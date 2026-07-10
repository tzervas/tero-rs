//! Tests for `crate::scheduler` — M-709 (single-queue baseline) / M-861 (per-worker deques +
//! steal-on-empty) / RFC-0008 RT1·RT2·RT3.
//!
//! M-797 in-crate test layout: all tests live here, not in `scheduler.rs`.
//!
//! # DoD coverage (M-861, amended by M-864)
//!
//! 1. **Construction refusals** stay fail-closed (`ZeroWorkers`/`ZeroCapacity`), unchanged by the
//!    deque redesign.
//! 2. **Results stay in spawn order** regardless of steal activity (RT2-comparable output).
//! 3. **RT2 sequentialization differential extended under stealing:** many randomized
//!    `(values, workers)` configurations — deliberately biased toward `workers` small relative to
//!    job count, to force steal activity — assert `parallel == sequential reference`.
//! 4. **Liveness** (every job runs exactly once) holds under random worker/job-count
//!    configurations, including single-worker (no stealing possible) and many-worker (steal-heavy)
//!    extremes.
//! 5. **Peak pending depth == job count (M-864 — backpressure REMOVED).** M-861's demand-signalled
//!    `capacity` backpressure was the feeder's bare-block point and the cause of a reproduced
//!    nested-submission deadlock, so it is gone (module docs / DN-67): the pool queue is unbounded
//!    and a batch materializes all `n` jobs across its lanes up front, so the peak *total* pending
//!    depth is exactly `n` and `capacity` no longer bounds anything. (Was: "backpressure stays
//!    `Exact`, peak ≤ capacity".)
//! 6. **`StealPolicy::select_victim` (RT3 EXPLAIN) is total, deterministic, and inspectable:**
//!    same inputs → same `StealDecision`; returns `None` iff every other deque is empty; the
//!    returned `victim` is never the thief itself and always has nonzero occupancy in the snapshot.
//! 7. **Steal activity is actually exercised — a real mutant-witness, not just an isolated-policy
//!    check.** `run_indexed`'s `steal_count` out-param counts jobs completed via a cross-deque
//!    steal; under a steal-forcing shape (few workers, many jobs) it must be `> 0`. A scheduler
//!    that silently regressed to single-queue/no-steal dispatch would still pass checks 1–5 (the
//!    *outputs* would still be correct) but this test would catch the regression directly.
//! 8. **M-864 nested submission:** deadlock-free at any depth (forced-low-`P` tests), panic-safe,
//!    deterministic — AND a *characterizing* test for the help-steal frame-stack growth under
//!    deep+wide low-`P` nesting (the moderate safe region; the `O(depth)`-stack leapfrogging fix is
//!    the tracked follow-up M-868). See DN-67 §3.4.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use proptest::prelude::*;

use crate::pool::Pool;
use crate::scheduler::{Scheduler, SchedulerError, StealPolicy};

#[test]
fn zero_workers_is_refused() {
    // Mutant witness: dropping the check would return Ok(_), so unwrap_err would panic.
    assert_eq!(
        Scheduler::with_workers(0, 4).unwrap_err(),
        SchedulerError::ZeroWorkers,
        "zero workers must fail closed (G2)"
    );
}

#[test]
fn zero_capacity_is_refused() {
    assert_eq!(
        Scheduler::with_workers(4, 0).unwrap_err(),
        SchedulerError::ZeroCapacity,
        "zero capacity must fail closed (G2)"
    );
}

#[test]
fn new_has_at_least_one_worker() {
    // Even on a single-core probe, available_parallelism fallback is 1 (never 0 — no silent hang).
    let s = Scheduler::new();
    assert!(s.workers() >= 1, "scheduler must have ≥ 1 worker");
    assert!(s.capacity() >= 1, "scheduler must have ≥ 1 capacity");
}

#[test]
fn new_uses_default_steal_policy() {
    let s = Scheduler::new();
    assert_eq!(
        s.steal_policy(),
        StealPolicy::RoundRobin,
        "Scheduler::new must use the documented default steal policy"
    );
}

#[test]
fn empty_job_set_returns_empty() {
    let s = Scheduler::with_workers(4, 8).unwrap();
    let out: Vec<i64> = s.run_indexed(Vec::<fn() -> i64>::new(), None, None);
    assert!(out.is_empty(), "no jobs ⇒ empty result (no hang)");
}

#[test]
fn results_are_in_spawn_order() {
    // Mutant witness: if results were collected in completion order rather than by spawn index,
    // this deterministic-order assertion would fail under real parallelism — including under
    // steal-heavy configurations (few workers, many jobs, so most jobs are stolen).
    let s = Scheduler::with_workers(4, 8).unwrap();
    let jobs: Vec<_> = (0..32usize).map(|i| move || i * 10).collect();
    let out = s.run_indexed(jobs, None, None);
    let expected: Vec<usize> = (0..32).map(|i| i * 10).collect();
    assert_eq!(
        out, expected,
        "output must be in spawn order (RT2-comparable)"
    );
}

#[test]
fn results_are_in_spawn_order_steal_heavy() {
    // Deliberately steal-heavy: 2 workers, 200 jobs, so nearly every worker will empty its own
    // deque and steal repeatedly. Output must still be spawn-order.
    let s = Scheduler::with_workers(2, 4).unwrap();
    let jobs: Vec<_> = (0..200usize).map(|i| move || i).collect();
    let out = s.run_indexed(jobs, None, None);
    let expected: Vec<usize> = (0..200).collect();
    assert_eq!(
        out, expected,
        "spawn-order output must hold even under heavy steal activity"
    );
}

#[test]
fn stealing_actually_occurs_under_a_lopsided_workload() {
    // Mutant witness for "stealing never happens": worker 0's round-robin share (even spawn
    // indices) is all slow (parks briefly), worker 1's share (odd indices) is all instant. Worker 1
    // will drain its own deque long before worker 0 finishes even its first slow job, so worker 1
    // MUST steal from worker 0's backlog to make progress — a scheduler that silently regressed to
    // single-queue/no-steal dispatch (or one where steal-selection was a no-op) would report
    // `steal_count == 0` here, and this assertion would catch it.
    let s = Scheduler::with_workers(2, 64).unwrap();
    let jobs: Vec<Box<dyn FnOnce() -> usize + Send>> = (0..64usize)
        .map(|i| -> Box<dyn FnOnce() -> usize + Send> {
            if i % 2 == 0 {
                Box::new(move || {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    i
                })
            } else {
                Box::new(move || i)
            }
        })
        .collect();
    let mut steals = 0usize;
    let out = s.run_indexed(jobs, None, Some(&mut steals));
    let expected: Vec<usize> = (0..64).collect();
    assert_eq!(
        out, expected,
        "spawn-order output must hold under a lopsided workload too"
    );
    assert!(
        steals > 0,
        "worker 1 (all-instant jobs) must have stolen at least one job from worker 0's \
         (all-slow) backlog — steal_count was 0, indicating stealing did not occur"
    );
}

// ── RT3: StealPolicy::select_victim is total, deterministic, inspectable ──────────────────────

#[test]
fn select_victim_none_when_all_empty() {
    let occupancy = vec![0usize; 4];
    let decision = StealPolicy::RoundRobin.select_victim(4, 0, &occupancy);
    assert_eq!(
        decision, None,
        "no candidate has work ⇒ select_victim must return None, never a spurious pick"
    );
}

#[test]
fn select_victim_never_targets_the_thief() {
    // thief = 1; only worker 3 (a neighbor, not the thief) has work.
    let occupancy = vec![0usize, 0, 0, 5];
    let decision = StealPolicy::RoundRobin
        .select_victim(4, 1, &occupancy)
        .expect("worker 3 has work to steal");
    assert_ne!(
        decision.victim, decision.thief,
        "a thief must never steal from itself"
    );
    assert_eq!(
        decision.victim, 3,
        "the only nonempty neighbor must be selected"
    );
}

#[test]
fn select_victim_ignores_thiefs_own_occupancy() {
    // Even if occupancy[thief] were nonzero (a caller violating the documented precondition —
    // "the caller only asks once its own deque is empty"), the scan starts at offset 1, so the
    // thief itself can never be the returned victim — here every OTHER worker is empty, so the
    // result must be None despite occupancy[thief] being nonzero.
    let occupancy = vec![0usize, 5, 0, 0];
    let decision = StealPolicy::RoundRobin.select_victim(4, 1, &occupancy);
    assert_eq!(
        decision, None,
        "select_victim must never report the thief's own occupancy as a steal target"
    );
}

#[test]
fn select_victim_picks_first_nonempty_in_rotation() {
    // thief=0, workers 1..3 scanned in order; worker 2 is the first nonempty.
    let occupancy = vec![0usize, 0, 3, 7];
    let decision = StealPolicy::RoundRobin
        .select_victim(4, 0, &occupancy)
        .expect("worker 2 has work");
    assert_eq!(
        decision.victim, 2,
        "round-robin must pick the first nonempty candidate"
    );
    assert_eq!(
        decision.victim_depth, 3,
        "the decision must record the victim's occupancy"
    );
    assert_eq!(
        decision.candidates_scanned, 2,
        "candidates_scanned counts worker 1 (empty) then worker 2 (chosen)"
    );
}

#[test]
fn select_victim_is_deterministic() {
    // Mutant witness: if selection consulted any hidden/random state, two calls with identical
    // inputs could disagree.
    let occupancy = vec![2usize, 0, 4, 0, 1];
    let d1 = StealPolicy::RoundRobin.select_victim(5, 1, &occupancy);
    let d2 = StealPolicy::RoundRobin.select_victim(5, 1, &occupancy);
    assert_eq!(
        d1, d2,
        "select_victim must be a pure, deterministic function of its inputs"
    );
}

proptest! {
    // RT2 sequentialization differential, EXTENDED UNDER STEALING (M-861): the parallel run
    // (per-worker deques + steal-on-empty) equals the spawn-order sequential reference, across
    // randomized worker counts including steal-forcing shapes (few workers, many jobs). Tagged
    // Empirical (this is the checked basis).
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn parallel_run_equals_sequential_reference_under_stealing(
        values in proptest::collection::vec(any::<i32>(), 0..128usize),
        workers in 1usize..8,
    ) {
        let s = Scheduler::with_workers(workers, workers * 2).unwrap();
        // Pure task: a deterministic function of the (captured) value — no shared state (RT1).
        let seq_ref: Vec<i64> = values.iter().map(|&v| i64::from(v).wrapping_mul(3)).collect();
        let jobs: Vec<_> = values
            .iter()
            .map(|&v| move || i64::from(v).wrapping_mul(3))
            .collect();
        let parallel = s.run_indexed(jobs, None, None);
        prop_assert_eq!(
            parallel, seq_ref,
            "parallel run (with per-worker deques + steal-on-empty) must equal the sequential \
             reference (RT2) — stealing reorders execution, never the observable result"
        );
    }

    // Same differential, but deliberately steal-heavy: worker count is held small (1..4) while job
    // count ranges wider, so most workers exhaust their own round-robin share and must steal.
    #[test]
    fn parallel_run_equals_sequential_reference_steal_heavy(
        values in proptest::collection::vec(any::<i32>(), 0..256usize),
        workers in 1usize..4,
    ) {
        let s = Scheduler::with_workers(workers, workers * 2).unwrap();
        let seq_ref: Vec<i64> = values.iter().map(|&v| i64::from(v).wrapping_mul(7).wrapping_add(1)).collect();
        let jobs: Vec<_> = values
            .iter()
            .map(|&v| move || i64::from(v).wrapping_mul(7).wrapping_add(1))
            .collect();
        let parallel = s.run_indexed(jobs, None, None);
        prop_assert_eq!(
            parallel, seq_ref,
            "steal-heavy configuration (few workers, many jobs) must still equal the sequential \
             reference (RT2)"
        );
    }

    // Liveness: every submitted job runs exactly once (no job dropped, none run twice), under
    // random worker/job-count configurations spanning no-steal (workers >= n) through
    // steal-heavy (workers << n).
    #[test]
    fn every_job_runs_exactly_once(
        n in 1usize..200,
        workers in 1usize..8,
    ) {
        let s = Scheduler::with_workers(workers, workers * 2).unwrap();
        // Each job returns its own index; the multiset of outputs must be exactly 0..n.
        let jobs: Vec<_> = (0..n).map(|i| move || i).collect();
        let mut out = s.run_indexed(jobs, None, None);
        out.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        prop_assert_eq!(out, expected, "each job runs exactly once (liveness), regardless of steal activity");
    }

    // Peak pending depth is exactly `n` (M-864: the whole batch is materialized across its lanes
    // before any lane drains — the queue is unbounded, `capacity` no longer bounds it). This
    // replaces M-861's `ready_queue_never_exceeds_capacity`: that backpressure bound was the
    // feeder's bare-block point and the reproduced-deadlock cause, so it was removed (module docs /
    // DN-67). `cap` is passed only to confirm it is genuinely ignored — peak is `n`, not `≤ cap`.
    #[test]
    fn peak_pending_depth_is_the_job_count_capacity_no_longer_bounds(
        n in 1usize..200,
        workers in 1usize..6,
        cap in 1usize..8,
    ) {
        let s = Scheduler::with_workers(workers, cap).unwrap();
        let jobs: Vec<_> = (0..n).map(|i| move || i).collect();
        let mut peak = 0usize;
        let _ = s.run_indexed(jobs, Some(&mut peak), None);
        prop_assert_eq!(
            peak, n,
            "M-864: all n jobs are enqueued before any lane drains, so peak == n (capacity no \
             longer bounds the queue)"
        );
    }

    // RT3: select_victim never returns the thief as its own victim, and (when it returns Some) the
    // reported victim_depth matches the snapshot passed in — the EXPLAIN record is faithful, not
    // approximate.
    #[test]
    fn select_victim_decision_is_faithful_to_snapshot(
        occupancy in proptest::collection::vec(0usize..10, 2..12),
        thief_seed in any::<usize>(),
    ) {
        let workers = occupancy.len();
        let thief = thief_seed % workers;
        let mut occ = occupancy.clone();
        occ[thief] = 0; // precondition: the thief consults the policy only once its own deque is empty
        let decision = StealPolicy::RoundRobin.select_victim(workers, thief, &occ);
        if let Some(d) = decision {
            prop_assert_ne!(d.victim, thief, "victim must never be the thief");
            prop_assert_eq!(d.victim_depth, occ[d.victim], "EXPLAIN record must match the snapshot");
            prop_assert!(d.victim_depth > 0, "a chosen victim must have nonzero occupancy");
            prop_assert!(d.candidates_scanned >= 1 && d.candidates_scanned < workers);
        } else {
            // None is only valid if every non-thief worker was empty in the snapshot.
            let any_other_nonempty = occ.iter().enumerate().any(|(i, &d)| i != thief && d > 0);
            prop_assert!(!any_other_nonempty, "select_victim returned None despite a nonempty candidate existing");
        }
    }
}

// ── M-864: nested `run_indexed` submission on the persistent pool ─────────────────────────────
//
// These stress the property `run_indexed`'s doc/`crate::pool` module docs claim but the tests above
// never exercise: a job that itself calls `Scheduler::run_indexed` (nested submission), at various
// depths/widths/shapes, must (1) never deadlock (checked here via a wall-clock timeout, not just an
// unbounded `cargo test` hang), (2) produce results identical to a plain sequential reference, stably
// across many repeated runs, and (3) never grow the OS thread count with nesting depth (the whole
// point of replacing per-call `thread::scope` with the persistent pool).

/// Run `f` on a fresh thread and wait up to `timeout` for it to finish, panicking with a clear
/// "suspected deadlock" message instead of hanging forever if it does not. `f`'s spawned thread is
/// never joined on timeout (a genuine deadlock would leak it) — acceptable for a test: the process
/// still exits normally, since only `main`/the test-harness thread returning matters for exit, and a
/// clear, fast test failure is far more useful here than an unbounded hang.
fn run_with_timeout<T: Send + 'static>(
    timeout: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(timeout).unwrap_or_else(|_| {
        panic!(
            "nested run_indexed did not complete within {timeout:?} — suspected deadlock (M-864 \
             help-steal invariant violated)"
        )
    })
}

/// The pure, unscheduled sequential reference for the nested shapes below: `widths[0]` is the
/// root's fan-out, `widths[1..]` the fan-out at every deeper level (uniform per level, but the
/// shape as a whole can still mix zero/one/many — see the mixed-shape test). A leaf (`widths` empty)
/// contributes exactly `1`.
fn nested_reference_shape(widths: &[usize]) -> u64 {
    match widths.split_first() {
        None => 1,
        Some((&w, rest)) => (0..w).map(|_| nested_reference_shape(rest)).sum(),
    }
}

/// The nested-`run_indexed` counterpart of [`nested_reference_shape`]: every non-leaf level fans out
/// across a **fresh** `Scheduler::new()` batch, each of whose jobs recurses into the next level —
/// i.e. every non-leaf `run_indexed` call here is itself invoked from *inside* a job of its parent's
/// `run_indexed` call (nested submission, arbitrarily deep). Must equal
/// [`nested_reference_shape`] exactly, regardless of shape or repeated runs (M-864 DoD).
fn nested_parallel_shape(widths: &[usize]) -> u64 {
    match widths.split_first() {
        None => 1,
        Some((&w, rest)) => {
            let rest = rest.to_vec(); // owned so each job closure can be `'static` (M-864 contract)
            let jobs: Vec<_> = (0..w)
                .map(|_| {
                    let rest = rest.clone();
                    move || nested_parallel_shape(&rest)
                })
                .collect();
            Scheduler::new()
                .run_indexed(jobs, None, None)
                .into_iter()
                .sum()
        }
    }
}

#[test]
fn nested_deep_chain_matches_sequential_reference_no_deadlock() {
    // A single, deep (40-level) chain — width 1 at every level, so exactly one nested `run_indexed`
    // call is ever in flight at a time, but 40 of them are simultaneously on the call stack at the
    // deepest point (each still waiting — via help-steal, never a bare block — on its own child).
    let shape = vec![1usize; 40];
    let expected = nested_reference_shape(&shape);
    let actual = run_with_timeout(Duration::from_secs(30), move || {
        nested_parallel_shape(&shape)
    });
    assert_eq!(
        actual, expected,
        "a 40-deep nested chain of run_indexed calls must match the sequential reference"
    );
}

#[test]
fn nested_wide_fanout_matches_sequential_reference_no_deadlock() {
    // 3 levels deep, wide fan-out at each level, and — unlike the deep-chain test — EVERY sibling at
    // every level also recurses, so many nested `run_indexed` batches are concurrently in flight,
    // all funnelled through the one shared, bounded, persistent pool (M-864's target scenario).
    let shape = vec![15usize, 15, 6];
    let expected = nested_reference_shape(&shape);
    let actual = run_with_timeout(Duration::from_secs(60), move || {
        nested_parallel_shape(&shape)
    });
    assert_eq!(
        actual, expected,
        "wide nested fan-out (concurrent nested batches) must match the sequential reference"
    );
}

#[test]
fn nested_mixed_batch_sizes_including_empty_and_single_item_match_reference() {
    // A deliberately irregular shape: a zero-width level (a branch that fans out to NOTHING — the
    // `n == 0` fast path, reached from *inside* a nested call), a width-1 level (no real parallelism,
    // `workers.min(n) == 1`), and ordinary wider levels, all nested inside one another.
    let shape = vec![6usize, 0, 4, 1, 5, 2];
    let expected = nested_reference_shape(&shape);
    let actual = run_with_timeout(Duration::from_secs(60), move || {
        nested_parallel_shape(&shape)
    });
    assert_eq!(
        actual, expected,
        "mixed batch-size nesting (including empty/single-item batches) must match the sequential \
         reference"
    );
}

#[test]
fn nested_recursion_is_deterministic_across_many_repeated_runs() {
    // The M-864 DoD explicitly asks for determinism "stable across many repeated runs", not a
    // one-off pass — covers a deep chain, a wide fan-out, and a mixed shape, 50 runs each.
    let shapes: [Vec<usize>; 3] = [
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        vec![12, 8],
        vec![5, 0, 3, 1, 4],
    ];
    for shape in &shapes {
        let expected = nested_reference_shape(shape);
        for run in 0..50 {
            let owned_shape = shape.clone();
            let actual = run_with_timeout(Duration::from_secs(30), move || {
                nested_parallel_shape(&owned_shape)
            });
            assert_eq!(
                actual, expected,
                "nested run_indexed must be exactly reproducible run-to-run (run {run}, shape \
                 {shape:?})"
            );
        }
    }
}

#[test]
fn nested_empty_and_single_item_batches_never_hang() {
    // A nested call whose OWN batch is empty (n == 0) short-circuits before touching the pool at all
    // (the `if n == 0` fast path in `run_indexed`) — and a nested call with exactly one job must not
    // deadlock either (`workers.min(1) == 1` lane, no steal candidates ever consulted).
    let outer_jobs: Vec<Box<dyn FnOnce() -> usize + Send>> = vec![
        Box::new(|| {
            let inner: Vec<u64> =
                Scheduler::new().run_indexed(Vec::<fn() -> u64>::new(), None, None);
            inner.len()
        }),
        Box::new(|| {
            let inner: Vec<u64> = Scheduler::new().run_indexed(vec![|| 7u64], None, None);
            inner.len()
        }),
    ];
    let result = run_with_timeout(Duration::from_secs(10), move || {
        Scheduler::new().run_indexed(outer_jobs, None, None)
    });
    assert_eq!(
        result,
        vec![0, 1],
        "nested empty/single-item batches must resolve without hanging"
    );
}

/// Best-effort OS thread count for THIS process (Linux only — `/proc/self/status`). Used only by
/// [`nested_recursion_thread_count_is_bounded_not_growing_with_depth`] as a regression witness for
/// "the persistent pool never grows with nesting depth" (M-864 DoD). Every other nested-recursion
/// property above (determinism, no-deadlock, liveness) is checked platform-independently.
#[cfg(target_os = "linux")]
fn os_thread_count() -> usize {
    let status = std::fs::read_to_string("/proc/self/status")
        .expect("mycelium-sched test: /proc/self/status must be readable on Linux");
    status
        .lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .and_then(|s| s.trim().parse::<usize>().ok())
        .expect("mycelium-sched test: /proc/self/status must have a `Threads:` line")
}

#[test]
#[cfg(target_os = "linux")]
fn nested_recursion_thread_count_is_bounded_not_growing_with_depth() {
    fn deep_chain(depth: usize, peak: &Arc<AtomicUsize>) -> u64 {
        peak.fetch_max(os_thread_count(), Ordering::Relaxed);
        if depth == 0 {
            return 1;
        }
        let peak = Arc::clone(peak);
        let jobs: Vec<Box<dyn FnOnce() -> u64 + Send>> =
            vec![Box::new(move || deep_chain(depth - 1, &peak))];
        Scheduler::new()
            .run_indexed(jobs, None, None)
            .into_iter()
            .sum()
    }

    // Warm the persistent pool first (the first-ever `run_indexed` call anywhere lazily spawns its
    // `available_parallelism()` OS threads — see `crate::pool::get`) so the baseline below reflects
    // its steady state, not a cold start.
    let _: Vec<i32> = Scheduler::new().run_indexed(vec![|| 1], None, None);
    thread::sleep(Duration::from_millis(20));
    let baseline = os_thread_count();

    let peak = Arc::new(AtomicUsize::new(baseline));
    let peak_for_run = Arc::clone(&peak);
    let depth = 40usize;
    let result = run_with_timeout(Duration::from_secs(30), move || {
        deep_chain(depth, &peak_for_run)
    });
    assert_eq!(
        result, 1,
        "a width-1 chain always sums to 1 regardless of depth"
    );

    let observed_peak = peak.load(Ordering::Relaxed);
    let available = thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    // Generous, depth-INDEPENDENT slack (test-harness/runtime threads). The point: peak stays a
    // small constant, never `O(depth)` — a regression to per-call `thread::scope` would blow this at
    // depth 40 (dozens of threads, growing linearly with depth); the persistent pool stays flat.
    assert!(
        observed_peak <= available + baseline + 16,
        "OS thread count grew with nesting depth (observed peak {observed_peak}, baseline \
         {baseline}, available_parallelism {available}) — the pool must stay bounded regardless of \
         how deep run_indexed nests (M-864)"
    );
}

// ── M-864 correctness rewrite: FORCED-LOW-WORKER-COUNT nested tests ───────────────────────────
//
// The tests above run on the GLOBAL pool (sized to `available_parallelism()`), so on a many-core
// box they cannot reproduce the nested-submission deadlock the original M-864 implementation had:
// its feeder bare-blocked on backpressure (`total >= capacity`) BEFORE help-stealing, so with
// enough nesting every pool thread bare-blocked and nothing drained the queue. That hang manifests
// only when the pool worker count `P` is SMALL relative to the fan-out width (width > cap + P).
//
// These tests force an explicit small `P` via `Pool::with_workers_for_test` + `Scheduler::run_
// indexed_on`, so the whole nested tree (every level) runs on the forced-count pool. They are the
// hardware-independent validation the coordinator required: on the PRE-FIX code they DEADLOCK (the
// `run_with_timeout` wrapper turns the hang into a fast, explicit test failure); on the fixed code
// (unbounded queue, feed-then-run, no feeder/lane bare-block) they PASS. Verified by hand against a
// scratch revert of the feeder-block (see DN-67 §3 / the PR report).

/// Nested `run_indexed`, threading an EXPLICIT `pool` through every level (so a forced-small worker
/// count holds for the whole tree, not just the top batch). `sched` fixes the per-batch lane count.
fn nested_parallel_shape_on(pool: &Arc<Pool>, sched: Scheduler, widths: &[usize]) -> u64 {
    match widths.split_first() {
        None => 1,
        Some((&w, rest)) => {
            let rest = rest.to_vec();
            let jobs: Vec<_> = (0..w)
                .map(|_| {
                    let rest = rest.clone();
                    let pool = Arc::clone(pool);
                    move || nested_parallel_shape_on(&pool, sched, &rest)
                })
                .collect();
            sched
                .run_indexed_on(pool, jobs, None, None)
                .into_iter()
                .sum()
        }
    }
}

#[test]
fn forced_low_p_wide_fanout_does_not_deadlock_p1_through_p4() {
    // THE regression test for the reproduced hang. The `[15,15,6]` shape (width 15 ≫ any of the
    // small forced worker counts + the old `capacity`) is exactly what deadlocked the pre-fix
    // feeder on a ≤4-core machine. Run it at forced P ∈ {1,2,3,4}; each must complete and match the
    // sequential reference under a wall-clock timeout. A single hang at ANY P fails the test.
    let shape = vec![15usize, 15, 6];
    let expected = nested_reference_shape(&shape);
    for p in 1usize..=4 {
        let shape = shape.clone();
        let actual = run_with_timeout(Duration::from_secs(60), move || {
            let pool = Pool::with_workers_for_test(p);
            // Lane count per batch: use a modest fixed count so batches genuinely fan out across
            // lanes even when P is tiny (the lanes are pool TASKS, run by P workers + the helping
            // caller — the point is that nested feeders never bare-block regardless of P vs width).
            let sched = Scheduler::with_workers(4, 8).unwrap();
            nested_parallel_shape_on(&pool, sched, &shape)
        });
        assert_eq!(
            actual, expected,
            "forced P={p}: wide nested fan-out [15,15,6] must complete without deadlock and match \
             the sequential reference (this shape hangs on the pre-fix feeder-block code)"
        );
    }
}

#[test]
fn forced_low_p_deep_chain_and_mixed_shapes_do_not_deadlock() {
    // A deep chain and irregular mixed shapes (incl. empty/single-item sub-batches), all at forced
    // low P — the nested-submission stress the global-pool tests can't force on a many-core box.
    let shapes: [Vec<usize>; 4] = [
        vec![1usize; 30],    // deep chain
        vec![8, 0, 5, 1, 4], // mixed incl. empty + single-item
        vec![10, 10],        // two wide levels
        vec![3, 3, 3, 3, 3], // moderate fan-out, moderately deep
    ];
    for p in 1usize..=4 {
        for shape in &shapes {
            let expected = nested_reference_shape(shape);
            let shape = shape.clone();
            let actual = run_with_timeout(Duration::from_secs(60), move || {
                let pool = Pool::with_workers_for_test(p);
                let sched = Scheduler::with_workers(4, 8).unwrap();
                nested_parallel_shape_on(&pool, sched, &shape)
            });
            assert_eq!(
                actual, expected,
                "forced P={p}: nested shape must complete without deadlock and match the reference"
            );
        }
    }
}

#[test]
fn forced_p1_single_worker_nested_is_the_hardest_case_and_still_completes() {
    // P=1 is the tightest: a single pool worker, so ALL concurrency comes from the caller's own
    // `help_while` recruiting itself as an extra helper. If any lane or feeder bare-blocked, P=1
    // would wedge immediately. A width-6, depth-3 tree (216 leaves) must still complete.
    let shape = vec![6usize, 6, 6];
    let expected = nested_reference_shape(&shape);
    let actual = run_with_timeout(Duration::from_secs(60), move || {
        let pool = Pool::with_workers_for_test(1);
        let sched = Scheduler::with_workers(4, 8).unwrap();
        nested_parallel_shape_on(&pool, sched, &shape)
    });
    assert_eq!(
        actual, expected,
        "forced P=1 (all concurrency via the caller's help_while) must still complete a nested tree"
    );
}

// ── M-864 Defect 2: a panicking job must not hang the join or kill the pool ───────────────────

#[test]
fn a_panicking_job_propagates_at_join_without_hanging_and_pool_survives() {
    // A job that panics must (1) NOT hang the batch (the RAII drop-guard decrements `remaining` on
    // unwind), (2) NOT kill the persistent pool worker (each job runs under catch_unwind), and
    // (3) re-raise the panic in the calling thread at the join (thread::scope-like propagation).
    // Use a forced pool so a killed worker (regression) would be observable as a subsequent hang.
    let pool = Pool::with_workers_for_test(2);
    let sched = Scheduler::with_workers(4, 8).unwrap();

    // (3) the panic is surfaced to the caller (not swallowed).
    let pool_a = Arc::clone(&pool);
    let panicked = run_with_timeout(Duration::from_secs(30), move || {
        let jobs: Vec<Box<dyn FnOnce() -> u64 + Send>> = vec![
            Box::new(|| 1u64),
            Box::new(|| panic!("job boom (expected by the test)")),
            Box::new(|| 3u64),
        ];
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            sched.run_indexed_on(&pool_a, jobs, None, None)
        }))
        .is_err()
    });
    assert!(
        panicked,
        "a panicking job must re-raise at the join (thread::scope-like), not be swallowed"
    );

    // (1)+(2): the SAME pool must still work after a panicking batch — a subsequent batch completes
    // (the worker did not die; the join did not hang). If either regressed, this would time out.
    let pool_b = Arc::clone(&pool);
    let sum = run_with_timeout(Duration::from_secs(30), move || {
        let sched = Scheduler::with_workers(4, 8).unwrap();
        let jobs: Vec<_> = (0..50usize).map(|i| move || i as u64).collect();
        sched
            .run_indexed_on(&pool_b, jobs, None, None)
            .into_iter()
            .sum::<u64>()
    });
    assert_eq!(
        sum,
        (0..50u64).sum::<u64>(),
        "the pool must survive a panicking job — a later batch on the same pool must complete"
    );
}

#[test]
fn a_nested_panic_propagates_up_through_the_nesting_without_hanging() {
    // A panic in a DEEPLY nested job must propagate all the way up (each level's help_while returns,
    // each level re-raises), never hang a mid-level join. Forced P=2 so a hang would be immediate.
    let pool = Pool::with_workers_for_test(2);
    let raised = run_with_timeout(Duration::from_secs(30), move || {
        let sched = Scheduler::with_workers(4, 8).unwrap();
        let pool_inner = Arc::clone(&pool);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            // outer batch → each job runs an inner batch → one inner job panics.
            let outer: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..3usize)
                .map(|k| -> Box<dyn FnOnce() -> u64 + Send> {
                    let pool_k = Arc::clone(&pool_inner);
                    Box::new(move || {
                        let inner: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..3usize)
                            .map(|j| -> Box<dyn FnOnce() -> u64 + Send> {
                                if k == 1 && j == 2 {
                                    Box::new(|| panic!("deep boom (expected)"))
                                } else {
                                    Box::new(move || (j as u64) + 1)
                                }
                            })
                            .collect();
                        sched
                            .run_indexed_on(&pool_k, inner, None, None)
                            .into_iter()
                            .sum()
                    })
                })
                .collect();
            sched.run_indexed_on(&pool_inner, outer, None, None)
        }))
        .is_err()
    });
    assert!(
        raised,
        "a deeply-nested job panic must propagate up through every join, never hang a mid-level one"
    );
}

// ── M-864: help-steal frame-stack growth under deep+wide low-P nesting (CHARACTERIZING) ────────
//
// `Pool::help_while` pops from the shared queue INDISCRIMINATELY — any batch's lane-loop, not just
// tasks descending from the waiter's own subtree — so a nested pop → nested `help_while` stacks a
// call frame on ONE OS thread. Under DEEP+WIDE nesting at low P, a single thread can accumulate
// help-steal frames from many sibling/cousin batches (worst case ~O(w^(d-1))), so the frame STACK
// grows with the live-internal-batch count. The deadlock-freedom induction (module docs) proves
// logical PROGRESS but NOT bounded stack — so `run_indexed` is deadlock-free / panic-safe /
// deterministic at any depth, but only stack-SAFE for MODERATE depth×width (never-silent, VR-5).
//
// This test CHARACTERIZES the safe region rather than asserting "any depth". Measured boundary
// (debug build, ~2 MiB default thread stack, forced P=1): shapes up to depth 5 at every tested
// width (incl. [8,8,8,8] = 4096 leaves) and depth 6 at width 3 COMPLETE; depth 6 width 4, depth 8
// width 3, and depth 16 width 2 STACK-OVERFLOW (a crash, not a hang). Width amplifies depth, as the
// O(w^(d-1)) worst case predicts. The O(depth)-stack fix — Cilk-style leapfrogging, where
// `help_while` runs ONLY tasks descending from its own batch — is the tracked follow-up M-868;
// see DN-67 §3.4. Current consumers (M-860/M-862) do not nest at all, so they are trivially inside
// the safe region.
#[test]
fn deep_and_wide_low_p_completes_within_a_normal_stack_moderate_region() {
    // [4,4,4,4]: depth 4, width 4 — 256 leaves, 85 internal batches, all funnelled through 1–2 pool
    // workers plus the caller's own help_while. Genuinely deep AND wide, with ample margin below the
    // measured overflow boundary (≈ depth 6), so it completes within a normal stack and is not
    // scheduling-flaky. This documents the moderate safe region; it deliberately does NOT probe to
    // overflow (that would crash the test process, not fail an assertion).
    let shape = vec![4usize, 4, 4, 4];
    let expected = nested_reference_shape(&shape);
    for p in 1usize..=2 {
        let shape = shape.clone();
        let actual = run_with_timeout(Duration::from_secs(60), move || {
            let pool = Pool::with_workers_for_test(p);
            let sched = Scheduler::with_workers(4, 8).unwrap();
            nested_parallel_shape_on(&pool, sched, &shape)
        });
        assert_eq!(
            actual, expected,
            "forced P={p}: a moderate deep+wide nested tree [4,4,4,4] must complete within a normal \
             stack (characterizes the safe region; deeper+wider overflows — tracked as M-868)"
        );
    }
}
