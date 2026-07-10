//! `Scheduler` — a real **OS-thread** work-stealing scheduler (M-709 / M-861 / M-864 / RFC-0008
//! RT1·RT2·RT3 / E12-1 / E25-1).
//!
//! The v0 R1 surface ([`crate::colony`]) ran tasks cooperatively on the calling thread. M-709 grew
//! that into a fixed pool of OS worker threads over a single shared FIFO queue. M-861 grew it
//! again: **per-worker deques with steal-on-empty** (LIFO-own / FIFO-steal), which cuts contention
//! on the single shared queue while preserving every guarantee M-709 established. **M-864** replaces
//! the per-call `std::thread::scope` (fresh OS threads every call) with dispatch onto the
//! process-wide **persistent** [`crate::pool`] — see that module's docs for the help-stealing design
//! that makes *nested* `run_indexed` submission safe on a **fixed**-size pool.
//!
//! # M-864 correctness rewrite (2026-07-01 — an adversarial review reproduced a real hang)
//!
//! The **first** M-864 implementation was **unsound**: it kept M-861's demand-signalled
//! backpressure (a feeder that `Condvar::wait`s while the per-lane deques hold `capacity` items) and
//! reached that wait **before** entering the help-steal loop. A nested `run_indexed` call (a pool
//! worker running a job that submits its own batch) could therefore **bare-block** the feeder while
//! its lane-tasks sat unrun — with enough nesting, every pool thread bare-blocks and nothing drains
//! the queue: a permanent hang (reproduced at forced low worker counts with a wide fan-out, e.g.
//! `[15,15,6]` at `P ∈ {1,2,3,4}`). A second defect: a **panicking job** skipped the batch's
//! completion decrement, hanging the join and permanently killing the pool worker.
//!
//! Both are fixed at the root:
//!
//! 1. **No bare-block anywhere on the batch's own progress.** The per-lane deques are **fully
//!    populated up front** (no `capacity` gate — the pool queue is unbounded; memory is bounded by
//!    the batch size = program size, which the caller already materialized in `jobs: Vec<F>`), then
//!    the lane-loop tasks are submitted. Because every deque is populated **before** any lane runs,
//!    a lane never has to *wait* for more work: it pops its own deque, steals when empty, and
//!    **exits** the instant nothing is left anywhere. No feeder `Condvar`, no lane `Condvar` — the
//!    lane-loop is *totally non-blocking*. The only "wait" left in the whole batch is
//!    [`crate::pool::Pool::help_while`], which is a help loop (it *runs* pending tasks), never a bare
//!    park. This restores the deadlock-freedom induction (module docs on `pool`): every thread that
//!    would otherwise wait is instead actively draining the shared queue.
//! 2. **Panic-safe join.** Each lane-loop runs each job under [`std::panic::catch_unwind`] (so a
//!    panicking job never kills the persistent pool worker) and the first captured panic is
//!    re-raised in the calling thread after the join ([`std::panic::resume_unwind`]) — matching
//!    `std::thread::scope`'s panic-propagates-at-join semantics as closely as safe std allows. An
//!    RAII drop-guard decrements the batch's outstanding-lane counter on **every** exit path
//!    (normal *or* unwind), so no panic can leave the join hanging.
//!
//! # Honesty (VR-5)
//!
//! - **RT2 sequentialization differential — `Empirical`, unchanged by stealing or by M-864's pool
//!   redesign.** RT1 (tasks share no mutable state) makes the observable result order-independent —
//!   the differential ("parallel run ≡ spawn-order sequential reference") asserts *result*-equality,
//!   never *scheduling*-equality, so work-stealing (which only reorders *execution*, never the
//!   RT1/RT2 observable) leaves the differential's claim unchanged, and neither does *how many OS
//!   threads* execute that reordered work (M-864: persistent pool vs. per-call fresh threads is an
//!   execution-substrate change, not an observable one). It is *checked* by a property test
//!   ([`tests`]) run under many randomized worker/steal configurations, not assumed — but it is not
//!   `Proven` (no mechanized theorem), so it stays `Empirical`.
//! - **RT3 — stealing is kept semantics-free.** The victim-selection policy
//!   ([`StealPolicy`]/[`StealDecision`]) is a small, total, deterministic, inspectable decision
//!   procedure — the same posture RFC-0008 §4.1 RT3 requires of a placement policy (mirroring the
//!   reserved `forage` construct's EXPLAIN posture, without depending on the unbuilt `forage`
//!   mechanism or on `mycelium-select`'s heavier RFC-0005 machinery, which is out of scope for this
//!   crate). Completion order and worker identity are never surfaced through the public API —
//!   [`Scheduler::run_indexed`] still returns outputs **in spawn order** — so RT2's deterministic
//!   default is preserved regardless of which worker executed which job or in what order steals
//!   occurred.
//! - **Backpressure — dropped as of M-864, honestly (was `Exact` under M-861).** The demand-signalled
//!   `capacity` bound on pending work was the *cause* of the reproduced deadlock (it was the feeder's
//!   bare-block point), and was a **non-normative implementation detail** to begin with (DN-61 §A.2:
//!   the R1 scheduler's only normative commitments are RT2 determinism + fuel-cooperative stepping +
//!   RT7 scope). It is removed: the pool queue is unbounded, and a batch's peak pending depth is
//!   simply its job count `n` (memory bounded by the program-sized `jobs` vector the caller already
//!   holds). `capacity` is **retained on the `Scheduler` API for source compatibility** but **no
//!   longer bounds anything** — see [`Scheduler::capacity`] (never-silent: documented, not quietly
//!   repurposed). DN-67 records the trade.
//! - **Liveness (every submitted job runs exactly once) — `Empirical`.** Property-tested over
//!   random job sets and random worker/steal configurations; not `Proven`.
//! - **Nested-submission deadlock-freedom (M-864) — `Empirical`.** With every batch-progress
//!   bare-block removed (point 1 above), [`crate::pool::Pool::help_while`]'s structural argument
//!   (module docs) holds: a **fixed**-size pool never deadlocks under arbitrarily deep nested
//!   `run_indexed` submission. Checked by **forced-low-worker-count** nested stress tests
//!   (`P ∈ {1,2,3,4}`, wide fan-out incl. the `[15,15,6]` shape that reproduced the original hang)
//!   under a wall-clock timeout ([`tests`]) — the tests that hang on the pre-fix code and pass on
//!   this one. Not mechanically proven, so `Empirical`, not `Proven` (VR-5).
//! - **Bounded *progress*, not bounded *stack* (M-864 — never-silent, VR-5).** Deadlock-freedom is a
//!   progress result; it does **not** bound the call stack. `help_while` pops the shared queue
//!   indiscriminately, so under **deep-AND-wide** low-`P` nesting a single OS thread can stack
//!   help-steal frames from many sibling batches (~`O(w^(d-1))`) → a **stack overflow**, not a hang.
//!   So nested `run_indexed` is deadlock-free / panic-safe / deterministic at any depth, but only
//!   *stack*-safe for **moderate** depth×width (measured region + boundary: `crate::pool` module
//!   docs, the `deep_and_wide_low_p_*` test, DN-67 §3.4). Current consumers submit a single,
//!   non-nested batch, so they are trivially safe. The `O(depth)`-stack (leapfrogging) fix is the
//!   tracked follow-up **M-868**.
//!
//! # The `'static` contract (M-864 — ratified: `docs/notes/DN-67-Persistent-Work-Stealing-Pool.md`)
//!
//! [`Scheduler::run_indexed`] now requires `F: 'static` and `T: 'static` (previously only `F: Send`
//! and `T: Send`, borrowing freely within the `std::thread::scope` call). A **persistent** pool's
//! worker threads outlive any single `run_indexed` call, so a job closure can no longer safely
//! borrow data from the calling stack frame the way `thread::scope`'s scoped threads could — it must
//! own (or `Arc`-share) everything it touches. Every current caller already passes owned data or was
//! adjusted to (M-860's per-node `Node::clone`, M-862's `Arc`-shared fuel counter and cloned
//! `Interpreter`); see DN-67 for the full rationale and the caller-by-caller audit.
//!
//! The crate stays `#![forbid(unsafe_code)]`: [`crate::pool`] is built from `Arc`/`Mutex`/`Condvar`/
//! `VecDeque`/`thread::spawn` — no `unsafe`, no `rayon`/`crossbeam` (a Chase-Lev lock-free deque
//! needs `unsafe` or an external crate; both are out of scope here, ADR/DN ratified: zero new
//! dependencies).

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use mycelium_core::GuaranteeStrength;

use crate::pool::{self, Pool};

/// Guarantee strength for the scheduler's RT2 sequentialization differential.
///
/// `Empirical`: the parallel run equals the sequential reference by RT1 (no shared mutable state),
/// checked by a property test — not `Proven` (no mechanized theorem). Unchanged by work-stealing
/// (M-861): stealing reorders *execution*, never the RT1/RT2 *observable*. (RFC-0008 RT2.)
pub const SCHEDULER_RT2_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

// NOTE (M-864): the former `SCHEDULER_BACKPRESSURE_STRENGTH` (`Exact` — total pending ≤ `capacity`)
// is **removed**. The `capacity` bound was the feeder's bare-block point and the cause of a
// reproduced nested-submission deadlock (see module docs); the pool queue is now unbounded (memory
// bounded by the batch's job count), so there is no backpressure guarantee to tag. `capacity`
// survives on the API for source compatibility but no longer bounds anything (documented on
// [`Scheduler::capacity`]; never-silent per DN-61 §A.2 / DN-67).

/// Guarantee strength for liveness (every submitted job runs exactly once).
///
/// `Empirical`: property-tested over random job sets and random worker/steal configurations; not
/// `Proven`.
pub const SCHEDULER_LIVENESS_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

/// Guarantee strength for the steal-victim-selection policy's determinism/inspectability (RT3).
///
/// `Exact`: [`StealPolicy::select_victim`] is a total, deterministic function of its inputs (worker
/// count, thief index, deque occupancy snapshot) — same inputs always produce the same
/// [`StealDecision`], every decision is inspectable. This is the RT3 "reified, named, explained"
/// obligation for the one piece of scheduling that is not RT1/RT2-neutral by inspection alone: a
/// caller can ask *why* a steal targeted a given worker.
pub const STEAL_POLICY_STRENGTH: GuaranteeStrength = GuaranteeStrength::Exact;

/// Why constructing a [`Scheduler`] refused — always explicit, never a silent fallback (G2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    /// A scheduler with zero workers can make no progress; rejected at construction (fail-closed,
    /// G2) rather than silently substituting a single worker.
    ZeroWorkers,
    /// A ready queue with zero capacity can never accept a job; rejected at construction (G2).
    ZeroCapacity,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::ZeroWorkers => f.write_str(
                "scheduler refused: zero workers cannot make progress (G2: fail-closed, never a \
                 silent single-worker substitution)",
            ),
            SchedulerError::ZeroCapacity => f.write_str(
                "scheduler refused: a zero-capacity ready queue can never accept a job (G2: \
                 fail-closed)",
            ),
        }
    }
}

impl std::error::Error for SchedulerError {}

/// The victim-selection policy for a worker whose own deque is empty (RFC-0008 RT3).
///
/// A policy is a **total, deterministic** procedure: same `(workers, thief, occupancy)` in ⇒ same
/// [`StealDecision`] out. This keeps stealing a placement-only concern (RT3: "where a hypha runs
/// may change performance, never the observable") — never a source of surprise, and always
/// EXPLAIN-able via [`StealPolicy::select_victim`]'s returned [`StealDecision`].
///
/// v0 ships exactly one policy, [`StealPolicy::RoundRobin`]; the type is an enum (not a bare
/// function) so a future policy is additive, not a breaking change — mirroring the reserved
/// `forage` construct's posture (a content-addressed, swappable decision procedure) without
/// depending on the unbuilt `forage` mechanism or the heavier RFC-0005 `mycelium-select` machinery
/// (out of scope for this crate; FLAG for a future placement-policy unification, see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StealPolicy {
    /// Starting one slot after the thief's own index, scan the other workers' deques in a fixed
    /// deterministic rotation and steal from the first non-empty one found (FIFO — `pop_front` —
    /// from the victim's deque, so the oldest work at the victim is taken first, minimizing
    /// disruption to the victim's own LIFO-recency locality).
    #[default]
    RoundRobin,
}

impl StealPolicy {
    /// Decide which worker `thief` should steal from, given a snapshot of every worker's deque
    /// length (`occupancy[i]` = worker `i`'s pending-job count, `occupancy[thief]` == 0 by the
    /// caller's precondition — a worker only consults the policy once its own deque is empty).
    ///
    /// Returns `None` if every other worker's deque is empty too (nothing to steal). Total,
    /// deterministic, EXPLAIN-able: the returned [`StealDecision`] records exactly which worker was
    /// picked, its occupancy, and how many candidates were scanned before it — the RT3 obligation.
    ///
    /// Guarantee: **Exact** ([`STEAL_POLICY_STRENGTH`]) — a pure function of its inputs, no hidden
    /// state, no randomness.
    #[must_use]
    pub fn select_victim(
        &self,
        workers: usize,
        thief: usize,
        occupancy: &[usize],
    ) -> Option<StealDecision> {
        debug_assert_eq!(
            occupancy.len(),
            workers,
            "occupancy snapshot must cover every worker"
        );
        match self {
            StealPolicy::RoundRobin => {
                for offset in 1..workers {
                    let candidate = (thief + offset) % workers;
                    let depth = occupancy[candidate];
                    if depth > 0 {
                        return Some(StealDecision {
                            thief,
                            victim: candidate,
                            victim_depth: depth,
                            candidates_scanned: offset,
                        });
                    }
                }
                None
            }
        }
    }
}

/// The EXPLAIN record for one [`StealPolicy::select_victim`] decision (RFC-0008 RT3: "every
/// departure from RT2's fragment ... is an explicit construct whose decision procedure ... [has]
/// mandatory EXPLAIN"). Inspectable, never silent — a caller (or a test) can reconstruct exactly
/// why a given worker was chosen as the victim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StealDecision {
    /// The worker whose own deque was empty and who is looking to steal.
    pub thief: usize,
    /// The worker chosen as the steal source.
    pub victim: usize,
    /// The victim's deque depth *at the time of the decision* (a snapshot — the victim's actual
    /// deque may change before the steal executes under the lock; the steal itself re-checks).
    pub victim_depth: usize,
    /// How many candidates (including the chosen victim) were scanned before landing on `victim`.
    pub candidates_scanned: usize,
}

/// A real OS-thread scheduler: **per-batch lanes with steal-on-empty** (LIFO-own / FIFO-steal)
/// dispatched onto the process-wide persistent [`crate::pool`] (RFC-0008 RT1·RT2·RT3; M-709/M-861/
/// M-864).
///
/// # Guarantee
/// - RT2 sequentialization: **`Empirical`** ([`SCHEDULER_RT2_STRENGTH`]), unchanged by stealing.
/// - RT3 steal-policy determinism/inspectability: **`Exact`** ([`STEAL_POLICY_STRENGTH`]).
/// - Nested-submission deadlock-freedom (M-864): **`Empirical`** (forced-low-worker-count nested
///   stress tests; see module docs). *Progress only* — the help-steal frame stack grows with the
///   live-internal-batch count, so nesting is stack-safe for **moderate** depth×width, not literally
///   unbounded (deep+wide low-`P` can overflow; leapfrogging fix tracked as M-868 — module docs).
/// - Liveness (each job runs once): **`Empirical`** ([`SCHEDULER_LIVENESS_STRENGTH`]).
/// - Backpressure: **removed at M-864** (was the deadlock cause; `capacity` no longer bounds the
///   queue — see module docs and [`Scheduler::capacity`]).
#[derive(Debug, Clone, Copy)]
pub struct Scheduler {
    workers: usize,
    capacity: usize,
    steal_policy: StealPolicy,
}

impl Scheduler {
    /// A scheduler sized to the host's available parallelism (fallback: 1 worker), with a ready
    /// queue capacity of `2 × workers` (allows up to two pending jobs per worker, independent of
    /// in-flight work), using the default [`StealPolicy`].
    ///
    /// Guarantee: **Exact** (construction is deterministic given the probed parallelism).
    #[must_use]
    pub fn new() -> Self {
        let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        Scheduler {
            workers,
            capacity: workers.saturating_mul(2),
            steal_policy: StealPolicy::default(),
        }
    }

    /// A scheduler with exactly `workers` OS threads and a ready-queue `capacity`, using the
    /// default [`StealPolicy`].
    ///
    /// # Errors
    /// [`SchedulerError::ZeroWorkers`] if `workers == 0`; [`SchedulerError::ZeroCapacity`] if
    /// `capacity == 0` (both fail-closed — G2: never a silent substitution).
    pub fn with_workers(workers: usize, capacity: usize) -> Result<Self, SchedulerError> {
        Self::with_workers_and_policy(workers, capacity, StealPolicy::default())
    }

    /// A scheduler with exactly `workers` OS threads, a ready-queue `capacity`, and an explicit
    /// [`StealPolicy`] — the RT3 EXPLAIN entry point: a caller who cares *which* deterministic
    /// victim-selection procedure is in effect can name it, rather than relying on the default.
    ///
    /// # Errors
    /// [`SchedulerError::ZeroWorkers`] if `workers == 0`; [`SchedulerError::ZeroCapacity`] if
    /// `capacity == 0` (both fail-closed — G2: never a silent substitution).
    pub fn with_workers_and_policy(
        workers: usize,
        capacity: usize,
        steal_policy: StealPolicy,
    ) -> Result<Self, SchedulerError> {
        if workers == 0 {
            return Err(SchedulerError::ZeroWorkers);
        }
        if capacity == 0 {
            return Err(SchedulerError::ZeroCapacity);
        }
        Ok(Scheduler {
            workers,
            capacity,
            steal_policy,
        })
    }

    /// The number of **lanes** a batch is split across (`min(workers, job-count)`) — the per-worker
    /// deque count, not the pool's OS-thread count (which is the process-wide
    /// [`available_parallelism`](std::thread::available_parallelism), shared by every scheduler).
    #[must_use]
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// The configured `capacity` value.
    ///
    /// **M-864: this no longer bounds anything.** Under M-861 it was a demand-signalled backpressure
    /// ceiling; that bound was the feeder's bare-block point and the cause of a reproduced
    /// nested-submission deadlock, so it was removed (the pool queue is unbounded — memory is bounded
    /// by a batch's job count, which the caller already materialized). The value is retained only for
    /// source compatibility of [`Scheduler::with_workers`]; it is never consulted by
    /// [`Scheduler::run_indexed`]. (Never-silent: documented here, not quietly repurposed — DN-61
    /// §A.2 / DN-67.)
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The steal-victim-selection policy this scheduler uses (RT3 EXPLAIN entry point).
    #[must_use]
    pub fn steal_policy(&self) -> StealPolicy {
        self.steal_policy
    }

    /// Run `jobs` across the persistent, process-wide work-stealing pool ([`crate::pool`], M-864)
    /// and return their outputs **in spawn order** (so the result vector is directly comparable to
    /// the sequential reference — the RT2 differential).
    ///
    /// Dispatch: the `jobs` are distributed round-robin across `min(workers, n)` **lanes** (per-batch
    /// deques). Each lane's loop pops its **own** deque LIFO (`pop_back` — recency locality) and, once
    /// empty, consults [`StealPolicy::select_victim`] to steal FIFO (`pop_front`) from another lane's
    /// deque, exiting the instant nothing remains anywhere. Completion order and which physical thread
    /// ran which job are **never observable** through this API (RT3-neutral: only the returned,
    /// spawn-order-indexed result vector is visible) — so RT2's deterministic-result default holds
    /// regardless of the steal schedule. Liveness (every job runs exactly once) is `Empirical`.
    ///
    /// **M-864 — persistent pool, help-stealing join, no bare-block (see module docs' correctness
    /// note).** Every lane's deque is populated **before** the lane-loop tasks are submitted, so a
    /// lane never *waits* for work — it drains and exits. The `min(workers, n)` lane-loops are
    /// submitted to the shared [`crate::pool`]; the calling thread then **helps** the pool drain (any
    /// pending task, from this batch or — under nested submission — any other) until every lane of
    /// this batch has finished ([`crate::pool::Pool::help_while`]). Nothing on this batch's own
    /// critical path ever bare-blocks, which is what makes a **nested** `run_indexed` call (submitted
    /// from inside a job running on a pool worker) provably deadlock-free on a **fixed**-size pool.
    ///
    /// The pool queue is **unbounded** — a batch materializes all `n` jobs across its lanes up front,
    /// so its peak pending depth is `n` (memory bounded by the batch size = the `jobs` vector the
    /// caller already holds). There is **no backpressure/`capacity` bound** any more (it was the
    /// deadlock cause — see [`Scheduler::capacity`] and module docs).
    ///
    /// `peak_depth` (when `Some`) records the batch's peak pending depth, which is exactly `n` (all
    /// jobs are enqueued before any lane drains).
    ///
    /// `steal_count` (when `Some`) records how many jobs were completed via a steal (`pop_front` from
    /// another lane's deque) rather than from the popping lane's own deque — a mutant-witness: a
    /// scheduler that silently regressed to single-lane dispatch would report `0` under a
    /// steal-forcing job shape (see `tests::scheduler`).
    ///
    /// # Panic behaviour (M-864, thread::scope-like)
    /// A job that panics does **not** kill the persistent pool worker (each job runs under
    /// [`std::panic::catch_unwind`]) and does **not** hang the join (an RAII drop-guard decrements the
    /// outstanding-lane count on every exit path). The **first** captured job panic is re-raised in
    /// this calling thread once the batch has joined ([`std::panic::resume_unwind`]) — matching
    /// `std::thread::scope`'s panic-propagates-at-join semantics as closely as safe std allows.
    ///
    /// # The `'static` contract (M-864)
    /// `F`/`T` must be `'static`: the shared pool's worker threads outlive this call, so a job can no
    /// longer borrow from the caller's stack frame the way the pre-M-864 `std::thread::scope`
    /// allowed. See the module docs and `docs/notes/DN-67-Persistent-Work-Stealing-Pool.md`.
    ///
    /// Guarantee: outputs equal the sequential reference — **`Empirical`** (RT2; RT1 ⇒
    /// schedule-independence, unaffected by stealing or by the M-864 pool redesign). Pure tasks only
    /// (the [`crate::task`] purity contract is `Declared`).
    #[must_use]
    pub fn run_indexed<T, F>(
        &self,
        jobs: Vec<F>,
        peak_depth: Option<&mut usize>,
        steal_count: Option<&mut usize>,
    ) -> Vec<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.run_indexed_on(&pool::get(), jobs, peak_depth, steal_count)
    }

    /// [`Scheduler::run_indexed`] against an **explicit** pool rather than the process-wide global.
    ///
    /// Public `run_indexed` is exactly `self.run_indexed_on(&pool::get(), …)`. This `pub(crate)`
    /// entry point exists so the tests can drive a batch (and its nested sub-batches) on a pool with
    /// a **forced, small worker count** ([`crate::pool::Pool::with_workers_for_test`]) — the only way
    /// to reproduce the nested-submission deadlock the M-864 rewrite fixes on a machine with many
    /// cores (the global pool is sized to `available_parallelism()`). A nested call inside a job must
    /// pass the **same** pool for the forced-count to hold at every level.
    pub(crate) fn run_indexed_on<T, F>(
        &self,
        pool: &Arc<Pool>,
        jobs: Vec<F>,
        peak_depth: Option<&mut usize>,
        steal_count: Option<&mut usize>,
    ) -> Vec<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let n = jobs.len();
        if n == 0 {
            if let Some(slot) = peak_depth {
                *slot = 0;
            }
            if let Some(slot) = steal_count {
                *slot = 0;
            }
            return Vec::new();
        }

        let lanes = self.workers.min(n); // no point creating more lanes than jobs
        let policy = self.steal_policy;

        // Populate every lane's deque UP FRONT, round-robin by spawn index. No `capacity` gate, no
        // feeder condvar: the whole batch is materialized before any lane runs, so a lane never has
        // to wait for more work (the root fix for the reproduced feeder bare-block — module docs).
        // `deques[i]` is lane `i`'s own deque: LIFO `pop_back` for its own work, FIFO `pop_front`
        // when another lane steals from it.
        let mut initial: Vec<VecDeque<(usize, F)>> = (0..lanes).map(|_| VecDeque::new()).collect();
        for (idx, job) in jobs.into_iter().enumerate() {
            initial[idx % lanes].push_back((idx, job));
        }

        // One lock guards every lane's deque together (so a steal's occupancy snapshot and its
        // `pop_front` are one atomic step — no thief can race a victim empty between them).
        struct Lanes<F> {
            deques: Vec<VecDeque<(usize, F)>>,
            steals: usize,
        }
        let state = Arc::new(Mutex::new(Lanes::<F> {
            deques: initial,
            steals: 0,
        }));
        // Results, pre-sized and written by spawn index → the output stays in spawn order (RT2).
        let results = Arc::new(Mutex::new((0..n).map(|_| None::<T>).collect::<Vec<_>>()));
        // The first job panic captured across all lanes (re-raised after the join — panic-safe join).
        let first_panic: Arc<Mutex<Option<Box<dyn std::any::Any + Send>>>> =
            Arc::new(Mutex::new(None));
        // Outstanding-lane countdown — the join condition `help_while` polls. A per-lane RAII guard
        // decrements it on EVERY exit path (normal or unwind), so no panic can hang the join.
        let remaining = Arc::new(Mutex::new(lanes));

        for me in 0..lanes {
            let state = Arc::clone(&state);
            let results = Arc::clone(&results);
            let first_panic = Arc::clone(&first_panic);
            let guard_remaining = Arc::clone(&remaining);
            let guard_pool = Arc::clone(pool);
            pool.submit(Box::new(move || {
                // The drop-guard: decrement `remaining` and wake the join on EVERY exit path
                // (normal return OR an unexpected unwind), so a panic can never leave the join
                // hanging. Constructed first thing, so it covers the whole lane body.
                let _lane_guard = LaneGuard {
                    remaining: guard_remaining,
                    pool: guard_pool,
                };
                loop {
                    // Pull the next job — own deque first (LIFO), then steal (FIFO) — or EXIT the
                    // instant nothing remains anywhere. Totally non-blocking: every deque was
                    // populated before this lane started, and no work is ever added later, so
                    // "nothing to pop or steal" means the batch is fully claimed → done.
                    let item = {
                        let mut s = state.lock().expect("mycelium-sched: scheduler mutex poisoned");
                        if let Some(item) = s.deques[me].pop_back() {
                            Some(item)
                        } else {
                            let occupancy: Vec<usize> = s.deques.iter().map(VecDeque::len).collect();
                            match policy.select_victim(lanes, me, &occupancy) {
                                Some(decision) => {
                                    let item = s.deques[decision.victim].pop_front().expect(
                                        "victim_depth > 0 under the same held lock ⇒ pop_front succeeds",
                                    );
                                    s.steals += 1;
                                    Some(item)
                                }
                                None => None, // nothing anywhere → this lane is done
                            }
                        }
                    };
                    match item {
                        Some((idx, job)) => {
                            // Run the job under catch_unwind so a panic never kills this persistent
                            // pool worker; capture the first panic to re-raise at the join.
                            match std::panic::catch_unwind(AssertUnwindSafe(job)) {
                                Ok(out) => {
                                    let mut r = results
                                        .lock()
                                        .expect("mycelium-sched: results mutex poisoned");
                                    r[idx] = Some(out);
                                }
                                Err(payload) => {
                                    let mut slot = first_panic
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    if slot.is_none() {
                                        *slot = Some(payload);
                                    }
                                    // Continue draining remaining items: their result slots stay
                                    // `None`, but the captured panic is re-raised before any unwrap,
                                    // so those slots are never read (join-propagation, not liveness).
                                }
                            }
                        }
                        None => break,
                    }
                }
                // `_lane_guard` drops here → `remaining -= 1` + wake the join.
            }));
        }

        // The nested-join wait: help the shared pool drain — this batch's own lanes, or (under nested
        // submission) anyone else's — until every lane of THIS batch has exited. Never a bare block:
        // `help_while` RUNS pending tasks (M-864's deadlock-freedom argument; see `pool` module docs).
        {
            let remaining = Arc::clone(&remaining);
            pool.help_while(move || {
                *remaining
                    .lock()
                    .expect("mycelium-sched: remaining-lanes mutex poisoned")
                    == 0
            });
        }

        // Surface the first job panic (thread::scope-like join propagation) BEFORE touching results.
        if let Some(payload) = first_panic
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            std::panic::resume_unwind(payload);
        }

        if peak_depth.is_some() || steal_count.is_some() {
            let s = state
                .lock()
                .expect("mycelium-sched: scheduler mutex poisoned");
            if let Some(slot) = peak_depth {
                *slot = n; // all n jobs are enqueued before any lane drains
            }
            if let Some(slot) = steal_count {
                *slot = s.steals;
            }
        }

        // Every lane has exited (`remaining == 0`), and each lane's writes into `results`
        // happen-before its guard's decrement, which happens-before this thread observing
        // `remaining == 0` (via the mutex) — so every write is visible here.
        let contents = std::mem::take(
            &mut *results
                .lock()
                .expect("mycelium-sched: results mutex poisoned"),
        );
        // No panic occurred (checked above), so every slot is `Some` (liveness); unwrap in order.
        contents
            .into_iter()
            .map(|o| o.expect("liveness: every submitted job ran exactly once (RT2 join)"))
            .collect()
    }
}

/// RAII guard that decrements a batch's outstanding-lane count on **every** exit path (normal return
/// or an unwind) and wakes the join, so a panicking lane can never leave [`Pool::help_while`] hanging
/// (M-864 panic-safety, Defect 2). Poison-tolerant: a poisoned lock during unwind still decrements
/// (never a hang) via [`std::sync::PoisonError::into_inner`].
struct LaneGuard {
    remaining: Arc<Mutex<usize>>,
    pool: Arc<Pool>,
}

impl Drop for LaneGuard {
    fn drop(&mut self) {
        let mut r = self
            .remaining
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *r = r.saturating_sub(1);
        drop(r);
        // Wake any thread parked in `help_while` so it re-checks its `done` condition promptly
        // (the poll-interval backstop makes this a latency optimization, not a correctness need).
        self.pool.notify_all();
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
