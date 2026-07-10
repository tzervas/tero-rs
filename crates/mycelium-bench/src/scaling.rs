//! **Multicore scaling curves** (M-859) — how a *batch of independent programs* on one backend
//! scales as worker count grows, via the real OS-thread [`Scheduler`](mycelium_std_runtime::scheduler::Scheduler)
//! (M-709/RFC-0008 RT1·RT2). Measurement only: no backend's semantics or execution path changes —
//! this module drives the *same* `run_once` dispatch (`crate::backend::run_once`) that the
//! single-core harness uses, just fanned out across a worker pool.
//!
//! ## What "scaling" means here
//! A **batch** is `N` independent invocations of the *same* corpus case (each one a fresh,
//! self-contained job: its own [`Node`] clone + its own [`Engines`]/artifact — no shared mutable
//! state crosses threads, matching the Scheduler's RT1 purity contract). We time the batch
//! sequentially (1 worker) and then across `2..=workers` OS threads, and report the **speedup**
//! (`t_1worker / t_Nworkers`) against the **ideal-linear** reference (`speedup == workers`).
//!
//! ## Honesty (VR-5)
//! - Every batch timing is **`Empirical`**: a wall-clock measurement over an explicit **trial count**
//!   (`ScalingPoint::batch_size` jobs × `ScalingPoint::repeats` repeated timed batches), on **this
//!   host** (`ScalingRun::host_note`). No number here is pre-written; the whole point is to *find
//!   out* whether/how far each backend scales.
//! - The **Amdahl serial-fraction estimate** derived from a scaling curve is `Empirical` (fit from
//!   the measured points) — it is a *derived statistic*, not a proof; it uses two points only (the
//!   1-worker and max-worker points) rather than a full least-squares fit, so it is a coarse,
//!   explicitly-labeled estimate, never a target.
//! - The **process-spawn-bound backends** (`direct-llvm`, `mlir-dialect`, via
//!   [`Backend::is_process_spawn_bound`]) are flagged in the scaling report exactly as they are in
//!   the single-core report: for a trivial kernel, scaling a *spawn-dominated* per-job cost mostly
//!   measures OS process-creation contention, not kernel compute — surfaced, not buried.
//! - A **skip** (toolchain absent / capability loss on the case) is recorded as [`ScalingOutcome::Skipped`]/
//!   [`ScalingOutcome::Unmeasurable`] — never silently dropped from the curve (G2).

use std::time::Instant;

use mycelium_core::Node;
use mycelium_std_runtime::scheduler::Scheduler;

use crate::backend::{run_once, Backend, Engines, Outcome};
use crate::corpus::Case;

/// One batch-timing measurement: `batch_size` independent jobs run across `workers` OS threads.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ScalingSample {
    /// How many OS worker threads the [`Scheduler`] used for this sample.
    pub workers: usize,
    /// Wall-clock nanoseconds for the whole batch (all `batch_size` jobs), the fastest of `repeats`
    /// timed batches (the same min-of-batches, least-noise convention as [`crate::timing::bench`]).
    pub batch_ns: f64,
    /// Per-job nanoseconds (`batch_ns / batch_size`) — comparable across worker counts.
    pub ns_per_job: f64,
}

/// Why a case could not be scaling-measured on a backend — never-silent (G2): a scaling curve that
/// could not be built still says *why*, it does not just omit the backend.
#[derive(Debug, Clone, serde::Serialize)]
pub enum ScalingOutcome {
    /// Measured: one [`ScalingSample`] per worker count in `1..=workers`.
    Measured(Vec<ScalingSample>),
    /// The backend was skipped for this case (toolchain absent / feature off) — the single-job probe
    /// outcome's reason, carried over verbatim.
    Skipped(String),
    /// The backend cannot lower this case at all (capability loss) — no batch was attempted.
    Unmeasurable(String),
}

/// One case's full scaling curve on one backend.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScalingPoint {
    /// The case id.
    pub case_id: String,
    /// The backend measured.
    pub backend: Backend,
    /// How many independent jobs made up each timed batch (the trial-count denominator).
    pub batch_size: u32,
    /// How many timed batches were run per worker count (the fastest is reported — see
    /// [`ScalingSample::batch_ns`]); this is the trial-count numerator, so the full trial count for
    /// one [`ScalingSample`] is `batch_size * repeats`.
    pub repeats: u32,
    /// The measurement (or the honest reason none was taken).
    pub outcome: ScalingOutcome,
}

impl ScalingPoint {
    /// Speedup at each measured worker count vs the 1-worker sample (`t_1 / t_n`), paired with the
    /// worker count and the **ideal-linear** reference (`== workers`) for comparison. `None` when
    /// this point has no measured samples (skip/unmeasurable/empty).
    #[must_use]
    pub fn speedups(&self) -> Option<Vec<(usize, f64, f64)>> {
        let ScalingOutcome::Measured(samples) = &self.outcome else {
            return None;
        };
        let base = samples.iter().find(|s| s.workers == 1)?.ns_per_job;
        if base <= 0.0 {
            return None;
        }
        Some(
            samples
                .iter()
                .map(|s| {
                    let speedup = if s.ns_per_job > 0.0 {
                        base / s.ns_per_job
                    } else {
                        0.0
                    };
                    #[allow(clippy::cast_precision_loss)]
                    let ideal = s.workers as f64;
                    (s.workers, speedup, ideal)
                })
                .collect(),
        )
    }

    /// A coarse **Amdahl serial-fraction** estimate from the 1-worker and highest-worker-count
    /// samples: solving `speedup = 1 / (s + (1-s)/n)` for `s` at the max measured `n`. `Empirical`
    /// (fit from these two measured points, not a proof) — `None` when there are fewer than two
    /// distinct worker-count samples or the fit is degenerate (speedup <= 0 or >= n, both of which
    /// would imply a nonsensical negative/undefined serial fraction).
    #[must_use]
    pub fn amdahl_serial_fraction(&self) -> Option<f64> {
        let sp = self.speedups()?;
        let (_, s1, _) = *sp.iter().find(|(w, _, _)| *w == 1)?;
        let (n, sn, _) = *sp.iter().max_by_key(|(w, _, _)| *w)?;
        if n <= 1 || s1 <= 0.0 || sn <= 0.0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = n as f64;
        // speedup(n) = 1 / (s + (1-s)/n)  =>  s = (n/speedup - 1) / (n - 1), clamped to [0, 1] since a
        // measured speedup can exceed the Amdahl prediction (superlinear, e.g. cache effects) or fall
        // short of even the fully-serial floor (contention) — both are honestly clamped, not asserted.
        let raw = (n / sn - 1.0) / (n - 1.0);
        Some(raw.clamp(0.0, 1.0))
    }
}

/// The full scaling run: every measured (case, backend) point, plus the host + trial-shape metadata
/// needed to interpret the numbers honestly.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScalingRun {
    /// A short, best-effort host note (arch/os/hw-thread-count) — provenance only, matches the
    /// single-core report's `host_note` convention.
    pub host_note: String,
    /// The worker counts exercised, `1..=max_workers` (ascending; always includes `1` as the
    /// sequential reference the speedup is computed against).
    pub worker_counts: Vec<usize>,
    /// Every measured point (case × backend).
    pub points: Vec<ScalingPoint>,
}

/// Build a fresh [`Node`] + [`Engines`] and run `backend` on `case` once — the unit of work fanned
/// out across the [`Scheduler`]'s worker pool. Each job is fully self-contained (its own clone / own
/// engines / own compile, for the compiled backends), so jobs share **no** mutable state (RT1) and
/// the closure is `Send` without borrowing anything cross-thread.
fn independent_job(case_src: &'static str, backend: Backend) -> impl FnOnce() -> Outcome + Send {
    move || {
        // Re-derive the node from source inside the job (never share a parsed `Node`/`Engines`
        // across threads) — this mirrors "N independent program instances", not "N threads racing
        // one shared artifact". Parse/check/elaborate failures here would be a corpus regression,
        // already caught by `Case::elaborate` at single-core measurement time; a defensive
        // panic-with-context is honest (never a silent wrong-answer substitution, G2).
        let nodule = mycelium_l1::parse(case_src).expect("scaling job: corpus case re-parses");
        let env = mycelium_l1::check_nodule(&nodule).expect("scaling job: corpus case re-checks");
        let node: Node =
            mycelium_l1::elaborate(&env, "main").expect("scaling job: corpus case re-elaborates");
        let eng = Engines::default();
        run_once(backend, &node, &eng)
    }
}

/// Time one batch of `batch_size` independent jobs across exactly `workers` OS threads, taking the
/// fastest of `repeats` timed batches (min-of-batches, the same least-noise convention as
/// [`crate::timing::bench`]). Returns `None` if every job in the batch failed to produce a value
/// (nothing meaningful to time — the caller falls back to the outcome-only probe).
fn time_batch(
    case_src: &'static str,
    backend: Backend,
    workers: usize,
    batch_size: u32,
    repeats: u32,
) -> Option<ScalingSample> {
    let scheduler = Scheduler::with_workers(workers, (workers * 2).max(1)).ok()?;
    let mut best = f64::INFINITY;
    let mut any_value = false;
    for _ in 0..repeats.max(1) {
        let jobs: Vec<_> = (0..batch_size.max(1))
            .map(|_| independent_job(case_src, backend))
            .collect();
        let t = Instant::now();
        // `run_indexed(jobs, peak_depth, steal_count)` — scaling measurement needs neither the
        // queue-depth nor the work-steal instrumentation (M-861 added `steal_count`), so both are
        // `None`; we only want the wall time of the whole batch across `workers` threads.
        let outcomes = scheduler.run_indexed(jobs, None, None);
        let elapsed = t.elapsed();
        if outcomes.iter().any(|o| matches!(o, Outcome::Value(_))) {
            any_value = true;
        }
        #[allow(clippy::cast_precision_loss)]
        let batch_ns = elapsed.as_nanos() as f64;
        best = best.min(batch_ns);
    }
    if !any_value {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let ns_per_job = best / f64::from(batch_size.max(1));
    Some(ScalingSample {
        workers,
        batch_ns: best,
        ns_per_job,
    })
}

/// Measure one case's scaling curve on one backend, across `1..=max_workers` OS threads.
///
/// A single probe run first decides *whether* this (case, backend) pair is even measurable
/// (matches the single-core harness's own probe-before-time discipline in
/// [`crate::backend::warm_runner`]): a `Skipped`/`Unlowerable`/`Error` probe short-circuits to the
/// corresponding never-silent [`ScalingOutcome`] variant with no batch timing attempted.
#[must_use]
pub fn measure_case_scaling(
    case: &Case,
    backend: Backend,
    max_workers: usize,
    batch_size: u32,
    repeats: u32,
) -> ScalingPoint {
    let max_workers = max_workers.max(1);
    let node = case
        .elaborate()
        .unwrap_or_else(|e| panic!("corpus case `{}` failed to elaborate: {e}", case.id));
    let eng = Engines::default();
    let probe = run_once(backend, &node, &eng);

    let outcome = match &probe {
        Outcome::Skipped(reason) => ScalingOutcome::Skipped(reason.clone()),
        Outcome::Unlowerable(reason) => ScalingOutcome::Unmeasurable(reason.clone()),
        Outcome::Error(reason) => ScalingOutcome::Unmeasurable(format!(
            "runtime error on the single-job probe (not a capability boundary, but no batch was \
             attempted): {reason}"
        )),
        Outcome::Value(_) => {
            let samples: Vec<ScalingSample> = (1..=max_workers)
                .filter_map(|w| time_batch(case.src, backend, w, batch_size, repeats))
                .collect();
            if samples.is_empty() {
                // The probe produced a value but every batch attempt failed to reproduce one — an
                // honest, surfaced anomaly rather than a silently-empty curve.
                ScalingOutcome::Unmeasurable(
                    "single-job probe produced a value but no scaling batch did (non-deterministic \
                     or environment-sensitive outcome) — recorded, not hidden"
                        .to_string(),
                )
            } else {
                ScalingOutcome::Measured(samples)
            }
        }
    };

    ScalingPoint {
        case_id: case.id.to_string(),
        backend,
        batch_size,
        repeats,
        outcome,
    }
}

/// Run the scaling suite: every `(case, backend)` pair in `cases` × `Backend::all()` (baseline
/// interpreter included — it is the natural single-core-vs-multicore comparison point too, unlike
/// the WIN/LOSS differential where it is excluded as the anchor). `max_workers` is host-derived by
/// the caller (`Scheduler::new().workers()` is the natural default — see `bin/bench.rs`).
#[must_use]
pub fn run_scaling(
    cases: &[Case],
    max_workers: usize,
    batch_size: u32,
    repeats: u32,
) -> ScalingRun {
    let host_note = crate::host_note_for_scaling();
    let worker_counts: Vec<usize> = (1..=max_workers.max(1)).collect();
    let mut points = Vec::new();
    for case in cases {
        for backend in Backend::all() {
            points.push(measure_case_scaling(
                case,
                backend,
                max_workers,
                batch_size,
                repeats,
            ));
        }
    }
    ScalingRun {
        host_note,
        worker_counts,
        points,
    }
}
