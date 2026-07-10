//! Dependency-light timing, in the house style (no `criterion`): a warmup pass, then the minimum
//! mean over several batches (the fastest batch's per-call mean — the least-noise estimate).
//! Mirrors `xtask/src/e1.rs::bench` so the two perf harnesses time the same way.
//!
//! **Honesty:** every number this produces is `Empirical` — a measurement, never a target (VR-5).
//! A debug build is *refused* for perf numbers ([`refuse_debug_build`]); micro-timing caveats
//! (warmup, process-spawn cost for the compiled paths, debug-vs-release) are surfaced in the report,
//! not buried.

use std::hint::black_box;
use std::time::Instant;

/// Default number of timed batches; the fastest batch's mean is reported (least-noise estimate).
pub const BATCHES: u32 = 5;

/// A single backend/case timing result: the per-call nanoseconds and the trial accounting that makes
/// the number honest (warmup iters, timed iters, batches, the batch spread).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Timing {
    /// The reported per-call time: the mean of the fastest batch, in nanoseconds.
    pub ns_per_call: f64,
    /// Iterations per batch (also the warmup count).
    pub iters: u32,
    /// Number of timed batches.
    pub batches: u32,
    /// The slowest batch's per-call mean, in ns — the honest upper end of the observed spread.
    pub ns_per_call_worst: f64,
}

impl Timing {
    /// The observed best/worst spread ratio (`worst / best`), `1.0` when only one batch was timed or
    /// the best was zero. A large ratio is a noise flag the report surfaces (microbench caveat).
    #[must_use]
    pub fn spread(&self) -> f64 {
        if self.ns_per_call > 0.0 {
            self.ns_per_call_worst / self.ns_per_call
        } else {
            1.0
        }
    }
}

/// Time `f`: `iters` warmup calls, then [`BATCHES`] timed batches of `iters` calls each; report the
/// fastest batch's per-call mean (and keep the slowest for the honest spread). `f` is `black_box`-fed
/// at the call site by the caller; this only boxes nothing itself beyond the loop fence.
///
/// `iters` is floored to `1` (never-panic: a zero is treated as one rather than dividing by zero or
/// asserting). The closure is run `iters.max(1) * (1 + BATCHES)` times in total.
#[must_use]
pub fn bench(iters: u32, mut f: impl FnMut()) -> Timing {
    // Floor to 1 so the per-call division below can never divide by zero (defensive, not asserted —
    // the harness should degrade gracefully, not panic, on a misconfigured iteration count).
    let iters = iters.max(1);

    // Warmup — fill caches / branch predictors / let the allocator settle.
    for _ in 0..iters {
        f();
    }

    let mut best = f64::INFINITY;
    let mut worst = 0.0_f64;
    for _ in 0..BATCHES {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = t.elapsed();
        #[allow(clippy::cast_precision_loss)]
        let per_call = elapsed.as_nanos() as f64 / f64::from(iters);
        best = best.min(per_call);
        worst = worst.max(per_call);
    }

    Timing {
        ns_per_call: best,
        iters,
        batches: BATCHES,
        ns_per_call_worst: worst,
    }
}

/// `true` when this binary was compiled with debug assertions on (a debug build). Perf numbers from a
/// debug build are meaningless; the harness refuses them.
#[must_use]
pub fn is_debug_build() -> bool {
    cfg!(debug_assertions)
}

/// Refuse to produce perf numbers from a debug build — print the fix and exit `2`. Called by the
/// `bench` binary before any timing. (A debug build leaves overflow checks + no optimisation on, so a
/// "win" or "loss" measured there would be a lie; G2 — never a misleading number.)
pub fn refuse_debug_build() {
    if is_debug_build() {
        eprintln!(
            "mycelium-bench: refusing to measure a debug build — run with `--release` \
             (`cargo run --release -p mycelium-bench --bin bench`). A debug build's timings are not \
             representative (no optimisation, overflow checks on), so a WIN/LOSS verdict from it \
             would be dishonest (VR-5/G2)."
        );
        std::process::exit(2);
    }
}

/// A tiny self-test fence so the closure is not optimised away in our own unit tests.
#[doc(hidden)]
#[must_use]
pub fn fence<T>(t: T) -> T {
    black_box(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_runs_the_closure_and_reports_a_finite_time() {
        let mut counter = 0u64;
        let t = bench(100, || {
            counter = counter.wrapping_add(fence(1));
        });
        // The closure ran warmup + BATCHES*iters times; the counter advanced accordingly.
        assert_eq!(counter, 100 * (1 + u64::from(BATCHES)));
        assert!(t.ns_per_call.is_finite());
        assert_eq!(t.iters, 100);
        assert_eq!(t.batches, BATCHES);
        // best <= worst by construction.
        assert!(t.ns_per_call <= t.ns_per_call_worst + f64::EPSILON);
        assert!(t.spread() >= 1.0);
    }

    #[test]
    fn iters_floor_is_one() {
        let mut ran = 0u32;
        let _ = bench(0, || ran += 1);
        // even with iters=0 requested, the floor of 1 means it ran 1 + BATCHES times.
        assert_eq!(ran, 1 + BATCHES);
    }
}
