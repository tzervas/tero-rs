//! Drive every backend over every corpus case: capture each backend's [`Outcome`] and (when it
//! produced a value) its [`Timing`], then classify it against the trusted interpreter into a
//! [`Verdict`]. The output is a structured [`RunRecord`] the report module renders.
//!
//! **Honesty in the timing budget:** in-process backends (interp, AOT env-machine, JIT) are timed
//! with many iterations; the process-spawn-bound backends (direct-LLVM, MLIR-dialect) are timed with
//! far fewer — each call spawns a fresh native process — and the report captions that this figure is
//! spawn-dominated for a trivial kernel (M-602/E1). The compiled paths also compile-once per case
//! before timing (so the timed figure is per-run, not per-compile); a compile failure is recorded.
//!
//! No correctness is assumed: the differential is computed *every* run, so a divergence is caught.

use mycelium_core::Node;

use crate::backend::{warm_runner, Backend, Engines, Outcome};
use crate::corpus::{Case, Fragment};
use crate::timing::{bench, Timing};
use crate::verdict::{classify, Verdict};

/// Per-backend timing budgets (warmup == timed iters per batch). In-process paths get many iters;
/// spawn-bound native paths get few (each call is a process spawn). Reified here (no black box).
#[must_use]
fn iters_for(backend: Backend) -> u32 {
    match backend {
        // In-process, cheap per call.
        Backend::Interp => 20_000,
        Backend::AotEnv => 20_000,
        // In-process but heavier (dlopen + FFI call).
        Backend::Jit => 2_000,
        // Each call spawns + execs a native artifact — keep the count small (mirrors xtask e1 §2).
        Backend::DirectLlvm | Backend::MlirDialect => 40,
    }
}

/// One backend's measured result on one case: its outcome, optional timing, and the classified
/// verdict vs the interpreter.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendResult {
    /// Which backend.
    pub backend: Backend,
    /// A short status word (`value` / `skipped` / `unlowerable` / `error`).
    pub outcome_status: &'static str,
    /// The non-value reason (empty for a value outcome).
    pub outcome_reason: String,
    /// The per-run timing, if a value was produced and it was timed (the warm `.run()`/`.call()` /
    /// in-process eval cost — NOT including any one-time compile).
    pub timing: Option<Timing>,
    /// The one-time setup (compile) nanoseconds for a compiled backend, when a compile happened.
    /// Reported separately so the per-run `timing` stays honest (compile cost is amortized, not
    /// charged to every invocation).
    pub compile_ns: Option<f64>,
    /// The verdict vs the trusted interpreter baseline.
    pub verdict: Verdict,
}

/// All backends' results on one case.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CaseRecord {
    /// The case id.
    pub id: String,
    /// The case's fragment.
    pub fragment: Fragment,
    /// The case's one-line note.
    pub note: String,
    /// The v0-calculus source (for reproducibility / audit).
    pub src: String,
    /// The interpreter baseline's per-call time (ns), if it produced a value — the comparison anchor.
    pub baseline_ns: Option<f64>,
    /// Each non-baseline backend's result (interp is excluded — it *is* the baseline).
    pub backends: Vec<BackendResult>,
}

/// The full execution-backend run: every case's record, in corpus order.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunRecord {
    /// Per-case records.
    pub cases: Vec<CaseRecord>,
}

/// Run + time one backend on one already-elaborated node using the compile-once/run-many
/// [`warm_runner`]: the compiled paths compile their artifact **once** (its cost captured separately
/// as `compile_ns`), then only the warm `.run()`/`.call()` is timed — so the per-run figure is not
/// polluted by per-call compilation. A capability loss / skip / error is detected by the probe and is
/// never timed. Returns `(probe_outcome, per_run_timing?, compile_ns?)`.
fn measure_backend(
    backend: Backend,
    node: &Node,
    eng: &Engines,
) -> (Outcome, Option<Timing>, Option<f64>) {
    let warm = warm_runner(backend, node, eng);
    let compile_ns = warm.compile_ns;
    let timing = warm.run.as_ref().map(|run| {
        let iters = iters_for(backend);
        bench(iters, || {
            // Correctness is asserted via the probe's value + the verdict; here we only time.
            let _ = std::hint::black_box(run());
        })
    });
    (warm.probe, timing, compile_ns)
}

/// Measure all backends on one case and classify each vs the interpreter baseline.
#[must_use]
pub fn measure_case(case: &Case, eng: &Engines) -> CaseRecord {
    let node = case
        .elaborate()
        .unwrap_or_else(|e| panic!("corpus case `{}` failed to elaborate: {e}", case.id));

    // The trusted baseline first.
    let (interp_outcome, interp_timing, _) = measure_backend(Backend::Interp, &node, eng);
    let baseline_ns = interp_timing.map(|t| t.ns_per_call);

    let mut backends = Vec::new();
    for backend in Backend::all() {
        if backend.is_baseline() {
            continue; // the interpreter is the anchor, not a row compared against itself.
        }
        let (outcome, timing, compile_ns) = measure_backend(backend, &node, eng);
        let verdict = classify(
            backend,
            (&interp_outcome, interp_timing),
            (&outcome, timing),
        );
        backends.push(BackendResult {
            backend,
            outcome_status: outcome.status(),
            outcome_reason: outcome.reason().to_string(),
            timing,
            compile_ns,
            verdict,
        });
    }

    CaseRecord {
        id: case.id.to_string(),
        fragment: case.fragment,
        note: case.note.to_string(),
        src: case.src.to_string(),
        baseline_ns,
        backends,
    }
}

/// Run the whole corpus, in order. (The caller supplies the corpus so a focused run is possible.)
#[must_use]
pub fn run_corpus(cases: &[Case], eng: &Engines) -> RunRecord {
    let cases = cases.iter().map(|c| measure_case(c, eng)).collect();
    RunRecord { cases }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::corpus;

    #[test]
    fn measuring_a_bit_case_yields_a_baseline_and_per_backend_verdicts() {
        let eng = Engines::default();
        let case = corpus()
            .into_iter()
            .find(|c| c.id == "bit-xor-not")
            .expect("the bit-xor-not case exists");
        let rec = measure_case(&case, &eng);
        assert_eq!(rec.id, "bit-xor-not");
        // The interpreter baseline must have produced a timed value.
        assert!(
            rec.baseline_ns.is_some(),
            "the trusted base must produce a baseline time"
        );
        // Every non-baseline backend has a verdict (4 of them).
        assert_eq!(rec.backends.len(), 4);
        // The AOT env-machine must agree with the interpreter (a value verdict, never a correctness
        // loss / baseline failure) — it spans the same fragment.
        let aot = rec
            .backends
            .iter()
            .find(|b| b.backend == Backend::AotEnv)
            .unwrap();
        assert!(
            !matches!(
                aot.verdict,
                Verdict::CorrectnessLoss { .. } | Verdict::BaselineFailed { .. }
            ),
            "AOT must not diverge from the interpreter on a bit case: {:?}",
            aot.verdict
        );
    }

    #[test]
    fn a_recursion_case_records_capability_loss_for_compiled_paths() {
        let eng = Engines::default();
        let case = corpus()
            .into_iter()
            .find(|c| c.id == "rec-self")
            .expect("the rec-self case exists");
        let rec = measure_case(&case, &eng);
        // The interpreter + AOT handle recursion; JIT + direct-LLVM cannot — capability loss or skip.
        for b in &rec.backends {
            match b.backend {
                Backend::AotEnv => assert!(
                    !matches!(b.verdict, Verdict::CapabilityLoss { .. }),
                    "AOT env-machine handles recursion"
                ),
                Backend::Jit | Backend::DirectLlvm => assert!(
                    matches!(
                        b.verdict,
                        Verdict::CapabilityLoss { .. } | Verdict::Skipped { .. }
                    ),
                    "compiled path must record a capability loss / skip on recursion: {:?}",
                    b.verdict
                ),
                Backend::MlirDialect => { /* feature-gated; skip or capability loss either way */ }
                Backend::Interp => unreachable!("interp is the baseline, not a row"),
            }
        }
    }
}
