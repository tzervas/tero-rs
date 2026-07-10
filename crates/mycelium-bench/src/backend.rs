//! The backend abstraction: the five execution paths under measurement, and a uniform [`Outcome`]
//! that distinguishes — never-silently (G2) — a produced value, a *graceful skip* (toolchain
//! absent), a *capability loss* (an unlowerable node, with its reason), and a hard error.
//!
//! The [`Backend::Interp`] reference interpreter is the **trusted base** (NFR-7/ADR-007): the
//! differential measures every other backend *against it*. A divergence from the interpreter is a
//! recorded correctness LOSS; an `Unsupported`/`Unlowerable` outcome is a recorded *capability* LOSS.
//!
//! All backends consume the same closed Core IR [`Node`] and dispatch through the same trusted prim
//! registry + certified swap engine, so "two execution paths" cannot mean "two semantics".

use mycelium_cert::BinaryTernarySwapEngine;
use mycelium_core::{CoreValue, Node};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_mlir::AotError;

/// One of the execution backends being measured. `Interp` is the trusted base; the rest are compared
/// against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    /// The reference interpreter (trusted base, NFR-7) — small-step substitution, in-process.
    Interp,
    /// The AOT env-machine (`mycelium_mlir::run_core`) — a big-step ANF evaluator, in-process.
    AotEnv,
    /// The JIT backend (`mycelium_mlir::jit_run`) — emits LLVM IR, `clang -shared`, `dlopen`, call.
    Jit,
    /// The direct-LLVM backend (`mycelium_mlir::compile_and_run`) — emits LLVM IR, `llc`+`clang`,
    /// native artifact, run + read-back stdout.
    DirectLlvm,
    /// The MLIR-dialect path (`mycelium_mlir::mlir_compile_and_run`, behind the `mlir-dialect`
    /// feature) — emits an `arith`/`func` MLIR module, drives it through `mlir-opt`/`mlir-translate`
    /// + `clang`. Records a Skip when the feature is off or the libMLIR toolchain is absent.
    MlirDialect,
}

impl Backend {
    /// All backends, in a stable order (interp first — it is the differential baseline).
    #[must_use]
    pub fn all() -> [Backend; 5] {
        [
            Backend::Interp,
            Backend::AotEnv,
            Backend::Jit,
            Backend::DirectLlvm,
            Backend::MlirDialect,
        ]
    }

    /// A short, stable label for the report (matches the serde rename).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Backend::Interp => "interp",
            Backend::AotEnv => "aot-env",
            Backend::Jit => "jit",
            Backend::DirectLlvm => "direct-llvm",
            Backend::MlirDialect => "mlir-dialect",
        }
    }

    /// Whether this backend is the trusted differential baseline.
    #[must_use]
    pub fn is_baseline(self) -> bool {
        matches!(self, Backend::Interp)
    }

    /// Whether this backend executes a freshly-spawned process per invocation (a microbench caveat
    /// the report surfaces: for a trivial kernel the per-invocation time is spawn-dominated, exactly
    /// the honest M-602/E1 finding). JIT `dlopen`s an in-process `.so` and calls it directly, so it is
    /// **not** spawn-bound; direct-LLVM and the MLIR-dialect path exec a native artifact per call.
    #[must_use]
    pub fn is_process_spawn_bound(self) -> bool {
        matches!(self, Backend::DirectLlvm | Backend::MlirDialect)
    }
}

/// The result of running one backend on one case — uniform across backends, never-silent.
#[derive(Debug, Clone)]
pub enum Outcome {
    /// The backend produced a value (lifted to a [`CoreValue`] for the differential — a bare-`Value`
    /// backend is wrapped as `CoreValue::Repr`). Boxed: a `CoreValue` is far larger than the small
    /// `String` reason variants, so boxing keeps the enum compact (clippy `large_enum_variant`).
    Value(Box<CoreValue>),
    /// The backend was skipped for an *environmental* reason (a missing toolchain / a feature that is
    /// off). NOT a loss — the harness simply could not measure it here. Carries the reason.
    Skipped(String),
    /// The backend *cannot* lower this program — an explicit unlowerable/unsupported node. This is a
    /// recorded **capability loss** with its reason (G2: never omitted).
    Unlowerable(String),
    /// The backend errored at run time (overflow, depth limit, a compile/exec failure that is not a
    /// toolchain-absence skip). A recorded failure, surfaced honestly.
    Error(String),
}

impl Outcome {
    /// A value outcome (boxes the value to keep the enum compact).
    #[must_use]
    pub fn value_outcome(v: CoreValue) -> Self {
        Outcome::Value(Box::new(v))
    }

    /// The produced value, if any.
    #[must_use]
    pub fn value(&self) -> Option<&CoreValue> {
        match self {
            Outcome::Value(v) => Some(v),
            _ => None,
        }
    }

    /// A short status word for the report.
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            Outcome::Value(_) => "value",
            Outcome::Skipped(_) => "skipped",
            Outcome::Unlowerable(_) => "unlowerable",
            Outcome::Error(_) => "error",
        }
    }

    /// The reason string for a non-value outcome (empty for a value).
    #[must_use]
    pub fn reason(&self) -> &str {
        match self {
            Outcome::Value(_) => "",
            Outcome::Skipped(m) | Outcome::Unlowerable(m) | Outcome::Error(m) => m,
        }
    }
}

/// **Observational equivalence** of two results — the honest differential equality. Compares the
/// *observable* (`repr + payload + guarantee` for a repr value; content-identity + guarantee for a
/// datum), **excluding** dynamic `Meta` provenance. This mirrors the M-210 `ObservationalEquiv`
/// checker and the existing three-way differential test (`crates/mycelium-l1/tests/differential.rs`),
/// which compare on the observable, not full structural `Value`/`Meta` equality.
///
/// This matters for honesty: the compiled backends (JIT/direct-LLVM) read a raw value back from
/// native execution and stamp `Provenance::Root` (they carry no derivation chain), while the
/// interpreter records a `Derived` provenance. A full `==` would flag that as a "correctness
/// divergence" even though the *result* (repr+payload+guarantee) is identical — a **false** loss.
/// Provenance is dynamic metadata, explicitly excluded from content identity (RFC-0001 §4.6); a true
/// correctness loss is a different repr/payload/datum or a weaker guarantee, which this does catch.
#[must_use]
pub fn observable_eq(a: &CoreValue, b: &CoreValue) -> bool {
    match (a, b) {
        (CoreValue::Repr(x), CoreValue::Repr(y)) => {
            x.repr() == y.repr()
                && x.payload() == y.payload()
                && x.meta().guarantee() == y.meta().guarantee()
        }
        (CoreValue::Data(x), CoreValue::Data(y)) => {
            // Content hash = constructor identity + field content (Meta/provenance excluded,
            // RFC-0001 §4.6); the guarantee is the meet-summary. Both must match.
            x.content_hash() == y.content_hash() && x.guarantee() == y.guarantee()
        }
        // A repr vs a datum is a genuine divergence.
        _ => false,
    }
}

/// The shared trusted engines every backend dispatches through (prim registry + certified swap).
/// Constructed once and reused, so a backend swap cannot smuggle in a second semantics.
pub struct Engines {
    /// The trusted primitive-op registry.
    pub prims: PrimRegistry,
    /// The certified binary<->ternary swap engine.
    pub swap: BinaryTernarySwapEngine,
}

impl Default for Engines {
    fn default() -> Self {
        Self {
            prims: PrimRegistry::with_builtins(),
            swap: BinaryTernarySwapEngine,
        }
    }
}

/// Map a direct-LLVM/JIT [`AotError`] onto a never-silent [`Outcome`]: a toolchain absence is a
/// graceful `Skipped`; an unsupported node/repr/prim is a `Unlowerable` capability loss; everything
/// else (overflow, depth limit, compile/exec failure) is a recorded `Error`.
fn classify_aot_error(e: &AotError) -> Outcome {
    match e {
        AotError::ToolchainMissing(tool) => {
            Outcome::Skipped(format!("native toolchain absent ({tool})"))
        }
        AotError::UnsupportedRepr(_)
        | AotError::UnsupportedPrim(_)
        | AotError::UnsupportedNode(_)
        | AotError::UnsupportedScheme(_) => Outcome::Unlowerable(e.to_string()),
        // Overflow / depth limit / free-variable / width / compile / run / parse / wf are run-time
        // failures, recorded honestly (not a capability boundary, not an environment skip).
        _ => Outcome::Error(e.to_string()),
    }
}

/// Run the **reference interpreter** (trusted base) on `node`. Spans the whole fragment (repr + data
/// + recursion), returning a [`CoreValue`]. The differential baseline.
#[must_use]
pub fn run_interp(node: &Node, eng: &Engines) -> Outcome {
    let interp = Interpreter::new(
        // A fresh registry/engine clone-equivalent: the interpreter takes ownership, so build its own
        // (same builtins / same certified swap — identical semantics to `eng`).
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let _ = eng; // `eng` documents the shared engines; the interpreter owns an identical pair.
    match interp.eval_core(node) {
        Ok(v) => Outcome::value_outcome(v),
        Err(e) => Outcome::Error(e.to_string()),
    }
}

/// Run the **AOT env-machine** (`mycelium_mlir::run_core`) on `node` — a big-step ANF evaluator,
/// in-process, spanning the whole fragment.
#[must_use]
pub fn run_aot_env(node: &Node, eng: &Engines) -> Outcome {
    match mycelium_mlir::run_core(node, &eng.prims, &eng.swap) {
        Ok(v) => Outcome::value_outcome(v),
        Err(e) => Outcome::Error(e.to_string()),
    }
}

/// Run the **JIT** backend (`mycelium_mlir::jit_run`) on `node`. Bit/trit-neg subset only; an
/// out-of-subset node is a capability loss, a missing `clang` is a graceful skip. Returns a bare
/// `Value` lifted to `CoreValue::Repr`.
#[must_use]
pub fn run_jit(node: &Node) -> Outcome {
    match mycelium_mlir::jit_run(node) {
        Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
        Err(e) => classify_aot_error(&e),
    }
}

/// Run the **direct-LLVM** backend (`mycelium_mlir::compile_and_run`) on `node`. Bit + trit
/// arithmetic + bounded non-recursive data subset; an out-of-subset node is a capability loss, a
/// missing `llc`/`clang` is a graceful skip. Returns a bare `Value` lifted to `CoreValue::Repr`.
#[must_use]
pub fn run_direct_llvm(node: &Node) -> Outcome {
    match mycelium_mlir::compile_and_run(node) {
        Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
        Err(e) => classify_aot_error(&e),
    }
}

/// Run the **MLIR-dialect** backend on `node`. Only compiled in when the `mlir-dialect` feature is
/// on; otherwise this records a `Skipped` ("feature off"). When on, it probes for the libMLIR
/// toolchain at runtime and records a `Skipped` if it is absent, a capability loss on an
/// out-of-element-wise-fragment node, and an error on a compile/run failure. Bare `Value` lifted.
#[must_use]
pub fn run_mlir_dialect(node: &Node) -> Outcome {
    #[cfg(feature = "mlir-dialect")]
    {
        use mycelium_mlir::DialectError;
        // Probe first so an absent toolchain is a clean Skip (never a hard error / silent pass).
        if !mycelium_mlir::MlirTools::is_available() {
            return Outcome::Skipped(
                "MLIR toolchain absent (mlir-opt/mlir-translate/clang not resolved)".into(),
            );
        }
        match mycelium_mlir::mlir_compile_and_run(node) {
            Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
            Err(DialectError::ToolchainMissing(t)) => {
                Outcome::Skipped(format!("MLIR toolchain absent ({t})"))
            }
            Err(DialectError::Unsupported(m)) => Outcome::Unlowerable(m),
            Err(e) => Outcome::Error(e.to_string()),
        }
    }
    #[cfg(not(feature = "mlir-dialect"))]
    {
        let _ = node;
        Outcome::Skipped(
            "the `mlir-dialect` feature is off (build with --features mlir-dialect)".into(),
        )
    }
}

/// Dispatch: run one backend on one node once, returning its [`Outcome`]. (Timing is layered on top
/// in `measure`; this is the single-shot call the timing loop repeats.)
#[must_use]
pub fn run_once(backend: Backend, node: &Node, eng: &Engines) -> Outcome {
    match backend {
        Backend::Interp => run_interp(node, eng),
        Backend::AotEnv => run_aot_env(node, eng),
        Backend::Jit => run_jit(node),
        Backend::DirectLlvm => run_direct_llvm(node),
        Backend::MlirDialect => run_mlir_dialect(node),
    }
}

/// A **warm runner**: a backend prepared so its *per-run* cost can be timed honestly, separating any
/// one-time setup (a compile) from the repeated execution. For the compiled backends this compiles
/// the artifact **once** (recording the one-time compile nanoseconds), then the closure times only
/// `.run()`/`.call()` — so the timed figure is per-invocation, not per-compile (otherwise a "loss"
/// would be wrongly attributed to compile cost). For the in-process backends the closure just re-runs
/// the evaluator.
///
/// The `probe` is the outcome of the first warm run (value / skip / unlowerable / error) — used for
/// the differential + verdict category; `run` repeats the warm execution for timing (cloned via the
/// boxed closure). `compile_ns` is the one-time setup cost when one applied.
pub struct WarmRun {
    /// The probe outcome (the first warm execution's result).
    pub probe: Outcome,
    /// The repeatable warm-execution closure (times only the run, not any compile). `None` when the
    /// probe was not a value (nothing to time).
    pub run: Option<Box<dyn Fn() -> Outcome>>,
    /// The one-time setup (compile) cost in nanoseconds, when a compile happened (compiled backends).
    pub compile_ns: Option<f64>,
}

impl WarmRun {
    fn in_process(probe: Outcome, run: Box<dyn Fn() -> Outcome>) -> Self {
        let run = matches!(probe, Outcome::Value(_)).then_some(run);
        Self {
            probe,
            run,
            compile_ns: None,
        }
    }

    fn not_timed(probe: Outcome, compile_ns: Option<f64>) -> Self {
        Self {
            probe,
            run: None,
            compile_ns,
        }
    }
}

/// Build a [`WarmRun`] for one backend on one node — the compile-once/run-many split that makes the
/// per-run timing honest.
#[must_use]
pub fn warm_runner(backend: Backend, node: &Node, eng: &Engines) -> WarmRun {
    match backend {
        // In-process: prepare the evaluator ONCE (outside the timed closure) and reuse it across the
        // timed runs — so the per-run number measures evaluation, not Interpreter/registry/engine
        // construction. This gives the in-process backends the same compile-once/run-many parity the
        // compiled backends below get, so the baseline/AOT numbers are comparable to JIT/direct-LLVM
        // (Copilot #265). We clone the node into the closure so it is `'static`.
        Backend::Interp => {
            let probe = run_interp(node, eng);
            let n = node.clone();
            // Build the interpreter once; `eval_core(&self, …)` serves every timed run (same builtins /
            // certified swap `run_interp` uses — identical semantics, construction just hoisted out).
            let interp = Interpreter::new(
                PrimRegistry::with_builtins(),
                Box::new(BinaryTernarySwapEngine),
            );
            WarmRun::in_process(
                probe,
                Box::new(move || match interp.eval_core(&n) {
                    Ok(v) => Outcome::value_outcome(v),
                    Err(e) => Outcome::Error(e.to_string()),
                }),
            )
        }
        Backend::AotEnv => {
            let probe = run_aot_env(node, eng);
            let n = node.clone();
            // Build the engines once and reuse them — `run_aot_env` reads `eng.prims`/`eng.swap`, so a
            // single owned `Engines` avoids rebuilding `PrimRegistry::with_builtins()` on every run.
            let owned = Engines::default();
            WarmRun::in_process(probe, Box::new(move || run_aot_env(&n, &owned)))
        }
        // Compiled: compile ONCE (timed), then the closure times only the artifact call.
        Backend::Jit => warm_jit(node),
        Backend::DirectLlvm => warm_direct_llvm(node),
        Backend::MlirDialect => warm_mlir_dialect(node),
    }
}

/// JIT: `compile_so` once (timed), then time `artifact.call()` (in-process `dlopen`ed `.so`).
fn warm_jit(node: &Node) -> WarmRun {
    use std::rc::Rc;
    use std::time::Instant;
    let t = Instant::now();
    match mycelium_mlir::compile_so(node) {
        Ok(artifact) => {
            #[allow(clippy::cast_precision_loss)]
            let compile_ns = t.elapsed().as_nanos() as f64;
            // Probe once for the outcome category.
            let probe = match artifact.call() {
                Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
                Err(e) => classify_aot_error(&e),
            };
            let art = Rc::new(artifact);
            let run: Box<dyn Fn() -> Outcome> = {
                let art = Rc::clone(&art);
                Box::new(move || match art.call() {
                    Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
                    Err(e) => classify_aot_error(&e),
                })
            };
            let mut w = WarmRun::in_process(probe, run);
            w.compile_ns = Some(compile_ns);
            w
        }
        // Compile failed: a capability loss / skip / error, with no warm phase to time.
        Err(e) => WarmRun::not_timed(classify_aot_error(&e), None),
    }
}

/// Direct-LLVM: `compile` once (timed), then time `artifact.run()` (native artifact exec per call —
/// process-spawn-bound, captioned as such in the report).
fn warm_direct_llvm(node: &Node) -> WarmRun {
    use std::rc::Rc;
    use std::time::Instant;
    let t = Instant::now();
    match mycelium_mlir::compile(node) {
        Ok(artifact) => {
            #[allow(clippy::cast_precision_loss)]
            let compile_ns = t.elapsed().as_nanos() as f64;
            let probe = match artifact.run() {
                Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
                Err(e) => classify_aot_error(&e),
            };
            let art = Rc::new(artifact);
            let run: Box<dyn Fn() -> Outcome> = {
                let art = Rc::clone(&art);
                Box::new(move || match art.run() {
                    Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
                    Err(e) => classify_aot_error(&e),
                })
            };
            let mut w = WarmRun::in_process(probe, run);
            w.compile_ns = Some(compile_ns);
            w
        }
        Err(e) => WarmRun::not_timed(classify_aot_error(&e), None),
    }
}

/// MLIR-dialect: `mlir_compile` once (timed), then time `artifact.run()`. Feature-gated; skips when
/// the feature is off or the toolchain is absent.
fn warm_mlir_dialect(node: &Node) -> WarmRun {
    #[cfg(feature = "mlir-dialect")]
    {
        use mycelium_mlir::DialectError;
        use std::rc::Rc;
        use std::time::Instant;
        if !mycelium_mlir::MlirTools::is_available() {
            return WarmRun::not_timed(
                Outcome::Skipped(
                    "MLIR toolchain absent (mlir-opt/mlir-translate/clang not resolved)".into(),
                ),
                None,
            );
        }
        let t = Instant::now();
        match mycelium_mlir::mlir_compile(node) {
            Ok(artifact) => {
                #[allow(clippy::cast_precision_loss)]
                let compile_ns = t.elapsed().as_nanos() as f64;
                let probe = match artifact.run() {
                    Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
                    Err(DialectError::ToolchainMissing(tt)) => {
                        Outcome::Skipped(format!("MLIR toolchain absent ({tt})"))
                    }
                    Err(DialectError::Unsupported(m)) => Outcome::Unlowerable(m),
                    Err(e) => Outcome::Error(e.to_string()),
                };
                let art = Rc::new(artifact);
                let run: Box<dyn Fn() -> Outcome> = {
                    let art = Rc::clone(&art);
                    Box::new(move || match art.run() {
                        Ok(v) => Outcome::value_outcome(CoreValue::Repr(v)),
                        Err(DialectError::ToolchainMissing(tt)) => {
                            Outcome::Skipped(format!("MLIR toolchain absent ({tt})"))
                        }
                        Err(DialectError::Unsupported(m)) => Outcome::Unlowerable(m),
                        Err(e) => Outcome::Error(e.to_string()),
                    })
                };
                let mut w = WarmRun::in_process(probe, run);
                w.compile_ns = Some(compile_ns);
                w
            }
            Err(DialectError::ToolchainMissing(tt)) => WarmRun::not_timed(
                Outcome::Skipped(format!("MLIR toolchain absent ({tt})")),
                None,
            ),
            Err(DialectError::Unsupported(m)) => WarmRun::not_timed(Outcome::Unlowerable(m), None),
            Err(e) => WarmRun::not_timed(Outcome::Error(e.to_string()), None),
        }
    }
    #[cfg(not(feature = "mlir-dialect"))]
    {
        let _ = node;
        WarmRun::not_timed(
            Outcome::Skipped(
                "the `mlir-dialect` feature is off (build with --features mlir-dialect)".into(),
            ),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::corpus;

    #[test]
    fn interp_and_aot_agree_on_every_corpus_case() {
        // The trusted base and the AOT env-machine both span the whole fragment — they must agree on
        // every case (this is the M-151/M-342 differential, re-pinned here as a harness sanity gate).
        let eng = Engines::default();
        for case in corpus() {
            let node = case.elaborate().expect("elaborates");
            let i = run_interp(&node, &eng);
            let a = run_aot_env(&node, &eng);
            let (iv, av) = (
                i.value().expect("interp produces a value"),
                a.value().expect("aot produces a value"),
            );
            // Compare the OBSERVABLE (repr+payload+guarantee / content-identity, provenance excluded —
            // RFC-0001 §4.6), the same differential obligation the harness itself uses and that
            // mycelium-l1/tests/differential.rs follows: robust to provenance/meta-only differences,
            // still catching a true repr/payload/guarantee divergence (Copilot #265).
            assert!(
                observable_eq(iv, av),
                "interp vs AOT diverged (observable: repr+payload+guarantee) on `{}`",
                case.id
            );
        }
    }

    #[test]
    fn recursion_and_swap_are_capability_losses_and_data_is_never_silent() {
        // The honest, *measured* capability boundary (RFC-0004 §2/§11) — note it is more nuanced than
        // a flat "compiled paths can't do data": both compiled backends can lower a **flat
        // non-recursive match/construct to a repr** (e.g. `Sign -> Ternary{1}`), so a `Data` case may
        // legitimately produce a VALUE. What is *always* a capability loss is **recursion**
        // (`Fix`/`FixGroup`) — neither compiled backend lowers that (no native call-frame/stack
        // machinery in the emitted IR).
        //
        // `Swap` is **no longer** in that always-a-loss bucket: M-852 (PR #823) landed native `Swap`
        // codegen for the certified binary<->ternary LEGAL-pair class in the shared `lower_program`
        // path both JIT (`jit.rs::emit_kernel_fn`) and direct-LLVM (`llvm.rs::compile_and_run`) route
        // through (`swap_codegen::lower_swap`) — so a swap case can now legitimately produce a VALUE
        // on both compiled backends, exactly like a flat data case. Verified directly: on this
        // corpus's `swap-roundtrip` case (a legal `(8,6)` binary<->ternary<->binary round trip), both
        // JIT and direct-LLVM now return `Outcome::Value` whose `repr`/`payload`/`guarantee` match the
        // trusted interpreter's (`0b0010_1010` round-trips losslessly; only `Meta::provenance` differs
        // — `Root` vs `Derived`, the documented dynamic-metadata exclusion, RFC-0001 §4.6). An
        // *illegal*-pair or otherwise-unsupported swap is still refused explicitly
        // (`AotError::UnsupportedNode`/`UnsupportedScheme`, never a silently-wrong transcode — G2/
        // VR-5), so `Swap` now shares `Data`'s never-silent obligation rather than recursion's
        // always-a-loss one. This is the prior test's stale assumption corrected, not weakened: it
        // used to hard-code "swap is always a capability loss" as a pre-asserted fact, which is
        // exactly the VR-5 anti-pattern the `Data` arm already avoided — folding `Swap` into that arm
        // fixes the staleness while keeping the real obligation (never a wrong answer slipped through)
        // fully enforced.
        //
        // So we assert two things, both honest:
        //  1. recursion ⇒ a capability loss / skip on BOTH compiled backends, and
        //  2. on data + swap, BOTH compiled backends are **never-silent**: a value, a capability loss,
        //     or a skip — never a wrong answer slipped through (the harness's whole purpose).
        // (Which data/swap cases each backend actually lowers is a *measured* output of the harness,
        // not a pre-asserted fact — VR-5: we don't hard-code a capability claim we'd have to keep in
        // sync.)
        use crate::corpus::Fragment;
        for case in corpus() {
            let node = case.elaborate().expect("elaborates");

            match case.fragment {
                // (1) recursion: always out of the compiled subset.
                Fragment::Recursion => {
                    for (label, outcome) in [
                        ("jit", run_jit(&node)),
                        ("direct-llvm", run_direct_llvm(&node)),
                    ] {
                        assert!(
                            matches!(outcome, Outcome::Unlowerable(_) | Outcome::Skipped(_)),
                            "{label} must record a capability loss / skip on `{}` ({}), got {:?}",
                            case.id,
                            case.fragment.label(),
                            outcome.status()
                        );
                    }
                }
                // (2) data + swap: never-silent on both (value OR explicit capability loss / skip).
                Fragment::Data | Fragment::Swap => {
                    for (label, outcome) in [
                        ("jit", run_jit(&node)),
                        ("direct-llvm", run_direct_llvm(&node)),
                    ] {
                        assert!(
                            matches!(
                                outcome,
                                Outcome::Value(_) | Outcome::Unlowerable(_) | Outcome::Skipped(_)
                            ),
                            "{label} on {} `{}` must be never-silent (value / capability loss / \
                             skip), got {:?}",
                            case.fragment.label(),
                            case.id,
                            outcome.status()
                        );
                    }
                }
                Fragment::BitSubset => { /* covered by the differential test */ }
            }
        }
    }
}
