//! Harness-level parallel dispatch for the **AOT-compiled (direct-LLVM)** and **in-process JIT**
//! execution paths (M-865; RFC-0008 §8; DN-61 §A.2; scope ratified by the maintainer 2026-07-01).
//!
//! # What this is — and is not
//! M-862 gave the reference **interpreter** a bounded, top-level-only parallel-eval path
//! ([`mycelium_interp::parallel`]): the direct argument list of a top-level, pure, ≥2-argument
//! `Node::Op`/`Node::Construct` is fanned across [`mycelium_sched::scheduler::Scheduler::run_indexed`],
//! each argument then reduced sequentially by the trusted small-step interpreter. This module extends
//! *that exact fragment gate* to the two compiled execution paths this crate owns — **not** a new
//! LLVM-IR-level concurrency primitive, **not** a second scheduler, and **not** "AOT hypha/colony/async
//! parity" (the language has no executable concurrency surface today — `hypha` is ratified-not-lexed,
//! `async` is unimplemented, and both the interpreter and every compiled path here still run one
//! sequential program; M-869 tracks the real concurrency-parity workstream once the language has a
//! spawn/hypha surface to drive it). The gap this module closes is narrower and already real: the AOT/
//! JIT paths had **no** harness-level parallel dispatch of their own to validate against M-862's, even
//! though M-860 already proved the pattern (parallel *codegen* of independent top-level programs via
//! this same `Scheduler`) for a sibling case.
//!
//! # Dispatch, at the Rust harness level (reusing M-860's precedent, no new bridge)
//! [`plan_concurrent`] reuses [`mycelium_interp::parallel::plan_parallel`] verbatim (DRY/KC-3 — one
//! purity/shape gate, never a re-derived copy) and narrows it to the **`Op`-headed** case only (see
//! below for why `Construct` stays out of *this* dispatcher's scope). For an eligible node:
//! - each argument is submitted as its own job to `Scheduler::run_indexed` (the same M-861/M-864
//!   work-stealing pool `mycelium_sched` already provides, and the same entry point M-860's
//!   `emit_llvm_ir_many` dispatches through — zero new scheduler surface);
//! - each job runs the **exact same trusted runner** the sequential reference uses for that path
//!   (`llvm::compile_and_run_with_swap_mode` for AOT, `jit::jit_run` for JIT) on its own argument,
//!   compiling/running it as its own **closed, standalone program** — valid because a top-level
//!   batch argument is by construction a closed, pure sub-fragment (RFC-0008 §4.2), so evaluating it
//!   standalone is observably identical to evaluating it in place;
//! - the batch's results are then **recomposed by re-invoking that same trusted runner once more** on
//!   a tiny reconstructed node (the original `prim` applied to `Node::Const` of each computed result)
//!   — so composition (prim application) is never hand-reimplemented at the Rust level; it goes
//!   through the identical codegen/JIT path the sequential reference itself uses. This keeps the
//!   "no shim / no second semantics" property M-865's own issue body asks about: the only new code is
//!   *scheduling* (which argument runs where), never *meaning*.
//!
//! Any fragment outside this gate (impure, fewer than two top-level args, or **not** headed by `Op`)
//! runs wholesale through the ordinary sequential entry point — never a partial/mixed order (G2).
//!
//! # Honest scope narrowing: `Op`-headed batches only, not `Construct` (grounded, not silent)
//! M-862's interpreter-side fragment also covers a top-level `Construct` (an "independent pure
//! Construct elements" batch, per the M-862 issue). This harness-level dispatcher does **not** extend
//! to that case, for a concrete, checked reason rather than a hand-wave: the direct-LLVM whole-program
//! contract requires the **top-level result to reduce to a representation `Lane`**
//! (`lower_program_with_swap_mode`'s `result_ev.into_lane(...)` in `llvm.rs`) — a bare top-level
//! `Construct` lowers to an `EnvValue::Datum`, which `into_lane` explicitly refuses. Every existing
//! `Construct` case in the codegen differentials (`tests/unified_threeway_differential.rs`) is
//! therefore always wrapped in a `Match` that extracts a `Lane`-valued field before it reaches
//! `compile_and_run`/`mlir_compile_and_run` — there is no standalone "compile a bare `Construct`"
//! entry point to submit as a per-argument job or recompose through in the first place. Composing
//! parallel per-field AOT results back into a native `Datum` would require new Datum-lane composition
//! plumbing this issue's harness-level-reuse mandate does not call for. This is flagged, not dropped:
//! extending the batch to `Construct` heads (and to the MLIR-dialect leg) is explicitly open follow-on
//! work, not claimed here.
//!
//! JIT carries its own pre-existing, independently-documented `Construct`/`Match` exclusion (M-727;
//! `tests/unified_threeway_differential.rs` module docs) — so narrowing this dispatcher to `Op` heads
//! costs JIT nothing it did not already lack, and only narrows the *newly-added* AOT-parallel surface
//! to match a boundary that already existed on the JIT side.
//!
//! # Determinism (Exact by construction) — extends M-860's argument
//! Every argument job is a pure function of its own closed `Node` (fresh `Ssa`/`Bbc` counters, no
//! shared mutable state across jobs — the same non-argument M-860 already makes for
//! `emit_llvm_ir_many`), and `Scheduler::run_indexed` returns outputs in **spawn order** (never
//! completion order), so the batch's `Vec` of computed argument values is deterministic input order —
//! restoring the exact original argument order before recomposition, regardless of worker count or
//! steal schedule. So `compile_and_run_concurrent(node) == compile_and_run(node)` and
//! `jit_run_concurrent(node) == jit_run(node)`, byte-for-byte — asserted by the differential in
//! `tests/concurrent_threeway_differential.rs`.
//!
//! # Tag: Empirical (differential-checked)
//! `compile_and_run_concurrent(e) == compile_and_run(e)` and `jit_run_concurrent(e) == jit_run(e)` for
//! `e` in the `Op`-headed pure batch fragment are checked by a corpus differential
//! (`tests/concurrent_threeway_differential.rs`) plus a mutant witness, never proven — **Empirical** on
//! the transparency lattice (never upgraded to `Proven` without a checked side-condition, VR-5).

use std::sync::Arc;

use mycelium_core::Node;
use mycelium_interp::parallel::{plan_parallel, BatchHead, ParallelPlan};
use mycelium_sched::scheduler::Scheduler;

use crate::jit;
use crate::llvm::{self, AotError};
use crate::swap_codegen::SwapCertMode;

/// The **reified, EXPLAIN-able** decision this module's dispatchers make for a given fragment — never
/// a silent/opaque choice (house rule #2/G2). Deliberately narrower than
/// [`mycelium_interp::parallel::ParallelPlan`]: a `Construct`-headed batch is folded into
/// [`ConcurrentPlan::Sequential`] here (see module docs for why), so a caller can tell — just from
/// this type — exactly which fragments this harness-level dispatcher parallelizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrentPlan {
    /// Not an eligible batch for *this* dispatcher — impure, fewer than two top-level arguments, a
    /// non-`Op` head (including a pure `Construct` batch — out of this dispatcher's scope, see module
    /// docs), or a `Let`/`Match`/`App`/`Fix`/… head. Runs wholly through the ordinary sequential entry
    /// point, never reordered.
    Sequential,
    /// A pure, top-level `Op` with **≥2** independent arguments — the batch this dispatcher fans out
    /// across [`Scheduler::run_indexed`], one job per argument (a single, non-nested fan-out, mirroring
    /// M-862's own top-level-only bound).
    OpBatch {
        /// The batch width (number of independent arguments fanned out).
        width: usize,
    },
}

/// Compute the [`ConcurrentPlan`] for `node` — reuses
/// [`mycelium_interp::parallel::plan_parallel`] verbatim for the purity/shape gate (DRY/KC-3: one
/// fragment definition, never a re-derived copy that could silently drift from M-862's), narrowed to
/// the `Op`-headed case this module's dispatchers actually parallelize.
#[must_use]
pub fn plan_concurrent(node: &Node) -> ConcurrentPlan {
    match plan_parallel(node) {
        ParallelPlan::TopLevelBatch {
            head: BatchHead::Op,
            width,
        } => ConcurrentPlan::OpBatch { width },
        // Impure, no batch, or a `Construct`-headed batch (out of this dispatcher's scope; module
        // docs) — all fold to the same wholesale-sequential outcome.
        ParallelPlan::SequentialImpure
        | ParallelPlan::SequentialNoBatch
        | ParallelPlan::TopLevelBatch {
            head: BatchHead::Construct,
            ..
        } => ConcurrentPlan::Sequential,
    }
}

/// Fan an `Op`-headed top-level batch's arguments across `Scheduler::run_indexed`, each job running
/// `runner` (the exact trusted sequential entry point for whichever compiled path called this — never
/// a second implementation of "what an argument evaluates to"), then recompose by calling `runner`
/// **once more** on the reconstructed `prim(const_0, .., const_n)` node. Shared by both
/// [`compile_and_run_concurrent_with_swap_mode`] and [`jit_run_concurrent`] (DRY: one dispatcher body,
/// parameterized by which trusted runner evaluates/composes).
///
/// Any argument job's error is surfaced directly (never swallowed or reinterpreted) — unlike M-862's
/// interpreter-side batch there is no *shared mutable fuel counter* a failing sibling could corrupt
/// (each compiled/JIT job is fully independent), so there is nothing to discard-and-retry: the first
/// error encountered while collecting results is the answer, exactly as the sequential path would have
/// produced running the same failing argument.
fn concurrent_op_batch<R>(node: &Node, runner: R) -> Result<mycelium_core::Value, AotError>
where
    R: Fn(&Node) -> Result<mycelium_core::Value, AotError> + Send + Sync + 'static,
{
    let Node::Op { prim, args } = node else {
        // Unreachable: only ever called for a `ConcurrentPlan::OpBatch`, which is only produced for a
        // top-level `Node::Op`. Refuse explicitly rather than panic (never-silent, G2).
        return Err(AotError::Run(
            "concurrent_op_batch reached a non-Op node".to_owned(),
        ));
    };

    let runner = Arc::new(runner);
    let jobs: Vec<_> = args
        .iter()
        .map(|arg| {
            let arg = arg.clone();
            let runner = Arc::clone(&runner);
            move || runner(&arg)
        })
        .collect();
    let results: Vec<Result<mycelium_core::Value, AotError>> =
        Scheduler::new().run_indexed(jobs, None, None);

    let mut values = Vec::with_capacity(results.len());
    for r in results {
        values.push(r?);
    }

    let recomposed = Node::Op {
        prim: prim.clone(),
        args: values.into_iter().map(Node::Const).collect(),
    };
    runner(&recomposed)
}

/// The harness-level parallel entry point for the **direct-LLVM AOT** path (default
/// [`SwapCertMode::Recheck`]) — see the module docs for the dispatch/recomposition/scope contract.
/// Falls back wholesale to [`llvm::compile_and_run`] for anything outside the `Op`-headed batch
/// fragment ([`ConcurrentPlan::Sequential`]).
pub fn compile_and_run_concurrent(node: &Node) -> Result<mycelium_core::Value, AotError> {
    compile_and_run_concurrent_with_swap_mode(node, SwapCertMode::Recheck)
}

/// [`compile_and_run_concurrent`] under an **explicit** native swap cert mode (M-852 parameterization
/// carried through, mirroring [`llvm::compile_and_run_with_swap_mode`]).
pub fn compile_and_run_concurrent_with_swap_mode(
    node: &Node,
    swap_mode: SwapCertMode,
) -> Result<mycelium_core::Value, AotError> {
    match plan_concurrent(node) {
        ConcurrentPlan::Sequential => llvm::compile_and_run_with_swap_mode(node, swap_mode),
        ConcurrentPlan::OpBatch { .. } => concurrent_op_batch(node, move |n| {
            llvm::compile_and_run_with_swap_mode(n, swap_mode)
        }),
    }
}

/// The harness-level parallel entry point for the **in-process JIT** path — see the module docs for
/// the dispatch/recomposition/scope contract. Falls back wholesale to [`jit::jit_run`] for anything
/// outside the `Op`-headed batch fragment ([`ConcurrentPlan::Sequential`]); this costs the JIT nothing
/// it did not already lack, since `Construct`/`Match` are outside its compiled subset regardless
/// (M-727).
pub fn jit_run_concurrent(node: &Node) -> Result<mycelium_core::Value, AotError> {
    match plan_concurrent(node) {
        ConcurrentPlan::Sequential => jit::jit_run(node),
        ConcurrentPlan::OpBatch { .. } => concurrent_op_batch(node, jit::jit_run),
    }
}
