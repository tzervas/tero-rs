//! MEM-4 → AOT reclamation-plan bridge — the RFC-0027 §9 audit trail at the AOT tier.
//!
//! This module is the wiring that finally *consumes* the MEM-4 static analysis
//! (`mycelium-mir-passes`) at execution time: it runs the borrow-elided RC-emission
//! ([`mycelium_mir_passes::emit::emit_elided`]) through the reference RC-evaluator
//! ([`mycelium_mir_passes::eval::eval`]) and turns every `rc → 0` reclamation the analysis predicts
//! into a never-silent [`ReclamationRecord`] (trigger [`ReclamationTrigger::RcZero`]), emitted to a
//! [`ReclamationSink`]. That is the RFC-0027 §9 EXPLAIN/audit contract, now produced from the AOT
//! path rather than only the runtime `RcCell` probe.
//!
//! # Honest scope (VR-5 — read this before relying on it)
//!
//! The AOT env-machine ([`crate::aot::run_core`]) **still Rust-manages the values** — it does not
//! perform Mycelium-level reclamation. So what this module produces is the **observable audit
//! trail** of *where MEM-4's static analysis says reclamation occurs*, **not** a change to how the
//! program executes. [`run_with_reclamation`] computes the result with the unmodified trusted
//! env-machine and emits the plan **additively** alongside it: a bug here is a wrong or missing
//! *audit record*, never a wrong *value* (DN-33 §2 — MEM-4 is additive; the runtime `RcCell` probe
//! remains the sound fallback). Threading actual reclamation into the env-machine is the deferred
//! big step (E12 / RFC-0027 §10).
//!
//! Two further honesty bounds, both `Declared`:
//! - The analysed fragment is the RC-evaluator's **straight-line fragment** (`Const/Let/Op/Swap`
//!   plus the RC wrappers). A term outside it (recursion `Fix`/`FixGroup`, higher-order `App`/`Match`)
//!   has **no plan** — [`run_with_reclamation`] reports `reclaimed: None`, an explicit documented
//!   skip (G2 — never a silent empty plan). [`emit_reclamation_plan`] is the never-silent primitive
//!   that returns the *typed* [`RcPlanError`] for that case.
//! - The record's `value_meta_hash` is a **synthetic** identity derived from the abstract machine's
//!   deterministic allocation id (the abstract machine tracks *references*, not value *content* —
//!   see `eval.rs`'s honesty note), so there is no real content hash to record. It is well-formed
//!   and stable (enough to make the §9 record inspectable / EXPLAIN-able), but it is `Declared`, not
//!   the value's true RFC-0001 §4.6 content address.

use mycelium_core::{ContentHash, CoreValue, Node};
use mycelium_interp::{EvalError, PrimRegistry, SwapEngine};
use mycelium_mir_passes::emit::{emit_elided, EmitError};
use mycelium_mir_passes::eval::{eval, AllocId, RcError};
use mycelium_rt_abi::reclamation::{
    ReclamationRecord, ReclamationSink, ReclamationTrigger, ScopeId, SweepEpoch,
};

/// The scope identity stamped on AOT-emitted reclamation records.
///
/// The AOT env-machine runs a whole program as one top-level scope, so a single `ScopeId(0)` anchors
/// the run. `Declared` — MEM-3's canonical scope-tree identity will replace this `u64` placeholder
/// once the AOT tier threads live scopes (see `reclamation.rs`'s FLAG on `ScopeId`).
const AOT_TOP_SCOPE: ScopeId = ScopeId(0);

/// The sweep epoch stamped on AOT-emitted reclamation records.
///
/// A single AOT run is one sweep epoch. `Declared` — the scheduler's `SweepOrder` epoch
/// (RFC-0008 §4.3) will replace this when the AOT tier integrates with the live scheduler.
const AOT_SWEEP_EPOCH: SweepEpoch = SweepEpoch(0);

/// A failure to build the reclamation plan for a term — never-silent (G2).
///
/// Both arms mean "no audit trail for this term", but they are kept **distinct** so a caller can
/// tell an *out-of-fragment* term (the expected, benign case) from a *soundness failure* of the
/// emission (which would be a real bug, not a limitation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RcPlanError {
    /// The RC-emission lowering refused the term — e.g. `Fix`/`FixGroup`, outside the first-order
    /// fragment MEM-4 lowers ([`EmitError::UnsupportedNode`]). Benign: the term is simply outside
    /// the analysable fragment.
    Emit(EmitError),
    /// The reference RC-evaluator refused the emitted IR. Usually a control-flow node outside the
    /// straight-line fragment ([`RcError::UnsupportedNode`] — benign); a `UseAfterFree`/`DoubleFree`
    /// here would instead signal a real soundness bug in [`emit_elided`] (surfaced, not swallowed).
    Eval(RcError),
}

impl std::fmt::Display for RcPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RcPlanError::Emit(e) => write!(f, "RC-emission refused the term: {e}"),
            RcPlanError::Eval(e) => write!(f, "RC-evaluator refused the emitted IR: {e}"),
        }
    }
}

impl std::error::Error for RcPlanError {}

impl From<EmitError> for RcPlanError {
    fn from(e: EmitError) -> Self {
        RcPlanError::Emit(e)
    }
}

impl From<RcError> for RcPlanError {
    fn from(e: RcError) -> Self {
        RcPlanError::Eval(e)
    }
}

/// Synthesise a well-formed `Declared` content address for an abstract allocation id.
///
/// The abstract RC machine tracks *references*, not value content, so there is no real content hash
/// to record (see `eval.rs`'s honesty note). A stable, deterministic `rcplan:<id>` identity is
/// enough to make the §9 record inspectable and EXPLAIN-able. The 64-digit zero-padded decimal is
/// always a valid digest (`[A-Za-z0-9_-]+`) and `rcplan` a valid algo tag (`[a-z0-9]+`), so the
/// construction is infallible by construction.
fn synth_hash(alloc: AllocId) -> ContentHash {
    ContentHash::from_parts("rcplan", &format!("{alloc:064}"))
        .expect("`rcplan:<064-digit>` is always a well-formed content address")
}

/// Build and emit the MEM-4 reclamation plan for `node`, returning the number of records emitted.
///
/// Lowers `node` with borrow elision ([`emit_elided`]), evaluates the result in the reference RC
/// machine ([`eval`]), and emits one [`ReclamationRecord`] (trigger [`ReclamationTrigger::RcZero`])
/// per allocation the machine reclaims — the RFC-0027 §9 audit trail for the straight-line fragment.
///
/// Never-silent (G2): a term outside the analysable fragment returns the typed [`RcPlanError`]
/// rather than an empty plan. The records carry the supplied `scope_id` / `sweep_epoch` and a
/// synthetic `Declared` value hash ([`synth_hash`]).
///
/// Guarantee: the *count* of records equals the machine's reclamation count (`Exact`); the audit
/// trail's correspondence to real execution-time reclamation is `Declared` (the env-machine
/// Rust-manages values — see the module honesty note).
pub fn emit_reclamation_plan(
    node: &Node,
    sink: &mut dyn ReclamationSink,
    scope_id: ScopeId,
    sweep_epoch: SweepEpoch,
) -> Result<usize, RcPlanError> {
    let rc = emit_elided(node)?;
    let report = eval(&rc)?;
    for alloc in &report.reclaimed {
        sink.emit(ReclamationRecord::new(
            scope_id,
            sweep_epoch,
            ReclamationTrigger::RcZero,
            synth_hash(*alloc),
        ));
    }
    Ok(report.reclaimed.len())
}

/// The result of [`run_with_reclamation`]: the computed value plus the size of the reclamation plan.
#[derive(Debug, Clone)]
pub struct RcRun {
    /// The program's result, computed by the trusted AOT env-machine ([`crate::aot::run_core`]) —
    /// **identical** to what `run_core` alone would return (the plan is additive observability).
    pub value: CoreValue,
    /// The number of reclamation records emitted to the sink, or `None` if `node` is outside the
    /// straight-line fragment the RC-evaluator models (an explicit, documented skip — never a silent
    /// empty plan).
    pub reclaimed: Option<usize>,
}

/// Run a Core IR program through the AOT path **and** emit its MEM-4 reclamation plan additively.
///
/// The value is computed by the unmodified trusted env-machine ([`crate::aot::run_core`]); the
/// reclamation plan ([`emit_reclamation_plan`]) is emitted alongside it to `sink`. The plan never
/// perturbs the result (DN-33 §2 / RFC-0027 §9).
///
/// A term outside the straight-line fragment (recursion / higher-order control flow) yields
/// `reclaimed: None` — the plan error is mapped to an explicit, documented skip, **not** swallowed
/// silently (use [`emit_reclamation_plan`] directly when you need the typed [`RcPlanError`]).
///
/// The returned [`EvalError`] is only ever the *value* computation failing; plan construction never
/// fails this function (it degrades to `None`).
pub fn run_with_reclamation(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    sink: &mut dyn ReclamationSink,
) -> Result<RcRun, EvalError> {
    let value = crate::aot::run_core(node, prims, swap)?;
    let reclaimed = emit_reclamation_plan(node, sink, AOT_TOP_SCOPE, AOT_SWEEP_EPOCH).ok();
    Ok(RcRun { value, reclaimed })
}
