//! The swap extension point for the interpreter (M-110).
//!
//! A [`Node::Swap`](mycelium_core::Node::Swap) is the *only* representation-changing node
//! (RFC-0001 §4.5 WF1); evaluating one is delegated to a [`SwapEngine`]. The real, certificate-
//! emitting binary↔ternary swap is **M-120** (`docs/spec/swaps/binary-ternary.md`); this module
//! defines the interface plus the trivial [`IdentitySwapEngine`] (a same-`Repr` swap, which is
//! exactly lossless) so the interpreter can evaluate swap nodes today. An unsupported swap is an
//! explicit error — **never** a silent coercion (SC-3; G2).

use mycelium_core::{ContentHash, Meta, Provenance, Repr, Value};

use crate::EvalError;

/// Evaluates a `Swap` node. Implementations must be *never silent*: an out-of-domain or unsupported
/// conversion returns an [`EvalError`] (the interpreter surfaces it), it does not coerce.
///
/// **`Sync` (M-862) + `Send` (M-864):** the parallel-pure-fragment evaluator (`crate::parallel`)
/// shares the engine across the M-861 scheduler's worker threads (`mycelium_sched`) via `&Interpreter`
/// — every engine must be safely readable from multiple threads at once (`Sync`). M-864's persistent
/// pool tightened `run_indexed` to `'static` job closures, which can no longer *borrow*
/// `&Interpreter` from the caller's stack frame (the pool's worker threads outlive any single call);
/// `Interpreter` is now `Clone` and stores its engine as `Arc<dyn SwapEngine>` so a job can own a
/// cheap handle to it, which additionally requires `Send` (an `Arc<T>` is `Send` only if `T: Send +
/// Sync`). Every shipped/known implementation (the identity engine here, the M-120 certified engines
/// in `mycelium-cert`) is a plain, interior-mutability-free struct and satisfies both automatically;
/// a future engine that needs interior mutability would need its own synchronization (e.g. a
/// `Mutex`), same as any other `Send + Sync` type.
pub trait SwapEngine: Send + Sync {
    /// Convert `src` to `target` under `policy`, returning the converted [`Value`] or an error. The
    /// result's [`Meta`] must record `policy_used` (ADR-006) and an honest guarantee/bound.
    fn swap(&self, src: &Value, target: &Repr, policy: &ContentHash) -> Result<Value, EvalError>;
}

/// The trivial swap engine: a swap whose `target` equals the source `Repr` is the identity — exactly
/// lossless, `guarantee` preserved. Any *cross-paradigm* swap returns
/// [`EvalError::UnsupportedSwap`], deferring to the certified M-120 engine. This keeps M-110 honest:
/// it executes the swaps it can prove trivially exact and refuses the rest rather than guessing.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentitySwapEngine;

impl SwapEngine for IdentitySwapEngine {
    fn swap(&self, src: &Value, target: &Repr, policy: &ContentHash) -> Result<Value, EvalError> {
        if src.repr() != target {
            return Err(EvalError::UnsupportedSwap {
                from: src.repr().clone(),
                to: target.clone(),
            });
        }
        // Same representation → identity. The value is unchanged; metadata records that it was
        // produced by a swap (policy_used set, ADR-006) and keeps the source's guarantee/bound.
        //
        // The `EvalError::Wf` arms below are *defensive*, not reachable from the public API via
        // this engine (A4-04): the guarantee/bound are copied verbatim from the already-validated
        // `src.meta()` (so the M-I1 coupling still holds, and `policy_used` is independent of it),
        // and the repr/payload are `src`'s own (so payload↔repr still agrees). They remain explicit
        // errors so a *custom* `SwapEngine` reusing this pattern — or a future change that derives
        // rather than copies the meta — refuses honestly rather than panicking (G2).
        let src_meta = src.meta();
        let meta = Meta::new(
            Provenance::Derived {
                op: mycelium_core::operation_hash("swap.identity"),
                inputs: vec![src.content_hash()],
            },
            src_meta.guarantee(),
            src_meta.bound().cloned(),
            src_meta.sparsity(),
            src_meta.physical(),
            Some(policy.clone()),
        )
        .map_err(EvalError::Wf)?;
        Value::new(src.repr().clone(), src.payload().clone(), meta).map_err(EvalError::Wf)
    }
}
