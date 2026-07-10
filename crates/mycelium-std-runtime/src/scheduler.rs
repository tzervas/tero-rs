//! Re-export of the work-stealing OS-thread `Scheduler` (M-709 / M-861), relocated to
//! `mycelium-sched` (E25/M-862 dependency-cycle fix).
//!
//! # Why a re-export
//!
//! `mycelium-std-runtime` also depends on `mycelium-interp` (E12-1 / M-713: it reuses the M-356
//! supervision kernel — `CancelToken`/`TaskOutcome`/`Supervisor`). The Scheduler itself needs none
//! of that — only `mycelium_core::GuaranteeStrength` — so it now lives in the foundational
//! `mycelium-sched` crate (below `mycelium-interp`), which lets the interpreter depend on the
//! Scheduler too (`mycelium-interp -> mycelium-sched`) without a
//! `mycelium-interp -> mycelium-std-runtime -> mycelium-interp` cycle.
//!
//! This module re-exports the full public surface at the **same path** existing consumers use
//! (`mycelium_std_runtime::scheduler::Scheduler`, `StealPolicy`, `StealDecision`,
//! `SchedulerError`, and the `*_STRENGTH` guarantee constants) so the M-859 bench and any other
//! consumer compile unchanged. See `mycelium-sched`'s crate docs for the full relocation
//! rationale and the resulting dependency graph.
//!
//! **M-864:** the former `SCHEDULER_BACKPRESSURE_STRENGTH` re-export is dropped — the demand-signalled
//! backpressure bound it tagged was removed (it was the cause of a reproduced nested-submission
//! deadlock; the pool queue is now unbounded, memory bounded by batch size). See `mycelium-sched`'s
//! scheduler docs and DN-67.
pub use mycelium_sched::scheduler::{
    Scheduler, SchedulerError, StealDecision, StealPolicy, SCHEDULER_LIVENESS_STRENGTH,
    SCHEDULER_RT2_STRENGTH, STEAL_POLICY_STRENGTH,
};
