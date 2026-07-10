//! Re-export of the structured-concurrency supervision/cancellation surface (M-713 / RFC-0008
//! RT4·RT7), relocated to `mycelium-rt-abi` (M-883/M-884 runtime-ABI seam extraction).
//!
//! # Why a re-export
//!
//! `mycelium-mlir` (AOT tier; architectural tier `core`) consumes this surface directly
//! (`runtime.rs`'s `reclaim` driver — DN-58 §B, M-817), but a direct
//! `mycelium-mlir -> mycelium-std-runtime` normal dependency is an upward-tier edge (`core`
//! depending on `std`) — flagged by the `no-upward-tier-edges` gate (`xtask/deps-strata.toml`,
//! M-879). This module's only internal coupling was `crate::scheduler::Scheduler` — itself just a
//! thin re-export of `mycelium_sched::scheduler::Scheduler` — plus the M-356 composition kernel
//! re-exported from `mycelium-interp`; both are already below `mycelium-mlir`. So the whole surface
//! now lives in the foundational `mycelium-rt-abi` crate, which lets `mycelium-mlir` depend on it
//! directly without the upward edge.
//!
//! This module re-exports the full public surface at the **same path** existing consumers use
//! (`mycelium_std_runtime::supervision::*`) so every existing consumer (`colony`, and their tests)
//! compiles unchanged. See `mycelium-rt-abi`'s crate docs for the full relocation rationale and the
//! resulting dependency graph.
pub use mycelium_rt_abi::supervision::{
    run_supervised, supervise_with_restart, CancelToken, CancelTree, Cancelled, Escalation,
    RestartIntensity, SupervisedFailure, SupervisedRun, SupervisionAction, SupervisionRecord,
    Supervisor, TaskOutcome, SUPERVISION_PROPAGATION_STRENGTH, SUPERVISION_RESTART_BOUND_STRENGTH,
};
