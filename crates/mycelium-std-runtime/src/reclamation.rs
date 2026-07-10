//! Re-export of the reclamation EXPLAIN/audit record (RFC-0027 §9 / MEM-1), relocated to
//! `mycelium-rt-abi` (M-883/M-884 runtime-ABI seam extraction).
//!
//! # Why a re-export
//!
//! `mycelium-mlir` (AOT tier; architectural tier `core`) consumes this record type
//! (`rc_plan.rs`), but a direct `mycelium-mlir -> mycelium-std-runtime` normal dependency is an
//! upward-tier edge (`core` depending on `std`) — flagged by the `no-upward-tier-edges` gate
//! (`xtask/deps-strata.toml`, M-879). `ReclamationRecord`/`ReclamationSink`/`ReclamationTrigger`/
//! `ScopeId`/`SweepEpoch`/`ChannelId`/`ExplainRecord`/`CollectingSink` need only
//! `mycelium_core::ContentHash` (no other std-runtime coupling), so they now live in the
//! foundational `mycelium-rt-abi` crate (below `mycelium-mlir`), which lets `mycelium-mlir` depend
//! on them directly without the upward edge.
//!
//! This module re-exports the full public surface at the **same path** existing consumers use
//! (`mycelium_std_runtime::reclamation::*`) so every existing consumer (`network`, `rc`, `region`,
//! `scope_region`, and their tests) compiles unchanged. See `mycelium-rt-abi`'s crate docs for the
//! full relocation rationale and the resulting dependency graph.
pub use mycelium_rt_abi::reclamation::{
    ChannelId, CollectingSink, ExplainRecord, ReclamationRecord, ReclamationSink,
    ReclamationTrigger, ScopeId, SweepEpoch,
};
