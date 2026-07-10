//! `mycelium-rt-abi` — the runtime-ABI seam (M-883/M-884; ADR-038 §2.2).
//!
//! # Why this crate exists
//!
//! `mycelium-mlir` (AOT tier; architectural tier `core`) needs two pieces of runtime machinery
//! that used to live in `mycelium-std-runtime` (tier `std`):
//! - the reclamation EXPLAIN/audit record ([`reclamation`], RFC-0027 §9), consumed by
//!   `mycelium-mlir::rc_plan` (MEM-4 → AOT bridge, DN-33 §2).
//! - the structured-concurrency supervision/cancellation surface ([`supervision`], M-713 /
//!   RFC-0008 RT4·RT7), consumed by `mycelium-mlir::runtime` (DN-58 §B, M-817).
//!
//! A direct `mycelium-mlir -> mycelium-std-runtime` normal dependency is an **upward architectural
//! edge** (`core` depending on `std`) — flagged by the `no-upward-tier-edges` gate
//! (`xtask/deps-strata.toml`, M-879). This crate is the fix: it holds exactly the ABI surface both
//! `mycelium-mlir` and `mycelium-std-runtime` need, sitting **below** both of them (it depends only
//! on `mycelium-core`, `mycelium-interp`, and `mycelium-sched` — all already `core`-tier, already
//! below `mycelium-mlir`). `mycelium-std-runtime` re-exports this crate's modules at their original
//! paths (`mycelium_std_runtime::reclamation::*` / `::supervision::*`) so every existing std-runtime
//! consumer keeps compiling unchanged (no public-API break).
//!
//! This is the same relocation shape M-861/M-862/M-864 used for the work-stealing `Scheduler`
//! (moved to `mycelium-sched`, below `mycelium-interp`) — see that crate's docs for the precedent.
//!
//! # `wild`-free
//!
//! This crate is `wild`-free: no raw pointer transmutes, no `unsafe` blocks (ADR-014).
#![forbid(unsafe_code)]

pub mod reclamation;
pub mod supervision;

#[cfg(test)]
mod tests;
