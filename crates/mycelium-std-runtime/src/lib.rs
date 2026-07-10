//! `std.runtime` — the fungal concurrency surface (M-521 / ADR-020).
//!
//! Implements the v0 R1 API surface decided in ADR-020 (Accepted 2026-06-20):
//! [`colony::Colony`]/[`colony::Scope`] structured concurrency,
//! [`task::Task`]/[`task::TaskCtx`]/[`task::Poll`],
//! sweep ordering ([`task::SweepOrder`], [`task::Deadlock`]), and the channel surface
//! ([`network::Network`], [`network::Sender`], [`network::Receiver`], [`network::TrySend`], [`network::TryRecv`]).
//!
//! # Guarantee matrix
//!
//! Every exported operation has a row in [`guarantee_matrix::MATRIX`].
//! The matrix is tested, not prose-only (SC-2 / VR-5).
//!
//! # Execution maturity (E12-1: M-709 / M-711 / M-713)
//!
//! Beyond the cooperative v0 surface, the crate now executes on real OS threads:
//! - [`scheduler::Scheduler`] (M-709 / M-861) — a per-worker-deque work-stealing OS-thread pool
//!   with demand-signalled, bounded backpressure (RFC-0008 RT1·RT2·RT3·§4.3); the RT2
//!   sequentialization differential is property-tested (`Empirical`). **Relocated to
//!   `mycelium-sched`** (E25/M-862): a foundational crate below `mycelium-interp`, so the
//!   interpreter can use it too, without an `interp <-> std-runtime` cycle. [`scheduler`] here is
//!   a thin re-export — see its module docs and `mycelium-sched`'s crate docs for the
//!   dependency-graph rationale.
//! - [`dataflow::run_dataflow`] / [`dataflow::run_dataflow_scheduled`] (M-711) — deadlock-freedom
//!   for communicating tasks: a no-progress sweep is an explicit [`task::Deadlock`], never a silent
//!   hang (G2 / RFC-0008 §4.3), checked on both the cooperative path and the OS-thread pool.
//! - [`supervision`] (M-713) — structured-concurrency cancellation ([`supervision::CancelTree`]),
//!   explicit per-child outcome collection ([`supervision::run_supervised`]), and an EXPLAIN-able
//!   bounded-cascade restart policy ([`supervision::supervise_with_restart`]) (RFC-0008 RT4·RT7),
//!   reusing the M-356 composition kernel from `mycelium-interp`. **Relocated to `mycelium-rt-abi`**
//!   (M-883/M-884): a foundational crate below `mycelium-mlir`, so the AOT tier can use it too,
//!   without an upward `mycelium-mlir -> mycelium-std-runtime` edge. [`supervision`] here is a
//!   thin re-export — see its module docs and `mycelium-rt-abi`'s crate docs for the
//!   dependency-graph rationale.
//!
//! # Reserved vocabulary (Glossary ⟂)
//!
//! The RFC-0008 §4.5 surface **constructs** (`hypha`, `fuse`, `xloc`, `cyst`, `graft`,
//! `forage`, `backbone`, `mesh`, `tier`, `reclaim`) remain **reserved, not yet activated** as
//! L1 language surface (their elaboration is M-710, gated on the M-667 L1 surface). The runtime
//! *machinery* they will dispatch to (scheduler, deadlock detection, supervision) is what this
//! crate now provides.
//!
//! # `wild`-free
//!
//! This crate is `wild`-free: no raw pointer transmutes, no `unsafe`
//! blocks, no `wild` keyword constructs (ADR-014).
//!
//! Design: `docs/adr/ADR-020-Runtime-Colony-Phylum-Placement.md`;
//! spec: `docs/spec/stdlib/runtime.md`; tasks M-521, E12-1 (M-709/M-711/M-713).
//!
//! # Memory model (E12 MEM-1/MEM-2/MEM-3 + live wiring)
//!
//! - [`reclamation`] (MEM-1) — the reclamation EXPLAIN/audit record and never-silent sink
//!   contract (RFC-0027 §9). **Relocated to `mycelium-rt-abi`** (M-883/M-884), for the same
//!   upward-tier-edge reason as [`supervision`] above; [`reclamation`] here is a thin re-export.
//! - [`rc`] (MEM-2) — non-atomic intra-hypha RC cell + `rc`-probe decision (DN-32 §2.2).
//! - [`region`] (MEM-3) — region-based batched scope-exit reclamation (DN-32 §2.3 / RFC-0027
//!   §10.3): [`region::Region`] accumulates deferred entries and bulk-emits `ScopeExit` records
//!   at scope-exit; [`region::ScopeNodeId`] / [`region::RegionEpoch`] are the canonical forms
//!   of the MEM-1 `ScopeId`/`SweepEpoch` placeholder types.
//! - [`scope_region`] — the **live-executor wiring**: structured `with_region` /
//!   [`scope_region::RegionScope`] tie a [`region::Region`]'s lifecycle to a single-hypha
//!   structured-concurrency scope, closing it (emitting the batched `ScopeExit` records) at
//!   scope-exit — reclamation fires from the running executor, not just the data structure.
//!   Nested scopes give child-before-parent epoch order for free.
//! - [`network`] also carries the **third live trigger**, `ChannelClose`: closing a channel that
//!   still holds buffered values in transit reclaims them, emitting one
//!   `ReclamationRecord(ChannelClose)` per value (RFC-0027 §9 / §7.3), with a canonical
//!   [`network::ChannelNodeId`] resolving the MEM-1 `ChannelId` placeholder.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/runtime.md` (spec status:
//! Accepted, v0 R1 surface (2026-06-21)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! Unlike the other 25 `mycelium-std-*` crates, this one was **load-bearing** (consumed directly by
//! `mycelium-mlir`), not reference-only, per [DN-66 §4.c](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//!
//! **FLAG (M-883/M-884, this crate's own change):** the specific basis for DN-66 §4.c —
//! `mycelium-mlir` depending on this crate *directly* — no longer holds after the runtime-ABI seam
//! extraction: `mycelium-mlir` now consumes the relocated `reclamation`/`supervision` surfaces via
//! `mycelium-rt-abi` (below `mycelium-mlir` in the dependency graph) instead of via this crate, and
//! `mycelium-std-runtime` is removed from its `Cargo.toml` entirely. Whether this crate remains
//! load-bearing (e.g. via `mycelium-bench`, or other v0 R1 surface usage) or reverts to an ordinary
//! retirement candidate under RFC-0031 D6 is an **orchestrator/maintainer-level re-review of
//! DN-66 §4.c**, not decided here — this note only records that its stated factual basis changed.
//!
//! # R2 tail — mechanized policy capture/set + the residual ledger (M-963 / DN-78)
//!
//! The M-828 capture-and-set tail, decided buildable by DN-78 §3 under the 2026-07-02
//! delegation (frz Lane C, epic E31-1):
//!
//! - [`policy_mech`] — mechanized `SelectionPolicy` **capture** ([`policy_mech::capture`] /
//!   [`policy_mech::replay`] — record-vs-replay, never a silent pass) and **setting**
//!   ([`policy_mech::PolicySlot`] — a reified setter whose every transition is recorded),
//!   riding the existing RFC-0005 machinery in `mycelium-select` (no new mechanism, KC-3).
//! - [`r2_residual`] — the never-silent ledger + refusal surface for everything the DN-78 §4
//!   split defers (the four remaining §4.5 construct activations, the capture/set L1 surface,
//!   multi-node placement maturity): one tested row per deferral, an explicit typed refusal
//!   per item (G2).
//!
//! These modules are **additive** to the DN-66 frozen v0 R1 baseline (a spec amendment for
//! `docs/spec/stdlib/runtime.md` is FLAGged to the integrator — DN-78 FLAG-F2, not a silent
//! extension).
#![forbid(unsafe_code)]

pub mod colony;
pub mod dataflow;
pub mod guarantee_matrix;
pub mod network;
pub mod policy_mech;
pub mod r2_residual;
pub mod rc;
pub mod reclamation;
pub mod region;
pub mod scheduler;
pub mod scope_region;
pub mod supervision;
pub mod task;

#[cfg(test)]
mod tests;
