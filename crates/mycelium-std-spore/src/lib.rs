//! `std.spore` — the deployable / reconstruction-manifest library (M-522, issue #163).
//!
//! `std.spore` is the ergonomic, value-semantic library face of the ADR-013 content-addressed
//! deployable unit and the RFC-0003 §6 reconstruction manifest. It **consumes** the landed
//! `mycelium-spore` packager (M-368), `std.content`'s hash primitives, and `std.vsa`'s decode;
//! it mints no new hash and performs no reconstruction itself (KC-3).
//!
//! # Honesty crux
//!
//! - **C4 / ADR-003:** a spore's identity *is* its canonical content hash; metadata is **not**
//!   identity, so the build hash is deterministic and metadata-invariant.
//! - **FR-C2 / VR-5 ceiling:** probabilistic VSA regrowth is tagged **`Empirical` at most**,
//!   carries its δ, and is **never** `Proven`. Enforced structurally: any manifest whose decode
//!   is `Resonator` and whose bound basis exceeds `Empirical` returns
//!   `Err(MalformedManifest::ResonatorOverStrength)` from [`ReconManifest::validate`].
//! - **C1 / G2:** a hash mismatch or missing/ambiguous publish input is an **explicit `Err`
//!   naming the offending input** — never a silent accept and never a partial artifact.
//!
//! # Guarantee matrix
//!
//! Every exported op has a row in [`guarantee_matrix::MATRIX`] (RFC-0016 §4.5 / spec §4).
//! The matrix is asserted in tests, not prose-only (C2 / VR-5).
//!
//! # Out of scope (FLAGS carried from spec §7)
//!
//! - **Native deploy / germination** — Implemented in M-620 (see [`deploy`] module):
//!   [`deploy::germinate`], [`deploy::explain_deploy`], [`deploy::DeployTarget`],
//!   [`deploy::DeployResult`], [`deploy::DeployError`], [`deploy::DeployVerification`].
//!   FLAG Q2 is resolved; the seam is live and guarded end-to-end (G2 / VR-4).
//! - **Approx coupling** (FLAG Q4a — RESOLVED) — [`RegrowthResult`] carries the manifest's full
//!   certificate `Bound` and projects to `std.numerics::Approx<Factorization>` via
//!   [`recon_manifest::RegrowthResult::as_approx`] (strength derived from the bound's basis, never
//!   upgraded — VR-5; held at the `Empirical` ceiling, FR-C2). It carries `Factorization` rather
//!   than `Value` because the resonator decode yields VSA factor atoms, not a reconstructed
//!   `Value` (that mapping is `std.vsa`'s). `std.spore` now depends on `mycelium-std-numerics`.
//! - **Ring placement discrepancy** (FLAG Q1) — RFC-0016 §4.2 lists `spore` under Ring 2 while
//!   §4.3 / the stdlib index files it under Tier A. This spec follows Tier-A / Ring-1 —
//!   FLAGGED §7 Q1.
//! - **Packaging schema fields** (FLAG Q3) — the `mycelium-proj.toml` field set is
//!   M-368/M-359's to define; this crate consumes whatever they fix — FLAGGED §7 Q3.
//! - **Spore ↔ vsa seam** (FLAG Q4b) — `std.vsa` owns the regrowth op; `spore` only packages
//!   and validates the manifest — co-design pending — FLAGGED §7 Q4.
//!
//! Design spec: `docs/spec/stdlib/spore.md`; ADR-013; RFC-0003 §6; task M-522, issue #163.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A):** Spore deployment records the representation manifest in
//! the content-addressed certificate: the packed representation of each component is part of the
//! canonical hash (ADR-003/RFC-0001 §4.6), so a packing change produces a different spore
//! identity. There is never a silent packing change at deploy time — the identity itself proves
//! the packing.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/spore.md` (spec status:
//! Accepted, library/manifest half (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference. A `.myc` port now exists
//! (`lib/std/spore.myc`, M-934 — kickoff `opp`, RFC-0031 D5), with this crate as its live
//! Rust oracle (`crates/mycelium-l1/tests/std_spore.rs`, including the M-934 content-address
//! parity check — the port carries this crate's hashes verbatim, never minting its own); per D6
//! the crate is **retained**, not retired (retirement is the post-1.0 M-867 decision), and no
//! item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod deploy;
pub mod guarantee_matrix;
pub mod recon_manifest;
pub mod spore_ops;

// Re-export the key consumer surface.
pub use mycelium_core::{ContentHash, GuaranteeStrength};
pub use mycelium_spore::{Spore as RawSpore, SporeError};

pub use deploy::{
    explain_deploy, germinate, DeployError, DeployResult, DeployTarget, DeployVerification,
};
pub use guarantee_matrix::MATRIX;
pub use recon_manifest::{MalformedManifest, ReconManifest, ReconMode, RegrowthResult};
pub use spore_ops::{explain_spore, identity, manifest_of, verify, SporeErr, SporeUnit};
