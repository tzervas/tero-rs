//! `mycelium-proj` — the **project-metadata layer** (M-359; DN-06 §6; the
//! `docs/spec/Nodule-Header-and-Project-Manifest.md` schema, Accepted 2026-06-16).
//!
//! Three pieces, all *above* the kernel (KC-3 — nothing in the trusted base depends on this):
//!
//! - [`header`] — the **structured nodule header**: the `// @key: value` metadata lines (the closed
//!   v0 key set) that may follow the `// nodule:` marker (M-358). Unknown/duplicate keys and
//!   malformed values are explicit errors (G2/VR-5).
//! - [`manifest`] — the **`mycelium-proj.toml` manifest**, read by a deliberately **minimal,
//!   no-new-dependency TOML-subset** reader (the workspace keeps its deps few/vetted; **adding** a
//!   full TOML crate would be an ADR, not a build detail). It is honestly a subset, named as one.
//! - [`mod@resolve`] — **top-down inheritance** (`in-file > manifest`) with per-field provenance and an
//!   `EXPLAIN`, so a field's effective value and *source* are never ambient (G2).
//! - [`cert_scope`] — **`@certification` mode resolution & scoping** (RFC-0034 §6; M-790): the active
//!   [`CertMode`](mycelium_core::cert_mode::CertMode) resolved most-specific-wins over the
//!   `global > phylum > nodule` lattice (reusing RFC-0012's scoped-override mechanism), plus the
//!   explicit cross-mode-composition boundary ([`cert_scope::compose`]), plus the
//!   **generation≠consumption split** (RFC-0034 §7; M-792): the always-on [`ModeSignal`] /
//!   [`generate_mode_signal`] + tunable [`ConsumptionTier`] / [`render_mode_signal`] so the
//!   inspectability history is captured in every mode and consumption is dial-up.
//!
//! **Metadata is not identity (ADR-003).** Nothing here perturbs a definition's content hash — these
//! are associated, queryable fields, the human/release layer on top of content-addressing.

pub mod cert_scope;
pub mod header;
pub mod manifest;
pub mod resolve;

#[cfg(test)]
mod tests;

pub use cert_scope::{
    cert_mode_word, compose, explain_mode, generate_mode_signal, parse_cert_mode,
    render_mode_signal, resolve_mode, CertDecl, CertScope, ConsumptionTier, CrossModeEvent,
    ModeSignal, ResolvedMode,
};
pub use header::{
    parse_header, Deprecated, HeaderError, HeaderFields, StructuredHeader, HEADER_KEYS,
};
pub use manifest::{
    parse_manifest, Dependency, Manifest, ManifestError, Project, ProjectKind, SporeConfig,
    Surface, Toolchain,
};
pub use resolve::{explain, resolve, Origin, Resolved, ResolvedHeader};
