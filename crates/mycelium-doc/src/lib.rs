//! `mycelium-doc` — **`myc-doc`**, the M-363 documentation BUILD pipeline (spec
//! `docs/spec/Narrative-Authoring-Pipeline.md`, Accepted).
//!
//! The architecture is **one content-addressed doc-IR, many renderers** (§3): the cited corpus
//! (RFCs/ADRs/notes/specs), the code + M-359 nodule-header metadata, and the JSON schemas are all
//! *projected* into one navigable [`ir::DocModel`], and HTML, Typst (→ PDF) and machine JSON are
//! *views of that one model* — never parallel truths (ADR-003/G11). Generation is **projection, not
//! authorship**: an item that cannot be grounded is **flagged "undocumented," never invented** (the
//! prose analogue of G2).
//!
//! On top of the model runs the **§4.1 quality-bar lint** ([`doc_lint`], spec §6): eight explicit
//! pass/fail checks (single-template conformance · navigability · progressive disclosure · checked
//! examples · no-dead-xref · dual-projection parity · no-hallucinated-prose · legibility/
//! accessibility). This is what activates the dormant lint named in `mycelium_lint`.
//!
//! KC-3: this is a tool **above the kernel** (no kernel change); it reuses the kernel's BLAKE3
//! `ContentHash` shape and the trusted L1 checker, and adds **no new workspace dependency**.

pub mod apiref;
pub mod book;
pub mod build;
pub mod corpus;
pub mod doc_lint;
pub mod emit;
pub mod hash;
pub mod ir;
pub mod lib_index;

#[cfg(test)]
mod tests;

pub use book::{build_book, load_manifest, BookError, BookManifest, ChapterSpec};
pub use build::{build, emit_all, BuildInput};
pub use doc_lint::{
    lint, CheckOutcome, CheckStatus, DocLintReport, Finding, Severity, CHECK_NAMES,
};
pub use ir::{DocModel, Level, Node, Payload, Provenance, SourceKind};
