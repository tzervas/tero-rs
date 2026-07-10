//! `mycelium-l1` — the **L1 surface prototype** (RFC-0006; **NON-NORMATIVE** until RFC-0006 is
//! ratified). A hand-written lexer + recursive-descent parser for the ratified DN-02 surface
//! vocabulary, validated against the `docs/spec/grammar/` conformance corpus (the
//! WebAssembly-spec pattern, T3.1-B: the corpus is the ground truth, not any single parser).
//!
//! The L1 track so far (RFC-0006 §3; RFC-0007): the lexer + recursive-descent parser prove the
//! grammar is real by parsing every `accept/` program and explicitly rejecting every `reject/`
//! one (`tests/conformance.rs`); the v0 monomorphic typechecker + structural totality checker
//! ([`checkty`], [`totality`]; RFC-0007 §4.4/§4.5) gate `matured` on checked totality; the
//! fuel-guarded big-step evaluator ([`eval`]; §4.6) runs every checked program over the *same*
//! trusted prim/swap engines as the L0 paths; and the elaborator ([`elab`]; §4.6) lowers the
//! evaluation-complete fragment to closed L0 terms — refusing everything else with an explicit
//! `Residual`, never a partial artifact. The three-way differential (L1-eval ↔ elaborate→L0-interp
//! ↔ AOT, validated through the M-210 shared checker) lives in `tests/differential.rs` (NFR-7).
//! `match` covers data types and `Binary`/`Ternary` literal patterns *and* **nested** patterns
//! (M-320): a literal arm fires on `repr + payload` equality, and coverage is decided by the
//! **Maranget usefulness** algorithm (`usefulness`) — exhaustiveness (a `_` must not be useful; its
//! witness names a concrete missing case) and redundancy (an arm covered by earlier rows is
//! unreachable) are both *checked* (W7 — never assumed; a `Binary`/`Ternary` value domain is never
//! enumerated, so a literal match still needs a `_`/binder default). The Maranget *compilation* to the
//! flat kernel `Match` (RFC-0007 §3, the elaborator/L0 path) lands with full L1-in-Core-IR (the
//! RFC-0001 revision).
//!
//! Honesty: every malformed input is an explicit [`ParseError`] with a source position — the
//! parser never panics and never silently accepts (S5/G2). The lexer disambiguates the one tricky
//! token (`<` opening a ternary literal vs a type-argument list) by lookahead, and a malformed
//! ternary literal is an explicit error, not a silent truncation.
//!
//! **Trusted-kernel discipline (ADR-014, KC-3):** this crate is `#![forbid(unsafe_code)]` — the
//! reference interpreter is **machine-proven `unsafe`-free**. Host-stack management for the recursive
//! checker/elaborator (the deep worker stack) is deliberately kept *outside* this kernel, in the
//! `mycelium-stack` crate, which the kernel uses only through its safe API; the explicit depth budgets
//! (`parse::MAX_EXPR_DEPTH`, `checkty::MAX_CHECK_DEPTH`, the evaluator's clock) are the portable
//! primitive that carries to the self-hosted frontend.
#![forbid(unsafe_code)]

/// The static affine **use-once** tracker for `Substrate` bindings (M-903; DN-71 Model S §4.2) —
/// piggybacked on [`checkty::Cx`]'s own scope, not a parallel analysis (KC-3/DRY). Internal to the
/// frontend; not part of the public surface.
mod affine;
pub mod ambient;
pub mod ast;
pub mod checkty;
pub(crate) mod decision;
pub mod elab;
pub mod error;
pub mod eval;
/// DN-58 §A (M-965) — the `Fuse` prelude trait (F-A1) + the semilattice-law checker (F-A2), the
/// [`checkty`] post-pass driven by [`checkty::check_nodule`]/[`checkty::check_phylum`]. Internal
/// to the frontend, like [`grade`]; not part of the public surface.
pub(crate) mod fuse;
/// RFC-0018 stage-1a static guarantee grading (Design A) — the [`checkty`] post-pass that enforces
/// the guarantee lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` statically. Internal to the
/// frontend (driven by [`checkty::check_nodule`]); not part of the public surface.
mod grade;
pub mod lexer;
pub mod mono;
pub mod nodule;
pub mod parse;
/// The `Substrate` v0 value form (M-902; DN-71 Model S §4.1) — an interpreter-level opaque affine
/// handle at the L1 evaluator level. No new L0 node / no `Repr` growth (KC-3). The affine use-once
/// **runtime backstop** now lives here too (M-903 — [`substrate::SubstrateHandle::try_consume`]);
/// the primary enforcement is the static pass ([`affine`]) run by [`checkty::check_nodule`]. The
/// `consume` **lowering** (real execution through existing paths) is still M-904.
pub mod substrate;
pub mod token;
pub mod totality;
pub(crate) mod usefulness;

#[cfg(test)]
mod tests;

pub use ambient::{
    expand_phylum_to_source, expand_to_source, resolve, resolve_report, AmbientError, Resolved,
};
pub use ast::{Nodule, Phylum, UsePath, Vis};
pub use checkty::{
    check_and_resolve, check_nodule, check_nodule_matured, check_phylum, check_phylum_matured,
    CheckError, Env, PhylumEnv, Ty,
};
pub use elab::{
    elaborate, elaborate_colony, elaborate_direct, elaborate_lower_rule, elaborate_reclaim,
    ElabError,
};
pub use error::ParseError;
pub use eval::{Evaluator, ForageDecision, ForageError, L1Error, L1Value};
pub use mono::{
    monomorphize, monomorphize_with_selections, ClosureSpecialization, InstanceSelection,
    MonoSelections,
};
pub use nodule::{parse_nodule_header, NoduleHeader, NoduleHeaderError};
pub use parse::{parse, parse_phylum};
pub use substrate::{ReleaseEvent, SubstrateError, SubstrateHandle, SubstrateProvenance};
pub use totality::Totality;
