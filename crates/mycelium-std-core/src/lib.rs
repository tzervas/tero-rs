//! `std.core` — Ring-0 prelude: the honest value model, re-exported (M-515).
//!
//! The thin Ring-0 surface every other stdlib module imports to talk about values
//! honestly. It **re-exports** `mycelium-core`'s value model (RFC-0001) — `Value`,
//! `Repr`, `Meta`, the runtime sum `CoreValue`/`Datum`, the `GuaranteeStrength`
//! lattice (`Exact ⊐ Proven ⊐ Empirical ⊐ Declared`), the `Bound`/`BoundBasis`
//! companion, and the kernel's content-identity type — plus a thin §4.8 *query
//! surface* over them.
//!
//! Honesty crux (inherited, not invented): the kernel forbids a silent `Repr`
//! change and a spurious guarantee upgrade, and `std.core` is where that floor is
//! *named* for the whole library. It is Ring 0 and adds **no trusted code** (KC-3):
//! every re-export resolves to an `mycelium-core` (M-101) item, and the query
//! functions below are pure, total delegations to kernel accessors — never an
//! approximation or a selection of their own.
//!
//! Design spec: `docs/spec/stdlib/core.md`; contract: RFC-0016 §4.1 (C1–C6);
//! guarantee matrix: §4.5 (every row `Exact`/total — the honest floor for a
//! no-accuracy re-export surface).
//!
//! Scope note (boundary, spec §2): `std.core` exposes the *types* and the read-only
//! query surface; the *verbs* live elsewhere — representation change is `std.swap`
//! (M-516), ε/δ numeric helpers `std.numerics` (M-512), `Option`/`Result`
//! *combinators* `std.error` (M-527), content-addressing *as a library*
//! `std.content` (M-523).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 0):** Ring-0 re-exports make the representation machinery
//! available to the whole library; no ambient choice is imposed here — callers select
//! the representation explicitly. `repr_of`, `meta_of`, and `guarantee_of` expose the
//! reified representation of any value as a pure, total query (never an inferred default).
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/core.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference. A `.myc` port of the
//! [`GUARANTEE_MATRIX`] DATA now exists (`lib/std/core.myc`, M-927, kickoff `opp`) — but the
//! kernel re-export facade that is the bulk of this crate's surface (the `mycelium-core` type
//! re-exports, the `prelude`, the §4.8 query fns over `CoreValue`, and the `error_scaffold`)
//! remains Rust-only per RFC-0031 D1 (no `.myc`-surface FFI/kernel-type construction or
//! value-reflection mechanism exists yet). So the D6 retirement trigger has still NOT fired for
//! this crate as a whole (D6 retires a crate only once its full public surface has a `.myc`
//! counterpart, not just part of it — M-867 is post-1.0 regardless); no item here is
//! `#[deprecated]`.
#![forbid(unsafe_code)]

// ---- shared stdlib error scaffold (M-535, E5-1; DN-17 §2.4/§4 P3) ----------------
//
// The non-coupling, mechanical-only scaffold every `mycelium-std-*` error type can use:
// the `StdError` marker trait (+ blanket impl), the `impl_std_error!` boilerplate macro,
// and the `assert_is_std_error` / `assert_display_contains` test helpers. It factors out
// *only* the repeated `impl std::error::Error` marker / `source()` delegate / `*_is_std_error`
// test — never a `Display` message, a variant, a derive, or a guarantee tag (VR-5 / DN-17 §5).
pub mod error_scaffold;

// ---- value model re-exports (RFC-0001 §4.1–§4.3) ---------------------------------
pub use mycelium_core::bound::{Bound, BoundBasis, BoundKind, NormKind};
pub use mycelium_core::datum::{CoreValue, Datum};
pub use mycelium_core::guarantee::GuaranteeStrength;
pub use mycelium_core::id::ContentHash;
pub use mycelium_core::meta::{Meta, PackScheme, PhysicalLayout, Provenance, SparsityObs};
pub use mycelium_core::repr::{Repr, ScalarKind, SparsityClass};
pub use mycelium_core::value::{Payload, Trit, Value};

/// The curated default prelude (spec §3 / FLAG Q1). `use mycelium_std_core::prelude::*;`
/// brings the value model, the lattice tags, and the query surface into scope. The
/// final membership is a ratification call (RFC-0016 §8-Q3); this is the proposed
/// minimal set, kept consistent across the module specs by the orchestrator.
pub mod prelude {
    pub use super::{
        bound_of, guarantee_of, meta_of, provenance_of, repr_of, Bound, BoundBasis, BoundKind,
        CoreValue, Datum, GuaranteeStrength, Meta, NormKind, Payload, Provenance, Repr, Trit,
        Value,
    };
}

// ---- §4.8 runtime query surface (inspectability; spec §3/§4) ---------------------
//
// These are thin, pure, *total* delegations to the kernel's own accessors. They are
// honest by construction:
//   * `repr_of` / `meta_of` return `Option` — a `CoreValue::Data` (an algebraic
//     `Datum`) has no `Repr`/`Meta`, so the absence is reported explicitly (C1
//     never-silent), never a fabricated default.
//   * `guarantee_of` is total: every `CoreValue` carries a guarantee (a `Datum`'s is
//     the meet-summary of its fields).
//   * `bound_of` / `provenance_of` follow `meta_of` and so are `Option` too.
//
// None of these *select*, *convert*, or *approximate*, so each is `Exact` (C2) and
// has nothing of its own to EXPLAIN (C3) — they are the window through which a
// *downstream* op's tag/bound/provenance is inspected (RFC-0001 §4.8).

/// The representation of `v`, or `None` if `v` is algebraic data (no `Repr`).
#[must_use]
pub fn repr_of(v: &CoreValue) -> Option<&Repr> {
    v.as_repr().map(Value::repr)
}

/// The metadata of `v`, or `None` if `v` is algebraic data (no `Meta`).
#[must_use]
pub fn meta_of(v: &CoreValue) -> Option<&Meta> {
    v.as_repr().map(Value::meta)
}

/// The guarantee tag of `v` (total — every value carries one).
#[must_use]
pub fn guarantee_of(v: &CoreValue) -> GuaranteeStrength {
    v.guarantee()
}

/// The bound attached to `v`, or `None` when there is no metadata or no bound.
#[must_use]
pub fn bound_of(v: &CoreValue) -> Option<&Bound> {
    meta_of(v).and_then(Meta::bound)
}

/// The provenance of `v`, or `None` if `v` is algebraic data (no `Meta`).
#[must_use]
pub fn provenance_of(v: &CoreValue) -> Option<&Provenance> {
    meta_of(v).map(Meta::provenance)
}

// ---- guarantee matrix, as checked data (RFC-0016 §4.5) ---------------------------

/// One row of the module guarantee matrix (RFC-0016 §4.5): an exported item, its
/// honest guarantee tag, whether it is fallible (and the explicit error shape), its
/// declared effects, and whether it exposes an EXPLAIN artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// The exported op / item name.
    pub op: &'static str,
    /// Its honest guarantee tag on the lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.
    pub tag: GuaranteeStrength,
    /// The explicit fallibility: `"total"`, or the `Option`/`Result` shape returned.
    pub fallibility: &'static str,
    /// Declared effects (`"none"` for this pure re-export surface).
    pub effects: &'static str,
    /// Whether the item surfaces an inspectable EXPLAIN artifact.
    pub explainable: bool,
}

/// The `std.core` guarantee matrix (spec §4). Every row is `Exact` and effect-free:
/// `std.core` introduces no operation that selects, converts, or approximates, so the
/// honest tag for each is `Exact` (RFC-0016 §4.1 C2) — the honest *floor*, not an
/// upgrade. The query rows that surface a downstream value's own tag/bound/provenance
/// are marked EXPLAIN-able (they are the inspection window, RFC-0001 §4.8).
pub const GUARANTEE_MATRIX: &[GuaranteeRow] = &[
    GuaranteeRow {
        op: "Value/Repr/Meta (type re-exports)",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "CoreValue/Datum (type re-exports)",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "GuaranteeStrength (lattice tags)",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "Bound/BoundBasis (type re-exports)",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "repr_of",
        tag: GuaranteeStrength::Exact,
        fallibility: "Option<&Repr> (None for algebraic data)",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "meta_of",
        tag: GuaranteeStrength::Exact,
        fallibility: "Option<&Meta> (None for algebraic data)",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "guarantee_of",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "bound_of",
        tag: GuaranteeStrength::Exact,
        fallibility: "Option<&Bound> (None when no meta/bound)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "provenance_of",
        tag: GuaranteeStrength::Exact,
        fallibility: "Option<&Provenance> (None for algebraic data)",
        effects: "none",
        explainable: true,
    },
];

#[cfg(test)]
mod tests;
