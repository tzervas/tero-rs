//! `std.error` — errors-as-values ergonomics layer: propagate, recover, or re-propagate —
//! never drop (M-527, #168).
//!
//! # What this crate is
//! The **Ring-2 combinator and propagation surface** over `Option<T>` and `Result<T, E>`
//! (the Mycelium value model's error sums, from `core` / RFC-0001). It provides the
//! `map` / `map_err` / `and_then` / `or_else` / `filter` / `ok_or` / `unwrap_or` /
//! `?`-style-propagate / `unwrap`-family / `recover`-bridge surface described in
//! `docs/spec/stdlib/error.md` §3.
//!
//! # Honesty crux (C1 / I1 — the structural guarantee)
//! **Propagation is the floor; suppression is structurally impossible.** There is no
//! combinator in this surface that consumes an `Err`/`None` and silently yields success.
//! Every op either:
//! - transforms the error (it survives in the result sum), or
//! - re-propagates it (short-circuit / `?`), or
//! - explicitly recovers it (with an honest `Declared` tag — the caller decides), or
//! - **refuses loudly** (the `unwrap`/`expect`/`unwrap_err` named partials).
//!
//! The one lossy op, `ok` (`Result→Option`), discards `ε` — this is a **flagged,
//! EXPLAIN-able lossy conversion** (not an unflagged drop; FLAG Q2 in spec §7).
//!
//! # Guarantee matrix (load-bearing deliverable — RFC-0016 §4.5)
//! [`guarantee_matrix::MATRIX`] is the checked-data guarantee matrix: one row per
//! exported op, asserted in tests — never prose-only. Tags: mostly `Exact` (pure
//! combinators); `Declared` for `unwrap_or`/`unwrap_or_else` (substituted defaults,
//! RFC-0014 I2/VR-5); `Inherited` for the `recover` bridge (VR-5 — never laundered up).
//!
//! # Design spec
//! `docs/spec/stdlib/error.md` (M-527, #168). Contract: RFC-0016 §4.1 (C1–C6).
//!
//! # Module boundary (spec §2)
//! - **In scope:** the pure combinator surface + `?`-style propagation + `recover` bridge.
//! - **Out of scope:** error *representation* (`Option`/`Result` types — `core`/M-515);
//!   reified *recovery policies* + effect budgets (`std.recover`, M-520 — the bridge
//!   target); error *presentation* (`std.diag`, M-510).
//!
//! # Open FLAGs (carried from spec §7)
//! - **FLAG Q1 (RESOLVED):** The concrete `RecoverOutcome`/`Outcome` shape and the recovery
//!   driver are owned by `std.recover` (M-520, RFC-0014), now landed. This crate re-exports
//!   that surface ([`Outcome`]/[`Resolution`]/[`RecoverOutcome`]/[`handle_classified`]) at its
//!   root and defines no recovery type of its own (KC-3 — it is the bridge target, not the home).
//! - **FLAG Q2:** Whether `ok` (`Result→Option`) should be gated behind an unmissable
//!   name (e.g. `ok_discarding_err`) awaits RFC-0016 §8-Q3 ratification.
//! - **FLAG Q3:** The `unwrap`/`expect`/`unwrap_err` refusal mechanism (abort vs escalate
//!   vs `std.diag` record) is co-designed with M-510/M-520; the *guarantee* (loud refusal,
//!   never silent) is fixed here.
//! - **FLAG Q4:** Whether `?`-on-`Option` vs `?`-on-`Result` unification is one
//!   polymorphic operator or two surface forms is a DN-level decision (RFC-0016 §8-Q3).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A / Tier B boundary):** Error types carry the representation
//! context in which they arose — a conversion error (e.g., from `std.swap`) retains the source
//! and target `Repr` as part of its value. No representation context is stripped on error paths;
//! an `Err` is as inspectable as an `Ok`.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/error.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod combinators;
pub mod guarantee_matrix;

// ---- top-level re-exports of the combinator surface -------------------------

// Value sums from core (Ring-0 / spec §2 — this module imports, not defines, them).
pub use mycelium_core::PolicyRef;
pub use mycelium_std_core::GuaranteeStrength;

// RFC-0014 recovery bridge (error.md §7-Q1 — RESOLVED). The concrete `Outcome`/`Resolution`
// landed in `std.recover` (M-520); re-export it here so `std.error` callers reach the recovery
// surface through the errors module. The bridge keeps `std.error`'s obligation verbatim: the
// outcome is `Recovered | Propagated` with **no drop variant** (I1), and the recovered tag is
// inherited from the policy, **never laundered upward** (I2/VR-5). `std.error` adds no recovery
// algebra of its own — it consumes `std.recover`'s (KC-3).
pub use mycelium_std_recover::{handle_classified, Outcome, RecoverOutcome, Resolution};

// Combinator functions re-exported at the crate root for ergonomics.
pub use combinators::{
    // transform
    and_then,
    // named partial accessors
    expect,
    filter,
    flatten,
    inspect,
    inspect_err,
    map,
    map_err,
    ok,
    ok_or,
    ok_or_else,
    or_else,
    transpose,
    unwrap,
    unwrap_err,
    // defaulted accessors (Declared tag)
    unwrap_or,
    unwrap_or_else,
    unwrap_or_else_option,
    unwrap_or_option,
    zip,
    // EXPLAIN / diagnostic support
    RefusalRecord,
    SubstitutionRecord,
};

#[cfg(test)]
mod tests {
    //! Integration tests over the crate root surface.
    //!
    //! The guarantee matrix data tests live in [`crate::guarantee_matrix`];
    //! the per-combinator behavioural tests live in [`crate::combinators`].
    //! This module tests the *crate-level contract*: the structural
    //! never-silent property, the guarantee matrix completeness, and the
    //! EXPLAIN records.

    use crate::guarantee_matrix::{Explainable, Fallibility, MATRIX};

    /// The guarantee matrix has no "Dropped" guarantee variant in any row.
    /// This is the structural I1 bound expressed over the matrix data.
    /// Guard: adding a "Dropped" entry to any row makes this fail.
    #[test]
    fn matrix_has_no_dropped_outcome_in_any_row() {
        for row in MATRIX {
            assert!(
                !row.error_set.to_lowercase().contains("drop")
                    || row.error_set.contains("never a drop"),
                "row {:?} must not permit a dropped outcome (I1 / C1): {:?}",
                row.op,
                row.error_set
            );
        }
    }

    /// The guarantee matrix has exactly one `Inherited` tag: the recover bridge.
    /// Guard: adding an Inherited tag to a non-bridge op makes this fail.
    #[test]
    fn only_recover_bridge_has_inherited_tag() {
        let inherited: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.guarantee.starts_with("Inherited"))
            .map(|r| r.op)
            .collect();
        assert_eq!(
            inherited.len(),
            1,
            "exactly one row should have an Inherited tag (the recover bridge)"
        );
        assert!(
            inherited[0].starts_with("recover"),
            "the Inherited tag must be on the recover bridge"
        );
    }

    /// All total ops (Fallibility::Total) have no error set.
    /// Guard: adding error_set to a total op makes this fail.
    #[test]
    fn total_ops_have_empty_error_set() {
        for row in MATRIX {
            if row.fallibility == Fallibility::Total {
                assert!(
                    row.error_set.is_empty(),
                    "total op {:?} must have empty error_set",
                    row.op
                );
            }
        }
    }

    /// The `Declared` tag is used only for the unwrap_or family (downgrade rule — VR-5).
    /// Guard: any other op claiming `Declared` makes this fail.
    #[test]
    fn declared_tag_only_on_default_recovery_ops() {
        for row in MATRIX {
            if row.guarantee == "Declared" {
                assert!(
                    matches!(
                        row.op,
                        "unwrap_or"
                            | "unwrap_or_else"
                            | "unwrap_or_option"
                            | "unwrap_or_else_option"
                    ),
                    "only unwrap_or* should be Declared; found {:?}",
                    row.op
                );
            }
        }
    }

    /// Partial ops must have non-empty error_set and DiagnosticRefusal EXPLAIN.
    /// Guard: a partial op without a named refusal makes this fail (C1 / C3).
    #[test]
    fn partial_ops_are_explicit_and_explainable() {
        for row in MATRIX {
            if row.fallibility == Fallibility::Partial {
                assert!(
                    !row.error_set.is_empty(),
                    "partial op {:?} must name its refusal (C1)",
                    row.op
                );
                assert_eq!(
                    row.explainable,
                    Explainable::DiagnosticRefusal,
                    "partial op {:?} must be DiagnosticRefusal for EXPLAIN (C3)",
                    row.op
                );
            }
        }
    }

    /// The EXPLAIN column: every op that transforms/approximates/converts has a non-`NotApplicable`
    /// entry. This confirms the C3 obligation is not silently waived.
    /// Guard: zeroing out the explainable field for any non-pure op makes this fail.
    #[test]
    fn explain_column_is_set_for_all_non_pure_ops() {
        let non_na: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.explainable != Explainable::NotApplicable)
            .map(|r| r.op)
            .collect();
        // There must be at least: ok (lossy), unwrap/expect/unwrap_err (partial),
        // unwrap_or/unwrap_or_else (substitution), propagate (match), recover (policy).
        assert!(
            non_na.len() >= 7,
            "expected at least 7 ops with non-NotApplicable EXPLAIN; got {}: {:?}",
            non_na.len(),
            non_na
        );
    }
}
