//! The `std.numerics` guarantee matrix — encoded as data, asserted in tests (RFC-0016 §4.5).
//!
//! Every exported helper op has exactly one row describing:
//! - Its honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
//! - Its fallibility and the explicit error/refusal set (C1; RFC-0013).
//! - Its declared effects (C6 — all are `"none"` here; every helper is pure).
//! - Whether it surfaces an inspectable EXPLAIN artifact (C3; G11).
//!
//! The matrix is the **load-bearing deliverable** of the spec (spec §4 / RFC-0016 §4.5): it is
//! encoded as data so the test suite can assert invariants structurally, not just as prose.
//!
//! # Guarantee tag justification (VR-5 — spec §4)
//!
//! The honesty crux: the helpers are **structural** — they construct, attach, project, or check
//! a bound — so the **helper op itself** has no accuracy/precision semantics of its own and is
//! `Exact` (the RFC-0016 §4.1-C2 "len-style" case: no approximate numeric quantity is computed).
//! The thing that has a strength tag is the **carried `Bound`** — and its strength is **whatever
//! its `BoundBasis` supports**, set by the kernel/theorem, **never asserted by this module**.
//!
//! `declared` is the one exception: it is `Declared` (not `Exact`) because the *bound it
//! attaches* is always `UserDeclared` (M-I4), so the helper's own observable effect is a
//! strength downgrade — the tag of the output is `Declared`, so that is the row's tag.
//!
//! All helpers are pure (`effects: "none"`); C6 holds across the board.

use mycelium_core::GuaranteeStrength;

/// One row of the `std.numerics` guarantee matrix (RFC-0016 §4.5; spec §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// The exported helper op name.
    pub op: &'static str,
    /// The honest guarantee tag of the **helper op itself** (not the carried bound — see above).
    pub tag: GuaranteeStrength,
    /// Whether the op can return a `Result::Err` / `None`.
    pub fallible: bool,
    /// The explicit error/refusal set (or `"total"` when infallible).
    pub error_set: &'static str,
    /// Declared effects (C6). `"none"` for all pure helpers.
    pub effects: &'static str,
    /// Whether the op surfaces an inspectable EXPLAIN artifact (C3; G11).
    pub explainable: bool,
}

/// The `std.numerics` guarantee matrix (spec §4; RFC-0016 §4.5).
///
/// Structural invariants are verified by [`assert_matrix_invariants`] and the `#[test]`s below.
pub const GUARANTEE_MATRIX: &[GuaranteeRow] = &[
    // ── carrier constructors / projections ────────────────────────────────────
    GuaranteeRow {
        op: "exact",
        // FR: exact T; no bound (M-I1); tag = Exact on the value, not on a bound.
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: false, // No bound — no EXPLAIN artifact (M-I1).
    },
    GuaranteeRow {
        op: "value_of",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "bound_of",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: true, // Projects the carried bound — that *is* the EXPLAIN artifact (C3).
    },
    GuaranteeRow {
        op: "strength_of",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: true, // Projects the carried strength (part of the EXPLAIN surface).
    },
    // ── bound constructors ────────────────────────────────────────────────────
    GuaranteeRow {
        op: "error_bound",
        // The helper is Exact (it constructs a carrier; ε is the caller's datum, spec §4).
        // The carried bound's strength = basis-implied (ADR-011) — never set here.
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(NumErr::BadEps) when eps < 0 or non-finite",
        effects: "none",
        explainable: true, // The constructed bound + its basis is the EXPLAIN artifact.
    },
    GuaranteeRow {
        op: "prob_bound",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(NumErr::BadDelta) when delta not in [0,1] or non-finite",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "declared",
        // `declared` is Declared by construction (basis = UserDeclared; M-I4; always flagged).
        // The helper's own observable effect is producing a Declared-strength carrier, so the
        // row tag is Declared — the one non-Exact carrier constructor (VR-5 justification §4).
        tag: GuaranteeStrength::Declared,
        fallible: false,
        error_set: "total (the assertion never fails; it is unverified — M-I4/VR-5)",
        effects: "none",
        explainable: true, // Surfaced "declared, unverified" (M-I4/VR-5) — always EXPLAIN-able.
    },
    GuaranteeRow {
        op: "attach",
        // Structural: the tag of the helper is Exact (it attaches; carries/checks, no approx).
        // The CARRIED tag = basis-implied (M-I2/M-I3/M-I4) — never set by this helper.
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: true, // The basis ⇒ tag derivation is the EXPLAIN artifact.
    },
    // ── composition helpers ───────────────────────────────────────────────────
    GuaranteeRow {
        op: "combine",
        // Structural wrapper around `compose_error_bound`; helper is Exact.
        // Carried strength = meet of inputs, basis re-derived (spec §4; FR-N2).
        // Refuses when no sound ε rule exists — never fabricates (M-204 / C1).
        // Bound addition is outward-rounded (banked guard 1 / A2-01).
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(NumErr::NoRule) when no sound ε rule; Err(NumErr::Overflow) on overflow",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "union_delta",
        // Structural wrapper around the δ union monoid; helper is Exact.
        // Carried strength = meet of inputs.
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set:
            "Err(NumErr::NoRule) on empty input; Err(NumErr::BadDelta) on malformed input δ",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "accuracy_to_probability",
        // The rule application is exact; the carried δ is honestly worst-case when outside tol.
        // Never a silent tighten — outside tol ⇒ δ = 1 (honest worst case; ADR-010 §4).
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set:
            "Err(NumErr::NoRule) if input is not Error kind; Err(NumErr::BadDelta) if acc_delta not in [0,1]",
        effects: "none",
        explainable: true,
    },
    // ── inspection / EXPLAIN ──────────────────────────────────────────────────
    GuaranteeRow {
        op: "explain",
        // Pure projection of the certificate — exact (it IS the EXPLAIN artifact; G11; C3).
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: true, // It is the EXPLAIN artifact itself.
    },
    // ── tier-i re-validation checker ─────────────────────────────────────────
    GuaranteeRow {
        op: "check_error",
        // Re-validation verdict — exact (the trust anchor; never silent; ADR-010 tier-i).
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set:
            "Err(CheckErr::Rejected{recomputed,claimed}) when claim is tighter than re-derivation; \
             Err(CheckErr::Malformed) when ill-formed",
        effects: "none",
        explainable: true, // The verdict + recomputed-vs-claimed delta is the EXPLAIN artifact.
    },
    GuaranteeRow {
        op: "check_union",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set:
            "Err(CheckErr::Rejected{recomputed,claimed}) when claim is too tight; \
             Err(CheckErr::Malformed) on malformed input",
        effects: "none",
        explainable: true,
    },
];

/// Assert structural invariants on [`GUARANTEE_MATRIX`] — the RFC-0016 §4.5 obligation.
///
/// Called from the test suite. Panics with a descriptive message on any violation.
pub fn assert_matrix_invariants() {
    for row in GUARANTEE_MATRIX {
        // Non-empty op name.
        assert!(!row.op.is_empty(), "matrix row has empty op name");
        // Non-empty error-set string.
        assert!(
            !row.error_set.is_empty(),
            "op '{}': error_set must be non-empty",
            row.op
        );
        // Effects must be "none" (all numerics helpers are pure; C6).
        assert_eq!(
            row.effects, "none",
            "op '{}': all std.numerics helpers must be effect-free (C6)",
            row.op
        );
        // Non-Exact ops must be EXPLAIN-able (C3 — carries an accuracy/bound artifact).
        if row.tag != GuaranteeStrength::Exact {
            assert!(
                row.explainable,
                "op '{}': non-Exact ops must be EXPLAIN-able (C3)",
                row.op
            );
        }
        // Fallible ops must name their error set (not "total").
        if row.fallible {
            assert_ne!(
                row.error_set, "total",
                "op '{}': fallible op must name its error set",
                row.op
            );
        }
        // Infallible ops must have error_set = "total".
        if !row.fallible {
            // "total" is the canonical string for infallible (but the string may also carry
            // additional notes — accept any string that starts with "total").
            assert!(
                row.error_set.starts_with("total"),
                "op '{}': infallible op must have error_set starting with 'total', got '{}'",
                row.op,
                row.error_set
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::GuaranteeStrength;

    /// Matrix structural invariants hold (RFC-0016 §4.5).
    #[test]
    fn matrix_invariants_hold() {
        assert_matrix_invariants();
    }

    /// All expected ops appear in the matrix exactly once (coverage + no-duplicate guard).
    ///
    /// Mutation witness: removing any op from the matrix makes this test fail.
    #[test]
    fn matrix_contains_all_ops_exactly_once() {
        let expected = [
            "exact",
            "value_of",
            "bound_of",
            "strength_of",
            "error_bound",
            "prob_bound",
            "declared",
            "attach",
            "combine",
            "union_delta",
            "accuracy_to_probability",
            "explain",
            "check_error",
            "check_union",
        ];
        for op in &expected {
            let count = GUARANTEE_MATRIX.iter().filter(|r| r.op == *op).count();
            assert_eq!(count, 1, "op '{op}' must appear exactly once in the matrix");
        }
        // Matrix must have exactly as many rows as the expected list.
        assert_eq!(
            GUARANTEE_MATRIX.len(),
            expected.len(),
            "matrix row count ({}) must match the expected op list ({})",
            GUARANTEE_MATRIX.len(),
            expected.len()
        );
    }

    /// All ops declare `effects = "none"` (C6 — all helpers are pure).
    ///
    /// Mutation witness: adding an effect to any row breaks this test.
    #[test]
    fn all_ops_are_effect_free() {
        for row in GUARANTEE_MATRIX {
            assert_eq!(
                row.effects, "none",
                "op '{}': all std.numerics helpers must be effect-free (C6)",
                row.op
            );
        }
    }

    /// Non-Exact ops are EXPLAIN-able (C3). Only `declared` is non-Exact in this module.
    ///
    /// Mutation witness: marking `declared` as not-explainable breaks this.
    #[test]
    fn non_exact_ops_are_explainable() {
        for row in GUARANTEE_MATRIX {
            if row.tag != GuaranteeStrength::Exact {
                assert!(
                    row.explainable,
                    "op '{}': non-Exact op must be EXPLAIN-able (C3)",
                    row.op
                );
            }
        }
    }

    /// Structural/exact helpers are `Exact`; only `declared` is `Declared`.
    ///
    /// Mutation witness: changing `declared` to `Exact` or any other op to `Declared` fires this.
    #[test]
    fn only_declared_constructor_is_declared_strength() {
        let declared_ops: Vec<&str> = GUARANTEE_MATRIX
            .iter()
            .filter(|r| r.tag == GuaranteeStrength::Declared)
            .map(|r| r.op)
            .collect();
        assert_eq!(
            declared_ops,
            ["declared"],
            "only the 'declared' constructor should carry Declared strength (M-I4/VR-5); \
             all structural helpers are Exact (spec §4 tag justification)"
        );
    }

    /// `declared` is explicitly marked EXPLAIN-able (surfaces "declared, unverified"; M-I4/VR-5).
    ///
    /// Mutation witness: removing EXPLAIN from `declared` conceals the unverified flag.
    #[test]
    fn declared_is_explainable() {
        let row = GUARANTEE_MATRIX
            .iter()
            .find(|r| r.op == "declared")
            .expect("declared must be in the matrix");
        assert!(
            row.explainable,
            "declared must be EXPLAIN-able (surfaces 'declared, unverified' M-I4/VR-5)"
        );
    }

    /// Fallible ops name their error set (C1 — explicit refusals, never silent).
    ///
    /// Mutation witness: making a fallible op's error_set = "total" would conceal the refusal.
    #[test]
    fn fallible_ops_name_error_set() {
        for row in GUARANTEE_MATRIX {
            if row.fallible {
                assert_ne!(
                    row.error_set, "total",
                    "op '{}': fallible op must name its explicit error set (C1)",
                    row.op
                );
            }
        }
    }

    /// Checker ops (`check_error`, `check_union`) are marked fallible (they can Reject/Malform).
    ///
    /// Mutation witness: making either checker infallible means it can never refuse.
    #[test]
    fn checker_ops_are_fallible() {
        for name in ["check_error", "check_union"] {
            let row = GUARANTEE_MATRIX.iter().find(|r| r.op == name).unwrap();
            assert!(
                row.fallible,
                "checker op '{name}' must be fallible (it can Reject or Malform)"
            );
        }
    }

    /// Composition helpers (`combine`, `union_delta`) are marked fallible (can NoRule/Overflow).
    ///
    /// Mutation witness: marking composition ops infallible means they could never refuse.
    #[test]
    fn composition_ops_are_fallible() {
        for name in ["combine", "union_delta"] {
            let row = GUARANTEE_MATRIX.iter().find(|r| r.op == name).unwrap();
            assert!(
                row.fallible,
                "composition op '{name}' must be fallible (can refuse; M-204/C1)"
            );
        }
    }

    /// `explain` is total and EXPLAIN-able (it is the EXPLAIN artifact itself; G11; C3).
    ///
    /// Mutation witness: making `explain` fallible would mean a bound might not be inspectable.
    #[test]
    fn explain_is_total_and_explainable() {
        let row = GUARANTEE_MATRIX
            .iter()
            .find(|r| r.op == "explain")
            .expect("explain must be in the matrix");
        assert!(
            !row.fallible,
            "explain must be total (G11; C3 — always produces an artifact)"
        );
        assert!(
            row.explainable,
            "explain must be EXPLAIN-able (it is the artifact)"
        );
    }
}
