//! The `std.math` guarantee matrix — encoded as data, asserted in tests (RFC-0016 §4.5).
//!
//! Every exported op has exactly one row describing:
//! - Its honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.
//! - Its fallibility and the explicit error set (C1).
//! - Its declared effects (C6 — all are `"none"` here).
//! - Whether it exposes an EXPLAIN artifact (C3).
//!
//! The matrix is the *load-bearing deliverable* of the spec (spec §4 / RFC-0016 §4.5): it is
//! encoded as data so the test suite can assert invariants structurally, not just as prose.
//!
//! # Honesty note on approximate-op tags (VR-5)
//!
//! All approximate ops (`sqrt`, `cbrt`, `exp`, `log`, `logb`, `pow`, `hypot`, `sin`, `cos`, `tan`,
//! `asin`, `acos`, `atan`, `atan2`) carry `Declared` in this implementation because the
//! transcendental compute floor is the platform libm via Rust's `f64` intrinsics — an unaudited
//! `wild` floor (FLAG in lib.rs; M-541 / §8-Q6). When M-541 lands with an audited `std-sys`
//! surface, the relevant ops can be upgraded to `Empirical` (empirically fit bound) or `Proven`
//! (cited theorem with checked side-conditions) — but only with a checked basis (VR-5).

use mycelium_core::GuaranteeStrength;

/// One row of the `std.math` guarantee matrix (RFC-0016 §4.5; spec §4).
///
/// Encoded as data so tests can assert invariants structurally — never prose-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// The exported op's name.
    pub op: &'static str,
    /// The honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.
    ///
    /// For approximate ops this is the strength the `bound.basis` justifies (VR-5); it is never
    /// asserted above what the evidence supports.
    pub tag: GuaranteeStrength,
    /// Whether the op returns a `Result` / is fallible.
    pub fallible: bool,
    /// The explicit error set (or `"total"` for total ops).
    pub error_set: &'static str,
    /// Declared effects (`"none"` for all pure math ops, per C6).
    pub effects: &'static str,
    /// Whether the op surfaces an inspectable EXPLAIN artifact (C3).
    pub explainable: bool,
}

/// The `std.math` guarantee matrix (spec §4; RFC-0016 §4.5).
///
/// All approximate rows carry `Declared` (see the module honesty note). Every invariant is
/// asserted via [`assert_matrix_invariants`].
pub const GUARANTEE_MATRIX: &[GuaranteeRow] = &[
    // ---- exact, total ----
    GuaranteeRow {
        op: "abs",
        tag: GuaranteeStrength::Exact,
        fallible: true, // Err(Overflow) at i64::MIN
        error_set: "Err(Overflow) when x == i64::MIN",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "neg",
        tag: GuaranteeStrength::Exact,
        fallible: true, // Err(Overflow) at i64::MIN
        error_set: "Err(Overflow) when x == i64::MIN",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "signum",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "min",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total (tie rule: first arg; documented, never silent)",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "max",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_set: "total (tie rule: first arg; documented, never silent)",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "gcd",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(Overflow) when the true gcd is 2^63 (inputs drawn from {0, i64::MIN})",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "lcm",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(Overflow)",
        effects: "none",
        explainable: false,
    },
    // ---- exact, domain-restricted ----
    GuaranteeRow {
        op: "checked_div",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(DivByZero | Overflow)",
        effects: "none",
        explainable: true, // the refusal record names the restriction
    },
    GuaranteeRow {
        op: "checked_rem",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(DivByZero | Overflow)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "ratio",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(DivByZero | Overflow)",
        effects: "none",
        explainable: true,
    },
    // ---- exact rounding (mode is reified/EXPLAIN-able) ----
    GuaranteeRow {
        op: "floor",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN/infinite; Err(Overflow) when out of i64 range",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "ceil",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN/infinite; Err(Overflow) when out of i64 range",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "trunc",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN/infinite; Err(Overflow) when out of i64 range",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "round",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN/infinite; Err(Overflow) when out of i64 range",
        effects: "none",
        explainable: true, // mode is the EXPLAIN artifact (C3)
    },
    // ---- approximate: Declared (libm floor, unaudited; FLAG M-541 / §8-Q6) ----
    GuaranteeRow {
        op: "sqrt",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(NegativeDomain)",
        effects: "none",
        explainable: true, // carries Bound + basis cert
    },
    GuaranteeRow {
        op: "cbrt",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN or infinite",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "exp",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(Overflow | OutOfDomain)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "log",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(NonPositiveDomain)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "logb",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(NonPositiveDomain | BadBase)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "pow",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(DivByZero | OutOfDomain | Overflow)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "hypot",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain | Overflow)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "sin",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN or infinite",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "cos",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN or infinite",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "tan",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain | PoleDomain)",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "asin",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain) when |x| > 1",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "acos",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain) when |x| > 1",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "atan",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain) for NaN or infinite",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "atan2",
        tag: GuaranteeStrength::Declared,
        fallible: true,
        error_set: "Err(OutOfDomain | PoleDomain)",
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
        // Non-empty error set string.
        assert!(
            !row.error_set.is_empty(),
            "op {}: error_set must be non-empty",
            row.op
        );
        // Effects string must be "none" (all math ops are pure; C6).
        assert_eq!(
            row.effects, "none",
            "op {}: all math ops must be effect-free (C6)",
            row.op
        );
        // Approximate ops (Declared/Empirical/Proven for non-exact) must be EXPLAIN-able (C3).
        if row.tag != GuaranteeStrength::Exact {
            assert!(
                row.explainable,
                "op {}: non-Exact ops must be EXPLAIN-able (C3 — carries bound cert)",
                row.op
            );
        }
        // Every fallible op must have a non-empty error_set (not "total").
        if row.fallible {
            assert_ne!(
                row.error_set, "total",
                "op {}: fallible op must name its error set",
                row.op
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matrix structural invariants hold (RFC-0016 §4.5).
    #[test]
    fn matrix_invariants_hold() {
        assert_matrix_invariants();
    }

    /// All expected ops appear in the matrix exactly once.
    #[test]
    fn matrix_contains_all_ops_exactly_once() {
        let expected = [
            "abs",
            "neg",
            "signum",
            "min",
            "max",
            "gcd",
            "lcm",
            "checked_div",
            "checked_rem",
            "ratio",
            "floor",
            "ceil",
            "trunc",
            "round",
            "sqrt",
            "cbrt",
            "exp",
            "log",
            "logb",
            "pow",
            "hypot",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
        ];
        for op in &expected {
            let count = GUARANTEE_MATRIX.iter().filter(|r| r.op == *op).count();
            assert_eq!(count, 1, "op '{op}' must appear exactly once in the matrix");
        }
        // Matrix must have exactly as many rows as the expected list.
        assert_eq!(
            GUARANTEE_MATRIX.len(),
            expected.len(),
            "matrix row count must match the expected op list"
        );
    }

    /// Exact ops are never EXPLAIN-able for accuracy (they have no accuracy semantics).
    ///
    /// Exception: exact ops that carry a refusal record (checked_div, checked_rem, ratio, round)
    /// ARE EXPLAIN-able — the refusal record / mode is the EXPLAIN artifact (C3). This test
    /// verifies the full policy: non-exact ops MUST be explain-able; exact ops that are NOT
    /// refusal-carriers must NOT be explain-able.
    #[test]
    fn exact_non_refusal_ops_are_not_explainable() {
        // These exact ops have no accuracy semantics and no EXPLAIN artifact.
        let exact_non_explain = [
            "abs", "neg", "signum", "min", "max", "gcd", "lcm", "floor", "ceil", "trunc",
        ];
        for op in &exact_non_explain {
            let row = GUARANTEE_MATRIX.iter().find(|r| r.op == *op).unwrap();
            assert!(
                !row.explainable,
                "op {op}: exact non-refusal op must not be marked explainable"
            );
        }
    }

    /// Non-exact ops (approximate family) are all EXPLAIN-able (C3 — carry bound cert).
    #[test]
    fn non_exact_ops_are_explainable() {
        for row in GUARANTEE_MATRIX {
            if row.tag != GuaranteeStrength::Exact {
                assert!(
                    row.explainable,
                    "op {}: non-Exact op must be EXPLAIN-able (C3)",
                    row.op
                );
            }
        }
    }

    /// All approximate ops carry `Declared` strength (VR-5 honesty — libm floor, FLAG M-541).
    ///
    /// Mutation witness: changing any approximate op to `Empirical`/`Proven` → assertion fires.
    #[test]
    fn approximate_ops_carry_declared_not_higher() {
        let approximate_ops = [
            "sqrt", "cbrt", "exp", "log", "logb", "pow", "hypot", "sin", "cos", "tan", "asin",
            "acos", "atan", "atan2",
        ];
        for op in &approximate_ops {
            let row = GUARANTEE_MATRIX.iter().find(|r| r.op == *op).unwrap();
            assert_eq!(
                row.tag,
                GuaranteeStrength::Declared,
                "op {op}: approximate ops must carry Declared (not higher) while libm floor is \
                 unaudited (VR-5; FLAG M-541 — upgrade requires a checked basis, never asserted)"
            );
        }
    }

    /// All ops declare effects = "none" (C6 — all math ops are pure).
    #[test]
    fn all_ops_are_effect_free() {
        for row in GUARANTEE_MATRIX {
            assert_eq!(
                row.effects, "none",
                "op {}: all std.math ops must be effect-free (C6)",
                row.op
            );
        }
    }
}
