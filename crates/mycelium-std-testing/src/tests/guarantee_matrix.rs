//! White-box unit tests for `guarantee_matrix.rs` (the RFC-0016 Â§4.5 matrix as checked data).
//!
//! Extracted as-touched per the test-layout rule (CLAUDE.md Â§Test layout).

use crate::guarantee_matrix::*;
use mycelium_core::GuaranteeStrength;

/// The matrix has exactly 5 rows (spec §4 lists five ops).
#[test]
fn matrix_has_five_rows() {
    assert_eq!(
        MATRIX.len(),
        5,
        "spec §4 lists five ops in the guarantee matrix"
    );
}

/// Every row is `Exact` (spec §4 tag justification: harness ops are Exact mechanisms).
/// Guard: accidentally tagging any row Empirical/Declared would overclaim subject strength.
#[test]
fn all_rows_are_exact() {
    for row in MATRIX {
        assert_eq!(
            row.tag,
            GuaranteeStrength::Exact,
            "{} must be Exact — harness ops are Exact mechanisms (spec §4 / VR-5)",
            row.op
        );
    }
}

/// All five rows are EXPLAIN-able (the harness ops all surface inspection artifacts — C3).
#[test]
fn all_rows_are_explainable() {
    for row in MATRIX {
        assert!(
            row.explainable,
            "{} must be EXPLAIN-able (C3/G11/SC-3 — no black boxes)",
            row.op
        );
    }
}

/// Op names are unique (no duplicate rows).
#[test]
fn op_names_are_unique() {
    let mut names: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for row in MATRIX {
        assert!(
            names.insert(row.op),
            "duplicate op name '{}' in guarantee matrix",
            row.op
        );
    }
}

/// The harness ops that declare IO effects are `golden` and `differential` only.
/// (Spec §4: `golden` declares `io (read baseline)`; `differential` declares `io per backend`.)
/// Guard: accidentally marking effect-free ops as IO-declaring makes this fail.
#[test]
fn only_golden_and_differential_declare_io() {
    for row in MATRIX {
        if row.op == "golden" || row.op == "differential" {
            assert!(
                row.effects.contains("io"),
                "{} must declare IO effects (spec §4)",
                row.op
            );
        } else {
            assert!(
                row.effects.starts_with("none"),
                "{} must be effect-free (spec §4); got '{}'",
                row.op,
                row.effects
            );
        }
    }
}

/// The fallibility column of `summarize` and `is_green` is "total".
/// (These are total functions over verdicts — spec §4.)
#[test]
fn aggregator_rows_are_total() {
    for row in MATRIX {
        if row.op == "summarize" || row.op == "is_green" {
            assert_eq!(
                row.fallibility, "total",
                "{} must be total (spec §4)",
                row.op
            );
        }
    }
}
