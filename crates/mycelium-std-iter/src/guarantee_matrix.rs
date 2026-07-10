//! `std.iter` guarantee matrix — RFC-0016 §4.5, as checked data.
//!
//! This module encodes the spec §4 table as a static, exhaustively-checked constant. Every row
//! is asserted in tests rather than living as prose-only claims (the RFC-0016 §4.5 obligation).
//!
//! **Key column: `totality_preserving`** — the load-bearing column for `std.iter` (RFC-0016
//! §4.4: "total + terminating where the kernel guarantees it"). `true` means the combinator,
//! applied to a finite `Foldable`, lowers to or composes a single RFC-0007 §4.8 `for` fold,
//! whose synthesized helper descends structurally on the spine and is classified `Total` with
//! zero extension by the §4.5 checker. Termination is **inherited**, not re-proved here (KC-3).
//!
//! # Row count note
//! The spec §4 table has 18 rows, treating `any`/`all` as one row and `find`/`position` as one
//! row. This implementation splits each into separate rows for finer test granularity (+2) and
//! adds a row for `zip_exact` (the fallible companion to `zip`), for **21 rows total**. All split
//! rows carry identical guarantee tags; the test asserts the exact count.
//!
//! # Tag justification (VR-5 — downgrade rather than overclaim)
//! - Every eager combinator is `Exact`/total: it inherits the kernel fold's `Total`
//!   classification (RFC-0007 §4.8/§4.5). The inheritance is the module's whole honesty claim.
//! - `lazy_unfold` is `Declared`: it asserts (not proves) an unbounded source. It is the sole
//!   non-total entry; every other row is `Exact`.
//! - `reduce`, `find`, `position` are `Exact` AND total (they terminate) but **fallible**
//!   (`Option` return). Fallibility and totality are orthogonal (spec §4 "Tag justification").
//! - `step_by` is `Exact` AND total but **fallible** (`Result<_, ZeroStep>`).

use mycelium_core::GuaranteeStrength;

/// One row of the `std.iter` guarantee matrix (spec §4 / RFC-0016 §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// The exported op name.
    pub op: &'static str,
    /// Honest guarantee tag on the lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.
    pub tag: GuaranteeStrength,
    /// Whether the combinator preserves totality when applied to a finite `Foldable`.
    pub totality_preserving: bool,
    /// The fallibility shape: `"total"`, `"Option<E>"`, `"Result<_, ZeroStep>"`, etc.
    pub fallibility: &'static str,
    /// Declared effects (`"none"` for every op in this pure module).
    pub effects: &'static str,
    /// Whether the op surfaces an inspectable EXPLAIN artifact (C3).
    pub explainable: bool,
}

/// The full `std.iter` guarantee matrix — all ops (spec §4, 18 spec rows; 22 implementation rows
/// after splitting `any`/`all` and `find`/`position` into separate entries for test granularity).
///
/// Encoded as a static array so every cell can be asserted in tests.
pub const MATRIX: &[GuaranteeRow] = &[
    // ── transforms (5 rows) ─────────────────────────────────────────────────
    GuaranteeRow {
        op: "map",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "filter",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "scan",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "enumerate",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "flat_map",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    // ── reductions (7 rows — any/all and find/position split for granularity) ─
    GuaranteeRow {
        op: "fold",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "reduce",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "Option<E> (None on empty input)",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "count",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    // any/all: done-flag fold — total (walks full spine). FLAG Q3 / RFC-0007 §4.8.
    GuaranteeRow {
        op: "any",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: true, // AnyAllWitness carries the first-match index
    },
    GuaranteeRow {
        op: "all",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: true, // AnyAllWitness carries the first-failure index
    },
    // find/position: done-flag fold — total. None on no-match (C1).
    GuaranteeRow {
        op: "find",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "Option<E> (None when no element matches)",
        effects: "none",
        explainable: true, // done-flag fold; find carries its witness implicitly in the Option
    },
    GuaranteeRow {
        op: "position",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "Option<usize> (None when no element matches)",
        effects: "none",
        explainable: false,
    },
    // ── pair / merge (2 rows) ─────────────────────────────────────────────────
    GuaranteeRow {
        op: "zip",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total (truncates to min; ZipOutcome records the truncation)",
        effects: "none",
        explainable: true, // ZipOutcome carries the truncation point (Q1)
    },
    GuaranteeRow {
        op: "zip_exact",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility:
            "Err(ZipLengthMismatch { left, right }) on unequal lengths — never a silent truncation",
        effects: "none",
        explainable: true, // the ZipLengthMismatch is the reified refusal (C3)
    },
    GuaranteeRow {
        op: "chain",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    // ── bounded slicing (3 rows) ──────────────────────────────────────────────
    GuaranteeRow {
        op: "take",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "skip",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "step_by",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "Result<Foldable<E>, ZeroStep> (Err when k = 0)",
        effects: "none",
        explainable: false,
    },
    // ── transducer (1 row) ────────────────────────────────────────────────────
    GuaranteeRow {
        op: "transduce",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true,
        fallibility: "total",
        effects: "none",
        explainable: true, // Transducer::describe() exposes the fused step pipeline
    },
    // ── lazy surface (2 rows) — the one honest exception ──────────────────────
    GuaranteeRow {
        op: "lazy_unfold",
        // NOT total — the source may be unbounded. VR-5 downgrade: Declared, never Exact.
        tag: GuaranteeStrength::Declared,
        totality_preserving: false,
        fallibility:
            "total-call (source may be unbounded — the Lazy<E> type carries the Declared tag)",
        effects: "none",
        explainable: true, // the Lazy<E> type itself is the EXPLAIN artifact
    },
    GuaranteeRow {
        op: "lazy_take",
        tag: GuaranteeStrength::Exact,
        totality_preserving: true, // re-bounds a Lazy back to a finite Foldable
        fallibility: "total (given the Nat bound)",
        effects: "none",
        explainable: true, // the n parameter records the bound applied
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The matrix must have exactly 21 rows: 18 spec rows + 2 for splitting any/all and
    /// find/position into separate rows, + 1 for `zip_exact` (the fallible companion to `zip`).
    #[test]
    fn matrix_has_expected_row_count() {
        assert_eq!(
            MATRIX.len(),
            21,
            "matrix must have 21 rows (18 spec rows + 2 split rows + zip_exact)"
        );
    }

    #[test]
    fn all_eager_ops_are_exact() {
        for row in MATRIX {
            if row.op == "lazy_unfold" {
                continue;
            }
            assert_eq!(
                row.tag,
                mycelium_core::GuaranteeStrength::Exact,
                "eager op '{}' must be Exact (inherited from kernel fold)",
                row.op
            );
        }
    }

    #[test]
    fn lazy_unfold_is_declared_and_not_total() {
        let row = MATRIX.iter().find(|r| r.op == "lazy_unfold").unwrap();
        assert_eq!(row.tag, mycelium_core::GuaranteeStrength::Declared);
        assert!(!row.totality_preserving);
    }

    #[test]
    fn lazy_take_is_exact_and_totality_preserving() {
        let row = MATRIX.iter().find(|r| r.op == "lazy_take").unwrap();
        assert_eq!(row.tag, mycelium_core::GuaranteeStrength::Exact);
        assert!(row.totality_preserving);
    }

    #[test]
    fn all_rows_are_effect_free() {
        for row in MATRIX {
            assert_eq!(
                row.effects, "none",
                "op '{}' must declare no effects",
                row.op
            );
        }
    }

    #[test]
    fn all_eager_ops_preserve_totality() {
        for row in MATRIX {
            if row.op != "lazy_unfold" {
                assert!(
                    row.totality_preserving,
                    "eager op '{}' must preserve totality",
                    row.op
                );
            }
        }
    }

    #[test]
    fn exactly_one_non_total_row() {
        let non_total: Vec<&str> = MATRIX
            .iter()
            .filter(|r| !r.totality_preserving)
            .map(|r| r.op)
            .collect();
        assert_eq!(
            non_total,
            vec!["lazy_unfold"],
            "exactly one op must be non-total: lazy_unfold"
        );
    }

    #[test]
    fn explainable_ops_are_the_decision_bearing_ones() {
        let explainable: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.explainable)
            .map(|r| r.op)
            .collect();
        // Must include all decision-bearing ops.
        for op in &[
            "zip",
            "any",
            "all",
            "find",
            "transduce",
            "lazy_unfold",
            "lazy_take",
        ] {
            assert!(
                explainable.contains(op),
                "op '{}' should be EXPLAIN-able",
                op
            );
        }
        // Must NOT include pure structural transforms.
        for op in &["map", "filter", "fold", "count", "chain", "skip", "take"] {
            assert!(
                !explainable.contains(op),
                "op '{}' should NOT be EXPLAIN-able (pure structural transform)",
                op
            );
        }
    }
}
