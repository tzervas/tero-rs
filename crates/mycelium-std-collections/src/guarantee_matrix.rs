//! The `std.collections` guarantee matrix encoded as **data** (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has one row in [`MATRIX`]. The matrix is the load-bearing C2
//! deliverable (RFC-0016 §4.1 C2 / VR-5): guarantee tags are asserted in tests, not prose-only.
//!
//! # Guarantee tag justification (all `Exact`)
//! No collection op carries accuracy, precision, or probability semantics: `len` counts,
//! `get` retrieves, `insert` builds a new value — each is a deterministic structural fact.
//! RFC-0016 C2 makes `Exact` the floor explicitly ("an op with no accuracy semantics … is
//! simply `Exact`"). There is nothing to upgrade and nothing to overclaim (VR-5). The
//! substantive honesty claims live in the **fallibility** and **order** columns, not the
//! tag column (spec §4 tag justification).
//!
//! # Fallibility column
//! - `Total` — the op cannot fail for any well-formed input.
//! - `Fallible` — the op returns an explicit `Option` / `Result`; the error set is named (C1).
//!
//! # Effects column
//! Every op is effect-free (C6). Allocation for a new persistent node is intrinsic/bounded
//! and budget-free at this layer (`none*` in the spec table; recorded as `"none*"` here).
//!
//! # EXPLAIN column
//! Most ops select/convert/approximate nothing — `NotApplicable`. The ops that make a
//! visible decision expose it: `update`/`slice` refusals carry a diagnostic record;
//! `get_or`'s default is reified; and the iteration-order contract is itself the
//! inspectable artifact for `keys`/`values`/`entries`/`union`/…

/// Fallibility classification for an exported op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// The op is total: it cannot fail for any well-formed input.
    Total,
    /// The op can fail; the error set is described in `error_set`.
    Fallible,
}

/// Whether an op has an EXPLAIN obligation (C3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op selects / converts / approximates and must expose a reified artifact.
    Yes,
    /// The op reports a deterministic fact; there is no hidden decision to explain.
    NotApplicable,
}

/// One row in the guarantee matrix (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation name.
    pub op: &'static str,
    /// Guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
    /// All rows are `"Exact"` — no collection op has accuracy semantics (spec §4).
    pub guarantee: &'static str,
    /// Whether the op can fail and what the explicit error set is.
    pub fallibility: Fallibility,
    /// The explicit error / none-case description (empty string for total ops).
    pub error_set: &'static str,
    /// Declared effects (C6). `"none"` = no effects; `"none*"` = bounded allocation only.
    pub effects: &'static str,
    /// Whether the op has a C3 EXPLAIN obligation.
    pub explainable: Explainable,
}

/// The `std.collections` guarantee matrix. One row per exported op, encoded as data and
/// asserted in `tests` — never prose-only (RFC-0016 §4.5; spec §4).
///
/// **All rows are `Exact`** — no accuracy semantics anywhere in `std.collections`.
/// The load-bearing honesty is in the **fallibility** column (C1: explicit `Option`/`Result`)
/// and the **documented-order invariant** (the no-silent-reorder crux from RFC-0016 §4.4),
/// captured in the `explainable` field for the order-surfacing ops.
pub const MATRIX: &[MatrixRow] = &[
    // ─── Seq ─────────────────────────────────────────────────────────────────
    MatrixRow {
        op: "Seq::len",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::is_empty",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::get",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "None when index out of range (C1)",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::first",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "None on empty (C1)",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::push",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::pop",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "None on empty (C1, never a silent no-op)",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::update",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "Err(IndexOOB) when i >= len (C1)",
        effects: "none*",
        explainable: Explainable::Yes, // refusal record (RFC-0013 structured diagnostic)
    },
    MatrixRow {
        op: "Seq::concat",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::slice",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "Err(IndexOOB) on lo > hi or hi > len (no silent clamp, C1)",
        effects: "none*",
        explainable: Explainable::Yes, // refusal record (RFC-0013 structured diagnostic)
    },
    MatrixRow {
        op: "Seq::foldable",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ─── Map ─────────────────────────────────────────────────────────────────
    MatrixRow {
        op: "Map::get",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "None on missing key (C1, never a default)",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::contains_key",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::insert",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::remove",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::get_or",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::Yes, // the default is a reified named arg (spec §3)
    },
    MatrixRow {
        op: "Map::keys",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::Yes, // documented insertion order is the inspectable artifact
    },
    MatrixRow {
        op: "Map::values",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::Yes, // same documented order as keys
    },
    MatrixRow {
        op: "Map::entries",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::Yes, // documented order is the inspectable artifact
    },
    // ─── Set ─────────────────────────────────────────────────────────────────
    MatrixRow {
        op: "Set::contains",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Set::insert",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Set::remove",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Set::union",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::Yes, // result order (self-first, then other-only) is documented
    },
    MatrixRow {
        op: "Set::intersection",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::Yes, // result order (self's order, filtered) is documented
    },
    MatrixRow {
        op: "Set::difference",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none*",
        explainable: Explainable::Yes, // result order (self's order, filtered) is documented
    },
    MatrixRow {
        op: "Set::foldable",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ---- constructors & size observers (pure, total, Exact) --------------------
    MatrixRow {
        op: "Seq::empty",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Seq::from_slice",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::empty",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::len",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Map::is_empty",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Set::empty",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Set::len",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "Set::is_empty",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
];

#[cfg(test)]
mod tests {
    use super::{Explainable, Fallibility, MATRIX};

    /// Every exported op named in the spec §3 surface appears in the matrix.
    /// Guard: removing any op from MATRIX makes this fail.
    #[test]
    fn matrix_contains_all_expected_ops() {
        let expected = [
            "Seq::len",
            "Seq::is_empty",
            "Seq::get",
            "Seq::first",
            "Seq::push",
            "Seq::pop",
            "Seq::update",
            "Seq::concat",
            "Seq::slice",
            "Seq::foldable",
            "Map::get",
            "Map::contains_key",
            "Map::insert",
            "Map::remove",
            "Map::get_or",
            "Map::keys",
            "Map::values",
            "Map::entries",
            "Set::contains",
            "Set::insert",
            "Set::remove",
            "Set::union",
            "Set::intersection",
            "Set::difference",
            "Set::foldable",
            "Seq::empty",
            "Seq::from_slice",
            "Map::empty",
            "Map::len",
            "Map::is_empty",
            "Set::empty",
            "Set::len",
            "Set::is_empty",
        ];
        for name in &expected {
            assert!(
                MATRIX.iter().any(|r| r.op == *name),
                "matrix is missing op {:?} (spec §3)",
                name
            );
        }
        assert_eq!(
            MATRIX.len(),
            expected.len(),
            "matrix has unexpected extra rows"
        );
    }

    /// Every row carries the `Exact` guarantee tag (spec §4 / VR-5).
    /// Guard: changing any tag to a weaker label makes this fail.
    #[test]
    fn every_row_is_exact() {
        for row in MATRIX {
            assert_eq!(
                row.guarantee, "Exact",
                "op {:?} must be Exact — no collection op has accuracy semantics (RFC-0016 C2)",
                row.op
            );
        }
    }

    /// Total ops have an empty error set; fallible ops have a non-empty one (C1).
    /// Guard: setting a fallible op's error_set to "" makes this fail.
    #[test]
    fn fallibility_and_error_set_are_consistent() {
        for row in MATRIX {
            match row.fallibility {
                Fallibility::Total => assert!(
                    row.error_set.is_empty(),
                    "total op {:?} must have an empty error_set",
                    row.op
                ),
                Fallibility::Fallible => assert!(
                    !row.error_set.is_empty(),
                    "fallible op {:?} must name its error set (C1)",
                    row.op
                ),
            }
        }
    }

    /// The fallible ops are exactly those that can return `None` or `Err(IndexOOB)`.
    /// Guard: marking a total op as fallible (or vice versa) makes this fail.
    #[test]
    fn exactly_the_right_fallible_ops() {
        let fallible: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.fallibility == Fallibility::Fallible)
            .map(|r| r.op)
            .collect();
        // From spec §4: Seq::get, Seq::first, Seq::pop, Seq::update, Seq::slice, Map::get.
        let expected_fallible = [
            "Seq::get",
            "Seq::first",
            "Seq::pop",
            "Seq::update",
            "Seq::slice",
            "Map::get",
        ];
        for op in &expected_fallible {
            assert!(
                fallible.contains(op),
                "op {op:?} must be Fallible in the matrix"
            );
        }
        assert_eq!(
            fallible.len(),
            expected_fallible.len(),
            "unexpected extra/missing fallible ops: got {fallible:?}"
        );
    }

    /// The EXPLAIN-able ops are those that make a visible decision (C3).
    /// Guard: removing EXPLAIN from a decision-bearing op makes this fail.
    #[test]
    fn explainable_ops_are_decision_bearing() {
        let explainable: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.explainable == Explainable::Yes)
            .map(|r| r.op)
            .collect();
        // Spec §4: update/slice (refusal records), get_or (reified default),
        //          keys/values/entries/union/intersection/difference (documented order).
        let expected_explainable = [
            "Seq::update",
            "Seq::slice",
            "Map::get_or",
            "Map::keys",
            "Map::values",
            "Map::entries",
            "Set::union",
            "Set::intersection",
            "Set::difference",
        ];
        for op in &expected_explainable {
            assert!(
                explainable.contains(op),
                "op {op:?} must be Explainable::Yes in the matrix (C3)"
            );
        }
        assert_eq!(
            explainable.len(),
            expected_explainable.len(),
            "unexpected extra/missing Explainable::Yes ops: got {explainable:?}"
        );
    }

    /// Map::remove is total (returns `(Map, Option<V>)`) — the absence is in the Option, not in fallibility.
    #[test]
    fn map_remove_is_total_absence_in_option() {
        let row = MATRIX.iter().find(|r| r.op == "Map::remove").unwrap();
        assert_eq!(
            row.fallibility,
            Fallibility::Total,
            "Map::remove is total — absence is in the Option<V> second component, not a Fallible op"
        );
    }
}
