//! The `std.content` guarantee matrix encoded as **data** (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has one row in [`MATRIX`]. The matrix is the load-bearing C2
//! deliverable (RFC-0016 §4.1 C2 / VR-5): guarantee tags are asserted in tests, not prose-only.
//!
//! # Guarantee tag justification (all `Exact`)
//! A content hash is a *pure function of normalized structure* (RFC-0001 §4.6 WF4). There is no
//! accuracy / precision / probability semantics anywhere in `std.content`, so the lattice floor
//! `Exact` applies directly to every op (RFC-0016 C2: "an op with no accuracy semantics … is
//! simply `Exact`"). Determinism is the substantive claim: identical normalized inputs always
//! yield the same digest; the digest never depends on names, spans, formatting, or dynamic
//! metadata (ADR-003; RFC-0001 §4.6 / §4.8).
//!
//! # Fallibility column
//! - `Total` — the op cannot fail; returning `Exact` is guaranteed for all inputs.
//! - `Fallible` — the op returns an explicit `Option` / `Result`; the error set is named.
//!
//! # Effects column
//! Every op is effect-free (C6: no IO, time, randomness, or unbounded allocation). The name
//! lookups (`resolve_name` / `names_of`) read a registry that is itself content-addressed and
//! append-only; reads are effect-free observations.
//!
//! # EXPLAIN-able column
//! `n/a` throughout — EXPLAIN / policy artifacts (C3) reify *selection / conversion /
//! approximation* decisions. `std.content` does none of those; it reports deterministic facts.
//! A digest is its own witness (the identity *is* the hash), so there is no hidden decision to
//! make inspectable. This is the honest reading of C3, not a waiver (spec §5/C3).

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
    /// The op reports a deterministic fact; there is no hidden decision to explain (C3 `n/a`).
    NotApplicable,
}

/// One row in the guarantee matrix (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation name.
    pub op: &'static str,
    /// Guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
    /// All rows are `"Exact"` because content hashing is deterministic (spec §4).
    pub guarantee: &'static str,
    /// Whether the op can fail and what the explicit error set is.
    pub fallibility: Fallibility,
    /// The explicit error / none-case description (empty string for total ops).
    pub error_set: &'static str,
    /// Declared effects (C6). Empty string means no effects.
    pub effects: &'static str,
    /// Whether the op has a C3 EXPLAIN obligation.
    pub explainable: Explainable,
}

/// The `std.content` guarantee matrix.  One row per exported op, encoded as data and asserted in
/// `tests` — never prose-only (RFC-0016 §4.5; spec §4).
///
/// **All rows are `Exact`** (deterministic): a content hash is a pure function of normalized
/// structure (RFC-0001 §4.6 WF4; ADR-003). **All rows declare no effects** (C6: pure reads).
/// **All rows are `NotApplicable` for EXPLAIN** (C3: no selection/conversion/approximation).
pub const MATRIX: &[MatrixRow] = &[
    MatrixRow {
        op: "hash_of_value",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "hash_of_def",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "digest_eq",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "as_ref",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "parse_ref",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "Err(MalformedDigest) on bad shape",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "resolve_name",
        guarantee: "Exact",
        fallibility: Fallibility::Fallible,
        error_set: "None when name unbound",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    MatrixRow {
        op: "names_of",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
];

#[cfg(test)]
mod tests {
    use super::{Explainable, Fallibility, MatrixRow, MATRIX};

    /// Every exported op named in the spec §3 surface appears in the matrix.
    /// Guard: removing any op from MATRIX makes this fail.
    #[test]
    fn matrix_contains_all_seven_exported_ops() {
        let expected = [
            "hash_of_value",
            "hash_of_def",
            "digest_eq",
            "as_ref",
            "parse_ref",
            "resolve_name",
            "names_of",
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
                "op {:?} must be Exact — content hashing is deterministic (RFC-0001 §4.6 WF4)",
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

    /// Every row declares no effects (C6).
    /// Guard: adding an effect to any row makes this fail.
    #[test]
    fn every_row_declares_no_effects() {
        for row in MATRIX {
            assert_eq!(
                row.effects, "none",
                "op {:?} declares unexpected effects (C6): {:?}",
                row.op, row.effects
            );
        }
    }

    /// Every row has EXPLAIN = NotApplicable (spec §4 / C3 n/a).
    /// Guard: marking any row Yes makes this fail.
    #[test]
    fn every_row_is_not_applicable_for_explain() {
        for row in MATRIX {
            assert_eq!(
                row.explainable,
                Explainable::NotApplicable,
                "op {:?} unexpectedly requires EXPLAIN (C3 is n/a for std.content — spec §5/C3)",
                row.op
            );
        }
    }

    /// The two fallible ops in the spec are `parse_ref` and `resolve_name`.
    /// Guard: changing either op's fallibility makes this fail.
    #[test]
    fn exactly_two_fallible_ops() {
        let fallible: Vec<&MatrixRow> = MATRIX
            .iter()
            .filter(|r| r.fallibility == Fallibility::Fallible)
            .collect();
        assert_eq!(fallible.len(), 2, "expected exactly 2 fallible ops");
        assert!(fallible.iter().any(|r| r.op == "parse_ref"));
        assert!(fallible.iter().any(|r| r.op == "resolve_name"));
    }
}
