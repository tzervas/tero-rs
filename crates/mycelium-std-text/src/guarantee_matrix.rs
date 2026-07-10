//! The `std.text` guarantee matrix encoded as **data** (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has one row in [`MATRIX`]. The matrix is the load-bearing C2
//! deliverable (RFC-0016 §4.1 C2 / VR-5): guarantee tags are asserted in tests, not prose-only.
//!
//! # Guarantee tag justification (all `Exact`)
//!
//! `std.text` carries **no accuracy/precision/probability semantics** — there is no ε bound, no
//! probability, no approximation. A string op either returns the correct characters/bytes or an
//! explicit `Err`. Per RFC-0016 §4.1 C2: "an op with no accuracy semantics … is simply `Exact`".
//! The `Exact` tag for each row is the *honest floor*, not an overclaim.
//!
//! The honesty load is entirely in the **fallibility column**: every restricted op names its
//! explicit error set, carries **where** it failed (byte/char index), and never returns a
//! sentinel, a clamp, or a partial result (C1 / G2).
//!
//! # Fallibility column
//! - `"total"` — the op cannot fail for any well-formed input.
//! - Anything else — the explicit `Result`/`Option` shape and the error set.
//!
//! # Effects column
//! Every op is effect-free (C6: no IO, time, randomness, or global state).
//!
//! # EXPLAIN-able column
//! `"yes"` for ops that can reject/fail (their error type is the inspectable EXPLAIN artifact
//! per RFC-0013 §4.1 I1). `"n/a"` for total ops that neither select, convert, nor approximate.

/// One row in the `std.text` guarantee matrix (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation name.
    pub op: &'static str,
    /// Guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
    /// All rows are `"Exact"` (spec §4).
    pub guarantee: &'static str,
    /// The fallibility description: `"total"` or the explicit error set.
    pub fallibility: &'static str,
    /// Declared effects (C6); `"none"` throughout.
    pub effects: &'static str,
    /// Whether the op exposes an EXPLAIN artifact (C3 / RFC-0013 §4.1 I1).
    /// `"yes"` for fallible ops (the error is the artifact); `"n/a"` for total ops.
    pub explainable: &'static str,
}

/// The `std.text` guarantee matrix — one row per exported op, encoded as data and asserted
/// in `tests` — never prose-only (RFC-0016 §4.5; spec §4).
///
/// **All rows are `Exact`** (no accuracy semantics).
/// **All rows declare no effects** (C6: pure value ops).
/// **Fallible rows are EXPLAIN-able** (their error type carries `where` + `expected`/`found`).
pub const MATRIX: &[MatrixRow] = &[
    // ─── Construction ─────────────────────────────────────────────────────
    MatrixRow {
        op: "from_chars",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "from_utf8",
        guarantee: "Exact",
        fallibility: "Err(Utf8Error::Invalid { byte, reason }) — never a silent U+FFFD",
        effects: "none",
        explainable: "yes",
    },
    MatrixRow {
        op: "concat",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "join",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    // ─── Immutable transforms ─────────────────────────────────────────────
    MatrixRow {
        op: "to_upper",
        guarantee: "Exact",
        fallibility: "total (returns a NEW value; never in-place)",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "to_lower",
        guarantee: "Exact",
        fallibility: "total (returns a NEW value; never in-place)",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "trim",
        guarantee: "Exact",
        fallibility: "total (returns a NEW value; never in-place)",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "replace",
        guarantee: "Exact",
        fallibility: "total (returns a NEW value; never in-place)",
        effects: "none",
        explainable: "n/a",
    },
    // ─── Length / iteration ────────────────────────────────────────────────
    MatrixRow {
        op: "len_bytes",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "len_chars",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "len_graphemes",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "chars",
        guarantee: "Exact",
        fallibility: "total",
        effects: "none",
        explainable: "n/a",
    },
    // ─── Slicing / indexing ────────────────────────────────────────────────
    MatrixRow {
        op: "slice",
        guarantee: "Exact",
        fallibility: "Err(BoundaryError::{ OutOfRange | InvalidRange | NotCharBoundary | NotGraphemeBoundary }) — never a silent snap or panic",
        effects: "none",
        explainable: "yes",
    },
    MatrixRow {
        op: "char_at",
        guarantee: "Exact",
        fallibility: "Err(BoundaryError::OutOfRange { len, asked }) — never a sentinel char",
        effects: "none",
        explainable: "yes",
    },
    // ─── Parse ────────────────────────────────────────────────────────────
    MatrixRow {
        op: "parse_int",
        guarantee: "Exact",
        fallibility: "Err(ParseErr::{ Empty | Invalid { at, expected, found } | OutOfRange { at, target } }) — never a sentinel 0",
        effects: "none",
        explainable: "yes",
    },
    MatrixRow {
        op: "parse_bool",
        guarantee: "Exact",
        fallibility: "Err(ParseErr::{ Empty | Invalid { at, expected, found } }) — never a sentinel false",
        effects: "none",
        explainable: "yes",
    },
    // ─── Encoding / transcoding ────────────────────────────────────────────
    MatrixRow {
        op: "encode_utf8",
        guarantee: "Exact",
        fallibility: "total (lossless: Text is already UTF-8)",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "to_utf16",
        guarantee: "Exact",
        fallibility: "total (lossless: UTF-8 → UTF-16)",
        effects: "none",
        explainable: "n/a",
    },
    MatrixRow {
        op: "to_latin1",
        guarantee: "Exact",
        fallibility: "Err(EncodeError::Unrepresentable { ch, at, target_encoding }) — never a silent U+FFFD",
        effects: "none",
        explainable: "yes",
    },
    MatrixRow {
        op: "to_latin1_lossy",
        guarantee: "Exact",
        fallibility: "total — returns Lossy{ value, substituted, marker }; substitution count is in the value",
        effects: "none",
        explainable: "yes",
    },
    MatrixRow {
        op: "from_utf16",
        guarantee: "Exact",
        fallibility: "Err(TranscodeError::{ UnpairedSurrogate { at } | Invalid { at } }) — never silent U+FFFD",
        effects: "none",
        explainable: "yes",
    },
];

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{MatrixRow, MATRIX};

    /// Every exported op named in the spec §3 surface appears in the matrix.
    /// Guard: removing any op from MATRIX makes this fail.
    #[test]
    fn matrix_contains_all_exported_ops() {
        let expected = [
            "from_chars",
            "from_utf8",
            "concat",
            "join",
            "to_upper",
            "to_lower",
            "trim",
            "replace",
            "len_bytes",
            "len_chars",
            "len_graphemes",
            "chars",
            "slice",
            "char_at",
            "parse_int",
            "parse_bool",
            "encode_utf8",
            "to_utf16",
            "to_latin1",
            "to_latin1_lossy",
            "from_utf16",
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
            "matrix has unexpected extra rows (expected {}, got {})",
            expected.len(),
            MATRIX.len()
        );
    }

    /// Every row carries the `Exact` guarantee tag (spec §4 / VR-5).
    /// Guard: changing any tag to a weaker label makes this fail.
    #[test]
    fn every_row_is_exact() {
        for row in MATRIX {
            assert_eq!(
                row.guarantee, "Exact",
                "op {:?} must be Exact — text has no accuracy semantics (spec §4 / C2)",
                row.op
            );
        }
    }

    /// Every row declares no effects (C6).
    /// Guard: adding an effect to any row makes this fail.
    #[test]
    fn every_row_declares_no_effects() {
        for row in MATRIX {
            assert_eq!(
                row.effects, "none",
                "op {:?} declares unexpected effects (C6 — all text ops are pure)",
                row.op
            );
        }
    }

    /// Fallible ops (those with non-`"total"` fallibility) are all EXPLAIN-able (`"yes"`).
    /// Guard: marking a fallible op as `"n/a"` for EXPLAIN makes this fail.
    #[test]
    fn fallible_ops_are_explainable() {
        for row in MATRIX {
            let is_total = row.fallibility.starts_with("total");
            if !is_total {
                assert_eq!(
                    row.explainable, "yes",
                    "fallible op {:?} must be EXPLAIN-able (C3 / RFC-0013 §4.1 I1)",
                    row.op
                );
            }
        }
    }

    /// Most total ops are EXPLAIN `"n/a"` (C3: no selection/conversion/approximation to explain).
    /// Exception: `to_latin1_lossy` is total but EXPLAIN-able because its `Lossy<T>` return type
    /// is itself the reified substitution artifact (the Lossy record carries `substituted` + `marker`,
    /// so the caller always sees what was substituted — C3 / spec §4).
    ///
    /// Guard: marking a total op other than `to_latin1_lossy` as EXPLAIN `"yes"` makes this fail.
    #[test]
    fn total_ops_not_explainable_except_lossy_variants() {
        // `to_latin1_lossy` is the only total op that is EXPLAIN-able (its Lossy return IS the
        // EXPLAIN artifact). All other total ops must be `"n/a"`.
        for row in MATRIX {
            let is_total = row.fallibility.starts_with("total");
            if is_total && row.op != "to_latin1_lossy" {
                assert_eq!(
                    row.explainable, "n/a",
                    "total op {:?} should not have EXPLAIN obligation (C3 n/a for total ops)",
                    row.op
                );
            }
        }
    }

    /// The spec's 5 fallible ops are exactly: `from_utf8`, `slice`, `char_at`, `parse_int`,
    /// `parse_bool`, `to_latin1`, `from_utf16`. `to_latin1_lossy` is total (its lossiness
    /// is reified in the Lossy return type — EXPLAIN-able but not fallible).
    /// Guard: changing the fallibility of any of these ops makes this fail.
    #[test]
    fn exactly_seven_strict_fallible_ops() {
        let expected_fallible = [
            "from_utf8",
            "slice",
            "char_at",
            "parse_int",
            "parse_bool",
            "to_latin1",
            "from_utf16",
        ];
        let fallible: Vec<&MatrixRow> = MATRIX
            .iter()
            .filter(|r| !r.fallibility.starts_with("total"))
            .collect();
        assert_eq!(
            fallible.len(),
            expected_fallible.len(),
            "expected {} fallible ops, got {}",
            expected_fallible.len(),
            fallible.len()
        );
        for name in &expected_fallible {
            assert!(
                fallible.iter().any(|r| r.op == *name),
                "expected fallible op {:?} not found in matrix",
                name
            );
        }
    }

    /// `to_latin1_lossy` is total but EXPLAIN-able (its Lossy<T> return type reifies the lossiness).
    /// Guard: marking `to_latin1_lossy` as not-explainable makes this fail.
    #[test]
    fn to_latin1_lossy_is_total_and_explainable() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "to_latin1_lossy")
            .expect("row exists");
        assert!(
            row.fallibility.starts_with("total"),
            "to_latin1_lossy must be total (Lossy carries the substitution count in the value)"
        );
        assert_eq!(
            row.explainable, "yes",
            "to_latin1_lossy must be EXPLAIN-able (the Lossy record reifies substitutions)"
        );
    }

    /// The matrix count matches the expected 21 rows (one per exported op).
    /// Guard: adding/removing ops without updating this test makes this fail.
    #[test]
    fn matrix_has_twenty_one_rows() {
        assert_eq!(
            MATRIX.len(),
            21,
            "spec §4 lists 21 exported ops; got {}",
            MATRIX.len()
        );
    }
}
