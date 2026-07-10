//! The `std.io` + `serialize` guarantee matrix encoded as **data** (RFC-0016 §4.5).
//!
//! Every exported operation has exactly one row in [`MATRIX`].  The matrix is the
//! load-bearing C2/VR-5 deliverable: guarantee tags are asserted in tests, not
//! prose-only.
//!
//! # Tag justification summary (VR-5 — downgrade rather than overclaim)
//!
//! | Tag | Rows | Reason |
//! |---|---|---|
//! | `Exact` | `serialize`, `to_json`, `read_all`, `read`, `write` | No accuracy/precision/probability semantics (RFC-0016 C2 "no accuracy semantics → Exact"). `serialize`/`to_json` are `Exact` *when `Ok`* but **fallible**: a non-finite `f64` has no JSON form and is refused (never a silent `null`). |
//! | `Empirical` | `deserialize`, `from_json`, `read_value` | Round-trip property established by proptest corpus; no checked theorem → `Empirical`, not `Proven` (VR-5 / spec §7-Q2) |
//!
//! # Effect column (C6)
//! - `"none"` — the op is pure over its byte input; no OS facility touched.
//! - `"io"` — the op reads from or writes to a `Source`/`Sink`; the `io` effect
//!   is declared on the signature (RFC-0014 §4.5).
//! - `"io + alloc(Budget)"` — chunked `read` additionally allocates a buffer
//!   bounded by the declared `Budget` (C6/ADR-015).
//!
//! # EXPLAIN-able column (C3)
//! - `"n/a"` — `serialize`/`to_json` are faithful projections; `read_all`/`read`/
//!   `write` are pure byte-movement.  None selects, converts, or approximates, so
//!   there is no hidden decision to expose.
//! - `"yes"` — the fallible ops (`deserialize`/`from_json`/`read_value`) carry an
//!   RFC-0013 diagnostic record with the failure locus (byte offset / field path),
//!   making decode failures legible (C3/G11).

/// Guarantee tag on the honesty lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`
/// (RFC-0016 §4.1 C2; VR-5).
///
/// All values are `'static` strings matching the lattice names so they can be
/// asserted in tests without a dependency on the core lattice type from this
/// matrix module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuaranteeTag {
    /// No accuracy / precision / probability semantics; the operation is the
    /// honest floor `Exact` (RFC-0016 C2).
    Exact,
    /// The property is established by a **checked side-condition theorem** (VR-5).
    /// Not used in this matrix — no round-trip theorem has been checked.
    Proven,
    /// The property holds over a **generated corpus** (proptest); not `Proven`
    /// because no theorem with checked side-conditions exists (VR-5).
    Empirical,
    /// The property is **asserted without a checked basis**; always FLAGGED (VR-5).
    Declared,
}

impl GuaranteeTag {
    /// Human-readable name matching the lattice notation (`"Exact"`, etc.).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            GuaranteeTag::Exact => "Exact",
            GuaranteeTag::Proven => "Proven",
            GuaranteeTag::Empirical => "Empirical",
            GuaranteeTag::Declared => "Declared",
        }
    }
}

/// Fallibility classification for an exported op (C1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// The op cannot fail for any well-formed input.
    Total,
    /// The op returns an explicit `Result` or `Option`; the error set is named in
    /// `error_set`.
    Fallible,
}

/// Whether the op surfaces an EXPLAIN artifact (C3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op carries an RFC-0013 diagnostic record with the failure locus —
    /// the machine-legible EXPLAIN surface.
    Yes,
    /// The op neither selects, converts, nor approximates; there is no hidden
    /// decision to expose (C3 `n/a`).
    NotApplicable,
}

/// One row in the `std.io` + `serialize` guarantee matrix (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// Exported operation name.
    pub op: &'static str,
    /// Guarantee tag (VR-5).
    pub guarantee: GuaranteeTag,
    /// Fallibility: total or fallible.
    pub fallibility: Fallibility,
    /// The explicit error set (empty string for total ops).
    pub error_set: &'static str,
    /// Declared effects (C6): `"none"`, `"io"`, or `"io + alloc(Budget)"`.
    pub effects: &'static str,
    /// Whether the op surfaces a C3 EXPLAIN artifact.
    pub explainable: Explainable,
}

/// The `std.io` + `serialize` guarantee matrix.
///
/// Eight rows — one per exported op (spec §4 guarantee matrix / RFC-0016 §4.5).
/// Asserted in `tests` — never prose-only (C2 / VR-5).
pub const MATRIX: &[MatrixRow] = &[
    // ── serialize: Value → bytes (total, Exact) ───────────────────────────────
    MatrixRow {
        op: "serialize",
        guarantee: GuaranteeTag::Exact,
        fallibility: Fallibility::Fallible,
        error_set: "Err(OutOfDomain) — non-finite f64 has no JSON form (never silent null)",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── deserialize: bytes → Value (fallible, Empirical) ──────────────────────
    MatrixRow {
        op: "deserialize",
        guarantee: GuaranteeTag::Empirical,
        fallibility: Fallibility::Fallible,
        error_set: "Err(Truncated|Malformed|UnknownTag|OutOfDomain|BudgetExceeded) @locus",
        effects: "none",
        explainable: Explainable::Yes,
    },
    // ── to_json: canonical JSON (total, Exact) ────────────────────────────────
    MatrixRow {
        op: "to_json",
        guarantee: GuaranteeTag::Exact,
        fallibility: Fallibility::Fallible,
        error_set: "Err(OutOfDomain) — non-finite f64 has no JSON form (never silent null)",
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── from_json: canonical JSON → Value (fallible, Empirical) ──────────────
    // Empirical = round-trip fidelity (from_json∘to_json ≡ id), established by a proptest corpus,
    // not a theorem (VR-5). NOTE (DN-16, scope-distinct framing): std.fmt tags its delegating
    // `from_json` `Exact` — a *different* claim (decode determinism, no accuracy semantics), not a
    // contradiction. See crates/mycelium-std-fmt/src/lib.rs "Tag-framing note". Both retained.
    MatrixRow {
        op: "from_json",
        guarantee: GuaranteeTag::Empirical,
        fallibility: Fallibility::Fallible,
        error_set: "Err(Malformed|UnknownTag|OutOfDomain|BudgetExceeded) @locus",
        effects: "none",
        explainable: Explainable::Yes,
    },
    // ── read_all: Source → bytes (total, Exact, declares io) ─────────────────
    MatrixRow {
        op: "read_all",
        guarantee: GuaranteeTag::Exact,
        fallibility: Fallibility::Fallible,
        error_set: "Err(UnexpectedEof|Refused|EffectBudget)",
        effects: "io",
        explainable: Explainable::Yes,
    },
    // ── read: chunked (total, Exact, declares io + alloc(Budget)) ─────────────
    MatrixRow {
        op: "read",
        guarantee: GuaranteeTag::Exact,
        fallibility: Fallibility::Fallible,
        error_set: "Err(Refused|EffectBudget)",
        effects: "io + alloc(Budget)",
        explainable: Explainable::Yes,
    },
    // ── write: Sink ← bytes (total, Exact, declares io) ──────────────────────
    MatrixRow {
        op: "write",
        guarantee: GuaranteeTag::Exact,
        fallibility: Fallibility::Fallible,
        error_set: "Err(Refused|EffectBudget)",
        effects: "io",
        explainable: Explainable::Yes,
    },
    // ── read_value: Source → Value (fallible, Empirical, declares io) ─────────
    MatrixRow {
        op: "read_value",
        guarantee: GuaranteeTag::Empirical,
        fallibility: Fallibility::Fallible,
        error_set: "Err(ReadValueError::Ser(_)|ReadValueError::Io(_)) @locus",
        effects: "io",
        explainable: Explainable::Yes,
    },
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{Explainable, Fallibility, GuaranteeTag, MATRIX};

    /// Every op named in the spec §3 surface appears in the matrix exactly once.
    /// Guard: removing or renaming any op from MATRIX makes this fail.
    #[test]
    fn matrix_contains_all_eight_exported_ops() {
        let expected = [
            "serialize",
            "deserialize",
            "to_json",
            "from_json",
            "read_all",
            "read",
            "write",
            "read_value",
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
            "matrix has unexpected extra or missing rows"
        );
    }

    /// The `Exact` ops are exactly the five stated in spec §4 (serialize, to_json,
    /// read_all, read, write).  The `Empirical` ops are the three round-trip ops.
    /// Guard: upgrading any `Empirical` to `Proven` makes this fail (VR-5).
    #[test]
    fn guarantee_tags_match_spec() {
        let exact_ops = ["serialize", "to_json", "read_all", "read", "write"];
        let empirical_ops = ["deserialize", "from_json", "read_value"];

        for op in &exact_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing from matrix", op));
            assert_eq!(
                row.guarantee,
                GuaranteeTag::Exact,
                "op {:?} must be Exact (spec §4)",
                op
            );
        }
        for op in &empirical_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing from matrix", op));
            assert_eq!(
                row.guarantee,
                GuaranteeTag::Empirical,
                "op {:?} must be Empirical (spec §4 / VR-5 — round-trip property \
                 established by proptest, not by a checked theorem)",
                op
            );
        }
    }

    /// No op is `Proven` (VR-5 — no round-trip theorem has been checked).
    /// Guard: upgrading any row to `Proven` without a checked theorem makes this fail.
    #[test]
    fn no_op_is_proven_without_a_checked_theorem() {
        // VR-5: `Proven` is allowed ONLY with a theorem whose side-conditions are
        // checked.  This spec does not assert `Proven`; it fixes the discipline
        // (spec §7-Q2).  This test is the mechanical guard against an inadvertent
        // upgrade.
        for row in MATRIX {
            assert_ne!(
                row.guarantee,
                GuaranteeTag::Proven,
                "op {:?} claims Proven without a checked theorem — \
                 downgrade to Empirical (VR-5 / spec §4.2 / §7-Q2)",
                row.op
            );
        }
    }

    /// No op is `Declared` (VR-5 — an asserted-but-unchecked claim must be FLAGGED
    /// and is not the right tag for any op in this module).
    #[test]
    fn no_op_is_declared() {
        for row in MATRIX {
            assert_ne!(
                row.guarantee,
                GuaranteeTag::Declared,
                "op {:?} uses the Declared tag (VR-5 FLAG — must be FLAGGED explicitly)",
                row.op
            );
        }
    }

    /// Fallible ops have a non-empty error set; total ops have an empty one (C1).
    /// Guard: setting a fallible op's error_set to "" makes this fail.
    #[test]
    fn fallibility_and_error_set_are_consistent() {
        for row in MATRIX {
            match row.fallibility {
                Fallibility::Total => assert!(
                    row.error_set.is_empty(),
                    "total op {:?} must have an empty error_set (C1)",
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

    /// The `serialize` and `to_json` ops are fallible: they refuse a `Value` carrying a non-finite
    /// `f64` (JSON has no such literal; `serde_json` would silently emit `null` — a lossy,
    /// identity-colliding encoding). Refusing is never-silent (C1/G2).
    /// Guard: flipping either back to `Total` (re-introducing the silent-null path) makes this fail.
    #[test]
    fn serialize_and_to_json_refuse_non_finite_fallibly() {
        for op in &["serialize", "to_json"] {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing", op));
            assert_eq!(
                row.fallibility,
                Fallibility::Fallible,
                "op {:?} must be fallible — non-finite f64 has no faithful JSON form (C1/G2)",
                op
            );
        }
    }

    /// The io ops declare the `io` effect (C6).
    /// Guard: changing any io op's effect to "none" makes this fail.
    #[test]
    fn io_ops_declare_io_effect() {
        let io_ops = ["read_all", "read", "write", "read_value"];
        for op in &io_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing", op));
            assert!(
                row.effects.contains("io"),
                "op {:?} must declare the io effect (C6 / spec §4)",
                op
            );
        }
    }

    /// The serialize ops are pure (no effects).
    /// Guard: adding an effect to serialize/deserialize/to_json/from_json makes this fail.
    #[test]
    fn serialize_ops_are_pure() {
        let pure_ops = ["serialize", "deserialize", "to_json", "from_json"];
        for op in &pure_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing", op));
            assert_eq!(
                row.effects, "none",
                "op {:?} must be pure/effect-free (C6 / spec §4)",
                op
            );
        }
    }

    /// `read` declares the additional `alloc(Budget)` effect (C6/ADR-015).
    #[test]
    fn read_declares_alloc_budget() {
        let row = MATRIX.iter().find(|r| r.op == "read").expect("read row");
        assert!(
            row.effects.contains("alloc"),
            "read must declare alloc(Budget) effect (C6/ADR-015)"
        );
    }

    /// The three fallible decode ops carry EXPLAIN artifacts (C3 — RFC-0013 locus).
    /// Guard: removing EXPLAIN from any decode op makes this fail.
    #[test]
    fn decode_ops_are_explainable() {
        let explain_ops = ["deserialize", "from_json", "read_value"];
        for op in &explain_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing", op));
            assert_eq!(
                row.explainable,
                Explainable::Yes,
                "op {:?} must be EXPLAIN-able (C3 — RFC-0013 diagnostic @locus)",
                op
            );
        }
    }

    /// The serialize/io write ops are `NotApplicable` for EXPLAIN (no
    /// selection/conversion/approximation).
    #[test]
    fn projection_ops_are_not_explainable() {
        let na_ops = ["serialize", "to_json"];
        for op in &na_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("op {:?} missing", op));
            assert_eq!(
                row.explainable,
                Explainable::NotApplicable,
                "op {:?} must be NotApplicable for EXPLAIN (C3 n/a — faithful projection)",
                op
            );
        }
    }
}
