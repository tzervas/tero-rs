//! The `std.diag` guarantee matrix — encoded as **data**, asserted in tests (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has exactly one row describing:
//! - Its honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
//! - Its fallibility and the explicit error set (C1 — never-silent).
//! - Its declared effects (C6 — "none" for all `diag` ops except `sink`).
//! - Whether it exposes a C3 EXPLAIN artifact.
//! - The never-silent property: how the underlying error is preserved (I1 — presentation never
//!   gates propagation; no row permits a silent drop).
//!
//! # Guarantee tag justification (VR-5 — downgrade rather than overclaim)
//!
//! **All `diag` ops are `Exact`** because the module has **no accuracy semantics of its own**: a
//! diagnostic is a faithful, content-addressed *re-presentation* of an explicit error (RFC-0016
//! C2, the `len`-style case). `present` does not compute an approximate result; it pairs a truth
//! with a view of it. The honesty work `diag` does is *structural* (I1 — never suppress) and
//! *reporting* (RT5/VR-5 — surface a tag honestly), not *bounding*.
//!
//! The lattice tags that appear in `guarantee` / `audit_of` are **reported data, not op
//! guarantees**: `guarantee` returns the route's delivery strength (≤ `Declared` in v0, `None`
//! for `null`), and `audit_of` reports each crossing's bound as recorded — both downgrade-honest,
//! never upgraded (RT5/VR-5).
//!
//! # Never-silent property (I1 / RFC-0013 / RFC-0016 C1)
//!
//! The structural proof that no row can suppress the presented error:
//! - `present` is total and returns the error **unchanged** alongside the diagnostic (I1).
//! - Every fallible op (`from_json`, `resolve`, `on`, `from_file`, `resolve_route`, `sink`) fails
//!   **explicitly** — `Err(...)` / `None` / `Some(Err(...))`, never a sentinel or silent default.
//! - Route/sink resolution is **dispatched outside `present`** so even a `null` route or an
//!   `UnknownRoute` leaves the underlying error already surfaced and propagating.

/// Fallibility classification for a `std.diag` exported op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// The op always returns a value for all well-formed inputs — total in the error sense.
    Total,
    /// The op yields an explicit `Result`/`Option` / `Some(Result)`.
    Explicit,
}

/// Whether an op exposes a C3 EXPLAIN artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op is itself the EXPLAIN record (e.g. `present`, `to_human`, `to_json`).
    IsExplainRecord,
    /// The op yields a content-addressed EXPLAIN handle (e.g. `policy_ref`, `content_id`).
    ContentAddressedHandle,
    /// The op is not applicable for EXPLAIN (pure identity/value function).
    NotApplicable,
    /// The op names the known set — the closed vocabulary is the EXPLAIN artifact.
    ClosedVocabulary,
}

/// One row in the `std.diag` guarantee matrix (RFC-0016 §4.5; spec §4).
///
/// Encoded as data so tests can assert invariants structurally — never prose-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported op name (matches spec §3 table).
    pub op: &'static str,
    /// Honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
    pub guarantee: &'static str,
    /// Fallibility of the op.
    pub fallibility: Fallibility,
    /// The explicit error/none-case description (empty for total ops).
    pub error_set: &'static str,
    /// Declared effects (C6). "none" for all `diag` ops except `sink`.
    pub effects: &'static str,
    /// C3 EXPLAIN classification.
    pub explainable: Explainable,
    /// The never-silent property (I1): how the underlying error is preserved — never "silently
    /// dropped". This is the structural guarantee that no row can suppress the presented error.
    pub never_silent_property: &'static str,
}

/// The `std.diag` guarantee matrix (spec §4; RFC-0016 §4.5).
///
/// 14 rows — one per exported op in the spec §3 surface. All rows are `Exact` (VR-5); no row
/// permits a silent drop of the underlying error (I1/C1). Asserted structurally via `tests`.
pub const MATRIX: &[MatrixRow] = &[
    // ── presentation / identity ────────────────────────────────────────────────────────────────
    MatrixRow {
        op: "present",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total — returns the error UNCHANGED alongside the diagnostic (I1); \
                    cannot fail in a way that drops it",
        effects: "none (pure)",
        explainable: Explainable::IsExplainRecord,
        never_silent_property: "the error is returned unchanged in Presentation.error — \
                                the structural proof that the renderer cannot suppress it (I1)",
    },
    MatrixRow {
        op: "content_id",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total",
        effects: "none",
        explainable: Explainable::ContentAddressedHandle,
        never_silent_property: "pure identity function; no error path",
    },
    // ── dual projection (G11 / RFC-0013 I3) ────────────────────────────────────────────────────
    MatrixRow {
        op: "to_human",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total",
        effects: "none",
        explainable: Explainable::IsExplainRecord,
        never_silent_property: "renders the diagnostic (level-graded); carries the content id (I3)",
    },
    MatrixRow {
        op: "to_json",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total",
        effects: "none",
        explainable: Explainable::IsExplainRecord,
        never_silent_property: "lossless machine projection; carries the content id (I3)",
    },
    MatrixRow {
        op: "from_json",
        guarantee: "Exact",
        fallibility: Fallibility::Explicit,
        error_set:
            "Err(ParseErr) on malformed input — explicit, never a partial/sentinel record (C1)",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "explicit Err on malformed input (never a silent partial record)",
    },
    // ── error-class registry (X1 — looked up, never evaluated) ────────────────────────────────
    MatrixRow {
        op: "resolve (class)",
        guarantee: "Exact",
        fallibility: Fallibility::Explicit,
        error_set: "Err(UnknownClass) — class not in registry; no eval (X1)",
        effects: "none",
        explainable: Explainable::ClosedVocabulary,
        never_silent_property: "unknown class is an explicit error, never silently ignored (X1/G2)",
    },
    MatrixRow {
        op: "register (class)",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total — known-set insert; extension is membership, never eval (X1)",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "pure insert; no error path",
    },
    // ── reified policy (I4 — presentation/routing only) ───────────────────────────────────────
    MatrixRow {
        op: "on (add policy rule)",
        guarantee: "Exact",
        fallibility: Fallibility::Explicit,
        error_set: "Err(UnknownClass) — rule names an unregistered class (X1)",
        effects: "none",
        explainable: Explainable::ContentAddressedHandle,
        never_silent_property: "unknown class is an explicit error; the resulting policy is \
                                content-addressed (EXPLAIN-able)",
    },
    MatrixRow {
        op: "policy_ref",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total",
        effects: "none",
        explainable: Explainable::ContentAddressedHandle,
        never_silent_property: "pure content-hash projection; it IS the EXPLAIN handle",
    },
    MatrixRow {
        op: "from_file (policy)",
        guarantee: "Exact",
        fallibility: Fallibility::Explicit,
        error_set:
            "Err(UnknownClass) — whole-file reject on first unknown class (X1); never partial",
        effects: "none",
        explainable: Explainable::IsExplainRecord,
        never_silent_property: "whole-file reject; never partially/silently applied (X1/G2)",
    },
    // ── routes → RFC-0008 sinks (RT5 — honest delivery guarantees) ────────────────────────────
    MatrixRow {
        op: "resolve_route",
        guarantee: "Exact",
        fallibility: Fallibility::Explicit,
        error_set: "Err(UnknownRoute) — not in the closed v0 set; never a silent misroute",
        effects: "none",
        explainable: Explainable::ClosedVocabulary,
        never_silent_property: "unknown route is an explicit error, never a silent misroute (I1)",
    },
    MatrixRow {
        op: "sink (dispatch)",
        guarantee: "Exact",
        fallibility: Fallibility::Explicit,
        error_set: "Some(Err(UnknownRoute)) for a bad route; None for no route — OUTSIDE present, \
                    so it never gates propagation (I1)",
        effects: "io (the actual sink transport, RFC-0008; bounded, declared at the call)",
        explainable: Explainable::ContentAddressedHandle,
        never_silent_property: "dispatched OUTSIDE present (I1): the error has already surfaced; \
                                a null/unknown route cannot gate it",
    },
    // ── honest reporting of delivery guarantee and audit bound ────────────────────────────────
    MatrixRow {
        op: "guarantee (of a Delivery)",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total — REPORTS the sink's honest strength (None for Null; ≤ Declared in v0); \
                    never upgrades it (RT5/VR-5)",
        effects: "none",
        explainable: Explainable::IsExplainRecord,
        never_silent_property: "reports the delivery guarantee on the lattice; the null sink \
                                honestly says None (not delivered — RT5/VR-5)",
    },
    MatrixRow {
        op: "audit_of",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "total — a crossing with no statically-derivable certificate reports \
                    honesty: None (unknown ≠ Exact), never a fabricated bound (VR-5)",
        effects: "none",
        explainable: Explainable::IsExplainRecord,
        never_silent_property: "read-only audit projection (I5); honesty is READ OFF each \
                                certificate, never upgraded (VR-5)",
    },
];

// ─── Tests ────────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::{Explainable, Fallibility, MATRIX};

    /// The spec §3 lists 14 exported surface rows (the full §4.5 matrix). Verify coverage.
    /// Mutation witness: adding or removing a row without updating this test makes it fail.
    #[test]
    fn matrix_covers_all_spec_ops() {
        let expected_ops = [
            "present",
            "content_id",
            "to_human",
            "to_json",
            "from_json",
            "resolve (class)",
            "register (class)",
            "on (add policy rule)",
            "policy_ref",
            "from_file (policy)",
            "resolve_route",
            "sink (dispatch)",
            "guarantee (of a Delivery)",
            "audit_of",
        ];
        for name in &expected_ops {
            assert!(
                MATRIX.iter().any(|r| r.op == *name),
                "matrix is missing op {name:?} (spec §3/§4 — RFC-0016 §4.5)"
            );
        }
        assert_eq!(
            MATRIX.len(),
            14,
            "expected exactly 14 rows (spec §3 — all 14 exported ops)"
        );
    }

    /// **Every row is `Exact`** — `diag` has no accuracy semantics of its own (RFC-0016 C2 /
    /// spec §4 tag justification / VR-5).
    /// Mutation witness: accidentally marking any row as `Declared`/`Empirical`/`Proven` fails here.
    #[test]
    fn all_diag_ops_are_exact() {
        for row in MATRIX {
            assert_eq!(
                row.guarantee, "Exact",
                "op {:?} must be Exact (diag has no accuracy semantics — RFC-0016 C2/VR-5)",
                row.op
            );
        }
    }

    /// **Every row states a non-empty never_silent_property** (C1 — the structural guarantee that
    /// no op silently drops the underlying error must be stated per row; I1).
    /// Mutation witness: leaving `never_silent_property` empty on any row makes this fail.
    #[test]
    fn every_row_states_never_silent_property() {
        for row in MATRIX {
            assert!(
                !row.never_silent_property.is_empty(),
                "row {:?} must state its never_silent_property (C1/I1)",
                row.op
            );
        }
    }

    /// **Every row states its effects** (C6 — even if `"none"`).
    /// Mutation witness: leaving `effects` empty on any row makes this fail.
    #[test]
    fn every_row_states_effects() {
        for row in MATRIX {
            assert!(
                !row.effects.is_empty(),
                "row {:?} must state its declared effects (C6)",
                row.op
            );
        }
    }

    /// **Explicit ops state a non-empty `error_set`** (C1 — the error case must be named, never
    /// anonymous).
    /// Mutation witness: making an `Explicit` row have an empty `error_set` breaks this test.
    #[test]
    fn explicit_ops_have_nonempty_error_set() {
        for row in MATRIX {
            if row.fallibility == Fallibility::Explicit {
                assert!(
                    !row.error_set.is_empty(),
                    "explicit op {:?} must name its error set (C1)",
                    row.op
                );
            }
        }
    }

    /// `present` is the I1 crux: it must be `Total`, `Exact`, `IsExplainRecord`, and its error_set
    /// must explicitly name "UNCHANGED" (the structural proof that it cannot suppress the error).
    /// Mutation witness: changing `present`'s fallibility to `Explicit` or removing "UNCHANGED"
    /// from the error_set breaks this test.
    #[test]
    fn present_is_the_i1_crux() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "present")
            .expect("present must be in the matrix (I1 crux)");
        assert_eq!(row.guarantee, "Exact", "present must be Exact");
        assert_eq!(
            row.fallibility,
            Fallibility::Total,
            "present must be Total (I1 — the error always surfaces)"
        );
        assert_eq!(
            row.explainable,
            Explainable::IsExplainRecord,
            "present must be IsExplainRecord (the diagnostic IS the EXPLAIN artifact)"
        );
        assert!(
            row.error_set.contains("UNCHANGED"),
            "present's error_set must state the error is returned UNCHANGED (I1): {:?}",
            row.error_set
        );
    }

    /// `sink (dispatch)` is dispatched **outside** `present` (I1): its error_set and
    /// never_silent_property must explicitly state this.
    /// Mutation witness: removing "OUTSIDE" from either field fails this test.
    #[test]
    fn sink_is_dispatched_outside_present() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "sink (dispatch)")
            .expect("sink must be in the matrix");
        assert!(
            row.error_set.contains("OUTSIDE") || row.never_silent_property.contains("OUTSIDE"),
            "sink's error_set or never_silent_property must state it is OUTSIDE present (I1): \
             error_set={:?} never_silent_property={:?}",
            row.error_set,
            row.never_silent_property
        );
        assert_eq!(
            row.effects, "io (the actual sink transport, RFC-0008; bounded, declared at the call)",
            "sink must declare its io effect (C6)"
        );
    }

    /// `guarantee (of a Delivery)` must explicitly say it REPORTS the strength (RT5/VR-5), never
    /// upgrades it, and that the null sink yields None.
    /// Mutation witness: removing "None" or "Declared" from the error_set fails this test.
    #[test]
    fn guarantee_op_reports_without_upgrading() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "guarantee (of a Delivery)")
            .expect("guarantee must be in the matrix");
        assert!(
            row.error_set.contains("None"),
            "guarantee's error_set must mention None for the null sink (RT5/VR-5): {:?}",
            row.error_set
        );
        assert!(
            row.error_set.contains("Declared"),
            "guarantee's error_set must mention Declared as the v0 ceiling (RT5): {:?}",
            row.error_set
        );
    }

    /// `audit_of` must explicitly say honesty is READ OFF each certificate and never upgraded, and
    /// that `None` represents unknown (VR-5), not `Exact`.
    /// Mutation witness: removing "None" or "never upgraded"/"VR-5" fails this test.
    #[test]
    fn audit_of_reports_honesty_without_upgrading() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "audit_of")
            .expect("audit_of must be in the matrix");
        assert!(
            row.error_set.contains("None"),
            "audit_of's error_set must mention None (unknown ≠ Exact — VR-5): {:?}",
            row.error_set
        );
        assert!(
            row.never_silent_property.contains("VR-5") || row.error_set.contains("VR-5"),
            "audit_of must cite VR-5 (honesty read-off, never upgraded): {:?}",
            row.error_set
        );
    }

    /// Registry ops (`resolve`, `register`, `on`, `from_file`) all cite the X1 never-eval
    /// discipline.
    /// Mutation witness: removing "X1" from any registry op's error_set / never_silent_property
    /// breaks this test.
    #[test]
    fn registry_ops_cite_x1_never_eval() {
        let registry_ops = [
            "resolve (class)",
            "register (class)",
            "on (add policy rule)",
            "from_file (policy)",
        ];
        for op_name in &registry_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op_name)
                .unwrap_or_else(|| panic!("registry op {op_name:?} must be in the matrix"));
            let cites_x1 = row.error_set.contains("X1") || row.never_silent_property.contains("X1");
            assert!(
                cites_x1,
                "registry op {:?} must cite X1 (looked up, never evaluated): \
                 error_set={:?} never_silent_property={:?}",
                op_name, row.error_set, row.never_silent_property
            );
        }
    }

    /// Only `sink` has an `io` effect; all other rows are `"none"`.
    /// Mutation witness: adding `io` to any non-sink row breaks this test (C6 — effects declared
    /// explicitly per op, never smuggled in).
    #[test]
    fn only_sink_has_io_effect() {
        for row in MATRIX {
            if row.op == "sink (dispatch)" {
                assert!(
                    row.effects.contains("io"),
                    "sink must declare its io effect (C6): {:?}",
                    row.effects
                );
            } else {
                assert!(
                    !row.effects.contains("io"),
                    "op {:?} must not claim io effects — only sink does (C6): {:?}",
                    row.op,
                    row.effects
                );
            }
        }
    }

    /// `content_id` and `policy_ref` are the content-addressed handle ops (ADR-003/ADR-006).
    /// Mutation witness: changing their `explainable` to another variant fails this test.
    #[test]
    fn content_id_and_policy_ref_are_content_addressed_handles() {
        for op_name in &["content_id", "policy_ref"] {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *op_name)
                .unwrap_or_else(|| panic!("{op_name:?} must be in the matrix"));
            assert_eq!(
                row.explainable,
                Explainable::ContentAddressedHandle,
                "{op_name:?} must be ContentAddressedHandle (ADR-003/ADR-006)"
            );
        }
    }
}
