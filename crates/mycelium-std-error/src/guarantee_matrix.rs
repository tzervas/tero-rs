//! The `std.error` guarantee matrix encoded as **data** (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has one row in [`MATRIX`]. The matrix is the load-bearing C2
//! deliverable (RFC-0016 §4.1 C2 / VR-5): guarantee tags are asserted in tests, not
//! prose-only.
//!
//! # Guarantee tag justification
//! Pure combinators (`map`, `map_err`, `and_then`, `or_else`, `filter`, `ok_or`,
//! `ok_or_else`, `ok`, `transpose`, `flatten`, `zip`, `inspect`, `inspect_err`,
//! `propagate`-style, `unwrap`/`expect`/`unwrap_err`) are `Exact`: they are pure value
//! transformers with no accuracy/precision/probability semantics (RFC-0016 C2 "len-style"
//! case).
//!
//! `unwrap_or` / `unwrap_or_else` are `Declared`: the substituted default value is
//! *asserted*, not proven — RFC-0014 I2 ("recovery never fabricates or upgrades a
//! guarantee"). Downgrade is the rule (VR-5).
//!
//! `recover` (the RFC-0014 bridge) carries an **inherited** tag — it accepts whatever
//! honest tag the RFC-0014 policy attaches to a recovered value. This module never
//! launders that tag upward (VR-5 / I2). Because the tag varies with the policy, the
//! matrix records `"Inherited-from-policy (≤ Declared by I2)"` — the narrowest honest
//! characterisation without fabricating a concrete tag (G2/VR-5).
//!
//! # Fallibility / never-silent (C1)
//! No row permits a silent drop. Every combinator either:
//! - transforms the `Err`/`None` (it survives in the result), or
//! - explicitly re-propagates it, or
//! - explicitly recovers it (with an honest tag), or
//! - refuses loudly (partial accessors: `unwrap`/`expect`/`unwrap_err`).
//!
//! The one "lossy" op, `ok` (`Result→Option`), discards `ε` — **flagged as an explicit
//! lossy conversion** (EXPLAIN-able, C3; FLAG Q2 in spec §7), not an unflagged drop.

/// Fallibility classification for an exported op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// The op is total: it always returns a value for all well-formed inputs (no panic /
    /// abort on the *normal* path). Note: `unwrap`/`expect`/`unwrap_err` are `Partial`
    /// (see below) — they are total-when-successful but abort on the wrong variant.
    Total,
    /// The op is partial: on the wrong variant it refuses loudly (abort + diagnostic),
    /// never a silent default. This is the explicitly-named partial accessor family.
    Partial,
    /// The op yields an explicit `Option`/`Result` (it is a combinator whose *output*
    /// is a sum — the caller must inspect it).
    Combinatorial,
}

/// Whether an op has a C3 EXPLAIN obligation (selects / converts / approximates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op has no selection/conversion/approximation; no EXPLAIN artifact needed.
    NotApplicable,
    /// The op has a flagged lossy conversion that must be EXPLAIN-noted (the `ok` op).
    LossyConversion,
    /// The op is inspectable: its outcome records the `PolicyRef` (the `recover` bridge).
    PolicyRef,
    /// The op elaborates to an inspectable `Match` node (the `?`-style propagation form).
    MatchElaboration,
    /// The op records a diagnostic refusal (partial accessors).
    DiagnosticRefusal,
    /// The substituted default is recorded (the `unwrap_or` family).
    SubstitutedDefault,
}

/// One row in the `std.error` guarantee matrix (RFC-0016 §4.5; spec §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation name.
    pub op: &'static str,
    /// Guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
    pub guarantee: &'static str,
    /// Whether the op is total, partial, or combinatorial.
    pub fallibility: Fallibility,
    /// The explicit error/none-case description (empty for total/combinatorial with no
    /// special outcome, non-empty for partial ops and the recover bridge).
    pub error_set: &'static str,
    /// Declared effects (C6). "none" means no effects.
    pub effects: &'static str,
    /// Whether the op has a C3 EXPLAIN obligation.
    pub explainable: Explainable,
    /// The "never-silent" property: how errors/nones are handled — propagated, transformed,
    /// recovered-explicitly, or refused-loudly. No "silently dropped" variant (I1/C1).
    pub never_silent_property: &'static str,
}

/// The `std.error` guarantee matrix. One row per exported op, encoded as data and asserted
/// in `tests` — never prose-only (RFC-0016 §4.5; spec §4).
///
/// **No row permits a silent drop of an error** (RFC-0014 I1 / RFC-0016 C1). Every row
/// either transforms the sum (the error survives in the result), re-propagates it,
/// recovers it explicitly with an honest tag, or — for the partial accessors — refuses
/// loudly.
pub const MATRIX: &[MatrixRow] = &[
    // ---- transform (keep the sum shape) ----------------------------------------
    MatrixRow {
        op: "map",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Err passes through unchanged (error preserved in sum)",
    },
    MatrixRow {
        op: "map_err",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Ok passes through; Err transformed (error preserved in sum)",
    },
    MatrixRow {
        op: "and_then",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Err short-circuits and propagates (never dropped)",
    },
    MatrixRow {
        op: "or_else",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "explicit recovery hook; must yield a Result (recover or re-propagate, never a drop)",
    },
    MatrixRow {
        op: "filter",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Some->None is a typed transition (named absence), not a silent loss",
    },
    MatrixRow {
        op: "inspect",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none (closure may declare its own — transparent to combinator per C6)",
        explainable: Explainable::NotApplicable,
        never_silent_property: "peek Ok; value and sum shape unchanged",
    },
    MatrixRow {
        op: "inspect_err",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none (closure may declare its own — transparent to combinator per C6)",
        explainable: Explainable::NotApplicable,
        never_silent_property: "peek Err; value and propagation unchanged",
    },
    // ---- convert between Option and Result -------------------------------------
    MatrixRow {
        op: "ok_or",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "None becomes Err(e): names the absence explicitly (never a drop)",
    },
    MatrixRow {
        op: "ok_or_else",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none (closure may declare its own — transparent to combinator per C6)",
        explainable: Explainable::NotApplicable,
        never_silent_property: "None becomes Err(e): names the absence explicitly (never a drop)",
    },
    MatrixRow {
        op: "ok",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::LossyConversion,
        never_silent_property: "Err->None: FLAGGED lossy conversion (EXPLAIN-able, spec Q2); not an unflagged drop",
    },
    MatrixRow {
        op: "transpose",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Err inside Option propagates out; no error is dropped",
    },
    MatrixRow {
        op: "flatten",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "inner Err propagates to outer; no wrapping discarded",
    },
    MatrixRow {
        op: "zip",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "either None short-circuits to None; no value silently dropped",
    },
    // ---- defaulted accessors (recover with honest tag) -------------------------
    MatrixRow {
        op: "unwrap_or",
        guarantee: "Declared",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::SubstitutedDefault,
        never_silent_property:
            "recovers with explicitly-supplied default (Declared tag per I2/VR-5); never upgrades",
    },
    MatrixRow {
        op: "unwrap_or_else",
        guarantee: "Declared",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none (closure may declare its own)",
        explainable: Explainable::SubstitutedDefault,
        never_silent_property:
            "recovers with computed default (Declared tag per I2/VR-5); honest tag from closure",
    },
    MatrixRow {
        op: "unwrap_or_option",
        guarantee: "Declared",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::SubstitutedDefault,
        never_silent_property:
            "Option variant: recovers None with explicitly-supplied default (Declared per I2/VR-5)",
    },
    MatrixRow {
        op: "unwrap_or_else_option",
        guarantee: "Declared",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none (closure may declare its own)",
        explainable: Explainable::SubstitutedDefault,
        never_silent_property:
            "Option variant: recovers None with computed default (Declared per I2/VR-5)",
    },
    // ---- explicit propagation --------------------------------------------------
    MatrixRow {
        op: "propagate (?-style)",
        guarantee: "Exact",
        fallibility: Fallibility::Combinatorial,
        error_set: "",
        effects: "none",
        explainable: Explainable::MatchElaboration,
        never_silent_property:
            "propagates Err/None to caller (the default posture); elaborates to Match (RFC-0014 §4.3)",
    },
    // ---- explicit named partial accessors --------------------------------------
    MatrixRow {
        op: "unwrap",
        guarantee: "Exact",
        fallibility: Fallibility::Partial,
        error_set: "on Err: refuses loudly (abort + diagnostic); never a silent default",
        effects: "none",
        explainable: Explainable::DiagnosticRefusal,
        never_silent_property: "EXPLICIT named partial; refuses loudly on wrong variant (never silent)",
    },
    MatrixRow {
        op: "expect",
        guarantee: "Exact",
        fallibility: Fallibility::Partial,
        error_set: "on Err: refuses loudly with caller-supplied msg; never a silent default",
        effects: "none",
        explainable: Explainable::DiagnosticRefusal,
        never_silent_property:
            "EXPLICIT named partial with caller reason; refuses loudly on wrong variant",
    },
    MatrixRow {
        op: "unwrap_err",
        guarantee: "Exact",
        fallibility: Fallibility::Partial,
        error_set: "on Ok: refuses loudly (abort + diagnostic); never a silent default",
        effects: "none",
        explainable: Explainable::DiagnosticRefusal,
        never_silent_property:
            "symmetric partial: EXPLICIT named partial; refuses loudly on Ok variant",
    },
    // ---- RFC-0014 bridge (re-exported from std.recover — M-520) ----------------
    //
    // The concrete `Outcome`/`RecoverOutcome` and the recovery driver are owned by
    // `std.recover` (M-520, RFC-0014), now landed and re-exported at this crate's root. This
    // row records the bridge `std.error` surfaces. Tag is "Inherited-from-policy" — this module
    // never launders the policy's tag (VR-5 / I2).
    MatrixRow {
        op: "recover (RFC-0014 bridge — re-exported from std.recover, M-520)",
        guarantee: "Inherited-from-policy (Declared floor per I2/VR-5; see spec §7-Q1)",
        fallibility: Fallibility::Combinatorial,
        error_set: "Recovered(t,tag) | Propagated(e') — never a drop (I1); EffectBudgetExhausted on overrun (I4)",
        effects: "declared by the policy (retry/alloc/io/cascade); bounded (RFC-0014 I3/I4)",
        explainable: Explainable::PolicyRef,
        never_silent_property:
            "yields Recovered or Propagated; never a drop (I1); Ok passes through unrecovered",
    },
];

#[cfg(test)]
mod tests {
    use super::{Explainable, Fallibility, MATRIX};

    /// The spec §3 lists 19 exported surface rows (counting recover as one). Verify coverage.
    /// Guard: adding or removing a row without updating this test makes it fail.
    #[test]
    fn matrix_covers_all_spec_ops() {
        let expected_ops = [
            "map",
            "map_err",
            "and_then",
            "or_else",
            "filter",
            "inspect",
            "inspect_err",
            "ok_or",
            "ok_or_else",
            "ok",
            "transpose",
            "flatten",
            "zip",
            "unwrap_or",
            "unwrap_or_else",
            "unwrap_or_option",
            "unwrap_or_else_option",
            "propagate (?-style)",
            "unwrap",
            "expect",
            "unwrap_err",
        ];
        for name in &expected_ops {
            assert!(
                MATRIX
                    .iter()
                    .any(|r| r.op == *name || r.op.starts_with(name)),
                "matrix is missing op {name:?} (spec §3)"
            );
        }
        // The recover bridge must also be present.
        assert!(
            MATRIX.iter().any(|r| r.op.starts_with("recover")),
            "matrix must have the recover bridge row (spec §3)"
        );
        assert_eq!(
            MATRIX.len(),
            22,
            "expected 22 rows (21 core incl. the two Option unwrap_or variants + 1 recover bridge)"
        );
    }

    /// Every pure combinator row carries `Exact` (RFC-0016 C2 / VR-5).
    /// The `unwrap_or` family is `Declared` (I2). The recover bridge is `Inherited-...`.
    /// Guard: changing a pure combinator's tag to `Declared` or vice-versa makes this fail.
    #[test]
    fn tags_match_spec_lattice() {
        for row in MATRIX {
            match row.op {
                "unwrap_or" | "unwrap_or_else" | "unwrap_or_option" | "unwrap_or_else_option" => {
                    assert_eq!(
                        row.guarantee, "Declared",
                        "{} must be Declared (I2/VR-5 — fallback substitution)",
                        row.op
                    );
                }
                op if op.starts_with("recover") => {
                    assert!(
                        row.guarantee.starts_with("Inherited"),
                        "recover bridge tag must start with 'Inherited' (VR-5 — no launder); got {:?}",
                        row.guarantee
                    );
                }
                _ => {
                    assert_eq!(
                        row.guarantee, "Exact",
                        "{} must be Exact (pure combinator, RFC-0016 C2)",
                        row.op
                    );
                }
            }
        }
    }

    /// Partial ops have a non-empty error_set (C1 — explicit refusal description).
    /// Total/combinatorial pure-combinator rows may have empty error_set.
    /// Guard: making a partial op's error_set empty makes this fail.
    #[test]
    fn partial_ops_have_nonempty_error_set() {
        for row in MATRIX {
            if row.fallibility == Fallibility::Partial {
                assert!(
                    !row.error_set.is_empty(),
                    "partial op {:?} must name its explicit refusal (C1)",
                    row.op
                );
            }
        }
    }

    /// Exactly the `unwrap_or` family carries `Declared` (not `Exact`).
    /// Guard: adding `Declared` to a non-default-recovery op makes this fail.
    #[test]
    fn only_default_recovery_ops_are_declared() {
        let declared_ops: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.guarantee == "Declared")
            .map(|r| r.op)
            .collect();
        assert_eq!(
            declared_ops,
            [
                "unwrap_or",
                "unwrap_or_else",
                "unwrap_or_option",
                "unwrap_or_else_option"
            ],
            "only the unwrap_or family (incl. Option variants) should be Declared (I2/VR-5)"
        );
    }

    /// The `unwrap_or` family is marked `SubstitutedDefault` for EXPLAIN.
    /// Guard: changing the explainable field for unwrap_or makes this fail.
    #[test]
    fn unwrap_or_family_is_explain_substituted_default() {
        for row in MATRIX {
            if row.op == "unwrap_or" || row.op == "unwrap_or_else" {
                assert_eq!(
                    row.explainable,
                    Explainable::SubstitutedDefault,
                    "{} must be SubstitutedDefault for EXPLAIN (spec §4/§5)",
                    row.op
                );
            }
        }
    }

    /// The `ok` combinator is the only one with `LossyConversion` EXPLAIN (spec §7-Q2).
    /// Guard: adding LossyConversion to a non-ok op makes this fail.
    #[test]
    fn only_ok_is_lossy_conversion() {
        let lossy: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.explainable == Explainable::LossyConversion)
            .map(|r| r.op)
            .collect();
        assert_eq!(
            lossy,
            ["ok"],
            "only the 'ok' op is a flagged lossy conversion"
        );
    }

    /// The partial accessors must all be marked `DiagnosticRefusal` for EXPLAIN.
    /// Guard: changing a partial op's explainable field makes this fail.
    #[test]
    fn partial_ops_are_diagnostic_refusal() {
        for row in MATRIX {
            if row.fallibility == Fallibility::Partial {
                assert_eq!(
                    row.explainable,
                    Explainable::DiagnosticRefusal,
                    "partial op {:?} must be DiagnosticRefusal for EXPLAIN (spec §5/C3)",
                    row.op
                );
            }
        }
    }

    /// Every row has a non-empty `never_silent_property` (C1 — the structural guarantee
    /// that no combinator silently drops an error must be stated per row).
    /// Guard: leaving never_silent_property empty on any row makes this fail.
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

    /// The recover bridge row is present and has `PolicyRef` EXPLAIN.
    /// Guard: removing the recover bridge or changing its explainable makes this fail.
    #[test]
    fn recover_bridge_is_present_and_has_policy_ref() {
        let recover_row = MATRIX
            .iter()
            .find(|r| r.op.starts_with("recover"))
            .expect("recover bridge must be in the matrix (spec §3/§4)");
        assert_eq!(
            recover_row.explainable,
            Explainable::PolicyRef,
            "recover bridge must be PolicyRef for EXPLAIN (RFC-0014 §4.4)"
        );
        assert!(
            recover_row.error_set.contains("Recovered")
                && recover_row.error_set.contains("Propagated"),
            "recover bridge error_set must name both Recovered and Propagated outcomes (I1)"
        );
        // The error_set may say "never a drop" (which is correct — it explicitly rules it out).
        // What it must NOT say is something that implies a silent drop is permitted.
        // We check the string states the "never a drop" guarantee, not just any occurrence of "drop".
        assert!(
            recover_row.error_set.contains("never a drop"),
            "recover bridge error_set must explicitly state 'never a drop' (I1): {:?}",
            recover_row.error_set
        );
    }

    /// The `?`-propagate row elaborates to a Match (RFC-0014 §4.3).
    /// Guard: changing its explainable field makes this fail.
    #[test]
    fn propagate_is_match_elaboration() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "propagate (?-style)")
            .expect("propagate op must be in the matrix");
        assert_eq!(
            row.explainable,
            Explainable::MatchElaboration,
            "?-propagate must be MatchElaboration for EXPLAIN (RFC-0014 §4.3)"
        );
    }
}
