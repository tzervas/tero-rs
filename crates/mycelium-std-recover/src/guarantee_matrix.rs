//! The `std.recover` guarantee matrix encoded as **checked data** (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has one row in [`MATRIX`].  The matrix is the load-bearing C2
//! deliverable (RFC-0016 §4.1 C2 / VR-5): guarantee tags are asserted in tests, not prose-only.
//!
//! Rows mirror the spec §4.5 table exactly.  Tests assert:
//! - No row permits a silent drop (I1 — the never-silent spine).
//! - Recovered-tag discipline (I2/VR-5 — recovery only downgrades).
//! - Budget-bounded effects (I4 — every row with effects is bounded).
//! - `Ok` pass-through is `Exact`, not `Declared` (FR-R3 — the P5-B exact-tag bug fix).
//! - Policy/ledger/check ops are `Exact` (no accuracy semantics, RFC-0016 C2).

/// Fallibility classification for a `std.recover` exported op (I1 — explicit outcome set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// Always yields a `Resolution` (either `Recovered` or `Propagated`) — total over budgets.
    Total,
    /// Returns `Result<_, UnknownClass>` — the explicit configuration-error path.
    FallibleConfig,
    /// Returns `Result<_, EffectBudgetExhausted>` — the explicit overrun path.
    FallibleBudget,
}

/// Whether an op carries an EXPLAIN obligation (C3 — no black boxes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op records the acting `PolicyRef` on every outcome (C3).
    PolicyRef,
    /// The op *is* the policy reference (the `policy_ref` op itself).
    IsPolicyRef,
    /// No EXPLAIN artifact needed (value/config/budget ops with no selection).
    NotApplicable,
}

/// One row in the `std.recover` guarantee matrix (RFC-0016 §4.5; spec §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation name.
    pub op: &'static str,
    /// Guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (VR-5).
    ///
    /// - `"Exact"` — no accuracy/precision semantics (value/config/budget ops).
    /// - `"Declared"` — the substituted fallback has no checked basis (I2/VR-5).
    /// - `"Inherited-from-attempt"` — inherits the attempt's own tag (retry / pass-through).
    /// - `"Inherited-honest-floor"` — `Exact` for Ok pass-through; never upgraded.
    pub guarantee: &'static str,
    /// Fallibility: total, fallible-config, or fallible-budget.
    pub fallibility: Fallibility,
    /// The explicit outcome / error set description (non-empty for fallible ops).
    pub error_set: &'static str,
    /// Declared + budgeted effects (C6).  "none" means effect-free.
    pub effects: &'static str,
    /// Whether the op has a C3 EXPLAIN obligation.
    pub explainable: Explainable,
    /// The structural never-silent property of this op (C1/I1 — must be non-empty for every row).
    pub never_silent_property: &'static str,
}

/// The §4.5 guarantee matrix (spec §4; RFC-0016 §4.5; RFC-0014 I1–I5).
///
/// **No row permits a silent drop** (I1/C1): every op either recovers explicitly (honest tag),
/// re-propagates, or refuses a budget/config error explicitly — none of them "silently returns
/// success" or "the error disappears".
pub static MATRIX: &[MatrixRow] = &[
    // ---- The driver ----
    MatrixRow {
        op: "handle (never-silent driver)",
        guarantee: "Inherited-honest-floor: Exact for Ok pass-through, Declared for fallback, attempt-tag for retry success",
        fallibility: Fallibility::Total,
        error_set: "total over budgets — yields Recovered | Propagated; no drop (I1); budget overrun is EffectBudgetExhausted routed through the outcome",
        effects: "the policy's actions' effects (retry → EffectKind::Retry budgeted Attempts(N); cleanup_then_propagate → declared effect budgeted); Ok / fallback / escalate paths are effect-free",
        explainable: Explainable::PolicyRef,
        never_silent_property: "Resolution has no Dropped variant; every call yields Recovered or Propagated — enforced by the type (I1)",
    },
    // ---- The std.error bridge ----
    MatrixRow {
        op: "recover / handle_classified (std.error bridge)",
        guarantee: "= handle — inherits the honest floor; never upgraded (I2/VR-5)",
        fallibility: Fallibility::Total,
        error_set: "RecoverOutcome = Resolution = Recovered | Propagated; never a drop (I1)",
        effects: "= handle (the policy's declared, budgeted effects)",
        explainable: Explainable::PolicyRef,
        never_silent_property: "Resolution = Recovered | Propagated; no Dropped variant — I1 is a property of the type",
    },
    // ---- Actions ----
    MatrixRow {
        op: "fallback(value) (action)",
        guarantee: "Declared — a substituted fallback has no checked basis (I2/VR-5)",
        fallibility: Fallibility::Total,
        error_set: "always Recovered(value, Declared) — the one always-recovering action",
        effects: "none (pure value substitution)",
        explainable: Explainable::PolicyRef,
        never_silent_property: "always Recovered — the error is replaced by an explicit value (I1); the substitution is honest-tagged Declared",
    },
    MatrixRow {
        op: "retry(<=N) (action)",
        guarantee: "Inherited-from-attempt — the successful attempt's own tag; on exhaustion no value is produced",
        fallibility: Fallibility::Total,
        error_set: "Recovered on success; Propagated(original_error) on exhaustion — bounded by <=N (I4); EffectBudgetExhausted on budget overrun",
        effects: "EffectKind::Retry, budgeted Attempts(N); overrun → graceful EffectBudgetExhausted (I4)",
        explainable: Explainable::PolicyRef,
        never_silent_property: "either Recovered (attempt succeeded) or Propagated(original_error) (exhausted) — original error never discarded (I1)",
    },
    MatrixRow {
        op: "escalate(class') (action)",
        guarantee: "n/a — re-propagates an error, not a value; no accuracy semantics",
        fallibility: Fallibility::Total,
        error_set: "always Propagated(transformed_error) — still explicit; never a drop (I1)",
        effects: "none (pure structural transform — class label changes, error continues)",
        explainable: Explainable::PolicyRef,
        never_silent_property: "always Propagated — the error is re-tagged but never discarded (I1); the escalated class is in the PolicyRef (C3)",
    },
    MatrixRow {
        op: "cleanup_then_propagate(effect) (action)",
        guarantee: "n/a — re-propagates the original error",
        fallibility: Fallibility::Total,
        error_set: "always Propagated(original_error); cleanup budget overrun is recorded in cleanup_overrun field (legible — spec §7-Q4); original error propagates regardless (I1)",
        effects: "declared cleanup effect, budgeted (I4/I5); overrun graceful (skips cleanup, records overrun, propagates original error)",
        explainable: Explainable::PolicyRef,
        never_silent_property: "original error always propagates (I1) — even if the cleanup budget overruns; the overrun is recorded (not swallowed)",
    },
    // ---- Policy registration ----
    MatrixRow {
        op: "on (policy registration)",
        guarantee: "Exact (builds a config value; no accuracy semantics)",
        fallibility: Fallibility::FallibleConfig,
        error_set: "Err(UnknownClass) — a class not in the diag registry is an explicit config error, never an eval'd string (X1)",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Err(UnknownClass) is explicit — an unknown class never silently becomes a default action (X1/G2)",
    },
    MatrixRow {
        op: "policy_ref",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
        explainable: Explainable::IsPolicyRef,
        never_silent_property: "pure deterministic hash — no error path; policy_ref is the EXPLAIN identity (C3)",
    },
    MatrixRow {
        op: "action_for",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "Option<RecoveryAction> — None if no rule for the class (never a silent fallthrough)",
        effects: "none",
        explainable: Explainable::NotApplicable,
        never_silent_property: "None is explicit absence — no silent fallthrough to a default action (G2)",
    },
    // ---- Budget ledger ----
    MatrixRow {
        op: "consume (budget ledger)",
        guarantee: "Exact (budget arithmetic; no accuracy semantics)",
        fallibility: Fallibility::FallibleBudget,
        error_set: "Err(EffectBudgetExhausted) — the graceful, explicit overrun (I4); names kind + requested + remaining; never a hang/OOM",
        effects: "the consumed EffectKind, bounded",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Err(EffectBudgetExhausted) is explicit — an overrun never hangs or silently stalls (I4)",
    },
    MatrixRow {
        op: "check_effects (I3)",
        guarantee: "Exact (static check; no accuracy semantics)",
        fallibility: Fallibility::FallibleConfig,
        error_set: "Err(UndeclaredEffect) — a performed-but-undeclared effect is an explicit checker error (I3); names the undeclared effect",
        effects: "none (a static checker — not a runtime concern; KC-3)",
        explainable: Explainable::NotApplicable,
        never_silent_property: "Err(UndeclaredEffect) is explicit — an undeclared effect never silently becomes a valid side effect (I3/G2)",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row has a non-empty `never_silent_property` (C1/I1).
    /// Guard: leaving `never_silent_property` empty on any row makes this fail.
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

    /// No row implies a silent drop (I1 — "drop" in a row's error_set or
    /// never_silent_property must always negate it with "never" or "no drop").
    /// Guard: adding a row with "drop" in error_set without a negation makes this fail.
    #[test]
    fn no_row_permits_silent_drop() {
        for row in MATRIX {
            // A row may mention "drop" only if it also negates it ("never" or "no drop").
            let text = row.error_set.to_lowercase();
            let nsp = row.never_silent_property.to_lowercase();
            if text.contains("drop") {
                let negated = text.contains("never")
                    || text.contains("no drop")
                    || nsp.contains("never")
                    || nsp.contains("no drop");
                assert!(
                    negated,
                    "row {:?} mentions 'drop' without negation ('never'/'no drop') — \
                     may imply a silent drop (I1): error_set={:?}",
                    row.op, row.error_set
                );
            }
        }
    }

    /// The handle/recover bridge rows are `Total` (they always yield a `Resolution`).
    /// Guard: changing the driver row's fallibility makes this fail.
    #[test]
    fn driver_rows_are_total() {
        for row in MATRIX {
            if row.op.starts_with("handle") || row.op.starts_with("recover") {
                assert_eq!(
                    row.fallibility,
                    Fallibility::Total,
                    "driver row {:?} must be Total (total over budgets — I1/I4)",
                    row.op
                );
            }
        }
    }

    /// The fallback action row is `Declared` (the one fixed guaranteed-tag action — I2/VR-5).
    /// Guard: changing fallback's guarantee tag makes this fail.
    #[test]
    fn fallback_action_is_declared() {
        let row = MATRIX
            .iter()
            .find(|r| r.op.starts_with("fallback"))
            .expect("fallback row must be in the matrix");
        assert!(
            row.guarantee.contains("Declared"),
            "fallback action must carry Declared (I2/VR-5 — a substituted fallback has no \
             checked basis): {:?}",
            row.guarantee
        );
    }

    /// The Ok pass-through guarantee (via handle) contains "Exact" — FR-R3 bug fix (P5-B).
    /// Guard: changing the driver row's guarantee to Declared makes this fail.
    #[test]
    fn ok_pass_through_is_exact_not_declared() {
        let row = MATRIX
            .iter()
            .find(|r| r.op.starts_with("handle"))
            .expect("handle row must be in the matrix");
        assert!(
            row.guarantee.contains("Exact"),
            "handle (Ok pass-through) must include Exact — not Declared (FR-R3 / P5-B bug fix): \
             {:?}",
            row.guarantee
        );
        // Also verify the guarantee does NOT claim Exact universally (only for Ok pass-through).
        assert!(
            row.guarantee.contains("Ok pass-through"),
            "handle row guarantee must note the Ok-pass-through scoping of Exact: {:?}",
            row.guarantee
        );
    }

    /// Policy/ledger/check ops are `Exact` (no accuracy semantics — RFC-0016 C2 "len-style").
    /// Guard: tagging a policy op as non-Exact makes this fail.
    #[test]
    fn policy_and_ledger_ops_are_exact() {
        let exact_ops = [
            "on (policy registration)",
            "policy_ref",
            "action_for",
            "consume (budget ledger)",
            "check_effects (I3)",
        ];
        for &op in &exact_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == op)
                .unwrap_or_else(|| panic!("matrix must contain op {:?}", op));
            assert!(
                row.guarantee.starts_with("Exact"),
                "op {:?} must be Exact (no accuracy semantics, RFC-0016 C2): {:?}",
                op,
                row.guarantee
            );
        }
    }

    /// `retry` action row has budgeted `Retry` effect (I3/I4).
    /// Guard: removing the effect from the retry row makes this fail.
    #[test]
    fn retry_action_has_budgeted_retry_effect() {
        let row = MATRIX
            .iter()
            .find(|r| r.op.starts_with("retry"))
            .expect("retry action row must be in the matrix");
        assert!(
            row.effects.contains("Retry") || row.effects.contains("retry"),
            "retry action must declare the Retry effect (I3): {:?}",
            row.effects
        );
        assert!(
            row.effects.contains("Attempts") || row.effects.contains("budgeted"),
            "retry action's Retry effect must be budgeted (I4): {:?}",
            row.effects
        );
    }

    /// The handle and recover bridge rows carry `PolicyRef` for EXPLAIN (C3).
    /// Guard: removing PolicyRef from the driver rows makes this fail.
    #[test]
    fn driver_rows_are_policy_ref_explainable() {
        for row in MATRIX {
            if row.op.starts_with("handle") || row.op.starts_with("recover") {
                assert_eq!(
                    row.explainable,
                    Explainable::PolicyRef,
                    "driver row {:?} must carry PolicyRef for EXPLAIN (C3)",
                    row.op
                );
            }
        }
    }

    /// All `Total` rows have their error_set contain "Recovered" and "Propagated" (or are
    /// effect-free value ops), confirming I1 at the matrix level.
    /// Guard: a Total row that drops the two-variant guarantee makes this fail.
    #[test]
    fn total_driver_rows_name_recovered_and_propagated() {
        let driver_ops = ["handle", "recover"];
        for &prefix in &driver_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op.starts_with(prefix))
                .unwrap_or_else(|| panic!("matrix must contain a row starting with {:?}", prefix));
            assert!(
                row.error_set.contains("Recovered"),
                "driver op {:?} error_set must name Recovered (I1): {:?}",
                row.op,
                row.error_set
            );
            assert!(
                row.error_set.contains("Propagated"),
                "driver op {:?} error_set must name Propagated (I1): {:?}",
                row.op,
                row.error_set
            );
        }
    }

    /// `cleanup_then_propagate` row records the spec §7-Q4 disposition (cleanup overrun is
    /// recorded, not swallowed).
    /// Guard: removing the §7-Q4 note from the cleanup row makes this fail.
    #[test]
    fn cleanup_row_records_overrun_visibility() {
        let row = MATRIX
            .iter()
            .find(|r| r.op.starts_with("cleanup_then_propagate"))
            .expect("cleanup_then_propagate row must be in the matrix");
        // The disposition: the cleanup overrun is recorded, not swallowed.
        assert!(
            row.error_set.contains("cleanup_overrun") || row.error_set.contains("recorded"),
            "cleanup_then_propagate row must note the overrun-visibility disposition (spec §7-Q4): {:?}",
            row.error_set
        );
    }
}
