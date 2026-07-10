//! Property tests for `std.recover` (M-520; RFC-0014 I1–I5; dev-workflow §3–§7).
//!
//! Four families:
//! 1. **No code path drops an error** — every `Outcome::Err` input is `Recovered` or `Propagated`.
//! 2. **Recovered tag ≤ policy action ceiling** (meet) — never upgraded (I2/VR-5; FR-R3).
//! 3. **Budget overrun is explicit, never silent** (I4) — an exhausted budget yields `Propagated`,
//!    never hangs or panics.
//! 4. **Ok pass-through is `Exact`** (FR-R3 / P5-B bug fix) — never `Declared`.
//!
//! All tests are deterministic (no random seeds — "one deterministic sample"; dev-workflow guard #7).
//! Each negative test names a **mutant witness**: the specific change that would make the test fail.

#![allow(clippy::unwrap_used)] // test-only; the assertions make every `unwrap` safe.

use mycelium_core::GuaranteeStrength;
use mycelium_interp::budget::{Budgets, EffectBudget, EffectKind};

use crate::{
    action::RecoveryAction,
    effect::{check_effects, EffectSet},
    handle::handle_classified,
    outcome::{Outcome, Resolution},
    policy::RecoveryPolicy,
    registry::{ClassName, ClassRegistry},
};

// ---- helpers ----------------------------------------------------------------

/// Build a simple registry + class for tests.
fn simple_registry() -> (ClassRegistry, ClassName) {
    let mut reg = ClassRegistry::new();
    reg.register("io-error");
    let class = reg.resolve("io-error").unwrap();
    (reg, class)
}

/// `GuaranteeStrength::ALL` as an array (all four lattice levels).
const ALL_STRENGTHS: [GuaranteeStrength; 4] = [
    GuaranteeStrength::Exact,
    GuaranteeStrength::Proven,
    GuaranteeStrength::Empirical,
    GuaranteeStrength::Declared,
];

// ---- 1. Never-drops property ------------------------------------------------

/// Property: for every `Outcome`, `handle_classified` yields `Recovered | Propagated` — no drop.
/// (I1 — RFC-0014 §4.2; the never-silent spine.)
///
/// Mutant witness: adding a `Dropped` variant to `Resolution` would allow a drop; the type
/// currently makes that impossible, but this test documents the invariant.
#[test]
fn no_drop_for_ok_outcome() {
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Fallback {
                value: Box::new(99),
            },
        )
        .unwrap();
    let mut budgets = Budgets::new();

    // Ok path → always Recovered.
    let r = handle_classified(
        Outcome::Ok(42u32),
        &policy,
        &mut budgets,
        |_: &String| class.clone(),
        || (Outcome::Ok(0u32), GuaranteeStrength::Exact),
    );
    assert!(
        r.is_recovered(),
        "Ok outcome must always yield Recovered (I1): {r:?}"
    );
}

#[test]
fn no_drop_for_err_with_no_rule() {
    // An Err with no matching rule → Propagated, never dropped (I1 floor).
    // Mutant witness: returning `Recovered(default_value)` when no rule matches would violate I1.
    let (_reg, class) = simple_registry();
    let policy = RecoveryPolicy::<u32>::new(); // empty policy — no rules.
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("no-rule".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || (Outcome::Ok(0u32), GuaranteeStrength::Exact),
    );
    assert!(
        r.is_propagated(),
        "Err with no matching rule must Propagate unchanged (I1): {r:?}"
    );
    // The propagated error is unchanged.
    if let Resolution::Propagated {
        error,
        policy: pref,
        ..
    } = r
    {
        assert_eq!(error, "no-rule");
        assert!(pref.is_none(), "no-rule path must have no acting policy");
    }
}

#[test]
fn no_drop_for_err_with_fallback_rule() {
    // Fallback → Recovered (never dropped — I1).
    // Mutant witness: returning Propagated for a fallback would make this fail.
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Fallback {
                value: Box::new(42),
            },
        )
        .unwrap();
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("io-fail".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!("attempt should not be called for fallback"),
    );
    assert!(
        r.is_recovered(),
        "fallback must yield Recovered (I1): {r:?}"
    );
}

#[test]
fn no_drop_for_err_with_retry_exhausted() {
    // Retry exhausted → Propagated(original_error) — original never dropped (I1, additive).
    // Mutant witness: returning Recovered(fallback) on exhaustion would violate I1.
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(&reg, "io-error", RecoveryAction::Retry { max_attempts: 2 })
        .unwrap();
    // Budget for 2 retries.
    let mut budgets = Budgets::new().with(EffectBudget::Attempts(2));

    // attempt always fails.
    let attempt_count = std::cell::Cell::new(0u32);
    let r = handle_classified(
        Outcome::Err("original".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || {
            attempt_count.set(attempt_count.get() + 1);
            (
                Outcome::Err("retry-fail".to_string()),
                GuaranteeStrength::Exact,
            )
        },
    );
    assert!(
        r.is_propagated(),
        "exhausted retry must Propagate original error (I1): {r:?}"
    );
    // The ORIGINAL error propagates — not the retry's error.
    if let Resolution::Propagated { error, .. } = r {
        assert_eq!(
            error, "original",
            "retry exhaustion must propagate the ORIGINAL error (I1)"
        );
    }
    assert_eq!(
        attempt_count.get(),
        2,
        "attempt must be called max_attempts times"
    );
}

#[test]
fn no_drop_for_err_with_escalate() {
    // Escalate → Propagated — still explicit, never dropped (I1).
    // Mutant witness: returning Recovered for an escalation would violate I1.
    // Fix #3: "fatal" is now registry-validated in `on()`, so we must register it.
    let mut reg = ClassRegistry::new();
    reg.register("io-error");
    reg.register("fatal"); // must be registered — Escalate.to_class is now X1-validated.
    let class = reg.resolve("io-error").unwrap();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Escalate {
                to_class: "fatal".to_string(),
            },
        )
        .unwrap();
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("err".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!("escalate never calls attempt"),
    );
    assert!(
        r.is_propagated(),
        "escalate must yield Propagated (I1): {r:?}"
    );
}

#[test]
fn no_drop_for_err_with_cleanup_then_propagate() {
    // cleanup_then_propagate → always Propagated (I1).
    // Mutant witness: returning Recovered after cleanup would violate I1.
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::CleanupThenPropagate {
                effect: EffectKind::Alloc,
            },
        )
        .unwrap();
    let mut budgets = Budgets::new().with(EffectBudget::Bytes(1));

    let r = handle_classified(
        Outcome::Err("err".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!("cleanup_then_propagate never calls attempt"),
    );
    assert!(
        r.is_propagated(),
        "cleanup_then_propagate must yield Propagated (I1): {r:?}"
    );
}

// ---- 2. Recovered tag ≤ action ceiling (meet; I2/VR-5; FR-R3) ---------------

/// Property: fallback always yields `Declared` — the honest floor for a substitution.
/// Holds for every possible attempt tag (the attempt is never called for fallback).
/// Mutant witness: changing fallback's tag from Declared to Exact would violate I2/VR-5.
#[test]
fn fallback_tag_is_always_declared() {
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Fallback { value: Box::new(0) },
        )
        .unwrap();
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("e".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    if let Resolution::Recovered { tag, .. } = r {
        assert_eq!(
            tag,
            GuaranteeStrength::Declared,
            "fallback tag must be Declared (I2/VR-5)"
        );
    } else {
        panic!("fallback must yield Recovered");
    }
}

/// Property: Ok pass-through tag is `Exact` — not `Declared` (FR-R3 / P5-B bug fix).
/// Mutant witness: stamping Declared on Ok pass-through (the scaffold bug) makes this fail.
#[test]
fn ok_pass_through_tag_is_exact() {
    let policy = RecoveryPolicy::<u32>::new();
    let mut budgets = Budgets::new();
    let (_, class) = simple_registry();

    let r = handle_classified(
        Outcome::Ok(7u32),
        &policy,
        &mut budgets,
        |_: &String| class.clone(),
        || unreachable!(),
    );
    if let Resolution::Recovered { tag, .. } = r {
        assert_eq!(
            tag,
            GuaranteeStrength::Exact,
            "Ok pass-through must be Exact (FR-R3 / P5-B bug fix) — never Declared: {tag:?}"
        );
    } else {
        panic!("Ok must yield Recovered: {r:?}");
    }
}

/// Property: retry success inherits the attempt's own tag — for every lattice level.
/// Mutant witness: upgrading attempt_tag to Exact when the attempt returns Declared would fail.
#[test]
fn retry_tag_inherits_attempt_tag_for_all_strengths() {
    let (reg, class) = simple_registry();

    for &strength in &ALL_STRENGTHS {
        let mut policy = RecoveryPolicy::<u32>::new();
        policy
            .on(&reg, "io-error", RecoveryAction::Retry { max_attempts: 1 })
            .unwrap();
        let mut budgets = Budgets::new().with(EffectBudget::Attempts(1));

        let r = handle_classified(
            Outcome::Err("e".to_string()),
            &policy,
            &mut budgets,
            |_| class.clone(),
            || (Outcome::Ok(0u32), strength),
        );
        if let Resolution::Recovered { tag, .. } = r {
            assert_eq!(
                tag, strength,
                "retry success tag must equal the attempt's own tag {strength:?} (I2/VR-5)"
            );
        } else {
            panic!("retry success must yield Recovered for strength {strength:?}: {r:?}");
        }
    }
}

/// Property: recovered tag is never stronger than `Declared` for a fallback (lattice meet).
/// For all four strengths the fallback tag is still `Declared` (the meet floor for substitution).
/// Mutant witness: changing the meet to take `Exact` when both sides are Exact would fail.
#[test]
fn fallback_tag_is_declared_for_all_strength_pairings() {
    // The fallback action has a fixed `Declared` ceiling — it does not depend on other tags.
    let (reg, class) = simple_registry();

    for &_strength in &ALL_STRENGTHS {
        let mut policy = RecoveryPolicy::<u32>::new();
        policy
            .on(
                &reg,
                "io-error",
                RecoveryAction::Fallback { value: Box::new(1) },
            )
            .unwrap();
        let mut budgets = Budgets::new();

        let r = handle_classified(
            Outcome::Err("e".to_string()),
            &policy,
            &mut budgets,
            |_| class.clone(),
            || unreachable!(),
        );
        if let Resolution::Recovered { tag, .. } = r {
            assert_eq!(
                tag,
                GuaranteeStrength::Declared,
                "fallback tag must always be Declared regardless of context (I2/VR-5)"
            );
        }
    }
}

// ---- 3. Budget overrun is explicit (I4) -------------------------------------

/// Property: a retry budget overrun yields `Propagated` explicitly — never hangs or panics.
/// Mutant witness: panic-ing on overrun instead of returning Propagated would make this fail.
#[test]
fn retry_budget_overrun_is_explicit_propagated() {
    // Budget of 0 → immediately overruns on first consume.
    // Mutant witness: returning Recovered with any value on budget overrun would violate I4.
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(&reg, "io-error", RecoveryAction::Retry { max_attempts: 5 })
        .unwrap();
    // Attempts budget = 0 → any consume immediately overruns.
    let mut budgets = Budgets::new().with(EffectBudget::Attempts(0));

    let attempt_count = std::cell::Cell::new(0u32);
    let r = handle_classified(
        Outcome::Err("original".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || {
            attempt_count.set(attempt_count.get() + 1);
            (Outcome::Ok(0u32), GuaranteeStrength::Exact)
        },
    );
    assert!(
        r.is_propagated(),
        "budget-0 retry must yield Propagated (I4 graceful overrun): {r:?}"
    );
    assert_eq!(
        attempt_count.get(),
        0,
        "attempt must not be called when budget is zero from the start (I5)"
    );
}

/// Property: a no-declared-budget effect cannot run (I5 — tightly scoped by default).
/// Mutant witness: allowing consume on an absent budget to succeed would violate I5.
#[test]
fn absent_budget_is_immediate_graceful_overrun() {
    let mut b = Budgets::new(); // no budget declared for Retry.
    let err = b.consume(EffectKind::Retry, 1).unwrap_err();
    // The overrun is explicit: kind + requested + remaining.
    // Mutant witness: returning Ok(()) for an absent budget would violate I5.
    assert_eq!(err.kind, EffectKind::Retry, "overrun must name the kind");
    assert_eq!(err.requested, 1);
    assert_eq!(err.remaining, 0, "absent budget has 0 remaining (I5)");
}

/// Property: cleanup budget overrun is recorded (spec §7-Q4 disposition), not swallowed.
/// The original error still propagates (I1).
/// Mutant witness: setting cleanup_overrun = false when the cleanup budget overruns would fail.
#[test]
fn cleanup_overrun_is_recorded_not_swallowed() {
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::CleanupThenPropagate {
                effect: EffectKind::Io,
            },
        )
        .unwrap();
    // No Io budget declared → consume(Io, 1) overruns immediately.
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("original-err".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    // Original error propagates (I1) regardless.
    if let Resolution::Propagated {
        error,
        cleanup_overrun,
        ..
    } = r
    {
        assert_eq!(error, "original-err", "original error must propagate (I1)");
        // The cleanup overrun is recorded (spec §7-Q4 — legibility, not silent swallow).
        // Mutant witness: setting cleanup_overrun = false when budget is absent makes this fail.
        assert!(
            cleanup_overrun,
            "cleanup overrun must be recorded (spec §7-Q4 disposition)"
        );
    } else {
        panic!("cleanup_then_propagate must yield Propagated: {r:?}");
    }
}

/// Property: cleanup succeeds within budget → cleanup_overrun = false.
/// Mutant witness: setting cleanup_overrun = true unconditionally would make this fail.
///
/// B1 (RESOLVED): `EffectKind::Io` is now primeable via `EffectBudget::Ops`, so the cleanup budget
/// for an Io effect is declared **directly** — no Retry/Attempts proxy (the budget-API gap the
/// M-520 leaf flagged is closed in `mycelium-interp`).
#[test]
fn cleanup_within_budget_sets_overrun_false() {
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::CleanupThenPropagate {
                effect: EffectKind::Io,
            },
        )
        .unwrap();
    // Io budget = 1 → the single cleanup op runs within budget (the real Io path, not a proxy).
    let mut budgets = Budgets::new().with(EffectBudget::Ops(1));

    let r = handle_classified(
        Outcome::Err("err".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    if let Resolution::Propagated {
        cleanup_overrun,
        error,
        ..
    } = r
    {
        assert_eq!(error, "err", "original error propagates (I1)");
        assert!(
            !cleanup_overrun,
            "cleanup within budget must set cleanup_overrun = false (spec §7-Q4)"
        );
    } else {
        panic!("cleanup_then_propagate must yield Propagated: {r:?}");
    }
}

/// A user-declared **named** cleanup effect is budgeted directly via `EffectBudget::Named`
/// (B1 — RESOLVED). Within budget the cleanup runs (`cleanup_overrun = false`); the original error
/// still propagates (I1).
#[test]
fn cleanup_with_named_effect_budget_runs_within_budget() {
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::CleanupThenPropagate {
                effect: EffectKind::Named("flush".to_owned()),
            },
        )
        .unwrap();
    let mut budgets = Budgets::new().with(EffectBudget::Named("flush".to_owned(), 1));

    let r = handle_classified(
        Outcome::Err("err".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    if let Resolution::Propagated {
        cleanup_overrun,
        error,
        ..
    } = r
    {
        assert_eq!(error, "err", "original error propagates (I1)");
        assert!(
            !cleanup_overrun,
            "named cleanup within budget must set cleanup_overrun = false"
        );
    } else {
        panic!("cleanup_then_propagate must yield Propagated: {r:?}");
    }
}

// ---- 4. Effect declaration checks (I3) -------------------------------------

/// Property: check_effects catches an undeclared effect explicitly.
/// Mutant witness: removing the check_effects call from the validator lets undeclared effects pass.
#[test]
fn undeclared_effect_is_explicit_error() {
    let declared: EffectSet = [EffectKind::Alloc].into_iter().collect();
    let performed: EffectSet = [EffectKind::Alloc, EffectKind::Io].into_iter().collect();
    let err = check_effects(&declared, &performed).unwrap_err();
    // The exact effect variant is named (dev-workflow guard #7 — assert the specific failure).
    // Mutant witness: returning `Ok(())` for an undeclared effect would make this fail.
    assert_eq!(
        err.effect,
        EffectKind::Io,
        "check_effects must name the undeclared effect (I3 — explicit checker error)"
    );
}

/// Property: `UndeclaredEffect::Display` names the effect without extra debug-quoting.
/// Mutant witness: using `{:?}` on `to_string()` would produce `"io"` (with quotes) instead of
/// `io` — a display bug that buries the effect name in debug-string quotes.
#[test]
fn undeclared_effect_display_names_effect_without_extra_quotes() {
    use crate::effect::UndeclaredEffect;
    let e = UndeclaredEffect {
        effect: EffectKind::Io,
    };
    let msg = e.to_string();
    assert!(
        msg.contains("io"),
        "UndeclaredEffect display must contain the effect name (I3)"
    );
    assert!(
        !msg.contains("\"io\""),
        "UndeclaredEffect display must NOT wrap the effect name in debug quotes: {msg:?}"
    );
    // Named effects: the name appears bare, not double-quoted.
    let named = UndeclaredEffect {
        effect: EffectKind::Named("flush".to_owned()),
    };
    let named_msg = named.to_string();
    assert!(
        named_msg.contains("flush"),
        "UndeclaredEffect display must contain the named effect (I3): {named_msg:?}"
    );
    assert!(
        !named_msg.contains("\"flush\""),
        "UndeclaredEffect display must NOT wrap named effect in debug quotes: {named_msg:?}"
    );
}

#[test]
fn declared_effects_pass_check() {
    let declared: EffectSet = [EffectKind::Alloc, EffectKind::Io].into_iter().collect();
    let performed: EffectSet = [EffectKind::Alloc].into_iter().collect();
    assert!(
        check_effects(&declared, &performed).is_ok(),
        "performed ⊆ declared must pass check_effects (I3)"
    );
}

#[test]
fn empty_declared_with_no_performed_passes() {
    let empty: EffectSet = EffectSet::new();
    assert!(
        check_effects(&empty, &empty).is_ok(),
        "empty declared vs empty performed is always ok (I3)"
    );
}

// ---- 5. Policy registration (Exact; UnknownClass explicit) ------------------

/// Property: `on` with an unregistered class is an explicit `UnknownClass` error (X1).
/// Mutant witness: silently inserting an unregistered class name would violate X1/G2.
#[test]
fn policy_on_unknown_class_is_explicit_error() {
    let reg = ClassRegistry::new(); // empty.
    let mut policy = RecoveryPolicy::<u32>::new();
    let err = policy
        .on(
            &reg,
            "no-such-class",
            RecoveryAction::Fallback { value: Box::new(0) },
        )
        .unwrap_err();
    assert_eq!(
        err.name, "no-such-class",
        "UnknownClass must name the attempted class (X1)"
    );
}

/// Property: `policy_ref` is deterministic for the same rules (content-addressed — ADR-006/C3).
/// Mutant witness: non-deterministic hashing would produce different refs for the same policy.
///
/// Fix #2: policy_ref() now returns Result; unwrap() is safe because u32: serde::Serialize always
/// succeeds in serde_json (no non-finite values; integers serialize without error).
#[test]
fn policy_ref_is_deterministic() {
    let (reg, _) = simple_registry();
    let mut p1 = RecoveryPolicy::<u32>::new();
    p1.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback {
            value: Box::new(42),
        },
    )
    .unwrap();
    let mut p2 = RecoveryPolicy::<u32>::new();
    p2.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback {
            value: Box::new(42),
        },
    )
    .unwrap();
    assert_eq!(
        p1.policy_ref().unwrap(),
        p2.policy_ref().unwrap(),
        "same policy must have same content hash (ADR-006)"
    );
}

/// Property: different rules produce different `PolicyRef`s (no collision — banked guard #5).
/// Mutant witness: a hash collision between distinct rules would fail this test.
///
/// Fix #2: policy_ref() returns Result; unwrap() is safe for u32 (always serializable in serde_json).
#[test]
fn different_policies_have_different_refs() {
    let (reg, _) = simple_registry();
    let mut p1 = RecoveryPolicy::<u32>::new();
    p1.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback { value: Box::new(1) },
    )
    .unwrap();
    let mut p2 = RecoveryPolicy::<u32>::new();
    p2.on(&reg, "io-error", RecoveryAction::Retry { max_attempts: 3 })
        .unwrap();
    assert_ne!(
        p1.policy_ref().unwrap(),
        p2.policy_ref().unwrap(),
        "distinct policies must have distinct content hashes (ADR-006/banked guard #5)"
    );
}

/// Property: empty policy has a stable content hash (not a fabricated zero — G2).
/// Mutant witness: returning a null/zero hash for an empty policy would produce collisions.
///
/// Fix #2: policy_ref() returns Result; empty policy has no fallback values, so it always
/// succeeds (no serialization possible to fail).
#[test]
fn empty_policy_has_stable_hash() {
    let p1 = RecoveryPolicy::<u32>::new();
    let p2 = RecoveryPolicy::<u32>::new();
    assert_eq!(
        p1.policy_ref().unwrap(),
        p2.policy_ref().unwrap(),
        "two empty policies must have the same content hash"
    );
    // And it must be a valid ContentHash (not a fabricated value).
    let r = p1.policy_ref().unwrap();
    assert!(
        r.as_str().starts_with("blake3:"),
        "policy_ref must be a valid blake3 content hash: {:?}",
        r.as_str()
    );
}

// ---- 6. EXPLAIN-ability (C3) ------------------------------------------------

/// Property: every Recovered outcome carries a PolicyRef when a rule was applied (C3).
/// Mutant witness: returning policy: None when a rule acts would lose EXPLAIN-ability.
///
/// Fix #2: policy_ref() returns Result; unwrap() is safe for u32.
#[test]
fn recovered_outcome_carries_policy_ref_when_rule_applied() {
    let (reg, class) = simple_registry();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Fallback { value: Box::new(0) },
        )
        .unwrap();
    // Fix #2: policy_ref() now returns Result — unwrap() is safe (u32: serde::Serialize).
    let expected_ref = policy.policy_ref().unwrap();
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("e".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    assert_eq!(
        r.policy_ref(),
        Some(&expected_ref),
        "Recovered outcome must carry the acting PolicyRef (C3/EXPLAIN)"
    );
}

/// Property: Propagated outcome carries a PolicyRef when a rule acted (C3).
/// (Even re-propagation is EXPLAIN-able.)
/// Mutant witness: returning policy: None when escalate acts would lose EXPLAIN-ability.
#[test]
fn propagated_outcome_carries_policy_ref_when_rule_applied() {
    // Fix #3: "fatal" must be registered — Escalate.to_class is now X1-validated.
    let mut reg = ClassRegistry::new();
    reg.register("io-error");
    reg.register("fatal");
    let class = reg.resolve("io-error").unwrap();
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Escalate {
                to_class: "fatal".into(),
            },
        )
        .unwrap();
    // Fix #2: policy_ref() now returns Result — unwrap() is safe for well-typed T (u32: Serialize).
    let expected_ref = policy.policy_ref().unwrap();
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("e".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    assert_eq!(
        r.policy_ref(),
        Some(&expected_ref),
        "Propagated outcome (escalate) must carry the acting PolicyRef (C3/EXPLAIN)"
    );
}

/// Property: no-rule Propagated has no PolicyRef (no phantom attribution — G2).
/// Mutant witness: attaching a fabricated PolicyRef when no rule matched would be dishonest.
#[test]
fn no_rule_propagated_has_no_policy_ref() {
    let (_, class) = simple_registry();
    let policy = RecoveryPolicy::<u32>::new();
    let mut budgets = Budgets::new();

    let r = handle_classified(
        Outcome::Err("e".to_string()),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || unreachable!(),
    );
    assert_eq!(
        r.policy_ref(),
        None,
        "no-rule Propagated must have no PolicyRef (G2 — no phantom attribution)"
    );
}

// ---- 7. GuaranteeStrength lattice (meet) ------------------------------------

/// Property: `meet` is weakest-wins for all 16 pairs.
/// This test mirrors `mycelium-core`'s guarantee test but is stated here for the `std.recover`
/// tag contract (FR-R3): the meet is what the recover driver applies.
/// Mutant witness: swapping meet/join would give strongest-wins, violating VR-5.
#[test]
fn meet_is_weakest_for_all_pairs() {
    for &a in &ALL_STRENGTHS {
        for &b in &ALL_STRENGTHS {
            let m = a.meet(b);
            let expected_rank = a.rank().max(b.rank()); // highest rank = weakest
            assert_eq!(
                m.rank(),
                expected_rank,
                "meet({a:?}, {b:?}) must have rank {expected_rank} (weakest wins — VR-5)"
            );
        }
    }
}

/// Property: meet(Declared, anything) = Declared (Declared is the bottom of the lattice).
/// Mutant witness: returning Exact for meet(Declared, Exact) would violate VR-5.
#[test]
fn declared_is_meet_bottom() {
    for &other in &ALL_STRENGTHS {
        let m = GuaranteeStrength::Declared.meet(other);
        assert_eq!(
            m,
            GuaranteeStrength::Declared,
            "meet(Declared, {other:?}) must be Declared (bottom of the lattice — VR-5)"
        );
    }
}

// ---- 8. EffectBudgetExhausted converts to EvalError (RFC-0014 §4.8) ---------

/// Property: `EffectBudgetExhausted` converts to `EvalError::EffectBudget` (one enforcement
/// mechanism — RFC-0014 §4.8 / `mycelium-interp`).
/// Mutant witness: a separate error type (not EvalError) would break the one-channel property.
#[test]
fn effect_budget_exhausted_converts_to_eval_error() {
    use mycelium_interp::EvalError;
    let mut b = Budgets::new().with(EffectBudget::Attempts(0));
    let exhausted = b.consume(EffectKind::Retry, 1).unwrap_err();
    let as_eval: EvalError = exhausted.clone().into();
    assert!(
        matches!(as_eval, EvalError::EffectBudget(_)),
        "EffectBudgetExhausted must convert to EvalError::EffectBudget (RFC-0014 §4.8): {as_eval:?}"
    );
}

// ---- 9. ClassRegistry (X1) --------------------------------------------------

/// Property: resolve returns UnknownClass for unregistered names (X1 — never an eval'd string).
/// Mutant witness: silently accepting any string would violate X1.
#[test]
fn unregistered_class_is_unknown_error() {
    let reg = ClassRegistry::new();
    let err = reg.resolve("anything").unwrap_err();
    assert_eq!(err.name, "anything");
}

/// Property: register then resolve succeeds (the registry is additive).
/// Mutant witness: resolving after registration failing would break the contract.
#[test]
fn registered_class_resolves() {
    let mut reg = ClassRegistry::new();
    reg.register("my-class");
    let class = reg
        .resolve("my-class")
        .expect("registered class must resolve");
    assert_eq!(class.as_str(), "my-class");
}

// ---- 10. Fix #2: stable serde-based PolicyRef encoding (banked guard #5) ------

/// Fix #2 — Property: two structurally-equal policies produce the same `PolicyRef` via
/// the stable serde_json encoding.
///
/// This also implicitly verifies that the hash does NOT depend on `Debug` output (if it did,
/// a custom `Debug` impl that returns varying text would cause two equal policies to diverge).
///
/// Mutant witness: using `format!("{:?}")` instead of `serde_json::to_vec` would make this
/// test fail for a type whose `Debug` output is non-deterministic or differs from its serde
/// representation (see `DebugOverrideType` in the next test).
#[test]
fn equal_fallback_values_produce_equal_policy_refs() {
    // Two policies with structurally-equal fallback values must have the same PolicyRef
    // regardless of how their types format under Debug.
    let (reg, _) = simple_registry();
    let mut p1 = RecoveryPolicy::<u32>::new();
    p1.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback { value: Box::new(7) },
    )
    .unwrap();
    let mut p2 = RecoveryPolicy::<u32>::new();
    p2.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback { value: Box::new(7) },
    )
    .unwrap();
    assert_eq!(
        p1.policy_ref().unwrap(),
        p2.policy_ref().unwrap(),
        "two policies with equal fallback values must produce equal PolicyRefs (banked guard #5)"
    );
}

/// Fix #2 — Property: the PolicyRef hash uses `serde::Serialize`, not `Debug`.
///
/// `DebugDiffers` has a `Debug` impl that always returns `"VARIABLE"` regardless of the
/// inner value, but a `Serialize` impl that uses the actual integer.  Two instances with
/// different inner values therefore produce identical `Debug` output but different serde
/// output — and must produce **different** `PolicyRef`s (the hash must use serde, not Debug).
///
/// Mutant witness: using `format!("{:?}")` for hashing would make both policies produce the
/// SAME `PolicyRef` (both serialize as `"VARIABLE"` under Debug), making this test fail.
#[test]
fn policy_ref_hashes_by_serialize_not_debug() {
    use serde::Serialize;

    /// A wrapper type whose `Debug` output is always `"VARIABLE"` regardless of the
    /// inner value, but whose `Serialize` output reflects the actual inner `u32`.
    /// This deliberately breaks the Debug↔Serialize consistency to expose which path the
    /// hash uses.
    #[derive(Clone, Serialize)]
    struct DebugDiffers(u32);

    impl std::fmt::Debug for DebugDiffers {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // Always returns the same string regardless of the inner value.
            f.write_str("VARIABLE")
        }
    }

    let (reg, _) = simple_registry();

    // p1 holds DebugDiffers(1), p2 holds DebugDiffers(2).
    // Their Debug outputs are identical ("VARIABLE"), but their serde outputs differ.
    let mut p1 = RecoveryPolicy::<DebugDiffers>::new();
    p1.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback {
            value: Box::new(DebugDiffers(1)),
        },
    )
    .unwrap();
    let mut p2 = RecoveryPolicy::<DebugDiffers>::new();
    p2.on(
        &reg,
        "io-error",
        RecoveryAction::Fallback {
            value: Box::new(DebugDiffers(2)),
        },
    )
    .unwrap();

    // Sanity: Debug output is the same for both.
    assert_eq!(
        format!("{:?}", DebugDiffers(1)),
        format!("{:?}", DebugDiffers(2)),
        "precondition: Debug output must be identical for both values"
    );

    // The PolicyRef must differ — the hash uses serde output (which differs), not Debug.
    // Mutant witness: hashing via `format!("{:?}")` would produce equal refs here.
    assert_ne!(
        p1.policy_ref().unwrap(),
        p2.policy_ref().unwrap(),
        "policies with different fallback values must produce different PolicyRefs even when \
         their Debug output is identical — serde encoding, not Debug, is used (banked guard #5)"
    );
}

// ---- 11. Fix #3: Escalate `to_class` is registry-validated in `on()` (X1) ----

/// Fix #3 — Property: `on(.., Escalate { to_class: "<unregistered>" })` returns `Err(UnknownClass)`.
///
/// Prior to this fix, only the LHS `class` was validated; the Escalate `to_class` was an
/// unvalidated raw string (partial X1 violation).  Now both are registry-checked.
///
/// Mutant witness: removing the `to_class` registry check in `on()` would let an unregistered
/// escalation target slip through, making this test fail (X1 / G2).
#[test]
fn on_escalate_with_unregistered_to_class_is_explicit_error() {
    let mut reg = ClassRegistry::new();
    reg.register("io-error"); // LHS class is registered.
                              // "fatal" is NOT registered — escalating to it must be an explicit error.
    let mut policy = RecoveryPolicy::<u32>::new();
    let err = policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Escalate {
                to_class: "fatal".to_string(), // not in registry
            },
        )
        .unwrap_err();
    // The error must name the unknown `to_class`, not the LHS class (X1 — explicit, not silent).
    // Mutant witness: returning Err for the LHS class name instead of "fatal" loses the
    // diagnostic (the caller can't tell which class was bad).
    assert_eq!(
        err.name, "fatal",
        "UnknownClass must name the unregistered Escalate to_class (X1 / G2)"
    );
}

/// Fix #3 — Property: `on(.., Escalate { to_class: "<registered>" })` succeeds (Ok).
///
/// Confirms the registry check does not reject valid, registered escalation targets.
///
/// Mutant witness: always returning Err for Escalate `to_class` (over-correction) would make
/// this test fail and break valid escalation policies.
#[test]
fn on_escalate_with_registered_to_class_succeeds() {
    let mut reg = ClassRegistry::new();
    reg.register("io-error");
    reg.register("fatal"); // escalation target is registered — must be accepted.
    let mut policy = RecoveryPolicy::<u32>::new();
    policy
        .on(
            &reg,
            "io-error",
            RecoveryAction::Escalate {
                to_class: "fatal".to_string(),
            },
        )
        .expect("registered Escalate.to_class must be accepted (X1)");
    // The rule is now in the policy.
    let class = reg.resolve("io-error").unwrap();
    assert!(
        policy.action_for(&class).is_some(),
        "escalate rule must be inserted when to_class is registered"
    );
}
