//! The **never-silent driver** (RFC-0014 §4.2/§4.3; RFC-0016 C1).
//!
//! [`handle_classified`] is the spine of `std.recover`: it maps an [`Outcome`] through a
//! [`RecoveryPolicy`] and always yields a [`Resolution`] — **never a drop** (I1).
//!
//! - `Ok(v)` passes through as `Recovered` with tag **`Exact`** (FR-R3 fix — the scaffold stamped
//!   `Declared` here; that was wrong: a clean pass-through is not a fallback substitution).
//! - `Err` with **no** matching rule: re-propagates the error **unchanged** (I1 floor).
//! - `Err` with a matching rule: applies the action, respecting the closed set and budgets (I3/I4).
//!
//! # Tag meet (FR-R3 / I2 / VR-5)
//!
//! The recovered tag is chosen **honestly per action**:
//! - `Fallback` → **`Declared`** (fixed ceiling; a substituted value has no checked basis).
//! - `Retry` success → the **attempt's own tag** (inherited; never upgraded).
//! - `Retry` exhausted → `Propagated(original_error)`.
//! - `Escalate`, `CleanupThenPropagate` → `Propagated`.
//! - `Ok` pass-through → **`Exact`** (no substitution; no lower bound imposed).
//!
//! The tag discipline satisfies FR-R3: *recovered tag ≤ policy ceiling* (the meet). Since the
//! policy ceiling for `Fallback` is `Declared` and for `Retry` is the attempt's own tag (which is
//! already honest), the meet is either a no-op or a downgrade — never an upgrade (VR-5).

use mycelium_core::GuaranteeStrength;
use serde::Serialize;

use crate::action::RecoveryAction;
use crate::effect::Budgets;
use crate::outcome::{Outcome, Resolution};
use crate::policy::RecoveryPolicy;
use crate::registry::ClassName;

/// Handle an [`Outcome`] under a recovery policy, providing the error's class for rule lookup.
///
/// `class_of` extracts the error's class for policy lookup (never an eval'd string — X1). It
/// returns an **owned** [`ClassName`] (resolved through the registry at policy-build time; the
/// returned name is already validated — X1).
/// `attempt` is called at most `max_attempts` times for a `Retry` action, yielding a new
/// `(Outcome, GuaranteeStrength)` pair (the strength is the attempt value's honest tag).
///
/// # Never-silent (I1 / C1)
///
/// Every call yields `Recovered | Propagated`; no path drops an error.  The [`Resolution`] type
/// has no `Dropped` variant — I1 is enforced by the type.
///
/// # Guarantee tags (I2 / FR-R3 / VR-5)
///
/// - `Ok(v)` → `Recovered(v, Exact, None)` — the honest tag for a clean pass-through.
/// - `Fallback` → `Recovered(v, Declared, policy_ref)` — honest floor for a substitution.
/// - `Retry` success → `Recovered(v, attempt_tag, policy_ref)` — inherits the attempt's tag.
/// - `Retry` exhausted → `Propagated(original_error, policy_ref, cleanup_overrun: false)`.
/// - `Escalate` → `Propagated(error, policy_ref, cleanup_overrun: false)`.
/// - `CleanupThenPropagate` → `Propagated(error, policy_ref, cleanup_overrun: <bool>)`.
/// - No matching rule → `Propagated(error, None, cleanup_overrun: false)`.
///
/// # Effects (C6 / I3 / I4)
///
/// - `Retry` consumes `EffectKind::Retry` once per attempt from `budgets`. Budget overrun →
///   immediate `Propagated(original_error)`.
/// - `CleanupThenPropagate` consumes the action's declared `effect` once. A budget overrun sets
///   `cleanup_overrun: true` in the `Propagated` outcome (spec §7-Q4 disposition: record, not
///   swallow).
/// - `Fallback` and `Escalate` are effect-free.
///
/// # Policy identity (banked guard #5)
///
/// When a rule matches, the acting [`PolicyRef`](crate::policy::PolicyRef) is computed via
/// [`RecoveryPolicy::policy_ref`] (a stable serde_json encoding — not `Debug`).  If the fallback
/// value's `serde::Serialize` impl fails (e.g. a `NaN` float), the function propagates the
/// original error with `policy: None` — it refuses to apply the action without an established
/// policy identity (never-silent: the error is never dropped, I1).  This situation is a
/// configuration error in the caller's `T: Serialize` impl; the explicit `Propagated` makes it
/// diagnosable.
#[must_use]
pub fn handle_classified<T, E>(
    outcome: Outcome<T, E>,
    policy: &RecoveryPolicy<T>,
    budgets: &mut Budgets,
    class_of: impl Fn(&E) -> ClassName,
    mut attempt: impl FnMut() -> (Outcome<T, E>, GuaranteeStrength),
) -> Resolution<T, E>
where
    T: Serialize + std::fmt::Debug + Clone + Send + Sync + 'static,
{
    let error = match outcome {
        Outcome::Ok(value) => {
            // FR-R3 fix (P5-B): Ok pass-through is Exact — not Declared.
            // A clean Ok is not a fallback substitution; the value carries no lower guarantee
            // imposed by the recovery subsystem (its own guarantee is Exact unless the caller
            // produced it otherwise).  policy: None (no policy acted — nothing to EXPLAIN).
            return Resolution::Recovered {
                value,
                tag: GuaranteeStrength::Exact,
                policy: None,
            };
        }
        Outcome::Err(e) => e,
    };

    let class = class_of(&error);
    let Some(action) = policy.action_for(&class) else {
        // No matching rule → the error propagates UNCHANGED (I1 floor).
        return Resolution::Propagated {
            error,
            policy: None,
            cleanup_overrun: false,
        };
    };

    // Compute the content-addressed PolicyRef (banked guard #5 — stable serde_json encoding).
    // If the fallback value's serialization fails (a configuration error in T's Serialize impl),
    // propagate the original error with policy: None — never-silent (I1): the error is not
    // dropped, and the policy ref hash failure is diagnosable from the Propagated outcome.
    let pref = match policy.policy_ref() {
        Ok(r) => Some(r),
        Err(_) => {
            return Resolution::Propagated {
                error,
                policy: None,
                cleanup_overrun: false,
            };
        }
    };

    match action {
        RecoveryAction::Fallback { value } => {
            // A substituted fallback is honestly Declared — no checked basis (I2/VR-5).
            Resolution::Recovered {
                value: *value.clone(),
                tag: GuaranteeStrength::Declared,
                policy: pref,
            }
        }
        RecoveryAction::Retry { max_attempts } => {
            let max = *max_attempts;
            for _ in 0..max {
                // Consume one retry attempt from the budget ledger (I4/I5).
                if budgets
                    .consume(mycelium_interp::budget::EffectKind::Retry, 1)
                    .is_err()
                {
                    // Budget exhausted before this attempt — original error propagates (I1).
                    return Resolution::Propagated {
                        error,
                        policy: pref,
                        cleanup_overrun: false,
                    };
                }
                let (next_outcome, attempt_tag) = attempt();
                if let Outcome::Ok(value) = next_outcome {
                    // FR-R3 / I2: tag is the attempt's own tag (never upgraded beyond that).
                    return Resolution::Recovered {
                        value,
                        tag: attempt_tag,
                        policy: pref,
                    };
                }
                // Attempt failed — loop and try again (up to max).
            }
            // All retries exhausted → original error propagates (additive — I1).
            Resolution::Propagated {
                error,
                policy: pref,
                cleanup_overrun: false,
            }
        }
        RecoveryAction::Escalate { to_class } => {
            // Re-propagates explicitly (I1). `to_class` is recorded in the PolicyRef hash (C3)
            // but the generic driver cannot physically re-type the opaque `E` with the escalated
            // class. See `RecoveryAction::Escalate` for the full limitation note — this is the
            // Rust-first seam: honest and EXPLAIN-able, pending std.diag integration (M-510/M-520).
            let _ = to_class;
            Resolution::Propagated {
                error,
                policy: pref,
                cleanup_overrun: false,
            }
        }
        RecoveryAction::CleanupThenPropagate { effect } => {
            // Bounded cleanup; a budget overrun skips the cleanup ONLY — original error propagates
            // regardless (additive — I1).  Spec §7-Q4 disposition: record the overrun, not swallow.
            let cleanup_overrun = budgets.consume(effect.clone(), 1).is_err();
            Resolution::Propagated {
                error,
                policy: pref,
                cleanup_overrun,
            }
        }
    }
}

/// Convenience: bridge a `Result<T, E>` into a [`Resolution<T, E>`] under a policy.
///
/// This is [`handle_classified`] with a `Result` input.  `RecoverOutcome<T, E>` is
/// `Resolution<T, E>` — the concrete shape that resolves `error.md` §7-Q1 (no drop variant, I1;
/// honest inherited tag, I2).
#[must_use]
pub fn recover_classified<T, E>(
    result: Result<T, E>,
    policy: &RecoveryPolicy<T>,
    budgets: &mut Budgets,
    class_of: impl Fn(&E) -> ClassName,
    attempt: impl FnMut() -> (Outcome<T, E>, GuaranteeStrength),
) -> Resolution<T, E>
where
    T: Serialize + std::fmt::Debug + Clone + Send + Sync + 'static,
{
    handle_classified(
        Outcome::from_result(result),
        policy,
        budgets,
        class_of,
        attempt,
    )
}
