//! **Declarative error recovery & bounded effects** (M-352; RFC-0014, Accepted 2026-06-16).
//!
//! The recovery half RFC-0013 deferred: a **separable** subsystem (SoC, bounded blast radius) that lets
//! an error *trigger functionality* (fallback, retry, escalate, cleanup) while staying inside the
//! never-silent contract (G2). **Tooling layer only** — no kernel change (KC-3, zero new L0 nodes); no
//! Python (ADR-007). Three pillars:
//!
//! - **errors-as-propagating-values** ([`Outcome`] over a [`StructuredError`]; RFC-0001 substrate);
//! - **explicit declarative recovery** — [`handle`] applies a reified [`policy::RecoveryPolicy`]
//!   (RFC-0005 pattern; **shares RFC-0013's error-class registry**, §4.9) and yields a [`Resolution`]
//!   that is **always** either *recovered* or *re-propagated* — never a silent drop (I1);
//! - **declared, bounded effects** ([`effect`]: declared sets, per-kind budgets, a graceful
//!   `EffectBudgetExhausted` — I3/I4).
//!
//! The governing invariant (RFC-0014 §4.2 I1): a handler **acts on** an error and produces a *new
//! explicit outcome*; it never makes the error vanish unobserved. [`Resolution`] has no "dropped"
//! variant — never-silent is enforced by the type. Recovery never fabricates or upgrades a guarantee
//! (I2/VR-5): a substituted fallback is honestly `Declared`.
//!
//! **The L0 lowering target** — `handle` is a `Match` over the result sum (`Ok | Err`), so recovery
//! introduces **no new kernel node**; that `Match`-over-error-sums target is differentially verified
//! (L1-eval ≡ L0-interp ≡ AOT) in `mycelium-l1` (NFR-7). Wiring the [`effect::Budgets`] ledger into the
//! AOT env-machine's runtime budget resolver is the RFC-0008 integration (§4.8 boundary), not v0.

pub mod effect;
pub mod policy;

use mycelium_core::{ContentHash, GuaranteeStrength, Value};

use crate::diagnostics::registry::ClassName;
use effect::Budgets;
use policy::RecoveryAction;

pub use effect::{
    check_effects, Budgets as EffectBudgets, EffectBudget, EffectBudgetExhausted, EffectKind,
    EffectSet, UndeclaredEffect,
};
pub use policy::{RecoveryAction as Action, RecoveryPolicy};

/// The structured error value — the `Err` payload of the result sum (RFC-0001; the same structured
/// error RFC-0013 *presents*). Its class is a registry-resolved [`ClassName`] (X1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredError {
    /// The error class (registry-resolved).
    pub class: ClassName,
    /// The refusal reason.
    pub message: String,
    /// The site.
    pub site: String,
}

impl StructuredError {
    /// A structured error.
    #[must_use]
    pub fn new(class: ClassName, message: impl Into<String>, site: impl Into<String>) -> Self {
        StructuredError {
            class,
            message: message.into(),
            site: site.into(),
        }
    }
}

/// The result sum `Ok(τ) | Err(ε)` (RFC-0014 §4.1). v0's success domain is a kernel [`Value`]; the
/// `Err` payload is a [`StructuredError`]. Errors propagate by default — there is no silent unwind.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Success.
    Ok(Value),
    /// An explicit, propagating error.
    Err(StructuredError),
}

/// The outcome of handling: an error is **either recovered** (an explicit value with an honest
/// guarantee) **or re-propagated** (an explicit error — possibly transformed). There is deliberately
/// **no "dropped" variant**: never-silent (I1) is a property of this type, so a mutant handler that
/// discarded an error could not be expressed here.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// Recovered with an explicit value, honestly tagged (a substituted fallback is `Declared` — I2).
    Recovered {
        /// The recovered value.
        value: Value,
        /// Its honest guarantee (downgrade-only; never upgraded — VR-5).
        guarantee: GuaranteeStrength,
        /// The `PolicyRef` of the policy that recovered it, if any.
        policy: Option<ContentHash>,
    },
    /// Re-propagated — the error continues to surface (additive over the explicit error — I1).
    Propagated {
        /// The propagating error (possibly transformed/escalated).
        error: StructuredError,
        /// The `PolicyRef` of the policy that acted, if any.
        policy: Option<ContentHash>,
    },
}

/// Handle an [`Outcome`] under a reified recovery `policy`, drawing on a budget ledger and an
/// `attempt` thunk (re-run for `retry`). The result is **always** [`Resolution::Recovered`] or
/// [`Resolution::Propagated`] — never a drop (I1):
///
/// - `Ok(v)` passes through as `Recovered` carrying the value's own guarantee (nothing was recovered).
/// - `Err(e)` with **no** matching rule re-propagates `e` **unchanged** (I1).
/// - `fallback` recovers with an honest `Declared` value (I2/VR-5).
/// - `retry(<=N)` re-attempts up to `N` times; if all fail the **original error propagates** (additive).
/// - `escalate(c')` re-propagates a transformed error (still explicit).
/// - `cleanup_then_propagate` runs a **bounded** effect (a budget overrun is a graceful
///   `EffectBudgetExhausted`, swallowed *for the cleanup only*) and then propagates the original error.
#[must_use]
pub fn handle(
    outcome: Outcome,
    policy: &RecoveryPolicy,
    budgets: &mut Budgets,
    mut attempt: impl FnMut() -> Outcome,
) -> Resolution {
    let error = match outcome {
        Outcome::Ok(value) => {
            // Nothing to recover — the value passes through with its own (unmodified) guarantee.
            let guarantee = value.meta().guarantee();
            return Resolution::Recovered {
                value,
                guarantee,
                policy: None,
            };
        }
        Outcome::Err(e) => e,
    };

    let Some(action) = policy.action_for(&error.class) else {
        // No policy for this class → the error propagates UNCHANGED (I1).
        return Resolution::Propagated {
            error,
            policy: None,
        };
    };
    let pref = Some(policy.content_id());

    match action {
        RecoveryAction::Fallback { value } => Resolution::Recovered {
            value: value.as_ref().clone(),
            // A substituted fallback has no checked basis → honestly `Declared` (I2/VR-5).
            guarantee: GuaranteeStrength::Declared,
            policy: pref,
        },
        RecoveryAction::Retry { max_attempts } => {
            for _ in 0..*max_attempts {
                if let Outcome::Ok(value) = attempt() {
                    let guarantee = value.meta().guarantee();
                    return Resolution::Recovered {
                        value,
                        guarantee,
                        policy: pref,
                    };
                }
            }
            // Retries exhausted → the original error continues to propagate (additive — I1).
            Resolution::Propagated {
                error,
                policy: pref,
            }
        }
        RecoveryAction::Escalate { to } => Resolution::Propagated {
            error: StructuredError {
                class: to.clone(),
                message: format!("escalated from {}: {}", error.class, error.message),
                site: error.site,
            },
            policy: pref,
        },
        RecoveryAction::CleanupThenPropagate { effect } => {
            // Run the bounded cleanup effect against the ambient ledger; a budget overrun (incl. no
            // budget declared — I5) is a graceful EffectBudgetExhausted that skips the cleanup — but
            // the ORIGINAL error propagates regardless (additive — I1).
            let _ = budgets.consume(effect.clone(), 1);
            Resolution::Propagated {
                error,
                policy: pref,
            }
        }
    }
}
