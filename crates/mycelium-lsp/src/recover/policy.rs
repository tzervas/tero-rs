//! The **reified recovery policy** (RFC-0014 §4.4) — the RFC-0005 pattern (ADR-006) applied to
//! *control* (what happens on an error) rather than presentation (RFC-0013). It is the sibling of the
//! diagnostic policy and **shares RFC-0013's error-class registry** (§4.9): a rule names a class
//! resolved through the registry — never an evaluated string (X1). A recovery policy is
//! content-addressed (`PolicyRef`), so every recovered/re-propagated outcome can answer *"which policy
//! acted on this error, and what does it do?"*.
//!
//! The recovery-action set is **closed** in v0 (§8 resolved); each action is explicit and never-silent.

use std::collections::BTreeMap;

use mycelium_core::{ContentHash, Value};

use super::effect::EffectKind;
use crate::diagnostics::registry::{ClassName, ClassRegistry, UnknownClass};

/// The **closed** v0 recovery-action set (§4.4; §8 resolved). Each action yields an explicit
/// outcome — a recovered value or a re-propagated error — never a silent drop (I1).
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Recover with an explicit fallback value. The recovered value is honestly tagged **`Declared`**
    /// (a substituted fallback has no checked basis — I2/VR-5; recovery only ever downgrades).
    Fallback {
        /// The fallback value (boxed — it is much larger than the other actions' fields).
        value: Box<Value>,
    },
    /// Re-attempt the operation, **bounded** by `max_attempts` (I4). If every attempt fails the
    /// original error **continues to propagate** (additive — a policy is never a silent terminator).
    Retry {
        /// The maximum number of attempts.
        max_attempts: u64,
    },
    /// Transform and re-propagate as another (registry-resolved) error class — still explicit.
    Escalate {
        /// The class to escalate to.
        to: ClassName,
    },
    /// Run a **bounded** effect (consuming from the ambient budget ledger), then let the original
    /// error continue (additive). The effect's budget lives in the [`super::effect::Budgets`] ledger
    /// (one enforcement mechanism); a budget overrun is a graceful `EffectBudgetExhausted` that skips
    /// the cleanup, and the original error propagates regardless.
    CleanupThenPropagate {
        /// The effect the cleanup performs (its budget is declared in the ambient ledger; I5).
        effect: EffectKind,
    },
}

/// A reified recovery policy: a map from a **registry-resolved** [`ClassName`] to its [`RecoveryAction`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecoveryPolicy {
    rules: BTreeMap<ClassName, RecoveryAction>,
}

impl RecoveryPolicy {
    /// An empty policy.
    #[must_use]
    pub fn new() -> Self {
        RecoveryPolicy::default()
    }

    /// Add an action for `class`, **resolving the class name through the registry** (X1). Replaces and
    /// returns any prior action for the class.
    ///
    /// # Errors
    /// Returns [`UnknownClass`] if `class` is not in `registry` — an explicit configuration error.
    pub fn on(
        &mut self,
        registry: &ClassRegistry,
        class: &str,
        action: RecoveryAction,
    ) -> Result<Option<RecoveryAction>, UnknownClass> {
        let name = registry.resolve(class)?;
        Ok(self.rules.insert(name, action))
    }

    /// The recovery action for a resolved class, if any.
    #[must_use]
    pub fn action_for(&self, class: &ClassName) -> Option<&RecoveryAction> {
        self.rules.get(class)
    }

    /// The rules, in deterministic (class-sorted) order.
    pub fn rules(&self) -> impl Iterator<Item = (&ClassName, &RecoveryAction)> {
        self.rules.iter()
    }

    /// Whether the policy has no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The **content address** of this policy (RFC-0005 `PolicyRef`; ADR-006) — a deterministic BLAKE3
    /// over its canonical, sorted rules. Diffable and identity-stable.
    #[must_use]
    pub fn content_id(&self) -> ContentHash {
        let mut h = blake3::Hasher::new();
        let blob = |hasher: &mut blake3::Hasher, bytes: &[u8]| {
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        };
        blob(&mut h, b"mycelium.recovery-policy.v1");
        h.update(&(self.rules.len() as u64).to_le_bytes());
        for (class, action) in &self.rules {
            blob(&mut h, class.as_str().as_bytes());
            match action {
                RecoveryAction::Fallback { value } => {
                    h.update(&[0u8]);
                    blob(&mut h, value.content_hash().as_str().as_bytes());
                }
                RecoveryAction::Retry { max_attempts } => {
                    h.update(&[1u8]);
                    h.update(&max_attempts.to_le_bytes());
                }
                RecoveryAction::Escalate { to } => {
                    h.update(&[2u8]);
                    blob(&mut h, to.as_str().as_bytes());
                }
                RecoveryAction::CleanupThenPropagate { effect } => {
                    h.update(&[3u8]);
                    blob(&mut h, effect.to_string().as_bytes());
                }
            }
        }
        let hex = h.finalize().to_hex();
        ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest")
    }
}
