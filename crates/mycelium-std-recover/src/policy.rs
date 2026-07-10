//! The **reified, content-addressed recovery policy** (RFC-0014 §4.4; RFC-0005 pattern; ADR-006).
//!
//! A `RecoveryPolicy` is an `on <ErrorClass> => <RecoveryAction>` map.  Its identity is its
//! **content address** (`PolicyRef` = `ContentHash`; RFC-0001 §4.6 / ADR-003) — a BLAKE3 hash
//! over the canonical, sorted rules.  This makes every recovered/re-propagated outcome answerable
//! to *"which policy acted on this error, and what does it do?"* (C3 / EXPLAIN-able).
//!
//! # Content-hash validity (banked guard #5)
//!
//! Inputs that canonicalize ambiguously must be rejected **before** hashing.  The only inputs
//! here are class names (resolved through the registry — never raw strings, X1) and action
//! parameters (integers / strings / `EffectKind` / **stable serializations of fallback values**).
//!
//! Fallback values are serialized with `serde_json::to_vec` (the `T: serde::Serialize` bound),
//! which is deterministic for a given type and stable across Rust versions.  `Debug` is **not**
//! used for hashing — it is not a stable serialization format and can vary across implementations
//! or compiler versions, which would make two structurally-equal policies hash differently
//! (banked guard #5 violation).  On a serialization error `policy_ref` returns
//! `Err(PolicyHashError)` — it never silently skips a value (which would make two different
//! fallback values collide on the same hash).

use std::collections::BTreeMap;

use mycelium_core::ContentHash;
use mycelium_interp::budget::EffectKind;
use serde::Serialize;

use crate::action::RecoveryAction;
use crate::registry::{ClassName, ClassRegistry, UnknownClass};

/// The content address of a `RecoveryPolicy` (RFC-0001 §4.6 / ADR-006 / `PolicyRef`).
///
/// A deterministic BLAKE3 hash over the policy's canonical, sorted rules.  Stable across
/// serialization boundaries and suitable as a cache / deduplication key.
pub type PolicyRef = ContentHash;

/// An error computing the content address of a [`RecoveryPolicy`] (banked guard #5).
///
/// Returned by [`RecoveryPolicy::policy_ref`] when a fallback value cannot be serialized to a
/// stable canonical form.  This is a configuration error — a fallback value type must implement
/// [`serde::Serialize`] correctly (i.e. produce a stable, deterministic encoding) for the policy
/// to be content-addressed.
///
/// # Never-silent (I1 / banked guard #5)
///
/// `policy_ref` refuses to produce a `PolicyRef` rather than silently skipping the offending
/// value from the hash.  Skipping would cause two policies with different fallback values to
/// produce identical `PolicyRef`s — a content-address collision (banked guard #5 violation).
#[derive(Debug)]
pub struct PolicyHashError {
    /// The `serde_json` serialization error.
    pub source: serde_json::Error,
}

impl std::fmt::Display for PolicyHashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cannot compute stable PolicyRef: fallback value serialization failed (serde_json): {}",
            self.source
        )
    }
}

mycelium_std_core::impl_std_error!(PolicyHashError, source = |this| { Some(&this.source) });

/// A reified, content-addressed recovery policy.
///
/// Maps a **registry-resolved** [`ClassName`] to a [`RecoveryAction`].  Rules are stored in a
/// `BTreeMap` (sorted by class name) so the content hash is deterministic and the policy is
/// diffable / inspectable (C3).
///
/// # EXPLAIN-ability (C3)
///
/// A `RecoveryPolicy` is its own EXPLAIN artifact: the `rules()` iterator exposes every class →
/// action binding, and [`RecoveryPolicy::policy_ref`] is the stable identity tag embedded in
/// every [`crate::Resolution`] outcome.
///
/// # Type bound
///
/// The `T` type parameter is the fallback value type — it must be
/// [`serde::Serialize`] + [`std::fmt::Debug`] + [`Clone`] + `Send + Sync + 'static`.
/// `serde::Serialize` is required for stable, canonical content hashing of fallback values
/// (banked guard #5 — `Debug` is not a stable serialization and is not used for hashing).
#[derive(Debug, Clone)]
pub struct RecoveryPolicy<T> {
    rules: BTreeMap<ClassName, RecoveryAction<T>>,
}

impl<T> Default for RecoveryPolicy<T> {
    fn default() -> Self {
        RecoveryPolicy {
            rules: BTreeMap::new(),
        }
    }
}

impl<T: Serialize + std::fmt::Debug + Clone + Send + Sync + 'static> RecoveryPolicy<T> {
    /// An empty policy (no rules).
    #[must_use]
    pub fn new() -> Self {
        RecoveryPolicy::default()
    }

    /// Add an `on <class> => <action>` rule, resolving `class` **and** any `Escalate.to_class`
    /// through `registry` (X1 — both class names are registry-validated, never raw strings).
    ///
    /// Replaces and returns any prior action for the class.  An unknown class (either the LHS
    /// `class` or an `Escalate` target) is an explicit configuration error ([`UnknownClass`]),
    /// never a silent fabrication (G2).
    ///
    /// # Errors
    ///
    /// Returns [`UnknownClass`] if:
    /// - `class` is not in the registry (LHS class), **or**
    /// - the action is [`RecoveryAction::Escalate`] and `to_class` is not in the registry
    ///   (X1 — the escalation target must be a registered class, never an unvalidated string).
    pub fn on(
        &mut self,
        registry: &ClassRegistry,
        class: &str,
        action: RecoveryAction<T>,
    ) -> Result<Option<RecoveryAction<T>>, UnknownClass> {
        // X1: resolve the LHS class through the registry.
        let name = registry.resolve(class)?;

        // Fix #3 (Copilot review finding): also validate the Escalate `to_class` through the
        // registry.  The X1 invariant says "a class is a name resolved through the registry —
        // never an evaluated string"; the LHS was checked but the escalation target was not,
        // allowing an unregistered raw string to slip through.  Never silent — return an explicit
        // `Err(UnknownClass)` for an unregistered `to_class` (G2 / X1).
        if let RecoveryAction::Escalate { ref to_class } = action {
            // Validate that `to_class` is registered.  We deliberately do NOT resolve it to a
            // `ClassName` here because the generic driver (`handle_classified`) operates on an
            // opaque `E` and cannot physically re-type it; `to_class` is stored as a `String`
            // and hashed into the `PolicyRef` for EXPLAIN-ability (C3), but the structural
            // escalation is a Rust-first seam (see `RecoveryAction::Escalate` for the full note).
            // The registry check ensures X1 holds: only registered classes can be targeted.
            if !registry.contains(to_class) {
                return Err(UnknownClass {
                    name: to_class.clone(),
                });
            }
        }

        Ok(self.rules.insert(name, action))
    }

    /// The recovery action for a resolved class, if any.
    #[must_use]
    pub fn action_for(&self, class: &ClassName) -> Option<&RecoveryAction<T>> {
        self.rules.get(class)
    }

    /// The rules in deterministic (class-sorted) order (C3 — inspectable, diffable).
    pub fn rules(&self) -> impl Iterator<Item = (&ClassName, &RecoveryAction<T>)> {
        self.rules.iter()
    }

    /// Whether the policy has no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The **content address** of this policy (RFC-0005 `PolicyRef`; ADR-006).
    ///
    /// A deterministic BLAKE3 over the canonical sorted rules; diffable and identity-stable.
    /// Every outcome returned by [`crate::handle`] carries this `PolicyRef` when a rule was
    /// applied, so every outcome can be traced back to the exact policy that acted (C3).
    ///
    /// # Honesty (banked guard #5)
    ///
    /// All hashed fields use stable, canonical encodings:
    /// - Class names are registry-resolved strings (X1).
    /// - Integer action parameters (`max_attempts`) are hashed as little-endian bytes.
    /// - `EffectKind` values are hashed via their `Display` string.
    /// - Fallback values are serialized with **`serde_json::to_vec`** — a canonical, deterministic
    ///   encoding that is stable across Rust versions and implementations.  `Debug` is **not** used:
    ///   `Debug` is not a stable serialization format and its output can vary between Rust versions
    ///   or custom implementations, making content-addressed identity unreliable.
    /// - The rule count is hashed before iterating so a length-1 policy for class `"ab"` and a
    ///   length-2 policy for classes `"a"` / `"b"` do not collide.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyHashError`] if a fallback value cannot be serialized by `serde_json`.
    /// This is a configuration error — a value type whose `serde::Serialize` impl fails
    /// (e.g. contains a `NaN` float that `serde_json` refuses to encode) cannot participate in
    /// content-addressed policy identity.  Returning `Err` rather than silently skipping the
    /// value is the only correct behavior: skipping would allow two policies with different
    /// fallback values to produce the same `PolicyRef` — a content-address collision (banked
    /// guard #5 / G2 — never silent).
    pub fn policy_ref(&self) -> Result<PolicyRef, PolicyHashError> {
        let mut h = blake3::Hasher::new();
        // Length-prefix every variable-length field so different structures cannot collide.
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
                    // Serialize the fallback value to a stable, canonical JSON encoding (banked
                    // guard #5 fix).  serde_json preserves struct field order and is deterministic
                    // for a given type, making it stable across Rust versions — unlike `Debug`,
                    // which has no stability guarantee.  On error we refuse rather than silently
                    // skipping the value (a skip would make two different fallback values hash the
                    // same — a content-address collision).
                    let bytes = serde_json::to_vec(value.as_ref())
                        .map_err(|e| PolicyHashError { source: e })?;
                    blob(&mut h, &bytes);
                }
                RecoveryAction::Retry { max_attempts } => {
                    h.update(&[1u8]);
                    h.update(&max_attempts.to_le_bytes());
                }
                RecoveryAction::Escalate { to_class } => {
                    h.update(&[2u8]);
                    blob(&mut h, to_class.as_bytes());
                }
                RecoveryAction::CleanupThenPropagate { effect } => {
                    h.update(&[3u8]);
                    blob(&mut h, effect.to_string().as_bytes());
                }
            }
        }
        let hex = h.finalize().to_hex();
        Ok(ContentHash::from_parts("blake3", hex.as_str())
            .expect("blake3 hex is always a valid content hash"))
    }
}

/// The declared, closed effect set for a policy (I3 / RFC-0014 §4.5).
///
/// Returns the set of `EffectKind`s that the policy's actions may perform.  Used by
/// [`crate::effect::check_effects`] to enforce I3 (no undeclared effect).
pub fn policy_effects<T>(policy: &RecoveryPolicy<T>) -> std::collections::BTreeSet<EffectKind>
where
    T: Serialize + std::fmt::Debug + Clone + Send + Sync + 'static,
{
    let mut set = std::collections::BTreeSet::new();
    for (_, action) in policy.rules() {
        match action {
            RecoveryAction::Retry { .. } => {
                set.insert(EffectKind::Retry);
            }
            RecoveryAction::CleanupThenPropagate { effect } => {
                set.insert(effect.clone());
            }
            RecoveryAction::Fallback { .. } | RecoveryAction::Escalate { .. } => {}
        }
    }
    set
}
