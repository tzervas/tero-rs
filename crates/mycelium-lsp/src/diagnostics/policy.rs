//! The **reified per-definition error-handling policy** (RFC-0013 §4.4) — the RFC-0005 selection-
//! policy pattern (ADR-006) applied to *presentation/routing*.
//!
//! ```text
//! on <ErrorClass> => { message?, tags?, level?, route? }
//! ```
//!
//! A policy is a **content-addressed, inspectable artifact**: every diagnostic it shapes records its
//! [`content_id`](DiagnosticPolicy::content_id) (`PolicyRef`), so one can always answer *"which policy
//! shaped this diagnostic, and what does it do?"*. It configures **presentation/routing only** (I4):
//! no recovery, no fallback, no handler, no control-flow effect — that is RFC-0014's concern. The
//! explicit error it attaches to propagates unchanged (I1, enforced by [`super::record::present`]).
//!
//! `<ErrorClass>` is resolved **through the registry** (§4.5 X1): a rule cannot name a class the
//! registry does not know — an unknown class is an explicit error, never silently ignored.

use std::collections::{BTreeMap, BTreeSet};

use mycelium_core::ContentHash;
use serde::{Deserialize, Serialize};

use super::record::Level;
use super::registry::{ClassName, ClassRegistry, UnknownClass};

/// A single `on <ErrorClass> => { … }` rule. Presentation/routing only (I4); a `None` field leaves
/// the corresponding default in place.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    /// A presentation message override (does not change the error's identity or reason).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Free-form string tags (v0; §4.4 DN04-Q2).
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub tags: BTreeSet<String>,
    /// The default verbosity for diagnostics under this rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<Level>,
    /// An output route — *where* the presentation goes, never *whether* the error propagates (I1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
}

impl Rule {
    /// An empty rule (all defaults).
    #[must_use]
    pub fn new() -> Self {
        Rule::default()
    }
    /// Set the presentation message.
    #[must_use]
    pub fn message(mut self, m: impl Into<String>) -> Self {
        self.message = Some(m.into());
        self
    }
    /// Add a tag.
    #[must_use]
    pub fn tag(mut self, t: impl Into<String>) -> Self {
        self.tags.insert(t.into());
        self
    }
    /// Set the level.
    #[must_use]
    pub fn level(mut self, l: Level) -> Self {
        self.level = Some(l);
        self
    }
    /// Set the route from a free-form string (the on-the-wire/`PolicyFile` projection form). Prefer
    /// [`route_to`](Rule::route_to) for a route in the closed v0 set; a string that does not resolve to
    /// a [`Route`](super::sink::Route) is an explicit [`UnknownRoute`](super::sink::UnknownRoute) at
    /// sink-resolution time (never a silent misroute — RFC-0013 §8).
    #[must_use]
    pub fn route(mut self, r: impl Into<String>) -> Self {
        self.route = Some(r.into());
        self
    }
    /// Set the route from the **closed v0** [`Route`](super::sink::Route) vocabulary (the checked path).
    #[must_use]
    pub fn route_to(mut self, r: super::sink::Route) -> Self {
        self.route = Some(r.as_str().to_owned());
        self
    }
}

/// A reified error-handling policy: a map from a **registry-resolved** [`ClassName`] to its [`Rule`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiagnosticPolicy {
    rules: BTreeMap<ClassName, Rule>,
}

impl DiagnosticPolicy {
    /// An empty policy.
    #[must_use]
    pub fn new() -> Self {
        DiagnosticPolicy::default()
    }

    /// Add a rule for `class`, **resolving the class name through the registry** (X1). A duplicate
    /// class replaces the prior rule and returns it.
    ///
    /// # Errors
    /// Returns [`UnknownClass`] if `class` is not in `registry` — an explicit configuration error
    /// (never silently ignored, never coerced into code).
    pub fn on(
        &mut self,
        registry: &ClassRegistry,
        class: &str,
        rule: Rule,
    ) -> Result<Option<Rule>, UnknownClass> {
        let name = registry.resolve(class)?;
        Ok(self.rules.insert(name, rule))
    }

    /// The rule for a resolved class, if any.
    #[must_use]
    pub fn rule_for(&self, class: &ClassName) -> Option<&Rule> {
        self.rules.get(class)
    }

    /// The rules, in deterministic (class-sorted) order.
    pub fn rules(&self) -> impl Iterator<Item = (&ClassName, &Rule)> {
        self.rules.iter()
    }

    /// Whether the policy has no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The **content address** of this policy (RFC-0005 `PolicyRef`; ADR-006) — a deterministic
    /// BLAKE3 over its canonical, sorted rules. Diffable and identity-stable regardless of insertion
    /// order or which on-disk format (JSON/YAML/TOML) a file used (§4.7).
    #[must_use]
    pub fn content_id(&self) -> ContentHash {
        let mut h = blake3::Hasher::new();
        let blob = |hasher: &mut blake3::Hasher, bytes: &[u8]| {
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        };
        blob(&mut h, b"mycelium.diagnostic-policy.v1");
        h.update(&(self.rules.len() as u64).to_le_bytes());
        for (class, rule) in &self.rules {
            blob(&mut h, class.as_str().as_bytes());
            // The rule is hashed by its canonical JSON — serde gives a stable, field-ordered form
            // (BTreeSet tags are sorted; Option fields skip when None).
            let json = serde_json::to_string(rule).expect("a rule serializes");
            blob(&mut h, json.as_bytes());
        }
        let hex = h.finalize().to_hex();
        ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest")
    }
}

/// A serializable projection of a policy (RFC-0013 §4.7: a file is a *projection of* the canonical
/// declaration, not the source of truth). Class names are strings here; re-ingesting validates them
/// through the registry (X1), so an on-disk file can never smuggle in an unknown/evaluated class.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyFile {
    /// `class name -> rule`.
    #[serde(default)]
    pub on: BTreeMap<String, Rule>,
}

impl DiagnosticPolicy {
    /// Project this policy to a serializable [`PolicyFile`] (one on-disk form; §4.7).
    #[must_use]
    pub fn to_file(&self) -> PolicyFile {
        PolicyFile {
            on: self
                .rules
                .iter()
                .map(|(c, r)| (c.as_str().to_owned(), r.clone()))
                .collect(),
        }
    }

    /// Ingest a [`PolicyFile`], **resolving every class name through the registry** (X1). An unknown
    /// class anywhere in the file is an explicit error — the file is rejected as a whole, never
    /// partially/silently applied.
    ///
    /// # Errors
    /// Returns [`UnknownClass`] for the first class name not in `registry`.
    pub fn from_file(registry: &ClassRegistry, file: &PolicyFile) -> Result<Self, UnknownClass> {
        let mut policy = DiagnosticPolicy::new();
        for (class, rule) in &file.on {
            policy.on(registry, class, rule.clone())?;
        }
        Ok(policy)
    }
}
