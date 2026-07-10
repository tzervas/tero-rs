//! The **content-addressed diagnostic record** and its **dual human + JSON projection**
//! (RFC-0013 §4.2/§4.3), plus the never-silent **`present`** renderer (§4.1).
//!
//! A diagnostic is *one content-addressed value*; "human" and "JSON" are two **renderers of one
//! truth** (G11). The renderer is a **pure function of an already-emitted error** ([`ReasonedError`])
//! plus an optional policy: it is structurally incapable of catching, softening, or standing in for
//! that error — [`present`] returns the error **unchanged** alongside the presentation (I1).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use mycelium_core::ContentHash;
use serde::{Deserialize, Serialize};

use super::policy::DiagnosticPolicy;
use super::registry::ClassName;
use super::sink::{Route, SinkBinding, UnknownRoute};

/// A graded context **level** — a verbosity knob over *one* truth (§4.2). Ordered
/// `Minimal < Medium < Detailed`; raising it shows *more of* a diagnostic, never *whether* the
/// underlying error exists (I2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// The refusal, its reason, and its site. The error is always present at this level (I2).
    Minimal,
    /// Adds the `NotValidatedReason` / `FeedbackSummary`-style reason detail.
    Medium,
    /// Adds an **allowlisted** set of additional context fields (§4.5 X2) — never a wholesale dump.
    Detailed,
}

/// The **allowlist** for the detailed tier (§4.5, exclusion X2): the *only* context-field names a
/// detailed diagnostic may carry. Context not on this list is **not gathered** — there is no path by
/// which a wholesale environment / locals dump (and the secrets in it) reaches a diagnostic.
pub const DETAILED_ALLOWLIST: &[&str] = &[
    "from_repr",
    "to_repr",
    "honesty_bound",
    "policy",
    "lemma_ref",
    "trials",
    "expected_source",
    "element_index",
    "required_dim",
    "supplied_dim",
    "binary_width",
    "ternary_trits",
];

/// Keep only the allowlisted context fields (X2). Applied at record construction, so a record never
/// even *holds* a non-allowlisted field.
fn allowlist(context: BTreeMap<String, String>) -> BTreeMap<String, String> {
    context
        .into_iter()
        .filter(|(k, _)| DETAILED_ALLOWLIST.contains(&k.as_str()))
        .collect()
}

/// The **explicit, already-emitted reasoned error** this layer *presents* — never replaces (I1).
/// Built from the structured errors the kernel / checker / linter already emit (a swap refusal, a
/// `CheckVerdict::NotValidated`, a lint finding). The renderer is a pure function *of* it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonedError {
    /// The error class (resolved through the registry, never evaluated — X1).
    pub class: ClassName,
    /// The refusal message — always shown, even at [`Level::Minimal`] (I2).
    pub message: String,
    /// The site (breadcrumb / location).
    pub site: String,
    /// The reason detail surfaced at [`Level::Medium`] and above.
    pub reason: Option<String>,
    /// Candidate additional context, **allowlist-filtered** at projection (§4.5 X2). Fields not on
    /// [`DETAILED_ALLOWLIST`] are dropped before they ever reach a [`DiagnosticRecord`].
    pub context: BTreeMap<String, String>,
}

impl ReasonedError {
    /// A minimal reasoned error (class + message + site), no reason or context.
    #[must_use]
    pub fn new(class: ClassName, message: impl Into<String>, site: impl Into<String>) -> Self {
        ReasonedError {
            class,
            message: message.into(),
            site: site.into(),
            reason: None,
            context: BTreeMap::new(),
        }
    }

    /// Attach a medium-tier reason.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Attach a candidate detailed-tier context field (allowlist-filtered at projection).
    #[must_use]
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// One **content-addressed diagnostic** (§4.3) — the canonical truth. Human and JSON are projections
/// *of* this record; both carry its [`content_id`](DiagnosticRecord::content_id).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticRecord {
    /// The error class (the refusal's identity).
    pub class: String,
    /// The presentation message (a policy may set this; it does not change the error's identity).
    pub message: String,
    /// The site.
    pub site: String,
    /// The medium-tier reason detail, if any.
    pub reason: Option<String>,
    /// The verbosity level (set by policy; default [`Level::Minimal`]).
    pub level: Level,
    /// Free-form string tags (v0; §4.4 DN04-Q2).
    pub tags: BTreeSet<String>,
    /// The output route, if any. Routing concerns *where the presentation goes*, never *whether the
    /// error propagates* (I1).
    pub route: Option<String>,
    /// Allowlisted detailed-tier context (§4.5 X2) — already filtered; never a wholesale dump.
    pub context: BTreeMap<String, String>,
    /// The `PolicyRef` (content hash) of the policy that shaped this diagnostic, if any (§4.4).
    pub policy: Option<String>,
}

/// The result of presenting an error: the **additive** diagnostic *and* the explicit error, **still
/// propagating, unchanged** (I1). Returning the error here is the structural proof that the renderer
/// cannot suppress it — a mutant renderer that dropped it would not type-check / would fail the
/// never-silent invariant test (§5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Presentation {
    /// The additive presentation (§4.1).
    pub diagnostic: DiagnosticRecord,
    /// The explicit error — **unchanged** by any policy (I1).
    pub error: ReasonedError,
}

/// Present an explicit [`ReasonedError`] as a [`DiagnosticRecord`], optionally shaped by a policy.
///
/// The error is returned **unchanged** in [`Presentation::error`] (I1: no policy, level, tag, route,
/// or message can cause it not to surface). A policy sets only message / tags / level / route
/// (§4.4 I4); the class + site of the refusal are always present, and at [`Level::Minimal`] the
/// message names it (I2). The detailed-tier context is allowlist-filtered (X2). When a policy
/// applies, the record carries that policy's content hash (`PolicyRef`; §4.4).
#[must_use]
pub fn present(error: ReasonedError, policy: Option<&DiagnosticPolicy>) -> Presentation {
    let rule = policy.and_then(|p| p.rule_for(&error.class));

    let message = rule
        .and_then(|r| r.message.clone())
        .unwrap_or_else(|| error.message.clone());
    let level = rule.and_then(|r| r.level).unwrap_or(Level::Minimal);
    let tags = rule.map(|r| r.tags.clone()).unwrap_or_default();
    let route = rule.and_then(|r| r.route.clone());
    // The PolicyRef is recorded only when a policy actually shaped this diagnostic.
    let policy_ref = match (policy, rule) {
        (Some(p), Some(_)) => Some(p.content_id().as_str().to_owned()),
        _ => None,
    };

    let diagnostic = DiagnosticRecord {
        class: error.class.as_str().to_owned(),
        message,
        site: error.site.clone(),
        reason: error.reason.clone(),
        level,
        tags,
        route,
        context: allowlist(error.context.clone()),
        policy: policy_ref,
    };

    // I1, made structural: the error is handed back untouched. It still propagates.
    Presentation { diagnostic, error }
}

/// A canonical, injective byte encoder for content-addressing a record (mirrors the kernel's
/// length-prefixed BLAKE3 framing in `mycelium-core::content`, kept in the tooling layer so no kernel
/// dependency is added — KC-3).
struct Canon {
    h: blake3::Hasher,
}

impl Canon {
    fn new() -> Self {
        Canon {
            h: blake3::Hasher::new(),
        }
    }
    fn blob(&mut self, bytes: &[u8]) {
        self.h.update(&(bytes.len() as u64).to_le_bytes());
        self.h.update(bytes);
    }
    fn str(&mut self, s: &str) {
        self.blob(s.as_bytes());
    }
    fn opt(&mut self, s: Option<&str>) {
        match s {
            // Distinct tags so `None` and `Some("")` can never collide.
            None => {
                self.h.update(&[0u8]);
            }
            Some(v) => {
                self.h.update(&[1u8]);
                self.str(v);
            }
        }
    }
    fn finish(self) -> ContentHash {
        let hex = self.h.finalize().to_hex();
        ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest")
    }
}

impl DiagnosticRecord {
    /// Resolve this diagnostic's `route` to its RFC-0008 [`SinkBinding`] (M-354, RFC-0013 §8).
    /// `None` when no route is set; `Some(Err(_))` when the route string is not in the closed v0 set
    /// (an explicit [`UnknownRoute`], never a silent misroute). This is the **sink-dispatch** point and
    /// it lives **outside** [`present`] — so routing (or a failed route resolution) can never gate the
    /// error's propagation (I1): the error has already surfaced unchanged in [`Presentation::error`].
    #[must_use]
    pub fn sink(&self) -> Option<Result<SinkBinding, UnknownRoute>> {
        self.route
            .as_deref()
            .map(|r| Route::resolve(r).map(Route::binding))
    }

    /// The **content address** of this diagnostic (§4.3; ADR-003) — a deterministic BLAKE3 over its
    /// canonical fields. Identity-stable: the same diagnostic content always hashes the same, so the
    /// human and JSON projections share it (I3).
    #[must_use]
    pub fn content_id(&self) -> ContentHash {
        let mut c = Canon::new();
        c.str("mycelium.diagnostic.v1"); // domain separation
        c.str(&self.class);
        c.str(&self.message);
        c.str(&self.site);
        c.opt(self.reason.as_deref());
        c.str(match self.level {
            Level::Minimal => "minimal",
            Level::Medium => "medium",
            Level::Detailed => "detailed",
        });
        c.h.update(&(self.tags.len() as u64).to_le_bytes());
        for t in &self.tags {
            c.str(t);
        }
        c.opt(self.route.as_deref());
        c.h.update(&(self.context.len() as u64).to_le_bytes());
        for (k, v) in &self.context {
            c.str(k);
            c.str(v);
        }
        c.opt(self.policy.as_deref());
        c.finish()
    }

    /// The **JSON projection** (§4.3): the lossless, round-trippable machine record, with its
    /// content `id` embedded. `from_json(to_json(r))` recovers an equal record with an equal
    /// `content_id` (I3).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut value = serde_json::to_value(self).expect("a diagnostic record serializes");
        if let serde_json::Value::Object(map) = &mut value {
            map.insert(
                "id".to_owned(),
                serde_json::Value::String(self.content_id().as_str().to_owned()),
            );
        }
        serde_json::to_string(&value).expect("a json value serializes")
    }

    /// Recover a record from its JSON projection (I3). The embedded `id` is informational — it is
    /// recomputed from the recovered fields, so the round-trip is over the semantic content.
    ///
    /// # Errors
    /// Returns a [`serde_json::Error`] if the input is not a well-formed diagnostic record.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// The **human projection** (§4.3), graded by [`self.level`](Self::level). Minimal names the
    /// refusal (class + message + site; I2); medium adds the reason; detailed adds the allowlisted
    /// context. The content `id` is embedded so the human view carries the same identity as the JSON
    /// one (I3).
    #[must_use]
    pub fn to_human(&self) -> String {
        let mut out = String::new();
        // Minimal — the refusal is always named (I2).
        out.push_str(&format!(
            "[{}] {} (at {})",
            self.class, self.message, self.site
        ));
        if let Some(route) = &self.route {
            out.push_str(&format!(" → {route}"));
        }
        if !self.tags.is_empty() {
            let tags: Vec<&str> = self.tags.iter().map(String::as_str).collect();
            out.push_str(&format!("  #{}", tags.join(" #")));
        }
        // Medium — the reason detail.
        if self.level >= Level::Medium {
            if let Some(reason) = &self.reason {
                out.push_str(&format!("\n  reason: {reason}"));
            }
        }
        // Detailed — the allowlisted context (never a wholesale dump; X2).
        if self.level >= Level::Detailed {
            for (k, v) in &self.context {
                out.push_str(&format!("\n  {k}: {v}"));
            }
        }
        out.push_str(&format!("\n  id: {}", self.content_id().as_str()));
        out
    }
}

impl fmt::Display for DiagnosticRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_human())
    }
}
