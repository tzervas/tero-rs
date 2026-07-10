//! The **automatic baseline** (M-362; RFC-0015) — the automation layer *over* RFC-0013 (presentation)
//! and RFC-0014 (recovery).
//!
//! It derives a zero-config **baseline diagnostic policy** from the language's own structured mapping
//! — the error-class registry (RFC-0013 §4.5) and a **closed, total `class → (level, route)` table** —
//! optionally scoped per-definition by that definition's **declared effects** (the classes it can
//! raise). The result is an ordinary, content-addressed [`DiagnosticPolicy`] you can read, diff, and
//! `EXPLAIN`. The four honesty-boundary rules of RFC-0015 §4.1 hold by construction:
//!
//! - **(A1) additive only.** The baseline is a [`DiagnosticPolicy`] — presentation/routing only,
//!   structurally incapable of changing control flow (RFC-0013 I1). Auto-applying it can never swallow,
//!   soften, or hide an error.
//! - **(A2) recovery is opt-in, declared, bounded.** No recovery is ever auto-applied. A [`RecoveryProfile`]
//!   is produced **only** when explicitly requested, over the **explicitly supplied** classes, and its
//!   actions are bounded (RFC-0014 I4/I5).
//! - **(A3) reified + EXPLAIN.** The derived policy is content-addressed ([`DiagnosticPolicy::content_id`])
//!   and [`explain_baseline`] answers "what baseline applied here, and why?".
//! - **(A4) total, not learned.** [`baseline_for_class`] is a total, deterministic function of the class
//!   name (a closed table + a safe fallback) — never a learned or `eval`-ed guess (VR-5/RFC-0005).
//!
//! §8 resolutions and the prior-art grounding live in `research/06-automatic-baseline-diagnostics-RECORD.md`.

use crate::diagnostics::policy::{DiagnosticPolicy, Rule};
use crate::diagnostics::record::Level;
use crate::diagnostics::registry::{ClassRegistry, UnknownClass};
use crate::diagnostics::sink::Route;
use crate::recover::policy::{RecoveryAction, RecoveryPolicy};

/// The auto-derived baseline for one error class: its presentation level + route, and the *rationale*
/// the derivation used (surfaced by [`explain_baseline`] — A3/A4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineRule {
    /// The baseline presentation level.
    pub level: Level,
    /// The baseline route (an RFC-0013 §8 observability sink).
    pub route: Route,
    /// Why this class maps here — a fixed, inspectable reason (no learning; A4).
    pub rationale: &'static str,
}

/// The **total** baseline derivation (A4): a deterministic function of the class name — a closed table
/// over the v0 built-in classes plus a **safe additive fallback** (`Stream`/`Minimal`) for any other
/// registered class. Never learned, never `eval`-ed; the same input always yields the same rule.
#[must_use]
pub fn baseline_for_class(class: &str) -> BaselineRule {
    let (level, route, rationale) = match class {
        // Representation-crossing refusals → the durable crossing-audit view (RFC-0013 §4.6), medium
        // detail (the from/to repr matters). `UnsupportedSwapPair` is one but does not share the prefix.
        "UnsupportedSwapPair" => (
            Level::Medium,
            Route::Audit,
            "representation-crossing refusal: durable audit, medium detail (RFC-0013 §4.6)",
        ),
        c if c.starts_with("Swap") => (
            Level::Medium,
            Route::Audit,
            "representation-crossing refusal: durable audit, medium detail (RFC-0013 §4.6)",
        ),
        // Static-check refusals → the in-process diagnostic stream, medium (the reason is useful).
        "NotValidated" | "TypeMismatch" | "UnresolvedName" => (
            Level::Medium,
            Route::Stream,
            "static-check refusal: diagnostic stream, medium detail",
        ),
        // Honesty-load-bearing advisories → durable audit (you want a record of a Declared bound /
        // policy divergence), medium.
        "UnverifiedBound" | "PolicyDivergence" => (
            Level::Medium,
            Route::Audit,
            "honesty-load-bearing advisory: durable audit so the unverified/divergent case is recorded (VR-5)",
        ),
        // Authoring-invariant lints → the diagnostic stream, minimal (advisory while editing).
        "ImplicitSwap" | "PlaceholderPolicy" | "FreeVariable" => (
            Level::Minimal,
            Route::Stream,
            "authoring-invariant lint: diagnostic stream, minimal",
        ),
        // Safe additive fallback for any other registered class: log to the stream, minimally. Additive
        // (A1) and never silent — the error still propagates; this only routes its presentation.
        _ => (
            Level::Minimal,
            Route::Stream,
            "default: safe additive baseline — diagnostic stream, minimal (the class still propagates)",
        ),
    };
    BaselineRule {
        level,
        route,
        rationale,
    }
}

/// Derive the baseline [`DiagnosticPolicy`] for **every** class in `registry` (the broadest scope). A1:
/// the result is presentation-only — it cannot change control flow. Each rule is tagged `baseline` so an
/// auto-derived rule is distinguishable from a hand-written one.
#[must_use]
pub fn derive_baseline(registry: &ClassRegistry) -> DiagnosticPolicy {
    let classes: Vec<String> = registry.classes().map(str::to_owned).collect();
    // Every class is from the registry, so `on` cannot fail; build infallibly.
    build(registry, classes.iter().map(String::as_str))
        .expect("classes enumerated from the registry resolve")
}

/// Derive the baseline scoped to a **definition's declared effect classes** (the classes it can raise;
/// RFC-0014 I3) — the per-definition auto-wrap scope (§8-Q3). An unknown class is an explicit error
/// (X1; never silently dropped).
///
/// # Errors
/// Returns [`UnknownClass`] for the first `class` not in `registry`.
pub fn derive_baseline_for(
    registry: &ClassRegistry,
    classes: &[&str],
) -> Result<DiagnosticPolicy, UnknownClass> {
    build(registry, classes.iter().copied())
}

fn build<'a>(
    registry: &ClassRegistry,
    classes: impl Iterator<Item = &'a str>,
) -> Result<DiagnosticPolicy, UnknownClass> {
    let mut policy = DiagnosticPolicy::new();
    for class in classes {
        let b = baseline_for_class(class);
        let rule = Rule::new().level(b.level).route_to(b.route).tag("baseline");
        policy.on(registry, class, rule)?;
    }
    Ok(policy)
}

/// The `EXPLAIN` of the baseline derivation over `registry` (A3): every class with its derived level,
/// route, and rationale — so "what baseline applies, and why?" is always answerable. Deterministic
/// (class-sorted) and stable.
#[must_use]
pub fn explain_baseline(registry: &ClassRegistry) -> String {
    let mut out = String::from("baseline diagnostic policy (RFC-0015; derived, not learned):\n");
    for class in registry.classes() {
        let b = baseline_for_class(class);
        out.push_str(&format!(
            "  {class}: level={:?} route={} — {}\n",
            b.level,
            b.route.as_str(),
            b.rationale
        ));
    }
    out
}

/// The **closed v0** set of named, opt-in, bounded recovery profiles (RFC-0015 §8-Q2; A2). Recovery is
/// **never** auto-applied — a profile is built only on explicit request, over explicitly-supplied
/// classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryProfile {
    /// `strict` — propagate everything (the honest default: no recovery, all errors bubble).
    Strict,
    /// `resilient` — bounded `retry(<=3)` on the supplied classes; anything else propagates (I4/I5).
    Resilient,
}

impl RecoveryProfile {
    /// The canonical profile name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RecoveryProfile::Strict => "strict",
            RecoveryProfile::Resilient => "resilient",
        }
    }

    /// The closed v0 set, for enumeration / exhaustive tests.
    #[must_use]
    pub fn all() -> [RecoveryProfile; 2] {
        [RecoveryProfile::Strict, RecoveryProfile::Resilient]
    }

    /// Resolve a profile name against the closed set (looked up, never evaluated). `None` if unknown.
    #[must_use]
    pub fn resolve(s: &str) -> Option<RecoveryProfile> {
        RecoveryProfile::all().into_iter().find(|p| p.as_str() == s)
    }
}

/// The bounded retry ceiling the `resilient` profile applies (RFC-0015 §4.1 example `retry(<=3)`; I4).
pub const RESILIENT_MAX_ATTEMPTS: u64 = 3;

/// Build a [`RecoveryPolicy`] from a named [`RecoveryProfile`] over the **explicitly supplied** classes
/// (opt-in, I5). The actions are bounded (I4). This is the *only* way the automation layer produces
/// recovery — it is never derived or applied implicitly (A2).
///
/// # Errors
/// Returns [`UnknownClass`] for the first `class` not in `registry` (X1).
pub fn recovery_profile(
    profile: RecoveryProfile,
    registry: &ClassRegistry,
    classes: &[&str],
) -> Result<RecoveryPolicy, UnknownClass> {
    let mut policy = RecoveryPolicy::new();
    match profile {
        // The honest default: no recovery rules at all — everything propagates (I1).
        RecoveryProfile::Strict => {}
        // Bounded retry on each opted-in class; if every attempt fails the error still propagates.
        RecoveryProfile::Resilient => {
            for class in classes {
                policy.on(
                    registry,
                    class,
                    RecoveryAction::Retry {
                        max_attempts: RESILIENT_MAX_ATTEMPTS,
                    },
                )?;
            }
        }
    }
    Ok(policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::record::present;
    use crate::diagnostics::registry::ClassName;
    use crate::diagnostics::ReasonedError;

    fn registry() -> ClassRegistry {
        ClassRegistry::with_builtins()
    }

    // --- A4: total, deterministic derivation ---

    #[test]
    fn the_derivation_covers_every_registry_class() {
        let reg = registry();
        let policy = derive_baseline(&reg);
        // Every known class has a baseline rule (total).
        for class in reg.classes() {
            let name = reg.resolve(class).unwrap();
            assert!(policy.rule_for(&name).is_some(), "no baseline for {class}");
        }
    }

    #[test]
    fn the_derivation_is_deterministic() {
        let reg = registry();
        assert_eq!(
            derive_baseline(&reg).content_id(),
            derive_baseline(&reg).content_id()
        );
        assert_eq!(
            baseline_for_class("SwapOutOfRange"),
            baseline_for_class("SwapOutOfRange")
        );
    }

    #[test]
    fn an_unknown_class_in_a_scoped_baseline_is_explicit() {
        let reg = registry();
        assert!(derive_baseline_for(&reg, &["NotAClass"]).is_err());
        assert!(derive_baseline_for(&reg, &["SwapOutOfRange"]).is_ok());
    }

    // --- A1: additive only — a baseline can never suppress an error (I1) ---

    #[test]
    fn the_baseline_never_suppresses_the_error() {
        let reg = registry();
        let policy = derive_baseline(&reg);
        let class: ClassName = reg.resolve("SwapOutOfRange").unwrap();
        let err = ReasonedError::new(class, "swap out of range", "f/swap");
        let shown = present(err.clone(), Some(&policy));
        // I1, structural: the error is returned unchanged — it still propagates.
        assert_eq!(shown.error, err);
        // …and the baseline shaped the diagnostic (route to the durable audit for a swap refusal).
        assert_eq!(shown.diagnostic.route.as_deref(), Some("audit"));
        assert_eq!(shown.diagnostic.message, "swap out of range");
    }

    // --- A3: reified + EXPLAIN ---

    #[test]
    fn explain_names_every_class_and_reason() {
        let reg = registry();
        let ex = explain_baseline(&reg);
        assert!(
            ex.contains("SwapOutOfRange: level=Medium route=audit"),
            "{ex}"
        );
        assert!(
            ex.contains("ImplicitSwap: level=Minimal route=stream"),
            "{ex}"
        );
        // A derived policy is content-addressed (a real PolicyRef).
        assert!(derive_baseline(&reg)
            .content_id()
            .as_str()
            .starts_with("blake3:"));
    }

    // --- A2: recovery is opt-in, declared, bounded ---

    #[test]
    fn strict_profile_recovers_nothing() {
        let reg = registry();
        let p = recovery_profile(RecoveryProfile::Strict, &reg, &["SwapOutOfRange"]).unwrap();
        assert!(p.is_empty(), "strict must propagate everything");
    }

    #[test]
    fn resilient_profile_is_bounded_and_only_acts_on_opted_in_classes() {
        let reg = registry();
        let p = recovery_profile(RecoveryProfile::Resilient, &reg, &["SwapOutOfRange"]).unwrap();
        let acted = reg.resolve("SwapOutOfRange").unwrap();
        let not_acted = reg.resolve("TypeMismatch").unwrap();
        match p.action_for(&acted) {
            Some(RecoveryAction::Retry { max_attempts }) => {
                assert_eq!(*max_attempts, RESILIENT_MAX_ATTEMPTS); // bounded (I4)
            }
            other => panic!("expected bounded retry, got {other:?}"),
        }
        // A class not opted in gets no recovery — opt-in (I5).
        assert!(p.action_for(&not_acted).is_none());
    }

    #[test]
    fn profiles_resolve_through_the_closed_set() {
        assert_eq!(
            RecoveryProfile::resolve("resilient"),
            Some(RecoveryProfile::Resilient)
        );
        assert_eq!(
            RecoveryProfile::resolve("strict"),
            Some(RecoveryProfile::Strict)
        );
        assert_eq!(RecoveryProfile::resolve("yolo"), None);
    }
}
