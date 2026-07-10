//! `mycelium-build` — the **stable-component build layer** (M-311; RFC-0004 §4; ADR-003/009; KC-3).
//!
//! RFC-0004 §4 (normative): a definition is a *stable component*, and thus **AOT-eligible**, iff
//! (1) it is content-addressed and hash-frozen (Unison identity, ADR-003); (2) its spec is ratified;
//! (3) its verification obligations (swap certificates, bound checks, reference equivalence) are
//! discharged. **Promotion is an explicit act gated on automatic checks** — the checks must pass,
//! but marking-stable is deliberate. Everything else runs interpreted/JIT (ADR-009).
//!
//! This crate makes that gate executable and inspectable:
//! - [`check_eligibility`] runs the automatic §4 checks, returning the **specific** blocking reasons
//!   (never a silent "not eligible" — G2);
//! - [`decide`] routes a component (AOT only for an *eligible, explicitly promoted* one) and emits a
//!   [`BuildCertificate`] — the content-addressed ([`BuildCertificate::cert_ref`]), inspectable
//!   record of the decision (ADR-003);
//! - the certificate's invariants are re-checked on `Deserialize` (banked guard 3): a tampered
//!   certificate claiming `Aot` without discharged obligations is **rejected on the way in**, so an
//!   AOT route can never be forged from untrusted input.
//!
//! It lives **outside** the trusted kernel (KC-3): it depends only on `mycelium-core` for the
//! `ContentHash` identity, and nothing in the kernel depends on it.

use mycelium_core::{operation_hash, ContentHash};
use serde::{Deserialize, Deserializer, Serialize};

pub mod cache;
pub mod target;
pub use cache::{BuildCache, CacheOutcome};
pub use target::{
    realizable_targets, supported_targets, Arch, BuildError, BuildProfile, DispatchMiss, Os,
    Target, VariantTable,
};

/// The execution route a component takes (RFC-0004 §4 / ADR-009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRoute {
    /// AOT-compiled native code — **only** a promoted stable component (RFC-0004 §4).
    Aot,
    /// Interpreted/JIT — everything else (the safe default; ADR-009).
    Interpreted,
}

/// The RFC-0004 §4 verification obligations for a definition. Each is *discharged elsewhere* — swap
/// certificates by `mycelium-cert` (RFC-0002/M-210), bound checks by the ADR-010 tier-i checker,
/// reference equivalence by the interp↔AOT differential (NFR-7) — and recorded here as a checked
/// fact. The honest default is [`Obligations::none`] (nothing discharged).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Obligations {
    /// Every swap certificate validates through the shared checker (RFC-0002 / M-210).
    pub swap_certificates_valid: bool,
    /// Every bound check is discharged through the tier-i checker (ADR-010).
    pub bound_checks_discharged: bool,
    /// Interpreter↔AOT reference equivalence is checked (NFR-7 / RR-12).
    pub reference_equivalence_checked: bool,
}

impl Obligations {
    /// All three §4(3) obligations discharged.
    #[must_use]
    pub const fn all_discharged(self) -> bool {
        self.swap_certificates_valid
            && self.bound_checks_discharged
            && self.reference_equivalence_checked
    }

    /// The honest default: nothing discharged yet.
    #[must_use]
    pub const fn none() -> Self {
        Obligations {
            swap_certificates_valid: false,
            bound_checks_discharged: false,
            reference_equivalence_checked: false,
        }
    }

    /// All three discharged — the verified state.
    #[must_use]
    pub const fn all() -> Self {
        Obligations {
            swap_certificates_valid: true,
            bound_checks_discharged: true,
            reference_equivalence_checked: true,
        }
    }
}

/// A candidate definition for the stable/experimental decision (RFC-0004 §4). The `hash` is the
/// content-addressed, hash-frozen identity (§4(1), ADR-003) — by construction a [`ContentHash`], so
/// condition (1) is structural here and the gate checks (2) and (3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    /// (1) content-addressed identity (ADR-003).
    pub hash: ContentHash,
    /// (2) the spec governing this definition is ratified.
    pub spec_ratified: bool,
    /// (3) the verification obligations.
    pub obligations: Obligations,
}

/// The automatic-check verdict (RFC-0004 §4): whether the §4 conditions hold. Promotion is *gated*
/// on [`Eligibility::Eligible`] but is a separate, deliberate act ([`decide`]'s `promote`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Eligibility {
    /// All §4 conditions hold — eligible for the explicit promotion act.
    Eligible,
    /// One or more conditions fail; the specific reasons (never a silent refusal — G2).
    Blocked(Vec<&'static str>),
}

/// Run the automatic §4 eligibility checks. Condition (1) is structural (a [`ContentHash`] is by
/// construction content-addressed/hash-frozen); this checks (2) spec ratification and (3) the
/// discharged obligations, returning every failing reason.
#[must_use]
pub fn check_eligibility(c: &Component) -> Eligibility {
    let mut reasons = Vec::new();
    if !c.spec_ratified {
        reasons.push("spec not ratified (RFC-0004 §4(2))");
    }
    if !c.obligations.swap_certificates_valid {
        reasons.push("swap certificates not all valid (RFC-0002 / M-210)");
    }
    if !c.obligations.bound_checks_discharged {
        reasons.push("bound checks not discharged (ADR-010 tier-i)");
    }
    if !c.obligations.reference_equivalence_checked {
        reasons.push("interp↔AOT reference equivalence not checked (NFR-7)");
    }
    if reasons.is_empty() {
        Eligibility::Eligible
    } else {
        Eligibility::Blocked(reasons)
    }
}

/// An inspectable, content-addressed record of one build decision (RFC-0004 §4; ADR-003). Fields are
/// private so an *inconsistent* certificate is unrepresentable (banked guard 2): the only ways to
/// obtain one are [`decide`] and a re-validating [`Deserialize`] (guard 3) — a tampered certificate
/// claiming `Aot` without discharged obligations is rejected on the way in, so an AOT route can never
/// be forged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildCertificate {
    /// The component's content address (ADR-003).
    component: ContentHash,
    /// Whether the spec is ratified (§4(2)).
    spec_ratified: bool,
    /// The §4(3) obligations.
    obligations: Obligations,
    /// Whether the automatic §4 checks passed (eligible for promotion).
    eligible: bool,
    /// Whether promotion to stable was explicitly requested *and* granted (gated on `eligible`).
    promoted: bool,
    /// The resulting execution route.
    route: ExecutionRoute,
    /// The specific blocking reasons when not eligible (empty when eligible).
    blocked_on: Vec<String>,
}

impl BuildCertificate {
    /// The component this certifies.
    #[must_use]
    pub fn component(&self) -> &ContentHash {
        &self.component
    }
    /// Whether the spec is ratified.
    #[must_use]
    pub fn spec_ratified(&self) -> bool {
        self.spec_ratified
    }
    /// The recorded obligations.
    #[must_use]
    pub fn obligations(&self) -> Obligations {
        self.obligations
    }
    /// Whether the automatic §4 checks passed.
    #[must_use]
    pub fn eligible(&self) -> bool {
        self.eligible
    }
    /// Whether the component was explicitly promoted to stable.
    #[must_use]
    pub fn promoted(&self) -> bool {
        self.promoted
    }
    /// The execution route.
    #[must_use]
    pub fn route(&self) -> ExecutionRoute {
        self.route
    }
    /// The specific blocking reasons (empty when eligible).
    #[must_use]
    pub fn blocked_on(&self) -> &[String] {
        &self.blocked_on
    }

    /// The **content address** of this certificate (ADR-003 / RFC-0001 §4.6): the BLAKE3 hash of its
    /// canonical serialization, so a build decision is answerable and de-duplicable by hash. Two
    /// certificates with the same fields hash to the same `cert_ref` (the M-312 cache key).
    #[must_use]
    pub fn cert_ref(&self) -> ContentHash {
        let canonical = serde_json::to_string(self).expect("BuildCertificate serializes");
        operation_hash(&format!("build-certificate.v1:{canonical}"))
    }

    /// Re-check the certificate's internal consistency (the invariant a forged certificate would
    /// violate). Used by the validating [`Deserialize`].
    fn validate(&self) -> Result<(), &'static str> {
        // `eligible` must be exactly the §4 verdict for (spec_ratified, obligations).
        let computed_eligible = self.spec_ratified && self.obligations.all_discharged();
        if self.eligible != computed_eligible {
            return Err("`eligible` disagrees with (spec_ratified, obligations) — tampered");
        }
        // `blocked_on` is non-empty iff not eligible (eligible ⇒ no reasons).
        if self.blocked_on.is_empty() != self.eligible {
            return Err("`blocked_on` inconsistent with `eligible`");
        }
        // Promotion is gated on eligibility.
        if self.promoted && !self.eligible {
            return Err("`promoted` set on an ineligible component (promotion is gated, §4)");
        }
        // An AOT route requires an eligible, promoted component (the forge guard).
        match self.route {
            ExecutionRoute::Aot if !(self.eligible && self.promoted) => {
                Err("`Aot` route without eligible+promoted — forged AOT route (§4)")
            }
            _ => Ok(()),
        }
    }
}

/// Re-validate every certificate on deserialize (banked guard 3): the wire form re-runs the §4
/// consistency invariant, so a hand-edited certificate that claims an AOT route without discharged
/// obligations is **rejected**, never trusted.
impl<'de> Deserialize<'de> for BuildCertificate {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            component: ContentHash,
            spec_ratified: bool,
            obligations: Obligations,
            eligible: bool,
            promoted: bool,
            route: ExecutionRoute,
            blocked_on: Vec<String>,
        }
        let w = Wire::deserialize(deserializer)?;
        let cert = BuildCertificate {
            component: w.component,
            spec_ratified: w.spec_ratified,
            obligations: w.obligations,
            eligible: w.eligible,
            promoted: w.promoted,
            route: w.route,
            blocked_on: w.blocked_on,
        };
        cert.validate().map_err(serde::de::Error::custom)?;
        Ok(cert)
    }
}

/// Decide a component's build and emit its certificate (RFC-0004 §4). `promote` is the **explicit,
/// deliberate** marking-stable act: an eligible component runs interpreted/JIT *unless* `promote` is
/// requested, and a `promote` request for an ineligible component is **refused** (the route stays
/// `Interpreted` and the blocking reasons are recorded) — never a silent AOT (G2).
#[must_use]
pub fn decide(c: &Component, promote: bool) -> BuildCertificate {
    let eligibility = check_eligibility(c);
    let (eligible, blocked_on) = match &eligibility {
        Eligibility::Eligible => (true, Vec::new()),
        Eligibility::Blocked(reasons) => (false, reasons.iter().map(|s| (*s).to_owned()).collect()),
    };
    let promoted = eligible && promote;
    let route = if promoted {
        ExecutionRoute::Aot
    } else {
        ExecutionRoute::Interpreted
    };
    BuildCertificate {
        component: c.hash.clone(),
        spec_ratified: c.spec_ratified,
        obligations: c.obligations,
        eligible,
        promoted,
        route,
        blocked_on,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(tag: &str) -> ContentHash {
        operation_hash(tag)
    }

    fn verified(spec_ratified: bool) -> Component {
        Component {
            hash: hash("def.verified"),
            spec_ratified,
            obligations: Obligations::all(),
        }
    }

    #[test]
    fn eligible_and_promoted_routes_to_aot() {
        let cert = decide(&verified(true), true);
        assert!(cert.eligible());
        assert!(cert.promoted());
        assert_eq!(cert.route(), ExecutionRoute::Aot);
        assert!(cert.blocked_on().is_empty());
    }

    #[test]
    fn eligible_but_not_promoted_stays_interpreted() {
        // Promotion is deliberate (§4): an eligible component is NOT auto-promoted.
        // Mutant-witness: if `decide` promoted whenever eligible, the route would be Aot here.
        let cert = decide(&verified(true), false);
        assert!(cert.eligible());
        assert!(!cert.promoted());
        assert_eq!(cert.route(), ExecutionRoute::Interpreted);
        assert!(cert.blocked_on().is_empty());
    }

    #[test]
    fn promoting_an_unverified_component_is_refused_not_silent() {
        // Mutant-witness: dropping the eligibility gate would let an unverified component reach Aot.
        let mut c = verified(true);
        c.obligations.reference_equivalence_checked = false;
        let cert = decide(&c, true);
        assert!(!cert.eligible());
        assert!(!cert.promoted());
        assert_eq!(cert.route(), ExecutionRoute::Interpreted);
        assert!(
            cert.blocked_on()
                .iter()
                .any(|r| r.contains("reference equivalence")),
            "the specific blocking reason must be surfaced, got {:?}",
            cert.blocked_on()
        );
    }

    #[test]
    fn unratified_spec_blocks_promotion() {
        let cert = decide(&verified(false), true);
        assert!(!cert.eligible());
        assert_eq!(cert.route(), ExecutionRoute::Interpreted);
        assert!(cert
            .blocked_on()
            .iter()
            .any(|r| r.contains("spec not ratified")));
    }

    #[test]
    fn certificate_round_trips_and_is_content_addressed() {
        let cert = decide(&verified(true), true);
        let json = serde_json::to_string(&cert).unwrap();
        let back: BuildCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(cert, back);
        // Content address is deterministic and re-derivable.
        assert_eq!(cert.cert_ref(), back.cert_ref());
        assert_eq!(cert.cert_ref().algo(), "blake3");
    }

    #[test]
    fn a_forged_aot_certificate_is_rejected_on_deserialize() {
        // Hand-craft a certificate claiming an Aot route without discharged obligations — the
        // validating deserialize must reject it (banked guard 3; the forge guard).
        // Mutant-witness: dropping `validate()` in Deserialize would let this through.
        let forged = serde_json::json!({
            "component": "blake3:deadbeef",
            "spec_ratified": false,
            "obligations": {
                "swap_certificates_valid": false,
                "bound_checks_discharged": false,
                "reference_equivalence_checked": false
            },
            "eligible": true,
            "promoted": true,
            "route": "Aot",
            "blocked_on": []
        })
        .to_string();
        let err = serde_json::from_str::<BuildCertificate>(&forged)
            .expect_err("a forged AOT certificate must be rejected");
        assert!(
            err.to_string().contains("disagrees") || err.to_string().contains("forged"),
            "got: {err}"
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // deny_unknown_fields: a stray field is a tamper signal, not silently ignored.
        let extra = serde_json::json!({
            "component": "blake3:abc",
            "spec_ratified": true,
            "obligations": Obligations::all(),
            "eligible": true,
            "promoted": false,
            "route": "Interpreted",
            "blocked_on": [],
            "sneaky": 1
        })
        .to_string();
        assert!(serde_json::from_str::<BuildCertificate>(&extra).is_err());
    }
}
