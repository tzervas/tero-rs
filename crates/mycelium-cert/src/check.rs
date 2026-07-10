//! The **single translation-validation certificate checker** (M-210; RFC-0002 §2; RFC-0004 §3;
//! T1.1): one [`check`]`(A, B, R, claimed, evidence)` answering *"does artifact `B` refine
//! reference `A` under relation `R` within the claimed `{ε, δ, strength}`?"* — shared by
//! representation swaps (RFC-0002) **and** interpreter↔AOT equivalence (RFC-0004 §3). Build once,
//! use twice.
//!
//! The three relation instances (RFC-0002 §2):
//!
//! - [`RefinementRelation::Bijection`] — the M-120 binary↔ternary swap class. Discharges by
//!   **structural re-derivation equality**: the certificate's lemma reference and its `(n, m)`
//!   side-condition ([`legal_pair`]) are checked, then the swap is re-derived from `A` and compared
//!   payload-for-payload with `B` — the computational analogue of the SMT/`decide` discharge
//!   (RFC-0002 §4). Per-*instance* validation, cheap by construction (no per-value proof; the
//!   once-per-kind lemma is referenced, its instance recomputed). KC-4 measures this cost (M-212).
//! - [`RefinementRelation::BoundedSimilarity`] — lossy swaps (RFC-0002 §5). The measured `A`↔`B`
//!   deviation and the claim are re-validated through the **`mycelium-numerics` tier-i checker**
//!   (ADR-010 "Trusted base"): the certificate's ε must cover the *measured* deviation of this
//!   instance, and the claimed ε must not be tighter than the certificate's checked basis states
//!   (VR-5 — a claim never upgrades past its evidence). Theorem *citations* in a `ProvenThm` basis
//!   are accepted axiomatically; only the arithmetic instantiation is re-checked (RFC-0002 §7).
//! - [`RefinementRelation::ObservationalEquiv`] — interpreter↔AOT (RFC-0004 §3; folds in the M-151
//!   differential as an instance). Discharges by structural equality of the NFR-7 observable
//!   `(repr, payload, guarantee)`.
//!
//! **Never a silent pass.** Translation validation is incomplete — it may fail to validate a
//! correct artifact — so every non-validation is an explicit [`CheckVerdict::NotValidated`]
//! carrying its [`NotValidatedReason`] *and* the explicit [`Fallback`] path (RFC-0002 §2): keep the
//! reference artifact `A` (refuse the swap; run the trusted interpreter, ADR-007).

use mycelium_core::{
    BoundKind, ContentHash, CoreValue, Datum, GuaranteeStrength, NormKind, Payload, Repr, Value,
};
use mycelium_numerics::{
    basis_strength, check_error_claim, check_union_claim, Certificate, CheckOutcome, ErrorBound,
    ErrorOp, ProbBound,
};

use crate::{
    binary_to_ternary, dense_vsa, legal_pair, roundtrip_lemma_ref, ternary_to_binary,
    SwapCertificate,
};

/// The relation `R` under which `B` claims to refine `A` (RFC-0002 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementRelation {
    /// Exact-within-range bijection (the binary↔ternary class; RFC-0002 §4). Claim must be
    /// `{0, 0, Exact}`.
    Bijection,
    /// Approximate similarity-preservation within an ε bound (the lossy classes; RFC-0002 §5).
    BoundedSimilarity,
    /// Observational equivalence of two execution paths over the NFR-7 observable
    /// `(repr, payload, guarantee)` (RFC-0004 §3). Claim must be `{0, 0, Exact}`.
    ObservationalEquiv,
}

/// The evidence presented to the checker — the *certificate* of `(A, B, R, claimed, certificate)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Evidence<'a> {
    /// A swap certificate (the RFC-0002 instances, bijective or bounded).
    Swap(&'a SwapCertificate),
    /// The interp↔AOT instance (RFC-0004 §3): the claim is bare observational equality, so the
    /// artifacts themselves are the whole evidence — there is no auxiliary certificate object.
    Observational,
}

/// The explicit fallback path when validation fails — required by RFC-0002 §2 (TV incompleteness
/// must never become a silent pass *or* a silent drop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// Keep the reference artifact `A`: for a swap, refuse the converted value and continue with
    /// the source; for interp↔AOT, run the reference interpreter — the trusted base (ADR-007).
    UseReference,
}

/// Why the checker did not validate. [`Diverged`](NotValidatedReason::Diverged) is a genuine
/// counterexample; the others are explicit refusals (bad evidence, too-tight claim, or checker
/// incompleteness) — none of them is ever silent.
#[derive(Debug, Clone, PartialEq)]
pub enum NotValidatedReason {
    /// `B` does not refine `A` under `R` — the checker found a concrete divergence.
    Diverged {
        /// What diverged.
        detail: String,
    },
    /// The certificate does not bind to these artifacts/this relation (wrong kind, unknown lemma,
    /// failed side-condition, mismatched reprs/params, or a claim stronger than its basis — VR-5).
    CertificateMismatch {
        /// What failed to bind.
        detail: String,
    },
    /// The claim is **tighter** than the sound re-derivation — the tier-i rejection surfaced
    /// (ADR-010 "Trusted base"; RFC-0002 §2).
    ClaimTooTight {
        /// The bound the checker re-derives (measured deviation or certificate ε).
        recomputed: f64,
        /// The (too-tight) bound that was claimed.
        claimed: f64,
    },
    /// The checker cannot decide this instance (TV incompleteness) — *not* a counterexample. The
    /// caller must take the [`Fallback`], never assume validity.
    Incomplete {
        /// What the checker cannot decide.
        detail: String,
    },
}

/// The checker's verdict. There is no third state: an instance is validated at an honest strength,
/// or it is explicitly not — with a reason and a fallback (RFC-0002 §2).
#[derive(Debug, Clone, PartialEq)]
pub enum CheckVerdict {
    /// `B` refines `A` under `R` within the claim, established at `strength` (never stronger than
    /// the evidence's checked basis — VR-5).
    Validated {
        /// The strength at which the refinement is established.
        strength: GuaranteeStrength,
    },
    /// Not validated — reason + the explicit fallback path.
    NotValidated {
        /// Why.
        reason: NotValidatedReason,
        /// What to do instead (never silence).
        fallback: Fallback,
    },
}

fn not_validated(reason: NotValidatedReason) -> CheckVerdict {
    CheckVerdict::NotValidated {
        reason,
        fallback: Fallback::UseReference,
    }
}

fn mismatch(detail: impl Into<String>) -> CheckVerdict {
    not_validated(NotValidatedReason::CertificateMismatch {
        detail: detail.into(),
    })
}

fn diverged(detail: impl Into<String>) -> CheckVerdict {
    not_validated(NotValidatedReason::Diverged {
        detail: detail.into(),
    })
}

fn incomplete(detail: impl Into<String>) -> CheckVerdict {
    not_validated(NotValidatedReason::Incomplete {
        detail: detail.into(),
    })
}

/// The single shared checker: does artifact `B` refine reference `A` under `relation` within the
/// `claimed` `{ε, δ, strength}` certificate, given `evidence`? (RFC-0002 §2; RFC-0004 §3.)
///
/// Exact relations (bijection, observational) discharge by structural/re-derivation equality;
/// bounded relations discharge through the `mycelium-numerics` tier-i checker. Every failure is an
/// explicit [`CheckVerdict::NotValidated`] with a reason and a [`Fallback`] — never a silent pass.
#[must_use]
pub fn check(
    a: &Value,
    b: &Value,
    relation: RefinementRelation,
    claimed: Certificate,
    evidence: &Evidence<'_>,
) -> CheckVerdict {
    match relation {
        RefinementRelation::Bijection => check_bijection(a, b, claimed, evidence),
        RefinementRelation::BoundedSimilarity => check_bounded(a, b, claimed, evidence),
        RefinementRelation::ObservationalEquiv => check_observational(a, b, claimed, evidence),
    }
}

/// Bijection instance (RFC-0002 §4): lemma reference + `(n, m)` side-condition + repr/param
/// binding, then structural re-derivation equality (`enc`/`dec` re-run on `A`, compared with `B`).
fn check_bijection(
    a: &Value,
    b: &Value,
    claimed: Certificate,
    evidence: &Evidence<'_>,
) -> CheckVerdict {
    let Evidence::Swap(cert) = evidence else {
        return mismatch("Bijection requires swap-certificate evidence");
    };
    let SwapCertificate::Bijective {
        src,
        target,
        policy_used,
        lemma_ref,
        params,
    } = cert
    else {
        return mismatch("Bijection requires a Bijective certificate, got Bounded");
    };
    if claimed != Certificate::exact() {
        return mismatch("a bijective refinement claims exactly {ε: 0, δ: 0, Exact}");
    }
    if lemma_ref != &roundtrip_lemma_ref() {
        return mismatch(format!(
            "unknown bijection lemma {} (expected the M-121 round-trip lemma)",
            lemma_ref.as_str()
        ));
    }
    // The lemma's side-condition, re-checked here — `Proven` is only honored with its
    // side-conditions *checked* (the honesty rule; RFC-0002 §3).
    if !legal_pair(params.width, params.trits) {
        return mismatch(format!(
            "side-condition fails: (width {}, trits {}) is not a legal pair (B_n ⊄ T_m)",
            params.width, params.trits
        ));
    }
    if a.repr() != src {
        return mismatch("certificate src repr does not match artifact A");
    }
    if b.repr() != target {
        return mismatch("certificate target repr does not match artifact B");
    }
    let rederived = match (src, target) {
        (Repr::Binary { width }, Repr::Ternary { trits }) => {
            if *width != params.width || *trits != params.trits {
                return mismatch("certificate params do not bind its src/target reprs");
            }
            binary_to_ternary(a, *trits, policy_used)
        }
        (Repr::Ternary { trits }, Repr::Binary { width }) => {
            if *width != params.width || *trits != params.trits {
                return mismatch("certificate params do not bind its src/target reprs");
            }
            ternary_to_binary(a, *width, policy_used)
        }
        // TV incompleteness, stated: the only bijective swap kind that exists is binary↔ternary
        // (RFC-0002 §4); anything else cannot be re-derived here.
        _ => return incomplete(
            "no re-derivation for this swap kind (only binary↔ternary is bijective; RFC-0002 §4)",
        ),
    };
    match rederived {
        Err(e) => diverged(format!("re-derivation of the swap on A failed: {e}")),
        Ok((rv, _)) if rv.payload() == b.payload() => CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact,
        },
        Ok(_) => diverged("re-derived payload differs from B — B is not the swap of A"),
    }
}

/// Observational instance (RFC-0004 §3; NFR-7): structural equality of
/// `(repr, payload, guarantee)` — exactly the M-151 differential's observable.
fn check_observational(
    a: &Value,
    b: &Value,
    claimed: Certificate,
    evidence: &Evidence<'_>,
) -> CheckVerdict {
    let Evidence::Observational = evidence else {
        return mismatch("ObservationalEquiv takes Observational evidence, not a swap certificate");
    };
    if claimed != Certificate::exact() {
        return mismatch("observational equivalence claims exactly {ε: 0, δ: 0, Exact}");
    }
    if a.repr() != b.repr() {
        return diverged(format!(
            "observable reprs differ: {:?} vs {:?}",
            a.repr(),
            b.repr()
        ));
    }
    if a.payload() != b.payload() {
        return diverged("observable payloads differ");
    }
    if a.meta().guarantee() != b.meta().guarantee() {
        return diverged(format!(
            "observable guarantees differ: {:?} vs {:?}",
            a.meta().guarantee(),
            b.meta().guarantee()
        ));
    }
    CheckVerdict::Validated {
        strength: GuaranteeStrength::Exact,
    }
}

/// Observational equivalence over a whole [`CoreValue`] (RFC-0011 §4.6; NFR-7) — the M-151/M-210
/// observable **generalized from a representation [`Value`] to the data + recursion fragment**
/// (M-342). It is the *same* relation, one category up: a representation leaf is the existing
/// `(repr, payload, guarantee)` observable (`check_observational`, so path-dependent provenance is
/// excluded exactly as before); a [`Datum`] is its **constructor identity** + **meet-summary
/// guarantee** + **field-wise** observational equivalence (recursing into each field). Two values of
/// different category (a repr vs a datum) are an explicit divergence.
///
/// This lets the interp↔AOT/native differential validate **datum** results through the single shared
/// checker — closing M-302's "through the M-210 `ObservationalEquiv` checker" obligation for the
/// whole kernel corpus, not just representation results — and a mislabeled lowering (wrong
/// constructor, wrong field, weakened guarantee) is caught here, never a silent pass (NFR-7/VR-4).
#[must_use]
pub fn check_core(a: &CoreValue, b: &CoreValue) -> CheckVerdict {
    match (a, b) {
        (CoreValue::Repr(x), CoreValue::Repr(y)) => check(
            x,
            y,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        (CoreValue::Data(x), CoreValue::Data(y)) => check_data(x, y),
        (CoreValue::Repr(_), CoreValue::Data(_)) | (CoreValue::Data(_), CoreValue::Repr(_)) => {
            diverged("observable categories differ: a representation value vs a datum")
        }
    }
}

/// The datum instance of [`check_core`]: constructor identity, then the meet-summary guarantee, then
/// field-wise observational equivalence. The constructor + guarantee checks make a mislabeled or
/// guarantee-weakened lowering an explicit divergence; the field recursion bottoms out at the
/// representation leaves' exact observable.
fn check_data(a: &Datum, b: &Datum) -> CheckVerdict {
    if a.ctor() != b.ctor() {
        return diverged(format!(
            "constructors differ: {:?} vs {:?}",
            a.ctor(),
            b.ctor()
        ));
    }
    // A constructor match fixes the arity (WF6 saturation), but check defensively rather than index.
    if a.fields().len() != b.fields().len() {
        return diverged("datum arities differ for the same constructor (malformed datum)");
    }
    if a.guarantee() != b.guarantee() {
        return diverged(format!(
            "datum guarantee summaries differ: {:?} vs {:?}",
            a.guarantee(),
            b.guarantee()
        ));
    }
    // Field-wise: the first divergence propagates (with its reason), never a silent skip.
    for (x, y) in a.fields().iter().zip(b.fields()) {
        let verdict = check_core(x, y);
        if !matches!(verdict, CheckVerdict::Validated { .. }) {
            return verdict;
        }
    }
    // Structurally identical (same ctor, same guarantee, observationally-equal fields) ⇒ the
    // equivalence holds exactly, consistent with the representation-leaf verdict.
    CheckVerdict::Validated {
        strength: GuaranteeStrength::Exact,
    }
}

/// Bounded instance (RFC-0002 §5): the certificate's ε must cover the **measured** deviation of
/// this instance, and the claim must not be tighter than the certificate — both re-validated
/// through the `mycelium-numerics` tier-i checker (ADR-010).
fn check_bounded(
    a: &Value,
    b: &Value,
    claimed: Certificate,
    evidence: &Evidence<'_>,
) -> CheckVerdict {
    let Evidence::Swap(cert) = evidence else {
        return mismatch("BoundedSimilarity requires swap-certificate evidence");
    };
    let SwapCertificate::Bounded {
        src,
        target,
        policy_used,
        bound,
    } = cert
    else {
        return mismatch("BoundedSimilarity requires a Bounded certificate, got Bijective");
    };
    if a.repr() != src {
        return mismatch("certificate src repr does not match artifact A");
    }
    if b.repr() != target {
        return mismatch("certificate target repr does not match artifact B");
    }
    if claimed.strength() == GuaranteeStrength::Exact {
        return mismatch("a bounded claim cannot be Exact — Exact means no bound (M-I1)");
    }
    // VR-5: the claim may be *weaker* than the evidence's basis, never stronger.
    let evidence_strength = basis_strength(&bound.basis);
    if claimed.strength().rank() < evidence_strength.rank() {
        return mismatch(format!(
            "claimed strength {:?} upgrades past the certificate basis ({evidence_strength:?}) — VR-5",
            claimed.strength()
        ));
    }
    // δ certificates (the M-231 Dense↔VSA class) discharge by deterministic re-derivation; ε
    // certificates by measured deviation through the tier-i kernel.
    let (cert_eps, norm) = match bound.kind {
        BoundKind::Probability { delta: cert_delta } => {
            return check_bounded_prob(a, b, target, policy_used, cert_delta, claimed);
        }
        BoundKind::Error { eps, norm } => {
            if claimed.delta() != 0.0 {
                return mismatch("an ε certificate carries no δ side");
            }
            (eps, norm)
        }
        _ => {
            return incomplete(
                "only ε (ErrorBound) and δ (ProbabilityBound) certificates are checkable",
            )
        }
    };
    // The arithmetic instantiation, re-checked (RFC-0002 §7): the actual deviation of *this*
    // instance, in the certificate's own norm.
    let measured = match deviation(a, b, norm) {
        Ok(d) => d,
        Err(DeviationError::Unsupported(detail)) => return incomplete(detail),
        Err(DeviationError::Diverged(detail)) => return diverged(detail),
    };
    let Some(measured_eb) = ErrorBound::new(measured, norm) else {
        return diverged(format!(
            "measured deviation {measured} is not a finite bound — B is unboundedly far from A"
        ));
    };
    let Some(cert_eb) = ErrorBound::new(cert_eps, norm) else {
        return mismatch("certificate ε is not a well-formed bound");
    };
    let Some(claimed_eb) = ErrorBound::new(claimed.eps(), norm) else {
        return mismatch("claimed ε is not a well-formed bound");
    };
    // Tier-i (1): the certificate must cover the measured instance. A single-input `Add`
    // re-derivation is the identity, so this is "claim ≥ re-derived measurement" through the one
    // shared kernel comparison (single source of truth, incl. its tolerance).
    match check_error_claim(&[measured_eb], ErrorOp::Add, cert_eb) {
        CheckOutcome::Valid => {}
        CheckOutcome::Rejected {
            recomputed,
            claimed,
        } => {
            return not_validated(NotValidatedReason::ClaimTooTight {
                recomputed,
                claimed,
            })
        }
        CheckOutcome::Malformed => return incomplete("tier-i re-derivation was malformed"),
    }
    // Tier-i (2): the claim must not be tighter than its evidence states (VR-5).
    match check_error_claim(&[cert_eb], ErrorOp::Add, claimed_eb) {
        CheckOutcome::Valid => {}
        CheckOutcome::Rejected {
            recomputed,
            claimed,
        } => {
            return not_validated(NotValidatedReason::ClaimTooTight {
                recomputed,
                claimed,
            })
        }
        CheckOutcome::Malformed => return incomplete("tier-i re-derivation was malformed"),
    }
    CheckVerdict::Validated {
        strength: claimed.strength(),
    }
}

/// The δ instance (M-231): tier-i claim-vs-certificate through the union-bound kernel, then
/// **deterministic re-derivation equality** — the Dense↔VSA encoding/decoding is a pure function
/// of `A` and the versioned codebook, so the checker re-runs the swap (which re-checks the
/// capacity/profile side-conditions and re-derives the honest basis) and compares payloads with
/// `B`. A failed re-derivation means the certificate does not bind to this instance; a payload
/// difference is a genuine divergence; a basis stronger than the re-derivation supports is a
/// VR-5 rejection.
fn check_bounded_prob(
    a: &Value,
    b: &Value,
    target: &Repr,
    policy_used: &ContentHash,
    cert_delta: f64,
    claimed: Certificate,
) -> CheckVerdict {
    if claimed.eps() != 0.0 {
        return mismatch("a δ certificate carries no ε side");
    }
    let (Some(cert_pb), Some(claimed_pb)) =
        (ProbBound::new(cert_delta), ProbBound::new(claimed.delta()))
    else {
        return mismatch("certificate/claimed δ is not a well-formed probability");
    };
    // Tier-i: the claim must not be tighter than the certificate (VR-5), through the one shared
    // union-bound kernel comparison.
    match check_union_claim(&[cert_pb], claimed_pb) {
        CheckOutcome::Valid => {}
        CheckOutcome::Rejected {
            recomputed,
            claimed,
        } => {
            return not_validated(NotValidatedReason::ClaimTooTight {
                recomputed,
                claimed,
            })
        }
        CheckOutcome::Malformed => return incomplete("tier-i δ re-derivation was malformed"),
    }
    // Re-derive the swap on A (re-checking its side-conditions and honest basis as it goes).
    let rederived = match (a.repr(), target) {
        (Repr::Dense { .. }, Repr::Vsa { dim, .. }) => {
            dense_vsa::dense_to_vsa(a, *dim, cert_delta, policy_used)
        }
        (Repr::Vsa { .. }, Repr::Dense { dim, .. }) => {
            dense_vsa::vsa_to_dense(a, *dim, cert_delta, policy_used)
        }
        // TV incompleteness, stated: Dense↔VSA is the only δ-certified swap class (M-231).
        _ => return incomplete("no δ re-derivation for this swap kind (only Dense↔VSA, M-231)"),
    };
    match rederived {
        Err(e) => mismatch(format!(
            "certificate does not bind: re-derivation of the swap on A refused: {e}"
        )),
        Ok((rv, rcert)) => {
            // The evidence basis must not be stronger than the honest re-derived one (VR-5).
            if let SwapCertificate::Bounded {
                bound: ref rebound, ..
            } = rcert
            {
                if claimed.strength().rank() < basis_strength(&rebound.basis).rank() {
                    return mismatch(format!(
                        "claimed strength {:?} upgrades past the re-derived basis ({:?}) — VR-5",
                        claimed.strength(),
                        basis_strength(&rebound.basis)
                    ));
                }
            }
            if rv.payload() == b.payload() {
                CheckVerdict::Validated {
                    strength: claimed.strength(),
                }
            } else {
                diverged("re-derived payload differs from B — B is not the swap of A")
            }
        }
    }
}

/// How a deviation measurement can fail — split so the caller maps it honestly: `Unsupported` is
/// checker incompleteness, `Diverged` is a genuine counterexample.
enum DeviationError {
    Unsupported(String),
    Diverged(String),
}

/// The numeric payload slice of a value, if it has one (`Scalars`/`Hypervector`).
fn numeric_payload(v: &Value) -> Option<&[f64]> {
    match v.payload() {
        Payload::Scalars(xs) => Some(xs),
        Payload::Hypervector(xs) => Some(xs),
        // A sequence (RFC-0032 D3) and a byte string (RFC-0032 D4) have no flat numeric payload here
        // — explicitly None, not coerced. The scalar float (ADR-040; M-896) is deliberately None
        // too: no float swap/deviation metric is ratified yet (the float op/swap surface is
        // M-898+), so the checker reports honest incompleteness (`Unsupported`) rather than
        // improvising a metric over the in-band specials (NaN/±inf) — never a guessed number (G2).
        Payload::Bits(_) | Payload::Trits(_) | Payload::Seq(_) | Payload::Bytes(_) => None,
        Payload::Float(_) => None,
    }
}

/// The actual `A`↔`B` deviation in `norm`. `Rel` is the elementwise maximum relative deviation
/// (`0/0` contributes `0`; a nonzero `B` element against a zero reference is unbounded — an
/// explicit divergence, never a coerced number).
fn deviation(a: &Value, b: &Value, norm: NormKind) -> Result<f64, DeviationError> {
    let xs = numeric_payload(a).ok_or_else(|| {
        DeviationError::Unsupported("no deviation metric for non-numeric payloads here".to_owned())
    })?;
    let ys = numeric_payload(b).ok_or_else(|| {
        DeviationError::Unsupported("no deviation metric for non-numeric payloads here".to_owned())
    })?;
    if xs.len() != ys.len() {
        return Err(DeviationError::Diverged(format!(
            "payload lengths differ: {} vs {}",
            xs.len(),
            ys.len()
        )));
    }
    let diffs = xs.iter().zip(ys.iter()).map(|(x, y)| (x - y).abs());
    Ok(match norm {
        NormKind::L1 => diffs.sum(),
        NormKind::L2 => diffs.map(|d| d * d).sum::<f64>().sqrt(),
        NormKind::Linf => diffs.fold(0.0_f64, f64::max),
        NormKind::Rel => {
            let mut worst = 0.0_f64;
            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = (x - y).abs();
                if *x == 0.0 {
                    if d != 0.0 {
                        return Err(DeviationError::Diverged(
                            "relative deviation undefined: reference element is 0, candidate is not"
                                .to_owned(),
                        ));
                    }
                } else {
                    worst = worst.max(d / x.abs());
                }
            }
            worst
        }
    })
}
