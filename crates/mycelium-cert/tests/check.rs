//! M-210 acceptance — the single shared TV checker (RFC-0002 §2; RFC-0004 §3): the bijective
//! (M-120) instance discharges by re-derivation equality, the bounded instance through the E2-4
//! tier-i kernel, observational equivalence by structural equality of the NFR-7 observable; every
//! failure mode is an explicit `NotValidated{reason, fallback}` — never a silent pass.

use mycelium_cert::{
    binary_to_ternary, check, check_core, roundtrip_lemma_ref, ternary_to_binary, BinTernParams,
    CheckVerdict, Evidence, Fallback, NotValidatedReason, RefinementRelation, SwapCertificate,
};
use mycelium_core::{
    binary, operation_hash, ternary, Bound, BoundBasis, BoundKind, ContentHash, CoreValue, CtorRef,
    Datum, GuaranteeStrength, Meta, NormKind, Payload, Provenance, Repr, ScalarKind, Value,
};
use mycelium_numerics::Certificate;

fn policy() -> ContentHash {
    ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

fn byte_of(value: i64) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(binary::int_to_bits(value, 8).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn tern_of(value: i64, trits: u32) -> Value {
    Value::new(
        Repr::Ternary { trits },
        Payload::Trits(ternary::int_to_trits(value, trits).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn dense(xs: Vec<f64>, dtype: ScalarKind, meta: Meta) -> Value {
    Value::new(
        Repr::Dense {
            dim: u32::try_from(xs.len()).unwrap(),
            dtype,
        },
        Payload::Scalars(xs),
        meta,
    )
    .unwrap()
}

fn validated_exact() -> CheckVerdict {
    CheckVerdict::Validated {
        strength: GuaranteeStrength::Exact,
    }
}

/// Every `NotValidated` must carry the explicit fallback (RFC-0002 §2) — assert and extract.
fn reason_of(v: CheckVerdict) -> NotValidatedReason {
    match v {
        CheckVerdict::NotValidated { reason, fallback } => {
            assert_eq!(fallback, Fallback::UseReference);
            reason
        }
        CheckVerdict::Validated { .. } => panic!("expected NotValidated, got Validated"),
    }
}

// ---------- Bijection instance (the M-120 cert validates through the one checker) ----------

/// Both directions of the M-120 bijective swap validate, exhaustively over every byte (8↔6).
#[test]
fn bijective_cert_validates_both_directions() {
    for v in -128..=127 {
        let a = byte_of(v);
        let (b, cert) = binary_to_ternary(&a, 6, &policy()).unwrap();
        assert_eq!(
            check(
                &a,
                &b,
                RefinementRelation::Bijection,
                Certificate::exact(),
                &Evidence::Swap(&cert),
            ),
            validated_exact(),
            "enc instance for {v}"
        );
        let (back, dec_cert) = ternary_to_binary(&b, 8, &policy()).unwrap();
        assert_eq!(
            check(
                &b,
                &back,
                RefinementRelation::Bijection,
                Certificate::exact(),
                &Evidence::Swap(&dec_cert),
            ),
            validated_exact(),
            "dec instance for {v}"
        );
    }
}

/// A tampered target payload is a genuine counterexample — `Diverged`, never a pass.
#[test]
fn tampered_target_diverges() {
    let a = byte_of(42);
    let (_, cert) = binary_to_ternary(&a, 6, &policy()).unwrap();
    let forged = tern_of(43, 6); // wrong value under the same cert
    let reason = reason_of(check(
        &a,
        &forged,
        RefinementRelation::Bijection,
        Certificate::exact(),
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(reason, NotValidatedReason::Diverged { .. }));
}

/// An unknown lemma reference cannot bind — `CertificateMismatch`.
#[test]
fn unknown_lemma_is_a_mismatch() {
    let a = byte_of(7);
    let (b, _) = binary_to_ternary(&a, 6, &policy()).unwrap();
    let forged = SwapCertificate::Bijective {
        src: a.repr().clone(),
        target: b.repr().clone(),
        policy_used: policy(),
        lemma_ref: operation_hash("lemma.not.the.real.one"),
        params: BinTernParams { width: 8, trits: 6 },
    };
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::Bijection,
        Certificate::exact(),
        &Evidence::Swap(&forged),
    ));
    assert!(matches!(
        reason,
        NotValidatedReason::CertificateMismatch { .. }
    ));
}

/// A certificate whose `(n, m)` fails the legal-pair side-condition is rejected — `Proven` is
/// honored only with checked side-conditions (the honesty rule).
#[test]
fn failed_side_condition_is_a_mismatch() {
    let a = byte_of(0);
    let b = tern_of(0, 4);
    let forged = SwapCertificate::Bijective {
        src: a.repr().clone(),
        target: b.repr().clone(),
        policy_used: policy(),
        lemma_ref: roundtrip_lemma_ref(),
        params: BinTernParams { width: 8, trits: 4 }, // B_8 ⊄ T_4
    };
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::Bijection,
        Certificate::exact(),
        &Evidence::Swap(&forged),
    ));
    assert!(matches!(
        reason,
        NotValidatedReason::CertificateMismatch { .. }
    ));
}

/// A bijective claim must be exactly `{0, 0, Exact}` — anything else cannot bind.
#[test]
fn bijection_with_nonexact_claim_is_a_mismatch() {
    let a = byte_of(1);
    let (b, cert) = binary_to_ternary(&a, 6, &policy()).unwrap();
    let claimed = Certificate::new(0.25, 0.0, GuaranteeStrength::Proven).unwrap();
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::Bijection,
        claimed,
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(
        reason,
        NotValidatedReason::CertificateMismatch { .. }
    ));
}

/// Wrong evidence kind for the relation is explicit, both ways.
#[test]
fn relation_evidence_mismatch_is_explicit() {
    let a = byte_of(1);
    let (b, cert) = binary_to_ternary(&a, 6, &policy()).unwrap();
    let r = reason_of(check(
        &a,
        &b,
        RefinementRelation::Bijection,
        Certificate::exact(),
        &Evidence::Observational,
    ));
    assert!(matches!(r, NotValidatedReason::CertificateMismatch { .. }));
    let r = reason_of(check(
        &a,
        &a,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(r, NotValidatedReason::CertificateMismatch { .. }));
}

// ---------- Observational instance (RFC-0004 §3; the M-151 differential's relation) ----------

#[test]
fn observational_equality_validates() {
    let a = byte_of(99);
    let b = byte_of(99);
    assert_eq!(
        check(
            &a,
            &b,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        validated_exact()
    );
}

#[test]
fn observational_payload_divergence_is_caught() {
    let a = byte_of(99);
    let b = byte_of(98);
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    ));
    assert!(matches!(reason, NotValidatedReason::Diverged { .. }));
}

/// Guarantee strength is part of the NFR-7 observable: same payload, different honesty tag ⇒
/// divergence (two paths must never mean two *disclosures*).
#[test]
fn observational_guarantee_divergence_is_caught() {
    let xs = vec![1.0, 2.0];
    let a = dense(xs.clone(), ScalarKind::F32, Meta::exact(Provenance::Root));
    let declared = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(Bound {
            kind: BoundKind::Error {
                eps: 0.1,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::UserDeclared,
        }),
        None,
        None,
        None,
    )
    .unwrap();
    let b = dense(xs, ScalarKind::F32, declared);
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    ));
    assert!(matches!(reason, NotValidatedReason::Diverged { .. }));
}

// ---------- Bounded instance (synthetic certs; the real F32→BF16 swap is tested in dense.rs) ----

fn bounded_cert(src: &Value, target: &Value, eps: f64, basis: BoundBasis) -> SwapCertificate {
    SwapCertificate::Bounded {
        src: src.repr().clone(),
        target: target.repr().clone(),
        policy_used: policy(),
        bound: Bound {
            kind: BoundKind::Error {
                eps,
                norm: NormKind::Linf,
            },
            basis,
        },
    }
}

fn empirical() -> BoundBasis {
    BoundBasis::EmpiricalFit {
        trials: 10_000,
        method: "synthetic test fixture".to_owned(),
    }
}

/// A sound bounded certificate over the measured instance validates at the claimed strength.
#[test]
fn bounded_cert_validates_when_it_covers_the_measured_deviation() {
    let a = dense(
        vec![1.0, 2.0, 3.0],
        ScalarKind::F32,
        Meta::exact(Provenance::Root),
    );
    let b = dense(
        vec![1.01, 2.0, 2.99],
        ScalarKind::Bf16,
        Meta::exact(Provenance::Root),
    );
    let cert = bounded_cert(&a, &b, 0.05, empirical());
    let claimed = Certificate::new(0.05, 0.0, GuaranteeStrength::Empirical).unwrap();
    assert_eq!(
        check(
            &a,
            &b,
            RefinementRelation::BoundedSimilarity,
            claimed,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Empirical
        }
    );
}

/// A certificate tighter than the measured instance is the tier-i rejection, surfaced.
#[test]
fn bounded_cert_tighter_than_reality_is_rejected() {
    let a = dense(vec![1.0], ScalarKind::F32, Meta::exact(Provenance::Root));
    let b = dense(vec![1.5], ScalarKind::Bf16, Meta::exact(Provenance::Root));
    let cert = bounded_cert(&a, &b, 0.1, empirical()); // actual Linf deviation is 0.5
    let claimed = Certificate::new(0.1, 0.0, GuaranteeStrength::Empirical).unwrap();
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    ));
    match reason {
        NotValidatedReason::ClaimTooTight {
            recomputed,
            claimed,
        } => {
            assert!((recomputed - 0.5).abs() < 1e-12);
            assert!((claimed - 0.1).abs() < 1e-12);
        }
        other => panic!("expected ClaimTooTight, got {other:?}"),
    }
}

/// A claim tighter than the certificate's stated ε is rejected even when the measured instance
/// would pass — the claim never outruns its checked evidence (VR-5).
#[test]
fn claim_tighter_than_certificate_is_rejected() {
    let a = dense(vec![1.0], ScalarKind::F32, Meta::exact(Provenance::Root));
    let b = dense(vec![1.0], ScalarKind::Bf16, Meta::exact(Provenance::Root)); // measured 0
    let cert = bounded_cert(&a, &b, 0.1, empirical());
    let claimed = Certificate::new(0.01, 0.0, GuaranteeStrength::Empirical).unwrap();
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(reason, NotValidatedReason::ClaimTooTight { .. }));
}

/// A claimed strength stronger than the certificate basis supports is a VR-5 upgrade — rejected.
#[test]
fn strength_upgrade_past_the_basis_is_rejected() {
    let a = dense(vec![1.0], ScalarKind::F32, Meta::exact(Provenance::Root));
    let b = dense(vec![1.0], ScalarKind::Bf16, Meta::exact(Provenance::Root));
    let cert = bounded_cert(&a, &b, 0.1, empirical()); // Empirical basis
    let claimed = Certificate::new(0.1, 0.0, GuaranteeStrength::Proven).unwrap(); // upgrade attempt
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(
        reason,
        NotValidatedReason::CertificateMismatch { .. }
    ));
}

/// A δ claim against an ε certificate is a certificate mismatch — the two sides are distinct
/// kernels and never silently mixed (ADR-010; the δ class itself is checked via the M-231
/// Dense↔VSA instance, `tests/dense_vsa.rs`).
#[test]
fn delta_claims_against_an_eps_certificate_are_explicit() {
    let a = dense(vec![1.0], ScalarKind::F32, Meta::exact(Provenance::Root));
    let b = dense(vec![1.0], ScalarKind::Bf16, Meta::exact(Provenance::Root));
    let cert = bounded_cert(&a, &b, 0.1, empirical());
    let claimed = Certificate::new(0.1, 0.01, GuaranteeStrength::Empirical).unwrap();
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(
        reason,
        NotValidatedReason::CertificateMismatch { .. }
    ));
}

/// A bounded certificate over a payload with no deviation metric is `Incomplete`, not a guess.
#[test]
fn bounded_over_nonnumeric_payload_is_incomplete() {
    let a = byte_of(1);
    let b = byte_of(1);
    let cert = bounded_cert(&a, &b, 0.1, empirical());
    let claimed = Certificate::new(0.1, 0.0, GuaranteeStrength::Empirical).unwrap();
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(reason, NotValidatedReason::Incomplete { .. }));
}

/// `Rel`-norm deviation against a zero reference element is unbounded — an explicit divergence.
#[test]
fn relative_deviation_from_zero_reference_diverges() {
    let a = dense(
        vec![0.0, 1.0],
        ScalarKind::F32,
        Meta::exact(Provenance::Root),
    );
    let b = dense(
        vec![0.5, 1.0],
        ScalarKind::Bf16,
        Meta::exact(Provenance::Root),
    );
    let cert = SwapCertificate::Bounded {
        src: a.repr().clone(),
        target: b.repr().clone(),
        policy_used: policy(),
        bound: Bound {
            kind: BoundKind::Error {
                eps: 0.9,
                norm: NormKind::Rel,
            },
            basis: empirical(),
        },
    };
    let claimed = Certificate::new(0.9, 0.0, GuaranteeStrength::Empirical).unwrap();
    let reason = reason_of(check(
        &a,
        &b,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    ));
    assert!(matches!(reason, NotValidatedReason::Diverged { .. }));
}

// ---------- check_core: ObservationalEquiv over the data + recursion fragment (M-302) ----------

/// A shared `Nat`-like declaration hash; constructor 0 = `Z`, constructor 1 = `S(Nat)`.
fn nat_decl() -> ContentHash {
    ContentHash::parse("blake3:natdecl").unwrap()
}
fn z() -> CoreValue {
    CoreValue::Data(Datum::new(CtorRef::new(nat_decl(), 0), vec![]))
}
fn s(inner: CoreValue) -> CoreValue {
    CoreValue::Data(Datum::new(CtorRef::new(nat_decl(), 1), vec![inner]))
}

#[test]
fn check_core_validates_structurally_equal_datums() {
    // S(S(Z)) ≡ S(S(Z)) — same constructors, same fields, all-Exact summary ⇒ Validated{Exact}.
    assert_eq!(check_core(&s(s(z())), &s(s(z()))), validated_exact());
}

#[test]
fn check_core_on_a_repr_leaf_agrees_with_check_observational() {
    // A representation CoreValue routes to the existing observable — same verdict as `check`.
    let a = CoreValue::Repr(byte_of(99));
    let b = CoreValue::Repr(byte_of(99));
    assert_eq!(check_core(&a, &b), validated_exact());
    let c = CoreValue::Repr(byte_of(98));
    assert!(matches!(
        reason_of(check_core(&a, &c)),
        NotValidatedReason::Diverged { .. }
    ));
}

#[test]
fn check_core_catches_a_wrong_constructor() {
    // Mutant-witness: a lowering that built `Z` where the reference built `S(Z)` (wrong arm) is an
    // explicit divergence, never a silent pass (NFR-7/VR-4).
    assert!(matches!(
        reason_of(check_core(&s(z()), &z())),
        NotValidatedReason::Diverged { .. }
    ));
}

#[test]
fn check_core_catches_a_divergent_field_deep_in_the_tree() {
    // Mutant-witness at depth: S(S(Z)) vs S(Z) — the constructors agree at the root and the first S,
    // but the inner field diverges (S(Z) vs Z). The recursion must surface it.
    assert!(matches!(
        reason_of(check_core(&s(s(z())), &s(z()))),
        NotValidatedReason::Diverged { .. }
    ));
}

#[test]
fn check_core_catches_a_repr_field_divergence() {
    // A datum carrying a *representation* field — here the unary constructor (index 1 of the shared
    // decl) wrapping a `Binary{8}` value, the `Box = Mk(Binary{8})` shape: a wrong field byte is
    // caught at the representation leaf through the existing exact observable.
    let mk = |b: Value| CoreValue::Data(Datum::new(CtorRef::new(nat_decl(), 1), vec![b.into()]));
    assert_eq!(
        check_core(&mk(byte_of(10)), &mk(byte_of(10))),
        validated_exact()
    );
    assert!(matches!(
        reason_of(check_core(&mk(byte_of(10)), &mk(byte_of(11)))),
        NotValidatedReason::Diverged { .. }
    ));
}

#[test]
fn check_core_catches_a_category_mismatch() {
    // A representation value vs a datum are different observable categories — an explicit divergence,
    // not a coincidental pass.
    let repr = CoreValue::Repr(byte_of(0));
    assert!(matches!(
        reason_of(check_core(&repr, &z())),
        NotValidatedReason::Diverged { .. }
    ));
}
