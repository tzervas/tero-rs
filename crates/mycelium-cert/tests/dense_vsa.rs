//! M-231 — the Dense↔VSA bounded swap: certificates carry an honestly-derived δ (`ProvenThm`
//! where the capacity instantiation checks, `EmpiricalFit` where the trial-validated profile
//! covers, an explicit refusal elsewhere — RFC-0002 §5), both directions validate through the
//! single M-210 checker (SC-2/SC-3), and the declared empirical profile is exercised with
//! exactly its declared trial count.

use mycelium_cert::{
    check, dense_to_vsa, vsa_to_dense, CheckVerdict, Evidence, NotValidatedReason,
    RefinementRelation, SwapCertificate, SwapError, DENSE_VSA_EMP_DELTA,
};
use mycelium_core::{
    BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, Payload, Provenance, Repr,
    ScalarKind, Value,
};
use mycelium_numerics::Certificate;
use mycelium_vsa::capacity;

fn policy() -> ContentHash {
    ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

/// A deterministic bipolar Dense{n, F32} value (tiny LCG — house style).
fn bipolar_dense(n: u32, seed: u64) -> Value {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let xs: Vec<f64> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 63) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    Value::new(
        Repr::Dense {
            dim: n,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(xs),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

const N: u32 = 8;
const DELTA: f64 = 1e-2;

fn proven_dim() -> u32 {
    u32::try_from(capacity::required_dim(
        u64::from(N),
        DELTA,
        capacity::MARGIN_MU,
    ))
    .unwrap()
}

fn cert_bound(cert: &SwapCertificate) -> &mycelium_core::Bound {
    match cert {
        SwapCertificate::Bounded { bound, .. } => bound,
        SwapCertificate::Bijective { .. } => panic!("expected a Bounded certificate"),
    }
}

/// Proven path: the capacity side-condition checks, the basis is `ProvenThm`, the value
/// discloses `Proven` + the δ bound, and the cert validates through the one checker (SC-2).
#[test]
fn proven_enc_dec_round_trip_validates() {
    let d = proven_dim();
    let a = bipolar_dense(N, 1);
    let (hv, enc_cert) = dense_to_vsa(&a, d, DELTA, &policy()).expect("proven enc");
    assert_eq!(hv.meta().guarantee(), GuaranteeStrength::Proven);
    assert!(matches!(
        cert_bound(&enc_cert),
        mycelium_core::Bound {
            kind: BoundKind::Probability { delta },
            basis: BoundBasis::ProvenThm { .. },
        } if *delta == DELTA
    ));
    let claimed = Certificate::new(0.0, DELTA, GuaranteeStrength::Proven).unwrap();
    assert!(matches!(
        check(
            &a,
            &hv,
            RefinementRelation::BoundedSimilarity,
            claimed,
            &Evidence::Swap(&enc_cert),
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Proven
        }
    ));

    // Decode recovers the exact bipolar vector and validates too.
    let (back, dec_cert) = vsa_to_dense(&hv, N, DELTA, &policy()).expect("proven dec");
    assert_eq!(back.payload(), a.payload());
    assert!(matches!(
        check(
            &hv,
            &back,
            RefinementRelation::BoundedSimilarity,
            claimed,
            &Evidence::Swap(&dec_cert),
        ),
        CheckVerdict::Validated { .. }
    ));
}

/// Empirical path: below the proven dimension but inside the trial-validated profile the basis
/// honestly degrades to `EmpiricalFit` (never silently kept `Proven`), and a `Proven` claim over
/// it is rejected (VR-5).
#[test]
fn empirical_path_degrades_the_basis_honestly() {
    let a = bipolar_dense(N, 2);
    let d = 256; // ≥ 32·8, < requiredDim(8, 0.05)
    let (hv, cert) = dense_to_vsa(&a, d, DENSE_VSA_EMP_DELTA, &policy()).expect("empirical enc");
    assert_eq!(hv.meta().guarantee(), GuaranteeStrength::Empirical);
    assert!(matches!(
        cert_bound(&cert),
        mycelium_core::Bound {
            kind: BoundKind::Probability { delta },
            basis: BoundBasis::EmpiricalFit { .. },
        } if *delta == DENSE_VSA_EMP_DELTA
    ));
    let honest = Certificate::new(0.0, DENSE_VSA_EMP_DELTA, GuaranteeStrength::Empirical).unwrap();
    assert!(matches!(
        check(
            &a,
            &hv,
            RefinementRelation::BoundedSimilarity,
            honest,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Empirical
        }
    ));
    // Claiming Proven over empirical evidence is a VR-5 rejection, not a pass.
    let upgraded = Certificate::new(0.0, DENSE_VSA_EMP_DELTA, GuaranteeStrength::Proven).unwrap();
    assert!(matches!(
        check(
            &a,
            &hv,
            RefinementRelation::BoundedSimilarity,
            upgraded,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::NotValidated {
            reason: NotValidatedReason::CertificateMismatch { .. },
            ..
        }
    ));
}

/// A claim tighter than the certificate's δ is rejected through the tier-i union-bound kernel.
#[test]
fn tighter_delta_claim_is_rejected() {
    let d = proven_dim();
    let a = bipolar_dense(N, 3);
    let (hv, cert) = dense_to_vsa(&a, d, DELTA, &policy()).unwrap();
    let tighter = Certificate::new(0.0, DELTA / 10.0, GuaranteeStrength::Proven).unwrap();
    assert!(matches!(
        check(
            &a,
            &hv,
            RefinementRelation::BoundedSimilarity,
            tighter,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::NotValidated {
            reason: NotValidatedReason::ClaimTooTight { .. },
            ..
        }
    ));
}

/// A tampered conversion is caught by the deterministic re-derivation — a genuine divergence.
#[test]
fn tampered_encoding_diverges() {
    let d = proven_dim();
    let a = bipolar_dense(N, 4);
    let (hv, cert) = dense_to_vsa(&a, d, DELTA, &policy()).unwrap();
    let Payload::Hypervector(mut data) = hv.payload().clone() else {
        unreachable!()
    };
    data[0] += 2.0;
    let tampered = Value::new(
        hv.repr().clone(),
        Payload::Hypervector(data),
        hv.meta().clone(),
    )
    .unwrap();
    let claimed = Certificate::new(0.0, DELTA, GuaranteeStrength::Proven).unwrap();
    assert!(matches!(
        check(
            &a,
            &tampered,
            RefinementRelation::BoundedSimilarity,
            claimed,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::NotValidated {
            reason: NotValidatedReason::Diverged { .. },
            ..
        }
    ));
}

/// Every uncovered instance is an explicit typed refusal (RFC-0002 §5) — never a `Declared`
/// gamble: non-bipolar components, a dimension no basis reaches, a non-encoding decode source,
/// and an approximate source.
#[test]
fn refusals_are_explicit() {
    // Non-bipolar component.
    let half = Value::new(
        Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.0, 0.5]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_eq!(
        dense_to_vsa(&half, proven_dim(), DELTA, &policy()),
        Err(SwapError::NotBipolar { index: 1 })
    );
    // Neither the theorem nor the profile covers dim 64 for 8 components.
    let a = bipolar_dense(N, 5);
    assert!(matches!(
        dense_to_vsa(&a, 64, DELTA, &policy()),
        Err(SwapError::InsufficientCapacity { .. })
    ));
    // Decoding a hypervector that is not an enc product: its δ would describe nothing.
    let stray = Value::new(
        Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim: 256,
            sparsity: mycelium_core::SparsityClass::Dense,
        },
        Payload::Hypervector(vec![1.0; 256]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_eq!(
        vsa_to_dense(&stray, N, DELTA, &policy()),
        Err(SwapError::NotDenseVsaEncoding)
    );
}

/// The declared empirical profile holds at its worst covered point (n = 16, d = 32·n) over
/// exactly the declared trial count: the enc→dec round trip recovers the bipolar vector at a
/// failure rate ≤ δ.
#[test]
fn empirical_profile_holds_over_declared_trials() {
    let n = 16u32;
    let d = n * 32;
    let mut failures = 0u64;
    let trials = mycelium_cert::dense_vsa::DENSE_VSA_EMP_TRIALS;
    for trial in 0..trials {
        let a = bipolar_dense(n, 0xDE5E ^ trial);
        let (hv, _) = dense_to_vsa(&a, d, DENSE_VSA_EMP_DELTA, &policy()).unwrap();
        match vsa_to_dense(&hv, n, DENSE_VSA_EMP_DELTA, &policy()) {
            Ok((back, _)) => {
                if back.payload() != a.payload() {
                    failures += 1;
                }
            }
            Err(_) => failures += 1,
        }
    }
    let rate = failures as f64 / trials as f64;
    assert!(
        rate <= DENSE_VSA_EMP_DELTA,
        "round-trip failure rate {rate} exceeded the declared δ={DENSE_VSA_EMP_DELTA} \
         ({failures}/{trials})"
    );
}
