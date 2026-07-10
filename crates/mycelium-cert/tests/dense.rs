//! M-211 acceptance — the Dense `F32 → BF16` bounded swap (RFC-0002 §3/§5; ADR-010 §1): the
//! proven `Rel 2^−8` rounding bound holds on a property sweep, rounding is to-nearest-even, the
//! emitted `Bounded` certificate validates through the M-210 checker, and every out-of-theorem
//! input (NaN/Inf, non-f32, subnormal, overflow, approximate source) is an explicit refusal.

use mycelium_cert::{
    check, dense_f32_to_bf16, CertifiedSwapEngine, CheckVerdict, Evidence, NotValidatedReason,
    RefinementRelation, SwapCertificate, SwapError, BF16_REL_EPS,
};
use mycelium_core::{
    Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, NormKind, Payload,
    Provenance, Repr, ScalarKind, Value,
};
use mycelium_interp::SwapEngine;
use mycelium_numerics::Certificate;

fn policy() -> ContentHash {
    ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

fn dense_f32(xs: Vec<f64>) -> Value {
    Value::new(
        Repr::Dense {
            dim: u32::try_from(xs.len()).unwrap(),
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(xs),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn scalars(v: &Value) -> &[f64] {
    match v.payload() {
        Payload::Scalars(xs) => xs,
        other => panic!("expected scalars, got {other:?}"),
    }
}

/// A deterministic LCG over f32s spanning the normal exponent range (Phase-1 house style — no
/// `rand` dependency). Yields finite, normal-or-zero f32 values as exact f64s.
struct F32Gen(u64);

impl F32Gen {
    fn next_f32(&mut self) -> f32 {
        loop {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Mantissa from the top bits; exponent uniform in [-100, 100]; random sign.
            let mantissa = ((self.0 >> 40) as f64) / f64::from(1u32 << 24) + 1.0;
            let exp = i32::try_from((self.0 >> 16) % 201).unwrap() - 100;
            let sign = if self.0 & 1 == 0 { 1.0 } else { -1.0 };
            #[allow(clippy::cast_possible_truncation)]
            let x = (sign * mantissa * 2.0_f64.powi(exp)) as f32;
            if x.is_finite() && (x == 0.0 || x.abs() >= f32::MIN_POSITIVE) {
                return x;
            }
        }
    }
}

/// **Soundness property (ADR-010 §1):** over 20k generated f32s, the per-element relative rounding
/// error never exceeds the certified `u = 2^−8`, and rounding is idempotent (the output is on the
/// bf16 grid).
#[test]
fn relative_rounding_bound_holds_on_a_sweep() {
    let mut gen = F32Gen(0xE23_0211); // E2-3 / M-211 seed — deterministic, reproducible
    let mut xs = Vec::with_capacity(20_000);
    for _ in 0..20_000 {
        xs.push(f64::from(gen.next_f32()));
    }
    let a = dense_f32(xs.clone());
    let (b, _) = dense_f32_to_bf16(&a, &policy()).expect("swap");
    let ys = scalars(&b);
    for (x, y) in xs.iter().zip(ys.iter()) {
        if *x == 0.0 {
            assert_eq!(*y, 0.0);
            continue;
        }
        let rel = (x - y).abs() / x.abs();
        assert!(rel <= BF16_REL_EPS, "rel error {rel} > 2^-8 for x = {x}");
    }
    // Idempotence: swapping the bf16-gridded values again is the identity on payloads.
    let on_grid = dense_f32(ys.to_vec());
    let (again, _) = dense_f32_to_bf16(&on_grid, &policy()).expect("re-swap");
    assert_eq!(scalars(&again), ys);
}

/// Round-to-nearest-even spot checks at the midpoint between two bf16 neighbours of 1.0.
#[test]
fn rounding_is_to_nearest_even() {
    // 1 + 2^-8 is exactly midway between 1.0 (even mantissa) and 1 + 2^-7 → ties to even → 1.0.
    let tie = 1.0 + 0.003_906_25;
    // 1 + 3·2^-9 is past the midpoint → rounds up to 1 + 2^-7.
    let up = 1.0 + 3.0 * 0.001_953_125;
    let a = dense_f32(vec![tie, up]);
    let (b, _) = dense_f32_to_bf16(&a, &policy()).unwrap();
    assert_eq!(scalars(&b), &[1.0, 1.007_812_5]);
}

/// The emitted certificate is `Bounded` with the `Rel 2^−8` ε and a `ProvenThm` basis; the result
/// value honestly discloses `Proven` + the same bound (M-I2) and records the policy.
#[test]
fn certificate_and_meta_are_honest() {
    let a = dense_f32(vec![1.5, -2.25, 0.0]);
    let (b, cert) = dense_f32_to_bf16(&a, &policy()).unwrap();
    assert_eq!(
        b.repr(),
        &Repr::Dense {
            dim: 3,
            dtype: ScalarKind::Bf16
        }
    );
    assert_eq!(b.meta().guarantee(), GuaranteeStrength::Proven);
    assert_eq!(b.meta().policy_used(), Some(&policy()));
    match &cert {
        SwapCertificate::Bounded { bound, .. } => {
            assert!(matches!(bound.basis, BoundBasis::ProvenThm { .. }));
            assert_eq!(
                bound.kind,
                BoundKind::Error {
                    eps: BF16_REL_EPS,
                    norm: NormKind::Rel
                }
            );
            assert_eq!(b.meta().bound(), Some(bound));
        }
        SwapCertificate::Bijective { .. } => panic!("F32→BF16 is bounded, not bijective"),
    }
}

/// **M-211 acceptance:** the emitted certificate validates through the M-210 shared checker.
#[test]
fn bounded_cert_validates_through_the_shared_checker() {
    let a = dense_f32(vec![
        1.5,
        -2.25,
        0.0,
        1.000_976_562_5,
        f64::from(3.0e30_f32),
        f64::from(-7.0e-30_f32),
    ]);
    let (b, cert) = dense_f32_to_bf16(&a, &policy()).unwrap();
    let claimed = Certificate::new(BF16_REL_EPS, 0.0, GuaranteeStrength::Proven).unwrap();
    assert_eq!(
        check(
            &a,
            &b,
            RefinementRelation::BoundedSimilarity,
            claimed,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Proven
        }
    );
    // A weaker claim is allowed (downgrade is always honest) and validates at its own strength.
    let weaker = Certificate::new(BF16_REL_EPS, 0.0, GuaranteeStrength::Declared).unwrap();
    assert_eq!(
        check(
            &a,
            &b,
            RefinementRelation::BoundedSimilarity,
            weaker,
            &Evidence::Swap(&cert),
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Declared
        }
    );
}

/// A tampered conversion is caught by the checker: the measured deviation exceeds the
/// certificate's ε → the tier-i rejection surfaces.
#[test]
fn tampered_conversion_is_caught_by_the_checker() {
    let a = dense_f32(vec![1.0, 2.0]);
    let (b, cert) = dense_f32_to_bf16(&a, &policy()).unwrap();
    let mut forged = scalars(&b).to_vec();
    forged[1] *= 1.01; // 1% off — far beyond 2^-8 ≈ 0.39%
    let bad = Value::new(b.repr().clone(), Payload::Scalars(forged), b.meta().clone()).unwrap();
    let claimed = Certificate::new(BF16_REL_EPS, 0.0, GuaranteeStrength::Proven).unwrap();
    let verdict = check(
        &a,
        &bad,
        RefinementRelation::BoundedSimilarity,
        claimed,
        &Evidence::Swap(&cert),
    );
    assert!(
        matches!(
            verdict,
            CheckVerdict::NotValidated {
                reason: NotValidatedReason::ClaimTooTight { .. },
                ..
            }
        ),
        "got {verdict:?}"
    );
}

// ---------- explicit refusals (never silent; M-211 acceptance) ----------

#[test]
fn nan_and_inf_are_explicit() {
    let a = dense_f32(vec![1.0, f64::NAN]);
    assert_eq!(
        dense_f32_to_bf16(&a, &policy()),
        Err(SwapError::NonFinite { index: 1 })
    );
    let b = dense_f32(vec![f64::INFINITY]);
    assert_eq!(
        dense_f32_to_bf16(&b, &policy()),
        Err(SwapError::NonFinite { index: 0 })
    );
}

#[test]
fn non_f32_payload_is_explicit() {
    // 0.1 is not exactly representable in f32 — the payload contradicts dtype F32.
    let a = dense_f32(vec![0.1]);
    assert_eq!(
        dense_f32_to_bf16(&a, &policy()),
        Err(SwapError::NotAnF32 { index: 0 })
    );
}

#[test]
fn subnormal_is_outside_the_proven_range() {
    // The smallest positive f32 subnormal, exact as f64.
    let tiny = f64::from(f32::from_bits(1));
    let a = dense_f32(vec![tiny]);
    assert_eq!(
        dense_f32_to_bf16(&a, &policy()),
        Err(SwapError::SubnormalUnsupported { index: 0 })
    );
}

#[test]
fn rounding_overflow_is_explicit() {
    // f32::MAX rounds up past bf16's largest finite value → +Inf → refused.
    let a = dense_f32(vec![f64::from(f32::MAX)]);
    assert_eq!(
        dense_f32_to_bf16(&a, &policy()),
        Err(SwapError::RoundOverflow { index: 0 })
    );
}

#[test]
fn approximate_source_is_refused() {
    let declared = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(Bound {
            kind: BoundKind::Error {
                eps: 0.5,
                norm: NormKind::Rel,
            },
            basis: BoundBasis::UserDeclared,
        }),
        None,
        None,
        None,
    )
    .unwrap();
    let a = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.0]),
        declared,
    )
    .unwrap();
    assert_eq!(
        dense_f32_to_bf16(&a, &policy()),
        Err(SwapError::ApproximateSource)
    );
}

/// The engine over the complete certified surface performs the bounded swap and stays explicit
/// about everything else.
#[test]
fn certified_engine_covers_the_split_regime() {
    let a = dense_f32(vec![1.5, -2.25]);
    let out = CertifiedSwapEngine
        .swap(
            &a,
            &Repr::Dense {
                dim: 2,
                dtype: ScalarKind::Bf16,
            },
            &policy(),
        )
        .expect("bounded swap through the engine");
    assert_eq!(out.meta().guarantee(), GuaranteeStrength::Proven);
    // The reverse direction (BF16 → F32) has no certified swap — explicit, never silent.
    let err = CertifiedSwapEngine.swap(
        &out,
        &Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        &policy(),
    );
    assert!(matches!(
        err,
        Err(mycelium_interp::EvalError::UnsupportedSwap { .. })
    ));
}
