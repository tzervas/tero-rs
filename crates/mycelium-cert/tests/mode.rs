//! M-788 acceptance — **mode-gated swap-certificate emission + checking** (RFC-0034 §4/§5/§7;
//! RFC-0034 §13 mode-parametric). Across the three [`CertMode`] tiers and the cross-mode negatives:
//!
//! - `Fast` runs cert-free: **no** certificate emitted, **no** check, and the result never carries
//!   `Empirical`/`Proven` (the M-787/M-788 floor) — its bound basis is reconciled to `UserDeclared`.
//! - `Balanced` **emits** the certificate but does **not** check it (tags propagate unchanged).
//! - `Certified` **emits and checks** (today's full behaviour) — a non-validating check is surfaced
//!   never-silently.
//! - The `Meta` invariants M-I1…M-I4 hold in **every** mode (the result Value is constructible).
//! - Axis-B (out-of-range / illegal pair) stays an explicit error in **every** mode (not gated).

use mycelium_cert::{
    binary_to_ternary, dense_f32_to_bf16, gate_swap, CheckVerdict, GatedSwap, ModeGatedSwapEngine,
    SwapCertificate,
};
use mycelium_core::{
    binary, BoundBasis, CertMode, ContentHash, GuaranteeStrength, Meta, Payload, Provenance, Repr,
    ScalarKind, Value,
};
use mycelium_interp::{EvalError, SwapEngine};

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

/// An exact Dense{F32} source (so the bounded F32→BF16 swap accepts it).
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

// ---------- Bijective class (binary↔ternary, would-be Exact) across the three modes ----------

/// A bijective swap is structurally `Exact` in every mode (the free, structural tag passes the Fast
/// floor untouched). `Fast` emits/checks nothing; `Balanced` emits without checking; `Certified`
/// emits and validates. The result is `Exact`/bound-free (M-I1) in all three.
#[test]
fn bijective_across_modes() {
    let a = byte_of(42);
    let (raw_value, raw_cert) = binary_to_ternary(&a, 6, &policy()).unwrap();

    for mode in CertMode::ALL {
        let g: GatedSwap = gate_swap(&a, raw_value.clone(), raw_cert.clone(), mode).unwrap();
        // The value is Exact and bound-free in every mode (structural; M-I1).
        assert_eq!(g.value.meta().guarantee(), GuaranteeStrength::Exact);
        assert_eq!(g.value.meta().bound(), None);
        // The mode tag is recorded (never-silent).
        assert_eq!(g.value.meta().cert_mode(), mode);

        match mode {
            CertMode::Fast => {
                assert!(g.certificate.is_none(), "fast emits no certificate");
                assert!(g.check.is_none(), "fast checks nothing");
                assert!(!g.validated());
            }
            CertMode::Balanced => {
                assert!(g.certificate.is_some(), "balanced emits the certificate");
                assert!(g.check.is_none(), "balanced does not check");
                assert!(!g.validated(), "balanced makes no validation claim");
            }
            CertMode::Certified => {
                assert!(g.certificate.is_some(), "certified emits the certificate");
                assert_eq!(
                    g.check,
                    Some(CheckVerdict::Validated {
                        strength: GuaranteeStrength::Exact
                    }),
                    "certified validates the bijective certificate"
                );
                assert!(g.validated());
            }
        }
    }
}

// ---------- Bounded class (Dense F32→BF16, would-be Proven) across the three modes ----------

/// The bounded F32→BF16 swap is `Proven` in `Balanced`/`Certified` but **floors to `Declared`** in
/// `Fast`, with its computed ε bound's basis relabelled `ProvenThm → UserDeclared` (M-788). The
/// result constructs a `Meta` (M-I1…M-I4) in every mode.
#[test]
fn bounded_across_modes() {
    let src = dense_f32(vec![1.5, -2.25, 0.0]);
    let (raw_value, raw_cert) = dense_f32_to_bf16(&src, &policy()).unwrap();

    for mode in CertMode::ALL {
        let g = gate_swap(&src, raw_value.clone(), raw_cert.clone(), mode).unwrap();
        assert_eq!(g.value.meta().cert_mode(), mode);

        match mode {
            CertMode::Fast => {
                // Floored: Declared, bound kept but basis reconciled to UserDeclared (M-I4).
                assert_eq!(g.value.meta().guarantee(), GuaranteeStrength::Declared);
                let bound = g.value.meta().bound().expect("computed bound is kept");
                assert_eq!(bound.basis, BoundBasis::UserDeclared);
                assert!(g.certificate.is_none(), "fast emits no certificate");
                assert!(g.check.is_none(), "fast checks nothing");
            }
            CertMode::Balanced => {
                // Proven passes through; the cert is emitted but not checked.
                assert_eq!(g.value.meta().guarantee(), GuaranteeStrength::Proven);
                assert!(matches!(
                    g.value.meta().bound().unwrap().basis,
                    BoundBasis::ProvenThm { .. }
                ));
                assert!(g.certificate.is_some());
                assert!(g.check.is_none(), "balanced does not check");
            }
            CertMode::Certified => {
                assert_eq!(g.value.meta().guarantee(), GuaranteeStrength::Proven);
                assert!(g.certificate.is_some());
                // The emitted bounded certificate validates through the M-210 checker.
                assert!(
                    g.validated(),
                    "certified must validate the bounded certificate, got {:?}",
                    g.check
                );
            }
        }
    }
}

// ---------- Cross-mode NEGATIVE: fast never carries Empirical/Proven nor emits/checks ----------

/// The M-787/M-788 floor as a swept negative over both swap classes: a `Fast`-gated result is never
/// `Empirical`/`Proven`, never emits a certificate, and never checks one.
#[test]
fn fast_never_certifies_any_class() {
    let cases: Vec<(Value, SwapCertificate, Value)> = vec![
        {
            let a = byte_of(7);
            let (v, c) = binary_to_ternary(&a, 6, &policy()).unwrap();
            (a, c, v)
        },
        {
            let s = dense_f32(vec![3.0, -1.0]);
            let (v, c) = dense_f32_to_bf16(&s, &policy()).unwrap();
            (s, c, v)
        },
    ];
    for (src, cert, value) in cases {
        let g = gate_swap(&src, value, cert, CertMode::Fast).unwrap();
        let strength = g.value.meta().guarantee();
        assert!(
            strength != GuaranteeStrength::Empirical && strength != GuaranteeStrength::Proven,
            "fast result must never be Empirical/Proven, got {strength:?}"
        );
        assert!(g.certificate.is_none(), "fast emits no certificate");
        assert!(g.check.is_none(), "fast checks nothing");
        // Any surviving bound is UserDeclared (the reconciled basis) — never an unearned basis.
        if let Some(b) = g.value.meta().bound() {
            assert_eq!(b.basis, BoundBasis::UserDeclared);
        }
    }
}

// ---------- Axis-B (fallibility) is NOT gated: explicit error in every mode ----------

/// An out-of-range `dec` (ternary value outside `B_n`) is an explicit error in *every* mode — the
/// mode tunes certification, never fallibility (RFC-0034 §4; SC-3/G2). Driven through the engine.
#[test]
fn out_of_range_is_an_error_in_every_mode() {
    // 364 = all-`+` 6-trit value ∉ [−128, 127], so dec to Binary{8} is OutOfRange.
    let tern = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(mycelium_core::ternary::int_to_trits(364, 6).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    for mode in CertMode::ALL {
        let engine = ModeGatedSwapEngine::new(mode);
        let result = engine.swap(&tern, &Repr::Binary { width: 8 }, &policy());
        assert!(
            matches!(result, Err(EvalError::Swap(_))),
            "out-of-range dec must be an explicit error in {mode:?}, got {result:?}"
        );
    }
}

/// An illegal `(width, trits)` pair is an explicit error in every mode (Axis-B ungated).
#[test]
fn illegal_pair_is_an_error_in_every_mode() {
    let a = byte_of(1);
    for mode in CertMode::ALL {
        let engine = ModeGatedSwapEngine::new(mode);
        // (8, 1): Binary{8} ⊄ Ternary{1} — illegal.
        let result = engine.swap(&a, &Repr::Ternary { trits: 1 }, &policy());
        assert!(
            matches!(result, Err(EvalError::Swap(_))),
            "illegal pair must be an explicit error in {mode:?}, got {result:?}"
        );
    }
}

// ---------- Engine surface: default is Fast; Certified surfaces a non-validating check ----------

/// The default engine is `Fast` (RFC-0034 §5) and returns a value with no certificate/check.
#[test]
fn default_engine_is_fast() {
    let engine = ModeGatedSwapEngine::default();
    assert_eq!(engine.mode(), CertMode::Fast);
    let a = byte_of(3);
    let g = engine
        .swap_gated(&a, &Repr::Ternary { trits: 6 }, &policy())
        .unwrap();
    assert_eq!(g.value.meta().cert_mode(), CertMode::Fast);
    assert!(g.certificate.is_none());
    assert!(g.check.is_none());
}

/// The engine's `swap` returns the gated value when the (Certified) check validates — the common
/// path — confirming the never-silent guard does not reject a *valid* certified swap.
#[test]
fn certified_engine_returns_value_on_validation() {
    let engine = ModeGatedSwapEngine::new(CertMode::Certified);
    let a = byte_of(99);
    let value = engine
        .swap(&a, &Repr::Ternary { trits: 6 }, &policy())
        .expect("a valid certified swap returns its value");
    assert_eq!(value.meta().cert_mode(), CertMode::Certified);
    assert_eq!(value.meta().guarantee(), GuaranteeStrength::Exact);
}
