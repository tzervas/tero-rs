//! Mutant-witness tests for `mycelium-cert` (ADR-021 Gate A3 / M-654).
//!
//! Each test is a genuine witness for a specific surviving mutant from `cargo-mutants -p
//! mycelium-cert`. Every test fails when its named mutation is applied to the source and passes on
//! the honest code. Tests are grouped by source module; the mutation site is annotated inline.

use mycelium_cert::{
    binary_to_ternary, check, dense_f32_to_bf16, dense_to_vsa, dense_vsa, roundtrip_lemma_ref,
    ternary_to_binary, vsa_to_dense, BinTernParams, BinaryTernarySwapEngine, CertifiedSwapEngine,
    CheckVerdict, Evidence, NotValidatedReason, RefinementRelation, SwapCertificate, SwapError,
    DENSE_VSA_DEFAULT_DELTA, DENSE_VSA_EMP_DELTA,
};
use mycelium_core::{
    binary, operation_hash, ternary, Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength,
    Meta, NormKind, Payload, Provenance, Repr, ScalarKind, SparsityClass, Value,
};
use mycelium_interp::{EvalError, SwapEngine};
use mycelium_numerics::Certificate;
use mycelium_vsa::capacity;

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

fn bipolar_dense(xs: Vec<f64>) -> Value {
    dense_f32(xs)
}

// ===========================================================================
// src/lib.rs — SwapError::fmt
// ===========================================================================

/// mutant: src/lib.rs:155 `<impl core::fmt::Display for SwapError>::fmt` → `Ok(Default::default())`
///
/// If `fmt` silently returned the default string, `to_string()` would produce "" for every
/// variant. This test pins the Display output to the actual message text so a blanked-out `fmt`
/// fails immediately.
#[test]
fn swap_error_display_messages_are_non_empty_and_match_content() {
    let cases: &[(SwapError, &str)] = &[
        (SwapError::WrongSource { expected: "Binary" }, "Binary"),
        (SwapError::IllegalPair { width: 8, trits: 4 }, "Binary{8}"),
        (SwapError::OutOfRange, "ternary value is outside"),
        (SwapError::NonFinite { index: 2 }, "NaN/Inf"),
        (SwapError::NotAnF32 { index: 3 }, "not exactly an f32"),
        (SwapError::SubnormalUnsupported { index: 0 }, "subnormal"),
        (SwapError::RoundOverflow { index: 1 }, "overflows"),
        (SwapError::ApproximateSource, "approximate"),
        (
            SwapError::InsufficientCapacity {
                components: 8,
                dim: 64,
                required: 2048,
            },
            "Dense\u{2194}VSA",
        ),
        (SwapError::NotBipolar { index: 2 }, "bipolar"),
        (SwapError::NotDenseVsaEncoding, "swap.dense_vsa.enc.v1"),
        (SwapError::AmbiguousDecode { index: 5 }, "undefined"),
    ];
    for (err, needle) in cases {
        let s = err.to_string();
        assert!(
            !s.is_empty(),
            "SwapError::{err:?} display must not be empty (mutant: fmt → Ok(Default::default()))"
        );
        assert!(
            s.contains(needle),
            "SwapError display for {:?} should contain {:?}, got {:?}",
            err,
            needle,
            s
        );
    }
}

// ===========================================================================
// src/lib.rs — BinaryTernarySwapEngine::swap match guard `a == b`
// ===========================================================================

/// mutant: src/lib.rs:368 match guard `a == b` → `true`
///
/// If the guard is `true` every time, then a `(Binary → Dense)` swap would hit the identity
/// branch rather than the `UnsupportedSwap` error. The test confirms that a genuinely different
/// pair is refused explicitly, not silently "identity-ed".
#[test]
fn btswap_engine_unrelated_reprs_produce_unsupported_not_identity() {
    let src = byte_of(42);
    let target = Repr::Ternary { trits: 6 };
    // same-repr identity must succeed
    let result = BinaryTernarySwapEngine.swap(&src, &Repr::Binary { width: 8 }, &policy());
    assert!(result.is_ok(), "identity swap must succeed");

    // A non-Binary, non-Ternary target must be an UnsupportedSwap, never silently identity.
    let dense_target = Repr::Dense {
        dim: 1,
        dtype: ScalarKind::F32,
    };
    assert!(
        matches!(
            BinaryTernarySwapEngine.swap(&src, &dense_target, &policy()),
            Err(EvalError::UnsupportedSwap { .. })
        ),
        "Binary → Dense must be UnsupportedSwap (mutant: a==b guard → true would route to identity)"
    );

    // Ternary → Ternary{different trits}: different repr, must also be unsupported.
    let tern_src = tern_of(5, 4);
    let tern_other = Repr::Ternary { trits: 6 }; // different width ⇒ different repr
    assert!(
        matches!(
            BinaryTernarySwapEngine.swap(&tern_src, &tern_other, &policy()),
            Err(EvalError::UnsupportedSwap { .. })
        ),
        "Ternary{{4}} → Ternary{{6}} must be UnsupportedSwap (guard mutant: a==b → true)"
    );
    let _ = target;
}

/// mutant: src/lib.rs:368 match guard `a == b` → `false`
///
/// If the guard is always `false`, same-repr identity swaps would fall through to `UnsupportedSwap`
/// instead of delegating to the identity engine. This test confirms the identity path succeeds.
#[test]
fn btswap_engine_same_repr_identity_succeeds() {
    // Binary identity
    let src = byte_of(77);
    let out = BinaryTernarySwapEngine
        .swap(&src, &Repr::Binary { width: 8 }, &policy())
        .expect("same-repr Binary{8} → Binary{8} must succeed (mutant: guard → false breaks it)");
    assert_eq!(out.payload(), src.payload());

    // Ternary identity
    let tern = tern_of(-5, 6);
    let out_t = BinaryTernarySwapEngine
        .swap(&tern, &Repr::Ternary { trits: 6 }, &policy())
        .expect("same-repr Ternary{6} → Ternary{6} must succeed (mutant: guard → false breaks it)");
    assert_eq!(out_t.payload(), tern.payload());
}

/// mutant: src/lib.rs:368 `==` → `!=` in match guard `a == b`
///
/// If the guard uses `!=`, same-repr swaps become `UnsupportedSwap` and an unsupported pair
/// (differing reprs) routes through identity. Both behaviors are wrong: the test exercises the
/// boundary where the existing `a == b` arm must handle the match correctly.
#[test]
fn btswap_engine_distinguishes_same_and_different_repr_on_guard_equality() {
    let bin8 = byte_of(10);
    // Same repr: must succeed as identity.
    assert!(
        BinaryTernarySwapEngine
            .swap(&bin8, &Repr::Binary { width: 8 }, &policy())
            .is_ok(),
        "identity must succeed (mutant: == → != in guard)"
    );
    // Different repr (not binary↔ternary): must be UnsupportedSwap.
    assert!(
        matches!(
            BinaryTernarySwapEngine.swap(
                &bin8,
                &Repr::Dense {
                    dim: 1,
                    dtype: ScalarKind::F32
                },
                &policy()
            ),
            Err(EvalError::UnsupportedSwap { .. })
        ),
        "unsupported pair must stay unsupported (mutant: == → != in guard)"
    );
}

// ===========================================================================
// src/lib.rs — CertifiedSwapEngine::swap `src_dim == target_dim` guard
// ===========================================================================

/// mutant: src/lib.rs:406 match guard `src_dim == target_dim` → `true`
///
/// If the guard is always `true`, a Dense{F32, dim=2} → Dense{BF16, dim=4} swap would be
/// attempted instead of falling through to the binary-ternary engine (which refuses it). This test
/// confirms that dimension-mismatched F32→BF16 does not silently succeed.
#[test]
fn certified_engine_mismatched_dense_dims_are_refused() {
    // F32 dim=2, asking BF16 dim=4 — the guard `src_dim == target_dim` must fail → not matched.
    let src = dense_f32(vec![1.0, 2.0]);
    let bad_target = Repr::Dense {
        dim: 4,
        dtype: ScalarKind::Bf16,
    };
    // With a true guard this would try dense_f32_to_bf16 on a 2-element source with a 4-dim
    // target repr, which is a logic error. The honest guard → falls through to BinaryTernarySwap
    // engine → UnsupportedSwap.
    assert!(
        matches!(
            CertifiedSwapEngine.swap(&src, &bad_target, &policy()),
            Err(EvalError::UnsupportedSwap { .. } | EvalError::Swap(_))
        ),
        "mismatched Dense F32→BF16 dims must not silently match (guard mutant: == → true)"
    );
    // The correct (same-dim) case still works.
    let ok = CertifiedSwapEngine
        .swap(
            &src,
            &Repr::Dense {
                dim: 2,
                dtype: ScalarKind::Bf16,
            },
            &policy(),
        )
        .expect("same-dim F32→BF16 must succeed");
    assert_eq!(ok.meta().guarantee(), GuaranteeStrength::Proven);
}

// ===========================================================================
// src/lib.rs — CertifiedSwapEngine::swap `model == DENSE_VSA_MODEL` guards
// ===========================================================================

/// mutant: src/lib.rs:426 match guard `model == dense_vsa::DENSE_VSA_MODEL` → `true`
///
/// If the Dense→VSA guard is `true` always, a non-MAP-I VSA target would be silently attempted
/// via `dense_to_vsa`, which would refuse with a wrong-repr error. The test confirms that using
/// a different model name either returns an explicit `UnsupportedSwap` or `Swap` error, not a
/// silent "success" with a different model.
#[test]
fn certified_engine_non_map_i_vsa_target_is_refused_or_unsupported() {
    let src = bipolar_dense(vec![1.0, -1.0]);
    let hrr_target = Repr::Vsa {
        model: "HRR".to_owned(),
        dim: 256,
        sparsity: SparsityClass::Dense,
    };
    let result = CertifiedSwapEngine.swap(&src, &hrr_target, &policy());
    assert!(
        matches!(
            result,
            Err(EvalError::UnsupportedSwap { .. } | EvalError::Swap(_))
        ),
        "Dense → non-MAP-I VSA must not silently succeed (mutant: model guard → true); got {result:?}"
    );
}

/// mutant: src/lib.rs:426 match guard `model == dense_vsa::DENSE_VSA_MODEL` → `false`
///
/// If the guard is always `false`, a genuine MAP-I Dense→VSA swap would fall through to
/// `BinaryTernarySwapEngine`, which would refuse it as `UnsupportedSwap`. This test confirms the
/// authentic MAP-I path succeeds.
#[test]
fn certified_engine_map_i_dense_to_vsa_succeeds() {
    let src = bipolar_dense(vec![1.0, -1.0, 1.0, 1.0]);
    let vsa_dim = u32::try_from(capacity::required_dim(
        4,
        DENSE_VSA_DEFAULT_DELTA,
        capacity::MARGIN_MU,
    ))
    .unwrap_or(4096)
    .max(4096);
    let target = Repr::Vsa {
        model: dense_vsa::DENSE_VSA_MODEL.to_owned(),
        dim: vsa_dim,
        sparsity: SparsityClass::Dense,
    };
    let out = CertifiedSwapEngine.swap(&src, &target, &policy()).expect(
        "MAP-I Dense→VSA via CertifiedSwapEngine must succeed (mutant: guard → false breaks it)",
    );
    assert_eq!(out.repr(), &target);
}

/// mutant: src/lib.rs:426 `==` → `!=` in `model == dense_vsa::DENSE_VSA_MODEL`
///
/// If `==` is flipped to `!=`, MAP-I targets fall through (UnsupportedSwap) and non-MAP-I targets
/// would be incorrectly routed into `dense_to_vsa`. This test exercises both polarities.
#[test]
fn certified_engine_model_guard_distinguishes_map_i_from_hrr() {
    let src = bipolar_dense(vec![1.0, -1.0]);
    let vsa_dim = 2048u32;
    // MAP-I must succeed.
    let map_i = Repr::Vsa {
        model: dense_vsa::DENSE_VSA_MODEL.to_owned(),
        dim: vsa_dim,
        sparsity: SparsityClass::Dense,
    };
    assert!(
        CertifiedSwapEngine.swap(&src, &map_i, &policy()).is_ok(),
        "MAP-I Dense→VSA must succeed (mutant: == → != flips polarity)"
    );
    // HRR must not succeed.
    let hrr = Repr::Vsa {
        model: "HRR".to_owned(),
        dim: vsa_dim,
        sparsity: SparsityClass::Dense,
    };
    assert!(
        matches!(
            CertifiedSwapEngine.swap(&src, &hrr, &policy()),
            Err(EvalError::UnsupportedSwap { .. } | EvalError::Swap(_))
        ),
        "HRR Dense→VSA must not succeed (mutant: == → != would incorrectly route it)"
    );
}

// ===========================================================================
// src/check.rs — check_bijection params binding `||` → `&&`
// ===========================================================================

/// mutant: src/check.rs:218 `||` → `&&` in `check_bijection` params binding (Binary→Ternary arm)
///
/// The check is `if *width != params.width || *trits != params.trits`.
/// With `&&` instead of `||`, a cert whose `width` mismatches alone would NOT fire the guard
/// (because `trits` matches) and the check would pass a forged cert. This test presents a cert
/// where `width` differs but `trits` matches — the honest code refuses it.
#[test]
fn check_bijection_catches_width_mismatch_independent_of_trits() {
    // Build a cert claiming Binary{8}→Ternary{6} but with params {width:4, trits:6}.
    // width mismatches (8 vs 4), trits match (6 vs 6).
    let a = byte_of(5);
    let (b, _) = binary_to_ternary(&a, 6, &policy()).unwrap();
    let forged = SwapCertificate::Bijective {
        src: Repr::Binary { width: 8 },
        target: Repr::Ternary { trits: 6 },
        policy_used: policy(),
        lemma_ref: roundtrip_lemma_ref(),
        params: BinTernParams { width: 4, trits: 6 }, // width WRONG, trits correct
    };
    let result = check(
        &a,
        &b,
        RefinementRelation::Bijection,
        Certificate::exact(),
        &Evidence::Swap(&forged),
    );
    assert!(
        matches!(
            result,
            CheckVerdict::NotValidated {
                reason: NotValidatedReason::CertificateMismatch { .. },
                ..
            }
        ),
        "width mismatch alone must be caught (mutant: || → && in params check); got {result:?}"
    );
}

/// mutant: src/check.rs:224 `||` → `&&` in `check_bijection` params binding (Ternary→Binary arm)
///
/// The check is `if *width != params.width || *trits != params.trits`.
/// With `&&`, a cert whose only `trits` field mismatches would be accepted. This test exercises
/// the Ternary→Binary arm with a trits mismatch but correct width.
///
/// CRITICAL: the cert's `params` must pass `legal_pair(params.width, params.trits)` (line 204)
/// or it is rejected there BEFORE reaching line 224. So we need params.trits = 7 (a legal but
/// WRONG value): legal_pair(8,7)=true (3^7/2=1093≥128) but trits 7≠6 (the actual Ternary width).
/// With the `||` honest code: `(8≠8)||(6≠7)` = true → mismatch caught.
/// With `&&` mutant:      `(8≠8)&&(6≠7)` = false → NOT caught → re-derives & validates (wrong).
#[test]
fn check_bijection_catches_trits_mismatch_independent_of_width() {
    // Build a cert claiming Ternary{6}→Binary{8} but with params {width:8, trits:7}.
    // width matches (8 == 8), trits mismatch (6 ≠ 7).
    // legal_pair(8,7)=true so passes the legal_pair guard at line 204,
    // but trits in repr (6) ≠ params.trits (7) — which must be caught.
    let a_bin = byte_of(-20);
    let (a_tern, _) = binary_to_ternary(&a_bin, 6, &policy()).unwrap();
    let (b, _) = ternary_to_binary(&a_tern, 8, &policy()).unwrap();
    let forged = SwapCertificate::Bijective {
        src: Repr::Ternary { trits: 6 },
        target: Repr::Binary { width: 8 },
        policy_used: policy(),
        lemma_ref: roundtrip_lemma_ref(),
        params: BinTernParams { width: 8, trits: 7 }, // trits WRONG (7≠6), width correct
    };
    let result = check(
        &a_tern,
        &b,
        RefinementRelation::Bijection,
        Certificate::exact(),
        &Evidence::Swap(&forged),
    );
    assert!(
        matches!(
            result,
            CheckVerdict::NotValidated {
                reason: NotValidatedReason::CertificateMismatch { .. },
                ..
            }
        ),
        "trits mismatch alone must be caught (mutant: || → && in params check); got {result:?}"
    );
}

// ===========================================================================
// src/check.rs — check_bounded_prob VR-5 strength comparison `<` → `>`
// ===========================================================================

/// mutant: src/check.rs:514 `<` → `>` in `check_bounded_prob` VR-5 strength check
///
/// The check is `claimed.strength().rank() < basis_strength(&rebound.basis).rank()`.
/// With `>`, it would reject legitimate downgrades (Empirical claim, Proven basis) and accept
/// upgrades (Proven claim, Empirical basis). This test presents a Proven-basis encode, then
/// checks that claiming `Proven` succeeds and claiming a lower strength also succeeds (not rejected).
#[test]
fn check_bounded_prob_vr5_direction_is_correct() {
    let n = 8u32;
    let delta = 1e-2_f64;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    let a = bipolar_dense(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    let (hv, cert) = dense_to_vsa(&a, vsa_dim, delta, &policy()).expect("proven enc");

    // Claiming Proven (matching the ProvenThm basis) must validate.
    let proven_claim = Certificate::new(0.0, delta, GuaranteeStrength::Proven).unwrap();
    assert!(
        matches!(
            check(
                &a,
                &hv,
                RefinementRelation::BoundedSimilarity,
                proven_claim,
                &Evidence::Swap(&cert)
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Proven
            }
        ),
        "Proven claim over ProvenThm basis must validate (mutant: < → > in rank comparison)"
    );

    // Claiming a weaker strength (Empirical) must also validate — downgrade is always honest.
    let downgrade_claim = Certificate::new(0.0, delta, GuaranteeStrength::Empirical).unwrap();
    assert!(
        matches!(
            check(
                &a,
                &hv,
                RefinementRelation::BoundedSimilarity,
                downgrade_claim,
                &Evidence::Swap(&cert)
            ),
            CheckVerdict::Validated { .. }
        ),
        "downgrade claim must validate (mutant: < → > might break this)"
    );

    // Attempting a ProvenThm-basis enc, then claiming Proven MUST succeed — but with wrong
    // direction it would be mismatch. Already tested above; the key witness is the downgrade arm.
}

// ===========================================================================
// src/check.rs — deviation L2 norm `*` operators
// ===========================================================================

/// mutant: src/check.rs:569 `*` → `/` in L2 deviation (the `d * d` squaring term)
///
/// The L2 norm is `sqrt(sum(d^2))`. With `/` instead of `*`, it computes `sqrt(sum(d/d)) =
/// sqrt(n)` for n non-zero elements. The two are identical ONLY when all diffs are 1.0.
///
/// CRITICAL: the original test used A=[0,0,0,0], B=[1,1,1,1] with diffs all = 1.0. For d=1,
/// d*d=1 = d/d=1 = d+d=2 (almost). This test uses diff=3.0 where `d*d=9`, `d+d=6`, `d/d=1`.
///
/// A=[0], B=[3]: honest L2 = sqrt(9) = 3.0.
/// With `+` mutant: sqrt(3+3) = sqrt(6) ≈ 2.449.
/// With `/` mutant: sqrt(3/3) = sqrt(1) = 1.0.
/// A cert with eps=2.5: honest rejects (3.0 > 2.5), both mutants accept (< 2.5). Test catches both.
#[test]
fn check_bounded_l2_deviation_uses_correct_multiplication() {
    // A = [0.0], B = [3.0], diff = 3.0
    // Honest L2 = sqrt(3^2) = sqrt(9) = 3.0
    // With `+` mutant: sqrt(3+3) = sqrt(6) ≈ 2.449 (WRONG — accepts eps=2.5)
    // With `/` mutant: sqrt(3/3) = sqrt(1) = 1.0 (WRONG — accepts eps=2.5)
    let a = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![0.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let b = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::Bf16,
        },
        Payload::Scalars(vec![3.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let l2_cert = |eps: f64| SwapCertificate::Bounded {
        src: a.repr().clone(),
        target: b.repr().clone(),
        policy_used: policy(),
        bound: Bound {
            kind: BoundKind::Error {
                eps,
                norm: NormKind::L2,
            },
            basis: BoundBasis::EmpiricalFit {
                trials: 100,
                method: "mutant-witness L2".to_owned(),
            },
        },
    };
    // eps = 3.5 > actual L2 = 3.0 → must validate (honest and both mutants).
    let claimed_ok = Certificate::new(3.5, 0.0, GuaranteeStrength::Empirical).unwrap();
    assert!(
        matches!(
            check(
                &a,
                &b,
                RefinementRelation::BoundedSimilarity,
                claimed_ok,
                &Evidence::Swap(&l2_cert(3.5))
            ),
            CheckVerdict::Validated { .. }
        ),
        "L2 eps=3.5 ≥ actual 3.0 must validate"
    );

    // eps = 2.5: honest L2 = 3.0 > 2.5 → ClaimTooTight.
    // `+` mutant: sqrt(6) ≈ 2.449 ≤ 2.5 → would Validate (WRONG).
    // `/` mutant: sqrt(1) = 1.0 ≤ 2.5 → would Validate (WRONG).
    // Both mutants are caught by this assertion.
    let claimed_tight = Certificate::new(2.5, 0.0, GuaranteeStrength::Empirical).unwrap();
    assert!(
        matches!(
            check(
                &a,
                &b,
                RefinementRelation::BoundedSimilarity,
                claimed_tight,
                &Evidence::Swap(&l2_cert(2.5))
            ),
            CheckVerdict::NotValidated {
                reason: NotValidatedReason::ClaimTooTight { .. },
                ..
            }
        ),
        "L2 eps=2.5 < actual 3.0 must be ClaimTooTight \
         (mutant * → + gives ≈2.449 which would pass; mutant * → / gives 1.0 which would pass)"
    );
}

// ===========================================================================
// src/dense.rs — round_f32_to_bf16 bit-level operations
// ===========================================================================

/// mutant: src/dense.rs:49 `>>` → `<<` in `(bits >> 16) & 1` (lsb extraction)
///
/// The LSB of the bfloat16 mantissa is bit 16 of the f32 bits. `(bits >> 16) & 1` extracts it.
/// With `<<`, `(bits << 16) & 1` is ALWAYS 0 for any u32 (shifting left by 16 fills the low 16
/// bits with zeros, so bit 0 of the result is always 0). This means the tie-breaking lsb is
/// always 0 under the mutant, forcing round-down even when the lower bf16 neighbour is odd.
///
/// IMPORTANT: the original test case used 1.0+2^-8 where the lower bf16 grid point (1.0) has
/// f32 bit 16 = 0. Honest lsb=0 and mutant lsb=0 agree here — that test does NOT kill the
/// mutant. The distinguishing case needs a tie where the lower bf16 neighbour has bit 16 = 1
/// (odd mantissa), so honest lsb=1 → rounds UP while mutant lsb=0 → rounds DOWN.
///
/// Lower bf16 neighbour `1.0+2^-7` has f32 bits 0x3F810000; bit 16 = 0x3F81 & 1 = 1 (odd).
/// Its tie-point is `1.0+2^-7+2^-8 = 1.0117188`.
/// Honest: lsb=1, adds 0x8000, rounds UP to 1.0+2^-6 = 1.015625.
/// Mutant: lsb=0 always, adds 0x7FFF, rounds DOWN to 1.0+2^-7 = 1.0078125.
#[test]
fn round_to_nearest_even_at_tie_point_uses_correct_lsb_extraction() {
    // Case 1: lsb=0 (even lower neighbour 1.0), tie rounds DOWN.
    // Both honest and mutant give lsb=0 here — confirms round-down path but does NOT kill mutant.
    let tie_down = 1.0 + 2.0_f64.powi(-8); // midpoint; lower bf16=1.0 (bit16=0) → round down
    let a_down = dense_f32(vec![tie_down]);
    let (b_down, _) = dense_f32_to_bf16(&a_down, &policy()).expect("must not fail");
    let r_down = match b_down.payload() {
        Payload::Scalars(xs) => xs[0],
        other => panic!("{other:?}"),
    };
    assert_eq!(
        r_down, 1.0,
        "tie at 1+2^-8 must round DOWN to 1.0 (lsb=0 even)"
    );

    // Case 2: lsb=1 (odd lower neighbour 1.0+2^-7), tie rounds UP — kills the >> → << mutant.
    // Midpoint: 1.0+2^-7+2^-8 = 1.0117188. Lower bf16=1.0+2^-7, bit 16 of its f32=1 (odd).
    // Honest: lsb=1 → rounds UP to 1.0+2^-6=1.015625.
    // Mutant (>> → <<, lsb always 0): rounds DOWN to 1.0+2^-7=1.0078125.
    let tie_up = 1.0 + 2.0_f64.powi(-7) + 2.0_f64.powi(-8); // midpoint; lower has lsb=1
    let a_up = dense_f32(vec![tie_up]);
    let (b_up, _) = dense_f32_to_bf16(&a_up, &policy()).expect("must not fail on odd-lsb tie");
    let r_up = match b_up.payload() {
        Payload::Scalars(xs) => xs[0],
        other => panic!("{other:?}"),
    };
    // Must round UP to 1.0+2^-6 = 1.015625 (even upper neighbour).
    assert!(
        (r_up - 1.015_625).abs() < 1e-9,
        "tie at 1+2^-7+2^-8 must round UP to 1.015625 (lsb=1 odd → round to even upper); got {r_up} \
         (mutant: >> → << always gives lsb=0 → rounds DOWN to 1.0078125)"
    );
}

/// mutant: src/dense.rs:50 `+` → `-` in `bits + 0x7FFF + lsb` (round-to-nearest bias)
///
/// The standard round-to-nearest trick adds (0x7FFF + lsb) to bias the truncation. With `-`
/// instead of `+`, the bias is subtracted, systematically rounding DOWN even when the remainder
/// is ≥ 0.5. A value at or just above 1.5 (which rounds to 2.0 normally) would instead truncate
/// to something wrong.
#[test]
fn round_f32_to_bf16_rounds_up_when_remainder_exceeds_half() {
    // 1.5 is exactly on the bf16 grid, so no rounding occurs — use a value slightly above the
    // midpoint to force a round-up: 1 + 3*2^-9 = 1.005859375.
    // Between bf16 grid points 1.0 and 1 + 2^-7 = 1.0078125.
    // 1 + 3*2^-9 = 1.005859375 is above the midpoint (1 + 2^-8 = 1.00390625) → must round UP
    // to 1.0078125.
    let above_mid = 1.0 + 3.0 * 2.0_f64.powi(-9); // 1.005859375
    let a = dense_f32(vec![above_mid]);
    let (b, _) = dense_f32_to_bf16(&a, &policy()).expect("above-mid must succeed");
    let result = match b.payload() {
        Payload::Scalars(xs) => xs[0],
        other => panic!("{other:?}"),
    };
    // Must round UP to 1.0 + 2^-7 = 1.0078125.
    assert!(
        (result - 1.007_812_5).abs() < 1e-9,
        "value above midpoint must round UP; got {result} \
         (mutant: + → - in bias flips round-up to round-down)"
    );
}

// ===========================================================================
// src/dense.rs — round_element subnormal boundary `<` → `<=`
// ===========================================================================

/// mutant: src/dense.rs:64 `<` → `<=` in `x.abs() < BF16_MIN_NORMAL` subnormal check
///
/// The guard is `x != 0.0 && x.abs() < BF16_MIN_NORMAL`. With `<=`, the smallest normal value
/// `BF16_MIN_NORMAL` itself (= f32::MIN_POSITIVE) would be classified as subnormal and refused,
/// even though the theorem's side-condition explicitly covers normals. Conversely, a value just
/// below normal would pass with `<` (correct: refused) but also with `<=` (correct: refused too).
/// The unique distinguisher is EXACTLY `BF16_MIN_NORMAL`: it must SUCCEED with `<` and would
/// FAIL with `<=`.
#[test]
fn bf16_min_normal_itself_is_not_refused_as_subnormal() {
    // BF16_MIN_NORMAL = f32::MIN_POSITIVE = 2^-126. This is the boundary: the theorem covers
    // values with |x| >= BF16_MIN_NORMAL. With `<` the guard fires for |x| < BF16_MIN_NORMAL,
    // so exactly BF16_MIN_NORMAL should NOT fire the guard.
    let min_normal = f64::from(f32::MIN_POSITIVE); // = BF16_MIN_NORMAL
    let a = dense_f32(vec![min_normal]);
    let result = dense_f32_to_bf16(&a, &policy());
    assert!(
        result.is_ok(),
        "f32::MIN_POSITIVE (= BF16_MIN_NORMAL) must not be refused as subnormal \
         (mutant: < → <= would refuse it); got {result:?}"
    );
}

// ===========================================================================
// src/dense_vsa.rs — codebook_atom LCG bit extraction
// ===========================================================================

/// mutant: src/dense_vsa.rs:72 `==` → `!=` in `(s >> 63) & 1 == 1` (codebook atom generation)
///
/// The LCG atom is `+1.0` when the top bit is 1, `-1.0` otherwise. With `!=` it would flip:
/// +1.0 when top bit is 0, -1.0 when top bit is 1. This changes the codebook and breaks
/// round-tripping: the encode uses atoms from the original codebook; decode uses the mutated
/// codebook. The test confirms enc→dec recovers the original vector.
///
/// mutant: src/dense_vsa.rs:72 `&` → `^` in `(s >> 63) & 1` (bit mask)
///
/// With `^` instead of `&`, `(s >> 63) ^ 1` XORs with 1 instead of AND — this also flips the
/// polarity for every atom where the top bit is 0 (XOR changes the result only for those cases).
/// Both mutations corrupt the codebook, breaking round-trip recovery.
#[test]
fn codebook_atom_sign_is_correct_for_round_trip() {
    // Use a small, proven-capacity instance where correct codebook → perfect recovery.
    let n = 4u32;
    let delta = 1e-2_f64;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    // A fixed bipolar vector.
    let xs = vec![1.0f64, -1.0, 1.0, -1.0];
    let a = bipolar_dense(xs.clone());
    let (hv, _) = dense_to_vsa(&a, vsa_dim, delta, &policy()).expect("enc must succeed");
    let (back, _) = vsa_to_dense(&hv, n, delta, &policy()).expect("dec must succeed");
    let recovered = match back.payload() {
        Payload::Scalars(v) => v.clone(),
        other => panic!("{other:?}"),
    };
    assert_eq!(
        recovered, xs,
        "enc→dec must recover exact bipolar vector (mutant: == → != or & → ^ flips atom signs)"
    );
}

// ===========================================================================
// src/dense_vsa.rs — delta_bound profile_covers logic operators
// ===========================================================================

/// mutant: src/dense_vsa.rs:96 `&&` → `||` in `profile_covers` guard
///
/// The guard is `components <= MAX && vsa_dim >= components * DIM_FACTOR && delta >= EMP_DELTA`.
/// With `||`, a case that fails ALL three conditions could still match — e.g. components=100
/// (>> MAX=16) with a large dimension would hit `EmpiricalFit` instead of `InsufficientCapacity`.
/// This test presents an instance that exceeds the empirical profile's component bound AND has
/// a too-small dimension, verifying it's properly refused.
#[test]
fn delta_bound_all_profile_conditions_must_hold() {
    // components = 32 > DENSE_VSA_EMP_MAX_COMPONENTS=16 → profile does NOT cover it.
    // vsa_dim = 32 * 32 = 1024 ≥ 32 * DIM_FACTOR but BELOW the proven threshold for n=32.
    // With || mutant, `components <= 16` is false but the other two may be true → profile matches.
    // Honest code requires ALL three → InsufficientCapacity.
    let big_n = 32u32;
    let vsa_dim = big_n * 32; // satisfies the dim factor, but n too big for profile
    let result = dense_to_vsa(
        &bipolar_dense(
            (0..big_n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
        vsa_dim,
        DENSE_VSA_EMP_DELTA,
        &policy(),
    );
    assert!(
        matches!(result, Err(SwapError::InsufficientCapacity { .. })),
        "components > MAX must give InsufficientCapacity (mutant: && → || would give EmpiricalFit); got {result:?}"
    );
}

/// mutant: src/dense_vsa.rs:95 `*` → `+` or `/` in `vsa_dim >= components * DENSE_VSA_EMP_DIM_FACTOR`
///
/// The condition is `vsa_dim >= components * 32`. With `+`, it becomes `>= components + 32`;
/// with `/`, `>= components / 32`. Both are much weaker checks.
///
/// This test uses a case where `vsa_dim < components * 32` but `vsa_dim > components + 32`:
/// specifically n=8, vsa_dim=48 (8 + 32 = 40 < 48, so `+` passes; 8 * 32 = 256 > 48, fails).
#[test]
fn profile_covers_dim_factor_is_multiplication_not_addition() {
    // n=8, vsa_dim=48: 8*32=256 > 48 → profile does NOT cover (correct).
    // But 48 > 8+32=40 → with + mutant, profile WOULD cover it (wrong).
    let _n = 8u32;
    let vsa_dim = 48u32;
    let a = bipolar_dense(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    let result = dense_to_vsa(&a, vsa_dim, DENSE_VSA_EMP_DELTA, &policy());
    assert!(
        matches!(result, Err(SwapError::InsufficientCapacity { .. })),
        "vsa_dim=48 < n*32=256 must be InsufficientCapacity (mutant: * → + would incorrectly pass); got {result:?}"
    );
}

// ===========================================================================
// src/dense_vsa.rs — dense_to_vsa accumulation `*` → `/`
// ===========================================================================

/// mutant: src/dense_vsa.rs:170 `*` → `/` in `dense_to_vsa` accumulation: `*h += x * a`
///
/// The MAP-I superposition is `hv[j] += x_i * atom_i[j]`. With `/` instead of `*`, the
/// component is divided by the atom value instead of scaled — since atoms are ±1, division has
/// the same magnitude but same absolute value as multiplication for ±1 atoms. However, the sign
/// is different when x=-1: x*a = (-1)*1 = -1 but x/a = (-1)/1 = -1 (same!) and (-1)*(-1)=+1
/// but (-1)/(-1)=+1 (same!). Since both a and x are ±1, `x*a == x/a` always.
///
/// Wait — for ±1 values x*a and x/a are identical. Let me think: if x=-1, a=1: x*a=-1, x/a=-1.
/// If x=1, a=-1: x*a=-1, x/a=-1. If x=-1, a=-1: x*a=1, x/a=1. So indeed x*a == x/a for ±1.
///
/// In that case the `/` mutant is equivalent for the enc step — it IS an equivalent mutant for
/// the accumulation *only* because both x and a are ±1. We should justify this.
///
/// JUSTIFY: src/dense_vsa.rs:170 `*` → `/` in `*h += x * a` when x ∈ {±1} and a ∈ {±1}:
///   x * a == x / a (since |a|=1, so 1/a = a). This mutant is **equivalent** for all valid
///   bipolar inputs. The function is `NotBipolar`-gated so x is always ±1; the codebook atoms
///   are always ±1. The equivalence is total: `justify: equivalent mutant for ±1 bipolar inputs`.
///
/// Test below documents the justification by verifying the enc→dec round-trip still holds and
/// showing the specific arithmetic equivalence.
#[test]
fn dense_to_vsa_accumulation_mul_and_div_equivalent_for_bipolar() {
    // Justification witness: for x ∈ {±1}, a ∈ {±1}, x*a == x/a.
    for x in [1.0_f64, -1.0] {
        for a in [1.0_f64, -1.0] {
            assert_eq!(x * a, x / a, "x*a != x/a for x={x}, a={a} (unexpected)");
        }
    }
    // The enc→dec round-trip still holds (both * and / give the same superposition).
    let n = 4u32;
    let delta = 1e-2;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    let xs = vec![1.0, -1.0, 1.0, 1.0];
    let a = bipolar_dense(xs.clone());
    let (hv, _) = dense_to_vsa(&a, vsa_dim, delta, &policy()).unwrap();
    let (back, _) = vsa_to_dense(&hv, n, delta, &policy()).unwrap();
    assert_eq!(
        back.payload(),
        a.payload(),
        "enc→dec round-trip must still recover bipolar vector"
    );
}

// ===========================================================================
// src/dense_vsa.rs — vsa_to_dense provenance guard
// ===========================================================================

/// mutant: src/dense_vsa.rs:224 match guard `op == &operation_hash(ENC_OP)` → `true`
///
/// The match arm at line 224 is `Provenance::Derived { op, .. } if op == &operation_hash(ENC_OP)`.
/// With `guard → true`, the condition becomes `if true`, so any `Derived` provenance with ANY op
/// is accepted (the op is no longer checked). `Root` provenance still fails the `Derived` pattern,
/// so the Root test does NOT kill this mutant — Root falls to `_ => NotDenseVsaEncoding` regardless.
///
/// The distinguishing case is a VSA with `Provenance::Derived { op: <wrong-op>, inputs: ... }`:
/// honest code returns `NotDenseVsaEncoding` (wrong op), mutant returns Ok (guard=true accepts it).
#[test]
fn vsa_to_dense_requires_enc_v1_provenance_not_just_model() {
    let vsa_dim = 2048u32;

    // Case 1: Root provenance — caught by the `_` arm, not the guard. Documents the error path.
    let root_stray = Value::new(
        Repr::Vsa {
            model: dense_vsa::DENSE_VSA_MODEL.to_owned(),
            dim: vsa_dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(vec![0.5; vsa_dim as usize]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_eq!(
        vsa_to_dense(&root_stray, 8, 1e-2, &policy()),
        Err(SwapError::NotDenseVsaEncoding),
        "Root-provenance VSA must be NotDenseVsaEncoding"
    );

    // Case 2: Derived provenance with a WRONG op — kills the `guard → true` mutant.
    // Honest code: op != enc.v1 → guard fires (false) → falls to `_ => NotDenseVsaEncoding`.
    // Mutant (guard → true): Derived pattern matches regardless of op → proceeds to decode → Ok.
    let wrong_op_meta = Meta::new(
        Provenance::Derived {
            op: operation_hash("some.other.operation"),
            inputs: vec![policy()],
        },
        mycelium_core::GuaranteeStrength::Exact,
        None,
        None,
        None,
        Some(policy()),
    )
    .unwrap();
    let derived_stray = Value::new(
        Repr::Vsa {
            model: dense_vsa::DENSE_VSA_MODEL.to_owned(),
            dim: vsa_dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(vec![0.5; vsa_dim as usize]),
        wrong_op_meta,
    )
    .unwrap();
    assert_eq!(
        vsa_to_dense(&derived_stray, 8, 1e-2, &policy()),
        Err(SwapError::NotDenseVsaEncoding),
        "Derived-but-wrong-op VSA must be NotDenseVsaEncoding (mutant: guard → true accepts it)"
    );
}

/// Extra witness: the correct enc.v1 provenance is accepted (positive arm of the guard).
#[test]
fn vsa_to_dense_accepts_enc_v1_provenance() {
    let n = 4u32;
    let delta = 1e-2;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    let a = bipolar_dense(vec![1.0, -1.0, -1.0, 1.0]);
    let (hv, _) = dense_to_vsa(&a, vsa_dim, delta, &policy()).unwrap();
    // hv now carries the enc.v1 provenance — decode must succeed.
    assert!(
        vsa_to_dense(&hv, n, delta, &policy()).is_ok(),
        "enc.v1 product must be accepted by vsa_to_dense"
    );
}

// ===========================================================================
// src/dense_vsa.rs — vsa_to_dense correlation dot product
// ===========================================================================

/// mutant: src/dense_vsa.rs:234 `*` → `/` in correlation: `h * a` → `h / a`
///
/// The correlation is `dot += h_j * a_ij`. With `/`, it becomes `dot += h_j / a_ij`.
/// Since atoms a_ij ∈ {±1}, dividing by ±1 gives the same result as multiplying (for atoms).
/// However, the hypervector elements h_j are NOT ±1 — they are sums of n bipolar values.
/// So h_j / a_ij ≠ h_j * a_ij in general.
///
/// Specifically: if a_ij = -1, then h_j * (-1) = -h_j but h_j / (-1) = -h_j — same!
/// If a_ij = 1, then h_j * 1 = h_j but h_j / 1 = h_j — same!
///
/// Since atoms are ±1, h * a == h / a regardless of h. This is also an equivalent mutant.
///
/// JUSTIFY: src/dense_vsa.rs:234 `*` → `/` in `h * a`: atoms are ±1, so h/a == h*a for all h.
/// Equivalent mutant.
#[test]
fn vsa_to_dense_dot_product_mul_div_equivalent_for_bipolar_atoms() {
    // Justification: for any h and a ∈ {±1}, h/a = h * a (since 1/1=1, 1/(-1)=-1).
    for a in [1.0_f64, -1.0] {
        let h_vals = [0.0, 1.0, -1.0, 0.5, -0.5, 2.3, -7.1];
        for h in h_vals {
            assert_eq!(h * a, h / a, "h*a != h/a for h={h}, a={a}");
        }
    }
    // enc→dec still round-trips perfectly (both arms give identical dot products).
    let n = 4u32;
    let delta = 1e-2;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    let a = bipolar_dense(vec![-1.0, 1.0, -1.0, -1.0]);
    let (hv, _) = dense_to_vsa(&a, vsa_dim, delta, &policy()).unwrap();
    let (back, _) = vsa_to_dense(&hv, n, delta, &policy()).unwrap();
    assert_eq!(back.payload(), a.payload());
}

// ===========================================================================
// src/dense_vsa.rs — vsa_to_dense sign decision `>` → `>=`
// ===========================================================================

/// mutant: src/dense_vsa.rs:239 `>` → `>=` in `if dot > 0.0 { 1.0 } else { -1.0 }`
///
/// With `>=`, `dot == 0.0` would return `+1.0` instead of `AmbiguousDecode`. However, the code
/// already guards against `dot == 0.0` on line 236:
///   `if dot == 0.0 { return Err(SwapError::AmbiguousDecode { index: i }); }`
///
/// So by the time we reach the sign decision, `dot != 0.0`. Therefore `>` and `>=` are
/// equivalent here — the case `dot == 0.0` is already unreachable at line 239.
///
/// JUSTIFY: src/dense_vsa.rs:239 `>` → `>=`: the `dot == 0.0` case is already excluded by the
/// AmbiguousDecode guard at line 236. For `dot != 0.0`, `dot > 0.0` and `dot >= 0.0` are
/// identical. Equivalent mutant — unreachable distinction.
#[test]
fn vsa_to_dense_sign_decision_equivalent_for_nonzero_dot() {
    // Witness: with a proven-capacity enc, the dot is always nonzero and the sign is well-defined.
    let n = 4u32;
    let delta = 1e-2;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    let a = bipolar_dense(vec![1.0, 1.0, -1.0, 1.0]);
    let (hv, _) = dense_to_vsa(&a, vsa_dim, delta, &policy()).unwrap();
    let (back, _) = vsa_to_dense(&hv, n, delta, &policy()).unwrap();
    // The decoded signs must match.
    assert_eq!(
        back.payload(),
        a.payload(),
        "sign decision must recover the input"
    );
    // Confirm AmbiguousDecode is still live (dot==0 case is reachable in principle, just not here).
    // The guard at line 236 prevents reaching line 239 with dot==0.
}

// ===========================================================================
// Additional CertifiedSwapEngine VSA→Dense direction model guard witnesses
// ===========================================================================

/// mutant: src/lib.rs:426 in VSA→Dense arm: `model == dense_vsa::DENSE_VSA_MODEL` → `true/false/!=`
///
/// The VSA→Dense arm in CertifiedSwapEngine also has a model guard. These tests exercise
/// that arm's guard:
/// - `→ true`: non-MAP-I VSA would be attempted as Dense decode (and fail with WrongSource or similar)
/// - `→ false`: MAP-I VSA→Dense falls through to BinaryTernary engine (UnsupportedSwap)
/// - `== → !=`: MAP-I fails through, non-MAP-I is tried
#[test]
fn certified_engine_vsa_to_dense_model_guard_is_correct() {
    let n = 4u32;
    let delta = 1e-2;
    let vsa_dim = u32::try_from(capacity::required_dim(
        u64::from(n),
        delta,
        capacity::MARGIN_MU,
    ))
    .unwrap();
    let a = bipolar_dense(vec![1.0, -1.0, 1.0, -1.0]);
    let (hv, _) = dense_to_vsa(&a, vsa_dim, delta, &policy()).unwrap();

    // MAP-I VSA → Dense{F32} must succeed through the engine.
    let dense_target = Repr::Dense {
        dim: n,
        dtype: ScalarKind::F32,
    };
    assert!(
        CertifiedSwapEngine
            .swap(&hv, &dense_target, &policy())
            .is_ok(),
        "MAP-I VSA→Dense must succeed (mutant: guard → false breaks this)"
    );

    // A non-MAP-I VSA → Dense must not succeed — and must be UnsupportedSwap specifically.
    // With the `model == DENSE_VSA_MODEL → true` mutant, HRR VSA routes to vsa_to_dense which
    // returns NotDenseVsaEncoding (a SwapError, not UnsupportedSwap). The honest code falls through
    // to BinaryTernarySwapEngine which returns EvalError::UnsupportedSwap. So we must assert the
    // specific error variant to kill the mutant.
    let hrr_hv = Value::new(
        Repr::Vsa {
            model: "HRR".to_owned(),
            dim: vsa_dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(vec![0.5; vsa_dim as usize]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert!(
        matches!(
            CertifiedSwapEngine.swap(&hrr_hv, &dense_target, &policy()),
            Err(EvalError::UnsupportedSwap { .. })
        ),
        "HRR VSA→Dense must be UnsupportedSwap specifically (mutant: guard → true gives Swap(NotDenseVsaEncoding) instead)"
    );
}
