//! M-360 — hand-vectorized↔scalar dot-kernel differential corpus, **through the shared M-210
//! checker** (NFR-7; VR-4; RR-12; ADR-009; phase-3.md Batch L / E1).
//!
//! The SIMD kernels for **all three bitnet packings** (I2_S, TL1, TL2) must compute the *same* exact
//! ternary dot product as the corresponding scalar kernel (`compile_bitnet_dot_for(scheme)`,
//! the **oracle**) and the packing-independent `ternary_dot_ref`. The vector unpack is the
//! correctness-critical part, so this is a **corpus** that brackets the vector body width and the
//! tail:
//! - I2_S / TL1 (8-wide body): n ∈ {0,1,7,8,9,15,16,17,31,33,64,255,256,257,1000}
//! - TL2 (4-group body = 12 trits/iter): n ∈ {0,1,3,9,11,12,13,23,24,25,35,36,37,64,257,1000}
//!
//! Each pair is validated through the single shared M-210 checker (`ObservationalEquiv`,
//! `Certificate::exact()`), plus a discrimination test so a green pass is not vacuous (guard 7).
//! Skips when `clang` is absent.

mod common;
use common::i64_value;

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, PackScheme, Trit};
use mycelium_mlir::{
    compile_bitnet_dot_for, compile_bitnet_dot_simd, compile_bitnet_dot_simd_tl1,
    compile_bitnet_dot_simd_tl2, pack_trits, ternary_dot_ref, AotError,
};

fn weights(n: usize) -> Vec<Trit> {
    let mut s = 0x5151_2727_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            match (s >> 33) % 3 {
                0 => Trit::Neg,
                1 => Trit::Zero,
                _ => Trit::Pos,
            }
        })
        .collect()
}
fn activations(n: usize) -> Vec<i32> {
    let mut s = 0x1A2B_3C4D_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((s >> 40) % 201) as i64 - 100) as i32
        })
        .collect()
}

#[test]
fn simd_and_scalar_agree_through_the_shared_checker_over_the_corpus() {
    let simd = match compile_bitnet_dot_simd() {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return, // environment skip
        Err(e) => panic!("SIMD compile failed: {e}"),
    };
    let scalar = match compile_bitnet_dot_for(PackScheme::I2S) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("scalar compile failed: {e}"),
    };

    for n in [
        0usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 64, 255, 256, 257, 1000,
    ] {
        let w = weights(n);
        let x = activations(n);
        let packed = pack_trits(&w, PackScheme::I2S);
        let simd_sum = simd.call(&packed, &x, n).expect("SIMD kernel runs");
        let scalar_sum = scalar.call(&packed, &x, n).expect("scalar kernel runs");
        let oracle = ternary_dot_ref(&w, &x);

        assert_eq!(scalar_sum, oracle, "scalar (oracle) diverged at n={n}");
        assert_eq!(simd_sum, oracle, "SIMD diverged from oracle at n={n}");
        // The discriminating check: both compiled paths through the single shared M-210 checker.
        assert_eq!(
            check(
                &i64_value(scalar_sum),
                &i64_value(simd_sum),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "n={n}: the shared checker must validate the scalar↔SIMD pair"
        );
    }
}

#[test]
fn the_differential_discriminates_a_corrupted_buffer() {
    // Guard 7: feed the SIMD and scalar kernels *different* weight buffers and confirm the shared
    // checker reports the mismatch — so the corpus pass above is meaningful, not vacuous.
    let simd = match compile_bitnet_dot_simd() {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("SIMD compile failed: {e}"),
    };
    let scalar = match compile_bitnet_dot_for(PackScheme::I2S) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("scalar compile failed: {e}"),
    };
    let n = 64;
    let x = activations(n);
    let packed_a = pack_trits(&weights(n), PackScheme::I2S);
    // A different weight set → a different (non-trivial) dot product on this data.
    let packed_b = pack_trits(
        &weights(n)
            .iter()
            .map(|t| match t {
                Trit::Pos => Trit::Neg,
                Trit::Neg => Trit::Pos,
                Trit::Zero => Trit::Pos,
            })
            .collect::<Vec<_>>(),
        PackScheme::I2S,
    );
    let scalar_sum = scalar.call(&packed_a, &x, n).expect("scalar runs");
    let simd_sum = simd.call(&packed_b, &x, n).expect("SIMD runs");
    assert_ne!(
        scalar_sum, simd_sum,
        "the two buffers must differ on this data"
    );
    assert!(
        matches!(
            check(
                &i64_value(scalar_sum),
                &i64_value(simd_sum),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the shared checker must reject mismatched dot results"
    );
}

// ── TL1 SIMD differential ───────────────────────────────────────────────────────────────────────

#[test]
fn tl1_simd_and_scalar_agree_through_the_shared_checker_over_the_corpus() {
    // The TL1 SIMD kernel (8-wide body, `select(code==2,-1,code)` decode) must match the scalar
    // TL1 oracle and `ternary_dot_ref` over a corpus bracketing the 8-lane width and the tail.
    let simd = match compile_bitnet_dot_simd_tl1() {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("TL1 SIMD compile failed: {e}"),
    };
    let scalar = match compile_bitnet_dot_for(PackScheme::Tl1) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("TL1 scalar compile failed: {e}"),
    };

    for n in [
        0usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 64, 255, 256, 257, 1000,
    ] {
        let w = weights(n);
        let x = activations(n);
        let packed = pack_trits(&w, PackScheme::Tl1);
        let simd_sum = simd.call(&packed, &x, n).expect("TL1 SIMD kernel runs");
        let scalar_sum = scalar.call(&packed, &x, n).expect("TL1 scalar kernel runs");
        let oracle = ternary_dot_ref(&w, &x);

        assert_eq!(scalar_sum, oracle, "TL1 scalar (oracle) diverged at n={n}");
        assert_eq!(simd_sum, oracle, "TL1 SIMD diverged from oracle at n={n}");
        assert_eq!(
            check(
                &i64_value(scalar_sum),
                &i64_value(simd_sum),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "n={n}: the shared checker must validate the TL1 scalar↔SIMD pair"
        );
    }
}

// ── TL2 SIMD differential ───────────────────────────────────────────────────────────────────────

#[test]
fn tl2_simd_and_scalar_agree_through_the_shared_checker_over_the_corpus() {
    // The TL2 SIMD kernel (4-group = 12-trit body, 5-bit-code bitstream decode) must match the
    // scalar TL2 oracle and `ternary_dot_ref` over a corpus bracketing the 4-group width (12
    // trits) and the tail.
    let simd = match compile_bitnet_dot_simd_tl2() {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("TL2 SIMD compile failed: {e}"),
    };
    let scalar = match compile_bitnet_dot_for(PackScheme::Tl2) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("TL2 scalar compile failed: {e}"),
    };

    for n in [
        0usize, 1, 3, 9, 11, 12, 13, 23, 24, 25, 35, 36, 37, 64, 99, 100, 257, 1000,
    ] {
        let w = weights(n);
        let x = activations(n);
        let packed = pack_trits(&w, PackScheme::Tl2);
        let simd_sum = simd.call(&packed, &x, n).expect("TL2 SIMD kernel runs");
        let scalar_sum = scalar.call(&packed, &x, n).expect("TL2 scalar kernel runs");
        let oracle = ternary_dot_ref(&w, &x);

        assert_eq!(scalar_sum, oracle, "TL2 scalar (oracle) diverged at n={n}");
        assert_eq!(simd_sum, oracle, "TL2 SIMD diverged from oracle at n={n}");
        assert_eq!(
            check(
                &i64_value(scalar_sum),
                &i64_value(simd_sum),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "n={n}: the shared checker must validate the TL2 scalar↔SIMD pair"
        );
    }
}
