//! M-340 — specialized↔generic kernel differential, **through the shared M-210 checker** (NFR-7;
//! VR-4; RR-12; ADR-009; phase-3.md Batch L).
//!
//! The JIT runtime-specialization layer (`mycelium_mlir::compile_specialized_dot`: bake the weights
//! in → `clang -shared -O2` → `dlopen` → call) computes the *same* exact ternary dot product as the
//! generic runtime-pointer kernel (`compile_bitnet_dot_for`) and the packing-independent oracle
//! (`ternary_dot_ref`). This test asserts the two compiled paths agree on the observable scalar
//! result **and validates the pair through the single shared M-210 checker** (`ObservationalEquiv`,
//! `Certificate::exact()`) — the same checker the AOT/JIT differentials use — so a specialization bug
//! is caught by the same machinery, not a bespoke assertion. Skips when `clang` is absent.

mod common;
use common::i64_value;

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, PackScheme, Trit};
use mycelium_mlir::{
    compile_bitnet_dot_for, compile_specialized_dot, pack_trits, ternary_dot_ref, AotError,
};

/// Deterministic ternary weights / int activations (fixed LCGs, not a statistical sample).
fn weights(n: usize) -> Vec<Trit> {
    let mut s = 0x0F1E_2D3C_u64;
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
    let mut s = 0x2468_ACE0_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((s >> 40) % 201) as i64 - 100) as i32
        })
        .collect()
}

#[test]
fn specialized_and_generic_agree_through_the_shared_checker() {
    for n in [1usize, 4, 17, 64, 333, 1000] {
        let w = weights(n);
        let x = activations(n);

        // Generic runtime-pointer kernel (I2_S packing) vs the weight-specialized kernel.
        let generic = match compile_bitnet_dot_for(PackScheme::I2S) {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return, // environment skip
            Err(e) => panic!("generic compile failed at n={n}: {e}"),
        };
        let spec = match compile_specialized_dot(&w) {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("specialized compile failed at n={n}: {e}"),
        };

        let packed = pack_trits(&w, PackScheme::I2S);
        let generic_sum = generic.call(&packed, &x, n).expect("generic kernel runs");
        let spec_sum = spec.call(&x).expect("specialized kernel runs");

        // Both must equal the oracle (a wrong specialization would diverge here).
        let oracle = ternary_dot_ref(&w, &x);
        assert_eq!(generic_sum, oracle, "generic diverged from oracle at n={n}");
        assert_eq!(
            spec_sum, oracle,
            "specialized diverged from oracle at n={n}"
        );

        // The discriminating check: route the two compiled paths' observables through the *single
        // shared* M-210 checker. Mutant-witness: a specialized kernel that dropped a lane or flipped
        // a sign would produce a different `Value` and the checker would report `NotValidated`.
        assert_eq!(
            check(
                &i64_value(generic_sum),
                &i64_value(spec_sum),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "n={n}: the shared checker must validate the generic↔specialized pair"
        );
    }
}

#[test]
fn the_differential_discriminates_a_wrong_specialization() {
    // Guard 7: a pass is only meaningful if the checker can *fail*. Specialize on the negated weights
    // (a deliberate mutant) and confirm the shared checker reports the mismatch against the generic
    // kernel — so the green test above is not vacuous.
    let n = 64;
    let w = weights(n);
    let x = activations(n);

    let generic = match compile_bitnet_dot_for(PackScheme::I2S) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("generic compile failed: {e}"),
    };
    // Negated weights: a different (non-trivial) dot product for this mixed data.
    let negated: Vec<Trit> = w
        .iter()
        .map(|t| match t {
            Trit::Pos => Trit::Neg,
            Trit::Neg => Trit::Pos,
            Trit::Zero => Trit::Zero,
        })
        .collect();
    let mutant = match compile_specialized_dot(&negated) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("mutant compile failed: {e}"),
    };

    let packed = pack_trits(&w, PackScheme::I2S);
    let generic_sum = generic.call(&packed, &x, n).expect("generic runs");
    let mutant_sum = mutant.call(&x).expect("mutant runs");
    assert_ne!(
        generic_sum, mutant_sum,
        "negated weights must change the dot product on this data"
    );
    assert!(
        matches!(
            check(
                &i64_value(generic_sum),
                &i64_value(mutant_sum),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the shared checker must reject a wrong specialization"
    );
}
