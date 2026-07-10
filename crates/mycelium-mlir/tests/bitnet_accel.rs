//! M-728 — the **BitNet capability-flag + never-silent graceful-degradation** differential
//! (FR-C3; RFC-0029 §7.4; ADR-009; G2/VR-5).
//!
//! [`mycelium_mlir::accelerated_ternary_dot`] runs the ternary dot product through the M-728
//! capability gate. This suite pins the three obligations of M-728's Definition of Done:
//!
//! 1. **Correctness (`Empirical`):** the value equals the reference ternary dot
//!    ([`mycelium_mlir::ternary_dot_ref`]) **on either path** — accelerated or degraded — over a mixed
//!    corpus.
//! 2. **Explicit capability flag:** the path taken is governed by the compile-time feature
//!    (`bitnet-accel`) AND the runtime capability — and the chosen path is **recorded** in the
//!    [`AccelOutcome`], queryable and `EXPLAIN`-able.
//! 3. **Never-silent graceful degradation (G2):** when the capability is absent (the default build's
//!    feature-off case, or a host without `clang`), the reference path runs and the outcome records
//!    `Path::Reference(reason)` with an explicit reason — never a silent slow path, never an error.
//!
//! The default build (`--features` off) exercises the **degradation** branch deterministically; the
//! accelerated branch is exercised under `cargo test --features bitnet-accel` (skips `clang`-absent).
//!
//! **Guarantee:** the value is `Exact` on both paths (integer dot product); the *equivalence* of the
//! two paths is `Empirical` (this differential), never upgraded to `Proven` (VR-5).

use mycelium_core::Trit;
use mycelium_mlir::{
    accelerated_ternary_dot, ternary_dot_ref, AccelPath, BitnetCapability, DegradeReason,
};

/// Deterministic ternary/activation test data (small LCGs) — fixed, not a statistical sample.
/// Mirrors the generators in `bitnet.rs`'s tests so the corpus is the same shape.
fn weights(n: usize) -> Vec<Trit> {
    let mut s = 0x1234_5678_u64;
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
    let mut s = 0x9E37_79B9_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((s >> 40) % 201) as i64 - 100) as i32
        })
        .collect()
}

const SIZES: [usize; 7] = [1, 4, 5, 7, 64, 256, 1000];

/// M-728 correctness: the gated dot product equals the reference oracle **on whichever path runs** —
/// accelerated or degraded. This is the never-silent contract's correctness half: degradation must
/// never change the answer (G2). `Empirical`.
#[test]
fn accelerated_dot_equals_reference_on_either_path() {
    for n in SIZES {
        let w = weights(n);
        let x = activations(n);
        let expected = ternary_dot_ref(&w, &x);
        match accelerated_ternary_dot(&w, &x) {
            Ok(outcome) => {
                // Mutant-witness: a wrong kernel decode OR a buggy degradation would diverge here.
                assert_eq!(
                    outcome.value, expected,
                    "n={n}: gated dot ({}) ≠ reference ({expected}) on path {:?}",
                    outcome.value, outcome.path
                );
            }
            Err(e) => panic!("n={n}: accelerated_ternary_dot errored unexpectedly: {e}"),
        }
    }
}

/// M-728 never-silent: the outcome **records** which path ran, with an `EXPLAIN`-able reason — a
/// degradation is never silent (G2). The recorded path must be consistent with the detected capability
/// (and, in the default build, with the feature being OFF).
#[test]
fn the_chosen_path_is_recorded_and_consistent_with_capability() {
    let cap = BitnetCapability::detect();
    let w = weights(32);
    let x = activations(32);
    let outcome = accelerated_ternary_dot(&w, &x).expect("dot");

    // The recorded path must match the capability's verdict — no silent divergence between "what the
    // gate says" and "what ran".
    assert_eq!(
        outcome.was_accelerated(),
        cap.is_accelerated(),
        "recorded path {:?} disagrees with capability {} — a silent mismatch (G2 violation)",
        outcome.path,
        cap.explain()
    );

    // The EXPLAIN strings are non-empty and name the path (auditable, no black box — NFR-1).
    assert!(!outcome.explain().is_empty());
    assert!(!cap.explain().is_empty());

    match &outcome.path {
        AccelPath::Accelerated { .. } => {
            // Only reachable under the feature AND a present toolchain — the detected capability
            // (a runtime value) must agree that acceleration is engaged (no silent gate breach).
            assert!(
                BitnetCapability::detect().is_accelerated(),
                "accelerated path but the capability says not accelerated — gate breached"
            );
            assert!(outcome.explain().contains("ACCELERATED"));
        }
        AccelPath::Reference(reason) => {
            // The reason must be explicit and EXPLAIN-able — never a bare silent fallback.
            assert!(!reason.explain().is_empty());
            assert!(outcome.explain().contains("REFERENCE"));
            assert!(outcome.explain().contains("graceful degradation"));
        }
    }
}

/// M-728 capability flag: the **compile-time** gate is honest. In the **default** build (feature off)
/// the capability is never accelerated, the path is always a recorded `FeatureDisabled` degradation,
/// and `ACCEL_FEATURE_ENABLED` is `false`. This is the deterministic feature-off assertion (it does not
/// depend on the toolchain at all).
#[test]
#[cfg(not(feature = "bitnet-accel"))]
fn feature_off_always_degrades_explicitly_to_reference() {
    // This whole test is `#[cfg(not(feature = "bitnet-accel"))]`, so the feature is statically off
    // here; we assert the *behaviour* (a runtime capability that never accelerates), not the const
    // (which clippy rightly flags as a constant assertion under the cfg gate).
    let cap = BitnetCapability::detect();
    assert!(
        !cap.is_accelerated(),
        "feature-off build must never accelerate: {}",
        cap.explain()
    );

    let w = weights(50);
    let x = activations(50);
    let outcome = accelerated_ternary_dot(&w, &x).expect("dot");
    // The degradation is explicit and recorded as FeatureDisabled — never a silent slow path (G2).
    assert_eq!(
        outcome.path,
        AccelPath::Reference(DegradeReason::FeatureDisabled),
        "feature-off must degrade explicitly to the reference path with the FeatureDisabled reason"
    );
    // …and still produces the correct value.
    assert_eq!(outcome.value, ternary_dot_ref(&w, &x));
}

/// M-728 capability flag: under `--features bitnet-accel` the build *opted in*, so the compile-time
/// half of the gate is open. The runtime half (the JIT toolchain) is then what decides — and **either
/// outcome is honest**: accelerated (toolchain present) or a recorded `RuntimeUnavailable` degradation
/// (toolchain absent). Both must produce the correct value; neither is silent.
#[test]
#[cfg(feature = "bitnet-accel")]
fn feature_on_accelerates_or_degrades_explicitly_by_runtime() {
    // This whole test is `#[cfg(feature = "bitnet-accel")]`, so the compile-time half of the gate is
    // statically open here; we assert the *runtime behaviour* below, not the const (clippy flags a
    // constant assertion). The runtime capability decides accelerate-vs-degrade.
    let w = weights(64);
    let x = activations(64);
    let expected = ternary_dot_ref(&w, &x);
    let outcome = accelerated_ternary_dot(&w, &x).expect("dot");
    assert_eq!(
        outcome.value, expected,
        "feature-on value must equal the reference"
    );

    match outcome.path {
        // Toolchain present — the accelerated kernel ran and agrees with the oracle (Empirical).
        AccelPath::Accelerated { .. } => {
            assert!(outcome.was_accelerated());
        }
        // Toolchain absent — explicit, recorded RuntimeUnavailable degradation (never silent, G2).
        AccelPath::Reference(reason) => {
            assert_eq!(
                reason,
                DegradeReason::RuntimeUnavailable,
                "feature-on degradation must be RuntimeUnavailable (not FeatureDisabled)"
            );
        }
    }
}
