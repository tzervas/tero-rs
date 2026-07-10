//! In-crate white-box tests for [`crate::accel`] (M-728; CLAUDE.md test-layout rule). Covers the
//! capability detection + EXPLAIN surface and the private `probe_runtime`; the behavioural
//! accel↔reference differential lives in `tests/bitnet_accel.rs`.

use crate::accel::*;
use mycelium_core::Trit;

#[test]
fn capability_detect_is_consistent_with_the_feature_flag() {
    let cap = BitnetCapability::detect();
    // The accelerated verdict implies the feature was opted in (the compile-time half of the gate);
    // it can never be accelerated without the feature (G2: no silent engagement). Assert the
    // *runtime* EXPLAIN reflects this, not the const itself (clippy flags a constant assertion).
    let ex = cap.explain();
    if cap.is_accelerated() {
        // An accelerated verdict must say the feature is ON (the gate's compile-time half opened).
        assert!(
            ex.contains("feature `bitnet-accel` ON"),
            "accelerated verdict but EXPLAIN does not report the feature ON — gate breach: {ex}"
        );
    }
    // The EXPLAIN names the feature state and the verdict (no black box — NFR-1).
    assert!(ex.contains("bitnet-accel"));
    assert!(ex.contains("ACCELERATED") || ex.contains("REFERENCE"));
}

#[test]
fn probe_runtime_matches_a_real_compile() {
    // The runtime probe is a real compile, not a guess — it agrees with whether the bitnet kernel
    // actually compiles right now (VR-5: "available" means "we compiled one").
    assert_eq!(probe_runtime(), crate::bitnet::compile_bitnet_dot().is_ok());
}

#[test]
fn degrade_reasons_are_explainable() {
    // Every degradation reason carries a non-empty EXPLAIN — a fallback is never a bare silent skip.
    assert!(!DegradeReason::FeatureDisabled.explain().is_empty());
    assert!(!DegradeReason::RuntimeUnavailable.explain().is_empty());
    assert_ne!(
        DegradeReason::FeatureDisabled.explain(),
        DegradeReason::RuntimeUnavailable.explain()
    );
}

#[test]
fn outcome_records_the_path_and_explains_it() {
    let w = vec![Trit::Neg, Trit::Zero, Trit::Pos];
    let x = vec![7, 9, 4];
    let outcome = accelerated_ternary_dot(&w, &x).expect("dot");
    // -7 + 0 + 4 = -3 on either path (value is path-independent — degradation never changes it).
    assert_eq!(outcome.value, -3);
    // The recorded path is queryable + EXPLAIN-able (never a silent fallback, G2).
    assert_eq!(
        outcome.was_accelerated(),
        matches!(outcome.path, Path::Accelerated { .. })
    );
    assert!(!outcome.explain().is_empty());
}
