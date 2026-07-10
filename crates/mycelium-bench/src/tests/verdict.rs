use crate::backend::{Backend, Outcome};
use crate::timing::Timing;
use crate::verdict::*;
use mycelium_core::{CoreValue, Meta, Payload, Provenance, Repr, Value};

fn byte(b: u8) -> CoreValue {
    let bits: Vec<bool> = (0..8).map(|i| (b >> i) & 1 == 1).collect();
    CoreValue::Repr(
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(bits),
            Meta::exact(Provenance::Root),
        )
        .expect("valid byte"),
    )
}

fn timing(ns: f64) -> Timing {
    Timing {
        ns_per_call: ns,
        iters: 1000,
        batches: 5,
        ns_per_call_worst: ns,
    }
}

#[test]
fn equal_values_and_faster_backend_is_a_speed_win() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::value_outcome(byte(0xAB));
    // interp 100ns, backend 25ns => 4x faster => WIN.
    let v = classify(
        Backend::Jit,
        (&interp, Some(timing(100.0))),
        (&other, Some(timing(25.0))),
    );
    assert!(v.is_win(), "expected a speed win, got {v:?}");
    assert_eq!(v.status(), "WIN");
    assert_eq!(v.guarantee_tag(), "Empirical");
    if let Verdict::SpeedWin { ratio_x1000 } = v {
        assert_eq!(ratio_x1000, 4000, "4.0x => 4000 per-mille");
    } else {
        panic!("not a SpeedWin");
    }
}

#[test]
fn equal_values_and_slower_spawn_bound_backend_is_a_speed_loss_with_reason() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::value_outcome(byte(0xAB));
    // interp 100ns, direct-llvm 100000ns => far slower => LOSS, with the spawn-bound reason.
    let v = classify(
        Backend::DirectLlvm,
        (&interp, Some(timing(100.0))),
        (&other, Some(timing(100_000.0))),
    );
    assert!(v.is_loss(), "expected a loss, got {v:?}");
    match v {
        Verdict::SpeedLoss { reason, .. } => {
            assert!(
                reason.contains("process-spawn-bound"),
                "the spawn-bound reason must be surfaced honestly: {reason}"
            );
        }
        _ => panic!("expected SpeedLoss"),
    }
}

#[test]
fn provenance_differences_are_not_correctness_losses() {
    use crate::backend::observable_eq;
    use mycelium_core::ContentHash;
    // The compiled backends read a value back and stamp `Provenance::Root`; the interpreter
    // records a `Derived` chain. Same repr+payload+guarantee ⇒ observationally equal ⇒ NOT a
    // correctness loss (the false positive this guards against).
    let payload = Payload::Bits((0..8).map(|i| (0xAB_u8 >> i) & 1 == 1).collect());
    let root = CoreValue::Repr(
        Value::new(
            Repr::Binary { width: 8 },
            payload.clone(),
            Meta::exact(Provenance::Root),
        )
        .unwrap(),
    );
    let derived = CoreValue::Repr(
        Value::new(
            Repr::Binary { width: 8 },
            payload,
            Meta::exact(Provenance::Derived {
                op: ContentHash::parse("blake3:abc123").unwrap(),
                inputs: vec![ContentHash::parse("blake3:def456").unwrap()],
            }),
        )
        .unwrap(),
    );
    assert!(
        observable_eq(&root, &derived),
        "values differing only in provenance must be observationally equal (not a loss)"
    );
    // And the classifier must treat them as equal — a speed verdict, never a correctness loss.
    let v = classify(
        Backend::DirectLlvm,
        (&Outcome::value_outcome(derived), Some(timing(100.0))),
        (&Outcome::value_outcome(root), Some(timing(100.0))),
    );
    assert!(
        !matches!(v, Verdict::CorrectnessLoss { .. }),
        "a provenance-only difference must NOT be a correctness loss, got {v:?}"
    );
}

#[test]
fn diverging_values_is_a_correctness_loss_even_if_faster() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::value_outcome(byte(0xFF)); // wrong answer
                                                    // Backend is 10x faster, but the answer diverges — still a LOSS (correctness).
    let v = classify(
        Backend::Jit,
        (&interp, Some(timing(100.0))),
        (&other, Some(timing(10.0))),
    );
    assert!(v.is_loss());
    assert!(!v.is_win(), "a wrong-but-fast answer is never a win");
    assert!(matches!(v, Verdict::CorrectnessLoss { .. }));
    assert_eq!(v.status(), "LOSS (correctness)");
}

#[test]
fn unlowerable_node_is_a_capability_loss() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::Unlowerable("unsupported node for the AOT subset: Fix".into());
    let v = classify(
        Backend::DirectLlvm,
        (&interp, Some(timing(100.0))),
        (&other, None),
    );
    assert!(v.is_loss());
    assert!(matches!(v, Verdict::CapabilityLoss { .. }));
    assert_eq!(v.guarantee_tag(), "Declared");
    if let Verdict::CapabilityLoss { reason } = v {
        assert!(
            reason.contains("Fix"),
            "the unlowerable reason must be kept: {reason}"
        );
    }
}

#[test]
fn toolchain_absent_is_a_skip_not_a_loss() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::Skipped("native toolchain absent (clang)".into());
    let v = classify(
        Backend::DirectLlvm,
        (&interp, Some(timing(100.0))),
        (&other, None),
    );
    assert!(!v.is_loss(), "a skip is NOT a loss");
    assert!(!v.is_win());
    assert!(matches!(v, Verdict::Skipped { .. }));
    assert_eq!(v.status(), "skipped");
}

#[test]
fn runtime_error_is_recorded_not_a_loss_category() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::Error("trit arithmetic overflowed fixed width".into());
    let v = classify(
        Backend::DirectLlvm,
        (&interp, Some(timing(100.0))),
        (&other, None),
    );
    assert!(matches!(v, Verdict::RuntimeError { .. }));
    assert!(!v.is_loss());
}

#[test]
fn neutral_band_classifies_near_parity_as_neutral() {
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::value_outcome(byte(0xAB));
    // 100ns vs 105ns => within +-10% => Neutral.
    let v = classify(
        Backend::AotEnv,
        (&interp, Some(timing(100.0))),
        (&other, Some(timing(105.0))),
    );
    assert!(matches!(v, Verdict::SpeedNeutral { .. }), "got {v:?}");
    assert!(!v.is_loss() && !v.is_win());
}

#[test]
fn baseline_failure_is_flagged_loudly() {
    let interp = Outcome::Error("interpreter blew up".into());
    let other = Outcome::value_outcome(byte(0xAB));
    let v = classify(
        Backend::AotEnv,
        (&interp, None),
        (&other, Some(timing(10.0))),
    );
    assert!(matches!(v, Verdict::BaselineFailed { .. }));
}

#[test]
fn missing_timings_with_equal_values_is_neutral_not_a_false_win() {
    // If we couldn't time one side (e.g. a one-shot run), equal values => Neutral, never a
    // fabricated speed verdict.
    let interp = Outcome::value_outcome(byte(0xAB));
    let other = Outcome::value_outcome(byte(0xAB));
    let v = classify(Backend::AotEnv, (&interp, None), (&other, None));
    assert!(matches!(v, Verdict::SpeedNeutral { ratio_x1000: 1000 }));
}

// ─────────────────────────────── Regression gates (M-859) ─────────────────────────────────────

fn baseline() -> RegressionBaseline {
    RegressionBaseline {
        host_tag: "x86_64-linux, 4 hw threads".to_string(),
        captured: "2026-06-30 / claude/leaf/E25-M859-bench-scaling".to_string(),
        trial_iters: 20_000,
        entries: vec![BaselineEntry {
            case_id: "bit-xor-not".to_string(),
            backend: "aot-env".to_string(),
            ns_per_call: 100.0,
        }],
    }
}

#[test]
fn faster_than_baseline_beyond_band_is_an_improvement() {
    let b = baseline();
    // baseline 100ns, this run 50ns => 2x faster => Improvement.
    let v = regression_classify(&b, &b.host_tag, "bit-xor-not", Backend::AotEnv, Some(50.0));
    assert!(
        v.status() == "WIN (vs baseline)" || matches!(v, RegressionOutcome::Improvement { .. })
    );
    if let RegressionOutcome::Improvement { ratio_x1000 } = v {
        assert_eq!(ratio_x1000, 2000, "2.0x => 2000 per-mille");
    } else {
        panic!("expected Improvement, got {v:?}");
    }
}

#[test]
fn slower_than_baseline_beyond_band_is_a_regression() {
    let b = baseline();
    // baseline 100ns, this run 200ns => 2x slower => Regression.
    let v = regression_classify(&b, &b.host_tag, "bit-xor-not", Backend::AotEnv, Some(200.0));
    assert!(v.is_regression(), "expected a regression, got {v:?}");
    assert_eq!(v.status(), "REGRESSION");
}

#[test]
fn within_band_is_a_hold() {
    let b = baseline();
    // baseline 100ns, this run 110ns => within +-20% => Hold.
    let v = regression_classify(&b, &b.host_tag, "bit-xor-not", Backend::AotEnv, Some(110.0));
    assert!(matches!(v, RegressionOutcome::Hold { .. }), "got {v:?}");
    assert!(!v.is_regression());
}

#[test]
fn missing_this_run_timing_is_not_timed_never_a_fabricated_regression() {
    let b = baseline();
    let v = regression_classify(&b, &b.host_tag, "bit-xor-not", Backend::AotEnv, None);
    assert!(matches!(v, RegressionOutcome::NotTimed));
    assert!(
        !v.is_regression(),
        "a not-timed pair must never count as a regression"
    );
}

#[test]
fn unknown_case_backend_pair_is_no_baseline_never_a_fabricated_hold() {
    let b = baseline();
    let v = regression_classify(&b, &b.host_tag, "bit-not", Backend::AotEnv, Some(50.0));
    assert!(matches!(v, RegressionOutcome::NoBaseline));
}

#[test]
fn mismatched_host_tag_is_refused_never_silently_compared() {
    let b = baseline();
    let v = regression_classify(
        &b,
        "aarch64-macos, 8 hw threads",
        "bit-xor-not",
        Backend::AotEnv,
        Some(50.0),
    );
    match v {
        RegressionOutcome::HostMismatch {
            baseline_host,
            this_host,
        } => {
            assert_eq!(baseline_host, b.host_tag);
            assert_eq!(this_host, "aarch64-macos, 8 hw threads");
        }
        other => panic!("expected HostMismatch, got {other:?}"),
    }
}

#[test]
fn baseline_json_roundtrips() {
    let b = baseline();
    let json = b.to_json().expect("serializes");
    let back = RegressionBaseline::from_json(&json).expect("deserializes");
    assert_eq!(b, back, "baseline JSON must round-trip exactly");
}

#[test]
fn malformed_baseline_json_is_an_explicit_error_not_a_silent_empty_baseline() {
    let result = RegressionBaseline::from_json("{ not valid json");
    assert!(
        result.is_err(),
        "a malformed baseline must be an explicit Err, never silently treated as empty"
    );
}

#[test]
fn nonpositive_timings_are_not_timed_never_a_divide_by_zero_fabrication() {
    let b = baseline();
    let v = regression_classify(&b, &b.host_tag, "bit-xor-not", Backend::AotEnv, Some(0.0));
    assert!(matches!(v, RegressionOutcome::NotTimed));
}
