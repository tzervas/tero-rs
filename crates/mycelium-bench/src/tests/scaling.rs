use crate::corpus::{corpus, Case};
use crate::scaling::*;
use crate::Backend;

fn bit_case() -> Case {
    corpus()
        .into_iter()
        .find(|c| c.id == "bit-xor-not")
        .expect("bit-xor-not exists in the corpus")
}

#[test]
fn interp_scaling_curve_has_one_sample_per_worker_count() {
    let case = bit_case();
    // Small batch/repeats — this test only needs to confirm shape + honesty, not real speedup
    // (real timings are Empirical and never asserted as specific numbers, VR-5).
    let point = measure_case_scaling(&case, Backend::Interp, 3, 4, 1);
    match &point.outcome {
        ScalingOutcome::Measured(samples) => {
            let workers: Vec<usize> = samples.iter().map(|s| s.workers).collect();
            assert_eq!(
                workers,
                vec![1, 2, 3],
                "one sample per worker count 1..=max_workers"
            );
            for s in samples {
                assert!(s.batch_ns.is_finite() && s.batch_ns >= 0.0);
                assert!(s.ns_per_job.is_finite() && s.ns_per_job >= 0.0);
            }
        }
        other => panic!("interp on a bit case must be measurable, got {other:?}"),
    }
}

#[test]
fn speedups_include_the_ideal_linear_reference_and_start_at_one() {
    let case = bit_case();
    let point = measure_case_scaling(&case, Backend::Interp, 2, 4, 1);
    let sp = point
        .speedups()
        .expect("interp bit case must have speedups");
    let (w1, s1, ideal1) = sp[0];
    assert_eq!(w1, 1);
    assert!(
        (s1 - 1.0).abs() < 1e-9,
        "1-worker speedup is exactly 1.0x by definition"
    );
    assert!((ideal1 - 1.0).abs() < 1e-9);
    // Every ideal-linear reference must equal its own worker count.
    for (w, _, ideal) in &sp {
        #[allow(clippy::cast_precision_loss)]
        let expect_ideal = *w as f64;
        assert!((*ideal - expect_ideal).abs() < 1e-9);
    }
}

#[test]
fn amdahl_fraction_is_clamped_into_zero_one() {
    let case = bit_case();
    let point = measure_case_scaling(&case, Backend::Interp, 3, 4, 1);
    if let Some(s) = point.amdahl_serial_fraction() {
        assert!(
            (0.0..=1.0).contains(&s),
            "serial fraction must be in [0,1], got {s}"
        );
    }
    // No assertion that a fraction MUST be present (it legitimately can be None on a degenerate
    // fit) — this only pins the honesty invariant that when present, it is a valid fraction.
}

#[test]
fn recursion_case_is_unmeasurable_or_skipped_for_compiled_backends_never_silently_empty() {
    let case = corpus()
        .into_iter()
        .find(|c| c.id == "rec-self")
        .expect("rec-self exists");
    for backend in [Backend::Jit, Backend::DirectLlvm] {
        let point = measure_case_scaling(&case, backend, 2, 4, 1);
        assert!(
            matches!(
                point.outcome,
                ScalingOutcome::Unmeasurable(_) | ScalingOutcome::Skipped(_)
            ),
            "recursion must never silently produce a Measured curve on a compiled backend: {:?}",
            point.outcome
        );
        // Never-silent (G2): the reason string must be non-empty.
        match &point.outcome {
            ScalingOutcome::Unmeasurable(r) | ScalingOutcome::Skipped(r) => {
                assert!(!r.is_empty(), "the reason must be recorded, not blank");
            }
            ScalingOutcome::Measured(_) => unreachable!(),
        }
    }
}

#[test]
fn run_scaling_covers_every_case_backend_pair() {
    let cases: Vec<_> = corpus()
        .into_iter()
        .filter(|c| matches!(c.id, "bit-xor-not" | "rec-self"))
        .collect();
    let run = run_scaling(&cases, 2, 4, 1);
    assert_eq!(run.worker_counts, vec![1, 2]);
    // 2 cases x 5 backends = 10 points, never-silently short.
    assert_eq!(run.points.len(), 2 * Backend::all().len());
}

#[test]
fn process_spawn_bound_backends_are_flagged_consistently_with_the_single_core_harness() {
    // The scaling module reuses the same `is_process_spawn_bound` classification the single-core
    // report uses — a spot check that the flag is queryable from here without duplicating logic.
    assert!(Backend::DirectLlvm.is_process_spawn_bound());
    assert!(Backend::MlirDialect.is_process_spawn_bound());
    assert!(!Backend::Jit.is_process_spawn_bound());
    assert!(!Backend::Interp.is_process_spawn_bound());
    assert!(!Backend::AotEnv.is_process_spawn_bound());
}
