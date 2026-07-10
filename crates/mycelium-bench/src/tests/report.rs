use crate::backend::{Backend, Engines};
use crate::corpus::corpus;
use crate::measure::run_corpus;
use crate::report::*;
use crate::scaling::run_scaling;
use crate::verdict::{BaselineEntry, RegressionBaseline, NEUTRAL_BAND};

/// Build a small real report by measuring two bit cases (offline, deterministic enough for the
/// emission tests — we assert structure, never specific timings).
fn small_report() -> Report {
    let eng = Engines::default();
    let cases: Vec<_> = corpus()
        .into_iter()
        .filter(|c| matches!(c.id, "bit-xor-not" | "rec-self"))
        .collect();
    let run = run_corpus(&cases, &eng);
    Report {
        tool: "mycelium-bench-test",
        profile: "test",
        mlir_dialect_feature: cfg!(feature = "mlir-dialect"),
        host_note: "unit-test".into(),
        honesty: Honesty::default(),
        neutral_band: NEUTRAL_BAND,
        run,
        llm: None,
        scaling: None,
        regression: None,
    }
}

#[test]
fn json_projection_is_valid_and_roundtrips_as_value() {
    let r = small_report();
    let json = r.to_json().expect("serializes");
    // It must be valid JSON.
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert_eq!(v["tool"], "mycelium-bench-test");
    assert!(v["run"]["cases"].is_array());
    // Deterministic: serializing twice gives identical bytes.
    assert_eq!(json, r.to_json().unwrap());
}

#[test]
fn markdown_has_all_required_sections() {
    let r = small_report();
    let md = r.to_markdown();
    assert!(md.contains("# Mycelium honest benchmark report"));
    assert!(md.contains("## WIN / LOSS / regression table"));
    assert!(md.contains("## Per-case timings"));
    assert!(md.contains("## Where we're losing"));
    assert!(md.contains("## Multicore scaling"));
    assert!(md.contains("## Regression gate vs committed baseline"));
    assert!(md.contains("## LLM-harness leverage"));
    // The honesty posture + VR-5 must be stated.
    assert!(md.contains("VR-5"));
    assert!(md.contains("Empirical"));
    // The process-spawn caveat must be present (it is the headline honest finding).
    assert!(md.contains("process-spawn-bound"));
}

#[test]
fn the_recursion_case_surfaces_a_capability_loss_in_the_losses_section() {
    // rec-self cannot be lowered by jit/direct-llvm — the "where we're losing" section MUST name
    // it as a capability loss (unless the run skipped for toolchain absence, also acceptable).
    let r = small_report();
    let roll = r.loss_rollup();
    let md = r.to_markdown();
    // Either a capability loss is recorded, or the compiled paths were skipped (no toolchain).
    let tallies = r.tallies();
    let compiled_accounted = !roll.capability.is_empty() || tallies.skips > 0;
    assert!(
        compiled_accounted,
        "the recursion case must produce a capability loss OR a skip for the compiled paths"
    );
    if !roll.capability.is_empty() {
        assert!(
            md.contains("Capability losses"),
            "a recorded capability loss must appear in the losses section"
        );
    }
}

#[test]
fn md_escape_protects_table_rows() {
    assert_eq!(md_escape("a | b\nc"), "a \\| b c");
}

#[test]
fn llm_section_labels_synthetic_when_present() {
    use crate::llm::LlmReport;
    let sample = r#"{
      "harness":"mycelium-llm-validation","version":"0.1.0","run_id":"X","mode":"mock",
      "honesty_posture":{"never_silent":true,"guarantee_lattice":["Exact"],
        "model_allowed_tags":["Declared"],"vr5_rule":"r"},
      "summary":{"overall":"MOCK","total":1,"pass":0,"mock_pass":1,"skip":0,"fail":0,
        "exit_code":0,"mode":"mock","model":null},
      "results":[{"id":"V-01","status":"mock-PASS","guarantee_tag":"Declared",
        "message":"m","detail":{"mode":"mock"}}]
    }"#;
    let rep = LlmReport::from_json(sample).unwrap();
    let mut r = small_report();
    r.llm = Some(LlmSection::from_report(
        &rep,
        "sample.json".into(),
        rep.is_synthetic(),
    ));
    let md = r.to_markdown();
    assert!(
        md.contains("SYNTHETIC sample"),
        "synthetic must be labeled in the LLM section"
    );
    assert!(md.contains("V-01"));
}

// ─────────────────────────────── Scaling section (M-859) ──────────────────────────────────────

#[test]
fn scaling_section_absent_is_reported_empty_not_synthesized() {
    let r = small_report();
    let md = r.to_markdown();
    assert!(md.contains("No scaling run attached to this report"));
}

#[test]
fn scaling_section_renders_when_attached() {
    let mut r = small_report();
    let cases: Vec<_> = corpus()
        .into_iter()
        .filter(|c| c.id == "bit-xor-not")
        .collect();
    let run = run_scaling(&cases, 2, 4, 1);
    // A spawn-bound backend that was actually measured (a value, not a skip/toolchain-absent) must
    // carry the spawn-dominated caveat in the rendered row — asserted on the structured data first
    // (never-silent, precise) so the markdown substring check below is a genuine corroboration, not
    // the only evidence.
    let direct_llvm_measured = run.points.iter().any(|p| {
        p.backend == Backend::DirectLlvm
            && matches!(p.outcome, crate::scaling::ScalingOutcome::Measured(_))
    });
    r.scaling = Some(run);
    let md = r.to_markdown();
    assert!(md.contains("bit-xor-not"));
    assert!(md.contains("Amdahl serial fraction"));
    if direct_llvm_measured {
        assert!(
            md.contains("spawn-dominated"),
            "a measured spawn-bound backend row must carry the spawn-dominated note"
        );
    }
}

// ─────────────────────────────── Regression section (M-859) ───────────────────────────────────

fn baseline_for(host_tag: &str) -> RegressionBaseline {
    RegressionBaseline {
        host_tag: host_tag.to_string(),
        captured: "test-fixture".to_string(),
        trial_iters: 1000,
        entries: vec![BaselineEntry {
            case_id: "bit-xor-not".to_string(),
            backend: "aot-env".to_string(),
            ns_per_call: 100.0,
        }],
    }
}

#[test]
fn regression_section_absent_is_reported_empty_not_synthesized() {
    let r = small_report();
    let md = r.to_markdown();
    assert!(md.contains("No baseline supplied for this report"));
}

#[test]
fn regression_gate_on_matching_host_tag_never_reports_host_mismatch() {
    // Regression test for a real bug caught while dogfooding: `with_regression_gate` used to
    // compare the baseline's bare host tag against `Report::host_note` (which wraps the same
    // info in report-header prose, `"host: ... (provenance only)"`), so every row was a spurious
    // HostMismatch even on the exact host the baseline was captured on. Fixed by threading the
    // canonical bare tag explicitly through `with_regression_gate`'s `this_host_tag` parameter.
    let r = small_report();
    let baseline = baseline_for("exact-match-host-tag");
    let r = r.with_regression_gate("exact-match-host-tag", &baseline);
    let sec = r.regression.as_ref().expect("regression section attached");
    let bit_case_row = sec
        .rows
        .iter()
        .find(|row| row.case_id == "bit-xor-not" && row.backend == "aot-env")
        .expect("bit-xor-not/aot-env row present");
    assert!(
        bit_case_row.outcome.status() != "host-mismatch",
        "a matching bare host tag must never classify as host-mismatch: {:?}",
        bit_case_row.outcome
    );
}

#[test]
fn regression_gate_on_different_host_tag_is_host_mismatch() {
    let r = small_report();
    let baseline = baseline_for("some-other-host");
    let r = r.with_regression_gate("this-run-host", &baseline);
    let sec = r.regression.as_ref().expect("regression section attached");
    assert!(
        sec.rows
            .iter()
            .all(|row| row.outcome.status() == "host-mismatch"),
        "every row must be host-mismatch when the tags genuinely differ"
    );
    let md = r.to_markdown();
    assert!(md.contains("host-mismatch"));
}

#[test]
fn regression_section_renders_and_reports_zero_regressions_on_a_generous_baseline() {
    let r = small_report();
    // A baseline whose ns_per_call is deliberately huge — this run cannot possibly regress
    // against it, so we can assert the section renders "0 regression(s)" deterministically.
    let mut baseline = baseline_for("unit-test-host");
    baseline.entries[0].ns_per_call = 1.0e12;
    let r = r.with_regression_gate("unit-test-host", &baseline);
    let md = r.to_markdown();
    assert!(md.contains("Regression gate vs committed baseline"));
    assert!(md.contains("0 regression(s)"));
}
