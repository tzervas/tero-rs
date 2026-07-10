//! End-to-end harness test (offline, deterministic): run a small corpus across the backends, ingest
//! the committed SYNTHETIC LLM-harness sample, emit both report projections, and assert the report's
//! *structure* and honesty discipline — never a specific timing number (those are Empirical, VR-5).
//!
//! This pins the three deliverables the harness must always satisfy:
//! 1. it measures interp vs AOT vs the compiled paths over real programs and classifies each,
//! 2. it surfaces an honest LOSS (a capability loss for the compiled paths on a recursion case),
//! 3. it ingests the LLM-harness report and labels a synthetic sample as synthetic.

use mycelium_bench::backend::{Backend, Engines};
use mycelium_bench::corpus::corpus;
use mycelium_bench::llm::{parse_any_llm_json, LlmReport};
use mycelium_bench::measure::run_corpus;
use mycelium_bench::report::{neutral_band, Honesty, LlmSection, Report};
use mycelium_bench::verdict::Verdict;

/// A faithful synthetic sample matching the harness schema (so the test is fully self-contained).
const SYNTHETIC_LLM: &str = r#"{
  "harness": "mycelium-llm-validation",
  "version": "0.1.0",
  "run_id": "20260617T182214Z",
  "mode": "mock",
  "honesty_posture": {
    "never_silent": true,
    "guarantee_lattice": ["Exact", "Proven", "Empirical", "Declared"],
    "model_allowed_tags": ["Declared", "Empirical"],
    "vr5_rule": "Model-derived claims are Empirical or Declared — NEVER Proven or Exact."
  },
  "summary": {
    "overall": "MOCK", "total": 2, "pass": 0, "mock_pass": 2,
    "skip": 0, "fail": 0, "exit_code": 0, "mode": "mock", "model": null
  },
  "results": [
    {"id":"V-01-determinism","status":"mock-PASS","guarantee_tag":"Declared",
     "message":"[MOCK] determinism simulated","detail":{"mode":"mock","matched":true}},
    {"id":"V-04-latency-tokens","status":"mock-PASS","guarantee_tag":"Declared",
     "message":"[MOCK] latency simulated",
     "detail":{"mode":"mock","wall_seconds":0.0,
               "token_counts":{"prompt":12,"generated":7,"note":"Declared"}}}
  ]
}"#;

fn build_report() -> Report {
    let eng = Engines::default();
    // A representative slice: a bit case every backend can run, and a recursion case the compiled
    // paths cannot (so a capability loss is guaranteed to be surfaced).
    let cases: Vec<_> = corpus()
        .into_iter()
        .filter(|c| matches!(c.id, "bit-xor-not" | "rec-self"))
        .collect();
    assert_eq!(
        cases.len(),
        2,
        "the two pinned cases must exist in the corpus"
    );
    let run = run_corpus(&cases, &eng);

    let rep = LlmReport::from_json(SYNTHETIC_LLM).expect("synthetic sample parses");
    let synthetic = rep.is_synthetic();
    let llm = Some(LlmSection::from_report(
        &rep,
        "tools/llm-harness/reports/<synthetic-sample>.json".into(),
        synthetic,
    ));

    Report {
        tool: "mycelium-bench",
        profile: "test",
        mlir_dialect_feature: cfg!(feature = "mlir-dialect"),
        host_note: "integration-test".into(),
        honesty: Honesty::default(),
        neutral_band: neutral_band(),
        run,
        llm,
        scaling: None,
        regression: None,
    }
}

#[test]
fn the_harness_measures_interp_aot_and_compiled_paths_and_classifies_each() {
    let report = build_report();

    // The bit case: the interpreter produced a timed baseline, and the AOT env-machine agreed with it
    // (not a correctness loss / baseline failure). The compiled paths either won/lost on speed,
    // recorded a value, or skipped (no toolchain) — but never silently nothing.
    let bit = report
        .run
        .cases
        .iter()
        .find(|c| c.id == "bit-xor-not")
        .expect("bit case present");
    assert!(
        bit.baseline_ns.is_some(),
        "trusted interpreter baseline must be timed"
    );
    let aot = bit
        .backends
        .iter()
        .find(|b| b.backend == Backend::AotEnv)
        .unwrap();
    assert!(
        !matches!(
            aot.verdict,
            Verdict::CorrectnessLoss { .. } | Verdict::BaselineFailed { .. }
        ),
        "AOT env-machine must agree with the interpreter on the bit case: {:?}",
        aot.verdict
    );
    // Every backend row has a concrete verdict (never absent).
    assert_eq!(bit.backends.len(), 4, "four non-baseline backends measured");
}

#[test]
fn the_harness_surfaces_an_honest_loss() {
    let report = build_report();
    let roll = report.loss_rollup();
    let tallies = report.tallies();

    // The recursion case MUST account for the compiled paths as a capability loss (or a skip when the
    // native toolchain is absent) — never a silent success. At least one of the two must be true.
    let recursion_accounted = !roll.capability.is_empty() || tallies.skips > 0;
    assert!(
        recursion_accounted,
        "the recursion case must surface a capability loss or a toolchain skip for the compiled paths"
    );

    // The markdown must contain the explicit "where we're losing" section.
    let md = report.to_markdown();
    assert!(md.contains("## Where we're losing"));

    // If a capability loss was recorded (toolchain present), it must name the unlowerable reason and
    // appear in the section — the honest M-602/E1-style finding, surfaced not buried.
    if !roll.capability.is_empty() {
        assert!(md.contains("Capability losses"));
        let (_, backend, reason) = &roll.capability[0];
        assert!(
            matches!(*backend, "jit" | "direct-llvm" | "mlir-dialect"),
            "a capability loss must be attributed to a compiled backend, got {backend}"
        );
        assert!(
            !reason.is_empty(),
            "the capability-loss reason must be recorded (G2)"
        );
    }
}

#[test]
fn the_harness_ingests_the_llm_report_and_labels_synthetic() {
    let report = build_report();
    let sec = report.llm.as_ref().expect("LLM section present");
    assert!(
        sec.is_synthetic,
        "the mock sample must be flagged synthetic"
    );
    assert_eq!(sec.validations.len(), 2);

    let md = report.to_markdown();
    assert!(md.contains("## LLM-harness leverage"));
    assert!(
        md.contains("SYNTHETIC sample"),
        "a synthetic sample must be labeled synthetic in the report (VR-5 / the harness's V-03 rule)"
    );
    // The latency + token columns must render the ingested values.
    assert!(md.contains("V-04-latency-tokens"));
}

#[test]
fn both_report_projections_are_emitted_and_valid() {
    let report = build_report();

    // JSON is valid + deterministic.
    let json = report.to_json().expect("json serializes");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert_eq!(v["tool"], "mycelium-bench");
    assert_eq!(v["profile"], "test");
    assert!(v["run"]["cases"].is_array());
    assert!(v["llm"]["is_synthetic"].as_bool().unwrap());
    assert_eq!(
        json,
        report.to_json().unwrap(),
        "JSON emission is deterministic"
    );

    // Markdown carries every required section + the honesty posture.
    let md = report.to_markdown();
    for section in [
        "# Mycelium honest benchmark report",
        "## WIN / LOSS / regression table",
        "## Per-case timings",
        "## Where we're losing",
        "## LLM-harness leverage",
    ] {
        assert!(md.contains(section), "missing report section: {section}");
    }
    assert!(md.contains("VR-5"), "the no-target rule must be stated");
    assert!(
        md.contains("process-spawn-bound"),
        "the spawn caveat must be stated"
    );
}

#[test]
fn the_harness_ingests_grok_report_and_labels_synthetic_correctly() {
    // Load the committed Grok SYNTHETIC fixture. The path is crate-local (in tests/), so it is
    // always reachable from the integration-test's working directory.
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/SYNTHETIC-SAMPLE-grok-4.3-self-test.json"
    );
    let text = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|e| panic!("Grok fixture must be readable: {e}"));

    // The schema-dispatching entry-point must recognise the Grok schema and not error.
    let parsed = parse_any_llm_json(&text, fixture_path.into())
        .expect("parse_any_llm_json must ingest the Grok fixture without error");

    // Synthetic flag: the fixture has `honesty_posture.synthetic = true` and
    // `metadata.mode = "self-test"` — both mark it as a synthetic run.
    assert!(
        parsed.is_synthetic,
        "the Grok self-test fixture must be flagged synthetic (VR-5)"
    );
    assert!(
        parsed.provenance.contains("SYNTHETIC"),
        "provenance must surface the synthetic label: {}",
        parsed.provenance
    );

    // Per-outcome rows: the committed fixture has 3 outcomes (g01, g02, g04).
    assert_eq!(
        parsed.validations.len(),
        3,
        "three Grok outcomes → three validation rows"
    );

    // Guarantee tags must be preserved verbatim from the Grok outcomes — never upgraded (VR-5).
    for row in &parsed.validations {
        assert_eq!(
            row.guarantee_tag.as_deref(),
            Some("Declared"),
            "guarantee_tag must be preserved from the Grok outcome (VR-5): row id={}",
            row.id
        );
    }

    // Build a full report with the Grok-ingested LLM section and verify the markdown labels it
    // synthetic (the same path the bench harness uses for the original harness).
    let eng = Engines::default();
    let cases: Vec<_> = corpus()
        .into_iter()
        .filter(|c| c.id == "bit-xor-not")
        .collect();
    let run = run_corpus(&cases, &eng);
    let llm_section = LlmSection::from_parsed(parsed);
    assert!(
        llm_section.is_synthetic,
        "LlmSection built from Grok parsed must carry the synthetic flag"
    );

    let report = Report {
        tool: "mycelium-bench-grok-test",
        profile: "test",
        mlir_dialect_feature: cfg!(feature = "mlir-dialect"),
        host_note: "grok-bridge-test".into(),
        honesty: Honesty::default(),
        neutral_band: neutral_band(),
        run,
        llm: Some(llm_section),
        scaling: None,
        regression: None,
    };

    let md = report.to_markdown();
    assert!(
        md.contains("SYNTHETIC sample"),
        "the unified markdown report must label a Grok synthetic run as SYNTHETIC"
    );
    // Spot-check that at least one outcome row appears in the markdown.
    assert!(
        md.contains("g01"),
        "the Grok outcome g01 must appear in the report table"
    );

    // JSON projection: the llm section must serialize and carry is_synthetic = true.
    let json = report.to_json().expect("json serializes");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert!(
        v["llm"]["is_synthetic"].as_bool().unwrap_or(false),
        "JSON llm.is_synthetic must be true for the Grok synthetic fixture"
    );
}
