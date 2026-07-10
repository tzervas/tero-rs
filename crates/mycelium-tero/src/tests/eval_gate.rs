//! White-box tests for the Layer-2 **eval gate** (M-1018 / DN-87 §6.1). The binding assertion is on
//! **honesty, not on a win**: the verdict must be *computed, recorded, and honest*, a **Closed gate
//! is a first-class valid outcome (no win required)**, the served `layer2_enabled` flag defaults
//! **off**, latency is measured with explicit denominators, and provenance fidelity is preserved.
//!
//! The recorded *numbers* the DoD asks for come from the `tero-eval` binary over the full committed
//! index (`eval/VERDICT.md` / `eval/verdict.json`); these tests exercise the machinery hermetically
//! (fast) plus a bounded, skip-graceful smoke over the committed index.

use std::path::{Path, PathBuf};

use crate::eval::verdict::{decide_gate, GateEvidence, GateVerdict, REGRESSION_BAND};
use crate::{run_eval, EvalQuestion, EvalSuite, TeroIndexReport};

use super::fixture::corpus_report;

/// Build a small, honest fixture question set: each question is a fixture row's own title, its gold
/// anchor the row's anchor (guaranteed resolvable).
fn fixture_questions(report: &TeroIndexReport) -> Vec<EvalQuestion> {
    report
        .items
        .iter()
        .filter(|it| !it.title.is_empty())
        .take(4)
        .enumerate()
        .map(|(i, it)| EvalQuestion {
            id: format!("q{i}"),
            question: it.title.clone(),
            gold_anchor: it.anchor.clone(),
            seed: 4200 + i as u64,
        })
        .collect()
}

#[test]
fn run_eval_computes_and_records_an_honest_verdict_closed_is_valid() {
    let (_root, report) = corpus_report("l2-eval");
    let questions = fixture_questions(&report);
    let out = run_eval(&report, &questions, 5, 2);

    // The verdict is computed + recorded. Closed OR Open are both valid outcomes — we do NOT require a
    // Layer-2 win (a Closed gate is the honest, expected result; VR-5).
    let evidence = out.verdict.evidence();
    assert_eq!(evidence.questions, questions.len(), "denominator recorded");
    assert_eq!(evidence.k, 5);
    assert_eq!(
        evidence.band, REGRESSION_BAND,
        "the band is reified in evidence"
    );
    if let GateVerdict::Closed { reason, .. } = &out.verdict {
        assert!(
            !reason.is_empty(),
            "a Closed verdict states why (never-silent)"
        );
    }

    // Denominators everywhere (never a bare rate).
    assert_eq!(out.layer1.total, questions.len());
    assert_eq!(out.layer2.total, questions.len());
    assert_eq!(out.questions_total, questions.len());
    assert_eq!(out.codebook_len + out.refused_records, report.items.len());

    // Provenance fidelity is preserved: every returned Layer-2 citation resolves to a real row.
    assert!(
        (out.layer2.provenance_fidelity() - 1.0).abs() < f64::EPSILON,
        "Layer-2 provenance must be 1.0 (returned anchors resolve): {}",
        out.layer2.provenance_fidelity()
    );

    // Latency is measured, with a trial count (Empirical, single-machine) + a host tag.
    assert_eq!(out.layer1.trial_iters, 2);
    assert!(out.layer1.ns_per_query >= 0.0 && out.layer1.ns_per_query.is_finite());
    assert!(out.layer2.ns_per_query >= 0.0 && out.layer2.ns_per_query.is_finite());
    assert!(!out.host_tag.is_empty());

    // The report round-trips through its committed JSON shape.
    let json = serde_json::to_string(&out).expect("EvalReport serializes");
    assert!(json.contains("\"verdict\""));
}

fn evidence(l1c1: f64, l2c1: f64, prov: f64, l1ns: f64, l2ns: f64) -> GateEvidence {
    GateEvidence {
        questions: 10,
        k: 5,
        l1_correct_at_1: l1c1,
        l2_correct_at_1: l2c1,
        l1_correct_at_k: l1c1,
        l2_correct_at_k: l2c1,
        l2_provenance: prov,
        l1_ns_per_query: l1ns,
        l2_ns_per_query: l2ns,
        band: REGRESSION_BAND,
    }
}

#[test]
fn gate_is_closed_by_default_when_layer2_does_not_beat_baseline() {
    // Equal correctness ⇒ Layer-2 did not beat Layer-1 beyond the band ⇒ Closed.
    let v = decide_gate(evidence(0.5, 0.5, 1.0, 1000.0, 1000.0));
    assert!(!v.is_open(), "equal correctness must not open the gate");
    match v {
        GateVerdict::Closed { reason, .. } => assert!(reason.contains("correctness@1")),
        GateVerdict::Open { .. } => panic!("must be Closed"),
    }
}

#[test]
fn gate_can_open_on_a_real_win_so_the_test_is_not_vacuous() {
    // A genuine Layer-2 win (correctness beats beyond the band, provenance 1.0, latency within band)
    // ⇒ Open. This proves the gate is capable of opening — it is not wired shut.
    let v = decide_gate(evidence(0.20, 0.80, 1.0, 1000.0, 1000.0));
    assert!(v.is_open(), "a real win must open the gate");
}

#[test]
fn gate_stays_closed_if_provenance_or_latency_regress_despite_a_correctness_win() {
    // Correctness win but provenance < 1.0 ⇒ Closed (a Layer-2 answer with an unresolvable citation
    // is disqualifying).
    let v_prov = decide_gate(evidence(0.20, 0.80, 0.90, 1000.0, 1000.0));
    assert!(!v_prov.is_open());
    // Correctness win but latency 2x the baseline (beyond the band) ⇒ Closed.
    let v_lat = decide_gate(evidence(0.20, 0.80, 1.0, 1000.0, 2000.0));
    assert!(!v_lat.is_open());
}

#[test]
fn layer2_front_flag_defaults_off_serving_layer1() {
    // The served gate defaults closed: identify reports `layer2_enabled=false` (the M-1018 gate stays
    // shut until the eval gate opens — DN-87 §6.1). The system serves Layer-1 answers.
    let v = crate::front::core::identify_value(false);
    assert_eq!(v["layer2_enabled"], serde_json::Value::Bool(false));
}

/// The repo root, two levels above this crate's manifest dir (mirrors `query_latency.rs`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

#[test]
fn committed_index_smoke_is_skip_graceful() {
    // Skip-graceful (mirrors query_latency.rs): a checkout without the committed index/question set
    // yields no assertion, not a failure.
    let index_path = repo_root().join("docs/tero-index/index.json");
    let questions_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval/questions.json");
    if !index_path.exists() || !questions_path.exists() {
        return;
    }
    let full = crate::load_report(&index_path).expect("load committed index");
    let suite = EvalSuite::from_json(&std::fs::read_to_string(&questions_path).unwrap())
        .expect("parse committed questions");

    // Bound the codebook for test speed (the full-corpus run is the binary's job) while keeping every
    // gold anchor, so the committed data path is genuinely exercised.
    const BOUND: usize = 400;
    let golds: std::collections::BTreeSet<&str> = suite
        .questions
        .iter()
        .map(|q| q.gold_anchor.as_str())
        .collect();
    let mut kept: Vec<_> = full
        .items
        .iter()
        .filter(|it| golds.contains(it.anchor.as_str()))
        .cloned()
        .collect();
    for it in &full.items {
        if kept.len() >= BOUND {
            break;
        }
        if !golds.contains(it.anchor.as_str()) {
            kept.push(it.clone());
        }
    }
    let mut report = TeroIndexReport {
        items: kept,
        flagged: Vec::new(),
    };
    report.sort();

    let out = run_eval(&report, &suite.questions, 5, 1);
    // The machinery produces a recorded, honest verdict over committed data (Closed or Open both OK).
    assert_eq!(out.questions_total, suite.questions.len());
    assert_eq!(out.codebook_len + out.refused_records, report.items.len());
    assert!((out.layer2.provenance_fidelity() - 1.0).abs() < f64::EPSILON);
}
