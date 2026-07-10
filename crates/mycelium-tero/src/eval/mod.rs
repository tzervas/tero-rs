//! **The Layer-2 eval harness** (M-1018 / DN-87 §6.1): a question set drawn from real agent tasks
//! over *this* corpus, graded on **answer correctness** (`correctness@1` / `@k`, with explicit
//! denominators), **provenance fidelity** (every returned citation resolves to a real Layer-1 row —
//! must be 1.0), and **latency** (ns/query, `Empirical`, single-machine, with trial count + host
//! tag), comparing **Layer 2 (VSA)** against the **Layer-1 baseline** ([`crate::QueryEngine`] text
//! search). The [`verdict::decide_gate`] verdict is **Closed by default** and opens only on a real,
//! measured Layer-2 win that keeps provenance and latency honest.
//!
//! Honesty posture (G2/VR-5), bound throughout: a **Closed gate is a first-class, honest, expected
//! outcome** for a ~5k-row structured corpus. Nothing here tunes thresholds or curates the question
//! set to manufacture a pass — the two systems are compared as-is, and the recorded numbers are
//! whatever the harness measures.

pub mod verdict;

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::model::TeroIndexReport;
use crate::query::{Query, QueryEngine};
use crate::vsa2::Layer2Index;

use verdict::{decide_gate, GateEvidence, GateVerdict, REGRESSION_BAND};

/// One eval question, drawn from a real agent task over this corpus, with a **stable, resolvable gold
/// anchor** and a fixed `seed` (seed discipline: recorded per question for reproducibility). The gold
/// anchor is the Layer-1 row a correct answer must surface top-1/top-k.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalQuestion {
    /// A short question id (e.g. `q-landed-index`).
    pub id: String,
    /// The natural-language question, as an agent would pose it.
    pub question: String,
    /// The gold Layer-1 anchor a correct retrieval must return.
    pub gold_anchor: String,
    /// The fixed per-question seed (seed discipline — recorded, reproducible).
    pub seed: u64,
}

/// The committed question-set file shape (`eval/questions.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalSuite {
    /// The questions.
    pub questions: Vec<EvalQuestion>,
}

impl EvalSuite {
    /// Parse a suite from its committed JSON text. Never-silent: a malformed suite is an explicit
    /// `Err`, never an empty/partial question set.
    ///
    /// # Errors
    /// The `serde_json` parse error, verbatim.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// Per-system aggregate metrics, with **explicit denominators** (never a bare rate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Which system (`"layer1"` / `"layer2"`).
    pub system: String,
    /// Questions whose gold anchor was returned top-1.
    pub correct_at_1: usize,
    /// Questions whose gold anchor was returned within top-k.
    pub correct_at_k: usize,
    /// Total questions graded (the denominator).
    pub total: usize,
    /// Returned citations that resolved to a real Layer-1 row.
    pub provenance_ok: usize,
    /// Total returned citations (the provenance denominator).
    pub provenance_total: usize,
    /// Measured latency (ns/query, Empirical, single-machine).
    pub ns_per_query: f64,
    /// The trial iteration count the latency represents.
    pub trial_iters: u32,
}

impl SystemMetrics {
    /// correctness@1 as a rate in `[0, 1]` (`0` when no questions).
    #[must_use]
    pub fn rate_at_1(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct_at_1 as f64 / self.total as f64
        }
    }

    /// correctness@k as a rate in `[0, 1]`.
    #[must_use]
    pub fn rate_at_k(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct_at_k as f64 / self.total as f64
        }
    }

    /// Provenance fidelity in `[0, 1]`. Vacuously `1.0` when nothing was returned (no citation failed
    /// to resolve) — the honest reading of "no unresolved citation was served".
    #[must_use]
    pub fn provenance_fidelity(&self) -> f64 {
        if self.provenance_total == 0 {
            1.0
        } else {
            self.provenance_ok as f64 / self.provenance_total as f64
        }
    }
}

/// One question's per-system outcome — the auditable row behind the aggregates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerQuestion {
    /// The question id.
    pub id: String,
    /// The gold anchor.
    pub gold_anchor: String,
    /// The per-question seed (recorded).
    pub seed: u64,
    /// Layer-1 top-1 anchor (if any).
    pub l1_top: Option<String>,
    /// Layer-1 gold in top-1.
    pub l1_hit_at_1: bool,
    /// Layer-1 gold in top-k.
    pub l1_hit_at_k: bool,
    /// Layer-2 top-1 anchor (if any).
    pub l2_top: Option<String>,
    /// Layer-2 gold in top-1.
    pub l2_hit_at_1: bool,
    /// Layer-2 gold in top-k.
    pub l2_hit_at_k: bool,
}

/// The full eval report — the machine artifact (`eval/verdict.json`) and the source of the human
/// `eval/VERDICT.md` append.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// The `k` used for correctness@k.
    pub k: usize,
    /// The committed Layer-2 master seed the codebook was built from.
    pub seed_master: u64,
    /// The host tag the latency was measured on.
    pub host_tag: String,
    /// Questions graded.
    pub questions_total: usize,
    /// Encoded records in the Layer-2 codebook.
    pub codebook_len: usize,
    /// Rows refused at Layer-2 encode time (never-silent).
    pub refused_records: usize,
    /// Layer-1 baseline metrics.
    pub layer1: SystemMetrics,
    /// Layer-2 metrics.
    pub layer2: SystemMetrics,
    /// Per-question outcomes.
    pub per_question: Vec<PerQuestion>,
    /// The append-only gate verdict.
    pub verdict: GateVerdict,
}

/// A single-machine host tag for latency provenance: `"<arch>-<os>, <n> hw threads"`. Not portable —
/// a latency baseline is only ever compared against a run tagged the same (VR-5).
#[must_use]
pub fn host_tag() -> String {
    let threads = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
    format!(
        "{}-{}, {threads} hw threads",
        std::env::consts::ARCH,
        std::env::consts::OS
    )
}

/// The Layer-1 baseline's top-k anchors for a text question (ranked), or empty on a refusal.
fn layer1_topk(engine: &QueryEngine, question: &str, k: usize) -> Vec<String> {
    match engine.run(&Query::Text(question.to_owned())) {
        Ok(answer) => answer
            .items()
            .iter()
            .take(k)
            .map(|it| it.anchor.clone())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// The Layer-2 semantic layer's top-k anchors for a text question (ranked), or empty on a refusal.
fn layer2_topk(index: &Layer2Index, question: &str, k: usize) -> Vec<String> {
    match index.rank(question, k) {
        Ok(cands) => cands.into_iter().map(|c| c.anchor).collect(),
        Err(_) => Vec::new(),
    }
}

/// Run the eval harness: grade both systems over `questions`, measure latency across `trials` reps,
/// and decide the gate. Deterministic in its correctness metrics (retrieval is a pure function of the
/// report + seed); the latency figures are `Empirical` (single-machine, `trials`-averaged).
#[must_use]
pub fn run_eval(
    report: &TeroIndexReport,
    questions: &[EvalQuestion],
    k: usize,
    trials: u32,
) -> EvalReport {
    let engine = QueryEngine::new(report);
    let index = Layer2Index::build(report);

    // Provenance resolution set for Layer-1 (every real anchor). Layer-2 uses `index.resolve`.
    let l1_anchors: std::collections::BTreeSet<&str> =
        report.items.iter().map(|it| it.anchor.as_str()).collect();

    let mut per_question = Vec::with_capacity(questions.len());
    let (mut l1_c1, mut l1_ck) = (0usize, 0usize);
    let (mut l2_c1, mut l2_ck) = (0usize, 0usize);
    let (mut l1_prov_ok, mut l1_prov_total) = (0usize, 0usize);
    let (mut l2_prov_ok, mut l2_prov_total) = (0usize, 0usize);

    for q in questions {
        let l1 = layer1_topk(&engine, &q.question, k);
        let l2 = layer2_topk(&index, &q.question, k);

        let l1_hit1 = l1.first().is_some_and(|a| a == &q.gold_anchor);
        let l1_hitk = l1.iter().any(|a| a == &q.gold_anchor);
        let l2_hit1 = l2.first().is_some_and(|a| a == &q.gold_anchor);
        let l2_hitk = l2.iter().any(|a| a == &q.gold_anchor);

        l1_c1 += usize::from(l1_hit1);
        l1_ck += usize::from(l1_hitk);
        l2_c1 += usize::from(l2_hit1);
        l2_ck += usize::from(l2_hitk);

        // Provenance: every returned anchor must resolve to a real Layer-1 row.
        for a in &l1 {
            l1_prov_total += 1;
            l1_prov_ok += usize::from(l1_anchors.contains(a.as_str()));
        }
        for a in &l2 {
            l2_prov_total += 1;
            l2_prov_ok += usize::from(index.resolve(a).is_some());
        }

        per_question.push(PerQuestion {
            id: q.id.clone(),
            gold_anchor: q.gold_anchor.clone(),
            seed: q.seed,
            l1_top: l1.first().cloned(),
            l1_hit_at_1: l1_hit1,
            l1_hit_at_k: l1_hitk,
            l2_top: l2.first().cloned(),
            l2_hit_at_1: l2_hit1,
            l2_hit_at_k: l2_hitk,
        });
    }

    let total = questions.len();
    let l1_ns = measure_latency(trials, questions, |q| {
        let _ = layer1_topk(&engine, &q.question, k);
    });
    let l2_ns = measure_latency(trials, questions, |q| {
        let _ = layer2_topk(&index, &q.question, k);
    });

    let layer1 = SystemMetrics {
        system: "layer1".to_owned(),
        correct_at_1: l1_c1,
        correct_at_k: l1_ck,
        total,
        provenance_ok: l1_prov_ok,
        provenance_total: l1_prov_total,
        ns_per_query: l1_ns,
        trial_iters: trials,
    };
    let layer2 = SystemMetrics {
        system: "layer2".to_owned(),
        correct_at_1: l2_c1,
        correct_at_k: l2_ck,
        total,
        provenance_ok: l2_prov_ok,
        provenance_total: l2_prov_total,
        ns_per_query: l2_ns,
        trial_iters: trials,
    };

    let evidence = GateEvidence {
        questions: total,
        k,
        l1_correct_at_1: layer1.rate_at_1(),
        l2_correct_at_1: layer2.rate_at_1(),
        l1_correct_at_k: layer1.rate_at_k(),
        l2_correct_at_k: layer2.rate_at_k(),
        l2_provenance: layer2.provenance_fidelity(),
        l1_ns_per_query: layer1.ns_per_query,
        l2_ns_per_query: layer2.ns_per_query,
        band: REGRESSION_BAND,
    };
    let verdict = decide_gate(evidence);

    EvalReport {
        k,
        seed_master: crate::vsa2::TERO_L2_SEED,
        host_tag: host_tag(),
        questions_total: total,
        codebook_len: index.len(),
        refused_records: index.refused().len(),
        layer1,
        layer2,
        per_question,
        verdict,
    }
}

/// Time `run` over the whole question set, `trials` reps, returning ns/query (Empirical). Returns
/// `0.0` when there is nothing to time (no questions / zero trials) — never a divide-by-zero.
fn measure_latency<F: Fn(&EvalQuestion)>(trials: u32, questions: &[EvalQuestion], run: F) -> f64 {
    let denom = trials as usize * questions.len();
    if denom == 0 {
        return 0.0;
    }
    let start = Instant::now();
    for _ in 0..trials {
        for q in questions {
            run(q);
        }
    }
    start.elapsed().as_nanos() as f64 / denom as f64
}
