//! `tero-eval` — the Layer-2 **eval harness runner** (M-1018 / DN-87 §6.1). Loads the committed
//! question set + the committed Layer-1 index, runs [`mycelium_tero::run_eval`] (Layer-2 VSA vs the
//! Layer-1 baseline), prints correctness/provenance/latency **with denominators + seed + host tag**,
//! and records the **append-only gate verdict** — writing the machine `eval/verdict.json`, the fresh
//! `eval/latency-baseline.json`, and appending the human `eval/VERDICT.md`.
//!
//! **Honesty (G2/VR-5):** the verdict is *computed and recorded honestly* — it is **Closed by
//! default** and never stamped "beats RAG" without a real harness win. For this ~5k-row structured
//! corpus a **Closed gate is the expected, successful outcome**: the system keeps serving Layer-1
//! answers and the improved-on-RAG claim stays aspiration. Skip-graceful: a checkout without the
//! committed index yields a clean no-op (exit 0), not a failure.
//!
//! Usage:
//! ```text
//!   tero-eval [--index <path>] [--questions <path>] [--eval-dir <dir>] [--trials N] [--k N]
//! ```
//! Exit codes: `0` ok/skip · `64` usage · `66` I/O.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mycelium_tero::{
    latency_classify, load_report, run_eval, EvalReport, EvalSuite, GateVerdict, LatencyBaseline,
    LatencyEntry,
};

const EX_OK: u8 = 0;
const EX_USAGE: u8 = 64;
const EX_IO: u8 = 66;

/// The `(case, system)` id the aggregate latency is recorded under in the baseline.
const AGGREGATE_CASE: &str = "aggregate";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => ExitCode::from(code),
        Err((code, msg)) => {
            eprintln!("tero-eval: {msg}");
            ExitCode::from(code)
        }
    }
}

struct Config {
    index: PathBuf,
    questions: PathBuf,
    eval_dir: PathBuf,
    trials: u32,
    k: usize,
}

fn run(args: &[String]) -> Result<u8, (u8, String)> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest
        .ancestors()
        .nth(2)
        .unwrap_or(&manifest)
        .to_path_buf();
    let mut cfg = Config {
        index: repo_root.join("docs/tero-index/index.json"),
        questions: manifest.join("eval/questions.json"),
        eval_dir: manifest.join("eval"),
        trials: 5,
        k: 5,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => cfg.index = PathBuf::from(next(args, &mut i)?),
            "--questions" => cfg.questions = PathBuf::from(next(args, &mut i)?),
            "--eval-dir" => cfg.eval_dir = PathBuf::from(next(args, &mut i)?),
            "--trials" => {
                cfg.trials = next(args, &mut i)?.parse().map_err(|_| {
                    (
                        EX_USAGE,
                        format!("--trials must be an integer\n{}", usage()),
                    )
                })?;
            }
            "--k" => {
                cfg.k = next(args, &mut i)?
                    .parse()
                    .map_err(|_| (EX_USAGE, format!("--k must be an integer\n{}", usage())))?;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(EX_OK);
            }
            other => return Err((EX_USAGE, format!("unknown argument: {other}\n{}", usage()))),
        }
        i += 1;
    }

    // Skip-graceful: no committed index ⇒ a clean no-op (a stripped checkout, e.g. before the first
    // `tero-index` run). Never a failure (G2 — the absence is stated, not swallowed).
    if !cfg.index.exists() {
        println!(
            ">> tero-eval: no committed Layer-1 index at {} — skipping (run `tero-index` first)",
            cfg.index.display()
        );
        return Ok(EX_OK);
    }
    let report = load_report(&cfg.index).map_err(|e| (EX_IO, format!("load index: {e}")))?;

    let suite_text = std::fs::read_to_string(&cfg.questions).map_err(|e| {
        (
            EX_IO,
            format!("read questions {}: {e}", cfg.questions.display()),
        )
    })?;
    let suite =
        EvalSuite::from_json(&suite_text).map_err(|e| (EX_IO, format!("parse questions: {e}")))?;

    let report_eval = run_eval(&report, &suite.questions, cfg.k, cfg.trials);
    print_report(&report_eval, &report.items.len());

    // Latency regression classify vs the committed baseline (if any), then refresh it.
    let baseline_path = cfg.eval_dir.join("latency-baseline.json");
    classify_latency(&baseline_path, &report_eval);

    write_artifacts(&cfg.eval_dir, &baseline_path, &report_eval)?;
    Ok(EX_OK)
}

fn next(args: &[String], i: &mut usize) -> Result<String, (u8, String)> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or((EX_USAGE, format!("missing argument value\n{}", usage())))
}

fn print_report(r: &EvalReport, index_rows: &usize) {
    println!(
        ">> tero-eval (M-1018): Layer-2 VSA vs Layer-1 baseline over {index_rows} indexed rows"
    );
    println!(
        "   seed (Layer-2 master) = 0x{:016X} · host = {} · k = {} · trials = {}",
        r.seed_master, r.host_tag, r.k, r.layer1.trial_iters
    );
    println!(
        "   codebook = {} record(s) encoded, {} refused (never-silent)",
        r.codebook_len, r.refused_records
    );
    println!("   questions graded = {}", r.questions_total);
    println!(
        "   correctness@1 : Layer-1 {}/{} ({:.3}) · Layer-2 {}/{} ({:.3})",
        r.layer1.correct_at_1,
        r.layer1.total,
        r.layer1.rate_at_1(),
        r.layer2.correct_at_1,
        r.layer2.total,
        r.layer2.rate_at_1()
    );
    println!(
        "   correctness@{}: Layer-1 {}/{} ({:.3}) · Layer-2 {}/{} ({:.3})",
        r.k,
        r.layer1.correct_at_k,
        r.layer1.total,
        r.layer1.rate_at_k(),
        r.layer2.correct_at_k,
        r.layer2.total,
        r.layer2.rate_at_k()
    );
    println!(
        "   provenance    : Layer-1 {}/{} ({:.3}) · Layer-2 {}/{} ({:.3})  [must be 1.0 to open]",
        r.layer1.provenance_ok,
        r.layer1.provenance_total,
        r.layer1.provenance_fidelity(),
        r.layer2.provenance_ok,
        r.layer2.provenance_total,
        r.layer2.provenance_fidelity()
    );
    println!(
        "   latency (Empirical, ns/query): Layer-1 {:.0} · Layer-2 {:.0}",
        r.layer1.ns_per_query, r.layer2.ns_per_query
    );
    println!("   GATE VERDICT: {}", r.verdict.status());
    if let GateVerdict::Closed { reason, .. } = &r.verdict {
        println!("     reason: {reason}");
        println!(
            "     → serving Layer-1 answers; the improved-on-RAG claim stays aspiration (G2/VR-5)."
        );
    }
}

fn classify_latency(baseline_path: &Path, r: &EvalReport) {
    let Ok(text) = std::fs::read_to_string(baseline_path) else {
        println!("   latency vs baseline: no committed baseline yet (writing a fresh one)");
        return;
    };
    match LatencyBaseline::from_json(&text) {
        Ok(baseline) => {
            for (system, ns) in [
                ("layer1", r.layer1.ns_per_query),
                ("layer2", r.layer2.ns_per_query),
            ] {
                let outcome =
                    latency_classify(&baseline, &r.host_tag, AGGREGATE_CASE, system, Some(ns));
                println!("   latency vs baseline [{system}]: {}", outcome.status());
            }
        }
        // Never-silent: a malformed baseline is reported, not treated as "no baseline".
        Err(e) => eprintln!("   latency baseline unreadable ({e}) — not classified"),
    }
}

fn write_artifacts(
    eval_dir: &Path,
    baseline_path: &Path,
    r: &EvalReport,
) -> Result<(), (u8, String)> {
    std::fs::create_dir_all(eval_dir)
        .map_err(|e| (EX_IO, format!("create {}: {e}", eval_dir.display())))?;

    // Machine artifact: the full eval report.
    let verdict_json = serde_json::to_string_pretty(r)
        .map_err(|e| (EX_IO, format!("serialize verdict.json: {e}")))?;
    std::fs::write(eval_dir.join("verdict.json"), verdict_json + "\n")
        .map_err(|e| (EX_IO, format!("write verdict.json: {e}")))?;

    // Fresh latency baseline (this run's numbers, tagged with this host).
    let baseline = LatencyBaseline {
        host_tag: r.host_tag.clone(),
        captured: format!("tero-eval run over {} questions", r.questions_total),
        trial_iters: r.layer1.trial_iters,
        entries: vec![
            LatencyEntry {
                case_id: AGGREGATE_CASE.to_owned(),
                system: "layer1".to_owned(),
                ns_per_call: r.layer1.ns_per_query,
            },
            LatencyEntry {
                case_id: AGGREGATE_CASE.to_owned(),
                system: "layer2".to_owned(),
                ns_per_call: r.layer2.ns_per_query,
            },
        ],
    };
    let baseline_json = baseline
        .to_json()
        .map_err(|e| (EX_IO, format!("serialize latency-baseline.json: {e}")))?;
    std::fs::write(baseline_path, baseline_json + "\n")
        .map_err(|e| (EX_IO, format!("write latency-baseline.json: {e}")))?;

    // Human artifact: append (never overwrite) the gate verdict record.
    append_verdict_md(&eval_dir.join("VERDICT.md"), r)?;
    Ok(())
}

/// Append (append-only, DN-87 §6-style) a run section to `VERDICT.md`. A first run writes the header.
fn append_verdict_md(path: &Path, r: &EvalReport) -> Result<(), (u8, String)> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let run_n =
        existing.matches("\n## Run ").count() + usize::from(existing.starts_with("## Run ")) + 1;

    let mut out = String::new();
    if existing.is_empty() {
        out.push_str(
            "# Layer-2 eval gate — VERDICT (append-only)\n\n\
             The M-1018 gate record (DN-87 §6.1). Each run appends a section; history is never \
             rewritten. The gate is **Closed by default** and opens only on a measured Layer-2 win \
             that keeps provenance at 1.0 and latency within the band. A **Closed gate is the honest, \
             expected outcome** for this ~5k-row structured corpus — the system serves Layer-1 \
             answers and the improved-on-RAG claim stays aspiration (G2/VR-5).\n",
        );
    } else {
        out.push_str(&existing);
    }

    let reason = match &r.verdict {
        GateVerdict::Closed { reason, .. } => reason.clone(),
        GateVerdict::Open { .. } => "opened — Layer-2 met every gate criterion".to_owned(),
    };
    out.push_str(&format!(
        "\n## Run {run_n} — gate {status}\n\n\
         - host: {host}\n\
         - seed (Layer-2 master): 0x{seed:016X}\n\
         - questions: {q} · k = {k} · codebook = {cb} records ({refused} refused, never-silent)\n\
         - correctness@1: Layer-1 {l1c1}/{tot} ({l1r1:.3}) · Layer-2 {l2c1}/{tot} ({l2r1:.3})\n\
         - correctness@{k}: Layer-1 {l1ck}/{tot} ({l1rk:.3}) · Layer-2 {l2ck}/{tot} ({l2rk:.3})\n\
         - provenance fidelity: Layer-1 {l1p:.3} · Layer-2 {l2p:.3} (must be 1.0 to open)\n\
         - latency (Empirical, ns/query, {trials} trials): Layer-1 {l1ns:.0} · Layer-2 {l2ns:.0}\n\
         - verdict: {status} — {reason}\n",
        run_n = run_n,
        status = r.verdict.status(),
        host = r.host_tag,
        seed = r.seed_master,
        q = r.questions_total,
        k = r.k,
        cb = r.codebook_len,
        refused = r.refused_records,
        l1c1 = r.layer1.correct_at_1,
        l2c1 = r.layer2.correct_at_1,
        tot = r.layer1.total,
        l1r1 = r.layer1.rate_at_1(),
        l2r1 = r.layer2.rate_at_1(),
        l1ck = r.layer1.correct_at_k,
        l2ck = r.layer2.correct_at_k,
        l1rk = r.layer1.rate_at_k(),
        l2rk = r.layer2.rate_at_k(),
        l1p = r.layer1.provenance_fidelity(),
        l2p = r.layer2.provenance_fidelity(),
        trials = r.layer1.trial_iters,
        l1ns = r.layer1.ns_per_query,
        l2ns = r.layer2.ns_per_query,
        reason = reason,
    ));

    std::fs::write(path, out).map_err(|e| (EX_IO, format!("write VERDICT.md: {e}")))?;
    Ok(())
}

fn usage() -> String {
    "usage: tero-eval [--index <path>] [--questions <path>] [--eval-dir <dir>] [--trials N] [--k N]"
        .to_owned()
}
