//! `mycelium-bench` runnable harness — runs the execution-backend benchmark over the shared corpus,
//! ingests the LLM-harness report, and writes a deterministic markdown + JSON report into
//! `crates/mycelium-bench/reports/`.
//!
//! **Run with `--release`** (`cargo run --release -p mycelium-bench --bin bench`). A debug build is
//! refused — its timings are not representative (no optimisation, overflow checks on), so any WIN/LOSS
//! verdict from it would be dishonest (VR-5/G2).
//!
//! It prints a short human summary to stdout and writes the full report to `reports/`. Optional flags:
//! `--out <DIR>` to redirect the report directory; `--stdout` to also print the markdown;
//! `--scaling [N]` to also run the multicore scaling suite (M-859; `N` caps worker count, default
//! host parallelism — this is *slow*, opt-in, off by default); `--baseline <FILE>` to gate this run's
//! single-core timings against a committed [`RegressionBaseline`] JSON (M-859; opt-in, off by
//! default — no baseline is fabricated when the flag is absent).

use std::path::{Path, PathBuf};

use mycelium_bench::backend::Engines;
use mycelium_bench::corpus::corpus;
use mycelium_bench::llm::{LlmIngestError, LlmReport};
use mycelium_bench::measure::run_corpus;
use mycelium_bench::report::{neutral_band, Honesty, LlmSection, Report};
use mycelium_bench::scaling::run_scaling;
use mycelium_bench::timing::refuse_debug_build;
use mycelium_bench::verdict::RegressionBaseline;

fn main() {
    // 1. Honest profile gate: never measure a debug build.
    refuse_debug_build();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = parse_out_dir(&args).unwrap_or_else(default_reports_dir);
    let also_stdout = args.iter().any(|a| a == "--stdout");
    let scaling_max_workers = parse_scaling_flag(&args);
    let baseline_path = parse_baseline_flag(&args);

    eprintln!("mycelium-bench: running the execution-backend corpus (release build)...");
    let eng = Engines::default();
    let cases = corpus();
    let run = run_corpus(&cases, &eng);

    // 2. Ingest the LLM-harness report: prefer the newest real one; fall back to the committed
    //    SYNTHETIC sample (labeled synthetic). Absence is recorded, never synthesized.
    let llm = ingest_llm_section();

    // 3. Multicore scaling (M-859) — opt-in via `--scaling [N]`, off by default (it is
    //    substantially slower: every case x backend is timed across every worker count 1..=N).
    let scaling = scaling_max_workers.map(|max_workers| {
        eprintln!(
            "mycelium-bench: running the multicore scaling suite (1..={max_workers} workers, this \
             is the slow opt-in path)..."
        );
        run_scaling(&cases, max_workers, 8, 3)
    });

    let mut report = Report {
        tool: "mycelium-bench",
        profile: "release",
        mlir_dialect_feature: cfg!(feature = "mlir-dialect"),
        host_note: host_note(),
        honesty: Honesty::default(),
        neutral_band: neutral_band(),
        run,
        llm,
        scaling,
        regression: None,
    };

    // 4. Regression gate (M-859) — opt-in via `--baseline <FILE>`. A malformed/unreadable baseline
    //    is a loud failure (exit 1), never a silent skip of the gate the caller explicitly asked for.
    if let Some(path) = baseline_path {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!(
                "mycelium-bench: cannot read baseline {}: {e}",
                path.display()
            );
            std::process::exit(1);
        });
        let baseline = RegressionBaseline::from_json(&text).unwrap_or_else(|e| {
            eprintln!(
                "mycelium-bench: baseline {} is not valid JSON: {e}",
                path.display()
            );
            std::process::exit(1);
        });
        report = report.with_regression_gate(&mycelium_bench::host_tag(), &baseline);
    }

    // 3. Emit both projections (G11 dual projection), deterministically.
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!(
            "mycelium-bench: cannot create report dir {}: {e}",
            out_dir.display()
        );
        std::process::exit(1);
    }
    let md = report.to_markdown();
    let json = match report.to_json() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("mycelium-bench: failed to serialize JSON report: {e}");
            std::process::exit(1);
        }
    };
    // Deterministic stable filenames (latest-run convention) so the committed report is diffable.
    let md_path = out_dir.join("latest-report.md");
    let json_path = out_dir.join("latest-report.json");
    write_or_die(&md_path, &md);
    write_or_die(&json_path, &json);

    // 4. Short human summary to stdout.
    let t = report.tallies();
    println!("mycelium-bench summary:");
    println!(
        "  cases: {}   wins: {}   neutral: {}   speed-losses: {}   correctness-losses: {}   \
         capability-losses: {}   errors: {}   skips: {}",
        report.run.cases.len(),
        t.wins,
        t.neutral,
        t.speed_losses,
        t.correctness_losses,
        t.capability_losses,
        t.errors,
        t.skips,
    );
    if t.baseline_failures > 0 {
        println!(
            "  WARNING: {} baseline (interpreter) failure(s) — the trusted base failed; investigate.",
            t.baseline_failures
        );
    }
    println!("  report (markdown): {}", md_path.display());
    println!("  report (json)    : {}", json_path.display());
    if let Some(sec) = &report.llm {
        println!(
            "  llm-harness      : {} ({})",
            sec.source_path,
            if sec.is_synthetic {
                "SYNTHETIC sample"
            } else {
                "real run"
            }
        );
    } else {
        println!("  llm-harness      : none found (section recorded empty, not synthesized)");
    }
    if let Some(run) = &report.scaling {
        println!(
            "  scaling (M-859)  : {} points across worker counts {:?} (host: {})",
            run.points.len(),
            run.worker_counts,
            run.host_note,
        );
    } else {
        println!("  scaling (M-859)  : not run (pass --scaling [N] to opt in)");
    }
    if let Some(sec) = &report.regression {
        let regressions = sec
            .rows
            .iter()
            .filter(|r| r.outcome.is_regression())
            .count();
        println!(
            "  regression gate  : {} row(s) vs baseline (captured {}); {} REGRESSION(S)",
            sec.rows.len(),
            sec.baseline_captured,
            regressions,
        );
    } else {
        println!("  regression gate  : not run (pass --baseline <FILE> to opt in)");
    }

    if also_stdout {
        println!("\n{md}");
    }
}

/// `--out <DIR>` override for the report directory.
fn parse_out_dir(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--out" {
            return it.next().map(PathBuf::from);
        }
    }
    None
}

/// `--scaling [N]` — opt into the multicore scaling suite. `N` (optional) caps the worker count;
/// absent or unparsable, it defaults to the host's available parallelism
/// (`mycelium_std_runtime::scheduler::Scheduler::new().workers()`, floor 1). Returns `None` when the
/// flag was not passed at all (the suite is off by default — it is materially slower).
fn parse_scaling_flag(args: &[String]) -> Option<usize> {
    let idx = args.iter().position(|a| a == "--scaling")?;
    let default_workers = mycelium_std_runtime::scheduler::Scheduler::new().workers();
    let n = args
        .get(idx + 1)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default_workers);
    Some(n)
}

/// `--baseline <FILE>` — opt into the regression gate against a committed baseline JSON.
fn parse_baseline_flag(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--baseline" {
            return it.next().map(PathBuf::from);
        }
    }
    None
}

/// The default report dir: `<crate>/reports/`, resolved relative to this source file so it works from
/// any CWD (`CARGO_MANIFEST_DIR` is the crate root at build time).
fn default_reports_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reports")
}

/// Best-effort one-line host note for report provenance (target triple + thread count). No PII.
/// Delegates to [`mycelium_bench::host_note_for_scaling`] — a single canonical implementation (the
/// prior copy here duplicated it exactly, which is exactly the kind of drift that produced the
/// `host_tag()`-vs-`host_note()` format mismatch this module's regression gate had to fix, M-859).
fn host_note() -> String {
    mycelium_bench::host_note_for_scaling()
}

/// Find the LLM-harness report to ingest. Order:
/// 1. the newest `*-report.json` in `tools/llm-harness/reports/` (a real or fixture run), else
/// 2. the committed synthetic sample if one is present there, else
/// 3. `None` (recorded as "no report", never synthesized).
fn ingest_llm_section() -> Option<LlmSection> {
    let reports_dir = harness_reports_dir()?;
    let path = match LlmReport::newest_in_dir(&reports_dir) {
        Ok(Some(p)) => p,
        Ok(None) => {
            eprintln!(
                "mycelium-bench: no LLM-harness report found under {} — LLM section recorded empty.",
                reports_dir.display()
            );
            return None;
        }
        Err(LlmIngestError::Io(m)) | Err(LlmIngestError::Parse(m)) => {
            eprintln!("mycelium-bench: could not scan LLM reports dir: {m}");
            return None;
        }
    };
    match LlmReport::from_path(&path) {
        Ok(rep) => {
            let synthetic = rep.is_synthetic();
            if synthetic {
                eprintln!(
                    "mycelium-bench: ingesting SYNTHETIC llm-harness sample {} (labeled synthetic).",
                    path.display()
                );
            }
            Some(LlmSection::from_report(
                &rep,
                path.display().to_string(),
                synthetic,
            ))
        }
        Err(e) => {
            eprintln!(
                "mycelium-bench: failed to read LLM report {}: {e}",
                path.display()
            );
            None
        }
    }
}

/// Resolve `tools/llm-harness/reports/` relative to the workspace root (the crate is at
/// `<root>/crates/mycelium-bench`, so the harness dir is two levels up). Returns `None` if it does
/// not exist.
fn harness_reports_dir() -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest
        .parent() // crates/
        .and_then(Path::parent) // <root>/
        .map(|root| root.join("tools/llm-harness/reports"))?;
    dir.is_dir().then_some(dir)
}

fn write_or_die(path: &Path, contents: &str) {
    if let Err(e) = std::fs::write(path, contents) {
        eprintln!("mycelium-bench: failed to write {}: {e}", path.display());
        std::process::exit(1);
    }
}
