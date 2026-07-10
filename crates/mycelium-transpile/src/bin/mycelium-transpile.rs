//! CLI for `mycelium-transpile` (M-873, batch mode added in the follow-on wave; `--vet` added in
//! M-1000): `mycelium-transpile [--vet] <input> <out-dir>`.
//!
//! `<input>` is either:
//! - a single `.rs` file — writes `<out-dir>/<stem>.myc` + `<out-dir>/<stem>.gap.json`, then
//!   prints a one-line summary (unchanged single-file behavior); or
//! - a directory (typically a crate's `src/`) — recurses every `*.rs` file (skipping test
//!   infrastructure, `src/batch.rs::discover_rs_files`), transpiles each independently, writes
//!   the same per-file `<stem>.myc`/`<stem>.gap.json` pair for every discovered file **plus** two
//!   combined artifacts: `<out-dir>/summary.json` (per-file + aggregate counts) and
//!   `<out-dir>/union.gap.json` (every gap from every file, plus aggregate category counts).
//!
//! `--vet` (M-1000) runs the **real** `myc check` oracle over every emitted `.myc`, writes
//! `<out-dir>/vet.json` (per-file + aggregate vet records), and prints the **`checked_fraction`**
//! (myc-check-clean coverage) alongside the emission-only `expressible_fraction`. The oracle is the
//! pre-built `MYC_CHECK_CMD` binary when that env var is set (the sanctioned, build-lock-safe form
//! `scripts/checks/transpile-vet.sh` uses), else the `cargo run -p mycelium-check` fallback
//! (`crate::vet::MycChecker::from_env`). See `src/vet.rs` for the metric's stated denominator.
//!
//! Every emitted artifact is `Declared`/unvalidated (see `src/lib.rs`); the vet verdict is
//! `Empirical` (measured — see `src/vet.rs`). No `clap` dependency — plain `std::env::args`
//! (kickoff-scoped minimal deps).

use mycelium_transpile::batch::{discover_rs_files, summarize, transpile_batch};
use mycelium_transpile::vet::{vet_batch, MycChecker, VetInput, VetReport};
use mycelium_transpile::{transpile_file, GapReport};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    // Parse a minimal flag set: an optional `--vet` before the two positional args. Kept hand-rolled
    // (no `clap`) per the crate's minimal-deps stance.
    let mut vet = false;
    let mut positional: Vec<String> = Vec::new();
    for a in env::args().skip(1) {
        match a.as_str() {
            "--vet" => vet = true,
            _ => positional.push(a),
        }
    }
    if positional.len() != 2 {
        eprintln!("usage: mycelium-transpile [--vet] <input.rs | input-dir> <out-dir>");
        return ExitCode::FAILURE;
    }
    let input = Path::new(&positional[0]);
    let out_dir = Path::new(&positional[1]);

    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!(
            "mycelium-transpile: failed to create {}: {e}",
            out_dir.display()
        );
        return ExitCode::FAILURE;
    }

    if input.is_dir() {
        run_batch(input, out_dir, vet)
    } else {
        run_single_file(input, out_dir, vet)
    }
}

/// Run the vet loop over the written `.myc` files and report `checked_fraction` alongside
/// `expressible_fraction`. Advisory: a vet failure/tool-unavailable is reported (never silent, G2)
/// but does **not** change the process exit code — vetting is a measurement, not a gate.
fn run_vet(inputs: &[VetInput], out_dir: &Path) {
    if inputs.is_empty() {
        eprintln!("mycelium-transpile: --vet: no emitted .myc files to vet");
        return;
    }
    // Cargo-fallback runs in the current directory (typically the workspace root); the sanctioned
    // path is a pre-built `MYC_CHECK_CMD` binary, which carries its own absolute program path.
    let checker = MycChecker::from_env(env::current_dir().ok());
    let report = vet_batch(&checker, inputs);
    let vet_path = out_dir.join("vet.json");
    match serde_json::to_string_pretty(&report) {
        Ok(j) => {
            if let Err(e) = fs::write(&vet_path, j) {
                eprintln!(
                    "mycelium-transpile: failed to write {}: {e}",
                    vet_path.display()
                );
            }
        }
        Err(e) => eprintln!("mycelium-transpile: failed to serialize vet.json: {e}"),
    }
    print_vet_summary(&report, &vet_path);
}

fn print_vet_summary(report: &VetReport, vet_path: &Path) {
    let (clean_files, files_with_emissions) = report.clean_file_fraction();
    // Per-class file breakdown, deterministically ordered (BTreeMap).
    let classes = report
        .class_counts
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "mycelium-transpile: --vet over {} file(s) — checked_fraction {:.1}% ({}/{} items \
         myc-check-clean, file-gated) vs expressible_fraction {:.1}% ({}/{} items emitted); \
         {clean_files}/{files_with_emissions} file(s) with emissions fully clean [{classes}] -> {}",
        report.records.len(),
        report.checked_fraction() * 100.0,
        report.total_checked_clean_items,
        report.total_non_test_items,
        report.expressible_fraction() * 100.0,
        report.total_emitted_items,
        report.total_non_test_items,
        vet_path.display(),
    );
}

/// Write `<out_dir>/<stem>.myc` + `<out_dir>/<stem>.gap.json` for one already-transpiled file.
/// Shared by both single-file and batch mode so the two never drift.
fn write_pair(
    stem: &str,
    myc_text: &str,
    report: &GapReport,
    out_dir: &Path,
) -> Result<(), String> {
    let myc_path = out_dir.join(format!("{stem}.myc"));
    let gap_path = out_dir.join(format!("{stem}.gap.json"));
    fs::write(&myc_path, myc_text)
        .map_err(|e| format!("failed to write {}: {e}", myc_path.display()))?;
    let gap_json = serde_json::to_string_pretty(report)
        .map_err(|e| format!("failed to serialize gap report for {stem}: {e}"))?;
    fs::write(&gap_path, gap_json)
        .map_err(|e| format!("failed to write {}: {e}", gap_path.display()))?;
    Ok(())
}

fn run_single_file(input: &Path, out_dir: &Path, vet: bool) -> ExitCode {
    let (myc_text, report) = match transpile_file(input) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("mycelium-transpile: {e}");
            return ExitCode::FAILURE;
        }
    };

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    if let Err(e) = write_pair(stem, &myc_text, &report, out_dir) {
        eprintln!("mycelium-transpile: {e}");
        return ExitCode::FAILURE;
    }

    let emitted = report.emitted_items.len();
    let gapped = report.gaps.len();
    let non_test = report.non_test_item_count();
    println!(
        "mycelium-transpile: {} top-level item(s) ({} non-test) — {} emitted, {} gap(s) \
         recorded, {:.1}% expressible -> {}/{stem}.myc, {}/{stem}.gap.json",
        report.total_top_level_items,
        non_test,
        emitted,
        gapped,
        report.expressible_fraction() * 100.0,
        out_dir.display(),
        out_dir.display(),
    );

    if vet {
        let myc_path: PathBuf = out_dir.join(format!("{stem}.myc"));
        let inputs = vec![VetInput::from_report(myc_path, &report)];
        run_vet(&inputs, out_dir);
    }
    ExitCode::SUCCESS
}

fn run_batch(input_dir: &Path, out_dir: &Path, vet: bool) -> ExitCode {
    let files = match discover_rs_files(input_dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "mycelium-transpile: failed to walk {}: {e}",
                input_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };
    if files.is_empty() {
        eprintln!(
            "mycelium-transpile: no .rs files found under {} (after skipping test \
             infrastructure)",
            input_dir.display()
        );
        return ExitCode::FAILURE;
    }

    let (results, failures) = transpile_batch(&files);
    // A hard parse/read failure is never silently dropped from the run (G2) — it is reported and
    // fails the process, distinct from a per-item gap (which the summary/union artifacts do
    // capture).
    for (path, err) in &failures {
        eprintln!("mycelium-transpile: {}: {err}", path.display());
    }

    // Per-file artifacts, named by stem — collisions (two files sharing a stem, e.g. two
    // `mod.rs`) are resolved by keeping the *last* write and flagging it loudly (never silent),
    // since this PoC's per-file naming scheme has no path-qualification mechanism.
    let mut seen_stems: std::collections::HashMap<String, std::path::PathBuf> =
        std::collections::HashMap::new();
    // Collect one vet input per written `.myc` (only used when `--vet`); the myc path mirrors
    // `write_pair`'s naming so a stem collision (last-writer-wins, warned above) vets the file that
    // actually landed on disk.
    let mut vet_inputs: std::collections::BTreeMap<String, VetInput> =
        std::collections::BTreeMap::new();
    for r in &results {
        let stem = r
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string();
        if let Some(prev) = seen_stems.insert(stem.clone(), r.path.clone()) {
            eprintln!(
                "mycelium-transpile: WARNING stem collision `{stem}.myc`/`{stem}.gap.json` — \
                 {} overwrites {} (no path-qualification in this PoC's per-file naming)",
                r.path.display(),
                prev.display()
            );
        }
        if let Err(e) = write_pair(&stem, &r.myc, &r.report, out_dir) {
            eprintln!("mycelium-transpile: {e}");
            return ExitCode::FAILURE;
        }
        if vet {
            let myc_path = out_dir.join(format!("{stem}.myc"));
            vet_inputs.insert(stem, VetInput::from_report(myc_path, &r.report));
        }
    }

    let (batch_summary, union) = summarize(&results);

    let summary_path = out_dir.join("summary.json");
    match serde_json::to_string_pretty(&batch_summary) {
        Ok(j) => {
            if let Err(e) = fs::write(&summary_path, j) {
                eprintln!(
                    "mycelium-transpile: failed to write {}: {e}",
                    summary_path.display()
                );
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("mycelium-transpile: failed to serialize summary.json: {e}");
            return ExitCode::FAILURE;
        }
    }

    let union_path = out_dir.join("union.gap.json");
    match serde_json::to_string_pretty(&union) {
        Ok(j) => {
            if let Err(e) = fs::write(&union_path, j) {
                eprintln!(
                    "mycelium-transpile: failed to write {}: {e}",
                    union_path.display()
                );
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("mycelium-transpile: failed to serialize union.gap.json: {e}");
            return ExitCode::FAILURE;
        }
    }

    println!(
        "mycelium-transpile: batch over {} file(s) ({} failed to parse) — {} top-level item(s) \
         ({} non-test), {} emitted, {} gap(s), {:.1}% expressible -> {}, {}",
        results.len(),
        failures.len(),
        batch_summary.totals.total_items,
        batch_summary.totals.non_test_items,
        batch_summary.totals.emitted,
        batch_summary.totals.gaps,
        batch_summary.totals.expressible_pct,
        summary_path.display(),
        union_path.display(),
    );

    if vet {
        let inputs: Vec<VetInput> = vet_inputs.into_values().collect();
        run_vet(&inputs, out_dir);
    }

    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
