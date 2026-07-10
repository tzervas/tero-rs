//! `myc-sec` — security checks as tooling CLI (M-367; contract `docs/spec/Security-Checks-Contract.md`).
//!
//! Three families: the **`wild`-block audit** (in-repo, the new check), and **secrets** + **supply-chain**
//! (orchestrating the existing `scripts/checks/{secrets,deny}.sh`). The load-bearing honesty rule is
//! **skip ≠ pass**: a missing scanner is reported as *reduced coverage*, never folded into a clean bill
//! (G2/VR-5). Every finding cites *why*; severity is a fixed declared map.
//!
//! ```text
//! myc-sec [--project <dir>] [--strict] [--explain] [--no-secrets] [--no-supply-chain]
//! ```
//!
//! Exit codes: 0 clean (or only reduced-coverage) · 1 a failing finding (critical/high, a script failure,
//! or — with `--strict` — a medium) · 64 usage.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use mycelium_cli_common::Args;
use mycelium_sec::{audit_wild, collect_myc, explain_wild, Severity, WildAudit};

/// The status of an orchestrated (shell-script) family — the honest three-way (never just ok/fail).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Coverage {
    /// Ran fully, no findings.
    Ok,
    /// Ran, but a scanner was absent — partial coverage (NOT a pass).
    Reduced,
    /// Ran and found something (or the gate failed).
    Fail,
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: myc-sec [--project <dir>] [--strict] [--explain] [--no-secrets] [--no-supply-chain]"
    );
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let mut dir = PathBuf::from(".");
    let mut strict = false;
    let mut explain = false;
    let mut do_secrets = true;
    let mut do_supply = true;

    let mut args = Args::from_env();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--project" => match args.value() {
                Some(p) => dir = PathBuf::from(p),
                None => return usage(),
            },
            "--strict" => strict = true,
            "--explain" => explain = true,
            "--no-secrets" => do_secrets = false,
            "--no-supply-chain" => do_supply = false,
            _ => return usage(),
        }
    }

    // --- Family 1: the wild-block audit (in-repo) ---
    let audit = run_wild(&dir);
    let mut failing = false; // a finding that fails the gate
    let mut reduced = false;

    println!(
        "security: wild-audit — {} block(s), {} justified, {} unjustified",
        audit.inventory.len(),
        audit.justified(),
        audit.unjustified()
    );
    if explain {
        print!("{}", explain_wild(&audit));
    } else {
        for f in &audit.findings {
            println!(
                "  [{}] {} at {}: {}",
                f.severity.as_str(),
                f.rule,
                f.at,
                f.why
            );
        }
    }
    // An unjustified wild is `medium`: fails only under --strict (contract §3).
    if audit.findings.iter().any(|f| f.severity >= Severity::High) {
        failing = true;
    }
    if strict && !audit.findings.is_empty() {
        failing = true;
    }

    // --- Families 2 & 3: orchestrate the existing gates (skip ≠ pass) ---
    if do_secrets {
        match run_script(&dir, "scripts/checks/secrets.sh", "secrets") {
            Coverage::Ok => println!("security: secrets — ok"),
            Coverage::Reduced => {
                println!(
                    "security: secrets — REDUCED COVERAGE (a scanner is absent; not a clean bill)"
                );
                reduced = true;
            }
            Coverage::Fail => {
                println!("security: secrets — FAIL (potential secret; investigate)");
                failing = true;
            }
        }
    }
    if do_supply {
        match run_script(&dir, "scripts/checks/deny.sh", "supply-chain") {
            Coverage::Ok => println!("security: supply-chain — ok"),
            Coverage::Reduced => {
                println!(
                    "security: supply-chain — REDUCED COVERAGE (cargo-deny/audit absent; not a clean bill)"
                );
                reduced = true;
            }
            Coverage::Fail => {
                println!("security: supply-chain — FAIL (advisory/policy finding)");
                failing = true;
            }
        }
    }

    // The coverage receipt — an OK with reduced coverage is NOT a clean bill, and says so (the crux).
    let coverage = if reduced { "REDUCED" } else { "FULL" };
    if failing {
        eprintln!("myc-sec: findings present (coverage: {coverage})");
        ExitCode::from(1)
    } else {
        println!("myc-sec: no failing findings (coverage: {coverage})");
        ExitCode::SUCCESS
    }
}

/// Run the wild-audit over every `.myc` under `dir` (empty audit if the dir can't be read — reported, not
/// crashed; the absence of sources is honestly an empty inventory, not a hidden pass).
fn run_wild(dir: &Path) -> WildAudit {
    let files = match collect_myc(dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("myc-sec: {e}");
            return WildAudit::default();
        }
    };
    let mut sources = Vec::new();
    for f in files {
        if let Ok(src) = std::fs::read_to_string(&f) {
            let rel = f
                .strip_prefix(dir)
                .unwrap_or(&f)
                .to_string_lossy()
                .replace('\\', "/");
            sources.push((rel, src));
        }
    }
    audit_wild(&sources)
}

/// Orchestrate an existing check script, classifying its result honestly (skip ≠ pass). The scripts exit
/// 0 on success *or* graceful skip and print an `ok`/`skip`/`FAIL` marker — so exit code alone cannot tell
/// "clean" from "scanner absent". A missing script is itself reduced coverage (named, not silent).
fn run_script(dir: &Path, rel: &str, family: &str) -> Coverage {
    let script = dir.join(rel);
    if !script.is_file() {
        eprintln!("myc-sec: {family}: {rel} not found — reduced coverage");
        return Coverage::Reduced;
    }
    let output = match Command::new("bash").arg(&script).current_dir(dir).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("myc-sec: {family}: could not run {rel}: {e} — reduced coverage");
            return Coverage::Reduced;
        }
    };
    if !output.status.success() {
        return Coverage::Fail;
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if text.contains("FAIL") {
        Coverage::Fail
    } else if text.contains("skip") {
        Coverage::Reduced
    } else {
        Coverage::Ok
    }
}
