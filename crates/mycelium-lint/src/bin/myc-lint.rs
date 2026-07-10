//! `myc-lint` — lint + auto-fix CLI (M-366; contract `docs/spec/Lint-and-Autofix-Contract.md`).
//!
//! Surfaces the M-141 lints + header lints as **actionable, reified, opt-in** fixes (suggest / apply /
//! scaffold). **No silent rewrite** (G2): `--fix` applies only behaviour-preserving `apply` edits — and in
//! v0 there are none (every lint fix is suggest or scaffold; header canonicalization is `mycfmt`'s job), so
//! `--fix` rewrites nothing. A control-flow change (an explicit `swap`, a recovery handler) is always a
//! **scaffold**, never auto-applied (RFC-0014 I1/I5).
//!
//! ```text
//! myc-lint [--project <dir>] [--fix] [--explain] [<file.myc | ->...]
//! ```
//!
//! Exit codes: 0 clean (or warnings only) · 1 an error-severity finding · 64 usage · 66 I/O.

use std::path::Path;
use std::process::ExitCode;

use mycelium_cli_common::{read_source, walk_myc, Args};
use mycelium_lint::{doc_lint_status, lint_sources, LintReport};
use mycelium_lsp::Severity;

fn usage() -> ExitCode {
    eprintln!("usage: myc-lint [--project <dir>] [--fix] [--explain] [<file.myc | ->...]");
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let mut project: Option<String> = None;
    let mut fix = false;
    let mut explain = false;
    let mut paths: Vec<String> = Vec::new();

    let mut args = Args::from_env();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--project" => match args.value() {
                Some(p) => project = Some(p),
                None => return usage(),
            },
            "--fix" => fix = true,
            "--explain" => explain = true,
            "-" => paths.push("-".to_owned()),
            s if s.starts_with("--") => return usage(),
            s => paths.push(s.to_owned()),
        }
    }

    let sources = match collect_sources(project.as_deref(), &paths) {
        Ok(s) => s,
        Err(code) => return code,
    };
    if sources.is_empty() {
        eprintln!("myc-lint: no .myc sources to lint");
        return usage();
    }

    let report = lint_sources(&sources);
    print_report(&report, explain, fix);

    if report.has_errors() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn print_report(report: &LintReport, explain: bool, fix: bool) {
    for f in &report.findings {
        let sev = match f.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        let tier = f.fix.as_ref().map_or("", |x| x.tier.as_str());
        let tier_note = if tier.is_empty() {
            String::new()
        } else {
            format!(" (fix: {tier})")
        };
        println!(
            "{}: {sev} {} at {}{tier_note}: {}",
            f.file, f.code, f.at, f.message
        );
        if explain {
            if let Some(fx) = &f.fix {
                println!("    → {}", fx.description);
                if let Some(scaffold) = &fx.scaffold {
                    for line in scaffold.lines() {
                        println!("      {line}");
                    }
                }
            }
        }
    }

    let (apply, suggest, scaffold) = report.tier_counts();
    eprintln!(
        "myc-lint: {} finding(s) across {} file(s) — {apply} apply / {suggest} suggest / {scaffold} scaffold",
        report.findings.len(),
        report.files
    );
    if fix {
        // The never-silent boundary, made explicit: v0 has no behaviour-preserving auto-fix to apply.
        eprintln!(
            "myc-lint: --fix applied 0 edit(s) — v0 has no safe auto-fix (suggest/scaffold only; \
             header canonicalization is `mycfmt`'s job). Nothing was rewritten (G2)."
        );
    }
    if explain {
        eprintln!("myc-lint: {}", doc_lint_status());
    }
}

/// Resolve the sources: `--project <dir>` walks for `.myc`; explicit paths are read (`-` = stdin);
/// neither → the current directory. The `.myc` walk is `mycelium_cli_common::walk_myc` (shared with
/// `myc-sec`); explicit `-`/path reads go through `read_source` (shared with `mycfmt`/`myc-check`).
fn collect_sources(
    project: Option<&str>,
    paths: &[String],
) -> Result<Vec<(String, String)>, ExitCode> {
    let mut out = Vec::new();

    if let Some(dir) = project {
        for f in walk_myc(Path::new(dir)).map_err(|e| {
            eprintln!("myc-lint: {e}");
            ExitCode::from(66)
        })? {
            let src = std::fs::read_to_string(&f).map_err(|e| {
                eprintln!("myc-lint: io-error: {}: {e}", f.display());
                ExitCode::from(66)
            })?;
            let rel = f
                .strip_prefix(dir)
                .unwrap_or(&f)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, src));
        }
        return Ok(out);
    }

    if paths.is_empty() {
        // Default: walk the current directory. A file that cannot be read here is skipped (best-effort
        // over an ambient tree) — unchanged from the original, distinct from the explicit-path case.
        for f in walk_myc(Path::new(".")).map_err(|e| {
            eprintln!("myc-lint: {e}");
            ExitCode::from(66)
        })? {
            if let Ok(src) = std::fs::read_to_string(&f) {
                out.push((f.to_string_lossy().into_owned(), src));
            }
        }
        return Ok(out);
    }

    for p in paths {
        // `read_source` prints the same `myc-lint: io-error: …` line and treats `-` as stdin; a refusal
        // maps to exit 66 (EX_IOERR) here. Stdin keeps its `<stdin>` display name (read_source does not
        // name the source — the caller owns the label, as before).
        let src = read_source("myc-lint: io-error", p).map_err(|_| ExitCode::from(66))?;
        let name = if p == "-" {
            "<stdin>".to_owned()
        } else {
            p.clone()
        };
        out.push((name, src));
    }
    Ok(out)
}
