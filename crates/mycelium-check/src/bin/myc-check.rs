//! `myc-check` — the correctness/type-check driver CLI (M-365; contract
//! `docs/spec/Myc-Check-Driver-Contract.md`). The prototype grown up: it keeps the single-file **oracle**
//! mode (the KC-2 LLM-harness contract — exit 2 parse / 3 check / `--expect-main`) and adds a **project**
//! mode that checks a whole `phylum`/program and aggregates diagnostics routed via the M-362 baseline.
//!
//! ```text
//! myc-check [--expect-main <ret-type>] <file.myc | ->          # oracle (single file)
//! myc-check --project <dir> | --config <mycelium-proj.toml> [--explain]   # whole project (CI gate)
//! myc-check --phylum <dir> [--json]                            # whole-phylum cross-nodule check (M-1006)
//! ```
//!
//! Exit codes: 0 ok · 2 parse error · 3 check error · 5 project-resolution error · 64 usage · 66 I/O.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mycelium_check::{
    check_phylum_dir, check_project, check_sources, FindingKind, PhylumReport, Report,
};
use mycelium_cli_common::{read_source, Args};
use mycelium_l1::ast::{Item, TypeRef};
use mycelium_l1::{check_nodule, parse};

fn usage() -> ExitCode {
    eprintln!(
        "usage: myc-check [--expect-main <ret-type>] <file.myc | ->\n       \
         myc-check --project <dir> | --config <mycelium-proj.toml> [--explain]\n       \
         myc-check --phylum <dir> [--json]     # whole-phylum cross-nodule check (M-1006)"
    );
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let mut expect_main: Option<String> = None;
    let mut project: Option<String> = None;
    let mut config: Option<String> = None;
    let mut phylum: Option<String> = None;
    let mut explain = false;
    let mut json = false;
    let mut path: Option<String> = None;

    let mut args = Args::from_env();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--expect-main" => match args.value() {
                Some(t) => expect_main = Some(t),
                None => return usage(),
            },
            "--project" => match args.value() {
                Some(p) => project = Some(p),
                None => return usage(),
            },
            "--config" => match args.value() {
                Some(p) => config = Some(p),
                None => return usage(),
            },
            "--phylum" => match args.value() {
                Some(p) => phylum = Some(p),
                None => return usage(),
            },
            "--explain" => explain = true,
            "--json" => json = true,
            _ if path.is_none() => path = Some(a),
            _ => return usage(),
        }
    }

    // Phylum mode (explicit) — the whole-phylum cross-nodule check (M-1006). Mutually exclusive with
    // the oracle/project inputs; combining is a usage error (never a silent precedence pick; G2).
    if let Some(dir) = phylum.as_deref() {
        if path.is_some() || expect_main.is_some() || project.is_some() || config.is_some() {
            return usage();
        }
        return run_phylum(Path::new(dir), json);
    }
    // `--json` is only meaningful in --phylum mode — never silently ignored elsewhere (G2).
    if json {
        return usage();
    }

    // Project mode (explicit) — the whole-phylum CI gate.
    if project.is_some() || config.is_some() {
        if path.is_some() || expect_main.is_some() {
            return usage(); // project mode does not take a file/--expect-main
        }
        let dir = project
            .map(PathBuf::from)
            .or_else(|| {
                config.as_deref().map(|c| {
                    Path::new(c)
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
                })
            })
            .unwrap_or_else(|| PathBuf::from("."));
        return run_project(&dir, explain);
    }

    // Oracle mode (single file) — back-compatible with the prototype's exact contract.
    let Some(path) = path else { return usage() };
    run_oracle(&path, expect_main.as_deref())
}

/// Project mode: check every `.myc` under `dir`, aggregate, exit non-zero on any error (CI gate).
fn run_project(dir: &Path, explain: bool) -> ExitCode {
    match check_project(dir) {
        Ok(report) => {
            print_report(&report, explain);
            ExitCode::from(report.exit_code())
        }
        Err(e) => {
            eprintln!("myc-check: {e}");
            ExitCode::from(5)
        }
    }
}

/// Phylum mode (M-1006): check every `.myc` under `dir` **as one cross-resolving phylum**, exit
/// non-zero on any refusal. Emits a stable one-line `--json` object (for the transpiler vet loop) or
/// a human summary.
fn run_phylum(dir: &Path, json: bool) -> ExitCode {
    match check_phylum_dir(dir) {
        Ok(report) => {
            if json {
                println!("{}", phylum_report_json(&report));
            } else {
                print_phylum_report(&report);
            }
            ExitCode::from(report.exit_code())
        }
        Err(e) => {
            eprintln!("myc-check: {e}");
            ExitCode::from(5)
        }
    }
}

/// The stable `--phylum --json` contract: ONE line, one JSON object. `error` is `null` on success or
/// `{"kind":"parse|duplicate|check","site":..,"message":..}`; `nodules` holds one
/// `{"nodule":..,"class":"Clean"}` per nodule (only when clean — never a fabricated verdict, VR-5).
fn phylum_report_json(report: &PhylumReport) -> String {
    let error = match &report.error {
        None => "null".to_owned(),
        Some(e) => format!(
            "{{\"kind\":\"{}\",\"site\":\"{}\",\"message\":\"{}\"}}",
            e.kind.as_str(),
            json_escape(&e.site),
            json_escape(&e.message),
        ),
    };
    let nodules: Vec<String> = report
        .nodules
        .iter()
        .map(|n| {
            format!(
                "{{\"nodule\":\"{}\",\"class\":\"{}\"}}",
                json_escape(&n.nodule),
                json_escape(n.class),
            )
        })
        .collect();
    format!(
        "{{\"mode\":\"phylum\",\"ok\":{},\"files_checked\":{},\"error\":{},\"nodules\":[{}]}}",
        report.ok,
        report.files_checked,
        error,
        nodules.join(","),
    )
}

/// Minimal JSON string escaping (no serde dep): quote `"`/`\`, the common control escapes, and any
/// remaining control char as `\uXXXX` — so the `--json` line is always one well-formed line.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The human summary for `--phylum` (non-`--json`): the single refusal line if any, else the clean
/// line. Never a silent empty pass (G2).
fn print_phylum_report(report: &PhylumReport) {
    match &report.error {
        Some(e) => {
            let at = if e.site.is_empty() {
                String::new()
            } else {
                format!(" at `{}`", e.site)
            };
            eprintln!("myc-check: {}-error{at}: {}", e.kind.as_str(), e.message);
        }
        None => println!(
            "ok: {} nodule(s) checked as one phylum, no findings",
            report.nodules.len()
        ),
    }
}

fn print_report(report: &Report, explain: bool) {
    for f in &report.findings {
        let kind = match f.kind {
            FindingKind::Parse => "parse-error",
            FindingKind::Check => "check-error",
        };
        let at = if f.site.is_empty() {
            String::new()
        } else {
            format!(" in `{}`", f.site)
        };
        if explain && f.kind == FindingKind::Check {
            println!(
                "{}: {kind}{at} [level={:?} route={}]: {}",
                f.file,
                f.level,
                f.route.as_deref().unwrap_or("-"),
                f.message
            );
        } else {
            println!("{}: {kind}{at}: {}", f.file, f.message);
        }
    }
    if report.is_ok() {
        println!("ok: {} file(s) checked, no findings", report.files_checked);
    } else {
        eprintln!(
            "myc-check: {} finding(s) across {} file(s)",
            report.findings.len(),
            report.files_checked
        );
    }
}

/// Oracle mode: the prototype's exact behavior (M-002/KC-2 harness contract). A single file (or `-`),
/// optional `--expect-main`, machine-readable first line, exit 2 (parse) / 3 (check) / 0 (ok).
fn run_oracle(path: &str, expect_main: Option<&str>) -> ExitCode {
    // `read_source` prints the same `io-error: …` line (no tool-name tag, as the prototype oracle did);
    // a refusal maps to exit 66 (EX_IOERR) here, preserving the harness contract.
    let src = match read_source("io-error", path) {
        Ok(s) => s,
        Err(_) => return ExitCode::from(66),
    };

    let nodule = match parse(&src) {
        Ok(c) => c,
        Err(e) => {
            println!("parse-error: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = check_nodule(&nodule) {
        println!("check-error: {e}");
        return ExitCode::from(3);
    }
    if let Some(expected) = expect_main {
        let found = nodule.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.sig.name == "main" => Some(f),
            _ => None,
        });
        let Some(f) = found else {
            println!(
                "check-error: no `fn main` declared (task requires `fn main() -> {expected}`)"
            );
            return ExitCode::from(3);
        };
        if !f.sig.value_params.is_empty() {
            println!(
                "check-error: `main` must be nullary, has {} parameter(s)",
                f.sig.value_params.len()
            );
            return ExitCode::from(3);
        }
        let got = render_type(&f.sig.ret);
        if got != expected {
            println!("check-error: `main` returns {got}, task requires {expected}");
            return ExitCode::from(3);
        }
    }
    // Keep the driver library exercised on the same input (parity), then emit the oracle's `ok`.
    let _ = check_sources(&[(path.to_owned(), src)]);
    println!("ok");
    ExitCode::SUCCESS
}

/// Render a declared return type the way the surface writes it (for `--expect-main`). Ported verbatim
/// from the prototype oracle so the harness contract is unchanged.
fn render_type(t: &TypeRef) -> String {
    use mycelium_l1::ast::{BaseType, Scalar, Sparsity, Strength};
    let base = match &t.base {
        BaseType::Binary(n) => format!("Binary{{{n}}}"),
        BaseType::Ternary(m) => format!("Ternary{{{m}}}"),
        // v0 tuple type (M-826): `(T, U, …)`.
        BaseType::Tuple(elems) => {
            let inner: Vec<String> = elems.iter().map(render_type).collect();
            format!("({})", inner.join(", "))
        }
        BaseType::Dense(d, s) => format!(
            "Dense{{{d}, {}}}",
            match s {
                Scalar::F16 => "F16",
                Scalar::Bf16 => "BF16",
                Scalar::F32 => "F32",
                Scalar::F64 => "F64",
            }
        ),
        BaseType::Vsa {
            model,
            dim,
            sparsity,
        } => match sparsity {
            Sparsity::Dense => format!("VSA{{{model}, {dim}, Dense}}"),
            Sparsity::Sparse(k) => format!("VSA{{{model}, {dim}, Sparse{{{k}}}}}"),
        },
        BaseType::Substrate(tag) => format!("Substrate{{{tag}}}"),
        // RFC-0032 D3/D4 (M-749/M-750): `Seq{T, N}` / nullary `Bytes`.
        BaseType::Seq { elem, len } => format!("Seq{{{}, {len}}}", render_type(elem)),
        BaseType::Bytes => "Bytes".to_owned(),
        // ADR-040 (M-897): the nullary scalar-float repr keyword (binary64 only — FLAG-1).
        BaseType::Float => "Float".to_owned(),
        BaseType::Named(n, args) => {
            if args.is_empty() {
                n.clone()
            } else {
                let inner: Vec<String> = args.iter().map(render_type).collect();
                format!("{n}<{}>", inner.join(", "))
            }
        }
        BaseType::Ambient(params) => match params {
            mycelium_l1::ast::AmbientParams::Size(n) => format!("{{{n}}}"),
            mycelium_l1::ast::AmbientParams::Dense(d, s) => format!(
                "{{{d}, {}}}",
                match s {
                    Scalar::F16 => "F16",
                    Scalar::Bf16 => "BF16",
                    Scalar::F32 => "F32",
                    Scalar::F64 => "F64",
                }
            ),
            mycelium_l1::ast::AmbientParams::Vsa {
                model,
                dim,
                sparsity,
            } => match sparsity {
                Sparsity::Dense => format!("{{{model}, {dim}, Dense}}"),
                Sparsity::Sparse(k) => format!("{{{model}, {dim}, Sparse{{{k}}}}}"),
            },
        },
        // RFC-0024 §3: function type `A -> B` (right-associative). Parenthesize a function-typed
        // LHS so `(A -> B) -> C` is unambiguous, not `A -> B -> C` (Copilot #397).
        BaseType::Fn(a, b) => {
            let lhs = render_type(a);
            let lhs = if matches!(a.base, BaseType::Fn(..)) {
                format!("({lhs})")
            } else {
                lhs
            };
            format!("{lhs} -> {}", render_type(b))
        }
    };
    match t.guarantee {
        None => base,
        Some(Strength::Exact) => format!("{base} @ Exact"),
        Some(Strength::Proven) => format!("{base} @ Proven"),
        Some(Strength::Empirical) => format!("{base} @ Empirical"),
        Some(Strength::Declared) => format!("{base} @ Declared"),
    }
}
