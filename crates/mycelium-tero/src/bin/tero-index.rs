//! `tero-index` — regenerate the committed Layer-1 corpus index (`docs/tero-index/{INDEX.md,
//! index.json}`) from the repo corpus (M-1015 / DN-87). Deterministic: two runs at one commit
//! produce byte-identical output (the drift-gate contract, checked by `scripts/checks/tero-index.sh`).
//!
//! Usage:
//! ```text
//!   tero-index [--repo-root .] [--out docs/tero-index]
//! ```
//!
//! Exit codes (mirroring the repo toolchain): `0` ok · `64` usage · `66` I/O. Never-silent (G2):
//! every failure is an explicit message with remediation, never a panic.

use std::path::PathBuf;
use std::process::ExitCode;

use mycelium_tero::{build_tero_index, write_json, write_markdown};

const EX_OK: u8 = 0;
const EX_USAGE: u8 = 64;
const EX_IO: u8 = 66;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => ExitCode::from(code),
        Err((code, msg)) => {
            eprintln!("tero-index: {msg}");
            ExitCode::from(code)
        }
    }
}

fn run(args: &[String]) -> Result<u8, (u8, String)> {
    let mut repo_root = PathBuf::from(".");
    let mut out = PathBuf::from("docs/tero-index");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--repo-root" => {
                i += 1;
                repo_root = PathBuf::from(args.get(i).ok_or((EX_USAGE, usage()))?);
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(args.get(i).ok_or((EX_USAGE, usage()))?);
            }
            "-h" | "--help" => return Ok(print_usage()),
            other => return Err((EX_USAGE, format!("unknown argument: {other}\n{}", usage()))),
        }
        i += 1;
    }

    let report =
        build_tero_index(&repo_root).map_err(|e| (EX_IO, format!("tero-index build: {e}")))?;
    write_json(&report, &out).map_err(|e| (EX_IO, format!("tero-index (json): {e}")))?;
    write_markdown(&report, &out).map_err(|e| (EX_IO, format!("tero-index (markdown): {e}")))?;
    println!(
        ">> tero-index: {} row(s) indexed, {} flagged (Empirical/Declared heuristic — source is \
         ground truth) → {}/{{INDEX.md,index.json}}",
        report.items.len(),
        report.flagged.len(),
        out.display()
    );
    Ok(EX_OK)
}

fn print_usage() -> u8 {
    println!("{}", usage());
    EX_OK
}

fn usage() -> String {
    "usage: tero-index [--repo-root <dir>] [--out <dir>]".to_owned()
}
