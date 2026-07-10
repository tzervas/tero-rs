//! `mycfmt` — the canonical formatter CLI (M-364; contract `docs/spec/Mycfmt-Formatter-Contract.md`).
//!
//! Formatting is an **identity-preserving projection** (RFC-0001 §4.6/§4.8; ADR-003): it never changes a
//! definition's content-addressed identity, and it **never writes a partial or garbled rewrite** (G2) —
//! any refusal leaves the file exactly as it was.
//!
//! ```text
//! mycfmt [--check | --write] [--flatten | --readable [--expand-spine]] [--explain] [--config <mycelium-proj.toml>] <file.myc | ->...
//! ```
//!
//! Exit codes (contract §5): 0 ok · 1 `--check` would reformat · 2 parse error · 3 header error ·
//! 4 out-of-scope refusal (incl. a `[toolchain].format` pin mismatch) · 64 usage · 66 I/O.
//!
//! **`--flatten`** emits the single-line human↔stream form (M-819; DN-57 §2): the whole nodule on one
//! line, components separated by `; `.  The mandatory `;` terminator (M-818) makes this unambiguous.
//! Comments and structured-header metadata are stripped (not part of the surface AST).  The output
//! re-parses to the same surface AST as the canonical form (`Empirical` round-trip guarantee).
//! `--flatten` is incompatible with `--write` (the stream form is for stdout / pipe use).
//!
//! **`--readable`** emits the human-readable multi-line form (M-974; DN-82) — the inverse posture of
//! `--flatten`: long argument / field / variant / arm segments break across lines with line breaks
//! after commas; short segments stay inline. It preserves comments and the structured header (like the
//! default form) and is **presentation-only, functionally inert** (same surface AST — C1/C2). It is the
//! canonical form the `myc-fmt` gate enforces for the human-authored stdlib (`lib/std/*.myc`).
//! `--readable` and `--flatten` are mutually exclusive (opposite layout postures).
//!
//! **`--expand-spine`** (Shape-Dispatched Readable house-style knob, M-976; requires `--readable`)
//! selects the "expanded" spine style: a right-nested same-head chain (Cons/GLCons/bool_and/cat …)
//! STILL renders as a flat spine (each link at one indent, no pyramid), but every inner nested call
//! is broken onto its own lines (block-indented) instead of staying inline-when-it-fits. Both are
//! behavior-neutral (C1/C2); the default (omit the flag) is the compact `InlineWhenFits` style.

use std::process::ExitCode;

use mycelium_cli_common::{read_source, Args};
use mycelium_fmt::{
    flatten_source, format_source, format_source_readable, format_source_readable_cfg, Formatted,
    LayoutCfg, SpineInner,
};
use mycelium_proj::parse_manifest;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Print the formatted source to stdout (default).
    Stdout,
    /// Report which files would change; write nothing; exit 1 if any differ.
    Check,
    /// Rewrite the file in place (atomically), only after a successful identity-preserving format.
    Write,
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: mycfmt [--check | --write] [--flatten | --readable [--expand-spine]] [--explain] [--config <mycelium-proj.toml>] <file.myc | ->..."
    );
    ExitCode::from(64) // EX_USAGE
}

fn main() -> ExitCode {
    let mut mode = Mode::Stdout;
    let mut flatten = false;
    let mut readable = false;
    let mut expand_spine = false;
    let mut explain = false;
    let mut config: Option<String> = None;
    let mut paths: Vec<String> = Vec::new();

    let mut args = Args::from_env();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--check" => mode = Mode::Check,
            "--write" => mode = Mode::Write,
            "--flatten" => flatten = true,
            "--readable" => readable = true,
            "--expand-spine" => expand_spine = true,
            "--explain" => explain = true,
            "--config" => match args.value() {
                Some(p) => config = Some(p),
                None => return usage(),
            },
            "-" => paths.push("-".to_owned()),
            s if s.starts_with("--") => return usage(),
            s => paths.push(s.to_owned()),
        }
    }
    if paths.is_empty() {
        return usage();
    }
    if mode == Mode::Write && paths.iter().any(|p| p == "-") {
        eprintln!("mycfmt: --write cannot rewrite stdin; use the default (stdout) for `-`");
        return usage();
    }
    // --flatten is for stdout / pipe use; --write with --flatten is rejected (G2: never silent).
    if flatten && mode == Mode::Write {
        eprintln!(
            "mycfmt: --flatten cannot be used with --write; the stream form is stdout-only \
             (pipe it to a file if needed)"
        );
        return usage();
    }
    // --readable and --flatten are opposite layout postures; requesting both is a usage error (G2).
    if flatten && readable {
        eprintln!(
            "mycfmt: --flatten and --readable are mutually exclusive (opposite layout postures) — \
             pick one"
        );
        return usage();
    }
    // --expand-spine is the Readable house-style knob (M-976); it only makes sense with --readable.
    if expand_spine && !readable {
        eprintln!(
            "mycfmt: --expand-spine is a --readable house-style knob (M-976) — it requires --readable"
        );
        return usage();
    }

    // The `[toolchain].format` hard pin (M-364 §10.3), if a manifest is given/discoverable. A manifest
    // that does not parse is an explicit error — we never silently format ignoring a malformed pin (G2).
    let pin = match resolve_pin(config.as_deref(), &paths) {
        Ok(p) => p,
        Err(code) => return code,
    };

    let mut worst = 0u8; // highest exit code seen
    for path in &paths {
        let code = run_one(
            path,
            mode,
            flatten,
            readable,
            expand_spine,
            explain,
            pin.as_deref(),
        );
        worst = worst.max(code);
    }
    ExitCode::from(worst)
}

/// Resolve the `[toolchain].format` pin from an explicit `--config`, else by discovering
/// `mycelium-proj.toml` upward from the first real file path. Returns `Ok(None)` when there is no
/// manifest (the built-in default applies). A malformed manifest is a hard error (exit 66/4).
fn resolve_pin(config: Option<&str>, paths: &[String]) -> Result<Option<String>, ExitCode> {
    let manifest_path = if let Some(c) = config {
        Some(std::path::PathBuf::from(c))
    } else {
        paths
            .iter()
            .find(|p| *p != "-")
            .and_then(|p| discover_manifest(std::path::Path::new(p)))
    };
    let Some(mp) = manifest_path else {
        return Ok(None);
    };
    let text = match std::fs::read_to_string(&mp) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("mycfmt: io-error: {}: {e}", mp.display());
            return Err(ExitCode::from(66));
        }
    };
    match parse_manifest(&text) {
        Ok(m) => Ok(m.toolchain.and_then(|t| t.format)),
        Err(e) => {
            eprintln!("mycfmt: manifest-error: {}: {e}", mp.display());
            Err(ExitCode::from(4))
        }
    }
}

/// Walk up from `start`'s directory looking for `mycelium-proj.toml`.
fn discover_manifest(start: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut dir = start
        .parent()
        .map(std::path::Path::to_path_buf)
        .or_else(|| std::env::current_dir().ok())?;
    loop {
        let candidate = dir.join("mycelium-proj.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Format one path; return its exit code (contract §5).
fn run_one(
    path: &str,
    mode: Mode,
    flatten: bool,
    readable: bool,
    expand_spine: bool,
    explain: bool,
    pin: Option<&str>,
) -> u8 {
    // `read_source` prints the same `mycfmt: io-error: …` line the local copy did; a refusal maps to
    // the contract's I/O exit code 66 (EX_IOERR) here, where the exit-code newtype lives.
    let src = match read_source("mycfmt: io-error", path) {
        Ok(s) => s,
        Err(_) => return 66,
    };

    let result = if flatten {
        flatten_source(&src, pin)
    } else if readable && expand_spine {
        // The Readable "expanded" house style (M-976): flat spine, inner nested calls block-expanded.
        let cfg = LayoutCfg {
            spine_inner: SpineInner::AlwaysExpand,
            ..LayoutCfg::default()
        };
        format_source_readable_cfg(&src, pin, cfg)
    } else if readable {
        format_source_readable(&src, pin)
    } else {
        format_source(&src, pin)
    };

    match result {
        Ok(formatted) => emit(path, &src, &formatted, mode, explain),
        Err(e) => {
            eprintln!("mycfmt: {path}: {e}");
            e.exit_code()
        }
    }
}

fn emit(path: &str, src: &str, f: &Formatted, mode: Mode, explain: bool) -> u8 {
    if explain {
        eprintln!("mycfmt: {path}");
        for n in &f.notes {
            eprintln!("  - {n}");
        }
        eprintln!("  (identity preserved: surface AST round-trip verified)");
    }
    match mode {
        Mode::Stdout => {
            print!("{}", f.output);
            0
        }
        Mode::Check => {
            if f.changed {
                eprintln!("mycfmt: {path}: would reformat");
                1
            } else {
                0
            }
        }
        Mode::Write => {
            if !f.changed {
                return 0;
            }
            match write_atomic(path, &f.output) {
                Ok(()) => {
                    eprintln!("mycfmt: {path}: formatted");
                    let _ = src; // (kept for symmetry; the on-disk file is replaced atomically)
                    0
                }
                Err(e) => {
                    eprintln!("mycfmt: io-error: {path}: {e}");
                    66
                }
            }
        }
    }
}

/// Write `content` to `path` atomically (temp file in the same dir, then rename) so a crash mid-write can
/// never leave a partial file — the never-silent rule extended to the filesystem (G2).
fn write_atomic(path: &str, content: &str) -> std::io::Result<()> {
    let p = std::path::Path::new(path);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty());
    let tmp = match dir {
        Some(d) => d.join(format!(
            ".{}.mycfmt.tmp",
            p.file_name().and_then(|s| s.to_str()).unwrap_or("out")
        )),
        None => std::path::PathBuf::from(format!(
            ".{}.mycfmt.tmp",
            p.file_name().and_then(|s| s.to_str()).unwrap_or("out")
        )),
    };
    std::fs::write(&tmp, content)?;
    std::fs::rename(&tmp, p)
}
