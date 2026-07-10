//! `myc` — the one-command Mycelium toolchain driver (M-733).
//!
//! ```text
//! myc init  <name>                 # scaffold a new phylum
//! myc build [--config <manifest>]  # build the content-addressed spore
//! myc check [--config <manifest>]  # parse + type-check every .myc source
//! myc test  [--config <manifest>]  # run the available verification (check)
//! myc run   [--config <manifest>]  # run a project's `main` (M-908 v0 single-nodule; M-909 multi-nodule)
//! myc --stream [<file>]            # parse a `;`-terminated component stream (M-820/DN-57)
//! ```
//!
//! Every failure is a DN-22 structured [`Report`](mycelium_cli::Report) — `error[<code>]: …` with a
//! source location and an actionable `help:` line; no raw panic ever reaches the user (G2).
//!
//! Exit codes: 0 ok · 2 manifest · 64 usage · 65 source/eval/nodule-link error · 66 I/O · 70 a
//! program outside the evaluation-complete fragment (RFC-0007).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mycelium_cli::{
    build, check_project, corpus_context, init, reject_unbounded_in_corpus, run_stream_parse,
    run_with_options, unbounded_banner, Report, RunOptions,
};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         myc init  <name>\n  \
         myc build [--config <manifest>] [--unbounded]\n  \
         myc check [--config <manifest>]\n  \
         myc test  [--config <manifest>]\n  \
         myc run   [--config <manifest>] [--unbounded]  # single- or multi-nodule (M-908/M-909)\n  \
         myc --stream [<file>]\n\
         \n  \
         --unbounded  opt-in, NON-DETERMINISTIC: lift the recursion-depth ceiling (RFC-0041 §5).\n               \
         Machine-dependent; excluded from the conformance corpus; never for CI/reproducible builds."
    );
    ExitCode::from(64)
}

/// Print a [`Report`] to stderr and return its exit code.
fn fail(r: &Report) -> ExitCode {
    eprintln!("{}", r.render());
    ExitCode::from(r.exit)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        return usage();
    };
    let rest: Vec<String> = args.collect();

    match cmd.as_str() {
        "init" => match rest.as_slice() {
            [name] => match init(Path::new("."), name) {
                Ok(files) => {
                    println!("created {} file(s):", files.len());
                    for f in files {
                        println!("  {}", f.display());
                    }
                    println!("next: cd {name} && myc check");
                    ExitCode::SUCCESS
                }
                Err(r) => fail(&r),
            },
            _ => usage(),
        },
        // `build` and `run` accept the RFC-0041 §5 `--unbounded` escape hatch; `check`/`test` do not.
        "build" => with_run_options(&rest, "build", |m, _opts| cmd_build(m)),
        "check" => with_manifest(&rest, cmd_check),
        "test" => with_manifest(&rest, cmd_test),
        "run" => with_run_options(&rest, "run", |m, opts| match run_with_options(m, opts) {
            Ok(report) => {
                println!("{}", report.rendered);
                eprintln!("myc: ran `{}` in {}", report.entry, report.source);
                ExitCode::SUCCESS
            }
            Err(r) => fail(&r),
        }),
        "--stream" => cmd_stream(&rest),
        _ => usage(),
    }
}

/// Resolve the `--config <manifest>` flag (default `mycelium-proj.toml`) and dispatch. Used by
/// `check`/`test`, which do **not** accept `--unbounded` (RFC-0041 §5: CLI-flag-only, and only on the
/// execution/build drivers) — so an `--unbounded` here is an unknown flag (usage error).
fn with_manifest(rest: &[String], f: impl FnOnce(&Path) -> ExitCode) -> ExitCode {
    let mut manifest = PathBuf::from("mycelium-proj.toml");
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--config" => match it.next() {
                Some(p) => manifest = PathBuf::from(p),
                None => return usage(),
            },
            _ => return usage(),
        }
    }
    f(&manifest)
}

/// Resolve `--config <manifest>` **and** the RFC-0041 §5 `--unbounded` flag for `run`/`build`, then
/// dispatch. When `--unbounded` is engaged: (1) if a conformance-corpus / CI context is signalled
/// ([`corpus_context`]), it is **refused** never-silently ([`reject_unbounded_in_corpus`]) — the
/// deterministic corpus path must not run the machine-dependent mode; (2) otherwise a never-silent
/// banner ([`unbounded_banner`]) is printed to stderr before dispatch (G2). `cmd` (`"run"`/`"build"`)
/// tailors the banner's per-command effect line.
fn with_run_options(
    rest: &[String],
    cmd: &str,
    f: impl FnOnce(&Path, &RunOptions) -> ExitCode,
) -> ExitCode {
    let mut manifest = PathBuf::from("mycelium-proj.toml");
    let mut opts = RunOptions::default();
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--config" => match it.next() {
                Some(p) => manifest = PathBuf::from(p),
                None => return usage(),
            },
            "--unbounded" => opts.unbounded = true,
            _ => return usage(),
        }
    }
    if opts.unbounded {
        // Deterministic corpus/CI runs refuse the machine-dependent mode (never a silent downgrade).
        if corpus_context() {
            if let Err(r) = reject_unbounded_in_corpus(&opts) {
                return fail(&r);
            }
        }
        // Never-silent: announce the escape hatch on stderr before doing anything.
        eprintln!("{}", unbounded_banner(cmd));
    }
    f(&manifest, &opts)
}

fn cmd_build(manifest: &Path) -> ExitCode {
    match build(manifest) {
        Ok((spore, descriptor)) => {
            print!("{descriptor}");
            eprintln!("myc: built {} ({})", spore.name, spore.id.as_str());
            ExitCode::SUCCESS
        }
        Err(r) => fail(&r),
    }
}

fn cmd_check(manifest: &Path) -> ExitCode {
    match check_project(manifest) {
        Ok(report) => {
            for r in &report.failures {
                eprintln!("{}\n", r.render());
            }
            if report.ok() {
                eprintln!("myc: {} nodule(s) checked clean", report.checked.len());
                ExitCode::SUCCESS
            } else {
                eprintln!(
                    "myc: {} checked, {} failed",
                    report.checked.len(),
                    report.failures.len()
                );
                ExitCode::from(65)
            }
        }
        Err(r) => fail(&r),
    }
}

fn cmd_test(manifest: &Path) -> ExitCode {
    // `test` runs the available verification (type-check). Honest (VR-5): a dedicated `.myc`
    // unit-test runner does not exist yet — this does not pretend to have run user-authored tests.
    let code = cmd_check(manifest);
    eprintln!(
        "myc: note — `test` ran the type-check verification; a dedicated .myc unit-test runner is \
         future work (no user tests were discovered or executed)."
    );
    code
}

/// `myc --stream [<file>]` — parse a `;`-terminated Mycelium component stream (M-820 / DN-57).
///
/// Without a file argument, reads from stdin (`<stdin>`). With a file argument, opens and reads
/// that file. The source is lexed once and the token stream is segmented at `nodule` header tokens
/// into per-nodule components, each parsed with `mycelium_l1::parse`. The split is token-driven,
/// so it is comment-/string-safe by construction (a `nodule`/`;` inside a comment is never a token;
/// DN-57 §2). v0 I/O is whole-input-buffered (`Declared` — see [`mycelium_cli::stream_parse`]).
///
/// Every malformed component surfaces an explicit error with a component:line:col location (G2).
/// An unterminated component (its last item has no `;` before the next `nodule`/EOF) is likewise an
/// explicit error, never a silent partial accept (G2 / DN-57 §3.1).
///
/// Exit 0 on all-green; exit 65 if any component failed (or on lex error); exit 66 on I/O error.
fn cmd_stream(rest: &[String]) -> ExitCode {
    // Parse the optional file argument; reject anything else (unknown flags) as usage.
    let (reader, source_name): (Box<dyn std::io::Read>, String) = match rest {
        [] => (Box::new(std::io::stdin()), "<stdin>".to_owned()),
        [path] if !path.starts_with('-') => match std::fs::File::open(path) {
            Ok(f) => (Box::new(f), path.clone()),
            Err(e) => {
                let r = Report::new("myc-stream-io", format!("{path}: {e}"), 66)
                    .help("check that the file path is correct and the file is readable");
                return fail(&r);
            }
        },
        _ => return usage(),
    };

    match run_stream_parse(reader, &source_name) {
        Err(r) => fail(&r),
        Ok(report) => {
            // Print any failures to stderr, each as a structured DN-22 report.
            for f in &report.failures {
                eprintln!("{}\n", f.render());
            }
            if report.ok() {
                eprintln!(
                    "myc: stream `{}` — {} component(s) parsed clean",
                    report.source_name, report.parsed_ok,
                );
                ExitCode::SUCCESS
            } else {
                eprintln!(
                    "myc: stream `{}` — {} ok, {} failed",
                    report.source_name, report.parsed_ok, report.parsed_err,
                );
                ExitCode::from(65)
            }
        }
    }
}
