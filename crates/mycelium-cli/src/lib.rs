//! `mycelium-cli` — the **`myc`** one-command toolchain driver (M-733; E16-1).
//!
//! A single front door over the Mycelium toolchain: `myc init` scaffolds a phylum, `myc build`
//! packages it (the content-addressed spore — M-368), `myc check` type-checks it (parse + check via
//! the L1 front-end), `myc test` runs the available verification, `myc run` executes a project
//! (single- or **multi-nodule**, M-908/M-909) through the reference interpreter, and `myc --stream`
//! parses a `;`-delimited component stream from stdin or a file (M-820 / DN-57).
//!
//! ## Error-message quality bar (DN-22 / RFC-0013)
//! Every user-visible failure is a structured [`Report`]: a stable `code`, a human-readable
//! `message`, an optional source `location`, and an actionable `help`. No raw Rust panic ever
//! reaches the user (G2 — never opaque); a failure the driver cannot honestly act on is reported as
//! such, never swallowed and never faked (VR-5).
//!
//! ## Honesty about scope (`Declared`)
//! `init` / `build` / `check` do real end-to-end work. `test` runs `check` and is explicit that a
//! dedicated `.myc` unit-test *runner* does not exist yet (it does not pretend to have run tests
//! that were never written). `run` executes a project's `.myc` sources: a **single** source follows
//! the M-908 v0 path (parse → [`check_nodule`] → [`elaborate`](mycelium_l1::elaborate) its nullary
//! `main`); **two or more** sources follow the M-909 multi-nodule path (see [`run`]'s doc for the
//! full linking model). A missing `main`, a program outside the evaluation-complete fragment, or an
//! interpreter failure are each an explicit [`Report`] — never a silent narrowing to "the first file
//! found" and never a stub that pretends to have run (G2/VR-5). `--stream` is a **token-driven**
//! component splitter: it lexes the source once ([`mycelium_l1::lexer::lex`]), segments the token
//! stream at `nodule` header tokens (`;` as `Tok::Semi` is the per-item terminator — DN-57), and
//! parse each component slice with [`mycelium_l1::parse`]. Splitting on *tokens* (not raw text) makes
//! it comment-/string-safe by construction: a `nodule`/`;` inside a `//` comment is never a token, so
//! it can never mis-split (DN-57 §2). The per-component parse bounds parse state to one component at
//! a time. **v0 I/O is whole-input-buffered** (`Declared`); true per-`;`-component incremental I/O
//! would require a resumable L1 token-stream API that does not exist yet (flagged future work).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as StdRead;
use std::path::{Path, PathBuf};

use mycelium_l1::ast::{Item, Path as NoduleAstPath};
use mycelium_l1::lexer::lex;
use mycelium_l1::token::{Pos, Spanned, Tok};
use mycelium_l1::{
    check_nodule, check_phylum, elaborate, parse, CheckError, Env, Nodule, ParseError, Phylum,
    PhylumEnv, UsePath,
};
use mycelium_proj::parse_manifest;
use mycelium_spore::{build_spore, explain, Spore};

/// A structured, actionable diagnostic (the DN-22 quality bar; a projection of an RFC-0013
/// diagnostic). It renders as `error[<code>]: <message>` with optional `--> <location>` and
/// `help:` lines — never an opaque internal error (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// A stable, machine-readable diagnostic code (e.g. `myc-parse`, `myc-build`, `myc-run-unwired`).
    pub code: &'static str,
    /// The human-readable, specific message.
    pub message: String,
    /// An optional `path:line:col` (or `path`) the user can jump to.
    pub location: Option<String>,
    /// An optional actionable next step.
    pub help: Option<String>,
    /// The process exit code this report maps to (sysexits-flavoured; never 0).
    pub exit: u8,
}

impl Report {
    /// A report with a code, message and exit code (no location/help).
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>, exit: u8) -> Self {
        Report {
            code,
            message: message.into(),
            location: None,
            help: None,
            exit,
        }
    }

    /// Attach a `path:line:col` (or `path`) location.
    #[must_use]
    pub fn at(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Attach an actionable `help:` line.
    #[must_use]
    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Render the multi-line, structured form (no trailing newline).
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = format!("error[{}]: {}", self.code, self.message);
        if let Some(loc) = &self.location {
            s.push_str(&format!("\n  --> {loc}"));
        }
        if let Some(help) = &self.help {
            s.push_str(&format!("\n  help: {help}"));
        }
        s
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::error::Error for Report {}

/// `myc init <name>` — scaffold a new phylum named `name` under `parent`, returning the created
/// files. The name must be a simple lowercase identifier (`[a-z][a-z0-9_]*`); a dotted/empty/
/// mixed-case name is refused, never silently normalized (G2). An existing project at the target is
/// refused — `init` never overwrites (G2).
///
/// # Errors
/// [`Report`] (`myc-init-name` / `myc-init-exists` / `myc-io`) on a bad name, a pre-existing project,
/// or a filesystem failure.
pub fn init(parent: &Path, name: &str) -> Result<Vec<PathBuf>, Report> {
    validate_name(name)?;
    let dir = parent.join(name);
    let manifest_path = dir.join("mycelium-proj.toml");
    if manifest_path.exists() {
        return Err(Report::new(
            "myc-init-exists",
            format!("a project already exists at {}", manifest_path.display()),
            66,
        )
        .help(
            "choose a new name or remove the existing project — `myc init` never overwrites (G2)",
        ));
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| Report::new("myc-io", format!("{}: {e}", dir.display()), 66))?;

    let manifest = scaffold_manifest(name);
    let nodule = scaffold_nodule(name);
    let source_path = dir.join(format!("{name}.myc"));

    write_new(&manifest_path, &manifest)?;
    write_new(&source_path, &nodule)?;
    Ok(vec![manifest_path, source_path])
}

/// `myc build` — build the content-addressed spore for the project at `manifest_path`, returning the
/// built [`Spore`] and its descriptor text (M-368). A missing/ambiguous publish input is surfaced as
/// a structured [`Report`], never a partial artifact (G2).
///
/// # Errors
/// [`Report`] (`myc-io` / `myc-manifest` / `myc-build`) on a read failure, a malformed manifest, or a
/// refused build input.
pub fn build(manifest_path: &Path) -> Result<(Spore, String), Report> {
    let (manifest, project_dir) = load_manifest(manifest_path)?;
    let spore = build_spore(&manifest, &project_dir).map_err(|e| {
        Report::new("myc-build", e.to_string(), e.exit_code())
            .at(project_dir.display().to_string())
            .help("declare the [surface].exports, add a `.myc` source, or pin a dependency `hash` (ADR-003)")
    })?;
    // Compute the descriptor from a borrow, then move `spore` out by value (no clone).
    let descriptor = explain(&spore);
    Ok((spore, descriptor))
}

/// The outcome of [`check_project`]: which nodules type-checked, and the structured failures.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckReport {
    /// Source files that parsed and type-checked cleanly.
    pub checked: Vec<String>,
    /// Per-file structured failures (parse or type errors), each with a location (DN-22).
    pub failures: Vec<Report>,
}

impl CheckReport {
    /// Whether every checked file passed.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// `myc check` — parse and type-check every `.myc` source under the project directory containing
/// `manifest_path`. Each nodule is checked independently (per-nodule scope — honest `Declared`:
/// cross-nodule resolution is the elaborator's job, not re-implemented here). Returns a structured
/// [`CheckReport`]; a parse/type error becomes a located [`Report`] in `failures`, never a panic (G2).
///
/// # Errors
/// [`Report`] (`myc-io`) only when the source tree cannot be walked; per-file check failures are
/// carried in the returned [`CheckReport`], not as an `Err`.
pub fn check_project(manifest_path: &Path) -> Result<CheckReport, Report> {
    let (_, project_dir) = load_manifest(manifest_path)?;
    let sources =
        mycelium_cli_common::walk_myc(&project_dir).map_err(|e| Report::new("myc-io", e, 66))?;
    let mut report = CheckReport::default();
    for path in sources {
        let rel = path
            .strip_prefix(&project_dir)
            .unwrap_or(&path)
            .display()
            .to_string();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                report
                    .failures
                    .push(Report::new("myc-io", format!("{rel}: {e}"), 66).at(rel.clone()));
                continue;
            }
        };
        match parse(&text) {
            Err(pe) => report.failures.push(
                Report::new("myc-parse", pe.message.clone(), 65)
                    .at(format!("{rel}:{}:{}", pe.pos.line, pe.pos.col))
                    .help("fix the syntax error at the indicated position"),
            ),
            Ok(nodule) => match check_nodule(&nodule) {
                Err(ce) => report.failures.push(
                    Report::new("myc-check", ce.to_string(), 65)
                        .at(rel.clone())
                        .help("resolve the type error reported above"),
                ),
                Ok(_env) => report.checked.push(rel),
            },
        }
    }
    Ok(report)
}

/// The outcome of a successful `myc run` (M-908/M-909): which source ran, which entry function was
/// executed, and a rendering of the interpreter's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    /// The `.myc` source file that ran, relative to the project directory. For a multi-nodule
    /// project (M-909) this is the **entry nodule's** source file (the one declaring `main`) — the
    /// other linked nodules are named in [`Report`]s on failure, not in this success value.
    pub source: String,
    /// The entry function name that was executed (v0 convention: `main`).
    pub entry: String,
    /// A `{:?}`-rendered form of the interpreter's result value (`Declared` — a v0 debug rendering,
    /// not a stable/parseable format; a dedicated value-printer is follow-up work, not silently
    /// approximated here).
    pub rendered: String,
}

/// Options that tune a `myc run` (or `myc build`) invocation beyond `--config` (RFC-0041 §5 /
/// DN-84 §9.3). Additive: [`Default`] is the ordinary, deterministic, corpus-safe behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunOptions {
    /// The opt-in, **non-deterministic** `--unbounded` escape hatch (RFC-0041 §5, DN-84 §9.3 —
    /// design (C)): lift the deterministic recursion-depth ceiling for this invocation. It is
    /// **CLI-flag-only** (never a manifest/env/LSP-config knob), engaging it prints a never-silent
    /// banner ([`unbounded_banner`]), and it is **excluded from the conformance corpus** and refused
    /// under a corpus/CI run ([`reject_unbounded_in_corpus`]). Machine-dependent by construction, so
    /// never the default and never a reproducible-build input.
    pub unbounded: bool,
}

/// The reference interpreter to execute a `myc run` under, given [`RunOptions`]. Default: the
/// deterministic 4096-floor budget (unchanged). With `--unbounded`: the depth ceiling is lifted to
/// [`u32::MAX`] via [`mycelium_interp::Interpreter::with_depth`] (RFC-0041 §5) — refusal then bounds
/// on available memory/host stack, not the deterministic budget. Never-silent even so: the growable
/// deep stack keeps it an explicit refusal, never a `SIGABRT`.
fn interpreter_for(opts: &RunOptions) -> mycelium_interp::Interpreter {
    let interp = mycelium_interp::Interpreter::default();
    if opts.unbounded {
        interp.with_depth(u32::MAX)
    } else {
        interp
    }
}

/// The never-silent stderr banner printed when `--unbounded` is engaged (G2 — an explicit escape
/// hatch is announced, never silent). `cmd` is the subcommand (`"run"` / `"build"`) so the
/// per-command effect line is accurate: `run` actually lifts the interpreter ceiling; `build`
/// performs no interpreted evaluation, so the flag is accepted for interface parity but does not
/// alter its frontend passes (their depth ceilings are internal to `mycelium-l1` — a tracked
/// follow-on). The corpus/CI refusal ([`reject_unbounded_in_corpus`]) applies to both.
#[must_use]
pub fn unbounded_banner(cmd: &str) -> String {
    let mode = "myc: WARNING — `--unbounded` engaged: an opt-in, NON-DETERMINISTIC, \
                machine-dependent escape hatch (RFC-0041 §5 / DN-84 §9.3). It is excluded from the \
                conformance corpus and must never be used in CI or for a reproducible build.";
    let effect = match cmd {
        "run" => {
            "  effect: the interpreter's deterministic recursion-depth ceiling is DISABLED for this \
             run — a deep computation is now bounded only by available memory/host stack, not the \
             4096 depth budget (it still refuses never-silently, never a crash)."
        }
        _ => {
            "  effect: `myc build` performs no interpreted evaluation, so this flag does not alter \
             the build's frontend passes (their depth ceilings live in `mycelium-l1`, not the CLI — \
             a tracked follow-on). It is accepted for interface parity and still refused under a \
             corpus/CI run."
        }
    };
    format!("{mode}\n{effect}")
}

/// The conformance-corpus / CI guard (RFC-0041 §5): a corpus/CI run is the **deterministic** path, so
/// `--unbounded` (opt-in, machine-dependent) must be **refused** there — never silently downgraded to
/// a bounded run, never silently allowed to run non-deterministically (G2). A corpus runner (or the
/// CLI when a corpus/CI context is signalled — see [`corpus_context`]) calls this before executing.
///
/// # Errors
/// [`Report`] (`myc-unbounded-corpus`, exit 64) when `opts.unbounded` is set during a corpus/CI run.
pub fn reject_unbounded_in_corpus(opts: &RunOptions) -> Result<(), Report> {
    if opts.unbounded {
        return Err(Report::new(
            "myc-unbounded-corpus",
            "`--unbounded` is excluded from the conformance corpus and refused in CI: it is an \
             opt-in, non-deterministic, machine-dependent escape hatch (RFC-0041 §5 / DN-84 §9.3), \
             so a deterministic corpus/CI run must not use it",
            64,
        )
        .help(
            "drop `--unbounded` for corpus/CI runs — it is for interactive REPL/exploration only; \
             the corpus is the deterministic, reproducible path",
        ));
    }
    Ok(())
}

/// Whether the process is running under a **conformance-corpus / CI** context that must refuse
/// `--unbounded` (RFC-0041 §5). Signalled by the `MYC_CORPUS` environment variable being set to any
/// non-empty value — the corpus/CI job exports it so [`reject_unbounded_in_corpus`] fires. A pure,
/// side-effect-free read (never mutates the environment).
#[must_use]
pub fn corpus_context() -> bool {
    std::env::var_os("MYC_CORPUS").is_some_and(|v| !v.is_empty())
}

/// `myc run` — execute a project through the reference interpreter (M-908 v0 single-nodule;
/// M-909 multi-nodule).
///
/// The project directory (containing `manifest_path`) is walked for `.myc` sources
/// ([`mycelium_cli_common::walk_myc`]). **Zero** sources is refused (`myc-run-no-source`). **One**
/// source runs the M-908 v0 path directly. **Two or more** sources run the M-909 multi-nodule path:
///
/// 1. **Parse** every source independently (each file is a bare `nodule <path>; …` block — a
///    phylum-of-one in [`mycelium_l1`] terms).
/// 2. **Link-check** the parsed nodules before any type-checking, since [`check_phylum`] itself does
///    not guard against these (never-silent, G2 — each is a named, located [`Report`]):
///    - **duplicate** (`myc-run-nodule-duplicate`): two files declare the same `nodule <path>;`.
///    - **unresolved** (`myc-run-nodule-unresolved`): a `use <nodule>.<item>` (or `use <nodule>.*`)
///      names a nodule with no corresponding file in the project.
///    - **cyclic** (`myc-run-nodule-cyclic`): the nodule-level `use` dependency graph has a cycle. A
///      **v0 CLI policy choice** (`Declared`), not a fundamental limit of [`check_phylum`] (which
///      tolerates cyclic nodule refs at the type-check level via its two-pass export/coherence
///      build) — `myc run` v0 additionally requires the *link* graph to be acyclic, matching the
///      conservative "refuse rather than guess" posture used throughout this driver; this may be
///      lifted once a real project-scoped linker replaces the v0 flatten-by-name scheme below.
/// 3. Assemble the parsed nodules into one [`Phylum`] (no `phylum` header — `path: None`) and
///    [`check_phylum`] it, which enforces cross-nodule `pub`/`use` visibility and the phylum-wide
///    orphan rule (M-662). A check failure is `myc-check`.
/// 4. **Find the entry nodule**: exactly one of the checked nodules must declare a nullary `main` —
///    zero is `myc-run-no-entry`, more than one is `myc-run-entry-ambiguous` (never guesses which).
/// 5. **Link for elaboration**: [`check_phylum`]'s per-nodule [`Env`] only carries a nodule's own
///    declarations plus what it *directly* imports (RFC-0006 §4.3) — not the transitive closure a
///    call chain through an imported function may need (e.g. `main` imports `helper` from nodule
///    `B`, and `helper`'s body calls a second, *private* function of `B` that `main` never
///    imported). Since [`check_phylum`] has already validated every cross-nodule reference in the
///    program is legal, `myc run` v0 safely **flattens every checked nodule's `Env` into one merged
///    `Env`** (by simple name) purely for elaboration/execution — a v0 CLI-level linking policy, not
///    an `mycelium-l1`/`mycelium-interp` change. The one residual risk this reintroduces — two
///    *different* nodules independently declaring an item with the same simple name — is itself
///    checked during the merge and refused as `myc-run-nodule-fn-collision` if the declarations
///    differ (identical entries, e.g. a name re-exported through an import, are not a conflict).
/// 6. [`elaborate`] the entry nodule's `main` against the merged `Env` to a closed L0 Core IR node,
///    then run it on the trusted reference interpreter ([`mycelium_interp::Interpreter`]) — same as
///    the M-908 v0 path.
///
/// ## Scope (`Declared`, v0 — both single- and multi-nodule)
/// - **Entry convention:** the executed function must be named `main` and take no arguments (the
///   convention already used by the differential/conformance corpora) — a missing `main` is an
///   explicit refusal, never a silent pick of some other function.
/// - **Result fragment:** v0 observes only **representation-value** results
///   ([`mycelium_interp::Interpreter::eval`]); an entry that evaluates to an algebraic **data**
///   value (r3, RFC-0011) is refused rather than rendered ad hoc — a dedicated data-value printer is
///   follow-up work.
/// - **Swap engine:** v0 runs on the interpreter's default identity swap engine (same-representation
///   swap only, [`mycelium_interp::Interpreter::default`]); a program invoking the certified
///   binary↔ternary swap surfaces the interpreter's own explicit `UnsupportedSwap` error — never a
///   silent identity substitution for a real cross-paradigm conversion.
///
/// # Errors
/// [`Report`] on: no `.myc` source (`myc-run-no-source`), a parse/check failure (`myc-parse` /
/// `myc-check`), an unlinkable multi-nodule project (`myc-run-nodule-duplicate` /
/// `myc-run-nodule-unresolved` / `myc-run-nodule-cyclic` / `myc-run-nodule-fn-collision`), a
/// missing/ambiguous `main` (`myc-run-no-entry` / `myc-run-entry-ambiguous`), a program outside the
/// evaluation-complete fragment (`myc-run-residual`), or an interpreter-evaluation failure
/// (`myc-run-eval`) — every path is an explicit, located [`Report`], never a panic (G2).
pub fn run(manifest_path: &Path) -> Result<RunReport, Report> {
    run_with_options(manifest_path, &RunOptions::default())
}

/// [`run`] with an explicit [`RunOptions`] (RFC-0041 §5) — the entry the `myc` driver calls so a
/// `--unbounded` invocation threads its lifted depth ceiling into the interpreter. Behavior is
/// identical to [`run`] when `opts` is default.
///
/// # Errors
/// The same [`Report`] set as [`run`].
pub fn run_with_options(manifest_path: &Path, opts: &RunOptions) -> Result<RunReport, Report> {
    let (_, project_dir) = load_manifest(manifest_path)?;
    let sources =
        mycelium_cli_common::walk_myc(&project_dir).map_err(|e| Report::new("myc-io", e, 66))?;

    match sources.as_slice() {
        [] => Err(Report::new(
            "myc-run-no-source",
            format!("no `.myc` source found under {}", project_dir.display()),
            66,
        )
        .help("add a `.myc` source file to the project")),
        [single] => run_single_nodule(single, &project_dir, opts),
        multiple => run_multi_nodule(multiple, &project_dir, opts),
    }
}

/// The M-908 v0 path: exactly one `.myc` source — parse, [`check_nodule`], [`elaborate`] its
/// nullary `main`, then run on the reference interpreter. Behavior is unchanged from M-908 except
/// that `opts` selects the interpreter's depth budget ([`interpreter_for`] — RFC-0041 §5).
fn run_single_nodule(
    source_path: &Path,
    project_dir: &Path,
    opts: &RunOptions,
) -> Result<RunReport, Report> {
    let rel = rel_to_project(source_path, project_dir);

    let text = std::fs::read_to_string(source_path)
        .map_err(|e| Report::new("myc-io", format!("{}: {e}", source_path.display()), 66))?;

    let nodule = parse(&text).map_err(|ParseError { pos, message }| {
        Report::new("myc-parse", message, 65)
            .at(format!("{rel}:{}:{}", pos.line, pos.col))
            .help("fix the syntax error at the indicated position")
    })?;

    let env = check_nodule(&nodule).map_err(|ce| {
        Report::new("myc-check", ce.to_string(), 65)
            .at(rel.clone())
            .help("resolve the type error reported above (see `myc check`)")
    })?;

    const ENTRY: &str = "main";
    if env.fn_decl(ENTRY).is_none() {
        let mut available: Vec<&str> = env.fns.keys().map(String::as_str).collect();
        available.sort_unstable();
        let list = if available.is_empty() {
            "(none declared)".to_owned()
        } else {
            available.join(", ")
        };
        return Err(Report::new(
            "myc-run-no-entry",
            format!("no nullary `{ENTRY}` function in {rel} — v0 `myc run` executes `{ENTRY}`"),
            65,
        )
        .at(rel.clone())
        .help(format!(
            "declare a nullary `fn {ENTRY}() => …` entry point; declared function(s): {list}"
        )));
    }

    let node = elaborate(&env, ENTRY).map_err(|ee| {
        Report::new("myc-run-residual", ee.to_string(), 70)
            .at(rel.clone())
            .help(
                "the program uses a construct outside the evaluation-complete fragment \
                 (RFC-0007 §4.6); `myc run` v0 executes only the elaborated fragment",
            )
    })?;

    let interp = interpreter_for(opts);
    let value = interp.eval(&node).map_err(|ee| {
        Report::new("myc-run-eval", ee.to_string(), 65)
            .at(rel.clone())
            .help("the program failed during interpreted evaluation — see the error above")
    })?;

    Ok(RunReport {
        source: rel,
        entry: ENTRY.to_owned(),
        rendered: format!("{value:?}"),
    })
}

/// The M-909 multi-nodule path: manifest-driven project loading, nodule linking, and end-to-end
/// execution. See [`run`]'s doc for the full six-step model. `opts` selects the interpreter's depth
/// budget ([`interpreter_for`] — RFC-0041 §5).
fn run_multi_nodule(
    sources: &[PathBuf],
    project_dir: &Path,
    opts: &RunOptions,
) -> Result<RunReport, Report> {
    // Step 1: parse every source independently — each file is a bare `nodule <path>; …` block.
    let mut parsed: Vec<(String, Nodule)> = Vec::with_capacity(sources.len());
    for source_path in sources {
        let rel = rel_to_project(source_path, project_dir);
        let text = std::fs::read_to_string(source_path)
            .map_err(|e| Report::new("myc-io", format!("{}: {e}", source_path.display()), 66))?;
        let nodule = parse(&text).map_err(|ParseError { pos, message }| {
            Report::new("myc-parse", message, 65)
                .at(format!("{rel}:{}:{}", pos.line, pos.col))
                .help("fix the syntax error at the indicated position")
        })?;
        parsed.push((rel, nodule));
    }

    // Step 2: link-check before check_phylum (which does not itself guard duplicate nodule paths
    // or cyclic `use` graphs — G2: never let those corrupt the phylum-wide export table silently).
    check_no_duplicate_nodule_paths(&parsed)?;
    check_use_targets_resolve(&parsed)?;
    check_no_nodule_cycles(&parsed)?;

    // Step 3: assemble one Phylum (no header — path: None) and check it as a whole.
    let phylum = Phylum {
        path: None,
        nodules: parsed.iter().map(|(_, n)| n.clone()).collect(),
    };
    let phylum_env: PhylumEnv = check_phylum(&phylum).map_err(|ce: CheckError| {
        Report::new("myc-check", ce.to_string(), 65)
            .help("resolve the type error reported above (see `myc check`)")
    })?;

    // Step 4: find the single nodule declaring a nullary `main` (never guess between candidates).
    const ENTRY: &str = "main";
    let entry_path = find_entry_nodule(&phylum_env, &parsed)?;
    let entry_rel = parsed
        .iter()
        .find(|(_, n)| &n.path == entry_path)
        .map(|(rel, _)| rel.clone())
        .unwrap_or_else(|| entry_path.0.join("."));

    // Step 5: flatten every checked nodule's Env into one merged Env for elaboration (see `run`'s
    // doc — this is a v0 CLI-level linking policy, not an l1/interp change). A genuine simple-name
    // collision across two *different* nodules refuses rather than silently picking a winner.
    let merged = merge_phylum_env(&phylum_env)?;

    // Step 6: elaborate + run, same as the single-nodule path.
    let node = elaborate(&merged, ENTRY).map_err(|ee| {
        Report::new("myc-run-residual", ee.to_string(), 70)
            .at(entry_rel.clone())
            .help(
                "the program uses a construct outside the evaluation-complete fragment \
                 (RFC-0007 §4.6); `myc run` v0 executes only the elaborated fragment",
            )
    })?;

    let interp = interpreter_for(opts);
    let value = interp.eval(&node).map_err(|ee| {
        Report::new("myc-run-eval", ee.to_string(), 65)
            .at(entry_rel.clone())
            .help("the program failed during interpreted evaluation — see the error above")
    })?;

    Ok(RunReport {
        source: entry_rel,
        entry: ENTRY.to_owned(),
        rendered: format!("{value:?}"),
    })
}

/// `path`, relative to `project_dir` (falls back to the absolute path if stripping fails — never
/// panics, G2).
fn rel_to_project(path: &Path, project_dir: &Path) -> String {
    path.strip_prefix(project_dir)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The dot-joined nodule path (`a.b` for `nodule a.b;`), the key `myc run`'s M-909 linker uses to
/// identify a nodule across files.
fn nodule_path_string(path: &NoduleAstPath) -> String {
    path.0.join(".")
}

/// The nodule path a `use` targets: for a glob (`use a.b.*`) the whole path is the nodule; for a
/// specific import (`use a.b.Item`) the last segment is the imported item, so the nodule is the
/// prefix. Returns `None` for an unqualified specific `use` (a single-segment, non-glob path) — that
/// shape is malformed on its own terms and [`check_phylum`] reports it precisely; the M-909 linker
/// does not duplicate that diagnostic.
fn use_target_nodule_path(up: &UsePath) -> Option<String> {
    if up.glob {
        Some(up.path.0.join("."))
    } else if up.path.0.len() >= 2 {
        Some(up.path.0[..up.path.0.len() - 1].join("."))
    } else {
        None
    }
}

/// Never-silent (G2): two files declaring the same `nodule <path>;` would silently collide in
/// [`check_phylum`]'s qualified export table (`qualify` keys by nodule path); `myc run` v0 refuses
/// explicitly before that can happen.
fn check_no_duplicate_nodule_paths(parsed: &[(String, Nodule)]) -> Result<(), Report> {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for (rel, nodule) in parsed {
        let key = nodule_path_string(&nodule.path);
        if let Some(first_rel) = seen.get(&key) {
            return Err(Report::new(
                "myc-run-nodule-duplicate",
                format!(
                    "nodule `{key}` is declared in both {first_rel} and {rel} — every nodule path \
                     in a project must be unique"
                ),
                65,
            )
            .at(rel.clone())
            .help("rename one of the nodules, or merge their declarations into a single nodule"));
        }
        seen.insert(key, rel.clone());
    }
    Ok(())
}

/// Never-silent (G2): a `use` naming a nodule with no corresponding file in the project is refused
/// explicitly, rather than surfacing only as an opaque "unknown name" from [`check_phylum`].
fn check_use_targets_resolve(parsed: &[(String, Nodule)]) -> Result<(), Report> {
    let known: BTreeSet<String> = parsed
        .iter()
        .map(|(_, n)| nodule_path_string(&n.path))
        .collect();
    for (rel, nodule) in parsed {
        let from = nodule_path_string(&nodule.path);
        for item in &nodule.items {
            let Item::Use(up) = item else { continue };
            let Some(target) = use_target_nodule_path(up) else {
                continue;
            };
            if !known.contains(&target) {
                return Err(Report::new(
                    "myc-run-nodule-unresolved",
                    format!(
                        "nodule `{from}` ({rel}) references nodule `{target}` via `use`, but no \
                         nodule `{target}` exists in this project"
                    ),
                    65,
                )
                .at(rel.clone())
                .help(
                    "check the `use` path, or add the missing nodule's `.myc` source to the project",
                ));
            }
        }
    }
    Ok(())
}

/// Never-silent (G2), `Declared` v0 policy: `myc run` requires the nodule-level `use` dependency
/// graph to be acyclic (see [`run`]'s doc — this is stricter than [`check_phylum`] itself needs to
/// be, a deliberate v0 CLI simplification, not a kernel limitation).
fn check_no_nodule_cycles(parsed: &[(String, Nodule)]) -> Result<(), Report> {
    let known: BTreeSet<String> = parsed
        .iter()
        .map(|(_, n)| nodule_path_string(&n.path))
        .collect();
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (_, nodule) in parsed {
        let from = nodule_path_string(&nodule.path);
        let targets = edges.entry(from.clone()).or_default();
        for item in &nodule.items {
            let Item::Use(up) = item else { continue };
            let Some(target) = use_target_nodule_path(up) else {
                continue;
            };
            if target != from && known.contains(&target) {
                targets.insert(target);
            }
        }
    }

    // 3-color DFS marks: 0 = white (unvisited), 1 = gray (on the current path), 2 = black (done).
    let mut color: BTreeMap<String, u8> = edges.keys().cloned().map(|k| (k, 0)).collect();
    let starts: Vec<String> = edges.keys().cloned().collect();
    for start in starts {
        if color.get(&start).copied() != Some(0) {
            continue;
        }
        let mut path = Vec::new();
        if let Some(cycle) = dfs_find_cycle(&start, &edges, &mut color, &mut path) {
            let chain = cycle.join(" -> ");
            return Err(Report::new(
                "myc-run-nodule-cyclic",
                format!(
                    "cyclic nodule `use` dependency: {chain} — myc run v0 requires an acyclic \
                     nodule graph"
                ),
                65,
            )
            .help(
                "break the cycle by removing or restructuring the `use` that closes the loop; \
                 myc run v0 links nodules eagerly and does not support mutually-dependent \
                 nodules yet",
            ));
        }
    }
    Ok(())
}

/// Depth-first cycle search over the nodule `use` graph (3-color: white/gray/black), bounded by the
/// number of nodules in the project — recursion depth is at most the node count, never unbounded
/// input-driven recursion.
fn dfs_find_cycle(
    node: &str,
    edges: &BTreeMap<String, BTreeSet<String>>,
    color: &mut BTreeMap<String, u8>,
    path: &mut Vec<String>,
) -> Option<Vec<String>> {
    // 0 = white (unvisited), 1 = gray (on the current DFS path), 2 = black (fully explored).
    color.insert(node.to_owned(), 1);
    path.push(node.to_owned());
    if let Some(targets) = edges.get(node) {
        for t in targets {
            match color.get(t.as_str()).copied() {
                Some(1) => {
                    let start_idx = path.iter().position(|p| p == t).unwrap_or(0);
                    let mut cycle: Vec<String> = path[start_idx..].to_vec();
                    cycle.push(t.clone());
                    return Some(cycle);
                }
                Some(2) => continue,
                _ => {
                    if let Some(c) = dfs_find_cycle(t, edges, color, path) {
                        return Some(c);
                    }
                }
            }
        }
    }
    path.pop();
    color.insert(node.to_owned(), 2);
    None
}

/// The single nodule declaring a nullary `main`, or an explicit refusal — zero candidates is
/// `myc-run-no-entry`; more than one is `myc-run-entry-ambiguous` (never guesses between them, G2).
fn find_entry_nodule<'a>(
    phylum_env: &'a PhylumEnv,
    parsed: &[(String, Nodule)],
) -> Result<&'a NoduleAstPath, Report> {
    let candidates: Vec<&NoduleAstPath> = phylum_env
        .nodules
        .iter()
        .filter(|(_, env)| env.fn_decl("main").is_some())
        .map(|(path, _)| path)
        .collect();
    match candidates.as_slice() {
        [] => {
            let nodules: Vec<String> = parsed
                .iter()
                .map(|(_, n)| nodule_path_string(&n.path))
                .collect();
            Err(Report::new(
                "myc-run-no-entry",
                format!(
                    "no nodule declares a nullary `main` — v0 `myc run` executes `main`; nodules \
                     in this project: {}",
                    nodules.join(", ")
                ),
                65,
            )
            .help("declare a nullary `fn main() => …` in exactly one nodule of the project"))
        }
        [only] => Ok(only),
        many => {
            let names: Vec<String> = many.iter().map(|p| nodule_path_string(p)).collect();
            Err(Report::new(
                "myc-run-entry-ambiguous",
                format!(
                    "more than one nodule declares a nullary `main` ({}) — v0 `myc run` needs a \
                     single, unambiguous entry",
                    names.join(", ")
                ),
                65,
            )
            .help("keep a nullary `main` in exactly one nodule of the project"))
        }
    }
}

/// Flatten every checked nodule's [`Env`] into one merged `Env`, by simple name, for elaboration
/// (see [`run`]'s doc, step 5). [`check_phylum`] has already validated every cross-nodule reference
/// in the program is legal, so this merge is safe **except** for a genuine simple-name collision —
/// two different nodules independently declaring an item with the same name but a different
/// definition — which is refused as `myc-run-nodule-fn-collision` rather than silently picking a
/// winner (G2). An identical re-inserted entry (e.g. a name a nodule imported, cloned verbatim into
/// its own [`Env`] by [`check_phylum`]) is not a conflict.
fn merge_phylum_env(phylum_env: &PhylumEnv) -> Result<Env, Report> {
    let mut types = BTreeMap::new();
    let mut fns = BTreeMap::new();
    let mut totality = BTreeMap::new();
    let mut traits = BTreeMap::new();
    let mut instances = BTreeMap::new();
    let mut impls = BTreeMap::new();
    let mut lower_rules = BTreeMap::new();
    // DN-54 §10 Model A derive-site provenance (M-973): keyed by the same `(trait, head)` coherence
    // key as `instances`/`impls`, so it merges the same pub-key way across the phylum's nodules.
    let mut derived_provenance = BTreeMap::new();
    // `via`-delegation EXPLAIN provenance (M-966): keyed the same way, merges identically.
    let mut via_provenance = BTreeMap::new();
    let mut conflicts: Vec<String> = Vec::new();

    for (_, env) in &phylum_env.nodules {
        merge_map(&mut types, &env.types, String::clone, &mut conflicts);
        merge_map(&mut fns, &env.fns, String::clone, &mut conflicts);
        merge_map(&mut totality, &env.totality, String::clone, &mut conflicts);
        merge_map(&mut traits, &env.traits, String::clone, &mut conflicts);
        merge_map(&mut instances, &env.instances, fmt_pair_key, &mut conflicts);
        merge_map(&mut impls, &env.impls, fmt_pair_key, &mut conflicts);
        merge_map(
            &mut lower_rules,
            &env.lower_rules,
            String::clone,
            &mut conflicts,
        );
        merge_map(
            &mut derived_provenance,
            &env.derived_provenance,
            fmt_pair_key,
            &mut conflicts,
        );
        merge_map(
            &mut via_provenance,
            &env.via_provenance,
            fmt_pair_key,
            &mut conflicts,
        );
    }

    if !conflicts.is_empty() {
        conflicts.sort_unstable();
        conflicts.dedup();
        return Err(Report::new(
            "myc-run-nodule-fn-collision",
            format!(
                "myc run v0 links nodules by simple name; the following name(s) are declared \
                 differently by more than one nodule and cannot be unambiguously linked: {}",
                conflicts.join(", ")
            ),
            65,
        )
        .help(
            "rename one of the conflicting declarations — cross-nodule name collisions are not \
             yet disambiguated (v0; a future project-scoped linker will lift this)",
        ));
    }

    Ok(Env {
        types,
        fns,
        totality,
        traits,
        instances,
        impls,
        lower_rules,
        derived_provenance,
        via_provenance,
    })
}

/// `(String, String)`-keyed maps (`instances`/`impls`) format their key as `left::right` for a
/// collision report.
fn fmt_pair_key(k: &(String, String)) -> String {
    format!("{}::{}", k.0, k.1)
}

/// Merge `src` into `dst` by key: a new key is inserted; an existing key with an **equal** value is
/// left alone (e.g. the same declaration re-appearing via two nodules' imports); an existing key
/// with a **different** value is recorded (via `fmt_key`) in `conflicts` rather than silently
/// overwritten (G2 — the caller turns a non-empty `conflicts` into an explicit [`Report`]).
fn merge_map<K, V>(
    dst: &mut BTreeMap<K, V>,
    src: &BTreeMap<K, V>,
    fmt_key: impl Fn(&K) -> String,
    conflicts: &mut Vec<String>,
) where
    K: Ord + Clone,
    V: PartialEq + Clone,
{
    for (k, v) in src {
        match dst.get(k) {
            None => {
                dst.insert(k.clone(), v.clone());
            }
            Some(existing) if existing == v => {}
            Some(_) => conflicts.push(fmt_key(k)),
        }
    }
}

/// The outcome of a single nodule-component parse in [`stream_parse`].
///
/// Each entry corresponds to one nodule-component extracted from the stream.
/// `Ok(n)` records its 1-based component number on success; `Err(report)` carries the structured
/// diagnostic for a malformed component — never silent, never skipped (G2 / M-820).
pub type StreamComponent = Result<usize, Report>;

/// `myc --stream` — parse a `;`-delimited Mycelium component stream from `reader` (M-820 / DN-57).
///
/// ## Streaming semantics (`Declared` for the I/O strategy; `Empirical` for the split)
/// **v0 is whole-input-buffered for I/O.** The entire reader is read into a `String` first, then
/// the source is **lexer-split** into per-nodule components and each component is parsed
/// independently. This bounds the *parse* state to one component at a time (the per-component parse
/// is a [`mycelium_l1::parse`] call on the component's source slice, not the whole input), but the
/// *I/O* is fully buffered. True per-`;`-component **incremental** I/O would require the L1 lexer to
/// expose a resumable/incremental token-stream API (one does not exist yet); that is flagged as
/// future work (`Declared`). The *split* itself is `Empirical` — it is token-accurate (see below)
/// and tested, including comment-/string-safety.
///
/// ## Component granularity — token-driven, comment-safe (DN-57 §2)
/// The source is tokenized once via [`mycelium_l1::lexer::lex`]; the token stream is then segmented
/// at [`mycelium_l1::token::Tok::Nodule`] keyword tokens. Each "component" is a complete Mycelium
/// nodule block — from its `nodule` header token through all its `;`-terminated
/// ([`Tok::Semi`](mycelium_l1::token::Tok::Semi)) items, up to (but not including) the next `nodule`
/// header token. Crucially this is **not** a raw-text keyword scan: a `nodule` or `;` appearing
/// inside a `//` comment (or a future string literal) is **not** a `Tok::Nodule`/`Tok::Semi` token,
/// so it can never cause a mis-split (DN-57 §2: "the end-of-component is a *token*, not the *absence*
/// of more tokens" — a streaming parser must not scan ahead for the next item-opening *keyword text*).
///
/// ## Never-silent error contract (G2)
/// - A **lex** failure surfaces as an outer `Err(Report)` (`myc-stream-lex`) with the source
///   position — a lexically invalid stream is never silently truncated.
/// - A malformed component yields a [`Report`] (`myc-stream-parse`) with the 1-based component
///   index, the parse-error position within that component, and an actionable `help:` line. The
///   remaining components are still attempted — one bad component does not abort the stream.
/// - A component whose last token before the next `nodule`/EOF is **not** `Tok::Semi` is an
///   unterminated component: an explicit `myc-stream-eof` error (DN-57 §3.1 — mandatory `;`), never
///   a silent partial accept.
/// - An entirely empty stream (no tokens) or one with no `nodule` header is an explicit
///   `myc-stream-empty` / per-component error — never silently succeeded.
///
/// ## I/O errors
/// An I/O failure reading `reader` is returned as an outer `Err(Report)` (`myc-stream-io`, exit 66)
/// before any parse results.
///
/// ## Return value
/// Returns `Ok(Vec<StreamComponent>)` — one entry per component. `Err(report)` entries are
/// per-component parse / unterminated failures; `Ok(n)` entries confirm success. The outer `Result`
/// carries I/O, lex, or empty-stream errors that prevent any per-component parsing.
///
/// # Errors
/// Returns `Err(Report)` for a fatal I/O failure on `reader`, a lex failure, or an empty stream.
pub fn stream_parse(
    mut reader: impl StdRead,
    source_name: &str,
) -> Result<Vec<StreamComponent>, Report> {
    // --- Step 1: read the entire input (v0: full-input buffering; `Declared` limitation) ---
    let mut src = String::new();
    reader.read_to_string(&mut src).map_err(|e| {
        Report::new("myc-stream-io", format!("{source_name}: {e}"), 66)
            .help("check that the input source is readable and produces valid UTF-8")
    })?;

    // --- Step 2: lex once (never-silent: a lex error surfaces explicitly, G2) ---
    let toks = lex(&src).map_err(|ParseError { pos, message }| {
        Report::new(
            "myc-stream-lex",
            format!("`{source_name}` failed to lex: {message}"),
            65,
        )
        .at(format!("{source_name}:{}:{}", pos.line, pos.col))
        .help("fix the lexically invalid token at the indicated position")
    })?;

    // --- Step 3: segment the token stream at `nodule` header tokens (comment-safe by construction) ---
    // A `nodule`/`;` inside a `//` comment is never a `Tok::Nodule`/`Tok::Semi`, so this split is
    // immune to comment/string-literal mis-splits (DN-57 §2).
    let segments = segment_nodule_components(&toks);

    if segments.is_empty() {
        // No `nodule` header token — either an empty stream (only `Eof`) or content with no header.
        // Distinguish: a stream that is only `Eof` is empty; otherwise it is one malformed component.
        let non_eof = toks.iter().any(|s| s.tok != Tok::Eof);
        if !non_eof {
            return Err(Report::new(
                "myc-stream-empty",
                format!("`{source_name}` is empty — no components to parse"),
                65,
            )
            .help(
                "a Mycelium stream must contain at least one `nodule`-headed component (DN-57); \
                 check that the input is non-empty",
            ));
        }
        // Tokens present but no `nodule` header — surface as one explicit malformed component.
        return Ok(vec![parse_component(src.trim(), 1, source_name)]);
    }

    // --- Step 4: per-segment, slice the source and parse (or report unterminated) ---
    // Build a line-start byte index so a token `Pos` (1-based line/col) maps to a byte offset.
    let line_starts = line_start_offsets(&src);
    let mut results: Vec<StreamComponent> = Vec::with_capacity(segments.len());

    for (comp_idx, seg) in segments.iter().enumerate() {
        let one_based = comp_idx + 1;
        // The segment's source slice runs from its first token's byte offset to its end byte offset.
        let start_byte = pos_to_byte(&line_starts, &src, seg.start_pos);
        let end_byte = seg
            .end_pos
            .map_or(src.len(), |p| pos_to_byte(&line_starts, &src, p));
        let slice = src.get(start_byte..end_byte).unwrap_or("").trim();

        if !seg.terminated {
            // Never-silent: the last token before the boundary is not `Tok::Semi` (DN-57 §3.1).
            results.push(Err(Report::new(
                "myc-stream-eof",
                format!(
                    "component {one_based} in `{source_name}` is unterminated: \
                     its last item has no `;` terminator before the next component / EOF"
                ),
                65,
            )
            .at(format!(
                "{source_name}:{one_based}:{}:{}",
                seg.start_pos.line, seg.start_pos.col
            ))
            .help(
                "every Mycelium component must end with `;` after its last item (DN-57 §3.1); \
                 add `;` at the end of the component",
            )));
        } else {
            results.push(parse_component(slice, one_based, source_name));
        }
    }

    Ok(results)
}

/// One lexer-segmented nodule-component: where its `nodule` header token starts, where the next
/// component (or EOF) starts, and whether its final token is the mandatory `;` terminator.
struct NoduleSegment {
    /// Source position of the segment's opening `nodule` token (1-based line/col).
    start_pos: Pos,
    /// Source position of the *next* segment's opening `nodule` token, or `None` for the last
    /// segment (which runs to end-of-source).
    end_pos: Option<Pos>,
    /// Whether the last non-`Eof` token of this segment is `Tok::Semi` (DN-57 mandatory terminator).
    terminated: bool,
}

/// Segment a token stream into per-nodule components at `Tok::Nodule` header boundaries.
///
/// Each segment runs from one `Tok::Nodule` token up to (but not including) the next `Tok::Nodule`
/// token (or `Tok::Eof`). A segment is `terminated` iff its last non-`Eof` token is `Tok::Semi` —
/// the DN-57 mandatory component terminator. Comment-safe by construction: comments are never in the
/// token stream, so a `nodule`/`;` inside a comment cannot start or terminate a segment.
///
/// Guarantee: `Empirical` — validated by the stream tests (including comment-/string-safety).
fn segment_nodule_components(toks: &[Spanned]) -> Vec<NoduleSegment> {
    // Collect the indices of every `nodule` header token.
    let nodule_idxs: Vec<usize> = toks
        .iter()
        .enumerate()
        .filter(|(_, s)| s.tok == Tok::Nodule)
        .map(|(i, _)| i)
        .collect();

    let mut segments = Vec::with_capacity(nodule_idxs.len());
    for (n, &start_idx) in nodule_idxs.iter().enumerate() {
        // The token range of this segment: [start_idx, next_nodule_idx) — or to the end otherwise.
        let next_nodule_idx = nodule_idxs.get(n + 1).copied();
        let end_idx = next_nodule_idx.unwrap_or(toks.len());

        // The boundary position (start of the next component) — `None` for the last segment.
        let end_pos = next_nodule_idx.map(|i| toks[i].pos);

        // Terminated iff the last non-`Eof` token in [start_idx, end_idx) is `Tok::Semi`.
        let terminated = toks[start_idx..end_idx]
            .iter()
            .rev()
            .find(|s| s.tok != Tok::Eof)
            .is_some_and(|s| s.tok == Tok::Semi);

        segments.push(NoduleSegment {
            start_pos: toks[start_idx].pos,
            end_pos,
            terminated,
        });
    }
    segments
}

/// Byte offsets of the start of each 1-based source line (`line_starts[0]` = 0 = start of line 1).
/// Used to map a token [`Pos`](mycelium_l1::token::Pos) (1-based line/col) to a byte offset.
fn line_start_offsets(src: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, b) in src.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}

/// Map a 1-based `Pos` (line/col) to a byte offset in `src`, using a precomputed `line_starts`.
///
/// The lexer counts `col` in characters (1-based), so we walk `col - 1` chars from the line start to
/// land on the correct byte offset (handles multi-byte UTF-8). A position past end-of-line clamps to
/// the source length — never panics (G2).
fn pos_to_byte(line_starts: &[usize], src: &str, pos: Pos) -> usize {
    let line_idx = (pos.line as usize).saturating_sub(1);
    let Some(&line_byte) = line_starts.get(line_idx) else {
        return src.len();
    };
    // Walk `col - 1` characters from the line start.
    let col_offset = (pos.col as usize).saturating_sub(1);
    let rest = &src[line_byte..];
    match rest.char_indices().nth(col_offset) {
        Some((byte_in_line, _)) => line_byte + byte_in_line,
        None => {
            // `col` is past the last char of the line — clamp to the line end (next line start - 1)
            // or the source length for the final line.
            line_starts
                .get(line_idx + 1)
                .map_or(src.len(), |&next| next.saturating_sub(1))
        }
    }
}

/// Parse a single component's source slice as a Mycelium nodule.
///
/// Returns `Ok(component_idx)` on success; `Err(Report)` with a fully-located diagnostic on any
/// parse failure (G2: never silent, never panics — backed by [`mycelium_l1::parse`]'s own contract).
fn parse_component(text: &str, component_idx: usize, source_name: &str) -> StreamComponent {
    match parse(text) {
        Ok(_nodule) => Ok(component_idx),
        Err(ParseError { pos, message }) => Err(Report::new(
            "myc-stream-parse",
            format!("component {component_idx} in `{source_name}` failed to parse: {message}"),
            65,
        )
        .at(format!(
            "{source_name}:{component_idx}:{}:{}",
            pos.line, pos.col
        ))
        .help(
            "fix the syntax error at the indicated component:line:col position; \
             each component must be a valid Mycelium nodule terminated with `;`",
        )),
    }
}

/// The result of [`stream_parse`] summarised for the CLI.
///
/// Parallel to [`CheckReport`] but for streaming input rather than project files.
/// Carries the per-component results and the source name for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamReport {
    /// How many components parsed cleanly.
    pub parsed_ok: usize,
    /// How many components failed to parse (or were unterminated).
    pub parsed_err: usize,
    /// The structured failures, each located to a component.
    pub failures: Vec<Report>,
    /// Human-readable source label (e.g. `"<stdin>"` or a file path).
    pub source_name: String,
}

impl StreamReport {
    /// Whether every component parsed successfully.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Drive [`stream_parse`] and collect results into a [`StreamReport`].
///
/// Converts the per-component `Vec<StreamComponent>` from [`stream_parse`] into a summary
/// suitable for CLI display and test assertions.
///
/// # Errors
/// Returns `Err(Report)` for an I/O failure or an empty stream (no components found).
pub fn run_stream_parse(reader: impl StdRead, source_name: &str) -> Result<StreamReport, Report> {
    let components = stream_parse(reader, source_name)?;
    let mut report = StreamReport {
        parsed_ok: 0,
        parsed_err: 0,
        failures: Vec::new(),
        source_name: source_name.to_owned(),
    };
    for result in components {
        match result {
            Ok(_) => report.parsed_ok += 1,
            Err(r) => {
                report.parsed_err += 1;
                report.failures.push(r);
            }
        }
    }
    Ok(report)
}

// --- internals ---------------------------------------------------------------------------------

/// Load + parse the manifest at `manifest_path`, returning it with the project directory.
fn load_manifest(manifest_path: &Path) -> Result<(mycelium_proj::Manifest, PathBuf), Report> {
    let text = std::fs::read_to_string(manifest_path).map_err(|e| {
        Report::new("myc-io", format!("{}: {e}", manifest_path.display()), 66)
            .help("run `myc` from a project directory, or pass the manifest path")
    })?;
    let manifest = parse_manifest(&text).map_err(|e| {
        Report::new("myc-manifest", e.to_string(), 2).at(manifest_path.display().to_string())
    })?;
    let project_dir = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    Ok((manifest, project_dir))
}

/// Validate an `init` name: `[a-z][a-z0-9_]*`. A bad name is refused, never normalized (G2).
fn validate_name(name: &str) -> Result<(), Report> {
    let bad = || {
        Report::new(
            "myc-init-name",
            format!("{name:?} is not a valid phylum name"),
            64,
        )
        .help("use a lowercase identifier: a letter then letters/digits/underscores, e.g. `acme_geometry`")
    };
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return Err(bad()),
    }
    if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        return Err(bad());
    }
    Ok(())
}

/// Write `content` to `path`, refusing to clobber an existing file (G2).
fn write_new(path: &Path, content: &str) -> Result<(), Report> {
    if path.exists() {
        return Err(Report::new(
            "myc-init-exists",
            format!("{} already exists", path.display()),
            66,
        ));
    }
    std::fs::write(path, content)
        .map_err(|e| Report::new("myc-io", format!("{}: {e}", path.display()), 66))
}

/// The scaffolded `mycelium-proj.toml` for `name`.
fn scaffold_manifest(name: &str) -> String {
    format!(
        "# Scaffolded by `myc init`. A minimal, gate-clean phylum.\n\
         [project]\n\
         name    = \"{name}\"\n\
         kind    = \"phylum\"\n\
         version = \"0.1.0\"\n\
         license = \"MIT\"\n\
         summary = \"{name} — a new Mycelium phylum.\"\n\
         \n\
         [surface]\n\
         exports = [\"{name}\"]\n"
    )
}

/// The scaffolded starter nodule for `name`.
fn scaffold_nodule(name: &str) -> String {
    format!(
        "// nodule: {name}\n\
         // @summary: {name} — scaffolded by `myc init`; replace with your own definitions.\n\
         nodule {name};\n\
         \n\
         fn answer() => Binary{{8}} =\n  \
         0b0010_1010;\n"
    )
}

#[cfg(test)]
mod tests;
