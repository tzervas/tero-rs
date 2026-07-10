//! `mycelium-check` — the project-aware correctness/type-check **driver** (M-365; the `myc-check`
//! prototype grows up).
//!
//! It resolves a `mycelium-proj.toml` project, checks the **whole** `phylum`/program, and aggregates
//! every refusal as a structured diagnostic **routed through the M-362 auto-baseline** (RFC-0013 levels +
//! routes), exiting non-zero on any error so CI can gate on it. It changes nothing about *what* the
//! checker decides — the trusted M-210 checker ([`mycelium_l1::check_nodule`]) is the base (KC-3); this is
//! the *driver* above it: discovery + aggregation + honest routing.
//!
//! Honesty: a `ParseError`/`CheckError` is an **explicit** finding with a site (never a silent pass; G2),
//! and a check refusal is routed at the baseline level/route for the umbrella `NotValidated` class —
//! `CheckError` is a flat `{site, message}`, so the driver does **not** fabricate a finer class
//! (`TypeMismatch` vs `UnresolvedName`) it cannot structurally distinguish (VR-5: report what is known,
//! never invent). Per-op guarantee tags computed by the checker are untouched.

use std::path::{Path, PathBuf};

use mycelium_l1::{check_nodule, parse};
use mycelium_lsp::{
    derive_baseline, present, ClassRegistry, DiagnosticPolicy, Level, ReasonedError,
};

/// What kind of refusal a finding records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingKind {
    /// A syntactic refusal (parse).
    Parse,
    /// A static-check refusal (type/totality/name/validation).
    Check,
}

/// One aggregated diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The file it occurred in.
    pub file: String,
    /// Parse or check.
    pub kind: FindingKind,
    /// The site within the file (a definition name, or empty for a whole-file parse error).
    pub site: String,
    /// The author-facing message.
    pub message: String,
    /// The baseline presentation level (check refusals; `Minimal` for parse).
    pub level: Level,
    /// The baseline route, if any (check refusals).
    pub route: Option<String>,
}

impl Finding {
    /// Attach a baseline route, fluently (M-644 ergonomics). Additive builder; sets `route`.
    #[must_use]
    pub fn with_route(mut self, route: String) -> Self {
        self.route = Some(route);
        self
    }
}

/// The aggregated result of checking a set of sources.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Every finding, deterministically ordered (by file).
    pub findings: Vec<Finding>,
    /// How many files were checked.
    pub files_checked: usize,
}

impl Report {
    /// Push a finding, fluently (M-644 ergonomics). Additive builder; appends to `findings` (does
    /// **not** touch `files_checked` — set that explicitly with [`Report::with_files_checked`]).
    #[must_use]
    pub fn with_finding(mut self, finding: Finding) -> Self {
        self.findings.push(finding);
        self
    }

    /// Set the checked-file count, fluently (M-644 ergonomics). Additive builder.
    #[must_use]
    pub fn with_files_checked(mut self, files_checked: usize) -> Self {
        self.files_checked = files_checked;
        self
    }

    /// Whether the report is clean (no findings).
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.findings.is_empty()
    }

    /// The CI exit code: 2 if any parse error, else 3 if any check error, else 0.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        if self.findings.iter().any(|f| f.kind == FindingKind::Parse) {
            2
        } else if self.findings.iter().any(|f| f.kind == FindingKind::Check) {
            3
        } else {
            0
        }
    }
}

/// A project-resolution failure — no/ambiguous input (no sources, an unreadable file). Exit 5.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveError(pub String);

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "resolution-error: {}", self.0)
    }
}

impl std::error::Error for ResolveError {}

/// Check one source, appending any finding. Check refusals are routed through `policy` (the M-362
/// baseline) for their level/route; parse refusals are syntactic (pre-class) and presented minimally.
pub fn check_source(
    file: &str,
    src: &str,
    policy: &DiagnosticPolicy,
    registry: &ClassRegistry,
    out: &mut Vec<Finding>,
) {
    match parse(src) {
        Err(e) => out.push(Finding {
            file: file.to_owned(),
            kind: FindingKind::Parse,
            site: String::new(),
            message: e.to_string(),
            level: Level::Minimal,
            route: None,
        }),
        Ok(nodule) => {
            if let Err(ce) = check_nodule(&nodule) {
                // Route through the baseline at the umbrella static-check class (honest: the flat
                // CheckError carries no finer class to claim — VR-5).
                let class = registry
                    .resolve("NotValidated")
                    .expect("NotValidated is a builtin class");
                let p = present(
                    ReasonedError::new(class, ce.message.clone(), ce.site.clone()),
                    Some(policy),
                );
                out.push(Finding {
                    file: file.to_owned(),
                    kind: FindingKind::Check,
                    site: ce.site,
                    message: ce.message,
                    level: p.diagnostic.level,
                    route: p.diagnostic.route,
                });
            }
        }
    }
}

/// Check one source under the **default baseline policy** — the M-644 ergonomic convenience for the
/// common case where a caller has no custom `policy`/`registry`. Derives the builtin
/// [`ClassRegistry`] + the [`derive_baseline`] policy (exactly as [`check_sources`] does) and delegates
/// to the 5-arg [`check_source`]. A *new name* (Rust has no overloading; renaming `check_source` would
/// be breaking — M-644 is additive-only). For many sources prefer [`check_sources`], which builds the
/// registry/policy once.
pub fn check_source_default(file: &str, src: &str, out: &mut Vec<Finding>) {
    let registry = ClassRegistry::with_builtins();
    let policy = derive_baseline(&registry);
    check_source(file, src, &policy, &registry, out);
}

/// Check an explicit set of `(path, contents)` sources, aggregating findings deterministically.
#[must_use]
pub fn check_sources(sources: &[(String, String)]) -> Report {
    let registry = ClassRegistry::with_builtins();
    let policy = derive_baseline(&registry);
    let mut findings = Vec::new();
    for (file, src) in sources {
        check_source(file, src, &policy, &registry, &mut findings);
    }
    findings.sort_by(|a, b| a.file.cmp(&b.file));
    Report {
        findings,
        files_checked: sources.len(),
    }
}

/// Resolve and check a whole project: every `.myc` under `dir`.
///
/// # Errors
/// [`ResolveError`] when there are no `.myc` sources, or a source cannot be read (the project input is
/// missing/ambiguous; G2 — never a silent empty pass).
pub fn check_project(dir: &Path) -> Result<Report, ResolveError> {
    let files = collect_myc(dir)?;
    if files.is_empty() {
        return Err(ResolveError(format!(
            "no `.myc` sources under {} — nothing to check (a clean exit here would be a silent pass; G2)",
            dir.display()
        )));
    }
    let mut sources = Vec::with_capacity(files.len());
    for f in files {
        let src = std::fs::read_to_string(&f)
            .map_err(|e| ResolveError(format!("{}: {e}", f.display())))?;
        let rel = f
            .strip_prefix(dir)
            .unwrap_or(&f)
            .to_string_lossy()
            .replace('\\', "/");
        sources.push((rel, src));
    }
    Ok(check_sources(&sources))
}

// ---------------------------------------------------------------------------------------------------
// Phylum-check mode (M-1006) — check a set of `.myc` files as **one cross-resolving phylum**, not as
// isolated phyla-of-one. `check_sources`/`check_project` above run [`check_nodule`] per file, so a
// cross-nodule `use a.Foo;` cannot resolve (each file is a phylum-of-one). This mode assembles the
// files into a single `Phylum` and runs [`mycelium_l1::check_phylum`], the kernel's cross-nodule
// resolver — additive, alongside (never replacing) the per-file modes above.
//
// Honesty (VR-5/G2): `check_phylum` is **all-or-nothing** — it returns either the whole `PhylumEnv`
// or one `CheckError`. On success every nodule is reported `Clean`; on failure we report that single
// `CheckError` faithfully and fabricate **no** per-nodule verdicts we cannot know. A parse failure or
// a duplicate `nodule` path is refused **before** assembly (never a silent collision in the
// phylum-wide export table). The guarantee is `Empirical` (real toolchain).

/// What kind of refusal blocked a phylum from checking clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhylumErrorKind {
    /// A source failed to parse — the phylum cannot be assembled from an unparseable nodule.
    Parse,
    /// Two nodules declared the same dotted `nodule` path — an ambiguous phylum export table (G2).
    Duplicate,
    /// The assembled phylum failed [`mycelium_l1::check_phylum`] (a cross-nodule type/name refusal).
    Check,
}

impl PhylumErrorKind {
    /// The stable lowercase tag used in the `--json` contract (`"parse"`/`"duplicate"`/`"check"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            PhylumErrorKind::Parse => "parse",
            PhylumErrorKind::Duplicate => "duplicate",
            PhylumErrorKind::Check => "check",
        }
    }
}

/// The single refusal that blocked a phylum (all-or-nothing — there is at most one).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhylumError {
    /// Which class of refusal.
    pub kind: PhylumErrorKind,
    /// Where it occurred — a file label (parse) or a nodule path (duplicate/check site).
    pub site: String,
    /// The author-facing message.
    pub message: String,
}

/// A per-nodule verdict, emitted **only** when the whole phylum checked clean (never fabricated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoduleVerdict {
    /// The nodule's dotted path (`a`, `core.binary`).
    pub nodule: String,
    /// The verdict class — currently always `"Clean"` (only emitted on a clean phylum).
    pub class: &'static str,
}

/// The result of checking a set of sources **as one phylum**.
#[derive(Debug, Clone)]
pub struct PhylumReport {
    /// Whether the whole phylum checked clean.
    pub ok: bool,
    /// How many sources were assembled into the phylum.
    pub files_checked: usize,
    /// The single blocking refusal, if any (`None` iff `ok`).
    pub error: Option<PhylumError>,
    /// One `Clean` verdict per nodule — populated **only** when `ok` (VR-5: never a guessed verdict).
    pub nodules: Vec<NoduleVerdict>,
}

impl PhylumReport {
    /// The CI exit code: 0 if clean, 2 for a parse error, 3 for a check/duplicate refusal.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match &self.error {
            None => 0,
            Some(e) => match e.kind {
                PhylumErrorKind::Parse => 2,
                PhylumErrorKind::Check | PhylumErrorKind::Duplicate => 3,
            },
        }
    }
}

/// Check an explicit set of `(path, contents)` sources **as one cross-resolving phylum** (the
/// FS-free core, mirroring [`check_sources`]'s shape so it is unit-testable). See the module note
/// above for the honesty contract.
///
/// Steps: parse each source (any parse failure → an explicit `Parse` refusal — the phylum cannot be
/// assembled), guard against duplicate `nodule` paths (a `Duplicate` refusal — never a silent export
/// collision, G2), assemble one `Phylum { path: None, .. }`, and run [`mycelium_l1::check_phylum`]
/// (all-or-nothing: `Clean` verdicts on success, one faithful `Check` refusal on failure).
#[must_use]
pub fn check_phylum_sources(sources: &[(String, String)]) -> PhylumReport {
    let files_checked = sources.len();

    // 1. Parse each source. Any parse failure refuses the whole phylum (never a silent partial).
    let mut nodules: Vec<mycelium_l1::Nodule> = Vec::with_capacity(files_checked);
    for (file, src) in sources {
        match parse(src) {
            Ok(nodule) => nodules.push(nodule),
            Err(e) => {
                return PhylumReport {
                    ok: false,
                    files_checked,
                    error: Some(PhylumError {
                        kind: PhylumErrorKind::Parse,
                        site: file.clone(),
                        message: e.to_string(),
                    }),
                    nodules: Vec::new(),
                };
            }
        }
    }

    // 2. Never-silent duplicate-nodule-path guard (mirrors mycelium-cli's
    // `check_no_duplicate_nodule_paths`): `check_phylum` keys its export table by nodule path and
    // would otherwise let a second nodule of the same path collide silently (G2).
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for nodule in &nodules {
        let key = nodule.path.0.join(".");
        if !seen.insert(key.clone()) {
            return PhylumReport {
                ok: false,
                files_checked,
                error: Some(PhylumError {
                    kind: PhylumErrorKind::Duplicate,
                    site: key.clone(),
                    message: format!(
                        "nodule `{key}` is declared more than once — every nodule path in a phylum \
                         must be unique"
                    ),
                }),
                nodules: Vec::new(),
            };
        }
    }

    // 3. Assemble one header-less phylum and check it as a whole.
    let phylum = mycelium_l1::Phylum {
        path: None,
        nodules,
    };
    match mycelium_l1::check_phylum(&phylum) {
        Ok(_) => {
            let verdicts = phylum
                .nodules
                .iter()
                .map(|n| NoduleVerdict {
                    nodule: n.path.0.join("."),
                    class: "Clean",
                })
                .collect();
            PhylumReport {
                ok: true,
                files_checked,
                error: None,
                nodules: verdicts,
            }
        }
        // All-or-nothing: report the one CheckError faithfully; do NOT guess which nodules are clean.
        Err(ce) => PhylumReport {
            ok: false,
            files_checked,
            error: Some(PhylumError {
                kind: PhylumErrorKind::Check,
                site: ce.site,
                message: ce.message,
            }),
            nodules: Vec::new(),
        },
    }
}

/// Resolve and check every `.myc` under `dir` **as one phylum** (the FS wrapper over
/// [`check_phylum_sources`], mirroring [`check_project`]'s discovery/read path).
///
/// # Errors
/// [`ResolveError`] when there are no `.myc` sources, or a source cannot be read (missing/ambiguous
/// input; G2 — never a silent empty pass).
pub fn check_phylum_dir(dir: &Path) -> Result<PhylumReport, ResolveError> {
    let files = collect_myc(dir)?;
    if files.is_empty() {
        return Err(ResolveError(format!(
            "no `.myc` sources under {} — nothing to check (a clean exit here would be a silent pass; G2)",
            dir.display()
        )));
    }
    let mut sources = Vec::with_capacity(files.len());
    for f in files {
        let src = std::fs::read_to_string(&f)
            .map_err(|e| ResolveError(format!("{}: {e}", f.display())))?;
        let rel = f
            .strip_prefix(dir)
            .unwrap_or(&f)
            .to_string_lossy()
            .replace('\\', "/");
        sources.push((rel, src));
    }
    Ok(check_phylum_sources(&sources))
}

/// Collect every `.myc` under `dir` (recursively), sorted; skipping hidden entries and `target/`.
fn collect_myc(dir: &Path) -> Result<Vec<PathBuf>, ResolveError> {
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), ResolveError> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| ResolveError(format!("{}: {e}", dir.display())))?;
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "myc") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_program_checks_ok() {
        let r = check_sources(&[(
            "ok.myc".to_owned(),
            "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x;\n".to_owned(),
        )]);
        assert!(r.is_ok(), "{:?}", r.findings);
        assert_eq!(r.exit_code(), 0);
    }

    #[test]
    fn a_parse_error_is_an_explicit_finding_exit_2() {
        let r = check_sources(&[(
            "bad.myc".to_owned(),
            "nodule d\nfn f(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6})".to_owned(),
        )]);
        assert_eq!(r.findings.len(), 1);
        assert_eq!(r.findings[0].kind, FindingKind::Parse);
        assert_eq!(r.exit_code(), 2);
    }

    #[test]
    fn a_check_error_is_routed_through_the_baseline_exit_3() {
        // An undefined name is a check refusal (UnresolvedName-class), routed at the baseline level.
        let r = check_sources(&[(
            "c.myc".to_owned(),
            "nodule d;\nfn f() => Binary{8} = nope(0b0);\n".to_owned(),
        )]);
        assert_eq!(r.exit_code(), 3, "{:?}", r.findings);
        let c = r
            .findings
            .iter()
            .find(|f| f.kind == FindingKind::Check)
            .expect("a check finding");
        // The M-362 baseline routes static-check refusals to the diagnostic stream at medium detail.
        assert_eq!(c.level, Level::Medium, "{c:?}");
        assert_eq!(c.route.as_deref(), Some("stream"), "{c:?}");
    }

    #[test]
    fn aggregation_is_deterministic_and_reports_all_files() {
        let r = check_sources(&[
            (
                "b.myc".to_owned(),
                "nodule d;\nfn f() => Binary{8} = nope(0b0);\n".to_owned(),
            ),
            (
                "a.myc".to_owned(),
                "nodule d;\nfn g() => Binary{8} = also_nope(0b0);\n".to_owned(),
            ),
        ]);
        // Both files reported, sorted by name (a before b).
        assert_eq!(r.findings.len(), 2);
        assert_eq!(r.findings[0].file, "a.myc");
        assert_eq!(r.findings[1].file, "b.myc");
        assert_eq!(r.exit_code(), 3);
    }

    #[test]
    fn check_source_default_and_builders_are_additive_ergonomics() {
        // M-644: the default-policy convenience checks one source via the same baseline path as
        // check_sources (builds the builtin registry + derived policy, delegates to check_source).
        let mut out = Vec::new();
        check_source_default(
            "a.myc",
            "nodule d;\nfn g() => Binary{8} = also_nope(0b0);\n",
            &mut out,
        );
        assert!(
            !out.is_empty(),
            "an unresolved call is a recorded check finding"
        );
        // The fluent builders compose a Report additively (no canonical constructor changed).
        let f = out.remove(0).with_route("escalate".to_owned());
        assert_eq!(f.route.as_deref(), Some("escalate"));
        let r = Report::default().with_finding(f).with_files_checked(1);
        assert_eq!(r.findings.len(), 1);
        assert_eq!(r.files_checked, 1);
    }

    // --- Phylum-check mode (M-1006) -------------------------------------------------------------

    #[test]
    fn phylum_cross_nodule_reference_resolves() {
        // Nodule `a` exports `helper` (`pub fn`); nodule `b` imports it (`use a.*`) and calls it. As
        // one phylum this resolves (RFC-0006 §4.3, mirrors the l1 `cross_nodule_program_runs_three_way`).
        let a = (
            "a.myc".to_owned(),
            "nodule a;\npub fn helper(x: Binary{8}) => Binary{8} = not(x);\n".to_owned(),
        );
        let b = (
            "b.myc".to_owned(),
            "nodule b;\nuse a.*;\nfn g(x: Binary{8}) => Binary{8} = helper(x);\n".to_owned(),
        );
        let report = check_phylum_sources(&[a, b.clone()]);
        assert!(
            report.ok,
            "phylum should resolve `a.helper`: {:?}",
            report.error
        );
        assert_eq!(report.exit_code(), 0);
        assert_eq!(report.nodules.len(), 2);
        assert!(report.nodules.iter().all(|v| v.class == "Clean"));

        // Witness that the phylum path is what makes it resolve: the SAME `b.myc` checked in
        // isolation (a phylum-of-one, the per-file path) FAILS — `a.helper` is unresolved there.
        let isolated = check_sources(&[b]);
        assert!(
            !isolated.is_ok(),
            "b.myc must NOT resolve `a.*` in isolation (proves the phylum lever): {:?}",
            isolated.findings
        );
        assert_eq!(isolated.exit_code(), 3);
    }

    #[test]
    fn phylum_duplicate_nodule_path_is_refused() {
        // Two nodules both declare `nodule a;` — an ambiguous export table, refused never-silently (G2)
        // BEFORE reaching check_phylum.
        let report = check_phylum_sources(&[
            (
                "x.myc".to_owned(),
                "nodule a;\npub fn helper(x: Binary{8}) => Binary{8} = not(x);\n".to_owned(),
            ),
            (
                "y.myc".to_owned(),
                "nodule a;\npub fn other(x: Binary{8}) => Binary{8} = x;\n".to_owned(),
            ),
        ]);
        assert!(!report.ok);
        assert_eq!(
            report.error.as_ref().map(|e| e.kind),
            Some(PhylumErrorKind::Duplicate),
            "{:?}",
            report.error
        );
        assert_eq!(report.exit_code(), 3);
    }

    #[test]
    fn phylum_parse_error_is_reported() {
        // Missing `;` after the nodule header — an unparseable nodule; the phylum cannot be assembled.
        let report = check_phylum_sources(&[(
            "bad.myc".to_owned(),
            "nodule a\npub fn helper(x: Binary{8}) => Binary{8} = not(x);\n".to_owned(),
        )]);
        assert!(!report.ok);
        let e = report.error.as_ref().expect("a parse refusal");
        assert_eq!(e.kind, PhylumErrorKind::Parse);
        assert_eq!(e.site, "bad.myc", "parse site is the file label");
        assert_eq!(report.exit_code(), 2);
    }

    #[test]
    fn phylum_check_error_is_reported() {
        // A single nodule with a real check error (an unresolved call) — check_phylum refuses.
        let report = check_phylum_sources(&[(
            "c.myc".to_owned(),
            "nodule a;\nfn f() => Binary{8} = nope(0b0);\n".to_owned(),
        )]);
        assert!(!report.ok);
        assert_eq!(
            report.error.as_ref().map(|e| e.kind),
            Some(PhylumErrorKind::Check),
            "{:?}",
            report.error
        );
        assert_eq!(report.exit_code(), 3);
    }
}
