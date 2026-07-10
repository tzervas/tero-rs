//! The **project manifest** `mycelium-proj.toml` (M-359; spec §2) and a **minimal, auditable
//! TOML-subset reader**.
//!
//! The workspace keeps its external dependencies few and vetted (the `/security-review` ethos —
//! only pinned `serde`/`serde_json`/`blake3` in the toolchain crates). Rather than **add** a full
//! TOML crate (a new-dependency decision that is an ADR, not a build detail), this reads the
//! *subset* the manifest needs: `# comments`, `[table]` headers, and
//! single-line `key = value` where a value is a **basic string** (`"…"`), an **array** of values, an
//! **inline table** (`{ k = v, … }`), or a **boolean**. Anything outside the subset (a bare number, a
//! multi-line array, an unknown `[project]` key, an unknown `[project].kind`) is an **explicit** error
//! — never silently dropped or guessed (G2). It is honestly a *subset*, named as one; it is not a
//! conformant TOML parser.
//!
//! Only `[project]` is **typed and validated** in v0 (the fields headers inherit from); the optional
//! `[surface]`/`[dependencies]`/`[toolchain]`/`[spore]` tables are accepted but not interpreted yet
//! (their consumers are M-361). Metadata is **not** identity (ADR-003).

use crate::header::{is_iso_date, is_semver, is_spdx, is_url};

/// The shape of a Mycelium project (spec §2 — `[project].kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectKind {
    /// A library — a content-addressed `phylum`.
    Phylum,
    /// An executable program.
    Program,
    /// A single-file / small script.
    Script,
}

/// The typed `[project]` table (the v0 closed key set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Project {
    /// `name` — the project name (required).
    pub name: String,
    /// `kind` — `phylum` | `program` | `script` (required).
    pub kind: ProjectKind,
    /// `version` — semver release label.
    pub version: Option<String>,
    /// `license` — SPDX identifier.
    pub license: Option<String>,
    /// `authors`.
    pub authors: Option<Vec<String>>,
    /// `since` — first publication ISO date.
    pub since: Option<String>,
    /// `summary`.
    pub summary: Option<String>,
    /// `repository` — source URL.
    pub repository: Option<String>,
    /// `keywords` — discovery tags.
    pub keywords: Option<Vec<String>>,
    /// `lang` — the surface-language edition this project targets.
    pub lang: Option<String>,
    /// `certification` — the project-wide certification mode (RFC-0034 §6; M-790). This is the
    /// **phylum** tier of the `global > phylum > nodule` lattice (FLAG-B, `cert_scope.rs`); a nodule's
    /// `@certification` header overrides it. The value is the closed set `fast | balanced |
    /// certified`; an unknown word is an explicit error (G2). Metadata, not identity (ADR-003).
    pub certification: Option<mycelium_core::cert_mode::CertMode>,
}

/// The typed `[toolchain]` table (M-364): the optional pins the toolchain reads. v0 closed key set:
/// `format` (the formatter spelling/version — a **hard pin**, M-364 §10.3) and `lints` (the lint
/// profile). Unknown keys are explicit errors (G2). Metadata, not identity (ADR-003).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Toolchain {
    /// `format` — the formatter spelling/version pin (e.g. `"mycfmt-0"`). A hard pin: `mycfmt` refuses a
    /// mismatch rather than format with rules the project did not ask for (M-364 §10.3 / G2).
    pub format: Option<String>,
    /// `lints` — the lint profile (e.g. `"strict"`).
    pub lints: Option<String>,
}

/// The typed `[surface]` table (M-368): a phylum's **public exports** — the germination boundary. v0
/// closed key set: `exports` (a list of dotted nodule names). Metadata layer (ADR-003).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Surface {
    /// `exports` — the public nodule names a phylum germinates from.
    pub exports: Vec<String>,
}

/// One `[dependencies]` entry (M-368): another phylum, **content-addressed** (ADR-003) — pinned by
/// `hash` (authoritative) with a human `version` requirement. A `hash`-less dep is an explicit error at
/// publish (the spore build refuses an unpinned, non-reproducible input; G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    /// The dependency's local name (the `[dependencies]` key).
    pub name: String,
    /// `phylum` — the depended-on phylum's name.
    pub phylum: String,
    /// `version` — a human version requirement (e.g. `"^2"`), checked against the pinned hash's version.
    pub version: Option<String>,
    /// `hash` — the content-addressed pin (`blake3:…`); authoritative (ADR-003). **Parsed, not
    /// free-text** (DN-40 A3): a `ContentHash` is well-formed by construction (`Exact`), so a
    /// malformed pin is rejected at manifest-build time (an explicit `ManifestError`, never silent —
    /// G2) and can never flow downstream into a spore's identity edge.
    pub hash: Option<mycelium_core::ContentHash>,
}

/// The typed `[spore]` table (M-368): how the project publishes as a deployable (ADR-013). v0 closed key
/// set: `include` (what germinates; defaults to the public `[surface]`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SporeConfig {
    /// `include` — what germinates (e.g. `["surface"]`, or explicit nodule names).
    pub include: Vec<String>,
}

/// A parsed `mycelium-proj.toml` (v0: the typed `[project]` table + the optional `[toolchain]`,
/// `[surface]`, `[dependencies]`, and `[spore]` tables).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The required `[project]` table.
    pub project: Project,
    /// The optional `[toolchain]` pins (M-364; M-361). `None` when the table is absent.
    pub toolchain: Option<Toolchain>,
    /// The optional `[surface]` exports (M-368). `None` when the table is absent.
    pub surface: Option<Surface>,
    /// The `[dependencies]` (M-368); empty when the table is absent.
    pub dependencies: Vec<Dependency>,
    /// The optional `[spore]` packaging config (M-368). `None` when the table is absent.
    pub spore: Option<SporeConfig>,
}

/// An explicit manifest error (G2): a syntax error, an out-of-subset construct, or a bad value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestError {
    /// 1-based source line.
    pub line: u32,
    /// What is wrong.
    pub message: String,
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ManifestError {}

/// The closed v0 `[project]` key set.
const PROJECT_KEYS: &[&str] = &[
    "name",
    "kind",
    "version",
    "license",
    "authors",
    "since",
    "summary",
    "repository",
    "keywords",
    "lang",
    "certification",
];

/// A parsed TOML value (the supported subset).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Val {
    Str(String),
    Arr(Vec<Val>),
    Table(Vec<(String, Val)>),
    Bool(bool),
}

/// Parse a `mycelium-proj.toml` source into a [`Manifest`].
///
/// # Errors
/// Returns [`ManifestError`] on a syntax error, an out-of-subset construct, a missing/unknown
/// `[project]` key, or a malformed value.
pub fn parse_manifest(src: &str) -> Result<Manifest, ManifestError> {
    let mut current: Option<String> = None;
    // Collected `[project]` key→(value, line). Other tables stay accepted-but-uninterpreted (v0),
    // except `[toolchain]` — its first consumer is M-364 (`mycfmt` reads `[toolchain].format`).
    let mut project_kv: Vec<(String, Val, u32)> = Vec::new();
    let mut toolchain_kv: Vec<(String, Val, u32)> = Vec::new();
    let mut surface_kv: Vec<(String, Val, u32)> = Vec::new();
    let mut deps_kv: Vec<(String, Val, u32)> = Vec::new();
    let mut spore_kv: Vec<(String, Val, u32)> = Vec::new();
    let (mut saw_toolchain, mut saw_surface, mut saw_spore) = (false, false, false);
    // Track which tables have already been opened: a repeated `[table]` header silently merged its
    // keys into the first instance, which is last-wins-by-stealth. Reject it (G2, never-silent).
    let mut seen_tables: Vec<String> = Vec::new();

    for (idx, raw) in src.lines().enumerate() {
        let line_no = (idx + 1) as u32;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(table) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let name = table.trim().to_owned();
            // Only the v0-interpreted tables are single-instance-checked; other tables stay
            // accepted-but-uninterpreted (and so a duplicate of them is harmless / ignored anyway).
            if matches!(
                name.as_str(),
                "project" | "toolchain" | "surface" | "dependencies" | "spore"
            ) {
                if seen_tables.iter().any(|t| t == &name) {
                    return Err(ManifestError {
                        line: line_no,
                        message: format!(
                            "duplicate `[{name}]` table — each table may appear at most once (G2)"
                        ),
                    });
                }
                seen_tables.push(name.clone());
            }
            current = Some(name.clone());
            match name.as_str() {
                "toolchain" => saw_toolchain = true,
                "surface" => saw_surface = true,
                "spore" => saw_spore = true,
                _ => {}
            }
            continue;
        }
        let (key, rhs) = line.split_once('=').ok_or_else(|| ManifestError {
            line: line_no,
            message: format!("expected `key = value` or `[table]`, got {line:?}"),
        })?;
        let key = key.trim().to_owned();
        let val = parse_value(rhs.trim(), line_no)?;
        // Reject a duplicate key within the same table instance — a repeat would otherwise
        // last-wins silently (mirrors the header parser's `seen` set; G2, never-silent).
        let dup_in = |kv: &[(String, Val, u32)]| kv.iter().any(|(k, _, _)| k == &key);
        match current.as_deref() {
            Some("project") => {
                if dup_in(&project_kv) {
                    return Err(duplicate_key("project", &key, line_no));
                }
                project_kv.push((key, val, line_no));
            }
            Some("toolchain") => {
                if dup_in(&toolchain_kv) {
                    return Err(duplicate_key("toolchain", &key, line_no));
                }
                toolchain_kv.push((key, val, line_no));
            }
            Some("surface") => {
                if dup_in(&surface_kv) {
                    return Err(duplicate_key("surface", &key, line_no));
                }
                surface_kv.push((key, val, line_no));
            }
            Some("dependencies") => {
                if dup_in(&deps_kv) {
                    return Err(duplicate_key("dependencies", &key, line_no));
                }
                deps_kv.push((key, val, line_no));
            }
            Some("spore") => {
                if dup_in(&spore_kv) {
                    return Err(duplicate_key("spore", &key, line_no));
                }
                spore_kv.push((key, val, line_no));
            }
            // Other tables: accepted, not interpreted in v0.
            _ => {}
        }
    }

    let project = build_project(project_kv)?;
    let toolchain = saw_toolchain
        .then(|| build_toolchain(toolchain_kv))
        .transpose()?;
    let surface = saw_surface.then(|| build_surface(surface_kv)).transpose()?;
    let spore = saw_spore.then(|| build_spore(spore_kv)).transpose()?;
    let dependencies = build_dependencies(deps_kv)?;
    Ok(Manifest {
        project,
        toolchain,
        surface,
        dependencies,
        spore,
    })
}

/// Strip a trailing `#` comment that is **outside** a quoted string (single-line values only).
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Build the explicit error for a duplicate key within a single table instance (G2, never-silent).
/// `table` names the offending table and `key` the repeated key, so the diagnostic points at the
/// exact offending input rather than letting the repeat last-win silently.
fn duplicate_key(table: &str, key: &str, line: u32) -> ManifestError {
    ManifestError {
        line,
        message: format!(
            "duplicate `[{table}]` key `{key}` — each key may appear at most once (G2)"
        ),
    }
}

/// The closed v0 `[toolchain]` key set.
const TOOLCHAIN_KEYS: &[&str] = &["format", "lints"];

fn build_toolchain(kv: Vec<(String, Val, u32)>) -> Result<Toolchain, ManifestError> {
    let mut tc = Toolchain::default();
    for (key, val, line) in kv {
        if !TOOLCHAIN_KEYS.contains(&key.as_str()) {
            return Err(ManifestError {
                line,
                message: format!(
                    "unknown `[toolchain]` key `{key}` — the v0 set is closed: {} (G2)",
                    TOOLCHAIN_KEYS.join(", ")
                ),
            });
        }
        match key.as_str() {
            "format" => tc.format = Some(as_str(&val, "format", line)?),
            "lints" => tc.lints = Some(as_str(&val, "lints", line)?),
            _ => unreachable!("key membership checked above"),
        }
    }
    Ok(tc)
}

fn build_surface(kv: Vec<(String, Val, u32)>) -> Result<Surface, ManifestError> {
    let mut exports = None;
    for (key, val, line) in kv {
        match key.as_str() {
            "exports" => exports = Some(as_str_list(&val, "exports", line)?),
            other => {
                return Err(ManifestError {
                    line,
                    message: format!(
                        "unknown `[surface]` key `{other}` — the v0 set is closed: exports (G2)"
                    ),
                })
            }
        }
    }
    Ok(Surface {
        exports: exports.unwrap_or_default(),
    })
}

fn build_spore(kv: Vec<(String, Val, u32)>) -> Result<SporeConfig, ManifestError> {
    let mut include = None;
    for (key, val, line) in kv {
        match key.as_str() {
            "include" => include = Some(as_str_list(&val, "include", line)?),
            other => {
                return Err(ManifestError {
                    line,
                    message: format!(
                        "unknown `[spore]` key `{other}` — the v0 set is closed: include (G2)"
                    ),
                })
            }
        }
    }
    Ok(SporeConfig {
        include: include.unwrap_or_default(),
    })
}

/// The closed v0 `[dependencies]` inline-table key set.
const DEP_KEYS: &[&str] = &["phylum", "version", "hash"];

fn build_dependencies(kv: Vec<(String, Val, u32)>) -> Result<Vec<Dependency>, ManifestError> {
    let mut deps = Vec::with_capacity(kv.len());
    for (name, val, line) in kv {
        let Val::Table(pairs) = val else {
            return Err(ManifestError {
                line,
                message: format!(
                    "dependency `{name}` must be an inline table \
                     `{{ phylum = \"…\", version = \"…\", hash = \"blake3:…\" }}` (G2)"
                ),
            });
        };
        let (mut phylum, mut version, mut hash) = (None, None, None);
        for (k, v) in &pairs {
            if !DEP_KEYS.contains(&k.as_str()) {
                return Err(ManifestError {
                    line,
                    message: format!(
                        "unknown dependency key `{k}` in `{name}` — the v0 set is closed: {} (G2)",
                        DEP_KEYS.join(", ")
                    ),
                });
            }
            match k.as_str() {
                "phylum" => phylum = Some(as_str(v, "phylum", line)?),
                "version" => version = Some(as_str(v, "version", line)?),
                // Parse-don't-validate at the boundary that owns the input (DN-40 A3): the dependency
                // hash is the identity-bearing edge of a spore (ADR-003), so it is parsed into a typed
                // `ContentHash` here — a malformed pin is an explicit error (never-silent, G2), and a
                // well-formed `ContentHash` is `Exact` by construction.
                "hash" => hash = Some(as_content_hash(v, &name, line)?),
                _ => unreachable!("key membership checked above"),
            }
        }
        deps.push(Dependency {
            phylum: phylum.unwrap_or_else(|| name.clone()),
            name,
            version,
            hash,
        });
    }
    Ok(deps)
}

fn build_project(kv: Vec<(String, Val, u32)>) -> Result<Project, ManifestError> {
    if kv.is_empty() {
        return Err(ManifestError {
            line: 1,
            message: "no `[project]` table — a manifest needs at least `[project]` with `name` and `kind`".to_owned(),
        });
    }
    let mut name = None;
    let mut kind = None;
    let mut version = None;
    let mut license = None;
    let mut authors = None;
    let mut since = None;
    let mut summary = None;
    let mut repository = None;
    let mut keywords = None;
    let mut lang = None;
    let mut certification = None;

    for (key, val, line) in kv {
        if !PROJECT_KEYS.contains(&key.as_str()) {
            return Err(ManifestError {
                line,
                message: format!(
                    "unknown `[project]` key `{key}` — the v0 set is closed: {} (G2)",
                    PROJECT_KEYS.join(", ")
                ),
            });
        }
        match key.as_str() {
            "name" => name = Some(as_str(&val, "name", line)?),
            "kind" => kind = Some(as_kind(&as_str(&val, "kind", line)?, line)?),
            "version" => version = Some(checked_str(&val, "version", line, is_semver, "a semver")?),
            "license" => {
                license = Some(checked_str(
                    &val,
                    "license",
                    line,
                    is_spdx,
                    "a recognised SPDX id/expression",
                )?)
            }
            "since" => {
                since = Some(checked_str(
                    &val,
                    "since",
                    line,
                    is_iso_date,
                    "an ISO-8601 date",
                )?)
            }
            "repository" => {
                repository = Some(checked_str(&val, "repository", line, is_url, "a URL")?)
            }
            "summary" => summary = Some(as_str(&val, "summary", line)?),
            "lang" => lang = Some(as_str(&val, "lang", line)?),
            "certification" => {
                // RFC-0034 §6 — the closed mode set `fast | balanced | certified` (FLAG-A). An
                // unknown word is an explicit error (G2), never a silent default.
                let s = as_str(&val, "certification", line)?;
                certification = Some(
                    crate::cert_scope::parse_cert_mode(&s)
                        .map_err(|m| ManifestError { line, message: m })?,
                );
            }
            "authors" => authors = Some(as_str_list(&val, "authors", line)?),
            "keywords" => keywords = Some(as_str_list(&val, "keywords", line)?),
            _ => unreachable!("key membership checked above"),
        }
    }

    let project = Project {
        name: name.ok_or_else(|| ManifestError {
            line: 1,
            message: "`[project]` is missing the required `name`".to_owned(),
        })?,
        kind: kind.ok_or_else(|| ManifestError {
            line: 1,
            message: "`[project]` is missing the required `kind` (phylum | program | script)"
                .to_owned(),
        })?,
        version,
        license,
        authors,
        since,
        summary,
        repository,
        keywords,
        lang,
        certification,
    };
    Ok(project)
}

fn as_kind(s: &str, line: u32) -> Result<ProjectKind, ManifestError> {
    match s {
        "phylum" => Ok(ProjectKind::Phylum),
        "program" => Ok(ProjectKind::Program),
        "script" => Ok(ProjectKind::Script),
        other => Err(ManifestError {
            line,
            message: format!("`kind` must be `phylum`, `program`, or `script`, got {other:?} (G2)"),
        }),
    }
}

fn as_str(val: &Val, key: &str, line: u32) -> Result<String, ManifestError> {
    match val {
        Val::Str(s) => Ok(s.clone()),
        _ => Err(ManifestError {
            line,
            message: format!("`{key}` must be a string"),
        }),
    }
}

/// Parse a dependency `hash` into a typed [`mycelium_core::ContentHash`] (DN-40 A3 — parse,
/// don't validate). The value must be a string and a well-formed content address (`<algo>:<digest>`,
/// `ContentHash::parse`); for the kernel's fixed algorithm (`blake3`, M-103) the digest must
/// additionally be a real digest — exactly **64 lowercase hex** (what `blake3::hash().to_hex()`
/// emits), so a shape-valid-but-bogus stub like `"blake3:abc"` is rejected, not just `"blake3:"`.
/// A malformed pin is an **explicit** `ManifestError` naming the offending dependency and the bad
/// value (never silent — G2); a well-formed address is `Exact` by construction.
fn as_content_hash(
    val: &Val,
    dep_name: &str,
    line: u32,
) -> Result<mycelium_core::ContentHash, ManifestError> {
    let s = as_str(val, "hash", line)?;
    let malformed = |detail: &str| ManifestError {
        line,
        message: format!(
            "dependency `{dep_name}` has a malformed content-address `hash` {s:?} — {detail}; the \
             pin is identity-bearing (ADR-003) and is checked, never accepted as free text \
             (DN-40 A3 / G2)"
        ),
    };
    // Shape: `<algo>:<digest>` with the address charset (`ContentHash::parse`).
    let h = mycelium_core::ContentHash::parse(&s)
        .ok_or_else(|| malformed("expected `<algo>:<digest>` (e.g. `blake3:<64-hex>`)"))?;
    // Algorithm-aware digest check, delegated to the single source of truth in `mycelium-core`
    // (`ContentHash::digest_well_formed` / `has_well_formed_digest`) rather than re-deriving the
    // 64-hex rule here (DRY — the canonical rule moves in one place). This is what makes a
    // shaped-but-bogus stub like `blake3:abc` an error rather than a silent accept. We keep the
    // shape-vs-digest split so the two failures carry distinct, granular messages.
    if !h.has_well_formed_digest() {
        return Err(malformed(
            "a `blake3` digest must be exactly 64 lowercase hex characters (M-103)",
        ));
    }
    Ok(h)
}

fn checked_str(
    val: &Val,
    key: &str,
    line: u32,
    ok: fn(&str) -> bool,
    want: &str,
) -> Result<String, ManifestError> {
    let s = as_str(val, key, line)?;
    if ok(&s) {
        Ok(s)
    } else {
        Err(ManifestError {
            line,
            message: format!(
                "`{key}` value {s:?} is not {want} (checked, never fabricated — VR-5)"
            ),
        })
    }
}

fn as_str_list(val: &Val, key: &str, line: u32) -> Result<Vec<String>, ManifestError> {
    match val {
        Val::Arr(items) => items.iter().map(|v| as_str(v, key, line)).collect(),
        _ => Err(ManifestError {
            line,
            message: format!("`{key}` must be an array of strings"),
        }),
    }
}

// --- the minimal value scanner (single-line) ---

/// The maximum nesting depth of arrays/inline-tables the v0 reader will descend (DoS bound, G2).
/// A manifest value (a dependency table, a small export list) is shallow by design; anything deeper
/// is refused explicitly rather than recursed-into and risking stack exhaustion on adversarial input.
const MAX_VALUE_DEPTH: u32 = 16;

/// The maximum number of elements one array may hold (DoS bound, G2). Manifest arrays are small
/// (authors, keywords, exports); a pathological list is refused, never silently truncated.
const MAX_ARRAY_ELEMS: usize = 1024;

/// The maximum number of key/value pairs one inline table may hold (DoS bound, G2).
const MAX_TABLE_PAIRS: usize = 1024;

fn parse_value(s: &str, line: u32) -> Result<Val, ManifestError> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let v = scan_value(&chars, &mut i, line, 0)?;
    skip_ws(&chars, &mut i);
    if i != chars.len() {
        return Err(ManifestError {
            line,
            message: format!(
                "trailing characters after value: {:?}",
                chars[i..].iter().collect::<String>()
            ),
        });
    }
    Ok(v)
}

fn skip_ws(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

fn scan_value(chars: &[char], i: &mut usize, line: u32, depth: u32) -> Result<Val, ManifestError> {
    skip_ws(chars, i);
    match chars.get(*i) {
        Some('"') => scan_string(chars, i, line).map(Val::Str),
        Some('[') => scan_array(chars, i, line, depth),
        Some('{') => scan_inline_table(chars, i, line, depth),
        Some('t') | Some('f') => scan_bool(chars, i, line),
        Some(c) => Err(ManifestError {
            line,
            message: format!(
                "unsupported value starting with {c:?} — the v0 manifest reader supports strings, \
                 arrays, inline tables, and booleans only (G2; not a full TOML parser)"
            ),
        }),
        None => Err(ManifestError {
            line,
            message: "expected a value after `=`".to_owned(),
        }),
    }
}

fn scan_string(chars: &[char], i: &mut usize, line: u32) -> Result<String, ManifestError> {
    *i += 1; // opening quote
    let mut out = String::new();
    while let Some(&c) = chars.get(*i) {
        match c {
            '"' => {
                *i += 1;
                return Ok(out);
            }
            '\\' => {
                *i += 1;
                match chars.get(*i) {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some(other) => {
                        return Err(ManifestError {
                            line,
                            message: format!("unsupported escape `\\{other}` in string"),
                        })
                    }
                    None => break,
                }
                *i += 1;
            }
            _ => {
                out.push(c);
                *i += 1;
            }
        }
    }
    Err(ManifestError {
        line,
        message: "unterminated string (missing closing `\"`)".to_owned(),
    })
}

/// The explicit error returned when a value nests past [`MAX_VALUE_DEPTH`] (DoS bound, G2).
fn too_deep(line: u32) -> ManifestError {
    ManifestError {
        line,
        message: format!(
            "value nests deeper than the v0 limit of {MAX_VALUE_DEPTH} (arrays/inline-tables; \
             a deeply-nested value is refused, never recursed-into — G2)"
        ),
    }
}

fn scan_array(chars: &[char], i: &mut usize, line: u32, depth: u32) -> Result<Val, ManifestError> {
    if depth >= MAX_VALUE_DEPTH {
        return Err(too_deep(line));
    }
    *i += 1; // '['
    let mut items = Vec::new();
    loop {
        skip_ws(chars, i);
        match chars.get(*i) {
            Some(']') => {
                *i += 1;
                return Ok(Val::Arr(items));
            }
            Some(',') => {
                *i += 1;
            }
            None => return Err(ManifestError {
                line,
                message:
                    "unterminated array (missing `]`; multi-line arrays are not in the v0 subset)"
                        .to_owned(),
            }),
            _ => {
                if items.len() >= MAX_ARRAY_ELEMS {
                    return Err(ManifestError {
                        line,
                        message: format!(
                            "array holds more than the v0 limit of {MAX_ARRAY_ELEMS} elements \
                             (refused, never silently truncated — G2)"
                        ),
                    });
                }
                items.push(scan_value(chars, i, line, depth + 1)?);
            }
        }
    }
}

fn scan_inline_table(
    chars: &[char],
    i: &mut usize,
    line: u32,
    depth: u32,
) -> Result<Val, ManifestError> {
    if depth >= MAX_VALUE_DEPTH {
        return Err(too_deep(line));
    }
    *i += 1; // '{'
    let mut pairs = Vec::new();
    loop {
        skip_ws(chars, i);
        match chars.get(*i) {
            Some('}') => {
                *i += 1;
                return Ok(Val::Table(pairs));
            }
            Some(',') => {
                *i += 1;
            }
            None => {
                return Err(ManifestError {
                    line,
                    message: "unterminated inline table (missing `}`)".to_owned(),
                })
            }
            _ => {
                if pairs.len() >= MAX_TABLE_PAIRS {
                    return Err(ManifestError {
                        line,
                        message: format!(
                            "inline table holds more than the v0 limit of {MAX_TABLE_PAIRS} pairs \
                             (refused, never silently truncated — G2)"
                        ),
                    });
                }
                let key = scan_bare_key(chars, i, line)?;
                skip_ws(chars, i);
                if chars.get(*i) != Some(&'=') {
                    return Err(ManifestError {
                        line,
                        message: format!("expected `=` after inline-table key `{key}`"),
                    });
                }
                *i += 1;
                let v = scan_value(chars, i, line, depth + 1)?;
                pairs.push((key, v));
            }
        }
    }
}

fn scan_bare_key(chars: &[char], i: &mut usize, line: u32) -> Result<String, ManifestError> {
    skip_ws(chars, i);
    let mut out = String::new();
    while let Some(&c) = chars.get(*i) {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
            *i += 1;
        } else {
            break;
        }
    }
    if out.is_empty() {
        return Err(ManifestError {
            line,
            message: "expected a bare key in inline table".to_owned(),
        });
    }
    Ok(out)
}

fn scan_bool(chars: &[char], i: &mut usize, line: u32) -> Result<Val, ManifestError> {
    let rest: String = chars[*i..].iter().collect();
    if rest.starts_with("true") {
        *i += 4;
        Ok(Val::Bool(true))
    } else if rest.starts_with("false") {
        *i += 5;
        Ok(Val::Bool(false))
    } else {
        Err(ManifestError {
            line,
            message: format!("unrecognised bare token {rest:?} (expected `true`/`false`)"),
        })
    }
}
