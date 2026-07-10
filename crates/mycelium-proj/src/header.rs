//! The **structured nodule header** (M-359; spec §3) — the `// @key: value` metadata lines that may
//! follow the required `// nodule:` marker (DN-06 §6, recognised by [`mycelium_l1::parse_nodule_header`]).
//!
//! The v0 key set is **closed** (spec §7.3, ratified 2026-06-16, extended 2026-06-18 by RFC-0017,
//! 2026-06-24 by RFC-0034 §6 — `@certification`):
//! `version`, `license`, `authors`, `since`, `updated`, `summary`, `repository`, `keywords`,
//! `deprecated`, `matured`, `certification`. An **unknown** `@key`, a
//! **duplicate** key, a **malformed** value (non-SPDX `@license`, non-ISO `@since`/`@updated`,
//! ill-formed `@version`) — each is an **explicit** error, never silently ignored or guessed
//! (G2 / VR-5). Metadata is **not** identity (ADR-003): nothing here perturbs a definition's content
//! hash; these are associated, queryable fields.

use mycelium_l1::{parse_nodule_header, NoduleHeader, NoduleHeaderError};

/// The closed v0 metadata key set (spec §7.3). Kept here as the single source of truth for the
/// parser, the linter, and the "unknown key" message.
pub const HEADER_KEYS: &[&str] = &[
    "version",
    "license",
    "authors",
    "since",
    "updated",
    "summary",
    "repository",
    "keywords",
    "deprecated",
    "matured",
    "certification",
];

/// A `@deprecated` value: a bare flag or a reason string (spec §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Deprecated {
    /// `@deprecated: true` / `false`.
    Flag(bool),
    /// `@deprecated: <reason>` — a free-text supersession reason.
    Reason(String),
}

/// The parsed `@key` metadata of a header (all optional; absent fields inherit per the resolver).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HeaderFields {
    /// `@version` — semver release label (inherits from `project.version`).
    pub version: Option<String>,
    /// `@license` — SPDX id (inherits from `project.license`).
    pub license: Option<String>,
    /// `@authors` — comma-separated (inherits from `project.authors`).
    pub authors: Option<Vec<String>>,
    /// `@since` — first publication ISO date (inherits from `project.since`).
    pub since: Option<String>,
    /// `@updated` — last update ISO date (per-file, author-maintained; never inherited).
    pub updated: Option<String>,
    /// `@summary` — one-line description (per-file).
    pub summary: Option<String>,
    /// `@repository` — source URL (inherits from `project.repository`).
    pub repository: Option<String>,
    /// `@keywords` — comma-separated discovery tags (inherits from `project.keywords`).
    pub keywords: Option<Vec<String>>,
    /// `@deprecated` — flags this nodule superseded (per-file; never inherited).
    pub deprecated: Option<Deprecated>,
    /// `@matured` — the nodule/phylum is a matured (AOT) scope; RFC-0017; inherited top-down.
    pub matured: Option<bool>,
    /// `@certification` — the active certification mode (RFC-0034 §6; M-790). The **most-specific**
    /// (nodule) tier of the `global > phylum > nodule` lattice; overrides the manifest's. The value
    /// is the closed set `fast | balanced | certified` ([`mycelium_core::cert_mode::CertMode`]); an
    /// unknown word is an explicit error (G2). Resolved by [`crate::cert_scope::resolve_mode`].
    pub certification: Option<mycelium_core::cert_mode::CertMode>,
}

/// A parsed structured header: the `// nodule:` marker plus its `@key` metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredHeader {
    /// The DN-06 §6 marker (its `name` is `None` for the bare `// nodule`).
    pub marker: NoduleHeader,
    /// The parsed metadata fields.
    pub fields: HeaderFields,
}

/// An explicit header error (G2): a malformed marker, an unknown/duplicate key, or a bad value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderError {
    /// 1-based source line of the offending construct.
    pub line: u32,
    /// What is wrong, in author-facing terms.
    pub message: String,
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for HeaderError {}

impl From<NoduleHeaderError> for HeaderError {
    fn from(e: NoduleHeaderError) -> Self {
        HeaderError {
            line: e.line,
            message: e.message,
        }
    }
}

/// Parse the structured header from `src`.
///
/// - `Ok(Some(h))` — the first non-blank line is a valid `// nodule:` marker; any contiguous
///   `// @key: value` lines that follow it are parsed and validated.
/// - `Ok(None)` — the file declares no nodule (the first non-blank line is not a marker).
/// - `Err(_)` — a malformed marker, an unknown/duplicate key, or a malformed value (G2).
///
/// # Errors
/// See [`HeaderError`].
pub fn parse_header(src: &str) -> Result<Option<StructuredHeader>, HeaderError> {
    let Some(marker) = parse_nodule_header(src)? else {
        return Ok(None);
    };
    let lines: Vec<&str> = src.lines().collect();
    // Locate the marker line (the first non-blank line — already validated by parse_nodule_header).
    let mut i = 0;
    while i < lines.len() && lines[i].trim().is_empty() {
        i += 1;
    }
    let mut fields = HeaderFields::default();
    let mut seen: Vec<String> = Vec::new();
    // Parse the contiguous run of `// @key: value` lines immediately after the marker.
    for (offset, raw) in lines.iter().enumerate().skip(i + 1) {
        let line_no = (offset + 1) as u32;
        let t = raw.trim();
        let Some(at) = t
            .strip_prefix("//")
            .map(str::trim)
            .and_then(|c| c.strip_prefix('@'))
        else {
            break; // end of the header block
        };
        let (key, val) = at.split_once(':').ok_or_else(|| HeaderError {
            line: line_no,
            message: format!("metadata line `// @{at}` must be `// @<key>: <value>`"),
        })?;
        let key = key.trim();
        let val = val.trim();
        if !HEADER_KEYS.contains(&key) {
            return Err(HeaderError {
                line: line_no,
                message: format!(
                    "unknown header key `@{key}` — the v0 key set is closed: {} (spec §7.3; G2)",
                    HEADER_KEYS.join(", ")
                ),
            });
        }
        if seen.iter().any(|k| k == key) {
            return Err(HeaderError {
                line: line_no,
                message: format!(
                    "duplicate header key `@{key}` — each key may appear at most once (G2)"
                ),
            });
        }
        set_field(&mut fields, key, val, line_no)?;
        seen.push(key.to_owned());
    }
    Ok(Some(StructuredHeader { marker, fields }))
}

fn set_field(
    fields: &mut HeaderFields,
    key: &str,
    val: &str,
    line: u32,
) -> Result<(), HeaderError> {
    let nonempty = |v: &str| -> Result<String, HeaderError> {
        if v.is_empty() {
            Err(HeaderError {
                line,
                message: format!("`@{key}` has no value"),
            })
        } else {
            Ok(v.to_owned())
        }
    };
    match key {
        "version" => {
            let v = nonempty(val)?;
            if !is_semver(&v) {
                return Err(bad(
                    line,
                    "version",
                    &v,
                    "a semver `MAJOR.MINOR.PATCH` (e.g. `1.2.0`)",
                ));
            }
            fields.version = Some(v);
        }
        "license" => {
            let v = nonempty(val)?;
            if !is_spdx(&v) {
                return Err(bad(line, "license", &v, "a recognised SPDX identifier or expression (e.g. `Apache-2.0`, `MIT OR Apache-2.0`)"));
            }
            fields.license = Some(v);
        }
        "since" => {
            let v = nonempty(val)?;
            if !is_iso_date(&v) {
                return Err(bad(line, "since", &v, "an ISO-8601 date `YYYY-MM-DD`"));
            }
            fields.since = Some(v);
        }
        "updated" => {
            let v = nonempty(val)?;
            if !is_iso_date(&v) {
                return Err(bad(line, "updated", &v, "an ISO-8601 date `YYYY-MM-DD`"));
            }
            fields.updated = Some(v);
        }
        "summary" => fields.summary = Some(nonempty(val)?),
        "repository" => {
            let v = nonempty(val)?;
            if !is_url(&v) {
                return Err(bad(
                    line,
                    "repository",
                    &v,
                    "a URL (e.g. `https://…` or `git@…`)",
                ));
            }
            fields.repository = Some(v);
        }
        "authors" => fields.authors = Some(parse_list(val, "authors", line)?),
        "keywords" => fields.keywords = Some(parse_list(val, "keywords", line)?),
        "deprecated" => fields.deprecated = Some(parse_deprecated(val, line)?),
        "matured" => {
            fields.matured = Some(match val {
                "true" => true,
                "false" => false,
                _ => {
                    return Err(bad(line, "matured", val, "a boolean `true` or `false`"));
                }
            });
        }
        "certification" => {
            // RFC-0034 §6: the closed mode set `fast | balanced | certified` (FLAG-A, cert_scope.rs).
            // An unknown word is an explicit error (G2) — never a silent default.
            fields.certification = Some(
                crate::cert_scope::parse_cert_mode(val)
                    .map_err(|m| HeaderError { line, message: m })?,
            );
        }
        _ => unreachable!("key membership checked by caller"),
    }
    Ok(())
}

fn bad(line: u32, key: &str, got: &str, want: &str) -> HeaderError {
    HeaderError {
        line,
        message: format!("`@{key}` value {got:?} is not {want} (declared metadata is checked, never fabricated — VR-5)"),
    }
}

fn parse_list(val: &str, key: &str, line: u32) -> Result<Vec<String>, HeaderError> {
    let items: Vec<String> = val
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    if items.is_empty() {
        return Err(HeaderError {
            line,
            message: format!(
                "`@{key}` is empty — list at least one comma-separated value, or omit the key"
            ),
        });
    }
    Ok(items)
}

fn parse_deprecated(val: &str, line: u32) -> Result<Deprecated, HeaderError> {
    match val {
        "true" => Ok(Deprecated::Flag(true)),
        "false" => Ok(Deprecated::Flag(false)),
        "" => Err(HeaderError {
            line,
            message: "`@deprecated` needs `true`/`false` or a reason string".to_owned(),
        }),
        reason => Ok(Deprecated::Reason(reason.trim_matches('"').to_owned())),
    }
}

// --- value validators (shared with the manifest reader) ---

/// A `YYYY-MM-DD` ISO-8601 calendar date with a plausible month/day (cheap, dependency-free; the
/// honest claim is *well-formed*, not *calendar-exact* — a leap-day check is not load-bearing here).
#[must_use]
pub fn is_iso_date(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    let (y, m, d) = (parts[0], parts[1], parts[2]);
    if y.len() != 4 || m.len() != 2 || d.len() != 2 {
        return false;
    }
    if !(y.chars().all(|c| c.is_ascii_digit())
        && m.chars().all(|c| c.is_ascii_digit())
        && d.chars().all(|c| c.is_ascii_digit()))
    {
        return false;
    }
    let month: u32 = m.parse().unwrap_or(0);
    let day: u32 = d.parse().unwrap_or(0);
    (1..=12).contains(&month) && (1..=31).contains(&day)
}

/// A `MAJOR.MINOR.PATCH` semver core, with an optional `-prerelease` and/or `+build` suffix.
#[must_use]
pub fn is_semver(s: &str) -> bool {
    let core = s.split(['+', '-']).next().unwrap_or("");
    let parts: Vec<&str> = core.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// A non-empty, URL-shaped string (scheme-prefixed or `git@`-style). Declared metadata: checked for
/// shape, not reachability (VR-5).
#[must_use]
pub fn is_url(s: &str) -> bool {
    s.starts_with("http://")
        || s.starts_with("https://")
        || s.starts_with("git@")
        || s.starts_with("ssh://")
        || s.starts_with("git://")
}

/// The v0 known-SPDX subset — common OSI/FSF identifiers. A `@license` must be one of these (or an
/// expression composed of them with `OR`/`AND`/`WITH`), else it is an explicit error (G2). The set
/// is deliberately modest; it grows by explicit decision (not silently).
pub const KNOWN_SPDX: &[&str] = &[
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1-only",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "Unlicense",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "Zlib",
    "BSL-1.0",
    "0BSD",
];

/// A recognised SPDX identifier or a simple expression over [`KNOWN_SPDX`] (operators `OR`/`AND`/
/// `WITH`; a `LicenseRef-…` custom reference is also accepted).
#[must_use]
pub fn is_spdx(s: &str) -> bool {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }
    for tok in tokens {
        let t = tok.trim_matches(['(', ')']);
        if t.is_empty() || matches!(t, "OR" | "AND" | "WITH") {
            continue;
        }
        if KNOWN_SPDX.contains(&t) || t.starts_with("LicenseRef-") {
            continue;
        }
        return false;
    }
    true
}
