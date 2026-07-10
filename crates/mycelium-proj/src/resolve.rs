//! **Top-down inheritance resolution** (M-359; spec §4) with per-field provenance and an `EXPLAIN`.
//!
//! The *effective* header of a file is resolved most-specific-first — `in-file @key` > the
//! `mycelium-proj.toml` `[project]` table — and is **always inspectable** (no black box, G2): every
//! resolved field carries its [`Origin`], so "where did this license come from?" is answerable by
//! [`explain`]. Inherited fields (`version`/`license`/`authors`/`since`/`repository`/`keywords`) fall
//! back to the manifest; per-file fields (`updated`/`summary`/`deprecated`) never inherit. A local
//! value **overrides** the manifest (local wins) — that is an allowed override, not a conflict
//! (spec §4). Resolution produces *associated metadata* only — the content hash is unaffected
//! (ADR-003).
//!
//! Note (honest scope): the spec's middle tier — a nearest-ancestor *nodule-root* header — is a
//! multi-file concern; v0 resolves the single-file (`in-file > manifest`) case and names the
//! ancestor tier as deferred. Disallowed cross-tier conflicts (e.g. license-incompatible overrides)
//! are likewise a later compliance check (M-361), not fabricated here.

use crate::cert_scope::{cert_mode_word, resolve_mode, CertDecl, CertScope, ResolvedMode};
use crate::header::{Deprecated, HeaderFields, StructuredHeader};
use crate::manifest::Manifest;

/// Where a resolved field's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Set by an in-file `// @key:` line.
    Local,
    /// Inherited from the `mycelium-proj.toml` `[project]` table.
    ProjectManifest,
}

impl Origin {
    fn label(self) -> &'static str {
        match self {
            Origin::Local => "local",
            Origin::ProjectManifest => "mycelium-proj.toml",
        }
    }
}

/// A resolved field: its effective value and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved<T> {
    /// The effective value.
    pub value: T,
    /// Its provenance.
    pub origin: Origin,
}

/// The fully-resolved header — each inherited field annotated with its [`Origin`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedHeader {
    /// The nodule's dotted name (from the marker; `None` for a bare marker).
    pub name: Option<Vec<String>>,
    /// Effective `version`.
    pub version: Option<Resolved<String>>,
    /// Effective `license`.
    pub license: Option<Resolved<String>>,
    /// Effective `authors`.
    pub authors: Option<Resolved<Vec<String>>>,
    /// Effective `since`.
    pub since: Option<Resolved<String>>,
    /// Effective `repository`.
    pub repository: Option<Resolved<String>>,
    /// Effective `keywords`.
    pub keywords: Option<Resolved<Vec<String>>>,
    /// `updated` — per-file (always local; never inherited).
    pub updated: Option<String>,
    /// `summary` — per-file.
    pub summary: Option<String>,
    /// `deprecated` — per-file.
    pub deprecated: Option<Deprecated>,
    /// `matured` — the nodule/phylum is a matured (AOT) scope; RFC-0017; inherited top-down.
    pub matured: Option<Resolved<bool>>,
    /// `certification` — the active [`CertMode`](mycelium_core::cert_mode::CertMode), resolved
    /// **most-specific-wins** over the `global > phylum > nodule` lattice (RFC-0034 §6; M-790). The
    /// nodule `@certification` (header) overrides the phylum one (manifest); with neither, the project
    /// default [`CertMode::Fast`](mycelium_core::cert_mode::CertMode::Fast). Always present (it falls
    /// back to the default), and always carries its winning [`CertScope`] — never ambient (G2).
    pub certification: ResolvedMode,
}

/// Resolve a parsed header against an optional project manifest.
#[must_use]
pub fn resolve(header: &StructuredHeader, manifest: Option<&Manifest>) -> ResolvedHeader {
    let f: &HeaderFields = &header.fields;
    let p = manifest.map(|m| &m.project);

    // Inherited string field: local > manifest.
    let inherit_str = |local: &Option<String>, from_manifest: Option<&String>| {
        if let Some(v) = local {
            Some(Resolved {
                value: v.clone(),
                origin: Origin::Local,
            })
        } else {
            from_manifest.map(|v| Resolved {
                value: v.clone(),
                origin: Origin::ProjectManifest,
            })
        }
    };
    let inherit_list = |local: &Option<Vec<String>>, from_manifest: Option<&Vec<String>>| {
        if let Some(v) = local {
            Some(Resolved {
                value: v.clone(),
                origin: Origin::Local,
            })
        } else {
            from_manifest.map(|v| Resolved {
                value: v.clone(),
                origin: Origin::ProjectManifest,
            })
        }
    };
    let inherit_bool = |local: &Option<bool>, from_manifest: Option<bool>| {
        if let Some(v) = local {
            Some(Resolved {
                value: *v,
                origin: Origin::Local,
            })
        } else {
            from_manifest.map(|v| Resolved {
                value: v,
                origin: Origin::ProjectManifest,
            })
        }
    };

    ResolvedHeader {
        name: header.marker.name.clone(),
        version: inherit_str(&f.version, p.and_then(|p| p.version.as_ref())),
        license: inherit_str(&f.license, p.and_then(|p| p.license.as_ref())),
        authors: inherit_list(&f.authors, p.and_then(|p| p.authors.as_ref())),
        since: inherit_str(&f.since, p.and_then(|p| p.since.as_ref())),
        repository: inherit_str(&f.repository, p.and_then(|p| p.repository.as_ref())),
        keywords: inherit_list(&f.keywords, p.and_then(|p| p.keywords.as_ref())),
        // `matured` (RFC-0017) is *specified* to inherit top-down, but in this single-file resolver
        // both inheritance tiers are deferred: manifest-level `[project].matured` is not yet enacted
        // (R17-Q1) and multi-file ancestor-nodule resolution is out of scope here (see the module
        // header, §"deferred"). So `@matured` resolves **local-only** for now — the manifest source is
        // `None` and there is no ancestor tier; the inherited tiers land with those enactments.
        matured: inherit_bool(&f.matured, None),
        // Per-file: never inherited.
        updated: f.updated.clone(),
        summary: f.summary.clone(),
        deprecated: f.deprecated.clone(),
        // Certification mode (RFC-0034 §6; M-790): gather the `@certification` declarations from each
        // scope and resolve most-specific-wins via the shared `cert_scope` fold (reusing RFC-0012's
        // innermost-enclosing-wins mechanism). The nodule (in-file header) is the most-specific tier;
        // the manifest is the phylum tier (FLAG-B). `global` is reserved (no source in v0).
        certification: resolve_certification(f, p.and_then(|p| p.certification)),
    }
}

/// Gather the in-scope `@certification` declarations and resolve them most-specific-wins
/// (RFC-0034 §6) via [`resolve_mode`]. The header carries the **nodule** tier, the manifest the
/// **phylum** tier (FLAG-B in [`crate::cert_scope`]); `global` has no v0 source. With no declaration
/// at any scope the result is the project default ([`ResolvedMode::defaulted`]).
fn resolve_certification(
    header: &HeaderFields,
    manifest_mode: Option<mycelium_core::cert_mode::CertMode>,
) -> ResolvedMode {
    let mut decls: Vec<CertDecl> = Vec::new();
    if let Some(mode) = manifest_mode {
        decls.push(CertDecl {
            scope: CertScope::Phylum,
            mode,
        });
    }
    if let Some(mode) = header.certification {
        decls.push(CertDecl {
            scope: CertScope::Nodule,
            mode,
        });
    }
    resolve_mode(&decls)
}

/// The `EXPLAIN` of a resolved header — every field with its value and source, so nothing about the
/// metadata is ambient (G2). Stable, line-oriented, deterministic.
#[must_use]
pub fn explain(r: &ResolvedHeader) -> String {
    let mut out = String::new();
    let name = r
        .name
        .as_ref()
        .map_or_else(|| "(bare nodule)".to_owned(), |segs| segs.join("."));
    out.push_str(&format!("nodule: {name}\n"));

    let mut row = |field: &str, value: Option<String>, origin: Option<Origin>| match value {
        Some(v) => out.push_str(&format!(
            "  {field}: {v}  [{}]\n",
            origin.map_or("local", Origin::label)
        )),
        None => out.push_str(&format!("  {field}: —  [unset]\n")),
    };

    row(
        "version",
        r.version.as_ref().map(|x| x.value.clone()),
        r.version.as_ref().map(|x| x.origin),
    );
    row(
        "license",
        r.license.as_ref().map(|x| x.value.clone()),
        r.license.as_ref().map(|x| x.origin),
    );
    row(
        "authors",
        r.authors.as_ref().map(|x| x.value.join(", ")),
        r.authors.as_ref().map(|x| x.origin),
    );
    row(
        "since",
        r.since.as_ref().map(|x| x.value.clone()),
        r.since.as_ref().map(|x| x.origin),
    );
    row(
        "repository",
        r.repository.as_ref().map(|x| x.value.clone()),
        r.repository.as_ref().map(|x| x.origin),
    );
    row(
        "keywords",
        r.keywords.as_ref().map(|x| x.value.join(", ")),
        r.keywords.as_ref().map(|x| x.origin),
    );
    // Per-file fields are always local when present.
    row(
        "updated",
        r.updated.clone(),
        r.updated.as_ref().map(|_| Origin::Local),
    );
    row(
        "summary",
        r.summary.clone(),
        r.summary.as_ref().map(|_| Origin::Local),
    );
    let dep = r.deprecated.as_ref().map(|d| match d {
        Deprecated::Flag(b) => b.to_string(),
        Deprecated::Reason(s) => s.clone(),
    });
    row(
        "deprecated",
        dep,
        r.deprecated.as_ref().map(|_| Origin::Local),
    );
    // Certification mode (RFC-0034 §6): always present (defaults to `fast`); its source is a
    // `CertScope` (`global`/`phylum`/`nodule`) or `default` when no declaration was made — so it has
    // its own row rather than reusing the `Origin`-typed `row` closure. Never ambient (G2).
    let cert_src = r.certification.source.map_or("default", CertScope::label);
    out.push_str(&format!(
        "  certification: {}  [{cert_src}]\n",
        cert_mode_word(r.certification.mode)
    ));
    out
}
