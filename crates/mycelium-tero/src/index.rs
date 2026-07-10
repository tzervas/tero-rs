//! The top-level Layer-1 index build (M-1015 / DN-87 §2.1): run every corpus-family extractor over
//! the repo, collect their rows + never-silent flags, and sort into the canonical deterministic
//! order. Pure function of the on-disc corpus at a commit ⇒ byte-identical regeneration (the DN-87
//! §6.3 drift-gate contract), proved by the two-run test.
//!
//! Family coverage (each covered, its scope recorded — the M-1015 DoD): `docs/` + `research/`
//! markdown ([`crate::docs`]), `tools/github/issues.yaml` + `idmap.tsv` ([`crate::issues`]),
//! `CHANGELOG.md` ([`crate::changelog`]), `.claude/skills/*/SKILL.md` ([`crate::skills`]). The
//! `docs/api-index/` + `docs/lib-index/` outputs are *referenced* as sibling indices
//! ([`crate::model::SIBLING_INDICES`]), never re-indexed.

use std::path::Path;

use mycelium_doc::corpus::AnchorAlloc;

use crate::model::TeroIndexReport;
use crate::{changelog, docs, issues, skills};

/// Build the full tero-index report from the corpus rooted at `repo_root`.
///
/// Anchor allocation: the `docs`/`research` families share one [`AnchorAlloc`] (matching
/// `mycelium-doc`'s own globally-unique doc-anchor scheme); `changelog` and `skills` each namespace
/// their anchors (`cl--…` / `sk--…`) so no family's anchors can collide with another's; `issues`
/// use their own stable ids as anchors. The processing order is fixed, so allocation is
/// deterministic.
///
/// # Errors
/// Propagates the first filesystem error from any family walk over a present source tree — never a
/// silent skip of readable corpus.
pub fn build_tero_index(repo_root: &Path) -> std::io::Result<TeroIndexReport> {
    let mut report = TeroIndexReport::default();
    let mut doc_alloc = AnchorAlloc::new();
    let mut cl_alloc = AnchorAlloc::new();
    let mut sk_alloc = AnchorAlloc::new();

    docs::index_all(
        repo_root,
        &mut doc_alloc,
        &mut report.items,
        &mut report.flagged,
    )?;
    issues::index_all(repo_root, &mut report.items, &mut report.flagged)?;
    changelog::index_all(
        repo_root,
        &mut cl_alloc,
        &mut report.items,
        &mut report.flagged,
    )?;
    skills::index_all(
        repo_root,
        &mut sk_alloc,
        &mut report.items,
        &mut report.flagged,
    )?;

    report.sort();
    Ok(report)
}
