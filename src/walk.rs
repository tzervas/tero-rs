//! Small, deterministic filesystem helpers shared by the family extractors — a recursive
//! extension-filtered collector (sorted for reproducibility) and a repo-relative path normalizer.
//!
//! These mirror `mycelium_doc::lib_index`'s `collect_myc` / `repo_rel` shape (a directory walk +
//! path normalization is not a "parallel heuristic" — it is plumbing; `lib_index.rs` wrote its own
//! for exactly this reason). The one substantive parser this crate reuses rather than re-writes is
//! `mycelium_doc::corpus::ingest` (the markdown structure — see `docs.rs`).

use std::path::{Path, PathBuf};

/// Recursively collect files with extension `ext` under `root`, **sorted** for determinism. A
/// missing `root` is not an error — it yields an empty list (skip-graceful, like every gate here).
///
/// # Errors
/// Propagates the first filesystem error under a present `root` — never a silent skip of a readable
/// tree.
pub fn collect_ext(root: &Path, ext: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// `path` made repo-relative with `/` separators (the `mycelium_doc` `repo_rel` twin — a one-line
/// normalization, deliberately duplicated rather than reaching across the crate boundary for a
/// private helper).
#[must_use]
pub fn repo_rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

/// Whether a repo-relative path is an excluded fixture/reject/target artifact (mirrors
/// `mycelium_doc::build::is_excluded` — projecting must-fail fixtures or the reject corpus would
/// wrongly pollute the index). A tiny, documented duplication (that helper is private).
#[must_use]
pub fn is_excluded(rel: &str) -> bool {
    rel.contains("/tests/fixtures/")
        || rel.contains("/fixtures/")
        || rel.contains("/reject/")
        || rel.contains("/target/")
}
