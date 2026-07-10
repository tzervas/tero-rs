//! The `CHANGELOG.md` family — one row per release header (`## …`) and per dated entry (`### …
//! (YYYY-MM-DD)`), in document order (newest-first, as the changelog is written).
//!
//! Honesty (G2): the entry's own leading id (`M-996`, `DN-87`, `RFC-0034`, …) and its date are
//! extracted verbatim from the header where present; a header with neither is still indexed (its
//! full text is the title), never dropped.

use mycelium_doc::corpus::AnchorAlloc;

use crate::model::{Family, Flagged, TeroIndexItem};

/// Index `CHANGELOG.md` at the repo root. `alloc` namespaces changelog anchors (`cl--<slug>`) so
/// they never collide with doc anchors. Skip-graceful: a missing file yields nothing.
///
/// # Errors
/// Propagates a filesystem error reading a present `CHANGELOG.md`.
pub fn index_all(
    repo_root: &std::path::Path,
    alloc: &mut AnchorAlloc,
    items: &mut Vec<TeroIndexItem>,
    _flagged: &mut Vec<Flagged>,
) -> std::io::Result<()> {
    let path = repo_root.join("CHANGELOG.md");
    if !path.exists() {
        return Ok(());
    }
    let rel = "CHANGELOG.md";
    let src = std::fs::read_to_string(&path)?;

    for (i, line) in src.lines().enumerate() {
        let (kind, text) = if let Some(t) = line.strip_prefix("### ") {
            ("entry", t.trim())
        } else if let Some(t) = line.strip_prefix("## ") {
            ("release", t.trim())
        } else {
            continue;
        };
        let anchor = alloc.alloc(Some("cl"), text);
        let mut item = TeroIndexItem::new(
            anchor,
            Family::Changelog,
            kind,
            text.to_owned(),
            rel.to_owned(),
            (i + 1) as u32,
        );
        item.id = leading_id(text);
        item.summary = Some(crate::model::strip_md_links(text));
        items.push(item);
    }
    Ok(())
}

/// The first corpus id token in a changelog header (`M-<n>`, `E<n>`/`E<n>-<n>`, `RFC-<n>`,
/// `ADR-<n>`, `DN-<n>`), or `None`. Whole-token matched so `M-996` is not found inside a word.
pub(crate) fn leading_id(text: &str) -> Option<String> {
    for tok in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')) {
        if is_corpus_id(tok) {
            return Some(tok.to_owned());
        }
    }
    None
}

/// Whether `tok` is a well-formed corpus id: a known prefix then `-`? then digits (`E` also allows
/// the bare `E39`/`E39-1` epic forms).
fn is_corpus_id(tok: &str) -> bool {
    for prefix in ["RFC-", "ADR-", "DN-", "M-"] {
        if let Some(rest) = tok.strip_prefix(prefix) {
            return !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit());
        }
    }
    if let Some(rest) = tok.strip_prefix('E') {
        // E39 or E39-1
        let ok = rest
            .split('-')
            .all(|seg| !seg.is_empty() && seg.chars().all(|c| c.is_ascii_digit()));
        return ok && rest.chars().next().is_some_and(|c| c.is_ascii_digit());
    }
    false
}
