//! The `docs/` and `research/` markdown families.
//!
//! **DRY (house rule #5):** the markdown *structure* — the section tree, stable anchors, and
//! `file:line` provenance — is produced by `mycelium_doc::corpus::ingest`, the crate's existing
//! CommonMark-subset corpus parser, **not** a second parallel heuristic (M-1015 / DN-87 §2.1). This
//! module adds only what the doc-IR does not carry, because its consumer (the HTML/Typst doc site)
//! never needed it: the document's declared **status** and **guarantee tag** (scanned from the
//! `| **Status** |` / `| **Guarantee** |` metadata rows) and a one-line lead **summary**. Those
//! scans are new *metadata* extraction on top of the reused *structure* parse — the same
//! divergence-with-a-reason posture `lib_index::full_summary` documents.
//!
//! Honesty (G2): status/guarantee are the leading lattice keyword of the metadata cell, verbatim
//! from source; a doc family that should carry a status but does not is **flagged**, never assumed.

use mycelium_doc::corpus::{ingest, AnchorAlloc};
use mycelium_doc::{Level, Node, Payload, SourceKind};

use crate::model::{Family, Flagged, TeroIndexItem};
use crate::walk::{collect_ext, is_excluded, repo_rel};

/// The house-rule #3 status lattice keywords a doc `Status` row may lead with.
const STATUS_KEYWORDS: &[&str] = &[
    "Draft",
    "Proposed",
    "Accepted",
    "Enacted",
    "Superseded",
    "Resolved",
];

/// The guarantee lattice (`Exact ⊐ Proven ⊐ Empirical ⊐ Declared`, house rule #1).
const GUARANTEE_KEYWORDS: &[&str] = &["Exact", "Proven", "Empirical", "Declared"];

/// Index every markdown document under `docs/` (as [`Family::Doc`]) and `research/` (as
/// [`Family::Research`]) rooted at `repo_root`, appending rows + flags. A shared [`AnchorAlloc`]
/// keeps anchors globally unique across the whole corpus (collision-free deep links).
///
/// # Errors
/// Propagates the first filesystem error under a present root.
pub fn index_all(
    repo_root: &std::path::Path,
    alloc: &mut AnchorAlloc,
    items: &mut Vec<TeroIndexItem>,
    flagged: &mut Vec<Flagged>,
) -> std::io::Result<()> {
    index_tree(
        repo_root,
        &repo_root.join("docs"),
        Family::Doc,
        alloc,
        items,
        flagged,
    )?;
    index_tree(
        repo_root,
        &repo_root.join("research"),
        Family::Research,
        alloc,
        items,
        flagged,
    )?;
    Ok(())
}

/// Index one markdown tree under `root` into `family`.
fn index_tree(
    repo_root: &std::path::Path,
    root: &std::path::Path,
    family: Family,
    alloc: &mut AnchorAlloc,
    items: &mut Vec<TeroIndexItem>,
    flagged: &mut Vec<Flagged>,
) -> std::io::Result<()> {
    for path in collect_ext(root, "md")? {
        let rel = repo_rel(repo_root, &path);
        if is_excluded(&rel) || is_generated_index(&rel) {
            continue;
        }
        let src = std::fs::read_to_string(&path)?;
        let kind = classify(&rel);
        let node = ingest(&rel, &src, kind, alloc);
        emit_doc(&node, family, kind, &rel, &src, items, flagged);
    }
    Ok(())
}

/// Emit the document row + one row per section, from an ingested doc `node`.
fn emit_doc(
    node: &Node,
    family: Family,
    kind: SourceKind,
    rel: &str,
    src: &str,
    items: &mut Vec<TeroIndexItem>,
    flagged: &mut Vec<Flagged>,
) {
    let doc_id = doc_id(rel);
    let status = leading_keyword(src, "Status", STATUS_KEYWORDS);
    let guarantee = leading_keyword(src, "Guarantee", GUARANTEE_KEYWORDS);

    // A doc family that should declare a status but the index could not extract one — recorded,
    // never assumed (G2). Two DISTINCT reasons: the row is genuinely absent, vs. the row is present
    // but its value is not on the ratified lattice (Draft/Proposed/Accepted/Enacted/Superseded/
    // Resolved). Conflating them shipped a false "no Status row" reason for `Living`-status notes.
    if status.is_none() && matches!(kind, SourceKind::Rfc | SourceKind::Adr | SourceKind::Note) {
        let reason = match labeled_cell(src, "Status") {
            Some(cell) => format!(
                "{} document has a `| **Status** |` row but its value ({}) is not on the ratified \
                 lattice (Draft/Proposed/Accepted/Enacted/Superseded/Resolved) — status left unset \
                 (not coerced)",
                kind.as_str(),
                first_words(&cell, 4),
            ),
            None => format!(
                "{} document has no `| **Status** |` metadata row — status left unset (not \
                 invented)",
                kind.as_str()
            ),
        };
        flagged.push(Flagged {
            item: rel.to_owned(),
            reason,
        });
    }

    node.walk(&mut |n| match &n.payload {
        Payload::Document { .. } => {
            let doc_kind = match family {
                Family::Research => "record".to_owned(),
                _ => kind.as_str().to_owned(),
            };
            let mut item = TeroIndexItem::new(
                n.anchor.clone(),
                family,
                doc_kind,
                n.title.clone().unwrap_or_else(|| rel.to_owned()),
                rel.to_owned(),
                n.provenance.line,
            );
            item.id = doc_id.clone();
            item.status = status.clone();
            item.guarantee_tag = guarantee.clone();
            item.summary = lead_summary(n);
            items.push(item);
        }
        Payload::Section => {
            // A section IS its heading — the `title` column already carries it, so no separate
            // `summary` (which would duplicate it) and no status/guarantee of its own.
            let item = TeroIndexItem::new(
                n.anchor.clone(),
                family,
                "section",
                n.title.clone().unwrap_or_default(),
                rel.to_owned(),
                n.provenance.line,
            );
            items.push(item);
        }
        _ => {}
    });
}

/// The committed generated-index directories — this index's **own** output (`docs/tero-index/`) and
/// its **sibling** indices (`docs/api-index/`, `docs/lib-index/`). They are excluded from the docs
/// walk for two reasons: (1) self-indexing `docs/tero-index/` would break the drift-gate fixpoint
/// (regenerating changes the file that is itself being read); (2) the siblings are *referenced*, not
/// *duplicated* (M-1015) — their symbol domains are theirs to index, linked via
/// [`crate::model::SIBLING_INDICES`].
fn is_generated_index(rel: &str) -> bool {
    rel.starts_with("docs/tero-index/")
        || rel.starts_with("docs/api-index/")
        || rel.starts_with("docs/lib-index/")
}

/// Classify a repo-relative markdown path into its corpus family (mirrors
/// `mycelium_doc::build::classify`, which is `pub(crate)` — a tiny, documented duplication).
fn classify(rel: &str) -> SourceKind {
    if rel.contains("/rfcs/") {
        SourceKind::Rfc
    } else if rel.contains("/adr/") {
        SourceKind::Adr
    } else if rel.contains("/notes/") {
        SourceKind::Note
    } else if rel.contains("/devlog/") {
        SourceKind::Devlog
    } else if rel.contains("/spec/") {
        SourceKind::Spec
    } else {
        SourceKind::Other
    }
}

/// The document's own id (`RFC-0034`, `ADR-032`, `DN-87`), parsed from the filename stem prefix, or
/// `None` when the file is not an id-bearing decision doc.
pub(crate) fn doc_id(rel: &str) -> Option<String> {
    let stem = rel.rsplit('/').next().unwrap_or(rel);
    for prefix in ["RFC", "ADR", "DN"] {
        if let Some(rest) = stem.strip_prefix(prefix) {
            let rest = rest.strip_prefix('-')?;
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if !digits.is_empty() {
                return Some(format!("{prefix}-{digits}"));
            }
        }
    }
    None
}

/// The leading lattice keyword of a `| **<label>** | value |` metadata row (the first `keywords`
/// entry that appears as a whole word in the value cell). Verbatim from source; `None` when the row
/// or a keyword is absent.
pub(crate) fn leading_keyword(src: &str, label: &str, keywords: &[&str]) -> Option<String> {
    let cell = labeled_cell(src, label)?;
    // First keyword that appears as a whole alphanumeric token, scanning the cell left-to-right.
    let mut best: Option<(usize, &str)> = None;
    for kw in keywords {
        if let Some(pos) = whole_word_pos(&cell, kw) {
            if best.is_none_or(|(p, _)| pos < p) {
                best = Some((pos, kw));
            }
        }
    }
    best.map(|(_, kw)| kw.to_string())
}

/// The trimmed value cell of the first `| **<label>** | value |` metadata row, or `None` if no such
/// row exists (or its value cell is empty). Distinct from [`leading_keyword`] returning `None`: this
/// says the ROW is absent, not merely that its value carried no lattice keyword — so a "no Status
/// row" flag is never conflated with a "Status row present but off-lattice value" one (G2).
pub(crate) fn labeled_cell(src: &str, label: &str) -> Option<String> {
    let needle = format!("**{label}**");
    for line in src.lines() {
        let t = line.trim();
        if !t.starts_with('|') || !t.contains(&needle) {
            continue;
        }
        return t
            .split('|')
            .nth(2)
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .map(str::to_owned);
    }
    None
}

/// The first `n` whitespace-separated tokens of `s`, joined by a single space (with `…` appended if
/// truncated) — a bounded, deterministic excerpt for a flagged-reason message (never the whole,
/// possibly-long, markdown cell).
fn first_words(s: &str, n: usize) -> String {
    let toks: Vec<&str> = s.split_whitespace().collect();
    if toks.len() <= n {
        toks.join(" ")
    } else {
        format!("{}…", toks[..n].join(" "))
    }
}

/// The byte position of `word` in `hay` as a whole token (bounded by non-alphanumeric on both
/// sides), case-sensitive; `None` if absent. Prevents `Proven` matching inside `Proventest` etc.
fn whole_word_pos(hay: &str, word: &str) -> Option<usize> {
    let bytes = hay.as_bytes();
    let mut from = 0;
    while let Some(rel) = hay[from..].find(word) {
        let abs = from + rel;
        let before_ok = abs == 0 || !bytes[abs - 1].is_ascii_alphanumeric();
        let after = abs + word.len();
        let after_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return Some(abs);
        }
        from = abs + word.len();
        if from >= hay.len() {
            break;
        }
    }
    None
}

/// The document's one-line lead summary: the first meaningful line of the lead prose (the content
/// before the first section heading), squeezed to a single line and truncated. A metadata-table
/// line (`| … |`) is skipped — for a decision doc the lead is the `| Field | Value |` table, which
/// is not a summary — as is a leading blockquote marker. Verbatim-ish, never invented (G2); `None`
/// when the lead has no non-table prose line.
fn lead_summary(doc: &Node) -> Option<String> {
    for c in &doc.children {
        let (Payload::Prose { text }, Some(Level::Minimal)) = (&c.payload, c.level) else {
            continue;
        };
        let line = text.lines().find_map(|l| {
            // Strip a leading blockquote marker so a `> vision …` lead reads cleanly.
            let t = l.trim_start().trim_start_matches('>').trim_start();
            (!t.is_empty() && !t.starts_with('|')).then_some(t)
        });
        if let Some(line) = line {
            let s = one_line(line, 200);
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}

/// Collapse to a single line (first line, whitespace-squeezed) and truncate to `max` chars at a
/// word boundary, appending `…` when cut. A faithful excerpt of source text; embedded markdown-link
/// markup is reduced to its text ([`crate::model::strip_md_links`]) so a summary carries no
/// relocated/broken relative link into `INDEX.md`.
pub(crate) fn one_line(text: &str, max: usize) -> String {
    let text = crate::model::strip_md_links(text);
    let first = text.lines().next().unwrap_or(&text);
    let squeezed = first.split_whitespace().collect::<Vec<_>>().join(" ");
    if squeezed.chars().count() <= max {
        return squeezed;
    }
    let mut cut = String::new();
    for word in squeezed.split(' ') {
        if cut.chars().count() + word.chars().count() + 1 > max {
            break;
        }
        if !cut.is_empty() {
            cut.push(' ');
        }
        cut.push_str(word);
    }
    if cut.is_empty() {
        cut = squeezed.chars().take(max).collect();
    }
    cut.push('…');
    cut
}
