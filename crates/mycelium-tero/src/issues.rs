//! The `tools/github/issues.yaml` (+ `tools/github/idmap.tsv`) family.
//!
//! **A purpose-built YAML *subset* reader, honestly named as one** — the same discipline as
//! `mycelium_doc::corpus`'s CommonMark subset and the manifest TOML subset reader. It is *not* a
//! general YAML parser: it reads exactly the shape `issues.yaml` uses — a top-level `issues:`
//! sequence of `- id:` entries with a fixed vocabulary of 4-space-indented scalar / inline-list /
//! block-list / block-scalar fields. This choice is deliberate (M-1015):
//!
//! - `serde_yaml` is **not** an in-workspace dependency and is upstream-unmaintained (a
//!   supply-chain/`cargo-deny` risk); every existing `issues.yaml` consumer in the repo is Python
//!   (PyYAML). Adding a Rust YAML crate for five known fields is disproportionate (KISS/YAGNI).
//! - Keeping the whole index in one Rust toolchain gives one determinism story and one drift gate.
//!
//! Honesty (G2): the reader never guesses. An entry it cannot parse (missing title, an
//! unrecognized field shape) is **flagged**, never silently dropped; a duplicate `id` (the
//! union-merge hazard, CLAUDE.md mitigation #2) is flagged, never silently deduped. The extracted
//! entry count is cross-checked against an independent `- id:` line count in the tests.

use std::collections::BTreeMap;
use std::path::Path;

use crate::docs::one_line;
use crate::model::{Family, Flagged, TeroIndexItem};

/// Index every entry in `tools/github/issues.yaml`, enriched with the GitHub issue number from
/// `tools/github/idmap.tsv` where present. Skip-graceful: a missing `issues.yaml` yields nothing.
///
/// # Errors
/// Propagates a filesystem error reading a present `issues.yaml` / `idmap.tsv`.
pub fn index_all(
    repo_root: &Path,
    items: &mut Vec<TeroIndexItem>,
    flagged: &mut Vec<Flagged>,
) -> std::io::Result<()> {
    let path = repo_root.join("tools/github/issues.yaml");
    if !path.exists() {
        return Ok(());
    }
    let rel = "tools/github/issues.yaml";
    let src = std::fs::read_to_string(&path)?;
    let idmap = load_idmap(repo_root)?;

    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    for entry in split_entries(&src) {
        let raw = parse_entry(&entry);
        if let Some(prev_line) = seen.insert(raw.id.clone(), entry.start_line) {
            flagged.push(Flagged {
                item: raw.id.clone(),
                reason: format!(
                    "duplicate id in issues.yaml (also at line {prev_line}) — both kept, not \
                     silently deduped (union-merge hazard, mitigation #2)"
                ),
            });
        }
        if raw.title.is_none() {
            flagged.push(Flagged {
                item: format!("{} (line {})", raw.id, entry.start_line),
                reason: "issue entry has no `title:` field — indexed under its id".to_owned(),
            });
        }

        let kind = if raw.is_epic { "epic" } else { "issue" };
        let mut item = TeroIndexItem::new(
            raw.id.clone(),
            Family::Issue,
            kind,
            raw.title.clone().unwrap_or_else(|| raw.id.clone()),
            rel.to_owned(),
            entry.start_line,
        );
        item.id = Some(raw.id.clone());
        item.status = raw.status;
        item.summary = raw.summary;
        item.epic = raw.epic;
        item.depends_on = raw.depends_on;
        item.doc_refs = raw.doc_refs;
        item.gh_issue = idmap.get(&raw.id).cloned();
        items.push(item);
    }
    Ok(())
}

/// A parsed issue entry (only the fields this index surfaces).
#[derive(Debug, Default)]
struct RawEntry {
    id: String,
    title: Option<String>,
    status: Option<String>,
    epic: Option<String>,
    depends_on: Vec<String>,
    doc_refs: Vec<String>,
    summary: Option<String>,
    is_epic: bool,
}

/// A raw entry's source slice + the 1-based line its `- id:` sits on.
struct EntrySlice<'a> {
    lines: Vec<&'a str>,
    start_line: u32,
}

/// Split the file into per-entry slices, delimited by top-level `  - id:` lines (exactly two
/// leading spaces — a `- id:` inside a deeper-indented `body` block never matches).
fn split_entries(src: &str) -> Vec<EntrySlice<'_>> {
    let lines: Vec<&str> = src.lines().collect();
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("  - id:"))
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(lines.len());
        out.push(EntrySlice {
            lines: lines[start..end].to_vec(),
            start_line: (start + 1) as u32,
        });
    }
    out
}

/// Parse one entry slice into a [`RawEntry`].
fn parse_entry(entry: &EntrySlice<'_>) -> RawEntry {
    // The `- id:` value is on the first line: `  - id: M-1015`.
    let id = entry
        .lines
        .first()
        .and_then(|l| l.split_once("id:"))
        .map(|(_, v)| dequote(v.trim()))
        .unwrap_or_default();
    let mut raw = RawEntry {
        id,
        ..RawEntry::default()
    };

    let mut i = 1;
    while i < entry.lines.len() {
        let line = entry.lines[i];
        // A field key sits at exactly 4-space indent: `    key: ...`.
        let Some(rest) = line.strip_prefix("    ") else {
            i += 1;
            continue;
        };
        if rest.starts_with(' ') {
            // Deeper indent handled inline by whichever block field consumed it; skip here.
            i += 1;
            continue;
        }
        let Some((key, value)) = rest.split_once(':') else {
            i += 1;
            continue;
        };
        let value = value.trim();
        match key {
            "title" => raw.title = Some(dequote(value)),
            "epic" => raw.epic = non_empty(dequote(value)),
            "labels" => {
                let labels = parse_inline_list(value);
                raw.status = labels
                    .iter()
                    .find_map(|l| l.strip_prefix("status:").map(str::to_owned));
                raw.is_epic = labels.iter().any(|l| l == "type:epic");
            }
            "depends_on" => {
                let (list, consumed) = parse_list(value, &entry.lines[i + 1..]);
                raw.depends_on = list;
                i += consumed;
            }
            "doc_refs" => {
                let (list, consumed) = parse_list(value, &entry.lines[i + 1..]);
                raw.doc_refs = list;
                i += consumed;
            }
            "body" => {
                raw.summary = first_block_line(&entry.lines[i + 1..]).map(|s| one_line(&s, 200));
            }
            _ => {}
        }
        i += 1;
    }
    raw
}

/// Parse a list field that is either inline (`[a, b]`, value non-empty) or a following block
/// sequence (`- item` lines indented deeper than the key). Returns the items and how many *extra*
/// lines the block form consumed (0 for the inline form).
fn parse_list(value: &str, following: &[&str]) -> (Vec<String>, usize) {
    if !value.is_empty() {
        return (parse_inline_list(value), 0);
    }
    let mut out = Vec::new();
    let mut consumed = 0;
    for line in following {
        let t = line.trim_start();
        if let Some(item) = t.strip_prefix("- ") {
            // Confirm it is deeper-indented than the 4-space key (a block-seq item, ≥6 spaces).
            if line.starts_with("      ") {
                out.push(dequote(item.trim()));
                consumed += 1;
                continue;
            }
        }
        break;
    }
    (out, consumed)
}

/// Split an inline `[a, b, c]` (or a bare `a, b`) list into trimmed, dequoted, non-empty items.
pub(crate) fn parse_inline_list(value: &str) -> Vec<String> {
    let inner = value.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(|s| dequote(s.trim()))
        .filter(|s| !s.is_empty())
        .collect()
}

/// The first non-blank line of a `body: |` block (lines indented deeper than the 4-space key),
/// trimmed. `None` when the block is empty or absent.
fn first_block_line(following: &[&str]) -> Option<String> {
    for line in following {
        // A block-scalar continuation is indented deeper than the key (≥5 spaces); a new
        // 4-space key or a shallower line ends the block.
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with("     ") {
            return Some(line.trim().to_owned());
        }
        break;
    }
    None
}

/// Strip a single pair of surrounding double quotes and unescape `\"`/`\\` (issues.yaml titles are
/// double-quoted); pass through an unquoted scalar unchanged.
pub(crate) fn dequote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        s.to_owned()
    }
}

fn non_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

/// Load `tools/github/idmap.tsv` → `task_id → issue_number`. Comment (`#`) and short lines are
/// skipped. Skip-graceful: a missing file yields an empty map.
///
/// # Errors
/// Propagates a filesystem error reading a present `idmap.tsv`.
fn load_idmap(repo_root: &Path) -> std::io::Result<BTreeMap<String, String>> {
    let path = repo_root.join("tools/github/idmap.tsv");
    let mut map = BTreeMap::new();
    if !path.exists() {
        return Ok(map);
    }
    let src = std::fs::read_to_string(&path)?;
    for line in src.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 2 && !cols[0].is_empty() && !cols[1].is_empty() {
            map.insert(cols[0].trim().to_owned(), cols[1].trim().to_owned());
        }
    }
    Ok(map)
}
