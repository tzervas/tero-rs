//! The `gen-apiref` projection (output (d), fully automated — spec §4). API reference is **pure
//! projection from code + schemas + M-359 metadata**, no interpretive layer: a `.myc` nodule's header
//! ([`mycelium_proj::parse_header`]) and `fn` signatures, and the JSON schemas, become api-item IR
//! nodes. A missing `@summary` / schema `description` is an explicit [`Payload::ApiItem`] with
//! `summary: None` (rendered "undocumented") — **never invented** (the prose form of G2). The whole
//! `.myc` source is also captured as a *checked* example, so the §4.1 checked-examples lint type-checks
//! the real, dogfooded code (T7.1/T7.5).

use mycelium_proj::parse_header;
use serde_json::Value;

use crate::corpus::AnchorAlloc;
use crate::ir::{Level, Node, Payload, Provenance, SourceKind};

/// Project a `.myc` source into a [`Payload::Document`] (`source_kind: api`) of api-item nodes.
///
/// Children: the nodule itself (signature + `@summary` or undocumented), one api-item per `fn`
/// signature (currently undocumented — the doc-comment surface is later, spec §4 note), and the whole
/// source as a *checked* example.
#[must_use]
pub fn project_nodule(path: &str, src: &str, alloc: &mut AnchorAlloc) -> Node {
    let nodule_name = nodule_name(src).unwrap_or_else(|| path_stem(path));
    let doc_anchor = alloc.alloc(None, &format!("api {nodule_name}"));

    // The nodule header's @summary, if the header parses and carries one. A malformed header is not
    // *our* error to raise (myc-check/myc-lint own it) — here it simply yields no summary (honest:
    // the item renders as undocumented rather than crashing the build).
    let summary = parse_header(src)
        .ok()
        .flatten()
        .and_then(|h| h.fields.summary);

    let mut children = Vec::new();
    children.push(Node::new(
        alloc.alloc(Some(&doc_anchor), "nodule"),
        Some(format!("nodule {nodule_name}")),
        Some(Level::Medium),
        Provenance {
            source: path.to_owned(),
            line: 1,
        },
        Payload::ApiItem {
            signature: Some(format!("nodule {nodule_name}")),
            summary,
        },
        vec![],
    ));

    // One api-item per `fn` signature. The summary is the contiguous `//` doc-comment block
    // immediately preceding the `fn` (M-736) — extracted from source, never invented; a `fn` with
    // no preceding comment stays `None` (rendered "undocumented", an explicit honest gap — G2).
    // The source is split into lines once here (not per `fn`) so projection stays O(#lines).
    let lines: Vec<&str> = src.lines().collect();
    for (sig, line) in fn_signatures(src) {
        let name = fn_name(&sig).unwrap_or_else(|| "fn".to_owned());
        children.push(Node::new(
            alloc.alloc(Some(&doc_anchor), &format!("fn {name}")),
            Some(sig.clone()),
            Some(Level::Detailed),
            Provenance {
                source: path.to_owned(),
                line,
            },
            Payload::ApiItem {
                signature: Some(sig),
                summary: preceding_doc(&lines, line),
            },
            vec![],
        ));
    }

    // The whole source as a checked example (it is real, type-checked code — §4.1 #4 / T7.1).
    children.push(Node::new(
        alloc.alloc(Some(&doc_anchor), "source"),
        Some("Source".to_owned()),
        Some(Level::Detailed),
        Provenance {
            source: path.to_owned(),
            line: 1,
        },
        Payload::Example {
            lang: "myc".to_owned(),
            source: src.to_owned(),
            checked: true,
        },
        vec![],
    ));

    Node::new(
        doc_anchor,
        Some(format!("nodule {nodule_name}")),
        None,
        Provenance {
            source: path.to_owned(),
            line: 1,
        },
        Payload::Document {
            source_kind: SourceKind::Api,
        },
        children,
    )
}

/// Project a JSON-schema file into a [`Payload::Document`] of api-item nodes (one per top-level
/// property). A property with no `description` is an explicit undocumented api-item.
#[must_use]
pub fn project_schema(path: &str, json: &str, alloc: &mut AnchorAlloc) -> Option<Node> {
    let v: Value = serde_json::from_str(json).ok()?;
    let title = v
        .get("title")
        .and_then(Value::as_str)
        .map_or_else(|| path_stem(path), str::to_owned);
    let doc_anchor = alloc.alloc(None, &format!("schema {title}"));

    let mut children = Vec::new();
    if let Some(desc) = v.get("description").and_then(Value::as_str) {
        children.push(Node::new(
            alloc.alloc(Some(&doc_anchor), "overview"),
            None,
            Some(Level::Minimal),
            Provenance {
                source: path.to_owned(),
                line: 1,
            },
            Payload::Prose {
                text: desc.to_owned(),
            },
            vec![],
        ));
    }
    if let Some(props) = v.get("properties").and_then(Value::as_object) {
        for (name, spec) in props {
            let ty = spec
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("object")
                .to_owned();
            let summary = spec
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned);
            children.push(Node::new(
                alloc.alloc(Some(&doc_anchor), &format!("field {name}")),
                Some(format!("{name}: {ty}")),
                Some(Level::Detailed),
                Provenance {
                    source: path.to_owned(),
                    line: 1,
                },
                Payload::ApiItem {
                    signature: Some(format!("{name}: {ty}")),
                    summary,
                },
                vec![],
            ));
        }
    }

    Some(Node::new(
        doc_anchor,
        Some(format!("schema {title}")),
        None,
        Provenance {
            source: path.to_owned(),
            line: 1,
        },
        Payload::Document {
            source_kind: SourceKind::Api,
        },
        children,
    ))
}

/// The dotted nodule name from a `nodule X.Y` declaration (or the `// nodule:` marker).
///
/// `pub(crate)`: reused by [`crate::lib_index`] (the `docs/lib-index/` M-1004 extractor) rather
/// than re-implemented — DRY, one nodule-name heuristic for both consumers.
pub(crate) fn nodule_name(src: &str) -> Option<String> {
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("nodule ") {
            // Every real `.myc` file spells this a `nodule X.Y;` *statement* (semicolon-terminated,
            // not the `{`-block this trim originally targeted) — found while building the M-1004
            // lib-index extractor, which would otherwise index every nodule as e.g. `std.cmp;`.
            // Fixed here (once, DRY) rather than stripped a second time in the caller.
            return Some(rest.trim().trim_end_matches(['{', ';', ' ']).to_owned());
        }
    }
    // Fall back to the marker comment.
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("// nodule:") {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

/// Extract `fn NAME(...) -> Ty` / `fn NAME(...) => Ty` signatures with their 1-based line numbers.
///
/// A parameter list may itself span several lines (e.g. `lib/std/text.myc::decode_two`, where the
/// closing `)` + return type + body `=` land a few lines below `fn decode_two(`) — found while
/// building the M-1004 lib-index extractor, which was truncating ~1.3% of `lib/`'s signatures to
/// their bare `fn NAME(` opening line. Handled by joining lines until the body-introducing bare `=`
/// is found (never open-ended: bounded by the file's own line count).
///
/// `pub(crate)`: reused by [`crate::lib_index`] (DRY — see [`nodule_name`]).
pub(crate) fn fn_signatures(src: &str) -> Vec<(String, u32)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        let Some(rest) = t.strip_prefix("fn ") else {
            i += 1;
            continue;
        };
        let start_line = (i + 1) as u32;
        let mut acc = rest.to_owned();
        let mut j = i;
        while body_separator(&acc) == acc.as_str() && j + 1 < lines.len() {
            j += 1;
            acc.push(' ');
            acc.push_str(lines[j].trim());
        }
        let sig = clean_join_spacing(body_separator(&acc).trim());
        out.push((format!("fn {sig}"), start_line));
        i = j + 1;
    }
    out
}

/// Undo the cosmetic artifact of joining indented continuation lines with a single space: a
/// multi-line parameter list's own opening `(`/closing `)` picks up a stray adjacent space (e.g.
/// `decode_two( b: Bytes, i: Binary{8} )` instead of `decode_two(b: Bytes, i: Binary{8})`). No
/// real `.myc` signature uses this spacing (checked over every `lib/` file), so this is a safe,
/// content-preserving cleanup, not a risk of eating meaningful whitespace.
fn clean_join_spacing(sig: &str) -> String {
    sig.replace("( ", "(").replace(" )", ")")
}

/// Split a `fn` line at its body-introducing bare `=`, keeping the return-type arrow `=>` intact.
///
/// Mycelium's real surface syntax (every `.myc` file under `lib/`) writes the return type with a
/// **fat arrow**: `fn f(x: T) => U = <body>`. A naive `split_once('=')` (the original M-736 cut)
/// finds the `=` **inside** `=>` first and truncates the signature there, silently dropping the
/// return type for every real `.myc` file — an accuracy bug, not a cosmetic one (found while
/// building the M-1004 `docs/lib-index/` extractor, which reuses this function). Fixed here (once,
/// DRY) rather than duplicated with a workaround in the caller: scan for the first `=` that is
/// **not** immediately followed by `>` (i.e. not part of `=>`); that is the body separator. A
/// signature with no such separator (e.g. truncated input) returns the line unchanged.
fn body_separator(rest: &str) -> &str {
    let bytes = rest.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        if b == b'=' && bytes.get(idx + 1) != Some(&b'>') {
            return &rest[..idx];
        }
    }
    rest
}

/// The contiguous `//` doc-comment block immediately above the `fn` at `fn_line` (1-based), over
/// the already-split source `lines`. The scan walks backward, joining `//` comment lines into one
/// summary, and stops at the first blank line, non-comment line, or header line (`// nodule…` /
/// `// @key:` are metadata, not doc prose). Returns `None` when the `fn` has no preceding comment —
/// an honest, explicit gap (never invented filler, G2). The text is taken verbatim from source, so
/// it always traces to its provenance. Takes a `&[&str]` so the caller splits the source once.
///
/// `pub(crate)`: reused by [`crate::lib_index`] (DRY — see [`nodule_name`]).
pub(crate) fn preceding_doc(lines: &[&str], fn_line: u32) -> Option<String> {
    if fn_line == 0 || (fn_line as usize) > lines.len() {
        return None;
    }
    let mut idx = (fn_line as usize) - 1; // 0-based index of the `fn` line itself
    let mut collected: Vec<String> = Vec::new();
    while idx > 0 {
        idx -= 1;
        let t = lines[idx].trim();
        let Some(rest) = t.strip_prefix("//") else {
            break; // a non-comment line ends the doc block
        };
        let rest = rest.trim();
        // Header lines and blank comments are not doc prose — they bound the block.
        if rest.is_empty() || rest.starts_with('@') || rest.starts_with("nodule") {
            break;
        }
        collected.push(rest.to_owned());
    }
    if collected.is_empty() {
        return None;
    }
    collected.reverse();
    // A `// ── <heading> ──…` block is a SECTION DIVIDER (a corpus-wide `.myc` convention —
    // introduces a group of items, e.g. `// ── Width-generic comparison helpers … ──────`), not
    // doc prose for the one item directly beneath it — and unlike a real doc comment, it's never
    // blank-line-separated from that item, so the backward scan above collects it. Verbatim source
    // text attributed to the wrong item is still misleading (found while building the M-1004
    // lib-index extractor's `type`-declaration summaries, which hit this far more often than the
    // pre-existing `fn` fixtures happened to); an explicit "undocumented" is more honest than a
    // wrong attribution (G2), so the whole block is discarded when its first line is a divider.
    if collected[0].starts_with('─') {
        return None;
    }
    Some(collected.join(" "))
}

/// The function name from a `fn NAME(...)` signature.
///
/// `pub(crate)`: reused by [`crate::lib_index`] (DRY — see [`nodule_name`]).
pub(crate) fn fn_name(sig: &str) -> Option<String> {
    let rest = sig.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// `pub(crate)`: reused by [`crate::lib_index`] (DRY — see [`nodule_name`]).
pub(crate) fn path_stem(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.rsplit_once('.').map_or(file, |(s, _)| s).to_owned()
}
