//! `docs/lib-index/` extraction (M-1004) — the `docs/api-index/` analogue for the self-hosted
//! `.myc` tree under `lib/`. Walks each phylum directory (`lib/std/`, `lib/compiler/`, and any
//! future sibling — `phylum_roots` discovers them, so a new phylum is never silently omitted),
//! extracts per-nodule symbol/doc info, and emits `docs/lib-index/{INDEX.md,index.json}` grouped
//! by phylum/nodule, mirroring the `docs/api-index/` shape (grep-friendly MD + machine JSON +
//! honesty tag + a never-silent `flagged` section, G2).
//!
//! **DRY, not a parallel heuristic:** the nodule-header scan, `fn`-signature extraction, and the
//! backward doc-comment joiner are the *exact same* `pub(crate)` functions [`crate::apiref`]
//! already uses for the corpus doc-IR (`nodule_name` / `fn_signatures` / `preceding_doc` /
//! `fn_name` / `path_stem`) — reused here, not reimplemented. Building this extractor surfaced a
//! real bug in that shared code (`fn_signatures` truncated every signature at Mycelium's `=>`
//! return-type arrow, since *every* real `.myc` file uses `=>`, not the `->` the pre-existing test
//! fixtures used) — fixed once in `apiref.rs`, not worked around here (see its `body_separator`).
//!
//! **What's genuinely new here** (not in `apiref.rs`, because its consumer — the corpus doc-IR
//! that feeds the HTML/Typst/JSON doc *site* — never needed it): `type NAME = Ctor1(...) |
//! Ctor2(...);` declaration extraction (single- or multi-line, with best-effort per-constructor
//! line attribution), and a forward-joining `full_summary` that recovers a multi-line `@summary`
//! header (`mycelium_proj::parse_header` only returns the first line — its own scan stops at the
//! first non-`@`-prefixed comment line, which is exactly the header's own continuation prose).
//! Both divergences are deliberate and documented, per the M-1004 issue's "if you must diverge,
//! record the divergence" instruction.
//!
//! Honesty (G2/VR-5): this is the same class of heuristic as `tools/docgen/code_index.py` — a
//! line/regex scan over source text, never a real parse; source is ground truth. Every extracted
//! item carries the same declared per-item `tag` ([`ITEM_TAG`]) — a uniform tag is exactly as
//! strong a claim as a shared heuristic basis supports; inventing finer per-item confidence would
//! be an unchecked upgrade (VR-5). An unhandled construct, a malformed header, or a file with no
//! nodule marker is a `flagged` entry, never a silent drop.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::apiref::{fn_name, fn_signatures, nodule_name, path_stem, preceding_doc};

/// The top-level honesty tag (mirrors `tools/docgen/code_index.py::HONESTY_TAG`, adapted to the
/// `.myc` source this extractor reads instead of a `cargo-public-api` snapshot).
///
/// No inline code-span backticks in this string: it is itself wrapped in a single backtick pair
/// when rendered into `INDEX.md`'s honesty line, and Markdown code spans don't nest — an inner
/// `` `.myc` `` produced `MD038` (spaces-inside-code-span) findings (M-1004 markdown-gate finding).
pub const HONESTY_TAG: &str = "Empirical/Declared — line/regex heuristic over .myc source \
(mirrors tools/docgen/code_index.py's approach one level up the stack); source is ground truth. \
Use this index to find where to Read, not as an authoritative reference.";

/// The per-item honesty tag every extracted row carries. The M-1004 DoD requires entries to carry
/// "file:line + honesty tag" (distinct from `docs/api-index/index.json`'s always-`null`
/// `guarantee_tag` placeholder) — every row here was produced by the *same* heuristic, so a
/// uniform tag is exactly as strong a claim as the extraction can support (VR-5: never invent
/// finer per-item confidence than the shared basis).
pub const ITEM_TAG: &str = "Empirical/Declared";

/// The directory under the repo root this extractor walks for phyla.
const LIB_DIR: &str = "lib";

/// One indexed row: a nodule marker, an `fn`, a `type`, or a `type`'s constructor.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LibIndexItem {
    /// The dotted, nodule-qualified symbol (e.g. `std.cmp::is_lt`, `std.cmp::Ordering::Lt`, or
    /// just the nodule name itself for a `kind: "nodule"` row).
    pub symbol: String,
    /// `"nodule"` | `"fn"` | `"type"` | `"ctor"`.
    pub kind: String,
    /// The phylum directory name under `lib/` (e.g. `"std"`, `"compiler"`).
    pub phylum: String,
    /// The dotted nodule name (e.g. `"std.cmp"`).
    pub nodule: String,
    /// Repo-relative source path.
    pub file: String,
    /// 1-based source line.
    pub line: u32,
    /// The signature/declaration text, verbatim from source (whitespace-joined for a multi-line
    /// declaration), or `None` when there is nothing to show beyond the symbol itself.
    pub signature: Option<String>,
    /// The doc-comment/summary text, verbatim from source; `None` is an explicit "undocumented"
    /// gap (never invented — G2), not a missing field.
    pub summary: Option<String>,
    /// The honesty tag ([`ITEM_TAG`]) — populated, not a placeholder.
    pub tag: String,
}

/// A construct the heuristic could not (or does not yet) extract — never silently dropped (G2).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Flagged {
    /// What the flag is about (a symbol, a nodule, or a `"<nodule> (line N)"` locator).
    pub item: String,
    /// Why it's flagged, in author-facing terms.
    pub reason: String,
}

/// The full build result: every extracted item plus every flagged gap, in stable sorted order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LibIndexReport {
    pub items: Vec<LibIndexItem>,
    pub flagged: Vec<Flagged>,
}

/// Build the full `docs/lib-index/` report by walking every phylum directory under `lib/`.
///
/// # Errors
/// Propagates the first filesystem error (with its path implied by the walk) — never a silent
/// skip of a present source.
pub fn build_lib_index(repo_root: &Path) -> std::io::Result<LibIndexReport> {
    let mut items = Vec::new();
    let mut flagged = Vec::new();

    for (phylum, dir) in phylum_roots(repo_root)? {
        for path in collect_myc(&dir)? {
            let rel = repo_rel(repo_root, &path);
            let src = std::fs::read_to_string(&path)?;
            index_file(&phylum, &rel, &src, &mut items, &mut flagged);
        }
    }

    items.sort_by(|a, b| {
        (&a.phylum, &a.nodule, a.line, &a.symbol).cmp(&(&b.phylum, &b.nodule, b.line, &b.symbol))
    });
    flagged.sort_by(|a, b| a.item.cmp(&b.item).then(a.reason.cmp(&b.reason)));
    Ok(LibIndexReport { items, flagged })
}

/// Write `docs/lib-index/index.json`.
///
/// # Errors
/// Any filesystem error creating the directory or writing the file.
pub fn write_json(report: &LibIndexReport, output_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    #[derive(Serialize)]
    struct Payload<'a> {
        generated: &'a str,
        items: &'a [LibIndexItem],
        flagged: &'a [Flagged],
    }
    let payload = Payload {
        generated: HONESTY_TAG,
        items: &report.items,
        flagged: &report.flagged,
    };
    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(output_dir.join("index.json"), json + "\n")
}

/// Write `docs/lib-index/INDEX.md`.
///
/// # Errors
/// Any filesystem error creating the directory or writing the file.
pub fn write_markdown(report: &LibIndexReport, output_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let mut out = String::new();
    out.push_str("# Mycelium Lib Index — the self-hosted `.myc` reference\n\n");
    out.push_str(&format!("> **Honesty:** `{HONESTY_TAG}`\n"));
    out.push_str("> Use the index to find where to `Read`, not as an authoritative reference.\n\n");

    let mut phyla: Vec<&str> = report.items.iter().map(|i| i.phylum.as_str()).collect();
    phyla.sort_unstable();
    phyla.dedup();

    for phylum in phyla {
        out.push_str(&format!("## {phylum}\n\n"));
        let mut nodules: Vec<&str> = report
            .items
            .iter()
            .filter(|i| i.phylum == phylum)
            .map(|i| i.nodule.as_str())
            .collect();
        nodules.sort_unstable();
        nodules.dedup();
        for nodule in nodules {
            out.push_str(&format!("### {nodule}\n\n"));
            out.push_str("| Symbol | Kind | File:Line | Signature | Summary | Tag |\n");
            out.push_str("|---|---|---|---|---|---|\n");
            for item in report
                .items
                .iter()
                .filter(|i| i.phylum == phylum && i.nodule == nodule)
            {
                let symbol = format!("`{}`", md_escape(&item.symbol));
                let file_line = format!("`{}:{}`", item.file, item.line);
                let sig = item
                    .signature
                    .as_deref()
                    .map(|s| format!("`{}`", md_escape(s)))
                    .unwrap_or_else(|| "—".to_owned());
                let summary = item
                    .summary
                    .as_deref()
                    .map(md_escape_prose)
                    .unwrap_or_else(|| "—".to_owned());
                out.push_str(&format!(
                    "| {symbol} | {} | {file_line} | {sig} | {summary} | {} |\n",
                    item.kind, item.tag
                ));
            }
            out.push('\n');
        }
    }

    out.push_str("## Flagged items\n\n");
    out.push_str(
        "Constructs/gaps the heuristic could not (or does not yet) extract (G2: never silently \
         dropped):\n\n",
    );
    if report.flagged.is_empty() {
        out.push_str("*(none)*\n\n");
    } else {
        out.push_str("| Item | Reason |\n|---|---|\n");
        for f in &report.flagged {
            out.push_str(&format!(
                "| `{}` | {} |\n",
                md_escape(&f.item),
                md_escape_prose(&f.reason)
            ));
        }
        out.push('\n');
    }

    // Exactly one trailing newline, no dangling blank line before EOF (matches
    // `docs/api-index/INDEX.md`'s convention; a stray blank line here tripped `MD012`).
    let out = out.trim_end().to_owned() + "\n";
    std::fs::write(output_dir.join("INDEX.md"), out)
}

/// Escape `|` — safe inside an already-backtick-wrapped code-span cell (a code span already
/// suppresses Markdown emphasis, so `*`/`_` need no escaping there; `|` still must be escaped or
/// it corrupts the table's own column syntax).
pub(crate) fn md_escape(s: &str) -> String {
    s.replace('|', "\\|")
}

/// Escape `|`/`*` for a cell rendered as **plain prose** (summary text, a flagged-item reason) —
/// content that is NOT wrapped in a code span. `.myc` doc-comments are copied verbatim from source
/// (never invented, G2) and occasionally contain literal math notation like `3^m * carry`;
/// unescaped, a linter reads `* c` as emphasis markup (M-1004's `MD037` finding). Deliberately does
/// **not** escape `_` — CommonMark/GFM already suppresses intraword `_` emphasis (`snake_case_name`
/// never renders as emphasis), and identifiers with underscores are extremely common in this
/// prose; escaping them would insert a literal backslash that breaks a plain-text grep for the
/// identifier — the opposite of this index's grep-friendly purpose. `index.json` is unaffected
/// (JSON, not Markdown — no escaping needed there).
pub(crate) fn md_escape_prose(s: &str) -> String {
    s.replace('|', "\\|").replace('*', "\\*")
}

// ── discovery / filesystem ──────────────────────────────────────────────────────────────────────

/// Discover phylum roots under `lib/` (sorted, deterministic): every immediate subdirectory. A new
/// phylum directory is picked up automatically — the extractor never needs to be told about it.
pub(crate) fn phylum_roots(repo_root: &Path) -> std::io::Result<Vec<(String, PathBuf)>> {
    let lib = repo_root.join(LIB_DIR);
    let mut out = Vec::new();
    if !lib.exists() {
        return Ok(out);
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&lib)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        let Some(name) = dir.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        out.push((name, dir));
    }
    Ok(out)
}

/// Recursively collect `.myc` files under `root`, sorted for determinism.
pub(crate) fn collect_myc(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("myc") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// `path` made repo-relative with `/` separators (the `build.rs::repo_rel` twin; not reused
/// directly since that helper is private to `build.rs` and this module has no other need of it —
/// a one-line duplication, not a parallel heuristic).
pub(crate) fn repo_rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

// ── per-file extraction ─────────────────────────────────────────────────────────────────────────

/// Index one `.myc` file: the nodule marker, every `fn`, every `type` + its constructors, and any
/// unrecognised top-level construct (flagged, never dropped).
pub(crate) fn index_file(
    phylum: &str,
    rel: &str,
    src: &str,
    items: &mut Vec<LibIndexItem>,
    flagged: &mut Vec<Flagged>,
) {
    let nodule = nodule_name(src).unwrap_or_else(|| path_stem(rel));
    if nodule_name(src).is_none() {
        flagged.push(Flagged {
            item: rel.to_owned(),
            reason: "no `// nodule:` marker / `nodule X.Y;` declaration found — grouped by \
                filename instead"
                .to_owned(),
        });
    }

    let summary = full_summary(src);
    if let Err(e) = mycelium_proj::parse_header(src) {
        flagged.push(Flagged {
            item: format!("{nodule} (header)"),
            reason: format!(
                "header metadata parse error at line {}: {} — @summary/@version/etc. may be \
                 unavailable; fn/type extraction still ran",
                e.line, e.message
            ),
        });
    }

    items.push(LibIndexItem {
        symbol: nodule.clone(),
        kind: "nodule".to_owned(),
        phylum: phylum.to_owned(),
        nodule: nodule.clone(),
        file: rel.to_owned(),
        line: nodule_decl_line(src),
        signature: Some(format!("nodule {nodule}")),
        summary,
        tag: ITEM_TAG.to_owned(),
    });

    let lines: Vec<&str> = src.lines().collect();
    for (sig, line) in fn_signatures(src) {
        let name = fn_name(&sig).unwrap_or_else(|| "fn".to_owned());
        items.push(LibIndexItem {
            symbol: format!("{nodule}::{name}"),
            kind: "fn".to_owned(),
            phylum: phylum.to_owned(),
            nodule: nodule.clone(),
            file: rel.to_owned(),
            line,
            signature: Some(sig),
            summary: preceding_doc(&lines, line),
            tag: ITEM_TAG.to_owned(),
        });
    }

    let (decls, problems) = type_declarations(src);
    for decl in decls {
        items.push(LibIndexItem {
            symbol: format!("{nodule}::{}", decl.name),
            kind: "type".to_owned(),
            phylum: phylum.to_owned(),
            nodule: nodule.clone(),
            file: rel.to_owned(),
            line: decl.start_line,
            signature: Some(decl.text.clone()),
            summary: preceding_doc(&lines, decl.start_line),
            tag: ITEM_TAG.to_owned(),
        });
        for (ctor_name, ctor_text, ctor_line) in &decl.ctors {
            items.push(LibIndexItem {
                symbol: format!("{nodule}::{}::{ctor_name}", decl.name),
                kind: "ctor".to_owned(),
                phylum: phylum.to_owned(),
                nodule: nodule.clone(),
                file: rel.to_owned(),
                line: *ctor_line,
                signature: Some(ctor_text.clone()),
                summary: None,
                tag: ITEM_TAG.to_owned(),
            });
        }
    }
    for (name, line, reason) in problems {
        flagged.push(Flagged {
            item: format!("{nodule}::{name}"),
            reason: format!("{reason} (starting line {line})"),
        });
    }

    for (kw, line) in unrecognized_top_level(src) {
        flagged.push(Flagged {
            item: format!("{nodule} (line {line})"),
            reason: format!(
                "unextracted top-level construct `{kw}` at line {line} — only nodule/fn/type are \
                 indexed today; not silently dropped, just out of scope (no `lib/` nodule uses \
                 this DN-26 `Item` form yet)"
            ),
        });
    }
}

/// The 1-based line of the file's nodule declaration (the `nodule X.Y;` statement if present, else
/// the `// nodule:` marker comment, else `1` as an honest last resort). Mirrors `apiref::nodule_name`'s
/// two-pass scan so the reported line matches the name it found.
pub(crate) fn nodule_decl_line(src: &str) -> u32 {
    for (i, line) in src.lines().enumerate() {
        if line.trim().starts_with("nodule ") {
            return (i + 1) as u32;
        }
    }
    for (i, line) in src.lines().enumerate() {
        if line.trim().starts_with("// nodule:") {
            return (i + 1) as u32;
        }
    }
    1
}

/// The full (possibly multi-line) `@summary` text: the structured header's first line plus any
/// contiguous, non-`@`, non-blank `//` continuation lines immediately following it.
/// `mycelium_proj::parse_header`'s own scan stops at the first such line (it only consumes
/// contiguous `// @key: value` lines), so a multi-line summary — common in `lib/` headers, e.g.
/// `lib/compiler/ambient.myc`'s four-line `@summary` — is otherwise silently truncated to its
/// first line. Mirrors `apiref::preceding_doc`'s backward joiner, scanning forward instead — a
/// deliberate, documented divergence from `apiref.rs` (its consumer only ever uses the first-line
/// summary, so it never needed this). Verbatim from source, never invented (G2); `None` when there
/// is no `@summary` line at all.
pub(crate) fn full_summary(src: &str) -> Option<String> {
    let lines: Vec<&str> = src.lines().collect();
    let start = lines.iter().position(|line| {
        line.trim()
            .strip_prefix("//")
            .map(str::trim)
            .is_some_and(|c| c.starts_with("@summary:"))
    })?;
    let first = lines[start]
        .trim()
        .strip_prefix("//")?
        .trim()
        .strip_prefix("@summary:")?
        .trim()
        .to_owned();
    let mut parts = vec![first];
    for line in &lines[start + 1..] {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("//") else {
            break;
        };
        let rest = rest.trim();
        if rest.is_empty() || rest.starts_with('@') || rest.starts_with("nodule") {
            break;
        }
        parts.push(rest.to_owned());
    }
    let joined = parts.join(" ");
    (!joined.is_empty()).then_some(joined)
}

// ── `type` declaration extraction (new — not in `apiref.rs`; see the module doc) ────────────────

/// One `type NAME[...] = Ctor1(...) | Ctor2(...);` declaration.
pub(crate) struct TypeDecl {
    pub(crate) name: String,
    pub(crate) start_line: u32,
    /// The whole declaration, source lines whitespace-joined, trailing `;` stripped — e.g.
    /// `type Ordering = Lt | Eq | Gt`.
    pub(crate) text: String,
    /// Each top-level (depth-0, `|`-separated) constructor: (name, full text, best-effort line).
    pub(crate) ctors: Vec<(String, String, u32)>,
}

/// Extract every top-level `type NAME = ...;` declaration. Terminated by the first line (from the
/// `type` line onward) whose trimmed text ends with `;` — Mycelium constructor argument lists
/// don't themselves contain `;`, so this is unambiguous over every real `lib/` file (checked by
/// hand across `lib/std/` + `lib/compiler/`). Returns the parsed declarations plus a
/// `(name, start_line, reason)` list for any declaration the heuristic could not close (never
/// silently dropped — G2; the caller turns these into `Flagged` entries).
pub(crate) fn type_declarations(src: &str) -> (Vec<TypeDecl>, Vec<(String, u32, String)>) {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut problems = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        let Some(after_kw) = t.strip_prefix("type ") else {
            i += 1;
            continue;
        };
        let start_line = (i + 1) as u32;
        let name: String = after_kw
            .trim_start()
            .chars()
            .take_while(|c| !matches!(c, '=' | '[') && !c.is_whitespace())
            .collect();

        let mut block: Vec<(&str, u32)> = vec![(lines[i], start_line)];
        let mut j = i;
        let mut closed = lines[i].trim_end().ends_with(';');
        while !closed && j + 1 < lines.len() {
            j += 1;
            block.push((lines[j], (j + 1) as u32));
            closed = lines[j].trim_end().ends_with(';');
        }
        if !closed {
            problems.push((
                name,
                start_line,
                "type declaration never reached a terminating `;` before EOF — not indexed"
                    .to_owned(),
            ));
            i += 1;
            continue;
        }

        // Comment-only lines (the corpus's `// ── section divider ──` convention appears INSIDE
        // multi-line type blocks, e.g. lib/compiler/token.myc::Tok) must not reach the joined
        // body: a comment line carries no `|`, so its text would splice onto the neighboring
        // constructor. Raw `block` lines still drive per-ctor line attribution below.
        let joined: String = block
            .iter()
            .map(|(l, _)| l.trim())
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = joined.trim_end_matches(';').trim().to_owned();
        let body_after_eq = joined.split_once('=').map_or(joined.as_str(), |(_, b)| b);
        let body = body_after_eq.trim().trim_end_matches(';').trim();
        let ctors = split_top_level_ctors(body, &block);

        out.push(TypeDecl {
            name,
            start_line,
            text,
            ctors,
        });
        i = j + 1;
    }
    (out, problems)
}

/// Split a type body into top-level (`|`-separated, depth-0 over `()[]{}`) constructor texts, then
/// best-effort attribute each to the source line it appears on.
pub(crate) fn split_top_level_ctors(
    body: &str,
    block: &[(&str, u32)],
) -> Vec<(String, String, u32)> {
    let mut ctors: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut cur = String::new();
    for ch in body.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                cur.push(ch);
            }
            '|' if depth == 0 => {
                ctors.push(std::mem::take(&mut cur).trim().to_owned());
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        ctors.push(cur.trim().to_owned());
    }

    let fallback_line = block.first().map_or(0, |(_, l)| *l);
    ctors
        .into_iter()
        .filter(|c| !c.is_empty())
        .map(|text| {
            let name: String = text
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let line = attribute_line(block, &name).unwrap_or(fallback_line);
            (name, text, line)
        })
        .collect()
}

/// Find the line in `block` whose trimmed text contains `ident` as a whole word (bounded by
/// non-identifier characters on both sides) — a best-effort constructor→line attribution. `None`
/// when no line matches (the caller falls back to the type's start line): a heuristic-on-heuristic,
/// deliberately not claimed more precise than that.
pub(crate) fn attribute_line(block: &[(&str, u32)], ident: &str) -> Option<u32> {
    if ident.is_empty() {
        return None;
    }
    for (raw, line) in block {
        let t = raw.trim();
        let bytes = t.as_bytes();
        let mut search_from = 0;
        while let Some(pos) = t[search_from..].find(ident) {
            let abs = search_from + pos;
            let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
            let after = abs + ident.len();
            let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
            if before_ok && after_ok {
                return Some(*line);
            }
            search_from = abs + ident.len();
            if search_from >= t.len() {
                break;
            }
        }
    }
    None
}

pub(crate) fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Keywords for DN-26 `Item` constructs this extractor does not (yet) parse: `use` (cross-nodule
/// imports/re-exports), `trait`, `impl`, `object`, `derive`, `lower`, `default`. None appear in
/// `lib/` today, but a future nodule using one must be FLAGGED, never silently absent (G2).
const UNRECOGNIZED_KEYWORDS: &[&str] = &[
    "use ", "trait ", "impl ", "object ", "derive ", "lower ", "default ",
];

/// Top-level (column-0) lines starting with an unrecognised `Item` keyword.
pub(crate) fn unrecognized_top_level(src: &str) -> Vec<(&'static str, u32)> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        if line.starts_with(|c: char| c.is_whitespace()) || line.trim().is_empty() {
            continue; // not top-level, or blank
        }
        for kw in UNRECOGNIZED_KEYWORDS.iter().copied() {
            if line.starts_with(kw) {
                out.push((kw.trim(), (i + 1) as u32));
                break;
            }
        }
    }
    out
}
