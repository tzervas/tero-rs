//! Markdown → doc-IR projection (the `gen-manual` / `gen-book` corpus path, spec §4). Projects an
//! RFC/ADR/note/spec/devlog markdown file into a [`Node`] with `Payload::Document` — level-graded sections, prose,
//! checked-or-illustrative code examples, and cross-references — **verbatim projection, never a
//! rewrite** (the prose stays the corpus's; we only structure it). Dependency-free: a small,
//! purpose-built CommonMark *subset* parser (headings, fenced code, paragraphs, inline links) — the
//! same "honestly a subset, named as one" discipline as the manifest TOML reader.

use std::collections::BTreeSet;

use crate::ir::{Level, Node, Payload, Provenance, SourceKind, XrefResolution, XrefTarget};

/// Allocates globally-unique, stable anchor slugs (so deep links never collide — §4.1 navigability).
#[derive(Debug, Default)]
pub struct AnchorAlloc {
    used: BTreeSet<String>,
}

impl AnchorAlloc {
    /// A fresh allocator.
    #[must_use]
    pub fn new() -> Self {
        AnchorAlloc::default()
    }

    /// Slugify `base` (optionally namespaced under `ns`) and make it unique by `-N` suffixing.
    pub fn alloc(&mut self, ns: Option<&str>, base: &str) -> String {
        let slug = slugify(base);
        let slug = if slug.is_empty() {
            "x".to_owned()
        } else {
            slug
        };
        let candidate = match ns {
            Some(n) => format!("{n}--{slug}"),
            None => slug,
        };
        if self.used.insert(candidate.clone()) {
            return candidate;
        }
        for n in 2.. {
            let c = format!("{candidate}-{n}");
            if self.used.insert(c.clone()) {
                return c;
            }
        }
        unreachable!("the integer suffix space is unbounded")
    }
}

/// A GitHub-style anchor slug: lowercase, non-alphanumerics → `-`, collapsed, trimmed.
#[must_use]
pub fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_owned()
}

/// A tokenized markdown block (the parser's intermediate before the section tree is built).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Block {
    Heading {
        level: u8,
        text: String,
        line: u32,
    },
    Code {
        lang: String,
        source: String,
        line: u32,
    },
    Para {
        text: String,
        line: u32,
    },
}

/// Tokenize markdown into a flat block list (a CommonMark subset: ATX headings, ```fenced code,
/// paragraphs). Table rows and lists fall into paragraphs — fine for projection (we structure, we
/// do not re-typeset).
fn tokenize(src: &str) -> Vec<Block> {
    let lines: Vec<&str> = src.lines().collect();
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let trimmed = raw.trim_start();
        // Fenced code block.
        if let Some(rest) = trimmed.strip_prefix("```") {
            let lang = rest.trim().to_owned();
            let start = (i + 1) as u32;
            let mut body = String::new();
            i += 1;
            while i < lines.len() && !lines[i].trim_start().starts_with("```") {
                body.push_str(lines[i]);
                body.push('\n');
                i += 1;
            }
            if i < lines.len() {
                i += 1; // consume the closing fence
            }
            blocks.push(Block::Code {
                lang,
                source: body,
                line: start,
            });
            continue;
        }
        // ATX heading.
        if let Some((level, text)) = parse_heading(trimmed) {
            blocks.push(Block::Heading {
                level,
                text,
                line: (i + 1) as u32,
            });
            i += 1;
            continue;
        }
        // Blank line — skip.
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        // Paragraph: gather until a blank line, a heading, or a fence.
        let start = (i + 1) as u32;
        let mut para = String::new();
        while i < lines.len() {
            let t = lines[i].trim_start();
            if t.is_empty() || t.starts_with("```") || parse_heading(t).is_some() {
                break;
            }
            if !para.is_empty() {
                para.push('\n');
            }
            para.push_str(lines[i].trim_end());
            i += 1;
        }
        blocks.push(Block::Para {
            text: para,
            line: start,
        });
    }
    blocks
}

/// Parse an ATX heading line into `(level, text)` (1–6 `#`, then required space).
fn parse_heading(trimmed: &str) -> Option<(u8, String)> {
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &trimmed[hashes..];
    let text = rest.strip_prefix(' ')?.trim();
    Some((hashes as u8, text.to_owned()))
}

/// Extract inline `[text](target)` link targets from a paragraph (the cross-reference seed).
#[must_use]
pub fn extract_links(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b']' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            // find the matching close paren
            let mut j = i + 2;
            let mut depth = 1;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
                j += 1;
            }
            if j <= bytes.len() {
                let target = &text[i + 2..j];
                // Drop a `"title"` suffix and surrounding whitespace.
                let target = target.split_whitespace().next().unwrap_or("").to_owned();
                if !target.is_empty() {
                    out.push(target);
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Whether a fenced example is held to the type-check bar (§4.1 #4). Only a `myc-checked` info
/// string opts in — a plain ```myc fence stays *illustrative* (most prose snippets are partial, not
/// complete nodule programs; tagging every one would falsely redden the gate). Complete `.myc`
/// programs in the `examples/` corpus are the checked source the apiref generator captures.
fn is_checked_lang(lang: &str) -> bool {
    let l = lang.split_whitespace().next().unwrap_or("");
    matches!(l, "myc-checked")
}

/// The fence language, normalized (the first token of the info string).
fn norm_lang(lang: &str) -> String {
    let l = lang.split_whitespace().next().unwrap_or("");
    if l.is_empty() {
        "text".to_owned()
    } else {
        l.to_owned()
    }
}

/// Project a markdown source into a [`Payload::Document`] node.
///
/// `path` is the repo-relative source (provenance); `source_kind` selects the corpus family. The
/// document title is the first H1 (else the path stem). The lead prose before the first heading is
/// the document's `minimal` summary; H2 sections are `medium`, deeper sections `detailed` (graded
/// depth, §4.1 #3).
#[must_use]
pub fn ingest(path: &str, src: &str, source_kind: SourceKind, alloc: &mut AnchorAlloc) -> Node {
    let blocks = tokenize(src);
    let stem = path_stem(path);
    let doc_anchor = alloc.alloc(None, &stem);

    // Title = first H1, else the path stem.
    let title = blocks
        .iter()
        .find_map(|b| match b {
            Block::Heading { level: 1, text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_else(|| stem.clone());

    let mut children = Vec::new();

    // The lead summary is everything before the first section heading (level ≥ 2), minus the H1
    // title itself — the document's `minimal` depth. The rest is the heading-rooted section forest.
    let first_sec = blocks
        .iter()
        .position(|b| matches!(b, Block::Heading { level, .. } if *level >= 2))
        .unwrap_or(blocks.len());
    let lead: Vec<Block> = blocks[..first_sec]
        .iter()
        .filter(|b| !matches!(b, Block::Heading { level: 1, .. }))
        .cloned()
        .collect();
    for n in blocks_to_nodes(&lead, path, &doc_anchor, Level::Minimal, alloc) {
        children.push(n);
    }

    let rest: Vec<Block> = blocks[first_sec..]
        .iter()
        .filter(|b| !matches!(b, Block::Heading { level: 1, .. }))
        .cloned()
        .collect();
    let (sections, _) = build_sections(&rest, 0, 2, path, &doc_anchor, alloc);
    children.extend(sections);

    Node::new(
        doc_anchor,
        Some(title),
        None,
        Provenance {
            source: path.to_owned(),
            line: 1,
        },
        Payload::Document { source_kind },
        children,
    )
}

/// The file stem (no directory, no extension).
fn path_stem(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.rsplit_once('.').map_or(file, |(s, _)| s).to_owned()
}

/// Recursively build sections from `blocks` starting at `start`, consuming everything until a heading
/// of level `<= parent_level`. Returns `(nodes, next_index)`.
fn build_sections(
    blocks: &[Block],
    parent_level: u8,
    grade_from: u8,
    path: &str,
    ns: &str,
    alloc: &mut AnchorAlloc,
) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();
    let mut i = 0;
    while i < blocks.len() {
        match &blocks[i] {
            Block::Heading { level, .. } if *level <= parent_level => break,
            Block::Heading { level, text, line } => {
                let lvl = *level;
                let anchor = alloc.alloc(Some(ns), text);
                // The block run immediately under this heading (until the next heading of any level).
                let mut body_blocks = Vec::new();
                i += 1;
                while i < blocks.len() && !matches!(blocks[i], Block::Heading { .. }) {
                    body_blocks.push(blocks[i].clone());
                    i += 1;
                }
                let graded = if lvl <= grade_from {
                    Level::Medium
                } else {
                    Level::Detailed
                };
                let mut sec_children = blocks_to_nodes(&body_blocks, path, &anchor, graded, alloc);
                // Recurse into deeper headings nested under this one.
                let (sub, consumed) =
                    build_sections(&blocks[i..], lvl, grade_from, path, &anchor, alloc);
                i += consumed;
                sec_children.extend(sub);
                nodes.push(Node::new(
                    anchor,
                    Some(text.clone()),
                    Some(graded),
                    Provenance {
                        source: path.to_owned(),
                        line: *line,
                    },
                    Payload::Section,
                    sec_children,
                ));
            }
            _ => {
                // Stray content at this level (no enclosing heading) — attach as prose/examples.
                let mut run = Vec::new();
                while i < blocks.len() && !matches!(blocks[i], Block::Heading { .. }) {
                    run.push(blocks[i].clone());
                    i += 1;
                }
                nodes.extend(blocks_to_nodes(&run, path, ns, Level::Medium, alloc));
            }
        }
    }
    (nodes, i)
}

/// Turn a run of leaf blocks (paras + code) into prose / example / xref nodes.
fn blocks_to_nodes(
    blocks: &[Block],
    path: &str,
    ns: &str,
    level: Level,
    alloc: &mut AnchorAlloc,
) -> Vec<Node> {
    let mut nodes = Vec::new();
    for b in blocks {
        match b {
            Block::Para { text, line } => {
                let anchor = alloc.alloc(Some(ns), &first_words(text));
                nodes.push(Node::new(
                    anchor.clone(),
                    None,
                    Some(level),
                    Provenance {
                        source: path.to_owned(),
                        line: *line,
                    },
                    Payload::Prose { text: text.clone() },
                    vec![],
                ));
                for raw in extract_links(text) {
                    let xa = alloc.alloc(Some(&anchor), "xref");
                    nodes.push(Node::new(
                        xa,
                        None,
                        None,
                        Provenance {
                            source: path.to_owned(),
                            line: *line,
                        },
                        Payload::Xref {
                            target: XrefTarget {
                                raw,
                                // Unresolved at ingest; the build's resolver fills the verdict.
                                resolution: XrefResolution::Unresolved,
                            },
                        },
                        vec![],
                    ));
                }
            }
            Block::Code { lang, source, line } => {
                let anchor = alloc.alloc(Some(ns), "example");
                nodes.push(Node::new(
                    anchor,
                    None,
                    Some(level),
                    Provenance {
                        source: path.to_owned(),
                        line: *line,
                    },
                    Payload::Example {
                        lang: norm_lang(lang),
                        source: source.clone(),
                        checked: is_checked_lang(lang),
                    },
                    vec![],
                ));
            }
            Block::Heading { .. } => unreachable!("headings handled by build_sections"),
        }
    }
    nodes
}

/// The first few words of a paragraph, for a readable anchor slug.
fn first_words(text: &str) -> String {
    text.split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_is_github_style() {
        assert_eq!(slugify("Hello, World!"), "hello-world");
        assert_eq!(slugify("§4.1 Quality bar"), "4-1-quality-bar");
        assert_eq!(slugify("  --x--  "), "x");
    }

    #[test]
    fn anchors_are_unique() {
        let mut a = AnchorAlloc::new();
        assert_eq!(a.alloc(None, "Intro"), "intro");
        assert_eq!(a.alloc(None, "Intro"), "intro-2");
        assert_eq!(a.alloc(Some("doc"), "Intro"), "doc--intro");
    }

    #[test]
    fn headings_parse() {
        assert_eq!(
            parse_heading("## Hi there"),
            Some((2, "Hi there".to_owned()))
        );
        assert_eq!(parse_heading("#nope"), None);
        assert_eq!(parse_heading("not a heading"), None);
        assert_eq!(parse_heading("####### too deep"), None);
    }

    #[test]
    fn links_extract_inline_targets() {
        let links = extract_links("see [the RFC](RFC-0013.md#levels) and [home](https://x.io).");
        assert_eq!(links, vec!["RFC-0013.md#levels", "https://x.io"]);
    }

    #[test]
    fn tokenize_separates_headings_code_and_paras() {
        let src = "# Title\n\nIntro para.\n\n## Sec\n\n```myc\nfn f() = 0\n```\n\nmore.\n";
        let blocks = tokenize(src);
        assert!(matches!(blocks[0], Block::Heading { level: 1, .. }));
        assert!(matches!(&blocks[1], Block::Para { text, .. } if text == "Intro para."));
        assert!(matches!(blocks[2], Block::Heading { level: 2, .. }));
        assert!(matches!(&blocks[3], Block::Code { lang, .. } if lang == "myc"));
    }

    #[test]
    fn ingest_builds_a_graded_document_tree() {
        let src = "# My Doc\n\nThe summary line.\n\n## First\n\nBody of first.\n\n### Deep\n\nDeep body.\n\n## Second\n\nBody two.\n";
        let mut alloc = AnchorAlloc::new();
        let doc = ingest("docs/spec/my-doc.md", src, SourceKind::Spec, &mut alloc);
        assert_eq!(doc.title.as_deref(), Some("My Doc"));
        // Lead summary is minimal-graded prose.
        let lead = doc
            .children
            .iter()
            .find(|n| matches!(n.payload, Payload::Prose { .. }))
            .unwrap();
        assert_eq!(lead.level, Some(Level::Minimal));
        // Two H2 sections at medium; the H3 nests under "First" at detailed.
        let secs: Vec<&Node> = doc
            .children
            .iter()
            .filter(|n| matches!(n.payload, Payload::Section))
            .collect();
        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].level, Some(Level::Medium));
        let deep = secs[0]
            .children
            .iter()
            .find(|n| matches!(n.payload, Payload::Section))
            .unwrap();
        assert_eq!(deep.level, Some(Level::Detailed));
    }

    #[test]
    fn a_myc_checked_fence_is_marked_checked() {
        let src =
            "# D\n\n```myc-checked\nfn f() -> Binary{8} = 0b0\n```\n\n```text\nillustrative\n```\n";
        let mut alloc = AnchorAlloc::new();
        let doc = ingest("d.md", src, SourceKind::Note, &mut alloc);
        let examples: Vec<&Node> = {
            let mut v = Vec::new();
            doc.walk(&mut |n| {
                if matches!(n.payload, Payload::Example { .. }) {
                    v.push(n);
                }
            });
            v
        };
        assert_eq!(examples.len(), 2);
        let checked = examples
            .iter()
            .filter(|n| matches!(&n.payload, Payload::Example { checked: true, .. }))
            .count();
        assert_eq!(checked, 1);
    }
}
