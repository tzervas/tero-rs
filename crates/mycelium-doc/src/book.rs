//! The **BOOK** output — M-363's output (b), *"the full language book"* (spec §4 `gen-book`): a
//! curated, linear, chaptered reading order over the doc-IR, with per-page prev/next navigation and a
//! client-side search index. This is a **fifth renderer alongside HTML/Typst/JSON**
//! ([`crate::emit`]) — it composes the *existing* honest per-page HTML projection into a book; it
//! does **not** re-author content (spec §4: "projection, not authorship").
//!
//! ## Curated ordering, not a parallel truth
//! A book needs a reading order the flat, alphabetical-by-`SourceKind` corpus index doesn't have
//! (the ratified spec §4 calls `gen-book` **"projection + light interpretation"** — sequencing is the
//! interpretive part; the *content* on each page is still pure projection). The order is a small,
//! committed manifest ([`docs/book-manifest.json`](../../../../docs/book-manifest.json)), **not**
//! hand-edited generated output: each chapter lists explicit `sources` (curated order) and/or
//! `globs` (drift-proof — a new stdlib/RFC/ADR/DN file is picked up automatically, the same
//! `tools/docgen/code_index.py` discipline). A manifest entry that resolves to **no** ingested
//! document is a **build error** (never a silently-dropped chapter, never a dead link — G2).
//!
//! ## Composing, not re-rendering
//! Each book page's body is the **same** `<article>` HTML [`crate::emit::html`] already produced for
//! the corpus site (byte-identical `data-cid` attributes and all) — this module renders a *scoped*
//! [`DocModel`] (exactly the book's pages) through [`crate::emit::html::render`] and re-wraps the
//! extracted article in a book-specific shell (chapter breadcrumb, prev/next, a ToC/search
//! sidebar link). Two non-corpus sources are honestly, explicitly composed in too:
//! - `CONTRIBUTING.md` (repo root, outside `docs/`) rides in via [`crate::build::BuildInput::extra_md_files`]
//!   so it is a genuine ingested [`Node`], not a special case here.
//! - `docs/spec/grammar/mycelium.ebnf` is not markdown, so [`crate::build::build`] never walks it; this
//!   module synthesizes a single [`Payload::Document`] node wrapping its content **verbatim** as an
//!   unchecked [`Payload::Example`] (grounded — the exact file bytes — never invented prose).
//!
//! No new dependency: manifest parsing reuses `serde`/`serde_json` (already vetted, KC-3); the search
//! index and its client-side filter are hand-rolled JSON + vanilla JS, the same "no heavy dep"
//! posture as the rest of this crate.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::emit::{html_escape, Artifacts};
use crate::ir::{DocModel, Level, Node, Payload, Provenance, SourceKind};

/// The repo-relative default location of the committed chapter manifest.
pub const DEFAULT_MANIFEST_PATH: &str = "docs/book-manifest.json";

/// A never-silent book-build error (a broken manifest entry, a bad manifest, an anchor collision) —
/// surfaced with enough detail to fix it, never a silently-dropped chapter (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookError(pub String);

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BookError {}

/// The committed chapter manifest (`docs/book-manifest.json`) — curated order, drift-proof globs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookManifest {
    /// The book's title (the ToC page `<h1>`).
    pub title: String,
    /// A short, hand-authored preface (the one piece of new prose this module authors — spec
    /// "minimal new authoring"; everything else is projected).
    pub preface: String,
    /// Chapters, in reading order.
    pub chapters: Vec<ChapterSpec>,
}

/// One chapter: an ordered list of explicit sources, optionally extended by drift-proof globs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterSpec {
    /// The chapter title.
    pub title: String,
    /// Explicit repo-relative source paths, in curated reading order.
    #[serde(default)]
    pub sources: Vec<String>,
    /// Glob patterns (a single `*` wildcard per pattern — a hand-rolled subset, not a full glob
    /// engine, the same "honestly a subset" discipline as [`crate::corpus`]'s markdown parser).
    /// Matches are resolved against every ingested document's source path, sorted, and appended
    /// after `sources` — so a new file under the globbed directory is picked up automatically.
    #[serde(default)]
    pub globs: Vec<String>,
    /// Source paths to exclude from a `globs` match (e.g. a module `README.md`/template file).
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// Load the committed manifest from `<repo_root>/docs/book-manifest.json`.
///
/// # Errors
/// A missing or unparseable manifest is a `BookError`, never a silent empty book.
pub fn load_manifest(repo_root: &Path) -> Result<BookManifest, BookError> {
    let path = repo_root.join(DEFAULT_MANIFEST_PATH);
    let src = std::fs::read_to_string(&path)
        .map_err(|e| BookError(format!("reading {}: {e}", path.display())))?;
    serde_json::from_str(&src).map_err(|e| BookError(format!("parsing {}: {e}", path.display())))
}

/// A single wildcard-per-pattern match: `prefix` + `*` + `suffix` — a hand-rolled subset (no `**`,
/// no character classes), matching this crate's "honestly a subset, named as one" convention.
fn glob_match(pattern: &str, candidate: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == candidate,
        Some((prefix, suffix)) => {
            candidate.len() >= prefix.len() + suffix.len()
                && candidate.starts_with(prefix)
                && candidate.ends_with(suffix)
        }
    }
}

/// Synthesize a single [`Payload::Document`] node wrapping a non-markdown file **verbatim** as an
/// unchecked example — grounded (the file's exact bytes), never invented. Used for the one
/// non-`.md` book source (`docs/spec/grammar/mycelium.ebnf`); `checked: false` is honest (it is a
/// grammar fragment, not a `.myc` program — the checked-examples lint only ever applies to real
/// nodule source, §4.1 #4).
fn synth_verbatim_node(anchor: &str, title: &str, path: &str, lang: &str, src: &str) -> Node {
    let prov = Provenance {
        source: path.to_owned(),
        line: 1,
    };
    let body = Node::new(
        format!("{anchor}--source"),
        None,
        Some(Level::Detailed),
        prov.clone(),
        Payload::Example {
            lang: lang.to_owned(),
            source: src.to_owned(),
            checked: false,
        },
        vec![],
    );
    Node::new(
        anchor.to_owned(),
        Some(title.to_owned()),
        None,
        prov,
        Payload::Document {
            source_kind: SourceKind::Spec,
        },
        vec![body],
    )
}

/// One resolved book page: which chapter it belongs to, and the doc-IR node it projects.
struct Page {
    chapter_idx: usize,
    node: Node,
}

/// Resolve every chapter's `sources`/`globs` against the model, synthesizing the one honest
/// exception (the grammar EBNF). Never-silent: an entry that resolves to nothing is a `BookError`.
fn resolve_pages(
    model: &DocModel,
    manifest: &BookManifest,
    repo_root: &Path,
) -> Result<Vec<Page>, BookError> {
    let by_source: BTreeMap<&str, &Node> = model
        .documents
        .iter()
        .map(|d| (d.provenance.source.as_str(), d))
        .collect();

    let mut pages = Vec::new();
    let mut seen_anchors: BTreeSet<String> = BTreeSet::new();
    let mut seen_sources: BTreeSet<String> = BTreeSet::new();

    for (chapter_idx, chapter) in manifest.chapters.iter().enumerate() {
        let mut ordered_paths: Vec<String> = chapter.sources.clone();
        if !chapter.globs.is_empty() {
            let mut matched: Vec<String> = model
                .documents
                .iter()
                .map(|d| d.provenance.source.clone())
                .filter(|p| chapter.globs.iter().any(|g| glob_match(g, p)))
                .filter(|p| !chapter.exclude.contains(p))
                .collect();
            matched.sort();
            ordered_paths.extend(matched);
        }
        if ordered_paths.is_empty() {
            return Err(BookError(format!(
                "chapter '{}' resolves to zero pages (empty sources/globs, or every glob match was \
                 excluded) — a chapter with no content is a broken book, not a silent skip",
                chapter.title
            )));
        }
        for path in ordered_paths {
            if !seen_sources.insert(path.clone()) {
                return Err(BookError(format!(
                    "'{path}' appears in more than one book chapter — a page must have exactly one \
                     place in the reading order"
                )));
            }
            let node = if let Some(&n) = by_source.get(path.as_str()) {
                n.clone()
            } else if path.ends_with(".ebnf") {
                let full = repo_root.join(&path);
                let src = std::fs::read_to_string(&full).map_err(|e| {
                    BookError(format!(
                        "chapter '{}': cannot read grammar source {path}: {e}",
                        chapter.title
                    ))
                })?;
                synth_verbatim_node(
                    "book-grammar-ebnf",
                    "Mycelium Grammar (EBNF)",
                    &path,
                    "ebnf",
                    &src,
                )
            } else {
                return Err(BookError(format!(
                    "chapter '{}' references '{path}' but no such document was ingested — fix \
                     docs/book-manifest.json or the source path (never a silently-dropped chapter)",
                    chapter.title
                )));
            };
            if !seen_anchors.insert(node.anchor.clone()) {
                return Err(BookError(format!(
                    "duplicate book page anchor '{}' (from '{path}') — an anchor collision would \
                     silently merge two distinct pages",
                    node.anchor
                )));
            }
            pages.push(Page { chapter_idx, node });
        }
    }
    Ok(pages)
}

/// Extract the `<article>...</article>` body from a full rendered page — reusing the *exact* honest
/// projection [`crate::emit::html::render`] already produced (same `data-cid`s), never re-deriving
/// content (spec §4: "projection, not authorship").
fn extract_article(page_html: &str) -> &str {
    let start = page_html.find("<article").unwrap_or(0);
    let end = page_html
        .rfind("</article>")
        .map_or(page_html.len(), |i| i + "</article>".len());
    &page_html[start..end]
}

/// The shared visual language (the same design tokens as [`crate::emit::html`]'s one template,
/// §5 — "one reviewed template" — extended with the chapter-nav/prev-next/search chrome a linear
/// book needs and the wiki-style corpus index does not).
const BOOK_STYLE: &str = "\
:root{--fg:#1a1a2e;--bg:#fdfdfd;--accent:#3a5;--dim:#667;--code:#f4f4f8}\
*{box-sizing:border-box}\
body{margin:0;font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}\
header,nav,main,footer{max-width:54rem;margin:0 auto;padding:1rem 1.25rem}\
header{border-bottom:2px solid var(--accent)}\
nav ul{list-style:none;padding-left:1rem}\
a{color:var(--accent)}a.unresolved{color:#b00;text-decoration:line-through}\
h1,h2,h3,h4,h5,h6{line-height:1.25}\
pre{background:var(--code);padding:.75rem 1rem;border-radius:6px;overflow:auto}\
code{font:0.9em ui-monospace,SFMono-Regular,Menlo,monospace}\
.undocumented{color:var(--dim);font-style:italic;border-left:3px solid #c93;padding-left:.5rem}\
.level{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}\
.checked{color:var(--accent);font-size:.75rem}\
.crumb{font-size:.85rem;color:var(--dim)}\
.pager{display:flex;justify-content:space-between;margin:2rem 0;font-size:.9rem}\
.pager a{border:1px solid #ddd;border-radius:6px;padding:.5rem .9rem}\
#book-search-box{width:100%;padding:.5rem;font-size:1rem;border:1px solid #ddd;border-radius:6px}\
#book-search-results li{margin:.5rem 0}\
footer{color:var(--dim);font-size:.85rem;border-top:1px solid #ddd;margin-top:2rem}";

fn book_shell(book_title: &str, page_title: &str, nav: &str, main: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n\
         <meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n\
         <title>{page_title} — {book_title}</title>\n\
         <style>{BOOK_STYLE}</style>\n\
         </head>\n<body>\n\
         <header><h1 class=\"site-title\">{book_title}</h1>\
         <p>A curated linear composition of the honest corpus projection — never a parallel truth \
         (ADR-003/G11). <a href=\"../search.html\">Search the book</a></p></header>\n\
         {nav}\n<main>\n{main}\n</main>\n\
         <footer>Generated by <code>myc-doc book</code> — every page composes the same \
         content-addressed article the corpus site renders (dual-projection parity by \
         construction). Undocumented items are flagged, never invented (G2).</footer>\n\
         </body>\n</html>\n",
        book_title = html_escape(book_title),
        page_title = html_escape(page_title),
    )
}

fn page_title(node: &Node) -> &str {
    node.title.as_deref().unwrap_or(&node.anchor)
}

/// Build every book artifact: the ToC/landing page, one page per chapter entry (prev/next nav), and
/// the search index + its page.
///
/// # Errors
/// A manifest entry that does not resolve to an ingested document (or a duplicate/collision) is a
/// `BookError` — a broken book is a build failure, never a silently-incomplete one (§4.1 "never a
/// half-build").
pub fn build_book(
    model: &DocModel,
    manifest: &BookManifest,
    repo_root: &Path,
) -> Result<Artifacts, BookError> {
    let pages = resolve_pages(model, manifest, repo_root)?;

    // Render every page's article through the SAME html renderer as the corpus site (composition,
    // not re-authorship) — scoped to exactly the book's pages.
    let scoped = DocModel::new(pages.iter().map(|p| p.node.clone()).collect());
    let rendered = crate::emit::html::render(&scoped);

    let mut arts = Artifacts::new();

    // ── per-page HTML, with prev/next + chapter breadcrumb ──────────────────────────────────────
    for (i, page) in pages.iter().enumerate() {
        let full_page = rendered
            .files
            .get(&format!("pages/{}.html", page.node.anchor))
            .map_or("", String::as_str);
        let article = extract_article(full_page);
        let chapter = &manifest.chapters[page.chapter_idx];

        let prev_link = i
            .checked_sub(1)
            .map(|j| &pages[j])
            .map(|p| {
                format!(
                    "<a href=\"{}.html\">← {}</a>",
                    html_escape(&p.node.anchor),
                    html_escape(page_title(&p.node))
                )
            })
            .unwrap_or_else(|| "<a href=\"../index.html\">← Table of contents</a>".to_owned());
        let next_link = pages
            .get(i + 1)
            .map(|p| {
                format!(
                    "<a href=\"{}.html\">{} →</a>",
                    html_escape(&p.node.anchor),
                    html_escape(page_title(&p.node))
                )
            })
            .unwrap_or_else(|| "<a href=\"../index.html\">Table of contents →</a>".to_owned());

        let nav = format!(
            "<nav aria-label=\"Chapter\"><p class=\"crumb\"><a href=\"../index.html\">Table of \
             contents</a> · Chapter {}: {}</p></nav>",
            page.chapter_idx + 1,
            html_escape(&chapter.title)
        );
        let main = format!(
            "{article}\n<div class=\"pager\"><span>{prev_link}</span><span>{next_link}</span></div>"
        );
        arts.put(
            format!("book/pages/{}.html", page.node.anchor),
            book_shell(&manifest.title, page_title(&page.node), &nav, &main),
        );
    }

    // ── the ToC / landing page ──────────────────────────────────────────────────────────────────
    let mut toc = String::from("<nav aria-label=\"Table of contents\">\n");
    for (ci, chapter) in manifest.chapters.iter().enumerate() {
        toc.push_str(&format!(
            "<section><h2>{}. {}</h2>\n<ol>\n",
            ci + 1,
            html_escape(&chapter.title)
        ));
        for page in pages.iter().filter(|p| p.chapter_idx == ci) {
            toc.push_str(&format!(
                "  <li><a href=\"pages/{a}.html\" data-cid=\"{cid}\">{t}</a></li>\n",
                a = html_escape(&page.node.anchor),
                cid = html_escape(page.node.id.as_str()),
                t = html_escape(page_title(&page.node)),
            ));
        }
        toc.push_str("</ol></section>\n");
    }
    toc.push_str("</nav>");
    let main = format!(
        "<p>{}</p>\n<p><a href=\"search.html\">Search the book</a> · {} chapters, {} pages.</p>",
        html_escape(&manifest.preface),
        manifest.chapters.len(),
        pages.len()
    );
    arts.put(
        "book/index.html",
        book_index_shell(&manifest.title, &toc, &main),
    );

    // ── search index + search page (client-side, no new dep) ───────────────────────────────────
    let search_index = render_search_index(&pages, manifest);
    arts.put("book/search-index.json", search_index);
    arts.put("book/assets/search.js", SEARCH_JS.to_owned());
    arts.put(
        "book/search.html",
        book_index_shell(&manifest.title, "", SEARCH_PAGE_BODY),
    );

    Ok(arts)
}

fn book_index_shell(book_title: &str, nav: &str, main: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n\
         <meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n\
         <title>{book_title}</title>\n\
         <style>{BOOK_STYLE}</style>\n\
         </head>\n<body>\n\
         <header><h1 class=\"site-title\">{book_title}</h1>\
         <p>A curated linear composition of the honest corpus projection — never a parallel truth \
         (ADR-003/G11). <a href=\"search.html\">Search the book</a></p></header>\n\
         {nav}\n<main>\n{main}\n</main>\n\
         <footer>Generated by <code>myc-doc book</code> from the M-363 doc-IR. Undocumented items \
         are flagged, never invented (G2).</footer>\n\
         </body>\n</html>\n",
        book_title = html_escape(book_title),
    )
}

/// One search record — title, the chapter it lives in, its page URL (relative to `book/`), and a
/// short snippet (the document's lead prose, when present — grounded, never invented).
#[derive(Debug, Serialize)]
struct SearchRecord<'a> {
    title: &'a str,
    chapter: &'a str,
    url: String,
    snippet: String,
}

fn lead_snippet(node: &Node) -> String {
    let mut snippet = String::new();
    node.walk(&mut |n| {
        if snippet.is_empty() {
            if let Payload::Prose { text } = &n.payload {
                snippet = text.chars().take(200).collect();
            }
        }
    });
    snippet
}

fn render_search_index(pages: &[Page], manifest: &BookManifest) -> String {
    let records: Vec<SearchRecord<'_>> = pages
        .iter()
        .map(|p| SearchRecord {
            title: page_title(&p.node),
            chapter: &manifest.chapters[p.chapter_idx].title,
            url: format!("pages/{}.html", p.node.anchor),
            snippet: lead_snippet(&p.node),
        })
        .collect();
    serde_json::to_string_pretty(&records).expect("search records are always serializable")
}

/// A small, dependency-free client-side substring filter over `search-index.json` — no search
/// engine, no heavy dep; keeps the crate's dependency posture (KC-3).
const SEARCH_JS: &str = "\
async function mycBookSearch() {
  const res = await fetch('search-index.json');
  const records = await res.json();
  const input = document.getElementById('book-search-box');
  const out = document.getElementById('book-search-results');
  function render(q) {
    out.innerHTML = '';
    if (!q) { return; }
    const needle = q.toLowerCase();
    const hits = records.filter(r =>
      r.title.toLowerCase().includes(needle) ||
      r.chapter.toLowerCase().includes(needle) ||
      r.snippet.toLowerCase().includes(needle)
    );
    for (const r of hits) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = r.url;
      a.textContent = r.title + ' (' + r.chapter + ')';
      li.appendChild(a);
      out.appendChild(li);
    }
    if (hits.length === 0) {
      out.innerHTML = '<li>No matches.</li>';
    }
  }
  input.addEventListener('input', () => render(input.value));
}
mycBookSearch();
";

const SEARCH_PAGE_BODY: &str = "\
<p>Search across every page in the book (title, chapter, and lead summary — client-side, no \
server round-trip).</p>
<input id=\"book-search-box\" type=\"search\" placeholder=\"Search the book…\" \
aria-label=\"Search the book\">
<ul id=\"book-search-results\" aria-live=\"polite\"></ul>
<script src=\"assets/search.js\"></script>";
