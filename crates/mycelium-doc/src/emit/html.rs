//! The static-HTML renderer (spec §8.1 — static HTML path). One reviewed template (§5): a `<header>`,
//! an index→detail `<nav>`, a level-graded `<main>`, and a provenance `<footer>` — **semantic HTML by
//! construction** (the §4.1 legibility/accessibility bar: heading order never skips, code carries a
//! `language-*` class, the nav is labelled). Every node element carries `data-cid="blake3:…"`, its
//! content address — the hook the dual-projection-parity lint checks against the JSON view.

use crate::emit::{html_escape, Artifacts};
use crate::ir::{DocModel, Node, Payload, SourceKind, XrefResolution};

/// The one reviewed template's CSS (the shared visual language, §5). Its content feeds the pinned
/// template hash recorded in every page footer (provenance, §6).
const STYLE: &str = "\
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
footer{color:var(--dim);font-size:.85rem;border-top:1px solid #ddd;margin-top:2rem}";

/// The pinned template content hash (provenance, §6) — the address of the shared template/style.
#[must_use]
pub fn template_hash() -> String {
    use crate::hash::DocHasher;
    let mut h = DocHasher::new();
    h.tag(200).str(STYLE);
    h.finish().as_str().to_owned()
}

/// Render the whole model to an HTML site: `index.html` plus one `pages/<anchor>.html` per document.
#[must_use]
pub fn render(model: &DocModel) -> Artifacts {
    let mut arts = Artifacts::new();
    arts.put("index.html", render_index(model));
    for doc in &model.documents {
        arts.put(format!("pages/{}.html", doc.anchor), render_page(doc));
    }
    arts
}

/// The concatenation of every page (for the parity/legibility lints, which scan the rendered output).
#[must_use]
pub fn render_concat(model: &DocModel) -> String {
    let mut s = render_index(model);
    for doc in &model.documents {
        s.push('\n');
        s.push_str(&render_page(doc));
    }
    s
}

fn doc_title(doc: &Node) -> &str {
    doc.title.as_deref().unwrap_or(&doc.anchor)
}

fn source_kind(doc: &Node) -> SourceKind {
    match &doc.payload {
        Payload::Document { source_kind } => *source_kind,
        _ => SourceKind::Other,
    }
}

fn page_shell(title: &str, nav: &str, main: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n\
         <meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n\
         <title>{title} — Mycelium</title>\n\
         <style>{STYLE}</style>\n\
         </head>\n<body>\n\
         <header><h1 class=\"site-title\">Mycelium Documentation</h1>\
         <p>A projection of the cited corpus — never a parallel truth (ADR-003/G11).</p></header>\n\
         {nav}\n<main>\n{main}\n</main>\n\
         <footer>Generated from the Mycelium corpus · one template (hash <code>{th}</code>) · \
         every block is content-addressed (ADR-003). Undocumented items are flagged, never invented (G2).</footer>\n\
         </body>\n</html>\n",
        title = html_escape(title),
        th = html_escape(&short_hash(&template_hash())),
    )
}

fn short_hash(h: &str) -> String {
    // `blake3:<12 hex>…` — readable provenance without the full 64 hex.
    match h.split_once(':') {
        Some((algo, digest)) => format!("{algo}:{}…", &digest[..digest.len().min(12)]),
        None => h.to_owned(),
    }
}

/// The index→detail entry point (§4.1 #2): documents grouped by corpus family, each a deep link.
fn render_index(model: &DocModel) -> String {
    let groups = [
        (SourceKind::Spec, "Specifications & contracts"),
        (SourceKind::Rfc, "RFCs"),
        (SourceKind::Adr, "Architecture decisions"),
        (SourceKind::Note, "Design notes"),
        (SourceKind::Api, "API reference"),
        (SourceKind::Devlog, "Devlog"),
        (SourceKind::Other, "Other"),
    ];
    let mut nav = String::from("<nav aria-label=\"Documentation index\">\n");
    for (kind, label) in groups {
        let mut items = String::new();
        for doc in &model.documents {
            if source_kind(doc) == kind {
                items.push_str(&format!(
                    "  <li><a href=\"pages/{a}.html\" data-cid=\"{cid}\">{t}</a></li>\n",
                    a = html_escape(&doc.anchor),
                    cid = html_escape(doc.id.as_str()),
                    t = html_escape(doc_title(doc)),
                ));
            }
        }
        if !items.is_empty() {
            nav.push_str(&format!(
                "<section><h2>{}</h2>\n<ul>\n{items}</ul></section>\n",
                html_escape(label)
            ));
        }
    }
    nav.push_str("</nav>");
    let main = "<p>Pick a document from the index. Each page offers graded depth \
                (minimal · medium · detailed — RFC-0013 levels reused).</p>"
        .to_owned();
    page_shell("Index", &nav, &main)
}

fn render_page(doc: &Node) -> String {
    let mut main = String::new();
    main.push_str(&format!(
        "<article id=\"{id}\" data-cid=\"{cid}\"><h1>{t}</h1>\n",
        id = html_escape(&doc.anchor),
        cid = html_escape(doc.id.as_str()),
        t = html_escape(doc_title(doc)),
    ));
    for child in &doc.children {
        render_node(child, 2, &doc.anchor, &mut main);
    }
    main.push_str("</article>");
    let nav =
        "<nav aria-label=\"Site\"><ul><li><a href=\"../index.html\">← Index</a></li></ul></nav>";
    page_shell(doc_title(doc), nav, &main)
}

/// Render one node at heading `depth` (2..=6, clamped — heading order never skips, §4.1 #8).
/// `doc_anchor` is the enclosing document's anchor (so a cross-document xref gets the right page href).
fn render_node(node: &Node, depth: usize, doc_anchor: &str, buf: &mut String) {
    let cid = html_escape(node.id.as_str());
    match &node.payload {
        Payload::Section => {
            let h = depth.clamp(2, 6);
            let lvl = node
                .level
                .map(|l| format!(" <span class=\"level\">{}</span>", l.as_str()))
                .unwrap_or_default();
            buf.push_str(&format!(
                "<section data-cid=\"{cid}\" id=\"{id}\">\n<h{h}>{t}{lvl}</h{h}>\n",
                id = html_escape(&node.anchor),
                t = html_escape(node.title.as_deref().unwrap_or("")),
            ));
            for c in &node.children {
                render_node(c, depth + 1, doc_anchor, buf);
            }
            buf.push_str("</section>\n");
        }
        Payload::Prose { text } => {
            buf.push_str(&format!(
                "<p data-cid=\"{cid}\">{}</p>\n",
                html_escape(text)
            ));
        }
        Payload::Example {
            lang,
            source,
            checked,
        } => {
            let badge = if *checked {
                " <span class=\"checked\" title=\"type-checked in CI\">✓ checked</span>"
            } else {
                " <span class=\"level\" title=\"illustrative, not CI-checked\">illustrative</span>"
            };
            buf.push_str(&format!(
                "<figure data-cid=\"{cid}\">{badge}\n<pre><code class=\"language-{lang}\">{src}</code></pre>\n</figure>\n",
                lang = html_escape(lang),
                src = html_escape(source),
            ));
        }
        Payload::Xref { target } => {
            let (href, class) = match &target.resolution {
                XrefResolution::Internal { anchor } => {
                    // Same page → a bare fragment; cross-page → the sibling page + fragment.
                    let target_doc = anchor.split("--").next().unwrap_or(anchor);
                    let href = if target_doc == doc_anchor {
                        format!("#{}", html_escape(anchor))
                    } else {
                        format!("{}.html#{}", html_escape(target_doc), html_escape(anchor))
                    };
                    (href, "")
                }
                XrefResolution::ExternalUrl | XrefResolution::OutOfScope => {
                    (html_escape(&target.raw), "")
                }
                XrefResolution::Dead { .. } | XrefResolution::Unresolved => {
                    (html_escape(&target.raw), " class=\"unresolved\"")
                }
            };
            buf.push_str(&format!(
                "<a data-cid=\"{cid}\" href=\"{href}\"{class}>{t}</a>\n",
                t = html_escape(&target.raw),
            ));
        }
        Payload::ApiItem { signature, summary } => {
            let h = depth.clamp(2, 6);
            buf.push_str(&format!(
                "<section data-cid=\"{cid}\" id=\"{id}\">\n<h{h}><code>{sig}</code></h{h}>\n",
                id = html_escape(&node.anchor),
                sig = html_escape(signature.as_deref().unwrap_or("")),
            ));
            match summary {
                Some(s) => buf.push_str(&format!("<p>{}</p>\n", html_escape(s))),
                None => buf.push_str(
                    "<p class=\"undocumented\">undocumented — no summary projected from source (G2)</p>\n",
                ),
            }
            for c in &node.children {
                render_node(c, depth + 1, doc_anchor, buf);
            }
            buf.push_str("</section>\n");
        }
        Payload::Undocumented { what } => {
            buf.push_str(&format!(
                "<p data-cid=\"{cid}\" class=\"undocumented\">undocumented: {}</p>\n",
                html_escape(what),
            ));
        }
        Payload::Document { .. } | Payload::Index => {
            // Nested documents/index are not expected inside a page body; render children flatly.
            for c in &node.children {
                render_node(c, depth, doc_anchor, buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{ingest, AnchorAlloc};
    use crate::ir::SourceKind;

    fn model() -> DocModel {
        let mut a = AnchorAlloc::new();
        let src = "# Doc\n\nLead.\n\n## Sec\n\nBody with [a link](other.md#x).\n\n```myc-checked\nfn f() -> Binary{8} = 0b0\n```\n";
        let doc = ingest("docs/spec/doc.md", src, SourceKind::Spec, &mut a);
        DocModel::new(vec![doc])
    }

    #[test]
    fn the_site_has_an_index_and_a_page_per_doc() {
        let m = model();
        let arts = render(&m);
        assert!(arts.files.contains_key("index.html"));
        assert_eq!(
            arts.files
                .keys()
                .filter(|k| k.starts_with("pages/"))
                .count(),
            1
        );
    }

    #[test]
    fn every_node_id_is_embedded_for_parity() {
        let m = model();
        let html = render_concat(&m);
        for id in m.id_set() {
            assert!(html.contains(&id), "missing cid {id} in HTML");
        }
    }

    #[test]
    fn the_template_is_one_and_pinned() {
        let m = model();
        let html = render_concat(&m);
        let th = template_hash();
        assert!(th.starts_with("blake3:"));
        // The footer pins the (short) template hash on every page.
        assert!(html.contains("one template"));
    }

    #[test]
    fn output_is_semantic_and_accessible() {
        let m = model();
        let html = render_concat(&m);
        assert!(html.contains("<main>"));
        assert!(html.contains("aria-label"));
        assert!(html.contains("lang=\"en\""));
        assert!(html.contains("class=\"language-"));
    }

    #[test]
    fn an_undocumented_api_item_renders_a_visible_marker() {
        let mut a = AnchorAlloc::new();
        let doc = crate::apiref::project_nodule(
            "x.myc",
            "// nodule: x\nnodule x\nfn g() -> Binary{8} = 0b0\n",
            &mut a,
        );
        let m = DocModel::new(vec![doc]);
        let html = render_concat(&m);
        assert!(html.contains("undocumented"));
    }
}
