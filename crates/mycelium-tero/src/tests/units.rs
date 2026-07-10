//! White-box unit tests of the trickiest extraction helpers (`pub(crate)` access) — the metadata
//! keyword scan (whole-word, first-wins), the id parse, the changelog id scan, the one-line
//! summariser, and the issues.yaml subset helpers.

use crate::changelog::leading_id;
use crate::docs::{doc_id, leading_keyword, one_line};
use crate::issues::{dequote, parse_inline_list};
use crate::model::strip_md_links;

const STATUS: &[&str] = &[
    "Draft",
    "Proposed",
    "Accepted",
    "Enacted",
    "Superseded",
    "Resolved",
];
const GUARANTEE: &[&str] = &["Exact", "Proven", "Empirical", "Declared"];

#[test]
fn leading_keyword_takes_the_first_lattice_word_in_the_status_cell() {
    let src = "# D\n\n| Field | Value |\n|---|---|\n\
               | **Status** | **Enacted** — advanced from **Accepted** (2026) |\n";
    assert_eq!(
        leading_keyword(src, "Status", STATUS).as_deref(),
        Some("Enacted")
    );
}

#[test]
fn leading_keyword_matches_whole_words_only() {
    // "Provenance" must NOT satisfy the `Proven` guarantee keyword.
    let src = "| **Guarantee** | Provenance is recorded, tag Declared |\n";
    assert_eq!(
        leading_keyword(src, "Guarantee", GUARANTEE).as_deref(),
        Some("Declared")
    );
}

#[test]
fn leading_keyword_is_none_without_a_row_or_keyword() {
    assert_eq!(
        leading_keyword("no metadata here\n", "Status", STATUS),
        None
    );
    assert_eq!(
        leading_keyword("| **Status** | pending review |\n", "Status", STATUS),
        None
    );
}

#[test]
fn doc_id_parses_decision_prefixes_only() {
    assert_eq!(
        doc_id("docs/rfcs/RFC-0034-Foo.md").as_deref(),
        Some("RFC-0034")
    );
    assert_eq!(
        doc_id("docs/adr/ADR-032-Bar.md").as_deref(),
        Some("ADR-032")
    );
    assert_eq!(doc_id("docs/notes/DN-87-Baz.md").as_deref(), Some("DN-87"));
    assert_eq!(doc_id("docs/spec/Some-Spec.md"), None);
    assert_eq!(doc_id("docs/guide/README.md"), None);
}

#[test]
fn changelog_leading_id_finds_the_first_corpus_id() {
    assert_eq!(
        leading_id("M-996 — AOT TCO (2026-07-06)").as_deref(),
        Some("M-996")
    );
    assert_eq!(
        leading_id("DN-87 — captured; E39-1 minted").as_deref(),
        Some("DN-87")
    );
    assert_eq!(leading_id("E21-1 landed").as_deref(), Some("E21-1"));
    assert_eq!(leading_id("Just a prose header").as_deref(), None);
}

#[test]
fn one_line_squeezes_and_truncates_with_an_ellipsis() {
    assert_eq!(one_line("first line\nsecond line", 200), "first line");
    let long = "word ".repeat(100);
    let s = one_line(&long, 20);
    assert!(s.chars().count() <= 21 && s.ends_with('…'));
}

#[test]
fn issues_dequote_strips_and_unescapes() {
    assert_eq!(
        dequote("\"M-1 — a \\\"quoted\\\" title\""),
        "M-1 — a \"quoted\" title"
    );
    assert_eq!(dequote("bare-scalar"), "bare-scalar");
}

#[test]
fn strip_md_links_reduces_a_link_to_its_text_and_keeps_the_rest() {
    // A heading with an embedded relative cross-link (the real docs/tero-index/ links-gate case).
    assert_eq!(
        strip_md_links("4. Guarantee matrix — [RFC-0016 §4.5](../../rfcs/RFC-0016.md) here"),
        "4. Guarantee matrix — RFC-0016 §4.5 here"
    );
    // Multiple links + a unicode em dash (UTF-8 byte-safety).
    assert_eq!(strip_md_links("see [a](x) — [b](y)"), "see a — b");
    // A non-link bracket run is copied through unchanged (never a lossy guess).
    assert_eq!(
        strip_md_links("array[i] and (paren)"),
        "array[i] and (paren)"
    );
    assert_eq!(strip_md_links("no links here"), "no links here");
    // The result never contains the `](` link token (what the links gate scans for).
    assert!(!strip_md_links("[t](u)").contains("]("));
}

#[test]
fn issues_parse_inline_list_handles_brackets_and_commas() {
    assert_eq!(
        parse_inline_list("[phase:8, type:epic, status:todo]"),
        vec!["phase:8", "type:epic", "status:todo"]
    );
    assert_eq!(parse_inline_list("[]"), Vec::<String>::new());
    assert_eq!(parse_inline_list("[M-1]"), vec!["M-1"]);
}
