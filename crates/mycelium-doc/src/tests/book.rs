//! White-box tests for [`crate::book`] — a hermetic mini-corpus + a hand-built manifest exercise
//! chapter resolution, prev/next nav, the search index, and the never-silent error paths.

use crate::book::*;
use crate::build::{build, BuildInput};
use std::fs;
use std::path::{Path, PathBuf};

fn temp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("mycdoc-book-{tag}-{nanos}"));
    fs::create_dir_all(&p).unwrap();
    p
}

fn write(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

/// A small, realistic-shaped corpus: two wiki pages, three stdlib module specs (drift-proof glob),
/// a grammar EBNF file (the one synthesized non-`.md` source), and CONTRIBUTING.md at the repo root.
fn small_corpus() -> PathBuf {
    let root = temp_dir("corpus");
    write(&root, "docs/wiki/Home.md", "# Home\n\nWelcome.\n");
    write(
        &root,
        "docs/wiki/Getting-Started.md",
        "# Getting Started\n\nInstall it.\n",
    );
    write(
        &root,
        "docs/spec/stdlib/alpha.md",
        "# alpha\n\nThe alpha module.\n",
    );
    write(
        &root,
        "docs/spec/stdlib/beta.md",
        "# beta\n\nThe beta module.\n",
    );
    write(
        &root,
        "docs/spec/stdlib/README.md",
        "# stdlib index\n\nNot a module — must be excluded.\n",
    );
    write(
        &root,
        "docs/spec/grammar/mycelium.ebnf",
        "program ::= nodule_header item*;\n",
    );
    write(&root, "CONTRIBUTING.md", "# Contributing\n\nRead this.\n");
    root
}

fn small_manifest() -> BookManifest {
    BookManifest {
        title: "Test Book".to_owned(),
        preface: "A tiny preface.".to_owned(),
        chapters: vec![
            ChapterSpec {
                title: "Getting Started".to_owned(),
                sources: vec![
                    "docs/wiki/Home.md".to_owned(),
                    "docs/wiki/Getting-Started.md".to_owned(),
                ],
                globs: vec![],
                exclude: vec![],
            },
            ChapterSpec {
                title: "Reference".to_owned(),
                sources: vec!["docs/spec/grammar/mycelium.ebnf".to_owned()],
                globs: vec![],
                exclude: vec![],
            },
            ChapterSpec {
                title: "Standard Library".to_owned(),
                sources: vec![],
                globs: vec!["docs/spec/stdlib/*.md".to_owned()],
                exclude: vec!["docs/spec/stdlib/README.md".to_owned()],
            },
            ChapterSpec {
                title: "Contributing".to_owned(),
                sources: vec!["CONTRIBUTING.md".to_owned()],
                globs: vec![],
                exclude: vec![],
            },
        ],
    }
}

fn built_model(root: &Path) -> crate::ir::DocModel {
    let mut input = BuildInput::conventional(root);
    input.extra_md_files = vec![root.join("CONTRIBUTING.md")];
    build(&input).expect("build succeeds")
}

#[test]
fn a_small_book_builds_every_page_navigably() {
    let root = small_corpus();
    let model = built_model(&root);
    let manifest = small_manifest();

    let arts = build_book(&model, &manifest, &root).expect("book build succeeds");

    assert!(arts.files.contains_key("book/index.html"));
    assert!(arts.files.contains_key("book/search-index.json"));
    assert!(arts.files.contains_key("book/assets/search.js"));
    assert!(arts.files.contains_key("book/search.html"));

    // Home, Getting-Started, the grammar synth page, alpha, beta (README excluded), CONTRIBUTING.
    let page_count = arts
        .files
        .keys()
        .filter(|k| k.starts_with("book/pages/"))
        .count();
    assert_eq!(page_count, 6, "pages: {:?}", arts.files.keys());

    // The stdlib README (anchor "readme", from its filename stem) must never appear as a book page
    // — it is excluded from the Standard Library chapter's glob.
    assert!(!arts.files.contains_key("book/pages/readme.html"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn every_page_carries_prev_next_nav_and_a_chapter_breadcrumb() {
    let root = small_corpus();
    let model = built_model(&root);
    let manifest = small_manifest();
    let arts = build_book(&model, &manifest, &root).expect("book build succeeds");

    let home = arts.files.get("book/pages/home.html").expect("home page");
    // First page: prev goes to the ToC, next goes forward.
    assert!(home.contains("Table of contents →") || home.contains("Table of contents"));
    assert!(home.contains("Chapter 1: Getting Started"));

    let getting_started = arts
        .files
        .get("book/pages/getting-started.html")
        .expect("getting-started page");
    assert!(getting_started.contains("← Home"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn the_grammar_ebnf_is_synthesized_verbatim() {
    let root = small_corpus();
    let model = built_model(&root);
    let manifest = small_manifest();
    let arts = build_book(&model, &manifest, &root).expect("book build succeeds");

    let grammar_page = arts
        .files
        .get("book/pages/book-grammar-ebnf.html")
        .expect("the synthesized grammar page exists");
    assert!(grammar_page.contains("program ::= nodule_header item*;"));
    assert!(grammar_page.contains("language-ebnf"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn the_search_index_has_one_record_per_page_with_a_grounded_snippet() {
    let root = small_corpus();
    let model = built_model(&root);
    let manifest = small_manifest();
    let arts = build_book(&model, &manifest, &root).expect("book build succeeds");

    let idx = arts.files.get("book/search-index.json").unwrap();
    let records: Vec<serde_json::Value> = serde_json::from_str(idx).unwrap();
    assert_eq!(records.len(), 6);
    let home = records
        .iter()
        .find(|r| r["title"] == "Home")
        .expect("Home is indexed");
    assert!(home["snippet"].as_str().unwrap().contains("Welcome"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn a_manifest_entry_with_no_ingested_document_is_a_build_error_not_a_silent_gap() {
    let root = small_corpus();
    let model = built_model(&root);
    let mut manifest = small_manifest();
    manifest.chapters.push(ChapterSpec {
        title: "Broken".to_owned(),
        sources: vec!["docs/does-not-exist.md".to_owned()],
        globs: vec![],
        exclude: vec![],
    });

    let err = build_book(&model, &manifest, &root).expect_err("must fail, never silently skip");
    assert!(err.0.contains("does-not-exist.md"));
    assert!(err.0.contains("Broken"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn an_empty_chapter_is_a_build_error() {
    let root = small_corpus();
    let model = built_model(&root);
    let mut manifest = small_manifest();
    manifest.chapters.push(ChapterSpec {
        title: "Empty".to_owned(),
        sources: vec![],
        globs: vec![],
        exclude: vec![],
    });

    let err = build_book(&model, &manifest, &root).expect_err("an empty chapter must fail");
    assert!(err.0.contains("Empty"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn a_page_listed_in_two_chapters_is_a_build_error() {
    let root = small_corpus();
    let model = built_model(&root);
    let mut manifest = small_manifest();
    manifest.chapters[1]
        .sources
        .push("docs/wiki/Home.md".to_owned());

    let err = build_book(&model, &manifest, &root).expect_err("a duplicate page must fail");
    assert!(err.0.contains("docs/wiki/Home.md"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn the_committed_repo_manifest_parses_and_matches_the_documented_schema() {
    // A structural sanity check on docs/book-manifest.json itself (not a full build — that needs
    // the whole real corpus, exercised instead by the committed `just docs-book` recipe).
    let manifest_src = include_str!("../../../../docs/book-manifest.json");
    let manifest: BookManifest =
        serde_json::from_str(manifest_src).expect("the committed manifest parses");
    assert!(!manifest.chapters.is_empty());
    assert!(manifest.chapters.iter().all(|c| !c.title.is_empty()));
}
