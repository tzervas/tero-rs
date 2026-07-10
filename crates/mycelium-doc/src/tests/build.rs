//! White-box tests for [`crate::build`] — extracted from the logic file (as-touched, CLAUDE.md test
//! layout rule) when `extra_md_files` was added. Uses `pub(crate)` access to `classify`,
//! `classify_target`, `normalize_join`, and `ResolveCtx`.

use crate::build::*;
use crate::ir::SourceKind;
use crate::ir::XrefResolution;

#[test]
fn normalize_join_resolves_dot_dot() {
    assert_eq!(
        normalize_join("docs/rfcs", "../adr/ADR-003.md"),
        "docs/adr/ADR-003.md"
    );
    assert_eq!(normalize_join("docs", "./Glossary.md"), "docs/Glossary.md");
    assert_eq!(normalize_join("docs/spec", "x.md"), "docs/spec/x.md");
}

#[test]
fn classify_maps_paths_to_families() {
    assert_eq!(classify("docs/rfcs/RFC-0001.md"), SourceKind::Rfc);
    assert_eq!(classify("docs/adr/ADR-010.md"), SourceKind::Adr);
    assert_eq!(classify("docs/notes/DN-06.md"), SourceKind::Note);
    assert_eq!(classify("docs/devlog/x.md"), SourceKind::Devlog);
    assert_eq!(classify("docs/spec/SPEC.md"), SourceKind::Spec);
    assert_eq!(classify("docs/Glossary.md"), SourceKind::Other);
    // A repo-root file outside `docs/` (e.g. CONTRIBUTING.md via `extra_md_files`) is `Other` too —
    // no substring match, same as any other unclassified corpus doc.
    assert_eq!(classify("CONTRIBUTING.md"), SourceKind::Other);
}

fn ctx_with(files: &[(&str, &str)], anchors: &[&str], corpus: &str) -> ResolveCtx {
    ResolveCtx {
        anchors: anchors.iter().map(|s| (*s).to_owned()).collect(),
        file_index: files
            .iter()
            .map(|(p, a)| ((*p).to_owned(), (*a).to_owned()))
            .collect(),
        corpus_rel: Some(corpus.to_owned()),
    }
}

#[test]
fn an_external_url_is_out_of_scope_not_dead() {
    let ctx = ctx_with(&[], &[], "docs");
    assert_eq!(
        classify_target("https://example.com", "d--x", "docs/a.md", &ctx),
        XrefResolution::ExternalUrl
    );
}

#[test]
fn a_resolving_internal_md_link_is_internal() {
    let ctx = ctx_with(
        &[("docs/rfcs/RFC-0013.md", "rfc-0013")],
        &["rfc-0013", "rfc-0013--levels"],
        "docs",
    );
    // file-level
    assert_eq!(
        classify_target("../rfcs/RFC-0013.md", "spec--x", "docs/spec/a.md", &ctx),
        XrefResolution::Internal {
            anchor: "rfc-0013".to_owned()
        }
    );
    // fragment-level
    assert_eq!(
        classify_target(
            "../rfcs/RFC-0013.md#levels",
            "spec--x",
            "docs/spec/a.md",
            &ctx
        ),
        XrefResolution::Internal {
            anchor: "rfc-0013--levels".to_owned()
        }
    );
}

#[test]
fn a_broken_internal_corpus_link_is_dead() {
    let ctx = ctx_with(&[], &[], "docs");
    match classify_target("../rfcs/RFC-9999.md", "spec--x", "docs/spec/a.md", &ctx) {
        XrefResolution::Dead { .. } => {}
        other => panic!("expected Dead, got {other:?}"),
    }
}

#[test]
fn a_link_outside_the_corpus_is_out_of_scope() {
    let ctx = ctx_with(&[], &[], "docs");
    // README at repo root — links.sh owns it, not the doc-IR.
    assert_eq!(
        classify_target("../../README.md", "spec--x", "docs/spec/a.md", &ctx),
        XrefResolution::OutOfScope
    );
    // a non-markdown target
    assert_eq!(
        classify_target("../../scripts/lib.sh", "spec--x", "docs/spec/a.md", &ctx),
        XrefResolution::OutOfScope
    );
}

#[test]
fn a_missing_fragment_falls_back_to_the_document_top() {
    let ctx = ctx_with(
        &[("docs/x.md", "x")],
        &["x"], // no x--nope anchor
        "docs",
    );
    assert_eq!(
        classify_target("x.md#nope", "y--a", "docs/y.md", &ctx),
        XrefResolution::Internal {
            anchor: "x".to_owned()
        }
    );
}

#[test]
fn extra_md_files_are_ingested_and_their_xrefs_resolve() {
    // A hermetic tiny tree: docs/rfcs/RFC-0001.md is the corpus; CONTRIBUTING.md sits at the repo
    // root and links to it — proving extra_md_files goes through the SAME resolve pipeline as the
    // corpus walk (not a silently-unresolved bolt-on).
    let root = std::env::temp_dir().join(format!(
        "mycdoc-extra-md-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("docs/rfcs")).unwrap();
    std::fs::write(
        root.join("docs/rfcs/RFC-0001-Thing.md"),
        "# RFC-0001 — Thing\n\nAbstract.\n",
    )
    .unwrap();
    std::fs::write(
        root.join("CONTRIBUTING.md"),
        "# Contributing\n\nSee [RFC-0001](docs/rfcs/RFC-0001-Thing.md).\n",
    )
    .unwrap();

    let mut input = BuildInput::conventional(&root);
    input.extra_md_files = vec![root.join("CONTRIBUTING.md")];
    let model = build(&input).expect("build succeeds");

    let contributing = model
        .documents
        .iter()
        .find(|d| d.provenance.source == "CONTRIBUTING.md")
        .expect("CONTRIBUTING.md was ingested");
    assert_eq!(contributing.title.as_deref(), Some("Contributing"));

    let mut found_internal = false;
    contributing.walk(&mut |n| {
        if let crate::ir::Payload::Xref { target } = &n.payload {
            if matches!(target.resolution, XrefResolution::Internal { .. }) {
                found_internal = true;
            }
        }
    });
    assert!(
        found_internal,
        "CONTRIBUTING.md's link to the RFC should resolve internally"
    );

    std::fs::remove_dir_all(&root).ok();
}
