//! End-to-end: build a small corpus tree from disk, emit every view, and run the §4.1 lint — the
//! filesystem-walking + xref-resolution path that the unit tests stub. Hermetic (a temp dir).

use std::fs;
use std::path::{Path, PathBuf};

use mycelium_doc::build::{build, emit_all, BuildInput};
use mycelium_doc::doc_lint;

/// A unique temp directory under the system temp root.
fn temp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("mycdoc-{tag}-{nanos}"));
    fs::create_dir_all(&p).unwrap();
    p
}

fn write(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

#[test]
fn a_small_corpus_builds_emits_and_passes_the_quality_bar() {
    let root = temp_dir("ok");

    // An RFC that cross-references an ADR (resolves) and an external URL (out of scope).
    write(
        &root,
        "docs/rfcs/RFC-0001-Thing.md",
        "# RFC-0001 — Thing\n\nThe abstract summary line.\n\n## Guide-level explanation\n\n\
         See [the decision](../adr/ADR-001-Choice.md) and the [spec](https://example.com/x).\n\n\
         ### A deeper subsection\n\nDetail here.\n",
    );
    write(
        &root,
        "docs/adr/ADR-001-Choice.md",
        "# ADR-001 — Choice\n\nWe chose X.\n\n## Context\n\nBecause Y.\n",
    );
    // A schema (api reference) with one documented + one undocumented field.
    write(
        &root,
        "docs/spec/schemas/thing.schema.json",
        r#"{"title":"Thing","description":"A thing.","properties":{"a":{"type":"string","description":"the a"},"b":{"type":"number"}}}"#,
    );
    // A real, type-checking example nodule.
    write(
        &root,
        "examples/demo/mycelium-proj.toml",
        "[project]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    );
    write(
        &root,
        "examples/demo/demo.myc",
        "// nodule: demo\n// @summary: A demo nodule.\nnodule demo;\n\nfn id(x: Binary{8}) => Binary{8} =\n  x;\n",
    );

    let input = BuildInput::conventional(&root);
    let model = build(&input).expect("build succeeds");

    // The corpus + schema + example all projected.
    assert!(model.documents.len() >= 4, "got {}", model.documents.len());

    // Emit every view; the artifact set includes the index, a page per doc, JSON, Typst, EPUB note.
    let arts = emit_all(&model);
    assert!(arts.files.contains_key("index.html"));
    assert!(arts.files.contains_key("doc-model.json"));
    assert!(arts.files.contains_key("doc.typ"));
    assert!(arts.files.contains_key("EPUB-DEFERRED.txt"));
    let out = root.join("out");
    let n = arts.write_to(&out).expect("write artifacts");
    assert_eq!(n, arts.files.len());
    assert!(out.join("index.html").exists());

    // The §4.1 lint passes (green-and-real): no error-severity findings.
    let report = doc_lint::lint(&model);
    assert!(
        !report.has_errors(),
        "unexpected errors: {:?}",
        report.errors()
    );

    // And it is genuinely exercising the checks: the example type-checked, the ADR xref resolved.
    let checked = report
        .outcomes
        .iter()
        .find(|o| o.name == "checked-examples")
        .unwrap();
    assert!(
        checked.summary.contains("1 checked examples")
            || checked.summary.contains("checked examples")
    );
    let xref = report
        .outcomes
        .iter()
        .find(|o| o.name == "no-dead-xref")
        .unwrap();
    assert!(xref.summary.contains("internal xrefs resolve"));

    fs::remove_dir_all(&root).ok();
}

#[test]
fn a_broken_internal_link_is_caught_by_the_gate() {
    let root = temp_dir("dead");
    // A doc linking to a sibling corpus doc that does not exist → a dead internal xref.
    write(
        &root,
        "docs/notes/DN-99-Note.md",
        "# DN-99 — Note\n\nLead.\n\n## Body\n\nSee [the missing one](./DN-100-Missing.md).\n",
    );
    let input = BuildInput::conventional(&root);
    let model = build(&input).expect("build succeeds");
    let report = doc_lint::lint(&model);
    assert!(
        report.errors().iter().any(|f| f.check == "no-dead-xref"),
        "the dead internal link must fail the gate; errors: {:?}",
        report.errors()
    );
    fs::remove_dir_all(&root).ok();
}
