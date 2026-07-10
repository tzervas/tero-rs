//! `load_report` — the `index.json` round-trip (M-1016): write a report, load it back, and confirm
//! it is exactly the report that was written (the read-side twin of `determinism.rs`'s write-side
//! byte-identical check).

use crate::load::load_report;
use crate::tests::fixture::{temp_dir, write_corpus};
use crate::{build_tero_index, write_json, Family, Query, QueryEngine, TeroIndexItem};

#[test]
fn a_written_report_loads_back_identical() {
    let root = temp_dir("load-roundtrip");
    write_corpus(&root, true); // defects included: exercises duplicate ids, missing fields, etc.
    let written = build_tero_index(&root).unwrap();

    let out = temp_dir("load-roundtrip-out");
    write_json(&written, &out).unwrap();

    let loaded = load_report(&out.join("index.json")).unwrap();
    assert_eq!(loaded.items, written.items);
    assert_eq!(loaded.flagged, written.flagged);
}

#[test]
fn loading_a_missing_file_is_an_io_error_not_a_silent_empty_report() {
    let root = temp_dir("load-missing");
    let err = load_report(&root.join("does-not-exist.json")).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn loading_malformed_json_is_an_invalid_data_error() {
    let root = temp_dir("load-malformed");
    std::fs::write(root.join("bad.json"), "{ not json").unwrap();
    let err = load_report(&root.join("bad.json")).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn loading_ignores_the_top_level_fields_it_does_not_round_trip() {
    // `generated`/`item_tag`/`siblings` are the crate's own constants, not read back — a payload
    // that carries only `items`/`flagged` (the minimal shape `write_json` always emits a superset
    // of) must still load cleanly.
    let root = temp_dir("load-minimal");
    std::fs::write(root.join("minimal.json"), r#"{"items": [], "flagged": []}"#).unwrap();
    let loaded = load_report(&root.join("minimal.json")).unwrap();
    assert!(loaded.items.is_empty());
    assert!(loaded.flagged.is_empty());
}

#[test]
fn a_shuffled_but_valid_report_loads_into_canonical_order() {
    // Regression (M-1016 review): `load_report` used to trust the file's row order outright, so a
    // validly-shaped but out-of-order `index.json` (hand-edited, or written by some future producer
    // that forgets to sort) would load with an order that silently diverges from `QueryEngine`'s
    // `order_by = "canonical index order"` claim. `load_report` must now re-canonicalize
    // unconditionally, regardless of the file's on-disc order.
    let root = temp_dir("load-shuffled");
    let a = TeroIndexItem::new("a-anchor", Family::Doc, "section", "A", "a.md", 1);
    let b = TeroIndexItem::new("b-anchor", Family::Doc, "section", "B", "a.md", 2);
    let z = TeroIndexItem::new("z-anchor", Family::Doc, "section", "Z", "z.md", 5);

    #[derive(serde::Serialize)]
    struct Payload {
        items: Vec<TeroIndexItem>,
        flagged: Vec<crate::Flagged>,
    }
    let payload = Payload {
        items: vec![z, a, b], // deliberately NOT in (family, file, line, anchor) order
        flagged: Vec::new(),
    };
    let path = root.join("shuffled.json");
    std::fs::write(&path, serde_json::to_string(&payload).unwrap()).unwrap();

    let loaded = load_report(&path).unwrap();
    let anchors: Vec<&str> = loaded.items.iter().map(|it| it.anchor.as_str()).collect();
    assert_eq!(
        anchors,
        vec!["a-anchor", "b-anchor", "z-anchor"],
        "load_report must canonicalize row order, not trust the file's on-disc order"
    );

    // The query engine, built over the now-canonicalized report, answers in that order too — the
    // `Explain::order_by` "canonical index order" claim stays accurate for a *loaded* report, not
    // only a freshly built one.
    let engine = QueryEngine::new(&loaded);
    let answer = engine.run(&Query::Kind("section".to_owned())).unwrap();
    let hit_anchors: Vec<&str> = answer.items().iter().map(|it| it.anchor.as_str()).collect();
    assert_eq!(hit_anchors, vec!["a-anchor", "b-anchor", "z-anchor"]);
}
