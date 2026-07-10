//! Determinism — the DN-87 §6.3 byte-identical-regeneration contract, checked at the report level
//! (two builds equal) and at the emitted-artifact level (two `INDEX.md`/`index.json` writes are
//! byte-for-byte identical). This is the in-crate twin of `scripts/checks/tero-index.sh`.

use crate::build_tero_index;
use crate::emit::{write_json, write_markdown};
use crate::tests::fixture::{temp_dir, write_corpus};

#[test]
fn two_builds_produce_an_identical_report() {
    let root = temp_dir("det-report");
    write_corpus(&root, true);
    let a = build_tero_index(&root).unwrap();
    let b = build_tero_index(&root).unwrap();
    assert_eq!(a.items, b.items);
    assert_eq!(a.flagged, b.flagged);
}

#[test]
fn two_emissions_are_byte_identical() {
    let root = temp_dir("det-emit");
    write_corpus(&root, true);
    let report = build_tero_index(&root).unwrap();

    let out_a = temp_dir("det-out-a");
    let out_b = temp_dir("det-out-b");
    write_markdown(&report, &out_a).unwrap();
    write_json(&report, &out_a).unwrap();
    write_markdown(&report, &out_b).unwrap();
    write_json(&report, &out_b).unwrap();

    for name in ["INDEX.md", "index.json"] {
        let a = std::fs::read(out_a.join(name)).unwrap();
        let b = std::fs::read(out_b.join(name)).unwrap();
        assert_eq!(a, b, "{name} must be byte-identical across regenerations");
    }
}

#[test]
fn emitted_markdown_ends_with_exactly_one_newline() {
    let root = temp_dir("det-nl");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let out = temp_dir("det-nl-out");
    write_markdown(&report, &out).unwrap();
    let md = std::fs::read_to_string(out.join("INDEX.md")).unwrap();
    assert!(md.ends_with('\n') && !md.ends_with("\n\n"));
    // The honesty header + sibling references are present (posture carried into the artifact).
    assert!(md.contains("source is ground truth"));
    assert!(md.contains("Sibling indices"));
    assert!(md.contains("`api-index`") && md.contains("`lib-index`"));
}
