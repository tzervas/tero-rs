//! Anchor stability + uniqueness — deep links must be stable across regeneration and collision-free
//! (a citation anchor that moves or aliases is worthless).

use std::collections::BTreeSet;

use crate::build_tero_index;
use crate::model::Family;
use crate::tests::fixture::{temp_dir, write_corpus};

#[test]
fn anchors_are_unique_across_a_clean_corpus() {
    let root = temp_dir("anchor-uniq");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let mut seen = BTreeSet::new();
    for item in &report.items {
        assert!(
            seen.insert(item.anchor.clone()),
            "duplicate anchor {} — anchors must be globally unique in a clean corpus",
            item.anchor
        );
    }
}

#[test]
fn anchors_are_stable_across_regeneration() {
    let root = temp_dir("anchor-stable");
    write_corpus(&root, false);
    let a: Vec<String> = build_tero_index(&root)
        .unwrap()
        .items
        .iter()
        .map(|i| i.anchor.clone())
        .collect();
    let b: Vec<String> = build_tero_index(&root)
        .unwrap()
        .items
        .iter()
        .map(|i| i.anchor.clone())
        .collect();
    assert_eq!(a, b);
}

#[test]
fn doc_section_anchors_are_namespaced_under_their_document() {
    let root = temp_dir("anchor-ns");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    // The RFC-0099 document anchor + its section anchors, which must be prefixed by the doc anchor.
    let doc = report
        .items
        .iter()
        .find(|i| i.id.as_deref() == Some("RFC-0099") && i.kind != "section")
        .unwrap();
    let sections: Vec<&crate::TeroIndexItem> = report
        .items
        .iter()
        .filter(|i| i.family == Family::Doc && i.kind == "section" && i.file == doc.file)
        .collect();
    assert!(!sections.is_empty());
    for s in sections {
        assert!(
            s.anchor.starts_with(&format!("{}--", doc.anchor)),
            "section anchor {} not namespaced under doc anchor {}",
            s.anchor,
            doc.anchor
        );
    }
}

#[test]
fn changelog_and_skill_anchors_are_family_namespaced() {
    let root = temp_dir("anchor-fam");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    for i in &report.items {
        match i.family {
            Family::Changelog => assert!(i.anchor.starts_with("cl--")),
            Family::Skill => assert!(i.anchor.starts_with("sk--")),
            Family::Issue => assert!(i.id.as_deref() == Some(i.anchor.as_str())),
            _ => {}
        }
    }
}
