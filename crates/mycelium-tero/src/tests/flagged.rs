//! The never-silent flagged path (G2): an extraction limit is *recorded*, never dropped, and the
//! affected row (where one still exists) is still emitted.

use crate::build_tero_index;
use crate::model::Family;
use crate::tests::fixture::{temp_dir, write_corpus};

fn has_flag(report: &crate::TeroIndexReport, item_contains: &str, reason_contains: &str) -> bool {
    report
        .flagged
        .iter()
        .any(|f| f.item.contains(item_contains) && f.reason.contains(reason_contains))
}

#[test]
fn a_note_without_a_status_row_is_flagged_not_assumed() {
    // Add a note with no `| **Status** |` row on top of the defect-free corpus.
    let root = temp_dir("flag-status");
    write_corpus(&root, false);
    std::fs::write(
        root.join("docs/notes/No-Status-Note.md"),
        "# A statusless note\n\nprose\n\n## Section\n\ns\n",
    )
    .unwrap();
    let report = build_tero_index(&root).unwrap();
    assert!(has_flag(
        &report,
        "No-Status-Note.md",
        "no `| **Status** |`"
    ));
    // The note is STILL indexed (flagged, not dropped) — just with status unset.
    let note = report
        .items
        .iter()
        .find(|i| i.file.ends_with("No-Status-Note.md") && i.kind == "note")
        .unwrap();
    assert_eq!(note.status, None);
}

#[test]
fn a_note_with_an_off_lattice_status_is_flagged_distinctly_from_a_missing_row() {
    // A note whose `| **Status** |` row IS present but carries a non-lattice value (`Living`) must
    // NOT be flagged as "no Status row" (the PR #1237 review HIGH — the two cases were conflated).
    let root = temp_dir("flag-offlattice");
    write_corpus(&root, false);
    std::fs::write(
        root.join("docs/notes/Living-Note.md"),
        "# A living note\n\n\
         | Field | Value |\n|---|---|\n| **Status** | **Living — initial capture** |\n\n\
         prose\n\n## Section\n\ns\n",
    )
    .unwrap();
    let report = build_tero_index(&root).unwrap();
    // Flagged with the ROW-PRESENT reason, quoting the off-lattice value — never the missing-row one.
    assert!(has_flag(
        &report,
        "Living-Note.md",
        "not on the ratified lattice"
    ));
    assert!(has_flag(&report, "Living-Note.md", "Living"));
    assert!(!has_flag(&report, "Living-Note.md", "no `| **Status** |`"));
    // Still indexed, status unset (not coerced to a lattice value).
    let note = report
        .items
        .iter()
        .find(|i| i.file.ends_with("Living-Note.md") && i.kind == "note")
        .unwrap();
    assert_eq!(note.status, None);
}

#[test]
fn a_duplicate_issue_id_is_flagged_and_both_rows_kept() {
    let root = temp_dir("flag-dup");
    write_corpus(&root, true); // defects include a duplicate M-0099
    let report = build_tero_index(&root).unwrap();
    assert!(has_flag(&report, "M-0099", "duplicate id"));
    // Both entries survive — never silently deduped (honest reflection of the corpus).
    let dups = report
        .items
        .iter()
        .filter(|i| i.anchor == "M-0099" && i.family == Family::Issue)
        .count();
    assert_eq!(dups, 2);
}

#[test]
fn an_issue_with_no_title_is_flagged_and_kept_under_its_id() {
    let root = temp_dir("flag-notitle");
    write_corpus(&root, true); // M-0100 has no title
    let report = build_tero_index(&root).unwrap();
    assert!(has_flag(&report, "M-0100", "no `title:`"));
    let row = report.items.iter().find(|i| i.anchor == "M-0100").unwrap();
    assert_eq!(row.title, "M-0100"); // falls back to the id, not invented
}

#[test]
fn a_skill_without_frontmatter_or_name_is_flagged_not_indexed() {
    let root = temp_dir("flag-skill");
    write_corpus(&root, true); // adds nofront + noname skills
    let report = build_tero_index(&root).unwrap();
    assert!(has_flag(
        &report,
        "nofront/SKILL.md",
        "no `--- … ---` YAML frontmatter"
    ));
    assert!(has_flag(&report, "noname/SKILL.md", "no `name:`"));
    // Only the one valid skill is indexed.
    assert_eq!(
        report
            .items
            .iter()
            .filter(|i| i.family == Family::Skill)
            .count(),
        1
    );
}

#[test]
fn a_clean_corpus_has_no_flags() {
    let root = temp_dir("flag-clean");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    assert!(
        report.flagged.is_empty(),
        "clean fixture must be flag-free: {:?}",
        report.flagged
    );
}
