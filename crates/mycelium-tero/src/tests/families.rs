//! Per-family coverage + counts — over the hermetic fixture (exact oracle) *and* over the live
//! repo corpus cross-checked against independent line counts (the lib_index "cross-check against an
//! independent grep" precedent, and the M-1015 DoD's "every corpus family covered").

use std::path::{Path, PathBuf};

use crate::build_tero_index;
use crate::model::Family;
use crate::tests::fixture::{temp_dir, write_corpus};

fn doc_count(items: &[crate::TeroIndexItem], family: Family, doc_kind_not_section: bool) -> usize {
    items
        .iter()
        .filter(|i| i.family == family && (!doc_kind_not_section || i.kind != "section"))
        .count()
}

#[test]
fn fixture_family_counts_match_the_oracle() {
    let root = temp_dir("families");
    let expected = write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();

    assert_eq!(
        doc_count(&report.items, Family::Doc, true),
        expected.docs,
        "doc Document rows"
    );
    assert_eq!(
        doc_count(&report.items, Family::Research, true),
        expected.research,
        "research record rows"
    );
    assert_eq!(
        doc_count(&report.items, Family::Issue, false),
        expected.issues,
        "issue rows"
    );
    assert_eq!(
        doc_count(&report.items, Family::Changelog, false),
        expected.changelog,
        "changelog rows"
    );
    assert_eq!(
        doc_count(&report.items, Family::Skill, false),
        expected.skills,
        "skill rows"
    );
    // Every family is represented (none silently empty) + sections were emitted for docs.
    assert!(report.items.iter().any(|i| i.kind == "section"));
}

#[test]
fn fixture_extracts_status_guarantee_and_issue_fields() {
    let root = temp_dir("fields");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let find = |id: &str| {
        report
            .items
            .iter()
            .find(|i| i.id.as_deref() == Some(id))
            .unwrap()
    };

    let rfc = find("RFC-0099");
    assert_eq!(rfc.status.as_deref(), Some("Accepted"));
    assert_eq!(rfc.guarantee_tag.as_deref(), Some("Proven"));
    assert_eq!(
        rfc.summary.as_deref(),
        Some("The lead prose that becomes the summary.")
    );

    let dn = find("DN-99");
    assert_eq!(dn.status.as_deref(), Some("Proposed"));
    assert_eq!(dn.guarantee_tag.as_deref(), Some("Declared"));

    let issue = report
        .items
        .iter()
        .find(|i| i.anchor == "M-0099" && i.family == Family::Issue);
    let issue = issue.unwrap();
    assert_eq!(issue.kind, "issue");
    assert_eq!(issue.status.as_deref(), Some("todo"));
    assert_eq!(issue.epic.as_deref(), Some("E99-1"));
    assert_eq!(issue.depends_on, vec!["M-0001", "M-0002"]);
    assert_eq!(
        issue.doc_refs,
        vec!["corpus:RFC-0099", "src:crates/mycelium-tero/src/lib.rs"]
    );
    assert_eq!(issue.gh_issue.as_deref(), Some("4242")); // from idmap.tsv
    assert_eq!(
        issue.summary.as_deref(),
        Some("The body first line becomes the summary.")
    );

    let epic = report.items.iter().find(|i| i.anchor == "E99-1").unwrap();
    assert_eq!(epic.kind, "epic");
    assert_eq!(epic.status.as_deref(), Some("in-progress"));
}

// ── live-repo cross-check (independent-grep parity) ─────────────────────────────────────────────

/// The repo root, two levels above this crate's manifest dir.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

/// Count lines in `path` whose trimmed-left form starts with `prefix` (an independent grep).
fn count_lines_starting(path: &Path, prefix: &str) -> usize {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| l.starts_with(prefix)).count())
        .unwrap_or(0)
}

#[test]
fn live_issue_count_matches_independent_grep() {
    let root = repo_root();
    let issues_yaml = root.join("tools/github/issues.yaml");
    if !issues_yaml.exists() {
        return; // skip-graceful in a stripped checkout
    }
    let report = build_tero_index(&root).unwrap();
    let indexed = report
        .items
        .iter()
        .filter(|i| i.family == Family::Issue)
        .count();
    // Independent oracle: `grep -c '^  - id:'`.
    let grepped = count_lines_starting(&issues_yaml, "  - id:");
    assert_eq!(indexed, grepped, "issue rows vs independent `- id:` count");
    assert!(indexed > 0);
}

#[test]
fn live_changelog_and_skill_counts_match_independent_greps() {
    let root = repo_root();
    let changelog = root.join("CHANGELOG.md");
    if !changelog.exists() {
        return;
    }
    let report = build_tero_index(&root).unwrap();

    let indexed_cl = report
        .items
        .iter()
        .filter(|i| i.family == Family::Changelog)
        .count();
    let grepped_cl =
        count_lines_starting(&changelog, "## ") + count_lines_starting(&changelog, "### ");
    assert_eq!(indexed_cl, grepped_cl, "changelog rows vs `##`+`###` count");

    // Skills: one row per SKILL.md that has a name; independent count of SKILL.md files is the
    // upper bound (a nameless/frontmatter-less one is flagged out, so indexed <= files).
    let skills_dir = root.join(".claude/skills");
    if skills_dir.exists() {
        let files = walk_skill_files(&skills_dir);
        let indexed_sk = report
            .items
            .iter()
            .filter(|i| i.family == Family::Skill)
            .count();
        assert!(indexed_sk <= files, "skill rows <= SKILL.md files");
        assert!(indexed_sk > 0);
    }
}

/// Count `SKILL.md` files under `dir` (independent of the extractor's own walk).
fn walk_skill_files(dir: &Path) -> usize {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn live_regeneration_is_deterministic() {
    let root = repo_root();
    if !root.join("docs").exists() {
        return;
    }
    let a = build_tero_index(&root).unwrap();
    let b = build_tero_index(&root).unwrap();
    assert_eq!(
        a.items, b.items,
        "two live builds must be identical (determinism)"
    );
    assert_eq!(a.flagged, b.flagged);
}
