//! A hermetic temp-dir corpus fixture shared by the behavioural tests (the mycelium-doc
//! `lib_index` / `book` / `build` precedent). `write_corpus` lays down a *known* mini-corpus so a
//! test can assert exact per-family counts, status/guarantee extraction, anchor stability, and the
//! never-silent flagged path — without depending on the live repo. (The live-repo cross-check is a
//! separate test in `families.rs`.)

use std::fs;
use std::path::{Path, PathBuf};

/// A unique temp dir (nanos-suffixed, no collision across runs — matches the lib_index fixture).
pub(crate) fn temp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("myctero-{tag}-{nanos}"));
    fs::create_dir_all(&p).unwrap();
    p
}

/// Write `contents` to `root/rel`, creating parent dirs.
pub(crate) fn write(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

/// The counts a defect-free [`write_corpus`] must produce — the test oracle.
pub(crate) struct Expected {
    /// doc-family Document rows (kind != "section").
    pub docs: usize,
    /// research-family Document rows (kind == "record").
    pub research: usize,
    /// issue-family rows.
    pub issues: usize,
    /// changelog-family rows (releases + entries).
    pub changelog: usize,
    /// skill-family rows.
    pub skills: usize,
}

/// Lay down a known mini-corpus under `root`. With `defects`, add the never-silent cases (a note
/// with no Status row, a duplicate issue id, an issue with no title, a SKILL.md with no
/// frontmatter, a SKILL.md with no name). Returns the *defect-free* expected counts (the defect
/// rows are asserted separately in `flagged.rs`).
pub(crate) fn write_corpus(root: &Path, defects: bool) -> Expected {
    // ── docs/ ───────────────────────────────────────────────────────────────────────────────────
    write(
        root,
        "docs/rfcs/RFC-0099-Test.md",
        "# RFC-0099 — A Test RFC\n\n\
         | Field | Value |\n|---|---|\n| **Status** | **Accepted** (2026-01-01) |\n\
         | **Guarantee** | `Proven` where the theorem is checked |\n\n\
         The lead prose that becomes the summary.\n\n\
         ## §1 First\n\nBody one.\n\n## §2 Second\n\nBody two.\n",
    );
    write(
        root,
        "docs/adr/ADR-099-Test.md",
        "# ADR-099 — A Test ADR\n\n\
         | Field | Value |\n|---|---|\n| **Status** | **Enacted** (2026-01-02) |\n\n\
         Intro.\n\n## Context\n\nc\n\n## Decision\n\nd\n",
    );
    write(
        root,
        "docs/notes/DN-99-Test.md",
        "# DN-99 — A Test Note\n\n\
         | Field | Value |\n|---|---|\n| **Status** | **Proposed** (2026-01-03) |\n\
         | **Guarantee** | Declared throughout |\n\n\
         A note lead.\n\n## Only Section\n\ns\n",
    );
    write(
        root,
        "docs/spec/Some-Spec.md",
        "# A Spec\n\nSpecs need no status row.\n\n## Clause\n\nx\n",
    );
    // ── research/ ───────────────────────────────────────────────────────────────────────────────
    write(
        root,
        "research/01-test-RECORD.md",
        "# A Research Record\n\nFindings lead.\n\n## Method\n\nm\n",
    );
    // ── CHANGELOG.md (1 release + 2 dated entries) ──────────────────────────────────────────────
    write(
        root,
        "CHANGELOG.md",
        "# Changelog\n\n## [Unreleased]\n\n\
         ### M-0099 — a test change (2026-01-04)\n\n- did a thing\n\n\
         ### DN-99 — captured (2026-01-03)\n\n- noted a thing\n",
    );
    // ── .claude/skills (one valid) ──────────────────────────────────────────────────────────────
    write(
        root,
        ".claude/skills/good/SKILL.md",
        "---\nname: good-skill\ndescription: >-\n  A folded description that\n  spans two lines.\n\
         when_to_use: Use it when testing.\n---\n\n# Good Skill\n\nbody\n",
    );

    // ── issues.yaml + idmap.tsv ─────────────────────────────────────────────────────────────────
    let mut issues = String::from(
        "issues:\n\
         \x20 - id: M-0099\n\
         \x20   title: \"M-0099 — a test issue\"\n\
         \x20   milestone: \"Phase 8\"\n\
         \x20   labels: [phase:8, type:feature, status:todo]\n\
         \x20   epic: E99-1\n\
         \x20   depends_on: [M-0001, M-0002]\n\
         \x20   doc_refs:\n\
         \x20     - corpus:RFC-0099\n\
         \x20     - src:crates/mycelium-tero/src/lib.rs\n\
         \x20   body: |\n\
         \x20     The body first line becomes the summary.\n\
         \x20     A second body line ignored.\n\
         \x20 - id: E99-1\n\
         \x20   title: \"E99-1 (epic) — a test epic\"\n\
         \x20   milestone: \"Phase 8\"\n\
         \x20   labels: [phase:8, type:epic, status:in-progress]\n\
         \x20   epic: E99\n\
         \x20   depends_on: [M-0099]\n\
         \x20   body: |\n\
         \x20     Epic body.\n",
    );
    let mut expected_issues = 2;
    if defects {
        // A duplicate id (union-merge hazard) and an entry with no title — both kept + flagged.
        issues.push_str(
            "\x20 - id: M-0099\n\
             \x20   title: \"M-0099 — a duplicate\"\n\
             \x20   labels: [status:done]\n\
             \x20 - id: M-0100\n\
             \x20   labels: [status:todo]\n\
             \x20   body: |\n\
             \x20     No title on this one.\n",
        );
        expected_issues += 2;
    }
    write(root, "tools/github/issues.yaml", &issues);
    write(
        root,
        "tools/github/idmap.tsv",
        "# task_id\tissue_number\tissue_db_id\nM-0099\t4242\t99887766\n",
    );

    if defects {
        write(
            root,
            ".claude/skills/nofront/SKILL.md",
            "# No Frontmatter\n\nbody\n",
        );
        write(
            root,
            ".claude/skills/noname/SKILL.md",
            "---\ndescription: has a description but no name\n---\n\nbody\n",
        );
    }

    Expected {
        docs: 4, // RFC-0099, ADR-099, DN-99, Some-Spec
        research: 1,
        issues: expected_issues,
        changelog: 3, // ## [Unreleased] + 2 ### entries
        skills: 1,    // only the valid one is indexed
    }
}

/// Build a real Layer-1 report from a hermetic, defect-free mini-corpus — the shared oracle for the
/// M-1017 front tests (known ids: `M-0099`, `RFC-0099`, `E99-1`; statuses `todo`/`Accepted`/…).
/// Returns the corpus root (for a `refresh` index path) and the built, canonically-sorted report.
pub(crate) fn corpus_report(tag: &str) -> (PathBuf, crate::TeroIndexReport) {
    let root = temp_dir(tag);
    write_corpus(&root, false);
    let report = crate::build_tero_index(&root).expect("fixture corpus builds");
    (root, report)
}

/// Emit `report` as an `index.json` under `root/tero-index/` and return that file's path — for the
/// `refresh` tests, which reload the served index from disk (`load_report`).
pub(crate) fn emit_index(root: &Path, report: &crate::TeroIndexReport) -> PathBuf {
    let dir = root.join("tero-index");
    crate::write_json(report, &dir).expect("fixture index emits");
    dir.join("index.json")
}
