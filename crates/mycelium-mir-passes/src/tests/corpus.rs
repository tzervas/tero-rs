//! Tests for `crate::corpus` — the MEM-4 measurement / Q5 gate (DN-33 §8.1) and its Increment-2 +
//! audit-trail companion.

use crate::corpus::{
    measure, measure_mem4, measure_mem4_standard, measure_standard, standard_corpus,
};

#[test]
fn standard_corpus_is_a_mix() {
    // The corpus must contain BOTH elision-friendly and elision-neutral terms, so the measured
    // ratio is honest (not a cherry-picked best case).
    let corpus = standard_corpus();
    assert!(corpus.len() >= 8, "corpus should be reasonably sized");
    let report = measure(&corpus).expect("measurement must not fault");
    let with_win = report.rows.iter().filter(|(_, o, e, _)| o > e).count();
    let neutral = report.rows.iter().filter(|(_, o, e, _)| o == e).count();
    assert!(with_win >= 3, "corpus must include elision wins");
    assert!(
        neutral >= 2,
        "corpus must include elision-neutral terms (honest mix)"
    );
}

#[test]
fn q5_gate_elision_reduces_dups_and_preserves_semantics() {
    // THE Q5 GATE: across the representative corpus, borrow elision must (a) preserve semantics for
    // EVERY term (same reclamation multiset, no use-after-free), and (b) measurably reduce the
    // emitted Dup count. Both are required before Increment 2 may be committed (DN-33 §8.1 Q5).
    let report = measure_standard().expect("measurement must not fault");

    assert!(
        report.all_semantics_preserved,
        "every term's elision must be semantics-preserving (Q3)"
    );
    assert!(
        report.elided_dups < report.owned_dups,
        "elision must reduce the aggregate Dup count: owned={}, elided={}",
        report.owned_dups,
        report.elided_dups
    );
    assert!(
        report.reduction_ratio() > 0.0,
        "the Q5 dup-reduction ratio must be positive (got {:.3})",
        report.reduction_ratio()
    );
}

#[test]
fn elided_never_exceeds_owned_per_term() {
    // Per-term monotonicity: elision never INCREASES Dups for any term (it only removes them).
    let report = measure_standard().expect("measurement must not fault");
    for (name, owned, elided, preserved) in &report.rows {
        assert!(
            elided <= owned,
            "term {name}: elision increased Dups ({owned} -> {elided})"
        );
        assert!(
            preserved,
            "term {name}: elision was not semantics-preserving"
        );
    }
}

#[test]
fn report_ratio_is_exact_arithmetic() {
    // The ratio is an exact count ratio (Exact tag): (owned - elided) / owned.
    let report = measure_standard().expect("measurement must not fault");
    let expected = (report.owned_dups - report.elided_dups) as f64 / report.owned_dups as f64;
    assert!((report.reduction_ratio() - expected).abs() < f64::EPSILON);
    assert_eq!(
        report.dups_removed(),
        report.owned_dups - report.elided_dups
    );
}

// ── Increment 2 + audit-trail measurement ────────────────────────────────────

#[test]
fn mem4_reuse_is_sound_and_semantics_preserving_over_corpus() {
    // Across the representative corpus, EVERY `rc == 1` reuse annotation must be machine-verified
    // sound (reached at rc == 1 — no `UnsoundUnique`/UAF) and reclaim the same multiset as the owned
    // emission. This is the Increment-2 analogue of the Q5 gate.
    let report = measure_mem4_standard().expect("measurement must not fault");
    assert!(
        report.all_reuse_sound,
        "every reuse annotation must be machine-verified sound"
    );
    assert!(
        report.all_semantics_preserved,
        "every reuse emission must preserve the owned reclamation multiset"
    );
}

#[test]
fn mem4_corpus_exercises_reuse_sites_and_audit_records() {
    // The measurement must actually find reuse sites (else it proves nothing) AND a non-trivial audit
    // trail. The corpus carries sole-owned moves (`result_move`, `borrow_then_sole_move`,
    // `sole_move_after_drop`), so the aggregate reuse-site count is positive.
    let report = measure_mem4_standard().expect("measurement must not fault");
    assert!(
        report.reuse_sites >= 3,
        "corpus must exercise several reuse sites (got {})",
        report.reuse_sites
    );
    assert!(
        report.reclamations >= report.n_terms,
        "each non-trivial term reclaims at least one value (records={}, n={})",
        report.reclamations,
        report.n_terms
    );
}

#[test]
fn mem4_per_term_rows_are_consistent() {
    // The aggregate fields must equal the sum of the per-term rows (no double-count / drop).
    let report = measure_mem4(&standard_corpus()).expect("measurement must not fault");
    let sites: usize = report.rows.iter().map(|r| r.reuse_sites).sum();
    let records: usize = report.rows.iter().map(|r| r.reclamations).sum();
    assert_eq!(
        sites, report.reuse_sites,
        "reuse-site total must match rows"
    );
    assert_eq!(
        records, report.reclamations,
        "reclamation total must match rows"
    );
    assert_eq!(report.rows.len(), report.n_terms);
    // Every row sound + preserved (the corpus is straight-line, sound by construction — measured).
    for row in &report.rows {
        assert!(row.sound, "term {}: reuse annotation unsound", row.name);
        assert!(row.preserved, "term {}: not semantics-preserving", row.name);
    }
}

#[test]
fn mem4_specific_reuse_sites_are_located() {
    // Pin the reuse annotations to the terms that should carry them (a mutation witness: if the reuse
    // predicate regressed, these specific counts would move).
    let report = measure_mem4_standard().expect("measurement must not fault");
    let row = |name: &str| {
        report
            .rows
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("missing corpus row {name}"))
    };
    // `let x = c in x`: the single move is a sole-owned reuse site.
    assert_eq!(row("result_move").reuse_sites, 1);
    // `let x = c in let y = reads(x,2) in y`: x is borrow-elided (no reuse), y is a sole-owned move.
    assert_eq!(row("borrow_then_sole_move").reuse_sites, 1);
    // A reader-heavy let with no move escape carries NO reuse site (all uses are borrows).
    assert_eq!(row("reader_x4").reuse_sites, 0);
}
