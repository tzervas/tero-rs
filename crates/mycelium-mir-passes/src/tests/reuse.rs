//! Tests for MEM-4 Increment 2 — the `rc == 1` reuse annotation (`emit::emit_reuse` →
//! `RcNode::MoveUnique`), machine-verified by the evaluator. DN-33 §6 D-3 / §8.1 Q3.

use crate::corpus::standard_corpus;
use crate::emit::{emit_elided, emit_owned, emit_reuse, is_sole_owned_move};
use crate::eval::{count_move_unique, eval, RcError};
use crate::rc_ir::RcNode;

use super::common::{c, let_, op, var};

// ── analysis ─────────────────────────────────────────────────────────────────

#[test]
fn is_sole_owned_move_classification() {
    // `let x = c in x`: x used once, in a move position (the result) → sole-owned move.
    assert!(is_sole_owned_move(&"x".to_owned(), &var("x")));
    // op(x, x): two uses, both borrow positions → NOT a sole-owned move.
    assert!(!is_sole_owned_move(
        &"x".to_owned(),
        &op("p", vec![var("x"), var("x")])
    ));
    // op(x): one use, but a borrow position (reader arg) → NOT a move (it's borrow-elidable).
    assert!(!is_sole_owned_move(
        &"x".to_owned(),
        &op("p", vec![var("x")])
    ));
}

// ── emission ─────────────────────────────────────────────────────────────────

#[test]
fn sole_owned_move_becomes_move_unique() {
    // `let x = c in x`: emit_reuse annotates the single move as MoveUnique; emit_elided does not.
    let n = let_("x", c(), var("x"));
    let reuse = emit_reuse(&n).unwrap();
    assert_eq!(
        count_move_unique(&reuse),
        1,
        "the sole-owned move must be annotated MoveUnique"
    );
    assert_eq!(
        count_move_unique(&emit_elided(&n).unwrap()),
        0,
        "Increment 1 (borrow only) does not annotate reuse"
    );
}

#[test]
fn borrowable_takes_precedence_over_reuse() {
    // op(x, x) is fully borrowable → borrow elision wins; no MoveUnique (the value is read, not moved).
    let n = let_("x", c(), op("p", vec![var("x"), var("x")]));
    let reuse = emit_reuse(&n).unwrap();
    assert_eq!(
        count_move_unique(&reuse),
        0,
        "a borrowed value is not a unique move"
    );
}

#[test]
fn nested_sole_owned_moves_each_annotated() {
    // `let x = c in let y = x in y`: x moved once into y (move), y moved once as the result.
    // Both are sole-owned moves → two MoveUnique annotations.
    let n = let_("x", c(), let_("y", var("x"), var("y")));
    let reuse = emit_reuse(&n).unwrap();
    assert_eq!(
        count_move_unique(&reuse),
        2,
        "both single-move bindings annotated"
    );
}

// ── machine-verified soundness ───────────────────────────────────────────────

#[test]
fn eval_verifies_every_reuse_annotation_on_corpus() {
    // For every corpus term, evaluating the reuse-annotated emission must NOT error — i.e. every
    // MoveUnique really is reached at rc == 1 (the static claim is machine-verified), and the
    // reclamations still match the owned emission (semantics preserved).
    for (name, term) in standard_corpus() {
        let reuse = emit_reuse(&term).unwrap();
        let owned = emit_owned(&term).unwrap();
        let reuse_report =
            eval(&reuse).unwrap_or_else(|e| panic!("term {name}: reuse annotation unsound: {e}"));
        let owned_report = eval(&owned).unwrap_or_else(|e| panic!("term {name}: owned eval: {e}"));
        assert_eq!(
            reuse_report.reclaimed_sorted(),
            owned_report.reclaimed_sorted(),
            "term {name}: reuse annotation changed the reclamation multiset"
        );
    }
}

#[test]
fn unsound_reuse_annotation_is_detected() {
    // Hand-built UNSOUND term: dup x (rc=2), then MoveUnique(x) — the rc is 2, not 1, at the
    // annotation. The evaluator must catch it (mutation witness: without the rc==1 check an unsound
    // reuse annotation would slip through).
    let bad = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(RcNode::Const(super::common::val(true))),
        body: Box::new(RcNode::Dup {
            var: "x".to_owned(),
            body: Box::new(RcNode::MoveUnique("x".to_owned())),
        }),
    };
    match eval(&bad) {
        Err(RcError::UnsoundUnique { var, found }) => {
            assert_eq!(var, "x");
            assert_eq!(found, 2, "the rc was 2, not the claimed 1");
        }
        other => panic!("expected UnsoundUnique, got {other:?}"),
    }
}

#[test]
fn sound_reuse_annotation_at_rc_one_is_accepted() {
    // The well-formed case: a single owning reference, consumed by MoveUnique → accepted, reclaimed.
    let good = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(RcNode::Const(super::common::val(true))),
        body: Box::new(RcNode::MoveUnique("x".to_owned())),
    };
    let report = eval(&good).expect("rc==1 MoveUnique is sound");
    assert_eq!(
        report.reclaimed.len(),
        1,
        "the unique owner reclaims its value"
    );
}

// ── measurement ──────────────────────────────────────────────────────────────

#[test]
fn reuse_annotation_count_is_positive_on_corpus() {
    // Increment 2 finds real reuse sites across the representative corpus (the measurable effect).
    let total: usize = standard_corpus()
        .iter()
        .map(|(_, t)| count_move_unique(&emit_reuse(t).unwrap()))
        .sum();
    assert!(
        total > 0,
        "the reuse annotation must find sole-owner sites on the corpus"
    );
}
