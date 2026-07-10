//! Tests for MEM-4 Increment 1 — borrow elision (`emit::emit_elided`) and the differential harness
//! (`eval::differential`), DN-33 §8.1 Q3.

use mycelium_core::Node;

use crate::emit::{borrow_occurrences, emit_elided, emit_owned, is_fully_borrowable};
use crate::eval::{differential, eval, RcError};
use crate::rc_ir::RcNode;

use super::common::{c, let_, op, var};

/// Count `Dup`/`Drop`(+`DropAfter`)/`Borrow` nodes in an `RcNode` (test probe).
fn counts(node: &RcNode) -> (usize, usize, usize) {
    fn go(n: &RcNode, d: &mut usize, dr: &mut usize, b: &mut usize) {
        match n {
            RcNode::Const(_) | RcNode::Var(_) | RcNode::MoveUnique(_) => {}
            RcNode::Borrow(_) => *b += 1,
            RcNode::Dup { body, .. } => {
                *d += 1;
                go(body, d, dr, b);
            }
            RcNode::Drop { body, .. } | RcNode::DropAfter { body, .. } => {
                *dr += 1;
                go(body, d, dr, b);
            }
            RcNode::Let { bound, body, .. } => {
                go(bound, d, dr, b);
                go(body, d, dr, b);
            }
            RcNode::Op { args, .. } | RcNode::Construct { args, .. } => {
                for a in args {
                    go(a, d, dr, b);
                }
            }
            RcNode::Swap { src, .. } => go(src, d, dr, b),
            RcNode::Match { scrutinee, .. } => go(scrutinee, d, dr, b),
            RcNode::Lam { body, .. } => go(body, d, dr, b),
            RcNode::App { func, arg } => {
                go(func, d, dr, b);
                go(arg, d, dr, b);
            }
        }
    }
    let (mut d, mut dr, mut b) = (0, 0, 0);
    go(node, &mut d, &mut dr, &mut b);
    (d, dr, b)
}

// ── borrow analysis ──────────────────────────────────────────────────────────

#[test]
fn borrow_occurrences_count_reader_args() {
    // op(x, x) — both x are reader-primitive args → 2 borrow positions.
    assert_eq!(
        borrow_occurrences(&"x".to_owned(), &op("p", vec![var("x"), var("x")])),
        2
    );
    // a bare x (result/move position) is NOT a borrow position.
    assert_eq!(borrow_occurrences(&"x".to_owned(), &var("x")), 0);
}

#[test]
fn fully_borrowable_classification() {
    // op(x, x): every use is a borrow → fully borrowable.
    assert!(is_fully_borrowable(
        &"x".to_owned(),
        &op("p", vec![var("x"), var("x")])
    ));
    // x as the result (a move) → NOT fully borrowable (it escapes).
    assert!(!is_fully_borrowable(&"x".to_owned(), &var("x")));
    // mixed: one borrow + one move (x flows to the result) → NOT fully borrowable.
    assert!(!is_fully_borrowable(
        &"x".to_owned(),
        &let_("y", op("p", vec![var("x")]), var("x"))
    ));
}

// ── elided emission shape ────────────────────────────────────────────────────

#[test]
fn elided_borrowable_let_uses_borrows_no_dup_one_dropafter() {
    // let x = c in op(x, x, x): fully borrowable → 0 dups, 3 Borrows, 1 DropAfter.
    let n = let_("x", c(), op("p", vec![var("x"), var("x"), var("x")]));
    let rc = emit_elided(&n).unwrap();
    let (dups, drops, borrows) = counts(&rc);
    assert_eq!(dups, 0, "borrow elision removes all Dups");
    assert_eq!(borrows, 3, "each reader use becomes a Borrow");
    assert_eq!(drops, 1, "reclaimed once by a DropAfter");
    // The owned emission, by contrast, has 2 Dups and 0 Borrows.
    let owned = emit_owned(&n).unwrap();
    assert_eq!(counts(&owned), (2, 0, 0));
}

#[test]
fn elided_result_move_stays_owned() {
    // let x = c in x: x is the result (escapes) → NOT borrowable → identical to owned (no Borrow).
    let n = let_("x", c(), var("x"));
    let elided = emit_elided(&n).unwrap();
    assert_eq!(
        counts(&elided),
        (0, 0, 0),
        "a moved-out binding is not borrow-elided"
    );
    assert_eq!(elided, emit_owned(&n).unwrap());
}

// ── differential: elision is semantics-preserving + reduces Dups ─────────────

#[test]
fn differential_preserves_reclamations_over_corpus() {
    // For each straight-line term, the owned and elided emissions must reclaim the SAME multiset of
    // values with no use-after-free, and elision must never increase the Dup count.
    let corpus: Vec<Node> = vec![
        c(),
        let_("x", c(), var("x")),
        let_("x", c(), op("p", vec![var("x"), var("x")])),
        let_("x", c(), op("p", vec![var("x"), var("x"), var("x")])),
        let_("x", c(), op("id", vec![var("x")])),
        // nested: y owns (result), x is borrowed by a reader.
        let_(
            "y",
            c(),
            let_("x", c(), op("p", vec![var("x"), var("x"), var("y")])),
        ),
    ];
    for (i, t) in corpus.iter().enumerate() {
        let owned = emit_owned(t).unwrap();
        let elided = emit_elided(t).unwrap();
        let d = differential(&owned, &elided)
            .unwrap_or_else(|e| panic!("term {i} differential errored: {e}"));
        assert!(
            d.is_semantics_preserving(),
            "term {i}: elision changed the reclamation multiset"
        );
        assert!(
            d.elided_dups <= d.owned_dups,
            "term {i}: elision must not increase Dups ({} -> {})",
            d.owned_dups,
            d.elided_dups
        );
    }
}

#[test]
fn differential_shows_dup_reduction_for_multi_read() {
    // let x = c in op(x, x, x): owned needs 2 Dups; elided needs 0 → 2 removed, same reclamations.
    let t = let_("x", c(), op("p", vec![var("x"), var("x"), var("x")]));
    let d = differential(&emit_owned(&t).unwrap(), &emit_elided(&t).unwrap()).unwrap();
    assert!(d.is_semantics_preserving());
    assert_eq!(d.owned_dups, 2);
    assert_eq!(d.elided_dups, 0);
    assert_eq!(d.dups_removed(), 2, "borrow elision removed both Dups");
}

// ── evaluator soundness witnesses ────────────────────────────────────────────

#[test]
fn eval_elided_term_has_no_use_after_free() {
    // The elided emission of a fully-borrowable term evaluates cleanly (the DropAfter keeps the
    // value live through its borrows).
    let t = let_("x", c(), op("p", vec![var("x"), var("x")]));
    let report = eval(&emit_elided(&t).unwrap()).expect("elided term must evaluate without UAF");
    assert_eq!(report.reclaimed.len(), 1, "x reclaimed exactly once");
}

#[test]
fn eval_detects_use_after_free() {
    // Hand-built BAD term: drop x, then borrow x → use-after-free (mutation witness for the
    // evaluator — if it did not check liveness, an unsound elision would slip through).
    let bad = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(RcNode::Const(super::common::val(true))),
        body: Box::new(RcNode::Drop {
            var: "x".to_owned(),
            body: Box::new(RcNode::Borrow("x".to_owned())),
        }),
    };
    assert_eq!(
        eval(&bad),
        Err(RcError::UseAfterFree("x".to_owned())),
        "borrowing a reclaimed value must be caught (G2)"
    );
}

#[test]
fn eval_detects_double_free() {
    // drop x twice (single owning reference) → reference count goes negative → double free.
    let bad = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(RcNode::Const(super::common::val(true))),
        body: Box::new(RcNode::Drop {
            var: "x".to_owned(),
            body: Box::new(RcNode::Drop {
                var: "x".to_owned(),
                body: Box::new(RcNode::Const(super::common::val(false))),
            }),
        }),
    };
    assert_eq!(eval(&bad), Err(RcError::DoubleFree("x".to_owned())));
}

#[test]
fn eval_rejects_non_straight_line_nodes() {
    // App is outside the straight-line fragment — explicit refusal, never a silent wrong answer.
    let app_node = RcNode::App {
        func: Box::new(RcNode::Const(super::common::val(true))),
        arg: Box::new(RcNode::Const(super::common::val(false))),
    };
    assert_eq!(eval(&app_node), Err(RcError::UnsupportedNode("App")));
}
