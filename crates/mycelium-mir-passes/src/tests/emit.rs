//! Tests for `crate::emit` — naive fully-owned RC-emission (MEM-4·B0).

use mycelium_core::Node;

use crate::balance::check_balance;
use crate::emit::{count_occurrences, emit_owned, EmitError};
use crate::rc_ir::{Mode, RcNode};

use super::common::{app, c, lam, let_, op, var};

/// Count `Dup`/`Drop` wrapper nodes anywhere in an `RcNode` (test-only structural probe).
fn count_rc_ops(node: &RcNode) -> (usize, usize) {
    fn go(n: &RcNode, dups: &mut usize, drops: &mut usize) {
        match n {
            RcNode::Const(_) | RcNode::Var(_) | RcNode::Borrow(_) | RcNode::MoveUnique(_) => {}
            RcNode::Dup { body, .. } => {
                *dups += 1;
                go(body, dups, drops);
            }
            RcNode::Drop { body, .. } | RcNode::DropAfter { body, .. } => {
                *drops += 1;
                go(body, dups, drops);
            }
            RcNode::Let { bound, body, .. } => {
                go(bound, dups, drops);
                go(body, dups, drops);
            }
            RcNode::Op { args, .. } | RcNode::Construct { args, .. } => {
                for a in args {
                    go(a, dups, drops);
                }
            }
            RcNode::Swap { src, .. } => go(src, dups, drops),
            RcNode::Match {
                scrutinee,
                alts,
                default,
            } => {
                go(scrutinee, dups, drops);
                for alt in alts {
                    match alt {
                        crate::rc_ir::RcAlt::Ctor { body, .. }
                        | crate::rc_ir::RcAlt::Lit { body, .. } => go(body, dups, drops),
                    }
                }
                if let Some(d) = default {
                    go(d, dups, drops);
                }
            }
            RcNode::Lam { body, .. } => go(body, dups, drops),
            RcNode::App { func, arg } => {
                go(func, dups, drops);
                go(arg, dups, drops);
            }
        }
    }
    let (mut d, mut r) = (0, 0);
    go(node, &mut d, &mut r);
    (d, r)
}

// ── occurrence counting (shadowing-aware) ────────────────────────────────────

#[test]
fn occurrences_count_free_uses() {
    // op(x, x, x) has three free uses of x.
    let body = op("p", vec![var("x"), var("x"), var("x")]);
    assert_eq!(count_occurrences(&"x".to_owned(), &body), 3);
}

#[test]
fn occurrences_respect_shadowing() {
    // let x = c in (let x = c in x): the inner `x` shadows the outer, so the OUTER x has 0 free
    // uses in the body. Mutant witness: ignoring shadowing would count 1.
    let inner = let_("x", c(), var("x"));
    let outer_body = inner;
    assert_eq!(
        count_occurrences(&"x".to_owned(), &outer_body),
        0,
        "the inner binder shadows the outer x (A4-01)"
    );
}

// ── emission shape ───────────────────────────────────────────────────────────

#[test]
fn const_and_var_pass_through() {
    assert_eq!(
        emit_owned(&c()).unwrap(),
        RcNode::Const(super::common::val(true))
    );
    assert_eq!(emit_owned(&var("x")).unwrap(), RcNode::Var("x".to_owned()));
}

#[test]
fn let_single_use_emits_no_dup_no_drop() {
    // let x = c in x  → k == 1 → 0 dups, 0 drops.
    let n = let_("x", c(), var("x"));
    let rc = emit_owned(&n).unwrap();
    assert_eq!(
        count_rc_ops(&rc),
        (0, 0),
        "one use needs no Dup and no Drop"
    );
    check_balance(&rc).expect("must balance");
}

#[test]
fn let_unused_emits_one_drop() {
    // let x = c in c  → k == 0 → exactly one Drop, no Dup.
    let n = let_("x", c(), c());
    let rc = emit_owned(&n).unwrap();
    assert_eq!(
        count_rc_ops(&rc),
        (0, 1),
        "an unused binding is reclaimed by exactly one Drop (never leaked, G2)"
    );
    check_balance(&rc).expect("must balance");
}

#[test]
fn let_three_uses_emits_two_dups() {
    // let x = c in p(x, x, x)  → k == 3 → 2 dups, 0 drops.
    let n = let_("x", c(), op("p", vec![var("x"), var("x"), var("x")]));
    let rc = emit_owned(&n).unwrap();
    assert_eq!(
        count_rc_ops(&rc),
        (2, 0),
        "three uses need two Dups (one reference per use)"
    );
    check_balance(&rc).expect("must balance");
}

#[test]
fn lam_param_balanced_owned() {
    // λx. x  → param used once → owned, no dup/drop; mode Owned.
    let rc = emit_owned(&lam("x", var("x"))).unwrap();
    match &rc {
        RcNode::Lam { mode, .. } => assert_eq!(*mode, Mode::Owned, "B0 emits Owned params"),
        other => panic!("expected Lam, got {other:?}"),
    }
    assert_eq!(count_rc_ops(&rc), (0, 0));
    check_balance(&rc).expect("must balance");

    // λx. c  → param unused → one Drop.
    let rc0 = emit_owned(&lam("x", c())).unwrap();
    assert_eq!(count_rc_ops(&rc0), (0, 1), "unused param dropped");
    check_balance(&rc0).expect("must balance");
}

#[test]
fn shadowing_emits_drop_for_outer_and_balances() {
    // let x = c in (let x = c in x): outer x unused (shadowed) → outer Drop; inner x used once.
    let n = let_("x", c(), let_("x", c(), var("x")));
    let rc = emit_owned(&n).unwrap();
    // One Drop (outer x), no dups.
    assert_eq!(count_rc_ops(&rc), (0, 1));
    check_balance(&rc).expect("shadowed emission must still balance");
}

// ── recursion is refused (never-silent, G2) ──────────────────────────────────

#[test]
fn fix_is_unsupported_not_mis_emitted() {
    let fix = Node::Fix {
        name: "f".to_owned(),
        body: Box::new(var("f")),
    };
    assert_eq!(
        emit_owned(&fix),
        Err(EmitError::UnsupportedNode("Fix")),
        "recursion must be refused explicitly, not silently mis-emitted (G2)"
    );
}

#[test]
fn fixgroup_is_unsupported() {
    let fg = Node::FixGroup {
        defs: vec![("f".to_owned(), Box::new(var("g")))],
        body: Box::new(var("f")),
    };
    assert_eq!(emit_owned(&fg), Err(EmitError::UnsupportedNode("FixGroup")));
}

#[test]
fn unsupported_propagates_through_subterms() {
    // A Fix nested inside a Let must still surface the error (no silent partial emission).
    let n = let_(
        "x",
        c(),
        Node::Fix {
            name: "f".to_owned(),
            body: Box::new(var("f")),
        },
    );
    assert_eq!(emit_owned(&n), Err(EmitError::UnsupportedNode("Fix")));
}

// ── end-to-end: every emitted term balances ──────────────────────────────────

#[test]
fn emission_output_always_balances() {
    // A battery of first-order terms; emission must produce a balanced IR for each (the core B0
    // property: the naive emission never leaks and never over-releases).
    let terms: Vec<Node> = vec![
        c(),
        var("x"),
        let_("x", c(), var("x")),
        let_("x", c(), c()),
        let_("x", c(), op("p", vec![var("x"), var("x")])),
        lam("x", var("x")),
        lam("x", c()),
        app(lam("x", var("x")), c()),
        let_(
            "y",
            c(),
            let_("x", var("y"), op("p", vec![var("x"), var("y")])),
        ),
    ];
    for (i, t) in terms.iter().enumerate() {
        let rc = emit_owned(t).unwrap_or_else(|e| panic!("term {i} emit failed: {e}"));
        check_balance(&rc).unwrap_or_else(|e| panic!("term {i} unbalanced: {e}"));
    }
}
