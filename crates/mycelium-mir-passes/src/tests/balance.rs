//! Tests for `crate::balance` — the structural balance invariant (MEM-4·B0 / DN-33 §8.1 Q3).
//!
//! These are the **mutation witnesses** for the invariant: hand-built *unbalanced* IR must be
//! caught (so the check is not vacuous), and the `Borrowed` clause is exercised ahead of Increment 1.

use crate::balance::{check_balance, BalanceError};
use crate::rc_ir::{Mode, RcNode};

use super::common::val;

fn konst() -> RcNode {
    RcNode::Const(val(true))
}

fn rc_var(s: &str) -> RcNode {
    RcNode::Var(s.to_owned())
}

// ── balanced cases ───────────────────────────────────────────────────────────

#[test]
fn owned_single_use_balances() {
    // let x = c in x  →  1 + 0 dups == 1 use + 0 drops. OK.
    let n = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(rc_var("x")),
    };
    check_balance(&n).expect("single use balances");
}

#[test]
fn owned_unused_with_drop_balances() {
    // let x = c in (drop x; c)  →  1 + 0 == 0 uses + 1 drop. OK.
    let n = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(RcNode::drop_one(&"x".to_owned(), konst())),
    };
    check_balance(&n).expect("unused-with-drop balances");
}

#[test]
fn owned_two_uses_with_one_dup_balances() {
    // let x = c in (dup x; p(x, x))  →  1 + 1 dup == 2 uses + 0 drops. OK.
    let body = RcNode::dup_n(
        &"x".to_owned(),
        1,
        RcNode::Op {
            prim: "p".to_owned(),
            args: vec![rc_var("x"), rc_var("x")],
        },
    );
    let n = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(body),
    };
    check_balance(&n).expect("two uses + one dup balances");
}

// ── mutation witnesses: unbalanced IR is detected ────────────────────────────

#[test]
fn owned_missing_dup_is_detected() {
    // let x = c in p(x, x) WITHOUT the dup → 1 + 0 != 2 + 0. Must fail (else the check is vacuous).
    let n = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(RcNode::Op {
            prim: "p".to_owned(),
            args: vec![rc_var("x"), rc_var("x")],
        }),
    };
    match check_balance(&n) {
        Err(BalanceError::OwnedUnbalanced {
            var,
            dups,
            uses,
            drops,
        }) => {
            assert_eq!(var, "x");
            assert_eq!((dups, uses, drops), (0, 2, 0));
        }
        other => panic!("expected OwnedUnbalanced, got {other:?}"),
    }
}

#[test]
fn owned_unused_without_drop_is_detected() {
    // let x = c in c (no drop) → 1 + 0 != 0 + 0. Must fail (a leak would be silent otherwise — G2).
    let n = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(konst()),
    };
    assert!(
        matches!(check_balance(&n), Err(BalanceError::OwnedUnbalanced { .. })),
        "an unused binding with no Drop must be flagged (would otherwise leak)"
    );
}

#[test]
fn owned_extra_drop_is_detected() {
    // let x = c in (drop x; drop x; c) → 1 + 0 != 0 + 2. Over-release must be flagged.
    let inner = RcNode::drop_one(&"x".to_owned(), RcNode::drop_one(&"x".to_owned(), konst()));
    let n = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(inner),
    };
    assert!(matches!(
        check_balance(&n),
        Err(BalanceError::OwnedUnbalanced { drops: 2, .. })
    ));
}

// ── Borrowed clause (forward-compatible with Increment 1) ────────────────────

#[test]
fn borrowed_param_with_no_rcops_is_ok() {
    // λ(borrowed x). x  → a borrowed read needs no Dup/Drop. OK regardless of use count.
    let n = RcNode::Lam {
        param: "x".to_owned(),
        mode: Mode::Borrowed,
        body: Box::new(RcNode::Op {
            prim: "read".to_owned(),
            args: vec![rc_var("x"), rc_var("x")],
        }),
    };
    check_balance(&n).expect("borrowed param with no RC ops is balanced by definition");
}

#[test]
fn borrowed_param_with_a_dup_is_detected() {
    // A borrowed binding must carry no Dup/Drop; one slipping in is a bug (Increment 1 invariant).
    let n = RcNode::Lam {
        param: "x".to_owned(),
        mode: Mode::Borrowed,
        body: Box::new(RcNode::dup_n(&"x".to_owned(), 1, rc_var("x"))),
    };
    match check_balance(&n) {
        Err(BalanceError::BorrowedHasRcOps { var, dups, .. }) => {
            assert_eq!(var, "x");
            assert_eq!(dups, 1);
        }
        other => panic!("expected BorrowedHasRcOps, got {other:?}"),
    }
}

// ── nested binders are all checked ───────────────────────────────────────────

#[test]
fn nested_binders_each_checked() {
    // An OUTER balanced binding wrapping an INNER unbalanced one must still fail (the walk
    // descends into every binding, not just the top).
    let inner_bad = RcNode::Let {
        id: "y".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(RcNode::Op {
            prim: "p".to_owned(),
            args: vec![rc_var("y"), rc_var("y")], // y used twice, no dup → unbalanced
        }),
    };
    let outer = RcNode::Let {
        id: "x".to_owned(),
        bound: Box::new(konst()),
        body: Box::new(RcNode::Drop {
            var: "x".to_owned(),
            body: Box::new(inner_bad),
        }),
    };
    assert!(
        matches!(
            check_balance(&outer),
            Err(BalanceError::OwnedUnbalanced { var, .. }) if var == "y"
        ),
        "the inner unbalanced binding y must be detected"
    );
}
