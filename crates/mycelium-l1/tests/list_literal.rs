//! RFC-0040 / M-977 — the `[…]` **Vec list literal** desugaring, and its behaviour-neutrality proof.
//!
//! A `[e1, …, en]` literal checked against a **cons-list-shaped** user ADT (exactly two constructors —
//! one nullary "nil", one binary `Cons(A, Self)` "cons") desugars to the right-nested
//! `Cons(e1, Cons(…, Nil))` chain and is checked as that chain. The proof of behaviour-neutrality is
//! **AST identity** (strictly stronger than eval-equivalence): after `check_nodule`, the elaborated fn
//! body of the `[…]` program is **byte-for-byte equal** to the body of the hand-written `Cons` chain —
//! so every downstream path (L1-eval, elaborate→L0-interp, AOT) is identical *by construction*.
//! Guarantee: `Empirical` (verified by execution here + the `mycelium-std-conformance` three-way
//! suites), never `Proven`.

use mycelium_l1::{check_nodule, parse};

/// `Vec[A] = Nil | Cons(A, Vec[A])` prelude — the canonical cons-list ADT.
const VEC_PRELUDE: &str = "nodule d;\ntype Vec[A] = Nil | Cons(A, Vec[A]);\n";

/// The elaborated body of `f` in a checked nodule (panics with a clear message on any failure).
fn checked_body(src: &str) -> mycelium_l1::ast::Expr {
    let nodule = parse(src).unwrap_or_else(|e| panic!("parse failed: {e}\n{src}"));
    let env = check_nodule(&nodule).unwrap_or_else(|e| panic!("check failed: {e}\n{src}"));
    env.fns
        .get("f")
        .unwrap_or_else(|| panic!("no fn `f`"))
        .body
        .clone()
}

/// BEHAVIOR-NEUTRALITY (`Empirical`): a `[…]` literal against a `Vec[_]` context elaborates to the byte-identical AST as the
/// hand-written `Cons` chain — so it is behaviour-neutral by construction (same surface AST after
/// desugaring ⇒ identical eval / elaborate / AOT).
#[test]
fn list_literal_desugars_to_cons_chain_identically() {
    let list = format!(
        "{VEC_PRELUDE}fn f() => Vec[Binary{{8}}] = [0b0000_0001, 0b0000_0010, 0b0000_0011];\n"
    );
    let cons = format!(
        "{VEC_PRELUDE}fn f() => Vec[Binary{{8}}] = \
         Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\n"
    );
    assert_eq!(
        checked_body(&list),
        checked_body(&cons),
        "the desugared list body must equal the hand-written Cons chain (AST identity)"
    );
}

/// An empty `[]` in a `Vec` context desugars to the bare nil constructor (no guessed element type —
/// the context supplies it; contrast the no-context `[]` which is still the RFC-0032 error).
#[test]
fn empty_list_desugars_to_nil() {
    let list = format!("{VEC_PRELUDE}fn f() => Vec[Binary{{8}}] = [];\n");
    let nil = format!("{VEC_PRELUDE}fn f() => Vec[Binary{{8}}] = Nil;\n");
    assert_eq!(checked_body(&list), checked_body(&nil));
}

/// A `[…]` against a `Seq{T, N}` context is UNCHANGED (RFC-0032 D3): the Seq literal is a distinct
/// expected type, so the desugaring never reinterprets it — no ambiguity, never a silent reinterpret.
#[test]
fn seq_literal_still_types_as_seq_when_expected() {
    let src = "nodule d;\nfn f() => Seq{Binary{8}, 2} = [0b0000_0001, 0b0000_0010];\n";
    check_nodule(&parse(src).expect("parses")).expect("Seq literal still checks as Seq");
}

/// A non-list-shaped 2-ctor ADT is NOT a desugaring target — a `[…]` against it is still refused
/// (the structural recogniser requires a nullary nil + a `Cons(A, Self)` recursive cons).
#[test]
fn non_list_two_ctor_adt_is_not_a_desugaring_target() {
    // `Pair` has two ctors but neither is a nullary nil + a recursive `Cons(A, Self)` — must NOT
    // accept `[…]`.
    let src =
        "nodule d;\ntype Pair[A] = MkA(A) | MkB(A, A);\nfn f() => Pair[Binary{8}] = [0b0000_0001];\n";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a `[…]` against a non-cons-list ADT must be refused, not silently reinterpreted"
    );
}
