//! M-965 (DN-58 §A F-A1/F-A2) — the `Fuse` prelude trait + the semilattice-law checker. One test
//! per law (idempotence/commutativity/associativity — each isolated so a violation of exactly
//! that law, and no other, is what's exercised), plus the prelude-availability and
//! non-enumerable-domain-is-unchecked cases.

use crate::checkty::*;
use crate::parse;

fn env(src: &str) -> Env {
    check_nodule(&parse(src).expect("parses")).expect("checks")
}

fn check_err(src: &str) -> CheckError {
    check_nodule(&parse(src).expect("parses")).expect_err("must fail to check")
}

/// F-A1: `Fuse` is available with **no** `trait Fuse` declaration in source, and a lawful instance
/// (`Flag`/logical-OR — the textbook two-element semilattice) checks cleanly: all three laws hold,
/// so `check_fuse_laws` finds nothing to refuse.
#[test]
fn fuse_prelude_trait_is_builtin_and_a_lawful_instance_checks() {
    env("nodule d;\n\
         type Flag = Off | On;\n\
         impl Fuse[Flag] for Flag {\n\
           fn join(a: Flag, b: Flag) => Flag =\n\
             match a {\n\
               Off => match b { Off => Off, On => On },\n\
               On => match b { Off => On, On => On },\n\
             };\n\
         };\n\
         fn combine(x: Flag, y: Flag) => Flag = fuse(x, y);");
}

/// A user cannot shadow the built-in `Fuse` trait with their own declaration (never a silent
/// shadow of the prelude — G2, mirrors `Bool` redeclaration being refused).
#[test]
fn redeclaring_the_builtin_fuse_trait_is_refused() {
    let err = check_err("nodule d;\ntrait Fuse[T] { fn join(a: T, b: T) => T; };");
    assert!(
        err.message.contains("Fuse") && err.message.contains("built-in"),
        "expected a built-in-redeclaration refusal, got: {}",
        err.message
    );
}

/// Idempotence law: `join(a, a) = a`. A `join` that always returns a fixed constructor is not
/// idempotent for any other input — refused at `impl` time, never at a `fuse` call site.
#[test]
fn idempotence_violation_is_refused_at_definition() {
    let err = check_err(
        "nodule d;\n\
         type Flag = Off | On;\n\
         impl Fuse[Flag] for Flag { fn join(a: Flag, b: Flag) => Flag = Off; };",
    );
    assert!(
        err.message.contains("idempotence"),
        "expected an idempotence-law refusal, got: {}",
        err.message
    );
}

/// Commutativity law: `join(a, b) = join(b, a)`. Left-projection (`join(a, b) = a`) is idempotent
/// and associative but NOT commutative — isolates the commutativity law exactly.
#[test]
fn commutativity_violation_is_refused_at_definition() {
    let err = check_err(
        "nodule d;\n\
         type Flag = Off | On;\n\
         impl Fuse[Flag] for Flag { fn join(a: Flag, b: Flag) => Flag = a; };",
    );
    assert!(
        err.message.contains("commutativity"),
        "expected a commutativity-law refusal, got: {}",
        err.message
    );
}

/// Associativity law: `join(join(a, b), c) = join(a, join(b, c))`. The classic
/// rock/paper/scissors "winner" operation is idempotent (`join(x, x) = x`) and commutative (the
/// winner of a pair doesn't depend on argument order) but famously NOT associative — isolates the
/// associativity law exactly (join(join(Rock, Paper), Scissors) = Scissors, but
/// join(Rock, join(Paper, Scissors)) = Rock).
#[test]
fn associativity_violation_is_refused_at_definition() {
    let err = check_err(
        "nodule d;\n\
         type Move = Rock | Paper | Scissors;\n\
         impl Fuse[Move] for Move {\n\
           fn join(a: Move, b: Move) => Move =\n\
             match a {\n\
               Rock => match b { Rock => Rock, Paper => Paper, Scissors => Rock },\n\
               Paper => match b { Rock => Paper, Paper => Paper, Scissors => Scissors },\n\
               Scissors => match b { Rock => Rock, Paper => Scissors, Scissors => Scissors },\n\
             };\n\
         };",
    );
    assert!(
        err.message.contains("associativity"),
        "expected an associativity-law refusal, got: {}",
        err.message
    );
}

/// A `for_ty` with a fielded constructor is not a finite enumerable domain in v0 (DN-58 §A.6
/// F-A3, deferred) — the law checker honestly skips it rather than guessing, so an arbitrary
/// (unverified) `join` still checks. This pins the documented v0 scope boundary, not a bug.
#[test]
fn non_enumerable_for_ty_is_left_unchecked_not_refused() {
    env("nodule d;\n\
         type Wrap = Mk(Binary{8});\n\
         impl Fuse[Wrap] for Wrap { fn join(a: Wrap, b: Wrap) => Wrap = a; };");
}

/// `impl Fuse[T] for T` over a *different* pair of types (arity satisfied, but `trait_args[0]` and
/// `for_ty` diverge) is out of this task's scope to forbid — documented here as a scope pin, not a
/// missing check: `register_instances`'s coherence key is `(trait_name, type_head(for_ty))`
/// only, so nothing currently requires the two to agree structurally beyond both resolving
/// concretely. This test only pins that `Fuse[Flag] for Flag` (the intended "self" idiom) is
/// accepted; it does not assert anything about the mismatched form (no FLAG needed — the checker
/// still only ever runs `join` at `for_ty`'s own domain, so the law check is sound either way).
#[test]
fn fuse_self_idiom_impl_is_accepted() {
    env("nodule d;\n\
         type Sign = Neg | Zero | Pos;\n\
         impl Fuse[Sign] for Sign {\n\
           fn join(a: Sign, b: Sign) => Sign =\n\
             match a {\n\
               Neg => match b { Neg => Neg, Zero => Neg, Pos => Neg },\n\
               Zero => match b { Neg => Neg, Zero => Zero, Pos => Zero },\n\
               Pos => match b { Neg => Neg, Zero => Zero, Pos => Pos },\n\
             };\n\
         };");
}
