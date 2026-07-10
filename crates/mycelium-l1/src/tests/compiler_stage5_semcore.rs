//! M-740 Stage 5, increment 1 (DN-26 §7.3 row 5 / §9 flag-1) — the self-hosted `compiler.semcore`
//! PARTIAL first increment: the type vocabulary (`Ty`/`Width`/`DataInfo`/`CtorInfo`) plus the
//! Maranget usefulness/decision-tree pipeline, the affine use-once tracker, and the static
//! guarantee-grading pass. `lib/compiler/semcore.myc`'s own header documents in full what is IN
//! this increment and what is DEFERRED (fuse.rs / checkty.rs's checking logic / elab.rs / eval.rs
//! / mono.rs — feasibility-gated on M-986/M-987); this module is the unit differential gate for
//! the ported subset only.
//!
//! **FLAG-semcore-10, RESOLVED (moved in-crate to restore a live-oracle differential).** The
//! original gate lived as an EXTERNAL integration test (`crates/mycelium-l1/tests/
//! compiler_stage5_semcore.rs`), which could only see this crate's `pub` surface — and
//! `usefulness.rs`/`decision.rs`/`affine.rs`/`grade.rs` are `pub(crate) mod`/plain `mod` (unlike
//! every prior stage's `pub mod`), so that external test could not call `usefulness::useful`,
//! `decision::compile`, `affine`'s `Tracker`/`slot`/`outcome` types, or `grade::check_guarantees`
//! directly. It fell back to HAND-DERIVED expected values (`Empirical`, weaker than every other
//! stage's live-oracle posture). This module is the **in-crate** (`#[cfg(test)] mod tests`)
//! replacement: white-box `use crate::<mod>::*` access (the established `src/tests/*.rs`
//! convention — see `src/tests/{usefulness,decision,affine}.rs`) reaches every `pub(crate)` item
//! with **zero visibility change to the trusted logic modules** (none of `usefulness.rs`/
//! `decision.rs`/`affine.rs`/`grade.rs` were touched — only this test module and the one `mod`
//! line in `src/tests/mod.rs` were added).
//!
//! **Per-leg live-oracle status (the honest accounting VR-5 requires):**
//! - **usefulness legs** — fully LIVE: each case calls the real `crate::usefulness::useful` on the
//!   same registry/matrix the `.myc` driver encodes, and the verdict/witness asserted against the
//!   `.myc` port is the ORACLE's own computed result (never a literal a human reasoned out).
//! - **decision legs** — fully LIVE: `crate::decision::compile` + `crate::decision::has_reachable_fail`
//!   compute the real tree/verdict; arm routing is checked against a small local `eval_tree`
//!   reference walker (duplicated from `src/tests/decision.rs`'s own — that sibling test module's
//!   helpers are private to `crate::tests::decision`, not reachable from this sibling module, so a
//!   small, honest re-implementation stands in, same shape as `compiler_stage3.rs`'s own `fp`
//!   mirror module).
//! - **affine legs** — fully LIVE: `crate::affine::Tracker::{seeded, use_at}` and the free
//!   `crate::affine::union_merge_into` are called directly; the `.myc` port's outcome/slot codes
//!   are asserted against the real tracker's computed outcome.
//! - **grade legs** — LIVE, via a documented indirection (**FLAG-semcore-10-b, a new finding this
//!   leaf surfaced**): the only `pub(crate)` entry point into `grade.rs` is
//!   `crate::grade::check_guarantees`, which returns `Result<(), CheckError>` — pass/fail against
//!   a function's OWN declared return demand, not the raw computed `Strength`. The per-body walk
//!   that computes the raw grade (`Gx::grade`, `check_fn_grades`) has NO visibility modifier, so it
//!   is private to `crate::grade` itself and not reachable even from this in-crate sibling module.
//!   The exact live grade is recovered by **probing**: since `Strength::satisfies` is monotone in
//!   the demand and the lattice has 4 total-ordered levels (`Exact ⊐ Proven ⊐ Empirical ⊐
//!   Declared`), re-checking the SAME body under a synthetic return-demand at each level (strongest
//!   first) and taking the first that still passes recovers the body's exact grade — no fabricated
//!   constant, every bit sourced from a real `check_guarantees` call. The two demand-VIOLATION
//!   cases need no probing at all (the failure is internal to an ascription/argument check, so
//!   `check_guarantees`'s Err is already the live verdict, at any demand). **FLAGGED UP** for the
//!   maintainer/DN-26 owner exactly as the original FLAG-semcore-10 was: lifting `Gx`/
//!   `check_fn_grades` to `pub(crate)` would let a future revision skip the probing indirection and
//!   read the grade directly.
//!
//! M-981 applies as in every prior self-hosted-compiler-scale stage: only the L1-eval leg is
//! exercised (the L0-substitution interpreter / AOT three-way smoke is skipped for this partial
//! increment — every input here is a small synthetic fixture, not drawn from a corpus, so the
//! marginal value of a three-way leg is low relative to its eval-depth cost, M-987).

use std::collections::BTreeMap;

use crate::affine::{union_merge_into, Slot, Tracker, UseOutcome, UseSite};
use crate::ast::{BaseType, Expr, FnDecl, FnSig, Literal, Param, Path, Strength, TypeRef, Vis};
use crate::checkty::{check_nodule, CtorInfo, DataInfo, Ty};
use crate::decision::{compile, has_reachable_fail, Head, Tree};
use crate::elab::build_registry;
use crate::eval::Evaluator;
use crate::grade::check_guarantees;
use crate::mono::monomorphize;
use crate::parse;
use crate::usefulness::{render, useful, Pat};
use mycelium_core::Payload;

/// Extract a `Binary{N}` `CoreValue`'s bits as a `u32` (MSB-first) — the established convention
/// from every prior stage's harness (`compiler_stage1.rs`'s own `core_bits_as_u32`).
fn core_bits_as_u32(v: &mycelium_core::CoreValue) -> u32 {
    let repr_val = v
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr CoreValue, got {v:?}"));
    match repr_val.payload() {
        Payload::Bits(bits) => bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b)),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

const SEMCORE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/semcore.myc"
));

/// The shared driver prelude: small `.myc` encoder helpers every scenario's `main()` calls, plus
/// two small shared fixtures (`opt2_types`, `optlist_types`). Kept separate from each scenario's
/// own `main()` so every test only differs in the one expression under test.
fn driver_prelude() -> String {
    r#"
// ---- shared fixtures -----------------------------------------------------------------------
// Opt2: a 2-nullary-ctor enum (like `Bool`) — the smallest possible finite signature.
fn opt2_types() => Vec[DataInfo] =
  Cons(DI("Opt2", Nil, Cons(CI("OA", Nil), Cons(CI("OB", Nil), Nil))), Nil);

// OptList: a recursive 1-field-ctor + nullary-ctor enum (like `Vec`) — exercises a non-nullary
// constructor's field-type expansion (`Vec[DataInfo]` order: OptList before Opt2, since ONil's
// field references Opt2 and lookup is order-independent — the checker's own shell-first
// registration note applies equally to this VALUE-level registry list, which is looked up by
// name, not position).
fn optlist_types() => Vec[DataInfo] =
  Cons(DI("OptList", Nil, Cons(CI("OCons", Cons(TyData("Opt2", Nil), Cons(TyData("OptList", Nil), Nil))), Cons(CI("ONil", Nil), Nil))),
  Cons(DI("Opt2", Nil, Cons(CI("OA", Nil), Cons(CI("OB", Nil), Nil))), Nil));

// ---- usefulness verdict encoders -----------------------------------------------------------
fn useful_verdict(res: Result[Option[Vec[Pat]], Bytes]) => Binary{32} =
  match res {
    Err(_) => 0b0000_0000_0000_0000_0000_0000_0000_0011,
    Ok(o) => match o {
      None => 0b0000_0000_0000_0000_0000_0000_0000_0000,
      Some(_) => 0b0000_0000_0000_0000_0000_0000_0000_0001
    }
  };

fn useful_witness_is(res: Result[Option[Vec[Pat]], Bytes], want: Bytes) => Binary{32} =
  match res {
    Err(_) => 0b0000_0000_0000_0000_0000_0000_0000_0000,
    Ok(o) => match o {
      None => 0b0000_0000_0000_0000_0000_0000_0000_0000,
      Some(w) => match bytes_eq(render_list(w), want) {
        0b1 => 0b0000_0000_0000_0000_0000_0000_0000_0001,
        _ => 0b0000_0000_0000_0000_0000_0000_0000_0000
      }
    }
  };

// ---- decision verdict encoders -------------------------------------------------------------
fn compile_ok(res: Result[Tree, Bytes]) => Tree =
  match res { Ok(t) => t, Err(_) => Fail };

fn bool_code(b: Bool) => Binary{32} =
  match b { True => 0b0000_0000_0000_0000_0000_0000_0000_0001, False => 0b0000_0000_0000_0000_0000_0000_0000_0000 };

// tree_arm_code: the routed arm index, or the sentinel `0xFFFF_FFFF` for a reached `Fail` (arm
// indices in every fixture below are 0/1, so the all-ones sentinel cannot collide).
fn tree_arm_code(o: Option[Binary{32}]) => Binary{32} =
  match o { None => 0b1111_1111_1111_1111_1111_1111_1111_1111, Some(a) => a };

// ---- affine verdict encoders ----------------------------------------------------------------
fn outcome_code(o: UseOutcome) => Binary{32} =
  match o {
    NotAffine => 0b0000_0000_0000_0000_0000_0000_0000_0000,
    FirstUse => 0b0000_0000_0000_0000_0000_0000_0000_0001,
    DoubleUse(_, _, _) => 0b0000_0000_0000_0000_0000_0000_0000_0010
  };

fn slot_code(s: Slot) => Binary{32} =
  match s {
    Skip => 0b0000_0000_0000_0000_0000_0000_0000_0000,
    Live(_) => 0b0000_0000_0000_0000_0000_0000_0000_0001,
    Moved(_, _) => 0b0000_0000_0000_0000_0000_0000_0000_0010
  };

fn slots_first(v: Vec[Slot]) => Slot =
  match v { Cons(h, _) => h, Nil => Skip };

// ---- grade verdict encoders ------------------------------------------------------------------
fn strength_code(s: Strength) => Binary{32} =
  match s {
    GDeclared => 0b0000_0000_0000_0000_0000_0000_0000_0000,
    GEmpirical => 0b0000_0000_0000_0000_0000_0000_0000_0001,
    GProven => 0b0000_0000_0000_0000_0000_0000_0000_0010,
    GExact => 0b0000_0000_0000_0000_0000_0000_0000_0011
  };

// grade_result_code: the sentinel `255` for `Err` (a real grading violation) — no valid
// `Strength` code below in this file's fixtures reaches that value.
fn grade_result_code(res: Result[Strength, CheckError]) => Binary{32} =
  match res {
    Err(_) => 0b0000_0000_0000_0000_0000_0000_1111_1111,
    Ok(s) => strength_code(s)
  };

fn bytes_ty() => TypeRef = TR(KwBytes, None);
fn bytes_ty_g(g: Strength) => TypeRef = TR(KwBytes, Some(g));
fn one_path(name: Bytes) => Path = Pth(Cons(name, Nil));
"#
    .to_owned()
}

fn program(driver: &str) -> String {
    format!("{SEMCORE_SRC}\n{}\n{driver}", driver_prelude())
}

/// L1-eval-only assertion (the M-981 convention every self-hosted-compiler-scale stage uses).
fn assert_l1_only_u32(label: &str, src: &str, expected_u32: u32) {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));
    let got = core_bits_as_u32(&l1_core);
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the LIVE-ORACLE expected value {expected_u32}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The structural gate: `semcore.myc` parses and type-checks green (no driver needed).
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn semcore_myc_parses_and_checks() {
    let nodule = parse(SEMCORE_SRC).unwrap_or_else(|e| panic!("semcore.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("semcore.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// usefulness.rs: `useful` (LIVE — `crate::usefulness::useful` computes each verdict/witness
// directly; the `.myc` port's result is asserted against THAT, not a hand-derived constant).
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn ctor(n: &str, subs: Vec<Pat>) -> Pat {
    Pat::Ctor(n.to_owned(), subs)
}

/// Opt2: a 2-nullary-ctor enum (like `Bool`) — mirrors the `.myc` driver's `opt2_types()`.
fn opt2_registry() -> BTreeMap<String, DataInfo> {
    let mut m = BTreeMap::new();
    m.insert(
        "Opt2".to_owned(),
        DataInfo {
            name: "Opt2".to_owned(),
            params: vec![],
            ctors: vec![
                CtorInfo {
                    name: "OA".to_owned(),
                    fields: vec![],
                },
                CtorInfo {
                    name: "OB".to_owned(),
                    fields: vec![],
                },
            ],
        },
    );
    m
}

/// OptList: a recursive 1-field-ctor + nullary-ctor enum (like `Vec`) — mirrors the `.myc`
/// driver's `optlist_types()`.
fn optlist_registry() -> BTreeMap<String, DataInfo> {
    let mut m = opt2_registry();
    m.insert(
        "OptList".to_owned(),
        DataInfo {
            name: "OptList".to_owned(),
            params: vec![],
            ctors: vec![
                CtorInfo {
                    name: "OCons".to_owned(),
                    fields: vec![
                        Ty::Data("Opt2".to_owned(), vec![]),
                        Ty::Data("OptList".to_owned(), vec![]),
                    ],
                },
                CtorInfo {
                    name: "ONil".to_owned(),
                    fields: vec![],
                },
            ],
        },
    );
    m
}

/// Exhaustive: the matrix already covers both `Opt2` constructors (`OA`, `OB`), so `_` is NOT
/// useful w.r.t. it — LIVE: `crate::usefulness::useful` computes the verdict directly.
#[test]
fn useful_exhaustive_two_ctors() {
    let types = opt2_registry();
    let matrix = vec![vec![ctor("OA", vec![])], vec![ctor("OB", vec![])]];
    let col_types = vec![Ty::Data("Opt2".to_owned(), vec![])];
    let oracle =
        useful(&types, &matrix, &[Pat::Wild], &col_types).expect("within the recursion budget");
    let expected = u32::from(oracle.is_some());
    let driver = r#"
fn main() => Binary{32} =
  useful_verdict(useful(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Cons(Cons(MpCtor("OB", Nil), Nil), Nil)),
    Cons(MpWild, Nil),
    Cons(TyData("Opt2", Nil), Nil)));
"#;
    assert_l1_only_u32(
        "useful: OA+OB matrix, _ vs the LIVE-ORACLE verdict",
        &program(driver),
        expected,
    );
}

/// Exhaustive (redundancy-flavored): a PRIOR wildcard row already subsumes everything, so a later
/// arm's own concrete pattern is NOT useful w.r.t. it — LIVE-checked against `useful`.
#[test]
fn useful_exhaustive_prior_wildcard_subsumes() {
    let types = opt2_registry();
    let matrix = vec![vec![Pat::Wild]];
    let col_types = vec![Ty::Data("Opt2".to_owned(), vec![])];
    let q = vec![ctor("OA", vec![])];
    let oracle = useful(&types, &matrix, &q, &col_types).expect("within the recursion budget");
    let expected = u32::from(oracle.is_some());
    let driver = r#"
fn main() => Binary{32} =
  useful_verdict(useful(opt2_types(),
    Cons(Cons(MpWild, Nil), Nil),
    Cons(MpCtor("OA", Nil), Nil),
    Cons(TyData("Opt2", Nil), Nil)));
"#;
    assert_l1_only_u32(
        "useful: prior wildcard row subsumes OA vs the LIVE-ORACLE verdict",
        &program(driver),
        expected,
    );
}

/// Non-exhaustive: the matrix covers only `OA`, so `_` IS useful w.r.t. it — both the verdict AND
/// the rendered witness are asserted against the LIVE `useful`/`render` oracle.
#[test]
fn useful_non_exhaustive_missing_ob() {
    let types = opt2_registry();
    let matrix = vec![vec![ctor("OA", vec![])]];
    let col_types = vec![Ty::Data("Opt2".to_owned(), vec![])];
    let oracle =
        useful(&types, &matrix, &[Pat::Wild], &col_types).expect("within the recursion budget");
    let expected_verdict = u32::from(oracle.is_some());
    let driver = r#"
fn main() => Binary{32} =
  useful_verdict(useful(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Nil),
    Cons(MpWild, Nil),
    Cons(TyData("Opt2", Nil), Nil)));
"#;
    assert_l1_only_u32(
        "useful: OA-only matrix, _ vs the LIVE-ORACLE verdict",
        &program(driver),
        expected_verdict,
    );

    let witness = oracle.expect("non-exhaustive: the LIVE oracle must return a witness");
    let want: String = witness.iter().map(render).collect::<Vec<_>>().join(", ");
    let driver_witness = format!(
        r#"
fn main() => Binary{{32}} =
  useful_witness_is(useful(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Nil),
    Cons(MpWild, Nil),
    Cons(TyData("Opt2", Nil), Nil)), "{want}");
"#
    );
    assert_l1_only_u32(
        "useful: OA-only matrix witness vs the LIVE-ORACLE rendered witness",
        &program(&driver_witness),
        1,
    );
}

/// Non-exhaustive over a 2-column matrix (exercising the recursive column-2 case + the
/// non-nullary-constructor field expansion via `OptList::OCons`) — the witness is asserted
/// against the LIVE `useful`/`render` oracle, not a hand-traced derivation.
#[test]
fn useful_non_exhaustive_two_columns() {
    let types = optlist_registry();
    let matrix = vec![vec![ctor("ONil", vec![]), Pat::Wild]];
    let col_types = vec![
        Ty::Data("OptList".to_owned(), vec![]),
        Ty::Data("Opt2".to_owned(), vec![]),
    ];
    let q = vec![Pat::Wild, Pat::Wild];
    let oracle = useful(&types, &matrix, &q, &col_types).expect("within the recursion budget");
    let witness = oracle.expect("non-exhaustive: the LIVE oracle must return a witness");
    let want: String = witness.iter().map(render).collect::<Vec<_>>().join(", ");
    let driver = format!(
        r#"
fn main() => Binary{{32}} =
  useful_witness_is(useful(optlist_types(),
    Cons(Cons(MpCtor("ONil", Nil), Cons(MpWild, Nil)), Nil),
    Cons(MpWild, Cons(MpWild, Nil)),
    Cons(TyData("OptList", Nil), Cons(TyData("Opt2", Nil), Nil))), "{want}");
"#
    );
    assert_l1_only_u32(
        "useful: OptList ONil-only 2-column matrix witness vs the LIVE-ORACLE rendered witness",
        &program(&driver),
        1,
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// decision.rs: `compile` + `has_reachable_fail` (LIVE — `crate::decision::compile` builds the
// real tree; arm routing is checked with a small local reference walker, `eval_tree` below,
// duplicated from `src/tests/decision.rs`'s own private helper of the same shape — that sibling
// test module's items are not reachable from here, mirroring `compiler_stage3.rs`'s own `fp`
// mirror-module precedent for a small, honest, test-only duplication).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Project the sub-value of `value` at `occurrence` (`value` is a concrete [`Pat`] — no `Wild`).
fn project<'a>(value: &'a Pat, occurrence: &[usize]) -> &'a Pat {
    let mut v = value;
    for &j in occurrence {
        match v {
            Pat::Ctor(_, subs) => v = &subs[j],
            _ => panic!("occurrence {occurrence:?} steps into a non-constructor value"),
        }
    }
    v
}

fn head_matches(head: &Head, value: &Pat) -> bool {
    match (head, value) {
        (Head::Ctor(n, a), Pat::Ctor(m, subs)) => n == m && *a == subs.len(),
        (Head::Lit(k), Pat::Lit(j)) => k == j,
        _ => false,
    }
}

/// Evaluate a decision tree against a concrete value — the test-only reference evaluator that
/// lets this leaf check `compile`'s tree against a concrete routing, exactly as `src/tests/
/// decision.rs`'s own (private, unreachable-from-here) `eval_tree` does.
fn eval_tree(tree: &Tree, value: &Pat) -> Option<usize> {
    match tree {
        Tree::Leaf(a) => Some(*a),
        Tree::Fail => None,
        Tree::Switch {
            occurrence,
            cases,
            default,
        } => {
            let sub = project(value, occurrence);
            match cases.iter().find(|(h, _)| head_matches(h, sub)) {
                Some((_, subtree)) => eval_tree(subtree, value),
                None => default.as_deref().and_then(|d| eval_tree(d, value)),
            }
        }
    }
}

/// `tree_arm_code`'s Rust-side mirror: an `Option<usize>` arm index -> its `Binary{32}` code (the
/// `0xFFFF_FFFF` sentinel for an unreached/`None` arm, matching the `.myc` `tree_arm_code` encoder).
fn arm_code(arm: Option<usize>) -> u32 {
    match arm {
        None => 0xFFFF_FFFF,
        Some(a) => a as u32,
    }
}

/// The fixed inputs shared by every `compile`-driven test below (the `Opt2` registry, its
/// exhaustive 2-arm `[[OA]] -> 0, [[OB]] -> 1]` matrix, the root occurrence, and its column
/// type) — a small named struct rather than a 4-tuple return (clippy::type_complexity).
struct Opt2TwoArmMatrix {
    types: BTreeMap<String, DataInfo>,
    matrix: Vec<Vec<Pat>>,
    occ: Vec<Vec<usize>>,
    tys: Vec<Ty>,
}

fn opt2_two_arm_matrix() -> Opt2TwoArmMatrix {
    Opt2TwoArmMatrix {
        types: opt2_registry(),
        matrix: vec![vec![ctor("OA", vec![])], vec![ctor("OB", vec![])]],
        occ: vec![vec![]],
        tys: vec![Ty::Data("Opt2".to_owned(), vec![])],
    }
}

/// Exhaustive: `compile` on `[[OA]] -> arm0, [[OB]] -> arm1` produces a tree with no reachable
/// `Fail` — LIVE: `crate::decision::{compile, has_reachable_fail}` compute the verdict directly.
#[test]
fn compile_exhaustive_no_reachable_fail() {
    let Opt2TwoArmMatrix {
        types,
        matrix,
        occ,
        tys,
    } = opt2_two_arm_matrix();
    let tree = compile(&types, &matrix, &[0, 1], &occ, &tys)
        .expect("compiles within the RFC-0041 recursion budget");
    let expected = u32::from(has_reachable_fail(&tree));
    let driver = r#"
fn main() => Binary{32} =
  bool_code(has_reachable_fail(compile_ok(compile(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Cons(Cons(MpCtor("OB", Nil), Nil), Nil)),
    Cons(0b0000_0000_0000_0000_0000_0000_0000_0000, Cons(0b0000_0000_0000_0000_0000_0000_0000_0001, Nil)),
    Cons(Nil, Nil),
    Cons(TyData("Opt2", Nil), Nil)))));
"#;
    assert_l1_only_u32(
        "compile: exhaustive OA/OB match vs the LIVE-ORACLE reachable-Fail verdict",
        &program(driver),
        expected,
    );
}

/// Both concrete inputs route to their LIVE-ORACLE-computed arm via the local `eval_tree`
/// reference walker on the same exhaustive tree `compile` produces.
#[test]
fn compile_exhaustive_routes_to_expected_arms() {
    let Opt2TwoArmMatrix {
        types,
        matrix,
        occ,
        tys,
    } = opt2_two_arm_matrix();
    let tree = compile(&types, &matrix, &[0, 1], &occ, &tys)
        .expect("compiles within the RFC-0041 recursion budget");

    let expected_oa = arm_code(eval_tree(&tree, &ctor("OA", vec![])));
    let driver_oa = r#"
fn main() => Binary{32} =
  tree_arm_code(tree_eval(compile_ok(compile(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Cons(Cons(MpCtor("OB", Nil), Nil), Nil)),
    Cons(0b0000_0000_0000_0000_0000_0000_0000_0000, Cons(0b0000_0000_0000_0000_0000_0000_0000_0001, Nil)),
    Cons(Nil, Nil),
    Cons(TyData("Opt2", Nil), Nil))), MpCtor("OA", Nil)));
"#;
    assert_l1_only_u32(
        "compile: OA routes to the LIVE-ORACLE arm",
        &program(driver_oa),
        expected_oa,
    );

    let expected_ob = arm_code(eval_tree(&tree, &ctor("OB", vec![])));
    let driver_ob = r#"
fn main() => Binary{32} =
  tree_arm_code(tree_eval(compile_ok(compile(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Cons(Cons(MpCtor("OB", Nil), Nil), Nil)),
    Cons(0b0000_0000_0000_0000_0000_0000_0000_0000, Cons(0b0000_0000_0000_0000_0000_0000_0000_0001, Nil)),
    Cons(Nil, Nil),
    Cons(TyData("Opt2", Nil), Nil))), MpCtor("OB", Nil)));
"#;
    assert_l1_only_u32(
        "compile: OB routes to the LIVE-ORACLE arm",
        &program(driver_ob),
        expected_ob,
    );
}

/// Incomplete: `compile` on a 1-arm matrix covering only `OA` produces a tree WITH a reachable
/// `Fail` — LIVE-checked, plus the uncovered `OB` input's routing to that `Fail`.
#[test]
fn compile_incomplete_has_reachable_fail() {
    let types = opt2_registry();
    let matrix = vec![vec![ctor("OA", vec![])]];
    let occ = vec![vec![]];
    let tys = vec![Ty::Data("Opt2".to_owned(), vec![])];
    let tree = compile(&types, &matrix, &[0], &occ, &tys)
        .expect("compiles within the RFC-0041 recursion budget");
    let expected_fail = u32::from(has_reachable_fail(&tree));
    let driver = r#"
fn main() => Binary{32} =
  bool_code(has_reachable_fail(compile_ok(compile(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Nil),
    Cons(0b0000_0000_0000_0000_0000_0000_0000_0000, Nil),
    Cons(Nil, Nil),
    Cons(TyData("Opt2", Nil), Nil)))));
"#;
    assert_l1_only_u32(
        "compile: OA-only match vs the LIVE-ORACLE reachable-Fail verdict",
        &program(driver),
        expected_fail,
    );

    // The uncovered `OB` input routes to that reachable `Fail` (tree_eval -> None -> sentinel).
    let expected_ob = arm_code(eval_tree(&tree, &ctor("OB", vec![])));
    let driver_fail = r#"
fn main() => Binary{32} =
  tree_arm_code(tree_eval(compile_ok(compile(opt2_types(),
    Cons(Cons(MpCtor("OA", Nil), Nil), Nil),
    Cons(0b0000_0000_0000_0000_0000_0000_0000_0000, Nil),
    Cons(Nil, Nil),
    Cons(TyData("Opt2", Nil), Nil))), MpCtor("OB", Nil)));
"#;
    assert_l1_only_u32(
        "compile: OB (uncovered) routes to the LIVE-ORACLE reachable Fail",
        &program(driver_fail),
        expected_ob,
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// affine.rs: `Tracker::{seeded, use_at}` (first use, then a double-consume) + the free
// `union_merge_into` (conservative branch merge) — LIVE, called directly.
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn outcome_code(o: &UseOutcome) -> u32 {
    match o {
        UseOutcome::NotAffine => 0,
        UseOutcome::FirstUse => 1,
        UseOutcome::DoubleUse { .. } => 2,
    }
}

fn slot_code(s: &Slot) -> u32 {
    match s {
        Slot::Skip => 0,
        Slot::Live { .. } => 1,
        Slot::Moved { .. } => 2,
    }
}

/// First use of a live `Substrate` binding — LIVE: `crate::affine::Tracker::{seeded, use_at}`
/// computes the outcome directly (mirrors `affine.rs::Tracker::use_at`'s `Live -> Moved`
/// transition).
#[test]
fn affine_first_use_is_first_use() {
    let tracker = Tracker::seeded(&[("h".to_owned(), Ty::Substrate("H".to_owned()))]);
    let expected = outcome_code(&tracker.use_at(0));
    let driver = r#"
fn main() => Binary{32} =
  match slots_use_at(slots_seeded(Cons(TySubstrate("H"), Nil)), 0b0000_0000_0000_0000_0000_0000_0000_0000, 0b0000_0000_0000_0000_0000_0000_0000_0000) {
    Pr(_, outcome) => outcome_code(outcome)
  };
"#;
    assert_l1_only_u32(
        "affine: first use of a live Substrate vs the LIVE-ORACLE outcome",
        &program(driver),
        expected,
    );
}

/// A second use of the SAME binding (feeding the updated slots back in) is a double-consume —
/// LIVE: the same `Tracker` instance's second `use_at(0)` call (mirrors `affine.rs::Tracker::
/// use_at`'s `Moved -> DoubleUse` outcome).
#[test]
fn affine_second_use_is_double_use() {
    let tracker = Tracker::seeded(&[("h".to_owned(), Ty::Substrate("H".to_owned()))]);
    let _first = tracker.use_at(0);
    let expected = outcome_code(&tracker.use_at(0));
    let driver = r#"
fn main() => Binary{32} =
  match slots_use_at(slots_seeded(Cons(TySubstrate("H"), Nil)), 0b0000_0000_0000_0000_0000_0000_0000_0000, 0b0000_0000_0000_0000_0000_0000_0000_0000) {
    Pr(slots2, _) => match slots_use_at(slots2, 0b0000_0000_0000_0000_0000_0000_0000_0000, 0b0000_0000_0000_0000_0000_0000_0000_0001) {
      Pr(_, outcome2) => outcome_code(outcome2)
    }
  };
"#;
    assert_l1_only_u32(
        "affine: second use of the same binding vs the LIVE-ORACLE DoubleUse outcome",
        &program(driver),
        expected,
    );
}

/// `union_merge_into` — a slot `Live` in `acc` but `Moved` in `other` becomes `Moved` (the
/// conservative "moved in either branch is moved afterward" rule) — LIVE: the free
/// `crate::affine::union_merge_into` computes the merged slot directly.
#[test]
fn affine_union_merge_moved_wins_over_live() {
    let mut acc = vec![Slot::Live {
        tag: "H".to_owned(),
    }];
    let other = vec![Slot::Moved {
        tag: "H".to_owned(),
        first_use: UseSite { ordinal: 0 },
    }];
    union_merge_into(&mut acc, &other);
    let expected = slot_code(&acc[0]);
    let driver = r#"
fn main() => Binary{32} =
  slot_code(slots_first(union_merge_into(
    Cons(Live("H"), Nil),
    Cons(Moved("H", 0b0000_0000_0000_0000_0000_0000_0000_0000), Nil))));
"#;
    assert_l1_only_u32(
        "affine: union_merge_into vs the LIVE-ORACLE merged slot (Moved wins over Live)",
        &program(driver),
        expected,
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// grade.rs: `grade_fn_body` — Let/If/Ascribe/App(call)/Wild, both an accepted and a demand-
// violation ("Err") outcome, mirroring RFC-0018 §4.3's documented rules — LIVE, via
// `crate::grade::check_guarantees` (see the module doc's FLAG-semcore-10-b account of the
// probing technique the passing cases need, since `check_guarantees` exposes pass/fail only).
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn fn_decl(name: &str, value_params: Vec<Param>, ret: TypeRef, body: Expr) -> FnDecl {
    FnDecl {
        vis: Vis::Private,
        thaw: false,
        tier: None,
        sig: FnSig {
            name: name.to_owned(),
            params: vec![],
            value_params,
            ret,
            effects: vec![],
            effect_budgets: BTreeMap::new(),
        },
        body,
    }
}

fn bytes_ty() -> TypeRef {
    TypeRef {
        base: BaseType::Bytes,
        guarantee: None,
    }
}

fn bytes_ty_g(g: Strength) -> TypeRef {
    TypeRef {
        base: BaseType::Bytes,
        guarantee: Some(g),
    }
}

fn str_lit(s: &str) -> Expr {
    Expr::Lit(Literal::Str(s.to_owned()))
}

fn path1(name: &str) -> Expr {
    Expr::Path(Path(vec![name.to_owned()]))
}

/// Does `fd`'s body satisfy a synthetic return-demand of `probe` (LIVE, via the real
/// `crate::grade::check_guarantees`)? Builds a clone of `fd` whose return type carries `probe` as
/// its `@ g` demand — same body, same aux fns — and asks the oracle whether the body's (private,
/// unobservable-from-here) computed grade is `⊒ probe`.
fn oracle_body_satisfies(aux_fns: &BTreeMap<String, FnDecl>, fd: &FnDecl, probe: Strength) -> bool {
    let mut probed = fd.clone();
    probed.sig.ret.guarantee = Some(probe);
    let mut fns = aux_fns.clone();
    fns.insert(probed.sig.name.clone(), probed.clone());
    let mut own = BTreeMap::new();
    own.insert(probed.sig.name.clone(), probed);
    check_guarantees(&fns, &own, &[]).is_ok()
}

/// The LIVE-ORACLE `Strength` of `fd`'s body (FLAG-semcore-10-b: recovered by probing the four
/// demand levels strongest-first and taking the first the body satisfies — sound because
/// `Strength::satisfies` is monotone in the demand and every grade satisfies the `Declared`
/// lattice floor, so the scan always terminates with the body's exact rank).
fn oracle_body_grade(aux_fns: &BTreeMap<String, FnDecl>, fd: &FnDecl) -> Strength {
    for probe in [
        Strength::Exact,
        Strength::Proven,
        Strength::Empirical,
        Strength::Declared,
    ] {
        if oracle_body_satisfies(aux_fns, fd, probe) {
            return probe;
        }
    }
    unreachable!("every grade satisfies the Declared (rank 0) floor demand")
}

fn strength_code(s: Strength) -> u32 {
    u32::from(s.rank())
}

/// `let x = "s" in x` with no ascription — LIVE-checked against `oracle_body_grade` (per G-Let,
/// with no annotation to weaken against, this is `Exact`).
#[test]
fn grade_let_no_ascription_is_exact() {
    let main_fn = fn_decl(
        "main_fn",
        vec![],
        bytes_ty(),
        Expr::Let {
            name: "x".to_owned(),
            ty: None,
            bound: Box::new(str_lit("s")),
            body: Box::new(path1("x")),
        },
    );
    let expected = strength_code(oracle_body_grade(&BTreeMap::new(), &main_fn));
    let driver = r#"
fn main_fn() => FnDecl =
  FD(Private, False, None,
     FS("main_fn", Nil, Nil, bytes_ty(), Nil, Nil),
     Let("x", None, Lit(Str("s")), Path(one_path("x"))));
fn main() => Binary{32} =
  grade_result_code(grade_fn_body(Nil, main_fn()));
"#;
    assert_l1_only_u32(
        "grade: unascribed let vs the LIVE-ORACLE grade",
        &program(driver),
        expected,
    );
}

/// `"s" @ Empirical`: body grade `Exact` satisfies the `Empirical` demand, so the ascription
/// succeeds and the result carries the WEAKENED `Empirical` grade — LIVE-checked (per G-Weaken).
#[test]
fn grade_ascribe_weakens_to_empirical() {
    let main_fn = fn_decl(
        "main_fn",
        vec![],
        bytes_ty(),
        Expr::Ascribe(Box::new(str_lit("s")), bytes_ty_g(Strength::Empirical)),
    );
    let expected = strength_code(oracle_body_grade(&BTreeMap::new(), &main_fn));
    let driver = r#"
fn main_fn() => FnDecl =
  FD(Private, False, None,
     FS("main_fn", Nil, Nil, bytes_ty(), Nil, Nil),
     Ascribe(Lit(Str("s")), bytes_ty_g(GEmpirical)));
fn main() => Binary{32} =
  grade_result_code(grade_fn_body(Nil, main_fn()));
"#;
    assert_l1_only_u32(
        "grade: Exact literal ascribed @ Empirical vs the LIVE-ORACLE weakened grade",
        &program(driver),
        expected,
    );
}

/// `wild(...) @ Exact`: `wild` is graded `Declared` (the FFI floor), which does NOT satisfy the
/// `Exact` demand — the failure is INTERNAL to the ascription check, so `check_guarantees`'s
/// `Err` is the live verdict at any outer demand (no probing needed).
#[test]
fn grade_ascribe_wild_violates_exact_demand() {
    let main_fn = fn_decl(
        "main_fn",
        vec![],
        bytes_ty(),
        Expr::Ascribe(
            Box::new(Expr::Wild(Box::new(str_lit("s")))),
            bytes_ty_g(Strength::Exact),
        ),
    );
    let mut fns = BTreeMap::new();
    fns.insert("main_fn".to_owned(), main_fn.clone());
    let mut own = BTreeMap::new();
    own.insert("main_fn".to_owned(), main_fn);
    assert!(
        check_guarantees(&fns, &own, &[]).is_err(),
        "LIVE ORACLE: wild @ Exact must be a demand violation"
    );
    let driver = r#"
fn main_fn() => FnDecl =
  FD(Private, False, None,
     FS("main_fn", Nil, Nil, bytes_ty(), Nil, Nil),
     Ascribe(Wild(Lit(Str("s"))), bytes_ty_g(GExact)));
fn main() => Binary{32} =
  grade_result_code(grade_fn_body(Nil, main_fn()));
"#;
    assert_l1_only_u32(
        "grade: wild @ Exact is a LIVE-ORACLE-confirmed demand violation (Err)",
        &program(driver),
        0xFF,
    );
}

/// `if "c" then ("t" @ Proven) else wild("f")` — the condition's grade does not degrade the
/// result; LIVE-checked (per RFC-0018 §4.5 G-Match/A restated for `if`).
#[test]
fn grade_if_meets_branch_bodies_not_condition() {
    let main_fn = fn_decl(
        "main_fn",
        vec![],
        bytes_ty(),
        Expr::If {
            cond: Box::new(str_lit("c")),
            conseq: Box::new(Expr::Ascribe(
                Box::new(str_lit("t")),
                bytes_ty_g(Strength::Proven),
            )),
            alt: Box::new(Expr::Wild(Box::new(str_lit("f")))),
        },
    );
    let expected = strength_code(oracle_body_grade(&BTreeMap::new(), &main_fn));
    let driver = r#"
fn main_fn() => FnDecl =
  FD(Private, False, None,
     FS("main_fn", Nil, Nil, bytes_ty(), Nil, Nil),
     If(Lit(Str("c")), Ascribe(Lit(Str("t")), bytes_ty_g(GProven)), Wild(Lit(Str("f")))));
fn main() => Binary{32} =
  grade_result_code(grade_fn_body(Nil, main_fn()));
"#;
    assert_l1_only_u32(
        "grade: if meets Proven/Declared branches vs the LIVE-ORACLE grade",
        &program(driver),
        expected,
    );
}

/// G-App: a callee `idfn(x: Bytes @ Empirical) => Bytes @ Empirical = x` called with an `Exact`
/// literal argument — LIVE-checked (the call's result is the callee's DECLARED return grade, not
/// the argument's own grade).
#[test]
fn grade_app_known_callee_satisfied_demand() {
    let idfn_decl = fn_decl(
        "idfn",
        vec![Param {
            name: "x".to_owned(),
            ty: bytes_ty_g(Strength::Empirical),
        }],
        bytes_ty_g(Strength::Empirical),
        path1("x"),
    );
    let main_fn = fn_decl(
        "main_fn",
        vec![],
        bytes_ty(),
        Expr::App {
            head: Box::new(path1("idfn")),
            args: vec![str_lit("s")],
        },
    );
    let mut aux_fns = BTreeMap::new();
    aux_fns.insert("idfn".to_owned(), idfn_decl);
    let expected = strength_code(oracle_body_grade(&aux_fns, &main_fn));
    let driver = r#"
fn idfn_decl() => FnDecl =
  FD(Private, False, None,
     FS("idfn", Nil, Cons(Prm("x", bytes_ty_g(GEmpirical)), Nil), bytes_ty_g(GEmpirical), Nil, Nil),
     Path(one_path("x")));
fn main_fn() => FnDecl =
  FD(Private, False, None,
     FS("main_fn", Nil, Nil, bytes_ty(), Nil, Nil),
     App(Path(one_path("idfn")), Cons(Lit(Str("s")), Nil)));
fn main() => Binary{32} =
  grade_result_code(grade_fn_body(Cons(Pr("idfn", idfn_decl()), Nil), main_fn()));
"#;
    assert_l1_only_u32(
        "grade: App to a known callee vs the LIVE-ORACLE declared return grade",
        &program(driver),
        expected,
    );
}

/// G-App failure: a callee demanding `Exact` on its parameter, called with a `wild`-graded
/// (`Declared`) argument — the failure is INTERNAL to `grade_app`'s own argument check, so
/// `check_guarantees`'s `Err` is the live verdict at any outer demand (no probing needed).
#[test]
fn grade_app_known_callee_violates_demand() {
    let strictfn_decl = fn_decl(
        "strictfn",
        vec![Param {
            name: "x".to_owned(),
            ty: bytes_ty_g(Strength::Exact),
        }],
        bytes_ty_g(Strength::Exact),
        path1("x"),
    );
    let main_fn = fn_decl(
        "main_fn",
        vec![],
        bytes_ty(),
        Expr::App {
            head: Box::new(path1("strictfn")),
            args: vec![Expr::Wild(Box::new(str_lit("s")))],
        },
    );
    let mut fns = BTreeMap::new();
    fns.insert("strictfn".to_owned(), strictfn_decl);
    fns.insert("main_fn".to_owned(), main_fn.clone());
    let mut own = BTreeMap::new();
    own.insert("main_fn".to_owned(), main_fn);
    assert!(
        check_guarantees(&fns, &own, &[]).is_err(),
        "LIVE ORACLE: a Declared-graded arg must violate an Exact demand"
    );
    let driver = r#"
fn strictfn_decl() => FnDecl =
  FD(Private, False, None,
     FS("strictfn", Nil, Cons(Prm("x", bytes_ty_g(GExact)), Nil), bytes_ty_g(GExact), Nil, Nil),
     Path(one_path("x")));
fn main_fn() => FnDecl =
  FD(Private, False, None,
     FS("main_fn", Nil, Nil, bytes_ty(), Nil, Nil),
     App(Path(one_path("strictfn")), Cons(Wild(Lit(Str("s"))), Nil)));
fn main() => Binary{32} =
  grade_result_code(grade_fn_body(Cons(Pr("strictfn", strictfn_decl()), Nil), main_fn()));
"#;
    assert_l1_only_u32(
        "grade: App to a known callee with a LIVE-ORACLE-confirmed demand violation (Err)",
        &program(driver),
        0xFF,
    );
}
