//! Differential tests for `std.iter` (M-715, E13-1) — the self-hosted first-order iterator surface
//! over the `List<A>` cons-list shape.
//!
//! The nodule source is loaded verbatim via `include_str!` (the single source of truth), then a
//! typed driver `fn` is appended to pin every generic parameter to a concrete type (`Binary{8}`).
//! Without explicit pinning the monomorphizer emits a never-silent `Residual` (undetermined type
//! parameter — G2), so every driver uses typed helpers and explicit return types.
//!
//! # Honesty tags
//! - **`Exact`** — constructors (`Nil`/`Cons`) and the total discriminator `is_empty_l` — total
//!   over the finite domain (RFC-0016).
//! - **`Declared`** — the type-level contract of `length` (O(n) spine-walk) — a structural check,
//!   not a theorem.
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), validated
//!   by trial on the programs below; not a machine-checked proof.
//!
//! # Recursive HOF combinators now execute three-way (M-715 closed — rsm S3)
//! `map`, `filter`, `foldl`, `any`, `all`, `find` — all recursive HOF combinators — now execute
//! three-way (L1-eval ≡ L0-interp ≡ AOT). The previously-flagged gap is CLOSED: stage-1
//! defunctionalization (RFC-0024 §4, M-687) handled *saturated* HOF application (`f(x)`) but not a
//! *recursive call that re-passes a HOF parameter* (`map(rest, f)`). M-715 extends
//! `mono::resolve_fn_args`: when the fn-valued argument is a HOF VALUE PARAMETER already bound to a
//! static specialization in the current emit scope (`fn_param_subst`), it is threaded through as the
//! SAME specialization the outer call pinned (so the recursive self-call resolves to e.g. `map$inc`,
//! the fn-arg dropped — no runtime closure). Still deferred (M-704, never faked): closures / lambdas,
//! multi-arg arrows (a true binary `foldl` f: A -> B -> B), and partial application.
//!
//! # What three-way covers
//! - `is_empty_l` — total discriminator (Exact), three-way green
//! - `length` — O(n) spine-walk (Declared), three-way green; never-silent add_u overflow (Empirical)
//! - `map` / `filter` / `foldl` / `any` / `all` / `find` — recursive HOF combinators over a named
//!   top-level fn arg (Declared contract; Empirical three-way agreement)

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// The std.iter nodule source, loaded at compile time — the single source of truth.
const ITER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/iter.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    format!("{ITER_SRC}\n{driver}")
}

/// Run the three-way differential on `src` — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT — and
/// assert all three paths agree AND equal the `expected` reference value.
///
/// Honesty: differential agreement is `Empirical` (trials); the type-level contract is `Declared`.
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));

    assert!(
        mono.fns.values().all(|fd| fd.sig.params.is_empty())
            && mono.types.values().all(|d| d.params.is_empty())
            && mono.traits.is_empty()
            && mono.instances.is_empty()
            && mono.impls.is_empty(),
        "{label}: monomorphized env must be closed (no generics/traits)"
    );

    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));

    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));

    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: elaborate failed: {e}"));
    let l0_core = interp
        .eval_core(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));

    let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
        .unwrap_or_else(|e| panic!("{label}: AOT run_core failed: {e}"));

    assert_eq!(
        l1_core, l0_core,
        "{label}: L1-eval(mono) vs elaborate→L0-interp diverged"
    );
    assert_eq!(l0_core, aot_core, "{label}: L0-interp vs AOT diverged");

    for (x, y, pair) in [
        (&l1_core, &l0_core, "L1↔interp"),
        (&l0_core, &aot_core, "interp↔AOT"),
    ] {
        assert_eq!(
            check_core(x, y),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "{label}: the shared checker must validate the {pair} pair"
        );
    }

    let ref_env = check_nodule(
        &parse(expected_src).unwrap_or_else(|e| panic!("{label}: ref parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("{label}: ref check failed: {e}"));
    let ref_node = elaborate(&ref_env, "main")
        .unwrap_or_else(|e| panic!("{label}: ref elaborate failed: {e}"));
    let expected = interp
        .eval_core(&ref_node)
        .unwrap_or_else(|e| panic!("{label}: ref eval failed: {e}"));

    assert_eq!(
        l1_core, expected,
        "{label}: result does not match expected reference value"
    );
}

// ── is_empty_l ────────────────────────────────────────────────────────────────────────────────────

/// `is_empty_l(Nil)` → `True` (Exact: the empty case always returns True).
/// Expected (hand-computed, three-way verified): is_empty_l on empty List returns True.
#[test]
fn is_empty_l_on_nil_returns_true() {
    let driver = "fn mk_nil() => List[Binary{8}] = Nil;\nfn main() => Bool = is_empty_l(mk_nil());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_empty_l(Nil)", &src, expected);
}

/// `is_empty_l(Cons(x, Nil))` → `False` (Exact: the Cons arm always returns False).
/// Expected (hand-computed, three-way verified): is_empty_l on non-empty List returns False.
#[test]
fn is_empty_l_on_cons_returns_false() {
    let driver = "fn mk_one() => List[Binary{8}] = Cons(0b0000_0001, Nil);\nfn main() => Bool = is_empty_l(mk_one());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_empty_l(Cons)", &src, expected);
}

// ── length ────────────────────────────────────────────────────────────────────────────────────────

/// `length([0b01, 0b02])` → `0b0000_0010`. O(n) spine-walk; Declared.
/// Expected (hand-computed, three-way verified): same provenance-matching rationale as
/// std_collections.rs::len_of_two_element_list — the reference uses add_u, not a literal,
/// to match the Derived provenance produced by `length`'s `add_u` spine.
#[test]
fn length_of_two_element_list() {
    let driver = "fn mk_two() => List[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Nil));\nfn main() => Binary{8} = length(mk_two());";
    let src = program(driver);
    // length([e1, e2]) = add_u(1, add_u(1, 0)) = 2 — Derived provenance matches.
    let expected =
        "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0001, add_u(0b0000_0001, 0b0000_0000));";
    assert_three_way("length([1,2])", &src, expected);
}

/// `length(Nil)` → `0b0000_0000`. Base case (Exact: match-defined, returns the literal).
/// Expected (hand-computed, three-way verified).
#[test]
fn length_of_nil_is_zero() {
    let driver =
        "fn mk_nil() => List[Binary{8}] = Nil;\nfn main() => Binary{8} = length(mk_nil());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0000;";
    assert_three_way("length(Nil)", &src, expected);
}

/// `length` of a three-element list → `0b0000_0011`. Declared.
/// Expected (hand-computed, three-way verified).
#[test]
fn length_of_three_element_list() {
    let driver = "fn mk_three() => List[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Binary{8} = length(mk_three());";
    let src = program(driver);
    // length([e1,e2,e3]) = add_u(1, add_u(1, add_u(1, 0))) = 3
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, 0b0000_0000)));";
    assert_three_way("length([1,2,3])", &src, expected);
}

/// `length` never-silent overflow bound: `add_u(0b0000_0001, 0b1111_1111)` refuses on ALL paths.
/// This pins the Binary{8} capacity ceiling of `length`, mirroring
/// std_collections.rs::len_bound_add_u_overflow_refuses_on_every_path. Empirical.
#[test]
fn length_bound_add_u_overflow_refuses_on_every_path() {
    let src = program("fn main() => Binary{8} = add_u(0b0000_0001, 0b1111_1111);");

    let env =
        check_nodule(&parse(&src).expect("length_bound: parse must succeed (overflow is runtime)"))
            .expect("length_bound: check must succeed (overflow is runtime contract)");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "length_bound: L1-eval must refuse the add_u overflow (never a silent wrap to 0)"
    );
    let node = elaborate(&env, "main").expect("length_bound: must elaborate");
    assert!(
        interp.eval(&node).is_err(),
        "length_bound: L0-interp must refuse the overflow"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "length_bound: AOT must refuse the overflow"
    );
}

// ── Recursive HOF combinators — executable three-way (M-715 closed) ─────────────────────────────────
//
// map/filter/foldl/any/all/find now run three-way (L1-eval ≡ L0-interp ≡ AOT) over a single named
// top-level fn argument. The recursive re-pass of the HOF parameter (`map(rest, f)` etc.) is threaded
// as the same static specialization (mono::resolve_fn_args). References share Derived/Root provenance
// with the computed value: `map`'s elements are `add_u(h, 1)` (Derived); `filter`/`find` return the
// original element (Root). Closures / multi-arg arrows stay deferred (M-704) — only NAMED fns here.

// A reusable Binary{8} successor as a top-level fn (a valid RFC-0024 §4 defunctionalization target).
const INC: &str = "fn inc(x: Binary{8}) => Binary{8} = add_u(x, 0b0000_0001);\n";
// A reusable Binary{8} predicate (== 0b10) as a top-level fn.
const IS_TWO: &str =
    "fn is_two(x: Binary{8}) => Bool = match eq(x, 0b0000_0010) { 0b1 => True, _ => False };\n";

/// `map([1,2], inc)` → `[2,3]`. The recursive `map(rest, f)` threads `inc` through. Declared/Empirical.
#[test]
fn map_applies_fn_to_each_element() {
    let src = program(&format!(
        "{INC}fn mk() => List[Binary{{8}}] = Cons(0b0000_0001, Cons(0b0000_0010, Nil));\nfn main() => List[Binary{{8}}] = map(mk(), inc);"
    ));
    // Reference: recompute via the same add_u so the mapped elements share Derived provenance.
    let expected = program(
        "fn main() => List[Binary{8}] = Cons(add_u(0b0000_0001, 0b0000_0001), Cons(add_u(0b0000_0010, 0b0000_0001), Nil));",
    );
    assert_three_way("map([1,2], inc)=[2,3]", &src, &expected);
}

/// `map(Nil, inc)` → `Nil` — empty passes through (never-silent). Exact.
#[test]
fn map_over_nil_is_nil() {
    let src = program(&format!(
        "{INC}fn mk() => List[Binary{{8}}] = Nil;\nfn main() => List[Binary{{8}}] = map(mk(), inc);"
    ));
    let expected = program("fn main() => List[Binary{8}] = Nil;");
    assert_three_way("map(Nil, inc)=Nil", &src, &expected);
}

/// `filter([1,2,1], is_one)` → `[1,1]` — keeps the matching ORIGINAL elements (Root provenance).
#[test]
fn filter_keeps_matching_elements() {
    let src = program(
        "fn is_one(x: Binary{8}) => Bool = match eq(x, 0b0000_0001) { 0b1 => True, _ => False };\nfn mk() => List[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0001, Nil)));\nfn main() => List[Binary{8}] = filter(mk(), is_one);",
    );
    let expected =
        program("fn main() => List[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0001, Nil));");
    assert_three_way("filter([1,2,1], is_one)=[1,1]", &src, &expected);
}

/// `foldl([1,2,3], inc, 0)` → `inc(3)` = `4`. Per the nodule contract, the `f: A -> B` foldl discards
/// the accumulator and returns `f(last)` for a non-empty list (Derived). Empirical.
#[test]
fn foldl_returns_f_of_last_for_nonempty() {
    let src = program(&format!(
        "{INC}fn mk() => List[Binary{{8}}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Binary{{8}} = foldl(mk(), inc, 0b0000_0000);"
    ));
    let expected = program("fn main() => Binary{8} = add_u(0b0000_0011, 0b0000_0001);");
    assert_three_way("foldl([1,2,3], inc, 0)=inc(3)=4", &src, &expected);
}

/// `foldl(Nil, inc, 5)` → `5` — the initial acc is returned on the empty list (never-silent). Root.
#[test]
fn foldl_over_nil_returns_initial_acc() {
    let src = program(&format!(
        "{INC}fn mk() => List[Binary{{8}}] = Nil;\nfn main() => Binary{{8}} = foldl(mk(), inc, 0b0000_0101);"
    ));
    let expected = program("fn main() => Binary{8} = 0b0000_0101;");
    assert_three_way("foldl(Nil, inc, 5)=5", &src, &expected);
}

/// `any([1,2], is_two)` → `True` (the second element matches). Declared/Empirical.
#[test]
fn any_true_when_an_element_matches() {
    let src = program(&format!(
        "{IS_TWO}fn mk() => List[Binary{{8}}] = Cons(0b0000_0001, Cons(0b0000_0010, Nil));\nfn main() => Bool = any(mk(), is_two);"
    ));
    assert_three_way(
        "any([1,2], is_two)=True",
        &src,
        "nodule ref;\nfn main() => Bool = True;",
    );
}

/// `any([1,3], is_two)` → `False`; and `any(Nil, is_two)` → `False` (never a fabricated True).
#[test]
fn any_false_when_no_element_matches() {
    let src = program(&format!(
        "{IS_TWO}fn mk() => List[Binary{{8}}] = Cons(0b0000_0001, Cons(0b0000_0011, Nil));\nfn main() => Bool = any(mk(), is_two);"
    ));
    assert_three_way(
        "any([1,3], is_two)=False",
        &src,
        "nodule ref;\nfn main() => Bool = False;",
    );
}

/// `all([2,2], is_two)` → `True`; `all([2,1], is_two)` → `False` (short-circuits at the first miss).
#[test]
fn all_true_only_when_every_element_matches() {
    let yes = program(&format!(
        "{IS_TWO}fn mk() => List[Binary{{8}}] = Cons(0b0000_0010, Cons(0b0000_0010, Nil));\nfn main() => Bool = all(mk(), is_two);"
    ));
    assert_three_way(
        "all([2,2], is_two)=True",
        &yes,
        "nodule ref;\nfn main() => Bool = True;",
    );
    let no = program(&format!(
        "{IS_TWO}fn mk() => List[Binary{{8}}] = Cons(0b0000_0010, Cons(0b0000_0001, Nil));\nfn main() => Bool = all(mk(), is_two);"
    ));
    assert_three_way(
        "all([2,1], is_two)=False",
        &no,
        "nodule ref;\nfn main() => Bool = False;",
    );
}

/// `find([1,2,3], is_two)` → `Some(2)` — the first matching ORIGINAL element (Root). Never-silent.
#[test]
fn find_returns_first_match() {
    let src = program(&format!(
        "{IS_TWO}fn mk() => List[Binary{{8}}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Option[Binary{{8}}] = find(mk(), is_two);"
    ));
    let expected = program("fn main() => Option[Binary{8}] = Some(0b0000_0010);");
    assert_three_way("find([1,2,3], is_two)=Some(2)", &src, &expected);
}

/// `find([1,3], is_two)` → `None` — a miss is an explicit None, never a fabricated element (G2).
#[test]
fn find_miss_returns_none() {
    let src = program(&format!(
        "{IS_TWO}fn mk() => List[Binary{{8}}] = Cons(0b0000_0001, Cons(0b0000_0011, Nil));\nfn main() => Option[Binary{{8}}] = find(mk(), is_two);"
    ));
    let expected = program("fn main() => Option[Binary{8}] = None;");
    assert_three_way("find([1,3], is_two)=None", &src, &expected);
}

// Note (M-704 boundary, documented not tested): the M-715 fix threads a NAMED top-level fn and its
// recursive re-pass only. Closures / lambdas, multi-arg arrows, and partial application remain
// deferred (RFC-0024 §5 / M-704) — they are not constructable in the stage-1 surface, so there is no
// program to exercise; the deferral is recorded in iter.myc's per-fn FLAGs, never silently claimed.
