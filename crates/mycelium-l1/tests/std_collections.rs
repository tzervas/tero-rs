//! Differential tests for `std.collections` (M-716, #461) — the self-hosted `Vec`/`Map`/`Set`
//! core nodule.
//!
//! The nodule source is loaded verbatim via `include_str!` (the single source of truth), then a
//! typed driver `fn` is appended to pin every generic parameter to a concrete type (e.g.
//! `Binary{8}`). Without explicit pinning the monomorphizer emits a never-silent `Residual`
//! (undetermined type parameter — G2), so every driver carries typed helper functions.
//!
//! # Honesty tags
//! - **`Exact`** — constructors (`Nil`/`Cons`/`MNil`/`MCons`/`SNil`/`SCons`) and total
//!   discriminators (`is_empty`) — total, RFC-0016 §4.1 C2 / docs/spec/stdlib/collections.md §3.
//! - **`Declared`** — the type-level contract of every eliminator/transformer (`head`/`tail`/`get`/
//!   `snoc`/`reverse`/`map_get`/`set_contains`) — a structural check, not a theorem.
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), validated
//!   by trial on the programs below; not a machine-checked proof.
//! - **`Empirical`** — the `len`-fits-`Binary{8}` bound (add_u refuses at 256 on every path —
//!   the overflow test pins this; not a type-level proof).
//!
//! # Grounding
//! Expected values are hand-computed and verified three-way (L1≡L0≡AOT). The Rust crate
//! `crates/mycelium-std-collections` exists but is **Seq-backed** (a different representation): it
//! shares the `is_empty`/`get`/`len`/`contains` semantics, but has no `head`/`tail`/`snoc`/`reverse`
//! (those are cons-list ops of this `.myc` port). So it is a value oracle for the shared-semantics
//! subset only — not a structural reference.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// The std.collections nodule source, loaded at compile time — the single source of truth.
const COLLECTIONS_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/collections.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    format!("{COLLECTIONS_SRC}\n{driver}")
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

// ── Vec: is_empty ──────────────────────────────────────────────────────────────────────────────────

/// `is_empty(Nil)` → `True` (Exact: the empty case always returns True).
/// Expected (hand-computed, three-way verified): Vec::is_empty on an empty list returns true.
#[test]
fn is_empty_on_nil_returns_true() {
    let driver = "fn mk_nil() => Vec[Binary{8}] = Nil;\nfn main() => Bool = is_empty(mk_nil());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_empty(Nil)", &src, expected);
}

/// `is_empty(Cons(x, Nil))` → `False` (Exact: the Cons arm always returns False).
/// Expected (hand-computed, three-way verified): Vec::is_empty on a non-empty list returns false.
#[test]
fn is_empty_on_cons_returns_false() {
    let driver = "fn mk_one() => Vec[Binary{8}] = Cons(0b0000_0001, Nil);\nfn main() => Bool = is_empty(mk_one());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_empty(Cons)", &src, expected);
}

// ── Vec: head ──────────────────────────────────────────────────────────────────────────────────────

/// `head(Nil)` → `None` — never-silent (G2): empty Vec never fabricates a value. Declared.
/// Expected (hand-computed, three-way verified): Vec::head on empty returns None.
#[test]
fn head_on_nil_returns_none() {
    let driver =
        "fn mk_nil() => Vec[Binary{8}] = Nil;\nfn main() => Option[Binary{8}] = head(mk_nil());";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = None;";
    assert_three_way("head(Nil)", &src, expected);
}

/// `head(Cons(x, rest))` → `Some(x)` — first element is returned. Declared.
/// Expected (hand-computed, three-way verified): Vec::head on Cons(0b0000_0001, Nil) returns Some(0b0000_0001).
#[test]
fn head_on_cons_returns_some() {
    let driver = "fn mk_one() => Vec[Binary{8}] = Cons(0b0000_0001, Nil);\nfn main() => Option[Binary{8}] = head(mk_one());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b0000_0001);";
    assert_three_way("head(Cons)", &src, expected);
}

// ── Vec: tail ──────────────────────────────────────────────────────────────────────────────────────

/// `tail(Nil)` → `None` — never-silent (G2). Declared.
/// Expected (hand-computed, three-way verified): Vec::tail on empty returns None.
#[test]
fn tail_on_nil_returns_none() {
    let driver = "fn mk_nil() => Vec[Binary{8}] = Nil;\nfn main() => Option[Vec[Binary{8}]] = tail(mk_nil());";
    let src = program(driver);
    // The inner Vec is the empty Nil — Option<Vec<Binary{8}>> = None.
    let expected = "nodule ref;\ntype Vec[A] = Nil | Cons(A, Vec[A]);\ntype Option[A] = Some(A) | None;\nfn main() => Option[Vec[Binary{8}]] = None;";
    assert_three_way("tail(Nil)", &src, expected);
}

/// `tail(Cons(x, rest))` → `Some(rest)` — returns the spine after the head. Declared.
/// Expected (hand-computed, three-way verified): tail on [1, 2] returns Some([2]).
#[test]
fn tail_on_cons_returns_some() {
    let driver = "fn mk_two() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Nil));\nfn main() => Option[Vec[Binary{8}]] = tail(mk_two());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Vec[A] = Nil | Cons(A, Vec[A]);\ntype Option[A] = Some(A) | None;\nfn main() => Option[Vec[Binary{8}]] = Some(Cons(0b0000_0010, Nil));";
    assert_three_way("tail(Cons)", &src, expected);
}

// ── Vec: len ───────────────────────────────────────────────────────────────────────────────────────

/// `len` over a two-element list → `0b0000_0010`. O(n) spine-walk; Declared.
/// Expected (hand-computed, three-way verified): Vec::len on [1, 2] returns 2.
/// The reference program uses `add_u` (not a literal) to match the `Derived` provenance produced
/// by `len`'s `add_u` spine — a literal `0b0000_0010` has `Root` provenance and fails `assert_eq`.
/// `len([1,2]) = add_u(1, add_u(1, 0))`: same ops and the same `Derived` provenance, which
/// `CoreValue` equality requires (a `Root`-provenance literal `0b0000_0010` would fail `assert_eq`).
/// (Empirical basis; the three-way agreement is separately asserted above.)
#[test]
fn len_of_two_element_list() {
    let driver = "fn mk_two() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Nil));\nfn main() => Binary{8} = len(mk_two());";
    let src = program(driver);
    // add_u(1, add_u(1, 0)) = 2 via the same op tree as len([e1, e2])
    let expected =
        "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0001, add_u(0b0000_0001, 0b0000_0000));";
    assert_three_way("len([1,2])", &src, expected);
}

/// `len` over a three-element list → `0b0000_0011`. Declared.
/// Expected (hand-computed, three-way verified): Vec::len on [1, 2, 3] returns 3.
/// Same provenance-matching rationale: add_u(1, add_u(1, add_u(1, 0))) = 3.
#[test]
fn len_of_three_element_list() {
    let driver = "fn mk_three() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Binary{8} = len(mk_three());";
    let src = program(driver);
    // add_u(1, add_u(1, add_u(1, 0))) = 3
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, 0b0000_0000)));";
    assert_three_way("len([1,2,3])", &src, expected);
}

/// `len`-bound: the `add_u` mechanism underlying `len`'s `Binary{8}` count refuses at 256 on ALL
/// three paths — never a silent wrap (G2/VR-5). Empirical (pinned by trial on the programs below).
///
/// Why not test via a 256-element list: `len`'s recursion reaches the L1 evaluator's depth limit
/// (`DEFAULT_DEPTH = 64`) long before reaching 256 elements; and the L0 interpreter does not use
/// the same deep-worker-stack machinery, so 256-deep `fill` recursion overflows the Rust thread
/// stack instead of being a clean `is_err()`. Both are never-silent refusals — but neither is the
/// `add_u` arithmetic overflow. We test the actual mechanism (add_u overflow at `Binary{8}`
/// boundary) directly, exactly as `enablement.rs::add_u_overflow_refuses_on_every_path` does.
/// The `len` connection: `len(xs)` is `add_u(1, len(rest))` — the 256th step would compute
/// `add_u(0b0000_0001, 0b1111_1111) = 256` which this test pins. Empirical.
///
/// Expected (hand-computed, three-way verified): Vec::len fails (add_u overflows) on a > 255-element list.
#[test]
fn len_bound_add_u_overflow_refuses_on_every_path() {
    // add_u(0b0000_0001, 0b1111_1111) = 256, which overflows Binary{8} — the exact operation
    // that len would execute on its 256th element. This is the never-silent (G2) contract for
    // len's Binary{8} index width. Uses the collections nodule source as context for consistency.
    let src = program("fn main() => Binary{8} = add_u(0b0000_0001, 0b1111_1111);");

    let env = check_nodule(
        &parse(&src).expect("len_bound: parse must succeed (overflow is runtime, not static)"),
    )
    .expect("len_bound: check must succeed (overflow is a runtime contract)");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "len_bound: L1-eval must refuse the add_u overflow (never a silent wrap to 0)"
    );
    let node = elaborate(&env, "main").expect("len_bound: must elaborate");
    assert!(
        interp.eval(&node).is_err(),
        "len_bound: L0-interp must refuse the overflow"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "len_bound: AOT must refuse the overflow"
    );
}

// ── Vec: get ───────────────────────────────────────────────────────────────────────────────────────

/// `get([1,2,3], 0)` → `Some(1)` — index 0 returns the head. Declared.
/// Expected (hand-computed, three-way verified): Vec::get on [1,2,3] at 0 returns Some(1).
#[test]
fn get_index_0_returns_head() {
    let driver = "fn mk_three() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Option[Binary{8}] = get(mk_three(), 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b0000_0001);";
    assert_three_way("get([1,2,3], 0)", &src, expected);
}

/// `get([1,2,3], 1)` → `Some(2)` — index 1 returns the second element. Declared.
/// Expected (hand-computed, three-way verified): Vec::get on [1,2,3] at 1 returns Some(2).
#[test]
fn get_index_1_returns_second() {
    let driver = "fn mk_three() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Option[Binary{8}] = get(mk_three(), 0b0000_0001);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b0000_0010);";
    assert_three_way("get([1,2,3], 1)", &src, expected);
}

/// `get([1,2,3], 5)` → `None` — OOB → None, never-silent (G2). Declared.
/// Expected (hand-computed, three-way verified): Vec::get on [1,2,3] at 5 returns None.
#[test]
fn get_out_of_bounds_returns_none() {
    let driver = "fn mk_three() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Option[Binary{8}] = get(mk_three(), 0b0000_0101);";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = None;";
    assert_three_way("get([1,2,3], OOB)", &src, expected);
}

// ── Vec: snoc ──────────────────────────────────────────────────────────────────────────────────────

/// `snoc([1,2], 3)` → `[1,2,3]` — appends at the end. Declared.
/// Expected (hand-computed, three-way verified): Vec::snoc on [1,2] with 3 returns [1,2,3] (Cons(1,Cons(2,Cons(3,Nil)))).
#[test]
fn snoc_appends_at_end() {
    let driver = "fn mk_two() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Nil));\nfn main() => Vec[Binary{8}] = snoc(mk_two(), 0b0000_0011);";
    let src = program(driver);
    // snoc([1,2], 3) = [1,2,3] = Cons(1, Cons(2, Cons(3, Nil)))
    let expected = "nodule ref;\ntype Vec[A] = Nil | Cons(A, Vec[A]);\nfn main() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));";
    assert_three_way("snoc([1,2], 3)", &src, expected);
}

// ── Vec: reverse ───────────────────────────────────────────────────────────────────────────────────

/// `reverse([1,2,3])` → `[3,2,1]` — snoc-based recursion reverses the spine (O(n²); the O(n) accumulator form is blocked under RFC-0007 §11.3 — see the nodule comment). Declared.
/// Expected (hand-computed, three-way verified): Vec::reverse on [1,2,3] returns [3,2,1].
#[test]
fn reverse_of_three_element_list() {
    let driver = "fn mk_three() => Vec[Binary{8}] = Cons(0b0000_0001, Cons(0b0000_0010, Cons(0b0000_0011, Nil)));\nfn main() => Vec[Binary{8}] = reverse(mk_three());";
    let src = program(driver);
    // reverse([1,2,3]) = [3,2,1] = Cons(3, Cons(2, Cons(1, Nil)))
    let expected = "nodule ref;\ntype Vec[A] = Nil | Cons(A, Vec[A]);\nfn main() => Vec[Binary{8}] = Cons(0b0000_0011, Cons(0b0000_0010, Cons(0b0000_0001, Nil)));";
    assert_three_way("reverse([1,2,3])", &src, expected);
}

// ── Map: map_get ───────────────────────────────────────────────────────────────────────────────────
//
// map_get is WIDTH-GENERIC over the key width N and fully generic over the value type V (M-718/M-753).
// These drivers pin N=8, V=Binary{8} from the concrete map; the Binary{16}-key and non-Binary{8}-value
// specialisations are covered below. Lookup is O(n) linear scan; first match wins; missing key → None
// (G2). The recursive scan is itself width-generic — enabled by the width-var pass-through in `unify`.

/// `map_get(MCons(k,v, MNil), k)` → `Some(v)` — key present, hit on first entry. Declared.
/// Expected (hand-computed, three-way verified): Map::get on {1→10} with key 1 returns Some(10).
#[test]
fn map_get_hit_returns_some() {
    let driver = "fn mk_map() => Map[Binary{8}, Binary{8}] = MCons(0b0000_0001, 0b0000_1010, MNil);\nfn main() => Option[Binary{8}] = map_get(mk_map(), 0b0000_0001);";
    let src = program(driver);
    // map_get({1→10}, 1) = Some(10) = Some(0b0000_1010)
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b0000_1010);";
    assert_three_way("map_get(hit)", &src, expected);
}

/// `map_get(MCons(k,v, MNil), k2)` where `k2 ≠ k` → `None`. Never-silent (G2). Declared.
/// Expected (hand-computed, three-way verified): Map::get on {1→10} with key 2 returns None.
#[test]
fn map_get_miss_returns_none() {
    let driver = "fn mk_map() => Map[Binary{8}, Binary{8}] = MCons(0b0000_0001, 0b0000_1010, MNil);\nfn main() => Option[Binary{8}] = map_get(mk_map(), 0b0000_0010);";
    let src = program(driver);
    // map_get({1→10}, 2) = None
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = None;";
    assert_three_way("map_get(miss)", &src, expected);
}

/// `map_get` with two entries, shadowed key: insert order wins (first MCons wins). Declared.
/// Expected (hand-computed, three-way verified): Map::get on {2→20, 1→10} with key 2 returns Some(20).
#[test]
fn map_get_multi_entry_first_wins() {
    // map_insert(2, 20, map_insert(1, 10, map_empty)) = MCons(2, 20, MCons(1, 10, MNil))
    // map_get that, key=2 → Some(20)
    let driver = "fn mk_map() => Map[Binary{8}, Binary{8}] = MCons(0b0000_0010, 0b0001_0100, MCons(0b0000_0001, 0b0000_1010, MNil));\nfn main() => Option[Binary{8}] = map_get(mk_map(), 0b0000_0010);";
    let src = program(driver);
    // map_get({2→20, 1→10}, 2) = Some(20) = Some(0b0001_0100)
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b0001_0100);";
    assert_three_way("map_get(multi, first-wins)", &src, expected);
}

// ── Set: set_contains ──────────────────────────────────────────────────────────────────────────────
//
// set_contains is WIDTH-GENERIC over the element width N (M-718/M-753; same width-typed eq constraint
// as map_get). These drivers pin N=8; the Binary{16} specialisation is covered below. O(n) scan. Declared.

/// `set_contains(SCons(x, SNil), x)` → `True` — element present. Declared.
/// Expected (hand-computed, three-way verified): Set::contains on {1} with 1 returns True.
#[test]
fn set_contains_present_returns_true() {
    let driver = "fn mk_set() => Set[Binary{8}] = SCons(0b0000_0001, SNil);\nfn main() => Bool = set_contains(mk_set(), 0b0000_0001);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("set_contains(present)", &src, expected);
}

/// `set_contains(SCons(x, SNil), y)` where `y ≠ x` → `False`. Never-silent (G2). Declared.
/// Expected (hand-computed, three-way verified): Set::contains on {1} with 2 returns False.
#[test]
fn set_contains_absent_returns_false() {
    let driver = "fn mk_set() => Set[Binary{8}] = SCons(0b0000_0001, SNil);\nfn main() => Bool = set_contains(mk_set(), 0b0000_0010);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("set_contains(absent)", &src, expected);
}

/// `set_contains` on empty set → `False`. Never-silent (G2). Declared.
/// Expected (hand-computed, three-way verified): Set::contains on {} with any key returns False.
#[test]
fn set_contains_empty_returns_false() {
    let driver = "fn mk_empty() => Set[Binary{8}] = SNil;\nfn main() => Bool = set_contains(mk_empty(), 0b0000_0001);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("set_contains(empty)", &src, expected);
}

// ── Width-generic key/value specialisations (M-718) ─────────────────────────────────────────────────
//
// The SAME recursive map_get/set_contains definitions, specialised to a DIFFERENT key width and to a
// non-Binary{8} value type — proving they are genuinely width-generic over the key (not a renamed
// Binary{8} monomorph) and fully generic over the value. The recursive linear scan over an abstract
// width is what the `unify` width-var pass-through (M-718) enables.

/// `map_get` with `Binary{16}` keys — a two-entry map scanned past the first entry (so the RECURSIVE
/// call executes at the abstract width before monomorphizing to N=16). Key 256 → Some(512). Declared.
/// Hand-computed: {1→10, 256→512} get 256 = Some(512); 256/512 are unrepresentable at Binary{8}.
#[test]
fn map_get_binary16_key_recurses() {
    let driver = "fn mk_map() => Map[Binary{16}, Binary{16}] = MCons(0b0000_0000_0000_0001, 0b0000_0000_0000_1010, MCons(0b0000_0001_0000_0000, 0b0000_0010_0000_0000, MNil));\nfn main() => Option[Binary{16}] = map_get(mk_map(), 0b0000_0001_0000_0000);";
    let src = program(driver);
    // map_get({1→10, 256→512}, 256) = Some(512) = Some(0b0000_0010_0000_0000)
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{16}] = Some(0b0000_0010_0000_0000);";
    assert_three_way("map_get(Binary{16} key, recurse)", &src, expected);
}

/// `map_get` with a NON-Binary{8} value type (`V = Bool`) at `Binary{8}` keys — proves the value type
/// is fully generic (only carried, never compared). Key 2 present → Some(False). Declared.
#[test]
fn map_get_bool_value_is_generic() {
    let driver = "fn mk_map() => Map[Binary{8}, Bool] = MCons(0b0000_0001, True, MCons(0b0000_0010, False, MNil));\nfn main() => Option[Bool] = map_get(mk_map(), 0b0000_0010);";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Bool] = Some(False);";
    assert_three_way("map_get(Bool value)", &src, expected);
}

/// `set_contains` with `Binary{16}` elements — scanned past the first element so the RECURSIVE call
/// runs at the abstract width before monomorphizing to N=16. Element 256 present → True. Declared.
#[test]
fn set_contains_binary16_recurses() {
    let driver = "fn mk_set() => Set[Binary{16}] = SCons(0b0000_0000_0000_0001, SCons(0b0000_0001_0000_0000, SNil));\nfn main() => Bool = set_contains(mk_set(), 0b0000_0001_0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("set_contains(Binary{16}, recurse)", &src, expected);
}

/// `map_get` mixing a `Binary{8}` key map with a `Binary{16}` lookup key is a never-silent width
/// mismatch — the key width `N` cannot be both 8 and 16 (DN-42 §4 / VR-5 / S1). Never a silent widen.
#[test]
fn map_get_mixed_key_widths_refuses() {
    let driver = "fn mk_map() => Map[Binary{8}, Binary{8}] = MCons(0b0000_0001, 0b0000_1010, MNil);\nfn main() => Option[Binary{8}] = map_get(mk_map(), 0b0000_0001_0000_0000);";
    let src = program(driver);
    let parsed = parse(&src).expect("parse should succeed");
    let err = check_nodule(&parsed)
        .expect_err("expected a never-silent key-width-mismatch refusal, but check succeeded")
        .to_string();
    assert!(
        err.contains("Binary{16}")
            && (err.contains("cannot match") || err.contains("width") || err.contains("swap")),
        "refusal must name the key-width mismatch (never-silent), got: {err}"
    );
}

// ── Constructors / builders: empty / push_front / map_empty / map_insert / set_empty / set_add ───────
//
// M-719 honesty boundary (VR-5/G2) — NOT a fabricated pass. The six constructor/builder ops
// (`empty`/`push_front`/`map_empty`/`map_insert`/`set_empty`/`set_add`) are EXPORTED but cannot yet be
// exercised through a runnable three-way differential under the current language surface, because of the
// RFC-0007 §11.3 nullary-generic-constructor / type-argument-inference limitation:
//
//   - `empty()` / `map_empty()` / `set_empty()` are generic NULLARY functions. A call determines neither
//     `A`/`K`/`V` from its (absent) arguments, and the checker does NOT propagate the *enclosing
//     function's* return-type ascription into the call (a return-type ascription is not a call-result
//     ascription, and there is no surface syntax to ascribe a nullary call's result). So `is_empty(empty())`,
//     `let e: Vec[Binary{8}] = empty() in …`, and a typed wrapper `fn e8() => Vec[Binary{8}] = empty()`
//     all REFUSE at check time with a never-silent message: "`empty` is generic over `A`, but this call
//     does not determine it — ascribe an argument or the result (RFC-0007 §11.3, never a guessed default)".
//   - `push_front(x, Nil)` / `map_insert(k, v, MNil)` / `set_add(x, SNil)` reach the checker but FAIL at
//     monomorphization: a bare nullary constructor (`Nil`/`MNil`/`SNil`) in argument position cannot have
//     its type argument re-inferred ("constructor `Nil` of generic `Vec<…>` needs its type argument(s)
//     from context — ascribe the value (RFC-0007 §11.3, never a guess)").
//
// This is the SAME limitation the nodule itself documents for `snoc` (why it writes `push_front(x, xs)`
// rather than `Cons(x, Nil)`) and for the O(n) `reverse` form. Rather than fake a `Declared` "pass" for an
// op that does not execute (which would violate VR-5), the never-silent refusal is PINNED below as the
// honest, asserted conformance fact for these ops, and the runnable-differential gap is FLAGGED for M-719
// close-out (the runnable coverage lands when RFC-0007 §11.3 ascription syntax / nullary-call type-argument
// inference is available — tracked as a never-silent open item, not a silent omission).

/// The exported generic constructors `empty`/`map_empty`/`set_empty` are NOT silently defaulted — a call
/// that fails to determine the element/key/value type is a never-silent check refusal (G2 / RFC-0007 §11.3),
/// never a guessed `A`. This pins that behaviour for all three so the "no silent default" contract is a
/// trial-checked fact, not just prose. (Data-driven over the three constructor calls — a body that asserts
/// over a case table, per the test-layout rule.)
#[test]
fn nullary_constructors_refuse_undetermined_type_never_silent() {
    // (label, driver) — each calls a nullary generic constructor whose type parameter is undetermined.
    const CASES: &[(&str, &str)] = &[
        ("empty", "fn main() => Bool = is_empty(empty());"),
        (
            "map_empty",
            "fn main() => Option[Binary{8}] = map_get(map_empty(), 0b0000_0001);",
        ),
        (
            "set_empty",
            "fn main() => Bool = set_contains(set_empty(), 0b0000_0001);",
        ),
    ];
    for (label, driver) in CASES {
        let src = program(driver);
        let parsed = parse(&src).unwrap_or_else(|e| panic!("{label}: parse should succeed: {e}"));
        let err = check_nodule(&parsed)
            .err()
            .unwrap_or_else(|| {
                panic!("{label}: expected a never-silent undetermined-type refusal, but check succeeded")
            })
            .to_string();
        // The refusal must name the never-silent rule (RFC-0007 §11.3) — not a guessed default.
        assert!(
            err.contains("does not determine it") || err.contains("§11.3"),
            "{label}: refusal must be the never-silent undetermined-type message, got: {err}"
        );
    }
}

/// The builder ops `push_front`/`map_insert`/`set_add` over a bare nullary base constructor are NOT
/// silently defaulted either: the type argument of `Nil`/`MNil`/`SNil` in argument position cannot be
/// re-inferred at monomorphization, and that is a never-silent refusal (G2 / RFC-0007 §11.3), never a
/// guessed structure. Pinned here so the builders' "no silent default base" contract is trial-checked.
#[test]
fn builders_over_bare_base_constructor_refuse_never_silent() {
    // (label, driver) — each builder wraps a bare nullary base constructor whose type argument is
    // undetermined in argument position; check passes but monomorphize refuses (never-silent).
    const CASES: &[(&str, &str)] = &[
        (
            "push_front",
            "fn mk() => Vec[Binary{8}] = push_front(0b0000_0111, Nil);\nfn main() => Option[Binary{8}] = head(mk());",
        ),
        (
            "map_insert",
            "fn mk() => Map[Binary{8},Binary{8}] = map_insert(0b0000_0011, 0b0010_0000, MNil);\nfn main() => Option[Binary{8}] = map_get(mk(), 0b0000_0011);",
        ),
        (
            "set_add",
            "fn mk() => Set[Binary{8}] = set_add(0b0000_0101, SNil);\nfn main() => Bool = set_contains(mk(), 0b0000_0101);",
        ),
    ];
    for (label, driver) in CASES {
        let src = program(driver);
        let parsed = parse(&src).unwrap_or_else(|e| panic!("{label}: parse should succeed: {e}"));
        let env = check_nodule(&parsed)
            .unwrap_or_else(|e| panic!("{label}: check should succeed (refusal is at mono): {e}"));
        let err = monomorphize(&env, "main")
            .err()
            .unwrap_or_else(|| {
                panic!("{label}: expected a never-silent mono refusal, but monomorphize succeeded")
            })
            .to_string();
        assert!(
            err.contains("type argument") || err.contains("§11.3") || err.contains("from context"),
            "{label}: refusal must be the never-silent type-argument message, got: {err}"
        );
    }
}
