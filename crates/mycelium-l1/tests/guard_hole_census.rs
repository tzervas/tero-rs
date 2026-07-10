//! RFC-0041 §4.7/§5 — the guard-hole **census** (W0 safety net; RR-29 guard-hole inventory turned
//! into tracked failing tests, one per hole this crate owns).
//!
//! Each test below is a REAL repro: it constructs a genuinely deep/wide input and calls the named
//! hole's entry point. Today every one of them either SIGABRTs (an uncatchable host-stack overflow —
//! Rust's default stack-overflow handler aborts the process directly, it does **not** go through the
//! panic/unwind machinery, so `std::panic::catch_unwind` cannot intercept it) or — for
//! `parse_type_ref` — already refuses cleanly (a documented near-miss, not an open hole). That is
//! exactly why every test here stays `#[ignore = "Wn"]`: running one for real would crash the whole
//! test binary, not just fail an assertion (RFC-0041 §5 "durability gates" / §7 census-tags). When the
//! wave named in each `#[ignore]` reason lands, the ignore is removed and the assertion must pass.
//!
//! Never-silent (G2); honestly tagged `Declared` (RFC-0041 §Posture) — these are trackers, not proofs.

use mycelium_l1::{check_nodule, parse, L1Value};

/// `Vec[A] = Nil | Cons(A, Vec[A])` — the canonical cons-list ADT (mirrors `tests/list_literal.rs`).
const VEC_PRELUDE: &str = "nodule d;\ntype Vec[A] = Nil | Cons(A, Vec[A]);\n";

/// `[e, e, …]` (`n` elements) checked against a `Vec[Binary{8}]` context.
///
/// RFC-0040/M-977 (`checkty::check_list`, `crates/mycelium-l1/src/checkty.rs:5568`) desugars this
/// **after parsing** to a right-nested `Cons(e, Cons(…, Nil))` chain and re-checks it as that chain.
/// The list-literal SOURCE is a flat, comma-separated production (`comma_separated`, a loop — not
/// recursion), so it never charges the parser's `MAX_EXPR_DEPTH` (256) guard no matter how many
/// elements `n` has; but the *desugared* chain is `n` deep, and `self.check` walks it via genuine
/// Rust call recursion. Hence "surface-reachable, not bounded by the 256 nesting cap" (RFC-0041 §4.7).
fn deep_list_source(n: usize) -> String {
    let elems = std::iter::repeat_n("0b0000_0001", n)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{VEC_PRELUDE}fn f() => Vec[Binary{{8}}] = [{elems}];\n")
}

#[test]
fn check_list_deep_list_literal_refuses_cleanly() {
    // Hole: `checkty::check_list` (crates/mycelium-l1/src/checkty.rs:5568). RFC-0041 §4.7 (W1):
    // `check_list` now routes the list's DATA-spine through iteration + a work-step charge (not a
    // control-depth `try_enter` per element), so it no longer refuses a large list *as if* it were
    // n-deep control recursion — a large list is checked without abort, bounded by the work-step
    // budget. The desugared `Cons` chain is still n-deep DATA, so the *grade* pass (its own §4.7 depth
    // budget) is what refuses this 50_000-deep spine — cleanly (`CheckError`), never a SIGABRT. So the
    // end-to-end contract is a clean explicit refusal, not a crash (and no longer a control-depth
    // false-refusal from `check_list` itself).
    let src = deep_list_source(50_000);
    let nodule = parse(&src).expect("the SOURCE is not deeply nested (flat list) — parses fine");
    let result = check_nodule(&nodule);
    assert!(
        result.is_err(),
        "expected an explicit over-budget refusal (RFC-0041 §4.7), not success or a SIGABRT"
    );
}

/// A single constructor of arity `n` (`Mk(Binary{8}, Binary{8}, …)`) plus a two-arm match over it
/// (`Mk(_, _, …) => …, _ => …`) — the "structural twin" of the list-literal repro (RFC-0041 §4.7):
/// constructor ARITY is also a flat, comma-separated grammar production (`comma_separated`), so each
/// of the `n` sibling arguments/patterns charges and releases the SAME parser depth level rather than
/// accumulating — `n` can defeat `MAX_EXPR_DEPTH` (256) by width alone, without ever nesting. Pattern-
/// match COMPILATION (`usefulness::useful` / `decision::compile_rows`) then walks the wide constructor
/// itself, unguarded by that cap ("the tuple/ctor-arity→depth spine, surface-reachable" — RFC-0041
/// §4.7).
fn wide_arity_match_source(n: usize) -> String {
    let field_tys = std::iter::repeat_n("Binary{8}", n)
        .collect::<Vec<_>>()
        .join(", ");
    let args = std::iter::repeat_n("0b0000_0001", n)
        .collect::<Vec<_>>()
        .join(", ");
    let wilds = std::iter::repeat_n("_", n).collect::<Vec<_>>().join(", ");
    format!(
        "nodule d;\ntype Big = Mk({field_tys});\n\
         fn f() => Binary{{8}} = match Mk({args}) {{ Mk({wilds}) => 0b0000_0001, _ => 0b0000_0000 }};\n"
    )
}

#[test]
fn usefulness_wide_arity_match_refuses_cleanly() {
    // Hole: `usefulness::useful` (crates/mycelium-l1/src/usefulness.rs:147) — exhaustiveness
    // checking over the wide-arity constructor pattern above.
    let src = wide_arity_match_source(20_000);
    let nodule = parse(&src).expect("arity is a flat comma list — parses fine at any width");
    let result = check_nodule(&nodule);
    assert!(
        result.is_err(),
        "expected an explicit over-budget refusal, not success or a SIGABRT"
    );
}

#[test]
fn decision_wide_arity_match_refuses_cleanly() {
    // Hole: `decision::compile_rows` (crates/mycelium-l1/src/decision.rs:113) — decision-tree
    // compilation over the SAME wide-arity match construct as the `usefulness` test above (both
    // functions run while checking one `match` expression's arms; each is its own RFC §4.7 table
    // row / tracked hole, so it gets its own census test even though the repro input is identical).
    let src = wide_arity_match_source(20_000);
    let nodule = parse(&src).expect("arity is a flat comma list — parses fine at any width");
    let result = check_nodule(&nodule);
    assert!(
        result.is_err(),
        "expected an explicit over-budget refusal, not success or a SIGABRT"
    );
}

#[test]
fn grade_deep_list_literal_with_annotation_refuses_cleanly() {
    // Hole: `grade::Gx::grade` (crates/mycelium-l1/src/grade.rs:137). `check_guarantees`
    // (checkty.rs:2205, invoked from `check_and_resolve`/`check_nodule`) walks the RESOLVED
    // (already-desugared) function body — so the same list-literal desugaring that stresses
    // `check_list` above also drives this independent recursive grade walk `n`-deep over the
    // resulting `Expr::App` chain (`grade.rs`'s `Expr::App => self.grade_app(..)` arm recurses).
    let elems = std::iter::repeat_n("0b0000_0001", 50_000)
        .collect::<Vec<_>>()
        .join(", ");
    let src = format!("{VEC_PRELUDE}fn f() => Vec[Binary{{8}}] @ Exact = [{elems}];\n");
    let nodule = parse(&src).expect("the SOURCE is not deeply nested (flat list) — parses fine");
    let result = check_nodule(&nodule);
    assert!(
        result.is_err(),
        "expected an explicit over-budget refusal, not success or a SIGABRT"
    );
}

/// **Documented near-miss, not an open hole.** `parse_type_ref` (`crates/mycelium-l1/src/parse.rs`)
/// is depth-guarded via the shared `MAX_EXPR_DEPTH` budget — a crafted `A -> A -> … -> A` chain past
/// the cap refuses cleanly (an explicit `ParseError`, never a SIGABRT). RFC-0041 §4.2/§7 (W1) RAISED
/// that cap `256 → 4096` (parse.rs:21) to unify it with the shared recursion budget; this test tracks
/// that raise — the guard must keep firing cleanly at the **new** cap, so the crafted chain now
/// exceeds 4096 (the parser runs under the 256 MiB deep worker stack, which supports far more than
/// 4096 parser frames, so the guard fires well before any host-stack overflow).
///
/// **Honesty (VR-5):** unlike the SIGABRT repros in this file, this one always refuses cleanly (it is
/// a guarded near-miss, not an open hole) — kept in the census for uniformity and to pin the raised
/// cap's guard behaviour.
#[test]
fn parse_type_ref_near_miss_guard_fires_at_cap() {
    let chain = "Binary{8} -> ".repeat(5000); // > 4096 `->` hops — past the raised cap (RFC-0041 W1)
    let src = format!("nodule d;\nfn f(g: {chain}Binary{{8}}) => Binary{{8}} = g;\n");
    let result = parse(&src);
    assert!(
        result.is_err(),
        "parse_type_ref is guarded (raised 4096 cap, RFC-0041 §4.2/§7) — a crafted chain past it must \
         refuse cleanly, never SIGABRT (the near-miss the census tracks, not an open hole)"
    );
}

/// A right-nested `Cons`-shaped [`L1Value::Data`] chain, `n` deep — acyclic by construction (the
/// type's own doc comment: "Data values are immutable and acyclic … every field existed before its
/// containing value").
fn deep_cons(n: usize) -> L1Value {
    let byte = mycelium_core::Value::new(
        mycelium_core::Repr::Binary { width: 8 },
        mycelium_core::Payload::Bits(vec![false; 8]),
        mycelium_core::Meta::exact(mycelium_core::Provenance::Root),
    )
    .expect("a well-formed Binary{8} const");
    let mut acc = L1Value::Data {
        ty: "Vec".to_owned(),
        ctor: "Nil".to_owned(),
        fields: std::sync::Arc::new(vec![]),
    };
    for _ in 0..n {
        acc = L1Value::Data {
            ty: "Vec".to_owned(),
            ctor: "Cons".to_owned(),
            fields: std::sync::Arc::new(vec![L1Value::Repr(byte.clone()), acc]),
        };
    }
    acc
}

/// Hole: `L1Value`'s compiler-derived recursive `Clone`/`Drop` glue, plus the hand-written recursive
/// `to_core` (`crates/mycelium-l1/src/eval.rs:117` `to_core`, `:155` `value_contains_substrate_id`) —
/// none depth-guarded. A deep `Cons` chain (built directly here, bypassing the parser entirely — the
/// same shape `check_list`'s desugaring produces at runtime) walks all three `n`-deep.
///
/// **Honesty (FLAG, VR-5):** unlike the checker holes above, `Clone`/`Drop` are compiler-generated —
/// there is no `Result` to assert a clean refusal on today, and `to_core` returns `Option<CoreValue>`
/// (an absent-registry-entry signal, not a depth-budget signal). So this test cannot assert a "clean
/// refusal"; it only constructs and exercises the real repro (the call itself, if unignored on a
/// large enough `n`, is the SIGABRT). `value_contains_substrate_id` (eval.rs:155) is private and
/// reachable only via a live `Substrate`-escape scope-exit path through the full evaluator — building
/// that scenario is out of scope for this census test; it is documented here, not silently dropped.
/// RFC-0041 §4.5/§6 (W3, the within-freeze behavior-preserving-hardening channel) converts the
/// recursive-destruction class to iterative worklists; `to_core`/`value_contains_substrate_id` are
/// expected to gain (or be joined by) an explicit budget at the same time.
#[test]
// RFC-0041 §4.5 (W5, M-979) + M-994 fix (b): `L1Value::Data.fields` is now `Arc`-shared, so `Clone` is
// a derived O(1) refcount bump (no spine walk at all — sharing is sound because `Data` is immutable +
// acyclic), and `Drop` is a hand-written iterative worklist that dismantles a *uniquely-owned* deep
// spine with O(1) host-stack recursion. So a deep `Cons` chain clones (O(1)) and drops (iterative) with
// no SIGABRT. The assertion (construct + clone + drop a 200k-deep chain without aborting) passes.
// (`to_core` / `value_contains_substrate_id` remain recursive — documented above — but are not
// exercised here.)
fn l1value_deep_cons_clone_drop_no_sigabrt() {
    let deep = deep_cons(200_000);
    let cloned = deep.clone(); // derived O(1) `Arc` clone (M-994 fix (b)) — shares the spine, no recursion
    drop(cloned); // shared drop is an O(1) refcount decrement (the unique owner `deep` still holds it)
    drop(deep); // now the unique owner: iterative Drop (RFC-0041 §4.5) dismantles n-deep, no recursion
}
