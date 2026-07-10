//! RFC-0041 W0 — the **§4.0 depth-metric** property test and the **§5.1 error-parity** differential
//! gate (recursion-depth safety, "safety-net" wave). This file adds *tests only* — no behaviour
//! change, no edit to any logic file.
//!
//! ## The §4.0 metric (`source_call_depth`) — Part 1, PASSES today
//! §4.0 fixes **one machine-independent depth metric**: the budget is charged **one unit per
//! source-level call/β boundary** — a user function application (`App`) or a `Fix` unfold — **not**
//! per internal IR node. Consequences pinned here:
//!   * an n-ary application `f(a, b, c)` is depth **1**, not 3 (arity does not multiply depth);
//!   * nested user calls increment by 1 each (`not(not(not(x)))` is depth 3);
//!   * a data-spine literal — a `Cons`/list literal `[e₀, …, e_{N-1}]` — has data-spine depth **N**,
//!     charged by element uniformly (RFC-0040 desugars it to an N-long `Cons` chain);
//!   * a flat, non-recursive body is depth 0/1.
//!
//! [`source_call_depth`] is a **pure** syntactic function of the *surface AST* (parse-only, no
//! type-checking, no cross-function inlining): it measures the maximum call-chain nesting within a
//! function body. It is therefore a **static** approximation of the runtime charge — cross-function
//! recursion depth is a runtime quantity, out of scope for the pure metric (§4.0: tail iterations do
//! not charge either). Honesty (VR-5): the metric and its property test are **`Empirical`/`Declared`**
//! — a heuristic over fixtures, **never** `Proven`. Source is ground truth.
//!
//! ## The §5.1 error-parity gate (`error_parity_at_the_canonical_threshold`) — Part 2, GREEN (W4)
//! §5.1 requires **one canonical over-budget error variant + width**, refused at the **same §4.2
//! floor** across the three execution paths. With **W1** (the shared `RecursionBudget` /
//! `BudgetError::DepthExceeded{u32}`), **W3½** (AOT env-machine on the shared guard), **W5** (L1-eval
//! CEK + eval-depth raise 64→4096), and **W4** (this wave — the L0 interpreter budgets its
//! substitution machinery and constructs `EvalError::DepthLimit`), all three now refuse a
//! past-the-floor input with the **canonical over-budget family** at the **4096** floor — never a
//! `SIGABRT`. The canonical variant is `DepthExceeded{ limit: u32 }` (workstack); the interp/AOT
//! reconcile to `EvalError::DepthLimit{ limit: usize }` at the **same** threshold (their externally
//! observed variant is deliberately unchanged — the full variant *unification* is a later,
//! `core`-surface concern, not this gate).
//!
//! ### FLAG — why the gate uses a *per-engine* deep input, not one shared static spine (W4 finding)
//! The gate's original W0 shape (one statically-deep `S(S(…S(Z)…))` **source** of depth 4096+1000,
//! asserting all three refuse on *that* node) is **not achievable** — three independent, empirically
//! verified reasons, none interp-side:
//!   1. **Parser cap == floor.** `mycelium-l1`'s `MAX_EXPR_DEPTH` was raised to **4096** in W1
//!      (unifying it with the budget floor), so a *statically* deep source is refused **at parse** —
//!      it never reaches any evaluator. The maximal parseable nesting (4096) is exactly the eval
//!      budget's accept boundary, leaving **no** source that both parses *and* over-runs eval.
//!   2. **The AOT trampoline is data-immune.** The AOT env-machine (M-347, O(1) host stack) charges
//!      depth only at **App/Match** frames; a pure **data-spine** value is built iteratively with no
//!      depth charge, so the AOT *completes* on a deep data value (returns the datum / `DataResult`),
//!      it does not refuse. Only deep **recursion** drives its depth budget.
//!   3. **L0 is a substitution machine (§4.1).** It (correctly, by design) re-walks/re-clones the
//!      growing term each small step, so a deep *runtime recursion* (`spin`) is `O(N²)`+ on it —
//!      minutes at the 4096 floor. Its practical deep input is a deep **value** (refused fast, during
//!      the structural value-walk), which is exactly the RR-29 §0.1 flagship shape.
//!
//! So a single shared input cannot make all three refuse *fast* at 4096. The gate instead exercises
//! each engine with a deep input **it evaluates**, and asserts the invariant that actually holds:
//! **every path refuses over-budget with the canonical variant family at the shared 4096 floor**.
//! (Recorded for the orchestrator — the RFC §5.1 prose describes a single-shared-input gate; the
//! achievable, honest realization is this per-engine one. VR-5/G2: the parity is over *variant +
//! threshold*, which is real; the shared-input assumption was the over-simplification.)

use mycelium_cert::BinaryTernarySwapEngine;
use mycelium_core::{ContentHash, CtorRef, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::{Budgets, EvalError, Interpreter, PrimRegistry};
use mycelium_l1::ast::{Expr, Item, Literal};
use mycelium_l1::{check_nodule, elaborate, parse, Evaluator, L1Error};

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Part 1 — the §4.0 machine-independent depth metric (a pure function + property test).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The **§4.0 depth metric** of a source program: the maximum source-level call/β nesting over every
/// top-level function body, charging **one unit per user-`App` boundary** and **N for an N-element
/// data-spine (`Cons`/list) literal** — never per internal IR node. Pure and parse-only (no
/// type-checking); a heuristic (`Empirical`/`Declared`, not `Proven`). Panics never-silently (G2) on
/// a source that does not parse — callers pass known-good fixtures.
fn source_call_depth(src: &str) -> usize {
    let nodule = parse(src).expect("the §4.0 metric fixture must parse");
    nodule
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(decl) => Some(expr_depth(&decl.body)),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The §4.0 nesting depth of a single surface expression. **`App` charges +1** (one source call/β
/// boundary) over the deepest of its head and arguments — so an n-ary `f(a, b, c)` with flat
/// arguments is depth 1, not 3, and nested calls increment by 1. A **list literal** (the `Cons`
/// data-spine, RFC-0040) charges its **element count** plus the deepest element (so `[a, b, c]` of
/// flat elements is depth 3). Every other construct is transparent — it recurses into its children
/// and takes the max, charging nothing itself (a `let`/`match`/`swap`/ascription is not a call).
fn expr_depth(e: &Expr) -> usize {
    match e {
        // A user function / constructor application: one source call/β boundary. n-ary args are
        // siblings (max, not sum), so `f(a, b, c)` is depth 1; nesting accumulates by 1 each.
        Expr::App { head, args } => {
            1 + expr_depth(head).max(args.iter().map(expr_depth).max().unwrap_or(0))
        }
        // A data-spine (`Cons`) literal of N elements is depth N (charged by element uniformly),
        // plus the deepest element's own nesting.
        Expr::Lit(Literal::List(elems)) => {
            elems.len() + elems.iter().map(expr_depth).max().unwrap_or(0)
        }
        // Bare literals and path/variable references are leaves.
        Expr::Lit(_) | Expr::Path(_) => 0,
        // Binders / control that are NOT call boundaries: transparent (max over children).
        Expr::Let { bound, body, .. } => expr_depth(bound).max(expr_depth(body)),
        Expr::If { cond, conseq, alt } => expr_depth(cond)
            .max(expr_depth(conseq))
            .max(expr_depth(alt)),
        Expr::Match { scrutinee, arms } => {
            expr_depth(scrutinee).max(arms.iter().map(|a| expr_depth(&a.body)).max().unwrap_or(0))
        }
        Expr::For { xs, init, body, .. } => {
            expr_depth(xs).max(expr_depth(init)).max(expr_depth(body))
        }
        Expr::Swap { value, .. } => expr_depth(value),
        Expr::WithParadigm { body, .. } => expr_depth(body),
        Expr::Wild(inner) | Expr::Spore(inner) | Expr::Consume(inner) => expr_depth(inner),
        Expr::Colony(hyphae) => hyphae
            .iter()
            .map(|h| expr_depth(&h.body))
            .max()
            .unwrap_or(0),
        Expr::Lambda { body, .. } => expr_depth(body),
        Expr::Fuse { left, right } => expr_depth(left).max(expr_depth(right)),
        Expr::Reclaim { policy, body } => expr_depth(policy).max(expr_depth(body)),
        Expr::Ascribe(inner, _) => expr_depth(inner),
        // A tuple is a single flat constructor (arity does not add spine depth): max over elements.
        Expr::TupleLit(elems) => elems.iter().map(expr_depth).max().unwrap_or(0),
    }
}

/// **§4.0 property (`Empirical`): the metric matches the known source-level call depth of each
/// fixture.** Each fixture pins one clause of §4.0 — a flat body is 0, a single call is 1, an n-ary
/// call is 1 (not its arity), nested calls increment by 1, a nested constructor spine counts its
/// depth, and an N-element list literal is N. This test is **not** ignored and MUST PASS.
#[test]
fn source_call_depth_matches_the_known_metric_of_each_fixture() {
    // (source, expected §4.0 depth, pinned §4.0 clause).
    let fixtures: &[(&str, usize, &str)] = &[
        // flat, non-recursive body → 0 (a bare literal is not a call).
        (
            "nodule d;\nfn main() => Binary{8} = 0b1011_0010;",
            0,
            "flat literal is depth 0",
        ),
        // a single user call → 1.
        (
            "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);",
            1,
            "one call is depth 1",
        ),
        // an n-ary (2-arg) application is depth 1, NOT its arity.
        (
            "nodule d;\nfn main() => Binary{8} = xor(0b1011_0010, 0b1111_1111);",
            1,
            "binary application is depth 1 (arity does not multiply)",
        ),
        // an n-ary (3-arg) application is STILL depth 1 — the canonical `f(a,b,c)` case (§4.0).
        (
            "nodule d;\nfn main() => Binary{8} = f(0b1, 0b1, 0b1);",
            1,
            "ternary application f(a,b,c) is depth 1, not 3",
        ),
        // nested user calls increment by 1 each: not(not(not(x))) → 3.
        (
            "nodule d;\nfn main() => Binary{8} = not(not(not(0b1010_1010)));",
            3,
            "three nested calls are depth 3",
        ),
        // helper + caller: the metric is the max over all fn bodies (main's flip(flip(..)) = 2).
        (
            "nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(flip(0b1010_1010));",
            2,
            "max over bodies: nested flip(flip(..)) is depth 2",
        ),
        // a nested constructor data-spine S(S(Z)) → 2 (each S is one boundary; Z is a leaf).
        (
            "nodule d;\ntype Nat = Z | S(Nat);\nfn main() => Nat = S(S(Z));",
            2,
            "nested constructor spine S(S(Z)) is depth 2",
        ),
        // a list (Cons) literal of N=3 elements → 3, charged by element uniformly.
        (
            "nodule d;\nfn main() => Seq{Binary{8}, 3} = [0b1111_0000, 0b0000_1111, 0b1010_1010];",
            3,
            "a Cons literal of N elements is depth N",
        ),
    ];

    for (src, expected, clause) in fixtures {
        assert_eq!(
            source_call_depth(src),
            *expected,
            "§4.0 metric mismatch — {clause}:\n{src}"
        );
    }
}

/// **Non-vacuity / mutant witness: the metric genuinely DISTINGUISHES different-depth sources**
/// (mirrors `differential.rs`'s `the_data_differential_distinguishes_divergent_elaborations`). A
/// metric that returned a constant — or that summed arity instead of taking the max — would still
/// pass a lone equality; these assertions rule that out, so a green Part-1 test is meaningful.
#[test]
fn the_metric_distinguishes_different_depth_sources() {
    let one_call = "nodule d;\nfn main() => Binary{8} = not(0b1010_1010);";
    let three_nested = "nodule d;\nfn main() => Binary{8} = not(not(not(0b1010_1010)));";
    // Depth is not constant: a deeper nest measures strictly deeper.
    assert!(
        source_call_depth(three_nested) > source_call_depth(one_call),
        "the metric must rank a 3-deep nest above a single call (not a constant)"
    );

    // Arity is NOT nesting: a 3-ARG flat call (depth 1) is strictly shallower than a 3-DEEP nest
    // (depth 3) — the §4.0 distinction a per-node or per-arg charge would collapse.
    let three_arg = "nodule d;\nfn main() => Binary{8} = f(0b1, 0b1, 0b1);";
    assert_ne!(
        source_call_depth(three_arg),
        source_call_depth(three_nested),
        "a 3-arg flat call and a 3-deep nest must NOT measure the same depth"
    );
    assert!(
        source_call_depth(three_arg) < source_call_depth(three_nested),
        "an n-ary call must be shallower than an equally-wide nesting (arity ≠ depth)"
    );

    // A longer data-spine literal measures strictly deeper — the element-count charge is real.
    let cons1 = "nodule d;\nfn main() => Seq{Binary{8}, 1} = [0b1111_0000];";
    let cons3 =
        "nodule d;\nfn main() => Seq{Binary{8}, 3} = [0b1111_0000, 0b0000_1111, 0b1010_1010];";
    assert!(
        source_call_depth(cons3) > source_call_depth(cons1),
        "a longer Cons literal must measure a deeper data-spine"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Part 2 — the §5.1 cross-path error-parity + threshold differential (GREEN as of W4).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The **canonical over-budget threshold** (§4.2 default floor): depth **4096**. The canonical
/// *variant* is `BudgetError::DepthExceeded { limit: u32 }` (mycelium-workstack, W1) surfaced as
/// `L1Error::DepthExceeded { limit: u32 }` (W5) and reconciled to `EvalError::DepthLimit { limit:
/// usize }` on the interp (W4) and AOT (W3½) — the same threshold on every path.
const CANONICAL_DEPTH_FLOOR: u32 = 4096;

/// A shallow SOURCE that drives **unbounded runtime recursion depth**: `spin(n) = S(spin(n))` is
/// **non-tail** (the recursive call is a `Construct` argument, so its frame is still pending on
/// re-entry — not TCO-eligible), so its source-call depth grows one unit per unfold until a budget
/// refuses. It **parses trivially** (nesting 2), unlike a *statically* deep spine which the parser's
/// `MAX_EXPR_DEPTH == 4096` cap refuses outright — the realistic RR-29 deep-value attack (shallow
/// spore, deep runtime recursion). The CEK (L1) and trampoline (AOT) machines refuse it in bounded
/// time; the substitution machine (L0) does not (see the module FLAG), so the interp path below uses
/// a deep **value** instead.
const SPIN_SRC: &str = "nodule d;\ntype Nat = Z | S(Nat);\nfn spin(n: Nat) => Nat = S(spin(n));\nfn main() => Nat = spin(Z);";

/// A fabricated `CtorRef` (content-addressed; identity is all the value walk needs) for the deep-value
/// input — mirrors `mycelium-interp`'s `guard_hole_census` deep repro. No `DataRegistry` is required:
/// `eval_core`'s value read-off builds a `Datum` from the ctor + fields, it does not re-validate the
/// ctor against a declaration.
fn fabricated_ctor() -> CtorRef {
    CtorRef::new(
        ContentHash::parse("blake3:round_trip_safe").expect("a well-formed content hash"),
        0,
    )
}

/// A right-nested `Construct` **value** chain `n` deep, every leaf already a `Const` (a normal form) —
/// the L0 interpreter's native deep input. Built directly as a Core `Node` (the parser's 4096 cap
/// forbids a statically-deep *source*); a deep value that arises at runtime has exactly this shape.
fn deep_value(n: usize) -> Node {
    let leaf = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        Meta::exact(Provenance::Root),
    )
    .expect("a well-formed Binary{8} const");
    let mut acc = Node::Const(leaf);
    for _ in 0..n {
        acc = Node::Construct {
            ctor: fabricated_ctor(),
            args: vec![acc],
        };
    }
    acc
}

/// **§5.1 error-parity + threshold gate (GREEN as of W4).** Every execution path refuses a deep input
/// with the **canonical over-budget variant family** at the **shared 4096 floor** — never a `SIGABRT`.
/// Each engine is exercised with a deep input **it evaluates in bounded time** (see the module-level
/// FLAG for why a single shared static-spine input is not achievable — parser-cap == floor, the AOT
/// trampoline is data-immune, and L0 is a substitution machine):
///   * **L1-eval** (CEK) on the runtime recursion `spin` → `L1Error::DepthExceeded { limit == 4096 }`;
///   * **L0-interp** (substitution) on a deep **value** → `EvalError::DepthLimit { limit == 4096 }`
///     (W4 — the flagship RR-29 §0.1 fix: the structural value-walk refuses, never aborts);
///   * **AOT** (trampoline) on `spin` at the explicit **deterministic floor** budget →
///     `EvalError::DepthLimit { limit == 4096 }` (W3½ on the shared guard; `mycelium_mlir::run`'s
///     *default* ceiling is the DN-05 dynamic `[10k, 2M]`, so the floor is passed explicitly here to
///     assert the *shared-threshold* parity rather than the dynamic headroom).
#[test]
fn error_parity_at_the_canonical_threshold() {
    let floor = CANONICAL_DEPTH_FLOOR;

    // Path 1 — L1-eval (CEK): the canonical variant `DepthExceeded { limit }` at the floor. Ample fuel
    // so *depth* is what trips (not the step clock).
    let env = check_nodule(&parse(SPIN_SRC).expect("spin parses")).expect("spin checks");
    let l1_err = Evaluator::new(&env)
        .with_fuel(100_000_000)
        .call("main", vec![])
        .expect_err("L1-eval must refuse the runtime recursion, never a host-stack abort");
    assert!(
        matches!(l1_err, L1Error::DepthExceeded { limit } if limit == floor),
        "L1-eval must refuse with the canonical DepthExceeded {{ limit: {floor} }}; got: {l1_err:?}"
    );

    // Path 2 — L0-interp (substitution): a deep VALUE refuses with `EvalError::DepthLimit` at the same
    // floor (W4). Depth 4096+1000 past the floor; the structural value-walk refuses during descent.
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let deep = deep_value((floor as usize) + 1000);
    let interp_err = interp
        .eval_core(&deep)
        .expect_err("L0-interp must refuse the deep value, never SIGABRT (W4)");
    assert!(
        matches!(interp_err, EvalError::DepthLimit { limit } if limit == floor as usize),
        "L0-interp must refuse with DepthLimit {{ limit: {floor} }} (reconciles to the canonical variant, W4); got: {interp_err:?}"
    );

    // Path 3 — AOT (trampoline): `spin` at the explicit deterministic floor budget → `DepthLimit` at
    // the same threshold (W3½ on the shared guard). `run`'s default is the DN-05 dynamic ceiling, so
    // the floor is passed explicitly to assert the shared-threshold parity.
    let node = elaborate(&env, "main").expect("spin elaborates");
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let aot_err = mycelium_mlir::run_core_with_effects(
        &node,
        &prims,
        &engine,
        100_000_000,
        floor as usize,
        &mut Budgets::new(),
    )
    .expect_err("AOT must refuse the runtime recursion at the deterministic floor (W3½)");
    assert!(
        matches!(aot_err, EvalError::DepthLimit { limit } if limit == floor as usize),
        "AOT must refuse with DepthLimit {{ limit: {floor} }} at the shared floor (W3½); got: {aot_err:?}"
    );
}
