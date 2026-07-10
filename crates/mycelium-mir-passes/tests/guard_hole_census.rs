//! RFC-0041 §4.7/§5 — the guard-hole **census** (W0 safety net; RR-29 guard-hole inventory turned
//! into tracked failing tests, one per hole this crate owns).
//!
//! Real repros: each test constructs a genuinely deep [`Node`] and calls the hole's entry point.
//! Rust's default stack-overflow handler aborts the process directly (never through panic/unwind),
//! so none of this is `catch_unwind`-able — a still-open hole's test stays `#[ignore = "Wn"]`d;
//! running one for real would crash the whole test binary. **W1 closes both holes this crate owns**
//! (`emit_owned`/`count_occurrences`, RFC-0041 §4.7 W1) via the shared `mycelium-workstack` budget +
//! guarded deep stack, so their `#[ignore]` is dropped below and the assertion holds for real.

use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_mir_passes::emit::{count_occurrences, emit_owned};
use mycelium_workstack::{ensure_sufficient_stack, RecursionBudget};

/// A right-nested `Node::Let` chain, `n` deep, referencing `var` at its innermost leaf.
///
/// **Fixture note (not an `emit`/RFC-0041 hole):** `Node`'s `Box<Node>` chain has the ordinary
/// derived (recursive) `Drop` glue, so a 200,000-deep value built by this helper must be built,
/// exercised, *and dropped* on a guarded deep stack — a bare `let deep = deep_let(...)` on the
/// test's own (default-sized) thread would itself SIGABRT when `deep` goes out of scope, entirely
/// independent of whichever `emit`-side hole the test targets. Every caller below therefore does
/// the whole construct→call→drop lifecycle inside one [`ensure_sufficient_stack`] closure.
fn deep_let(n: usize, var: &str) -> Node {
    let byte = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        Meta::exact(Provenance::Root),
    )
    .expect("a well-formed Binary{8} const");
    let mut body = Node::Var(var.to_owned());
    for i in 0..n {
        body = Node::Let {
            id: format!("y{i}"),
            bound: Box::new(Node::Const(byte.clone())),
            body: Box::new(body),
        };
    }
    body
}

#[test]
fn emit_owned_deep_let_chain_refuses_cleanly() {
    // Was: `emit_owned` (crates/mycelium-mir-passes/src/emit.rs) — recurses through `Let`'s `bound`
    // and `body`. Closed (RFC-0041 §4.7 W1): the outer call runs on the `mycelium-workstack` deep
    // worker stack and every recursive step charges a `RecursionBudget::try_enter` (default depth
    // ceiling 4096), so a chain deeper than the ceiling refuses cleanly well before the host stack
    // is at risk.
    //
    // **Depth choice (10,000, not this file's other tests' 200,000):** `emit_owned` charges
    // `count_occurrences` once per binder — the `O(N²)` re-walk flagged as a W2/profiling residual
    // (see the `count_occurrences` test below) — so the *cost of reaching the refusal* scales with
    // `min(depth_ceiling, n) * n`. 10,000 is still comfortably beyond the 4096 ceiling (so the guard
    // genuinely trips on deep input, and 10,000 native `Let` frames would SIGABRT an unguarded
    // default thread stack too) while keeping this test's wall-clock reasonable; 200,000 here would
    // (correctly, but needlessly slowly) re-derive the same `O(N²)` residual this file already notes.
    //
    // The whole construct→call→drop lifecycle runs inside `ensure_sufficient_stack` (see the
    // `deep_let` fixture note): `emit_owned` itself would refuse at depth 4096 regardless, but the
    // *fixture's* drop needs the guarded stack too.
    let budget = RecursionBudget::default();
    let refused = ensure_sufficient_stack(&budget, || {
        let deep = deep_let(10_000, "x");
        emit_owned(&deep).is_err()
    });
    assert!(
        refused,
        "expected an explicit over-budget refusal, not success or a SIGABRT"
    );
}

/// Was: `count_occurrences` (`crates/mycelium-mir-passes/src/emit.rs`).
///
/// **Honesty (FLAG, VR-5):** `count_occurrences` returns a plain `usize` — infallible, so this test
/// cannot assert a "clean refusal" the way the fallible `emit_owned` test above does. RFC-0041 §4.7
/// W1 closes the **host-stack-overflow** hole (the entry point now runs on the `mycelium-workstack`
/// deep worker stack, so this 200,000-deep chain completes rather than SIGABRTing); it does **not**
/// add a CPU/work-step bound (that needs an infallible→fallible signature change, which would ripple
/// into `is_fully_borrowable`/`is_sole_owned_move` and the `emit_elided`/`emit_reuse` path — out of
/// this leaf's scope). The `O(N²)` re-walk cost (this function is called once per binder) is an
/// explicitly flagged residual for W2/profiling.
#[test]
fn count_occurrences_deep_let_chain() {
    // Whole construct→call→drop lifecycle inside `ensure_sufficient_stack` — see the `deep_let`
    // fixture note (the 200,000-deep fixture's own `Drop` needs the guarded stack, independent of
    // `count_occurrences`'s own guard).
    let budget = RecursionBudget::default();
    let got = ensure_sufficient_stack(&budget, || {
        let deep = deep_let(200_000, "x");
        count_occurrences(&"x".to_owned(), &deep)
    });
    assert_eq!(
        got, 1,
        "the innermost leaf is the sole free occurrence of `x`"
    );
}
