//! RFC-0041 §4.7/§5 — the guard-hole **census** (W0 safety net; RR-29 guard-hole inventory turned
//! into a tracked failing test for this crate's hole).
//!
//! Real repro: constructs a genuinely deep [`Node`] and calls the hole's entry point. Rust's default
//! stack-overflow handler aborts the process directly (never through panic/unwind), so this is not
//! `catch_unwind`-able — the test stayed `#[ignore = "W1"]`d until the wave landed. **RFC-0041 §4.7
//! W1 (this crate):** `llm_canonical` now wraps `render_node` in
//! [`mycelium_workstack::ensure_sufficient_stack`] (`crates/mycelium-lsp/src/project.rs`), so the
//! deep render below runs on the grown 256 MiB worker stack and completes cleanly instead of
//! aborting the test binary — the `#[ignore]` is dropped and the census now asserts clean completion.

use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_lsp::project::llm_canonical;
use mycelium_workstack::{ensure_sufficient_stack, RecursionBudget};

/// A right-nested `Node::Let` chain, `n` deep — mirrors the shape `render_node`'s `Node::Let` arm
/// recurses on (`crates/mycelium-lsp/src/project.rs` `render_node`, dispatched from the public
/// [`llm_canonical`]).
fn deep_let(n: usize) -> Node {
    let byte = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        Meta::exact(Provenance::Root),
    )
    .expect("a well-formed Binary{8} const");
    let mut acc = Node::Const(byte.clone());
    for i in 0..n {
        acc = Node::Let {
            id: format!("x{i}"),
            bound: Box::new(Node::Const(byte.clone())),
            body: Box::new(acc),
        };
    }
    acc
}

/// Closed hole: `render_node` (`crates/mycelium-lsp/src/project.rs`), reached via the public
/// [`llm_canonical`] (RFC-0021 §4.6 `LlmCanonical` projection), now runs on the grown host stack.
///
/// **Honesty (VR-5):** at W1, `llm_canonical` returned a plain `String` — infallible by signature —
/// so this test asserted only **clean completion**, never a `Result` refusal. **RFC-0041 §4.2/§9 W7
/// (process-arena coverage, `docs/notes/W7-arena-coverage-audit.md`):** `llm_canonical` is now
/// `Result<String, BudgetError>` — it charges the shared process-wide arena before rendering — but
/// this 20,000-deep fixture's estimated cost (~1.3 MB) sits well under the crate's declared default
/// ceiling (256 MiB), so it still renders successfully; the arena wiring never changes this test's
/// depth-safety claim. The W1 fix is "never a host-stack abort", not "reject deep input" — a
/// *depth*-refusal budget for this projection remains later work (§4.7); the census here documents
/// exactly that scope. A synthetic *arena* refusal (a tiny injected ceiling) is covered separately by
/// `src/tests/project.rs::large_synthetic_input_trips_out_of_budget`.
///
/// **FLAG (test-harness artifact, out of scope for this leaf):** `Node`'s `Drop` glue is itself
/// recursive (one frame per nesting level, compiler-generated for the `Box<Node>` fields), so
/// dropping a deep fixture on the test harness's default per-test thread stack would overflow
/// *that* — a hazard of this synthetic fixture's construction/teardown, not of the `render_node`
/// guard hole under test (a real LSP host never constructs/drops a whole document tree in one
/// recursive call the way this fixture builder's `Drop` does). The whole test body — build, render,
/// assert, drop — runs inside the same [`ensure_sufficient_stack`] worker the production fix uses,
/// so the fixture's construction and teardown get the same 256 MiB headroom as the render.
///
/// **Depth choice:** `n = 20_000` is ~78× the L1 parser's 256-frame depth guard (and ~312× the
/// evaluator's 64) — comfortably past the old unguarded default host-stack size (a few MiB), which
/// is what made the original `#[ignore]`d repro SIGABRT. It is *not* `RecursionBudget::DEFAULT_DEPTH_LIMIT`
/// (4096) sized up to the crate's historical repro constant (200,000): this render is `O(depth²)`
/// (each `Node::Let`/`Node::App`/… arm's `format!` copies its already-rendered sub-`String` into a
/// new buffer one level up — the same re-walk the shared budget's `WorkSteps` charge exists to
/// guard, per `mycelium-workstack`'s docs), so 200,000 levels pushes a **debug** build's per-frame
/// stack usage past the 256 MiB guard *even wrapped* (empirically confirmed) and the runtime into
/// tens of seconds; 20,000 keeps the census fast and comfortably within budget while still being
/// unambiguously deeper than anything the pre-fix code could survive.
#[test]
fn render_node_deep_let_chain() {
    let budget = RecursionBudget::with_depth_default(u64::MAX, u64::MAX);
    ensure_sufficient_stack(&budget, || {
        let n = 20_000;
        let deep = deep_let(n);
        let rendered = llm_canonical(&deep)
            .expect("20k-deep chain's estimated cost stays well under the 256 MiB default ceiling");

        // Clean completion: the process did not abort, and the render is well-formed — one
        // `(let [xI …` opener per nesting level (`deep_let` nests right-to-left, so the outermost
        // binder is `x{n-1}` and the innermost is `x0`, wrapping the original constant), plus the
        // constant payload.
        assert_eq!(rendered.matches("(let [x").count(), n);
        assert!(rendered.starts_with(&format!("(let [x{} ", n - 1)));
        assert!(rendered.contains("(let [x0 "));
        assert!(rendered.contains("(const 0b00000000"));
    });
}
