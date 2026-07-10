//! RFC-0041 §4.7/§5 — the guard-hole **census** (W0 safety net; RR-29 guard-hole inventory; W1
//! closes this crate's hole).
//!
//! Real repro: builds a deep [`mycelium_l1::ast::Expr`] tree directly (bypassing the mycfmt public
//! text entry points on purpose — those parse through `mycelium_l1::parse`, which is ALREADY
//! depth-guarded at 256, so a text-based deep input never reaches this crate's own render family;
//! this crate's render recursion is reachable only from an AST built some other way, e.g. a future
//! non-text AST producer) and calls the private render helper directly (white-box, `use crate::*` —
//! CLAUDE.md test layout). Rust's default stack-overflow handler aborts the process directly (never
//! through panic/unwind), so pre-W1 this test could not assert on the outcome and stayed
//! `#[ignore = "W1"]`d — running it for real would have crashed the whole test binary.
//!
//! **W1 closure:** `render_expr_canonical` (`crates/mycelium-fmt/src/lib.rs`) now wraps its
//! recursive descent in `mycelium_workstack::ensure_sufficient_stack` (a 256 MiB lazily-committed
//! worker stack), so the render itself — not a host-stack overflow — is what returns. The sibling
//! sites named below (`render_flat`, `render_item_flat`/`render_impl_flat`, `render_body_with_comments`)
//! are closed the same way, each wrapped once at its own outer entry (KISS — not threaded through
//! every internal fn).

use crate::*;
use mycelium_l1::ast::{Expr, Literal};

/// A right-nested `Expr::Let` chain, `n` deep.
fn deep_let(n: usize) -> Expr {
    let mut acc = Expr::Lit(Literal::Int(0));
    for i in 0..n {
        acc = Expr::Let {
            name: format!("x{i}"),
            ty: None,
            bound: Box::new(Expr::Lit(Literal::Int(0))),
            body: Box::new(acc),
        };
    }
    acc
}

/// Hole: the "fmt render family" — `render_expr_canonical` and its siblings
/// (`crates/mycelium-fmt/src/lib.rs` `render_expr_canonical`, `render_flat`, `render_item_flat`,
/// `render_impl_flat`, `render_body_with_comments`).
///
/// **Honesty (VR-5):** `render_expr_canonical` returns a plain `String` — infallible — so this test
/// cannot assert a "clean refusal" (there is no error variant to check). What W1 buys is narrower and
/// checked here directly: the call **completes and returns**, on the grown worker stack, instead of
/// aborting the process. The real per-op budget refusal (`BudgetError::DepthExceeded`) is future work
/// (a later wave that charges `RecursionBudget::try_enter` at each frame instead of leaving the depth
/// ceiling unbounded, per this function's `u64::MAX` W1 budget) — not silently upgraded to "guarded"
/// here (G2/VR-5): this test's assertion is exactly "no abort", nothing stronger.
///
/// **Depth choice (`Empirical`).** `50_000` — well past the [`mycelium_l1::parse::MAX_EXPR_DEPTH`]
/// 256 cap and the shared [`mycelium_workstack::RecursionBudget::DEFAULT_DEPTH_LIMIT`] 4096 ceiling
/// (~12×), so this is a genuinely deep stress input, not a near-miss — while staying safely inside
/// the 256 MiB worker's headroom for this function's (debug-build, unoptimized) per-frame size.
/// Empirically: 60,000 still completes; 100,000 overflows the 256 MiB worker in a debug build (this
/// function's frames are larger than `mycelium-stack`'s own synthetic 200,000-deep regression test,
/// which uses a much smaller per-frame footprint) — 50,000 keeps a comfortable margin under that
/// boundary rather than riding it.
#[test]
fn render_expr_canonical_deep_let_chain() {
    let deep = deep_let(50_000);
    let out = render_expr_canonical(&deep);
    // Infallible + grown-stack (W1): the call must complete and hand back a real rendering, not abort.
    assert!(
        !out.is_empty(),
        "render_expr_canonical must complete on the grown worker stack and return a non-empty String"
    );
    // `deep`'s compiler-derived recursive `Drop` glue (walking the `Box<Expr>` chain `n` deep) is a
    // SEPARATE, already-tracked guard hole — `crates/mycelium-l1/tests/guard_hole_census.rs`'s
    // `l1value_deep_cons_clone_drop_no_sigabrt`, explicitly `#[ignore = "W3"]` (RFC-0041 §4.5/§6: the
    // recursive-destruction class converts to iterative worklists in W3, not W1). Dropping `deep`
    // normally here would abort on *that* unrelated hole and falsely implicate this test's actual
    // subject (`render_expr_canonical`'s own recursion). `mem::forget` sidesteps it so this test
    // verifies exactly what W1 closes — nothing stronger, nothing conflated (VR-5).
    std::mem::forget(deep);
}
