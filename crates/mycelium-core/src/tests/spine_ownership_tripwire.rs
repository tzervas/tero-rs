//! RFC-0041 §4.5 (W7 / item #10) — **Box-owned / acyclic spine tripwire** for the frozen types whose
//! `Drop` was made iterative (`mem::replace`/`mem::take` worklist teardown): [`crate::node::Node`]
//! (+ its [`crate::node::Alt`] arm) and the [`crate::datum::Datum`] ↔ [`crate::datum::CoreValue`]
//! cluster.
//!
//! ## The invariant being guarded
//! Those iterative `Drop`s are double-free-safe **only** under the *Box-owned / acyclic / no-shared-
//! spine* precondition (RFC-0041 §4.5; the `Low freeze11` recorded precondition in `node.rs`/
//! `datum.rs`): every recursive child is uniquely owned via `Box<…>` / `Vec<…>`, so the worklist
//! visits and frees each node **exactly once**. A future intern/DAG cache putting `Rc<…>`/`Arc<…>`
//! (or any shared-ownership handle) on the value spine would **silently** invalidate that — the same
//! node reached by two owners would be torn down (and freed) twice: a **double-free**. Because the
//! iterative `Drop` looks correct in isolation, nothing else would flag the regression; this tripwire
//! is the guard.
//!
//! This is the *within-freeze* hardening channel of **DN-56 §6** (RFC-0041 §6): a behavior-preserving
//! recursion→iteration transform admitted on frozen types under an explicit bar — of which "the
//! Box-owned invariant still holds" is a standing precondition, not a one-time check. So it earns a
//! standing test.
//!
//! ## Why a source-structural scan (mechanism choice — VR-5)
//! A `static_assertions`/trait-bound check on the *current* field types would (a) need a new
//! dependency (the workspace manifest is orchestrator-owned — out of a leaf's edit scope) and, worse,
//! (b) only pin the field types that exist *today* — it would **not** catch a newly-added
//! `Rc`/`Arc`-typed **intern-cache field**, which is exactly the change that breaks the invariant. A
//! lexical scan of the (non-test portion of the) frozen modules for `Rc`/`Arc`/`Weak` catches **both**
//! a field-type change (`Vec<CoreValue>` → `Vec<Rc<CoreValue>>`) **and** an added shared-ownership
//! field or import anywhere in the type's module. These modules contain **zero** legitimate `Rc`/`Arc`
//! today (a `#![forbid(unsafe_code)]` value core), so any appearance is a real signal. This is the
//! most reliable available mechanism for "fail if a future interning change puts shared ownership on
//! the spine".
//!
//! ## Out-of-crate sibling (FLAG): `mycelium-l1::eval::L1Value`
//! The same invariant backs the iterative `L1Value` teardown in `mycelium-l1` (RFC-0041 §4.5, W5). It
//! lives in a **different crate** (outside this leaf's `mycelium-core` scope), so it is **not** covered
//! here — the identical `Rc`/`Arc`-on-spine tripwire should be added in `mycelium-l1`'s in-crate tests.
//! Flagged up for the wave (see the leaf report).

/// The non-test portion of a source file: everything before its first `#[cfg(test)]`. Scanning only
/// this region keeps the tripwire focused on the frozen *type definitions + their `Drop`/`Clone`
/// machinery* and immune to any `Rc`/`Arc` a future *test* helper might legitimately use.
fn logic_head(src: &str) -> &str {
    match src.find("#[cfg(test)]") {
        Some(idx) => &src[..idx],
        None => src,
    }
}

/// Shared-ownership handles whose presence on the value spine would break the "each node freed exactly
/// once" precondition of the iterative `Drop`s (double-free). `Rc`/`Arc`/their `Weak`s, spelled either
/// bare or path-qualified.
const SHARED_OWNERSHIP_TOKENS: &[&str] = &[
    "Rc<",
    "Arc<",
    "Weak<",
    "Rc::",
    "Arc::",
    "rc::Rc",
    "sync::Arc",
];

fn assert_no_shared_ownership(module: &str, src: &str) {
    let head = logic_head(src);
    for tok in SHARED_OWNERSHIP_TOKENS {
        assert!(
            !head.contains(tok),
            "{module}: found shared-ownership token `{tok}` on/near the frozen value spine. The \
             iterative Drop (RFC-0041 §4.5; DN-56 §6 within-freeze channel) is double-free-safe ONLY \
             under the Box-owned/acyclic/no-shared-spine invariant — an Rc/Arc intern or DAG cache on \
             the spine SILENTLY breaks it (a node reached by two owners is freed twice). If this is a \
             deliberate interning change, the iterative Drop/Clone MUST be reworked (ref-count-aware) \
             FIRST, then this tripwire updated deliberately."
        );
    }
}

#[test]
fn node_spine_stays_box_owned_acyclic() {
    let src = include_str!("../node.rs");
    assert_no_shared_ownership("node.rs (Node/Alt)", src);
    // Positively confirm we are scanning the real recursive spine: `Node`'s children are `Box`/`Vec`
    // owned. If the spine field shapes change, revisit the invariant (and this assertion) on purpose.
    let head = logic_head(src);
    assert!(
        head.contains("Box<Node>") && head.contains("Vec<Node>"),
        "node.rs: expected the recursive spine to be Box<Node>/Vec<Node>-owned; the spine shape \
         changed — re-audit the iterative Drop precondition (RFC-0041 §4.5)"
    );
}

#[test]
fn datum_cluster_spine_stays_box_owned_acyclic() {
    let src = include_str!("../datum.rs");
    assert_no_shared_ownership("datum.rs (Datum/CoreValue)", src);
    // `Datum`'s recursive spine is `Vec<CoreValue>` (uniquely owned). Confirm we scanned it.
    let head = logic_head(src);
    assert!(
        head.contains("Vec<CoreValue>"),
        "datum.rs: expected Datum's recursive spine to be Vec<CoreValue>-owned; the spine shape \
         changed — re-audit the iterative Drop precondition (RFC-0041 §4.5)"
    );
}
