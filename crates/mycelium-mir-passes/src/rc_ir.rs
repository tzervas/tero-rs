//! The RC-annotated IR (`RcNode`) — DN-33 §8.1 Q2 / MEM-4.
//!
//! This is the **separate** intermediate representation the maintainer ratified (DN-33 §8.1 Q2):
//! the own/borrow mode and the emitted `Dup`/`Drop` reference-counting operations live here, on a
//! mirror of the Core IR `Node` grammar — **not** on the trusted Core IR. `mycelium-core`'s `Node`
//! stays pristine (KC-3 / DN-33 §4: the audited kernel does not grow; MEM-4's correctness obligation
//! is confined to this optimisation-only crate).
//!
//! # Relationship to the Core IR
//!
//! `RcNode` mirrors `mycelium_core::Node` over the **first-order fragment**
//! (`Const/Var/Let/Op/Swap/Construct/Match/Lam/App`) and adds two reference-counting wrapper nodes:
//!
//! - [`RcNode::Dup`] — `dup x; body`: increments `x`'s reference count, then evaluates `body`.
//! - [`RcNode::Drop`] — `drop x; body`: decrements `x`'s reference count (reclaiming at zero — the
//!   point a `ReclamationRecord(RcZero)` would fire in the runtime, RFC-0027 §10.1), then `body`.
//!
//! Each binding form ([`RcNode::Lam`]) carries an explicit [`Mode`] — `Owned` in the naive emission
//! (MEM-4·B0); `Borrowed` is the *output* of MEM-4 Increment 1's borrow elision.
//!
//! # Recursion is out of scope here (G2 — never-silent)
//!
//! `Fix`/`FixGroup` (recursion) are **deliberately absent** from `RcNode`: reference counting of
//! recursive bindings is a later increment (DN-33 §6). The emission ([`crate::emit`]) returns an
//! explicit error for them rather than mis-handling them silently.
//!
//! Guarantee: the IR is a structural mirror — `Exact` by construction (no approximation in the
//! representation itself).

use mycelium_core::{CtorRef, PolicyRef, Prim, Repr, Value, VarId};

/// The ownership mode of a binding (DN-33 §2 / §8.1 Q2).
///
/// `Owned` is the naive default (MEM-4·B0): the binding holds a reference that must be `Dup`'d per
/// extra use and `Drop`'d if unused. `Borrowed` is the output of MEM-4 Increment 1's non-escaping
/// borrow elision — a borrowed binding is *read without consuming*, so it needs **no** `Dup`/`Drop`.
///
/// Guarantee: `Exact` — a two-state tag with no approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// The binding owns a reference: extra uses `Dup`, an unused binding `Drop`s (naive default).
    Owned,
    /// The binding is borrowed (read-only, non-consuming): no `Dup`/`Drop` (Increment 1 output).
    Borrowed,
}

/// One alternative of a flat [`RcNode::Match`] — mirrors `mycelium_core::Alt`.
#[derive(Debug, Clone, PartialEq)]
pub enum RcAlt {
    /// A constructor arm binding the constructor's fields to `binders` (in declaration order).
    Ctor {
        /// The constructor matched.
        ctor: CtorRef,
        /// The field binders, in declaration order.
        binders: Vec<VarId>,
        /// The arm body, in scope of `binders`.
        body: RcNode,
    },
    /// A literal arm matching a representation value.
    Lit {
        /// The literal value to match.
        value: Value,
        /// The arm body.
        body: RcNode,
    },
}

/// An RC-annotated Core IR node (the first-order fragment + `Dup`/`Drop`).
///
/// Mirrors `mycelium_core::Node` and adds the [`RcNode::Dup`]/[`RcNode::Drop`] reference-counting
/// wrappers and the per-binding [`Mode`]. Produced by [`crate::emit`]; checked by
/// [`crate::balance`].
///
/// Guarantee: `Exact` — a faithful structural mirror; the RC ops are inserted by the emission pass.
#[derive(Debug, Clone, PartialEq)]
pub enum RcNode {
    /// A constant value (allocates one reference when evaluated).
    Const(Value),
    /// A variable reference — a **consuming use** (move) of one reference of the variable.
    Var(VarId),
    /// A **borrowing read** of a variable — non-consuming (the reference count is unchanged; the
    /// value must be live). Produced by MEM-4 Increment 1 (borrow elision) for reads in
    /// reader-primitive positions (`Op`/`Swap` arguments) in the immutable value model, where a
    /// primitive reads its operands and produces a fresh result, retaining nothing.
    Borrow(VarId),
    /// A **sole-owner consuming move** — semantically identical to [`RcNode::Var`] (it consumes one
    /// reference), but **statically proven** to be the unique owner at this point (reference count
    /// is exactly 1). Produced by MEM-4 Increment 2 (`rc == 1` reuse annotation, DN-33 §6 D-3): the
    /// runtime `RcCell::drop_ref` `UniqueOwner` branch is guaranteed to fire here, so the value's
    /// allocation is **FBIP-reuse-eligible** (the freed cell may be recycled by codegen). The static
    /// claim is **machine-verified** by [`crate::eval`] (it errors if the rc is not 1 here).
    MoveUnique(VarId),
    /// `dup var; body` — increment `var`'s reference count, then evaluate `body`.
    Dup {
        /// The variable whose reference count is incremented.
        var: VarId,
        /// The continuation, evaluated after the increment.
        body: Box<RcNode>,
    },
    /// `drop var; body` — decrement `var`'s reference count (reclaim at zero), then evaluate `body`.
    Drop {
        /// The variable whose reference count is decremented.
        var: VarId,
        /// The continuation, evaluated after the decrement.
        body: Box<RcNode>,
    },
    /// `let r = body in (drop var; r)` — evaluate `body` to its result, **then** decrement `var`'s
    /// reference count (reclaim at zero), then yield the result. Distinct from [`RcNode::Drop`]
    /// (which decrements *before* its body): the borrowed value must stay live **through** its reads
    /// inside `body`, so its reclamation is placed *after* the body completes. Produced by MEM-4
    /// Increment 1 to reclaim a fully-borrowed binding once its reads are done.
    DropAfter {
        /// The variable reclaimed after `body` completes.
        var: VarId,
        /// The body evaluated (and read from) before the decrement.
        body: Box<RcNode>,
    },
    /// A let binding. `id` is owned within `body`; its `Dup`/`Drop` annotations are placed at the
    /// top of `body` by the emission pass.
    Let {
        /// Bound name.
        id: VarId,
        /// The bound expression.
        bound: Box<RcNode>,
        /// The body in which `id` is in scope.
        body: Box<RcNode>,
    },
    /// A paradigm-specific primitive application.
    Op {
        /// The primitive.
        prim: Prim,
        /// Operands.
        args: Vec<RcNode>,
    },
    /// The representation-changing node (mirrors `Node::Swap`).
    Swap {
        /// The value being converted.
        src: Box<RcNode>,
        /// The target representation.
        target: Repr,
        /// The policy that chose/justified the swap.
        policy: PolicyRef,
    },
    /// A saturated constructor application.
    Construct {
        /// The constructor.
        ctor: CtorRef,
        /// The field expressions, in declaration order.
        args: Vec<RcNode>,
    },
    /// A flat pattern match.
    Match {
        /// The value being scrutinised.
        scrutinee: Box<RcNode>,
        /// The alternatives, tried first-match.
        alts: Vec<RcAlt>,
        /// The catch-all branch.
        default: Option<Box<RcNode>>,
    },
    /// A lambda abstraction with an explicit ownership [`Mode`] on its parameter.
    Lam {
        /// The bound parameter.
        param: VarId,
        /// The parameter's ownership mode (`Owned` in the naive emission; `Borrowed` after elision).
        mode: Mode,
        /// The body, in scope of `param`.
        body: Box<RcNode>,
    },
    /// Application — apply `func` to `arg`, call-by-value.
    App {
        /// The function being applied.
        func: Box<RcNode>,
        /// The argument.
        arg: Box<RcNode>,
    },
}

impl RcNode {
    /// Wrap `body` in `n` nested `Dup { var }` nodes (innermost is `body`).
    ///
    /// Used by the emission pass to give an owned binding one reference per extra use.
    ///
    /// Guarantee: `Exact` — produces exactly `n` `Dup` wrappers.
    #[must_use]
    pub fn dup_n(var: &VarId, n: usize, body: RcNode) -> RcNode {
        let mut acc = body;
        for _ in 0..n {
            acc = RcNode::Dup {
                var: var.clone(),
                body: Box::new(acc),
            };
        }
        acc
    }

    /// Wrap `body` in a single `Drop { var }` node.
    ///
    /// Guarantee: `Exact` — produces exactly one `Drop` wrapper.
    #[must_use]
    pub fn drop_one(var: &VarId, body: RcNode) -> RcNode {
        RcNode::Drop {
            var: var.clone(),
            body: Box::new(body),
        }
    }

    /// Wrap `body` in a single `DropAfter { var }` node (reclaim `var` *after* `body`).
    ///
    /// Guarantee: `Exact` — produces exactly one `DropAfter` wrapper.
    #[must_use]
    pub fn drop_after(var: &VarId, body: RcNode) -> RcNode {
        RcNode::DropAfter {
            var: var.clone(),
            body: Box::new(body),
        }
    }
}
