//! Naive (fully-owned) RC-emission lowering `Node → RcNode` — MEM-4·B0 / DN-33 §6.
//!
//! This is the **foundation** the investigation (DN-33 §6.1) found missing: a lowering that *emits*
//! the reference-counting operations, so that MEM-4's elision (Increment 1) has something to elide.
//! It is **naive / fully-owned**: every binding owns its reference, so a binding used `k` times gets
//! `k-1` `Dup`s (one reference per use) and each use consumes one; a binding used `0` times gets one
//! `Drop`. No borrowing yet — that is Increment 1.
//!
//! # Scope — the first-order fragment (G2: never-silent on the rest)
//!
//! Total over `Const/Var/Let/Op/Swap/Construct/Match/Lam/App`. Recursion (`Fix`/`FixGroup`) is
//! **out of scope** for this increment (RC of recursive bindings is harder — DN-33 §6) and returns
//! an explicit [`EmitError::UnsupportedNode`] rather than being silently mis-emitted.
//!
//! # The emission rule (per owned binding)
//!
//! For a binding of `x` with `k` consuming uses in its scope body:
//! - `k == 0` → prepend one `Drop x` (the bound value is reclaimed immediately; never leaked).
//! - `k >= 1` → prepend `k-1` `Dup x` (so there are `k` references; the `k` uses consume them).
//!
//! The bound value starts with one reference (produced by evaluating `bound`), so the net is
//! `1 + (k-1) == k` references created and `k` consumed → balance zero (verified independently by
//! [`crate::balance`]).
//!
//! Guarantee: the emission is `Exact` **for the balance property** — by construction every owned
//! binding is reference-balanced (proven independently by the balance check, and mutation-tested).
//! No performance claim is made (B0 emits the *most* RC ops; Increment 1 removes the redundant
//! ones).

use mycelium_core::{Alt, Node, VarId};
use mycelium_workstack::{ensure_sufficient_stack, BudgetError, RecursionBudget};

use crate::rc_ir::{Mode, RcAlt, RcNode};

/// Why RC-emission could not lower a node.
///
/// Exhaustive and never-silent (G2): a node outside the supported fragment is an explicit error,
/// not a silent pass-through or a wrong emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// The node is outside the first-order fragment this increment supports (e.g. `Fix`/`FixGroup`).
    /// Carries the node kind for diagnostics.
    UnsupportedNode(&'static str),
    /// `emit_owned`'s own AST-traversal recursion (over `Let`'s `bound`/`body`, `Match` arms, `App`
    /// spines, …) exceeded the shared [`mycelium_workstack::RecursionBudget`] depth ceiling
    /// (RFC-0041 §4.7/§5.1 — guard hole RR-29). Refused cleanly here rather than overflowing the
    /// host stack (G2); reconciles with the interp/AOT `DepthLimit` family in W4/W3½.
    DepthExceeded {
        /// The depth ceiling (source-call/β-nesting frames) that was reached.
        limit: u32,
    },
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::UnsupportedNode(kind) => write!(
                f,
                "RC-emission does not yet support `{kind}` (recursion is a later MEM-4 increment — \
                 DN-33 §6); refusing rather than mis-emitting (G2)"
            ),
            EmitError::DepthExceeded { limit } => write!(
                f,
                "RC-emission's own AST-traversal recursion exceeded the depth budget (limit \
                 {limit} frames) — refusing rather than overflowing the host stack (RFC-0041 §4.7, G2)"
            ),
        }
    }
}

impl std::error::Error for EmitError {}

impl From<BudgetError> for EmitError {
    /// `emit_owned` only ever charges the depth guard ([`RecursionBudget::try_enter`]), so in
    /// practice this always sees [`BudgetError::DepthExceeded`]; the `OutOfBudget` arm is handled
    /// defensively (never a silent drop of the ceiling, G2) should a future increment add
    /// byte/step charging here.
    fn from(err: BudgetError) -> Self {
        match err {
            BudgetError::DepthExceeded { limit } => EmitError::DepthExceeded { limit },
            BudgetError::OutOfBudget { limit, .. } => EmitError::DepthExceeded {
                limit: u32::try_from(limit).unwrap_or(u32::MAX),
            },
        }
    }
}

/// Lower a Core IR [`Node`] to the naive fully-owned [`RcNode`] (MEM-4·B0).
///
/// Returns [`EmitError::UnsupportedNode`] for `Fix`/`FixGroup` (G2 — never-silent on recursion), and
/// [`EmitError::DepthExceeded`] if the traversal's own recursion exceeds the shared
/// [`RecursionBudget`] depth ceiling (RFC-0041 §4.7 — guard hole RR-29): the outermost call runs on
/// the deep worker stack ([`ensure_sufficient_stack`]) so a genuinely deep input never SIGABRTs the
/// host, and each recursive step charges [`RecursionBudget::try_enter`] so a pathological input
/// refuses cleanly at the depth ceiling instead of exhausting even the deep worker stack.
///
/// Guarantee: `Exact` for the balance property (every owned binding is reference-balanced by
/// construction). See [`crate::balance::check_balance`] for the independent verification.
pub fn emit_owned(node: &Node) -> Result<RcNode, EmitError> {
    // W1: the outer `budget` arg is not yet consulted for sizing (the deep worker stack is already
    // generous); the REAL depth guard is the budget created and charged inside the closure, entirely
    // on the worker thread (mirrors `mycelium-workstack`'s own `ensure_sufficient_stack` usage
    // precedent: `RecursionBudget` is `Send` but not `Sync`, so it is owned inside `f`, not borrowed
    // across the thread boundary).
    let outer = RecursionBudget::default();
    ensure_sufficient_stack(&outer, || {
        let budget = RecursionBudget::default();
        emit_owned_guarded(node, &budget)
    })
}

/// The guarded recursive core of [`emit_owned`]: identical traversal, but every recursive step
/// charges `budget.try_enter()` (RAII-released on return) so the depth ceiling — not a host-stack
/// overflow — always bounds a pathological input (RFC-0041 §4.0/§4.7).
///
/// The per-binder occurrence counts route through [`count_occurrences_inner`] (not the public
/// [`count_occurrences`]): this function only ever runs inside [`emit_owned`]'s
/// [`ensure_sufficient_stack`] closure, i.e. already on the deep worker stack, so re-entering the
/// public wrapper here would needlessly spawn a fresh 256 MiB worker thread **per binder** — turning
/// the already-`O(N²)` re-walk (§4.7 residual) into an `O(N)` **thread-spawn** multiplier too.
fn emit_owned_guarded(node: &Node, budget: &RecursionBudget) -> Result<RcNode, EmitError> {
    let _guard = budget.try_enter()?;
    match node {
        Node::Const(v) => Ok(RcNode::Const(v.clone())),
        Node::Var(x) => Ok(RcNode::Var(x.clone())),
        Node::Let { id, bound, body } => {
            let rc_bound = emit_owned_guarded(bound, budget)?;
            let k = count_occurrences_inner(id, body);
            let rc_body = emit_owned_guarded(body, budget)?;
            Ok(RcNode::Let {
                id: id.clone(),
                bound: Box::new(rc_bound),
                body: Box::new(balance_binder(id, k, rc_body)),
            })
        }
        Node::Op { prim, args } => Ok(RcNode::Op {
            prim: prim.clone(),
            args: emit_args(args, budget)?,
        }),
        Node::Swap {
            src,
            target,
            policy,
        } => Ok(RcNode::Swap {
            src: Box::new(emit_owned_guarded(src, budget)?),
            target: target.clone(),
            policy: policy.clone(),
        }),
        Node::Construct { ctor, args } => Ok(RcNode::Construct {
            ctor: ctor.clone(),
            args: emit_args(args, budget)?,
        }),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let rc_scrutinee = Box::new(emit_owned_guarded(scrutinee, budget)?);
            let mut rc_alts = Vec::with_capacity(alts.len());
            for alt in alts {
                rc_alts.push(emit_alt(alt, budget)?);
            }
            let rc_default = match default {
                Some(d) => Some(Box::new(emit_owned_guarded(d, budget)?)),
                None => None,
            };
            Ok(RcNode::Match {
                scrutinee: rc_scrutinee,
                alts: rc_alts,
                default: rc_default,
            })
        }
        Node::Lam { param, body } => {
            let k = count_occurrences_inner(param, body);
            let rc_body = emit_owned_guarded(body, budget)?;
            Ok(RcNode::Lam {
                param: param.clone(),
                mode: Mode::Owned,
                body: Box::new(balance_binder(param, k, rc_body)),
            })
        }
        Node::App { func, arg } => Ok(RcNode::App {
            func: Box::new(emit_owned_guarded(func, budget)?),
            arg: Box::new(emit_owned_guarded(arg, budget)?),
        }),
        Node::Fix { .. } => Err(EmitError::UnsupportedNode("Fix")),
        Node::FixGroup { .. } => Err(EmitError::UnsupportedNode("FixGroup")),
    }
}

/// Emit each argument of an `Op`/`Construct`, short-circuiting on the first error. A recursion point
/// (RFC-0041 §4.7): each element re-enters [`emit_owned_guarded`], which charges the shared `budget`.
fn emit_args(args: &[Node], budget: &RecursionBudget) -> Result<Vec<RcNode>, EmitError> {
    args.iter().map(|a| emit_owned_guarded(a, budget)).collect()
}

/// Emit one match alternative, balancing each of its (owned) field binders against its uses. A
/// recursion point (RFC-0041 §4.7): its body re-enters [`emit_owned_guarded`], charging `budget`.
fn emit_alt(alt: &Alt, budget: &RecursionBudget) -> Result<RcAlt, EmitError> {
    match alt {
        Alt::Ctor {
            ctor,
            binders,
            body,
        } => {
            let rc_body = emit_owned_guarded(body, budget)?;
            // Each binder is an owned binding scoped to this arm body; balance each against its
            // occurrence count. Nesting order is irrelevant to balance (each binder is independent).
            // `count_occurrences_inner` (not the public wrapper) — see `emit_owned_guarded`'s doc:
            // already on the deep worker stack, so no per-binder thread-spawn is needed.
            let wrapped = binders.iter().fold(rc_body, |acc, b| {
                let k = count_occurrences_inner(b, body);
                balance_binder(b, k, acc)
            });
            Ok(RcAlt::Ctor {
                ctor: ctor.clone(),
                binders: binders.clone(),
                body: wrapped,
            })
        }
        Alt::Lit { value, body } => Ok(RcAlt::Lit {
            value: value.clone(),
            body: emit_owned_guarded(body, budget)?,
        }),
    }
}

/// Place the owned-binding RC annotations at the top of `body`:
/// `k == 0` → one `Drop`; `k >= 1` → `k-1` `Dup`s.
fn balance_binder(var: &mycelium_core::VarId, k: usize, body: RcNode) -> RcNode {
    if k == 0 {
        RcNode::drop_one(var, body)
    } else {
        RcNode::dup_n(var, k - 1, body)
    }
}

/// Count the **free** consuming occurrences of `var` in `node`, respecting shadowing.
///
/// An inner binder that re-binds `var` shadows it: occurrences under that inner scope do **not**
/// count for the outer `var` (rubric A4-01 — analysis across binders must respect shadowing).
/// Total over the whole `Node` grammar (including `Fix`/`FixGroup`) so it is correct even where
/// emission later refuses — counting is never the thing that silently goes wrong.
///
/// **RFC-0041 §4.7 (guard hole RR-29):** this is called once per binder from [`emit_owned_guarded`]
/// (and the elision path), re-walking `body` each time — an `O(N²)` cost over an `N`-deep binder
/// chain. It is also **infallible** (`usize`), so unlike `emit_owned` it cannot refuse past a
/// work-step ceiling without a signature change (which would ripple into the `bool`-returning
/// [`is_fully_borrowable`]/[`is_sole_owned_move`] and the `emit_elided`/`emit_reuse` path — out of
/// this leaf's scope). **W1 minimum (per the guard-hole plan):** wrap the entry point in
/// [`ensure_sufficient_stack`] so the traversal itself never SIGABRTs the host on a deep chain — the
/// `O(N²)` re-walk cost is an explicitly flagged residual for W2/profiling (a work-step
/// [`RecursionBudget::charge_steps`] CPU bound needs the infallible→fallible signature change).
///
/// Guarantee: `Exact` — a deterministic structural count.
//
// FLAG(W7 residual, RFC-0041 §9): the SIGABRT/host-stack hole is CLOSED (this entry point runs the
// traversal on the `mycelium-workstack` deep worker stack, so a deep chain completes rather than
// aborting — see `tests/guard_hole_census.rs::count_occurrences_deep_let_chain`). What is DEFERRED is
// the `O(N²)` re-walk (called once per binder from `emit_*`) and, relatedly, a work-step CPU bound:
// this function is infallible (`usize`), so it cannot refuse past a `RecursionBudget::charge_steps`
// ceiling without an infallible→fallible signature change that would ripple into
// `is_fully_borrowable`/`is_sole_owned_move`/`borrow_occurrences` and the `emit_elided`/`emit_reuse`
// path. That quadratic-scan/work-step fix is a W2/profiling item, not a self-DoS (SIGABRT) risk.
#[must_use]
pub fn count_occurrences(var: &mycelium_core::VarId, node: &Node) -> usize {
    let budget = RecursionBudget::default();
    ensure_sufficient_stack(&budget, || count_occurrences_inner(var, node))
}

/// The recursive core of [`count_occurrences`], run entirely on the deep worker stack the public
/// entry point spawns — so nested calls recurse on the *same* grown stack rather than each
/// re-spawning a worker (which would multiply the thread-spawn cost across every AST node and could
/// exhaust virtual address space on a deep chain).
fn count_occurrences_inner(var: &mycelium_core::VarId, node: &Node) -> usize {
    match node {
        Node::Const(_) => 0,
        Node::Var(x) => usize::from(x == var),
        Node::Let { id, bound, body } => {
            let in_bound = count_occurrences_inner(var, bound);
            // `id` shadows `var` inside `body` only if they are the same name.
            let in_body = if id == var {
                0
            } else {
                count_occurrences_inner(var, body)
            };
            in_bound + in_body
        }
        Node::Op { args, .. } | Node::Construct { args, .. } => {
            args.iter().map(|a| count_occurrences_inner(var, a)).sum()
        }
        Node::Swap { src, .. } => count_occurrences_inner(var, src),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let mut n = count_occurrences_inner(var, scrutinee);
            for alt in alts {
                n += match alt {
                    Alt::Ctor { binders, body, .. } => {
                        if binders.iter().any(|b| b == var) {
                            0 // shadowed by a field binder
                        } else {
                            count_occurrences_inner(var, body)
                        }
                    }
                    Alt::Lit { body, .. } => count_occurrences_inner(var, body),
                };
            }
            n + default
                .as_deref()
                .map_or(0, |d| count_occurrences_inner(var, d))
        }
        Node::Lam { param, body } => {
            if param == var {
                0
            } else {
                count_occurrences_inner(var, body)
            }
        }
        Node::App { func, arg } => {
            count_occurrences_inner(var, func) + count_occurrences_inner(var, arg)
        }
        Node::Fix { name, body } => {
            if name == var {
                0
            } else {
                count_occurrences_inner(var, body)
            }
        }
        Node::FixGroup { defs, body } => {
            if defs.iter().any(|(name, _)| name == var) {
                0 // the group binds all its names everywhere in defs + body
            } else {
                defs.iter()
                    .map(|(_, d)| count_occurrences_inner(var, d))
                    .sum::<usize>()
                    + count_occurrences_inner(var, body)
            }
        }
    }
}

// ── MEM-4 Increment 1 — non-escaping borrow elision ──────────────────────────
//
// In the immutable value model, a reader primitive (`Op`/`Swap`) *reads* its operands and produces
// a fresh result, retaining nothing — so an operand position is a **borrow** (non-consuming read),
// not a move. A `let` binding whose every use is such a borrow is **fully borrowable**: it needs no
// `Dup` (one reference serves all reads) and is reclaimed by a single `DropAfter` once its reads are
// done. This is strictly fewer RC ops than the naive owned emission (`k-1` `Dup`s → `0`), and it is
// **semantics-preserving** (the same value is reclaimed, exactly once) — verified by the differential
// harness in [`crate::eval`].
//
// Increment 1 is intraprocedural and conservative: it elides only **fully-borrowable `let`
// bindings**. A binding with ANY escaping use (the binding flows to the result, into a `Construct`,
// or to an `App`/`Match`) stays fully **owned** (the naive emission). `Lam` parameters stay `Owned`
// (interprocedural borrowing — `Mode::Borrowed` at a call boundary — is a later increment).

use std::collections::HashSet;

use crate::rc_ir::RcNode as N;

/// Lower a Core IR [`Node`] with MEM-4 Increment 1 **borrow elision** applied.
///
/// Identical to [`emit_owned`] except that every **fully-borrowable** `let` binding (every use is a
/// reader-primitive read — [`is_fully_borrowable`]) is emitted with its uses as
/// [`RcNode::Borrow`](crate::rc_ir::RcNode::Borrow), **no** `Dup`, and a single
/// [`RcNode::DropAfter`](crate::rc_ir::RcNode::DropAfter) reclaiming it after its reads.
///
/// Returns [`EmitError::UnsupportedNode`] for `Fix`/`FixGroup` (G2 — never-silent on recursion), and
/// [`EmitError::DepthExceeded`] if the traversal's own recursion exceeds the shared
/// [`RecursionBudget`] depth ceiling (RFC-0041 §4.7 — the W7 guard hole in this AOT RC/ownership
/// pass): the outermost call runs on the deep worker stack so a genuinely deep `Node` never
/// SIGABRTs `myc build`, and each recursive step charges [`RecursionBudget::try_enter`] so a
/// pathological input refuses cleanly at the depth ceiling.
///
/// Guarantee: the elision is **semantics-preserving** — `Empirical` (the differential harness in
/// [`crate::eval`] checks that, for a corpus of terms, the multiset of reclaimed values is identical
/// to the owned emission's, with no use-after-free), backed by the structural `DropAfter`-after-reads
/// argument. The `dup`-count reduction is `Exact` (read off the IR); the *performance* benefit of
/// that reduction stays `Declared` until measured (DN-33 §8.1 Q5).
pub fn emit_elided(node: &Node) -> Result<RcNode, EmitError> {
    // reuse = false: borrow elision only (Increment 1).
    emit_ann_toplevel(node, false)
}

/// Lower a Core IR [`Node`] with MEM-4 Increment 1 (**borrow elision**) **and** Increment 2
/// (**`rc == 1` reuse annotation**) applied.
///
/// A superset of [`emit_elided`]: in addition to borrow-eliding fully-borrowable bindings, a `let`
/// binding that is a **sole-owned single move** ([`is_sole_owned_move`] — used exactly once, in a
/// move position) has that move emitted as [`RcNode::MoveUnique`](crate::rc_ir::RcNode::MoveUnique),
/// recording that the runtime `UniqueOwner` branch is statically guaranteed (FBIP-reuse-eligible).
///
/// Returns [`EmitError::UnsupportedNode`] for `Fix`/`FixGroup` (G2), and [`EmitError::DepthExceeded`]
/// on a deeper-than-ceiling `Node` (RFC-0041 §4.7 — see [`emit_elided`]).
///
/// Guarantee: the reuse annotation is **semantics-preserving** and its soundness is
/// **machine-verified** — [`crate::eval`] errors ([`crate::eval::RcError::UnsoundUnique`]) if any
/// `MoveUnique` is reached at a reference count other than 1. Tag `Empirical` (differential + the
/// verifying evaluator), not `Proven`.
pub fn emit_reuse(node: &Node) -> Result<RcNode, EmitError> {
    // reuse = true: borrow elision + the rc==1 reuse annotation (Increment 2).
    emit_ann_toplevel(node, true)
}

/// Shared guarded entry for the annotated (elision / reuse) emission ([`emit_elided`] /
/// [`emit_reuse`]). Runs the traversal on the `mycelium-workstack` deep worker stack
/// ([`ensure_sufficient_stack`]) and charges a fresh [`RecursionBudget`] at every recursive step
/// (RFC-0041 §4.7 — the W7 guard hole in this AOT RC/ownership pass), so a deep `Node` refuses
/// cleanly ([`EmitError::DepthExceeded`]) instead of SIGABRT-ing the host stack (G2). Mirrors
/// [`emit_owned`]'s W1 guard.
///
/// The per-binder occurrence / borrow analysis routes through the `_inner` helpers (not the public
/// [`count_occurrences`] / [`borrow_occurrences`] wrappers): this closure already runs on the deep
/// worker stack, so re-entering a public wrapper here would needlessly spawn a fresh worker thread
/// **per binder** (turning the already-`O(N²)` re-walk residual into an `O(N)` thread-spawn
/// multiplier too — the same reasoning as `emit_owned_guarded`).
fn emit_ann_toplevel(node: &Node, reuse: bool) -> Result<RcNode, EmitError> {
    let outer = RecursionBudget::default();
    ensure_sufficient_stack(&outer, || {
        let budget = RecursionBudget::default();
        emit_ann(node, &Ann::new(reuse), &budget)
    })
}

/// Annotation context threaded through emission: the in-scope `borrowed` and `unique` variable sets
/// and whether the `rc == 1` reuse annotation (Increment 2) is enabled.
#[derive(Clone)]
struct Ann {
    borrowed: HashSet<VarId>,
    unique: HashSet<VarId>,
    reuse: bool,
}

impl Ann {
    fn new(reuse: bool) -> Self {
        Ann {
            borrowed: HashSet::new(),
            unique: HashSet::new(),
            reuse,
        }
    }

    fn with_borrowed(&self, id: &VarId) -> Ann {
        let mut a = self.clone();
        a.borrowed.insert(id.clone());
        a
    }

    fn with_unique(&self, id: &VarId) -> Ann {
        let mut a = self.clone();
        a.unique.insert(id.clone());
        a
    }
}

/// Emit `node` under the annotation context: every use of a `borrowed` variable becomes a
/// non-consuming [`RcNode::Borrow`]; every use of a `unique` variable becomes a sole-owner
/// [`RcNode::MoveUnique`]; fully-borrowable `let`s are borrow-elided; sole-owned single-move `let`s
/// are reuse-annotated (when `reuse` is enabled).
///
/// **RFC-0041 §4.7 (the W7 guard hole in this AOT RC/ownership pass):** every recursive step charges
/// `budget.try_enter()` (RAII-released on return), so a pathological `Node` refuses cleanly with
/// [`EmitError::DepthExceeded`] at the depth ceiling instead of overflowing the host stack (G2). The
/// per-binder occurrence/borrow analysis routes through the `_inner` helpers ([`count_occurrences_inner`],
/// [`is_fully_borrowable_inner`], [`is_sole_owned_move_inner`], [`borrow_occurrences_inner`]) — **not**
/// the public [`ensure_sufficient_stack`]-wrapping wrappers — because this function already runs on the
/// deep worker stack (via [`emit_ann_toplevel`]), so re-entering a wrapper here would needlessly spawn a
/// fresh worker thread **per binder** (the same reasoning as [`emit_owned_guarded`]).
fn emit_ann(node: &Node, ann: &Ann, budget: &RecursionBudget) -> Result<RcNode, EmitError> {
    let _guard = budget.try_enter()?;
    match node {
        Node::Const(v) => Ok(N::Const(v.clone())),
        Node::Var(x) => Ok(if ann.borrowed.contains(x) {
            N::Borrow(x.clone())
        } else if ann.unique.contains(x) {
            N::MoveUnique(x.clone())
        } else {
            N::Var(x.clone())
        }),
        Node::Let { id, bound, body } => {
            let rc_bound = emit_ann(bound, ann, budget)?;
            if is_fully_borrowable_inner(id, body) {
                // Borrow-elide: read `id` without consuming, reclaim with a single DropAfter.
                let rc_body = emit_ann(body, &ann.with_borrowed(id), budget)?;
                Ok(N::Let {
                    id: id.clone(),
                    bound: Box::new(rc_bound),
                    body: Box::new(N::drop_after(id, rc_body)),
                })
            } else if ann.reuse && is_sole_owned_move_inner(id, body) {
                // Reuse-annotate (Increment 2): `id` is used exactly once, in a move position, so its
                // reference count is statically 1 at that consume → emit it as MoveUnique. k == 1 ⇒
                // no Dup, no Drop (the single move reclaims it).
                let rc_body = emit_ann(body, &ann.with_unique(id), budget)?;
                Ok(N::Let {
                    id: id.clone(),
                    bound: Box::new(rc_bound),
                    body: Box::new(balance_binder(id, 1, rc_body)),
                })
            } else {
                // Owned (naive) emission for this binding.
                let k = count_occurrences_inner(id, body);
                let rc_body = emit_ann(body, ann, budget)?;
                Ok(N::Let {
                    id: id.clone(),
                    bound: Box::new(rc_bound),
                    body: Box::new(balance_binder(id, k, rc_body)),
                })
            }
        }
        Node::Op { prim, args } => Ok(N::Op {
            prim: prim.clone(),
            args: emit_args_a(args, ann, budget)?,
        }),
        Node::Swap {
            src,
            target,
            policy,
        } => Ok(N::Swap {
            src: Box::new(emit_ann(src, ann, budget)?),
            target: target.clone(),
            policy: policy.clone(),
        }),
        Node::Construct { ctor, args } => Ok(N::Construct {
            ctor: ctor.clone(),
            args: emit_args_a(args, ann, budget)?,
        }),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let rc_scrutinee = Box::new(emit_ann(scrutinee, ann, budget)?);
            let mut rc_alts = Vec::with_capacity(alts.len());
            for alt in alts {
                rc_alts.push(emit_alt_a(alt, ann, budget)?);
            }
            let rc_default = match default {
                Some(d) => Some(Box::new(emit_ann(d, ann, budget)?)),
                None => None,
            };
            Ok(N::Match {
                scrutinee: rc_scrutinee,
                alts: rc_alts,
                default: rc_default,
            })
        }
        Node::Lam { param, body } => {
            // Lam params stay Owned (interprocedural borrowing is a later increment).
            let k = count_occurrences_inner(param, body);
            let rc_body = emit_ann(body, ann, budget)?;
            Ok(N::Lam {
                param: param.clone(),
                mode: Mode::Owned,
                body: Box::new(balance_binder(param, k, rc_body)),
            })
        }
        Node::App { func, arg } => Ok(N::App {
            func: Box::new(emit_ann(func, ann, budget)?),
            arg: Box::new(emit_ann(arg, ann, budget)?),
        }),
        Node::Fix { .. } => Err(EmitError::UnsupportedNode("Fix")),
        Node::FixGroup { .. } => Err(EmitError::UnsupportedNode("FixGroup")),
    }
}

/// Emit each argument of an `Op`/`Construct` under the annotation context, short-circuiting on the
/// first error. A recursion point (RFC-0041 §4.7): each element re-enters [`emit_ann`], which charges
/// the shared `budget`.
fn emit_args_a(
    args: &[Node],
    ann: &Ann,
    budget: &RecursionBudget,
) -> Result<Vec<RcNode>, EmitError> {
    args.iter().map(|a| emit_ann(a, ann, budget)).collect()
}

/// Emit one annotated match alternative. A recursion point (RFC-0041 §4.7): its body re-enters
/// [`emit_ann`], charging `budget`; the per-binder counts route through [`count_occurrences_inner`]
/// (already on the deep worker stack — see [`emit_ann`]).
fn emit_alt_a(alt: &Alt, ann: &Ann, budget: &RecursionBudget) -> Result<RcAlt, EmitError> {
    match alt {
        Alt::Ctor {
            ctor,
            binders,
            body,
        } => {
            let rc_body = emit_ann(body, ann, budget)?;
            let wrapped = binders.iter().fold(rc_body, |acc, b| {
                let k = count_occurrences_inner(b, body);
                balance_binder(b, k, acc)
            });
            Ok(RcAlt::Ctor {
                ctor: ctor.clone(),
                binders: binders.clone(),
                body: wrapped,
            })
        }
        Alt::Lit { value, body } => Ok(RcAlt::Lit {
            value: value.clone(),
            body: emit_ann(body, ann, budget)?,
        }),
    }
}

/// Whether `var`'s binding is a **sole-owned single move**: used **exactly once**, and that use is a
/// **move** (not a borrow position). At such a use the reference count is statically 1, so the
/// runtime `UniqueOwner` branch is guaranteed — the `rc == 1` reuse annotation (Increment 2) applies.
///
/// Runs its structural walks on the `mycelium-workstack` deep worker stack ([`ensure_sufficient_stack`])
/// so a deep `body` never SIGABRTs the host (RFC-0041 §4.7); the emission path calls
/// [`is_sole_owned_move_inner`] directly (already on the deep stack) to avoid a per-binder thread spawn.
///
/// Guarantee: `Exact` — a deterministic structural test (conservative: only the unambiguous
/// single-move case; multi-move last-consume is a later refinement).
#[must_use]
pub fn is_sole_owned_move(var: &VarId, body: &Node) -> bool {
    let budget = RecursionBudget::default();
    ensure_sufficient_stack(&budget, || is_sole_owned_move_inner(var, body))
}

/// The deep-stack-agnostic core of [`is_sole_owned_move`] — the bare structural test, run either on
/// the worker stack the public wrapper spawns or (from [`emit_ann`]) on the emission's existing deep
/// stack. Uses the bare [`count_occurrences_inner`]/[`borrow_occurrences_inner`] recursions (no
/// per-call thread spawn).
fn is_sole_owned_move_inner(var: &VarId, body: &Node) -> bool {
    count_occurrences_inner(var, body) == 1 && borrow_occurrences_inner(var, body) == 0
}

/// Whether `var`'s binding is **fully borrowable** over `body`: it is used at least once and **every**
/// use is in a reader-primitive (borrow) position (`Op` argument or `Swap` source). Such a binding
/// never escapes (it does not flow to the result, into a `Construct`, or to an `App`/`Match`), so it
/// can be read without consuming and reclaimed once at the end.
///
/// Runs its structural walks on the deep worker stack ([`ensure_sufficient_stack`]) so a deep `body`
/// never SIGABRTs the host (RFC-0041 §4.7); the emission path calls [`is_fully_borrowable_inner`]
/// directly (already on the deep stack) to avoid a per-binder thread spawn.
///
/// Guarantee: `Exact` — a deterministic structural test (conservative: any escaping use makes it
/// `false`, keeping the binding owned — never wrongly elided).
#[must_use]
pub fn is_fully_borrowable(var: &VarId, body: &Node) -> bool {
    let budget = RecursionBudget::default();
    ensure_sufficient_stack(&budget, || is_fully_borrowable_inner(var, body))
}

/// The deep-stack-agnostic core of [`is_fully_borrowable`] (see [`is_sole_owned_move_inner`]).
fn is_fully_borrowable_inner(var: &VarId, body: &Node) -> bool {
    let total = count_occurrences_inner(var, body);
    total >= 1 && borrow_occurrences_inner(var, body) == total
}

/// Count occurrences of `var` in **borrow positions** (direct `Op` argument / `Swap` source),
/// respecting shadowing. A bare `Var`, a `Construct`/`App`/`Match`/tail occurrence is **not** a
/// borrow position (those are moves/escapes).
///
/// **RFC-0041 §4.7 (guard hole RR-29 sibling):** like [`count_occurrences`], the entry point wraps
/// the traversal in [`ensure_sufficient_stack`] so a deep `node` completes on the deep worker stack
/// rather than SIGABRTing the host. Also infallible (`usize`), so it carries the same explicitly-flagged
/// `O(N²)`/work-step residual (see [`count_occurrences`]).
#[must_use]
pub fn borrow_occurrences(var: &VarId, node: &Node) -> usize {
    let budget = RecursionBudget::default();
    ensure_sufficient_stack(&budget, || borrow_occurrences_inner(var, node))
}

/// The recursive core of [`borrow_occurrences`], run on the deep worker stack the public entry point
/// spawns (or, from the emission/classifier paths, on their existing deep stack) — so nested calls
/// recurse on the *same* grown stack rather than each re-spawning a worker.
fn borrow_occurrences_inner(var: &VarId, node: &Node) -> usize {
    match node {
        Node::Const(_) | Node::Var(_) => 0,
        Node::Let { id, bound, body } => {
            borrow_occurrences_inner(var, bound)
                + if id == var {
                    0
                } else {
                    borrow_occurrences_inner(var, body)
                }
        }
        // Op args and Swap src ARE borrow positions: a direct `Var(var)` child counts; a deeper
        // child is recursed (its own immediate parent decides).
        Node::Op { args, .. } => args.iter().map(|a| arg_borrow(var, a)).sum(),
        Node::Swap { src, .. } => arg_borrow(var, src),
        // Construct args are MOVES (stored into the data value): a direct `Var(var)` is NOT a borrow;
        // only deeper reader positions count → recurse with `borrow_occurrences` (not `arg_borrow`).
        Node::Construct { args, .. } => args.iter().map(|a| borrow_occurrences_inner(var, a)).sum(),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            // The scrutinee is a move (deconstructed), not a borrow → recurse for deeper readers.
            let mut n = borrow_occurrences_inner(var, scrutinee);
            for alt in alts {
                n += match alt {
                    Alt::Ctor { binders, body, .. } => {
                        if binders.iter().any(|b| b == var) {
                            0
                        } else {
                            borrow_occurrences_inner(var, body)
                        }
                    }
                    Alt::Lit { body, .. } => borrow_occurrences_inner(var, body),
                };
            }
            n + default
                .as_deref()
                .map_or(0, |d| borrow_occurrences_inner(var, d))
        }
        Node::Lam { param, body } => {
            if param == var {
                0
            } else {
                borrow_occurrences_inner(var, body)
            }
        }
        Node::App { func, arg } => {
            borrow_occurrences_inner(var, func) + borrow_occurrences_inner(var, arg)
        }
        Node::Fix { name, body } => {
            if name == var {
                0
            } else {
                borrow_occurrences_inner(var, body)
            }
        }
        Node::FixGroup { defs, body } => {
            if defs.iter().any(|(name, _)| name == var) {
                0
            } else {
                defs.iter()
                    .map(|(_, d)| borrow_occurrences_inner(var, d))
                    .sum::<usize>()
                    + borrow_occurrences_inner(var, body)
            }
        }
    }
}

/// One reader-primitive argument: a direct `Var(var)` is a borrow position (count 1); anything else
/// recurses (its own structure decides where `var`'s borrows are).
fn arg_borrow(var: &VarId, arg: &Node) -> usize {
    match arg {
        Node::Var(x) if x == var => 1,
        other => borrow_occurrences_inner(var, other),
    }
}
