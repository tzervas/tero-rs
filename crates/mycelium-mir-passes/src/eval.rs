//! Reference RC-evaluator + differential harness — MEM-4 / DN-33 §8.1 Q3 (differential half).
//!
//! An **abstract reference-counting machine** over the RC-annotated IR's straight-line fragment
//! (`Const/Let/Op/Swap/Var/Borrow/Dup/Drop/DropAfter`). It does **not** compute values — it tracks
//! *references* and *reclamations* — and so serves as the executable semantics against which the
//! borrow elision is checked: [`differential`] runs a term's owned emission and its borrow-elided
//! emission and asserts they reclaim **the same multiset of values** (semantics-preserving) with
//! **no use-after-free**.
//!
//! # Accounting model (closed program)
//!
//! Each allocation (`Const`, and the fresh result of an `Op`/`Swap`) gets a distinct [`AllocId`]
//! with reference count 1. `Dup` increments; `Drop`/`DropAfter` and a consuming `Var` **move**
//! decrement (reclaiming at zero, logged in order); a `Borrow` is a non-consuming **read** that
//! asserts the value is still live. Reclamation order/identity is deterministic (allocation order is
//! fixed by the term skeleton, which is identical for the owned and elided emissions), so the two
//! reclamation logs are directly comparable.
//!
//! # Honesty (VR-5)
//!
//! This is an **abstract** machine (references + reclamation, not data): the consuming/reader
//! distinction is modelled, but operand *content* is not, so it checks the RC discipline, not value
//! correctness. Control-flow nodes (`App/Match/Construct/Lam`) and recursion are **out of the
//! straight-line fragment** and return an explicit [`RcError::UnsupportedNode`] (G2 — never-silent);
//! the differential corpus is straight-line. The elision's semantics-preservation is therefore
//! `Empirical` (differential trials over a corpus), not `Proven`.

use std::collections::HashMap;

use mycelium_core::VarId;
use mycelium_workstack::{ensure_sufficient_stack, BudgetError, RecursionBudget};

use crate::rc_ir::RcNode;

/// A distinct allocation identity (assigned in evaluation order).
pub type AllocId = u64;

/// An RC-discipline violation detected by the evaluator (never-silent, G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RcError {
    /// A variable was used with no binding in scope.
    UnboundVar(VarId),
    /// A value was read (`Borrow`) or moved after it had already been reclaimed (rc was 0).
    UseAfterFree(VarId),
    /// A reference count was decremented below zero (over-release / double free).
    DoubleFree(VarId),
    /// A `MoveUnique` annotation (Increment 2) was reached where the reference count was **not** 1 —
    /// i.e. the static "sole owner" claim is false. This catches an unsound reuse annotation.
    UnsoundUnique {
        /// The annotated variable.
        var: VarId,
        /// The actual reference count found (≠ 1).
        found: i64,
    },
    /// A node outside the straight-line fragment (e.g. `App`/`Match`/`Construct`/`Lam`).
    UnsupportedNode(&'static str),
    /// The evaluator's own [`RcNode`]-traversal recursion (over `Let`'s `bound`/`body`, `Op`
    /// arguments, `Dup`/`Drop`/`DropAfter` wrappers, …) exceeded the shared
    /// [`mycelium_workstack::RecursionBudget`] depth ceiling (RFC-0041 §4.7 — the W7 guard hole in
    /// this AOT RC/ownership pass). Refused cleanly here rather than overflowing the host stack and
    /// SIGABRT-ing `myc build` (G2); reconciles with the `emit`-side [`crate::emit::EmitError::DepthExceeded`]
    /// and the interp/AOT `DepthLimit` family.
    DepthExceeded {
        /// The depth ceiling (recursion frames) that was reached.
        limit: u32,
    },
}

impl std::fmt::Display for RcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RcError::UnboundVar(v) => write!(f, "unbound variable `{v}`"),
            RcError::UseAfterFree(v) => write!(f, "use-after-free: `{v}` read/moved after reclaim"),
            RcError::DoubleFree(v) => {
                write!(f, "double free: reference count of `{v}` went negative")
            }
            RcError::UnsoundUnique { var, found } => write!(
                f,
                "unsound rc==1 annotation: `{var}` was marked sole-owner (MoveUnique) but its \
                 reference count was {found}, not 1"
            ),
            RcError::UnsupportedNode(k) => {
                write!(
                    f,
                    "RC-evaluator does not support `{k}` (straight-line fragment only)"
                )
            }
            RcError::DepthExceeded { limit } => write!(
                f,
                "RC-evaluator's own IR-traversal recursion exceeded the depth budget (limit \
                 {limit} frames) — refusing rather than overflowing the host stack (RFC-0041 §4.7, G2)"
            ),
        }
    }
}

impl std::error::Error for RcError {}

impl From<BudgetError> for RcError {
    /// [`eval`] only ever charges the depth guard ([`RecursionBudget::try_enter`]), so in practice
    /// this always sees [`BudgetError::DepthExceeded`]; the `OutOfBudget` arm is handled defensively
    /// (never a silent drop of the ceiling, G2) should a future increment add byte/step charging.
    fn from(err: BudgetError) -> Self {
        match err {
            BudgetError::DepthExceeded { limit } => RcError::DepthExceeded { limit },
            BudgetError::OutOfBudget { limit, .. } => RcError::DepthExceeded {
                limit: u32::try_from(limit).unwrap_or(u32::MAX),
            },
        }
    }
}

/// The outcome of evaluating a term: its result allocation and the reclamation log (in order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalReport {
    /// The allocation yielded as the term's result (it escapes — not necessarily reclaimed).
    pub result: AllocId,
    /// The allocations reclaimed (reference count reached zero), in reclamation order.
    pub reclaimed: Vec<AllocId>,
}

impl EvalReport {
    /// The reclamation log as a sorted multiset — the comparison key for [`differential`]
    /// (order-independent: elision may change *when* a value is reclaimed, not *whether*).
    #[must_use]
    pub fn reclaimed_sorted(&self) -> Vec<AllocId> {
        let mut v = self.reclaimed.clone();
        v.sort_unstable();
        v
    }
}

struct Machine {
    next: AllocId,
    rc: HashMap<AllocId, i64>,
    reclaimed: Vec<AllocId>,
}

impl Machine {
    fn new() -> Self {
        Machine {
            next: 0,
            rc: HashMap::new(),
            reclaimed: Vec::new(),
        }
    }

    fn alloc(&mut self) -> AllocId {
        let id = self.next;
        self.next += 1;
        self.rc.insert(id, 1);
        id
    }

    fn dup(&mut self, a: AllocId) {
        *self.rc.entry(a).or_insert(0) += 1;
    }

    /// Decrement; reclaim at zero, error below zero.
    fn dec(&mut self, a: AllocId, var: &VarId) -> Result<(), RcError> {
        let r = self.rc.entry(a).or_insert(0);
        *r -= 1;
        if *r == 0 {
            self.reclaimed.push(a);
            Ok(())
        } else if *r < 0 {
            Err(RcError::DoubleFree(var.clone()))
        } else {
            Ok(())
        }
    }

    fn assert_live(&self, a: AllocId, var: &VarId) -> Result<(), RcError> {
        if self.rc.get(&a).copied().unwrap_or(0) > 0 {
            Ok(())
        } else {
            Err(RcError::UseAfterFree(var.clone()))
        }
    }
}

/// Evaluate an [`RcNode`] in the abstract RC machine, returning its reclamation report.
///
/// Errors (never-silent) on an unbound variable, a use-after-free, a double-free, or a node outside
/// the straight-line fragment — and, per RFC-0041 §4.7 (the W7 guard hole in this AOT RC/ownership
/// pass), on [`RcError::DepthExceeded`] if the traversal's own recursion exceeds the shared
/// [`RecursionBudget`] depth ceiling. The outermost call runs on the deep worker stack
/// ([`ensure_sufficient_stack`]) so a genuinely deep input never SIGABRTs `myc build`, and every
/// recursive step charges [`RecursionBudget::try_enter`] so a pathological input refuses cleanly at
/// the depth ceiling instead of exhausting even the deep worker stack.
pub fn eval(node: &RcNode) -> Result<EvalReport, RcError> {
    // The outer budget is not consulted for sizing (the deep worker stack is generous); the real
    // depth guard is the budget created and charged inside the closure, entirely on the worker
    // thread (mirrors `emit::emit_owned` and `mycelium-workstack`'s own precedent: `RecursionBudget`
    // is `Send` but not `Sync`, so it is owned inside `f`, not borrowed across the thread boundary).
    let outer = RecursionBudget::default();
    ensure_sufficient_stack(&outer, || {
        let budget = RecursionBudget::default();
        let mut m = Machine::new();
        let env = HashMap::new();
        let result = go(node, &env, &mut m, &budget)?;
        Ok(EvalReport {
            result,
            reclaimed: m.reclaimed,
        })
    })
}

fn lookup(env: &HashMap<VarId, AllocId>, x: &VarId) -> Result<AllocId, RcError> {
    env.get(x)
        .copied()
        .ok_or_else(|| RcError::UnboundVar(x.clone()))
}

/// The guarded recursive core of [`eval`]: identical accounting, but every recursive step charges
/// `budget.try_enter()` (RAII-released on return) so the depth ceiling — not a host-stack overflow —
/// always bounds a pathological input (RFC-0041 §4.7).
fn go(
    node: &RcNode,
    env: &HashMap<VarId, AllocId>,
    m: &mut Machine,
    budget: &RecursionBudget,
) -> Result<AllocId, RcError> {
    let _guard = budget.try_enter()?;
    match node {
        RcNode::Const(_) => Ok(m.alloc()),
        RcNode::Var(x) => {
            // Move: consume one reference of x.
            let a = lookup(env, x)?;
            m.dec(a, x)?;
            Ok(a)
        }
        RcNode::Borrow(x) => {
            // Read: the value must be live; reference count unchanged.
            let a = lookup(env, x)?;
            m.assert_live(a, x)?;
            Ok(a)
        }
        RcNode::MoveUnique(x) => {
            // Verify the Increment-2 static claim: the reference count MUST be exactly 1 here.
            let a = lookup(env, x)?;
            let rc = m.rc.get(&a).copied().unwrap_or(0);
            if rc != 1 {
                return Err(RcError::UnsoundUnique {
                    var: x.clone(),
                    found: rc,
                });
            }
            // Then it consumes (the unique owner → reclaim), exactly like a Var move.
            m.dec(a, x)?;
            Ok(a)
        }
        RcNode::Dup { var, body } => {
            let a = lookup(env, var)?;
            m.dup(a);
            go(body, env, m, budget)
        }
        RcNode::Drop { var, body } => {
            let a = lookup(env, var)?;
            m.dec(a, var)?;
            go(body, env, m, budget)
        }
        RcNode::DropAfter { var, body } => {
            // Evaluate the body (its reads of `var` happen here), THEN reclaim `var`.
            let r = go(body, env, m, budget)?;
            let a = lookup(env, var)?;
            m.dec(a, var)?;
            Ok(r)
        }
        RcNode::Let { id, bound, body } => {
            let a = go(bound, env, m, budget)?;
            let mut e2 = env.clone();
            e2.insert(id.clone(), a);
            go(body, &e2, m, budget)
        }
        RcNode::Op { args, .. } => {
            for arg in args {
                go(arg, env, m, budget)?;
            }
            Ok(m.alloc()) // the primitive produces a fresh result
        }
        RcNode::Swap { src, .. } => {
            go(src, env, m, budget)?;
            Ok(m.alloc())
        }
        RcNode::Construct { .. } => Err(RcError::UnsupportedNode("Construct")),
        RcNode::Match { .. } => Err(RcError::UnsupportedNode("Match")),
        RcNode::Lam { .. } => Err(RcError::UnsupportedNode("Lam")),
        RcNode::App { .. } => Err(RcError::UnsupportedNode("App")),
    }
}

/// The verdict of a differential run on one term (DN-33 §8.1 Q3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Differential {
    /// Whether the owned and elided emissions reclaim the **same multiset** of values.
    pub same_reclamations: bool,
    /// `Dup` count of the owned emission.
    pub owned_dups: usize,
    /// `Dup` count of the elided emission.
    pub elided_dups: usize,
}

impl Differential {
    /// The elision is **semantics-preserving** iff the reclamation multisets match.
    #[must_use]
    pub fn is_semantics_preserving(&self) -> bool {
        self.same_reclamations
    }

    /// `Dup`s removed by elision (≥ 0; the optimisation's effect — `Exact`, read off the IR).
    #[must_use]
    pub fn dups_removed(&self) -> usize {
        self.owned_dups.saturating_sub(self.elided_dups)
    }
}

/// Count `Dup` nodes anywhere in an [`RcNode`].
///
/// **RFC-0041 §4.7 (guard hole RR-29 sibling):** the entry point runs the traversal on the
/// `mycelium-workstack` deep worker stack ([`ensure_sufficient_stack`]) so a deep `RcNode` — even one
/// built directly by an external caller, outside the [`differential`] path — completes rather than
/// SIGABRTing the host (the "no input SIGABRTs any public pass" §9-DoD close).
//
// FLAG(W7 residual, RFC-0041 §9): the SIGABRT/host-stack hole is now CLOSED (deep-stack wrap above,
// like `emit::count_occurrences`/`borrow_occurrences`). What stays DEFERRED to W2/profiling is the
// `O(N²)`/work-step CPU bound: this and [`count_move_unique`] are infallible (`usize`), so they cannot
// refuse past a `RecursionBudget::charge_steps` ceiling without an infallible→fallible signature change
// (out of this leaf's scope). Not a self-DoS in any current caller.
#[must_use]
pub fn count_dups(node: &RcNode) -> usize {
    let budget = RecursionBudget::default();
    ensure_sufficient_stack(&budget, || count_dups_inner(node))
}

/// The recursive core of [`count_dups`], run on the deep worker stack the public entry point spawns
/// — so nested calls recurse on the *same* grown stack rather than each re-spawning a worker.
fn count_dups_inner(node: &RcNode) -> usize {
    match node {
        RcNode::Const(_) | RcNode::Var(_) | RcNode::Borrow(_) | RcNode::MoveUnique(_) => 0,
        RcNode::Dup { body, .. } => 1 + count_dups_inner(body),
        RcNode::Drop { body, .. } | RcNode::DropAfter { body, .. } => count_dups_inner(body),
        RcNode::Let { bound, body, .. } => count_dups_inner(bound) + count_dups_inner(body),
        RcNode::Op { args, .. } | RcNode::Construct { args, .. } => {
            args.iter().map(count_dups_inner).sum()
        }
        RcNode::Swap { src, .. } => count_dups_inner(src),
        RcNode::Match {
            scrutinee,
            alts,
            default,
        } => {
            count_dups_inner(scrutinee)
                + alts
                    .iter()
                    .map(|a| match a {
                        crate::rc_ir::RcAlt::Ctor { body, .. }
                        | crate::rc_ir::RcAlt::Lit { body, .. } => count_dups_inner(body),
                    })
                    .sum::<usize>()
                + default.as_deref().map_or(0, count_dups_inner)
        }
        RcNode::Lam { body, .. } => count_dups_inner(body),
        RcNode::App { func, arg } => count_dups_inner(func) + count_dups_inner(arg),
    }
}

/// Count [`RcNode::MoveUnique`] annotations (Increment 2 `rc == 1` reuse sites) anywhere in an
/// [`RcNode`] — the FBIP-reuse-eligible consume points. `Exact` (read off the IR).
///
/// **RFC-0041 §4.7:** deep-stack-wrapped like [`count_dups`] (see its note) — a deep external input
/// completes on the worker stack rather than SIGABRTing; the `O(N²)`/work-step bound stays a deferred
/// W2 residual.
#[must_use]
pub fn count_move_unique(node: &RcNode) -> usize {
    let budget = RecursionBudget::default();
    ensure_sufficient_stack(&budget, || count_move_unique_inner(node))
}

/// The recursive core of [`count_move_unique`] (see [`count_dups_inner`]).
fn count_move_unique_inner(node: &RcNode) -> usize {
    match node {
        RcNode::Const(_) | RcNode::Var(_) | RcNode::Borrow(_) => 0,
        RcNode::MoveUnique(_) => 1,
        RcNode::Dup { body, .. } | RcNode::Drop { body, .. } | RcNode::DropAfter { body, .. } => {
            count_move_unique_inner(body)
        }
        RcNode::Let { bound, body, .. } => {
            count_move_unique_inner(bound) + count_move_unique_inner(body)
        }
        RcNode::Op { args, .. } | RcNode::Construct { args, .. } => {
            args.iter().map(count_move_unique_inner).sum()
        }
        RcNode::Swap { src, .. } => count_move_unique_inner(src),
        RcNode::Match {
            scrutinee,
            alts,
            default,
        } => {
            count_move_unique_inner(scrutinee)
                + alts
                    .iter()
                    .map(|a| match a {
                        crate::rc_ir::RcAlt::Ctor { body, .. }
                        | crate::rc_ir::RcAlt::Lit { body, .. } => count_move_unique_inner(body),
                    })
                    .sum::<usize>()
                + default.as_deref().map_or(0, count_move_unique_inner)
        }
        RcNode::Lam { body, .. } => count_move_unique_inner(body),
        RcNode::App { func, arg } => count_move_unique_inner(func) + count_move_unique_inner(arg),
    }
}

/// Run the differential check: evaluate the owned and elided emissions of the **same** Core IR term
/// and compare. Returns the [`Differential`] verdict, or an [`RcError`] if either emission
/// use-after-frees / double-frees (which would itself be a soundness failure of the elision).
///
/// The two emissions are supplied as already-lowered [`RcNode`]s so this function stays independent
/// of `crate::emit` (the caller pairs `emit_owned(t)` with `emit_elided(t)`).
pub fn differential(owned: &RcNode, elided: &RcNode) -> Result<Differential, RcError> {
    let owned_report = eval(owned)?;
    let elided_report = eval(elided)?;
    Ok(Differential {
        same_reclamations: owned_report.reclaimed_sorted() == elided_report.reclaimed_sorted(),
        owned_dups: count_dups(owned),
        elided_dups: count_dups(elided),
    })
}
