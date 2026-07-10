//! The structural balance invariant — MEM-4·B0 / DN-33 §8.1 Q3 (structural-invariant half).
//!
//! This **independently** verifies that an [`RcNode`] is reference-balanced: it re-walks the
//! *emitted* IR (not the source `Node`) and, for every owned binding, checks
//!
//! ```text
//! 1 (the binding's initial reference) + #Dup(x)  ==  #consuming-uses(x) + #Drop(x)
//! ```
//!
//! within that binding's scope. Because it re-derives the counts from the final IR rather than
//! trusting the emission's own bookkeeping, it is a genuine check: a buggy emission (an off-by-one
//! `Dup`, a missing `Drop`) makes the invariant fail (mutation-tested in the suite).
//!
//! This is the **structural-invariant** half of the ratified Q3 soundness strategy (DN-33 §8.1);
//! the **differential** half (run with/without elision, compare reclamation traces) lands with
//! MEM-4 Increment 1, where there are two emissions to compare.
//!
//! # Mode-awareness (forward-compatible with Increment 1)
//!
//! - An **`Owned`** binding must satisfy the balance equation above.
//! - A **`Borrowed`** binding (Increment 1 output) must carry **no** `Dup`/`Drop` of its variable —
//!   a borrowed value is read without consuming, so it is neither dup'd nor dropped. Its uses are
//!   non-consuming reads and impose no balance obligation.
//!
//! (B0 emits only `Owned`; the `Borrowed` clause is here so the same checker validates Increment 1.)
//!
//! Guarantee: `Exact` — the tallies are deterministic structural counts; the invariant is an
//! exact equation, not an approximation.

use mycelium_core::VarId;

use crate::rc_ir::{Mode, RcAlt, RcNode};

/// A balance-invariant violation for one binding (never-silent diagnostics, G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalanceError {
    /// An owned binding's references do not balance: `1 + dups != uses + drops`.
    OwnedUnbalanced {
        /// The binding variable.
        var: VarId,
        /// `Dup` count of `var` in scope.
        dups: usize,
        /// Consuming-use count of `var` in scope.
        uses: usize,
        /// `Drop` count of `var` in scope.
        drops: usize,
    },
    /// A borrowed binding carries a `Dup`/`Drop` of its variable (it must carry neither).
    BorrowedHasRcOps {
        /// The binding variable.
        var: VarId,
        /// `Dup` count (must be 0 for a borrowed binding).
        dups: usize,
        /// `Drop` count (must be 0 for a borrowed binding).
        drops: usize,
    },
}

impl std::fmt::Display for BalanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BalanceError::OwnedUnbalanced {
                var,
                dups,
                uses,
                drops,
            } => write!(
                f,
                "owned binding `{var}` is RC-unbalanced: 1 + dups({dups}) != uses({uses}) + \
                 drops({drops})"
            ),
            BalanceError::BorrowedHasRcOps { var, dups, drops } => write!(
                f,
                "borrowed binding `{var}` must carry no Dup/Drop, but has dups({dups}) + \
                 drops({drops})"
            ),
        }
    }
}

impl std::error::Error for BalanceError {}

/// The (dups, drops, uses) tally of a variable within a scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Tally {
    dups: usize,
    drops: usize,
    uses: usize,
}

impl Tally {
    fn add(self, other: Tally) -> Tally {
        Tally {
            dups: self.dups + other.dups,
            drops: self.drops + other.drops,
            uses: self.uses + other.uses,
        }
    }
}

/// Verify the balance invariant over `node` and every binding nested within it.
///
/// Returns the first violation found (never-silent — the bad binding is named), or `Ok(())` if
/// every binding balances.
///
/// Guarantee: `Exact` — a deterministic pass over the IR.
pub fn check_balance(node: &RcNode) -> Result<(), BalanceError> {
    match node {
        RcNode::Const(_) | RcNode::Var(_) | RcNode::Borrow(_) | RcNode::MoveUnique(_) => Ok(()),
        RcNode::Dup { body, .. } | RcNode::Drop { body, .. } | RcNode::DropAfter { body, .. } => {
            check_balance(body)
        }
        RcNode::Let { id, bound, body } => {
            check_balance(bound)?;
            check_owned(id, body)?;
            check_balance(body)
        }
        RcNode::Op { args, .. } | RcNode::Construct { args, .. } => {
            for a in args {
                check_balance(a)?;
            }
            Ok(())
        }
        RcNode::Swap { src, .. } => check_balance(src),
        RcNode::Match {
            scrutinee,
            alts,
            default,
        } => {
            check_balance(scrutinee)?;
            for alt in alts {
                match alt {
                    RcAlt::Ctor { binders, body, .. } => {
                        for b in binders {
                            check_owned(b, body)?;
                        }
                        check_balance(body)?;
                    }
                    RcAlt::Lit { body, .. } => check_balance(body)?,
                }
            }
            if let Some(d) = default {
                check_balance(d)?;
            }
            Ok(())
        }
        RcNode::Lam { param, mode, body } => {
            let t = tally(param, body);
            match mode {
                Mode::Owned => check_owned_tally(param, t)?,
                Mode::Borrowed => {
                    if t.dups != 0 || t.drops != 0 {
                        return Err(BalanceError::BorrowedHasRcOps {
                            var: param.clone(),
                            dups: t.dups,
                            drops: t.drops,
                        });
                    }
                }
            }
            check_balance(body)
        }
        RcNode::App { func, arg } => {
            check_balance(func)?;
            check_balance(arg)
        }
    }
}

/// Check the owned-balance equation for binding `var` over `body`.
fn check_owned(var: &VarId, body: &RcNode) -> Result<(), BalanceError> {
    check_owned_tally(var, tally(var, body))
}

fn check_owned_tally(var: &VarId, t: Tally) -> Result<(), BalanceError> {
    // 1 (initial reference) + dups == uses + drops.
    if 1 + t.dups == t.uses + t.drops {
        Ok(())
    } else {
        Err(BalanceError::OwnedUnbalanced {
            var: var.clone(),
            dups: t.dups,
            uses: t.uses,
            drops: t.drops,
        })
    }
}

/// Tally `var`'s `Dup`/`Drop`/use counts within `body`, respecting shadowing (an inner binder of
/// the same name stops the tally for that sub-scope).
fn tally(var: &VarId, body: &RcNode) -> Tally {
    match body {
        RcNode::Const(_) => Tally::default(),
        // A `MoveUnique` is a consuming move, exactly like `Var` for the balance equation.
        RcNode::Var(x) | RcNode::MoveUnique(x) => Tally {
            uses: usize::from(x == var),
            ..Tally::default()
        },
        // A borrow is a non-consuming read: it changes no reference count, so it contributes
        // nothing to the balance equation (its liveness — that a reference is live when it reads —
        // is the reference evaluator's obligation, not the structural balance's).
        RcNode::Borrow(_) => Tally::default(),
        RcNode::Dup { var: v, body } => {
            let base = tally(var, body);
            if v == var {
                Tally {
                    dups: base.dups + 1,
                    ..base
                }
            } else {
                base
            }
        }
        RcNode::Drop { var: v, body } | RcNode::DropAfter { var: v, body } => {
            let base = tally(var, body);
            if v == var {
                Tally {
                    drops: base.drops + 1,
                    ..base
                }
            } else {
                base
            }
        }
        RcNode::Let { id, bound, body } => {
            // `bound` is in the outer scope; `body` only if `id` does not shadow `var`.
            let in_bound = tally(var, bound);
            if id == var {
                in_bound
            } else {
                in_bound.add(tally(var, body))
            }
        }
        RcNode::Op { args, .. } | RcNode::Construct { args, .. } => args
            .iter()
            .fold(Tally::default(), |acc, a| acc.add(tally(var, a))),
        RcNode::Swap { src, .. } => tally(var, src),
        RcNode::Match {
            scrutinee,
            alts,
            default,
        } => {
            let mut acc = tally(var, scrutinee);
            for alt in alts {
                acc = acc.add(match alt {
                    RcAlt::Ctor { binders, body, .. } => {
                        if binders.iter().any(|b| b == var) {
                            Tally::default() // shadowed
                        } else {
                            tally(var, body)
                        }
                    }
                    RcAlt::Lit { body, .. } => tally(var, body),
                });
            }
            if let Some(d) = default {
                acc = acc.add(tally(var, d));
            }
            acc
        }
        RcNode::Lam { param, body, .. } => {
            if param == var {
                Tally::default() // shadowed
            } else {
                tally(var, body)
            }
        }
        RcNode::App { func, arg } => tally(var, func).add(tally(var, arg)),
    }
}
