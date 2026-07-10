//! **Declared, bounded effects** (RFC-0014 §4.5) — the safety discipline: effects are *declared* on a
//! signature (no unknown side effects, I3) and any effect that could be unbounded carries an explicit
//! *budget* whose overrun is an **explicit, graceful** [`EffectBudgetExhausted`] (I4) — never a hang, a
//! stack overflow, or an OOM. This is the direct generalisation of the `Fix`/`FixGroup` fuel clock
//! (RFC-0007 §4.5), the M-347 depth ceiling, and DN-05 budgets: **separate named budgets, one
//! enforcement mechanism** (§8 resolved). The default scope is the narrowest — an effect with no budget
//! set cannot run (you opt into a broader effect by *declaring its budget*, I5).
//!
//! ## The shared budget primitive lives in `mycelium-interp` (RFC-0014 §4.8, completed)
//! The ledger primitive — [`EffectKind`], [`EffectBudget`], [`EffectBudgetExhausted`], [`Budgets`] —
//! now lives in [`mycelium_interp::budget`], the common ancestor both this recovery subsystem
//! (`mycelium-lsp`) and the AOT env-machine (`mycelium-mlir`) depend on. A budget overrun routes
//! through [`mycelium_interp::EvalError::EffectBudget`], so the ledger is no longer tooling-only: it is
//! the **same** type the env-machine threads, on the **same** runtime refusal channel as
//! `FuelExhausted`/`DepthLimit`. This module re-exports it (RFC-0014's enacted API is preserved) and
//! keeps the *checker* half — the [`EffectSet`] / [`UndeclaredEffect`] / [`check_effects`]
//! no-undeclared-effect check (I3) — in the tooling layer (it is not a runtime concern; KC-3).

use std::collections::BTreeSet;
use std::fmt;

// The shared, runtime-enforced budget primitive (RFC-0014 §4.8 — now in `mycelium-interp`, where the
// fuel/depth clocks live). Re-exported so the recovery subsystem's enacted surface is unchanged.
pub use mycelium_interp::budget::{Budgets, EffectBudget, EffectBudgetExhausted, EffectKind};

/// A definition's **declared** effect set (§4.5 I3) — what it says it can do, on its signature.
pub type EffectSet = BTreeSet<EffectKind>;

/// An effect a definition performs but did **not** declare (I3) — an explicit checker error, never
/// silent. This is the "no unknown side effects" guarantee. A *checker* concern (tooling layer), not a
/// runtime one, so it stays here rather than in the shared `mycelium-interp` budget primitive (KC-3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndeclaredEffect {
    /// The performed-but-undeclared effect.
    pub effect: EffectKind,
}

impl fmt::Display for UndeclaredEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "undeclared effect {:?}: a definition may not perform an effect absent from its signature \
             (RFC-0014 §4.5 I3 — no unknown side effects); declare it",
            self.effect.to_string()
        )
    }
}

impl std::error::Error for UndeclaredEffect {}

/// The **compositional no-undeclared-effect check** (I3): every effect a definition *performs* (its own
/// plus its callees', composed up the call graph) must be in its *declared* set. Returns the first
/// undeclared effect, if any. This *checks* declared effects compose — it never *infers* one (an effect
/// can never become implicit; §8 resolved: manual-declare + compositional-check).
///
/// # Errors
/// Returns [`UndeclaredEffect`] for the first performed effect not in `declared`.
pub fn check_effects(declared: &EffectSet, performed: &EffectSet) -> Result<(), UndeclaredEffect> {
    for e in performed {
        if !declared.contains(e) {
            return Err(UndeclaredEffect { effect: e.clone() });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ledger_primitive_is_the_shared_interp_type() {
        // The recovery ledger and the env-machine consume the *same* `Budgets`: a `mycelium_interp`
        // re-export, so an overrun is a runtime `EvalError::EffectBudget` (RFC-0014 §4.8, completed).
        let mut b = Budgets::new().with(EffectBudget::Attempts(1));
        assert!(b.consume(EffectKind::Retry, 1).is_ok());
        let exhausted = b.consume(EffectKind::Retry, 1).unwrap_err();
        let as_eval: mycelium_interp::EvalError = exhausted.into();
        assert!(matches!(
            as_eval,
            mycelium_interp::EvalError::EffectBudget(_)
        ));
    }

    #[test]
    fn an_undeclared_effect_is_an_explicit_checker_error() {
        let declared: EffectSet = [EffectKind::Alloc].into_iter().collect();
        let performed: EffectSet = [EffectKind::Alloc, EffectKind::Io].into_iter().collect();
        let err = check_effects(&declared, &performed).unwrap_err();
        assert_eq!(err.effect, EffectKind::Io);
    }
}
