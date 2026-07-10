//! **Declared, bounded effects** (RFC-0014 §4.5) — re-exported from `mycelium_interp::budget`
//! plus the tooling-layer checker (`check_effects` / `UndeclaredEffect`).
//!
//! The shared budget primitive lives in `mycelium-interp` (RFC-0014 §4.8 — one enforcement
//! mechanism over separate named budgets).  This module re-exports the runtime types unchanged
//! and adds the **declaration checker** (I3 — a tooling concern, not a runtime one; KC-3).
//!
//! # Effect discipline (I3/I4/I5)
//!
//! - **(I3) Effects are declared.** A definition may not perform an effect absent from its
//!   declared set.  [`check_effects`] is the compositional enforcement check.
//! - **(I4) Budgets overrun gracefully.** Every effect is bounded; an overrun is an explicit
//!   [`EffectBudgetExhausted`] — never a hang, stack overflow, or OOM.
//! - **(I5) Tightly scoped by default.** An effect with **no** declared budget cannot run
//!   ([`Budgets::consume`] on an absent budget is an immediate `EffectBudgetExhausted`).

use std::collections::BTreeSet;

// The shared runtime-enforced budget primitive (RFC-0014 §4.8 — now in `mycelium-interp`).
// Re-exported so callers of `std.recover` see one consistent surface.
pub use mycelium_interp::budget::{Budgets, EffectBudget, EffectBudgetExhausted, EffectKind};

/// A definition's **declared** effect set (§4.5 I3) — the set it names on its signature.
pub type EffectSet = BTreeSet<EffectKind>;

/// A performed-but-undeclared effect (I3) — an explicit checker error, never silent.
///
/// A definition may not perform an effect absent from its declared set.  This is a *checker*
/// concern (tooling layer), not a runtime one — it lives here rather than in the shared
/// `mycelium-interp` budget primitive (KC-3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndeclaredEffect {
    /// The effect that was performed but not declared.
    pub effect: EffectKind,
}

impl std::fmt::Display for UndeclaredEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "undeclared effect {}: a definition may not perform an effect absent from its \
             declared signature set (RFC-0014 §4.5 I3 — no unknown side effects); declare it",
            self.effect
        )
    }
}

mycelium_std_core::impl_std_error!(UndeclaredEffect);

/// The **compositional no-undeclared-effect check** (I3).
///
/// Every effect a definition *performs* (its own plus its callees', composed up the call graph)
/// must be in its *declared* set.  Returns the first undeclared effect found, if any.
///
/// This *checks* that declared effects compose — it never *infers* one.  An effect can never
/// become implicit; the discipline is: **manual-declare + compositional-check** (§8 resolved).
///
/// # Errors
///
/// Returns [`UndeclaredEffect`] for the first effect in `performed` that is absent from
/// `declared`.
pub fn check_effects(declared: &EffectSet, performed: &EffectSet) -> Result<(), UndeclaredEffect> {
    for e in performed {
        if !declared.contains(e) {
            return Err(UndeclaredEffect { effect: e.clone() });
        }
    }
    Ok(())
}
