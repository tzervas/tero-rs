//! The result sums `Outcome<T, E>` and `Resolution<T, E>` (RFC-0014 §4.1–§4.2).
//!
//! - [`Outcome`] is the input sum: `Ok(T) | Err(E)`.  Errors propagate by default.
//! - [`Resolution`] is the output sum of [`crate::handle`]: `Recovered | Propagated` — there is
//!   **no `Dropped` variant** (I1).  Never-silent is enforced by the type: a handler cannot
//!   express "discard the error" using this sum.

use mycelium_core::GuaranteeStrength;
use mycelium_diag::Diag;

use crate::policy::PolicyRef;

/// The input result sum `Ok(T) | Err(E)` (RFC-0014 §4.1).
///
/// Errors propagate by default — there is no silent unwinding; an `Err` must be explicitly acted
/// on by a recovery policy or it continues to surface.
///
/// This is the **input** type to [`crate::handle`] and `crate::recover`.  The `E` error payload
/// should carry a [`Diag`] diagnostic record per FR-R5.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome<T, E> {
    /// The operation succeeded.
    Ok(T),
    /// The operation failed with an explicit propagating error (never a silent unwind).
    Err(E),
}

impl<T, E> Outcome<T, E> {
    /// Construct from a standard `Result`.
    pub fn from_result(r: Result<T, E>) -> Self {
        match r {
            Result::Ok(v) => Outcome::Ok(v),
            Result::Err(e) => Outcome::Err(e),
        }
    }

    /// Convert to a standard `Result`.
    pub fn into_result(self) -> Result<T, E> {
        match self {
            Outcome::Ok(v) => Result::Ok(v),
            Outcome::Err(e) => Result::Err(e),
        }
    }

    /// Whether this is `Ok`.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Outcome::Ok(_))
    }

    /// Whether this is `Err`.
    #[must_use]
    pub fn is_err(&self) -> bool {
        matches!(self, Outcome::Err(_))
    }
}

/// A diagnosed error value: the error payload `E` bundled with its [`Diag`] record (FR-R5).
///
/// `recover` and `handle` accept a generic `E` but the invariant FR-R5 says that a recovered or
/// re-propagated error carries a `Diag`.  `DiagError<E>` is the canonical bundling.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagError<E> {
    /// The structured error value.
    pub error: E,
    /// The diagnostic record attached to this error (RFC-0013 / FR-R5).
    pub diag: Diag,
}

impl<E> DiagError<E> {
    /// Attach a diagnostic record to an error value.
    pub fn new(error: E, diag: Diag) -> Self {
        DiagError { error, diag }
    }
}

/// The **outcome of handling** an [`Outcome`] under a recovery policy (RFC-0014 §4.2).
///
/// There is **no `Dropped` variant** (I1): a mutant handler that discarded an unmatched error
/// cannot be expressed using this type.
///
/// - [`Resolution::Recovered`]: the error was replaced by a concrete value, tagged honestly
///   (never upgraded — I2/VR-5).
/// - [`Resolution::Propagated`]: the error continues to surface (possibly transformed); it was
///   never discarded.
///
/// Both variants carry the [`PolicyRef`] of the acting policy (if any) so every outcome is
/// EXPLAIN-able (C3).
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution<T, E> {
    /// The error was explicitly recovered: a concrete value with an honest, policy-inherited tag.
    ///
    /// # Guarantee tag (I2/VR-5)
    ///
    /// `tag` is **never upgraded** by the policy: it is at most the action's honest ceiling
    /// (`Declared` for a `Fallback`; the attempt's own tag for a `Retry`; `Exact` for a clean
    /// `Ok` pass-through).  The `meet` of the action ceiling and the policy's declared tag is
    /// taken to ensure the tag never rises (FR-R3).
    Recovered {
        /// The recovered value.
        value: T,
        /// The honest guarantee tag (meet of the action's ceiling and the policy's tag; ≤ policy
        /// tag; never `Exact` unless the recovered value provably justifies it — I2/VR-5).
        tag: GuaranteeStrength,
        /// The content-addressed policy that recovered this value (C3 / EXPLAIN-able).
        policy: Option<PolicyRef>,
    },
    /// The error was re-propagated — it never vanished (I1).
    ///
    /// The error may have been transformed (e.g. class-escalated) but it continues to surface.
    /// A budget overrun for a `CleanupThenPropagate` cleanup effect is recorded in
    /// `cleanup_overrun` rather than swallowed (spec §7-Q4 disposition: record it for legibility).
    Propagated {
        /// The propagating error (possibly transformed/escalated).
        error: E,
        /// The content-addressed policy that acted, if any (C3 / EXPLAIN-able).
        policy: Option<PolicyRef>,
        /// Whether a `cleanup_then_propagate` cleanup effect overran its budget (spec §7-Q4).
        ///
        /// `true` means the cleanup was **skipped** because of an `EffectBudgetExhausted`; the
        /// original error propagates regardless.  `false` means either no cleanup was attempted
        /// or the cleanup ran successfully within its budget.
        cleanup_overrun: bool,
    },
}

impl<T, E> Resolution<T, E> {
    /// Whether the error was recovered (a value is available).
    #[must_use]
    pub fn is_recovered(&self) -> bool {
        matches!(self, Resolution::Recovered { .. })
    }

    /// Whether the error was re-propagated (never dropped — I1).
    #[must_use]
    pub fn is_propagated(&self) -> bool {
        matches!(self, Resolution::Propagated { .. })
    }

    /// The acting `PolicyRef`, if any (C3 / EXPLAIN-able).
    #[must_use]
    pub fn policy_ref(&self) -> Option<&PolicyRef> {
        match self {
            Resolution::Recovered { policy, .. } | Resolution::Propagated { policy, .. } => {
                policy.as_ref()
            }
        }
    }
}
