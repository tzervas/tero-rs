//! **Shared mode-parametric test harness** (M-795; RFC-0034 §13; DN-20).
//!
//! Provides the reusable building blocks every mode-sensitive test in this crate (and, via
//! re-export from this in-crate tests module, potentially adapted crates) uses:
//!
//! 1. **Canonical bound fixtures** — one well-formed, invariant-consistent `Bound` for each of the
//!    four [`GuaranteeStrength`] variants. Used wherever a test needs a valid `(guarantee, bound)`
//!    pre-image for [`CertMode::gate_result`] or [`Meta::new`].
//!
//! 2. **`for_each_mode`** — a thin wrapper that iterates [`CertMode::ALL`] and calls a closure
//!    with each mode, providing a uniform "mode-parametric sweep" without repeating the iterator
//!    in every test body.
//!
//! 3. **`ModeScope`** — a typed predicate set that describes which modes satisfy a property; used
//!    by `assert_mode_scope` to check that an invariant fires in the modes it applies to AND is
//!    correctly absent in the modes it does not. This is the **cross-mode negative** pattern made
//!    first-class: every mode-sensitive assertion declares its expected scope explicitly, preventing
//!    the silent "invariant holds where it should not" defect.
//!
//! 4. **`assert_mode_scope`** — the primary assertion helper: given a `ModeScope` and a predicate,
//!    asserts `predicate(mode) == true` for each mode in scope and `== false` for each mode outside
//!    scope, with a clear panic message on either direction of failure.
//!
//! ## Harness API decision (FLAG for maintainer — M-795)
//!
//! This module is **`#[cfg(test)]`-only**, declared as `mod mode_harness` inside
//! `src/tests/mod.rs`. It is *not* exported as a public item of `mycelium-core`. This is the
//! smallest defensible choice (KC-3 / YAGNI): the harness is sufficient for the in-crate test
//! suite and avoids enlarging the public API surface.
//!
//! For crates that want to reuse these fixtures (e.g. `mycelium-cert`), the current options are:
//! (a) duplicate the few fixture fns locally (minimal duplication — the code is small),
//! (b) introduce a `mycelium-test-support` dev-only crate (the "correct" solution if many crates
//! need it; escalate as a follow-on task rather than prematurely publishing a support crate).
//! The M-794 consolidation gate is the right moment to evaluate option (b).
//! **Until then, cross-crate callers duplicate locally or wait for M-794.**
//!
//! Guarantee tag: `Declared` — the harness shapes are illustrative fixtures, not verified bounds.

use crate::bound::{Bound, BoundBasis, BoundKind, NormKind};
use crate::cert_mode::CertMode;
use crate::guarantee::GuaranteeStrength;
use crate::meta::{Meta, Provenance};

// ---------------------------------------------------------------------------
// § 1. Canonical bound fixtures
// ---------------------------------------------------------------------------

/// A [`BoundBasis::ProvenThm`] ε-error bound — the canonical pre-image for
/// [`GuaranteeStrength::Proven`] tests (matches the Dense F32→BF16 swap's emission).
///
/// **Mutant-witness:** replacing `BoundBasis::ProvenThm` with `BoundBasis::UserDeclared` would
/// cause `Meta::new` with `Proven` to fail M-I2, catching any test that forgets to check the
/// basis coupling.
pub fn proven_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.003_906_25, // 2^{-8}: the round-to-nearest relative bound for BF16
            norm: NormKind::Rel,
        },
        basis: BoundBasis::ProvenThm {
            citation: "round-to-nearest relative error theorem".to_owned(),
        },
    }
}

/// A [`BoundBasis::EmpiricalFit`] failure-probability bound — the canonical pre-image for
/// [`GuaranteeStrength::Empirical`] tests (matches the Dense↔VSA swap's empirical-profile
/// emission).
///
/// **Mutant-witness:** replacing `trials: 10_000` with `trials: 0` would fail `Bound::well_formed`
/// (A6-02), catching any test that does not validate the basis constraints.
pub fn empirical_bound() -> Bound {
    Bound {
        kind: BoundKind::Probability { delta: 0.05 },
        basis: BoundBasis::EmpiricalFit {
            trials: 10_000,
            method: "Monte-Carlo round trip".to_owned(),
        },
    }
}

/// A [`BoundBasis::UserDeclared`] ε-error bound — the canonical pre-image for
/// [`GuaranteeStrength::Declared`] tests.
///
/// **Mutant-witness:** replacing `BoundBasis::UserDeclared` with `BoundBasis::ProvenThm` would
/// cause `Meta::new` with `Declared` to fail M-I4, catching any test that forgets the basis
/// constraint.
pub fn declared_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.1,
            norm: NormKind::L2,
        },
        basis: BoundBasis::UserDeclared,
    }
}

/// Select the canonical bound for a given [`GuaranteeStrength`], consistent with M-I1…M-I4:
/// `Exact` → `None`; all others → `Some(<canonical bound>)`.
///
/// This is the single source of truth for "which bound pairs with which strength" in mode-
/// parametric loops, so tests do not construct inconsistent pre-images.
pub fn canonical_bound(g: GuaranteeStrength) -> Option<Bound> {
    match g {
        GuaranteeStrength::Exact => None,
        GuaranteeStrength::Proven => Some(proven_bound()),
        GuaranteeStrength::Empirical => Some(empirical_bound()),
        GuaranteeStrength::Declared => Some(declared_bound()),
    }
}

/// Assert that a `(guarantee, bound)` pair is accepted by [`Meta::new`] (the M-I1…M-I4
/// checker). This is the central contract: `gate_result` and other policy primitives must always
/// yield a Meta-constructible pair. Panics with a descriptive message on failure.
///
/// **Guarantee tag:** `Declared` — this is a test-support helper, not a verified bound.
pub fn assert_meta_constructs(g: GuaranteeStrength, b: Option<Bound>) {
    let meta = Meta::new(Provenance::Root, g, b.clone(), None, None, None);
    assert!(
        meta.is_ok(),
        "pair must be Meta-constructible (g={g:?}, bound={b:?}), got Err({:?})",
        meta.err()
    );
}

// ---------------------------------------------------------------------------
// § 2. Mode-parametric iteration
// ---------------------------------------------------------------------------

/// Run `f(mode)` for every mode in [`CertMode::ALL`] (weakest → strongest: Fast, Balanced,
/// Certified).
///
/// Use this wherever a test must assert the same property across all three tiers, making the
/// "this holds in every mode" scope explicit (RFC-0034 §13; DN-20: every mode-sensitive
/// assertion states its scope, never leaves it implicitly all-on).
///
/// **Example:**
/// ```ignore
/// for_each_mode(|mode| {
///     let (g, b) = mode.gate_result(GuaranteeStrength::Exact, None);
///     assert_meta_constructs(g, b);
/// });
/// ```
pub fn for_each_mode(mut f: impl FnMut(CertMode)) {
    for &mode in &CertMode::ALL {
        f(mode);
    }
}

// ---------------------------------------------------------------------------
// § 3. Cross-mode negative pattern — ModeScope + assert_mode_scope
// ---------------------------------------------------------------------------

/// A predicate set describing in which [`CertMode`] tiers a property is expected to hold.
///
/// The cross-mode negative pattern (RFC-0034 §13; M-795) requires asserting not only that an
/// invariant fires in the modes it applies to, but also that it is **correctly absent/relaxed**
/// in the modes it does not. `ModeScope` makes the intended scope explicit and machine-checkable,
/// preventing the "invariant holds where it shouldn't" defect.
///
/// ## Predefined scopes (cover the common RFC-0034 cases)
/// - [`FAST_ONLY`](ModeScope::FAST_ONLY): property holds only in `Fast`.
/// - [`CERTIFIED_ONLY`](ModeScope::CERTIFIED_ONLY): property holds only in `Certified` (e.g.
///   swap-cert *checking*).
/// - [`EMIT_MODES`](ModeScope::EMIT_MODES): `Balanced` + `Certified` — modes that emit
///   certificates (RFC-0034 §5). Equivalent to [`NON_FAST`](ModeScope::NON_FAST).
/// - [`ALL_MODES`](ModeScope::ALL_MODES): property holds in every mode (e.g. Axis-B
///   never-silent fallibility, cert_mode tag presence).
/// - [`NON_FAST`](ModeScope::NON_FAST): `Balanced` + `Certified` — modes where the machinery
///   runs and `Empirical`/`Proven` are reachable.
///
/// Custom scopes are built by providing a `[bool; 3]` in `[Fast, Balanced, Certified]` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeScope {
    /// `in_scope[i]` = true means `CertMode::ALL[i]` is in scope. Indices: 0=Fast, 1=Balanced,
    /// 2=Certified (the same order as [`CertMode::ALL`]).
    pub in_scope: [bool; 3],
}

impl ModeScope {
    /// Property holds in **every** mode (e.g. Axis-B never-silent fallibility, cert_mode tag
    /// presence). In scope: Fast, Balanced, Certified.
    pub const ALL_MODES: ModeScope = ModeScope {
        in_scope: [true, true, true],
    };

    /// Property holds **only in `Fast`** (e.g. the floor from `Proven`/`Empirical` to `Declared`,
    /// cert suppression). In scope: Fast only.
    pub const FAST_ONLY: ModeScope = ModeScope {
        in_scope: [true, false, false],
    };

    /// Property holds in **`Balanced` and `Certified`** — the modes where the certification
    /// machinery runs and `Empirical`/`Proven` tags are reachable (RFC-0034 §5). In scope:
    /// Balanced, Certified.
    pub const NON_FAST: ModeScope = ModeScope {
        in_scope: [false, true, true],
    };

    /// Property holds **only in `Certified`** (e.g. certificate *checking*; RFC-0034 §5). In
    /// scope: Certified only.
    pub const CERTIFIED_ONLY: ModeScope = ModeScope {
        in_scope: [false, false, true],
    };

    /// Property holds in **`Balanced` and `Certified`** — the modes that *emit* swap certificates
    /// (RFC-0034 §5). Alias for [`NON_FAST`](ModeScope::NON_FAST) expressed by function. In
    /// scope: Balanced, Certified.
    pub const EMIT_MODES: ModeScope = ModeScope {
        in_scope: [false, true, true],
    };

    /// Returns `true` iff the given mode is in scope.
    pub fn contains(self, mode: CertMode) -> bool {
        // CertMode::ALL order: [Fast=0, Balanced=1, Certified=2] matches `depth()`.
        self.in_scope[mode.depth() as usize]
    }
}

/// Assert that `predicate(mode)` returns `true` for every mode **in** `scope` and `false` for
/// every mode **outside** `scope`, sweeping [`CertMode::ALL`].
///
/// This is the mechanical implementation of the cross-mode negative pattern (M-795): it catches
/// both (a) a property not holding when it should, **and** (b) a property holding when it should
/// not ("the invariant holding where it shouldn't").
///
/// `desc` is a human-readable description of the property being tested, included in panic messages.
///
/// **Mutant-witness for the negative arm:** if the predicate were `|_| true` (always holds), the
/// `FAST_ONLY` scope would panic on `Balanced`/`Certified` because `holds=true` but
/// `expected=false`. Conversely, `|_| false` would panic on `Fast` because `holds=false` but
/// `expected=true`. Both directions are caught.
///
/// **Example — cert emission scope:**
/// ```ignore
/// assert_mode_scope(
///     ModeScope::EMIT_MODES,
///     |mode| gate_swap(&src, value.clone(), cert.clone(), mode)
///         .unwrap()
///         .certificate
///         .is_some(),
///     "swap-cert emission",
/// );
/// ```
pub fn assert_mode_scope(scope: ModeScope, predicate: impl Fn(CertMode) -> bool, desc: &str) {
    for &mode in &CertMode::ALL {
        let holds = predicate(mode);
        let expected = scope.contains(mode);
        if holds && !expected {
            panic!(
                "cross-mode NEGATIVE failed: `{desc}` holds in {mode:?} but should NOT \
                 (scope={scope:?}). The invariant fires where it shouldn't."
            );
        }
        if !holds && expected {
            panic!(
                "cross-mode POSITIVE failed: `{desc}` does NOT hold in {mode:?} but SHOULD \
                 (scope={scope:?}). The invariant is absent where it must fire."
            );
        }
    }
}
