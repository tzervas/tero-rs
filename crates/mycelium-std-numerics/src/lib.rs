//! `std.numerics` — the honest ε/δ carrier and meet-composition surface (M-512, issue #153).
//!
//! `std.numerics` is the library home for *carrying a value together with its `Meta`-attached
//! `{Bound, GuaranteeStrength}`* ([`Approx<T>`]) and for the **meet-composition /
//! refuse-without-a-rule** posture that keeps bounds honest. The verified ε/δ kernels live in
//! `mycelium-numerics` (ADR-010); this module is their **ergonomic, never-upgrading surface**
//! (KC-3 — it adds no trusted bound algebra, it consumes it).
//!
//! # Honesty crux (C2 / VR-5)
//!
//! A helper's guarantee tag is exactly what its basis supports and is **never upgraded**: strength
//! composes by [`GuaranteeStrength::meet`] (weakest-wins), bounds propagate with **outward
//! (directed) rounding** (via `mycelium-numerics`), and an op lacking a sound rule **refuses**
//! (`Err`) rather than fabricating a bound. `Proven` is constructible **only** via a
//! [`ProvenThm`]-typed witness token — there is no other path (FR-N3, type-level; sealed module).
//!
//! # Design grounding
//!
//! - Spec: `docs/spec/stdlib/numerics.md` (§3 exported-op surface, §4 guarantee matrix)
//! - ADR-010 (the two bound kernels + shared certificate), ADR-011 (`BoundBasis` universal)
//! - `mycelium-numerics` — M-201 `::error`, M-202 `::prob`, M-203 `::cert` (Done 2026-06-09)
//! - RFC-0016 §4.1 C1–C6, §4.2 Ring-1, §4.5 guarantee matrix
//! - RFC-0001 §4.3/§4.7 (`Meta`, guarantee lattice, M-I1–M-I4)
//! - Task M-512, issue #153
//!
//! # ε-ownership (NFR-N2 — RESOLVED)
//!
//! This module is the ε-carrier home, so it now **owns** the `Declared`-strength float ε that
//! `std.math` previously held locally: [`DECLARED_FLOAT_EPS`]. `std.math` re-exports it from here
//! rather than restating it (the math.md §7-Q2 / README §5 ε-ownership FLAG). The value is a
//! **`Declared`** placeholder (`UserDeclared` basis) for the unaudited libm transcendental floor;
//! its honest upgrade to a `Proven` magnitude is owned by the kernel (ADR-010 / `mycelium-numerics`)
//! and the audited `wild`/FFI floor (M-541), reachable only via a checked basis (VR-5) — never by
//! restating a tighter number here.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Numerics ops are representation-agnostic (the bound
//! algebra operates over floats and fixed-point values regardless of `Repr`); however, every
//! result always carries an explicit `Bound` with its `BoundBasis` — no precision is implicitly
//! reduced or silently upgraded. The `Proven` strength is reachable only via a `ProvenThm`
//! witness token; there is no ambient path to `Proven`.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/numerics.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod matrix;

use mycelium_core::{Bound, BoundBasis, BoundKind, GuaranteeStrength, NormKind};
use mycelium_numerics::{
    accuracy_to_probability as kern_accuracy_to_probability, basis_strength, check_error_claim,
    check_union_claim, compose_error_bound, CheckOutcome, ProbBound,
};

// Re-export kernel types so consumers can use them without depending on mycelium-numerics directly.
// `ErrorBound` and `ErrorOp` and `ProbBound` are the kernel types this module surfaces (NFR-N2:
// cite ε constants from `mycelium-numerics`; restate none).
pub use mycelium_numerics::{ErrorBound, ErrorOp, ProbBound as KernelProbBound};

/// The `Declared`-strength ε upper bound for `f64` operations whose compute floor is the
/// platform libm (an unaudited `wild`/FFI floor — ADR-014; M-541).
///
/// Conservative: `2 · f64::EPSILON ≈ 4.44e-16` (Linf norm). Its basis is **`UserDeclared`** — there
/// is no checked-side-condition theorem backing the libm call yet — so any value carrying it tags at
/// most `Declared` (VR-5: downgrade to stay honest, never upgrade without a checked basis). The
/// honest *Proven* magnitude is the kernel's (ADR-010 / `mycelium-numerics`) and the audited floor's
/// (M-541) to supply; do not restate a tighter number here.
///
/// This is the single home for the constant (NFR-N2): `std.math` re-exports it from here.
pub const DECLARED_FLOAT_EPS: f64 = 2.0 * f64::EPSILON;

// ──────────────────────────────────────────────────────────────────────────────
// Sealed witness token for `Proven` construction (FR-N3)
// ──────────────────────────────────────────────────────────────────────────────

/// Module-private seal so [`ProvenThm`] cannot be constructed outside this crate.
///
/// `ProvenThm` carries a `_seal: Sealed` field (private). Because `Sealed` is in a private module
/// and its constructor is not exported, the only way to obtain a `ProvenThm` is via
/// [`ProvenThm::new`] — which is the sanctioned, validating entry point.
///
/// This enforces FR-N3 at the type level: there is **no** `unsafe` or reflection route around it
/// in Rust's type system (the `#![forbid(unsafe_code)]` lint above backs this up).
mod sealed {
    /// Private zero-size sentinel that cannot be named outside this crate.
    #[derive(Debug, Clone, PartialEq)]
    pub(super) struct Sealed;
}

/// A checked-theorem witness required to construct an [`Approx`] with `Proven` strength (FR-N3).
///
/// The only way to obtain a `ProvenThm` is via [`ProvenThm::new`], which validates that the
/// citation is non-empty (ADR-011 / `bound.schema.json`: `citation` must not be blank). The
/// private `sealed::Sealed` sentinel field makes it impossible to construct `ProvenThm { .. }`
/// by hand outside this crate — type-system enforcement, no `unsafe` required.
///
/// # Example
///
/// ```rust
/// # use mycelium_std_numerics::ProvenThm;
/// let thm = ProvenThm::new("ADR-010 §1 affine-arithmetic ε-composition").unwrap();
/// ```
///
/// # Compile-time enforcement
///
/// The following does **not** compile because `_seal` is private:
///
/// ```compile_fail
/// use mycelium_std_numerics::ProvenThm;
/// // Error: field `_seal` is private
/// let _ = ProvenThm { citation: "my theorem".to_owned(), _seal: () };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProvenThm {
    /// The theorem citation (non-empty — validated in [`ProvenThm::new`]).
    pub citation: String,
    /// Private sentinel — cannot be constructed outside this crate.
    _seal: sealed::Sealed,
}

impl ProvenThm {
    /// Construct a [`ProvenThm`] witness with the given `citation`.
    ///
    /// Returns `None` if `citation` is empty or whitespace-only (ADR-011: every `ProvenThm`
    /// basis must carry a non-empty, non-blank citation; an empty string would make the tag
    /// vacuous and thus dishonest — VR-5).
    ///
    /// Mutation witness: removing the `trim().is_empty()` check allows an empty citation to
    /// produce a `Proven` `Approx`, silently violating ADR-011.
    #[must_use]
    pub fn new(citation: impl Into<String>) -> Option<Self> {
        let citation = citation.into();
        if citation.trim().is_empty() {
            return None;
        }
        Some(ProvenThm {
            citation,
            _seal: sealed::Sealed,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Structured errors (spec §3 `NumErr` / `CheckErr`; RFC-0013 refusal record)
// ──────────────────────────────────────────────────────────────────────────────

/// Structured refusal record for `std.numerics` helpers (C1; RFC-0013; spec §3 `NumErr`).
///
/// Every failure mode is an explicit variant — no silent coercion, no fabricated default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumErr {
    /// `eps < 0` or non-finite (out-of-range ε — mirrors the M-203 range-checked constructor).
    BadEps,
    /// `delta ∉ [0, 1]` or non-finite (out-of-range δ — M-203).
    BadDelta,
    /// No sound ε propagation rule exists for this op / input configuration (M-204 posture:
    /// **refuse, never fabricate**). Inputs may be non-`Error` bounds, have mismatched norms,
    /// have wrong arity, or the input slice is empty.
    NoRule,
    /// Input norms disagree and cannot be combined without a norm-coercion rule (never silent).
    NormMismatch,
    /// A composed ε overflowed to non-finite — refused, not emitted as a vacuous bound (A2-04).
    Overflow,
}

impl core::fmt::Display for NumErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NumErr::BadEps => write!(f, "eps is negative or non-finite (out-of-range ε)"),
            NumErr::BadDelta => {
                write!(f, "delta is not in [0,1] or non-finite (out-of-range δ)")
            }
            NumErr::NoRule => write!(
                f,
                "no sound ε propagation rule for this op/configuration (refuse, never fabricate)"
            ),
            NumErr::NormMismatch => {
                write!(
                    f,
                    "input norms disagree — cannot combine without norm-coercion rule"
                )
            }
            NumErr::Overflow => write!(
                f,
                "composed ε overflowed to non-finite — refused (not a vacuous bound)"
            ),
        }
    }
}

/// Structured verdict for the tier-i re-validation checker (spec §3 `CheckErr`; RFC-0013).
#[derive(Debug, Clone, PartialEq)]
pub enum CheckErr {
    /// The claimed bound is **tighter** than the kernel re-derivation (never a silent pass).
    Rejected {
        /// The bound the kernel re-derives (the sound floor).
        recomputed: f64,
        /// The (too-tight) bound that was claimed.
        claimed: f64,
    },
    /// The certificate is ill-formed (bad arity, norm mismatch, non-finite magnitudes).
    Malformed,
}

impl core::fmt::Display for CheckErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CheckErr::Rejected {
                recomputed,
                claimed,
            } => write!(
                f,
                "claimed ε={claimed:.2e} is tighter than the re-derivation {recomputed:.2e} \
                 (rejected — not a silent pass)"
            ),
            CheckErr::Malformed => write!(f, "certificate is ill-formed (malformed)"),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Approx<T> carrier (FR-N1)
// ──────────────────────────────────────────────────────────────────────────────

/// A thin view pairing a value with its `{Bound, strength}` (RFC-0001 §4.3 `Meta`) — **not** a
/// new numeric type and **no kernel change** (FR-N1 / KC-3).
///
/// `Approx<T>` is the carrier that `math` (M-525) and `dense` (M-518) defer to this module
/// (spec §3, §7-Q1): it is a plain value with its `Meta`-attached `{Bound, strength}`, not a
/// parallel numeric type, so it composes with content-addressing (ADR-003) and adds no new
/// representation kind.
///
/// # Strength invariant (VR-5)
///
/// `strength` is **derived** from `bound.basis` by the constructors and is never set by the
/// caller directly. Invariant: `self.strength == basis_strength(&self.bound.basis)`.
///
/// # Proven-strength gating (FR-N3)
///
/// The [`proven`](Approx::proven) constructor is the sole `Proven`-strength path and requires a
/// [`ProvenThm`] witness (private sealed token — type-level enforcement; no `unsafe` route).
#[derive(Debug, Clone, PartialEq)]
pub struct Approx<T> {
    /// The carried value.
    pub value: T,
    /// The error/probability bound certifying `value` (carries its `BoundBasis` per ADR-011).
    pub bound: Bound,
    /// The honest guarantee strength — derived from `bound.basis` by the constructors, **never**
    /// set independently (VR-5). Invariant: `strength == basis_strength(&bound.basis)`.
    ///
    /// **Private (the VR-5 seal):** a public field would let a caller write
    /// `Approx { strength: Proven, .. }` and bypass the [`ProvenThm`] witness; keeping it private
    /// means the only way to build an `Approx` is the basis-deriving constructors. Read it via
    /// [`Approx::strength`].
    strength: GuaranteeStrength,
}

impl<T> Approx<T> {
    /// Construct a `Declared`-strength approximation (basis = `UserDeclared`, always-flagged,
    /// user-asserted bound — M-I4 / VR-5).
    ///
    /// The strength is derived from the bound's basis, not asserted. This is the explicit,
    /// always-flagged escape hatch (spec §3 `declared(...)` / M-I4); `explain` surfaces the
    /// "declared, unverified" marker (M-I4/VR-5).
    #[must_use]
    pub fn declared(value: T, bound: Bound) -> Self {
        let strength = basis_strength(&bound.basis);
        Self {
            value,
            bound,
            strength,
        }
    }

    /// Construct an `Empirical`-strength approximation (basis = `EmpiricalFit{trials, method}`).
    ///
    /// The strength is derived from the bound's basis (VR-5). If the bound carries a
    /// `UserDeclared` basis the strength will be `Declared` (basis-derived, not asserted).
    #[must_use]
    pub fn empirical(value: T, bound: Bound) -> Self {
        let strength = basis_strength(&bound.basis);
        Self {
            value,
            bound,
            strength,
        }
    }

    /// Construct a `Proven`-strength approximation, gated by a [`ProvenThm`] witness (FR-N3).
    ///
    /// The `witness` argument is a checked-theorem token that can only be obtained via
    /// [`ProvenThm::new`] (private sentinel field — type-level enforcement). Without a real
    /// `ProvenThm` this function cannot be called, ensuring `Proven` strength is unattainable
    /// without evidence.
    ///
    /// The bound's basis is replaced with the witness's citation as a `BoundBasis::ProvenThm` so
    /// the `basis_strength` derivation produces `GuaranteeStrength::Proven`.
    ///
    /// Mutation witness: removing the `witness` parameter or making `ProvenThm` publicly
    /// constructible would allow `Proven` without a checked basis, violating VR-5.
    #[must_use]
    pub fn proven(value: T, bound: Bound, witness: ProvenThm) -> Self {
        // Replace the bound's basis with the witness's citation — this is the only place that
        // can produce a `ProvenThm`-basis bound in this crate, gated by the token.
        let proven_bound = Bound {
            kind: bound.kind,
            basis: BoundBasis::ProvenThm {
                citation: witness.citation,
            },
        };
        let strength = basis_strength(&proven_bound.basis);
        Self {
            value,
            bound: proven_bound,
            strength,
        }
    }

    /// Attach a bound to a value: the strength is derived from `bound.basis` (VR-5 — never
    /// asserted). This is the generic `attach` helper from spec §3: the tag is exactly what the
    /// basis supports (M-I2/M-I3/M-I4).
    ///
    /// Prefer the named constructors ([`Approx::declared`], [`Approx::empirical`],
    /// [`Approx::proven`]) for clarity; use this for programmatic use when the basis is not
    /// statically known.
    #[must_use]
    pub fn attach(value: T, bound: Bound) -> Self {
        let strength = basis_strength(&bound.basis);
        Self {
            value,
            bound,
            strength,
        }
    }

    /// The honest strength of this approximation (derived from the basis, invariant).
    #[must_use]
    pub fn strength(&self) -> GuaranteeStrength {
        self.strength
    }

    /// Project the carried value.
    #[must_use]
    pub fn value_of(&self) -> &T {
        &self.value
    }

    /// Project the bound (total; every `Approx` carries a bound).
    #[must_use]
    pub fn bound_of(&self) -> &Bound {
        &self.bound
    }

    /// Compose two `Approx<T>` values by taking the **meet** of their strengths and propagating
    /// the ε bound through the `mycelium-numerics` affine-arithmetic kernel under `op` (FR-N2).
    ///
    /// Returns `Err` (refuses, never fabricates) when:
    /// - `op` has no sound ε-propagation rule for the given input kinds (`NumErr::NoRule`,
    ///   M-204 posture): inputs are not `Error` bounds, norms disagree, or arity is wrong.
    /// - the composed ε overflows to non-finite (`NumErr::Overflow` — A2-04).
    ///
    /// The composed bound is computed via `mycelium-numerics::compose_error_bound`, which uses
    /// **outward-rounded** addition (banked guard 1 / A2-01): the stored ε is a true upper bound,
    /// never round-to-nearest (which can fall below the real value).
    ///
    /// The `combined_value` argument is the caller-supplied composed value (this helper manages
    /// only the bound/strength per spec §3; the value arithmetic is the caller's).
    ///
    /// # Errors
    ///
    /// Returns [`NumErr`] when no sound rule exists or the composition overflows.
    pub fn combine(
        &self,
        other: &Approx<T>,
        combined_value: T,
        op: ErrorOp,
    ) -> Result<Approx<T>, NumErr> {
        let composed =
            compose_error_bound(&[&self.bound, &other.bound], op).ok_or(NumErr::NoRule)?;
        // Re-validate: overflow to non-finite is a refusal (A2-04).
        if !composed.bound.well_formed() {
            return Err(NumErr::Overflow);
        }
        Ok(Approx {
            value: combined_value,
            bound: composed.bound,
            strength: composed.strength,
        })
    }

    /// Map a function over the value, carrying the bound and strength unchanged.
    ///
    /// The bound/strength of `self` applies to the *result* only when `f` is exact (introduces
    /// no new approximation). The caller asserts this; the bound propagates without modification.
    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Approx<U> {
        Approx {
            value: f(self.value),
            bound: self.bound,
            strength: self.strength,
        }
    }
}

/// The `explain` artifact for an [`Approx<T>`] (C3; G11 dual human/machine projection).
///
/// Projects `{kind, eps|delta, norm, basis, strength}` and **why** the bound holds: which
/// `ProvenThm{citation}`, which `EmpiricalFit{trials, method}`, or the `UserDeclared`
/// "declared, unverified" marker. `eps` is `+∞` for non-`Error` bounds (honest, never `NaN`).
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    /// The bound kind tag ("Error" / "Probability" / "Crosstalk" / "Capacity").
    pub kind: String,
    /// The ε magnitude, or `+∞` when the bound is not an `Error` kind (honest; never `NaN`).
    pub eps: f64,
    /// The norm in which `eps` is expressed (defaults to `Linf` for non-`Error` kinds).
    pub norm: NormKind,
    /// The δ failure probability (or `0.0` when the bound is not a `Probability` kind).
    pub delta: f64,
    /// The basis: which theorem, which empirical fit, or `UserDeclared`.
    pub basis: BoundBasis,
    /// The honest guarantee strength.
    pub strength: GuaranteeStrength,
    /// Human-readable one-line summary (G11 human projection).
    pub summary: String,
}

impl<T: core::fmt::Debug> Approx<T> {
    /// Project this carrier to a dual human/machine EXPLAIN record (G11; C3; spec §3 `explain`).
    ///
    /// Total — every `Approx<T>` carries a bound. For non-`Error` kinds the `eps` is `+∞`
    /// (the honest unstated bound); `NaN` is **never** returned (NFR-N1 / C1).
    #[must_use]
    pub fn explain(&self) -> Explanation {
        let (kind, eps, norm, delta) = match &self.bound.kind {
            BoundKind::Error { eps, norm } => ("Error".to_owned(), *eps, *norm, 0.0),
            BoundKind::Probability { delta } => {
                // eps is unstated — represent honestly as +∞, never NaN (NFR-N1 / C1).
                (
                    "Probability".to_owned(),
                    f64::INFINITY,
                    NormKind::Linf,
                    *delta,
                )
            }
            BoundKind::Crosstalk { expected, .. } => {
                ("Crosstalk".to_owned(), *expected, NormKind::Linf, 0.0)
            }
            BoundKind::Capacity { .. } => {
                ("Capacity".to_owned(), f64::INFINITY, NormKind::Linf, 0.0)
            }
        };
        let basis_desc = match &self.bound.basis {
            BoundBasis::ProvenThm { citation } => format!("ProvenThm: {citation}"),
            BoundBasis::EmpiricalFit { trials, method } => {
                format!("EmpiricalFit ({trials} trials): {method}")
            }
            BoundBasis::UserDeclared => {
                "UserDeclared (declared, unverified — not Proven; M-I4/VR-5)".to_owned()
            }
        };
        let summary = format!(
            "{:?} [{kind}] value={:?}; ε={eps:.2e} ({norm:?}); δ={delta:.2e}; basis={basis_desc}",
            self.strength, self.value,
        );
        Explanation {
            kind,
            eps,
            norm,
            delta,
            basis: self.bound.basis.clone(),
            strength: self.strength,
            summary,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bound constructors (spec §3; ADR-011)
// ──────────────────────────────────────────────────────────────────────────────

/// Construct an `ErrorBound{eps, norm, basis}` (spec §3 `error_bound`).
///
/// Returns `Err(NumErr::BadEps)` when `eps < 0` or non-finite — mirrors the M-203 range-checked
/// constructor; never a silent coercion (C1/A2-03).
///
/// # Errors
///
/// - [`NumErr::BadEps`] when `eps < 0` or non-finite.
pub fn error_bound(eps: f64, norm: NormKind, basis: BoundBasis) -> Result<Bound, NumErr> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(NumErr::BadEps);
    }
    Ok(Bound {
        kind: BoundKind::Error { eps, norm },
        basis,
    })
}

/// Construct a `ProbabilityBound{delta, basis}` (spec §3 `prob_bound`).
///
/// Returns `Err(NumErr::BadDelta)` when `delta ∉ [0, 1]` or non-finite — mirrors M-203.
///
/// # Errors
///
/// - [`NumErr::BadDelta`] when `delta ∉ [0, 1]` or non-finite.
pub fn prob_bound(delta: f64, basis: BoundBasis) -> Result<Bound, NumErr> {
    if !delta.is_finite() || !(0.0..=1.0).contains(&delta) {
        return Err(NumErr::BadDelta);
    }
    Ok(Bound {
        kind: BoundKind::Probability { delta },
        basis,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Composition helpers (spec §3; FR-N2; ADR-010)
// ──────────────────────────────────────────────────────────────────────────────

/// Compose the **δ union bound** of a slice of `Probability`-kind bounds, taking the **meet** of
/// their strengths (spec §3 `union_delta`; M-202; ADR-010 §2).
///
/// `P(⋃ Eᵢ) ≤ min(1, Σδᵢ)`. The sum is **outward-rounded** by the `mycelium-numerics` kernel
/// so the composed δ is a true upper bound (banked guard 1 / A2-01).
///
/// Returns `Err(NumErr::NoRule)` if the slice is empty (refuse, not fabricate; M-204 posture).
/// Returns `Err(NumErr::BadDelta)` if any input is not a `Probability` bound or has a malformed δ.
///
/// # Errors
///
/// - [`NumErr::NoRule`] when `bounds` is empty.
/// - [`NumErr::BadDelta`] when any bound is not a `Probability` kind or has an invalid δ.
pub fn union_delta(bounds: &[&Bound]) -> Result<Bound, NumErr> {
    if bounds.is_empty() {
        return Err(NumErr::NoRule);
    }
    // Extract ProbBound from each Bound; refuse if any is not Probability kind.
    let prob_bounds: Vec<ProbBound> = bounds
        .iter()
        .map(|b| match b.kind {
            BoundKind::Probability { delta } => ProbBound::new(delta).ok_or(NumErr::BadDelta),
            _ => Err(NumErr::BadDelta),
        })
        .collect::<Result<_, _>>()?;

    let composed = ProbBound::union(prob_bounds.iter());

    // Meet of all input strengths.
    let strength = bounds
        .iter()
        .map(|b| basis_strength(&b.basis))
        .fold(GuaranteeStrength::TOP, GuaranteeStrength::meet);

    // Re-derive the basis matching the meet strength.
    let basis = composed_prob_basis(strength, bounds);

    Ok(Bound {
        kind: BoundKind::Probability {
            delta: composed.delta(),
        },
        basis,
    })
}

/// The single sanctioned cross-kernel inference (spec §3 `accuracy_to_probability`; ADR-010 §4).
///
/// Given an `ErrorBound`-kind bound and a tolerance `tau ≥ 0`, returns a `Probability`-kind bound:
/// - `eps ≤ tau` → failure probability = `acc_delta` (the accuracy confidence carries over).
/// - `eps > tau` → honest worst case `δ = 1` (the bound permits a violation; never a silent tighten).
///
/// `acc_delta` is the failure probability of the accuracy claim itself (e.g. from a
/// `Probability`-kind companion); pass `0.0` for an accuracy bound that holds with certainty.
///
/// This is the **one** legal ε↔δ mixing; no other cross-kernel inference is exposed (ADR-010 §4).
///
/// # Errors
///
/// - [`NumErr::NoRule`] when `acc` is not an `Error`-kind bound.
/// - [`NumErr::BadEps`] when `acc` has a malformed ε.
/// - [`NumErr::BadDelta`] when `tau < 0` or `acc_delta ∉ [0, 1]`.
pub fn accuracy_to_probability(acc: &Bound, tau: f64, acc_delta: f64) -> Result<Bound, NumErr> {
    let (eps, norm) = match acc.kind {
        BoundKind::Error { eps, norm } => (eps, norm),
        _ => return Err(NumErr::NoRule),
    };
    let error_bound_val = ErrorBound::new(eps, norm).ok_or(NumErr::BadEps)?;

    let prob =
        kern_accuracy_to_probability(error_bound_val, tau, acc_delta).ok_or(NumErr::BadDelta)?;

    // Strength: inherits the accuracy bound's basis-implied strength (the conversion is exact).
    let strength = basis_strength(&acc.basis);
    let basis = match strength {
        GuaranteeStrength::Proven => BoundBasis::ProvenThm {
            citation: format!(
                "ADR-010 §4 accuracy→probability: eps={eps:.2e} ≤ tau={tau:.2e} ({})",
                match &acc.basis {
                    BoundBasis::ProvenThm { citation } => citation.as_str(),
                    _ => "inherited",
                }
            ),
        },
        GuaranteeStrength::Empirical => {
            let (trials, method) = match &acc.basis {
                BoundBasis::EmpiricalFit { trials, method } => (*trials, method.as_str()),
                _ => (0, "inherited"),
            };
            BoundBasis::EmpiricalFit {
                trials,
                method: format!("ADR-010 §4 accuracy→probability: {method}"),
            }
        }
        _ => BoundBasis::UserDeclared,
    };

    Ok(Bound {
        kind: BoundKind::Probability {
            delta: prob.delta(),
        },
        basis,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tier-i re-validation checker (spec §3 `check`; M-203; ADR-010 "Trusted base")
// ──────────────────────────────────────────────────────────────────────────────

/// Re-validate a claimed ε bound for `op` over `input_bounds` via the M-203 tier-i checker.
///
/// - **`Ok(())`**: the claimed bound is `≥` the kernel re-derivation (sound).
/// - **`Err(CheckErr::Rejected{..})`**: the claim is tighter — rejected (not a silent pass;
///   RFC-0002 §2; ADR-010 tier-i).
/// - **`Err(CheckErr::Malformed)`**: the certificate cannot be re-derived (bad arity, norm
///   mismatch).
///
/// # Errors
///
/// Returns [`CheckErr`] when the claim is tighter than the re-derivation or malformed.
pub fn check_error(
    input_bounds: &[ErrorBound],
    op: ErrorOp,
    claimed: ErrorBound,
) -> Result<(), CheckErr> {
    match check_error_claim(input_bounds, op, claimed) {
        CheckOutcome::Valid => Ok(()),
        CheckOutcome::Rejected {
            recomputed,
            claimed,
        } => Err(CheckErr::Rejected {
            recomputed,
            claimed,
        }),
        CheckOutcome::Malformed => Err(CheckErr::Malformed),
    }
}

/// Re-validate a claimed δ union bound over `input_bounds` via the M-203 tier-i checker.
///
/// - **`Ok(())`**: the claimed δ is `≥` the re-derivation (sound).
/// - **`Err(CheckErr::Rejected{..})`**: the claim is tighter — rejected (not a silent pass).
/// - **`Err(CheckErr::Malformed)`**: reserved; currently fires only on empty input.
///
/// # Errors
///
/// Returns [`CheckErr`] when the claim is tighter or malformed.
pub fn check_union(input_bounds: &[ProbBound], claimed: ProbBound) -> Result<(), CheckErr> {
    match check_union_claim(input_bounds, claimed) {
        CheckOutcome::Valid => Ok(()),
        CheckOutcome::Rejected {
            recomputed,
            claimed,
        } => Err(CheckErr::Rejected {
            recomputed,
            claimed,
        }),
        CheckOutcome::Malformed => Err(CheckErr::Malformed),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Derive an honest `BoundBasis` for a composed probability bound at the meet `strength`.
/// Mirrors the `composed_basis` logic in `mycelium-numerics::cert` (spec §4.4; VR-5).
fn composed_prob_basis(strength: GuaranteeStrength, bases: &[&Bound]) -> BoundBasis {
    match strength {
        GuaranteeStrength::Exact | GuaranteeStrength::Proven => {
            // All inputs were Proven — the union bound is the proof (ADR-010 §2).
            let inputs: Vec<&str> = bases
                .iter()
                .filter_map(|b| match &b.basis {
                    BoundBasis::ProvenThm { citation } => Some(citation.as_str()),
                    _ => None,
                })
                .collect();
            let citation = if inputs.is_empty() {
                "ADR-010 §2 union-bound δ-composition".to_owned()
            } else {
                format!(
                    "ADR-010 §2 union-bound δ-composition over [{}]",
                    inputs.join("; ")
                )
            };
            BoundBasis::ProvenThm { citation }
        }
        GuaranteeStrength::Empirical => {
            let trials = bases
                .iter()
                .filter_map(|b| match &b.basis {
                    BoundBasis::EmpiricalFit { trials, .. } => Some(*trials),
                    _ => None,
                })
                .min()
                .unwrap_or(0);
            BoundBasis::EmpiricalFit {
                trials,
                method: "composed (ADR-010 §2 union-bound δ)".to_owned(),
            }
        }
        GuaranteeStrength::Declared => BoundBasis::UserDeclared,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{BoundBasis, BoundKind, GuaranteeStrength, NormKind};
    use mycelium_numerics::ErrorOp;

    // ── test helpers ──────────────────────────────────────────────────────────

    fn mk_error_proven(eps: f64) -> Bound {
        Bound {
            kind: BoundKind::Error {
                eps,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::ProvenThm {
                citation: "test-theorem".to_owned(),
            },
        }
    }

    fn mk_error_empirical(eps: f64) -> Bound {
        Bound {
            kind: BoundKind::Error {
                eps,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::EmpiricalFit {
                trials: 100,
                method: "test-fit".to_owned(),
            },
        }
    }

    fn mk_error_declared(eps: f64) -> Bound {
        Bound {
            kind: BoundKind::Error {
                eps,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::UserDeclared,
        }
    }

    fn mk_prob_proven(delta: f64) -> Bound {
        Bound {
            kind: BoundKind::Probability { delta },
            basis: BoundBasis::ProvenThm {
                citation: "test-union-theorem".to_owned(),
            },
        }
    }

    fn mk_prob_declared(delta: f64) -> Bound {
        Bound {
            kind: BoundKind::Probability { delta },
            basis: BoundBasis::UserDeclared,
        }
    }

    fn approx_with(eps: f64, strength: GuaranteeStrength) -> Approx<f64> {
        let bound = match strength {
            GuaranteeStrength::Exact | GuaranteeStrength::Proven => mk_error_proven(eps),
            GuaranteeStrength::Empirical => mk_error_empirical(eps),
            GuaranteeStrength::Declared => mk_error_declared(eps),
        };
        Approx {
            value: 0.0_f64,
            bound,
            strength,
        }
    }

    // ── FR-N1: Approx<T> carrier ──────────────────────────────────────────────

    #[test]
    fn declared_constructor_derives_declared_strength() {
        let b = mk_error_declared(0.1);
        let a = Approx::declared(42.0_f64, b);
        // Mutation witness: swapping `basis_strength` for a constant breaks this.
        assert_eq!(a.strength, GuaranteeStrength::Declared);
    }

    #[test]
    fn empirical_constructor_derives_empirical_strength() {
        let b = mk_error_empirical(0.1);
        let a = Approx::empirical(42.0_f64, b);
        assert_eq!(a.strength, GuaranteeStrength::Empirical);
    }

    #[test]
    fn proven_constructor_requires_witness() {
        let thm = ProvenThm::new("ADR-010 §1 affine-arithmetic ε-composition").unwrap();
        let b = mk_error_proven(0.1);
        let a = Approx::proven(42.0_f64, b, thm);
        assert_eq!(a.strength, GuaranteeStrength::Proven);
        assert!(matches!(a.bound.basis, BoundBasis::ProvenThm { .. }));
    }

    #[test]
    fn proven_witness_empty_citation_returns_none() {
        // Mutation witness: removing `trim().is_empty()` allows empty citations → Proven bypass.
        assert!(
            ProvenThm::new("").is_none(),
            "empty citation must not produce a witness"
        );
        assert!(
            ProvenThm::new("   ").is_none(),
            "whitespace-only citation must not produce a witness"
        );
    }

    #[test]
    fn attach_derives_strength_from_basis() {
        let cases: &[(Bound, GuaranteeStrength)] = &[
            (mk_error_proven(0.0), GuaranteeStrength::Proven),
            (mk_error_empirical(0.1), GuaranteeStrength::Empirical),
            (mk_error_declared(0.5), GuaranteeStrength::Declared),
        ];
        for (bound, expected) in cases {
            let a = Approx::attach(1.0_f64, bound.clone());
            assert_eq!(
                a.strength, *expected,
                "attach must derive strength from basis (VR-5)"
            );
        }
    }

    // ── FR-N2: combine / meet + outward rounding ──────────────────────────────

    /// Property: `combine(a, b).strength == a.strength.meet(b.strength)` over ALL strength pairs
    /// reachable in `Approx<T>` (exhaustive — the space is finite; spec FR-N2 / dev-workflow §3).
    ///
    /// `Approx<T>` always carries a bound (M-I1: `Exact` means *no* bound — `Exact` strength is
    /// the identity of `meet` but is unreachable in a bound-carrying carrier). The reachable set
    /// is `{Proven, Empirical, Declared}` — 3×3 = 9 pairs, fully enumerated here.
    ///
    /// The `GuaranteeStrength::meet` lattice laws (commutativity, associativity, idempotence,
    /// identity `Exact`) are verified exhaustively over all 16 pairs in `mycelium-core`; this
    /// test verifies that `combine` correctly implements the meet on the reachable strength set.
    ///
    /// Mutation witness: replacing `.meet()` with any other combinator breaks at least one pair.
    #[test]
    fn combine_strength_is_meet_for_all_strength_pairs_exhaustive() {
        // The three strengths reachable in a bound-carrying Approx<T> (M-I1: Exact = no bound).
        const BOUND_CARRYING_STRENGTHS: [GuaranteeStrength; 3] = [
            GuaranteeStrength::Proven,
            GuaranteeStrength::Empirical,
            GuaranteeStrength::Declared,
        ];
        for &s_a in &BOUND_CARRYING_STRENGTHS {
            for &s_b in &BOUND_CARRYING_STRENGTHS {
                let a = approx_with(0.1, s_a);
                let b = approx_with(0.2, s_b);
                let combined = a
                    .combine(&b, 3.0_f64, ErrorOp::Add)
                    .unwrap_or_else(|e| panic!("combine({s_a:?},{s_b:?}) failed: {e}"));
                let expected = s_a.meet(s_b);
                assert_eq!(
                    combined.strength, expected,
                    "combine({s_a:?},{s_b:?}).strength must equal the meet ({expected:?})"
                );
            }
        }
    }

    /// Property: the composed ε is ≥ the real sum of the two input εs (outward rounding — FR-N2).
    ///
    /// For `ErrorOp::Add` and inputs where `eps_a + eps_b` rounds DOWN under round-to-nearest,
    /// `combine` must produce a result strictly greater than the rounded-down sum.
    ///
    /// Mutation witness: replacing the kernel's `add_up` with plain `+` causes this to fail
    /// for any input pair where the sum rounds down.
    #[test]
    fn combine_add_eps_is_outward_rounded() {
        // 1.0 + 1e-17 rounds to exactly 1.0 under round-to-nearest — below the true sum.
        let eps_a = 1.0_f64;
        let eps_b = 1e-17_f64;
        let bound_a = Bound {
            kind: BoundKind::Error {
                eps: eps_a,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::ProvenThm {
                citation: "t1".to_owned(),
            },
        };
        let bound_b = Bound {
            kind: BoundKind::Error {
                eps: eps_b,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::ProvenThm {
                citation: "t2".to_owned(),
            },
        };
        let a = Approx {
            value: 0.0_f64,
            bound: bound_a,
            strength: GuaranteeStrength::Proven,
        };
        let b = Approx {
            value: 0.0_f64,
            bound: bound_b,
            strength: GuaranteeStrength::Proven,
        };
        let combined = a.combine(&b, 0.0_f64, ErrorOp::Add).unwrap();
        let naive_sum = eps_a + eps_b; // Rounds down to 1.0 — omits eps_b.
        if let BoundKind::Error {
            eps: composed_eps, ..
        } = combined.bound.kind
        {
            assert!(
                composed_eps >= naive_sum,
                "outward-rounding: composed ε={composed_eps} must be ≥ naive sum {naive_sum}"
            );
            // The true sum is > 1.0 (eps_b is lost under round-to-nearest), so add_up must exceed 1.0.
            assert!(
                composed_eps > 1.0_f64,
                "outward-rounding: composed ε={composed_eps} must be > 1.0 (naive rounds down)"
            );
        } else {
            panic!("expected Error bound kind");
        }
    }

    #[test]
    fn map_preserves_bound_and_strength() {
        let b = mk_error_proven(0.5);
        let a = Approx {
            value: 42_i32,
            bound: b.clone(),
            strength: GuaranteeStrength::Proven,
        };
        let mapped = a.map(|v| v as f64);
        assert_eq!(mapped.value, 42.0_f64);
        assert_eq!(mapped.bound, b);
        assert_eq!(mapped.strength, GuaranteeStrength::Proven);
    }

    // ── FR-N3: Proven only via witness ────────────────────────────────────────

    // The compile-fail property is documented on the ProvenThm type (/// doc block).
    // The sealed token pattern is verified here at runtime by testing the only constructible path:
    #[test]
    fn proven_strength_only_via_witness_path() {
        // Empty citation → no witness → no Proven strength (type-level + runtime guard).
        assert!(ProvenThm::new("").is_none());
        // Valid citation → witness → Proven strength.
        let w = ProvenThm::new("ADR-010 §1").unwrap();
        let b = mk_error_proven(0.1);
        let a = Approx::proven(1.0_f64, b, w);
        assert_eq!(a.strength, GuaranteeStrength::Proven);
    }

    // ── NFR-N1: explain is total, never NaN ──────────────────────────────────

    #[test]
    fn explain_error_bound_is_total_and_never_nan() {
        let b = mk_error_proven(0.25);
        let a = Approx {
            value: 1.618_f64, // golden-ratio-ish test value
            bound: b,
            strength: GuaranteeStrength::Proven,
        };
        let ex = a.explain();
        assert!(!ex.summary.is_empty(), "explain summary must be non-empty");
        assert!(
            !ex.eps.is_nan(),
            "explain eps must never be NaN (NFR-N1 / C1)"
        );
        assert!(ex.eps >= 0.0, "explain eps must be non-negative");
        assert_eq!(ex.kind, "Error");
        assert_eq!(ex.norm, NormKind::Linf);
        assert!(
            ex.summary.contains("Proven"),
            "summary must mention the strength: {}",
            ex.summary
        );
    }

    #[test]
    fn explain_probability_bound_uses_infinity_not_nan() {
        let b = mk_prob_proven(0.05);
        let a = Approx {
            value: 1.0_f64,
            bound: b,
            strength: GuaranteeStrength::Proven,
        };
        let ex = a.explain();
        assert_eq!(
            ex.eps,
            f64::INFINITY,
            "non-Error bound must report eps=+∞, not NaN"
        );
        assert!(!ex.eps.is_nan(), "eps must never be NaN (NFR-N1 / C1)");
        assert_eq!(ex.kind, "Probability");
    }

    #[test]
    fn explain_declared_mentions_unverified_flag() {
        let b = mk_error_declared(0.1);
        let a = Approx::declared(1.0_f64, b);
        let ex = a.explain();
        assert!(
            ex.summary.contains("declared, unverified") || ex.summary.contains("UserDeclared"),
            "explain for Declared must surface the unverified flag (M-I4/VR-5): {}",
            ex.summary
        );
    }

    // ── Bound constructors ────────────────────────────────────────────────────

    #[test]
    fn error_bound_rejects_negative_eps() {
        // Mutation witness: removing the `eps < 0` check allows negative eps.
        assert_eq!(
            error_bound(-0.1, NormKind::Linf, BoundBasis::UserDeclared),
            Err(NumErr::BadEps)
        );
        assert_eq!(
            error_bound(f64::NEG_INFINITY, NormKind::Linf, BoundBasis::UserDeclared),
            Err(NumErr::BadEps)
        );
        assert_eq!(
            error_bound(f64::NAN, NormKind::Linf, BoundBasis::UserDeclared),
            Err(NumErr::BadEps)
        );
        assert_eq!(
            error_bound(f64::INFINITY, NormKind::Linf, BoundBasis::UserDeclared),
            Err(NumErr::BadEps)
        );
    }

    #[test]
    fn error_bound_accepts_zero_and_positive() {
        assert!(error_bound(0.0, NormKind::Linf, BoundBasis::UserDeclared).is_ok());
        assert!(error_bound(1e-10, NormKind::L2, BoundBasis::UserDeclared).is_ok());
    }

    #[test]
    fn prob_bound_rejects_out_of_range_delta() {
        // Mutation witness: removing the range check allows δ > 1 or δ < 0.
        assert_eq!(
            prob_bound(-0.01, BoundBasis::UserDeclared),
            Err(NumErr::BadDelta)
        );
        assert_eq!(
            prob_bound(1.01, BoundBasis::UserDeclared),
            Err(NumErr::BadDelta)
        );
        assert_eq!(
            prob_bound(f64::NAN, BoundBasis::UserDeclared),
            Err(NumErr::BadDelta)
        );
        assert_eq!(
            prob_bound(f64::INFINITY, BoundBasis::UserDeclared),
            Err(NumErr::BadDelta)
        );
    }

    #[test]
    fn prob_bound_accepts_valid_delta() {
        assert!(prob_bound(0.0, BoundBasis::UserDeclared).is_ok());
        assert!(prob_bound(0.5, BoundBasis::UserDeclared).is_ok());
        assert!(prob_bound(1.0, BoundBasis::UserDeclared).is_ok());
    }

    // ── union_delta ───────────────────────────────────────────────────────────

    #[test]
    fn union_delta_empty_returns_no_rule() {
        // Mutation witness: returning δ=0 instead of Err(NoRule) fabricates a bound.
        assert_eq!(union_delta(&[]), Err(NumErr::NoRule));
    }

    #[test]
    fn union_delta_sums_and_clamps() {
        let b1 = mk_prob_declared(0.4);
        let b2 = mk_prob_declared(0.7);
        let result = union_delta(&[&b1, &b2]).unwrap();
        // min(1, 0.4 + 0.7) = 1.0
        if let BoundKind::Probability { delta } = result.kind {
            assert!(
                (delta - 1.0).abs() < 1e-12,
                "union(0.4, 0.7) should clamp to 1.0, got {delta}"
            );
        } else {
            panic!("expected Probability kind");
        }
    }

    #[test]
    fn union_delta_meet_strength_proven_proven_is_proven() {
        let b1 = mk_prob_proven(0.1);
        let b2 = mk_prob_proven(0.2);
        let result = union_delta(&[&b1, &b2]).unwrap();
        assert_eq!(basis_strength(&result.basis), GuaranteeStrength::Proven);
    }

    #[test]
    fn union_delta_meet_strength_proven_declared_is_declared() {
        let b1 = mk_prob_proven(0.1);
        let b2 = mk_prob_declared(0.2);
        let result = union_delta(&[&b1, &b2]).unwrap();
        // Proven ∧ Declared = Declared (weakest-wins, M-204 / FR-N2).
        // Mutation witness: using max-rank (strongest-wins) produces Proven.
        assert_eq!(basis_strength(&result.basis), GuaranteeStrength::Declared);
    }

    #[test]
    fn union_delta_rejects_non_probability_bound() {
        let b = mk_error_declared(0.1);
        assert_eq!(union_delta(&[&b]), Err(NumErr::BadDelta));
    }

    // ── accuracy_to_probability ───────────────────────────────────────────────

    #[test]
    fn accuracy_to_prob_within_tau_returns_acc_delta() {
        // eps = 0.05 ≤ tau = 0.1 → result.delta = acc_delta = 0.0
        let acc = mk_error_proven(0.05);
        let result = accuracy_to_probability(&acc, 0.1, 0.0).unwrap();
        if let BoundKind::Probability { delta } = result.kind {
            assert!(
                delta.abs() < 1e-12,
                "eps ≤ tau: delta should be acc_delta=0.0, got {delta}"
            );
        } else {
            panic!("expected Probability kind");
        }
    }

    #[test]
    fn accuracy_to_prob_outside_tau_returns_worst_case() {
        // eps = 0.2 > tau = 0.1 → honest worst case δ = 1.0 (never a silent tighten).
        let acc = mk_error_proven(0.2);
        let result = accuracy_to_probability(&acc, 0.1, 0.0).unwrap();
        if let BoundKind::Probability { delta } = result.kind {
            assert!(
                (delta - 1.0).abs() < 1e-12,
                "eps > tau: delta should be worst case 1.0, got {delta}"
            );
        } else {
            panic!("expected Probability kind");
        }
    }

    #[test]
    fn accuracy_to_prob_bad_delta_rejected() {
        let acc = mk_error_proven(0.1);
        assert_eq!(
            accuracy_to_probability(&acc, 0.2, 1.5),
            Err(NumErr::BadDelta),
            "acc_delta=1.5 out of [0,1] must be Err(BadDelta)"
        );
    }

    #[test]
    fn accuracy_to_prob_non_error_bound_rejected() {
        let b = mk_prob_proven(0.1);
        assert_eq!(
            accuracy_to_probability(&b, 0.2, 0.0),
            Err(NumErr::NoRule),
            "non-Error bound must be Err(NoRule)"
        );
    }

    // ── check_error / check_union ─────────────────────────────────────────────

    #[test]
    fn check_error_valid_claim() {
        // eps(add) = 0.1 + 0.2 = 0.3; claiming 0.4 (looser) → Valid.
        let a = ErrorBound::new(0.1, NormKind::Linf).unwrap();
        let b = ErrorBound::new(0.2, NormKind::Linf).unwrap();
        let claimed = ErrorBound::new(0.4, NormKind::Linf).unwrap();
        assert!(check_error(&[a, b], ErrorOp::Add, claimed).is_ok());
    }

    #[test]
    fn check_error_too_tight_is_rejected() {
        // Claiming eps=0.1 for add([0.1, 0.2]) — true sum is 0.3 — must be Rejected.
        // Mutation witness: accepting any claim removes the never-silent posture.
        let a = ErrorBound::new(0.1, NormKind::Linf).unwrap();
        let b = ErrorBound::new(0.2, NormKind::Linf).unwrap();
        let claimed = ErrorBound::new(0.1, NormKind::Linf).unwrap();
        let result = check_error(&[a, b], ErrorOp::Add, claimed);
        assert!(
            matches!(result, Err(CheckErr::Rejected { .. })),
            "too-tight claim must be Rejected, got {result:?}"
        );
    }

    #[test]
    fn check_union_valid_claim() {
        let a = ProbBound::new(0.1).unwrap();
        let b = ProbBound::new(0.2).unwrap();
        let claimed = ProbBound::new(0.4).unwrap();
        assert!(check_union(&[a, b], claimed).is_ok());
    }

    #[test]
    fn check_union_too_tight_is_rejected() {
        // union(0.3, 0.4) = 0.7; claiming 0.2 is too tight.
        let a = ProbBound::new(0.3).unwrap();
        let b = ProbBound::new(0.4).unwrap();
        let claimed = ProbBound::new(0.2).unwrap();
        let result = check_union(&[a, b], claimed);
        assert!(
            matches!(result, Err(CheckErr::Rejected { .. })),
            "too-tight union claim must be Rejected"
        );
    }

    // ── NFR-N2: ε constants resolve to mycelium-numerics symbols ─────────────

    /// NFR-N2: ε constants must resolve to `mycelium-numerics` symbols (not restated here).
    ///
    /// This test verifies that the `combine` path routes through `mycelium-numerics`
    /// `compose_error_bound` by asserting that a composed bound passes the M-203 tier-i checker.
    /// If this crate restated its own ε constants or used plain `+`, the checker would disagree.
    ///
    /// Mutation witness: replacing `compose_error_bound` with a custom add that uses `+` (not
    /// `add_up`) may produce a composed bound that fails this checker call.
    #[test]
    fn eps_constants_resolve_to_mycelium_numerics_via_combine_and_check() {
        use mycelium_numerics::{check_error_claim, CheckOutcome};

        let bound_a = mk_error_proven(0.1);
        let bound_b = mk_error_proven(0.2);
        let approx_a = Approx {
            value: 1.0_f64,
            bound: bound_a,
            strength: GuaranteeStrength::Proven,
        };
        let approx_b = Approx {
            value: 2.0_f64,
            bound: bound_b,
            strength: GuaranteeStrength::Proven,
        };
        let combined = approx_a.combine(&approx_b, 3.0_f64, ErrorOp::Add).unwrap();

        if let BoundKind::Error {
            eps: composed_eps,
            norm,
        } = combined.bound.kind
        {
            let input_a = ErrorBound::new(0.1, NormKind::Linf).unwrap();
            let input_b = ErrorBound::new(0.2, NormKind::Linf).unwrap();
            let claimed = ErrorBound::new(composed_eps, norm).unwrap();
            // The kernel checker must accept the composed bound as valid (it was produced by the
            // kernel in the first place — this is a round-trip sanity check).
            assert_eq!(
                check_error_claim(&[input_a, input_b], ErrorOp::Add, claimed),
                CheckOutcome::Valid,
                "ε produced by combine must satisfy the M-203 tier-i checker"
            );
        } else {
            panic!("expected Error bound kind from combine");
        }
    }
}
