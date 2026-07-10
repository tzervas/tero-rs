//! The **`ErrorBound` (ε) kernel** — affine arithmetic (M-201; ADR-010 §1; RFC-0001 §4.7).
//!
//! ε-magnitude bounds compose through **affine arithmetic** (the ADR-010 §1 choice — sound,
//! compositional, *correlation-aware*; the Daisy/Rosa + FloVer lineage). The semantic domain is the
//! [`AffineForm`] `x̂ = x₀ + Σ xᵢ·εᵢ` over noise symbols `εᵢ ∈ [−1, +1]`; its **radius** `Σ|xᵢ|` is the
//! sound ε on the deviation from the central value. Linear ops (`add`/`sub`/`neg`/`scale`) are exact
//! in their *algebraic structure* (shared noise symbols cancel — the correlation awareness); `mul` is
//! nonlinear, so its second-order remainder is over-approximated onto a fresh noise symbol.
//!
//! **The compositions are computed in f64, and a `Proven` ε that travels in a bound must be a *true*
//! upper bound** — so every bound-increasing quantity is rounded **outward** (the `crate::round`
//! helpers), and each operation folds the magnitude of its own floating-point round-off into the
//! reserved [`ROUNDOFF_SYM`] term so [`AffineForm::radius`] stays a sound enclosure under f64
//! arithmetic (WS1; A2-01). An operation that is exact in f64 (e.g. integer sums, power-of-two
//! scales) adds no round-off term, so an exact composition stays exact.
//!
//! The scalar [`ErrorBound`] `{eps, norm}` is the projection that travels in a [`mycelium_core::Bound`]
//! (`BoundKind::Error`). Its compositions are the *conservative* (worst-case, correlation-free)
//! projections used when only the magnitudes — not the affine structure — are available (the
//! interpreter's case, M-204), likewise outward-rounded. All three normative composition properties
//! hold: **Soundness** (the composed `eps` is a true upper bound on the deviation, including f64
//! round-off), **Monotonicity** (never tighter than the inputs justify), **Determinism** (identical
//! inputs → identical `eps`).

use std::collections::BTreeMap;

use mycelium_core::NormKind;

use crate::round;

/// A noise-symbol identifier. Distinct symbols model *independent* uncertainty sources; a shared
/// symbol models a *correlated* one (the affine-arithmetic advantage over interval arithmetic).
pub type NoiseSym = u64;

/// The reserved noise symbol carrying the **accumulated floating-point round-off of the affine
/// operations themselves** (WS1; A2-01). Each op folds the magnitude of its own center/coefficient
/// rounding here, so [`AffineForm::radius`] stays a sound enclosure under f64 rounding. It is never a
/// user symbol — callers must not pass `u64::MAX` to [`AffineForm::uncertain`] or as a `mul` fresh
/// symbol.
pub const ROUNDOFF_SYM: NoiseSym = u64::MAX;

/// An affine form `x₀ + Σ xᵢ·εᵢ` with noise symbols `εᵢ ∈ [−1, +1]` (ADR-010 §1). The
/// **concretization** is the interval `[x₀ − radius, x₀ + radius]` with `radius = Σ|xᵢ|`; linear
/// operations are *exact* functions of the shared assignment, so correlated uncertainty cancels.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineForm {
    center: f64,
    /// Noise terms `symbol → coefficient`. Zero coefficients are never stored, so equality is the
    /// mathematical equality of forms.
    terms: BTreeMap<NoiseSym, f64>,
}

impl AffineForm {
    /// The exact constant `c` (no uncertainty; `radius == 0`).
    #[must_use]
    pub fn constant(center: f64) -> Self {
        AffineForm {
            center,
            terms: BTreeMap::new(),
        }
    }

    /// `center ± radius` carried on a single noise symbol `sym`, or `None` if `center` is non-finite,
    /// `radius` is non-finite, or `radius` is negative — an out-of-range uncertainty is an explicit
    /// refusal, **never** a silent collapse to an exact form (house rule 2; A2-03). A `radius` of
    /// exactly `0` is the [`constant`](Self::constant) (no term stored). `sym` must not be
    /// [`ROUNDOFF_SYM`].
    #[must_use]
    pub fn uncertain(center: f64, sym: NoiseSym, radius: f64) -> Option<Self> {
        if !center.is_finite() || !radius.is_finite() || radius < 0.0 {
            return None;
        }
        debug_assert_ne!(
            sym, ROUNDOFF_SYM,
            "ROUNDOFF_SYM is reserved for accumulated round-off"
        );
        let mut terms = BTreeMap::new();
        if radius > 0.0 {
            terms.insert(sym, radius);
        }
        Some(AffineForm { center, terms })
    }

    /// The central value `x₀`.
    #[must_use]
    pub fn center(&self) -> f64 {
        self.center
    }

    /// The total deviation `radius = Σ|xᵢ|` — the sound ε on `|value − center|` (ADR-010 §1). The sum
    /// is accumulated with **outward rounding** (A2-01), so the returned radius is never below the
    /// real Σ|xᵢ|; it includes the [`ROUNDOFF_SYM`] term carrying the operations' own round-off.
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.terms
            .values()
            .map(|c| c.abs())
            .fold(0.0, round::add_up)
    }

    /// Evaluate the form at a noise assignment `ε(sym) ∈ [−1, +1]`. Used to *check* soundness: for a
    /// linear op this equals the corresponding real operation exactly (the soundness property test).
    #[must_use]
    pub fn eval(&self, assign: impl Fn(NoiseSym) -> f64) -> f64 {
        self.center
            + self
                .terms
                .iter()
                .map(|(sym, coeff)| coeff * assign(*sym))
                .sum::<f64>()
    }

    /// Insert/accumulate a term, dropping it if the coefficient lands on exactly zero (keeps forms
    /// canonical, so structurally-equal forms are `==`).
    fn accumulate(terms: &mut BTreeMap<NoiseSym, f64>, sym: NoiseSym, coeff: f64) {
        if coeff == 0.0 {
            return;
        }
        let entry = terms.entry(sym).or_insert(0.0);
        *entry += coeff;
        if *entry == 0.0 {
            terms.remove(&sym);
        }
    }

    /// Negation — exact (`−x̂`).
    #[must_use]
    pub fn neg(&self) -> Self {
        AffineForm {
            center: -self.center,
            terms: self.terms.iter().map(|(s, c)| (*s, -c)).collect(),
        }
    }

    /// Addition — *exact* on the form's structure (shared noise symbols combine, so correlated
    /// uncertainty cancels), with the new center's own f64 round-off folded into [`ROUNDOFF_SYM`] so
    /// the concretization stays a sound enclosure (A2-01). When the center sum is exact the round-off
    /// is `0` and no term is added.
    #[must_use]
    pub fn add(&self, other: &AffineForm) -> Self {
        let mut terms = self.terms.clone();
        for (sym, coeff) in &other.terms {
            Self::accumulate(&mut terms, *sym, *coeff);
        }
        Self::accumulate(
            &mut terms,
            ROUNDOFF_SYM,
            round::add_err(self.center, other.center).abs(),
        );
        AffineForm {
            center: self.center + other.center,
            terms,
        }
    }

    /// Subtraction — exact (`x̂ − ŷ`); `x̂ − x̂ == 0` with `radius 0` (the correlation advantage).
    #[must_use]
    pub fn sub(&self, other: &AffineForm) -> Self {
        self.add(&other.neg())
    }

    /// Scaling by a constant `c` (`c·x̂`), with the round-off of the center and each scaled
    /// coefficient folded into [`ROUNDOFF_SYM`] (A2-01). Exact (no round-off term) when every product
    /// is exact — e.g. scaling by a power of two.
    #[must_use]
    pub fn scale(&self, c: f64) -> Self {
        let mut terms = BTreeMap::new();
        let mut roundoff = round::mul_err(c, self.center).abs();
        for (sym, coeff) in &self.terms {
            roundoff = round::add_up(roundoff, round::mul_err(c, *coeff).abs());
            Self::accumulate(&mut terms, *sym, c * coeff);
        }
        Self::accumulate(&mut terms, ROUNDOFF_SYM, roundoff);
        AffineForm {
            center: c * self.center,
            terms,
        }
    }

    /// Multiplication — *nonlinear*. The linear part `x₀ŷ + y₀x̂` is kept exactly; the second-order
    /// remainder (magnitude `≤ radius(x)·radius(y)`) is over-approximated onto a **fresh** noise
    /// symbol `fresh` (standard affine multiplication, ADR-010 §1). Sound: the true product lies in
    /// `[center ± radius]` for every noise assignment. `fresh` must not already appear in either
    /// operand.
    #[must_use]
    pub fn mul(&self, other: &AffineForm, fresh: NoiseSym) -> Self {
        debug_assert!(
            !self.terms.contains_key(&fresh) && !other.terms.contains_key(&fresh),
            "AffineForm::mul: fresh symbol {fresh} already appears in an operand (A2-06)"
        );
        debug_assert_ne!(fresh, ROUNDOFF_SYM, "fresh symbol must not be ROUNDOFF_SYM");
        let mut terms: BTreeMap<NoiseSym, f64> = BTreeMap::new();
        // Linear part: x0·(other terms) + y0·(self terms), tracking each product's round-off.
        let mut roundoff = round::mul_err(self.center, other.center).abs();
        for (sym, coeff) in &other.terms {
            roundoff = round::add_up(roundoff, round::mul_err(self.center, *coeff).abs());
            Self::accumulate(&mut terms, *sym, self.center * coeff);
        }
        for (sym, coeff) in &self.terms {
            roundoff = round::add_up(roundoff, round::mul_err(other.center, *coeff).abs());
            Self::accumulate(&mut terms, *sym, other.center * coeff);
        }
        // Second-order remainder (outward-rounded) onto the caller's fresh symbol; the linear
        // products' own f64 round-off goes to the reserved channel so the form stays sound (A2-01).
        let remainder = round::mul_up(self.radius(), other.radius());
        Self::accumulate(&mut terms, fresh, remainder);
        Self::accumulate(&mut terms, ROUNDOFF_SYM, roundoff);
        AffineForm {
            center: self.center * other.center,
            terms,
        }
    }
}

/// A scalar ε-magnitude bound `{eps ≥ 0, norm}` — the [`AffineForm::radius`] projection that travels
/// in a [`mycelium_core::Bound`] (`BoundKind::Error`). Compositions are the conservative (worst-case)
/// projections used when only magnitudes are available. Fields are **private** so a bound can only be
/// built via [`new`](Self::new)/[`exact`](Self::exact) or the outward-rounded compositions — a tight,
/// under-rounded `eps` cannot be injected by direct construction (A2-05; keeps A2-01 structural).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorBound {
    // `pub(crate)`: the kernel's own modules compose these directly, but **external** crates cannot
    // name the fields — so a tight, under-rounded bound cannot be injected by struct construction;
    // they must go through `new`/`exact`/the outward-rounded compositions, or the accessors (A2-05).
    pub(crate) eps: f64,
    pub(crate) norm: NormKind,
}

impl ErrorBound {
    /// The error magnitude (`>= 0`, finite).
    #[must_use]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The norm `eps` is measured in.
    #[must_use]
    pub fn norm(&self) -> NormKind {
        self.norm
    }

    /// An exact bound (`eps == 0`) in `norm` — the identity of [`add`](Self::add).
    #[must_use]
    pub const fn exact(norm: NormKind) -> Self {
        ErrorBound { eps: 0.0, norm }
    }

    /// A well-formed bound, or `None` if `eps` is negative or non-finite (never a silent coercion).
    #[must_use]
    pub fn new(eps: f64, norm: NormKind) -> Option<Self> {
        (eps.is_finite() && eps >= 0.0).then_some(ErrorBound { eps, norm })
    }

    /// `eps(x ± y) = eps(x) + eps(y)` — the affine-`add` projection (1-Lipschitz; ℓ-norm triangle
    /// inequality). Sound and monotone; the sum is **rounded outward** so the composed bound is never
    /// below the real `eps(x) + eps(y)` (A2-01). `None` if the norms differ (no silent norm coercion).
    #[must_use]
    pub fn add(&self, other: &ErrorBound) -> Option<Self> {
        (self.norm == other.norm).then_some(ErrorBound {
            eps: round::add_up(self.eps, other.eps),
            norm: self.norm,
        })
    }

    /// `eps(x − y) = eps(x) + eps(y)` — identical magnitude composition to [`add`](Self::add).
    #[must_use]
    pub fn sub(&self, other: &ErrorBound) -> Option<Self> {
        self.add(other)
    }

    /// `eps(−x) = eps(x)` — negation does not change the magnitude.
    #[must_use]
    pub fn neg(&self) -> Self {
        *self
    }

    /// `eps(c·x) = |c|·eps(x)` — scaling, **rounded outward** (A2-01).
    #[must_use]
    pub fn scale(&self, c: f64) -> Self {
        ErrorBound {
            eps: round::mul_up(c.abs(), self.eps),
            norm: self.norm,
        }
    }

    /// `eps(x·y) ≤ |x₀|·eps(y) + |y₀|·eps(x) + eps(x)·eps(y)` — sound first-order error propagation
    /// for a product about central magnitudes `x0_mag = |x₀|`, `y0_mag = |y₀|`. Every product and sum
    /// is **rounded outward** so the composed bound stays a true upper bound (A2-01). `None` if the
    /// norms differ.
    #[must_use]
    pub fn mul(&self, other: &ErrorBound, x0_mag: f64, y0_mag: f64) -> Option<Self> {
        if self.norm != other.norm {
            return None;
        }
        let xy = round::mul_up(x0_mag.abs(), other.eps);
        let yx = round::mul_up(y0_mag.abs(), self.eps);
        let ee = round::mul_up(self.eps, other.eps);
        Some(ErrorBound {
            eps: round::add_up(round::add_up(xy, yx), ee),
            norm: self.norm,
        })
    }
}
