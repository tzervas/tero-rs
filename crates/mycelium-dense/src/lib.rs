//! `mycelium-dense` — the **Dense paradigm operational surface** (M-230; RFC-0001 §4.1;
//! RFC-0002 §5): typed, dimension-tracked `Dense{dim, dtype}` values and elementwise/embedding
//! ops — the Dense analogue of the VSA `VsaModel` surface.
//!
//! Dimension and dtype are part of the type ([`DenseSpace`] binds both); a mismatch is a typed
//! error, never a silent broadcast or coercion (G2). Per the honesty rule, every op carries an
//! honest per-op tag ([`DenseSpace::op_guarantee`]):
//!
//! - **`neg`** is **`Exact`** — negation never rounds (the dtype grids are symmetric).
//! - **`add`/`sub`/`scale`** are **`Proven`**, carrying a per-element relative ε
//!   ([`Bound`] `Error{eps, Rel}`) with a `ProvenThm` basis: the standard round-to-nearest
//!   relative-error theorem (Higham 2002, Thm 2.2 — the same basis as the M-211 `F32→BF16`
//!   swap), with its side-conditions **checked per element** (exact on-grid inputs; finite,
//!   zero-or-normal, non-overflowing results). A violated side-condition is an explicit
//!   [`DenseError`], never a bound the theorem does not cover (VR-5).
//! - **`dot`/`similarity`** are *measurement helpers* returning bare `f64` (no `Meta` to tag),
//!   mirroring `VsaModel::similarity`. Their value-returning twins **`dot_value`/
//!   `similarity_value`** (M-891) deliver the same `f64` as a `Dense{1, F64}` result carrying a
//!   **`Proven` absolute (ℓ∞) bound**: over exact on-grid F32/BF16 operands every product is
//!   *exact* in the binary64 accumulator (≤ 24-bit significands ⇒ ≤ 48-bit products), so the
//!   only error is the recursive-summation rounding — bounded by the standard `γ`-analysis
//!   (Higham 2002, §4.2), disclosed via [`DenseSpace::dot_abs_eps`] /
//!   [`DenseSpace::similarity_abs_eps`]. Note this is deliberately **not** [`DenseSpace::op_rel_eps`]:
//!   the dtype's per-element rounding ε never enters (inputs are exact, accumulation is f64), and a
//!   per-element *relative* claim on a dot product is false under cancellation — the honest bound
//!   is absolute and dimension-dependent (VR-5).
//!
//! **Honest scope (v1, same as M-211).** Sources must be `Exact`: composing an approximate
//! input's own bound with the op's rounding ε needs the magnitude-aware Dense composition rule
//! that is still open (recorded at M-204/M-211) — refused explicitly via
//! [`DenseError::ApproximateSource`], never fabricated. `F16`/`F64` dtypes are explicitly
//! unsupported (`F64` ops cannot be re-derived against an exact reference in `f64`; `F16` lands
//! with a use case), and subnormal results are refused (outside the cited theorem's
//! side-conditions).

use mycelium_core::{
    operation_hash, Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, NormKind,
    Payload, Provenance, Repr, ScalarKind, Value, WfError,
};

/// Single-rounding relative bound for native `f32` ops: the unit roundoff `u = β^(1−p)/2 = 2^−24`
/// for IEEE binary32 (`p = 24`) under round-to-nearest (Higham 2002, Thm 2.2).
pub const F32_OP_REL_EPS: f64 = 5.960_464_477_539_063e-8; // 2⁻²⁴, exact in f64

/// Two-rounding relative bound for BF16 ops: the op is computed as a native `f32` op
/// (`u₁ = 2^−24`) and rounded to the bfloat16 grid (`u₂ = 2^−8`); the composition
/// `(1+δ₁)(1+δ₂) − 1 ≤ u₁ + u₂ + u₁u₂ ≤ 2^−8 + 2^−23` (the slack absorbs the cross term).
pub const BF16_OP_REL_EPS: f64 = 0.003_906_25 + 1.192_092_895_507_812_5e-7; // 2⁻⁸ + 2⁻²³

/// Smallest positive *normal* magnitude on both the `f32` and bfloat16 grids (`2^−126` — bf16
/// keeps f32's exponent range). Below it the relative-error theorem's side-condition fails.
pub const DENSE_MIN_NORMAL: f64 = f32::MIN_POSITIVE as f64;

/// Unit roundoff of the binary64 accumulator the measurement ops (`dot`/`similarity`) sum in:
/// `u = β^(1−p)/2 = 2^−53` for IEEE binary64 (`p = 53`) under round-to-nearest (Higham 2002,
/// Thm 2.2) — the ε unit of [`DenseSpace::dot_abs_eps`]/[`DenseSpace::similarity_abs_eps`].
pub const F64_ACC_U: f64 = 1.110_223_024_625_156_5e-16; // 2⁻⁵³, exact in f64 (pinned by test)

const F32_OP_CITATION: &str = "round-to-nearest relative error ≤ u = β^(1−p)/2 = 2^−24 for IEEE \
     binary32 (β=2, p=24) — Higham, Accuracy and Stability of Numerical Algorithms (2002), Thm 2.2; \
     native f32 op (single rounding); side-conditions checked per element: exact on-grid inputs, \
     finite zero-or-normal result, no overflow";

const BF16_OP_CITATION: &str = "two-rounding composition (1+δ₁)(1+δ₂)−1 ≤ 2^−8 + 2^−23: native f32 \
     op (u₁ = 2^−24) then bfloat16 round-to-nearest (u₂ = 2^−8) — Higham (2002), Thm 2.2 applied at \
     each rounding; side-conditions checked per element at both steps: exact on-grid inputs, finite \
     zero-or-normal results, no overflow";

const DOT_CITATION: &str = "recursive binary64 summation of exact products: each x_i*y_i is exact \
     in f64 (operands checked exact on the F32/BF16 grid, so significands are <= 24 bits — products \
     need <= 48 <= 53 — and product exponents lie in binary64's normal range), and the (dim−1)-add \
     recursive sum satisfies |fl(Σ) − Σ| <= γ_{dim−1}·Σ|x_i·y_i| with γ_k = k·u/(1−k·u), u = 2^−53 \
     — Higham, Accuracy and Stability of Numerical Algorithms (2002), §4.2 + Thm 2.2; additions \
     that underflow are exact (Hauser), so no subnormal side-condition arises. The stored ε = \
     1.05·dim·u·Σ̂ (Σ̂ = the computed abs-product sum) over-covers γ_{dim−1}·Σ|x_i·y_i| for every \
     dim < 2^32: the ×1.05 slack absorbs Σ̂'s own summation rounding and the ε-formula's evaluation \
     rounding. Side-conditions checked: per-element (exact on-grid finite inputs) and on the result \
     (finite — overflow is unreachable for dim < 2^32 since every |product| < 2^257)";

const SIMILARITY_CITATION: &str = "cosine dot/(‖a‖·‖b‖) in binary64 over exact products (see \
     dense.dot): |computed − true| <= 2.1·(dim+2)·u, u = 2^−53 — first-order sum of the numerator \
     term γ_{dim−1} (|fl(Σx_i·y_i) − Σ| <= γ_{dim−1}·Σ|x_i·y_i| <= γ_{dim−1}·‖a‖·‖b‖ by \
     Cauchy–Schwarz) and the denominator/quotient term (two positive-term norm sums + sqrt + \
     product + quotient roundings, <= (dim+3)·u to first order) times |cos| <= 1 — Higham (2002) \
     §4.2 + Thm 2.2; the slack in 2.1·(dim+2) over the first-order (2·dim+2)·u absorbs all \
     second-order γ-terms for dim < 2^32 and the ε-formula's own rounding. A zero-norm operand \
     returns the documented convention 0 exactly (an operand norm is 0 iff that operand is the \
     zero vector: squares of nonzero on-grid elements are >= 2^−298, so they cannot underflow to 0 \
     in f64). Side-conditions checked: per-element (exact on-grid finite inputs) and on the result \
     (finite)";

/// The Dense operations this surface supplies (RFC-0001 §4.1 — the Dense analogue of
/// [`VsaOp`](https://docs.rs/mycelium-vsa)'s closed op set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseOp {
    /// Elementwise addition.
    Add,
    /// Elementwise subtraction.
    Sub,
    /// Elementwise negation.
    Neg,
    /// Scalar multiplication.
    Scale,
    /// Dot-product measurement (M-891).
    Dot,
    /// Cosine-similarity measurement (M-891).
    Similarity,
}

/// Why a Dense operation could not be performed — always explicit, never a silent coercion (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseError {
    /// Operand dimensionality disagrees with the space's `dim` — a type error (M-230 acceptance).
    DimMismatch {
        /// Expected dimensionality.
        expected: u32,
        /// Actual dimensionality.
        got: u32,
    },
    /// Operand dtype disagrees with the space's `dtype` — a type error, never re-rounded.
    DtypeMismatch {
        /// The dtype the space expected.
        expected: ScalarKind,
    },
    /// The dtype has no supported op set in v1 (`F16`/`F64` — honest scope, see crate docs).
    UnsupportedDtype {
        /// The unsupported dtype.
        dtype: ScalarKind,
    },
    /// The value handed in is not a `Dense` value at all.
    NotDense,
    /// An element is NaN/±Inf — no rounding bound is defined for it.
    NonFinite {
        /// Index of the offending element.
        index: usize,
    },
    /// An element is not exactly representable on the declared dtype grid — the payload
    /// contradicts its own representation; refused, never silently re-rounded.
    NotOnGrid {
        /// Index of the offending element.
        index: usize,
    },
    /// The scale factor is non-finite or off the dtype grid (same contract as the elements).
    ScalarOffGrid,
    /// A result element is subnormal — outside the cited theorem's side-conditions (v1 scope,
    /// same honest refusal as M-211).
    SubnormalUnsupported {
        /// Index of the offending element.
        index: usize,
    },
    /// A result element overflows the dtype's finite range — explicit, never a silent ±Inf.
    Overflow {
        /// Index of the offending element.
        index: usize,
    },
    /// The source value is itself approximate; composing its bound with the op's rounding ε is
    /// not a defined rule yet (recorded at M-204/M-211) — refused, never fabricated.
    ApproximateSource,
    /// A constructed result violated a Core IR invariant.
    Wf(WfError),
}

impl core::fmt::Display for DenseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DenseError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            DenseError::DtypeMismatch { expected } => {
                write!(f, "dtype mismatch: expected {expected:?}")
            }
            DenseError::UnsupportedDtype { dtype } => {
                write!(
                    f,
                    "dtype {dtype:?} has no supported Dense op set (M-230 v1 scope)"
                )
            }
            DenseError::NotDense => write!(f, "expected a Dense value"),
            DenseError::NonFinite { index } => {
                write!(f, "element {index} is NaN/Inf — no defined rounding bound")
            }
            DenseError::NotOnGrid { index } => {
                write!(f, "element {index} is not on the declared dtype grid")
            }
            DenseError::ScalarOffGrid => {
                write!(f, "scale factor is non-finite or off the dtype grid")
            }
            DenseError::SubnormalUnsupported { index } => write!(
                f,
                "result element {index} is subnormal — outside the proven relative-bound range"
            ),
            DenseError::Overflow { index } => {
                write!(
                    f,
                    "result element {index} overflows the dtype's finite range"
                )
            }
            DenseError::ApproximateSource => write!(
                f,
                "source is approximate; composing its bound with the op ε is not a defined rule yet"
            ),
            DenseError::Wf(e) => write!(f, "well-formedness violation: {e}"),
        }
    }
}

impl std::error::Error for DenseError {}

/// Round an `f32` to the nearest bfloat16 (ties to even), widened back to `f32` bit-exactly —
/// the same grid the M-211 swap targets. Caller has excluded NaN/Inf.
fn round_f32_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16)
}

/// Whether `x` is exactly representable on the `dtype` grid (finite values only).
fn on_grid(dtype: ScalarKind, x: f64) -> bool {
    #[allow(clippy::cast_possible_truncation)] // representability is exactly what we check
    let xf = x as f32;
    if f64::from(xf) != x {
        return false;
    }
    match dtype {
        ScalarKind::F32 => true,
        ScalarKind::Bf16 => round_f32_to_bf16(xf) == xf,
        // Unreachable behind DenseSpace::new's dtype gate; conservatively off-grid.
        ScalarKind::F16 | ScalarKind::F64 => false,
    }
}

/// A typed Dense space: every value it constructs or operates on has exactly this `dim` and
/// `dtype` (dim-in-the-type, M-230 acceptance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseSpace {
    /// Dimensionality.
    pub dim: u32,
    /// Element dtype (`F32` or `BF16` in v1).
    pub dtype: ScalarKind,
}

impl DenseSpace {
    /// A Dense space of `dim`-vectors over `dtype`. `F16`/`F64` are an explicit
    /// [`DenseError::UnsupportedDtype`] (v1 scope — see crate docs).
    pub fn new(dim: u32, dtype: ScalarKind) -> Result<Self, DenseError> {
        match dtype {
            ScalarKind::F32 | ScalarKind::Bf16 => Ok(DenseSpace { dim, dtype }),
            ScalarKind::F16 | ScalarKind::F64 => Err(DenseError::UnsupportedDtype { dtype }),
        }
    }

    /// The `Repr` of this space's values.
    #[must_use]
    pub fn repr(&self) -> Repr {
        Repr::Dense {
            dim: self.dim,
            dtype: self.dtype,
        }
    }

    /// The honest intrinsic guarantee per op: `neg` never rounds (`Exact`); `add`/`sub`/`scale`
    /// round once (or twice for BF16) under the cited theorem (`Proven`); `dot`/`similarity`
    /// carry the proven binary64 accumulation bound (M-891 — see [`Self::dot_abs_eps`]/
    /// [`Self::similarity_abs_eps`]; `Proven`, never `Exact`: the recursive sum rounds).
    #[must_use]
    pub fn op_guarantee(op: DenseOp) -> GuaranteeStrength {
        match op {
            DenseOp::Neg => GuaranteeStrength::Exact,
            DenseOp::Add | DenseOp::Sub | DenseOp::Scale | DenseOp::Dot | DenseOp::Similarity => {
                GuaranteeStrength::Proven
            }
        }
    }

    /// The per-element relative ε this space's rounding ops carry.
    #[must_use]
    pub fn op_rel_eps(&self) -> f64 {
        match self.dtype {
            ScalarKind::Bf16 => BF16_OP_REL_EPS,
            // `new` admits only F32 | Bf16; F32 is the remaining case.
            _ => F32_OP_REL_EPS,
        }
    }

    fn op_citation(&self) -> &'static str {
        match self.dtype {
            ScalarKind::Bf16 => BF16_OP_CITATION,
            _ => F32_OP_CITATION,
        }
    }

    /// The disclosed **absolute** (ℓ∞ over the single result element) error bound of
    /// [`Self::dot_value`], given `abs_sum` = the computed `fl(Σ|xᵢ·yᵢ|)`:
    /// `1.05 · dim · u · abs_sum` with `u =` [`F64_ACC_U`] — a conservative cover of the
    /// summation bound `γ_{dim−1}·Σ|xᵢ·yᵢ|` (products are exact; see [`DOT_CITATION`] via the
    /// result's `ProvenThm` basis). Deliberately **not** [`Self::op_rel_eps`]: the dtype's
    /// per-element rounding ε never enters (inputs are exact on-grid, accumulation is binary64),
    /// and a *relative* claim on a dot product is false under cancellation (VR-5).
    #[must_use]
    pub fn dot_abs_eps(&self, abs_sum: f64) -> f64 {
        1.05 * f64::from(self.dim) * F64_ACC_U * abs_sum
    }

    /// The disclosed **absolute** error bound of [`Self::similarity_value`]:
    /// `2.1 · (dim + 2) · u` with `u =` [`F64_ACC_U`] — a conservative cover of the first-order
    /// bound `(2·dim + 2)·u` (numerator summation over exact products, via Cauchy–Schwarz, plus
    /// the norm/sqrt/quotient roundings against `|cos| ≤ 1`; see the result's `ProvenThm`
    /// citation). Input-independent: cosine's normalization caps the absolute error by
    /// construction.
    #[must_use]
    pub fn similarity_abs_eps(&self) -> f64 {
        2.1 * (f64::from(self.dim) + 2.0) * F64_ACC_U
    }

    /// Construct an **`Exact`** Dense value, checking every element is finite and exactly on the
    /// dtype grid — an off-grid payload would contradict its own `Repr` (refused, never
    /// re-rounded).
    pub fn value(&self, xs: Vec<f64>) -> Result<Value, DenseError> {
        self.check_elements(&xs)?;
        Value::new(
            self.repr(),
            Payload::Scalars(xs),
            Meta::exact(Provenance::Root),
        )
        .map_err(DenseError::Wf)
    }

    fn check_elements(&self, xs: &[f64]) -> Result<(), DenseError> {
        if xs.len() != self.dim as usize {
            return Err(DenseError::DimMismatch {
                expected: self.dim,
                got: u32::try_from(xs.len()).unwrap_or(u32::MAX),
            });
        }
        for (index, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(DenseError::NonFinite { index });
            }
            if !on_grid(self.dtype, x) {
                return Err(DenseError::NotOnGrid { index });
            }
        }
        Ok(())
    }

    /// Extract the scalar payload of a value belonging to this space, re-checking the contract
    /// the ops rely on (right space, `Exact` source, on-grid elements).
    fn scalars_of<'a>(&self, v: &'a Value) -> Result<&'a [f64], DenseError> {
        let Repr::Dense { dim, dtype } = *v.repr() else {
            return Err(DenseError::NotDense);
        };
        if dtype != self.dtype {
            return Err(DenseError::DtypeMismatch {
                expected: self.dtype,
            });
        }
        if dim != self.dim {
            return Err(DenseError::DimMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        if v.meta().guarantee() != GuaranteeStrength::Exact {
            return Err(DenseError::ApproximateSource);
        }
        let Payload::Scalars(xs) = v.payload() else {
            return Err(DenseError::NotDense);
        };
        self.check_elements(xs)?;
        Ok(xs)
    }

    /// One elementwise result under the theorem's checked side-conditions: compute as a native
    /// `f32` op, round to the dtype grid, and refuse (explicitly) any element the cited bound
    /// does not cover.
    fn round_result(&self, y32: f32, index: usize) -> Result<f64, DenseError> {
        if !y32.is_finite() {
            return Err(DenseError::Overflow { index });
        }
        if y32 != 0.0 && f64::from(y32.abs()) < DENSE_MIN_NORMAL {
            return Err(DenseError::SubnormalUnsupported { index });
        }
        let rounded = match self.dtype {
            ScalarKind::Bf16 => {
                let r = round_f32_to_bf16(y32);
                if !r.is_finite() {
                    return Err(DenseError::Overflow { index });
                }
                if r != 0.0 && f64::from(r.abs()) < DENSE_MIN_NORMAL {
                    return Err(DenseError::SubnormalUnsupported { index });
                }
                r
            }
            _ => y32,
        };
        Ok(f64::from(rounded))
    }

    /// Wrap a rounded result with its honest `Proven` rounding bound (M-I2: the basis is the
    /// checked theorem instantiation, never an assertion).
    fn wrap_proven(
        &self,
        data: Vec<f64>,
        op: &str,
        inputs: Vec<ContentHash>,
    ) -> Result<Value, DenseError> {
        let bound = Bound {
            kind: BoundKind::Error {
                eps: self.op_rel_eps(),
                norm: NormKind::Rel,
            },
            basis: BoundBasis::ProvenThm {
                citation: self.op_citation().to_owned(),
            },
        };
        let meta = Meta::new(
            Provenance::Derived {
                op: operation_hash(op),
                inputs,
            },
            GuaranteeStrength::Proven,
            Some(bound),
            None,
            None,
            None,
        )
        .map_err(DenseError::Wf)?;
        Value::new(self.repr(), Payload::Scalars(data), meta).map_err(DenseError::Wf)
    }

    /// Elementwise `a + b` (**`Proven`**, per-element relative ε — see crate docs).
    pub fn add_values(&self, a: &Value, b: &Value) -> Result<Value, DenseError> {
        self.elementwise(a, b, "dense.add", |x, y| x + y)
    }

    /// Elementwise `a − b` (**`Proven`**, same bound as `add`).
    pub fn sub_values(&self, a: &Value, b: &Value) -> Result<Value, DenseError> {
        self.elementwise(a, b, "dense.sub", |x, y| x - y)
    }

    fn elementwise(
        &self,
        a: &Value,
        b: &Value,
        op: &str,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<Value, DenseError> {
        let xs = self.scalars_of(a)?;
        let ys = self.scalars_of(b)?;
        let mut out = Vec::with_capacity(xs.len());
        for (index, (&x, &y)) in xs.iter().zip(ys).enumerate() {
            // On-grid checked above, so the f32 narrowing is exact.
            #[allow(clippy::cast_possible_truncation)]
            let y32 = f(x as f32, y as f32);
            out.push(self.round_result(y32, index)?);
        }
        self.wrap_proven(out, op, vec![a.content_hash(), b.content_hash()])
    }

    /// Elementwise negation (**`Exact`** — the grids are symmetric, so no element ever rounds).
    pub fn neg_value(&self, a: &Value) -> Result<Value, DenseError> {
        let xs = self.scalars_of(a)?;
        let out: Vec<f64> = xs.iter().map(|&x| -x).collect();
        let meta = Meta::new(
            Provenance::Derived {
                op: operation_hash("dense.neg"),
                inputs: vec![a.content_hash()],
            },
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .map_err(DenseError::Wf)?;
        Value::new(self.repr(), Payload::Scalars(out), meta).map_err(DenseError::Wf)
    }

    /// Scalar multiplication `c · a` (**`Proven`**). `c` must be finite and on the dtype grid —
    /// the same contract as the elements (else [`DenseError::ScalarOffGrid`]).
    pub fn scale_value(&self, a: &Value, c: f64) -> Result<Value, DenseError> {
        if !c.is_finite() || !on_grid(self.dtype, c) {
            return Err(DenseError::ScalarOffGrid);
        }
        let xs = self.scalars_of(a)?;
        let mut out = Vec::with_capacity(xs.len());
        for (index, &x) in xs.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)] // on-grid checked: narrowing is exact
            let y32 = (c as f32) * (x as f32);
            out.push(self.round_result(y32, index)?);
        }
        self.wrap_proven(out, "dense.scale", vec![a.content_hash()])
    }

    /// The shared measurement accumulation: the recursive (left-to-right) binary64 sum of the
    /// products and of their absolute values. Every product is **exact** in f64 (both factors
    /// are on the F32/BF16 grid — `scalars_of` has checked them), so the returned sums carry
    /// only the recursive-summation rounding the `DOT_CITATION` bound covers.
    fn dot_sums(xs: &[f64], ys: &[f64]) -> (f64, f64) {
        let mut sum = 0.0;
        let mut abs_sum = 0.0;
        for (&x, &y) in xs.iter().zip(ys) {
            let p = x * y; // exact in f64: ≤ 24-bit significands, normal-range exponents
            sum += p;
            abs_sum += p.abs();
        }
        (sum, abs_sum)
    }

    /// The shared cosine computation (`0` by documented convention if either norm is `0` — which
    /// happens iff that operand is exactly the zero vector; see `SIMILARITY_CITATION`).
    fn cosine(xs: &[f64], ys: &[f64]) -> f64 {
        let (dot, _) = Self::dot_sums(xs, ys);
        let na: f64 = xs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = ys.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Dot product in `f64` — a *measurement* helper (no `Meta` to tag), mirroring
    /// `VsaModel::similarity`. Typed errors for space mismatches. The value-returning twin
    /// with the honest `Proven` bound attached is [`Self::dot_value`] (M-891).
    pub fn dot(&self, a: &Value, b: &Value) -> Result<f64, DenseError> {
        let xs = self.scalars_of(a)?;
        let ys = self.scalars_of(b)?;
        Ok(Self::dot_sums(xs, ys).0)
    }

    /// Cosine similarity in `[-1, 1]` (`0` if either operand has zero norm) — a measurement
    /// helper in `f64`. The value-returning twin with the honest `Proven` bound attached is
    /// [`Self::similarity_value`] (M-891).
    pub fn similarity(&self, a: &Value, b: &Value) -> Result<f64, DenseError> {
        let xs = self.scalars_of(a)?;
        let ys = self.scalars_of(b)?;
        Ok(Self::cosine(xs, ys))
    }

    /// Wrap a scalar `f64` measurement as a **`Dense{1, F64}`** result `Value` carrying its
    /// honest `Proven` absolute (`Linf` over the single element) error bound (M-891). The
    /// payload is the `f64` **exactly as computed** — never re-rounded onto the operand dtype
    /// grid (that would add error and spurious overflow/subnormal refusals to a measurement).
    /// `F64` has no Dense op set (v1 scope), so a measurement cannot silently feed back into
    /// dense arithmetic — re-entry would be an explicit `UnsupportedDtype` refusal (G2).
    fn wrap_measurement(
        &self,
        y: f64,
        eps: f64,
        citation: &'static str,
        op: &str,
        inputs: Vec<ContentHash>,
    ) -> Result<Value, DenseError> {
        // Checked side-condition on the result (unreachable for dim < 2^32 — see the citation's
        // no-overflow argument — but checked, never assumed; VR-5).
        if !y.is_finite() || !eps.is_finite() {
            return Err(DenseError::Overflow { index: 0 });
        }
        let bound = Bound {
            kind: BoundKind::Error {
                eps,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::ProvenThm {
                citation: citation.to_owned(),
            },
        };
        let meta = Meta::new(
            Provenance::Derived {
                op: operation_hash(op),
                inputs,
            },
            GuaranteeStrength::Proven,
            Some(bound),
            None,
            None,
            None,
        )
        .map_err(DenseError::Wf)?;
        Value::new(
            Repr::Dense {
                dim: 1,
                dtype: ScalarKind::F64,
            },
            Payload::Scalars(vec![y]),
            meta,
        )
        .map_err(DenseError::Wf)
    }

    /// Dot product as a **`Dense{1, F64}` value** with its honest per-op tag (M-891):
    /// **`Proven`**, absolute (`Linf`) ε = [`Self::dot_abs_eps`] of the computed abs-product
    /// sum, `ProvenThm` basis (binary64 summation over exact products — the citation carries
    /// the full argument and the checked side-conditions). Same operand contract as
    /// [`Self::dot`] (equal dim + dtype, `Exact` on-grid sources — typed errors otherwise).
    pub fn dot_value(&self, a: &Value, b: &Value) -> Result<Value, DenseError> {
        let xs = self.scalars_of(a)?;
        let ys = self.scalars_of(b)?;
        let (sum, abs_sum) = Self::dot_sums(xs, ys);
        self.wrap_measurement(
            sum,
            self.dot_abs_eps(abs_sum),
            DOT_CITATION,
            "dense.dot",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Cosine similarity as a **`Dense{1, F64}` value** with its honest per-op tag (M-891):
    /// **`Proven`**, absolute (`Linf`) ε = [`Self::similarity_abs_eps`] (input-independent —
    /// normalization caps the absolute error), `ProvenThm` basis. Zero-norm operands yield the
    /// documented convention `0` exactly (disclosed in the citation). Same operand contract as
    /// [`Self::similarity`].
    pub fn similarity_value(&self, a: &Value, b: &Value) -> Result<Value, DenseError> {
        let xs = self.scalars_of(a)?;
        let ys = self.scalars_of(b)?;
        self.wrap_measurement(
            Self::cosine(xs, ys),
            self.similarity_abs_eps(),
            SIMILARITY_CITATION,
            "dense.similarity",
            vec![a.content_hash(), b.content_hash()],
        )
    }
}

#[cfg(test)]
mod tests;
