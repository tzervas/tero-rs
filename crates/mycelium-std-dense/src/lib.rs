//! `std.dense` — Ring 1 / Tier A capability surface (M-518; RFC-0016 §4.1).
//!
//! An ergonomic, value-semantic surface over [`mycelium_dense`] (M-230) typed, dimension-tracked
//! dense values (`Dense{dim, dtype}` tensors / embeddings). `dim` and `dtype` are carried in the
//! [`DenseSpace`] descriptor so a shape or precision mismatch is an explicit typed error, never a
//! silent broadcast, truncation, or re-round (C1/G2).
//!
//! ## Contract (RFC-0016 §4.1 C1–C6)
//!
//! - **C1 — never-silent:** dim/dtype mismatch → [`StdDenseError`]; zero-norm cosine → explicit
//!   `Err(ZeroNorm)`; off-grid / wrong-length input → explicit `Err`; no `NaN`/sentinel escapes.
//! - **C2 — honest per-op tag:** integer ops → `Exact`; float **elementwise** (`add`/`sub`/`scale`/
//!   `hadamard`) → `Proven` (Q1 — finalized: the ADR-010 per-element IEEE backward-error
//!   instantiation, M-512 delivered); float **accumulation** (`sum`/`dot`/`l1_norm`) → `Empirical`
//!   (the `nγ_n` bound is a distinct theorem, not yet checked); `l2_norm`/`cosine` → `Empirical`
//!   (FLAG Q2 — sqrt/division composition side-conditions not yet fully checked; honest downgrade
//!   per VR-5). Tags degrade by meet; never upgrade without a checked basis.
//! - **C3 — EXPLAIN:** every float result carries a reified [`OpBound`] (ε, norm, basis string),
//!   inspectable without further processing. A `from_slice` or `cosine` failure names which check
//!   failed.
//! - **C4 — value-semantic:** all ops are pure functions; `Dense` values are immutable.
//! - **C5 — above the kernel (KC-3):** consumes `mycelium-dense` + `mycelium-core`; no new trusted
//!   code, no `unsafe`, no FFI.
//! - **C6 — declared effects:** every op here is pure (`effects: none`) except `map`, whose effects
//!   are its function argument's declared effects only.
//!
//! ## Guarantee matrix (RFC-0016 §4.5 — encoded as data, asserted in tests)
//!
//! See [`GUARANTEE_MATRIX`] and the `guarantee_matrix_*` tests.
//!
//! ## Scope boundary
//!
//! - Representation changes (`Dense{F32}→Dense{BF16}`, `Dense↔VSA`) → **`std.swap`** (M-516).
//! - The ε/δ bound algebra → **`std.numerics`** (M-512; ADR-010); this module *consumes*, never
//!   defines.
//! - Non-representation scalar conversion → **`std.cmp`/`convert`** (M-532).
//! - VSA hypervector algebra → **`std.vsa`** (M-513).
//!
//! ## Q1 (RESOLVED for elementwise; accumulation stays `Empirical`) — float `Proven` disposition
//!
//! The `Proven` tags on `add`/`sub`/`scale`/`hadamard` (float DT) are **finalized** (DN-16,
//! 2026-06-19; maintainer-ratified): they rest on the ADR-010 `ErrorBound` per-element IEEE
//! backward-error instantiation (`fl(a∘b) = (a∘b)(1+δ)`, `|δ| ≤ u` — Higham 2002, Thm 2.2), whose
//! single side-condition (operand finiteness / no overflow) is **guarded at runtime** (non-finite
//! inputs → kernel [`DenseError::NonFinite`], surfaced as [`StdDenseError::Kernel`]) and re-validated
//! by the numerics checker (M-512, delivered).
//! The accumulation ops (`sum`, `dot`, `l1_norm`) stay **`Empirical`** conservatively: the `nγ_n`
//! accumulation bound is a *distinct* theorem not yet discharged by a checked instantiation;
//! upgrade accumulation to `Proven` only when that checked accumulation theorem is delivered (VR-5).
//!
//! ## FLAG Q2 — `Empirical` for l2_norm / cosine
//!
//! `l2_norm` and `cosine` compose a bounded dot/sum-of-squares with a sqrt (and cosine adds a
//! division). The composed bound's side-conditions are guarded at runtime (`ZeroNorm` for cosine)
//! but **not yet fully checkable** through the ADR-010 affine/error kernel. Tagged `Empirical`
//! pending a checked composition from M-512. Upgrade requires a checked theorem (VR-5).
//!
//! ## FLAG Q3 — integer dtype ops (design claims; kernel pending)
//!
//! The v1 `mycelium-dense` supports only `F32`/`BF16`; integer dtype ops are specified in the
//! guarantee matrix as `Exact` design claims but are not implemented in the underlying kernel
//! (M-230 v1 scope, FLAG Q3). They will be implemented when the kernel adds integer dtype support.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A):** Dense operations always operate in the `Dense`
//! representation; no implicit cross-representation conversion occurs. Shape and dtype are carried
//! in the [`DenseSpace`] descriptor (never inferred), and a shape or dtype mismatch is an explicit
//! `Err`, never a silent broadcast or re-round (C1). Representation changes (`Dense{F32}→Dense{BF16}`,
//! `Dense↔VSA`) are explicit swaps in `std.swap` — never automatic.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/dense.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub use mycelium_core::{Bound, BoundBasis, BoundKind, GuaranteeStrength, NormKind, ScalarKind};
pub use mycelium_dense::{
    DenseError, DenseOp, DenseSpace, BF16_OP_REL_EPS, DENSE_MIN_NORMAL, F32_OP_REL_EPS,
};

// ────────────────────────────────────────────────────────────────────────────
// § 1 — Public error type (C1: explicit, named, never sentinel)
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the `std.dense` capability surface (C1/G2: explicit typed errors, never sentinels).
///
/// This is a superset of [`DenseError`] (from the `mycelium-dense` kernel): it adds
/// `ZeroNorm` (for `cosine` on a zero-norm vector) and wraps kernel errors transparently.
#[derive(Debug, Clone, PartialEq)]
pub enum StdDenseError {
    /// Dimension mismatch: the input slice length does not match the space's `dim`.
    ///
    /// EXPLAIN: names the expected and actual length so the caller knows which check failed (C3).
    LenMismatch {
        /// The dimension the space requires.
        expected: usize,
        /// The actual slice length.
        got: usize,
    },
    /// An element is not exactly representable on the declared dtype grid.
    ///
    /// EXPLAIN: names the dtype and element index so the caller knows which grid was violated (C3).
    OffGrid {
        /// The dtype whose grid was violated.
        dtype: ScalarKind,
        /// The index of the offending element.
        index: usize,
    },
    /// `cosine` called on a zero-norm vector — the result is mathematically undefined.
    ///
    /// EXPLAIN: returned explicitly; never `NaN`, never `0.0` silently (C1/G2).
    ZeroNorm,
    /// A kernel-level error from `mycelium-dense` (propagated transparently).
    Kernel(DenseError),
}

impl core::fmt::Display for StdDenseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StdDenseError::LenMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            StdDenseError::OffGrid { dtype, index } => {
                write!(f, "element {index} is not on the {dtype:?} grid")
            }
            StdDenseError::ZeroNorm => write!(f, "zero-norm vector: cosine is undefined"),
            StdDenseError::Kernel(e) => write!(f, "kernel error: {e}"),
        }
    }
}

impl std::error::Error for StdDenseError {}

impl From<DenseError> for StdDenseError {
    fn from(e: DenseError) -> Self {
        // Map kernel dim-mismatch → our LenMismatch for the from_slice boundary;
        // other variants propagate as Kernel.
        match e {
            DenseError::DimMismatch { expected, got } => StdDenseError::LenMismatch {
                expected: expected as usize,
                got: got as usize,
            },
            other => StdDenseError::Kernel(other),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// § 2 — OpBound: reified, inspectable ε-bound artifact (C3/EXPLAIN)
// ────────────────────────────────────────────────────────────────────────────

/// A reified ε-bound artifact: the inspectable record of a float op's accuracy claim (C3/EXPLAIN;
/// RFC-0001 §4.3 `Bound`; SC-3/G11).
///
/// Every float op returns an `OpBound` carrying:
/// - `eps`: the ε-bound magnitude in the norm `norm` (a *true upper bound*, A2-01),
/// - `norm`: the norm kind (`Rel` for relative, `L2` for L2, etc.),
/// - `strength`: the guarantee lattice tag (`Proven`/`Empirical`),
/// - `basis`: the human-readable citation or fit description — the EXPLAIN artifact.
///
/// The bound is always a *true upper bound* (not a round-to-nearest approximation) as required
/// by the dev-workflow banked guard (A2-01): the stored `eps` is ≥ the mathematical value.
#[derive(Debug, Clone, PartialEq)]
pub struct OpBound {
    /// Error magnitude (ε ≥ 0, finite).
    pub eps: f64,
    /// The norm in which `eps` is expressed.
    pub norm: NormKind,
    /// The lattice tag (`Proven`/`Empirical`; never `Exact` — `OpBound` is only issued for
    /// approximate ops).
    pub strength: GuaranteeStrength,
    /// The EXPLAIN artifact: citation or empirical-fit description.
    pub basis: String,
}

impl OpBound {
    /// Convert to a [`Bound`] suitable for attaching to a [`mycelium_core::Meta`].
    ///
    /// `Proven` → `ProvenThm { citation: basis }`; `Empirical` → `EmpiricalFit { trials: 1,
    /// method: basis }` (the minimum evidence-present requirement from A6-02/B2-03).
    /// `Declared`/`Exact` → `UserDeclared` (conservative).
    #[must_use]
    pub fn to_core_bound(&self) -> Bound {
        let basis = match self.strength {
            GuaranteeStrength::Proven => BoundBasis::ProvenThm {
                citation: self.basis.clone(),
            },
            GuaranteeStrength::Empirical => BoundBasis::EmpiricalFit {
                // trials: 1 is the minimum for a non-evidence-free EmpiricalFit (A6-02/B2-03).
                // The "1" marks "at least one confirmed sample", not a purely statistical basis.
                trials: 1,
                method: self.basis.clone(),
            },
            _ => BoundBasis::UserDeclared,
        };
        Bound {
            kind: BoundKind::Error {
                eps: self.eps,
                norm: self.norm,
            },
            basis,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// § 3 — Guarantee matrix (RFC-0016 §4.5 — encoded as data, asserted in tests)
// ────────────────────────────────────────────────────────────────────────────

/// One row of the guarantee matrix (§4 of `docs/spec/stdlib/dense.md`).
///
/// Encoded as a struct so the matrix is a Rust constant — asserted in tests, never prose only
/// (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// Operation name (matches the spec matrix op column).
    pub op: &'static str,
    /// The lattice tag (the weakest of intrinsic + input tags).
    pub tag: GuaranteeStrength,
    /// Whether the op returns `Result` (fallible).
    pub fallible: bool,
    /// The explicit error variants when `fallible == true` (empty for total ops).
    pub error_variants: &'static [&'static str],
    /// Whether the op exposes an inspectable bound artifact (C3/EXPLAIN).
    pub explainable: bool,
    /// Declared effects beyond pure computation ("none" = pure).
    pub effects: &'static str,
}

/// The guarantee matrix for `std.dense` (RFC-0016 §4.5).
///
/// Rows = exported ops; columns = tag · fallibility · error variants · EXPLAIN-able · effects.
/// Every row is asserted in the `guarantee_matrix_*` tests.
///
/// **Design claims (FLAG Q3):** Integer dtype rows (`*_int`) are design claims for when the
/// kernel adds integer dtype support. Float rows correspond to the live M-230 v1 surface (F32/BF16).
pub const GUARANTEE_MATRIX: &[GuaranteeRow] = &[
    // --- constructors ---
    GuaranteeRow {
        op: "zeros",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    GuaranteeRow {
        op: "full",
        tag: GuaranteeStrength::Exact,
        fallible: true, // Err if x is off-grid (C1)
        error_variants: &["OffGrid"],
        explainable: true, // names which grid check failed (C3)
        effects: "none",
    },
    GuaranteeRow {
        op: "from_slice",
        tag: GuaranteeStrength::Exact,
        fallible: true,
        error_variants: &["LenMismatch", "OffGrid"],
        explainable: true, // names which check failed (C3)
        effects: "none",
    },
    // --- elementwise — int DT (design claims; FLAG Q3: kernel integer support pending) ---
    GuaranteeRow {
        op: "add_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    GuaranteeRow {
        op: "sub_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    GuaranteeRow {
        op: "hadamard_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    GuaranteeRow {
        op: "scale_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    // --- elementwise — float DT (live M-230 v1; Proven finalized, Q1; M-512-checked) ---
    GuaranteeRow {
        op: "neg_float",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    GuaranteeRow {
        op: "add_float",
        tag: GuaranteeStrength::Proven,
        fallible: false, // shape/dtype = static type contract of the DenseSpace
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    GuaranteeRow {
        op: "sub_float",
        tag: GuaranteeStrength::Proven,
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    GuaranteeRow {
        op: "hadamard_float",
        tag: GuaranteeStrength::Proven,
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    GuaranteeRow {
        op: "scale_float",
        tag: GuaranteeStrength::Proven,
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    // --- map (tag = meet of input tag and f's tag; never upgraded; VR-5) ---
    GuaranteeRow {
        op: "map",
        tag: GuaranteeStrength::Exact, // intrinsic = Exact; actual tag = meet with f_tag
        fallible: false,               // fallibility is f's
        error_variants: &[],           // f's error set
        explainable: false,            // iff f is
        effects: "f_effects",
    },
    // --- reductions — int DT (design claims; FLAG Q3) ---
    GuaranteeRow {
        op: "sum_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    GuaranteeRow {
        op: "l1_norm_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    // --- reductions — float DT ---
    // FLAG Q1: accumulation bound conservatively Empirical (pending M-512 Higham instantiation).
    GuaranteeRow {
        op: "sum_float",
        tag: GuaranteeStrength::Empirical,
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    GuaranteeRow {
        op: "l1_norm_float",
        tag: GuaranteeStrength::Empirical,
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    // FLAG Q2: l2_norm Empirical (sqrt composition side-conditions not fully checked).
    GuaranteeRow {
        op: "l2_norm_float",
        tag: GuaranteeStrength::Empirical,
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    // --- similarity — int DT (design claim; FLAG Q3) ---
    GuaranteeRow {
        op: "dot_int",
        tag: GuaranteeStrength::Exact,
        fallible: false,
        error_variants: &[],
        explainable: false,
        effects: "none",
    },
    // --- similarity — float DT ---
    GuaranteeRow {
        op: "dot_float",
        tag: GuaranteeStrength::Empirical, // accumulation bound (FLAG Q1)
        fallible: false,
        error_variants: &[],
        explainable: true,
        effects: "none",
    },
    // FLAG Q2: cosine Empirical (sqrt + division; ZeroNorm guarded explicitly — C1).
    GuaranteeRow {
        op: "cosine_float",
        tag: GuaranteeStrength::Empirical,
        fallible: true,
        error_variants: &["ZeroNorm"],
        explainable: true,
        effects: "none",
    },
];

// ────────────────────────────────────────────────────────────────────────────
// § 4 — Accumulation bound constants and helpers (FLAG Q1 conservative)
// ────────────────────────────────────────────────────────────────────────────

/// Conservative empirical ε for floating-point accumulation ops (`sum`, `l1_norm`, `dot`).
///
/// The Higham backward-error bound for a sum of `n` IEEE binary32 terms is approximately
/// `n · u` where `u = F32_OP_REL_EPS = 2^-24`. Because M-512 has not yet delivered a checked
/// instantiation of this bound, we tag these ops **`Empirical`** and use `2 · n · u` as a
/// *conservative outward-rounded upper bound* (the factor of 2 provides slack ensuring the stored
/// ε is a true upper bound — dev-workflow banked guard A2-01).
///
/// Disposition: once M-512 delivers the checked Higham instantiation, upgrade to `Proven`
/// and replace this with the exact `nγ_n` constant.
#[must_use]
pub fn accumulation_eps_f32(n: usize) -> f64 {
    // Outward-rounded: 2 · n · u. The factor of 2 is a conservative slack (A2-01).
    2.0 * (n as f64) * F32_OP_REL_EPS
}

/// BF16 analogue of [`accumulation_eps_f32`].
#[must_use]
pub fn accumulation_eps_bf16(n: usize) -> f64 {
    2.0 * (n as f64) * BF16_OP_REL_EPS
}

/// Empirical basis string for accumulation ops (FLAG Q1).
pub const ACCUMULATION_EMPIRICAL_BASIS: &str =
    "empirical upper bound: 2·n·u (u = IEEE binary32 unit roundoff 2^−24; factor-of-2 slack is \
     a conservative outward-rounded upper bound pending a checked nγ_n accumulation theorem for a \
     Proven upgrade — Q1, std.dense §7)";

/// Empirical basis string for BF16 accumulation ops (FLAG Q1).
pub const ACCUMULATION_BF16_EMPIRICAL_BASIS: &str =
    "empirical upper bound: 2·n·u (u = BF16 two-rounding ε = 2^−8 + 2^−23; conservative \
     outward-rounded upper bound pending a checked nγ_n accumulation theorem — Q1, std.dense §7)";

/// Empirical basis string for L2-norm / cosine ops (FLAG Q2).
pub const SQRT_COMPOSITION_EMPIRICAL_BASIS: &str =
    "empirical upper bound: sqrt/division composition; side-conditions (non-negativity for sqrt, \
     non-zero denominator for division — guarded by ZeroNorm) not yet fully checkable via \
     ADR-010 affine/error kernel — FLAG Q2, std.dense §7; pending M-512 checked composition";

// ────────────────────────────────────────────────────────────────────────────
// § 5 — StdDense: the ergonomic Ring-1 capability surface
// ────────────────────────────────────────────────────────────────────────────

/// The ergonomic Ring-1 capability surface over a typed `Dense{dim, dtype}` space (M-518).
///
/// `StdDense` wraps a [`DenseSpace`] and provides:
/// - Typed constructors: [`zeros`](Self::zeros), [`full`](Self::full),
///   [`from_slice`](Self::from_slice)
/// - Elementwise ops: [`add`](Self::add), [`sub`](Self::sub), [`neg`](Self::neg),
///   [`hadamard`](Self::hadamard), [`scale`](Self::scale), [`map`](Self::map)
/// - Reductions: [`sum`](Self::sum), [`l1_norm`](Self::l1_norm), [`l2_norm`](Self::l2_norm)
/// - Similarity: [`dot`](Self::dot), [`cosine`](Self::cosine)
///
/// Float ops return their result alongside an [`OpBound`] artifact (C3/EXPLAIN).
///
/// ## Usage
///
/// ```rust
/// use mycelium_std_dense::{StdDense, ScalarKind, GuaranteeStrength};
///
/// let space = StdDense::new(4, ScalarKind::F32).unwrap();
/// let a = space.from_slice(&[1.0_f64, 0.0, 0.0, 0.0]).unwrap();
/// let b = space.from_slice(&[0.0_f64, 1.0, 0.0, 0.0]).unwrap();
/// let (result, bound) = space.add(&a, &b).unwrap();
/// assert_eq!(bound.strength, GuaranteeStrength::Proven);
/// ```
#[derive(Debug)]
pub struct StdDense {
    space: DenseSpace,
}

impl StdDense {
    /// Construct a `StdDense` surface for a `dim`-dimensional space over `dtype`.
    ///
    /// Returns `Err` if `dtype` is unsupported in v1 (F32/BF16 only; F16/F64 are
    /// `Kernel(UnsupportedDtype)` — honest scope, FLAG Q3).
    pub fn new(dim: u32, dtype: ScalarKind) -> Result<Self, StdDenseError> {
        DenseSpace::new(dim, dtype)
            .map(|space| StdDense { space })
            .map_err(StdDenseError::Kernel)
    }

    /// The underlying [`DenseSpace`] descriptor.
    #[must_use]
    pub fn space(&self) -> &DenseSpace {
        &self.space
    }

    /// The dimensionality.
    #[must_use]
    pub fn dim(&self) -> u32 {
        self.space.dim
    }

    /// The element dtype.
    #[must_use]
    pub fn dtype(&self) -> ScalarKind {
        self.space.dtype
    }

    // ── constructors ────────────────────────────────────────────────────────

    /// Construct an **Exact** zero vector (guarantee matrix: `zeros` — `Exact`, total).
    #[must_use]
    pub fn zeros(&self) -> mycelium_core::Value {
        // zeros are always on-grid and finite for any supported dtype.
        self.space
            .value(vec![0.0_f64; self.space.dim as usize])
            .expect("zeros: zero vector always passes DenseSpace::value validation")
    }

    /// Construct an **Exact** constant vector with every element equal to `x`.
    ///
    /// Guarantee matrix: `full` — `Exact`, fallible (`OffGrid` if `x` is not on the dtype grid).
    ///
    /// # Errors
    ///
    /// [`StdDenseError::Kernel`] wrapping [`DenseError::NotOnGrid`] or [`DenseError::NonFinite`]
    /// if `x` is not finite or not exactly representable on the `dtype` grid.
    pub fn full(&self, x: f64) -> Result<mycelium_core::Value, StdDenseError> {
        self.space
            .value(vec![x; self.space.dim as usize])
            .map_err(StdDenseError::from)
    }

    /// Construct a value from a slice, checking length and grid alignment.
    ///
    /// Guarantee matrix: `from_slice` — `Exact`, fallible (`LenMismatch` / `OffGrid`),
    /// EXPLAIN-able (names which check failed — C3).
    ///
    /// # Errors
    ///
    /// - [`StdDenseError::LenMismatch`]: `xs.len() != dim`.
    /// - [`StdDenseError::OffGrid`]: an element is not exactly representable on the `dtype` grid.
    /// - [`StdDenseError::Kernel`]: non-finite element or other kernel-level issue.
    pub fn from_slice(&self, xs: &[f64]) -> Result<mycelium_core::Value, StdDenseError> {
        // Check length first (C3: EXPLAIN which check failed).
        if xs.len() != self.space.dim as usize {
            return Err(StdDenseError::LenMismatch {
                expected: self.space.dim as usize,
                got: xs.len(),
            });
        }
        // Check grid alignment before delegating (C3: OffGrid names dtype + index).
        for (index, &x) in xs.iter().enumerate() {
            if !is_on_grid(self.space.dtype, x) {
                return Err(StdDenseError::OffGrid {
                    dtype: self.space.dtype,
                    index,
                });
            }
        }
        self.space.value(xs.to_vec()).map_err(StdDenseError::from)
    }

    // ── elementwise ops ─────────────────────────────────────────────────────

    /// Elementwise `a + b` — float DT: `Proven` (FLAG Q1 — uses kernel bound).
    ///
    /// Returns `(result, bound)` — `bound` is the EXPLAIN artifact (C3).
    ///
    /// # Errors
    ///
    /// Kernel errors propagated as [`StdDenseError::Kernel`].
    pub fn add(
        &self,
        a: &mycelium_core::Value,
        b: &mycelium_core::Value,
    ) -> Result<(mycelium_core::Value, OpBound), StdDenseError> {
        let v = self.space.add_values(a, b).map_err(StdDenseError::Kernel)?;
        Ok((v, self.elementwise_bound()))
    }

    /// Elementwise `a − b` — same contract as [`add`](Self::add).
    pub fn sub(
        &self,
        a: &mycelium_core::Value,
        b: &mycelium_core::Value,
    ) -> Result<(mycelium_core::Value, OpBound), StdDenseError> {
        let v = self.space.sub_values(a, b).map_err(StdDenseError::Kernel)?;
        Ok((v, self.elementwise_bound()))
    }

    /// Elementwise negation — **Exact** (the dtype grid is symmetric; no rounding).
    ///
    /// Returns `result` only — Exact ops carry no bound (M-I1).
    ///
    /// # Errors
    ///
    /// Kernel errors propagated as [`StdDenseError::Kernel`].
    pub fn neg(&self, a: &mycelium_core::Value) -> Result<mycelium_core::Value, StdDenseError> {
        self.space.neg_value(a).map_err(StdDenseError::Kernel)
    }

    /// Elementwise (Hadamard) product `a ⊙ b` — float DT: `Proven` (FLAG Q1).
    ///
    /// Implemented element-by-element via the kernel's `scale_value` path (each element of `a`
    /// is scaled by the corresponding element of `b`), so the Proven bound is inherited from
    /// the single-multiply Higham Thm 2.2 instantiation (same as `scale`).
    ///
    /// Returns `(result, bound)`.
    ///
    /// # Errors
    ///
    /// Kernel errors propagated as [`StdDenseError::Kernel`].
    pub fn hadamard(
        &self,
        a: &mycelium_core::Value,
        b: &mycelium_core::Value,
    ) -> Result<(mycelium_core::Value, OpBound), StdDenseError> {
        let scalars_a = extract_scalars(a)?;
        let scalars_b = extract_scalars(b)?;
        let dim = self.space.dim as usize;
        if scalars_a.len() != dim {
            return Err(StdDenseError::LenMismatch {
                expected: dim,
                got: scalars_a.len(),
            });
        }
        if scalars_b.len() != dim {
            return Err(StdDenseError::LenMismatch {
                expected: dim,
                got: scalars_b.len(),
            });
        }
        // Delegate per-element to the kernel's scale_value, which enforces the proven bound.
        let space1 = DenseSpace::new(1, self.space.dtype).map_err(StdDenseError::Kernel)?;
        let mut result_scalars = Vec::with_capacity(dim);
        for i in 0..dim {
            let ai = scalars_a[i];
            let bi = scalars_b[i];
            let va = space1.value(vec![ai]).map_err(StdDenseError::Kernel)?;
            let vr = space1.scale_value(&va, bi).map_err(StdDenseError::Kernel)?;
            let r = extract_scalars(&vr)?;
            result_scalars.push(r[0]);
        }
        let result = self
            .space
            .value(result_scalars)
            .map_err(StdDenseError::Kernel)?;
        Ok((result, self.elementwise_bound()))
    }

    /// Scalar multiplication `s · a` — float DT: `Proven` (FLAG Q1).
    ///
    /// `s` must be finite and on the `dtype` grid; [`DenseError::ScalarOffGrid`] otherwise.
    ///
    /// Returns `(result, bound)`.
    ///
    /// # Errors
    ///
    /// - [`StdDenseError::Kernel`] wrapping [`DenseError::ScalarOffGrid`] if `s` is off-grid.
    /// - Other kernel errors propagated.
    pub fn scale(
        &self,
        a: &mycelium_core::Value,
        s: f64,
    ) -> Result<(mycelium_core::Value, OpBound), StdDenseError> {
        let v = self
            .space
            .scale_value(a, s)
            .map_err(StdDenseError::Kernel)?;
        Ok((v, self.elementwise_bound()))
    }

    /// Map a function `f` over every element (tag = meet of input tag and `f_tag` — VR-5).
    ///
    /// The output's guarantee tag is `meet(input.guarantee, f_tag)` — `map` never *upgrades*
    /// a value's strength. Declare `f_tag` honestly: if `f` is exact, use `Exact`; if `f` is
    /// an approximation, use its actual tag.
    ///
    /// **Guarantee matrix: `map`** — intrinsic `Exact`, actual tag = meet with `f_tag`; effects =
    /// `f`'s; fallibility = `f`'s. No `OpBound` is returned — the tag is the EXPLAIN artifact.
    ///
    /// **Note on non-Exact results:** when `f_tag` is not `Exact` (or the input is not `Exact`),
    /// the result carries a `UserDeclared` bound with `f64::MAX` ε as a finite-but-vacuous
    /// conservative placeholder (infinite ε is not a valid bound per A2-03). This is honest:
    /// `map` cannot know `f`'s ε without `f` providing it. Callers who need a tighter bound
    /// should use [`add`](Self::add), [`sub`](Self::sub), [`scale`](Self::scale), or
    /// [`hadamard`](Self::hadamard), which have known and inspectable bounds.
    ///
    /// # Errors
    ///
    /// - [`StdDenseError::LenMismatch`] if the input's payload does not match `dim`.
    /// - Any error returned by `f`, via `E: Into<StdDenseError>`.
    pub fn map<E>(
        &self,
        a: &mycelium_core::Value,
        f_tag: GuaranteeStrength,
        f: impl Fn(f64) -> Result<f64, E>,
    ) -> Result<(mycelium_core::Value, GuaranteeStrength), StdDenseError>
    where
        E: Into<StdDenseError>,
    {
        let scalars = extract_scalars(a)?;
        if scalars.len() != self.space.dim as usize {
            return Err(StdDenseError::LenMismatch {
                expected: self.space.dim as usize,
                got: scalars.len(),
            });
        }
        let mut out = Vec::with_capacity(scalars.len());
        for &x in scalars {
            out.push(f(x).map_err(|e| e.into())?);
        }
        // The result tag is the meet of the input's guarantee and f's declared tag (VR-5).
        let input_tag = a.meta().guarantee();
        let result_tag = input_tag.meet(f_tag);

        let result = if result_tag == GuaranteeStrength::Exact {
            // Exact: build via the kernel (no bound).
            self.space.value(out).map_err(StdDenseError::Kernel)?
        } else {
            // Non-Exact: carry a UserDeclared bound with f64::MAX ε as a finite-but-vacuous
            // conservative placeholder. This is honest: `map` cannot derive f's bound — the
            // caller must use add/sub/scale/hadamard for ops with known bounds. We use f64::MAX
            // rather than f64::INFINITY because Bound::well_formed() requires eps.is_finite()
            // (A2-03: bounds must be finite — infinite uncertainty is not a bound).
            use mycelium_core::{Meta, Payload, Provenance, Value};
            let bound = Bound {
                kind: BoundKind::Error {
                    eps: f64::MAX, // finite but vacuous — honest: unknown f bound (A2-03)
                    norm: NormKind::Rel,
                },
                basis: BoundBasis::UserDeclared,
            };
            let meta = Meta::new(
                Provenance::Root,
                GuaranteeStrength::Declared,
                Some(bound),
                None,
                None,
                None,
            )
            .map_err(|e| StdDenseError::Kernel(DenseError::Wf(e)))?;
            Value::new(self.space.repr(), Payload::Scalars(out), meta)
                .map_err(|e| StdDenseError::Kernel(DenseError::Wf(e)))?
        };
        Ok((result, result_tag))
    }

    // ── reductions ──────────────────────────────────────────────────────────

    /// Sum all elements — float DT: `Empirical` (FLAG Q1: accumulation bound, conservative
    /// downgrade from `Proven` pending M-512 Higham instantiation).
    ///
    /// Returns `(scalar_sum, bound)` — `bound` is the EXPLAIN artifact (C3).
    ///
    /// # Errors
    ///
    /// - [`StdDenseError::LenMismatch`] if payload length mismatches `dim`.
    /// - [`StdDenseError::Kernel`] for malformed input (not a `Dense` scalars value).
    pub fn sum(&self, a: &mycelium_core::Value) -> Result<(f64, OpBound), StdDenseError> {
        let scalars = self.scalars_checked(a)?;
        let s: f64 = scalars.iter().copied().sum();
        let n = scalars.len();
        let bound = OpBound {
            eps: self.accumulation_eps(n),
            norm: NormKind::Rel,
            strength: GuaranteeStrength::Empirical,
            basis: self.accumulation_basis().to_owned(),
        };
        Ok((s, bound))
    }

    /// L1 norm (sum of |xᵢ|) — float DT: `Empirical` (same accumulation argument as `sum`,
    /// FLAG Q1).
    ///
    /// Returns `(l1_norm, bound)`.
    ///
    /// # Errors
    ///
    /// Same as [`sum`](Self::sum).
    pub fn l1_norm(&self, a: &mycelium_core::Value) -> Result<(f64, OpBound), StdDenseError> {
        let scalars = self.scalars_checked(a)?;
        let l1: f64 = scalars.iter().map(|x| x.abs()).sum();
        let n = scalars.len();
        let bound = OpBound {
            eps: self.accumulation_eps(n),
            norm: NormKind::Rel,
            strength: GuaranteeStrength::Empirical,
            basis: self.accumulation_basis().to_owned(),
        };
        Ok((l1, bound))
    }

    /// L2 (Euclidean) norm — float DT: `Empirical` (FLAG Q2: sqrt composition not fully checked).
    ///
    /// Returns `(l2_norm, bound)`. Computed as `sqrt(Σ xᵢ²)` in `f64`.
    ///
    /// # Errors
    ///
    /// Same as [`sum`](Self::sum).
    pub fn l2_norm(&self, a: &mycelium_core::Value) -> Result<(f64, OpBound), StdDenseError> {
        let scalars = self.scalars_checked(a)?;
        let sum_sq: f64 = scalars.iter().map(|x| x * x).sum();
        let l2 = sum_sq.sqrt();
        let n = scalars.len();
        // ε: conservative — accumulation eps for the sum-of-squares step + 0.5·u for the sqrt.
        // Outward-rounded sum (A2-01).
        let eps = self.accumulation_eps(n) + 0.5 * self.unit_roundoff();
        let bound = OpBound {
            eps,
            norm: NormKind::Rel,
            strength: GuaranteeStrength::Empirical,
            basis: SQRT_COMPOSITION_EMPIRICAL_BASIS.to_owned(),
        };
        Ok((l2, bound))
    }

    // ── similarity ops ───────────────────────────────────────────────────────

    /// Dot product `⟨a, b⟩` — float DT: `Empirical` (FLAG Q1: accumulation bound).
    ///
    /// Returns `(dot, bound)`. Computed in `f64`.
    ///
    /// # Errors
    ///
    /// - [`StdDenseError::LenMismatch`] if either payload length mismatches `dim`.
    /// - [`StdDenseError::Kernel`] for malformed input.
    pub fn dot(
        &self,
        a: &mycelium_core::Value,
        b: &mycelium_core::Value,
    ) -> Result<(f64, OpBound), StdDenseError> {
        let sa = self.scalars_checked(a)?;
        let sb = self.scalars_checked(b)?;
        let d: f64 = sa.iter().zip(sb.iter()).map(|(x, y)| x * y).sum();
        let n = sa.len();
        let bound = OpBound {
            eps: self.accumulation_eps(n),
            norm: NormKind::Rel,
            strength: GuaranteeStrength::Empirical,
            basis: self.accumulation_basis().to_owned(),
        };
        Ok((d, bound))
    }

    /// Cosine similarity — float DT: `Empirical` (FLAG Q2: sqrt + division composition).
    ///
    /// Returns `(cosine, bound)` in `(−1, 1]`. The zero-norm check is explicit (C1/G2):
    /// a zero vector is `Err(ZeroNorm)`, never `NaN` or a silent `0.0`.
    ///
    /// # Errors
    ///
    /// - [`StdDenseError::ZeroNorm`]: either operand is the zero vector.
    /// - [`StdDenseError::LenMismatch`]: payload length mismatch.
    /// - [`StdDenseError::Kernel`]: malformed input.
    pub fn cosine(
        &self,
        a: &mycelium_core::Value,
        b: &mycelium_core::Value,
    ) -> Result<(f64, OpBound), StdDenseError> {
        let sa = self.scalars_checked(a)?;
        let sb = self.scalars_checked(b)?;
        let dot: f64 = sa.iter().zip(sb.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = sa.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = sb.iter().map(|x| x * x).sum::<f64>().sqrt();
        // C1/G2: zero norm is explicit ZeroNorm — never NaN, never silent 0.0.
        if na == 0.0 || nb == 0.0 {
            return Err(StdDenseError::ZeroNorm);
        }
        let cosine = dot / (na * nb);
        let n = sa.len();
        // Conservative composed ε: accumulation + two sqrt steps + one divide.
        // All additions are outward-rounded (A2-01).
        let u = self.unit_roundoff();
        let eps = self.accumulation_eps(n) + u + u + u; // 3u slack for sqrt, sqrt, divide
        let bound = OpBound {
            eps,
            norm: NormKind::Rel,
            strength: GuaranteeStrength::Empirical,
            basis: SQRT_COMPOSITION_EMPIRICAL_BASIS.to_owned(),
        };
        Ok((cosine, bound))
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// The per-element relative ε (unit roundoff) for this space's dtype.
    fn unit_roundoff(&self) -> f64 {
        self.space.op_rel_eps()
    }

    /// The accumulation ε for `n` terms (outward-rounded; FLAG Q1).
    fn accumulation_eps(&self, n: usize) -> f64 {
        match self.space.dtype {
            ScalarKind::Bf16 => accumulation_eps_bf16(n),
            _ => accumulation_eps_f32(n),
        }
    }

    /// The accumulation empirical basis string for this space's dtype.
    fn accumulation_basis(&self) -> &'static str {
        match self.space.dtype {
            ScalarKind::Bf16 => ACCUMULATION_BF16_EMPIRICAL_BASIS,
            _ => ACCUMULATION_EMPIRICAL_BASIS,
        }
    }

    /// The per-element `OpBound` for elementwise ops (add/sub/scale/hadamard) — Proven.
    fn elementwise_bound(&self) -> OpBound {
        OpBound {
            eps: self.space.op_rel_eps(),
            norm: NormKind::Rel,
            strength: GuaranteeStrength::Proven,
            basis: elementwise_citation(self.space.dtype),
        }
    }

    /// Extract and length-check the scalar payload of `v` for this space's dim.
    fn scalars_checked<'v>(&self, v: &'v mycelium_core::Value) -> Result<&'v [f64], StdDenseError> {
        let scalars = extract_scalars(v)?;
        if scalars.len() != self.space.dim as usize {
            return Err(StdDenseError::LenMismatch {
                expected: self.space.dim as usize,
                got: scalars.len(),
            });
        }
        Ok(scalars)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// § 6 — Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Extract the scalar payload from a `Dense` value.
fn extract_scalars(v: &mycelium_core::Value) -> Result<&[f64], StdDenseError> {
    use mycelium_core::Payload;
    match v.payload() {
        Payload::Scalars(xs) => Ok(xs.as_slice()),
        _ => Err(StdDenseError::Kernel(DenseError::NotDense)),
    }
}

/// Whether `x` is exactly representable on the given `dtype` grid (finite values only).
///
/// Mirrors the `on_grid` check in `mycelium-dense` (not re-exported) — used here to surface
/// `OffGrid` with the specific `dtype` and `index` for C3/EXPLAIN in `from_slice`.
fn is_on_grid(dtype: ScalarKind, x: f64) -> bool {
    if !x.is_finite() {
        return false;
    }
    #[allow(clippy::cast_possible_truncation)] // representability is exactly what we check
    let xf = x as f32;
    if f64::from(xf) != x {
        return false;
    }
    match dtype {
        ScalarKind::F32 => true,
        ScalarKind::Bf16 => {
            // Round to BF16 (ties to even) and check the round-trip.
            let bits = xf.to_bits();
            let lsb = (bits >> 16) & 1;
            let rounded = f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16);
            rounded == xf
        }
        ScalarKind::F16 | ScalarKind::F64 => false,
    }
}

/// The citation string for element-wise `Proven` bounds (add/sub/scale/hadamard).
fn elementwise_citation(dtype: ScalarKind) -> String {
    match dtype {
        ScalarKind::Bf16 => {
            "two-rounding composition (1+δ₁)(1+δ₂)−1 ≤ 2^−8 + 2^−23: native f32 op then \
             bfloat16 round-to-nearest — Higham (2002), Thm 2.2; side-conditions checked per \
             element (Q1 finalized; M-512-checked)"
                .to_owned()
        }
        _ => "round-to-nearest relative error ≤ u = 2^−24 for IEEE binary32 — Higham (2002), \
             Thm 2.2; side-conditions checked per element (Q1 finalized; M-512-checked)"
            .to_owned(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// § 7 — Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::GuaranteeStrength::{Empirical, Exact, Proven};

    // ── guarantee matrix tests (RFC-0016 §4.5 — encoded as data, asserted here) ──

    /// Assert that the guarantee matrix has no duplicate op names.
    #[test]
    fn guarantee_matrix_no_duplicate_ops() {
        // Mutant-witness: if two rows share an op name, this catches the duplicate.
        let mut seen = std::collections::HashSet::new();
        for row in GUARANTEE_MATRIX {
            assert!(
                seen.insert(row.op),
                "duplicate op name in GUARANTEE_MATRIX: {}",
                row.op
            );
        }
    }

    /// Total Exact ops must have no error variants.
    #[test]
    fn guarantee_matrix_exact_total_ops_have_no_variants() {
        for row in GUARANTEE_MATRIX {
            if row.tag == Exact && !row.fallible {
                // Mutant-witness: add an error_variant to zeros — this test catches it.
                assert!(
                    row.error_variants.is_empty(),
                    "op '{}': Exact total ops must have no error_variants",
                    row.op
                );
            }
        }
    }

    /// Every fallible op (except `map`) must name at least one error variant.
    #[test]
    fn guarantee_matrix_fallible_ops_name_variants() {
        for row in GUARANTEE_MATRIX {
            // `map` is special: fallibility/variants are f's, which are caller-supplied.
            if row.fallible && row.op != "map" {
                // Mutant-witness: remove an error_variant from a fallible row — this test fails.
                assert!(
                    !row.error_variants.is_empty(),
                    "op '{}': fallible ops must name at least one error variant",
                    row.op
                );
            }
        }
    }

    /// Non-Exact ops that are EXPLAIN-able must have a float (Proven/Empirical) tag,
    /// except `from_slice` which is Exact+EXPLAIN-able (names which check failed — C3).
    #[test]
    fn guarantee_matrix_explainable_implies_float_tag_or_from_slice() {
        for row in GUARANTEE_MATRIX {
            if row.explainable && row.tag == Exact {
                assert!(
                    row.op == "from_slice" || row.op == "full",
                    "unexpected Exact+explainable op: {} (only from_slice/full are Exact+explainable)",
                    row.op
                );
            }
        }
    }

    /// Integer design-claim rows must be Exact and non-fallible.
    #[test]
    fn guarantee_matrix_int_rows_are_exact_and_total() {
        let int_rows = [
            "add_int",
            "sub_int",
            "hadamard_int",
            "scale_int",
            "sum_int",
            "l1_norm_int",
            "dot_int",
        ];
        for name in int_rows {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == name)
                .unwrap_or_else(|| panic!("missing integer design-claim row: {name}"));
            // Mutant-witness: change an int row's tag to Empirical — this test catches it.
            assert_eq!(row.tag, Exact, "integer op '{name}' must be Exact");
            assert!(!row.fallible, "integer op '{name}' must be total");
            assert!(
                !row.explainable,
                "integer op '{name}' has no bound to explain"
            );
        }
    }

    /// Float accumulation/reduction/similarity rows must be Empirical (FLAG Q1/Q2 conservative).
    #[test]
    fn guarantee_matrix_float_accumulation_rows_are_empirical() {
        let empirical_rows = [
            "sum_float",
            "l1_norm_float",
            "l2_norm_float",
            "dot_float",
            "cosine_float",
        ];
        for name in empirical_rows {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == name)
                .unwrap_or_else(|| panic!("missing float accumulation row: {name}"));
            // Mutant-witness: change sum_float tag to Proven — this test catches it.
            assert_eq!(
                row.tag, Empirical,
                "float accumulation op '{name}' must be Empirical (FLAG Q1/Q2)"
            );
            assert!(row.explainable, "float op '{name}' must be EXPLAIN-able");
        }
    }

    /// Float elementwise ops (add/sub/scale/hadamard — not neg) must be Proven (FLAG Q1 proviso).
    #[test]
    fn guarantee_matrix_float_elementwise_are_proven() {
        let proven_rows = ["add_float", "sub_float", "hadamard_float", "scale_float"];
        for name in proven_rows {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == name)
                .unwrap_or_else(|| panic!("missing float elementwise row: {name}"));
            // Mutant-witness: change add_float to Empirical — this test catches it.
            assert_eq!(
                row.tag, Proven,
                "float elementwise op '{name}' must be Proven (Q1 finalized; M-512-checked)"
            );
            assert!(
                row.explainable,
                "float elementwise op '{name}' must be EXPLAIN-able"
            );
        }
    }

    /// neg must be Exact (no rounding on a symmetric grid).
    #[test]
    fn guarantee_matrix_neg_is_exact() {
        let row = GUARANTEE_MATRIX
            .iter()
            .find(|r| r.op == "neg_float")
            .expect("neg_float row missing");
        assert_eq!(row.tag, Exact, "neg must be Exact");
        assert!(!row.fallible, "neg must be total");
        assert!(!row.explainable, "neg carries no bound (M-I1)");
    }

    /// cosine must be fallible with ZeroNorm (C1/G2).
    #[test]
    fn guarantee_matrix_cosine_has_zero_norm_error() {
        let row = GUARANTEE_MATRIX
            .iter()
            .find(|r| r.op == "cosine_float")
            .expect("cosine_float row missing");
        // Mutant-witness: removing ZeroNorm from cosine_float.error_variants makes this fail.
        assert!(row.fallible, "cosine must be fallible");
        assert!(
            row.error_variants.contains(&"ZeroNorm"),
            "cosine must name ZeroNorm as an error variant (C1/G2)"
        );
    }

    // ── C1 — never-silent tests ───────────────────────────────────────────────

    /// from_slice returns LenMismatch for the wrong slice length — not a panic, not silent.
    #[test]
    fn from_slice_len_mismatch_is_explicit() {
        let s = StdDense::new(4, ScalarKind::F32).unwrap();
        // Mutant-witness: if LenMismatch were a panic instead of a Result, this test fails.
        let err = s.from_slice(&[1.0_f64, 2.0, 3.0]).unwrap_err();
        assert_eq!(
            err,
            StdDenseError::LenMismatch {
                expected: 4,
                got: 3
            },
            "from_slice: wrong-length input must be LenMismatch"
        );
    }

    /// from_slice returns OffGrid for a non-representable value — not a silent re-round.
    #[test]
    fn from_slice_off_grid_is_explicit() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        // 0.1 is not exactly representable as f32.
        // Mutant-witness: if OffGrid were silently rounded, this would return Ok.
        let err = s.from_slice(&[1.0_f64, 0.1]).unwrap_err();
        assert_eq!(
            err,
            StdDenseError::OffGrid {
                dtype: ScalarKind::F32,
                index: 1
            },
            "from_slice: off-grid element must be OffGrid, not silently re-rounded"
        );
    }

    /// cosine on a zero vector returns ZeroNorm — not NaN, not a silent 0.0.
    #[test]
    fn cosine_zero_norm_is_explicit() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 0.0, 0.0]).unwrap();
        let zero = s.zeros();
        // Mutant-witness: if cosine returned Ok(0.0) instead of Err(ZeroNorm), this fails.
        let err = s.cosine(&a, &zero).unwrap_err();
        assert_eq!(
            err,
            StdDenseError::ZeroNorm,
            "cosine on zero vector must be ZeroNorm"
        );
        let err2 = s.cosine(&zero, &a).unwrap_err();
        assert_eq!(
            err2,
            StdDenseError::ZeroNorm,
            "cosine: zero first arg must also be ZeroNorm"
        );
    }

    /// cosine result is not NaN (never a sentinel).
    #[test]
    fn cosine_result_is_never_nan() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 0.0]).unwrap();
        let b = s.from_slice(&[0.0_f64, 1.0]).unwrap();
        let (c, _) = s.cosine(&a, &b).unwrap();
        assert!(!c.is_nan(), "cosine must never be NaN");
    }

    // ── C2 — honest per-op tag tests ──────────────────────────────────────────

    /// zeros produces an Exact value (no bound — M-I1).
    #[test]
    fn zeros_is_exact_no_bound() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let z = s.zeros();
        assert_eq!(z.meta().guarantee(), Exact, "zeros must be Exact");
        assert!(
            z.meta().bound().is_none(),
            "Exact ops carry no bound (M-I1)"
        );
        if let mycelium_core::Payload::Scalars(xs) = z.payload() {
            assert_eq!(xs.len(), 3);
            assert!(xs.iter().all(|&x| x == 0.0));
        } else {
            panic!("zeros: expected Scalars payload");
        }
    }

    /// full produces an Exact value.
    #[test]
    fn full_is_exact() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let v = s.full(1.5).unwrap();
        assert_eq!(v.meta().guarantee(), Exact, "full must be Exact");
        if let mycelium_core::Payload::Scalars(xs) = v.payload() {
            assert!(xs.iter().all(|&x| x == 1.5));
        } else {
            panic!("full: expected Scalars payload");
        }
    }

    /// full returns a kernel error for off-grid scalar (C1).
    #[test]
    fn full_off_grid_is_explicit() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        // 0.1 is not exactly f32 — must error, not silently re-round.
        let err = s.full(0.1_f64).unwrap_err();
        assert!(
            matches!(err, StdDenseError::Kernel(_)),
            "full with off-grid scalar must return a Kernel error"
        );
    }

    /// add returns Proven tag and correct eps for F32.
    #[test]
    fn add_carries_proven_bound_f32() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0]).unwrap();
        let b = s.from_slice(&[0.5_f64, -1.0]).unwrap();
        let (v, bound) = s.add(&a, &b).unwrap();
        // Mutant-witness: if add were tagged Empirical instead of Proven, this fails.
        assert_eq!(v.meta().guarantee(), Proven, "add must be Proven (FLAG Q1)");
        assert_eq!(bound.strength, Proven);
        assert_eq!(bound.norm, NormKind::Rel);
        assert!(bound.eps > 0.0 && bound.eps.is_finite());
        assert_eq!(
            bound.eps, F32_OP_REL_EPS,
            "F32 add eps must equal F32_OP_REL_EPS"
        );
    }

    /// sub carries a Proven bound.
    #[test]
    fn sub_carries_proven_bound() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[2.0_f64, 3.0]).unwrap();
        let b = s.from_slice(&[0.5_f64, 1.0]).unwrap();
        let (v, bound) = s.sub(&a, &b).unwrap();
        assert_eq!(v.meta().guarantee(), Proven);
        assert_eq!(bound.strength, Proven);
    }

    /// neg is Exact and returns no OpBound (M-I1).
    #[test]
    fn neg_is_exact_no_op_bound() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, -2.0, 3.0]).unwrap();
        let v = s.neg(&a).unwrap();
        // Mutant-witness: if neg were tagged Proven, meta().bound() would be Some — this catches it.
        assert_eq!(v.meta().guarantee(), Exact, "neg must be Exact");
        assert!(
            v.meta().bound().is_none(),
            "neg: Exact ops carry no bound (M-I1)"
        );
        if let mycelium_core::Payload::Scalars(xs) = v.payload() {
            assert_eq!(xs, &[-1.0_f64, 2.0, -3.0]);
        }
    }

    /// scale carries a Proven bound.
    #[test]
    fn scale_carries_proven_bound() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0]).unwrap();
        let (v, bound) = s.scale(&a, 2.0).unwrap();
        assert_eq!(v.meta().guarantee(), Proven);
        assert_eq!(bound.strength, Proven);
        assert_eq!(bound.eps, F32_OP_REL_EPS);
    }

    /// sum returns Empirical bound with eps = 2·n·F32_OP_REL_EPS (FLAG Q1 conservative).
    #[test]
    fn sum_carries_empirical_bound_proportional_to_n() {
        let s = StdDense::new(4, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 1.0, 1.0, 1.0]).unwrap();
        let (total, bound) = s.sum(&a).unwrap();
        assert_eq!(total, 4.0);
        // Mutant-witness: if sum were tagged Proven instead of Empirical, this fails.
        assert_eq!(bound.strength, Empirical, "sum must be Empirical (FLAG Q1)");
        assert_eq!(bound.norm, NormKind::Rel);
        let expected_eps = accumulation_eps_f32(4);
        assert_eq!(
            bound.eps, expected_eps,
            "sum eps must equal accumulation_eps_f32(4)"
        );
        assert!(bound.eps > 0.0);
    }

    /// l1_norm returns Empirical bound.
    #[test]
    fn l1_norm_carries_empirical_bound() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, -2.0, 3.0]).unwrap();
        let (norm, bound) = s.l1_norm(&a).unwrap();
        assert!((norm - 6.0).abs() < 1e-10, "l1_norm of [1,-2,3] must be 6");
        assert_eq!(bound.strength, Empirical, "l1_norm must be Empirical");
    }

    /// l2_norm returns Empirical bound and correct value for a 3-4-5 triple.
    #[test]
    fn l2_norm_carries_empirical_bound_and_correct_value() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[3.0_f64, 4.0, 0.0]).unwrap();
        let (norm, bound) = s.l2_norm(&a).unwrap();
        assert!((norm - 5.0).abs() < 1e-10, "l2_norm of [3,4,0] must be 5");
        // Mutant-witness: if l2_norm were tagged Proven instead of Empirical, this fails.
        assert_eq!(
            bound.strength, Empirical,
            "l2_norm must be Empirical (FLAG Q2)"
        );
        assert!(bound.eps > 0.0 && bound.eps.is_finite());
    }

    /// dot returns Empirical bound.
    #[test]
    fn dot_carries_empirical_bound() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 0.0, 0.0]).unwrap();
        let b = s.from_slice(&[1.0_f64, 0.0, 0.0]).unwrap();
        let (d, bound) = s.dot(&a, &b).unwrap();
        assert_eq!(d, 1.0);
        assert_eq!(bound.strength, Empirical, "dot must be Empirical (FLAG Q1)");
    }

    /// cosine of orthogonal unit vectors is ~0 with Empirical bound.
    #[test]
    fn cosine_orthogonal_is_near_zero() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 0.0]).unwrap();
        let b = s.from_slice(&[0.0_f64, 1.0]).unwrap();
        let (c, bound) = s.cosine(&a, &b).unwrap();
        assert!(c.abs() < 1e-12, "cosine of orthogonal vectors must be ~0");
        assert_eq!(
            bound.strength, Empirical,
            "cosine must be Empirical (FLAG Q2)"
        );
        assert!(bound.eps > 0.0 && bound.eps.is_finite());
    }

    /// cosine(v, v) ≈ 1.
    #[test]
    fn cosine_self_is_one() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0, 2.0]).unwrap();
        let (c, _) = s.cosine(&a, &a).unwrap();
        assert!((c - 1.0).abs() < 1e-12, "cosine(v, v) must be ~1");
    }

    // ── C3 — EXPLAIN / bound artifact tests ───────────────────────────────────

    /// OpBound.to_core_bound: Proven → ProvenThm, Empirical → EmpiricalFit(trials≥1).
    #[test]
    fn op_bound_to_core_bound_basis_matches_strength() {
        let proven = OpBound {
            eps: F32_OP_REL_EPS,
            norm: NormKind::Rel,
            strength: Proven,
            basis: "some theorem".to_owned(),
        };
        let core = proven.to_core_bound();
        // Mutant-witness: if Proven mapped to EmpiricalFit instead of ProvenThm, this fails.
        assert!(
            matches!(core.basis, BoundBasis::ProvenThm { .. }),
            "Proven strength must produce ProvenThm basis"
        );

        let empirical = OpBound {
            eps: 1e-6,
            norm: NormKind::Rel,
            strength: Empirical,
            basis: "some fit".to_owned(),
        };
        let core = empirical.to_core_bound();
        // Mutant-witness: if EmpiricalFit had trials=0, it would fail the A6-02 well-formedness check.
        assert!(
            matches!(core.basis, BoundBasis::EmpiricalFit { trials: 1, .. }),
            "Empirical strength must produce EmpiricalFit with trials=1 (A6-02)"
        );
    }

    /// EXPLAIN artifacts (basis strings) are non-empty for all float ops.
    #[test]
    fn explain_artifacts_are_non_empty() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0]).unwrap();
        let b = s.from_slice(&[0.5_f64, 1.0]).unwrap();

        let (_, bound) = s.add(&a, &b).unwrap();
        assert!(
            !bound.basis.is_empty(),
            "add EXPLAIN basis must be non-empty"
        );
        let (_, bound) = s.dot(&a, &b).unwrap();
        assert!(
            !bound.basis.is_empty(),
            "dot EXPLAIN basis must be non-empty"
        );
        let (_, bound) = s.l2_norm(&a).unwrap();
        assert!(
            !bound.basis.is_empty(),
            "l2_norm EXPLAIN basis must be non-empty"
        );
        let (_, bound) = s.cosine(&a, &b).unwrap();
        assert!(
            !bound.basis.is_empty(),
            "cosine EXPLAIN basis must be non-empty"
        );
    }

    // ── Accumulation bound property tests (A2-01 outward-rounding) ────────────

    /// accumulation_eps_f32 is outward-rounded: eps >= n * F32_OP_REL_EPS (A2-01).
    #[test]
    fn accumulation_eps_f32_is_outward_rounded() {
        for n in [0_usize, 1, 4, 128, 1024, 65536] {
            let eps = accumulation_eps_f32(n);
            let lower_bound = (n as f64) * F32_OP_REL_EPS;
            assert!(
                eps >= lower_bound,
                "accumulation_eps_f32({n}) = {eps} must be >= n * F32_OP_REL_EPS = {lower_bound} (A2-01)"
            );
        }
    }

    /// accumulation_eps_bf16 is outward-rounded.
    #[test]
    fn accumulation_eps_bf16_is_outward_rounded() {
        for n in [0_usize, 1, 4, 128, 1024] {
            let eps = accumulation_eps_bf16(n);
            let lower_bound = (n as f64) * BF16_OP_REL_EPS;
            assert!(
                eps >= lower_bound,
                "accumulation_eps_bf16({n}) = {eps} must be >= n * BF16_OP_REL_EPS = {lower_bound}"
            );
        }
    }

    // ── map tag composition tests ─────────────────────────────────────────────

    /// map(Exact input, Exact f_tag) → Exact (meet(Exact, Exact) = Exact).
    #[test]
    fn map_exact_exact_produces_exact() {
        let s = StdDense::new(3, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
        let (result, tag) = s
            .map(&a, Exact, |x| -> Result<f64, StdDenseError> { Ok(x) })
            .unwrap();
        // Mutant-witness: if map upgraded Exact→Proven, this fails.
        assert_eq!(tag, Exact, "map(Exact, Exact) must be Exact");
        assert_eq!(result.meta().guarantee(), Exact);
    }

    /// map(Exact input, Empirical f_tag) → Empirical (meet degrades — VR-5).
    #[test]
    fn map_exact_empirical_degrades_to_empirical() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0]).unwrap();
        let (_, tag) = s
            .map(&a, Empirical, |x| -> Result<f64, StdDenseError> {
                Ok(x * 2.0)
            })
            .unwrap();
        // Mutant-witness: if map upgraded Empirical→Exact, this fails.
        assert_eq!(
            tag, Empirical,
            "map(Exact, Empirical) must degrade to Empirical (VR-5)"
        );
    }

    // ── BF16 sanity ───────────────────────────────────────────────────────────

    /// BF16 space from_slice + add carry BF16 Proven eps.
    #[test]
    fn bf16_add_carries_bf16_proven_eps() {
        let s = StdDense::new(2, ScalarKind::Bf16).unwrap();
        let a = s.from_slice(&[1.5_f64, -2.0]).unwrap();
        let b = s.from_slice(&[1.5_f64, 2.0]).unwrap();
        let (v, bound) = s.add(&a, &b).unwrap();
        assert_eq!(v.meta().guarantee(), Proven);
        assert_eq!(
            bound.eps, BF16_OP_REL_EPS,
            "BF16 eps must be BF16_OP_REL_EPS"
        );
    }

    /// BF16 off-grid check: 1.501953125 is f32-exact but off the BF16 grid.
    #[test]
    fn bf16_from_slice_off_grid_is_explicit() {
        let s = StdDense::new(2, ScalarKind::Bf16).unwrap();
        // 1.5 is on the BF16 grid; 1.501953125 = 1.5 + 2^-9 is f32-exact but off BF16.
        let err = s.from_slice(&[1.5_f64, 1.501_953_125]).unwrap_err();
        assert_eq!(
            err,
            StdDenseError::OffGrid {
                dtype: ScalarKind::Bf16,
                index: 1
            },
            "BF16 off-grid element must be OffGrid"
        );
    }

    // ── Unsupported dtype ─────────────────────────────────────────────────────

    /// F16 and F64 return UnsupportedDtype (honest v1 scope, FLAG Q3).
    #[test]
    fn unsupported_dtypes_return_kernel_error() {
        let err = StdDense::new(4, ScalarKind::F64).unwrap_err();
        // Mutant-witness: if F64 were silently coerced to F32, this would return Ok.
        assert!(
            matches!(
                err,
                StdDenseError::Kernel(DenseError::UnsupportedDtype { .. })
            ),
            "F64 must be Kernel(UnsupportedDtype)"
        );
        let err2 = StdDense::new(4, ScalarKind::F16).unwrap_err();
        assert!(
            matches!(
                err2,
                StdDenseError::Kernel(DenseError::UnsupportedDtype { .. })
            ),
            "F16 must be Kernel(UnsupportedDtype)"
        );
    }

    // ── hadamard ─────────────────────────────────────────────────────────────

    /// hadamard([1,2], [3,4]) = [3,8] with Proven tag.
    #[test]
    fn hadamard_correct_value_and_proven_tag() {
        let s = StdDense::new(2, ScalarKind::F32).unwrap();
        let a = s.from_slice(&[1.0_f64, 2.0]).unwrap();
        let b = s.from_slice(&[3.0_f64, 4.0]).unwrap();
        let (v, bound) = s.hadamard(&a, &b).unwrap();
        assert_eq!(bound.strength, Proven, "hadamard must be Proven");
        if let mycelium_core::Payload::Scalars(xs) = v.payload() {
            assert!((xs[0] - 3.0).abs() < 1e-10, "hadamard: 1*3=3");
            assert!((xs[1] - 8.0).abs() < 1e-10, "hadamard: 2*4=8");
        } else {
            panic!("hadamard: expected Scalars payload");
        }
    }

    // ── Randomized property tests (deterministic LCG, no proptest dep) ────────

    /// Property: the sum declared ε (Empirical) is a true upper bound over the actual f64
    /// accumulation error vs a naive reference (50 deterministic samples, FLAG Q1 conservative).
    #[test]
    fn sum_empirical_bound_holds_on_sample() {
        const N: usize = 64;
        let s = StdDense::new(N as u32, ScalarKind::F32).unwrap();
        // Deterministic LCG (Knuth multiplicative): one fixed seed, 50 samples.
        let mut rng: u64 = 0xdead_beef_cafe_babe;
        let samples = 50_usize;
        for _ in 0..samples {
            let xs: Vec<f64> = (0..N)
                .map(|_| {
                    rng = rng
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    // Build a normal f32 in [2.0, 4.0) — always on the f32 grid.
                    let bits = (((rng >> 33) as u32) & 0x007F_FFFF) | 0x4000_0000_u32;
                    f64::from(f32::from_bits(bits))
                })
                .collect();
            let a = s.from_slice(&xs).unwrap();
            let (total, bound) = s.sum(&a).unwrap();
            let reference: f64 = xs.iter().copied().sum();
            assert!(bound.eps >= 0.0 && bound.eps.is_finite());
            let rel_diff = if reference.abs() > 1e-300 {
                (total - reference).abs() / reference.abs()
            } else {
                0.0
            };
            // The declared eps (2·N·u ≈ 7.6e-6) must be >= the actual relative error
            // (f64 accumulation over f32 values gives ~1e-15 actual error).
            assert!(
                bound.eps >= rel_diff,
                "sum declared eps ({}) must be >= actual rel error ({}) — A2-01",
                bound.eps,
                rel_diff
            );
        }
    }

    /// Property: the dot product declared ε holds on a deterministic sample (50 vectors, FLAG Q1).
    #[test]
    fn dot_empirical_bound_holds_on_sample() {
        const N: usize = 32;
        let s = StdDense::new(N as u32, ScalarKind::F32).unwrap();
        let mut rng: u64 = 0xfeed_face_dead_beef;
        let samples = 50_usize;
        for _ in 0..samples {
            let gen = |rng: &mut u64| -> Vec<f64> {
                (0..N)
                    .map(|_| {
                        *rng = rng
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let bits = ((*rng >> 33) as u32 & 0x007F_FFFF) | 0x4000_0000_u32;
                        f64::from(f32::from_bits(bits))
                    })
                    .collect()
            };
            let xs = gen(&mut rng);
            let ys = gen(&mut rng);
            let a = s.from_slice(&xs).unwrap();
            let b = s.from_slice(&ys).unwrap();
            let (d, bound) = s.dot(&a, &b).unwrap();
            let reference: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
            assert!(bound.eps >= 0.0 && bound.eps.is_finite());
            let rel_diff = if reference.abs() > 1e-300 {
                (d - reference).abs() / reference.abs()
            } else {
                0.0
            };
            assert!(
                bound.eps >= rel_diff,
                "dot declared eps ({}) must be >= actual rel error ({}) — A2-01",
                bound.eps,
                rel_diff
            );
        }
    }
}
