//! Native direct-LLVM codegen of **`Repr::Dense` element-wise ops** — the un-quantized F32/BF16
//! fragment (M-853; epic E25-1; **RFC-0039 §5.1** *Native Dense lowering*, Accepted 2026-06-30;
//! ADR-034 re-gating native AOT into lang 1.0.0; RFC-0004 §6 *no-opaque-lowering* / §11 the additive
//! direct-LLVM increment pattern; ADR-030 quant-as-`Meta.physical`; DN-01).
//!
//! ## What this lowers
//! The `mycelium-dense::DenseSpace` element-wise surface over the **un-quantized F32/BF16 fragment**
//! (RFC-0039 §5.1, OQ-2/OQ-4 resolved — F32/BF16 first; the ADR-030 quantized dtypes widen as E20-1
//! lands `QuantDesc`, refused never-silently until then):
//!
//! - **`dense.add` / `dense.sub`** — elementwise `a ± b`, computed as a native `f32` op then rounded
//!   to the dtype grid (bf16 round-to-nearest-even done in IR on the bit pattern), mirroring
//!   `DenseSpace::add_values`/`sub_values` digit-for-digit. Reference tag **`Proven`** (Higham 2002
//!   Thm 2.2; `F32_OP_REL_EPS = 2⁻²⁴`, `BF16_OP_REL_EPS = 2⁻⁸ + 2⁻²³`).
//! - **`dense.neg`** — elementwise `-a`, exact (`fneg`; the grids are symmetric so no element rounds),
//!   mirroring `DenseSpace::neg_value`. Reference tag **`Exact`**.
//! - **`dense.scale`** — scalar `c · a`, same rounding discipline as `add`, mirroring
//!   `DenseSpace::scale_value`. Reference tag **`Proven`**.
//! - **`dense.dot` / `dense.similarity`** — bare-`f64` *measurements* (no `Meta` tag), mirroring
//!   `DenseSpace::dot`/`similarity`. `dot` is an `f64` accumulation; `similarity` is
//!   `dot / (‖a‖·‖b‖)` (0 on a zero-norm operand).
//!
//! Every step is **explicit per-element textual IR** — one op per element, no opaque pass (RFC-0004
//! §6). A leading IR comment records the op, dim, dtype, rounding, the inspectable `Meta.physical`
//! schedule, and the guarantee (no black box; ADR-006/G2).
//!
//! ## Faithfulness to the reference (the load-bearing decision)
//! The native lowering is the **performance layer**, never the source of meaning — `mycelium-dense`
//! (and the interpreter above it) is the trusted base (NFR-7). The native op's payload is
//! **observably equal** to `DenseSpace`'s (`repr + payload + guarantee`), and the per-element
//! side-conditions the reference checks (finite, exactly on-grid input, normal-not-subnormal /
//! non-overflowing result) are **re-checked at runtime in the emitted IR**, refusing **never-silently**
//! through a sentinel read-back (matching `DenseError::NonFinite`/`NotOnGrid`/`SubnormalUnsupported`/
//! `Overflow`). The native path **does not** ship a second, divergent Dense semantics (DRY).
//!
//! ## Guarantee tag (VR-5 — never upgraded past the basis)
//! The read-back [`Value`] carries the **reference's** per-op tag (`Proven` for `add`/`sub`/`scale`,
//! `Exact` for `neg`) so the differential's observable matches — but the **codegen's own confidence
//! that native ≡ reference is `Empirical`**, established by the M-210 three-way differential plus the
//! `cargo-mutants` witness, **not** by a proof object linked into this codegen. Carrying the Higham
//! bound *value* is honest (the rounding the IR performs is the same single/double rounding the
//! theorem bounds, and the side-conditions are re-checked at runtime), but **the theorem is referenced,
//! not re-derived with the proof in hand here** — exactly the `swap_codegen` reasoning. Upgrading the
//! *codegen-correctness* claim to `Proven` would need the M-211/Higham proof wired as a checked basis
//! in this module — **flagged, not assumed** (VR-5). So: the *value* is the reference's
//! `Proven`/`Exact` tag (it must be, to be the same value), and the *codegen* claim is `Empirical`.
//!
//! ## Never-silent refusals (G2)
//! - **`F16`/`F64`** dtype → [`DenseAotError::UnsupportedDtype`] (matches `DenseSpace::new`).
//! - **Any quantized Dense value** → [`DenseAotError::QuantRefused`] — the ADR-030 quant descriptor is
//!   not yet in the value model (E20-1; RFC-0039 §5.1 honesty note), so a quantized Dense is refused,
//!   never silently treated as un-quantized.
//! - **Non-finite / off-grid input** → refused at lowering (matches `DenseError::NonFinite`/`NotOnGrid`).
//! - **Subnormal / overflowing result** → refused **never-silently** at runtime via the sentinel
//!   read-back (matches `DenseError::SubnormalUnsupported`/`Overflow`).
//! - **dim / dtype mismatch** → refused (matches `DenseError::DimMismatch`/`DtypeMismatch`).
//!
//! ## Direct-LLVM first; dialect later (RFC-0039 §5.1 / RFC-0004 §11)
//! This is the direct-LLVM increment. The MLIR-dialect path honestly **refuses** Dense
//! (`DialectError::Unsupported`, `dialect/native.rs`), so the three-way differential's dialect leg is a
//! never-faked refusal (the differential reduces to two-way for Dense).
//!
//! **Submodule confinement:** zero `unsafe` (compiler-enforced by the crate's `#![forbid]`).

use std::fmt;
use std::fmt::Write as _; // `writeln!` into a String never fails — call sites discard the Result.
use std::process::Command;

use mycelium_core::{
    operation_hash, Bound, BoundBasis, BoundKind, GuaranteeStrength, Meta, NormKind, Payload,
    PhysicalLayout, Provenance, Repr, ScalarKind, Value, WfError,
};
use mycelium_dense::{BF16_OP_REL_EPS, DENSE_MIN_NORMAL, F32_OP_REL_EPS};

use crate::llvm::{path, run_tool, unique_tmp_dir, TmpDir};

// ─── the Dense op surface this module lowers (un-quantized F32/BF16; RFC-0039 §5.1) ─────────────

/// The Dense element-wise ops native codegen lowers — the `mycelium-dense::DenseSpace` surface
/// (RFC-0039 §5.1). `Dot`/`Similarity` are bare-`f64` *measurements* (no `Meta` tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseCgOp {
    /// Elementwise `a + b` (Proven rounding bound).
    Add,
    /// Elementwise `a − b` (Proven rounding bound).
    Sub,
    /// Elementwise `-a` (Exact — symmetric grids).
    Neg,
    /// Scalar `c · a` (Proven rounding bound). The scalar `c` rides the program (see [`DenseProgram`]).
    Scale,
    /// `Σ aᵢ·bᵢ` in `f64` — a bare measurement (no `Meta`).
    Dot,
    /// Cosine similarity in `[-1, 1]` — a bare measurement (no `Meta`).
    Similarity,
}

impl DenseCgOp {
    /// The `mycelium-dense` operation name (matches the reference's `operation_hash` keys), recorded in
    /// the IR comment + provenance so the lowered op is never anonymous (G2).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            DenseCgOp::Add => "dense.add",
            DenseCgOp::Sub => "dense.sub",
            DenseCgOp::Neg => "dense.neg",
            DenseCgOp::Scale => "dense.scale",
            DenseCgOp::Dot => "dense.dot",
            DenseCgOp::Similarity => "dense.similarity",
        }
    }

    /// Whether the op produces a Dense `Value` (vs a bare-`f64` measurement). `dot`/`similarity` are
    /// measurements.
    #[must_use]
    pub fn is_value_op(self) -> bool {
        !matches!(self, DenseCgOp::Dot | DenseCgOp::Similarity)
    }

    /// The honest reference guarantee for a value op (mirrors `DenseSpace::op_guarantee`). `None` for a
    /// measurement op (no `Meta` to tag).
    #[must_use]
    pub fn reference_guarantee(self) -> Option<GuaranteeStrength> {
        match self {
            DenseCgOp::Neg => Some(GuaranteeStrength::Exact),
            DenseCgOp::Add | DenseCgOp::Sub | DenseCgOp::Scale => Some(GuaranteeStrength::Proven),
            DenseCgOp::Dot | DenseCgOp::Similarity => None,
        }
    }
}

/// A native-Dense lowering program: one element-wise op over a Dense space, plus its operand(s).
/// The operands are reference `Value`s the caller has built through `DenseSpace` (so they are exact,
/// on-grid, and dim/dtype-consistent). Single-source-of-truth for [`emit_dense_llvm_ir`],
/// [`dense_compile`], and the read-back shape (so they can never disagree).
#[derive(Debug, Clone)]
pub struct DenseProgram {
    /// The op to lower.
    pub op: DenseCgOp,
    /// Dimensionality of the Dense space.
    pub dim: u32,
    /// Element dtype (`F32` or `Bf16` — F16/F64 are an explicit refusal).
    pub dtype: ScalarKind,
    /// First operand elements (length `dim`, exactly on the dtype grid).
    pub a: Vec<f64>,
    /// Second operand elements, for binary ops (`add`/`sub`/`dot`/`similarity`). `None` for unary
    /// (`neg`/`scale`).
    pub b: Option<Vec<f64>>,
    /// The scalar factor for `scale` (on-grid). `None` for non-`scale` ops.
    pub scale: Option<f64>,
}

/// What a Dense native op produces: a Dense `Value` (for `add`/`sub`/`neg`/`scale`) or a bare-`f64`
/// measurement (for `dot`/`similarity`). Never-silent: the variant is the op's honest output shape.
/// The `Value` is **boxed** — a `Value` is ~240 bytes while a measurement is 8, so an unboxed enum
/// would bloat every `Measurement` to the `Value` size (clippy `large_enum_variant`); boxing keeps the
/// common-case measurement small without changing the never-silent shape.
#[derive(Debug, Clone, PartialEq)]
pub enum DenseResult {
    /// A Dense `Value` (boxed) carrying the reference's per-op guarantee tag.
    Value(Box<Value>),
    /// A bare-`f64` measurement (no `Meta` — mirrors `DenseSpace::dot`/`similarity`).
    Measurement(f64),
}

// ─── explicit, never-silent failure of the native Dense path (G2) ───────────────────────────────

/// Why the native Dense path could not lower/run a program — **always explicit, never silent** (G2).
/// Mirrors the reference [`mycelium_dense::DenseError`] refusals where they overlap, and adds the
/// native-path-specific toolchain / quant-gate refusals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseAotError {
    /// `F16`/`F64` — outside the un-quantized F32/BF16 fragment (matches `DenseSpace::new`; RFC-0039
    /// §5.1). The trusted interpreter + `mycelium-dense` refuse these too — never a silent coercion.
    UnsupportedDtype(ScalarKind),
    /// A **quantized** Dense value (an ADR-030 `QuantDesc`-bearing repr). The descriptor is not yet in
    /// the value model (E20-1; RFC-0039 §5.1 honesty note) — refused, never treated as un-quantized.
    QuantRefused(String),
    /// An operand's dimensionality disagrees with the program's `dim` (matches
    /// `DenseError::DimMismatch`).
    DimMismatch {
        /// Expected dimension.
        expected: u32,
        /// Got.
        got: usize,
    },
    /// An input element is non-finite (matches `DenseError::NonFinite`).
    NonFinite(usize),
    /// An input element / the scale factor is not exactly on the dtype grid (matches
    /// `DenseError::NotOnGrid`/`ScalarOffGrid`).
    OffGrid(String),
    /// The program is malformed for its op (e.g. a binary op with no second operand) — an internal
    /// contract violation, surfaced explicitly rather than panicking.
    Malformed(String),
    /// A result element is subnormal — outside the cited theorem's side-conditions (matches
    /// `DenseError::SubnormalUnsupported`). Detected at runtime, surfaced via the sentinel read-back.
    Subnormal,
    /// A result element overflowed the dtype's finite range (matches `DenseError::Overflow`).
    /// Detected at runtime, surfaced via the sentinel read-back — never a silent ±Inf.
    Overflow,
    /// The native toolchain (`llc`/`clang`) is absent — callers **skip**, not fail (house idiom).
    ToolchainMissing(String),
    /// `llc`/`clang` ran but returned non-zero (compile failure).
    Compile(String),
    /// The artifact failed to run or produced unreadable output.
    Run(String),
    /// The native stdout did not parse back into the expected shape.
    Parse(String),
    /// Reconstructing the result `Value` failed its well-formedness check.
    Wf(String),
}

impl fmt::Display for DenseAotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenseAotError::UnsupportedDtype(d) => write!(
                f,
                "dtype {d:?} is outside the native Dense F32/BF16 fragment (M-853; matches \
                 DenseSpace::new — F16/F64 refused, never coerced; G2)"
            ),
            DenseAotError::QuantRefused(s) => write!(
                f,
                "quantized Dense value refused: {s} — the ADR-030 QuantDesc is not yet in the value \
                 model (E20-1; RFC-0039 §5.1); refused, never treated as un-quantized (G2)"
            ),
            DenseAotError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            DenseAotError::NonFinite(i) => {
                write!(
                    f,
                    "input element {i} is NaN/Inf — no defined rounding bound"
                )
            }
            DenseAotError::OffGrid(s) => write!(f, "off the declared dtype grid: {s}"),
            DenseAotError::Malformed(s) => write!(f, "malformed Dense program: {s}"),
            DenseAotError::Subnormal => write!(
                f,
                "result element is subnormal — outside the proven relative-bound range (matches \
                 DenseError::SubnormalUnsupported; never silent — G2)"
            ),
            DenseAotError::Overflow => write!(
                f,
                "result element overflows the dtype's finite range (matches DenseError::Overflow; \
                 never a silent ±Inf — G2/SC-3)"
            ),
            DenseAotError::ToolchainMissing(t) => write!(f, "native toolchain missing: {t}"),
            DenseAotError::Compile(e) => write!(f, "native compile failed: {e}"),
            DenseAotError::Run(e) => write!(f, "native run failed: {e}"),
            DenseAotError::Parse(e) => write!(f, "native output parse failed: {e}"),
            DenseAotError::Wf(e) => write!(f, "result well-formedness violation: {e}"),
        }
    }
}

impl std::error::Error for DenseAotError {}

// ─── the inspectable EXPLAIN record (RFC-0004 §6; ADR-006 — no black box) ───────────────────────

/// The inspectable record of how a Dense op was lowered — the EXPLAIN payload (RFC-0004 §6; no black
/// box). Carries the op, dim, dtype, the per-op relative ε (`0` for exact `neg` / measurements), the
/// inspectable `Meta.physical` schedule (`DenseArray` for the un-quantized fragment — the
/// schedule-as-metadata discipline, DN-01/ADR-030), the never-upgraded codegen guarantee, and the
/// quant status (always `un-quantized` today; the E20-1 widening is recorded here when it lands).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseExplain {
    /// The op name (`dense.add`, …).
    pub op: &'static str,
    /// Dimensionality.
    pub dim: u32,
    /// Element dtype.
    pub dtype: ScalarKind,
    /// The per-element relative ε the value op carries (`0.0` for exact `neg` / measurements).
    pub rel_eps: f64,
    /// The inspectable physical schedule (`Some(DenseArray)` for a value op; `None` for a measurement
    /// whose output is a bare `f64`). The ADR-030 quant granularity / accumulator width / packing
    /// schedule extend this record as E20-1 lands them (RFC-0039 §5.1) — never a hidden choice.
    pub physical: Option<PhysicalLayout>,
    /// The reference guarantee the read-back value carries (`None` for a measurement).
    pub reference_guarantee: Option<GuaranteeStrength>,
    /// The **codegen-correctness** guarantee (`Empirical` — differential + mutant-witness, never a
    /// proof object linked here; VR-5).
    pub codegen_guarantee: GuaranteeStrength,
    /// The quant status — `un-quantized` today; the E20-1 widening records the `QuantDesc` here.
    pub quant: &'static str,
}

/// The codegen-correctness guarantee for the native Dense path: **`Empirical`** (the basis is the
/// M-210 differential together with the `cargo-mutants` witness; no proof object is linked into this
/// codegen — VR-5). Exposed so callers / EXPLAIN consumers read the honest codegen-confidence tag,
/// distinct from the reference *value* tag the read-back carries.
pub const DENSE_CODEGEN_GUARANTEE: GuaranteeStrength = GuaranteeStrength::Empirical;

const F32_OP_CITATION: &str = "round-to-nearest relative error ≤ u = β^(1−p)/2 = 2^−24 for IEEE \
     binary32 (β=2, p=24) — Higham, Accuracy and Stability of Numerical Algorithms (2002), Thm 2.2; \
     native f32 op (single rounding); side-conditions checked per element: exact on-grid inputs, \
     finite zero-or-normal result, no overflow";

const BF16_OP_CITATION: &str = "two-rounding composition (1+δ₁)(1+δ₂)−1 ≤ 2^−8 + 2^−23: native f32 \
     op (u₁ = 2^−24) then bfloat16 round-to-nearest (u₂ = 2^−8) — Higham (2002), Thm 2.2 applied at \
     each rounding; side-conditions checked per element at both steps: exact on-grid inputs, finite \
     zero-or-normal results, no overflow";

// ─── grid / dtype helpers (mirror mycelium-dense, kept native so codegen has its own basis) ─────

/// Round an `f32` to the nearest bfloat16 (ties to even), widened back to `f32` bit-exactly — the
/// same grid the reference targets (`mycelium_dense`'s private `round_f32_to_bf16`, re-implemented so
/// the native path has its own host-side basis for grid checks). Caller has excluded NaN/Inf.
/// `pub(crate)` for white-box mutant-witness testing (the bit-twiddle is a correctness helper).
pub(crate) fn round_f32_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16)
}

/// Whether `x` is exactly representable on the `dtype` grid (finite values only) — mirrors
/// `mycelium_dense`'s private `on_grid`. `pub(crate)` for white-box mutant-witness testing.
pub(crate) fn on_grid(dtype: ScalarKind, x: f64) -> bool {
    #[allow(clippy::cast_possible_truncation)] // representability is exactly what we check
    let xf = x as f32;
    if f64::from(xf) != x {
        return false;
    }
    match dtype {
        ScalarKind::F32 => true,
        ScalarKind::Bf16 => round_f32_to_bf16(xf) == xf,
        ScalarKind::F16 | ScalarKind::F64 => false,
    }
}

/// The per-op relative ε for a value op under `dtype` (mirrors `DenseSpace::op_rel_eps`).
fn op_rel_eps(dtype: ScalarKind) -> f64 {
    match dtype {
        ScalarKind::Bf16 => BF16_OP_REL_EPS,
        _ => F32_OP_REL_EPS,
    }
}

/// The cited theorem behind the `Proven` rounding bound, per dtype — F32 single-rounding vs BF16
/// two-rounding (the citation IS the transparency record the `ProvenThm` basis carries; a wrong/blank
/// citation would mis-attribute the bound, so it is mutant-witnessed). `pub(crate)` for white-box test.
pub(crate) fn op_citation(dtype: ScalarKind) -> &'static str {
    match dtype {
        ScalarKind::Bf16 => BF16_OP_CITATION,
        _ => F32_OP_CITATION,
    }
}

// ─── program validation (mirrors the reference's per-element side-condition checks) ─────────────

impl DenseProgram {
    /// Validate the program against the same contract the reference enforces: a supported dtype, the
    /// un-quantized fragment, dim-consistent operands, and finite, exactly-on-grid input elements (and
    /// scale factor). Returns an explicit [`DenseAotError`] for any violation — never a silent coercion
    /// (G2), exactly as `DenseSpace` refuses. `pub(crate)` so the MLIR-dialect sibling emitter
    /// (`dialect::native::dense`, M-856b) reuses the *same* validation — never a second, divergent copy
    /// (DRY).
    pub(crate) fn validate(&self) -> Result<(), DenseAotError> {
        // dtype gate — F16/F64 refused (matches DenseSpace::new).
        match self.dtype {
            ScalarKind::F32 | ScalarKind::Bf16 => {}
            d @ (ScalarKind::F16 | ScalarKind::F64) => {
                return Err(DenseAotError::UnsupportedDtype(d));
            }
        }
        // dim consistency.
        if self.a.len() != self.dim as usize {
            return Err(DenseAotError::DimMismatch {
                expected: self.dim,
                got: self.a.len(),
            });
        }
        // per-element input checks (finite + on-grid) for operand a.
        self.check_elements(&self.a)?;
        // operand b: required for binary ops, must be dim-consistent + on-grid; forbidden for unary.
        match self.op {
            DenseCgOp::Add | DenseCgOp::Sub | DenseCgOp::Dot | DenseCgOp::Similarity => {
                let b = self.b.as_ref().ok_or_else(|| {
                    DenseAotError::Malformed(format!("{} needs a second operand", self.op.name()))
                })?;
                if b.len() != self.dim as usize {
                    return Err(DenseAotError::DimMismatch {
                        expected: self.dim,
                        got: b.len(),
                    });
                }
                self.check_elements(b)?;
            }
            DenseCgOp::Neg => {}
            DenseCgOp::Scale => {
                let c = self.scale.ok_or_else(|| {
                    DenseAotError::Malformed("scale needs a scalar factor".to_owned())
                })?;
                if !c.is_finite() || !on_grid(self.dtype, c) {
                    return Err(DenseAotError::OffGrid(format!(
                        "scale factor {c} is non-finite or off the {:?} grid",
                        self.dtype
                    )));
                }
            }
        }
        Ok(())
    }

    fn check_elements(&self, xs: &[f64]) -> Result<(), DenseAotError> {
        for (i, &x) in xs.iter().enumerate() {
            if !x.is_finite() {
                return Err(DenseAotError::NonFinite(i));
            }
            if !on_grid(self.dtype, x) {
                return Err(DenseAotError::OffGrid(format!(
                    "element {i} ({x}) is not on the {:?} grid",
                    self.dtype
                )));
            }
        }
        Ok(())
    }
}

// ─── the never-silent read-back protocol (float bit-patterns) ───────────────────────────────────

/// The sentinel line a Dense artifact prints when a result element is subnormal — a side-condition the
/// reference also refuses (`DenseError::SubnormalUnsupported`). The read-back turns it into an explicit
/// [`DenseAotError`], never a silent value (G2/SC-3).
pub(crate) const DENSE_SUBNORMAL_SENTINEL: &str = "SUBNORMAL";
/// The overflow sentinel (a non-finite result element — `DenseError::Overflow`).
pub(crate) const DENSE_OVERFLOW_SENTINEL: &str = "OVERFLOW";

// ─── IR emission ────────────────────────────────────────────────────────────────────────────────

/// Emit textual LLVM IR for a Dense element-wise program — a `main` that computes each result element
/// and prints it, then `ret`. One op per element (no opaque pass — RFC-0004 §6). Returns an explicit
/// [`DenseAotError`] for anything outside the F32/BF16 un-quantized fragment. Also returns the
/// inspectable [`DenseExplain`].
pub fn emit_dense_llvm_ir(prog: &DenseProgram) -> Result<(String, DenseExplain), DenseAotError> {
    prog.validate()?;
    let explain = mk_explain(prog);

    let mut out = String::from(
        "; mycelium direct-LLVM Dense codegen (un-quantized F32/BF16 element-wise; M-853; \
         RFC-0039 §5.1)\n",
    );
    emit_explain_comment(&explain, &mut out);
    // printf for the read-back protocol; the never-silent subnormal/overflow path prints a sentinel.
    out.push_str("declare i32 @printf(i8*, ...)\n");
    out.push_str("@.fmt_u64 = private constant [6 x i8] c\"%llu \\00\"\n");
    out.push_str("@.fmt_nl = private constant [2 x i8] c\"\\0A\\00\"\n");
    out.push_str("@.s_sub = private constant [10 x i8] c\"SUBNORMAL\\00\"\n");
    out.push_str("@.s_ovf = private constant [9 x i8] c\"OVERFLOW\\00\"\n");
    out.push('\n');
    out.push_str("define i32 @main() {\nentry:\n");

    let mut ssa = Ssa(0);
    let mut body = String::new();
    match prog.op {
        DenseCgOp::Add => emit_elementwise(prog, "fadd", &mut ssa, &mut body)?,
        DenseCgOp::Sub => emit_elementwise(prog, "fsub", &mut ssa, &mut body)?,
        DenseCgOp::Neg => emit_neg(prog, &mut ssa, &mut body),
        DenseCgOp::Scale => emit_scale(prog, &mut ssa, &mut body)?,
        DenseCgOp::Dot => emit_dot(prog, &mut ssa, &mut body)?,
        DenseCgOp::Similarity => emit_similarity(prog, &mut ssa, &mut body)?,
    }
    out.push_str(&body);
    out.push_str("  ret i32 0\n}\n");
    Ok((out, explain))
}

fn mk_explain(prog: &DenseProgram) -> DenseExplain {
    let reference_guarantee = prog.op.reference_guarantee();
    let rel_eps = match prog.op {
        DenseCgOp::Add | DenseCgOp::Sub | DenseCgOp::Scale => op_rel_eps(prog.dtype),
        DenseCgOp::Neg | DenseCgOp::Dot | DenseCgOp::Similarity => 0.0,
    };
    DenseExplain {
        op: prog.op.name(),
        dim: prog.dim,
        dtype: prog.dtype,
        rel_eps,
        // The un-quantized value ops record the `DenseArray` schedule (DN-01 — the inspectable
        // `Meta.physical`); measurements produce a bare `f64`, so no physical schedule.
        physical: prog.op.is_value_op().then_some(PhysicalLayout::DenseArray),
        reference_guarantee,
        codegen_guarantee: DENSE_CODEGEN_GUARANTEE,
        quant: "un-quantized (F32/BF16; ADR-030 QuantDesc gated on E20-1)",
    }
}

/// Emit the dumpable EXPLAIN comment into the IR (RFC-0004 §6 — the op's basis is visible in the
/// `.ll`; never a black box, G2).
fn emit_explain_comment(e: &DenseExplain, out: &mut String) {
    let _ = writeln!(
        out,
        "; dense {} | dim={} dtype={:?} | rel_eps={:e} | physical={:?} | ref-guarantee={:?} | \
         codegen-guarantee={:?} | quant={}",
        e.op,
        e.dim,
        e.dtype,
        e.rel_eps,
        e.physical,
        e.reference_guarantee,
        e.codegen_guarantee,
        e.quant,
    );
}

/// Emit one element-wise binary op (`fadd`/`fsub`): per element, compute as a native `f32` op, round
/// to the dtype grid (bf16 round-to-nearest-even in IR), check finite/normal/overflow with a
/// never-silent trap, and print the f64 bit pattern. Mirrors `DenseSpace::elementwise`.
fn emit_elementwise(
    prog: &DenseProgram,
    fop: &str,
    ssa: &mut Ssa,
    body: &mut String,
) -> Result<(), DenseAotError> {
    let b = prog
        .b
        .as_ref()
        .ok_or_else(|| DenseAotError::Malformed("binary op needs operand b".to_owned()))?;
    for (&ai, &bi) in prog.a.iter().zip(b.iter()) {
        // Operands are exact, on-grid (validated), so the f32 constants are exact.
        let x = f32_const(ai);
        let y = f32_const(bi);
        let r = ssa.fresh();
        let _ = writeln!(body, "  {r} = {fop} float {x}, {y}");
        let rounded = emit_round_to_grid(prog.dtype, &r, ssa, body);
        emit_check_and_print(&rounded, ssa, body);
    }
    emit_newline(ssa, body);
    Ok(())
}

/// Emit elementwise negation: `fneg`, exact (no rounding — symmetric grids), then print. Mirrors
/// `DenseSpace::neg_value`.
fn emit_neg(prog: &DenseProgram, ssa: &mut Ssa, body: &mut String) {
    for &ai in &prog.a {
        let x = f32_const(ai);
        let r = ssa.fresh();
        let _ = writeln!(body, "  {r} = fneg float {x}");
        // neg is exact on a symmetric grid: a negated on-grid value is on-grid and finite. Extend
        // f32→f64 and print the bit pattern (no rounding / no side-condition trap needed).
        let d = ssa.fresh();
        let _ = writeln!(body, "  {d} = fpext float {r} to double");
        emit_print_f64_bits(&d, ssa, body);
    }
    emit_newline(ssa, body);
}

/// Emit scalar `c · a`: per element `c·xᵢ` as a native `f32` mul, rounded to grid, checked, printed.
/// Mirrors `DenseSpace::scale_value`.
fn emit_scale(prog: &DenseProgram, ssa: &mut Ssa, body: &mut String) -> Result<(), DenseAotError> {
    let c = prog
        .scale
        .ok_or_else(|| DenseAotError::Malformed("scale needs a factor".to_owned()))?;
    let cc = f32_const(c);
    for &ai in &prog.a {
        let x = f32_const(ai);
        let r = ssa.fresh();
        let _ = writeln!(body, "  {r} = fmul float {cc}, {x}");
        let rounded = emit_round_to_grid(prog.dtype, &r, ssa, body);
        emit_check_and_print(&rounded, ssa, body);
    }
    emit_newline(ssa, body);
    Ok(())
}

/// Emit `dot = Σ aᵢ·bᵢ` in `f64` (a bare measurement — mirrors `DenseSpace::dot`, which sums in
/// `f64`). Each product/accumulate is in `f64` to match the reference exactly.
fn emit_dot(prog: &DenseProgram, ssa: &mut Ssa, body: &mut String) -> Result<(), DenseAotError> {
    let b = prog
        .b
        .as_ref()
        .ok_or_else(|| DenseAotError::Malformed("dot needs operand b".to_owned()))?;
    let acc = emit_dot_acc(&prog.a, b, ssa, body);
    emit_print_f64_bits(&acc, ssa, body);
    emit_newline(ssa, body);
    Ok(())
}

/// Emit cosine similarity `dot / (‖a‖·‖b‖)`, `0` on a zero-norm operand (mirrors
/// `DenseSpace::similarity` — all in `f64`).
fn emit_similarity(
    prog: &DenseProgram,
    ssa: &mut Ssa,
    body: &mut String,
) -> Result<(), DenseAotError> {
    let b = prog
        .b
        .as_ref()
        .ok_or_else(|| DenseAotError::Malformed("similarity needs operand b".to_owned()))?;
    let dot = emit_dot_acc(&prog.a, b, ssa, body);
    let na2 = emit_dot_acc(&prog.a, &prog.a, ssa, body); // Σ aᵢ²
    let nb2 = emit_dot_acc(b, b, ssa, body); // Σ bᵢ²
    let na = ssa.fresh();
    let _ = writeln!(body, "  {na} = call double @llvm.sqrt.f64(double {na2})");
    let nb = ssa.fresh();
    let _ = writeln!(body, "  {nb} = call double @llvm.sqrt.f64(double {nb2})");
    // denom = na·nb; if either norm is 0.0 → 0.0 (matches the reference's `if na == 0 || nb == 0`).
    let denom = ssa.fresh();
    let _ = writeln!(body, "  {denom} = fmul double {na}, {nb}");
    let na_z = ssa.fresh();
    let _ = writeln!(body, "  {na_z} = fcmp oeq double {na}, 0.0");
    let nb_z = ssa.fresh();
    let _ = writeln!(body, "  {nb_z} = fcmp oeq double {nb}, 0.0");
    let any_z = ssa.fresh();
    let _ = writeln!(body, "  {any_z} = or i1 {na_z}, {nb_z}");
    let q = ssa.fresh();
    let _ = writeln!(body, "  {q} = fdiv double {dot}, {denom}");
    let sim = ssa.fresh();
    let _ = writeln!(body, "  {sim} = select i1 {any_z}, double 0.0, double {q}");
    emit_print_f64_bits(&sim, ssa, body);
    emit_newline(ssa, body);
    Ok(())
}

/// Accumulate `Σ xᵢ·yᵢ` in `f64`, left-to-right, returning the accumulator register. Mirrors the
/// reference's `f64` `.sum()` (which folds left-to-right). Each step explicit IR (§6).
fn emit_dot_acc(xs: &[f64], ys: &[f64], ssa: &mut Ssa, body: &mut String) -> String {
    let mut acc = "0.0".to_owned();
    for (x, y) in xs.iter().zip(ys.iter()) {
        let xc = f64_const(*x);
        let yc = f64_const(*y);
        let p = ssa.fresh();
        let _ = writeln!(body, "  {p} = fmul double {xc}, {yc}");
        let next = ssa.fresh();
        let _ = writeln!(body, "  {next} = fadd double {acc}, {p}");
        acc = next;
    }
    acc
}

/// Round an `f32` SSA value to the dtype grid in IR. F32: identity. BF16: round-to-nearest-even on the
/// bit pattern (`(bits + 0x7FFF + lsb) >> 16 << 16`), mirroring `round_f32_to_bf16` digit-for-digit.
/// Returns the rounded `f32` SSA register.
fn emit_round_to_grid(dtype: ScalarKind, val: &str, ssa: &mut Ssa, body: &mut String) -> String {
    match dtype {
        ScalarKind::F32 => val.to_owned(),
        ScalarKind::Bf16 => {
            // bits = bitcast f32 → i32
            let bits = ssa.fresh();
            let _ = writeln!(body, "  {bits} = bitcast float {val} to i32");
            // lsb = (bits >> 16) & 1
            let sh = ssa.fresh();
            let _ = writeln!(body, "  {sh} = lshr i32 {bits}, 16");
            let lsb = ssa.fresh();
            let _ = writeln!(body, "  {lsb} = and i32 {sh}, 1");
            // rounded_bits = ((bits + 0x7FFF + lsb) >> 16) << 16
            let add1 = ssa.fresh();
            let _ = writeln!(body, "  {add1} = add i32 {bits}, 32767");
            let add2 = ssa.fresh();
            let _ = writeln!(body, "  {add2} = add i32 {add1}, {lsb}");
            let shr = ssa.fresh();
            let _ = writeln!(body, "  {shr} = lshr i32 {add2}, 16");
            let shl = ssa.fresh();
            let _ = writeln!(body, "  {shl} = shl i32 {shr}, 16");
            // back to f32
            let r = ssa.fresh();
            let _ = writeln!(body, "  {r} = bitcast i32 {shl} to float");
            r
        }
        // unreachable behind validate(); conservatively pass through.
        ScalarKind::F16 | ScalarKind::F64 => val.to_owned(),
    }
}

/// Check a rounded `f32` result element for the reference's side-conditions and print its f64 bit
/// pattern. A non-finite (overflow) or subnormal-nonzero element prints the never-silent sentinel and
/// aborts the run (the read-back surfaces it as `Overflow`/`Subnormal` — matches `DenseError`; G2).
fn emit_check_and_print(val: &str, ssa: &mut Ssa, body: &mut String) {
    // Overflow: a non-finite result. Test |x| == +Inf (an overflowed f32 op yields ±Inf), or NaN.
    let absf = ssa.fresh();
    let _ = writeln!(body, "  {absf} = call float @llvm.fabs.f32(float {val})");
    let is_inf = ssa.fresh();
    let _ = writeln!(
        body,
        "  {is_inf} = fcmp oeq float {absf}, 0x7FF0000000000000"
    );
    let is_nan = ssa.fresh();
    let _ = writeln!(body, "  {is_nan} = fcmp uno float {val}, 0.0");
    let nonfinite = ssa.fresh();
    let _ = writeln!(body, "  {nonfinite} = or i1 {is_inf}, {is_nan}");
    // Subnormal: x != 0 and |x| < DENSE_MIN_NORMAL (2^-126).
    let is_zero = ssa.fresh();
    let _ = writeln!(body, "  {is_zero} = fcmp oeq float {val}, 0.0");
    let min_normal = f32_const(DENSE_MIN_NORMAL);
    let lt_min = ssa.fresh();
    let _ = writeln!(body, "  {lt_min} = fcmp olt float {absf}, {min_normal}");
    let nz = ssa.fresh();
    let _ = writeln!(body, "  {nz} = xor i1 {is_zero}, true");
    let subnormal = ssa.fresh();
    let _ = writeln!(body, "  {subnormal} = and i1 {nz}, {lt_min}");
    // Branch: overflow → print OVERFLOW sentinel + ret; subnormal → SUBNORMAL + ret; else print bits.
    let ovf_lbl = ssa.fresh_label();
    let chk_sub_lbl = ssa.fresh_label();
    let sub_lbl = ssa.fresh_label();
    let ok_lbl = ssa.fresh_label();
    let _ = writeln!(
        body,
        "  br i1 {nonfinite}, label %{ovf_lbl}, label %{chk_sub_lbl}"
    );
    let _ = writeln!(body, "{ovf_lbl}:");
    emit_print_sentinel("@.s_ovf", 9, ssa, body);
    let _ = writeln!(body, "  ret i32 0");
    let _ = writeln!(body, "{chk_sub_lbl}:");
    let _ = writeln!(
        body,
        "  br i1 {subnormal}, label %{sub_lbl}, label %{ok_lbl}"
    );
    let _ = writeln!(body, "{sub_lbl}:");
    emit_print_sentinel("@.s_sub", 10, ssa, body);
    let _ = writeln!(body, "  ret i32 0");
    let _ = writeln!(body, "{ok_lbl}:");
    // In range: extend to f64 and print the bit pattern.
    let d = ssa.fresh();
    let _ = writeln!(body, "  {d} = fpext float {val} to double");
    emit_print_f64_bits(&d, ssa, body);
}

/// Print one `f64` value's IEEE-754 bit pattern as a decimal `u64` (so the read-back is bit-exact).
fn emit_print_f64_bits(d: &str, ssa: &mut Ssa, body: &mut String) {
    let bits = ssa.fresh();
    let _ = writeln!(body, "  {bits} = bitcast double {d} to i64");
    let p = ssa.fresh();
    let _ = writeln!(
        body,
        "  {p} = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* \
         @.fmt_u64, i64 0, i64 0), i64 {bits})"
    );
}

/// Print a sentinel string (the never-silent refusal marker).
fn emit_print_sentinel(global: &str, len: usize, ssa: &mut Ssa, body: &mut String) {
    let p = ssa.fresh();
    let _ = writeln!(
        body,
        "  {p} = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([{len} x i8], \
         [{len} x i8]* {global}, i64 0, i64 0))"
    );
}

/// Print the trailing newline that terminates the result line.
fn emit_newline(ssa: &mut Ssa, body: &mut String) {
    let p = ssa.fresh();
    let _ = writeln!(
        body,
        "  {p} = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* \
         @.fmt_nl, i64 0, i64 0))"
    );
}

/// Render an `f64` (already on the dtype grid / a scalar factor) as an exact LLVM `float` constant —
/// the hex `0x…` form, so the textual constant is bit-exact (no decimal round-trip). The value is a
/// finite on-grid f32, so the f64→f32 narrowing is exact.
fn f32_const(x: f64) -> String {
    #[allow(clippy::cast_possible_truncation)] // on-grid checked upstream: exact narrowing
    let xf = x as f32;
    // LLVM's `float` hex constant is the *double* bit pattern of the f32 value (LLVM widens an f32 hex
    // literal from the double form). So emit the f64 bits of the widened f32.
    let bits = f64::from(xf).to_bits();
    format!("0x{bits:016X}")
}

/// Render an `f64` as an exact LLVM `double` constant (hex form — bit-exact).
fn f64_const(x: f64) -> String {
    format!("0x{:016X}", x.to_bits())
}

// ─── SSA / label counters (local; the llvm.rs ones are pub(crate) but coupled to that module) ───

/// SSA register counter for the Dense module (separate from `llvm::Ssa` so the two never collide).
struct Ssa(usize);
impl Ssa {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("%r{n}")
    }
    fn fresh_label(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("bb{n}")
    }
}

// ─── compile / run (drives llc + clang; reuses the llvm.rs toolchain helpers) ───────────────────

/// A compiled native Dense artifact: the executable on disk (cleaned on drop) plus the read-back
/// shape (op, dim, dtype) needed to reconstruct the result `Value`/measurement.
pub struct DenseArtifact {
    _dir: TmpDir,
    bin: std::path::PathBuf,
    op: DenseCgOp,
    dim: u32,
    dtype: ScalarKind,
}

impl DenseArtifact {
    /// Build an artifact around an **already-compiled** executable — the MLIR-dialect sibling emitter
    /// (`dialect::native::dense`, M-856b) compiles through its own pipeline (`mlir-opt` /
    /// `mlir-translate` / `clang`, not `llc`/`clang`), then reuses this constructor so [`Self::run`]'s
    /// read-back/reconstruct logic runs **verbatim** for both compiled paths — the two can never
    /// silently diverge on how a result `Value` is stamped (DRY; VR-5). `pub(crate)`, and gated to
    /// the `mlir-dialect` feature — its only caller — so a default (feature-off) build carries no
    /// dead-code warning for a constructor that exists solely for that path.
    #[cfg(feature = "mlir-dialect")]
    pub(crate) fn from_binary(
        dir: TmpDir,
        bin: std::path::PathBuf,
        op: DenseCgOp,
        dim: u32,
        dtype: ScalarKind,
    ) -> Self {
        DenseArtifact {
            _dir: dir,
            bin,
            op,
            dim,
            dtype,
        }
    }

    /// Run the artifact and read its result back. A value op reconstructs a Dense [`Value`] carrying
    /// the reference's per-op guarantee tag; a measurement op returns a bare `f64`. A sentinel line
    /// (subnormal/overflow) is surfaced as an explicit [`DenseAotError`] — never a silent value (G2).
    pub fn run(&self) -> Result<DenseResult, DenseAotError> {
        let output = Command::new(&self.bin)
            .output()
            .map_err(|e| DenseAotError::Run(format!("exec {}: {e}", self.bin.display())))?;
        if !output.status.success() {
            return Err(DenseAotError::Run(format!(
                "artifact exited {}",
                output.status
            )));
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|e| DenseAotError::Parse(format!("non-utf8 output: {e}")))?;
        let line = stdout.lines().next().unwrap_or("").trim();
        // Never-silent sentinels (matches DenseError::Overflow / SubnormalUnsupported). A sentinel
        // can appear **anywhere** on the line, not only at its start: a result element that overflows
        // / goes subnormal at index `i > 0` is printed *after* the earlier in-range elements (the
        // artifact emits each element space-separated, then the sentinel for the first failing one),
        // so the line reads e.g. `"<bits0> OVERFLOW"`. Scan every whitespace-split token for a
        // sentinel and surface the matching `DenseAotError` — never misclassify an overflow/subnormal
        // as a `Parse` failure of the sentinel token (G2: the refusal stays the right variant).
        let mut tokens = line.split_whitespace();
        let mut bits: Vec<u64> = Vec::new();
        for tok in tokens.by_ref() {
            if tok == DENSE_OVERFLOW_SENTINEL {
                return Err(DenseAotError::Overflow);
            }
            if tok == DENSE_SUBNORMAL_SENTINEL {
                return Err(DenseAotError::Subnormal);
            }
            bits.push(
                tok.parse::<u64>()
                    .map_err(|e| DenseAotError::Parse(format!("non-u64 token {tok:?}: {e}")))?,
            );
        }
        if self.op.is_value_op() {
            self.reconstruct_value(&bits)
        } else {
            // A measurement is one f64.
            if bits.len() != 1 {
                return Err(DenseAotError::Parse(format!(
                    "measurement expected 1 element, got {}",
                    bits.len()
                )));
            }
            Ok(DenseResult::Measurement(f64::from_bits(bits[0])))
        }
    }

    /// Reconstruct the Dense `Value` from the printed f64 bit patterns, carrying the reference's per-op
    /// guarantee tag (so the observable matches `DenseSpace`).
    fn reconstruct_value(&self, bits: &[u64]) -> Result<DenseResult, DenseAotError> {
        if bits.len() != self.dim as usize {
            return Err(DenseAotError::Parse(format!(
                "expected {} elements, got {}",
                self.dim,
                bits.len()
            )));
        }
        let xs: Vec<f64> = bits.iter().map(|&b| f64::from_bits(b)).collect();
        let repr = Repr::Dense {
            dim: self.dim,
            dtype: self.dtype,
        };
        let meta = self.result_meta()?;
        Value::new(repr, Payload::Scalars(xs), meta)
            .map(|v| DenseResult::Value(Box::new(v)))
            .map_err(|e| DenseAotError::Wf(e.to_string()))
    }

    /// Build the result `Meta` mirroring the reference's per-op guarantee: `Exact` (no bound) for
    /// `neg`, `Proven` (the Higham `ProvenThm` rounding bound) for `add`/`sub`/`scale`. The guarantee +
    /// bound — which the differential's `(repr, payload, guarantee)` observable checks — match the
    /// reference exactly. `Meta.physical = DenseArray` records the inspectable schedule (RFC-0039 §5.1;
    /// DN-01) without affecting the observable.
    fn result_meta(&self) -> Result<Meta, DenseAotError> {
        let map_wf = |e: WfError| DenseAotError::Wf(e.to_string());
        match self.op {
            DenseCgOp::Neg => Meta::new(
                Provenance::Root,
                GuaranteeStrength::Exact,
                None,
                None,
                Some(PhysicalLayout::DenseArray),
                None,
            )
            .map_err(map_wf),
            DenseCgOp::Add | DenseCgOp::Sub | DenseCgOp::Scale => {
                let bound = Bound {
                    kind: BoundKind::Error {
                        eps: op_rel_eps(self.dtype),
                        norm: NormKind::Rel,
                    },
                    basis: BoundBasis::ProvenThm {
                        citation: op_citation(self.dtype).to_owned(),
                    },
                };
                Meta::new(
                    Provenance::Derived {
                        op: operation_hash(self.op.name()),
                        inputs: vec![],
                    },
                    GuaranteeStrength::Proven,
                    Some(bound),
                    None,
                    Some(PhysicalLayout::DenseArray),
                    None,
                )
                .map_err(map_wf)
            }
            // measurements never reconstruct a Value.
            DenseCgOp::Dot | DenseCgOp::Similarity => Err(DenseAotError::Malformed(
                "measurement op has no result Meta".to_owned(),
            )),
        }
    }
}

/// Compile a Dense program to a native executable (emit IR → `llc` → `clang`) without running it.
/// Returns [`DenseAotError::ToolchainMissing`] when `llc`/`clang` are absent (callers skip); any
/// out-of-fragment construct is the same explicit refusal as [`emit_dense_llvm_ir`].
pub fn dense_compile(prog: &DenseProgram) -> Result<DenseArtifact, DenseAotError> {
    let (mut ir, _explain) = emit_dense_llvm_ir(prog)?;
    // Declare the intrinsics each op pulls in (only where needed — keep the module minimal so a reader
    // sees exactly what each op requires).
    let mut decls = String::new();
    if matches!(prog.op, DenseCgOp::Add | DenseCgOp::Sub | DenseCgOp::Scale) {
        decls.push_str("declare float @llvm.fabs.f32(float)\n");
    }
    if matches!(prog.op, DenseCgOp::Similarity) {
        decls.push_str("declare double @llvm.sqrt.f64(double)\n");
    }
    if !decls.is_empty() {
        // Insert the declares right before `define i32 @main()`.
        ir = ir.replacen(
            "define i32 @main()",
            &format!("{decls}define i32 @main()"),
            1,
        );
    }
    ensure_toolchain()?;
    let dir = unique_tmp_dir().map_err(aot_to_dense)?;
    let ll = dir.join("dense.ll");
    let obj = dir.join("dense.o");
    let bin = dir.join("dense");
    let guard = TmpDir(dir);
    std::fs::write(&ll, ir.as_bytes()).map_err(|e| DenseAotError::Run(format!("write IR: {e}")))?;
    run_tool(
        "llc",
        &[
            "-relocation-model=pic",
            "-filetype=obj",
            path(&ll).map_err(aot_to_dense)?,
            "-o",
            path(&obj).map_err(aot_to_dense)?,
        ],
    )
    .map_err(aot_to_dense)?;
    run_tool(
        "clang",
        &[
            path(&obj).map_err(aot_to_dense)?,
            "-o",
            path(&bin).map_err(aot_to_dense)?,
        ],
    )
    .map_err(aot_to_dense)?;
    Ok(DenseArtifact {
        _dir: guard,
        bin,
        op: prog.op,
        dim: prog.dim,
        dtype: prog.dtype,
    })
}

/// Compile + run a Dense program: the compiled execution path the M-853 differential checks against
/// the `mycelium-dense` reference.
pub fn dense_compile_and_run(prog: &DenseProgram) -> Result<DenseResult, DenseAotError> {
    dense_compile(prog)?.run()
}

/// Map a `llvm::AotError` (from the reused toolchain helpers) into a `DenseAotError`, preserving the
/// never-silent classification (toolchain-missing stays a skip; a real compile/run failure stays an
/// error).
fn aot_to_dense(e: crate::llvm::AotError) -> DenseAotError {
    use crate::llvm::AotError;
    match e {
        AotError::ToolchainMissing(t) => DenseAotError::ToolchainMissing(t),
        AotError::Compile(s) => DenseAotError::Compile(s),
        AotError::Run(s) => DenseAotError::Run(s),
        AotError::Parse(s) => DenseAotError::Parse(s),
        other => DenseAotError::Run(other.to_string()),
    }
}

fn ensure_toolchain() -> Result<(), DenseAotError> {
    for tool in ["llc", "clang"] {
        Command::new(tool)
            .arg("--version")
            .output()
            .map_err(|_| DenseAotError::ToolchainMissing(tool.to_owned()))?;
    }
    Ok(())
}
