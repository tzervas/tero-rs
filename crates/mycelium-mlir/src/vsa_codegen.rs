//! Native direct-LLVM codegen of **`Repr::Vsa` hypervector ops** — the real-`Vec<f64>` fragment of
//! the 1.0.0-mandatory **MAP-I / BSC / HRR / FHRR** models (M-854; epic E25-1; **RFC-0039 §5.2**
//! *Native VSA lowering*, Accepted 2026-06-30; RFC-0003 §4.1 the per-op guarantee matrix; ADR-031 the
//! element space / sparsity / complex carrier; ADR-034 re-gating native AOT into lang 1.0.0;
//! RFC-0004 §6 *no-opaque-lowering* / §11 the additive direct-LLVM increment pattern; DN-01).
//!
//! ## What this lowers
//! The `mycelium-vsa::VsaModel` hypervector surface over the **real-`Vec<f64>` fragment** of the four
//! **1.0.0-native-mandatory standard models** (OQ-3 resolved 2026-06-30 — MAP-I, BSC, HRR, FHRR; the
//! niche **SBC, MAP-B** extend post-mandate and are refused never-silently here):
//!
//! - **`vsa.bind` / `vsa.unbind`** — the model's binding algebra:
//!   - **MAP-I**: elementwise product `aᵢ·bᵢ`; self-inverse on the `±1` alphabet — `bind`/`unbind`
//!     both **`Exact`** (`MapI::bind`).
//!   - **BSC**: elementwise XOR on `{0,1}` (computed as `|aᵢ − bᵢ|`); self-inverse — **`Exact`**
//!     (`Bsc::bind`).
//!   - **HRR**: circular convolution `(a⊛b)[k] = Σᵢ aᵢ·b[(k+d−i) mod d]` (`bind` **`Exact`**); circular
//!     correlation `unbind = a ⊛ involution(b)`, the approximate inverse — **`Empirical`** (the residual
//!     weak link, RFC-0003 §4.1 / T1.2; `Hrr::unbind`).
//!   - **FHRR**: elementwise phase add `wrap((aᵢ+bᵢ))` (`bind` **`Exact`**); phase sub `wrap((aᵢ−bᵢ))`
//!     (`unbind` **`Empirical`** — the weak-link assignment; `Fhrr::unbind`).
//! - **`vsa.bundle`** — superposition:
//!   - **MAP-I**: elementwise integer sum `Σ items`; **`Proven`** capacity bound **only** when the
//!     reference's checked side-condition `dim ≥ requiredDim(m, δ)` holds
//!     (`capacity::proven_capacity_bound`), else an explicit `InsufficientCapacity` refusal (never an
//!     unbacked `Proven`; VR-5/M-I2). Multi-hop capacity (M-832) is **never** `Proven` (it is unfinished
//!     research — RFC-0039 §5.2 honesty note).
//!   - **BSC**: elementwise majority (tie copies the first operand's bit); value-level **`Empirical`**,
//!     profile-gated (`BSC_BUNDLE_PROFILE` — odd `m ≤ 5`, `dim ≥ 1024`; `Bsc::bundle_values_empirical`).
//!   - **HRR**: elementwise sum; **`Empirical`** (Gaussian capacity only).
//!   - **FHRR**: per-component complex-sum-renormalized phasor `arg Σ e^{iθ}`; **`Empirical`**, with a
//!     never-silent `DegenerateBundleComponent` refusal when a phasor sum vanishes (`Fhrr::bundle`).
//! - **`vsa.permute`** — cyclic left rotation by `shift` (`rem_euclid`); **`Exact`** for every model.
//! - **`vsa.similarity`** — a bare-`f64` *measurement* (no `Meta` tag), per model: cosine
//!   (MAP-I/HRR), centered Hamming `1 − 2·d_H/d` (BSC), mean `cos(θa − θb)` (FHRR).
//!
//! Every step is **explicit per-element textual IR** computed in `f64` (`double`), mirroring the
//! reference's `f64` arithmetic **digit-for-digit and in the same operation order** — so the native
//! read-back hypervector is **bit-identical** to `mycelium-vsa`'s, which is exactly what the M-210
//! observational-equivalence checker requires (it compares `Payload::Hypervector` bit-exactly). No
//! opaque pass (RFC-0004 §6). A leading IR comment records the op, model, dim, the inspectable
//! `Meta.physical` schedule (`VsaStore`), and the guarantee (no black box; ADR-006/G2).
//!
//! ## Faithfulness to the reference (the load-bearing decision)
//! The native lowering is the **performance layer**, never the source of meaning — `mycelium-vsa`
//! (and the interpreter above it) is the trusted base (NFR-7). The native op's payload is
//! **observably equal** to the reference's (`repr + payload + guarantee`); the alphabet / regime
//! side-conditions the reference checks (`±1` for MAP-I, `{0,1}` for BSC, in-range phases for FHRR,
//! the single-factor empirical regime + minimum dim for HRR/FHRR `unbind`, the capacity side-condition
//! for MAP-I `bundle`) are **re-checked host-side at lowering**, refusing **never-silently** through a
//! dedicated `VsaAotError`. The native path **does not** ship a second, divergent VSA semantics (DRY).
//!
//! ## Guarantee tag (VR-5 — never upgraded past the basis)
//! The read-back [`Value`] carries the **reference's** RFC-0003 §4.1 per-op tag (so the differential's
//! observable matches) — but the **codegen's own confidence that native ≡ reference is `Empirical`**,
//! established by the M-210 differential plus the `cargo-mutants` witness, **not** by a proof object
//! linked into this codegen ([`VSA_CODEGEN_GUARANTEE`]). The MAP-I `bundle` `Proven` capacity *value*
//! tag is carried **only** by replaying the reference's checked instantiation
//! (`capacity::proven_capacity_bound`) — the same side-condition the reference checks — never on the
//! strength of the in-progress multi-hop research (M-832, which never stamps `Proven`; RFC-0039 §5.2
//! honesty note). So: the *value* tag is the reference's RFC-0003 §4.1 tag, and the *codegen* claim is
//! `Empirical`.
//!
//! ## Never-silent refusals (G2)
//! - **SBC / MAP-B** model → [`VsaAotError::UnsupportedModel`] — out of the 1.0.0-native-mandatory set
//!   (OQ-3; extend post-mandate), refused, never silently served by a different model.
//! - **Block-sparse / complex (ADR-031) carrier** → [`VsaAotError::UnsupportedCarrier`] — the
//!   `VsaElem`/`VsaSparsity`/`HypervectorC` fields are not yet in the value model (E20-1; RFC-0039 §5.2
//!   honesty note), so only the real-`Vec<f64>` dense fragment is lowered; a sparse repr is refused.
//! - **Off-alphabet / out-of-range / out-of-regime input** → refused at lowering (matches the
//!   reference's `NonAlphabetComponent` / `OutsideEmpiricalProfile`).
//! - **MAP-I `bundle` below `requiredDim`** → [`VsaAotError::InsufficientCapacity`] (matches the
//!   reference; no unbacked `Proven`).
//! - **FHRR degenerate bundle component** → [`VsaAotError::DegenerateBundleComponent`], surfaced
//!   never-silently at runtime via the sentinel read-back (matches `Fhrr::bundle`).
//! - **dim / operand-count mismatch** → refused (matches `VsaError::DimMismatch`/`EmptyBundle`).
//!
//! ## Direct-LLVM first; dialect later (RFC-0039 §5.2 / RFC-0004 §11)
//! This is the direct-LLVM increment. The MLIR-dialect path honestly **refuses** VSA
//! (`DialectError::Unsupported` naming "Dense/VSA stay on the interpreter / direct-LLVM path",
//! `dialect/native.rs`), so the three-way differential's dialect leg is a never-faked refusal (the
//! differential reduces to two-way for VSA, exactly as for Dense).
//!
//! **Submodule confinement:** zero `unsafe` (compiler-enforced by the crate's `#![forbid]`).

use std::fmt;
use std::fmt::Write as _; // `writeln!` into a String never fails — call sites discard the Result.
use std::process::Command;

use mycelium_core::{
    operation_hash, Bound, GuaranteeStrength, Meta, Payload, PhysicalLayout, Provenance, Repr,
    SparsityClass, Value, WfError,
};
use mycelium_vsa::bsc::BSC_BUNDLE_PROFILE;
use mycelium_vsa::capacity::proven_capacity_bound;
use mycelium_vsa::fhrr::FHRR_UNBIND_PROFILE;
use mycelium_vsa::hrr::HRR_UNBIND_PROFILE;
use mycelium_vsa::EmpiricalProfile;

use crate::llvm::{path, run_tool, unique_tmp_dir, TmpDir};

// ─── earned Empirical capacity profiles for HRR/FHRR bundle (M-854; FLAG-0 resolution 2026-06-30) ─

/// The trial-validated regime backing the native HRR-`bundle` `Empirical` δ (M-854; RFC-0039 §5.2,
/// per the maintainer's FLAG-0 resolution 2026-06-30 — HRR/FHRR bundle moves from `Declared` to a real
/// `Empirical` tag earned by measured trials). `mycelium-vsa` exposes no value-level HRR-bundle profile,
/// so this profile is **derived here** (in the native codegen, over the *same* `mycelium-vsa` reference
/// algebra) and validated by [`crate::tests::vsa_codegen`]'s Monte-Carlo trial, exactly mirroring the
/// `BSC_BUNDLE_PROFILE` / `*_UNBIND_PROFILE` derivation: the membership-decode failure rate at the worst
/// covered point (`max_items` members, `min_dim`) stays ≤ `delta` over `trials` independent trials.
///
/// **Measured envelope (CPU, this environment, 2026-06-30):** at `dim = 256`, codebook 16, the
/// membership-decode failure rate is `0` for `m ≤ 4` and `1e-4` (1/10 000) at `m = 5` — comfortably
/// ≤ `δ = 1e-2`. The `δ = 1e-2` declared here is the family-consistent guaranteed tail (a 10 000-trial
/// basis cannot distinguish `1e-4` from `0`), **earned** by the trial, never fabricated. Outside this
/// envelope (`dim < 256` or `m > 5`) native HRR `bundle` is an explicit `OutsideEmpiricalProfile`
/// refusal — the bound is **never claimed beyond what was measured** (VR-5). The large-dim / many-vector
/// extension is GPU-deferred (the heavy profiling test; see `tests/vsa_differential.rs`).
pub const HRR_BUNDLE_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 5,
    // Sum-superposition has no majority-tie asymmetry (unlike BSC), so even item counts are covered.
    odd_items_only: false,
    min_dim: 256,
    delta: 1e-2,
    trials: 10_000,
    method: "Monte-Carlo HRR sum-bundle membership decode (N(0,1/d) atoms, m ≤ 5, d ≥ 256, \
             codebook 16, depth 1; measured worst-point rate 1e-4 ≤ δ at d=256/m=5)",
};

/// The trial-validated regime backing the native FHRR-`bundle` `Empirical` δ (M-854; the FLAG-0
/// resolution 2026-06-30). Derived + validated here exactly as [`HRR_BUNDLE_PROFILE`], over the
/// `mycelium-vsa` FHRR phasor-bundle algebra. **Measured envelope (CPU, 2026-06-30):** at `dim = 256`,
/// codebook 16, the membership-decode failure rate is `0` for every `m ≤ 5` — well ≤ `δ = 1e-2`. The
/// `δ = 1e-2` is the family-consistent earned tail; outside the envelope native FHRR `bundle` is an
/// explicit `OutsideEmpiricalProfile` refusal, and the FHRR degenerate-phasor-component refusal
/// (`DegenerateBundleComponent`) is unchanged (a vanished phasor sum is still refused never-silently).
pub const FHRR_BUNDLE_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 5,
    odd_items_only: false,
    min_dim: 256,
    delta: 1e-2,
    trials: 10_000,
    method:
        "Monte-Carlo FHRR phasor-bundle membership decode (uniform phasor atoms, m ≤ 5, d ≥ 256, \
             codebook 16, depth 1; measured worst-point rate 0 ≤ δ at d=256/m=5)",
};

// ─── the VSA op surface this module lowers (real-Vec<f64> fragment; RFC-0039 §5.2) ──────────────

/// The VSA ops native codegen lowers — the `mycelium-vsa::VsaModel` surface (RFC-0039 §5.2).
/// `Similarity` is a bare-`f64` *measurement* (no `Meta` tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsaCgOp {
    /// Bind (associate) two hypervectors.
    Bind,
    /// Unbind (recover a factor) — the (approximate or exact) inverse of [`VsaCgOp::Bind`].
    Unbind,
    /// Bundle (superpose) a non-empty set of hypervectors.
    Bundle,
    /// Permute (cyclic left shift) by `shift`.
    Permute,
    /// Cosine / Hamming / phase similarity in `[-1, 1]` — a bare measurement (no `Meta`).
    Similarity,
}

impl VsaCgOp {
    /// Whether the op produces a VSA `Value` (vs a bare-`f64` measurement). `similarity` is a
    /// measurement.
    #[must_use]
    pub fn is_value_op(self) -> bool {
        !matches!(self, VsaCgOp::Similarity)
    }
}

/// The 1.0.0-native-mandatory VSA models (OQ-3 resolved 2026-06-30). SBC / MAP-B are **not** in this
/// set — they are an explicit [`VsaAotError::UnsupportedModel`] (extend post-mandate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsaModelId {
    /// Multiply-Add-Permute (integer/bipolar). Bind/unbind/permute Exact; bundle Proven (capacity).
    MapI,
    /// Binary Spatter Code. Bind/unbind (XOR)/permute Exact; bundle Empirical (majority).
    Bsc,
    /// Holographic Reduced Representations (circular convolution). Bind/permute Exact; unbind/bundle
    /// Empirical.
    Hrr,
    /// Fourier HRR (phasor). Bind/permute Exact; unbind/bundle Empirical.
    Fhrr,
}

impl VsaModelId {
    /// The `mycelium-vsa` registry model id (matches `Repr::Vsa { model }` + the reference's
    /// `operation_hash` keys), recorded in the IR comment + provenance so the lowered op is never
    /// anonymous (G2).
    #[must_use]
    pub fn registry_id(self) -> &'static str {
        match self {
            VsaModelId::MapI => "MAP-I",
            VsaModelId::Bsc => "BSC",
            VsaModelId::Hrr => "HRR",
            VsaModelId::Fhrr => "FHRR",
        }
    }

    /// Parse a registry model id into a native-mandatory model, or `None` for a non-mandatory model
    /// (SBC / MAP-B / unknown) — the caller turns `None` into an explicit
    /// [`VsaAotError::UnsupportedModel`] (never a silent substitution; G2).
    #[must_use]
    pub fn from_registry_id(id: &str) -> Option<Self> {
        match id {
            "MAP-I" => Some(VsaModelId::MapI),
            "BSC" => Some(VsaModelId::Bsc),
            "HRR" => Some(VsaModelId::Hrr),
            "FHRR" => Some(VsaModelId::Fhrr),
            _ => None,
        }
    }

    /// The `mycelium-vsa` operation-name prefix for this model (e.g. `vsa.map_i`), used to build the
    /// provenance / EXPLAIN op key matching the reference's `operation_hash` keys.
    #[must_use]
    pub fn op_prefix(self) -> &'static str {
        match self {
            VsaModelId::MapI => "vsa.map_i",
            VsaModelId::Bsc => "vsa.bsc",
            VsaModelId::Hrr => "vsa.hrr",
            VsaModelId::Fhrr => "vsa.fhrr",
        }
    }

    /// The full op name for `(model, op)` matching the reference's keys, e.g. `vsa.map_i.bind`.
    /// `similarity` has no reference op key (it is a bare measurement), so it returns `None`.
    #[must_use]
    pub fn op_name(self, op: VsaCgOp) -> Option<String> {
        let suffix = match op {
            VsaCgOp::Bind => "bind",
            VsaCgOp::Unbind => "unbind",
            VsaCgOp::Bundle => "bundle",
            VsaCgOp::Permute => "permute",
            VsaCgOp::Similarity => return None,
        };
        Some(format!("{}.{suffix}", self.op_prefix()))
    }

    /// The honest **value-level** guarantee the native read-back carries for `(model, op)` — derived
    /// from the reference's value-level surface (RFC-0003 §4.1), never upgraded past it (VR-5):
    /// - `permute` and `bind` are algebraically exact — **`Exact`** for every mandatory model;
    /// - `unbind` is **`Exact`** (self-inverse) for MAP-I/BSC, **`Empirical`** (the weak link, the
    ///   reference's `*_unbind` profile) for HRR/FHRR;
    /// - `bundle` is **`Proven`** for MAP-I (the checked capacity bound) and **`Empirical`** for BSC,
    ///   HRR, and FHRR — each carrying a trial-validated capacity profile: BSC the reference's
    ///   `BSC_BUNDLE_PROFILE`, and HRR/FHRR the **codegen-derived** [`HRR_BUNDLE_PROFILE`] /
    ///   [`FHRR_BUNDLE_PROFILE`] (M-854 FLAG-0 resolution, 2026-06-30 — moved from `Declared` to a real
    ///   `Empirical` earned by measured trials, validated over the `mycelium-vsa` reference algebra; no
    ///   fabricated bound — VR-5). The `Empirical` tag holds **only within** the measured envelope
    ///   (`max_items` / `min_dim`); outside it native `bundle` is an explicit `OutsideEmpiricalProfile`
    ///   refusal — the bound is never claimed past what was measured.
    ///
    /// `None` for a measurement (`similarity` — no `Meta`).
    #[must_use]
    pub fn reference_guarantee(self, op: VsaCgOp) -> Option<GuaranteeStrength> {
        use GuaranteeStrength::{Empirical, Exact, Proven};
        let g = match (self, op) {
            // permute is a coordinate bijection — Exact for every model.
            (_, VsaCgOp::Permute) => Exact,
            // bind is algebraic/deterministic — Exact for every mandatory model.
            (_, VsaCgOp::Bind) => Exact,
            // unbind: self-inverse exact for MAP-I/BSC; the weak link (Empirical) for HRR/FHRR.
            (VsaModelId::MapI | VsaModelId::Bsc, VsaCgOp::Unbind) => Exact,
            (VsaModelId::Hrr | VsaModelId::Fhrr, VsaCgOp::Unbind) => Empirical,
            // bundle: MAP-I value-level Proven (checked capacity); BSC/HRR/FHRR Empirical (trial profile).
            (VsaModelId::MapI, VsaCgOp::Bundle) => Proven,
            (VsaModelId::Bsc | VsaModelId::Hrr | VsaModelId::Fhrr, VsaCgOp::Bundle) => Empirical,
            // similarity is a measurement — no Meta tag.
            (_, VsaCgOp::Similarity) => return None,
        };
        Some(g)
    }
}

/// Resolve a `Repr::Vsa { model, sparsity }` to the native [`VsaModelId`] for lowering, or an
/// **explicit, never-silent** refusal — the boundary entry point that turns a registry id + sparsity
/// class into a lowerable model (G2). Refuses:
/// - a **non-mandatory model** (SBC / MAP-B / unknown) → [`VsaAotError::UnsupportedModel`] (OQ-3;
///   extend post-mandate), never silently served by a different model;
/// - a **sparse carrier** (ADR-031 block-sparse, not yet in the value model) →
///   [`VsaAotError::UnsupportedCarrier`] (E20-1 gate), never silently flattened to dense.
///
/// This is how a caller (e.g. an AOT lowering pass dispatching a `Repr::Vsa` const) reaches native VSA
/// codegen: it resolves the model here and gets a typed refusal for anything out of the real-`Vec<f64>`
/// MAP-I/BSC/HRR/FHRR fragment, routing those to the interpreter/reference (NFR-7).
pub fn resolve_model(model: &str, sparsity: SparsityClass) -> Result<VsaModelId, VsaAotError> {
    match sparsity {
        SparsityClass::Dense => {}
        SparsityClass::Sparse { max_active } => {
            return Err(VsaAotError::UnsupportedCarrier(format!(
                "sparse≤{max_active} (ADR-031 block-sparse carrier, E20-1)"
            )));
        }
    }
    VsaModelId::from_registry_id(model)
        .ok_or_else(|| VsaAotError::UnsupportedModel(model.to_owned()))
}

/// A native-VSA lowering program: one op over a model's hypervector carrier, plus its operand(s).
/// The operands are the reference hypervector payloads the caller has built through `mycelium-vsa`
/// (so they are alphabet-valid + dim-consistent). Single-source-of-truth for [`emit_vsa_llvm_ir`],
/// [`vsa_compile`], and the read-back shape (so they can never disagree).
#[derive(Debug, Clone)]
pub struct VsaProgram {
    /// The op to lower.
    pub op: VsaCgOp,
    /// The model whose algebra is lowered.
    pub model: VsaModelId,
    /// Hypervector dimensionality.
    pub dim: u32,
    /// The operands (each length `dim`). `bind`/`unbind`/`similarity` use the first two; `permute`
    /// uses the first; `bundle` uses all (≥ 1).
    pub items: Vec<Vec<f64>>,
    /// The shift for `permute` (`None` for non-`permute` ops).
    pub shift: Option<i64>,
    /// The target failure probability δ for a MAP-I `bundle`'s `Proven` capacity bound (`None` for
    /// non-MAP-I-bundle ops; an explicit `Malformed` if a MAP-I bundle omits it).
    pub bundle_delta: Option<f64>,
}

/// What a VSA native op produces: a VSA `Value` (`bind`/`unbind`/`bundle`/`permute`) or a bare-`f64`
/// measurement (`similarity`). Never-silent: the variant is the op's honest output shape. The `Value`
/// is **boxed** — a `Value` is large while a measurement is 8 bytes, so an unboxed enum would bloat
/// every `Measurement` to the `Value` size (clippy `large_enum_variant`); boxing keeps the
/// common-case measurement small without changing the never-silent shape.
#[derive(Debug, Clone, PartialEq)]
pub enum VsaResult {
    /// A VSA `Value` (boxed) carrying the reference's per-op guarantee tag.
    Value(Box<Value>),
    /// A bare-`f64` measurement (no `Meta` — mirrors `VsaModel::similarity`).
    Measurement(f64),
}

// ─── explicit, never-silent failure of the native VSA path (G2) ─────────────────────────────────

/// Why the native VSA path could not lower/run a program — **always explicit, never silent** (G2).
/// Mirrors the reference [`mycelium_vsa::VsaError`] refusals where they overlap, and adds the
/// native-path-specific toolchain / model-gate / carrier-gate refusals.
#[derive(Debug, Clone, PartialEq)]
pub enum VsaAotError {
    /// The model is outside the 1.0.0-native-mandatory set {MAP-I, BSC, HRR, FHRR} (OQ-3). SBC / MAP-B
    /// (and any unknown model) extend post-mandate — refused, never silently served by a different
    /// model (G2).
    UnsupportedModel(String),
    /// A block-sparse / complex (ADR-031) carrier — not yet in the value model (E20-1; RFC-0039 §5.2
    /// honesty note). Only the real-`Vec<f64>` dense fragment is lowered; a sparse repr is refused,
    /// never silently flattened.
    UnsupportedCarrier(String),
    /// An operand's dimensionality disagrees with the program's `dim` (matches
    /// `VsaError::DimMismatch`).
    DimMismatch {
        /// Expected dimension.
        expected: u32,
        /// Got.
        got: usize,
    },
    /// A bundle was requested over zero items (matches `VsaError::EmptyBundle`).
    EmptyBundle,
    /// A component is outside the model's alphabet (`±1` for MAP-I, `{0,1}` for BSC, an in-range phase
    /// for FHRR) — the algebra is undefined there (matches `VsaError::NonAlphabetComponent`). The
    /// index names the offending component.
    NonAlphabetComponent {
        /// The model whose alphabet was violated.
        model: &'static str,
        /// Index of the offending component.
        index: usize,
    },
    /// An `Empirical` op was requested outside its trial-validated profile's side-conditions (matches
    /// `VsaError::OutsideEmpiricalProfile`) — issuing the tag there would outrun the evidence (VR-5).
    OutsideEmpiricalProfile(String),
    /// A MAP-I `Proven` bundle was requested but `dim < requiredDim(items, δ)` — the cited capacity
    /// theorem's side-condition fails (matches `VsaError::InsufficientCapacity`; M-I2/VR-5). No
    /// unbacked `Proven` is stamped.
    InsufficientCapacity {
        /// Number of items bundled.
        items: u64,
        /// The dimension supplied.
        dim: u64,
        /// The dimension the theorem requires.
        required: u64,
    },
    /// An FHRR bundle component's phasor sum vanished — its phase is undefined (matches
    /// `VsaError::DegenerateBundleComponent`). Detected at runtime, surfaced via the sentinel
    /// read-back — never an arbitrary phase pick (G2).
    DegenerateBundleComponent,
    /// The program is malformed for its op (e.g. a binary op with < 2 operands, a `permute` with no
    /// shift, a MAP-I bundle with no δ) — an internal contract violation, surfaced explicitly rather
    /// than panicking.
    Malformed(String),
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

impl fmt::Display for VsaAotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VsaAotError::UnsupportedModel(m) => write!(
                f,
                "VSA model {m:?} is outside the 1.0.0-native-mandatory set {{MAP-I, BSC, HRR, FHRR}} \
                 (OQ-3; SBC/MAP-B extend post-mandate) — refused, never served by another model (G2)"
            ),
            VsaAotError::UnsupportedCarrier(s) => write!(
                f,
                "VSA carrier refused: {s} — the ADR-031 element-space/sparsity/complex carrier is not \
                 yet in the value model (E20-1; RFC-0039 §5.2); only the real-Vec<f64> dense fragment \
                 is lowered, never silently flattened (G2)"
            ),
            VsaAotError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            VsaAotError::EmptyBundle => write!(f, "bundle requires at least one item"),
            VsaAotError::NonAlphabetComponent { model, index } => write!(
                f,
                "component {index} is outside the {model} alphabet — the algebra is undefined there; \
                 refused, never coerced (G2)"
            ),
            VsaAotError::OutsideEmpiricalProfile(detail) => {
                write!(f, "outside the trial-validated empirical profile: {detail}")
            }
            VsaAotError::InsufficientCapacity {
                items,
                dim,
                required,
            } => write!(
                f,
                "insufficient capacity for a Proven bound: bundling {items} items needs dim ≥ \
                 {required}, got {dim} (no unbacked Proven — VR-5)"
            ),
            VsaAotError::DegenerateBundleComponent => write!(
                f,
                "FHRR bundle component has a vanished phasor sum — its phase is undefined; refused, \
                 never an arbitrary pick (matches VsaError::DegenerateBundleComponent; G2)"
            ),
            VsaAotError::Malformed(s) => write!(f, "malformed VSA program: {s}"),
            VsaAotError::ToolchainMissing(t) => write!(f, "native toolchain missing: {t}"),
            VsaAotError::Compile(e) => write!(f, "native compile failed: {e}"),
            VsaAotError::Run(e) => write!(f, "native run failed: {e}"),
            VsaAotError::Parse(e) => write!(f, "native output parse failed: {e}"),
            VsaAotError::Wf(e) => write!(f, "result well-formedness violation: {e}"),
        }
    }
}

impl std::error::Error for VsaAotError {}

// ─── the inspectable EXPLAIN record (RFC-0004 §6; ADR-006 — no black box) ───────────────────────

/// The inspectable record of how a VSA op was lowered — the EXPLAIN payload (RFC-0004 §6; no black
/// box). Carries the op, model, dim, the inspectable `Meta.physical` schedule (`VsaStore` for the
/// dense fragment — the schedule-as-metadata discipline, DN-01/ADR-031), the never-upgraded codegen
/// guarantee, the reference per-op tag the read-back value carries, and the carrier status (always
/// `real-Vec<f64> dense` today; the E20-1 element-space widening is recorded here when it lands).
#[derive(Debug, Clone, PartialEq)]
pub struct VsaExplain {
    /// The op name (`vsa.map_i.bind`, …; `vsa.<model>.similarity` for the measurement).
    pub op: String,
    /// The registry model id (`MAP-I`, …).
    pub model: &'static str,
    /// Dimensionality.
    pub dim: u32,
    /// The inspectable physical schedule (`Some(VsaStore{sparse:false})` for a value op; `None` for a
    /// measurement whose output is a bare `f64`). The ADR-031 element-space / sparsity layout extends
    /// this record as E20-1 lands it (RFC-0039 §5.2) — never a hidden choice.
    pub physical: Option<PhysicalLayout>,
    /// The reference guarantee the read-back value carries (`None` for a measurement).
    pub reference_guarantee: Option<GuaranteeStrength>,
    /// The **codegen-correctness** guarantee (`Empirical` — differential + mutant-witness, never a
    /// proof object linked here; VR-5).
    pub codegen_guarantee: GuaranteeStrength,
    /// The carrier status — `real-Vec<f64> dense` today; the E20-1 widening records the ADR-031
    /// element space / sparsity here.
    pub carrier: &'static str,
}

/// The codegen-correctness guarantee for the native VSA path: **`Empirical`** (the basis is the M-210
/// differential together with the `cargo-mutants` witness; no proof object is linked into this
/// codegen — VR-5). Exposed so callers / EXPLAIN consumers read the honest codegen-confidence tag,
/// distinct from the reference *value* tag the read-back carries.
pub const VSA_CODEGEN_GUARANTEE: GuaranteeStrength = GuaranteeStrength::Empirical;

/// The carrier status string the EXPLAIN records — the real-`Vec<f64>` dense fragment (RFC-0039 §5.2;
/// the ADR-031 element-space/sparsity/complex carrier widens it as E20-1 lands those `Repr` fields).
const CARRIER_STATUS: &str =
    "real-Vec<f64> dense (ADR-031 element-space/sparsity/complex gated on E20-1)";

// ─── alphabet / regime helpers (mirror mycelium-vsa, kept native so codegen has its own basis) ──

/// Whether every component is `±1` — the MAP-I bipolar alphabet (mirrors `MapI::check_bipolar`).
/// `pub(crate)` for white-box mutant-witness testing. Returns the index of the first violation.
pub(crate) fn first_non_bipolar(v: &[f64]) -> Option<usize> {
    v.iter().position(|&x| x != 1.0 && x != -1.0)
}

/// Whether every component is `0`/`1` — the BSC binary alphabet (mirrors `Bsc::check_binary`).
/// `pub(crate)` for white-box mutant-witness testing.
pub(crate) fn first_non_binary(v: &[f64]) -> Option<usize> {
    v.iter().position(|&x| x != 0.0 && x != 1.0)
}

/// Whether every component is a finite phase in `(−π, π]` — the FHRR phasor alphabet (mirrors
/// `Fhrr::check_phases`). `pub(crate)` for white-box mutant-witness testing.
pub(crate) fn first_off_phase(v: &[f64]) -> Option<usize> {
    v.iter()
        .position(|&t| !t.is_finite() || t <= -std::f64::consts::PI || t > std::f64::consts::PI)
}

// ─── program validation (mirrors the reference's per-operand side-condition checks) ─────────────

impl VsaProgram {
    /// The operand arity an op requires (the count of `items` it reads). `bundle` reads all (≥ 1).
    fn required_operands(&self) -> usize {
        match self.op {
            VsaCgOp::Permute => 1,
            VsaCgOp::Bind | VsaCgOp::Unbind | VsaCgOp::Similarity => 2,
            VsaCgOp::Bundle => self.items.len().max(1),
        }
    }

    /// The operands this op actually reads (the first `required_operands` of `items`).
    fn operands(&self) -> &[Vec<f64>] {
        &self.items[..self.required_operands().min(self.items.len())]
    }

    /// Validate the program against the same contract the reference enforces: dim-consistent operands,
    /// the model's alphabet, and the op's empirical/capacity regime. Returns an explicit
    /// [`VsaAotError`] for any violation — never a silent coercion (G2), exactly as `mycelium-vsa`
    /// refuses.
    pub(crate) fn validate(&self) -> Result<(), VsaAotError> {
        // Operand-count / malformed-shape gate.
        match self.op {
            VsaCgOp::Bind | VsaCgOp::Unbind | VsaCgOp::Similarity => {
                if self.items.len() < 2 {
                    return Err(VsaAotError::Malformed(format!(
                        "{:?} needs two operands, got {}",
                        self.op,
                        self.items.len()
                    )));
                }
            }
            VsaCgOp::Permute => {
                if self.items.is_empty() {
                    return Err(VsaAotError::Malformed(
                        "permute needs one operand".to_owned(),
                    ));
                }
                if self.shift.is_none() {
                    return Err(VsaAotError::Malformed("permute needs a shift".to_owned()));
                }
            }
            VsaCgOp::Bundle => {
                if self.items.is_empty() {
                    return Err(VsaAotError::EmptyBundle);
                }
            }
        }
        // dim consistency over every operand the op reads.
        for v in self.operands() {
            if v.len() != self.dim as usize {
                return Err(VsaAotError::DimMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
        }
        // Per-model alphabet + op-regime checks.
        self.check_alphabet()?;
        self.check_regime()?;
        Ok(())
    }

    /// Check every operand against the model's alphabet, exactly where the reference does (MAP-I
    /// `±1`, BSC `{0,1}`, FHRR phase in `(−π,π]`). HRR has no alphabet constraint (real vectors).
    fn check_alphabet(&self) -> Result<(), VsaAotError> {
        let model = self.model.registry_id();
        for v in self.operands() {
            let bad = match self.model {
                VsaModelId::MapI => first_non_bipolar(v),
                VsaModelId::Bsc => first_non_binary(v),
                VsaModelId::Fhrr => first_off_phase(v),
                VsaModelId::Hrr => None,
            };
            if let Some(index) = bad {
                return Err(VsaAotError::NonAlphabetComponent { model, index });
            }
        }
        Ok(())
    }

    /// Check the op's empirical / capacity regime, replaying the side-conditions: the BSC/HRR/FHRR
    /// bundle profiles, the HRR/FHRR unbind profile minimum dim, and the MAP-I bundle capacity
    /// side-condition (the `Proven` gate). Never stamps a tag past the basis (VR-5) — an op outside its
    /// trial-validated envelope is an explicit `OutsideEmpiricalProfile` refusal, not an Empirical-anyway.
    fn check_regime(&self) -> Result<(), VsaAotError> {
        match (self.model, self.op) {
            // BSC value-level bundle is profile-gated (odd m ≤ 5, dim ≥ 1024) — the same gate the
            // reference's `bundle_values_empirical` runs via `BSC_BUNDLE_PROFILE.check`.
            (VsaModelId::Bsc, VsaCgOp::Bundle) => BSC_BUNDLE_PROFILE
                .check(self.items.len(), self.dim)
                .map_err(|e| VsaAotError::OutsideEmpiricalProfile(e.to_string())),
            // HRR/FHRR value-level bundle are profile-gated (m ≤ 5, dim ≥ 256) — the codegen-derived
            // HRR_BUNDLE_PROFILE / FHRR_BUNDLE_PROFILE (M-854 FLAG-0 resolution): Empirical within the
            // measured envelope, an explicit refusal beyond it (never claimed past what was measured).
            (VsaModelId::Hrr, VsaCgOp::Bundle) => HRR_BUNDLE_PROFILE
                .check(self.items.len(), self.dim)
                .map_err(|e| VsaAotError::OutsideEmpiricalProfile(e.to_string())),
            (VsaModelId::Fhrr, VsaCgOp::Bundle) => FHRR_BUNDLE_PROFILE
                .check(self.items.len(), self.dim)
                .map_err(|e| VsaAotError::OutsideEmpiricalProfile(e.to_string())),
            // HRR/FHRR value-level unbind require the profile minimum dim (the single-factor regime's
            // structural-provenance check is the reference's; native codegen lowers from raw payloads,
            // so it gates on the *checkable* minimum-dim side-condition and records the regime in the
            // EXPLAIN, never claiming a tighter basis).
            (VsaModelId::Hrr, VsaCgOp::Unbind) => HRR_UNBIND_PROFILE
                .check(1, self.dim)
                .map_err(|e| VsaAotError::OutsideEmpiricalProfile(e.to_string())),
            (VsaModelId::Fhrr, VsaCgOp::Unbind) => FHRR_UNBIND_PROFILE
                .check(1, self.dim)
                .map_err(|e| VsaAotError::OutsideEmpiricalProfile(e.to_string())),
            // MAP-I bundle requires the checked capacity side-condition for its Proven tag.
            (VsaModelId::MapI, VsaCgOp::Bundle) => {
                let delta = self.bundle_delta.ok_or_else(|| {
                    VsaAotError::Malformed(
                        "MAP-I bundle needs a target δ for the Proven capacity bound".to_owned(),
                    )
                })?;
                let m = self.items.len() as u64;
                let dim = u64::from(self.dim);
                if proven_capacity_bound(m, dim, delta).is_none() {
                    return Err(VsaAotError::InsufficientCapacity {
                        items: m,
                        dim,
                        required: mycelium_vsa::capacity::required_dim(
                            m,
                            delta,
                            mycelium_vsa::capacity::MARGIN_MU,
                        ),
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

// ─── the never-silent read-back protocol (float bit-patterns + a degenerate sentinel) ───────────

/// The sentinel line a VSA artifact prints when an FHRR bundle component's phasor sum vanishes — a
/// condition the reference also refuses (`VsaError::DegenerateBundleComponent`). The read-back turns
/// it into an explicit [`VsaAotError::DegenerateBundleComponent`], never a silent value (G2).
pub(crate) const VSA_DEGENERATE_SENTINEL: &str = "DEGENERATE";

// ─── IR emission ────────────────────────────────────────────────────────────────────────────────

/// Emit textual LLVM IR for a VSA program — a `main` that computes each result component (or the
/// scalar measurement) in `f64` and prints its bit pattern, then `ret`. One op per element (no opaque
/// pass — RFC-0004 §6). Returns an explicit [`VsaAotError`] for anything outside the supported
/// fragment. Also returns the inspectable [`VsaExplain`].
pub fn emit_vsa_llvm_ir(prog: &VsaProgram) -> Result<(String, VsaExplain), VsaAotError> {
    prog.validate()?;
    let explain = mk_explain(prog);

    let mut out = String::from(
        "; mycelium direct-LLVM VSA codegen (real-Vec<f64> MAP-I/BSC/HRR/FHRR; M-854; RFC-0039 §5.2)\n",
    );
    emit_explain_comment(&explain, &mut out);
    // printf for the read-back protocol; the never-silent degenerate-phasor path prints a sentinel.
    out.push_str("declare i32 @printf(i8*, ...)\n");
    out.push_str("@.fmt_u64 = private constant [6 x i8] c\"%llu \\00\"\n");
    out.push_str("@.fmt_nl = private constant [2 x i8] c\"\\0A\\00\"\n");
    out.push_str("@.s_deg = private constant [11 x i8] c\"DEGENERATE\\00\"\n");
    out.push('\n');
    out.push_str("define i32 @main() {\nentry:\n");

    let mut ssa = Ssa(0);
    let mut body = String::new();
    match prog.op {
        VsaCgOp::Bind => emit_bind(prog, false, &mut ssa, &mut body)?,
        VsaCgOp::Unbind => emit_bind(prog, true, &mut ssa, &mut body)?,
        VsaCgOp::Bundle => emit_bundle(prog, &mut ssa, &mut body)?,
        VsaCgOp::Permute => emit_permute(prog, &mut ssa, &mut body)?,
        VsaCgOp::Similarity => emit_similarity(prog, &mut ssa, &mut body)?,
    }
    out.push_str(&body);
    out.push_str("  ret i32 0\n}\n");
    Ok((out, explain))
}

pub(crate) fn mk_explain(prog: &VsaProgram) -> VsaExplain {
    VsaExplain {
        op: prog
            .model
            .op_name(prog.op)
            .unwrap_or_else(|| format!("{}.similarity", prog.model.op_prefix())),
        model: prog.model.registry_id(),
        dim: prog.dim,
        // Value ops record the `VsaStore{sparse:false}` schedule (DN-01 — the inspectable
        // `Meta.physical`); the measurement produces a bare `f64`, so no physical schedule.
        physical: prog
            .op
            .is_value_op()
            .then_some(PhysicalLayout::VsaStore { sparse: false }),
        reference_guarantee: prog.model.reference_guarantee(prog.op),
        codegen_guarantee: VSA_CODEGEN_GUARANTEE,
        carrier: CARRIER_STATUS,
    }
}

/// Emit the dumpable EXPLAIN comment into the IR (RFC-0004 §6 — the op's basis is visible in the
/// `.ll`; never a black box, G2).
pub(crate) fn emit_explain_comment(e: &VsaExplain, out: &mut String) {
    let _ = writeln!(
        out,
        "; vsa {} | model={} dim={} | physical={:?} | ref-guarantee={:?} | codegen-guarantee={:?} \
         | carrier={}",
        e.op, e.model, e.dim, e.physical, e.reference_guarantee, e.codegen_guarantee, e.carrier,
    );
}

/// Emit `bind`/`unbind` for the program's model. `inverse` selects unbind. Each model's algebra is
/// emitted as explicit per-element `f64` IR, mirroring the reference op-for-op.
fn emit_bind(
    prog: &VsaProgram,
    inverse: bool,
    ssa: &mut Ssa,
    body: &mut String,
) -> Result<(), VsaAotError> {
    let a = &prog.items[0];
    let b = &prog.items[1];
    match prog.model {
        // MAP-I: elementwise product (self-inverse — unbind == bind).
        VsaModelId::MapI => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let p = ssa.fresh();
                let _ = writeln!(
                    body,
                    "  {p} = fmul double {}, {}",
                    f64_const(ai),
                    f64_const(bi)
                );
                emit_print_f64_bits(&p, ssa, body);
            }
        }
        // BSC: elementwise XOR on {0,1} == |a − b| (self-inverse).
        VsaModelId::Bsc => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let d = ssa.fresh();
                let _ = writeln!(
                    body,
                    "  {d} = fsub double {}, {}",
                    f64_const(ai),
                    f64_const(bi)
                );
                let r = ssa.fresh();
                let _ = writeln!(body, "  {r} = call double @llvm.fabs.f64(double {d})");
                emit_print_f64_bits(&r, ssa, body);
            }
        }
        // HRR: circular convolution; unbind convolves with the involution of b.
        VsaModelId::Hrr => {
            let bv: Vec<f64> = if inverse {
                hrr_involution(b)
            } else {
                b.clone()
            };
            emit_cconv(a, &bv, ssa, body);
        }
        // FHRR: phase add (bind) / phase sub (unbind), each wrapped to (−π, π].
        VsaModelId::Fhrr => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let raw = ssa.fresh();
                let fop = if inverse { "fsub" } else { "fadd" };
                let _ = writeln!(
                    body,
                    "  {raw} = {fop} double {}, {}",
                    f64_const(ai),
                    f64_const(bi)
                );
                let wrapped = emit_wrap_phase(&raw, ssa, body);
                emit_print_f64_bits(&wrapped, ssa, body);
            }
        }
    }
    emit_newline(ssa, body);
    Ok(())
}

/// The HRR involution `b~[i] = b[(−i) mod d]` (host-side; the convolution operands are constants, so
/// the involution is folded at emit time, mirroring `Hrr::involution`). `pub(crate)` for white-box
/// mutant-witness testing.
pub(crate) fn hrr_involution(b: &[f64]) -> Vec<f64> {
    let d = b.len();
    (0..d).map(|i| b[(d - i) % d]).collect()
}

/// Emit HRR circular convolution `out[k] = Σᵢ a[i]·b[(k+d−i) mod d]` in `f64`, accumulating
/// left-to-right exactly as `Hrr::cconv` (the trusted reference's naive `O(d²)` form). Each
/// product/accumulate is explicit IR (§6).
fn emit_cconv(a: &[f64], b: &[f64], ssa: &mut Ssa, body: &mut String) {
    let d = a.len();
    for k in 0..d {
        let mut acc = "0.0".to_owned();
        for (i, &ai) in a.iter().enumerate() {
            let bi = b[(k + d - i) % d];
            let p = ssa.fresh();
            let _ = writeln!(
                body,
                "  {p} = fmul double {}, {}",
                f64_const(ai),
                f64_const(bi)
            );
            let next = ssa.fresh();
            let _ = writeln!(body, "  {next} = fadd double {acc}, {p}");
            acc = next;
        }
        emit_print_f64_bits(&acc, ssa, body);
    }
}

/// Emit `bundle` for the program's model. MAP-I/HRR sum; BSC majority; FHRR complex-sum phasor.
fn emit_bundle(prog: &VsaProgram, ssa: &mut Ssa, body: &mut String) -> Result<(), VsaAotError> {
    let items = &prog.items;
    let dim = prog.dim as usize;
    match prog.model {
        // MAP-I / HRR: elementwise sum, accumulating left-to-right (matches the reference's `+=`).
        VsaModelId::MapI | VsaModelId::Hrr => {
            for idx in 0..dim {
                let mut acc = f64_const(items[0][idx]);
                for item in &items[1..] {
                    let next = ssa.fresh();
                    let _ = writeln!(
                        body,
                        "  {next} = fadd double {acc}, {}",
                        f64_const(item[idx])
                    );
                    acc = next;
                }
                // `acc` may be a bare constant (single-item bundle) or an SSA register — print either.
                emit_print_f64_bits(&acc, ssa, body);
            }
        }
        // BSC: majority — count ones; > half → 1, < half → 0, tie → first operand's bit. Folded
        // host-side (operands are constants and on the {0,1} alphabet), then the bit is emitted as a
        // constant; this mirrors `Bsc::bundle` exactly (n compared to items.len()/2).
        VsaModelId::Bsc => {
            let half = items.len() as f64 / 2.0;
            for idx in 0..dim {
                let n: f64 = items.iter().map(|v| v[idx]).sum();
                let bit = if n > half {
                    1.0
                } else if n < half {
                    0.0
                } else {
                    items[0][idx]
                };
                // The bit is an exact f64 constant — emit its bit pattern directly (no IR arithmetic
                // needed; the result equals the reference's majority bit bit-for-bit).
                emit_print_const_f64_bits(bit, body);
            }
        }
        // FHRR: per-component re = Σ cos θ, im = Σ sin θ; |sum| < 1e-9 → degenerate (never-silent
        // sentinel); else wrap(atan2(im, re)). All in f64, mirroring `Fhrr::bundle`.
        VsaModelId::Fhrr => {
            emit_fhrr_bundle(items, dim, ssa, body);
        }
    }
    emit_newline(ssa, body);
    Ok(())
}

/// Emit the FHRR bundle: for each component accumulate `re = Σ cos`, `im = Σ sin` in `f64`, branch to
/// the never-silent `DEGENERATE` sentinel if `√(re²+im²) < 1e-9`, else print `wrap(atan2(im, re))`.
fn emit_fhrr_bundle(items: &[Vec<f64>], dim: usize, ssa: &mut Ssa, body: &mut String) {
    for idx in 0..dim {
        let mut re = "0.0".to_owned();
        let mut im = "0.0".to_owned();
        for item in items {
            let theta = f64_const(item[idx]);
            let c = ssa.fresh();
            let _ = writeln!(body, "  {c} = call double @cos(double {theta})");
            let s = ssa.fresh();
            let _ = writeln!(body, "  {s} = call double @sin(double {theta})");
            let re_next = ssa.fresh();
            let _ = writeln!(body, "  {re_next} = fadd double {re}, {c}");
            let im_next = ssa.fresh();
            let _ = writeln!(body, "  {im_next} = fadd double {im}, {s}");
            re = re_next;
            im = im_next;
        }
        // magnitude = sqrt(re*re + im*im).
        let re2 = ssa.fresh();
        let _ = writeln!(body, "  {re2} = fmul double {re}, {re}");
        let im2 = ssa.fresh();
        let _ = writeln!(body, "  {im2} = fmul double {im}, {im}");
        let sumsq = ssa.fresh();
        let _ = writeln!(body, "  {sumsq} = fadd double {re2}, {im2}");
        let mag = ssa.fresh();
        let _ = writeln!(body, "  {mag} = call double @llvm.sqrt.f64(double {sumsq})");
        let deg = ssa.fresh();
        // 1e-9 threshold — matches the reference's `< 1e-9`.
        let _ = writeln!(body, "  {deg} = fcmp olt double {mag}, {}", f64_const(1e-9));
        let deg_lbl = ssa.fresh_label();
        let ok_lbl = ssa.fresh_label();
        let _ = writeln!(body, "  br i1 {deg}, label %{deg_lbl}, label %{ok_lbl}");
        let _ = writeln!(body, "{deg_lbl}:");
        emit_print_sentinel("@.s_deg", 11, ssa, body);
        let _ = writeln!(body, "  ret i32 0");
        let _ = writeln!(body, "{ok_lbl}:");
        let theta = ssa.fresh();
        // `atan2` has no LLVM intrinsic — call the libm symbol directly (linked with `-lm`), matching
        // the reference's `f64::atan2` (which is the same libm `atan2`).
        let _ = writeln!(
            body,
            "  {theta} = call double @atan2(double {im}, double {re})"
        );
        let wrapped = emit_wrap_phase(&theta, ssa, body);
        emit_print_f64_bits(&wrapped, ssa, body);
    }
}

/// Emit `permute` — cyclic left rotation by `shift` (`result[i] = a[(i + shift) mod d]`, `rem_euclid`).
/// Folded host-side (the operand + shift are constants), mirroring `rotate`. The permuted components
/// are the input's own exact `f64`s, so they are emitted as constants (a coordinate bijection — Exact).
fn emit_permute(prog: &VsaProgram, ssa: &mut Ssa, body: &mut String) -> Result<(), VsaAotError> {
    let a = &prog.items[0];
    let shift = prog
        .shift
        .ok_or_else(|| VsaAotError::Malformed("permute needs a shift".to_owned()))?;
    let d = a.len() as i64;
    for i in 0..a.len() {
        let src = (i as i64 + shift).rem_euclid(d) as usize;
        emit_print_const_f64_bits(a[src], body);
    }
    emit_newline(ssa, body);
    Ok(())
}

/// Emit `similarity` — the per-model measurement in `f64`, mirroring each model's `similarity` exactly:
/// cosine (MAP-I/HRR), centered Hamming `1 − 2·d_H/d` (BSC), mean `cos(θa − θb)` (FHRR). Prints one
/// `f64` measurement.
fn emit_similarity(prog: &VsaProgram, ssa: &mut Ssa, body: &mut String) -> Result<(), VsaAotError> {
    let a = &prog.items[0];
    let b = &prog.items[1];
    let sim = match prog.model {
        VsaModelId::MapI | VsaModelId::Hrr => emit_cosine(a, b, ssa, body),
        VsaModelId::Bsc => emit_hamming_sim(a, b, ssa, body),
        VsaModelId::Fhrr => emit_phase_sim(a, b, ssa, body),
    };
    emit_print_f64_bits(&sim, ssa, body);
    emit_newline(ssa, body);
    Ok(())
}

/// Emit cosine `dot / (‖a‖·‖b‖)`, `0` on a zero-norm operand (mirrors `wrap::cosine` / `MapI::similarity`
/// — all in `f64`, summed left-to-right). Returns the result SSA register.
pub(crate) fn emit_cosine(a: &[f64], b: &[f64], ssa: &mut Ssa, body: &mut String) -> String {
    let dot = emit_dot_acc(a, b, ssa, body);
    let na2 = emit_dot_acc(a, a, ssa, body);
    let nb2 = emit_dot_acc(b, b, ssa, body);
    let na = ssa.fresh();
    let _ = writeln!(body, "  {na} = call double @llvm.sqrt.f64(double {na2})");
    let nb = ssa.fresh();
    let _ = writeln!(body, "  {nb} = call double @llvm.sqrt.f64(double {nb2})");
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
    sim
}

/// Emit the BSC centered-Hamming similarity `1 − 2·d_H/d` (mirrors `Bsc::similarity`). `d_H` counts
/// positions where `a ≠ b`; the count + the `1 − 2·h/d` are in `f64`, matching the reference exactly.
pub(crate) fn emit_hamming_sim(a: &[f64], b: &[f64], ssa: &mut Ssa, body: &mut String) -> String {
    let mut hamm = "0.0".to_owned();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        // h += (a == b) ? 0.0 : 1.0
        let eq = ssa.fresh();
        let _ = writeln!(
            body,
            "  {eq} = fcmp oeq double {}, {}",
            f64_const(ai),
            f64_const(bi)
        );
        let inc = ssa.fresh();
        let _ = writeln!(body, "  {inc} = select i1 {eq}, double 0.0, double 1.0");
        let next = ssa.fresh();
        let _ = writeln!(body, "  {next} = fadd double {hamm}, {inc}");
        hamm = next;
    }
    let len = a.len() as f64;
    // sim = 1 - 2*h/len
    let ratio = ssa.fresh();
    let _ = writeln!(body, "  {ratio} = fdiv double {hamm}, {}", f64_const(len));
    let two = ssa.fresh();
    let _ = writeln!(body, "  {two} = fmul double {}, {ratio}", f64_const(2.0));
    let sim = ssa.fresh();
    let _ = writeln!(body, "  {sim} = fsub double {}, {two}", f64_const(1.0));
    sim
}

/// Emit the FHRR phase similarity `mean cos(θa − θb)` (mirrors `Fhrr::similarity`) — summed
/// left-to-right then divided by `len`, all in `f64`.
pub(crate) fn emit_phase_sim(a: &[f64], b: &[f64], ssa: &mut Ssa, body: &mut String) -> String {
    let mut acc = "0.0".to_owned();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let diff = ssa.fresh();
        let _ = writeln!(
            body,
            "  {diff} = fsub double {}, {}",
            f64_const(ai),
            f64_const(bi)
        );
        let c = ssa.fresh();
        let _ = writeln!(body, "  {c} = call double @cos(double {diff})");
        let next = ssa.fresh();
        let _ = writeln!(body, "  {next} = fadd double {acc}, {c}");
        acc = next;
    }
    let len = a.len() as f64;
    let sim = ssa.fresh();
    let _ = writeln!(body, "  {sim} = fdiv double {acc}, {}", f64_const(len));
    sim
}

/// Accumulate `Σ xᵢ·yᵢ` in `f64`, left-to-right, returning the accumulator register. Mirrors the
/// reference's `f64` `.sum()` (which folds left-to-right). Each step explicit IR (§6).
fn emit_dot_acc(xs: &[f64], ys: &[f64], ssa: &mut Ssa, body: &mut String) -> String {
    let mut acc = "0.0".to_owned();
    for (x, y) in xs.iter().zip(ys.iter()) {
        let p = ssa.fresh();
        let _ = writeln!(
            body,
            "  {p} = fmul double {}, {}",
            f64_const(*x),
            f64_const(*y)
        );
        let next = ssa.fresh();
        let _ = writeln!(body, "  {next} = fadd double {acc}, {p}");
        acc = next;
    }
    acc
}

/// Emit `wrap_phase(theta)` = `let u = theta.rem_euclid(TAU); if u > π { u − TAU } else u`, mirroring
/// `fhrr::wrap_phase` digit-for-digit. The `rem_euclid` is emitted as **exactly** Rust's `f64`
/// algorithm — `let r = theta % TAU; if r < 0 { r + TAU } else { r }` (TAU > 0 so `|TAU| = TAU`) —
/// using the LLVM `frem` for `%`. This matches `f64::rem_euclid` **bit-for-bit including the `-0.0`
/// sign** (a `floor`-based identity `theta − TAU·floor(theta/TAU)` agrees on every magnitude *except*
/// `theta = -0.0`, which it would flip to `+0.0` — verified over 2·10⁶ samples; using `frem` closes
/// that edge so the read-back stays bit-exact for a `-0.0` phase sum). Returns the wrapped register.
pub(crate) fn emit_wrap_phase(theta: &str, ssa: &mut Ssa, body: &mut String) -> String {
    let tau = f64_const(std::f64::consts::TAU);
    let pi = f64_const(std::f64::consts::PI);
    // r = theta % TAU  (frem); rem_euclid: if r < 0 { r + TAU } else { r }.
    let r0 = ssa.fresh();
    let _ = writeln!(body, "  {r0} = frem double {theta}, {tau}");
    let neg = ssa.fresh();
    let _ = writeln!(body, "  {neg} = fcmp olt double {r0}, 0.0");
    let plus = ssa.fresh();
    let _ = writeln!(body, "  {plus} = fadd double {r0}, {tau}");
    let u = ssa.fresh();
    let _ = writeln!(body, "  {u} = select i1 {neg}, double {plus}, double {r0}");
    // if u > π { u − TAU } else { u }
    let gt = ssa.fresh();
    let _ = writeln!(body, "  {gt} = fcmp ogt double {u}, {pi}");
    let shifted = ssa.fresh();
    let _ = writeln!(body, "  {shifted} = fsub double {u}, {tau}");
    let r = ssa.fresh();
    let _ = writeln!(body, "  {r} = select i1 {gt}, double {shifted}, double {u}");
    r
}

/// Print one `f64` SSA value's IEEE-754 bit pattern as a decimal `u64` (so the read-back is bit-exact).
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

/// Print a *constant* `f64`'s bit pattern directly (for ops folded host-side — permute, the BSC
/// majority bit). The constant's bits equal the reference's, so the read-back is bit-exact.
fn emit_print_const_f64_bits(x: f64, body: &mut String) {
    let _ = writeln!(
        body,
        "  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* \
         @.fmt_u64, i64 0, i64 0), i64 {})",
        x.to_bits()
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

/// Render an `f64` as an exact LLVM `double` constant (hex form — bit-exact, no decimal round-trip).
pub(crate) fn f64_const(x: f64) -> String {
    format!("0x{:016X}", x.to_bits())
}

// ─── SSA / label counters (local; the llvm.rs ones are pub(crate) but coupled to that module) ───

/// SSA register counter for the VSA module (separate from `llvm::Ssa` so the two never collide).
/// `pub(crate)` so the JIT store-sink emitter (`vsa_jit.rs`, M-855) shares the exact same counter
/// discipline as this module's print-sink emitter — one SSA-naming scheme, never two that could drift.
pub(crate) struct Ssa(usize);
impl Ssa {
    /// A fresh SSA counter starting at register `%r0` (mirrors this module's own `Ssa(0)` construction).
    pub(crate) fn new() -> Self {
        Ssa(0)
    }
    pub(crate) fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("%r{n}")
    }
    pub(crate) fn fresh_label(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("bb{n}")
    }
}

// ─── compile / run (drives llc + clang; reuses the llvm.rs toolchain helpers) ───────────────────

/// A compiled native VSA artifact: the executable on disk (cleaned on drop) plus the read-back shape
/// (op, model, dim) needed to reconstruct the result `Value`/measurement.
pub struct VsaArtifact {
    _dir: TmpDir,
    bin: std::path::PathBuf,
    op: VsaCgOp,
    model: VsaModelId,
    dim: u32,
    /// The MAP-I bundle δ (carried so the read-back can re-issue the same checked `Proven` bound).
    bundle_delta: Option<f64>,
    /// The bundle item count (for the MAP-I capacity bound's `items`).
    item_count: u64,
}

impl VsaArtifact {
    /// Build a read-back-**shape-only** artifact — carries the `(op, model, dim, bundle_delta,
    /// item_count)` shape but no executable (`bin`/`_dir` are empty placeholders, matching
    /// [`Self::for_readback_test`]'s no-execution contract). `pub(crate)`, always available (not
    /// `#[cfg(test)]`) so the **dynamic-VSA JIT** (`vsa_jit.rs`, M-855) can reuse this module's
    /// read-back methods (`reconstruct_value` / `result_meta` / `result_bound`) verbatim over its own
    /// `dlopen`-sourced `u64` buffer — the JIT read-back Meta/guarantee construction is then *provably*
    /// identical to the AOT path's (DRY; the two execution modes can never silently diverge on how a
    /// result `Value` is stamped).
    pub(crate) fn for_shape(
        op: VsaCgOp,
        model: VsaModelId,
        dim: u32,
        bundle_delta: Option<f64>,
        item_count: u64,
    ) -> Self {
        VsaArtifact {
            _dir: TmpDir(std::path::PathBuf::new()),
            bin: std::path::PathBuf::new(),
            op,
            model,
            dim,
            bundle_delta,
            item_count,
        }
    }

    /// Build an artifact around an **already-compiled** executable — the MLIR-dialect sibling emitter
    /// (`dialect::native::vsa`, M-856b) compiles through its own pipeline (`mlir-opt` / `mlir-translate`
    /// / `clang`, not `llc`/`clang`), then reuses this constructor so [`Self::run`]'s read-back/
    /// reconstruct logic runs **verbatim** for both compiled paths — the two can never silently diverge
    /// on how a result `Value` is stamped (DRY; VR-5). `pub(crate)`, and gated to the `mlir-dialect`
    /// feature — its only caller — so a default (feature-off) build carries no dead-code warning for
    /// a constructor that exists solely for that path.
    #[cfg(feature = "mlir-dialect")]
    #[allow(clippy::too_many_arguments)] // mirrors the read-back shape exactly; no natural grouping
    pub(crate) fn from_binary(
        dir: TmpDir,
        bin: std::path::PathBuf,
        op: VsaCgOp,
        model: VsaModelId,
        dim: u32,
        bundle_delta: Option<f64>,
        item_count: u64,
    ) -> Self {
        VsaArtifact {
            _dir: dir,
            bin,
            op,
            model,
            dim,
            bundle_delta,
            item_count,
        }
    }

    /// White-box constructor for the **toolchain-independent read-back tests** (M-854 mutant-witness).
    /// The read-back metadata methods (`result_bound` / `result_meta` / `reconstruct_value`) read only
    /// the *shape* fields (`op`, `model`, `dim`, `bundle_delta`, `item_count`) — never `bin`/`_dir` —
    /// so a test can witness those code paths with a placeholder binary, no `llc`/`clang` required (the
    /// emission-assertion discipline that keeps the mutant catch-rate environment-independent; VR-5,
    /// the M-725 `ran_mlir` non-vacuity lesson). The `bin` it carries is never executed by those
    /// methods, so it points nowhere; the empty `TmpDir` drop is a no-op.
    #[cfg(test)]
    pub(crate) fn for_readback_test(
        op: VsaCgOp,
        model: VsaModelId,
        dim: u32,
        bundle_delta: Option<f64>,
        item_count: u64,
    ) -> Self {
        VsaArtifact {
            _dir: TmpDir(std::path::PathBuf::new()),
            bin: std::path::PathBuf::new(),
            op,
            model,
            dim,
            bundle_delta,
            item_count,
        }
    }

    /// White-box constructor pointing `bin` at an **arbitrary executable** (a universal POSIX utility
    /// like `/bin/true` / `/bin/false`, **never** `llc`/`clang`), so [`Self::run`]'s process-level
    /// exit-status guard can be witnessed toolchain-independently (no AOT toolchain — the vacuity-prone
    /// leg is the `llc`/`clang` differential, not coreutils). The read-back-shape fields are
    /// placeholders (the status guard fires before they matter).
    #[cfg(test)]
    pub(crate) fn for_exec_test(bin: std::path::PathBuf, op: VsaCgOp) -> Self {
        VsaArtifact {
            _dir: TmpDir(std::path::PathBuf::new()),
            bin,
            op,
            model: VsaModelId::Hrr,
            dim: 1,
            bundle_delta: None,
            item_count: 1,
        }
    }

    /// Run the artifact and read its result back. A value op reconstructs a VSA [`Value`] carrying the
    /// reference's per-op guarantee tag; a measurement op returns a bare `f64`. A sentinel line
    /// (degenerate phasor) is surfaced as an explicit [`VsaAotError`] — never a silent value (G2).
    pub fn run(&self) -> Result<VsaResult, VsaAotError> {
        let output = Command::new(&self.bin)
            .output()
            .map_err(|e| VsaAotError::Run(format!("exec {}: {e}", self.bin.display())))?;
        if !output.status.success() {
            return Err(VsaAotError::Run(format!(
                "artifact exited {}",
                output.status
            )));
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|e| VsaAotError::Parse(format!("non-utf8 output: {e}")))?;
        self.parse_stdout(&stdout)
    }

    /// Parse a captured artifact stdout into the result `VsaResult`, applying the never-silent
    /// read-back protocol: scan every token of the first line, surface the `DEGENERATE` sentinel as an
    /// explicit refusal (G2), reconstruct a `Value` for a value op, and require exactly one element for
    /// a measurement. Split out of [`Self::run`] so the read-back logic is **witnessable without the
    /// `llc`/`clang` toolchain** (the env-independent mutant-witness; VR-5) — `run` only adds the
    /// process exec + exit-status check around it.
    pub(crate) fn parse_stdout(&self, stdout: &str) -> Result<VsaResult, VsaAotError> {
        let line = stdout.lines().next().unwrap_or("").trim();
        // Never-silent sentinel (matches VsaError::DegenerateBundleComponent). A sentinel can appear
        // anywhere on the line (after the earlier in-range components), so scan every token.
        let mut bits: Vec<u64> = Vec::new();
        for tok in line.split_whitespace() {
            if tok == VSA_DEGENERATE_SENTINEL {
                return Err(VsaAotError::DegenerateBundleComponent);
            }
            bits.push(
                tok.parse::<u64>()
                    .map_err(|e| VsaAotError::Parse(format!("non-u64 token {tok:?}: {e}")))?,
            );
        }
        if self.op.is_value_op() {
            self.reconstruct_value(&bits)
        } else {
            if bits.len() != 1 {
                return Err(VsaAotError::Parse(format!(
                    "measurement expected 1 element, got {}",
                    bits.len()
                )));
            }
            Ok(VsaResult::Measurement(f64::from_bits(bits[0])))
        }
    }

    /// Reconstruct the VSA `Value` from the printed f64 bit patterns, carrying the reference's per-op
    /// guarantee tag (so the observable matches the reference). `pub(crate)` for the toolchain-free
    /// white-box read-back mutant-witness (the `dim`-mismatch guard + the per-op `Meta` are pinned
    /// directly, not only behind a compiled run).
    pub(crate) fn reconstruct_value(&self, bits: &[u64]) -> Result<VsaResult, VsaAotError> {
        if bits.len() != self.dim as usize {
            return Err(VsaAotError::Parse(format!(
                "expected {} components, got {}",
                self.dim,
                bits.len()
            )));
        }
        let xs: Vec<f64> = bits.iter().map(|&b| f64::from_bits(b)).collect();
        let repr = Repr::Vsa {
            model: self.model.registry_id().to_owned(),
            dim: self.dim,
            sparsity: SparsityClass::Dense,
        };
        let meta = self.result_meta()?;
        Value::new(repr, Payload::Hypervector(xs), meta)
            .map(|v| VsaResult::Value(Box::new(v)))
            .map_err(|e| VsaAotError::Wf(e.to_string()))
    }

    /// Build the result `Meta` mirroring the reference's RFC-0003 §4.1 value-level guarantee: `Exact`
    /// for the algebraically-exact ops (bind/unbind on MAP-I/BSC, permute), `Empirical` for the
    /// weak-link ops with a reference value-level profile (HRR/FHRR unbind, BSC bundle) carrying that
    /// profile bound, `Proven` for the MAP-I bundle carrying the **checked** capacity bound (the same
    /// side-condition the reference checks — VR-5/M-I2), and `Declared` (with a flagged `UserDeclared`
    /// bound) for HRR/FHRR bundle, which the reference exposes no value-level bound for — the honest
    /// downgrade (never a fabricated `Empirical`; VR-5). `Meta.physical = VsaStore` records the
    /// inspectable schedule. `pub(crate)` for the toolchain-free read-back mutant-witness.
    pub(crate) fn result_meta(&self) -> Result<Meta, VsaAotError> {
        let map_wf = |e: WfError| VsaAotError::Wf(e.to_string());
        let physical = Some(PhysicalLayout::VsaStore { sparse: false });
        let op_name = self.model.op_name(self.op).ok_or_else(|| {
            VsaAotError::Malformed("measurement op has no result Meta".to_owned())
        })?;
        let guarantee = self
            .model
            .reference_guarantee(self.op)
            .ok_or_else(|| VsaAotError::Malformed("measurement op has no guarantee".to_owned()))?;
        let provenance = Provenance::Derived {
            op: operation_hash(&op_name),
            inputs: vec![],
        };
        let bound: Option<Bound> = self.result_bound(guarantee)?;
        Meta::new(provenance, guarantee, bound, None, physical, None).map_err(map_wf)
    }

    /// The bound the read-back `Meta` carries, matching how the op was verified: the trial profile
    /// bound for an `Empirical` op (BSC/HRR/FHRR bundle, HRR/FHRR unbind), the **checked** capacity
    /// bound for the MAP-I `Proven` bundle, none for `Exact`. `pub(crate)` so the toolchain-free
    /// read-back mutant-witness can pin the exact `Option<Bound>` each `(model, op, strength)` arm
    /// returns (deleting any arm flips a checked value; VR-5).
    pub(crate) fn result_bound(
        &self,
        guarantee: GuaranteeStrength,
    ) -> Result<Option<Bound>, VsaAotError> {
        Ok(match (self.model, self.op, guarantee) {
            (_, _, GuaranteeStrength::Exact) => None,
            // MAP-I Proven bundle: replay the reference's checked capacity bound (same side-condition).
            (VsaModelId::MapI, VsaCgOp::Bundle, GuaranteeStrength::Proven) => {
                let delta = self.bundle_delta.ok_or_else(|| {
                    VsaAotError::Malformed("MAP-I bundle missing δ at read-back".to_owned())
                })?;
                Some(
                    proven_capacity_bound(self.item_count, u64::from(self.dim), delta).ok_or_else(
                        || {
                            VsaAotError::Malformed(
                                "MAP-I bundle capacity side-condition failed at read-back"
                                    .to_owned(),
                            )
                        },
                    )?,
                )
            }
            // Empirical bundle ops carry their trial-validated profile bound (the EmpiricalFit δ the
            // §profile measured): BSC the reference's profile, HRR/FHRR the codegen-derived profiles.
            (VsaModelId::Bsc, VsaCgOp::Bundle, GuaranteeStrength::Empirical) => {
                Some(BSC_BUNDLE_PROFILE.bound())
            }
            (VsaModelId::Hrr, VsaCgOp::Bundle, GuaranteeStrength::Empirical) => {
                Some(HRR_BUNDLE_PROFILE.bound())
            }
            (VsaModelId::Fhrr, VsaCgOp::Bundle, GuaranteeStrength::Empirical) => {
                Some(FHRR_BUNDLE_PROFILE.bound())
            }
            // Empirical unbind ops carry the reference's trial-validated profile bound.
            (VsaModelId::Hrr, VsaCgOp::Unbind, GuaranteeStrength::Empirical) => {
                Some(HRR_UNBIND_PROFILE.bound())
            }
            (VsaModelId::Fhrr, VsaCgOp::Unbind, GuaranteeStrength::Empirical) => {
                Some(FHRR_UNBIND_PROFILE.bound())
            }
            _ => None,
        })
    }
}

/// The `declare` block of intrinsics a `(model, op)` pulls in — only what each op actually emits, so a
/// reader sees exactly its dependencies (no over-declaration). Pure (no toolchain), so the
/// per-`(model, op)` declaration guards are **witnessable without `llc`/`clang`** (the env-independent
/// mutant-witness; VR-5). Each `&&` here is load-bearing — a `&& → ||` widening would declare an
/// intrinsic the op never calls (an unjustified, never-silent-violating decl); the `if !decls`
/// idempotence and the nested `Bundle`-only block keep `cos` (bundle+similarity) separate from
/// `sin`/`atan2`/`sqrt` (bundle-only).
pub(crate) fn intrinsic_decls(model: VsaModelId, op: VsaCgOp) -> String {
    let mut decls = String::new();
    // BSC bind/unbind: |a−b| uses `fabs`.
    if matches!(model, VsaModelId::Bsc) && matches!(op, VsaCgOp::Bind | VsaCgOp::Unbind) {
        decls.push_str("declare double @llvm.fabs.f64(double)\n");
    }
    // FHRR bundle + similarity use libm `cos`/`sin` (bit-exact with the reference's `f64::cos`/`sin`);
    // FHRR bind/unbind only `frem`-wrap a phase (no trig). FHRR bundle additionally needs `atan2`
    // (a libm symbol — no LLVM intrinsic) and `sqrt` (the magnitude guard).
    if matches!(model, VsaModelId::Fhrr) && matches!(op, VsaCgOp::Bundle | VsaCgOp::Similarity) {
        decls.push_str("declare double @cos(double)\n");
        if matches!(op, VsaCgOp::Bundle) {
            decls.push_str("declare double @sin(double)\n");
            decls.push_str("declare double @atan2(double, double)\n");
            decls.push_str("declare double @llvm.sqrt.f64(double)\n");
        }
    }
    // Cosine similarity (MAP-I/HRR) needs `sqrt` for the norms.
    if matches!(op, VsaCgOp::Similarity) && matches!(model, VsaModelId::MapI | VsaModelId::Hrr) {
        decls.push_str("declare double @llvm.sqrt.f64(double)\n");
    }
    decls
}

/// Assemble the **complete** module IR `vsa_compile` feeds to `llc`: the per-element body from
/// [`emit_vsa_llvm_ir`] with the op's [`intrinsic_decls`] block prepended ahead of `@main` (only when
/// the op pulls any intrinsic in — `if !decls.is_empty()`, so a no-intrinsic op's IR is untouched).
/// Pure (no toolchain), split out of [`vsa_compile`] so the decl-prepend guard is **witnessable
/// without `llc`/`clang`** (the env-independent mutant-witness; VR-5). Returns the same explicit
/// refusal as [`emit_vsa_llvm_ir`] for an out-of-fragment program.
pub(crate) fn assemble_compile_ir(prog: &VsaProgram) -> Result<String, VsaAotError> {
    let (mut ir, _explain) = emit_vsa_llvm_ir(prog)?;
    let decls = intrinsic_decls(prog.model, prog.op);
    if !decls.is_empty() {
        ir = ir.replacen(
            "define i32 @main()",
            &format!("{decls}define i32 @main()"),
            1,
        );
    }
    Ok(ir)
}

/// Compile a VSA program to a native executable (emit IR → `llc` → `clang`) without running it.
/// Returns [`VsaAotError::ToolchainMissing`] when `llc`/`clang` are absent (callers skip); any
/// out-of-fragment construct is the same explicit refusal as [`emit_vsa_llvm_ir`].
pub fn vsa_compile(prog: &VsaProgram) -> Result<VsaArtifact, VsaAotError> {
    let ir = assemble_compile_ir(prog)?;
    ensure_toolchain()?;
    let dir = unique_tmp_dir().map_err(aot_to_vsa)?;
    let ll = dir.join("vsa.ll");
    let obj = dir.join("vsa.o");
    let bin = dir.join("vsa");
    let guard = TmpDir(dir);
    std::fs::write(&ll, ir.as_bytes()).map_err(|e| VsaAotError::Run(format!("write IR: {e}")))?;
    run_tool(
        "llc",
        &[
            "-relocation-model=pic",
            "-filetype=obj",
            path(&ll).map_err(aot_to_vsa)?,
            "-o",
            path(&obj).map_err(aot_to_vsa)?,
        ],
    )
    .map_err(aot_to_vsa)?;
    run_tool(
        "clang",
        &[
            path(&obj).map_err(aot_to_vsa)?,
            "-o",
            path(&bin).map_err(aot_to_vsa)?,
            "-lm",
        ],
    )
    .map_err(aot_to_vsa)?;
    Ok(VsaArtifact {
        _dir: guard,
        bin,
        op: prog.op,
        model: prog.model,
        dim: prog.dim,
        bundle_delta: prog.bundle_delta,
        item_count: prog.items.len() as u64,
    })
}

/// Compile + run a VSA program: the compiled execution path the M-854 differential checks against the
/// `mycelium-vsa` reference.
pub fn vsa_compile_and_run(prog: &VsaProgram) -> Result<VsaResult, VsaAotError> {
    vsa_compile(prog)?.run()
}

/// Map a `llvm::AotError` (from the reused toolchain helpers) into a `VsaAotError`, preserving the
/// never-silent classification (toolchain-missing stays a skip; a real compile/run failure stays an
/// error).
pub(crate) fn aot_to_vsa(e: crate::llvm::AotError) -> VsaAotError {
    use crate::llvm::AotError;
    match e {
        AotError::ToolchainMissing(t) => VsaAotError::ToolchainMissing(t),
        AotError::Compile(s) => VsaAotError::Compile(s),
        AotError::Run(s) => VsaAotError::Run(s),
        AotError::Parse(s) => VsaAotError::Parse(s),
        other => VsaAotError::Run(other.to_string()),
    }
}

fn ensure_toolchain() -> Result<(), VsaAotError> {
    for tool in ["llc", "clang"] {
        Command::new(tool)
            .arg("--version")
            .output()
            .map_err(|_| VsaAotError::ToolchainMissing(tool.to_owned()))?;
    }
    Ok(())
}
