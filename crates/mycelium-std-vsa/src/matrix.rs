//! The `std.vsa` guarantee matrix — RFC-0016 §4.5.
//!
//! Encodes, as a data table, the per-`(model, op)` guarantee tags that the
//! **normative RFC-0003 §4 matrix** (as corrected by the r3 §4.1 erratum) assigns.  The table
//! **mirrors** `mycelium_vsa::RFC0003_MATRIX` (M-242) — it does not re-derive the tags.  Tests
//! assert the table against the kernel matrix so divergence is caught mechanically (VR-5/C2).
//!
//! ## Why a separate table here?
//!
//! `mycelium-vsa::RFC0003_MATRIX` encodes the *operation-level* literature tags (the model × op
//! × strength triple) and is the authoritative source.  This table pairs each `(model, op)` tag
//! with the **EXPLAIN-able?** column from `vsa.md §4`.  Fallibility/effects are **not** encoded
//! here — each op documents its explicit error set on its `ops.rs` signature; the [`OpGuarantee`]
//! data model below carries `tag` + `explain_able` only, and the tests assert exactly that.

use mycelium_core::GuaranteeStrength;
use mycelium_vsa::VsaOp;

/// One row of the `std.vsa` guarantee matrix (RFC-0016 §4.5 / vsa.md §4).
///
/// Each row covers one `(model_id, op)` pair.  The guarantee tag is read from the authoritative
/// RFC-0003 §4 matrix (`mycelium_vsa::matrix_tag`), not invented here (C2 / VR-5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpGuarantee {
    /// The VSA model id, e.g. `"MAP-I"`.
    pub model_id: &'static str,
    /// The operation.
    pub op: VsaOp,
    /// The guarantee tag (the RFC-0003 §4 matrix entry for this pair).
    pub tag: GuaranteeStrength,
    /// Is the result EXPLAIN-able (carries an inspectable artifact such as a bound or trace)?
    /// `Exact` ops are `false` — they carry no bound (M-I1); all approximate ops are `true`.
    pub explain_able: bool,
}

/// The `std.vsa` guarantee matrix (vsa.md §4), encoded as data (RFC-0016 §4.5).
///
/// Rows = `(model, op)` pairs.  Tag = the RFC-0003 §4 normative entry for that pair (copied from
/// `mycelium_vsa::RFC0003_MATRIX`; divergence is caught in tests).  `explain_able` = whether
/// the op exposes an inspectable bound/trace artifact (C3).
///
/// Notes matching the spec:
/// - `bind` is `Exact` for MAP-I/MAP-B/BSC/HRR/FHRR (algebraic; §4.1 erratum) and `Proven` for
///   SBC (the Bloom analysis — `matrix.rs` `("SBC", Bind, Proven)`).
/// - `unbind` is `Exact` for MAP-I/MAP-B/BSC (self-inverse); `Empirical` for HRR/FHRR (the
///   residual weak link; §4.1); `Proven` for SBC.
/// - `bundle` is `Proven` for MAP-I/MAP-B/BSC/SBC and `Empirical` for HRR/FHRR.
/// - `permute` is `Exact` for **all** models (§4.1 erratum — a fixed coordinate bijection).
///
/// # FLAG Q3 (SBC bind/bundle `Proven`)
///
/// The SBC `bind` and `bundle` cells are `Proven` in the kernel matrix (`matrix.rs`), reflecting
/// the Bloom / Counting-Bloom analysis cited in RFC-0003 §4/§5.  This table mirrors that cell
/// faithfully.  The exact theorem constants and checked side-conditions are owned by RFC-0003 §5
/// and the encoded matrix; they are **cited here, not restated** (VR-5 / vsa.md §7-Q3).
///
/// # FLAG Q3 (HRR/FHRR `bundle` `Empirical`)
///
/// HRR/FHRR `bundle` is `Empirical` (Gaussian/asymptotic).  The sub-Gaussian phasor upgrade to
/// `Proven` (Thomas) is **not** pre-claimed here — it requires checked instantiation not yet
/// discharged in this crate (VR-5 / vsa.md §7-Q3).
pub const GUARANTEE_MATRIX: &[OpGuarantee] = &[
    // --- MAP-I ---
    OpGuarantee {
        model_id: "MAP-I",
        op: VsaOp::Bind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    OpGuarantee {
        model_id: "MAP-I",
        op: VsaOp::Unbind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    OpGuarantee {
        model_id: "MAP-I",
        op: VsaOp::Bundle,
        tag: GuaranteeStrength::Proven,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "MAP-I",
        op: VsaOp::Permute,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // --- MAP-B ---
    OpGuarantee {
        model_id: "MAP-B",
        op: VsaOp::Bind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    OpGuarantee {
        model_id: "MAP-B",
        op: VsaOp::Unbind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // MAP-B bundle is Proven (membership-only, Clarkson Thm 16) with the RR-13 nesting refusal.
    OpGuarantee {
        model_id: "MAP-B",
        op: VsaOp::Bundle,
        tag: GuaranteeStrength::Proven,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "MAP-B",
        op: VsaOp::Permute,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // --- BSC ---
    OpGuarantee {
        model_id: "BSC",
        op: VsaOp::Bind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    OpGuarantee {
        model_id: "BSC",
        op: VsaOp::Unbind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // BSC bundle is Proven on-expectation (Heim / Yi & Achour — weaker form; still Proven in matrix).
    OpGuarantee {
        model_id: "BSC",
        op: VsaOp::Bundle,
        tag: GuaranteeStrength::Proven,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "BSC",
        op: VsaOp::Permute,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // --- HRR ---
    // HRR bind is Exact (algebraic circular convolution; §4.1 erratum).
    OpGuarantee {
        model_id: "HRR",
        op: VsaOp::Bind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // HRR unbind is Empirical — the residual weak link (§4.1); the approximate inverse is lossy.
    OpGuarantee {
        model_id: "HRR",
        op: VsaOp::Unbind,
        tag: GuaranteeStrength::Empirical,
        explain_able: true,
    },
    // HRR bundle is Empirical (Gaussian/asymptotic). FLAG Q3: sub-Gaussian upgrade not claimed.
    OpGuarantee {
        model_id: "HRR",
        op: VsaOp::Bundle,
        tag: GuaranteeStrength::Empirical,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "HRR",
        op: VsaOp::Permute,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // --- FHRR ---
    OpGuarantee {
        model_id: "FHRR",
        op: VsaOp::Bind,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    OpGuarantee {
        model_id: "FHRR",
        op: VsaOp::Unbind,
        tag: GuaranteeStrength::Empirical,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "FHRR",
        op: VsaOp::Bundle,
        tag: GuaranteeStrength::Empirical,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "FHRR",
        op: VsaOp::Permute,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
    // --- SBC (sparse) ---
    // FLAG Q3: SBC bind/unbind/bundle are Proven per the Bloom analysis (Clarkson Thms 22–23).
    // Exact constants are owned by RFC-0003 §5 + the encoded matrix, cited not restated here.
    OpGuarantee {
        model_id: "SBC",
        op: VsaOp::Bind,
        tag: GuaranteeStrength::Proven,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "SBC",
        op: VsaOp::Unbind,
        tag: GuaranteeStrength::Proven,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "SBC",
        op: VsaOp::Bundle,
        tag: GuaranteeStrength::Proven,
        explain_able: true,
    },
    OpGuarantee {
        model_id: "SBC",
        op: VsaOp::Permute,
        tag: GuaranteeStrength::Exact,
        explain_able: false,
    },
];

/// Look up a row in [`GUARANTEE_MATRIX`] by model id and op.  Returns `None` for an unregistered
/// model — an unregistered model has no honest tag to offer (the same convention as
/// `mycelium_vsa::matrix_tag`).
#[must_use]
pub fn std_matrix_tag(model_id: &str, op: VsaOp) -> Option<GuaranteeStrength> {
    GUARANTEE_MATRIX
        .iter()
        .find(|r| r.model_id == model_id && r.op == op)
        .map(|r| r.tag)
}
