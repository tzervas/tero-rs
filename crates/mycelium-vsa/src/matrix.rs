//! The **RFC-0003 §4 guarantee-tag matrix**, encoded as a single source-of-truth table (M-242).
//!
//! Honest tags per the literature (RFC-0003 §4, normative; research T1.2): every implemented
//! model's [`intrinsic_guarantee`](crate::VsaModel::intrinsic_guarantee) must match this table —
//! asserted model-by-model, op-by-op in `tests/matrix.rs`, so the code and the normative matrix
//! cannot silently diverge (VR-5: the matrix is the per-model × per-op tag ledger).
//!
//! Row bases (RFC-0003 §4):
//! - **MAP-I** — bind/unbind self-inverse `Exact`; bundle `Proven` (Clarkson Thm 6 / Thomas
//!   Thms 2/7); permute `Exact` (T1.2 "permute Exact everywhere"; the §4 sequence-capacity note
//!   — error quadratic in sequence length, Clarkson Thm 9 — is about *retrieval from sequence
//!   encodings*, not the shift itself).
//! - **MAP-B** — bind/unbind self-inverse `Exact`; bundle **membership-only** `Proven`
//!   (Clarkson Thm 16) with deep nesting **forbidden under `Proven`** (reliability decays
//!   `1/2 + 1/2^r` with depth `r` — RR-13; enforced by
//!   [`VsaError::NestedBundleUnsupported`](crate::VsaError)); permute `Exact`.
//! - **BSC** — XOR bind/unbind `Exact`; bundle `Proven` **on expectation** (Heim / Yi & Achour —
//!   weaker than w.p. ≥ 1−δ, tagged accordingly in the docs); circular-shift permute `Exact`.
//! - **HRR / FHRR** — bind algebraic `Exact` (RFC-0003 §4.1 erratum, r3: `bind` is the exact
//!   algebraic op — convolution / complex product — while `unbind` is the lossy one); **unbind the
//!   residual `Empirical` weak link** (not self-inverse, needs cleanup; at most `Empirical`
//!   single-factor); bundle `Empirical` (Gaussian/asymptotic); permute `Exact`.
//! - **SBC (sparse)** — algebraic part `Proven`; bundle `Proven` via the Bloom / Counting-Bloom
//!   analysis (Clarkson Thms 22–23); permute `Exact`.

use mycelium_core::GuaranteeStrength;

use crate::VsaOp;

use GuaranteeStrength::{Empirical, Exact, Proven};
use VsaOp::{Bind, Bundle, Permute, Unbind};

/// The §4 matrix: `(model id, op, normative tag)` for every implemented model × operation.
pub const RFC0003_MATRIX: &[(&str, VsaOp, GuaranteeStrength)] = &[
    ("MAP-I", Bind, Exact),
    ("MAP-I", Unbind, Exact),
    ("MAP-I", Bundle, Proven),
    ("MAP-I", Permute, Exact),
    ("MAP-B", Bind, Exact),
    ("MAP-B", Unbind, Exact),
    ("MAP-B", Bundle, Proven),
    ("MAP-B", Permute, Exact),
    ("BSC", Bind, Exact),
    ("BSC", Unbind, Exact),
    // A3-06/C1-04: this `Proven` is the literature's *operation-level, on-expectation* tag (Heim /
    // Yi & Achour: minimum size to hit a target accuracy *in expectation*) — strictly weaker than a
    // value-level w.p. ≥ 1−δ guarantee, and weaker than MAP-I's tail-bound `Proven` even though the
    // lattice renders both as the same `Proven`. The lattice cannot carry the "on expectation"
    // qualifier, so a matrix consumer must read it here: the BSC bundle's *value* path is correctly
    // `Empirical` (a δ from `BSC_BUNDLE_PROFILE`), and no value carries a w.p.≥1−δ `Proven` (M-I2).
    ("BSC", Bundle, Proven),
    ("BSC", Permute, Exact),
    ("HRR", Bind, Exact),
    ("HRR", Unbind, Empirical),
    ("HRR", Bundle, Empirical),
    ("HRR", Permute, Exact),
    ("FHRR", Bind, Exact),
    ("FHRR", Unbind, Empirical),
    ("FHRR", Bundle, Empirical),
    ("FHRR", Permute, Exact),
    ("SBC", Bind, Proven),
    ("SBC", Unbind, Proven),
    ("SBC", Bundle, Proven),
    ("SBC", Permute, Exact),
];

/// Look up the normative tag for `(model_id, op)`; `None` for a model the matrix does not cover
/// (an unregistered model has no honest tag to offer — never a default).
#[must_use]
pub fn matrix_tag(model_id: &str, op: VsaOp) -> Option<GuaranteeStrength> {
    RFC0003_MATRIX
        .iter()
        .find(|(m, o, _)| *m == model_id && *o == op)
        .map(|(_, _, g)| *g)
}
