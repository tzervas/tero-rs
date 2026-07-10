//! **FHRR** — Fourier Holographic Reduced Representations (frequency-domain / phasor) (M-241;
//! RFC-0003 §4; T1.2).
//!
//! Hypervector components are **phase angles** `θᵢ ∈ (−π, π]` standing for the unit phasors
//! `e^{iθᵢ}` (the natural `Vec<f64>` encoding of unit-magnitude complex components). The algebra:
//! - **bind** = elementwise **phase addition** (complex multiplication of phasors) — algebraic
//!   and deterministic; **unbind** = phase subtraction (conjugate multiplication), the
//!   approximate-inverse role in use (decoding from superpositions needs cleanup) — tagged
//!   **`Empirical`**, the same weak-link assignment as HRR (RFC-0003 §4 / T1.2; never upgraded
//!   even though pure-pair recovery is near-exact, the matrix is normative).
//! - **bundle** = per-component **complex sum renormalized to a phasor** (`arg Σ e^{iθ}`) —
//!   lossy by construction (magnitude is discarded); **`Empirical`** (RFC-0003 §4). A component
//!   whose phasor sum has (near-)zero magnitude has no defined phase — an explicit
//!   [`VsaError::DegenerateBundleComponent`], never an arbitrary pick.
//! - **permute** = cyclic shift — **`Exact`**.
//! - **similarity** = mean `cos(θa − θb)` (the real part of the normalized Hermitian inner
//!   product) in `[-1, 1]`.

use mycelium_core::{operation_hash, GuaranteeStrength, Provenance, Value};

use crate::wrap::{hv_of, rotate, wrap, wrap_exact};
use crate::{EmpiricalProfile, VsaError, VsaModel, VsaOp};

/// The trial-validated regime backing the Value-level FHRR unbind's `Empirical` δ
/// (`tests/empirical_profiles.rs` runs exactly these trials).
pub const FHRR_UNBIND_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 1,
    odd_items_only: false,
    min_dim: 256,
    delta: 1e-2,
    trials: 10_000,
    method: "Monte-Carlo bind→unbind→cleanup recovery (uniform phasor atoms, single bind factor, \
             codebook ≤ 16, d ≥ 256)",
};

/// Wrap an angle to `(−π, π]`.
pub(crate) fn wrap_phase(theta: f64) -> f64 {
    let t = theta.rem_euclid(std::f64::consts::TAU);
    if t > std::f64::consts::PI {
        t - std::f64::consts::TAU
    } else {
        t
    }
}

/// The FHRR model at a fixed dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fhrr {
    /// Hypervector dimensionality (number of phasor components).
    pub dim: u32,
}

impl Fhrr {
    /// An FHRR model of dimension `dim`.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Fhrr { dim }
    }

    fn check_len(&self, v: &[f64]) -> Result<(), VsaError> {
        if v.len() == self.dim as usize {
            Ok(())
        } else {
            Err(VsaError::DimMismatch {
                expected: self.dim as usize,
                got: v.len(),
            })
        }
    }

    /// Components must be phases in `(−π, π]` — anything else is outside the phasor alphabet
    /// (refused, never silently wrapped: an out-of-range payload contradicts its encoding).
    fn check_phases(v: &[f64]) -> Result<(), VsaError> {
        match v
            .iter()
            .position(|&t| !t.is_finite() || t <= -std::f64::consts::PI || t > std::f64::consts::PI)
        {
            Some(index) => Err(VsaError::NonAlphabetComponent { index }),
            None => Ok(()),
        }
    }

    /// Value-level `bind` (deterministic phasor algebra).
    pub fn bind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let out = self.bind(
            hv_of(self.model_id(), self.dim, a)?,
            hv_of(self.model_id(), self.dim, b)?,
        )?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.fhrr.bind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level **`Empirical` unbind** (the RFC-0003 §4 weak-link tag, like HRR): the decoded
    /// vector carries the [`FHRR_UNBIND_PROFILE`] δ and is completed through a
    /// [`CleanupMemory`](crate::CleanupMemory). Gated to the validated regime: `a` must be a
    /// single `vsa.fhrr.bind` product.
    pub fn unbind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        FHRR_UNBIND_PROFILE.check(1, self.dim)?;
        match a.meta().provenance() {
            Provenance::Derived { op, .. } if op == &operation_hash("vsa.fhrr.bind") => {}
            _ => {
                return Err(VsaError::OutsideEmpiricalProfile {
                    detail: "input is not a single vsa.fhrr.bind product (the validated \
                             single-factor regime)"
                        .to_owned(),
                })
            }
        }
        let out = self.unbind(
            hv_of(self.model_id(), self.dim, a)?,
            hv_of(self.model_id(), self.dim, b)?,
        )?;
        wrap(
            self.model_id(),
            self.dim,
            out,
            "vsa.fhrr.unbind",
            vec![a.content_hash(), b.content_hash()],
            GuaranteeStrength::Empirical,
            Some(FHRR_UNBIND_PROFILE.bound()),
        )
    }

    /// Value-level `permute` (Exact): cyclic shift by `shift` (M-892 — completes the FHRR
    /// Value-level bind group so the `vsa.permute` prim dispatches uniformly across the model
    /// set). A pure component rotation — no phase arithmetic occurs, so the `Exact` tag needs no
    /// alphabet guard beyond the model/dim check (the same posture as
    /// [`Bsc::permute_value`](crate::Bsc::permute_value) and `MapI::permute_value`).
    pub fn permute_value(&self, a: &Value, shift: i64) -> Result<Value, VsaError> {
        let out = self.permute(hv_of(self.model_id(), self.dim, a)?, shift)?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.fhrr.permute",
            vec![a.content_hash()],
        )
    }
}

impl VsaModel for Fhrr {
    fn model_id(&self) -> &'static str {
        "FHRR"
    }

    fn self_inverse(&self) -> bool {
        false
    }

    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength {
        match op {
            // Algebraic, deterministic phasor ops.
            VsaOp::Bind | VsaOp::Permute => GuaranteeStrength::Exact,
            // The weak-link assignment (RFC-0003 §4 / T1.2) — normative, never upgraded.
            VsaOp::Unbind | VsaOp::Bundle => GuaranteeStrength::Empirical,
        }
    }

    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Self::check_phases(a)?;
        Self::check_phases(b)?;
        Ok(a.iter().zip(b).map(|(x, y)| wrap_phase(x + y)).collect())
    }

    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Self::check_phases(a)?;
        Self::check_phases(b)?;
        Ok(a.iter().zip(b).map(|(x, y)| wrap_phase(x - y)).collect())
    }

    fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
        if items.is_empty() {
            return Err(VsaError::EmptyBundle);
        }
        for item in items {
            self.check_len(item)?;
            Self::check_phases(item)?;
        }
        let mut out = Vec::with_capacity(self.dim as usize);
        for index in 0..self.dim as usize {
            let re: f64 = items.iter().map(|v| v[index].cos()).sum();
            let im: f64 = items.iter().map(|v| v[index].sin()).sum();
            // A vanished phasor sum has no phase: explicit, never an arbitrary pick (G2).
            if (re * re + im * im).sqrt() < 1e-9 {
                return Err(VsaError::DegenerateBundleComponent { index });
            }
            out.push(wrap_phase(im.atan2(re)));
        }
        Ok(out)
    }

    fn permute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        Ok(rotate(a, shift))
    }

    fn unpermute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
        self.permute(a, -shift)
    }

    /// Mean `cos(θa − θb)` — the real part of the normalized Hermitian inner product of the
    /// phasor vectors, in `[-1, 1]`.
    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        a.iter().zip(b).map(|(x, y)| (x - y).cos()).sum::<f64>() / a.len() as f64
    }
}
