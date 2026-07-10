//! **HRR** — Holographic Reduced Representations (circular convolution) (M-241; RFC-0003 §4;
//! T1.2).
//!
//! Hypervectors are real vectors (atoms drawn ~`N(0, 1/d)` in practice). The algebra:
//! - **bind** = **circular convolution** — algebraic and deterministic (the issue's "bind
//!   (convolution) algebraic"); binding is *not* self-inverse.
//! - **unbind** = **circular correlation** (convolution with the involution `b~[i] = b[−i mod d]`)
//!   — an **approximate** inverse, lossy, needing cleanup: **the residual `Empirical` weak link**
//!   (RFC-0003 §4 / T1.2). It is never tagged above `Empirical`, and the Value-level form carries
//!   a trial-validated δ ([`HRR_UNBIND_PROFILE`]).
//! - **permute** = cyclic shift — **`Exact`**; **bundle** = elementwise sum — **`Empirical`**
//!   (Gaussian-style capacity only; RFC-0003 §4).
//! - **similarity** = cosine.
//!
//! # The Value-level unbind regime
//! [`Hrr::unbind_values`] is gated to exactly the regime its profile's trials validate: the input
//! must be a **single `vsa.hrr.bind` product** (checked via provenance — the structural witness of
//! the "single-factor" side-condition T1.2 scopes the Empirical tag to) at the profile's minimum
//! dimension or above. Unbinding from bundles or unknown-provenance vectors stays available as
//! *algebra* (measurement world, no tag is issued); recovering multi-factor products needs the
//! Phase-3 resonator (RFC-0003 §6) and is out of scope here.

use mycelium_core::{operation_hash, GuaranteeStrength, Provenance, Value};

use crate::wrap::{cosine, hv_of, rotate, wrap, wrap_exact};
use crate::{EmpiricalProfile, VsaError, VsaModel, VsaOp};

/// The trial-validated regime backing the Value-level HRR unbind's `Empirical` δ
/// (`tests/empirical_profiles.rs` runs exactly these trials).
///
/// **Honest limitations of this profile (A3-10):**
/// - `trials = 2_000` is **thinner** than the other profiles' `10_000` at the same `δ = 1e-2`, so
///   this `Empirical` δ is estimated at a *coarser resolution* (a 1e-2 tail estimated from 2k
///   trials has wide confidence; 10k would tighten it). The count is held here deliberately — it is
///   an empirical-evidence decision, not a build detail, so it is documented rather than silently
///   bumped (do not raise it without re-running the trials).
/// - The **codebook-size side-condition** (`codebook ≤ 16`) is recorded **only in the free-text
///   `method` string**, which [`EmpiricalProfile::check`] cannot inspect — so `check` gates on item
///   count / parity / dimension but *not* on codebook size. A caller cleaning up against a larger
///   codebook is outside the validated regime without an explicit refusal; the constraint is honest
///   documentation, not an enforced guard.
pub const HRR_UNBIND_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 1,
    odd_items_only: false,
    min_dim: 256,
    delta: 1e-2,
    trials: 2_000,
    method: "Monte-Carlo bind→unbind→cleanup recovery (N(0,1/d) atoms, single bind factor, \
             codebook ≤ 16, d ≥ 256)",
};

/// The HRR model at a fixed dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hrr {
    /// Hypervector dimensionality.
    pub dim: u32,
}

impl Hrr {
    /// An HRR model of dimension `dim`.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Hrr { dim }
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

    /// Circular convolution `(a ⊛ b)[k] = Σᵢ a[i]·b[(k−i) mod d]` (naive `O(d²)` — the trusted
    /// reference; an FFT path is a perf-path concern, not a semantics one).
    fn cconv(a: &[f64], b: &[f64]) -> Vec<f64> {
        let d = a.len();
        let mut out = vec![0.0; d];
        for (k, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, &ai) in a.iter().enumerate() {
                acc += ai * b[(k + d - i) % d];
            }
            *o = acc;
        }
        out
    }

    /// The involution `b~[i] = b[(−i) mod d]`, turning convolution into correlation.
    fn involution(b: &[f64]) -> Vec<f64> {
        let d = b.len();
        (0..d).map(|i| b[(d - i) % d]).collect()
    }

    /// Value-level `bind` (deterministic algebra; binding is where HRR is honest — the
    /// approximation lives in `unbind`).
    pub fn bind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let out = self.bind(
            hv_of(self.model_id(), self.dim, a)?,
            hv_of(self.model_id(), self.dim, b)?,
        )?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.hrr.bind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level **`Empirical` unbind** — the documented weak link (RFC-0003 §4). The result is
    /// the *noisy* correlation decode tagged with the [`HRR_UNBIND_PROFILE`] δ (M-I3); recovery is
    /// completed by routing it through a [`CleanupMemory`](crate::CleanupMemory), whose
    /// confidence/margin keep the retrieval inspectable (FR-S4/G2). Gated to the validated
    /// regime: `a` must be a single `vsa.hrr.bind` product (module docs).
    pub fn unbind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        HRR_UNBIND_PROFILE.check(1, self.dim)?;
        match a.meta().provenance() {
            Provenance::Derived { op, .. } if op == &operation_hash("vsa.hrr.bind") => {}
            _ => {
                return Err(VsaError::OutsideEmpiricalProfile {
                    detail: "input is not a single vsa.hrr.bind product (the validated \
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
            "vsa.hrr.unbind",
            vec![a.content_hash(), b.content_hash()],
            GuaranteeStrength::Empirical,
            Some(HRR_UNBIND_PROFILE.bound()),
        )
    }
}

impl VsaModel for Hrr {
    fn model_id(&self) -> &'static str {
        "HRR"
    }

    fn self_inverse(&self) -> bool {
        false
    }

    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength {
        match op {
            // Algebraic, deterministic (issue #61: "bind (convolution) algebraic").
            VsaOp::Bind | VsaOp::Permute => GuaranteeStrength::Exact,
            // The residual weak link: approximate inverse, needs cleanup — at most Empirical
            // (RFC-0003 §4 / T1.2); never upgraded.
            VsaOp::Unbind => GuaranteeStrength::Empirical,
            // Gaussian/asymptotic capacity only (RFC-0003 §4).
            VsaOp::Bundle => GuaranteeStrength::Empirical,
        }
    }

    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Ok(Self::cconv(a, b))
    }

    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        // Circular correlation: convolve with the involution — the approximate inverse.
        Ok(Self::cconv(a, &Self::involution(b)))
    }

    fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
        let Some((first, rest)) = items.split_first() else {
            return Err(VsaError::EmptyBundle);
        };
        self.check_len(first)?;
        let mut acc = first.to_vec();
        for item in rest {
            self.check_len(item)?;
            for (a, x) in acc.iter_mut().zip(item.iter()) {
                *a += x;
            }
        }
        Ok(acc)
    }

    fn permute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        Ok(rotate(a, shift))
    }

    fn unpermute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
        self.permute(a, -shift)
    }

    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        cosine(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CleanupMemory;
    use mycelium_core::{Meta, Payload, Repr, SparsityClass};

    /// Deterministic ~N(0, 1/d) atom (Box–Muller over a tiny LCG — house style).
    fn gaussian_atom(dim: u32, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut unif = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
        };
        let scale = 1.0 / f64::from(dim).sqrt();
        (0..dim)
            .map(|_| {
                let (u1, u2) = (unif(), unif());
                scale * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect()
    }

    fn hv_value(dim: u32, seed: u64) -> Value {
        Value::new(
            Repr::Vsa {
                model: "HRR".to_owned(),
                dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(gaussian_atom(dim, seed)),
            Meta::exact(mycelium_core::Provenance::Root),
        )
        .unwrap()
    }

    const D: u32 = 256;

    #[test]
    fn bind_is_not_self_inverse_and_unbind_is_approximate() {
        let m = Hrr::new(D);
        assert!(!m.self_inverse());
        let a = gaussian_atom(D, 1);
        let b = gaussian_atom(D, 2);
        let bound = m.bind(&a, &b).unwrap();
        let recovered = m.unbind(&bound, &b).unwrap();
        // Approximate: clearly similar to a, but not equal (the weak link).
        assert_ne!(recovered, a);
        let sim = m.similarity(&recovered, &a);
        assert!(sim > 0.5, "decode similarity {sim} should be high");
        assert_eq!(
            m.intrinsic_guarantee(VsaOp::Unbind),
            GuaranteeStrength::Empirical
        );
    }

    #[test]
    fn unbind_then_cleanup_recovers_the_filler() {
        let m = Hrr::new(D);
        let role = gaussian_atom(D, 10);
        let filler = gaussian_atom(D, 20);
        let bound = m.bind(&role, &filler).unwrap();
        let mut mem = CleanupMemory::new(D);
        mem.insert("filler", filler.clone()).unwrap();
        mem.insert("other1", gaussian_atom(D, 21)).unwrap();
        mem.insert("other2", gaussian_atom(D, 22)).unwrap();
        let noisy = m.unbind(&bound, &role).unwrap();
        let hit = mem.cleanup(&noisy, &m).unwrap();
        assert_eq!(hit.label, "filler");
        assert!(hit.margin > 0.1, "margin {}", hit.margin);
    }

    #[test]
    fn value_unbind_is_empirical_and_regime_gated() {
        let m = Hrr::new(D);
        let a = hv_value(D, 1);
        let b = hv_value(D, 2);
        let bound = m.bind_values(&a, &b).unwrap();
        assert_eq!(bound.meta().guarantee(), GuaranteeStrength::Exact);
        let noisy = m.unbind_values(&bound, &b).unwrap();
        assert_eq!(noisy.meta().guarantee(), GuaranteeStrength::Empirical);
        assert!(matches!(
            noisy.meta().bound().map(|x| &x.basis),
            Some(mycelium_core::BoundBasis::EmpiricalFit { trials, .. })
                if *trials == HRR_UNBIND_PROFILE.trials
        ));
        // A Root-provenance vector is outside the validated single-factor regime — explicit.
        assert!(matches!(
            m.unbind_values(&a, &b),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
        // Below the profile dimension — explicit.
        let small = Hrr::new(64);
        let sa = hv_value(64, 3);
        let sb = hv_value(64, 4);
        let sbound = small.bind_values(&sa, &sb).unwrap();
        assert!(matches!(
            small.unbind_values(&sbound, &sb),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
    }

    #[test]
    fn permute_round_trips() {
        let m = Hrr::new(D);
        let a = gaussian_atom(D, 5);
        let p = m.permute(&a, 9).unwrap();
        assert_eq!(m.unpermute(&p, 9).unwrap(), a);
    }
}
