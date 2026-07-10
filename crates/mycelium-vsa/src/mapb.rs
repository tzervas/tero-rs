//! **MAP-B** — Multiply-Add-Permute, bipolar with **sign-rounded bundling** (M-240; RFC-0003 §4).
//!
//! Hypervectors are bipolar `±1` vectors. The algebra:
//! - **bind** = elementwise product; **self-inverse** (`x·x = 1`) — **`Exact`** (algebraic), so
//!   `unbind` is the same op.
//! - **permute** = cyclic shift — **`Exact`**.
//! - **bundle** = **sign-rounded** elementwise sum (the result stays bipolar; ties on an even
//!   operand count copy the first operand's component — deterministic, documented). The matrix
//!   tag is **`Proven` membership-only** (Clarkson Thm 16), but its reliability decays
//!   `1/2 + 1/2^r` with nesting depth `r` — deep nesting is **forbidden under `Proven`**
//!   (RFC-0003 §4; RR-13).
//! - **similarity** = cosine.
//!
//! # Honesty of the Value-level bundle
//! The corpus carries a *checked-instantiation formula* (`requiredDim`) only for MAP-I's additive
//! bundle (M-131); no MAP-B analogue is ratified, so a `Proven` **value** cannot be issued here
//! (M-I2/VR-5 — the matrix's `Proven` is a literature tag about the operation). The Value-level
//! [`MapB::bundle_values_empirical`] therefore carries an **`Empirical`** δ from the
//! trial-validated [`MAPB_BUNDLE_PROFILE`] (M-I3), refusing explicitly outside the profile's
//! side-conditions — including any input that is itself a MAP-B bundle (**RR-13**: depth > 1 is
//! [`VsaError::NestedBundleUnsupported`], never a silent accuracy loss; M-242).

use mycelium_core::{operation_hash, GuaranteeStrength, Provenance, Value};

use crate::wrap::{cosine, hv_of, rotate, wrap, wrap_exact};
use crate::{EmpiricalProfile, VsaError, VsaModel, VsaOp};

/// The op-hash name of the MAP-B bundle — also the marker the RR-13 nesting check looks for in
/// an input's provenance.
const BUNDLE_OP: &str = "vsa.map_b.bundle";

/// The trial-validated regime backing the Value-level MAP-B bundle's `Empirical` δ
/// (`tests/empirical_profiles.rs` runs exactly these trials).
pub const MAPB_BUNDLE_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 5,
    odd_items_only: true,
    min_dim: 1024,
    delta: 1e-2,
    trials: 10_000,
    method: "Monte-Carlo sign-bundle membership decode (bipolar atoms, odd m ≤ 5, d ≥ 1024, \
             depth 1)",
};

/// The MAP-B model at a fixed dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapB {
    /// Hypervector dimensionality.
    pub dim: u32,
}

impl MapB {
    /// A MAP-B model of dimension `dim`.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        MapB { dim }
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

    /// Every component must be `±1` — MAP-B's algebra is defined on the bipolar alphabet only.
    fn check_bipolar(v: &[f64]) -> Result<(), VsaError> {
        match v.iter().position(|&x| x != 1.0 && x != -1.0) {
            Some(index) => Err(VsaError::NonAlphabetComponent { index }),
            None => Ok(()),
        }
    }

    /// Value-level `bind` (Exact).
    ///
    /// The `Exact` tag rests on the bipolar self-inverse identity (`x·x = 1`), which holds **only**
    /// on the `±1` alphabet. Non-bipolar components are refused with
    /// [`VsaError::NonAlphabetComponent`] rather than stamped `Exact` on a wrong result
    /// (A3-04; M-I2/VR-5; G2).
    pub fn bind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let av = hv_of(self.model_id(), self.dim, a)?;
        let bv = hv_of(self.model_id(), self.dim, b)?;
        Self::check_bipolar(av)?;
        Self::check_bipolar(bv)?;
        let out = self.bind(av, bv)?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.map_b.bind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level `unbind` (Exact; self-inverse).
    ///
    /// As with [`bind_values`](Self::bind_values), the `Exact` self-inverse identity holds only on
    /// the `±1` alphabet; non-bipolar components are refused, never stamped `Exact` (A3-04).
    pub fn unbind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let av = hv_of(self.model_id(), self.dim, a)?;
        let bv = hv_of(self.model_id(), self.dim, b)?;
        Self::check_bipolar(av)?;
        Self::check_bipolar(bv)?;
        let out = self.unbind(av, bv)?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.map_b.unbind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level `permute` (Exact).
    pub fn permute_value(&self, a: &Value, shift: i64) -> Result<Value, VsaError> {
        let out = self.permute(hv_of(self.model_id(), self.dim, a)?, shift)?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.map_b.permute",
            vec![a.content_hash()],
        )
    }

    /// Value-level **`Empirical` bundle**: sign-rounded superposition carrying the
    /// [`MAPB_BUNDLE_PROFILE`] δ (see module docs for why this is `Empirical`, not `Proven`).
    /// Refusals are explicit: outside the profile (item count, parity, dimension, non-bipolar
    /// inputs) and **nested bundles** (RR-13 — an input whose provenance op is itself the MAP-B
    /// bundle).
    pub fn bundle_values_empirical(&self, items: &[&Value]) -> Result<Value, VsaError> {
        MAPB_BUNDLE_PROFILE.check(items.len(), self.dim)?;
        for v in items {
            if let Provenance::Derived { op, .. } = v.meta().provenance() {
                if op == &operation_hash(BUNDLE_OP) {
                    return Err(VsaError::NestedBundleUnsupported { model: "MAP-B" });
                }
            }
        }
        let hvs: Vec<&[f64]> = items
            .iter()
            .map(|v| hv_of(self.model_id(), self.dim, v))
            .collect::<Result<_, _>>()?;
        for hv in &hvs {
            Self::check_bipolar(hv)?;
        }
        let data = self.bundle(&hvs)?;
        let inputs = items.iter().map(|v| v.content_hash()).collect();
        wrap(
            self.model_id(),
            self.dim,
            data,
            BUNDLE_OP,
            inputs,
            GuaranteeStrength::Empirical,
            Some(MAPB_BUNDLE_PROFILE.bound()),
        )
    }
}

impl VsaModel for MapB {
    fn model_id(&self) -> &'static str {
        "MAP-B"
    }

    fn self_inverse(&self) -> bool {
        true
    }

    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength {
        match op {
            // Algebraic, lossless (RFC-0003 §4).
            VsaOp::Bind | VsaOp::Unbind | VsaOp::Permute => GuaranteeStrength::Exact,
            // Membership-only Proven (Clarkson Thm 16) — at depth 1; deeper nesting is forbidden
            // under Proven (RR-13). The Value-level bundle is Empirical (module docs).
            VsaOp::Bundle => GuaranteeStrength::Proven,
        }
    }

    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
    }

    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        // Self-inverse on bipolar atoms: unbind == bind.
        self.bind(a, b)
    }

    fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
        let Some((first, rest)) = items.split_first() else {
            return Err(VsaError::EmptyBundle);
        };
        self.check_len(first)?;
        let mut sums = first.to_vec();
        for item in rest {
            self.check_len(item)?;
            for (s, x) in sums.iter_mut().zip(item.iter()) {
                *s += x;
            }
        }
        // Sign-round; a tie (sum 0, even operand count) copies the first operand's component.
        Ok(sums
            .iter()
            .zip(first.iter())
            .map(|(&s, &f)| {
                if s > 0.0 {
                    1.0
                } else if s < 0.0 {
                    -1.0
                } else {
                    f
                }
            })
            .collect())
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
    use mycelium_core::{Meta, Payload, Repr, SparsityClass};

    /// Deterministic bipolar atom (tiny LCG — house style, no rand dependency).
    fn bipolar(dim: u32, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if (s >> 63) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    fn hv_value(dim: u32, seed: u64) -> Value {
        Value::new(
            Repr::Vsa {
                model: "MAP-B".to_owned(),
                dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(bipolar(dim, seed)),
            Meta::exact(mycelium_core::Provenance::Root),
        )
        .unwrap()
    }

    const D: u32 = 1024;

    #[test]
    fn bind_is_self_inverse_exact() {
        let m = MapB::new(D);
        assert!(m.self_inverse());
        let a = bipolar(D, 1);
        let b = bipolar(D, 2);
        let bound = m.bind(&a, &b).unwrap();
        assert_eq!(m.unbind(&bound, &b).unwrap(), a);
        assert_eq!(m.intrinsic_guarantee(VsaOp::Bind), GuaranteeStrength::Exact);
    }

    #[test]
    fn bundle_is_bipolar_and_close_to_members() {
        let m = MapB::new(D);
        let items: Vec<Vec<f64>> = (0..3).map(|i| bipolar(D, 50 + i)).collect();
        let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
        let bundle = m.bundle(&refs).unwrap();
        assert!(
            bundle.iter().all(|&x| x == 1.0 || x == -1.0),
            "sign-rounded"
        );
        let member = m.similarity(&bundle, &items[0]);
        let stranger = m.similarity(&bundle, &bipolar(D, 999));
        assert!(
            member > stranger + 0.2,
            "member {member} vs stranger {stranger}"
        );
    }

    #[test]
    fn even_count_ties_copy_the_first_operand() {
        let m = MapB::new(4);
        let a = vec![1.0, -1.0, 1.0, -1.0];
        let b = vec![-1.0, 1.0, -1.0, 1.0]; // every position ties
        assert_eq!(m.bundle(&[&a, &b]).unwrap(), a);
    }

    #[test]
    fn value_bind_unbind_refuse_non_bipolar_a3_04() {
        // A3-04 regression: bind/unbind_values stamp Exact on the bipolar self-inverse identity,
        // which holds only on the ±1 alphabet. A non-bipolar component must be refused, never
        // mis-tagged Exact. Mutant-witness: removing the check_bipolar guards in
        // bind_values/unbind_values makes these return an (Exact-tagged) Value.
        let m = MapB::new(D);
        let mut data = bipolar(D, 3);
        data[7] = 0.5;
        let bad = Value::new(
            Repr::Vsa {
                model: "MAP-B".to_owned(),
                dim: D,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(data),
            Meta::exact(mycelium_core::Provenance::Root),
        )
        .unwrap();
        let ok = hv_value(D, 4);
        assert_eq!(
            m.bind_values(&bad, &ok),
            Err(VsaError::NonAlphabetComponent { index: 7 })
        );
        assert_eq!(
            m.unbind_values(&ok, &bad),
            Err(VsaError::NonAlphabetComponent { index: 7 })
        );
        // A bipolar bind still returns an unchanged Exact result.
        let good = m.bind_values(&ok, &hv_value(D, 5)).unwrap();
        assert_eq!(good.meta().guarantee(), GuaranteeStrength::Exact);
    }

    #[test]
    fn value_bundle_is_empirical_within_the_profile() {
        let m = MapB::new(D);
        let vals: Vec<Value> = (0..3).map(|i| hv_value(D, 10 + i)).collect();
        let refs: Vec<&Value> = vals.iter().collect();
        let bundle = m.bundle_values_empirical(&refs).unwrap();
        assert_eq!(bundle.meta().guarantee(), GuaranteeStrength::Empirical);
        match bundle.meta().bound() {
            Some(b) => {
                assert!(matches!(
                    b.kind,
                    mycelium_core::BoundKind::Probability { delta } if delta == MAPB_BUNDLE_PROFILE.delta
                ));
                assert!(matches!(
                    b.basis,
                    mycelium_core::BoundBasis::EmpiricalFit { trials, .. }
                        if trials == MAPB_BUNDLE_PROFILE.trials
                ));
            }
            None => panic!("an Empirical bundle must carry its bound (M-I1)"),
        }
    }

    #[test]
    fn outside_profile_is_explicit() {
        let m = MapB::new(D);
        let vals: Vec<Value> = (0..4).map(|i| hv_value(D, 20 + i)).collect();
        // Even item count: outside the validated profile.
        let refs: Vec<&Value> = vals.iter().collect();
        assert!(matches!(
            m.bundle_values_empirical(&refs),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
        // Undersized dimension.
        let small = MapB::new(64);
        let small_vals: Vec<Value> = (0..3).map(|i| hv_value(64, i)).collect();
        let small_refs: Vec<&Value> = small_vals.iter().collect();
        assert!(matches!(
            small.bundle_values_empirical(&small_refs),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
    }

    #[test]
    fn nested_bundle_is_refused_rr13() {
        let m = MapB::new(D);
        let vals: Vec<Value> = (0..3).map(|i| hv_value(D, 30 + i)).collect();
        let refs: Vec<&Value> = vals.iter().collect();
        let depth1 = m.bundle_values_empirical(&refs).unwrap();
        // Bundling a bundle (depth 2) is an explicit refusal, never a silent accuracy loss.
        let nested = [&depth1, &vals[0], &vals[1]];
        assert_eq!(
            m.bundle_values_empirical(&nested),
            Err(VsaError::NestedBundleUnsupported { model: "MAP-B" })
        );
    }

    #[test]
    fn non_bipolar_inputs_are_refused() {
        let m = MapB::new(D);
        let mut data = bipolar(D, 7);
        data[2] = 0.5;
        let bad = Value::new(
            Repr::Vsa {
                model: "MAP-B".to_owned(),
                dim: D,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(data),
            Meta::exact(mycelium_core::Provenance::Root),
        )
        .unwrap();
        let ok1 = hv_value(D, 8);
        let ok2 = hv_value(D, 9);
        assert_eq!(
            m.bundle_values_empirical(&[&bad, &ok1, &ok2]),
            Err(VsaError::NonAlphabetComponent { index: 2 })
        );
    }
}
