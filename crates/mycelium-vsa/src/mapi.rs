//! **MAP-I** — Multiply-Add-Permute (integer/bipolar) (RFC-0003 §4, T2.6).
//!
//! Hypervectors are real vectors (bipolar `±1` atoms in practice). The algebra:
//! - **bind** = elementwise product; **self-inverse** on bipolar atoms (`x·x = 1`), so `unbind` is
//!   the same op — both **`Exact`** (algebraic).
//! - **permute** = cyclic shift; **`Exact`**, inverted by the opposite shift.
//! - **bundle** = elementwise sum (integer superposition); the retrieval capacity bound is
//!   **`Proven`** (Clarkson/Thomas) but is derived + validated in **M-131**, not here.
//! - **similarity** = cosine.

use mycelium_core::{
    operation_hash, ContentHash, GuaranteeStrength, Meta, Payload, Provenance, Repr, SparsityClass,
    Value,
};

use crate::{capacity, VsaError, VsaModel, VsaOp};

/// The MAP-I model at a fixed dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapI {
    /// Hypervector dimensionality.
    pub dim: u32,
}

impl MapI {
    /// A MAP-I model of dimension `dim`.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        MapI { dim }
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

    /// Extract the hypervector data from a MAP-I `Value` (checking model id + dim).
    fn hv_of<'a>(&self, v: &'a Value) -> Result<&'a [f64], VsaError> {
        match (v.repr(), v.payload()) {
            (Repr::Vsa { model, dim, .. }, Payload::Hypervector(h))
                if model == self.model_id() && *dim == self.dim =>
            {
                Ok(h)
            }
            _ => Err(VsaError::NotThisModel {
                expected: self.model_id(),
            }),
        }
    }

    /// Wrap a result vector into an `Exact` MAP-I `Value` with honest `Derived` provenance.
    fn wrap_exact(
        &self,
        data: Vec<f64>,
        op: &str,
        inputs: Vec<ContentHash>,
    ) -> Result<Value, VsaError> {
        let meta = Meta::new(
            Provenance::Derived {
                op: operation_hash(op),
                inputs,
            },
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .map_err(VsaError::Wf)?;
        Value::new(
            Repr::Vsa {
                model: self.model_id().to_owned(),
                dim: self.dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(data),
            meta,
        )
        .map_err(VsaError::Wf)
    }

    /// Value-level `bind` (Exact): `bind(a, b)` with `Derived` provenance over both inputs.
    ///
    /// The `Exact` tag rests on the bipolar self-inverse identity (`x·x = 1`), which holds **only**
    /// on the `±1` alphabet. A non-bipolar component would make the stamped `Exact` a wrong result,
    /// so we guard both operands and refuse with [`VsaError::NonAlphabetComponent`] rather than
    /// mis-tag (A3-04; M-I2/VR-5; G2).
    pub fn bind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let (av, bv) = (self.hv_of(a)?, self.hv_of(b)?);
        Self::check_bipolar(av)?;
        Self::check_bipolar(bv)?;
        let out = self.bind(av, bv)?;
        self.wrap_exact(
            out,
            "vsa.map_i.bind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level `unbind` (Exact): recover a factor (self-inverse for MAP-I).
    ///
    /// As with [`bind_values`](Self::bind_values), the `Exact` self-inverse identity holds only on
    /// the `±1` alphabet; non-bipolar components are refused, never stamped `Exact` (A3-04).
    pub fn unbind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let (av, bv) = (self.hv_of(a)?, self.hv_of(b)?);
        Self::check_bipolar(av)?;
        Self::check_bipolar(bv)?;
        let out = self.unbind(av, bv)?;
        self.wrap_exact(
            out,
            "vsa.map_i.unbind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level `permute` (Exact): cyclic shift by `shift`.
    pub fn permute_value(&self, a: &Value, shift: i64) -> Result<Value, VsaError> {
        let out = self.permute(self.hv_of(a)?, shift)?;
        self.wrap_exact(out, "vsa.map_i.permute", vec![a.content_hash()])
    }

    /// Value-level **certified `bundle`** (M-131): superpose `items` and attach a **`Proven`**
    /// `CapacityBound` targeting failure probability `delta` — but **only** when the checked
    /// side-condition `dim ≥ requiredDim(m, δ)` holds (the M-001 checked-instantiation pattern,
    /// citing Clarkson/Thomas). If the dimension is insufficient the theorem does not apply and we
    /// return [`VsaError::InsufficientCapacity`] rather than stamping an unbacked `Proven` tag
    /// (M-I2/VR-5). The result `Value` is `Proven` with the bound and `Derived` provenance over all
    /// inputs.
    pub fn bundle_values_certified(&self, items: &[&Value], delta: f64) -> Result<Value, VsaError> {
        if items.is_empty() {
            return Err(VsaError::EmptyBundle);
        }
        // Check the cited theorem's CHECKABLE side-conditions before issuing a Proven bound
        // (A3-03/H6; M-I2/VR-5): the dimension instantiation alone is not enough — the
        // Clarkson/Thomas capacity bound also assumes (i) bipolar (±1) atoms and (ii) distinct
        // items. Without these the Proven tag is unbacked, so we refuse rather than stamp it.
        let hvs: Vec<&[f64]> = items
            .iter()
            .map(|v| self.hv_of(v))
            .collect::<Result<_, _>>()?;
        for hv in &hvs {
            Self::check_bipolar(hv)?;
        }
        let inputs: Vec<ContentHash> = items.iter().map(|v| v.content_hash()).collect();
        if let Some(index) = first_duplicate(&inputs) {
            return Err(VsaError::DuplicateBundleItems { index });
        }
        let m = items.len() as u64;
        let dim = u64::from(self.dim);
        let bound = capacity::proven_capacity_bound(m, dim, delta).ok_or_else(|| {
            VsaError::InsufficientCapacity {
                items: m,
                dim,
                required: capacity::required_dim(m, delta, capacity::MARGIN_MU),
            }
        })?;
        let data = self.bundle(&hvs)?;
        let meta = Meta::new(
            Provenance::Derived {
                op: operation_hash("vsa.map_i.bundle"),
                inputs,
            },
            GuaranteeStrength::Proven,
            Some(bound),
            None,
            None,
            None,
        )
        .map_err(VsaError::Wf)?;
        Value::new(
            Repr::Vsa {
                model: self.model_id().to_owned(),
                dim: self.dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(data),
            meta,
        )
        .map_err(VsaError::Wf)
    }

    /// Every component must be `±1` — the MAP-I capacity theorem assumes bipolar atoms (A3-03/H6).
    fn check_bipolar(v: &[f64]) -> Result<(), VsaError> {
        match v.iter().position(|&x| x != 1.0 && x != -1.0) {
            Some(index) => Err(VsaError::NonAlphabetComponent { index }),
            None => Ok(()),
        }
    }
}

/// The index of the first content hash that repeats an earlier one, if any. Item counts are small,
/// so the quadratic scan is fine and keeps the order deterministic (first repeat reported).
fn first_duplicate(hashes: &[ContentHash]) -> Option<usize> {
    (0..hashes.len()).find(|&i| hashes[..i].contains(&hashes[i]))
}

impl VsaModel for MapI {
    fn model_id(&self) -> &'static str {
        "MAP-I"
    }

    fn self_inverse(&self) -> bool {
        true
    }

    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength {
        match op {
            // Algebraic, lossless (RFC-0003 §4).
            VsaOp::Bind | VsaOp::Unbind | VsaOp::Permute => GuaranteeStrength::Exact,
            // Capacity-bounded superposition: Proven per the cited theorem — but the *value*-level
            // Proven bound (M-I2) is derived and validated in M-131, not asserted here.
            VsaOp::Bundle => GuaranteeStrength::Proven,
        }
    }

    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
    }

    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        // MAP-I bind is self-inverse on bipolar atoms: unbind == bind.
        self.bind(a, b)
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
        let d = a.len() as i64;
        // result[i] = a[(i + shift) mod d]  (left rotation by `shift`).
        Ok((0..a.len())
            .map(|i| a[(i as i64 + shift).rem_euclid(d) as usize])
            .collect())
    }

    fn unpermute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
        self.permute(a, -shift)
    }

    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic bipolar (`±1`) hypervector from a seed (a tiny LCG — no rand dependency).
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

    const D: u32 = 1024;

    #[test]
    fn bind_is_self_inverse_exact() {
        let m = MapI::new(D);
        assert!(m.self_inverse());
        let a = bipolar(D, 1);
        let b = bipolar(D, 2);
        let bound = m.bind(&a, &b).unwrap();
        // unbind(bind(a,b), b) == a exactly (bipolar atoms).
        let recovered = m.unbind(&bound, &b).unwrap();
        assert_eq!(recovered, a);
        assert_eq!(m.intrinsic_guarantee(VsaOp::Bind), GuaranteeStrength::Exact);
    }

    #[test]
    fn permute_is_invertible_exact() {
        let m = MapI::new(D);
        let a = bipolar(D, 7);
        let p = m.permute(&a, 3).unwrap();
        assert_eq!(m.unpermute(&p, 3).unwrap(), a);
        // A non-trivial shift actually moves things.
        assert_ne!(p, a);
        assert_eq!(
            m.intrinsic_guarantee(VsaOp::Permute),
            GuaranteeStrength::Exact
        );
    }

    #[test]
    fn permute_wraps_cyclically() {
        let m = MapI::new(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(m.permute(&a, 1).unwrap(), vec![2.0, 3.0, 4.0, 1.0]);
        assert_eq!(m.permute(&a, -1).unwrap(), vec![4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn bundle_is_similar_to_its_members() {
        // A bundle of a few near-orthogonal atoms is far more similar to a member than a random hv.
        let m = MapI::new(D);
        let items: Vec<Vec<f64>> = (0..4).map(|i| bipolar(D, 100 + i)).collect();
        let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
        let bundle = m.bundle(&refs).unwrap();
        let member_sim = m.similarity(&bundle, &items[0]);
        let stranger = bipolar(D, 9999);
        let stranger_sim = m.similarity(&bundle, &stranger);
        assert!(
            member_sim > 0.3,
            "a member should be clearly present: {member_sim}"
        );
        assert!(
            member_sim > stranger_sim + 0.2,
            "member ({member_sim}) should beat a stranger ({stranger_sim})"
        );
    }

    #[test]
    fn bundle_tag_is_proven_but_value_bound_is_m131() {
        // The literature tag is Proven (RFC-0003 §4); the Value-level Proven bound is M-131.
        assert_eq!(
            MapI::new(D).intrinsic_guarantee(VsaOp::Bundle),
            GuaranteeStrength::Proven
        );
    }

    #[test]
    fn dim_mismatch_and_empty_bundle_are_explicit() {
        let m = MapI::new(4);
        assert_eq!(
            m.bind(&[1.0, 2.0], &[1.0, 2.0]),
            Err(VsaError::DimMismatch {
                expected: 4,
                got: 2
            })
        );
        assert_eq!(m.bundle(&[]), Err(VsaError::EmptyBundle));
    }

    // --- Value-level adapters (Exact ops carry honest Meta) ------------------------------------

    fn hv_value(dim: u32, seed: u64) -> Value {
        Value::new(
            Repr::Vsa {
                model: "MAP-I".to_owned(),
                dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(bipolar(dim, seed)),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    #[test]
    fn value_bind_unbind_round_trips_with_derived_provenance() {
        let m = MapI::new(D);
        let a = hv_value(D, 11);
        let b = hv_value(D, 22);
        let bound = m.bind_values(&a, &b).unwrap();
        assert_eq!(bound.meta().guarantee(), GuaranteeStrength::Exact);
        match bound.meta().provenance() {
            Provenance::Derived { op, inputs } => {
                assert_eq!(op, &operation_hash("vsa.map_i.bind"));
                assert_eq!(inputs, &vec![a.content_hash(), b.content_hash()]);
            }
            other => panic!("expected Derived, got {other:?}"),
        }
        // unbind(bind(a,b), b) recovers a's payload.
        let recovered = m.unbind_values(&bound, &b).unwrap();
        assert_eq!(recovered.payload(), a.payload());
    }

    #[test]
    fn value_bind_unbind_refuse_non_bipolar_a3_04() {
        // A3-04 regression: bind/unbind_values stamp Exact on the bipolar self-inverse identity,
        // which holds only on the ±1 alphabet. A non-bipolar component must be refused, never
        // mis-tagged Exact. Mutant-witness: removing the check_bipolar guards in
        // bind_values/unbind_values makes these return an (Exact-tagged) Value.
        let m = MapI::new(D);
        let mut data = bipolar(D, 3);
        data[5] = 0.5;
        let bad = Value::new(
            Repr::Vsa {
                model: "MAP-I".to_owned(),
                dim: D,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(data),
            Meta::exact(Provenance::Root),
        )
        .unwrap();
        let ok = hv_value(D, 4);
        assert_eq!(
            m.bind_values(&bad, &ok),
            Err(VsaError::NonAlphabetComponent { index: 5 })
        );
        assert_eq!(
            m.unbind_values(&ok, &bad),
            Err(VsaError::NonAlphabetComponent { index: 5 })
        );
        // A bipolar bind still returns an unchanged Exact result.
        let good = m.bind_values(&ok, &hv_value(D, 5)).unwrap();
        assert_eq!(good.meta().guarantee(), GuaranteeStrength::Exact);
    }

    #[test]
    fn value_adapter_rejects_foreign_values() {
        let m = MapI::new(D);
        let wrong_dim = hv_value(D / 2, 1);
        assert!(matches!(
            m.bind_values(&wrong_dim, &wrong_dim),
            Err(VsaError::NotThisModel { .. })
        ));
    }

    #[test]
    fn value_permute_round_trips() {
        let m = MapI::new(D);
        let a = hv_value(D, 5);
        let p = m.permute_value(&a, 7).unwrap();
        // Permuting back by -7 at the data level recovers the original payload.
        let back = m.permute(
            match p.payload() {
                Payload::Hypervector(h) => h,
                _ => unreachable!(),
            },
            -7,
        );
        match (back, a.payload()) {
            (Ok(b), Payload::Hypervector(orig)) => assert_eq!(&b, orig),
            _ => panic!("permute round-trip failed"),
        }
    }
}
