//! **BSC** — Binary Spatter Code (M-240; RFC-0003 §4).
//!
//! Hypervectors are binary `{0, 1}` vectors. The algebra:
//! - **bind** = elementwise **XOR**; **self-inverse** — **`Exact`** (algebraic), so `unbind` is
//!   the same op.
//! - **permute** = cyclic shift — **`Exact`**.
//! - **bundle** = elementwise **majority** (the result stays binary; ties on an even operand
//!   count copy the first operand's bit — deterministic, documented). The matrix tag is
//!   **`Proven` on expectation** (Heim / Yi & Achour: minimum size to hit a target accuracy *in
//!   expectation*) — deliberately **weaker than w.p. ≥ 1−δ**, and tagged accordingly
//!   (RFC-0003 §4).
//! - **similarity** = normalized Hamming agreement mapped to `[-1, 1]` (`1 − 2·d_H/d`) — the
//!   metric appropriate to the binary alphabet (cosine of `{0,1}` vectors is not centered);
//!   documented deviation from the trait's cosine default.
//!
//! # Honesty of the Value-level bundle
//! As with MAP-B, no checked-instantiation formula for the BSC majority bundle is ratified in the
//! corpus (and the literature form is on-expectation, weaker than a δ tail), so a `Proven`
//! **value** cannot be issued (M-I2/VR-5). [`Bsc::bundle_values_empirical`] carries an
//! **`Empirical`** δ from the trial-validated [`BSC_BUNDLE_PROFILE`] (M-I3), refusing explicitly
//! outside its side-conditions.

use mycelium_core::{GuaranteeStrength, Value};

use crate::wrap::{hv_of, rotate, wrap, wrap_exact};
use crate::{EmpiricalProfile, VsaError, VsaModel, VsaOp};

/// The trial-validated regime backing the Value-level BSC bundle's `Empirical` δ
/// (`tests/empirical_profiles.rs` runs exactly these trials).
pub const BSC_BUNDLE_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 5,
    odd_items_only: true,
    min_dim: 1024,
    delta: 1e-2,
    trials: 10_000,
    method: "Monte-Carlo majority-bundle membership decode (binary atoms, odd m ≤ 5, d ≥ 1024, \
             depth 1)",
};

/// The BSC model at a fixed dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bsc {
    /// Hypervector dimensionality.
    pub dim: u32,
}

impl Bsc {
    /// A BSC model of dimension `dim`.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Bsc { dim }
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

    /// Every component must be `0` or `1` — XOR/majority are undefined elsewhere; refused, never
    /// coerced (G2).
    fn check_binary(v: &[f64]) -> Result<(), VsaError> {
        match v.iter().position(|&x| x != 0.0 && x != 1.0) {
            Some(index) => Err(VsaError::NonAlphabetComponent { index }),
            None => Ok(()),
        }
    }

    /// Value-level `bind` (Exact, XOR).
    pub fn bind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let out = self.bind(
            hv_of(self.model_id(), self.dim, a)?,
            hv_of(self.model_id(), self.dim, b)?,
        )?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.bsc.bind",
            vec![a.content_hash(), b.content_hash()],
        )
    }

    /// Value-level `unbind` (Exact; XOR is self-inverse).
    pub fn unbind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let out = self.unbind(
            hv_of(self.model_id(), self.dim, a)?,
            hv_of(self.model_id(), self.dim, b)?,
        )?;
        wrap_exact(
            self.model_id(),
            self.dim,
            out,
            "vsa.bsc.unbind",
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
            "vsa.bsc.permute",
            vec![a.content_hash()],
        )
    }

    /// Value-level **`Empirical` bundle**: majority superposition carrying the
    /// [`BSC_BUNDLE_PROFILE`] δ (module docs). Explicit refusals outside the profile and for
    /// non-binary inputs.
    pub fn bundle_values_empirical(&self, items: &[&Value]) -> Result<Value, VsaError> {
        BSC_BUNDLE_PROFILE.check(items.len(), self.dim)?;
        let hvs: Vec<&[f64]> = items
            .iter()
            .map(|v| hv_of(self.model_id(), self.dim, v))
            .collect::<Result<_, _>>()?;
        for hv in &hvs {
            Self::check_binary(hv)?;
        }
        let data = self.bundle(&hvs)?;
        let inputs = items.iter().map(|v| v.content_hash()).collect();
        wrap(
            self.model_id(),
            self.dim,
            data,
            "vsa.bsc.bundle",
            inputs,
            GuaranteeStrength::Empirical,
            Some(BSC_BUNDLE_PROFILE.bound()),
        )
    }
}

impl VsaModel for Bsc {
    fn model_id(&self) -> &'static str {
        "BSC"
    }

    fn self_inverse(&self) -> bool {
        true
    }

    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength {
        match op {
            // XOR self-inverse / circular shift: algebraic (RFC-0003 §4).
            VsaOp::Bind | VsaOp::Unbind | VsaOp::Permute => GuaranteeStrength::Exact,
            // A3-06/C1-04: this `Proven` is the literature's *operation-level, on-expectation* tag
            // (Heim / Yi & Achour) — strictly weaker than a value-level w.p. ≥ 1−δ guarantee, even
            // though the lattice renders it identically to MAP-I's tail-bound `Proven`. The lattice
            // cannot carry the "on expectation" qualifier; a matrix consumer reads it in
            // `matrix.rs`. The Value-level bundle is correctly `Empirical` (module docs).
            VsaOp::Bundle => GuaranteeStrength::Proven,
        }
    }

    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Self::check_binary(a)?;
        Self::check_binary(b)?;
        // XOR on {0,1} == |x − y|.
        Ok(a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect())
    }

    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        // XOR is self-inverse: unbind == bind.
        self.bind(a, b)
    }

    fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
        let Some((first, rest)) = items.split_first() else {
            return Err(VsaError::EmptyBundle);
        };
        self.check_len(first)?;
        Self::check_binary(first)?;
        let mut ones = first.to_vec();
        for item in rest {
            self.check_len(item)?;
            Self::check_binary(item)?;
            for (s, x) in ones.iter_mut().zip(item.iter()) {
                *s += x;
            }
        }
        let half = items.len() as f64 / 2.0;
        // Majority; a tie (exactly half ones, even operand count) copies the first operand's bit.
        Ok(ones
            .iter()
            .zip(first.iter())
            .map(|(&n, &f)| {
                if n > half {
                    1.0
                } else if n < half {
                    0.0
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

    /// Normalized Hamming agreement mapped to `[-1, 1]`: `1 − 2·d_H/d` (identical → `1`,
    /// complementary → `−1`, uncorrelated → `≈0`).
    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let hamming: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| if x == y { 0.0 } else { 1.0 })
            .sum();
        1.0 - 2.0 * hamming / a.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{Meta, Payload, Provenance, Repr, SparsityClass};

    /// Deterministic binary atom (tiny LCG — house style).
    fn binary_atom(dim: u32, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if (s >> 63) & 1 == 1 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn hv_value(dim: u32, seed: u64) -> Value {
        Value::new(
            Repr::Vsa {
                model: "BSC".to_owned(),
                dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(binary_atom(dim, seed)),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    const D: u32 = 1024;

    #[test]
    fn xor_bind_is_self_inverse_exact() {
        let m = Bsc::new(D);
        assert!(m.self_inverse());
        let a = binary_atom(D, 1);
        let b = binary_atom(D, 2);
        let bound = m.bind(&a, &b).unwrap();
        assert_eq!(m.unbind(&bound, &b).unwrap(), a);
        assert_eq!(m.intrinsic_guarantee(VsaOp::Bind), GuaranteeStrength::Exact);
    }

    #[test]
    fn permute_round_trips() {
        let m = Bsc::new(D);
        let a = binary_atom(D, 3);
        let p = m.permute(&a, 5).unwrap();
        assert_eq!(m.unpermute(&p, 5).unwrap(), a);
        assert_ne!(p, a);
    }

    #[test]
    fn majority_bundle_is_binary_and_close_to_members() {
        let m = Bsc::new(D);
        let items: Vec<Vec<f64>> = (0..3).map(|i| binary_atom(D, 40 + i)).collect();
        let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
        let bundle = m.bundle(&refs).unwrap();
        assert!(
            bundle.iter().all(|&x| x == 0.0 || x == 1.0),
            "binary result"
        );
        let member = m.similarity(&bundle, &items[0]);
        let stranger = m.similarity(&bundle, &binary_atom(D, 777));
        assert!(
            member > stranger + 0.2,
            "member {member} vs stranger {stranger}"
        );
    }

    #[test]
    fn even_count_ties_copy_the_first_operand_a3_09() {
        // A3-09 regression: the documented BSC majority tie-break copies the first operand's bit on
        // an exact tie (half ones, even operand count) — the analogue of MAP-B's tie test. Without a
        // test this behavior is unverified. Mutant-witness: changing the `else { f }` arm in
        // `bundle` (e.g. to a constant) flips this assertion.
        let m = Bsc::new(4);
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 1.0]; // every position ties (one 1, one 0)
        assert_eq!(m.bundle(&[&a, &b]).unwrap(), a);
    }

    #[test]
    fn similarity_is_centered() {
        let m = Bsc::new(4);
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let not_a = vec![0.0, 1.0, 0.0, 1.0];
        assert_eq!(m.similarity(&a, &a), 1.0);
        assert_eq!(m.similarity(&a, &not_a), -1.0);
    }

    #[test]
    fn value_bundle_is_empirical_and_profile_gated() {
        let m = Bsc::new(D);
        let vals: Vec<Value> = (0..3).map(|i| hv_value(D, 10 + i)).collect();
        let refs: Vec<&Value> = vals.iter().collect();
        let bundle = m.bundle_values_empirical(&refs).unwrap();
        assert_eq!(bundle.meta().guarantee(), GuaranteeStrength::Empirical);
        assert!(matches!(
            bundle.meta().bound().map(|b| &b.kind),
            Some(mycelium_core::BoundKind::Probability { delta }) if *delta == BSC_BUNDLE_PROFILE.delta
        ));
        // Even operand count: outside the validated profile, explicit.
        let four: Vec<Value> = (0..4).map(|i| hv_value(D, 60 + i)).collect();
        let four_refs: Vec<&Value> = four.iter().collect();
        assert!(matches!(
            m.bundle_values_empirical(&four_refs),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
    }

    #[test]
    fn non_binary_components_are_refused() {
        let m = Bsc::new(4);
        assert_eq!(
            m.bind(&[1.0, 0.0, 2.0, 1.0], &[0.0, 0.0, 1.0, 1.0]),
            Err(VsaError::NonAlphabetComponent { index: 2 })
        );
    }
}
