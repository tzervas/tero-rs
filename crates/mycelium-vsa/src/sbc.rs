//! **SBC** — Sparse Block Codes (k-active, one-hot-per-block) (M-242; RFC-0003 §4/§5; T1.3).
//!
//! The vector is partitioned into `blocks` blocks of `block_len` components; a well-formed SBC
//! vector has **exactly one active (`1.0`) component per block** (`k = blocks` active overall —
//! the declared sparsity class `Sparse{max_active: blocks}`, a *static refinement*, T1.3). The
//! algebra (block-local circular convolution — the binding of one-hots is index addition):
//!
//! - **bind** = per-block index addition mod `block_len`; **unbind** = per-block index
//!   subtraction (an exact algebraic inverse; *not* self-inverse). The matrix tags the algebraic
//!   part **`Proven`** (RFC-0003 §4 — encoded as stated; `Exact` would be an upgrade past the
//!   normative row, and downgrades are always honest).
//! - **bundle** = elementwise sum (a counting-Bloom-style superposition; membership is **`Proven`**
//!   via the Bloom / Counting-Bloom analysis, Clarkson Thms 22–23). The bundle *result* exceeds
//!   the one-hot refinement by construction (it is a multiset), so no `Sparse{max_active: blocks}`
//!   **value** is produced for bundles here — the Value-level bundle (with its checked-formula
//!   bound) is future work recorded in the M-242 plan entry, not silently approximated.
//! - **permute** = within-block index rotation — **`Exact`** (preserves the block structure;
//!   T1.2 "permute Exact everywhere").
//! - **similarity** = cosine (for one-hot-per-block operands this is the fraction of agreeing
//!   blocks).
//!
//! **Sparsity placement (T1.3).** The *declared* class is in the `Repr`
//! (`Sparse{max_active: blocks}`); the *observed* sparsity is recorded as runtime metadata
//! (`Meta.sparsity = SparsityObs{active, density}`) on every constructed value.

use mycelium_core::{
    operation_hash, GuaranteeStrength, Meta, Payload, Provenance, Repr, SparsityClass, SparsityObs,
    Value,
};

use crate::wrap::cosine;
use crate::{VsaError, VsaModel, VsaOp};

/// The SBC model: `blocks` blocks of `block_len` components (`dim = blocks · block_len`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sbc {
    /// Number of blocks (= the number of active components, `k`).
    pub blocks: u32,
    /// Components per block.
    pub block_len: u32,
}

impl Sbc {
    /// An SBC model with `blocks` blocks of `block_len` components.
    #[must_use]
    pub fn new(blocks: u32, block_len: u32) -> Self {
        Sbc { blocks, block_len }
    }

    /// Total dimensionality.
    #[must_use]
    pub fn dim(&self) -> u32 {
        self.blocks * self.block_len
    }

    fn check_len(&self, v: &[f64]) -> Result<(), VsaError> {
        if v.len() == self.dim() as usize {
            Ok(())
        } else {
            Err(VsaError::DimMismatch {
                expected: self.dim() as usize,
                got: v.len(),
            })
        }
    }

    /// The active index of each block of a one-hot-per-block vector; the alphabet violation
    /// (`index` = the offending block's first component) is explicit, never coerced (G2).
    fn block_indices(&self, v: &[f64]) -> Result<Vec<u32>, VsaError> {
        self.check_len(v)?;
        let bl = self.block_len as usize;
        let mut indices = Vec::with_capacity(self.blocks as usize);
        for (b, block) in v.chunks_exact(bl).enumerate() {
            let mut active: Option<u32> = None;
            for (i, &x) in block.iter().enumerate() {
                if x == 1.0 {
                    if active.is_some() {
                        return Err(VsaError::NonAlphabetComponent { index: b * bl });
                    }
                    active = Some(i as u32);
                } else if x != 0.0 {
                    return Err(VsaError::NonAlphabetComponent { index: b * bl });
                }
            }
            match active {
                Some(i) => indices.push(i),
                None => return Err(VsaError::NonAlphabetComponent { index: b * bl }),
            }
        }
        Ok(indices)
    }

    /// Rebuild the one-hot-per-block vector from per-block active indices.
    fn vector_of_indices(&self, indices: &[u32]) -> Vec<f64> {
        let bl = self.block_len as usize;
        let mut out = vec![0.0; self.dim() as usize];
        for (b, &i) in indices.iter().enumerate() {
            out[b * bl + i as usize] = 1.0;
        }
        out
    }

    /// The SBC `Repr`: the declared sparsity class is the static refinement
    /// `Sparse{max_active: blocks}` (T1.3).
    #[must_use]
    pub fn repr(&self) -> Repr {
        Repr::Vsa {
            model: "SBC".to_owned(),
            dim: self.dim(),
            sparsity: SparsityClass::Sparse {
                max_active: self.blocks,
            },
        }
    }

    /// The observed sparsity of a one-hot-per-block vector (recorded as runtime `Meta`, T1.3).
    fn observed(&self) -> SparsityObs {
        SparsityObs {
            active: u64::from(self.blocks),
            density: f64::from(self.blocks) / f64::from(self.dim()),
        }
    }

    /// Construct an **`Exact`** SBC value from per-block active indices, carrying the declared
    /// `Sparse` class in the `Repr` and the observed sparsity in `Meta` (T1.3).
    pub fn value(&self, indices: &[u32]) -> Result<Value, VsaError> {
        if indices.len() != self.blocks as usize {
            return Err(VsaError::DimMismatch {
                expected: self.blocks as usize,
                got: indices.len(),
            });
        }
        if let Some(pos) = indices.iter().position(|&i| i >= self.block_len) {
            return Err(VsaError::NonAlphabetComponent {
                index: pos * self.block_len as usize,
            });
        }
        self.wrap(self.vector_of_indices(indices), Provenance::Root)
    }

    fn wrap(&self, data: Vec<f64>, provenance: Provenance) -> Result<Value, VsaError> {
        let meta = Meta::new(
            provenance,
            GuaranteeStrength::Exact,
            None,
            Some(self.observed()),
            None,
            None,
        )
        .map_err(VsaError::Wf)?;
        Value::new(self.repr(), Payload::Hypervector(data), meta).map_err(VsaError::Wf)
    }

    fn hv_of<'a>(&self, v: &'a Value) -> Result<&'a [f64], VsaError> {
        match (v.repr(), v.payload()) {
            (Repr::Vsa { model, dim, .. }, Payload::Hypervector(h))
                if model == "SBC" && *dim == self.dim() =>
            {
                Ok(h)
            }
            _ => Err(VsaError::NotThisModel { expected: "SBC" }),
        }
    }

    /// Value-level `bind`: per-block index addition; the result keeps the one-hot refinement,
    /// the declared `Sparse` class, and the observed sparsity.
    pub fn bind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let out = self.bind(self.hv_of(a)?, self.hv_of(b)?)?;
        self.wrap(
            out,
            Provenance::Derived {
                op: operation_hash("vsa.sbc.bind"),
                inputs: vec![a.content_hash(), b.content_hash()],
            },
        )
    }

    /// Value-level `unbind`: per-block index subtraction (the exact algebraic inverse of `bind`).
    pub fn unbind_values(&self, a: &Value, b: &Value) -> Result<Value, VsaError> {
        let out = self.unbind(self.hv_of(a)?, self.hv_of(b)?)?;
        self.wrap(
            out,
            Provenance::Derived {
                op: operation_hash("vsa.sbc.unbind"),
                inputs: vec![a.content_hash(), b.content_hash()],
            },
        )
    }
}

impl VsaModel for Sbc {
    fn model_id(&self) -> &'static str {
        "SBC"
    }

    fn self_inverse(&self) -> bool {
        false
    }

    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength {
        match op {
            // The §4 sparse row tags the algebraic part and the Bloom-analysis bundle Proven
            // (Clarkson Thms 22–23) — encoded as stated, never upgraded past the normative row.
            VsaOp::Bind | VsaOp::Unbind | VsaOp::Bundle => GuaranteeStrength::Proven,
            // T1.2: permute Exact everywhere.
            VsaOp::Permute => GuaranteeStrength::Exact,
        }
    }

    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        let (ia, ib) = (self.block_indices(a)?, self.block_indices(b)?);
        let indices: Vec<u32> = ia
            .iter()
            .zip(&ib)
            .map(|(x, y)| (x + y) % self.block_len)
            .collect();
        Ok(self.vector_of_indices(&indices))
    }

    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        let (ia, ib) = (self.block_indices(a)?, self.block_indices(b)?);
        let indices: Vec<u32> = ia
            .iter()
            .zip(&ib)
            .map(|(x, y)| (x + self.block_len - y) % self.block_len)
            .collect();
        Ok(self.vector_of_indices(&indices))
    }

    fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
        let Some((first, rest)) = items.split_first() else {
            return Err(VsaError::EmptyBundle);
        };
        // Counting-Bloom superposition: validate each operand's one-hot structure, then sum.
        self.block_indices(first)?;
        let mut acc = first.to_vec();
        for item in rest {
            self.block_indices(item)?;
            for (a, x) in acc.iter_mut().zip(item.iter()) {
                *a += x;
            }
        }
        Ok(acc)
    }

    /// Within-block index rotation (preserves the block structure; the whole-vector shift would
    /// destroy it).
    fn permute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
        let ia = self.block_indices(a)?;
        let bl = i64::from(self.block_len);
        let indices: Vec<u32> = ia
            .iter()
            .map(|&i| ((i64::from(i) + shift).rem_euclid(bl)) as u32)
            .collect();
        Ok(self.vector_of_indices(&indices))
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

    const B: u32 = 16; // blocks
    const L: u32 = 32; // block length

    fn indices(seed: u64, n: u32, modulus: u32) -> Vec<u32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) % u64::from(modulus)) as u32
            })
            .collect()
    }

    #[test]
    fn bind_unbind_round_trips_exactly() {
        let m = Sbc::new(B, L);
        assert!(!m.self_inverse());
        let a = m.vector_of_indices(&indices(1, B, L));
        let b = m.vector_of_indices(&indices(2, B, L));
        let bound = m.bind(&a, &b).unwrap();
        assert_eq!(m.unbind(&bound, &b).unwrap(), a);
    }

    #[test]
    fn permute_rotates_within_blocks_and_round_trips() {
        let m = Sbc::new(B, L);
        let a = m.vector_of_indices(&indices(3, B, L));
        let p = m.permute(&a, 7).unwrap();
        assert_eq!(m.unpermute(&p, 7).unwrap(), a);
        // Still a well-formed SBC vector (one active per block).
        assert!(m.block_indices(&p).is_ok());
        assert_ne!(p, a);
    }

    #[test]
    fn bundle_supports_membership_queries() {
        let m = Sbc::new(B, L);
        let items: Vec<Vec<f64>> = (0..3)
            .map(|i| m.vector_of_indices(&indices(40 + i, B, L)))
            .collect();
        let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
        let bundle = m.bundle(&refs).unwrap();
        let member = m.similarity(&bundle, &items[0]);
        let stranger = m.similarity(&bundle, &m.vector_of_indices(&indices(99, B, L)));
        assert!(
            member > stranger + 0.2,
            "member {member} vs stranger {stranger}"
        );
    }

    #[test]
    fn malformed_vectors_are_refused() {
        let m = Sbc::new(2, 4);
        // Two active components in block 1.
        let two_active = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq!(
            m.bind(&two_active, &two_active),
            Err(VsaError::NonAlphabetComponent { index: 4 })
        );
        // A non-binary component in block 0.
        let fractional = vec![0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            m.bind(&fractional, &fractional),
            Err(VsaError::NonAlphabetComponent { index: 0 })
        );
        // An empty block.
        let empty_block = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            m.bind(&empty_block, &empty_block),
            Err(VsaError::NonAlphabetComponent { index: 4 })
        );
    }

    #[test]
    fn values_carry_declared_class_and_observed_sparsity() {
        let m = Sbc::new(B, L);
        let v = m.value(&indices(5, B, L)).unwrap();
        assert_eq!(
            v.repr(),
            &Repr::Vsa {
                model: "SBC".to_owned(),
                dim: B * L,
                sparsity: SparsityClass::Sparse { max_active: B },
            }
        );
        let obs = v.meta().sparsity().expect("observed sparsity recorded");
        assert_eq!(obs.active, u64::from(B));
        assert!((obs.density - f64::from(B) / f64::from(B * L)).abs() < 1e-15);
        // Value-level bind/unbind preserve the refinement and round-trip.
        let w = m.value(&indices(6, B, L)).unwrap();
        let bound = m.bind_values(&v, &w).unwrap();
        assert!(bound.meta().sparsity().is_some());
        let back = m.unbind_values(&bound, &w).unwrap();
        assert_eq!(back.payload(), v.payload());
        // Out-of-range index: explicit.
        assert!(matches!(
            m.value(&vec![L; B as usize]),
            Err(VsaError::NonAlphabetComponent { .. })
        ));
    }
}
