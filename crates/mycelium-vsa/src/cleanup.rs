//! **Cleanup / item memory** (M-132; FR-S4; RFC-0003 §3).
//!
//! An associative memory over a labelled codebook of clean atoms. A noisy query — e.g. the result
//! of an *approximate* `unbind` or a `bundle` decode — is snapped to the nearest stored atom by
//! [similarity](crate::VsaModel::similarity), returning the recovered item **and a confidence**
//! (the match similarity) plus the **margin** to the runner-up. The confidence/margin are what make
//! an approximate unbind *usable* (FR-S4): a caller can threshold on them rather than trusting a
//! silent nearest-neighbour pick (G2 — the decision is inspectable, never hidden).

use crate::{VsaError, VsaModel};

/// A cleanup hit: the recovered codebook item plus how confident the match is.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The recovered item's label.
    pub label: String,
    /// Its index in the codebook.
    pub index: usize,
    /// Match similarity to the query (cosine, in `[-1, 1]`) — the confidence.
    pub confidence: f64,
    /// Gap to the next-best item (`confidence − second_best`; `second_best = −1` when singleton).
    /// A small margin means the retrieval is ambiguous and should be treated with suspicion.
    pub margin: f64,
}

/// A labelled item memory at a fixed dimensionality.
#[derive(Debug, Clone, Default)]
pub struct CleanupMemory {
    dim: u32,
    items: Vec<(String, Vec<f64>)>,
}

impl CleanupMemory {
    /// An empty memory for `dim`-dimensional atoms.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        CleanupMemory {
            dim,
            items: Vec::new(),
        }
    }

    /// Store an atom under `label`. Errors if its length disagrees with the memory's `dim`.
    pub fn insert(&mut self, label: impl Into<String>, atom: Vec<f64>) -> Result<(), VsaError> {
        if atom.len() != self.dim as usize {
            return Err(VsaError::DimMismatch {
                expected: self.dim as usize,
                got: atom.len(),
            });
        }
        self.items.push((label.into(), atom));
        Ok(())
    }

    /// Number of stored items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the memory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Dimensionality of the stored atoms.
    #[must_use]
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// The codebook atoms in index order, as `(label, atom)` pairs. Read-only — needed by the
    /// resonator's softmax-superposition cleanup, which forms `Σⱼ wⱼ·cᵢ,ⱼ` over the raw atoms
    /// (a projection `CleanupMemory::cleanup` alone cannot express). Mirrors `len`/`is_empty`.
    pub fn atoms(&self) -> impl Iterator<Item = (&str, &[f64])> {
        self.items
            .iter()
            .map(|(label, atom)| (label.as_str(), atom.as_slice()))
    }

    /// Clean up `query` against the codebook using `model`'s similarity: return the best-matching
    /// item with its confidence and margin, or `None` if the memory is empty or `query`'s length
    /// disagrees with `dim`. Never coerces silently — a low confidence/margin is *reported*, not
    /// hidden (the caller decides; FR-S4/G2).
    #[must_use]
    pub fn cleanup<M: VsaModel>(&self, query: &[f64], model: &M) -> Option<Match> {
        if self.items.is_empty() || query.len() != self.dim as usize {
            return None;
        }
        // Score every item; track the best and second-best similarity.
        let mut best: (usize, f64) = (0, f64::NEG_INFINITY);
        let mut second = f64::NEG_INFINITY;
        for (i, (_, atom)) in self.items.iter().enumerate() {
            let sim = model.similarity(query, atom);
            if sim > best.1 {
                second = best.1;
                best = (i, sim);
            } else if sim > second {
                second = sim;
            }
        }
        // With a single item there is no runner-up; treat the floor of cosine (−1) as the baseline.
        if !second.is_finite() {
            second = -1.0;
        }
        let (index, confidence) = best;
        Some(Match {
            label: self.items[index].0.clone(),
            index,
            confidence,
            margin: confidence - second,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MapI;

    /// Deterministic bipolar atom (tiny LCG; no rand dependency).
    fn atom(dim: usize, seed: u64) -> Vec<f64> {
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

    fn memory_of(labels: &[(&str, u64)]) -> CleanupMemory {
        let mut mem = CleanupMemory::new(D);
        for (label, seed) in labels {
            mem.insert(*label, atom(D as usize, *seed)).unwrap();
        }
        mem
    }

    #[test]
    fn exact_atom_cleans_to_itself_with_full_confidence() {
        let model = MapI::new(D);
        let mem = memory_of(&[("alpha", 1), ("beta", 2), ("gamma", 3)]);
        let query = atom(D as usize, 2); // == beta
        let hit = mem.cleanup(&query, &model).expect("non-empty");
        assert_eq!(hit.label, "beta");
        assert_eq!(hit.index, 1);
        assert!(
            (hit.confidence - 1.0).abs() < 1e-9,
            "conf={}",
            hit.confidence
        );
        assert!(hit.margin > 0.8, "margin={}", hit.margin); // clearly separated
    }

    #[test]
    fn empty_memory_and_dim_mismatch_return_none() {
        let model = MapI::new(D);
        assert!(CleanupMemory::new(D)
            .cleanup(&atom(D as usize, 1), &model)
            .is_none());
        let mem = memory_of(&[("a", 1)]);
        assert!(mem.cleanup(&atom(8, 1), &model).is_none()); // wrong length
    }

    #[test]
    fn insert_rejects_wrong_dimension() {
        let mut mem = CleanupMemory::new(D);
        assert!(matches!(
            mem.insert("x", vec![1.0, -1.0]),
            Err(VsaError::DimMismatch { .. })
        ));
    }

    /// The headline use case (FR-S4): bundle role⊗filler pairs, then *approximately* unbind by a
    /// role and clean up the noisy result to the right filler — with positive confidence/margin.
    #[test]
    fn cleanup_makes_approximate_unbind_usable() {
        let model = MapI::new(D);
        // Roles and fillers.
        let role_color = atom(D as usize, 10);
        let role_shape = atom(D as usize, 11);
        let red = atom(D as usize, 20);
        let cube = atom(D as usize, 21);
        // A record = bundle( color⊗red , shape⊗cube ).
        let cr = model.bind(&role_color, &red).unwrap();
        let sc = model.bind(&role_shape, &cube).unwrap();
        let record = model.bundle(&[&cr, &sc]).unwrap();

        // Filler codebook for cleanup.
        let mut fillers = CleanupMemory::new(D);
        fillers.insert("red", red.clone()).unwrap();
        fillers.insert("cube", cube.clone()).unwrap();
        fillers.insert("green", atom(D as usize, 22)).unwrap();
        fillers.insert("sphere", atom(D as usize, 23)).unwrap();

        // Approximate unbind by the colour role → noisy ≈ red.
        let noisy = model.unbind(&record, &role_color).unwrap();
        let hit = fillers.cleanup(&noisy, &model).expect("non-empty");
        assert_eq!(hit.label, "red", "should recover the colour filler");
        assert!(hit.confidence > 0.3, "confidence={}", hit.confidence);
        assert!(hit.margin > 0.2, "margin={}", hit.margin);

        // And the shape role recovers the cube.
        let noisy_shape = model.unbind(&record, &role_shape).unwrap();
        assert_eq!(fillers.cleanup(&noisy_shape, &model).unwrap().label, "cube");
    }

    #[test]
    fn singleton_memory_reports_a_margin_against_the_cosine_floor() {
        let model = MapI::new(D);
        let mem = memory_of(&[("only", 7)]);
        let hit = mem.cleanup(&atom(D as usize, 7), &model).unwrap();
        assert_eq!(hit.label, "only");
        // confidence ≈ 1, margin = confidence − (−1) ≈ 2.
        assert!(hit.margin > 1.9, "margin={}", hit.margin);
    }
}
