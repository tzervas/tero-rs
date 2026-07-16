//! Self-contained MAP-I algebra + cleanup memory for tero Layer-2.
//! Replaces former `mycelium-vsa` path dependency (language project, not required in-tree).

/// Algebra / memory errors (explicit, never silent).
#[derive(Debug, Clone, PartialEq)]
pub enum VsaError {
    DimMismatch { expected: usize, got: usize },
    EmptyBundle,
}

impl std::fmt::Display for VsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyBundle => write!(f, "bundle requires at least one item"),
        }
    }
}

impl std::error::Error for VsaError {}

/// MAP-I model at fixed dimensionality (elementwise product bind, sum bundle, cosine sim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapI {
    pub dim: u32,
}

impl MapI {
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Self { dim }
    }

    #[must_use]
    pub const fn model_id(&self) -> &'static str {
        "MAP-I"
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

    /// Elementwise product (self-inverse on ±1).
    pub fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.check_len(a)?;
        self.check_len(b)?;
        Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
    }

    /// Unbind == bind for MAP-I bipolar atoms.
    pub fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
        self.bind(a, b)
    }

    /// Superposition (elementwise sum).
    pub fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
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

    /// Cosine similarity.
    #[must_use]
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot / denom
        }
    }
}

/// Cleanup hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub label: String,
    pub index: usize,
    pub confidence: f64,
    pub margin: f64,
}

/// Labelled item memory.
#[derive(Debug, Clone, Default)]
pub struct CleanupMemory {
    dim: u32,
    items: Vec<(String, Vec<f64>)>,
}

impl CleanupMemory {
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Self {
            dim,
            items: Vec::new(),
        }
    }

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

    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn atoms(&self) -> impl Iterator<Item = (&str, &[f64])> {
        self.items
            .iter()
            .map(|(label, atom)| (label.as_str(), atom.as_slice()))
    }

    /// Nearest neighbour by cosine similarity.
    ///
    /// Argument order matches the historical mycelium-vsa API used by tero:
    /// `cleanup(query, model)`.
    /// Returns `None` when the codebook is empty or the query dim mismatches.
    pub fn cleanup(&self, query: &[f64], model: &MapI) -> Option<Match> {
        if self.items.is_empty() || query.len() != self.dim as usize {
            return None;
        }
        let mut best_i = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut second = f64::NEG_INFINITY;
        for (i, (_, atom)) in self.items.iter().enumerate() {
            let s = model.similarity(query, atom);
            if s > best {
                second = best;
                best = s;
                best_i = i;
            } else if s > second {
                second = s;
            }
        }
        if second == f64::NEG_INFINITY {
            second = -1.0;
        }
        Some(Match {
            label: self.items[best_i].0.clone(),
            index: best_i,
            confidence: best,
            margin: best - second,
        })
    }
}

/// Capacity bound helpers (formula only; returns whether dim suffices).
pub mod capacity {
    pub const MARGIN_MU: f64 = 0.1;

    #[must_use]
    pub fn required_dim(items: u64, delta: f64, mu: f64) -> u64 {
        if items == 0
            || !delta.is_finite()
            || delta <= 0.0
            || delta > 1.0
            || !mu.is_finite()
            || mu <= 0.0
        {
            return u64::MAX;
        }
        let val = (2.0 / (mu * mu)) * (items as f64 / delta).ln();
        if !val.is_finite() || val < 0.0 {
            return 0;
        }
        val.ceil() as u64
    }

    /// `Some(())` when dim is sufficient for a proven-capacity claim (no mycelium Bound type).
    #[must_use]
    pub fn proven_capacity_bound(items: u64, dim: u64, delta: f64) -> Option<()> {
        let required = required_dim(items, delta, MARGIN_MU);
        if dim < required {
            None
        } else {
            Some(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_is_self_inverse_on_bipolar() {
        let m = MapI::new(4);
        let a = vec![1.0, -1.0, 1.0, -1.0];
        let b = vec![-1.0, -1.0, 1.0, 1.0];
        let c = m.bind(&a, &b).unwrap();
        let back = m.bind(&c, &b).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn capacity_table() {
        assert_eq!(capacity::required_dim(3, 1e-2, capacity::MARGIN_MU), 1141);
    }
}
