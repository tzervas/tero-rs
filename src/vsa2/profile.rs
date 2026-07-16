//! Layer-2 empirical profile (self-contained; no mycelium types).

/// Declared encode regime for Layer-2 record bundles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalProfile {
    pub max_items: usize,
    pub odd_items_only: bool,
    pub min_dim: u32,
    pub delta: f64,
    pub trials: u64,
    pub method: &'static str,
}

impl EmpiricalProfile {
    /// Refuse out-of-regime encode parameters explicitly.
    pub fn check(&self, items: usize, dim: u32) -> Result<(), String> {
        if items == 0 || items > self.max_items {
            return Err(format!(
                "validated for 1..={} items, got {items}",
                self.max_items
            ));
        }
        if self.odd_items_only && items % 2 == 0 {
            return Err(format!("validated for an odd item count only, got {items}"));
        }
        if dim < self.min_dim {
            return Err(format!("validated for dim ≥ {}, got {dim}", self.min_dim));
        }
        Ok(())
    }
}

/// Layer-2 hypervector dimensionality.
pub const L2_DIM: u32 = 4096;

/// Target per-record failure probability for capacity checks.
pub const L2_DELTA: f64 = 1e-2;

/// Per-field term cap.
pub const L2_TERM_CAP: usize = 8;

/// Declared Layer-2 profile (`trials = 0` until harness upgrades it).
pub const L2_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 64,
    odd_items_only: false,
    min_dim: L2_DIM,
    delta: L2_DELTA,
    trials: 0,
    method:
        "Declared — Layer-2 record-bundle regime; no trial validation discharged yet (M-1018); \
             the eval harness measures retrieval, it does not upgrade this profile to Empirical",
};
