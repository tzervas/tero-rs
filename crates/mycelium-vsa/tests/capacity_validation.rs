//! M-131 — empirical validation of the MAP-I bundle capacity bound (SC-2).
//!
//! The `Proven` capacity bound (`mycelium_vsa::capacity`) cites Clarkson/Thomas and is issued only
//! when the checked side-condition `dim ≥ requiredDim(m, δ)` holds (M-001 pattern). This test
//! *empirically validates* that the bound is not vacuous: over **≥10⁴ independent trials** at a
//! dimension that satisfies the side-condition, the measured retrieval-failure rate stays at or
//! below the proven target `δ`. (It does not re-prove the theorem — it checks the instantiation
//! behaves as claimed, the SC-2 obligation.)
//!
//! ## proptest migration (M-654 / ADR-021 Gate A3)
//!
//! The original hand-rolled LCG trial loop (`for trial in 0..TRIALS { Lcg::new(0xC0FFEE ^ trial)
//! }`) is replaced by a `proptest!` test that generates a `Vec<u64>` of exactly `TRIALS` seeds
//! using proptest's value tree.  This preserves the statistical property (failure rate ≤ δ) while
//! gaining:
//! - Proper shrinking: when a batch fails, proptest minimises the seed list.
//! - `PROPTEST_CASES` control: the outer proptest loop (number of independent batches to draw)
//!   respects the env-var; CI can rotate seeds across runs.
//! - The LCG is retained as the per-trial atom generator — it is cheap, no-alloc, and matches the
//!   distributions the crate's empirical constants were calibrated on.

use mycelium_core::{Meta, Payload, Provenance, Repr, SparsityClass, Value};
use mycelium_vsa::{capacity, MapI, VsaError};
use proptest::prelude::*;

/// Deterministic bipolar (`±1`) atom generator (a tiny LCG — reproducible, no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn bit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if (self.0 >> 63) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    }
    fn atom(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.bit()).collect()
    }
}

const M: u64 = 3; // items per bundle
const DELTA: f64 = 1e-2; // proven target failure probability
const N: usize = 8; // codebook size
const TRIALS: usize = 10_000; // ≥ 1e4 (SC-2)

/// Run a single capacity trial with the given seed. Returns `true` on failure (some non-member
/// out-ranks some member — cleanup would mis-retrieve).
fn capacity_trial_fails(seed: u64, dim: usize) -> bool {
    let mut rng = Lcg::new(seed);
    // A fresh codebook of N atoms; bundle the first M of them.
    let codebook: Vec<Vec<f64>> = (0..N).map(|_| rng.atom(dim)).collect();
    let mut bundle = vec![0.0f64; dim];
    for atom in codebook.iter().take(M as usize) {
        for (b, x) in bundle.iter_mut().zip(atom) {
            *b += x;
        }
    }
    // Dot of the bundle with each codebook atom (norms are equal, so dot ranks = cosine ranks).
    let dot = |atom: &[f64]| -> f64 { bundle.iter().zip(atom).map(|(b, x)| b * x).sum() };
    let member_min = (0..M as usize)
        .map(|i| dot(&codebook[i]))
        .fold(f64::INFINITY, f64::min);
    let stranger_max = (M as usize..N)
        .map(|j| dot(&codebook[j]))
        .fold(f64::NEG_INFINITY, f64::max);
    // Failure: some non-member out-ranks some member (cleanup would mis-retrieve).
    member_min <= stranger_max
}

proptest! {
    // Default to ONE batch (the original single-batch LCG behaviour) so `cargo test`/`just check`
    // stay fast; `PROPTEST_CASES=N` opts into N independent batches (CI seed rotation / extra power).
    #![proptest_config(ProptestConfig { cases: 1, ..ProptestConfig::default() })]
    /// SC-2: with `dim ≥ requiredDim(M, δ)`, the empirical retrieval-failure rate is `≤ δ` over ≥10⁴
    /// trials — every bundled member out-scores every non-member by nearest-neighbour cleanup.
    ///
    /// A3-08 (preserved): The Clarkson/Thomas bound is a *sufficient* condition (`dim ≥ requiredDim
    /// ⟹ failProb ≤ δ`), so a trial run at the boundary dim confirms non-vacuity but **cannot**
    /// confirm tightness. Tamper-protection for the bound's constant is the pinned-constant unit test
    /// in `mycelium_vsa::capacity`; this test guards non-vacuity only.
    ///
    /// proptest generates `TRIALS` independent seeds per case; `PROPTEST_CASES` controls how many
    /// independent batches run (default **1** — the single-batch LCG behaviour; set `PROPTEST_CASES=N`
    /// for N batches). CI rotates seeds across runs automatically.
    #[test]
    fn bundle_capacity_holds_over_1e4_trials(
        seeds in proptest::collection::vec(any::<u64>(), TRIALS)
    ) {
        let dim = capacity::required_dim(M, DELTA, capacity::MARGIN_MU) as usize; // 1141
        prop_assert!(dim >= 1141);

        let failures: usize = seeds
            .iter()
            .filter(|&&seed| capacity_trial_fails(seed, dim))
            .count();
        let rate = failures as f64 / TRIALS as f64;
        prop_assert!(
            rate <= DELTA,
            "empirical failure rate {rate} exceeded the proven δ={DELTA} \
             (failures={failures}/{TRIALS}, dim={dim})"
        );
    }
}

/// The certified Value-level bundle issues a `Proven` `CapacityBound` exactly when the dimension
/// meets the side-condition, and refuses (explicitly) when it does not — the honest downgrade.
#[test]
fn certified_bundle_is_proven_only_when_dimension_suffices() {
    let dim = capacity::required_dim(M, DELTA, capacity::MARGIN_MU) as u32; // 1141
    let model = MapI::new(dim);

    let mut rng = Lcg::new(42);
    let items: Vec<Value> = (0..M)
        .map(|_| {
            Value::new(
                Repr::Vsa {
                    model: "MAP-I".to_owned(),
                    dim,
                    sparsity: SparsityClass::Dense,
                },
                Payload::Hypervector(rng.atom(dim as usize)),
                Meta::exact(Provenance::Root),
            )
            .unwrap()
        })
        .collect();
    let refs: Vec<&Value> = items.iter().collect();

    // Sufficient dimension → Proven bound.
    let bundle = model.bundle_values_certified(&refs, DELTA).expect("proven");
    assert_eq!(
        bundle.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    match bundle.meta().bound() {
        Some(b) => {
            assert!(matches!(
                b.basis,
                mycelium_core::BoundBasis::ProvenThm { .. }
            ));
            assert!(matches!(
                b.kind,
                mycelium_core::BoundKind::Capacity { items: 3, .. }
            ));
        }
        None => panic!("a Proven bundle must carry a bound (M-I1)"),
    }

    // Undersized model → explicit InsufficientCapacity, never an unbacked Proven tag.
    let small = MapI::new(64);
    // Distinct (per-item seed) bipolar atoms so this isolates the dimension side-condition — the
    // certified path also refuses duplicate/non-bipolar items (H6), which would otherwise mask it.
    let small_items: Vec<Value> = (0..M)
        .map(|i| {
            Value::new(
                Repr::Vsa {
                    model: "MAP-I".to_owned(),
                    dim: 64,
                    sparsity: SparsityClass::Dense,
                },
                Payload::Hypervector(Lcg::new(100 + i).atom(64)),
                Meta::exact(Provenance::Root),
            )
            .unwrap()
        })
        .collect();
    let small_refs: Vec<&Value> = small_items.iter().collect();
    assert!(matches!(
        small.bundle_values_certified(&small_refs, DELTA),
        Err(VsaError::InsufficientCapacity { .. })
    ));
}

#[test]
fn certified_bundle_refuses_unchecked_side_conditions() {
    // A3-03/H6 regression: the cited capacity theorem assumes bipolar (±1) atoms and distinct items.
    // The certified path must refuse both rather than stamp an unbacked Proven tag (M-I2/VR-5).
    // Mutant-witness: removing the check_bipolar / first_duplicate guards in bundle_values_certified
    // makes these return a Proven bundle.
    let dim = capacity::required_dim(M, DELTA, capacity::MARGIN_MU) as u32; // sufficient
    let model = MapI::new(dim);
    let vsa = |hv: Vec<f64>| {
        Value::new(
            Repr::Vsa {
                model: "MAP-I".to_owned(),
                dim,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(hv),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    };

    // Non-bipolar component (a 0.5) → NonAlphabetComponent, not a Proven bound.
    let mut rng = Lcg::new(11);
    let mut bad = rng.atom(dim as usize);
    bad[0] = 0.5;
    let a = vsa(bad);
    let b = vsa(rng.atom(dim as usize));
    let c = vsa(rng.atom(dim as usize));
    assert!(matches!(
        model.bundle_values_certified(&[&a, &b, &c], DELTA),
        Err(VsaError::NonAlphabetComponent { index: 0 })
    ));

    // Duplicate item (same content) → DuplicateBundleItems.
    let d = vsa(rng.atom(dim as usize));
    let e = vsa(rng.atom(dim as usize));
    assert!(matches!(
        model.bundle_values_certified(&[&d, &e, &d], DELTA),
        Err(VsaError::DuplicateBundleItems { index: 2 })
    ));

    // Distinct bipolar items at sufficient dim still certify Proven.
    let f = vsa(rng.atom(dim as usize));
    assert!(model.bundle_values_certified(&[&d, &e, &f], DELTA).is_ok());
}
