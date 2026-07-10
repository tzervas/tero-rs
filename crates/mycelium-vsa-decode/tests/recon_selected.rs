//! RFC-0010 — Value-level **auto-selected** factor decode (`reconstruct_factors_selected`).
//!
//! Relocated from `mycelium-vsa`'s `tests/recon.rs` (M-971) alongside the function under test, which
//! moved to `mycelium-vsa-decode` so `mycelium-vsa` no longer depends on `mycelium-select` (breaking
//! the `{interp, select, vsa}` cycle — DN-68). The small resonator-dimension fixtures (`DR`,
//! `hv_value_r`, `resonator_bound`) are duplicated here (integration tests cannot share private
//! helpers across crates); the `mycelium-vsa` side keeps its own copies for the select-free
//! `reconstruct_factors`/`reconstruct_role` cases that stayed there.

use mycelium_core::{
    Bound, BoundBasis, BoundKind, DecodeProcedure, DecodeSpec, GuaranteeStrength, Meta, Payload,
    Provenance, ReconInfo, ReconMode, Repr, SparsityClass, Value,
};
use mycelium_vsa::{reconstruct_factors, CleanupMemory, MapI, VsaError, VsaModel};
use mycelium_vsa_decode::{reconstruct_factors_selected, DecodeMethod, DEFAULT_ENUM_BUDGET};

const DR: u32 = 4096; // ≥ MAPI_RESONATOR_PROFILE.min_dim

fn hv_value_r(data: Vec<f64>) -> Value {
    Value::new(
        Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim: DR,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(data),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn resonator_bound() -> Bound {
    Bound {
        kind: BoundKind::Probability { delta: 0.01 },
        basis: BoundBasis::EmpiricalFit {
            trials: 1_000,
            method: "resonator profile".to_owned(),
        },
    }
}

/// A sequential bipolar generator matching `tests/decode_select.rs` so instances proven there recover
/// here too (the resonator arm is instance-sensitive near `τ_lock`; reuse a known-good draw).
struct LcgR(u64);
impl LcgR {
    fn new(seed: u64) -> Self {
        LcgR(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn bipolar(&mut self) -> Vec<f64> {
        (0..DR)
            .map(|_| {
                if (self.next_u64() >> 63) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }
}

/// Build `f` codebooks of `k` bipolar atoms (one `Lcg` seeded with `seed`, slot by slot — identical to
/// `decode_select`'s generator) and the record `s = ⊛ chosen atoms` for `truth`.
fn build_instance(
    model: &MapI,
    f: usize,
    k: usize,
    truth: &[usize],
    seed: u64,
) -> (Vec<CleanupMemory>, Value) {
    let mut lcg = LcgR::new(seed);
    let mut mems = Vec::with_capacity(f);
    let mut chosen: Vec<Vec<f64>> = Vec::with_capacity(f);
    for (i, &t) in truth.iter().enumerate().take(f) {
        let mut c = CleanupMemory::new(DR);
        for j in 0..k {
            let a = lcg.bipolar();
            c.insert(format!("{i}:{j}"), a.clone()).unwrap();
            if j == t {
                chosen.push(a);
            }
        }
        mems.push(c);
    }
    let mut prod = chosen[0].clone();
    for a in &chosen[1..] {
        prod = model.bind(&prod, a).unwrap();
    }
    (mems, hv_value_r(prod))
}

/// A `Resonator` manifest over `record` (factor refs are cosmetic for the selected decode — the
/// executor uses the passed `codebooks`).
fn resonator_manifest(record: &Value) -> ReconInfo {
    ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DR,
        vec![record.content_hash()],
        None,
        DecodeSpec {
            procedure: DecodeProcedure::Resonator,
            cleanup_threshold: None,
            factors: Some(vec![record.content_hash()]),
            iteration_budget: Some(50),
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: Some(7),
        },
        resonator_bound(),
    )
    .unwrap()
}

/// A small instance (∏k ≤ budget) is auto-upgraded to a brute-force **`Exact`** decode (RFC-0010).
#[test]
fn selected_small_instance_is_brute_force_exact() {
    let model = MapI::new(DR);
    let truth = [3usize, 5];
    let (mems, record) = build_instance(&model, 2, 8, &truth, 10_000); // ∏=64
    let manifest = resonator_manifest(&record);
    let out =
        reconstruct_factors_selected(&model, &manifest, &record, &mems, DEFAULT_ENUM_BUDGET, None)
            .expect("recovers");
    assert_eq!(out.method, DecodeMethod::BruteForceExact);
    assert_eq!(out.guarantee, GuaranteeStrength::Exact);
    assert_eq!([out.factors[0].index, out.factors[1].index], truth);
}

/// In-regime but over a tight enumeration budget ⇒ the **`Empirical`** resonator arm runs.
#[test]
fn selected_in_regime_over_budget_is_resonator() {
    let model = MapI::new(DR);
    let truth = [1usize, 6, 3]; // the known-good draw from decode_select (codebooks(3,8,2))
    let (mems, record) = build_instance(&model, 3, 8, &truth, 2); // ∏=512
    let manifest = resonator_manifest(&record);
    // Budget 64 < ∏=512 routes to the resonator arm (vs brute force).
    let out = reconstruct_factors_selected(&model, &manifest, &record, &mems, 64, None)
        .expect("recovers");
    assert_eq!(out.method, DecodeMethod::Resonator);
    assert_eq!(out.guarantee, GuaranteeStrength::Empirical);
    assert!(out.resonator_trace.is_some());
    assert_eq!(
        [
            out.factors[0].index,
            out.factors[1].index,
            out.factors[2].index
        ],
        truth
    );
}

/// **The capability gain:** `F=4, k=8` (∏=4096) is *outside* the resonator's `max_factors=3` regime —
/// `reconstruct_factors` refuses it — yet it is enumerable, so the auto path recovers it **exactly**
/// by brute force (RFC-0010 §4.4: brute force is `Exact` for any factor count). Same manifest, two
/// outcomes: the plain decode refuses, the selected decode delivers an Exact factorization.
#[test]
fn selected_out_of_resonator_regime_but_enumerable_is_exact() {
    let model = MapI::new(DR);
    let truth = [2usize, 7, 0, 5];
    let (mems, record) = build_instance(&model, 4, 8, &truth, 30_000); // ∏=4096 = DEFAULT_ENUM_BUDGET
    let manifest = resonator_manifest(&record);

    // The plain (resonator-only) decode refuses: F=4 is outside the validated regime.
    assert!(matches!(
        reconstruct_factors(&model, &manifest, &record, &mems),
        Err(VsaError::OutsideEmpiricalProfile { .. })
    ));

    // The selected decode upgrades to brute-force Exact and recovers the 4-tuple.
    let out =
        reconstruct_factors_selected(&model, &manifest, &record, &mems, DEFAULT_ENUM_BUDGET, None)
            .expect("brute-force recovers");
    assert_eq!(out.method, DecodeMethod::BruteForceExact);
    assert_eq!(out.guarantee, GuaranteeStrength::Exact);
    assert_eq!(
        [
            out.factors[0].index,
            out.factors[1].index,
            out.factors[2].index,
            out.factors[3].index
        ],
        truth
    );
}

/// A non-`Resonator` manifest is the wrong procedure for the factor decode — explicit, not guessed.
#[test]
fn selected_non_resonator_manifest_is_rejected() {
    let model = MapI::new(DR);
    let truth = [0usize, 0];
    let (mems, record) = build_instance(&model, 2, 8, &truth, 40_000);
    let cleanup_manifest = ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DR,
        vec![record.content_hash()],
        None,
        DecodeSpec {
            procedure: DecodeProcedure::Cleanup,
            cleanup_threshold: Some(0.2),
            factors: None,
            iteration_budget: None,
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        },
        resonator_bound(),
    )
    .unwrap();
    assert!(matches!(
        reconstruct_factors_selected(
            &model,
            &cleanup_manifest,
            &record,
            &mems,
            DEFAULT_ENUM_BUDGET,
            None
        ),
        Err(VsaError::NotCompositional)
    ));
}
