//! M-260 — the reconstruction manifest (RFC-0003 §6; SPEC §10.10): a `ReconInfo` distinguishing
//! indexed retrieval from compositional reconstruction, codebooks referenced by content hash,
//! with an attached honest bound. **Exit criterion:** the compositional path recovers a *novel
//! combination* not present in the codebook. Plus the round-trip test: the manifest travels in
//! `Meta.reconstruction` and survives the value wire form.

use std::collections::BTreeMap;

use mycelium_core::{
    Bound, BoundBasis, BoundKind, CleanupShape, DecodeProcedure, DecodeSpec, GuaranteeStrength,
    InitStrategy, Meta, Payload, Provenance, Recipe, ReconInfo, ReconMode, Repr, SparsityClass,
    Value,
};
use mycelium_vsa::{
    capacity, reconstruct_factors, reconstruct_role, CleanupMemory, MapI, StopReason, VsaError,
};

const D: u32 = 2048; // ≥ requiredDim(2, 1e-2) — the record bundles two bound pairs
const DELTA: f64 = 1e-2;

/// Deterministic bipolar atom (tiny LCG — house style).
fn atom(seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..D)
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

fn hv_value(data: Vec<f64>) -> Value {
    Value::new(
        Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim: D,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(data),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// The §6 exit criterion, end to end: build a record `bundle(color⊗red, shape⊗cube)` whose bound
/// pairs are **not** in any codebook, describe it with a compositional `ReconInfo` (codebooks by
/// content hash, recipe naming the roles, cleanup decode + threshold, `Proven` capacity bound via
/// the checked instantiation), attach the manifest to the record's `Meta`, round-trip the whole
/// value over the wire, and reconstruct the **novel combination** through the manifest.
#[test]
fn compositional_path_recovers_a_novel_combination() {
    let model = MapI::new(D);

    // Atoms: roles and fillers.
    let role_color = hv_value(atom(10));
    let role_shape = hv_value(atom(11));
    let red = hv_value(atom(20));
    let cube = hv_value(atom(21));
    let green = hv_value(atom(22));
    let sphere = hv_value(atom(23));

    // The record: bundle(color⊗red, shape⊗cube) — the *pairs* are novel, never stored anywhere.
    let cr = model.bind_values(&role_color, &red).unwrap();
    let sc = model.bind_values(&role_shape, &cube).unwrap();
    let record = model
        .bundle_values_certified(&[&cr, &sc], DELTA)
        .expect("2 items into 2048 dims satisfies the capacity side-condition");

    // The manifest: content-addressed codebooks + recipe + cleanup decode + the honest bound
    // (the same checked capacity instantiation the record's own bundle carries).
    let filler_codebook_ref = red.content_hash(); // the filler memory, content-addressed
    let bound = capacity::proven_capacity_bound(2, u64::from(D), DELTA)
        .expect("side-condition holds at D=2048");
    let manifest = ReconInfo::new(
        ReconMode::CompositionalReconstruction,
        "MAP-I",
        D,
        vec![filler_codebook_ref],
        Some(Recipe {
            roles: vec!["color".to_owned(), "shape".to_owned()],
            structure: BTreeMap::from([
                ("color".to_owned(), role_color.content_hash()),
                ("shape".to_owned(), role_shape.content_hash()),
            ]),
        }),
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
        bound,
    )
    .expect("well-formed compositional manifest");

    // The manifest travels with the value (Meta.reconstruction) and survives the wire.
    let described = Value::new(
        record.repr().clone(),
        record.payload().clone(),
        record.meta().clone().with_reconstruction(manifest.clone()),
    )
    .unwrap();
    let json = serde_json::to_string(&described).unwrap();
    let back: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(back, described);
    let carried = back.meta().reconstruction().expect("manifest attached");
    assert_eq!(carried.mode(), ReconMode::CompositionalReconstruction);
    assert_eq!(carried.model(), "MAP-I");

    // The filler item memory the codebook reference names.
    let mut fillers = CleanupMemory::new(D);
    for (label, v) in [
        ("red", &red),
        ("cube", &cube),
        ("green", &green),
        ("sphere", &sphere),
    ] {
        let Payload::Hypervector(h) = v.payload() else {
            unreachable!()
        };
        fillers.insert(label, h.clone()).unwrap();
    }

    // Compositional reconstruction of the NOVEL combinations: neither color⊗red nor shape⊗cube
    // exists in any codebook, yet the recipe + algebraic inverse recover both fillers.
    let hit = reconstruct_role(&model, carried, &back, "color", &role_color, &fillers).unwrap();
    assert_eq!(hit.label, "red");
    assert!(hit.confidence >= 0.2, "confidence {}", hit.confidence);
    assert!(hit.margin > 0.1, "margin {}", hit.margin);
    let hit = reconstruct_role(&model, carried, &back, "shape", &role_shape, &fillers).unwrap();
    assert_eq!(hit.label, "cube");

    // The attached bound is honest: a Proven capacity bound from the checked instantiation.
    assert!(matches!(
        carried.bound().kind,
        BoundKind::Capacity { items: 2, .. }
    ));
    assert_eq!(record.meta().guarantee(), GuaranteeStrength::Proven);

    // Outside the recipe is an explicit refusal, never a guess.
    assert!(matches!(
        reconstruct_role(&model, carried, &back, "texture", &role_color, &fillers),
        Err(VsaError::UnknownRole { .. })
    ));
}

/// The indexed-vs-compositional distinction is operational: an `IndexedRetrieval` manifest
/// refuses the compositional path, and a below-threshold retrieval is an explicit error.
#[test]
fn indexed_manifests_and_weak_retrievals_refuse_explicitly() {
    let model = MapI::new(D);
    let role = hv_value(atom(1));
    let filler = hv_value(atom(2));
    let record = model.bind_values(&role, &filler).unwrap();

    let empirical_bound = Bound {
        kind: BoundKind::Probability { delta: 0.05 },
        basis: mycelium_core::BoundBasis::EmpiricalFit {
            trials: 10_000,
            method: "test".to_owned(),
        },
    };
    let indexed = ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        D,
        vec![filler.content_hash()],
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
        empirical_bound.clone(),
    )
    .unwrap();
    let mut memory = CleanupMemory::new(D);
    let Payload::Hypervector(h) = filler.payload() else {
        unreachable!()
    };
    memory.insert("filler", h.clone()).unwrap();
    assert!(matches!(
        reconstruct_role(&model, &indexed, &record, "color", &role, &memory),
        Err(VsaError::NotCompositional)
    ));

    // A3-07 regression: an EMPTY codebook surfaces as EmptyCodebook, not EmptyBundle (which means
    // "a bundle of zero operands" — a semantically different condition). Mutant-witness: reverting
    // reconstruct_role's `ok_or(VsaError::EmptyCodebook)` to `EmptyBundle` flips this assertion.
    let well_formed = ReconInfo::new(
        ReconMode::CompositionalReconstruction,
        "MAP-I",
        D,
        vec![filler.content_hash()],
        Some(Recipe {
            roles: vec!["color".to_owned()],
            structure: BTreeMap::from([("color".to_owned(), role.content_hash())]),
        }),
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
        empirical_bound.clone(),
    )
    .unwrap();
    let empty_memory = CleanupMemory::new(D);
    assert!(matches!(
        reconstruct_role(&model, &well_formed, &record, "color", &role, &empty_memory),
        Err(VsaError::EmptyCodebook)
    ));

    // A compositional manifest with an unreachable threshold: the weak retrieval is explicit.
    let strict = ReconInfo::new(
        ReconMode::CompositionalReconstruction,
        "MAP-I",
        D,
        vec![filler.content_hash()],
        Some(Recipe {
            roles: vec!["color".to_owned()],
            structure: BTreeMap::from([("color".to_owned(), role.content_hash())]),
        }),
        DecodeSpec {
            procedure: DecodeProcedure::Cleanup,
            cleanup_threshold: Some(1.0),
            factors: None,
            iteration_budget: None,
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        },
        empirical_bound,
    )
    .unwrap();
    // Unbind by the WRONG role so the retrieval is genuinely weak.
    let wrong_role = hv_value(atom(99));
    assert!(matches!(
        reconstruct_role(&model, &strict, &record, "color", &wrong_role, &memory),
        Err(VsaError::BelowCleanupThreshold { .. })
    ));
}

// --- Resonator decode (RFC-0009; M-350) ---------------------------------------------------------

const DR: u32 = 4096; // ≥ MAPI_RESONATOR_PROFILE.min_dim

/// A deterministic bipolar atom at the resonator dimension.
fn atom_r(seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..DR)
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

/// End-to-end: a `Resonator` manifest (with the r4 decode params) factorizes a known two-factor bind
/// product into the right codebook atoms, gated by the trial-validated profile (RFC-0009; §10.2/§11).
#[test]
fn resonator_decode_recovers_factors_end_to_end() {
    let model = MapI::new(DR);

    // Two codebooks of 8 bipolar atoms each (the in-regime point F=2, k=8, d=4096).
    let mut c0 = CleanupMemory::new(DR);
    let mut c1 = CleanupMemory::new(DR);
    let mut a0 = Vec::new();
    let mut a1 = Vec::new();
    for j in 0..8u64 {
        let x = atom_r(1000 + j);
        let y = atom_r(2000 + j);
        c0.insert(format!("c0:{j}"), x.clone()).unwrap();
        c1.insert(format!("c1:{j}"), y.clone()).unwrap();
        a0.push(x);
        a1.push(y);
    }
    // True tuple (3, 5); the product s = x₃ ⊛ y₅.
    let x3 = hv_value_r(a0[3].clone());
    let y5 = hv_value_r(a1[5].clone());
    let record = model.bind_values(&x3, &y5).unwrap();

    // A Resonator manifest carrying the r4 decode params (codebooks referenced by content hash).
    let manifest = ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DR,
        vec![record.content_hash()], // manifest codebook refs (content-addressed)
        None,
        DecodeSpec {
            procedure: DecodeProcedure::Resonator,
            cleanup_threshold: None,
            factors: Some(vec![x3.content_hash(), y5.content_hash()]),
            iteration_budget: Some(50),
            cleanup: Some(CleanupShape::Softmax),
            beta: Some(6.0),
            tau_lock: Some(0.9),
            init: Some(InitStrategy::UniformSuperposition),
            seed: Some(7),
        },
        resonator_bound(),
    )
    .unwrap();

    let out = reconstruct_factors(&model, &manifest, &record, &[c0, c1]).expect("recovers factors");
    assert_eq!(out.trace.stop, StopReason::Converged);
    assert_eq!(out.factors[0].index, 3);
    assert_eq!(out.factors[1].index, 5);
}

/// An out-of-regime request (k = 32 > the profile's max_codebook=16 — past the §10.3 validated edge)
/// is an explicit refusal, never a stretched tag (RFC-0009 §5.2). The profile gate fires before the loop.
#[test]
fn resonator_decode_refuses_out_of_regime() {
    let model = MapI::new(DR);
    let codebooks: Vec<CleanupMemory> = (0..2)
        .map(|i| {
            let mut c = CleanupMemory::new(DR);
            for j in 0..32u64 {
                c.insert(format!("{i}:{j}"), atom_r(7000 + i * 100 + j))
                    .unwrap();
            }
            c
        })
        .collect();
    let record = hv_value_r(atom_r(1));
    let manifest = ReconInfo::new(
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
            seed: None,
        },
        resonator_bound(),
    )
    .unwrap();
    assert!(matches!(
        reconstruct_factors(&model, &manifest, &record, &codebooks),
        Err(VsaError::OutsideEmpiricalProfile { .. })
    ));
}
