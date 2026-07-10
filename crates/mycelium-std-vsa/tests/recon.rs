//! Reconstruction surface tests — compositional reconstruction and resonator factorization.
//!
//! Tests:
//! - C1: explicit errors on non-compositional manifests, unknown roles, below-threshold.
//! - C3: `ResonatorTrace` is returned on non-convergence (EXPLAIN-able failure).
//! - FR-C2 ceiling: resonator factorization is never `Proven`; the resonator profile check
//!   refuses out-of-regime requests explicitly.
//! - Round-trip: `reconstruct_role` recovers the correct filler in a bind-then-reconstruct cycle.

use std::collections::BTreeMap;

use mycelium_core::{
    operation_hash, Bound, BoundBasis, BoundKind, DecodeProcedure, DecodeSpec, Provenance, Recipe,
    ReconInfo, ReconMode,
};
use mycelium_core::{Meta, Payload, Repr, SparsityClass, Value};
use mycelium_std_vsa::{reconstruct_factors, reconstruct_role, CleanupMemory, VsaError};
use mycelium_vsa::{MapI, VsaModel};

const DIM: u32 = 4096; // resonator profile requires d ≥ 4096

fn bipolar(dim: u32, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 63) & 1 == 1 {
                1.0_f64
            } else {
                -1.0_f64
            }
        })
        .collect()
}

fn hv_value(dim: u32, data: Vec<f64>) -> Value {
    Value::new(
        Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(data),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn empirical_bound() -> Bound {
    Bound {
        kind: BoundKind::Probability { delta: 0.02 },
        basis: BoundBasis::EmpiricalFit {
            trials: 1_000,
            method: "MAPI_RESONATOR_PROFILE".to_owned(),
        },
    }
}

fn compositional_manifest(role: &str, role_hash: mycelium_core::ContentHash) -> ReconInfo {
    let recipe = Recipe {
        roles: vec![role.to_owned()],
        structure: BTreeMap::from([(role.to_owned(), role_hash)]),
    };
    let decode = DecodeSpec {
        procedure: DecodeProcedure::Cleanup,
        cleanup_threshold: Some(0.2),
        factors: None,
        iteration_budget: None,
        cleanup: None,
        beta: None,
        tau_lock: None,
        init: None,
        seed: None,
    };
    ReconInfo::new(
        ReconMode::CompositionalReconstruction,
        "MAP-I",
        DIM,
        vec![operation_hash("codebook")],
        Some(recipe),
        decode,
        empirical_bound(),
    )
    .unwrap()
}

fn indexed_manifest() -> ReconInfo {
    let decode = DecodeSpec {
        procedure: DecodeProcedure::Cleanup,
        cleanup_threshold: Some(0.2),
        factors: None,
        iteration_budget: None,
        cleanup: None,
        beta: None,
        tau_lock: None,
        init: None,
        seed: None,
    };
    ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DIM,
        vec![operation_hash("codebook")],
        None,
        decode,
        empirical_bound(),
    )
    .unwrap()
}

// --- C1: reconstruct_role errors ---

/// An indexed-retrieval manifest is `Err(NotCompositional)` — the §6 distinction, made
/// operational (C1 / G2).
#[test]
fn reconstruct_role_non_compositional_manifest_is_explicit() {
    let m = MapI::new(DIM);
    let role_atom = hv_value(DIM, bipolar(DIM, 1));
    let record = hv_value(DIM, bipolar(DIM, 2));
    let mem = CleanupMemory::new(DIM);
    let manifest = indexed_manifest();
    assert!(
        matches!(
            reconstruct_role(&m, &manifest, &record, "color", &role_atom, &mem),
            Err(VsaError::NotCompositional)
        ),
        "indexed manifest must fail with NotCompositional"
    );
}

/// An unknown role name is `Err(UnknownRole)`, never a silent best-guess.
///
/// Mutant-witness: remove the `UnknownRole` check in `recon::reconstruct_role` — this
/// proceeds to a wrong unbind rather than an explicit error.
#[test]
fn reconstruct_role_unknown_role_is_explicit() {
    let m = MapI::new(DIM);
    let role_data = bipolar(DIM, 10);
    let role_atom = hv_value(DIM, role_data.clone());
    let role_hash = role_atom.content_hash();
    let record_data = m.bind(&role_data, &bipolar(DIM, 11)).unwrap();
    let record = hv_value(DIM, record_data);
    let mut mem = CleanupMemory::new(DIM);
    mem.insert("filler", bipolar(DIM, 11)).unwrap();
    let manifest = compositional_manifest("color", role_hash);
    assert!(
        matches!(
            reconstruct_role(&m, &manifest, &record, "NOT-A-ROLE", &role_atom, &mem),
            Err(VsaError::UnknownRole { role }) if role == "NOT-A-ROLE"
        ),
        "unknown role must fail with UnknownRole"
    );
}

// --- Compositional reconstruction round-trip ---

/// Build a role⊗filler record, reconstruct via `reconstruct_role` → recovers the filler.
/// This exercises the full C3 (EXPLAIN) path: the returned `Match` carries `(confidence, margin)`.
#[test]
fn reconstruct_role_round_trip() {
    let m = MapI::new(DIM);
    let role_data = bipolar(DIM, 100);
    let filler_data = bipolar(DIM, 101);
    let role_atom = hv_value(DIM, role_data.clone());
    let role_hash = role_atom.content_hash();

    let record_data = m.bind(&role_data, &filler_data).unwrap();
    let record = hv_value(DIM, record_data);

    let mut mem = CleanupMemory::new(DIM);
    mem.insert("filler", filler_data).unwrap();
    mem.insert("stranger", bipolar(DIM, 999)).unwrap();

    let manifest = compositional_manifest("color", role_hash);
    let hit = reconstruct_role(&m, &manifest, &record, "color", &role_atom, &mem)
        .expect("compositional reconstruction should succeed");
    assert_eq!(hit.label, "filler", "should recover the correct filler");
    assert!(hit.confidence > 0.3, "confidence={}", hit.confidence);
    assert!(hit.margin > 0.1, "margin={}", hit.margin);
}

// --- Resonator factorization ---

fn resonator_manifest(factor_hashes: Vec<mycelium_core::ContentHash>) -> ReconInfo {
    let decode = DecodeSpec {
        procedure: DecodeProcedure::Resonator,
        cleanup_threshold: None,
        factors: Some(factor_hashes),
        iteration_budget: Some(50),
        cleanup: None,
        beta: None,
        tau_lock: None,
        init: None,
        seed: None,
    };
    ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DIM,
        vec![operation_hash("cb0"), operation_hash("cb1")],
        None,
        decode,
        empirical_bound(),
    )
    .unwrap()
}

fn codebook(k: usize, base: u64) -> (CleanupMemory, Vec<Vec<f64>>) {
    let mut mem = CleanupMemory::new(DIM);
    let mut atoms = Vec::with_capacity(k);
    let mut lcg = Lcg::new(base);
    for j in 0..k {
        let atom = lcg.bipolar(DIM);
        mem.insert(format!("{base}:{j}"), atom.clone()).unwrap();
        atoms.push(atom);
    }
    (mem, atoms)
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn bipolar(&mut self, dim: u32) -> Vec<f64> {
        (0..dim)
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

/// FR-C2: the resonator factorization result is `Empirical` (never `Proven`).  Checked via the
/// profile bound whose basis is `EmpiricalFit`.
#[test]
fn resonator_profile_bound_is_empirical_never_proven() {
    use mycelium_vsa::MAPI_RESONATOR_PROFILE;
    let b = MAPI_RESONATOR_PROFILE.bound();
    assert_eq!(
        b.basis.strength(),
        mycelium_core::GuaranteeStrength::Empirical,
        "MAPI_RESONATOR_PROFILE bound must be Empirical (FR-C2 — never Proven)"
    );
}

/// Out-of-regime resonator request is an explicit `OutsideEmpiricalProfile`, never stretched.
///
/// F=3, k=32 (∏k=32768) is outside the validated MAP-I profile (k≤16, ∏k≤4096 at d=4096).
///
/// Mutant-witness: remove the profile check in `recon::reconstruct_factors` — this runs the
/// resonator outside its validated envelope instead of refusing explicitly.
#[test]
fn resonator_out_of_regime_is_explicit() {
    let m = MapI::new(DIM);
    // Build a too-large codebook (k=32 > max_codebook=16).
    let (c0, a0) = codebook(32, 500);
    let (c1, a1) = codebook(32, 501);
    let (c2, a2) = codebook(32, 502);
    let s_data = {
        let m2 = MapI::new(DIM);
        let tmp = m2.bind(&a0[0], &a1[0]).unwrap();
        m2.bind(&tmp, &a2[0]).unwrap()
    };
    let s = hv_value(DIM, s_data);
    let fh: Vec<_> = [
        operation_hash("f0"),
        operation_hash("f1"),
        operation_hash("f2"),
    ]
    .to_vec();
    let manifest = ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DIM,
        vec![operation_hash("cb")],
        None,
        DecodeSpec {
            procedure: DecodeProcedure::Resonator,
            cleanup_threshold: None,
            factors: Some(fh),
            iteration_budget: Some(50),
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        },
        empirical_bound(),
    )
    .unwrap();
    assert!(
        matches!(
            reconstruct_factors(&m, &manifest, &s, &[c0, c1, c2]),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ),
        "F=3 k=32 is outside the validated profile and must fail explicitly"
    );
}

/// Resonator factorization in-regime converges and recovers the correct factors.
/// This is one deterministic trial (seed-based LCG), not a statistical claim.
#[test]
fn resonator_factorization_in_regime_converges() {
    let m = MapI::new(DIM);
    let (c0, a0) = codebook(8, 600);
    let (c1, a1) = codebook(8, 601);
    let s_data = m.bind(&a0[3], &a1[5]).unwrap();
    let s = hv_value(DIM, s_data);
    let fh = vec![operation_hash("f0"), operation_hash("f1")];
    let manifest = resonator_manifest(fh);
    let result = reconstruct_factors(&m, &manifest, &s, &[c0, c1])
        .expect("F=2 k=8 is in-regime; should converge");
    assert_eq!(result.factors[0].index, 3, "slot 0 should recover index 3");
    assert_eq!(result.factors[1].index, 5, "slot 1 should recover index 5");
}

/// Non-convergence (budget=1) is an explicit error with an inspectable trace — never a returned
/// factor set (C1/C3 / RFC-0009 §5/§6).
///
/// Mutant-witness: return `Ok(Factorization { ... })` from a BudgetExhausted run — this test
/// catches it.
#[test]
fn resonator_budget_exhausted_carries_trace() {
    let m = MapI::new(DIM);
    let (c0, a0) = codebook(8, 700);
    let (c1, a1) = codebook(8, 701);
    let s_data = m.bind(&a0[1], &a1[2]).unwrap();
    let s = hv_value(DIM, s_data);
    let fh = vec![operation_hash("f0"), operation_hash("f1")];
    let manifest = ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        DIM,
        vec![operation_hash("cb")],
        None,
        DecodeSpec {
            procedure: DecodeProcedure::Resonator,
            cleanup_threshold: None,
            factors: Some(fh),
            iteration_budget: Some(1), // far too few iterations
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        },
        empirical_bound(),
    )
    .unwrap();
    match reconstruct_factors(&m, &manifest, &s, &[c0, c1]) {
        Err(VsaError::ResonatorBudgetExhausted { trace }) => {
            assert!(
                trace.iterations <= 1,
                "trace should record ≤ 1 iterations with a budget of 1"
            );
            // The trace is inspectable (C3): it carries the trajectory.
            assert!(
                !trace.trajectory.is_empty(),
                "trace must contain at least one trajectory record"
            );
        }
        other => panic!("expected ResonatorBudgetExhausted, got {other:?}"),
    }
}
