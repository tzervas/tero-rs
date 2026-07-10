//! Unit tests for [`crate::fhrr`] (extracted from the former inline `mod tests` as-touched —
//! the M-797 lazy retrofit; white-box access via `use crate::fhrr::*`).

use crate::fhrr::{wrap_phase, Fhrr};
use crate::{CleanupMemory, VsaError, VsaModel, VsaOp};
use mycelium_core::{GuaranteeStrength, Meta, Payload, Repr, SparsityClass, Value};

/// Deterministic uniform-phase atom (tiny LCG — house style).
fn phasor_atom(dim: u32, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            wrap_phase(std::f64::consts::TAU * u)
        })
        .collect()
}

fn hv_value(dim: u32, seed: u64) -> Value {
    Value::new(
        Repr::Vsa {
            model: "FHRR".to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(phasor_atom(dim, seed)),
        Meta::exact(mycelium_core::Provenance::Root),
    )
    .unwrap()
}

const D: u32 = 256;

#[test]
fn bind_unbind_recovers_up_to_rounding() {
    let m = Fhrr::new(D);
    assert!(!m.self_inverse());
    let a = phasor_atom(D, 1);
    let b = phasor_atom(D, 2);
    let bound = m.bind(&a, &b).unwrap();
    let recovered = m.unbind(&bound, &b).unwrap();
    let sim = m.similarity(&recovered, &a);
    assert!(
        sim > 0.999,
        "pure-pair recovery should be near-exact: {sim}"
    );
    // Still tagged Empirical — the matrix is normative (never upgraded past it).
    assert_eq!(
        m.intrinsic_guarantee(VsaOp::Unbind),
        GuaranteeStrength::Empirical
    );
}

#[test]
fn bundle_is_phasor_valued_and_member_similar() {
    let m = Fhrr::new(D);
    let items: Vec<Vec<f64>> = (0..3).map(|i| phasor_atom(D, 30 + i)).collect();
    let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
    let bundle = m.bundle(&refs).unwrap();
    assert!(bundle
        .iter()
        .all(|&t| t > -std::f64::consts::PI && t <= std::f64::consts::PI));
    let member = m.similarity(&bundle, &items[0]);
    let stranger = m.similarity(&bundle, &phasor_atom(D, 555));
    assert!(
        member > stranger + 0.2,
        "member {member} vs stranger {stranger}"
    );
}

#[test]
fn degenerate_bundle_component_is_explicit() {
    let m = Fhrr::new(2);
    // Opposite phasors cancel exactly at every component.
    let a = vec![0.5, -1.0];
    let b = vec![
        wrap_phase(0.5 + std::f64::consts::PI),
        wrap_phase(-1.0 + std::f64::consts::PI),
    ];
    assert_eq!(
        m.bundle(&[&a, &b]),
        Err(VsaError::DegenerateBundleComponent { index: 0 })
    );
}

#[test]
fn out_of_range_phases_are_refused() {
    let m = Fhrr::new(2);
    assert_eq!(
        m.bind(&[0.1, 7.0], &[0.2, 0.3]),
        Err(VsaError::NonAlphabetComponent { index: 1 })
    );
}

#[test]
fn value_unbind_is_empirical_and_regime_gated() {
    let m = Fhrr::new(D);
    let a = hv_value(D, 1);
    let b = hv_value(D, 2);
    let bound = m.bind_values(&a, &b).unwrap();
    let noisy = m.unbind_values(&bound, &b).unwrap();
    assert_eq!(noisy.meta().guarantee(), GuaranteeStrength::Empirical);
    // Root provenance → outside the validated single-factor regime.
    assert!(matches!(
        m.unbind_values(&a, &b),
        Err(VsaError::OutsideEmpiricalProfile { .. })
    ));
}

#[test]
fn unbind_then_cleanup_recovers_the_filler() {
    let m = Fhrr::new(D);
    let role = phasor_atom(D, 10);
    let filler = phasor_atom(D, 20);
    let bound = m.bind(&role, &filler).unwrap();
    let mut mem = CleanupMemory::new(D);
    mem.insert("filler", filler).unwrap();
    mem.insert("other", phasor_atom(D, 21)).unwrap();
    let noisy = m.unbind(&bound, &role).unwrap();
    let hit = mem.cleanup(&noisy, &m).unwrap();
    assert_eq!(hit.label, "filler");
    assert!(hit.confidence > 0.9);
}

/// M-892: the Value-level `permute` (completing the FHRR bind group for the `vsa.permute` prim) —
/// an `Exact` cyclic rotation, inverted by the complementary shift, with `Derived` provenance.
#[test]
fn value_permute_is_exact_and_cyclic() {
    let m = Fhrr::new(D);
    let a = hv_value(D, 7);
    let p = m.permute_value(&a, 3).unwrap();
    assert_eq!(p.meta().guarantee(), GuaranteeStrength::Exact);
    assert!(p.meta().bound().is_none(), "Exact results carry no bound");
    assert_ne!(p.payload(), a.payload(), "a nonzero shift moves components");
    // The complementary shift restores the original components exactly (rotation is lossless).
    let back = m.permute_value(&p, i64::from(D) - 3).unwrap();
    assert_eq!(back.payload(), a.payload());
    // Model/dim guard: a foreign value is an explicit refusal.
    let wrong_dim = hv_value(64, 1);
    assert!(matches!(
        m.permute_value(&wrong_dim, 1),
        Err(VsaError::NotThisModel { .. })
    ));
}
