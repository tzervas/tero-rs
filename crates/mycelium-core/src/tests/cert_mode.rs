//! White-box tests for [`crate::cert_mode`] — the certification-mode policy primitives
//! ([`CertMode::gate_guarantee`], [`CertMode::gate_result`]) and their never-silent / VR-5 floors
//! (RFC-0034 §5/§7; M-786/M-787/M-788). Mode-parametric (RFC-0034 §13): every assertion sweeps the
//! finite [`CertMode::ALL`] × [`GuaranteeStrength::ALL`] case space where it applies.
//!
//! Bound fixtures and the Meta-constructibility helper delegate to the shared harness
//! ([`super::mode_harness`]; M-795) so they stay in one place and any fixture change propagates
//! automatically.

use super::mode_harness::{assert_meta_constructs, declared_bound, empirical_bound, proven_bound};
use crate::bound::{Bound, BoundBasis, BoundKind};
use crate::cert_mode::CertMode;
use crate::guarantee::GuaranteeStrength;

// --- M-786/M-787 carried-over tests (extracted from the logic file, as-touched per M-797) ---

#[test]
fn default_is_fast() {
    // RFC-0034 §5: `fast` is the project default.
    assert_eq!(CertMode::default(), CertMode::Fast);
}

#[test]
fn depth_orders_fast_balanced_certified() {
    // Strictly increasing certification depth (RFC-0034 §5).
    assert!(CertMode::Fast.depth() < CertMode::Balanced.depth());
    assert!(CertMode::Balanced.depth() < CertMode::Certified.depth());
    // `ALL` is in depth order and exhaustive (the value space is finite — a complete check).
    let depths: Vec<u8> = CertMode::ALL.iter().map(|m| m.depth()).collect();
    assert_eq!(depths, vec![0, 1, 2]);
}

#[test]
fn fast_never_yields_empirical_or_proven() {
    // The M-787 invariant (VR-5 floor): exhaustive over the finite strength space.
    use GuaranteeStrength as G;
    for &g in &G::ALL {
        let gated = CertMode::Fast.gate_guarantee(g);
        assert!(
            gated != G::Empirical && gated != G::Proven,
            "fast must never compute Empirical/Proven (got {gated:?} from {g:?})"
        );
    }
    // Specifically: structural Exact passes; everything else floors to Declared.
    assert_eq!(CertMode::Fast.gate_guarantee(G::Exact), G::Exact);
    assert_eq!(CertMode::Fast.gate_guarantee(G::Proven), G::Declared);
    assert_eq!(CertMode::Fast.gate_guarantee(G::Empirical), G::Declared);
    assert_eq!(CertMode::Fast.gate_guarantee(G::Declared), G::Declared);
}

#[test]
fn balanced_and_certified_pass_every_strength_through() {
    // The machinery runs in these modes, so tag assignment is unchanged (mechanism preserved).
    use GuaranteeStrength as G;
    for &g in &G::ALL {
        assert_eq!(CertMode::Balanced.gate_guarantee(g), g);
        assert_eq!(CertMode::Certified.gate_guarantee(g), g);
    }
}

#[test]
fn serde_form_is_the_bare_variant_string() {
    // Mirrors GuaranteeStrength's wire form (RFC-0034 / guarantee.schema.json convention).
    for (mode, json) in [
        (CertMode::Fast, "\"Fast\""),
        (CertMode::Balanced, "\"Balanced\""),
        (CertMode::Certified, "\"Certified\""),
    ] {
        assert_eq!(serde_json::to_string(&mode).unwrap(), json);
        assert_eq!(serde_json::from_str::<CertMode>(json).unwrap(), mode);
    }
}

// --- M-788: gate_result — the (guarantee, bound) reconciliation against M-I1…M-I4 ---
//
// Fixture functions (`proven_bound`, `empirical_bound`, `declared_bound`) and
// `assert_meta_constructs` are imported from the shared harness (super::mode_harness; M-795).
// The local `kind_of` helper remains since it is cert_mode-specific and not harness-general.

/// The bound's `kind` payload (the computed ε/δ *value*), independent of its basis. Local helper
/// for cert_mode tests that check the value survives the gate unchanged.
fn kind_of(b: &Bound) -> BoundKind {
    b.kind.clone()
}

#[test]
fn fast_exact_stays_exact_and_bound_free_m_i1() {
    // Fast + Exact: structural, bound = None (M-I1). The pair must construct a Meta.
    let (g, b) = CertMode::Fast.gate_result(GuaranteeStrength::Exact, None);
    assert_eq!(g, GuaranteeStrength::Exact);
    assert_eq!(b, None, "Exact must be bound-free (M-I1)");
    assert_meta_constructs(g, b);
}

#[test]
fn fast_floors_proven_keeping_value_relabelling_basis_m_i4() {
    // Fast floors Proven → Declared; the computed ε *value* is kept, the basis is demoted to
    // UserDeclared (M-I4 + VR-5: computed, asserted-not-verified in fast).
    let input = proven_bound();
    let (g, b) = CertMode::Fast.gate_result(GuaranteeStrength::Proven, Some(input.clone()));
    assert_eq!(
        g,
        GuaranteeStrength::Declared,
        "fast floors Proven → Declared"
    );
    let b = b.expect("a computed bound must survive (value kept)");
    assert_eq!(
        b.basis,
        BoundBasis::UserDeclared,
        "basis demoted to UserDeclared (M-I4)"
    );
    assert_eq!(
        kind_of(&b),
        kind_of(&input),
        "the computed ε value is preserved"
    );
    assert_meta_constructs(g, Some(b));
}

#[test]
fn fast_floors_empirical_keeping_value_relabelling_basis_m_i4() {
    // Same reconciliation for an Empirical δ.
    let input = empirical_bound();
    let (g, b) = CertMode::Fast.gate_result(GuaranteeStrength::Empirical, Some(input.clone()));
    assert_eq!(g, GuaranteeStrength::Declared);
    let b = b.expect("a computed bound must survive");
    assert_eq!(b.basis, BoundBasis::UserDeclared);
    assert_eq!(
        kind_of(&b),
        kind_of(&input),
        "the computed δ value is preserved"
    );
    assert_meta_constructs(g, Some(b));
}

#[test]
fn fast_declared_is_idempotent() {
    // An already-Declared (UserDeclared-basis) intent is unchanged by the relabel — a no-op.
    let input = declared_bound();
    let (g, b) = CertMode::Fast.gate_result(GuaranteeStrength::Declared, Some(input.clone()));
    assert_eq!(g, GuaranteeStrength::Declared);
    assert_eq!(
        b,
        Some(input),
        "Declared + UserDeclared passes through untouched"
    );
    assert_meta_constructs(g, b);
}

#[test]
fn fast_exact_drops_any_stray_bound() {
    // Defensive: an Exact intent that (buggily) carries a bound is reconciled to bound = None,
    // never a silent carry that would violate M-I1.
    let (g, b) = CertMode::Fast.gate_result(GuaranteeStrength::Exact, Some(proven_bound()));
    assert_eq!(g, GuaranteeStrength::Exact);
    assert_eq!(b, None, "Exact must drop a stray bound (M-I1)");
    assert_meta_constructs(g, b);
}

#[test]
fn balanced_and_certified_pass_the_pair_through_unchanged() {
    // The machinery runs → the earned (guarantee, bound) pair is unchanged (mechanism preserved),
    // and it still constructs a Meta (the inputs were already invariant-consistent).
    for mode in [CertMode::Balanced, CertMode::Certified] {
        // Proven + ProvenThm.
        let (g, b) = mode.gate_result(GuaranteeStrength::Proven, Some(proven_bound()));
        assert_eq!(g, GuaranteeStrength::Proven);
        assert_eq!(b, Some(proven_bound()));
        assert_meta_constructs(g, b);

        // Empirical + EmpiricalFit.
        let (g, b) = mode.gate_result(GuaranteeStrength::Empirical, Some(empirical_bound()));
        assert_eq!(g, GuaranteeStrength::Empirical);
        assert_eq!(b, Some(empirical_bound()));
        assert_meta_constructs(g, b);

        // Exact + None.
        let (g, b) = mode.gate_result(GuaranteeStrength::Exact, None);
        assert_eq!(g, GuaranteeStrength::Exact);
        assert_eq!(b, None);
        assert_meta_constructs(g, b);
    }
}

#[test]
fn gate_result_guarantee_matches_gate_guarantee_in_every_mode() {
    // The guarantee component of gate_result must agree with gate_guarantee (single source of the
    // mode→tag policy; no second, divergent policy). Exhaustive over the finite case space.
    for &mode in &CertMode::ALL {
        for &g in &GuaranteeStrength::ALL {
            // Pair an invariant-consistent bound with the intent so the pre-image is realistic.
            let bound = match g {
                GuaranteeStrength::Exact => None,
                GuaranteeStrength::Proven => Some(proven_bound()),
                GuaranteeStrength::Empirical => Some(empirical_bound()),
                GuaranteeStrength::Declared => Some(declared_bound()),
            };
            let (gated_g, _) = mode.gate_result(g, bound);
            assert_eq!(
                gated_g,
                mode.gate_guarantee(g),
                "gate_result's guarantee must equal gate_guarantee (mode={mode:?}, intended={g:?})"
            );
        }
    }
}

#[test]
fn gate_result_output_is_always_meta_constructible_exhaustive() {
    // The central M-788 contract, swept exhaustively over CertMode::ALL × the four realistic
    // (intent, bound) pre-images: the gated pair ALWAYS satisfies the same M-I1…M-I4 checker the
    // Meta constructor uses — in every mode (RFC-0034 §13: mode-parametric, cross-mode).
    for &mode in &CertMode::ALL {
        let cases = [
            (GuaranteeStrength::Exact, None),
            (GuaranteeStrength::Proven, Some(proven_bound())),
            (GuaranteeStrength::Empirical, Some(empirical_bound())),
            (GuaranteeStrength::Declared, Some(declared_bound())),
        ];
        for (intent, bound) in cases {
            let (g, b) = mode.gate_result(intent, bound);
            assert_meta_constructs(g, b);
        }
    }
}

#[test]
fn fast_result_never_carries_empirical_or_proven_pair() {
    // The M-787/M-788 cross-mode NEGATIVE: no Fast-gated result is ever tagged Empirical/Proven,
    // nor does it carry a ProvenThm/EmpiricalFit basis (the basis is always reconciled away).
    let cases = [
        (GuaranteeStrength::Exact, None),
        (GuaranteeStrength::Proven, Some(proven_bound())),
        (GuaranteeStrength::Empirical, Some(empirical_bound())),
        (GuaranteeStrength::Declared, Some(declared_bound())),
    ];
    for (intent, bound) in cases {
        let (g, b) = CertMode::Fast.gate_result(intent, bound);
        assert!(
            g != GuaranteeStrength::Empirical && g != GuaranteeStrength::Proven,
            "fast result must never be Empirical/Proven (intent={intent:?} → {g:?})"
        );
        if let Some(b) = b {
            assert!(
                matches!(b.basis, BoundBasis::UserDeclared),
                "a fast result's bound basis must be UserDeclared (intent={intent:?}, basis={:?})",
                b.basis
            );
        }
    }
}
