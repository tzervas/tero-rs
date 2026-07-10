//! White-box tests for [`crate::meta`] — the `Meta` well-formedness invariants M-I1…M-I4,
//! accessor mutant-witnesses, the M-I5 lossless-physical invariant, the cert_mode tag, and the
//! wrapping opt-out (RFC-0034 §10). Extracted from the logic file as-touched (M-797; M-791).

use crate::bound::{Bound, BoundBasis, BoundKind, NormKind};
use crate::guarantee::GuaranteeStrength;
use crate::meta::{Meta, PackScheme, PhysicalLayout, Provenance, SparsityObs};
use crate::WfError;

fn proven_capacity() -> Bound {
    Bound {
        kind: BoundKind::Capacity {
            items: 3,
            dim: 10_000,
        },
        basis: BoundBasis::ProvenThm {
            citation: "Clarkson-Ubaru-Yang 2023".to_owned(),
        },
    }
}

#[test]
fn exact_without_bound_is_ok() {
    assert!(Meta::new(
        Provenance::Root,
        GuaranteeStrength::Exact,
        None,
        None,
        None,
        None
    )
    .is_ok());
}

#[test]
fn exact_with_bound_violates_m_i1() {
    let m = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Exact,
        Some(proven_capacity()),
        None,
        None,
        None,
    );
    assert_eq!(m.unwrap_err(), WfError::GuaranteeBoundMismatch);
}

#[test]
fn proven_requires_proven_basis() {
    // Proven + ProvenThm: ok (M-I2).
    assert!(Meta::new(
        Provenance::Root,
        GuaranteeStrength::Proven,
        Some(proven_capacity()),
        None,
        None,
        None,
    )
    .is_ok());
    // Declared cannot claim a ProvenThm basis (M-I4).
    let bad = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(proven_capacity()),
        None,
        None,
        None,
    );
    assert_eq!(bad.unwrap_err(), WfError::GuaranteeBoundMismatch);
}

#[test]
fn non_exact_requires_a_bound() {
    let m = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Proven,
        None,
        None,
        None,
        None,
    );
    assert_eq!(m.unwrap_err(), WfError::GuaranteeBoundMismatch);
}

#[test]
fn out_of_range_bound_is_malformed() {
    let b = Bound {
        kind: BoundKind::Probability { delta: 1.5 },
        basis: BoundBasis::UserDeclared,
    };
    let m = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(b),
        None,
        None,
        None,
    );
    assert_eq!(m.unwrap_err(), WfError::MalformedBound);
}

#[test]
fn out_of_range_sparsity_is_malformed_sparsity() {
    // A6-08 mutant-witness: an out-of-range `density` is a sparsity-observation error, not a
    // bound error — so it must be `MalformedSparsity`, never the misleading `MalformedBound`.
    let bad_sparsity = SparsityObs {
        active: 10,
        density: 1.5,
    };
    let m = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Exact,
        None,
        Some(bad_sparsity),
        None,
        None,
    );
    assert_eq!(m.unwrap_err(), WfError::MalformedSparsity);
}

#[test]
fn with_physical_is_lossless_m_i5() {
    // M-I5: recording a layout touches only `physical` — guarantee, bound, and every other
    // field are untouched, so the value's type and guarantee cannot change.
    let base = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Proven,
        Some(proven_capacity()),
        None,
        None,
        None,
    )
    .unwrap();
    let recorded = base.clone().with_physical(PhysicalLayout::TritPacked {
        scheme: PackScheme::Tl2,
    });
    assert_eq!(
        recorded.physical(),
        Some(PhysicalLayout::TritPacked {
            scheme: PackScheme::Tl2
        })
    );
    // Everything that defines type/guarantee is identical (M-I5: lossless).
    assert_eq!(recorded.guarantee(), base.guarantee());
    assert_eq!(recorded.bound(), base.bound());
    assert_eq!(recorded.provenance(), base.provenance());
    // Re-recording a different layout still changes nothing but `physical`.
    let rerecorded = recorded.clone().with_physical(PhysicalLayout::TritPacked {
        scheme: PackScheme::I2S,
    });
    assert_eq!(rerecorded.guarantee(), base.guarantee());
    assert_eq!(rerecorded.bound(), base.bound());
}

#[test]
fn error_bound_uses_norm() {
    let b = Bound {
        kind: BoundKind::Error {
            eps: 0.004,
            norm: NormKind::L2,
        },
        basis: BoundBasis::EmpiricalFit {
            trials: 10_000,
            method: "Frady-Sommer Gaussian".to_owned(),
        },
    };
    assert!(Meta::new(
        Provenance::Root,
        GuaranteeStrength::Empirical,
        Some(b),
        None,
        None,
        None,
    )
    .is_ok());
}

// Mutant-witnesses for Meta accessor methods (meta.rs:189, 194, 204, 209):
// bound(), sparsity(), reconstruction(), policy_used() must each return the value passed to
// new() (not always None). Tests construct a Meta with each optional field set and assert the
// accessor returns Some with the correct value.
#[test]
fn accessors_return_the_constructed_optional_fields() {
    // bound() — must return the bound passed to new(), not None.
    let b = proven_capacity();
    let m = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Proven,
        Some(b.clone()),
        None,
        None,
        None,
    )
    .unwrap();
    assert!(
        m.bound().is_some(),
        "bound() must return Some when a bound was passed"
    );
    assert_eq!(m.bound().unwrap(), &b);

    // sparsity() — must return the SparsityObs, not None.
    let sp = SparsityObs {
        active: 10,
        density: 0.01,
    };
    let m2 = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Exact,
        None,
        Some(sp),
        None,
        None,
    )
    .unwrap();
    assert!(
        m2.sparsity().is_some(),
        "sparsity() must return Some when a SparsityObs was passed"
    );
    assert_eq!(m2.sparsity().unwrap(), sp);

    // policy_used() — must return the ContentHash, not None.
    let hash = crate::id::ContentHash::parse("blake3:round_trip_safe").unwrap();
    let m3 = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Exact,
        None,
        None,
        None,
        Some(hash.clone()),
    )
    .unwrap();
    assert!(
        m3.policy_used().is_some(),
        "policy_used() must return Some when a hash was passed"
    );
    assert_eq!(m3.policy_used().unwrap(), &hash);

    // reconstruction() — must return the ReconInfo, not None. Built via with_reconstruction().
    use crate::content::operation_hash;
    use crate::recon::{DecodeProcedure, DecodeSpec, ReconInfo, ReconMode};
    let recon = ReconInfo::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        1024,
        vec![operation_hash("cb")],
        None,
        DecodeSpec {
            procedure: DecodeProcedure::Cleanup,
            cleanup_threshold: Some(0.5),
            factors: None,
            iteration_budget: None,
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        },
        Bound {
            kind: BoundKind::Probability { delta: 0.01 },
            basis: BoundBasis::EmpiricalFit {
                trials: 1_000,
                method: "frady".to_owned(),
            },
        },
    )
    .unwrap();
    let m4 = Meta::exact(Provenance::Root).with_reconstruction(recon.clone());
    assert!(
        m4.reconstruction().is_some(),
        "reconstruction() must return Some after with_reconstruction()"
    );
    assert_eq!(m4.reconstruction().unwrap(), &recon);
}

// Mutant-witness (meta.rs:279:60): the guard `!want_proven && !want_empirical` (for Declared)
// becomes `!want_proven || !want_empirical` when mutated. For Declared we call
// `basis_ok(bound, false, false)`, so `!false && !false = true` which correctly accepts it.
// Under the `||` mutant, `!false || !false = true` also accepts it — so **UserDeclared with
// Declared** does not distinguish the two. BUT consider a ProvenThm basis with Declared tag:
// `basis_ok(Some(ProvenThm), false, false)` → `want_proven = false` → returns `false` (correct);
// under `||`, `!false || !false = true` would also return `false` for the *outer* match arm,
// but the *inner* basis match sees `Some(ProvenThm) => want_proven`, and `want_proven = false`,
// so it still returns `false`. The real distinction is with an EmpiricalFit basis + Declared:
// `basis_ok(Some(EmpiricalFit), false, false)` → `want_empirical = false` → returns false
// (correct). Under `||`, the expression is `!want_proven || !want_empirical` = `!false || !false`
// = true, so `want_empirical` wouldn't matter — but the basis match returns `want_empirical = false`.
// The inner match still returns false in both cases. Actually the || vs && matters only when
// both flags are true simultaneously — which never happens for `(false, false)`. The real mutant
// kill is: `Declared` + `ProvenThm` basis must be REJECTED (existing `proven_requires_proven_basis`
// covers this). Additionally we need `Empirical` + `UserDeclared` to be rejected.
//
// Mutant-witness (meta.rs:291:22): the guard `basis_ok(bound, false, true)` for Empirical is
// replaced with `true`, making Empirical always succeed regardless of basis. We need a test
// that shows Empirical + UserDeclared (or ProvenThm) must be rejected.
#[test]
fn empirical_requires_empirical_fit_basis() {
    // Empirical + EmpiricalFit: ok (M-I3).
    let empirical_ok = Bound {
        kind: BoundKind::Error {
            eps: 0.1,
            norm: NormKind::L2,
        },
        basis: BoundBasis::EmpiricalFit {
            trials: 1_000,
            method: "frady".to_owned(),
        },
    };
    assert!(Meta::new(
        Provenance::Root,
        GuaranteeStrength::Empirical,
        Some(empirical_ok),
        None,
        None,
        None,
    )
    .is_ok());

    // Empirical + UserDeclared: rejected (M-I3). This kills the `true` replacement mutant.
    let declared_basis = Bound {
        kind: BoundKind::Error {
            eps: 0.1,
            norm: NormKind::L2,
        },
        basis: BoundBasis::UserDeclared,
    };
    let bad = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Empirical,
        Some(declared_basis),
        None,
        None,
        None,
    );
    assert_eq!(
        bad.unwrap_err(),
        WfError::GuaranteeBoundMismatch,
        "Empirical with UserDeclared basis must be rejected (M-I3)"
    );

    // Empirical + ProvenThm: also rejected — Empirical is lower than Proven (M-I3 requires
    // EmpiricalFit specifically). This reinforces that the guard is basis-specific.
    let proven_basis = Bound {
        kind: BoundKind::Error {
            eps: 0.05,
            norm: NormKind::L2,
        },
        basis: BoundBasis::ProvenThm {
            citation: "some theorem".to_owned(),
        },
    };
    let bad2 = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Empirical,
        Some(proven_basis),
        None,
        None,
        None,
    );
    assert_eq!(
        bad2.unwrap_err(),
        WfError::GuaranteeBoundMismatch,
        "Empirical with ProvenThm basis must be rejected (M-I3 demands EmpiricalFit)"
    );
}

// Mutant-witness for meta.rs:279:60 (the && → || in Declared check): Declared + ProvenThm
// or Declared + EmpiricalFit must both be rejected; Declared + UserDeclared must be accepted.
#[test]
fn declared_accepts_only_user_declared_basis() {
    // Declared + UserDeclared: ok (M-I4).
    let declared_ok = Bound {
        kind: BoundKind::Error {
            eps: 0.2,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    };
    assert!(Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(declared_ok),
        None,
        None,
        None,
    )
    .is_ok());

    // Declared + EmpiricalFit: rejected. Under `||` mutant, `!false || !true = !false || false`
    // = true — so the branch would (incorrectly) accept. Pinning the rejection kills the mutant.
    let empirical_basis = Bound {
        kind: BoundKind::Error {
            eps: 0.2,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::EmpiricalFit {
            trials: 10,
            method: "m".to_owned(),
        },
    };
    let bad = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(empirical_basis),
        None,
        None,
        None,
    );
    assert_eq!(
        bad.unwrap_err(),
        WfError::GuaranteeBoundMismatch,
        "Declared with EmpiricalFit basis must be rejected (M-I4)"
    );
}

#[test]
fn every_meta_carries_a_cert_mode_defaulting_to_fast() {
    // Never-silent (RFC-0034 §3.1; M-786): every Meta carries a mode; the default is Fast (§5).
    assert_eq!(
        Meta::exact(Provenance::Root).cert_mode(),
        crate::CertMode::Fast
    );
    let m = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Exact,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(m.cert_mode(), crate::CertMode::Fast);
}

#[test]
fn with_cert_mode_sets_the_tag_without_changing_guarantee() {
    // The mode is not a guarantee strength — setting it never upgrades the value (VR-5).
    let m = Meta::exact(Provenance::Root).with_cert_mode(crate::CertMode::Certified);
    assert_eq!(m.cert_mode(), crate::CertMode::Certified);
    assert_eq!(m.guarantee(), GuaranteeStrength::Exact);
}

#[test]
fn deserialize_resolves_mode_to_fast_never_silently_stronger() {
    // cert_mode is a runtime tag not carried on the wire (M-786, deferred — documented on
    // `MetaWire`). A loaded value resolves to the WEAKEST mode (Fast), never silently claiming a
    // stronger one (the VR-5 floor).
    let certified = Meta::exact(Provenance::Root).with_cert_mode(crate::CertMode::Certified);
    let json = serde_json::to_string(&certified).expect("serialize");
    let back: Meta = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.cert_mode(), crate::CertMode::Fast);
}
