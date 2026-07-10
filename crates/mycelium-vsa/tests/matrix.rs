//! M-242 — every implemented model's intrinsic tags **match the RFC-0003 §4 matrix** exactly
//! (the single source-of-truth table in `mycelium_vsa::matrix`), and the matrix covers every
//! implemented model × op. The honesty rule, mechanically: code and the normative ledger cannot
//! silently diverge (VR-5).

use mycelium_core::GuaranteeStrength;
use mycelium_vsa::{matrix_tag, Bsc, Fhrr, Hrr, MapB, MapI, Sbc, VsaModel, VsaOp, RFC0003_MATRIX};

const OPS: [VsaOp; 4] = [VsaOp::Bind, VsaOp::Unbind, VsaOp::Bundle, VsaOp::Permute];

fn assert_matches_matrix(model: &dyn VsaModel) {
    for op in OPS {
        let normative = matrix_tag(model.model_id(), op)
            .unwrap_or_else(|| panic!("{} missing from the §4 matrix", model.model_id()));
        assert_eq!(
            model.intrinsic_guarantee(op),
            normative,
            "{} × {op:?}: intrinsic tag diverges from RFC-0003 §4",
            model.model_id()
        );
    }
}

/// Every implemented model × op tag equals the normative §4 matrix entry.
#[test]
fn every_model_matches_the_rfc0003_matrix() {
    assert_matches_matrix(&MapI::new(1024));
    assert_matches_matrix(&MapB::new(1024));
    assert_matches_matrix(&Bsc::new(1024));
    assert_matches_matrix(&Hrr::new(256));
    assert_matches_matrix(&Fhrr::new(256));
    assert_matches_matrix(&Sbc::new(16, 32));
}

/// The matrix is total over its models (4 ops each, no gaps or duplicates) and an unregistered
/// model honestly has no tag.
#[test]
fn matrix_is_total_and_closed() {
    let models = ["MAP-I", "MAP-B", "BSC", "HRR", "FHRR", "SBC"];
    assert_eq!(RFC0003_MATRIX.len(), models.len() * OPS.len());
    for m in models {
        for op in OPS {
            assert!(matrix_tag(m, op).is_some(), "{m} × {op:?} missing");
        }
    }
    assert_eq!(matrix_tag("NOT-A-MODEL", VsaOp::Bind), None);
}

/// The HRR/FHRR unbind weak link is pinned at `Empirical` — the residual honest gap the corpus
/// documents (RFC-0003 §4 / T1.2); any upgrade must fail here first.
#[test]
fn the_unbind_weak_link_stays_empirical() {
    for model in ["HRR", "FHRR"] {
        assert_eq!(
            matrix_tag(model, VsaOp::Unbind),
            Some(GuaranteeStrength::Empirical)
        );
    }
}
