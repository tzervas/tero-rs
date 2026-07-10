//! M-513 guarantee matrix tests — RFC-0016 §4.5.
//!
//! The guarantee matrix is data, not prose; these tests assert it:
//! 1. **Mirrors the kernel matrix** — every `(model, op)` tag in `GUARANTEE_MATRIX` must match
//!    the normative `mycelium_vsa::matrix_tag` entry (VR-5 / C2).
//! 2. **Covers every kernel model × op** — no row is missing from `GUARANTEE_MATRIX`.
//! 3. **EXPLAIN-able flag is honest** — `Exact` ops have `explain_able == false`; approximate ops
//!    have `explain_able == true`.
//! 4. **Specific pins** — the HRR/FHRR `unbind` weak link stays `Empirical`, `permute` is
//!    `Exact` for every model, and the BSC `bundle` on-expectation note is `Proven`.

use mycelium_core::GuaranteeStrength;
use mycelium_std_vsa::matrix::{std_matrix_tag, GUARANTEE_MATRIX};
use mycelium_vsa::{matrix_tag, VsaOp, RFC0003_MATRIX};

const OPS: [VsaOp; 4] = [VsaOp::Bind, VsaOp::Unbind, VsaOp::Bundle, VsaOp::Permute];

/// Every tag in `GUARANTEE_MATRIX` matches the kernel RFC-0003 §4 matrix.
///
/// Mutant-witness: flip one tag in `GUARANTEE_MATRIX` (e.g. change HRR Unbind from `Empirical`
/// to `Exact`) and this test fails.
#[test]
fn std_matrix_mirrors_the_kernel_matrix() {
    for row in GUARANTEE_MATRIX {
        let kernel_tag = matrix_tag(row.model_id, row.op).unwrap_or_else(|| {
            panic!(
                "GUARANTEE_MATRIX row ({}, {:?}) not covered by the kernel matrix",
                row.model_id, row.op
            )
        });
        assert_eq!(
            row.tag, kernel_tag,
            "GUARANTEE_MATRIX ({}, {:?}): std tag {:?} diverges from kernel {:?} (VR-5)",
            row.model_id, row.op, row.tag, kernel_tag
        );
    }
}

/// Every kernel matrix model × op appears in `GUARANTEE_MATRIX` (completeness check).
#[test]
fn std_matrix_is_complete_over_kernel_models() {
    for (model_id, op, _) in RFC0003_MATRIX {
        assert!(
            std_matrix_tag(model_id, *op).is_some(),
            "GUARANTEE_MATRIX is missing ({model_id}, {op:?})"
        );
    }
}

/// `GUARANTEE_MATRIX` has the same number of rows as the kernel matrix (no extras, no gaps).
#[test]
fn std_matrix_length_matches_kernel() {
    assert_eq!(
        GUARANTEE_MATRIX.len(),
        RFC0003_MATRIX.len(),
        "GUARANTEE_MATRIX has {} rows, kernel has {}",
        GUARANTEE_MATRIX.len(),
        RFC0003_MATRIX.len()
    );
}

/// `Exact` ops have `explain_able == false`; approximate ops have `explain_able == true`.
///
/// This encodes the C3 (EXPLAIN) contract: an `Exact` result carries no bound (M-I1), so
/// there is nothing to explain; an approximate result always carries its bound/trace.
///
/// Mutant-witness: set `explain_able = true` for any `Exact` row — this test catches it.
#[test]
fn explain_able_flag_matches_guarantee_strength() {
    for row in GUARANTEE_MATRIX {
        match row.tag {
            GuaranteeStrength::Exact => {
                assert!(
                    !row.explain_able,
                    "({}, {:?}): Exact ops carry no bound; explain_able must be false",
                    row.model_id, row.op
                );
            }
            _ => {
                assert!(
                    row.explain_able,
                    "({}, {:?}): approximate ops must have explain_able = true (C3)",
                    row.model_id, row.op
                );
            }
        }
    }
}

/// `permute` is `Exact` for every model (§4.1 erratum — a fixed coordinate bijection).
///
/// Mutant-witness: change any permute row from `Exact` to `Proven` — this test catches it.
#[test]
fn permute_is_exact_for_all_models() {
    let permute_rows: Vec<_> = GUARANTEE_MATRIX
        .iter()
        .filter(|r| r.op == VsaOp::Permute)
        .collect();
    assert!(
        !permute_rows.is_empty(),
        "no Permute rows in GUARANTEE_MATRIX"
    );
    for row in permute_rows {
        assert_eq!(
            row.tag,
            GuaranteeStrength::Exact,
            "({}, Permute): §4.1 erratum says permute is Exact for all models",
            row.model_id
        );
    }
}

/// HRR and FHRR `unbind` stays `Empirical` — the residual weak link (RFC-0003 §4 / §4.1).
///
/// Any upgrade must fail here first (VR-5).
///
/// Mutant-witness: change HRR/FHRR Unbind to `Proven` — this test catches it immediately.
#[test]
fn hrr_fhrr_unbind_weak_link_stays_empirical() {
    for model in ["HRR", "FHRR"] {
        let tag = std_matrix_tag(model, VsaOp::Unbind)
            .unwrap_or_else(|| panic!("{model} Unbind missing from GUARANTEE_MATRIX"));
        assert_eq!(
            tag,
            GuaranteeStrength::Empirical,
            "{model} Unbind must stay Empirical (the residual weak link; RFC-0003 §4.1)"
        );
    }
}

/// SBC `bind`, `unbind`, and `bundle` are `Proven` (Bloom analysis; matrix.rs).
/// FLAG Q3: the exact theorem constants are owned by RFC-0003 §5 / matrix.rs, cited here.
#[test]
fn sbc_bloom_ops_are_proven() {
    for op in [VsaOp::Bind, VsaOp::Unbind, VsaOp::Bundle] {
        let tag = std_matrix_tag("SBC", op)
            .unwrap_or_else(|| panic!("SBC {op:?} missing from GUARANTEE_MATRIX"));
        assert_eq!(
            tag,
            GuaranteeStrength::Proven,
            "SBC {op:?} must be Proven (Bloom analysis; matrix.rs)"
        );
    }
}

/// MAP-I/MAP-B/BSC `bind` and `unbind` are `Exact` (self-inverse algebraic).
#[test]
fn self_inverse_models_bind_unbind_are_exact() {
    for model in ["MAP-I", "MAP-B", "BSC"] {
        for op in [VsaOp::Bind, VsaOp::Unbind] {
            let tag = std_matrix_tag(model, op)
                .unwrap_or_else(|| panic!("{model} {op:?} missing from GUARANTEE_MATRIX"));
            assert_eq!(
                tag,
                GuaranteeStrength::Exact,
                "{model} {op:?} must be Exact (self-inverse algebraic; RFC-0003 §4.1)"
            );
        }
    }
}

/// HRR/FHRR `bind` is `Exact` (the algebraic op itself; §4.1 erratum — bind is exact,
/// unbind is the lossy one).
#[test]
fn hrr_fhrr_bind_is_exact() {
    for model in ["HRR", "FHRR"] {
        let tag = std_matrix_tag(model, VsaOp::Bind)
            .unwrap_or_else(|| panic!("{model} Bind missing from GUARANTEE_MATRIX"));
        assert_eq!(
            tag,
            GuaranteeStrength::Exact,
            "{model} Bind must be Exact (algebraic op; §4.1 erratum — the bind is not the lossy one)"
        );
    }
}

/// HRR/FHRR `bundle` is `Empirical` (Gaussian/asymptotic; FLAG Q3 — sub-Gaussian upgrade
/// not claimed without a checked instantiation).
#[test]
fn hrr_fhrr_bundle_is_empirical() {
    for model in ["HRR", "FHRR"] {
        let tag = std_matrix_tag(model, VsaOp::Bundle)
            .unwrap_or_else(|| panic!("{model} Bundle missing from GUARANTEE_MATRIX"));
        assert_eq!(
            tag,
            GuaranteeStrength::Empirical,
            "{model} Bundle must be Empirical (Gaussian; FLAG Q3 — sub-Gaussian upgrade not claimed)"
        );
    }
}

/// An unregistered model returns `None` — never a default tag.
#[test]
fn unregistered_model_returns_none() {
    for op in OPS {
        assert_eq!(
            std_matrix_tag("NOT-A-MODEL", op),
            None,
            "unregistered model must return None for {op:?}"
        );
    }
}
