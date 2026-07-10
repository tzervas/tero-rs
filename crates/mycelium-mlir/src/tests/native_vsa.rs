//! In-crate white-box tests for `dialect/native/vsa.rs` (M-856b; RFC-0039 §5.2; CLAUDE.md
//! test-layout rule). Pure **emission** checks (no toolchain): each model's `bind` lowers through
//! its own algebra in `arith`/`math` ops, the FHRR degenerate-bundle branch scaffold appears only
//! for FHRR `bundle`, the phase wrap uses `llvm.frem` (never `arith.remf` — see the module doc for
//! why), and emission is deterministic. The toolchain-dependent compile/run three-way differential
//! lives in `tests/vsa_differential.rs`.
//!
//! Feature-gated: `dialect::native` only compiles under `mlir-dialect` (`src/tests/mod.rs` gates
//! this module to match).

use crate::dialect::native::vsa::emit_vsa_mlir;
use crate::vsa_codegen::{VsaCgOp, VsaModelId, VsaProgram};

fn prog(
    op: VsaCgOp,
    model: VsaModelId,
    items: Vec<Vec<f64>>,
    shift: Option<i64>,
    bundle_delta: Option<f64>,
) -> VsaProgram {
    let dim = items[0].len() as u32;
    VsaProgram {
        op,
        model,
        dim,
        items,
        shift,
        bundle_delta,
    }
}

#[test]
fn map_i_bind_emits_a_plain_multiply() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        vec![vec![1.0, -1.0], vec![-1.0, -1.0]],
        None,
        None,
    );
    let m = emit_vsa_mlir(&p).expect("emit");
    assert!(m.starts_with("// mycelium MLIR-dialect VSA codegen"));
    assert!(m.contains("arith.mulf"), "expected arith.mulf in:\n{m}");
    assert!(m.contains("llvm.call @printf"));
    assert!(
        !m.contains("define i32 @main"),
        "must not be the direct-LLVM textual shape"
    );
}

#[test]
fn bsc_bind_emits_subtract_then_absf() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Bsc,
        vec![vec![0.0, 1.0], vec![1.0, 1.0]],
        None,
        None,
    );
    let m = emit_vsa_mlir(&p).expect("emit");
    assert!(m.contains("arith.subf"));
    assert!(
        m.contains("math.absf"),
        "BSC bind (XOR == |a-b|) must use math.absf:\n{m}"
    );
}

#[test]
fn hrr_bind_emits_the_unrolled_convolution() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Hrr,
        vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        None,
        None,
    );
    let m = emit_vsa_mlir(&p).expect("emit");
    // dim=3 circular convolution unrolls to 3*3 = 9 multiply-accumulate pairs.
    assert_eq!(
        m.matches("arith.mulf").count(),
        9,
        "expected 9 unrolled mul-adds in:\n{m}"
    );
}

#[test]
fn fhrr_bind_emits_phase_add_and_the_frem_based_wrap() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Fhrr,
        vec![vec![0.1, 1.0], vec![0.5, -0.1]],
        None,
        None,
    );
    let m = emit_vsa_mlir(&p).expect("emit");
    assert!(m.contains("arith.addf"), "bind is a phase add:\n{m}");
    assert!(
        m.contains("llvm.frem"),
        "the phase wrap must use llvm.frem (fmod semantics), never arith.remf \
         (IEEE remainder — verified to disagree with fmod/Rust's %):\n{m}"
    );
    assert!(!m.contains("arith.remf"));
}

#[test]
fn fhrr_unbind_emits_phase_subtract() {
    // FHRR unbind is Empirical and profile-gated (dim >= 256, HRR_UNBIND_PROFILE) — use a
    // profile-satisfying dim so `validate` (shared with direct-LLVM, DRY) does not refuse it.
    let dim = 256usize;
    let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.01).sin()).collect();
    let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.02).cos() * 0.1).collect();
    let p = prog(VsaCgOp::Unbind, VsaModelId::Fhrr, vec![a, b], None, None);
    let m = emit_vsa_mlir(&p).expect("emit");
    assert!(m.contains("arith.subf"), "unbind is a phase subtract:\n{m}");
}

#[test]
fn only_fhrr_bundle_emits_the_degenerate_branch_scaffold() {
    let fhrr_bundle = prog(
        VsaCgOp::Bundle,
        VsaModelId::Fhrr,
        vec![vec![0.1; 256], vec![0.2; 256], vec![0.3; 256]],
        None,
        None,
    );
    let m = emit_vsa_mlir(&fhrr_bundle).expect("emit");
    assert!(
        m.contains("cf.cond_br"),
        "FHRR bundle must branch on the degenerate check:\n{m}"
    );
    assert!(m.contains("s_deg"));
    assert!(m.contains("math.atan2"));

    let map_i_bind = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        vec![vec![1.0, -1.0], vec![-1.0, -1.0]],
        None,
        None,
    );
    let m2 = emit_vsa_mlir(&map_i_bind).expect("emit");
    assert!(!m2.contains("cf.cond_br"));
    assert!(!m2.contains("s_deg"));
}

#[test]
fn bsc_bundle_folds_the_majority_bit_host_side_no_runtime_branch() {
    // BSC bundle is profile-gated (odd item count <= 5, dim >= 1024, BSC_BUNDLE_PROFILE).
    let dim = 1024usize;
    let items = (1..=3)
        .map(|s| {
            (0..dim)
                .map(|i| if (i + s) % 2 == 0 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Bsc, items, None, None);
    let m = emit_vsa_mlir(&p).expect("emit");
    // Host-folded (constant operands): no runtime arithmetic needed for the majority bit.
    assert!(!m.contains("cf.cond_br"));
}

#[test]
fn permute_is_host_folded_no_runtime_arithmetic() {
    let p = prog(
        VsaCgOp::Permute,
        VsaModelId::Hrr,
        vec![vec![0.1, 0.2, 0.3, 0.4]],
        Some(2),
        None,
    );
    let m = emit_vsa_mlir(&p).expect("emit");
    assert!(!m.contains("arith.mulf") && !m.contains("arith.addf"));
    assert!(m.contains("arith.constant"));
}

#[test]
fn emission_is_deterministic() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Hrr,
        vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        None,
        None,
    );
    let a = emit_vsa_mlir(&p).unwrap();
    let b = emit_vsa_mlir(&p).unwrap();
    assert_eq!(a, b);
}

#[test]
fn malformed_program_is_an_explicit_refusal_not_a_panic() {
    let p = prog(VsaCgOp::Bind, VsaModelId::MapI, vec![vec![1.0]], None, None);
    let err = emit_vsa_mlir(&p).expect_err("bind with only one operand must be refused");
    assert!(matches!(err, crate::vsa_codegen::VsaAotError::Malformed(_)));
}
