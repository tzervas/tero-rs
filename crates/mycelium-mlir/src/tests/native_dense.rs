//! In-crate white-box tests for `dialect/native/dense.rs` (M-856b; RFC-0039 §5.1; CLAUDE.md
//! test-layout rule). Pure **emission** checks (no toolchain): the module carries genuine
//! `arith`/`math`/`llvm`-dialect ops (not the direct-LLVM text), the never-silent overflow/subnormal
//! branch scaffold appears only where the op needs it, and emission is deterministic. The
//! toolchain-dependent compile/run three-way differential lives in `tests/dense_differential.rs`.
//!
//! Feature-gated: `dialect::native` only compiles under `mlir-dialect` (`src/tests/mod.rs` gates
//! this module to match).

use crate::dense_codegen::{DenseCgOp, DenseProgram};
use crate::dialect::native::dense::emit_dense_mlir;
use mycelium_core::ScalarKind;

fn prog(
    op: DenseCgOp,
    dtype: ScalarKind,
    a: Vec<f64>,
    b: Option<Vec<f64>>,
    scale: Option<f64>,
) -> DenseProgram {
    let dim = a.len() as u32;
    DenseProgram {
        op,
        dim,
        dtype,
        a,
        b,
        scale,
    }
}

#[test]
fn add_emits_arith_and_math_ops_not_llvm_text() {
    let p = prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.5, 2.5],
        Some(vec![0.25, -1.0]),
        None,
    );
    let m = emit_dense_mlir(&p).expect("emit");
    assert!(m.starts_with("// mycelium MLIR-dialect Dense codegen"));
    assert!(m.contains("module {"));
    assert!(m.contains("func.func @main()"));
    assert!(m.contains("arith.addf"), "expected arith.addf in:\n{m}");
    assert!(
        m.contains("math.absf"),
        "expected the subnormal/overflow check in:\n{m}"
    );
    assert!(
        m.contains("llvm.call @printf"),
        "expected the printf read-back in:\n{m}"
    );
    // Not the direct-LLVM textual shape.
    assert!(!m.contains("define i32 @main"));
    assert!(!m.contains("fadd float"));
}

#[test]
fn neg_and_measurement_ops_never_emit_the_overflow_branch_scaffold() {
    for p in [
        prog(DenseCgOp::Neg, ScalarKind::F32, vec![1.5, -0.5], None, None),
        prog(
            DenseCgOp::Dot,
            ScalarKind::F32,
            vec![1.0, 2.0],
            Some(vec![0.5, -1.0]),
            None,
        ),
        prog(
            DenseCgOp::Similarity,
            ScalarKind::F32,
            vec![1.0, 0.0],
            Some(vec![0.0, 1.0]),
            None,
        ),
    ] {
        let m = emit_dense_mlir(&p).expect("emit");
        assert!(
            !m.contains("cf.cond_br"),
            "{:?} must not emit the overflow/subnormal branch (no rounding side-condition to \
             check): {m}",
            p.op
        );
        assert!(!m.contains("s_ovf") && !m.contains("s_sub"));
    }
}

#[test]
fn add_sub_scale_emit_the_overflow_branch_scaffold() {
    for p in [
        prog(
            DenseCgOp::Add,
            ScalarKind::F32,
            vec![1.0],
            Some(vec![1.0]),
            None,
        ),
        prog(
            DenseCgOp::Sub,
            ScalarKind::F32,
            vec![1.0],
            Some(vec![1.0]),
            None,
        ),
        prog(
            DenseCgOp::Scale,
            ScalarKind::F32,
            vec![1.0],
            None,
            Some(2.0),
        ),
    ] {
        let m = emit_dense_mlir(&p).expect("emit");
        assert!(
            m.contains("cf.cond_br"),
            "{:?} must emit the branch scaffold: {m}",
            p.op
        );
        assert!(m.contains("s_ovf") && m.contains("s_sub"));
    }
}

#[test]
fn bf16_scale_emits_the_round_to_grid_bit_trick() {
    let p = prog(
        DenseCgOp::Scale,
        ScalarKind::Bf16,
        vec![1.0, 2.0],
        None,
        Some(0.5),
    );
    let m = emit_dense_mlir(&p).expect("emit");
    assert!(
        m.contains("arith.shrui"),
        "expected the bf16 round-to-grid shift in:\n{m}"
    );
    assert!(m.contains("arith.andi"));
    assert!(m.contains("arith.shli"));
}

#[test]
fn emission_is_deterministic() {
    let p = prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.5, 2.5],
        Some(vec![0.25, -1.0]),
        None,
    );
    let a = emit_dense_mlir(&p).unwrap();
    let b = emit_dense_mlir(&p).unwrap();
    assert_eq!(a, b);
}

#[test]
fn malformed_program_is_an_explicit_refusal_not_a_panic() {
    // Add with no operand b — same `validate`/malformed contract as direct-LLVM (DRY).
    let p = prog(DenseCgOp::Add, ScalarKind::F32, vec![1.0], None, None);
    let err = emit_dense_mlir(&p).expect_err("binary op with no b must be refused");
    assert!(matches!(
        err,
        crate::dense_codegen::DenseAotError::Malformed(_)
    ));
}
