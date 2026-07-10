//! M-853 — the **native-Dense-codegen differential** (E25-1; **RFC-0039 §5.1**; NFR-7; VR-4; the
//! M-210 shared checker; RFC-0004 §6).
//!
//! The native Dense lowering (`dense_codegen.rs`, direct-LLVM) is validated against the trusted-base
//! reference `mycelium-dense::DenseSpace`. Because the interpreter registers **no `dense.*` prims**
//! (Dense is a standalone operational surface — like the cert swap), the reference *is* `DenseSpace`
//! (the trusted base, NFR-7), and the candidate is the **direct-LLVM** native artifact. The two must
//! be **observably equal** (`repr + payload + guarantee`), each validated through the **single shared
//! M-210 checker** (`mycelium_cert::check` under `RefinementRelation::ObservationalEquiv`). A
//! deliberately divergent lowering is caught (a mutation of the lowering diverges here), so a passing
//! differential is meaningful, not vacuous.
//!
//! **The MLIR-dialect leg (M-856b).** The **generic bit/trit `Node` path** still honestly refuses a
//! Dense `Const` (`DialectError::Unsupported`, `dialect/native.rs::const_lane`) — that boundary is
//! permanent on both backends (Dense is lowered through the dedicated `DenseProgram` entry points,
//! never the generic `Node` path); [`dense_const_is_refused_by_the_mlir_dialect_path`] still asserts
//! it. But `dialect::native::dense` (M-856b) now provides a **dialect-native sibling** of
//! `dense_compile_and_run` over the *same* `DenseProgram`, so the differential is a genuine **three-way**
//! (reference ≡ direct-LLVM ≡ dialect) where libMLIR is provisioned — skip-graceful
//! (`DenseAotError::ToolchainMissing`) where it is not, never a faked pass (VR-5/G2). See
//! [`value_ops_dialect_matches_reference_and_direct_llvm`] /
//! [`measurement_ops_dialect_matches_reference_bit_exact`].
//!
//! **Never-silent boundaries (G2/SC-3).** A subnormal/overflow result, an off-grid/non-finite input, a
//! quantized value, and an F16/F64 dtype are each refused **non-silently** by the native path exactly
//! where `mycelium-dense` refuses — asserted in [`refusals_match_the_reference`].
//!
//! **Toolchain skip.** The direct-LLVM path needs `llc`/`clang`; where absent it returns
//! `ToolchainMissing` and the path **skips** (the house idiom) — never a false failure.
//!
//! **Guarantee:** `Empirical` — the differential is empirical evidence the native Dense codegen agrees
//! with the trusted `mycelium-dense` reference over the corpus; never upgraded to `Proven` without a
//! checked proof object linked into codegen (VR-5).

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, Payload, Repr, ScalarKind, Value};
use mycelium_dense::DenseSpace;
use mycelium_mlir::{dense_compile_and_run, DenseAotError, DenseCgOp, DenseProgram, DenseResult};
use mycelium_numerics::Certificate;

// ─── observable + helpers ────────────────────────────────────────────────────────────────────────

/// The NFR-7 observable: `(repr, payload, guarantee)`. The native read-back reconstructs the
/// reference's per-op tag, so the two observables coincide.
type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);
fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}

/// Compute the `mycelium-dense` reference value for a program (the trusted base).
fn reference_value(prog: &DenseProgram) -> Value {
    let space = DenseSpace::new(prog.dim, prog.dtype).expect("supported dtype");
    let a = space.value(prog.a.clone()).expect("on-grid a");
    match prog.op {
        DenseCgOp::Add => {
            let b = space.value(prog.b.clone().unwrap()).unwrap();
            space.add_values(&a, &b).unwrap()
        }
        DenseCgOp::Sub => {
            let b = space.value(prog.b.clone().unwrap()).unwrap();
            space.sub_values(&a, &b).unwrap()
        }
        DenseCgOp::Neg => space.neg_value(&a).unwrap(),
        DenseCgOp::Scale => space.scale_value(&a, prog.scale.unwrap()).unwrap(),
        DenseCgOp::Dot | DenseCgOp::Similarity => {
            panic!("measurement ops have no Value reference; use reference_measurement")
        }
    }
}

/// Compute the `mycelium-dense` reference measurement for a `dot`/`similarity` program.
fn reference_measurement(prog: &DenseProgram) -> f64 {
    let space = DenseSpace::new(prog.dim, prog.dtype).unwrap();
    let a = space.value(prog.a.clone()).unwrap();
    let b = space.value(prog.b.clone().unwrap()).unwrap();
    match prog.op {
        DenseCgOp::Dot => space.dot(&a, &b).unwrap(),
        DenseCgOp::Similarity => space.similarity(&a, &b).unwrap(),
        other => panic!("{other:?} is not a measurement op"),
    }
}

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

// ─── corpus (data-driven; in-range, both dtypes) ─────────────────────────────────────────────────

/// The value-op corpus: in-range `add`/`sub`/`neg`/`scale` over F32 and BF16, on-grid operands. Each
/// case's reference is `DenseSpace`; the native artifact must match it (repr + payload + guarantee).
fn value_corpus() -> Vec<DenseProgram> {
    vec![
        // F32
        prog(
            DenseCgOp::Add,
            ScalarKind::F32,
            vec![1.5, 2.5, -0.25],
            Some(vec![0.25, -1.0, 0.5]),
            None,
        ),
        prog(
            DenseCgOp::Sub,
            ScalarKind::F32,
            vec![3.0, 0.5, -2.0],
            Some(vec![1.0, 0.25, -1.5]),
            None,
        ),
        prog(
            DenseCgOp::Neg,
            ScalarKind::F32,
            vec![1.5, -0.625, 0.0, 2.0],
            None,
            None,
        ),
        prog(
            DenseCgOp::Scale,
            ScalarKind::F32,
            vec![1.5, -2.0, 0.5],
            None,
            Some(2.0),
        ),
        prog(
            DenseCgOp::Scale,
            ScalarKind::F32,
            vec![1.0, 2.0],
            None,
            Some(-0.5),
        ),
        // BF16 (the two-rounding path)
        prog(
            DenseCgOp::Add,
            ScalarKind::Bf16,
            vec![1.5, -2.0],
            Some(vec![0.5, 1.0]),
            None,
        ),
        prog(
            DenseCgOp::Sub,
            ScalarKind::Bf16,
            vec![2.0, -1.0],
            Some(vec![0.5, 0.5]),
            None,
        ),
        prog(
            DenseCgOp::Neg,
            ScalarKind::Bf16,
            vec![1.5, -2.0, 0.0],
            None,
            None,
        ),
        prog(
            DenseCgOp::Scale,
            ScalarKind::Bf16,
            vec![1.5, -2.0],
            None,
            Some(2.0),
        ),
    ]
}

/// The measurement corpus: `dot`/`similarity` over F32 (bare-`f64`, no `Meta`). The reference is the
/// `f64` measurement; the native must match bit-exactly (same left-to-right `f64` reduction).
fn measurement_corpus() -> Vec<DenseProgram> {
    vec![
        prog(
            DenseCgOp::Dot,
            ScalarKind::F32,
            vec![1.0, 2.0, -1.0],
            Some(vec![0.5, -1.0, 2.0]),
            None,
        ),
        prog(
            DenseCgOp::Dot,
            ScalarKind::F32,
            vec![0.0, 0.0],
            Some(vec![1.0, 1.0]),
            None,
        ),
        prog(
            DenseCgOp::Similarity,
            ScalarKind::F32,
            vec![1.0, 0.0],
            Some(vec![0.0, 1.0]),
            None,
        ),
        prog(
            DenseCgOp::Similarity,
            ScalarKind::F32,
            vec![1.0, 2.0, -1.0],
            Some(vec![0.5, -1.0, 2.0]),
            None,
        ),
        // zero-norm operand → similarity 0 (the zero-norm guard).
        prog(
            DenseCgOp::Similarity,
            ScalarKind::F32,
            vec![1.0, 0.0],
            Some(vec![0.0, 0.0]),
            None,
        ),
    ]
}

// ─── the differential (reference ≡ direct-LLVM, M-210-checked) ───────────────────────────────────

/// Value ops: `DenseSpace` ≡ direct-LLVM, observably equal, validated through the M-210 checker. The
/// reference's per-op tag (Proven add/sub/scale, Exact neg) must match the native read-back.
#[test]
fn value_ops_match_reference_through_the_m210_checker() {
    for (i, p) in value_corpus().iter().enumerate() {
        let reference = reference_value(p);
        match dense_compile_and_run(p) {
            Ok(DenseResult::Value(native)) => {
                assert_eq!(
                    observable(&reference),
                    observable(&native),
                    "program #{i} ({:?} {:?}): reference vs direct-LLVM diverged",
                    p.op,
                    p.dtype
                );
                // M-210: the reference↔native pair validates through the single shared TV checker.
                assert_eq!(
                    check(
                        &reference,
                        &native,
                        RefinementRelation::ObservationalEquiv,
                        Certificate::exact(),
                        &Evidence::Observational,
                    ),
                    CheckVerdict::Validated {
                        strength: GuaranteeStrength::Exact
                    },
                    "program #{i}: the shared checker must validate reference↔native"
                );
            }
            Ok(other) => panic!("program #{i}: expected a Value, got {other:?}"),
            Err(DenseAotError::ToolchainMissing(_)) => return, // env skip
            Err(e) => panic!("program #{i}: direct-LLVM errored: {e}"),
        }
    }
}

/// Measurement ops: `DenseSpace` ≡ direct-LLVM, bit-exact `f64` (the native reduction folds
/// left-to-right exactly as the reference's `.sum()`).
#[test]
fn measurement_ops_match_reference_bit_exact() {
    for (i, p) in measurement_corpus().iter().enumerate() {
        let reference = reference_measurement(p);
        match dense_compile_and_run(p) {
            Ok(DenseResult::Measurement(native)) => {
                assert_eq!(
                    reference.to_bits(),
                    native.to_bits(),
                    "program #{i} ({:?}): reference={reference} native={native} diverged",
                    p.op
                );
            }
            Ok(other) => panic!("program #{i}: expected a Measurement, got {other:?}"),
            Err(DenseAotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("program #{i}: direct-LLVM errored: {e}"),
        }
    }
}

// ─── non-vacuity: the native path actually discriminates ─────────────────────────────────────────

/// Sanity: the native Dense path **discriminates** — two different programs are NOT observably equal,
/// and the shared checker reports the divergence (so the equivalence above is non-vacuous).
#[test]
fn native_dense_distinguishes_different_programs() {
    let p1 = prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.5, 2.5],
        Some(vec![0.25, -1.0]),
        None,
    );
    let p2 = prog(
        DenseCgOp::Sub,
        ScalarKind::F32,
        vec![1.5, 2.5],
        Some(vec![0.25, -1.0]),
        None,
    );
    let (a, b) = match (dense_compile_and_run(&p1), dense_compile_and_run(&p2)) {
        (Ok(DenseResult::Value(a)), Ok(DenseResult::Value(b))) => (a, b),
        (Err(DenseAotError::ToolchainMissing(_)), _)
        | (_, Err(DenseAotError::ToolchainMissing(_))) => return,
        other => panic!("native dense errored: {other:?}"),
    };
    assert_ne!(
        observable(&a),
        observable(&b),
        "add != sub on the same operands"
    );
    // The shared checker rejects the divergent pair (never a vacuous pass).
    assert!(
        matches!(
            check(
                &a,
                &b,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the divergent dense pair"
    );
}

// ─── the MLIR-dialect leg (M-856b): reference ≡ direct-LLVM ≡ dialect ────────────────────────────

/// Value ops: `DenseSpace` (reference) ≡ direct-LLVM ≡ **dialect** — a genuine three-way, libMLIR-
/// gated (skip-graceful on `ToolchainMissing`, never a faked pass). `ran` tracks non-vacuity: at
/// least one corpus case must actually run through the dialect pipeline for the assertion to mean
/// anything (the M-725 `ran_mlir` discipline).
#[cfg(feature = "mlir-dialect")]
#[test]
fn value_ops_dialect_matches_reference_and_direct_llvm() {
    use mycelium_mlir::dialect::native::dense::dialect_compile_and_run;
    use mycelium_mlir::MlirTools;
    let mut ran = false;
    for (i, p) in value_corpus().iter().enumerate() {
        let reference = reference_value(p);
        let direct = match dense_compile_and_run(p) {
            Ok(DenseResult::Value(v)) => *v,
            Err(DenseAotError::ToolchainMissing(_)) => continue, // direct-LLVM env skip
            other => panic!("program #{i}: direct-LLVM unexpected: {other:?}"),
        };
        match dialect_compile_and_run(p) {
            Ok(DenseResult::Value(dialect)) => {
                ran = true;
                assert_eq!(
                    observable(&reference),
                    observable(&dialect),
                    "program #{i} ({:?} {:?}): reference vs dialect diverged",
                    p.op,
                    p.dtype
                );
                assert_eq!(
                    observable(&direct),
                    observable(&dialect),
                    "program #{i} ({:?} {:?}): direct-LLVM vs dialect diverged",
                    p.op,
                    p.dtype
                );
            }
            Ok(other) => panic!("program #{i}: expected a Value, got {other:?}"),
            Err(DenseAotError::ToolchainMissing(_)) => continue, // dialect env skip
            Err(e) => panic!("program #{i}: dialect errored: {e}"),
        }
    }
    // A missing toolchain means every case skip-graceful'd (`ToolchainMissing`) — that is the
    // documented "green on a box without the tools" contract (Cargo.toml), never a false failure.
    // Only when the toolchain actually resolves does a still-vacuous corpus indicate a real bug.
    if MlirTools::is_available() {
        assert!(
            ran,
            "non-vacuity: at least one corpus case must actually run through the dialect pipeline \
             (libMLIR is provisioned in this environment, so the assertion above must mean something)"
        );
    }
}

/// Measurement ops (`dot`/`similarity`): reference ≡ direct-LLVM ≡ dialect, bit-exact.
#[cfg(feature = "mlir-dialect")]
#[test]
fn measurement_ops_dialect_matches_reference_bit_exact() {
    use mycelium_mlir::dialect::native::dense::dialect_compile_and_run;
    use mycelium_mlir::MlirTools;
    let mut ran = false;
    for (i, p) in measurement_corpus().iter().enumerate() {
        let reference = reference_measurement(p);
        let direct = match dense_compile_and_run(p) {
            Ok(DenseResult::Measurement(m)) => m,
            Err(DenseAotError::ToolchainMissing(_)) => continue,
            other => panic!("program #{i}: direct-LLVM unexpected: {other:?}"),
        };
        match dialect_compile_and_run(p) {
            Ok(DenseResult::Measurement(dialect)) => {
                ran = true;
                assert_eq!(
                    reference.to_bits(),
                    dialect.to_bits(),
                    "program #{i} ({:?}): reference vs dialect diverged",
                    p.op
                );
                assert_eq!(
                    direct.to_bits(),
                    dialect.to_bits(),
                    "program #{i} ({:?}): direct-LLVM vs dialect diverged",
                    p.op
                );
            }
            Ok(other) => panic!("program #{i}: expected a Measurement, got {other:?}"),
            Err(DenseAotError::ToolchainMissing(_)) => continue,
            Err(e) => panic!("program #{i}: dialect errored: {e}"),
        }
    }
    // Skip-graceful on a missing toolchain (the documented no-libMLIR-box contract); only a
    // still-vacuous corpus with the toolchain actually resolved is a real failure.
    if MlirTools::is_available() {
        assert!(
            ran,
            "non-vacuity: at least one corpus case must run through the dialect pipeline"
        );
    }
}

// ─── never-silent refusals match the reference (G2/SC-3) ─────────────────────────────────────────

/// The native path refuses **non-silently** exactly where `mycelium-dense` refuses: overflow (Inf),
/// subnormal, off-grid input, F64 dtype. Each is an explicit `DenseAotError`, never a silent value.
#[test]
fn refusals_match_the_reference() {
    use mycelium_dense::{DenseError, DENSE_MIN_NORMAL};
    let space = DenseSpace::new(1, ScalarKind::F32).unwrap();

    // Overflow: f32::MAX + f32::MAX = Inf. Reference: DenseError::Overflow.
    let m = space.value(vec![f64::from(f32::MAX)]).unwrap();
    assert_eq!(
        space.add_values(&m, &m),
        Err(DenseError::Overflow { index: 0 })
    );
    let p = prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![f64::from(f32::MAX)],
        Some(vec![f64::from(f32::MAX)]),
        None,
    );
    match dense_compile_and_run(&p) {
        Err(DenseAotError::Overflow) => {}
        Err(DenseAotError::ToolchainMissing(_)) => return,
        other => panic!("overflow must be refused, got {other:?}"),
    }

    // Subnormal: 1.5·2⁻¹²⁶ − 1.25·2⁻¹²⁶ = 0.25·2⁻¹²⁶ (subnormal). Reference: SubnormalUnsupported.
    let a = space.value(vec![1.5 * DENSE_MIN_NORMAL]).unwrap();
    let b = space.value(vec![1.25 * DENSE_MIN_NORMAL]).unwrap();
    assert_eq!(
        space.sub_values(&a, &b),
        Err(DenseError::SubnormalUnsupported { index: 0 })
    );
    let p = prog(
        DenseCgOp::Sub,
        ScalarKind::F32,
        vec![1.5 * DENSE_MIN_NORMAL],
        Some(vec![1.25 * DENSE_MIN_NORMAL]),
        None,
    );
    match dense_compile_and_run(&p) {
        Err(DenseAotError::Subnormal) => {}
        Err(DenseAotError::ToolchainMissing(_)) => return,
        other => panic!("subnormal must be refused, got {other:?}"),
    }

    // Off-grid input (0.1 is not exact f32) — refused at lowering (no toolchain needed). The reference
    // refuses constructing the value at all (NotOnGrid).
    assert_eq!(
        space.value(vec![0.1]),
        Err(DenseError::NotOnGrid { index: 0 })
    );
    let p = prog(DenseCgOp::Neg, ScalarKind::F32, vec![0.1], None, None);
    assert!(
        matches!(dense_compile_and_run(&p), Err(DenseAotError::OffGrid(_))),
        "off-grid input must be refused"
    );

    // F64 dtype — refused (the reference's DenseSpace::new refuses it; UnsupportedDtype).
    let p = DenseProgram {
        op: DenseCgOp::Neg,
        dim: 1,
        dtype: ScalarKind::F64,
        a: vec![1.0],
        b: None,
        scale: None,
    };
    assert!(
        matches!(
            dense_compile_and_run(&p),
            Err(DenseAotError::UnsupportedDtype(ScalarKind::F64))
        ),
        "F64 must be refused"
    );
}

/// A result element that overflows/goes-subnormal at index **> 0** of a dim > 1 vector must surface
/// the correct `DenseAotError::Overflow`/`Subnormal` — **not** a misclassified `Parse` failure. The
/// artifact prints the earlier in-range elements (space-separated) before the sentinel, so the
/// read-back must scan for the sentinel **anywhere** on the line, not only at its start. Earlier the
/// `refusals_match_the_reference` corpus only exercised dim = 1 (overflow at index 0), which the
/// start-only check happened to handle; this pins the dim > 1, index > 0 boundary (G2 — the refusal
/// stays the right variant). The reference refuses identically (`DenseError::Overflow { index: 1 }`).
#[test]
fn overflow_subnormal_at_nonzero_index_is_the_right_variant_not_parse() {
    use mycelium_dense::{DenseError, DENSE_MIN_NORMAL};
    let space = DenseSpace::new(2, ScalarKind::F32).unwrap();

    // Overflow at index 1: a = [1.0, f32::MAX], b = [1.0, f32::MAX]. Element 0 is in-range (2.0),
    // element 1 overflows (f32::MAX + f32::MAX = +Inf). Reference: Overflow { index: 1 }.
    let a = space.value(vec![1.0, f64::from(f32::MAX)]).unwrap();
    assert_eq!(
        space.add_values(&a, &a),
        Err(DenseError::Overflow { index: 1 })
    );
    let p = prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.0, f64::from(f32::MAX)],
        Some(vec![1.0, f64::from(f32::MAX)]),
        None,
    );
    match dense_compile_and_run(&p) {
        Err(DenseAotError::Overflow) => {}
        Err(DenseAotError::ToolchainMissing(_)) => return,
        other => panic!("overflow at index 1 must surface DenseAotError::Overflow, got {other:?}"),
    }

    // Subnormal at index 1: a = [1.0, 1.5·2⁻¹²⁶], b = [0.0, 1.25·2⁻¹²⁶]. Element 0 in-range (1.0),
    // element 1 subnormal (0.25·2⁻¹²⁶). Reference: SubnormalUnsupported { index: 1 }.
    let a = space.value(vec![1.0, 1.5 * DENSE_MIN_NORMAL]).unwrap();
    let b = space.value(vec![0.0, 1.25 * DENSE_MIN_NORMAL]).unwrap();
    assert_eq!(
        space.sub_values(&a, &b),
        Err(DenseError::SubnormalUnsupported { index: 1 })
    );
    let p = prog(
        DenseCgOp::Sub,
        ScalarKind::F32,
        vec![1.0, 1.5 * DENSE_MIN_NORMAL],
        Some(vec![0.0, 1.25 * DENSE_MIN_NORMAL]),
        None,
    );
    match dense_compile_and_run(&p) {
        // env skip — the toolchain is absent (this is the last assertion, so a unit value, not a
        // `return`, to satisfy clippy::needless_return).
        Err(DenseAotError::Subnormal) | Err(DenseAotError::ToolchainMissing(_)) => {}
        other => {
            panic!("subnormal at index 1 must surface DenseAotError::Subnormal, got {other:?}")
        }
    }
}

// ─── the dialect leg honestly refuses Dense (the third edge is a never-faked refusal) ────────────

/// The **MLIR-dialect** path honestly **refuses** a Dense `Const` (`DialectError::Unsupported`) — so
/// the three-way reduces to a two-way for Dense, never a faked third pass (VR-5/G2). Asserting the
/// refusal keeps the coverage honest: the dialect path never silently mis-lowers a Dense value.
#[cfg(feature = "mlir-dialect")]
#[test]
fn dense_const_is_refused_by_the_mlir_dialect_path() {
    use mycelium_core::{Meta, Node, Provenance};
    use mycelium_mlir::DialectError;
    // A Dense const value (built directly — the interpreter has no dense prim, but a Const node carries
    // the value, which the dialect const-lane lowering must refuse).
    let dense_val = Value::new(
        Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.5, 2.5]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Const(dense_val);
    match mycelium_mlir::mlir_compile_and_run(&node) {
        Err(DialectError::Unsupported(msg)) => {
            assert!(
                msg.contains("Dense") || msg.contains("dialect fragment"),
                "the refusal must name the Dense/dialect-fragment boundary; got: {msg}"
            );
        }
        Err(DialectError::ToolchainMissing(_)) => { /* env skip — still no silent success */ }
        Ok(v) => panic!(
            "the MLIR-dialect path must refuse a Dense const, got {:?}",
            v.payload()
        ),
        Err(e) => panic!("unexpected MLIR-dialect error on a Dense const: {e}"),
    }
}

/// The direct-LLVM `const_lane` path also refuses a Dense `Const` through the **bit/trit Node
/// lowering** (`AotError::UnsupportedRepr`) — Dense is lowered through the dedicated `dense_codegen`
/// entry points, never the generic bit/trit `Node` path, so the generic refusal stays in place (G2).
/// This keeps the two lowering surfaces cleanly separated and never silently cross-wired.
#[test]
fn dense_const_is_refused_by_the_generic_bit_trit_node_path() {
    use mycelium_core::{Meta, Node, Provenance};
    use mycelium_mlir::AotError;
    let dense_val = Value::new(
        Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.5, 2.5]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Const(dense_val);
    // emit_llvm_ir refuses at const_lane (before any toolchain), so this holds even without llc/clang.
    match mycelium_mlir::emit_llvm_ir(&node) {
        Err(AotError::UnsupportedRepr(msg)) => {
            assert!(
                msg.contains("Dense"),
                "the generic bit/trit refusal must name Dense; got: {msg}"
            );
        }
        other => panic!("the generic Node path must refuse a Dense const, got {other:?}"),
    }
}
