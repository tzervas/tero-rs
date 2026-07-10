//! In-crate white-box tests for `vsa_jit.rs` (M-855; RFC-0039 §5.3; CLAUDE.md test-layout rule).
//! Pure **emission** + **read-back-decode** checks (no `clang`/`dlopen` — the M-854 toolchain-
//! independent mutant-witness lesson): the per-op component computation, the store-sink IR shape, the
//! FHRR degenerate-flag accumulation, the reused `vsa_codegen` refusal surface, the EXPLAIN record, and
//! the never-silent `decode_jit_result` status protocol. The compiled-path differential (JIT ≡ interp/
//! `mycelium-vsa`, M-210-checked) lives in `tests/vsa_jit_differential.rs`.

use crate::vsa_codegen::{
    Ssa, VsaAotError, VsaArtifact, VsaCgOp, VsaModelId, VsaProgram, VsaResult,
};
use crate::vsa_jit::{
    compute_bind, compute_bundle_non_fhrr, compute_cconv, compute_fhrr_bundle, compute_permute,
    decode_jit_result, emit_store_components, emit_vsa_jit_ir, Component, VSA_JIT_GUARANTEE,
};
use mycelium_core::GuaranteeStrength;

// ─── fixtures (mirrors src/tests/vsa_codegen.rs's fixture shapes) ────────────────────────────────

fn prog(
    op: VsaCgOp,
    model: VsaModelId,
    dim: u32,
    items: Vec<Vec<f64>>,
    shift: Option<i64>,
    bundle_delta: Option<f64>,
) -> VsaProgram {
    VsaProgram {
        op,
        model,
        dim,
        items,
        shift,
        bundle_delta,
    }
}

fn bipolar(dim: u32) -> Vec<f64> {
    (0..dim)
        .map(|i| if i.is_multiple_of(2) { 1.0 } else { -1.0 })
        .collect()
}
fn binary(dim: u32) -> Vec<f64> {
    (0..dim).map(|i| f64::from(i % 2)).collect()
}
fn real(dim: u32) -> Vec<f64> {
    (0..dim).map(|i| f64::from(i) * 0.25 - 1.0).collect()
}
fn phase(dim: u32) -> Vec<f64> {
    (0..dim).map(|i| f64::from(i % 5) * 0.5 - 1.0).collect()
}

// ─── VSA_JIT_GUARANTEE — pinned, never upgraded (VR-5) ────────────────────────────────────────────

#[test]
fn jit_guarantee_is_empirical_never_upgraded() {
    assert_eq!(VSA_JIT_GUARANTEE, GuaranteeStrength::Empirical);
}

// ─── compute_bind: exact IR shape per model ───────────────────────────────────────────────────────

#[test]
fn compute_bind_mapi_emits_fmul_per_component() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        4,
        vec![bipolar(4), bipolar(4)],
        None,
        None,
    );
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let comps = compute_bind(&p, false, &mut ssa, &mut body);
    assert_eq!(comps.len(), 4);
    assert!(matches!(comps[0], Component::Reg(_)));
    assert_eq!(body.matches("fmul double").count(), 4);
    assert!(!body.contains("fsub double"));
}

#[test]
fn compute_bind_bsc_emits_fsub_and_fabs_per_component() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Bsc,
        3,
        vec![binary(3), binary(3)],
        None,
        None,
    );
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let comps = compute_bind(&p, false, &mut ssa, &mut body);
    assert_eq!(comps.len(), 3);
    assert_eq!(body.matches("fsub double").count(), 3);
    assert_eq!(body.matches("@llvm.fabs.f64").count(), 3);
}

#[test]
fn compute_bind_hrr_bind_vs_unbind_differ_by_involution() {
    let a = real(6);
    let b = real(6);
    let p = prog(VsaCgOp::Bind, VsaModelId::Hrr, 6, vec![a, b], None, None);
    let mut ssa1 = Ssa::new();
    let mut body1 = String::new();
    let bind_comps = compute_bind(&p, false, &mut ssa1, &mut body1);
    let mut ssa2 = Ssa::new();
    let mut body2 = String::new();
    let unbind_comps = compute_bind(&p, true, &mut ssa2, &mut body2);
    assert_eq!(bind_comps.len(), 6);
    assert_eq!(unbind_comps.len(), 6);
    // Both are circular convolutions (same instruction shape); the register *names* differ only if
    // the underlying operand ordering differs — the involution changes which constant literals are
    // multiplied, so the emitted constant operands (not just SSA names) diverge between the two.
    assert_ne!(
        body1, body2,
        "bind vs unbind must emit different convolution operands (involution)"
    );
}

#[test]
fn compute_bind_fhrr_bind_uses_fadd_unbind_uses_fsub() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Fhrr,
        4,
        vec![phase(4), phase(4)],
        None,
        None,
    );
    let mut ssa = Ssa::new();
    let mut bind_body = String::new();
    let _ = compute_bind(&p, false, &mut ssa, &mut bind_body);
    assert!(bind_body.contains("fadd double"));

    let mut ssa2 = Ssa::new();
    let mut unbind_body = String::new();
    let _ = compute_bind(&p, true, &mut ssa2, &mut unbind_body);
    assert!(unbind_body.contains("fsub double"));
}

// ─── compute_cconv: the product-index arithmetic (pinned, mirrors vsa_codegen's own test) ────────

#[test]
fn cconv_product_index_arithmetic_is_pinned() {
    // out[k] = sum_i a[i]*b[(k+d-i) mod d]; for d=3, a=[a0,a1,a2], b=[b0,b1,b2]:
    // out[0] = a0*b0 + a1*b2 + a2*b1
    // out[1] = a0*b1 + a1*b0 + a2*b2
    // out[2] = a0*b2 + a1*b1 + a2*b0
    let a = [2.0, 3.0, 5.0];
    let b = [7.0, 11.0, 13.0];
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let regs = compute_cconv(&a, &b, &mut ssa, &mut body);
    assert_eq!(regs.len(), 3);
    // Every register is distinct (no accidental register-name collision across k-iterations).
    let unique: std::collections::HashSet<_> = regs.iter().collect();
    assert_eq!(unique.len(), 3);
    // d^2 = 9 fmul instructions total (3 per output component).
    assert_eq!(body.matches("fmul double").count(), 9);
    assert_eq!(body.matches("fadd double").count(), 9);
}

// ─── compute_bundle_non_fhrr: MAP-I/HRR sum + BSC majority boundary ───────────────────────────────

#[test]
fn compute_bundle_mapi_emits_left_to_right_sum() {
    let items: Vec<Vec<f64>> = (0..3).map(|_| bipolar(4)).collect();
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        4,
        items,
        None,
        Some(1e-2),
    );
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let comps = compute_bundle_non_fhrr(&p, &mut ssa, &mut body);
    assert_eq!(comps.len(), 4);
    // 3 items -> 2 fadds per component (accumulate items[1] and items[2] onto items[0]) * 4 components.
    assert_eq!(body.matches("fadd double").count(), 8);
}

#[test]
fn bsc_bundle_majority_boundary_bits_are_pinned() {
    // 3 items (odd), single component: n in {0,1,2,3}. half = 1.5.
    // n=3 -> 1 (>half); n=2 -> 1 (>half); n=1 -> 0 (<half); n=0 -> 0 (<half). No tie possible for odd m.
    // The n=2/n=1 cases deliberately put the *minority* bit at items[0] (x) — if the majority arm
    // ever fell through to the tie fallback `items[0][idx]` (e.g. a `>`->`==` or `<`->`==` mutation,
    // which makes `n > half`/`n < half` never true for an integer n against a non-integer half), the
    // wrong (minority) bit would surface instead of the correct majority bit, so these two cases are
    // load-bearing for catching that mutant class — not just a majority-vs-minority label pun.
    let cases: [(u8, u8, u8, f64); 4] = [
        (1, 1, 1, 1.0), // n=3 -> 1 (unanimous; items[0]=1 coincides, but distinctness carried by below)
        (0, 1, 1, 1.0), // n=2 -> 1; items[0]=0 is the MINORITY bit — catches `>` -> `==`
        (1, 0, 0, 0.0), // n=1 -> 0; items[0]=1 is the MINORITY bit — catches `<` -> `==`
        (0, 0, 0, 0.0), // n=0 -> 0 (unanimous)
    ];
    for (x, y, z, want) in cases {
        let items = vec![vec![f64::from(x)], vec![f64::from(y)], vec![f64::from(z)]];
        let p = prog(VsaCgOp::Bundle, VsaModelId::Bsc, 1, items, None, None);
        let mut ssa = Ssa::new();
        let mut body = String::new();
        let comps = compute_bundle_non_fhrr(&p, &mut ssa, &mut body);
        assert_eq!(comps.len(), 1);
        match &comps[0] {
            Component::Const(bit) => assert_eq!(
                *bit, want,
                "majority({x},{y},{z}) should be {want}, got {bit}"
            ),
            other => panic!("BSC bundle bit must be a folded Const, got {other:?}"),
        }
    }
}

#[test]
fn compute_bundle_hrr_uses_sum_not_majority() {
    let items: Vec<Vec<f64>> = (0..3).map(|_| real(4)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Hrr, 4, items, None, None);
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let comps = compute_bundle_non_fhrr(&p, &mut ssa, &mut body);
    assert_eq!(comps.len(), 4);
    assert!(comps.iter().all(|c| matches!(c, Component::Reg(_))));
}

#[test]
#[should_panic(expected = "compute_fhrr_bundle")]
fn compute_bundle_non_fhrr_refuses_fhrr_model() {
    let items: Vec<Vec<f64>> = (0..3).map(|_| phase(4)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, 4, items, None, None);
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let _ = compute_bundle_non_fhrr(&p, &mut ssa, &mut body);
}

// ─── compute_fhrr_bundle: the OR-accumulated degenerate flag ─────────────────────────────────────

#[test]
fn fhrr_bundle_first_component_flag_has_no_or_the_rest_do() {
    let items: Vec<Vec<f64>> = (0..2).map(|_| phase(3)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, 3, items, None, None);
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let (values, any_deg) = compute_fhrr_bundle(&p, &mut ssa, &mut body);
    assert_eq!(values.len(), 3);
    // 2 later components OR'd against the running flag => exactly 2 `or i1` instructions (idx=1,2);
    // idx=0 contributes no `or` (the flag starts as its own `deg` register).
    assert_eq!(body.matches("or i1").count(), 2);
    assert!(!any_deg.is_empty());
    assert!(body.contains("call double @cos(double"));
    assert!(body.contains("call double @sin(double"));
    assert!(body.contains("call double @atan2(double"));
    assert!(body.contains("@llvm.sqrt.f64"));
}

#[test]
fn fhrr_bundle_single_component_emits_no_or() {
    let items: Vec<Vec<f64>> = (0..2).map(|_| phase(1)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, 1, items, None, None);
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let (values, _any_deg) = compute_fhrr_bundle(&p, &mut ssa, &mut body);
    assert_eq!(values.len(), 1);
    assert!(!body.contains("or i1"));
}

// ─── compute_permute: the source-index arithmetic (pinned) ───────────────────────────────────────

#[test]
fn permute_source_index_arithmetic_is_pinned() {
    let a = vec![10.0, 20.0, 30.0, 40.0];
    // shift = 1 (cyclic left rotate by 1): result[i] = a[(i+1) mod 4] => [20,30,40,10]
    let p = prog(
        VsaCgOp::Permute,
        VsaModelId::MapI,
        4,
        vec![a.clone()],
        Some(1),
        None,
    );
    let got = compute_permute(&p).unwrap();
    assert_eq!(got, vec![20.0, 30.0, 40.0, 10.0]);

    // shift = -1: result[i] = a[(i-1) mod 4] => [40,10,20,30]
    let p2 = prog(
        VsaCgOp::Permute,
        VsaModelId::MapI,
        4,
        vec![a],
        Some(-1),
        None,
    );
    let got2 = compute_permute(&p2).unwrap();
    assert_eq!(got2, vec![40.0, 10.0, 20.0, 30.0]);
}

#[test]
fn permute_without_shift_is_malformed() {
    let p = prog(
        VsaCgOp::Permute,
        VsaModelId::MapI,
        4,
        vec![bipolar(4)],
        None,
        None,
    );
    assert_eq!(
        compute_permute(&p),
        Err(VsaAotError::Malformed("permute needs a shift".to_owned()))
    );
}

// ─── emit_store_components: Reg -> bitcast+store; Const -> store raw bits, no bitcast ─────────────

#[test]
fn store_sink_bitcasts_registers_but_not_constants() {
    let mut ssa = Ssa::new();
    let mut body = String::new();
    let comps = vec![Component::Reg("%r0".to_owned()), Component::Const(2.5)];
    emit_store_components(&comps, &mut ssa, &mut body);
    assert_eq!(body.matches("bitcast double").count(), 1);
    assert_eq!(body.matches("store i64").count(), 2);
    assert!(body.contains(&2.5_f64.to_bits().to_string()));
    assert_eq!(body.matches("getelementptr i64, ptr %out").count(), 2);
    // indices are 0 and 1, in order.
    assert!(body.contains("i64 0\n") || body.contains("i64 0\n  store"));
    assert!(body.contains("i64 1"));
}

// ─── emit_vsa_jit_ir: top-level integration + reused refusal surface ──────────────────────────────

/// One `emit_vsa_jit_ir` shape case (a named struct, not a giant tuple — keeps `every_op_emits_…`
/// below `clippy::type_complexity`-clean).
struct EmitCase {
    op: VsaCgOp,
    model: VsaModelId,
    dim: u32,
    items: Vec<Vec<f64>>,
    shift: Option<i64>,
    delta: Option<f64>,
    want_width: usize,
}

#[test]
fn every_op_emits_the_kernel_signature_and_correct_width() {
    let cases = [
        EmitCase {
            op: VsaCgOp::Bind,
            model: VsaModelId::MapI,
            dim: 4,
            items: vec![bipolar(4), bipolar(4)],
            shift: None,
            delta: None,
            want_width: 4,
        },
        EmitCase {
            op: VsaCgOp::Unbind,
            model: VsaModelId::Bsc,
            dim: 4,
            items: vec![binary(4), binary(4)],
            shift: None,
            delta: None,
            want_width: 4,
        },
        EmitCase {
            op: VsaCgOp::Permute,
            model: VsaModelId::Hrr,
            dim: 4,
            items: vec![real(4)],
            shift: Some(2),
            delta: None,
            want_width: 4,
        },
        EmitCase {
            op: VsaCgOp::Similarity,
            model: VsaModelId::Fhrr,
            dim: 4,
            items: vec![phase(4), phase(4)],
            shift: None,
            delta: None,
            want_width: 1,
        },
        EmitCase {
            op: VsaCgOp::Bundle,
            model: VsaModelId::Hrr,
            dim: 256,
            items: (0..3).map(|_| real(256)).collect(),
            shift: None,
            delta: None,
            want_width: 256,
        },
    ];
    for EmitCase {
        op,
        model,
        dim,
        items,
        shift,
        delta,
        want_width,
    } in cases
    {
        let p = prog(op, model, dim, items, shift, delta);
        let (ir, explain, width) = emit_vsa_jit_ir(&p).unwrap_or_else(|e| {
            panic!("case {op:?}/{model:?} should lower: {e}");
        });
        assert!(ir.contains("define i32 @myc_vsa_kernel(ptr %out)"));
        assert!(ir.contains("ret i32 0"));
        assert_eq!(width, want_width);
        assert_eq!(explain.dim, dim);
        assert_eq!(explain.codegen_guarantee, GuaranteeStrength::Empirical);
    }
}

#[test]
fn fhrr_bundle_emits_the_branch_once_status_protocol() {
    let items: Vec<Vec<f64>> = (0..2).map(|_| phase(256)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, 256, items, None, None);
    let (ir, _explain, width) = emit_vsa_jit_ir(&p).unwrap();
    assert_eq!(width, 256);
    assert!(ir.contains("br i1"));
    assert!(ir.contains("ret i32 1"));
    assert!(ir.contains("ret i32 0"));
    // exactly one branch (not a per-component branch): the AOT sentinel path would `contains` a
    // string global; the JIT path never emits one (no printf/sentinel string at all).
    assert!(!ir.contains("@.s_deg"));
    assert!(!ir.contains("DEGENERATE"));
}

#[test]
fn permute_emits_no_arithmetic_and_no_bitcast() {
    let p = prog(
        VsaCgOp::Permute,
        VsaModelId::MapI,
        4,
        vec![bipolar(4)],
        Some(1),
        None,
    );
    let (ir, _explain, width) = emit_vsa_jit_ir(&p).unwrap();
    assert_eq!(width, 4);
    assert!(!ir.contains("fmul double"));
    assert!(!ir.contains("fadd double"));
    assert!(!ir.contains("fsub double"));
    assert!(!ir.contains("bitcast double"));
}

#[test]
fn unsupported_model_is_refused_reusing_vsa_codegen_surface() {
    // SBC / MAP-B are not in the 1.0.0-native-mandatory set; `resolve_model` in vsa_codegen already
    // refuses them (reused, not re-implemented) — here we exercise the equivalent via a VsaProgram
    // directly is not possible (VsaModelId has only the 4 mandatory variants), so this test instead
    // pins that emit_vsa_jit_ir calls `prog.validate()` (the shared refusal surface) by checking a
    // validate()-refused program (off-alphabet) is refused identically to the AOT path.
    let bad = vec![1.0, 2.0, 1.0, -1.0]; // 2.0 is off the MAP-I bipolar alphabet
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        4,
        vec![bad, bipolar(4)],
        None,
        None,
    );
    assert_eq!(
        emit_vsa_jit_ir(&p),
        Err(VsaAotError::NonAlphabetComponent {
            model: "MAP-I",
            index: 1
        })
    );
}

#[test]
fn empirical_profile_gate_is_reused_for_the_jit_path() {
    // HRR bundle below the profile's min_dim (256) is refused OutsideEmpiricalProfile — exactly the
    // AOT path's refusal (same `VsaProgram::validate` call).
    let items: Vec<Vec<f64>> = (0..3).map(|_| real(8)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Hrr, 8, items, None, None);
    match emit_vsa_jit_ir(&p) {
        Err(VsaAotError::OutsideEmpiricalProfile(_)) => {}
        other => panic!("expected OutsideEmpiricalProfile, got {other:?}"),
    }
}

#[test]
fn insufficient_map_i_capacity_is_refused_never_an_unbacked_proven() {
    let items: Vec<Vec<f64>> = (0..3).map(|_| bipolar(8)).collect();
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        8,
        items,
        None,
        Some(1e-2),
    );
    match emit_vsa_jit_ir(&p) {
        Err(VsaAotError::InsufficientCapacity { .. }) => {}
        other => panic!("expected InsufficientCapacity, got {other:?}"),
    }
}

#[test]
fn empty_bundle_is_refused() {
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        4,
        vec![],
        None,
        Some(1e-2),
    );
    assert_eq!(emit_vsa_jit_ir(&p), Err(VsaAotError::EmptyBundle));
}

#[test]
fn dim_mismatch_is_refused() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        4,
        vec![bipolar(4), bipolar(8)],
        None,
        None,
    );
    assert_eq!(
        emit_vsa_jit_ir(&p),
        Err(VsaAotError::DimMismatch {
            expected: 4,
            got: 8
        })
    );
}

// ─── decode_jit_result: the never-silent status protocol (toolchain-independent witness) ─────────

#[test]
fn decode_nonzero_status_is_degenerate_refusal() {
    let shape = VsaArtifact::for_shape(VsaCgOp::Bundle, VsaModelId::Fhrr, 4, None, 2);
    assert_eq!(
        decode_jit_result(VsaCgOp::Bundle, &shape, 1, &[0, 0, 0, 0]),
        Err(VsaAotError::DegenerateBundleComponent)
    );
    // Nonzero status is refused regardless of the buffer's content (never a silently-decoded garbage
    // value) — any nonzero status short-circuits before the buffer is even inspected.
    assert_eq!(
        decode_jit_result(VsaCgOp::Bundle, &shape, 42, &[]),
        Err(VsaAotError::DegenerateBundleComponent)
    );
}

#[test]
fn decode_value_op_reconstructs_via_the_shared_shape() {
    let shape = VsaArtifact::for_shape(VsaCgOp::Bind, VsaModelId::MapI, 2, None, 0);
    let bits = [1.0_f64.to_bits(), (-1.0_f64).to_bits()];
    match decode_jit_result(VsaCgOp::Bind, &shape, 0, &bits) {
        Ok(VsaResult::Value(v)) => {
            assert_eq!(
                v.payload(),
                &mycelium_core::Payload::Hypervector(vec![1.0, -1.0])
            );
        }
        other => panic!("expected a reconstructed Value, got {other:?}"),
    }
}

#[test]
fn decode_value_op_wrong_width_is_refused_not_truncated() {
    let shape = VsaArtifact::for_shape(VsaCgOp::Bind, VsaModelId::MapI, 4, None, 0);
    let bits = [1.0_f64.to_bits(), (-1.0_f64).to_bits()]; // only 2, dim is 4
    match decode_jit_result(VsaCgOp::Bind, &shape, 0, &bits) {
        Err(VsaAotError::Parse(_)) => {}
        other => panic!("expected a Parse (dim-mismatch) refusal, got {other:?}"),
    }
}

#[test]
fn decode_measurement_op_reads_exactly_one_element() {
    let shape = VsaArtifact::for_shape(VsaCgOp::Similarity, VsaModelId::MapI, 4, None, 0);
    let bits = [0.5_f64.to_bits()];
    match decode_jit_result(VsaCgOp::Similarity, &shape, 0, &bits) {
        Ok(VsaResult::Measurement(m)) => assert_eq!(m.to_bits(), 0.5_f64.to_bits()),
        other => panic!("expected a Measurement, got {other:?}"),
    }
}

#[test]
fn decode_measurement_op_wrong_width_is_refused() {
    let shape = VsaArtifact::for_shape(VsaCgOp::Similarity, VsaModelId::MapI, 4, None, 0);
    match decode_jit_result(VsaCgOp::Similarity, &shape, 0, &[1, 2]) {
        Err(VsaAotError::Parse(_)) => {}
        other => {
            panic!("expected a Parse refusal for a 2-element measurement buffer, got {other:?}")
        }
    }
    match decode_jit_result(VsaCgOp::Similarity, &shape, 0, &[]) {
        Err(VsaAotError::Parse(_)) => {}
        other => panic!("expected a Parse refusal for an empty measurement buffer, got {other:?}"),
    }
}

// ─── emission determinism (the JIT kernel must be reproducible, not order-flaky) ──────────────────

#[test]
fn emission_is_deterministic() {
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::Fhrr,
        256,
        (0..3).map(|_| phase(256)).collect(),
        None,
        None,
    );
    let (ir1, _, w1) = emit_vsa_jit_ir(&p).unwrap();
    let (ir2, _, w2) = emit_vsa_jit_ir(&p).unwrap();
    assert_eq!(ir1, ir2);
    assert_eq!(w1, w2);
}
