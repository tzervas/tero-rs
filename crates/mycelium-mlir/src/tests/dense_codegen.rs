//! In-crate white-box tests for `dense_codegen.rs` (M-853; RFC-0039 §5.1; CLAUDE.md test-layout
//! rule). These are pure **emission** + **logic** checks (no toolchain): the per-element
//! side-condition validation, the never-silent refusals, the inspectable `DenseExplain` /
//! dumpable IR comment (RFC-0004 §6), the honest reference-vs-codegen guarantee split (VR-5), and
//! that the emitted IR carries the explicit per-element ops (no opaque pass). The compiled-path
//! differential (native ≡ `mycelium-dense`, M-210-checked, mutant-witnessed) lives in
//! `tests/dense_differential.rs`.

use crate::dense_codegen::{
    emit_dense_llvm_ir, on_grid, op_citation, round_f32_to_bf16, DenseAotError, DenseCgOp,
    DenseExplain, DenseProgram, DENSE_CODEGEN_GUARANTEE,
};
use mycelium_core::{GuaranteeStrength, PhysicalLayout, ScalarKind};

// ─── fixtures ────────────────────────────────────────────────────────────────────────────────────

/// A well-formed program for `op` over `dtype` with the given operands (small, on-grid).
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

/// A canonical well-formed F32 program per op (used to assert emission + EXPLAIN shape).
fn canonical(op: DenseCgOp) -> DenseProgram {
    match op {
        DenseCgOp::Add => prog(
            op,
            ScalarKind::F32,
            vec![1.5, 2.5],
            Some(vec![0.25, -1.0]),
            None,
        ),
        DenseCgOp::Sub => prog(
            op,
            ScalarKind::F32,
            vec![3.0, 0.5],
            Some(vec![1.0, 0.25]),
            None,
        ),
        DenseCgOp::Neg => prog(op, ScalarKind::F32, vec![1.5, -0.625, 0.0], None, None),
        DenseCgOp::Scale => prog(op, ScalarKind::F32, vec![1.5, -2.0], None, Some(2.0)),
        DenseCgOp::Dot => prog(
            op,
            ScalarKind::F32,
            vec![1.0, 2.0, -1.0],
            Some(vec![0.5, -1.0, 2.0]),
            None,
        ),
        DenseCgOp::Similarity => prog(
            op,
            ScalarKind::F32,
            vec![1.0, 2.0, -1.0],
            Some(vec![0.5, -1.0, 2.0]),
            None,
        ),
    }
}

const ALL_OPS: [DenseCgOp; 6] = [
    DenseCgOp::Add,
    DenseCgOp::Sub,
    DenseCgOp::Neg,
    DenseCgOp::Scale,
    DenseCgOp::Dot,
    DenseCgOp::Similarity,
];

// ─── op metadata (mirrors DenseSpace) ────────────────────────────────────────────────────────────

/// The reference guarantee per op mirrors `DenseSpace::op_guarantee`: `neg` Exact;
/// `add`/`sub`/`scale` Proven; `dot`/`similarity` are measurements (None).
#[test]
fn reference_guarantee_mirrors_the_dense_surface() {
    assert_eq!(
        DenseCgOp::Neg.reference_guarantee(),
        Some(GuaranteeStrength::Exact)
    );
    for op in [DenseCgOp::Add, DenseCgOp::Sub, DenseCgOp::Scale] {
        assert_eq!(
            op.reference_guarantee(),
            Some(GuaranteeStrength::Proven),
            "{op:?} must be Proven (the rounding bound)"
        );
    }
    for op in [DenseCgOp::Dot, DenseCgOp::Similarity] {
        assert_eq!(
            op.reference_guarantee(),
            None,
            "{op:?} is a bare measurement (no Meta)"
        );
        assert!(!op.is_value_op(), "{op:?} is not a value op");
    }
    for op in [
        DenseCgOp::Add,
        DenseCgOp::Sub,
        DenseCgOp::Neg,
        DenseCgOp::Scale,
    ] {
        assert!(op.is_value_op(), "{op:?} produces a Value");
    }
}

/// Op names match the `mycelium-dense` operation keys (so provenance / EXPLAIN are never anonymous).
#[test]
fn op_names_match_the_dense_keys() {
    assert_eq!(DenseCgOp::Add.name(), "dense.add");
    assert_eq!(DenseCgOp::Sub.name(), "dense.sub");
    assert_eq!(DenseCgOp::Neg.name(), "dense.neg");
    assert_eq!(DenseCgOp::Scale.name(), "dense.scale");
    assert_eq!(DenseCgOp::Dot.name(), "dense.dot");
    assert_eq!(DenseCgOp::Similarity.name(), "dense.similarity");
}

// ─── the inspectable EXPLAIN record + dumpable IR comment (RFC-0004 §6 — no black box) ────────────

/// Every op emits the dumpable EXPLAIN comment (op, dim, dtype, guarantees, quant) — never a hidden
/// lowering (G2). The codegen-guarantee is always Empirical; the value ops carry the reference tag.
#[test]
fn every_op_emits_the_dumpable_explain_comment() {
    for op in ALL_OPS {
        let (ir, explain) = emit_dense_llvm_ir(&canonical(op)).expect("canonical program lowers");
        assert!(
            ir.contains(&format!("; dense {}", op.name())),
            "{op:?} IR must carry the dumpable EXPLAIN comment:\n{ir}"
        );
        assert!(
            ir.contains("codegen-guarantee=Empirical"),
            "{op:?} IR must record the Empirical codegen guarantee (VR-5):\n{ir}"
        );
        assert!(
            ir.contains("quant=un-quantized"),
            "{op:?} IR must record the un-quantized status (E20-1 gate):\n{ir}"
        );
        // EXPLAIN fields match the op.
        assert_eq!(explain.op, op.name());
        assert_eq!(explain.codegen_guarantee, GuaranteeStrength::Empirical);
        assert_eq!(explain.reference_guarantee, op.reference_guarantee());
    }
}

/// The codegen-correctness guarantee is **Empirical** (VR-5 — the differential + mutant-witness are
/// the basis; no proof object linked here). Pinned so it cannot silently upgrade past its basis.
#[test]
fn codegen_guarantee_is_empirical_never_upgraded() {
    assert_eq!(DENSE_CODEGEN_GUARANTEE, GuaranteeStrength::Empirical);
    for op in ALL_OPS {
        let (_, explain) = emit_dense_llvm_ir(&canonical(op)).unwrap();
        assert_eq!(
            explain.codegen_guarantee,
            GuaranteeStrength::Empirical,
            "{op:?} codegen guarantee must stay Empirical (VR-5)"
        );
    }
}

/// A value op records the inspectable `Meta.physical = DenseArray` schedule (DN-01; the
/// schedule-as-metadata discipline); a measurement op has no physical schedule (bare f64 output).
#[test]
fn value_ops_record_the_dense_array_schedule() {
    for op in [
        DenseCgOp::Add,
        DenseCgOp::Sub,
        DenseCgOp::Neg,
        DenseCgOp::Scale,
    ] {
        let (_, explain) = emit_dense_llvm_ir(&canonical(op)).unwrap();
        assert_eq!(
            explain.physical,
            Some(PhysicalLayout::DenseArray),
            "{op:?} must record the DenseArray schedule"
        );
    }
    for op in [DenseCgOp::Dot, DenseCgOp::Similarity] {
        let (_, explain) = emit_dense_llvm_ir(&canonical(op)).unwrap();
        assert_eq!(
            explain.physical, None,
            "{op:?} (measurement) has no physical schedule"
        );
    }
}

/// The per-op relative ε in the EXPLAIN matches the cited Higham bound exactly: F32 `add`/`sub`/`scale`
/// carry 2⁻²⁴, BF16 carries 2⁻⁸ + 2⁻²³, and `neg`/measurements carry 0 (no rounding).
#[test]
fn explain_rel_eps_matches_the_cited_bound() {
    let f32_add = emit_dense_llvm_ir(&prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.0],
        Some(vec![1.0]),
        None,
    ))
    .unwrap()
    .1;
    assert_eq!(f32_add.rel_eps, 2f64.powi(-24), "F32 op ε = 2⁻²⁴");

    let bf16_add = emit_dense_llvm_ir(&prog(
        DenseCgOp::Add,
        ScalarKind::Bf16,
        vec![1.0],
        Some(vec![1.0]),
        None,
    ))
    .unwrap()
    .1;
    assert_eq!(
        bf16_add.rel_eps,
        2f64.powi(-8) + 2f64.powi(-23),
        "BF16 op ε = 2⁻⁸ + 2⁻²³"
    );

    let neg = emit_dense_llvm_ir(&canonical(DenseCgOp::Neg)).unwrap().1;
    assert_eq!(neg.rel_eps, 0.0, "neg is exact (no ε)");
    let dot = emit_dense_llvm_ir(&canonical(DenseCgOp::Dot)).unwrap().1;
    assert_eq!(dot.rel_eps, 0.0, "dot is a measurement (no ε)");
}

// ─── the IR transcode shape (no opaque pass — §6) ────────────────────────────────────────────────

/// `add`/`sub` emit the explicit per-element float op (`fadd`/`fsub`) and the never-silent
/// side-condition trap (`SUBNORMAL`/`OVERFLOW` sentinels + the branch). One op per element.
#[test]
fn add_sub_emit_explicit_elementwise_ir_with_traps() {
    for (op, fop) in [(DenseCgOp::Add, "fadd"), (DenseCgOp::Sub, "fsub")] {
        let p = canonical(op);
        let dim = p.dim as usize;
        let (ir, _) = emit_dense_llvm_ir(&p).unwrap();
        assert!(
            ir.matches(&format!("{fop} float")).count() == dim,
            "{op:?} must emit one {fop} per element (one op per element — §6):\n{ir}"
        );
        // Never-silent side-condition trap *branches* (subnormal / overflow), matching DenseError —
        // the `getelementptr … @.s_…` *use* (not just the always-present header declaration).
        assert!(
            ir.contains("@.s_sub, i64 0, i64 0") && ir.contains("@.s_ovf, i64 0, i64 0"),
            "{op:?} must emit the never-silent subnormal/overflow trap branches (G2):\n{ir}"
        );
        assert!(
            ir.contains("@llvm.fabs.f32") && ir.contains("br i1"),
            "{op:?} must emit the side-condition check + branch IR:\n{ir}"
        );
    }
}

/// `neg` emits an `fneg` per element and **no rounding / no side-condition trap** (it is exact on the
/// symmetric grid — a negated on-grid value is on-grid and finite). The sentinel globals are always
/// *declared* in the header; the marker that `neg` emits **no trap** is that its body never *uses*
/// them (no `getelementptr … @.s_ovf`/`@.s_sub` trap branch) and emits no `br i1` / `fabs`.
#[test]
fn neg_emits_exact_fneg_with_no_trap() {
    let p = canonical(DenseCgOp::Neg);
    let dim = p.dim as usize;
    let (ir, _) = emit_dense_llvm_ir(&p).unwrap();
    assert_eq!(
        ir.matches("fneg float").count(),
        dim,
        "neg must emit one fneg per element:\n{ir}"
    );
    // The trap *use* of a sentinel is a `getelementptr … @.s_…` inside a `printf` call — neg emits none.
    assert!(
        !ir.contains("@.s_ovf, i64 0, i64 0") && !ir.contains("@.s_sub, i64 0, i64 0"),
        "neg is exact — it must NOT emit subnormal/overflow trap branches:\n{ir}"
    );
    assert!(
        !ir.contains("@llvm.fabs.f32") && !ir.contains("br i1"),
        "neg must emit no side-condition check / branch (it is exact):\n{ir}"
    );
}

/// `scale` emits an `fmul` per element (the scalar times each element) plus the side-condition trap.
#[test]
fn scale_emits_fmul_per_element() {
    let p = canonical(DenseCgOp::Scale);
    let dim = p.dim as usize;
    let (ir, _) = emit_dense_llvm_ir(&p).unwrap();
    assert_eq!(
        ir.matches("fmul float").count(),
        dim,
        "scale must emit one fmul per element:\n{ir}"
    );
    assert!(ir.contains("@.s_ovf"), "scale must emit the overflow trap");
}

/// `dot` accumulates in `f64` (`fmul double` + `fadd double` per element); `similarity` adds the
/// sqrt + fdiv + the zero-norm guard. Both print a single f64 measurement.
#[test]
fn dot_and_similarity_emit_f64_reductions() {
    let dot = emit_dense_llvm_ir(&canonical(DenseCgOp::Dot)).unwrap().0;
    assert!(
        dot.contains("fmul double") && dot.contains("fadd double"),
        "dot must accumulate in f64:\n{dot}"
    );
    let sim = emit_dense_llvm_ir(&canonical(DenseCgOp::Similarity))
        .unwrap()
        .0;
    assert!(
        sim.contains("@llvm.sqrt.f64") && sim.contains("fdiv double"),
        "similarity must emit the norm sqrt + division:\n{sim}"
    );
    assert!(
        sim.contains("select i1") && sim.contains("fcmp oeq double"),
        "similarity must emit the never-silent zero-norm guard:\n{sim}"
    );
}

/// BF16 ops emit the round-to-nearest-even bit-twiddle (`bitcast … to i32`, `add … 32767`, the shift
/// chain) that mirrors `round_f32_to_bf16` — the grid rounding is explicit IR, not a hidden coercion.
#[test]
fn bf16_op_emits_the_round_to_nearest_even_bit_twiddle() {
    let p = prog(
        DenseCgOp::Add,
        ScalarKind::Bf16,
        vec![1.5],
        Some(vec![0.5]),
        None,
    );
    let (ir, _) = emit_dense_llvm_ir(&p).unwrap();
    assert!(
        ir.contains("bitcast float") && ir.contains("32767"),
        "BF16 rounding must emit the explicit round-to-nearest-even bit twiddle (§6):\n{ir}"
    );
    // F32 add does NOT round (no bit twiddle).
    let f32_ir = emit_dense_llvm_ir(&prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.5],
        Some(vec![0.5]),
        None,
    ))
    .unwrap()
    .0;
    assert!(
        !f32_ir.contains("32767"),
        "F32 add must not emit the BF16 rounding twiddle:\n{f32_ir}"
    );
}

// ─── emission determinism ────────────────────────────────────────────────────────────────────────

#[test]
fn emission_is_deterministic() {
    for op in ALL_OPS {
        let p = canonical(op);
        assert_eq!(
            emit_dense_llvm_ir(&p).map(|(ir, _)| ir),
            emit_dense_llvm_ir(&p).map(|(ir, _)| ir),
            "{op:?} emission must be deterministic"
        );
    }
}

// ─── never-silent refusals (G2) — the validation half, no toolchain needed ───────────────────────

/// F16/F64 dtypes are refused (`UnsupportedDtype`) — matches `DenseSpace::new`. Never a silent
/// coercion to F32 (G2). The refusal is at lowering (no toolchain needed).
#[test]
fn unsupported_dtypes_are_refused() {
    for dt in [ScalarKind::F16, ScalarKind::F64] {
        let p = prog(DenseCgOp::Neg, dt, vec![1.0], None, None);
        match emit_dense_llvm_ir(&p) {
            Err(DenseAotError::UnsupportedDtype(got)) => assert_eq!(got, dt),
            other => panic!("dtype {dt:?} must be refused (UnsupportedDtype), got {other:?}"),
        }
    }
}

/// A non-finite or off-grid input element is refused (matches `DenseError::NonFinite`/`NotOnGrid`) —
/// never silently re-rounded. Data-driven over a small table.
#[test]
fn non_finite_and_off_grid_inputs_are_refused() {
    // (input element, expected refusal predicate name)
    // 0.1 is not exactly an f32; NaN/Inf are non-finite.
    let nan = prog(DenseCgOp::Neg, ScalarKind::F32, vec![f64::NAN], None, None);
    assert!(
        matches!(emit_dense_llvm_ir(&nan), Err(DenseAotError::NonFinite(0))),
        "NaN input must be NonFinite-refused"
    );
    let inf = prog(
        DenseCgOp::Neg,
        ScalarKind::F32,
        vec![f64::INFINITY],
        None,
        None,
    );
    assert!(
        matches!(emit_dense_llvm_ir(&inf), Err(DenseAotError::NonFinite(0))),
        "Inf input must be NonFinite-refused"
    );
    let offgrid = prog(DenseCgOp::Neg, ScalarKind::F32, vec![0.1], None, None);
    assert!(
        matches!(emit_dense_llvm_ir(&offgrid), Err(DenseAotError::OffGrid(_))),
        "0.1 (not exact f32) must be OffGrid-refused"
    );
    // A bf16-off-grid value that is exact f32 (1.5 + 2^-9) is still off the bf16 grid.
    let bf16_off = prog(
        DenseCgOp::Neg,
        ScalarKind::Bf16,
        vec![1.501_953_125],
        None,
        None,
    );
    assert!(
        matches!(
            emit_dense_llvm_ir(&bf16_off),
            Err(DenseAotError::OffGrid(_))
        ),
        "an f32-exact but bf16-off value must be OffGrid-refused"
    );
}

/// A `scale` factor that is off the dtype grid is refused (matches `DenseError::ScalarOffGrid`).
#[test]
fn off_grid_scale_factor_is_refused() {
    let p = prog(
        DenseCgOp::Scale,
        ScalarKind::F32,
        vec![1.0],
        None,
        Some(0.1),
    );
    assert!(
        matches!(emit_dense_llvm_ir(&p), Err(DenseAotError::OffGrid(_))),
        "an off-grid scale factor must be refused"
    );
}

/// A dimension mismatch between operand `a` and `b` (binary op) is refused (matches
/// `DenseError::DimMismatch`).
#[test]
fn dim_mismatch_is_refused() {
    let p = prog(
        DenseCgOp::Add,
        ScalarKind::F32,
        vec![1.0, 2.0],
        Some(vec![1.0, 2.0, 3.0]),
        None,
    );
    // dim is set from a.len()=2; b.len()=3 mismatches.
    match emit_dense_llvm_ir(&p) {
        Err(DenseAotError::DimMismatch { expected, got }) => {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        }
        other => panic!("dim mismatch must be refused, got {other:?}"),
    }
}

/// A binary op with no second operand, or a scale with no factor, is a malformed program — refused
/// explicitly, never panicking.
#[test]
fn malformed_programs_are_refused() {
    let no_b = DenseProgram {
        op: DenseCgOp::Add,
        dim: 1,
        dtype: ScalarKind::F32,
        a: vec![1.0],
        b: None,
        scale: None,
    };
    assert!(matches!(
        emit_dense_llvm_ir(&no_b),
        Err(DenseAotError::Malformed(_))
    ));
    let no_scale = DenseProgram {
        op: DenseCgOp::Scale,
        dim: 1,
        dtype: ScalarKind::F32,
        a: vec![1.0],
        b: None,
        scale: None,
    };
    assert!(matches!(
        emit_dense_llvm_ir(&no_scale),
        Err(DenseAotError::Malformed(_))
    ));
}

// ─── the DenseExplain record is constructible + carries the honest split ─────────────────────────

/// The `DenseExplain` carries BOTH the reference value tag and the (distinct) codegen tag — the
/// inspectable, never-conflated honest split (VR-5).
#[test]
fn dense_explain_carries_the_honest_guarantee_split() {
    let e = DenseExplain {
        op: "dense.add",
        dim: 2,
        dtype: ScalarKind::F32,
        rel_eps: 2f64.powi(-24),
        physical: Some(PhysicalLayout::DenseArray),
        reference_guarantee: Some(GuaranteeStrength::Proven),
        codegen_guarantee: GuaranteeStrength::Empirical,
        quant: "un-quantized",
    };
    // The reference VALUE tag (Proven) and the CODEGEN tag (Empirical) are distinct — never conflated.
    assert_eq!(e.reference_guarantee, Some(GuaranteeStrength::Proven));
    assert_eq!(e.codegen_guarantee, GuaranteeStrength::Empirical);
    assert_ne!(
        e.reference_guarantee.unwrap(),
        e.codegen_guarantee,
        "the reference value tag and the codegen-correctness tag must stay distinct (VR-5)"
    );
}

// ─── mutant-witness for the host-side grid helpers (kill the round_f32_to_bf16 / on_grid mutants) ─

/// Direct witness for `round_f32_to_bf16` (the host-side bf16 grid-rounding helper). Pins the exact
/// round of a value that is **f32-exact but bf16-off** to its correct nearest-even bf16 — this is the
/// case where the `>> ↔ <<` and `+ ↔ -`/`+ ↔ *` bit-twiddle mutations diverge (they survive a corpus
/// of only on-grid values, which round to themselves under several mutations). Mirrors
/// `mycelium_dense`'s rounding bit-for-bit (the differential's BF16 leg checks the *emitted* IR; this
/// pins the host helper the grid check relies on).
#[test]
fn round_f32_to_bf16_is_bit_exact() {
    // The bf16-off values are built by *arithmetic* (1.5 ± a sub-bf16-ULP fraction) rather than a
    // decimal literal — a literal like 1.501953125 trips clippy::excessive_precision, and the bit
    // intent is clearer this way. `2⁻⁹` and `2⁻⁷` are both exact f32, so these sums are exact f32.
    let half_ulp_down = 1.5_f32 + 2.0_f32.powi(-9); // 1.5 + 2⁻⁹ = exactly half a bf16 ULP above 1.5
    let above_half = 1.5_f32 + 2.0_f32.powi(-7); // 1.5 + 2⁻⁷ = above the half-ULP (rounds up)

    // 1.5 is on the bf16 grid → rounds to itself.
    assert_eq!(round_f32_to_bf16(1.5), 1.5);
    // 1.5 + 2⁻⁹ is f32-exact but bf16-off. The dropped low bits are exactly half an ULP, so
    // ties-to-even rounds DOWN to 1.5 (1.5's last kept bit is even). A wrong shift / rounding-bias
    // mutation lands elsewhere.
    assert_eq!(
        round_f32_to_bf16(half_ulp_down),
        1.5,
        "ties-to-even must round 1.5+2⁻⁹ down to 1.5 (the correct bf16 round)"
    );
    // Above the half-ULP: rounds UP to the next bf16. Pin that it is strictly above 1.5 and a valid
    // bf16 (low 16 bits zero) — a sign/shift mutation lands on a different grid point.
    let up = round_f32_to_bf16(above_half);
    assert!(
        up > 1.5 && up.to_bits() & 0x0000_FFFF == 0,
        "a >half-ULP value must round UP to a valid bf16 (low 16 bits zero), got {up}"
    );
    // ── The ties-to-even *odd-kept-bit* case — kills the `(bits >> 16) ↔ (bits << 16)` lsb-extraction
    // mutation. That mutation forces the rounding lsb to 0 (round-half-DOWN instead of
    // round-half-to-EVEN), which only differs at an exact half-ULP tie whose kept bit is ODD: there
    // the correct round goes UP (to the even neighbour) but the mutant goes DOWN. The even-kept-bit
    // tie above does NOT distinguish them (both round down). ──
    // bf16 grid in [1,2) has ULP 2⁻⁷, half-ULP 2⁻⁸. Start at the ODD grid point 1 + 2⁻⁷ (mantissa LSB
    // set) and add exactly half a ULP: 1 + 2⁻⁷ + 2⁻⁸. Ties-to-even rounds UP to the EVEN neighbour
    // 1 + 2⁻⁶; the lsb-forced-0 mutant rounds DOWN to 1 + 2⁻⁷.
    let odd_tie = 1.0_f32 + 2.0_f32.powi(-7) + 2.0_f32.powi(-8);
    let even_neighbour = 1.0_f32 + 2.0_f32.powi(-6);
    assert_eq!(
        round_f32_to_bf16(odd_tie),
        even_neighbour,
        "ties-to-even must round an odd-kept-bit tie UP to the even neighbour (kills the lsb >>/<< \
         mutation that would round half-down instead)"
    );

    // The rounded value is always on the bf16 grid (low 16 bits zero) — a `<<`/`>>` swap would leave
    // non-zero low bits.
    for v in [0.5_f32, 2.0, -3.0, half_ulp_down, above_half, odd_tie] {
        let r = round_f32_to_bf16(v);
        assert_eq!(
            r.to_bits() & 0x0000_FFFF,
            0,
            "round_f32_to_bf16({v}) = {r} must be on the bf16 grid (low 16 bits zero)"
        );
    }
}

/// `on_grid` accepts exactly the values on each dtype's grid and rejects off-grid ones — the host
/// check the lowering's input validation relies on. Kills the `on_grid` `== ↔ !=` / branch mutants
/// and (transitively) the `round_f32_to_bf16` mutants via the bf16-off case.
#[test]
fn on_grid_accepts_and_rejects_exactly() {
    // 1.5 + 2⁻⁹ is f32-exact but bf16-OFF (the discriminating case) — built by arithmetic to keep the
    // bit intent clear (and dodge clippy::excessive_precision on the decimal form).
    let bf16_off = 1.5_f32 + 2.0_f32.powi(-9);
    // F32: any exact f32 is on-grid; 0.1 is not exactly an f32.
    assert!(on_grid(ScalarKind::F32, 1.5));
    assert!(on_grid(ScalarKind::F32, f64::from(bf16_off))); // f32-exact
    assert!(!on_grid(ScalarKind::F32, 0.1));
    // BF16: 1.5 is on-grid; the bf16-off value is f32-exact but off the bf16 grid.
    assert!(on_grid(ScalarKind::Bf16, 1.5));
    assert!(
        !on_grid(ScalarKind::Bf16, f64::from(bf16_off)),
        "1.5+2⁻⁹ is f32-exact but off the bf16 grid — must be rejected"
    );
    // F16/F64 are never on-grid (refused dtypes).
    assert!(!on_grid(ScalarKind::F16, 1.5));
    assert!(!on_grid(ScalarKind::F64, 1.5));
}

/// The `Proven` rounding-bound citation is dtype-specific and non-blank — F32 cites single-rounding,
/// BF16 cites the two-rounding composition. The citation IS the transparency record the `ProvenThm`
/// basis carries (a wrong/blank/dtype-confused citation mis-attributes the bound — VR-5/ADR-006), so
/// it is pinned. Kills the `op_citation -> "xyzzy"` and `delete ScalarKind::Bf16 arm` mutants.
#[test]
fn op_citation_is_dtype_specific_and_nonblank() {
    let f32 = op_citation(ScalarKind::F32);
    let bf16 = op_citation(ScalarKind::Bf16);
    assert!(
        !f32.is_empty() && !bf16.is_empty(),
        "citations must be non-blank"
    );
    assert_ne!(
        f32, bf16,
        "F32 and BF16 citations must differ (distinct bounds)"
    );
    // F32 cites the single-rounding 2⁻²⁴ bound; BF16 cites the two-rounding 2⁻⁸ + 2⁻²³ composition.
    assert!(
        f32.contains("2^−24") && f32.contains("binary32"),
        "F32 citation must name the single-rounding binary32 bound; got: {f32}"
    );
    assert!(
        bf16.contains("two-rounding") && bf16.contains("bfloat16"),
        "BF16 citation must name the two-rounding bfloat16 composition; got: {bf16}"
    );
}

/// The `DenseAotError` `Display` strings discriminate the variants (kills the
/// `fmt -> Ok(Default::default())` mutant, which would blank every message — a never-silent refusal
/// must say *what* was refused, G2/ADR-006).
#[test]
fn error_display_messages_discriminate_and_are_nonempty() {
    let cases: [(DenseAotError, &str); 5] = [
        (DenseAotError::UnsupportedDtype(ScalarKind::F64), "F32/BF16"),
        (DenseAotError::Subnormal, "subnormal"),
        (DenseAotError::Overflow, "overflow"),
        (DenseAotError::QuantRefused("q".to_owned()), "quantized"),
        (DenseAotError::NonFinite(3), "NaN/Inf"),
    ];
    for (err, needle) in cases {
        let msg = err.to_string();
        assert!(!msg.is_empty(), "{err:?} Display must be non-empty (G2)");
        assert!(
            msg.contains(needle),
            "{err:?} Display must name the refusal ({needle:?}); got: {msg}"
        );
    }
}
