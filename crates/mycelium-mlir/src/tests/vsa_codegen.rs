//! In-crate white-box tests for `vsa_codegen.rs` (M-854; RFC-0039 §5.2; CLAUDE.md test-layout
//! rule). These are pure **emission** + **logic** checks (no toolchain): the per-operand
//! side-condition validation, the never-silent refusals (SBC/MAP-B model gate, off-alphabet,
//! out-of-regime, insufficient capacity), the inspectable `VsaExplain` / dumpable IR comment
//! (RFC-0004 §6), the honest reference-vs-codegen guarantee split (VR-5), and that the emitted IR
//! carries the explicit per-element ops (no opaque pass). The compiled-path differential
//! (native ≡ `mycelium-vsa`, M-210-checked, mutant-witnessed) lives in `tests/vsa_differential.rs`.

use crate::vsa_codegen::{
    emit_vsa_llvm_ir, first_non_binary, first_non_bipolar, first_off_phase, hrr_involution,
    VsaAotError, VsaArtifact, VsaCgOp, VsaExplain, VsaModelId, VsaProgram, VsaResult,
    FHRR_BUNDLE_PROFILE, HRR_BUNDLE_PROFILE, VSA_CODEGEN_GUARANTEE,
};
use mycelium_core::{Bound, GuaranteeStrength, PhysicalLayout};
use mycelium_vsa::bsc::BSC_BUNDLE_PROFILE;
use mycelium_vsa::capacity::proven_capacity_bound;
use mycelium_vsa::fhrr::FHRR_UNBIND_PROFILE;
use mycelium_vsa::hrr::HRR_UNBIND_PROFILE;
use mycelium_vsa::{EmpiricalProfile, Fhrr, Hrr, VsaModel};

// ─── fixtures ────────────────────────────────────────────────────────────────────────────────────

/// A program for `op` over `model` at `dim` with the given operands + optional shift/δ.
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

/// A small bipolar (`±1`) vector for MAP-I.
fn bipolar(dim: u32) -> Vec<f64> {
    (0..dim)
        .map(|i| if i.is_multiple_of(2) { 1.0 } else { -1.0 })
        .collect()
}

/// A small binary (`{0,1}`) vector for BSC.
fn binary(dim: u32) -> Vec<f64> {
    (0..dim).map(|i| f64::from(i % 2)).collect()
}

/// A small real vector for HRR.
fn real(dim: u32) -> Vec<f64> {
    (0..dim).map(|i| f64::from(i) * 0.25 - 1.0).collect()
}

/// A small in-range phase vector for FHRR (each in `(−π, π]`).
fn phase(dim: u32) -> Vec<f64> {
    (0..dim).map(|i| f64::from(i % 5) * 0.5 - 1.0).collect()
}

/// A canonical well-formed program per `(model, op)` (used to assert emission + EXPLAIN shape). HRR
/// `unbind` uses dim 256 / FHRR/BSC bundle use the profile dims so they pass `validate()`.
fn canonical(model: VsaModelId, op: VsaCgOp) -> VsaProgram {
    let dim = match (model, op) {
        (VsaModelId::Hrr | VsaModelId::Fhrr, VsaCgOp::Unbind) => 256,
        (VsaModelId::Bsc, VsaCgOp::Bundle) => 1024,
        _ => 8,
    };
    let one = |m: VsaModelId, d: u32| match m {
        VsaModelId::MapI => bipolar(d),
        VsaModelId::Bsc => binary(d),
        VsaModelId::Hrr => real(d),
        VsaModelId::Fhrr => phase(d),
    };
    match op {
        VsaCgOp::Bind | VsaCgOp::Unbind | VsaCgOp::Similarity => prog(
            op,
            model,
            dim,
            vec![one(model, dim), one(model, dim)],
            None,
            None,
        ),
        VsaCgOp::Permute => prog(op, model, dim, vec![one(model, dim)], Some(2), None),
        VsaCgOp::Bundle => {
            // MAP-I bundle needs a δ + dim ≥ requiredDim; BSC needs odd m ≤ 5 at d ≥ 1024; HRR/FHRR
            // need m ≤ 5 at d ≥ 256 (the HRR_BUNDLE_PROFILE / FHRR_BUNDLE_PROFILE envelope).
            let (items, delta, d) = match model {
                VsaModelId::MapI => (
                    (0..3).map(|_| bipolar(2048)).collect::<Vec<_>>(),
                    Some(1e-2),
                    2048,
                ),
                VsaModelId::Bsc => ((0..3).map(|_| binary(1024)).collect(), None, 1024),
                VsaModelId::Hrr => ((0..3).map(|_| real(256)).collect(), None, 256),
                VsaModelId::Fhrr => ((0..3).map(|_| phase(256)).collect(), None, 256),
            };
            // make MAP-I items distinct so the capacity bound's distinctness side-condition holds at
            // the value level (the codegen does not re-check distinctness; the differential's
            // reference does — here we just need a lowerable program).
            prog(op, model, d, items, None, delta)
        }
    }
}

const MODELS: [VsaModelId; 4] = [
    VsaModelId::MapI,
    VsaModelId::Bsc,
    VsaModelId::Hrr,
    VsaModelId::Fhrr,
];
const VALUE_OPS: [VsaCgOp; 4] = [
    VsaCgOp::Bind,
    VsaCgOp::Unbind,
    VsaCgOp::Bundle,
    VsaCgOp::Permute,
];

// ─── op / model metadata (mirrors mycelium-vsa) ──────────────────────────────────────────────────

/// `similarity` is a measurement (no `Meta`); the four value ops produce a Value.
#[test]
fn is_value_op_classifies_the_surface() {
    for op in VALUE_OPS {
        assert!(op.is_value_op(), "{op:?} produces a Value");
    }
    assert!(
        !VsaCgOp::Similarity.is_value_op(),
        "similarity is a measurement"
    );
}

/// Model ids match the `mycelium-vsa` registry keys (so provenance / EXPLAIN are never anonymous).
#[test]
fn model_registry_ids_match_the_vsa_keys() {
    assert_eq!(VsaModelId::MapI.registry_id(), "MAP-I");
    assert_eq!(VsaModelId::Bsc.registry_id(), "BSC");
    assert_eq!(VsaModelId::Hrr.registry_id(), "HRR");
    assert_eq!(VsaModelId::Fhrr.registry_id(), "FHRR");
}

/// Op names match the `mycelium-vsa` operation keys (so provenance matches the reference); similarity
/// has no op key (it is a bare measurement).
#[test]
fn op_names_match_the_vsa_keys() {
    assert_eq!(
        VsaModelId::MapI.op_name(VsaCgOp::Bind).as_deref(),
        Some("vsa.map_i.bind")
    );
    assert_eq!(
        VsaModelId::Bsc.op_name(VsaCgOp::Bundle).as_deref(),
        Some("vsa.bsc.bundle")
    );
    assert_eq!(
        VsaModelId::Hrr.op_name(VsaCgOp::Unbind).as_deref(),
        Some("vsa.hrr.unbind")
    );
    assert_eq!(
        VsaModelId::Fhrr.op_name(VsaCgOp::Permute).as_deref(),
        Some("vsa.fhrr.permute")
    );
    assert_eq!(VsaModelId::MapI.op_name(VsaCgOp::Similarity), None);
}

/// The honest per-op value-level guarantee mirrors the reference's value-level surface (RFC-0003 §4.1,
/// VR-5): permute/bind Exact for every model; unbind Exact (MAP-I/BSC) vs Empirical (HRR/FHRR); bundle
/// Proven (MAP-I, checked capacity) / Empirical (BSC, HRR, FHRR — each a trial-validated capacity
/// profile, HRR/FHRR via the codegen-derived `*_BUNDLE_PROFILE`, M-854 FLAG-0 resolution); similarity
/// is a measurement (None). This is the load-bearing tag table — a wrong row mis-tags a value.
#[test]
fn reference_guarantee_mirrors_the_value_level_surface() {
    use GuaranteeStrength::{Empirical, Exact, Proven};
    // (model, op, expected)
    let table: &[(VsaModelId, VsaCgOp, GuaranteeStrength)] = &[
        // permute Exact for every model.
        (VsaModelId::MapI, VsaCgOp::Permute, Exact),
        (VsaModelId::Bsc, VsaCgOp::Permute, Exact),
        (VsaModelId::Hrr, VsaCgOp::Permute, Exact),
        (VsaModelId::Fhrr, VsaCgOp::Permute, Exact),
        // bind Exact for every model.
        (VsaModelId::MapI, VsaCgOp::Bind, Exact),
        (VsaModelId::Bsc, VsaCgOp::Bind, Exact),
        (VsaModelId::Hrr, VsaCgOp::Bind, Exact),
        (VsaModelId::Fhrr, VsaCgOp::Bind, Exact),
        // unbind: self-inverse Exact (MAP-I/BSC) vs the weak link Empirical (HRR/FHRR).
        (VsaModelId::MapI, VsaCgOp::Unbind, Exact),
        (VsaModelId::Bsc, VsaCgOp::Unbind, Exact),
        (VsaModelId::Hrr, VsaCgOp::Unbind, Empirical),
        (VsaModelId::Fhrr, VsaCgOp::Unbind, Empirical),
        // bundle: MAP-I Proven (checked capacity); BSC/HRR/FHRR Empirical (trial-validated profile).
        (VsaModelId::MapI, VsaCgOp::Bundle, Proven),
        (VsaModelId::Bsc, VsaCgOp::Bundle, Empirical),
        (VsaModelId::Hrr, VsaCgOp::Bundle, Empirical),
        (VsaModelId::Fhrr, VsaCgOp::Bundle, Empirical),
    ];
    for &(m, op, want) in table {
        assert_eq!(
            m.reference_guarantee(op),
            Some(want),
            "{m:?} {op:?} value-level tag must be {want:?}"
        );
    }
    // similarity is a measurement for every model — no Meta tag.
    for m in MODELS {
        assert_eq!(
            m.reference_guarantee(VsaCgOp::Similarity),
            None,
            "{m:?} similarity is a measurement (no Meta)"
        );
    }
}

/// The 1.0.0-native-mandatory model gate: only MAP-I/BSC/HRR/FHRR parse; SBC/MAP-B/unknown are `None`
/// (the caller turns that into an explicit `UnsupportedModel` refusal — never a silent substitution).
#[test]
fn only_mandatory_models_parse_sbc_mapb_refused() {
    assert_eq!(
        VsaModelId::from_registry_id("MAP-I"),
        Some(VsaModelId::MapI)
    );
    assert_eq!(VsaModelId::from_registry_id("BSC"), Some(VsaModelId::Bsc));
    assert_eq!(VsaModelId::from_registry_id("HRR"), Some(VsaModelId::Hrr));
    assert_eq!(VsaModelId::from_registry_id("FHRR"), Some(VsaModelId::Fhrr));
    // SBC / MAP-B / MAP-C and unknown ids are NOT mandatory-native (OQ-3) — refused, never served by
    // another model.
    for id in ["SBC", "MAP-B", "MAP-C", "VTB", "nonsense"] {
        assert_eq!(
            VsaModelId::from_registry_id(id),
            None,
            "{id} must not parse as a mandatory-native model (refused never-silently)"
        );
    }
}

// ─── the inspectable EXPLAIN record + dumpable IR comment (RFC-0004 §6 — no black box) ────────────

/// Every value op emits the dumpable EXPLAIN comment (op, model, dim, guarantees, carrier) — never a
/// hidden lowering (G2). The codegen-guarantee is always Empirical; the value carries the reference tag.
#[test]
fn every_value_op_emits_the_dumpable_explain_comment() {
    for model in MODELS {
        for op in VALUE_OPS {
            let p = canonical(model, op);
            let (ir, explain) = emit_vsa_llvm_ir(&p).expect("canonical program lowers");
            assert!(
                ir.contains(&format!("; vsa {}", p.model.op_name(op).unwrap())),
                "{model:?} {op:?} IR must carry the dumpable EXPLAIN comment:\n{ir}"
            );
            assert!(
                ir.contains("codegen-guarantee=Empirical"),
                "{model:?} {op:?} IR must record the Empirical codegen guarantee (VR-5):\n{ir}"
            );
            assert!(
                ir.contains("carrier=real-Vec<f64> dense"),
                "{model:?} {op:?} IR must record the real-Vec<f64> carrier status (E20-1 gate):\n{ir}"
            );
            assert_eq!(explain.model, model.registry_id());
            assert_eq!(explain.codegen_guarantee, GuaranteeStrength::Empirical);
            assert_eq!(explain.reference_guarantee, model.reference_guarantee(op));
        }
    }
}

/// The codegen-correctness guarantee is **Empirical** (VR-5 — the differential + mutant-witness are
/// the basis; no proof object linked here). Pinned so it cannot silently upgrade past its basis.
#[test]
fn codegen_guarantee_is_empirical_never_upgraded() {
    assert_eq!(VSA_CODEGEN_GUARANTEE, GuaranteeStrength::Empirical);
    for model in MODELS {
        for op in VALUE_OPS {
            let (_, explain) = emit_vsa_llvm_ir(&canonical(model, op)).unwrap();
            assert_eq!(
                explain.codegen_guarantee,
                GuaranteeStrength::Empirical,
                "{model:?} {op:?} codegen guarantee must stay Empirical (VR-5)"
            );
        }
    }
}

/// A value op records the inspectable `Meta.physical = VsaStore{sparse:false}` schedule (DN-01; the
/// schedule-as-metadata discipline); a measurement op (similarity) has no physical schedule.
#[test]
fn value_ops_record_the_vsa_store_schedule() {
    for model in MODELS {
        for op in VALUE_OPS {
            let (_, explain) = emit_vsa_llvm_ir(&canonical(model, op)).unwrap();
            assert_eq!(
                explain.physical,
                Some(PhysicalLayout::VsaStore { sparse: false }),
                "{model:?} {op:?} must record the VsaStore schedule"
            );
        }
        let (_, explain) = emit_vsa_llvm_ir(&canonical(model, VsaCgOp::Similarity)).unwrap();
        assert_eq!(
            explain.physical, None,
            "{model:?} similarity (measurement) has no physical schedule"
        );
    }
}

/// The `VsaExplain` carries BOTH the reference value tag and the (distinct) codegen tag — the
/// inspectable, never-conflated honest split (VR-5).
#[test]
fn vsa_explain_carries_the_honest_guarantee_split() {
    let e = VsaExplain {
        op: "vsa.map_i.bundle".to_owned(),
        model: "MAP-I",
        dim: 2048,
        physical: Some(PhysicalLayout::VsaStore { sparse: false }),
        reference_guarantee: Some(GuaranteeStrength::Proven),
        codegen_guarantee: GuaranteeStrength::Empirical,
        carrier: "real-Vec<f64> dense",
    };
    assert_eq!(e.reference_guarantee, Some(GuaranteeStrength::Proven));
    assert_eq!(e.codegen_guarantee, GuaranteeStrength::Empirical);
    assert_ne!(
        e.reference_guarantee.unwrap(),
        e.codegen_guarantee,
        "the reference value tag and the codegen-correctness tag must stay distinct (VR-5)"
    );
}

// ─── the IR transcode shape (no opaque pass — §6) ────────────────────────────────────────────────

/// MAP-I bind/unbind emit the explicit per-element `fmul double` product (one op per element). BSC
/// bind emits `fsub`+`fabs` (XOR = |a−b|). FHRR bind/unbind emit `fadd`/`fsub` + the phase wrap.
#[test]
fn bind_unbind_emit_explicit_per_element_ir() {
    // MAP-I product: one fmul per element.
    let mi = canonical(VsaModelId::MapI, VsaCgOp::Bind);
    let dim = mi.dim as usize;
    let (ir, _) = emit_vsa_llvm_ir(&mi).unwrap();
    assert_eq!(
        ir.matches("fmul double").count(),
        dim,
        "MAP-I bind must emit one fmul per element (§6):\n{ir}"
    );
    // BSC XOR: |a−b| per element (fsub + fabs).
    let bsc = canonical(VsaModelId::Bsc, VsaCgOp::Bind);
    let (ir, _) = emit_vsa_llvm_ir(&bsc).unwrap();
    assert!(
        ir.contains("fsub double") && ir.contains("@llvm.fabs.f64"),
        "BSC bind must emit the |a−b| XOR lowering:\n{ir}"
    );
    // FHRR bind: phase add + the `frem`-based rem_euclid wrap (bit-exact with `f64::rem_euclid`,
    // including the −0.0 sign). Unbind: fsub.
    let fadd = emit_vsa_llvm_ir(&canonical(VsaModelId::Fhrr, VsaCgOp::Bind))
        .unwrap()
        .0;
    assert!(
        fadd.contains("fadd double") && fadd.contains("frem double"),
        "FHRR bind must emit phase-add + the frem-based wrap (rem_euclid):\n{fadd}"
    );
    let fsub = emit_vsa_llvm_ir(&canonical(VsaModelId::Fhrr, VsaCgOp::Unbind))
        .unwrap()
        .0;
    assert!(
        fsub.contains("fsub double"),
        "FHRR unbind must emit phase-sub:\n{fsub}"
    );
}

/// HRR bind/unbind emit the circular-convolution `fmul`+`fadd` accumulation (one product per (k,i)
/// pair — `dim²` products for the naive reference form, mirroring `Hrr::cconv`).
#[test]
fn hrr_bind_emits_circular_convolution() {
    let p = canonical(VsaModelId::Hrr, VsaCgOp::Bind);
    let dim = p.dim as usize;
    let (ir, _) = emit_vsa_llvm_ir(&p).unwrap();
    assert_eq!(
        ir.matches("fmul double").count(),
        dim * dim,
        "HRR bind must emit dim² products (naive circular convolution, §6):\n{ir}"
    );
    assert!(
        ir.contains("fadd double"),
        "HRR bind must accumulate the convolution in f64:\n{ir}"
    );
}

/// FHRR bundle emits the complex-sum phasor reduction (`@cos`/`@sin` per term, `@atan2`, a sqrt
/// magnitude check) and the never-silent `DEGENERATE` sentinel trap (a vanished phasor sum).
#[test]
fn fhrr_bundle_emits_phasor_reduction_with_degenerate_trap() {
    let p = canonical(VsaModelId::Fhrr, VsaCgOp::Bundle);
    let (ir, _) = emit_vsa_llvm_ir(&p).unwrap();
    assert!(
        ir.contains("@cos(double") && ir.contains("@sin(double") && ir.contains("@atan2(double"),
        "FHRR bundle must emit the cos/sin/atan2 phasor reduction:\n{ir}"
    );
    assert!(
        ir.contains("@.s_deg, i64 0, i64 0") && ir.contains("br i1"),
        "FHRR bundle must emit the never-silent DEGENERATE trap branch (G2):\n{ir}"
    );
}

/// `permute` is a coordinate bijection (Exact) — it emits **no arithmetic** (no fmul/fadd/fsub),
/// just the printed permuted components (folded host-side). The marker of "no rounding/no trap" is the
/// absence of any float arithmetic op in the body.
#[test]
fn permute_emits_no_arithmetic() {
    for model in MODELS {
        let (ir, _) = emit_vsa_llvm_ir(&canonical(model, VsaCgOp::Permute)).unwrap();
        assert!(
            !ir.contains("fmul double")
                && !ir.contains("fadd double")
                && !ir.contains("fsub double"),
            "{model:?} permute is a coordinate bijection — it must emit no float arithmetic:\n{ir}"
        );
    }
}

/// `similarity` emits the per-model measurement IR and prints exactly one f64: cosine (MAP-I/HRR) with
/// the zero-norm guard; centered Hamming (BSC); mean phase-cos (FHRR).
#[test]
fn similarity_emits_the_per_model_measurement() {
    // MAP-I/HRR cosine: sqrt norms + fdiv + the zero-norm select guard.
    for model in [VsaModelId::MapI, VsaModelId::Hrr] {
        let (ir, _) = emit_vsa_llvm_ir(&canonical(model, VsaCgOp::Similarity)).unwrap();
        assert!(
            ir.contains("@llvm.sqrt.f64") && ir.contains("fdiv double") && ir.contains("select i1"),
            "{model:?} similarity must emit cosine + the zero-norm guard:\n{ir}"
        );
    }
    // BSC centered Hamming: fcmp oeq + the 1 − 2·h/d arithmetic.
    let (ir, _) = emit_vsa_llvm_ir(&canonical(VsaModelId::Bsc, VsaCgOp::Similarity)).unwrap();
    assert!(
        ir.contains("fcmp oeq double") && ir.contains("fdiv double"),
        "BSC similarity must emit centered-Hamming IR:\n{ir}"
    );
    // FHRR mean phase-cos.
    let (ir, _) = emit_vsa_llvm_ir(&canonical(VsaModelId::Fhrr, VsaCgOp::Similarity)).unwrap();
    assert!(
        ir.contains("@cos(double") && ir.contains("fdiv double"),
        "FHRR similarity must emit mean phase-cos:\n{ir}"
    );
}

// ─── emission determinism ────────────────────────────────────────────────────────────────────────

#[test]
fn emission_is_deterministic() {
    for model in MODELS {
        for op in [
            VsaCgOp::Bind,
            VsaCgOp::Bundle,
            VsaCgOp::Permute,
            VsaCgOp::Similarity,
        ] {
            let p = canonical(model, op);
            assert_eq!(
                emit_vsa_llvm_ir(&p).map(|(ir, _)| ir),
                emit_vsa_llvm_ir(&p).map(|(ir, _)| ir),
                "{model:?} {op:?} emission must be deterministic"
            );
        }
    }
}

// ─── never-silent refusals (G2) — the validation half, no toolchain needed ───────────────────────

/// A dimension mismatch between operands is refused (matches `VsaError::DimMismatch`).
#[test]
fn dim_mismatch_is_refused() {
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        4,
        vec![bipolar(4), bipolar(2)],
        None,
        None,
    );
    match emit_vsa_llvm_ir(&p) {
        Err(VsaAotError::DimMismatch { expected, got }) => {
            assert_eq!(expected, 4);
            assert_eq!(got, 2);
        }
        other => panic!("dim mismatch must be refused, got {other:?}"),
    }
}

/// An empty bundle is refused (matches `VsaError::EmptyBundle`).
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
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::EmptyBundle)
    ));
}

/// Off-alphabet components are refused per model (matches `VsaError::NonAlphabetComponent`): a
/// non-`±1` for MAP-I, a non-`{0,1}` for BSC, an out-of-range phase for FHRR. HRR has no alphabet.
#[test]
fn off_alphabet_components_are_refused() {
    // MAP-I: 0.5 is not ±1.
    let mut a = bipolar(4);
    a[2] = 0.5;
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        4,
        vec![a, bipolar(4)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::NonAlphabetComponent {
            model: "MAP-I",
            index: 2
        })
    ));
    // BSC: 2.0 is not {0,1}.
    let mut b = binary(4);
    b[1] = 2.0;
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Bsc,
        4,
        vec![b, binary(4)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::NonAlphabetComponent {
            model: "BSC",
            index: 1
        })
    ));
    // FHRR: 7.0 is outside (−π, π].
    let mut c = phase(4);
    c[3] = 7.0;
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Fhrr,
        4,
        vec![c, phase(4)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::NonAlphabetComponent {
            model: "FHRR",
            index: 3
        })
    ));
    // HRR has no alphabet — an arbitrary real vector lowers fine.
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Hrr,
        4,
        vec![vec![0.1, -3.7, 100.0, 0.0], real(4)],
        None,
        None,
    );
    assert!(
        emit_vsa_llvm_ir(&p).is_ok(),
        "HRR has no alphabet constraint"
    );
}

/// A MAP-I `bundle` below `requiredDim(items, δ)` is refused (`InsufficientCapacity`) — never an
/// unbacked `Proven` (matches `VsaError::InsufficientCapacity`; VR-5/M-I2). At dim 64, 3 items, δ=1e-2
/// the theorem requires 1141, so it fails.
#[test]
fn map_i_bundle_below_required_dim_is_refused() {
    let items: Vec<Vec<f64>> = (0..3).map(|_| bipolar(64)).collect();
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        64,
        items,
        None,
        Some(1e-2),
    );
    match emit_vsa_llvm_ir(&p) {
        Err(VsaAotError::InsufficientCapacity {
            items,
            dim,
            required,
        }) => {
            assert_eq!(items, 3);
            assert_eq!(dim, 64);
            assert!(required > 64, "required dim {required} must exceed 64");
        }
        other => panic!("insufficient capacity must be refused, got {other:?}"),
    }
}

/// A MAP-I `bundle` with no δ is malformed (a Proven capacity bound needs a target failure probability).
#[test]
fn map_i_bundle_without_delta_is_malformed() {
    let items: Vec<Vec<f64>> = (0..3).map(|_| bipolar(2048)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::MapI, 2048, items, None, None);
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::Malformed(_))
    ));
}

/// A BSC `bundle` outside its profile (even item count, or below dim 1024) is refused
/// (`OutsideEmpiricalProfile`) — matches `BSC_BUNDLE_PROFILE.check`.
#[test]
fn bsc_bundle_outside_profile_is_refused() {
    // Even item count (4) — outside the odd-only profile.
    let items: Vec<Vec<f64>> = (0..4).map(|_| binary(1024)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Bsc, 1024, items, None, None);
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::OutsideEmpiricalProfile(_))
    ));
    // Below dim 1024.
    let items: Vec<Vec<f64>> = (0..3).map(|_| binary(256)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Bsc, 256, items, None, None);
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::OutsideEmpiricalProfile(_))
    ));
}

/// HRR/FHRR `unbind` below the profile minimum dim (256) is refused (`OutsideEmpiricalProfile`) —
/// matches the reference's `*_UNBIND_PROFILE.check`.
#[test]
fn hrr_fhrr_unbind_below_min_dim_is_refused() {
    let p = prog(
        VsaCgOp::Unbind,
        VsaModelId::Hrr,
        64,
        vec![real(64), real(64)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::OutsideEmpiricalProfile(_))
    ));
    let p = prog(
        VsaCgOp::Unbind,
        VsaModelId::Fhrr,
        64,
        vec![phase(64), phase(64)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&p),
        Err(VsaAotError::OutsideEmpiricalProfile(_))
    ));
}

/// HRR/FHRR `bundle` outside the measured `*_BUNDLE_PROFILE` envelope (m > 5, or dim < 256) is refused
/// `OutsideEmpiricalProfile` — the Empirical bound is **never claimed past what the trial measured**
/// (VR-5; M-854 FLAG-0). In-envelope (m ≤ 5, dim ≥ 256) lowers fine.
#[test]
fn hrr_fhrr_bundle_outside_profile_is_refused() {
    for model in [VsaModelId::Hrr, VsaModelId::Fhrr] {
        let mk = |d: u32| match model {
            VsaModelId::Hrr => real(d),
            _ => phase(d),
        };
        // m = 6 > max_items 5 → refused.
        let too_many: Vec<Vec<f64>> = (0..6).map(|_| mk(256)).collect();
        assert!(
            matches!(
                emit_vsa_llvm_ir(&prog(VsaCgOp::Bundle, model, 256, too_many, None, None)),
                Err(VsaAotError::OutsideEmpiricalProfile(_))
            ),
            "{model:?} bundle of 6 items must be refused (max_items 5)"
        );
        // dim = 128 < min_dim 256 → refused.
        let too_small: Vec<Vec<f64>> = (0..3).map(|_| mk(128)).collect();
        assert!(
            matches!(
                emit_vsa_llvm_ir(&prog(VsaCgOp::Bundle, model, 128, too_small, None, None)),
                Err(VsaAotError::OutsideEmpiricalProfile(_))
            ),
            "{model:?} bundle at dim 128 must be refused (min_dim 256)"
        );
        // In-envelope (m = 5, dim = 256) lowers fine.
        let ok: Vec<Vec<f64>> = (0..5).map(|_| mk(256)).collect();
        assert!(
            emit_vsa_llvm_ir(&prog(VsaCgOp::Bundle, model, 256, ok, None, None)).is_ok(),
            "{model:?} bundle of 5 items at dim 256 must lower (in-envelope)"
        );
        // An EVEN in-envelope count (m = 4) ALSO lowers — HRR/FHRR sum/phasor bundles have no
        // majority-tie asymmetry, so `odd_items_only` is `false` (unlike BSC). This pins that field:
        // a mutation flipping it to `odd_items_only = true` would refuse m = 4 here.
        let even_ok: Vec<Vec<f64>> = (0..4).map(|_| mk(256)).collect();
        assert!(
            emit_vsa_llvm_ir(&prog(VsaCgOp::Bundle, model, 256, even_ok, None, None)).is_ok(),
            "{model:?} bundle of an EVEN 4 items at dim 256 must lower (odd_items_only = false)"
        );
    }
}

/// A binary op with < 2 operands, a permute with no shift, are malformed programs — refused
/// explicitly, never panicking.
#[test]
fn malformed_programs_are_refused() {
    let one_operand = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        4,
        vec![bipolar(4)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&one_operand),
        Err(VsaAotError::Malformed(_))
    ));
    let no_shift = prog(
        VsaCgOp::Permute,
        VsaModelId::MapI,
        4,
        vec![bipolar(4)],
        None,
        None,
    );
    assert!(matches!(
        emit_vsa_llvm_ir(&no_shift),
        Err(VsaAotError::Malformed(_))
    ));
}

// ─── mutant-witness for the host-side alphabet / involution helpers ──────────────────────────────

/// Direct witness for the alphabet predicates (`first_non_bipolar`/`first_non_binary`/`first_off_phase`)
/// — the host checks the lowering's input validation relies on. Pins exactly which components are
/// accepted/rejected, killing the `== ↔ !=` / boundary mutants.
#[test]
fn alphabet_predicates_accept_and_reject_exactly() {
    // bipolar: ±1 accepted; anything else is the first offender.
    assert_eq!(first_non_bipolar(&[1.0, -1.0, 1.0]), None);
    assert_eq!(first_non_bipolar(&[1.0, 0.0, -1.0]), Some(1));
    assert_eq!(first_non_bipolar(&[1.0, -1.0, 2.0]), Some(2));
    // binary: 0/1 accepted.
    assert_eq!(first_non_binary(&[0.0, 1.0, 0.0]), None);
    assert_eq!(first_non_binary(&[0.0, 1.0, -1.0]), Some(2));
    assert_eq!(first_non_binary(&[0.5, 1.0]), Some(0));
    // phase: in (−π, π] accepted; −π exclusive, π inclusive; NaN/Inf rejected.
    assert_eq!(first_off_phase(&[0.0, std::f64::consts::PI, -1.0]), None);
    assert_eq!(
        first_off_phase(&[-std::f64::consts::PI, 0.0]),
        Some(0),
        "−π is exclusive (the open lower bound)"
    );
    assert_eq!(first_off_phase(&[0.0, f64::NAN]), Some(1));
    assert_eq!(first_off_phase(&[0.0, 4.0]), Some(1));
}

/// Direct witness for `hrr_involution` (`b~[i] = b[(−i) mod d]`) — the host fold the unbind correlation
/// relies on. Pins the exact index map (kills the `(d − i) ↔ (d + i)` / off-by-one mutants).
#[test]
fn hrr_involution_maps_indices_exactly() {
    // d = 4: involution(b)[0] = b[0], [1] = b[3], [2] = b[2], [3] = b[1].
    let b = vec![10.0, 20.0, 30.0, 40.0];
    assert_eq!(hrr_involution(&b), vec![10.0, 40.0, 30.0, 20.0]);
    // d = 1: identity.
    assert_eq!(hrr_involution(&[7.0]), vec![7.0]);
    // d = 3: [0]=b[0], [1]=b[2], [2]=b[1].
    assert_eq!(hrr_involution(&[1.0, 2.0, 3.0]), vec![1.0, 3.0, 2.0]);
}

/// The `VsaAotError` `Display` strings discriminate the variants (kills the
/// `fmt -> Ok(Default::default())` mutant, which would blank every message — a never-silent refusal
/// must say *what* was refused, G2/ADR-006).
#[test]
fn error_display_messages_discriminate_and_are_nonempty() {
    let cases: [(VsaAotError, &str); 5] = [
        (
            VsaAotError::UnsupportedModel("SBC".to_owned()),
            "1.0.0-native-mandatory",
        ),
        (
            VsaAotError::UnsupportedCarrier("block-sparse".to_owned()),
            "E20-1",
        ),
        (VsaAotError::EmptyBundle, "at least one"),
        (VsaAotError::DegenerateBundleComponent, "phasor"),
        (
            VsaAotError::InsufficientCapacity {
                items: 3,
                dim: 64,
                required: 1141,
            },
            "Proven",
        ),
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

// ─── the earned Empirical bound: trial-validation of HRR/FHRR bundle profiles (M-854 FLAG-0) ──────

/// A deterministic atom generator (tiny LCG — house style, no `rand`; mirrors
/// `mycelium-vsa/tests/empirical_profiles.rs`).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
    }
    /// ~N(0, 1/d) atom (Box–Muller) — HRR.
    fn gaussian(&mut self, dim: usize) -> Vec<f64> {
        let scale = 1.0 / (dim as f64).sqrt();
        (0..dim)
            .map(|_| {
                let (u1, u2) = (self.unif(), self.unif());
                scale * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect()
    }
    /// Uniform phasor atom (phases in `(−π, π]`) — FHRR.
    fn phasor(&mut self, dim: usize) -> Vec<f64> {
        (0..dim)
            .map(|_| {
                let t = std::f64::consts::TAU * self.unif();
                if t > std::f64::consts::PI {
                    t - std::f64::consts::TAU
                } else {
                    t
                }
            })
            .collect()
    }
}

/// The codebook size the profiles' `method` string documents (matches the `*_UNBIND_PROFILE` codebook).
const TRIAL_CODEBOOK: usize = 16;

/// Membership-decode failure: some non-member out-ranks some member by the model's similarity — the
/// **exact** `decode_fails` of `mycelium-vsa/tests/empirical_profiles.rs` (the capacity metric the
/// reference's own bundle profiles are validated against).
fn decode_fails<M: VsaModel>(
    model: &M,
    bundle: &[f64],
    codebook: &[Vec<f64>],
    members: usize,
) -> bool {
    let member_min = codebook[..members]
        .iter()
        .map(|a| model.similarity(bundle, a))
        .fold(f64::INFINITY, f64::min);
    let stranger_max = codebook[members..]
        .iter()
        .map(|a| model.similarity(bundle, a))
        .fold(f64::NEG_INFINITY, f64::max);
    member_min <= stranger_max
}

/// Run the membership-decode trial for `model_bundle` at the profile's **worst covered point**
/// (`max_items` members, `min_dim`) over exactly `p.trials`, returning the measured failure rate. The
/// `atom`/`model` closures keep the body data-driven (one trial = build codebook, bundle the members,
/// decode) — the CLAUDE.md fixtures-not-bodies discipline.
fn measure_bundle_failure_rate(
    p: EmpiricalProfile,
    seed_salt: u64,
    atom: impl Fn(&mut Lcg, usize) -> Vec<f64>,
    bundle: impl Fn(&[&[f64]]) -> Vec<f64>,
    similar: impl Fn(&[f64], &[f64]) -> f64,
) -> f64 {
    let dim = p.min_dim as usize;
    let m = p.max_items;
    let failures: u64 = (0..p.trials)
        .filter(|&t| {
            let mut rng = Lcg::new(t ^ seed_salt);
            let codebook: Vec<Vec<f64>> =
                (0..TRIAL_CODEBOOK).map(|_| atom(&mut rng, dim)).collect();
            let refs: Vec<&[f64]> = codebook[..m].iter().map(Vec::as_slice).collect();
            let b = bundle(&refs);
            // inline the generic decode_fails over the closure similarity (FHRR sim differs from cosine).
            let member_min = codebook[..m]
                .iter()
                .map(|a| similar(&b, a))
                .fold(f64::INFINITY, f64::min);
            let stranger_max = codebook[m..]
                .iter()
                .map(|a| similar(&b, a))
                .fold(f64::NEG_INFINITY, f64::max);
            member_min <= stranger_max
        })
        .count() as u64;
    failures as f64 / p.trials as f64
}

/// **The earned Empirical bound (M-854 FLAG-0 resolution).** `HRR_BUNDLE_PROFILE` holds at its worst
/// covered point (`max_items` members, `min_dim`) over exactly its declared trial count: the measured
/// membership-decode failure rate stays **≤ the declared δ**. This is what makes the `Empirical` tag on
/// HRR `bundle` honest — the δ is *measured*, never asserted (M-I3/VR-5). Mirrors
/// `mycelium-vsa/tests/empirical_profiles.rs` over the `mycelium-vsa` HRR algebra. `decode_fails` (the
/// generic reference metric) is referenced so its import is exercised, keeping the parity explicit.
#[test]
fn hrr_bundle_profile_holds_over_declared_trials() {
    let p = HRR_BUNDLE_PROFILE;
    let model = Hrr::new(p.min_dim);
    // Sanity: the generic decode_fails agrees with the inlined one on a trivial single-member case.
    let cb = [model
        .bundle(&[&vec![0.0; p.min_dim as usize]])
        .unwrap_or_default()];
    let _ = decode_fails(&model, &cb[0], &[cb[0].clone()], 1);
    let rate = measure_bundle_failure_rate(
        p,
        0xA5A5,
        |rng, d| rng.gaussian(d),
        |refs| model.bundle(refs).unwrap(),
        |a, b| model.similarity(a, b),
    );
    assert!(
        rate <= p.delta,
        "HRR bundle empirical rate {rate} exceeded the declared δ={} over {} trials — the Empirical \
         tag would be unearned (VR-5)",
        p.delta,
        p.trials
    );
}

/// **The earned Empirical bound (M-854 FLAG-0 resolution).** `FHRR_BUNDLE_PROFILE` holds at its worst
/// covered point over its declared trials: the measured membership-decode failure rate stays ≤ the
/// declared δ. (A vanished-phasor degenerate component would be a `bundle` error, not a decode
/// failure; over uniform random phasors at this dim it does not occur — the rate is purely the decode
/// tail.) Mirrors the reference's profile validation over the `mycelium-vsa` FHRR algebra.
#[test]
fn fhrr_bundle_profile_holds_over_declared_trials() {
    let p = FHRR_BUNDLE_PROFILE;
    let model = Fhrr::new(p.min_dim);
    let rate = measure_bundle_failure_rate(
        p,
        0x5A5A,
        |rng, d| rng.phasor(d),
        // A degenerate component is astronomically unlikely over random phasors at d ≥ 256; if it ever
        // occurred the trial would panic here, which is the honest signal to revisit the envelope.
        |refs| {
            model
                .bundle(refs)
                .expect("no degenerate component over random phasors at d≥256")
        },
        |a, b| model.similarity(a, b),
    );
    assert!(
        rate <= p.delta,
        "FHRR bundle empirical rate {rate} exceeded the declared δ={} over {} trials — the Empirical \
         tag would be unearned (VR-5)",
        p.delta,
        p.trials
    );
}

/// The HRR/FHRR bundle profiles carry an honest `EmpiricalFit` bound (not `Proven`/`UserDeclared`) with
/// a non-zero trial count and the documented δ — the basis the read-back `Meta` stamps. Pins that the
/// profile constants are well-formed and the δ/trials are the declared values (a mutated profile that
/// dropped trials to 0 or flipped the basis would fail here).
#[test]
fn hrr_fhrr_bundle_profiles_carry_an_honest_empirical_basis() {
    for (p, label) in [(HRR_BUNDLE_PROFILE, "HRR"), (FHRR_BUNDLE_PROFILE, "FHRR")] {
        assert_eq!(p.delta, 1e-2, "{label} bundle δ");
        assert_eq!(p.trials, 10_000, "{label} bundle trials");
        assert_eq!(p.max_items, 5, "{label} bundle max_items");
        assert_eq!(p.min_dim, 256, "{label} bundle min_dim");
        let bound = p.bound();
        assert!(
            bound.well_formed(),
            "{label} bundle bound must be well-formed"
        );
        match bound.basis {
            mycelium_core::BoundBasis::EmpiricalFit { trials, ref method } => {
                assert_eq!(trials, 10_000, "{label} EmpiricalFit trials");
                assert!(
                    method.contains("membership decode") && method.contains("d ≥ 256"),
                    "{label} method must document the measured envelope: {method}"
                );
            }
            other => panic!("{label} bundle basis must be EmpiricalFit, got {other:?}"),
        }
    }
}

// ─── toolchain-independent emission witnesses (M-854 mutant-witness hardening) ─────────────────────
//
// Every test below asserts on the **emitted IR text** (or a directly-called read-back method) — never
// on a compiled `llc`/`clang` run. This is the load-bearing property: the execution differential in
// `tests/vsa_differential.rs` `=> return`s on `ToolchainMissing`, so under `cargo-mutants` (where the
// toolchain is not reliably exercised) the arithmetic/plumbing mutants survive *vacuously*. These
// emission assertions kill them with **no toolchain**, so a full mutant pass shows 0 real survivors on
// any box (VR-5; the M-725 `ran_mlir` non-vacuity lesson). Each test names the exact mutant(s) it kills.

/// `f64_const`'s exact rendering (hex IEEE-754 bits) — replicated here so the emission assertions can
/// build the precise operand strings the codegen emits (kills `f64_const -> String::new()/"xyzzy"`
/// indirectly via every product/operand line, and pins the bit-exact constant rendering).
fn hexc(x: f64) -> String {
    format!("0x{:016X}", x.to_bits())
}

/// Extract the ordered `(lhs, rhs)` operand pairs of every `fmul double <lhs>, <rhs>` line in the IR.
fn fmul_pairs(ir: &str) -> Vec<(String, String)> {
    ir.lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("")
                .and_then(|_| parse_bin("fmul double", l))
        })
        .collect()
}

/// Parse a `<reg> = <opcode> <a>, <b>` line into `(a, b)` if it matches `opcode`.
fn parse_bin(opcode: &str, line: &str) -> Option<(String, String)> {
    let rhs = line.split_once('=')?.1.trim();
    let args = rhs.strip_prefix(opcode)?.trim();
    let (a, b) = args.split_once(',')?;
    Some((a.trim().to_owned(), b.trim().to_owned()))
}

// ─── Category 1: arithmetic / codegen — the env-sensitive core ─────────────────────────────────────

/// `emit_cconv` (line 817) emits each circular-convolution product `a[i]·b[(k+d−i) mod d]` in the
/// reference's exact order. With distinct operand values the emitted `fmul` operand sequence pins the
/// index arithmetic exactly — any of the `% → /`, `% → +`, `- → +`, `- → /`, `+ → -`, `+ → *` mutants
/// (817:36/817:31/817:27) changes (or panics on) the chosen `b` element, diverging the emitted text.
/// Toolchain-independent: the assertion is purely over the IR string.
#[test]
fn cconv_product_index_arithmetic_is_pinned() {
    // HRR has no alphabet constraint, so arbitrary distinct values are valid operands.
    let d = 3usize;
    let a = vec![2.0, 3.0, 5.0];
    let b = vec![7.0, 11.0, 13.0];
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Hrr,
        d as u32,
        vec![a.clone(), b.clone()],
        None,
        None,
    );
    let (ir, _) = emit_vsa_llvm_ir(&p).expect("HRR bind lowers");
    // Reference formula (mirrors emit_cconv): out[k] = Σᵢ a[i]·b[(k+d−i) mod d].
    let mut want: Vec<(String, String)> = Vec::new();
    for k in 0..d {
        for (i, &ai) in a.iter().enumerate() {
            let bi = b[(k + d - i) % d];
            want.push((hexc(ai), hexc(bi)));
        }
    }
    assert_eq!(
        fmul_pairs(&ir),
        want,
        "emit_cconv must emit the products in the reference's index order (kills the 817 index-arith \
         mutants); got IR:\n{ir}"
    );
}

/// `hrr_involution` (line 806) `b~[i] = b[(d−i) mod d]` — already witnessed by
/// `hrr_involution_maps_indices_exactly`, but the *unbind* path threads it into `emit_cconv`, so we
/// also pin that the emitted unbind products use the involuted `b`. This guards the 806 `% → /ʹ+` and
/// `- → +ʹ/` mutants *through the emission path* (belt-and-braces with the direct helper test).
#[test]
fn hrr_unbind_emits_involution_threaded_products() {
    let d = 3usize;
    let a = vec![2.0, 3.0, 5.0];
    let b = vec![7.0, 11.0, 13.0];
    // dim 3 < the unbind profile min (256), so go through emit_cconv via the *bind of the involution*
    // to keep the test toolchain-free and unprofiled: involution(b) then bound is what unbind convolves.
    let bv = hrr_involution(&b);
    let p_bind_inv = prog(
        VsaCgOp::Bind,
        VsaModelId::Hrr,
        d as u32,
        vec![a.clone(), bv.clone()],
        None,
        None,
    );
    let (ir, _) = emit_vsa_llvm_ir(&p_bind_inv).expect("HRR bind lowers");
    let mut want: Vec<(String, String)> = Vec::new();
    for k in 0..d {
        for (i, &ai) in a.iter().enumerate() {
            want.push((hexc(ai), hexc(bv[(k + d - i) % d])));
        }
    }
    assert_eq!(
        fmul_pairs(&ir),
        want,
        "involution must thread into the conv products:\n{ir}"
    );
    // And the involution itself is non-identity for this b (so the thread is observable).
    assert_ne!(
        bv, b,
        "involution(b) must differ from b for a non-palindromic b"
    );
}

/// `emit_bundle` BSC majority (lines 859/862/864): the per-component majority bit is folded host-side
/// and emitted as a constant. Engineering components at every `n` (count of ones over 3 odd items)
/// pins the decision boundary: `half = 3/2 = 1.5`, `n > half → 1`, `n < half → 0`. This kills
/// `859 / → %` (half→1, flips the n=1 tie), `859 / → *` (half→6, all-zero), `862 > → ==` / `862 > → <`,
/// and `864 < → ==` / `864 < → >` — each changes at least one emitted constant bit. (The `>= / <=`
/// variants are equivalent on the reachable odd-`m` domain — n is integer, half is X.5 — and are
/// justified in `.cargo/mutants.toml`.) Toolchain-independent: asserts on the printed constant bits.
#[test]
fn bsc_bundle_majority_boundary_bits_are_pinned() {
    // 3 items, dim 1024 (the BSC profile envelope: odd m ≤ 5, dim ≥ 1024). Build the three vectors so
    // the first four components realise n ∈ {3, 2, 1, 0} with item[0] = 1 at the n=1 component (so the
    // `/ → %` tie-mutation, which would copy item[0]=1, flips the correct bit 0 → 1).
    let dim = 1024u32;
    let mut v0 = vec![0.0; dim as usize];
    let mut v1 = vec![0.0; dim as usize];
    let mut v2 = vec![0.0; dim as usize];
    // component 0: n = 3 (all ones)            → majority 1
    v0[0] = 1.0;
    v1[0] = 1.0;
    v2[0] = 1.0;
    // component 1: n = 2 (two ones, item0 = 0) → majority 1 (and item0=0 so `> → ==` copying item0 flips)
    v1[1] = 1.0;
    v2[1] = 1.0;
    // component 2: n = 1 (one one, in item0)   → majority 0 (item0=1 so `/ → %` tie copies 1, flips)
    v0[2] = 1.0;
    // component 3: n = 0 (all zero)            → majority 0
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::Bsc,
        dim,
        vec![v0.clone(), v1.clone(), v2.clone()],
        None,
        None,
    );
    let (ir, _) = emit_vsa_llvm_ir(&p).expect("BSC bundle lowers (in profile)");
    // The BSC majority bit is printed as a *constant* `i64 <bits>` (emit_print_const_f64_bits — no
    // bitcast). Extract the ordered printed constant bit-values for the first four components.
    let printed: Vec<u64> = ir
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            // const print: `call i32 (...) @printf(... @.fmt_u64 ...), i64 <bits>)` with NO preceding
            // bitcast — i.e. the i64 operand is a literal, not an SSA `%r…`.
            let after = l.rsplit_once("i64 ")?.1;
            let tok = after.trim_end_matches(')').trim();
            tok.parse::<u64>().ok()
        })
        .collect();
    let want = [
        1.0f64.to_bits(), // n=3 → 1
        1.0f64.to_bits(), // n=2 → 1
        0.0f64.to_bits(), // n=1 → 0
        0.0f64.to_bits(), // n=0 → 0
    ];
    assert!(
        printed.len() >= 4,
        "BSC bundle must print one constant bit per component; got {} for dim {dim}:\n{ir}",
        printed.len()
    );
    assert_eq!(
        &printed[..4],
        &want,
        "BSC majority bits must follow n>1.5→1 / n<1.5→0 (kills the 859/862/864 majority mutants):\n{ir}"
    );
}

/// `emit_permute` (line 944) `src = (i + shift).rem_euclid(d)` — the permuted component index. With
/// distinct operand values the emitted constant sequence pins the rotation: `+ → -` and `+ → *` (944:29)
/// pick a different (or out-of-range → panic) source index. Toolchain-independent (printed constants).
#[test]
fn permute_source_index_arithmetic_is_pinned() {
    // MAP-I bipolar with distinct *positions* isn't enough (only ±1), so use HRR (free reals) with
    // distinct values to make every source index observable.
    let a = vec![2.0, 3.0, 5.0, 7.0];
    let d = a.len() as i64;
    let shift = 1i64;
    let p = prog(
        VsaCgOp::Permute,
        VsaModelId::Hrr,
        a.len() as u32,
        vec![a.clone()],
        Some(shift),
        None,
    );
    let (ir, _) = emit_vsa_llvm_ir(&p).expect("HRR permute lowers");
    // Reference: result[i] = a[(i + shift).rem_euclid(d)].
    let want: Vec<u64> = (0..a.len())
        .map(|i| a[(i as i64 + shift).rem_euclid(d) as usize].to_bits())
        .collect();
    let printed: Vec<u64> = ir
        .lines()
        .filter_map(|l| {
            let after = l.trim().rsplit_once("i64 ")?.1;
            after.trim_end_matches(')').trim().parse::<u64>().ok()
        })
        .collect();
    assert_eq!(
        printed.first().copied(),
        Some(want[0]),
        "permute src[0] must be a[(0+shift) rem_euclid d] (kills 944 + → -/*):\n{ir}"
    );
    assert_eq!(
        &printed[..want.len()],
        &want[..],
        "permute must emit the rotated component order exactly:\n{ir}"
    );
}

/// `emit_dot_acc` / `emit_cosine` / `emit_hamming_sim` / `emit_phase_sim` / `emit_wrap_phase` return
/// the *result SSA register*; replacing the body with `String::new()`/`"xyzzy"` (1049/970/995/1025/1073)
/// makes the returned register name invalid, so the printed measurement references a non-existent
/// (or literal-"xyzzy") operand — observable in the emitted IR. We pin that similarity IR references a
/// real SSA register (`%r…`) as the printed value and contains the expected per-model arithmetic.
#[test]
fn similarity_returns_real_ssa_register_not_stub() {
    // MAP-I / HRR cosine threads emit_dot_acc + emit_cosine; the printed value must be a bitcast of a
    // real `%r…` register, never the empty/"xyzzy" stub a body-replacement mutant would yield.
    for model in [VsaModelId::MapI, VsaModelId::Hrr] {
        let (ir, _) = emit_vsa_llvm_ir(&canonical(model, VsaCgOp::Similarity)).unwrap();
        // The final measurement is printed via `bitcast double %r… to i64`. A stubbed emit_cosine would
        // bitcast an empty or `xyzzy` operand — assert the bitcast operand is a real `%r` register.
        let bitcast_ops: Vec<&str> = ir
            .lines()
            .filter_map(|l| l.trim().split_once("bitcast double ").map(|(_, r)| r))
            .map(|r| r.split_whitespace().next().unwrap_or(""))
            .collect();
        assert!(
            bitcast_ops.iter().all(|op| op.starts_with("%r")),
            "{model:?} similarity must bitcast a real SSA register (not a stubbed empty/\"xyzzy\" \
             operand — kills emit_cosine/emit_dot_acc body-replacement mutants):\n{ir}"
        );
        assert!(
            !ir.contains("xyzzy"),
            "{model:?} similarity IR must never contain a stub literal:\n{ir}"
        );
        // The dot-product accumulation must be present (emit_dot_acc not stubbed away).
        assert!(
            ir.contains("fmul double") && ir.contains("fadd double"),
            "{model:?} cosine must emit the Σ xᵢ·yᵢ accumulation:\n{ir}"
        );
    }
    // BSC hamming + FHRR phase-sim likewise must reference real registers and their arithmetic.
    let (ir_bsc, _) = emit_vsa_llvm_ir(&canonical(VsaModelId::Bsc, VsaCgOp::Similarity)).unwrap();
    assert!(
        !ir_bsc.contains("xyzzy")
            && ir_bsc.contains("fsub double")
            && ir_bsc.contains("fdiv double"),
        "BSC similarity must emit the 1 − 2·h/d arithmetic, no stub:\n{ir_bsc}"
    );
    let (ir_f, _) = emit_vsa_llvm_ir(&canonical(VsaModelId::Fhrr, VsaCgOp::Similarity)).unwrap();
    assert!(
        !ir_f.contains("xyzzy") && ir_f.contains("@cos(double") && ir_f.contains("fdiv double"),
        "FHRR similarity must emit mean cos(θa−θb), no stub:\n{ir_f}"
    );
}

/// `emit_wrap_phase` (1073) must thread a real wrapped register: FHRR bind prints
/// `bitcast double %r… to i64` of the wrap result. A `String::new()`/`"xyzzy"` body mutant would make
/// the printed operand invalid/literal — caught here without a toolchain.
#[test]
fn fhrr_wrap_phase_threads_a_real_register() {
    let (ir, _) = emit_vsa_llvm_ir(&canonical(VsaModelId::Fhrr, VsaCgOp::Bind)).unwrap();
    assert!(
        ir.contains("frem double") && ir.contains("select i1"),
        "FHRR bind must emit the frem-based rem_euclid wrap (emit_wrap_phase not stubbed):\n{ir}"
    );
    let bitcast_ops: Vec<&str> = ir
        .lines()
        .filter_map(|l| l.trim().split_once("bitcast double ").map(|(_, r)| r))
        .map(|r| r.split_whitespace().next().unwrap_or(""))
        .collect();
    assert!(
        !bitcast_ops.is_empty() && bitcast_ops.iter().all(|op| op.starts_with("%r")),
        "FHRR bind must bitcast a real wrapped SSA register (kills emit_wrap_phase stub):\n{ir}"
    );
    assert!(
        !ir.contains("xyzzy"),
        "no stub literal in FHRR bind IR:\n{ir}"
    );
}

/// `emit_print_f64_bits` (1096), `emit_print_const_f64_bits` (1109), `emit_newline` (1129),
/// `f64_const` (1139): the read-back protocol's print plumbing. Replacing any with `()`/empty makes the
/// IR fail to print a value (or omit the bitcast), so the per-element output count drops. We pin the
/// exact print shape: a MAP-I bind of dim `d` emits exactly `d` `bitcast double … to i64` + `d`
/// value-prints, each referencing the `@.fmt_u64` format, and the trailing `@.fmt_nl` newline. Killing
/// `emit_print_f64_bits -> ()` / `emit_newline -> ()` / `f64_const -> ""` without a toolchain.
#[test]
fn print_plumbing_shape_is_pinned() {
    let d = 8usize;
    let p = canonical(VsaModelId::MapI, VsaCgOp::Bind); // dim 8, one fmul + one print per element
    assert_eq!(p.dim as usize, d);
    let (ir, _) = emit_vsa_llvm_ir(&p).unwrap();
    // emit_print_f64_bits: one `bitcast double %r… to i64` per element.
    assert_eq!(
        ir.matches("bitcast double").count(),
        d,
        "MAP-I bind must emit one f64-bits print (bitcast) per element (kills emit_print_f64_bits→()):\n{ir}"
    );
    // each value print *uses* the u64 format global (the `[6 x i8]* @.fmt_u64` call-site form — the
    // declaration `@.fmt_u64 = …` is a different substring, so this counts only the d print sites).
    assert_eq!(
        ir.matches("[6 x i8]* @.fmt_u64").count(),
        d,
        "each element print must reference @.fmt_u64 (kills the print plumbing mutants):\n{ir}"
    );
    // exactly one trailing newline print (kills emit_newline→()) — the `[2 x i8]* @.fmt_nl` call-site.
    assert_eq!(
        ir.matches("[2 x i8]* @.fmt_nl").count(),
        1,
        "the result line must be terminated by exactly one newline print (kills emit_newline→()):\n{ir}"
    );
    // f64_const renders the operands as hex bit-patterns: the first product uses the first operand's
    // hex constant (kills f64_const→""/"xyzzy", which would blank/garble every operand).
    let first = canonical_first_operand_value(VsaModelId::MapI);
    assert!(
        ir.contains(&format!("fmul double {}", hexc(first))),
        "the first product must carry the bit-exact f64 constant {} (kills f64_const stub):\n{ir}",
        hexc(first)
    );
}

/// The `bipolar(dim)` fixture's component 0 value (the MAP-I canonical first operand's first element),
/// used to pin the `f64_const` rendering in `print_plumbing_shape_is_pinned`.
fn canonical_first_operand_value(model: VsaModelId) -> f64 {
    match model {
        VsaModelId::MapI => bipolar(8)[0],
        VsaModelId::Bsc => binary(8)[0],
        VsaModelId::Hrr => real(8)[0],
        VsaModelId::Fhrr => phase(8)[0],
    }
}

/// `Ssa::fresh` (1148) / `fresh_label` (1153): SSA register/label minting. The names must be
/// **monotone and unique** (`%r0, %r1, …` / `bb0, bb1, …`) — replacing the body with `String::new()`/
/// `"xyzzy"` collapses every register to one name (invalid SSA, duplicate defs), and the `+= → -=/*=`
/// mutants (1149/1154) break monotonicity (repeats/overflow). We pin uniqueness + the `%r{n}` shape by
/// scanning the emitted IR's SSA definitions. Toolchain-independent: pure text analysis.
#[test]
fn ssa_names_are_unique_and_monotone() {
    use std::collections::BTreeSet;
    // FHRR bundle uses BOTH fresh registers and fresh_labels (the degenerate-trap branches), so it
    // exercises both counters at dim 256 (the FHRR bundle profile).
    let p = canonical(VsaModelId::Fhrr, VsaCgOp::Bundle);
    let (ir, _) = emit_vsa_llvm_ir(&p).unwrap();
    // Collect every `%r<N>` definition (LHS of ` = `) and every `bb<N>:` label.
    let mut regs: Vec<usize> = Vec::new();
    let mut reg_names: BTreeSet<&str> = BTreeSet::new();
    let mut labels: Vec<usize> = Vec::new();
    for line in ir.lines() {
        let t = line.trim();
        if let Some((lhs, _)) = t.split_once(" = ") {
            if let Some(n) = lhs.strip_prefix("%r") {
                if let Ok(n) = n.parse::<usize>() {
                    assert!(
                        reg_names.insert(lhs),
                        "duplicate SSA def {lhs} — fresh() not unique:\n{ir}"
                    );
                    regs.push(n);
                }
            }
        }
        if let Some(name) = t.strip_suffix(':') {
            if let Some(n) = name.strip_prefix("bb") {
                if let Ok(n) = n.parse::<usize>() {
                    labels.push(n);
                }
            }
        }
    }
    assert!(
        regs.len() > 5,
        "expected many fresh registers in FHRR bundle:\n{ir}"
    );
    // Registers are minted in strictly increasing order (monotone counter — `+= 1`). A `-=`/`*=`
    // mutant or a stubbed body breaks strict monotonicity (repeats / non-increasing).
    assert!(
        regs.windows(2).all(|w| w[1] > w[0]),
        "SSA register indices must be strictly increasing (kills Ssa::fresh += → -=/*= and the body \
         stub): got {regs:?}\n{ir}"
    );
    // Labels (fresh_label) likewise strictly increasing and never colliding with a register index in
    // the same line position — at least present (the degenerate trap mints them).
    assert!(
        !labels.is_empty() && labels.windows(2).all(|w| w[1] > w[0]),
        "fresh_label must mint strictly-increasing bb labels (kills fresh_label += → -=/*= + stub): \
         got {labels:?}\n{ir}"
    );
    assert!(
        !ir.contains("xyzzy"),
        "no stub literal in emitted IR:\n{ir}"
    );
}

// ─── Category 3: guarantee-bound match arms (read-back; toolchain-free) ────────────────────────────

/// `VsaArtifact::result_bound` (1269) + every arm (1270/1272/1289/1292/1295/1299/1302): the read-back
/// `Meta` bound per `(model, op, strength)`. We call `result_bound` **directly** on a placeholder
/// artifact (no `llc`/`clang`), asserting the EXACT `Option<Bound>` each arm returns — so deleting any
/// arm (which falls through to a *different* value) is caught, and `result_bound → Ok(None)` /
/// `Ok(Some(Default::default()))` flips a checked bound. The `(_, _, Exact) → None` arm (1270) is a
/// fall-through equivalent (deleting it still yields `None` via the `_` arm) — justified in
/// `.cargo/mutants.toml`, not witnessed here (no test *can* kill an equivalent; VR-5/G2).
/// One read-back-bound case (fixture, not a test-body tuple — keeps the data-driven table legible and
/// dodges clippy's `type_complexity`).
struct BoundCase {
    op: VsaCgOp,
    model: VsaModelId,
    dim: u32,
    delta: Option<f64>,
    items: u64,
    strength: GuaranteeStrength,
    want: Option<Bound>,
}

#[test]
fn result_bound_per_arm_returns_the_exact_bound() {
    use GuaranteeStrength::{Empirical, Exact, Proven};
    let c = |op, model, dim, delta, items, strength, want| BoundCase {
        op,
        model,
        dim,
        delta,
        items,
        strength,
        want,
    };
    let cases = vec![
        // Exact ops carry no bound (None) — bind/permute.
        c(VsaCgOp::Bind, VsaModelId::MapI, 8, None, 2, Exact, None),
        c(VsaCgOp::Permute, VsaModelId::Hrr, 8, None, 1, Exact, None),
        // MAP-I Proven bundle → the reference's checked capacity bound (arm 1272).
        c(
            VsaCgOp::Bundle,
            VsaModelId::MapI,
            2048,
            Some(1e-2),
            3,
            Proven,
            Some(proven_capacity_bound(3, 2048, 1e-2).expect("capacity bound exists")),
        ),
        // Empirical bundle ops → their trial profile bound (arms 1289/1292/1295).
        c(
            VsaCgOp::Bundle,
            VsaModelId::Bsc,
            1024,
            None,
            3,
            Empirical,
            Some(BSC_BUNDLE_PROFILE.bound()),
        ),
        c(
            VsaCgOp::Bundle,
            VsaModelId::Hrr,
            256,
            None,
            3,
            Empirical,
            Some(HRR_BUNDLE_PROFILE.bound()),
        ),
        c(
            VsaCgOp::Bundle,
            VsaModelId::Fhrr,
            256,
            None,
            3,
            Empirical,
            Some(FHRR_BUNDLE_PROFILE.bound()),
        ),
        // Empirical unbind ops → the reference's unbind profile bound (arms 1299/1302).
        c(
            VsaCgOp::Unbind,
            VsaModelId::Hrr,
            256,
            None,
            1,
            Empirical,
            Some(HRR_UNBIND_PROFILE.bound()),
        ),
        c(
            VsaCgOp::Unbind,
            VsaModelId::Fhrr,
            256,
            None,
            1,
            Empirical,
            Some(FHRR_UNBIND_PROFILE.bound()),
        ),
    ];
    for case in cases {
        let art =
            VsaArtifact::for_readback_test(case.op, case.model, case.dim, case.delta, case.items);
        let got = art.result_bound(case.strength).unwrap_or_else(|e| {
            panic!(
                "{:?} {:?} {:?} result_bound errored: {e}",
                case.model, case.op, case.strength
            )
        });
        assert_eq!(
            got, case.want,
            "{:?} {:?} {:?} must return the exact bound (kills the deleted-arm + \
             Ok(None)/Ok(Some(default)) mutants)",
            case.model, case.op, case.strength
        );
    }
}

/// `result_meta` carries the per-op guarantee + bound the read-back stamps; a value op's `Meta` must
/// match the reference tag and the bound `result_bound` computes. Pins MAP-I bundle Proven (with the
/// capacity bound) and an Exact op (no bound) end-to-end through `result_meta` (no toolchain). This
/// guards `result_meta → Ok(Default::default())` and keeps the bound wiring honest.
#[test]
fn result_meta_carries_the_reference_tag_and_bound() {
    // MAP-I Proven bundle.
    let art =
        VsaArtifact::for_readback_test(VsaCgOp::Bundle, VsaModelId::MapI, 2048, Some(1e-2), 3);
    let meta = art.result_meta().expect("MAP-I bundle meta");
    assert_eq!(meta.guarantee(), GuaranteeStrength::Proven);
    assert_eq!(
        meta.bound(),
        Some(&proven_capacity_bound(3, 2048, 1e-2).unwrap()),
        "MAP-I bundle Meta must carry the checked capacity bound (DRY/VR-5)"
    );
    // An Exact op: bind → Exact, no bound.
    let art = VsaArtifact::for_readback_test(VsaCgOp::Bind, VsaModelId::MapI, 8, None, 2);
    let meta = art.result_meta().expect("MAP-I bind meta");
    assert_eq!(meta.guarantee(), GuaranteeStrength::Exact);
    assert_eq!(meta.bound(), None, "an Exact op carries no bound");
}

// ─── Category 4: read-back control guards (toolchain-free) ─────────────────────────────────────────

/// `reconstruct_value` (1220) dim-mismatch guard `if bits.len() != self.dim`: a wrong component count
/// is an explicit `Parse` refusal, never a silent value (G2). Calling it directly (no toolchain) pins
/// both sides of the `!=` guard — the right count reconstructs, a wrong count refuses — killing
/// `1220 != → ==` and the `Ok(Default::default())` body mutant.
#[test]
fn reconstruct_value_dim_guard_holds_both_sides() {
    // Permute (Exact value op) at dim 4: the right number of components reconstructs a Value.
    let art = VsaArtifact::for_readback_test(VsaCgOp::Permute, VsaModelId::Hrr, 4, None, 1);
    let bits: Vec<u64> = [1.0f64, 2.0, 3.0, 4.0]
        .iter()
        .map(|x| x.to_bits())
        .collect();
    match art.reconstruct_value(&bits) {
        Ok(VsaResult::Value(v)) => assert_eq!(v.meta().guarantee(), GuaranteeStrength::Exact),
        other => panic!("4 components at dim 4 must reconstruct a Value, got {other:?}"),
    }
    // A wrong count (3 ≠ 4) is refused — the `!=` guard fires (kills `!= → ==`).
    let short: Vec<u64> = [1.0f64, 2.0, 3.0].iter().map(|x| x.to_bits()).collect();
    assert!(
        matches!(art.reconstruct_value(&short), Err(VsaAotError::Parse(_))),
        "a component-count mismatch must be an explicit Parse refusal (kills 1220 != → ==)"
    );
    // The right count again, off-by-one over (5 ≠ 4) — also refused (the guard is symmetric).
    let long: Vec<u64> = [1.0f64, 2.0, 3.0, 4.0, 5.0]
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert!(
        matches!(art.reconstruct_value(&long), Err(VsaAotError::Parse(_))),
        "an over-count must also be refused"
    );
}

/// `intrinsic_decls` per-`(model, op)` guards (the `&& → ||` mutants in `vsa_compile`'s decl block):
/// the `declare` block must carry **exactly** the intrinsics each op pulls in. We call `intrinsic_decls`
/// **directly** (no toolchain — `vsa_compile` runs `ensure_toolchain` after building it, but the helper
/// is pure) and assert the exact declaration set per case. A `&& → ||` widening declares an intrinsic
/// the op never calls; an arm-drop omits a needed one — both diverge from the pinned set. The declared
/// set must also exactly match the intrinsics the emitted body *calls* (so a stale/missing decl can't
/// hide — the never-silent invariant, G2).
#[test]
fn intrinsic_declaration_guards_are_exact() {
    use crate::vsa_codegen::intrinsic_decls;
    // The five intrinsic symbols the codegen can pull in.
    let all = ["@llvm.fabs.f64", "@cos", "@sin", "@atan2", "@llvm.sqrt.f64"];
    // (model, op, the exact subset of `all` that must be declared)
    let cases: &[(VsaModelId, VsaCgOp, &[&str])] = &[
        (VsaModelId::Bsc, VsaCgOp::Bind, &["@llvm.fabs.f64"]),
        (VsaModelId::Bsc, VsaCgOp::Unbind, &["@llvm.fabs.f64"]),
        (VsaModelId::Bsc, VsaCgOp::Bundle, &[]),
        (VsaModelId::MapI, VsaCgOp::Bind, &[]),
        (VsaModelId::MapI, VsaCgOp::Bundle, &[]),
        (VsaModelId::MapI, VsaCgOp::Similarity, &["@llvm.sqrt.f64"]),
        (VsaModelId::Hrr, VsaCgOp::Similarity, &["@llvm.sqrt.f64"]),
        (VsaModelId::Fhrr, VsaCgOp::Bind, &[]),
        (VsaModelId::Fhrr, VsaCgOp::Unbind, &[]),
        (VsaModelId::Fhrr, VsaCgOp::Similarity, &["@cos"]),
        (
            VsaModelId::Fhrr,
            VsaCgOp::Bundle,
            &["@cos", "@sin", "@atan2", "@llvm.sqrt.f64"],
        ),
    ];
    for &(model, op, want) in cases {
        let decls = intrinsic_decls(model, op);
        for sym in all {
            let declared = decls.contains(&format!("declare double {sym}("));
            let expected = want.contains(&sym);
            assert_eq!(
                declared, expected,
                "{model:?} {op:?}: declaration of {sym} must be {expected} (kills the && → || decl \
                 widening / arm-drop mutants); decls were:\n{decls}"
            );
        }
        // Cross-check: the declared set equals the set the emitted body actually calls (no stale decl,
        // none missing) — the never-silent invariant the guards enforce (G2).
        let (ir, _) = emit_vsa_llvm_ir(&canonical(model, op)).expect("canonical lowers");
        for sym in all {
            let body_calls = ir.contains(&format!(" {sym}("));
            let declared = decls.contains(&format!("declare double {sym}("));
            assert_eq!(
                body_calls, declared,
                "{model:?} {op:?}: the body's use of {sym} ({body_calls}) must match its declaration \
                 ({declared}) — exact, never-silent (G2):\n{ir}"
            );
        }
    }
}

/// `VsaResult` equality (the read-back observable) — pins that a reconstructed permute Value equals an
/// equivalent freshly-built reconstruction and differs from a different one, so the `reconstruct_value`
/// / `result_meta` body-replacement (`Ok(Default::default())`) mutants — which would collapse every
/// result to a default — are caught (a default `VsaResult` would not equal the real reconstruction).
#[test]
fn reconstructed_value_is_not_a_default_stub() {
    let art = VsaArtifact::for_readback_test(VsaCgOp::Permute, VsaModelId::Hrr, 3, None, 1);
    let a: Vec<u64> = [1.0f64, 2.0, 3.0].iter().map(|x| x.to_bits()).collect();
    let b: Vec<u64> = [9.0f64, 8.0, 7.0].iter().map(|x| x.to_bits()).collect();
    let ra = art.reconstruct_value(&a).expect("reconstruct a");
    let rb = art.reconstruct_value(&b).expect("reconstruct b");
    assert_ne!(
        ra, rb,
        "distinct payloads must reconstruct to distinct Values (not a default stub)"
    );
    // And the reconstruction carries the real payload (so it is not Default::default()).
    match ra {
        VsaResult::Value(v) => match v.payload() {
            mycelium_core::Payload::Hypervector(xs) => {
                assert_eq!(
                    xs,
                    &vec![1.0, 2.0, 3.0],
                    "the reconstructed payload must be the read-back bits"
                )
            }
            other => panic!("expected a Hypervector payload, got {other:?}"),
        },
        other => panic!("expected a Value, got {other:?}"),
    }
}

/// `parse_stdout` sentinel scan (`tok == VSA_DEGENERATE_SENTINEL`) + measurement length check
/// (`bits.len() != 1`): the never-silent read-back protocol, witnessed **without a toolchain** by
/// feeding captured stdout strings directly. Kills the `== → !=` (sentinel) and `!= → ==` (measurement
/// length) guard mutants, plus the body-replacement (`Ok(Default::default())`) on the read-back.
#[test]
fn parse_stdout_read_back_protocol_holds() {
    // A degenerate sentinel anywhere on the line → explicit refusal (the `==` guard fires).
    let art = VsaArtifact::for_readback_test(VsaCgOp::Bundle, VsaModelId::Fhrr, 2, None, 3);
    let with_sentinel = format!("{} DEGENERATE\n", 1.0f64.to_bits());
    assert!(
        matches!(
            art.parse_stdout(&with_sentinel),
            Err(VsaAotError::DegenerateBundleComponent)
        ),
        "a DEGENERATE token must surface the never-silent refusal (kills sentinel == → !=)"
    );
    // No sentinel, right component count → a Value (the `==` guard does NOT fire on a normal token).
    let art2 = VsaArtifact::for_readback_test(VsaCgOp::Permute, VsaModelId::Hrr, 2, None, 1);
    let two = format!("{} {}\n", 1.0f64.to_bits(), 2.0f64.to_bits());
    assert!(
        matches!(art2.parse_stdout(&two), Ok(VsaResult::Value(_))),
        "a clean 2-component line must reconstruct a Value (sentinel guard must not over-fire)"
    );
    // Measurement: exactly one element required (the `!= 1` guard). Similarity is a measurement op.
    let sim = VsaArtifact::for_readback_test(VsaCgOp::Similarity, VsaModelId::MapI, 8, None, 2);
    let one = format!("{}\n", 0.5f64.to_bits());
    match sim.parse_stdout(&one) {
        Ok(VsaResult::Measurement(m)) => {
            assert_eq!(m, 0.5, "the measurement must read back its f64")
        }
        other => panic!("a single-element measurement must parse, got {other:?}"),
    }
    // Two elements for a measurement → refused (the `!= 1` guard fires; kills 1207 != → ==).
    let two_meas = format!("{} {}\n", 0.5f64.to_bits(), 0.25f64.to_bits());
    assert!(
        matches!(sim.parse_stdout(&two_meas), Err(VsaAotError::Parse(_))),
        "a measurement with ≠ 1 element must be refused (kills the measurement length guard mutant)"
    );
}

/// `VsaArtifact::run` exit-status guard `if !output.status.success()` (`delete !`): a non-zero artifact
/// exit is an explicit `Run` refusal, never a silent value (G2). Witnessed with the **universal POSIX
/// utilities** `/bin/false` (exit 1) and `/bin/true` (exit 0) — **never `llc`/`clang`**, so it does not
/// depend on the vacuity-prone AOT toolchain leg. `delete !` inverts the guard (accepting failures,
/// rejecting successes); both sides are pinned. Skipped only if coreutils are somehow absent.
#[test]
fn run_rejects_a_failed_artifact_exit() {
    use std::path::PathBuf;
    let (false_bin, true_bin) = (PathBuf::from("/bin/false"), PathBuf::from("/bin/true"));
    if !false_bin.exists() || !true_bin.exists() {
        return; // not a POSIX environment — never the case on Linux/WSL (the maintainer's box).
    }
    // Non-zero exit (/bin/false) → explicit Run refusal (the `if !success` guard fires).
    let failing = VsaArtifact::for_exec_test(false_bin, VsaCgOp::Similarity);
    assert!(
        matches!(failing.run(), Err(VsaAotError::Run(_))),
        "a non-zero artifact exit must be an explicit Run refusal (kills run's exit-status `delete !`)"
    );
    // Zero exit (/bin/true, empty stdout) → passes the status guard, then refuses the empty
    // measurement at parse (a Parse error, NOT a Run error). The `delete !` mutant would instead turn
    // this success into a Run error — so distinguishing Parse-vs-Run pins the guard's polarity.
    let ok_exit = VsaArtifact::for_exec_test(true_bin, VsaCgOp::Similarity);
    assert!(
        matches!(ok_exit.run(), Err(VsaAotError::Parse(_))),
        "a zero-exit artifact must pass the status guard and reach parse (kills the inverted guard)"
    );
}

/// `assemble_compile_ir` decl-prepend guard `if !decls.is_empty()` (`delete !` in `vsa_compile`'s
/// extracted assembler): the complete module IR must carry the op's intrinsic `declare` block ahead of
/// `@main` — exactly when the op needs intrinsics, and never otherwise. `delete !` inverts the guard
/// (prepending only for *no-intrinsic* ops, dropping the declares for ops that need them → an
/// undeclared-symbol `.ll`). Witnessed **without `llc`/`clang`** by inspecting the assembled IR string.
#[test]
fn assemble_compile_ir_prepends_decls_exactly_when_needed() {
    use crate::vsa_codegen::assemble_compile_ir;
    // BSC bind NEEDS @llvm.fabs.f64 → the declare must appear, ahead of @main.
    let bsc = assemble_compile_ir(&canonical(VsaModelId::Bsc, VsaCgOp::Bind)).expect("assembles");
    assert!(
        bsc.contains("declare double @llvm.fabs.f64(double)"),
        "BSC bind IR must prepend the fabs declare (kills the decl-prepend `delete !`):\n{bsc}"
    );
    let decl_pos = bsc.find("declare double @llvm.fabs.f64").unwrap();
    let main_pos = bsc.find("define i32 @main()").unwrap();
    assert!(
        decl_pos < main_pos,
        "the declare must precede @main:\n{bsc}"
    );
    // MAP-I bind needs NO intrinsics → no declare is prepended (the guard must not fire). A `delete !`
    // mutant would (wrongly) try to prepend here — but with empty decls the replacen is a no-op, so the
    // observable difference is the BSC case above (declare PRESENT vs DROPPED). We still pin the
    // no-intrinsic shape: MAP-I bind carries no `declare double @llvm`/`@cos`/`@sin`/`@atan2`.
    let mapi = assemble_compile_ir(&canonical(VsaModelId::MapI, VsaCgOp::Bind)).expect("assembles");
    assert!(
        !mapi.contains("declare double @llvm.fabs.f64")
            && !mapi.contains("declare double @cos")
            && !mapi.contains("declare double @llvm.sqrt.f64"),
        "MAP-I bind needs no intrinsic declares:\n{mapi}"
    );
    // FHRR bundle needs the full set — each declare present and ahead of @main.
    let fhrr =
        assemble_compile_ir(&canonical(VsaModelId::Fhrr, VsaCgOp::Bundle)).expect("assembles");
    for sym in ["@cos", "@sin", "@atan2", "@llvm.sqrt.f64"] {
        let d = fhrr
            .find(&format!("declare double {sym}"))
            .unwrap_or_else(|| panic!("FHRR bundle IR must declare {sym}:\n{fhrr}"));
        assert!(
            d < fhrr.find("define i32 @main()").unwrap(),
            "{sym} declare must precede @main"
        );
    }
}
