//! M-855 — the **dynamic-VSA JIT differential** (E25-1; **RFC-0039 §5.3**; NFR-7; VR-4; the M-210
//! shared checker; RFC-0003 §4.1 the per-op guarantee matrix).
//!
//! The in-process dynamic-VSA JIT (`vsa_jit.rs`: emit IR at runtime → `clang -shared` → `dlopen` →
//! `dlsym` → call) is validated against the trusted-base reference `mycelium-vsa` for the same
//! 1.0.0-native-mandatory **MAP-I / BSC / HRR / FHRR** models the M-854 AOT path covers — bind / unbind
//! / bundle / permute / similarity. This mirrors `tests/vsa_differential.rs`'s corpus and checker
//! pattern exactly (the JIT reuses `vsa_codegen`'s program/error/read-back shapes verbatim, so the two
//! differentials are structurally the same test over two different sinks).
//!
//! **Bit-exact by construction.** The JIT kernel computes every component in `f64`, mirroring the
//! reference's `f64` arithmetic digit-for-digit (the same helpers `vsa_codegen` uses), so the read-back
//! hypervector is **bit-identical** to `mycelium-vsa`'s — what the M-210 observational checker requires.
//!
//! **The "dynamic" dimension exercised for real.** [`jit_specializes_per_call_from_runtime_data`]
//! builds a small corpus of `(model, dim)` pairs from a **runtime** iterator (not Rust-compile-time
//! constants) and JIT-compiles + runs each — the exact RFC-0039 §5.3 scenario (a caller that only knows
//! the model/dimension at runtime gets a kernel specialized to exactly that shape, fresh each call).
//!
//! **Toolchain skip.** The JIT needs `clang` (no `llc`); where absent it returns `ToolchainMissing` and
//! the path **skips** (the house idiom) — never a false failure.
//!
//! **Guarantee:** `Empirical` — the differential is empirical evidence the dynamic-VSA JIT agrees with
//! the trusted `mycelium-vsa` reference over the corpus; never upgraded to `Proven` without a checked
//! proof object linked into codegen (VR-5).

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{
    operation_hash, Bound, GuaranteeStrength, Meta, Payload, PhysicalLayout, Provenance, Repr,
    SparsityClass, Value,
};
use mycelium_mlir::{
    vsa_jit_compile_and_run, VsaAotError, VsaCgOp, VsaModelId, VsaProgram, VsaResult,
    FHRR_BUNDLE_PROFILE, HRR_BUNDLE_PROFILE,
};
use mycelium_numerics::Certificate;
use mycelium_vsa::bsc::BSC_BUNDLE_PROFILE;
use mycelium_vsa::capacity::proven_capacity_bound;
use mycelium_vsa::fhrr::FHRR_UNBIND_PROFILE;
use mycelium_vsa::hrr::HRR_UNBIND_PROFILE;
use mycelium_vsa::{Bsc, Fhrr, Hrr, MapI, VsaModel};

// ─── observable + helpers (mirrors tests/vsa_differential.rs) ────────────────────────────────────

type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);
fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}

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

fn vsa_repr(model: VsaModelId, dim: u32) -> Repr {
    Repr::Vsa {
        model: model.registry_id().to_owned(),
        dim,
        sparsity: SparsityClass::Dense,
    }
}

fn reference_value(
    model: VsaModelId,
    op: VsaCgOp,
    dim: u32,
    payload: Vec<f64>,
    item_count: u64,
    delta: Option<f64>,
) -> Value {
    let guarantee = model.reference_guarantee(op).expect("value op");
    let op_name = model.op_name(op).expect("value op has a key");
    let provenance = Provenance::Derived {
        op: operation_hash(&op_name),
        inputs: vec![],
    };
    let bound: Option<Bound> = match (model, op, guarantee) {
        (_, _, GuaranteeStrength::Exact) => None,
        (VsaModelId::MapI, VsaCgOp::Bundle, GuaranteeStrength::Proven) => {
            Some(proven_capacity_bound(item_count, u64::from(dim), delta.unwrap()).unwrap())
        }
        (VsaModelId::Bsc, VsaCgOp::Bundle, GuaranteeStrength::Empirical) => {
            Some(BSC_BUNDLE_PROFILE.bound())
        }
        (VsaModelId::Hrr, VsaCgOp::Bundle, GuaranteeStrength::Empirical) => {
            Some(HRR_BUNDLE_PROFILE.bound())
        }
        (VsaModelId::Fhrr, VsaCgOp::Bundle, GuaranteeStrength::Empirical) => {
            Some(FHRR_BUNDLE_PROFILE.bound())
        }
        (VsaModelId::Hrr, VsaCgOp::Unbind, GuaranteeStrength::Empirical) => {
            Some(HRR_UNBIND_PROFILE.bound())
        }
        (VsaModelId::Fhrr, VsaCgOp::Unbind, GuaranteeStrength::Empirical) => {
            Some(FHRR_UNBIND_PROFILE.bound())
        }
        _ => None,
    };
    let meta = Meta::new(
        provenance,
        guarantee,
        bound,
        None,
        Some(PhysicalLayout::VsaStore { sparse: false }),
        None,
    )
    .expect("well-formed reference meta");
    Value::new(vsa_repr(model, dim), Payload::Hypervector(payload), meta)
        .expect("well-formed reference value")
}

fn reference_payload(model: VsaModelId, op: VsaCgOp, p: &VsaProgram) -> Vec<f64> {
    let dim = p.dim;
    let a = &p.items[0];
    match model {
        VsaModelId::MapI => {
            let m = MapI::new(dim);
            match op {
                VsaCgOp::Bind | VsaCgOp::Unbind => m.bind(a, &p.items[1]).unwrap(),
                VsaCgOp::Permute => m.permute(a, p.shift.unwrap()).unwrap(),
                VsaCgOp::Bundle => {
                    let refs: Vec<&[f64]> = p.items.iter().map(Vec::as_slice).collect();
                    m.bundle(&refs).unwrap()
                }
                VsaCgOp::Similarity => unreachable!("measurement"),
            }
        }
        VsaModelId::Bsc => {
            let m = Bsc::new(dim);
            match op {
                VsaCgOp::Bind | VsaCgOp::Unbind => m.bind(a, &p.items[1]).unwrap(),
                VsaCgOp::Permute => m.permute(a, p.shift.unwrap()).unwrap(),
                VsaCgOp::Bundle => {
                    let refs: Vec<&[f64]> = p.items.iter().map(Vec::as_slice).collect();
                    m.bundle(&refs).unwrap()
                }
                VsaCgOp::Similarity => unreachable!("measurement"),
            }
        }
        VsaModelId::Hrr => {
            let m = Hrr::new(dim);
            match op {
                VsaCgOp::Bind => m.bind(a, &p.items[1]).unwrap(),
                VsaCgOp::Unbind => m.unbind(a, &p.items[1]).unwrap(),
                VsaCgOp::Permute => m.permute(a, p.shift.unwrap()).unwrap(),
                VsaCgOp::Bundle => {
                    let refs: Vec<&[f64]> = p.items.iter().map(Vec::as_slice).collect();
                    m.bundle(&refs).unwrap()
                }
                VsaCgOp::Similarity => unreachable!("measurement"),
            }
        }
        VsaModelId::Fhrr => {
            let m = Fhrr::new(dim);
            match op {
                VsaCgOp::Bind => m.bind(a, &p.items[1]).unwrap(),
                VsaCgOp::Unbind => m.unbind(a, &p.items[1]).unwrap(),
                VsaCgOp::Permute => m.permute(a, p.shift.unwrap()).unwrap(),
                VsaCgOp::Bundle => {
                    let refs: Vec<&[f64]> = p.items.iter().map(Vec::as_slice).collect();
                    m.bundle(&refs).unwrap()
                }
                VsaCgOp::Similarity => unreachable!("measurement"),
            }
        }
    }
}

fn reference_similarity(model: VsaModelId, p: &VsaProgram) -> f64 {
    let dim = p.dim;
    let (a, b) = (&p.items[0], &p.items[1]);
    match model {
        VsaModelId::MapI => MapI::new(dim).similarity(a, b),
        VsaModelId::Bsc => Bsc::new(dim).similarity(a, b),
        VsaModelId::Hrr => Hrr::new(dim).similarity(a, b),
        VsaModelId::Fhrr => Fhrr::new(dim).similarity(a, b),
    }
}

// ─── corpus (small-dim CPU-runnable, every model + op — mirrors vsa_differential.rs) ──────────────

fn bipolar(dim: u32, seed: u64) -> Vec<f64> {
    (0..u64::from(dim))
        .map(|i| {
            if (i * 2 + seed).is_multiple_of(3) {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
fn binary(dim: u32, seed: u64) -> Vec<f64> {
    (0..u64::from(dim))
        .map(|i| f64::from(u32::try_from((i * 3 + seed) % 2).unwrap()))
        .collect()
}
fn real(dim: u32, seed: u64) -> Vec<f64> {
    (0..dim)
        .map(|i| (f64::from(i) + seed as f64) * 0.125 - 0.5)
        .collect()
}
fn phase(dim: u32, seed: u64) -> Vec<f64> {
    (0..dim)
        .map(|i| {
            let t = ((u64::from(i) * 11 + seed * 7) % 19) as f64 / 19.0;
            (t * 1.8 - 0.9) * std::f64::consts::PI
        })
        .collect()
}

fn value_corpus() -> Vec<(VsaModelId, VsaCgOp, VsaProgram)> {
    let mut v = Vec::new();
    for (model, mk) in [
        (VsaModelId::MapI, bipolar as fn(u32, u64) -> Vec<f64>),
        (VsaModelId::Bsc, binary),
        (VsaModelId::Hrr, real),
        (VsaModelId::Fhrr, phase),
    ] {
        v.push((
            model,
            VsaCgOp::Bind,
            prog(VsaCgOp::Bind, model, vec![mk(8, 1), mk(8, 2)], None, None),
        ));
        v.push((
            model,
            VsaCgOp::Permute,
            prog(VsaCgOp::Permute, model, vec![mk(8, 3)], Some(3), None),
        ));
    }
    v.push((
        VsaModelId::MapI,
        VsaCgOp::Unbind,
        prog(
            VsaCgOp::Unbind,
            VsaModelId::MapI,
            vec![bipolar(8, 4), bipolar(8, 5)],
            None,
            None,
        ),
    ));
    v.push((
        VsaModelId::Bsc,
        VsaCgOp::Unbind,
        prog(
            VsaCgOp::Unbind,
            VsaModelId::Bsc,
            vec![binary(8, 4), binary(8, 5)],
            None,
            None,
        ),
    ));
    v.push((
        VsaModelId::Hrr,
        VsaCgOp::Unbind,
        prog(
            VsaCgOp::Unbind,
            VsaModelId::Hrr,
            vec![real(256, 4), real(256, 5)],
            None,
            None,
        ),
    ));
    v.push((
        VsaModelId::Fhrr,
        VsaCgOp::Unbind,
        prog(
            VsaCgOp::Unbind,
            VsaModelId::Fhrr,
            vec![phase(256, 4), phase(256, 5)],
            None,
            None,
        ),
    ));
    v.push((
        VsaModelId::MapI,
        VsaCgOp::Bundle,
        prog(
            VsaCgOp::Bundle,
            VsaModelId::MapI,
            (1..=3).map(|s| bipolar(2048, s)).collect(),
            None,
            Some(1e-2),
        ),
    ));
    v.push((
        VsaModelId::Bsc,
        VsaCgOp::Bundle,
        prog(
            VsaCgOp::Bundle,
            VsaModelId::Bsc,
            (1..=3).map(|s| binary(1024, s)).collect(),
            None,
            None,
        ),
    ));
    v.push((
        VsaModelId::Hrr,
        VsaCgOp::Bundle,
        prog(
            VsaCgOp::Bundle,
            VsaModelId::Hrr,
            (1..=3).map(|s| real(256, s)).collect(),
            None,
            None,
        ),
    ));
    v.push((
        VsaModelId::Fhrr,
        VsaCgOp::Bundle,
        prog(
            VsaCgOp::Bundle,
            VsaModelId::Fhrr,
            (1..=3).map(|s| phase(256, s)).collect(),
            None,
            None,
        ),
    ));
    v
}

fn measurement_corpus() -> Vec<(VsaModelId, VsaProgram)> {
    vec![
        (
            VsaModelId::MapI,
            prog(
                VsaCgOp::Similarity,
                VsaModelId::MapI,
                vec![bipolar(8, 1), bipolar(8, 2)],
                None,
                None,
            ),
        ),
        (
            VsaModelId::Bsc,
            prog(
                VsaCgOp::Similarity,
                VsaModelId::Bsc,
                vec![binary(8, 1), binary(8, 2)],
                None,
                None,
            ),
        ),
        (
            VsaModelId::Hrr,
            prog(
                VsaCgOp::Similarity,
                VsaModelId::Hrr,
                vec![real(8, 1), real(8, 2)],
                None,
                None,
            ),
        ),
        (
            VsaModelId::Fhrr,
            prog(
                VsaCgOp::Similarity,
                VsaModelId::Fhrr,
                vec![phase(8, 1), phase(8, 2)],
                None,
                None,
            ),
        ),
    ]
}

// ─── the differential (reference ≡ dynamic-VSA JIT, M-210-checked) ───────────────────────────────

/// Value ops: `mycelium-vsa` ≡ the in-process JIT, observably equal, validated through the M-210
/// checker. The reference's per-op tag must match the JIT read-back, and the payload must be
/// **bit-exact**.
#[test]
fn value_ops_match_reference_through_the_m210_checker() {
    for (i, (model, op, p)) in value_corpus().iter().enumerate() {
        let payload = reference_payload(*model, *op, p);
        let reference = reference_value(
            *model,
            *op,
            p.dim,
            payload,
            p.items.len() as u64,
            p.bundle_delta,
        );
        match vsa_jit_compile_and_run(p) {
            Ok(VsaResult::Value(native)) => {
                assert_eq!(
                    observable(&reference),
                    observable(&native),
                    "case #{i} ({model:?} {op:?}): reference vs JIT diverged"
                );
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
                    "case #{i}: the shared checker must validate reference↔JIT"
                );
            }
            Ok(other) => panic!("case #{i}: expected a Value, got {other:?}"),
            Err(VsaAotError::ToolchainMissing(_)) => return, // env skip
            Err(e) => panic!("case #{i} ({model:?} {op:?}): JIT errored: {e}"),
        }
    }
}

/// Measurement ops: `mycelium-vsa` ≡ JIT, **bit-exact** `f64`.
#[test]
fn measurement_ops_match_reference_bit_exact() {
    for (i, (model, p)) in measurement_corpus().iter().enumerate() {
        let reference = reference_similarity(*model, p);
        match vsa_jit_compile_and_run(p) {
            Ok(VsaResult::Measurement(native)) => {
                assert_eq!(
                    reference.to_bits(),
                    native.to_bits(),
                    "case #{i} ({model:?} similarity): reference={reference} native={native} diverged"
                );
            }
            Ok(other) => panic!("case #{i}: expected a Measurement, got {other:?}"),
            Err(VsaAotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("case #{i}: JIT errored: {e}"),
        }
    }
}

/// FHRR `-0.0` phase-wrap edge — the JIT must wrap **bit-exactly** like the reference (mirrors the AOT
/// edge case; the JIT reuses the same `emit_wrap_phase` helper, so this is really pinning the reuse).
#[test]
fn fhrr_negative_zero_phase_wraps_bit_exact() {
    let neg_zero = vec![-0.0_f64];
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Fhrr,
        vec![neg_zero.clone(), neg_zero],
        None,
        None,
    );
    let reference = Fhrr::new(1).bind(&[-0.0], &[-0.0]).unwrap();
    match vsa_jit_compile_and_run(&p) {
        Ok(VsaResult::Value(native)) => {
            let native_payload = match native.payload() {
                Payload::Hypervector(h) => h.clone(),
                other => panic!("expected a hypervector, got {other:?}"),
            };
            assert_eq!(
                reference[0].to_bits(),
                native_payload[0].to_bits(),
                "FHRR -0.0 phase wrap must be bit-exact on the JIT path too"
            );
        }
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => panic!("FHRR -0.0 bind errored: {other:?}"),
    }
}

/// FHRR degenerate bundle component — the **executed JIT kernel** returns the nonzero status (never a
/// silently-decoded garbage buffer), and the read-back surfaces an explicit
/// [`VsaAotError::DegenerateBundleComponent`], exactly where the reference refuses. Built at dim 256
/// (the `FHRR_BUNDLE_PROFILE` envelope) with two pairwise-opposite phasor items so every component
/// cancels. Exercises the real `clang`/`dlopen` artifact, not just IR inspection.
#[test]
fn fhrr_degenerate_bundle_is_refused_by_the_executed_kernel() {
    use std::f64::consts::{PI, TAU};
    let opp = |t: f64| {
        let u = (t + PI).rem_euclid(TAU);
        if u > PI {
            u - TAU
        } else {
            u
        }
    };
    let a = phase(256, 7);
    let b: Vec<f64> = a.iter().map(|&t| opp(t)).collect();
    assert_eq!(
        Fhrr::new(256).bundle(&[&a, &b]),
        Err(mycelium_vsa::VsaError::DegenerateBundleComponent { index: 0 })
    );
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, vec![a, b], None, None);
    match vsa_jit_compile_and_run(&p) {
        Err(VsaAotError::DegenerateBundleComponent) => {}
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => {
            panic!("FHRR degenerate bundle must surface DegenerateBundleComponent, got {other:?}")
        }
    }
}

// ─── non-vacuity: the JIT path actually discriminates (a divergent pair is rejected) ─────────────

/// The JIT path **discriminates**: binding the same operand against `+1` (identity) vs `-1` (negation)
/// gives opposite payloads, and the shared checker rejects the divergent pair (so the equivalence
/// above is non-vacuous, not a checker that would validate anything).
#[test]
fn jit_distinguishes_different_ops() {
    let a = bipolar(8, 1);
    let plus_one = vec![1.0; 8];
    let minus_one = vec![-1.0; 8];
    let p_pos = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        vec![a.clone(), plus_one],
        None,
        None,
    );
    let p_neg = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        vec![a, minus_one],
        None,
        None,
    );
    let (x, y) = match (
        vsa_jit_compile_and_run(&p_pos),
        vsa_jit_compile_and_run(&p_neg),
    ) {
        (Ok(VsaResult::Value(x)), Ok(VsaResult::Value(y))) => (x, y),
        (Err(VsaAotError::ToolchainMissing(_)), _) | (_, Err(VsaAotError::ToolchainMissing(_))) => {
            return
        }
        other => panic!("jit errored: {other:?}"),
    };
    assert_ne!(
        x.payload(),
        y.payload(),
        "bind(a,+1) must differ from bind(a,-1)"
    );
    assert!(
        matches!(
            check(
                &x,
                &y,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the divergent jit pair"
    );
}

/// A deliberately wrong reference (bind compared against unbind, HRR is not self-inverse) is rejected
/// by the checker — pins that a passing differential is content, not coincidence.
#[test]
fn a_divergent_lowering_is_rejected_by_the_checker() {
    let a = real(8, 1);
    let b = real(8, 2);
    let p_bind = prog(
        VsaCgOp::Bind,
        VsaModelId::Hrr,
        vec![a.clone(), b.clone()],
        None,
        None,
    );
    let native = match vsa_jit_compile_and_run(&p_bind) {
        Ok(VsaResult::Value(v)) => v,
        Err(VsaAotError::ToolchainMissing(_)) => return,
        other => panic!("hrr bind errored: {other:?}"),
    };
    let wrong_payload = Hrr::new(8).unbind(&a, &b).unwrap();
    let wrong = reference_value(VsaModelId::Hrr, VsaCgOp::Bind, 8, wrong_payload, 0, None);
    assert!(
        matches!(
            check(
                &wrong,
                &native,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the jit bind against the wrong (unbind) reference payload"
    );
}

// ─── the capacity-bound parity case (RFC-0039 §5.4) ──────────────────────────────────────────────

/// The MAP-I `bundle` issues **`Proven` iff the reference does** on the JIT path too — the checked
/// `dim ≥ requiredDim` side-condition, reused verbatim from `vsa_codegen` (never a second, potentially
/// divergent capacity check).
#[test]
fn map_i_bundle_capacity_parity_proven_iff_reference() {
    let items: Vec<Vec<f64>> = (1..=3).map(|s| bipolar(2048, s)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::MapI, items, None, Some(1e-2));
    match vsa_jit_compile_and_run(&p) {
        Ok(VsaResult::Value(native)) => {
            assert_eq!(native.meta().guarantee(), GuaranteeStrength::Proven);
            let want = proven_capacity_bound(3, 2048, 1e-2).unwrap();
            assert_eq!(native.meta().bound(), Some(&want));
        }
        Err(VsaAotError::ToolchainMissing(_)) => return,
        other => panic!("sufficient-dim MAP-I bundle errored: {other:?}"),
    }

    // Insufficient dim -> explicit InsufficientCapacity refusal, reused at emission (no toolchain
    // needed even without clang).
    let small_items: Vec<Vec<f64>> = (1..=3).map(|s| bipolar(64, s)).collect();
    let small = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        small_items,
        None,
        Some(1e-2),
    );
    match mycelium_mlir::vsa_jit_compile(&small) {
        Err(VsaAotError::InsufficientCapacity { items, dim, .. }) => {
            assert_eq!((items, dim), (3, 64));
        }
        Err(e) => panic!("insufficient-dim MAP-I bundle must be InsufficientCapacity, got {e}"),
        Ok(_) => panic!("insufficient-dim MAP-I bundle must be refused, got Ok"),
    }
}

// ─── never-silent refusals reused from vsa_codegen (SBC/MAP-B + the carrier gate) ─────────────────

/// SBC / MAP-B (and any non-mandatory model) are refused **never-silently** at the shared
/// model-resolution boundary — the JIT path does not re-implement this gate, it reuses
/// `resolve_vsa_model` (the exact function the AOT path calls).
#[test]
fn sbc_mapb_and_sparse_carrier_are_refused_never_silently_on_the_jit_path_too() {
    use mycelium_mlir::resolve_vsa_model;
    for id in ["SBC", "MAP-B", "nonsense"] {
        match resolve_vsa_model(id, SparsityClass::Dense) {
            Err(VsaAotError::UnsupportedModel(got)) => assert_eq!(got, id),
            other => panic!("{id} must be refused with UnsupportedModel, got {other:?}"),
        }
    }
    assert!(matches!(
        resolve_vsa_model("MAP-I", SparsityClass::Sparse { max_active: 8 }),
        Err(VsaAotError::UnsupportedCarrier(_))
    ));
}

// ─── the "dynamic" scenario for real: model/dim chosen from runtime data (RFC-0039 §5.3) ─────────

/// The scenario RFC-0039 §5.3 names: a **data-dependent dimension** and a **runtime-chosen model** —
/// built here from a runtime iterator (never a Rust-compile-time constant per case), each JIT-compiled
/// fresh. Every case must still match the reference bit-exactly; this is the same M-210 check as
/// [`value_ops_match_reference_through_the_m210_checker`], just driven by data the emitter has no
/// static knowledge of ahead of the call.
#[test]
fn jit_specializes_per_call_from_runtime_data() {
    // A runtime-constructed plan: model id (as a string, as if read from a config/stream) + dim, only
    // resolved to a `VsaModelId` at call time via the shared `resolve_vsa_model` boundary.
    let runtime_plan: Vec<(&str, u32)> = vec![("MAP-I", 8), ("BSC", 12), ("HRR", 16), ("FHRR", 20)];
    for (i, (model_id, dim)) in runtime_plan.into_iter().enumerate() {
        let model = mycelium_mlir::resolve_vsa_model(model_id, SparsityClass::Dense).unwrap();
        let (a, b) = match model {
            VsaModelId::MapI => (bipolar(dim, 1), bipolar(dim, 2)),
            VsaModelId::Bsc => (binary(dim, 1), binary(dim, 2)),
            VsaModelId::Hrr => (real(dim, 1), real(dim, 2)),
            VsaModelId::Fhrr => (phase(dim, 1), phase(dim, 2)),
        };
        let p = prog(VsaCgOp::Bind, model, vec![a, b], None, None);
        let reference_payload = reference_payload(model, VsaCgOp::Bind, &p);
        let reference = reference_value(model, VsaCgOp::Bind, dim, reference_payload, 2, None);
        match vsa_jit_compile_and_run(&p) {
            Ok(VsaResult::Value(native)) => {
                assert_eq!(
                    observable(&reference),
                    observable(&native),
                    "runtime-plan case #{i} ({model_id}, dim={dim}) diverged"
                );
            }
            Err(VsaAotError::ToolchainMissing(_)) => return,
            other => panic!("runtime-plan case #{i} errored: {other:?}"),
        }
    }
}
