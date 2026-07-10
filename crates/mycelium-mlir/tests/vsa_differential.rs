//! M-854 — the **native-VSA-codegen differential** (E25-1; **RFC-0039 §5.2**; NFR-7; VR-4; the
//! M-210 shared checker; RFC-0004 §6; RFC-0003 §4.1 the per-op guarantee matrix).
//!
//! The native VSA lowering (`vsa_codegen.rs`, direct-LLVM) is validated against the trusted-base
//! reference `mycelium-vsa` for the 1.0.0-native-mandatory **MAP-I / BSC / HRR / FHRR** models. The
//! interpreter registers **no `vsa.*` prims** (VSA is a standalone operational surface — like Dense
//! and the cert swap), so the reference *is* `mycelium-vsa` (the trusted base, NFR-7), and the
//! candidate is the **direct-LLVM** native artifact. The two must be **observably equal** (`repr +
//! payload + guarantee`), validated through the **single shared M-210 checker**
//! (`mycelium_cert::check` under `RefinementRelation::ObservationalEquiv`).
//!
//! **Bit-exact by construction.** The native lowering computes every component in `f64` (`double`),
//! mirroring the reference's `f64` arithmetic digit-for-digit and in the same operation order (it
//! calls the same libm `cos`/`sin`/`atan2` for FHRR), so the read-back hypervector is **bit-identical**
//! to `mycelium-vsa`'s — exactly what the M-210 observational checker requires (it compares
//! `Payload::Hypervector` bit-exactly). A deliberately divergent lowering is caught (a mutation of the
//! lowering diverges here), so a passing differential is meaningful, not vacuous.
//!
//! **The MLIR-dialect leg (M-856b).** The **generic bit/trit `Node` path** still honestly refuses a
//! VSA `Const` (`DialectError::Unsupported` naming "Dense/VSA stay on the interpreter / direct-LLVM
//! path", `dialect/native.rs::const_lane`) — that boundary is permanent on both backends (VSA is
//! lowered through the dedicated `VsaProgram` entry points, never the generic `Node` path);
//! [`vsa_const_is_refused_by_the_mlir_dialect_path`] still asserts it. But `dialect::native::vsa`
//! (M-856b) now provides a **dialect-native sibling** of `vsa_compile_and_run` over the *same*
//! `VsaProgram` — covering `bind`/`unbind`/`bundle`/`permute`/`similarity` over all four
//! 1.0.0-mandatory models — so the differential is a genuine **three-way** (reference ≡ direct-LLVM ≡
//! dialect) over the existing small-dim corpus where libMLIR is provisioned — skip-graceful
//! (`VsaAotError::ToolchainMissing`) where it is not, never a faked pass (VR-5/G2). See
//! [`value_ops_dialect_matches_reference_and_direct_llvm`] /
//! [`measurement_ops_dialect_matches_reference_bit_exact`]. Kept deliberately to the **existing**
//! light small-dim corpus (no new heavy/large-dim VSA test infrastructure — the heavy GPU-scale
//! mutant-durability pass stays the maintainer's follow-up, per the module doc above).
//!
//! **The capacity-bound parity case** (RFC-0039 §5.4): the native MAP-I `bundle` issues `Proven` **iff**
//! the reference does (the checked `dim ≥ requiredDim` side-condition), and refuses
//! (`InsufficientCapacity`) otherwise — never an unbacked `Proven` (VR-5).
//!
//! **Honest tags carried** (RFC-0003 §4.1, VR-5): `Exact` (bind/permute, MAP-I/BSC unbind), `Empirical`
//! (HRR/FHRR unbind via the reference's profile; BSC bundle via `BSC_BUNDLE_PROFILE`; HRR/FHRR bundle
//! via the codegen-derived `HRR_BUNDLE_PROFILE`/`FHRR_BUNDLE_PROFILE`, earned by measured trials — the
//! M-854 FLAG-0 resolution, moved from `Declared` to `Empirical`-within-envelope), `Proven` (MAP-I
//! bundle via the checked capacity bound). Outside an op's measured envelope it is refused
//! `OutsideEmpiricalProfile`, never claimed past what was measured.
//!
//! **Toolchain skip.** The direct-LLVM path needs `llc`/`clang`; where absent it returns
//! `ToolchainMissing` and the path **skips** (the house idiom) — never a false failure.
//!
//! **GPU siderail (maintainer constraint, 2026-06-30).** The **heavy** VSA test set (large-dim /
//! many-vector corpora) requires a GPU this environment lacks; those tests are
//! `#[ignore = "heavy VSA — requires GPU; maintainer local follow-up (2026-06-30 PM)"]` — NOT run here,
//! NOT claimed passed, visibly deferred (G2). They run locally via
//! `cargo test -p mycelium-mlir -- --ignored`. The CPU-runnable small-dim differential below proceeds
//! normally and gates the PR (the lowering is dim-independent, so the small-dim corpus exercises every
//! op/model/arithmetic path; the GPU set is a *scale/throughput* follow-up, not a different code path).
//!
//! **Guarantee:** `Empirical` — the differential is empirical evidence the native VSA codegen agrees
//! with the trusted `mycelium-vsa` reference over the corpus; never upgraded to `Proven` without a
//! checked proof object linked into codegen (VR-5).

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{
    operation_hash, Bound, GuaranteeStrength, Meta, Payload, PhysicalLayout, Provenance, Repr,
    SparsityClass, Value,
};
use mycelium_mlir::{
    vsa_compile_and_run, VsaAotError, VsaCgOp, VsaModelId, VsaProgram, VsaResult,
    FHRR_BUNDLE_PROFILE, HRR_BUNDLE_PROFILE,
};
use mycelium_numerics::Certificate;
use mycelium_vsa::bsc::BSC_BUNDLE_PROFILE;
use mycelium_vsa::capacity::proven_capacity_bound;
use mycelium_vsa::fhrr::FHRR_UNBIND_PROFILE;
use mycelium_vsa::hrr::HRR_UNBIND_PROFILE;
use mycelium_vsa::{Bsc, Fhrr, Hrr, MapI, VsaModel};

// ─── observable + helpers ────────────────────────────────────────────────────────────────────────

/// The NFR-7 observable: `(repr, payload, guarantee)`. The native read-back reconstructs the
/// reference's per-op tag, so the two observables coincide.
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

/// Build the VSA `Repr` for a model at `dim` (dense).
fn vsa_repr(model: VsaModelId, dim: u32) -> Repr {
    Repr::Vsa {
        model: model.registry_id().to_owned(),
        dim,
        sparsity: SparsityClass::Dense,
    }
}

/// Build a reference `Value` from a raw hypervector payload + a `(model, op)` Meta matching what the
/// native read-back stamps (the same provenance op key + the same guarantee/bound). This is the
/// trusted-base observable the native artifact is checked against: the **payload** is `mycelium-vsa`'s
/// own algebra output (computed below via the model trait), and the **Meta** mirrors the reference's
/// value-level surface (or the honest downgrade where it has none).
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

/// Compute the reference hypervector payload for a value op via the `mycelium-vsa` model trait (the
/// trusted base's own algebra).
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

/// Compute the reference `f64` measurement (`similarity`) via the model trait.
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

// ─── corpus (data-driven; small-dim CPU-runnable, every model + op) ──────────────────────────────

/// A small bipolar (`±1`) vector (MAP-I) from a deterministic pattern.
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
/// A small binary (`{0,1}`) vector (BSC).
fn binary(dim: u32, seed: u64) -> Vec<f64> {
    (0..u64::from(dim))
        .map(|i| f64::from(u32::try_from((i * 3 + seed) % 2).unwrap()))
        .collect()
}
/// A small real vector (HRR).
fn real(dim: u32, seed: u64) -> Vec<f64> {
    (0..dim)
        .map(|i| (f64::from(i) + seed as f64) * 0.125 - 0.5)
        .collect()
}
/// A small in-range phase vector (FHRR), each strictly inside `(−π, π]`. Built from a bounded
/// fraction of `π` so no component can ever land on/past the alphabet boundary the reference refuses.
fn phase(dim: u32, seed: u64) -> Vec<f64> {
    (0..dim)
        .map(|i| {
            // a deterministic value in [−0.9π, 0.9π] (strictly inside the (−π, π] phasor alphabet).
            let t = ((u64::from(i) * 11 + seed * 7) % 19) as f64 / 19.0; // [0, 1)
            (t * 1.8 - 0.9) * std::f64::consts::PI
        })
        .collect()
}

/// The value-op corpus: `(model, op, program)` triples over the four mandatory models, small-dim,
/// alphabet-valid, in the lowerable regime (HRR/FHRR unbind at dim 256; BSC bundle at dim 1024; MAP-I
/// bundle at dim 2048 ≥ requiredDim). Each case's reference is the `mycelium-vsa` trait; the native
/// artifact must match it (repr + payload + guarantee).
fn value_corpus() -> Vec<(VsaModelId, VsaCgOp, VsaProgram)> {
    let mut v = Vec::new();
    // bind / permute (Exact) at small dim 8 for every model.
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
    // unbind: MAP-I/BSC Exact (small dim); HRR/FHRR Empirical (dim 256 — the profile minimum).
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
    // bundle: MAP-I Proven (dim 2048, δ 1e-2 — distinct items so the value-level distinctness holds);
    // BSC Empirical (dim 1024, 3 items); HRR/FHRR Empirical (dim 256, 3 items — in the measured
    // *_BUNDLE_PROFILE envelope: m ≤ 5, d ≥ 256).
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

/// The measurement corpus: `similarity` over each model (bare-`f64`, no `Meta`).
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
        // zero-vector cosine → 0 (the zero-norm guard) for MAP-I.
        (
            VsaModelId::MapI,
            prog(
                VsaCgOp::Similarity,
                VsaModelId::MapI,
                vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, -1.0, 1.0, -1.0]],
                None,
                None,
            ),
        ),
    ]
}

// ─── the differential (reference ≡ direct-LLVM, M-210-checked) ───────────────────────────────────

/// Value ops: `mycelium-vsa` ≡ direct-LLVM, observably equal, validated through the M-210 checker. The
/// reference's per-op tag (Exact / Empirical / Proven / Declared) must match the native read-back, and
/// the payload must be **bit-exact** (native f64 mirrors the reference's f64 op-for-op).
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
        match vsa_compile_and_run(p) {
            Ok(VsaResult::Value(native)) => {
                assert_eq!(
                    observable(&reference),
                    observable(&native),
                    "case #{i} ({model:?} {op:?}): reference vs direct-LLVM diverged"
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
                    "case #{i}: the shared checker must validate reference↔native"
                );
            }
            Ok(other) => panic!("case #{i}: expected a Value, got {other:?}"),
            Err(VsaAotError::ToolchainMissing(_)) => return, // env skip
            Err(e) => panic!("case #{i} ({model:?} {op:?}): direct-LLVM errored: {e}"),
        }
    }
}

/// Measurement ops: `mycelium-vsa` ≡ direct-LLVM, **bit-exact** `f64` (the native reduction folds
/// left-to-right exactly as the reference's `.sum()` and calls the same libm trig).
#[test]
fn measurement_ops_match_reference_bit_exact() {
    for (i, (model, p)) in measurement_corpus().iter().enumerate() {
        let reference = reference_similarity(*model, p);
        match vsa_compile_and_run(p) {
            Ok(VsaResult::Measurement(native)) => {
                assert_eq!(
                    reference.to_bits(),
                    native.to_bits(),
                    "case #{i} ({model:?} similarity): reference={reference} native={native} diverged"
                );
            }
            Ok(other) => panic!("case #{i}: expected a Measurement, got {other:?}"),
            Err(VsaAotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("case #{i}: direct-LLVM errored: {e}"),
        }
    }
}

/// FHRR `-0.0` phase-wrap edge: a phase sum that lands on `-0.0` must wrap **bit-exactly** like the
/// reference's `f64::rem_euclid` (which preserves the `-0.0` sign), not flip to `+0.0`. The native
/// `wrap_phase` emits `frem` + the `if r<0 {r+TAU}` algorithm precisely for this reason (a `floor`-based
/// identity would flip the sign here — verified over 2·10⁶ samples). `bind([-0.0], [-0.0]) = wrap(-0.0)`
/// exercises the edge: `(-0.0)+(-0.0) = -0.0` reaches `wrap_phase`. The reference and native must agree
/// bit-for-bit (G2 — never a silent ±0 divergence).
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
    match vsa_compile_and_run(&p) {
        Ok(VsaResult::Value(native)) => {
            let native_payload = match native.payload() {
                Payload::Hypervector(h) => h.clone(),
                other => panic!("expected a hypervector, got {other:?}"),
            };
            assert_eq!(
                reference[0].to_bits(),
                native_payload[0].to_bits(),
                "FHRR -0.0 phase wrap must be bit-exact (ref={}, native={}); the frem-based \
                 rem_euclid preserves the -0.0 sign",
                reference[0],
                native_payload[0]
            );
        }
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => panic!("FHRR -0.0 bind errored: {other:?}"),
    }
}

/// FHRR degenerate bundle component: when a component's phasor sum vanishes, its phase is undefined —
/// the **executed native artifact** must print the `DEGENERATE` sentinel and the read-back surfaces an
/// explicit [`VsaAotError::DegenerateBundleComponent`], **exactly where the reference refuses**
/// (`VsaError::DegenerateBundleComponent`) — never an arbitrary phase (G2/SC-3). Built at dim 256 (the
/// `FHRR_BUNDLE_PROFILE` envelope, so the profile gate passes and the runtime degenerate-trap is the
/// surfaced failure) with two pairwise-opposite phasor items (`θ` and `wrap(θ + π)`) — every component
/// cancels, so component 0 is degenerate. This exercises the never-silent trap through a real
/// `llc`/`clang` artifact, not just IR inspection.
#[test]
fn fhrr_degenerate_bundle_is_refused_by_the_executed_artifact() {
    use std::f64::consts::{PI, TAU};
    // The opposite phasor of θ is wrap(θ + π) in (−π, π] — `rem_euclid` then shift, matching the
    // reference's `wrap_phase`.
    let opp = |t: f64| {
        let u = (t + PI).rem_euclid(TAU);
        if u > PI {
            u - TAU
        } else {
            u
        }
    };
    let a = phase(256, 7); // in-range phasor vector (alphabet-valid)
    let b: Vec<f64> = a.iter().map(|&t| opp(t)).collect(); // pairwise opposite — every component cancels
                                                           // Reference: the bundle of opposite phasors is a degenerate-component refusal at index 0.
    assert_eq!(
        Fhrr::new(256).bundle(&[&a, &b]),
        Err(mycelium_vsa::VsaError::DegenerateBundleComponent { index: 0 })
    );
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, vec![a, b], None, None);
    match vsa_compile_and_run(&p) {
        Err(VsaAotError::DegenerateBundleComponent) => {}
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => {
            panic!("FHRR degenerate bundle must surface DegenerateBundleComponent, got {other:?}")
        }
    }
}

// ─── non-vacuity: the native path actually discriminates ─────────────────────────────────────────

/// Sanity: the native VSA path **discriminates** — binding the same operand against `+1` (the
/// identity, yielding `a`) vs against `−1` (yielding `−a`) gives **opposite** payloads, and the shared
/// checker reports the divergence (so the equivalence above is non-vacuous). Using the identity/negation
/// binders makes the discrimination robust to operand symmetry (a random permute can coincide).
#[test]
fn native_vsa_distinguishes_different_ops() {
    let a = bipolar(8, 1);
    let plus_one = vec![1.0; 8]; // bind(a, +1) = a
    let minus_one = vec![-1.0; 8]; // bind(a, −1) = −a
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
        vec![a.clone(), minus_one],
        None,
        None,
    );
    let (x, y) = match (vsa_compile_and_run(&p_pos), vsa_compile_and_run(&p_neg)) {
        (Ok(VsaResult::Value(x)), Ok(VsaResult::Value(y))) => (x, y),
        (Err(VsaAotError::ToolchainMissing(_)), _) | (_, Err(VsaAotError::ToolchainMissing(_))) => {
            return
        }
        other => panic!("native vsa errored: {other:?}"),
    };
    assert_ne!(
        x.payload(),
        y.payload(),
        "bind(a, +1) = a must differ from bind(a, −1) = −a (the payloads must differ)"
    );
    // The shared checker rejects the divergent pair (never a vacuous pass).
    assert!(
        matches!(
            check(
                &x,
                &y,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the divergent vsa pair"
    );
}

/// A deliberately wrong lowering (a bind result compared against the reference's *unbind* result for a
/// non-self-inverse model) is rejected by the checker — pins that the differential's pass is content,
/// not coincidence. HRR bind ≠ HRR unbind (HRR is not self-inverse).
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
    let native = match vsa_compile_and_run(&p_bind) {
        Ok(VsaResult::Value(v)) => v,
        Err(VsaAotError::ToolchainMissing(_)) => return,
        other => panic!("hrr bind errored: {other:?}"),
    };
    // The WRONG reference: HRR unbind (the involution-correlation), which differs from bind.
    let wrong_payload = Hrr::new(8).unbind(&a, &b).unwrap();
    let wrong = reference_value(VsaModelId::Hrr, VsaCgOp::Bind, 8, wrong_payload, 0, None);
    assert!(
        matches!(
            check(
                &wrong,
                &native,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the native bind against the wrong (unbind) reference payload"
    );
}

// ─── the capacity-bound parity case (RFC-0039 §5.4) ──────────────────────────────────────────────

/// The MAP-I `bundle` issues **`Proven` iff the reference does** — the checked `dim ≥ requiredDim`
/// side-condition. At dim 2048 (≥ requiredDim(3, 1e-2) = 1141) the native bundle is `Proven` carrying
/// the same capacity bound as the reference; at dim 64 (< 1141) it is refused
/// (`InsufficientCapacity`), never an unbacked `Proven` (VR-5/M-I2). This is the §5.4 capacity-parity
/// obligation.
#[test]
fn map_i_bundle_capacity_parity_proven_iff_reference() {
    // Sufficient dim → Proven, with the same checked capacity bound as the reference.
    let items: Vec<Vec<f64>> = (1..=3).map(|s| bipolar(2048, s)).collect();
    let p = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        items.clone(),
        None,
        Some(1e-2),
    );
    match vsa_compile_and_run(&p) {
        Ok(VsaResult::Value(native)) => {
            assert_eq!(
                native.meta().guarantee(),
                GuaranteeStrength::Proven,
                "sufficient dim must yield a Proven bundle (the checked capacity bound)"
            );
            // The native bound is exactly the reference's checked capacity bound.
            let want = proven_capacity_bound(3, 2048, 1e-2).unwrap();
            assert_eq!(
                native.meta().bound(),
                Some(&want),
                "the native Proven bound must be the reference's checked capacity bound (DRY/VR-5)"
            );
        }
        Err(VsaAotError::ToolchainMissing(_)) => return,
        other => panic!("sufficient-dim MAP-I bundle errored: {other:?}"),
    }

    // Insufficient dim → explicit InsufficientCapacity refusal (no unbacked Proven). This is at
    // lowering (no toolchain needed), so it holds even without llc/clang.
    let small_items: Vec<Vec<f64>> = (1..=3).map(|s| bipolar(64, s)).collect();
    let small = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        small_items,
        None,
        Some(1e-2),
    );
    match mycelium_mlir::emit_vsa_llvm_ir(&small) {
        Err(VsaAotError::InsufficientCapacity {
            items,
            dim,
            required,
        }) => {
            assert_eq!((items, dim), (3, 64));
            assert!(required > 64, "the theorem requires more than dim 64");
        }
        other => panic!("insufficient-dim MAP-I bundle must be refused, got {other:?}"),
    }
    // And the reference agrees the side-condition fails.
    assert_eq!(
        proven_capacity_bound(3, 64, 1e-2),
        None,
        "the reference also refuses a Proven bound at dim 64 (parity)"
    );
}

// ─── the MLIR-dialect leg (M-856b): reference ≡ direct-LLVM ≡ dialect ────────────────────────────

/// Value ops: `mycelium-vsa` (reference) ≡ direct-LLVM ≡ **dialect**, over the existing small-dim
/// corpus (every model × op) — a genuine three-way, libMLIR-gated (skip-graceful, never a faked
/// pass). `ran` tracks non-vacuity (the M-725 `ran_mlir` discipline): kept **light** per the module
/// doc — no new heavy corpus, just the existing light corpus run through one more (real) backend.
#[cfg(feature = "mlir-dialect")]
#[test]
fn value_ops_dialect_matches_reference_and_direct_llvm() {
    use mycelium_mlir::dialect::native::vsa::dialect_compile_and_run;
    use mycelium_mlir::MlirTools;
    let mut ran = false;
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
        let direct = match vsa_compile_and_run(p) {
            Ok(VsaResult::Value(v)) => *v,
            Err(VsaAotError::ToolchainMissing(_)) => continue, // direct-LLVM env skip
            other => panic!("case #{i} ({model:?} {op:?}): direct-LLVM unexpected: {other:?}"),
        };
        match dialect_compile_and_run(p) {
            Ok(VsaResult::Value(dialect)) => {
                ran = true;
                assert_eq!(
                    observable(&reference),
                    observable(&dialect),
                    "case #{i} ({model:?} {op:?}): reference vs dialect diverged"
                );
                assert_eq!(
                    observable(&direct),
                    observable(&dialect),
                    "case #{i} ({model:?} {op:?}): direct-LLVM vs dialect diverged"
                );
            }
            Ok(other) => panic!("case #{i}: expected a Value, got {other:?}"),
            Err(VsaAotError::ToolchainMissing(_)) => continue, // dialect env skip
            Err(e) => panic!("case #{i} ({model:?} {op:?}): dialect errored: {e}"),
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

/// Measurement ops (`similarity`): reference ≡ direct-LLVM ≡ dialect, bit-exact, over every model.
#[cfg(feature = "mlir-dialect")]
#[test]
fn measurement_ops_dialect_matches_reference_bit_exact() {
    use mycelium_mlir::dialect::native::vsa::dialect_compile_and_run;
    use mycelium_mlir::MlirTools;
    let mut ran = false;
    for (i, (model, p)) in measurement_corpus().iter().enumerate() {
        let reference = reference_similarity(*model, p);
        let direct = match vsa_compile_and_run(p) {
            Ok(VsaResult::Measurement(m)) => m,
            Err(VsaAotError::ToolchainMissing(_)) => continue,
            other => panic!("case #{i}: direct-LLVM unexpected: {other:?}"),
        };
        match dialect_compile_and_run(p) {
            Ok(VsaResult::Measurement(dialect)) => {
                ran = true;
                assert_eq!(
                    reference.to_bits(),
                    dialect.to_bits(),
                    "case #{i} ({model:?} similarity): reference vs dialect diverged"
                );
                assert_eq!(
                    direct.to_bits(),
                    dialect.to_bits(),
                    "case #{i} ({model:?} similarity): direct-LLVM vs dialect diverged"
                );
            }
            Ok(other) => panic!("case #{i}: expected a Measurement, got {other:?}"),
            Err(VsaAotError::ToolchainMissing(_)) => continue,
            Err(e) => panic!("case #{i}: dialect errored: {e}"),
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

/// The dialect leg's FHRR `-0.0` phase-wrap edge, mirroring
/// [`fhrr_negative_zero_phase_wraps_bit_exact`]: the dialect emitter uses `llvm.frem` (not
/// `arith.remf`, which is MLIR's *IEEE remainder* — verified empirically to disagree with `fmod`;
/// see `dialect::native::vsa`'s module doc), so it must preserve the `-0.0` sign exactly like the
/// reference and the direct-LLVM path.
#[cfg(feature = "mlir-dialect")]
#[test]
fn fhrr_negative_zero_phase_wraps_bit_exact_through_the_dialect() {
    use mycelium_mlir::dialect::native::vsa::dialect_compile_and_run;
    let neg_zero = vec![-0.0_f64];
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::Fhrr,
        vec![neg_zero.clone(), neg_zero],
        None,
        None,
    );
    let reference = Fhrr::new(1).bind(&[-0.0], &[-0.0]).unwrap();
    match dialect_compile_and_run(&p) {
        Ok(VsaResult::Value(dialect)) => {
            let dialect_payload = match dialect.payload() {
                Payload::Hypervector(h) => h.clone(),
                other => panic!("expected a hypervector, got {other:?}"),
            };
            assert_eq!(
                reference[0].to_bits(),
                dialect_payload[0].to_bits(),
                "FHRR -0.0 phase wrap must be bit-exact through the dialect too (ref={}, \
                 dialect={})",
                reference[0],
                dialect_payload[0]
            );
        }
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => panic!("FHRR -0.0 bind (dialect) errored: {other:?}"),
    }
}

/// The dialect leg's FHRR degenerate-bundle refusal, mirroring
/// [`fhrr_degenerate_bundle_is_refused_by_the_executed_artifact`]: the `cf.cond_br` branch to the
/// `DEGENERATE` sentinel must fire through the real compiled dialect artifact too, never an
/// arbitrary phase (G2/SC-3).
#[cfg(feature = "mlir-dialect")]
#[test]
fn fhrr_degenerate_bundle_is_refused_by_the_dialect_artifact() {
    use mycelium_mlir::dialect::native::vsa::dialect_compile_and_run;
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
    let p = prog(VsaCgOp::Bundle, VsaModelId::Fhrr, vec![a, b], None, None);
    match dialect_compile_and_run(&p) {
        Err(VsaAotError::DegenerateBundleComponent) => {}
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => panic!(
            "FHRR degenerate bundle (dialect) must surface DegenerateBundleComponent, got {other:?}"
        ),
    }
}

/// The dialect leg carries the **same** MAP-I bundle capacity-parity discipline as direct-LLVM
/// (RFC-0039 §5.4's capacity-bound parity case) — inherited for free from the shared
/// `VsaProgram::validate` (DRY), asserted directly here so the parity is pinned on this leg too.
#[cfg(feature = "mlir-dialect")]
#[test]
fn map_i_bundle_capacity_parity_proven_iff_reference_through_the_dialect() {
    use mycelium_mlir::dialect::native::vsa::{dialect_compile_and_run, emit_vsa_mlir};
    let items: Vec<Vec<f64>> = (1..=3).map(|s| bipolar(2048, s)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::MapI, items, None, Some(1e-2));
    match dialect_compile_and_run(&p) {
        Ok(VsaResult::Value(dialect)) => {
            assert_eq!(
                dialect.meta().guarantee(),
                GuaranteeStrength::Proven,
                "sufficient dim must yield a Proven bundle through the dialect too"
            );
            let want = proven_capacity_bound(3, 2048, 1e-2).unwrap();
            assert_eq!(
                dialect.meta().bound(),
                Some(&want),
                "the dialect Proven bound must be the reference's checked capacity bound (DRY/VR-5)"
            );
        }
        Err(VsaAotError::ToolchainMissing(_)) => return,
        other => panic!("sufficient-dim MAP-I bundle (dialect) errored: {other:?}"),
    }

    let small_items: Vec<Vec<f64>> = (1..=3).map(|s| bipolar(64, s)).collect();
    let small = prog(
        VsaCgOp::Bundle,
        VsaModelId::MapI,
        small_items,
        None,
        Some(1e-2),
    );
    match emit_vsa_mlir(&small) {
        Err(VsaAotError::InsufficientCapacity { items, dim, .. }) => {
            assert_eq!((items, dim), (3, 64));
        }
        other => panic!("insufficient-dim MAP-I bundle (dialect) must be refused, got {other:?}"),
    }
}

// ─── never-silent refusals: SBC/MAP-B + the carrier gate (G2) ────────────────────────────────────

/// SBC / MAP-B (and any non-mandatory model) are refused **never-silently** at the model-resolution
/// boundary (`resolve_vsa_model`) — they are not in the 1.0.0-native-mandatory set {MAP-I, BSC, HRR,
/// FHRR} (OQ-3). The reference *does* implement them (so they stay interpreter/reference-served), but
/// the native path refuses with an explicit `UnsupportedModel`, never silently serving a different
/// model (G2). A **sparse carrier** is likewise an explicit `UnsupportedCarrier` (the ADR-031 carrier
/// is not yet in the value model — E20-1 gate), never silently flattened to dense.
#[test]
fn sbc_mapb_and_sparse_carrier_are_refused_never_silently() {
    use mycelium_core::SparsityClass;
    use mycelium_mlir::resolve_vsa_model;
    // SBC / MAP-B / unknown → explicit UnsupportedModel (the reference serves them; the native path
    // refuses and routes there — NFR-7).
    for id in ["SBC", "MAP-B", "MAP-C", "VTB", "nonsense"] {
        match resolve_vsa_model(id, SparsityClass::Dense) {
            Err(VsaAotError::UnsupportedModel(got)) => assert_eq!(got, id),
            other => panic!("{id} must be refused with UnsupportedModel, got {other:?}"),
        }
    }
    // The four mandatory models resolve (dense).
    for (id, want) in [
        ("MAP-I", VsaModelId::MapI),
        ("BSC", VsaModelId::Bsc),
        ("HRR", VsaModelId::Hrr),
        ("FHRR", VsaModelId::Fhrr),
    ] {
        assert_eq!(resolve_vsa_model(id, SparsityClass::Dense), Ok(want));
    }
    // A sparse carrier on a mandatory model is still refused (ADR-031/E20-1 gate) — never flattened.
    assert!(matches!(
        resolve_vsa_model("MAP-I", SparsityClass::Sparse { max_active: 8 }),
        Err(VsaAotError::UnsupportedCarrier(_))
    ));
}

// ─── the dialect leg honestly refuses VSA (the third edge is a never-faked refusal) ──────────────

/// The **MLIR-dialect** path honestly **refuses** a VSA `Const` (`DialectError::Unsupported` naming
/// "Dense/VSA stay on the interpreter / direct-LLVM path") — so the three-way reduces to a two-way for
/// VSA, never a faked third pass (VR-5/G2). Asserting the refusal keeps the coverage honest: the
/// dialect path never silently mis-lowers a VSA value.
#[cfg(feature = "mlir-dialect")]
#[test]
fn vsa_const_is_refused_by_the_mlir_dialect_path() {
    use mycelium_core::{Meta, Node, Provenance};
    use mycelium_mlir::DialectError;
    let vsa_val = Value::new(
        vsa_repr(VsaModelId::MapI, 4),
        Payload::Hypervector(vec![1.0, -1.0, 1.0, -1.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Const(vsa_val);
    match mycelium_mlir::mlir_compile_and_run(&node) {
        Err(DialectError::Unsupported(msg)) => {
            assert!(
                msg.contains("VSA") || msg.contains("dialect fragment"),
                "the refusal must name the VSA/dialect-fragment boundary; got: {msg}"
            );
        }
        Err(DialectError::ToolchainMissing(_)) => { /* env skip — still no silent success */ }
        Ok(v) => panic!(
            "the MLIR-dialect path must refuse a VSA const, got {:?}",
            v.payload()
        ),
        Err(e) => panic!("unexpected MLIR-dialect error on a VSA const: {e}"),
    }
}

/// The direct-LLVM `const_lane` path also refuses a VSA `Const` through the **generic bit/trit Node
/// lowering** (`AotError::UnsupportedRepr`) — VSA is lowered through the dedicated `vsa_codegen` entry
/// points, never the generic bit/trit `Node` path, so the generic refusal stays in place (G2). This
/// keeps the two lowering surfaces cleanly separated and never silently cross-wired.
#[test]
fn vsa_const_is_refused_by_the_generic_bit_trit_node_path() {
    use mycelium_core::{Meta, Node, Provenance};
    use mycelium_mlir::AotError;
    let vsa_val = Value::new(
        vsa_repr(VsaModelId::Bsc, 4),
        Payload::Hypervector(vec![1.0, 0.0, 1.0, 0.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Const(vsa_val);
    // emit_llvm_ir refuses at const_lane (before any toolchain), so this holds even without llc/clang.
    match mycelium_mlir::emit_llvm_ir(&node) {
        Err(AotError::UnsupportedRepr(msg)) => {
            assert!(
                msg.contains("Vsa"),
                "the generic bit/trit refusal must name Vsa; got: {msg}"
            );
        }
        other => panic!("the generic Node path must refuse a VSA const, got {other:?}"),
    }
}

// ─── GPU-deferred heavy set (siderail) — NOT run here; run with `--ignored` ───────────────────────

/// **GPU-deferred (siderail, 2026-06-30).** The heavy VSA differential at literature-scale dimensions
/// (`dim ≥ 1024` across many vectors / the trial-validated profile dims) requires a GPU this
/// environment lacks. This test is `#[ignore]` — NOT run here and NOT claimed passed; run it locally
/// with `cargo test -p mycelium-mlir -- --ignored`. The lowering is dim-independent, so the small-dim
/// CPU differential above already witnesses every op/model/arithmetic path; this is a **scale**
/// follow-up (throughput at HDC-realistic dims), not a different code path. Any correctness claim that
/// depends *only* on this not-run test stays `Declared` until the maintainer's GPU pass (VR-5).
#[test]
#[ignore = "heavy VSA — requires GPU; maintainer local follow-up (2026-06-30 PM)"]
fn heavy_large_dim_differential_gpu_deferred() {
    // dim 1024 MAP-I bind + bundle + HRR convolution at the literature scale — the same lowering,
    // exercised at GPU-realistic size. Runs on the maintainer's GPU box; here it is a visible defer.
    let a = bipolar(1024, 1);
    let b = bipolar(1024, 2);
    let p = prog(
        VsaCgOp::Bind,
        VsaModelId::MapI,
        vec![a.clone(), b.clone()],
        None,
        None,
    );
    let reference = reference_value(
        VsaModelId::MapI,
        VsaCgOp::Bind,
        1024,
        reference_payload(VsaModelId::MapI, VsaCgOp::Bind, &p),
        2,
        None,
    );
    match vsa_compile_and_run(&p) {
        Ok(VsaResult::Value(native)) => {
            assert_eq!(observable(&reference), observable(&native));
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
                }
            );
        }
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => panic!("heavy MAP-I bind errored: {other:?}"),
    }
}

/// **GPU-deferred (siderail, 2026-06-30) — the HRR/FHRR bundle *profile-extension* trial.** The CPU
/// `HRR_BUNDLE_PROFILE`/`FHRR_BUNDLE_PROFILE` envelope (m ≤ 5, d ≥ 256, codebook 16) is the **earned**
/// Empirical basis here; **widening** it (larger dim, more components / larger codebook — the heavy
/// HDC-realistic regime) requires the many-trial Monte-Carlo at scale that needs a GPU this environment
/// lacks. This test sketches that heavy profiling (a large-dim, many-component bundle decode) and is
/// `#[ignore]` — NOT run here, NOT claimed passed. Until the maintainer's GPU pass widens the profile,
/// **native HRR/FHRR bundle beyond the CPU-measured envelope honestly refuses `OutsideEmpiricalProfile`**
/// (asserted CPU-side in `hrr_fhrr_bundle_outside_profile_is_refused`) — the bound is never claimed past
/// what was measured (VR-5). Run locally with `cargo test -p mycelium-mlir -- --ignored`.
#[test]
#[ignore = "heavy VSA — requires GPU; maintainer local follow-up (2026-06-30 PM)"]
fn heavy_hrr_fhrr_bundle_profile_extension_gpu_deferred() {
    // A large-dim bundle decode at m = 11, dim = 4096 — outside the CPU envelope (m > 5). On a GPU the
    // many-trial validation would widen HRR_BUNDLE_PROFILE/FHRR_BUNDLE_PROFILE to cover it; here we only
    // (a) confirm the lowering still compiles+runs bit-exactly at scale (dim-independence), and
    // (b) document that the *tag* for this regime stays refused until the GPU profile widens.
    let items: Vec<Vec<f64>> = (1..=11u64).map(|s| real(4096, s)).collect();
    // CPU truth #1: beyond the measured envelope, the native path refuses (never an unearned Empirical).
    let cpu = prog(VsaCgOp::Bundle, VsaModelId::Hrr, items.clone(), None, None);
    assert!(
        matches!(
            mycelium_mlir::emit_vsa_llvm_ir(&cpu),
            Err(VsaAotError::OutsideEmpiricalProfile(_))
        ),
        "m=11 is outside the CPU-measured HRR bundle envelope — must refuse until a GPU pass widens it"
    );
    // CPU truth #2 (the bit-exact lowering is dim-independent): the *payload* of an in-envelope large
    // bundle still matches the reference. A GPU pass would run this decode-trial at scale; here we just
    // exercise the compiled artifact at a larger-but-in-envelope point (m = 5, dim = 4096).
    let big: Vec<Vec<f64>> = (1..=5u64).map(|s| real(4096, s)).collect();
    let p = prog(VsaCgOp::Bundle, VsaModelId::Hrr, big, None, None);
    let payload = reference_payload(VsaModelId::Hrr, VsaCgOp::Bundle, &p);
    let reference = reference_value(VsaModelId::Hrr, VsaCgOp::Bundle, 4096, payload, 5, None);
    match vsa_compile_and_run(&p) {
        Ok(VsaResult::Value(native)) => {
            assert_eq!(observable(&reference), observable(&native));
        }
        Err(VsaAotError::ToolchainMissing(_)) => {}
        other => panic!("heavy in-envelope HRR bundle errored: {other:?}"),
    }
}
