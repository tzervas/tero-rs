//! M-250 acceptance — the **schedule-staged packing selector** (E2-7; RFC-0004 §5; DN-01;
//! RFC-0005 §4): a cost model evaluated **exhaustively** over the fixed bitnet.cpp candidate set
//! (`I2_S`/`TL1`/`TL2`) through the *one* selection mechanism (`select_packing`), choosing a
//! [`PhysicalLayout`] recorded on `Meta.physical` (M-I5 lossless). Determinism + override are
//! pinned here; the E3 wrong-layout soundness differential is M-251 (`mycelium-mlir`).

use mycelium_core::{
    GuaranteeStrength, Meta, PackScheme, PhysicalLayout, Provenance, Repr, ScalarKind,
    SparsityClass,
};
use mycelium_select::{
    bitnet_packing_policy, record_packing_layout, select_layout, Candidate, SelectError,
    SelectionInputs, BITNET_PACKINGS,
};

/// A ternary source value's queryable inputs (the packing site's subject is a ternary value).
fn ternary_inputs(trits: u32) -> SelectionInputs {
    let meta = Meta::exact(Provenance::Root);
    SelectionInputs::from_meta(Repr::Ternary { trits }, &meta)
}

#[test]
fn candidate_set_is_exactly_the_fixed_bitnet_three() {
    // The set is small + fixed (T1.4) — exhaustive evaluation, not autoscheduling.
    assert_eq!(
        BITNET_PACKINGS,
        [PackScheme::I2S, PackScheme::Tl1, PackScheme::Tl2]
    );
    let policy = bitnet_packing_policy();
    assert_eq!(policy.candidates().len(), 3);
    for (c, s) in policy.candidates().iter().zip(BITNET_PACKINGS.iter()) {
        assert_eq!(c, &Candidate::Packing(*s));
    }
}

#[test]
fn exhaustive_cost_model_picks_tl2_and_explains_every_candidate() {
    let policy = bitnet_packing_policy();
    let inputs = ternary_inputs(64);
    let (layout, explain) = select_layout(&policy, &inputs, None).unwrap();

    // TL2 is the cheapest (1.67 b/w < 2.0); the exhaustive cheapest is deterministic.
    assert_eq!(
        layout,
        PhysicalLayout::TritPacked {
            scheme: PackScheme::Tl2
        }
    );
    // Mandatory EXPLAIN: every candidate is costed (the full ranking, RFC-0005 §2.2), and the
    // Always→Cheapest rule fired.
    assert_eq!(explain.costs.len(), 3);
    assert_eq!(explain.matched_rule, Some(0));
    assert!(!explain.overridden);
    assert_eq!(explain.chosen, Candidate::Packing(PackScheme::Tl2));

    // The costs are real storage bits (64 trits × bits/element): I2_S/TL1 = 128, TL2 = 1.67×64.
    let cost_of = |s: PackScheme| {
        explain
            .costs
            .iter()
            .find(|c| c.candidate == Candidate::Packing(s))
            .unwrap()
            .cost
    };
    assert!((cost_of(PackScheme::I2S) - 128.0).abs() < 1e-9);
    assert!((cost_of(PackScheme::Tl1) - 128.0).abs() < 1e-9);
    assert!(cost_of(PackScheme::Tl2) < cost_of(PackScheme::I2S));
}

#[test]
fn selection_is_deterministic() {
    let policy = bitnet_packing_policy();
    let inputs = ternary_inputs(48);
    let a = select_layout(&policy, &inputs, None).unwrap();
    let b = select_layout(&policy, &inputs, None).unwrap();
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1); // identical EXPLAIN — same (policy, inputs) → same trace.
}

#[test]
fn override_forces_a_layout_deterministically() {
    let policy = bitnet_packing_policy();
    let inputs = ternary_inputs(64);

    // Force I2_S (index 0 — the lossless multiply-add default) regardless of cost.
    let (layout, explain) = select_layout(&policy, &inputs, Some(0)).unwrap();
    assert_eq!(
        layout,
        PhysicalLayout::TritPacked {
            scheme: PackScheme::I2S
        }
    );
    assert!(explain.overridden);
    assert_eq!(explain.matched_rule, None); // override bypasses the table.

    // Forcing TL1 (index 1) is likewise honored and stable across calls.
    let force_tl1 = || select_layout(&policy, &inputs, Some(1)).unwrap().0;
    assert_eq!(
        force_tl1(),
        PhysicalLayout::TritPacked {
            scheme: PackScheme::Tl1
        }
    );
    assert_eq!(force_tl1(), force_tl1());
}

#[test]
fn out_of_range_override_is_an_explicit_error() {
    let policy = bitnet_packing_policy();
    let inputs = ternary_inputs(64);
    assert!(select_layout(&policy, &inputs, Some(3)).is_err()); // never a silent fallback (G2).
}

#[test]
fn recorded_layout_is_lossless_on_meta_m_i5() {
    // The chosen layout is recorded on `Meta.physical`; the guarantee/bound are untouched (M-I5).
    let policy = bitnet_packing_policy();
    let src = Repr::Ternary { trits: 32 };
    let meta = Meta::exact(Provenance::Root);
    let (recorded, _explain) = record_packing_layout(&policy, &src, &meta, None).unwrap();
    assert_eq!(
        recorded.physical(),
        Some(PhysicalLayout::TritPacked {
            scheme: PackScheme::Tl2
        })
    );
    // Lossless: the guarantee (and absence of a bound, an Exact value) survive recording.
    assert_eq!(recorded.guarantee(), GuaranteeStrength::Exact);
    assert_eq!(recorded.bound(), None);
}

#[test]
fn element_count_scales_cost_but_not_the_winner() {
    // The winner is invariant to the source size — TL2's lower bits/element wins at any element
    // count (deterministic across sizes).
    let policy = bitnet_packing_policy();
    for trits in [1u32, 16, 64, 4096] {
        let inputs = ternary_inputs(trits);
        assert_eq!(
            select_layout(&policy, &inputs, None).unwrap().0,
            PhysicalLayout::TritPacked {
                scheme: PackScheme::Tl2
            }
        );
    }
}

#[test]
fn a_trit_packed_layout_for_a_non_ternary_source_is_refused() {
    // A5-02 mutant-witness: before the fix, `select_layout`/`record_packing_layout` would happily
    // produce (and record) a `TritPacked` layout for a non-ternary source — a silent latent
    // mis-tag, a layout that contradicts its own representation. A `TritPacked` record only
    // describes how *trits* sit in bytes (RFC-0004 §5; DN-01), so a non-`Ternary` src is now the
    // explicit `NonTernarySource` refusal, never a coercion or a quiet record (G2; never-silent).
    let meta = Meta::exact(Provenance::Root);
    let vsa_src = Repr::Vsa {
        model: "MAP-I".into(),
        dim: 100,
        sparsity: SparsityClass::Dense,
    };
    let policy = bitnet_packing_policy();

    let vsa_inputs = SelectionInputs::from_meta(vsa_src.clone(), &meta);
    assert_eq!(
        select_layout(&policy, &vsa_inputs, None),
        Err(SelectError::NonTernarySource {
            src: vsa_src.clone()
        })
    );

    // The one-call recorder refuses too — no mis-tagged `Meta` ever escapes.
    assert_eq!(
        record_packing_layout(&policy, &vsa_src, &meta, None),
        Err(SelectError::NonTernarySource { src: vsa_src })
    );

    // Binary and Dense sources are equally refused (only `Ternary` admits a trit packing).
    for non_ternary in [
        Repr::Binary { width: 64 },
        Repr::Dense {
            dim: 8,
            dtype: ScalarKind::F32,
        },
    ] {
        let inputs = SelectionInputs::from_meta(non_ternary.clone(), &meta);
        assert_eq!(
            select_layout(&policy, &inputs, None),
            Err(SelectError::NonTernarySource { src: non_ternary })
        );
    }
}
