//! In-crate unit tests for the Dense operational surface (extracted from `lib.rs` per the
//! test-layout rule — logic files carry no test code; white-box access via `use super::*`).

use super::*;

#[test]
fn op_rel_eps_constants_match_their_cited_formulas() {
    // A5-07: pin the disclosed per-op ε to the exact formulas of their ProvenThm citation, so
    // the literals cannot silently drift from `F32_OP_CITATION`/`BF16_OP_CITATION` (VR-5).
    // F32: single-rounding unit roundoff u = β^(1−p)/2 = 2⁻²⁴ (IEEE binary32, p = 24).
    assert_eq!(F32_OP_REL_EPS, 2f64.powi(-24));
    // BF16: two-rounding composition u₁ + u₂ + u₁u₂ ≤ 2⁻⁸ + 2⁻²³ (the disclosed slack bound).
    assert_eq!(BF16_OP_REL_EPS, 2f64.powi(-8) + 2f64.powi(-23));
    // M-891: the binary64 accumulator's unit roundoff u = 2⁻⁵³ (the measurement-bound unit).
    assert_eq!(F64_ACC_U, 2f64.powi(-53));
}

#[test]
fn unsupported_dtypes_are_explicit() {
    assert_eq!(
        DenseSpace::new(4, ScalarKind::F64),
        Err(DenseError::UnsupportedDtype {
            dtype: ScalarKind::F64
        })
    );
    assert_eq!(
        DenseSpace::new(4, ScalarKind::F16),
        Err(DenseError::UnsupportedDtype {
            dtype: ScalarKind::F16
        })
    );
}

#[test]
fn construction_checks_the_grid() {
    let s = DenseSpace::new(2, ScalarKind::F32).unwrap();
    assert!(s.value(vec![1.5, -0.625]).is_ok());
    // 0.1 is not exactly an f32.
    assert_eq!(
        s.value(vec![1.0, 0.1]),
        Err(DenseError::NotOnGrid { index: 1 })
    );
    assert_eq!(
        s.value(vec![f64::NAN, 0.0]),
        Err(DenseError::NonFinite { index: 0 })
    );
    let b = DenseSpace::new(2, ScalarKind::Bf16).unwrap();
    // 1.5 is on the bf16 grid; 1.501953125 (1.5 + 2^-9) is f32-exact but off the bf16 grid.
    assert!(b.value(vec![1.5, -2.0]).is_ok());
    assert_eq!(
        b.value(vec![1.5, 1.501_953_125]),
        Err(DenseError::NotOnGrid { index: 1 })
    );
}

#[test]
fn neg_is_exact() {
    let s = DenseSpace::new(3, ScalarKind::F32).unwrap();
    let a = s.value(vec![1.5, -0.625, 0.0]).unwrap();
    let n = s.neg_value(&a).unwrap();
    assert_eq!(n.meta().guarantee(), GuaranteeStrength::Exact);
    assert_eq!(n.payload(), &Payload::Scalars(vec![-1.5, 0.625, 0.0]));
    assert_eq!(
        DenseSpace::op_guarantee(DenseOp::Neg),
        GuaranteeStrength::Exact
    );
}

#[test]
fn add_carries_the_proven_rounding_bound() {
    let s = DenseSpace::new(2, ScalarKind::F32).unwrap();
    let a = s.value(vec![1.5, 2.5]).unwrap();
    let b = s.value(vec![0.25, -1.0]).unwrap();
    let y = s.add_values(&a, &b).unwrap();
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { .. },
        }) => {
            assert_eq!(*eps, F32_OP_REL_EPS);
            assert_eq!(*norm, NormKind::Rel);
        }
        other => panic!("expected a ProvenThm Error bound, got {other:?}"),
    }
    match y.meta().provenance() {
        Provenance::Derived { op, inputs } => {
            assert_eq!(op, &operation_hash("dense.add"));
            assert_eq!(inputs, &vec![a.content_hash(), b.content_hash()]);
        }
        other => panic!("expected Derived, got {other:?}"),
    }
}

#[test]
fn mismatches_and_approximate_sources_are_typed_errors() {
    let s = DenseSpace::new(2, ScalarKind::F32).unwrap();
    let a = s.value(vec![1.0, 2.0]).unwrap();
    let wrong_dim = DenseSpace::new(3, ScalarKind::F32)
        .unwrap()
        .value(vec![1.0, 2.0, 3.0])
        .unwrap();
    assert_eq!(
        s.add_values(&a, &wrong_dim),
        Err(DenseError::DimMismatch {
            expected: 2,
            got: 3
        })
    );
    let wrong_dtype = DenseSpace::new(2, ScalarKind::Bf16)
        .unwrap()
        .value(vec![1.0, 2.0])
        .unwrap();
    assert_eq!(
        s.add_values(&a, &wrong_dtype),
        Err(DenseError::DtypeMismatch {
            expected: ScalarKind::F32
        })
    );
    // An approximate source is refused (no composition rule yet) — built via the M-204-style
    // derived bound to simulate one.
    let approx = Value::new(
        s.repr(),
        Payload::Scalars(vec![1.0, 2.0]),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(Bound {
                kind: BoundKind::Error {
                    eps: 0.1,
                    norm: NormKind::Rel,
                },
                basis: BoundBasis::UserDeclared,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        s.add_values(&a, &approx),
        Err(DenseError::ApproximateSource)
    );
}

#[test]
fn overflow_and_subnormal_results_are_explicit() {
    let s = DenseSpace::new(1, ScalarKind::F32).unwrap();
    let max = s.value(vec![f64::from(f32::MAX)]).unwrap();
    assert_eq!(
        s.add_values(&max, &max),
        Err(DenseError::Overflow { index: 0 })
    );
    // 1.5·2⁻¹²⁶ − 1.25·2⁻¹²⁶ = 0.25·2⁻¹²⁶: subnormal, refused.
    let a = s.value(vec![1.5 * DENSE_MIN_NORMAL]).unwrap();
    let b = s.value(vec![1.25 * DENSE_MIN_NORMAL]).unwrap();
    assert_eq!(
        s.sub_values(&a, &b),
        Err(DenseError::SubnormalUnsupported { index: 0 })
    );
}

#[test]
fn scale_checks_the_factor() {
    let s = DenseSpace::new(2, ScalarKind::Bf16).unwrap();
    let a = s.value(vec![1.5, -2.0]).unwrap();
    assert_eq!(s.scale_value(&a, 0.1), Err(DenseError::ScalarOffGrid));
    let y = s.scale_value(&a, 2.0).unwrap();
    assert_eq!(y.payload(), &Payload::Scalars(vec![3.0, -4.0]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
}

#[test]
fn similarity_is_a_measurement_helper() {
    let s = DenseSpace::new(2, ScalarKind::F32).unwrap();
    let a = s.value(vec![1.0, 0.0]).unwrap();
    let b = s.value(vec![0.0, 1.0]).unwrap();
    assert!((s.similarity(&a, &b).unwrap()).abs() < 1e-12);
    assert!((s.similarity(&a, &a).unwrap() - 1.0).abs() < 1e-12);
    let z = s.value(vec![0.0, 0.0]).unwrap();
    assert_eq!(s.similarity(&a, &z).unwrap(), 0.0);
    assert_eq!(s.dot(&a, &b).unwrap(), 0.0);
}

// ── M-891 (`enb` Gap C): the value-returning measurement pair `dot_value`/`similarity_value` ────

/// A `Dense{n, F32}` value from on-grid elements (test fixture).
fn dense_f32(xs: Vec<f64>) -> Value {
    let n = u32::try_from(xs.len()).expect("test dims are small");
    DenseSpace::new(n, ScalarKind::F32)
        .expect("F32 is a supported dtype")
        .value(xs)
        .expect("fixture elements are finite and on-grid")
}

#[test]
fn measurement_ops_are_proven_in_op_guarantee() {
    // The per-op tag: a recursive sum rounds, so Dot/Similarity can never be Exact (VR-5).
    assert_eq!(
        DenseSpace::op_guarantee(DenseOp::Dot),
        GuaranteeStrength::Proven
    );
    assert_eq!(
        DenseSpace::op_guarantee(DenseOp::Similarity),
        GuaranteeStrength::Proven
    );
}

#[test]
fn dot_value_carries_the_proven_accumulation_bound() {
    let s = DenseSpace::new(3, ScalarKind::F32).unwrap();
    let a = dense_f32(vec![1.5, 2.0, -0.5]);
    let b = dense_f32(vec![2.0, 0.25, 4.0]);
    let y = s.dot_value(&a, &b).unwrap();
    // 3.0 + 0.5 − 2.0 = 1.5: every product and partial sum exact in f64.
    assert_eq!(
        y.repr(),
        &Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64
        },
        "the measurement result form is Dense{{1, F64}} — the f64 exactly as computed"
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![1.5]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    // The bound is the disclosed absolute accumulation bound (Linf), NOT op_rel_eps (which is a
    // per-element dtype-rounding relative ε — it never enters: inputs exact, accumulation f64).
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { citation },
        }) => {
            assert_eq!(*eps, s.dot_abs_eps(3.0 + 0.5 + 2.0));
            assert_eq!(*norm, NormKind::Linf);
            assert!(
                citation.contains("Higham") && citation.contains("2^−53"),
                "the ProvenThm citation must carry the theorem + the u = 2⁻⁵³ unit: {citation}"
            );
        }
        other => panic!("expected the ProvenThm Linf accumulation bound, got {other:?}"),
    }
    match y.meta().provenance() {
        Provenance::Derived { op, inputs } => {
            assert_eq!(op, &operation_hash("dense.dot"));
            assert_eq!(inputs, &vec![a.content_hash(), b.content_hash()]);
        }
        other => panic!("expected Derived, got {other:?}"),
    }
    // The value-returning twin delivers the SAME f64 the bare measurement helper does.
    assert_eq!(y.payload(), &Payload::Scalars(vec![s.dot(&a, &b).unwrap()]));
}

#[test]
fn similarity_value_carries_the_input_independent_bound() {
    let s = DenseSpace::new(2, ScalarKind::F32).unwrap();
    let a = dense_f32(vec![1.0, 0.0]);
    let b = dense_f32(vec![0.0, 1.0]);
    // Orthogonal: products are 0 each — the cosine is exactly 0.
    let y = s.similarity_value(&a, &b).unwrap();
    assert_eq!(
        y.repr(),
        &Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64
        }
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![0.0]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { .. },
        }) => {
            assert_eq!(*eps, s.similarity_abs_eps());
            assert_eq!(*norm, NormKind::Linf);
        }
        other => panic!("expected the ProvenThm Linf similarity bound, got {other:?}"),
    }
    // Self-similarity is 1 within the disclosed bound.
    let Payload::Scalars(sim) = s.similarity_value(&a, &a).unwrap().payload().clone() else {
        panic!("similarity must return scalars");
    };
    assert!((sim[0] - 1.0).abs() <= s.similarity_abs_eps());
    // The zero-norm convention: exactly 0 (documented — an operand norm is 0 iff the operand is
    // the zero vector), still carried with the Proven bound (0 error ≤ ε).
    let z = dense_f32(vec![0.0, 0.0]);
    let zc = s.similarity_value(&a, &z).unwrap();
    assert_eq!(zc.payload(), &Payload::Scalars(vec![0.0]));
    assert_eq!(zc.meta().guarantee(), GuaranteeStrength::Proven);
    // Twin-consistency with the bare helper.
    assert_eq!(
        zc.payload(),
        &Payload::Scalars(vec![s.similarity(&a, &z).unwrap()])
    );
}

/// **Property test (the disclosed bound, incl. the cancellation case a relative claim would
/// fail):** over cases whose *true* real-arithmetic dot is known analytically, the computed
/// payload differs from the truth by at most the ε the value's own bound discloses.
#[test]
fn dot_value_respects_its_own_disclosed_bound() {
    // (xs, ys, exact dot in real arithmetic). All elements are on the F32 grid.
    // The last case cancels catastrophically: fl(2⁶⁰ + 1) = 2⁶⁰ in f64, so the computed dot is
    // 0 against a true 1 — a per-element relative bound is FALSE here; the absolute
    // accumulation bound must (and does) cover it.
    let two30 = f64::from(2f32.powi(30));
    let cases: [(&[f64], &[f64], f64); 4] = [
        (&[1.5, 2.0, -0.5], &[2.0, 0.25, 4.0], 1.5),
        (&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0], 20.0),
        (&[0.0, 0.0], &[1.0, -1.0], 0.0),
        (&[two30, 1.0, -two30], &[two30, 1.0, two30], 1.0),
    ];
    for (xs, ys, exact) in cases {
        let n = u32::try_from(xs.len()).unwrap();
        let s = DenseSpace::new(n, ScalarKind::F32).unwrap();
        let a = s.value(xs.to_vec()).unwrap();
        let b = s.value(ys.to_vec()).unwrap();
        let y = s.dot_value(&a, &b).unwrap();
        let Payload::Scalars(out) = y.payload() else {
            panic!("dot_value must return scalars")
        };
        let Some(Bound {
            kind: BoundKind::Error { eps, .. },
            ..
        }) = y.meta().bound()
        else {
            panic!("dot_value must carry its Error bound")
        };
        assert!(
            (out[0] - exact).abs() <= *eps,
            "|{} − {exact}| exceeds the value's own disclosed ε = {eps}",
            out[0]
        );
    }
}

#[test]
fn measurement_ops_share_the_elementwise_operand_contract() {
    let s = DenseSpace::new(2, ScalarKind::F32).unwrap();
    let a = dense_f32(vec![1.0, 2.0]);
    let wrong_dim = dense_f32(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        s.dot_value(&a, &wrong_dim),
        Err(DenseError::DimMismatch {
            expected: 2,
            got: 3
        })
    );
    let wrong_dtype = DenseSpace::new(2, ScalarKind::Bf16)
        .unwrap()
        .value(vec![1.0, 2.0])
        .unwrap();
    assert_eq!(
        s.similarity_value(&a, &wrong_dtype),
        Err(DenseError::DtypeMismatch {
            expected: ScalarKind::F32
        })
    );
    // An approximate source is refused — same honest scope as the elementwise ops.
    let approx = s.add_values(&a, &dense_f32(vec![0.5, 0.5])).unwrap();
    assert_eq!(s.dot_value(&a, &approx), Err(DenseError::ApproximateSource));
    assert_eq!(
        s.similarity_value(&approx, &a),
        Err(DenseError::ApproximateSource)
    );
}

#[test]
fn measurement_ops_work_over_bf16_spaces() {
    // The exact-product argument holds a fortiori for BF16 (8-bit significands).
    let s = DenseSpace::new(2, ScalarKind::Bf16).unwrap();
    let a = s.value(vec![1.5, -2.0]).unwrap();
    let b = s.value(vec![2.0, 0.5]).unwrap();
    let y = s.dot_value(&a, &b).unwrap();
    assert_eq!(y.payload(), &Payload::Scalars(vec![2.0])); // 3.0 − 1.0, exact
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
}
