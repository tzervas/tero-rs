//! M-110 golden corpus — programs paired with their expected evaluated results, exercising every
//! small-step rule (E-Let-Bind/Step, E-Op-Arg/Apply, E-Swap-Arg/Apply), honest metadata threading
//! (guarantee `meet`, `Derived` provenance, `policy_used`), and the never-silent error paths.

use mycelium_core::{
    operation_hash, Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, Node,
    NormKind, Payload, Provenance, Repr, Trit, Value,
};
use mycelium_interp::{EvalError, Interpreter, Step};

fn policy() -> ContentHash {
    ContentHash::parse("blake3:round_trip_safe").unwrap()
}

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

const A: [bool; 8] = [true, false, true, true, false, false, true, false]; // 0b1011_0010
const ONES: [bool; 8] = [true; 8];

fn bits_of(v: &Value) -> Vec<bool> {
    match v.payload() {
        Payload::Bits(b) => b.clone(),
        other => panic!("expected bits, got {other:?}"),
    }
}

fn run(node: &Node) -> Value {
    Interpreter::default().eval(node).expect("evaluates")
}

#[test]
fn const_evaluates_to_itself() {
    let v = byte(A);
    assert_eq!(run(&Node::Const(v.clone())), v);
}

#[test]
fn let_binds_and_substitutes() {
    // let a = 0b1011_0010 in a  ⟶*  0b1011_0010
    let node = Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte(A))),
        body: Box::new(Node::Var("a".into())),
    };
    assert_eq!(run(&node), byte(A));
}

#[test]
fn inner_let_shadows_outer() {
    // let x = A in (let x = ONES in x)  ⟶*  ONES   (shadowing is respected by subst)
    let node = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(byte(A))),
        body: Box::new(Node::Let {
            id: "x".into(),
            bound: Box::new(Node::Const(byte(ONES))),
            body: Box::new(Node::Var("x".into())),
        }),
    };
    assert_eq!(bits_of(&run(&node)), ONES.to_vec());
}

#[test]
fn outer_binding_visible_in_inner_body() {
    // let x = A in (let y = ONES in x)  ⟶*  A   (outer binding survives the inner, distinct binder)
    let node = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(byte(A))),
        body: Box::new(Node::Let {
            id: "y".into(),
            bound: Box::new(Node::Const(byte(ONES))),
            body: Box::new(Node::Var("x".into())),
        }),
    };
    assert_eq!(bits_of(&run(&node)), A.to_vec());
}

#[test]
fn op_id_is_identity_with_derived_provenance() {
    let node = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(byte(A))],
    };
    let out = run(&node);
    assert_eq!(bits_of(&out), A.to_vec());
    assert_eq!(out.meta().guarantee(), GuaranteeStrength::Exact);
    // Honest provenance: Derived{ op = hash(core.id), inputs = [hash(input value)] }.
    match out.meta().provenance() {
        Provenance::Derived { op, inputs } => {
            assert_eq!(op, &operation_hash("core.id"));
            assert_eq!(inputs, &vec![byte(A).content_hash()]);
        }
        other => panic!("expected Derived provenance, got {other:?}"),
    }
}

#[test]
fn bit_not_complements() {
    let node = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let expected: Vec<bool> = A.iter().map(|&b| !b).collect();
    assert_eq!(bits_of(&run(&node)), expected);
}

#[test]
fn bit_xor_combines_two_args() {
    // bit.xor(A, ONES) = !A
    let node = Node::Op {
        prim: "bit.xor".into(),
        args: vec![Node::Const(byte(A)), Node::Const(byte(ONES))],
    };
    let expected: Vec<bool> = A.iter().map(|&b| !b).collect();
    assert_eq!(bits_of(&run(&node)), expected);
}

#[test]
fn let_feeds_an_op() {
    // let a = A in bit.not(a)
    let node = Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte(A))),
        body: Box::new(Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Var("a".into())],
        }),
    };
    let expected: Vec<bool> = A.iter().map(|&b| !b).collect();
    assert_eq!(bits_of(&run(&node)), expected);
}

#[test]
fn trit_neg_flips_signs() {
    // ⟨0,−1,0,0,+1,0⟩  ⟶  ⟨0,+1,0,0,−1,0⟩  (digit-wise sign flip)
    let tern = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![
            Trit::Zero,
            Trit::Neg,
            Trit::Zero,
            Trit::Zero,
            Trit::Pos,
            Trit::Zero,
        ]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Op {
        prim: "trit.neg".into(),
        args: vec![Node::Const(tern)],
    };
    let out = run(&node);
    assert_eq!(
        out.payload(),
        &Payload::Trits(vec![
            Trit::Zero,
            Trit::Pos,
            Trit::Zero,
            Trit::Zero,
            Trit::Neg,
            Trit::Zero,
        ])
    );
}

fn tern(value: i64, m: u32) -> Value {
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(mycelium_core::ternary::int_to_trits(value, m).expect("in range")),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn trit_value_of(v: &Value) -> i64 {
    match v.payload() {
        Payload::Trits(t) => mycelium_core::ternary::trits_to_int(t),
        other => panic!("expected trits, got {other:?}"),
    }
}

#[test]
fn trit_add_evaluates() {
    // 5 + (−3) = 2, in 4 trits.
    let node = Node::Op {
        prim: "trit.add".into(),
        args: vec![Node::Const(tern(5, 4)), Node::Const(tern(-3, 4))],
    };
    assert_eq!(trit_value_of(&run(&node)), 2);
}

#[test]
fn trit_sub_and_mul_evaluate() {
    let sub = Node::Op {
        prim: "trit.sub".into(),
        args: vec![Node::Const(tern(7, 4)), Node::Const(tern(9, 4))],
    };
    assert_eq!(trit_value_of(&run(&sub)), -2);

    let mul = Node::Op {
        prim: "trit.mul".into(),
        args: vec![Node::Const(tern(6, 4)), Node::Const(tern(-6, 4))],
    };
    assert_eq!(trit_value_of(&run(&mul)), -36); // fits in 4 trits (|·| ≤ 40)
}

#[test]
fn trit_arithmetic_overflow_is_explicit() {
    // 30 + 30 = 60 > max(3 trits)=13 → explicit Overflow, never a silent wrap.
    let node = Node::Op {
        prim: "trit.add".into(),
        args: vec![Node::Const(tern(11, 3)), Node::Const(tern(11, 3))],
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::Overflow { .. })
    ));
}

#[test]
fn trit_width_mismatch_is_type_error() {
    let node = Node::Op {
        prim: "trit.add".into(),
        args: vec![Node::Const(tern(1, 3)), Node::Const(tern(1, 4))],
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::PrimType { .. })
    ));
}

#[test]
fn nested_ternary_arithmetic() {
    // let x = 4 in trit.mul(trit.add(x, x), x)  =  (4+4)*4 = 32, in 4 trits.
    let node = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(tern(4, 4))),
        body: Box::new(Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Op {
                    prim: "trit.add".into(),
                    args: vec![Node::Var("x".into()), Node::Var("x".into())],
                },
                Node::Var("x".into()),
            ],
        }),
    };
    assert_eq!(trit_value_of(&run(&node)), 32);
}

#[test]
fn identity_swap_preserves_value_and_records_policy() {
    // swap(A, to: Binary{8}, policy)  ⟶  A, with policy_used recorded (same-repr identity swap).
    let node = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Binary { width: 8 },
        policy: policy(),
    };
    let out = run(&node);
    assert_eq!(bits_of(&out), A.to_vec());
    assert_eq!(out.meta().guarantee(), GuaranteeStrength::Exact);
    assert_eq!(out.meta().policy_used(), Some(&policy()));
}

#[test]
fn nested_program_with_let_op_and_swap() {
    // let a = A in swap(bit.not(a), to: Binary{8}, policy)
    let node = Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte(A))),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("a".into())],
            }),
            target: Repr::Binary { width: 8 },
            policy: policy(),
        }),
    };
    let out = run(&node);
    let expected: Vec<bool> = A.iter().map(|&b| !b).collect();
    assert_eq!(bits_of(&out), expected);
    assert_eq!(out.meta().policy_used(), Some(&policy()));
}

#[test]
fn evaluation_is_deterministic() {
    let node = Node::Op {
        prim: "bit.xor".into(),
        args: vec![Node::Const(byte(A)), Node::Const(byte(ONES))],
    };
    let interp = Interpreter::default();
    assert_eq!(interp.eval(&node), interp.eval(&node));
}

#[test]
fn step_count_matches_expected_reductions() {
    // Op{bit.not, [Const]} : args already values → E-Op-Apply (1 step) → Const → Value.
    let interp = Interpreter::default();
    let n0 = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let n1 = match interp.step(&n0).unwrap() {
        Step::Next(n) => *n,
        Step::Value => panic!("first step should reduce"),
    };
    assert!(matches!(n1, Node::Const(_)));
    assert_eq!(interp.step(&n1).unwrap(), Step::Value);
}

// --- never-silent error paths -----------------------------------------------------------------

#[test]
fn free_variable_is_explicit_error() {
    let err = Interpreter::default().eval(&Node::Var("oops".into()));
    assert_eq!(err, Err(EvalError::FreeVariable("oops".into())));
}

#[test]
fn unknown_prim_is_explicit_error() {
    let node = Node::Op {
        prim: "bit.frobnicate".into(),
        args: vec![Node::Const(byte(A))],
    };
    assert_eq!(
        Interpreter::default().eval(&node),
        Err(EvalError::UnknownPrim("bit.frobnicate".into()))
    );
}

#[test]
fn width_mismatch_is_type_error() {
    let small = Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![true]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Op {
        prim: "bit.and".into(),
        args: vec![Node::Const(byte(A)), Node::Const(small)],
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::PrimType { .. })
    ));
}

#[test]
fn bit_op_on_ternary_is_type_error() {
    let tern = Value::new(
        Repr::Ternary { trits: 1 },
        Payload::Trits(vec![Trit::Pos]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(tern)],
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::PrimType { .. })
    ));
}

#[test]
fn cross_paradigm_swap_is_unsupported_not_silent() {
    // Binary → Ternary is the certified M-120 swap; the identity engine refuses it explicitly.
    let node = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Ternary { trits: 6 },
        policy: policy(),
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::UnsupportedSwap { .. })
    ));
}

#[test]
fn composing_an_approximate_input_is_refused() {
    // An approximate (Declared) value fed to an exact prim → refused, not silently downgraded.
    let approx = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(A.to_vec()),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(Bound {
                kind: BoundKind::Probability { delta: 0.01 },
                basis: BoundBasis::UserDeclared,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let node = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(approx)],
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::ApproxCompositionUnsupported { .. })
    ));
}

// --- M-204: honest approximate composition via the verified-numerics kernel ---------------------

/// A ternary value carrying an `Error` bound at the given strength (the M-I strength↔basis coupling
/// is the caller's to keep consistent — `Meta::new` enforces it).
fn tern_err(value: i64, m: u32, strength: GuaranteeStrength, eps: f64, basis: BoundBasis) -> Value {
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(mycelium_core::ternary::int_to_trits(value, m).expect("in range")),
        Meta::new(
            Provenance::Root,
            strength,
            Some(Bound {
                kind: BoundKind::Error {
                    eps,
                    norm: NormKind::Linf,
                },
                basis,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

fn proven(cite: &str) -> BoundBasis {
    BoundBasis::ProvenThm {
        citation: cite.to_owned(),
    }
}

fn result_error(v: &Value) -> (f64, GuaranteeStrength) {
    match v.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, .. },
            ..
        }) => (*eps, v.meta().guarantee()),
        other => panic!("expected an Error bound, got {other:?}"),
    }
}

#[test]
fn trit_add_composes_approximate_error_bounds() {
    // Two Proven-bounded approximate ternaries → trit.add carries the affine-composed ε (1.0+2.0)
    // and stays Proven (affine composition is itself sound; value is still 5+(-3)=2).
    let node = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(tern_err(
                5,
                4,
                GuaranteeStrength::Proven,
                1.0,
                proven("thm-x"),
            )),
            Node::Const(tern_err(
                -3,
                4,
                GuaranteeStrength::Proven,
                2.0,
                proven("thm-y"),
            )),
        ],
    };
    let out = run(&node);
    assert_eq!(trit_value_of(&out), 2);
    let (eps, strength) = result_error(&out);
    assert!((eps - 3.0).abs() < 1e-12);
    assert_eq!(strength, GuaranteeStrength::Proven);
    assert!(matches!(
        out.meta().bound().unwrap().basis,
        BoundBasis::ProvenThm { .. }
    ));
}

#[test]
fn trit_neg_preserves_error_magnitude() {
    // Negation is affine: ε is unchanged, strength stays Empirical.
    let node = Node::Op {
        prim: "trit.neg".into(),
        args: vec![Node::Const(tern_err(
            4,
            4,
            GuaranteeStrength::Empirical,
            0.5,
            BoundBasis::EmpiricalFit {
                trials: 10_000,
                method: "fit".into(),
            },
        ))],
    };
    let out = run(&node);
    assert_eq!(trit_value_of(&out), -4);
    let (eps, strength) = result_error(&out);
    assert!((eps - 0.5).abs() < 1e-12);
    assert_eq!(strength, GuaranteeStrength::Empirical);
}

#[test]
fn core_id_passes_bound_through_unchanged() {
    // core.id preserves the bound verbatim, including its original citation (identity changes nothing).
    let v = tern_err(
        2,
        4,
        GuaranteeStrength::Proven,
        0.25,
        proven("original-thm"),
    );
    let node = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(v.clone())],
    };
    let out = run(&node);
    assert_eq!(out.meta().bound(), v.meta().bound());
    assert_eq!(out.meta().guarantee(), GuaranteeStrength::Proven);
}

#[test]
fn approximate_composition_meets_strength_down() {
    // Proven ⊕ Declared → Declared (the weakest input governs; VR-5).
    let node = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(tern_err(
                5,
                4,
                GuaranteeStrength::Proven,
                1.0,
                proven("thm"),
            )),
            Node::Const(tern_err(
                -3,
                4,
                GuaranteeStrength::Declared,
                2.0,
                BoundBasis::UserDeclared,
            )),
        ],
    };
    let out = run(&node);
    let (_eps, strength) = result_error(&out);
    assert_eq!(strength, GuaranteeStrength::Declared);
    assert_eq!(out.meta().bound().unwrap().basis, BoundBasis::UserDeclared);
}

#[test]
fn trit_mul_still_refuses_approximate_input() {
    // No multiplicative ε rule yet (needs Dense magnitudes, E2-1) → explicit refusal, never silent.
    let node = Node::Op {
        prim: "trit.mul".into(),
        args: vec![
            Node::Const(tern_err(
                3,
                4,
                GuaranteeStrength::Proven,
                1.0,
                proven("thm"),
            )),
            Node::Const(tern(2, 4)),
        ],
    };
    assert!(matches!(
        Interpreter::default().eval(&node),
        Err(EvalError::ApproxCompositionUnsupported { .. })
    ));
}

#[test]
fn fuel_exhaustion_is_reported() {
    // A program that needs ≥1 reduction, run with zero fuel → explicit FuelExhausted.
    let node = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(byte(A))],
    };
    assert_eq!(
        Interpreter::default().with_fuel(0).eval(&node),
        Err(EvalError::FuelExhausted)
    );
}

#[test]
fn fuel_exhaustion_at_depth_is_reported_not_a_hang(// A4-04: the fuel guard must trip *partway through* a deeply nested reduction, not only at the
    // trivial zero-fuel boundary. A right-nested chain of `core.id`s needs one δ-reduction per
    // layer; budgeting fewer steps than the chain is deep forces FuelExhausted mid-evaluation —
    // an explicit error, never a hang and never a silent truncated result (the never-silent path).
) {
    const DEPTH: usize = 64;
    let mut node = Node::Const(byte(A));
    for _ in 0..DEPTH {
        node = Node::Op {
            prim: "core.id".into(),
            args: vec![node],
        };
    }
    // Ample fuel evaluates the whole nest to the original value.
    let full = Interpreter::default().eval(&node).expect("evaluates fully");
    assert_eq!(bits_of(&full), A.to_vec());
    // A budget smaller than the nesting depth must exhaust strictly *inside* the reduction.
    assert_eq!(
        Interpreter::default()
            .with_fuel(DEPTH as u64 / 2)
            .eval(&node),
        Err(EvalError::FuelExhausted)
    );
}

#[test]
fn malformed_swap_meta_surfaces_as_wf_not_a_panic() {
    // A4-04: `EvalError::Wf` is the never-panic guard for an internally inconsistent constructed
    // result. It is unreachable from the *built-in* prims and the identity swap engine (see the
    // doc comments at the construction sites in prims.rs/swap.rs), so we exercise it through the
    // public `Interpreter::new` extension point with a deliberately broken swap engine that emits a
    // Value whose payload length contradicts its repr. The interpreter must surface this as an
    // explicit `EvalError::Wf`, never a panic and never a silently malformed value (G2).
    use mycelium_core::{ContentHash, Meta, Provenance, Repr, WfError};
    use mycelium_interp::{PrimRegistry, SwapEngine};

    struct MalformedSwap;
    impl SwapEngine for MalformedSwap {
        fn swap(
            &self,
            _src: &Value,
            _target: &Repr,
            _policy: &ContentHash,
        ) -> Result<Value, EvalError> {
            // Claim Binary{8} but hand back a single bit → PayloadReprMismatch on Value::new.
            Value::new(
                Repr::Binary { width: 8 },
                Payload::Bits(vec![true]),
                Meta::exact(Provenance::Root),
            )
            .map_err(EvalError::Wf)
        }
    }

    let node = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Binary { width: 8 },
        policy: policy(),
    };
    let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(MalformedSwap));
    assert_eq!(
        interp.eval(&node),
        Err(EvalError::Wf(WfError::PayloadReprMismatch))
    );
}
