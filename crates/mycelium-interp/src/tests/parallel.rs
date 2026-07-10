//! M-862: the `parallel_eval == sequential_eval` differential over a corpus of pure fragments, plus
//! `is_pure`/`plan_parallel` gate tests (the EXPLAIN-able, top-level-bounded selection). White-box
//! access via `use crate::…::*` (CLAUDE.md test-layout rule).
use crate::parallel::{is_pure, plan_parallel, BatchHead, ParallelPlan};
use crate::{EvalError, Interpreter};
use mycelium_core::{
    Alt, CtorSpec, DataRegistry, DeclSpec, FieldSpec, Meta, Node, Payload, Provenance, Repr, Value,
};
use std::collections::BTreeMap;

fn byte(bits: [bool; 8]) -> Node {
    Node::Const(
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(bits.to_vec()),
            Meta::exact(Provenance::Root),
        )
        .unwrap(),
    )
}

fn op(prim: &str, args: Vec<Node>) -> Node {
    Node::Op {
        prim: prim.to_owned(),
        args,
    }
}

/// A `Binary{width: 4}` constant — deliberately a different width than [`byte`] so a `bit.and`/
/// `bit.or`/`bit.xor` of the two is a deterministic width-mismatch [`EvalError::PrimType`].
fn nibble(bits: [bool; 4]) -> Node {
    Node::Const(
        Value::new(
            Repr::Binary { width: 4 },
            Payload::Bits(bits.to_vec()),
            Meta::exact(Provenance::Root),
        )
        .unwrap(),
    )
}

/// `Fix(f, Var f)` — pure (per [`is_pure`]) and **non-terminating**: every reduction step is a fresh
/// redex, so it consumes fuel forever and never reaches a value. Used to starve a shared fuel pool.
fn spin() -> Node {
    Node::Fix {
        name: "f".into(),
        body: Box::new(Node::Var("f".into())),
    }
}

/// `Pair(Binary{8}, Binary{8})` and `type Nat = Z | S(Nat)` — enough data shape for
/// `Construct`/`Match` fixtures.
fn registry() -> DataRegistry {
    let mut m = BTreeMap::new();
    m.insert(
        "Pair".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec {
                fields: vec![
                    FieldSpec::Repr(Repr::Binary { width: 8 }),
                    FieldSpec::Repr(Repr::Binary { width: 8 }),
                ],
            }],
        },
    );
    m.insert(
        "Nat".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] },
                CtorSpec {
                    fields: vec![FieldSpec::Data("Nat".to_owned())],
                },
            ],
        },
    );
    DataRegistry::build(&m).unwrap()
}

fn z(r: &DataRegistry) -> Node {
    Node::Construct {
        ctor: r.ctor_ref("Nat", 0).unwrap(),
        args: vec![],
    }
}
fn s(r: &DataRegistry, n: Node) -> Node {
    Node::Construct {
        ctor: r.ctor_ref("Nat", 1).unwrap(),
        args: vec![n],
    }
}

/// `drop_ = Fix(f, λn. match n { Z => Z, S(m) => f m })` — reused from `r4_tests`'s fixture shape.
fn drop_(r: &DataRegistry) -> Node {
    Node::Fix {
        name: "f".into(),
        body: Box::new(Node::Lam {
            param: "n".into(),
            body: Box::new(Node::Match {
                scrutinee: Box::new(Node::Var("n".into())),
                alts: vec![
                    Alt::Ctor {
                        ctor: r.ctor_ref("Nat", 0).unwrap(),
                        binders: vec![],
                        body: z(r),
                    },
                    Alt::Ctor {
                        ctor: r.ctor_ref("Nat", 1).unwrap(),
                        binders: vec!["m".into()],
                        body: Node::App {
                            func: Box::new(Node::Var("f".into())),
                            arg: Box::new(Node::Var("m".into())),
                        },
                    },
                ],
                default: None,
            }),
        }),
    }
}

/// A corpus of **pure** Core IR fragments spanning every node family the parallel evaluator
/// special-cases (`Op`/`Construct` fan-out, `App`/`Let`/`Match`/`Fix` ordering).
fn pure_corpus() -> Vec<Node> {
    let r = registry();
    vec![
        // A bare constant.
        byte([true; 8]),
        // A single Op.
        op("bit.not", vec![byte([false; 8])]),
        // Nested, independent Op args — the exact "independent Construct/Op element" shape M-862
        // targets: `and(not(a), or(b, c))`.
        op(
            "bit.and",
            vec![
                op(
                    "bit.not",
                    vec![byte([true, false, true, false, true, false, true, false])],
                ),
                op(
                    "bit.or",
                    vec![
                        byte([false; 8]),
                        byte([false, false, false, false, true, true, true, true]),
                    ],
                ),
            ],
        ),
        // A Construct whose two fields are themselves independent Op subterms.
        Node::Construct {
            ctor: r.ctor_ref("Pair", 0).unwrap(),
            args: vec![
                op("bit.not", vec![byte([true; 8])]),
                op("bit.and", vec![byte([true; 8]), byte([false; 8])]),
            ],
        },
        // A deeper Construct: S(S(Z)).
        s(&r, s(&r, z(&r))),
        // Let over a pure Op.
        Node::Let {
            id: "x".into(),
            bound: Box::new(op("bit.not", vec![byte([false; 8])])),
            body: Box::new(op("bit.and", vec![Node::Var("x".into()), byte([true; 8])])),
        },
        // Beta reduction: (λx. not(x)) applied to a value.
        Node::App {
            func: Box::new(Node::Lam {
                param: "x".into(),
                body: Box::new(op("bit.not", vec![Node::Var("x".into())])),
            }),
            arg: Box::new(byte([false, false, false, false, true, true, true, true])),
        },
        // Curried application (both App positions are independent closed subterms).
        Node::App {
            func: Box::new(Node::App {
                func: Box::new(Node::Lam {
                    param: "x".into(),
                    body: Box::new(Node::Lam {
                        param: "y".into(),
                        body: Box::new(op(
                            "bit.xor",
                            vec![Node::Var("x".into()), Node::Var("y".into())],
                        )),
                    }),
                }),
                arg: Box::new(byte([true, true, true, true, false, false, false, false])),
            }),
            arg: Box::new(byte([false, false, false, false, true, true, true, true])),
        },
        // Match selecting a constructor arm.
        Node::Match {
            scrutinee: Box::new(s(&r, z(&r))),
            alts: vec![
                Alt::Ctor {
                    ctor: r.ctor_ref("Nat", 0).unwrap(),
                    binders: vec![],
                    body: z(&r),
                },
                Alt::Ctor {
                    ctor: r.ctor_ref("Nat", 1).unwrap(),
                    binders: vec!["m".into()],
                    body: Node::Var("m".into()),
                },
            ],
            default: None,
        },
        // Fix-driven structural recursion (drop_ applied to S(S(S(Z)))).
        Node::App {
            func: Box::new(drop_(&r)),
            arg: Box::new(s(&r, s(&r, s(&r, z(&r))))),
        },
    ]
}

#[test]
fn pure_corpus_fragments_are_all_marked_pure() {
    for (i, node) in pure_corpus().iter().enumerate() {
        assert!(is_pure(node), "corpus[{i}] expected pure: {node:?}");
    }
}

/// The M-862 headline claim: `eval_core_parallel == eval_core` over the pure corpus (Empirical,
/// differential-checked — never `Proven`, VR-5).
#[test]
fn parallel_eval_matches_sequential_eval_over_the_pure_corpus() {
    let interp = Interpreter::default();
    for (i, node) in pure_corpus().iter().enumerate() {
        let seq = interp.eval_core(node);
        let par = interp.eval_core_parallel(node);
        assert_eq!(seq, par, "corpus[{i}] diverged: {node:?}");
    }
}

/// Repeating the differential many times catches any nondeterminism a data race would introduce
/// (the scheduler's steal/execution order varies run to run; the spawn-order-indexed *result* must
/// not — RT2).
#[test]
fn parallel_eval_is_deterministic_across_repeated_runs() {
    let interp = Interpreter::default();
    for node in pure_corpus() {
        let first = interp.eval_core_parallel(&node);
        for _ in 0..25 {
            assert_eq!(
                first,
                interp.eval_core_parallel(&node),
                "nondeterministic: {node:?}"
            );
        }
    }
}

#[test]
fn wild_prim_is_never_pure_even_deeply_nested() {
    // A `wild:` op anywhere in the tree makes the WHOLE fragment ineligible (all-or-nothing gate).
    let wild_leaf = op("wild:foreign", vec![byte([true; 8])]);
    assert!(!is_pure(&wild_leaf));

    let nested = op("bit.and", vec![byte([true; 8]), wild_leaf]);
    assert!(
        !is_pure(&nested),
        "a nested wild: op must taint the whole fragment"
    );
}

#[test]
fn an_impure_fragment_falls_back_to_the_sequential_reference_and_still_matches() {
    // `eval_core_parallel` on an impure fragment must equal plain `eval_core` (the fallback path),
    // not merely "run without crashing".
    let interp = Interpreter::default();
    let wild = op("wild:foreign", vec![byte([true; 8])]);
    assert_eq!(interp.eval_core(&wild), interp.eval_core_parallel(&wild));
    assert!(matches!(
        interp.eval_core_parallel(&wild).unwrap_err(),
        EvalError::UnknownPrim(p) if p == "wild:foreign"
    ));
}

#[test]
fn swap_is_conservatively_never_pure() {
    // `SwapEngine` is an opaque `Box<dyn>` — purity can't be verified structurally, so every `Swap`
    // is a parallelism boundary even though the shipped `IdentitySwapEngine` happens to be pure.
    let swap = Node::Swap {
        src: Box::new(byte([true; 8])),
        target: Repr::Binary { width: 8 },
        policy: mycelium_core::operation_hash("policy"),
    };
    assert!(!is_pure(&swap));

    let interp = Interpreter::default();
    assert_eq!(interp.eval_core(&swap), interp.eval_core_parallel(&swap));
}

#[test]
fn fuel_exhaustion_agrees_between_sequential_and_parallel() {
    // Fix(f, f) loops; both paths must refuse identically under a tight fuel budget — never a hang,
    // never a silent divergence between the two evaluators.
    let spin = Node::Fix {
        name: "f".into(),
        body: Box::new(Node::Var("f".into())),
    };
    assert!(is_pure(&spin));
    let interp = Interpreter::default().with_fuel(64);
    assert_eq!(
        interp.eval_core(&spin).unwrap_err(),
        EvalError::FuelExhausted
    );
    assert_eq!(
        interp.eval_core_parallel(&spin).unwrap_err(),
        EvalError::FuelExhausted
    );
}

#[test]
fn eval_parallel_repr_entry_point_agrees_with_eval() {
    let interp = Interpreter::default();
    let node = op("bit.not", vec![byte([false; 8])]);
    assert_eq!(
        interp.eval(&node).unwrap(),
        interp.eval_parallel(&node).unwrap()
    );
}

#[test]
fn eval_parallel_on_a_data_result_is_an_explicit_refusal_like_eval() {
    let r = registry();
    let interp = Interpreter::default();
    assert_eq!(
        interp.eval_parallel(&z(&r)).unwrap_err(),
        EvalError::DataResult
    );
}

// ---- plan_parallel: the reified, EXPLAIN-able, top-level-BOUNDED selection (never silent, G2) ----

#[test]
fn plan_parallelizes_only_a_top_level_multi_arg_op_or_construct() {
    let r = registry();
    // A top-level ≥2-arg Op → a top-level Op batch of that width.
    assert_eq!(
        plan_parallel(&op("bit.and", vec![byte([true; 8]), byte([false; 8])])),
        ParallelPlan::TopLevelBatch {
            head: BatchHead::Op,
            width: 2
        }
    );
    // A top-level ≥2-arg Construct (Pair) → a top-level Construct batch.
    let pair = Node::Construct {
        ctor: r.ctor_ref("Pair", 0).unwrap(),
        args: vec![byte([true; 8]), byte([false; 8])],
    };
    assert_eq!(
        plan_parallel(&pair),
        ParallelPlan::TopLevelBatch {
            head: BatchHead::Construct,
            width: 2
        }
    );
}

#[test]
fn plan_does_not_parallelize_below_the_top_level_or_below_two_args() {
    let r = registry();
    // A single-arg Op is not worth a thread → sequential (no batch).
    assert_eq!(
        plan_parallel(&op("bit.not", vec![byte([true; 8])])),
        ParallelPlan::SequentialNoBatch
    );
    // A zero-arg Construct (Z) → sequential.
    assert_eq!(plan_parallel(&z(&r)), ParallelPlan::SequentialNoBatch);
    // A Let/App/Match/Fix HEAD is never a top-level batch (the parallelism is bounded to the
    // outermost node — nested Op/Construct arg lists are evaluated sequentially within a worker).
    let let_node = Node::Let {
        id: "x".into(),
        bound: Box::new(op("bit.not", vec![byte([false; 8])])),
        body: Box::new(op("bit.and", vec![Node::Var("x".into()), byte([true; 8])])),
    };
    assert_eq!(plan_parallel(&let_node), ParallelPlan::SequentialNoBatch);
    let app = Node::App {
        func: Box::new(Node::Lam {
            param: "x".into(),
            body: Box::new(op("bit.not", vec![Node::Var("x".into())])),
        }),
        arg: Box::new(byte([false; 8])),
    };
    assert_eq!(plan_parallel(&app), ParallelPlan::SequentialNoBatch);
}

#[test]
fn plan_marks_impure_fragments_sequential_impure() {
    // A `wild:` op or a `Swap` (even at the top) is never parallelized — the wholesale-sequential,
    // never-reordered path (G2).
    assert_eq!(
        plan_parallel(&op("wild:foreign", vec![byte([true; 8]), byte([false; 8])])),
        ParallelPlan::SequentialImpure
    );
    let swap = Node::Swap {
        src: Box::new(byte([true; 8])),
        target: Repr::Binary { width: 8 },
        policy: mycelium_core::operation_hash("policy"),
    };
    assert_eq!(plan_parallel(&swap), ParallelPlan::SequentialImpure);
}

#[test]
fn a_deeply_nested_pure_fragment_parallelizes_only_the_top_batch_and_still_matches() {
    // The bound: even though every level here is a pure multi-arg Op, only the OUTERMOST batch is
    // fanned out — the nested Op arg lists are reduced sequentially inside each worker (so total
    // spawned threads are capped at the top width, never O(depth * fan-out)). Correctness is
    // unaffected: the result still equals the sequential reference.
    let deep = op(
        "bit.and",
        vec![
            op(
                "bit.or",
                vec![
                    op("bit.xor", vec![byte([true; 8]), byte([false; 8])]),
                    op("bit.not", vec![byte([false; 8])]),
                ],
            ),
            op(
                "bit.xor",
                vec![
                    op("bit.and", vec![byte([true; 8]), byte([true; 8])]),
                    byte([false, true, false, true, false, true, false, true]),
                ],
            ),
        ],
    );
    assert!(is_pure(&deep));
    // Only the outermost `bit.and`'s two args form the batch — width 2, regardless of depth.
    assert_eq!(
        plan_parallel(&deep),
        ParallelPlan::TopLevelBatch {
            head: BatchHead::Op,
            width: 2
        }
    );
    let interp = Interpreter::default();
    assert_eq!(interp.eval_core(&deep), interp.eval_core_parallel(&deep));
}

#[test]
fn eval_parallel_exercises_the_batch_path_through_the_repr_entry_point() {
    // A ≥2-arg top-level Op reaches the parallel batch path via `eval_parallel` (the repr entry
    // point) and still equals the sequential `eval`.
    let interp = Interpreter::default();
    let node = op("bit.xor", vec![byte([true; 8]), byte([false; 8])]);
    assert!(matches!(
        plan_parallel(&node),
        ParallelPlan::TopLevelBatch { .. }
    ));
    assert_eq!(
        interp.eval(&node).unwrap(),
        interp.eval_parallel(&node).unwrap()
    );
}

// ---- fuel-starvation divergence fix: a non-terminating sibling must never starve an earlier
// ---- arg's fuel into a different (wrong) error than the sequential reference (VR-5/G2) ----

/// The headline regression case. `A = bit.and(bit.not(byte8), nibble4)` is a top-level-*nested*
/// width mismatch: sequentially, reducing the leftmost argument `A` needs only ~1-2 ticks before it
/// deterministically errors `PrimType` (E-Op-Arg finds the innermost `bit.not` redex, then E-Op-Apply
/// on `A` itself hits the width mismatch) — the sequential reference's strict left-to-right
/// short-circuit means the second top-level argument `spin` (pure, non-terminating) is **never**
/// touched. Dispatching both as concurrent jobs against one shared fuel pool (the pre-fix behaviour)
/// lets `spin` race ahead and drain the pool, starving `A`'s job into `FuelExhausted` instead —
/// schedule-dependent and wrong. The fix (discard any batch with an `Err` and defer wholesale to
/// `eval_core`) must make `eval_core_parallel` agree with `eval_core` exactly, every time.
#[test]
fn fuel_starved_sibling_never_diverges_from_the_sequential_reference() {
    let a = op(
        "bit.and",
        vec![
            op("bit.not", vec![byte([true; 8])]),
            nibble([true, false, true, false]),
        ],
    );
    let node = op("bit.and", vec![a, spin()]);
    assert!(is_pure(&node));
    assert_eq!(
        plan_parallel(&node),
        ParallelPlan::TopLevelBatch {
            head: BatchHead::Op,
            width: 2
        }
    );

    // Tight fuel: comfortably enough for the sequential reference's couple of ticks, nowhere near
    // enough for `spin` to ever terminate — it always errors, so every parallel attempt must be
    // discarded and re-run through the sequential reference (never a partially-trusted result).
    let interp = Interpreter::default().with_fuel(5);
    let expected = interp.eval_core(&node);
    assert_eq!(
        expected,
        Err(EvalError::PrimType {
            prim: "bit.and".to_owned(),
            why: "width mismatch: 8 vs 4".to_owned(),
        }),
        "sequential reference must deterministically hit the width-mismatch PrimType error on `A` \
         without ever reaching `spin`"
    );

    // Repeated runs exercise whatever interleaving the scheduler actually produces; the parallel
    // result must equal `expected` — never `FuelExhausted` — every single time (schedule-independent).
    for _ in 0..50 {
        assert_eq!(
            interp.eval_core_parallel(&node),
            expected,
            "eval_core_parallel diverged from eval_core under a fuel-starving sibling"
        );
    }
}

/// The all-success fast path (both batch jobs return `Ok`) must still parallelize and still match
/// the sequential reference exactly — the fix only changes behaviour on the any-error path.
#[test]
fn all_success_batch_still_parallelizes_and_matches_sequential() {
    let node = op(
        "bit.and",
        vec![
            op("bit.not", vec![byte([true; 8])]),
            op(
                "bit.or",
                vec![
                    byte([false; 8]),
                    byte([true, false, true, false, true, false, true, false]),
                ],
            ),
        ],
    );
    assert!(is_pure(&node));
    assert_eq!(
        plan_parallel(&node),
        ParallelPlan::TopLevelBatch {
            head: BatchHead::Op,
            width: 2
        }
    );
    let interp = Interpreter::default();
    assert_eq!(interp.eval_core(&node), interp.eval_core_parallel(&node));
}
