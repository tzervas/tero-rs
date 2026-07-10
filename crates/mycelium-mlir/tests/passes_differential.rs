//! M-726 — the **optimization-pass differential** (RFC-0029 §7.2; NFR-7; G2; phase-6).
//!
//! The never-silent correctness bar for the inlining / CSE / DCE passes: for a corpus of programs
//! where each pass actually fires, the **optimized** program must be observably equal to the
//! **unoptimized** program **and** to the **reference interpreter** —
//! `eval(passes(ir)) == eval(ir) == interp(source)` — each pair validated through the **single shared
//! M-210 observational-equivalence checker** (`mycelium_cert::check`, `repr + payload + guarantee`),
//! exactly as the three-way native differential (`threeway_differential.rs`) does. So a passing
//! differential is meaningful, not vacuous: a deliberately divergent transform is caught by the same
//! checker (the sentinel test below).
//!
//! **Guarantee:** `Empirical` — empirical evidence (trials over the corpus) that the passes preserve
//! the interpreter's semantics; never upgraded to `Proven` absent a checked equivalence proof (VR-5).
//!
//! The optimized/unoptimized programs are evaluated by **round-tripping the pass IR back to a
//! `mycelium_core::Node`** and running the **trusted** env-machine (`mycelium_mlir::run`) — the passes
//! never re-implement evaluation (DRY), so this checks them against the trusted base.

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::passes::{optimize, Pass, Program};
use mycelium_numerics::Certificate;

// ─── fixtures (local; small deterministic set, not a statistical sample) ─────────────────────────

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn tern(trits: Vec<Trit>) -> Value {
    let m = trits.len() as u32;
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(trits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

const A: [bool; 8] = [true, false, true, true, false, false, true, false];
const B: [bool; 8] = [false, false, true, false, true, false, true, true];

type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);
fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}

fn interp_eval(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate the corpus")
}

fn aot_eval(node: &Node) -> Value {
    mycelium_mlir::run(node, &PrimRegistry::with_builtins(), &IdentitySwapEngine)
        .expect("AOT must evaluate the corpus")
}

/// The shared M-210 checker verdict for an observationally-equivalent pair (the same call shape the
/// three-way native differential uses).
fn validates(a: &Value, b: &Value) -> bool {
    check(
        a,
        b,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    ) == CheckVerdict::Validated {
        strength: GuaranteeStrength::Exact,
    }
}

// ─── corpus: each program makes a specific pass fire ─────────────────────────────────────────────

/// `(pass-that-must-fire, program)` pairs.
fn corpus() -> Vec<(Pass, Node)> {
    vec![
        // inline — alias fold (a source `let`).
        (
            Pass::Inline,
            Node::Let {
                id: "x".into(),
                bound: Box::new(Node::Op {
                    prim: "bit.not".into(),
                    args: vec![Node::Const(byte(A))],
                }),
                body: Box::new(Node::Op {
                    prim: "bit.not".into(),
                    args: vec![Node::Var("x".into())],
                }),
            },
        ),
        // inline — single-use closure β-reduce.
        (
            Pass::Inline,
            Node::App {
                func: Box::new(Node::Lam {
                    param: "z".into(),
                    body: Box::new(Node::Op {
                        prim: "bit.and".into(),
                        args: vec![Node::Var("z".into()), Node::Const(byte(B))],
                    }),
                }),
                arg: Box::new(Node::Const(byte(A))),
            },
        ),
        // cse — repeated `a xor b`.
        (
            Pass::Cse,
            Node::Let {
                id: "y".into(),
                bound: Box::new(Node::Op {
                    prim: "bit.xor".into(),
                    args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
                }),
                body: Box::new(Node::Op {
                    prim: "bit.or".into(),
                    args: vec![
                        Node::Op {
                            prim: "bit.xor".into(),
                            args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
                        },
                        Node::Var("y".into()),
                    ],
                }),
            },
        ),
        // cse — ternary lane, repeated `trit.neg` combined by `x - x` (stays in range).
        (
            Pass::Cse,
            Node::Let {
                id: "n".into(),
                bound: Box::new(Node::Op {
                    prim: "trit.neg".into(),
                    args: vec![Node::Const(tern(vec![Trit::Pos, Trit::Zero, Trit::Neg]))],
                }),
                body: Box::new(Node::Op {
                    prim: "trit.sub".into(),
                    args: vec![
                        Node::Op {
                            prim: "trit.neg".into(),
                            args: vec![Node::Const(tern(vec![Trit::Pos, Trit::Zero, Trit::Neg]))],
                        },
                        Node::Var("n".into()),
                    ],
                }),
            },
        ),
        // dce — a dead binding.
        (
            Pass::Dce,
            Node::Let {
                id: "dead".into(),
                bound: Box::new(Node::Op {
                    prim: "bit.not".into(),
                    args: vec![Node::Const(byte(A))],
                }),
                body: Box::new(Node::Const(byte(B))),
            },
        ),
    ]
}

/// The core M-726 differential: for every corpus program, `optimized ≡ unoptimized ≡ interp`, each
/// pair validated through the shared M-210 checker; and the designated pass actually fired (so the
/// equivalence is not over a vacuous "optimization").
#[test]
fn optimized_equals_unoptimized_equals_interp_via_shared_checker() {
    for (i, (must_fire, node)) in corpus().iter().enumerate() {
        let (opt, log) = optimize(node);
        let unopt = Program::lower(node);

        assert!(
            log.fired(*must_fire),
            "program #{i}: the {must_fire} pass was expected to fire but did not:\n{}",
            log.explain()
        );

        let opt_val = aot_eval(&opt.to_node());
        let unopt_val = aot_eval(&unopt.to_node());
        let interp_val = interp_eval(node);

        // Observable equality (fast path) …
        assert_eq!(
            observable(&opt_val),
            observable(&interp_val),
            "program #{i}: optimized result diverged from the interpreter (never-silent violation)"
        );
        // … and the shared M-210 checker validates each edge of the triangle.
        assert!(
            validates(&opt_val, &unopt_val),
            "program #{i}: shared checker must validate optimized ↔ unoptimized"
        );
        assert!(
            validates(&unopt_val, &interp_val),
            "program #{i}: shared checker must validate unoptimized ↔ interpreter"
        );
        assert!(
            validates(&opt_val, &interp_val),
            "program #{i}: shared checker must validate optimized ↔ interpreter"
        );
    }
}

/// Sentinel (mutant-witness): the differential is meaningful — a deliberately wrong transform (a
/// different constant) is **rejected** by the same shared checker. So a green differential is not
/// vacuous.
#[test]
fn the_shared_checker_rejects_a_wrong_transform() {
    let (_must, node) = &corpus()[2]; // the CSE case
    let (opt, _) = optimize(node);
    let honest = aot_eval(&opt.to_node());
    let wrong = aot_eval(&Node::Const(byte([true; 8])));

    assert_ne!(
        observable(&honest),
        observable(&wrong),
        "a wrong transform must be observably different"
    );
    assert!(
        !validates(&honest, &wrong),
        "the shared checker must REJECT a divergent (sabotaged) optimization"
    );
}
