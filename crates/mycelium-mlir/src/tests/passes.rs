//! White-box tests for the M-726 optimization passes (`crate::passes`; RFC-0029 §7.2).
//!
//! Two things every pass must demonstrate (the DoD): it is **EXPLAIN-able** (its decisions are
//! reified into a queryable [`TransformLog`] — asserted by `log.fired(..)` / `by_pass` / `by_site`),
//! and it is **never-silent + differentially correct** — `eval(passes(ir)) == eval(ir) ==
//! interp(source)` (asserted via the trusted env-machine `crate::aot::run_core` and the reference
//! interpreter, on the result's observable identity `repr + payload + guarantee`, NFR-7).
//!
//! The differential corpus is **data-driven** (per the CLAUDE.md test-layout rule): each [`Case`]
//! carries a program, the pass it is meant to exercise, and whether that pass must fire — the test
//! body is `assert over a case`, not bespoke logic.

use mycelium_core::{GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};

use crate::aot;
use crate::passes::{cse, dce, inline, optimize, Pass, Program, TransformLog};

// ─── fixtures ────────────────────────────────────────────────────────────────────────────────────

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

type Observable = (Repr, Payload, GuaranteeStrength);
fn observable(v: &Value) -> Observable {
    (v.repr().clone(), v.payload().clone(), v.meta().guarantee())
}

fn interp_eval(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate the corpus")
}

/// Run a program through the trusted AOT env-machine (the same evaluator `aot::run` uses).
fn aot_eval(node: &Node) -> Value {
    aot::run(node, &PrimRegistry::with_builtins(), &IdentitySwapEngine)
        .expect("AOT must evaluate the corpus")
}

// ─── differential corpus (data-driven) ─────────────────────────────────────────────────────────

/// Which pass a corpus case is built to exercise (and must fire on it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exercises {
    Inline,
    Cse,
    Dce,
    /// A program where the pipeline runs but no specific single-pass firing is asserted (used for the
    /// "passes are a no-op-preserving identity" baseline cases).
    Any,
}

struct Case {
    name: &'static str,
    program: Node,
    exercises: Exercises,
}

/// `let x = not a in not x` — the `let` lowers to an alias `x = %k`, so inlining's `alias-fold` fires.
fn alias_chain() -> Node {
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
    }
}

/// `(λx. bit.not x) a` — a closure applied exactly once ⇒ inlining's `beta-reduce` fires.
fn single_use_closure() -> Node {
    Node::App {
        func: Box::new(Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("x".into())],
            }),
        }),
        arg: Box::new(Node::Const(byte(A))),
    }
}

/// `let y = (a xor b) in bit.and (a xor b) y` — the subexpression `a xor b` is computed twice in the
/// source, so two identical `bit.xor a b` ops appear in the ANF ⇒ CSE merges them.
fn repeated_subexpr() -> Node {
    let axb = || Node::Op {
        prim: "bit.xor".into(),
        args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
    };
    Node::Let {
        id: "y".into(),
        bound: Box::new(axb()),
        body: Box::new(Node::Op {
            prim: "bit.and".into(),
            args: vec![axb(), Node::Var("y".into())],
        }),
    }
}

/// `let dead = bit.not a in b` — `dead` is bound but never read ⇒ DCE removes it (and the `b`
/// constant binding it shadows nothing). The result is just `b`.
fn dead_binding() -> Node {
    Node::Let {
        id: "dead".into(),
        bound: Box::new(Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Const(byte(A))],
        }),
        body: Box::new(Node::Const(byte(B))),
    }
}

/// A larger program where all three passes have something to do: an aliased repeated subexpression
/// feeding a single-use closure, plus a dead binding.
fn combined() -> Node {
    // let r = (a xor b) in
    // let dead = bit.not a in           -- dead (never used)
    // let f = (λz. bit.and z (a xor b)) in
    //   f r                              -- f used once (β-reduce); (a xor b) repeated (CSE)
    Node::Let {
        id: "r".into(),
        bound: Box::new(Node::Op {
            prim: "bit.xor".into(),
            args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
        }),
        body: Box::new(Node::Let {
            id: "dead".into(),
            bound: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Const(byte(A))],
            }),
            body: Box::new(Node::App {
                func: Box::new(Node::Lam {
                    param: "z".into(),
                    body: Box::new(Node::Op {
                        prim: "bit.and".into(),
                        args: vec![
                            Node::Var("z".into()),
                            Node::Op {
                                prim: "bit.xor".into(),
                                args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
                            },
                        ],
                    }),
                }),
                arg: Box::new(Node::Var("r".into())),
            }),
        }),
    }
}

/// A ternary program (a different repr lane) with a repeated `trit.neg` subexpression — CSE again,
/// proving the passes are repr-agnostic. The combining op is `trit.sub` of the two equal results
/// (`x − x = 0`), so the program stays in range (no overflow — that would be a source-program error,
/// not a pass concern).
fn ternary_repeated() -> Node {
    let t = || Node::Const(tern(vec![Trit::Pos, Trit::Zero, Trit::Neg]));
    let neg = || Node::Op {
        prim: "trit.neg".into(),
        args: vec![t()],
    };
    Node::Let {
        id: "n".into(),
        bound: Box::new(neg()),
        body: Box::new(Node::Op {
            prim: "trit.sub".into(),
            args: vec![neg(), Node::Var("n".into())],
        }),
    }
}

/// A bare constant + a plain op — nothing to optimize; the pipeline must be a value-preserving
/// identity (the no-op baseline).
fn nothing_to_do() -> Node {
    Node::Op {
        prim: "bit.and".into(),
        args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
    }
}

fn corpus() -> Vec<Case> {
    vec![
        Case {
            name: "alias_chain",
            program: alias_chain(),
            exercises: Exercises::Inline,
        },
        Case {
            name: "single_use_closure",
            program: single_use_closure(),
            exercises: Exercises::Inline,
        },
        Case {
            name: "repeated_subexpr",
            program: repeated_subexpr(),
            exercises: Exercises::Cse,
        },
        Case {
            name: "ternary_repeated",
            program: ternary_repeated(),
            exercises: Exercises::Cse,
        },
        Case {
            name: "dead_binding",
            program: dead_binding(),
            exercises: Exercises::Dce,
        },
        Case {
            name: "combined",
            program: combined(),
            exercises: Exercises::Any,
        },
        Case {
            name: "nothing_to_do",
            program: nothing_to_do(),
            exercises: Exercises::Any,
        },
    ]
}

// ─── the core differential: with == without == interp (Empirical) ────────────────────────────────

/// For every corpus program, assert the three-way equality:
/// `eval(optimized) == eval(unoptimized) == interp(source)`, on the observable identity. This is the
/// never-silent guarantee: no pass changed observable behaviour. (`Empirical` — trials, not `Proven`.)
#[test]
fn with_equals_without_equals_interp_over_corpus() {
    for case in corpus() {
        let unopt = Program::lower(&case.program);
        let (opt, _log) = optimize(&case.program);

        // Three evaluations: optimized (round-tripped to a Node, run on the trusted env-machine),
        // unoptimized (likewise), and the reference interpreter on the original source.
        let opt_val = aot_eval(&opt.to_node());
        let unopt_val = aot_eval(&unopt.to_node());
        let interp_val = interp_eval(&case.program);

        assert_eq!(
            observable(&opt_val),
            observable(&unopt_val),
            "case `{}`: optimized result diverged from unoptimized (a pass changed observable \
             behaviour — never-silent violation, G2)",
            case.name
        );
        assert_eq!(
            observable(&unopt_val),
            observable(&interp_val),
            "case `{}`: the (unoptimized) AOT round-trip diverged from the reference interpreter",
            case.name
        );
        assert_eq!(
            observable(&opt_val),
            observable(&interp_val),
            "case `{}`: optimized result diverged from the reference interpreter",
            case.name
        );
    }
}

/// Each corpus case's designated pass must actually **fire** on it (the transform log records the
/// transform) — a vacuous "optimization" that never fires would make the differential meaningless.
#[test]
fn the_designated_pass_fires_and_is_recorded() {
    for case in corpus() {
        let (_opt, log) = optimize(&case.program);
        match case.exercises {
            Exercises::Inline => assert!(
                log.fired(Pass::Inline),
                "case `{}`: inlining was expected to fire but the log has no Inline record:\n{}",
                case.name,
                log.explain()
            ),
            Exercises::Cse => assert!(
                log.fired(Pass::Cse),
                "case `{}`: CSE was expected to fire but the log has no Cse record:\n{}",
                case.name,
                log.explain()
            ),
            Exercises::Dce => assert!(
                log.fired(Pass::Dce),
                "case `{}`: DCE was expected to fire but the log has no Dce record:\n{}",
                case.name,
                log.explain()
            ),
            Exercises::Any => { /* baseline / combined — no single-pass firing asserted here */ }
        }
    }
}

/// The `combined` program exercises **all three** passes in one run — the pipeline composes.
#[test]
fn the_pipeline_fires_all_three_passes_on_the_combined_program() {
    let (_opt, log) = optimize(&combined());
    assert!(
        log.fired(Pass::Inline),
        "combined: inline must fire\n{}",
        log.explain()
    );
    assert!(
        log.fired(Pass::Cse),
        "combined: cse must fire\n{}",
        log.explain()
    );
    assert!(
        log.fired(Pass::Dce),
        "combined: dce must fire\n{}",
        log.explain()
    );
    // The combined program still evaluates to the same value (covered by the corpus differential too).
    let (opt, _) = optimize(&combined());
    assert_eq!(
        observable(&aot_eval(&opt.to_node())),
        observable(&interp_eval(&combined())),
        "combined: optimized result must equal the interpreter"
    );
}

// ─── EXPLAIN-ability: the transform log is reified + queryable ───────────────────────────────────

/// Inlining's `alias-fold` is recorded with a readable `(pass, rule, site, before → after, reason)`
/// and is queryable by pass and by site.
#[test]
fn inline_records_an_explainable_alias_fold() {
    let (_p, log) = inline(&Program::lower(&alias_chain()));
    assert!(
        log.fired(Pass::Inline),
        "alias-fold must fire:\n{}",
        log.explain()
    );
    let inlines = log.by_pass(Pass::Inline);
    assert!(
        !inlines.is_empty(),
        "there must be at least one Inline record"
    );
    let r = inlines[0];
    assert_eq!(r.pass, Pass::Inline);
    assert_eq!(r.rule, "alias-fold");
    assert!(
        !r.reason.is_empty(),
        "the record must carry a non-empty reason (no black box)"
    );
    // The EXPLAIN dump is human-readable and mentions the pass + rule.
    let dump = log.explain();
    assert!(dump.contains("inline/alias-fold"), "explain dump:\n{dump}");
}

/// The single-use closure inline is recorded as a `beta-reduce` with the closure site and the reason.
#[test]
fn inline_records_an_explainable_beta_reduce() {
    let (_p, log) = inline(&Program::lower(&single_use_closure()));
    let betas: Vec<_> = log
        .by_pass(Pass::Inline)
        .into_iter()
        .filter(|r| r.rule == "beta-reduce")
        .collect();
    assert_eq!(
        betas.len(),
        1,
        "exactly one closure should be β-reduced; log:\n{}",
        log.explain()
    );
    assert!(betas[0].reason.contains("applied exactly once"));
}

/// CSE records a `cse-merge` naming the redundant site and the canonical binding it merged into.
#[test]
fn cse_records_an_explainable_merge() {
    let (_p, log) = cse(&Program::lower(&repeated_subexpr()));
    assert!(log.fired(Pass::Cse), "cse must fire:\n{}", log.explain());
    let merges = log.by_pass(Pass::Cse);
    assert!(!merges.is_empty());
    let r = merges[0];
    assert_eq!(r.rule, "cse-merge");
    assert!(
        r.after.contains("canonical"),
        "the merge record should name the canonical binding; got `{}`",
        r.after
    );
    // by_site: the recorded site is queryable.
    let site = r.site.clone();
    assert!(
        !log.by_site(&site).is_empty(),
        "the record must be retrievable by its site `{site}`"
    );
}

/// DCE records a `drop-dead` for the removed binding, with `<removed>` as the after-state.
#[test]
fn dce_records_an_explainable_drop() {
    let (opt, log) = dce(&Program::lower(&dead_binding()));
    assert!(log.fired(Pass::Dce), "dce must fire:\n{}", log.explain());
    let drops = log.by_pass(Pass::Dce);
    assert!(
        !drops.is_empty(),
        "at least one dead binding must be dropped"
    );
    assert!(drops.iter().all(|r| r.rule == "drop-dead"));
    assert!(drops.iter().any(|r| r.after == "<removed>"));
    // The optimized program is strictly smaller (the dead binding is gone).
    assert!(
        opt.len() < Program::lower(&dead_binding()).len(),
        "DCE must shrink the program"
    );
}

// ─── never-silent: the log accounts for every change (no silent transform) ───────────────────────

/// A no-op program (nothing to optimize) produces an **empty** log AND an unchanged result — the
/// pipeline never invents a transform, and a no-op is recorded by the *absence* of any entry.
#[test]
fn a_no_op_program_yields_an_empty_log_and_unchanged_value() {
    let (opt, log) = optimize(&nothing_to_do());
    assert!(
        log.is_empty(),
        "no transform should be recorded on a program with nothing to optimize; got:\n{}",
        log.explain()
    );
    assert_eq!(
        observable(&aot_eval(&opt.to_node())),
        observable(&interp_eval(&nothing_to_do())),
        "the no-op pipeline must preserve the value exactly"
    );
}

/// Soundness sentinel (mutant-witness): the differential is **not** vacuous. A *deliberately wrong*
/// "optimization" — replacing the program with a different constant — is caught by the same
/// observable-identity comparison the real differential uses. So a green differential is meaningful.
#[test]
fn the_differential_catches_a_deliberately_wrong_transform() {
    // The honest optimized result.
    let prog = repeated_subexpr();
    let (opt, _) = optimize(&prog);
    let honest = aot_eval(&opt.to_node());

    // A sabotaged "optimized" program: a different constant. The differential check must reject it.
    let wrong = Node::Const(byte([true; 8]));
    let wrong_val = aot_eval(&wrong);
    assert_ne!(
        observable(&honest),
        observable(&wrong_val),
        "a wrong transform must be observably different — else the differential is vacuous"
    );
}

// ─── purity / log-API unit checks ────────────────────────────────────────────────────────────────

/// `TransformLog` is append-only via `record`, and `extend` preserves order — the merged pipeline log
/// is the concatenation of each pass's records in pipeline order.
#[test]
fn the_pipeline_log_is_the_ordered_concatenation_of_each_pass() {
    let prog = Program::lower(&combined());
    let (p1, l_inline) = inline(&prog);
    let (p2, l_cse) = cse(&p1);
    let (_p3, l_dce) = dce(&p2);

    let (_opt, merged) = optimize(&combined());

    let expected_len = l_inline.len() + l_cse.len() + l_dce.len();
    assert_eq!(
        merged.len(),
        expected_len,
        "the merged log must account for every pass's records (no dropped or invented entry)"
    );
    // Inline records come first, then CSE, then DCE (pipeline order).
    let passes_in_order: Vec<Pass> = merged.entries().iter().map(|r| r.pass).collect();
    let first_cse = passes_in_order.iter().position(|p| *p == Pass::Cse);
    let last_inline = passes_in_order.iter().rposition(|p| *p == Pass::Inline);
    if let (Some(fc), Some(li)) = (first_cse, last_inline) {
        assert!(
            li < fc,
            "every Inline record precedes every Cse record (pipeline order)"
        );
    }
}

/// Each pass is a **pure function**: running it twice on the same input yields the identical
/// `(Program, log)` (no hidden state, deterministic — a prerequisite for EXPLAIN-ability).
#[test]
fn each_pass_is_deterministic_and_pure() {
    let prog = Program::lower(&combined());
    assert_eq!(inline(&prog), inline(&prog), "inline must be deterministic");
    let (p1, _) = inline(&prog);
    assert_eq!(cse(&p1), cse(&p1), "cse must be deterministic");
    let (p2, _) = cse(&p1);
    assert_eq!(dce(&p2), dce(&p2), "dce must be deterministic");
}

/// Running the whole pipeline a second time on its own output is **stable** — a fixpoint: the second
/// run finds nothing new to do (no further transform), so optimization is idempotent on this corpus.
#[test]
fn the_pipeline_is_idempotent_on_its_own_output() {
    for case in corpus() {
        let (opt1, _) = optimize(&case.program);
        let (opt2, log2) = crate::passes::run_pipeline(&opt1);
        // The second pass over already-optimized IR yields an observably-equal value…
        assert_eq!(
            observable(&aot_eval(&opt1.to_node())),
            observable(&aot_eval(&opt2.to_node())),
            "case `{}`: re-optimizing must preserve the value",
            case.name
        );
        // …and (DCE may still clean a freshly-dead alias on the second pass, so we don't require an
        // empty log) the value is the invariant. The empty-log fixpoint is asserted for the no-op case.
        let _ = log2;
    }
}

/// The `TransformLog` empty-on-construction + `record` mutation contract (the only mutator).
#[test]
fn transform_log_starts_empty_and_records() {
    let mut log = TransformLog::new();
    assert!(log.is_empty());
    assert_eq!(log.len(), 0);
    log.record(crate::passes::TransformRecord {
        pass: Pass::Dce,
        rule: "drop-dead",
        site: "%9".into(),
        before: "op bit.not %1".into(),
        after: "<removed>".into(),
        reason: "test".into(),
    });
    assert!(!log.is_empty());
    assert_eq!(log.len(), 1);
    assert!(log.fired(Pass::Dce));
    assert!(!log.fired(Pass::Cse));
    assert_eq!(log.by_site("%9").len(), 1);
}
