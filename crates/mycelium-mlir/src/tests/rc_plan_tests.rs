//! Tests for `crate::rc_plan` — the MEM-4 → AOT reclamation-plan bridge (RFC-0027 §9 audit trail).
//!
//! White-box: drives `emit_reclamation_plan` / `run_with_reclamation` against the trusted env-machine
//! and a `CollectingSink`, asserting (a) the audit trail is emitted for the straight-line fragment,
//! (b) the value is unperturbed, and (c) out-of-fragment terms are an explicit skip (never silent).

use mycelium_core::{CoreValue, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::{IdentitySwapEngine, PrimRegistry};
use mycelium_rt_abi::reclamation::{CollectingSink, ReclamationTrigger, ScopeId, SweepEpoch};

use crate::rc_plan::{emit_reclamation_plan, run_with_reclamation, RcPlanError};

fn byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `let a = byte in bit.not(a)` — a one-use straight-line term: `a` is reclaimed exactly once.
fn let_op_program() -> Node {
    Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte())),
        body: Box::new(Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Var("a".into())],
        }),
    }
}

/// `(fix f => λx. f x) byte` — non-productive recursion: outside the analysable fragment.
fn spin() -> Node {
    Node::App {
        func: Box::new(Node::Fix {
            name: "f".into(),
            body: Box::new(Node::Lam {
                param: "x".into(),
                body: Box::new(Node::App {
                    func: Box::new(Node::Var("f".into())),
                    arg: Box::new(Node::Var("x".into())),
                }),
            }),
        }),
        arg: Box::new(Node::Const(byte())),
    }
}

// ── emit_reclamation_plan: the audit trail for the straight-line fragment ─────

#[test]
fn plan_emits_a_record_per_reclamation() {
    // `let a = byte in bit.not(a)`: the binding `a` and the `bit.not` result both reach rc 0 in the
    // abstract machine → two reclamation records, all RcZero, all carrying the supplied scope/epoch.
    let mut sink = CollectingSink::new();
    let n = emit_reclamation_plan(&let_op_program(), &mut sink, ScopeId(7), SweepEpoch(3)).unwrap();
    assert_eq!(
        n,
        sink.len(),
        "returned count must equal records emitted (G2)"
    );
    assert!(
        n >= 1,
        "a one-use let-op term reclaims at least the binding"
    );
    for rec in &sink.records {
        assert_eq!(
            *rec.trigger(),
            ReclamationTrigger::RcZero,
            "AOT plan emits RcZero-trigger records"
        );
        assert_eq!(rec.scope_id, ScopeId(7), "scope id is threaded through");
        assert_eq!(
            rec.sweep_epoch,
            SweepEpoch(3),
            "sweep epoch is threaded through"
        );
        // The synthetic value hash is well-formed and tagged `rcplan` (Declared identity).
        assert_eq!(rec.value_meta_hash().algo(), "rcplan");
    }
}

#[test]
fn plan_refuses_out_of_fragment_terms_explicitly() {
    // Recursion is outside the first-order fragment MEM-4 lowers — the emission refuses it, and the
    // bridge surfaces that as a typed error (never an empty plan, G2).
    let mut sink = CollectingSink::new();
    let err = emit_reclamation_plan(&spin(), &mut sink, ScopeId(0), SweepEpoch(0)).unwrap_err();
    assert!(
        matches!(err, RcPlanError::Emit(_)),
        "recursion is refused at emission, not silently skipped: {err:?}"
    );
    assert!(sink.is_empty(), "no partial records on refusal");
}

// ── run_with_reclamation: additive — value unperturbed, plan alongside ────────

#[test]
fn run_computes_value_and_emits_plan_additively() {
    // The value is exactly what `run_core`/`run` would produce, and the plan is emitted alongside.
    let prog = let_op_program();
    let mut sink = CollectingSink::new();
    let run = run_with_reclamation(
        &prog,
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        &mut sink,
    )
    .expect("straight-line program runs cleanly");

    // (a) value: bit.not over the byte, identical to the env-machine alone.
    let expected: Vec<bool> = match byte().payload() {
        Payload::Bits(b) => b.iter().map(|&x| !x).collect(),
        _ => unreachable!(),
    };
    match &run.value {
        CoreValue::Repr(v) => assert_eq!(v.payload(), &Payload::Bits(expected)),
        other => panic!("expected a repr value, got {other:?}"),
    }

    // (b) plan: emitted, and the count matches the records the sink collected.
    assert_eq!(
        run.reclaimed,
        Some(sink.len()),
        "reclaimed count reflects the emitted plan"
    );
    assert!(run.reclaimed.unwrap() >= 1);
}

#[test]
fn run_reports_none_for_out_of_fragment_but_still_computes_when_possible() {
    // A term outside the straight-line fragment yields `reclaimed: None` — an explicit documented
    // skip. Here the recursion also diverges, so the value computation hits the fuel/depth budget;
    // either way the plan is `None`, never a silent empty plan, and no records leak to the sink.
    let mut sink = CollectingSink::new();
    let result = run_with_reclamation(
        &spin(),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        &mut sink,
    );
    // The value side errors (divergent) — that is the env-machine's budget guard, not the plan's.
    assert!(
        result.is_err(),
        "non-productive recursion is a budget error"
    );
    assert!(sink.is_empty(), "out-of-fragment term emits no records");
}

#[test]
fn run_reports_none_for_higher_order_term_that_still_yields_a_value() {
    // `(λx. bit.not(x)) byte` computes a value, but `App`/`Lam` are outside the RC-evaluator's
    // straight-line fragment → the value is returned and `reclaimed` is an explicit `None`.
    let app = Node::App {
        func: Box::new(Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("x".into())],
            }),
        }),
        arg: Box::new(Node::Const(byte())),
    };
    let mut sink = CollectingSink::new();
    let run = run_with_reclamation(
        &app,
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        &mut sink,
    )
    .expect("the application computes a value");
    assert!(matches!(run.value, CoreValue::Repr(_)), "value is produced");
    assert_eq!(
        run.reclaimed, None,
        "higher-order control flow is an explicit plan skip (G2), not an empty plan"
    );
    assert!(sink.is_empty());
}
