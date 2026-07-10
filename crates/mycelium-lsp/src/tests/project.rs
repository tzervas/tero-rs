use crate::project::llm_canonical;
use mycelium_core::{
    Bound, BoundBasis, BoundKind, CtorRef, GuaranteeStrength, Meta, Node, NormKind, Payload,
    Provenance, Repr, ScalarKind, Trit, Value,
};

fn byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn trits() -> Value {
    Value::new(
        Repr::Ternary { trits: 4 },
        Payload::Trits(vec![Trit::Zero, Trit::Zero, Trit::Pos, Trit::Neg]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

// P2 witness: an approximate value carrying a Declared guarantee + a bound.
fn declared_dense() -> Value {
    let bound = Bound {
        kind: BoundKind::Error {
            eps: 0.5,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    };
    Value::new(
        Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.0, 2.0]),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(bound),
            None,
            None,
            None,
        )
        .expect("well-formed meta"),
    )
    .unwrap()
}

/// Totality (RFC-0021 §4.2): every one of the 11 node kinds renders without panicking and yields
/// a non-empty s-expression. The exhaustive `match` makes this true by construction; this test is
/// the witness that the feasibility assessment (`research/11` T11.4) holds for the full grammar.
#[test]
fn total_over_every_node_kind() {
    let ctor = CtorRef::new(
        mycelium_core::ContentHash::parse("blake3:00ctor00").unwrap(),
        0,
    );
    let nodes: Vec<Node> = vec![
        Node::Const(byte()),
        Node::Var("x".into()),
        Node::Let {
            id: "a".into(),
            bound: Box::new(Node::Const(byte())),
            body: Box::new(Node::Var("a".into())),
        },
        Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Const(byte())],
        },
        Node::Swap {
            src: Box::new(Node::Const(byte())),
            target: Repr::Ternary { trits: 6 },
            policy: mycelium_core::ContentHash::parse("blake3:po1icy00").unwrap(),
        },
        Node::Construct {
            ctor: ctor.clone(),
            args: vec![Node::Const(byte())],
        },
        Node::Match {
            scrutinee: Box::new(Node::Const(byte())),
            alts: vec![],
            default: Some(Box::new(Node::Const(byte()))),
        },
        Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Var("x".into())),
        },
        Node::App {
            func: Box::new(Node::Var("f".into())),
            arg: Box::new(Node::Const(byte())),
        },
        Node::Fix {
            name: "f".into(),
            body: Box::new(Node::Var("f".into())),
        },
        Node::FixGroup {
            defs: vec![
                ("f".into(), Box::new(Node::Var("g".into()))),
                ("g".into(), Box::new(Node::Var("f".into()))),
            ],
            body: Box::new(Node::Var("f".into())),
        },
    ];
    // One rule per node kind; the closed v0 grammar is exactly these 11.
    assert_eq!(nodes.len(), 11, "the v0 L1 grammar is 11 node kinds");
    for n in &nodes {
        let s = llm_canonical(n).expect("small fixture fits the default arena ceiling");
        assert!(!s.is_empty(), "every node kind renders non-empty: {n:?}");
    }
}

/// P3 (RFC-0021 §4.3): a `Swap` is rendered explicitly and never elided — its marker and policy
/// always appear.
#[test]
fn swap_is_never_elided() {
    let prog = Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte())),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Var("a".into())),
            target: Repr::Ternary { trits: 6 },
            policy: mycelium_core::ContentHash::parse("blake3:po1icy00").unwrap(),
        }),
    };
    let s = llm_canonical(&prog).expect("small fixture fits the default arena ceiling");
    assert!(
        s.contains("(swap!"),
        "P3: the Swap node must be rendered: {s}"
    );
    assert!(s.contains(":to Ternary{6}"), "the target survives: {s}");
    assert!(s.contains(":policy"), "the policy reference survives: {s}");
}

/// P2 (RFC-0021 §4.3): the guarantee tag is part of every rendered constant, and an approximate
/// value surfaces its bound — the honesty tag can never be silently dropped.
#[test]
fn guarantee_tags_survive() {
    assert!(
        llm_canonical(&Node::Const(byte()))
            .expect("small fixture fits the default arena ceiling")
            .contains("@Exact"),
        "an Exact value renders its tag"
    );
    let s = llm_canonical(&Node::Const(declared_dense()))
        .expect("small fixture fits the default arena ceiling");
    assert!(
        s.contains("@Declared"),
        "a Declared value renders its tag: {s}"
    );
    assert!(
        s.contains(":bound"),
        "an approximate value surfaces its bound: {s}"
    );
}

/// Determinism (RFC-0021 §3.1): same node in, same surface out.
#[test]
fn deterministic() {
    let prog = Node::Op {
        prim: "trit.add".into(),
        args: vec![Node::Const(trits()), Node::Const(trits())],
    };
    assert_eq!(
        llm_canonical(&prog).expect("small fixture fits the default arena ceiling"),
        llm_canonical(&prog).expect("small fixture fits the default arena ceiling")
    );
}

/// A large `Const` payload is summarized by length, not inlined element-wise — bounded output on
/// arbitrarily wide values, and never silently dropped (it states its shape).
#[test]
fn large_payloads_are_summarized() {
    let wide = Value::new(
        Repr::Binary { width: 256 },
        Payload::Bits(vec![false; 256]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let s =
        llm_canonical(&Node::Const(wide)).expect("small fixture fits the default arena ceiling");
    assert!(s.contains("0b<256 bits>"), "wide binary is summarized: {s}");
    // A small value is still inlined verbatim (the byte fixture is 8 bits).
    assert!(llm_canonical(&Node::Const(byte()))
        .expect("small fixture fits the default arena ceiling")
        .contains("0b10110010"));
}

/// RFC-0041 §4.2/§9 (W7 process-arena coverage): a large synthetic `Node` — one deep enough that its
/// estimated byte cost exceeds a deliberately tiny arena ceiling — refuses `OutOfBudget`
/// never-silently, rather than proceeding to render unbounded. Uses
/// [`crate::project::llm_canonical_with_arena`] (`pub(crate)`) to inject the tiny ceiling; production
/// callers go through [`llm_canonical`], which supplies the crate's declared default.
#[test]
fn large_synthetic_input_trips_out_of_budget() {
    use crate::project::llm_canonical_with_arena;
    use mycelium_workstack::{BudgetError, BudgetKind, ProcessArena};

    // A 5,000-deep right-nested `Let` chain: comfortably beyond a 1-byte ceiling's tolerance for the
    // per-node estimate, and not pathologically deep for the guarded stack (this crate's guard-hole
    // census already exercises 20,000).
    let mut prog = Node::Const(byte());
    for i in 0..5_000 {
        prog = Node::Let {
            id: format!("x{i}"),
            bound: Box::new(Node::Const(byte())),
            body: Box::new(prog),
        };
    }

    let tiny_arena = ProcessArena::new(1);
    match llm_canonical_with_arena(&prog, &tiny_arena) {
        Err(BudgetError::OutOfBudget {
            kind: BudgetKind::Bytes,
            limit,
            ..
        }) => assert_eq!(limit, 1, "the refusal reports the configured ceiling"),
        other => panic!("expected an explicit OutOfBudget refusal, got {other:?}"),
    }
}

/// The normal-input twin of the test above: the same 5,000-deep chain, rendered against the crate's
/// real declared default ceiling ([`llm_canonical`]), passes unchanged — the arena wiring never
/// perturbs ordinary-sized input.
#[test]
fn normal_sized_input_passes_unchanged() {
    let mut prog = Node::Const(byte());
    for i in 0..5_000 {
        prog = Node::Let {
            id: format!("x{i}"),
            bound: Box::new(Node::Const(byte())),
            body: Box::new(prog),
        };
    }
    let s = llm_canonical(&prog).expect("well within the default 256 MiB arena ceiling");
    assert_eq!(
        s.matches("(let [x").count(),
        5_000,
        "every binder is still rendered"
    );
}
