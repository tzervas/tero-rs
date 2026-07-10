use crate::lint::*;
use mycelium_core::{
    ContentHash, GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, ScalarKind, Trit, Value,
};

fn binary8() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true; 8]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn ternary6() -> Value {
    Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![Trit::Zero; 6]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn declared() -> Value {
    Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.0]),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(mycelium_core::Bound {
                kind: mycelium_core::BoundKind::Probability { delta: 0.1 },
                basis: mycelium_core::BoundBasis::UserDeclared,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

fn codes(diags: &[Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.code).collect()
}

// --- implicit-swap: positive + negative ---

#[test]
fn implicit_swap_fires_on_mixed_paradigms() {
    let node = Node::Op {
        prim: "f".into(),
        args: vec![Node::Const(binary8()), Node::Const(ternary6())],
    };
    let d = lint(&node);
    assert!(codes(&d).contains(&"implicit-swap"));
    assert!(has_errors(&d));
}

#[test]
fn implicit_swap_clean_on_same_paradigm() {
    let node = Node::Op {
        prim: "f".into(),
        args: vec![Node::Const(binary8()), Node::Const(binary8())],
    };
    assert!(!codes(&lint(&node)).contains(&"implicit-swap"));
}

// --- unverified-bound: positive + negative ---

#[test]
fn unverified_bound_fires_on_declared() {
    let d = lint(&Node::Const(declared()));
    assert_eq!(codes(&d), vec!["unverified-bound"]);
    assert_eq!(d[0].severity, Severity::Warning);
}

#[test]
fn unverified_bound_clean_on_exact() {
    assert!(lint(&Node::Const(binary8())).is_empty());
}

// --- placeholder-policy: positive + negative ---

fn swap_with(policy: &str) -> Node {
    Node::Swap {
        src: Box::new(Node::Const(binary8())),
        target: Repr::Ternary { trits: 6 },
        policy: ContentHash::parse(policy).unwrap(),
    }
}

#[test]
fn placeholder_policy_fires_on_stub() {
    assert!(codes(&lint(&swap_with("blake3:00000000"))).contains(&"placeholder-policy"));
    assert!(codes(&lint(&swap_with("policy:todo"))).contains(&"placeholder-policy"));
}

#[test]
fn placeholder_policy_clean_on_real_ref() {
    assert!(!codes(&lint(&swap_with("blake3:Hh3kQ_x-1A"))).contains(&"placeholder-policy"));
}

// --- free-variable: positive + negative ---

#[test]
fn free_variable_fires_when_unbound() {
    let d = lint(&Node::Var("x".into()));
    assert_eq!(codes(&d), vec!["free-variable"]);
}

#[test]
fn free_variable_clean_when_bound() {
    let node = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(binary8())),
        body: Box::new(Node::Var("x".into())),
    };
    assert!(lint(&node).is_empty());
}

#[test]
fn lint_is_deterministic() {
    let node = Node::Op {
        prim: "f".into(),
        args: vec![Node::Const(binary8()), Node::Const(ternary6())],
    };
    assert_eq!(lint(&node), lint(&node));
}

#[test]
fn scoping_respected_in_nested_lets() {
    // `y` is free in the body even though `x` is bound.
    let node = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(binary8())),
        body: Box::new(Node::Var("y".into())),
    };
    assert_eq!(codes(&lint(&node)), vec!["free-variable"]);
}

// --- nodule-header (M-141 source-text companion; DN-06 §6) ---

#[test]
fn nodule_header_clean_on_valid_or_absent_marker() {
    assert!(lint_nodule_header("// nodule: geometry.shapes\nnodule geometry.shapes;\n").is_empty());
    assert!(lint_nodule_header("// nodule\nnodule g;\n").is_empty());
    assert!(lint_nodule_header("nodule g\nfn f() -> Binary{8} = 0b0").is_empty());
}

#[test]
fn nodule_header_fires_on_malformed_named_marker() {
    let d = lint_nodule_header("// nodule: 9bad\n");
    assert_eq!(codes(&d), vec!["nodule-header"]);
    assert_eq!(d[0].severity, Severity::Error);
    assert!(has_errors(&d));
    assert!(!lint_nodule_header("// nodule:\n").is_empty());
}

#[test]
fn structured_header_checks_keys_and_values() {
    // Clean on a valid structured header and on a file with no header.
    assert!(
        lint_structured_header("// nodule: g\n// @license: MIT\n// @updated: 2026-06-16\n")
            .is_empty()
    );
    assert!(lint_structured_header("fn f() -> Binary{8} = 0b0").is_empty());
    // Fires on an unknown key, a bad SPDX license, and a non-ISO date (G2).
    assert_eq!(
        codes(&lint_structured_header("// nodule: g\n// @bogus: x\n")),
        vec!["nodule-header"]
    );
    assert!(has_errors(&lint_structured_header(
        "// nodule: g\n// @license: Nope\n"
    )));
    assert!(has_errors(&lint_structured_header(
        "// nodule: g\n// @since: 2026-13-99\n"
    )));
}

#[test]
fn diagnostic_path_is_the_navigable_breadcrumb() {
    // M-310: the `at` breadcrumb splits into a structured, navigable path.
    let d = Diagnostic {
        code: "x",
        severity: Severity::Warning,
        at: "let a/swap/op f".to_owned(),
        message: String::new(),
    };
    assert_eq!(d.path(), vec!["let a", "swap", "op f"]);
    // An empty breadcrumb (the program root) yields an empty path, not `[""]`.
    let root = Diagnostic {
        at: String::new(),
        ..d.clone()
    };
    assert!(root.path().is_empty());
}
