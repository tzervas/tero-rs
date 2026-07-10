//! Shared Core IR builders for the MEM-4 tests — data-driven fixtures, not bespoke logic in test
//! bodies (CLAUDE.md test-layout rule).

use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Value};

/// A trivial 1-bit constant value (content is irrelevant to RC emission/balance — only term
/// structure matters here).
pub fn val(bit: bool) -> Value {
    Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![bit]),
        Meta::exact(Provenance::Root),
    )
    .expect("1-bit value is well-formed")
}

/// A constant node.
pub fn c() -> Node {
    Node::Const(val(true))
}

/// A variable reference.
pub fn var(name: &str) -> Node {
    Node::Var(name.to_owned())
}

/// `let name = bound in body`.
pub fn let_(name: &str, bound: Node, body: Node) -> Node {
    Node::Let {
        id: name.to_owned(),
        bound: Box::new(bound),
        body: Box::new(body),
    }
}

/// `λ param. body`.
pub fn lam(param: &str, body: Node) -> Node {
    Node::Lam {
        param: param.to_owned(),
        body: Box::new(body),
    }
}

/// `func arg`.
pub fn app(func: Node, arg: Node) -> Node {
    Node::App {
        func: Box::new(func),
        arg: Box::new(arg),
    }
}

/// A primitive application `prim(args…)`.
pub fn op(prim: &str, args: Vec<Node>) -> Node {
    Node::Op {
        prim: prim.to_owned(),
        args,
    }
}
