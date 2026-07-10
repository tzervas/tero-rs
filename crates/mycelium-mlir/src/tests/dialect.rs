//! In-crate tests for `dialect.rs` (CLAUDE.md test-layout rule).
//! White-box access via `use crate::dialect::*`; logic file carries no inline `#[cfg(test)]` code.
use crate::dialect::*;
use mycelium_core::{ContentHash, Meta, Node, Payload, Provenance, Repr, Value};

fn byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn program() -> Node {
    Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte())),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Var("a".into())),
            target: Repr::Ternary { trits: 6 },
            policy: ContentHash::parse("blake3:round_trip_safe").unwrap(),
        }),
    }
}

#[test]
fn emits_a_module_with_one_op_per_binding() {
    let m = emit(&program());
    assert!(m.starts_with("module {"));
    assert!(m.contains("func.func @kernel"));
    assert!(m.contains("\"ternary.const\""));
    assert!(m.contains("\"ternary.swap\""));
    assert!(m.contains("func.return"));
    // target + policy attributes are present (no opaque pass).
    assert!(m.contains("target = \"ternary<6>\""));
    assert!(m.contains("policy = \"blake3:round_trip_safe\""));
}

#[test]
fn emission_is_deterministic() {
    assert_eq!(emit(&program()), emit(&program()));
}
