//! M-390 (R7-Q4): the feedback facade surfaces a content-addressed **prim declaration** at every
//! `Op` site — EXPLAIN over prims (DN-10 §3.2 step 4; G2/SC-3). A primitive is no longer a black
//! box: its `#p` reference, intrinsic guarantee, and arity are inspectable; an unrecognized prim is
//! surfaced as a diagnostic, never silently dropped.

use mycelium_core::{GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_lsp::feedback::analyze;

fn byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

#[test]
fn op_sites_surface_their_content_addressed_prim_declaration() {
    // `bit.not(byte)` — a known kernel prim.
    let program = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte())],
    };
    let fb = analyze(&program);
    assert_eq!(fb.prims.len(), 1, "one prim site per Op node");
    let site = &fb.prims[0];
    assert_eq!(site.name, "bit.not");
    assert!(site.reference.is_some(), "a known prim resolves to its #p");
    assert_eq!(site.intrinsic, Some(GuaranteeStrength::Exact));
    assert_eq!(site.arity, Some(1));
    // It is a clean program — the prim resolved, so no diagnostic was raised for it.
    assert!(fb.summary().is_clean());
    assert_eq!(fb.summary().prims, 1);
}

#[test]
fn an_unknown_prim_is_surfaced_never_silent() {
    let program = Node::Op {
        prim: "bit.nope".into(),
        args: vec![Node::Const(byte())],
    };
    let fb = analyze(&program);
    // The site is still recorded (with no reference), AND an explicit diagnostic is raised.
    assert_eq!(fb.prims.len(), 1);
    assert!(fb.prims[0].reference.is_none());
    assert!(
        fb.diagnostics.iter().any(|d| d.code == "unknown-prim"),
        "an unrecognized prim must surface an `unknown-prim` diagnostic (never silent)"
    );
    assert!(!fb.summary().is_clean());
}
