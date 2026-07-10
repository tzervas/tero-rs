//! In-crate white-box tests for `trampoline.rs` — the M-850 direct-LLVM heap-trampoline (non-tail
//! `Fix` / `FixGroup`). White-box access via `use crate::trampoline::*` (the `Member`/`Cont`/analysis
//! internals are `pub(crate)`/private). These are **pure analysis/emission** checks — no toolchain;
//! the compiled interp≡native differential lives in `tests/recursion_trampoline_differential.rs`.
//!
//! Guarantee tag: **Declared** at the unit level (shape assertions over the analyzer/destructurer);
//! the *differential* leg is what raises the lowering to **Empirical** (see the integration test).
use crate::trampoline::*;

use mycelium_core::lower::{lower_to_anf, Anf};
use mycelium_core::{Alt, Meta, Node, Payload, Provenance, Repr, Value};

// ─── fixtures ───────────────────────────────────────────────────────────────────────────────────

fn byte_n(n: u8) -> Value {
    let bits: Vec<bool> = (0..8).map(|i| (n >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .expect("8-bit value")
}

/// Lower a `Fix { name, body }`'s body to the nested ANF the trampoline destructures. The
/// destructurer takes the Fix *body* (a `λparam. Match …` node), lowered.
fn fix_body_anf(body: Node) -> Anf {
    lower_to_anf(&body)
}

/// A canonical tail-or-base single `Fix` body: `λn. Match n { Lit 0 → 0xAA ; default → self(0) }`.
fn tail_fix_body() -> Node {
    Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Lit {
                value: byte_n(0),
                body: Node::Const(byte_n(0xAA)),
            }],
            default: Some(Box::new(Node::App {
                func: Box::new(Node::Var("self".into())),
                arg: Box::new(Node::Const(byte_n(0))),
            })),
        }),
    }
}

/// A non-tail single `Fix` body: `λn. Match n { Lit 0 → 0xAA ; default → bit.not(self(0)) }`.
fn non_tail_fix_body() -> Node {
    Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Lit {
                value: byte_n(0),
                body: Node::Const(byte_n(0xAA)),
            }],
            default: Some(Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::App {
                    func: Box::new(Node::Var("self".into())),
                    arg: Box::new(Node::Const(byte_n(0))),
                }],
            })),
        }),
    }
}

// ─── destructuring ──────────────────────────────────────────────────────────────────────────────

#[test]
fn destructure_fix_accepts_canonical_lam_match() {
    // Mutant-witness: a destructurer that dropped the param/arm extraction would yield the wrong
    // member count or mis-name the member.
    let members = destructure_fix("self", &fix_body_anf(tail_fix_body()))
        .expect("canonical λ.Match Fix must destructure");
    assert_eq!(members.len(), 1, "a single Fix is a one-member group");
}

#[test]
fn destructure_fix_refuses_non_lam_body() {
    // A Fix body that is a bare Const (not `λparam. Match …`) is refused (G2). Mutant-witness:
    // skipping the Lam check would let a non-canonical body through to fragile IR.
    let anf = fix_body_anf(Node::Const(byte_n(0xAA)));
    assert!(
        destructure_fix("self", &anf).is_err(),
        "a non-λ Fix body must be refused"
    );
}

#[test]
fn destructure_fix_refuses_ctor_arm_on_param() {
    // A Ctor arm on the recursion param is outside the Binary{8} Lit-arm fragment (G2).
    let body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Ctor {
                ctor: mycelium_core::CtorRef::new(
                    mycelium_core::ContentHash::parse("blake3:pair").expect("hash"),
                    0,
                ),
                binders: vec!["a".into(), "b".into()],
                body: Node::Const(byte_n(0)),
            }],
            default: None,
        }),
    };
    assert!(
        destructure_fix("self", &fix_body_anf(body)).is_err(),
        "a Ctor arm on the recursion param must be refused"
    );
}

#[test]
fn destructure_fixgroup_accepts_two_members() {
    // Two canonical members → a two-member group. Mutant-witness: a destructurer that returned only
    // the first member would break sibling resolution.
    let defs = vec![
        ("f".to_string(), fix_body_anf(tail_fix_body())),
        ("g".to_string(), fix_body_anf(non_tail_fix_body())),
    ];
    let members = destructure_fixgroup(&defs).expect("two canonical members destructure");
    assert_eq!(members.len(), 2);
}

// ─── tail-vs-trampoline classification ──────────────────────────────────────────────────────────

#[test]
fn pure_tail_single_fix_is_classified_as_tail() {
    // The tail/base-only single Fix is the fast iterative-loop fragment — `is_pure_tail_single_fix`
    // must return true so `llvm.rs` keeps the byte-for-byte tail loop. Mutant-witness: returning
    // false here would needlessly route the tail fragment through the heavier trampoline.
    let members = destructure_fix("self", &fix_body_anf(tail_fix_body())).expect("destructure");
    assert!(
        is_pure_tail_single_fix(&members).expect("classify"),
        "a tail/base-only single Fix is the tail-loop fragment"
    );
}

#[test]
fn non_tail_single_fix_is_classified_as_trampoline() {
    // A non-tail (pending-op) arm forces the trampoline — `is_pure_tail_single_fix` must be false.
    // Mutant-witness: returning true would route a non-tail program to the tail loop, which cannot
    // express the pending continuation (a wrong/fragile lowering).
    let members = destructure_fix("self", &fix_body_anf(non_tail_fix_body())).expect("destructure");
    assert!(
        !is_pure_tail_single_fix(&members).expect("classify"),
        "a non-tail single Fix must route to the trampoline"
    );
}

#[test]
fn multi_member_group_is_never_the_tail_fragment() {
    // A FixGroup (≥2 members) is never the single-Fix tail-loop fragment, regardless of arm shapes.
    let defs = vec![
        ("f".to_string(), fix_body_anf(tail_fix_body())),
        ("g".to_string(), fix_body_anf(tail_fix_body())),
    ];
    let members = destructure_fixgroup(&defs).expect("destructure group");
    assert!(
        !is_pure_tail_single_fix(&members).expect("classify"),
        "a multi-member group is never the tail-loop fragment"
    );
}

// ─── runtime emission ───────────────────────────────────────────────────────────────────────────

#[test]
fn trampoline_runtime_is_self_contained_and_safe() {
    // The frame-stack runtime declares its alloc/free seams and traps OOM with a defined @abort
    // (never raw UB) — and carries no `unsafe` token (the submodule-confinement invariant; DN-21).
    let rt = trampoline_runtime();
    assert!(rt.contains("@myc_tramp_alloc"), "declares the alloc seam");
    assert!(rt.contains("@myc_tramp_free"), "declares the free seam");
    assert!(
        rt.contains("call void @abort()"),
        "OOM takes a defined-trap, never silent UB (G2)"
    );
    assert!(
        !rt.contains("unsafe"),
        "the emitted runtime is a safe @malloc/@free structure"
    );
}

#[test]
fn trampoline_runtime_emission_is_deterministic() {
    assert_eq!(trampoline_runtime(), trampoline_runtime());
}
