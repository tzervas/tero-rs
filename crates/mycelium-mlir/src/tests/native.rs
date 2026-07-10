//! In-crate tests for `dialect/native.rs` (CLAUDE.md test-layout rule; extracted as-touched by
//! M-725 from the former inline `#[cfg(test)] mod tests`). White-box access via
//! `use crate::dialect::native::*`; the logic file carries no inline `#[cfg(test)]` code.
//!
//! Feature-gated: `dialect::native` only compiles under `mlir-dialect`, so this module is gated to
//! match (`super::super` declares it `#[cfg(feature = "mlir-dialect")]`). These tests are pure
//! **emission** checks — they exercise `emit_mlir` (deterministic text, no toolchain) and the
//! refusal boundary; the toolchain-dependent compile/run differential lives in
//! `tests/threeway_differential.rs`.

use crate::dialect::native::*;
use mycelium_core::{
    Alt, CtorSpec, DataRegistry, DeclSpec, FieldSpec, Meta, Node, Payload, Provenance, Repr, Trit,
    Value,
};
use std::collections::BTreeMap;

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

fn not_a_xor_b() -> Node {
    let b = byte([false, false, true, false, true, false, true, true]);
    Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Op {
            prim: "bit.xor".into(),
            args: vec![Node::Const(byte(A)), Node::Const(b)],
        }],
    }
}

/// A `trit.add` over two 4-trit constants whose sum stays in range (no overflow).
fn trit_add_in_range() -> Node {
    Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg, Trit::Pos])),
            Node::Const(tern(vec![Trit::Zero, Trit::Neg, Trit::Pos, Trit::Neg])),
        ],
    }
}

/// A `trit.sub` over two 3-trit constants with a named numeric oracle: `3 - 1 = 2`, in range (no
/// overflow). Balanced-ternary MSB-first: `3 = [0,+,0]` (0·9 + 1·3 + 0·1), `1 = [0,0,+]`, and the
/// difference `2 = [0,+,-]` (0·9 + 1·3 + (-1)·1) all fit 3 trits. Mirrors `trit_add_in_range`'s
/// named-helper style so the emission test exercises a known in-range pair, not an anonymous one.
fn trit_sub_in_range() -> Node {
    Node::Op {
        prim: "trit.sub".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
            Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
        ],
    }
}

#[test]
fn emits_a_real_arith_func_module() {
    let (m, kind, width) = emit_mlir(&not_a_xor_b()).expect("emit");
    assert!(m.starts_with("module {"));
    assert!(m.contains("func.func @main()"));
    assert!(m.contains("func.func private @putchar"));
    // Real arith ops (the lowering, not the textual skeleton):
    assert!(m.contains("arith.xori"), "expected arith.xori in:\n{m}");
    assert!(m.contains("func.call @putchar"));
    assert!(m.contains("func.return"));
    assert_eq!(kind, ResultKind::Binary);
    assert_eq!(width, 8);
}

#[test]
fn emission_is_deterministic() {
    assert_eq!(
        emit_mlir(&not_a_xor_b()).unwrap().0,
        emit_mlir(&not_a_xor_b()).unwrap().0
    );
    // The trit-carry module (M-725) is deterministic too (no nondeterministic SSA naming).
    assert_eq!(
        emit_mlir(&trit_add_in_range()).unwrap().0,
        emit_mlir(&trit_add_in_range()).unwrap().0
    );
}

/// M-725: `trit.add` now lowers through the real dialect path — a ripple-carry over `arith` ops
/// with the shared overflow-sentinel read-back branch (`cf.cond_br`). Asserts the genuine carry
/// arithmetic (`arith.remsi`/`arith.divsi`) and the never-silent overflow branch are emitted.
#[test]
fn trit_add_emits_real_ripple_carry_with_overflow_branch() {
    let (m, kind, width) = emit_mlir(&trit_add_in_range()).expect("emit trit.add");
    assert_eq!(kind, ResultKind::Ternary);
    assert_eq!(width, 4);
    // The balanced-ternary carry step (`x = s + 4`, then `srem 3 − 1` / `sdiv 3 − 1`):
    assert!(m.contains("arith.remsi"), "expected arith.remsi in:\n{m}");
    assert!(m.contains("arith.divsi"), "expected arith.divsi in:\n{m}");
    // The never-silent overflow read-back: a conditional branch on the folded overflow flag.
    assert!(m.contains("cf.cond_br"), "expected cf.cond_br in:\n{m}");
    assert!(m.contains("^ovf:"), "expected ^ovf block in:\n{m}");
    assert!(m.contains("^ok:"), "expected ^ok block in:\n{m}");
    // Both terminating blocks return.
    assert!(m.contains("func.return"));
}

/// `trit.sub` lowers via `add(a, neg(b))` — same ripple + overflow branch (M-725). Exercises the
/// named `3 - 1 = 2` in-range pair (numeric oracle, mirroring `trit_add_in_range`).
#[test]
fn trit_sub_lowers_through_the_dialect_path() {
    let (m, kind, width) = emit_mlir(&trit_sub_in_range()).expect("emit trit.sub");
    assert_eq!(kind, ResultKind::Ternary);
    assert_eq!(width, 3);
    assert!(m.contains("arith.remsi"), "expected ripple-carry in:\n{m}");
    assert!(
        m.contains("cf.cond_br"),
        "expected overflow branch in:\n{m}"
    );
}

/// An overflow-free, purely element-wise program emits NO control flow — the M-601 single-block
/// shape is preserved exactly (the branch is added only when a trit additive op needs it).
#[test]
fn element_wise_program_has_no_overflow_branch() {
    let (m, _, _) = emit_mlir(&not_a_xor_b()).expect("emit");
    assert!(
        !m.contains("cf.cond_br"),
        "element-wise program must stay single-block (no cf.cond_br):\n{m}"
    );
}

/// A `trit.mul` over two 3-trit constants with a named numeric oracle: `2 · 3 = 6`, in range (no
/// overflow). Balanced-ternary MSB-first: `2 = [0,+,-]` (1·3 + (−1)·1), `3 = [0,+,0]` (1·3), and the
/// product `6 = [+,-,0]` (1·9 + (−1)·3 + 0·1) all fit 3 trits. Mirrors the `trit_add_in_range` /
/// `trit_sub_in_range` named-helper style so the emission test exercises a known in-range pair.
fn trit_mul_in_range() -> Node {
    Node::Op {
        prim: "trit.mul".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
        ],
    }
}

/// M-857: `trit.mul` now lowers through the real dialect path — shifted accumulation of `±a` into a
/// 2m-trit buffer (`arith.muli` per digit) plus the shared ripple adder, with the never-silent
/// overflow read-back branch (`cf.cond_br`). Asserts the genuine multiply + carry arithmetic and the
/// overflow branch are emitted (mirrors `trit_add_emits_real_ripple_carry_with_overflow_branch`).
#[test]
fn trit_mul_emits_real_shifted_accumulate_with_overflow_branch() {
    let (m, kind, width) = emit_mlir(&trit_mul_in_range()).expect("emit trit.mul");
    assert_eq!(kind, ResultKind::Ternary);
    assert_eq!(width, 3);
    // The per-digit scaling factor (`±a / 0`) is an integer multiply:
    assert!(m.contains("arith.muli"), "expected arith.muli in:\n{m}");
    // The shared balanced-ternary carry step resolves the accumulation (`x = s + 4`, `remsi`/`divsi`):
    assert!(m.contains("arith.remsi"), "expected arith.remsi in:\n{m}");
    assert!(m.contains("arith.divsi"), "expected arith.divsi in:\n{m}");
    // The never-silent overflow read-back: a conditional branch on the folded overflow flag.
    assert!(m.contains("cf.cond_br"), "expected cf.cond_br in:\n{m}");
    assert!(m.contains("^ovf:"), "expected ^ovf block in:\n{m}");
    assert!(m.contains("^ok:"), "expected ^ok block in:\n{m}");
    assert!(m.contains("func.return"));
}

fn policy() -> mycelium_core::ContentHash {
    mycelium_core::ContentHash::parse("blake3:round_trip_safe").unwrap()
}

#[test]
fn out_of_fragment_nodes_are_explicitly_refused() {
    // M-856 moved the `Swap` boundary: the certified binary↔ternary class now lowers (tested
    // below), so the still-refused case is an **illegal** pair — `Binary{8}` → `Ternary{2}` is
    // illegal (2^7=128 > (3^2-1)/2=4) — refused at compile time by the `Recheck` re-check, never
    // silently emitted.
    let illegal_swap = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Ternary { trits: 2 },
        policy: policy(),
    };
    match emit_mlir(&illegal_swap) {
        Err(DialectError::Unsupported(msg)) => {
            assert!(
                msg.contains("legal pair") || msg.contains("recheck"),
                "the refusal must name the legal-pair re-check; got: {msg}"
            );
        }
        other => panic!("an illegal-pair Swap must be Unsupported, got {other:?}"),
    }
    // A Dense/VSA swap target stays refused too (Dense/VSA are out of the fragment entirely).
    let dense_swap = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Dense {
            dim: 8,
            dtype: mycelium_core::ScalarKind::F32,
        },
        policy: policy(),
    };
    match emit_mlir(&dense_swap) {
        Err(DialectError::Unsupported(_)) => {}
        other => panic!("a Swap to Dense must be Unsupported, got {other:?}"),
    }
    // Everything richer than the fixed-width bit/trit arithmetic + the M-856 data/swap fragment is
    // still refused — here a closure (`Lam`) stays on the direct-LLVM / interp path. The message
    // routes it explicitly (never a silent drop).
    let lam = Node::Lam {
        param: "x".into(),
        body: Box::new(Node::Var("x".into())),
    };
    match emit_mlir(&lam) {
        Err(DialectError::Unsupported(_)) => {}
        other => panic!("a closure (Lam) must be Unsupported (the new boundary), got {other:?}"),
    }
}

#[test]
fn toolchain_resolves_or_skips() {
    // Either the tools resolve (this container) or we get a graceful ToolchainMissing — never a
    // panic, never a silent mismatch.
    match resolve_tools() {
        Ok(t) => {
            assert!(t.mlir_opt.contains("mlir-opt"));
            assert!(t.mlir_translate.contains("mlir-translate"));
            assert!(t.llvm_major >= 1);
        }
        Err(DialectError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected toolchain error: {e}"),
    }
}

// ─── M-856: `Swap` (certified binary↔ternary + identity) ──────────────────────────────────────

/// `(8, 6)` is a legal pair: `2^7 = 128 <= (3^6-1)/2 = 364`. The dialect Swap now lowers a real
/// `enc` transcode (accumulate + balanced-division `arith` ops), never touching the toolchain at
/// emission time.
#[test]
fn legal_binary_to_ternary_swap_emits_a_real_transcode() {
    let swap = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Ternary { trits: 6 },
        policy: policy(),
    };
    let (m, kind, width) = emit_mlir(&swap).expect("legal pair must lower");
    assert_eq!(kind, ResultKind::Ternary);
    assert_eq!(width, 6);
    assert!(m.contains("arith.extui"), "expected bits decode in:\n{m}");
    assert!(
        m.contains("arith.remsi"),
        "expected balanced-division encode in:\n{m}"
    );
    assert!(
        m.contains("arith.divsi"),
        "expected balanced-division encode in:\n{m}"
    );
    // The enc final-quotient honest-net check still threads the overflow read-back branch.
    assert!(
        m.contains("cf.cond_br"),
        "expected the overflow read-back branch in:\n{m}"
    );
}

/// `(4, 3)` is legal (2^3=8 <= (3^3-1)/2=13). The `dec` (Ternary -> Binary) direction lowers via
/// Horner-decode + range-checked bit encode.
#[test]
fn legal_ternary_to_binary_swap_emits_a_real_transcode() {
    let swap = Node::Swap {
        src: Box::new(Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg]))),
        target: Repr::Binary { width: 4 },
        policy: policy(),
    };
    let (m, kind, width) = emit_mlir(&swap).expect("legal pair must lower");
    assert_eq!(kind, ResultKind::Binary);
    assert_eq!(width, 4);
    assert!(m.contains("arith.extsi"), "expected trits decode in:\n{m}");
    assert!(
        m.contains("arith.shrui"),
        "expected bit-encode readback in:\n{m}"
    );
    assert!(
        m.contains("cf.cond_br"),
        "expected the out-of-range read-back branch in:\n{m}"
    );
}

/// A same-`Repr` swap is the identity — no transcode arithmetic, no overflow branch (the lane
/// passes through unchanged, mirroring `crate::swap_codegen`'s identity engine).
#[test]
fn identity_swap_has_no_transcode_or_overflow_branch() {
    let swap = Node::Swap {
        src: Box::new(Node::Const(byte(A))),
        target: Repr::Binary { width: 8 },
        policy: policy(),
    };
    let (m, kind, width) = emit_mlir(&swap).expect("identity swap must lower");
    assert_eq!(kind, ResultKind::Binary);
    assert_eq!(width, 8);
    assert!(
        !m.contains("cf.cond_br"),
        "identity swap must not thread an overflow branch:\n{m}"
    );
    assert!(
        !m.contains("arith.extui"),
        "identity swap must not transcode:\n{m}"
    );
}

// ─── M-856: `Construct`/`Match` (non-recursive data fragment) ─────────────────────────────────

/// A single-constructor, single-field type: `type Box = Box(Binary{8})`.
fn box_registry() -> DataRegistry {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Box".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec {
                fields: vec![FieldSpec::Repr(Repr::Binary { width: 8 })],
            }],
        },
    );
    DataRegistry::build(&specs).expect("Box registry must build")
}

/// A two-constructor, no-field type: `type Color = Red | Blue`.
fn color_registry() -> DataRegistry {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Color".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec { fields: vec![] }, CtorSpec { fields: vec![] }],
        },
    );
    DataRegistry::build(&specs).expect("Color registry must build")
}

/// `Construct(Box, A)` then `Match` extracting the field — the M-373 Increment-1 shape, mirrored
/// through the dialect: the tag is an `arith.constant … : i64`, the field is carried forward as a
/// plain SSA value (no `alloca`/`getelementptr`/`load`/`store`), and `Match` dispatches with
/// `cf.switch`.
#[test]
fn construct_and_match_emit_a_switch_and_no_memory_ops() {
    let reg = box_registry();
    let node = Node::Match {
        scrutinee: Box::new(Node::Construct {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            args: vec![Node::Const(byte(A))],
        }),
        alts: vec![Alt::Ctor {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            binders: vec!["b".to_owned()],
            body: Node::Var("b".to_owned()),
        }],
        default: None,
    };
    let (m, kind, width) = emit_mlir(&node).expect("emit Construct/Match");
    assert_eq!(kind, ResultKind::Binary);
    assert_eq!(width, 8);
    assert!(m.contains("cf.switch"), "expected cf.switch in:\n{m}");
    // A single Ctor arm with a matching binder count is total — the interpreter never actually
    // executes a no-default trap here, but the `Match` still has no default, so the switch's
    // default label traps with `@abort` (a defined trap, never raw UB).
    assert!(
        m.contains("func.func private @abort"),
        "expected @abort declared in:\n{m}"
    );
    assert!(
        m.contains("func.call @abort"),
        "expected the trap call in:\n{m}"
    );
    // No memory ops — the field is a plain SSA value, never alloca/getelementptr/load/store.
    for op in ["alloca", "getelementptr", "memref.load", "memref.store"] {
        assert!(!m.contains(op), "must not use memory ({op}) in:\n{m}");
    }
}

/// A two-arm `Match` (both constructors present) with a `default` needs no `@abort` trap — the
/// declaration is emitted only when actually used (mirrors the overflow-branch minimality).
#[test]
fn match_with_default_never_declares_abort() {
    let col = color_registry();
    let node = Node::Match {
        scrutinee: Box::new(Node::Construct {
            ctor: col.ctor_ref("Color", 1).unwrap(), // Blue
            args: vec![],
        }),
        alts: vec![Alt::Ctor {
            ctor: col.ctor_ref("Color", 0).unwrap(), // Red
            binders: vec![],
            body: Node::Const(byte(A)),
        }],
        default: Some(Box::new(Node::Const(byte([false; 8])))),
    };
    let (m, kind, width) = emit_mlir(&node).expect("emit Match with default");
    assert_eq!(kind, ResultKind::Binary);
    assert_eq!(width, 8);
    assert!(
        !m.contains("@abort"),
        "a Match with a default must never declare/call @abort:\n{m}"
    );
}

/// Emission is deterministic for the data fragment too (no nondeterministic SSA/label naming).
#[test]
fn construct_match_emission_is_deterministic() {
    let reg = box_registry();
    let mk = || Node::Match {
        scrutinee: Box::new(Node::Construct {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            args: vec![Node::Const(byte(A))],
        }),
        alts: vec![Alt::Ctor {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            binders: vec!["b".to_owned()],
            body: Node::Var("b".to_owned()),
        }],
        default: None,
    };
    assert_eq!(emit_mlir(&mk()).unwrap().0, emit_mlir(&mk()).unwrap().0);
}

/// Match on a bare repr-lane scrutinee (the `Lit`-arm branch primitive) stays an explicit refusal
/// — it is tied to the Increment-3 recursion base case (`Fix`/`FixGroup`), out of scope here.
#[test]
fn match_on_a_repr_lane_scrutinee_is_refused() {
    let node = Node::Match {
        scrutinee: Box::new(Node::Const(byte(A))),
        alts: vec![Alt::Lit {
            value: byte(A),
            body: Node::Const(byte(A)),
        }],
        default: Some(Box::new(Node::Const(byte([false; 8])))),
    };
    match emit_mlir(&node) {
        Err(DialectError::Unsupported(_)) => {}
        other => panic!("Match on a repr lane must be Unsupported, got {other:?}"),
    }
}

/// A `trit.add` **inside a `Match` default arm** that is actually taken (M-856). Its overflow flag
/// is local to that arm and must be re-exported through the merge block's own block argument to
/// stay dominance-safe (the module doc comment's design rationale) — this is the compile/run-level
/// confirmation of that design (the in-crate `emit_mlir` tests only assert the *emitted text*).
///
/// **FLAG (M-856b candidate, direct-LLVM only, not fixed here — `llvm.rs` is read-only for this
/// task):** the analogous program on the **direct-LLVM** backend (`crate::llvm::compile_and_run`)
/// fails `llc`'s IR verifier with *"Instruction does not dominate all uses!"* — `crate::llvm`
/// folds a `Match` arm's per-op overflow flags into the **same shared list** as the enclosing
/// scope's flags (see `lower_program`'s `flags: &mut Vec<String>` threaded into `lower_match`), so
/// an overflow flag computed only inside one arm is referenced by the final `fold_or` at a program
/// point that arm's block does not dominate whenever the *other* arm is the one actually taken.
/// Discovered empirically while building this test (see `threeway_differential.rs`'s `data_corpus`
/// case 6, which therefore checks interp<->MLIR-dialect only, not the three-way, for this shape).
#[test]
fn match_default_arm_with_trit_add_compiles_and_runs_through_the_dialect_path() {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Color".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec { fields: vec![] }, CtorSpec { fields: vec![] }],
        },
    );
    let col = DataRegistry::build(&specs).unwrap();
    let node = Node::Match {
        scrutinee: Box::new(Node::Construct {
            ctor: col.ctor_ref("Color", 1).unwrap(),
            args: vec![],
        }),
        alts: vec![Alt::Ctor {
            ctor: col.ctor_ref("Color", 0).unwrap(),
            binders: vec![],
            body: Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Zero])),
        }],
        default: Some(Box::new(Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
            ],
        })),
    };
    match compile_and_run(&node) {
        // 1 + 1 = 2 = [0, +, -] in balanced ternary (3 trits).
        Ok(v) => assert_eq!(
            v.payload(),
            &Payload::Trits(vec![Trit::Zero, Trit::Pos, Trit::Neg])
        ),
        Err(DialectError::ToolchainMissing(_)) => {}
        Err(e) => panic!("dialect path errored: {e}"),
    }
}
