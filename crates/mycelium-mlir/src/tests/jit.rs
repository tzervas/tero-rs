//! In-crate white-box tests for [`crate::jit`] (M-340; CLAUDE.md test-layout rule). Extracted from
//! the logic file (M-797 lazy retrofit, as-touched by M-727/M-729). `use crate::jit::*` gives
//! white-box access to the private `emit_kernel_fn` emitter.

use crate::jit::*;
use crate::llvm::AotError;
use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Trit, Value};

fn binary(bits: Vec<bool>) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

#[test]
fn jit_kernel_emits_a_function_writing_to_out() {
    let prog = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(binary(vec![true, false]))],
    };
    let (ir, _, width) = emit_kernel_fn(&prog).unwrap();
    assert!(ir.contains("define i32 @myc_kernel(ptr %out)"));
    assert!(ir.contains("store i8")); // writes results into the out buffer
    assert!(ir.contains("ret i32 0")); // ok status (no overflow path for a bit op)
    assert_eq!(width, 2);
}

#[test]
fn jit_bit_not_matches_interpreter() {
    // Mutant-witness: a wrong store offset / fn signature would read back a different payload.
    let prog = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(binary(vec![true, false, true, true]))],
    };
    match jit_run(&prog) {
        Ok(v) => {
            assert_eq!(v.payload(), &Payload::Bits(vec![false, true, false, false]));
            assert_eq!(v.repr(), &Repr::Binary { width: 4 });
        }
        Err(AotError::ToolchainMissing(_)) => { /* environment skip */ }
        Err(e) => panic!("unexpected JIT error: {e}"),
    }
}

#[test]
fn jit_trit_neg_matches_interpreter() {
    let prog = Node::Op {
        prim: "trit.neg".into(),
        args: vec![Node::Const(
            Value::new(
                Repr::Ternary { trits: 3 },
                Payload::Trits(vec![Trit::Pos, Trit::Zero, Trit::Neg]),
                Meta::exact(Provenance::Root),
            )
            .unwrap(),
        )],
    };
    match jit_run(&prog) {
        Ok(v) => assert_eq!(
            v.payload(),
            &Payload::Trits(vec![Trit::Neg, Trit::Zero, Trit::Pos])
        ),
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected JIT error: {e}"),
    }
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

#[test]
fn jit_trit_add_matches_oracle() {
    // 5 + 4 = 9 in 3 trits: [+,-,-] + [0,+,+] = [+,0,0] — the in-process ripple-carry path.
    let prog = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
        ],
    };
    match jit_run(&prog) {
        Ok(v) => assert_eq!(
            v.payload(),
            &Payload::Trits(vec![Trit::Pos, Trit::Zero, Trit::Zero])
        ),
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected JIT error: {e}"),
    }
}

#[test]
fn jit_trit_overflow_is_explicit() {
    // 4 + 4 = 8 in 2 trits overflows: the kernel returns the non-zero status, surfaced as an
    // explicit Overflow — never a silently-wrapped (unwritten) buffer. Mutant-witness: a `void`
    // kernel (no status) could not signal this in-process.
    let prog = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
            Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
        ],
    };
    match jit_run(&prog) {
        Ok(v) => panic!("overflow must not produce a value, got {:?}", v.payload()),
        Err(AotError::Overflow(_)) => { /* expected */ }
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected JIT error: {e}"),
    }
}
