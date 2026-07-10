//! In-crate tests for `llvm.rs` — the direct-LLVM AOT backend (CLAUDE.md test-layout rule;
//! extracted as-touched from the former inline `#[cfg(test)] mod tests` by M-850, the M-797 lazy
//! retrofit). White-box access via `use crate::llvm::*`; the logic file carries no inline
//! `#[cfg(test)]` code.
//!
//! These are pure **emission** checks (deterministic textual IR, no toolchain) plus the
//! `compile_and_run` compiled-path smoke tests, which **skip gracefully** when `llc`/`clang` are
//! absent (`AotError::ToolchainMissing` — the house idiom). The richer recursion/closure differential
//! lives in `tests/recursion_differential.rs`, `tests/recursion_trampoline_differential.rs`, and the
//! three-way harness `tests/threeway_differential.rs`.

use crate::llvm::*;
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

fn ternary(trits: Vec<Trit>) -> Value {
    let m = trits.len() as u32;
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(trits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn not_program() -> Node {
    Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(binary(vec![true, false, true, true]))],
    }
}

fn neg_program() -> Node {
    Node::Op {
        prim: "trit.neg".into(),
        args: vec![Node::Const(ternary(vec![Trit::Pos, Trit::Zero, Trit::Neg]))],
    }
}

#[test]
fn emits_module_for_bit_not() {
    let ir = emit_llvm_ir(&not_program()).unwrap();
    assert!(ir.contains("declare i32 @putchar(i32)"));
    assert!(ir.contains("define i32 @main()"));
    assert!(ir.contains("xor i32")); // bit.not lowers to xor with 1
    assert!(ir.contains("call i32 @putchar"));
    assert!(ir.contains("ret i32 0"));
}

#[test]
fn emission_is_deterministic() {
    assert_eq!(emit_llvm_ir(&not_program()), emit_llvm_ir(&not_program()));
}

#[test]
fn emits_module_for_trit_neg() {
    let ir = emit_llvm_ir(&neg_program()).unwrap();
    assert!(ir.contains("sub i32 0,")); // trit.neg lowers to 0 - x per trit
                                        // Ternary output uses the '-'(45)/'0'(48)/'+'(43) select chain.
    assert!(ir.contains("select i1") && ir.contains("i32 45") && ir.contains("i32 43"));
    assert!(ir.contains("ret i32 0"));
}

#[test]
fn ternary_const_is_supported() {
    // M-301 trit slice: a Ternary const is now lowered (was UnsupportedRepr in the bit-only
    // slice). Mutant-witness: reverting const_lane to Binary-only would refuse this.
    let v = ternary(vec![Trit::Pos, Trit::Zero, Trit::Neg]);
    assert!(emit_llvm_ir(&Node::Const(v)).is_ok());
}

fn binop(prim: &str, a: Vec<Trit>, b: Vec<Trit>) -> Node {
    Node::Op {
        prim: prim.into(),
        args: vec![Node::Const(ternary(a)), Node::Const(ternary(b))],
    }
}

#[test]
fn trit_add_emits_ripple_carry_ir() {
    // Mutant-witness: a non-carry (elementwise) add would not emit the srem/sdiv-by-3 balancing
    // or the icmp overflow flag the read-back protocol branches on.
    let ir = emit_llvm_ir(&binop(
        "trit.add",
        vec![Trit::Pos, Trit::Neg, Trit::Neg],
        vec![Trit::Zero, Trit::Pos, Trit::Pos],
    ))
    .unwrap();
    assert!(ir.contains("srem i32") && ir.contains("sdiv i32")); // balanced-digit normalisation
    assert!(ir.contains("icmp ne i32")); // overflow flag
    assert!(ir.contains("br i1")); // read-back branch
    assert!(ir.contains("putchar(i32 33)")); // overflow sentinel '!'
}

#[test]
fn arithmetic_emission_is_deterministic() {
    let p = binop(
        "trit.mul",
        vec![Trit::Zero, Trit::Pos, Trit::Neg],
        vec![Trit::Zero, Trit::Pos, Trit::Zero],
    );
    assert_eq!(emit_llvm_ir(&p), emit_llvm_ir(&p));
}

#[test]
fn refuses_arithmetic_width_mismatch() {
    // Mutant-witness: dropping the width check would emit a ragged ripple-carry.
    let prog = binop("trit.add", vec![Trit::Pos, Trit::Zero], vec![Trit::Pos]);
    assert!(matches!(
        emit_llvm_ir(&prog),
        Err(AotError::WidthMismatch { .. })
    ));
}

#[test]
fn refuses_bit_arithmetic_on_binary_lane() {
    // Mutant-witness: dropping require_kind would let trit.add ripple over a binary lane.
    let prog = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(binary(vec![true, false])),
            Node::Const(binary(vec![false, true])),
        ],
    };
    assert!(matches!(
        emit_llvm_ir(&prog),
        Err(AotError::UnsupportedPrim(_))
    ));
}

#[test]
fn refuses_bit_op_on_ternary_lane() {
    // Mutant-witness: dropping require_kind would let bit.not mis-lower a ternary lane.
    let prog = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(ternary(vec![Trit::Pos, Trit::Neg]))],
    };
    assert!(matches!(
        emit_llvm_ir(&prog),
        Err(AotError::UnsupportedPrim(_))
    ));
}

#[test]
fn lowers_legal_swap_and_refuses_unsupported_swap_kinds() {
    // M-852 (E25-1): a `Swap` over the certified binary↔ternary class is now **lowered** natively
    // (was refused before M-852). `(2,2)` is a legal pair (2^1=2 ≤ (3^2−1)/2=4), so it emits a
    // transcode module — no longer an UnsupportedNode. (The cert basis is recorded in a dumpable IR
    // comment; the interp↔native equivalence is the `tests/swap_differential.rs` gate.)
    let legal = Node::Swap {
        src: Box::new(Node::Const(binary(vec![true, false]))),
        target: Repr::Ternary { trits: 2 },
        policy: mycelium_core::ContentHash::parse("blake3:x").unwrap(),
    };
    let ir = emit_llvm_ir(&legal).expect("a legal binary↔ternary swap now lowers (M-852)");
    assert!(
        ir.contains("; swap"),
        "the dumpable swap comment must be emitted:\n{ir}"
    );

    // Mutant-witness / never-silent (G2): an **illegal** pair `(2,4)`… is legal, so pick a clearly
    // illegal one: `(8,2)` — 2^7=128 > (3^2−1)/2=4 — must still be an explicit refusal in the default
    // (Recheck) mode. A swap kind outside bit↔ternary is never silently lowered.
    let illegal = Node::Swap {
        src: Box::new(Node::Const(binary(vec![
            true, false, true, true, false, false, true, false,
        ]))),
        target: Repr::Ternary { trits: 2 },
        policy: mycelium_core::ContentHash::parse("blake3:x").unwrap(),
    };
    assert!(
        matches!(emit_llvm_ir(&illegal), Err(AotError::UnsupportedNode(_))),
        "an illegal-pair swap must be refused in the default Recheck mode (never silent; G2)"
    );
}

#[test]
fn refuses_width_mismatch() {
    // Mutant-witness: dropping the width check would emit a ragged elementwise op.
    let prog = Node::Op {
        prim: "bit.and".into(),
        args: vec![
            Node::Const(binary(vec![true, false, true])),
            Node::Const(binary(vec![true, false])),
        ],
    };
    assert!(matches!(
        emit_llvm_ir(&prog),
        Err(AotError::WidthMismatch { .. })
    ));
}

// --- compiled-path smoke test (skips when llc/clang are absent) ---------------------------

#[test]
fn native_bit_not_matches_interpreter() {
    let prog = not_program();
    match compile_and_run(&prog) {
        Ok(v) => {
            // Mutant-witness: if bit.not lowered to `or`/`and` instead of `xor _, 1`, the
            // payload would differ from the complemented input.
            assert_eq!(v.payload(), &Payload::Bits(vec![false, true, false, false]));
            assert_eq!(v.repr(), &Repr::Binary { width: 4 });
        }
        Err(AotError::ToolchainMissing(_)) => { /* environment skip — house idiom */ }
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

#[test]
fn native_trit_neg_matches_interpreter() {
    // Mutant-witness: if trit.neg lowered to anything but `0 - x` (or the output select chain
    // mapped the wrong char), the negated payload `[-,0,+]` would differ.
    match compile_and_run(&neg_program()) {
        Ok(v) => {
            assert_eq!(
                v.payload(),
                &Payload::Trits(vec![Trit::Neg, Trit::Zero, Trit::Pos])
            );
            assert_eq!(v.repr(), &Repr::Ternary { trits: 3 });
        }
        Err(AotError::ToolchainMissing(_)) => { /* environment skip */ }
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

#[test]
fn native_trit_add_matches_oracle() {
    // 5 + 4 = 9 in 3 trits: [+,-,-] + [0,+,+] = [+,0,0]. Mutant-witness: a missing carry would
    // yield the elementwise (wrong) sum, and a wrong balancing constant would mis-encode.
    let prog = binop(
        "trit.add",
        vec![Trit::Pos, Trit::Neg, Trit::Neg],
        vec![Trit::Zero, Trit::Pos, Trit::Pos],
    );
    match compile_and_run(&prog) {
        Ok(v) => assert_eq!(
            v.payload(),
            &Payload::Trits(vec![Trit::Pos, Trit::Zero, Trit::Zero])
        ),
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

#[test]
fn native_trit_sub_matches_oracle() {
    // 9 - 4 = 5 in 3 trits: [+,0,0] - [0,+,+] = [+,-,-].
    let prog = binop(
        "trit.sub",
        vec![Trit::Pos, Trit::Zero, Trit::Zero],
        vec![Trit::Zero, Trit::Pos, Trit::Pos],
    );
    match compile_and_run(&prog) {
        Ok(v) => assert_eq!(
            v.payload(),
            &Payload::Trits(vec![Trit::Pos, Trit::Neg, Trit::Neg])
        ),
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

#[test]
fn native_trit_mul_matches_oracle() {
    // 2 * 3 = 6 in 3 trits: [0,+,-] * [0,+,0] = [+,-,0]. Mutant-witness: a wrong shift in the
    // shifted-accumulate, or reading the high (overflow) half, would diverge.
    let prog = binop(
        "trit.mul",
        vec![Trit::Zero, Trit::Pos, Trit::Neg],
        vec![Trit::Zero, Trit::Pos, Trit::Zero],
    );
    match compile_and_run(&prog) {
        Ok(v) => assert_eq!(
            v.payload(),
            &Payload::Trits(vec![Trit::Pos, Trit::Neg, Trit::Zero])
        ),
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

#[test]
fn native_trit_add_overflow_is_explicit() {
    // 4 + 4 = 8 in 2 trits (max magnitude 4) overflows. The native path must report it through
    // the read-back protocol — an explicit Overflow, never a silent wrap. Mutant-witness:
    // dropping the final-carry flag would print a wrapped result instead.
    let prog = binop(
        "trit.add",
        vec![Trit::Pos, Trit::Pos],
        vec![Trit::Pos, Trit::Pos],
    );
    match compile_and_run(&prog) {
        Ok(v) => panic!("overflow must not produce a value, got {:?}", v.payload()),
        Err(AotError::Overflow(_)) => { /* expected */ }
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

#[test]
fn native_trit_mul_overflow_is_explicit() {
    // 4 * 4 = 16 in 2 trits overflows (high trits non-zero).
    let prog = binop(
        "trit.mul",
        vec![Trit::Pos, Trit::Pos],
        vec![Trit::Pos, Trit::Pos],
    );
    match compile_and_run(&prog) {
        Ok(v) => panic!("overflow must not produce a value, got {:?}", v.payload()),
        Err(AotError::Overflow(_)) => {}
        Err(AotError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected AOT error: {e}"),
    }
}

// ─── M-860: parallel per-function codegen — byte-identical vs sequential ──────────────────────

/// A multi-function batch: several distinct, independent programs (different prims/operands, so
/// distinct content hashes → the content-key sort actually reorders the parallel work relative to
/// input order, exercising the scatter-back-to-original-index path, not a no-op sort).
fn multi_function_batch() -> Vec<Node> {
    vec![
        not_program(),
        neg_program(),
        binop(
            "trit.add",
            vec![Trit::Pos, Trit::Zero],
            vec![Trit::Neg, Trit::Pos],
        ),
        binop(
            "trit.sub",
            vec![Trit::Zero, Trit::Pos],
            vec![Trit::Pos, Trit::Neg],
        ),
        binop(
            "trit.mul",
            vec![Trit::Pos, Trit::Zero],
            vec![Trit::Neg, Trit::Zero],
        ),
        Node::Const(binary(vec![true, false, false, true])),
    ]
}

#[test]
fn parallel_emit_matches_sequential_emit_byte_identical() {
    let nodes = multi_function_batch();
    let sequential: Vec<Result<String, AotError>> = nodes.iter().map(emit_llvm_ir).collect();
    let parallel = emit_llvm_ir_many(&nodes);
    assert_eq!(
        parallel.len(),
        sequential.len(),
        "parallel batch must return exactly one result per input"
    );
    for (i, (p, s)) in parallel.iter().zip(sequential.iter()).enumerate() {
        assert_eq!(
            p, s,
            "program #{i}: parallel emit diverged from sequential emit"
        );
    }
}

#[test]
fn parallel_emit_output_order_is_input_order_not_content_hash_order() {
    // A stronger check than the byte-equality above: the OUTPUT VEC's i-th element must be the i-th
    // INPUT's result, not merely "the same multiset of results" (a scatter-by-completion-order bug
    // would still pass a naive set-equality check but fail this).
    let nodes = multi_function_batch();
    let parallel = emit_llvm_ir_many(&nodes);
    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(
            parallel[i],
            emit_llvm_ir(node),
            "program #{i}: output position must match its own input's emission, not another \
             program's (a scatter/reorder bug)"
        );
    }
}

#[test]
fn parallel_emit_is_stable_under_repeated_runs() {
    // Rerunning the parallel batch (fresh thread-pool scheduling each time) must not perturb the
    // output — Exact by construction (M-860 DoD), not "usually agrees".
    let nodes = multi_function_batch();
    let first = emit_llvm_ir_many(&nodes);
    for _ in 0..5 {
        assert_eq!(emit_llvm_ir_many(&nodes), first);
    }
}

#[test]
fn parallel_emit_handles_an_empty_batch() {
    let empty: Vec<Node> = Vec::new();
    assert_eq!(
        emit_llvm_ir_many(&empty),
        Vec::<Result<String, AotError>>::new()
    );
}

#[test]
fn parallel_emit_preserves_a_refusal_at_its_original_index() {
    // Mix a well-formed program with one that `emit_llvm_ir` refuses (a width mismatch); the
    // refusal must land at its own index, not swallow or misplace the others.
    let bad = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(ternary(vec![Trit::Pos, Trit::Zero])),
            Node::Const(ternary(vec![Trit::Neg])),
        ],
    };
    let nodes = vec![not_program(), bad, neg_program()];
    let parallel = emit_llvm_ir_many(&nodes);
    assert!(parallel[0].is_ok());
    assert!(matches!(parallel[1], Err(AotError::WidthMismatch { .. })));
    assert!(parallel[2].is_ok());
}
