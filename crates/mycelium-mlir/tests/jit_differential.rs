//! M-340 — interp↔**JIT** differential (NFR-7; VR-4; RR-12; ADR-009; phase-3.md Batch L).
//!
//! The in-process JIT path (`mycelium_mlir::jit_run`: emit fn IR → `clang -shared` → `dlopen` →
//! call) must agree with the M-110 reference interpreter on the observable (`repr + payload +
//! guarantee`) and **validate through the single shared M-210 checker** (`ObservationalEquiv`) — the
//! same checker the AOT/env-machine differentials use. Skips when `clang` is absent (the house
//! "skip gracefully" idiom).

mod common;
use common::{byte, tern, A, B};

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, Node, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::AotError;
use mycelium_numerics::Certificate;

/// The bit/trit subset the JIT lowers — the same shape the AOT differential uses.
fn corpus() -> Vec<Node> {
    vec![
        Node::Const(byte(A)),
        Node::Op {
            prim: "bit.xor".into(),
            args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
        },
        Node::Op {
            prim: "trit.neg".into(),
            args: vec![Node::Const(tern(vec![
                Trit::Pos,
                Trit::Zero,
                Trit::Neg,
                Trit::Pos,
            ]))],
        },
        // M-301 trit carry arithmetic (in range): add 5+4=9 and mul 2*3=6 over 3 trits, in-process.
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
            ],
        },
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
            ],
        },
    ]
}

#[test]
fn interp_and_jit_agree_through_the_shared_checker() {
    let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine));
    for (i, node) in corpus().iter().enumerate() {
        let jit = match mycelium_mlir::jit_run(node) {
            Ok(v) => v,
            Err(AotError::ToolchainMissing(_)) => return, // environment skip
            Err(e) => panic!("program #{i}: JIT errored: {e}"),
        };
        let reference = interp.eval(node).expect("interp evaluates");
        let obs = |v: &Value| (v.repr().clone(), v.payload().clone(), v.meta().guarantee());
        // Mutant-witness: a wrong store offset or fn signature in the JIT kernel would diverge here.
        assert_eq!(
            obs(&reference),
            obs(&jit),
            "program #{i} interp↔JIT diverged"
        );
        assert_eq!(
            check(
                &reference,
                &jit,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "program #{i}: the shared checker must validate the interp↔JIT pair"
        );
    }
}
