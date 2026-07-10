//! Increment-3 — tail-recursive Fix as an iterative LLVM loop (RFC-0004 §11.6; DN-15 §8).
//!
//! Exercises the native tail-Fix backend against the M-110 reference interpreter for terminating
//! programs (value-equality), verifies the graceful depth-limit on non-terminating programs (both
//! sides refuse explicitly, never crash/hang), and confirms refusals for out-of-scope shapes
//! (FixGroup, non-tail Fix, non-λ.Match Fix body).
//!
//! Guarantee tag: **Declared** — hand-written LLVM IR iterative loop; the differential is
//! empirical evidence, not a proof (VR-5; never upgraded without a checked basis). Skips
//! gracefully when `llc`/`clang` are absent (house idiom; `AotError::ToolchainMissing`).

mod common;
use common::{byte, observable, A, B, ONES};

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{Alt, GuaranteeStrength, Node, Payload, Repr, Value};
use mycelium_interp::{EvalError, IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::AotError;
use mycelium_numerics::Certificate;

// ─── helpers ──────────────────────────────────────────────────────────────────────────────────

/// Evaluate `node` with the reference interpreter using a bounded fuel clock (so a diverging
/// program surfaces as `EvalError::FuelExhausted` instead of hanging forever). The fuel is
/// intentionally small — enough for a few recursion steps but not for a runaway loop.
fn interp_bounded(node: &Node, fuel: u64) -> Result<Value, EvalError> {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .with_fuel(fuel)
        .eval(node)
}

// ─── Terminating tail-Fix programs ────────────────────────────────────────────────────────────

/// Build a `Binary{8}` `Value` from an integer (LSB-first, truncated to 8 bits).
/// Element `i` = bit `i` of `n`.
fn byte_n(n: u8) -> Value {
    let bits: Vec<bool> = (0..8).map(|i| (n >> i) & 1 == 1).collect();
    byte(bits.try_into().expect("8 bits"))
}

/// Countdown program: `f = λn. Match n { Lit 2 → App(f,1) ; Lit 1 → App(f,0) ; _ → App(f,0) }`
/// but with a base case at Lit 0 → return B.
///
/// Concretely:
/// ```
/// f = Fix(self, λn. Match n {
///     Lit byte(2) => App(self, byte(1))    [tail]
///     Lit byte(1) => App(self, byte(0))    [tail]
///     Lit byte(0) => byte(B)               [base]
///     default     => App(self, byte(0))    [tail → also reaches base case]
/// })
/// App(f, byte(2))
/// ```
/// Expected: B after 3 iterations (2→1→0→B). Differential: interp == native == byte(B).
fn countdown_program() -> Node {
    let b0 = Node::Const(byte_n(0));
    let b1 = Node::Const(byte_n(1));
    let b2 = Node::Const(byte_n(2));
    let result_b = Node::Const(byte(B));

    // f = Fix(self, λn. Match n { ... })
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![
                // Lit byte(2) → tail call f(1)
                Alt::Lit {
                    value: byte_n(2),
                    body: Node::App {
                        func: Box::new(Node::Var("self".into())),
                        arg: Box::new(b1.clone()),
                    },
                },
                // Lit byte(1) → tail call f(0)
                Alt::Lit {
                    value: byte_n(1),
                    body: Node::App {
                        func: Box::new(Node::Var("self".into())),
                        arg: Box::new(b0.clone()),
                    },
                },
                // Lit byte(0) → base case: return B
                Alt::Lit {
                    value: byte_n(0),
                    body: result_b,
                },
            ],
            // default → tail call f(0) (so any other value converges too)
            default: Some(Box::new(Node::App {
                func: Box::new(Node::Var("self".into())),
                arg: Box::new(b0),
            })),
        }),
    };

    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(b2),
    }
}

/// One-step trivial program: `f = λn. Match n { Lit A → byte(B) ; _ → App(f, byte(A)) }`,
/// applied to `byte(ONES)`. Since ONES != A, the default arm fires once (tail), then the Lit A
/// arm fires (base), returning B. Differential: interp == native == byte(B).
fn one_step_program() -> Node {
    let result_b = Node::Const(byte(B));

    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![
                // Lit A → base case
                Alt::Lit {
                    value: byte(A),
                    body: result_b,
                },
            ],
            // default → tail call f(A)
            default: Some(Box::new(Node::App {
                func: Box::new(Node::Var("self".into())),
                arg: Box::new(Node::Const(byte(A))),
            })),
        }),
    };

    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte(ONES))),
    }
}

// ─── Non-terminating program ───────────────────────────────────────────────────────────────────

/// Purely diverging program in the canonical tail-Fix shape: `f = λn. Match n { default → App(f, n) }`,
/// applied to `byte(A)`.
///
/// There are no Lit arms and no base case — the default arm always tail-calls back. This fits the
/// canonical λ.Match shape but never terminates. Both paths must refuse explicitly and gracefully:
///
/// - native: `AotError::DepthLimit` (depth ceiling reached; DEPTHLIMIT_SENTINEL read-back)
/// - interp: `EvalError::FuelExhausted` (bounded fuel exhausted)
///
/// Neither is `Ok(_)` — both refusals are explicit (G2 / SC-3 / DN-05 #1).
fn diverging_program() -> Node {
    // f = λn. Match n { default → App(f, n) }  — no base case, loops forever
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![], // no Lit arms
            default: Some(Box::new(Node::App {
                func: Box::new(Node::Var("self".into())),
                arg: Box::new(Node::Var("n".into())),
            })),
        }),
    };

    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte(A))),
    }
}

// ─── Out-of-scope refusal programs ────────────────────────────────────────────────────────────

/// A `FixGroup` (mutual recursion) — always refused: `UnsupportedNode` (G2).
fn fixgroup_program() -> Node {
    // `fixgroup { f = λn. App(g, n) ; g = λn. App(f, n) } in App(f, byte(A))`
    // We can't build FixGroup directly in Core IR as Node doesn't have it at top-level easily.
    // Instead use the outer Node::FixGroup which is the mutual-recursion form.
    Node::FixGroup {
        defs: vec![
            (
                "f".into(),
                Box::new(Node::Lam {
                    param: "n".into(),
                    body: Box::new(Node::App {
                        func: Box::new(Node::Var("g".into())),
                        arg: Box::new(Node::Var("n".into())),
                    }),
                }),
            ),
            (
                "g".into(),
                Box::new(Node::Lam {
                    param: "n".into(),
                    body: Box::new(Node::App {
                        func: Box::new(Node::Var("f".into())),
                        arg: Box::new(Node::Var("n".into())),
                    }),
                }),
            ),
        ],
        body: Box::new(Node::App {
            func: Box::new(Node::Var("f".into())),
            arg: Box::new(Node::Const(byte(A))),
        }),
    }
}

/// A non-tail Fix: the self-name appears as a non-tail-position App (the result is `bit.not` of
/// the self-call, not the self-call itself). Must return `UnsupportedNode` (G2).
fn non_tail_fix_program() -> Node {
    // f = λn. bit.not(App(f, n))   — self used but not in tail position
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::App {
                func: Box::new(Node::Var("self".into())),
                arg: Box::new(Node::Var("n".into())),
            }],
        }),
    };
    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte(A))),
    }
}

/// A Fix whose body is NOT a Lam (just a Const) — refused as `UnsupportedNode` (G2).
fn non_lam_fix_program() -> Node {
    // Fix(self, byte(A))  — body is a constant, not a Lam
    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(Node::Const(byte(A))),
        }),
        arg: Box::new(Node::Const(byte(A))),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────────────────────

/// Countdown: interp and native agree on the 3-step result (byte B).
/// Mutant-witness: if the loop emitted the wrong back-edge (e.g., returned on the first
/// iteration), the native result would differ from the interpreter's.
#[test]
fn countdown_interp_and_native_agree() {
    let prog = countdown_program();
    let native = match mycelium_mlir::compile_and_run(&prog) {
        Ok(v) => v,
        Err(AotError::ToolchainMissing(_)) => return, // env skip
        Err(e) => panic!("native path errored on countdown: {e}"),
    };
    let interp = interp_bounded(&prog, 10_000).expect("interp must eval countdown");

    assert_eq!(
        observable(&interp),
        observable(&native),
        "countdown: interp={:?} vs native={:?}",
        interp.payload(),
        native.payload()
    );
    // M-210 shared checker.
    assert_eq!(
        check(
            &interp,
            &native,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "M-210 checker must validate the countdown interp↔native pair"
    );
    // Sanity: the result is actually B.
    assert_eq!(
        native.payload(),
        &Payload::Bits(B.to_vec()),
        "countdown must produce byte(B)"
    );
    assert_eq!(native.repr(), &Repr::Binary { width: 8 });
}

/// Increment-3 review (Copilot #224): a tail-recursive arm whose **next step is computed via a
/// nested `Match`** — a `Match` in the pre-tail binding sequence. `f = λn. Match n { Lit 0 → B ;
/// default → App(self, Match n { Lit 2 → 1 ; _ → 0 }) }`, applied to `byte(2)`. The program is valid
/// (the interpreter evaluates it 2 → 1 → 0 → B), but the **native** path refuses it: a `Match`
/// introduces basic blocks that would invalidate the loop back-edge phi, so it is an explicit
/// `UnsupportedNode` (deferred — DN-15 §8.5), never fragile IR (G2).
fn step_via_match_program() -> Node {
    // The recursion argument is itself a Match (a pre-tail binding after ANF flattening).
    let step_match = || Node::Match {
        scrutinee: Box::new(Node::Var("n".into())),
        alts: vec![Alt::Lit {
            value: byte_n(2),
            body: Node::Const(byte_n(1)),
        }],
        default: Some(Box::new(Node::Const(byte_n(0)))),
    };
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Lit {
                value: byte_n(0),
                body: Node::Const(byte(B)),
            }],
            default: Some(Box::new(Node::App {
                func: Box::new(Node::Var("self".into())),
                arg: Box::new(step_match()),
            })),
        }),
    };
    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte_n(2))),
    }
}

#[test]
fn step_via_match_in_pre_tail_bindings_is_explicitly_refused() {
    // Computing the next step via a nested `Match` puts a `Match` in the tail arm's pre-tail binding
    // sequence. That introduces basic blocks which would invalidate the loop back-edge phi, so the
    // native path REFUSES it explicitly (UnsupportedNode) — never fragile/incorrect IR (G2; DN-15
    // §8.5 deferred). The interpreter evaluates it fine (it's a valid program); the boundary is a
    // native-codegen limitation, honestly surfaced, not a semantic restriction.
    let prog = step_via_match_program();
    match mycelium_mlir::compile_and_run(&prog) {
        Err(AotError::UnsupportedNode(_)) => { /* expected explicit refusal */ }
        Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
        Ok(v) => panic!(
            "a Match-in-pre-tail-bindings program must be refused; native returned {:?}",
            v.payload()
        ),
        Err(e) => panic!("step-via-match errored with an unexpected variant: {e}"),
    }
    // The interpreter still evaluates it (sanity: the program itself is well-formed).
    assert!(
        interp_bounded(&prog, 10_000).is_ok(),
        "the interpreter should evaluate the (valid) step-via-match program"
    );
}

/// One-step: interp and native agree (single default-arm iteration, then base B).
#[test]
fn one_step_interp_and_native_agree() {
    let prog = one_step_program();
    let native = match mycelium_mlir::compile_and_run(&prog) {
        Ok(v) => v,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("native path errored on one_step: {e}"),
    };
    let interp = interp_bounded(&prog, 10_000).expect("interp must eval one_step");

    assert_eq!(
        observable(&interp),
        observable(&native),
        "one_step: interp={:?} vs native={:?}",
        interp.payload(),
        native.payload()
    );
    assert_eq!(
        check(
            &interp,
            &native,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        }
    );
    assert_eq!(
        native.payload(),
        &Payload::Bits(B.to_vec()),
        "one_step must produce byte(B)"
    );
}

/// Diverging: the canonical `λ.Match` tail-Fix with no base case loops forever. Native MUST
/// return `AotError::DepthLimit` (graceful, sentinel-based), never crash/hang. The interpreter
/// MUST return `Err(EvalError::FuelExhausted)` with bounded fuel. Neither side may produce
/// `Ok(_)` — both refusals are explicit (G2 / DN-05 #1 / SC-3).
///
/// Mutant-witness: if the depth-limit sentinel path was not emitted (e.g. the depth-check `br`
/// was dropped), the native executable would loop until OS-killed; with it the test completes
/// in bounded time via the graceful exit.
#[test]
fn diverging_native_is_depth_limit_never_hang() {
    let prog = diverging_program();

    // Native must refuse with DepthLimit.
    match mycelium_mlir::compile_and_run(&prog) {
        Ok(v) => panic!(
            "diverging program must not produce a value; got {:?}",
            v.payload()
        ),
        Err(AotError::DepthLimit(_)) => { /* expected — graceful explicit refusal */ }
        Err(AotError::ToolchainMissing(_)) => return, // env skip
        Err(e) => panic!("diverging program: unexpected native error: {e}"),
    }

    // Interpreter must also refuse (FuelExhausted with a tiny budget).
    let interp_result = interp_bounded(&prog, 100);
    assert!(
        matches!(interp_result, Err(EvalError::FuelExhausted)),
        "diverging program: interpreter must FuelExhaust with bounded fuel; got {interp_result:?}"
    );
}

/// `FixGroup` (mutual recursion) is always refused with `UnsupportedNode` (never silent — G2).
#[test]
fn fixgroup_is_unsupported_node() {
    let prog = fixgroup_program();
    assert!(
        matches!(
            mycelium_mlir::emit_llvm_ir(&prog),
            Err(AotError::UnsupportedNode(_))
        ),
        "FixGroup must return UnsupportedNode; got {:?}",
        mycelium_mlir::emit_llvm_ir(&prog)
    );
}

/// Non-tail Fix (self-ref not in tail position) is refused with `UnsupportedNode` (G2).
#[test]
fn non_tail_fix_is_unsupported_node() {
    let prog = non_tail_fix_program();
    assert!(
        matches!(
            mycelium_mlir::emit_llvm_ir(&prog),
            Err(AotError::UnsupportedNode(_))
        ),
        "non-tail Fix must return UnsupportedNode; got {:?}",
        mycelium_mlir::emit_llvm_ir(&prog)
    );
}

/// A Fix whose body is not a Lam is refused with `UnsupportedNode` (G2).
#[test]
fn non_lam_fix_body_is_unsupported_node() {
    let prog = non_lam_fix_program();
    assert!(
        matches!(
            mycelium_mlir::emit_llvm_ir(&prog),
            Err(AotError::UnsupportedNode(_))
        ),
        "Fix with non-Lam body must return UnsupportedNode; got {:?}",
        mycelium_mlir::emit_llvm_ir(&prog)
    );
}
