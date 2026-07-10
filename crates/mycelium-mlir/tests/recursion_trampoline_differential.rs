//! M-850 (epic E25-1) — **direct-LLVM heap-trampoline** for full object-level recursion: non-tail
//! single `Fix` and mutual-recursion `FixGroup`, run on an explicit `@malloc`'d control stack (never
//! the C stack). This is the differential that promotes the trampoline tag from `Declared` to
//! `Empirical`: it value-checks **interp ≡ direct-LLVM** over a recursion corpus the *tail*-loop path
//! ([`recursion_differential.rs`]) cannot lower (non-tail continuations, `FixGroup`), confirms the
//! deep-recursion ceiling reaches a graceful `DepthLimit` (no C-stack overflow — DN-05 #1; DN-15
//! §8.4), and pins the still-refused boundaries (Match-in-pre-tail step, non-`λ.Match` shapes) as
//! honest `UnsupportedNode` (G2/VR-5 — never a silent mis-lowering).
//!
//! Guarantee tag: **Empirical** — the differential below is checked (interp ≡ direct-LLVM over the
//! corpus) and a `cargo-mutants` witness of the frame/continuation logic is caught by it (M-850
//! report). It is **not** `Proven` — hand-written textual LLVM IR has no machine-checked refinement
//! theorem (VR-5: never upgraded past the checked basis). Skips gracefully when `llc`/`clang` are
//! absent (`AotError::ToolchainMissing` — the house idiom).
//!
//! The MLIR-dialect leg is **not** exercised here: `dialect::native` explicitly refuses recursion
//! (`Fix`/`FixGroup` → `UnsupportedNode`; `src/dialect/native.rs`), so for this corpus the third
//! differential edge is an honest refusal, not a vacuous skip. The element-wise three-way leg lives
//! in `tests/threeway_differential.rs` (with its `ran_mlir` non-vacuity guard). **M-858:** this
//! paragraph used to be an unchecked doc-comment claim; it is now a checked fact — see
//! `tests/unified_threeway_differential.rs::dialect_honestly_refuses_closures_and_recursion`, which
//! actually calls `mlir_compile_and_run` on a `Fix`-based program and asserts the
//! `DialectError::Unsupported` refusal, never just asserting it in prose.

mod common;
use common::observable;

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{Alt, GuaranteeStrength, Node, Payload, Repr, Value};
use mycelium_interp::{EvalError, IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::AotError;
use mycelium_numerics::Certificate;

// ─── helpers ──────────────────────────────────────────────────────────────────────────────────

/// A `Binary{8}` `Value` from an integer (LSB-first; element `i` = bit `i` of `n`). The trampoline's
/// narrow ABI carries exactly `Binary{8}` accumulators.
fn byte_n(n: u8) -> Value {
    let bits: Vec<bool> = (0..8).map(|i| (n >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits),
        mycelium_core::Meta::exact(mycelium_core::Provenance::Root),
    )
    .expect("8-bit binary value")
}

/// The reference interpreter under a bounded fuel clock — a diverging program surfaces as
/// `EvalError::FuelExhausted` rather than hanging (the interpreter is O(1)-stack; it refuses by
/// fuel, the native path by depth — both never-silent, DN-15 §8.4).
fn interp_bounded(node: &Node, fuel: u64) -> Result<Value, EvalError> {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .with_fuel(fuel)
        .eval(node)
}

/// Assert interp ≡ direct-LLVM on a **terminating** trampoline program: both observable triples are
/// equal **and** the shared M-210 checker validates the pair at `Exact`. Skips when the toolchain is
/// absent. `expected` pins the actual value so the test is not merely self-consistent.
fn assert_interp_eq_native(label: &str, prog: &Node, expected: &[bool]) {
    let native = match mycelium_mlir::compile_and_run(prog) {
        Ok(v) => v,
        Err(AotError::ToolchainMissing(_)) => return, // env skip — house idiom
        Err(e) => panic!("{label}: direct-LLVM path errored: {e}"),
    };
    let interp =
        interp_bounded(prog, 100_000).unwrap_or_else(|e| panic!("{label}: interp errored: {e:?}"));

    assert_eq!(
        observable(&interp),
        observable(&native),
        "{label}: interp={:?} vs native={:?}",
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
        },
        "{label}: the shared M-210 checker must validate the interp↔native pair"
    );
    assert_eq!(
        native.payload(),
        &Payload::Bits(expected.to_vec()),
        "{label}: native produced an unexpected value"
    );
    assert_eq!(native.repr(), &Repr::Binary { width: 8 });
}

// ─── Non-tail single `Fix` corpus (Cont::{Not, And, Or, Xor}) ───────────────────────────────────

/// `f = λn. Match n { Lit 0 → base ; default → wrap(App(self, byte 0)) }`, applied to `byte 1`.
/// The default arm fires once (a non-tail call whose result is wrapped by `wrap`), then the base arm
/// (Lit 0) returns `base` — so the observable result is `wrap(base)`. The single non-tail frame
/// exercises the heap control stack + the defunctionalized continuation `wrap`.
fn non_tail_fix(base: u8, wrap: Node) -> Node {
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Lit {
                value: byte_n(0),
                body: Node::Const(byte_n(base)),
            }],
            default: Some(Box::new(wrap)),
        }),
    };
    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte_n(1))),
    }
}

/// A single recursive call `App(self, byte 0)` (the inner non-tail call the wrapping op consumes).
fn self_call_0() -> Node {
    Node::App {
        func: Box::new(Node::Var("self".into())),
        arg: Box::new(Node::Const(byte_n(0))),
    }
}

fn bits(n: u8) -> Vec<bool> {
    (0..8).map(|i| (n >> i) & 1 == 1).collect()
}

/// `Cont::Not`: `f(1) = bit.not(f(0)) = bit.not(0xAA) = 0x55`.
#[test]
fn non_tail_not_interp_eq_native() {
    let prog = non_tail_fix(
        0xAA,
        Node::Op {
            prim: "bit.not".into(),
            args: vec![self_call_0()],
        },
    );
    assert_interp_eq_native("non-tail bit.not", &prog, &bits(!0xAA));
}

/// `Cont::And`: `f(1) = bit.and(f(0), 0x0F) = bit.and(0xAA, 0x0F) = 0x0A`. The saved operand `0x0F`
/// is a pre-call binding materialized onto the heap frame.
#[test]
fn non_tail_and_interp_eq_native() {
    let prog = non_tail_fix(
        0xAA,
        Node::Op {
            prim: "bit.and".into(),
            args: vec![self_call_0(), Node::Const(byte_n(0x0F))],
        },
    );
    assert_interp_eq_native("non-tail bit.and", &prog, &bits(0xAA & 0x0F));
}

/// `Cont::Or`: `f(1) = bit.or(f(0), 0x0F) = bit.or(0xAA, 0x0F) = 0xAF`.
#[test]
fn non_tail_or_interp_eq_native() {
    let prog = non_tail_fix(
        0xAA,
        Node::Op {
            prim: "bit.or".into(),
            args: vec![self_call_0(), Node::Const(byte_n(0x0F))],
        },
    );
    assert_interp_eq_native("non-tail bit.or", &prog, &bits(0xAA | 0x0F));
}

/// `Cont::Xor`: `f(1) = bit.xor(f(0), 0xFF) = bit.xor(0xAA, 0xFF) = 0x55`. The xor-with-operand path
/// (distinct from `Not`'s xor-255) — a wrong operand source on the frame would diverge here.
#[test]
fn non_tail_xor_interp_eq_native() {
    let prog = non_tail_fix(
        0xAA,
        Node::Op {
            prim: "bit.xor".into(),
            args: vec![self_call_0(), Node::Const(byte_n(0xFF))],
        },
    );
    assert_interp_eq_native("non-tail bit.xor", &prog, &bits(0xAA ^ 0xFF));
}

/// **Two stacked non-tail frames** — `f(2) = not(f(1))`, `f(1) = not(f(0))`, `f(0) = 0xAA`. The
/// result is `not(not(0xAA)) = 0xAA`. This exercises *more than one* live heap frame and the LIFO
/// unwind order: a frame-index or unwind-order bug (e.g. applying the continuations in the wrong
/// order, or off-by-one on `sp`) is caught because the two `Not`s must compose in the right order to
/// round-trip. Built directly (the recursion descends 2 → 1 → 0 via the Lit arms).
fn two_frame_program() -> Node {
    // f = λn. Match n { Lit 0 → 0xAA ; Lit 1 → not(f(0)) ; default(2) → not(f(1)) }, App(f, 2).
    let not_call = |arg: u8| Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::App {
            func: Box::new(Node::Var("self".into())),
            arg: Box::new(Node::Const(byte_n(arg))),
        }],
    };
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![
                Alt::Lit {
                    value: byte_n(0),
                    body: Node::Const(byte_n(0xAA)),
                },
                Alt::Lit {
                    value: byte_n(1),
                    body: not_call(0),
                },
            ],
            default: Some(Box::new(not_call(1))),
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
fn two_stacked_frames_interp_eq_native() {
    // not(not(0xAA)) = 0xAA — the double-complement round-trips iff both frames unwind correctly.
    assert_interp_eq_native(
        "two stacked non-tail frames",
        &two_frame_program(),
        &bits(0xAA),
    );
}

// ─── Mutual recursion via `FixGroup` ────────────────────────────────────────────────────────────

/// A two-member mutual recursion that crosses the member boundary on the heap stack:
/// `f(n) = match n { 0 → 0xAA ; default → g(0) }`,
/// `g(n) = match n { 0 → bit.not(f(0)) ; default → f(0) }`, entered at `App(f, 1)`.
/// `f(1) → g(0) → bit.not(f(0)) → bit.not(0xAA) = 0x55`. The single non-tail frame is pushed inside
/// member `g`, applied to member `f`'s base result — so the shared member-dispatch + the frame's
/// member-crossing are both exercised.
fn fixgroup_program() -> Node {
    let f = (
        "f".to_string(),
        Box::new(Node::Lam {
            param: "n".into(),
            body: Box::new(Node::Match {
                scrutinee: Box::new(Node::Var("n".into())),
                alts: vec![Alt::Lit {
                    value: byte_n(0),
                    body: Node::Const(byte_n(0xAA)),
                }],
                default: Some(Box::new(Node::App {
                    func: Box::new(Node::Var("g".into())),
                    arg: Box::new(Node::Const(byte_n(0))),
                })),
            }),
        }),
    );
    let g = (
        "g".to_string(),
        Box::new(Node::Lam {
            param: "n".into(),
            body: Box::new(Node::Match {
                scrutinee: Box::new(Node::Var("n".into())),
                alts: vec![Alt::Lit {
                    value: byte_n(0),
                    body: Node::Op {
                        prim: "bit.not".into(),
                        args: vec![Node::App {
                            func: Box::new(Node::Var("f".into())),
                            arg: Box::new(Node::Const(byte_n(0))),
                        }],
                    },
                }],
                default: Some(Box::new(Node::App {
                    func: Box::new(Node::Var("f".into())),
                    arg: Box::new(Node::Const(byte_n(0))),
                })),
            }),
        }),
    );
    Node::FixGroup {
        defs: vec![f, g],
        body: Box::new(Node::App {
            func: Box::new(Node::Var("f".into())),
            arg: Box::new(Node::Const(byte_n(1))),
        }),
    }
}

#[test]
fn fixgroup_mutual_interp_eq_native() {
    // f(1) → g(0) → not(f(0)) = not(0xAA) = 0x55. Mutual recursion now LOWERS (was UnsupportedNode
    // before M-850); the trampoline's shared member dispatch resolves the sibling call.
    assert_interp_eq_native(
        "FixGroup mutual recursion",
        &fixgroup_program(),
        &bits(!0xAA),
    );
}

/// A `FixGroup` entered at the **second** member — `App(g, 1)` — so the trampoline's `entry` index is
/// non-zero. `g(1) → f(0) = 0xAA`. A wrong entry-member resolution would diverge here.
#[test]
fn fixgroup_entry_member_index_is_respected() {
    // `Node` now implements `Drop` (RFC-0041 §4.5 iterative destruction), so a field cannot be
    // moved out of an owned `Node` by value (E0509); take `defs` by-ref via `mem::take` instead.
    let mut base = fixgroup_program();
    let prog = match &mut base {
        Node::FixGroup { defs, .. } => Node::FixGroup {
            defs: std::mem::take(defs),
            body: Box::new(Node::App {
                func: Box::new(Node::Var("g".into())),
                arg: Box::new(Node::Const(byte_n(1))),
            }),
        },
        _ => unreachable!(),
    };
    assert_interp_eq_native("FixGroup entry at member g", &prog, &bits(0xAA));
}

// ─── Deep recursion → graceful `DepthLimit` (DN-05 #1; no C-stack overflow) ──────────────────────

/// A non-tail recursion with no reachable base case: `f = λn. Match n { Lit 0 → bit.not(f(0)) ;
/// default → bit.not(f(0)) }`, applied to `byte 0`. Every step pushes a `Not` frame and recurses on
/// `byte 0`, which re-enters the same arm — so the heap control stack grows without bound until the
/// **`AutoDepthBudget` ceiling** is hit. The native path must reach a **graceful `DepthLimit`** via
/// the sentinel read-back (never a SIGSEGV / OOM-abort / hang), in parity with the interpreter's
/// `FuelExhausted`. This is the deep-recursion robustness guarantee: the recursion runs on the
/// **heap** stack, so the host **C stack stays O(1)** (DN-15 §8.1/§8.4) — a C-recursion lowering
/// would overflow the platform stack instead.
fn deep_diverge_program() -> Node {
    let not_self_0 = || Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::App {
            func: Box::new(Node::Var("self".into())),
            arg: Box::new(Node::Const(byte_n(0))),
        }],
    };
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Lit {
                value: byte_n(0),
                body: not_self_0(),
            }],
            default: Some(Box::new(not_self_0())),
        }),
    };
    Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte_n(0))),
    }
}

#[test]
fn deep_non_tail_recursion_is_graceful_depth_limit_never_overflow() {
    let prog = deep_diverge_program();

    // Native: must refuse with a graceful DepthLimit — NOT a SIGSEGV, NOT a SIGABRT, NOT a hang. The
    // process completes in bounded time precisely because the depth-guard `br` fires before the
    // ceiling-sized frame stack overflows (the read-back maps the sentinel to AotError::DepthLimit).
    match mycelium_mlir::compile_and_run(&prog) {
        Err(AotError::DepthLimit(_)) => { /* expected — graceful, explicit refusal */ }
        Err(AotError::ToolchainMissing(_)) => return, // env skip
        Ok(v) => panic!(
            "deep non-tail recursion must not produce a value; native returned {:?}",
            v.payload()
        ),
        Err(e) => panic!("deep recursion: unexpected native error variant: {e}"),
    }

    // Interpreter parity: it also refuses (FuelExhausted) — the parity DN-15 §8.4 asserts. The
    // reference interpreter is O(1) in *Mycelium* stack but a tree-walking eval of a deeply-nested
    // **non-tail** body grows the *host Rust* call stack with depth before fuel runs out (and the
    // substitution cost is super-linear in fuel) — a property of the interpreter's recursion descent,
    // orthogonal to the M-850 native guarantee. So we run this leg on a thread with a roomy stack and
    // a small fuel budget (enough to exhaust, cheap to run): the native leg above is the
    // C-stack-safety guarantee under test; this only confirms the interpreter's refusal *variant*
    // matches (both refuse, never silent — the tail-cycle parity at full depth lives in
    // `tail_fixgroup_divergence_is_graceful_depth_limit`, where the interpreter is O(1)-stack).
    let prog_for_interp = prog.clone();
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || interp_bounded(&prog_for_interp, 200))
        .expect("spawn interp thread");
    let interp = handle.join().expect("interp thread panicked");
    assert!(
        matches!(interp, Err(EvalError::FuelExhausted)),
        "deep recursion: interpreter must FuelExhaust with bounded fuel; got {interp:?}"
    );
}

/// A **tail** `FixGroup` cycle with no base case (`f(n) → g(n)`, `g(n) → f(n)`) — a tail call still
/// advances the depth counter (no frame pushed, but `sp` bumped), so a non-terminating *tail* mutual
/// recursion reaches the SAME graceful `DepthLimit`, never an unbounded loop (G2/DN-05 #1).
fn tail_fixgroup_diverge() -> Node {
    let member = |callee: &str| {
        (
            // member name filled by caller
            String::new(),
            Node::Lam {
                param: "n".into(),
                body: Box::new(Node::Match {
                    scrutinee: Box::new(Node::Var("n".into())),
                    alts: vec![],
                    default: Some(Box::new(Node::App {
                        func: Box::new(Node::Var(callee.into())),
                        arg: Box::new(Node::Var("n".into())),
                    })),
                }),
            },
        )
    };
    let (_, fbody) = member("g");
    let (_, gbody) = member("f");
    Node::FixGroup {
        defs: vec![
            ("f".to_string(), Box::new(fbody)),
            ("g".to_string(), Box::new(gbody)),
        ],
        body: Box::new(Node::App {
            func: Box::new(Node::Var("f".into())),
            arg: Box::new(Node::Const(byte_n(1))),
        }),
    }
}

#[test]
fn tail_fixgroup_divergence_is_graceful_depth_limit() {
    let prog = tail_fixgroup_diverge();
    match mycelium_mlir::compile_and_run(&prog) {
        Err(AotError::DepthLimit(_)) => { /* expected — tail cycle still depth-bounded */ }
        Err(AotError::ToolchainMissing(_)) => return,
        Ok(v) => panic!(
            "diverging tail FixGroup must not produce a value; got {:?}",
            v.payload()
        ),
        Err(e) => panic!("tail FixGroup divergence: unexpected native error: {e}"),
    }
    assert!(
        matches!(interp_bounded(&prog, 1_000), Err(EvalError::FuelExhausted)),
        "diverging tail FixGroup: interpreter must FuelExhaust"
    );
}

// ─── Still-refused boundaries (the honest edge — G2/VR-5) ────────────────────────────────────────

/// A `Match` in a recursive arm's **pre-call** (step-computing) binding sequence stays refused — the
/// step computed via a nested `Match` (`App(self, Match n { … })`) introduces basic blocks the
/// straight-line pre-call lowering does not handle (DN-15 §8.5, an *independently* deferred
/// codegen-shape limitation distinct from the non-tail/FixGroup work M-850 lands). It is an explicit
/// `UnsupportedNode`, never fragile IR; the reference interpreter still evaluates the (valid) program.
fn step_via_match_program() -> Node {
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
                body: Node::Const(byte_n(0xAA)),
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
fn match_in_pre_call_step_stays_refused() {
    let prog = step_via_match_program();
    assert!(
        matches!(
            mycelium_mlir::emit_llvm_ir(&prog),
            Err(AotError::UnsupportedNode(_))
        ),
        "a Match-in-pre-call-step program must be refused (UnsupportedNode); got {:?}",
        mycelium_mlir::emit_llvm_ir(&prog)
    );
    // The program is well-formed: the interpreter evaluates it (2 → 1 → 0 → 0xAA). The boundary is a
    // native-codegen limitation honestly surfaced, never a semantic restriction.
    assert!(
        interp_bounded(&prog, 100_000).is_ok(),
        "the interpreter should evaluate the (valid) step-via-match program"
    );
}

/// More than one recursive call in a single arm is refused — only **linear** non-tail recursion (one
/// self/sibling call per arm) is supported (G2). `f = λn. Match n { Lit 0 → 0xAA ; default →
/// bit.and(f(0), f(0)) }` has two calls in the default arm.
#[test]
fn two_calls_per_arm_stays_refused() {
    let call0 = || Node::App {
        func: Box::new(Node::Var("self".into())),
        arg: Box::new(Node::Const(byte_n(0))),
    };
    let fix_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Match {
            scrutinee: Box::new(Node::Var("n".into())),
            alts: vec![Alt::Lit {
                value: byte_n(0),
                body: Node::Const(byte_n(0xAA)),
            }],
            default: Some(Box::new(Node::Op {
                prim: "bit.and".into(),
                args: vec![call0(), call0()],
            })),
        }),
    };
    let prog = Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(fix_body),
        }),
        arg: Box::new(Node::Const(byte_n(1))),
    };
    assert!(
        matches!(
            mycelium_mlir::emit_llvm_ir(&prog),
            Err(AotError::UnsupportedNode(_))
        ),
        "a two-recursive-calls-per-arm program must be refused; got {:?}",
        mycelium_mlir::emit_llvm_ir(&prog)
    );
}

/// A `FixGroup` member whose body is not the canonical `λparam. Match param { … }` (here a bare
/// `λn. App(g, n)` with no `Match`) is refused — the trampoline destructures only the canonical
/// shape (G2). This pins that the FixGroup support did not silently widen the accepted shape.
#[test]
fn fixgroup_non_canonical_member_stays_refused() {
    let prog = Node::FixGroup {
        defs: vec![
            (
                "f".to_string(),
                Box::new(Node::Lam {
                    param: "n".into(),
                    body: Box::new(Node::App {
                        func: Box::new(Node::Var("g".into())),
                        arg: Box::new(Node::Var("n".into())),
                    }),
                }),
            ),
            (
                "g".to_string(),
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
            arg: Box::new(Node::Const(byte_n(0))),
        }),
    };
    assert!(
        matches!(
            mycelium_mlir::emit_llvm_ir(&prog),
            Err(AotError::UnsupportedNode(_))
        ),
        "a non-λ.Match FixGroup member must be refused; got {:?}",
        mycelium_mlir::emit_llvm_ir(&prog)
    );
}
