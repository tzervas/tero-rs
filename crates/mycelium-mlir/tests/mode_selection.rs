//! M-727 — the **explicit, never-silently-selected execution mode** differential (RFC-0029 §7.3;
//! ADR-009; NFR-7; G2/VR-5).
//!
//! [`mycelium_mlir::run_mode`] dispatches on an explicit [`ExecMode`] (`Interpreter` / `Aot` / `Jit`).
//! This suite pins the two obligations of M-727's Definition of Done:
//!
//! 1. **Correctness (`Empirical`):** `JIT == interpreter` over the bit/trit subset (the same corpus
//!    `jit_differential.rs` uses), routed through the *named-mode* dispatcher rather than the raw
//!    `jit_run`/`Interpreter::eval` entry points — so the formalized API is what is checked. Skips
//!    `clang`-absent (the house idiom).
//! 2. **Never-silent selection (G2):** the JIT is reachable **only** by naming `ExecMode::Jit`. There
//!    is no `Default`/`Auto` mode and no fallback arm, so a caller that names `Interpreter`/`Aot`
//!    provably never engages the JIT, and a caller that names `Jit` on a node the JIT can't compile
//!    gets an **explicit refusal** (`ModeError::Unsupported`/`ToolchainMissing`), never a silent
//!    substitution of a different mode.
//!
//! **Guarantee:** `Empirical` — the equivalence is evidence over the corpus, never upgraded to
//! `Proven` (VR-5).

mod common;
use common::{byte, tern, A, B};

use mycelium_core::{Meta, Node, Payload, Provenance, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, PrimRegistry, SwapEngine};
use mycelium_mlir::{ExecMode, ModeError};

/// Fresh interpreter/AOT config for one `run_mode` call (the dispatcher consumes it, since the
/// reference interpreter owns its config — see `mode::run`'s signature rationale).
fn config() -> (PrimRegistry, Box<dyn SwapEngine>) {
    (PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
}

/// The bit/trit subset the JIT compiles — the corpus shared with `jit_differential.rs`. All three
/// modes (interpreter, AOT, JIT) must agree on the observable over this set.
fn jit_subset_corpus() -> Vec<Node> {
    vec![
        Node::Const(byte(A)),
        Node::Op {
            prim: "bit.xor".into(),
            args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
        },
        Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Const(byte(A))],
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
        // M-301 trit carry arithmetic, in range.
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

fn obs(
    v: &Value,
) -> (
    mycelium_core::Repr,
    mycelium_core::Payload,
    mycelium_core::GuaranteeStrength,
) {
    (v.repr().clone(), v.payload().clone(), v.meta().guarantee())
}

/// M-727 correctness: `run_mode(Jit) == run_mode(Interpreter)` over the subset — the named-mode API
/// (not the raw entry points) is what is differential-checked. `Empirical`.
#[test]
fn jit_mode_equals_interpreter_mode_over_the_subset() {
    let mut ran_jit = false;
    for (i, node) in jit_subset_corpus().iter().enumerate() {
        let (p, s) = config();
        let reference = mycelium_mlir::run_mode(ExecMode::Interpreter, node, p, s)
            .unwrap_or_else(|e| panic!("program #{i}: interpreter mode errored: {e}"));

        let (p, s) = config();
        let jit = match mycelium_mlir::run_mode(ExecMode::Jit, node, p, s) {
            Ok(v) => v,
            Err(ModeError::ToolchainMissing(_)) => continue, // environment skip (clang absent)
            Err(e) => panic!("program #{i}: JIT mode errored: {e}"),
        };
        ran_jit = true;
        // Mutant-witness: a wrong store offset / fn signature in the JIT kernel would diverge here.
        assert_eq!(
            obs(&reference),
            obs(&jit),
            "program #{i}: interp mode ≠ jit mode"
        );
    }
    // Non-vacuity guard: if `clang` is present, the JIT must actually have run on ≥1 program (never a
    // silent vacuous pass). Same guard `threeway_codegen_differential.rs` uses.
    if clang_present() {
        assert!(ran_jit, "clang present but JIT ran on no program — vacuous");
    }
}

/// Probe whether `clang` can compile a trivial in-subset kernel right now (the JIT toolchain). Used by
/// the non-vacuity guard — `true` means the JIT genuinely should have run.
fn clang_present() -> bool {
    let trivial = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let (p, s) = config();
    !matches!(
        mycelium_mlir::run_mode(ExecMode::Jit, &trivial, p, s),
        Err(ModeError::ToolchainMissing(_))
    )
}

/// M-727 never-silent selection (G2): the AOT mode is **also** equivalent to the interpreter over the
/// subset (the three named modes agree), so a caller's explicit choice between them never changes the
/// result — only the path. This pins that the named modes are interchangeable on correctness, which is
/// what makes an *explicit* choice safe (the choice is about path/perf, never about answer).
#[test]
fn aot_mode_equals_interpreter_mode_over_the_subset() {
    for (i, node) in jit_subset_corpus().iter().enumerate() {
        let (p, s) = config();
        let reference = mycelium_mlir::run_mode(ExecMode::Interpreter, node, p, s)
            .unwrap_or_else(|e| panic!("program #{i}: interpreter mode errored: {e}"));
        let (p, s) = config();
        let aot = mycelium_mlir::run_mode(ExecMode::Aot, node, p, s)
            .unwrap_or_else(|e| panic!("program #{i}: AOT mode errored: {e}"));
        assert_eq!(
            obs(&reference),
            obs(&aot),
            "program #{i}: interp mode ≠ aot mode"
        );
    }
}

/// M-727 never-silent selection (G2): naming `ExecMode::Jit` on a node **outside** the JIT subset is
/// an **explicit refusal** (`Unsupported`), never a silent fallback to the interpreter/AOT — even
/// though those modes *could* run it. The caller must re-select deliberately; the dispatcher will not
/// quietly run a different path.
#[test]
fn jit_mode_refuses_out_of_subset_explicitly_never_falls_back() {
    // A `Dense` (embedding) value passed through `core.id` — outside the compiled bit/trit + data +
    // closure/recursion subset (`const_lane` lowers only `Binary`/`Ternary`), but a valid value the
    // interpreter evaluates. (The prior identity-*closure* example is no longer out-of-subset: the JIT
    // shares `llvm.rs`'s lowering, which since M-851 lowers closures by inlining — so a still-refused
    // *representation* is the faithful "out-of-subset" probe now.)
    let dense = Value::new(
        mycelium_core::Repr::Dense {
            dim: 3,
            dtype: mycelium_core::ScalarKind::F64,
        },
        Payload::Scalars(vec![1.0, -2.0, 0.5]),
        Meta::exact(Provenance::Root),
    )
    .expect("a well-formed dense value");
    let out_of_subset = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(dense)],
    };

    // The interpreter mode runs it fine (proving the node is valid, just not JIT-able).
    let (p, s) = config();
    let interp = mycelium_mlir::run_mode(ExecMode::Interpreter, &out_of_subset, p, s);
    assert!(
        interp.is_ok(),
        "the out-of-subset node must be valid under the interpreter, got {interp:?}"
    );

    // The JIT mode must REFUSE explicitly — never silently run the interpreter's answer. The refusal
    // is an explicit `Err` (a `Dense` const maps through `AotError::UnsupportedRepr` →
    // `ModeError::Jit`; a structural out-of-subset node would map through `UnsupportedNode` →
    // `ModeError::Unsupported`). The G2 invariant under test is precisely *no silent `Ok` fallback* —
    // any explicit refusal `Err` satisfies it; only an `Ok` (a silently-run interpreter answer) fails.
    let (p, s) = config();
    match mycelium_mlir::run_mode(ExecMode::Jit, &out_of_subset, p, s) {
        Err(ModeError::ToolchainMissing(_)) => { /* env skip — clang absent, still no silent run */
        }
        Err(ModeError::Unsupported(_)) | Err(ModeError::Jit(_)) => { /* explicit refusal — expected */
        }
        Ok(v) => panic!(
            "JIT mode must refuse the out-of-subset node explicitly, got a value {:?} \
             (silent fallback would be the G2 violation)",
            v.payload()
        ),
        Err(e) => panic!("unexpected JIT-mode error variant (expected an explicit refusal): {e}"),
    }
}

/// M-727: the named modes are inspectable (`EXPLAIN`-able) and there is no hidden/default mode — the
/// `ALL` list is exactly the three named modes, each with a stable name and a truthful
/// toolchain-availability flag. (The *type-level* guarantee that there is no `Auto`/`Default` variant
/// is enforced by `ExecMode` having no `Default` impl and no such variant; this asserts the surface.)
#[test]
fn exec_modes_are_named_and_inspectable() {
    assert_eq!(
        ExecMode::ALL,
        [ExecMode::Interpreter, ExecMode::Aot, ExecMode::Jit]
    );
    assert_eq!(ExecMode::Interpreter.name(), "interpreter");
    assert_eq!(ExecMode::Aot.name(), "aot");
    assert_eq!(ExecMode::Jit.name(), "jit");
    // The two pure-Rust trusted paths are always available; the JIT needs a toolchain — a truthful,
    // queryable flag so the caller chooses deliberately rather than discovering a missing tool.
    assert!(ExecMode::Interpreter.is_always_available());
    assert!(ExecMode::Aot.is_always_available());
    assert!(!ExecMode::Jit.is_always_available());
}
