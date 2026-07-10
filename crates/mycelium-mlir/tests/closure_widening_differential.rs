//! M-851 (epic E25-1) — direct-LLVM **closure-ABI widening** differential: the narrow packed-`i64`
//! `Binary{8}` closure ABI (M-378) is replaced by a **specialize-at-application (inlining)** lowering,
//! so closures over **any repr/width** (`Binary{w}`, `Ternary{m}`), **curried application**, and
//! **returned closures** lower natively (DN-15 §7.1 — the "uniform … any repr/width" widening;
//! RFC-0004 §11.7; ADR-034). A `Lam` builds a suspended closure value (a free-var snapshot); an `App`
//! inlines the body with the param bound to the (concrete-shape) argument — so every shape is
//! statically resolved at the call site, never a fixed wire ABI nor a guessed width.
//!
//! This is the **interp ≡ direct-LLVM** differential that backs the widening's `Empirical` tag: each
//! terminating closure-heavy program is value-checked against the M-110 reference interpreter through
//! the shared **M-210** checker (`ObservationalEquiv`), and a divergence-distinguishing test guards
//! against a vacuous pass. The still-refused shape (a closure-valued *program result* — a closure is
//! not printable by the read-back protocol; DN-15 §7.4) is pinned as an honest `UnsupportedNode`
//! (G2/VR-5 — never a silent mis-lowering).
//!
//! **MLIR-dialect leg.** The `dialect::native` path refuses closures (`Lam`/`App` →
//! `DialectError::Unsupported`), so for this corpus the third differential edge is an **honest
//! refusal**, not a vacuous skip — exactly as the recursion trampoline differential is two-way
//! (interp ≡ direct-LLVM). The element-wise three-way leg with its `ran_mlir` non-vacuity guard lives
//! in `tests/threeway_differential.rs`. **M-858:** this paragraph used to be an unchecked doc-comment
//! claim; `tests/unified_threeway_differential.rs::dialect_honestly_refuses_closures_and_recursion`
//! now actually calls `mlir_compile_and_run` on an `App`/`Lam` program and asserts the
//! `DialectError::Unsupported` refusal, turning the claim into a checked fact.
//!
//! Guarantee tag: **Empirical** — hand-written textual LLVM IR with a *checked* empirical basis (this
//! differential + the `cargo-mutants` witness of the closure inlining fns), never `Proven` (VR-5: no
//! machine-checked refinement theorem for the emitted IR). Skips gracefully when `llc`/`clang` are
//! absent (`AotError::ToolchainMissing` — the house idiom).

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::AotError;
use mycelium_numerics::Certificate;

// ─── helpers ──────────────────────────────────────────────────────────────────────────────────

/// A `Binary{w}` value from a bit vector (any width — the point of the widening).
fn bits(v: Vec<bool>) -> Value {
    let width = v.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(v),
        Meta::exact(Provenance::Root),
    )
    .expect("binary value")
}

/// A `Ternary{m}` value from a trit vector (closures over ternary — new under the widening).
fn trits(v: Vec<Trit>) -> Value {
    let m = v.len() as u32;
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(v),
        Meta::exact(Provenance::Root),
    )
    .expect("ternary value")
}

fn lam(param: &str, body: Node) -> Node {
    Node::Lam {
        param: param.to_owned(),
        body: Box::new(body),
    }
}
fn app(f: Node, a: Node) -> Node {
    Node::App {
        func: Box::new(f),
        arg: Box::new(a),
    }
}
fn var(x: &str) -> Node {
    Node::Var(x.to_owned())
}
fn let_(id: &str, bound: Node, body: Node) -> Node {
    Node::Let {
        id: id.to_owned(),
        bound: Box::new(bound),
        body: Box::new(body),
    }
}
fn op2(prim: &str, a: Node, b: Node) -> Node {
    Node::Op {
        prim: prim.into(),
        args: vec![a, b],
    }
}
fn op1(prim: &str, a: Node) -> Node {
    Node::Op {
        prim: prim.into(),
        args: vec![a],
    }
}

fn interp_eval(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate the closure-widening corpus")
}

/// Assert interp ≡ direct-LLVM on a terminating closure program: observable triples equal **and** the
/// shared M-210 checker validates the pair at `Exact`. Skips when the toolchain is absent.
fn assert_interp_eq_native(label: &str, prog: &Node) {
    let native = match mycelium_mlir::compile_and_run(prog) {
        Ok(v) => v,
        Err(AotError::ToolchainMissing(_)) => return, // env skip — house idiom
        Err(e) => panic!("{label}: direct-LLVM closure path errored: {e}"),
    };
    let interp = interp_eval(prog);
    assert_eq!(
        (interp.repr(), interp.payload(), interp.meta().guarantee()),
        (native.repr(), native.payload(), native.meta().guarantee()),
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
}

// ─── any-width binary closures (was: only Binary{8}) ───────────────────────────────────────────

/// Identity over **Binary{4}** — the narrow ABI refused any width but 8; the widening lowers it.
#[test]
fn identity_over_binary4() {
    let prog = app(
        lam("x", var("x")),
        Node::Const(bits(vec![true, false, true, true])),
    );
    assert_interp_eq_native("identity Binary{4}", &prog);
}

/// Identity over **Binary{16}** — a wider lane than the old packed-`i64`-friendly 8.
#[test]
fn identity_over_binary16() {
    let v: Vec<bool> = (0..16).map(|i| i % 3 == 0).collect();
    let prog = app(lam("x", var("x")), Node::Const(bits(v)));
    assert_interp_eq_native("identity Binary{16}", &prog);
}

/// Capture + op over **Binary{12}**: `let y=B12 in (λx. x ⊕ y) A12` — a wide capture and a wide arg.
#[test]
fn wide_capture_and_arg_binary12() {
    let a: Vec<bool> = (0..12).map(|i| i % 2 == 0).collect();
    let b: Vec<bool> = (0..12).map(|i| i % 5 == 0).collect();
    let prog = let_(
        "y",
        Node::Const(bits(b)),
        app(
            lam("x", op2("bit.xor", var("x"), var("y"))),
            Node::Const(bits(a)),
        ),
    );
    assert_interp_eq_native("wide capture+arg Binary{12}", &prog);
}

// ─── ternary closures (new under the widening) ─────────────────────────────────────────────────

/// Identity over **Ternary{3}** — a closure over a balanced-ternary value (the narrow ABI was
/// Binary-only). Mutant-witness: a wrong box tag / sext would corrupt the trits.
#[test]
fn identity_over_ternary() {
    let prog = app(
        lam("x", var("x")),
        Node::Const(trits(vec![Trit::Pos, Trit::Zero, Trit::Neg])),
    );
    assert_interp_eq_native("identity Ternary{3}", &prog);
}

/// `trit.neg` inside a closure body over **Ternary{4}**: `(λx. trit.neg x) T` — a ternary lane crosses
/// the boundary, is negated digit-wise, and crosses back. Exercises a non-identity ternary closure.
#[test]
fn ternary_neg_in_closure_body() {
    let prog = app(
        lam("x", op1("trit.neg", var("x"))),
        Node::Const(trits(vec![Trit::Pos, Trit::Zero, Trit::Neg, Trit::Pos])),
    );
    assert_interp_eq_native("trit.neg in closure Ternary{4}", &prog);
}

/// Capture a ternary value, return it via the identity-ish closure: `let y=T in (λx. trit.neg y) z`
/// (the body ignores `x`, negates the captured ternary `y`). A ternary **capture** across the boundary.
#[test]
fn ternary_capture() {
    let prog = let_(
        "y",
        Node::Const(trits(vec![Trit::Neg, Trit::Pos, Trit::Zero])),
        app(
            lam("x", op1("trit.neg", var("y"))),
            Node::Const(trits(vec![Trit::Zero, Trit::Zero, Trit::Zero])),
        ),
    );
    assert_interp_eq_native("ternary capture", &prog);
}

// ─── currying / returned closures (new under the widening) ─────────────────────────────────────

/// Curried application `((λx. λy. x ⊕ y) A) B → A ⊕ B`: the outer closure **returns a closure**
/// capturing `x`, which is then applied to `B`. The narrow ABI refused a closure-as-result; the
/// widening lowers it. Mutant-witness: a wrong capture-box order would diverge.
#[test]
fn curried_xor() {
    let a = vec![true, false, true, true, false, false, true, false];
    let b = vec![false, false, true, false, true, false, true, true];
    let curry = lam("x", lam("y", op2("bit.xor", var("x"), var("y"))));
    let prog = app(app(curry, Node::Const(bits(a))), Node::Const(bits(b)));
    assert_interp_eq_native("curried xor", &prog);
}

/// Curried over a **non-8 width**: `((λx. λy. x ∧ y) A6) B6 → A6 ∧ B6`. Currying *and* a wide lane.
#[test]
fn curried_and_binary6() {
    let a = vec![true, true, false, true, false, true];
    let b = vec![true, false, false, true, true, true];
    let curry = lam("x", lam("y", op2("bit.and", var("x"), var("y"))));
    let prog = app(app(curry, Node::Const(bits(a))), Node::Const(bits(b)));
    assert_interp_eq_native("curried and Binary{6}", &prog);
}

/// A closure **bound, then applied later** — the returned closure flows through a `let`:
/// `let f = (λx. λy. x ⊕ y) A in f B → A ⊕ B`. The intermediate closure is a first-class value.
#[test]
fn returned_closure_through_let() {
    let a = vec![true, false, true, true, false, false, true, false];
    let b = vec![false, false, true, false, true, false, true, true];
    let curry = lam("x", lam("y", op2("bit.xor", var("x"), var("y"))));
    let prog = let_(
        "f",
        app(curry, Node::Const(bits(a))),
        app(var("f"), Node::Const(bits(b))),
    );
    assert_interp_eq_native("returned closure through let", &prog);
}

/// A **3-argument curry** `(((λa. λb. λc. (a ⊕ b) ∧ c) A) B) C` — two nested returned closures, fully
/// applied. Stresses the chained-currying return-shape threading (each App resolves the next).
#[test]
fn triple_curry() {
    let a = vec![true, false, true, true, false, false, true, false];
    let b = vec![false, false, true, false, true, false, true, true];
    let c = vec![true; 8];
    let curry = lam(
        "a",
        lam(
            "b",
            lam(
                "c",
                op2("bit.and", op2("bit.xor", var("a"), var("b")), var("c")),
            ),
        ),
    );
    let prog = app(
        app(app(curry, Node::Const(bits(a))), Node::Const(bits(b))),
        Node::Const(bits(c)),
    );
    assert_interp_eq_native("triple curry", &prog);
}

// ─── non-vacuity guard ─────────────────────────────────────────────────────────────────────────

/// The widened closure path **discriminates**: two curried programs with the same shape but different
/// final ops (`⊕` vs `∧`) produce different results, and the shared checker reports the divergence.
/// Guards against a vacuous pass of the equivalence tests above.
#[test]
fn widened_closure_path_distinguishes_programs() {
    let a = vec![true, false, true, true, false, false, true, false];
    let b = vec![false, false, true, false, true, false, true, true];
    let mk = |prim: &str| {
        let curry = lam("x", lam("y", op2(prim, var("x"), var("y"))));
        app(
            app(curry, Node::Const(bits(a.clone()))),
            Node::Const(bits(b.clone())),
        )
    };
    let (x, y) = match (
        mycelium_mlir::compile_and_run(&mk("bit.xor")),
        mycelium_mlir::compile_and_run(&mk("bit.and")),
    ) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(AotError::ToolchainMissing(_)), _) | (_, Err(AotError::ToolchainMissing(_))) => return,
        (x, y) => panic!("widened closure path errored: {x:?} / {y:?}"),
    };
    assert_ne!(
        (x.repr(), x.payload()),
        (y.repr(), y.payload()),
        "curried A⊕B and A∧B must differ"
    );
    assert!(
        matches!(
            check(
                &x,
                &y,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the divergent curried pair"
    );
}

// ─── honest still-refused boundaries (G2/VR-5) ─────────────────────────────────────────────────

/// A closure-valued **program result** stays an explicit refusal (a closure is not printable by the
/// read-back protocol): a bare `λx.x`, and a curry left **partially applied** (`(λx.λy.x ⊕ y) A`,
/// whose result is the inner closure). Currying *itself* lowers — only a closure on the *result* is
/// refused (DN-15 §7.4; the boundary moved, it did not vanish).
#[test]
fn closure_valued_program_result_is_refused() {
    let bare = lam("x", var("x"));
    let partial = app(
        lam("x", lam("y", op2("bit.xor", var("x"), var("y")))),
        Node::Const(bits(vec![true; 8])),
    );
    for (label, prog) in [("bare lam", &bare), ("partial curry", &partial)] {
        match mycelium_mlir::compile_and_run(prog) {
            Err(AotError::UnsupportedNode(_)) => { /* expected explicit refusal */ }
            Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
            Ok(v) => panic!(
                "{label}: a closure result must be refused; got {:?}",
                v.payload()
            ),
            Err(e) => panic!("{label}: unexpected error variant: {e}"),
        }
    }
}

/// A **nested capture of the outer param** `(λx. (λw. w ⊕ x) x) A → A ⊕ A = 0`: the inner closure
/// captures the *outer* parameter `x` and is applied to it. The **specialize-at-application** lowering
/// (M-851) resolves this fully — the call-site argument `A` pins `x`, the inner closure inlines with
/// `w ← x`, so the shape flows in directly (no per-closure type inference, no guessed width). interp ≡
/// direct-LLVM. (Mutant-witness: a wrong capture binding would diverge from `A ⊕ A = 0`.)
#[test]
fn nested_capture_of_outer_param() {
    let prog = app(
        lam(
            "x",
            app(lam("w", op2("bit.xor", var("w"), var("x"))), var("x")),
        ),
        Node::Const(bits(vec![
            true, false, true, true, false, false, true, false,
        ])),
    );
    assert_interp_eq_native("nested capture of outer param", &prog);
    // `A ⊕ A` is the all-zero byte — pin the actual value, not just self-consistency.
    if let Ok(v) = mycelium_mlir::compile_and_run(&prog) {
        assert_eq!(
            v.payload(),
            &Payload::Bits(vec![false; 8]),
            "A ⊕ A must be 0"
        );
    }
}
