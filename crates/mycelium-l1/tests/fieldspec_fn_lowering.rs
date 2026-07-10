//! **ADR-033 / DN-74 (M-923)** — surface lowering for `Ty::Fn` record fields to the kernel
//! `FieldSpec::Fn { arity, sig }` primitive (Path A, the DN-74-ratified FLAG-1 disposition).
//!
//! **Honest scope note (new evidence found while implementing this leaf).** `crate::mono`'s closure
//! defunctionalization (RFC-0024 §4A/M-704) unconditionally rewrites *every* reachable `Ty::Fn`
//! field into a closure tag-sum before `mycelium_l1::elab::build_registry` ever runs, so through the
//! standard [`mycelium_l1::elaborate`] entry point the fixed `field_spec`'s `FieldSpec::Fn` arm is
//! never reached (defunctionalization already produces a closed, executing term for that case,
//! verified by `tests/closures.rs`'s existing three-way differential). Making `FieldSpec::Fn`
//! reachable from a real, parsed-and-checked program therefore uses
//! [`mycelium_l1::elaborate_direct`] — a narrow, additive sibling of `elaborate` that skips the
//! `mono.rs` pre-pass (added by this leaf; `elaborate`'s own behavior and the existing
//! differential corpus are unchanged). This is the honest, scoped resolution: changing `mono.rs`'s
//! defunctionalization scope itself is out of this leaf's owned files (`elab.rs`/`checkty.rs`/
//! `eval.rs` only) and is FLAGged in the PR description, not attempted here.
//!
//! Three properties, each lifted from "kernel-unit-only" (the `mycelium-core` `fn_*` property
//! suite) to "driven by a real `mycelium-l1`-parsed-and-checked program's own registry":
//! 1. distinct `Fn` signatures at the same field position produce distinct `CtorRef`s (the ADR-033
//!    §10.1 collision, closed) — via *this crate's* `field_spec`/`build_registry`, not a hand-rolled
//!    `DeclSpec`;
//! 2. a program constructing and dispatching through an `Fn`-typed record field runs **L1-eval ≡
//!    elaborate_direct→L0-interp**, with **AOT's refusal recorded explicitly** (see the AOT note
//!    below — the ADR-033 §5.1 three-way differential, closed where forms close, honestly narrowed
//!    where AOT cannot yet close);
//! 3. a cross-typed projection (a value built under one `Fn`-signature's `CtorRef`, matched against
//!    a different signature's) is an explicit, never-silent refusal (`EvalError::NonExhaustiveMatch`)
//!    on **both** L0-interp and AOT — never a silent accept, never a panic (G2). This case is
//!    necessarily adversarial (hand-assembled `Node`s): a well-typed surface program cannot express
//!    it — the checker already rejects a genuinely cross-typed `match` arm — so this demonstrates
//!    the *kernel's* rejection of an out-of-band/adversarial term, driven by this crate's own
//!    registry (not `mycelium-core`'s isolated unit tests).
//!
//! **AOT closure/refusal, recorded (further new evidence, `mycelium-mlir`, not owned by this
//! leaf).** `mycelium-mlir`'s env-machine reifies every `Construct` field to a [`CoreValue`]
//! *immediately* at construction time (`as_core`, `aot.rs`), and [`CoreValue`] is `Repr | Data`
//! only — it has no function-value variant at all (unlike the small-step L0 interpreter, which
//! keeps an unreduced `Node::Lam` field as a legitimate intermediate value and only needed the one
//! `guarantee_of_value` fix this leaf also made in `mycelium-interp`, §"cross-crate" below). So
//! constructing an `Fn`-typed field **always** refuses on AOT today with an explicit
//! [`mycelium_interp::EvalError::FunctionResult`] — never a silent wrong value, never a panic (G2)
//! — for exactly the reason ADR-033 §5.1 anticipates ("the AOT path may be partially stubbed at
//! this stage"). Closing this for AOT needs a `CoreValue`/`Datum` (or an AOT-local) representation
//! for a function-valued field — a `mycelium-core`/`mycelium-mlir` architecture change outside this
//! leaf's owned files (`mycelium-l1`'s `elab.rs`/`checkty.rs`/`eval.rs`); FLAGged in the PR
//! description as the concrete follow-on, not attempted here.
//!
//! **Cross-crate note.** This leaf also made one small, necessary fix in `mycelium-interp/src/lib.rs`
//! (`guarantee_of_value` gained a `Node::Lam { .. } => Ok(GuaranteeStrength::Exact)` arm) — without
//! it, L0-interp itself could not evaluate a `Match` whose scrutinee carries an `Fn`-typed field (the
//! guarantee-meet step had never seen a `Lam`-valued field before, since `FieldSpec::Fn` was
//! previously dead code end-to-end). `mycelium-interp` is not in this leaf's originally-scoped file
//! set; the change is minimal, mechanical, and required for the DoD — flagged prominently in the PR
//! description for review as an out-of-scope-file touch.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::{
    Alt, CtorRef, GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Value,
};
use mycelium_interp::{EvalError, Interpreter, PrimRegistry};
use mycelium_l1::elab::{build_registry, elaborate_direct};
use mycelium_l1::{check_nodule, parse, Evaluator};

/// The shared dictionary-dispatch corpus: two record types carrying an `Fn`-typed field of
/// **different** signatures at the same field position/arity (mirrors ADR-033 §5.1's
/// `MkDict_Eq8`/`MkDict_Eq16` shape), each dispatched through an ordinary named top-level function
/// (never a `lambda` literal — the bare-named-fn-as-value surface form `elab.rs`'s new `Path` arm
/// lowers, ADR-033 §2.1's "no captured environment" design).
fn dict_corpus_src() -> &'static str {
    "nodule d;\n\
     type Dict8 = MkDict8(Binary{8} => Binary{8});\n\
     type Dict16 = MkDict16(Binary{16} => Binary{16});\n\
     fn negate8(x: Binary{8}) => Binary{8} = not(x);\n\
     fn negate16(x: Binary{16}) => Binary{16} = not(x);\n\
     fn dispatch8(d: Dict8, v: Binary{8}) => Binary{8} = match d { MkDict8(f) => f(v) };\n\
     fn main() => Binary{8} = dispatch8(MkDict8(negate8), 0b0000_0001);"
}

/// Property 1: `field_spec`/`build_registry` (this crate's own lowering, not a hand-rolled
/// `mycelium-core` `DeclSpec`) gives `Dict8` and `Dict16` — same field position, same arity,
/// **different** `Fn` signatures — distinct `CtorRef`s. Closes the ADR-033 §10.1 same-arity
/// collision at the level M-923 owns: a real parsed-and-checked program's registry.
#[test]
fn distinct_fn_signatures_at_the_same_arity_get_distinct_ctor_refs() {
    let env = check_nodule(&parse(dict_corpus_src()).expect("parses")).expect("checks");
    // `build_registry` runs directly on the checked (NOT monomorphized) env — the previously-`None`
    // `field_spec` arm for `Ty::Fn` is what makes this succeed at all; before this leaf, `Dict8`/
    // `Dict16` (each carrying a `Ty::Fn` field) would have been silently skipped by `build_registry`
    // (the `continue 'types` staged-residual path), so `ctor_ref` would return `None` for both.
    let registry = build_registry(&env).expect("Ty::Fn fields now build a real registry entry");
    let dict8 = registry
        .ctor_ref("Dict8", 0)
        .expect("MkDict8 is in the registry");
    let dict16 = registry
        .ctor_ref("Dict16", 0)
        .expect("MkDict16 is in the registry");
    assert_ne!(
        dict8, dict16,
        "Dict8 (Binary{{8}} => Binary{{8}}) and Dict16 (Binary{{16}} => Binary{{16}}) share an \
         arity but differ in signature — they must NOT share a CtorRef (ADR-033 §10.1)"
    );
}

/// Property 2: the ADR-033 §5.1 program-level differential, closed **where forms close** (honest
/// per the module doc comment's AOT note). `main` constructs a `Dict8` value from an ordinary named
/// function, projects the field via `match`, and applies it:
/// - **L1-eval ≡ elaborate_direct→L0-interp** must agree on the result (`not(0b0000_0001) =
///   0b1111_1110`) — both close.
/// - **AOT** refuses **explicitly** — `mycelium-mlir`'s env-machine cannot yet represent an
///   `Fn`-typed field's value at all (`CoreValue` is `Repr | Data`, no function-value variant), so
///   it fails at `Construct`-reification time with `EvalError::FunctionResult`, never a silent wrong
///   value and never a panic. Recorded, not swept aside — the AOT closure is a follow-on (flagged in
///   the PR description), not this leaf's owned files.
#[test]
fn l1_eval_and_l0_interp_agree_through_a_fieldspec_fn_dispatch_aot_refuses_explicitly() {
    let src = dict_corpus_src();
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let registry = build_registry(&env).expect("the Fn-typed dictionary registry builds");

    // Path 1: L1-eval, directly on the checked (non-monomorphized) env — `Evaluator` needs no
    // monomorphization to run a name-based, dynamically-dispatched program (RFC-0007 §4.6's
    // superset-of-elaboration contract).
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .expect("L1-eval dispatches through the Fn-typed field");
    let l1_core = l1
        .to_core(&env, &registry)
        .expect("the result is a plain Binary{8} value, in the r3/repr fragment");

    // Path 2: elaborate_direct → L0-interp. `elaborate_direct` skips `mono.rs`'s closure
    // defunctionalization (see the module doc comment) so the `FieldSpec::Fn` primitive this leaf's
    // `field_spec` fix produces is what actually gets exercised, not a defunctionalized tag-sum.
    let node = elaborate_direct(&env, "main").expect("elaborates via the FieldSpec::Fn path");
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let l0_core = interp.eval_core(&node).expect("L0-interp dispatches");

    assert_eq!(
        l1_core, l0_core,
        "diverged: L1-eval vs elaborate_direct→L0-interp on a FieldSpec::Fn dispatch"
    );
    assert_eq!(
        check_core(&l1_core, &l0_core),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "the shared M-210 checker must validate the L1↔interp pair"
    );

    // Path 3: AOT — explicit, honest refusal (not a silent divergence, not a panic). See the module
    // doc comment's "AOT closure/refusal, recorded" note for why (`CoreValue` has no function-value
    // variant; `mycelium-mlir`'s `as_core` reifies every Construct field immediately).
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    assert_eq!(
        mycelium_mlir::run_core(&node, &prims, &engine),
        Err(EvalError::FunctionResult),
        "AOT must refuse an Fn-typed field explicitly (G2) — never a silent wrong value, and never \
         a fabricated agreement with L1-eval/L0-interp"
    );
}

/// A minimal, well-formed `Binary{n}` constant — the field payload for the adversarial `Construct`
/// in the no-match test below (the kernel does not re-validate a field's *value* shape against its
/// `FieldSpec` at `Construct`/`Match` time — only `CtorRef` identity is checked; that is exactly the
/// property under test).
fn bin_const(width: u32) -> Node {
    Node::Const(
        Value::new(
            Repr::Binary { width },
            Payload::Bits(vec![false; width as usize]),
            Meta::exact(Provenance::Root),
        )
        .expect("a well-formed all-zero Binary{n} constant"),
    )
}

/// Property 3: the never-silent cross-typed no-match. A value built under `Dict8`'s `CtorRef` is
/// matched against `Dict16`'s `CtorRef` — same arity, different `Fn` signature (ADR-033 §10.1's
/// exact collision shape, before Path A). Necessarily hand-assembled: a well-typed `.myc` program
/// cannot express a cross-typed `match` arm (the checker already refuses it), so this is the
/// kernel-level guarantee that an out-of-band/adversarial `Node::Match` — never a program the
/// surface can produce — is refused explicitly, not silently accepted, on **both** L0-interp and
/// AOT. Driven by *this crate's* registry (`dict_corpus_src`, parsed and checked), not
/// `mycelium-core`'s isolated `DeclSpec` unit tests — the DN-74 gap this property closes.
#[test]
fn cross_typed_fieldspec_fn_projection_is_a_never_silent_no_match() {
    let env = check_nodule(&parse(dict_corpus_src()).expect("parses")).expect("checks");
    let registry = build_registry(&env).expect("the Fn-typed dictionary registry builds");
    let dict8: CtorRef = registry.ctor_ref("Dict8", 0).expect("MkDict8 in registry");
    let dict16: CtorRef = registry
        .ctor_ref("Dict16", 0)
        .expect("MkDict16 in registry");
    assert_ne!(dict8, dict16, "precondition: the two CtorRefs must differ");

    // A value honestly built as `MkDict8(<some Binary{8} payload>)`.
    let dict8_value = Node::Construct {
        ctor: dict8,
        args: vec![bin_const(8)],
    };
    // Adversarially matched against `MkDict16`'s CtorRef, with no default arm — WF7's escape hatch
    // for a checker that (by construction) never emits this pairing; here we bypass the checker
    // entirely to probe the kernel's own guarantee.
    let node = Node::Match {
        scrutinee: Box::new(dict8_value),
        alts: vec![Alt::Ctor {
            ctor: dict16,
            binders: vec!["f".to_owned()],
            body: Node::Var("f".to_owned()),
        }],
        default: None,
    };

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    assert_eq!(
        interp.eval_core(&node),
        Err(EvalError::NonExhaustiveMatch),
        "L0-interp must refuse a cross-typed FieldSpec::Fn projection explicitly (G2), never \
         silently accept or panic"
    );

    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    assert_eq!(
        mycelium_mlir::run_core(&node, &prims, &engine),
        Err(EvalError::NonExhaustiveMatch),
        "AOT must agree with L0-interp: an explicit refusal, never a silent divergence (NFR-7/G2)"
    );
}
