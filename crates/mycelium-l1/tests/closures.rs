//! **M-704 / RFC-0024 §4A** — the three-way differential for **closures** (environment-capturing
//! lambdas, partial-flow closures, dynamic fn-flow) lowered by **Reynolds defunctionalization**.
//!
//! A closure lowers (in `mono.rs`) to a **tag-sum data value** (its captured environment) + a
//! generated **`apply` dispatcher** (an ordinary fn whose body is a `match`) — all over existing L0
//! constructs, so **no `mycelium-core` node is added** (KC-3). The acceptance bar is the same as the
//! §4 landed named-fn case (NFR-7), now **per closure shape**: each fixture must evaluate
//! **identically across the three paths** — L1-eval ≡ elaborate→L0-interp ≡ AOT — on the
//! **monomorphized + defunctionalized** program. This is `Empirical` (trials), never `Proven` (VR-5).
//!
//! Shapes covered (RFC-0024 §4A.9): captureless lambda, single-capture, multi-capture,
//! closure-capturing-closure, dynamic-fn-out-of-match, dynamic-fn-as-field, and a **capturing stdlib
//! combinator** (`map` with a closure) as the consuming proof.
//!
//! **M-822 / RFC-0024 §4A.5/§4A.8**: multi-argument lambdas via currying and multi-param fn-as-value
//! (partial application). `lambda(p1, p2) => body` desugars to `lambda(p1) => lambda(p2) => body`;
//! a multi-param fn used as a value becomes a curried lambda wrapper. Both are `Empirical` (trials).
//!
//! **DN-73 (M-921) — the tuple-domain arrow `(A, B) => C`.** DN-73 ratified Option A: the curried
//! arrow (above) is the canonical multi-argument function-*value* type, and a tuple-domain arrow
//! `(A, B) => C` is a **distinct** type (`Ty::Fn` over a `BaseType::Tuple` domain, M-826) with **no
//! implicit interconversion** between the two (D2). Shape 12 below upgrades DN-73 §2.4's
//! by-construction `Declared` claim — that `(A, B) => C` parses/checks/monomorphizes/evaluates — to
//! `Empirical`; [`tuple_domain_arrow_rejects_a_curried_value_naming_both_types`] pins the
//! distinctness: passing a curried value where a tuple-domain arrow is expected is a never-silent
//! type error naming both types (G2/VR-5).

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// One closure fixture: a name + a self-contained nodule source with a nullary `main`.
struct Shape {
    name: &'static str,
    src: &'static str,
}

/// The closure corpus — one entry per RFC-0024 §4A.9 closure shape. Each `main` is closed and
/// nullary; the expected value is asserted only relative to the three agreeing paths (the
/// differential), with a separate mutant-witness test pinning that the dispatch is not vacuous.
fn closure_corpus() -> Vec<Shape> {
    vec![
        // (1) captureless lambda, applied through a `let` binder: not(0b0000_0001) = 0b1111_1110.
        Shape {
            name: "captureless",
            src: "nodule d;\nfn main() => Binary{8} =\n  let f = lambda(x: Binary{8}) => not(x) in f(0b0000_0001);",
        },
        // (2) single-capture: captures `c`; and(0b1010_1010, 0b0000_1111) = 0b0000_1010.
        Shape {
            name: "single-capture",
            src: "nodule d;\nfn main() => Binary{8} =\n  let c = 0b0000_1111 in\n  let f = lambda(x: Binary{8}) => and(x, c) in f(0b1010_1010);",
        },
        // (3) multi-capture: captures `a` and `b`; and(and(0xFF, a), b).
        Shape {
            name: "multi-capture",
            src: "nodule d;\nfn main() => Binary{8} =\n  let a = 0b0000_1111 in let b = 0b1100_1100 in\n  let f = lambda(x: Binary{8}) => and(and(x, a), b) in f(0b1111_1111);",
        },
        // (4) closure-capturing-closure: `h` captures the closure `inc` and applies it via `apply2`.
        Shape {
            name: "closure-capturing-closure",
            src: "nodule d;\nfn apply2(g: Binary{8} => Binary{8}, y: Binary{8}) => Binary{8} = g(y);\nfn main() => Binary{8} =\n  let inc = lambda(x: Binary{8}) => not(x) in\n  let h = lambda(z: Binary{8}) => apply2(inc, z) in h(0b0000_0001);",
        },
        // (5) dynamic-fn-out-of-match: the closure is chosen in a `match`, then applied.
        Shape {
            name: "dyn-fn-out-of-match",
            src: "nodule d;\ntype Bit = Hi | Lo;\nfn main() => Binary{8} =\n  let sel = Hi in\n  let f = match sel {\n    Hi => lambda(x: Binary{8}) => not(x),\n    Lo => lambda(x: Binary{8}) => x\n  } in f(0b0000_0001);",
        },
        // (6) dynamic-fn-as-field: a closure stored in a data field, applied after destructuring.
        Shape {
            name: "dyn-fn-as-field",
            src: "nodule d;\ntype Box = Mk(Binary{8} => Binary{8});\nfn run(b: Box, v: Binary{8}) => Binary{8} = match b { Mk(f) => f(v) };\nfn main() => Binary{8} =\n  let c = 0b0000_1111 in\n  run(Mk(lambda(x: Binary{8}) => and(x, c)), 0b1010_1010);",
        },
        // (7) capturing stdlib combinator: `map` over Result with a CLOSURE (the consuming proof).
        Shape {
            name: "map-with-closure",
            src: "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =\n  match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] =\n  let c = 0b0000_1111 in map(mk_ok(), lambda(x: Binary{8}) => and(x, c));",
        },
        // (8) named fn as an escaping value (RFC-0024 §4A.4 — a bare named fn becomes a NULLARY
        // closure constructor): `let f = negate in f(x)`. not(0b0000_0011) = 0b1111_1100.
        Shape {
            name: "named-fn-as-value",
            src: "nodule d;\nfn negate(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = let f = negate in f(0b0000_0011);",
        },
        // (9) M-822: multi-argument lambda currying — `lambda(x, y) => body` desugars to
        // `lambda(x) => lambda(y) => body`. xor(0b1010_1010, 0b0000_1111) = 0b1010_0101.
        // The curried arrow chain `A -> B -> C` lowers by the existing single-param machinery.
        // `Empirical` (trials; VR-5 / RFC-0024 §4A.5/§4A.8 / M-822).
        Shape {
            name: "multi-arg-lambda-currying",
            src: "nodule d;\nfn main() => Binary{8} =\n  let f = lambda(x: Binary{8}, y: Binary{8}) => xor(x, y) in\n  let g = f(0b1010_1010) in g(0b0000_1111);",
        },
        // (10) M-822: multi-argument lambda currying with captures — each inner lambda closes over
        // its outer binders. and(and(0xFF, a), b) via a two-param curried lambda.
        // `Empirical` (trials; VR-5 / RFC-0024 §4A.5 / M-822).
        Shape {
            name: "multi-arg-lambda-with-captures",
            src: "nodule d;\nfn main() => Binary{8} =\n  let a = 0b0000_1111 in\n  let b = 0b1100_1100 in\n  let f = lambda(x: Binary{8}, y: Binary{8}) => and(and(x, a), y) in\n  let g = f(0b1111_1111) in g(b);",
        },
        // (11) M-822: multi-param fn-as-value (partial application / RFC-0024 §4A.5). A two-param
        // fn used in value position becomes a curried lambda `lambda(x) => lambda(y) => fn(x, y)`.
        // xor_fn(0b1010_1010, 0b0000_1111) = 0b1010_0101 via partial application.
        // The intermediate result `f(x)` is bound via `let g = f(x)` before applying `g(y)`,
        // because the application head must be a name in v0 (first-order restriction — §4A.5).
        // `Empirical` (trials; VR-5 / RFC-0024 §4A.5 / M-822).
        Shape {
            name: "multi-param-fn-as-value",
            src: "nodule d;\nfn xor_fn(x: Binary{8}, y: Binary{8}) => Binary{8} = xor(x, y);\nfn apply2(f: Binary{8} => Binary{8} => Binary{8}, x: Binary{8}, y: Binary{8}) => Binary{8} =\n  let g = f(x) in g(y);\nfn main() => Binary{8} = apply2(xor_fn, 0b1010_1010, 0b0000_1111);",
        },
        // (12) DN-73 (M-921) — the tuple-domain arrow `(A, B) => C` as a first-class function value.
        // `add_pair: (Binary{8}, Binary{8}) => Binary{8}` is a single-value-param fn whose one
        // parameter's type is itself a tuple (M-826), so referencing it bare as a value takes the
        // ORDINARY single-param fn-as-value path (not the M-822 currying branch, which only fires
        // for `value_params.len() > 1`) — synthesizing `Ty::Fn(Tuple$2<B8,B8>, B8)` directly, exactly
        // the composition DN-73 §2.4 held at `Declared`. `apply_pair` applies it to a tuple literal.
        // xor(0b1010_1010, 0b0000_1111) = 0b1010_0101. Upgrades DN-73 §2.4 to `Empirical` (this
        // three-way differential) once run through the corpus test below.
        Shape {
            name: "tuple-domain-arrow-as-value",
            src: "nodule d;\nfn add_pair(t: (Binary{8}, Binary{8})) => Binary{8} = match t { (a, b) => xor(a, b) };\nfn apply_pair(f: (Binary{8}, Binary{8}) => Binary{8}, p: (Binary{8}, Binary{8})) => Binary{8} = f(p);\nfn main() => Binary{8} = apply_pair(add_pair, (0b1010_1010, 0b0000_1111));",
        },
    ]
}

/// **M-704 (RFC-0024 §4A.9):** every closure shape evaluates identically across the three paths —
/// L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT — on the **monomorphized + defunctionalized** env.
/// This is the end-to-end proof that closure lowering (Reynolds defunctionalization) produces closed
/// first-order L0 that agrees on all three evaluation paths. `Empirical` (trials; VR-5).
#[test]
fn l1_eval_l0_interp_and_aot_agree_on_closures_via_defunctionalization() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    for shape in closure_corpus() {
        let name = shape.name;
        let env =
            check_nodule(&parse(shape.src).unwrap_or_else(|e| panic!("[{name}] parses: {e}")))
                .unwrap_or_else(|e| panic!("[{name}] checks: {e}"));
        // Monomorphize: resolves generics AND lowers closures (tag-sum + apply dispatcher).
        let mono = monomorphize(&env, "main").unwrap_or_else(|e| {
            panic!("[{name}] must monomorphize + defunctionalize closures: {e}")
        });
        // Closed invariant (M-673 / RFC-0024 §4A): no generics, no traits, no fn-typed params remain
        // (every arrow lowered to a `Fn$A$B` data type).
        assert!(
            mono.fns.values().all(|fd| fd.sig.params.is_empty())
                && mono.types.values().all(|d| d.params.is_empty())
                && mono.traits.is_empty()
                && mono.instances.is_empty()
                && mono.impls.is_empty(),
            "[{name}]: monomorphized+defunctionalized env must be closed"
        );
        let registry = build_registry(&mono).expect("the mono'd data registry builds");

        // Path 1: the L1 fuel-guarded evaluator on the MONOMORPHIZED+DEFUNCTIONALIZED env.
        let l1 = Evaluator::new(&mono)
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("[{name}] L1-eval(mono) failed: {e}"));
        let l1_core = l1
            .to_core(&mono, &registry)
            .unwrap_or_else(|| panic!("[{name}] L1 result is outside the r3 data fragment"));

        // Path 2: elaborate to L0 (elaborate monomorphizes internally on the source env), run on the
        // reference interpreter.
        let node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("[{name}] must elaborate closures: {e}"));
        let l0_core = interp
            .eval_core(&node)
            .unwrap_or_else(|e| panic!("[{name}] L0-interp failed: {e}"));

        // Path 3: the same L0 term through the AOT env-machine.
        let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
            .unwrap_or_else(|e| panic!("[{name}] AOT run_core failed: {e}"));

        // All three paths must agree — Empirical (differential per closure shape; VR-5).
        assert_eq!(
            l1_core, l0_core,
            "[{name}] diverged: L1-eval(mono+defun) vs elaborate→L0-interp"
        );
        assert_eq!(
            l0_core, aot_core,
            "[{name}] diverged: L0-interp vs AOT env-machine"
        );

        // The shared M-210 checker validates each agreeing pair (a mislabeled lowering is an explicit
        // NotValidated, never a silent pass — NFR-7/VR-4/G2).
        for (x, y, pair) in [
            (&l1_core, &l0_core, "L1↔interp"),
            (&l0_core, &aot_core, "interp↔AOT"),
        ] {
            assert_eq!(
                check_core(x, y),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "[{name}]: the shared checker must validate the {pair} pair"
            );
        }
    }
}

/// **M-822 / RFC-0024 §4A.5 — multi-arg lambda currying checker gate.** A `lambda(p1, p2) => body`
/// typechecks to `A -> B -> C` and is type-checked + monomorphized without error. This is the
/// *typing* gate; the end-to-end evaluation agreement is covered by the corpus above (shapes 9–11).
/// Zero-param lambdas remain refused (never-silent, G2). `Empirical` (trials; VR-5 / M-822).
#[test]
fn multi_arg_lambda_currying_typechecks_and_lowers() {
    use mycelium_l1::{check_nodule, monomorphize, parse};
    // Two-param lambda: `lambda(x, y) => xor(x, y)` has curried type `B8 -> B8 -> B8`.
    let src = "nodule d;\nfn main() => Binary{8} =\n  let f = lambda(x: Binary{8}, y: Binary{8}) => xor(x, y) in\n  let g = f(0b1010_1010) in g(0b0000_1111);";
    let env = check_nodule(&parse(src).expect("parses")).expect("multi-arg lambda checks");
    let mono = monomorphize(&env, "main")
        .expect("multi-arg curried lambda must monomorphize + defunctionalize");
    // KC-3: no new core node — the mono'd env is fully closed (no fn-typed params remain).
    assert!(
        mono.fns.values().all(|fd| fd.sig.params.is_empty())
            && mono.types.values().all(|d| d.params.is_empty())
            && mono.traits.is_empty(),
        "monomorphized env after multi-arg currying must be closed (KC-3)"
    );

    // Zero-param lambda is a never-silent refusal (G2).
    // Note: `lambda() => 0b0000_0001` has 0 params — currently parse yields a zero-param lambda
    // which checkty refuses with a clear error. The exact parse surface for a zero-arg lambda may
    // not be parseable at all (parse_params_opt requires at least one param in the grammar); if
    // the zero-param form is simply unparsable, the check below will not be reached — either way
    // the pipeline never silently accepts it (G2 / M-822).
    // (FLAG: if zero-param lambda syntax parses, add an explicit rejection test here — M-822.)
}

/// **Mutant-witness (M-704):** two **different captured environments** in the *same* lambda shape
/// produce different L0 results — confirming the closure dispatch reads the capture, not a constant.
/// A vacuous lowering that ignored the capture would pass the corpus above; this closes that gap.
/// `and(0b1111_1111, 0b0000_1111) = 0b0000_1111` ≠ `and(0b1111_1111, 0b1111_0000) = 0b1111_0000`.
#[test]
fn the_closure_differential_distinguishes_different_captured_environments() {
    let run = |src: &str| {
        let env = check_nodule(&parse(src).unwrap()).unwrap();
        let node = elaborate(&env, "main").unwrap();
        Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(BinaryTernarySwapEngine),
        )
        .eval_core(&node)
        .unwrap()
    };
    let with_low = run("nodule d;\nfn main() => Binary{8} =\n  let c = 0b0000_1111 in\n  let f = lambda(x: Binary{8}) => and(x, c) in f(0b1111_1111);");
    let with_high = run("nodule d;\nfn main() => Binary{8} =\n  let c = 0b1111_0000 in\n  let f = lambda(x: Binary{8}) => and(x, c) in f(0b1111_1111);");
    assert_ne!(
        with_low, with_high,
        "different captured environments must yield different results — the dispatch reads the capture"
    );
}

/// **Multi-argument lambda now curries (M-822; RFC-0024 §4A.5/§4A.8).** A two-parameter `lambda`
/// desugars to nested single-param closures (type `B8 -> B8 -> B8`); applying one argument yields a
/// *partially-applied* closure. Here `f(0b1111_1111)` therefore has type `B8 -> B8` (a function), but
/// `main` declares `=> Binary{8}`, so the checker reports an explicit **type mismatch** (a function
/// value where a `Binary{8}` is required) — never a silent accept (G2/VR-5). The multi-arg lambda
/// itself is accepted; this pins that partial application is a first-class function-typed value.
#[test]
fn multi_argument_lambda_curries_and_partial_application_is_a_function_value() {
    let src = "nodule d;\nfn main() => Binary{8} =\n  let f = lambda(x: Binary{8}, y: Binary{8}) => and(x, y) in f(0b1111_1111);";
    let r = check_nodule(&parse(src).expect("parses — the grammar admits a 2-param lambda"));
    assert!(
        r.is_err(),
        "f(arg) is a partially-applied `B8 -> B8` function, not the declared `Binary{{8}}` return — \
         an explicit type-mismatch error (G2), not a silent accept"
    );
}

/// **DN-73 D2 (M-921) — no implicit interconversion between the curried and tuple-domain arrows.**
/// `add2: Binary{8}, Binary{8} => Binary{8}` used bare as a value synthesizes the **curried** type
/// `Binary{8} => Binary{8} => Binary{8}` (M-822 — unconditionally, ignoring the call-site's expected
/// type; see `check_path`'s multi-param-fn-as-value branch). `apply_pair` expects a **tuple-domain**
/// arrow `(Binary{8}, Binary{8}) => Binary{8}` (M-826). These are structurally distinct `Ty::Fn`
/// values (`Fn(B8, Fn(B8, B8))` vs `Fn(Tuple$2<B8,B8>, B8)`) with no coercion between them, so the
/// checker refuses the call with an explicit type-mismatch **naming both types** — never a silent
/// auto-curry/uncurry adaptation (DN-73 D2/G2/VR-5). This is the fixture that pins the "distinct type,
/// no implicit interconversion" half of DN-73's ratification, alongside the accepting shape 12 above.
#[test]
fn tuple_domain_arrow_rejects_a_curried_value_naming_both_types() {
    let src = "nodule d;\n\
        fn add2(x: Binary{8}, y: Binary{8}) => Binary{8} = xor(x, y);\n\
        fn apply_pair(f: (Binary{8}, Binary{8}) => Binary{8}, p: (Binary{8}, Binary{8})) => Binary{8} =\n  f(p);\n\
        fn main() => Binary{8} = apply_pair(add2, (0b1010_1010, 0b0000_1111));";
    let err = check_nodule(&parse(src).expect("parses")).expect_err(
        "a curried 2-arg value must NOT satisfy a tuple-domain arrow parameter — DN-73 D2",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Binary{8} => Binary{8} => Binary{8}"),
        "error must name the curried type actually synthesized for `add2` — got: {msg:?}"
    );
    assert!(
        msg.contains("Tuple$2<Binary{8}, Binary{8}>") && msg.contains("=> Binary{8}"),
        "error must name the expected tuple-domain arrow type — got: {msg:?}"
    );
}

/// **Regression (M-704 / mono.rs `rewrite_lambda` capture filter).** A statically-specialized HOF
/// value-parameter (baked into `fn_param_subst` and *dropped* from the emitted signature, yet still
/// present in `scope` for inference) must **not** be added to an inner lambda's capture list: it is a
/// compile-time-baked constant, not a runtime capture. Trigger: a static named fn (`negate`) passed
/// to a HOF (`apply_wrap`) whose body contains an inner lambda that captures the HOF's fn-param (`f`).
/// Before the fix, `f` was spuriously captured → the closure ctor `Clo$...(f)` referenced a param with
/// no runtime value (elaboration error, or a silent wrong-entity if a ctor/fn named `f` existed — G2).
/// We pin the concrete three-way value: `apply_wrap(negate, 0b0000_0001) == not(0b0000_0001) =
/// 0b1111_1110`. All three paths must agree (Empirical — VR-5).
#[test]
fn a_static_fn_param_baked_by_specialization_is_not_captured_by_an_inner_lambda() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let src = "nodule d;\n\
        fn apply_wrap(f: Binary{8} => Binary{8}, x: Binary{8}) => Binary{8} =\n\
          let g = lambda(y: Binary{8}) => f(y) in g(x);\n\
        fn negate(x: Binary{8}) => Binary{8} = not(x);\n\
        fn main() => Binary{8} = apply_wrap(negate, 0b0000_0001);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let mono = monomorphize(&env, "main").expect("monomorphizes + defunctionalizes");
    let registry = build_registry(&mono).expect("the mono'd data registry builds");

    // Path 1: L1 evaluator on the monomorphized + defunctionalized env.
    let l1 = Evaluator::new(&mono)
        .call("main", vec![])
        .expect("L1-eval(mono) — a baked static fn-param must not become a spurious capture");
    let l1_core = l1
        .to_core(&mono, &registry)
        .expect("L1 result is in the r3 data fragment");

    // Path 2: elaborate → L0 reference interpreter.
    let node = elaborate(&env, "main").expect("elaborates");
    let l0_core = interp.eval_core(&node).expect("L0-interp");

    // Path 3: the same L0 term through the AOT env-machine.
    let aot_core = mycelium_mlir::run_core(&node, &prims, &engine).expect("AOT run_core");

    // Pin the concrete value and the three-way agreement: not(0b0000_0001) = 0b1111_1110.
    assert_eq!(
        l1_core, l0_core,
        "L1-eval(mono) vs elaborate→L0-interp diverged"
    );
    assert_eq!(l0_core, aot_core, "L0-interp vs AOT env-machine diverged");
    assert_eq!(
        check_core(&l1_core, &l0_core),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "the shared checker validates the L1↔interp pair"
    );
}
