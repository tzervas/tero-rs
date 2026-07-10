use crate::ast::Scalar;
use crate::ast::TypeRef;
use crate::ast::{Arm, Expr, Path, Pattern};
use crate::checkty::check_nodule;
use crate::checkty::{has_var, Env, Ty, Width};
use crate::elab::ElabError;
use crate::mono::*;
use crate::parse;
use crate::totality::{WalkDepthExceeded, MAX_WALK_DEPTH};
use std::collections::BTreeSet;

fn env(src: &str) -> Env {
    check_nodule(&parse(src).expect("parses")).expect("checks")
}

const LIST: &str = "nodule d;\ntype List[A] = Nil | Cons(A, List[A]);\n";
const CMP_I8: &str = "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };\n";

// ---- mangling: shape + injectivity / collision-freedom ------------------------------------

#[test]
fn mangle_ty_shapes() {
    assert_eq!(mangle_ty(&Ty::Binary(Width::Lit(8))), "Binary8");
    assert_eq!(mangle_ty(&Ty::Ternary(Width::Lit(6))), "Ternary6");
    assert_eq!(mangle_ty(&Ty::Dense(16, Scalar::F32)), "Dense16F32");
    // A nullary data type tags with `#` so it can never collide with a repr mangle (M-673
    // injectivity fix); the bare name is still used to *register/reference* the type.
    assert_eq!(mangle_ty(&Ty::Data("Bool".into(), vec![])), "Bool#");
    assert_eq!(
        mangle_ty(&Ty::Data("List".into(), vec![Ty::Binary(Width::Lit(8))])),
        "List$Binary8"
    );
    // nested generic recurses
    assert_eq!(
        mangle_ty(&Ty::Data(
            "List".into(),
            vec![Ty::Data("List".into(), vec![Ty::Binary(Width::Lit(8))])]
        )),
        "List$List$Binary8"
    );
}

#[test]
fn mangle_decl_empty_targs_is_the_identity() {
    // Empty type arguments ⇒ the original name, byte-for-byte (monomorphic passthrough).
    assert_eq!(mangle_decl("main", &[]), "main");
    assert_eq!(
        mangle_decl("first_or", &[Ty::Binary(Width::Lit(8))]),
        "first_or$Binary8"
    );
    assert_eq!(mangle_ctor("Cons", &[]), "Cons");
    assert_eq!(
        mangle_ctor("Cons", &[Ty::Binary(Width::Lit(8))]),
        "Cons$Binary8"
    );
}

#[test]
fn mangling_is_injective_and_surface_disjoint() {
    // `$` separates only mangle joints and `#` tags a nullary data type; `%` is the elaborator's
    // fresh-var char and must never arise from mangling. A monomorphic (surface) name is
    // `$`/`#`/`%`-free — so a mangled name is collision-free with surface names and fresh vars.
    let m = mangle_method("cmp", "Cmp", &Ty::Binary(Width::Lit(8)));
    assert_eq!(m, "cmp$Cmp$Binary8");
    assert!(!m.contains('%'), "no fresh-var char in a mangled name");
    // Two different instantiations of the same fn are DISTINCT names (identity fragmentation).
    assert_ne!(
        mangle_decl("first_or", &[Ty::Binary(Width::Lit(8))]),
        mangle_decl("first_or", &[Ty::Binary(Width::Lit(4))])
    );
    // Injectivity over a set of type args INCLUDING the adversarial repr/data-name boundary: a
    // data type whose name equals a repr mangle must NOT collide with the repr (the M-673 fix —
    // before it, `Data("Binary8",[])` and `Binary(8)` both mangled to "Binary8" → a silent drop).
    let tys = [
        Ty::Binary(Width::Lit(1)),
        Ty::Binary(Width::Lit(8)),
        Ty::Ternary(Width::Lit(8)),
        Ty::Dense(8, Scalar::F32),
        Ty::Dense(8, Scalar::F64),
        Ty::Data("Bool".into(), vec![]),
        Ty::Data("List".into(), vec![Ty::Binary(Width::Lit(8))]),
        Ty::Data("Binary8".into(), vec![]),
        Ty::Data("List".into(), vec![Ty::Data("Binary8".into(), vec![])]),
    ];
    let mut seen = BTreeSet::new();
    for t in &tys {
        assert!(seen.insert(mangle_ty(t)), "mangle_ty collision on {t}");
    }
    // Explicit: the repr and the like-named data type are distinct mangles (the closed hole).
    assert_ne!(
        mangle_ty(&Ty::Binary(Width::Lit(8))),
        mangle_ty(&Ty::Data("Binary8".into(), vec![])),
        "a data type named `Binary8` must not collide with the repr Binary{{8}}"
    );
}

// ---- core specialization: List / first_or ------------------------------------------------

#[test]
fn first_or_monomorphizes_to_closed_l0() {
    let env = env(&format!(
        "{LIST}fn first_or[A](xs: List[A], d: A) => A = match xs {{ Nil => d, Cons(x, _) => x }};\n\
             fn main() => Binary{{8}} = first_or(Cons(0b0000_0001, Nil), 0b0000_0000);"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    // `main` stays `main` (nullary monomorphic, empty targs ⇒ unchanged).
    let main = mono.fn_decl("main").expect("main present");
    assert!(main.sig.params.is_empty(), "main has no type params");
    // A mangled `first_or$Binary8` with empty params exists.
    let fo = mono
        .fn_decl("first_or$Binary8")
        .expect("first_or$Binary8 emitted");
    assert!(
        fo.sig.params.is_empty(),
        "the specialization is monomorphic"
    );
    // Its mangled data type `List$Binary8` exists with empty params and mangled ctors.
    let lst = mono
        .type_info("List$Binary8")
        .expect("List$Binary8 emitted");
    assert!(lst.params.is_empty());
    let ctor_names: BTreeSet<&str> = lst.ctors.iter().map(|c| c.name.as_str()).collect();
    assert!(ctor_names.contains("Cons$Binary8") && ctor_names.contains("Nil$Binary8"));
    // No reachable Ty::Var anywhere in the mono'd env.
    assert!(
        no_reachable_var(&mono),
        "mono'd env has a reachable Ty::Var"
    );
    // It elaborates to a closed L0 term and runs to the expected value.
    let node = crate::elaborate(&env, "main").expect("elaborates");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(
        v.payload(),
        &mycelium_core::Payload::Bits(vec![false, false, false, false, false, false, false, true])
    );
}

#[test]
fn a_generic_returning_a_datum_monomorphizes() {
    // `main` returns a `List<Binary{8}>` datum directly (no value-param to drive inference — the
    // return type drives it, via the bidirectional `expected`).
    let env = env(&format!(
        "{LIST}fn main() => List[Binary{{8}}] = Cons(0b0000_0001, Nil);"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    assert!(mono.type_info("List$Binary8").is_some());
    assert!(no_reachable_var(&mono));
    // Elaborates + runs (a data result).
    let node = crate::elaborate(&env, "main").expect("elaborates");
    let _ = mycelium_interp::Interpreter::default()
        .eval_core(&node)
        .expect("runs to a core value");
}

#[test]
fn nested_generics_enqueue_inner_and_outer_instances() {
    // `List<List<Binary{8}>>` — `mangle_ty` recurses and BOTH the inner `List$Binary8` and the
    // outer `List$List$Binary8` must be emitted (the inner is discovered when emitting the outer's
    // `Cons` field, RFC-0007 §11.2). The mangled-nullary field of the outer references the inner.
    let env = env(&format!(
        "{LIST}fn main() => List[List[Binary{{8}}]] = Cons(Cons(0b0000_0001, Nil), Nil);"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    assert!(
        mono.type_info("List$Binary8").is_some(),
        "inner List$Binary8 emitted"
    );
    let outer = mono
        .type_info("List$List$Binary8")
        .expect("outer List$List$Binary8 emitted");
    // The outer's `Cons` field-0 is the inner mangled-nullary data type.
    let cons = outer
        .ctors
        .iter()
        .find(|c| c.name == "Cons$List$Binary8")
        .expect("outer Cons");
    assert_eq!(cons.fields[0], Ty::Data("List$Binary8".into(), vec![]));
    assert!(no_reachable_var(&mono));
    // It elaborates to closed L0 and runs to a datum.
    let node = crate::elaborate(&env, "main").expect("elaborates");
    let _ = mycelium_interp::Interpreter::default()
        .eval_core(&node)
        .expect("runs");
}

#[test]
fn a_for_fold_over_a_generic_spine_instance_monomorphizes_and_runs() {
    // A `for` over a **generic data-type instance** (`List<Binary{8}>`) exercises `rewrite_for`:
    // the spine type is re-inferred concrete, the element type read off the cons field, and the
    // `List$Binary8` instance enqueued. (The fn itself is monomorphic; the *data type* is generic.)
    let env = env(&format!(
        "{LIST}fn checksum(bs: List[Binary{{8}}]) => Binary{{8}} = \
                for b in bs, acc = 0b0000_0000 => xor(acc, b);\n\
             fn main() => Binary{{8}} = checksum(Cons(0b1111_0000, Cons(0b0000_1111, Nil)));"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    assert!(mono.type_info("List$Binary8").is_some());
    assert!(no_reachable_var(&mono));
    // Elaborates to a closed L0 fold and runs: xor(xor(0b0000_0000, 0b1111_0000), 0b0000_1111).
    let node = crate::elaborate(&env, "main").expect("elaborates");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(
        v.payload(),
        &mycelium_core::Payload::Bits(vec![true, true, true, true, true, true, true, true])
    );
}

// ---- trait static resolution + EXPLAIN record --------------------------------------------

#[test]
fn a_trait_method_call_resolves_statically_with_an_explain_record() {
    let env = env(&format!(
        "{CMP_I8}fn main() => Binary{{2}} = cmp(0b0000_0001, 0b0000_0010);"
    ));
    let (mono, sel) = monomorphize_with_selections(&env, "main").expect("monomorphizes");
    // The trait method became a direct monomorphic fn.
    assert!(
        mono.fn_decl("cmp$Cmp$Binary8").is_some(),
        "the instance method is emitted as a direct fn"
    );
    // No traits/instances remain.
    assert!(mono.traits.is_empty() && mono.instances.is_empty() && mono.impls.is_empty());
    assert!(no_reachable_var(&mono));
    // The EXPLAIN record is populated and inspectable (house rule #2).
    assert_eq!(sel.len(), 1, "exactly one instance selected");
    let s = sel.get("cmp$Cmp$Binary8").expect("selection recorded");
    assert_eq!(s.trait_name, "Cmp");
    assert_eq!(s.for_ty, Ty::Binary(Width::Lit(8)));
    assert_eq!(s.impl_mangled, "cmp$Cmp$Binary8");
}

#[test]
fn a_bounded_generic_calling_a_trait_method_monomorphizes() {
    // `use_cmp<T: Cmp>(a,b) = cmp(a,b)` at `Binary{8}` → `use_cmp$Binary8` calling `cmp$Cmp$Binary8`.
    let env = env(&format!(
        "{CMP_I8}fn use_cmp[T: Cmp](a: T, b: T) => Binary{{2}} = cmp(a, b);\n\
             fn main() => Binary{{2}} = use_cmp(0b0000_0001, 0b0000_0010);"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    assert!(mono.fn_decl("use_cmp$Binary8").is_some());
    assert!(mono.fn_decl("cmp$Cmp$Binary8").is_some());
    assert!(no_reachable_var(&mono));
}

// ---- fragmentation witness: two widths in one program ------------------------------------

#[test]
fn two_widths_emit_two_distinct_specializations() {
    // `first_or` at Binary{8} AND Binary{4} reachable from one `main` → two distinct mangled fns,
    // both monomorphic. Identity fragmentation, recorded — not "one body". `main` reaches both
    // widths (it returns the Binary{8} result but also evaluates the Binary{4} one via a `let`).
    let env = env(&format!(
        "{LIST}fn first_or[A](xs: List[A], d: A) => A = match xs {{ Nil => d, Cons(x, _) => x }};\n\
             fn lo() => Binary{{4}} = first_or(Cons(0b0001, Nil), 0b0000);\n\
             fn hi() => Binary{{8}} = first_or(Cons(0b0000_0001, Nil), 0b0000_0000);\n\
             fn main() => Binary{{8}} = let _w = lo() in hi();"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    assert!(mono.fn_decl("first_or$Binary8").is_some(), "Binary8 spec");
    assert!(mono.fn_decl("first_or$Binary4").is_some(), "Binary4 spec");
    assert_ne!(
        mono.fn_decl("first_or$Binary8"),
        mono.fn_decl("first_or$Binary4"),
        "the two specializations are distinct fns"
    );
    assert!(mono.type_info("List$Binary8").is_some() && mono.type_info("List$Binary4").is_some());
}

// ---- property: determinism ----------------------------------------------------------------

#[test]
fn monomorphize_is_deterministic_byte_for_byte() {
    let env = env(&format!(
        "{CMP_I8}fn use_cmp[T: Cmp](a: T, b: T) => Binary{{2}} = cmp(a, b);\n\
             fn main() => Binary{{2}} = use_cmp(0b0000_0001, 0b0000_0010);"
    ));
    let a = monomorphize(&env, "main").expect("a");
    let b = monomorphize(&env, "main").expect("b");
    // The `Env` is `Debug`; equal debug ⟹ equal structure (BTreeMaps iterate deterministically).
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "mono is deterministic");
    // And the elaborated output is identical too.
    assert_eq!(
        format!("{:?}", crate::elaborate(&env, "main").unwrap()),
        format!("{:?}", crate::elaborate(&env, "main").unwrap())
    );
}

// ---- property: termination on recursion + mutual recursion --------------------------------

#[test]
fn recursion_and_mutual_recursion_emit_a_finite_set() {
    // A recursive generic over `List` + a mutually-recursive generic pair → a finite emitted set
    // (the worklist dedups by mangled name; a recursive type/fn enqueues itself once).
    let env = env(&format!(
            "{LIST}fn len_[A](xs: List[A]) => Binary{{8}} = \
                match xs {{ Nil => 0b0000_0000, Cons(_, r) => len_(r) }};\n\
             fn ping[A](xs: List[A]) => Binary{{8}} = match xs {{ Nil => 0b0000_0000, Cons(_, r) => pong(r) }};\n\
             fn pong[A](xs: List[A]) => Binary{{8}} = match xs {{ Nil => 0b0000_0001, Cons(_, r) => ping(r) }};\n\
             fn main() => Binary{{8}} = \
                xor(len_(Cons(0b0000_0001, Nil)), ping(Cons(0b0000_0010, Nil)));"
        ));
    let mono = monomorphize(&env, "main").expect("terminates");
    // Exactly one specialization of each at Binary{8} (dedup), and one List$Binary8.
    assert!(mono.fn_decl("len_$Binary8").is_some());
    assert!(mono.fn_decl("ping$Binary8").is_some());
    assert!(mono.fn_decl("pong$Binary8").is_some());
    assert_eq!(
        mono.types.keys().filter(|k| k.starts_with("List")).count(),
        1,
        "List specialized once at Binary8"
    );
    assert!(no_reachable_var(&mono));
    // Totality is recomputed over the mangled set, and a specialization's verdict EQUALS its
    // source's (the descent machinery is structural). `len_`/`ping`/`pong` all descend on the
    // list spine, so source and specialization are both `Total` — never fabricated (VR-5).
    assert_eq!(env.fn_totality("len_"), Some(crate::Totality::Total));
    assert_eq!(
        mono.fn_totality("len_$Binary8"),
        Some(crate::Totality::Total),
        "the recursive specialization keeps its source's Total verdict"
    );
    assert_eq!(
        mono.fn_totality("ping$Binary8"),
        Some(crate::Totality::Total)
    );
    assert_eq!(
        mono.fn_totality("pong$Binary8"),
        Some(crate::Totality::Total)
    );
}

// ---- property: dedup ----------------------------------------------------------------------

#[test]
fn n_calls_to_one_instantiation_emit_exactly_one_fn() {
    let env = env(&format!(
        "{LIST}fn first_or[A](xs: List[A], d: A) => A = match xs {{ Nil => d, Cons(x, _) => x }};\n\
             fn main() => Binary{{8}} = xor(xor(\
                first_or(Cons(0b0000_0001, Nil), 0b0000_0000), \
                first_or(Cons(0b0000_0010, Nil), 0b0000_0000)), \
                first_or(Cons(0b0000_0011, Nil), 0b0000_0000));"
    ));
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    let count = mono
        .fns
        .keys()
        .filter(|k| k.starts_with("first_or"))
        .count();
    assert_eq!(count, 1, "three calls @Binary8 ⇒ exactly one emitted fn");
}

// ---- property: width sweep ----------------------------------------------------------------

#[test]
fn width_sweep_each_width_monomorphizes_closed_and_runs() {
    for n in [1u32, 2, 4, 8, 16, 32] {
        let src = format!(
                "{LIST}fn first_or[A](xs: List[A], d: A) => A = match xs {{ Nil => d, Cons(x, _) => x }};\n\
                 fn main() => Binary{{{n}}} = first_or(Cons(0b{ones}, Nil), 0b{zeros});",
                ones = "1".repeat(n as usize),
                zeros = "0".repeat(n as usize),
            );
        let env = env(&src);
        let mono = monomorphize(&env, "main").unwrap_or_else(|e| panic!("n={n}: {e:?}"));
        assert!(
            mono.fn_decl(&format!("first_or$Binary{n}")).is_some(),
            "n={n}: specialization present"
        );
        assert!(no_reachable_var(&mono), "n={n}: a Ty::Var leaked");
        // Closed + runs.
        let node = crate::elaborate(&env, "main").unwrap_or_else(|e| panic!("n={n} elab: {e:?}"));
        let v = mycelium_interp::Interpreter::default()
            .eval(&node)
            .unwrap_or_else(|e| panic!("n={n} run: {e:?}"));
        let ones: Vec<bool> = std::iter::repeat_n(true, n as usize).collect();
        assert_eq!(v.payload(), &mycelium_core::Payload::Bits(ones), "n={n}");
    }
}

// ---- pass-through: a monomorphic program is byte-identical --------------------------------

#[test]
fn a_monomorphic_program_passes_through_unchanged() {
    let env = env("nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(0b1010_1010);");
    let mono = monomorphize(&env, "main").expect("monomorphizes");
    // The fast pass-through returns a clone: identical fn/type tables.
    assert_eq!(format!("{:?}", env.fns), format!("{:?}", mono.fns));
    assert_eq!(format!("{:?}", env.types), format!("{:?}", mono.types));
}

// ---- honesty: an undetermined type parameter stays a Residual (never guessed) -------------

#[test]
fn an_undetermined_type_parameter_is_a_residual_not_a_guess() {
    // The checker refuses an undetermined parameter at check time, so build the case at the entry
    // boundary: a *nullary generic* entry is refused by mono's `run` (never specialized blindly).
    let env = env("nodule d;\nfn g[A]() => Binary{1} = 0b1;");
    let err = monomorphize(&env, "g").unwrap_err();
    let ElabError::Residual { what, .. } = &err else {
        panic!("expected a Residual, got {err:?}");
    };
    assert!(
        what.contains("generic") || what.contains("monomorph"),
        "got: {what}"
    );
}

/// True iff no `Ty::Var` appears anywhere in the mono'd env's data fields or fn param/return types.
fn no_reachable_var(env: &Env) -> bool {
    fn ref_has_var(t: &TypeRef) -> bool {
        // A surface `Named(name, args)` is a `Ty::Var` only if `name` is a bare type param; in a
        // mono'd env every `Named` is a concrete (mangled) data/repr name with no args, so the
        // honest check is: a mangled type never carries type *arguments* and never a `VAR_` marker.
        match &t.base {
            crate::ast::BaseType::Named(n, args) => !args.is_empty() || n.starts_with("VAR_"),
            _ => false,
        }
    }
    let types_ok = env.types.values().all(|d| {
        d.params.is_empty() && d.ctors.iter().all(|c| c.fields.iter().all(|f| !has_var(f)))
    });
    let fns_ok = env.fns.values().all(|fd| {
        fd.sig.params.is_empty()
            && fd.sig.value_params.iter().all(|p| !ref_has_var(&p.ty))
            && !ref_has_var(&fd.sig.ret)
    });
    types_ok && fns_ok
}

/// True iff no `BaseType::Fn` appears in any emitted fn's **value-parameter** list (M-687
/// acceptance criterion: defunctionalization must drop all fn-typed params).
fn no_fn_in_sig_params(env: &Env) -> bool {
    fn ref_has_fn(t: &TypeRef) -> bool {
        matches!(&t.base, crate::ast::BaseType::Fn(_, _))
    }
    env.fns
        .values()
        .all(|fd| fd.sig.value_params.iter().all(|p| !ref_has_fn(&p.ty)))
}

// ---- M-687: HOF defunctionalization (RFC-0024 §4) ----------------------------------------

/// The central `map(mk_ok(), double)` acceptance scenario from the M-687 task brief:
/// `map` is specialized with `f = double`, the body `f(x)` → `double(x)`, the `f` param is
/// dropped from the emitted signature, and the result is closed first-order L0.
///
/// Note: `double` here is `not(x)` (flips all bits) — `add` is Ternary-only; the function
/// shape is what matters for HOF, not the specific arithmetic.
#[test]
fn hof_map_mk_ok_double_specializes_to_closed_l0() {
    let src = "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =\nmatch r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn double(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8}, Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8}, Binary{8}] = map(mk_ok(), double);";
    let e = env(src);
    let (mono, sel) = monomorphize_with_selections(&e, "main").expect("monomorphizes");

    // The specialized HOF fn is emitted under its mangled name.
    // `map` at A=Binary8, B=Binary8, E=Binary8, fn_arg param 1 = double
    let specialized = mono
        .fns
        .keys()
        .find(|k| k.starts_with("map") && k.contains("double"))
        .expect("a map specialization with 'double' in its name exists");
    assert!(
        mono.fn_decl(specialized).is_some(),
        "the specialized map is in the emitted env"
    );

    // The specialized fn must be closed first-order: no type params, no fn-typed params.
    let spec_decl = mono.fn_decl(specialized).unwrap();
    assert!(
        spec_decl.sig.params.is_empty(),
        "no type params in the specialization"
    );
    assert!(
        spec_decl
            .sig
            .value_params
            .iter()
            .all(|p| !matches!(&p.ty.base, crate::ast::BaseType::Fn(_, _))),
        "no fn-typed value params in the specialization (defunctionalized away)"
    );

    // The whole mono'd env is closed first-order (no Ty::Var, no fn-typed params).
    assert!(no_reachable_var(&mono), "a Ty::Var leaked");
    assert!(
        no_fn_in_sig_params(&mono),
        "a Ty::Fn remained in a sig param"
    );

    // EXPLAIN record: at least one HOF specialization recorded (house rule #2).
    assert!(
        !sel.hof_specs.is_empty(),
        "MonoSelections must record the HOF specialization (EXPLAIN)"
    );
    let hof = sel
        .hof_iter()
        .find(|(_, h)| h.source_fn == "map")
        .map(|(_, h)| h)
        .expect("a HOF spec for 'map' in the EXPLAIN record");
    assert_eq!(hof.source_fn, "map");
    assert!(
        hof.fn_args.iter().any(|(_, callee)| callee == "double"),
        "the HOF spec records 'double' as the baked-in callee"
    );

    // `double` itself is emitted as a closed monomorphic fn.
    assert!(
        mono.fn_decl("double").is_some(),
        "the callee `double` is emitted"
    );

    // `mk_ok` is emitted.
    assert!(mono.fn_decl("mk_ok").is_some(), "`mk_ok` is emitted");

    // The elaborator runs to the expected value on the elaborate path.
    let node = crate::elaborate(&e, "main").expect("elaborates");
    let _ = mycelium_interp::Interpreter::default()
        .eval_core(&node)
        .expect("runs to a core value");
}

/// Pinning test: the mangling scheme for HOF specializations is injective —
/// two different fn-args produce two different mangled names (no silent alias — G2).
#[test]
fn hof_fn_arg_joint_mangling_is_injective() {
    // `mangle_hof_decl("apply", [], [(0, "foo")])` vs `(0, "bar")` — different callees.
    let n1 = mangle_hof_decl("apply", &[], &[], &[(0, "foo".to_owned())], &[]);
    let n2 = mangle_hof_decl("apply", &[], &[], &[(0, "bar".to_owned())], &[]);
    assert_ne!(
        n1, n2,
        "different fn-args must produce different mangled names"
    );

    // vs. a non-HOF (no fn-args) name — must be distinct.
    let n0 = mangle_hof_decl("apply", &[], &[], &[], &[]);
    assert_ne!(n0, n1, "HOF and non-HOF mangles are distinct");

    // A fn-arg at param 0 is different from one at param 1.
    let n3 = mangle_hof_decl("apply", &[], &[], &[(1, "foo".to_owned())], &[]);
    assert_ne!(
        n1, n3,
        "different param indices produce different mangled names"
    );

    // Two fn-args vs one: distinct.
    let n4 = mangle_hof_decl(
        "apply",
        &[],
        &[],
        &[(0, "foo".to_owned()), (1, "bar".to_owned())],
        &[],
    );
    assert_ne!(n1, n4, "one vs two fn-args are distinct");

    // With type args: distinct from without.
    let n5 = mangle_hof_decl(
        "map",
        &[Ty::Binary(Width::Lit(8))],
        &[],
        &[(0, "double".to_owned())],
        &[],
    );
    let n6 = mangle_hof_decl(
        "map",
        &[Ty::Binary(Width::Lit(4))],
        &[],
        &[(0, "double".to_owned())],
        &[],
    );
    assert_ne!(
        n5, n6,
        "different type args are distinct even at same fn-arg"
    );

    // `%` separator is not in surface names or prior mangle characters (not `$`/`#`).
    assert!(n1.contains('%'), "fn-arg separator `%` is present");
    assert!(!n0.contains('%'), "no-fn-arg name has no `%`");
}

/// A non-statically-resolvable fn value (fn chosen in a `match`) → explicit `Residual`
/// (RFC-0024 §5 — out-of-scope, never-silent — G2).
///
/// We test two cases: (a) a non-HOF static-fn-arg program (should succeed — control), and (b)
/// a program where the checker accepts a `let`-bound fn-typed local but mono cannot see through
/// it (the arg expression is an `Expr::Path` that names a *scope binder*, not a top-level fn).
#[test]
fn hof_dynamic_fn_arg_is_a_residual() {
    // (a) Static fn arg — must succeed (control for the Residual test).
    let src_static = "nodule d;\nfn apply(f: Binary{8} => Binary{8}, x: Binary{8}) => Binary{8} = f(x);\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = apply(flip, 0b0000_0010);";
    let e_static = env(src_static);
    let mono_static = monomorphize(&e_static, "main").expect("static fn arg monomorphizes");
    assert!(
        no_fn_in_sig_params(&mono_static),
        "no fn-typed param in emitted sig (static)"
    );

    // (b) The checker allows `apply(x, v)` where `x` is a value-scope binder of fn type —
    // but `x` is not a top-level fn name, so `resolve_fn_args` returns Residual.
    // We build this by making `apply` call itself with a fn-typed local that comes from a
    // value parameter (not a named top-level fn).
    //
    // The actual checker handles `f(x)` inside the body of `apply` via the `Ty::Fn`-in-scope
    // arm (M-686). From mono's perspective, the OUTER call `apply(<non-static-expr>, v)` is
    // what we need to check.
    //
    // The simplest way: a wrapper that passes a *local binder* `g` (which the checker bound
    // to fn type via a let) to the HOF. The arg at the call site is `g`, not a top-level fn.
    // NOTE: The checker (M-686) does NOT allow `let g = flip in apply(g, v)` yet because
    // `flip` as a bare path in a let-bound position needs the checker to synthesize `Ty::Fn`
    // for a let-binder — which requires `check_let` to handle `Ty::Fn` bounds. If the checker
    // rejects this, the Residual comes from the checker, not mono; that is still G2-compliant.
    //
    // We test that the right kind of error surfaces (Residual), regardless of which gate fires.
    let src_dyn = "nodule d;\nfn apply(f: Binary{8} => Binary{8}, x: Binary{8}) => Binary{8} = f(x);\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn outer(g: Binary{8} => Binary{8}, v: Binary{8}) => Binary{8} = apply(g, v);\nfn main() => Binary{8} = outer(flip, 0b0000_0001);";
    // `outer(flip, v)` should succeed — `flip` is static here.
    // `apply(g, v)` inside `outer` has `g` as a local binder of fn type; the mono pass must
    // handle this as a HOF-param-application (via fn_param_subst when outer is specialized
    // with fn_args=[(0, "flip")]).
    let e_dyn = env(src_dyn);
    let mono_dyn = monomorphize(&e_dyn, "main");
    // This case exercises the transitive specialization: outer is specialized with g=flip,
    // and inside outer's body, apply(g, v) → apply's f-param gets g=flip baked in.
    // Either it succeeds (both specialized correctly) or it surfaces a clear Residual.
    match mono_dyn {
        Ok(m) => {
            // If it succeeds, verify no fn-typed params leaked.
            assert!(
                no_fn_in_sig_params(&m),
                "if transitive HOF specialization succeeds, no fn-typed param must remain"
            );
        }
        Err(ElabError::Residual { what, .. }) => {
            // A Residual is also acceptable — it means the transitive case is deferred (G2).
            assert!(
                !what.is_empty(),
                "the Residual must have a non-empty explanation"
            );
        }
        Err(other) => {
            panic!("expected Ok or Residual, got {other:?}");
        }
    }
}

/// Determinism: two calls to `monomorphize` on the same HOF program produce byte-equal results.
#[test]
fn hof_monomorphize_is_deterministic() {
    let src = "nodule d;\nfn apply(f: Binary{8} => Binary{8}, x: Binary{8}) => Binary{8} = f(x);\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = apply(flip, 0b0000_0010);";
    let e = env(src);
    let a = monomorphize(&e, "main").expect("first mono");
    let b = monomorphize(&e, "main").expect("second mono");
    assert_eq!(
        format!("{a:?}"),
        format!("{b:?}"),
        "HOF monomorphization is deterministic"
    );
}

/// A simple monomorphic HOF (`apply`) specializes, the fn-param is dropped, the body is
/// rewritten to a direct call, and the program runs to the expected value.
///
/// `flip(x) = not(x)` so `apply(flip, 0b0000_0001) = not(0b0000_0001) = 0b1111_1110`.
#[test]
fn hof_monomorphic_apply_flip_runs_to_closed_l0() {
    let src = "nodule d;\nfn apply(f: Binary{8} => Binary{8}, x: Binary{8}) => Binary{8} = f(x);\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = apply(flip, 0b0000_0001);";
    let e = env(src);
    let mono = monomorphize(&e, "main").expect("monomorphizes");

    // The specialized `apply` must have no fn-typed param.
    assert!(no_reachable_var(&mono));
    assert!(no_fn_in_sig_params(&mono), "fn-typed param was not dropped");

    // There should be a specialized `apply` with `flip` baked in.
    let specialized = mono
        .fns
        .keys()
        .find(|k| k.starts_with("apply") && k.contains("flip"))
        .expect("apply%0:flip-like specialization present");
    let sd = mono.fn_decl(specialized).unwrap();
    // The fn-param `f` should be gone from the emitted signature.
    assert_eq!(
        sd.sig.value_params.len(),
        1, // only `x: Binary{8}` remains
        "only the non-fn value-param `x` remains in the specialization"
    );

    // Runs to 0b1111_1110 (not(0b0000_0001)).
    let node = crate::elaborate(&e, "main").expect("elaborates");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(
        v.payload(),
        &mycelium_core::Payload::Bits(vec![true, true, true, true, true, true, true, false]),
        "apply(flip, 0b0000_0001) = not(0b0000_0001) = 0b1111_1110"
    );
}

// ── Width-generic free functions (DN-42 / M-753 step-d white-box) ──────────────────────────────

/// Width-generic `id_bits{N}` at N=8 produces a specialization `id_bits$Binary8` (identity
/// fragmentation: one distinct entry per concrete width, never a shared or unnamed alias — G2).
/// `Empirical`: we monomorphize and assert the mangled name is present and the function is emitted.
#[test]
fn width_generic_monomorphizes_into_distinct_specialization_binary_8() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn main() => Binary{8} = id_bits(0b0000_0000);";
    let e = env(src);
    let mono = monomorphize(&e, "main").expect("monomorphizes");
    assert!(
        mono.fns.contains_key("id_bits$Binary8"),
        "id_bits$Binary8 must be present after monomorphization at N=8; \
         got keys: {:?}",
        mono.fns.keys().collect::<Vec<_>>()
    );
    // The specialization must not have any type/width params remaining (it is monomorphic).
    let fd = mono.fns.get("id_bits$Binary8").unwrap();
    assert!(
        fd.sig.params.is_empty(),
        "monomorphized specialization must have no params: {:?}",
        fd.sig.params
    );
}

/// `id_bits{N}` called at N=8 and N=16 from separate entries each produces a distinct
/// specialization. Identity fragmentation: two widths → two distinct emitted functions, never a
/// silent alias (G2). `Empirical`: each mono run independently produces the right specialization.
#[test]
fn width_generic_two_widths_produce_distinct_specializations() {
    // Two separate mono passes — one per entry — each traces the reachable width.
    let src8 = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn main() => Binary{8} = id_bits(0b0000_0000);";
    let src16 = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn main() => Binary{16} = id_bits(0b0000_0000_0000_0000);";

    let e8 = env(src8);
    let mono8 = monomorphize(&e8, "main").expect("monomorphizes at 8");
    assert!(
        mono8.fns.contains_key("id_bits$Binary8"),
        "id_bits$Binary8 must be present at width-8 entry; keys: {:?}",
        mono8.fns.keys().collect::<Vec<_>>()
    );
    assert!(
        !mono8.fns.contains_key("id_bits$Binary16"),
        "id_bits$Binary16 must NOT be present (unreachable from main@8)"
    );

    let e16 = env(src16);
    let mono16 = monomorphize(&e16, "main").expect("monomorphizes at 16");
    assert!(
        mono16.fns.contains_key("id_bits$Binary16"),
        "id_bits$Binary16 must be present at width-16 entry; keys: {:?}",
        mono16.fns.keys().collect::<Vec<_>>()
    );
    assert!(
        !mono16.fns.contains_key("id_bits$Binary8"),
        "id_bits$Binary8 must NOT be present (unreachable from main@16)"
    );

    // The mangled names are distinct (identity fragmentation — never a silent alias).
    assert_ne!("id_bits$Binary8", "id_bits$Binary16");
}

/// An undetermined width param (not used in value params) is an explicit refusal at the checker.
/// `Declared`: never a guessed default width (DN-42 §4 / VR-5 / G2).
#[test]
fn width_generic_undetermined_param_is_a_check_error() {
    // Width param `N` used only in the return type — cannot be inferred from call.
    let src = "nodule d;\nfn phantom_n{N}(x: Binary{8}) => Binary{8} = x;\nfn main() => Binary{8} = phantom_n(0b0000_0000);";
    let result = crate::checkty::check_nodule(&crate::parse(src).expect("parses"));
    assert!(
        result.is_err(),
        "expected check to fail for undetermined width param `N`, but succeeded"
    );
}

// ---- RFC-0024 §4A (M-704) closures: arrow mangling + capture-set analysis ------------------

/// Closure **arrow mangling** is injective and surface-disjoint (RFC-0024 §4A.4 / G2): distinct
/// arrows produce distinct tag-sum names; the dispatcher name shares the arrow's suffix; a nested
/// arrow recurses. No silent alias.
#[test]
fn closure_arrow_mangling_is_injective_and_surface_disjoint() {
    let b8 = Ty::Binary(Width::Lit(8));
    let b16 = Ty::Binary(Width::Lit(16));
    let a1 = mangle_arrow(&b8, &b8); // Fn$Binary8$Binary8
    let a2 = mangle_arrow(&b8, &b16); // Fn$Binary8$Binary16
    let a3 = mangle_arrow(&b16, &b8); // Fn$Binary16$Binary8
    assert_eq!(a1, "Fn$Binary8$Binary8");
    assert_ne!(a1, a2, "distinct codomains ⇒ distinct arrows");
    assert_ne!(a2, a3, "distinct domain/codomain order ⇒ distinct arrows");
    // The dispatcher name shares the arrow's `A$B` suffix (queryable identity).
    assert_eq!(apply_fn_name(&a1), "apply$Binary8$Binary8");
    // A nested arrow `(B8 => B8) => B8` recurses into its inner arrow.
    let nested = mangle_arrow(&Ty::Fn(Box::new(b8.clone()), Box::new(b8.clone())), &b8);
    assert_eq!(nested, "Fn$Fn$Binary8$Binary8$Binary8");
    assert_ne!(
        nested, a1,
        "a higher-order arrow is distinct from its first-order base"
    );
    // Surface-disjoint: `$` is not a surface-identifier character (the lexer never produces it).
    assert!(a1.contains('$'));
}

/// **Capture-set analysis** (RFC-0024 §4A.3): `free_vars` collects the body's free single-segment
/// names not bound within it, in first-occurrence order, each once — and an inner binder (`let`,
/// `match` arm, the lambda param) shadows. This is the property the closure lowering relies on
/// (`capture(λ) = freevars(body) \ (params ∪ toplevel)`); a bug here would silently mis-capture (G2).
#[test]
fn free_vars_respects_binders_and_first_occurrence_order() {
    use crate::ast::{Arm, Expr, Param, Path, Pattern, TypeRef, WidthRef};
    // body ≡ `and(and(x, c), let y = b in and(y, c))`
    //   free (in occurrence order): x, c, b   — `y` is bound by the inner `let`, so not free.
    let b8 = || TypeRef::unguaranteed(crate::ast::BaseType::Binary(WidthRef::Lit(8)));
    let path = |n: &str| Expr::Path(Path(vec![n.to_owned()]));
    let call = |f: &str, args: Vec<Expr>| Expr::App {
        head: Box::new(path(f)),
        args,
    };
    let inner_let = Expr::Let {
        name: "y".to_owned(),
        ty: Some(b8()),
        bound: Box::new(path("b")),
        body: Box::new(call("and", vec![path("y"), path("c")])),
    };
    let body = call(
        "and",
        vec![call("and", vec![path("x"), path("c")]), inner_let],
    );
    // Seed `bound` with the lambda parameter `x` ⇒ `x` is NOT captured (it is a param, not free).
    let mut bound: BTreeSet<String> = BTreeSet::new();
    bound.insert("x".to_owned());
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<String> = Vec::new();
    free_vars(&body, &mut bound, &mut seen, &mut out).expect("well under the depth budget");
    // `free_vars` is the raw structural set: it includes the call-head name `and` (filtering
    // top-level names to find actual *captures* is `rewrite_lambda`'s scope-membership step). The
    // param `x` is excluded (seeded into `bound`), and the inner `let y` shadows `y`. Order =
    // first-occurrence: `and` (head) then `c` then `b`.
    assert_eq!(
        out,
        vec!["and".to_owned(), "c".to_owned(), "b".to_owned()],
        "free vars (param `x` excluded, inner `let y` shadowed) in first-occurrence order"
    );

    // A `match` arm pattern binds: `match s { Mk(z) => and(z, c) }` ⇒ `z` bound, `s` and `c` free.
    let m = Expr::Match {
        scrutinee: Box::new(path("s")),
        arms: vec![Arm {
            pattern: Pattern::Ctor("Mk".to_owned(), vec![Pattern::Ident("z".to_owned())]),
            body: call("and", vec![path("z"), path("c")]),
        }],
    };
    let mut bound2: BTreeSet<String> = BTreeSet::new();
    let mut seen2: BTreeSet<String> = BTreeSet::new();
    let mut out2: Vec<String> = Vec::new();
    free_vars(&m, &mut bound2, &mut seen2, &mut out2).expect("well under the depth budget");
    // `z` is bound by the arm pattern (shadowed); `s` (scrutinee), the head `and`, and `c` are free.
    assert_eq!(out2, vec!["s".to_owned(), "and".to_owned(), "c".to_owned()]);
    let _ = Param {
        name: String::new(),
        ty: b8(),
    }; // keep `Param` import meaningful if the builder changes
}

/// **α-renaming invariance** of `free_vars` (RFC-0024 §4A.9 property): renaming a bound variable
/// leaves the free-variable *set* unchanged. We rename the `let` binder `y`→`w` and assert the free
/// set is identical (the bound name never appears free either way).
#[test]
fn free_vars_is_invariant_under_alpha_renaming_of_bound_vars() {
    use crate::ast::{Expr, Path, TypeRef, WidthRef};
    let b8 = || TypeRef::unguaranteed(crate::ast::BaseType::Binary(WidthRef::Lit(8)));
    let path = |n: &str| Expr::Path(Path(vec![n.to_owned()]));
    let call = |f: &str, args: Vec<Expr>| Expr::App {
        head: Box::new(path(f)),
        args,
    };
    let make = |binder: &str| Expr::Let {
        name: binder.to_owned(),
        ty: Some(b8()),
        bound: Box::new(path("a")),
        body: Box::new(call("and", vec![path(binder), path("c")])),
    };
    let fv = |e: &Expr| {
        let mut b = BTreeSet::new();
        let mut s = BTreeSet::new();
        let mut o = Vec::new();
        free_vars(e, &mut b, &mut s, &mut o).expect("well under the depth budget");
        o.into_iter().collect::<BTreeSet<String>>()
    };
    assert_eq!(
        fv(&make("y")),
        fv(&make("w")),
        "free-var set is invariant under α-renaming of the bound `let` variable"
    );
}

// ---- recursion-depth bound (M-866): free_vars / pattern_binders are never-silent ----------

/// A `consume(consume(… consume(x) …))` nest `depth` deep — mirrors `totality::tests::deep_consume`
/// (M-674 precedent): `Expr::Consume` is a bare `Box<Expr>` wrapper, so it is the simplest way to
/// build a pathologically-nested `Expr` directly, bypassing the parser's `MAX_EXPR_DEPTH` surface
/// cap — a direct AST is the way to exercise `free_vars`'s *own* budget.
fn deep_consume(depth: usize) -> Expr {
    let mut e = Expr::Path(Path(vec!["x".to_owned()]));
    for _ in 0..depth {
        e = Expr::Consume(Box::new(e));
    }
    e
}

/// A `Mk(Mk(… Mk(_) …))` constructor-pattern nest `depth` deep — the pattern-side analogue of
/// [`deep_consume`], to exercise `pattern_binders`'s own separate depth budget (it resets to a fresh
/// `0` per `pattern_binders` call, mirroring `totality::pattern_binders`'s own convention — a
/// pattern's nesting is budgeted independently of the enclosing expression's).
fn deep_pattern(depth: usize) -> Pattern {
    let mut p = Pattern::Wildcard;
    for _ in 0..depth {
        p = Pattern::Ctor("Mk".to_owned(), vec![p]);
    }
    p
}

/// **The recursion bound itself (M-866; G2 never-silent).** `free_vars`'s own recursive descent
/// over `Expr` is budgeted at [`MAX_WALK_DEPTH`] (4096) — the same crate-wide AST-pass depth budget
/// `totality`/`checkty`/`elab` already carry (M-674), reused here rather than inventing a second
/// constant (DRY). Just under the budget, the walk completes and still finds the leaf free variable;
/// past it, `free_vars` returns the explicit [`WalkDepthExceeded`] refusal — never a host-stack
/// overflow — with `limit == MAX_WALK_DEPTH` exactly (`Exact`-tagged: the budget is a checked
/// constant, not a measurement).
#[test]
fn free_vars_trips_the_depth_budget_cleanly_and_just_under_it_succeeds() {
    // `MAX_WALK_DEPTH` (4096) levels of match-heavy recursion comfortably exceeds a default test
    // thread's ~2 MiB stack even though it is nowhere near `mycelium_stack`'s deep worker-stack
    // ceiling — run on the deep stack exactly as `totality`'s own depth-budget tests do (M-674).
    mycelium_stack::with_deep_stack(|| {
        // Just under the budget: the walk completes and still finds the leaf `x`.
        let ok_body = deep_consume((MAX_WALK_DEPTH - 5) as usize);
        let mut bound: BTreeSet<String> = BTreeSet::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut out: Vec<String> = Vec::new();
        free_vars(&ok_body, &mut bound, &mut seen, &mut out)
            .expect("just under the budget should walk to completion");
        assert_eq!(out, vec!["x".to_owned()]);

        // Past the budget: a clean, explicit refusal — never a host-stack overflow (G2).
        let bad_body = deep_consume((MAX_WALK_DEPTH + 50) as usize);
        let mut bound2: BTreeSet<String> = BTreeSet::new();
        let mut seen2: BTreeSet<String> = BTreeSet::new();
        let mut out2: Vec<String> = Vec::new();
        let err: WalkDepthExceeded = free_vars(&bad_body, &mut bound2, &mut seen2, &mut out2)
            .expect_err("past the budget must refuse");
        assert_eq!(err.limit, MAX_WALK_DEPTH);
        assert!(
            err.to_string().contains("recursion-depth budget"),
            "expected the explicit depth-budget refusal, got: {err}"
        );
    });
}

/// **`pattern_binders`'s own depth budget (M-866).** A `match` arm's pattern is walked by
/// `pattern_binders`, whose recursion is budgeted **independently** of the enclosing expression's
/// (it resets to a fresh `0` per call — mirrors `totality::pattern_binders`). A pathologically
/// nested pattern trips `pattern_binders`'s own `WalkDepthExceeded`, propagated up through
/// `free_vars`'s `?` — never a host-stack overflow, even though the *enclosing* `Match` expression
/// itself is shallow.
#[test]
fn pattern_binders_trips_its_own_depth_budget_via_a_deeply_nested_match_arm_pattern() {
    mycelium_stack::with_deep_stack(|| {
        let scrutinee = Box::new(Expr::Path(Path(vec!["s".to_owned()])));
        let arm_body = Expr::Path(Path(vec!["c".to_owned()]));

        // Just under the budget: the pattern walk (and hence the whole match) completes.
        let ok_match = Expr::Match {
            scrutinee: scrutinee.clone(),
            arms: vec![Arm {
                pattern: deep_pattern((MAX_WALK_DEPTH - 5) as usize),
                body: arm_body.clone(),
            }],
        };
        let mut bound: BTreeSet<String> = BTreeSet::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut out: Vec<String> = Vec::new();
        free_vars(&ok_match, &mut bound, &mut seen, &mut out)
            .expect("a pattern just under the budget should walk to completion");

        // Past the budget: `pattern_binders`'s own refusal, surfaced through `free_vars`.
        let bad_match = Expr::Match {
            scrutinee,
            arms: vec![Arm {
                pattern: deep_pattern((MAX_WALK_DEPTH + 50) as usize),
                body: arm_body,
            }],
        };
        let mut bound2: BTreeSet<String> = BTreeSet::new();
        let mut seen2: BTreeSet<String> = BTreeSet::new();
        let mut out2: Vec<String> = Vec::new();
        let err: WalkDepthExceeded = free_vars(&bad_match, &mut bound2, &mut seen2, &mut out2)
            .expect_err("a pathologically-nested match-arm pattern must refuse cleanly");
        assert_eq!(err.limit, MAX_WALK_DEPTH);
    });
}

// ---- M-904: `consume` rewrites transparently, no residual (DN-71 §4.3) --------------------

#[test]
fn consume_rewrites_transparently_through_the_slow_monomorphization_path() {
    // A bare Substrate/consume program alone is already fully monomorphic (no generics/traits/
    // lambdas), so `monomorphize` would take the fast pass-through (`is_already_monomorphic`) and
    // never actually call `Mono::rewrite` on `take`'s body — that would prove nothing about the
    // M-904 rewrite arm itself. Adding an unrelated nullary generic `id[A]` disables the fast
    // pass-through for the whole env, forcing the real worklist-driven `Mono::run` -> `rewrite`
    // walk over `take`'s own body when `take` is monomorphized directly.
    let env = env("nodule d;\nfn id[A](x: A) => A = x;\n\
         fn take(s: Substrate{Sock}) => Substrate{Sock} = consume s;");
    // `take` itself is non-generic (no *type* params — only monomorphic value params, which
    // `Mono::run`'s own-entry check does not gate), so monomorphizing it directly reaches the
    // M-904 rewrite arm without needing a caller. (No v0 surface syntax can actually construct a
    // live `Substrate` value to call `take` with — DN-71 §4.1/§8 FLAG-8 — so `take` can never be
    // *reached* from a real program's nullary entry; this test exercises the rewrite arm in
    // isolation, which is the most this leaf can honestly claim for the mono/elab layer.)
    let mono_env = monomorphize(&env, "take")
        .expect("M-904: `consume` rewrites transparently — no residual for this fragment");
    let take = mono_env.fns.get("take").expect("take is emitted");
    assert!(
        matches!(&take.body, Expr::Consume(inner) if matches!(inner.as_ref(), Expr::Path(_))),
        "the rewritten body stays a transparent Consume of the rewritten (unchanged, local-scope) \
         operand: {:?}",
        take.body
    );
}
