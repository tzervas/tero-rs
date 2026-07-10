//! DN-53 §A.3 / M-811 — the `object` composition surface and its desugaring equivalence.
//!
//! Three-way agreement test (Empirical guarantee, DN-53 §A.3.3): the `object` surface form and its
//! hand-written lowered form (`type`+`impl`+`fn`) must produce the same `Env` — same type in
//! registry, same trait instance registered, same inherent fn registered. This differential pins
//! `observe(object Foo) == observe(lower(object Foo))` for the stage-1 checker (KC-3: zero kernel
//! growth — the `object` keyword is a pure frontend desugaring with no new checker machinery).
//!
//! Note on path naming: `object` and `via` are now active keywords and cannot appear as nodule
//! path segments (the path parser calls `ident()` which refuses keywords — G2/never-silent).
//! Fixture paths use plain identifiers only.

use mycelium_l1::{check_nodule, parse};

/// The `object` surface form (DN-53 §A.3.1).
/// Nodule path: `composition.test` — no keyword segments.
/// Constructor `;` is mandatory per DN-53 §A.2.1 / RFC-0037 D2.
const OBJECT_SRC: &str = r#"
nodule composition.test;

trait Wrap[A] {
    fn wrap(x: A) => A;
};

object BoxVal {
    Mk(Binary{8});
    impl Wrap[Binary{8}] for BoxVal {
        fn wrap(x: Binary{8}) => Binary{8} = x;
    };
    fn unbox(b: BoxVal) => Binary{8} = match b { Mk(v) => v };
};
"#;

/// The equivalent hand-written lowered form (what the `object` desugars to — DN-53 §A.3.2).
const LOWERED_SRC: &str = r#"
nodule composition.test;

trait Wrap[A] {
    fn wrap(x: A) => A;
};

type BoxVal = Mk(Binary{8});

impl Wrap[Binary{8}] for BoxVal {
    fn wrap(x: Binary{8}) => Binary{8} = x;
};

fn unbox(b: BoxVal) => Binary{8} = match b { Mk(v) => v };
"#;

/// Three-way equivalence: `object` and hand-written lowering produce the same `Env` (same type,
/// same trait instance, same inherent fn). This is the `observe(object Foo) == observe(lower(Foo))`
/// differential (DN-53 §A.3.3; Empirical guarantee — structural agreement, not a theorem).
#[test]
fn object_and_lowered_form_produce_equivalent_envs() {
    let obj_env = check_nodule(&parse(OBJECT_SRC).expect("object form parses"))
        .expect("object form type-checks");
    let low_env = check_nodule(&parse(LOWERED_SRC).expect("lowered form parses"))
        .expect("lowered form type-checks");

    // Both register `BoxVal` as a data type.
    assert!(
        obj_env.types.contains_key("BoxVal"),
        "object form: `BoxVal` type must be registered"
    );
    assert!(
        low_env.types.contains_key("BoxVal"),
        "lowered form: `BoxVal` type must be registered"
    );
    // Both have the same constructor.
    let obj_ctor = &obj_env.types["BoxVal"].ctors;
    let low_ctor = &low_env.types["BoxVal"].ctors;
    assert_eq!(
        obj_ctor.len(),
        low_ctor.len(),
        "constructor count must agree"
    );
    assert_eq!(
        obj_ctor[0].name, low_ctor[0].name,
        "constructor name must agree"
    );

    // Both register `Wrap` as a trait.
    assert!(
        obj_env.traits.contains_key("Wrap"),
        "object form: `Wrap` trait must be registered"
    );
    assert!(
        low_env.traits.contains_key("Wrap"),
        "lowered form: `Wrap` trait must be registered"
    );

    // Both register a `Wrap` instance for `BoxVal`.
    let wrap_key =
        |env: &mycelium_l1::Env| env.instances.keys().find(|(tr, _)| tr == "Wrap").cloned();
    let obj_inst = wrap_key(&obj_env).expect("object form: Wrap instance for BoxVal must exist");
    let low_inst = wrap_key(&low_env).expect("lowered form: Wrap instance for BoxVal must exist");
    assert_eq!(
        obj_inst, low_inst,
        "Wrap instance key (trait, type_head) must agree"
    );

    // Both register the `unbox` inherent fn.
    assert!(
        obj_env.fns.contains_key("unbox"),
        "object form: `unbox` fn must be registered"
    );
    assert!(
        low_env.fns.contains_key("unbox"),
        "lowered form: `unbox` fn must be registered"
    );
}

/// An `object` body with no constructor is a never-silent parse error (DN-53 §A.3.1, G2).
#[test]
fn object_with_no_constructor_is_refused() {
    let bad_src = "nodule bad;\nobject Empty { }";
    let err = mycelium_l1::parse(bad_src).unwrap_err();
    assert!(
        err.to_string()
            .contains("must have at least one constructor clause"),
        "expected constructor-missing error, got: {}",
        err
    );
}

/// A `pub object` declaration exports its type name and inherent fns to other nodules (M-662).
#[test]
fn pub_object_is_parsed_and_typechecks() {
    // Note: `pub` and `object` are both keywords and cannot appear in nodule paths.
    let src = r#"
nodule counter.test;

pub object Counter {
    Mk(Binary{8});
    fn make(v: Binary{8}) => Counter = Mk(v);
    fn get(c: Counter) => Binary{8} = match c { Mk(v) => v };
};
"#;
    let nodule = mycelium_l1::parse(src).expect("`pub object` should parse");
    let env = check_nodule(&nodule).expect("`pub object` should type-check");
    assert!(
        env.types.contains_key("Counter"),
        "Counter type must be registered"
    );
    assert!(env.fns.contains_key("make"), "make fn must be registered");
    assert!(env.fns.contains_key("get"), "get fn must be registered");
}

/// A `via` delegation clause with an out-of-range field index is a never-silent `CheckError` (G2).
#[test]
fn via_delegation_out_of_range_is_refused() {
    // `BoxVal2` has one field (at index 0); `via 1` is out of range → explicit CheckError.
    // Note: `via` is now a keyword; `bad.delegation` uses no keyword path segments.
    let bad_src = r#"
nodule bad.delegation;

trait Wrap[A] {
    fn wrap(x: A) => A;
};

object BoxVal2 {
    Mk(Binary{8});
    via 1 : Wrap;
};
"#;
    let nodule = mycelium_l1::parse(bad_src).expect("parses OK");
    let err = check_nodule(&nodule).unwrap_err();
    assert!(
        err.to_string().contains("out of range"),
        "expected out-of-range error for `via 1` on a 1-field ctor, got: {}",
        err
    );
}

/// A `via` clause targeting an unknown trait is a never-silent `CheckError` (G2).
#[test]
fn via_unknown_trait_is_refused() {
    // Note: `via` is now a keyword; `bad.unknown` uses no keyword path segments.
    let bad_src = r#"
nodule bad.unknown;

object BoxVal3 {
    Mk(Binary{8});
    via 0 : NoSuchTrait;
};
"#;
    let nodule = mycelium_l1::parse(bad_src).expect("parses OK");
    let err = check_nodule(&nodule).unwrap_err();
    assert!(
        err.to_string().contains("NoSuchTrait"),
        "expected unknown-trait error, got: {}",
        err
    );
}
