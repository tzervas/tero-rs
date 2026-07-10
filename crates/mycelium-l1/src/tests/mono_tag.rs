//! M-967 / M-844 (executes DN-64 OQ-S; RFC-0018 §4, RFC-0019 §4.4): **per-instantiation
//! guarantee-tag context through monomorphization**. Grade-checking (`crate::grade::check_guarantees`)
//! runs once, pre-mono, over each *declaration's* own written `@ g` — mono must not silently lose,
//! merge, or upgrade that tag when it re-emits a declaration's specialized copy under its mangled
//! name. These are white-box, data-driven differentials: build a small checked `Env` with real
//! `@ g` annotations, monomorphize it, and assert the emitted `FnDecl`'s `TypeRef::guarantee`s equal
//! — never merely satisfy, exactly equal — what the source declared for that specific instantiation.

use crate::ast::Strength;
use crate::checkty::check_nodule;
use crate::checkty::Env;
use crate::mono::monomorphize_with_selections;
use crate::parse;

fn env(src: &str) -> Env {
    check_nodule(&parse(src).expect("parses")).expect("checks")
}

/// A `Cmp[A]` trait with **two** instances whose methods declare **different** return guarantees —
/// the DN-64 OQ-S scenario verbatim: "a generic function monomorphized at two different guarantee
/// levels." The two instances must sit on **different type heads** (RFC-0019 §4.5 coherence keys
/// per `(trait, head)`; two widths of the *same* head, e.g. `Binary{8}`/`Binary{4}`, would collide
/// as overlapping instances) — so `Binary{8}`'s instance is `@ Proven` and `Ternary{4}`'s is
/// `@ Empirical`. A generic `use_cmp[T: Cmp]` calls the trait method unqualified; two wrapper fns
/// instantiate it at the two heads, both reachable from one nullary `main` (mirrors
/// `two_widths_emit_two_distinct_specializations` in `tests/mono.rs`).
const CMP_TWO_GRADES: &str = "nodule d;\n\
    trait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\n\
    impl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} @ Proven = 0b00; };\n\
    impl Cmp[Ternary{4}] for Ternary{4} { fn cmp(a: Ternary{4}, b: Ternary{4}) => Binary{2} @ Empirical = 0b01; };\n\
    fn use_cmp[T: Cmp](a: T, b: T) => Binary{2} = cmp(a, b);\n\
    fn lo() => Binary{2} = use_cmp(0t0+-0, 0t000-);\n\
    fn hi() => Binary{2} = use_cmp(0b0000_0001, 0b0000_0010);\n\
    fn main() => Binary{2} = let _w = lo() in hi();\n";

/// The central differential: two call sites of the same generic (`use_cmp`), reaching two
/// **different** `Cmp` instances (`Binary{8}` vs `Ternary{4}`) whose own declarations carry
/// **different** guarantee tags. After mono, each mangled specialization must carry **exactly its
/// own** instance's tag — neither lost (both `None`), nor merged into one shared value, nor
/// upgraded past what its own `impl` wrote (VR-5).
#[test]
fn two_call_sites_reaching_different_trait_instances_keep_distinct_per_instantiation_tags() {
    let e = env(CMP_TWO_GRADES);
    let (mono, sel) = monomorphize_with_selections(&e, "main").expect("monomorphizes");

    // Both instance specializations were reached and resolved (EXPLAIN record, house rule #2).
    assert_eq!(sel.len(), 2, "both Cmp instances were statically resolved");

    let cmp8 = mono
        .fn_decl("cmp$Cmp$Binary8")
        .expect("the Binary{8} instance's cmp is emitted");
    let cmp4 = mono
        .fn_decl("cmp$Cmp$Ternary4")
        .expect("the Ternary{4} instance's cmp is emitted");

    // Not lost: neither specialization's return tag silently dropped to `None`.
    assert!(
        cmp8.sig.ret.guarantee.is_some(),
        "cmp$Cmp$Binary8 must keep its declared return tag, not lose it through mono"
    );
    assert!(
        cmp4.sig.ret.guarantee.is_some(),
        "cmp$Cmp$Ternary4 must keep its declared return tag, not lose it through mono"
    );

    // Exactly its own source's tag — not merged, not upgraded, not downgraded (equality is the
    // strongest form of "the differential shows no silent upgrade or loss").
    assert_eq!(
        cmp8.sig.ret.guarantee,
        Some(Strength::Proven),
        "cmp$Cmp$Binary8 keeps exactly its own impl's `@ Proven`"
    );
    assert_eq!(
        cmp4.sig.ret.guarantee,
        Some(Strength::Empirical),
        "cmp$Cmp$Ternary4 keeps exactly its own impl's `@ Empirical`"
    );

    // Not merged: the two instantiations' tags are distinct (a shared/merged tag would make them
    // equal — the failure mode this task exists to close).
    assert_ne!(
        cmp8.sig.ret.guarantee, cmp4.sig.ret.guarantee,
        "two instantiations at different guarantee levels must NOT bleed into a shared tag"
    );

    // VR-5, explicit: neither tag is stronger than its own source wrote (`Proven`/`Empirical` are
    // themselves what was written — `rank` equality here *is* the no-upgrade witness; a bug that
    // upgraded either to `Exact` would fail this).
    assert_eq!(
        cmp8.sig.ret.guarantee.map(Strength::rank),
        Some(Strength::Proven.rank()),
        "cmp$Cmp$Binary8's tag must not be upgraded past `Proven` (VR-5)"
    );
    assert_eq!(
        cmp4.sig.ret.guarantee.map(Strength::rank),
        Some(Strength::Empirical.rank()),
        "cmp$Cmp$Ternary4's tag must not be upgraded past `Empirical` (VR-5)"
    );
}

/// The generic wrapper `use_cmp` itself writes **no** `@ g` on its own return — mono must not
/// fabricate one. Both of its instantiations (`use_cmp$Binary8`, `use_cmp$Ternary4`) stay untagged,
/// exactly mirroring the unannotated source (never silently upgrading "no claim" into a claim).
#[test]
fn an_unannotated_generic_return_stays_untagged_after_mono() {
    let e = env(CMP_TWO_GRADES);
    let mono = crate::mono::monomorphize(&e, "main").expect("monomorphizes");
    let u8_ = mono
        .fn_decl("use_cmp$Binary8")
        .expect("use_cmp$Binary8 emitted");
    let u4 = mono
        .fn_decl("use_cmp$Ternary4")
        .expect("use_cmp$Ternary4 emitted");
    assert_eq!(
        u8_.sig.ret.guarantee, None,
        "an unannotated source return must not gain a tag through mono"
    );
    assert_eq!(
        u4.sig.ret.guarantee, None,
        "an unannotated source return must not gain a tag through mono"
    );
}

/// Data-driven differential (house-rule test layout: a table, not bespoke per-case logic): one
/// generic fn `tag_id[A](d: A @ Empirical) => A @ Empirical = d` instantiated at three different
/// widths. Each case names the width and the expected (param-tag, return-tag) pair on the emitted
/// mangled specialization — every row expects the **same** `Empirical` tag, unchanged by width,
/// demonstrating the tag threads through **per instantiation** without loss (the pre-M-967 bug
/// dropped every one of these to `None`) and without drift between instantiations.
#[test]
fn a_generic_tag_threads_unchanged_across_every_instantiation() {
    struct Case {
        width: u32,
        mangled: &'static str,
    }
    let cases = [
        Case {
            width: 4,
            mangled: "tag_id$Binary4",
        },
        Case {
            width: 8,
            mangled: "tag_id$Binary8",
        },
        Case {
            width: 16,
            mangled: "tag_id$Binary16",
        },
    ];
    // One `main` reaches all three instantiations via nested `let`s (each wrapper fn pins one
    // concrete width so mono's worklist discovers all three specializations from a single entry).
    let src = "nodule d;\n\
        fn tag_id[A](d: A @ Empirical) => A @ Empirical = d;\n\
        fn w4() => Binary{4} = tag_id(0b0001);\n\
        fn w8() => Binary{8} = tag_id(0b0000_0001);\n\
        fn w16() => Binary{16} = tag_id(0b0000_0000_0000_0001);\n\
        fn main() => Binary{16} = let _a = w4() in let _b = w8() in w16();\n";
    let e = env(src);
    let mono = crate::mono::monomorphize(&e, "main").expect("monomorphizes");

    for c in &cases {
        let fd = mono
            .fn_decl(c.mangled)
            .unwrap_or_else(|| panic!("{} must be emitted (width {})", c.mangled, c.width));
        assert_eq!(
            fd.sig.value_params[0].ty.guarantee,
            Some(Strength::Empirical),
            "{}'s param must keep its source's `@ Empirical`, not lose it (width {})",
            c.mangled,
            c.width
        );
        assert_eq!(
            fd.sig.ret.guarantee,
            Some(Strength::Empirical),
            "{}'s return must keep its source's `@ Empirical`, not lose it (width {})",
            c.mangled,
            c.width
        );
    }
}
