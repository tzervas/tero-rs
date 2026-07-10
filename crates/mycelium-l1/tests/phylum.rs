//! **Phylum + cross-nodule model** integration tests (M-662; RFC-0006 §4.3; DN-06; RFC-0019 §4.5).
//!
//! Cover the whole additive layer end-to-end through the public API:
//! - a phylum header + multiple `nodule` blocks parse + type-check;
//! - cross-`use` of a `pub` fn/type across nodules type-checks (and the private control refuses);
//! - the orphan rule generalized **phylum-wide** (cross-nodule `impl` accepts; the both-outside
//!   control still orphan-rejects);
//! - **never-silent** name resolution: unknown / private / duplicate / glob-vs-glob-ambiguous are
//!   explicit `CheckError`s, never silent winners (G2), with disambiguation controls;
//! - `pub` + phylum-header + `use`(specific+glob) **round-trip** through `expand_phylum_to_source`;
//! - backward-compat: a bare nodule is a phylum-of-one (`check_nodule` ≡ phylum-of-one).
//!
//! Every "the check fires" test is paired with a **control** that proves the check is not vacuous
//! (the same shape, minus the violation, is accepted). Honesty: the coherence/orphan rule is
//! `Declared` (RFC-0019 §4.5) — these tests pin the never-silent *behavior*, not a proof.

use mycelium_l1::{check_phylum, expand_phylum_to_source, parse_phylum, CheckError, Phylum};

/// Parse + check a phylum source, returning the per-nodule envs.
fn check(src: &str) -> Result<mycelium_l1::PhylumEnv, CheckError> {
    let ph = parse_phylum(src).expect("parses as a phylum");
    check_phylum(&ph)
}

/// Parse + check, expecting a never-silent `CheckError`; returns its message.
fn check_err(src: &str) -> String {
    let ph = parse_phylum(src).expect("parses as a phylum");
    check_phylum(&ph).expect_err("must fail to check").message
}

// ---------------------------------------------------------------------------------------------
// Parse: header + multiple nodules.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_phylum_header_with_two_nodules_parses_into_two_nodule_blocks() {
    let ph = parse_phylum(
        "phylum app.core\nnodule a;\nfn f() => Binary{8} = 0b0000_0000;\nnodule b;\nfn g() => Binary{8} = 0b0000_0001;",
    )
    .expect("parses");
    assert_eq!(
        ph.path.as_ref().map(|p| p.0.clone()),
        Some(vec!["app".to_owned(), "core".to_owned()])
    );
    assert_eq!(ph.nodules.len(), 2);
    assert_eq!(ph.nodules[0].path.0, vec!["a".to_owned()]);
    assert_eq!(ph.nodules[1].path.0, vec!["b".to_owned()]);
    // Each nodule's items end where the next `nodule` begins (M-662).
    assert_eq!(ph.nodules[0].items.len(), 1);
    assert_eq!(ph.nodules[1].items.len(), 1);
}

#[test]
fn a_header_less_single_nodule_is_a_phylum_of_one() {
    // `parse_phylum` is a strict superset of `parse`: a bare nodule parses to `path: None`.
    let ph = parse_phylum("nodule solo;\nfn f() => Binary{8} = 0b0;").expect("parses");
    assert!(ph.path.is_none(), "no header ⇒ phylum-of-one (path None)");
    assert_eq!(ph.nodules.len(), 1);
}

#[test]
fn a_phylum_header_with_no_nodule_is_an_explicit_error() {
    // A phylum groups nodules — a header alone names a grouping with nothing in it (G2).
    let err = parse_phylum("phylum app\n").expect_err("must reject");
    assert!(
        err.message.contains("at least one `nodule`"),
        "got: {}",
        err.message
    );
}

// ---------------------------------------------------------------------------------------------
// Cross-`use` accept: a `pub` fn / `pub` type imported by a sibling nodule type-checks.
// ---------------------------------------------------------------------------------------------

#[test]
fn nodule_b_uses_a_pub_fn_from_nodule_a_and_type_checks() {
    // Cross-`use` accept (the headline deliverable): `a` exports `pub fn id`; `b` imports it and
    // calls it. The call types because the imported `pub` signature is in `b`'s checking registry.
    let penv = check(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.id;\nfn use_it(y: Binary{8}) => Binary{8} = id(y);",
    )
    .expect("a cross-`use` of a pub fn type-checks");
    // `b`'s env resolves `use_it` and sees the imported `id`.
    let b = penv.nodule(&path(&["b"])).expect("nodule b present");
    assert!(b.fn_decl("use_it").is_some());
    assert!(
        b.fn_decl("id").is_some(),
        "imported fn is visible in b's env"
    );
}

#[test]
fn nodule_b_uses_a_pub_type_from_nodule_a_and_type_checks() {
    // A `pub type` crosses too: `b` imports `Flag` and uses it as a parameter type + constructor.
    check(
        "phylum p\nnodule a;\npub type Flag = Off | On;\nnodule b;\nuse a.Flag;\nfn f(x: Flag) => Flag = On;",
    )
    .expect("a cross-`use` of a pub type type-checks");
}

#[test]
fn intra_nodule_a_private_name_is_still_visible_within_its_own_nodule() {
    // `pub` gates ONLY cross-nodule visibility: a *private* fn is fully usable inside its own nodule.
    check(
        "phylum p\nnodule a;\nfn helper(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = helper(0b0000_0001);",
    )
    .expect("a private fn is visible intra-nodule");
}

// ---------------------------------------------------------------------------------------------
// `use` of a non-`pub` (private) name → never-silent CheckError; control: pub resolves.
// ---------------------------------------------------------------------------------------------

#[test]
fn use_of_a_private_name_is_a_never_silent_error_distinguishing_private_from_absent() {
    // `a.secret` exists but is NOT pub ⇒ the refusal must say "exists but is not `pub`" (honest +
    // helpful — distinct from "no such name"), never a silent skip (G2).
    let msg = check_err(
        "phylum p\nnodule a;\nfn secret(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.secret;\nfn f(y: Binary{8}) => Binary{8} = secret(y);",
    );
    assert!(
        msg.contains("not `pub`") && msg.contains("secret"),
        "a private import must refuse as exists-but-private, got: {msg}"
    );
}

#[test]
fn control_the_same_name_made_pub_resolves() {
    // The non-vacuous control for the private-import refusal: marking `secret` `pub` makes the very
    // same program check. (Proves the refusal above is about `pub`-ness, not the name/shape.)
    check(
        "phylum p\nnodule a;\npub fn secret(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.secret;\nfn f(y: Binary{8}) => Binary{8} = secret(y);",
    )
    .expect("the same import resolves once the name is pub");
}

#[test]
fn use_of_a_non_existent_name_is_a_never_silent_error() {
    // `a.nope` is declared by no nodule ⇒ "no such name" (distinct from private), never silent (G2).
    let msg = check_err(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.nope;\nfn f(y: Binary{8}) => Binary{8} = id(y);",
    );
    assert!(
        msg.contains("no such name") && msg.contains("nope"),
        "an unknown import must refuse as no-such-name, got: {msg}"
    );
}

#[test]
fn two_explicit_uses_binding_the_same_name_is_a_duplicate_import_error() {
    // Two explicit `use`s bind `id` ⇒ never-silent duplicate-import refusal (G2). Both `a.id` and
    // `c.id` are pub, so neither is unknown/private — the refusal is specifically the duplicate.
    let msg = check_err(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule c;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.id;\nuse c.id;\nfn f(y: Binary{8}) => Binary{8} = id(y);",
    );
    assert!(
        msg.contains("duplicate import") && msg.contains("id"),
        "two explicit imports of the same name must refuse, got: {msg}"
    );
}

// ---------------------------------------------------------------------------------------------
// Glob `use a.*`: accept; glob-vs-glob ambiguity → never-silent; control: explicit disambiguates.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_glob_use_brings_in_pub_names_and_type_checks() {
    // `use a.*` imports every pub name under `a` (here `id`). The reference `id(y)` resolves.
    check(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.*;\nfn f(y: Binary{8}) => Binary{8} = id(y);",
    )
    .expect("a glob import brings in pub names");
}

#[test]
fn a_glob_skips_private_names_but_imports_the_pub_ones() {
    // A glob imports a nodule's PUBLIC surface: `a` has a pub `id` and a private `secret`; `use a.*`
    // brings `id` (usable) but not `secret` (a reference to it is the normal unresolved-name error,
    // not a silent import). Here we only exercise that the pub one resolved.
    check(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nfn secret(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.*;\nfn f(y: Binary{8}) => Binary{8} = id(y);",
    )
    .expect("a glob imports the pub names");
}

#[test]
fn glob_vs_glob_ambiguity_on_a_referenced_name_is_a_never_silent_error() {
    // `a` and `c` both export a pub `dup`; `b` globs both and REFERENCES `dup` ⇒ never-silent
    // ambiguity (G2 — never a silent winner). The error must name the ambiguity + suggest explicit.
    let msg = check_err(
        "phylum p\nnodule a;\npub fn dup(x: Binary{8}) => Binary{8} = x;\nnodule c;\npub fn dup(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.*;\nuse c.*;\nfn f(y: Binary{8}) => Binary{8} = dup(y);",
    );
    assert!(
        msg.contains("ambiguous") && msg.contains("dup"),
        "a referenced glob-vs-glob collision must be an explicit ambiguity, got: {msg}"
    );
}

#[test]
fn control_glob_ambiguity_is_resolved_by_an_explicit_use() {
    // The non-vacuous control: the same two globs, but an explicit `use a.dup` shadows them — the
    // explicit binding wins deterministically (documented precedence), so `dup(y)` resolves.
    check(
        "phylum p\nnodule a;\npub fn dup(x: Binary{8}) => Binary{8} = x;\nnodule c;\npub fn dup(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.*;\nuse c.*;\nuse a.dup;\nfn f(y: Binary{8}) => Binary{8} = dup(y);",
    )
    .expect("an explicit use disambiguates the glob-vs-glob collision");
}

#[test]
fn control_an_unreferenced_glob_ambiguity_is_not_an_error() {
    // Ambiguity fires only on a *reference* (G2 wording): two globs collide on `dup`, but `b` never
    // references `dup`, so the program checks. (The collision is latent, refused only if used.)
    check(
        "phylum p\nnodule a;\npub fn dup(x: Binary{8}) => Binary{8} = x;\nnodule c;\npub fn dup(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.*;\nuse c.*;\nfn f(y: Binary{8}) => Binary{8} = not(y);",
    )
    .expect("an unreferenced glob collision is latent, not an error");
}

#[test]
fn own_decl_shadows_an_imported_glob_name_deterministically() {
    // Precedence (2) own-decl > (4) glob: `b` globs `a.*` (bringing `id`) but ALSO declares its own
    // `id` — its own wins, deterministically (documented shadowing, not a silent swap). The own
    // `id` returns Ternary, so a body typed against it proves the own decl is the one in scope.
    let penv = check(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse a.*;\nfn id(x: Ternary{6}) => Ternary{6} = x;\nfn f(y: Ternary{6}) => Ternary{6} = id(y);",
    )
    .expect("own decl shadows the glob import");
    let b = penv.nodule(&path(&["b"])).expect("nodule b;");
    // `b`'s `id` is the Ternary one (own), not the imported Binary one.
    let id = b.fn_decl("id").expect("id present");
    assert!(
        matches!(
            id.sig.ret.base,
            mycelium_l1::ast::BaseType::Ternary(mycelium_l1::ast::WidthRef::Lit(6))
        ),
        "own `id` (Ternary) must shadow the imported (Binary) one"
    );
}

// ---------------------------------------------------------------------------------------------
// Orphan rule — phylum-wide (pub-blind). Cross-nodule impl ACCEPTS; both-outside control rejects.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_cross_nodule_impl_is_accepted_trait_in_a_type_in_b_impl_in_b() {
    // The headline orphan generalization: trait `Negate` in `a` (pub), type `T` in `b`, impl in `b`.
    // Pre-M-662 this orphan-rejected (neither head local to `b`); now it ACCEPTS — `Negate` is
    // declared in the phylum (pub-blind coherence) and `T` is local to `b`. `b` imports the trait
    // name to reference it.
    check(
        "phylum p\nnodule a;\npub trait Negate[A] { fn negate(x: A) => A; };\nnodule b;\nuse a.Negate;\ntype T = Mk(Binary{8});\nimpl Negate[T] for T { fn negate(x: T) => T = x; };",
    )
    .expect("a cross-nodule impl (trait in a, type in b) is accepted phylum-wide");
}

#[test]
fn a_cross_nodule_impl_on_a_primitive_for_a_phylum_trait_is_accepted() {
    // Trait `Show` in `a` (pub), impl for the primitive `Binary{8}` in `b`. The trait is phylum-local
    // (coherence pub-blind), so even on a primitive `for`-type the impl is in-phylum ⇒ accepted.
    check(
        "phylum p\nnodule a;\npub trait Show[A] { fn show(x: A) => A; };\nnodule b;\nuse a.Show;\nimpl Show[Binary{8}] for Binary{8} { fn show(x: Binary{8}) => Binary{8} = x; };",
    )
    .expect("an impl for a primitive of a phylum-local trait is accepted");
}

#[test]
fn the_coherence_view_is_pub_blind_a_private_trait_still_satisfies_the_orphan_rule() {
    // Pub-blindness of coherence: `a`'s trait `Negate` is PRIVATE, yet the orphan rule still sees it
    // (coherence is enforcement authority, not the pub namespace). The impl lives in `a` (same
    // nodule), so no `use` is needed; this isolates "private trait + orphan rule sees it".
    check(
        "phylum p\nnodule a;\ntrait Negate[A] { fn negate(x: A) => A; };\nimpl Negate[Binary{8}] for Binary{8} { fn negate(x: Binary{8}) => Binary{8} = not(x); };\nnodule b;\nfn f() => Binary{8} = 0b0000_0000;",
    )
    .expect("a private trait still satisfies the orphan rule (coherence is pub-blind)");
}

#[test]
fn control_an_impl_whose_trait_and_type_are_both_outside_the_phylum_is_a_never_silent_refusal() {
    // The both-outside CONTROL (G2). In the phylum-wide model an impl can only reach the checker once
    // its trait + `for`-type *resolve*, and resolving a name implies an in-phylum declaration (own or
    // imported). So "both heads outside the phylum" manifests as the trait failing to resolve at all:
    // an explicit, never-silent refusal — the impl is **not** silently accepted. (The orphan *arm*
    // itself — trait+type resolvable yet neither phylum-local — is exercised directly as a checkty
    // unit test, since it is unreachable through resolvable surface names; see
    // `checkty::tests` orphan-arm coverage.)
    let msg = check_err(
        "phylum p\nnodule b;\nimpl Ext[Binary{8}] for Binary{8} { fn e(x: Binary{8}) => Binary{8} = x; };",
    );
    assert!(
        msg.contains("unknown trait") || msg.contains("orphan"),
        "an impl with neither head in the phylum must be a never-silent refusal, got: {msg}"
    );
}

// ---------------------------------------------------------------------------------------------
// Round-trip through `expand_phylum_to_source` (parse → print → parse stable).
// ---------------------------------------------------------------------------------------------

#[test]
fn pub_phylum_header_and_use_round_trip_through_expand() {
    let src = "phylum app.core\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\npub type Flag = Off | On;\nnodule b;\nuse a.id;\nuse a.*;\nfn f(y: Binary{8}) => Binary{8} = id(y);";
    let ph1 = parse_phylum(src).expect("parses");
    let printed = expand_phylum_to_source(&ph1);
    // The longhand twin must carry the phylum header, the `pub` markers, and both `use` forms.
    assert!(
        printed.contains("phylum app.core"),
        "header re-emitted:\n{printed}"
    );
    assert!(
        printed.contains("pub fn id"),
        "pub fn re-emitted:\n{printed}"
    );
    assert!(
        printed.contains("pub type Flag"),
        "pub type re-emitted:\n{printed}"
    );
    assert!(
        printed.contains("use a.id;\n"),
        "specific use re-emitted:\n{printed}"
    );
    assert!(
        printed.contains("use a.*;\n"),
        "glob use re-emitted:\n{printed}"
    );
    // Re-parsing the printed form yields a structurally-equal phylum (round-trip stable).
    let ph2 = parse_phylum(&printed).expect("re-parses");
    assert_eq!(ph1, ph2, "parse→print→parse must be stable");
}

#[test]
fn a_header_less_phylum_does_not_gain_a_phylum_line() {
    // An unmarked (header-less) phylum-of-one must NOT sprout a `phylum` line (never invented).
    let ph = parse_phylum("nodule solo;\npub fn f() => Binary{8} = 0b0;").expect("parses");
    let printed = expand_phylum_to_source(&ph);
    assert!(
        !printed.contains("phylum "),
        "no header must not be invented:\n{printed}"
    );
    assert!(
        printed.contains("pub fn f"),
        "the pub marker still round-trips:\n{printed}"
    );
}

// ---------------------------------------------------------------------------------------------
// Backward-compat: a bare nodule is a phylum-of-one; `check_phylum(of_one)` ≡ the single-nodule env.
// ---------------------------------------------------------------------------------------------

#[test]
fn check_phylum_of_one_matches_check_nodule() {
    let src =
        "nodule d;\nfn widen(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6}, policy: rt);";
    let nodule = mycelium_l1::parse(src).expect("parses");
    let via_nodule = mycelium_l1::check_nodule(&nodule).expect("checks");
    let via_phylum = check_phylum(&Phylum::of_one(nodule.clone())).expect("checks");
    let single = via_phylum.single().expect("phylum-of-one has one env");
    // The single-nodule env and the phylum-of-one's single env agree on the fn table + totality.
    assert_eq!(
        via_nodule.fns.keys().collect::<Vec<_>>(),
        single.fns.keys().collect::<Vec<_>>()
    );
    assert_eq!(via_nodule.totality, single.totality);
}

#[test]
fn a_private_cross_nodule_call_without_a_use_is_a_never_silent_unresolved_name() {
    // Without a `use`, a name from another nodule is simply not in scope: a reference is the normal
    // never-silent unknown-name error (cross-nodule visibility requires an explicit import — G2).
    let msg = check_err(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nfn f(y: Binary{8}) => Binary{8} = id(y);",
    );
    assert!(
        msg.contains("id") && (msg.contains("unknown") || msg.contains("prim")),
        "an un-imported cross-nodule name must be unresolved, got: {msg}"
    );
}

#[test]
fn a_single_segment_use_is_a_never_silent_teaching_refusal() {
    // M-662 (Copilot #369): `use X` names no nodule — a never-silent refusal with a teaching diagnostic
    // (qualify it as `use <nodule>.X`), NOT the confusing downstream "no such name" miss (G2).
    let msg = check_err(
        "phylum p\nnodule a;\npub fn id(x: Binary{8}) => Binary{8} = x;\nnodule b;\nuse id;\nfn f(y: Binary{8}) => Binary{8} = id(y);",
    );
    assert!(
        msg.contains("nodule-qualified") && msg.contains("id"),
        "a single-segment use must teach nodule-qualification, got: {msg}"
    );
}

/// Build a `Path` from segments (test helper).
fn path(segs: &[&str]) -> mycelium_l1::ast::Path {
    mycelium_l1::ast::Path(segs.iter().map(|s| (*s).to_owned()).collect())
}
