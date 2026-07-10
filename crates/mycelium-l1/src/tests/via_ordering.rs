//! `via` delegation trait-registry ordering (M-966, DN-53 §A.3.2).
//!
//! Three properties are pinned here, data-driven over small case tables (per the house test-layout
//! rule — a test body is *assert over a case*, not bespoke logic):
//!
//! 1. **Single-delegate happy path** — one `via` clause per trait resolves, registers a coherent
//!    instance, and is `EXPLAIN`-able via `Env::via_provenance` (the `(field_idx, object_name)` the
//!    forwarding impl came from — DN-53 §A.3.3's "never a hidden dispatch" promise).
//! 2. **Ambiguity refusal (red-then-green)** — two `via` clauses on the *same* object delegating the
//!    *same* trait have no deterministic tiebreak and are refused explicitly (G2), naming the trait
//!    and both candidate field indices.
//! 3. **Deterministic order across runs** — the same ambiguous/valid source produces the byte-identical
//!    `Env` (`via_provenance`/`instances`) and the byte-identical refusal message across repeated
//!    `check_nodule` calls, since resolution walks `via_decls` (an ordered `Vec`) into `BTreeMap`s,
//!    never a `HashMap` — no run-to-run iteration-order dependence.
//!
//! ### Fixture shape (a real, load-bearing finding while wiring up the happy path)
//!
//! DN-53 §A.3.2's own worked lowering generates `impl Iface for Foo { fn iface_method(self: Foo) ->
//! … = (match self { Mk(f1, _) = f1 }).iface_method() }` — the forwarding method's **self** parameter
//! is typed `Foo` (the *composing* object), not the delegate field's own type. Driving that through
//! `check_nodule` end-to-end surfaced that `expand_object_via_decls` had never actually substituted
//! *anything* into the trait's abstract signature (`sig: sig.clone()`) — so no `via` clause could
//! ever fully type-check (an "unknown type" `CheckError` on the trait's own abstract parameter,
//! independent of M-966). Fixed here by substituting the trait's params with **this `via` clause's
//! own trait arguments** (`subst_type_param_in_sig`, reusing `derive`'s M-973 substitution
//! machinery) — which only agrees with `check_impl_methods`'s independent structural check when the
//! `via` clause's trait argument names the **composing object's own type** (`via N : Trait[Foo]`,
//! matching DN-53's `self: Foo`), not the delegate field's type. The delegate field's own type must
//! *separately* already carry an instance of the trait (DN-53: "the delegate must implement the
//! trait"), resolved independently at the forwarding call site.
//!
//! A further, narrower and orthogonal limitation (kept out of fixtures below, not fixed here): a
//! trait whose **return type** also mentions the abstract parameter (e.g. `fn wrap(x: A) => A`)
//! cannot be delegated at all under stage-1's call-site unification — the outer signature demands a
//! `Foo`-typed return, but the actual forwarded value is the delegate's own (different) type, an
//! unsatisfiable unification. Fixtures here use `fn method(x: A) => Bool`-shaped traits (return type
//! independent of `A`) to stay clear of that separate gap; it is flagged in the PR description as a
//! pre-existing residual, not one M-966 is scoped to close.

use crate::checkty::*;
use crate::parse;

fn env(src: &str) -> Env {
    check_nodule(&parse(src).expect("parses")).expect("checks")
}

fn check_err(src: &str) -> CheckError {
    check_nodule(&parse(src).expect("parses")).expect_err("must fail to check")
}

/// Fixture prelude shared by every case below: a single-parameter trait `Wrap[A]` (return type
/// `Bool`, independent of `A` — see the module doc's "orthogonal limitation" note) plus its base
/// instance for `Binary{8}` (the concrete type every `via` delegate field carries in these
/// fixtures — DN-53: "the delegate must implement the trait"). `object_src` must start with its own
/// `nodule …;` header; the prelude is spliced in right after it.
fn with_wrap_prelude(object_src: &str) -> String {
    let (header, rest) = object_src
        .split_once('\n')
        .expect("fixture must have a `nodule …;` header line");
    format!(
        "{header}\n\
         trait Wrap[A] {{ fn wrap(x: A) => Bool; }};\n\
         impl Wrap[Binary{{8}}] for Binary{{8}} {{ fn wrap(x: Binary{{8}}) => Bool = True; }};\n\
         {rest}"
    )
}

/// A second single-parameter trait (`Peek[A]`), for the "different traits are not ambiguous" case.
fn with_wrap_and_peek_prelude(object_src: &str) -> String {
    let (header, rest) = object_src
        .split_once('\n')
        .expect("fixture must have a `nodule …;` header line");
    format!(
        "{header}\n\
         trait Wrap[A] {{ fn wrap(x: A) => Bool; }};\n\
         impl Wrap[Binary{{8}}] for Binary{{8}} {{ fn wrap(x: Binary{{8}}) => Bool = True; }};\n\
         trait Peek[A] {{ fn peek(x: A) => Bool; }};\n\
         impl Peek[Binary{{8}}] for Binary{{8}} {{ fn peek(x: Binary{{8}}) => Bool = True; }};\n\
         {rest}"
    )
}

// ---- 1. Single-delegate happy path: resolves + is EXPLAIN-able ------------------------------

/// One case per (trait, field position) — a single `via` clause always resolves and records
/// provenance naming exactly the delegate that provided it. `via N : Wrap[<ObjectName>]` names the
/// *composing* object's own type as the trait argument (DN-53 §A.3.2's `self: Foo` lowering).
struct SingleDelegateCase {
    /// A descriptive case name (for assertion messages only).
    name: &'static str,
    /// Nodule source: a two-field object, one `via` clause on the given field for `Wrap`.
    object_src: &'static str,
    /// The expected `(field_idx, object_name)` provenance entry.
    expect_provenance: (u32, &'static str),
}

const SINGLE_DELEGATE_CASES: &[SingleDelegateCase] = &[
    SingleDelegateCase {
        name: "delegates through field 0",
        object_src: "nodule single.zero;\n\
              object BoxA {\n\
                  Mk(Binary{8}, Binary{8});\n\
                  via 0 : Wrap[BoxA];\n\
              };",
        expect_provenance: (0, "BoxA"),
    },
    SingleDelegateCase {
        name: "delegates through field 1 (non-zero index)",
        object_src: "nodule single.one;\n\
              object BoxB {\n\
                  Mk(Binary{8}, Binary{8});\n\
                  via 1 : Wrap[BoxB];\n\
              };",
        expect_provenance: (1, "BoxB"),
    },
];

#[test]
fn single_via_delegate_resolves_and_is_explainable() {
    for case in SINGLE_DELEGATE_CASES {
        let e = env(&with_wrap_prelude(case.object_src));
        // Resolves: exactly one *object-side* `Wrap` instance is registered (the base
        // `impl Wrap[Binary{8}] for Binary{8}` in the prelude is the other, unrelated, instance).
        let matches: Vec<_> = e
            .instances
            .keys()
            .filter(|(tr, head)| tr == "Wrap" && head != "Binary")
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "case `{}`: expected exactly one object-side `Wrap` instance, got {:?}",
            case.name,
            matches
        );
        // EXPLAIN-able: `via_provenance` names the exact delegate (field_idx, object_name) — never
        // a silent/opaque dispatch (DN-53 §A.3.3, G2).
        let key = matches[0].clone();
        let (field_idx, object_name) = e.via_provenance.get(&key).unwrap_or_else(|| {
            panic!("case `{}`: missing via_provenance for {:?}", case.name, key)
        });
        assert_eq!(
            (*field_idx, object_name.as_str()),
            case.expect_provenance,
            "case `{}`: via_provenance mismatch",
            case.name
        );
    }
}

// ---- 2. Ambiguity refusal (red-then-green) --------------------------------------------------

/// Two `via` clauses on the same object delegating the same trait — no deterministic tiebreak
/// between two delegates, refused explicitly (never a silent first-match pick). The ambiguity
/// check runs before any signature substitution, so it is independent of the `Trait[ObjectName]`
/// argument convention above; these fixtures use a bare `via N : Wrap` for brevity.
struct AmbiguityCase {
    name: &'static str,
    object_src: &'static str,
    /// The two field indices the error message must name (in declaration order).
    field_idxs: (u32, u32),
}

const AMBIGUITY_CASES: &[AmbiguityCase] = &[
    AmbiguityCase {
        name: "two via clauses, adjacent fields, same trait",
        object_src: "nodule ambiguous.adjacent;\n\
              object Pair {\n\
                  Mk(Binary{8}, Binary{8});\n\
                  via 0 : Wrap;\n\
                  via 1 : Wrap;\n\
              };",
        field_idxs: (0, 1),
    },
    AmbiguityCase {
        name: "three fields, ambiguity between the last two",
        object_src: "nodule ambiguous.last_two;\n\
              object Triple {\n\
                  Mk(Binary{8}, Binary{8}, Binary{8});\n\
                  via 1 : Wrap;\n\
                  via 2 : Wrap;\n\
              };",
        field_idxs: (1, 2),
    },
];

#[test]
fn via_ambiguous_delegation_same_trait_is_refused_never_silently() {
    for case in AMBIGUITY_CASES {
        // Red: without the fix (this test), two `via` clauses on the same trait would either
        // silently pick the first-declared delegate or hit a generic, unhelpful coherence error
        // from `register_instances` — neither of which names the ambiguity. Green (post-fix):
        // an explicit `CheckError` naming the trait and both candidate field indices.
        let err = check_err(&with_wrap_prelude(case.object_src));
        let msg = err.to_string();
        assert!(
            msg.contains("ambiguous") && msg.contains("Wrap"),
            "case `{}`: expected an ambiguous-`via`-delegation error naming `Wrap`, got: {}",
            case.name,
            msg
        );
        let (first, second) = case.field_idxs;
        assert!(
            msg.contains(&format!("via {first} :")) && msg.contains(&format!("via {second} :")),
            "case `{}`: expected the error to name both candidate field indices {} and {}, got: {}",
            case.name,
            first,
            second,
            msg
        );
    }
}

/// A single `via` clause per trait is unaffected by the ambiguity check (no false positive when
/// two `via` clauses target *different* traits on the same object).
#[test]
fn via_clauses_for_different_traits_on_the_same_object_are_not_ambiguous() {
    let e = env(&with_wrap_and_peek_prelude(
        "nodule not_ambiguous.different_traits;\n\
         object Dual {\n\
             Mk(Binary{8}, Binary{8});\n\
             via 0 : Wrap[Dual];\n\
             via 1 : Peek[Dual];\n\
         };",
    ));
    assert!(e
        .instances
        .keys()
        .any(|(tr, head)| tr == "Wrap" && head == "Data:Dual"));
    assert!(e
        .instances
        .keys()
        .any(|(tr, head)| tr == "Peek" && head == "Data:Dual"));
    assert_eq!(
        e.via_provenance
            .get(&("Wrap".to_owned(), "Data:Dual".to_owned())),
        Some(&(0, "Dual".to_owned()))
    );
    assert_eq!(
        e.via_provenance
            .get(&("Peek".to_owned(), "Data:Dual".to_owned())),
        Some(&(1, "Dual".to_owned()))
    );
}

// ---- 3. Deterministic order across runs -----------------------------------------------------

/// Re-checking the same source repeatedly must produce the byte-identical `Env` (`instances` +
/// `via_provenance`) every time — resolution walks an ordered `Vec` (`via_decls`) into `BTreeMap`s,
/// never a `HashMap`, so there is no run-to-run iteration-order dependence (`Empirical`, pinned here
/// rather than `Proven` — repeated runs in one process are evidence, not a proof of the absence of
/// any nondeterministic source).
#[test]
fn repeated_checks_of_the_same_via_source_are_byte_identical() {
    let src = with_wrap_and_peek_prelude(
        "nodule repeat.check;\n\
         object Dual {\n\
             Mk(Binary{8}, Binary{8});\n\
             via 0 : Wrap[Dual];\n\
             via 1 : Peek[Dual];\n\
         };",
    );
    let first = env(&src);
    for run in 1..=9 {
        let again = env(&src);
        assert_eq!(
            first.instances, again.instances,
            "run {run}: `instances` diverged across repeated checks of identical source"
        );
        assert_eq!(
            first.via_provenance, again.via_provenance,
            "run {run}: `via_provenance` diverged across repeated checks of identical source"
        );
    }
}

/// The ambiguity refusal's message is itself deterministic — repeated checks of the same
/// ambiguous source name the same two candidates every time (never a run-dependent pick of
/// "which" is first).
#[test]
fn repeated_checks_of_ambiguous_via_source_name_the_same_candidates() {
    let src = with_wrap_prelude(
        "nodule repeat.ambiguous;\n\
         object Pair {\n\
             Mk(Binary{8}, Binary{8});\n\
             via 0 : Wrap;\n\
             via 1 : Wrap;\n\
         };",
    );
    let first_msg = check_err(&src).to_string();
    for run in 1..=9 {
        let again_msg = check_err(&src).to_string();
        assert_eq!(
            first_msg, again_msg,
            "run {run}: ambiguity error message diverged across repeated checks of identical source"
        );
    }
}
