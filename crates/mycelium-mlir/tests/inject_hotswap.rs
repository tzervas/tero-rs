//! The **hot-inject safety argument under test** (M-341; ADR-017; ADR-016 call ABI; NFR-7).
//!
//! Demonstrates, on the in-process `dlopen` JIT, the claims ADR-017 makes:
//! - a call **resolves to a compiled entry if present, else interprets** (the continuum, RFC-0004 §9.1);
//! - **injection never mutates a live entry** — an in-flight call to the *old* hash finishes on the
//!   old code while a new caller dispatches to the *new* hash (the atomicity hazard dissolves because
//!   a change is a new hash under a new entry, ADR-017 decision 4);
//! - the **recompile set is the changed dependency-closure** by hash reachability (decision 3);
//! - the **injected-compiled path is observationally equivalent to the interpreter**, validated
//!   through the shared M-210 TV checker (`mycelium_cert::check`, `ObservationalEquiv`).
//!
//! The compiled path needs `clang`; where it is absent the JIT returns `ToolchainMissing` and the
//! compiled-path assertions are skipped (the pure dispatch/closure assertions still run) — the same
//! graceful skip the M-340 JIT tests use.

use std::collections::HashMap;

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{ContentHash, GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::Interpreter;
use mycelium_mlir::inject::{recompile_closure, Image, InjectError, Resolution};
use mycelium_mlir::inject_gate::{Admission, InjectMode};
use mycelium_mlir::AotError;
use mycelium_numerics::Certificate;

fn binary(bits: Vec<bool>) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `not(<bits>)` — a closed bit-subset program (the JIT's domain today).
fn not_prog(bits: Vec<bool>) -> Node {
    Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(binary(bits))],
    }
}

/// Whether the `clang` toolchain is present; if an inject hits `ToolchainMissing`, skip the
/// compiled-path assertions (environment skip, never a false failure).
fn toolchain_missing(e: &InjectError) -> bool {
    matches!(e, InjectError::Compile(AotError::ToolchainMissing(_)))
}

/// The default (loose, unsigned) resolutions on `Image::new()` — the M-961 I1 G2 tag.
fn interpreted_loose_unsigned() -> Resolution {
    Resolution::Interpreted {
        inject_mode: InjectMode::Loose,
        admission: Admission::Unsigned,
    }
}
fn compiled_loose_unsigned() -> Resolution {
    Resolution::Compiled {
        inject_mode: InjectMode::Loose,
        admission: Admission::Unsigned,
    }
}

#[test]
fn a_call_prefers_the_compiled_entry_else_interprets() {
    let mut img = Image::new();
    let prog = not_prog(vec![true, false, true, true]);

    // Defined-only → resolves to the interpreter.
    let hash = img
        .define(prog.clone())
        .expect("loose image admits unsigned");
    assert_eq!(img.resolve(&hash), interpreted_loose_unsigned());
    let interpreted = img.call(&hash).expect("interpreted call runs");

    // After injecting the same definition, the same hash now resolves to the compiled entry.
    match img.inject(&prog) {
        Ok(h) => {
            assert_eq!(h, hash, "injection key is the content hash");
            assert_eq!(img.resolve(&hash), compiled_loose_unsigned());
            let compiled = img.call(&hash).expect("compiled call runs");
            // NFR-7: the compiled and interpreted observables agree, via the shared M-210 checker.
            assert_observably_equal(&interpreted, &compiled, "interp ≡ injected-compiled");
        }
        Err(e) if toolchain_missing(&e) => { /* environment skip */ }
        Err(e) => panic!("unexpected inject error: {e}"),
    }
}

#[test]
fn injecting_a_new_hash_never_disturbs_the_old_live_entry() {
    // The core safety argument: an edit is a *new hash under a new entry*; the old entry is untouched,
    // so an in-flight call to the old hash finishes on the old code while a new caller dispatches new.
    let mut img = Image::new();
    let old = not_prog(vec![true, false, true, true]); // not(1011) = 0100
    let new = not_prog(vec![false, false, false, false]); // not(0000) = 1111  (a different definition)

    let hash_old = match img.inject(&old) {
        Ok(h) => h,
        Err(e) if toolchain_missing(&e) => return, // skip: no compiled path in this environment
        Err(e) => panic!("unexpected inject error: {e}"),
    };
    let result_old_before = img.call(&hash_old).expect("old compiled call runs");
    assert_eq!(img.injected_count(), 1);

    // Inject the edited definition: a brand-new hash, a brand-new entry.
    let hash_new = img
        .inject(&new)
        .expect("second inject runs (toolchain present)");
    assert_ne!(
        hash_old, hash_new,
        "an edit is a new hash (ADR-017 decision 4)"
    );
    assert_eq!(
        img.injected_count(),
        2,
        "a new entry was added, none overwritten"
    );

    // The old hash STILL dispatches to the old code and yields the old result (no live-entry mutation).
    assert_eq!(img.resolve(&hash_old), compiled_loose_unsigned());
    let result_old_after = img.call(&hash_old).expect("old hash still callable");
    assert_eq!(
        result_old_before.payload(),
        result_old_after.payload(),
        "in-flight old hash finishes on old code, unchanged by the injection"
    );
    assert_eq!(
        result_old_after.payload(),
        &Payload::Bits(vec![false, true, false, false])
    );

    // A new caller — referencing the new hash — dispatches to the new code.
    let result_new = img.call(&hash_new).expect("new hash callable");
    assert_eq!(
        result_new.payload(),
        &Payload::Bits(vec![true, true, true, true])
    );
    assert_ne!(
        result_old_after.payload(),
        result_new.payload(),
        "old and new entries are genuinely distinct code"
    );
}

#[test]
fn re_injecting_the_same_definition_is_publish_once_idempotent() {
    // Content-addressing: re-injecting the same definition keeps the live entry (no overwrite, no
    // recompile) — the dispatch table does not grow.
    let mut img = Image::new();
    let prog = not_prog(vec![true, true, false, false]);
    let h1 = match img.inject(&prog) {
        Ok(h) => h,
        Err(e) if toolchain_missing(&e) => return,
        Err(e) => panic!("unexpected inject error: {e}"),
    };
    assert_eq!(img.injected_count(), 1);
    let h2 = img.inject(&prog).expect("re-inject runs");
    assert_eq!(h1, h2);
    assert_eq!(
        img.injected_count(),
        1,
        "publish-once: no second entry for the same hash"
    );
}

#[test]
fn injected_compiled_equals_interpreter_across_a_corpus_via_m210() {
    // NFR-7 over a small bit/trit corpus: for each program the injected-compiled result is
    // observationally equivalent to the reference interpreter, validated by the shared M-210 checker.
    let interp = Interpreter::default();
    let corpus = [
        not_prog(vec![true, false, true, true]),
        Node::Op {
            prim: "bit.xor".into(),
            args: vec![
                Node::Const(binary(vec![true, false, true, false])),
                Node::Const(binary(vec![true, true, true, true])),
            ],
        },
        Node::Op {
            prim: "trit.neg".into(),
            args: vec![Node::Const(
                Value::new(
                    Repr::Ternary { trits: 3 },
                    Payload::Trits(vec![
                        mycelium_core::Trit::Pos,
                        mycelium_core::Trit::Zero,
                        mycelium_core::Trit::Neg,
                    ]),
                    Meta::exact(Provenance::Root),
                )
                .unwrap(),
            )],
        },
    ];

    let mut img = Image::new();
    for (i, prog) in corpus.iter().enumerate() {
        let reference = interp.eval(prog).expect("interpreter runs the reference");
        match img.inject(prog) {
            Ok(hash) => {
                let compiled = img.call(&hash).expect("compiled call runs");
                assert_observably_equal(&reference, &compiled, &format!("program #{i}"));
            }
            Err(e) if toolchain_missing(&e) => return, // skip whole corpus if no toolchain
            Err(e) => panic!("program #{i}: unexpected inject error: {e}"),
        }
    }
}

#[test]
fn recompile_set_is_the_changed_dependency_closure() {
    // ADR-017 decision 3: editing a definition recompiles it + its transitive dependents, by hash
    // reachability — nothing else. Modeled on the content-hash dependency graph (no AST diff).
    let h = |s: &str| ContentHash::parse(&format!("blake3:{s}")).unwrap();
    let (main, helper, leaf, sibling) = (h("main"), h("helper"), h("leaf"), h("sibling"));
    // main -> helper -> leaf ;  main -> sibling  (sibling does not depend on leaf)
    let mut deps: HashMap<ContentHash, Vec<ContentHash>> = HashMap::new();
    deps.insert(main.clone(), vec![helper.clone(), sibling.clone()]);
    deps.insert(helper.clone(), vec![leaf.clone()]);
    deps.insert(leaf.clone(), vec![]);
    deps.insert(sibling.clone(), vec![]);

    let set = recompile_closure(&deps, std::slice::from_ref(&leaf));
    // leaf changed → helper and main (its transitive callers) recompile; sibling does NOT.
    assert!(set.contains(&leaf) && set.contains(&helper) && set.contains(&main));
    assert!(
        !set.contains(&sibling),
        "an unchanged, independent definition keeps its compiled entry"
    );
    assert_eq!(set.len(), 3);
}

#[test]
fn an_unknown_hash_is_an_explicit_dispatch_miss_not_a_guess() {
    let img = Image::new();
    let unknown = ContentHash::parse("blake3:nonexistent").unwrap();
    assert_eq!(img.resolve(&unknown), Resolution::Miss);
    assert!(matches!(
        img.call(&unknown),
        Err(InjectError::DispatchMiss(_))
    ));
}

/// Assert two values are observationally equal *through the shared M-210 TV checker* (the same
/// `ObservationalEquiv` instance used by the L1/interp/AOT differential) — never a bespoke compare.
fn assert_observably_equal(a: &Value, b: &Value, ctx: &str) {
    assert_eq!(
        check(
            a,
            b,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "{ctx}: the M-210 checker must validate interp ≡ injected-compiled"
    );
}
