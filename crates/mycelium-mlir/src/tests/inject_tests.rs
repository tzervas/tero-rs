//! Tests for `crate::inject` — hash-keyed dispatch, injection, recompile closure — M-789 / RFC-0034 §8.
//!
//! Extracted from the old inline `#[cfg(test)]` block in `inject.rs` (CLAUDE.md test-layout rule).
//! New additions here (M-789):
//!
//! * `dispatch_key_is_independent_of_cert_mode` — verifies that the ABI dispatch key (`ContentHash`
//!   produced by `Image::define`) is NOT influenced by `CertMode`. `CertMode` is a `Meta` field;
//!   `Meta` is excluded from the content hash (RFC-0001 §4.6; ADR-003); therefore the dispatch key
//!   is a *compile/deploy-phase* identity that survives a cert-off runtime (RFC-0034 §8). The
//!   `mycelium-core::content` module already tests this at the core level
//!   (`cert_mode_is_excluded_from_the_content_hash`); this test locks it in at the dispatch layer.
//!
//! * `define_is_idempotent_same_node_same_key` — `Image::define` is idempotent: the same `Node`
//!   always returns the same `ContentHash` and the definition set does not grow.
//!
//! Guarantee tags:
//! * `dispatch_key_is_independent_of_cert_mode` — `Proven` by construction (RFC-0001 §4.6 excludes
//!   `Meta`; `CertMode ⊆ Meta`). Exhaustive over `CertMode::ALL`.
//! * `define_is_idempotent_same_node_same_key` — `Proven` (by `HashMap::entry` idempotence).

use std::collections::HashMap;

use mycelium_core::{CertMode, ContentHash, Meta, Payload, Provenance, Repr, Value};

use crate::inject_gate::{Admission, InjectMode};

use crate::inject::{recompile_closure, Image, InjectError, Resolution};

// ─── helpers ──────────────────────────────────────────────────────────────────

fn binary(bits: Vec<bool>) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `not(<bits>)` — a closed bit-subset program (the JIT's domain).
fn not_prog(bits: Vec<bool>) -> mycelium_core::Node {
    mycelium_core::Node::Op {
        prim: "bit.not".into(),
        args: vec![mycelium_core::Node::Const(binary(bits))],
    }
}

fn h(s: &str) -> ContentHash {
    ContentHash::parse(&format!("blake3:{s}")).unwrap()
}

/// The default (loose, unsigned) interpreted resolution — the I1 G2-tag on the dev path (M-961).
fn interpreted_loose_unsigned() -> Resolution {
    Resolution::Interpreted {
        inject_mode: InjectMode::Loose,
        admission: Admission::Unsigned,
    }
}

// ─── pre-existing tests (extracted from inject.rs inline #[cfg(test)]) ────────

#[test]
fn defined_definition_resolves_to_interpreted_and_calls() {
    // No toolchain needed: an interpret-only definition dispatches to the interpreter.
    let mut img = Image::new();
    let prog = not_prog(vec![true, false, true, true]);
    let hash = img.define(prog).expect("loose image admits unsigned");
    assert_eq!(img.resolve(&hash), interpreted_loose_unsigned());
    let v = img.call(&hash).expect("interpreted call runs");
    assert_eq!(v.payload(), &Payload::Bits(vec![false, true, false, false]));
}

#[test]
fn an_unknown_hash_is_an_explicit_dispatch_miss() {
    let img = Image::new();
    let miss = h("deadbeef");
    assert_eq!(img.resolve(&miss), Resolution::Miss);
    assert_eq!(img.call(&miss), Err(InjectError::DispatchMiss(miss)));
}

#[test]
fn different_programs_get_different_hashes() {
    // The injection key is the content hash — an edit is a new hash (ADR-017 decision 4).
    let a = not_prog(vec![true, false]).content_hash();
    let b = not_prog(vec![false, false]).content_hash();
    assert_ne!(a, b);
}

#[test]
fn recompile_closure_is_the_reverse_reachable_set() {
    // Graph: main -> helper -> leaf ; other (independent).
    let (main, helper, leaf, other) = (h("main"), h("helper"), h("leaf"), h("other"));
    let mut deps: HashMap<ContentHash, Vec<ContentHash>> = HashMap::new();
    deps.insert(main.clone(), vec![helper.clone()]);
    deps.insert(helper.clone(), vec![leaf.clone()]);
    deps.insert(leaf.clone(), vec![]);
    deps.insert(other.clone(), vec![]);

    // Editing `leaf` must recompile leaf + helper + main, but not the independent `other`.
    let set = recompile_closure(&deps, std::slice::from_ref(&leaf));
    assert_eq!(
        set,
        std::collections::HashSet::from([leaf.clone(), helper.clone(), main.clone()])
    );
    assert!(!set.contains(&other));

    // Editing a leaf with no dependents recompiles only itself.
    assert_eq!(
        recompile_closure(&deps, std::slice::from_ref(&other)),
        std::collections::HashSet::from([other])
    );
}

#[test]
fn recompile_closure_terminates_on_a_cycle() {
    // Mutual reference (a hash cycle) must not loop forever — closure is still finite.
    let (a, b) = (h("a"), h("b"));
    let mut deps: HashMap<ContentHash, Vec<ContentHash>> = HashMap::new();
    deps.insert(a.clone(), vec![b.clone()]);
    deps.insert(b.clone(), vec![a.clone()]);
    assert_eq!(
        recompile_closure(&deps, std::slice::from_ref(&a)),
        std::collections::HashSet::from([a, b])
    );
}

// ─── M-789 / RFC-0034 §8: ABI dispatch keys are independent of CertMode ───────

/// **Test (M-789 DoD): the ABI dispatch key does not depend on `CertMode` (RFC-0034 §8).**
///
/// The dispatch key is `Node::content_hash()` (ADR-016/017; ADR-003). Per RFC-0001 §4.6, `Meta`
/// (which carries `CertMode`) is excluded from the content hash by construction. Therefore:
/// - A `Node::Const(value)` where the value carries a different `CertMode` tag in its `Meta`
///   produces the same `ContentHash` (the same dispatch key).
/// - Hot-inject / ABI dispatch survives a cert-off runtime (RFC-0034 §8): changing `CertMode`
///   never changes which compiled entry a hash resolves to.
///
/// Guarantee: `Proven` by construction (RFC-0001 §4.6 excludes `Meta`; `CertMode ⊆ Meta`).
/// Exhaustive over `CertMode::ALL` (three tiers: Fast, Balanced, Certified).
///
/// Mutant-witness: if `CertMode` were accidentally included in `Node::content_hash()`, the
/// three tiers would produce three distinct hashes, and `all_equal` would fail.
#[test]
fn dispatch_key_is_independent_of_cert_mode() {
    let bits = vec![true, false, true, false];

    // Build the same program node body, but embed a value with different CertMode tags.
    // CertMode is a Meta tag excluded from the content hash (RFC-0001 §4.6; ADR-003).
    let mut keys: Vec<ContentHash> = Vec::new();
    for mode in &CertMode::ALL {
        let val = Value::new(
            Repr::Binary { width: 4 },
            Payload::Bits(bits.clone()),
            Meta::exact(Provenance::Root).with_cert_mode(*mode),
        )
        .unwrap();
        let prog = mycelium_core::Node::Op {
            prim: "bit.not".into(),
            args: vec![mycelium_core::Node::Const(val)],
        };
        // Use `Image::define` to get the canonical dispatch key (the same path as production).
        let mut img = Image::new();
        let key = img.define(prog).expect("loose image admits unsigned");
        keys.push(key);
    }

    // All three tiers yield the same dispatch key — mode is excluded from identity (ADR-003).
    let reference = &keys[0];
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            key, reference,
            "CertMode tier {} produced a different ABI dispatch key — CertMode must not enter the \
             content hash (RFC-0034 §8; ADR-003; RFC-0001 §4.6)",
            i
        );
    }
}

/// **Test: `Image::define` is idempotent across repeated registration of the same node.**
///
/// ADR-017 decision 4: a definition registered multiple times always yields the same `ContentHash`
/// and does not grow the definition set. This holds regardless of `CertMode` (RFC-0034 §8).
///
/// Guarantee: `Proven` (by the hash-keyed idempotence of `HashMap::entry` in `Image::define`).
/// Mutant-witness: a mutation making `define` return a fresh hash on each call would break the
/// `defined_count == 1` assertion.
#[test]
fn define_is_idempotent_same_node_same_key() {
    let mut img = Image::new();
    let prog = not_prog(vec![true, true]);
    let h1 = img
        .define(prog.clone())
        .expect("loose image admits unsigned");
    let h2 = img.define(prog).expect("loose image admits unsigned");
    assert_eq!(h1, h2, "same definition must yield the same key every time");
    assert_eq!(img.defined_count(), 1, "no duplicate registration");
}
