//! Tests for `crate::data` — the data registry, `FieldSpec`, `FieldTy`, `CtorRef`, and the
//! ADR-033 §10 PATH-A full-signature encoding for `FieldSpec::Fn`.
//!
//! Tests extracted from the former inline `#[cfg(test)] mod tests` block in `data.rs`
//! (test layout rule M-797: no inline tests in logic files; as-touched extraction).
//!
//! New tests (ADR-033):
//!   - `fn_distinct_signatures_hash_distinctly` — distinct fn types → distinct declaration hashes
//!     (`Empirical` evidence for ADR-033 §10.5 property 1/2; moves FLAG-1 from `Declared` to
//!     `Empirical` — VR-5: not `Proven`, unmechanized).
//!   - `fn_same_signature_hashes_stably` — same fn type → same hash (determinism).
//!   - `fn_field_vs_repr_field_hash_distinctly` — `FIELD_FN` tag does not collide with `FIELD_REPR`.
//!   - `fn_arity_mismatch_is_explicit_error` — never-silent G2 for `arity ≠ params.len()`.
//!   - `fn_sig_data_ref_in_cycle_hashes_correctly` — FLAG-3: `Data` leaf inside a `FnSig` uses the
//!     in-cycle placeholder, not a circular hash.
//!   - `fn_sig_data_ref_out_of_cycle_resolves` — `Data` leaf inside a `FnSig` resolves to hash.
//!   - `fn_dangling_sig_ref_is_explicit_error` — a `FieldTyRef::Data` referencing an unknown decl.
//!   - `fn_nested_sig_distinct_hashes` — higher-order `FieldTyRef::Fn` nesting hashes distinctly.

use std::collections::BTreeMap;

use crate::data::{
    CtorSpec, DataRegistry, DeclSpec, FieldSpec, FieldTy, FieldTyRef, FnSig, RegistryError,
    ResolvedFieldTyRef,
};
use crate::repr::Repr;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn nat_spec() -> BTreeMap<String, DeclSpec> {
    // type Nat = Z | S(Nat)
    let mut m = BTreeMap::new();
    m.insert(
        "Nat".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] },
                CtorSpec {
                    fields: vec![FieldSpec::Data("Nat".to_owned())],
                },
            ],
        },
    );
    m
}

/// A `FieldSpec::Fn` with a single `Repr` parameter and a `Repr` return — no Data refs.
fn fn_spec_repr_only(param: Repr, ret: Repr, arity: u32) -> FieldSpec {
    FieldSpec::Fn {
        arity,
        sig: FnSig {
            arity,
            params: vec![FieldTyRef::Repr(param)],
            ret: Box::new(FieldTyRef::Repr(ret)),
        },
    }
}

/// Build a single-constructor declaration with the given fields.
fn single_ctor(fields: Vec<FieldSpec>) -> DeclSpec {
    DeclSpec {
        ctors: vec![CtorSpec { fields }],
    }
}

// ---------------------------------------------------------------------------
// Existing tests (extracted from data.rs inline block — M-797)
// ---------------------------------------------------------------------------

#[test]
fn self_recursive_decl_hashes_without_looping() {
    let reg = DataRegistry::build(&nat_spec()).expect("builds");
    let z = reg.ctor_ref("Nat", 0).expect("Z");
    let s = reg.ctor_ref("Nat", 1).expect("S");
    assert_eq!(z.decl(), s.decl(), "same declaration");
    assert_eq!(z.index(), 0);
    assert_eq!(s.index(), 1);
    assert_eq!(reg.field_count(&z), Some(0));
    assert_eq!(reg.field_count(&s), Some(1));
    assert_eq!(reg.ctor_count(&s), Some(2));
}

#[test]
fn identity_is_structural_not_nominal() {
    // The same structure under a different *name* gets the same declaration hash (names are not
    // identity — ADR-003).
    let mut renamed = BTreeMap::new();
    renamed.insert(
        "Peano".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] },
                CtorSpec {
                    fields: vec![FieldSpec::Data("Peano".to_owned())],
                },
            ],
        },
    );
    let nat = DataRegistry::build(&nat_spec()).unwrap();
    let peano = DataRegistry::build(&renamed).unwrap();
    assert_eq!(
        nat.decl_hash("Nat"),
        peano.decl_hash("Peano"),
        "α-equivalent declarations collide regardless of name"
    );
}

#[test]
fn constructor_order_is_identity_bearing() {
    // Z | S(Nat)  vs  S(Nat) | Z  are different declarations (order significant).
    let mut swapped = BTreeMap::new();
    swapped.insert(
        "Nat".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec {
                    fields: vec![FieldSpec::Data("Nat".to_owned())],
                },
                CtorSpec { fields: vec![] },
            ],
        },
    );
    let a = DataRegistry::build(&nat_spec()).unwrap();
    let b = DataRegistry::build(&swapped).unwrap();
    assert_ne!(a.decl_hash("Nat"), b.decl_hash("Nat"));
}

#[test]
fn field_repr_is_identity_bearing() {
    // type B8 = Wrap(Binary{8})  vs  type B8 = Wrap(Binary{4}) differ.
    let mk = |w| {
        let mut m = BTreeMap::new();
        m.insert(
            "W".to_owned(),
            DeclSpec {
                ctors: vec![CtorSpec {
                    fields: vec![FieldSpec::Repr(Repr::Binary { width: w })],
                }],
            },
        );
        DataRegistry::build(&m).unwrap()
    };
    assert_ne!(mk(8).decl_hash("W"), mk(4).decl_hash("W"));
}

#[test]
fn a_dangling_reference_is_an_explicit_error() {
    let mut m = BTreeMap::new();
    m.insert(
        "Tree".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec {
                fields: vec![FieldSpec::Data("Forest".to_owned())], // not declared
            }],
        },
    );
    assert_eq!(
        DataRegistry::build(&m).unwrap_err(),
        RegistryError::UnknownTypeRef {
            in_decl: "Tree".to_owned(),
            missing: "Forest".to_owned(),
        }
    );
}

#[test]
fn mutual_recursion_orders_canonically_and_name_independently() {
    // type Tree = Leaf | Node(Forest);  type Forest = Empty | Cons(Tree, Forest)
    // A mutually-recursive 2-cycle. Building it under two different *name* sets for the same
    // structure must yield the same group identity (R7-Q3: names are not identity, ADR-003).
    let mk = |t: &str, f: &str| {
        let mut m = BTreeMap::new();
        m.insert(
            t.to_owned(),
            DeclSpec {
                ctors: vec![
                    CtorSpec { fields: vec![] },
                    CtorSpec {
                        fields: vec![FieldSpec::Data(f.to_owned())],
                    },
                ],
            },
        );
        m.insert(
            f.to_owned(),
            DeclSpec {
                ctors: vec![
                    CtorSpec { fields: vec![] },
                    CtorSpec {
                        fields: vec![FieldSpec::Data(t.to_owned()), FieldSpec::Data(f.to_owned())],
                    },
                ],
            },
        );
        DataRegistry::build(&m).unwrap()
    };
    let a = mk("Tree", "Forest");
    let b = mk("Arbol", "Bosque");
    // The structurally-corresponding declarations collide across the renaming.
    assert_eq!(a.decl_hash("Tree"), b.decl_hash("Arbol"));
    assert_eq!(a.decl_hash("Forest"), b.decl_hash("Bosque"));
    // Building twice is deterministic.
    assert_eq!(a.decl_hash("Tree"), mk("Tree", "Forest").decl_hash("Tree"));
}

#[test]
fn out_of_cycle_reference_resolves_dependencies_first() {
    // type Byte = MkByte(Binary{8});  type Pair = MkPair(Byte, Byte) — no cycle, Byte first.
    let mut m = BTreeMap::new();
    m.insert(
        "Byte".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec {
                fields: vec![FieldSpec::Repr(Repr::Binary { width: 8 })],
            }],
        },
    );
    m.insert(
        "Pair".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec {
                fields: vec![
                    FieldSpec::Data("Byte".to_owned()),
                    FieldSpec::Data("Byte".to_owned()),
                ],
            }],
        },
    );
    let reg = DataRegistry::build(&m).expect("builds");
    let pair = reg.ctor_ref("Pair", 0).expect("MkPair");
    let byte_decl = reg.decl_hash("Byte").unwrap().clone();
    // The Pair constructor's two fields both resolve to the Byte declaration hash.
    let decl = reg.ctor(&pair).unwrap();
    assert_eq!(
        decl.fields,
        vec![FieldTy::Data(byte_decl.clone()), FieldTy::Data(byte_decl)]
    );
}

#[test]
fn ctor_ref_display_is_hash_prefixed_and_non_empty() {
    let nat = DataRegistry::build(&nat_spec()).expect("builds");
    let z = nat.ctor_ref("Nat", 0).expect("Z constructor");
    let s_ref = format!("{z}");
    assert!(!s_ref.is_empty(), "CtorRef Display must not be empty");
    assert!(
        s_ref.starts_with('#'),
        "CtorRef Display must start with '#': got {s_ref:?}"
    );
    let s = nat.ctor_ref("Nat", 1).expect("S constructor");
    let s_str = format!("{s}");
    assert_ne!(s_ref, s_str, "Z and S CtorRef must display differently");
}

#[test]
fn registry_error_display_is_non_empty() {
    let err = RegistryError::UnknownTypeRef {
        in_decl: "Tree".to_owned(),
        missing: "Forest".to_owned(),
    };
    let msg = format!("{err}");
    assert!(!msg.is_empty(), "RegistryError Display must not be empty");
    assert!(
        msg.contains("Tree"),
        "must mention the declaring type: {msg:?}"
    );
    assert!(
        msg.contains("Forest"),
        "must mention the missing type: {msg:?}"
    );
}

#[test]
fn ctor_ref_out_of_range_index_returns_none() {
    let nat = DataRegistry::build(&nat_spec()).expect("builds");
    assert!(nat.ctor_ref("Nat", 0).is_some(), "index 0 must exist");
    assert!(nat.ctor_ref("Nat", 1).is_some(), "index 1 must exist");
    assert_eq!(
        nat.ctor_ref("Nat", 2),
        None,
        "index 2 (== ctor count) must return None"
    );
}

#[test]
fn decl_returns_some_for_registered_hash() {
    let nat = DataRegistry::build(&nat_spec()).expect("builds");
    let hash = nat.decl_hash("Nat").expect("Nat is registered");
    let decl = nat.decl(hash);
    assert!(
        decl.is_some(),
        "decl() must return Some for a registered hash"
    );
    assert_eq!(decl.unwrap().ctors.len(), 2, "Nat must have 2 constructors");
}

// ---------------------------------------------------------------------------
// ADR-033 §10 PATH-A tests — FLAG-1 fix (moves guarantee from Declared → Empirical)
// Tag: Empirical — tested below; NOT Proven (no mechanized injectivity proof, VR-5).
// ---------------------------------------------------------------------------

/// Property 1/2 (ADR-033 §5.2 extended): two declarations differing *only* in the `Repr`s of a
/// `Fn` field's parameters produce **distinct** content hashes.
/// Closes the soundness gap from ADR-033 §10.1: `MkDict_Eq8 ≠ MkDict_Eq16`.
#[test]
fn fn_distinct_param_reprs_hash_distinctly() {
    // MkDict_Eq8  has a Fn field: (Binary{8},  Binary{8})  -> Binary{1}
    // MkDict_Eq16 has a Fn field: (Binary{16}, Binary{16}) -> Binary{1}
    let mk = |param_w: u32| {
        let mut m = BTreeMap::new();
        m.insert(
            "Dict".to_owned(),
            single_ctor(vec![FieldSpec::Fn {
                arity: 2,
                sig: FnSig {
                    arity: 2,
                    params: vec![
                        FieldTyRef::Repr(Repr::Binary { width: param_w }),
                        FieldTyRef::Repr(Repr::Binary { width: param_w }),
                    ],
                    ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
                },
            }]),
        );
        DataRegistry::build(&m).unwrap()
    };
    let eq8 = mk(8);
    let eq16 = mk(16);
    assert_ne!(
        eq8.decl_hash("Dict"),
        eq16.decl_hash("Dict"),
        "MkDict_Eq8 and MkDict_Eq16 must have distinct content hashes (ADR-033 §10.1 fix)"
    );
}

/// Property 2: two declarations differing *only* in the return type of a `Fn` field produce
/// distinct hashes. Covers the Q1 decision from ADR-033 §10.6: return type IS encoded.
#[test]
fn fn_distinct_return_reprs_hash_distinctly() {
    let mk = |ret_w: u32| {
        let mut m = BTreeMap::new();
        m.insert(
            "F".to_owned(),
            single_ctor(vec![fn_spec_repr_only(
                Repr::Binary { width: 8 },
                Repr::Binary { width: ret_w },
                1,
            )]),
        );
        DataRegistry::build(&m).unwrap()
    };
    assert_ne!(
        mk(1).decl_hash("F"),
        mk(8).decl_hash("F"),
        "distinct return types must produce distinct hashes"
    );
}

/// `FIELD_FN` tag does not collide with `FIELD_REPR` (ADR-033 §5.2 property 2 / §2.3).
/// A declaration with a `Fn { arity:1, Binary{8}->Binary{8} }` field and one with a bare
/// `Repr(Binary{8})` field must hash differently.
#[test]
fn fn_field_vs_repr_field_hashes_distinctly() {
    let fn_decl = {
        let mut m = BTreeMap::new();
        m.insert(
            "T".to_owned(),
            single_ctor(vec![fn_spec_repr_only(
                Repr::Binary { width: 8 },
                Repr::Binary { width: 8 },
                1,
            )]),
        );
        DataRegistry::build(&m).unwrap()
    };
    let repr_decl = {
        let mut m = BTreeMap::new();
        m.insert(
            "T".to_owned(),
            single_ctor(vec![FieldSpec::Repr(Repr::Binary { width: 8 })]),
        );
        DataRegistry::build(&m).unwrap()
    };
    assert_ne!(
        fn_decl.decl_hash("T"),
        repr_decl.decl_hash("T"),
        "FIELD_FN tag must not collide with FIELD_REPR"
    );
}

/// Same fn signature → same hash (determinism / hash stability).
#[test]
fn fn_same_signature_hashes_stably() {
    let mk = || {
        let mut m = BTreeMap::new();
        m.insert(
            "D".to_owned(),
            single_ctor(vec![fn_spec_repr_only(
                Repr::Binary { width: 8 },
                Repr::Binary { width: 1 },
                1,
            )]),
        );
        DataRegistry::build(&m).unwrap()
    };
    assert_eq!(
        mk().decl_hash("D"),
        mk().decl_hash("D"),
        "same fn signature must always produce the same hash"
    );
}

/// `arity ≠ params.len()` is an explicit error — never silent (G2).
#[test]
fn fn_arity_mismatch_is_explicit_error() {
    let mut m = BTreeMap::new();
    m.insert(
        "F".to_owned(),
        single_ctor(vec![FieldSpec::Fn {
            arity: 2, // stated arity: 2
            sig: FnSig {
                arity: 2,
                params: vec![FieldTyRef::Repr(Repr::Binary { width: 8 })], // only 1 param
                ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
            },
        }]),
    );
    let err = DataRegistry::build(&m).unwrap_err();
    match err {
        RegistryError::FnArityMismatch {
            in_decl,
            arity,
            params_len,
        } => {
            assert_eq!(in_decl, "F");
            assert_eq!(arity, 2);
            assert_eq!(params_len, 1);
        }
        other => panic!("expected FnArityMismatch, got {other:?}"),
    }
}

/// `FnArityMismatch` Display is non-empty and mentions the key values (G2).
#[test]
fn fn_arity_mismatch_display_is_informative() {
    let err = RegistryError::FnArityMismatch {
        in_decl: "Foo".to_owned(),
        arity: 3,
        params_len: 1,
    };
    let msg = format!("{err}");
    assert!(!msg.is_empty());
    assert!(msg.contains("Foo"), "must name the decl: {msg:?}");
    assert!(msg.contains('3'), "must name the stated arity: {msg:?}");
    assert!(msg.contains('1'), "must name params_len: {msg:?}");
}

/// A dangling `FieldTyRef::Data` inside a `Fn` signature is an explicit error.
#[test]
fn fn_dangling_sig_ref_is_explicit_error() {
    let mut m = BTreeMap::new();
    m.insert(
        "Dict".to_owned(),
        single_ctor(vec![FieldSpec::Fn {
            arity: 1,
            sig: FnSig {
                arity: 1,
                params: vec![FieldTyRef::Data("Unknown".to_owned())],
                ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
            },
        }]),
    );
    assert_eq!(
        DataRegistry::build(&m).unwrap_err(),
        RegistryError::UnknownTypeRef {
            in_decl: "Dict".to_owned(),
            missing: "Unknown".to_owned(),
        }
    );
}

/// FLAG-3 (ADR-033 §10.2 Q3): a `Fn` field whose parameter type is the enclosing self-recursive
/// data declaration must use the in-cycle placeholder — it must hash without looping.
/// Specifically: `type SelfDict = MkSelfDict(Fn { (SelfDict) -> Binary{1} })`.
#[test]
fn fn_sig_data_ref_self_recursive_hashes_without_looping() {
    let mut m = BTreeMap::new();
    m.insert(
        "SelfDict".to_owned(),
        single_ctor(vec![FieldSpec::Fn {
            arity: 1,
            sig: FnSig {
                arity: 1,
                params: vec![FieldTyRef::Data("SelfDict".to_owned())],
                ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
            },
        }]),
    );
    // Must build without panic/loop (the in-cycle placeholder is used for the Data leaf).
    let reg = DataRegistry::build(&m).expect("builds without looping");
    let ctor = reg.ctor_ref("SelfDict", 0).expect("constructor 0");
    assert_eq!(reg.field_count(&ctor), Some(1));
    // The resolved field must be a Fn variant (not Data or Repr).
    let decl = reg.ctor(&ctor).unwrap();
    match &decl.fields[0] {
        FieldTy::Fn { arity, sig } => {
            assert_eq!(*arity, 1);
            // The resolved param is the SelfDict hash (now known after hashing).
            let hash = reg.decl_hash("SelfDict").unwrap();
            assert_eq!(sig.params[0], ResolvedFieldTyRef::Data(hash.clone()));
        }
        other => panic!("expected FieldTy::Fn, got {other:?}"),
    }
}

/// A `FieldTyRef::Data` out-of-cycle inside a `Fn` signature resolves to the referenced decl's
/// hash (FLAG-3 Path A, non-cycle branch).
#[test]
fn fn_sig_data_ref_out_of_cycle_resolves() {
    // type Byte = MkByte(Binary{8});  type F = MkF(Fn { (Byte) -> Binary{1} })
    let mut m = BTreeMap::new();
    m.insert(
        "Byte".to_owned(),
        single_ctor(vec![FieldSpec::Repr(Repr::Binary { width: 8 })]),
    );
    m.insert(
        "F".to_owned(),
        single_ctor(vec![FieldSpec::Fn {
            arity: 1,
            sig: FnSig {
                arity: 1,
                params: vec![FieldTyRef::Data("Byte".to_owned())],
                ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
            },
        }]),
    );
    let reg = DataRegistry::build(&m).expect("builds");
    let byte_hash = reg.decl_hash("Byte").unwrap().clone();
    let f_ctor = reg.ctor_ref("F", 0).expect("MkF");
    let decl = reg.ctor(&f_ctor).unwrap();
    match &decl.fields[0] {
        FieldTy::Fn { arity, sig } => {
            assert_eq!(*arity, 1);
            assert_eq!(sig.params[0], ResolvedFieldTyRef::Data(byte_hash));
        }
        other => panic!("expected FieldTy::Fn, got {other:?}"),
    }
}

/// Higher-order (nested `FieldTyRef::Fn`) hashing: two declarations with nested fn types that
/// differ in an inner param must hash distinctly.
#[test]
fn fn_nested_sig_distinct_hashes() {
    // type F8  = MkF(Fn { (Fn { Binary{8}  -> Binary{1} }) -> Binary{1} })
    // type F16 = MkF(Fn { (Fn { Binary{16} -> Binary{1} }) -> Binary{1} })
    let mk = |inner_w: u32| {
        let mut m = BTreeMap::new();
        m.insert(
            "F".to_owned(),
            single_ctor(vec![FieldSpec::Fn {
                arity: 1,
                sig: FnSig {
                    arity: 1,
                    params: vec![FieldTyRef::Fn(Box::new(FnSig {
                        arity: 1,
                        params: vec![FieldTyRef::Repr(Repr::Binary { width: inner_w })],
                        ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
                    }))],
                    ret: Box::new(FieldTyRef::Repr(Repr::Binary { width: 1 })),
                },
            }]),
        );
        DataRegistry::build(&m).unwrap()
    };
    assert_ne!(
        mk(8).decl_hash("F"),
        mk(16).decl_hash("F"),
        "nested Fn types with distinct inner param reprs must hash distinctly"
    );
}

/// The `FieldSpec::Fn` field does *not* create a declaration-level dependency (it references
/// function values, not declarations) — unless its signature contains a `Data` leaf that names
/// a declaration. A `Fn`-only declaration with `Repr` leaves only is a singleton SCC.
/// (This guards FLAG-3: only `Data` inside a sig creates edges, not `Fn { Repr }` itself.)
#[test]
fn fn_repr_only_sig_does_not_create_declaration_dep() {
    // type F = MkF(Fn { (Binary{8}) -> Binary{1} }) — no Data refs anywhere.
    // The SCC algorithm must not create any dependency edge here.
    let mut m = BTreeMap::new();
    m.insert(
        "F".to_owned(),
        single_ctor(vec![fn_spec_repr_only(
            Repr::Binary { width: 8 },
            Repr::Binary { width: 1 },
            1,
        )]),
    );
    // Should build without issue (singleton SCC).
    let reg = DataRegistry::build(&m).expect("builds");
    assert!(reg.decl_hash("F").is_some());
}
