//! White-box tests for [`crate::content`]. Extracted from the logic file (test-layout rule, M-797).

use crate::content::Names;
use crate::id::ContentHash;
use crate::meta::{Meta, Provenance};
use crate::node::Node;
use crate::repr::{Repr, ScalarKind};
use crate::value::{Payload, Value};

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed byte")
}

const B: [bool; 8] = [true, false, true, true, false, false, true, false];

fn swap_def(binder: &str) -> Node {
    let policy = ContentHash::parse("blake3:round_trip_safe").expect("hash");
    Node::Let {
        id: binder.to_owned(),
        bound: Box::new(Node::Const(byte(B))),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Var(binder.to_owned())),
            target: Repr::Ternary { trits: 6 },
            policy,
        }),
    }
}

#[test]
fn hash_is_well_shaped_blake3() {
    let h = swap_def("a").content_hash();
    assert_eq!(h.algo(), "blake3");
    assert_eq!(h.digest().len(), 64); // BLAKE3 → 32 bytes → 64 hex chars
    assert!(ContentHash::parse(h.as_str()).is_some());
}

#[test]
fn identical_defs_collide() {
    assert_eq!(swap_def("a").content_hash(), swap_def("a").content_hash());
}

#[test]
fn trivial_renames_do_not_change_identity() {
    // Same structure, different binder name (and matching bound-var use) → α-equivalent.
    assert_eq!(
        swap_def("a").content_hash(),
        swap_def("longer_name").content_hash(),
        "α-renaming a binder must not change identity (RFC-0001 §4.6)"
    );
}

#[test]
fn dynamic_metadata_is_not_hashed() {
    // Two constants with identical repr+payload but different provenance must collide.
    let exact = byte(B);
    let derived = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(B.to_vec()),
        Meta::new(
            Provenance::Derived {
                op: ContentHash::parse("blake3:some_op").unwrap(),
                inputs: vec![],
            },
            crate::guarantee::GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .expect("exact meta"),
    )
    .expect("well-formed");
    assert_eq!(exact.content_hash(), derived.content_hash());
    assert_eq!(
        Node::Const(exact).content_hash(),
        Node::Const(derived).content_hash()
    );
}

#[test]
fn cert_mode_is_excluded_from_the_content_hash() {
    // RFC-0034 §3.1 / ADR-003 (M-786): the certification mode rides `Meta` (dynamic metadata), so
    // switching it must never perturb a value's content identity (RFC-0001 §4.6). Exhaustive over
    // the finite mode space — a complete check, not sampling.
    use crate::cert_mode::CertMode;
    let mk = |mode: CertMode| {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(B.to_vec()),
            Meta::exact(Provenance::Root).with_cert_mode(mode),
        )
        .expect("well-formed")
    };
    for mode in CertMode::ALL {
        assert_eq!(
            mk(CertMode::Fast).content_hash(),
            mk(mode).content_hash(),
            "cert_mode must not change a value's content identity (ADR-003)"
        );
        assert_eq!(
            Node::Const(mk(CertMode::Fast)).content_hash(),
            Node::Const(mk(mode)).content_hash(),
            "cert_mode must not change a definition's content identity (RFC-0001 §4.6)"
        );
    }
}

#[test]
fn paradigm_change_changes_identity() {
    // A definition differing only in representation paradigm gets a different hash (§4.6).
    let bin = Node::Const(byte(B));
    let tern = Node::Const(
        Value::new(
            Repr::Ternary { trits: 6 },
            Payload::Trits(vec![crate::value::Trit::Zero; 6]),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed"),
    );
    assert_ne!(bin.content_hash(), tern.content_hash());
}

#[test]
fn distinct_literals_differ() {
    let mut flipped = B;
    flipped[0] = !flipped[0];
    assert_ne!(
        Node::Const(byte(B)).content_hash(),
        Node::Const(byte(flipped)).content_hash()
    );
}

#[test]
fn scalar_precision_is_identity_bearing() {
    // Dense{dim, F32} and Dense{dim, F64} are distinct types (precision bounds error).
    let f32v = Node::Const(
        Value::new(
            Repr::Dense {
                dim: 2,
                dtype: ScalarKind::F32,
            },
            Payload::Scalars(vec![1.0, 2.0]),
            Meta::exact(Provenance::Root),
        )
        .unwrap(),
    );
    let f64v = Node::Const(
        Value::new(
            Repr::Dense {
                dim: 2,
                dtype: ScalarKind::F64,
            },
            Payload::Scalars(vec![1.0, 2.0]),
            Meta::exact(Provenance::Root),
        )
        .unwrap(),
    );
    assert_ne!(f32v.content_hash(), f64v.content_hash());
}

#[test]
fn op_operator_name_is_identity_bearing() {
    let add = Node::Op {
        prim: "add_binary".to_owned(),
        args: vec![Node::Const(byte(B))],
    };
    let sub = Node::Op {
        prim: "sub_binary".to_owned(),
        args: vec![Node::Const(byte(B))],
    };
    assert_ne!(add.content_hash(), sub.content_hash());
}

#[test]
fn free_variables_keep_their_names() {
    // Distinct free names are distinct contracts → distinct identity (not α-renamable).
    assert_ne!(
        Node::Var("x".to_owned()).content_hash(),
        Node::Var("y".to_owned()).content_hash()
    );
}

#[test]
fn names_are_metadata_outside_identity() {
    // The same definition can carry different human names; identity is unchanged.
    let h = swap_def("a").content_hash();
    let mut names = Names::new();
    assert!(names.is_empty());
    assert_eq!(names.bind(h.clone(), "to_ternary"), None);
    assert_eq!(names.name_of(&h), Some("to_ternary"));
    // Re-binding a new name does not (and cannot) change the hash.
    assert_eq!(
        names.bind(h.clone(), "as_balanced"),
        Some("to_ternary".into())
    );
    assert_eq!(names.name_of(&h), Some("as_balanced"));
    assert_eq!(swap_def("renamed_binder").content_hash(), h);
    assert_eq!(names.len(), 1);
}

// Mutant-witnesses for Canon::f64 (content.rs:131:9 → `()`): if f64 is a no-op,
// all Dense Scalar values with different float payloads hash identically.
#[test]
fn f64_is_included_in_scalar_payload_hash() {
    use crate::repr::ScalarKind;
    use crate::value::Payload;
    let v1 = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let v2 = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![2.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    // Different float payloads must yield different hashes.
    // If Canon::f64 is a no-op, both would hash identically.
    assert_ne!(
        v1.content_hash(),
        v2.content_hash(),
        "Dense values with different float payloads must have different content hashes"
    );
}

// Mutant-witness for Canon::ctor_ref (content.rs:148:9 → `()`): if ctor_ref is a no-op,
// two Construct nodes using different constructors would hash identically.
#[test]
fn ctor_ref_is_included_in_construct_node_hash() {
    use crate::data::{CtorSpec, DataRegistry, DeclSpec};
    use std::collections::BTreeMap;
    let mut m = BTreeMap::new();
    m.insert(
        "Bool".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] }, // False
                CtorSpec { fields: vec![] }, // True
            ],
        },
    );
    let reg = DataRegistry::build(&m).unwrap();
    let false_ref = reg.ctor_ref("Bool", 0).unwrap();
    let true_ref = reg.ctor_ref("Bool", 1).unwrap();
    let false_node = Node::Construct {
        ctor: false_ref,
        args: vec![],
    };
    let true_node = Node::Construct {
        ctor: true_ref,
        args: vec![],
    };
    // Different constructors (same decl, different index) must yield different hashes.
    // If Canon::ctor_ref is a no-op, both hash to the same value.
    assert_ne!(
        false_node.content_hash(),
        true_node.content_hash(),
        "Construct nodes with different ctors must have different content hashes"
    );
}

// Mutant-witness for Canon::prim_paradigm (content.rs:175:9 → `()`): if prim_paradigm
// is a no-op, prims with different paradigms hash identically.
#[test]
fn prim_paradigm_is_included_in_prim_decl_hash() {
    use crate::guarantee::GuaranteeStrength;
    use crate::prim::{PrimDecl, PrimParadigm, PrimSig, WidthRel};
    let bin_sig = PrimSig {
        operands: vec![PrimParadigm::Binary],
        result: PrimParadigm::Binary,
        width: WidthRel::Uniform,
    };
    let tern_sig = PrimSig {
        operands: vec![PrimParadigm::Ternary],
        result: PrimParadigm::Ternary,
        width: WidthRel::Uniform,
    };
    let bin_decl = PrimDecl {
        sig: bin_sig,
        intrinsic: GuaranteeStrength::Exact,
    };
    let tern_decl = PrimDecl {
        sig: tern_sig,
        intrinsic: GuaranteeStrength::Exact,
    };
    // Different paradigms must yield different hashes.
    // If Canon::prim_paradigm is a no-op, both hash identically.
    assert_ne!(
        bin_decl.content_hash(),
        tern_decl.content_hash(),
        "PrimDecls with different paradigms must have different content hashes"
    );
}

// Mutant-witness for Canon::prim_decl's WidthRel arm (content.rs: the `WidthRel::Collapse` tag,
// RFC-0032 D1): if the width relation is dropped or both arms emit the same tag, a width-`Uniform`
// decl and an otherwise-identical width-`Collapse` decl hash the same — and a collapsing prim
// would alias a uniform one in the content-addressed registry.
#[test]
fn prim_decl_width_rel_is_included_in_hash() {
    use crate::guarantee::GuaranteeStrength;
    use crate::prim::{PrimDecl, PrimParadigm, PrimSig, WidthRel};
    // Identical sig + intrinsic; the ONLY difference is the width relation.
    let mk = |width| PrimDecl {
        sig: PrimSig {
            operands: vec![PrimParadigm::Binary, PrimParadigm::Binary],
            result: PrimParadigm::Binary,
            width,
        },
        intrinsic: GuaranteeStrength::Exact,
    };
    assert_ne!(
        mk(WidthRel::Uniform).content_hash(),
        mk(WidthRel::Collapse).content_hash(),
        "PrimDecls differing only in WidthRel (Uniform vs Collapse) must hash differently"
    );
}

// Mutant-witnesses for de Bruijn arithmetic (content.rs:287:50, 287:54 → `+` or `/`):
// The subtraction `scope.len() - 1 - pos` computes the de Bruijn index.
// - For a single binder, pos=0 and scope.len()=1, so 1-1-0=0 always (mutations also give 0).
// - For a deeper binder (pos>0), the mutation produces a wrong index:
//   scope.len()=2, pos=1: correct=2-1-1=0; mutant `+pos`=2-1+1=2 (wrong).
// Test: let x = let y = Var("x") in y in x — outer let binds "x" at pos=1 in inner scope.
#[test]
fn de_bruijn_correctly_resolves_outer_binder_in_nested_scope() {
    // Build: let x = const in let y = const in Var("x")
    // When hashing Var("x"), scope = ["x", "y"], pos = 0 (x's index in scope), len = 2.
    // de Bruijn for x = 2 - 1 - 0 = 1 (x is 1 binder deep from the inner let).
    // vs let x = const in let y = const in Var("y")
    // de Bruijn for y = 2 - 1 - 1 = 0 (y is bound immediately).
    // These should hash DIFFERENTLY; α-renaming both lets should NOT change which is which.
    let node_ref_outer = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(byte(B))),
        body: Box::new(Node::Let {
            id: "y".into(),
            bound: Box::new(Node::Const(byte(B))),
            body: Box::new(Node::Var("x".into())), // references outer let
        }),
    };
    let node_ref_inner = Node::Let {
        id: "x".into(),
        bound: Box::new(Node::Const(byte(B))),
        body: Box::new(Node::Let {
            id: "y".into(),
            bound: Box::new(Node::Const(byte(B))),
            body: Box::new(Node::Var("y".into())), // references inner let
        }),
    };
    // Referencing outer binder vs inner binder must yield different hashes.
    // If de Bruijn arithmetic is wrong (+ instead of -), pos=0 gives index 2 for x and
    // pos=1 gives 2 for y too, potentially causing false collisions or wrong renaming.
    assert_ne!(
        node_ref_outer.content_hash(),
        node_ref_inner.content_hash(),
        "Nodes referencing outer vs inner binder must have different content hashes"
    );

    // α-equivalence: renaming the binders doesn't change which variable is referenced.
    let node_ref_outer_renamed = Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte(B))),
        body: Box::new(Node::Let {
            id: "b".into(),
            bound: Box::new(Node::Const(byte(B))),
            body: Box::new(Node::Var("a".into())), // still references outer
        }),
    };
    assert_eq!(
        node_ref_outer.content_hash(),
        node_ref_outer_renamed.content_hash(),
        "α-equivalent nodes (same outer binding reference) must hash identically"
    );
}

// Mutant-witnesses for Names::len (content.rs:481:9 → 1) and
// Names::is_empty (content.rs:487:9 → true):
// - Names::len → 1: a fresh Names.len() would return 1 (wrong — it's 0).
// - Names::is_empty → true: a Names with 2 entries would report empty (wrong).
#[test]
fn names_len_and_is_empty_reflect_actual_count() {
    let mut names = Names::new();
    // Empty table: len=0, is_empty=true.
    assert_eq!(
        names.len(),
        0,
        "empty Names must have len=0 (not 1 constant)"
    );
    assert!(names.is_empty(), "empty Names must be is_empty=true");

    // Use two Const nodes with different payloads — guaranteed distinct hashes.
    use crate::value::Payload;
    let v1 = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let v2 = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true; 8]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let h1 = Node::Const(v1).content_hash();
    let h2 = Node::Const(v2).content_hash();
    assert_ne!(
        h1, h2,
        "must be distinct hashes for the test to be meaningful"
    );

    names.bind(h1, "name1");
    assert_eq!(names.len(), 1, "after 1 bind: len=1");
    assert!(!names.is_empty(), "after 1 bind: not empty");

    names.bind(h2, "name2");
    assert_eq!(names.len(), 2, "after 2 binds: len=2 (not 1 constant)");
    assert!(
        !names.is_empty(),
        "after 2 binds: is_empty=false (not true constant)"
    );
}

// --- ADR-040 §3 (M-896): scalar-float identity + the NO-REHASH address-stability regression ------

use crate::repr::{FloatWidth, SparsityClass};
use crate::value::{Trit, CANONICAL_NAN_BITS};

fn val(repr: Repr, payload: Payload) -> Value {
    Value::new(repr, payload, Meta::exact(Provenance::Root)).expect("well-formed")
}

fn float_val(x: f64) -> Value {
    val(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(x),
    )
}

/// **The content-address note made checkable — NO rehash occurred (ADR-040 §3; RFC-0033 §7).**
/// These digests were computed on the pre-`Repr::Float` base (dev @ 942770c, 2026-07-02) for one
/// representative value of every pre-existing payload form, a composite node, and a prim
/// operation hash. Adding the `Float` arm to the prefix-tagged `Canon` encoder must leave every
/// one of them byte-identical — the frozen-tag append-only guarantee. A failing run here means an
/// existing identity shifted: a rehash was spent, which ADR-040 defers to E20-1. `Exact` (a pinned
/// equality over fixed inputs).
#[test]
fn adding_float_spent_no_rehash_existing_addresses_stable() {
    let binary = val(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
    );
    assert_eq!(
        binary.content_hash().as_str(),
        "blake3:9c2cfd3d03f00ca309eb0be84d4c948569ae4ad1cacdee052b5fe3f528170bc0"
    );

    let ternary = val(
        Repr::Ternary { trits: 4 },
        Payload::Trits(vec![Trit::Neg, Trit::Zero, Trit::Pos, Trit::Zero]),
    );
    assert_eq!(
        ternary.content_hash().as_str(),
        "blake3:86d53f1cf885cc00c8e772879c96f83984a4ff72bd5f7840332bcbe83866f1d5"
    );

    // NOTE the raw-bits NaN in this Dense payload: the existing `Canon::f64` paths are NOT
    // canonicalized by M-896 (ADR-040 FLAG-5 — the uniform rule rides the E20-1 settlement), so
    // this digest pins that they were left untouched.
    let dense = val(
        Repr::Dense {
            dim: 3,
            dtype: ScalarKind::F64,
        },
        Payload::Scalars(vec![1.5, -0.0, f64::from_bits(0x7ff8_0000_0000_0000)]),
    );
    assert_eq!(
        dense.content_hash().as_str(),
        "blake3:44b8877cfa8568cf751a4c5b91725334d2d57b6d5673e2deaf2eb420599ea93e"
    );

    let vsa = val(
        Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim: 4,
            sparsity: SparsityClass::Sparse { max_active: 2 },
        },
        Payload::Hypervector(vec![0.25, -1.0, 0.0, 2.0]),
    );
    assert_eq!(
        vsa.content_hash().as_str(),
        "blake3:87156d3912732cdf3305d9891ee80fbd04155f0c764fe635228c1771f45d281b"
    );

    let elem = Repr::Binary { width: 2 };
    let seq = val(
        Repr::Seq {
            elem: Box::new(elem.clone()),
            len: 2,
        },
        Payload::Seq(vec![
            val(elem.clone(), Payload::Bits(vec![true, false])),
            val(elem, Payload::Bits(vec![false, true])),
        ]),
    );
    assert_eq!(
        seq.content_hash().as_str(),
        "blake3:76953731fd6aaa0e319911bd90bfc67f63dfd92fdad960f160ac4aa7f1523b2d"
    );

    let bytes = val(Repr::Bytes, Payload::Bytes(vec![0xde, 0xad, 0xbe, 0xef]));
    assert_eq!(
        bytes.content_hash().as_str(),
        "blake3:43a189a6d443e741f41503f07a563693907f3132660bb9833c22c9c6e96f4681"
    );

    let node = Node::Let {
        id: "x".to_owned(),
        bound: Box::new(Node::Const(val(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        ))),
        body: Box::new(Node::Op {
            prim: "band".to_owned(),
            args: vec![Node::Var("x".to_owned()), Node::Var("x".to_owned())],
        }),
    };
    assert_eq!(
        node.content_hash().as_str(),
        "blake3:08792f5cde75d318ddcc90f15d62b124dc2645d590c69a3fb4be2f08688762e3"
    );

    assert_eq!(
        crate::content::operation_hash("band").as_str(),
        "blake3:20e9c439d4a46280150a666b91603efeb76348926b5dc60dc139b88f5788c7b9"
    );
}

/// Every NaN is ONE content address (ADR-040 §2.3): quiet/signaling, any payload, either sign —
/// all collide with the canonical quiet NaN. Identity never forks on platform NaN bits.
#[test]
fn every_nan_is_one_content_address() {
    let canonical = float_val(f64::from_bits(CANONICAL_NAN_BITS)).content_hash();
    for bits in [
        0x7ff8_0000_0000_0001_u64, // quiet, non-zero payload
        0xfff8_0000_0000_0000,     // quiet, sign bit set
        0x7ff0_0000_0000_0001,     // signaling
        0xfff7_ffff_ffff_ffff,     // signaling, sign bit set, max payload
    ] {
        assert_eq!(
            float_val(f64::from_bits(bits)).content_hash(),
            canonical,
            "NaN bits {bits:#018x} forked identity"
        );
    }
}

/// `+0.0` and `-0.0` are TWO content addresses (ADR-040 §2.3: observably distinct values are never
/// aliased), even though they are IEEE-equal.
#[test]
fn signed_zeros_are_two_content_addresses() {
    assert_ne!(
        float_val(0.0).content_hash(),
        float_val(-0.0).content_hash()
    );
}

/// The scalar float is a DISTINCT identity from every same-bits encoding in another paradigm —
/// no implicit scalar↔rank-0-tensor identification (ADR-040 §5): `Float(1.5)` ≠ `Dense{1,F64}[1.5]`.
#[test]
fn float_identity_distinct_from_dense_dim1() {
    let dense1 = val(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64,
        },
        Payload::Scalars(vec![1.5]),
    );
    assert_ne!(float_val(1.5).content_hash(), dense1.content_hash());
}

/// Distinct finite floats get distinct addresses; identical floats collide (determinism).
#[test]
fn float_addresses_are_deterministic_and_injective_on_bits() {
    assert_eq!(float_val(1.5).content_hash(), float_val(1.5).content_hash());
    assert_ne!(float_val(1.5).content_hash(), float_val(2.5).content_hash());
    // The specials are in-band, inspectable — and distinct identities (ADR-040 §2.4).
    assert_ne!(
        float_val(f64::INFINITY).content_hash(),
        float_val(f64::NEG_INFINITY).content_hash()
    );
}
