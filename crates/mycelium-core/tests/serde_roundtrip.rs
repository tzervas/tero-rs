//! M-104 — Core IR (de)serialization round-trips and schema-shape pinning.
//!
//! Two guarantees are checked here:
//!
//! 1. **Faithful round-trip** (RFC-0001 §4.8): `from_json(to_json(v)) == v`, including `Meta`, over
//!    a corpus spanning all four paradigms, every guarantee level, every bound kind/basis, every
//!    physical layout, and both provenance variants.
//! 2. **Schema agreement**: the serializer's output for representative values is *exactly* the
//!    committed `docs/spec/schemas/examples/value/valid/*.json` instances — and those files are what
//!    CI validates against `value.schema.json` (`scripts/checks/schema.sh`). So pinning emitted
//!    output to those files is what ties "the code emits schema-valid JSON" to a checked artifact.

use mycelium_core::bound::{Bound, BoundBasis, BoundKind, NormKind};
use mycelium_core::guarantee::GuaranteeStrength;
use mycelium_core::id::ContentHash;
use mycelium_core::meta::{Meta, PackScheme, PhysicalLayout, Provenance, SparsityObs};
use mycelium_core::repr::{Repr, ScalarKind, SparsityClass};
use mycelium_core::value::{Payload, Trit, Value};

fn hash(s: &str) -> ContentHash {
    ContentHash::parse(s).expect("valid content hash")
}

/// A spread of `(guarantee, bound)` pairs that satisfy M-I1…M-I4 — one per basis kind, plus Exact.
fn meta_variants() -> Vec<Meta> {
    let derived = Provenance::Derived {
        op: hash("blake3:op01"),
        inputs: vec![hash("blake3:in_a"), hash("blake3:in_b")],
    };
    vec![
        // Exact, no bound (M-I1).
        Meta::exact(Provenance::Root),
        // Proven + ProvenThm capacity bound (M-I2), with rich optional fields.
        Meta::new(
            derived.clone(),
            GuaranteeStrength::Proven,
            Some(Bound {
                kind: BoundKind::Capacity {
                    items: 3,
                    dim: 10_000,
                },
                basis: BoundBasis::ProvenThm {
                    citation: "Clarkson-Ubaru-Yang 2023".into(),
                },
            }),
            Some(SparsityObs {
                active: 97,
                density: 0.0097,
            }),
            Some(PhysicalLayout::VsaStore { sparse: true }),
            Some(hash("blake3:policy_x9")),
        )
        .unwrap(),
        // Empirical + EmpiricalFit error bound (M-I3), with a packed physical layout.
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Empirical,
            Some(Bound {
                kind: BoundKind::Error {
                    eps: 0.004,
                    norm: NormKind::L2,
                },
                basis: BoundBasis::EmpiricalFit {
                    trials: 10_000,
                    method: "Frady-Sommer Gaussian".into(),
                },
            }),
            None,
            Some(PhysicalLayout::TritPacked {
                scheme: PackScheme::Tl2,
            }),
            None,
        )
        .unwrap(),
        // Declared + UserDeclared probability bound (M-I4).
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(Bound {
                kind: BoundKind::Probability { delta: 0.01 },
                basis: BoundBasis::UserDeclared,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
        // Declared crosstalk WITH a tail (exercises the optional field round-trip).
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(Bound {
                kind: BoundKind::Crosstalk {
                    expected: 0.02,
                    tail: Some(0.1),
                },
                basis: BoundBasis::UserDeclared,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
        // Declared crosstalk WITHOUT a tail (the omitted-field path).
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(Bound {
                kind: BoundKind::Crosstalk {
                    expected: 0.02,
                    tail: None,
                },
                basis: BoundBasis::UserDeclared,
            }),
            None,
            Some(PhysicalLayout::DenseArray),
            None,
        )
        .unwrap(),
    ]
}

/// One value per paradigm, each paired with each `Meta` variant — the full round-trip corpus.
fn value_corpus() -> Vec<Value> {
    let reprs_payloads: Vec<(Repr, Payload)> = vec![
        (
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        ),
        (
            Repr::Ternary { trits: 6 },
            Payload::Trits(vec![
                Trit::Zero,
                Trit::Neg,
                Trit::Zero,
                Trit::Zero,
                Trit::Pos,
                Trit::Zero,
            ]),
        ),
        (
            Repr::Dense {
                dim: 3,
                dtype: ScalarKind::Bf16,
            },
            Payload::Scalars(vec![0.5, -1.25, 2.0]),
        ),
        (
            Repr::Vsa {
                model: "MAP-I".into(),
                dim: 4,
                sparsity: SparsityClass::Sparse { max_active: 2 },
            },
            Payload::Hypervector(vec![1.0, 0.0, 0.0, -1.0]),
        ),
        // RFC-0032 D3 (M-749): an indexed homogeneous sequence of `Binary{1}` elements. The wire
        // form carries self-describing element values; faithful round-trip is the test (each
        // element rides its own `Value` (de)serialization). The inner elements carry `Exact` meta;
        // only the *outer* value's `Meta` is varied by the corpus pairing.
        (
            Repr::Seq {
                elem: Box::new(Repr::Binary { width: 1 }),
                len: 2,
            },
            Payload::Seq(vec![
                Value::new(
                    Repr::Binary { width: 1 },
                    Payload::Bits(vec![true]),
                    Meta::exact(Provenance::Root),
                )
                .unwrap(),
                Value::new(
                    Repr::Binary { width: 1 },
                    Payload::Bits(vec![false]),
                    Meta::exact(Provenance::Root),
                )
                .unwrap(),
            ]),
        ),
        // RFC-0032 D4 (M-750): a byte string. The wire form is a lowercase-hex string; faithful
        // round-trip (incl. the outer `Meta` variants) is the test.
        (Repr::Bytes, Payload::Bytes(vec![0xde, 0xad, 0xbe, 0xef])),
    ];
    let mut out = Vec::new();
    for (repr, payload) in reprs_payloads {
        for meta in meta_variants() {
            out.push(Value::new(repr.clone(), payload.clone(), meta).unwrap());
        }
    }
    out
}

#[test]
fn json_round_trip_is_faithful() {
    for v in value_corpus() {
        let json = serde_json::to_string(&v).expect("serialize");
        let back: Value = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            v, back,
            "round-trip must preserve the value incl. Meta\n{json}"
        );
        // Stable: re-serializing the round-tripped value yields byte-identical JSON.
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2, "serialization must be deterministic");
    }
}

#[test]
fn bf16_renders_as_bf16() {
    // The Rust spelling is `Bf16`; the wire/schema spelling is `BF16`.
    let v = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::Bf16,
        },
        Payload::Scalars(vec![1.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let j: serde_json::Value = serde_json::to_value(&v).unwrap();
    assert_eq!(j["repr"]["dtype"], "BF16");
}

/// Pin the serializer's output to the committed, CI-schema-validated example files. Comparing as
/// parsed `serde_json::Value` makes this insensitive to whitespace/key-order formatting.
fn assert_matches_example(value: &Value, example_path: &str) {
    let file = std::fs::read_to_string(example_path)
        .unwrap_or_else(|e| panic!("read {example_path}: {e}"));
    let from_file: serde_json::Value = serde_json::from_str(&file).expect("example parses");
    let emitted: serde_json::Value = serde_json::to_value(value).expect("serialize");
    assert_eq!(
        emitted, from_file,
        "serializer output must equal {example_path} (the schema-validated artifact)"
    );
    // And the committed file must deserialize back into the same value.
    let from_file_val: Value = serde_json::from_str(&file).expect("example deserializes to Value");
    assert_eq!(&from_file_val, value);
}

const EX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/spec/schemas/examples/value/valid"
);

#[test]
fn emitted_value_matches_committed_examples() {
    let binary = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_matches_example(&binary, &format!("{EX}/binary-const.json"));

    let ternary = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![
            Trit::Zero,
            Trit::Neg,
            Trit::Zero,
            Trit::Zero,
            Trit::Pos,
            Trit::Zero,
        ]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_matches_example(&ternary, &format!("{EX}/ternary-const.json"));

    let dense = Value::new(
        Repr::Dense {
            dim: 3,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![0.5, -1.25, 2.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_matches_example(&dense, &format!("{EX}/dense-const.json"));

    let vsa = Value::new(
        Repr::Vsa {
            model: "MAP-I".into(),
            dim: 4,
            sparsity: SparsityClass::Sparse { max_active: 2 },
        },
        Payload::Hypervector(vec![1.0, 0.0, 0.0, -1.0]),
        Meta::new(
            Provenance::Derived {
                op: hash("blake3:bundle_Op01"),
                inputs: vec![hash("blake3:hv_a01"), hash("blake3:hv_b02")],
            },
            GuaranteeStrength::Proven,
            Some(Bound {
                kind: BoundKind::Capacity {
                    items: 3,
                    dim: 10_000,
                },
                basis: BoundBasis::ProvenThm {
                    citation: "Clarkson-Ubaru-Yang 2023".into(),
                },
            }),
            Some(SparsityObs {
                active: 97,
                density: 0.0097,
            }),
            Some(PhysicalLayout::VsaStore { sparse: true }),
            Some(hash("blake3:policy_x9")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_matches_example(&vsa, &format!("{EX}/vsa-proven-capacity.json"));
}

// --- Deserialization is never silently lenient: malformed wire forms are rejected. -------------

#[test]
fn rejects_payload_repr_mismatch() {
    // Binary{8} with a 2-bit payload — caught by Value::new on the way in.
    let json = r#"{ "repr": { "kind": "Binary", "width": 8 }, "payload": { "bits": "10" },
                   "meta": { "provenance": { "kind": "Root" }, "guarantee": "Exact" } }"#;
    assert!(serde_json::from_str::<Value>(json).is_err());
}

#[test]
fn rejects_exact_with_bound() {
    // M-I1 violation re-checked on deserialize.
    let json = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                   "meta": { "provenance": { "kind": "Root" }, "guarantee": "Exact",
                             "bound": { "kind": "ProbabilityBound", "delta": 0.1,
                                        "basis": { "kind": "UserDeclared" } } } }"#;
    assert!(serde_json::from_str::<Value>(json).is_err());
}

#[test]
fn rejects_declared_claiming_proven_basis() {
    // M-I4 violation: Declared cannot carry a ProvenThm basis.
    let json = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                   "meta": { "provenance": { "kind": "Root" }, "guarantee": "Declared",
                             "bound": { "kind": "CapacityBound", "items": 3, "dim": 9,
                                        "basis": { "kind": "ProvenThm", "citation": "x" } } } }"#;
    assert!(serde_json::from_str::<Value>(json).is_err());
}

#[test]
fn rejects_malformed_content_hash() {
    let json = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                   "meta": { "provenance": { "kind": "Derived", "op": "NOT A HASH", "inputs": [] },
                             "guarantee": "Exact" } }"#;
    assert!(serde_json::from_str::<Value>(json).is_err());
}

#[test]
fn rejects_bad_trit_glyph() {
    let json = r#"{ "repr": { "kind": "Ternary", "trits": 2 }, "payload": { "trits": "0x" },
                   "meta": { "provenance": { "kind": "Root" }, "guarantee": "Exact" } }"#;
    assert!(serde_json::from_str::<Value>(json).is_err());
}

/// `deny_unknown_fields` makes the schemas' `additionalProperties: false` a real contract: an
/// unknown field at the value level or the meta level is rejected, not silently dropped (A6-02).
/// Mutant-witness: removing `#[serde(deny_unknown_fields)]` from `ValueWire`/`MetaWire` makes these
/// `from_str` calls succeed.
#[test]
fn rejects_unknown_wire_fields() {
    // Baseline (no extra field) parses.
    let ok = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                  "meta": { "provenance": { "kind": "Root" }, "guarantee": "Exact" } }"#;
    assert!(serde_json::from_str::<Value>(ok).is_ok());
    // Unknown field at the value level.
    let value_extra = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                          "meta": { "provenance": { "kind": "Root" }, "guarantee": "Exact" },
                          "EXTRA_FIELD": true }"#;
    assert!(serde_json::from_str::<Value>(value_extra).is_err());
    // Unknown field at the meta level.
    let meta_extra = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                         "meta": { "provenance": { "kind": "Root" }, "guarantee": "Exact",
                                   "BOGUS": 1 } }"#;
    assert!(serde_json::from_str::<Value>(meta_extra).is_err());
}

/// Pin the exact wire spellings of every bound-kind/basis/layout variant so that enum-spelling
/// drift (e.g. `TL2` → `Tl2x`, `ErrorBound` → `Error`) is caught immediately (A6-03).
///
/// Mutant-witness (A6-03): removing or changing the `#[serde(rename = "TL2")]` attribute on
/// `PackScheme::Tl2` to any other string (e.g. `"Tl2x"`) makes the `scheme == "TL2"` assertion
/// fail — the symmetric round-trip test alone would have passed silently because it re-reads
/// whatever the serializer wrote. This test breaks that blind spot.
#[test]
fn wire_spellings_are_pinned_per_bound_kind_basis_and_layout() {
    // --- BoundKind wire spellings ---

    // ErrorBound + L2 norm + EmpiricalFit basis
    let err_l2 = Bound {
        kind: BoundKind::Error {
            eps: 0.004,
            norm: NormKind::L2,
        },
        basis: BoundBasis::EmpiricalFit {
            trials: 10_000,
            method: "Frady-Sommer Gaussian".into(),
        },
    };
    let j = serde_json::to_value(&err_l2).unwrap();
    assert_eq!(j["kind"], "ErrorBound", "A6-03: ErrorBound kind spelling");
    assert_eq!(j["norm"], "L2", "A6-03: L2 norm spelling");
    assert_eq!(
        j["basis"]["kind"], "EmpiricalFit",
        "A6-03: EmpiricalFit basis spelling"
    );
    assert!(j["basis"]["trials"].is_number(), "A6-03: trials present");

    // ErrorBound + Linf norm (distinct spelling)
    let err_linf = Bound {
        kind: BoundKind::Error {
            eps: 0.1,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    };
    let j = serde_json::to_value(&err_linf).unwrap();
    assert_eq!(j["norm"], "Linf", "A6-03: Linf norm spelling");
    assert_eq!(
        j["basis"]["kind"], "UserDeclared",
        "A6-03: UserDeclared basis spelling"
    );

    // ProbabilityBound + UserDeclared basis
    let prob = Bound {
        kind: BoundKind::Probability { delta: 0.01 },
        basis: BoundBasis::UserDeclared,
    };
    let j = serde_json::to_value(&prob).unwrap();
    assert_eq!(
        j["kind"], "ProbabilityBound",
        "A6-03: ProbabilityBound kind spelling"
    );

    // CrosstalkBound with tail
    let ct_with = Bound {
        kind: BoundKind::Crosstalk {
            expected: 0.02,
            tail: Some(0.1),
        },
        basis: BoundBasis::UserDeclared,
    };
    let j = serde_json::to_value(&ct_with).unwrap();
    assert_eq!(
        j["kind"], "CrosstalkBound",
        "A6-03: CrosstalkBound kind spelling"
    );
    assert!(
        j.get("tail").is_some(),
        "A6-03: tail field present when Some"
    );

    // CrosstalkBound without tail (optional field omitted)
    let ct_no = Bound {
        kind: BoundKind::Crosstalk {
            expected: 0.02,
            tail: None,
        },
        basis: BoundBasis::UserDeclared,
    };
    let j = serde_json::to_value(&ct_no).unwrap();
    assert_eq!(
        j["kind"], "CrosstalkBound",
        "A6-03: CrosstalkBound (no tail) kind spelling"
    );
    assert!(
        j.get("tail").is_none(),
        "A6-03: tail field absent when None"
    );

    // CapacityBound + ProvenThm basis
    let cap = Bound {
        kind: BoundKind::Capacity {
            items: 3,
            dim: 10_000,
        },
        basis: BoundBasis::ProvenThm {
            citation: "Clarkson-Ubaru-Yang 2023".into(),
        },
    };
    let j = serde_json::to_value(&cap).unwrap();
    assert_eq!(
        j["kind"], "CapacityBound",
        "A6-03: CapacityBound kind spelling"
    );
    assert_eq!(
        j["basis"]["kind"], "ProvenThm",
        "A6-03: ProvenThm basis spelling"
    );

    // --- PhysicalLayout / PackScheme wire spellings ---

    // TritPacked + TL2
    let tl2 = PhysicalLayout::TritPacked {
        scheme: PackScheme::Tl2,
    };
    let j = serde_json::to_value(tl2).unwrap();
    assert_eq!(
        j["layout"], "TritPacked",
        "A6-03: TritPacked layout spelling"
    );
    assert_eq!(j["scheme"], "TL2", "A6-03: TL2 scheme spelling");

    // TritPacked + TL1
    let tl1 = PhysicalLayout::TritPacked {
        scheme: PackScheme::Tl1,
    };
    let j = serde_json::to_value(tl1).unwrap();
    assert_eq!(j["scheme"], "TL1", "A6-03: TL1 scheme spelling");

    // DenseArray
    let da = PhysicalLayout::DenseArray;
    let j = serde_json::to_value(da).unwrap();
    assert_eq!(
        j["layout"], "DenseArray",
        "A6-03: DenseArray layout spelling"
    );

    // VsaStore
    let vs = PhysicalLayout::VsaStore { sparse: true };
    let j = serde_json::to_value(vs).unwrap();
    assert_eq!(j["layout"], "VsaStore", "A6-03: VsaStore layout spelling");

    // BinaryWords — the remaining PhysicalLayout variant (Copilot #306).
    let bw = serde_json::to_value(PhysicalLayout::BinaryWords).unwrap();
    assert_eq!(
        bw["layout"], "BinaryWords",
        "A6-03: BinaryWords layout spelling"
    );

    // EVERY PackScheme wire spelling — TL1/TL2 alone left Unpacked / TwoBitPerTrit /
    // FiveTritPerByte / I2S unpinned (Copilot #306). Default serde renders the variant name
    // verbatim except the explicitly-renamed TL1/TL2.
    for (scheme, wire) in [
        (PackScheme::Unpacked, "Unpacked"),
        (PackScheme::TwoBitPerTrit, "TwoBitPerTrit"),
        (PackScheme::FiveTritPerByte, "FiveTritPerByte"),
        (PackScheme::I2S, "I2S"),
        (PackScheme::Tl1, "TL1"),
        (PackScheme::Tl2, "TL2"),
    ] {
        let j = serde_json::to_value(PhysicalLayout::TritPacked { scheme }).unwrap();
        assert_eq!(j["scheme"], wire, "A6-03: {scheme:?} scheme spelling");
    }

    // Compile-enforced exhaustiveness: adding a PackScheme or PhysicalLayout variant breaks these
    // matches, forcing whoever adds it to pin the new wire spelling above — so a spelling can never
    // drift in unnoticed (the standing A6-03 concern, made a build error rather than a runtime gap).
    let _pin_pack = |s: PackScheme| match s {
        PackScheme::Unpacked
        | PackScheme::TwoBitPerTrit
        | PackScheme::FiveTritPerByte
        | PackScheme::I2S
        | PackScheme::Tl1
        | PackScheme::Tl2 => {}
    };
    let _pin_layout = |l: &PhysicalLayout| match l {
        PhysicalLayout::BinaryWords
        | PhysicalLayout::TritPacked { .. }
        | PhysicalLayout::DenseArray
        | PhysicalLayout::VsaStore { .. } => {}
    };
}

/// An `Empirical` guarantee whose bound rests on **zero trials** is evidence-free and must be
/// rejected on the wire, not silently accepted (A6-02/B2-03). Mutant-witness: reverting
/// `Bound::well_formed` to skip the basis check makes the `trials: 0` case parse.
#[test]
fn rejects_evidence_free_empirical_bound() {
    let zero = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                   "meta": { "provenance": { "kind": "Root" }, "guarantee": "Empirical",
                             "bound": { "kind": "ErrorBound", "eps": 0.5, "norm": "Linf",
                                        "basis": { "kind": "EmpiricalFit", "trials": 0,
                                                   "method": "frady" } } } }"#;
    assert!(serde_json::from_str::<Value>(zero).is_err());
    // The same bound backed by real trials deserializes fine.
    let ok = r#"{ "repr": { "kind": "Binary", "width": 1 }, "payload": { "bits": "0" },
                  "meta": { "provenance": { "kind": "Root" }, "guarantee": "Empirical",
                            "bound": { "kind": "ErrorBound", "eps": 0.5, "norm": "Linf",
                                       "basis": { "kind": "EmpiricalFit", "trials": 10000,
                                                  "method": "frady" } } } }"#;
    assert!(serde_json::from_str::<Value>(ok).is_ok());
}
