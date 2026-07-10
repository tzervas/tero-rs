//! M-120 acceptance — binary↔ternary swap: exhaustive round-trip (`dec(enc x) = Some x`, SC-1),
//! the never-silent out-of-range and illegal-pair paths (SC-3), the emitted `Bijective` certificate,
//! and interpreter integration via the `SwapEngine`.

use mycelium_cert::{
    binary_to_ternary, legal_pair, roundtrip_lemma_ref, ternary_to_binary, BinTernParams,
    BinaryTernarySwapEngine, SwapCertificate, SwapError,
};
use mycelium_core::{
    binary, ternary, ContentHash, GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Trit,
    Value,
};
use mycelium_interp::{EvalError, Interpreter};

fn policy() -> ContentHash {
    ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

fn byte_of(value: i64) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(binary::int_to_bits(value, 8).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// **SC-1 / P1:** `dec(enc b) = Some b` for *every* byte, over the canonical 8↔6 pair. Also checks
/// the result is `Exact`, records `policy_used`, and that the certificate is `Bijective{8,6}`.
#[test]
fn roundtrip_8x6_exhaustive() {
    assert!(legal_pair(8, 6));
    for v in -128..=127 {
        let b = byte_of(v);
        let (tern, cert) = binary_to_ternary(&b, 6, &policy()).expect("enc");
        // Encoded value denotes the same integer.
        match tern.payload() {
            Payload::Trits(t) => assert_eq!(ternary::trits_to_int(t), v),
            other => panic!("expected trits, got {other:?}"),
        }
        assert_eq!(tern.meta().guarantee(), GuaranteeStrength::Exact);
        assert_eq!(tern.meta().policy_used(), Some(&policy()));
        match &cert {
            SwapCertificate::Bijective {
                params, lemma_ref, ..
            } => {
                assert_eq!(*params, BinTernParams { width: 8, trits: 6 });
                assert_eq!(lemma_ref, &roundtrip_lemma_ref());
            }
            SwapCertificate::Bounded { .. } => panic!("binary↔ternary is bijective, not bounded"),
        }
        // dec(enc b) == b
        let (back, _) = ternary_to_binary(&tern, 8, &policy()).expect("dec");
        assert_eq!(back.payload(), b.payload());
        assert_eq!(back.repr(), b.repr());
    }
}

/// **P4:** decoding a ternary value outside `B_8` is an explicit error, never a silent wrap. The
/// all-`+` 6-trit value is 364 ∉ [−128,127] (`binary-ternary.md` §5).
#[test]
fn out_of_range_decode_is_explicit() {
    let all_plus = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![Trit::Pos; 6]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_eq!(ternary::trits_to_int(&[Trit::Pos; 6]), 364);
    assert_eq!(
        ternary_to_binary(&all_plus, 8, &policy()),
        Err(SwapError::OutOfRange)
    );
}

#[test]
fn illegal_pair_is_a_type_error_not_a_gamble() {
    // 8 bits need |·| up to 128, but 4 trits only reach (3^4−1)/2 = 40 → not a legal pair.
    assert!(!legal_pair(8, 4));
    let b = byte_of(0);
    assert_eq!(
        binary_to_ternary(&b, 4, &policy()),
        Err(SwapError::IllegalPair { width: 8, trits: 4 })
    );
}

#[test]
fn wrong_source_paradigm_is_rejected() {
    let tern = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![Trit::Zero; 6]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert!(matches!(
        binary_to_ternary(&tern, 6, &policy()),
        Err(SwapError::WrongSource { .. })
    ));
}

#[test]
fn certificate_round_trips_through_serde() {
    let (_, cert) = binary_to_ternary(&byte_of(-78), 6, &policy()).unwrap();
    let json = serde_json::to_string(&cert).unwrap();
    let back: SwapCertificate = serde_json::from_str(&json).unwrap();
    assert_eq!(cert, back);
    // Shape: tagged "Bijective" with the schema's required keys.
    let v: serde_json::Value = serde_json::to_value(&cert).unwrap();
    assert_eq!(v["kind"], "Bijective");
    assert_eq!(v["src"]["kind"], "Binary");
    assert_eq!(v["target"]["kind"], "Ternary");
    assert_eq!(v["params"]["width"], 8);
    assert_eq!(v["params"]["trits"], 6);
    assert!(v["lemma_ref"].as_str().unwrap().starts_with("blake3:"));
}

/// Pin the serializer output to the committed, CI-schema-validated example (the binding from "the
/// code emits it" to "check-jsonschema validates it" in `scripts/checks/schema.sh`).
#[test]
fn emitted_certificate_matches_committed_example() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/spec/schemas/examples/swap-certificate/valid/bijective-bin-tern-emitted.json"
    );
    let file = std::fs::read_to_string(path).expect("read example");
    let from_file: serde_json::Value = serde_json::from_str(&file).expect("parse");
    let (_, cert) = binary_to_ternary(&byte_of(-78), 6, &policy()).unwrap();
    let emitted: serde_json::Value = serde_json::to_value(&cert).unwrap();
    assert_eq!(emitted, from_file);
}

// --- interpreter integration (the engine plugs into M-110) ------------------------------------

fn interp() -> Interpreter {
    Interpreter::new(
        mycelium_interp::PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    )
}

#[test]
fn interpreter_evaluates_a_certified_swap() {
    // swap(0b1011_0010, to: Ternary{6}, policy)  ⟶  ⟨0,−1,0,0,+1,0⟩  (= −78)
    let node = Node::Swap {
        src: Box::new(Node::Const(byte_of(-78))),
        target: Repr::Ternary { trits: 6 },
        policy: policy(),
    };
    let out = interp().eval(&node).expect("evaluates");
    assert_eq!(
        out.payload(),
        &Payload::Trits(ternary::int_to_trits(-78, 6).unwrap())
    );
    assert_eq!(out.meta().policy_used(), Some(&policy()));
}

#[test]
fn interpreter_roundtrips_through_a_let() {
    // let b = 42 in swap(swap(b, →Ternary{6}), →Binary{8})  ⟶  42
    let node = Node::Let {
        id: "b".into(),
        bound: Box::new(Node::Const(byte_of(42))),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Swap {
                src: Box::new(Node::Var("b".into())),
                target: Repr::Ternary { trits: 6 },
                policy: policy(),
            }),
            target: Repr::Binary { width: 8 },
            policy: policy(),
        }),
    };
    let out = interp().eval(&node).expect("evaluates");
    assert_eq!(
        out.payload(),
        &Payload::Bits(binary::int_to_bits(42, 8).unwrap())
    );
}

#[test]
fn interpreter_reports_out_of_range_swap() {
    let all_plus = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![Trit::Pos; 6]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let node = Node::Swap {
        src: Box::new(Node::Const(all_plus)),
        target: Repr::Binary { width: 8 },
        policy: policy(),
    };
    assert!(matches!(interp().eval(&node), Err(EvalError::Swap(_))));
}

#[test]
fn interpreter_refuses_unrelated_swap() {
    let node = Node::Swap {
        src: Box::new(Node::Const(byte_of(1))),
        target: Repr::Dense {
            dim: 4,
            dtype: mycelium_core::ScalarKind::F32,
        },
        policy: policy(),
    };
    assert!(matches!(
        interp().eval(&node),
        Err(EvalError::UnsupportedSwap { .. })
    ));
}
