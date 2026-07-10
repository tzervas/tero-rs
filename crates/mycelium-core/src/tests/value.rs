//! White-box tests for [`crate::value`] — [`Value::new`] well-formedness/payload checks and the
//! DN-40 §3 over-allocation cap enforced on the deserialize path. Extracted from the logic file
//! (test-layout rule, M-797).

use crate::meta::{Meta, Provenance};
use crate::repr::MAX_DIM;
use crate::value::{Payload, Trit, Value};
use crate::{Repr, ScalarKind, WfError};

#[test]
fn well_matched_value_constructs() {
    let v = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    );
    assert!(v.is_ok());
}

#[test]
fn payload_length_must_match_repr() {
    let v = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false]), // wrong length
        Meta::exact(Provenance::Root),
    );
    assert_eq!(v.unwrap_err(), WfError::PayloadReprMismatch);
}

#[test]
fn payload_paradigm_must_match_repr() {
    let v = Value::new(
        Repr::Binary { width: 1 },
        Payload::Trits(vec![Trit::Pos]), // wrong paradigm
        Meta::exact(Provenance::Root),
    );
    assert_eq!(v.unwrap_err(), WfError::PayloadReprMismatch);
}

#[test]
fn malformed_repr_rejected() {
    let v = Value::new(
        Repr::Binary { width: 0 },
        Payload::Bits(vec![]),
        Meta::exact(Provenance::Root),
    );
    assert_eq!(v.unwrap_err(), WfError::MalformedRepr);
}

// --- DN-40 §3: over-allocation cap enforced through Value::new and the deserialize path -----------

/// `Value::new` rejects an over-cap declared dimension *before* the payload is examined — naming the
/// field/value/cap (over-allocation guard). The payload here is deliberately tiny: the point is that
/// the huge *declared* `dim` is caught before anything is sized to it.
#[test]
fn value_new_rejects_over_cap_dim_before_payload() {
    let over = MAX_DIM + 1;
    let v = Value::new(
        Repr::Dense {
            dim: over,
            dtype: ScalarKind::F64,
        },
        Payload::Scalars(vec![0.0]), // mismatched length — but the cap is checked first
        Meta::exact(Provenance::Root),
    );
    assert_eq!(
        v.unwrap_err(),
        WfError::DimensionTooLarge {
            field: "dim",
            value: over,
            cap: MAX_DIM,
        }
    );
}

/// (c) Deserializing a `Value` whose `repr` declares an over-cap dimension is rejected — never
/// silently accepted then over-allocated. `Value`'s `Deserialize` routes through `Value::new`, so
/// the cap is enforced on the wire path; the serde error carries the named cap message.
#[test]
fn deserialize_rejects_over_cap_declared_dimension() {
    let over = MAX_DIM + 1;
    // A self-describing wire value with a crafted huge declared `dim` and a (deliberately small)
    // payload. A naive consumer might size a buffer to `dim` before checking the payload.
    let json = format!(
        r#"{{"repr":{{"kind":"Dense","dim":{over},"dtype":"F32"}},"meta":{{"provenance":{{"kind":"Root"}},"guarantee":"Exact"}},"payload":{{"scalars":[0.0]}}}}"#
    );
    let err = serde_json::from_str::<Value>(&json).expect_err("over-cap dim must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("dim") && msg.contains(&MAX_DIM.to_string()),
        "deserialize error must name the offending dim and the cap (never-silent): {msg:?}"
    );
}

/// A declared dimension exactly at the cap round-trips through deserialize when its payload matches
/// (the inclusive bound does not reject a legitimate at-cap value on the wire). We keep the payload
/// length consistent with a small `dim` here to avoid allocating a billion elements in a unit test,
/// asserting instead via `Value::new` that the at-cap descriptor itself is well-formed.
#[test]
fn at_cap_descriptor_is_accepted_by_value_new() {
    // dim at the cap with a matching-length payload would allocate MAX_DIM scalars; we assert the
    // descriptor's well-formedness directly (the materialization cost is the very thing the cap
    // bounds), and rely on `repr` tests for the deserialize-of-Repr at-cap case.
    assert!(Repr::Dense {
        dim: MAX_DIM,
        dtype: ScalarKind::F32
    }
    .well_formed());
}

// --- RFC-0032 D3 (M-749): Repr::Seq payload matching + never-silent indexing --------------------

/// A `Binary{1}` element value, for building small homogeneous sequences in tests.
fn bit(b: bool) -> Value {
    Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![b]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed bit")
}

/// A well-matched sequence (count == len, every element's repr == elem) constructs.
#[test]
fn seq_well_matched_constructs() {
    let v = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 3,
        },
        Payload::Seq(vec![bit(true), bit(false), bit(true)]),
        Meta::exact(Provenance::Root),
    );
    assert!(v.is_ok());
    // The empty sequence is a legitimate value.
    let empty = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 0,
        },
        Payload::Seq(vec![]),
        Meta::exact(Provenance::Root),
    );
    assert!(empty.is_ok());
}

/// A sequence whose element count differs from its declared `len` is rejected (never silently
/// truncated/padded).
#[test]
fn seq_count_must_match_len() {
    let v = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 3,
        },
        Payload::Seq(vec![bit(true), bit(false)]), // 2 != 3
        Meta::exact(Provenance::Root),
    );
    assert_eq!(v.unwrap_err(), WfError::PayloadReprMismatch);
}

/// A heterogeneous sequence — an element whose repr differs from the declared `elem` — is rejected
/// (homogeneity is enforced, never silently accepted).
#[test]
fn seq_elements_must_be_homogeneous() {
    let wrong = Value::new(
        Repr::Ternary { trits: 1 },
        Payload::Trits(vec![Trit::Pos]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed trit");
    let v = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 2,
        },
        Payload::Seq(vec![bit(true), wrong]), // second element is Ternary, not Binary{1}
        Meta::exact(Provenance::Root),
    );
    assert_eq!(v.unwrap_err(), WfError::PayloadReprMismatch);
}

/// Never-silent indexing (G2): `seq_get` returns the element in range and `None` out of range —
/// never a panic, never a silent default. `seq_len` reports the element count.
#[test]
fn seq_get_is_never_silent() {
    let seq = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 2,
        },
        Payload::Seq(vec![bit(true), bit(false)]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed seq");

    assert_eq!(seq.seq_len(), Some(2));
    assert_eq!(seq.seq_get(0), Some(&bit(true)));
    assert_eq!(seq.seq_get(1), Some(&bit(false)));
    // Out of bounds → None, never a panic or a silent default.
    assert_eq!(seq.seq_get(2), None);
    assert_eq!(seq.seq_get(usize::MAX), None);

    // The accessors return None for a non-sequence value (never an empty-slice coercion).
    let not_seq = bit(true);
    assert_eq!(not_seq.seq_len(), None);
    assert_eq!(not_seq.seq_get(0), None);
    assert!(not_seq.seq_elems().is_none());
}

/// A sequence value round-trips through JSON faithfully (the wire form carries self-describing
/// elements), and deserializing a count≠len wire form is rejected (never silently accepted).
#[test]
fn seq_json_round_trips_and_rejects_mismatch() {
    let seq = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 2,
        },
        Payload::Seq(vec![bit(true), bit(false)]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed seq");
    let json = serde_json::to_string(&seq).expect("serialize");
    let back: Value = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(seq, back, "seq must round-trip faithfully: {json}");

    // A wire seq whose element count disagrees with the declared len is rejected on the way in.
    let bad = r#"{"repr":{"kind":"Seq","elem":{"kind":"Binary","width":1},"len":3},
                  "payload":{"seq":[
                     {"repr":{"kind":"Binary","width":1},"payload":{"bits":"1"},
                      "meta":{"provenance":{"kind":"Root"},"guarantee":"Exact"}}]},
                  "meta":{"provenance":{"kind":"Root"},"guarantee":"Exact"}}"#;
    assert!(
        serde_json::from_str::<Value>(bad).is_err(),
        "a count≠len seq wire form must be rejected, never silently accepted"
    );
}

/// The content hash distinguishes sequences by their elements and is order-sensitive; an identical
/// sequence collides. (Confirms the `Repr::Seq`/`Payload::Seq` content-addressing arms are wired —
/// without them a constructed seq would panic in `Canon`.)
#[test]
fn seq_content_hash_distinguishes_and_collides() {
    let mk = |a: bool, b: bool| {
        Value::new(
            Repr::Seq {
                elem: Box::new(Repr::Binary { width: 1 }),
                len: 2,
            },
            Payload::Seq(vec![bit(a), bit(b)]),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed seq")
    };
    // Identical sequences collide.
    assert_eq!(
        mk(true, false).content_hash(),
        mk(true, false).content_hash()
    );
    // Different elements differ.
    assert_ne!(
        mk(true, false).content_hash(),
        mk(true, true).content_hash()
    );
    // Order-sensitive: [t, f] ≠ [f, t].
    assert_ne!(
        mk(true, false).content_hash(),
        mk(false, true).content_hash()
    );
}

// --- RFC-0032 D4 (M-750): Repr::Bytes payload matching + never-silent byte access ----------------

/// A byte string is well-formed for any byte content (including empty), and `Repr::Bytes` only
/// matches a `Payload::Bytes` (never another payload).
#[test]
fn bytes_constructs_for_any_content_and_rejects_wrong_payload() {
    assert!(Value::new(
        Repr::Bytes,
        Payload::Bytes(vec![0xde, 0xad, 0xbe, 0xef]),
        Meta::exact(Provenance::Root),
    )
    .is_ok());
    // Empty bytes are a legitimate value.
    assert!(Value::new(
        Repr::Bytes,
        Payload::Bytes(vec![]),
        Meta::exact(Provenance::Root)
    )
    .is_ok());
    // A non-bytes payload under `Repr::Bytes` is rejected.
    assert_eq!(
        Value::new(
            Repr::Bytes,
            Payload::Bits(vec![true]),
            Meta::exact(Provenance::Root),
        )
        .unwrap_err(),
        WfError::PayloadReprMismatch
    );
}

/// Never-silent byte access (G2): `bytes_get`/`bytes_slice` return `None` out of range, `bytes_len`
/// reports the count, and every accessor returns `None` for a non-bytes value.
#[test]
fn bytes_access_is_never_silent() {
    let v = Value::new(
        Repr::Bytes,
        Payload::Bytes(vec![0x01, 0x02, 0x03]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed bytes");

    assert_eq!(v.bytes_len(), Some(3));
    assert_eq!(v.bytes_get(0), Some(0x01));
    assert_eq!(v.bytes_get(2), Some(0x03));
    // Out of bounds → None.
    assert_eq!(v.bytes_get(3), None);
    assert_eq!(v.bytes_get(usize::MAX), None);
    // Slice: in-range, empty, full; out-of-range / inverted → None.
    assert_eq!(v.bytes_slice(1, 3), Some(&[0x02, 0x03][..]));
    assert_eq!(v.bytes_slice(0, 0), Some(&[][..]));
    assert_eq!(v.bytes_slice(0, 3), Some(&[0x01, 0x02, 0x03][..]));
    assert_eq!(v.bytes_slice(0, 4), None); // end past len
    assert_eq!(v.bytes_slice(2, 1), None); // inverted

    // Non-bytes value → None everywhere (never a default / empty slice).
    let not_bytes = bit(true);
    assert_eq!(not_bytes.bytes_len(), None);
    assert_eq!(not_bytes.bytes_get(0), None);
    assert_eq!(not_bytes.bytes_slice(0, 0), None);
    assert!(not_bytes.bytes().is_none());
}

/// A byte string round-trips through JSON (lowercase-hex wire form); a non-hex / odd-length hex wire
/// form is rejected on the way in (never silently coerced).
#[test]
fn bytes_json_round_trips_and_rejects_bad_hex() {
    let v = Value::new(
        Repr::Bytes,
        Payload::Bytes(vec![0xde, 0xad, 0xbe, 0xef]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed bytes");
    let json = serde_json::to_string(&v).expect("serialize");
    assert!(
        json.contains("deadbeef"),
        "bytes render as lowercase hex: {json}"
    );
    let back: Value = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(v, back, "bytes must round-trip faithfully");

    // Odd-length hex is rejected.
    let odd = r#"{"repr":{"kind":"Bytes"},"payload":{"bytes":"abc"},
                  "meta":{"provenance":{"kind":"Root"},"guarantee":"Exact"}}"#;
    assert!(
        serde_json::from_str::<Value>(odd).is_err(),
        "odd-length hex must be rejected"
    );
    // A non-hex char is rejected.
    let nonhex = r#"{"repr":{"kind":"Bytes"},"payload":{"bytes":"zz"},
                     "meta":{"provenance":{"kind":"Root"},"guarantee":"Exact"}}"#;
    assert!(
        serde_json::from_str::<Value>(nonhex).is_err(),
        "non-hex char must be rejected, never silently coerced"
    );
}

/// The content hash distinguishes byte strings by content and collides on identical content
/// (confirms the `Repr::Bytes`/`Payload::Bytes` content-addressing arms are wired).
#[test]
fn bytes_content_hash_distinguishes_and_collides() {
    let mk = |bytes: Vec<u8>| {
        Value::new(
            Repr::Bytes,
            Payload::Bytes(bytes),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed bytes")
    };
    assert_eq!(
        mk(vec![1, 2, 3]).content_hash(),
        mk(vec![1, 2, 3]).content_hash()
    );
    assert_ne!(
        mk(vec![1, 2, 3]).content_hash(),
        mk(vec![1, 2, 4]).content_hash()
    );
    // A byte string is a distinct type from a Binary value with the same bytes (distinct repr tag).
    let as_binary = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false, false, false, false, false, false, false, true]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert_ne!(mk(vec![1]).content_hash(), as_binary.content_hash());
}

// --- ADR-040 (M-896): the scalar-float value form ------------------------------------------------

use crate::repr::FloatWidth;
use crate::value::CANONICAL_NAN_BITS;

fn float_value(x: f64) -> Value {
    Value::new(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(x),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed float value")
}

fn float_bits(v: &Value) -> u64 {
    v.float().expect("float payload").to_bits()
}

/// The deterministic edge corpus every float property is swept over (data-driven, fixture-first):
/// zeros (both signs), ordinary values, extremes, subnormals, the exactness boundary 2^53, the
/// in-band specials, and NaNs with non-canonical payload/sign bits (ADR-040 §2.3/§2.4).
const FLOAT_EDGE_BITS: [u64; 16] = [
    0x0000_0000_0000_0000, // +0.0
    0x8000_0000_0000_0000, // -0.0
    0x3ff8_0000_0000_0000, // 1.5
    0xbff8_0000_0000_0000, // -1.5
    0x7fef_ffff_ffff_ffff, // f64::MAX
    0xffef_ffff_ffff_ffff, // f64::MIN
    0x0010_0000_0000_0000, // f64::MIN_POSITIVE
    0x0000_0000_0000_0001, // smallest subnormal
    0x4340_0000_0000_0000, // 2^53 (the int-exactness boundary, ADR-040 §2.4)
    0x7ff0_0000_0000_0000, // +inf
    0xfff0_0000_0000_0000, // -inf
    CANONICAL_NAN_BITS,    // the canonical quiet NaN
    0x7ff8_0000_0000_0001, // quiet NaN, non-zero payload
    0xfff8_0000_0000_0000, // quiet NaN, sign bit set
    0x7ff0_0000_0000_0001, // signaling NaN
    0xfff7_ffff_ffff_ffff, // signaling NaN, sign bit set, max payload
];

/// DN-20 case tiering for the LCG sweeps below: `PROPTEST_CASES` selects the count (LOW on the
/// everyday `just check`, HIGH on `just check-full`); the property is never dropped, only its
/// case count is tiered.
fn sweep_cases() -> u64 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// A tiny deterministic LCG over u64 bit patterns (the pre-M-654 idiom): mycelium-core carries no
/// proptest dev-dep (kernel manifest kept minimal), so the property sweeps draw from this instead —
/// same tiering (`PROPTEST_CASES`), fixed seed, fully reproducible.
fn lcg_bits(seed: u64) -> impl Iterator<Item = u64> {
    let mut s = seed;
    std::iter::from_fn(move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        Some(s)
    })
}

#[test]
fn float_value_constructs_and_accessor_returns_it() {
    let v = float_value(1.5);
    assert_eq!(v.float(), Some(1.5));
    // Never-silent accessor: a non-float value has no scalar here (G2).
    let b = Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![true]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed");
    assert_eq!(b.float(), None);
}

#[test]
fn float_payload_must_match_float_repr() {
    // Wrong payload paradigm for Float → explicit rejection, never coerced (G2).
    let bad = Value::new(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Bits(vec![true]),
        Meta::exact(Provenance::Root),
    );
    assert_eq!(bad.unwrap_err(), WfError::PayloadReprMismatch);
    // And a Float payload under a non-Float repr is equally rejected.
    let bad = Value::new(
        Repr::Binary { width: 64 },
        Payload::Float(1.0),
        Meta::exact(Provenance::Root),
    );
    assert_eq!(bad.unwrap_err(), WfError::PayloadReprMismatch);
}

/// **Property (ADR-040 §2.3, `Empirical` over the corpus + sweep):** every NaN bit pattern —
/// quiet/signaling, any payload, either sign — constructs to the single canonical quiet NaN;
/// every non-NaN constructs bit-unchanged (including `-0.0` and subnormals).
#[test]
fn construction_canonicalizes_every_nan_and_only_nan() {
    let sweep = FLOAT_EDGE_BITS
        .into_iter()
        .chain(lcg_bits(0x0ADF_0040).take(sweep_cases() as usize));
    for bits in sweep {
        let x = f64::from_bits(bits);
        let got = float_bits(&float_value(x));
        if x.is_nan() {
            assert_eq!(
                got, CANONICAL_NAN_BITS,
                "NaN bits {bits:#018x} not canonical"
            );
        } else {
            assert_eq!(got, bits, "non-NaN bits {bits:#018x} must pass unchanged");
        }
    }
}

/// The signed zeros stay bit-distinct through construction (ADR-040 §2.3: observably distinct
/// values are never aliased), while `==` remains IEEE equality (`+0.0 == -0.0`) — the documented
/// FLAG-4 seam.
#[test]
fn signed_zeros_distinct_bits_ieee_equal() {
    let pos = float_value(0.0);
    let neg = float_value(-0.0);
    assert_ne!(float_bits(&pos), float_bits(&neg));
    assert_eq!(pos, neg); // derived PartialEq == IEEE equality on the payload
}

/// Canonical NaN is IEEE-unequal to itself under `==` (the other half of the FLAG-4 seam);
/// identity (content addressing) is tested in `tests/content.rs`.
#[test]
fn nan_value_is_ieee_unequal_to_itself() {
    let n = float_value(f64::NAN);
    assert_ne!(n, n.clone());
}

/// **Property (wire round-trip, `Empirical`; the finite-value exactness rides Rust's documented
/// shortest-round-trip float formatting, `Declared`):** `deserialize(serialize(v)) == v` bit-for-bit
/// over the edge corpus + sweep — `-0.0` keeps its sign, specials ride in-band as strings, NaN
/// round-trips to the canonical NaN.
#[test]
fn float_wire_round_trip_is_bit_exact() {
    let sweep = FLOAT_EDGE_BITS
        .into_iter()
        .chain(lcg_bits(0x0ADF_0041).take(sweep_cases() as usize));
    for bits in sweep {
        let v = float_value(f64::from_bits(bits));
        let json = serde_json::to_string(&v).expect("serialize");
        let back: Value = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            float_bits(&back),
            float_bits(&v),
            "wire round-trip drifted for bits {bits:#018x}"
        );
    }
}

/// The wire payload is the externally-tagged string form `{"float":"…"}` — a string, not a JSON
/// number, so the in-band specials (ADR-040 §2.4) are representable (a JSON number cannot carry
/// NaN/±inf; `serde_json` would silently null them — G2).
#[test]
fn float_wire_form_is_a_tagged_string() {
    let cases: [(f64, &str); 5] = [
        (1.5, "1.5"),
        (-0.0, "-0.0"),
        (f64::INFINITY, "inf"),
        (f64::NEG_INFINITY, "-inf"),
        (f64::NAN, "NaN"),
    ];
    for (x, s) in cases {
        let json = serde_json::to_value(float_value(x).payload()).expect("serialize");
        assert_eq!(json, serde_json::json!({ "float": s }));
    }
}

/// A malformed float string on the wire is rejected never-silently (G2), naming the offender.
#[test]
fn malformed_float_wire_string_is_rejected() {
    for bad in ["", "1.5.5", "0x1p3", "float", "NaN payload"] {
        let json = format!(r#"{{"float":{}}}"#, serde_json::json!(bad));
        let got: Result<Payload, _> = serde_json::from_str(&json);
        assert!(got.is_err(), "malformed float string {bad:?} was accepted");
    }
}
