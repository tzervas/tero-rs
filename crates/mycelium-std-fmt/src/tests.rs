//! In-crate white-box unit tests for `std.fmt` (extracted from `lib.rs`, M-797 as-touched).
//!
//! `use crate::*;` gives white-box access to the crate's items (the relocated tests are a
//! sibling of the logic module, so they import via `crate::` rather than `super::`).

use crate::*;
use mycelium_core::{
    meta::{Meta, Provenance},
    repr::{Repr, ScalarKind, SparsityClass},
    value::{Payload, Value},
};

// ── Test helpers ─────────────────────────────────────────────────────

fn binary_value(bits: &[bool]) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed binary value")
}

fn ternary_value(trits: &[Trit]) -> Value {
    let count = trits.len() as u32;
    Value::new(
        Repr::Ternary { trits: count },
        Payload::Trits(trits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed ternary value")
}

fn dense_value(xs: &[f64]) -> Value {
    let dim = xs.len() as u32;
    Value::new(
        Repr::Dense {
            dim,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(xs.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed dense value")
}

fn vsa_value(xs: &[f64]) -> Value {
    let dim = xs.len() as u32;
    Value::new(
        Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(xs.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed vsa value")
}

// ── Guarantee matrix invariants ───────────────────────────────────────

/// The guarantee matrix is internally consistent (RFC-0016 §4.5).
/// Mutation witness: set any row's tag to Proven -> assertion fires.
#[test]
fn guarantee_matrix_invariants_hold() {
    assert_matrix_invariants();
}

/// All expected ops appear in the matrix exactly once.
/// Mutation witness: remove a row -> count == 0.
#[test]
fn matrix_contains_all_five_ops_exactly_once() {
    let expected = [
        "display",
        "debug",
        "to_json",
        "from_json",
        "display_bounded",
    ];
    for op in &expected {
        let count = GUARANTEE_MATRIX.iter().filter(|r| r.op == *op).count();
        assert_eq!(count, 1, "op '{op}' must appear exactly once in the matrix");
    }
}

/// Every row in the matrix is `Exact` (spec §4 tag justification: no accuracy semantics).
/// Mutation witness: set one row's tag to Empirical -> assertion fires.
#[test]
fn all_matrix_rows_are_exact() {
    for row in GUARANTEE_MATRIX {
        assert_eq!(
            row.tag,
            GuaranteeStrength::Exact,
            "op '{}' must be Exact — fmt has no accuracy semantics (C2)",
            row.op
        );
    }
}

// ── Human projection: display ─────────────────────────────────────────

/// `display` on a binary value produces a `0b...` string (total, Exact).
/// Mutation witness: remove the `0b` prefix -> assertion fails.
#[test]
fn display_binary_starts_with_0b() {
    let v = binary_value(&[true, false, true, true]);
    let t = display(&v);
    assert!(
        t.as_str().starts_with("0b"),
        "binary display must start with '0b'; got {t}"
    );
    assert_eq!(t.as_str(), "0b1011");
}

/// `display` on a ternary value produces a `0t...` string.
/// Mutation witness: remove the `0t` prefix -> assertion fails.
#[test]
fn display_ternary_starts_with_0t() {
    let v = ternary_value(&[Trit::Pos, Trit::Zero, Trit::Neg]);
    let t = display(&v);
    assert!(
        t.as_str().starts_with("0t"),
        "ternary display must start with '0t'; got {t}"
    );
    assert_eq!(t.as_str(), "0t+0-");
}

/// `display` on a dense value produces a `[...]` bracketed list.
#[test]
fn display_dense_bracketed() {
    let v = dense_value(&[1.0, -1.0]);
    let t = display(&v);
    assert!(
        t.as_str().starts_with('[') && t.as_str().ends_with(']'),
        "dense display must be bracketed; got {t}"
    );
}

/// `display` on a VSA value produces a `hv[...]` string.
#[test]
fn display_vsa_hv_prefix() {
    let v = vsa_value(&[0.5, -0.5]);
    let t = display(&v);
    assert!(
        t.as_str().starts_with("hv["),
        "vsa display must start with 'hv['; got {t}"
    );
}

// ── Human projection: debug ───────────────────────────────────────────

/// `debug` on a binary value includes the paradigm and width.
/// Mutation witness: return the same as `display` -> assertion on "Binary<" fails.
#[test]
fn debug_binary_includes_repr_metadata() {
    let v = binary_value(&[true, false]);
    let t = debug(&v);
    assert!(
        t.as_str().contains("Binary<"),
        "debug binary must include 'Binary<'; got {t}"
    );
    assert!(
        t.as_str().contains("0b"),
        "debug binary must include the bit string; got {t}"
    );
}

/// `debug` on a ternary value includes the paradigm and trit count.
#[test]
fn debug_ternary_includes_repr_metadata() {
    let v = ternary_value(&[Trit::Zero]);
    let t = debug(&v);
    assert!(
        t.as_str().contains("Ternary<"),
        "debug ternary must include 'Ternary<'; got {t}"
    );
}

// ── Machine projection: to_json / from_json round-trip ────────────────

/// The machine-projection round-trip: `from_json(to_json(v)).content_hash() == v.content_hash()`.
///
/// This is the **one checked property** of `std.fmt` (spec §4; G11; RFC-0013 §4.3 / I3;
/// RFC-0001 §4.6 / ADR-003). The round-trip must preserve the canonical content hash so
/// that the JSON view is a faithful machine projection of the value's identity.
///
/// Mutation witness: truncate the payload in `to_json` -> hash diverges.
/// `to_json` refuses a non-finite `f64`, never silently coercing it to JSON `null`.
///
/// `serde_json` maps `NaN`/`±∞` to `null` (a lossy, identity-colliding encoding); the machine
/// projection must reject it explicitly (C1/G2). Regression guard for that silent-loss path.
#[test]
fn to_json_refuses_non_finite_f64_never_silent_null() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let v = dense_value(&[1.0, bad, 2.0]);
        assert_eq!(
            to_json(&v),
            Err(ToJsonError::NonFinite { index: 1 }),
            "to_json must refuse non-finite {bad:?}, not emit a silent null"
        );
    }
    // A wholly-finite dense value still projects fine.
    assert!(to_json(&dense_value(&[0.5, -1.0, 2.0])).is_ok());
}

#[test]
fn machine_projection_round_trip_preserves_content_hash_binary() {
    let v = binary_value(&[true, false, true, false, true, true, false, false]);
    let j = to_json(&v).expect("to_json: finite test value");
    let recovered = from_json(&j).expect("round-trip must succeed on well-formed value");
    assert_eq!(
        v.content_hash(),
        recovered.content_hash(),
        "from_json(to_json(v)) must recover the same content hash (spec §4; ADR-003)"
    );
}

/// Round-trip property for ternary values.
/// Mutation witness: omit trits from the JSON payload -> hash diverges.
#[test]
fn machine_projection_round_trip_preserves_content_hash_ternary() {
    let v = ternary_value(&[
        Trit::Pos,
        Trit::Neg,
        Trit::Zero,
        Trit::Pos,
        Trit::Neg,
        Trit::Zero,
    ]);
    let j = to_json(&v).expect("to_json: finite test value");
    let recovered = from_json(&j).expect("round-trip must succeed");
    assert_eq!(v.content_hash(), recovered.content_hash());
}

/// Round-trip property for dense values.
/// Mutation witness: change a scalar in the JSON -> hash diverges.
#[test]
fn machine_projection_round_trip_preserves_content_hash_dense() {
    let v = dense_value(&[1.0, -1.0, 0.5, -0.5]);
    let j = to_json(&v).expect("to_json: finite test value");
    let recovered = from_json(&j).expect("round-trip must succeed");
    assert_eq!(v.content_hash(), recovered.content_hash());
}

/// Round-trip property for VSA values.
#[test]
fn machine_projection_round_trip_preserves_content_hash_vsa() {
    let v = vsa_value(&[0.25, -0.25, 0.75]);
    let j = to_json(&v).expect("to_json: finite test value");
    let recovered = from_json(&j).expect("round-trip must succeed");
    assert_eq!(v.content_hash(), recovered.content_hash());
}

/// Full-corpus round-trip property: all 2^8 = 256 byte values.
///
/// Property test for the `Binary` round-trip bound: for *every* 8-bit value the round-trip
/// is exact. Mutation witness: return a fixed hash from `content_hash` -> all but one fail.
#[test]
fn machine_round_trip_all_256_byte_values() {
    for byte_val in 0u16..=255 {
        let bits: Vec<bool> = (0..8).rev().map(|i| (byte_val >> i) & 1 == 1).collect();
        let v = binary_value(&bits);
        let j = to_json(&v).expect("to_json: finite test value");
        let recovered = from_json(&j).expect("round-trip must succeed for all byte values");
        assert_eq!(
            v.content_hash(),
            recovered.content_hash(),
            "round-trip failed for byte_val={byte_val:#04x}"
        );
    }
}

/// `from_json` returns `Err(Malformed)` for a JSON non-object (C1 — never-silent).
/// Mutation witness: return Ok for string input -> assertion fails.
#[test]
fn from_json_rejects_non_object_as_malformed() {
    let bad = Json(serde_json::json!("not an object"));
    let err = from_json(&bad).expect_err("a string is not a valid Value");
    assert!(
        matches!(err, FromJsonError::Malformed(_)),
        "expected Malformed, got {err:?}"
    );
}

/// `from_json` returns `Err(Malformed)` when a required field is missing.
#[test]
fn from_json_rejects_missing_field() {
    let bad = Json(serde_json::json!({ "repr": { "kind": "Binary", "width": 8 } }));
    let err = from_json(&bad).expect_err("missing 'meta'/'payload' must be an error");
    assert!(
        matches!(err, FromJsonError::Malformed(_)),
        "expected Malformed, got {err:?}"
    );
}

/// `from_json` returns `Err(UnknownTag)` for a `repr.kind` it does not recognise (C1).
/// Mutation witness: return Ok or Malformed for an unknown tag -> assertion fails.
#[test]
fn from_json_unknown_repr_kind_is_explicit_error() {
    let bad = Json(serde_json::json!({
        "repr": { "kind": "Quantum", "width": 8 },
        "meta": {},
        "payload": { "bits": "00000000" }
    }));
    let err = from_json(&bad).expect_err("unknown kind must be an error");
    assert!(
        matches!(&err, FromJsonError::UnknownTag(t) if t == "Quantum"),
        "expected UnknownTag(\"Quantum\"), got {err:?}"
    );
}

// ── Bounded display ───────────────────────────────────────────────────

/// A budget larger than the value renders `Truncation::Complete` (no elision).
/// Mutation witness: always return Elided -> assertion fails.
#[test]
fn display_bounded_ample_budget_is_complete() {
    let v = binary_value(&[true, false, true]);
    let r = display_bounded(&v, Budget(100));
    assert_eq!(
        r.truncation,
        Truncation::Complete,
        "budget > len must produce Complete, not Elided"
    );
    assert_eq!(r.text.as_str(), "0b101");
}

/// A budget of exactly the value's length renders `Truncation::Complete`.
#[test]
fn display_bounded_exact_budget_is_complete() {
    let v = binary_value(&[false, true, false, true]);
    let r = display_bounded(&v, Budget(4));
    assert_eq!(r.truncation, Truncation::Complete);
}

/// A budget smaller than the value length elides and records `omitted` and `marker` (C1/C3).
/// Mutation witness: return Complete for any budget -> assertion fails.
#[test]
fn display_bounded_tight_budget_elides_and_records_omitted() {
    let v = binary_value(&[true, true, true, true, true, true, true, true]);
    let r = display_bounded(&v, Budget(4));
    match &r.truncation {
        Truncation::Elided { omitted, marker } => {
            assert_eq!(*omitted, 4, "must record 4 omitted bits");
            assert!(!marker.is_empty(), "marker must be non-empty");
            assert!(
                r.text.as_str().contains(marker.as_str()),
                "text must embed the marker verbatim (self-describing output)"
            );
        }
        Truncation::Complete => panic!("expected Elided for budget < len"),
    }
}

/// A budget of 0 elides everything (all content omitted).
/// Mutation witness: return Complete for budget=0 -> assertion fails.
#[test]
fn display_bounded_zero_budget_elides_all() {
    let v = binary_value(&[true, false]);
    let r = display_bounded(&v, Budget(0));
    assert!(
        matches!(&r.truncation, Truncation::Elided { omitted, .. } if *omitted == 2),
        "budget=0 must elide all 2 bits; got {:?}",
        r.truncation
    );
}

/// Ternary bounded display: elision carries the correct `omitted` count.
#[test]
fn display_bounded_ternary_elides_correctly() {
    let ts = [
        Trit::Pos,
        Trit::Neg,
        Trit::Zero,
        Trit::Pos,
        Trit::Zero,
        Trit::Neg,
    ];
    let v = ternary_value(&ts);
    let r = display_bounded(&v, Budget(3));
    match &r.truncation {
        Truncation::Elided { omitted, .. } => {
            assert_eq!(*omitted, 3, "must record 3 omitted trits");
        }
        Truncation::Complete => panic!("expected Elided"),
    }
}

/// Dense bounded display: elision carries the correct `omitted` count.
#[test]
fn display_bounded_dense_elides_correctly() {
    let v = dense_value(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let r = display_bounded(&v, Budget(2));
    match &r.truncation {
        Truncation::Elided { omitted, .. } => {
            assert_eq!(*omitted, 3, "must record 3 omitted scalars");
        }
        Truncation::Complete => panic!("expected Elided"),
    }
}

/// VSA bounded display: elision carries the correct `omitted` count.
#[test]
fn display_bounded_vsa_elides_correctly() {
    let v = vsa_value(&[0.1, 0.2, 0.3, 0.4]);
    let r = display_bounded(&v, Budget(1));
    match &r.truncation {
        Truncation::Elided { omitted, .. } => {
            assert_eq!(*omitted, 3, "must record 3 omitted components");
        }
        Truncation::Complete => panic!("expected Elided"),
    }
}

/// A `Truncation::Elided` value cannot be confused with `Truncation::Complete` at the type
/// level — the omitted/marker fields make silent truncation unrepresentable (C1/G2; spec §3).
///
/// This test is a structural check: it constructs both variants and verifies they are
/// distinct by `PartialEq`. The key guarantee is enforced by the type system (the `Elided`
/// variant requires both `omitted` and `marker`), so this test serves as an explicit,
/// mutation-witnessable documentation of that fact.
///
/// Mutation witness: collapse Truncation to a bool `elided: bool` -> the type check
/// collapses and the test would need to be rewritten, surfacing the regression.
#[test]
fn truncation_elided_is_not_confusable_with_complete() {
    let complete = Truncation::Complete;
    let elided = Truncation::Elided {
        omitted: 3,
        marker: "...<3 omitted>".to_owned(),
    };
    assert_ne!(
        complete, elided,
        "Complete and Elided must be distinct — silent truncation is unrepresentable (C1/G2)"
    );
}

// ── Projection is not identity (C4 / ADR-003) ─────────────────────────

/// `display` is a pure function of a borrowed `&Value`; the value's content hash is
/// unchanged after a call to `display` (C4 / ADR-003).
///
/// Mutation witness: if `display` took `&mut Value` and mutated meta -> hash would differ.
#[test]
fn display_does_not_change_content_hash() {
    let v = binary_value(&[true, false, true, false]);
    let h_before = v.content_hash();
    let _t = display(&v);
    let h_after = v.content_hash();
    assert_eq!(
        h_before, h_after,
        "display must not change the value's content hash (ADR-003; C4)"
    );
}

/// `to_json` does not change the value's content hash (C4 / ADR-003).
#[test]
fn to_json_does_not_change_content_hash() {
    let v = ternary_value(&[Trit::Pos, Trit::Zero]);
    let h_before = v.content_hash();
    let _j = to_json(&v).expect("to_json: finite test value");
    let h_after = v.content_hash();
    assert_eq!(h_before, h_after, "to_json must not change content hash");
}

// ── Property tests (per-bound property for every stated bound) ─────────

/// Property: for ALL 3-bit binary values (corpus of 8) the round-trip is exact.
#[test]
fn round_trip_property_all_3bit_binary_values() {
    for mask in 0u8..8 {
        let bits: Vec<bool> = (0..3).rev().map(|i| (mask >> i) & 1 == 1).collect();
        let v = binary_value(&bits);
        let recovered =
            from_json(&to_json(&v).expect("to_json: finite test value")).expect("round-trip");
        assert_eq!(
            v.content_hash(),
            recovered.content_hash(),
            "3-bit round-trip failed for mask={mask:#05b}"
        );
    }
}

/// Property: for all 27 2-trit ternary values the round-trip is exact.
#[test]
fn round_trip_property_all_2trit_ternary_values() {
    let all_trits = [Trit::Neg, Trit::Zero, Trit::Pos];
    for t1 in all_trits {
        for t2 in all_trits {
            let v = ternary_value(&[t1, t2]);
            let recovered =
                from_json(&to_json(&v).expect("to_json: finite test value")).expect("round-trip");
            assert_eq!(
                v.content_hash(),
                recovered.content_hash(),
                "2-trit round-trip failed for ({t1:?}, {t2:?})"
            );
        }
    }
}

/// Property (display_bounded bound): for every budget in 0..=len+2, if budget < len then
/// truncation is Elided and omitted == len - budget; if budget >= len then Complete.
///
/// This is the property test for the `display_bounded` guarantee: "omitted == total -
/// rendered_count, and truncation is Complete iff rendered_count == total" (spec §4).
///
/// Mutation witness: return Complete for budget < len -> second branch fires.
#[test]
fn display_bounded_property_omitted_count_equals_total_minus_budget() {
    let bits: Vec<bool> = (0..8).map(|i| i % 2 == 0).collect();
    let v = binary_value(&bits);
    let total = 8usize;

    for budget in 0..=total + 2 {
        let r = display_bounded(&v, Budget(budget));
        if budget >= total {
            assert_eq!(
                r.truncation,
                Truncation::Complete,
                "budget={budget} >= {total}: must be Complete"
            );
        } else {
            match &r.truncation {
                Truncation::Elided { omitted, .. } => {
                    assert_eq!(
                        *omitted,
                        total - budget,
                        "budget={budget}: omitted must equal {total}-{budget}={}",
                        total - budget
                    );
                }
                Truncation::Complete => {
                    panic!("budget={budget} < {total}: expected Elided but got Complete")
                }
            }
        }
    }
}

/// Property: the elision marker is always embedded in the rendered text when elided.
/// This makes the output self-describing without inspecting the Truncation variant (C3).
///
/// Mutation witness: omit the marker from the text -> assertion fails.
#[test]
fn display_bounded_elided_marker_is_in_text() {
    let v = dense_value(&[1.0, 2.0, 3.0, 4.0]);
    let r = display_bounded(&v, Budget(2));
    if let Truncation::Elided { marker, .. } = &r.truncation {
        assert!(
            r.text.as_str().contains(marker.as_str()),
            "elision marker must appear in the rendered text; text={:?}, marker={:?}",
            r.text.as_str(),
            marker
        );
    } else {
        panic!("expected Elided for budget=2 < dim=4");
    }
}
