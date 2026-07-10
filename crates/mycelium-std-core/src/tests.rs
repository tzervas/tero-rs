//! Structural tests for `std.core` (M-515) — extracted verbatim from `lib.rs` per the house
//! test-layout rule (no tests in logic files; the `mycelium-std-recover/src/tests.rs` precedent),
//! as the M-927 pre-port polish (behavior-neutral — the test bodies are unchanged).

use super::*;
use mycelium_core::data::CtorRef;

fn exact_repr_value() -> CoreValue {
    let meta = Meta::exact(Provenance::Root);
    let v = Value::new(
        Repr::Binary { width: 2 },
        Payload::Bits(vec![true, false]),
        meta,
    )
    .expect("well-formed binary value");
    CoreValue::Repr(v)
}

fn nil_datum() -> CoreValue {
    let ctor = CtorRef::new(ContentHash::parse("blake3:nil").expect("hash"), 0);
    CoreValue::Data(Datum::new(ctor, vec![]))
}

#[test]
fn matrix_is_all_exact_and_effect_free() {
    // Spec §4: every row of the Ring-0 re-export surface is the honest `Exact`
    // floor and declares no effects. This guards against an accidental overclaim
    // (a `Proven`/`Empirical` tag here would itself violate VR-5).
    assert_eq!(GUARANTEE_MATRIX.len(), 9, "spec §4 lists nine rows");
    for row in GUARANTEE_MATRIX {
        assert_eq!(
            row.tag,
            GuaranteeStrength::Exact,
            "{} must be Exact",
            row.op
        );
        assert_eq!(row.effects, "none", "{} must be effect-free", row.op);
    }
}

#[test]
fn only_query_rows_are_explainable() {
    // The EXPLAIN window is exactly the value-tag/bound/provenance queries.
    let explainable: Vec<&str> = GUARANTEE_MATRIX
        .iter()
        .filter(|r| r.explainable)
        .map(|r| r.op)
        .collect();
    assert_eq!(explainable, ["guarantee_of", "bound_of", "provenance_of"]);
}

#[test]
fn queries_on_a_repr_value_are_present() {
    let v = exact_repr_value();
    assert_eq!(repr_of(&v), Some(&Repr::Binary { width: 2 }));
    assert!(meta_of(&v).is_some());
    assert_eq!(guarantee_of(&v), GuaranteeStrength::Exact);
    assert_eq!(provenance_of(&v), Some(&Provenance::Root));
    assert_eq!(bound_of(&v), None); // an exact value carries no bound
}

#[test]
fn queries_on_algebraic_data_report_absence_never_silently() {
    // C1 never-silent: a Datum has no Repr/Meta; the queries say so explicitly
    // with `None` rather than fabricating a default.
    let d = nil_datum();
    assert_eq!(repr_of(&d), None);
    assert_eq!(meta_of(&d), None);
    assert_eq!(bound_of(&d), None);
    assert_eq!(provenance_of(&d), None);
    // guarantee_of stays total even for data.
    let _g = guarantee_of(&d);
}

#[test]
fn lattice_meet_never_upgrades() {
    // Sanity re-check of the re-exported floor: composition cannot strengthen.
    use GuaranteeStrength::{Declared, Empirical, Exact, Proven};
    for a in [Exact, Proven, Empirical, Declared] {
        for b in [Exact, Proven, Empirical, Declared] {
            let m = a.meet(b);
            assert!(m.rank() >= a.rank().max(b.rank()));
        }
    }
}
