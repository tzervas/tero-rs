//! White-box tests for [`crate::repr`] — the [`Repr`] well-formedness predicate, the never-silent
//! [`Repr::check_well_formed`], and the [`MAX_DIM`] over-allocation cap (DN-40 §3). Extracted from
//! the logic file (test-layout rule, M-797).

use crate::repr::{FloatWidth, Repr, ScalarKind, SparsityClass, MAX_DIM};
use crate::WfError;

#[test]
fn well_formed_accepts_positive() {
    assert!(Repr::Binary { width: 8 }.well_formed());
    assert!(Repr::Ternary { trits: 6 }.well_formed());
    assert!(Repr::Dense {
        dim: 768,
        dtype: ScalarKind::F32
    }
    .well_formed());
    assert!(Repr::Vsa {
        model: "MAP-I".to_string(),
        dim: 10_000,
        sparsity: SparsityClass::Sparse { max_active: 100 },
    }
    .well_formed());
}

#[test]
fn well_formed_rejects_zero_and_empty() {
    assert!(!Repr::Binary { width: 0 }.well_formed());
    assert!(!Repr::Ternary { trits: 0 }.well_formed());
    assert!(!Repr::Vsa {
        model: String::new(),
        dim: 10_000,
        sparsity: SparsityClass::Dense,
    }
    .well_formed());
    assert!(!Repr::Vsa {
        model: "MAP-I".to_string(),
        dim: 10_000,
        sparsity: SparsityClass::Sparse { max_active: 0 },
    }
    .well_formed());
}

// Mutant-witness (repr.rs Dense{dim} and Vsa{dim}): the guard is `> 0` (strictly positive), NOT
// `>= 0`. A zero-dim Repr must be rejected (above). A dim-1 Repr MUST be accepted — pins the `>`
// side. Combined, both tests together kill the `> → >=` mutant on the Dense and Vsa dim checks.
#[test]
fn well_formed_rejects_zero_dim_accepts_one() {
    // Dense dim == 0 is rejected; dim == 1 is accepted (strict lower bound is 1, not 0).
    assert!(!Repr::Dense {
        dim: 0,
        dtype: ScalarKind::F16
    }
    .well_formed());
    assert!(Repr::Dense {
        dim: 1,
        dtype: ScalarKind::F16
    }
    .well_formed());
    // Vsa dim == 0 is rejected; dim == 1 is accepted.
    assert!(!Repr::Vsa {
        model: "MAP-I".to_string(),
        dim: 0,
        sparsity: SparsityClass::Dense,
    }
    .well_formed());
    assert!(Repr::Vsa {
        model: "MAP-I".to_string(),
        dim: 1,
        sparsity: SparsityClass::Dense,
    }
    .well_formed());
}

// --- DN-40 §3: over-allocation cap (MAX_DIM) -----------------------------------------------------

/// (a) A dimension *at* the cap is well-formed — the bound is inclusive, so the most extreme
/// legitimate value is still accepted (pins the `> MAX_DIM`, not `>= MAX_DIM`, side).
#[test]
fn dimension_at_cap_is_well_formed() {
    assert!(Repr::Binary { width: MAX_DIM }.well_formed());
    assert!(Repr::Ternary { trits: MAX_DIM }.well_formed());
    assert!(Repr::Dense {
        dim: MAX_DIM,
        dtype: ScalarKind::F64
    }
    .well_formed());
    assert!(Repr::Vsa {
        model: "MAP-I".to_string(),
        dim: MAX_DIM,
        sparsity: SparsityClass::Sparse {
            max_active: MAX_DIM
        },
    }
    .well_formed());
}

/// (b) A dimension *above* the cap is rejected, never-silently, with the error naming the offending
/// field, its value, and the cap. Each variant's dimension field is exercised.
#[test]
fn dimension_above_cap_rejected_naming_field() {
    let over = MAX_DIM + 1;

    assert_eq!(
        Repr::Binary { width: over }
            .check_well_formed()
            .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "width",
            value: over,
            cap: MAX_DIM,
        }
    );
    assert_eq!(
        Repr::Ternary { trits: over }
            .check_well_formed()
            .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "trits",
            value: over,
            cap: MAX_DIM,
        }
    );
    assert_eq!(
        Repr::Dense {
            dim: over,
            dtype: ScalarKind::F32
        }
        .check_well_formed()
        .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "dim",
            value: over,
            cap: MAX_DIM,
        }
    );
    assert_eq!(
        Repr::Vsa {
            model: "MAP-I".to_string(),
            dim: over,
            sparsity: SparsityClass::Dense,
        }
        .check_well_formed()
        .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "dim",
            value: over,
            cap: MAX_DIM,
        }
    );
    // The Sparse `max_active` field is also capped (its own over-alloc surface).
    assert_eq!(
        Repr::Vsa {
            model: "MAP-I".to_string(),
            dim: 10_000,
            sparsity: SparsityClass::Sparse { max_active: over },
        }
        .check_well_formed()
        .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "max_active",
            value: over,
            cap: MAX_DIM,
        }
    );

    // …and `well_formed()` (the bool predicate) agrees with `check_well_formed()`.
    assert!(!Repr::Binary { width: over }.well_formed());
    assert!(!Repr::Dense {
        dim: u32::MAX,
        dtype: ScalarKind::F64
    }
    .well_formed());
}

/// The never-silent message names the field, the value, and the cap (G2 — not a bare "malformed").
#[test]
fn dimension_too_large_display_names_field_value_cap() {
    let err = Repr::Dense {
        dim: MAX_DIM + 1,
        dtype: ScalarKind::F32,
    }
    .check_well_formed()
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("dim"), "must name the field: {msg:?}");
    assert!(
        msg.contains(&(MAX_DIM + 1).to_string()),
        "must name the offending value: {msg:?}"
    );
    assert!(
        msg.contains(&MAX_DIM.to_string()),
        "must name the cap: {msg:?}"
    );
}

/// A non-positive dimension still maps to [`WfError::MalformedRepr`] (the cap check does not change
/// the existing lower-guard contract).
#[test]
fn zero_dim_still_malformed_repr_not_too_large() {
    assert_eq!(
        Repr::Binary { width: 0 }.check_well_formed().unwrap_err(),
        WfError::MalformedRepr
    );
}

// --- RFC-0032 D3 (M-749): Repr::Seq well-formedness ---------------------------------------------

/// A sequence of a well-formed element repr is well-formed for any `len` within the cap — including
/// `len == 0` (the empty sequence is a legitimate value, unlike a scalar paradigm's `dim`).
#[test]
fn seq_well_formed_for_valid_elem_and_len_including_empty() {
    let elem = Box::new(Repr::Binary { width: 8 });
    assert!(Repr::Seq {
        elem: elem.clone(),
        len: 3
    }
    .well_formed());
    // len == 0 is well-formed (the empty seq) — the distinguishing rule from `dim_in_range`.
    assert!(Repr::Seq {
        elem: elem.clone(),
        len: 0
    }
    .well_formed());
    // len at the cap is still well-formed (inclusive bound).
    assert!(Repr::Seq { elem, len: MAX_DIM }.well_formed());
}

/// A sequence over a **malformed element repr** is rejected, never-silently — the nested
/// `check_well_formed` recurses, so a `Binary{0}` element propagates its `MalformedRepr` (G2).
#[test]
fn seq_rejects_malformed_element_repr() {
    let bad = Repr::Seq {
        elem: Box::new(Repr::Binary { width: 0 }),
        len: 3,
    };
    assert_eq!(bad.check_well_formed().unwrap_err(), WfError::MalformedRepr);
    // …and an over-cap *inner* dimension propagates the named `DimensionTooLarge`.
    let over = MAX_DIM + 1;
    assert_eq!(
        Repr::Seq {
            elem: Box::new(Repr::Dense {
                dim: over,
                dtype: ScalarKind::F32
            }),
            len: 3,
        }
        .check_well_formed()
        .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "dim",
            value: over,
            cap: MAX_DIM,
        }
    );
}

/// A sequence whose own `len` exceeds the cap is rejected never-silently, naming the `len` field.
#[test]
fn seq_len_above_cap_rejected_naming_field() {
    let over = MAX_DIM + 1;
    assert_eq!(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 8 }),
            len: over,
        }
        .check_well_formed()
        .unwrap_err(),
        WfError::DimensionTooLarge {
            field: "len",
            value: over,
            cap: MAX_DIM,
        }
    );
}

/// Nested sequences (a `Seq` of `Seq`) recurse through well-formedness correctly — a malformed
/// innermost element is still caught (G2 all the way down).
#[test]
fn nested_seq_recurses_well_formedness() {
    let good = Repr::Seq {
        elem: Box::new(Repr::Seq {
            elem: Box::new(Repr::Ternary { trits: 4 }),
            len: 2,
        }),
        len: 3,
    };
    assert!(good.well_formed());
    let bad = Repr::Seq {
        elem: Box::new(Repr::Seq {
            elem: Box::new(Repr::Ternary { trits: 0 }), // malformed innermost
            len: 2,
        }),
        len: 3,
    };
    assert_eq!(bad.check_well_formed().unwrap_err(), WfError::MalformedRepr);
}

// --- RFC-0032 D4 (M-750): Repr::Bytes well-formedness -------------------------------------------

/// A byte string is well-formed unconditionally (any byte content; no declared length to bound).
#[test]
fn bytes_is_always_well_formed() {
    assert!(Repr::Bytes.well_formed());
    assert!(Repr::Bytes.check_well_formed().is_ok());
    // A `Seq` of `Bytes` is also well-formed (the element repr `Bytes` recurses to Ok).
    assert!(Repr::Seq {
        elem: Box::new(Repr::Bytes),
        len: 4,
    }
    .well_formed());
}

// --- ADR-040 (M-896): Repr::Float — the scalar-float value form ---------------------------------

/// A scalar float is well-formed unconditionally: it declares no dimension, and the width enum is
/// the whole static constraint (ADR-040 §2.1). `Exact` — a total decidable predicate.
#[test]
fn float_is_always_well_formed() {
    let f = Repr::Float {
        width: FloatWidth::F64,
    };
    assert!(f.well_formed());
    assert!(f.check_well_formed().is_ok());
    // A `Seq` of scalar floats is also well-formed (the element repr recurses to Ok).
    assert!(Repr::Seq {
        elem: Box::new(f),
        len: 3,
    }
    .well_formed());
}

/// The `FloatWidth` content-address tag registry is FROZEN (ADR-040 §2.1, the `ScalarKind::tag`
/// discipline): `F64 == 0`, forever. A failing run here means an existing float value's identity
/// shifted — a rehash, which ADR-040 §3 defers to E20-1 and this change must NOT spend.
#[test]
fn float_width_tags_are_frozen() {
    assert_eq!(FloatWidth::F64.tag(), 0);
}

/// The `ScalarKind` tag registry stays frozen too (regression guard: adding `FloatWidth` must not
/// disturb the Dense dtype registry).
#[test]
fn scalar_kind_tags_stay_frozen() {
    assert_eq!(ScalarKind::F16.tag(), 0);
    assert_eq!(ScalarKind::Bf16.tag(), 1);
    assert_eq!(ScalarKind::F32.tag(), 2);
    assert_eq!(ScalarKind::F64.tag(), 3);
}

/// The serde wire form is `{"kind":"Float","width":"F64"}` — internally tagged on `kind` like every
/// other `Repr`, faithfully round-trippable (M-104 discipline).
#[test]
fn float_repr_wire_form_round_trips() {
    let f = Repr::Float {
        width: FloatWidth::F64,
    };
    let json = serde_json::to_value(&f).expect("serialize");
    assert_eq!(json, serde_json::json!({"kind": "Float", "width": "F64"}));
    let back: Repr = serde_json::from_value(json).expect("deserialize");
    assert_eq!(back, f);
}
