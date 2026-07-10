//! Mutant-witness tests for prims.rs survivors (M-654 Gate A3).
use crate::prims::*;
use crate::EvalError;
use mycelium_core::{Meta, Payload, Provenance, Repr, Value};

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

// ---- prims.rs:61 — PrimRegistry::empty → Default::default() ----
// JUSTIFIED: PrimRegistry derives Default (BTreeMap::new()), and `empty()` also constructs
// BTreeMap::new(). The two are semantically identical — both produce an empty registry with no
// registered prims. This mutant is genuinely equivalent and is excluded via mutants.toml.

// ---- prims.rs:169 — expect_arity → Ok(()) ----
// Mutant: expect_arity always succeeds, even with wrong arity — arity errors are never raised.
// Kill: invoking a prim with wrong arity must return a PrimType error, not succeed silently.
#[test]
fn expect_arity_rejects_wrong_arity() {
    // Mutant-witness: prims.rs:169 replace expect_arity → Ok(()).
    // bit.not requires exactly 1 arg; providing 0 or 2 must be a PrimType error.
    // Test via the PrimRegistry public API.
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bit.not").expect("bit.not registered");
    let b = byte([true; 8]);
    // Zero args → PrimType.
    assert!(
        matches!(f("bit.not", &[]), Err(EvalError::PrimType { .. })),
        "bit.not with 0 args must be PrimType"
    );
    // Two args → PrimType.
    assert!(
        matches!(f("bit.not", &[&b, &b]), Err(EvalError::PrimType { .. })),
        "bit.not with 2 args must be PrimType"
    );
    // One arg → Ok (correct arity).
    assert!(
        f("bit.not", &[&b]).is_ok(),
        "bit.not with 1 arg must succeed"
    );
}

// ---- prims.rs:240 — prim_bit_and: & → | or ^ ----
// Mutant A (& → |): AND is replaced by OR — (1&0)=0 but (1|0)=1.
// Mutant B (& → ^): AND is replaced by XOR — (1&1)=1 but (1^1)=0.
// Kill: test a case where AND, OR, and XOR all differ (e.g. a=1,b=0 and a=1,b=1).
#[test]
fn bit_and_is_conjunction_not_disjunction_or_xor() {
    // Mutant-witness: prims.rs:240 & → | or ^.
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bit.and").expect("bit.and registered");

    // Operands: a = [true; 8], b = [false; 8].
    // AND: all false. OR: all true. XOR: all true. AND ≠ OR,XOR.
    let a = byte([true; 8]);
    let b_zeros = byte([false; 8]);
    let result = f("bit.and", &[&a, &b_zeros]).expect("bit.and evaluates");
    assert_eq!(
        result.payload(),
        &Payload::Bits(vec![false; 8]),
        "bit.and([1;8], [0;8]) must be [0;8] (AND), not [1;8] (OR/XOR)"
    );

    // Operands: a = [true; 8], b = [true; 8].
    // AND: all true. OR: all true. XOR: all false. AND ≠ XOR here.
    let b_ones = byte([true; 8]);
    let result2 = f("bit.and", &[&a, &b_ones]).expect("bit.and evaluates");
    assert_eq!(
        result2.payload(),
        &Payload::Bits(vec![true; 8]),
        "bit.and([1;8], [1;8]) must be [1;8] (AND/OR), distinguishing from XOR ([0;8])"
    );
}

// ---- prims.rs:243 — prim_bit_or: | → & or ^ ----
// Mutant A (| → &): OR is replaced by AND — (1|0)=1 but (1&0)=0.
// Mutant B (| → ^): OR is replaced by XOR — (1|1)=1 but (1^1)=0.
// Kill: test case where OR, AND, XOR all differ.
#[test]
fn bit_or_is_disjunction_not_conjunction_or_xor() {
    // Mutant-witness: prims.rs:243 | → & or ^.
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bit.or").expect("bit.or registered");

    // Operands: a = [true; 8], b = [false; 8].
    // OR: all true. AND: all false. XOR: all true. OR ≠ AND.
    let a = byte([true; 8]);
    let b_zeros = byte([false; 8]);
    let result = f("bit.or", &[&a, &b_zeros]).expect("bit.or evaluates");
    assert_eq!(
        result.payload(),
        &Payload::Bits(vec![true; 8]),
        "bit.or([1;8], [0;8]) must be [1;8] (OR), not [0;8] (AND)"
    );

    // Operands: a = [true; 8], b = [true; 8].
    // OR: all true. AND: all true. XOR: all false. OR ≠ XOR here.
    let b_ones = byte([true; 8]);
    let result2 = f("bit.or", &[&a, &b_ones]).expect("bit.or evaluates");
    assert_eq!(
        result2.payload(),
        &Payload::Bits(vec![true; 8]),
        "bit.or([1;8], [1;8]) must be [1;8] (OR/AND), distinguishing from XOR ([0;8])"
    );

    // Mixed: a=[T,F,T,F,T,F,T,F], b=[F,F,F,F,F,F,F,F].
    // OR=[T,F,T,F,T,F,T,F], AND=[F;8], XOR=[T,F,T,F,T,F,T,F] — OR and XOR agree here.
    // But the two tests above already distinguish OR from both AND and XOR.
}

// ---- RFC-0032 D1 (M-747): comparison/equality prims ----

/// MSB-first bit vector from a string (e.g. `"1010_0000"`, underscores ignored).
fn bits(s: &str) -> Vec<bool> {
    s.chars().filter(|c| *c != '_').map(|c| c == '1').collect()
}

/// A `Binary{8}` value from an MSB-first bit string (e.g. `"1010_0000"`, underscores ignored).
fn b8(s: &str) -> Value {
    let v = bits(s);
    assert_eq!(v.len(), 8, "b8 expects 8 bits");
    let mut a = [false; 8];
    a.copy_from_slice(&v);
    byte(a)
}

#[test]
fn cmp_eq_is_structural_equality_returning_binary1() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("cmp.eq").expect("cmp.eq registered");
    let a = b8("1010_0000");
    let same = b8("1010_0000");
    let diff = b8("1010_0001");
    // Equal ⇒ Binary{1} = 0b1; the repr collapses from Binary{8} to Binary{1}.
    let r = f("cmp.eq", &[&a, &same]).expect("cmp.eq evaluates");
    assert_eq!(r.repr(), &Repr::Binary { width: 1 });
    assert_eq!(r.payload(), &Payload::Bits(vec![true]));
    // Unequal ⇒ 0b0 (never a silent 0b1).
    let r = f("cmp.eq", &[&a, &diff]).expect("cmp.eq evaluates");
    assert_eq!(r.payload(), &Payload::Bits(vec![false]));
}

#[test]
fn cmp_lt_is_unsigned_magnitude_strict() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("cmp.lt").expect("cmp.lt registered");
    let lo = b8("1000_0000"); // 128
    let hi = b8("1010_0000"); // 160
                              // 128 < 160 ⇒ true.
    assert_eq!(
        f("cmp.lt", &[&lo, &hi]).expect("lt").payload(),
        &Payload::Bits(vec![true])
    );
    // Strict: not less when equal, and not less when greater.
    assert_eq!(
        f("cmp.lt", &[&hi, &hi]).expect("lt").payload(),
        &Payload::Bits(vec![false])
    );
    assert_eq!(
        f("cmp.lt", &[&hi, &lo]).expect("lt").payload(),
        &Payload::Bits(vec![false])
    );
}

#[test]
fn cmp_width_mismatch_is_never_silent() {
    // A `Binary{8}` vs `Binary{1}` comparison is an explicit PrimType error — never a silent
    // false (G2). (Same-paradigm, mismatched width.)
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("cmp.eq").expect("cmp.eq registered");
    let wide = b8("0000_0000");
    let narrow = Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert!(
        matches!(
            f("cmp.eq", &[&wide, &narrow]),
            Err(EvalError::PrimType { .. })
        ),
        "mismatched-width eq must be PrimType, never a silent false"
    );
}

// ---- RFC-0032 D2 (M-748): never-silent binary arithmetic ----

#[test]
fn bit_add_in_range_and_overflow_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bit.add").expect("bit.add registered");
    // 1 + 2 = 3, carries propagate MSB-first correctly.
    let r = f("bit.add", &[&b8("0000_0001"), &b8("0000_0010")]).expect("add");
    assert_eq!(r.payload(), &Payload::Bits(bits("0000_0011")));
    // 0b0000_1111 (15) + 0b0000_0001 (1) = 0b0001_0000 (16) — carry chain across the nibble.
    let r = f("bit.add", &[&b8("0000_1111"), &b8("0000_0001")]).expect("add");
    assert_eq!(r.payload(), &Payload::Bits(bits("0001_0000")));
    // 255 + 1 overflows Binary{8}: explicit Overflow, never a silent wrap to 0.
    assert!(
        matches!(
            f("bit.add", &[&b8("1111_1111"), &b8("0000_0001")]),
            Err(EvalError::Overflow { .. })
        ),
        "add overflow must be explicit, never a silent wrap"
    );
}

#[test]
fn bit_sub_in_range_and_underflow_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bit.sub").expect("bit.sub registered");
    // 5 - 2 = 3, borrow chain correct.
    let r = f("bit.sub", &[&b8("0000_0101"), &b8("0000_0010")]).expect("sub");
    assert_eq!(r.payload(), &Payload::Bits(bits("0000_0011")));
    // 16 - 1 = 15 — borrow across the nibble.
    let r = f("bit.sub", &[&b8("0001_0000"), &b8("0000_0001")]).expect("sub");
    assert_eq!(r.payload(), &Payload::Bits(bits("0000_1111")));
    // 0 - 1 underflows (no unsigned negative): explicit Overflow, never a silent wrap to 255.
    assert!(
        matches!(
            f("bit.sub", &[&b8("0000_0000"), &b8("0000_0001")]),
            Err(EvalError::Overflow { .. })
        ),
        "sub underflow must be explicit, never a silent wrap"
    );
}

// ---- RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): never-silent two's-complement multiply ----

/// A `Binary{width}` value of all-`false` bits, then patched via `set` — used to build wide
/// (> 8-bit) operands the `b8` helper can't express.
fn wide_binary(width: usize, ones_at_msb_first: &[usize]) -> Value {
    let mut bits = vec![false; width];
    for &i in ones_at_msb_first {
        bits[i] = true;
    }
    Value::new(
        Repr::Binary {
            width: width as u32,
        },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

#[test]
fn bin_mul_in_range_positive_and_negative() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.mul").expect("bin.mul registered");
    // 3 * 4 = 12 (0b0000_0011 * 0b0000_0100 = 0b0000_1100).
    let r = f("bin.mul", &[&b8("0000_0011"), &b8("0000_0100")]).expect("mul");
    assert_eq!(r.payload(), &Payload::Bits(bits("0000_1100")));
    assert_eq!(r.repr(), &Repr::Binary { width: 8 });
    // -3 * 4 = -12: -3 is 0b1111_1101, -12 is 0b1111_0100.
    let r = f("bin.mul", &[&b8("1111_1101"), &b8("0000_0100")]).expect("mul");
    assert_eq!(r.payload(), &Payload::Bits(bits("1111_0100")));
    // -3 * -4 = 12.
    let r = f("bin.mul", &[&b8("1111_1101"), &b8("1111_1100")]).expect("mul");
    assert_eq!(r.payload(), &Payload::Bits(bits("0000_1100")));
}

/// The classic two's-complement overflow edge: `i8::MIN * -1 = 128`, out of `B_8 = [-128, 127]` —
/// an explicit `Overflow`, never a silent wrap back to `-128`.
#[test]
fn bin_mul_min_times_neg_one_overflows() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.mul").expect("bin.mul registered");
    let min = b8("1000_0000"); // -128
    let neg_one = b8("1111_1111"); // -1
    assert!(
        matches!(
            f("bin.mul", &[&min, &neg_one]),
            Err(EvalError::Overflow { .. })
        ),
        "i8::MIN * -1 must be an explicit overflow, never a silent wrap"
    );
}

#[test]
fn bin_mul_overflow_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.mul").expect("bin.mul registered");
    // 127 * 2 = 254, out of B_8 ([-128, 127]).
    assert!(
        matches!(
            f("bin.mul", &[&b8("0111_1111"), &b8("0000_0010")]),
            Err(EvalError::Overflow { .. })
        ),
        "mul overflow must be explicit, never a silent wrap"
    );
}

#[test]
fn bin_mul_width_mismatch_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.mul").expect("bin.mul registered");
    let wide = b8("0000_0001");
    let narrow = Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    assert!(
        matches!(
            f("bin.mul", &[&wide, &narrow]),
            Err(EvalError::PrimType { .. })
        ),
        "mismatched-width mul must be PrimType, never a silent coercion"
    );
}

/// A width beyond the current `bin.mul` cap (`mycelium_core::binary::MUL_MAX_WIDTH`) is an explicit
/// `PrimType` refusal — distinct from an in-range-width `Overflow` — never a silently-truncated
/// native-int computation (M-887 scope boundary; FLAGged for the Gap-B follow-ons).
#[test]
fn bin_mul_over_cap_width_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.mul").expect("bin.mul registered");
    let width = mycelium_core::binary::MUL_MAX_WIDTH + 1;
    let a = wide_binary(width, &[]);
    let b = wide_binary(width, &[]);
    assert!(
        matches!(f("bin.mul", &[&a, &b]), Err(EvalError::PrimType { .. })),
        "an over-cap width must be an explicit PrimType refusal, never a silent truncation"
    );
}

/// **Property test (the overflow bound):** for every in-range pair at a small width, `bin.mul`'s
/// result agrees with an `i64` oracle; every out-of-range pair is an explicit `Overflow`. Mirrors
/// `mycelium_core::binary`'s own `mul_matches_integer_oracle` at the codec layer, one level up
/// through the prim's dispatch + never-silent-error mapping.
#[test]
fn bin_mul_matches_integer_oracle_at_width6() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.mul").expect("bin.mul registered");
    let n: u32 = 6;
    let lo = -(1i64 << (n - 1));
    let hi = (1i64 << (n - 1)) - 1;
    for x in lo..=hi {
        for y in lo..=hi {
            let av = mycelium_core::binary::int_to_bits(x, n).unwrap();
            let bv = mycelium_core::binary::int_to_bits(y, n).unwrap();
            let a = Value::new(
                Repr::Binary { width: n },
                Payload::Bits(av),
                Meta::exact(Provenance::Root),
            )
            .unwrap();
            let b = Value::new(
                Repr::Binary { width: n },
                Payload::Bits(bv),
                Meta::exact(Provenance::Root),
            )
            .unwrap();
            let expected = i128::from(x) * i128::from(y);
            let got = f("bin.mul", &[&a, &b]);
            if expected >= i128::from(lo) && expected <= i128::from(hi) {
                let want_bits = mycelium_core::binary::int_to_bits(expected as i64, n).unwrap();
                assert_eq!(
                    got.expect("in-range mul must succeed").payload(),
                    &Payload::Bits(want_bits),
                    "mul {x}*{y} at n={n}"
                );
            } else {
                assert!(
                    matches!(got, Err(EvalError::Overflow { .. })),
                    "mul {x}*{y} at n={n} should overflow, got {got:?}"
                );
            }
        }
    }
}

// ---- RFC-0033 §4.1.2/§4.1.3 (M-888, `enb` Gap B): never-silent unsigned division/remainder ----

/// A `Binary{n}` value from a non-negative `u64`, built via `mycelium_core::binary::uint_to_bits`.
fn u_bin(value: u64, n: u32) -> Value {
    let bits = mycelium_core::binary::uint_to_bits(value, n).expect("in range");
    Value::new(
        Repr::Binary { width: n },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

#[test]
fn bin_div_and_rem_worked_examples() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div").expect("bin.div registered");
    let rem = reg.get("bin.rem").expect("bin.rem registered");
    // 7 / 2 = 3 remainder 1.
    let a = u_bin(7, 8);
    let b = u_bin(2, 8);
    let q = div("bin.div", &[&a, &b]).expect("7 / 2");
    let r = rem("bin.rem", &[&a, &b]).expect("7 % 2");
    assert_eq!(
        q.payload(),
        &Payload::Bits(mycelium_core::binary::uint_to_bits(3, 8).unwrap())
    );
    assert_eq!(
        r.payload(),
        &Payload::Bits(mycelium_core::binary::uint_to_bits(1, 8).unwrap())
    );
}

#[test]
fn bin_div_by_zero_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div").expect("bin.div registered");
    let rem = reg.get("bin.rem").expect("bin.rem registered");
    let a = u_bin(7, 8);
    let zero = u_bin(0, 8);
    assert!(
        matches!(
            div("bin.div", &[&a, &zero]),
            Err(EvalError::PrimType { .. })
        ),
        "division by zero must be an explicit PrimType refusal, never a panic or silent value"
    );
    assert!(
        matches!(
            rem("bin.rem", &[&a, &zero]),
            Err(EvalError::PrimType { .. })
        ),
        "remainder by zero must be an explicit PrimType refusal, never a panic or silent value"
    );
}

#[test]
fn bin_div_rem_width_mismatch_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div").expect("bin.div registered");
    let wide = u_bin(1, 8);
    let narrow = u_bin(1, 1);
    assert!(
        matches!(
            div("bin.div", &[&wide, &narrow]),
            Err(EvalError::PrimType { .. })
        ),
        "mismatched-width div must be PrimType, never a silent coercion"
    );
}

/// A width beyond the current `bin.div`/`bin.rem` cap (`mycelium_core::binary::DIV_MAX_WIDTH`) is
/// an explicit `PrimType` refusal — never a silently-truncated native-int computation (M-888 scope
/// boundary, mirroring `bin.mul`'s `MUL_MAX_WIDTH` refusal).
#[test]
fn bin_div_over_cap_width_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div").expect("bin.div registered");
    let width = mycelium_core::binary::DIV_MAX_WIDTH + 1;
    let a = wide_binary(width, &[]);
    let b = wide_binary(width, &[]);
    assert!(
        matches!(div("bin.div", &[&a, &b]), Err(EvalError::PrimType { .. })),
        "an over-cap width must be an explicit PrimType refusal, never a silent truncation"
    );
}

/// **Property test (the Euclidean identity):** for every pair at a small width with a nonzero
/// divisor, `bin.div`/`bin.rem` satisfy `a == (a/b)*b + (a%b)` bit-exactly, with `remainder <
/// divisor`; every zero-divisor pair is an explicit `PrimType` refusal, never a panic. Mirrors
/// `mycelium_core::binary`'s own `div_rem_matches_euclidean_identity_oracle` at the codec layer,
/// one level up through the prim's dispatch + never-silent-error mapping.
#[test]
// The `x / y` / `x % y` in the `y != 0` branch are the trusted native oracles this test checks the
// prim against; they must stay plain (clippy 1.96 `manual_checked_ops` would obscure the oracle).
#[allow(clippy::manual_checked_ops)]
fn bin_div_rem_satisfy_euclidean_identity_at_width6() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div").expect("bin.div registered");
    let rem = reg.get("bin.rem").expect("bin.rem registered");
    let n: u32 = 6;
    let hi: u64 = (1u64 << n) - 1;
    for x in 0..=hi {
        for y in 0..=hi {
            let a = u_bin(x, n);
            let b = u_bin(y, n);
            let got_q = div("bin.div", &[&a, &b]);
            let got_r = rem("bin.rem", &[&a, &b]);
            if y == 0 {
                assert!(
                    matches!(got_q, Err(EvalError::PrimType { .. })),
                    "div by zero at x={x} must refuse, got {got_q:?}"
                );
                assert!(
                    matches!(got_r, Err(EvalError::PrimType { .. })),
                    "rem by zero at x={x} must refuse, got {got_r:?}"
                );
            } else {
                let q_val = got_q.expect("in-range div must succeed");
                let r_val = got_r.expect("in-range rem must succeed");
                let Payload::Bits(q_bits) = q_val.payload() else {
                    panic!("bin.div must return Payload::Bits")
                };
                let Payload::Bits(r_bits) = r_val.payload() else {
                    panic!("bin.rem must return Payload::Bits")
                };
                let qv = mycelium_core::binary::bits_to_uint(q_bits);
                let rv = mycelium_core::binary::bits_to_uint(r_bits);
                assert_eq!(qv, x / y, "quotient {x}/{y} at n={n}");
                assert_eq!(rv, x % y, "remainder {x}/{y} at n={n}");
                assert_eq!(
                    qv * y + rv,
                    x,
                    "Euclidean identity {x} == ({x}/{y})*{y} + {x}%{y}"
                );
                assert!(rv < y, "remainder must be < divisor");
            }
        }
    }
}

// ---- RFC-0033 §4.1.2/§4.1.3 (M-889, `enb` Gap B): never-silent logical shift ----

#[test]
fn bin_shl_and_shr_worked_examples() {
    let reg = PrimRegistry::with_builtins();
    let shl = reg.get("bin.shl").expect("bin.shl registered");
    let shr = reg.get("bin.shr").expect("bin.shr registered");
    // 1 << 3 = 8, 8 >> 3 = 1.
    let one = u_bin(1, 8);
    let three = u_bin(3, 8);
    let r = shl("bin.shl", &[&one, &three]).expect("1 << 3");
    assert_eq!(
        r.payload(),
        &Payload::Bits(mycelium_core::binary::uint_to_bits(8, 8).unwrap())
    );
    let eight = u_bin(8, 8);
    let r = shr("bin.shr", &[&eight, &three]).expect("8 >> 3");
    assert_eq!(
        r.payload(),
        &Payload::Bits(mycelium_core::binary::uint_to_bits(1, 8).unwrap())
    );
    // Logical (zero-filling) right shift: 0x80 >> 4 = 0x08, never sign-extended.
    let hi_bit = u_bin(0b1000_0000, 8);
    let four = u_bin(4, 8);
    let r = shr("bin.shr", &[&hi_bit, &four]).expect("0x80 >> 4");
    assert_eq!(
        r.payload(),
        &Payload::Bits(mycelium_core::binary::uint_to_bits(0b0000_1000, 8).unwrap())
    );
}

#[test]
fn bin_shift_by_zero_is_identity() {
    let reg = PrimRegistry::with_builtins();
    let shl = reg.get("bin.shl").expect("bin.shl registered");
    let shr = reg.get("bin.shr").expect("bin.shr registered");
    let a = u_bin(0b1010_1010, 8);
    let zero = u_bin(0, 8);
    let r = shl("bin.shl", &[&a, &zero]).expect("shl by 0");
    assert_eq!(r.payload(), a.payload());
    let r = shr("bin.shr", &[&a, &zero]).expect("shr by 0");
    assert_eq!(r.payload(), a.payload());
}

/// A shift amount `>= width` is an explicit `PrimType` refusal — never UB, a silently wrapped
/// shift amount, or a silently-zeroed result.
#[test]
fn bin_shift_amount_at_or_above_width_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let shl = reg.get("bin.shl").expect("bin.shl registered");
    let shr = reg.get("bin.shr").expect("bin.shr registered");
    let a = u_bin(1, 8);
    let width = u_bin(8, 8);
    assert!(
        matches!(
            shl("bin.shl", &[&a, &width]),
            Err(EvalError::PrimType { .. })
        ),
        "shift-amount == width must be an explicit PrimType refusal, never UB/wrap"
    );
    assert!(
        matches!(
            shr("bin.shr", &[&a, &width]),
            Err(EvalError::PrimType { .. })
        ),
        "shift-amount == width must be an explicit PrimType refusal, never UB/wrap"
    );
    let above = u_bin(255, 8);
    assert!(matches!(
        shl("bin.shl", &[&a, &above]),
        Err(EvalError::PrimType { .. })
    ));
    assert!(matches!(
        shr("bin.shr", &[&a, &above]),
        Err(EvalError::PrimType { .. })
    ));
}

#[test]
fn bin_shift_width_mismatch_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let shl = reg.get("bin.shl").expect("bin.shl registered");
    let wide = u_bin(1, 8);
    let narrow = u_bin(1, 1);
    assert!(
        matches!(
            shl("bin.shl", &[&wide, &narrow]),
            Err(EvalError::PrimType { .. })
        ),
        "mismatched-width shift must be PrimType, never a silent coercion"
    );
}

/// A width beyond the current `bin.shl`/`bin.shr` cap (`mycelium_core::binary::SHIFT_MAX_WIDTH`)
/// is an explicit `PrimType` refusal — never a silently-truncated native-int computation (M-889
/// scope boundary, mirroring `bin.mul`/`bin.div`'s width-cap refusals).
#[test]
fn bin_shift_over_cap_width_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let shl = reg.get("bin.shl").expect("bin.shl registered");
    let width = mycelium_core::binary::SHIFT_MAX_WIDTH + 1;
    let a = wide_binary(width, &[]);
    let b = wide_binary(width, &[]);
    assert!(
        matches!(shl("bin.shl", &[&a, &b]), Err(EvalError::PrimType { .. })),
        "an over-cap width must be an explicit PrimType refusal, never a silent truncation"
    );
}

/// **Property test (the shift-amount bound):** for every value/shift-amount pair at a small width,
/// `bin.shl`/`bin.shr` agree with a native `u64` shift for in-range amounts and refuse explicitly
/// for `k >= n`. Mirrors `mycelium_core::binary`'s own `shift_matches_native_oracle` at the codec
/// layer, one level up through the prim's dispatch + never-silent-error mapping.
#[test]
fn bin_shift_matches_native_oracle_at_width6() {
    let reg = PrimRegistry::with_builtins();
    let shl = reg.get("bin.shl").expect("bin.shl registered");
    let shr = reg.get("bin.shr").expect("bin.shr registered");
    let n: u32 = 6;
    let hi: u64 = (1u64 << n) - 1;
    for v in 0..=hi {
        for k in 0..=hi {
            let a = u_bin(v, n);
            let kb = u_bin(k, n);
            let got_shl = shl("bin.shl", &[&a, &kb]);
            let got_shr = shr("bin.shr", &[&a, &kb]);
            if k >= u64::from(n) {
                assert!(
                    matches!(got_shl, Err(EvalError::PrimType { .. })),
                    "shl {v}<<{k} at n={n} should refuse, got {got_shl:?}"
                );
                assert!(
                    matches!(got_shr, Err(EvalError::PrimType { .. })),
                    "shr {v}>>{k} at n={n} should refuse, got {got_shr:?}"
                );
            } else {
                let mask = (1u64 << n) - 1;
                let expected_shl = (v << k) & mask;
                let expected_shr = v >> k;
                let shl_val = got_shl.expect("in-range shl must succeed");
                let shr_val = got_shr.expect("in-range shr must succeed");
                let Payload::Bits(shl_bits) = shl_val.payload() else {
                    panic!("bin.shl must return Payload::Bits")
                };
                let Payload::Bits(shr_bits) = shr_val.payload() else {
                    panic!("bin.shr must return Payload::Bits")
                };
                assert_eq!(
                    mycelium_core::binary::bits_to_uint(shl_bits),
                    expected_shl,
                    "shl {v}<<{k} at n={n}"
                );
                assert_eq!(
                    mycelium_core::binary::bits_to_uint(shr_bits),
                    expected_shr,
                    "shr {v}>>{k} at n={n}"
                );
            }
        }
    }
}

// ---- RFC-0033 §4.1.2/§4.1.3 (M-767, `enb` Gap B): the signedness-split signed op set ----------
//
// `bin.div_s`/`bin.rem_s`/`bin.shr_s`/`cmp.lt_s` — the signed counterparts to the unsigned
// `bin.div`/`bin.rem`/`bin.shr` and the D1 unsigned `cmp.lt` (ADR-028: distinct named ops).
// These tests pin the wrapper's never-silent error mapping (div-by-zero / width mismatch /
// over-cap / out-of-range shift amount → `PrimType`; the single signed-division overflow
// `min ÷ −1` → `Overflow`) and the signed semantics the `_u` twins would get wrong (truncation
// toward zero, sign extension, the −1-sorts-below-0 order). Deep coverage of the codecs lives in
// `mycelium-core/src/tests/binary.rs`; here the fixture is the corpus table + assert-over-a-case.

/// A `Ternary{n}` value from an MSB-first digit string over `{-, 0, +}` (test fixture).
fn trits(s: &str) -> Value {
    use mycelium_core::Trit;
    let ds: Vec<Trit> = s
        .chars()
        .map(|c| match c {
            '-' => Trit::Neg,
            '0' => Trit::Zero,
            '+' => Trit::Pos,
            other => panic!("bad trit digit {other:?}"),
        })
        .collect();
    let n = u32::try_from(ds.len()).expect("small test widths");
    Value::new(
        Repr::Ternary { trits: n },
        Payload::Trits(ds),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A `Binary{n}` value from a signed `i64`, built via `mycelium_core::binary::int_to_bits`.
fn s_bin(value: i64, n: u32) -> Value {
    let bits = mycelium_core::binary::int_to_bits(value, n).expect("in range");
    Value::new(
        Repr::Binary { width: n },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// Worked examples pinning **truncation toward zero** (SMT-LIB `bvsdiv`/`bvsrem`; the module
/// rounding-convention note): `-7 / 2 = -3` r `-1` — floored division would answer `-4` r `1`.
#[test]
fn bin_div_s_and_rem_s_pin_truncation_toward_zero() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div_s").expect("bin.div_s registered");
    let rem = reg.get("bin.rem_s").expect("bin.rem_s registered");
    // (a, b, q, r) — the four sign quadrants at Binary{8}.
    for (a, b, q, r) in [
        (7i64, 2i64, 3i64, 1i64),
        (-7, 2, -3, -1),
        (7, -2, -3, 1),
        (-7, -2, 3, -1),
    ] {
        let got_q = div("bin.div_s", &[&s_bin(a, 8), &s_bin(b, 8)]).expect("quotient");
        let got_r = rem("bin.rem_s", &[&s_bin(a, 8), &s_bin(b, 8)]).expect("remainder");
        assert_eq!(
            got_q.payload(),
            &Payload::Bits(mycelium_core::binary::int_to_bits(q, 8).unwrap()),
            "{a} / {b}"
        );
        assert_eq!(
            got_r.payload(),
            &Payload::Bits(mycelium_core::binary::int_to_bits(r, 8).unwrap()),
            "{a} % {b}"
        );
    }
}

/// The §4.1.3 signed overflow-detect case: `-128 ÷ -1` (true quotient `+128`, out of `B_8`) is an
/// explicit `Overflow` — never a silent wrap back to `-128` and never SMT-LIB's defined wrap.
/// `rem_s(-128, -1) = 0` fits `B_8` exactly and succeeds.
#[test]
fn bin_div_s_min_by_neg_one_is_explicit_overflow() {
    let reg = PrimRegistry::with_builtins();
    let div = reg.get("bin.div_s").expect("bin.div_s registered");
    let rem = reg.get("bin.rem_s").expect("bin.rem_s registered");
    let min = s_bin(-128, 8);
    let neg_one = s_bin(-1, 8);
    assert!(
        matches!(
            div("bin.div_s", &[&min, &neg_one]),
            Err(EvalError::Overflow { .. })
        ),
        "-128 / -1 must be an explicit Overflow, never a silent wrap"
    );
    let r = rem("bin.rem_s", &[&min, &neg_one]).expect("-128 % -1 = 0 fits B_8");
    assert_eq!(
        r.payload(),
        &Payload::Bits(mycelium_core::binary::int_to_bits(0, 8).unwrap())
    );
}

#[test]
fn bin_div_s_and_rem_s_by_zero_are_never_silent() {
    let reg = PrimRegistry::with_builtins();
    for name in ["bin.div_s", "bin.rem_s"] {
        let f = reg.get(name).expect("registered");
        let a = s_bin(-7, 8);
        let zero = s_bin(0, 8);
        assert!(
            matches!(f(name, &[&a, &zero]), Err(EvalError::PrimType { .. })),
            "{name}: division by zero must be an explicit PrimType refusal, never a panic"
        );
    }
}

#[test]
fn bin_div_s_width_mismatch_and_over_cap_are_never_silent() {
    let reg = PrimRegistry::with_builtins();
    for name in ["bin.div_s", "bin.rem_s"] {
        let f = reg.get(name).expect("registered");
        assert!(
            matches!(
                f(name, &[&s_bin(1, 8), &s_bin(1, 4)]),
                Err(EvalError::PrimType { .. })
            ),
            "{name}: mismatched widths must be PrimType, never a silent coercion"
        );
        let width = mycelium_core::binary::DIV_MAX_WIDTH + 1;
        let one = wide_binary(width, &[width - 1]);
        assert!(
            matches!(f(name, &[&one, &one]), Err(EvalError::PrimType { .. })),
            "{name}: an over-cap width must be an explicit PrimType refusal"
        );
    }
}

/// `bin.shr_s` sign-extends (SMT-LIB `bvashr`): `-128 >> 4 = -8` (`0b1000_0000` → `0b1111_1000`),
/// where the logical `bin.shr` answers `+8` (`0b0000_1000`) — pinned side by side so the
/// signedness split is visible in one test. `-1` is a fixed point; non-negatives agree with the
/// logical shift.
#[test]
fn bin_shr_s_sign_extends_where_logical_shr_zero_fills() {
    let reg = PrimRegistry::with_builtins();
    let shr_s = reg.get("bin.shr_s").expect("bin.shr_s registered");
    let shr_u = reg.get("bin.shr").expect("bin.shr registered");
    let a = b8("1000_0000"); // -128 signed; 128 unsigned.
    let k = b8("0000_0100"); // shift by 4.
    let signed = shr_s("bin.shr_s", &[&a, &k]).expect("arithmetic shift");
    let logical = shr_u("bin.shr", &[&a, &k]).expect("logical shift");
    assert_eq!(signed.payload(), &Payload::Bits(bits("1111_1000")), "-8");
    assert_eq!(logical.payload(), &Payload::Bits(bits("0000_1000")), "+8");
    // -1 >> 3 = -1 (all-ones is a fixed point of sign extension).
    let r = shr_s("bin.shr_s", &[&b8("1111_1111"), &b8("0000_0011")]).expect("shift");
    assert_eq!(r.payload(), &Payload::Bits(bits("1111_1111")));
    // A non-negative value agrees with the logical shift: 64 >> 3 = 8.
    let r = shr_s("bin.shr_s", &[&b8("0100_0000"), &b8("0000_0011")]).expect("shift");
    assert_eq!(r.payload(), &Payload::Bits(bits("0000_1000")));
}

/// The shift-amount bound holds for the arithmetic shift exactly as for the logical one: `k >= N`
/// is an explicit `PrimType` refusal — never an implicit "all sign bits" result.
#[test]
fn bin_shr_s_out_of_range_amount_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.shr_s").expect("bin.shr_s registered");
    for k in ["0000_1000", "1111_1111"] {
        assert!(
            matches!(
                f("bin.shr_s", &[&b8("1111_1111"), &b8(k)]),
                Err(EvalError::PrimType { .. })
            ),
            "shift amount {k} >= width must refuse explicitly"
        );
    }
}

/// `cmp.lt_s` is the two's-complement order: `0b1111_1111` is `-1 < 0` here, while the D1
/// `cmp.lt` reads the same bits as `255 > 0` — the distinguishing case pinned against both prims.
#[test]
fn cmp_lt_s_orders_two_complement_where_lt_orders_magnitude() {
    let reg = PrimRegistry::with_builtins();
    let lt_s = reg.get("cmp.lt_s").expect("cmp.lt_s registered");
    let lt_u = reg.get("cmp.lt").expect("cmp.lt registered");
    let neg_one = b8("1111_1111");
    let zero = b8("0000_0000");
    let signed = lt_s("cmp.lt_s", &[&neg_one, &zero]).expect("signed order");
    let unsigned = lt_u("cmp.lt", &[&neg_one, &zero]).expect("unsigned order");
    assert_eq!(signed.payload(), &Payload::Bits(vec![true]), "-1 < 0");
    assert_eq!(unsigned.payload(), &Payload::Bits(vec![false]), "255 !< 0");
    // min < max; equal is not less; a positive pair agrees with the unsigned order.
    let cases = [
        ("1000_0000", "0111_1111", true),  // -128 < 127
        ("0111_1111", "1000_0000", false), // 127 !< -128
        ("0000_0101", "0000_0101", false), // 5 !< 5
        ("0000_0011", "0000_0101", true),  // 3 < 5
    ];
    for (a, b, expected) in cases {
        let r = lt_s("cmp.lt_s", &[&b8(a), &b8(b)]).expect("orderable");
        assert_eq!(r.payload(), &Payload::Bits(vec![expected]), "{a} lt_s {b}");
        assert_eq!(r.repr(), &Repr::Binary { width: 1 });
    }
}

/// `cmp.lt_s` refusal surface: a width mismatch, a ternary pair (its D1 order is already the
/// signed order — refused with that routing, never a silently duplicated order), and a
/// cross-paradigm pair are explicit `PrimType` refusals — never a silent `false` (G2).
#[test]
fn cmp_lt_s_refuses_non_binary_and_mismatched_operands() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("cmp.lt_s").expect("cmp.lt_s registered");
    // Width mismatch.
    assert!(
        matches!(
            f("cmp.lt_s", &[&s_bin(0, 8), &s_bin(0, 4)]),
            Err(EvalError::PrimType { .. })
        ),
        "mismatched widths must refuse, never a silent false"
    );
    // Ternary operands: the balanced-ternary D1 order is already signed — explicit routing.
    let t = trits("00+-");
    assert!(
        matches!(f("cmp.lt_s", &[&t, &t]), Err(EvalError::PrimType { .. })),
        "a ternary pair must refuse with the cmp.lt routing"
    );
    // Cross-paradigm.
    assert!(
        matches!(
            f("cmp.lt_s", &[&s_bin(0, 4), &trits("00+-")]),
            Err(EvalError::PrimType { .. })
        ),
        "a cross-paradigm pair must refuse"
    );
}

// ── M-890 (`enb` Gap C): the dense elementwise prim group ───────────────────────────────────────
//
// `dense.add`/`dense.sub`/`dense.neg`/`dense.scale` — the first tensor-valued prims. The kernel
// (`mycelium-dense`) constructs the result `Value` with its honest per-op tag; the wrapper carries
// it through unchanged (VR-5). These tests pin: (1) the Π-table intrinsic ↔ kernel `op_guarantee`
// consistency (the cross-crate guard `mycelium-core` cannot host), (2) accept-path payloads +
// carried tags/bounds, (3) the never-silent reject surface (shape/dtype mismatch, overflow,
// approximate sources, malformed scale factors), and (4) the per-element relative-error bound
// against an exact f64 oracle (the cheap property the `Proven` tag discloses).

use mycelium_core::{Bound, BoundBasis, BoundKind, NormKind, PrimTable, ScalarKind};
use mycelium_dense::{DenseOp, DenseSpace};

/// A `Dense{n, F32}` value from on-grid elements (test fixture).
fn dense_f32(xs: Vec<f64>) -> Value {
    let n = u32::try_from(xs.len()).expect("test dims are small");
    DenseSpace::new(n, ScalarKind::F32)
        .expect("F32 is a supported dtype")
        .value(xs)
        .expect("fixture elements are finite and on-grid")
}

/// The Π-table intrinsic must equal the kernel's per-op tag — the VR-5 "carried, never upgraded"
/// contract, guarded here because `mycelium-core` (Π) cannot depend on `mycelium-dense` (kernel).
#[test]
fn dense_prim_table_intrinsics_match_the_kernel_op_guarantees() {
    let table = PrimTable::builtins();
    for (name, op) in [
        ("dense.add", DenseOp::Add),
        ("dense.sub", DenseOp::Sub),
        ("dense.neg", DenseOp::Neg),
        ("dense.scale", DenseOp::Scale),
        ("dense.dot", DenseOp::Dot),
        ("dense.similarity", DenseOp::Similarity),
    ] {
        assert_eq!(
            table.intrinsic(name),
            Some(DenseSpace::op_guarantee(op)),
            "{name}: the Π intrinsic must be carried verbatim from DenseSpace::op_guarantee"
        );
    }
}

#[test]
fn dense_add_carries_the_kernel_proven_tag_and_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.add").expect("dense.add registered");
    let a = dense_f32(vec![1.5, 2.5]);
    let b = dense_f32(vec![0.25, -1.0]);
    let y = f("dense.add", &[&a, &b]).expect("in-range add");
    assert_eq!(y.payload(), &Payload::Scalars(vec![1.75, 1.5]));
    // The tag is the KERNEL's, carried unchanged: Proven + the per-element relative ε under a
    // ProvenThm basis (never re-derived by compose_result, whose intrinsic is Exact).
    assert_eq!(
        y.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    let space = DenseSpace::new(2, ScalarKind::F32).unwrap();
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { .. },
        }) => {
            assert_eq!(
                *eps,
                space.op_rel_eps(),
                "ε must be the kernel's op_rel_eps"
            );
            assert_eq!(*norm, NormKind::Rel);
        }
        other => panic!("expected the kernel's ProvenThm Error bound, got {other:?}"),
    }
    // Provenance is the kernel's Derived{op: hash("dense.add"), inputs}.
    match y.meta().provenance() {
        Provenance::Derived { op, inputs } => {
            assert_eq!(op, &mycelium_core::operation_hash("dense.add"));
            assert_eq!(inputs, &vec![a.content_hash(), b.content_hash()]);
        }
        other => panic!("expected Derived provenance, got {other:?}"),
    }
}

#[test]
fn dense_sub_and_neg_accept_paths() {
    let reg = PrimRegistry::with_builtins();
    let sub = reg.get("dense.sub").expect("dense.sub registered");
    let neg = reg.get("dense.neg").expect("dense.neg registered");
    let a = dense_f32(vec![1.5, 2.5]);
    let b = dense_f32(vec![0.5, -1.0]);
    let d = sub("dense.sub", &[&a, &b]).expect("in-range sub");
    assert_eq!(d.payload(), &Payload::Scalars(vec![1.0, 3.5]));
    assert_eq!(
        d.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    // neg is Exact (the grids are symmetric — never rounds) with no bound.
    let n = neg("dense.neg", &[&a]).expect("neg is total over on-grid inputs");
    assert_eq!(n.payload(), &Payload::Scalars(vec![-1.5, -2.5]));
    assert_eq!(
        n.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Exact
    );
    assert!(n.meta().bound().is_none(), "Exact results carry no bound");
}

#[test]
fn dense_scale_takes_a_dense1_factor() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.scale").expect("dense.scale registered");
    let a = dense_f32(vec![1.5, -2.0]);
    let c = dense_f32(vec![2.0]); // the pre-Gap-A scalar form: Dense{1, same dtype}
    let y = f("dense.scale", &[&a, &c]).expect("on-grid scale");
    assert_eq!(y.payload(), &Payload::Scalars(vec![3.0, -4.0]));
    assert_eq!(
        y.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
}

#[test]
fn dense_shape_mismatch_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.add").expect("dense.add registered");
    let a = dense_f32(vec![1.0, 2.0]);
    let b3 = dense_f32(vec![1.0, 2.0, 3.0]);
    // Dim mismatch → explicit PrimType naming expected/got — never a broadcast (G2).
    let err = f("dense.add", &[&a, &b3]).expect_err("dim mismatch must refuse");
    match err {
        EvalError::PrimType { prim, why } => {
            assert_eq!(prim, "dense.add");
            assert!(
                why.contains("dimension mismatch"),
                "the refusal must name the shape mismatch: {why}"
            );
        }
        other => panic!("expected PrimType, got {other:?}"),
    }
    // Dtype mismatch → explicit PrimType, never a re-round.
    let bf = DenseSpace::new(2, ScalarKind::Bf16)
        .unwrap()
        .value(vec![1.5, -2.0])
        .unwrap();
    assert!(
        matches!(f("dense.add", &[&a, &bf]), Err(EvalError::PrimType { .. })),
        "dtype mismatch must be an explicit refusal"
    );
    // A non-Dense operand → explicit PrimType.
    let bits = byte([true; 8]);
    assert!(
        matches!(
            f("dense.add", &[&bits, &a]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Dense first operand must be an explicit refusal"
    );
    assert!(
        matches!(
            f("dense.add", &[&a, &bits]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Dense second operand must be an explicit refusal"
    );
    // Wrong arity → explicit PrimType.
    assert!(matches!(
        f("dense.add", &[&a]),
        Err(EvalError::PrimType { .. })
    ));
}

#[test]
fn dense_overflow_and_approx_sources_refuse_explicitly() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.add").expect("dense.add registered");
    // Overflow: f32::MAX + f32::MAX exceeds the dtype's finite range → EvalError::Overflow.
    let max = dense_f32(vec![f64::from(f32::MAX)]);
    assert!(
        matches!(
            f("dense.add", &[&max, &max]),
            Err(EvalError::Overflow { .. })
        ),
        "an out-of-range result must be an explicit Overflow, never ±Inf"
    );
    // An approximate source has no defined composition rule (M-204/M-211) →
    // ApproxCompositionUnsupported — carried from the kernel's ApproximateSource refusal.
    let a = dense_f32(vec![1.0, 2.0]);
    let approx = f("dense.add", &[&a, &dense_f32(vec![0.5, 0.5])]).expect("a Proven value");
    assert!(
        matches!(
            f("dense.add", &[&a, &approx]),
            Err(EvalError::ApproxCompositionUnsupported { .. })
        ),
        "an approximate (Proven) source must refuse — no composition rule yet"
    );
}

#[test]
fn dense_scale_factor_contract_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.scale").expect("dense.scale registered");
    let a = dense_f32(vec![1.5, -2.0]);
    // A non-Dense{1} factor (wrong dim) → explicit PrimType.
    let c2 = dense_f32(vec![2.0, 2.0]);
    assert!(
        matches!(
            f("dense.scale", &[&a, &c2]),
            Err(EvalError::PrimType { .. })
        ),
        "a Dense{{2}} factor must refuse — the scalar form is Dense{{1}}"
    );
    // A non-Dense factor → explicit PrimType.
    let bits = byte([false; 8]);
    assert!(matches!(
        f("dense.scale", &[&a, &bits]),
        Err(EvalError::PrimType { .. })
    ));
    // A factor of the wrong dtype → explicit PrimType (never a silent re-round).
    let cbf = DenseSpace::new(1, ScalarKind::Bf16)
        .unwrap()
        .value(vec![2.0])
        .unwrap();
    assert!(matches!(
        f("dense.scale", &[&a, &cbf]),
        Err(EvalError::PrimType { .. })
    ));
    // An approximate factor → ApproxCompositionUnsupported (no defined composition rule).
    let one = dense_f32(vec![1.0]);
    let approx_c = reg.get("dense.add").unwrap()("dense.add", &[&one, &one]).expect("Proven");
    assert!(matches!(
        f("dense.scale", &[&a, &approx_c]),
        Err(EvalError::ApproxCompositionUnsupported { .. })
    ));
}

/// **Property test (the disclosed bound):** over an on-grid corpus, each `dense.add`/`dense.sub`
/// result element differs from the exact `f64` oracle by at most the disclosed per-element
/// relative ε (`op_rel_eps` — the very bound the carried `Proven` tag claims), and
/// `dense.neg` is an exact involution (`neg(neg(x)) == x`, the `Exact` claim). Loop-corpus style,
/// mirroring `bin_mul_matches_integer_oracle_at_width6`.
#[test]
fn dense_elementwise_results_respect_the_disclosed_relative_bound() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("dense.add").unwrap();
    let sub = reg.get("dense.sub").unwrap();
    let neg = reg.get("dense.neg").unwrap();
    let space = DenseSpace::new(4, ScalarKind::F32).unwrap();
    let eps = space.op_rel_eps();
    // On-grid f32 corpus spanning magnitudes and signs (all exactly representable in f32).
    let corpus: [[f64; 4]; 4] = [
        [1.5, -0.625, 1024.0, -3.25],
        [0.25, 7.5, -0.03125, 2048.0],
        [-1.0, 0.5, 100.5, -0.75],
        [3.0, -12.25, 0.125, 640.0],
    ];
    for xs in &corpus {
        for ys in &corpus {
            let a = dense_f32(xs.to_vec());
            let b = dense_f32(ys.to_vec());
            for (prim, f, oracle) in [
                ("dense.add", add, (|x, y| x + y) as fn(f64, f64) -> f64),
                ("dense.sub", sub, (|x, y| x - y) as fn(f64, f64) -> f64),
            ] {
                let y = f(prim, &[&a, &b]).expect("corpus results are in range");
                let Payload::Scalars(out) = y.payload() else {
                    panic!("{prim} must return Payload::Scalars")
                };
                for (i, (&got, (&x, &yv))) in out.iter().zip(xs.iter().zip(ys)).enumerate() {
                    let exact = oracle(x, yv);
                    // |got − exact| ≤ ε·|exact|: the per-element relative bound the Proven tag
                    // discloses (exact == 0 ⇒ got must be exactly 0 — no absolute slack).
                    assert!(
                        (got - exact).abs() <= eps * exact.abs(),
                        "{prim} element {i}: |{got} − {exact}| exceeds ε·|exact| (ε = {eps})"
                    );
                }
            }
            // Exact involution: neg(neg(a)) == a, payload-identical (the Exact claim).
            let n1 = neg("dense.neg", &[&a]).expect("neg is total");
            let n2 = neg("dense.neg", &[&n1]).expect("neg is total");
            assert_eq!(
                n2.payload(),
                a.payload(),
                "dense.neg must be an exact involution"
            );
        }
    }
}

// ── M-891 (`enb` Gap C): the dense measurement pair `dense.dot`/`dense.similarity` ──────────────
//
// The kernel constructs the `Dense{1, F64}` measurement value with its honest per-op tag —
// `Proven` with the **binary64 accumulation bound** (absolute/`Linf`, `dot_abs_eps`/
// `similarity_abs_eps` — deliberately NOT the dtype's per-element `op_rel_eps`; see the module
// note in `prims.rs`) — and the wrapper carries it through unchanged (VR-5). These tests pin:
// (1) the Π ↔ kernel tag consistency (folded into the M-890 guard above), (2) the accept-path
// payload + the carried tag/bound/provenance, (3) the EXPLAIN inspectability of the disclosed ε +
// its ProvenThm citation off the result value itself, (4) the never-silent reject surface, and
// (5) the disclosed-bound property over analytically-known dots, including the cancellation case
// a per-element relative claim would fail.

/// M-891 accept path: `dense.dot` returns the f64 measurement as `Dense{1, F64}` with the
/// kernel's `Proven` accumulation bound — and that bound is **EXPLAIN-able**: guarantee, ε,
/// norm, and the ProvenThm citation are all inspectable off the value's `Meta` (G2/SC-3).
#[test]
fn dense_dot_carries_the_inspectable_accumulation_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.dot").expect("dense.dot registered");
    let a = dense_f32(vec![1.5, 2.0, -0.5]);
    let b = dense_f32(vec![2.0, 0.25, 4.0]);
    let y = f("dense.dot", &[&a, &b]).expect("in-range dot");
    // 3.0 + 0.5 − 2.0 = 1.5 (every product and partial sum exact in f64).
    assert_eq!(
        y.repr(),
        &Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64
        },
        "the measurement result form is Dense{{1, F64}}"
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![1.5]));
    assert_eq!(
        y.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    let space = DenseSpace::new(3, ScalarKind::F32).unwrap();
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { citation },
        }) => {
            // ε is the kernel's disclosed absolute accumulation bound over the computed
            // abs-product sum (3.0 + 0.5 + 2.0) — NOT op_rel_eps (the dtype ε never enters).
            assert_eq!(*eps, space.dot_abs_eps(3.0 + 0.5 + 2.0));
            assert_eq!(*norm, NormKind::Linf);
            assert!(
                citation.contains("Higham"),
                "the EXPLAIN-able citation must name its theorem basis: {citation}"
            );
        }
        other => panic!("expected the kernel's ProvenThm Linf bound, got {other:?}"),
    }
    match y.meta().provenance() {
        Provenance::Derived { op, inputs } => {
            assert_eq!(op, &mycelium_core::operation_hash("dense.dot"));
            assert_eq!(inputs, &vec![a.content_hash(), b.content_hash()]);
        }
        other => panic!("expected Derived provenance, got {other:?}"),
    }
}

#[test]
fn dense_similarity_accept_paths_and_zero_convention() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.similarity").expect("registered");
    let a = dense_f32(vec![1.0, 0.0]);
    let b = dense_f32(vec![0.0, 1.0]);
    let space = DenseSpace::new(2, ScalarKind::F32).unwrap();
    // Orthogonal → exactly 0 (products are 0 each).
    let y = f("dense.similarity", &[&a, &b]).expect("similarity is total over on-grid inputs");
    assert_eq!(
        y.repr(),
        &Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64
        }
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![0.0]));
    assert_eq!(
        y.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { .. },
        }) => {
            assert_eq!(*eps, space.similarity_abs_eps());
            assert_eq!(*norm, NormKind::Linf);
        }
        other => panic!("expected the kernel's ProvenThm Linf bound, got {other:?}"),
    }
    // Self-similarity is 1 within the disclosed ε.
    let s = f("dense.similarity", &[&a, &a]).expect("self-similarity");
    let Payload::Scalars(sim) = s.payload() else {
        panic!("similarity must return scalars")
    };
    assert!((sim[0] - 1.0).abs() <= space.similarity_abs_eps());
    // The zero-norm convention (documented in the citation): exactly 0, never silent.
    let z = dense_f32(vec![0.0, 0.0]);
    let zc = f("dense.similarity", &[&a, &z]).expect("zero-norm convention");
    assert_eq!(zc.payload(), &Payload::Scalars(vec![0.0]));
}

#[test]
fn dense_measurement_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    for prim in ["dense.dot", "dense.similarity"] {
        let f = reg.get(prim).expect("registered");
        let a = dense_f32(vec![1.0, 2.0]);
        // Dim mismatch → explicit PrimType naming the offense — never a broadcast (G2).
        let b3 = dense_f32(vec![1.0, 2.0, 3.0]);
        let err = f(prim, &[&a, &b3]).expect_err("dim mismatch must refuse");
        match err {
            EvalError::PrimType { prim: p, why } => {
                assert_eq!(p, prim);
                assert!(why.contains("dimension mismatch"), "{prim}: {why}");
            }
            other => panic!("{prim}: expected PrimType, got {other:?}"),
        }
        // Dtype mismatch → explicit PrimType, never a re-round.
        let bf = DenseSpace::new(2, ScalarKind::Bf16)
            .unwrap()
            .value(vec![1.5, -2.0])
            .unwrap();
        assert!(matches!(
            f(prim, &[&a, &bf]),
            Err(EvalError::PrimType { .. })
        ));
        // A non-Dense operand (either side) → explicit PrimType.
        let bits = byte([true; 8]);
        assert!(matches!(
            f(prim, &[&bits, &a]),
            Err(EvalError::PrimType { .. })
        ));
        assert!(matches!(
            f(prim, &[&a, &bits]),
            Err(EvalError::PrimType { .. })
        ));
        // Wrong arity → explicit PrimType.
        assert!(matches!(f(prim, &[&a]), Err(EvalError::PrimType { .. })));
        // An approximate source → ApproxCompositionUnsupported (no composition rule yet).
        let approx = reg.get("dense.add").unwrap()("dense.add", &[&a, &dense_f32(vec![0.5, 0.5])])
            .expect("a Proven value");
        assert!(matches!(
            f(prim, &[&a, &approx]),
            Err(EvalError::ApproxCompositionUnsupported { .. })
        ));
    }
}

/// **Property test (the disclosed bound):** over cases whose *true* real-arithmetic dot is known
/// analytically, the computed payload differs from the truth by at most the ε **the value's own
/// bound discloses** — including the catastrophic-cancellation case (`fl(2⁶⁰ + 1) = 2⁶⁰`, so the
/// computed dot is 0 against a true 1) where a per-element relative claim (`op_rel_eps`) would be
/// flat-out false. The absolute accumulation bound must (and does) cover it.
#[test]
fn dense_dot_respects_its_own_disclosed_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("dense.dot").expect("registered");
    let two30 = f64::from(2f32.powi(30));
    let cases: [(&[f64], &[f64], f64); 4] = [
        (&[1.5, 2.0, -0.5], &[2.0, 0.25, 4.0], 1.5),
        (&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0], 20.0),
        (&[0.0, 0.0], &[1.0, -1.0], 0.0),
        (&[two30, 1.0, -two30], &[two30, 1.0, two30], 1.0),
    ];
    for (xs, ys, exact) in cases {
        let a = dense_f32(xs.to_vec());
        let b = dense_f32(ys.to_vec());
        let y = f("dense.dot", &[&a, &b]).expect("in-range dot");
        let Payload::Scalars(out) = y.payload() else {
            panic!("dense.dot must return Payload::Scalars")
        };
        let Some(Bound {
            kind: BoundKind::Error { eps, .. },
            ..
        }) = y.meta().bound()
        else {
            panic!("dense.dot must carry its Error bound")
        };
        assert!(
            (out[0] - exact).abs() <= *eps,
            "|{} − {exact}| exceeds the value's own disclosed ε = {eps}",
            out[0]
        );
    }
}

// ---- ADR-040 §2.5 (M-898, `enb` Gap A): the scalar-float arithmetic group ----------------------
//
// The reference-case corpus below is the **evidence** behind the `EmpiricalFit` basis every
// `flt.*` result carries (ADR-040 §2.6): every expected value is **hand-derived from IEEE-754
// binary64 RNE semantics** (exact-arithmetic rows, ties-to-even at the 2^53 boundary,
// overflow/underflow edges, signed zeros, the specials algebra, canonical-NaN identity) and
// written as an independent literal/constant — never recomputed with the op under test. The
// corpus row count is pinned to `FLT_CONFORMANCE_TRIALS`, so the trials the basis *records*
// equal the trials actually *run* (VR-5 — evidence never drifts from the claim).

use mycelium_core::{FloatWidth, GuaranteeStrength, CANONICAL_NAN_BITS};

/// An `Exact` `Float{F64}` value (the M-897 float-literal form — the ops' normal input).
fn fv(x: f64) -> Value {
    Value::new(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(x),
        Meta::exact(Provenance::Root),
    )
    .expect("a Float payload matches a Float repr")
}

/// The canonical quiet NaN (the single NaN identity — ADR-040 §2.3).
fn cnan() -> f64 {
    f64::from_bits(CANONICAL_NAN_BITS)
}

/// One reference row: op, operands, and the hand-derived expected bit pattern.
struct FltCase {
    op: &'static str,
    args: Vec<f64>,
    expected: f64,
    why: &'static str,
}

/// The M-898 IEEE-754 binary64 RNE reference corpus (see the section note). Exactly
/// [`FLT_CONFORMANCE_TRIALS`] rows — asserted by `flt_reference_case_corpus`.
fn flt_reference_cases() -> Vec<FltCase> {
    let c = |op, args: &[f64], expected, why| FltCase {
        op,
        args: args.to_vec(),
        expected,
        why,
    };
    vec![
        // flt.add — exact-arithmetic rows (all operands and results on the dyadic grid).
        c("flt.add", &[1.5, 2.25], 3.75, "exact dyadic sum"),
        c("flt.add", &[0.5, 0.25], 0.75, "exact dyadic sum"),
        c(
            "flt.add",
            &[-1.5, 1.5],
            0.0,
            "IEEE 6.3: x + (−x) is +0 under RNE",
        ),
        c(
            "flt.add",
            &[-0.0, -0.0],
            -0.0,
            "IEEE 6.3: (−0) + (−0) is −0",
        ),
        c(
            "flt.add",
            &[-0.0, 0.0],
            0.0,
            "IEEE 6.3: opposite-signed zeros sum to +0 under RNE",
        ),
        // Ties-to-even at the 2^53 representability edge (spacing 2): 2^53 + 1 is the midpoint
        // of {2^53, 2^53 + 2} → the even mantissa (2^53) wins; (2^53 + 2) + 1 is the midpoint of
        // {2^53 + 2, 2^53 + 4} → the even mantissa (2^53 + 4) wins.
        c(
            "flt.add",
            &[9_007_199_254_740_992.0, 1.0],
            9_007_199_254_740_992.0,
            "RNE tie at 2^53: midpoint rounds to the even mantissa (down)",
        ),
        c(
            "flt.add",
            &[9_007_199_254_740_994.0, 1.0],
            9_007_199_254_740_996.0,
            "RNE tie at 2^53 + 3: midpoint rounds to the even mantissa (up)",
        ),
        c(
            "flt.add",
            &[f64::MAX, f64::MAX],
            f64::INFINITY,
            "overflow → +inf, in-band (ratified FLAG-2)",
        ),
        c(
            "flt.add",
            &[-f64::MAX, -f64::MAX],
            f64::NEG_INFINITY,
            "overflow → −inf, in-band",
        ),
        c(
            "flt.add",
            &[f64::INFINITY, 1.0],
            f64::INFINITY,
            "inf + finite = inf",
        ),
        c(
            "flt.add",
            &[f64::INFINITY, f64::NEG_INFINITY],
            cnan(),
            "inf + (−inf) is invalid → NaN (canonical)",
        ),
        c(
            "flt.add",
            &[cnan(), 1.0],
            cnan(),
            "NaN propagates (canonical)",
        ),
        // flt.sub.
        c("flt.sub", &[3.75, 1.5], 2.25, "exact dyadic difference"),
        c("flt.sub", &[1.0, 1.0], 0.0, "x − x is +0 under RNE"),
        c("flt.sub", &[0.0, 0.0], 0.0, "(+0) − (+0) is +0 under RNE"),
        c(
            "flt.sub",
            &[-0.0, 0.0],
            -0.0,
            "(−0) − (+0) is (−0) + (−0) = −0",
        ),
        c(
            "flt.sub",
            &[f64::INFINITY, f64::INFINITY],
            cnan(),
            "inf − inf is invalid → NaN (canonical)",
        ),
        c(
            "flt.sub",
            &[1.0, f64::INFINITY],
            f64::NEG_INFINITY,
            "finite − inf = −inf",
        ),
        c(
            "flt.sub",
            &[f64::MAX, -f64::MAX],
            f64::INFINITY,
            "overflow → +inf, in-band",
        ),
        // flt.mul.
        c("flt.mul", &[1.5, 2.0], 3.0, "exact dyadic product"),
        c(
            "flt.mul",
            &[-1.5, 2.0],
            -3.0,
            "exact dyadic product, sign rule",
        ),
        c("flt.mul", &[0.5, 0.5], 0.25, "exact dyadic product"),
        c(
            "flt.mul",
            &[f64::MAX, 2.0],
            f64::INFINITY,
            "overflow → +inf, in-band",
        ),
        c(
            "flt.mul",
            &[0.0, f64::INFINITY],
            cnan(),
            "0 × inf is invalid → NaN (canonical)",
        ),
        c(
            "flt.mul",
            &[-1.0, 0.0],
            -0.0,
            "IEEE sign rule: (−1) × (+0) = −0",
        ),
        c(
            "flt.mul",
            &[f64::INFINITY, -2.0],
            f64::NEG_INFINITY,
            "inf × negative = −inf",
        ),
        // Underflow at the subnormal floor (spacing 2^-1074): (2^-1074) × 0.5 = 2^-1075 is the
        // midpoint of {0, 2^-1074} → the even candidate (0) wins under RNE.
        c(
            "flt.mul",
            &[5e-324, 0.5],
            0.0,
            "RNE tie at the subnormal floor: midpoint rounds to the even candidate 0",
        ),
        // flt.div.
        c("flt.div", &[3.0, 2.0], 1.5, "exact dyadic quotient"),
        c(
            "flt.div",
            &[1.0, 0.0],
            f64::INFINITY,
            "div-by-zero → +inf, in-band (never a trap — ratified FLAG-2)",
        ),
        c(
            "flt.div",
            &[-1.0, 0.0],
            f64::NEG_INFINITY,
            "div-by-zero, sign rule → −inf",
        ),
        c(
            "flt.div",
            &[1.0, -0.0],
            f64::NEG_INFINITY,
            "div by −0, sign rule → −inf (−0 is observably distinct — ADR-040 §2.3)",
        ),
        c(
            "flt.div",
            &[0.0, 0.0],
            cnan(),
            "0/0 is invalid → NaN (canonical)",
        ),
        c(
            "flt.div",
            &[f64::INFINITY, f64::INFINITY],
            cnan(),
            "inf/inf is invalid → NaN (canonical)",
        ),
        c("flt.div", &[1.0, f64::INFINITY], 0.0, "finite/inf = +0"),
        c(
            "flt.div",
            &[-1.0, f64::INFINITY],
            -0.0,
            "finite/inf, sign rule = −0",
        ),
        // flt.neg — sign-bit flip (exact; never rounds).
        c("flt.neg", &[1.5], -1.5, "sign flip"),
        c(
            "flt.neg",
            &[0.0],
            -0.0,
            "neg(+0) = −0 (bit-distinct — ADR-040 §2.3)",
        ),
        c("flt.neg", &[-0.0], 0.0, "neg(−0) = +0"),
        c(
            "flt.neg",
            &[f64::INFINITY],
            f64::NEG_INFINITY,
            "neg(inf) = −inf",
        ),
        c(
            "flt.neg",
            &[cnan()],
            cnan(),
            "neg(NaN) re-canonicalizes: NaN sign/payload bits are not observable (§2.3)",
        ),
    ]
}

/// **The conformance corpus (the `EmpiricalFit` evidence):** every row's delivered bit pattern
/// equals its hand-derived IEEE-754 RNE reference **bit-for-bit** (a payload `==` would pass
/// `-0.0 == 0.0` and fail NaN — bits do neither), and the row count equals the
/// `FLT_CONFORMANCE_TRIALS` the basis records.
#[test]
fn flt_reference_case_corpus() {
    let reg = PrimRegistry::with_builtins();
    let cases = flt_reference_cases();
    assert_eq!(
        cases.len() as u64,
        FLT_CONFORMANCE_TRIALS,
        "the recorded trials must equal the trials actually run (VR-5)"
    );
    for case in &cases {
        let f = reg.get(case.op).expect("flt prim registered");
        let args: Vec<Value> = case.args.iter().copied().map(fv).collect();
        let argrefs: Vec<&Value> = args.iter().collect();
        let y = f(case.op, &argrefs)
            .unwrap_or_else(|e| panic!("{}({:?}) must be total, got {e:?}", case.op, case.args));
        let Payload::Float(x) = y.payload() else {
            panic!("{}: result payload must be Float", case.op)
        };
        assert_eq!(
            x.to_bits(),
            case.expected.to_bits(),
            "{}({:?}): got {x:?}, want {:?} — {}",
            case.op,
            case.args,
            case.expected,
            case.why
        );
        assert_eq!(
            y.repr(),
            &Repr::Float {
                width: FloatWidth::F64
            },
            "{}: result repr must be Float{{F64}}",
            case.op
        );
    }
}

/// A value corpus for the property sweeps: finite grid points (exact + inexact decimals),
/// signed zeros, subnormals, the finite extremes, both infinities, and the canonical NaN.
fn flt_value_corpus() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.5,
        -2.5,
        0.1,
        0.2,
        1.0 / 3.0,
        1e10,
        -1e-300,
        5e-324,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        9_007_199_254_740_992.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        cnan(),
    ]
}

/// **Property (commutativity, bit-exact):** `flt.add`/`flt.mul` are commutative bit-for-bit over
/// the whole corpus — including specials and NaN, because every NaN result is canonical (one NaN,
/// one bit pattern; ADR-040 §2.3 is what makes float commutativity *bit*-exact, not just IEEE-==).
#[test]
fn flt_add_mul_commute_bitwise_on_the_corpus() {
    let reg = PrimRegistry::with_builtins();
    for op in ["flt.add", "flt.mul"] {
        let f = reg.get(op).expect("registered");
        for &a in &flt_value_corpus() {
            for &b in &flt_value_corpus() {
                let (va, vb) = (fv(a), fv(b));
                let xy = f(op, &[&va, &vb]).expect("total");
                let yx = f(op, &[&vb, &va]).expect("total");
                let (Payload::Float(p), Payload::Float(q)) = (xy.payload(), yx.payload()) else {
                    panic!("{op}: float results expected")
                };
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "{op}({a:?}, {b:?}) must commute bit-exactly"
                );
            }
        }
    }
}

/// **Property (additive identity):** `x + 0.0` is IEEE-equal to `x` for every non-NaN `x`, and
/// bit-identical for every `x` except `−0.0` (where IEEE itself defines `−0 + (+0) = +0` under
/// RNE — the documented identity-vs-equality seam, ADR-040 FLAG-4).
#[test]
fn flt_add_zero_is_the_identity_modulo_ieee() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("flt.add").expect("registered");
    let zero = fv(0.0);
    for &x in &flt_value_corpus() {
        let vx = fv(x);
        let y = f("flt.add", &[&vx, &zero]).expect("total");
        let Payload::Float(out) = y.payload() else {
            panic!("float result expected")
        };
        if x.is_nan() {
            assert_eq!(
                out.to_bits(),
                CANONICAL_NAN_BITS,
                "NaN + 0 is canonical NaN"
            );
        } else {
            assert_eq!(*out, x, "x + 0.0 must be IEEE-equal to x (x = {x:?})");
            if x.to_bits() != (-0.0f64).to_bits() {
                assert_eq!(out.to_bits(), x.to_bits(), "bit-identity for x ≠ −0.0");
            }
        }
    }
}

/// **Property (involution):** `flt.neg ∘ flt.neg` is a bit-identity over the whole corpus — the
/// signed zeros round-trip (`+0 → −0 → +0`), the infinities round-trip, and NaN re-canonicalizes
/// to itself.
#[test]
fn flt_neg_neg_is_a_bit_identity() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("flt.neg").expect("registered");
    for &x in &flt_value_corpus() {
        let vx = fv(x);
        let once = f("flt.neg", &[&vx]).expect("total");
        let twice = f("flt.neg", &[&once]).expect("total");
        let Payload::Float(out) = twice.payload() else {
            panic!("float result expected")
        };
        assert_eq!(
            out.to_bits(),
            x.to_bits(),
            "neg(neg({x:?})) must be a bit-identity"
        );
    }
}

/// **Property (one NaN, one address — ADR-040 §2.3):** every NaN any `flt.*` op produces over the
/// corpus carries exactly the canonical bits — no constructor path yields a non-canonical NaN.
#[test]
fn flt_nan_results_are_always_canonical() {
    let reg = PrimRegistry::with_builtins();
    for op in ["flt.add", "flt.sub", "flt.mul", "flt.div"] {
        let f = reg.get(op).expect("registered");
        for &a in &flt_value_corpus() {
            for &b in &flt_value_corpus() {
                let (va, vb) = (fv(a), fv(b));
                let y = f(op, &[&va, &vb]).expect("total");
                let Payload::Float(out) = y.payload() else {
                    panic!("float result expected")
                };
                if out.is_nan() {
                    assert_eq!(
                        out.to_bits(),
                        CANONICAL_NAN_BITS,
                        "{op}({a:?}, {b:?}): NaN must be canonical"
                    );
                }
            }
        }
    }
}

/// **The ADR-040 §2.6 tag contract, inspectable off the value (EXPLAIN — G2/SC-3):** every
/// `flt.*` result over `Exact` inputs is `Empirical` with the zero-deviation-vs-spec bound
/// (`eps = 0`, `Linf`) on the `EmpiricalFit{FLT_CONFORMANCE_TRIALS, …}` basis, with `Derived`
/// provenance — and the Π table's intrinsic agrees with what the wrapper delivers (the DN-10
/// §3.4 table↔kernel consistency, float form).
#[test]
fn flt_results_carry_the_adr040_empirical_tag_and_bound() {
    let reg = PrimRegistry::with_builtins();
    let table = PrimTable::builtins();
    let one = fv(1.0);
    let half = fv(0.5);
    for op in ["flt.add", "flt.sub", "flt.mul", "flt.div", "flt.neg"] {
        let f = reg.get(op).expect("registered");
        let args: Vec<&Value> = if op == "flt.neg" {
            vec![&half]
        } else {
            vec![&one, &half]
        };
        let y = f(op, &args).expect("total");
        assert_eq!(
            y.meta().guarantee(),
            GuaranteeStrength::Empirical,
            "{op}: the per-op tag is the ratified ADR-040 §2.6 Empirical (VR-5)"
        );
        assert_eq!(
            table.intrinsic(op),
            Some(GuaranteeStrength::Empirical),
            "{op}: Π intrinsic must agree with the delivered tag (DN-10 §3.4)"
        );
        match y.meta().bound() {
            Some(Bound {
                kind: BoundKind::Error { eps, norm },
                basis: BoundBasis::EmpiricalFit { trials, method },
            }) => {
                assert_eq!(*eps, 0.0, "{op}: zero deviation vs the RNE spec");
                assert_eq!(*norm, NormKind::Linf);
                assert_eq!(
                    *trials, FLT_CONFORMANCE_TRIALS,
                    "{op}: the basis records the corpus actually run"
                );
                assert!(!method.trim().is_empty());
            }
            other => panic!("{op}: expected the EmpiricalFit zero-deviation bound, got {other:?}"),
        }
        assert!(
            matches!(y.meta().provenance(), Provenance::Derived { .. }),
            "{op}: provenance must be Derived"
        );
    }
}

/// **Composition:** a `flt.*` result (Empirical, zero-deviation) is a legal input to the next
/// `flt.*` op — chained float arithmetic composes, and the chained result keeps the same honest
/// tag/bound form. An input carrying a *genuine* approximation bound (`eps > 0`) is an explicit
/// [`EvalError::ApproxCompositionUnsupported`] — no defined float ε-rule yet, refused, never
/// fabricated (G2/VR-5).
#[test]
fn flt_chaining_composes_and_true_approximations_refuse() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("flt.add").expect("registered");
    let mul = reg.get("flt.mul").expect("registered");
    // Chain: (1.5 × 2.0) + 0.25 = 3.25 — the intermediate is Empirical and composes.
    let prod = mul("flt.mul", &[&fv(1.5), &fv(2.0)]).expect("total");
    assert_eq!(prod.meta().guarantee(), GuaranteeStrength::Empirical);
    let sum = add("flt.add", &[&prod, &fv(0.25)]).expect("chained flt ops must compose");
    let Payload::Float(out) = sum.payload() else {
        panic!("float result expected")
    };
    assert_eq!(out.to_bits(), 3.25f64.to_bits());
    assert_eq!(sum.meta().guarantee(), GuaranteeStrength::Empirical);
    // A genuinely-approximate Float input (eps > 0) has no defined propagation rule — refuse.
    let approx = Value::new(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(1.0),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Empirical,
            Some(Bound {
                kind: BoundKind::Error {
                    eps: 1e-3,
                    norm: NormKind::Rel,
                },
                basis: BoundBasis::EmpiricalFit {
                    trials: 10,
                    method: "a synthetic approximate source".to_owned(),
                },
            }),
            None,
            None,
            None,
        )
        .expect("well-formed meta"),
    )
    .expect("well-formed value");
    assert!(
        matches!(
            add("flt.add", &[&approx, &fv(1.0)]),
            Err(EvalError::ApproxCompositionUnsupported { .. })
        ),
        "a true approximation must refuse explicitly, never a fabricated bound"
    );
}

/// **Never-silent type/arity discipline:** a non-`Float` operand and a wrong arity are explicit
/// [`EvalError::PrimType`] refusals — never a coercion (G2).
#[test]
fn flt_type_and_arity_refusals_are_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("flt.add").expect("registered");
    let neg = reg.get("flt.neg").expect("registered");
    let b = byte([false; 8]);
    let x = fv(1.0);
    assert!(
        matches!(add("flt.add", &[&b, &x]), Err(EvalError::PrimType { .. })),
        "a Binary operand must refuse"
    );
    assert!(
        matches!(add("flt.add", &[&x]), Err(EvalError::PrimType { .. })),
        "arity 1 for flt.add must refuse"
    );
    assert!(
        matches!(neg("flt.neg", &[&x, &x]), Err(EvalError::PrimType { .. })),
        "arity 2 for flt.neg must refuse"
    );
}

// ---- ADR-040 §2.4 (M-899, `enb` Gap A): scalar-float comparison + the named total order -------
//
// Two kinds of evidence, matching the section note in `prims.rs`:
//   - the **reference corpus** (`flt_cmp_reference_cases`) — hand-derived IEEE-754 §5.11
//     predicate rows and §5.10 totalOrder rows, the `EmpiricalFit` basis every comparison result
//     records (row count pinned to `FLT_CMP_CONFORMANCE_TRIALS`, VR-5);
//   - the **property sweeps** over `flt_value_corpus()` — trichotomy on non-NaN, NaN-unordered
//     on every predicate, lt/gt + le/ge duality, and the total-order laws (totality,
//     antisymmetry, transitivity, reflexivity, the −0/+0/NaN placement). The total-order laws
//     are exactly the **M-511 proof debt**: this corpus evidence is what keeps the tag honest at
//     `Empirical` — it is NOT a proof, and the tag must not move to `Proven` until M-511 lands.

/// The five IEEE-754 §5.11 partial-order predicates (NaN unordered → false on every one).
const FLT_CMP_PREDICATES: [&str; 5] = ["flt.lt", "flt.le", "flt.gt", "flt.ge", "flt.eq"];

/// Extract the `Binary{1}` truth bit of a comparison result (never a silent shape pass).
#[track_caller]
fn flt_cmp_truth(label: &str, v: &Value) -> bool {
    assert_eq!(
        v.repr(),
        &Repr::Binary { width: 1 },
        "{label}: result repr must be Binary{{1}}"
    );
    let Payload::Bits(bits) = v.payload() else {
        panic!(
            "{label}: result payload must be Bits, got {:?}",
            v.payload()
        )
    };
    assert_eq!(bits.len(), 1, "{label}: exactly one truth bit");
    bits[0]
}

/// Invoke a float-comparison prim over two `Exact` float operands and return its truth bit.
#[track_caller]
fn flt_cmp(reg: &PrimRegistry, op: &str, a: f64, b: f64) -> bool {
    let f = reg.get(op).expect("flt comparison prim registered");
    let (va, vb) = (fv(a), fv(b));
    let y = f(op, &[&va, &vb])
        .unwrap_or_else(|e| panic!("{op}({a:?}, {b:?}) must be total over Float operands: {e:?}"));
    flt_cmp_truth(op, &y)
}

/// One comparison reference row: op, operands, and the hand-derived expected truth value.
struct FltCmpCase {
    op: &'static str,
    a: f64,
    b: f64,
    expected: bool,
    why: &'static str,
}

/// The M-899 comparison reference corpus (see the section note). Every expected truth value is
/// hand-derived from IEEE-754 §5.11 (the five predicates) / §5.10 (`totalOrder`) — never
/// recomputed with the op under test. Exactly [`FLT_CMP_CONFORMANCE_TRIALS`] rows — asserted by
/// `flt_cmp_reference_case_corpus`.
fn flt_cmp_reference_cases() -> Vec<FltCmpCase> {
    let c = |op, a, b, expected, why| FltCmpCase {
        op,
        a,
        b,
        expected,
        why,
    };
    vec![
        // flt.lt — §5.11 compareQuietLess.
        c("flt.lt", 1.0, 2.0, true, "finite ordered pair"),
        c("flt.lt", 2.0, 1.0, false, "reversed pair"),
        c("flt.lt", 1.0, 1.0, false, "irreflexive"),
        c("flt.lt", -0.0, 0.0, false, "−0 == +0 under §5.11: not less"),
        c("flt.lt", cnan(), 1.0, false, "NaN unordered → false"),
        c(
            "flt.lt",
            1.0,
            cnan(),
            false,
            "NaN unordered (either side) → false",
        ),
        c(
            "flt.lt",
            cnan(),
            cnan(),
            false,
            "NaN vs NaN unordered → false",
        ),
        c(
            "flt.lt",
            f64::NEG_INFINITY,
            f64::INFINITY,
            true,
            "−inf < +inf",
        ),
        c(
            "flt.lt",
            f64::NEG_INFINITY,
            -f64::MAX,
            true,
            "−inf below the finite floor",
        ),
        c(
            "flt.lt",
            f64::INFINITY,
            f64::INFINITY,
            false,
            "inf not < itself",
        ),
        c("flt.lt", 0.0, 5e-324, true, "+0 < the smallest subnormal"),
        // flt.le — §5.11 compareQuietLessEqual.
        c("flt.le", 1.0, 1.0, true, "reflexive on non-NaN"),
        c("flt.le", -0.0, 0.0, true, "−0 == +0: le holds"),
        c("flt.le", 0.0, -0.0, true, "+0 == −0: le holds both ways"),
        c("flt.le", 2.0, 1.0, false, "reversed pair"),
        c(
            "flt.le",
            cnan(),
            1.0,
            false,
            "NaN unordered → false (le is NOT ¬gt on floats)",
        ),
        c(
            "flt.le",
            cnan(),
            cnan(),
            false,
            "le(NaN, NaN) is false — no reflexivity for NaN",
        ),
        c(
            "flt.le",
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            true,
            "−inf == −inf: le holds",
        ),
        // flt.gt — §5.11 compareQuietGreater.
        c("flt.gt", 2.0, 1.0, true, "finite ordered pair"),
        c("flt.gt", 1.0, 2.0, false, "reversed pair"),
        c(
            "flt.gt",
            cnan(),
            1.0,
            false,
            "NaN unordered → false (NaN is not \"the biggest\" under the partial order)",
        ),
        c(
            "flt.gt",
            1.0,
            cnan(),
            false,
            "NaN unordered (either side) → false",
        ),
        c(
            "flt.gt",
            f64::INFINITY,
            f64::MAX,
            true,
            "+inf above the finite ceiling",
        ),
        // flt.ge — §5.11 compareQuietGreaterEqual.
        c("flt.ge", 1.0, 1.0, true, "reflexive on non-NaN"),
        c("flt.ge", 0.0, -0.0, true, "+0 == −0: ge holds"),
        c("flt.ge", 1.0, 2.0, false, "reversed pair"),
        c("flt.ge", cnan(), cnan(), false, "ge(NaN, NaN) is false"),
        c(
            "flt.ge",
            f64::INFINITY,
            cnan(),
            false,
            "unordered even against +inf",
        ),
        // flt.eq — §5.11 compareQuietEqual.
        c("flt.eq", 1.0, 1.0, true, "equal finites"),
        c(
            "flt.eq",
            -0.0,
            0.0,
            true,
            "signed zeros compare EQUAL under §5.11 (the FLAG-4 seam — total_le separates them)",
        ),
        c("flt.eq", 1.0, 2.0, false, "distinct finites"),
        c(
            "flt.eq",
            cnan(),
            cnan(),
            false,
            "NaN ≠ NaN — THE unordered-equality row",
        ),
        c("flt.eq", cnan(), 1.0, false, "NaN equals nothing"),
        c(
            "flt.eq",
            f64::INFINITY,
            f64::INFINITY,
            true,
            "inf equals itself",
        ),
        c(
            "flt.eq",
            f64::INFINITY,
            f64::NEG_INFINITY,
            false,
            "opposite infinities differ",
        ),
        // flt.total_le — §5.10 totalOrder(a, b): −inf < … < −0 < +0 < … < +inf < NaN
        // (canonical positive quiet NaN sorts last — ADR-040 §2.3).
        c(
            "flt.total_le",
            -0.0,
            0.0,
            true,
            "−0 strictly precedes +0 in the total order",
        ),
        c(
            "flt.total_le",
            0.0,
            -0.0,
            false,
            "+0 does NOT precede −0 — the direction flt.eq cannot see",
        ),
        c(
            "flt.total_le",
            1.0,
            2.0,
            true,
            "agrees with le on ordered finites",
        ),
        c(
            "flt.total_le",
            2.0,
            1.0,
            false,
            "agrees with le on reversed finites",
        ),
        c(
            "flt.total_le",
            cnan(),
            cnan(),
            true,
            "REFLEXIVE on NaN (contrast flt.le(NaN, NaN) = false) — total means total",
        ),
        c(
            "flt.total_le",
            f64::INFINITY,
            cnan(),
            true,
            "+inf precedes NaN: canonical NaN sorts last",
        ),
        c(
            "flt.total_le",
            cnan(),
            f64::INFINITY,
            false,
            "NaN does not precede +inf",
        ),
        c("flt.total_le", 1.0, 1.0, true, "reflexive on finites"),
        c(
            "flt.total_le",
            f64::NEG_INFINITY,
            cnan(),
            true,
            "−inf (the total-order minimum) precedes NaN (the maximum)",
        ),
    ]
}

/// **The comparison conformance corpus (the `EmpiricalFit` evidence):** every row's delivered
/// truth bit equals its hand-derived IEEE-754 reference, and the row count equals the
/// `FLT_CMP_CONFORMANCE_TRIALS` the basis records (VR-5 — evidence never drifts from the claim).
#[test]
fn flt_cmp_reference_case_corpus() {
    let reg = PrimRegistry::with_builtins();
    let cases = flt_cmp_reference_cases();
    assert_eq!(
        cases.len() as u64,
        FLT_CMP_CONFORMANCE_TRIALS,
        "the recorded trials must equal the trials actually run (VR-5)"
    );
    for case in &cases {
        assert_eq!(
            flt_cmp(&reg, case.op, case.a, case.b),
            case.expected,
            "{}({:?}, {:?}): want {} — {}",
            case.op,
            case.a,
            case.b,
            case.expected,
            case.why
        );
    }
}

/// **Property (NaN is unordered — ADR-040 §2.4):** against every corpus value, in either operand
/// position, **all five** partial-order predicates yield `false` when NaN is involved — there is
/// no predicate under which NaN is less, greater, or equal to anything (itself included).
#[test]
fn flt_cmp_nan_is_unordered_on_every_predicate() {
    let reg = PrimRegistry::with_builtins();
    for op in FLT_CMP_PREDICATES {
        for &x in &flt_value_corpus() {
            assert!(
                !flt_cmp(&reg, op, cnan(), x),
                "{op}(NaN, {x:?}) must be false — NaN is unordered"
            );
            assert!(
                !flt_cmp(&reg, op, x, cnan()),
                "{op}({x:?}, NaN) must be false — NaN is unordered"
            );
        }
    }
}

/// **Property (trichotomy on the ordered domain):** for non-NaN `a`, `b`, exactly one of
/// `lt`/`eq`/`gt` holds, and the compound predicates are consistent (`le ⟺ lt ∨ eq`,
/// `ge ⟺ gt ∨ eq`). On the *full* domain trichotomy fails only for NaN — pinned by
/// `flt_cmp_nan_is_unordered_on_every_predicate` (all three false), which is exactly why the
/// order is *partial* (ADR-040 §2.4).
#[test]
fn flt_cmp_trichotomy_and_compound_consistency_on_non_nan() {
    let reg = PrimRegistry::with_builtins();
    for &a in &flt_value_corpus() {
        for &b in &flt_value_corpus() {
            if a.is_nan() || b.is_nan() {
                continue;
            }
            let (lt, eq, gt) = (
                flt_cmp(&reg, "flt.lt", a, b),
                flt_cmp(&reg, "flt.eq", a, b),
                flt_cmp(&reg, "flt.gt", a, b),
            );
            assert_eq!(
                u8::from(lt) + u8::from(eq) + u8::from(gt),
                1,
                "trichotomy: exactly one of lt/eq/gt for ({a:?}, {b:?})"
            );
            assert_eq!(
                flt_cmp(&reg, "flt.le", a, b),
                lt || eq,
                "le ⟺ lt ∨ eq for ({a:?}, {b:?})"
            );
            assert_eq!(
                flt_cmp(&reg, "flt.ge", a, b),
                gt || eq,
                "ge ⟺ gt ∨ eq for ({a:?}, {b:?})"
            );
        }
    }
}

/// **Property (duality):** `lt(a, b) ⟺ gt(b, a)` and `le(a, b) ⟺ ge(b, a)` over the *whole*
/// corpus — NaN included (both sides false on unordered pairs).
#[test]
fn flt_cmp_lt_gt_and_le_ge_are_duals() {
    let reg = PrimRegistry::with_builtins();
    for &a in &flt_value_corpus() {
        for &b in &flt_value_corpus() {
            assert_eq!(
                flt_cmp(&reg, "flt.lt", a, b),
                flt_cmp(&reg, "flt.gt", b, a),
                "lt({a:?}, {b:?}) must equal gt({b:?}, {a:?})"
            );
            assert_eq!(
                flt_cmp(&reg, "flt.le", a, b),
                flt_cmp(&reg, "flt.ge", b, a),
                "le({a:?}, {b:?}) must equal ge({b:?}, {a:?})"
            );
        }
    }
}

/// **Property (equality is reflexive exactly off NaN):** `flt.eq(x, x)` is true iff `x` is not
/// NaN — `¬flt.eq(x, x)` is the in-band NaN test the predicate set provides (G2: the no-order
/// case is observable, not swallowed).
#[test]
fn flt_eq_is_reflexive_iff_not_nan() {
    let reg = PrimRegistry::with_builtins();
    for &x in &flt_value_corpus() {
        assert_eq!(
            flt_cmp(&reg, "flt.eq", x, x),
            !x.is_nan(),
            "eq({x:?}, {x:?}) must be true exactly when x is not NaN"
        );
    }
}

/// **Property (the total-order laws — the M-511 proof debt's `Empirical` evidence, VR-5):**
/// over the whole corpus (NaN, ±0, ±inf, subnormals included), `flt.total_le` is **total**
/// (every pair ordered at least one way), **reflexive**, **antisymmetric** (mutual order ⟺
/// bit-identical — with the canonical NaN of ADR-040 §2.3 there is one bit pattern per
/// total-order equivalence class), and **transitive** (swept over all corpus triples). This
/// corpus sweep is evidence, NOT a proof: the tag stays `Empirical` until M-511 discharges the
/// proof debt — never upgraded on the strength of the host's documentation (VR-5).
#[test]
fn flt_total_le_satisfies_the_total_order_laws_on_the_corpus() {
    let reg = PrimRegistry::with_builtins();
    let corpus = flt_value_corpus();
    for &a in &corpus {
        assert!(
            flt_cmp(&reg, "flt.total_le", a, a),
            "reflexive: total_le({a:?}, {a:?})"
        );
        for &b in &corpus {
            let ab = flt_cmp(&reg, "flt.total_le", a, b);
            let ba = flt_cmp(&reg, "flt.total_le", b, a);
            assert!(
                ab || ba,
                "total: total_le must order ({a:?}, {b:?}) at least one way"
            );
            if ab && ba {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "antisymmetric: mutual total_le ⟹ bit-identical ({a:?}, {b:?})"
                );
            }
            for &c in &corpus {
                if ab && flt_cmp(&reg, "flt.total_le", b, c) {
                    assert!(
                        flt_cmp(&reg, "flt.total_le", a, c),
                        "transitive: total_le({a:?}, {b:?}) ∧ total_le({b:?}, {c:?}) ⟹ \
                         total_le({a:?}, {c:?})"
                    );
                }
            }
        }
    }
}

/// **Property (deterministic placement of the seam values — ADR-040 §2.3/§2.4):** the signed
/// zeros are IEEE-**equal** under `flt.eq` but **distinct and directed** under `flt.total_le`
/// (−0 precedes +0, not conversely — the FLAG-4 identity-vs-equality seam made orderable), and
/// the canonical NaN sorts **last**: every corpus value totally-precedes NaN, and NaN precedes
/// nothing but itself.
#[test]
fn flt_total_le_places_signed_zeros_and_nan_deterministically() {
    let reg = PrimRegistry::with_builtins();
    // The zeros: equal to flt.eq, directed to flt.total_le.
    assert!(
        flt_cmp(&reg, "flt.eq", -0.0, 0.0),
        "eq(−0, +0) — IEEE-equal"
    );
    assert!(
        flt_cmp(&reg, "flt.total_le", -0.0, 0.0),
        "total_le(−0, +0) — −0 precedes +0"
    );
    assert!(
        !flt_cmp(&reg, "flt.total_le", 0.0, -0.0),
        "¬total_le(+0, −0) — the zeros are DISTINCT under the total order"
    );
    // NaN: the total-order maximum (canonical positive quiet NaN — ADR-040 §2.3).
    for &x in &flt_value_corpus() {
        assert!(
            flt_cmp(&reg, "flt.total_le", x, cnan()),
            "total_le({x:?}, NaN): everything precedes-or-equals the canonical NaN"
        );
        assert_eq!(
            flt_cmp(&reg, "flt.total_le", cnan(), x),
            x.is_nan(),
            "total_le(NaN, {x:?}): NaN precedes nothing but itself"
        );
    }
}

/// **Property (the total order refines the partial order):** wherever the §5.11 partial order
/// *has* an answer, `flt.total_le` agrees with `flt.le` — the only divergences are exactly the
/// documented seams: NaN (unordered partially, placed totally) and the IEEE-equal-but-
/// bit-distinct signed-zero pair (equal partially, directed totally).
#[test]
fn flt_total_le_refines_le_off_the_documented_seams() {
    let reg = PrimRegistry::with_builtins();
    for &a in &flt_value_corpus() {
        for &b in &flt_value_corpus() {
            if a.is_nan() || b.is_nan() {
                continue; // the NaN seam — pinned by the placement test above.
            }
            #[allow(clippy::float_cmp)] // probing the IEEE-equal (not approximate) seam.
            if a == b && a.to_bits() != b.to_bits() {
                continue; // the ±0 seam — pinned by the placement test above.
            }
            assert_eq!(
                flt_cmp(&reg, "flt.total_le", a, b),
                flt_cmp(&reg, "flt.le", a, b),
                "total_le must agree with le off the seams ({a:?}, {b:?})"
            );
        }
    }
}

/// **The ADR-040 §2.6 tag contract, inspectable off the value (EXPLAIN — G2/SC-3):** every
/// comparison result over `Exact` inputs is `Empirical` with the zero-deviation-vs-spec bound
/// (`eps = 0`, `Linf`) on the `EmpiricalFit{FLT_CMP_CONFORMANCE_TRIALS, …}` basis, with
/// `Derived` provenance — and the Π table's intrinsic agrees (DN-10 §3.4). The
/// `flt.total_le` method string names the M-511 caveat explicitly, so EXPLAIN shows the
/// unproven total-order status, never hides it.
#[test]
fn flt_cmp_results_carry_the_adr040_empirical_tag_and_bound() {
    let reg = PrimRegistry::with_builtins();
    let table = PrimTable::builtins();
    let one = fv(1.0);
    let two = fv(2.0);
    for op in [
        "flt.lt",
        "flt.le",
        "flt.gt",
        "flt.ge",
        "flt.eq",
        "flt.total_le",
    ] {
        let f = reg.get(op).expect("registered");
        let y = f(op, &[&one, &two]).expect("total");
        assert_eq!(
            y.meta().guarantee(),
            GuaranteeStrength::Empirical,
            "{op}: the per-op tag is the ratified ADR-040 §2.6 Empirical (VR-5)"
        );
        assert_eq!(
            table.intrinsic(op),
            Some(GuaranteeStrength::Empirical),
            "{op}: Π intrinsic must agree with the delivered tag (DN-10 §3.4)"
        );
        match y.meta().bound() {
            Some(Bound {
                kind: BoundKind::Error { eps, norm },
                basis: BoundBasis::EmpiricalFit { trials, method },
            }) => {
                assert_eq!(*eps, 0.0, "{op}: zero deviation vs the IEEE predicate spec");
                assert_eq!(*norm, NormKind::Linf);
                assert_eq!(
                    *trials, FLT_CMP_CONFORMANCE_TRIALS,
                    "{op}: the basis records the corpus actually run"
                );
                assert!(
                    method.contains("M-511"),
                    "{op}: the recorded method must surface the M-511 total-order proof debt"
                );
            }
            other => panic!("{op}: expected the EmpiricalFit zero-deviation bound, got {other:?}"),
        }
        assert!(
            matches!(y.meta().provenance(), Provenance::Derived { .. }),
            "{op}: provenance must be Derived"
        );
    }
}

/// **Composition:** a `flt.*` arithmetic result (Empirical, zero-deviation) is a legal
/// comparison operand — comparing computed floats works — while a *genuinely* approximate input
/// (`eps > 0`) is an explicit [`EvalError::ApproxCompositionUnsupported`] refusal: an ε-ball
/// straddling the compare point could flip the bit, and no ε-rule is defined, so it refuses
/// rather than fabricating a truth value (G2/VR-5).
#[test]
fn flt_cmp_composes_over_flt_results_and_true_approximations_refuse() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("flt.add").expect("registered");
    let lt = reg.get("flt.lt").expect("registered");
    // (1.5 + 2.25) < 4.0 — the Empirical intermediate composes; 3.75 < 4.0 is true.
    let sum = add("flt.add", &[&fv(1.5), &fv(2.25)]).expect("total");
    assert_eq!(sum.meta().guarantee(), GuaranteeStrength::Empirical);
    let four = fv(4.0);
    let y = lt("flt.lt", &[&sum, &four]).expect("an flt.* result must compose into a comparison");
    assert!(flt_cmp_truth("flt.lt over a computed operand", &y));
    // A genuinely-approximate Float operand (eps > 0): refuse, never a fabricated bit.
    let approx = Value::new(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(1.0),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Empirical,
            Some(Bound {
                kind: BoundKind::Error {
                    eps: 1e-3,
                    norm: NormKind::Rel,
                },
                basis: BoundBasis::EmpiricalFit {
                    trials: 10,
                    method: "a synthetic approximate source".to_owned(),
                },
            }),
            None,
            None,
            None,
        )
        .expect("well-formed meta"),
    )
    .expect("well-formed value");
    for op in ["flt.lt", "flt.eq", "flt.total_le"] {
        let f = reg.get(op).expect("registered");
        assert!(
            matches!(
                f(op, &[&approx, &fv(1.0)]),
                Err(EvalError::ApproxCompositionUnsupported { .. })
            ),
            "{op}: a true approximation must refuse explicitly, never a fabricated truth bit"
        );
    }
}

/// **Never-silent type/arity discipline:** a non-`Float` operand and a wrong arity are explicit
/// [`EvalError::PrimType`] refusals on every comparison op — never a coercion (G2). And the D1
/// `cmp.eq`/`cmp.lt` prims still refuse `Float` operands *by routing*, naming the `flt.*`
/// predicates and the named total order (never a silently-wrong bitwise order).
#[test]
fn flt_cmp_type_and_arity_refusals_are_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let b = byte([false; 8]);
    let x = fv(1.0);
    for op in [
        "flt.lt",
        "flt.le",
        "flt.gt",
        "flt.ge",
        "flt.eq",
        "flt.total_le",
    ] {
        let f = reg.get(op).expect("registered");
        assert!(
            matches!(f(op, &[&b, &x]), Err(EvalError::PrimType { .. })),
            "{op}: a Binary operand must refuse"
        );
        assert!(
            matches!(f(op, &[&x]), Err(EvalError::PrimType { .. })),
            "{op}: arity 1 must refuse"
        );
    }
    // The D1 comparison prims route floats to the flt.* surface, explicitly.
    for op in ["cmp.eq", "cmp.lt"] {
        let f = reg.get(op).expect("registered");
        match f(op, &[&x, &x]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("flt.total_le"),
                "{op}: the Float refusal must name the flt.* routing, got: {why}"
            ),
            other => panic!("{op} over Float must refuse with PrimType, got {other:?}"),
        }
    }
}

// ── M-892 (`enb` Gap C): the model-dispatched VSA bind group ────────────────────────────────────
//
// `vsa.bind`/`vsa.unbind`/`vsa.permute` — model-dispatched (MAP-I/FHRR/BSC) on the first
// operand's `Repr::Vsa` model id; the `mycelium-vsa` kernel constructs the result `Value` with
// its honest per-model tag and the wrapper carries it through unchanged (VR-5). These tests pin:
// (1) accept-path payloads + carried per-model tags/bounds/provenance, (2) the cheap
// bind→unbind roundtrip property per model (exact recovery for the self-inverse MAP-I/BSC;
// FHRR recovery within its disclosed-Empirical regime), (3) the never-silent reject surface
// (model mismatch, out-of-set model, dim mismatch, alphabet violations, the FHRR regime gate,
// non-Exact operands, arity), and (4) permute's cyclic inverse via the complementary shift.
// (The Π-table meet-tag consistency guard lives in `tests/prim_table.rs`.)

use mycelium_core::SparsityClass;
use mycelium_vsa::fhrr::FHRR_UNBIND_PROFILE;
use mycelium_vsa::{Fhrr, VsaModel};

/// A hypervector `Value` of `model` at `dim` (dense sparsity class, `Exact`/`Root` meta) — built
/// through core alone, exactly what a surface program's injected argument looks like.
fn vsa_hv(model: &str, dim: u32, data: Vec<f64>) -> Value {
    Value::new(
        Repr::Vsa {
            model: model.to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(data),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// Deterministic LCG stream (house style — the seed fully determines the atom).
fn lcg_stream(dim: u32, seed: u64) -> impl Iterator<Item = f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim).map(move |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
    })
}

/// A bipolar (`±1`) MAP-I atom.
fn mapi_atom(dim: u32, seed: u64) -> Vec<f64> {
    lcg_stream(dim, seed)
        .map(|u| if u < 0.5 { -1.0 } else { 1.0 })
        .collect()
}

/// A phasor (`(−π, π]`) FHRR atom.
fn fhrr_atom(dim: u32, seed: u64) -> Vec<f64> {
    lcg_stream(dim, seed)
        .map(|u| {
            let t = std::f64::consts::TAU * u; // [0, τ)
            if t > std::f64::consts::PI {
                t - std::f64::consts::TAU
            } else {
                t
            }
        })
        .collect()
}

/// A binary (`{0, 1}`) BSC atom.
fn bsc_atom(dim: u32, seed: u64) -> Vec<f64> {
    lcg_stream(dim, seed)
        .map(|u| if u < 0.5 { 0.0 } else { 1.0 })
        .collect()
}

/// An unsigned `Binary{w}` shift-amount value (MSB-first), the `vsa.permute` second operand.
fn shift_bin(v: u64, w: u32) -> Value {
    let bits: Vec<bool> = (0..w).rev().map(|i| (v >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: w },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// The per-model accept corpus: (model id, dim, atom builder, the model-namespaced kernel op ids).
/// FHRR rides dim 256 (its unbind profile's `min_dim`); the self-inverse models use a small dim.
type AtomFn = fn(u32, u64) -> Vec<f64>;
fn vsa_corpus() -> [(&'static str, u32, AtomFn, [&'static str; 3]); 3] {
    [
        (
            "MAP-I",
            16,
            mapi_atom as AtomFn,
            ["vsa.map_i.bind", "vsa.map_i.unbind", "vsa.map_i.permute"],
        ),
        (
            "FHRR",
            256,
            fhrr_atom as AtomFn,
            ["vsa.fhrr.bind", "vsa.fhrr.unbind", "vsa.fhrr.permute"],
        ),
        (
            "BSC",
            16,
            bsc_atom as AtomFn,
            ["vsa.bsc.bind", "vsa.bsc.unbind", "vsa.bsc.permute"],
        ),
    ]
}

/// Accept path per model: `vsa.bind` produces a same-model/dim hypervector, tag **`Exact`**
/// carried from the kernel (no bound), provenance the **model-namespaced** kernel op over both
/// inputs — dispatch is recorded, inspectable, never silent (G2).
#[test]
fn vsa_bind_carries_the_kernel_tag_and_model_provenance_per_model() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bind").expect("vsa.bind registered");
    for (model, dim, atom, [bind_op, _, _]) in vsa_corpus() {
        let a = vsa_hv(model, dim, atom(dim, 1));
        let b = vsa_hv(model, dim, atom(dim, 2));
        let y = f("vsa.bind", &[&a, &b]).unwrap_or_else(|e| panic!("{model}: bind failed: {e}"));
        assert_eq!(
            y.repr(),
            &Repr::Vsa {
                model: model.to_owned(),
                dim,
                sparsity: SparsityClass::Dense,
            },
            "{model}: bind must preserve model + dim"
        );
        assert_eq!(
            y.meta().guarantee(),
            mycelium_core::GuaranteeStrength::Exact,
            "{model}: bind is Exact in every model of the dispatch set"
        );
        assert!(
            y.meta().bound().is_none(),
            "{model}: Exact carries no bound"
        );
        match y.meta().provenance() {
            Provenance::Derived { op, inputs } => {
                assert_eq!(
                    op,
                    &mycelium_core::operation_hash(bind_op),
                    "{model}: provenance records the dispatched model-namespaced op"
                );
                assert_eq!(inputs, &vec![a.content_hash(), b.content_hash()]);
            }
            other => panic!("{model}: expected Derived provenance, got {other:?}"),
        }
    }
}

/// The DoD property, per model, over a seeded corpus: `unbind(bind(a, b), b)` recovers `a` —
/// **exactly** (payload-equal) for the self-inverse `Exact` models (MAP-I, BSC), and for FHRR
/// within its disclosed basis: the result is **`Empirical`** carrying the trial-validated
/// `FHRR_UNBIND_PROFILE` δ bound, and pure-pair recovery is near-exact by the model's own
/// similarity (the kernel's documented behaviour; the δ = 1e-2 claim is about cleanup-completed
/// recovery, disclosed on the bound — VR-5: asserted at exactly that strength, no more).
#[test]
fn vsa_unbind_bind_roundtrip_recovers_the_operand_per_model() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let unbind = reg.get("vsa.unbind").expect("registered");
    for (model, dim, atom, _) in vsa_corpus() {
        for seed in [3u64, 5, 8, 13, 21, 34, 55, 89] {
            let a = vsa_hv(model, dim, atom(dim, seed));
            let b = vsa_hv(model, dim, atom(dim, seed + 1000));
            let ab = bind("vsa.bind", &[&a, &b]).unwrap();
            let rec = unbind("vsa.unbind", &[&ab, &b])
                .unwrap_or_else(|e| panic!("{model}/{seed}: unbind failed: {e}"));
            match model {
                "MAP-I" | "BSC" => {
                    assert_eq!(
                        rec.payload(),
                        a.payload(),
                        "{model}/{seed}: the self-inverse identity recovers a exactly"
                    );
                    assert_eq!(
                        rec.meta().guarantee(),
                        mycelium_core::GuaranteeStrength::Exact,
                        "{model}/{seed}: Exact carried from the kernel"
                    );
                }
                "FHRR" => {
                    assert_eq!(
                        rec.meta().guarantee(),
                        mycelium_core::GuaranteeStrength::Empirical,
                        "FHRR/{seed}: the weak-link Empirical tag carried from the kernel"
                    );
                    assert_eq!(
                        rec.meta().bound(),
                        Some(&FHRR_UNBIND_PROFILE.bound()),
                        "FHRR/{seed}: the trial-validated δ bound rides the value"
                    );
                    let (Payload::Hypervector(r), Payload::Hypervector(orig)) =
                        (rec.payload(), a.payload())
                    else {
                        panic!("FHRR/{seed}: hypervector payloads expected");
                    };
                    let sim = Fhrr::new(dim).similarity(r, orig);
                    assert!(
                        sim > 0.999,
                        "FHRR/{seed}: pure-pair recovery must be near-exact, got {sim}"
                    );
                }
                _ => unreachable!("corpus models"),
            }
        }
    }
}

/// `vsa.permute` per model: `Exact` carried from the kernel, and cyclic — the complementary
/// shift `dim − s` inverts it exactly (so the inverse permutation is expressible with the
/// unsigned `Binary{W}` shift operand; no negative-shift form is needed).
#[test]
fn vsa_permute_is_exact_and_inverted_by_the_complementary_shift() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.permute").expect("registered");
    for (model, dim, atom, [_, _, permute_op]) in vsa_corpus() {
        let a = vsa_hv(model, dim, atom(dim, 7));
        let s = shift_bin(3, 8);
        let p = f("vsa.permute", &[&a, &s]).unwrap_or_else(|e| panic!("{model}: {e}"));
        assert_eq!(
            p.meta().guarantee(),
            mycelium_core::GuaranteeStrength::Exact,
            "{model}: permute is Exact in every model of the dispatch set"
        );
        assert_ne!(
            p.payload(),
            a.payload(),
            "{model}: a nonzero shift moves components"
        );
        match p.meta().provenance() {
            Provenance::Derived { op, .. } => assert_eq!(
                op,
                &mycelium_core::operation_hash(permute_op),
                "{model}: provenance records the dispatched model-namespaced op"
            ),
            other => panic!("{model}: expected Derived provenance, got {other:?}"),
        }
        // The complementary shift restores the original components exactly (a pure rotation).
        let back = f("vsa.permute", &[&p, &shift_bin(u64::from(dim) - 3, 16)]).unwrap();
        assert_eq!(back.payload(), a.payload(), "{model}: cyclic inverse");
    }
}

/// The never-silent reject surface — model dispatch: a cross-model operand pair, a model outside
/// the introduction dispatch set (HRR — a kernel model with no surfaced Value-level set), a
/// non-Vsa operand, and a dim mismatch are all explicit `PrimType` refusals naming the offense
/// (G2), never a coercion or a guessed algebra.
#[test]
fn vsa_model_dispatch_rejects_are_explicit() {
    let reg = PrimRegistry::with_builtins();
    for prim in ["vsa.bind", "vsa.unbind"] {
        let f = reg.get(prim).expect("registered");
        // Cross-model operands: dispatch anchors on the FIRST operand's model; the kernel then
        // refuses the foreign second operand (never a silent cross-model bind).
        let mapi = vsa_hv("MAP-I", 16, mapi_atom(16, 1));
        let bsc = vsa_hv("BSC", 16, bsc_atom(16, 2));
        match f(prim, &[&mapi, &bsc]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("MAP-I"),
                "{prim}: the model-mismatch refusal names the expected model, got: {why}"
            ),
            other => panic!("{prim}: cross-model operands must refuse, got {other:?}"),
        }
        // An out-of-set model refuses naming the dispatch set (append-only widening, no guess).
        let hrr = vsa_hv("HRR", 16, mapi_atom(16, 3));
        match f(prim, &[&hrr, &hrr]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("MAP-I, FHRR, BSC"),
                "{prim}: the out-of-set refusal names the dispatch set, got: {why}"
            ),
            other => panic!("{prim}: an out-of-set model must refuse, got {other:?}"),
        }
        // A non-Vsa operand refuses explicitly.
        assert!(
            matches!(
                f(prim, &[&byte([true; 8]), &mapi]),
                Err(EvalError::PrimType { .. })
            ),
            "{prim}: a non-Vsa first operand must refuse"
        );
        // Dim mismatch: the kernel's DimMismatch/NotThisModel refusal, carried explicitly.
        let mapi32 = vsa_hv("MAP-I", 32, mapi_atom(32, 4));
        assert!(
            matches!(f(prim, &[&mapi, &mapi32]), Err(EvalError::PrimType { .. })),
            "{prim}: a dim mismatch must refuse"
        );
        // Arity is explicit.
        assert!(
            matches!(f(prim, &[&mapi]), Err(EvalError::PrimType { .. })),
            "{prim}: arity 1 must refuse"
        );
    }
}

/// Alphabet violations refuse explicitly per model (the kernel's guard, carried): a non-`±1`
/// MAP-I component, a non-`{0,1}` BSC component, an out-of-range FHRR phase — the tag would be
/// wrong off-alphabet, so the kernel refuses rather than mis-stamps (A3-04; VR-5/G2).
#[test]
fn vsa_alphabet_violations_refuse_explicitly() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bind").expect("registered");
    let cases: [(&str, u32, AtomFn, f64); 3] = [
        ("MAP-I", 16, mapi_atom, 0.5),
        ("BSC", 16, bsc_atom, 2.0),
        ("FHRR", 256, fhrr_atom, 7.0),
    ];
    for (model, dim, atom, bad_component) in cases {
        let mut data = atom(dim, 1);
        data[3] = bad_component;
        let bad = vsa_hv(model, dim, data);
        let ok = vsa_hv(model, dim, atom(dim, 2));
        match f("vsa.bind", &[&bad, &ok]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("alphabet") || why.contains("component"),
                "{model}: the refusal names the alphabet violation, got: {why}"
            ),
            other => panic!("{model}: off-alphabet operand must refuse, got {other:?}"),
        }
    }
}

/// The FHRR unbind regime gate, through the prim path: unbinding a value that is not a single
/// `vsa.fhrr.bind` product is an explicit refusal (the kernel's `OutsideEmpiricalProfile` — the
/// Empirical tag is issued only inside its trial-validated regime; VR-5), never a silently
/// mis-tagged decode.
#[test]
fn vsa_fhrr_unbind_is_regime_gated_through_the_prim_path() {
    let reg = PrimRegistry::with_builtins();
    let unbind = reg.get("vsa.unbind").expect("registered");
    let a = vsa_hv("FHRR", 256, fhrr_atom(256, 1));
    let b = vsa_hv("FHRR", 256, fhrr_atom(256, 2));
    // Root provenance → outside the validated single-factor regime → explicit refusal.
    match unbind("vsa.unbind", &[&a, &b]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("empirical profile"),
            "the refusal names the regime gate, got: {why}"
        ),
        other => panic!("an out-of-regime FHRR unbind must refuse, got {other:?}"),
    }
}

/// The Exact-input guard (the M-204 posture, VSA form): a non-`Exact` operand — here an
/// `Empirical` FHRR unbind result, a perfectly in-alphabet phase vector — must NOT come out of
/// `vsa.bind` re-stamped `Exact`. There is no defined δ-propagation rule through the algebra
/// yet, so the wrapper refuses explicitly (never a fabricated bound, never a silent upgrade —
/// G2/VR-5; the honest noisy-decode path is cleanup, M-894).
#[test]
fn vsa_non_exact_operands_refuse_composition() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let unbind = reg.get("vsa.unbind").expect("registered");
    let a = vsa_hv("FHRR", 256, fhrr_atom(256, 1));
    let b = vsa_hv("FHRR", 256, fhrr_atom(256, 2));
    let ab = bind("vsa.bind", &[&a, &b]).unwrap();
    let noisy = unbind("vsa.unbind", &[&ab, &b]).unwrap();
    assert_eq!(
        noisy.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Empirical
    );
    for prim in ["vsa.bind", "vsa.unbind", "vsa.permute"] {
        let f = reg.get(prim).expect("registered");
        let second: &Value = if prim == "vsa.permute" {
            &shift_bin(1, 8)
        } else {
            &a
        };
        assert!(
            matches!(
                f(prim, &[&noisy, second]),
                Err(EvalError::ApproxCompositionUnsupported { .. })
            ),
            "{prim}: a non-Exact operand must refuse composition, not re-stamp Exact"
        );
    }
}

// ── M-893 (`enb` Gap C): `vsa.bundle` — superposition via the certified path ────────────────────
//
// `vsa.bundle : (Seq{Vsa{m, d}, N≥1}, Float δ) → Vsa{m, d}` — MAP-I's `bundle_values_certified`
// (the M-131 checked-instantiation pattern). These tests pin: (1) the accept path — the kernel's
// `Proven` tag and its checked `CapacityBound` carried unchanged, the disclosed bound being the
// value's OWN (items = this bundle's m, dim = its d), model-namespaced provenance over every
// input hash; (2) the certified-singleton dispatch (FHRR/BSC refuse naming the certified set —
// their kernel bundles are Empirical-profile ops; an out-of-set model refuses naming the M-892
// set); (3) the never-silent reject surface (insufficient capacity naming the required dim,
// duplicates, off-alphabet, empty seq, non-Seq/non-Float operands, out-of-range δ, arity);
// (4) the elementwise Exact-input guard (a Proven bundle fed back in refuses — no nested-bundle
// δ-composition rule exists); and (5) the cheap capacity-bound property: below the checked bound
// every member stays recoverable by similarity against strangers.

use mycelium_vsa::capacity;
use mycelium_vsa::MapI;

/// A `Seq` value over hypervector elements (`Exact`/`Root` meta — exactly what the L1 list
/// literal builds). `elem_of` anchors the descriptor so an empty seq is constructible too.
fn vsa_seq_of(elem_of: &Value, items: &[Value]) -> Value {
    Value::new(
        Repr::Seq {
            elem: Box::new(elem_of.repr().clone()),
            len: u32::try_from(items.len()).expect("test seqs are small"),
        },
        Payload::Seq(items.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// Accept path: the kernel's **`Proven`** tag + checked `CapacityBound` are carried unchanged
/// (VR-5), the payload is the elementwise sum, and the disclosed bound is the **value's own**
/// (`Capacity{items, dim}` = this bundle's m and d, `ProvenThm` basis carrying the citation +
/// the checked side-condition record). Provenance is the model-namespaced `vsa.map_i.bundle`
/// over every input hash, in order — inspectable, EXPLAIN-able dispatch (G2).
#[test]
fn vsa_bundle_carries_the_kernel_proven_tag_and_its_own_capacity_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bundle").expect("vsa.bundle registered");
    let dim = 2048u32; // ≥ requiredDim(3, 1e-2) = 1141 — the checked side-condition holds.
    assert!(u64::from(dim) >= capacity::required_dim(3, 1e-2, capacity::MARGIN_MU));
    let items: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, 100 + i)))
        .collect();
    let seq = vsa_seq_of(&items[0], &items);
    let y = f("vsa.bundle", &[&seq, &fv(1e-2)]).expect("certified bundle accepts");
    assert_eq!(
        y.repr(),
        &Repr::Vsa {
            model: "MAP-I".to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        }
    );
    // Payload: the elementwise integer superposition (sum of the three bipolar atoms).
    let expected: Vec<f64> = (0..dim as usize)
        .map(|k| {
            items
                .iter()
                .map(|v| match v.payload() {
                    Payload::Hypervector(h) => h[k],
                    _ => unreachable!(),
                })
                .sum()
        })
        .collect();
    assert_eq!(y.payload(), &Payload::Hypervector(expected));
    // The kernel's Proven tag, carried unchanged — with its checked bound (never Proven bare).
    assert_eq!(
        y.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    match y.meta().bound() {
        Some(mycelium_core::Bound {
            kind: mycelium_core::BoundKind::Capacity { items: m, dim: d },
            basis: mycelium_core::BoundBasis::ProvenThm { citation },
        }) => {
            // The disclosed bound is the value's OWN: its m and d, not a generic table row.
            assert_eq!(*m, 3, "the bound discloses this bundle's item count");
            assert_eq!(*d, u64::from(dim), "the bound discloses this bundle's dim");
            assert!(
                citation.contains("Clarkson") && citation.contains("requiredDim"),
                "the ProvenThm basis records the citation + checked side-condition: {citation}"
            );
        }
        other => panic!("expected the kernel's checked Capacity/ProvenThm bound, got {other:?}"),
    }
    // Model-namespaced provenance over every input hash, in order (G2: dispatch is recorded).
    match y.meta().provenance() {
        Provenance::Derived { op, inputs } => {
            assert_eq!(op, &mycelium_core::operation_hash("vsa.map_i.bundle"));
            assert_eq!(
                inputs,
                &items.iter().map(Value::content_hash).collect::<Vec<_>>()
            );
        }
        other => panic!("expected Derived provenance, got {other:?}"),
    }
}

/// Insufficient dimension: the theorem's side-condition fails, so NO `Proven` bound can be issued
/// — an explicit refusal **naming the required dim** (the kernel's `InsufficientCapacity`),
/// never an unbacked tag and never a silently-weaker result (M-I2/VR-5; G2).
#[test]
fn vsa_bundle_refuses_insufficient_capacity_never_an_unbacked_proven() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bundle").expect("registered");
    let items: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", 16, mapi_atom(16, 200 + i)))
        .collect();
    let seq = vsa_seq_of(&items[0], &items);
    match f("vsa.bundle", &[&seq, &fv(1e-2)]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("insufficient capacity") && why.contains("1141"),
            "the refusal names the failed side-condition + the required dim, got: {why}"
        ),
        other => panic!("an under-dimensioned certified bundle must refuse, got {other:?}"),
    }
}

/// The certified-singleton dispatch: FHRR/BSC are in the M-892 bind-group set but have **no
/// certified Value-level bundle** — routing them through the certified prim would silently
/// re-tag their Empirical-profile evidence (VR-5), so each refuses explicitly naming the
/// certified set; an out-of-set model (HRR) refuses naming the M-892 dispatch set as usual.
#[test]
fn vsa_bundle_dispatch_is_the_certified_singleton() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bundle").expect("registered");
    for (model, dim, atom) in [
        ("FHRR", 256u32, fhrr_atom as AtomFn),
        ("BSC", 16, bsc_atom as AtomFn),
    ] {
        let items: Vec<Value> = (0..3)
            .map(|i| vsa_hv(model, dim, atom(dim, 10 + i)))
            .collect();
        let seq = vsa_seq_of(&items[0], &items);
        match f("vsa.bundle", &[&seq, &fv(1e-2)]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains(model) && why.contains("certified singleton"),
                "{model}: the refusal names the model + the certified set, got: {why}"
            ),
            other => panic!("{model}: an uncertified-model bundle must refuse, got {other:?}"),
        }
    }
    // Out-of-set model: the shared vsa_model_of refusal names the M-892 dispatch set.
    let hrr = vsa_hv("HRR", 16, mapi_atom(16, 3));
    let seq = vsa_seq_of(&hrr, std::slice::from_ref(&hrr));
    match f("vsa.bundle", &[&seq, &fv(1e-2)]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("MAP-I, FHRR, BSC"),
            "the out-of-set refusal names the dispatch set, got: {why}"
        ),
        other => panic!("an out-of-set model must refuse, got {other:?}"),
    }
}

/// The never-silent reject surface: empty seq, duplicate items, an off-alphabet component, a
/// non-Seq first operand, a non-Float δ, an out-of-range/non-finite δ, and arity — every one an
/// explicit `PrimType`, never a coercion or a defaulted parameter (G2). A dim/model-mismatched
/// *element* is refused upstream by the core `Seq` well-formedness invariant (pinned here too).
#[test]
fn vsa_bundle_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bundle").expect("registered");
    let dim = 2048u32;
    let a = vsa_hv("MAP-I", dim, mapi_atom(dim, 1));
    let b = vsa_hv("MAP-I", dim, mapi_atom(dim, 2));
    let delta = fv(1e-2);

    // Empty seq: no superposition is defined — refused with its own message.
    let empty = vsa_seq_of(&a, &[]);
    match f("vsa.bundle", &[&empty, &delta]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("at least one item"),
            "the empty-bundle refusal is named, got: {why}"
        ),
        other => panic!("an empty bundle must refuse, got {other:?}"),
    }

    // Duplicate items: the theorem assumes distinct atoms — the kernel's refusal is carried.
    let dup = vsa_seq_of(&a, &[a.clone(), a.clone()]);
    match f("vsa.bundle", &[&dup, &delta]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("distinct"),
            "the duplicate-items refusal is named, got: {why}"
        ),
        other => panic!("duplicate items must refuse, got {other:?}"),
    }

    // Off-alphabet component: the theorem assumes bipolar atoms — refused, never mis-tagged.
    let mut bad_data = mapi_atom(dim, 3);
    bad_data[5] = 0.5;
    let bad = vsa_hv("MAP-I", dim, bad_data);
    let off = vsa_seq_of(&a, &[a.clone(), bad]);
    match f("vsa.bundle", &[&off, &delta]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("alphabet") || why.contains("component"),
            "the off-alphabet refusal is named, got: {why}"
        ),
        other => panic!("an off-alphabet item must refuse, got {other:?}"),
    }

    // A dim/model-mismatched *element* is unreachable through this prim: the core `Seq`
    // well-formedness invariant (every element's repr matches the descriptor) refuses the
    // heterogeneous seq at *construction* (`Value::new` → `PayloadReprMismatch`), upstream of
    // the prim — verified here so the invariant this wrapper leans on cannot silently relax.
    // (The kernel's own per-item `hv_of` dim/model check stays as defense in depth.)
    let small = vsa_hv("MAP-I", 16, mapi_atom(16, 4));
    assert!(
        Value::new(
            Repr::Seq {
                elem: Box::new(a.repr().clone()),
                len: 2,
            },
            Payload::Seq(vec![a.clone(), small]),
            Meta::exact(Provenance::Root),
        )
        .is_err(),
        "a heterogeneous seq (dim-mismatched element) must be refused at construction"
    );

    // A non-Seq first operand / a non-Float δ: explicit type refusals.
    assert!(
        matches!(
            f("vsa.bundle", &[&a, &delta]),
            Err(EvalError::PrimType { .. })
        ),
        "a bare hypervector (non-Seq) first operand must refuse"
    );
    let pair = vsa_seq_of(&a, &[a.clone(), b.clone()]);
    assert!(
        matches!(
            f("vsa.bundle", &[&pair, &byte([false; 8])]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Float δ must refuse"
    );

    // δ outside (0, 1] or non-finite: refused with the named δ-domain message (not the kernel's
    // misleading required-dim-u64::MAX fold).
    for bad_delta in [0.0, -1e-3, 1.5, f64::NAN, f64::INFINITY] {
        match f("vsa.bundle", &[&pair, &fv(bad_delta)]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("(0, 1]"),
                "δ={bad_delta}: the δ-domain refusal is named, got: {why}"
            ),
            other => panic!("δ={bad_delta} must refuse, got {other:?}"),
        }
    }
    // δ = 1.0 is the domain's closed edge — the kernel's own domain, not a wrapper refusal.
    assert!(
        f("vsa.bundle", &[&pair, &fv(1.0)]).is_ok(),
        "δ = 1.0 is in the kernel's (0, 1] domain"
    );

    // Arity is explicit.
    assert!(
        matches!(f("vsa.bundle", &[&pair]), Err(EvalError::PrimType { .. })),
        "arity 1 must refuse"
    );
}

/// The elementwise Exact-input guard: a **`Proven` bundle fed back as an item** (an in-range
/// integer hypervector the alphabet guard alone would not stop… and the kernel's bipolar check
/// would — but the guard must refuse FIRST on the tag, since no δ-composition rule for nested
/// certified bundles exists) refuses as `ApproxCompositionUnsupported`, never a silently
/// re-certified nesting (VR-5/G2). A non-`Exact` seq value or δ refuses via the same guard.
#[test]
fn vsa_bundle_non_exact_elements_refuse_composition() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bundle").expect("registered");
    let dim = 2048u32;
    let items: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, 300 + i)))
        .collect();
    let seq = vsa_seq_of(&items[0], &items);
    let bundle = f("vsa.bundle", &[&seq, &fv(1e-2)]).expect("accepts");
    assert_eq!(
        bundle.meta().guarantee(),
        mycelium_core::GuaranteeStrength::Proven
    );
    // Feed the Proven bundle back in next to a fresh Exact atom.
    let fresh = vsa_hv("MAP-I", dim, mapi_atom(dim, 999));
    let nested = vsa_seq_of(&fresh, &[fresh.clone(), bundle]);
    assert!(
        matches!(
            f("vsa.bundle", &[&nested, &fv(1e-2)]),
            Err(EvalError::ApproxCompositionUnsupported { .. })
        ),
        "a non-Exact element must refuse composition, not re-certify a nested bundle"
    );
}

/// The cheap capacity-bound property (the M-893 DoD row): **below the checked bound, every
/// member stays recoverable** — for m ∈ {2, 3, 5} at dim 2048 ≥ requiredDim(m, δ=1e-2), each
/// bundled member's cosine similarity to the bundle strictly exceeds every stranger atom's, over
/// a deterministic multi-seed corpus. This exercises the bound's *operational content*
/// (member-vs-stranger separation at the certified dimension) — the δ-tail itself is the cited
/// theorem's claim, not re-proven here (Empirical evidence FOR the Proven instantiation, per the
/// M-131 posture). Also pins that the disclosed bound is each value's own (m, dim).
#[test]
fn vsa_bundle_members_recoverable_below_the_proven_capacity_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.bundle").expect("registered");
    let dim = 2048u32;
    let delta = 1e-2;
    let model = MapI::new(dim);
    for m in [2usize, 3, 5] {
        assert!(
            u64::from(dim) >= capacity::required_dim(m as u64, delta, capacity::MARGIN_MU),
            "corpus precondition: the checked side-condition holds at m={m}"
        );
        for base_seed in [0u64, 1, 2] {
            let items: Vec<Value> = (0..m as u64)
                .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, 1000 * (base_seed + 1) + i)))
                .collect();
            let seq = vsa_seq_of(&items[0], &items);
            let y = f("vsa.bundle", &[&seq, &fv(delta)]).expect("certified bundle accepts");
            // The disclosed bound is this value's own m and d.
            match y.meta().bound() {
                Some(mycelium_core::Bound {
                    kind: mycelium_core::BoundKind::Capacity { items: bm, dim: bd },
                    ..
                }) => {
                    assert_eq!(*bm, m as u64);
                    assert_eq!(*bd, u64::from(dim));
                }
                other => panic!("expected the value's own Capacity bound, got {other:?}"),
            }
            let bundle_hv = match y.payload() {
                Payload::Hypervector(h) => h.clone(),
                _ => unreachable!(),
            };
            let strangers: Vec<Vec<f64>> = (0..8u64)
                .map(|i| mapi_atom(dim, 777_000 + 100 * base_seed + i))
                .collect();
            let worst_stranger = strangers
                .iter()
                .map(|s| model.similarity(&bundle_hv, s))
                .fold(f64::NEG_INFINITY, f64::max);
            for (k, item) in items.iter().enumerate() {
                let member_hv = match item.payload() {
                    Payload::Hypervector(h) => h,
                    _ => unreachable!(),
                };
                let member_sim = model.similarity(&bundle_hv, member_hv);
                assert!(
                    member_sim > worst_stranger,
                    "m={m} seed={base_seed} member {k}: recoverability below the certified \
                     bound — member sim {member_sim} must exceed the best stranger \
                     {worst_stranger}"
                );
            }
        }
    }
}

// ── M-894 (`enb` Gap C): `vsa.cleanup` + `vsa.reconstruct` + `vsa.required_dim` ─────────────────
//
// The cleanup-memory retrieval, the RFC-0003 §6 compositional role-reconstruction, and the M-131
// capacity-bound query (FR-S4). These tests pin: (1) the accept paths — the `[index, confidence,
// margin]` decision triple with the query/record's own (strength, bound) pair carried through
// (the disclosed bound is the value's own — VR-5), composed `Derived{op: hash(prim)}` provenance;
// (2) the dispatch sets (cleanup: MAP-I/FHRR/BSC; reconstruct: {MAP-I, BSC}, an FHRR record an
// explicit refusal naming its unbind profile's regime); (3) the never-silent reject surface
// (empty codebook, model/dim mismatches, the RFC-0010 §4.4 identifiability tie, the below-
// threshold refusal naming confidence vs threshold, non-Exact non-carry operands, degenerate
// items/δ, arity); and (4) the **below-capacity property** (the M-894 DoD row): over a
// deterministic (m × seed) corpus at a dim certified by `vsa.required_dim`'s own answer,
// role-reconstruction from a certified bundle recovers every bundled filler, and the triple
// re-discloses the record's own `Proven` `CapacityBound`.

/// Unpack the `Seq{Float, 3}` decision triple.
fn triple_of(v: &Value) -> [f64; 3] {
    match v.payload() {
        Payload::Seq(elems) => {
            let xs: Vec<f64> = elems
                .iter()
                .map(|e| match e.payload() {
                    Payload::Float(x) => *x,
                    other => panic!("triple element must be a Float, got {other:?}"),
                })
                .collect();
            [xs[0], xs[1], xs[2]]
        }
        other => panic!("expected the Seq{{Float, 3}} decision triple, got {other:?}"),
    }
}

/// Decode a `Binary{64}` result (MSB-first) to a u64.
fn u64_of_bits(v: &Value) -> u64 {
    match (v.repr(), v.payload()) {
        (Repr::Binary { width: 64 }, Payload::Bits(bits)) => {
            bits.iter().fold(0u64, |acc, &b| (acc << 1) | u64::from(b))
        }
        other => panic!("expected a Binary{{64}} dimension, got {other:?}"),
    }
}

/// Accept path per model (MAP-I/FHRR/BSC — the procedure is model-generic): cleaning an exact
/// codebook atom recovers its own index with confidence ≈ 1 and a positive margin; the triple is
/// `Seq{Float, 3}`, `Exact`/no bound (every operand was `Exact`), with composed
/// `Derived{op: hash("vsa.cleanup"), inputs: the operands}` provenance (G2: inspectable).
#[test]
fn vsa_cleanup_returns_the_decision_triple_per_model() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.cleanup").expect("vsa.cleanup registered");
    for (model, dim, atom, _) in vsa_corpus() {
        let atoms: Vec<Value> = (0..4)
            .map(|i| vsa_hv(model, dim, atom(dim, 400 + i)))
            .collect();
        let codebook = vsa_seq_of(&atoms[0], &atoms);
        let query = atoms[2].clone();
        let y = f("vsa.cleanup", &[&query, &codebook])
            .unwrap_or_else(|e| panic!("{model}: cleanup failed: {e}"));
        assert_eq!(
            y.repr(),
            &Repr::Seq {
                elem: Box::new(Repr::Float {
                    width: FloatWidth::F64
                }),
                len: 3,
            },
            "{model}: the result is the Seq{{Float, 3}} decision triple"
        );
        let [index, confidence, margin] = triple_of(&y);
        assert_eq!(
            index, 2.0,
            "{model}: the exact atom cleans to its own index"
        );
        assert!(
            (confidence - 1.0).abs() < 1e-9,
            "{model}: an exact-atom query matches with full confidence, got {confidence}"
        );
        assert!(margin > 0.0, "{model}: unique arg-max, got margin {margin}");
        assert_eq!(
            y.meta().guarantee(),
            GuaranteeStrength::Exact,
            "{model}: all-Exact operands ⇒ an Exact decode triple (RFC-0010 §4.4)"
        );
        assert!(
            y.meta().bound().is_none(),
            "{model}: Exact carries no bound"
        );
        match y.meta().provenance() {
            Provenance::Derived { op, inputs } => {
                assert_eq!(op, &mycelium_core::operation_hash("vsa.cleanup"));
                assert_eq!(inputs, &vec![query.content_hash(), codebook.content_hash()]);
            }
            other => panic!("{model}: expected Derived provenance, got {other:?}"),
        }
    }
}

/// The FR-S4 headline path + the carry rule: an **FHRR unbind result** (`Empirical`, carrying its
/// trial-validated probability bound) used as the cleanup query yields the right atom, and the
/// triple carries the query's **own** (strength, bound) pair through the §4.7 meet — `Empirical`
/// with the same `Probability`/`EmpiricalFit` bound, never re-derived, never upgraded (VR-5).
#[test]
fn vsa_cleanup_carries_the_noisy_query_pair_through() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let unbind = reg.get("vsa.unbind").expect("registered");
    let cleanup = reg.get("vsa.cleanup").expect("registered");
    let dim = 256u32;
    let a = vsa_hv("FHRR", dim, fhrr_atom(dim, 21));
    let b = vsa_hv("FHRR", dim, fhrr_atom(dim, 22));
    let product = bind("vsa.bind", &[&a, &b]).expect("bind accepts");
    // A single vsa.fhrr.bind product is inside the validated unbind regime.
    let noisy = unbind("vsa.unbind", &[&product, &b]).expect("in-regime unbind accepts");
    assert_eq!(noisy.meta().guarantee(), GuaranteeStrength::Empirical);
    let query_bound = noisy
        .meta()
        .bound()
        .cloned()
        .expect("Empirical has a bound");

    let atoms: Vec<Value> = [21u64, 31, 41, 51]
        .into_iter()
        .map(|s| vsa_hv("FHRR", dim, fhrr_atom(dim, s)))
        .collect();
    let codebook = vsa_seq_of(&atoms[0], &atoms);
    let y = cleanup("vsa.cleanup", &[&noisy, &codebook]).expect("cleanup accepts");
    let [index, confidence, margin] = triple_of(&y);
    assert_eq!(index, 0.0, "the noisy unbind cleans up to the true atom");
    assert!(
        confidence > 0.9,
        "FHRR exact-inverse recovery, got {confidence}"
    );
    assert!(margin > 0.0);
    // The carried pair is the query's own (the M-204 Passthrough posture).
    assert_eq!(
        y.meta().guarantee(),
        GuaranteeStrength::Empirical,
        "the triple's strength is the meet — the noisy query's own Empirical"
    );
    assert_eq!(
        y.meta().bound(),
        Some(&query_bound),
        "the disclosed bound is the query's own, carried unchanged (VR-5)"
    );
}

/// Exact round trip per self-inverse model ({MAP-I, BSC} — the reconstruct dispatch set): with a
/// plain `Exact` bind product as the record, `vsa.reconstruct(record, role, fillers, thr)`
/// recovers the filler's index at confidence ≈ 1 with an `Exact`/no-bound triple.
#[test]
fn vsa_reconstruct_exact_roundtrip_per_self_inverse_model() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let f = reg
        .get("vsa.reconstruct")
        .expect("vsa.reconstruct registered");
    for (model, dim, atom) in [
        ("MAP-I", 64u32, mapi_atom as AtomFn),
        ("BSC", 64, bsc_atom as AtomFn),
    ] {
        let role = vsa_hv(model, dim, atom(dim, 500));
        let fillers: Vec<Value> = (0..3)
            .map(|i| vsa_hv(model, dim, atom(dim, 600 + i)))
            .collect();
        let codebook = vsa_seq_of(&fillers[0], &fillers);
        let record = bind("vsa.bind", &[&role, &fillers[1]]).expect("bind accepts");
        let y = f("vsa.reconstruct", &[&record, &role, &codebook, &fv(0.5)])
            .unwrap_or_else(|e| panic!("{model}: reconstruct failed: {e}"));
        let [index, confidence, margin] = triple_of(&y);
        assert_eq!(index, 1.0, "{model}: the bound filler is recovered");
        assert!(
            (confidence - 1.0).abs() < 1e-9,
            "{model}: self-inverse unbind recovers exactly, got {confidence}"
        );
        assert!(margin > 0.0, "{model}: unique arg-max");
        assert_eq!(y.meta().guarantee(), GuaranteeStrength::Exact);
        assert!(y.meta().bound().is_none());
    }
}

/// **The below-capacity property (the M-894 DoD row).** Data-driven corpus over m ∈ {2, 3, 5} ×
/// three seeds: bundle m role⊗filler pairs through the certified path at a dimension that
/// `vsa.required_dim`'s **own answer** certifies (dim ≥ requiredDim(m, δ) — the query and the
/// property exercise the same checked instantiation), then reconstruct **every** role — each
/// recovers its own filler index, clearing the threshold with a positive margin, and the triple
/// re-discloses the record's **own** `Proven` `CapacityBound` (`Capacity{m, dim}`, `ProvenThm`)
/// carried unchanged (VR-5: the disclosed bound is the value's own; the δ-tail itself is the
/// cited theorem's claim — this corpus is Empirical evidence *for* the Proven instantiation,
/// the M-131 posture).
#[test]
fn vsa_reconstruct_recovers_below_the_proven_capacity_bound() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let bundle = reg.get("vsa.bundle").expect("registered");
    let required_dim = reg.get("vsa.required_dim").expect("registered");
    let f = reg.get("vsa.reconstruct").expect("registered");
    let dim = 2048u32;
    let delta = 1e-2;
    for m in [2u64, 3, 5] {
        // The property's precondition through the surfaced query itself: dim ≥ requiredDim(m, δ).
        let items_v = shift_bin(m, 8);
        let req = required_dim("vsa.required_dim", &[&items_v, &fv(delta)])
            .expect("the capacity query accepts");
        assert!(
            u64::from(dim) >= u64_of_bits(&req),
            "corpus precondition: dim {dim} certifies m={m} (required {})",
            u64_of_bits(&req)
        );
        for base_seed in [0u64, 1, 2] {
            let s0 = 10_000 * (base_seed + 1);
            let roles: Vec<Value> = (0..m)
                .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, s0 + i)))
                .collect();
            let fillers: Vec<Value> = (0..m)
                .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, s0 + 100 + i)))
                .collect();
            let pairs: Vec<Value> = roles
                .iter()
                .zip(&fillers)
                .map(|(r, x)| bind("vsa.bind", &[r, x]).expect("bind accepts"))
                .collect();
            let seq = vsa_seq_of(&pairs[0], &pairs);
            let record = bundle("vsa.bundle", &[&seq, &fv(delta)]).expect("certified bundle");
            let record_bound = record.meta().bound().cloned().expect("Proven has a bound");
            let codebook = vsa_seq_of(&fillers[0], &fillers);
            for (k, role) in roles.iter().enumerate() {
                let y = f("vsa.reconstruct", &[&record, role, &codebook, &fv(0.2)]).unwrap_or_else(
                    |e| panic!("m={m} seed={base_seed} role {k}: reconstruct failed: {e}"),
                );
                let [index, confidence, margin] = triple_of(&y);
                assert_eq!(
                    index, k as f64,
                    "m={m} seed={base_seed}: role {k} recovers its own filler below capacity"
                );
                assert!(confidence >= 0.2 && margin > 0.0);
                // The carried pair is the record's own: Proven + its OWN CapacityBound.
                assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
                assert_eq!(
                    y.meta().bound(),
                    Some(&record_bound),
                    "the disclosed bound is the record's own Capacity bound (VR-5)"
                );
                match y.meta().bound() {
                    Some(Bound {
                        kind: BoundKind::Capacity { items, dim: d },
                        basis: BoundBasis::ProvenThm { .. },
                    }) => {
                        assert_eq!(*items, m);
                        assert_eq!(*d, u64::from(dim));
                    }
                    other => {
                        panic!("expected the record's Capacity/ProvenThm bound, got {other:?}")
                    }
                }
            }
        }
    }
}

/// Below-threshold retrieval refuses **explicitly, naming confidence vs threshold** — never a
/// silent low-quality answer (RFC-0003 §6; G2): the m = 3 crosstalk confidence (≈ 1/√3) cannot
/// clear a 0.9 threshold.
#[test]
fn vsa_reconstruct_below_threshold_refuses_explicitly() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let bundle = reg.get("vsa.bundle").expect("registered");
    let f = reg.get("vsa.reconstruct").expect("registered");
    let dim = 2048u32;
    let roles: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, 700 + i)))
        .collect();
    let fillers: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", dim, mapi_atom(dim, 800 + i)))
        .collect();
    let pairs: Vec<Value> = roles
        .iter()
        .zip(&fillers)
        .map(|(r, x)| bind("vsa.bind", &[r, x]).expect("bind accepts"))
        .collect();
    let seq = vsa_seq_of(&pairs[0], &pairs);
    let record = bundle("vsa.bundle", &[&seq, &fv(1e-2)]).expect("certified bundle");
    let codebook = vsa_seq_of(&fillers[0], &fillers);
    match f(
        "vsa.reconstruct",
        &[&record, &roles[0], &codebook, &fv(0.9)],
    ) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("below the threshold 0.9"),
            "the refusal names confidence vs threshold, got: {why}"
        ),
        other => panic!("a below-threshold retrieval must refuse, got {other:?}"),
    }
}

/// The reconstruct dispatch set is {MAP-I, BSC}: an FHRR record refuses **naming the ground** —
/// FHRR's `Empirical` unbind profile covers only a single `vsa.fhrr.bind` product, not a
/// reconstruction record (VR-5: never a stretched profile; surfacing it is append-only).
#[test]
fn vsa_reconstruct_dispatch_excludes_fhrr_explicitly() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.reconstruct").expect("registered");
    let dim = 256u32;
    let record = vsa_hv("FHRR", dim, fhrr_atom(dim, 1));
    let role = vsa_hv("FHRR", dim, fhrr_atom(dim, 2));
    let atoms: Vec<Value> = (0..2)
        .map(|i| vsa_hv("FHRR", dim, fhrr_atom(dim, 10 + i)))
        .collect();
    let codebook = vsa_seq_of(&atoms[0], &atoms);
    match f("vsa.reconstruct", &[&record, &role, &codebook, &fv(0.3)]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("FHRR") && why.contains("MAP-I, BSC"),
            "the refusal names FHRR and the dispatch set, got: {why}"
        ),
        other => panic!("an FHRR reconstruct must refuse, got {other:?}"),
    }
}

/// The cleanup never-silent reject surface: an out-of-set model (naming the M-892 set), an empty
/// codebook, a query↔codebook model/dim mismatch (naming both reprs), the RFC-0010 §4.4
/// identifiability tie, a non-`Exact` codebook atom (only the query slot carries a pair through),
/// a non-Seq codebook, and arity — every one explicit, never a coercion or a coin-flip (G2).
#[test]
fn vsa_cleanup_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.cleanup").expect("registered");
    let dim = 64u32;
    let a = vsa_hv("MAP-I", dim, mapi_atom(dim, 1));
    let b = vsa_hv("MAP-I", dim, mapi_atom(dim, 2));
    let codebook = vsa_seq_of(&a, &[a.clone(), b.clone()]);

    // Out-of-set model: the shared vsa_model_of refusal names the dispatch set.
    let hrr = vsa_hv("HRR", dim, mapi_atom(dim, 3));
    match f("vsa.cleanup", &[&hrr, &codebook]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("MAP-I, FHRR, BSC"),
            "the out-of-set refusal names the dispatch set, got: {why}"
        ),
        other => panic!("an out-of-set model must refuse, got {other:?}"),
    }

    // Empty codebook: nothing to clean up against.
    let empty = vsa_seq_of(&a, &[]);
    match f("vsa.cleanup", &[&a, &empty]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("at least one atom"),
            "the empty-codebook refusal is named, got: {why}"
        ),
        other => panic!("an empty codebook must refuse, got {other:?}"),
    }

    // Query↔codebook model/dim mismatch: named, never coerced (the codebook itself is
    // homogeneous — core Seq well-formedness — but may disagree with the query).
    let small = vsa_hv("MAP-I", 16, mapi_atom(16, 4));
    match f("vsa.cleanup", &[&small, &codebook]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("share the query's model and dim"),
            "the mismatch refusal is named, got: {why}"
        ),
        other => panic!("a query/codebook dim mismatch must refuse, got {other:?}"),
    }

    // Identifiability tie: two identical atoms — the query matches both, margin 0 — is an
    // explicit refusal, never a coin-flip between tied atoms (RFC-0010 §4.4).
    let dup = vsa_seq_of(&a, &[a.clone(), a.clone()]);
    match f("vsa.cleanup", &[&a, &dup]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("non-identifiable"),
            "the tie refusal is named, got: {why}"
        ),
        other => panic!("a tied retrieval must refuse, got {other:?}"),
    }

    // A non-Exact codebook atom: only the query slot carries a pair through.
    let bundle = reg.get("vsa.bundle").expect("registered");
    let big_dim = 2048u32;
    let items: Vec<Value> = (0..2)
        .map(|i| vsa_hv("MAP-I", big_dim, mapi_atom(big_dim, 20 + i)))
        .collect();
    let proven =
        bundle("vsa.bundle", &[&vsa_seq_of(&items[0], &items), &fv(1e-2)]).expect("accepts");
    let fresh = vsa_hv("MAP-I", big_dim, mapi_atom(big_dim, 30));
    let tainted = vsa_seq_of(&fresh, &[fresh.clone(), proven]);
    assert!(
        matches!(
            f("vsa.cleanup", &[&fresh, &tainted]),
            Err(EvalError::ApproxCompositionUnsupported { .. })
        ),
        "a non-Exact codebook atom must refuse composition"
    );

    // Non-Seq codebook / arity: explicit.
    assert!(
        matches!(f("vsa.cleanup", &[&a, &b]), Err(EvalError::PrimType { .. })),
        "a non-Seq codebook must refuse"
    );
    assert!(
        matches!(f("vsa.cleanup", &[&a]), Err(EvalError::PrimType { .. })),
        "arity 1 must refuse"
    );
}

/// The reconstruct-specific reject surface: a role↔record model/dim mismatch (naming both), a
/// non-`Exact` role (only the record slot carries a pair through), a non-Float / out-of-domain
/// threshold (naming the RFC-0003 §6 `[0, 1]` manifest domain), and arity.
#[test]
fn vsa_reconstruct_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let bind = reg.get("vsa.bind").expect("registered");
    let f = reg.get("vsa.reconstruct").expect("registered");
    let dim = 64u32;
    let role = vsa_hv("MAP-I", dim, mapi_atom(dim, 1));
    let filler = vsa_hv("MAP-I", dim, mapi_atom(dim, 2));
    let record = bind("vsa.bind", &[&role, &filler]).expect("accepts");
    let codebook = vsa_seq_of(&filler, std::slice::from_ref(&filler));

    // Role model/dim mismatch: named.
    let foreign_role = vsa_hv("BSC", dim, bsc_atom(dim, 1));
    match f(
        "vsa.reconstruct",
        &[&record, &foreign_role, &codebook, &fv(0.3)],
    ) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("share one model and"),
            "the role-mismatch refusal is named, got: {why}"
        ),
        other => panic!("a role model mismatch must refuse, got {other:?}"),
    }

    // A non-Exact role: no carry rule outside the record slot.
    let bundle = reg.get("vsa.bundle").expect("registered");
    let big_dim = 2048u32;
    let items: Vec<Value> = (0..2)
        .map(|i| vsa_hv("MAP-I", big_dim, mapi_atom(big_dim, 40 + i)))
        .collect();
    let proven =
        bundle("vsa.bundle", &[&vsa_seq_of(&items[0], &items), &fv(1e-2)]).expect("accepts");
    let big_record = bind("vsa.bind", &[&items[0], &items[1]]).expect("accepts");
    let big_codebook = vsa_seq_of(&items[1], std::slice::from_ref(&items[1]));
    assert!(
        matches!(
            f(
                "vsa.reconstruct",
                &[&big_record, &proven, &big_codebook, &fv(0.3)]
            ),
            Err(EvalError::ApproxCompositionUnsupported { .. })
        ),
        "a non-Exact role must refuse composition (only the record carries a pair through)"
    );

    // Threshold domain: finite ∈ [0, 1], named (the ReconInfo manifest domain).
    for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        match f("vsa.reconstruct", &[&record, &role, &codebook, &fv(bad)]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("[0, 1]"),
                "threshold={bad}: the domain refusal is named, got: {why}"
            ),
            other => panic!("threshold={bad} must refuse, got {other:?}"),
        }
    }
    // A non-Float threshold: explicit.
    assert!(
        matches!(
            f(
                "vsa.reconstruct",
                &[&record, &role, &codebook, &byte([false; 8])]
            ),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Float threshold must refuse"
    );
    // Arity.
    assert!(
        matches!(
            f("vsa.reconstruct", &[&record, &role, &codebook]),
            Err(EvalError::PrimType { .. })
        ),
        "arity 3 must refuse"
    );
}

/// `vsa.required_dim` matches the kernel's M-001 probe table and carries the kernel's **`Proven`**
/// `CapacityBound` for exactly the returned (items, dim, δ) instantiation — the query is
/// inspectable: `ProvenThm` basis with the citation + μ + the checked side-condition, composed
/// provenance over both operands. Minimality is pinned too: one dimension below the answer, the
/// kernel issues **no** Proven bound (the side-condition genuinely bites).
#[test]
fn vsa_required_dim_matches_the_kernel_probe_table_and_carries_the_proven_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg
        .get("vsa.required_dim")
        .expect("vsa.required_dim registered");
    for (items, delta, expected) in [
        (3u64, 1e-2, 1141u64),
        (10, 1e-3, 1843),
        (50, 1e-3, 2164),
        (100, 1e-4, 2764),
    ] {
        let items_v = shift_bin(items, 8);
        let delta_v = fv(delta);
        let y = f("vsa.required_dim", &[&items_v, &delta_v]).expect("the probe row accepts");
        assert_eq!(
            u64_of_bits(&y),
            expected,
            "items={items} δ={delta}: requiredDim must match the M-001 probe table"
        );
        assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
        match y.meta().bound() {
            Some(Bound {
                kind: BoundKind::Capacity { items: m, dim: d },
                basis: BoundBasis::ProvenThm { citation },
            }) => {
                assert_eq!(*m, items, "the bound discloses the queried item count");
                assert_eq!(*d, expected, "the bound discloses the returned dim");
                assert!(
                    citation.contains("Clarkson")
                        && citation.contains("requiredDim")
                        && citation.contains("0.1"),
                    "the ProvenThm basis records the citation, μ, and the checked \
                     side-condition: {citation}"
                );
            }
            other => panic!("expected the kernel's Capacity/ProvenThm bound, got {other:?}"),
        }
        match y.meta().provenance() {
            Provenance::Derived { op, inputs } => {
                assert_eq!(op, &mycelium_core::operation_hash("vsa.required_dim"));
                assert_eq!(
                    inputs,
                    &vec![items_v.content_hash(), delta_v.content_hash()]
                );
            }
            other => panic!("expected Derived provenance, got {other:?}"),
        }
        // Minimality: one below the answer, the kernel's checked side-condition fails — no
        // Proven bound exists (the returned dim is the smallest certifiable one).
        assert!(
            capacity::proven_capacity_bound(items, expected - 1, delta).is_none(),
            "items={items} δ={delta}: dim {} must NOT certify",
            expected - 1
        );
        assert!(
            capacity::proven_capacity_bound(items, expected, delta).is_some(),
            "items={items} δ={delta}: the returned dim must certify"
        );
    }
    // The degenerate-but-legal corner (items = 1, δ = 1): requiredDim is 0, disclosed as the
    // smallest well-formed dimension 1 — still sufficient (monotone), documented, never a
    // malformed zero-dim bound.
    let y = f("vsa.required_dim", &[&shift_bin(1, 8), &fv(1.0)]).expect("accepts");
    assert_eq!(u64_of_bits(&y), 1);
    assert!(matches!(
        y.meta().bound(),
        Some(Bound {
            kind: BoundKind::Capacity { items: 1, dim: 1 },
            ..
        })
    ));
}

/// The capacity-query reject surface: zero items (never the kernel's `u64::MAX` sentinel), δ
/// outside `(0, 1]` / non-finite (named), a non-Binary items operand, a non-Float δ, arity, and
/// the Exact-input guard (a non-Exact operand refuses — the query composes no bounds).
#[test]
fn vsa_required_dim_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("vsa.required_dim").expect("registered");
    let delta = fv(1e-2);

    match f("vsa.required_dim", &[&shift_bin(0, 8), &delta]) {
        Err(EvalError::PrimType { why, .. }) => assert!(
            why.contains("zero items"),
            "the zero-items refusal is named, got: {why}"
        ),
        other => panic!("zero items must refuse, got {other:?}"),
    }
    for bad_delta in [0.0, -1e-3, 1.5, f64::NAN, f64::INFINITY] {
        match f("vsa.required_dim", &[&shift_bin(3, 8), &fv(bad_delta)]) {
            Err(EvalError::PrimType { why, .. }) => assert!(
                why.contains("(0, 1]"),
                "δ={bad_delta}: the δ-domain refusal is named, got: {why}"
            ),
            other => panic!("δ={bad_delta} must refuse, got {other:?}"),
        }
    }
    assert!(
        matches!(
            f("vsa.required_dim", &[&fv(3.0), &delta]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Binary items operand must refuse"
    );
    assert!(
        matches!(
            f("vsa.required_dim", &[&shift_bin(3, 8), &byte([false; 8])]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Float δ must refuse"
    );
    assert!(
        matches!(
            f("vsa.required_dim", &[&shift_bin(3, 8)]),
            Err(EvalError::PrimType { .. })
        ),
        "arity 1 must refuse"
    );
}

// ---- M-912: `bytes.eq` / `hash.blake3` ----

/// A `Repr::Bytes` const value over `bytes`.
fn bytes_val(bytes: Vec<u8>) -> Value {
    Value::new(
        Repr::Bytes,
        Payload::Bytes(bytes),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// The deterministic official BLAKE3 test-vector input: a repeating sequence of 251 bytes
/// (`0, 1, 2, …, 250, 0, 1, …`) — exactly the generation rule documented by
/// `BLAKE3-team/BLAKE3`'s `test_vectors/test_vectors.json`.
fn blake3_test_vector_input(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// `hash.blake3` reproduces the **official BLAKE3 test vectors** (known digests, not merely
/// self-consistency against the same crate) at several input lengths, including a
/// chunk-boundary crossing (BLAKE3's chunk size is 1024 bytes) — pins the wrapper's
/// byte-for-byte correctness (right bytes, right order, right length), not just "it calls
/// some hash function". Guarantee `Exact` (M-912; justified by the kernel's own
/// deterministic BLAKE3 use for content addressing, M-103).
#[test]
fn hash_blake3_matches_official_test_vectors() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("hash.blake3").expect("hash.blake3 registered");
    // (input length, expected 32-byte digest as lowercase hex) — the official BLAKE3
    // test vectors (github.com/BLAKE3-team/BLAKE3, test_vectors/test_vectors.json),
    // truncated to the default 32-byte output (the vectors are extended-output; the
    // spec documents that the first 32 bytes must match the default-length output).
    for (len, expected_hex) in [
        (
            0,
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        ),
        (
            1,
            "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
        ),
        (
            3,
            "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f",
        ),
        (
            64,
            "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
        ),
        (
            1024,
            "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
        ),
    ] {
        let input = bytes_val(blake3_test_vector_input(len));
        let out = f("hash.blake3", &[&input]).unwrap_or_else(|e| panic!("len={len}: {e:?}"));
        assert_eq!(out.repr(), &Repr::Bytes, "len={len}: result must be Bytes");
        let Payload::Bytes(digest) = out.payload() else {
            panic!("len={len}: payload must be Bytes")
        };
        assert_eq!(digest.len(), 32, "len={len}: BLAKE3 digest is 32 bytes");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, expected_hex,
            "len={len}: digest must match the official BLAKE3 test vector"
        );
        assert_eq!(
            out.meta().guarantee(),
            GuaranteeStrength::Exact,
            "len={len}: hash.blake3 is Exact"
        );
    }
}

/// `hash.blake3` refuses a non-`Bytes` operand and the wrong arity — never-silent (G2).
#[test]
fn hash_blake3_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("hash.blake3").expect("hash.blake3 registered");
    let non_bytes = byte([true; 8]);
    assert!(
        matches!(
            f("hash.blake3", &[&non_bytes]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Bytes operand must refuse"
    );
    let b = bytes_val(vec![0x01]);
    assert!(
        matches!(f("hash.blake3", &[]), Err(EvalError::PrimType { .. })),
        "arity 0 must refuse"
    );
    assert!(
        matches!(f("hash.blake3", &[&b, &b]), Err(EvalError::PrimType { .. })),
        "arity 2 must refuse"
    );
}

/// `bytes.eq` accepts equal/unequal `Bytes` pairs (including the empty string, and pairs
/// differing only in length) with the `Exact` tag — the M-912 folded-in equality gap.
#[test]
fn bytes_eq_accepts_equal_and_unequal_pairs() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bytes.eq").expect("bytes.eq registered");

    let a = bytes_val(vec![0x01, 0x02, 0x03]);
    let a2 = bytes_val(vec![0x01, 0x02, 0x03]);
    let out = f("bytes.eq", &[&a, &a2]).expect("bytes.eq evaluates");
    assert_eq!(out.repr(), &Repr::Binary { width: 1 });
    assert_eq!(
        out.payload(),
        &Payload::Bits(vec![true]),
        "equal byte strings -> true"
    );
    assert_eq!(out.meta().guarantee(), GuaranteeStrength::Exact);

    let b = bytes_val(vec![0x01, 0x02, 0x04]);
    let out = f("bytes.eq", &[&a, &b]).expect("bytes.eq evaluates");
    assert_eq!(
        out.payload(),
        &Payload::Bits(vec![false]),
        "differing byte -> false"
    );

    // Different lengths (a length-sensitive comparison, not a prefix match).
    let c = bytes_val(vec![0x01, 0x02]);
    let out = f("bytes.eq", &[&a, &c]).expect("bytes.eq evaluates");
    assert_eq!(
        out.payload(),
        &Payload::Bits(vec![false]),
        "different-length byte strings -> false, never a prefix match"
    );

    // The empty string equals itself.
    let empty = bytes_val(vec![]);
    let empty2 = bytes_val(vec![]);
    let out = f("bytes.eq", &[&empty, &empty2]).expect("bytes.eq evaluates");
    assert_eq!(out.payload(), &Payload::Bits(vec![true]), "empty == empty");
}

/// `bytes.eq` refuses a non-`Bytes` operand (either position) and the wrong arity —
/// never-silent (G2).
#[test]
fn bytes_eq_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bytes.eq").expect("bytes.eq registered");
    let b = bytes_val(vec![0x01]);
    let non_bytes = byte([true; 8]);

    assert!(
        matches!(
            f("bytes.eq", &[&non_bytes, &b]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Bytes first operand must refuse"
    );
    assert!(
        matches!(
            f("bytes.eq", &[&b, &non_bytes]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Bytes second operand must refuse"
    );
    assert!(
        matches!(f("bytes.eq", &[&b]), Err(EvalError::PrimType { .. })),
        "arity 1 must refuse"
    );
    assert!(
        matches!(
            f("bytes.eq", &[&b, &b, &b]),
            Err(EvalError::PrimType { .. })
        ),
        "arity 3 must refuse"
    );
}

// --- ADR-040 §2.4 (CU-3): never-silent Binary↔Float conversions --------------------------------

/// An arbitrary-width `Binary{N}` value from an MSB-first bit vector (an `Exact` root value,
/// mirroring [`byte`]/[`b8`] but width-generic — CU-3's conversions exercise widths beyond 8).
fn binv(bits: Vec<bool>) -> Value {
    let width = u32::try_from(bits.len()).expect("test widths fit u32");
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `bin.to_flt` round-trips every in-range unsigned magnitude at `Binary{8}` — checked-exact,
/// `Empirical` tag with the shared zero-deviation bound (ADR-040 §2.6).
#[test]
fn bin_to_flt_round_trips_in_range_magnitudes() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.to_flt").expect("bin.to_flt registered");
    for v in 0u8..=255 {
        let bits: Vec<bool> = (0..8).rev().map(|i| (v >> i) & 1 == 1).collect();
        let a = binv(bits);
        let y = f("bin.to_flt", &[&a]).expect("in-range conversion");
        assert_eq!(
            y.repr(),
            &Repr::Float {
                width: FloatWidth::F64
            }
        );
        assert_eq!(y.payload(), &Payload::Float(f64::from(v)));
        assert_eq!(y.meta().guarantee(), GuaranteeStrength::Empirical);
    }
}

/// `bin.to_flt` refuses a magnitude past the binary64 exact-integer bound (`2^53`) — never a
/// silent lossy round (ADR-040 §2.4/§5: the lossy direction is a reified swap, not this prim).
#[test]
fn bin_to_flt_refuses_past_the_exact_bound() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.to_flt").expect("bin.to_flt registered");
    // 2^53 (54 bits, MSB set, rest zero) — the exact boundary, still in range.
    let mut at_bound = vec![false; 54];
    at_bound[0] = true;
    let a = binv(at_bound);
    assert!(f("bin.to_flt", &[&a]).is_ok(), "2^53 must be in range");

    // 2^54 — one bit past, out of range.
    let mut past_bound = vec![false; 55];
    past_bound[0] = true;
    let b = binv(past_bound);
    assert!(
        matches!(f("bin.to_flt", &[&b]), Err(EvalError::Overflow { .. })),
        "2^54 exceeds the exact-integer bound — must refuse, never round"
    );
}

/// `bin.to_flt` refuses a non-`Binary` operand and the wrong arity — never-silent (G2).
#[test]
fn bin_to_flt_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("bin.to_flt").expect("bin.to_flt registered");
    let a = binv(vec![true; 8]);
    let non_binary = fv(1.0);
    assert!(
        matches!(
            f("bin.to_flt", &[&non_binary]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Binary operand must refuse"
    );
    assert!(
        matches!(f("bin.to_flt", &[]), Err(EvalError::PrimType { .. })),
        "arity 0 must refuse"
    );
    assert!(
        matches!(f("bin.to_flt", &[&a, &a]), Err(EvalError::PrimType { .. })),
        "arity 2 must refuse"
    );
}

/// `flt.to_bin` round-trips every non-negative integer-valued `Float` that fits an 8-bit witness
/// — the width witness's bits are ignored, only its `Binary{M}` width is read (mirrors
/// `bit.width_cast`'s DN-41 shape).
#[test]
fn flt_to_bin_round_trips_in_range_integers() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("flt.to_bin").expect("flt.to_bin registered");
    let witness = binv(vec![false; 8]); // Binary{8} — only the width (8) matters.
    for v in 0u8..=255 {
        let x = fv(f64::from(v));
        let y = f("flt.to_bin", &[&x, &witness]).expect("in-range conversion");
        assert_eq!(y.repr(), &Repr::Binary { width: 8 });
        let bits: Vec<bool> = (0..8).rev().map(|i| (v >> i) & 1 == 1).collect();
        assert_eq!(y.payload(), &Payload::Bits(bits));
        assert_eq!(y.meta().guarantee(), GuaranteeStrength::Empirical);
    }
}

/// `flt.to_bin` refuses NaN, ±inf, a negative value, and a nonzero fractional part — never a
/// silent coercion (ADR-040 §2.4; G2).
#[test]
fn flt_to_bin_refuses_the_never_silent_domain() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("flt.to_bin").expect("flt.to_bin registered");
    let witness = binv(vec![false; 8]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 1.5] {
        let x = fv(bad);
        assert!(
            matches!(
                f("flt.to_bin", &[&x, &witness]),
                Err(EvalError::PrimType { .. })
            ),
            "flt.to_bin({bad}) must refuse (NaN/±inf/negative/fractional)"
        );
    }
}

/// `flt.to_bin` refuses a magnitude that does not fit the witness's target width — never a silent
/// truncation (ADR-040 §2.4/DN-41).
#[test]
fn flt_to_bin_refuses_out_of_target_width() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("flt.to_bin").expect("flt.to_bin registered");
    let witness8 = binv(vec![false; 8]);
    let x = fv(256.0); // one past Binary{8}'s unsigned range [0, 255].
    assert!(
        matches!(
            f("flt.to_bin", &[&x, &witness8]),
            Err(EvalError::Overflow { .. })
        ),
        "256 does not fit Binary{{8}} — must refuse, never truncate"
    );
}

/// `flt.to_bin` refuses a non-`Float` value operand and the wrong arity — never-silent (G2).
#[test]
fn flt_to_bin_reject_surface_is_never_silent() {
    let reg = PrimRegistry::with_builtins();
    let f = reg.get("flt.to_bin").expect("flt.to_bin registered");
    let witness = binv(vec![false; 8]);
    let non_float = binv(vec![true; 8]);
    assert!(
        matches!(
            f("flt.to_bin", &[&non_float, &witness]),
            Err(EvalError::PrimType { .. })
        ),
        "a non-Float value operand must refuse"
    );
    assert!(
        matches!(
            f("flt.to_bin", &[&fv(1.0)]),
            Err(EvalError::PrimType { .. })
        ),
        "arity 1 must refuse"
    );
}

/// Chained composition: converting the result of `flt.add` (an `Empirical` zero-deviation value)
/// through `flt.to_bin` succeeds and stays `Empirical` — the same composability rule the `flt.*`
/// arithmetic group uses ([`crate::prims::flt_result`]'s zero-deviation contract), never a
/// fabricated `Exact` upgrade (VR-5).
#[test]
fn flt_to_bin_composes_over_a_prior_flt_op_result() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("flt.add").expect("flt.add registered");
    let to_bin = reg.get("flt.to_bin").expect("flt.to_bin registered");
    let witness = binv(vec![false; 8]);
    let sum = add("flt.add", &[&fv(3.0), &fv(4.0)]).expect("exact dyadic sum");
    assert_eq!(sum.meta().guarantee(), GuaranteeStrength::Empirical);
    let y = to_bin("flt.to_bin", &[&sum, &witness]).expect("7.0 fits Binary{8}");
    assert_eq!(y.payload(), &Payload::Bits(bits("0000_0111")));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Empirical);
}

// --- RFC-0034 §10 (CU-5): the executable `wrapping` construct — eval-mode dispatch -------------
//
// `eval_wrapping` is **not** registered in `PrimRegistry` (no new `wrapping_*` prim name — RFC-0034
// §10's mode is dispatched here over the existing `bin.add`/`bin.sub`/`bin.mul`, per the CU-5 task
// ruling). These tests call it directly, as the future surface `wrapping { … }` construct's
// lowering will once `mycelium-l1`'s parser/elaborator gain that surface (FLAGged: absent today).

/// `eval_wrapping` over `bin.add`/`bin.sub`/`bin.mul` wraps modulo `2^n` exactly where the
/// non-wrapping prims refuse — tagged `Declared` with the [`WrappingOpt`] marker attached
/// (RFC-0034 §10), never the non-wrapping `Exact`/refuse contract.
#[test]
fn eval_wrapping_wraps_where_the_non_wrapping_prims_refuse() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("bin.add").expect("bin.add registered");
    let sub = reg.get("bin.sub").expect("bin.sub registered");
    let mul = reg.get("bin.mul").expect("bin.mul registered");

    // 127 + 1 = 128, out of B_8 = [-128, 127]: `bin.add` refuses, `eval_wrapping` wraps to -128.
    let a = binv(bits("0111_1111")); // 127
    let b = binv(bits("0000_0001")); // 1
    assert!(
        add("bin.add", &[&a, &b]).is_err(),
        "bin.add must refuse 127 + 1 (never-silent)"
    );
    let y = eval_wrapping("bin.add", &[&a, &b]).expect("wrapping never refuses on range");
    assert_eq!(
        y.payload(),
        &Payload::Bits(bits("1000_0000")), // -128
        "127 + 1 must wrap to -128 (RFC-0034 §10)"
    );
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Declared);
    assert!(
        y.meta().wrapping_opt().is_some(),
        "the WrappingOpt marker must be attached (RFC-0034 §10; M-791)"
    );

    // -128 - 1 = -129, out of range: `bin.sub` refuses, wraps to 127.
    let lo = binv(bits("1000_0000")); // -128
    let one = binv(bits("0000_0001")); // 1
    assert!(sub("bin.sub", &[&lo, &one]).is_err());
    let y = eval_wrapping("bin.sub", &[&lo, &one]).expect("wrapping never refuses on range");
    assert_eq!(y.payload(), &Payload::Bits(bits("0111_1111"))); // 127
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Declared);

    // 16 * 16 = 256 = 2^8, out of B_8: `bin.mul` refuses, wraps to 0.
    let sixteen = binv(bits("0001_0000"));
    assert!(mul("bin.mul", &[&sixteen, &sixteen]).is_err());
    let y = eval_wrapping("bin.mul", &[&sixteen, &sixteen]).expect("wrapping never refuses");
    assert_eq!(y.payload(), &Payload::Bits(bits("0000_0000")));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Declared);
}

/// `eval_wrapping` still agrees with the non-wrapping prim on an **in-range** result — wrapping
/// only opts out of the *range* refusal, never the arithmetic itself.
#[test]
fn eval_wrapping_agrees_with_the_non_wrapping_result_when_in_range() {
    let reg = PrimRegistry::with_builtins();
    let add = reg.get("bin.add").expect("bin.add registered");
    let a = binv(bits("0000_0011")); // 3
    let b = binv(bits("0000_0100")); // 4
    let non_wrapping = add("bin.add", &[&a, &b]).expect("3 + 4 = 7 is in range");
    let wrapping = eval_wrapping("bin.add", &[&a, &b]).expect("wrapping never refuses");
    assert_eq!(non_wrapping.payload(), wrapping.payload());
    assert_eq!(non_wrapping.meta().guarantee(), GuaranteeStrength::Exact);
    assert_eq!(wrapping.meta().guarantee(), GuaranteeStrength::Declared);
}

/// `eval_wrapping` refuses a structural mismatch (unequal widths) and an unsupported prim name —
/// `wrapping` only opts out of the range refusal, never the shape contract (G2).
#[test]
fn eval_wrapping_rejects_structural_mismatches_and_unsupported_prims() {
    let a8 = binv(vec![false; 8]);
    let a4 = binv(vec![false; 4]);
    assert!(
        matches!(
            eval_wrapping("bin.add", &[&a8, &a4]),
            Err(EvalError::PrimType { .. })
        ),
        "unequal widths must refuse"
    );
    assert!(
        matches!(
            eval_wrapping("bit.xor", &[&a8, &a8]),
            Err(EvalError::PrimType { .. })
        ),
        "eval_wrapping only supports bin.add/bin.sub/bin.mul (RFC-0034 §10 — no new prims)"
    );
    assert!(
        matches!(
            eval_wrapping("bin.add", &[&a8]),
            Err(EvalError::PrimType { .. })
        ),
        "arity 1 must refuse"
    );
}
