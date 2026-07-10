//! White-box tests for [`crate::binary`]. Extracted from the logic file as-touched by M-887 (test
//! layout rule, M-797) — the pre-existing codec tests plus the new [`crate::binary::mul`] coverage.
//! M-888 adds the unsigned [`crate::binary::div_rem`] coverage. M-889 adds the logical
//! [`crate::binary::shl`]/[`crate::binary::shr`] coverage. M-766 adds the shared two's-complement
//! [`crate::binary::add`]/[`crate::binary::sub`]/[`crate::binary::neg`] coverage.

use crate::binary::*;

#[test]
fn worked_example_byte() {
    // binary-ternary.md §5: 0b1011_0010 (MSB-first) = −78 in two's complement.
    let bits = [true, false, true, true, false, false, true, false];
    assert_eq!(bits_to_int(&bits), -78);
    assert_eq!(int_to_bits(-78, 8), Some(bits.to_vec()));
}

#[test]
fn range_edges() {
    assert_eq!(
        bits_to_int(&[true, false, false, false, false, false, false, false]),
        -128
    );
    assert_eq!(
        bits_to_int(&[false, true, true, true, true, true, true, true]),
        127
    );
    assert_eq!(int_to_bits(127, 8).map(|b| bits_to_int(&b)), Some(127));
    assert_eq!(int_to_bits(-128, 8).map(|b| bits_to_int(&b)), Some(-128));
    assert_eq!(int_to_bits(128, 8), None); // out of range
    assert_eq!(int_to_bits(-129, 8), None);
}

#[test]
fn round_trips_exhaustively_at_n8() {
    for v in -128..=127 {
        let bits = int_to_bits(v, 8).expect("in range");
        assert_eq!(bits.len(), 8);
        assert_eq!(bits_to_int(&bits), v);
    }
}

// Mutant-witness (binary.rs: `value == 0` → `value != 0` in `int_to_bits`'s n=0 guard):
// the mutation would let any non-zero value return `Some(Vec::new())` (the zero-width
// representation of any integer!) instead of `None`. `round_trips_exhaustively_at_n8` only
// covers n=8, missing this guard entirely.
#[test]
fn int_to_bits_n0_rejects_nonzero() {
    // n=0 has a zero-width repr that can only hold the value 0.
    assert_eq!(
        int_to_bits(0, 0),
        Some(Vec::new()),
        "0 in 0 bits is representable"
    );
    assert_eq!(int_to_bits(1, 0), None, "1 cannot be represented in 0 bits");
    assert_eq!(
        int_to_bits(-1, 0),
        None,
        "-1 cannot be represented in 0 bits"
    );
}

// ---- M-887: `mul` — never-silent two's-complement fixed-width multiply --------------------------

#[test]
fn mul_worked_examples() {
    // 3 * 4 = 12, in range at Binary{8}.
    let a = int_to_bits(3, 8).unwrap();
    let b = int_to_bits(4, 8).unwrap();
    assert_eq!(mul(&a, &b), int_to_bits(12, 8));

    // -3 * 4 = -12.
    let a = int_to_bits(-3, 8).unwrap();
    let b = int_to_bits(4, 8).unwrap();
    assert_eq!(mul(&a, &b), int_to_bits(-12, 8));

    // -3 * -4 = 12.
    let a = int_to_bits(-3, 8).unwrap();
    let b = int_to_bits(-4, 8).unwrap();
    assert_eq!(mul(&a, &b), int_to_bits(12, 8));

    // 0 * anything = 0.
    let zero = int_to_bits(0, 8).unwrap();
    let x = int_to_bits(-100, 8).unwrap();
    assert_eq!(mul(&zero, &x), int_to_bits(0, 8));
}

/// The classic two's-complement multiply-overflow edge: `i8::MIN * -1 = 128`, which does not fit
/// `B_8 = [-128, 127]` — an explicit `None`, never a silent wrap back to `-128`.
#[test]
fn mul_min_times_neg_one_overflows() {
    let min = int_to_bits(-128, 8).unwrap();
    let neg_one = int_to_bits(-1, 8).unwrap();
    assert_eq!(mul(&min, &neg_one), None);
}

#[test]
fn mul_overflow_and_in_range_boundary() {
    // 127 * 2 = 254, out of B_8 ([-128, 127]).
    let a = int_to_bits(127, 8).unwrap();
    let b = int_to_bits(2, 8).unwrap();
    assert_eq!(mul(&a, &b), None);

    // -128 * 1 = -128, exactly the low boundary, in range.
    let a = int_to_bits(-128, 8).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(mul(&a, &b), int_to_bits(-128, 8));

    // 63 * 2 = 126, in range; 64 * 2 = 128, out of range — the boundary is exact, not off-by-one.
    let sixty_three = int_to_bits(63, 8).unwrap();
    let sixty_four = int_to_bits(64, 8).unwrap();
    let two = int_to_bits(2, 8).unwrap();
    assert_eq!(mul(&sixty_three, &two), int_to_bits(126, 8));
    assert_eq!(mul(&sixty_four, &two), None);
}

#[test]
fn mul_rejects_unequal_widths() {
    let a = int_to_bits(1, 4).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(mul(&a, &b), None);
}

#[test]
fn mul_rejects_over_cap_width() {
    // n = 65 exceeds `MUL_MAX_WIDTH` (64) — an explicit `None`, never a silently-truncated
    // native-int computation.
    let a = vec![false; MUL_MAX_WIDTH + 1];
    let b = vec![false; MUL_MAX_WIDTH + 1];
    assert_eq!(mul(&a, &b), None);
    // At the cap itself (n = 64) the boundary is accepted (0 * 0 = 0 trivially).
    let a64 = vec![false; MUL_MAX_WIDTH];
    let b64 = vec![false; MUL_MAX_WIDTH];
    assert_eq!(mul(&a64, &b64), int_to_bits(0, 64));
}

// ---- CU-1: `mul_unsigned` — never-silent unsigned fixed-width multiply --------------------------

#[test]
fn mul_unsigned_worked_examples() {
    // 3 * 4 = 12, in range at U_8 = [0, 255].
    let a = uint_to_bits(3, 8).unwrap();
    let b = uint_to_bits(4, 8).unwrap();
    assert_eq!(mul_unsigned(&a, &b), uint_to_bits(12, 8));

    // 15 * 17 = 255, exactly the high boundary, in range.
    let a = uint_to_bits(15, 8).unwrap();
    let b = uint_to_bits(17, 8).unwrap();
    assert_eq!(mul_unsigned(&a, &b), uint_to_bits(255, 8));

    // 0 * anything = 0.
    let zero = uint_to_bits(0, 8).unwrap();
    let x = uint_to_bits(200, 8).unwrap();
    assert_eq!(mul_unsigned(&zero, &x), uint_to_bits(0, 8));
}

#[test]
fn mul_unsigned_overflow_and_boundary() {
    // 16 * 16 = 256, out of U_8 ([0, 255]) — explicit None, never a wrap to 0.
    let a = uint_to_bits(16, 8).unwrap();
    let b = uint_to_bits(16, 8).unwrap();
    assert_eq!(mul_unsigned(&a, &b), None);

    // 255 * 1 = 255, the high boundary, in range; 255 * 2 = 510 overflows — boundary is exact.
    let max = uint_to_bits(255, 8).unwrap();
    let one = uint_to_bits(1, 8).unwrap();
    let two = uint_to_bits(2, 8).unwrap();
    assert_eq!(mul_unsigned(&max, &one), uint_to_bits(255, 8));
    assert_eq!(mul_unsigned(&max, &two), None);
}

/// The overflow *criterion* differs from signed [`mul`]: `9 * 20 = 180` fits `U_8 = [0,255]` but
/// **not** `B_8 = [-128,127]`. This pins that `mul_unsigned` reads the unsigned range, not `mul`'s.
#[test]
fn mul_unsigned_criterion_differs_from_signed() {
    let a = uint_to_bits(9, 8).unwrap();
    let b = uint_to_bits(20, 8).unwrap();
    assert_eq!(mul_unsigned(&a, &b), uint_to_bits(180, 8)); // unsigned: in range
    assert_eq!(mul(&a, &b), None); // signed: 180 > 127, out of B_8
}

#[test]
fn mul_unsigned_rejects_unequal_and_over_cap_widths() {
    let a = uint_to_bits(1, 4).unwrap();
    let b = uint_to_bits(1, 8).unwrap();
    assert_eq!(mul_unsigned(&a, &b), None); // width mismatch
    let over = vec![false; MUL_MAX_WIDTH + 1];
    assert_eq!(mul_unsigned(&over, &over), None); // over-cap width
                                                  // At the cap (n = 64), 0 * 0 = 0 is accepted.
    let z64 = vec![false; MUL_MAX_WIDTH];
    assert_eq!(mul_unsigned(&z64, &z64), uint_to_bits(0, 64));
}

/// Exhaustive cross-check against the native `u16` oracle at `n = 8`: for every operand pair,
/// `mul_unsigned` returns the exact product iff it fits `U_8`, and `None` iff the native product
/// exceeds `255` — never a silent wrap (the property `mul_u` must satisfy, checked by enumeration).
#[test]
fn mul_unsigned_matches_native_oracle_u8() {
    for a in 0u16..256 {
        for b in 0u16..256 {
            let ab = uint_to_bits(a as u64, 8).unwrap();
            let bb = uint_to_bits(b as u64, 8).unwrap();
            let got = mul_unsigned(&ab, &bb);
            let prod = a * b; // u16 holds up to 255*255 = 65025 without overflow
            if prod <= 255 {
                assert_eq!(got, uint_to_bits(prod as u64, 8), "{a} * {b}");
            } else {
                assert_eq!(got, None, "{a} * {b} should overflow U_8");
            }
        }
    }
}

// ---- CU-6: popcount / clz / ctz — width-preserving bit-manipulation counts --------------------

#[test]
fn popcount_clz_ctz_worked_examples() {
    // 0b0110_1000 (MSB-first): popcount 3, clz 1, ctz 3.
    let x = uint_to_bits(0b0110_1000, 8).unwrap();
    assert_eq!(popcount(&x), uint_to_bits(3, 8).unwrap());
    assert_eq!(clz(&x), uint_to_bits(1, 8).unwrap());
    assert_eq!(ctz(&x), uint_to_bits(3, 8).unwrap());

    // 0b1000_0001: popcount 2, clz 0 (top bit set), ctz 0 (low bit set).
    let y = uint_to_bits(0b1000_0001, 8).unwrap();
    assert_eq!(popcount(&y), uint_to_bits(2, 8).unwrap());
    assert_eq!(clz(&y), uint_to_bits(0, 8).unwrap());
    assert_eq!(ctz(&y), uint_to_bits(0, 8).unwrap());
}

/// The all-zero and all-one boundary: `clz`/`ctz` of all-zero are `n` (no `1` bit); `popcount` of
/// all-one is `n`; and `n` itself fits the `n`-bit result (`popcount(0xFF) = 8 = 0b0000_1000`).
#[test]
fn popcount_clz_ctz_boundaries() {
    let zero = vec![false; 8];
    assert_eq!(popcount(&zero), uint_to_bits(0, 8).unwrap());
    assert_eq!(clz(&zero), uint_to_bits(8, 8).unwrap()); // all-zero ⇒ n
    assert_eq!(ctz(&zero), uint_to_bits(8, 8).unwrap());
    let ones = vec![true; 8];
    assert_eq!(popcount(&ones), uint_to_bits(8, 8).unwrap()); // n set bits
    assert_eq!(clz(&ones), uint_to_bits(0, 8).unwrap());
    assert_eq!(ctz(&ones), uint_to_bits(0, 8).unwrap());
}

/// Exhaustive cross-check against Rust's native `u8` methods at `n = 8` — `popcount`/`clz`/`ctz`
/// agree with `count_ones`/`leading_zeros`/`trailing_zeros` for every value (the property the prims
/// must satisfy). Guards the MSB-first bit order and the all-zero `= n` convention.
#[test]
fn popcount_clz_ctz_match_native_u8() {
    for v in 0u8..=255 {
        let bits = uint_to_bits(u64::from(v), 8).unwrap();
        assert_eq!(
            popcount(&bits),
            uint_to_bits(u64::from(v.count_ones()), 8).unwrap(),
            "popcount {v}"
        );
        assert_eq!(
            clz(&bits),
            uint_to_bits(u64::from(v.leading_zeros()), 8).unwrap(),
            "clz {v}"
        );
        assert_eq!(
            ctz(&bits),
            uint_to_bits(u64::from(v.trailing_zeros()), 8).unwrap(),
            "ctz {v}"
        );
    }
}

/// Arbitrary-width safety: at `n = 100` (> 64), the all-zero `clz` is `100`, which the count encoder
/// represents without hitting the `>> ≥ 64` shift UB `uint_to_bits` would.
#[test]
fn bit_counts_are_arbitrary_width_safe() {
    let zero = vec![false; 100];
    let clz100 = clz(&zero);
    assert_eq!(clz100.len(), 100);
    // 100 = 0b110_0100; the count occupies the low 7 bits, the rest 0.
    assert_eq!(bits_to_uint(&clz100[93..]), 100); // low 7 bits hold 100
    assert!(clz100[..93].iter().all(|&b| !b)); // high 93 bits all zero
}

#[test]
fn mul_n0_is_trivially_zero() {
    assert_eq!(mul(&[], &[]), Some(Vec::new()));
}

/// **Oracle property test (the overflow bound):** the codec-and-i128 multiplier agrees with a
/// direct `i128` oracle for *every* pair at small widths — in range it equals the exact product's
/// encoding, out of range it is `None`. Mirrors `ternary::mul_matches_integer_oracle`.
#[test]
fn mul_matches_integer_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();
                let got = mul(&a, &b);
                let expected = i128::from(x) * i128::from(y);
                if expected >= i128::from(lo) && expected <= i128::from(hi) {
                    let expected_i64 = expected as i64;
                    assert_eq!(got, int_to_bits(expected_i64, n), "mul {x}*{y} at n={n}");
                } else {
                    assert_eq!(got, None, "mul {x}*{y} should overflow at n={n}");
                }
            }
        }
    }
}

// ---- M-888: `div_rem` — never-silent unsigned fixed-width division/remainder -------------------

#[test]
fn uint_round_trips_exhaustively_at_n8() {
    for v in 0u64..=255 {
        let bits = uint_to_bits(v, 8).expect("in range");
        assert_eq!(bits.len(), 8);
        assert_eq!(bits_to_uint(&bits), v);
    }
}

#[test]
fn uint_to_bits_rejects_out_of_range() {
    assert_eq!(uint_to_bits(255, 8), Some(vec![true; 8]));
    assert_eq!(uint_to_bits(256, 8), None); // out of range for 8 bits
    assert_eq!(uint_to_bits(0, 0), Some(Vec::new()));
    assert_eq!(uint_to_bits(1, 0), None);
}

#[test]
fn div_rem_worked_examples() {
    // 7 / 2 = 3 remainder 1.
    let a = uint_to_bits(7, 8).unwrap();
    let b = uint_to_bits(2, 8).unwrap();
    let (q, r) = div_rem(&a, &b).expect("7 / 2");
    assert_eq!(bits_to_uint(&q), 3);
    assert_eq!(bits_to_uint(&r), 1);

    // 255 / 1 = 255 remainder 0 (upper boundary at n=8).
    let a = uint_to_bits(255, 8).unwrap();
    let b = uint_to_bits(1, 8).unwrap();
    let (q, r) = div_rem(&a, &b).expect("255 / 1");
    assert_eq!(bits_to_uint(&q), 255);
    assert_eq!(bits_to_uint(&r), 0);

    // 0 / anything nonzero = 0 remainder 0.
    let zero = uint_to_bits(0, 8).unwrap();
    let x = uint_to_bits(17, 8).unwrap();
    let (q, r) = div_rem(&zero, &x).expect("0 / 17");
    assert_eq!(bits_to_uint(&q), 0);
    assert_eq!(bits_to_uint(&r), 0);
}

/// Division by zero is an explicit `None` — never a panic, never a silently-defined value.
#[test]
fn div_rem_by_zero_refuses() {
    let a = uint_to_bits(7, 8).unwrap();
    let zero = uint_to_bits(0, 8).unwrap();
    assert_eq!(div_rem(&a, &zero), None);
    // Even 0 / 0 refuses — it is not special-cased to (0, 0).
    assert_eq!(div_rem(&zero, &zero), None);
}

#[test]
fn div_rem_rejects_unequal_widths() {
    let a = uint_to_bits(1, 4).unwrap();
    let b = uint_to_bits(1, 8).unwrap();
    assert_eq!(div_rem(&a, &b), None);
}

#[test]
fn div_rem_rejects_over_cap_width() {
    let a = vec![false; DIV_MAX_WIDTH + 1];
    let b = vec![false; DIV_MAX_WIDTH + 1];
    assert_eq!(div_rem(&a, &b), None);
    // At the cap itself the boundary is accepted (subject to the operands, not the width).
    let a64 = uint_to_bits(10, 64).unwrap();
    let b64 = uint_to_bits(3, 64).unwrap();
    let (q, r) = div_rem(&a64, &b64).expect("10 / 3 at n=64");
    assert_eq!(bits_to_uint(&q), 3);
    assert_eq!(bits_to_uint(&r), 1);
}

#[test]
fn div_rem_n0_is_div_by_zero() {
    // The zero-width bitvector's only value is 0; 0 / 0 refuses, it is not silently `(0, 0)`.
    assert_eq!(div_rem(&[], &[]), None);
}

/// **Oracle property test (the Euclidean identity):** for every pair at small widths with a
/// nonzero divisor, `a == quotient * b + remainder` holds bit-exactly and `remainder < b`; every
/// zero-divisor pair is an explicit `None`. Mirrors `mul_matches_integer_oracle`.
#[test]
// The `x / y` in the `y != 0` branch is the trusted native oracle this test checks `div_rem`
// against; it must stay plain (clippy 1.96 `manual_checked_ops` would obscure the oracle).
#[allow(clippy::manual_checked_ops)]
fn div_rem_matches_euclidean_identity_oracle() {
    for n in 1u32..=8 {
        let hi: u64 = (1u64 << n) - 1;
        for x in 0..=hi {
            for y in 0..=hi {
                let a = uint_to_bits(x, n).unwrap();
                let b = uint_to_bits(y, n).unwrap();
                let got = div_rem(&a, &b);
                if y == 0 {
                    assert_eq!(
                        got, None,
                        "div_rem {x}/{y} should refuse (div-by-zero) at n={n}"
                    );
                } else {
                    let (q, r) =
                        got.unwrap_or_else(|| panic!("div_rem {x}/{y} at n={n} must succeed"));
                    let qv = bits_to_uint(&q);
                    let rv = bits_to_uint(&r);
                    assert_eq!(qv, x / y, "quotient {x}/{y} at n={n}");
                    assert_eq!(rv, x % y, "remainder {x}/{y} at n={n}");
                    // Euclidean identity, bit-exact.
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
}

// ---- M-889: `shl`/`shr` — never-silent logical fixed-width shifts ------------------------------

#[test]
fn shl_worked_examples() {
    // 1 << 3 = 8 at Binary{8}.
    let a = uint_to_bits(1, 8).unwrap();
    let k = uint_to_bits(3, 8).unwrap();
    let got = shl(&a, &k).expect("1 << 3");
    assert_eq!(bits_to_uint(&got), 8);

    // Bits shifted past the MSB are dropped (never wrapped): 0b1111_1111 << 1 = 0b1111_1110.
    let a = uint_to_bits(255, 8).unwrap();
    let k = uint_to_bits(1, 8).unwrap();
    let got = shl(&a, &k).expect("255 << 1");
    assert_eq!(bits_to_uint(&got), 254);
}

#[test]
fn shr_worked_examples() {
    // 8 >> 3 = 1 at Binary{8}.
    let a = uint_to_bits(8, 8).unwrap();
    let k = uint_to_bits(3, 8).unwrap();
    let got = shr(&a, &k).expect("8 >> 3");
    assert_eq!(bits_to_uint(&got), 1);

    // Bits shifted past the LSB are dropped, zero-filled at the MSB (logical, not arithmetic):
    // 0b1000_0000 >> 4 = 0b0000_1000, not sign-extended.
    let a = uint_to_bits(0b1000_0000, 8).unwrap();
    let k = uint_to_bits(4, 8).unwrap();
    let got = shr(&a, &k).expect("0x80 >> 4");
    assert_eq!(bits_to_uint(&got), 0b0000_1000);
}

/// Shifting by `0` is the identity, for both directions.
#[test]
fn shift_by_zero_is_identity() {
    for n in 1u32..=8 {
        let hi: u64 = (1u64 << n) - 1;
        for v in 0..=hi {
            let a = uint_to_bits(v, n).unwrap();
            let zero = uint_to_bits(0, n).unwrap();
            assert_eq!(shl(&a, &zero), Some(a.clone()), "shl by 0 at v={v}, n={n}");
            assert_eq!(shr(&a, &zero), Some(a.clone()), "shr by 0 at v={v}, n={n}");
        }
    }
}

/// A shift amount `>= width` is an explicit `None` — never-silent (never UB, a modulo-wrapped
/// shift amount, or a silently-zeroed result).
#[test]
fn shift_amount_at_or_above_width_refuses() {
    let a = uint_to_bits(0b0000_0001, 8).unwrap();
    // Exactly at the width boundary.
    let k8 = uint_to_bits(8, 8).unwrap();
    assert_eq!(shl(&a, &k8), None, "shift by exactly the width must refuse");
    assert_eq!(shr(&a, &k8), None, "shift by exactly the width must refuse");
    // Above the width.
    let k255 = uint_to_bits(255, 8).unwrap();
    assert_eq!(shl(&a, &k255), None, "shift above the width must refuse");
    assert_eq!(shr(&a, &k255), None, "shift above the width must refuse");
    // One below the width boundary is in range (n=8, k=7 is the max valid shift).
    let k7 = uint_to_bits(7, 8).unwrap();
    assert!(shl(&a, &k7).is_some(), "shift by width-1 must succeed");
    assert!(shr(&a, &k7).is_some(), "shift by width-1 must succeed");
}

#[test]
fn shift_rejects_unequal_widths() {
    let a = uint_to_bits(1, 4).unwrap();
    let k = uint_to_bits(1, 8).unwrap();
    assert_eq!(shl(&a, &k), None);
    assert_eq!(shr(&a, &k), None);
}

#[test]
fn shift_rejects_over_cap_width() {
    let a = vec![false; SHIFT_MAX_WIDTH + 1];
    let k = vec![false; SHIFT_MAX_WIDTH + 1];
    assert_eq!(shl(&a, &k), None);
    assert_eq!(shr(&a, &k), None);
    // At the cap itself the boundary is accepted.
    let a64 = uint_to_bits(1, 64).unwrap();
    let k64 = uint_to_bits(63, 64).unwrap();
    assert_eq!(shl(&a64, &k64).map(|b| bits_to_uint(&b)), Some(1u64 << 63));
    let a64b = uint_to_bits(u64::MAX, 64).unwrap();
    assert_eq!(shr(&a64b, &k64).map(|b| bits_to_uint(&b)), Some(1u64));
}

/// The zero-width bitvector's only representable shift amount is `0`, and `0 >= width(0)`, so
/// `n == 0` always refuses — it is not special-cased to a trivial identity.
#[test]
fn shift_n0_always_refuses() {
    assert_eq!(shl(&[], &[]), None);
    assert_eq!(shr(&[], &[]), None);
}

/// **Oracle property test (the shift-amount bound):** for every value/shift-amount pair at a small
/// width, `shl`/`shr` agree with a native `u64` shift for in-range amounts and refuse explicitly
/// for `k >= n`. Mirrors `div_rem_matches_euclidean_identity_oracle`.
#[test]
fn shift_matches_native_oracle() {
    for n in 1u32..=8 {
        let hi: u64 = (1u64 << n) - 1;
        for v in 0..=hi {
            for k in 0..=hi {
                let a = uint_to_bits(v, n).unwrap();
                let kb = uint_to_bits(k, n).unwrap();
                let got_shl = shl(&a, &kb);
                let got_shr = shr(&a, &kb);
                if k >= u64::from(n) {
                    assert_eq!(got_shl, None, "shl {v}<<{k} at n={n} should refuse");
                    assert_eq!(got_shr, None, "shr {v}>>{k} at n={n} should refuse");
                } else {
                    let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
                    let expected_shl = (v << k) & mask;
                    let expected_shr = v >> k;
                    assert_eq!(
                        got_shl.map(|b| bits_to_uint(&b)),
                        Some(expected_shl),
                        "shl {v}<<{k} at n={n}"
                    );
                    assert_eq!(
                        got_shr.map(|b| bits_to_uint(&b)),
                        Some(expected_shr),
                        "shr {v}>>{k} at n={n}"
                    );
                }
            }
        }
    }
}

// ---- M-766: `add`/`sub`/`neg` — the shared two's-complement set's genuinely-missing members ------

#[test]
fn add_worked_examples() {
    // 3 + 4 = 7, in range at Binary{8}.
    let a = int_to_bits(3, 8).unwrap();
    let b = int_to_bits(4, 8).unwrap();
    assert_eq!(add(&a, &b), int_to_bits(7, 8));

    // -3 + 4 = 1.
    let a = int_to_bits(-3, 8).unwrap();
    let b = int_to_bits(4, 8).unwrap();
    assert_eq!(add(&a, &b), int_to_bits(1, 8));

    // -3 + -4 = -7.
    let a = int_to_bits(-3, 8).unwrap();
    let b = int_to_bits(-4, 8).unwrap();
    assert_eq!(add(&a, &b), int_to_bits(-7, 8));

    // 0 + anything = anything.
    let zero = int_to_bits(0, 8).unwrap();
    let x = int_to_bits(-100, 8).unwrap();
    assert_eq!(add(&zero, &x), int_to_bits(-100, 8));
}

/// The genuine gap `add` closes relative to `bit.add`: `5 + 3 = 8` is unsigned-in-range at
/// `Binary{4}` (`[0,15]`) but signed-out-of-range (`B_4 = [-8,7]`) — `add` refuses it, honoring the
/// two's-complement/signed domain `bit.add`'s unsigned overflow criterion would silently miss.
#[test]
fn add_refuses_where_unsigned_addition_would_not() {
    let a = int_to_bits(5, 4).unwrap();
    let b = int_to_bits(3, 4).unwrap();
    assert_eq!(
        add(&a, &b),
        None,
        "5 + 3 at Binary{{4}} is signed-out-of-range B_4 = [-8,7]"
    );
}

#[test]
fn add_overflow_and_in_range_boundary() {
    // 127 + 1 = 128, out of B_8 ([-128, 127]).
    let a = int_to_bits(127, 8).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(add(&a, &b), None);

    // -128 + 0 = -128, exactly the low boundary, in range.
    let a = int_to_bits(-128, 8).unwrap();
    let b = int_to_bits(0, 8).unwrap();
    assert_eq!(add(&a, &b), int_to_bits(-128, 8));

    // -128 + -1 = -129, out of range (the low-boundary overflow).
    let a = int_to_bits(-128, 8).unwrap();
    let b = int_to_bits(-1, 8).unwrap();
    assert_eq!(add(&a, &b), None);
}

#[test]
fn add_rejects_unequal_widths() {
    let a = int_to_bits(1, 4).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(add(&a, &b), None);
}

#[test]
fn add_rejects_over_cap_width() {
    let a = vec![false; TC_MAX_WIDTH + 1];
    let b = vec![false; TC_MAX_WIDTH + 1];
    assert_eq!(add(&a, &b), None);
    let a64 = vec![false; TC_MAX_WIDTH];
    let b64 = vec![false; TC_MAX_WIDTH];
    assert_eq!(add(&a64, &b64), int_to_bits(0, 64));
}

#[test]
fn add_n0_is_trivially_zero() {
    assert_eq!(add(&[], &[]), Some(Vec::new()));
}

/// **Oracle property test (the overflow bound):** `add` agrees with an `i128` oracle for every pair
/// at small widths — in range it equals the exact sum's encoding, out of range it is `None`.
#[test]
fn add_matches_integer_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();
                let got = add(&a, &b);
                let expected = i128::from(x) + i128::from(y);
                if expected >= i128::from(lo) && expected <= i128::from(hi) {
                    let expected_i64 = expected as i64;
                    assert_eq!(got, int_to_bits(expected_i64, n), "add {x}+{y} at n={n}");
                } else {
                    assert_eq!(got, None, "add {x}+{y} should overflow at n={n}");
                }
            }
        }
    }
}

#[test]
fn sub_worked_examples() {
    // 7 - 4 = 3, in range at Binary{8}.
    let a = int_to_bits(7, 8).unwrap();
    let b = int_to_bits(4, 8).unwrap();
    assert_eq!(sub(&a, &b), int_to_bits(3, 8));

    // -3 - 4 = -7.
    let a = int_to_bits(-3, 8).unwrap();
    let b = int_to_bits(4, 8).unwrap();
    assert_eq!(sub(&a, &b), int_to_bits(-7, 8));

    // 4 - (-3) = 7.
    let a = int_to_bits(4, 8).unwrap();
    let b = int_to_bits(-3, 8).unwrap();
    assert_eq!(sub(&a, &b), int_to_bits(7, 8));

    // anything - itself = 0.
    let x = int_to_bits(-100, 8).unwrap();
    assert_eq!(sub(&x, &x), int_to_bits(0, 8));
}

/// The genuine gap `sub` closes relative to `bit.sub`: `-8 - 1 = -9` has no unsigned borrow-out at
/// `Binary{4}` in the way `bit.sub` checks it (both operands' unsigned magnitudes: `8 - 1 = 7`, no
/// borrow), but is signed-out-of-range (`B_4 = [-8,7]`) — `sub` refuses it explicitly.
#[test]
fn sub_overflow_and_in_range_boundary() {
    // -128 - 1 = -129, out of B_8 ([-128, 127]).
    let a = int_to_bits(-128, 8).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(sub(&a, &b), None);

    // 127 - (-1) = 128, out of range (the high-boundary overflow).
    let a = int_to_bits(127, 8).unwrap();
    let b = int_to_bits(-1, 8).unwrap();
    assert_eq!(sub(&a, &b), None);

    // -128 - 0 = -128, exactly the low boundary, in range.
    let a = int_to_bits(-128, 8).unwrap();
    let b = int_to_bits(0, 8).unwrap();
    assert_eq!(sub(&a, &b), int_to_bits(-128, 8));
}

#[test]
fn sub_rejects_unequal_widths() {
    let a = int_to_bits(1, 4).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(sub(&a, &b), None);
}

#[test]
fn sub_rejects_over_cap_width() {
    let a = vec![false; TC_MAX_WIDTH + 1];
    let b = vec![false; TC_MAX_WIDTH + 1];
    assert_eq!(sub(&a, &b), None);
    let a64 = vec![false; TC_MAX_WIDTH];
    let b64 = vec![false; TC_MAX_WIDTH];
    assert_eq!(sub(&a64, &b64), int_to_bits(0, 64));
}

#[test]
fn sub_n0_is_trivially_zero() {
    assert_eq!(sub(&[], &[]), Some(Vec::new()));
}

/// **Oracle property test (the overflow bound):** `sub` agrees with an `i128` oracle for every pair
/// at small widths.
#[test]
fn sub_matches_integer_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();
                let got = sub(&a, &b);
                let expected = i128::from(x) - i128::from(y);
                if expected >= i128::from(lo) && expected <= i128::from(hi) {
                    let expected_i64 = expected as i64;
                    assert_eq!(got, int_to_bits(expected_i64, n), "sub {x}-{y} at n={n}");
                } else {
                    assert_eq!(got, None, "sub {x}-{y} should overflow at n={n}");
                }
            }
        }
    }
}

#[test]
fn neg_worked_examples() {
    let three = int_to_bits(3, 8).unwrap();
    assert_eq!(neg(&three), int_to_bits(-3, 8));

    let neg_three = int_to_bits(-3, 8).unwrap();
    assert_eq!(neg(&neg_three), int_to_bits(3, 8));

    let zero = int_to_bits(0, 8).unwrap();
    assert_eq!(neg(&zero), int_to_bits(0, 8));
}

/// The classic two's-complement negate-overflow edge: `B_n`'s minimum value `-2^(n-1)` has no
/// positive counterpart in `B_n` — an explicit `None`, never a silent wrap back to itself.
#[test]
fn neg_min_value_overflows() {
    let min8 = int_to_bits(-128, 8).unwrap();
    assert_eq!(
        neg(&min8),
        None,
        "-(-128) = 128 does not fit B_8 = [-128, 127]"
    );

    let min4 = int_to_bits(-8, 4).unwrap();
    assert_eq!(neg(&min4), None, "-(-8) = 8 does not fit B_4 = [-8, 7]");

    // The maximum value negates fine (it is not the boundary case).
    let max8 = int_to_bits(127, 8).unwrap();
    assert_eq!(neg(&max8), int_to_bits(-127, 8));
}

#[test]
fn neg_rejects_over_cap_width() {
    let a = vec![false; TC_MAX_WIDTH + 1];
    assert_eq!(neg(&a), None);
    let a64 = vec![false; TC_MAX_WIDTH];
    assert_eq!(neg(&a64), int_to_bits(0, 64));
}

#[test]
fn neg_n0_is_trivially_zero() {
    assert_eq!(neg(&[]), Some(Vec::new()));
}

/// **Oracle property test (the overflow bound):** `neg` agrees with an `i128` oracle for every value
/// at small widths — in range it equals the exact negation's encoding, the `MIN` value is `None`.
#[test]
fn neg_matches_integer_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for x in lo..=hi {
            let a = int_to_bits(x, n).unwrap();
            let got = neg(&a);
            let expected = -i128::from(x);
            if expected >= i128::from(lo) && expected <= i128::from(hi) {
                let expected_i64 = expected as i64;
                assert_eq!(got, int_to_bits(expected_i64, n), "neg {x} at n={n}");
            } else {
                assert_eq!(got, None, "neg {x} should overflow at n={n}");
            }
        }
    }
}

// ---- M-767: the signedness-split signed op set — div/rem/shr/cmp (RFC-0033 §4.1.2/§4.1.3) ------

use core::cmp::Ordering;

/// Truncation toward zero (SMT-LIB `bvsdiv`/`bvsrem`; the module doc's rounding-convention note):
/// `-7 / 2 = -3` remainder `-1` — a floored convention would give `-4` remainder `1`, so these
/// examples pin the convention itself, not just arithmetic.
#[test]
fn div_rem_signed_worked_examples_pin_truncation() {
    let enc = |v: i64| int_to_bits(v, 8).unwrap();
    // 7 / 2 = 3 r 1.
    assert_eq!(div_signed(&enc(7), &enc(2)), int_to_bits(3, 8));
    assert_eq!(rem_signed(&enc(7), &enc(2)), int_to_bits(1, 8));
    // -7 / 2 = -3 r -1 (truncated toward zero; remainder sign follows the dividend).
    assert_eq!(div_signed(&enc(-7), &enc(2)), int_to_bits(-3, 8));
    assert_eq!(rem_signed(&enc(-7), &enc(2)), int_to_bits(-1, 8));
    // 7 / -2 = -3 r 1.
    assert_eq!(div_signed(&enc(7), &enc(-2)), int_to_bits(-3, 8));
    assert_eq!(rem_signed(&enc(7), &enc(-2)), int_to_bits(1, 8));
    // -7 / -2 = 3 r -1.
    assert_eq!(div_signed(&enc(-7), &enc(-2)), int_to_bits(3, 8));
    assert_eq!(rem_signed(&enc(-7), &enc(-2)), int_to_bits(-1, 8));
}

/// The single signed-division overflow case: `B_8`'s minimum `-128 ÷ -1` has true quotient `+128`,
/// out of `B_8 = [-128, 127]` — an explicit refusal, never a wrap back to `-128` (RFC-0033 §4.1.3).
/// The remainder's exact result `0` fits, so `rem_signed` succeeds — deliberately not Rust's
/// `checked_rem` over-refusal (see the `rem_signed` doc comment).
#[test]
fn div_signed_min_by_neg_one_overflows_rem_succeeds() {
    let min = int_to_bits(-128, 8).unwrap();
    let neg_one = int_to_bits(-1, 8).unwrap();
    assert_eq!(
        div_signed(&min, &neg_one),
        None,
        "-128 / -1 must be an explicit overflow, never a silent wrap"
    );
    assert_eq!(
        rem_signed(&min, &neg_one),
        int_to_bits(0, 8),
        "-128 % -1 = 0 fits B_8 exactly and must succeed"
    );
    // The same edge at the width cap itself (n = 64): i64::MIN / -1.
    let min64 = int_to_bits(i64::MIN, 64).unwrap();
    let neg_one64 = int_to_bits(-1, 64).unwrap();
    assert_eq!(div_signed(&min64, &neg_one64), None);
    assert_eq!(rem_signed(&min64, &neg_one64), int_to_bits(0, 64));
}

#[test]
fn div_rem_signed_by_zero_refuses() {
    let a = int_to_bits(-7, 8).unwrap();
    let zero = int_to_bits(0, 8).unwrap();
    assert_eq!(div_signed(&a, &zero), None);
    assert_eq!(rem_signed(&a, &zero), None);
    // 0 / 0 refuses too (div-by-zero, not a special case defined away).
    assert_eq!(div_signed(&zero, &zero), None);
    assert_eq!(rem_signed(&zero, &zero), None);
    // n = 0: the only representable divisor is 0, so every n = 0 pair is div-by-zero.
    assert_eq!(div_signed(&[], &[]), None);
    assert_eq!(rem_signed(&[], &[]), None);
}

#[test]
fn div_rem_signed_reject_unequal_and_over_cap_widths() {
    let a = int_to_bits(1, 8).unwrap();
    let b = int_to_bits(1, 4).unwrap();
    assert_eq!(div_signed(&a, &b), None);
    assert_eq!(rem_signed(&a, &b), None);
    let wide = vec![false; DIV_MAX_WIDTH + 1];
    assert_eq!(div_signed(&wide, &wide), None);
    assert_eq!(rem_signed(&wide, &wide), None);
    // At the cap itself the boundary is accepted: 10 / 3 at n = 64.
    let a64 = int_to_bits(10, 64).unwrap();
    let b64 = int_to_bits(3, 64).unwrap();
    assert_eq!(div_signed(&a64, &b64), int_to_bits(3, 64));
    assert_eq!(rem_signed(&a64, &b64), int_to_bits(1, 64));
}

/// **Oracle property test (the signed-division identity on the domain):** for every operand pair
/// at small widths, either the pair refuses for the *right* reason (div-by-zero; the `min ÷ −1`
/// quotient overflow) or the truncated-division identity holds exactly against the `i64` oracle:
/// `a == q·b + r`, `|r| < |b|`, `sign(r) ∈ {0, sign(a)}`, and `q`/`r` equal Rust's own truncated
/// `/`/`%`. Mirrors `div_rem_matches_euclidean_identity_oracle` (the unsigned twin).
#[test]
fn div_rem_signed_match_truncated_identity_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();
                let got_q = div_signed(&a, &b);
                let got_r = rem_signed(&a, &b);
                if y == 0 {
                    assert_eq!(got_q, None, "div_signed {x}/{y} at n={n}: div-by-zero");
                    assert_eq!(got_r, None, "rem_signed {x}%{y} at n={n}: div-by-zero");
                    continue;
                }
                if x == lo && y == -1 {
                    assert_eq!(got_q, None, "div_signed min/-1 at n={n}: overflow");
                    assert_eq!(got_r, int_to_bits(0, n), "rem_signed min%-1 = 0 at n={n}");
                    continue;
                }
                let q = x / y; // Rust `/` is truncated toward zero — the pinned convention.
                let r = x % y; // Rust `%`: sign follows the dividend.
                assert_eq!(got_q, int_to_bits(q, n), "div_signed {x}/{y} at n={n}");
                assert_eq!(got_r, int_to_bits(r, n), "rem_signed {x}%{y} at n={n}");
                // The identity, from the raw oracle values (belt-and-braces over the encoding).
                assert_eq!(x, q * y + r, "identity a == q*b + r at {x}/{y}, n={n}");
                assert!(r.abs() < y.abs(), "|r| < |b| at {x}/{y}, n={n}");
                assert!(
                    r == 0 || (r < 0) == (x < 0),
                    "sign(r) follows the dividend at {x}/{y}, n={n}"
                );
            }
        }
    }
}

#[test]
fn shr_signed_worked_examples_pin_sign_extension() {
    let enc = |v: i64| int_to_bits(v, 8).unwrap();
    let k = |v: u64| uint_to_bits(v, 8).unwrap();
    // -8 >> 1 = -4 (sign bits shifted in, value halves toward −∞).
    assert_eq!(shr_signed(&enc(-8), &k(1)), int_to_bits(-4, 8));
    // -128 >> 4 = -8 (0b1000_0000 → 0b1111_1000: four copies of the sign bit shifted in —
    // the logical `shr` would give 0b0000_1000 = +8 instead).
    assert_eq!(shr_signed(&enc(-128), &k(4)), int_to_bits(-8, 8));
    // -1 >> k = -1 for every in-range k (all-ones is a fixed point of sign extension).
    for kk in 0..8 {
        assert_eq!(shr_signed(&enc(-1), &k(kk)), int_to_bits(-1, 8), "k={kk}");
    }
    // Non-negative values agree with the logical shift (the sign bit is 0).
    assert_eq!(shr_signed(&enc(64), &k(3)), int_to_bits(8, 8));
    // Shift by 0 is the identity.
    assert_eq!(shr_signed(&enc(-100), &k(0)), int_to_bits(-100, 8));
}

#[test]
fn shr_signed_refusals_mirror_the_logical_shift() {
    let a = int_to_bits(-1, 8).unwrap();
    // Shift amount >= width refuses (exactly the width, and above it).
    assert_eq!(shr_signed(&a, &uint_to_bits(8, 8).unwrap()), None);
    assert_eq!(shr_signed(&a, &uint_to_bits(255, 8).unwrap()), None);
    // Width mismatch refuses.
    assert_eq!(shr_signed(&a, &uint_to_bits(1, 4).unwrap()), None);
    // Over-cap width refuses; the cap boundary is accepted.
    let wide = vec![true; SHIFT_MAX_WIDTH + 1];
    assert_eq!(shr_signed(&wide, &wide), None);
    let a64 = int_to_bits(i64::MIN, 64).unwrap();
    let k63 = uint_to_bits(63, 64).unwrap();
    assert_eq!(shr_signed(&a64, &k63), int_to_bits(-1, 64));
    // n = 0 always refuses (the only representable amount, 0, is >= the width 0).
    assert_eq!(shr_signed(&[], &[]), None);
}

/// **Oracle property test (sign extension):** for every value/amount pair at small widths,
/// `shr_signed` agrees with the native `i64` arithmetic shift (`>>` on a signed operand) for
/// in-range amounts — `⌊a / 2^k⌋`, toward −∞ — and refuses explicitly for `k >= n`. Mirrors
/// `shift_matches_native_oracle` (the logical twin).
#[test]
fn shr_signed_matches_native_arithmetic_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for v in lo..=hi {
            for k in 0..=(u64::from(n) + 2) {
                let a = int_to_bits(v, n).unwrap();
                let Some(kb) = uint_to_bits(k, n) else {
                    continue; // amount not representable at this width — no operand to test.
                };
                let got = shr_signed(&a, &kb);
                if k >= u64::from(n) {
                    assert_eq!(got, None, "shr_signed {v}>>{k} at n={n} should refuse");
                } else {
                    let expected = v >> k; // native arithmetic shift on i64.
                    assert_eq!(
                        got,
                        int_to_bits(expected, n),
                        "shr_signed {v}>>{k} at n={n}"
                    );
                }
            }
        }
    }
}

#[test]
fn cmp_signed_worked_examples() {
    let enc = |v: i64| int_to_bits(v, 8).unwrap();
    // The distinguishing case vs the unsigned D1 order: -1 (0b1111_1111) < 0, where the unsigned
    // magnitude order says 255 > 0.
    assert_eq!(cmp_signed(&enc(-1), &enc(0)), Some(Ordering::Less));
    assert_eq!(cmp_signed(&enc(0), &enc(-1)), Some(Ordering::Greater));
    assert_eq!(cmp_signed(&enc(-128), &enc(127)), Some(Ordering::Less));
    assert_eq!(cmp_signed(&enc(5), &enc(5)), Some(Ordering::Equal));
    // Width mismatch is `None` (the caller refuses explicitly — never a silent ordering).
    assert_eq!(cmp_signed(&enc(0), &int_to_bits(0, 4).unwrap()), None);
    // The zero-width bitvector compares Equal (B_0 = {0}).
    assert_eq!(cmp_signed(&[], &[]), Some(Ordering::Equal));
}

/// `cmp_signed` is width-unbounded (purely structural — no integer decode): a 100-bit negative
/// sorts below a 100-bit non-negative, beyond the `i64` codec's 64-bit exactness cap.
#[test]
fn cmp_signed_orders_beyond_the_i64_codec_cap() {
    let mut neg = vec![false; 100];
    neg[0] = true; // sign bit set — a large-magnitude negative.
    let pos = vec![false; 100]; // zero.
    assert_eq!(cmp_signed(&neg, &pos), Some(Ordering::Less));
    assert_eq!(cmp_signed(&pos, &neg), Some(Ordering::Greater));
    assert_eq!(cmp_signed(&neg, &neg), Some(Ordering::Equal));
}

/// **Oracle property test (total order over the signed range):** exhaustively at small widths,
/// `cmp_signed` equals the `i64` value order under `bits_to_int` — which makes it a total order on
/// the domain (the oracle's order is), covering antisymmetry/transitivity by transport.
#[test]
fn cmp_signed_matches_the_value_order_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();
                assert_eq!(
                    cmp_signed(&a, &b),
                    Some(x.cmp(&y)),
                    "cmp_signed({x}, {y}) at n={n} must equal the value order"
                );
            }
        }
    }
}

// --- ADR-040 §2.4 (CU-3): never-silent Binary↔Float conversions --------------------------------

/// **Oracle property test (round-trip):** exhaustively at small widths, `checked_uint_to_f64` ∘
/// `uint_to_bits` agrees with the plain `f64` cast oracle for every in-range unsigned magnitude —
/// every such magnitude is far below `FLOAT_EXACT_MAX`, so the conversion never refuses here.
#[test]
fn checked_uint_to_f64_matches_the_oracle_and_round_trips() {
    for n in 0u32..=16 {
        let hi: u64 = if n == 0 { 0 } else { (1u64 << n) - 1 };
        for v in 0..=hi {
            let bits = uint_to_bits(v, n).unwrap();
            assert_eq!(
                checked_uint_to_f64(&bits),
                Some(v as f64),
                "checked_uint_to_f64({v}) at n={n} must match the f64 cast oracle"
            );
        }
    }
}

/// The exact-integer boundary: `2^53` round-trips exactly; `2^53 + 1` refuses (never a silent
/// lossy round — ADR-040 §2.4).
#[test]
fn checked_uint_to_f64_exact_boundary() {
    let at_bound = uint_to_bits(FLOAT_EXACT_MAX, 54).unwrap();
    assert_eq!(checked_uint_to_f64(&at_bound), Some(FLOAT_EXACT_MAX as f64));

    let past_bound = uint_to_bits(FLOAT_EXACT_MAX + 1, 54).unwrap();
    assert_eq!(
        checked_uint_to_f64(&past_bound),
        None,
        "2^53 + 1 exceeds binary64 exact-integer representability — must refuse, never round"
    );
}

/// An over-cap width (`> FLOAT_CONV_MAX_WIDTH`) refuses regardless of the encoded magnitude —
/// never a silent truncation of the width check itself.
#[test]
fn checked_uint_to_f64_rejects_over_cap_width() {
    let bits = vec![false; FLOAT_CONV_MAX_WIDTH + 1];
    assert_eq!(checked_uint_to_f64(&bits), None);
}

/// **Oracle property test (round-trip):** every non-negative integer magnitude at small target
/// widths converts to `Binary{M}` and back to the identical `f64`/magnitude pair via
/// `checked_f64_to_uint` — the never-silent `flt→bin` direction's total domain.
#[test]
fn checked_f64_to_uint_matches_the_oracle_and_round_trips() {
    for m in 0u32..=16 {
        let hi: u64 = if m == 0 { 0 } else { (1u64 << m) - 1 };
        for v in 0..=hi {
            assert_eq!(
                checked_f64_to_uint(v as f64, m),
                Some(v),
                "checked_f64_to_uint({v}) at m={m} must match the oracle"
            );
        }
        // One past the target width's range refuses (never a silent truncation).
        assert_eq!(
            checked_f64_to_uint((hi + 1) as f64, m),
            None,
            "{}(+1) does not fit Binary{{{m}}} — must refuse",
            hi + 1
        );
    }
}

/// Never-silent domain refusals: NaN, ±inf, a negative value, and a nonzero fractional part all
/// refuse — never a silent coercion/truncation (ADR-040 §2.4; G2).
#[test]
fn checked_f64_to_uint_refuses_the_never_silent_domain() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 1.5, -0.5] {
        assert_eq!(
            checked_f64_to_uint(bad, 8),
            None,
            "checked_f64_to_uint({bad}) must refuse (NaN/±inf/negative/fractional)"
        );
    }
    // `-0.0` is the one signed-zero edge that legitimately succeeds: it equals the integer 0.
    assert_eq!(checked_f64_to_uint(-0.0, 8), Some(0));
}

/// The exact-integer boundary mirrors [`checked_uint_to_f64_exact_boundary`]: `2^53` succeeds (at
/// a wide-enough target width); `2^54` (the next exactly-representable power of two beyond the
/// bound — `2^53 + 1` is not itself an expressible `f64` literal, since binary64 cannot represent
/// odd integers past `2^53`) refuses.
#[test]
fn checked_f64_to_uint_exact_boundary() {
    assert_eq!(
        checked_f64_to_uint(FLOAT_EXACT_MAX as f64, 54),
        Some(FLOAT_EXACT_MAX)
    );
    assert_eq!(
        checked_f64_to_uint((FLOAT_EXACT_MAX as f64) * 2.0, 64),
        None,
        "2^54 exceeds binary64 exact-integer representability (> 2^53) — must refuse"
    );
}

/// An over-cap target width (`> FLOAT_CONV_MAX_WIDTH`) refuses regardless of the value.
#[test]
fn checked_f64_to_uint_rejects_over_cap_width() {
    assert_eq!(
        checked_f64_to_uint(1.0, (FLOAT_CONV_MAX_WIDTH + 1) as u32),
        None
    );
}

// --- RFC-0034 §10 (CU-5): the executable `wrapping` construct — modular two's-complement -------

/// **Oracle property test:** exhaustively at small widths, `wrapping_add`/`wrapping_sub` equal
/// the exact integer sum/difference reduced modulo `2^n` into `B_n` (never `None` on range) — the
/// modular-reduction oracle any two's-complement wraparound implementation must match.
#[test]
fn wrapping_add_sub_match_the_modular_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        let modulus = 1i128 << n;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();

                let sum = i128::from(x) + i128::from(y);
                let wrapped_sum = sum.rem_euclid(modulus)
                    - if sum.rem_euclid(modulus) >= (modulus / 2) {
                        modulus
                    } else {
                        0
                    };
                assert_eq!(
                    wrapping_add(&a, &b),
                    int_to_bits(wrapped_sum as i64, n),
                    "wrapping_add {x}+{y} at n={n} must equal the modular reduction"
                );

                let diff = i128::from(x) - i128::from(y);
                let wrapped_diff = diff.rem_euclid(modulus)
                    - if diff.rem_euclid(modulus) >= (modulus / 2) {
                        modulus
                    } else {
                        0
                    };
                assert_eq!(
                    wrapping_sub(&a, &b),
                    int_to_bits(wrapped_diff as i64, n),
                    "wrapping_sub {x}-{y} at n={n} must equal the modular reduction"
                );
            }
        }
    }
}

/// **Oracle property test:** `wrapping_mul` equals the exact product reduced modulo `2^n`.
#[test]
fn wrapping_mul_matches_the_modular_oracle() {
    for n in 1u32..=8 {
        let lo = -(1i64 << (n - 1));
        let hi = (1i64 << (n - 1)) - 1;
        let modulus = 1i128 << n;
        for x in lo..=hi {
            for y in lo..=hi {
                let a = int_to_bits(x, n).unwrap();
                let b = int_to_bits(y, n).unwrap();
                let product = i128::from(x) * i128::from(y);
                let m = product.rem_euclid(modulus);
                let wrapped = m - if m >= (modulus / 2) { modulus } else { 0 };
                assert_eq!(
                    wrapping_mul(&a, &b),
                    int_to_bits(wrapped as i64, n),
                    "wrapping_mul {x}*{y} at n={n} must equal the modular reduction"
                );
            }
        }
    }
}

/// Worked examples pinning the genuine wraparound: `add`/`sub`/`mul` refuse exactly where
/// `wrapping_add`/`wrapping_sub`/`wrapping_mul` instead produce the modular result — the CU-5
/// contrast that makes `wrapping` meaningfully different from the non-wrapping ops.
#[test]
fn wrapping_worked_examples_where_the_non_wrapping_ops_refuse() {
    // B_8 = [-128, 127]. 127 + 1 = 128, out of range: `add` refuses, `wrapping_add` wraps to -128.
    let a = int_to_bits(127, 8).unwrap();
    let b = int_to_bits(1, 8).unwrap();
    assert_eq!(add(&a, &b), None, "127 + 1 must refuse (non-wrapping)");
    assert_eq!(
        wrapping_add(&a, &b),
        int_to_bits(-128, 8),
        "127 + 1 must wrap to -128 (RFC-0034 §10)"
    );

    // -128 - 1 = -129, out of range: `sub` refuses, `wrapping_sub` wraps to 127.
    let lo = int_to_bits(-128, 8).unwrap();
    let one = int_to_bits(1, 8).unwrap();
    assert_eq!(sub(&lo, &one), None, "-128 - 1 must refuse (non-wrapping)");
    assert_eq!(
        wrapping_sub(&lo, &one),
        int_to_bits(127, 8),
        "-128 - 1 must wrap to 127 (RFC-0034 §10)"
    );

    // 16 * 16 = 256, out of range at n=8 (B_8 max is 127): `mul` refuses, `wrapping_mul` wraps to 0.
    let sixteen = int_to_bits(16, 8).unwrap();
    assert_eq!(
        mul(&sixteen, &sixteen),
        None,
        "16 * 16 must refuse (non-wrapping)"
    );
    assert_eq!(
        wrapping_mul(&sixteen, &sixteen),
        int_to_bits(0, 8),
        "16 * 16 = 256 = 2^8 must wrap to 0 (RFC-0034 §10)"
    );
}

/// Structural mismatches (unequal widths, an over-cap width) still refuse — `wrapping` only opts
/// out of the *range* refusal, never the shape contract (G2).
#[test]
fn wrapping_rejects_structural_mismatches() {
    let a4 = int_to_bits(1, 4).unwrap();
    let b8 = int_to_bits(1, 8).unwrap();
    assert_eq!(wrapping_add(&a4, &b8), None, "unequal widths must refuse");
    assert_eq!(wrapping_sub(&a4, &b8), None, "unequal widths must refuse");
    assert_eq!(wrapping_mul(&a4, &b8), None, "unequal widths must refuse");

    let over_cap = vec![false; TC_MAX_WIDTH + 1];
    assert_eq!(
        wrapping_add(&over_cap, &over_cap),
        None,
        "an over-cap width must refuse"
    );
    let over_mul_cap = vec![false; MUL_MAX_WIDTH + 1];
    assert_eq!(
        wrapping_mul(&over_mul_cap, &over_mul_cap),
        None,
        "an over-cap width must refuse"
    );
}
