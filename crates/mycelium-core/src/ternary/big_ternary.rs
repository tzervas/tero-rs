//! Arbitrary-width balanced-ternary **integer** arithmetic (E20-1 / M-756; RFC-0033 §4.2; ADR-029).
//!
//! # Why this file exists
//! The fixed-width path in [`super`] (M-111) is `i64`-internal and **already never-silent** about its
//! ~40-trit cap ([`super::max_magnitude`] returns `None` at `m ≥ 41`; [`super::add`]/[`super::mul`]
//! return `None` on overflow). This module **removes the cap** by adding a growable representation that
//! *grows a new trit* instead of returning `None` — the bignum need the fixed-width comment
//! anticipated. It is **not** a bug-fix in Mycelium's code: the silent-overflow defect that motivates
//! an arbitrary-width path is `embeddonator`'s `dimensional::Tryte::max_value` (a different upstream
//! codebase, on the do-not-lift list), never `core::ternary`.
//!
//! # Design (KC-3: TRUSTED)
//! [`BigTernary`] is a digit-serial `Vec<Trit>` (least-significant-first, canonicalized) — the
//! obviously-correct, never-overflowing reference. A limbed/packed perf path (`PackedTernary`,
//! ≥40 trits/u64) is an explicit YAGNI follow-on (M-758) gated on a benchmark and, if added, MUST be
//! differentially proven bit-exact against this reference (RFC-0033 §4.2.2).
//!
//! # Never-silent boundary (G2)
//! [`BigTernary`] arithmetic NEVER overflows (the carry out of the top digit becomes a new digit). The
//! boundary is the **fixed-width** [`FixedWidthTrits`] (the in-memory image of `Repr::Ternary{N}`):
//! [`BigTernary::checked_to_width`] and [`checked_add_fixed`] return `Option` and yield `None` exactly
//! when the true result needs more than `N` trits — never a wrap or truncation. [`BigTernary::to_i128`]
//! is likewise `Option` (overflow-checked).
//!
//! # Guarantee lattice
//! Every operation here is **Exact** (closed integer arithmetic; the balanced-ternary digit algebra is
//! an exact integer identity, Knuth 4.1 / `docs/spec/swaps/binary-ternary.md` §1). The binary↔ternary
//! swap is `LosslessWithinRange` — lossless for the growable path, range-bounded for fixed width
//! (RFC-0033 §6.1).
//!
//! # Endianness
//! [`BigTernary`] is **least-significant-first** (index 0 least significant); the fixed-width [`super`]
//! codec/arithmetic is **most-significant-first**. The two are reconciled **only through the integer
//! value** ([`BigTernary::to_i128`] / [`super::trits_to_int`]), never by comparing trit vectors.

use super::{add_with_carry, digit, is_nonzero, is_zero, neg_trit};
use crate::value::Trit;

/// Arbitrary-width balanced-ternary integer (digit-serial reference form).
///
/// Invariant (canonical form): `digits` has no trailing (most-significant) `Zero` trits, EXCEPT that
/// zero is the empty vector. Enforced by [`BigTernary::canonicalize`] after every constructor/op, so
/// each integer has exactly **one** representation (non-redundant ⇒ content-addressing is well-defined;
/// RFC-0033 §4.2.4).
#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct BigTernary {
    /// Balanced trits, index 0 least significant. Canonical: no trailing `Zero`.
    digits: Vec<Trit>,
}

impl BigTernary {
    /// The additive identity (empty digit vector).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        BigTernary { digits: Vec::new() }
    }

    /// `true` iff this is exactly zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.digits.is_empty()
    }

    /// Number of significant trits (0 for zero).
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.digits.len()
    }

    /// Borrow the canonical digit slice (least-significant-first).
    #[inline]
    #[must_use]
    pub fn digits(&self) -> &[Trit] {
        &self.digits
    }

    /// Build from raw least-significant-first trits (any non-canonical input is accepted and
    /// canonicalized). Total — there is no invalid `Trit`, so this never fails.
    pub fn from_trits_lsf(trits: impl IntoIterator<Item = Trit>) -> Self {
        let mut b = BigTernary {
            digits: trits.into_iter().collect(),
        };
        b.canonicalize();
        b
    }

    /// Drop trailing (most-significant) `Zero` trits; zero becomes empty.
    fn canonicalize(&mut self) {
        while matches!(self.digits.last(), Some(Trit::Zero)) {
            self.digits.pop();
        }
    }

    /// Negate (flip every trit). Canonical form is preserved.
    #[must_use]
    pub fn neg(&self) -> Self {
        BigTernary {
            digits: self.digits.iter().map(|&t| neg_trit(t)).collect(),
        }
    }

    /// Addition. NEVER overflows — the final carry becomes a new digit. Digit-serial ripple of the
    /// shared [`super::add_with_carry`]; `O(max(width))`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.digits.len().max(other.digits.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = Trit::Zero;
        for i in 0..n {
            let a = self.digits.get(i).copied().unwrap_or(Trit::Zero);
            let b = other.digits.get(i).copied().unwrap_or(Trit::Zero);
            let (sum, c) = add_with_carry(a, b, carry);
            out.push(sum);
            carry = c;
        }
        if is_nonzero(carry) {
            out.push(carry);
        }
        let mut r = BigTernary { digits: out };
        r.canonicalize();
        r
    }

    /// Subtraction: `self + (−other)`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Multiplication (schoolbook over balanced trits): each `b_i ∈ {−1, 0, +1}`, so the partial
    /// product is `±self` shifted left by `i`, accumulated with [`add`](Self::add).
    /// `O(width(self) · width(other))`. A Karatsuba/Toom fast path is a YAGNI follow-on (M-759),
    /// equivalence-tested against this if added.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut acc = BigTernary::zero();
        for (i, &b) in other.digits.iter().enumerate() {
            if is_zero(b) {
                continue;
            }
            // partial = (±self) << i  (i leading zero trits, then the signed digits)
            let mut shifted = vec![Trit::Zero; i];
            let signed = if matches!(b, Trit::Neg) {
                self.neg()
            } else {
                self.clone()
            };
            shifted.extend_from_slice(&signed.digits);
            acc = acc.add(&BigTernary { digits: shifted });
        }
        acc.canonicalize();
        acc
    }

    // ---- bridges to/from machine integers (never-silent) ----

    /// Exact construction from `i128`.
    #[must_use]
    pub fn from_i128(mut value: i128) -> Self {
        let mut digits = Vec::new();
        while value != 0 {
            // Balanced residue in {−1, 0, +1}: `rem_euclid` gives 0,1,2; map 2 → −1. `value - rem` is
            // divisible by 3 (rem ≡ value mod 3), so the quotient is exact and applies the borrow.
            let m = value.rem_euclid(3);
            let rem: i128 = if m == 2 { -1 } else { m };
            value = (value - rem) / 3;
            digits.push(match rem {
                -1 => Trit::Neg,
                0 => Trit::Zero,
                1 => Trit::Pos,
                _ => unreachable!("balanced residue is in {{-1, 0, 1}}"),
            });
        }
        let mut b = BigTernary { digits };
        b.canonicalize();
        b
    }

    /// NEVER-SILENT conversion to `i128`: `None` if the value does not fit (overflow-checked Horner).
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        let mut acc: i128 = 0;
        let mut pow: i128 = 1;
        for (i, &t) in self.digits.iter().enumerate() {
            let term = i128::from(digit(t)).checked_mul(pow)?;
            acc = acc.checked_add(term)?;
            if i + 1 < self.digits.len() {
                pow = pow.checked_mul(3)?;
            }
        }
        Some(acc)
    }

    // ---- the fixed-width / never-silent boundary ----

    /// NEVER-SILENT narrowing to a fixed width of `n` trits: `Some` iff `width() ≤ n`; `None`
    /// otherwise. The single honest definition of "out of range" for `Repr::Ternary{trits:n}`.
    #[must_use]
    pub fn checked_to_width(&self, n: u32) -> Option<FixedWidthTrits> {
        if self.width() > n as usize {
            return None;
        }
        let mut digits = self.digits.clone();
        digits.resize(n as usize, Trit::Zero);
        Some(FixedWidthTrits { trits: digits })
    }
}

/// A balanced-ternary value pinned to exactly `trits.len()` trits — the in-memory image of
/// `Repr::Ternary{trits:N}`. Padding trits are `Zero`. Arithmetic that could overflow the width is
/// never-silent (see [`checked_add_fixed`]).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FixedWidthTrits {
    /// Exactly `N` trits, least-significant-first, `Zero`-padded.
    pub trits: Vec<Trit>,
}

impl FixedWidthTrits {
    /// Promote to the growable form (always exact).
    #[must_use]
    pub fn to_big(&self) -> BigTernary {
        BigTernary::from_trits_lsf(self.trits.iter().copied())
    }
}

/// NEVER-SILENT fixed-width addition: ripples the shared [`super::add_with_carry`] across `n` trits and
/// returns `None` iff the carry out of the top trit is non-zero (the true sum needs trit `n+1`). No
/// wrap, no truncation. Both inputs MUST be the same width (debug-asserted; a mismatch is a caller bug,
/// not a runtime value condition).
#[must_use]
pub fn checked_add_fixed(a: &FixedWidthTrits, b: &FixedWidthTrits) -> Option<FixedWidthTrits> {
    debug_assert_eq!(
        a.trits.len(),
        b.trits.len(),
        "width mismatch is a caller bug"
    );
    let n = a.trits.len();
    let mut out = Vec::with_capacity(n);
    let mut carry = Trit::Zero;
    for i in 0..n {
        let (sum, c) = add_with_carry(a.trits[i], b.trits[i], carry);
        out.push(sum);
        carry = c;
    }
    if is_nonzero(carry) {
        None // overflow — explicit, never silent
    } else {
        Some(FixedWidthTrits { trits: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bt(v: i128) -> BigTernary {
        BigTernary::from_i128(v)
    }

    #[test]
    fn roundtrip_i128() {
        for v in [-1_000_000i128, -42, -1, 0, 1, 13, 14, 364, 365, 9_999_999] {
            assert_eq!(bt(v).to_i128(), Some(v), "roundtrip {v}");
        }
    }

    #[test]
    fn add_matches_integer() {
        for a in [-50i128, -1, 0, 7, 121] {
            for b in [-121i128, -7, 0, 1, 50] {
                assert_eq!(bt(a).add(&bt(b)).to_i128(), Some(a + b), "{a}+{b}");
            }
        }
    }

    #[test]
    fn sub_matches_integer() {
        for a in [-50i128, -1, 0, 7, 121] {
            for b in [-121i128, -7, 0, 1, 50] {
                assert_eq!(bt(a).sub(&bt(b)).to_i128(), Some(a - b), "{a}-{b}");
            }
        }
    }

    #[test]
    fn mul_matches_integer() {
        for a in [-40i128, -3, 0, 1, 27] {
            for b in [-27i128, -1, 0, 3, 40] {
                assert_eq!(bt(a).mul(&bt(b)).to_i128(), Some(a * b), "{a}*{b}");
            }
        }
    }

    #[test]
    fn negation_round_trips() {
        for v in [-9_999i128, -7, -1, 0, 1, 42, 365] {
            assert_eq!(bt(v).neg().to_i128(), Some(-v), "neg {v}");
            assert_eq!(bt(v).neg().neg(), bt(v), "double-neg {v}");
        }
    }

    #[test]
    fn zero_is_canonical_empty() {
        assert!(BigTernary::zero().is_zero());
        assert_eq!(BigTernary::zero().width(), 0);
        assert_eq!(bt(0), BigTernary::zero());
        assert_eq!(bt(0).to_i128(), Some(0));
        // 1 + (−1) canonicalizes back to the empty zero, not a padded form.
        assert_eq!(bt(1).add(&bt(-1)), BigTernary::zero());
    }

    #[test]
    fn beyond_40_trits_is_exact_not_silent() {
        // 3^41 exceeds the fixed-width i64 path's range (which is never-silent there); BigTernary
        // grows to width 42 and stays exact. The headline "removes the cap" witness.
        let mut x = BigTernary::from_i128(1);
        let three = BigTernary::from_i128(3);
        for _ in 0..41 {
            x = x.mul(&three);
        }
        assert_eq!(x.width(), 42);
        assert_eq!(x.to_i128(), Some(3i128.pow(41)));
    }

    #[test]
    fn fixed_width_overflow_is_none() {
        // width 3 holds [−13, 13]. 13 + 1 = 14 overflows → None.
        let a = bt(13).checked_to_width(3).unwrap();
        let one = bt(1).checked_to_width(3).unwrap();
        assert_eq!(checked_add_fixed(&a, &one), None);
        // 6 + 6 = 12 still fits width 3.
        let six = bt(6).checked_to_width(3).unwrap();
        let r = checked_add_fixed(&six, &six).unwrap();
        assert_eq!(r.to_big().to_i128(), Some(12));
    }

    #[test]
    fn narrowing_is_never_silent() {
        assert!(bt(13).checked_to_width(3).is_some());
        assert!(bt(14).checked_to_width(3).is_none()); // 14 needs 4 trits
    }

    /// Cross-check the growable type against the EXISTING fixed-width [`super::super::add`] within
    /// range — two independent implementations must agree. Bridged **only through the integer value**
    /// (most-significant-first `add` vs least-significant-first `BigTernary`), never by comparing trit
    /// vectors directly — the honest way to reconcile the two endiannesses.
    #[test]
    fn big_agrees_with_fixed_width_add_in_range() {
        use super::super::{add, int_to_trits, max_magnitude, trits_to_int};
        let m = 4u32;
        let max = max_magnitude(m).unwrap();
        for x in -max..=max {
            for y in -max..=max {
                let big = BigTernary::from_i128(i128::from(x + y));
                let a = int_to_trits(x, m).unwrap();
                let b = int_to_trits(y, m).unwrap();
                match add(&a, &b) {
                    // in range ⇒ both agree on the integer value
                    Some(s) => {
                        assert_eq!(big.to_i128(), Some(i128::from(trits_to_int(&s))), "{x}+{y}");
                    }
                    // fixed-width overflow ⇒ the value needs > m trits ⇒ BigTernary is wider
                    None => assert!(big.width() > m as usize, "{x}+{y} should exceed width {m}"),
                }
            }
        }
    }
}
