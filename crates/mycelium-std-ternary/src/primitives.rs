//! First-class `Bit` and balanced `Trit{−1,0,+1}` primitives (FR-M2; M-111).
//!
//! These are the digit types at the top of the Ring-1 capability surface. Every constructor that
//! can fail returns `Option` with an explicit error — never a sentinel, silent clamp, or re-round
//! (C1/G2). Every op is `Exact` (C2): the balanced-ternary digit algebra is an exact integer
//! identity (Knuth 4.1; `docs/spec/swaps/binary-ternary.md` §1; M-111).
//!
//! **FLAG (Q1):** The spec (`docs/spec/stdlib/ternary.md` §7-Q1) leaves the `Bit`/`Trit` spelling
//! as a FLAGGED open question pending the DN-02/06 lexicon decision. We default to the M-111
//! kernel spellings (`Trit`, `digit`, `neg`) here.

use mycelium_core::Trit as CoreTrit;

// ── Trit ─────────────────────────────────────────────────────────────────────

/// A balanced trit in `{−1, 0, +1}` (FR-M2; M-111).
///
/// Constructors are total on their domain and explicit-`None` off it (C1/G2). All ops are
/// `Exact`: the balanced-ternary digit algebra is an exact integer identity (C2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    /// −1.
    Neg,
    /// 0.
    Zero,
    /// +1.
    Pos,
}

impl Trit {
    /// Construct a `Trit` from an integer.
    ///
    /// **Guarantee: `Exact`.** Returns `None` if `d ∉ {−1, 0, +1}` — an explicit
    /// off-domain error, never a silent clamp (C1/G2). (Mutant witness: changing the guard to
    /// `d < -2` would let `d = -2` through, breaking `None` on an off-domain input.)
    #[must_use]
    pub fn new(d: i64) -> Option<Trit> {
        match d {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }

    /// The signed integer value of this trit: `Neg↦−1, Zero↦0, Pos↦+1`.
    ///
    /// **Guarantee: `Exact`.** Total — every trit has a unique integer value (C2).
    #[must_use]
    pub fn digit(self) -> i64 {
        match self {
            Trit::Neg => -1,
            Trit::Zero => 0,
            Trit::Pos => 1,
        }
    }

    /// Convert to the `mycelium-core` kernel `Trit` (for passing to the arithmetic kernel).
    #[must_use]
    pub(crate) fn to_core(self) -> CoreTrit {
        match self {
            Trit::Neg => CoreTrit::Neg,
            Trit::Zero => CoreTrit::Zero,
            Trit::Pos => CoreTrit::Pos,
        }
    }

    /// Convert from the `mycelium-core` kernel `Trit`.
    #[must_use]
    pub(crate) fn from_core(t: CoreTrit) -> Trit {
        match t {
            CoreTrit::Neg => Trit::Neg,
            CoreTrit::Zero => Trit::Zero,
            CoreTrit::Pos => Trit::Pos,
        }
    }

    /// The MSB-first wire glyph for this trit: `-` / `0` / `+`
    /// (`docs/spec/swaps/binary-ternary.md` §1).
    #[must_use]
    pub fn to_wire_char(self) -> char {
        match self {
            Trit::Neg => '-',
            Trit::Zero => '0',
            Trit::Pos => '+',
        }
    }

    /// Parse a wire glyph back into a `Trit`.
    ///
    /// **Guarantee: `Exact`.** Returns `None` for any character outside `{'-', '0', '+'}` (C1).
    #[must_use]
    pub fn from_wire_char(c: char) -> Option<Trit> {
        match c {
            '-' => Some(Trit::Neg),
            '0' => Some(Trit::Zero),
            '+' => Some(Trit::Pos),
            _ => None,
        }
    }
}

// ── Bit ──────────────────────────────────────────────────────────────────────

/// A binary digit in `{0, 1}` (FR-M2).
///
/// Constructors are total on their domain and explicit-`None` off it (C1/G2). All ops are
/// `Exact`: Boolean algebra is an exact computation (C2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bit {
    /// 0.
    Zero,
    /// 1.
    One,
}

impl Bit {
    /// Construct a `Bit` from an integer.
    ///
    /// **Guarantee: `Exact`.** Returns `None` if `d ∉ {0, 1}` — explicit off-domain error (C1/G2).
    /// (Mutant witness: changing the guard to `d < 0` would let `d = 2` through.)
    #[must_use]
    pub fn new(d: i64) -> Option<Bit> {
        match d {
            0 => Some(Bit::Zero),
            1 => Some(Bit::One),
            _ => None,
        }
    }

    /// The unsigned integer value of this bit: `Zero↦0, One↦1`.
    ///
    /// **Guarantee: `Exact`.** Total (C2).
    #[must_use]
    pub fn digit(self) -> i64 {
        match self {
            Bit::Zero => 0,
            Bit::One => 1,
        }
    }

    /// Boolean AND.
    ///
    /// **Guarantee: `Exact`.** Total Boolean algebra (C2).
    #[must_use]
    pub fn and(self, other: Bit) -> Bit {
        match (self, other) {
            (Bit::One, Bit::One) => Bit::One,
            _ => Bit::Zero,
        }
    }

    /// Boolean OR.
    ///
    /// **Guarantee: `Exact`.** Total Boolean algebra (C2).
    #[must_use]
    pub fn or(self, other: Bit) -> Bit {
        match (self, other) {
            (Bit::Zero, Bit::Zero) => Bit::Zero,
            _ => Bit::One,
        }
    }

    /// Boolean XOR.
    ///
    /// **Guarantee: `Exact`.** Total Boolean algebra (C2).
    #[must_use]
    pub fn xor(self, other: Bit) -> Bit {
        if self == other {
            Bit::Zero
        } else {
            Bit::One
        }
    }
}

// ── std::ops::Neg ────────────────────────────────────────────────────────────

impl core::ops::Neg for Trit {
    type Output = Trit;

    /// Digit-wise negation: `value(−t) = −value(t)` exactly.
    ///
    /// **Guarantee: `Exact`.** Total — the balanced-ternary range is symmetric, so `−(+1) = −1`
    /// is in-range (no two's-complement asymmetry; `docs/spec/swaps/binary-ternary.md` §1; C2).
    fn neg(self) -> Trit {
        match self {
            Trit::Neg => Trit::Pos,
            Trit::Zero => Trit::Zero,
            Trit::Pos => Trit::Neg,
        }
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl core::fmt::Display for Trit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_wire_char())
    }
}

impl core::fmt::Display for Bit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.digit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Trit::new ─────────────────────────────────────────────────────────────

    #[test]
    fn trit_new_accepts_domain() {
        assert_eq!(Trit::new(-1), Some(Trit::Neg));
        assert_eq!(Trit::new(0), Some(Trit::Zero));
        assert_eq!(Trit::new(1), Some(Trit::Pos));
    }

    #[test]
    fn trit_new_rejects_off_domain() {
        // Mutant witness: if the guard were `d < -2`, `d = -2` would slip through.
        for &bad in &[-2i64, 2, 3, -100, 100, i64::MIN, i64::MAX] {
            assert_eq!(
                Trit::new(bad),
                None,
                "expected None for off-domain input {bad}"
            );
        }
    }

    // ── Trit::digit ───────────────────────────────────────────────────────────

    #[test]
    fn trit_digit_is_exact_total() {
        assert_eq!(Trit::Neg.digit(), -1);
        assert_eq!(Trit::Zero.digit(), 0);
        assert_eq!(Trit::Pos.digit(), 1);
    }

    // ── Trit::neg ─────────────────────────────────────────────────────────────

    #[test]
    fn trit_neg_is_exact_sign_flip() {
        // negation = digit-wise sign flip (docs/spec/swaps/binary-ternary.md §1)
        assert_eq!(-Trit::Pos, Trit::Neg);
        assert_eq!(-Trit::Neg, Trit::Pos);
        assert_eq!(-Trit::Zero, Trit::Zero);
    }

    #[test]
    fn trit_neg_is_involution() {
        // property: neg(neg(t)) == t for every trit (negation is its own inverse)
        // Mutant witness: if neg(Pos) returned Pos, this would fail for Pos.
        for t in [Trit::Neg, Trit::Zero, Trit::Pos] {
            assert_eq!(-(-t), t, "involution on {t:?}");
        }
    }

    #[test]
    fn trit_neg_digit_equals_arithmetic_neg() {
        // property: digit(neg t) == -digit(t) — verified exhaustively over all 3 trits.
        for t in [Trit::Neg, Trit::Zero, Trit::Pos] {
            assert_eq!((-t).digit(), -t.digit(), "neg.digit == -digit for {t:?}");
        }
    }

    // ── Trit wire chars ───────────────────────────────────────────────────────

    #[test]
    fn trit_wire_char_round_trips() {
        for t in [Trit::Neg, Trit::Zero, Trit::Pos] {
            let c = t.to_wire_char();
            assert_eq!(Trit::from_wire_char(c), Some(t));
        }
    }

    #[test]
    fn trit_from_wire_char_rejects_non_trit() {
        // Mutant witness: removing the `_ => None` arm would cause any char to be accepted.
        for &bad in &['a', '1', '2', ' ', '\n'] {
            assert_eq!(
                Trit::from_wire_char(bad),
                None,
                "expected None for non-trit char {bad:?}"
            );
        }
    }

    // ── Bit::new ──────────────────────────────────────────────────────────────

    #[test]
    fn bit_new_accepts_domain() {
        assert_eq!(Bit::new(0), Some(Bit::Zero));
        assert_eq!(Bit::new(1), Some(Bit::One));
    }

    #[test]
    fn bit_new_rejects_off_domain() {
        // Mutant witness: if guard were `d < 0`, `d = 2` would slip through.
        for &bad in &[-1i64, 2, 3, -100, 100, i64::MIN, i64::MAX] {
            assert_eq!(
                Bit::new(bad),
                None,
                "expected None for off-domain input {bad}"
            );
        }
    }

    // ── Boolean algebra ───────────────────────────────────────────────────────

    #[test]
    fn bit_and_truth_table() {
        use Bit::{One, Zero};
        assert_eq!(Zero.and(Zero), Zero);
        assert_eq!(Zero.and(One), Zero);
        assert_eq!(One.and(Zero), Zero);
        assert_eq!(One.and(One), One);
    }

    #[test]
    fn bit_or_truth_table() {
        use Bit::{One, Zero};
        assert_eq!(Zero.or(Zero), Zero);
        assert_eq!(Zero.or(One), One);
        assert_eq!(One.or(Zero), One);
        assert_eq!(One.or(One), One);
    }

    #[test]
    fn bit_xor_truth_table() {
        use Bit::{One, Zero};
        assert_eq!(Zero.xor(Zero), Zero);
        assert_eq!(Zero.xor(One), One);
        assert_eq!(One.xor(Zero), One);
        assert_eq!(One.xor(One), Zero);
    }

    #[test]
    fn bit_boolean_laws() {
        // De Morgan: not(a and b) == not(a) or not(b) — using xor-with-One as NOT.
        use Bit::{One, Zero};
        let not = |b: Bit| b.xor(One);
        for a in [Zero, One] {
            for b in [Zero, One] {
                // De Morgan AND: !(a & b) == !a | !b
                assert_eq!(
                    not(a.and(b)),
                    not(a).or(not(b)),
                    "De Morgan AND {a:?},{b:?}"
                );
                // De Morgan OR: !(a | b) == !a & !b
                assert_eq!(not(a.or(b)), not(a).and(not(b)), "De Morgan OR {a:?},{b:?}");
                // XOR == (a OR b) AND NOT(a AND b)
                assert_eq!(
                    a.xor(b),
                    a.or(b).and(not(a.and(b))),
                    "XOR identity {a:?},{b:?}"
                );
            }
        }
    }

    // ── Display ───────────────────────────────────────────────────────────────

    #[test]
    fn trit_display_matches_wire_char() {
        for t in [Trit::Neg, Trit::Zero, Trit::Pos] {
            assert_eq!(t.to_string(), t.to_wire_char().to_string());
        }
    }

    #[test]
    fn bit_display_matches_digit() {
        for b in [Bit::Zero, Bit::One] {
            assert_eq!(b.to_string(), b.digit().to_string());
        }
    }

    // ── core round-trip ───────────────────────────────────────────────────────

    #[test]
    fn trit_core_round_trip() {
        for t in [Trit::Neg, Trit::Zero, Trit::Pos] {
            assert_eq!(Trit::from_core(t.to_core()), t);
        }
    }
}
