//! The guarantee-strength lattice (RFC-0001 §3.4/§4.7; `guarantee.schema.json`).
//!
//! `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`, ordered strongest-to-weakest. The [`meet`]
//! (weakest-wins) composition is a meet-semilattice (commutative/associative/idempotent, identity
//! `Exact`); operations propagate guarantees by taking the meet of their inputs and their own
//! intrinsic strength (M-102; RFC-0001 §3.4/§4.7).
//!
//! [`meet`]: GuaranteeStrength::meet

/// How trustworthy a value's representation/bound is. Honesty is monotone-downward: an operation's
/// result is never stronger than its weakest input (the [`meet`](GuaranteeStrength::meet)).
///
/// The `serde` form is the bare string `"Exact"|"Proven"|"Empirical"|"Declared"`
/// (`guarantee.schema.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GuaranteeStrength {
    /// No approximation; `bound == None` (M-I1).
    Exact,
    /// Approximate with a machine-checked bound (basis `ProvenThm`).
    Proven,
    /// Approximate with an empirically-validated bound (basis `EmpiricalFit`).
    Empirical,
    /// Approximate with a user-asserted, unvalidated bound (basis `UserDeclared`); always flagged.
    Declared,
}

impl GuaranteeStrength {
    /// The strongest strength — the identity of [`meet`](Self::meet) and the unit of
    /// [`meet_all`](Self::meet_all): `meet(Exact, g) == g` for all `g`.
    pub const TOP: GuaranteeStrength = GuaranteeStrength::Exact;

    /// All four strengths, strongest-to-weakest — for exhaustive iteration in tests and tooling.
    pub const ALL: [GuaranteeStrength; 4] = [
        GuaranteeStrength::Exact,
        GuaranteeStrength::Proven,
        GuaranteeStrength::Empirical,
        GuaranteeStrength::Declared,
    ];

    /// Lattice rank, `0` = strongest (`Exact`) … `3` = weakest (`Declared`). The [`meet`](Self::meet)
    /// of two strengths is the one with the larger rank (the weakest).
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            GuaranteeStrength::Exact => 0,
            GuaranteeStrength::Proven => 1,
            GuaranteeStrength::Empirical => 2,
            GuaranteeStrength::Declared => 3,
        }
    }

    /// The lattice **meet** (greatest lower bound): the *weakest* of `self` and `other`
    /// (RFC-0001 §3.4/§4.7). Honesty is monotone-downward — a result can never be stronger than
    /// its weakest input — so composition takes the meet. The meet is the strength with the
    /// larger [`rank`](Self::rank); on a tie the operands are equal, so either is returned.
    ///
    /// Algebraically a meet-semilattice: **commutative**, **associative**, **idempotent**, with
    /// identity [`TOP`](Self::TOP) (`Exact`). These laws are verified by *exhaustion* over all
    /// 4×4(×4) tuples — the value space is finite, so this is a complete check, not sampling.
    #[must_use]
    pub fn meet(self, other: GuaranteeStrength) -> GuaranteeStrength {
        if self.rank() >= other.rank() {
            self
        } else {
            other
        }
    }

    /// Propagate guarantees through an operation (RFC-0001 §4.7):
    /// `guarantee(result) = meet(guarantee(v_1), …, guarantee(v_n), g_f)`, where `g_f` is the
    /// operation's own intrinsic strength. Folds [`meet`](Self::meet) over the inputs starting
    /// from `intrinsic`. With no inputs the result is `intrinsic` itself.
    ///
    /// This is the *only* sanctioned way to derive an operation's result strength — disclosure can
    /// only degrade, never spuriously upgrade (VR-3/VR-5).
    #[must_use]
    pub fn propagate(
        intrinsic: GuaranteeStrength,
        inputs: impl IntoIterator<Item = GuaranteeStrength>,
    ) -> GuaranteeStrength {
        inputs.into_iter().fold(intrinsic, GuaranteeStrength::meet)
    }

    /// The meet of a sequence of strengths, weakest-wins, starting from [`TOP`](Self::TOP)
    /// (`Exact`). An empty sequence yields `Exact` (the identity). Equivalent to
    /// [`propagate`](Self::propagate) with an `Exact` intrinsic.
    #[must_use]
    pub fn meet_all(strengths: impl IntoIterator<Item = GuaranteeStrength>) -> GuaranteeStrength {
        Self::propagate(GuaranteeStrength::TOP, strengths)
    }
}

#[cfg(test)]
mod tests {
    use super::GuaranteeStrength;
    use super::GuaranteeStrength::{Declared, Empirical, Exact, Proven};

    #[test]
    fn ranks_are_strongest_to_weakest() {
        assert!(Exact.rank() < Proven.rank());
        assert!(Proven.rank() < Empirical.rank());
        assert!(Empirical.rank() < Declared.rank());
    }

    /// The meet equals the weakest (larger-rank) operand — for every one of the 16 pairs.
    #[test]
    fn meet_is_weakest_for_all_pairs() {
        for a in GuaranteeStrength::ALL {
            for b in GuaranteeStrength::ALL {
                let m = a.meet(b);
                assert_eq!(m.rank(), a.rank().max(b.rank()), "meet({a:?},{b:?})");
                // The meet is one of its operands (it is a selection, never a fresh value).
                assert!(m == a || m == b);
            }
        }
    }

    /// **Commutativity**, exhaustively over all 4×4 pairs.
    #[test]
    fn meet_is_commutative() {
        for a in GuaranteeStrength::ALL {
            for b in GuaranteeStrength::ALL {
                assert_eq!(a.meet(b), b.meet(a), "commutativity {a:?},{b:?}");
            }
        }
    }

    /// **Idempotence**, all 4.
    #[test]
    fn meet_is_idempotent() {
        for a in GuaranteeStrength::ALL {
            assert_eq!(a.meet(a), a, "idempotence {a:?}");
        }
    }

    /// **Associativity**, exhaustively over all 4×4×4 triples.
    #[test]
    fn meet_is_associative() {
        for a in GuaranteeStrength::ALL {
            for b in GuaranteeStrength::ALL {
                for c in GuaranteeStrength::ALL {
                    assert_eq!(
                        a.meet(b).meet(c),
                        a.meet(b.meet(c)),
                        "associativity {a:?},{b:?},{c:?}"
                    );
                }
            }
        }
    }

    /// **Identity**: `Exact` (`TOP`) is the unit on both sides, for every strength.
    #[test]
    fn exact_is_the_identity() {
        assert_eq!(GuaranteeStrength::TOP, Exact);
        for a in GuaranteeStrength::ALL {
            assert_eq!(Exact.meet(a), a, "left identity {a:?}");
            assert_eq!(a.meet(Exact), a, "right identity {a:?}");
        }
    }

    /// `Declared` is the bottom: it absorbs everything (the weakest is always weakest).
    #[test]
    fn declared_is_absorbing() {
        for a in GuaranteeStrength::ALL {
            assert_eq!(Declared.meet(a), Declared, "{a:?}");
            assert_eq!(a.meet(Declared), Declared, "{a:?}");
        }
    }

    /// `propagate` realizes RFC-0001 §4.7: the result is the meet of all inputs and the intrinsic
    /// strength — an `Empirical` input drags an otherwise-`Exact` op down to `Empirical`.
    #[test]
    fn propagate_takes_the_meet_of_inputs_and_intrinsic() {
        // Exact op over [Exact, Empirical, Exact] → Empirical (worked example, §3.4).
        assert_eq!(
            GuaranteeStrength::propagate(Exact, [Exact, Empirical, Exact]),
            Empirical
        );
        // The op's own intrinsic strength also caps the result.
        assert_eq!(GuaranteeStrength::propagate(Proven, [Exact, Exact]), Proven);
        // No inputs → just the intrinsic.
        assert_eq!(GuaranteeStrength::propagate(Declared, []), Declared);
        // A single Declared input poisons a Proven op down to Declared.
        assert_eq!(
            GuaranteeStrength::propagate(Proven, [Exact, Declared, Empirical]),
            Declared
        );
    }

    /// `meet_all` is `propagate` from the `Exact` identity; empty → `Exact`.
    #[test]
    fn meet_all_folds_from_exact() {
        assert_eq!(GuaranteeStrength::meet_all([]), Exact);
        assert_eq!(GuaranteeStrength::meet_all([Proven, Empirical]), Empirical);
        assert_eq!(GuaranteeStrength::meet_all([Exact, Proven, Exact]), Proven);
    }
}
