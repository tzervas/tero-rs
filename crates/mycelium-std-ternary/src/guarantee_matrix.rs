//! The guarantee matrix for `std.ternary` (RFC-0016 §4.5; `docs/spec/stdlib/ternary.md` §4).
//!
//! The matrix is encoded as DATA — a table of [`OpGuarantee`] rows, one per exported op — and
//! asserted in tests. This is the load-bearing deliverable: **never prose-only** (RFC-0016 §4.5).
//!
//! Every row records:
//! - `op`: the operation name (matching the public API surface).
//! - `tag`: the guarantee tag on the `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` lattice (C2/VR-5).
//! - `fallibility`: whether the op is total or has an explicit error set.
//! - `effects`: declared effects (all `None` here — every op is pure; C6).
//! - `explainable`: whether the op exposes an inspectable artifact (C3; only the pack ops do).
//!
//! **Honesty justification (VR-5):** Every row tags `Exact` because:
//!
//! 1. The balanced-ternary algebra is an exact integer identity (Horner; digit-wise ops; Knuth 4.1;
//!    `docs/spec/swaps/binary-ternary.md` §1; M-111). No rounding, no ε, nothing to approximate.
//! 2. The packing codecs (I2S/TL1/TL2) are lossless re-encodings (DN-01 §2; RFC-0004 §5).
//!    `pack` then `unpack` is the identity on a well-formed input.
//!
//! The range boundary is handled by **fallibility** (the `None`/`Err` column), not by weakening
//! the tag. Downgrading to `Empirical`/`Declared` for these ops would be a dishonest downgrade
//! of an exact fact, which VR-5 forbids.

/// A guarantee-lattice tag (C2/VR-5). Mirrors `mycelium_core::GuaranteeStrength` without
/// re-importing that type in the guarantee matrix data (KC-3: the matrix is data, not a kernel
/// consumer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    /// No approximation; the result is bit-exact given the stated inputs.
    Exact,
    /// Approximate with a machine-checked bound (side-conditions verified in code).
    Proven,
    /// Approximate with an empirically-validated bound.
    Empirical,
    /// Approximate with a user-asserted, unvalidated bound.
    Declared,
}

/// Whether an op is total or returns an explicit error on some inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// Total on all well-formed inputs of the stated type.
    Total,
    /// Explicit `Option::None` when input is off-domain or out-of-range (C1/G2).
    NoneOnOffDomain(&'static str),
    /// Explicit `Err(E)` with the given error type and variants (C1/G2).
    ErrOn(&'static str),
}

/// Whether the op exposes an inspectable artifact for its selection/conversion (C3/G11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op selects, converts, or approximates nothing — C3 does not apply.
    NotApplicable,
    /// The op exposes an inspectable `Meta.physical` + EXPLAIN record (C3).
    Yes,
}

/// One row of the guarantee matrix (RFC-0016 §4.5; `docs/spec/stdlib/ternary.md` §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpGuarantee {
    /// The exported op name (matches the public API surface exactly).
    pub op: &'static str,
    /// The guarantee tag (C2/VR-5).
    pub tag: Tag,
    /// The fallibility — total or explicit error set (C1/G2).
    pub fallibility: Fallibility,
    /// Declared effects (RFC-0014); all `None` = pure (C6).
    pub effects: &'static str,
    /// Whether the op exposes an inspectable artifact (C3).
    pub explainable: Explainable,
}

/// The complete guarantee matrix for `std.ternary` (RFC-0016 §4.5).
///
/// Every row is `Exact` — see the module doc for the honesty justification (VR-5).
/// Asserted structurally in [`assert_matrix_invariants`] and in the `#[test]` below.
pub const MATRIX: &[OpGuarantee] = &[
    // ── Trit / Bit constructors ───────────────────────────────────────────────
    OpGuarantee {
        op: "Trit::new",
        tag: Tag::Exact,
        fallibility: Fallibility::NoneOnOffDomain("d ∉ {−1,0,+1}"),
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "Bit::new",
        tag: Tag::Exact,
        fallibility: Fallibility::NoneOnOffDomain("d ∉ {0,1}"),
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── Trit digit / neg ──────────────────────────────────────────────────────
    OpGuarantee {
        op: "Trit::digit",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "Neg::neg (Trit)",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── Bit ops ───────────────────────────────────────────────────────────────
    OpGuarantee {
        op: "Bit::digit",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "Bit::and",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "Bit::or",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "Bit::xor",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── int ↔ trits codec ────────────────────────────────────────────────────
    OpGuarantee {
        op: "trits_to_int",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "int_to_trits",
        tag: Tag::Exact,
        fallibility: Fallibility::NoneOnOffDomain("v ∉ [−(3^m−1)/2, +(3^m−1)/2]"),
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── balanced-ternary arithmetic ───────────────────────────────────────────
    OpGuarantee {
        op: "neg (Trits)",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "add",
        tag: Tag::Exact,
        fallibility: Fallibility::NoneOnOffDomain("fixed-width overflow or unequal widths"),
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "sub",
        tag: Tag::Exact,
        fallibility: Fallibility::NoneOnOffDomain("fixed-width overflow or unequal widths"),
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    OpGuarantee {
        op: "mul",
        tag: Tag::Exact,
        fallibility: Fallibility::NoneOnOffDomain("fixed-width overflow or unequal widths"),
        effects: "none",
        explainable: Explainable::NotApplicable,
    },
    // ── packed-ternary codecs ──────────────────────────────────────────────────
    OpGuarantee {
        op: "pack",
        tag: Tag::Exact,
        fallibility: Fallibility::ErrOn("PackError::OffGrid | PackError::Misaligned"),
        effects: "none",
        explainable: Explainable::Yes,
    },
    OpGuarantee {
        op: "unpack",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::Yes,
    },
    OpGuarantee {
        op: "scheme_of",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::Yes,
    },
    OpGuarantee {
        op: "explain",
        tag: Tag::Exact,
        fallibility: Fallibility::Total,
        effects: "none",
        explainable: Explainable::Yes,
    },
];

/// Structural invariants on the matrix — asserted in tests.
///
/// These are the machine-checkable version of the honesty rules applied to the matrix itself:
/// - C2/VR-5: every row must tag `Exact` (the justification is in the module doc; a non-`Exact`
///   row would be a violation of the spec's §4 guarantee matrix).
/// - C1/G2: any `Total` row must genuinely be total (no hidden conditions); any fallible row must
///   name its error set.
/// - C6: every row must declare `"none"` effects (all ops are pure; RFC-0014).
/// - C3: the explainable flag must be `Yes` for the pack ops and `NotApplicable` for the others.
pub fn assert_matrix_invariants() {
    // C2/VR-5: every tag is Exact.
    for row in MATRIX {
        assert_eq!(
            row.tag,
            Tag::Exact,
            "Matrix row '{}' must be Exact (C2/VR-5): every op in std.ternary \
             is an exact computation or a lossless re-encoding; a non-Exact tag \
             would be a dishonest downgrade. (Mutant witness: changing any tag to \
             Empirical would fail this assertion.)",
            row.op
        );
    }

    // C6: every effects column is "none".
    for row in MATRIX {
        assert_eq!(
            row.effects, "none",
            "Matrix row '{}' declares non-none effects '{}' — all std.ternary ops \
             are pure (no IO, time, randomness, or hidden allocation; C6/RFC-0014).",
            row.op, row.effects
        );
    }

    // C3: the pack ops are EXPLAIN-able; the rest are not.
    let explain_yes_ops = ["pack", "unpack", "scheme_of", "explain"];
    for row in MATRIX {
        let want = if explain_yes_ops.contains(&row.op) {
            Explainable::Yes
        } else {
            Explainable::NotApplicable
        };
        assert_eq!(
            row.explainable, want,
            "Matrix row '{}' has wrong EXPLAIN-able flag (C3).",
            row.op
        );
    }

    // C1: every fallible row names its error condition (non-empty string for NoneOnOffDomain /
    // ErrOn); Total rows have no condition string (trivially satisfied by the enum shape).
    for row in MATRIX {
        match row.fallibility {
            Fallibility::NoneOnOffDomain(cond) => {
                assert!(
                    !cond.is_empty(),
                    "Matrix row '{}' has NoneOnOffDomain with empty condition — \
                     the error set must be named (C1/G2).",
                    row.op
                );
            }
            Fallibility::ErrOn(variants) => {
                assert!(
                    !variants.is_empty(),
                    "Matrix row '{}' has ErrOn with empty variants — \
                     the error set must be named (C1/G2).",
                    row.op
                );
            }
            Fallibility::Total => {}
        }
    }

    // Coverage: every declared op name is non-empty and unique.
    let mut seen = std::collections::HashSet::new();
    for row in MATRIX {
        assert!(!row.op.is_empty(), "matrix row has empty op name");
        assert!(seen.insert(row.op), "duplicate op in matrix: '{}'", row.op);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarantee_matrix_invariants_hold() {
        // The load-bearing deliverable (RFC-0016 §4.5): the matrix is data, asserted here.
        // Mutant witness: changing any `Tag::Exact` to `Tag::Empirical` makes this fail.
        assert_matrix_invariants();
    }

    #[test]
    fn matrix_has_expected_op_count() {
        // 18 exported ops (as of the spec §4 table; update when ops are added/removed).
        assert_eq!(MATRIX.len(), 18, "op count mismatch — update the matrix");
    }

    #[test]
    fn all_tags_are_exact() {
        // C2/VR-5: re-assert directly so a CI failure message is clear.
        for row in MATRIX {
            assert_eq!(
                row.tag,
                Tag::Exact,
                "op '{}' must be Exact (balanced-ternary algebra is exact; \
                 codecs are lossless; C2/VR-5)",
                row.op
            );
        }
    }

    #[test]
    fn all_effects_are_none() {
        // C6: all ops are pure (RFC-0014).
        for row in MATRIX {
            assert_eq!(
                row.effects, "none",
                "op '{}' effects must be none (C6)",
                row.op
            );
        }
    }

    #[test]
    fn pack_ops_are_explainable() {
        // C3: the pack ops expose a scheme + EXPLAIN record.
        let pack_ops = ["pack", "unpack", "scheme_of", "explain"];
        for &name in &pack_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == name)
                .unwrap_or_else(|| panic!("op '{name}' not found in matrix — check for typo"));
            assert_eq!(
                row.explainable,
                Explainable::Yes,
                "op '{name}' must be EXPLAIN-able (C3)"
            );
        }
    }

    #[test]
    fn non_pack_ops_are_not_applicable_for_explain() {
        // C3: pure algebra ops do not select/convert/approximate — EXPLAIN not applicable.
        let non_pack_ops = [
            "Trit::new",
            "Bit::new",
            "Trit::digit",
            "Neg::neg (Trit)",
            "Bit::digit",
            "Bit::and",
            "Bit::or",
            "Bit::xor",
            "trits_to_int",
            "int_to_trits",
            "neg (Trits)",
            "add",
            "sub",
            "mul",
        ];
        for &name in &non_pack_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == name)
                .unwrap_or_else(|| panic!("op '{name}' not found in matrix — check for typo"));
            assert_eq!(
                row.explainable,
                Explainable::NotApplicable,
                "op '{name}' must not be EXPLAIN-able (no selection/conversion/approximation; C3)"
            );
        }
    }

    #[test]
    fn fallible_ops_name_their_error_set() {
        // C1/G2: every fallible op names its error condition — not just "some failure".
        for row in MATRIX {
            match row.fallibility {
                Fallibility::NoneOnOffDomain(cond) => {
                    assert!(
                        !cond.is_empty(),
                        "op '{}' NoneOnOffDomain has empty condition",
                        row.op
                    );
                }
                Fallibility::ErrOn(variants) => {
                    assert!(
                        !variants.is_empty(),
                        "op '{}' ErrOn has empty variants",
                        row.op
                    );
                }
                Fallibility::Total => {}
            }
        }
    }
}
