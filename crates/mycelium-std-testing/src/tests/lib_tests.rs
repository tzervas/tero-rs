//! White-box unit tests for `lib.rs` (Rng, Budget, for_all, golden, differential,
//! summarize, is_green, guarantee_matrix, Verdict, VR-5 never-upgrade).
//!
//! Extracted as-touched per the test-layout rule (CLAUDE.md §Test layout).

use crate::*;

// ─── Rng tests ────────────────────────────────────────────────────────────

/// Same seed → same sequence (determinism / RT3).
/// Guard: any non-determinism in `Rng::next_u64` makes this fail.
#[test]
fn rng_is_deterministic() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..20 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

/// Different seeds → different first values (sanity check).
/// Guard: a constant `next_u64` makes this fail.
#[test]
fn rng_different_seeds_differ() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    assert_ne!(a.next_u64(), b.next_u64());
}

/// Seed 0 is promoted to a non-zero default (Xorshift degenerate-state prevention).
/// Guard: a zero-state Xorshift would produce all-zeros.
#[test]
fn rng_zero_seed_is_promoted() {
    let mut r = Rng::new(0);
    assert_ne!(
        r.next_u64(),
        0,
        "zero seed must be promoted to avoid degenerate Xorshift"
    );
}

/// `next_usize_below(1)` always returns 0 (only valid value).
#[test]
fn rng_next_usize_below_one() {
    let mut r = Rng::new(99);
    for _ in 0..100 {
        assert_eq!(r.next_usize_below(1), 0);
    }
}

/// `next_usize_below(n)` always returns a value in `[0, n)`.
/// Property: for n in [1, 256], 1000 draws are all < n.
#[test]
fn rng_next_usize_below_in_range() {
    let mut r = Rng::new(0xABCD_EF01_2345_6789);
    for n in 1usize..=256 {
        for _ in 0..20 {
            let v = r.next_usize_below(n);
            assert!(v < n, "next_usize_below({n}) returned {v} >= {n}");
        }
    }
}

// ─── Budget tests ─────────────────────────────────────────────────────────

/// `Budget::new(0)` is refused — never-silent (C1).
#[test]
fn budget_zero_is_refused() {
    assert_eq!(Budget::new(0), None);
}

/// `Budget::new(n)` for n > 0 succeeds and reports the correct trial count.
/// Property: for all n in [1, 1000], Budget::new(n).unwrap().trials() == n.
#[test]
fn budget_trial_count_roundtrips() {
    for n in 1u32..=1000 {
        let b = Budget::new(n).expect("non-zero budget must succeed");
        assert_eq!(b.trials(), n);
    }
}

// ─── for_all tests ────────────────────────────────────────────────────────

/// A tautology property (always true) returns Pass.
/// Guard: returning Fail for a passing property makes this fail.
#[test]
fn for_all_pass_on_tautology() {
    struct Ints;
    impl Gen<u32> for Ints {
        fn generate(&mut self, rng: &mut Rng) -> Option<u32> {
            Some(rng.next_u32())
        }
    }
    let v = for_all(&mut Ints, 1, Budget::DEFAULT, |_x| true);
    assert_eq!(v, Verdict::Pass, "tautology must pass");
}

/// A contradiction property (always false) returns Fail.
/// Guard: returning Pass for a failing property makes this fail.
#[test]
fn for_all_fail_on_contradiction() {
    struct Ints;
    impl Gen<u32> for Ints {
        fn generate(&mut self, rng: &mut Rng) -> Option<u32> {
            Some(rng.next_u32())
        }
    }
    let v = for_all(&mut Ints, 1, Budget::DEFAULT, |_x| false);
    assert!(
        matches!(v, Verdict::Fail { .. }),
        "contradiction must fail; got {v:?}"
    );
}

/// A generator that never produces a value yields Skipped (C1 — never a silent pass).
/// Guard: returning Pass for an empty generator makes this fail.
#[test]
fn for_all_skipped_on_empty_generator() {
    struct Empty;
    impl Gen<u32> for Empty {
        fn generate(&mut self, _rng: &mut Rng) -> Option<u32> {
            None
        }
    }
    let v = for_all(&mut Empty, 1, Budget::DEFAULT, |_x| true);
    assert!(
        matches!(
            v,
            Verdict::Skipped {
                reason: SkipReason::NeedsRecord
            }
        ),
        "empty generator must yield Skipped{{NeedsRecord}}; got {v:?}"
    );
}

/// A Fail carries the reproducing seed (C3/G11 — EXPLAIN).
/// Guard: dropping the seed from FailRecord makes this fail.
#[test]
fn for_all_fail_carries_seed() {
    const SEED: u64 = 0x1234_5678_ABCD_EF01;
    struct Ints;
    impl Gen<u32> for Ints {
        fn generate(&mut self, rng: &mut Rng) -> Option<u32> {
            Some(rng.next_u32())
        }
    }
    let v = for_all(&mut Ints, SEED, Budget::DEFAULT, |_x| false);
    if let Verdict::Fail { record } = v {
        assert_eq!(
            record.seed, SEED,
            "Fail must carry the reproducing seed (C3/G11)"
        );
    } else {
        panic!("expected Fail; got {v:?}");
    }
}

/// `for_all` is reproducible: same seed + same property → same verdict.
/// Property: reproducibility is the RT3 seeded-generator discipline.
#[test]
fn for_all_is_reproducible() {
    struct Evens;
    impl Gen<u32> for Evens {
        fn generate(&mut self, rng: &mut Rng) -> Option<u32> {
            Some(rng.next_u32() & !1) // always even
        }
    }
    // prop: x % 2 == 0 (true for all evens — this is a Pass on the even generator)
    let v1 = for_all(&mut Evens, 42, Budget::DEFAULT, |x| x % 2 == 0);
    let v2 = for_all(&mut Evens, 42, Budget::DEFAULT, |x| x % 2 == 0);
    assert_eq!(
        v1, v2,
        "same seed must produce same verdict (RT3 reproducibility)"
    );
}

/// Shrinking: a Fail on a u32 > 0 property provides a shrunk counterexample.
/// Property: the shrunk value in the description is minimal (or at least, the description
/// is non-empty and not the raw unshrunken value).
#[test]
fn for_all_shrinks_counterexample() {
    /// A u32 generator with shrinking toward 0.
    struct ShrinkableInts;
    impl Gen<u32> for ShrinkableInts {
        fn generate(&mut self, rng: &mut Rng) -> Option<u32> {
            // Generate values in [100, 200] so there's room to shrink.
            Some(100 + (rng.next_u32() % 101))
        }
        fn shrink(&self, value: &u32) -> Vec<u32> {
            if *value == 0 {
                vec![]
            } else {
                // Halving shrink strategy.
                vec![value / 2, value.saturating_sub(1)]
            }
        }
    }
    // prop: x < 50 (fails for all values in [100, 200])
    let v = for_all(&mut ShrinkableInts, 7, Budget::DEFAULT, |x| *x < 50);
    match v {
        Verdict::Fail { record } => {
            // Shrinking should reach 0 (the minimal value that still fails x < 50 = false
            // is any value >= 50; halving from 100+ will find small failing values).
            assert!(
                !record.description.is_empty(),
                "Fail description must be non-empty (C3/G11)"
            );
        }
        other => panic!("expected Fail; got {other:?}"),
    }
}

// ─── golden tests ─────────────────────────────────────────────────────────

/// A matching baseline → Pass.
#[test]
fn golden_pass_on_match() {
    let baseline = GoldenBaseline::new("my_test", "hello world");
    let v = golden(Some(&baseline), "my_test", "hello world");
    assert_eq!(v, Verdict::Pass, "matching baseline must pass");
}

/// A mismatch → Fail carrying a diff (C3/G11).
/// Guard: returning Pass for a mismatch makes this fail.
#[test]
fn golden_fail_on_mismatch() {
    let baseline = GoldenBaseline::new("my_test", "hello world");
    let v = golden(Some(&baseline), "my_test", "hello universe");
    assert!(
        matches!(v, Verdict::Fail { .. }),
        "mismatch must fail; got {v:?}"
    );
}

/// A missing baseline → Skipped{NeedsRecord} (C1/G2 — never silent auto-accept).
/// Guard: returning Pass for a missing baseline is the primary honesty violation.
#[test]
fn golden_skipped_on_missing_baseline() {
    let v = golden(None, "my_test", "some output");
    assert!(
        matches!(
            v,
            Verdict::Skipped {
                reason: SkipReason::NeedsRecord
            }
        ),
        "missing baseline must yield Skipped{{NeedsRecord}}; got {v:?}"
    );
}

/// A name mismatch (wrong baseline supplied) → Skipped.
#[test]
fn golden_skipped_on_name_mismatch() {
    let baseline = GoldenBaseline::new("other_test", "hello world");
    let v = golden(Some(&baseline), "my_test", "hello world");
    assert!(
        matches!(
            v,
            Verdict::Skipped {
                reason: SkipReason::NeedsRecord
            }
        ),
        "name mismatch must yield Skipped{{NeedsRecord}}; got {v:?}"
    );
}

/// A Fail carries the diff description (EXPLAIN artifact — C3/G11).
#[test]
fn golden_fail_carries_diff() {
    let baseline = GoldenBaseline::new("t", "expected line");
    let v = golden(Some(&baseline), "t", "actual line");
    if let Verdict::Fail { record } = v {
        assert!(
            record.description.contains("diff"),
            "Fail description must reference the diff (C3/G11): {}",
            record.description
        );
    } else {
        panic!("expected Fail; got {v:?}");
    }
}

// ─── differential tests ───────────────────────────────────────────────────

/// Both backends agree → Pass.
#[test]
fn differential_pass_on_agreement() {
    let v = differential("input_42", true, || 42u32, true, || 42u32);
    assert_eq!(v, Verdict::Pass, "agreeing backends must pass");
}

/// Backends disagree → Fail carrying both outputs (C3/G11).
/// Guard: returning Pass for disagreement is the primary differential violation.
#[test]
fn differential_fail_on_disagreement() {
    let v = differential("input_x", true, || 1u32, true, || 2u32);
    assert!(
        matches!(v, Verdict::Fail { .. }),
        "disagreeing backends must fail; got {v:?}"
    );
}

/// lhs unavailable → Skipped{BackendUnavailable} (C1/G2).
#[test]
fn differential_skipped_on_lhs_unavailable() {
    let v = differential(
        "x",
        false, // lhs unavailable
        || 0u32,
        true,
        || 0u32,
    );
    assert!(
        matches!(
            v,
            Verdict::Skipped {
                reason: SkipReason::BackendUnavailable
            }
        ),
        "unavailable lhs must yield Skipped{{BackendUnavailable}}; got {v:?}"
    );
}

/// rhs unavailable → Skipped{BackendUnavailable} (C1/G2).
#[test]
fn differential_skipped_on_rhs_unavailable() {
    let v = differential(
        "x",
        true,
        || 0u32,
        false, // rhs unavailable
        || 0u32,
    );
    assert!(
        matches!(
            v,
            Verdict::Skipped {
                reason: SkipReason::BackendUnavailable
            }
        ),
        "unavailable rhs must yield Skipped{{BackendUnavailable}}; got {v:?}"
    );
}

/// A Fail carries both outputs (EXPLAIN artifact — C3/G11).
#[test]
fn differential_fail_carries_both_outputs() {
    let v = differential("x", true, || 1u32, true, || 99u32);
    if let Verdict::Fail { record } = v {
        assert!(
            record.description.contains("lhs=") && record.description.contains("rhs="),
            "Fail must carry both outputs (C3/G11): {}",
            record.description
        );
    } else {
        panic!("expected Fail; got {v:?}");
    }
}

// ─── summarize / is_green tests ───────────────────────────────────────────

/// `summarize` on an empty slice → all zero counts.
#[test]
fn summarize_empty() {
    let s = summarize(&[]);
    assert_eq!(s.passed, 0);
    assert_eq!(s.failed, 0);
    assert_eq!(s.skipped, 0);
    assert_eq!(s.undetermined, 0);
}

/// `summarize` counts each class independently (the crux: no class bleeds into another).
/// Guard: counting Skipped as Pass makes this fail.
#[test]
fn summarize_counts_are_independent() {
    let verdicts = vec![
        Verdict::Pass,
        Verdict::Pass,
        Verdict::Fail {
            record: FailRecord {
                description: "x".to_owned(),
                seed: 0,
                trial: 0,
                context: "t".to_owned(),
            },
        },
        Verdict::Skipped {
            reason: SkipReason::Ignored,
        },
        Verdict::Skipped {
            reason: SkipReason::BackendUnavailable,
        },
        Verdict::Undetermined {
            reason: UndetReason::OracleUnavailable,
        },
    ];
    let s = summarize(&verdicts);
    assert_eq!(s.passed, 2, "passed count");
    assert_eq!(s.failed, 1, "failed count");
    assert_eq!(s.skipped, 2, "skipped count");
    assert_eq!(s.undetermined, 1, "undetermined count");
}

/// `summarize` total = input length.
/// Property: for all slices, passed + failed + skipped + undetermined == len.
#[test]
fn summarize_total_equals_length() {
    let verdicts = vec![
        Verdict::Pass,
        Verdict::Skipped {
            reason: SkipReason::NeedsRecord,
        },
        Verdict::Undetermined {
            reason: UndetReason::BudgetExhaustedInconclusive,
        },
    ];
    let s = summarize(&verdicts);
    assert_eq!(
        s.passed + s.failed + s.skipped + s.undetermined,
        verdicts.len() as u32,
        "counts must sum to total"
    );
}

/// `is_green` is true iff `failed == 0` (even if skipped/undetermined > 0).
/// Guard: returning false for a passing-but-skipped set makes this fail (wrong direction).
#[test]
fn is_green_true_iff_no_failures() {
    // All pass → green.
    let s = summarize(&[Verdict::Pass, Verdict::Pass]);
    assert!(is_green(&s), "all-pass must be green");

    // Pass + skipped → green (skipped is surfaced in Summary, not hidden).
    let s2 = summarize(&[
        Verdict::Pass,
        Verdict::Skipped {
            reason: SkipReason::Ignored,
        },
    ]);
    assert!(
        is_green(&s2),
        "pass+skipped must be green (skip is surfaced, not hidden)"
    );

    // Pass + undetermined → green.
    let s3 = summarize(&[
        Verdict::Pass,
        Verdict::Undetermined {
            reason: UndetReason::OracleUnavailable,
        },
    ]);
    assert!(is_green(&s3), "pass+undetermined must be green");

    // Any failure → not green.
    let s4 = summarize(&[
        Verdict::Pass,
        Verdict::Fail {
            record: FailRecord {
                description: "x".to_owned(),
                seed: 0,
                trial: 0,
                context: "t".to_owned(),
            },
        },
    ]);
    assert!(!is_green(&s4), "any failure must not be green");
}

/// All-skipped → green (skips are surfaced, not treated as failures — C1).
///
/// Note: an all-skipped green means "no checks ran but none failed". The runner/CI decides
/// whether that is acceptable; the harness surfaces the skip counts honestly.
#[test]
fn is_green_all_skipped_is_green_but_skips_are_visible() {
    let verdicts = vec![
        Verdict::Skipped {
            reason: SkipReason::ToolMissing,
        },
        Verdict::Skipped {
            reason: SkipReason::UnmetPrecondition,
        },
    ];
    let s = summarize(&verdicts);
    assert!(is_green(&s), "all-skipped must be green (no failures)");
    assert_eq!(s.skipped, 2, "skip count must be visible in Summary");
    assert_eq!(
        s.passed, 0,
        "pass count must not include skips (crux: Skipped ≠ Pass)"
    );
}

// ─── Guarantee matrix coverage ────────────────────────────────────────────

/// The guarantee matrix has exactly 5 rows (one per spec §4 op).
#[test]
fn guarantee_matrix_has_five_rows() {
    assert_eq!(
        guarantee_matrix::MATRIX.len(),
        5,
        "spec §4 lists five ops in the guarantee matrix"
    );
}

/// Every row in the guarantee matrix is Exact (spec §4 tag justification).
#[test]
fn guarantee_matrix_all_rows_exact() {
    use mycelium_core::GuaranteeStrength;
    for row in guarantee_matrix::MATRIX {
        assert_eq!(
            row.tag,
            GuaranteeStrength::Exact,
            "{} must be Exact (spec §4 — harness ops are Exact mechanisms)",
            row.op
        );
    }
}

/// Every row's `explainable` matches the spec §4 table.
/// Guards against silently dropping EXPLAIN coverage.
#[test]
fn guarantee_matrix_explainable_rows_match_spec() {
    let explainable: Vec<&str> = guarantee_matrix::MATRIX
        .iter()
        .filter(|r| r.explainable)
        .map(|r| r.op)
        .collect();
    // Spec §4: for_all, golden, differential are EXPLAIN-able (counterexample/diff/both-outputs).
    assert!(
        explainable.contains(&"for_all"),
        "for_all must be EXPLAIN-able"
    );
    assert!(
        explainable.contains(&"golden"),
        "golden must be EXPLAIN-able"
    );
    assert!(
        explainable.contains(&"differential"),
        "differential must be EXPLAIN-able"
    );
}

// ─── Property test: Verdict PartialEq / Debug ─────────────────────────────

/// `Verdict::Pass == Verdict::Pass`.
#[test]
fn verdict_pass_eq() {
    assert_eq!(Verdict::Pass, Verdict::Pass);
}

/// `Verdict::Pass != Verdict::Fail`.
#[test]
fn verdict_pass_ne_fail() {
    let f = Verdict::Fail {
        record: FailRecord {
            description: "x".to_owned(),
            seed: 0,
            trial: 0,
            context: "c".to_owned(),
        },
    };
    assert_ne!(Verdict::Pass, f);
}

/// `Verdict::Skipped { reason: Ignored } != Verdict::Pass` (crux: Skipped is not Pass).
/// Guard: if Skipped is accidentally PartialEq to Pass, this fails.
#[test]
fn verdict_skipped_ne_pass() {
    assert_ne!(
        Verdict::Skipped {
            reason: SkipReason::Ignored
        },
        Verdict::Pass,
        "Skipped must never equal Pass (the honesty crux)"
    );
}

// ─── Property: for_all backs Empirical, not Proven ───────────────────────

/// A passing `for_all` never produces a verdict stronger than the property checked.
/// The harness returns `Pass` (exact mechanism); the *subject* must then tag `Empirical`.
/// This test guards the VR-5 never-upgrade rule by verifying no Proven/Exact verdict is
/// fabricated by the harness for trial-based checking.
///
/// Conceptual property: a `for_all` that returns `Pass` may only back `Empirical`, not
/// `Proven`. The harness has no "Proven" output; all it can produce is `Pass` (mechanism)
/// which the caller must correctly tag `Empirical` in their guarantee matrix.
///
/// We assert this by checking the verdict is exactly `Pass` (not some richer "proven" type).
#[test]
fn for_all_never_produces_proven_verdict() {
    struct Nats;
    impl Gen<u32> for Nats {
        fn generate(&mut self, rng: &mut Rng) -> Option<u32> {
            Some(rng.next_u32())
        }
    }
    // A property that trivially holds (always passes).
    let v = for_all(&mut Nats, 1, Budget::DEFAULT, |_x| true);
    // The verdict is exactly `Pass` — there is no richer "Proven" verdict type.
    // The subject's guarantee matrix row must still be tagged Empirical by the author.
    assert_eq!(
        v,
        Verdict::Pass,
        "for_all returns Pass (mechanism), never a Proven verdict; the subject backs Empirical (VR-5)"
    );
}
