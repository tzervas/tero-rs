//! Mode-parametric test suite for the shared harness (M-795; RFC-0034 §13; DN-20).
//!
//! Every test here:
//!   (a) states its mode-scope explicitly (via [`ModeScope`] or `for_each_mode`),
//!   (b) asserts the intended per-mode behaviour (not one tier's), and
//!   (c) uses the **cross-mode negative** pattern where applicable — asserting both POSITIVE
//!       (the invariant fires where it should) and NEGATIVE (the invariant is correctly absent
//!       where it should not fire).
//!
//! Tests use the harness primitives from [`super::mode_harness`] directly:
//! `proven_bound`, `empirical_bound`, `declared_bound`, `canonical_bound`, `for_each_mode`,
//! `assert_meta_constructs`, `ModeScope`, and `assert_mode_scope`.

use super::mode_harness::{
    assert_meta_constructs, assert_mode_scope, canonical_bound, declared_bound, empirical_bound,
    for_each_mode, proven_bound, ModeScope,
};
use crate::bound::BoundBasis;
use crate::cert_mode::CertMode;
use crate::guarantee::GuaranteeStrength;
use crate::meta::{Meta, Provenance};

// ---------------------------------------------------------------------------
// § 1. Harness self-tests — verify the harness itself is correct
// ---------------------------------------------------------------------------

/// Each canonical fixture satisfies `Bound::well_formed()` and can construct a `Meta` with the
/// strength it claims to represent. Tests the single-source-of-truth `canonical_bound` accessor.
///
/// **Mode-scope:** ALL_MODES (this is a fixture-correctness test, not mode-conditional).
/// **Mutant-witness:** replacing `canonical_bound(Proven)` with `canonical_bound(Declared)` would
/// produce a bound with `UserDeclared` basis that fails M-I2 for `Proven`, catching any test that
/// aliases the wrong fixture.
#[test]
fn canonical_bound_fixtures_are_meta_constructible() {
    for &g in &GuaranteeStrength::ALL {
        let b = canonical_bound(g);
        assert_meta_constructs(g, b.clone());
        // The bound's presence/absence matches M-I1 (Exact ⟺ None).
        match g {
            GuaranteeStrength::Exact => assert!(b.is_none(), "Exact must have no bound (M-I1)"),
            _ => assert!(b.is_some(), "{g:?} must have a bound"),
        }
    }
}

/// Each named fixture independently satisfies `Bound::well_formed()`.
///
/// **Mode-scope:** ALL_MODES (fixture correctness, mode-independent).
/// **Mutant-witness:** `empirical_bound` with `trials: 0` would fail `Bound::well_formed`
/// (A6-02 guard: evidence-free empirical basis is rejected).
#[test]
fn named_fixtures_are_well_formed() {
    assert!(
        proven_bound().well_formed(),
        "proven_bound must be well-formed"
    );
    assert!(
        empirical_bound().well_formed(),
        "empirical_bound must be well-formed"
    );
    assert!(
        declared_bound().well_formed(),
        "declared_bound must be well-formed"
    );
}

/// `for_each_mode` iterates all three modes exactly once each, in depth order.
///
/// **Mutant-witness:** if `CertMode::ALL` were missing a variant, the collected depths would
/// not equal `[0, 1, 2]`, catching any truncation.
#[test]
fn for_each_mode_visits_all_three_modes_in_depth_order() {
    let mut depths = Vec::new();
    for_each_mode(|mode| depths.push(mode.depth()));
    assert_eq!(
        depths,
        vec![0, 1, 2],
        "for_each_mode must yield Fast/Balanced/Certified in order"
    );
}

/// `ModeScope::contains` returns the right boolean for every scope variant.
///
/// **Mutant-witness:** off-by-one in `contains` (e.g. indexing with `depth() - 1`) would
/// return wrong results for Fast (depth=0 → panic or Certified's slot).
#[test]
fn mode_scope_contains_is_correct_for_all_predefined_scopes() {
    // ALL_MODES: all three in scope.
    assert!(ModeScope::ALL_MODES.contains(CertMode::Fast));
    assert!(ModeScope::ALL_MODES.contains(CertMode::Balanced));
    assert!(ModeScope::ALL_MODES.contains(CertMode::Certified));

    // FAST_ONLY: only Fast.
    assert!(ModeScope::FAST_ONLY.contains(CertMode::Fast));
    assert!(!ModeScope::FAST_ONLY.contains(CertMode::Balanced));
    assert!(!ModeScope::FAST_ONLY.contains(CertMode::Certified));

    // NON_FAST / EMIT_MODES: Balanced + Certified only.
    for scope in [ModeScope::NON_FAST, ModeScope::EMIT_MODES] {
        assert!(!scope.contains(CertMode::Fast));
        assert!(scope.contains(CertMode::Balanced));
        assert!(scope.contains(CertMode::Certified));
    }

    // CERTIFIED_ONLY: only Certified.
    assert!(!ModeScope::CERTIFIED_ONLY.contains(CertMode::Fast));
    assert!(!ModeScope::CERTIFIED_ONLY.contains(CertMode::Balanced));
    assert!(ModeScope::CERTIFIED_ONLY.contains(CertMode::Certified));
}

// ---------------------------------------------------------------------------
// § 2. Gate policy tests — using the harness for mode-parametric assertions
// ---------------------------------------------------------------------------

/// **`gate_result` output is always Meta-constructible across all modes** (the central M-788
/// contract). Exhaustive over `CertMode::ALL × GuaranteeStrength::ALL` using the harness's
/// `canonical_bound` for the consistent pre-image.
///
/// **Mode-scope:** ALL_MODES (the invariant is unconditional).
/// **Mutant-witness:** if `gate_result` returned `(Proven, None)` for any mode, `Meta::new`
/// would fail M-I2 and this test would catch it. If it returned `(Exact, proven_bound())`,
/// it would fail M-I1.
#[test]
fn gate_result_output_is_meta_constructible_in_every_mode_via_harness() {
    for_each_mode(|mode| {
        for &g in &GuaranteeStrength::ALL {
            let b = canonical_bound(g);
            let (gated_g, gated_b) = mode.gate_result(g, b);
            assert_meta_constructs(gated_g, gated_b);
        }
    });
}

/// **`Fast` floors `Proven`/`Empirical` to `Declared`; `Balanced`/`Certified` pass them through.**
///
/// Cross-mode negative pattern with `assert_mode_scope`:
/// - Predicate: "the gated guarantee equals the intended (no flooring occurred)"
/// - In `FAST_ONLY` scope → Fast *does* floor, so the predicate is FALSE for Fast (i.e. the
///   flooring scope where we expect floor to happen). We invert: assert the floor IS present
///   in Fast and absent in Balanced/Certified.
///
/// **Mode-scope:** FAST_ONLY for the floor; NON_FAST for pass-through.
/// **Mutant-witness:** if `gate_guarantee` passed `Proven` through in Fast (no floor), the
/// `FAST_ONLY` assertion would fail ("floor holds in Balanced too" is impossible, so the catch
/// is the POSITIVE arm: "floor does NOT hold in Fast"). The `NON_FAST` assertion would also
/// catch if Balanced started flooring (predicate becomes false, scope says it should hold).
#[test]
fn fast_floors_proven_empirical_to_declared_cross_mode_negative() {
    for intended in [GuaranteeStrength::Proven, GuaranteeStrength::Empirical] {
        let b = canonical_bound(intended);

        // The floor IS active in Fast: the result is Declared (not the intended strength).
        assert_mode_scope(
            ModeScope::FAST_ONLY,
            |mode| {
                let (g, _) = mode.gate_result(intended, b.clone());
                g == GuaranteeStrength::Declared
            },
            "fast floors Proven/Empirical to Declared",
        );

        // Pass-through IS active in Balanced/Certified: the result equals the intended strength.
        assert_mode_scope(
            ModeScope::NON_FAST,
            |mode| {
                let (g, _) = mode.gate_result(intended, b.clone());
                g == intended
            },
            "Balanced/Certified pass Proven/Empirical through unchanged",
        );
    }
}

/// **`Exact` is structural and passes through in every mode** (the free, structural tag is
/// never downgraded and never requires a bound).
///
/// **Mode-scope:** ALL_MODES.
/// **Mutant-witness:** if any mode downgraded `Exact` to `Declared`, the predicate would
/// be false for that mode, and ALL_MODES would catch the positive failure.
#[test]
fn exact_passes_through_in_every_mode() {
    assert_mode_scope(
        ModeScope::ALL_MODES,
        |mode| {
            let (g, b) = mode.gate_result(GuaranteeStrength::Exact, None);
            g == GuaranteeStrength::Exact && b.is_none()
        },
        "Exact is structural and passes through bound-free (M-I1) in every mode",
    );
}

/// **In `Fast`, a floored result's bound basis is always `UserDeclared`** (M-I4/M-788). The
/// basis is correctly NOT `ProvenThm`/`EmpiricalFit` — the machinery did not run.
///
/// Cross-mode negative: the `UserDeclared`-basis constraint is FAST_ONLY for the `Proven` and
/// `Empirical` intents; in Balanced/Certified the original basis is earned and preserved.
///
/// **Mutant-witness:** if Fast forgot to relabel the basis (passed ProvenThm through), the
/// `FAST_ONLY` predicate `b.basis == UserDeclared` would be false in Fast, catching the
/// positive failure. If Balanced started relabelling, the `NON_FAST` check would catch it.
#[test]
fn fast_bound_basis_is_user_declared_non_fast_preserves_original() {
    // Proven intent: ProvenThm basis is relabelled in Fast, preserved in Balanced/Certified.
    let p = proven_bound();

    assert_mode_scope(
        ModeScope::FAST_ONLY,
        |mode| {
            let (_, b) = mode.gate_result(GuaranteeStrength::Proven, Some(p.clone()));
            b.map(|b| b.basis == BoundBasis::UserDeclared)
                .unwrap_or(false)
        },
        "Fast relabels ProvenThm → UserDeclared (M-I4)",
    );

    assert_mode_scope(
        ModeScope::NON_FAST,
        |mode| {
            let (_, b) = mode.gate_result(GuaranteeStrength::Proven, Some(p.clone()));
            matches!(b.map(|b| b.basis), Some(BoundBasis::ProvenThm { .. }))
        },
        "Balanced/Certified preserve ProvenThm basis (earned — machinery runs)",
    );

    // Empirical intent: EmpiricalFit basis is relabelled in Fast, preserved in Balanced/Certified.
    let e = empirical_bound();

    assert_mode_scope(
        ModeScope::FAST_ONLY,
        |mode| {
            let (_, b) = mode.gate_result(GuaranteeStrength::Empirical, Some(e.clone()));
            b.map(|b| b.basis == BoundBasis::UserDeclared)
                .unwrap_or(false)
        },
        "Fast relabels EmpiricalFit → UserDeclared (M-I4)",
    );

    assert_mode_scope(
        ModeScope::NON_FAST,
        |mode| {
            let (_, b) = mode.gate_result(GuaranteeStrength::Empirical, Some(e.clone()));
            matches!(b.map(|b| b.basis), Some(BoundBasis::EmpiricalFit { .. }))
        },
        "Balanced/Certified preserve EmpiricalFit basis (earned — machinery runs)",
    );
}

/// **cert_mode tag is present on every `Meta` in every mode** (Axis-B never-silent, RFC-0034 §3.1).
///
/// This is the ALL_MODES positive: the mode tag is not relaxed by mode — it exists in Fast,
/// Balanced, and Certified equally. There is no cross-mode negative here (the tag is
/// unconditional), so we use `for_each_mode` directly.
///
/// **Mode-scope:** ALL_MODES.
/// **Mutant-witness:** if `Meta::cert_mode()` returned a hardcoded default and ignored
/// `with_cert_mode`, the three modes would all return `Fast`, and the `Balanced`/`Certified`
/// assertions would fail.
#[test]
fn cert_mode_tag_is_present_in_every_mode() {
    for_each_mode(|mode| {
        let m = Meta::exact(Provenance::Root).with_cert_mode(mode);
        assert_eq!(
            m.cert_mode(),
            mode,
            "cert_mode tag must be recorded exactly in every mode (never-silent; RFC-0034 §3.1)"
        );
        // The mode tag is independent of the guarantee strength (VR-5: not an upgrade).
        assert_eq!(
            m.guarantee(),
            GuaranteeStrength::Exact,
            "mode tag must not affect guarantee"
        );
    });
}

/// **`Empirical`/`Proven` are reachable in `Balanced`/`Certified`, never in `Fast`** (M-787; VR-5).
///
/// This is the main cross-mode negative for guarantee reachability (RFC-0034 §7): the Empirical
/// and Proven tags are ONLY reachable in the machinery modes. Fast cannot produce them by the gate.
///
/// **Mode-scope:** NON_FAST for "Proven/Empirical reachable"; FAST_ONLY "floors these to Declared".
/// (This test is the dual of `fast_floors_proven_empirical_to_declared_cross_mode_negative`:
/// that test was about the floor *being present*; this one is about the *tag never appearing* in Fast.)
///
/// **Mutant-witness:** if `gate_guarantee` passed `Proven` through in Fast (no floor), this
/// test's FAST_ONLY arm — "Fast floors" means the gated result is NOT Proven — would fail.
#[test]
fn proven_and_empirical_are_unreachable_in_fast_reachable_in_non_fast() {
    for intended in [GuaranteeStrength::Proven, GuaranteeStrength::Empirical] {
        // Fast: gated result is NEVER Proven or Empirical.
        assert_mode_scope(
            ModeScope::FAST_ONLY,
            |mode| {
                let (g, _) = mode.gate_result(intended, canonical_bound(intended));
                g != GuaranteeStrength::Proven && g != GuaranteeStrength::Empirical
            },
            "Fast never yields Proven/Empirical (M-787 floor)",
        );

        // Non-Fast: gated result IS the intended strength (Proven or Empirical).
        assert_mode_scope(
            ModeScope::NON_FAST,
            |mode| {
                let (g, _) = mode.gate_result(intended, canonical_bound(intended));
                g == intended
            },
            "Balanced/Certified can yield Proven/Empirical (machinery runs)",
        );
    }
}

/// **`gate_guarantee` and `gate_result`'s guarantee component agree in every mode** — single
/// source for the mode→tag policy (no divergent second policy).
///
/// **Mode-scope:** ALL_MODES (the consistency invariant is unconditional).
/// **Mutant-witness:** if `gate_result` had a bug where it used a different floor condition than
/// `gate_guarantee`, the assertion `gated_g == mode.gate_guarantee(g)` would fail for the
/// divergent (mode, g) pair.
#[test]
fn gate_result_guarantee_agrees_with_gate_guarantee_in_every_mode() {
    for_each_mode(|mode| {
        for &g in &GuaranteeStrength::ALL {
            let b = canonical_bound(g);
            let (gated_g, _) = mode.gate_result(g, b);
            assert_eq!(
                gated_g,
                mode.gate_guarantee(g),
                "gate_result's guarantee must equal gate_guarantee (mode={mode:?}, intent={g:?})"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// § 3. assert_mode_scope self-test — the harness primitive catches both directions
// ---------------------------------------------------------------------------

/// **`assert_mode_scope` catches a predicate that is always-true in a bounded scope.**
///
/// This verifies the cross-mode NEGATIVE arm of `assert_mode_scope`: if a predicate returns
/// `true` even for modes outside the scope, the assertion should panic. We catch the panic to
/// confirm the guard works.
///
/// **Guarantee tag:** `Declared` — this is a test-of-the-test, not a verified property.
#[test]
fn assert_mode_scope_panics_on_always_true_with_fast_only_scope() {
    let result = std::panic::catch_unwind(|| {
        // FAST_ONLY scope: the predicate must be false for Balanced and Certified.
        // An always-true predicate violates the NEGATIVE arm for Balanced/Certified.
        assert_mode_scope(
            ModeScope::FAST_ONLY,
            |_mode| true,
            "always true — must panic",
        );
    });
    assert!(
        result.is_err(),
        "assert_mode_scope must panic when the predicate holds outside the scope (NEGATIVE arm)"
    );
}

/// **`assert_mode_scope` catches a predicate that is always-false in a positive scope.**
///
/// Verifies the cross-mode POSITIVE arm: if a predicate returns `false` for a mode that is in
/// scope, the assertion should panic.
///
/// **Guarantee tag:** `Declared`.
#[test]
fn assert_mode_scope_panics_on_always_false_with_all_modes_scope() {
    let result = std::panic::catch_unwind(|| {
        // ALL_MODES scope: the predicate must be true for every mode.
        // An always-false predicate violates the POSITIVE arm for Fast (and all others).
        assert_mode_scope(
            ModeScope::ALL_MODES,
            |_mode| false,
            "always false — must panic",
        );
    });
    assert!(
        result.is_err(),
        "assert_mode_scope must panic when the predicate is absent inside the scope (POSITIVE arm)"
    );
}

/// **`assert_mode_scope` succeeds when the predicate exactly matches the scope.**
///
/// Verifies the happy path: if a predicate returns `true` exactly for the modes in scope and
/// `false` for all others, no panic occurs.
///
/// **Guarantee tag:** `Declared`.
#[test]
fn assert_mode_scope_succeeds_when_predicate_matches_scope_exactly() {
    // FAST_ONLY: predicate is true for Fast only.
    assert_mode_scope(
        ModeScope::FAST_ONLY,
        |mode| mode == CertMode::Fast,
        "predicate matches FAST_ONLY scope exactly",
    );

    // NON_FAST: predicate is true for Balanced and Certified only.
    assert_mode_scope(
        ModeScope::NON_FAST,
        |mode| mode != CertMode::Fast,
        "predicate matches NON_FAST scope exactly",
    );

    // ALL_MODES: predicate is always true.
    assert_mode_scope(
        ModeScope::ALL_MODES,
        |_mode| true,
        "predicate matches ALL_MODES scope",
    );
}
