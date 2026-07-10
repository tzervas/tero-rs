//! In-crate tests for [`crate::cert_mode_test`] — the scoped mode-parametric toolkit (M-796;
//! RFC-0034 §13).
//!
//! Every test:
//! (a) states its mode-scope explicitly (the DN-20 transparency rule),
//! (b) exercises the cross-mode negative pattern where applicable — asserting POSITIVE (invariant
//!     fires where it should) and NEGATIVE (invariant absent where it should not), and
//! (c) names a mutant-witness: one concrete mutation that makes the test fail (house rule §3 banked guard #7).
//!
//! Tests are organised in four sections:
//! - §1 `ModeScope` primitives
//! - §2 `ModeTestConfig` + shared resolver
//! - §3 `for_each_mode` / `for_each_mode_in` iteration helpers
//! - §4 `assert_mode_scope` / `assert_mode_negative` assertion helpers
//! - §5 Worked example / integration test

use crate::cert_mode_test::{
    assert_mode_negative, assert_mode_scope, for_each_mode, for_each_mode_in, ModeScope,
    ModeTestConfig, ModeVisit,
};
use mycelium_core::cert_mode::CertMode;
use mycelium_proj::{CertDecl, CertScope, ResolvedMode};

// ---------------------------------------------------------------------------
// §1 ModeScope primitives
// ---------------------------------------------------------------------------

/// All predefined scope constants agree with their documented membership.
///
/// **Mode-scope:** ALL_MODES (testing the scope type itself — mode-independent).
/// **Mutant-witness:** swapping `in_scope[0]` for `FAST_ONLY` from `true` to `false` would cause
/// `contains(Fast)` to return `false`, failing the `assert!(FAST_ONLY.contains(Fast))` check.
#[test]
fn mode_scope_predefined_constants_have_correct_membership() {
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
        assert!(
            !scope.contains(CertMode::Fast),
            "NON_FAST/EMIT_MODES must exclude Fast"
        );
        assert!(
            scope.contains(CertMode::Balanced),
            "NON_FAST/EMIT_MODES must include Balanced"
        );
        assert!(
            scope.contains(CertMode::Certified),
            "NON_FAST/EMIT_MODES must include Certified"
        );
    }

    // CERTIFIED_ONLY: only Certified.
    assert!(!ModeScope::CERTIFIED_ONLY.contains(CertMode::Fast));
    assert!(!ModeScope::CERTIFIED_ONLY.contains(CertMode::Balanced));
    assert!(ModeScope::CERTIFIED_ONLY.contains(CertMode::Certified));

    // BALANCED_ONLY: only Balanced.
    assert!(!ModeScope::BALANCED_ONLY.contains(CertMode::Fast));
    assert!(ModeScope::BALANCED_ONLY.contains(CertMode::Balanced));
    assert!(!ModeScope::BALANCED_ONLY.contains(CertMode::Certified));
}

/// `modes_in_scope` returns exactly the modes in the scope.
///
/// **Mutant-witness:** if `contains` used `depth() + 1` as the index, `Fast` (depth=0) would
/// panic (OOB), and `Balanced` would map to index 2 (Certified's slot) — wrong output.
#[test]
fn modes_in_scope_matches_contains() {
    for scope in [
        ModeScope::ALL_MODES,
        ModeScope::FAST_ONLY,
        ModeScope::NON_FAST,
        ModeScope::CERTIFIED_ONLY,
        ModeScope::BALANCED_ONLY,
    ] {
        let modes = scope.modes_in_scope();
        for &m in &CertMode::ALL {
            assert_eq!(
                modes.contains(&m),
                scope.contains(m),
                "modes_in_scope must agree with contains for mode={m:?} in scope={scope:?}"
            );
        }
    }
}

/// `modes_out_of_scope` is the complement of `modes_in_scope`.
///
/// **Mutant-witness:** if `modes_out_of_scope` used `contains` instead of `!contains`, the
/// intersection with `modes_in_scope` would be non-empty, failing the disjointness check.
#[test]
fn modes_out_of_scope_is_complement_of_in_scope() {
    for scope in [
        ModeScope::ALL_MODES,
        ModeScope::NON_FAST,
        ModeScope::CERTIFIED_ONLY,
    ] {
        let in_scope = scope.modes_in_scope();
        let out_scope = scope.modes_out_of_scope();
        // Disjoint:
        for m in &in_scope {
            assert!(
                !out_scope.contains(m),
                "mode {:?} must not appear in both modes_in_scope and modes_out_of_scope \
                 (scope={scope:?})",
                m
            );
        }
        // Union = ALL:
        let mut all = in_scope.clone();
        all.extend_from_slice(&out_scope);
        all.sort_by_key(|m| m.depth());
        let expected: Vec<CertMode> = CertMode::ALL.to_vec();
        assert_eq!(
            all, expected,
            "in_scope + out_of_scope must cover all modes (scope={scope:?})"
        );
    }
}

/// `count()` and `is_empty()` agree with `modes_in_scope().len()`.
///
/// **Mutant-witness:** if `count` iterated only two booleans, it would return the wrong value for
/// scopes where the third mode contributes to the count.
#[test]
fn count_and_is_empty_agree_with_modes_in_scope() {
    for scope in [
        ModeScope::ALL_MODES,
        ModeScope::FAST_ONLY,
        ModeScope::NON_FAST,
        ModeScope::CERTIFIED_ONLY,
        ModeScope::BALANCED_ONLY,
        ModeScope {
            in_scope: [false, false, false],
        }, // empty scope
    ] {
        let expected_count = scope.modes_in_scope().len();
        assert_eq!(
            scope.count(),
            expected_count,
            "count() must equal modes_in_scope().len() for scope={scope:?}"
        );
        assert_eq!(
            scope.is_empty(),
            expected_count == 0,
            "is_empty() must equal (count == 0) for scope={scope:?}"
        );
    }
}

/// `union` is commutative and its membership is the OR of both scopes.
///
/// **Mutant-witness:** if `union` used AND instead of OR, the union of FAST_ONLY and
/// CERTIFIED_ONLY would be an empty scope instead of {Fast, Certified}.
#[test]
fn scope_union_is_commutative_and_correct() {
    let a = ModeScope::FAST_ONLY;
    let b = ModeScope::CERTIFIED_ONLY;
    let ab = a.union(b);
    let ba = b.union(a);

    // Commutative:
    assert_eq!(ab, ba, "union must be commutative");

    // Membership is OR:
    for &m in &CertMode::ALL {
        assert_eq!(
            ab.contains(m),
            a.contains(m) || b.contains(m),
            "union membership must be OR (mode={m:?})"
        );
    }
}

/// `intersect` is commutative and its membership is the AND of both scopes.
///
/// **Mutant-witness:** if `intersect` used OR instead of AND, the intersection of FAST_ONLY and
/// NON_FAST would be {Fast, Balanced, Certified} instead of empty.
#[test]
fn scope_intersect_is_commutative_and_correct() {
    let a = ModeScope::FAST_ONLY;
    let b = ModeScope::NON_FAST;
    let ab = a.intersect(b);
    let ba = b.intersect(a);

    // Commutative:
    assert_eq!(ab, ba, "intersect must be commutative");

    // Empty intersection (FAST_ONLY ∩ NON_FAST = ∅):
    assert!(ab.is_empty(), "FAST_ONLY ∩ NON_FAST must be empty");

    // NON_FAST ∩ ALL_MODES = NON_FAST:
    let non_fast_all = ModeScope::NON_FAST.intersect(ModeScope::ALL_MODES);
    assert_eq!(
        non_fast_all,
        ModeScope::NON_FAST,
        "NON_FAST ∩ ALL = NON_FAST"
    );
}

/// `from_resolved_mode` maps mode → scope via the "at-or-above depth" rule.
///
/// **Mutant-witness:** if `from_resolved_mode(Balanced)` returned `CERTIFIED_ONLY`, a test
/// intending to cover Balanced would silently miss Balanced, and `modes_in_scope()` would
/// return `[Certified]` instead of `[Balanced, Certified]`.
#[test]
fn from_resolved_mode_maps_mode_to_scope_correctly() {
    let fast_r = ResolvedMode {
        mode: CertMode::Fast,
        source: None,
    };
    let balanced_r = ResolvedMode {
        mode: CertMode::Balanced,
        source: Some(CertScope::Phylum),
    };
    let certified_r = ResolvedMode {
        mode: CertMode::Certified,
        source: Some(CertScope::Nodule),
    };

    assert_eq!(
        ModeScope::from_resolved_mode(&fast_r),
        ModeScope::ALL_MODES,
        "Fast → ALL_MODES (widest coverage)"
    );
    assert_eq!(
        ModeScope::from_resolved_mode(&balanced_r),
        ModeScope::NON_FAST,
        "Balanced → NON_FAST (Balanced + Certified)"
    );
    assert_eq!(
        ModeScope::from_resolved_mode(&certified_r),
        ModeScope::CERTIFIED_ONLY,
        "Certified → CERTIFIED_ONLY"
    );
}

// ---------------------------------------------------------------------------
// §2 ModeTestConfig + shared resolver
// ---------------------------------------------------------------------------

/// Default `ModeTestConfig` resolves to `ALL_MODES` (no declarations → Fast default → ALL_MODES).
///
/// **Mode-scope:** ALL_MODES (this is a config-correctness test, mode-independent).
/// **Mutant-witness:** if `ModeTestConfig::default()` had a non-empty `decls`, `resolve()` might
/// return a non-ALL_MODES scope, failing the assertion.
#[test]
fn mode_test_config_default_resolves_to_all_modes() {
    let config = ModeTestConfig::default();
    assert_eq!(
        config.resolve(),
        ModeScope::ALL_MODES,
        "default config (no decls) must resolve to ALL_MODES via the Fast default"
    );
    // Provenance: mode=Fast, source=None (no declaration matched).
    let prov = config.provenance();
    assert_eq!(prov.mode, CertMode::Fast, "default provenance must be Fast");
    assert_eq!(
        prov.source, None,
        "default provenance must have no source (built-in default)"
    );
}

/// Phylum-level declaration overrides the global default (most-specific-wins).
///
/// **Mutant-witness:** if `resolve_mode` picked the least-specific instead of most-specific,
/// a `Phylum` declaration would be ignored when a `Global` is also present — the scope would
/// stay at the Global level, and this test would fail (wrong scope returned).
#[test]
fn phylum_declaration_overrides_global_default() {
    let config = ModeTestConfig::new(&[
        CertDecl {
            scope: CertScope::Global,
            mode: CertMode::Fast,
        },
        CertDecl {
            scope: CertScope::Phylum,
            mode: CertMode::Balanced,
        },
    ]);
    let scope = config.resolve();
    let prov = config.provenance();

    // Phylum (more specific) wins over Global.
    assert_eq!(
        scope,
        ModeScope::NON_FAST,
        "Phylum=Balanced → NON_FAST scope"
    );
    assert_eq!(prov.mode, CertMode::Balanced);
    assert_eq!(
        prov.source,
        Some(CertScope::Phylum),
        "Phylum must win over Global"
    );
}

/// Nodule-level declaration overrides phylum-level (most-specific-wins).
///
/// **Mutant-witness:** if the resolver picked `Phylum` over `Nodule`, the scope would be
/// `NON_FAST` instead of `CERTIFIED_ONLY`, and this test would fail.
#[test]
fn nodule_declaration_overrides_phylum() {
    let config = ModeTestConfig::new(&[
        CertDecl {
            scope: CertScope::Phylum,
            mode: CertMode::Balanced,
        },
        CertDecl {
            scope: CertScope::Nodule,
            mode: CertMode::Certified,
        },
    ]);
    let scope = config.resolve();
    let prov = config.provenance();

    // Nodule (most specific) wins.
    assert_eq!(
        scope,
        ModeScope::CERTIFIED_ONLY,
        "Nodule=Certified → CERTIFIED_ONLY"
    );
    assert_eq!(prov.mode, CertMode::Certified);
    assert_eq!(
        prov.source,
        Some(CertScope::Nodule),
        "Nodule must win over Phylum"
    );
}

/// Granular override wins over all scope tiers (above the lattice).
///
/// **Mutant-witness:** if `with_granular` were ignored (granular `None` always), the config
/// with a Nodule=Certified declaration and a granular Fast override would resolve to
/// CERTIFIED_ONLY instead of ALL_MODES, and this test would fail.
#[test]
fn granular_override_wins_over_all_scope_tiers() {
    let config = ModeTestConfig::new(&[CertDecl {
        scope: CertScope::Nodule,
        mode: CertMode::Certified,
    }])
    .with_granular(CertMode::Fast);

    let scope = config.resolve();
    let prov = config.provenance();

    // Granular Fast → ALL_MODES, source=None (granular is not a CertScope level).
    assert_eq!(scope, ModeScope::ALL_MODES, "granular Fast → ALL_MODES");
    assert_eq!(
        prov.mode,
        CertMode::Fast,
        "granular mode must be reflected in provenance"
    );
    assert_eq!(
        prov.source, None,
        "granular override has no CertScope source"
    );
}

/// `resolve()` is deterministic (same config → same scope) and pure.
///
/// **Mode-scope:** ALL_MODES (mode-independent).
/// **Mutant-witness:** any non-determinism in the resolver would produce different values on
/// repeated calls, failing the equality check.
#[test]
fn resolve_is_deterministic() {
    let config = ModeTestConfig::new(&[CertDecl {
        scope: CertScope::Phylum,
        mode: CertMode::Balanced,
    }]);
    let s1 = config.resolve();
    let s2 = config.resolve();
    assert_eq!(
        s1, s2,
        "resolve() must be deterministic (same config → same scope)"
    );
}

/// Provenance is never ambient: calling `provenance()` always returns the effective mode + its
/// source scope (never `None` for the mode, and `source=None` means "project default" — a
/// meaningful sentinel, not an absent value — G2 / RFC-0012 renderability).
///
/// **Mutant-witness:** if `provenance()` returned an uninitialised struct with mode=Fast and
/// source=None even when a Phylum declaration won, this test would still pass — but the *wrong
/// test* would pass. The combination of provenance check + scope check is what catches the bug.
#[test]
fn provenance_is_never_ambient_g2() {
    // Case 1: no decls → default Fast, source=None.
    let default_prov = ModeTestConfig::default().provenance();
    assert_eq!(default_prov.mode, CertMode::Fast);
    assert_eq!(default_prov.source, None);

    // Case 2: Phylum declaration → Balanced, source=Some(Phylum).
    let phylum_prov = ModeTestConfig::new(&[CertDecl {
        scope: CertScope::Phylum,
        mode: CertMode::Balanced,
    }])
    .provenance();
    assert_eq!(
        phylum_prov.mode,
        CertMode::Balanced,
        "phylum mode must be Balanced"
    );
    assert_eq!(
        phylum_prov.source,
        Some(CertScope::Phylum),
        "phylum provenance must carry source=Phylum (never ambient — G2)"
    );

    // Case 3: granular override → Fast, source=None (granular tier is above the CertScope lattice).
    let granular_prov = ModeTestConfig::new(&[CertDecl {
        scope: CertScope::Nodule,
        mode: CertMode::Certified,
    }])
    .with_granular(CertMode::Fast)
    .provenance();
    assert_eq!(
        granular_prov.mode,
        CertMode::Fast,
        "granular Fast must be reflected"
    );
    // source=None for granular (not a CertScope level — FLAG-INJECT explains why).
    assert_eq!(
        granular_prov.source, None,
        "granular override carries source=None"
    );
}

// ---------------------------------------------------------------------------
// §3 for_each_mode / for_each_mode_in
// ---------------------------------------------------------------------------

/// `for_each_mode` visits all three modes in depth order (Fast → Balanced → Certified).
///
/// **Mutant-witness:** if `CertMode::ALL` omitted a variant (e.g. `[Fast, Certified]`), the
/// collected depths would be `[0, 2]`, failing the `assert_eq!` against `[0, 1, 2]`.
#[test]
fn for_each_mode_visits_all_three_in_depth_order() {
    let mut depths = Vec::new();
    for_each_mode(|mode| depths.push(mode.depth()));
    assert_eq!(
        depths,
        vec![0, 1, 2],
        "for_each_mode must yield Fast/Balanced/Certified in CertMode::ALL order"
    );
}

/// `for_each_mode` visits exactly three modes (not more, not fewer).
///
/// **Mutant-witness:** if `for_each_mode` short-circuited after the first mode, the count would
/// be 1 instead of 3.
#[test]
fn for_each_mode_visits_exactly_three_modes() {
    let mut count = 0usize;
    for_each_mode(|_mode| count += 1);
    assert_eq!(count, 3, "for_each_mode must visit exactly three modes");
}

/// `for_each_mode_in(ALL_MODES, …)` visits all three modes, skips none.
///
/// **Mutant-witness:** if `for_each_mode_in` filtered inversely (visited out-of-scope modes
/// instead of in-scope), the visited list would be empty for ALL_MODES.
#[test]
fn for_each_mode_in_all_modes_visits_all() {
    let visit = for_each_mode_in(ModeScope::ALL_MODES, |_mode| {});
    assert!(
        visit.visited_all(),
        "ALL_MODES scope must visit all three modes"
    );
    assert!(
        visit.skipped.is_empty(),
        "ALL_MODES scope must skip nothing"
    );
    assert!(visit.matches_scope(ModeScope::ALL_MODES));
}

/// `for_each_mode_in(FAST_ONLY, …)` visits only Fast, skips Balanced and Certified.
///
/// **Mutant-witness:** if `contains` used the wrong index (e.g. `in_scope[1]` for Fast), it
/// would skip Fast (in_scope[0]=true would be ignored), and the visited list would be empty.
#[test]
fn for_each_mode_in_fast_only_visits_only_fast() {
    let mut visited_modes = Vec::new();
    let visit = for_each_mode_in(ModeScope::FAST_ONLY, |mode| visited_modes.push(mode));

    assert_eq!(
        visited_modes,
        vec![CertMode::Fast],
        "FAST_ONLY must visit only Fast"
    );
    assert!(
        visit.skipped.contains(&CertMode::Balanced),
        "Balanced must be skipped"
    );
    assert!(
        visit.skipped.contains(&CertMode::Certified),
        "Certified must be skipped"
    );
    assert!(visit.matches_scope(ModeScope::FAST_ONLY));
}

/// `for_each_mode_in(CERTIFIED_ONLY, …)` visits only Certified.
///
/// **Mutant-witness:** if `for_each_mode_in` iterated in reverse order, `visited[0]` would be
/// Certified but `matches_scope` would still pass; the depth-order check in
/// `for_each_mode_visits_all_three_in_depth_order` would catch the reverse order.
#[test]
fn for_each_mode_in_certified_only_visits_only_certified() {
    let mut visited_modes = Vec::new();
    let visit = for_each_mode_in(ModeScope::CERTIFIED_ONLY, |mode| visited_modes.push(mode));

    assert_eq!(
        visited_modes,
        vec![CertMode::Certified],
        "CERTIFIED_ONLY must visit only Certified"
    );
    assert_eq!(visit.skipped.len(), 2, "must skip Fast and Balanced");
    assert!(visit.matches_scope(ModeScope::CERTIFIED_ONLY));
}

/// `for_each_mode_in` on an empty scope visits nothing and skips everything (never-silent — C1).
///
/// A test configured with an empty scope is surfaced explicitly (all three modes in `skipped`),
/// not silently skipped as if nothing happened.
///
/// **Mutant-witness:** if an empty scope were treated as ALL_MODES (an unsound default), the
/// `visit.visited.len()` would be 3 instead of 0.
#[test]
fn for_each_mode_in_empty_scope_visits_nothing_c1() {
    let empty_scope = ModeScope {
        in_scope: [false, false, false],
    };
    let mut visited_modes = Vec::new();
    let visit = for_each_mode_in(empty_scope, |mode| visited_modes.push(mode));

    assert!(
        visited_modes.is_empty(),
        "empty scope must visit no modes (C1 — the empty is surfaced, not silently all-on)"
    );
    assert_eq!(
        visit.skipped.len(),
        3,
        "empty scope must have all three modes in skipped (never-silent — C1/G2)"
    );
    assert!(
        !visit.visited_all(),
        "empty scope visit must not claim visited_all"
    );
}

/// `ModeVisit::visited_all` returns `true` only when all three modes were visited.
///
/// **Mutant-witness:** if `visited_all` compared `len() >= 1` instead of `== 3`, a single-mode
/// visit would incorrectly report `visited_all = true`.
#[test]
fn mode_visit_visited_all_requires_three_modes() {
    // All three: visited_all = true.
    let full = ModeVisit {
        visited: CertMode::ALL.to_vec(),
        skipped: vec![],
    };
    assert!(full.visited_all());

    // Two modes: visited_all = false.
    let partial = ModeVisit {
        visited: vec![CertMode::Balanced, CertMode::Certified],
        skipped: vec![CertMode::Fast],
    };
    assert!(!partial.visited_all());

    // Zero modes: visited_all = false.
    let empty = ModeVisit {
        visited: vec![],
        skipped: CertMode::ALL.to_vec(),
    };
    assert!(!empty.visited_all());
}

// ---------------------------------------------------------------------------
// §4 assert_mode_scope / assert_mode_negative
// ---------------------------------------------------------------------------

/// `assert_mode_scope` does NOT panic when the predicate exactly matches the scope.
///
/// **Mode-scope:** ALL_MODES (testing the harness itself — mode-independent).
/// **Mutant-witness:** if `assert_mode_scope` always panicked (unconditional panic), this test
/// would fail. If it never panicked (no-op), §4-panic tests below would fail.
#[test]
fn assert_mode_scope_succeeds_when_predicate_matches_scope() {
    // FAST_ONLY: predicate is true for Fast, false for Balanced/Certified.
    assert_mode_scope(
        ModeScope::FAST_ONLY,
        |mode| mode == CertMode::Fast,
        "fast-only predicate matches FAST_ONLY scope",
    );

    // NON_FAST: predicate is true for Balanced and Certified only.
    assert_mode_scope(
        ModeScope::NON_FAST,
        |mode| mode != CertMode::Fast,
        "non-fast predicate matches NON_FAST scope",
    );

    // ALL_MODES: predicate is always true.
    assert_mode_scope(
        ModeScope::ALL_MODES,
        |_mode| true,
        "always-true predicate matches ALL_MODES",
    );

    // CERTIFIED_ONLY: predicate is true for Certified only.
    assert_mode_scope(
        ModeScope::CERTIFIED_ONLY,
        |mode| mode == CertMode::Certified,
        "certified-only predicate matches CERTIFIED_ONLY scope",
    );
}

/// `assert_mode_scope` panics on the NEGATIVE direction (predicate holds outside scope).
///
/// The NEGATIVE arm catches the "invariant fires where it doesn't apply" defect — the primary
/// value of the cross-mode negative pattern (RFC-0034 §13).
///
/// **Mutant-witness:** if `assert_mode_scope` only checked the POSITIVE direction (predicate
/// must be true in scope, no check outside scope), an always-true predicate with FAST_ONLY scope
/// would not panic, and the test below would not catch the over-broad invariant.
#[test]
fn assert_mode_scope_panics_negative_direction_predicate_holds_outside_scope() {
    // FAST_ONLY scope: an always-true predicate holds in Balanced/Certified — must panic.
    let result = std::panic::catch_unwind(|| {
        assert_mode_scope(
            ModeScope::FAST_ONLY,
            |_mode| true,
            "always-true with FAST_ONLY — must panic (NEGATIVE arm)",
        );
    });
    assert!(
        result.is_err(),
        "assert_mode_scope must panic when predicate holds outside scope (NEGATIVE arm)"
    );
}

/// `assert_mode_scope` panics on the POSITIVE direction (predicate absent inside scope).
///
/// **Mutant-witness:** if `assert_mode_scope` only checked the NEGATIVE direction, an
/// always-false predicate with ALL_MODES scope would not panic (predicate is correctly absent
/// outside the "empty outside" set), silently passing a test that should fail.
#[test]
fn assert_mode_scope_panics_positive_direction_predicate_absent_inside_scope() {
    // ALL_MODES scope: an always-false predicate must panic (must hold for Fast, Balanced, Certified).
    let result = std::panic::catch_unwind(|| {
        assert_mode_scope(
            ModeScope::ALL_MODES,
            |_mode| false,
            "always-false with ALL_MODES — must panic (POSITIVE arm)",
        );
    });
    assert!(
        result.is_err(),
        "assert_mode_scope must panic when predicate is absent inside scope (POSITIVE arm)"
    );
}

/// `assert_mode_scope` panic message names the violating mode (EXPLAIN — C3).
///
/// **Mutant-witness:** if the panic message were empty or generic (no mode name), the test
/// below would fail the `contains("{mode:?}")` check for the known violating mode.
#[test]
fn assert_mode_scope_panic_message_names_violating_mode() {
    // FAST_ONLY scope: an always-true predicate violates the NEGATIVE arm for Balanced.
    // The panic message must mention "Balanced" (the first mode outside FAST_ONLY).
    let result = std::panic::catch_unwind(|| {
        assert_mode_scope(
            ModeScope::FAST_ONLY,
            |_mode| true,
            "always-true: must name violating mode in message",
        );
    });
    let err = result.unwrap_err();
    // Extract the panic message.
    let msg = if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else {
        panic!("panic payload was not a String or &str — cannot check message content");
    };
    // The panic must mention the violating mode (Balanced is the first out-of-scope mode for FAST_ONLY).
    assert!(
        msg.contains("Balanced"),
        "panic message must name the violating mode (Balanced); got: {msg:?}"
    );
    assert!(
        msg.contains("NEGATIVE"),
        "panic message must indicate the NEGATIVE direction; got: {msg:?}"
    );
}

/// `assert_mode_negative` does NOT panic when the predicate is absent outside scope.
///
/// **Mutant-witness:** if `assert_mode_negative` always panicked, this test would fail. If it
/// never panicked, the panic test below would fail.
#[test]
fn assert_mode_negative_succeeds_when_predicate_absent_outside_scope() {
    // FAST_ONLY scope: predicate false for Balanced and Certified — no panic.
    assert_mode_negative(
        ModeScope::FAST_ONLY,
        |mode| mode == CertMode::Fast, // true only in scope — not outside
        "predicate absent outside FAST_ONLY scope",
    );

    // CERTIFIED_ONLY: predicate false for Fast and Balanced.
    assert_mode_negative(
        ModeScope::CERTIFIED_ONLY,
        |mode| mode == CertMode::Certified, // true only in scope
        "predicate absent outside CERTIFIED_ONLY scope",
    );
}

/// `assert_mode_negative` panics when the predicate holds outside scope.
///
/// **Mutant-witness:** if `assert_mode_negative` never panicked (no-op), the always-true
/// predicate with FAST_ONLY scope would not catch the over-broad invariant.
#[test]
fn assert_mode_negative_panics_when_predicate_holds_outside_scope() {
    // FAST_ONLY scope: an always-true predicate holds in Balanced/Certified — must panic.
    let result = std::panic::catch_unwind(|| {
        assert_mode_negative(
            ModeScope::FAST_ONLY,
            |_mode| true,
            "always-true outside FAST_ONLY — must panic",
        );
    });
    assert!(
        result.is_err(),
        "assert_mode_negative must panic when predicate holds outside scope"
    );
}

/// `assert_mode_negative` does NOT check the positive direction (only the negative).
///
/// An always-false predicate with ALL_MODES scope passes `assert_mode_negative` (the predicate
/// is trivially absent outside every scope — there is no "outside" for ALL_MODES). This is
/// correct: `assert_mode_negative` only checks the "absent outside scope" invariant; use
/// `assert_mode_scope` for the full two-direction check.
///
/// **Mutant-witness:** if `assert_mode_negative` also checked the positive direction, an
/// always-false predicate with ALL_MODES scope would panic, and this test would fail.
#[test]
fn assert_mode_negative_does_not_check_positive_direction() {
    // ALL_MODES: there are no modes outside scope, so no negative to check. No panic.
    assert_mode_negative(
        ModeScope::ALL_MODES,
        |_mode| false, // predicate always false — fine for assert_mode_negative
        "always-false with ALL_MODES — negative check trivially passes (no outside modes)",
    );
}

// ---------------------------------------------------------------------------
// §5 Worked example / integration test
// ---------------------------------------------------------------------------

/// **Worked example** (RFC-0034 §13 ¶8): a downstream developer gets per-tier + negative
/// coverage with zero boilerplate using `ModeTestConfig` + `assert_mode_scope`.
///
/// Demonstrates the full developer workflow:
/// 1. Declare scope via `ModeTestConfig` (project/nodule/granular levels).
/// 2. Resolve scope → `ModeScope`.
/// 3. Assert the property across modes, including the negative cases.
/// 4. Inspect provenance (never-ambient — G2).
///
/// **Mode-scope:** ALL_MODES (the example exercises the harness across all tiers).
/// **Guarantee tag:** `Declared` — this is an illustration of the API surface, not a claim
/// about the property under test.
#[test]
fn worked_example_downstream_dev_zero_boilerplate_per_tier_plus_negative() {
    // --- Step 1: Declare scope. A nodule scoped to Certified ---
    let config = ModeTestConfig::new(&[
        CertDecl {
            scope: CertScope::Phylum,
            mode: CertMode::Fast,
        }, // project default
        CertDecl {
            scope: CertScope::Nodule,
            mode: CertMode::Certified,
        }, // nodule override
    ]);

    // --- Step 2: Resolve ---
    let scope = config.resolve();
    let prov = config.provenance();

    assert_eq!(
        scope,
        ModeScope::CERTIFIED_ONLY,
        "Nodule=Certified → CERTIFIED_ONLY"
    );
    assert_eq!(
        prov.mode,
        CertMode::Certified,
        "provenance mode = Certified"
    );
    assert_eq!(
        prov.source,
        Some(CertScope::Nodule),
        "nodule scope won (most-specific-wins)"
    );

    // --- Step 3: Assert with cross-mode negative (zero boilerplate — RFC-0034 §13 ¶8) ---
    // The property: "is this the Certified mode?" — holds only in Certified.
    assert_mode_scope(
        scope,
        |mode| mode == CertMode::Certified,
        "the property holds only in Certified",
    );

    // --- Step 4: The visit record shows which tiers ran + which were skipped (never-silent) ---
    let mut ran_in = Vec::new();
    let visit = for_each_mode_in(scope, |mode| ran_in.push(mode));

    assert_eq!(ran_in, vec![CertMode::Certified], "only Certified ran");
    assert!(
        visit.skipped.contains(&CertMode::Fast),
        "Fast was skipped (not in CERTIFIED_ONLY scope)"
    );
    assert!(
        visit.skipped.contains(&CertMode::Balanced),
        "Balanced was skipped (not in CERTIFIED_ONLY scope)"
    );
    assert!(
        visit.matches_scope(scope),
        "visit must match the declared scope"
    );
}

/// **Worked example: nodule scope + granular override** showing the full three-tier dial.
///
/// A test author sets a Nodule-level Balanced default, then overrides a single test to Fast.
///
/// **Mutant-witness:** if `with_granular` were ignored, the scope would stay NON_FAST, and
/// `scope == ModeScope::ALL_MODES` would fail.
#[test]
fn worked_example_granular_override_wins_over_nodule() {
    // Nodule default: Balanced.
    let nodule_config = ModeTestConfig::new(&[CertDecl {
        scope: CertScope::Nodule,
        mode: CertMode::Balanced,
    }]);
    assert_eq!(
        nodule_config.resolve(),
        ModeScope::NON_FAST,
        "Nodule=Balanced → NON_FAST (Balanced + Certified)"
    );

    // Granular per-test override to Fast → ALL_MODES.
    let granular_config = nodule_config.with_granular(CertMode::Fast);
    let scope = granular_config.resolve();
    assert_eq!(scope, ModeScope::ALL_MODES, "granular Fast → ALL_MODES");

    // Provenance: granular wins, source=None (not a CertScope level).
    let prov = granular_config.provenance();
    assert_eq!(prov.mode, CertMode::Fast);
    assert_eq!(
        prov.source, None,
        "granular override has no CertScope source"
    );

    // The actual per-tier sweep runs across all modes.
    let mut visited = Vec::new();
    for_each_mode_in(scope, |mode| visited.push(mode));
    assert_eq!(
        visited,
        CertMode::ALL.to_vec(),
        "all three modes must run with ALL_MODES scope"
    );
}

/// **Worked example: negative-only helper** showing that `assert_mode_negative` is the right
/// tool when the positive check is handled by a separate mode-pinned assertion.
///
/// **Mutant-witness:** if `assert_mode_negative` were a no-op, the out-of-scope violation would
/// go undetected (the positive assertion in `assert_mode_scope` would catch it instead — but
/// `assert_mode_negative` is deliberately lighter, so the test below verifies it independently).
#[test]
fn worked_example_negative_only_helper() {
    // Property: "a value produced in Fast is not tagged Empirical" — this holds FAST_ONLY in the
    // sense that the absence of Empirical is guaranteed only in Fast (Balanced/Certified can
    // produce Empirical). We want to assert the NEGATIVE: in NON_FAST, the absence is NOT
    // guaranteed — i.e., the "no Empirical" property should NOT hold in Balanced/Certified.

    // The "no Empirical" property is absent in NON_FAST: assert_mode_negative(FAST_ONLY, ...)
    // checks that the property does not hold outside FAST_ONLY.
    assert_mode_negative(
        ModeScope::FAST_ONLY,
        // Simulated: in Balanced/Certified, the "Empirical absent" predicate is false
        // (Empirical IS reachable there). So this predicate returns false for non-Fast modes.
        |mode| mode == CertMode::Fast,
        "the 'no Empirical' property is absent in NON_FAST (Empirical is reachable there)",
    );
    // No panic → correct: the predicate is false for Balanced and Certified (not holding outside
    // FAST_ONLY scope).
}
