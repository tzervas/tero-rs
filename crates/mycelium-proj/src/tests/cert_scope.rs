//! White-box + property tests for [`crate::cert_scope`] — the M-790 and M-792 DoD laws
//! (RFC-0034 §6 / §3.1 / §7 / §13d).
//!
//! M-790 DoD property tests (existing):
//! 1. **Resolution precedence** — a `nodule` declaration overrides a `phylum` one, which overrides a
//!    `global` one (most-specific-wins), for *any* combination of modes and *any* declaration order.
//! 2. **Cross-mode never-silent-upgrade** — composing a value produced under one mode into a
//!    computation under any other mode never upgrades the value's guarantee strength (VR-5); a `fast`
//!    value entering a `certified` computation is always an explicit, visible boundary event.
//!
//! M-792 DoD property tests (new — RFC-0034 §7 / §13d):
//! 3. **EXPLAIN of the active mode is available in every mode** — `explain_mode` produces a non-empty,
//!    mode-naming string for any `ResolvedMode`, including the `fast` default (RFC-0034 §13d).
//! 4. **Signal generated even when consumption is lean** — `generate_mode_signal` succeeds in every
//!    mode; dialing `render_mode_signal` to a higher `ConsumptionTier` surfaces more of the *already-
//!    captured* history (no re-run or mode switch — RFC-0034 §7).

use crate::cert_scope::*;
use mycelium_core::cert_mode::CertMode;
use mycelium_core::guarantee::GuaranteeStrength;
use proptest::prelude::*;

/// Strategy over the three consumption tiers.
fn any_consumption_tier() -> impl Strategy<Value = ConsumptionTier> {
    prop::sample::select(ConsumptionTier::ALL.to_vec())
}

/// Strategy over the three certification modes.
fn any_mode() -> impl Strategy<Value = CertMode> {
    prop::sample::select(CertMode::ALL.to_vec())
}

/// Strategy over the three scopes.
fn any_scope() -> impl Strategy<Value = CertScope> {
    prop::sample::select(CertScope::ALL.to_vec())
}

/// Strategy over the four guarantee strengths.
fn any_strength() -> impl Strategy<Value = GuaranteeStrength> {
    prop::sample::select(GuaranteeStrength::ALL.to_vec())
}

// --- deterministic unit checks (the named, load-bearing cases) ---

#[test]
fn no_declarations_resolves_to_the_default() {
    let r = resolve_mode(&[]);
    assert_eq!(r, ResolvedMode::defaulted());
    assert_eq!(r.mode, CertMode::Fast);
    assert_eq!(r.source, None);
}

#[test]
fn parse_and_word_round_trip() {
    for mode in CertMode::ALL {
        assert_eq!(parse_cert_mode(cert_mode_word(mode)).unwrap(), mode);
    }
}

#[test]
fn parse_rejects_an_unknown_mode_word() {
    // Never-silent (G2): an out-of-set word is an explicit error, not a guess.
    let e = parse_cert_mode("turbo").unwrap_err();
    assert!(e.contains("unknown @certification mode"), "{e}");
    // The serde-capitalized spelling is *not* the surface spelling (FLAG-A) — also rejected.
    assert!(parse_cert_mode("Fast").is_err());
}

#[test]
fn scope_specificity_is_global_lt_phylum_lt_nodule() {
    assert!(CertScope::Global.specificity() < CertScope::Phylum.specificity());
    assert!(CertScope::Phylum.specificity() < CertScope::Nodule.specificity());
}

#[test]
fn fast_into_certified_is_an_explicit_boundary_never_an_upgrade() {
    // The concrete DoD case: a `fast` value (Empirical-intended) entering a `certified` computation.
    let ev = compose(
        CertMode::Fast,
        CertMode::Certified,
        GuaranteeStrength::Empirical,
    );
    // It is a visible boundary (producer ran less certification than consumer)…
    assert!(ev.is_boundary());
    // …and the value did NOT inherit a stronger guarantee — `Fast` floors Empirical to Declared.
    assert_eq!(ev.effective, GuaranteeStrength::Declared);
    assert!(!ev.upgraded_strength());
}

#[test]
fn structural_exact_survives_the_boundary() {
    // A structural `Exact` (e.g. a bijective swap) is not floored even under `Fast` (it earned the
    // strength structurally) — composing it never *upgrades*, and it stays Exact.
    let ev = compose(
        CertMode::Fast,
        CertMode::Certified,
        GuaranteeStrength::Exact,
    );
    assert_eq!(ev.effective, GuaranteeStrength::Exact);
    assert!(!ev.upgraded_strength());
}

// --- M-792 unit checks: EXPLAIN-of-mode + generation≠consumption (RFC-0034 §7 / §13d) ---

#[test]
fn explain_mode_is_available_in_every_mode_including_fast() {
    // RFC-0034 §13d: EXPLAIN of the active mode is always available.
    // The fast default (no declarations) must produce a non-empty, mode-naming output.
    let fast_default = ResolvedMode::defaulted();
    let ex = explain_mode(&fast_default);
    assert!(
        !ex.is_empty(),
        "explain_mode must produce output in fast/default"
    );
    assert!(ex.contains("fast"), "fast default must name 'fast': {ex}");
    assert!(
        ex.contains("default"),
        "fast default must name source 'default': {ex}"
    );

    // All three modes, all three scopes — every combination must explain.
    for mode in CertMode::ALL {
        for scope in CertScope::ALL {
            let r = ResolvedMode {
                mode,
                source: Some(scope),
            };
            let ex = explain_mode(&r);
            let mode_word = cert_mode_word(mode);
            assert!(
                ex.contains(mode_word),
                "explain_mode missing mode word for {:?}: {ex}",
                mode
            );
            assert!(
                ex.contains(scope.label()),
                "explain_mode missing scope label for {:?}: {ex}",
                scope
            );
        }
    }
}

#[test]
fn generate_mode_signal_is_available_in_every_mode() {
    // Generation is always-on — the signal is captured for any ResolvedMode.
    for mode in CertMode::ALL {
        let r = ResolvedMode::defaulted();
        let mut rm = r;
        rm.mode = mode;
        let sig = generate_mode_signal(&rm);
        assert_eq!(sig.resolved, rm);
        assert_eq!(sig.depth, mode.depth());
    }
}

#[test]
fn lean_consumption_is_identical_to_explain_mode() {
    // The Lean render is the EXPLAIN floor — it must produce the same text as explain_mode
    // (RFC-0034 §7: lean = "one compact line", which is exactly what explain_mode already is).
    // Mutant witness: if `render_mode_signal` with Lean diverged from `explain_mode`, this test
    // fails, revealing the contract break.
    for mode in CertMode::ALL {
        let r = ResolvedMode { mode, source: None };
        let sig = generate_mode_signal(&r);
        assert_eq!(
            render_mode_signal(&sig, ConsumptionTier::Lean),
            explain_mode(&r),
            "Lean render must equal explain_mode for {:?}",
            mode
        );
    }
}

#[test]
fn dialing_consumption_up_surfaces_more_without_rerun() {
    // The DoD invariant: a higher ConsumptionTier renders more information from the *same*
    // already-captured ModeSignal — no new generation step needed.
    // Concrete case: a fast default signal, rendered at Lean vs Medium vs Full.
    let r = ResolvedMode::defaulted();
    let sig = generate_mode_signal(&r);

    let lean = render_mode_signal(&sig, ConsumptionTier::Lean);
    let medium = render_mode_signal(&sig, ConsumptionTier::Medium);
    let full = render_mode_signal(&sig, ConsumptionTier::Full);

    // Lean ⊆ Medium ⊆ Full (medium/full are strictly longer and contain the lean prefix).
    // Mutant witness: swapping Lean/Full output makes this length check fail.
    assert!(
        medium.len() > lean.len(),
        "medium render must be longer than lean for the same signal"
    );
    assert!(
        full.len() > medium.len(),
        "full render must be longer than medium for the same signal"
    );
    // Lean prefix is preserved — the history is not replaced, only augmented.
    assert!(
        medium.starts_with(&lean),
        "medium render must start with lean prefix — same captured history, more surfaced"
    );
    // Full contains the generation≠consumption note (the already-captured marker).
    assert!(
        full.contains("already captured"),
        "full render must surface the already-captured note: {full}"
    );
}

#[test]
fn consumption_tier_ordering_is_lean_lt_medium_lt_full() {
    assert!(ConsumptionTier::Lean < ConsumptionTier::Medium);
    assert!(ConsumptionTier::Medium < ConsumptionTier::Full);
    assert!(ConsumptionTier::Full.is_at_least(ConsumptionTier::Lean));
    assert!(ConsumptionTier::Full.is_at_least(ConsumptionTier::Full));
    assert!(!ConsumptionTier::Lean.is_at_least(ConsumptionTier::Medium));
}

// --- DoD property tests ---

proptest! {
    /// DoD #1 — **resolution precedence**: with declarations present at a set of scopes, the resolved
    /// mode is exactly the one declared at the most-specific scope, regardless of declaration order.
    #[test]
    fn prop_most_specific_scope_wins(
        global in any_mode(),
        phylum in any_mode(),
        nodule in any_mode(),
        // Which scopes actually carry a declaration (at least one, else the law is "default").
        has_global in any::<bool>(),
        has_phylum in any::<bool>(),
        has_nodule in any::<bool>(),
    ) {
        let mut decls = Vec::new();
        if has_global { decls.push(CertDecl { scope: CertScope::Global, mode: global }); }
        if has_phylum { decls.push(CertDecl { scope: CertScope::Phylum, mode: phylum }); }
        if has_nodule { decls.push(CertDecl { scope: CertScope::Nodule, mode: nodule }); }

        let r = resolve_mode(&decls);

        // Expected winner: nodule > phylum > global; none ⇒ default.
        let (exp_mode, exp_src) = if has_nodule {
            (nodule, Some(CertScope::Nodule))
        } else if has_phylum {
            (phylum, Some(CertScope::Phylum))
        } else if has_global {
            (global, Some(CertScope::Global))
        } else {
            (CertMode::default(), None)
        };
        prop_assert_eq!(r.mode, exp_mode);
        prop_assert_eq!(r.source, exp_src);
    }

    /// DoD #1 (order-independence) — resolution picks by specificity, not by position: shuffling the
    /// declaration vector cannot change the result.
    #[test]
    fn prop_resolution_is_order_independent(
        a in any_scope(), ma in any_mode(),
        b in any_scope(), mb in any_mode(),
    ) {
        // Two declarations at (possibly distinct) scopes; if scopes collide, drop one (the parser
        // forbids two declarations at the same scope).
        let mut forward = vec![CertDecl { scope: a, mode: ma }];
        if b != a { forward.push(CertDecl { scope: b, mode: mb }); }
        let mut backward = forward.clone();
        backward.reverse();
        prop_assert_eq!(resolve_mode(&forward), resolve_mode(&backward));
    }

    /// DoD #2 — **cross-mode composition never silently upgrades**: for *any* producer/consumer modes
    /// and *any* incoming strength, the effective strength after crossing is never stronger than what
    /// the value came in with (VR-5), and an up-crossing is always a visible boundary event.
    #[test]
    fn prop_cross_mode_never_upgrades_strength(
        producer in any_mode(),
        consumer in any_mode(),
        incoming in any_strength(),
    ) {
        let ev = compose(producer, consumer, incoming);
        // Never an upgrade: the effective strength's rank is >= the incoming rank (weaker-or-equal).
        prop_assert!(!ev.upgraded_strength(),
            "compose({:?},{:?},{:?}) upgraded to {:?}", producer, consumer, incoming, ev.effective);
        prop_assert!(ev.effective.rank() >= incoming.rank());
        // The effective strength is floored by the PRODUCER's mode, not the consumer's — the value
        // keeps only the strength its own mode established.
        prop_assert_eq!(ev.effective, producer.gate_guarantee(incoming));
        // An up-crossing (producer weaker-certified than consumer) is flagged as a boundary.
        prop_assert_eq!(ev.is_boundary(), producer.depth() < consumer.depth());
    }

    /// M-792 DoD #3 — **EXPLAIN of the active mode is available in every mode** (RFC-0034 §13d):
    /// for *any* `ResolvedMode` (any combination of mode and optional source scope), `explain_mode`
    /// produces a non-empty string that names both the mode word and the source label.
    #[test]
    fn prop_explain_mode_available_in_every_mode(
        mode in any_mode(),
        scope in any_scope(),
        has_source in any::<bool>(),
    ) {
        let r = ResolvedMode {
            mode,
            source: if has_source { Some(scope) } else { None },
        };
        let ex = explain_mode(&r);
        // Non-empty and names the mode (RFC-0034 §13d — never ambient, G2).
        prop_assert!(!ex.is_empty());
        prop_assert!(ex.contains(cert_mode_word(mode)),
            "explain_mode did not name mode {:?}: {ex}", mode);
        // The source scope (or "default") is always named — never ambient.
        let expected_src = r.source.map_or("default", CertScope::label);
        prop_assert!(ex.contains(expected_src),
            "explain_mode did not name source {expected_src:?}: {ex}");
    }

    /// M-792 DoD #4 — **signal is always generated; dialing consumption up surfaces more**
    /// (RFC-0034 §7): for *any* `ResolvedMode` and *any* pair of consumption tiers where
    /// `higher >= lower`, `render_mode_signal` with the higher tier produces output that is
    /// at least as long as the lower tier — the already-captured history is only augmented,
    /// never discarded when consumption increases.
    #[test]
    fn prop_signal_generated_and_consumption_monotone(
        mode in any_mode(),
        scope in any_scope(),
        has_source in any::<bool>(),
        lower in any_consumption_tier(),
        higher in any_consumption_tier(),
    ) {
        let r = ResolvedMode {
            mode,
            source: if has_source { Some(scope) } else { None },
        };
        // Signal is generated for any mode — no mode-gating, no failure (RFC-0034 §7).
        let sig = generate_mode_signal(&r);
        // The signal captures the resolved mode faithfully.
        prop_assert_eq!(sig.resolved, r);
        prop_assert_eq!(sig.depth, mode.depth());

        // Consumption monotonicity: dialing up produces at least as many bytes.
        // (A higher tier can only add information, never remove it.)
        let output_lower = render_mode_signal(&sig, lower);
        let output_higher = render_mode_signal(&sig, higher);
        if higher >= lower {
            prop_assert!(
                output_higher.len() >= output_lower.len(),
                "render with {:?} ({} chars) shorter than {:?} ({} chars) — \
                 dialing up must never reduce output length",
                higher, output_higher.len(), lower, output_lower.len()
            );
        }

        // Lean output always names the mode and source — the EXPLAIN floor holds at every mode.
        let lean = render_mode_signal(&sig, ConsumptionTier::Lean);
        prop_assert!(lean.contains(cert_mode_word(mode)),
            "Lean render did not name mode {:?}: {lean}", mode);
        let expected_src = r.source.map_or("default", CertScope::label);
        prop_assert!(lean.contains(expected_src),
            "Lean render did not name source {expected_src:?}: {lean}");
    }
}
