//! Tests for [`crate::policy_mech`] (M-963; DN-78 §3 B-1/B-2).
//!
//! Fixture-driven: policies come from [`policy`]/[`inputs`] builders and the property test's
//! generated cases; test bodies assert over a case (M-797 layout).

use crate::policy_mech::{
    capture, replay, CaptureError, PolicySite, PolicySlot, ReplayError, SlotError,
};
use mycelium_core::{GuaranteeStrength, Repr};
use mycelium_select::{
    Action, Candidate, CostModel, PolicyRegistry, Predicate, Rule, SelectionInputs, SelectionPolicy,
};
use proptest::prelude::*;

/// A validated test policy over `Binary{width}` candidates.
fn policy(name: &str, widths: &[u32], rules: Vec<Rule>, default_choice: usize) -> SelectionPolicy {
    SelectionPolicy::new(
        name,
        widths
            .iter()
            .map(|w| Candidate::Repr(Repr::Binary { width: *w }))
            .collect(),
        rules,
        default_choice,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect("test policy must validate")
}

/// Swap/packing-shaped inputs (no decode facts) over a `Binary` source.
fn inputs(width: u32) -> SelectionInputs {
    SelectionInputs {
        src: Repr::Binary { width },
        guarantee: GuaranteeStrength::Exact,
        bound: None,
        sparsity: None,
        decode: None,
    }
}

// ── B-2: the reified setter surface ──

#[test]
fn set_appends_one_record_with_monotonic_seq_and_previous_chain() {
    let mut slot = PolicySlot::new(PolicySite::SwapTarget);
    let a = policy("a", &[8], vec![], 0);
    let b = policy("b", &[8, 16], vec![], 1);
    let a_ref = a.policy_ref();
    let b_ref = b.policy_ref();

    let first = slot.set(a).clone();
    assert_eq!(first.seq, 0);
    assert_eq!(first.previous, None, "first set has no previous policy");
    assert_eq!(first.new_policy, a_ref);
    assert_eq!(first.site, PolicySite::SwapTarget);

    let second = slot.set(b).clone();
    assert_eq!(second.seq, 1, "seq is per-slot monotonic");
    assert_eq!(
        second.previous,
        Some(a_ref),
        "the transition records the outgoing policy — never a silent override (G2)"
    );
    assert_eq!(second.new_policy, b_ref);

    assert_eq!(slot.transitions().len(), 2, "exactly one record per set");
    assert_eq!(
        slot.active().expect("policy b is active").policy_ref(),
        b_ref
    );
    // Mutant witness: dropping the record push, or reusing seq 0, fails here.
}

#[test]
fn select_without_active_policy_refuses_explicitly() {
    let mut slot = PolicySlot::new(PolicySite::Placement);
    let err = slot
        .select(&inputs(8), None)
        .expect_err("an unset slot must refuse, never silently default (G2/ADR-006)");
    assert_eq!(
        err,
        SlotError::NoActivePolicy {
            site: PolicySite::Placement
        }
    );
    let msg = err.to_string();
    assert!(
        msg.contains("placement") && msg.contains("no silent default"),
        "the refusal must teach: site + no-silent-default; got: {msg}"
    );
}

#[test]
fn select_through_slot_records_trace() {
    let mut slot = PolicySlot::new(PolicySite::Packing);
    let p = policy("trace", &[8, 16], vec![], 0);
    let p_ref = p.policy_ref();
    slot.set(p);

    let (_, e1) = slot.select(&inputs(8), None).expect("selection succeeds");
    let (_, e2) = slot.select(&inputs(16), None).expect("selection succeeds");

    assert_eq!(slot.trace().len(), 2, "one Explanation per selection");
    assert_eq!(slot.trace()[0], e1);
    assert_eq!(slot.trace()[1], e2);
    assert!(
        slot.trace().iter().all(|e| e.policy == p_ref),
        "every trace entry names the policy that decided (RFC-0005 §3 provenance)"
    );
}

// ── B-1: capture and replay ──

#[test]
fn capture_unknown_ref_refuses() {
    let mut slot = PolicySlot::new(PolicySite::SwapTarget);
    let p = policy("unregistered", &[8], vec![], 0);
    let p_ref = p.policy_ref();
    slot.set(p);
    let (_, record) = slot.select(&inputs(8), None).expect("selection succeeds");

    let empty = PolicyRegistry::new();
    let err = capture(&empty, &record)
        .expect_err("capture must refuse an unknown ref, never reconstruct (G2)");
    assert_eq!(err, CaptureError::UnknownPolicyRef { policy_ref: p_ref });
}

#[test]
fn capture_round_trip_replay_matches() {
    let mut registry = PolicyRegistry::new();
    let p = policy(
        "round-trip",
        &[8, 16, 32],
        vec![Rule {
            when: Predicate::Always,
            action: Action::Cheapest,
        }],
        2,
    );
    registry.register(p.clone());

    let mut slot = PolicySlot::new(PolicySite::SwapTarget);
    slot.set(p);
    let (_, recorded) = slot.select(&inputs(8), None).expect("selection succeeds");

    let captured = capture(&registry, &recorded).expect("registered ref must capture");
    assert_eq!(
        captured.policy.policy_ref(),
        captured.policy_ref,
        "the captured value is the policy the record names — checked, not assumed"
    );
    let replayed = replay(&captured, &recorded).expect("replay must reach the recorded decision");
    assert_eq!(replayed, recorded, "replay reproduces the full Explanation");
}

#[test]
fn replay_honors_recorded_override() {
    let mut registry = PolicyRegistry::new();
    // Cheapest would pick index 0 (8 bits); force index 1 so the override path is exercised.
    let p = policy("override", &[8, 16], vec![], 0);
    registry.register(p.clone());

    let mut slot = PolicySlot::new(PolicySite::Packing);
    slot.set(p);
    let (_, recorded) = slot
        .select(&inputs(8), Some(1))
        .expect("in-range override succeeds");
    assert!(recorded.overridden, "the override is recorded first-class");

    let captured = capture(&registry, &recorded).expect("capture succeeds");
    let replayed = replay(&captured, &recorded).expect("replay re-applies the recorded override");
    assert_eq!(replayed, recorded);
}

#[test]
fn replay_against_wrong_policy_refuses() {
    let mut registry = PolicyRegistry::new();
    let a = policy("policy-a", &[8], vec![], 0);
    let b = policy("policy-b", &[8, 16], vec![], 1);
    registry.register(a.clone());
    registry.register(b.clone());

    let mut slot = PolicySlot::new(PolicySite::SwapTarget);
    slot.set(a);
    let (_, recorded_by_a) = slot.select(&inputs(8), None).expect("selection succeeds");

    let captured_b = crate::policy_mech::CapturedPolicy {
        policy_ref: b.policy_ref(),
        policy: b,
    };
    let err = replay(&captured_b, &recorded_by_a)
        .expect_err("replaying a record against a different policy must refuse up front");
    assert!(
        matches!(err, ReplayError::PolicyMismatch { .. }),
        "expected PolicyMismatch, got {err:?}"
    );
}

#[test]
fn replay_divergence_is_explicit_not_silent() {
    let mut registry = PolicyRegistry::new();
    let p = policy("diverge", &[8, 16], vec![], 0);
    registry.register(p.clone());

    let mut slot = PolicySlot::new(PolicySite::SwapTarget);
    slot.set(p);
    let (_, mut recorded) = slot.select(&inputs(8), None).expect("selection succeeds");

    // Tamper with the record (a record from different code / a corrupted store): the replay
    // recomputes the true decision and must surface the difference, never absorb it.
    recorded.chosen_index = 1;
    recorded.chosen = Candidate::Repr(Repr::Binary { width: 16 });

    let captured = capture(&registry, &recorded).expect("capture succeeds");
    let err = replay(&captured, &recorded).expect_err("a diverging record must be surfaced (G2)");
    match err {
        ReplayError::Diverged { recorded, replayed } => {
            assert_eq!(recorded.chosen_index, 1, "the tampered record is carried");
            assert_eq!(replayed.chosen_index, 0, "the true decision is carried");
        }
        other => panic!("expected Diverged, got {other:?}"),
    }
}

// ── The record-vs-replay differential (property test; the `Empirical` basis for the
//    "Policy capture replay reaches the recorded decision" matrix row — VR-5) ──

proptest! {
    #[test]
    fn prop_capture_replay_differential(
        // 1..=4 candidate widths, distinct-enough for a real choice space.
        widths in proptest::collection::vec(1u32..512, 1..4),
        default_ix in 0usize..4,
        use_cheapest_rule in any::<bool>(),
        input_width in 1u32..512,
        force in proptest::option::of(0usize..4),
    ) {
        let default_choice = default_ix % widths.len();
        let rules = if use_cheapest_rule {
            vec![Rule { when: Predicate::Always, action: Action::Cheapest }]
        } else {
            vec![]
        };
        let p = policy("prop", &widths, rules, default_choice);

        let mut registry = PolicyRegistry::new();
        registry.register(p.clone());

        let mut slot = PolicySlot::new(PolicySite::SwapTarget);
        slot.set(p);

        let forced = force.map(|f| f % widths.len());
        let (_, recorded) = slot
            .select(&inputs(input_width), forced)
            .expect("in-range (possibly forced) selection on a validated policy succeeds");

        let captured = capture(&registry, &recorded).expect("registered ref captures");
        let replayed = replay(&captured, &recorded)
            .expect("replay must reach the recorded decision (record-vs-replay differential)");
        prop_assert_eq!(replayed, recorded);
    }
}
