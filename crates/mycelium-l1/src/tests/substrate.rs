//! White-box tests for the `Substrate` v0 value form (M-902; DN-71 Model S §4.1), its M-903
//! runtime use-once backstop (DN-71 §4.2), and its M-904 execution + drop posture (DN-71 §4.3/§8
//! FLAG-4).
//!
//! Covers the M-902 Definition of Done: creation (acquisition + provenance), passage (round-trip
//! through the evaluator's value-binding machinery), inspection (tag/id/provenance/EXPLAIN); the
//! M-903 `try_consume` runtime backstop (first-consume succeeds, a second consume — even through a
//! separate clone of the same identity — traps explicitly, never silently); the M-904 identity-move
//! execution (`consume` now runs end-to-end, and a runtime double-consume that evades the static
//! pass still traps, never silently); and the M-904 drop-without-consume v0 posture (a live,
//! abandoned handle is deterministically released and the release recorded, while a handle that
//! escapes — directly or nested in a constructed value — is never wrongly released).

use std::collections::BTreeMap;

use mycelium_core::DataRegistry;

use crate::checkty::{check_nodule, Env};
use crate::eval::{Evaluator, L1Error, L1Value};
use crate::parse;
use crate::substrate::{SubstrateError, SubstrateHandle, SubstrateProvenance};

fn env(src: &str) -> Env {
    check_nodule(&parse(src).expect("parses")).expect("checks")
}

fn a_handle(tag: &str) -> SubstrateHandle {
    SubstrateHandle::acquire(tag, SubstrateProvenance::new("wild:open", "test-fixture"))
}

// --- creation + inspection ---------------------------------------------------------------

#[test]
fn acquire_records_tag_and_provenance() {
    let h = a_handle("Socket");
    assert_eq!(h.tag(), "Socket");
    assert_eq!(h.provenance().acquired_via, "wild:open");
    assert_eq!(h.provenance().site, "test-fixture");
}

#[test]
fn explain_is_inspectable_and_names_the_provenance() {
    // House rule 2 (no black boxes): the EXPLAIN string surfaces tag + identity + provenance.
    let h = a_handle("Socket");
    let e = h.explain();
    assert!(e.contains("Substrate{Socket}"), "got: {e}");
    assert!(e.contains(&format!("#{}", h.id())), "got: {e}");
    assert!(e.contains("wild:open"), "got: {e}");
    assert!(e.contains("test-fixture"), "got: {e}");
}

#[test]
fn each_acquisition_has_a_distinct_identity() {
    // Identity is the external resource, not its content: two acquisitions of the "same" tag are
    // two distinct handles (DN-71 §4.1 — ADR-003 content-addressing does not apply).
    let a = a_handle("Socket");
    let b = a_handle("Socket");
    assert_ne!(
        a.id(),
        b.id(),
        "distinct acquisitions must have distinct ids"
    );
    assert_ne!(a, b, "distinct-identity handles must not be equal");
}

#[test]
fn clone_preserves_identity_the_passage_semantics() {
    // Cloning is *passage* (same resource), not a second acquisition: the id is preserved, so a
    // cloned handle is equal to its source. This is the mechanism the evaluator uses to pass a
    // Substrate through a binding; affinity is a checker property, not a non-Clone Rust bound.
    let a = a_handle("Socket");
    let b = a.clone();
    assert_eq!(a.id(), b.id());
    assert_eq!(a, b);
}

// --- L1Value discrimination --------------------------------------------------------------

#[test]
fn as_substrate_and_as_repr_are_discriminated() {
    let h = a_handle("Socket");
    let sv = L1Value::Substrate(h.clone());
    // A Substrate value inspects as a handle and has no repr (never-silent None).
    assert_eq!(sv.as_substrate(), Some(&h));
    assert!(sv.as_repr().is_none());
    // A non-Substrate value has no handle (never-silent None). A data value stands in for "other".
    let dv = L1Value::Data {
        ty: "Unit".into(),
        ctor: "Unit".into(),
        fields: std::sync::Arc::new(vec![]),
    };
    assert!(dv.as_substrate().is_none());
}

#[test]
fn substrate_has_no_l0_core_projection() {
    // A Substrate handle is not a kernel value (no Repr, no L0 node), so it has no CoreValue
    // projection — `to_core` is honestly `None`, never a fabricated lowering (DN-71 §4.1; G2).
    let env = env("nodule d;");
    let registry = DataRegistry::build(&BTreeMap::new()).expect("empty registry builds");
    let sv = L1Value::Substrate(a_handle("Socket"));
    assert!(sv.to_core(&env, &registry).is_none());
}

// --- passage through the evaluator (create -> pass -> inspect round-trip) -----------------

#[test]
fn substrate_passes_through_the_evaluator_unchanged() {
    // A passthrough fn binds a Substrate param and returns it (no `consume`): the handle rides the
    // ordinary value-binding machinery and comes back out identical (same tag + id + provenance).
    let env = env("nodule d;\nfn passthrough(s: Substrate{Res}) => Substrate{Res} = s;");
    let h = a_handle("Res");
    let out = Evaluator::new(&env)
        .call("passthrough", vec![L1Value::Substrate(h.clone())])
        .expect("passthrough evaluates");
    let got = out.as_substrate().expect("result is a Substrate handle");
    assert_eq!(got, &h, "the handle round-trips unchanged (same identity)");
}

// --- M-903: the runtime use-once backstop -------------------------------------------------

#[test]
fn try_consume_succeeds_once_and_marks_the_handle_consumed() {
    // The first move (DN-71 §4.2) succeeds and flips the shared `consumed` flag.
    let h = a_handle("Socket");
    assert!(!h.is_consumed(), "a freshly acquired handle starts live");
    let moved = h.try_consume().expect("first consume succeeds");
    assert_eq!(moved.tag(), "Socket");
    assert_eq!(moved.id(), h.id(), "the move preserves identity");
    assert!(h.is_consumed(), "the original observes the shared flag too");
    assert!(moved.is_consumed());
}

#[test]
fn try_consume_traps_a_double_consume_naming_the_tag_and_id() {
    // A second `try_consume` on the same identity is the never-silent runtime backstop tripping
    // (M-903; DN-71 §4.2) — it names both the tag and the id of the violated handle, never a silent
    // no-op or a fabricated second move (G2/VR-5).
    let h = a_handle("Socket");
    h.try_consume().expect("first consume succeeds");
    let err = h
        .try_consume()
        .expect_err("a second consume of the same identity must trap");
    assert_eq!(
        err,
        SubstrateError::AlreadyConsumed {
            tag: "Socket".into(),
            id: h.id(),
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("double-consume"), "got: {msg}");
    assert!(msg.contains("Socket"), "must name the tag: {msg}");
    assert!(msg.contains(&h.id().to_string()), "must name the id: {msg}");
    assert!(msg.contains("M-903"), "must cite the backstop: {msg}");
}

#[test]
fn the_consumed_flag_is_shared_across_clones_the_backstop_cannot_be_dodged() {
    // Cloning is passage, not re-acquisition (the type doc's contract) — consuming through ONE
    // clone must be visible through every other clone of the same identity, so the backstop cannot
    // be evaded by cloning before the second (illegitimate) consume.
    let a = a_handle("Socket");
    let b = a.clone();
    b.try_consume().expect("consume via the clone succeeds");
    assert!(a.is_consumed(), "the flag is shared, not per-clone");
    let err = a
        .try_consume()
        .expect_err("consuming the other clone must also trap");
    assert!(matches!(err, SubstrateError::AlreadyConsumed { .. }));
}

#[test]
fn distinct_identities_are_consumed_independently() {
    // Two separate acquisitions (distinct ids) have independent `consumed` flags — consuming one
    // never affects the other (identity, not tag, is what the backstop keys on).
    let a = a_handle("Socket");
    let b = a_handle("Socket");
    a.try_consume().expect("consuming `a` succeeds");
    assert!(a.is_consumed());
    assert!(!b.is_consumed(), "a distinct identity stays live");
    b.try_consume()
        .expect("consuming `b` independently succeeds");
}

// --- M-904: `consume` now executes (the identity-move lowering, DN-71 §4.3) ---------------

#[test]
fn surface_consume_executes_and_marks_the_handle_consumed() {
    // `consume s` type-checks (M-664 surface), its affine discipline is statically checked (M-903),
    // and it now **executes** as the identity-move (M-904; DN-71 §4.3): the evaluator returns the
    // same identity, now consumed — never a refusal, never a silent no-op.
    let env = env("nodule d;\nfn take(s: Substrate{Sock}) => Substrate{Sock} = consume s;");
    let h = a_handle("Sock");
    let out = Evaluator::new(&env)
        .call("take", vec![L1Value::Substrate(h.clone())])
        .expect("`consume` now executes end-to-end (M-904)");
    let got = out.as_substrate().expect("result is a Substrate handle");
    assert_eq!(got.id(), h.id(), "the move preserves identity");
    assert!(got.is_consumed(), "the moved-out handle is now consumed");
    assert!(
        h.is_consumed(),
        "the shared flag is visible through the original too"
    );
}

#[test]
fn a_full_acquire_use_consume_program_evaluates_through_the_real_interpreter() {
    // The M-904 DoD's end-to-end accept case: acquire (stood in by the Rust-level `acquire` — v0 has
    // no surface acquisition op yet, DN-71 §4.1) -> use (an intervening `let` binds and passes it
    // along unconsumed) -> consume, all evaluated by the real `Evaluator`, not a residual/refusal.
    let env = env("nodule d;\n\
         fn take(s: Substrate{Sock}) => Substrate{Sock} = let t = s in consume t;");
    let h = a_handle("Sock");
    let out = Evaluator::new(&env)
        .call("take", vec![L1Value::Substrate(h.clone())])
        .expect("acquire -> use -> consume runs end-to-end");
    assert_eq!(out.as_substrate().unwrap().id(), h.id());
    assert!(out.as_substrate().unwrap().is_consumed());
    // The intervening `let t = s in …` binding escaped into `consume t`'s result (it's the same
    // identity returned), so no scope-exit release fires for it — only the `consume` did the work.
    assert!(
        Evaluator::new(&env).release_events().is_empty(),
        "a fresh evaluator's own release log starts empty"
    );
}

#[test]
fn a_runtime_double_consume_that_evades_the_static_pass_still_traps_never_silently() {
    // The static pass (M-903) refuses double-consume at *check* time (see `tests/affine.rs`), so a
    // checker-accepted program never reaches this at runtime. This test exercises the **backstop**
    // directly (M-903's own escape hatch — a closure/loop-multiplicity gap `crate::affine`'s docs
    // name) by calling `take` twice with *clones of the same identity*: each individual call
    // type-checks (each call site only ever moves its own argument once), but the second call's
    // `consume` hits an already-consumed handle at runtime — an `L1Error::Stuck` (an evaluation state
    // the checker proves unreachable for a single well-typed program), never a silent second move
    // (G2/VR-5; DN-71 §4.2 backstop, wired into execution by M-904).
    let env = env("nodule d;\nfn take(s: Substrate{Sock}) => Substrate{Sock} = consume s;");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    ev.call("take", vec![L1Value::Substrate(h.clone())])
        .expect("first consume succeeds");
    let err = ev
        .call("take", vec![L1Value::Substrate(h.clone())])
        .expect_err("a second consume of the same identity must trap at runtime");
    let L1Error::Stuck { why, .. } = err else {
        panic!("expected Stuck (the runtime backstop), got {err:?}");
    };
    assert!(why.contains("double-consume"), "got: {why}");
    assert!(why.contains("M-903"), "must cite the backstop: {why}");
}

// --- M-904: the drop-without-consume v0 posture (DN-71 §8 FLAG-4) --------------------------

#[test]
fn a_never_consumed_parameter_is_released_at_scope_exit_and_the_release_is_recorded() {
    // `f` never references `s` at all — it is abandoned at the end of `f`'s own call frame. The v0
    // drop posture (accepted 2026-07-02) releases it deterministically and records the event: never
    // a silent leak (G2).
    let env = env("nodule d;\ntype Unit = Unit;\nfn f(s: Substrate{Sock}) => Unit = Unit;");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    ev.call("f", vec![L1Value::Substrate(h.clone())])
        .expect("f evaluates (dropping s is not a checker or runtime error in v0)");
    assert!(
        h.is_consumed(),
        "the abandoned handle is released (terminal)"
    );
    let events = ev.release_events();
    assert_eq!(events.len(), 1, "exactly one release event: {events:?}");
    assert_eq!(events[0].tag, "Sock");
    assert_eq!(events[0].id, h.id());
    let msg = events[0].to_string();
    assert!(msg.contains("released"), "got: {msg}");
    assert!(
        msg.contains("never `consume`d") || msg.contains("FLAG-4"),
        "got: {msg}"
    );
}

#[test]
fn a_tail_call_from_a_fn_with_a_substrate_param_is_not_tco_and_still_releases() {
    // RFC-0041 §4.6 (M-979) TCO precondition witness. A live `Substrate` parameter is PENDING
    // post-work (its scope-exit release runs *after* the body), so a tail call from such a fn must
    // NOT reuse its frame — else the release (+ `ReleaseEvent`) is silently skipped, a handle leak.
    // `f` takes a `Substrate` and its body is a direct TAIL call to `g`; because `f` has a substrate
    // param it is not TCO-eligible, its frame is kept, and the abandoned `s` is still released and
    // recorded. (If the precondition ignored substrate params, `f`'s frame would be elided at the
    // `g()` tail call and `s` would leak → zero release events.)
    let env = env("nodule d;\ntype Unit = Unit;\nfn g() => Unit = Unit;\n\
         fn f(s: Substrate{Sock}) => Unit = g();");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    ev.call("f", vec![L1Value::Substrate(h.clone())])
        .expect("f evaluates");
    assert!(
        h.is_consumed(),
        "the abandoned substrate param is released despite the tail call to g"
    );
    let events = ev.release_events();
    assert_eq!(
        events.len(),
        1,
        "the ReleaseEvent still fires — TCO must not elide a substrate-param frame: {events:?}"
    );
    assert_eq!(events[0].id, h.id());
}

#[test]
fn a_let_bound_substrate_abandoned_before_the_body_ends_is_released_at_its_own_scope_exit() {
    // `let t = s in True` — `t` is bound, never used, and its own `let` scope ends before `f`
    // returns. The release must fire at `t`'s own scope exit (inside the `let`), not merely at the
    // outer call boundary — both are wired (M-904), but this pins the inner one specifically.
    let env = env("nodule d;\nfn f(s: Substrate{Sock}) => Bool = let t = s in True;");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    ev.call("f", vec![L1Value::Substrate(h.clone())])
        .expect("f evaluates");
    assert!(
        h.is_consumed(),
        "the let-bound-and-abandoned handle is released"
    );
    let events = ev.release_events();
    assert_eq!(events.len(), 1, "exactly one release event: {events:?}");
    assert_eq!(events[0].id, h.id());
}

#[test]
fn a_substrate_returned_unconsumed_is_not_released_it_escapes() {
    // Plain passthrough (no `consume`): `s` is *returned*, not abandoned, so releasing it would
    // destroy a value the caller still legitimately holds live. The escape check must see through
    // the direct return.
    let env = env("nodule d;\nfn passthrough(s: Substrate{Sock}) => Substrate{Sock} = s;");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    let out = ev
        .call("passthrough", vec![L1Value::Substrate(h.clone())])
        .expect("passthrough evaluates");
    assert!(
        !out.as_substrate().unwrap().is_consumed(),
        "the returned handle must still be live — it escaped, it was not abandoned"
    );
    assert!(
        ev.release_events().is_empty(),
        "no release event for a value that escaped via return"
    );
}

#[test]
fn a_substrate_nested_in_a_returned_constructor_is_not_released_it_escapes_deeply() {
    // The escape check must see through a constructed value, not just a bare direct return — `Mk(s)`
    // returns `s` nested one level deep.
    let env = env("nodule d;\ntype Box = Mk(Substrate{Sock});\n\
         fn wrap(s: Substrate{Sock}) => Box = Mk(s);");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    let out = ev
        .call("wrap", vec![L1Value::Substrate(h.clone())])
        .expect("wrap evaluates");
    let L1Value::Data { ref fields, .. } = out else {
        panic!("expected a Data value");
    };
    let inner = fields[0].as_substrate().expect("field is the Substrate");
    assert!(
        !inner.is_consumed(),
        "a handle nested in the returned value must still be live"
    );
    assert!(
        ev.release_events().is_empty(),
        "no release event for a value that escaped nested in a constructor"
    );
}

#[test]
fn an_explicitly_consumed_substrate_never_also_gets_a_release_event() {
    // Consuming and releasing are the same terminal transition (KC-3 — no third state); an explicit
    // `consume` must not ALSO generate a spurious release event.
    let env = env("nodule d;\nfn take(s: Substrate{Sock}) => Substrate{Sock} = consume s;");
    let h = a_handle("Sock");
    let ev = Evaluator::new(&env);
    ev.call("take", vec![L1Value::Substrate(h.clone())])
        .expect("consume succeeds");
    assert!(
        ev.release_events().is_empty(),
        "an explicit consume must not also be logged as a release: {:?}",
        ev.release_events()
    );
}

#[test]
fn release_reuses_the_terminal_state_a_released_handle_cannot_be_consumed_afterward() {
    // `release` and `try_consume` share the same terminal flag (KC-3, substrate.rs docs): once
    // released, a subsequent `try_consume` on the same identity traps exactly like a double-consume,
    // never silently succeeding on an already-abandoned handle.
    let h = a_handle("Sock");
    let event = h.release("test-site").expect("a live handle releases once");
    assert_eq!(event.tag, "Sock");
    assert_eq!(event.id, h.id());
    assert!(h.is_consumed());
    let err = h
        .try_consume()
        .expect_err("consuming an already-released handle must trap");
    assert!(matches!(err, SubstrateError::AlreadyConsumed { .. }));
    // Releasing an already-terminal handle again is a legitimate no-op — `None`, not a duplicate
    // event and not an error.
    assert!(h.release("test-site-2").is_none());
}

#[test]
fn guarantee_index_on_a_substrate_is_refused() {
    // A Substrate carries no Meta/guarantee tag, so a guarantee-index assertion on it is an explicit
    // refusal (never a silently-passed assertion — VR-5/G2).
    let env = env("nodule d;");
    let err = Evaluator::new(&env)
        .assert_guarantee(
            "site",
            &L1Value::Substrate(a_handle("Socket")),
            crate::ast::Strength::Exact,
        )
        .expect_err("a guarantee index on a Substrate must be refused");
    assert!(matches!(err, L1Error::Unsupported { .. }), "got {err:?}");
}
