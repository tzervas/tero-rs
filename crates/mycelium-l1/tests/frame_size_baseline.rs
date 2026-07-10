//! **Frame-size CI baseline (RFC-0041 §4.2).** Pins `size_of` of the execution machines' value/frame
//! structs against [`mycelium_workstack::MAX_FRAME_BYTES`] so a toolchain/IR change that grows a frame
//! **fails CI, not production** — the determinism invariant (`mem_ceiling ≥ DEPTH_FLOOR × MAX_FRAME_BYTES`)
//! is only meaningful if `MAX_FRAME_BYTES` actually bounds the real structs (the ADR-041 lesson: a bump
//! must trip a gate, not surface as a runtime accept/reject shift).
//!
//! This mirrors the `reject_ledger.rs` pinned-count pattern: a data-driven table of `(name, measured
//! size)` asserted `<= MAX_FRAME_BYTES`, with a failure message telling the maintainer to **re-measure
//! all three machines and bump the baseline** on an *intended* frame-size change.
//!
//! ## Coverage note (transparency — G2)
//!
//! This gate is a test target, so it may depend on the UPWARD machine crates that the workstack *library*
//! must not (DN-68 downward-only holds for the lib; a dev-dependency back-edge is a permitted cargo
//! cycle). It pins the **publicly-reachable** value structs directly: the interp/L0 `CoreValue`/`Node`
//! (`mycelium-core`) and the L1 `L1Value` (`mycelium-l1`). The **AOT env-machine `Frame`/`AotVal`/`Env`
//! (`mycelium-mlir::aot`) are private**, so they cannot be measured from here; the current measured AOT
//! `Frame` (328 bytes, 64-bit) is the machine that *sets* `MAX_FRAME_BYTES` and must be pinned by an
//! in-crate baseline test inside `mycelium-mlir` (FLAGged to the integrator). The `<=` assertion is
//! deliberate (not `==`): narrower targets (32-bit) yield smaller structs and still pass; only genuine
//! growth past the baseline trips.

use mycelium_workstack::MAX_FRAME_BYTES;

/// The publicly-reachable per-machine value/frame structs and their measured `size_of`, pinned so that
/// a change to any of them that pushes it past [`MAX_FRAME_BYTES`] fails this gate.
fn measured_frames() -> Vec<(&'static str, usize)> {
    vec![
        // interp / L0 normal forms (the substitution + env machines share these).
        (
            "mycelium_core::CoreValue",
            size_of::<mycelium_core::CoreValue>(),
        ),
        ("mycelium_core::Node", size_of::<mycelium_core::Node>()),
        // L1 evaluator value.
        (
            "mycelium_l1::eval::L1Value",
            size_of::<mycelium_l1::eval::L1Value>(),
        ),
    ]
}

#[test]
fn frame_sizes_stay_within_the_pinned_baseline() {
    let max = MAX_FRAME_BYTES as usize;
    let mut over: Vec<String> = Vec::new();
    for (name, size) in measured_frames() {
        if size > max {
            over.push(format!("  {name}: {size} bytes > MAX_FRAME_BYTES ({max})"));
        }
    }
    assert!(
        over.is_empty(),
        "RFC-0041 §4.2 frame-size baseline exceeded — a machine's value/frame struct grew past the \
         pinned MAX_FRAME_BYTES:\n{}\n\nThis is intentional ONLY if you meant to grow a frame. If so: \
         (1) re-measure size_of for ALL three machines (interp/L0 CoreValue+Node, L1 L1Value, AND the \
         private AOT `Frame`/`AotVal` via mycelium-mlir's in-crate baseline test), (2) bump \
         `mycelium_workstack::MAX_FRAME_BYTES` to the new max, and (3) re-check the determinism \
         invariant `mem_ceiling >= DEPTH_FLOOR * MAX_FRAME_BYTES` still holds for configured ceilings. \
         Otherwise, shrink the frame (box a large variant) — an unintended frame-size bump would shift \
         the memory-ceiling accept/reject boundary (the ADR-041 lesson this gate exists to catch).",
        over.join("\n")
    );
}

#[test]
fn the_baseline_is_not_slack_below_the_measured_public_max() {
    // Guard against MAX_FRAME_BYTES drifting *far* above the real structs (which would make the §4.2
    // invariant demand an unnecessarily large memory ceiling). The public value structs are ~240 bytes
    // and the baseline (currently 384, set by the private AOT Frame at 328) should stay within a small
    // multiple of the largest measured public struct — a loose sanity bound, not a tight pin.
    let public_max = measured_frames()
        .into_iter()
        .map(|(_, s)| s)
        .max()
        .expect("at least one measured frame");
    assert!(
        MAX_FRAME_BYTES as usize <= public_max * 4,
        "MAX_FRAME_BYTES ({MAX_FRAME_BYTES}) is far above the largest measured public frame \
         ({public_max}); if the AOT Frame did not grow this may be stale — re-measure and tighten."
    );
}
