//! The fine-grained runtime-gated grow (RFC-0041 §4.3): it recurses past a deliberately small base
//! stack on a growth-available target, the availability probe is honest, and the no-grow floor refusal
//! (`check_floor`, white-box) is never-silent — simulated with a tiny fixed ceiling so it is exercised
//! on any host (a native test host always reports growth *available*, so the public probe never hits
//! the refusal branch there).

use crate::{check_floor, grow, stack_growth_available, StackError, NO_GROW_CEILING_BYTES};

#[test]
fn grow_recurses_past_a_small_base_stack() {
    // A 256 KiB worker stack that this recursion (~20k levels × a 1 KiB pad) would overflow WITHOUT
    // growing. Calling `grow` at each recursion point lets `stacker` allocate fresh segments on demand,
    // so it completes. Only meaningful where growth is available; on a no-grow target the startup guard
    // would refuse first, so we return the same sentinel rather than assert a false pass.
    fn descend(n: u64, pad: &[u8; 1024]) -> u64 {
        grow(|| {
            if n == 0 {
                u64::from(pad[0])
            } else {
                descend(n - 1, pad).wrapping_add(1)
            }
        })
    }
    let handle = std::thread::Builder::new()
        .name("small-base-stack".to_owned())
        .stack_size(256 * 1024)
        .spawn(|| {
            if stack_growth_available() {
                descend(20_000, &[9u8; 1024])
            } else {
                20_009 // no-grow target: the guard refuses before this would run; match the sentinel.
            }
        })
        .expect("spawn small-base-stack worker");
    // descend bottoms out at pad[0] == 9, then 20_000 increments → 20_009.
    assert_eq!(handle.join().expect("worker joins"), 20_009);
}

#[test]
fn the_growth_probe_is_honest_about_this_host() {
    // The probe must reflect what `stacker` can actually do on this target, not a baked-in constant.
    let available = stack_growth_available();
    assert_eq!(
        available,
        stacker::remaining_stack().is_some(),
        "the probe must agree with stacker's own remaining_stack signal"
    );
    // Native CI hosts (the only hosts this suite runs on) support growth — a regression to `false`
    // would mean the probe or the dependency broke.
    assert!(
        available,
        "a native host must report stack growth available"
    );
}

/// White-box cases over the pure `check_floor` decision, straddling the no-grow boundary. The public
/// `growable_ceiling_honors_floor` can only reach the `growth_available = true` arm on a native host,
/// so the refusal path is exercised here by simulating `growth_available = false` with a small ceiling.
struct FloorCase {
    growth_available: bool,
    ceiling: u64,
    floor: u32,
    frame: u64,
    accept: bool,
}

const FLOOR_CASES: &[FloorCase] = &[
    // Growth available ⇒ always OK, even when the fixed ceiling could never hold the floor (growth,
    // bounded by the depth budget, covers it).
    FloorCase {
        growth_available: true,
        ceiling: 1,
        floor: u32::MAX,
        frame: u64::MAX,
        accept: true,
    },
    // No-grow, realistic wasm ceiling (1 MiB) vs the default 4096 floor at ~16 KiB/frame ⇒ refuse.
    FloorCase {
        growth_available: false,
        ceiling: NO_GROW_CEILING_BYTES,
        floor: 4096,
        frame: 16 * 1024,
        accept: false,
    },
    // No-grow, exact equality (ceiling == floor*frame) ⇒ accept (the `<` boundary, not `<=`).
    FloorCase {
        growth_available: false,
        ceiling: 4096 * 16,
        floor: 4096,
        frame: 16,
        accept: true,
    },
    // No-grow, one byte short of equality ⇒ refuse.
    FloorCase {
        growth_available: false,
        ceiling: 4096 * 16 - 1,
        floor: 4096,
        frame: 16,
        accept: false,
    },
    // No-grow, product overflows u64 → saturates to u64::MAX > any sub-max ceiling ⇒ refuse (confirms
    // saturation, not a wrap to a small value that would spuriously accept).
    FloorCase {
        growth_available: false,
        ceiling: 1_000_000,
        floor: 2,
        frame: u64::MAX,
        accept: false,
    },
    // No-grow, floor 0 ⇒ required 0 ⇒ accept.
    FloorCase {
        growth_available: false,
        ceiling: 0,
        floor: 0,
        frame: u64::MAX,
        accept: true,
    },
];

#[test]
fn no_grow_floor_check_refuses_below_the_ceiling_and_accepts_at_or_above() {
    for (i, c) in FLOOR_CASES.iter().enumerate() {
        let got = check_floor(c.growth_available, c.ceiling, c.floor, c.frame);
        assert_eq!(
            got.is_ok(),
            c.accept,
            "case {i}: growth={} ceiling={} floor={} frame={} → {got:?}",
            c.growth_available,
            c.ceiling,
            c.floor,
            c.frame
        );
        if !c.accept {
            let StackError::FloorUnsatisfiableOnNoGrowTarget {
                floor,
                stack_bytes_per_frame,
                required,
                ceiling,
            } = got.expect_err("a rejection is an error");
            let expect_required = u64::from(c.floor).saturating_mul(c.frame);
            assert_eq!(floor, c.floor, "case {i}: floor echoed");
            assert_eq!(stack_bytes_per_frame, c.frame, "case {i}: frame echoed");
            assert_eq!(
                required, expect_required,
                "case {i}: required is the saturating floor*frame product"
            );
            assert_eq!(
                ceiling, c.ceiling,
                "case {i}: ceiling is the one under test"
            );
            assert!(
                ceiling < required,
                "case {i}: a rejection means ceiling < required"
            );
        }
    }
}

#[test]
fn the_public_floor_check_passes_on_a_growth_available_host() {
    // On the native test host growth is available, so the public entry (which reads the real probe and
    // the real NO_GROW_CEILING_BYTES) returns Ok even for a floor no fixed stack could hold.
    assert!(crate::growable_ceiling_honors_floor(u32::MAX, u64::MAX).is_ok());
}

#[test]
fn the_stack_error_display_is_actionable() {
    let err = check_floor(false, NO_GROW_CEILING_BYTES, 4096, 16 * 1024)
        .expect_err("this floor is unsatisfiable on a no-grow target");
    let msg = err.to_string();
    assert!(msg.contains("no-grow"), "names the no-grow cause: {msg}");
    assert!(
        msg.contains("Refusing to start"),
        "states the refusal: {msg}"
    );
}
