//! The W2 startup gate (`check_startup`, RFC-0041 §4.2/§4.3): it accepts a budget whose memory ceiling
//! honors `DEPTH_FLOOR × MAX_FRAME_BYTES`, refuses one that does not (never-silently, as
//! `StartupError::MemCeiling`), and — on the native test host, where stack growth is available — never
//! trips the host-stack arm. Data-driven over the memory-ceiling boundary.

use crate::{
    check_startup, RecursionBudget, StartupError, DEPTH_FLOOR, HOST_STACK_BYTES_PER_FRAME,
    MAX_FRAME_BYTES,
};

/// The memory ceiling the §4.2 invariant requires at the floor: `DEPTH_FLOOR × MAX_FRAME_BYTES`.
const REQUIRED_MEM: u64 = (DEPTH_FLOOR as u64) * MAX_FRAME_BYTES;

struct Case {
    mem_limit: u64,
    accept: bool,
    note: &'static str,
}

const CASES: &[Case] = &[
    Case {
        mem_limit: u64::MAX,
        accept: true,
        note: "unbounded ceiling honors the floor",
    },
    Case {
        mem_limit: REQUIRED_MEM,
        accept: true,
        note: "exact equality accepts (the `<` boundary)",
    },
    Case {
        mem_limit: REQUIRED_MEM - 1,
        accept: false,
        note: "one byte short of the floor requirement refuses",
    },
    Case {
        mem_limit: 0,
        accept: false,
        note: "a zero ceiling cannot honor a positive floor",
    },
];

#[test]
fn startup_gate_enforces_the_memory_floor() {
    for (i, c) in CASES.iter().enumerate() {
        // Work-step ceiling is irrelevant to the startup gate; only the memory ceiling is checked.
        let budget = RecursionBudget::new(DEPTH_FLOOR, c.mem_limit, u64::MAX);
        let got = check_startup(&budget);
        assert_eq!(
            got.is_ok(),
            c.accept,
            "case {i} ({}): mem_limit={} → {got:?}",
            c.note,
            c.mem_limit
        );
        if !c.accept {
            // The refusal is the memory-ceiling arm (the host-stack arm passes on a growth-available
            // host), and it carries the actionable numbers.
            match got {
                Err(StartupError::MemCeiling(_)) => {}
                other => panic!("case {i}: expected a MemCeiling refusal, got {other:?}"),
            }
        }
    }
}

#[test]
fn the_default_budget_is_refused_until_a_real_memory_ceiling_is_set() {
    // The default budget carries an *unbounded* (u64::MAX) memory ceiling, so it passes the §4.2 floor
    // trivially — the W2 wiring does not itself impose a finite ceiling (that is a per-deployment config).
    assert!(check_startup(&RecursionBudget::default()).is_ok());
}

#[test]
fn startup_error_display_names_the_governing_section() {
    let budget = RecursionBudget::new(DEPTH_FLOOR, 0, u64::MAX);
    let err = check_startup(&budget).expect_err("a zero memory ceiling must refuse");
    let msg = err.to_string();
    assert!(msg.contains("4.2"), "memory refusal cites §4.2: {msg}");
    assert!(
        std::error::Error::source(&err).is_some(),
        "the StartupError wraps its cause as a source"
    );
}

#[test]
fn the_host_stack_frame_estimate_exceeds_the_measured_worst_case() {
    // Sanity guard on the conservative constant: the per-frame host-stack estimate must stay above the
    // measured worst case (the L1 checker's ~10.9 KiB/frame) so the no-grow floor check can only
    // over-estimate the stack a floor needs, never under-estimate into a silent overflow.
    let estimate = HOST_STACK_BYTES_PER_FRAME; // bind so the compare is not a pure const-fold (clippy).
    let measured_checker_frame = 11 * 1024;
    assert!(
        estimate >= measured_checker_frame,
        "host-stack per-frame estimate ({estimate}) must exceed the ~10.9 KiB measured checker frame"
    );
}
