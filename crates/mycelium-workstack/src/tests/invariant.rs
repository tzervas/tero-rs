//! The §4.2 determinism invariant `assert_mem_ceiling_honors_floor`: accepts `mem_ceiling >= floor *
//! max_frame` and rejects `<`, never-silently. Data-driven over the boundary, with overflow saturation.

use crate::{assert_mem_ceiling_honors_floor, BudgetError, BudgetKind};

struct Case {
    mem_limit: u64,
    depth_floor: u32,
    max_frame: u64,
    accept: bool,
}

/// Cases straddling the `mem_ceiling >= floor * max_frame` boundary, including the exact-equality point
/// (must accept — mutant witness for `<` vs `<=`) and the overflow case (product saturates to `u64::MAX`).
const CASES: &[Case] = &[
    // floor * frame = 4096 * 1024 = 4_194_304
    Case {
        mem_limit: 4_194_304,
        depth_floor: 4096,
        max_frame: 1024,
        accept: true,
    }, // exact equality
    Case {
        mem_limit: 4_194_305,
        depth_floor: 4096,
        max_frame: 1024,
        accept: true,
    }, // above
    Case {
        mem_limit: 4_194_303,
        depth_floor: 4096,
        max_frame: 1024,
        accept: false,
    }, // one short
    Case {
        mem_limit: 0,
        depth_floor: 1,
        max_frame: 1,
        accept: false,
    },
    Case {
        mem_limit: 1,
        depth_floor: 1,
        max_frame: 1,
        accept: true,
    },
    Case {
        mem_limit: u64::MAX,
        depth_floor: 0,
        max_frame: u64::MAX,
        accept: true,
    }, // floor 0 → 0 required
    // Overflow: 2 * u64::MAX saturates to u64::MAX; a sub-max ceiling cannot honor it → reject
    // (and confirms the product saturates rather than wrapping to a small value that would accept).
    Case {
        mem_limit: 1_000_000,
        depth_floor: 2,
        max_frame: u64::MAX,
        accept: false,
    },
    // Saturating equality: 2 * u64::MAX saturates to exactly the u64::MAX ceiling → accept (`>=`).
    Case {
        mem_limit: u64::MAX,
        depth_floor: 2,
        max_frame: u64::MAX,
        accept: true,
    },
];

#[test]
fn honors_floor_accepts_at_or_above_and_rejects_below() {
    for (i, c) in CASES.iter().enumerate() {
        let got = assert_mem_ceiling_honors_floor(c.mem_limit, c.depth_floor, c.max_frame);
        assert_eq!(
            got.is_ok(),
            c.accept,
            "case {i}: mem={} floor={} frame={} → {got:?}",
            c.mem_limit,
            c.depth_floor,
            c.max_frame
        );
        if !c.accept {
            match got {
                Err(BudgetError::OutOfBudget {
                    kind: BudgetKind::Bytes,
                    limit,
                    requested,
                }) => {
                    assert_eq!(
                        limit, c.mem_limit,
                        "case {i}: limit is the ceiling under test"
                    );
                    let expect_required = u64::from(c.depth_floor).saturating_mul(c.max_frame);
                    assert_eq!(
                        requested, expect_required,
                        "case {i}: requested is the (saturating) floor*frame product"
                    );
                    assert!(
                        limit < requested,
                        "case {i}: a rejection means ceiling < required"
                    );
                }
                other => panic!("case {i}: a rejection must be OutOfBudget(Bytes), got {other:?}"),
            }
        }
    }
}
