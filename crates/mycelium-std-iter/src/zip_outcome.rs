//! [`ZipOutcome`] — the EXPLAIN artifact for `zip` (C3 / spec §4 / Q1).
//!
//! `zip` truncates to the shorter spine (the never-silent floor, C1). The [`ZipOutcome`] makes
//! the truncation point *inspectable*: a caller can always determine exactly which side was
//! longer and how many elements were dropped, without parsing the output (C3 no black boxes).

/// Records the outcome of a [`zip`](crate::zip) call — specifically, which side (if any) was
/// truncated and by how many elements.
///
/// # Guarantee tag: `Exact`
/// The lengths are exact measurements; no approximation.
///
/// # C3 (EXPLAIN): the truncation is never silent
/// `was_truncated()` → whether any elements were dropped.
/// `left_len()`, `right_len()`, `result_len()` → the exact counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZipOutcome {
    left: usize,
    right: usize,
    result: usize,
}

impl ZipOutcome {
    /// Construct from the three lengths. `result` = `left.min(right)` by construction.
    #[must_use]
    pub(crate) fn new(left: usize, right: usize, result: usize) -> Self {
        ZipOutcome {
            left,
            right,
            result,
        }
    }

    /// The number of elements in the left input.
    #[must_use]
    pub fn left_len(&self) -> usize {
        self.left
    }

    /// The number of elements in the right input.
    #[must_use]
    pub fn right_len(&self) -> usize {
        self.right
    }

    /// The number of pairs produced (= `min(left_len, right_len)`).
    #[must_use]
    pub fn result_len(&self) -> usize {
        self.result
    }

    /// `true` if the two inputs had different lengths (some elements were dropped).
    #[must_use]
    pub fn was_truncated(&self) -> bool {
        self.left != self.right
    }

    /// The number of elements dropped from the left side (0 if left was the shorter or equal).
    #[must_use]
    pub fn left_excess(&self) -> usize {
        self.left.saturating_sub(self.result)
    }

    /// The number of elements dropped from the right side (0 if right was the shorter or equal).
    #[must_use]
    pub fn right_excess(&self) -> usize {
        self.right.saturating_sub(self.result)
    }
}
