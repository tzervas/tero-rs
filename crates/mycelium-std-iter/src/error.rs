//! Error types for `std.iter` — all fallible ops return these explicitly (C1 never-silent).

use core::fmt;

/// Error returned by [`step_by`](crate::step_by) when `k = 0`.
///
/// A step of zero is a genuine error — not a clamp-to-1 (C1 / spec §4). The error type is a
/// unit struct: there is nothing more to say beyond "the step was zero".
///
/// # Guarantee tag: `Exact` (the error itself is a precise diagnosis)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroStep;

impl fmt::Display for ZeroStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("step_by: k = 0 is not a valid step size (must be ≥ 1)")
    }
}

mycelium_std_core::impl_std_error!(ZeroStep);

/// Error returned by [`zip_exact`](crate::zip_exact) when the left and right `Foldable`s have
/// different lengths.
///
/// Carries both lengths so the caller can surface the mismatch (C1 / C3 EXPLAIN).
///
/// # Guarantee tag: `Exact` (the error is a precise diagnosis — no approximation)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZipLengthMismatch {
    /// The number of elements in the left `Foldable`.
    pub left_len: usize,
    /// The number of elements in the right `Foldable`.
    pub right_len: usize,
}

impl fmt::Display for ZipLengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "zip_exact: length mismatch — left has {} elements, right has {}",
            self.left_len, self.right_len
        )
    }
}

mycelium_std_core::impl_std_error!(ZipLengthMismatch);
