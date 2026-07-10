//! White-box tests for crate-root items in [`crate`] (the [`crate::WfError`] surface). Extracted
//! from the logic file as-touched (test layout rule; M-797).

use crate::WfError;

// Mutant-witness (lib.rs WfError::fmt): replaced with `Ok(Default::default())` which emits an
// empty string. A non-empty, distinct error message for each variant is required so callers can
// distinguish errors (G2: never silent).
#[test]
fn wf_error_display_is_non_empty_and_variant_specific() {
    // Each variant must produce a non-empty, distinct message.
    let variants = [
        (WfError::GuaranteeBoundMismatch, "M-I"),
        (WfError::MalformedBound, "bound"),
        (WfError::MalformedRepr, "non-positive"),
        (WfError::PayloadReprMismatch, "payload"),
        (WfError::MalformedReconstruction, "manifest"),
        (WfError::MalformedSparsity, "sparsity"),
        (
            WfError::DimensionTooLarge {
                field: "dim",
                value: 2_000_000_000,
                cap: 1 << 30,
            },
            "exceeds",
        ),
    ];
    let mut messages = Vec::new();
    for (variant, expected_fragment) in &variants {
        let msg = format!("{variant}");
        assert!(
            !msg.is_empty(),
            "WfError::{variant:?} must not display as empty string"
        );
        assert!(
            msg.contains(expected_fragment),
            "WfError display must contain '{expected_fragment}': got {msg:?}"
        );
        messages.push(msg);
    }
    // All messages must be distinct (no constant replacement covers all variants).
    for i in 0..messages.len() {
        for j in (i + 1)..messages.len() {
            assert_ne!(
                messages[i], messages[j],
                "WfError messages must be distinct: [{i}]={:?} == [{j}]={:?}",
                messages[i], messages[j]
            );
        }
    }
}
