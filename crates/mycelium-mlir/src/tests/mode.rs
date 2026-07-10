//! In-crate white-box tests for [`crate::mode`] (M-727; CLAUDE.md test-layout rule). Covers the
//! private `jit_err` error-mapping (the never-silent error-kind preservation) and the `ExecMode`
//! surface; the behavioural interp/AOT/JIT differential lives in `tests/mode_selection.rs`.

use crate::llvm::AotError;
use crate::mode::*;

#[test]
fn jit_err_preserves_the_failure_kind() {
    // The mapping must keep a toolchain skip a skip, an overflow an overflow, an unsupported an
    // unsupported — never collapse them to a generic error (G2: the kind is the never-silent signal).
    assert!(matches!(
        jit_err(AotError::ToolchainMissing("clang".into())),
        ModeError::ToolchainMissing(_)
    ));
    assert!(matches!(
        jit_err(AotError::UnsupportedNode("closure".into())),
        ModeError::Unsupported(_)
    ));
    assert!(matches!(
        jit_err(AotError::Overflow("2-trit".into())),
        ModeError::Overflow(_)
    ));
    // Any other AotError flattens to the explicit `Jit` variant (still an error, never a silent run).
    assert!(matches!(
        jit_err(AotError::Run("dlopen".into())),
        ModeError::Jit(_)
    ));
}

#[test]
fn exec_mode_has_no_default_and_three_named_variants() {
    // ALL is exactly the three named modes — there is no Auto/Default mode (the never-silent contract
    // is partly type-level: a mode is engaged only by being named).
    assert_eq!(ExecMode::ALL.len(), 3);
    // Names are stable and distinct (for EXPLAIN/logs).
    let names: Vec<&str> = ExecMode::ALL.iter().map(|m| m.name()).collect();
    assert_eq!(names, ["interpreter", "aot", "jit"]);
}

#[test]
fn mode_error_display_is_legible_and_names_the_path() {
    // A never-silent error must be human-legible (no opaque discriminant).
    let e = ModeError::Unsupported("closures".into());
    assert!(e.to_string().contains("jit mode"));
    let e = ModeError::ToolchainMissing("clang".into());
    assert!(e.to_string().contains("toolchain missing"));
}
