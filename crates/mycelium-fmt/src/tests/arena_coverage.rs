//! RFC-0041 §4.2/§9 — the **process-arena coverage** wiring for `mycfmt`'s render family
//! (`docs/notes/W7-arena-coverage-audit.md`, item 2): [`format_source_styled_cfg`] and
//! [`flatten_source`] both reserve against the shared process-wide arena before rendering, and
//! refuse [`FmtError::OutOfBudget`] never-silently rather than proceeding unbounded.
//!
//! `PROCESS_BYTES_CHARGED` (the arena's underlying counter, in `mycelium-workstack`) is a
//! process-global static; this crate's test binary links its own copy (distinct from any other
//! crate's test binary), but tests *within this binary* still run concurrently by default. These
//! tests use ceilings sized so far apart (an effectively-zero tiny ceiling vs. the crate's real
//! 256 MiB default) that no plausible concurrent interference from other in-crate tests can flip
//! either assertion — see `mycelium-workstack/src/tests/arena.rs` for the stricter Mutex-serialized
//! pattern this crate does not need here.

use crate::*;
use std::error::Error as _;

/// A small, valid, already-flat nodule — reused as the "normal input" fixture for both render-family
/// entry points below.
const SMALL_SRC: &str = "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x;\n";

/// [`format_source_styled_cfg_with_arena`]: a tiny (1-byte) ceiling refuses the pre-render
/// reservation never-silently, even for this crate's smallest realistic fixture.
#[test]
fn format_source_trips_out_of_budget_on_tiny_ceiling() {
    let tiny_arena = ProcessArena::new(1);
    match format_source_styled_cfg_with_arena(
        SMALL_SRC,
        None,
        Style::Compact,
        LayoutCfg::default(),
        &tiny_arena,
    ) {
        Err(FmtError::OutOfBudget(mycelium_workstack::BudgetError::OutOfBudget {
            kind,
            limit,
            ..
        })) => {
            assert_eq!(kind, mycelium_workstack::BudgetKind::Bytes);
            assert_eq!(limit, 1, "the refusal reports the configured ceiling");
        }
        other => panic!("expected an explicit OutOfBudget refusal, got {other:?}"),
    }
}

/// The normal-input twin: the same fixture, formatted through the crate's real declared default
/// ([`format_source`]), succeeds unchanged by the arena wiring.
#[test]
fn format_source_normal_input_passes_unchanged() {
    let out = format_source(SMALL_SRC, None).expect("small fixture fits the default arena ceiling");
    // Not asserting byte-identity (the canonical layout may re-wrap even a short body); the arena
    // wiring's contract is "still formats successfully", which C1 (checked inside the function
    // itself) already guarantees is identity-preserving at the AST level.
    assert!(out.output.starts_with("nodule d;"));
    assert!(out.output.contains("fn f"));
}

/// A larger — but still entirely ordinary — source (many small fns) also passes under the real
/// default ceiling: the arena wiring does not regress everyday-sized formatting.
#[test]
fn format_source_moderate_input_passes_unchanged() {
    let mut src = String::from("nodule d;\n");
    for i in 0..500 {
        src.push_str(&format!("fn f{i}() => Binary{{1}} = 0b0;\n"));
    }
    let out = format_source(&src, None).expect("500-fn source fits the default 256 MiB ceiling");
    assert!(out.output.contains("fn f499"));
}

/// [`flatten_source_with_arena`]: the `--flatten` render-family entry point refuses the same way.
#[test]
fn flatten_source_trips_out_of_budget_on_tiny_ceiling() {
    let tiny_arena = ProcessArena::new(1);
    match flatten_source_with_arena(SMALL_SRC, None, &tiny_arena) {
        Err(FmtError::OutOfBudget(mycelium_workstack::BudgetError::OutOfBudget {
            kind,
            limit,
            ..
        })) => {
            assert_eq!(kind, mycelium_workstack::BudgetKind::Bytes);
            assert_eq!(limit, 1, "the refusal reports the configured ceiling");
        }
        other => panic!("expected an explicit OutOfBudget refusal, got {other:?}"),
    }
}

/// The normal-input twin for `--flatten`: the same fixture, through [`flatten_source`]'s real
/// declared default, succeeds unchanged.
#[test]
fn flatten_source_normal_input_passes_unchanged() {
    let out =
        flatten_source(SMALL_SRC, None).expect("small fixture fits the default arena ceiling");
    assert!(out.output.starts_with("nodule d;"));
}

/// `FmtError::OutOfBudget`'s CLI exit code is distinct from the three pre-existing refusal kinds
/// (contract §5 additive extension — `FmtError` is `#[non_exhaustive]`).
#[test]
fn out_of_budget_has_its_own_exit_code() {
    let tiny_arena = ProcessArena::new(1);
    let err = format_source_styled_cfg_with_arena(
        SMALL_SRC,
        None,
        Style::Compact,
        LayoutCfg::default(),
        &tiny_arena,
    )
    .expect_err("tiny ceiling refuses");
    assert_eq!(err.exit_code(), 5);
    assert!(
        err.source().is_some(),
        "the BudgetError is chained as the source (EXPLAIN)"
    );
}
