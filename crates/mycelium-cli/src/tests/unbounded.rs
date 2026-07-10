//! RFC-0041 §5 / DN-84 §9.3 — the `--unbounded` escape hatch: corpus refusal, never-silent banner,
//! and functional parity of a well-behaved run. White-box access via `use crate::*`.
//!
//! Guarantee: `Empirical` (behavioural checks over the guard + banner + one fixture), not `Proven`.

use crate::*;
use std::path::PathBuf;

/// The committed single-nodule fixture for `myc run` v0 (M-908) — reused to confirm `--unbounded`
/// does not change a well-behaved program's result.
fn run_fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/run-single-nodule/mycelium-proj.toml")
}

/// The corpus runner must REFUSE `--unbounded` (RFC-0041 §5): the deterministic corpus/CI path may not
/// run the opt-in, machine-dependent mode. This is the required "corpus-runner rejects it" test.
#[test]
fn a_corpus_run_refuses_unbounded() {
    let opts = RunOptions { unbounded: true };
    let err = reject_unbounded_in_corpus(&opts)
        .expect_err("a corpus/CI run must refuse `--unbounded`, never allow it");
    assert_eq!(err.code, "myc-unbounded-corpus");
    assert_eq!(
        err.exit, 64,
        "refusal is a usage-class exit, not a silent success"
    );
    assert!(
        err.message.contains("excluded from the conformance corpus"),
        "the refusal must be never-silent about WHY (G2); got: {}",
        err.message
    );
}

/// The complement: an ordinary (default) run is **not** refused by the corpus guard — the guard only
/// ever refuses the opt-in mode, never the deterministic default.
#[test]
fn a_corpus_run_allows_the_default() {
    assert!(
        reject_unbounded_in_corpus(&RunOptions::default()).is_ok(),
        "the deterministic default path must pass the corpus guard"
    );
}

/// `RunOptions` defaults to the ordinary, corpus-safe, deterministic behavior (unbounded off).
#[test]
fn run_options_default_is_bounded() {
    assert!(!RunOptions::default().unbounded);
}

/// The never-silent banner (G2) announces the mode, its non-determinism, and the corpus exclusion —
/// and its per-command effect line is accurate: `run` lifts the interpreter ceiling; `build` performs
/// no interpreted evaluation, so it is interface-parity only.
#[test]
fn the_banner_is_never_silent_and_command_accurate() {
    let run_banner = unbounded_banner("run");
    assert!(run_banner.contains("NON-DETERMINISTIC"), "{run_banner}");
    assert!(
        run_banner.contains("conformance corpus"),
        "the banner must state the corpus exclusion; got: {run_banner}"
    );
    assert!(
        run_banner.contains("DISABLED"),
        "the run banner must state the ceiling is disabled; got: {run_banner}"
    );

    let build_banner = unbounded_banner("build");
    assert!(
        build_banner.contains("no interpreted evaluation"),
        "the build banner must be honest that build does not run the interpreter; got: {build_banner}"
    );
}

/// Functional parity: a well-behaved program returns the **same** result under `--unbounded` as under
/// the deterministic default — `--unbounded` only lifts the ceiling, it does not change value
/// semantics. (The mode's machine-dependence is about *how deep* a program may go before refusal, not
/// the answer a shallow program computes.)
#[test]
fn unbounded_does_not_change_a_well_behaved_run_result() {
    let manifest = run_fixture_manifest();
    let bounded = run_with_options(&manifest, &RunOptions::default())
        .expect("the fixture runs end-to-end (default budget)");
    let unbounded = run_with_options(&manifest, &RunOptions { unbounded: true })
        .expect("the fixture runs end-to-end (unbounded budget)");
    assert_eq!(
        bounded.rendered, unbounded.rendered,
        "lifting the depth ceiling must not change a shallow program's result"
    );
}

/// `corpus_context()` is a pure read of `MYC_CORPUS` — a set, non-empty value signals the corpus/CI
/// context that refuses `--unbounded`. (Asserted directly rather than via env mutation, which would
/// race other parallel tests in-process.)
#[test]
fn corpus_context_reads_the_env_flag() {
    // Whatever the ambient env is, the function must not panic and must return a bool consistent with
    // the variable's presence — a cheap smoke over the signal `with_run_options` consults.
    let observed = corpus_context();
    let expected = std::env::var_os("MYC_CORPUS").is_some_and(|v| !v.is_empty());
    assert_eq!(observed, expected);
}
