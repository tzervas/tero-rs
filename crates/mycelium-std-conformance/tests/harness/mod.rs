//! Shared three-way + differential test-harness fixture for `.myc` stdlib nodule ports (M-925,
//! kickoff `opp`, RFC-0031 D5).
//!
//! Extracted verbatim from the `std_result.rs`/`std_option.rs` precedent so that every future
//! port (`std_diag.rs`, `std_core.rs`, … the M-926…M-934 wave) is **cases, not bespoke logic**
//! (house test-layout rule — "complex test logic lives in fixtures + parameterization, not in
//! test bodies"). A port's test file supplies only: the nodule's `include_str!` (the path is a
//! macro literal, so it cannot be centralized here — see "Usage" below), typed driver snippets,
//! and hand-computed expected values. All execution/comparison machinery lives here, once.
//!
//! # What this proves
//! For a nodule loaded verbatim (the single source of truth — no re-typing of the `.myc` under
//! test), the harness drives the **RFC-0007 §4.6 differential obligation**: the L1 fuel-guarded
//! evaluator (on the monomorphized env), `elaborate → L0` reference interpreter, and the AOT
//! env-machine must all agree on the observable, and every agreeing pair validates through the
//! M-210 shared TV checker (`mycelium_cert::check_core`) — never a silent pass.
//!
//! # Honesty tags (carry these into every port, VR-5 — never upgraded in translation)
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), validated
//!   by trial on the cases exercised; not a machine-checked proof.
//! - **`Declared`** — a combinator's type-level contract (a structural check, not a theorem),
//!   unless the port's own spec grounds a stronger tag (e.g. `Exact` for a total, match-defined,
//!   finite-domain op — see `std_cmp.rs`/`std_option.rs::is_some` for precedent).
//!
//! # Usage — new-port checklist (the D5 rows, RFC-0031 §5 / kickoff `opp`)
//! Every port task's Definition of Done walks these five rows; this fixture exists to make row 3
//! a drop-in rather than a re-derivation:
//!
//! 1. **Surface-check recorded before porting** (`Empirical`) — confirm the candidate crate's
//!    surface is expressible in `.myc` **today** (no H1 enabler needed) before starting the port.
//!    If it isn't, the task **STOPS and FLAGs** back to the `enb` epic — never forced (G2).
//! 2. **Pre-port polish committed separately**, Rust-side, behavior-neutral (ADR-038 §2.5) — clean
//!    up the Rust source's ambiguity *before* translating it; existing Rust tests must still pass
//!    unchanged, proving no behavior drifted.
//! 3. **`.myc` nodule + `include_str!` harness, three-way where forms close** — this module. Load
//!    the ported nodule verbatim (§ "Usage — wiring a new port" below), append a typed driver per
//!    case, and call [`assert_three_way`] (or the table-driven [`assert_cases`]).
//! 4. **Differential vs the Rust oracle green** (D5 bar; signature frozen) — the retained Rust
//!    crate (D6 — it is NOT retired, M-867 is post-1.0) is the oracle. `std_result`/`std_option`
//!    have **no** Rust-crate predecessor (M-649/M-715 — they were born self-hosted), so their
//!    "expected" side is a hand-computed reference `.myc` program run through the same path (the
//!    `expected_src` parameter below). The 9 `opp`-wave ports (`diag`…`spore`) DO have a retained
//!    Rust crate; wiring a direct Rust-vs-Mycelium comparison depends on that crate's own public
//!    API shape, which is out of this fixture's scope (M-925 owns `crates/mycelium-l1/tests/**`
//!    only) — **each port leaf FLAGs/implements its own Rust-oracle call** for row 4, using
//!    `expected_src` for the hand-computed form where no Rust comparison is wired yet, or a
//!    thin wrapper that runs the Rust fn and feeds its output into the same `Value`/`Repr`
//!    comparison this module already performs.
//! 5. **Per-op tags carried at the same strength** — never upgrade a tag in translation (VR-5);
//!    record the transpiler-assist fraction honestly in the port ledger
//!    (`docs/planning/self-hosting-port-ledger.md`), not here (this fixture is test-only).
//!
//! # Usage — wiring a new port
//! ```ignore
//! mod harness;
//!
//! const DIAG_SRC: &str = include_str!(concat!(
//!     env!("CARGO_MANIFEST_DIR"),
//!     "/../../lib/std/diag.myc"
//! ));
//!
//! fn program(driver: &str) -> String {
//!     harness::program(DIAG_SRC, driver)
//! }
//!
//! #[test]
//! fn some_combinator_case() {
//!     let driver = "fn main() => Bool = ...;";
//!     let src = program(driver);
//!     let expected = "nodule ref;\nfn main() => Bool = True;";
//!     harness::assert_three_way("some_combinator", &src, expected);
//! }
//! ```
//! For a port with many small cases sharing a shape, prefer the table-driven [`Case`]/
//! [`assert_cases`] pair over one `#[test]` per case when per-case CI granularity is not needed;
//! `std_result`/`std_option` keep one `#[test]` per case (unchanged from before this fixture) since
//! their existing test names are part of the CI-reporting surface and losing that granularity
//! would be a behavior change this migration must not make.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// Build a full test program by appending a driver to a nodule source that was itself loaded
/// verbatim via `include_str!` **at the call site** (the path argument to `include_str!` must be
/// a literal, so the load itself cannot be centralized in this shared module — only the
/// concatenation can). `nodule_src` is the single source of truth for the ported `.myc` file;
/// `driver` supplies the typed `main` (and any helper fns) that exercise one case.
pub fn program(nodule_src: &str, driver: &str) -> String {
    format!("{nodule_src}\n{driver}")
}

/// One differential case for the table-driven [`assert_cases`] entry point: a human-readable
/// label, the driver appended to the nodule source, and a small reference `.myc` program whose
/// `main` computes the expected value (ideally via the *same* underlying ops as the value under
/// test, so both share Derived — not Root — provenance; see the `std_result.rs`/`std_option.rs`
/// case comments for worked examples of this convention).
// `Case`/`assert_cases` are `#[allow(dead_code)]`: `std_result.rs`/`std_option.rs` (this
// migration's proof-it-generalizes consumers) deliberately keep one `#[test]` per case for
// CI-reporting granularity (calling `assert_three_way` directly, below) rather than the table
// form, so neither test binary references these two items yet. They are still `pub` API of this
// fixture for future ports (M-926…M-934) whose surface has many small, uniformly-shaped cases —
// never-silent about why the warning is suppressed (G2), not a claim the items are unused in
// principle.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct Case {
    pub label: &'static str,
    pub driver: &'static str,
    pub expected: &'static str,
}

/// Run every case in `cases` against `nodule_src` through [`assert_three_way`] — the fully
/// data-driven form for ports whose surface has many small, uniformly-shaped cases. Ports that
/// want per-case `#[test]` granularity in `cargo test` output (as `std_result`/`std_option` do)
/// should call [`assert_three_way`] directly from each `#[test]` fn instead; both forms share the
/// same underlying execution/comparison logic, so neither is "bespoke" (house test-layout rule).
#[allow(dead_code)]
pub fn assert_cases(nodule_src: &str, cases: &[Case]) {
    for case in cases {
        let src = program(nodule_src, case.driver);
        assert_three_way(case.label, &src, case.expected);
    }
}

/// Run the three-way differential on `src` — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT — and
/// assert all three paths agree AND equal the `expected_src` reference value (a `.myc` program
/// evaluated through the same L0-interp path).
///
/// Honesty: differential agreement is `Empirical` (trials); the type-level contract carried by
/// the ported combinators is `Declared` unless the port's own spec grounds a stronger tag.
pub fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    // Parse + type-check the test program.
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    // Monomorphize from `main` (every generic parameter must be fully determined — Residual
    // otherwise, a never-silent refusal — G2).
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));

    // M-673 closure invariant: the mono'd env must be closed (no generics, no traits).
    assert!(
        mono.fns.values().all(|fd| fd.sig.params.is_empty())
            && mono.types.values().all(|d| d.params.is_empty())
            && mono.traits.is_empty()
            && mono.instances.is_empty()
            && mono.impls.is_empty(),
        "{label}: monomorphized env must be closed (no generics/traits)"
    );

    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));

    // Path 1: L1 fuel-guarded evaluator on the monomorphized env.
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));

    // Path 2: elaborate→L0 reference interpreter.
    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: elaborate failed: {e}"));
    let l0_core = interp
        .eval_core(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));

    // Path 3: AOT env-machine.
    let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
        .unwrap_or_else(|e| panic!("{label}: AOT run_core failed: {e}"));

    // All three must agree (Empirical guarantee — trials).
    assert_eq!(
        l1_core, l0_core,
        "{label}: L1-eval(mono) vs elaborate→L0-interp diverged"
    );
    assert_eq!(l0_core, aot_core, "{label}: L0-interp vs AOT diverged");

    // Each agreeing pair validates through the M-210 shared checker (Empirical: never a silent
    // pass).
    for (x, y, pair) in [
        (&l1_core, &l0_core, "L1↔interp"),
        (&l0_core, &aot_core, "interp↔AOT"),
    ] {
        assert_eq!(
            check_core(x, y),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "{label}: the shared checker must validate the {pair} pair"
        );
    }

    // Compare against the reference value (hand-computed; the simplest honest reference is a
    // trivial direct program evaluated through the same three-way path).
    let ref_env = check_nodule(
        &parse(expected_src).unwrap_or_else(|e| panic!("{label}: ref parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("{label}: ref check failed: {e}"));
    let ref_node = elaborate(&ref_env, "main")
        .unwrap_or_else(|e| panic!("{label}: ref elaborate failed: {e}"));
    let expected = interp
        .eval_core(&ref_node)
        .unwrap_or_else(|e| panic!("{label}: ref eval failed: {e}"));

    assert_eq!(
        l1_core, expected,
        "{label}: result does not match expected reference value"
    );
}
