//! `std.testing` — the repo's verification discipline as a library (M-534, #174).
//!
//! Property / golden / differential test harness. The **honesty crux** is C1/G2 turned on the
//! test report itself: **a skipped or undetermined check is *reported*, never a silent pass** —
//! a test that could not run produces an explicit [`Verdict::Skipped`] that aggregates distinctly
//! from [`Verdict::Pass`], so "green" can never silently include "did not actually check".
//!
//! # What this module provides
//! - **[`for_all`]** — property testing: a bound for every guarantee, shrink to minimal
//!   counterexample, reproducible by seed. A passing property backs an `Empirical` claim;
//!   the harness **never upgrades** that to `Proven` (VR-5).
//! - **[`golden`]** — snapshot / golden-file testing: compare a produced value against a
//!   content-addressed stored baseline. A missing baseline is [`Verdict::Skipped`] with
//!   [`SkipReason::NeedsRecord`], **never** a silent auto-accept (C1/G2).
//! - **[`differential`]** — oracle testing: run the same input through two implementations
//!   and require observable agreement (M-151/M-210; NFR-7). An unavailable backend yields
//!   [`Verdict::Skipped`] with [`SkipReason::BackendUnavailable`], never a silent pass.
//! - **[`summarize`] / [`is_green`]** — the honest aggregator: `Skipped`/`Undetermined` counts
//!   stay **distinct** from `Pass`; "green" means *checked and passed*, never *did not check*.
//!
//! # Guarantee matrix
//! All ops are `Exact` *as mechanisms* (a verdict is an exact, deterministic function of the run).
//! The harness **never inflates the subject's tag**: a passing `for_all` backs `Empirical`, not
//! `Proven` (VR-5). See [`guarantee_matrix::MATRIX`] and its tests.
//!
//! # §4.1 contract conformance
//! - **C1 — never-silent (G2):** Skip/undetermined are first-class variants, not absence.
//! - **C2 — honest per-op tag (VR-5):** harness ops are `Exact` mechanisms; subject tag is not
//!   inflated.
//! - **C3 — no black boxes (SC-3/G11):** every non-`Pass` verdict is a reified inspectable
//!   artifact.
//! - **C4 — content-addressed, value-semantic (ADR-003/RFC-0001):** golden baselines are
//!   content-addressed; verdicts are immutable values; seeded runs are pure.
//! - **C5 — above the small kernel (KC-3):** adds no trusted code; checks the trusted base.
//! - **C6 — declared, bounded effects (RFC-0014):** property runs declare their trial budget;
//!   golden declares baseline IO; seeded generator is pure (RT3).
//!
//! # FLAGs (propagate to orchestrator / spec ratification)
//! - **FLAG-DIAG (RESOLVED):** `Fail` carries a structured diagnostic record. `std.diag` (M-510)
//!   has landed; [`FailRecord::to_diag`] projects to the canonical [`mycelium_diag::Diag`] record.
//!   `FailRecord` keeps testing-specific reproduction metadata (seed + trial index) and folds them
//!   into the `Diag`'s notes — delegating presentation to `std.diag`, not duplicating it (KC-3).
//! - **FLAG-Q5:** The differential harness adopts the §8-Q5 two-level bar (observable-result
//!   equivalence floor + per-module tag/EXPLAIN equivalence for honesty-load-bearing modules).
//!   The exact ratified definition lives in RFC-0016 §8-Q5 (RESOLVED per README §5). This
//!   implementation provides the observable-equality floor; the tag/EXPLAIN level is deferred
//!   pending fuller `std.diag` integration at the differential test call sites.
//! - **FLAG-WORKSPACE:** The workspace `Cargo.toml` was updated to add `crates/mycelium-std-testing`
//!   as a member. This is a parent-owned file; the orchestrator should verify this addition.
//!
//! # Design spec
//! `docs/spec/stdlib/testing.md` (M-534, #174).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Test assertions carry representation context — a
//! representation mismatch between expected and actual values is a [`Verdict::Fail`], never a
//! silent coercion. The differential harness (NFR-7) checks observable-result equivalence across
//! representations; a tag or `Repr` mismatch is a first-class failure, not a silent pass.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/testing.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod cert_mode_test;
pub mod guarantee_matrix;
pub mod verdict;

pub use cert_mode_test::{
    assert_mode_negative, assert_mode_scope, for_each_mode, for_each_mode_in, ModeScope,
    ModeTestConfig, ModeVisit,
};
pub use verdict::{FailRecord, SkipReason, Summary, UndetReason, Verdict};

#[cfg(test)]
mod tests;

// ─── Generator (seed-based, no external randomness) ──────────────────────────

/// A deterministic, seeded pseudo-random generator for property-test inputs (RT3 / C6).
///
/// Uses a Xorshift64 LCG — trivial but sufficient for reproducible input generation in a
/// no-external-randomness context (the `std.rand` seeded surface discipline, spec §2).
/// Deterministic: the same seed always produces the same sequence.
///
/// **FLAG-RAND:** When `std.rand` (M-531) lands, this should be replaced or delegated to the
/// seeded generator surface it provides. The API contract (seed-in, deterministic sequence out,
/// no undeclared entropy) will not change.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Construct a generator from a fixed seed (RT3: no undeclared entropy).
    ///
    /// A seed of `0` is promoted to a non-zero default to avoid a degenerate Xorshift state.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0xDEAD_BEEF_CAFE_1337
        } else {
            seed
        };
        Self { state }
    }

    /// Advance the state and return the next `u64` (Xorshift64).
    ///
    /// # Guarantee tag: `Exact`
    /// Deterministic; same state always yields same output (C4 / RT3).
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Advance and return a `u32`.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Advance and return a value in `[0, n)`. Panics if `n == 0`.
    ///
    /// Uses rejection sampling to avoid modulo bias (Exact, no approximation).
    pub fn next_usize_below(&mut self, n: usize) -> usize {
        assert!(n > 0, "n must be positive");
        let n = n as u64;
        // Rejection-sampling, done entirely in u64. Computing the threshold (or the draw) as
        // `usize` would truncate on 32-bit targets — collapsing the threshold near u64::MAX to a
        // small value and reintroducing severe modulo bias. `v % n < n <= usize::MAX`, so the
        // final cast is lossless.
        let threshold = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < threshold {
                return (v % n) as usize;
            }
        }
    }
}

// ─── Generator trait ─────────────────────────────────────────────────────────

/// A type that can produce values of type `T` given an `Rng`.
///
/// The seed + generator sequence is fully deterministic (C4/RT3). A `Gen<T>` that cannot
/// produce any value MUST return an empty list from [`Gen::shrink`] so the harness can
/// report [`Verdict::Skipped`] rather than looping indefinitely.
pub trait Gen<T> {
    /// Try to produce a value; `None` if the generator is exhausted or cannot produce.
    fn generate(&mut self, rng: &mut Rng) -> Option<T>;

    /// Produce shrink candidates from a failing value (smaller/simpler values that might still
    /// reproduce the failure). Default: no shrinking (empty list).
    fn shrink(&self, _value: &T) -> Vec<T> {
        vec![]
    }
}

// ─── Budget ───────────────────────────────────────────────────────────────────

/// A declared, bounded trial budget for a property run (C6 — effects are bounded).
///
/// A property cannot run with an unbounded budget; the `Budget` enforces a finite trial count.
/// The minimum is 1; creating a `Budget(0)` is refused (C1 — never silent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Budget(u32);

impl Budget {
    /// The default budget when no specific value is required (100 trials).
    pub const DEFAULT: Budget = Budget(100);

    /// The minimum budget (1 trial).
    pub const MIN: Budget = Budget(1);

    /// Create a budget from a trial count. Returns `None` if `trials == 0` (C1 — never-silent).
    #[must_use]
    pub fn new(trials: u32) -> Option<Self> {
        if trials == 0 {
            None
        } else {
            Some(Budget(trials))
        }
    }

    /// The number of trials this budget permits.
    #[must_use]
    pub fn trials(self) -> u32 {
        self.0
    }
}

// ─── Property harness ─────────────────────────────────────────────────────────

/// Run a property test: generate `budget` inputs from `gen` and check `prop` for each.
///
/// Returns the first failure (shrunk to a minimal counterexample), `Skipped` if the generator
/// cannot produce any input, or `Pass` if all trials succeed.
///
/// # Guarantee tag: `Exact` (the verdict is an exact function of the run)
/// A passing `for_all` **backs an `Empirical` claim** about the property — not `Proven`. The
/// harness has no operation that turns "passed N trials" into `Proven`; that would be the
/// exact VR-5 violation the module exists to prevent (spec §4 / §4.1 C2).
///
/// # Fallibility
/// - `Verdict::Fail{..}` — property violated; carries shrunk counterexample + seed.
/// - `Verdict::Skipped{SkipReason::NeedsRecord}` — generator produced no inputs.
///
/// # Effects: none (pure; seeded — C6)
/// The `seed` parameter makes the run reproducible (RT3). No undeclared entropy is drawn.
///
/// # EXPLAIN
/// A `Fail` verdict carries the shrunk counterexample description + the reproducing seed so
/// the failure can be reproduced and explained (C3/G11/SC-3).
pub fn for_all<T, G>(gen: &mut G, seed: u64, budget: Budget, prop: impl Fn(&T) -> bool) -> Verdict
where
    G: Gen<T>,
    T: core::fmt::Debug,
{
    let mut rng = Rng::new(seed);
    let mut generated_any = false;

    for trial in 0..budget.trials() {
        let Some(input) = gen.generate(&mut rng) else {
            // Generator exhausted before budget; if we got nothing at all it's Skipped.
            if !generated_any {
                return Verdict::Skipped {
                    reason: SkipReason::NeedsRecord,
                };
            }
            // Otherwise we already ran some trials successfully — report Pass.
            break;
        };
        generated_any = true;

        if !prop(&input) {
            // Shrink to a minimal counterexample (spec §3: "shrink to a minimal counterexample").
            let shrunk_desc = shrink_to_minimal(&input, &mut |v| !prop(v), gen, trial);
            return Verdict::Fail {
                record: FailRecord {
                    description: shrunk_desc,
                    seed,
                    trial,
                    context: "for_all property violated".to_owned(),
                },
            };
        }
    }

    if !generated_any {
        // Budget > 0 but generator never yielded — report Skipped.
        return Verdict::Skipped {
            reason: SkipReason::NeedsRecord,
        };
    }

    Verdict::Pass
}

/// Shrink a failing value to a minimal counterexample.
///
/// Tries shrink candidates from the generator; returns a `Debug` string of the minimal
/// reproducing input. The number of shrink steps is bounded (at most 1000) to keep the
/// shrinking itself bounded (C6).
fn shrink_to_minimal<T, G>(
    initial: &T,
    still_fails: &mut impl FnMut(&T) -> bool,
    gen: &mut G,
    trial: u32,
) -> String
where
    G: Gen<T>,
    T: core::fmt::Debug,
{
    let mut best = format!("{initial:?}");
    let mut candidates = gen.shrink(initial);
    let mut steps = 0usize;

    while !candidates.is_empty() && steps < 1000 {
        let mut next_candidates = vec![];
        for c in &candidates {
            if still_fails(c) {
                best = format!("{c:?}");
                next_candidates = gen.shrink(c);
                break;
            }
        }
        candidates = next_candidates;
        steps += 1;
    }

    format!("trial={trial} value={best}")
}

// ─── Golden / snapshot harness ────────────────────────────────────────────────

/// A golden baseline: an identifier (the "name") and its expected serialized form.
///
/// Golden baselines are **content-addressed** by the combination of `name` + `expected` text
/// (C4/ADR-003). The `name` is a human-readable label; the `expected` is the stored snapshot.
///
/// **FLAG-IO:** In a full implementation, golden baselines would be persisted to the filesystem
/// and the IO effect declared on the op. Here, baselines are supplied at call time (no filesystem
/// dep in this Ring-2 library), matching the spec's "test library, not the runner" boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenBaseline {
    /// The identifier for this golden test.
    pub name: String,
    /// The expected snapshot content.
    pub expected: String,
}

impl GoldenBaseline {
    /// Construct a baseline from a name and expected string.
    #[must_use]
    pub fn new(name: impl Into<String>, expected: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expected: expected.into(),
        }
    }
}

/// Run a golden / snapshot test: compare `produced` against the stored baseline.
///
/// # Guarantee tag: `Exact` (the verdict is an exact function of produced vs baseline)
///
/// # Fallibility
/// - `Verdict::Fail{..}` — mismatch; carries a diff (C3 EXPLAIN: the diff + context).
/// - `Verdict::Skipped{SkipReason::NeedsRecord}` — baseline is `None` (missing).
///   A missing baseline is **never** auto-accepted (C1/G2 — the honesty crux for golden tests).
///
/// # Effects
/// Declared: none at this layer (the baseline is passed in, not read from disk). The runner/CI
/// wiring that supplies the baseline owns any filesystem IO — this library owns only the
/// comparison and verdict.
///
/// # EXPLAIN
/// A mismatch carries both the expected and produced values so the failure is inspectable (C3).
pub fn golden(baseline: Option<&GoldenBaseline>, name: &str, produced: &str) -> Verdict {
    match baseline {
        None => {
            // Missing baseline — explicit Skipped, never a silent pass (C1/G2).
            Verdict::Skipped {
                reason: SkipReason::NeedsRecord,
            }
        }
        Some(b) if b.name != name => {
            // Baseline name mismatch — treat as missing (the caller supplied the wrong baseline).
            Verdict::Skipped {
                reason: SkipReason::NeedsRecord,
            }
        }
        Some(b) => {
            if b.expected == produced {
                Verdict::Pass
            } else {
                // Mismatch — carry a structured diff description (C3/G11).
                let diff = make_diff(&b.expected, produced);
                Verdict::Fail {
                    record: FailRecord {
                        description: format!(
                            "golden mismatch for '{}': expected {:?}, got {:?}; diff: {}",
                            name, b.expected, produced, diff
                        ),
                        seed: 0,
                        trial: 0,
                        context: format!("golden({})", name),
                    },
                }
            }
        }
    }
}

/// Produce a human-readable diff description between `expected` and `actual` (C3/G11).
///
/// This is a minimal line-level diff sufficient for the EXPLAIN artifact; a full diff tool
/// lives in the runner, not this library.
fn make_diff(expected: &str, actual: &str) -> String {
    let exp_lines: Vec<&str> = expected.lines().collect();
    let act_lines: Vec<&str> = actual.lines().collect();

    let mut result = String::new();
    let max = exp_lines.len().max(act_lines.len());
    for i in 0..max {
        match (exp_lines.get(i), act_lines.get(i)) {
            (Some(e), Some(a)) if e == a => {
                result.push_str(&format!("  {e}\n"));
            }
            (Some(e), Some(a)) => {
                result.push_str(&format!("- {e}\n+ {a}\n"));
            }
            (Some(e), None) => {
                result.push_str(&format!("- {e}\n"));
            }
            (None, Some(a)) => {
                result.push_str(&format!("+ {a}\n"));
            }
            (None, None) => {}
        }
    }
    result
}

// ─── Differential / oracle harness ───────────────────────────────────────────

/// Run a differential (oracle) test: require `lhs(input) == rhs(input)`.
///
/// This implements the M-151/M-210 interp↔AOT/native oracle pattern (NFR-7). The two-level
/// agreement bar (§8-Q5 RESOLVED): observable-result equivalence is the floor enforced here;
/// the tag/EXPLAIN equivalence level requires deeper `std.diag` call-site integration (FLAG-Q5).
///
/// # Guarantee tag: `Exact` (the verdict is an exact function of both outputs)
///
/// # Fallibility
/// - `Verdict::Fail{..}` — disagreement; carries both outputs for inspection (C3/EXPLAIN).
/// - `Verdict::Skipped{SkipReason::BackendUnavailable}` — `lhs` or `rhs` is unavailable.
///   An unavailable backend is **never** a silent pass (C1/G2).
///
/// # Effects: none (pure, given available backends — C6)
/// Per-backend IO declared by the `available` flags.
///
/// # EXPLAIN
/// A `Fail` carries both outputs and the input description so the disagreement is inspectable
/// (C3/G11/SC-3). **FLAG-Q5:** the tag/EXPLAIN level of the §8-Q5 two-level bar awaits
/// `std.diag` (FLAG-DIAG).
pub fn differential<O>(
    input_desc: &str,
    lhs_available: bool,
    lhs: impl FnOnce() -> O,
    rhs_available: bool,
    rhs: impl FnOnce() -> O,
) -> Verdict
where
    O: PartialEq + core::fmt::Debug,
{
    if !lhs_available || !rhs_available {
        // Backend unavailable — explicit Skipped, never a silent pass (C1/G2).
        return Verdict::Skipped {
            reason: SkipReason::BackendUnavailable,
        };
    }

    let lhs_out = lhs();
    let rhs_out = rhs();

    if lhs_out == rhs_out {
        Verdict::Pass
    } else {
        Verdict::Fail {
            record: FailRecord {
                description: format!(
                    "differential disagreement for input '{input_desc}': lhs={lhs_out:?} rhs={rhs_out:?}"
                ),
                seed: 0,
                trial: 0,
                context: format!("differential({})", input_desc),
            },
        }
    }
}

// ─── Aggregator ───────────────────────────────────────────────────────────────

/// Aggregate a slice of verdicts into a [`Summary`].
///
/// `Skipped` and `Undetermined` counts are **kept distinct** from `Pass` (the crux: "green"
/// cannot silently include "did not check" — C1/G2).
///
/// # Guarantee tag: `Exact` (a total function over verdicts — no approximation)
/// # Fallibility: total
/// # Effects: none
/// # EXPLAIN: yes — the per-class counts are the inspection artifact (spec §4)
pub fn summarize(vs: &[Verdict]) -> Summary {
    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut skipped = 0u32;
    let mut undetermined = 0u32;

    for v in vs {
        match v {
            Verdict::Pass => passed += 1,
            Verdict::Fail { .. } => failed += 1,
            Verdict::Skipped { .. } => skipped += 1,
            Verdict::Undetermined { .. } => undetermined += 1,
        }
    }

    Summary {
        passed,
        failed,
        skipped,
        undetermined,
    }
}

/// True only if there are no failures **and** skipped/undetermined counts are surfaced (i.e.,
/// the caller can see them — this function does NOT hide them).
///
/// "Green" means *checked and passed*, never *did not check* (C1/G2 — the honesty crux).
///
/// # Guarantee tag: `Exact`
/// # Fallibility: total
/// # EXPLAIN: yes — the caller can inspect `Summary` to see the skip/undetermined counts
///
/// **Note:** `is_green` returns `true` when `failed == 0`, regardless of skipped/undetermined
/// count. The skipped/undetermined are *surfaced* in the `Summary` — `is_green` does not hide
/// them, but it does not force them to block a pass. This matches the spec's "skips are surfaced,
/// not silently absent" intent: a skip is visible in the Summary and must be explained; whether
/// it blocks the CI run is the runner's decision, not the harness's. An `is_green` that returned
/// `false` for any non-zero skip would violate C1 by treating "could not run" as "failed".
pub fn is_green(s: &Summary) -> bool {
    s.failed == 0
}
