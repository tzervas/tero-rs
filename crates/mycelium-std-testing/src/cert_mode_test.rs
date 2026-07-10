//! **Scoped mode-parametric testing** (M-796; RFC-0034 §13) — the §13 conformance contract as a
//! first-class, natively-wired capability of the Mycelium testing toolkit.
//!
//! # What this provides
//!
//! A downstream developer marks a test or suite to run **across the `CertMode` tiers** with the
//! cross-mode negative pattern (fires-where-it-applies / absent-where-it-doesn't) as a **built-in**
//! — no hand-rolling required (RFC-0034 §13 ¶8: "give developers the tool + the default + the scope
//! dial, and let them choose").
//!
//! ## Core types
//! - **[`ModeScope`]** — a typed `[bool; 3]` predicate set declaring which `CertMode` tiers a
//!   property is expected to hold in. Predefined constants cover the common RFC-0034 cases; custom
//!   scopes are built from `[bool; 3]` in `[Fast, Balanced, Certified]` order.
//! - **[`ModeTestConfig`]** — the configurable scope for a test/suite, resolved most-specific-wins
//!   (`global ⊐ phylum ⊐ nodule ⊐ granular`) via M-790's `resolve_mode` / `CertDecl` mechanism
//!   (shared resolver — not a parallel one). Carries the `@certification` resolution provenance.
//!
//! ## Core functions
//! - **[`for_each_mode`]** — sweep all `CertMode` tiers, calling a closure for each. Always-on:
//!   every mode is visited and reported (never-silent — C1/G2). The simplest entry point.
//! - **[`for_each_mode_in`]** — like `for_each_mode`, but filtered by a `ModeScope`: only visits
//!   modes in scope. Returns the tiers visited and the tiers skipped, so the caller can verify
//!   that the right modes ran (never-silent about which tiers ran — C1/G2).
//! - **[`assert_mode_scope`]** — the primary assertion helper: given a `ModeScope` and a predicate,
//!   asserts `predicate(mode) == true` for each mode in scope and `== false` for each mode outside
//!   scope — the cross-mode **negative** pattern made first-class. Both directions are checked; a
//!   panic on either side surfaces the exact violating mode and direction.
//! - **[`assert_mode_negative`]** — negative-only helper: asserts a property is **absent** in all
//!   modes outside the given scope (the "invariant must not fire where it doesn't apply" check, the
//!   dual of a simple `assert_eq!`).
//! - **[`ModeTestConfig::resolve`]** — derive the effective `ModeScope` from a stack of
//!   `@certification` declarations (most-specific-wins), using the **shared** M-790 resolver
//!   ([`mycelium_proj::resolve_mode`]) — not a parallel implementation.
//!
//! ## CONFIGURABLE SCOPE (RFC-0034 §6 reused for testing)
//!
//! The `@certification` scoping mechanism (most-specific-wins over `global > phylum > nodule`,
//! RFC-0012 ambient/scoped-override) is **reused** here for test scope configuration:
//!
//! - **Project-wide default** — supplied as a `[CertDecl]` at the `Phylum` or `Global` scope.
//! - **Nodule-wide** — a `CertDecl` at `Nodule` scope overrides the project default.
//! - **Granular per-test** — [`ModeTestConfig::with_granular`] overrides at the finest granularity.
//!
//! Resolution uses `mycelium_proj::resolve_mode`, so the test scope and the `@certification`
//! attribute share a single resolution algorithm (no drift, no second implementation — KC-3).
//!
//! ## Never-forces (RFC-0034 §7 / the never-cornering stance)
//!
//! Nothing in this module forces a mode change on the test runner or on the values under test.
//! `ModeScope` and `ModeTestConfig` are advisory: they declare what the test author *intends* to
//! check, so the harness can surface mismatches (wrong tiers ran, property held where it shouldn't)
//! as explicit verdicts — never silently.
//!
//! ## Guarantee matrix
//! | Op | Tag | Fallibility | Effects | EXPLAIN |
//! |---|---|---|---|---|
//! | `for_each_mode` | Exact | total | none | yes (visits are reported) |
//! | `for_each_mode_in` | Exact | total | none | yes (`ModeVisit` carries visited/skipped sets) |
//! | `assert_mode_scope` | Exact | panics on mismatch | none | yes (mode + direction in panic msg) |
//! | `assert_mode_negative` | Exact | panics on mismatch | none | yes (mode in panic msg) |
//! | `ModeTestConfig::resolve` | Exact | total | none | yes (provenance in `ResolvedMode`) |
//!
//! **Guarantee tag: `Declared`** — this module exposes the *mechanism* for mode-parametric testing;
//! the guarantee strength of any *specific* property tested through this harness is determined by
//! the test author, never by the harness (VR-5 — no inflation).
//!
//! ## FLAGs for maintainer ratification
//! - **FLAG-SURFACE:** This module provides the *library-level* capability. The surface-syntax
//!   `@cert_scope` / `@mode_scope` annotations mentioned in RFC-0034 §13 are a future language-level
//!   feature (lands with surface-syntax work). Until then, `ModeTestConfig::with_granular` is the
//!   granular per-test override. This is the smallest honest version that satisfies the M-796 DoD;
//!   the surface-phase follow-on is flagged, not silently deferred.
//! - **FLAG-INJECT:** `ModeTestConfig::resolve` always uses the project-default `Fast` when no
//!   declarations are supplied (RFC-0034 §5 default). If the project needs a test-specific default
//!   of `Balanced`/`Certified`, supply a `[CertDecl { scope: Phylum, mode: CertMode::Balanced }]`.
//!   There is no separate "test-default" mechanism; project + nodule + granular is the resolution
//!   chain (the shared M-790 lattice, composed here for testing).

use mycelium_core::cert_mode::CertMode;
use mycelium_proj::{resolve_mode, ResolvedMode};

// Re-export the M-790 resolver surface types that downstream test code needs to build
// `ModeTestConfig` declarations. Exporting them from this module means downstream code only
// needs `use mycelium_std_testing::cert_mode_test::{ModeTestConfig, CertDecl, CertScope}`
// rather than also pulling in `mycelium_proj` directly (KC-3 — single import surface).
pub use mycelium_proj::CertDecl;
pub use mycelium_proj::CertScope;

// ---------------------------------------------------------------------------
// § 1. ModeScope — the typed per-mode predicate set
// ---------------------------------------------------------------------------

/// A typed predicate set describing in which [`CertMode`] tiers a property is expected to hold.
///
/// This is the first-class representation of the cross-mode scope declaration from RFC-0034 §13:
/// a test/suite *declares its intended scope*, and the harness checks both directions — the
/// property fires where it applies **and** is correctly absent where it does not.
///
/// ## Predefined scopes (the common RFC-0034 cases)
/// - [`ALL_MODES`](ModeScope::ALL_MODES): property holds in every mode (e.g. never-silent
///   fallibility, cert_mode tag presence — Axis-B invariants).
/// - [`FAST_ONLY`](ModeScope::FAST_ONLY): property holds only in `Fast` (e.g. the guarantee
///   floor from `Proven`/`Empirical` to `Declared`, cert suppression).
/// - [`NON_FAST`](ModeScope::NON_FAST): `Balanced` + `Certified` — modes where the certification
///   machinery runs and `Empirical`/`Proven` tags are reachable (RFC-0034 §5). Alias:
///   [`EMIT_MODES`](ModeScope::EMIT_MODES).
/// - [`CERTIFIED_ONLY`](ModeScope::CERTIFIED_ONLY): property holds only in `Certified` (e.g.
///   certificate *checking*).
///
/// ## Custom scopes
/// Build with `ModeScope { in_scope: [fast_in, balanced_in, certified_in] }` — the bool array
/// is indexed `[Fast=0, Balanced=1, Certified=2]`, matching [`CertMode::ALL`] order.
///
/// ## Scope × scope composition
/// `ModeScope` is `Copy` and composable: `ModeScope::union` / `ModeScope::intersect` let you build
/// derived scopes without naming all booleans.
///
/// # Guarantee tag: `Declared`
/// A `ModeScope` is a declaration, not a checked theorem. The harness checks conformance against
/// the declared scope, but whether the scope is the *right* scope for the property is the test
/// author's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeScope {
    /// `in_scope[i]` = true means `CertMode::ALL[i]` is in scope. Indices: 0=Fast, 1=Balanced,
    /// 2=Certified — the same order as [`CertMode::ALL`] (Fast `depth()=0` … Certified `depth()=2`).
    pub in_scope: [bool; 3],
}

impl ModeScope {
    /// Property holds in **every** mode — the unconditional scope (e.g. Axis-B never-silent
    /// fallibility, cert_mode tag presence). In scope: Fast, Balanced, Certified.
    pub const ALL_MODES: ModeScope = ModeScope {
        in_scope: [true, true, true],
    };

    /// Property holds **only in `Fast`** (e.g. the floor from `Proven`/`Empirical` to `Declared`,
    /// cert suppression in Fast). In scope: Fast only.
    pub const FAST_ONLY: ModeScope = ModeScope {
        in_scope: [true, false, false],
    };

    /// Property holds in **`Balanced` and `Certified`** — the modes where the certification
    /// machinery runs and `Empirical`/`Proven` tags are reachable (RFC-0034 §5). Alias:
    /// [`EMIT_MODES`](ModeScope::EMIT_MODES). In scope: Balanced, Certified.
    pub const NON_FAST: ModeScope = ModeScope {
        in_scope: [false, true, true],
    };

    /// Property holds **only in `Certified`** (e.g. certificate *checking*, RFC-0034 §5 Axis-C).
    /// In scope: Certified only.
    pub const CERTIFIED_ONLY: ModeScope = ModeScope {
        in_scope: [false, false, true],
    };

    /// Property holds in **`Balanced` and `Certified`** — the modes that *emit* swap certificates
    /// (RFC-0034 §5). Alias for [`NON_FAST`](ModeScope::NON_FAST). In scope: Balanced, Certified.
    pub const EMIT_MODES: ModeScope = ModeScope {
        in_scope: [false, true, true],
    };

    /// Property holds only in `Balanced` (e.g. a balanced-specific cost/precision tradeoff).
    /// In scope: Balanced only.
    pub const BALANCED_ONLY: ModeScope = ModeScope {
        in_scope: [false, true, false],
    };

    /// Returns `true` iff the given mode is in scope.
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    #[must_use]
    pub fn contains(self, mode: CertMode) -> bool {
        // CertMode::ALL order: [Fast=0, Balanced=1, Certified=2] — indexed by `depth()`.
        self.in_scope[mode.depth() as usize]
    }

    /// The set of modes **in** scope (at most 3 elements, in `CertMode::ALL` order).
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total (always returns a `Vec` of 0..=3 elements)
    #[must_use]
    pub fn modes_in_scope(self) -> Vec<CertMode> {
        CertMode::ALL
            .iter()
            .copied()
            .filter(|&m| self.contains(m))
            .collect()
    }

    /// The set of modes **outside** scope (the complement; at most 3 elements).
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    #[must_use]
    pub fn modes_out_of_scope(self) -> Vec<CertMode> {
        CertMode::ALL
            .iter()
            .copied()
            .filter(|&m| !self.contains(m))
            .collect()
    }

    /// The number of modes in scope (0..=3).
    ///
    /// # Guarantee tag: `Exact`
    #[must_use]
    pub fn count(self) -> usize {
        self.in_scope.iter().filter(|&&b| b).count()
    }

    /// `true` iff the scope is empty (no modes — a scope with nothing in it is a no-op but
    /// surfaced explicitly, not silently ignored — C1/G2).
    ///
    /// # Guarantee tag: `Exact`
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.count() == 0
    }

    /// The union of two scopes: a mode is in scope if it is in *either* scope.
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    #[must_use]
    pub fn union(self, other: ModeScope) -> ModeScope {
        ModeScope {
            in_scope: [
                self.in_scope[0] || other.in_scope[0],
                self.in_scope[1] || other.in_scope[1],
                self.in_scope[2] || other.in_scope[2],
            ],
        }
    }

    /// The intersection of two scopes: a mode is in scope only if it is in *both* scopes.
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    #[must_use]
    pub fn intersect(self, other: ModeScope) -> ModeScope {
        ModeScope {
            in_scope: [
                self.in_scope[0] && other.in_scope[0],
                self.in_scope[1] && other.in_scope[1],
                self.in_scope[2] && other.in_scope[2],
            ],
        }
    }

    /// Build a `ModeScope` from the effective mode of a [`ResolvedMode`]: the scope is "exactly
    /// the modes at or above the resolved depth" — i.e., `Fast` resolves to `ALL_MODES`,
    /// `Balanced` to `NON_FAST`, `Certified` to `CERTIFIED_ONLY`.
    ///
    /// This is the bridge between the `@certification` resolver and the `ModeScope` predicate:
    /// a `@certification: balanced` declaration means "this test cares about Balanced and
    /// Certified behaviour", not "this test only runs in Balanced".
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    ///
    /// **Mutant-witness:** if `from_resolved_mode(Balanced)` returned `CERTIFIED_ONLY`, a
    /// Balanced-scoped test would fail to check Balanced behaviour, and `modes_in_scope()` would
    /// return `[Certified]` instead of `[Balanced, Certified]`.
    #[must_use]
    pub fn from_resolved_mode(r: &ResolvedMode) -> ModeScope {
        match r.mode {
            // `fast` → all tiers are in scope (the test covers the full range).
            CertMode::Fast => ModeScope::ALL_MODES,
            // `balanced` → Balanced + Certified are in scope (the machinery modes).
            CertMode::Balanced => ModeScope::NON_FAST,
            // `certified` → only Certified is in scope (the strictest coverage).
            CertMode::Certified => ModeScope::CERTIFIED_ONLY,
        }
    }
}

// ---------------------------------------------------------------------------
// § 2. ModeTestConfig — configurable per-test scope with shared resolver
// ---------------------------------------------------------------------------

/// Configurable per-test / per-suite mode scope, resolved most-specific-wins via the shared
/// M-790 `resolve_mode` algorithm (RFC-0034 §6 / RFC-0012 ambient-override).
///
/// The scope is built from a stack of `@certification` declarations at `Global`, `Phylum`,
/// and `Nodule` scopes — the same three-tier lattice the `@certification` attribute uses in
/// production code — plus an optional **granular** override that wins over all scope tiers.
/// Resolution uses [`mycelium_proj::resolve_mode`] directly: the test scope and the production
/// `@certification` attribute share one algorithm (no parallel implementation, no drift — KC-3).
///
/// ## Building a config
/// ```rust
/// # use mycelium_std_testing::cert_mode_test::ModeTestConfig;
/// # use mycelium_core::cert_mode::CertMode;
/// # use mycelium_proj::{CertDecl, CertScope};
/// // Project default: fast (standard for most tests).
/// let default_config = ModeTestConfig::default();
///
/// // Project-wide default of Balanced, overridden per-nodule to Certified:
/// let config = ModeTestConfig::new(&[
///     CertDecl { scope: CertScope::Phylum, mode: CertMode::Balanced },
///     CertDecl { scope: CertScope::Nodule,  mode: CertMode::Certified },
/// ]);
///
/// // Granular per-test override — wins over all scope tiers:
/// let narrow = config.with_granular(CertMode::Fast);
/// ```
///
/// ## Resolving to a `ModeScope`
/// [`ModeTestConfig::resolve`] runs `resolve_mode` and maps the effective mode to a `ModeScope`
/// via [`ModeScope::from_resolved_mode`].
///
/// ## Provenance
/// [`ModeTestConfig::provenance`] returns the `ResolvedMode` — the effective mode + which scope
/// it came from — so the active test scope is never ambient (G2 / RFC-0012 renderability).
///
/// # Guarantee tag: `Declared` — a declaration, not a checked theorem (VR-5)
#[derive(Debug, Clone)]
pub struct ModeTestConfig {
    /// The `@certification` declarations at `Global`, `Phylum`, or `Nodule` scope. The resolver
    /// picks the most-specific winner; `Global` < `Phylum` < `Nodule` (RFC-0034 §6 lattice).
    decls: Vec<CertDecl>,
    /// An optional **granular** override that wins over all scope tiers — the per-test dial.
    /// `None` = no granular override (falls back to the `decls` resolution).
    granular: Option<CertMode>,
}

impl ModeTestConfig {
    /// Build a `ModeTestConfig` from a slice of `@certification` scope declarations.
    ///
    /// Declarations are not ordered (the resolver picks the highest-specificity winner). Supply
    /// at most one per scope level (duplicates at the same scope are resolved by
    /// `resolve_mode`'s max-specificity fold — the **last** duplicate found wins, per
    /// `Iterator::max_by_key`'s documented tie behaviour ("if several elements are equally
    /// maximum, the last element is returned"); in practice, the `@certification` parser
    /// upstream rejects duplicates, so this is just the resolver's behaviour for robustness).
    ///
    /// # Guarantee tag: `Exact` (pure construction)
    /// # Fallibility: total
    #[must_use]
    pub fn new(decls: &[CertDecl]) -> Self {
        ModeTestConfig {
            decls: decls.to_vec(),
            granular: None,
        }
    }

    /// Add (or replace) a **granular** per-test override — the most-specific tier, overrides
    /// all scope declarations in [`ModeTestConfig::new`].
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    #[must_use]
    pub fn with_granular(mut self, mode: CertMode) -> Self {
        self.granular = Some(mode);
        self
    }

    /// The provenance of the resolved test mode — the effective mode and the scope it came from.
    ///
    /// `source: None` means no declaration matched and the project default
    /// ([`CertMode::default`] = `Fast`) is in effect. `source: Some(CertScope::Nodule)` means a
    /// nodule-wide header won. When [`with_granular`](ModeTestConfig::with_granular) is set, the
    /// provenance reflects the granular override with `source: None` (the granular tier is not
    /// a `CertScope` level — it is above the lattice and wins unconditionally).
    ///
    /// The provenance is available for `EXPLAIN` rendering: a test that does not know which tiers
    /// ran can inspect the `ResolvedMode` and surface the scope decision (never ambient — G2).
    ///
    /// # Guarantee tag: `Exact`
    /// # Fallibility: total
    #[must_use]
    pub fn provenance(&self) -> ResolvedMode {
        // Granular override wins over all scope tiers (above the lattice).
        if let Some(g) = self.granular {
            return ResolvedMode {
                mode: g,
                source: None, // Granular is not a CertScope level — source is "granular/per-test".
            };
        }
        // Otherwise: delegate to the shared M-790 resolver.
        resolve_mode(&self.decls)
    }

    /// Resolve the effective `ModeScope` for this test/suite configuration.
    ///
    /// Uses [`ModeScope::from_resolved_mode`] on the [`provenance`](ModeTestConfig::provenance)
    /// result — so the mapping from `@certification` mode to test scope is canonical and shared.
    ///
    /// # Guarantee tag: `Exact` (pure, deterministic)
    /// # Fallibility: total
    ///
    /// **EXPLAIN:** call [`ModeTestConfig::provenance`] on the same config to see which scope
    /// tier produced the effective mode (never ambient — G2 / RFC-0012 renderability).
    #[must_use]
    pub fn resolve(&self) -> ModeScope {
        ModeScope::from_resolved_mode(&self.provenance())
    }
}

impl Default for ModeTestConfig {
    /// Default configuration: no declarations → project default `Fast` → `ALL_MODES` scope.
    ///
    /// This is the sensible default (RFC-0034 §13: "a mode-sensitive unit covers the tiers whose
    /// behaviour differs, plus the negatives"): a test with no `@certification` context runs
    /// across all tiers. Widening from `fast` is always safe.
    fn default() -> Self {
        ModeTestConfig {
            decls: vec![],
            granular: None,
        }
    }
}

// ---------------------------------------------------------------------------
// § 3. Mode-parametric iteration helpers
// ---------------------------------------------------------------------------

/// A summary of which `CertMode` tiers were visited and which were skipped by
/// [`for_each_mode_in`].
///
/// Carries the never-silent audit: the caller can assert that the expected tiers ran and the
/// right tiers were skipped. A test that only "happened to" cover the in-scope modes without
/// explicitly checking the out-of-scope modes gets no false confidence (C1/G2).
///
/// # Guarantee tag: `Declared` — a record of what ran; not a verdict on correctness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeVisit {
    /// The modes that were visited (in `CertMode::ALL` order).
    pub visited: Vec<CertMode>,
    /// The modes that were skipped because they were outside the scope (in `CertMode::ALL` order).
    pub skipped: Vec<CertMode>,
}

impl ModeVisit {
    /// `true` iff the visit covered all three modes (regardless of scope — useful for asserting
    /// that the full sweep happened).
    ///
    /// # Guarantee tag: `Exact`
    #[must_use]
    pub fn visited_all(&self) -> bool {
        self.visited.len() == CertMode::ALL.len()
    }

    /// `true` iff the visit covered exactly the given scope (no more, no fewer modes visited).
    ///
    /// # Guarantee tag: `Exact`
    #[must_use]
    pub fn matches_scope(&self, scope: ModeScope) -> bool {
        let expected: Vec<CertMode> = scope.modes_in_scope();
        self.visited == expected
    }
}

/// Run `f(mode)` for **every** mode in [`CertMode::ALL`] (weakest → strongest: Fast, Balanced,
/// Certified).
///
/// The simplest entry point for mode-parametric tests: the property is checked across the full
/// tier sweep. Use [`for_each_mode_in`] if you need scope filtering; use [`assert_mode_scope`]
/// if you also need the cross-mode negative check.
///
/// Never-silent: all three modes are always visited. There is no way to "skip a mode" through
/// this function (use `ModeScope` + `for_each_mode_in` for that).
///
/// # Guarantee tag: `Exact` (visits exactly the three modes in `CertMode::ALL` order)
/// # Fallibility: total (panics only if `f` panics)
/// # Effects: none (pure; the side-effects live in `f`)
///
/// # EXPLAIN
/// The visited tiers are `CertMode::ALL` — always `[Fast, Balanced, Certified]`. No filtering.
pub fn for_each_mode(mut f: impl FnMut(CertMode)) {
    for &mode in &CertMode::ALL {
        f(mode);
    }
}

/// Run `f(mode)` for each mode **in** `scope`, returning a [`ModeVisit`] that records which
/// modes were visited and which were skipped.
///
/// The caller can assert `visit.matches_scope(scope)` to confirm the right set ran, and inspect
/// `visit.skipped` to see which modes were out of scope (never silently absent — C1/G2).
///
/// # Guarantee tag: `Exact`
/// # Fallibility: total (panics only if `f` panics)
/// # Effects: none
///
/// # EXPLAIN
/// `ModeVisit` carries the full visited/skipped split — the caller can inspect it at any
/// verbosity level. The split is deterministic: `scope` fully determines which modes are in
/// `visited` vs `skipped`.
pub fn for_each_mode_in(scope: ModeScope, mut f: impl FnMut(CertMode)) -> ModeVisit {
    let mut visited = Vec::new();
    let mut skipped = Vec::new();

    for &mode in &CertMode::ALL {
        if scope.contains(mode) {
            f(mode);
            visited.push(mode);
        } else {
            skipped.push(mode);
        }
    }

    ModeVisit { visited, skipped }
}

// ---------------------------------------------------------------------------
// § 4. Cross-mode assertion helpers — the §13 conformance contract as built-ins
// ---------------------------------------------------------------------------

/// Assert that `predicate(mode)` returns **`true`** for every mode in `scope` and **`false`**
/// for every mode **outside** scope — the cross-mode **negative** pattern (RFC-0034 §13).
///
/// This is the mechanical implementation of the conformance contract: it catches **both**
/// (a) the property not holding when it should (positive failure), and (b) the property holding
/// when it should not ("invariant fires where it doesn't apply" — the negative failure that a
/// simple `assert!` cannot catch).
///
/// `desc` is a human-readable description of the property, included in panic messages (EXPLAIN
/// artifact — C3/G11).
///
/// # Guarantee tag: `Exact`
/// # Fallibility: panics on mismatch (the harness's "never-silent" is a panic, not a Verdict,
///   so the failure is immediately surfaced in the test runner — consistent with Rust's test
///   model)
/// # Effects: none (pure; side-effects live in `predicate`)
///
/// # EXPLAIN
/// A positive panic reports: which mode, the declared scope, and the description.
/// A negative panic reports: which mode (in scope but predicate false), the declared scope,
/// and the description.
///
/// **Mutant-witness for the negative arm:** if `predicate` always returns `true` (an invariant
/// that holds everywhere) and the scope is `FAST_ONLY`, the negative arm fires for `Balanced`
/// and `Certified`, catching the over-broad invariant. Conversely, if `predicate` always returns
/// `false`, the positive arm fires for `Fast`, catching the absent invariant.
///
/// # Example
/// ```rust
/// # use mycelium_std_testing::cert_mode_test::{ModeScope, assert_mode_scope};
/// # use mycelium_core::cert_mode::CertMode;
/// // Fast is the only mode that floors Proven to Declared:
/// assert_mode_scope(
///     ModeScope::FAST_ONLY,
///     |mode| mode == CertMode::Fast,
///     "Fast is the mode that applies the floor",
/// );
/// ```
pub fn assert_mode_scope(scope: ModeScope, predicate: impl Fn(CertMode) -> bool, desc: &str) {
    for &mode in &CertMode::ALL {
        let holds = predicate(mode);
        let expected = scope.contains(mode);
        if holds && !expected {
            panic!(
                "cross-mode NEGATIVE failed: `{desc}` holds in {mode:?} but scope={scope:?} \
                 says it should NOT. The invariant fires where it shouldn't (RFC-0034 §13)."
            );
        }
        if !holds && expected {
            panic!(
                "cross-mode POSITIVE failed: `{desc}` does NOT hold in {mode:?} but scope={scope:?} \
                 says it SHOULD. The invariant is absent where it must fire (RFC-0034 §13)."
            );
        }
    }
}

/// Assert that `predicate(mode)` returns **`false`** for every mode **outside** `scope` — the
/// negative-only half of the cross-mode negative pattern.
///
/// Use this when you want to assert the **absence** of a property in the out-of-scope modes,
/// without asserting the positive direction (you may have a separate positive check, e.g. the
/// standard `assert!` in a mode-pinned test body). This is the "invariant must be absent where
/// it doesn't apply" helper — the dual of a simple positive assertion.
///
/// # Guarantee tag: `Exact`
/// # Fallibility: panics when `predicate(mode)` returns `true` for a mode outside scope.
/// # Effects: none
///
/// # EXPLAIN
/// The panic carries the violating mode and the scope declaration (C3).
pub fn assert_mode_negative(scope: ModeScope, predicate: impl Fn(CertMode) -> bool, desc: &str) {
    for &mode in &CertMode::ALL {
        if !scope.contains(mode) && predicate(mode) {
            panic!(
                "cross-mode NEGATIVE failed: `{desc}` holds in {mode:?} but scope={scope:?} \
                 says it should NOT (property present outside its declared scope — RFC-0034 §13)."
            );
        }
    }
}

// ---------------------------------------------------------------------------
// § 5. Worked example (doc-only — used by tests in src/tests/cert_mode_test.rs)
// ---------------------------------------------------------------------------
//
// A downstream developer wanting per-tier + negative coverage with zero boilerplate writes:
//
// ```rust
// use mycelium_std_testing::cert_mode_test::{
//     assert_mode_scope, for_each_mode, ModeScope, ModeTestConfig,
// };
// use mycelium_core::cert_mode::CertMode;
// use mycelium_proj::{CertDecl, CertScope};
//
// /// Example: testing that `Fast` never yields `Empirical`/`Proven` guarantees.
// #[test]
// fn fast_never_yields_high_strength() {
//     // Scope: the property (no Empirical/Proven) holds only in Fast.
//     assert_mode_scope(
//         ModeScope::FAST_ONLY,
//         |mode| {
//             // Simulate: in Fast, guarantee is floored to Declared.
//             // In Balanced/Certified, Empirical/Proven are reachable.
//             mode == CertMode::Fast
//         },
//         "floor is active only in Fast",
//     );
// }
//
// /// Example: nodule-level scope configuration via ModeTestConfig.
// #[test]
// fn nodule_scoped_coverage() {
//     // Project default: Fast (all-modes scope). Nodule override: Certified.
//     let config = ModeTestConfig::new(&[
//         CertDecl { scope: CertScope::Phylum, mode: CertMode::Fast },
//         CertDecl { scope: CertScope::Nodule, mode: CertMode::Certified },
//     ]);
//
//     let scope = config.resolve();
//     let prov = config.provenance();
//
//     // Certified → CERTIFIED_ONLY scope.
//     assert_eq!(scope, ModeScope::CERTIFIED_ONLY);
//     assert_eq!(prov.mode, CertMode::Certified);
//     assert_eq!(prov.source, Some(CertScope::Nodule)); // nodule wins.
//
//     // Run the test only in Certified.
//     let visit = for_each_mode_in(scope, |mode| {
//         // The real check goes here — only runs in Certified.
//         assert_eq!(mode, CertMode::Certified);
//     });
//     assert!(visit.matches_scope(scope));
// }
// ```
