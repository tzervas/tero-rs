//! **RFC-0034 §13 conformance suite** (M-794 — the E21-1 capstone gate; scope narrowed by M-882).
//!
//! This is `mycelium-cert`'s share of the tunable-certification epic (E21-1) conformance suite: it
//! asserts the **three of the six RFC-0034 §13 conformance clauses that are this crate's own
//! contract** (a/b/c), each **parameterized over the [`CertMode`] tiers** (`fast`/`balanced`/
//! `certified`) **and** its **cross-mode negative** — the invariant must *fire* in the tiers it
//! applies to **and** be *correctly absent/relaxed* in the tiers it does not (the M-795
//! cross-mode-negative pattern). A `certified`-only invariant holding spuriously in `fast` is a
//! defect this suite catches, never silently passes (RFC-0034 §13 test contract). The remaining
//! three clauses (d/e/f) are verified in the crates that own their logic — see the M-882 note below.
//!
//! The conformance clauses (RFC-0034 §13):
//! - **(a)** every result carries a never-silent mode tag (§3.1) — verified **here**;
//! - **(b)** no `fast` result carries `Empirical`/`Proven` (§3.2) — verified **here**;
//! - **(c)** memory safety + Axis-B never-silent hold in **every** mode (§3.3) — verified **here**;
//! - **(d)** `EXPLAIN` of the active mode is always available (every mode, §7/§3.1) — verified in
//!   `crates/mycelium-proj/src/tests/cert_scope.rs` (see below);
//! - **(e)** spores are mintable with the runtime cert **off** (§8) — verified in
//!   `crates/mycelium-spore/src/tests/lib_tests.rs` (see below);
//! - **(f)** cross-mode composition surfaces the boundary, never a silent upgrade (§6/§3.1) —
//!   verified in `crates/mycelium-proj/src/tests/cert_scope.rs` (see below).
//!
//! ## M-882: clauses (d)/(e)/(f) relocated to their owning crates (dev-dep cycle break)
//! This suite previously pulled in `mycelium-proj`/`mycelium-spore` as **dev-dependencies** to
//! exercise clauses (d)/(e)/(f) "end-to-end". That created two dev-dep cycles: `cert →[dev] proj →
//! … → l1 → cert` and `cert →[dev] spore → proj → … → l1 → cert` (`mycelium-proj` depends on
//! `mycelium-l1`, which depends on `mycelium-cert`) — the "no cycle" claim the removed
//! `Cargo.toml` comment made was **wrong**, and M-882 is the fix.
//!
//! On inspection, clauses (d)/(e)/(f) as previously written never actually exercised any
//! `mycelium-cert` code path: `mycelium_proj::cert_scope::{explain_mode, compose, …}` and
//! `mycelium_spore::build_spore` operate entirely on `mycelium-core` types (`CertMode`/
//! `GuaranteeStrength`) with no wiring back through `mycelium-cert`'s own swap/gate surface — the
//! "genuinely cross-crate" framing this doc comment previously carried was an overclaim (VR-5:
//! corrected here rather than left standing). Each clause's real coverage already exists, more
//! thoroughly (property-based, `proptest`), in the crate that owns the logic:
//! - clause (d): `crates/mycelium-proj/src/tests/cert_scope.rs` —
//!   `explain_mode_is_available_in_every_mode_including_fast`,
//!   `generate_mode_signal_is_available_in_every_mode`, `lean_consumption_is_identical_to_explain_mode`,
//!   `dialing_consumption_up_surfaces_more_without_rerun`, plus the `prop_explain_mode_available_in_every_mode`
//!   / `prop_signal_generated_and_consumption_monotone` property tests;
//! - clause (e): `crates/mycelium-spore/src/tests/lib_tests.rs` — the M-789 DoD property test
//!   exhaustive over `CertMode::ALL` asserting spore mintability + content-identity independence
//!   from the runtime cert mode;
//! - clause (f): `crates/mycelium-proj/src/tests/cert_scope.rs` —
//!   `fast_into_certified_is_an_explicit_boundary_never_an_upgrade`,
//!   `structural_exact_survives_the_boundary`, plus the `prop_cross_mode_never_upgrades_strength`
//!   property test.
//!
//! Duplicating that coverage here via a dev-dependency added no additional assurance and cost a
//! dependency cycle; removing the duplication is not a coverage regression (the properties are
//! still checked, and checked with a wider proptest sweep than this suite ever ran) — it is
//! deduplication with an honest pointer, not a weakened test (never ship a hollow tautology in
//! place of real coverage, but also never keep a redundant cycle-inducing copy where the real
//! thing already exists and is stronger).
//!
//! ## Why this suite still spans (only) two crates
//! Clauses (a)/(b)/(c) are genuinely `mycelium-cert`'s own contract: they reach
//! [`mycelium_core::CertMode`] + the [`mycelium_cert`] gated-swap surface directly, with no need for
//! `mycelium-proj`/`mycelium-spore` at all.
//!
//! ## Data-driven, not bespoke (CLAUDE.md test-layout)
//! Each clause is a `#[test]` whose body is *assert over the [`CertMode::ALL`] sweep* via the
//! local [`for_each_mode`]/[`assert_mode_scope`] helpers (the M-795 harness shapes, duplicated
//! locally per the harness's own cross-crate note — `mode_harness.rs` is `#[cfg(test)]`-only in
//! `mycelium-core` and not exported). The harness fixtures
//! ([`proven_bound`]/[`empirical_bound`]/[`declared_bound`]) are the canonical per-strength
//! pre-images.
//!
//! **Guarantee tag:** the suite *checks* invariants; the strongest claim any assertion makes is the
//! one the code under test already establishes (`Exact` for a bijective swap; `Proven`/`Empirical`
//! only where the certified machinery earned it). The suite itself adds no guarantee tag — it is a
//! verification target (VR-5: it never upgrades a claim, it only asserts the floors hold).

use mycelium_cert::{
    binary_to_ternary, dense_f32_to_bf16, dense_to_vsa, gate_swap, GatedSwap, ModeGatedSwapEngine,
    SwapCertificate,
};
use mycelium_core::{
    binary, ternary, Bound, BoundBasis, BoundKind, CertMode, ContentHash, GuaranteeStrength, Meta,
    NormKind, Payload, Provenance, Repr, ScalarKind, Value, WrappingOpt,
};
use mycelium_interp::{EvalError, SwapEngine};

// ===========================================================================
// Shared harness (M-795 shapes, duplicated locally — see mode_harness.rs note)
// ===========================================================================

fn policy() -> ContentHash {
    ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

/// Run `f(mode)` for every mode in [`CertMode::ALL`] (Fast, Balanced, Certified).
fn for_each_mode(mut f: impl FnMut(CertMode)) {
    for &mode in &CertMode::ALL {
        f(mode);
    }
}

/// A predicate-set over the three tiers, in [`CertMode::ALL`] order `[Fast, Balanced, Certified]`.
/// Mirrors `mode_harness::ModeScope` (M-795): the cross-mode-negative pattern made first-class.
#[derive(Debug, Clone, Copy)]
struct ModeScope {
    in_scope: [bool; 3],
}

impl ModeScope {
    /// In scope in **every** mode (Axis-B never-silent, cert_mode tag presence, EXPLAIN-of-mode).
    const ALL_MODES: ModeScope = ModeScope {
        in_scope: [true, true, true],
    };
    /// In scope **only in `Fast`** (the `Proven`/`Empirical` → `Declared` floor; cert suppression).
    const FAST_ONLY: ModeScope = ModeScope {
        in_scope: [true, false, false],
    };
    /// In scope in **`Balanced` + `Certified`** — the cert-emitting / machinery-running tiers.
    const NON_FAST: ModeScope = ModeScope {
        in_scope: [false, true, true],
    };
    /// In scope **only in `Certified`** — certificate *checking*.
    const CERTIFIED_ONLY: ModeScope = ModeScope {
        in_scope: [false, false, true],
    };

    fn contains(self, mode: CertMode) -> bool {
        self.in_scope[mode.depth() as usize]
    }
}

/// Assert `predicate(mode) == scope.contains(mode)` for every mode — both the positive arm (fires
/// where it should) and the **negative** arm (absent where it should). The negative arm is the
/// whole point of the §13 contract: catch an invariant holding where it must not.
fn assert_mode_scope(scope: ModeScope, predicate: impl Fn(CertMode) -> bool, desc: &str) {
    for &mode in &CertMode::ALL {
        let holds = predicate(mode);
        let expected = scope.contains(mode);
        if holds && !expected {
            panic!(
                "cross-mode NEGATIVE failed: `{desc}` holds in {mode:?} but should NOT \
                 (the invariant fires where it shouldn't)."
            );
        }
        if !holds && expected {
            panic!(
                "cross-mode POSITIVE failed: `{desc}` does NOT hold in {mode:?} but SHOULD \
                 (the invariant is absent where it must fire)."
            );
        }
    }
}

// --- canonical per-strength bound pre-images (mirror mode_harness fixtures) ---

fn proven_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.003_906_25,
            norm: NormKind::Rel,
        },
        basis: BoundBasis::ProvenThm {
            citation: "round-to-nearest relative error theorem".to_owned(),
        },
    }
}

fn empirical_bound() -> Bound {
    Bound {
        kind: BoundKind::Probability { delta: 0.05 },
        basis: BoundBasis::EmpiricalFit {
            trials: 10_000,
            method: "Monte-Carlo round trip".to_owned(),
        },
    }
}

fn declared_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.1,
            norm: NormKind::L2,
        },
        basis: BoundBasis::UserDeclared,
    }
}

fn canonical_bound(g: GuaranteeStrength) -> Option<Bound> {
    match g {
        GuaranteeStrength::Exact => None,
        GuaranteeStrength::Proven => Some(proven_bound()),
        GuaranteeStrength::Empirical => Some(empirical_bound()),
        GuaranteeStrength::Declared => Some(declared_bound()),
    }
}

// --- value fixtures ---

fn byte_of(value: i64) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(binary::int_to_bits(value, 8).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// An exact Dense{F32} source (so the bounded `F32→BF16` / Dense→VSA swaps accept it). All values
/// are exactly representable in BF16 (1.0, 2.0) so the bounded swap is total.
fn dense_f32(xs: Vec<f64>) -> Value {
    Value::new(
        Repr::Dense {
            dim: u32::try_from(xs.len()).unwrap(),
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(xs),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A bipolar Dense{F32} source for the Dense↔VSA bounded swap (components must be ±1; M-231).
fn dense_bipolar(n: usize) -> Value {
    let xs: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    dense_f32(xs)
}

/// A ternary value outside `B_8` (`364` = all-`+` 6-trit ∉ [−128, 127]) — the Axis-B negative.
fn out_of_range_ternary() -> Value {
    Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(ternary::int_to_trits(364, 6).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// The three raw `(value, certificate)` swap pre-images exercised across modes: the **bijective**
/// (would-be `Exact`), the **bounded-ε** (would-be `Proven`), and the **bounded-δ** (would-be
/// `Empirical`) classes. Each is gated through every mode in the clause tests below.
fn raw_swaps() -> Vec<(&'static str, Value, Value, SwapCertificate)> {
    let bij_src = byte_of(42);
    let (bij_v, bij_c) = binary_to_ternary(&bij_src, 6, &policy()).unwrap();

    let eps_src = dense_f32(vec![1.0, 2.0, 1.0, 2.0]);
    let (eps_v, eps_c) = dense_f32_to_bf16(&eps_src, &policy()).unwrap();

    let delta_src = dense_bipolar(4);
    let (delta_v, delta_c) = dense_to_vsa(&delta_src, 2048, 1e-2, &policy()).unwrap();

    vec![
        ("bijective(Exact)", bij_src, bij_v, bij_c),
        ("bounded-ε(Proven)", eps_src, eps_v, eps_c),
        ("bounded-δ(Empirical)", delta_src, delta_v, delta_c),
    ]
}

// ===========================================================================
// Clause (a) — every result carries a never-silent mode tag (§3.1)
// ===========================================================================

/// **(a)** Every gated swap result, in **every** mode, carries the active [`CertMode`] as a tag on
/// its `Meta` — and it is exactly the mode it was produced under (never a silent default).
/// Cross-mode: the tag is present and *correct* in all three tiers (`ALL_MODES`).
#[test]
fn clause_a_every_result_carries_mode_tag() {
    for (name, src, raw_v, raw_c) in raw_swaps() {
        assert_mode_scope(
            ModeScope::ALL_MODES,
            |mode| {
                let g = gate_swap(&src, raw_v.clone(), raw_c.clone(), mode).unwrap();
                // The tag is present *and* equals the producing mode — not merely non-None.
                g.value.meta().cert_mode() == mode
            },
            &format!("{name}: result carries its exact CertMode tag"),
        );
    }
}

/// **(a) negative companion** — a deserialized `Meta` (no wire `cert_mode`, M-786) resolves to the
/// **weakest** mode `Fast`, never a silent `Certified`. The tag is never ambiently strong (VR-5).
#[test]
fn clause_a_default_mode_is_weakest_never_silently_strong() {
    let v = byte_of(7);
    assert_eq!(
        v.meta().cert_mode(),
        CertMode::Fast,
        "an untagged Meta must resolve to the weakest mode (Fast), never a silent Certified"
    );
}

// ===========================================================================
// Clause (b) — no `fast` result carries Empirical/Proven (§3.2)
// ===========================================================================

/// **(b)** A `fast` result **never** carries `Empirical`/`Proven`: the would-be `Proven` (ε) and
/// would-be `Empirical` (δ) swaps floor to `Declared` in `fast`, with their bound's basis relabelled
/// `UserDeclared`. Cross-mode **negative**: that flooring is present **only in `Fast`** — in
/// `Balanced`/`Certified` the earned strength passes through unchanged.
#[test]
fn clause_b_fast_never_empirical_or_proven() {
    // The bounded-ε swap is would-be `Proven`; the bounded-δ swap is would-be `Empirical`.
    for (name, src, raw_v, raw_c) in raw_swaps()
        .into_iter()
        .filter(|(n, ..)| *n != "bijective(Exact)")
    {
        // Positive+negative in one sweep: the strength is floored to Declared *only* in Fast.
        assert_mode_scope(
            ModeScope::FAST_ONLY,
            |mode| {
                let g = gate_swap(&src, raw_v.clone(), raw_c.clone(), mode).unwrap();
                g.value.meta().guarantee() == GuaranteeStrength::Declared
            },
            &format!("{name}: strength floored to Declared"),
        );
        // And, universally: in *no* mode does a Fast result carry Empirical/Proven.
        for_each_mode(|mode| {
            let g = gate_swap(&src, raw_v.clone(), raw_c.clone(), mode).unwrap();
            let s = g.value.meta().guarantee();
            if mode == CertMode::Fast {
                assert!(
                    s != GuaranteeStrength::Empirical && s != GuaranteeStrength::Proven,
                    "{name}: fast result carried {s:?} — the §3.2 floor was violated"
                );
                // The surviving bound's basis is the reconciled UserDeclared, never an unearned one.
                if let Some(b) = g.value.meta().bound() {
                    assert_eq!(
                        b.basis,
                        BoundBasis::UserDeclared,
                        "{name}: fast result kept an unearned bound basis {:?}",
                        b.basis
                    );
                }
            }
        });
    }
}

/// **(b) reachability negative** — `Empirical`/`Proven` *are* reachable in the non-`fast` tiers
/// (otherwise clause (b) would be vacuously true). The would-be `Proven` ε-swap surfaces a non-`fast`
/// strength stronger than `Declared` exactly in `Balanced`/`Certified` (`NON_FAST`).
#[test]
fn clause_b_strong_tags_reachable_only_outside_fast() {
    let src = dense_f32(vec![1.0, 2.0, 1.0, 2.0]);
    let (raw_v, raw_c) = dense_f32_to_bf16(&src, &policy()).unwrap();
    assert_mode_scope(
        ModeScope::NON_FAST,
        |mode| {
            let g = gate_swap(&src, raw_v.clone(), raw_c.clone(), mode).unwrap();
            // Stronger-than-Declared (rank < Declared's) is reachable only outside fast.
            g.value.meta().guarantee().rank() < GuaranteeStrength::Declared.rank()
        },
        "bounded-ε: a stronger-than-Declared tag is reachable only outside fast",
    );
}

/// **(b) gate_result invariant** — the [`CertMode::gate_result`] primitive that backs the floor is
/// itself mode-scoped over **every** canonical strength: in `fast` a `Proven`/`Empirical` intent
/// floors to `Declared`; outside `fast` it passes through. The reconciled pair is always
/// `Meta`-constructible (M-I1…M-I4).
#[test]
fn clause_b_gate_result_floors_only_in_fast() {
    for intended in [
        GuaranteeStrength::Exact,
        GuaranteeStrength::Proven,
        GuaranteeStrength::Empirical,
        GuaranteeStrength::Declared,
    ] {
        for_each_mode(|mode| {
            let (g, b) = mode.gate_result(intended, canonical_bound(intended));
            // The reconciled pair always constructs a Meta (the gate_result contract).
            Meta::new(Provenance::Root, g, b.clone(), None, None, None)
                .unwrap_or_else(|e| panic!("gate_result pair not Meta-constructible: {e:?}"));
            match mode {
                CertMode::Fast => {
                    assert!(
                        g != GuaranteeStrength::Empirical && g != GuaranteeStrength::Proven,
                        "fast gate_result yielded {g:?} for intent {intended:?}"
                    );
                }
                CertMode::Balanced | CertMode::Certified => {
                    assert_eq!(
                        g, intended,
                        "{mode:?} must pass the intended strength through unchanged"
                    );
                }
            }
        });
    }
}

// ===========================================================================
// Clause (c) — memory safety + Axis-B never-silent hold in every mode (§3.3)
// ===========================================================================

/// **(c) Axis-B never-silent** — an out-of-range `dec` is an explicit error in **every** mode (the
/// mode tunes certification, never fallibility). `ALL_MODES`: it must fail in all three, never
/// silently succeed in any.
#[test]
fn clause_c_out_of_range_is_error_in_every_mode() {
    let tern = out_of_range_ternary();
    assert_mode_scope(
        ModeScope::ALL_MODES,
        |mode| {
            ModeGatedSwapEngine::new(mode)
                .swap(&tern, &Repr::Binary { width: 8 }, &policy())
                .is_err()
        },
        "out-of-range dec is an explicit error",
    );
}

/// **(c) Axis-B never-silent** — an illegal `(width, trits)` pair is an explicit error in every mode.
#[test]
fn clause_c_illegal_pair_is_error_in_every_mode() {
    let a = byte_of(1);
    assert_mode_scope(
        ModeScope::ALL_MODES,
        |mode| {
            // (8, 1): Binary{8} ⊄ Ternary{1} — illegal.
            ModeGatedSwapEngine::new(mode)
                .swap(&a, &Repr::Ternary { trits: 1 }, &policy())
                .is_err()
        },
        "illegal pair is an explicit error",
    );
}

/// **(c) Axis-B opt-out is itself never-silent + orthogonal** — the explicit [`WrappingOpt`] marker
/// (RFC-0034 §10; M-791) is *absent by default* (the safe never-silent path needs no annotation) and,
/// when attached, does **not** silence Axis-A: a `wrapping` value keeps its honest guarantee + the
/// mode tag in every mode. Cross-mode: orthogonality holds in `ALL_MODES`.
#[test]
fn clause_c_wrapping_optout_is_explicit_and_orthogonal() {
    // Default: never-silent failability is on — no marker.
    assert!(
        byte_of(3).meta().wrapping_opt().is_none(),
        "Axis-B never-silent is the default — no wrapping marker without an explicit opt-out"
    );
    // When attached, it does not perturb the guarantee or the mode tag, in any mode.
    assert_mode_scope(
        ModeScope::ALL_MODES,
        |mode| {
            let meta = Meta::exact(Provenance::Root)
                .with_wrapping(WrappingOpt::new())
                .with_cert_mode(mode);
            let v = Value::new(
                Repr::Binary { width: 8 },
                Payload::Bits(vec![false; 8]),
                meta,
            )
            .unwrap();
            // Axis-A untouched (still Exact) AND the mode tag is intact AND the opt-out is visible.
            v.meta().wrapping_opt().is_some()
                && v.meta().guarantee() == GuaranteeStrength::Exact
                && v.meta().cert_mode() == mode
        },
        "explicit wrapping opt-out is visible and orthogonal to Axis-A + the mode tag",
    );
}

/// **(c) memory safety** — the trusted base is memory-safe **by construction, in every mode**: the
/// kernel crates this suite exercises (`mycelium-core`, `mycelium-cert`) compile under
/// `#![forbid(unsafe_code)]`, so *no* `CertMode` can introduce an `unsafe` escape (the guarantee is
/// compiler-enforced and mode-independent — RFC-0034 §3.3/§9, sharpening ADR-014).
///
/// **Guarantee tag: `Proven` (compiler-checked).** The basis is the crate-level `forbid` attribute,
/// re-asserted here so the conformance suite *records* the memory-safety clause rather than leaving
/// it implicit. The runtime assertion below is a tautology — `true` *is* memory-safe code running —
/// whose real witness is that this whole suite builds against the `forbid`-gated crates at all.
#[test]
fn clause_c_memory_safe_in_every_mode() {
    // Compile-time witness: the file header of mycelium-core/-cert is `#![forbid(unsafe_code)]`
    // (verified by `grep_check_forbid` below — a never-silent check, not a comment).
    for_each_mode(|mode| {
        // A safe, mode-tagged value is constructible in every mode — the runtime side of the
        // structural memory-safety guarantee (no mode reaches for unsafe).
        let v = byte_of(0);
        let tagged = v.meta().clone().with_cert_mode(mode);
        let out = Value::new(v.repr().clone(), v.payload().clone(), tagged);
        assert!(out.is_ok(), "a safe value must construct in {mode:?}");
    });
}

/// **(c) memory safety — the actual basis check.** The `Proven` memory-safety claim for the trusted
/// base rests on the crate-level `#![forbid(unsafe_code)]` attribute. This test *checks that
/// side-condition* (VR-5: a `Proven` claim is only allowed with its side-condition checked), reading
/// the kernel crates' source headers rather than asserting the property by fiat. If a future edit
/// removed the `forbid`, this fails loudly — the never-silent guard on the memory-safety clause.
#[test]
fn clause_c_trusted_base_forbids_unsafe() {
    // Resolve the workspace `crates/` dir relative to this test crate's manifest.
    let crates_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/mycelium-cert has a parent (crates/)");
    for krate in ["mycelium-core", "mycelium-cert"] {
        let lib = crates_dir.join(krate).join("src/lib.rs");
        let src =
            std::fs::read_to_string(&lib).unwrap_or_else(|e| panic!("read {}: {e}", lib.display()));
        assert!(
            src.contains("#![forbid(unsafe_code)]"),
            "{krate}/src/lib.rs must carry `#![forbid(unsafe_code)]` — the checked basis for the \
             RFC-0034 §3.3 memory-safety clause (Proven). Its removal would un-ground the claim."
        );
    }
}

// ===========================================================================
// Clauses (d)/(e)/(f) — relocated (M-882; see the module doc comment above)
// ===========================================================================
//
// These clauses were previously exercised here via `mycelium-proj`/`mycelium-spore`
// dev-dependencies, which introduced a dev-dep cycle back through `mycelium-l1` to this crate.
// None of the three actually reached any `mycelium-cert` code path (they only ever called
// `mycelium_proj::cert_scope::*` / `mycelium_spore::build_spore` on plain `mycelium-core` types),
// so the fix is a relocation, not a weakening: the identical properties are checked — more
// thoroughly, with `proptest` sweeps this suite never ran — in the crate that owns the logic:
//
// - clause (d) EXPLAIN-of-mode / generation≠consumption:
//   `crates/mycelium-proj/src/tests/cert_scope.rs`
//   (`explain_mode_is_available_in_every_mode_including_fast`,
//   `generate_mode_signal_is_available_in_every_mode`,
//   `lean_consumption_is_identical_to_explain_mode`,
//   `dialing_consumption_up_surfaces_more_without_rerun`,
//   `prop_explain_mode_available_in_every_mode`,
//   `prop_signal_generated_and_consumption_monotone`).
// - clause (e) spore mintability + mode-independent content identity:
//   `crates/mycelium-spore/src/tests/lib_tests.rs` (the M-789 DoD property test, exhaustive over
//   `CertMode::ALL`).
// - clause (f) cross-mode composition never upgrades:
//   `crates/mycelium-proj/src/tests/cert_scope.rs`
//   (`fast_into_certified_is_an_explicit_boundary_never_an_upgrade`,
//   `structural_exact_survives_the_boundary`, `prop_cross_mode_never_upgrades_strength`).

// ===========================================================================
// Capstone — clauses (a)/(b)/(c) converge on one gated swap (this crate's own witness)
// ===========================================================================

/// The capstone witness: a single bounded-ε swap, driven through the full
/// [`ModeGatedSwapEngine`] in every mode, exhibits the clause-(a) mode tag, the clause-(b) floor in
/// `fast`, and the cert-emission/checking scope — all consistent on one value, using only this
/// crate's own surface. (Clauses (d)/(e)/(f) — EXPLAIN-of-mode, spore mintability, cross-mode
/// composition — are verified in their owning crates; see the relocation note above. They are not
/// re-asserted here because doing so would require the `mycelium-proj`/`mycelium-spore`
/// dev-dependencies this change removes to break the M-882 cycle.)
#[test]
fn capstone_one_pipeline_exhibits_clauses_a_b_and_emission_scope() {
    let src = dense_f32(vec![1.0, 2.0, 1.0, 2.0]);
    for_each_mode(|mode| {
        let engine = ModeGatedSwapEngine::new(mode);
        let gated: GatedSwap = engine
            .swap_gated(
                &src,
                &Repr::Dense {
                    dim: 4,
                    dtype: ScalarKind::Bf16,
                },
                &policy(),
            )
            .expect("the bounded-ε swap succeeds in every mode (Axis-B not triggered)");

        // (a) the result carries its mode tag.
        assert_eq!(gated.value.meta().cert_mode(), mode, "(a) mode tag");

        // (b) in fast, never Empirical/Proven; outside fast, the earned strength is reachable.
        let strength = gated.value.meta().guarantee();
        match mode {
            CertMode::Fast => assert!(
                strength != GuaranteeStrength::Empirical && strength != GuaranteeStrength::Proven,
                "(b) fast must not carry {strength:?}"
            ),
            CertMode::Balanced | CertMode::Certified => assert!(
                strength.rank() <= GuaranteeStrength::Proven.rank(),
                "(b) the earned strength is reachable outside fast (got {strength:?})"
            ),
        }

        // (a/cert-emission) certificate present iff the mode emits (NON_FAST), checked iff Certified.
        assert_eq!(
            gated.certificate.is_some(),
            mode != CertMode::Fast,
            "certificate emitted iff non-fast"
        );
        assert_eq!(
            gated.check.is_some(),
            mode == CertMode::Certified,
            "certificate checked iff certified"
        );
    });
}

/// Mode-emission scope as a single cross-mode-negative assertion (the M-795 `EMIT_MODES` shape):
/// certificate **emission** is in `Balanced`+`Certified` (`NON_FAST`) and absent in `fast`; checking
/// is `CERTIFIED_ONLY`. Driven through the real engine on a bijective swap.
#[test]
fn capstone_emission_and_checking_scopes() {
    let src = byte_of(99);
    let engine_swap = |mode: CertMode| -> GatedSwap {
        ModeGatedSwapEngine::new(mode)
            .swap_gated(&src, &Repr::Ternary { trits: 6 }, &policy())
            .unwrap()
    };
    assert_mode_scope(
        ModeScope::NON_FAST,
        |mode| engine_swap(mode).certificate.is_some(),
        "swap-cert emission (Balanced + Certified, none in fast)",
    );
    assert_mode_scope(
        ModeScope::CERTIFIED_ONLY,
        |mode| engine_swap(mode).check.is_some(),
        "swap-cert checking (Certified only)",
    );
}

/// The never-silent engine guard, mode-scoped: in `Certified`, the [`SwapEngine::swap`] surface
/// returns the value on a *validating* check and would error on a non-validating one — a value is
/// never returned *as if validated* when it was not. Here the bijective swap validates, so all modes
/// return a value; the negative (a non-validating certified swap erroring) is covered by the
/// existing `mode.rs` suite — this asserts the positive end-to-end through the engine in every mode.
#[test]
fn capstone_engine_returns_value_in_every_mode_on_valid_swap() {
    let src = byte_of(42);
    for_each_mode(|mode| {
        let v = ModeGatedSwapEngine::new(mode)
            .swap(&src, &Repr::Ternary { trits: 6 }, &policy())
            .unwrap_or_else(|e: EvalError| {
                panic!("a valid bijective swap must return its value in {mode:?}: {e:?}")
            });
        assert_eq!(v.meta().cert_mode(), mode);
        assert_eq!(v.meta().guarantee(), GuaranteeStrength::Exact);
    });
}
