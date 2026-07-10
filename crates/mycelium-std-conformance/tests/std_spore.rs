//! Differential tests for `std.spore` (M-934, E29-1, kickoff `opp`) — the content-addressed
//! deployable-unit surface: identity carry + verification, the RFC-0003 §6 reconstruction
//! manifest with its FR-C2 resonator honesty ceiling, the ADR-013 deploy seam, and the
//! RFC-0016 §4.5 guarantee matrix as checked data.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-op three-way cases, and — the row this port owns
//! per the harness doc (§4) — the live comparisons against the **retained Rust oracle**,
//! `mycelium-std-spore` (RFC-0031 D6; the crate is NOT retired). String observables (hashes,
//! error/EXPLAIN renderings, matrix columns) reduce the `.myc` side to raw bytes ([`eval_bytes`])
//! and compare against the oracle's actual output **byte-for-byte**; scalar observables reduce to
//! bytes ([`eval_byte`], the `std_error.rs`/`std_recover.rs` precedent).
//!
//! # Content-address parity (the M-934 DoD extra — NO identity drift)
//! The `.myc` port **never mints a content hash** (FLAG-spore-2 in `lib/std/spore.myc`: BLAKE3 /
//! the canonical encoder is the RFC-0031 D1 kernel boundary; the `hash.*` prim is M-912-pending).
//! Identities are minted by the oracle (`SporeUnit::from_value` → the M-368 canonical encoder)
//! and injected into the `.myc` drivers verbatim; the differential then asserts the port carries
//! and compares them **byte-for-byte identically** to the oracle: same identity out
//! ([`oracle_identity_carries_verbatim`]), same verify outcome on the same hashes
//! ([`oracle_verify_round_trip_matches`], [`oracle_verify_mismatch_fields_match`]), same
//! determinism/distinctness behavior ([`oracle_from_value_determinism_and_distinctness`]), and
//! the same HashMismatch rendering naming both hashes. Because the `.myc` side consumes oracle
//! hashes verbatim and never computes one, identity drift is structurally impossible — and these
//! tests would catch any carry/comparison corruption.
//!
//! The **project-build** content-address path (`SporeUnit::from_manifest` — metadata invariance,
//! code-change sensitivity) is NOT drivable from this test: `from_manifest` takes a
//! `mycelium_proj::Manifest` and `mycelium-proj` is not a dev-dependency of `mycelium-l1` (the
//! oracle dev-dep set is orchestrator-owned — FLAGged, not edited). Those properties are covered
//! by the oracle's own in-crate tests (`cargo test -p mycelium-std-spore`, part of this task's
//! gate); the value-path determinism + content-sensitivity parity here exercises the same
//! canonical encoder end-to-end.
//!
//! # Surface-check (D5 row 1) and substitutions
//! See `lib/std/spore.myc`'s header for the full surface-check. PORTED: SporeErr (+ display),
//! the SporeUnit carry handle + identity/surface/manifest accessors, verify/verify_identity,
//! bytes_eq (composed from the D4 prims), ReconMode/DecodeProcedure/Basis/basis_strength,
//! MalformedManifest (+ display), ReconManifest (manifest_new/manifest_validate + accessors —
//! the FR-C2 ceiling), RegrowthResult (regrowth_new ceiling seal + strength predicates), the
//! deploy seam (germinate/detect_opaque_step/deploy_error_display/explain_deployed), and the
//! 15-row guarantee matrix. FLAGGED, not forced (VR-5/G2): the nodule name (`spore` is a
//! reserved keyword — `std.spores` substituted), content-hash minting (kernel D1; M-912),
//! from_manifest/from_value packaging (filesystem + kernel encoder), f64 bound scalars/`delta`/
//! `as_approx`, decimal integer rendering (AmbiguousInput display / `explain_spore`'s kernel
//! prefix), and `mycelium_vsa::Factorization` (generic payload substitution).
//!
//! Oracle-side reachability notes (each documented at its test): `RegrowthResult::new` needs a
//! `mycelium_vsa::Factorization` (not a dev-dep) — its ceiling refusal is covered three-way here
//! and behaviorally by the oracle's in-crate `regrowth_result_refuses_over_strength_basis`; the
//! oracle's `validate()`/`with_tampered_id` over-strength/tamper paths are `#[cfg(test)]`/
//! kernel-sealed — the `.myc` twins are compared against hand-built oracle values (`SporeErr` /
//! `MalformedManifest` are public, so the DISPLAY and field conventions compare live).
//!
//! # Pre-port polish (D5 row 2) — recorded clean
//! `mycelium-std-spore` froze under DN-66 (2026-07-01) and its surface carried no ambiguity this
//! port had to resolve; no behavior-neutral polish commit was needed (recorded here, not
//! skipped silently — G2). The one Rust-side edit is the lib.rs DN-66 note update (a `.myc` port
//! now exists; the crate is retained per D6 — the `std-ternary` M-933 precedent).
//!
//! # Honesty tags
//! - **`Declared`** — each ported op's type-level contract and the 15-row matrix transcription,
//!   carried at the SAME strength as `mycelium-std-spore`'s own guarantee matrix (VR-5: never
//!   upgraded in translation).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) AND the
//!   Rust-oracle differential below, both validated by trial on the programs in this file;
//!   neither is a machine-checked proof.

mod harness;

use mycelium_core::{
    binary::bits_to_int,
    bound::NormKind,
    content::operation_hash,
    recon::{DecodeProcedure, DecodeSpec},
    Bound, BoundBasis, BoundKind, ContentHash, CoreValue, Meta, Payload, Provenance, Repr, Value,
};
use mycelium_std_spore::{
    explain_deploy, germinate, identity as oracle_identity, verify as oracle_verify, DeployError,
    DeployResult, DeployTarget, MalformedManifest, ReconManifest, ReconMode, SporeErr, SporeUnit,
    MATRIX,
};

/// The std.spore nodule source, loaded at compile time — the single source of truth.
const SPORE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/spore.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(SPORE_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`] (same pattern as `std_recover.rs`).
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Shared driver fragments — the closed test vocabulary. Nullary generic constructors are pinned
// by annotated helper fns (the `std_recover.rs::empty_pol` precedent).
// ══════════════════════════════════════════════════════════════════════════════════════════════

const PRELUDE: &str = "fn no_manifest() => Option[ReconManifest] = None;\n\
fn value_surface() => Vec[Bytes] = Cons(\"value\", Nil);\n\
fn two_surface() => Vec[Bytes] = Cons(\"alpha\", Cons(\"beta\", Nil));\n\
fn one_codebook() => Vec[Bytes] = Cons(\"cb\", Nil);\n\
fn no_codebooks() => Vec[Bytes] = Nil;\n\
fn dim1024() => Binary{32} = 0b0000_0000_0000_0000_0000_0100_0000_0000;\n\
fn emp_basis() => Basis = EmpiricalFit(0b0000_0000_0000_0000_0000_0011_1110_1000, \"test\");\n\
fn prov_basis() => Basis = ProvenThm(\"test theorem\");\n\
fn decl_basis() => Basis = UserDeclared;\n";

/// Expected-side type mirrors — constructor order matches `lib/std/spore.myc` exactly
/// (structural identity for the CoreValue comparison).
const T_CORE: &str = "type Result[A, E] = Ok(A) | Err(E);\n\
type Option[A] = Some(A) | None;\n\
type Vec[A] = Nil | Cons(A, Vec[A]);\n\
type Unit = U;\n\
type Guarantee = GExact | GProven | GEmpirical | GDeclared;\n";

const T_SPORE: &str =
    "type SporeErr = HashMismatch(Bytes, Bytes) | PublishErr(Bytes) | IoErr(Bytes);\n";

const T_RECON: &str = "type ReconMode = IndexedRetrieval | CompositionalReconstruction;\n\
type DecodeProcedure = Cleanup | Resonator;\n\
type Basis = ProvenThm(Bytes) | EmpiricalFit(Binary{32}, Bytes) | UserDeclared;\n\
type MalformedManifest = ResonatorOverStrength | KernelWf;\n\
type ReconManifest = Manifest(ReconMode, Bytes, Binary{32}, Vec[Bytes], DecodeProcedure, Basis);\n";

const T_REGROWTH: &str = "type RegrowthResult[T] = Regrown(T, Basis);\n";

const T_DEPLOY: &str = "type DeployTarget = InMemory | Local(Bytes);\n\
type DeployVerification = Verification(Bool, Bool);\n\
type DeployError = MissingInput | AmbiguousInput(Vec[Bytes]) | DeployHashMismatch(Bytes, Bytes) | OpaqueStepDetected(Bytes);\n\
type DeployResult = Deployed(Bytes, DeployVerification) | Failed(DeployError);\n";

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Three-way differential cases (L1-eval ≡ elaborate→L0-interp ≡ AOT) — each against a
// hand-computed reference value (Declared data, Empirical agreement). Hash-shaped inputs here are
// SHORT placeholder byte strings (the depth-light structural cases); real oracle-minted 71-byte
// identities are exercised in the Rust-oracle section below.
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ── bytes_eq: the D4-prim equality composition ──────────────────────────────────────────────────

/// Equal byte strings compare `True` — including a length that exercises more than one 4-byte
/// recursion chunk (the depth-bounded frame design).
#[test]
fn bytes_eq_equal_is_true() {
    let driver = "fn main() => Bool = bytes_eq(\"blake3:abcdef\", \"blake3:abcdef\");";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("bytes_eq equal", &src, expected);
}

/// Same length, one divergent byte → `False` (the comparison half of a hash mismatch).
#[test]
fn bytes_eq_divergent_byte_is_false() {
    let driver = "fn main() => Bool = bytes_eq(\"blake3:abcdef\", \"blake3:abcdeX\");";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("bytes_eq divergent byte", &src, expected);
}

/// Different lengths → `False` without any per-byte probe (the length gate).
#[test]
fn bytes_eq_length_mismatch_is_false() {
    let driver = "fn main() => Bool = bytes_eq(\"blake3:ab\", \"blake3:abc\");";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("bytes_eq length mismatch", &src, expected);
}

// ── verify: equal → Ok(U); divergent → the named HashMismatch (C1/G2) ───────────────────────────

/// `verify_identity` on equal hashes is `Ok(U)`.
#[test]
fn verify_identity_equal_is_ok() {
    let driver =
        "fn main() => Result[Unit, SporeErr] = verify_identity(\"blake3:aa\", \"blake3:aa\");";
    let src = program(driver);
    let expected =
        format!("nodule ref;\n{T_CORE}{T_SPORE}fn main() => Result[Unit, SporeErr] = Ok(U);");
    assert_three_way("verify_identity equal Ok", &src, &expected);
}

/// A divergent recomputation is the explicit `Err(HashMismatch(declared, recomputed))` — both
/// hashes carried, `expected` = the DECLARED identity, `found` = the recomputed one (the oracle's
/// field convention) — never a silent accept (C1/G2/ADR-003).
#[test]
fn verify_identity_divergent_is_named_mismatch() {
    let driver =
        "fn main() => Result[Unit, SporeErr] = verify_identity(\"blake3:aa\", \"blake3:bb\");";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_SPORE}fn main() => Result[Unit, SporeErr] = Err(HashMismatch(\"blake3:aa\", \"blake3:bb\"));"
    );
    assert_three_way("verify_identity mismatch named", &src, &expected);
}

/// `verify` over the carry handle routes the DECLARED identity into the comparison.
#[test]
fn verify_spore_carry_round_trip_is_ok() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[Unit, SporeErr] = verify(spore_carry(\"blake3:id\", value_surface(), no_manifest()), \"blake3:id\");"
    );
    let src = program(&driver);
    let expected =
        format!("nodule ref;\n{T_CORE}{T_SPORE}fn main() => Result[Unit, SporeErr] = Ok(U);");
    assert_three_way("verify(carry) round-trip Ok", &src, &expected);
}

// ── manifest_of: None is honest absence, never a fabricated empty manifest (C1/G2) ──────────────

#[test]
fn manifest_of_is_none_when_absent() {
    let driver = format!(
        "{PRELUDE}fn main() => Option[ReconManifest] = manifest_of(spore_carry(\"blake3:id\", value_surface(), no_manifest()));"
    );
    let src = program(&driver);
    let expected =
        format!("nodule ref;\n{T_CORE}{T_RECON}fn main() => Option[ReconManifest] = None;");
    assert_three_way("manifest_of absent None", &src, &expected);
}

// ── manifest_new: the expressible kernel checks + the FR-C2 ceiling (all → KernelWf) ────────────

/// A valid IndexedRetrieval + Cleanup + EmpiricalFit manifest is accepted.
#[test]
fn manifest_new_valid_cleanup_is_accepted() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[ReconManifest, MalformedManifest] = manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Cleanup, emp_basis());"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_RECON}fn main() => Result[ReconManifest, MalformedManifest] = Ok(Manifest(IndexedRetrieval, \"MAP-I\", 0b0000_0000_0000_0000_0000_0100_0000_0000, Cons(\"cb\", Nil), Cleanup, EmpiricalFit(0b0000_0000_0000_0000_0000_0011_1110_1000, \"test\")));"
    );
    assert_three_way("manifest_new valid Ok", &src, &expected);
}

/// A Resonator + EmpiricalFit manifest is accepted (the canonical resonator case).
#[test]
fn manifest_new_resonator_empirical_is_accepted() {
    let driver = format!(
        "{PRELUDE}fn main() => Bool = match manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Resonator, emp_basis()) {{ Ok(_) => True, Err(_) => False }};"
    );
    let src = program(&driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("manifest_new resonator+empirical Ok", &src, expected);
}

/// A Resonator + UserDeclared manifest is accepted — the rule is "must not EXCEED Empirical",
/// not "must equal Empirical" (the oracle's `resonator_declared_basis_is_accepted` guard).
#[test]
fn manifest_new_resonator_declared_is_accepted() {
    let driver = format!(
        "{PRELUDE}fn main() => Bool = match manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Resonator, decl_basis()) {{ Ok(_) => True, Err(_) => False }};"
    );
    let src = program(&driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("manifest_new resonator+declared Ok", &src, expected);
}

/// A Resonator + ProvenThm manifest is REFUSED via the build path as `KernelWf` — the oracle's
/// `new()` maps the kernel's FR-C2 refusal to `MalformedManifest::KernelWf`, NOT
/// `ResonatorOverStrength` (that variant is `validate`'s — the next case).
#[test]
fn manifest_new_resonator_proven_is_kernel_wf() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[ReconManifest, MalformedManifest] = manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Resonator, prov_basis());"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_RECON}fn main() => Result[ReconManifest, MalformedManifest] = Err(KernelWf);"
    );
    assert_three_way("manifest_new resonator+proven KernelWf", &src, &expected);
}

/// The expressible kernel well-formedness checks each refuse with `KernelWf`: empty model,
/// zero dim, empty codebooks (FLAG-spore-5 — the recipe/decode-param invariants live on
/// un-mirrored kernel fields).
#[test]
fn manifest_new_kernel_wf_checks_refuse() {
    for (label, call) in [
        (
            "empty model",
            "manifest_new(IndexedRetrieval, \"\", dim1024(), one_codebook(), Cleanup, emp_basis())",
        ),
        (
            "zero dim",
            "manifest_new(IndexedRetrieval, \"MAP-I\", b32_zero(), one_codebook(), Cleanup, emp_basis())",
        ),
        (
            "empty codebooks",
            "manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), no_codebooks(), Cleanup, emp_basis())",
        ),
    ] {
        let driver = format!(
            "{PRELUDE}fn main() => Result[ReconManifest, MalformedManifest] = {call};"
        );
        let src = program(&driver);
        let expected = format!(
            "nodule ref;\n{T_CORE}{T_RECON}fn main() => Result[ReconManifest, MalformedManifest] = Err(KernelWf);"
        );
        assert_three_way(&format!("manifest_new {label} KernelWf"), &src, &expected);
    }
}

/// `manifest_validate` — the carry-in defense-in-depth path — refuses an over-strength resonator
/// manifest with the explicit `ResonatorOverStrength` (reachable here because `.myc` constructors
/// are open; in Rust the kernel seals it — see the nodule's substitution note).
#[test]
fn manifest_validate_over_strength_is_refused() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[ReconManifest, MalformedManifest] = manifest_validate(Manifest(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Resonator, prov_basis()));"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_RECON}fn main() => Result[ReconManifest, MalformedManifest] = Err(ResonatorOverStrength);"
    );
    assert_three_way("validate over-strength refused", &src, &expected);
}

/// `declared_strength` is derived from the basis — `GEmpirical` for an EmpiricalFit bound
/// (never fabricated, never upgraded — VR-5).
#[test]
fn declared_strength_is_basis_derived() {
    let driver = format!(
        "{PRELUDE}fn main() => Guarantee = match manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Resonator, emp_basis()) {{ Ok(m) => declared_strength(m), Err(_) => GExact }};"
    );
    let src = program(&driver);
    let expected = format!("nodule ref;\n{T_CORE}fn main() => Guarantee = GEmpirical;");
    assert_three_way("declared_strength basis-derived", &src, &expected);
}

// ── regrowth: the FR-C2 ceiling seal (Q4a) ──────────────────────────────────────────────────────

/// `regrowth_new` REFUSES a basis stronger than Empirical — the explicit ResonatorOverStrength,
/// never a silent accept (the oracle's `regrowth_result_refuses_over_strength_basis` twin;
/// oracle-side construction needs `mycelium_vsa::Factorization`, not a dev-dep — covered
/// behaviorally by the oracle's own in-crate test, which this task's gate runs).
#[test]
fn regrowth_new_refuses_over_strength_basis() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[RegrowthResult[Binary{{8}}], MalformedManifest] = regrowth_new(0b0000_0001, prov_basis());"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_RECON}{T_REGROWTH}fn main() => Result[RegrowthResult[Binary{{8}}], MalformedManifest] = Err(ResonatorOverStrength);"
    );
    assert_three_way("regrowth_new over-strength refused", &src, &expected);
}

/// An Empirical-basis regrowth is accepted; its strength predicates read off the basis honestly.
#[test]
fn regrowth_empirical_predicates_hold() {
    let driver = format!(
        "{PRELUDE}fn main() => Bool = match regrowth_new(0b0000_0001, emp_basis()) {{ Ok(r) => bool_and(is_empirical(r), bool_not(is_declared(r))), Err(_) => False }};"
    );
    let src = program(&driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("regrowth empirical predicates", &src, expected);
}

// ── germinate: the ADR-013 deploy seam, check order mirrored (G2/G11) ───────────────────────────

/// A `Local` target with an empty path refuses with `MissingInput` before any other work (G2).
#[test]
fn germinate_local_empty_path_is_missing_input() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[DeployResult, DeployError] = germinate(spore_carry(\"blake3:id\", value_surface(), no_manifest()), \"blake3:id\", Local(\"\"));"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_DEPLOY}fn main() => Result[DeployResult, DeployError] = Err(MissingInput);"
    );
    assert_three_way("germinate empty path MissingInput", &src, &expected);
}

/// An `InMemory` target over a multi-symbol surface refuses with ALL candidates listed (G11).
#[test]
fn germinate_in_memory_multi_surface_is_ambiguous() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[DeployResult, DeployError] = germinate(spore_carry(\"blake3:id\", two_surface(), no_manifest()), \"blake3:id\", InMemory);"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_DEPLOY}fn main() => Result[DeployResult, DeployError] = Err(AmbiguousInput(Cons(\"alpha\", Cons(\"beta\", Nil))));"
    );
    assert_three_way("germinate ambiguous candidates listed", &src, &expected);
}

/// A hash mismatch at deploy time is the named `DeployHashMismatch(declared, recomputed)` —
/// no silent overwrite, no partial deploy (C4/ADR-003).
#[test]
fn germinate_hash_mismatch_is_named() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[DeployResult, DeployError] = germinate(spore_carry(\"blake3:aa\", value_surface(), no_manifest()), \"blake3:bb\", InMemory);"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_DEPLOY}fn main() => Result[DeployResult, DeployError] = Err(DeployHashMismatch(\"blake3:aa\", \"blake3:bb\"));"
    );
    assert_three_way("germinate mismatch named", &src, &expected);
}

/// A clean single-surface InMemory deploy succeeds with the fully-checked verification record
/// (both invariants True — germinate refuses first otherwise).
#[test]
fn germinate_success_carries_verification() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[DeployResult, DeployError] = germinate(spore_carry(\"blake3:id\", value_surface(), no_manifest()), \"blake3:id\", InMemory);"
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\n{T_CORE}{T_DEPLOY}fn main() => Result[DeployResult, DeployError] = Ok(Deployed(\"blake3:id\", Verification(True, True)));"
    );
    assert_three_way("germinate success verification", &src, &expected);
}

/// The v0 pipeline detects no opaque step for either supported target (the oracle stub,
/// mirrored — Declared strength, carried).
#[test]
fn detect_opaque_step_is_none_for_supported_targets() {
    for (label, target) in [("InMemory", "InMemory"), ("Local", "Local(\"/tmp/x\")")] {
        let driver = format!("{PRELUDE}fn main() => Option[Bytes] = detect_opaque_step({target});");
        let src = program(&driver);
        let expected = format!("nodule ref;\n{T_CORE}fn main() => Option[Bytes] = None;");
        assert_three_way(&format!("detect_opaque_step {label} None"), &src, &expected);
    }
}

// ── guarantee matrix: structural checks over the 15-row table ───────────────────────────────────

/// Every row states a non-empty never_silent_property / op / guarantee (C1/G2 completeness),
/// the 12 selecting/converting rows are EXPLAIN-able (C3), and the 3 pure-read accessor rows are
/// not — the typed halves of guarantee_matrix.rs's tests.
#[test]
fn matrix_structural_checks_hold() {
    for (label, call) in [
        (
            "never_silent nonempty",
            "all_never_silent_nonempty(matrix())",
        ),
        ("ops nonempty", "all_ops_nonempty(matrix())"),
        ("guarantees nonempty", "all_guarantees_nonempty(matrix())"),
        (
            "manifest rows explainable",
            "manifest_rows_are_explainable()",
        ),
        (
            "accessor rows not explainable",
            "accessor_rows_not_explainable()",
        ),
    ] {
        let driver = format!("fn main() => Bool = {call};");
        let src = program(&driver);
        let expected = "nodule ref;\nfn main() => Bool = True;";
        assert_three_way(&format!("matrix {label}"), &src, expected);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Rust-oracle differential (D5 row 4 + the M-934 content-address DoD) — wired against the
// RETAINED `mycelium-std-spore` crate (RFC-0031 D6: NOT retired). String observables reduce the
// `.myc` side to raw bytes and compare against the oracle's actual strings byte-for-byte; the
// three-way obligation is covered by the cases above (these helpers bridge to the oracle only —
// the `std_recover.rs::eval_byte` precedent).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Decode a `Binary{8}` [`CoreValue`] to its signed byte (the `std_error.rs` codec).
fn extract_byte(cv: &CoreValue) -> i8 {
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Binary{{8}} repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bits(bits) => bits_to_int(bits) as i8,
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Decode a `Bytes` [`CoreValue`] to its raw byte string.
fn extract_bytes(cv: &CoreValue) -> Vec<u8> {
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Bytes repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bytes(b) => b.clone(),
        other => panic!("expected a Bytes payload, got {other:?}"),
    }
}

/// Run `driver`'s `main` through the L1 evaluator and return the resulting [`CoreValue`].
///
/// The depth budget is raised over `DEFAULT_DEPTH` (64): L1 charges depth **per AST node**
/// (eval.rs A4-03 — a *nesting* ceiling), and comparing a full 71-byte canonical
/// `blake3:<hex>` identity is a legitimately deep, terminating computation. Raising the budget
/// is the eval.rs-sanctioned mechanism ("the host stack is not the limit"); the three-way
/// differential cases above stay on the default budget with short strings.
fn eval_core(driver: &str) -> CoreValue {
    use mycelium_l1::elab::build_registry;
    use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

    let src = program(driver);
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("parse failed: {e}")))
        .unwrap_or_else(|e| panic!("check failed: {e}"));
    let mono = monomorphize(&env, "main").unwrap_or_else(|e| panic!("monomorphize failed: {e}"));
    let registry = build_registry(&mono).unwrap_or_else(|e| panic!("build_registry failed: {e}"));
    let val = Evaluator::new(&mono)
        .with_depth(1024)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("L1-eval failed: {e}"));
    val.to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("result is outside the r3 data fragment"))
}

/// Run a `main => Binary{8}` driver and decode the byte.
fn eval_byte(driver: &str) -> i8 {
    extract_byte(&eval_core(driver))
}

/// Run a `main => Bytes` driver and return the raw byte string.
fn eval_bytes(driver: &str) -> Vec<u8> {
    extract_bytes(&eval_core(driver))
}

/// The `.myc`-side reducers appended to oracle drivers.
const ORACLE_REDUCERS: &str = "fn verify_code(r: Result[Unit, SporeErr]) => Binary{8} = match r { Ok(_) => 0b0000_0000, Err(e) => match e { HashMismatch(_, _) => 0b0000_0001, PublishErr(_) => 0b0000_0010, IoErr(_) => 0b0000_0011 } };\n\
fn mm_expected(r: Result[Unit, SporeErr]) => Bytes = match r { Ok(_) => \"ok\", Err(e) => match e { HashMismatch(expected, _) => expected, PublishErr(m) => m, IoErr(m) => m } };\n\
fn mm_found(r: Result[Unit, SporeErr]) => Bytes = match r { Ok(_) => \"ok\", Err(e) => match e { HashMismatch(_, found) => found, PublishErr(m) => m, IoErr(m) => m } };\n\
fn dep_code(r: Result[DeployResult, DeployError]) => Binary{8} = match r { Ok(d) => match d { Deployed(_, _) => 0b0000_0000, Failed(_) => 0b0111_1111 }, Err(e) => match e { MissingInput => 0b0000_0001, AmbiguousInput(_) => 0b0000_0010, DeployHashMismatch(_, _) => 0b0000_0011, OpaqueStepDetected(_) => 0b0000_0100 } };\n\
fn dep_id(r: Result[DeployResult, DeployError]) => Bytes = match r { Ok(d) => match d { Deployed(id, _) => id, Failed(_) => \"failed\" }, Err(_) => \"err\" };\n\
fn dep_hash_ok(r: Result[DeployResult, DeployError]) => Bytes = match r { Ok(d) => match d { Deployed(_, v) => match v { Verification(h, _) => bool_text(h) }, Failed(_) => \"failed\" }, Err(_) => \"err\" };\n\
fn dep_opaque_ok(r: Result[DeployResult, DeployError]) => Bytes = match r { Ok(d) => match d { Deployed(_, v) => match v { Verification(_, o) => bool_text(o) }, Failed(_) => \"failed\" }, Err(_) => \"err\" };\n";

/// The Rust-oracle byte-value fixture (the oracle's own `spore_ops.rs` test constants).
fn byte_value(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed byte value")
}

const BITS_A: [bool; 8] = [true, false, true, true, false, false, true, false];
const BITS_B: [bool; 8] = [false, true, false, false, true, true, false, true];

/// Mint a real content-addressed identity via the oracle's canonical encoder (FLAG-spore-2:
/// the `.myc` side never computes one — it carries exactly this output).
fn oracle_hash(bits: [bool; 8]) -> String {
    let spore = SporeUnit::from_value(&byte_value(bits), None).expect("from_value is Ok");
    oracle_identity(&spore).as_str().to_owned()
}

// ── content-address parity (the M-934 DoD extra — NO identity drift) ────────────────────────────

/// The oracle's canonical encoder is deterministic (same value → same hash) and
/// content-sensitive (different value → different hash); the `.myc` `bytes_eq` agrees with the
/// oracle's `ContentHash` equality on exactly those hashes — the carried identities are the
/// oracle's own, byte-for-byte.
#[test]
fn oracle_from_value_determinism_and_distinctness() {
    let h1 = oracle_hash(BITS_A);
    let h2 = oracle_hash(BITS_A);
    let hb = oracle_hash(BITS_B);
    assert_eq!(h1, h2, "oracle: from_value identity must be deterministic");
    assert_ne!(h1, hb, "oracle: different values must hash differently");
    assert!(
        h1.starts_with("blake3:"),
        "oracle: canonical identity is blake3-prefixed: {h1}"
    );

    // The .myc side agrees on the SAME hashes (equality ↔ oracle ContentHash equality). These
    // full-length comparisons run through the raised-depth oracle bridge (see [`eval_core`]);
    // the three-way obligation for bytes_eq is covered by the short-string cases above.
    let eq_driver = format!("fn main() => Bytes = bool_text(bytes_eq(\"{h1}\", \"{h2}\"));");
    assert_eq!(
        eval_bytes(&eq_driver),
        b"true",
        ".myc bytes_eq must agree with oracle ContentHash equality (same hash)"
    );
    let ne_driver = format!("fn main() => Bytes = bool_text(bytes_eq(\"{h1}\", \"{hb}\"));");
    assert_eq!(
        eval_bytes(&ne_driver),
        b"false",
        ".myc bytes_eq must agree with oracle ContentHash inequality (distinct hashes)"
    );
}

/// A carried oracle identity comes back out VERBATIM — the `.myc` handle neither transforms nor
/// truncates the kernel-minted hash (no identity drift; the hash must match the oracle's output).
#[test]
fn oracle_identity_carries_verbatim() {
    let spore = SporeUnit::from_value(&byte_value(BITS_A), None).expect("builds");
    let h = oracle_identity(&spore).as_str().to_owned();
    let driver = format!(
        "{PRELUDE}fn main() => Bytes = identity(spore_carry(\"{h}\", value_surface(), no_manifest()));"
    );
    assert_eq!(
        eval_bytes(&driver),
        h.as_bytes(),
        "the .myc identity must be the oracle's hash byte-for-byte (ADR-003 — no drift)"
    );
}

/// `verify` round-trip parity on a REAL oracle-minted identity: the oracle's
/// `verify(from_value(v))` is Ok, and the `.myc` `verify` over the same carried hash is Ok too.
#[test]
fn oracle_verify_round_trip_matches() {
    let spore = SporeUnit::from_value(&byte_value(BITS_A), None).expect("builds");
    assert_eq!(
        oracle_verify(&spore),
        Ok(()),
        "oracle: verify(from_value(v)) must be Ok (round-trip)"
    );
    let h = oracle_identity(&spore).as_str().to_owned();
    let driver = format!(
        "{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = verify_code(verify(spore_carry(\"{h}\", value_surface(), no_manifest()), \"{h}\"));"
    );
    assert_eq!(
        eval_byte(&driver),
        0,
        "the .myc verify must be Ok on the oracle's own identity (round-trip parity)"
    );
}

/// Mismatch parity on real oracle hashes: two distinct oracle-minted identities produce the
/// named `HashMismatch` whose `expected`/`found` fields are the DECLARED and RECOMPUTED hashes
/// byte-for-byte (the oracle's field convention; G11 — both hashes carried).
#[test]
fn oracle_verify_mismatch_fields_match() {
    let ha = oracle_hash(BITS_A);
    let hb = oracle_hash(BITS_B);
    let call = format!("verify(spore_carry(\"{ha}\", value_surface(), no_manifest()), \"{hb}\")");
    let code_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = verify_code({call});");
    assert_eq!(eval_byte(&code_driver), 1, "mismatch must be HashMismatch");
    let exp_driver = format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Bytes = mm_expected({call});");
    assert_eq!(
        eval_bytes(&exp_driver),
        ha.as_bytes(),
        "HashMismatch.expected must be the DECLARED oracle hash byte-for-byte"
    );
    let found_driver = format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Bytes = mm_found({call});");
    assert_eq!(
        eval_bytes(&found_driver),
        hb.as_bytes(),
        "HashMismatch.found must be the RECOMPUTED oracle hash byte-for-byte"
    );
}

/// `SporeErr` Display parity — the `.myc` rendering equals the oracle's `to_string()`
/// byte-for-byte for all three variants (HashMismatch carries two REAL oracle hashes, so the
/// diagnostic-names-both-hashes property (G11) is compared on genuine identities).
#[test]
fn oracle_spore_err_display_matches() {
    let ha = oracle_hash(BITS_A);
    let hb = oracle_hash(BITS_B);
    let mismatch = SporeErr::HashMismatch {
        expected: ContentHash::parse(&ha).expect("oracle hash parses"),
        found: ContentHash::parse(&hb).expect("oracle hash parses"),
    };
    let cases: [(&str, String, String); 3] = [
        (
            "HashMismatch",
            format!("HashMismatch(\"{ha}\", \"{hb}\")"),
            mismatch.to_string(),
        ),
        (
            "PublishErr",
            "PublishErr(\"no sources: nothing to package\")".to_owned(),
            SporeErr::PublishErr("no sources: nothing to package".to_owned()).to_string(),
        ),
        (
            "IoErr",
            "IoErr(\"read failed: x.myc\")".to_owned(),
            SporeErr::IoErr("read failed: x.myc".to_owned()).to_string(),
        ),
    ];
    for (label, myc_err, oracle_display) in cases {
        let driver = format!("fn main() => Bytes = spore_err_display({myc_err});");
        assert_eq!(
            eval_bytes(&driver),
            oracle_display.as_bytes(),
            "{label}: .myc display must equal the oracle's Display byte-for-byte"
        );
    }
}

// ── manifest oracle parity (built from mycelium-core types — the kernel seam) ───────────────────

fn empirical_bound() -> Bound {
    Bound {
        kind: BoundKind::Probability { delta: 0.05 },
        basis: BoundBasis::EmpiricalFit {
            trials: 1000,
            method: "test".to_owned(),
        },
    }
}

fn proven_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: 0.01,
            norm: NormKind::L2,
        },
        basis: BoundBasis::ProvenThm {
            citation: "test theorem".to_owned(),
        },
    }
}

fn declared_bound() -> Bound {
    Bound {
        kind: BoundKind::Probability { delta: 0.1 },
        basis: BoundBasis::UserDeclared,
    }
}

fn cleanup_decode() -> DecodeSpec {
    DecodeSpec {
        procedure: DecodeProcedure::Cleanup,
        cleanup_threshold: Some(0.3),
        factors: None,
        iteration_budget: None,
        cleanup: None,
        beta: None,
        tau_lock: None,
        init: None,
        seed: None,
    }
}

fn resonator_decode() -> DecodeSpec {
    DecodeSpec {
        procedure: DecodeProcedure::Resonator,
        cleanup_threshold: None,
        factors: Some(vec![operation_hash("factor-a"), operation_hash("factor-b")]),
        iteration_budget: Some(50),
        cleanup: None,
        beta: None,
        tau_lock: None,
        init: None,
        seed: None,
    }
}

/// Drive the oracle's `ReconManifest::new` with a decode/bound pair and reduce the outcome to a
/// comparable code (Ok=0, ResonatorOverStrength=1, KernelWf=2).
fn oracle_manifest_code(decode: DecodeSpec, bound: Bound) -> i8 {
    match ReconManifest::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        1024,
        vec![operation_hash("codebook")],
        None,
        decode,
        bound,
    ) {
        Ok(_) => 0,
        Err(MalformedManifest::ResonatorOverStrength) => 1,
        Err(MalformedManifest::KernelWf) => 2,
    }
}

/// The build-path outcome parity table: the `.myc` `manifest_new` agrees with the LIVE oracle's
/// `ReconManifest::new` on accept/refuse AND the refusal variant, for every decode×basis pairing
/// the mirror carries (FLAG-spore-4: the bound's ε/δ scalars are kernel-side; the basis is the
/// honesty-relevant carry).
#[test]
fn oracle_manifest_new_outcomes_match() {
    let cases: [(&str, &str, &str, DecodeSpec, Bound); 5] = [
        (
            "cleanup+empirical",
            "Cleanup",
            "emp_basis()",
            cleanup_decode(),
            empirical_bound(),
        ),
        (
            "cleanup+proven",
            "Cleanup",
            "prov_basis()",
            cleanup_decode(),
            proven_bound(),
        ),
        (
            "resonator+empirical",
            "Resonator",
            "emp_basis()",
            resonator_decode(),
            empirical_bound(),
        ),
        (
            "resonator+declared",
            "Resonator",
            "decl_basis()",
            resonator_decode(),
            declared_bound(),
        ),
        (
            "resonator+proven",
            "Resonator",
            "prov_basis()",
            resonator_decode(),
            proven_bound(),
        ),
    ];
    for (label, myc_dec, myc_basis, decode, bound) in cases {
        let expected = oracle_manifest_code(decode, bound);
        let driver = format!(
            "{PRELUDE}fn main() => Binary{{8}} = match manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), {myc_dec}, {myc_basis}) {{ Ok(_) => 0b0000_0000, Err(e) => match e {{ ResonatorOverStrength => 0b0000_0001, KernelWf => 0b0000_0010 }} }};"
        );
        assert_eq!(
            eval_byte(&driver),
            expected,
            "{label}: manifest_new outcome must match the live oracle (FR-C2 ceiling parity)"
        );
    }
}

/// `declared_strength` parity: the oracle's strength for a resonator+empirical manifest is
/// `Empirical` and never stronger (rank ≥ 2); the `.myc` mirror reads the same strength off the
/// same basis (the shared lattice code map — VR-5: exact tag equality, never a weaker check).
#[test]
fn oracle_declared_strength_matches() {
    let m = ReconManifest::new(
        ReconMode::IndexedRetrieval,
        "MAP-I",
        1024,
        vec![operation_hash("codebook")],
        None,
        resonator_decode(),
        empirical_bound(),
    )
    .expect("valid resonator manifest");
    let oracle_rank = i8::try_from(m.declared_strength().rank()).expect("rank fits");
    let driver = format!(
        "{PRELUDE}fn main() => Binary{{8}} = match manifest_new(IndexedRetrieval, \"MAP-I\", dim1024(), one_codebook(), Resonator, emp_basis()) {{ Ok(m) => strength_rank(declared_strength(m)), Err(_) => 0b0111_1111 }};"
    );
    assert_eq!(
        eval_byte(&driver),
        oracle_rank,
        "declared_strength must agree with the oracle through the kernel rank map (VR-5)"
    );
    assert!(
        oracle_rank >= 2,
        "FR-C2: resonator strength never exceeds Empirical"
    );
}

/// `MalformedManifest` Display parity — both variants, byte-for-byte against the live oracle.
#[test]
fn oracle_malformed_display_matches() {
    for (label, myc_err, oracle_display) in [
        (
            "ResonatorOverStrength",
            "ResonatorOverStrength",
            MalformedManifest::ResonatorOverStrength.to_string(),
        ),
        (
            "KernelWf",
            "KernelWf",
            MalformedManifest::KernelWf.to_string(),
        ),
    ] {
        let driver = format!("fn main() => Bytes = malformed_display({myc_err});");
        assert_eq!(
            eval_bytes(&driver),
            oracle_display.as_bytes(),
            "{label}: .myc display must equal the oracle's Display byte-for-byte"
        );
    }
}

// ── deploy oracle parity ────────────────────────────────────────────────────────────────────────

/// Deploy parity on a REAL oracle spore (single-surface `spore(v)` — surface `["value"]`, the
/// from_value shape the `.myc` `value_surface()` fixture mirrors): the oracle germinates to
/// `Deployed` with both invariants verified; the `.myc` germinate over the SAME carried identity
/// agrees on outcome, deployed id (byte-for-byte), and both verification booleans.
/// (The multi-surface AmbiguousInput oracle path needs a `from_manifest` project build —
/// `mycelium-proj` is not a dev-dep; covered by the oracle's in-crate deploy tests + the
/// three-way ambiguous case above.)
#[test]
fn oracle_germinate_in_memory_deploy_matches() {
    let spore = SporeUnit::from_value(&byte_value(BITS_A), None).expect("builds");
    let result = germinate(&spore, &DeployTarget::InMemory).expect("oracle deploy succeeds");
    let DeployResult::Deployed {
        spore_id,
        verification,
    } = &result
    else {
        panic!("oracle: single-surface InMemory deploy must be Deployed");
    };
    let h = oracle_identity(&spore).as_str().to_owned();
    assert_eq!(spore_id.as_str(), h, "oracle: deployed id is the identity");

    let call = format!(
        "germinate(spore_carry(\"{h}\", value_surface(), no_manifest()), \"{h}\", InMemory)"
    );
    let code_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = dep_code({call});");
    assert_eq!(
        eval_byte(&code_driver),
        0,
        "deploy outcome must be Deployed on BOTH sides"
    );
    let id_driver = format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Bytes = dep_id({call});");
    assert_eq!(
        eval_bytes(&id_driver),
        spore_id.as_str().as_bytes(),
        "the deployed spore-id must match the oracle byte-for-byte (no identity drift)"
    );
    let hash_ok_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Bytes = dep_hash_ok({call});");
    assert_eq!(
        eval_bytes(&hash_ok_driver),
        if verification.content_hash_canonical {
            b"true".as_slice()
        } else {
            b"false".as_slice()
        },
        "content_hash_canonical must match the oracle"
    );
    let opaque_ok_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Bytes = dep_opaque_ok({call});");
    assert_eq!(
        eval_bytes(&opaque_ok_driver),
        if verification.no_opaque_lowering {
            b"true".as_slice()
        } else {
            b"false".as_slice()
        },
        "no_opaque_lowering must match the oracle"
    );
}

/// `Local` target with an empty path — `MissingInput` on BOTH sides (G2: refused before any
/// other work, never a guessed default).
#[test]
fn oracle_germinate_empty_path_matches() {
    let spore = SporeUnit::from_value(&byte_value(BITS_A), None).expect("builds");
    let err = germinate(
        &spore,
        &DeployTarget::Local {
            path: String::new(),
        },
    )
    .expect_err("oracle refuses an empty path");
    assert_eq!(err, DeployError::MissingInput);

    let h = oracle_identity(&spore).as_str().to_owned();
    let driver = format!(
        "{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = dep_code(germinate(spore_carry(\"{h}\", value_surface(), no_manifest()), \"{h}\", Local(\"\")));"
    );
    assert_eq!(
        eval_byte(&driver),
        1,
        "empty path must be MissingInput on BOTH sides"
    );
}

/// A populated `Local` path deploys on BOTH sides (the v0 stub's Local target performs the same
/// structural checks as InMemory without the ambiguity gate).
#[test]
fn oracle_germinate_local_path_matches() {
    let spore = SporeUnit::from_value(&byte_value(BITS_A), None).expect("builds");
    let result = germinate(
        &spore,
        &DeployTarget::Local {
            path: "/tmp/spore-target".to_owned(),
        },
    )
    .expect("oracle Local deploy succeeds");
    assert!(matches!(result, DeployResult::Deployed { .. }));

    let h = oracle_identity(&spore).as_str().to_owned();
    let driver = format!(
        "{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = dep_code(germinate(spore_carry(\"{h}\", value_surface(), no_manifest()), \"{h}\", Local(\"/tmp/spore-target\")));"
    );
    assert_eq!(
        eval_byte(&driver),
        0,
        "Local deploy must be Deployed on BOTH sides"
    );
}

/// `explain_deploy` parity for the Deployed arm — the `.myc` `explain_deployed` equals the
/// oracle's EXPLAIN string byte-for-byte (C3/VR-4: it names both the content-hash check and the
/// opaque-lowering check; determinism is the oracle's own `Exact` row).
#[test]
fn oracle_explain_deployed_matches() {
    let spore = SporeUnit::from_value(&byte_value(BITS_A), None).expect("builds");
    let result = germinate(&spore, &DeployTarget::InMemory).expect("deploys");
    let oracle_explain = explain_deploy(&result);
    let h = oracle_identity(&spore).as_str().to_owned();
    let driver =
        format!("fn main() => Bytes = explain_deployed(\"{h}\", Verification(True, True));");
    assert_eq!(
        eval_bytes(&driver),
        oracle_explain.as_bytes(),
        "the Deployed-arm EXPLAIN must equal the oracle's byte-for-byte (C3/VR-4)"
    );
}

/// `DeployError` Display parity for the three renderable variants (FLAG-spore-6: the
/// AmbiguousInput rendering interpolates a decimal count — not expressible; the `.myc` side
/// returns an explicit None for it, asserted here too — never a fabricated rendering).
#[test]
fn oracle_deploy_error_display_matches() {
    let ha = oracle_hash(BITS_A);
    let hb = oracle_hash(BITS_B);
    let cases: [(&str, String, String); 3] = [
        (
            "MissingInput",
            "MissingInput".to_owned(),
            DeployError::MissingInput.to_string(),
        ),
        (
            "HashMismatch",
            format!("DeployHashMismatch(\"{ha}\", \"{hb}\")"),
            DeployError::HashMismatch {
                expected: ha.clone(),
                actual: hb.clone(),
            }
            .to_string(),
        ),
        (
            "OpaqueStepDetected",
            "OpaqueStepDetected(\"jit-compile\")".to_owned(),
            DeployError::OpaqueStepDetected {
                step: "jit-compile".to_owned(),
            }
            .to_string(),
        ),
    ];
    for (label, myc_err, oracle_display) in cases {
        let driver = format!(
            "fn main() => Bytes = match deploy_error_display({myc_err}) {{ Some(s) => s, None => \"none\" }};"
        );
        assert_eq!(
            eval_bytes(&driver),
            oracle_display.as_bytes(),
            "{label}: .myc display must equal the oracle's Display byte-for-byte"
        );
    }
    // The FLAGged variant: an explicit None, never a fabricated rendering (G2).
    let driver = "fn main() => Bytes = match deploy_error_display(AmbiguousInput(Cons(\"a\", Cons(\"b\", Nil)))) { Some(_) => \"some\", None => \"none\" };";
    assert_eq!(
        eval_bytes(driver),
        b"none",
        "AmbiguousInput rendering is FLAG-spore-6 — explicit None, not a fabrication"
    );
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Guarantee-matrix oracle parity (the `std_diag.rs`/`std_recover.rs` precedent, strengthened):
// every expected value is computed LIVE from `mycelium_std_spore::MATRIX`, and every STRING
// column of every row is compared byte-for-byte — a transcription slip in the `.myc` table flips
// the oracle and fails the case.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The `.myc` row constructors, in `MATRIX` order (the parity loop's index map).
const MYC_ROW_FNS: [&str; 15] = [
    "row_build",
    "row_build_value",
    "row_identity",
    "row_explain",
    "row_manifest_of",
    "row_validate",
    "row_manifest_hash",
    "row_mode",
    "row_declared_strength",
    "row_reconstruct",
    "row_deploy",
    "row_germinate",
    "row_verify_hash_canonical",
    "row_no_opaque",
    "row_explain_deploy",
];

/// The `n`-deep `add_u` chain `matrix_len` expands to (the `std_diag.rs` provenance
/// convention: recompute via the SAME prims, not a bare literal).
fn myc_len_chain(n: u8) -> String {
    let mut expr = "0b0000_0000".to_owned();
    for _ in 0..n {
        expr = format!("add_u(0b0000_0001, {expr})");
    }
    expr
}

/// `matrix_len(matrix())` equals the live oracle's `MATRIX.len()` (15 rows), three-way.
#[test]
fn matrix_len_matches_rust_oracle_row_count() {
    let expected_count = u8::try_from(MATRIX.len()).expect("row count fits u8");
    let driver = "fn main() => Binary{8} = matrix_len(matrix());";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\nfn main() => Binary{{8}} = {};",
        myc_len_chain(expected_count)
    );
    assert_three_way("matrix_len == rust MATRIX.len()", &src, &expected);
}

/// Full per-row, per-column parity against the LIVE oracle: op / guarantee / fallibility /
/// effects / never_silent_property byte-for-byte, and explain_able through the shared
/// "true"/"false" rendering. This is the row-4 differential for the matrix data (Declared
/// transcription, Empirical agreement).
#[test]
fn matrix_rows_match_rust_oracle_by_column() {
    assert_eq!(
        MATRIX.len(),
        MYC_ROW_FNS.len(),
        "oracle row count and .myc row-constructor count must agree"
    );
    for (i, row_fn) in MYC_ROW_FNS.iter().enumerate() {
        let row = &MATRIX[i];
        let string_columns: [(&str, &str, &str); 5] = [
            ("op", "row_op", row.op),
            ("guarantee", "row_guarantee", row.guarantee),
            ("fallibility", "row_fallibility", row.fallibility),
            ("effects", "row_effects", row.effects),
            (
                "never_silent_property",
                "row_never_silent",
                row.never_silent_property,
            ),
        ];
        for (col, accessor, oracle_str) in string_columns {
            let driver = format!("fn main() => Bytes = {accessor}({row_fn}());");
            assert_eq!(
                eval_bytes(&driver),
                oracle_str.as_bytes(),
                "row {i} ({row_fn}) column {col} must equal the live oracle byte-for-byte"
            );
        }
        let driver = format!("fn main() => Bytes = bool_text(row_explainable({row_fn}()));");
        assert_eq!(
            eval_bytes(&driver),
            if row.explain_able {
                b"true".as_slice()
            } else {
                b"false".as_slice()
            },
            "row {i} ({row_fn}) explain_able must equal the live oracle"
        );
    }
}
