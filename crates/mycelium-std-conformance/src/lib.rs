//! `mycelium-std-conformance` — the standard-library **port-conformance test harness** (M-971).
//!
//! This crate has **no library surface**: it exists only to host the `lib/std/<name>.myc`
//! port-differential integration tests in `tests/`, which were relocated here from `mycelium-l1`
//! (M-971) so the trusted `core`-tier L1 crate no longer dev-depends on the `std`-tier
//! `mycelium-std-<name>` oracle crates — an upward-tier edge the DN-68 acyclic-deps invariant
//! forbids (`core` must not depend on `std`, in any dependency kind). The differential legitimately
//! belongs in a crate that may depend on **both** the L1 execution stack and the std oracles; this
//! is that crate (tier `std`, a leaf that nothing depends on — stratum 0, in no dependency cycle).
//!
//! Each `tests/std_<name>.rs` loads the ported nodule verbatim (`include_str!` of
//! `lib/std/<name>.myc`) and drives the RFC-0007 §4.6 three-way differential through the shared
//! [`tests/harness`] fixture (L1-eval ≡ elaborate→L0-interp ≡ AOT, validated pairwise by the M-210
//! shared checker), then compares against the retained Rust oracle (RFC-0031 D6 — the oracle crates
//! are NOT retired). See DN-68 (`docs/notes/DN-68-Acyclic-Deps-Invariant.md`) for the invariant and
//! `xtask/deps-strata.toml` for this crate's stratum/tier registration.

// Intentionally empty: all logic lives in the `tests/` integration binaries and their shared
// `harness` module. Nothing outside this crate depends on it.
