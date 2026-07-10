//! `mycelium-std-sys` — audited FFI/syscall floor for the Mycelium standard library.
//!
//! # Purpose
//!
//! This crate is the **single, audited phylum** for all `wild`/FFI/OS-syscall contact in the
//! Mycelium standard library tree. By routing every low-level interface through `std-sys`
//! exclusively, the pure `std` crates (`std-math`, `std-rand`, `std-fs`, `std-time`, …) can
//! earn a **`wild`-free badge** as soon as they wire through this phylum — no `unsafe` or
//! FFI contact anywhere in their own code (RFC-0016 §9).
//!
//! **Wiring** the pure std crates through this floor (M-722/M-723) now exists: the
//! `mycelium-std-sys-host` crate supplies `OsEntropy` (fills `std-rand`'s `EntropySource` from
//! [`rand::fill_bytes`]) and `OsClock` (drives `std-time`'s `ClockSource` from [`time`]), so
//! `std.rand`/`std.time` bottom out in this audited floor while staying `wild`-free. The remaining
//! step — the Mycelium-surface `wild:` per-op byte encoding that makes the byte-oriented
//! [`io`]/[`fs`] ops reachable from a `wild { io.write(…) }` block — is the RFC-0028 §4.4 host
//! encoding, deferred to the `@std-sys` author and not yet committed (tracked in E14-1). The pure
//! `std-math` path still uses Rust's `f64` intrinsics as a `Declared` placeholder, FLAGged here.
//!
//! # Honesty
//!
//! All functions in this crate carry the **`Declared`** guarantee tag (RFC-0016 §4.1 C2 /
//! VR-5). No audited theorem backs the libm precision, OS entropy quality, FS semantics, or
//! clock resolution. Promotion requires:
//! - `Empirical`: documented test coverage with measured error bounds.
//! - `Proven`: a verified theorem whose side-conditions are checked.
//!
//! Neither is established in v0 of this crate.
//!
//! # LR-9 rationale — `std-sys` as a phylum boundary
//!
//! LR-9 mandates that `wild` (unsafe FFI) appear in **exactly one place** in the std tree —
//! this crate. Placing syscall contact here (rather than scattered across individual std crates)
//! means the `wild` audit surface is bounded, inspectable, and `EXPLAIN`-able (G2/G11/SC-3).
//! The rest of the std library stays pure Rust with no `unsafe` blocks, satisfying KC-3.
//!
//! # Modules
//!
//! - [`math`] — transcendental function floor (libm via Rust `f64` intrinsics).
//! - [`rand`] — platform entropy floor (`fill_bytes`).
//! - [`fs`] — filesystem syscall floor (thin `std::fs` wrappers).
//! - [`time`] — OS clock floor (wall + monotonic + sleep).
//! - [`io`] — standard-stream I/O floor (stdin/stdout/stderr; RFC-0028 §4.5, M-722).
//! - [`sys`] — process / environment floor (exit, env vars, args; RFC-0028 §4.5, M-722).
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/sys.md` (spec status:
//! Accepted (2026-06-21)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.

#![forbid(unsafe_code)]

pub mod fs;
pub mod guarantee_matrix;
pub mod io;
pub mod math;
pub mod rand;
pub mod sys;
pub mod time;

// In-crate test modules (test-layout rule: logic files carry no test code). Migrated lazily
// (M-797): `sys` lives here; the other modules still use inline `#[cfg(test)] mod tests` until
// each is touched.
#[cfg(test)]
mod tests;
