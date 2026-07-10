# mycelium-std-sys-host

> Production host wiring: connects the pure std crates' injectable seams to the audited `mycelium-std-sys` OS floor.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

The pure `std` crates (`mycelium-std-rand`, `mycelium-std-time`, …) keep OS contact behind injectable
seams — `EntropySource`, `ClockSource` — so they stay `wild`-free and testable with deterministic
stubs. The audited OS floor lives in exactly one place: `mycelium-std-sys` (LR-9/RFC-0016 §8-Q6).
`mycelium-std-sys-host` is the production glue that fills those seams with the real floor: `OsEntropy`
fills `std-rand`'s `EntropySource` from `mycelium-std-sys::rand::fill_bytes`, and `OsClock` drives
`std-time`'s `ClockSource` from `mycelium-std-sys::time`. No `unsafe`, no kernel coupling; the
dependency arrow stays honest: pure std → (seam) ← host wiring → floor.

## Key items

- `OsEntropy` — `EntropySource` backed by the audited `std-sys` OS entropy floor (M-723).
- `OsClock` — `ClockSource` backed by the audited `std-sys` time floor (monotonic + wall + logical).

## Guarantee posture

Every read is `Declared` — a genuine OS source, but no checked precision/quality theorem. Failures
are explicit `Err` (no silent zero-fill, no clock wrap/clamp). Source is ground truth.

## Design references

- RFC-0028 §4.5 (host encoding); RFC-0016 §8-Q6 (std-sys floor); LR-9 (wild boundary).
- Tasks: M-722/M-723.

## Role in the workspace

The sole crate that depends on both the audited OS floor and the pure std crates; wires the production entropy and clock sources without touching `unsafe`. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-sys-host).

## Stability (DN-66 freeze, 2026-07-01)

This crate's public API (`OsEntropy`, `OsClock`) is the **frozen baseline** per
[DN-66](../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md). Unlike
its 25 siblings, `mycelium-std-sys-host` has **no `docs/spec/stdlib/sys-host.md`** yet (DN-66 §4.a
FLAGs this as a follow-up); the "Overview"/"Key items"/"Guarantee posture" sections above are the
current documented contract until a formal spec lands. No `.myc` port exists, so the RFC-0031 D6
retirement trigger has not fired and no item here is `#[deprecated]`.
