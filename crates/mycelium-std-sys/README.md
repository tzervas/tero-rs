# mycelium-std-sys

> Audited FFI/syscall floor for the Mycelium standard library — the single `wild`-contact phylum.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`mycelium-std-sys` is the single, audited phylum for all FFI/OS-syscall contact in the Mycelium
standard library tree. By routing every low-level interface through this crate exclusively, the pure
`std` crates (`std-math`, `std-rand`, `std-fs`, `std-time`, …) can earn a `wild`-free badge — no
`unsafe` or FFI anywhere in their own code (RFC-0016 §9/LR-9). The `mycelium-std-sys-host` crate
wires `OsEntropy` and `OsClock` through this floor so `std.rand`/`std.time` bottom out here while
remaining `wild`-free. All functions carry the `Declared` guarantee tag (VR-5): no audited theorem
backs libm precision, OS entropy quality, FS semantics, or clock resolution in v0.

## Key items

- `math` — transcendental function floor (libm via Rust `f64` intrinsics).
- `rand::fill_bytes` — platform entropy floor (`/dev/urandom` or equivalent).
- `fs` — filesystem syscall floor (thin `std::fs` wrappers).
- `time::mono_nanos` / `time::wall_nanos` / `time::sleep_nanos` — OS clock floor.
- `io` — standard-stream I/O floor (stdin/stdout/stderr; RFC-0028 §4.5, M-722).
- `sys` — process/environment floor (exit, env vars, args; RFC-0028 §4.5, M-722).
- `guarantee_matrix` — per-module `Declared` tags encoded as data.

## Design references

- RFC-0016 §8-Q6 (std-sys floor); RFC-0028 §4.5 (host encoding); LR-9 (wild boundary).
- Tasks: M-722/M-723.
- Spec: `docs/spec/stdlib/sys.md`.

## Role in the workspace

The only crate in the std tree that may touch OS/FFI; sits below the std layer with no mycelium workspace dependencies. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-sys).
