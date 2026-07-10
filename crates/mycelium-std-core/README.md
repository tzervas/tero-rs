# mycelium-std-core

> Ring-0 prelude for the Mycelium standard library — re-exports of the core value model.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.core` is the Ring-0 stdlib prelude: it re-exports the essential kernel types and functions
from `mycelium-core` so that all Ring-1/Ring-2 crates and user code import from one stable
surface. All nine exported operations are tagged `Exact` in the guarantee matrix — they are pure,
total functions over primitive types with no approximation. The `error_scaffold` module supplies
the shared `StdError` marker trait and `impl_std_error!` macro used by every other std crate.

## Key items

- `Value`, `Repr`, `Meta`, `Payload` — the RFC-0001 value model core types.
- `GuaranteeStrength`, `Bound`, `BoundBasis` — per-op tag lattice (`Exact ⊐ Proven ⊐ Empirical ⊐ Declared`).
- `ContentHash`, `Trit`, `Provenance` — identity, ternary digit, and provenance chain.
- `repr_of`, `meta_of`, `guarantee_of`, `bound_of`, `provenance_of` — total query functions (all `Exact`).
- `prelude` — curated re-export set for user-facing code.
- `error_scaffold` — shared `StdError` marker + `impl_std_error!` macro.
- `GUARANTEE_MATRIX` — 9-row guarantee table asserted in tests (RFC-0016 §4.5).

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0001 (value model).
- Spec: `docs/spec/stdlib/core.md` (M-515, #166).

## Role in the workspace

Foundation prelude for all `mycelium-std-*` crates; no user code should need to import `mycelium-core` directly. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-core).
