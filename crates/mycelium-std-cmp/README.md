# mycelium-std-cmp

> Ordering, equality, and non-repr value conversions for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.cmp` provides the canonical ordering and equality traits (`MycEq`, `MycOrd`,
`MycPartialOrd`), min/max/clamp combinators, and the `Widen`/`Narrow` conversion pair.
All ops are tagged `Exact` in the guarantee matrix; `clamp` and `Narrow` are explicitly
fallible (`Result`) — never silent. The `Narrow` trait covers lossy or range-restricted
casts; `Widen` is lossless and total. `Bf16Bits` enables BF16→F32 lossless widening.

## Key items

- `Ordering` — `Less`/`Equal`/`Greater` enum.
- `MycEq`, `MycOrd`, `MycPartialOrd` — comparison traits.
- `myc_min`, `myc_max` — total min/max over `MycOrd` values.
- `myc_clamp` — range clamp; returns `Err(ClampError::InvertedBounds)` if `lo > hi`.
- `Widen<To>` — lossless total widening conversion.
- `Narrow<To>` — fallible narrowing; `Err(NarrowError::OutOfRange)` or `Err(NarrowError::NotRepresentable)`.
- `Bf16Bits` — BF16 bit-pattern newtype enabling lossless BF16→F32 widening.
- `GUARANTEE_MATRIX` — 9-row guarantee table (all `Exact`); asserted in tests (RFC-0016 §4.5).

## Guarantee posture

All nine ops carry `Exact` tags. Fallibility (`clamp`, `Narrow`) is an explicit `Result`
error — never a silent clamp, sentinel, or truncation (C1/G2).

## Design references

- RFC-0016 (core + standard library contract, C1–C6).
- Spec: `docs/spec/stdlib/cmp.md` (M-532).

## Role in the workspace

Ordering/conversion primitives depended on by Ring-1/Ring-2 crates. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-cmp).
