# mycelium-dense

> Dense paradigm operational surface: typed, dimension-tracked `Dense{dim,dtype}` values and elementwise ops with honest per-op rounding bounds (RFC-0001 §4.1; RFC-0002 §5; M-230).

**Tier:** kernel  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-dense` provides the Dense-paradigm operation surface — the Dense analogue of the VSA `VsaModel` surface. Dimension and dtype are part of the type (`DenseSpace` binds both); a mismatch is a typed error, never a silent broadcast or coercion. Per the transparency rule every op carries an honest per-op tag: `neg` is `Exact`; `add`/`sub`/`scale` carry a per-element relative ε (`Bound{Error{eps, Rel}}`) with a `ProvenThm` basis citing Higham 2002, Thm 2.2, with its side-conditions checked per element. A violated side-condition is an explicit `DenseError`, never an unbacked bound.

`F16`/`F64` dtypes and approximate sources are explicitly out of scope in v1 — refused via `DenseError::UnsupportedDtype` and `DenseError::ApproximateSource` respectively — because the magnitude-aware composition rule is still open (M-204/M-211).

## Key items

- `DenseSpace` — a `(dim, dtype)` typed space; the entry point for all Dense ops.
- `DenseOp` — the closed op set: `Add`, `Sub`, `Neg`, `Scale`.
- `DenseError` — exhaustive explicit refusal type (dim/dtype mismatch, non-finite, not-on-grid, subnormal, approximate source, overflow, unsupported dtype).
- `F32_OP_REL_EPS` / `BF16_OP_REL_EPS` — the unit-roundoff constants with their citations.
- `DENSE_MIN_NORMAL` — the smallest positive normal magnitude (the theorem's side-condition boundary).

## Guarantee posture

`add`/`sub`/`scale` carry `Proven` per-element relative-ε bounds: the Higham 2002, Thm 2.2 side-conditions are checked per element and a violation returns an explicit `DenseError` (never a fabricated bound, VR-5). `neg` is `Exact`. Measurement helpers `dot`/`similarity` return bare `f64` with no `Meta` tag. Source is ground truth.

## Design references

- RFC-0001, RFC-0002, M-204, M-211, M-230

## Role in the workspace

Depends on `mycelium-core` only. Used by `mycelium-cert` (Dense↔VSA swap). See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-dense).
