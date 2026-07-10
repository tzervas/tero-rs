# mycelium-std-dense

> Dense tensor and embedding operations for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.dense` is the Ring-1/Tier-A dense tensor surface. It provides elementwise arithmetic,
inner products, norms, and distance measures over `StdDense` (an f64 tensor backed by
`DenseSpace`). Every operation returns an explicit `OpBound` carrying the guarantee tag and
numeric error bound — never a silent approximation. Domain violations (mismatched shapes,
non-unit vectors, etc.) return `Err(StdDenseError)`.

## Key items

- `StdDense` — f64 dense tensor; the primary value type.
- `DenseSpace` — shape/dimension descriptor for a dense tensor.
- `OpBound` — per-result carrier: `(value, GuaranteeStrength, ErrorBound)`.
- `StdDenseError` — explicit error for domain violations (shape mismatch, etc.).

## Guarantee posture

- Integer ops: `Exact`.
- Float elementwise (`add`, `sub`, `scale`, `hadamard`): `Proven` (Q1 finalized, DN-16, M-512-checked).
- Float accumulation (`sum`, `l1_norm`, `dot`): `Empirical`.
- `l2_norm`, `cosine`: `Empirical` (FLAG Q2 — full proof pending).

## Design references

- RFC-0016 (core + standard library contract, C1–C6); ADR-010 (ε/δ bound kernels); RFC-0001 (value model).
- Spec: `docs/spec/stdlib/dense.md` (M-518).

## Role in the workspace

Dense numeric substrate for embedding, ML, and VSA workloads. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-dense).
