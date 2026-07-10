# mycelium-std-vsa

> `std.vsa` — hypervector/VSA encoding capability surface: every approximating op exposes its guarantee tag and an inspectable trace, never a black box.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.vsa` is the Ring 1 / Tier A ergonomic capability surface over the landed VSA/HDC kernel
(M-513; RFC-0016 §4.2/§4.3). It exposes `bind`/`unbind`/`bundle`/`cleanup`/`permute`/`similarity`
and the sequence/set/role encoding family, plus a resonator-based reconstruction path for
factorization and role queries. Every `(model, op)` guarantee tag is read from the kernel matrix
(`mycelium_vsa::matrix_tag`), never fabricated, and the `GUARANTEE_MATRIX` encodes them as data
asserted in tests (RFC-0016 §4.5). Out-of-capacity, ambiguous, and model-mismatch outcomes are
explicit `Err` (C1); `cleanup` returns `(label, confidence, margin)` and a resonator run returns a
`ResonatorTrace` for full EXPLAIN capability (C3/G11).

## Key items

- `bind` / `unbind` — exact hypervector binding and unbinding.
- `bundle` / `cleanup` — bundling and winner-take-all cleanup; cleanup returns `(label, confidence, margin)`.
- `permute` / `unpermute` — permutation and inverse (exact).
- `similarity` — inner-product similarity between two hypervectors.
- `encode_seq` / `encode_set` — sequence and set encoding over VSA.
- `reconstruct_factors` / `reconstruct_role` — resonator-based factorization and role reconstruction with `ResonatorTrace`.
- `GUARANTEE_MATRIX` — per-`(model, op)` tags derived from the kernel matrix, asserted in tests.

## Guarantee posture

Per-`(model, op)` guarantee tags are derived from `mycelium_vsa::matrix_tag` (never fabricated) and
encoded in `GUARANTEE_MATRIX`, asserted in tests. Source is ground truth.

## Design references

- RFC-0016 §4.2/§4.3 (Ring 1/Tier A); RFC-0003 §4 (VSA guarantee matrix); RFC-0012 (ambient representation); ADR-003.
- Tasks: M-513.
- Spec: `docs/spec/stdlib/vsa.md`.

## Role in the workspace

Ring 1 / Tier A VSA/HDC capability surface; wraps `mycelium-vsa` without adding trusted code (KC-3). See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-vsa).
