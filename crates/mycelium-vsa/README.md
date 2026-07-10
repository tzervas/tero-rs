# mycelium-vsa

> VSA submodule: the `VsaModel` trait and its first model MAP-I, dependency-gated so the kernel stays small (RFC-0003; ADR-008; M-130).

**Tier:** kernel  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-vsa` provides the VSA algebra: the `VsaModel` trait defining the closed op set (`bind`, `unbind`, `bundle`, `permute`) and the first model **MAP-I** (multiply-add-permute, bipolar elementwise). The crate is dependency-gated: `mycelium-core` already type-checks `Repr::Vsa` mentions, so programs can name VSA values without pulling in this algebra (KC-3). The kernel stays small and auditable; VSA is opt-in.

For MAP-I: `bind`/`unbind` and `permute` are `Exact` (algebraic operations, elementwise product and cyclic shift on bipolar vectors). The `bundle` algebra ships here; the `Proven` Value-level bundle (which must carry the checked `CapacityBound` citing Clarkson/Thomas, M-131) is added in M-131 — the `Proven` tag is not stamped without a checked bound (VR-5). Zero `unsafe` — compiler-enforced.

## Key items

- `MapI` — MAP-I model: bind/unbind (elementwise bipolar product), permute (cyclic shift), bundle (elementwise superposition).
- `VsaModel` — the trait every model implements, declaring `intrinsic_guarantee` per op.
- `VsaOp` — the closed op set: `Bind`, `Unbind`, `Bundle`, `Permute`.
- `VsaError` — explicit refusals: `DimMismatch`, `EmptyBundle`, `InsufficientCapacity`, `DuplicateBundleItems`.
- `CleanupMemory` / `Match` — cleanup memory and nearest-neighbour match for decoding.
- `resonator::factorize` — resonator-network-based factorization for MAP-I.
- `decode_select` — `DecodeMethod` selection with mandatory `EXPLAIN` (RFC-0010).
- `matrix::RFC0003_MATRIX` — the RFC-0003 normative guarantee matrix.

## Guarantee posture

`bind`/`unbind`/`permute` are `Exact` (algebraic; no rounding). `bundle` in v1 (without M-131) emits values without a `Proven` capacity bound; the `Proven` bundle is deferred until the capacity theorem's side-conditions are checked (M-131). `VsaError::InsufficientCapacity` and `DuplicateBundleItems` enforce the theorem's preconditions rather than issuing an unbacked tag. Source is ground truth.

## Design references

- RFC-0003, RFC-0010, ADR-008, ADR-014, M-130, M-131, T2.6

## Role in the workspace

Depends on `mycelium-core` and `mycelium-select`. Used by `mycelium-cert` (Dense↔VSA swap). See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-vsa).
