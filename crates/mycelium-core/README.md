# mycelium-core

> Mycelium Core IR: `Value<Repr,Meta>`, the guarantee lattice, content-addressing, and the node grammar (RFC-0001).

**Tier:** kernel  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-core` is the trusted kernel: it defines every type that constitutes the Core IR — `Value<Repr, Meta>`, the guarantee lattice (`Exact ⊐ Proven ⊐ Empirical ⊐ Declared`), the bound vocabulary, the node grammar, and content-addressing via BLAKE3. The honesty invariants M-I1…M-I4 are enforced by construction (see `meta::Meta::new`). Zero `unsafe` — compiler-enforced (`#![forbid(unsafe_code)]`).

What is here so far: the guarantee `meet` composition and laws (M-102) and content-addressing (M-103). Serialization to the ratified JSON schemas (M-104) and the reference interpreter (M-110) are tracked separately.

## Key items

- `Value` — a `(Repr, Payload, Meta)` triple; the universal value type.
- `GuaranteeStrength` — the `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` lattice with `meet` composition.
- `Meta` / `Provenance` — per-value metadata: guarantee, bound, provenance, sparsity, and packing observations.
- `Node` — the Core IR node grammar (let, op, swap, construct, match, lam, app, fix, fix-group).
- `Repr` — the four paradigms: `Binary{width}`, `Ternary{trits}`, `Dense{dim,dtype}`, `Vsa{dim,model}`.
- `ContentHash` — BLAKE3-based content address; the identity of operations and policies.
- `DataRegistry` / `DataDecl` — algebraic data type declarations and constructor references.
- `WfError` — well-formedness errors for Core IR construction (RFC-0001 §4.3/§4.5).

## Guarantee posture

Per-op guarantee tags are carried by `Meta` and composed by `GuaranteeStrength::meet` — `Exact` is the identity, and any weaker strength propagates forward (never silently upgraded). Source is ground truth; the API enforces the M-I1…M-I4 invariants by construction.

## Design references

- RFC-0001 (r2–r5), ADR-014, DN-21

## Role in the workspace

The foundational crate depended on by every other crate. No upstream Mycelium dependencies. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-core).
