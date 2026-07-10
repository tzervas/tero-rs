# mycelium-numerics

> Verified-numerics foundation: `ErrorBound` (ε, affine arithmetic) + `ProbBound` (δ, union/apRHL) kernels meeting at one shared `{ε,δ,strength}` certificate with a tier-i Rust checker (ADR-010; E2-4).

**Tier:** kernel  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-numerics` provides the two bound kernels that honest approximate computation requires. The `error` kernel composes ε-magnitude bounds through affine arithmetic (`ErrorBound`, `AffineForm`); the `prob` kernel composes δ failure-probability bounds through the union bound and the apRHL `[SEQ]` rule (`ProbBound`, `ApRhlJudgment`). They are different monoids (a settled negative result per ADR-010/T0.1c) meeting at the shared `Certificate {ε, δ, strength}`, where `strength` composes by `meet`.

The tier-i Rust checker (`check_error_claim`/`check_union_claim`) re-derives a composition and rejects any claim tighter than the re-derivation — never a silent pass. The one sanctioned cross-kernel inference is `accuracy_to_probability`. Zero `unsafe` — compiler-enforced.

## Key items

- `Certificate` — the shared `{ε, δ, strength}` output consumed by `mycelium-cert` and the interpreter.
- `ErrorBound` / `AffineForm` / `NoiseSym` — affine-arithmetic ε composition.
- `ProbBound` / `ApRhlJudgment` — union-bound + apRHL δ composition.
- `check_error_claim` / `check_union_claim` — the tier-i re-derivation checker; rejects tighter-than-re-derived claims.
- `compose_error_bound` / `recompute_error` — bound composition helpers.
- `accuracy_to_probability` — the one sanctioned cross-kernel inference.

## Guarantee posture

`check_error_claim` and `check_union_claim` are `Proven` (re-derivation equality is the checked theorem). Three normative composition properties — Soundness, Monotonicity, Determinism (RFC-0001 §4.7) — are property-tested in `tests/properties.rs`. Source is ground truth.

## Design references

- ADR-010, ADR-011, E2-4, RFC-0001, RFC-0002

## Role in the workspace

Depends on `mycelium-core` (guarantee lattice, `Bound`/`BoundBasis` vocabulary). Used by `mycelium-interp`, `mycelium-cert`, and the differential test harness. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-numerics).
