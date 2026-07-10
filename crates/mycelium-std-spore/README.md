# mycelium-std-spore

> `std.spore` — content-addressed deployable unit and reconstruction-manifest library surface over the `mycelium-spore` packager.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.spore` is the ergonomic, value-semantic library face of the ADR-013 content-addressed
deployable unit and the RFC-0003 §6 reconstruction manifest. It consumes the landed `mycelium-spore`
packager (M-368) and adds no new hash or reconstruction logic (KC-3). A spore's identity is its
canonical content hash — metadata is not identity (C4/ADR-003). Probabilistic VSA regrowth is tagged
`Empirical` at most, carries its δ, and is never `Proven` (FR-C2/VR-5); the `ReconManifest::validate`
enforcer rejects any manifest whose resonator bound exceeds `Empirical`. Native deploy/germination
(`germinate`, `explain_deploy`) is implemented via M-620.

## Key items

- `SporeUnit` — the deployable unit value; identity is its `ContentHash`.
- `identity` / `verify` / `manifest_of` / `explain_spore` — core spore ops.
- `ReconManifest` / `ReconMode` / `RegrowthResult` — reconstruction manifest and regrowth result.
- `germinate` / `explain_deploy` / `DeployTarget` / `DeployResult` / `DeployError` — native deploy surface (M-620).
- `MATRIX` — per-op guarantee tags encoded as data, asserted in tests (RFC-0016 §4.5).
- `SporeErr` / `MalformedManifest` — never-silent error types (C1/G2).

## Guarantee posture

Per-op guarantee tags are encoded in `guarantee_matrix::MATRIX` and asserted in tests. VSA
regrowth is `Empirical` at most — never `Proven` (VR-5/FR-C2). Source is ground truth.

## Design references

- ADR-013 (content-addressed deployable); RFC-0003 §6 (reconstruction manifest); RFC-0016 §4.5; ADR-003.
- Tasks: M-522 (#163), M-620 (deploy).
- Spec: `docs/spec/stdlib/spore.md`.

## Role in the workspace

Ring 1 / Tier A deployable-unit library; composes `mycelium-spore`, `mycelium-std-vsa`, and `mycelium-std-numerics`. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-spore).
