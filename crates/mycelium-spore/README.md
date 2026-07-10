# mycelium-spore

> spore — packaging and publishing (M-368; ADR-013): builds a content-addressed deployable `spore` from a `mycelium-proj.toml` project.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-spore` builds a `Spore` — the deployable unit that germinates into a colony (DN-06/Glossary). Identity is the content-addressed DAG: source code by BLAKE3 hash + resolved dependency edges + germination surface. Metadata (`version`/`authors`/`summary`) travels with the spore but never defines it (ADR-003). Two builds of the same code and deps produce the same spore hash regardless of the version label.

A missing or ambiguous publish input is an explicit `SporeError`, never a guess (G2): a phylum with no surface, a project with no sources, or a dependency with no `hash` is refused — no partial artifact. The v0 on-disk encoding is a named-provisional reproducible form (M-368 §9.1), superseded append-only when the RFC-0008 R2 wire-schema lands.

## Key items

- `Spore` — the content-addressed artifact: `id` (BLAKE3 over code+deps+surface), `kind`, `surface`, `sources`, `deps`, plus metadata `name`/`version`.
- `SourceFile` — a content-addressed project source file (`path` + `blake3:<hex>` hash).
- `ResolvedDep` — a pinned dependency edge (name, phylum, content-hash, optional version).
- `SporeError` — explicit refusals: `Publish` (missing/ambiguous input, exit 3), `Io` (exit 66).
- `registry::publish` / `resolve` — the content-addressed local registry (M-732).
- `registry::artifact_hash` — compute the spore identity hash.

## Design references

- ADR-003, ADR-013, DN-06, M-368, M-732, RFC-0008

## Role in the workspace

Depends on `mycelium-core` (content-address type) and `mycelium-proj` (manifest). Provides the `spore` CLI binary. Dev-dependencies include `proptest` for the registry hash-verification bound (M-732 DoD). See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-spore).
