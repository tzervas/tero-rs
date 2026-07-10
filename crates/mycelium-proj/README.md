# mycelium-proj

> Project-metadata layer (M-359; DN-06 §6): the structured nodule header (`// @key: value`), the `mycelium-proj.toml` manifest (a minimal, dependency-free TOML-subset reader), the EXPLAIN-able top-down inheritance resolver, and `@certification` mode scoping.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-proj` is the project-metadata layer above the kernel (KC-3). It provides three pieces:
the structured nodule header (the `// @key: value` closed v0 key set that may follow the
`// nodule:` marker — unknown/duplicate keys and malformed values are explicit errors, G2); the
`mycelium-proj.toml` manifest parsed by a deliberately minimal, no-new-dependency TOML-subset
reader (adding a full TOML crate would be an ADR, not a build detail); and the EXPLAIN-able
top-down inheritance resolver (`in-file > manifest`) with per-field provenance so a field's
effective value and source are never ambient. Metadata is not identity (ADR-003): nothing here
perturbs a definition's content hash.

## Key items

- `parse_header` / `StructuredHeader` / `HeaderError` — the `// @key: value` structured header parser.
- `parse_manifest` / `Manifest` / `ManifestError` — the `mycelium-proj.toml` manifest reader.
- `resolve` / `explain` / `ResolvedHeader` — top-down inheritance resolver with per-field provenance.
- `cert_scope` — `@certification` mode resolution (RFC-0034 §6; M-790): `global > phylum > nodule` lattice, cross-mode-composition boundary, and the generation/consumption split (`ModeSignal` / `ConsumptionTier`).

## Design references

- M-359, M-790, M-792
- DN-06 §6
- RFC-0034 §6, RFC-0012
- ADR-003
- G2, VR-5, KC-3

## Role in the workspace

Depends on `mycelium-l1` and `mycelium-core`; consumed by `mycelium-fmt`, `mycelium-check`, `mycelium-lint`, `mycelium-doc`, `mycelium-lsp`, and `mycelium-cli`. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-proj).
