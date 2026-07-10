# mycelium-diag

> Canonical RFC-0013 structured-diagnostic record types — the failure-legibility substrate (`Diag`/`Severity`/`Locus`/`Trace`/`Code`) consumed by `std.diag`, `std.recover`, and `std.testing`.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-diag` is a small kernel crate that owns the single canonical RFC-0013 `Diag` record type,
extracted from the scattered definitions that had lived across `mycelium-check`, `mycelium-l1`,
`mycelium-interp`, and `mycelium-lsp`. A `Diag` is additive over an explicit error: it presents a
failure, it never is the failure's control flow, and presentation never gates propagation. A missing
locus is `None` (explicit), never a fabricated zero (G2). The crate provides human-readable and
lossless JSON dual projections of the same canonical record, plus a content hash (BLAKE3 over the
canonical fields, presentation-invariant per ADR-003). `mycelium-std-diag` re-exports and
ergonomically wraps these types; it does not redefine them.

## Key items

- `Diag` — the canonical RFC-0013 structured-diagnostic record.
- `Diag::human` — human-readable view with content id.
- `Diag::machine` — lossless JSON machine record (round-trips via `Diag::from_json`).
- `Diag::content_hash` — deterministic BLAKE3 over canonical fields, presentation-invariant (ADR-003).

## Design references

- M-510, M-520
- RFC-0013, RFC-0014
- ADR-003, G2, G11, KC-3

## Role in the workspace

Provides the canonical diagnostic record below the stdlib layer; depends on `mycelium-core`; consumed by `std.diag`, `std.recover`, and `std.testing`. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-diag).
