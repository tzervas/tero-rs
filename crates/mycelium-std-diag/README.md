# mycelium-std-diag

> Structured diagnostic surface for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.diag` is the Ring-1/Tier-A structured diagnostic surface. It re-exports the core
diagnostic types from `mycelium-diag` — `Diag`, `Severity`, `Locus`, `Trace`, `Code`,
and `ContentHash` — giving all Ring-2 crates and user code a single stable import point.
Diagnostics carry an explicit `Locus` (file/byte offset/field path) and `Severity` so
every reported issue is located and classifiable (RFC-0013 I1/never-silent).

## Key items

- `Diag` — the primary structured diagnostic value.
- `Severity` — diagnostic severity level (error, warning, note, etc.).
- `Locus` — source location carrier (file, byte offset, field path).
- `Trace` — diagnostic trace / call-chain context.
- `Code` — diagnostic code identifier.
- `ContentHash` — re-exported for attaching content identity to diagnostics.

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0013 (structured diagnostics, I1).
- Spec: `docs/spec/stdlib/diag.md` (M-510).

## Role in the workspace

Shared diagnostic vocabulary for compiler, interpreter, and stdlib error reporting. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-diag).
