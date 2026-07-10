# mycelium-std-error

> Option/Result combinators and recoverable-error surface for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.error` is the Ring-2 combinator surface for `Option` and `Result`. It re-exports the
standard functional combinators (`map`, `and_then`, `or_else`, `filter`, `zip`, `flatten`,
`transpose`, `inspect`, etc.) together with `RefusalRecord` and `SubstitutionRecord` for
auditable substitution of refused values. The `Outcome`/`RecoverOutcome`/`Resolution`
surface (from `mycelium-std-recover`) provides classified error handling with an explicit
recovery trace (RFC-0014).

## Key items

- `map`, `map_err`, `and_then`, `or_else`, `filter` — standard Option/Result combinators.
- `ok_or`, `ok_or_else`, `unwrap_or`, `unwrap_or_else` — Option→Result and default-value ops.
- `unwrap`, `expect`, `unwrap_err` — intentional-panic extractors (use sparingly; always explicit).
- `zip`, `flatten`, `transpose`, `inspect`, `inspect_err`, `ok`, `scan` — structural combinators.
- `RefusalRecord` — auditable record of a value refusal (C3/SC-3 EXPLAIN artifact).
- `SubstitutionRecord` — auditable record of a substituted default.
- `Outcome`, `RecoverOutcome`, `Resolution`, `handle_classified` — classified recovery surface (RFC-0014).

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0014 (declared effects / classified errors).
- Spec: `docs/spec/stdlib/error.md` (M-527).

## Role in the workspace

Ergonomic Option/Result handling layer for all Ring-2 std crates and user code. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-error).
