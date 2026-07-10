# mycelium-std-io

> Single-consumption I/O and canonical serialization for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.io` couples two surfaces over Mycelium's content-addressed value model. The **serialize**
surface projects a `Value` to/from bytes (`serialize`/`deserialize`) or canonical JSON
(`to_json`/`from_json`) using the RFC-0001 self-describing wire form (schema travels with data).
The **io** surface moves bytes over affine `Source`/`Sink` handles (LR-8: single-consumption
enforced at the type level by Rust move semantics). `read_value` bridges both: deserialize
directly from a `Source`. Every truncated, malformed, or failed decode is an explicit, located
error — no op returns a partially-filled `Value` (C1/G2). The `GUARANTEE_MATRIX` has 8 rows,
one per exported op, asserted in tests (RFC-0016 §4.5).

## Key items

- `serialize` / `deserialize` — RFC-0001 wire-format round-trip; `Err(SerError)` on failure.
- `to_json` / `from_json` — canonical JSON projection (one projection; delegates to this crate from `std.fmt`).
- `Source` / `Sink` — affine I/O handles (LR-8); consumed exactly once.
- `read_all` — consume a `Source` and return all bytes.
- `read` — chunked read threading the `Source` linearly; declares `alloc(Budget)` effect.
- `write` — write bytes to a `Sink`, threading it linearly.
- `read_value` — deserialize a `Value` directly from a `Source`.
- `Budget` — explicit alloc budget (required; never inferred).
- `Substrate` — in-memory `Vec<u8>` backend for testing without OS I/O.
- `SerError`, `IoError`, `ReadValueError` — explicit, located error types.
- `Format` — wire-format selector (e.g. `Format::Wire`).

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0001 (value model / wire form); LR-8 (affine handles).
- Spec: `docs/spec/stdlib/io.md` (M-514, #155).

## Role in the workspace

Canonical serialization and byte-I/O layer; `std.fs` builds on `Source`/`Sink`; `std.fmt` delegates JSON to this crate. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-io).
