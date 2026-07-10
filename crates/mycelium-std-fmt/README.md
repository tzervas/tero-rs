# mycelium-std-fmt

> Dual human/machine projection — display, debug, and JSON formatting for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.fmt` is the Ring-2/Tier-B dual projection surface. `display` and `debug` render a
`Value` to a human-readable `Text`, with optional `Budget` for bounded output; `display_bounded`
returns a `Truncation` indicating whether the result was `Complete` or `Elided`. `to_json` and
`from_json` provide the one canonical JSON projection, delegating to `mycelium-std-io` (ratified
2026-06-19). All five exported ops are tagged `Exact` in the guarantee matrix; `from_json` is
fallible (`Err(FromJsonError)`) — never silent on malformed input (C1/G2).

## Key items

- `display` — human-readable projection of a `Value` to `Text`.
- `debug` — machine-diagnostic projection of a `Value` to `Text`.
- `display_bounded` — bounded display returning `Truncation` (Complete or Elided{omitted, marker}).
- `to_json` — canonical JSON projection; `Err(ToJsonError)` on failure.
- `from_json` — canonical JSON parse; `Err(FromJsonError)` on malformed input.
- `Text` — immutable UTF-8 text carrier.
- `Budget` — explicit output-size budget (required; never inferred from ambient context).
- `Truncation` — `Complete` or `Elided { omitted: usize, marker: &str }`.
- `Rendering` — a completed display result with its truncation status.
- `Json` — the canonical JSON value type.
- `GUARANTEE_MATRIX` — 5-row guarantee table (all `Exact`); asserted in tests.

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0013 (structured diagnostics / EXPLAIN); ADR-003 (projection, not identity).
- Spec: `docs/spec/stdlib/fmt.md` (M-533).

## Role in the workspace

Human-readable and machine-readable output for all `std` crates and user-facing tooling. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-fmt).
