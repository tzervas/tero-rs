# mycelium-std-ternary

> `std.ternary` — balanced ternary and bit/trit capability surface: exact arithmetic, packed-ternary codecs, and a mandatory EXPLAIN for every packing decision.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.ternary` is the ergonomic, documented home for Mycelium's ternary differentiator (FR-M2;
M-111): first-class `Trit`/`Bit` primitives, exact balanced-ternary integer arithmetic with
explicit out-of-range, and packed-ternary helpers (I2S/TL1/TL2 codecs, RFC-0004 §5) that expose
their scheme via `scheme_of`/`explain` — packing is never a hidden lowering pass (C3/NFR-1/DN-01).
All ops tag `Exact`; the range boundary is fallibility, not a weakened tag. The guarantee matrix
is encoded as data and asserted in tests (RFC-0016 §4.5).

## Key items

- `Trit` / `Bit` — first-class balanced-ternary and binary primitives (FR-M2).
- `add` / `neg` / `sub` / `mul` — exact balanced-ternary integer arithmetic (out-of-range → `None`, never a clamp).
- `int_to_trits` / `trits_to_int` — exact integer ↔ trit-sequence codec; explicit `None` on overflow.
- `pack` / `unpack` — I2S/TL1/TL2 packing/unpacking (RFC-0004 §5), returning `PackError` on failure.
- `scheme_of` / `explain` — inspect or project any `Packed` value's scheme to an `ExplainRecord` (C3).
- `guarantee_matrix` — all ops encoded as data with `Exact` tags, asserted in tests.

## Guarantee posture

Every op is `Exact` — the balanced-ternary algebra and the I2S/TL1/TL2 codecs are exact; fallibility
is the never-silent out-of-range guard, not approximation. Source is ground truth.

## Design references

- RFC-0016 §4.2/§4.3 (Ring 1/Tier A); RFC-0004 §5 (packed ternary); RFC-0012 (ambient representation); DN-01 (packing); ADR-003.
- Tasks: M-517.
- Spec: `docs/spec/stdlib/ternary.md`.

## Role in the workspace

Ring 1 / Tier A exact ternary capability surface; wraps `mycelium-core::ternary` and adds no trusted code (KC-3). See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-ternary).
