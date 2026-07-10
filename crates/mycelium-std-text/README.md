# mycelium-std-text

> `std.text` — UTF-8 string type and operations: parse returns a `Result`, never a sentinel; lossy transcoding is always an explicit, named op.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.text` is the value-semantic, immutable UTF-8 string surface every program needs (M-524, #165):
construction from bytes/chars, slicing and indexing on validated char boundaries, a parse family
(`str → T` always returning `Result`), and encoding/transcoding between UTF-8 and other encodings.
The honesty crux is two-part: `parse` never silently coerces a malformed input to a sentinel (`0`,
`false`, `""`); and the only path to U+FFFD substitution is the explicitly-named `to_latin1_lossy`
whose `Lossy<Vec<u8>>` return type carries the substitution count — data cannot be lost silently.
All ops are `Exact` and effect-free; the guarantee matrix is asserted in tests (RFC-0016 §4.5).

## Key items

- `Text` — immutable, value-semantic UTF-8 string type (C4/ADR-003).
- `from_utf8` / `from_utf16` — strict construction; `Err` on invalid/lossy input, never silent U+FFFD.
- `parse_int` / `parse_bool` — `Result<T, ParseErr>`, never a sentinel on failure (C1/G2).
- `to_latin1` — strict Latin-1 encoding; `Err` on unmappable chars (C1).
- `to_latin1_lossy` — explicit lossy path; returns `Lossy<Vec<u8>>` carrying the substitution count (G2).
- `slice` / `char_at` — `Err` on off-boundary or out-of-range, never a silent truncation or sentinel char.
- `concat` / `join` / `replace` / `to_upper` / `to_lower` / `trim` — pure transforms returning new `Text` values (C4).
- `guarantee_matrix::MATRIX` — all ops encoded as `Exact`/effect-free tags, asserted in tests.

## Guarantee posture

Every op is `Exact` — text has no accuracy/precision semantics; all honesty is in the fallibility
column: `Result` everywhere, never a sentinel. Source is ground truth.

## Design references

- RFC-0016 §4.1 (C1–C6), §4.4; RFC-0013 §4.1 (diagnostic record I1); ADR-003; RFC-0012.
- Tasks: M-524 (#165).
- Spec: `docs/spec/stdlib/text.md`.

## Role in the workspace

Ring 2 / Tier B UTF-8 string surface; adds no trusted code and no `unsafe` (KC-3). See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-text).
