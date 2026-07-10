# mycelium-std-content

> Content-addressing and identity library for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.content` is the Ring-1/Tier-A content-addressing surface. It provides functions to
derive content hashes from values and definitions, compare and parse content references,
and resolve human-readable names to their canonical digest. The `NameRegistry` keeps the
name→hash mapping; `ContentRef` carries a typed reference (hash or name). No op invents
identity: every hash is derived from content, never assigned (ADR-003).

## Key items

- `hash_of_value` — derive a `ContentHash` from a `Value` (pure, `Exact`).
- `hash_of_def` — derive a `ContentHash` from a definition payload.
- `digest_eq` — constant-time equality check on two `ContentHash` values.
- `as_ref` — project a `ContentHash` to a `ContentRef`.
- `parse_ref` — parse a text digest or name into a `ContentRef`; `Err(MalformedDigest)` on failure.
- `resolve_name` — look up a name in a `NameRegistry`; returns `Option<ContentHash>`.
- `names_of` — return all names registered for a given `ContentHash`.
- `ContentRef`, `RefKind` — typed content reference and its kind (hash vs. name).
- `MalformedDigest` — parse error for invalid digest strings.
- `NameRegistry` — immutable name→hash registry.

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0001 (value model); ADR-003 (content-addressed identity).
- Spec: `docs/spec/stdlib/content.md` (M-523).

## Role in the workspace

Identity and name-resolution primitives for Ring-1/Ring-2 crates and the content-addressed value store. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-content).
