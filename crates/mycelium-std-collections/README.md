# mycelium-std-collections

> Immutable persistent collections — Seq, Map, and Set — for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.collections` provides three immutable, persistent data structures: `Seq<E>` (ordered
sequence), `Map<K,V>` (insertion-ordered key-value map), and `Set<E>` (insertion-ordered
element set). All collections are value-semantic and content-addressed. Out-of-bounds access
returns an explicit `Err(CollErr::IndexOOB)` — never a panic or silent default (C1/G2).

## Key items

- `Seq<E>` — immutable ordered sequence with `get`, `push`, `pop`, `len`, `foldable`.
- `Map<K,V>` — insertion-ordered key-value map with `get`, `insert`, `remove`, `contains_key`.
- `Set<E>` — insertion-ordered element set with `contains`, `insert`, `remove`.
- `CollErr::IndexOOB` — explicit out-of-bounds error returned by indexed access.

## Design references

- RFC-0016 (core + standard library contract, C1–C6).
- Spec: `docs/spec/stdlib/collections.md` (M-511).

## Role in the workspace

General-purpose immutable collections used across Ring-1/Ring-2 stdlib crates. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-collections).
