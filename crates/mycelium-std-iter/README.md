# mycelium-std-iter

> Iterator, fold, and transducer combinators for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.iter` is the Ring-2/Tier-B iterator surface. It provides `Foldable<E>` (an eager,
value-semantic collection backed by `Vec<E>`) and `Lazy<E>` (a deferred sequence), together
with the full combinator vocabulary: map, filter, fold, scan, zip, take, skip, step_by,
enumerate, flat_map, reduce, count, chain, any/all-with-witness, find, position, and
`transduce` (composable `Transducer` pipeline). `zip_exact` is fallible (`Err(ZipLengthMismatch)`)
and `step_by` rejects a zero step (`Err(ZeroStep)`) — no silent truncation or panic (C1/G2).
`lazy_unfold` is tagged `Declared` (potentially non-terminating); all other ops are total
under RFC-0007 (kernel `for`/fold totality-by-construction).

## Key items

- `Foldable<E>` — eager immutable sequence; primary combinators operate here.
- `Lazy<E>` — deferred sequence for bounded lazy evaluation.
- `Transducer` — composable pipeline stage; used with `transduce`.
- `ZipOutcome` — result of a `zip` (paired values or length mismatch info).
- `AnyAllWitness` — witness value returned by `any_with_witness` / `all_with_witness`.
- `map`, `filter`, `fold`, `scan`, `zip`, `zip_exact`, `take`, `skip` — core combinators.
- `step_by`, `enumerate`, `flat_map`, `reduce`, `count`, `chain` — additional combinators.
- `any_with_witness`, `all_with_witness`, `find`, `position` — search with witness.
- `transduce`, `lazy_take`, `lazy_unfold` — transducer / lazy evaluation ops.
- `ZeroStep` — error returned when `step_by(0)` is called.
- `ZipLengthMismatch` — error returned by `zip_exact` on unequal-length inputs.

## Design references

- RFC-0016 (core + standard library contract, C1–C6); RFC-0007 (kernel `for`/fold totality-by-construction).
- Spec: `docs/spec/stdlib/iter.md` (M-526).

## Role in the workspace

Iteration and pipeline combinators for Ring-2 std crates and user code. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-iter).
