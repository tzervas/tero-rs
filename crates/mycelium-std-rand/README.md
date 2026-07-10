# mycelium-std-rand

> `std.rand` — random number generation with reified, named nondeterminism (declared entropy effects).

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.rand` is the random-number surface for the Mycelium standard library, held to the RFC-0016 §4.1
contract. Its honesty crux is C6 in its sharpest form: nondeterminism is reified and named (RT3). A
generator that consumes real entropy carries a declared `EntropyEffect` on its signature, so a
deterministic-fragment program cannot pull randomness silently. The seeded surface (`Rng`) is a pure
immutable value — same seed, same sequence — with no ambient global RNG. The platform-entropy floor
is injectable and deferred to `mycelium-std-sys` (M-541).

## Key items

- `Rng` — seeded, immutable generator value `{algo, state}`; all draws return `(output, Rng')`.
- `EntropyRng<S>` — entropy-backed generator; construction and draws declare `EntropyEffect` (C6/RT3).
- `EntropySource` — injectable trait bridging the pure crate to the OS entropy floor.
- `EntropyEffect` — zero-cost declared-effect token; proves entropy was consumed, not fabricated.
- `StubEntropy` — deterministic test stub; exercises the full entropy path without OS calls.
- `seed` / `next_u64` / `split` — pure, `Exact`, reproducible seeded operations.
- `uniform_int` / `uniform_u64` / `bernoulli` / `choice` / `shuffle` — sampling ops (`Declared`).
- `normal` / `exponential` — continuous samplers (`Empirical`; Box-Muller / inverse-CDF).
- `seed_from_entropy` — draws entropy once (declared), returns a pure `Rng`.
- `GUARANTEE_MATRIX` — RFC-0016 §4.5 per-op tags encoded as data, asserted in tests.

## Guarantee posture

Per-op guarantee tags are encoded in `GUARANTEE_MATRIX` and asserted in tests (never prose-only).
Seeded-mechanism ops are `Exact`; sampling-correctness claims are `Declared`/`Empirical`; no op
reaches `Proven` without a checked theorem (VR-5). Source is ground truth.

## Design references

- RFC-0016 §4.1 (C1–C6); RFC-0014 (RT3 declared effects); ADR-003 (value-semantics).
- Tasks: M-531 (#171).
- Spec: `docs/spec/stdlib/rand.md`.

## Role in the workspace

Ring 2 / Tier B pure RNG surface; platform entropy is injected from `mycelium-std-sys`. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-rand).
