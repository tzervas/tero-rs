# mycelium-build

> Stable-component build layer (RFC-0004 §4; ADR-003/009): classifies AOT-eligible stable components vs interpreted/JIT and emits content-addressed build certificates.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-build` makes the RFC-0004 §4 AOT-eligibility gate executable and inspectable. A
definition is AOT-eligible only when it is content-addressed and hash-frozen (ADR-003), its spec
is ratified, and all verification obligations (swap certificates, bound checks, reference
equivalence) are discharged. Promotion to AOT is an explicit, deliberate act — the checks must
pass but marking stable is never automatic. A tampered `BuildCertificate` claiming AOT without
discharged obligations is rejected on deserialise, so the route cannot be forged from untrusted
input (G2).

## Key items

- `check_eligibility` — runs the automatic §4 checks; returns specific blocking reasons, never a silent "not eligible".
- `decide` — routes a component (AOT only for an eligible, explicitly promoted one) and emits a `BuildCertificate`.
- `BuildCertificate` — the content-addressed, inspectable record of the routing decision (ADR-003).
- `Obligations` — the three §4(3) verification obligations recorded as checked facts.
- `BuildCache` / `CacheOutcome` — build artefact cache layer.
- `Target` / `VariantTable` — target architecture and profile abstractions.

## Design references

- M-311, M-312
- RFC-0004 §4
- ADR-003, ADR-009
- KC-3

## Role in the workspace

Provides the AOT-eligibility gate and content-addressed build certificates; depends on `mycelium-core` only (outside the trusted kernel per KC-3). See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-build).
