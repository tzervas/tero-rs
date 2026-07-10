# mycelium-std-swap

> `std.swap` — certified, never-silent representation-change surface: every swap yields a value and an inspectable certificate, or an explicit error.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.swap` is the library form of RFC-0002: a swap is structurally impossible to obtain without its
`SwapCertificate`. Every swap yields a `Swapped{value, cert}` or an explicit `SwapError` — no
sentinel, no clamp, no silent coercion (C1/G2). The crate is a Ring 1 consumer over `mycelium-cert`'s
swap engines and adds no trusted code (KC-3/C5). The `explain` function projects any certificate to a
dual human/machine `ExplainRecord` (G11/C3). Tags are derived from the certificate's basis, never
asserted (VR-5/C2).

## Key items

- `Swapped` — the swap result: `{value, cert}`; `.explain()` projects to `ExplainRecord`.
- `bin_to_tern` / `tern_to_bin` — exact bijective binary↔ternary swaps (`Exact`, `Bijective` cert).
- `f32_to_bf16` — round-to-nearest Dense F32 → BF16 (`Proven` ε when ProvenThm conditions check).
- `dense_to_vsa` / `vsa_to_dense` — bipolar Dense ↔ VSA encoding (`Empirical` δ by default).
- `check_swap` — validate `b` refines `a` under `cert` via the M-210 shared checker (`Exact` verdict).
- `explain` — total projection of any `SwapCertificate` to an `ExplainRecord` (C3/G11).
- `legal_pair` — probe binary↔ternary legality without performing the swap.
- `GUARANTEE_MATRIX` — 7-row matrix encoded as data, asserted in tests (RFC-0016 §4.5).

## Guarantee posture

Per-op guarantee tags are encoded in `GUARANTEE_MATRIX` and asserted in tests. Bijective certs are
`Exact`; bounded certs are `Proven` or `Empirical` as derived from the cert basis (never asserted);
`Proven` is only reached when all cited theorem side-conditions are checked (VR-5). Source is ground
truth.

## Design references

- RFC-0002 (swap contract); RFC-0016 §4.1 (C1–C6); ADR-003; ADR-010 (BF16 rounding); RFC-0003/T0.2 (VSA capacity).
- Tasks: M-516.
- Spec: `docs/spec/stdlib/swap.md`.

## Role in the workspace

Ring 1 / Tier A certified representation-change surface; wraps `mycelium-cert` without enlarging the trusted base. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-swap).
