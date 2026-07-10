# mycelium-cert

> Swap certificates, the binary↔ternary certified swap (RFC-0002 §3/§4; M-120), and the single shared translation-validation checker (RFC-0002 §2; RFC-0004 §3; M-210).

**Tier:** compiler  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-cert` makes every swap inspectable and checkable. A swap produces a value in the target paradigm *and* an inspectable `SwapCertificate` — never silent (SC-3). The binary↔ternary swap over a legal `(n, m)` pair is bijective and `Exact` (`LosslessWithinRange`): it emits a `SwapCertificate::Bijective` referencing the once-per-`(n,m)` round-trip lemma (M-121) with concrete binding params. The inverse `dec` is partial: a ternary value outside the binary range is an explicit `SwapError::OutOfRange`, never a silent coercion.

The single translation-validation certificate checker (`check` module, M-210) validates bijective certificates by re-derivation equality, bounded certificates through the `mycelium-numerics` tier-i checker, and interp↔AOT observational equivalence — one checker, every instance. The serialized certificate form matches `docs/spec/schemas/swap-certificate.schema.json`. Zero `unsafe` — compiler-enforced.

## Key items

- `SwapCertificate` — the inspectable per-swap certificate (`Bijective` or `Bounded`).
- `SwapError` — explicit refusals: `WrongSource`, `IllegalPair`, `OutOfRange`, `NonFinite`, `NotAnF32`, `SubnormalUnsupported`.
- `check::check` / `check_core` — the unified translation-validation checker (M-210).
- `BinTernParams` — concrete parameters binding a bijection lemma to one use.
- `dense::dense_f32_to_bf16` — the `F32→BF16` certified swap (M-211).
- `dense_vsa::dense_to_vsa` / `vsa_to_dense` — Dense↔VSA swap (M-231).
- `mode::ModeGatedSwapEngine` / `GatedSwap` — RFC-0034/ADR-032 mode-gated swap engine.

## Guarantee posture

Bijective swaps produce `Exact` values (the round-trip lemma's side-conditions are checked). Bounded swaps produce values whose `Bound` is re-checked by `mycelium-numerics` against the claimed bound; a tighter-than-re-derived claim is refused. Posture is `Proven` where the theorem's side-conditions are verifiably checked, `Declared` for the lemma reference — source is ground truth.

## Design references

- RFC-0002, RFC-0004, RFC-0034, ADR-014, ADR-032, M-120, M-121, M-210, M-211, M-231

## Role in the workspace

Depends on `mycelium-core`, `mycelium-interp`, `mycelium-numerics`, and `mycelium-vsa`. Used by `mycelium-l1` and `mycelium-mlir`. Dev-dependencies include `mycelium-proj` and `mycelium-spore` for the RFC-0034 conformance suite. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-cert).
