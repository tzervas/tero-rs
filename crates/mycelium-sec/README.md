# mycelium-sec

> `myc-sec` — security checks as tooling (M-367): the `/security-review` posture as a suite tool, implementing the Mycelium-specific `wild`-block audit and orchestrating the existing secrets/supply-chain gates.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-sec` implements security checks as a first-class tooling step. The v0 library core is the
`wild`-block audit — the Mycelium-specific check no off-the-shelf scanner provides. `wild` is the
language's only unsafe escape hatch (LR-9/S6; DN-02 §5 — denied by default, lexically marked); the
audit inventories every `wild` block so the unsafe surface is known (never ambient) and requires
each to carry an ADR-014 `// SAFETY:` justification. An unjustified `wild` is an explicit finding
(G2). The audit surfaces the author's `// SAFETY:` claim; it does not adjudicate soundness
(VR-5 — report the claim, never fabricate a verdict). The secrets and supply-chain families
orchestrate the existing `scripts/checks/{secrets,deny}.sh` gates; a missing scanner is reduced
coverage, reported as such, never folded into a clean bill.

## Key items

- `audit_wild` — inventories every `wild` block in a source tree; produces a `WildAudit`.
- `WildAudit` — the full inventory (`Vec<WildBlock>`) plus the findings (one per unjustified block).
- `WildBlock` — one `wild` block: file, line, justified flag, and opener text.
- `Finding` — a security finding with family, rule code, severity, location, and why (G2).
- `Severity` — fixed declared map: `Info` / `Low` / `Medium` / `High` / `Critical`.
- `myc-sec` binary — orchestrates the wild audit, secrets scan, and supply-chain checks.

## Design references

- M-367
- LR-9, S6
- DN-02 §5
- ADR-014
- G2, VR-5, KC-3

## Role in the workspace

Depends on `mycelium-cli-common` only at the library level (a text-scan recogniser, `std`-only, no new dep); the binary orchestrates `scripts/checks/` via `std::process`. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-sec).
