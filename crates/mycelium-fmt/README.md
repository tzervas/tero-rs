# mycelium-fmt

> `mycfmt` — the canonical formatter (M-364): an identity-preserving projection over `.myc` sources that never changes a definition's content-addressed identity (RFC-0001 §4.6/§4.8; ADR-003).

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-fmt` implements the `mycfmt` formatter. Formatting is a projection: it rewrites a `.myc`
source into one canonical textual normal form and never changes a definition's content-addressed
identity. Three invariants are enforced — identity-preservation (C1, checked at runtime: a
mismatch is a refusal, never an emitted rewrite), idempotence (C2, tested), and header-preservation
(C3: the `// nodule:` marker and `// @key:` structured header are re-emitted canonically). Interior
comments are preserved by interleaving from the lexer comment table (M-690, Stage 2). Unparsable
input, a malformed header, a construct outside the round-trip-safe v0 scope, or an unplaceable
comment is an explicit error; `mycfmt` never writes a partial or garbled rewrite (G2). The
`[toolchain].format` hard pin (M-359) is enforced: a version mismatch is refused.

## Key items

- `format_source` — formats a source string in the compact canonical form; returns `Formatted` or `FmtError`.
- `format_source_readable` — the human-readable multi-line style (M-974/DN-82): long argument / field / variant / arm segments break across lines, short ones stay inline. Presentation-only and functionally inert (same surface AST — C1/C2).
- `Style` — `Compact` (default) or `Readable`; both are shared by `format_source_styled`.
- `Formatted` — the successful result: output text, a changed flag, and any notes.
- `FmtError` — explicit error variants (parse failure, header error, out-of-scope construct).
- `MYCFMT_VERSION` — the formatter spelling/version this build implements (used for the hard pin check).
- `mycfmt` binary — CLI entrypoint; `--check` / `--write`, `--flatten` (single-line stream form) and `--readable` (human multi-line; mutually exclusive with `--flatten`); reads stdin or a file path.

## Design references

- M-364, M-142, M-690, M-819, M-974
- RFC-0001 §4.6/§4.8
- ADR-003
- DN-06, DN-57, DN-82
- G2, VR-5, KC-3

## Role in the workspace

Depends on `mycelium-l1`, `mycelium-proj`, and `mycelium-cli-common`; provides the canonical formatter above the kernel. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-fmt).
