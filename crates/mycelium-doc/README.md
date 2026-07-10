# mycelium-doc

> `myc-doc` — the M-363 documentation build pipeline: a content-addressed doc-IR projected from the corpus, code, and nodule-header metadata, with HTML/Typst/JSON renderers and an eight-check §4.1 quality-bar lint.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-doc` implements the M-363 documentation build pipeline. The architecture is one
content-addressed doc-IR, many renderers: the cited corpus (RFCs/ADRs/notes/specs), the code and
M-359 nodule-header metadata, and JSON schemas are all projected into one navigable `DocModel`, and
HTML, Typst (→ PDF), and machine JSON are views of that one model — never parallel truths
(ADR-003/G11). Generation is projection, not authorship: an item that cannot be grounded is flagged
"undocumented," never invented (the prose analogue of G2). On top of the model runs the §4.1
quality-bar lint (`doc_lint`): eight explicit pass/fail checks including single-template
conformance, no-dead-xref, dual-projection parity, no-hallucinated-prose, and checked examples
(every inline example must actually type-check via the trusted L1 checker).

## Key items

- `build` / `BuildInput` — builds the `DocModel` from corpus, code, and metadata inputs.
- `emit_all` — emits all renderer outputs (HTML, Typst, JSON) from a built model.
- `DocModel` / `Node` / `Payload` / `Provenance` — the doc-IR types.
- `lint` / `DocLintReport` / `CHECK_NAMES` — the §4.1 quality-bar lint and its eight named checks.

## Design references

- M-363
- ADR-003
- G2, G11, KC-3

## Role in the workspace

Depends on `mycelium-core`, `mycelium-proj`, and `mycelium-l1`; provides the doc-IR and quality lint consumed by `mycelium-lint` (`DOC_QUALITY_CHECKS`) and the `myc-doc` binary. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-doc).
