# Changelog

## [Unreleased]

## [0.2.1] — 2026-07-21

### Added

- Optional **`memory` Cargo feature**: MCP tools `memory_store`, `memory_retrieve`, `memory_consolidate` backed by memory-gate-rs (`join/tero-memory-feature`). Scopes `memory-read` / `memory-write`; runtime gated by `TERO_MEMORY_ENABLED` + `TERO_MEMORY_DB`. Documented in README under "Optional memory tools".
- README **"What ships vs. what's gated"** section: states plainly that Layer-1 is what's serving, that Layer-2 (VSA) is implemented but gate-CLOSED (`layer2_enabled = false`, see `eval/VERDICT.md`), and that the `memory` feature is a separate, orthogonal surface — not Layer-2, not RAG.

### Changed

- CI: linux x64 jobs route to the self-hosted podman fleet runner; `fleet-ci.yml` / `fleet-security.yml` gained `workflow_dispatch` and pinned `ubuntu-latest` for the meta jobs; push/PR triggers enabled on the fleet workflows.

### Fixed

- Repository-root **MIT `LICENSE`** file added (`Cargo.toml` already declared `license = "MIT"`; the file itself was missing).

### Deferred

- Layer-2 VSA retrieval stays gate-CLOSED — no measured win over Layer-1 yet (correctness@1 0.375 vs 0.625, latency ~26x). See `eval/VERDICT.md` Run 1.
- History sanitize (retiring `v0.1.0`–`v0.1.3` tags/releases) not done this patch — still tracked in `docs/HISTORY_SANITIZE.md`.

## [0.2.0] — 2026-07-16 (standalone cut)

### Removed

- **All vendored `mycelium-*` crates** (language project residue). tero no longer path-depends on mycelium-doc, mycelium-vsa, mycelium-core, or any mycelium language crate.
- Multi-crate Mycelium-style workspace layout under `crates/mycelium-*`.

### Changed

- **Single-package repository layout**: package `tero` at repo root (`src/`, bins in `src/bin/`).
- Markdown corpus ingest inlined as `src/md/` (CommonMark-subset parser + doc IR).
- Layer-2 MAP-I + cleanup memory inlined as `src/vsa2/algebra.rs` (no external VSA crate).
- `scripts/check.sh` targets the single package only.
- Version **0.2.0** marks the standalone product line (breaks any consumer expecting path-deps on mycelium).

### Added

- `docs/HISTORY_SANITIZE.md` — plan to delete old releases/tags and optional history rewrite under branch protection.

### Migration for consumers

- Depend on git tag / path of this repo package `tero` only.
- Rebuild against 0.2.0; do not expect `mycelium_*` types in the public graph.

## [0.1.x] — retired

Pre-0.2.0 tags (`v0.1.0`–`v0.1.3`) shipped mycelium extraction residue. They should be deleted from GitHub Releases/tags after 0.2.0 lands (see HISTORY_SANITIZE.md).
