# Changelog

## [Unreleased]

### Added

- Optional **`memory` Cargo feature**: MCP tools `memory_store`, `memory_retrieve`, `memory_consolidate` backed by memory-gate-rs (`join/tero-memory-feature`). Scopes `memory-read` / `memory-write`; runtime gated by `TERO_MEMORY_ENABLED` + `TERO_MEMORY_DB`.

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
