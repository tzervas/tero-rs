# History & release sanitize plan (post–mycelium purge)

**Status:** Plan only — requires maintainer approval under branch protection.  
**Goal:** Public `tero-rs` history and releases must not carry mycelium language crates as if they belonged here.

## Why

1. `tero-rs` was extracted from the Mycelium monorepo and accidentally retained vendored `mycelium-*` crates and history.
2. Mycelium is a **separate language project**. tero is an index/query engine.
3. Old tags/releases (`v0.1.0`–`v0.1.3`) package the wrong dependency story for consumers.

## Recommended process (granular, protection-safe)

### Phase A — Land clean tree (this PR)

- Single-package layout, zero `mycelium-*` path deps.
- Green `./scripts/check.sh`.
- Merge to `main` via normal PR (no force-push yet).

### Phase B — Retire bad releases (human + admin)

1. List: `gh release list`, `gh api repos/tzervas/tero-rs/tags`
2. For each mycelium-era tag/release:
   - `gh release delete <tag> --yes` (if published)
   - `git push origin :refs/tags/<tag>` (delete remote tag)
3. Delete related GHCR images if any: `ghcr.io/tzervas/tero-rs:*` for those tags
4. Document deleted tags in CHANGELOG “Retired releases”

### Phase C — History rewrite (optional but preferred)

Branch protection blocks force-push to `main`. Options:

| Option | When |
|--------|------|
| **C1 — Soft** | Leave old history; new `0.2.0` is the first “honest” release. History still contains deleted crates (git objects). |
| **C2 — Orphan root** | Create orphan branch with current tree only; open PR is impossible → temporary disable protection or admin force-push with announcement. |
| **C3 — filter-repo** | `git filter-repo --path crates/mycelium- --invert-paths` then force-push; still needs protection exception. |

**Recommendation:** A + B immediately; C2 after team ack (cleanest for “wholly purged from history”).

### Phase D — Retag

- Tag `v0.2.0` on sanitized `main` only after A (+ preferred C).
- Publish GitHub Release + GHCR only from that tag.

## Do not

- Force-push without announcing consumers (tero-mcp, cabal).
- Keep path deps “just for history.”
- Mix mycelium monorepo CI into this repo.
