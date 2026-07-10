# Agent notes — tero-rs

**Tero-first (mandatory):** Start every analysis, edit, or assessment with tero (MCP `tero__*` or `/root/git/scripts/tero.sh tero-rs <tool> ...`). Cite every answer (citations + EXPLAIN on demand).

This repo is the Rust implementation of the tero query engine + MCP/HTTP fronts (Layer-1 primary; Layer-2 VSA gated). See crates/mycelium-tero/src/ and the bin/ frontends.

## Key references (from tero + wave)
- tooling-1.0-readiness-2026-07-10 wave (P1 toolchain): tero-rs, tero-mcp, cabal-devmelopner.
- plan.md (gaps, semver baseline, hygiene).
- dev-docs/WORKSPACE_CABAL_TERO_READINESS.md
- crates/mycelium-tero/Cargo.toml and src/ (M-1016/1017/1018 notes in comments).
- tero-mcp docs/ (the Python side + generator).

## Usage for agents
1. `search_tool` for "tero" schema if MCP not pre-listed.
2. tero__identify or script identify.
3. text_search for "semver", "hygiene", "roadmap", "1.0", "mcp", "front".
4. query_by_id for known (e.g. DN-87 if surfaced).
5. Then local fs only for verification.

Never silent (C0): refusals from tero when no citable rows are explicit.

## Branch / hygiene / land
- Working branch only: e.g. feature/1.0-readiness (evolved from chore/semver-baseline-v0.1.0).
- Always: `./scripts/check.sh` (or --fix) before commit/land.
- cargo test/clippy targeted on mycelium-tero + fronts.
- Append-only: docs/CHANGELOG.md, AGENTS.md, ROADMAP, this wave refs.
- Google-style comments on all code/docs changes: WHAT / WHY / WHY NOT other options.
- PR: feature/... → dev (pr-review adapted), dev → main --no-ff.
- Tero re-index + cabal --use-tero assessment post-edit.

## Semver + release path
See docs/ROADMAP.md and CHANGELOG.md for justification of current 0.1.0 path vs full 1.0 (supporting extracted crate; stable once fronts hardened + tests green across consumers).

## Self-improvement
Use cabal-devmelopner from inside:
```
cd /root/git/workspace/tero-rs
TERO_INDEX_PATH=docs/tero-index/index.json uv run --project /root/git/workspace/cabal-devmelopner cabal-devmelopner "..." --provider xai --use-tero --use-tools
```

Cites for this file (initial): tooling wave doc sections on tero-rs steps, prior semver chore, tero identify calls.

*Append only.*

## Secrets, .env and git-secrets protection (2026-07-10, tooling 1.0 wave)
**WHAT**:
- .gitignore now contains standard block for `.env`, `.env.local`, `.env.*.local`, `*.env`, `*.key`, `secrets/` (templates explicitly allowed with `!`).
- git-secrets protection activated: `git secrets --install -f` (hooks in .git/hooks including pre-commit/prepare-commit-msg), `--register-aws`, custom adds for `XAI_API_KEY` + variants + `ANTHROPIC_API_KEY` + `OPENAI_API_KEY` + sk- patterns.
- `.gitallowed` created with safe exceptions for key *names* in docs/comments/examples (real secret *values* will still be caught).
- `git secrets --scan` now clean across tree.
**WHY**: cabal-devmelopner and agents actively consume `XAI_API_KEY` (and will for Claude/ADK post-stability). Leaking keys in git history is a critical supply chain / compliance risk. Complements security-mcp scans. Enforces 1.0 "hardened" criteria at the source (pre-commit + hygiene).
**WHY NOT**: Relying on .gitignore alone is insufficient (doesn't scan code/comments for accidental pastes of values); git-secrets chosen as lightweight, established (awslabs), no heavy deps. Allowed patterns only for identifiers (not values).
**After fresh clone / in new worktree (mandatory)**:
```
git secrets --install
git secrets --register-aws
git secrets --add 'XAI_API_KEY'
git secrets --add 'ANTHROPIC_API_KEY'
git secrets --add 'OPENAI_API_KEY'
# then verify
git secrets --scan
```
Enhance `scripts/check.sh` with `git secrets --scan || exit 1`.
Cites: tooling-wave-1.0-readiness doc (this task), user request post dev-support tranche, tero hygiene sections, cabal XAI provider code + AGENTS.
All changes: Google-style WHAT/WHY/WHY-NOT, append-only, branch/worktree guarded, tero-first audit.
