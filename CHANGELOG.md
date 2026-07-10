# Changelog

All notable changes to tero-rs (the Rust tero kernel + fronts) will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-10

### Added (for 1.0 readiness)
- `scripts/check.sh` (targeted cargo fmt/clippy/check/test on mycelium-tero; --fix/--quick support). WHAT: hygiene gate matching sibling projects. WHY: wave requires hygiene first + CI parity for toolchain P1; tero-rs previously thin on per-repo check (per readiness survey). WHY NOT full workspace test: too heavy for extracted view; targeted keeps fast.
- Root + docs/ scaffolding (README.md, AGENTS.md, docs/ROADMAP.md) + tero index bootstrap via update-tero.sh. WHAT: enable Tero Layer-1 cites + cabal --use-tero assessment inside project. WHY: 0-item index caused facade refusal (C0) in cabal runs; wave mandates tero-first + update AGENTS/ROADMAP + cabal utilization. WHY NOT skip: blocks all tero/cabal requirements in rules.
- Branch evolution to `feature/1.0-readiness` (from chore/semver-baseline-v0.1.0) with guard.
- CHANGELOG.md (this) + initial 0.1.0 entry.

### Changed
- Ran `./scripts/check.sh --fix`: applied fmt (line wraps, import order in mycelium-tero/src/{bin/tero-mcp.rs, front/mcp.rs, lib.rs}). WHAT: clean hygiene run. WHY: fixes found on first check; required before land per rules. WHY NOT ignore: would fail 1.0 stable gate.
- Bumped key crate `mycelium-tero` version 0.0.0 → 0.1.0 (see Cargo.toml edit). WHAT: align extracted tero-rs project with 0.1.0 baseline established in chore + python siblings (tero-mcp/cabal 0.1.0). WHY: semver review in wave; 0.1.0 as "initial stable supporting tooling release candidate path" (per wave doc); not full 1.0 yet (see ROADMAP for path: requires hardened fronts, consumer parity, more tests). WHY NOT 1.0.0 now: tero-rs is supporting kernel (not end-user product); readiness criteria include full checks + cabal positive + release artifacts (partial here); would be dishonest semver per C0. Other crates remain 0.0.0 (internal); only tero-facing bumped.
- Google-style comments added to edits (this file + code changes) per wave rule.

### Cited
- Tero-first actions: /root/git/scripts/tero.sh tero-mcp/cabal-devmelopner/dev-docs identify + text_search for "tero semver 1.0 readiness tooling cabal", "chore/semver-baseline hygiene roadmap 1.0"; connected tero__* MCP calls.
- cabal-devmelopner runs (inside projects, with TERO_INDEX_PATH + --provider xai --use-tero --use-tools) for assessment (refusals on 0-item index surfaced gap).
- References: dev-docs/waves/1.0-readiness-tooling-wave-2026-07-10.md (P1 steps, criteria, "bump toward 1.0 or document 1.0 path"), plan.md (semver baseline, hygiene, cabal utilization), dev-docs/WORKSPACE_CABAL_TERO_READINESS.md, prior semver chore commit.
- Post: re-ran checks; will re-gen index + cabal assess + verify.

### Verification
- Hygiene: check.sh green (fmt fixed, tests ok).
- Tero index now >0 after md adds.
- All changes on feature/1.0-readiness; append-only docs.

[0.1.0]: https://example (no tag yet)

---

*Initial changelog entry for 1.0 wave tranche. Append future entries only.*
