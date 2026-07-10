# tero-rs — Tero Index (Layer 1)

> **Honesty:** Empirical/Declared — lite heading/line heuristic over markdown in tero-rs via tero-mcp/scripts/generate_lite_index.py; source files are ground truth. Generated 2026-07-10.
> Use this index to find where to Read, not as authoritative ground truth.

- **Items:** 24
- **Flagged:** 0
- **item_tag:** `Empirical/Declared`
- **Machine index:** [`index.json`](./index.json)
- **Manifest:** [`MANIFEST.toml`](./MANIFEST.toml)

## doc (17 entries)

| Anchor | Kind | Id | Title | File:Line | Status | Summary |
|---|---|---|---|---|---|---|
| `agents` | section | — | Agent notes — tero-rs | `AGENTS.md:1` | — | Tero-first (mandatory): Start every analysis, edit, or assessment with tero (MCP tero or /root/git/scripts/tero.sh tero-rs <tool> ...). Cite every answer (cita… |
| `agents--key-references-from-tero-wave` | section | — | Key references (from tero + wave) | `AGENTS.md:7` | — | - tooling-1.0-readiness-2026-07-10 wave (P1 toolchain): tero-rs, tero-mcp, cabal-devmelopner. |
| `agents--usage-for-agents` | section | — | Usage for agents | `AGENTS.md:14` | — | 1. searchtool for "tero" schema if MCP not pre-listed. |
| `agents--branch-hygiene-land` | section | — | Branch / hygiene / land | `AGENTS.md:23` | — | - Working branch only: e.g. feature/1.0-readiness (evolved from chore/semver-baseline-v0.1.0). |
| `agents--semver-release-path` | section | — | Semver + release path | `AGENTS.md:32` | — | See docs/ROADMAP.md and CHANGELOG.md for justification of current 0.1.0 path vs full 1.0 (supporting extracted crate; stable once fronts hardened + tests green… |
| `agents--self-improvement` | section | — | Self-improvement | `AGENTS.md:35` | — | Use cabal-devmelopner from inside: |
| `readme` | other | — | tero-rs | `README.md:1` | Extracted toolchain core (2026-07-10) | Status: Extracted toolchain core (2026-07-10) |
| `readme--quick-start-hygiene-tero` | section | — | Quick start (hygiene + tero) | `README.md:17` | — | cd tero-rs |
| `readme--after-docs-root-git-scripts-update-tero.sh-tero-rs` | other | — | After docs: /root/git/scripts/update-tero.sh tero-rs | `README.md:21` | — | Use tero MCP or /root/git/scripts/tero.sh tero-rs ... (after index) for all design/roadmap/historic context. See AGENTS.md. |
| `readme--query-root-git-scripts-tero.sh-tero-rs-textsearch-...` | other | — | Query: /root/git/scripts/tero.sh tero-rs text_search "..." | `README.md:22` | — | Use tero MCP or /root/git/scripts/tero.sh tero-rs ... (after index) for all design/roadmap/historic context. See AGENTS.md. |
| `readme--tero-first` | section | — | Tero-first | `README.md:25` | — | Use tero MCP or /root/git/scripts/tero.sh tero-rs ... (after index) for all design/roadmap/historic context. See AGENTS.md. |
| `readme--1.0-readiness` | section | — | 1.0 Readiness | `README.md:29` | — | Part of tooling-1.0-readiness-2026-07-10 wave (priority 1: Toolchain Core). |
| `roadmap` | note | — | tero-rs — Product Roadmap | `docs/ROADMAP.md:1` | Living (2026-07-10, tooling-1.0-readiness wave) | Status: Living (2026-07-10, tooling-1.0-readiness wave) |
| `roadmap--1.0-target-criteria-from-tooling-1.0-readiness-2026-07-10-wave` | section | — | 1.0 Target Criteria (from tooling-1.0-readiness-2026-07-10 wave) | `docs/ROADMAP.md:8` | — | - Stable: check.sh green, cargo test/clippy clean on tero fronts, no critical unwraps in prod paths. |
| `roadmap--current-state-tero-cited-surveys` | section | — | Current State (Tero-cited + surveys) | `docs/ROADMAP.md:15` | — | - On feature/1.0-readiness (evolved chore/semver-baseline-v0.1.0). |
| `roadmap--waves-steps-for-toolchain-p1` | section | — | Waves / Steps for Toolchain P1 | `docs/ROADMAP.md:22` | — | - Hygiene + branch guard (done this tranche). |
| `roadmap--relation-to-siblings` | section | — | Relation to siblings | `docs/ROADMAP.md:30` | — | - tero-mcp: Python lite + packaging over the Rust tero-mcp bin. See its ROADMAP for MCP surface stability. |

## changelog (7 entries)

| Anchor | Kind | Id | Title | File:Line | Status | Summary |
|---|---|---|---|---|---|---|
| `changelog` | entry | — | Changelog | `CHANGELOG.md:1` | — | All notable changes to tero-rs (the Rust tero kernel + fronts) will be documented here. |
| `changelog--unreleased` | section | — | [Unreleased] | `CHANGELOG.md:8` | — | - scripts/check.sh (targeted cargo fmt/clippy/check/test on mycelium-tero; --fix/--quick support). WHAT: hygiene gate matching sibling projects. WHY: wave requ… |
| `changelog--0.1.0-2026-07-10` | section | — | [0.1.0] - 2026-07-10 | `CHANGELOG.md:10` | — | - scripts/check.sh (targeted cargo fmt/clippy/check/test on mycelium-tero; --fix/--quick support). WHAT: hygiene gate matching sibling projects. WHY: wave requ… |
| `changelog--added-for-1.0-readiness` | section | — | Added (for 1.0 readiness) | `CHANGELOG.md:12` | — | - scripts/check.sh (targeted cargo fmt/clippy/check/test on mycelium-tero; --fix/--quick support). WHAT: hygiene gate matching sibling projects. WHY: wave requ… |
| `changelog--changed` | section | — | Changed | `CHANGELOG.md:18` | — | - Ran ./scripts/check.sh --fix: applied fmt (line wraps, import order in mycelium-tero/src/{bin/tero-mcp.rs, front/mcp.rs, lib.rs}). WHAT: clean hygiene run. W… |
| `changelog--cited` | section | — | Cited | `CHANGELOG.md:23` | — | - Tero-first actions: /root/git/scripts/tero.sh tero-mcp/cabal-devmelopner/dev-docs identify + textsearch for "tero semver 1.0 readiness tooling cabal", "chore… |
| `changelog--verification` | section | — | Verification | `CHANGELOG.md:29` | — | - Hygiene: check.sh green (fmt fixed, tests ok). |

