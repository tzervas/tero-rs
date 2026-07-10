# Changelog

All notable changes to tero-rs (the Rust tero kernel + fronts) will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-07-10

### Changed (rebrand — public crate rename)
- Renamed the public crate `mycelium-tero` → **`tero`** (and the directory `crates/mycelium-tero`
  → `crates/tero`; all `use mycelium_tero::…` paths → `tero::…`; the `identify`/`crate_summary`
  server-identity strings and their test assertions; all docs/comment prose). The binary names are
  unchanged (`tero-mcp`/`tero-http`/`tero-index`/`tero-eval` — already tero-prefixed) and the MCP
  protocol server name stays `tero-mcp`. WHY: the extracted project is "Tero"; carrying the
  `mycelium-` prefix on the crate was a leftover from the monorepo. WHY a patch bump (0.1.1 →
  0.1.2), not 0.1.1-stands: renaming the public crate name is an observable change to consumers
  (downstream `tero-mcp` now depends on `tero`, not `mycelium-tero`), so it is reflected in semver;
  but the shipped binary *behavior* is identical to 0.1.1.
- Added an "About the name" etymology note to the README: **Tero** honors Atsushi Tero, whose
  slime-mold (*Physarum*) experiments grew near-optimal transport networks (the Tokyo-rail-map
  result) — a fitting nod because Tero's indexing/search optimizes the route from a query to the
  relevant citations. (The crate `description` already carried a homage line; the README now states
  it plainly.)
- Removed a stray tracked `Cargo.toml.bak` backup (cruft).
- Added a real-committed-index end-to-end test to the `tero` crate: every citation's `file:line`
  resolves on disk, with engine/HTTP/MCP parity on real data (`src/tests/front_live_corpus.rs`).

### Removed (clean extraction — vendored language-crate residue)
- `tero-rs` was extracted from the Mycelium monorepo (where `tero`/`mycelium-tero` was built as a
  context/indexing dev tool) and carried the **entire** `mycelium-*` language workspace along as
  extraction residue. This release strips the residue: **removed 40 vendored `mycelium-*` crates**
  that are **not** in `tero`'s dependency closure (the `mycelium-std-*` family, `mycelium-cli`/
  `-lsp`/`-fmt`/`-lint`/`-check`/`-sec`/`-spore`/`-bench`/`-build`/`-diag`/`-transpile`/
  `-vsa-decode`, etc.). The workspace went from 57 crates to **17**. This is verified-safe: the
  removed set is manifest-closed — no remaining crate references any removed crate (checked via a
  path-dependency closure analysis + a green `cargo build/test -p tero` after removal). A welcome
  side effect: the workspace-wide clippy noise from the `mycelium-std-cmp` stub is gone with it.
- **Reverted** two out-of-scope edits that had been made to language crates (`mycelium-core`
  `chunks_exact`→`as_chunks`; `mycelium-l1` `map_or`→`unwrap_or`) and the tests added around them —
  the `mycelium-*` language crates are a **separate project**, vendored read-only here, and are not
  `tero-rs`'s to modify. Gates are now scoped to `tero` (`-p tero`, `clippy --no-deps`), not the
  whole workspace.

### FLAG (for maintainer decision — vendored genuine dependencies)
- `tero` still **vendors 16 `mycelium-*` crates** that are its genuine (transitive) dependencies:
  direct path-deps `mycelium-doc` + `mycelium-vsa`, pulling in `mycelium-core`, `-l1`, `-proj`,
  `-cert`, `-interp`, `-select`, `-numerics`, `-sched`, `-dense`, `-stack`, `-workstack`, plus
  `-mlir`/`-mir-passes`/`-rt-abi` (kept only because `mycelium-l1` has a **dev-dependency** on
  `mycelium-mlir` — `tero` does not build them, but they must remain workspace members for
  `mycelium-l1`'s manifest to resolve). These are vendored copies of a separately-maintained
  project; a future change should decide whether `tero` depends on the **published** mycelium
  crates instead of vendoring them (and whether `mycelium-l1`'s FFI-JIT `mycelium-mlir` dev-dep can
  be broken to drop the last 3). Not actioned here — flagged.

### Verified
- `cargo build --release -p tero` green; `cargo test -p tero --release` **113 passed**; `cargo clippy
  -p tero --all-targets --no-deps -- -D warnings` clean; the four tero binaries build. `cargo build
  --release --workspace` (now 17 crates) green. 1.0.0 remains gated on the broader readiness criteria
  (ROADMAP): hardened fronts, cabal positive assess, and a decision on the vendored-vs-published
  dependency question above.

## [0.1.1] - 2026-07-10

### Verified stable + published
- Real gate run (not claimed): `cargo build --release -p mycelium-tero` green;
  `cargo test -p mycelium-tero --release` 112/112 passed; `cargo clippy -p mycelium-tero
  --all-targets -- -D warnings` clean. `cargo build --release --workspace` also green (all ~55
  crates compile). `cargo clippy --workspace --all-targets -- -D warnings` (the full-workspace,
  not just tero-facing, gate) still fails on one unrelated finding in `mycelium-std-cmp` (a 0.0.0
  internal stub crate NOT in `mycelium-tero`'s dependency graph): a nightly-only
  `unstable_name_collisions` lint on a `widen()` method name that may collide with a future std
  API. Not a defect in the shipped surface, but it is why this is 0.1.1, not 1.0.0 (ROADMAP's own
  1.0 criteria require green checks ACROSS the workspace).
- Bumped `mycelium-tero` 0.1.0 -> 0.1.1 (patch: fixes/hygiene only, no API/behavior change to the
  shipped bins). Brings tero-rs to parity with the python siblings (tero-mcp/cabal already 0.1.1).
- Published: annotated tag `v0.1.1` on `origin`, `gh release create v0.1.1` with the built
  `tero-mcp` release binary attached as an asset, plus a GHCR container image
  `ghcr.io/tzervas/tero-rs` tagged `0.1.1`/`0.1`/`0`/`latest`. See the release page + package
  registry for the verification evidence (git ls-remote / gh release view / GHCR API), not this
  changelog entry alone (VR-5: this paragraph is itself `Declared` until cross-checked against
  those primary sources at publish time).

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

[0.1.1]: https://github.com/tzervas/tero-rs/releases/tag/v0.1.1
[0.1.0]: https://github.com/tzervas/tero-rs/releases/tag/v0.1.0

---

*Initial changelog entry for 1.0 wave tranche. Append future entries only.*
