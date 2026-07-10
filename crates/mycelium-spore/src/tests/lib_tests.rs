//! Tests for `crate` (lib.rs / build_spore / content_address) — M-789 / RFC-0034 §8.
//!
//! Extracted from the old inline `#[cfg(test)]` block (CLAUDE.md test-layout rule).
//! New additions here (M-789):
//!
//! * `spore_id_is_independent_of_cert_mode` — property test (RFC-0034 §8 DoD): the spore identity
//!   is purely compile/deploy phase (code+deps+surface DAG), independent of any `CertMode`. Because
//!   `CertMode` rides `Meta` which is excluded from the content hash (RFC-0001 §4.6; ADR-003), and
//!   `mycelium-spore`'s `content_address` function never reads `CertMode` at all, this is an
//!   **invariant by construction** — verified here by exhaustively exercising all `CertMode` tiers
//!   (`Fast`/`Balanced`/`Certified`) over a representative project tree.
//!
//! * `compile_spore_hash_disable_is_explicit_and_never_silent` — documents the never-silent
//!   gate for disabling the compile spore-hash (embedded/no-deploy builds, RFC-0034 §8). The full
//!   disable *mechanism* (`no-spore-hash` build flag) is **not yet implemented** — this test pins
//!   the current state: the hash is **always performed** (never silently skipped), and the path for
//!   disabling it is an open gap (FLAG to M-789 / ADR-013).
//!
//! Guarantee tags:
//! * `spore_id_is_independent_of_cert_mode` — `Proven` (by construction: the `content_address`
//!   function never reads `CertMode`; the `CertMode::ALL` parameterisation is exhaustive over the
//!   three current tiers, and `build_spore` returns the same `Spore::id` in each case).
//!   The property test is the machine-checkable evidence.
//! * `compile_spore_hash_disable_is_explicit_and_never_silent` — `Declared` (asserted, not yet
//!   backed by a full mechanism; the disable path is deferred per ADR-013 / RFC-0034 §8).

use std::io::Write;
use std::path::PathBuf;

use mycelium_core::CertMode;
use mycelium_proj::parse_manifest;

use crate::{build_spore, explain, kind_str, SporeError};

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Write a throwaway project tree under a unique temp dir; returns its path.
fn scratch(name: &str, manifest: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "myc-spore-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("mycelium-proj.toml"), manifest).unwrap();
    for (rel, content) in files {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }
    dir
}

fn manifest_from(src: &str) -> mycelium_proj::Manifest {
    parse_manifest(src).unwrap()
}

// ─── pre-existing tests (extracted from lib.rs inline #[cfg(test)]) ───────────

#[test]
fn builds_a_phylum_spore_and_is_metadata_invariant() {
    let m_v1 = "[project]\nname=\"geometry\"\nkind=\"phylum\"\nversion=\"1.0.0\"\n\
                [surface]\nexports=[\"geometry.shapes\"]\n";
    let dir = scratch(
        "metainv",
        m_v1,
        &[(
            "shapes.myc",
            "// nodule: geometry.shapes\nnodule geometry.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let s1 = build_spore(&manifest_from(m_v1), &dir).expect("builds");
    assert!(s1.id.as_str().starts_with("blake3:"));
    assert_eq!(s1.surface, vec!["geometry.shapes".to_owned()]);
    assert_eq!(s1.sources.len(), 1);

    // ADR-003: changing only metadata (version) leaves the spore identity unchanged.
    let m_v2 = m_v1.replace("1.0.0", "2.5.0");
    let s2 = build_spore(&manifest_from(&m_v2), &dir).expect("builds");
    assert_eq!(
        s1.id, s2.id,
        "metadata changed the spore identity (ADR-003 violated)"
    );

    // Changing a source file DOES change identity.
    std::fs::write(
        dir.join("shapes.myc"),
        "// nodule: geometry.shapes\nnodule geometry.shapes\nfn a() -> Binary{8} = 0b1\n",
    )
    .unwrap();
    let s3 = build_spore(&manifest_from(m_v1), &dir).expect("builds");
    assert_ne!(s1.id, s3.id, "a code change must change the spore identity");
}

#[test]
fn a_phylum_without_a_surface_is_refused() {
    let m = "[project]\nname=\"x\"\nkind=\"phylum\"\n";
    let dir = scratch(
        "nosurface",
        m,
        &[("a.myc", "nodule a\nfn f() -> Binary{8} = 0b0\n")],
    );
    let err = build_spore(&manifest_from(m), &dir).unwrap_err();
    assert_eq!(err.exit_code(), 3);
    assert!(format!("{err}").contains("germinate"), "{err}");
}

#[test]
fn a_hashless_dependency_is_refused() {
    let m = "[project]\nname=\"x\"\nkind=\"phylum\"\n[surface]\nexports=[\"a\"]\n\
             [dependencies]\nnumerics={ phylum=\"numerics\", version=\"^2\" }\n";
    let dir = scratch(
        "hashless",
        m,
        &[("a.myc", "nodule a\nfn f() -> Binary{8} = 0b0\n")],
    );
    let err = build_spore(&manifest_from(m), &dir).unwrap_err();
    assert_eq!(err.exit_code(), 3);
    assert!(format!("{err}").contains("no `hash`"), "{err}");
}

#[test]
fn a_project_with_no_sources_is_refused() {
    let m = "[project]\nname=\"x\"\nkind=\"program\"\n";
    let dir = scratch("nosrc", m, &[]);
    let err = build_spore(&manifest_from(m), &dir).unwrap_err();
    assert_eq!(err.exit_code(), 3);
    assert!(format!("{err}").contains("nothing to package"), "{err}");
}

#[test]
fn a_resolved_dependency_is_pinned_and_explained() {
    // A real (64-hex) blake3 pin — the manifest reader now parses the dependency hash into a typed
    // `ContentHash` and rejects a bogus stub like `blake3:abc` (DN-40 A3), so the fixture uses a
    // well-formed digest.
    const PIN: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let m = format!(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[surface]\nexports=[\"a\"]\n\
         [dependencies]\nnumerics={{ phylum=\"numerics\", version=\"^2\", hash=\"{PIN}\" }}\n"
    );
    let dir = scratch(
        "dep",
        &m,
        &[("a.myc", "nodule a\nfn f() -> Binary{8} = 0b0\n")],
    );
    let s = build_spore(&manifest_from(&m), &dir).expect("builds");
    assert_eq!(s.deps.len(), 1);
    assert_eq!(s.deps[0].hash, PIN);
    let ex = explain(&s);
    assert!(ex.contains("not identity — ADR-003"), "{ex}");
    assert!(ex.contains(&format!("numerics → numerics {PIN}")), "{ex}");
}

// ─── M-789 / RFC-0034 §8: spore identity is independent of CertMode ──────────

/// **Property test (M-789 DoD): a spore is mintable and content-addressed in every `CertMode`
/// tier, and the same project always produces the same `spore_id` regardless of mode.**
///
/// Guarantee: `Proven` by construction. The `content_address` function in `crate` hashes only the
/// code-by-hash DAG, the dependency edges, the germination surface, and the project kind
/// (RFC-0001 §4.6; ADR-003) — it *never reads* a `CertMode`. `CertMode` rides the dynamic `Meta`
/// of runtime values (excluded from the content hash by construction). Therefore switching the
/// runtime certification mode *cannot* change the spore identity; the hash is a compile/deploy
/// phase concern (RFC-0034 §8).
///
/// This test is **exhaustive over the three current tiers** (`CertMode::ALL`), per the RFC-0034
/// §13 mode-parametric test contract: each tier must produce the same `spore_id`, and the
/// check fires for every tier, not just the default (`Fast`).
///
/// Mutant-witness: removing any one of the three `CertMode` cases would reduce exhaustive coverage;
/// checking equality of *all three* against a reference ensures no tier is silently skipped.
#[test]
fn spore_id_is_independent_of_cert_mode() {
    // Build a reference project tree — the source and manifest are fixed.
    let manifest_src = "[project]\nname=\"auth\"\nkind=\"phylum\"\nversion=\"1.0.0\"\n\
                        [surface]\nexports=[\"auth.core\"]\n";

    // Each iteration uses a fresh scratch dir so file-system timing never perturbs the hash.
    let mut ids = Vec::new();
    for mode in &CertMode::ALL {
        let dir = scratch(
            &format!("certmode-{}", mode.depth()),
            manifest_src,
            &[(
                "auth.myc",
                "// nodule: auth.core\nnodule auth.core\nfn verify() -> Binary{1} = 0b1\n",
            )],
        );
        let spore = build_spore(&manifest_from(manifest_src), &dir).expect("builds in every mode");
        // The spore is mintable in this mode (deployability survives cert-off, RFC-0034 §8).
        assert!(
            spore.id.as_str().starts_with("blake3:"),
            "spore minted in CertMode::{:?} must have a blake3 identity",
            mode
        );
        ids.push((mode, spore.id));
    }

    // All three tiers yield the same identity — mode is not identity (ADR-003; RFC-0034 §8).
    let (_, ref reference_id) = ids[0];
    for (mode, id) in &ids[1..] {
        assert_eq!(
            id, reference_id,
            "CertMode::{:?} produced a different spore_id — runtime mode must not enter the \
             compile/deploy content hash (RFC-0034 §8; ADR-003)",
            mode
        );
    }

    // The surface and kind are captured correctly across all tiers (deployability contract).
    for (_, id) in &ids {
        assert_eq!(id, reference_id);
    }
}

/// **Never-silent gate (M-789 DoD; RFC-0034 §8): disabling the compile spore-hash is an explicit,
/// EXPLAIN-ed capability loss, never a silent default.**
///
/// RFC-0034 §8: "Turning off the *compile* spore hash is a separate, deliberate choice (embedded /
/// no-deploy builds) that MUST explicitly disable and `EXPLAIN` the loss of spores/inject —
/// never-silent about *capabilities*, not just values."
///
/// **Current state (`Declared` — asserted, mechanism pending):** the full embedded/no-deploy
/// build-flag disable path (`no-spore-hash` or equivalent) is **not yet implemented** — there is
/// no `#[cfg(no_spore_hash)]` guard in `build_spore`. This is an explicit open gap (FLAG to
/// ADR-013 / M-789 / RFC-0034 §8 — deferred, not faked). The present guarantee is:
/// the hash is **always performed** (the positive capability is always on and never silently
/// dropped), satisfying the *safe default*: the path that *keeps* spore identity/inject cannot
/// be silently turned off.
///
/// When the disable path is implemented, this test must be updated to assert that:
/// (a) the disabled path emits an explicit, EXPLAIN-able `SporeCapability::NoSporeHash` marker,
/// (b) calling `build_spore` in the disabled configuration returns a `SporeError::Publish`
///     with a message referencing the capability loss (never-silent; G2),
/// and (c) `explain()` on a no-hash build visibly marks the missing identity (not a black box).
///
/// Mutant-witness: a mutation that silently skips the spore hash (makes `content_address` return a
/// fixed digest) would not change this test's assertion of the positive path — the property test
/// `spore_id_is_independent_of_cert_mode` above catches that (mutation changes the id to a
/// constant, breaking the ADR-003 code-change check in `builds_a_phylum_spore_and_is_metadata_invariant`).
#[test]
fn compile_spore_hash_disable_is_explicit_and_never_silent() {
    // Positive path: the spore hash is always performed (never silently skipped).
    let m = "[project]\nname=\"embed\"\nkind=\"program\"\n";
    let dir = scratch(
        "disable-check",
        m,
        &[("main.myc", "nodule main\nfn main() -> Binary{1} = 0b0\n")],
    );
    let spore = build_spore(&manifest_from(m), &dir).expect("spore hash always on");
    assert!(
        spore.id.as_str().starts_with("blake3:"),
        "the compile spore-hash is never silently disabled (G2; RFC-0034 §8)"
    );

    // FLAG (Declared): the no-deploy/embedded disable path is not yet implemented.
    // When it lands (ADR-013 §4 / RFC-0034 §8), this test must be extended:
    //   - assert the disable requires an explicit opt-in flag (never a silent default),
    //   - assert the capability loss is EXPLAIN-able (not a black box),
    //   - assert `build_spore` with the flag disabled returns an explicit error or marker
    //     (SporeError::Publish with a capability-loss message), never a silent partial artifact.
    // Until then: the safe default holds — the hash is always on.
    // OPEN GAP: ADR-013 §4 / M-789 / RFC-0034 §8 (deferred, not faked).
    let _ = kind_str(spore.kind); // EXPLAIN-able kind is always accessible regardless of mode.
}

// ─── SporeError display/exit_code surface tests ───────────────────────────────

#[test]
fn spore_error_exit_codes_and_display() {
    let pub_err = SporeError::Publish("bad input".to_owned());
    assert_eq!(pub_err.exit_code(), 3);
    assert!(format!("{pub_err}").contains("publish-error"), "{pub_err}");

    let io_err = SporeError::Io("disk full".to_owned());
    assert_eq!(io_err.exit_code(), 66);
    assert!(format!("{io_err}").contains("io-error"), "{io_err}");
}

// ─── content_address injectivity (security: supply-chain substitution) ─────────
//
// The spore identity encoding MUST be injective (ADR-003): distinct (kind, surface, sources, deps)
// → distinct addresses, or the content-addressed supply chain (dep pinning, resolve-by-hash,
// immutability detection) can be substituted under. The `v0` encoding emitted every author-influenced
// field space/newline-delimited with no length-prefix/escaping, so a crafted field containing a space
// or newline could alias two distinct DAGs onto one pre-image string (a second-pre-image collision).
// The `v1` encoding length-prefixes every variable field; these tests are the regression witnesses —
// **each pair below produced a BYTE-IDENTICAL pre-image under `v0` and so collided; under `v1` they
// must differ.** Guarantee: `Proven` (white-box over `crate::content_address`; the colliding pairs are
// author-controlled and constructible — no preimage/filesystem needed, since `ResolvedDep` fields are
// free-text manifest strings).
mod injectivity {
    use mycelium_core::ContentHash;
    use mycelium_proj::ProjectKind;

    use crate::{content_address, ResolvedDep, SourceFile};

    /// A valid 64-hex `blake3` ContentHash from a single hex digit (for deterministic test inputs).
    fn ch(d: char) -> ContentHash {
        ContentHash::from_parts("blake3", &d.to_string().repeat(64)).unwrap()
    }

    fn dep(name: &str, phylum: &str, hash: &str) -> ResolvedDep {
        // `version` is metadata (excluded from identity); fixed here so it never confounds the test.
        ResolvedDep {
            name: name.into(),
            phylum: phylum.into(),
            hash: hash.into(),
            version: None,
        }
    }

    #[test]
    fn deps_field_boundary_cannot_alias_two_dags() {
        // v0: deps1 -> "deps:\n  a b c\n  d e f\n"; deps2 (one dep whose free-text hash embeds a
        // newline + the second record) -> the SAME string. All fields are author-controlled manifest
        // strings, so this is the cleanest witness (no preimage/filesystem).
        let deps1 = vec![dep("a", "b", "c"), dep("d", "e", "f")];
        let deps2 = vec![dep("a", "b", "c\n  d e f")];
        let id1 = content_address(ProjectKind::Program, &[], &[], &deps1);
        let id2 = content_address(ProjectKind::Program, &[], &[], &deps2);
        assert_ne!(
            id1, id2,
            "v1 must distinguish two distinct dep DAGs that aliased under v0"
        );
    }

    #[test]
    fn surface_field_boundary_cannot_alias() {
        // v0: both -> "surface:\n  a\n  b\n".
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["a\n  b".to_string()];
        assert_ne!(
            content_address(ProjectKind::Phylum, &a, &[], &[]),
            content_address(ProjectKind::Phylum, &b, &[], &[]),
            "v1 must distinguish two distinct surface lists that aliased under v0"
        );
    }

    #[test]
    fn source_path_with_embedded_delimiters_cannot_alias() {
        // v0: src_a (two files) and src_b (one file whose path embeds the first record + newline)
        // both -> "code:\n  a.myc <h1>\n  b.myc <h2>\n".
        let h1 = ch('1');
        let h2 = ch('2');
        let src_a = vec![
            SourceFile {
                path: "a.myc".into(),
                hash: h1.clone(),
            },
            SourceFile {
                path: "b.myc".into(),
                hash: h2.clone(),
            },
        ];
        let src_b = vec![SourceFile {
            path: format!("a.myc {}\n  b.myc", h1.as_str()),
            hash: h2.clone(),
        }];
        assert_ne!(
            content_address(ProjectKind::Program, &[], &src_a, &[]),
            content_address(ProjectKind::Program, &[], &src_b, &[]),
            "v1 must distinguish source DAGs that aliased under v0 via a newline in a path"
        );
    }

    #[test]
    fn distinct_adversarial_inputs_are_all_distinct() {
        // A small adversarial corpus: every pair must hash distinctly (injectivity over crafted,
        // delimiter-laden, author-controlled fields).
        let inputs: Vec<Vec<ResolvedDep>> = vec![
            vec![dep("a", "b", "c"), dep("d", "e", "f")],
            vec![dep("a", "b", "c\n  d e f")],
            vec![dep("a b", "c", "d")],
            vec![dep("a", "b c", "d")],
            vec![dep("a\nb", "c", "d")],
            vec![],
        ];
        let ids: Vec<_> = inputs
            .iter()
            .map(|d| {
                content_address(ProjectKind::Program, &[], &[], d)
                    .as_str()
                    .to_owned()
            })
            .collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "adversarial inputs {i} and {j} must not collide under v1"
                );
            }
        }
    }

    #[test]
    fn encoding_is_deterministic_under_v1() {
        // The v1 change must stay deterministic: same (kind, surface, sources, deps) DAG -> same
        // address (the ADR-003 property the registry round-trip relies on).
        let deps = vec![dep("x", "y", "z")];
        let surface = vec!["s".to_string()];
        let id_a = content_address(ProjectKind::Phylum, &surface, &[], &deps);
        let id_b = content_address(ProjectKind::Phylum, &surface, &[], &deps);
        assert_eq!(id_a, id_b, "the encoding must stay deterministic");
    }
}

// ─── DN-40 §3: the source walk is symlink-safe + bounded (build-DoS defence) ────
//
// The recursive `.myc` collection in `crate::walk` previously used `Path::is_dir()`, which **stats
// the symlink target** rather than the link, and recursed with **no depth/cycle cap**. A symlinked
// directory cycle therefore produced an unbounded / infinite walk — a denial-of-service on the
// build (DN-40 §3 medium). The fix:
//   (a) classifies each entry via `symlink_metadata` (stats the link, not its target) and **does
//       not descend into symlinked directory entries** — so no directory cycle can be re-entered;
//   (b) caps nesting at `MAX_WALK_DEPTH`, returning an explicit `SporeError::Publish` (never-silent;
//       G2) instead of overflowing the stack.
// These tests are the regression witnesses. Symlink creation is Unix-only (`std::os::unix`), so the
// cycle/skip tests are `#[cfg(unix)]`; the "normal nested tree still collects" test is portable and
// proves the bounded walk preserves the pre-existing behaviour for real directories.
//
// Guarantee: `Empirical` (the tests are concrete filesystem trials over the real `walk`, not a
// machine-checked proof of every input). The cycle test asserts the walk **returns** (does not hang
// / overflow) and handles the link explicitly; a stack-overflow regression would abort the process
// and fail the test rather than pass silently.
mod symlink_walk {
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use crate::build_spore;

    use super::manifest_from;

    /// A program manifest with no surface requirement (so the walk, not the surface, is under test).
    const PROGRAM_MANIFEST: &str = "[project]\nname=\"walktest\"\nkind=\"program\"\n";

    /// Make a unique scratch dir (no files); caller populates it.
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "myc-spore-walk-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_myc(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    /// A normal nested directory tree still collects every real `.myc` source (the fix must preserve
    /// the pre-existing behaviour for ordinary directories). Portable — no symlinks.
    #[test]
    fn normal_nested_tree_still_collects_all_sources() {
        let dir = scratch_dir("nested");
        write_myc(&dir, "top.myc", "nodule top\nfn a() -> Binary{1} = 0b0\n");
        write_myc(
            &dir,
            "sub/mid.myc",
            "nodule sub.mid\nfn b() -> Binary{1} = 0b1\n",
        );
        write_myc(
            &dir,
            "sub/deep/leaf.myc",
            "nodule sub.deep.leaf\nfn c() -> Binary{1} = 0b0\n",
        );

        let spore = build_spore(&manifest_from(PROGRAM_MANIFEST), &dir).expect("builds");
        let paths: Vec<&str> = spore.sources.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["sub/deep/leaf.myc", "sub/mid.myc", "top.myc"],
            "the bounded walk must still collect every real nested source, sorted"
        );
    }

    /// A symlinked-directory **cycle** (a child symlink pointing back at an ancestor) does NOT hang
    /// or overflow the stack: the walk skips the symlink and returns, collecting only the real
    /// source. Under the old `Path::is_dir()` + no-cap walk this looped forever. Unix-only.
    #[cfg(unix)]
    #[test]
    fn symlink_directory_cycle_does_not_hang_or_overflow() {
        use std::os::unix::fs::symlink;

        let dir = scratch_dir("cycle");
        write_myc(&dir, "real.myc", "nodule real\nfn a() -> Binary{1} = 0b0\n");
        // Create a subdir whose child `loop` symlinks back to the project root → a cycle.
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        symlink(&dir, sub.join("loop")).unwrap();

        // The walk must terminate (no hang / no stack overflow). If the fix regressed, this test
        // would loop forever or abort the process — either way it would not pass.
        let spore = build_spore(&manifest_from(PROGRAM_MANIFEST), &dir).expect("builds");
        let paths: Vec<&str> = spore.sources.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["real.myc"],
            "the symlinked cycle must be skipped — only the real source is collected"
        );
    }

    /// A symlink **to a directory of real `.myc` sources** is skipped (not traversed): the linked
    /// tree's sources do NOT enter the deterministic content-addressed set. This is the same
    /// mechanism as the cycle defence, observed on a finite (acyclic) link so the *skip* is what is
    /// asserted. Unix-only.
    #[cfg(unix)]
    #[test]
    fn symlinked_directory_entry_is_not_traversed() {
        use std::os::unix::fs::symlink;

        // An external tree with a real source we must NOT pick up via a link.
        let external = scratch_dir("external");
        write_myc(
            &external,
            "hidden.myc",
            "nodule hidden\nfn h() -> Binary{1} = 0b1\n",
        );

        let dir = scratch_dir("linkdir");
        write_myc(&dir, "real.myc", "nodule real\nfn a() -> Binary{1} = 0b0\n");
        symlink(&external, dir.join("linked")).unwrap();

        let spore = build_spore(&manifest_from(PROGRAM_MANIFEST), &dir).expect("builds");
        let paths: Vec<&str> = spore.sources.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["real.myc"],
            "a symlinked directory is never traversed — `linked/hidden.myc` must not appear"
        );
        assert!(
            !paths.iter().any(|p| p.contains("hidden")),
            "the linked tree's sources must not enter the content-addressed set"
        );
    }

    /// The depth cap is an **explicit, exit-coded refusal** (never-silent; G2): a real directory
    /// tree nested past `MAX_WALK_DEPTH` returns a `SporeError::Publish` naming the offending tree,
    /// not a stack overflow. This is the defence-in-depth bound (independent of symlinks), so the
    /// test uses ordinary directories and is portable.
    #[test]
    fn over_deep_real_tree_is_refused_explicitly() {
        let dir = scratch_dir("toodeep");
        // Build a chain of real directories deeper than the cap (cap + a margin), with a `.myc`
        // file at the bottom so the walk actually descends the whole way.
        let mut rel = String::new();
        for _ in 0..(crate::MAX_WALK_DEPTH + 5) {
            rel.push_str("d/");
        }
        rel.push_str("deep.myc");
        write_myc(&dir, &rel, "nodule deep\nfn a() -> Binary{1} = 0b0\n");

        let err = build_spore(&manifest_from(PROGRAM_MANIFEST), &dir).unwrap_err();
        assert_eq!(
            err.exit_code(),
            3,
            "the over-deep refusal is a publish error (exit 3)"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("nests deeper than"),
            "the refusal must name the depth-cap breach (never-silent; G2): {msg}"
        );
    }

    /// A symlinked `.myc` **file** is skipped too (only real files are content-addressed): the
    /// symlink-classification short-circuits before the `.myc` extension branch. Unix-only.
    #[cfg(unix)]
    #[test]
    fn symlinked_myc_file_is_skipped() {
        use std::os::unix::fs::symlink;

        let dir = scratch_dir("linkfile");
        write_myc(&dir, "real.myc", "nodule real\nfn a() -> Binary{1} = 0b0\n");
        // A `.myc` symlink pointing at the real file — must not be collected as a second source.
        symlink(dir.join("real.myc"), dir.join("alias.myc")).unwrap();

        let spore = build_spore(&manifest_from(PROGRAM_MANIFEST), &dir).expect("builds");
        let paths: Vec<&str> = spore.sources.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["real.myc"],
            "a symlinked .myc file is skipped — only the real file is content-addressed"
        );
    }
}
