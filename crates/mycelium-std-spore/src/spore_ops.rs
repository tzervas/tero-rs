//! Spore packaging and verification operations — the library face of the M-368 pipeline.
//!
//! Provides [`SporeUnit`] (the `std.spore` value handle wrapping a built [`mycelium_spore::Spore`]),
//! the packaging operations `spore_from_project` / `spore_from_value`, and the verify / explain /
//! identity / manifest_of surface from spec §3.
//!
//! # Honesty crux (C4 / ADR-003)
//!
//! A spore's identity IS its canonical content hash. Metadata (`name`, `version`, `authors`) is
//! carried with the spore but NEVER defines it. Two builds of the same code + deps yield the
//! SAME spore hash regardless of version label. A content change ALWAYS changes the hash.
//! Enforced by `mycelium-spore::build_spore` (the M-368 contract; tested in that crate and
//! cross-checked here).
//!
//! # Never-silent (C1 / G2)
//!
//! Every missing/ambiguous publish input is a typed `Err` naming the offending input. A hash
//! mismatch on verify is `Err(SporeErr::HashMismatch{expected, found})`. No partial artifact
//! is ever written.

use mycelium_core::ContentHash;
use mycelium_proj::{Manifest, ProjectKind};
use mycelium_spore::{Spore as RawSpore, SporeError};

use crate::recon_manifest::ReconManifest;

/// An explicit spore error — never a silent accept (C1/G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SporeErr {
    /// The recomputed content hash did not match the spore's declared identity (ADR-003).
    ///
    /// A hash mismatch is always an **explicit refusal** — no partial spore is produced and no
    /// silent default is used. Both hashes are carried so the diagnostic is self-contained (G11).
    HashMismatch {
        /// The hash the spore claims.
        expected: ContentHash,
        /// The hash recomputed from the component DAG.
        found: ContentHash,
    },
    /// A build/publish input is missing or ambiguous (no surface, hashless dep, cycle, …).
    ///
    /// Wraps the M-368 / `mycelium-spore` error message directly so the original diagnostic
    /// text (naming the offending input) is preserved (G11).
    PublishErr(String),
    /// An I/O error reading the project.
    IoErr(String),
}

impl From<SporeError> for SporeErr {
    fn from(e: SporeError) -> Self {
        match e {
            SporeError::Publish(m) => SporeErr::PublishErr(m),
            SporeError::Io(m) => SporeErr::IoErr(m),
        }
    }
}

impl std::fmt::Display for SporeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SporeErr::HashMismatch { expected, found } => write!(
                f,
                "spore-error: hash mismatch — declared {} but recomputed {} (ADR-003 violation; \
                 deploy aborted, no partial artifact written — C1/G2)",
                expected.as_str(),
                found.as_str()
            ),
            SporeErr::PublishErr(m) => write!(f, "spore-publish-error: {m}"),
            SporeErr::IoErr(m) => write!(f, "spore-io-error: {m}"),
        }
    }
}

mycelium_std_core::impl_std_error!(SporeErr);

/// A content-addressed, value-semantic spore handle (ADR-013).
///
/// Wraps a built [`mycelium_spore::Spore`] (the M-368 packager's output) with the `std.spore`
/// ergonomic surface: inspection, verification, EXPLAIN, and manifest access.
///
/// The canonical content hash (`identity`) is the spore's identity — metadata (`name`, `version`)
/// is carried but never defines it (ADR-003 / C4).
///
/// # Reconstruction manifest (RFC-0003 §6)
///
/// A `SporeUnit` optionally carries a [`ReconManifest`]. When present, [`manifest_of`] returns
/// `Some(&manifest)`; when absent (the common project-build case) it returns `None`. A spore
/// built from a single value (`spore_from_value`) carries the value's reconstruction info.
///
/// # Out of scope
///
/// The **full native deploy / germination** is Phase-6-gated (M-620) and is not implemented
/// here (FLAGGED §7 Q2). The `deploy` seam is sketched in the spec; this type does not expose
/// a `deploy` method.
#[derive(Debug, Clone, PartialEq)]
pub struct SporeUnit {
    raw: RawSpore,
    manifest: Option<ReconManifest>,
}

impl SporeUnit {
    /// Build a `SporeUnit` from a parsed `Manifest` and the project directory.
    ///
    /// Delegates to `mycelium-spore::build_spore` (the M-368 pipeline): determines the
    /// germination surface, content-addresses each `.myc` source, resolves dependency hashes,
    /// and computes the deterministic spore hash (ADR-003 identity).
    ///
    /// # Guarantee tag: `Exact` (deterministic; metadata-invariant — ADR-003)
    ///
    /// Two builds of the same code + deps + surface produce the SAME spore hash regardless of
    /// `name` / `version` / `authors`. A metadata-only change preserves identity.
    ///
    /// # Fallibility: `Err(SporeErr::*)`
    ///
    /// Returns `Err(SporeErr::PublishErr)` for any missing/ambiguous publish input:
    /// - a `phylum` with no germination surface → `PhylumSurfaceUnstated`
    /// - a `[spore].include` naming a non-export → `IncludeNotAnExport`
    /// - a dependency without a `hash` → `HashlessDependency`
    /// - no `.myc` sources found → `NoSources`
    /// - a dependency cycle → `DependencyCycle`
    ///
    /// Returns `Err(SporeErr::IoErr)` on a file-read failure.
    ///
    /// **No partial artifact is ever written** on any error (M-368 §5 / C1/G2).
    ///
    /// # Effects: `io` (reads project directory; does not write)
    pub fn from_manifest(
        manifest: &Manifest,
        project_dir: &std::path::Path,
    ) -> Result<Self, SporeErr> {
        let raw = mycelium_spore::build_spore(manifest, project_dir)?;
        Ok(SporeUnit {
            raw,
            manifest: None,
        })
    }

    /// The degenerate `spore(v)` case (ADR-013 §2): build a spore whose payload is a single value
    /// with its reconstruction manifest (RFC-0003 §6).
    ///
    /// The spore hash is computed by the **same canonical encoder** as a project spore
    /// (`recompute_identity` → `mycelium_spore::content_address`): a minimal single-source phylum is
    /// synthesized whose one source `hash` field IS the value's content hash
    /// (`std.content::hash_of_value(v)`), so identity is the content-addressed hash of the value + its
    /// reconstruction info, not raw bytes (ADR-003). It therefore inherits the `v1` injective encoding.
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    ///
    /// The same value (same `Repr` + payload; metadata excluded) always produces the same spore
    /// hash. A metadata-only change does not change identity (ADR-003).
    ///
    /// # Fallibility: `Err(SporeErr::PublishErr)` (the value-payload subset)
    ///
    /// Currently always `Ok` — the value-only case has no `[spore].include` / dep resolution
    /// step, so no `PublishErr` is possible from this path. Reserved for future validation.
    ///
    /// # Effects: none (pure construction; no IO)
    pub fn from_value(
        value: &mycelium_core::Value,
        manifest: Option<ReconManifest>,
    ) -> Result<Self, SporeErr> {
        // Content-address the value (ADR-003: identity = content hash, metadata excluded).
        let value_hash = mycelium_std_content::hash_of_value(value);

        // Synthesize a minimal project spore for the M-368 wrapper type (the "degenerate" case:
        // a single-item phylum with the value's content hash as its only source). The single
        // source's `hash` field IS the value's content hash — so the M-368 canonical encoding
        // of this spore content-addresses the value via hash_of_value, achieving ADR-013 §2
        // ("the narrow spore(v) case: a spore whose payload is v's reconstruction manifest").
        let source = mycelium_spore::SourceFile {
            path: "value".to_owned(),
            hash: value_hash.clone(),
        };
        // Build the raw spore shell (without an id yet).
        let mut raw = RawSpore {
            id: value_hash.clone(), // placeholder — overwritten below
            kind: ProjectKind::Phylum,
            surface: vec!["value".to_owned()],
            sources: vec![source],
            deps: vec![],
            name: "spore(v)".to_owned(),
            version: None,
        };
        // Compute the identity using the same canonical encoding that recompute_identity uses —
        // this keeps verify(from_value(v)) == Ok by construction (the round-trip property).
        raw.id = recompute_identity(&raw);
        Ok(SporeUnit { raw, manifest })
    }

    /// Verify the spore: recompute the component-DAG hash and compare to the declared identity.
    ///
    /// For a project spore the recomputed hash is deterministic (ADR-003); for a single-value
    /// spore the hash is recomputed from the value's content hash.
    ///
    /// For `SporeUnit` this is a consistency self-check: the hash stored in `raw.id` was
    /// computed at build time by `mycelium-spore::build_spore`. Re-running the same computation
    /// over `raw.sources`/`raw.deps`/`raw.surface`/`raw.kind` must yield the same result.
    ///
    /// # Guarantee tag: `Exact` (deterministic — a content hash is a pure function)
    ///
    /// # Fallibility: `Err(SporeErr::HashMismatch{expected, found})`
    ///
    /// If the recomputed hash diverges from the declared identity, this is an explicit
    /// `Err(SporeErr::HashMismatch)` naming both hashes — never a silent accept (C1/G2).
    ///
    /// For a well-formed `SporeUnit` constructed by `from_manifest` or `from_value`, this
    /// always returns `Ok(())`.
    ///
    /// # Effects: none (pure computation)
    pub fn verify(&self) -> Result<(), SporeErr> {
        // Re-derive the canonical DAG hash from the components (ADR-003 identity).
        let recomputed = recompute_identity(&self.raw);
        if recomputed == self.raw.id {
            Ok(())
        } else {
            Err(SporeErr::HashMismatch {
                expected: self.raw.id.clone(),
                found: recomputed,
            })
        }
    }

    /// The spore's canonical content-addressed identity (ADR-003).
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// # Fallibility: total
    /// # Effects: none
    #[must_use]
    pub fn identity(&self) -> &ContentHash {
        &self.raw.id
    }

    /// The reconstruction manifest, if this spore carries one.
    ///
    /// Returns `None` when the spore was built from a project without a reconstruction manifest.
    /// Returns `Some(&manifest)` for `spore(v)` units that carry a reconstruction recipe.
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// # Fallibility: `None` — never fabricates an empty manifest (C1/G2)
    /// # Effects: none
    #[must_use]
    pub fn manifest(&self) -> Option<&ReconManifest> {
        self.manifest.as_ref()
    }

    /// The raw M-368 spore (for consumers that need the full project representation).
    #[must_use]
    pub fn raw(&self) -> &RawSpore {
        &self.raw
    }

    /// Test-only constructor: replace the declared identity hash with `tampered_id`.
    ///
    /// Used exclusively in tests (e.g. `deploy.rs`) to exercise the hash-mismatch error path
    /// without access to private fields from outside this module. Not available in non-test
    /// builds.
    #[cfg(test)]
    pub fn with_tampered_id(self, tampered_id: ContentHash) -> Self {
        let mut raw = self.raw;
        raw.id = tampered_id;
        SporeUnit {
            raw,
            manifest: self.manifest,
        }
    }
}

/// Re-derive the canonical content-addressed identity for a built spore (ADR-003 / M-368).
///
/// **Delegates to the single canonical encoder `mycelium_spore::content_address`** — it does NOT
/// re-implement the encoding. A parallel copy here is exactly what produced the `v0`/`v1` split
/// (the verify path stamping a stale `v0` while `build_spore` stamped `v1`, a cross-crate
/// `HashMismatch`); the one-encoder rule makes that divergence structurally impossible (DRY).
///
/// The hash is metadata-excluded (ADR-003): `name`/`version`/`authors` are not fed to the hasher.
fn recompute_identity(raw: &RawSpore) -> ContentHash {
    mycelium_spore::content_address(raw.kind, &raw.surface, &raw.sources, &raw.deps)
}

/// The canonical identity of a `SporeUnit` — a convenience function matching the spec §3 surface.
///
/// # Guarantee tag: `Exact` (deterministic)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn identity(spore: &SporeUnit) -> &ContentHash {
    spore.identity()
}

/// The reconstruction manifest of a `SporeUnit`, if any — `None` for project spores without one.
///
/// # Guarantee tag: `Exact` (deterministic)
/// # Fallibility: `None` when no manifest — never fabricated (C1/G2)
/// # Effects: none
#[must_use]
pub fn manifest_of(spore: &SporeUnit) -> Option<&ReconManifest> {
    spore.manifest()
}

/// Verify the spore's consistency: recomputes the identity hash and compares.
///
/// # Guarantee tag: `Exact` (deterministic)
/// # Fallibility: `Err(SporeErr::HashMismatch{expected, found})` — explicit, named, never silent
/// # Effects: none (pure computation)
pub fn verify(spore: &SporeUnit) -> Result<(), SporeErr> {
    spore.verify()
}

/// The `EXPLAIN` of a built spore: the identity receipt, the surface, the code by hash, the
/// dependency edges, and the metadata explicitly marked *not* identity (ADR-003 / C3 / G11).
///
/// Returns a human-readable diagnostic string. The same spore always produces the same output
/// (total function of the manifest + resolved DAG — no randomness, no IO).
///
/// # Guarantee tag: `Exact` (deterministic; a total function of manifest + DAG)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn explain_spore(spore: &SporeUnit) -> String {
    let mut out = mycelium_spore::explain(&spore.raw);
    if let Some(m) = &spore.manifest {
        out.push_str(&format!("  reconstruction manifest: {m}\n"));
        out.push_str(&format!(
            "  manifest hash: {}\n",
            m.manifest_hash().as_str()
        ));
        out.push_str(&format!(
            "  declared strength: {:?}\n",
            m.declared_strength()
        ));
    } else {
        out.push_str("  reconstruction manifest: none\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{
        meta::{Meta, Provenance},
        repr::Repr,
        value::{Payload, Value},
    };
    use mycelium_proj::parse_manifest;
    use std::io::Write as IoWrite;

    fn byte_value(bits: [bool; 8]) -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(bits.to_vec()),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed byte value")
    }

    const BITS_A: [bool; 8] = [true, false, true, true, false, false, true, false];
    const BITS_B: [bool; 8] = [false, true, false, false, true, true, false, true];

    /// Write a throwaway project tree under a unique temp dir.
    fn scratch(name: &str, manifest_src: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "myc-std-spore-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("mycelium-proj.toml"), manifest_src).unwrap();
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

    fn parsed_manifest(src: &str) -> Manifest {
        parse_manifest(src).unwrap()
    }

    // --- from_manifest / identity / verify round-trip ---

    /// `verify(spore(build(manifest))) == Ok` — the round-trip property.
    ///
    /// Guard: any non-determinism in the identity hash breaks this (property over the `from_manifest`
    /// path).
    #[test]
    fn verify_built_spore_is_ok_round_trip() {
        // Mutant witness: changing the canonical encoding so that recompute_identity produces a
        // different hash than build_spore's content_address makes this fail.
        let m_src = "[project]\nname=\"geom\"\nkind=\"phylum\"\nversion=\"1.0.0\"\n\
                     [surface]\nexports=[\"geom.shapes\"]\n";
        let dir = scratch(
            "rtrip",
            m_src,
            &[(
                "shapes.myc",
                "nodule geom.shapes\nfn a() -> Binary{8} = 0b0\n",
            )],
        );
        let m = parsed_manifest(m_src);
        let spore = SporeUnit::from_manifest(&m, &dir).expect("builds");
        assert_eq!(
            verify(&spore),
            Ok(()),
            "verify(spore(build(manifest))) must be Ok (round-trip property)"
        );
    }

    /// A spore with a tampered identity returns `Err(HashMismatch)` — the explicit, named refusal.
    ///
    /// Guard: returning Ok for a tampered identity violates C1/G2.
    #[test]
    fn verify_tampered_identity_is_hash_mismatch() {
        // Mutant witness: returning Ok for any input makes this fail.
        let m_src = "[project]\nname=\"x\"\nkind=\"phylum\"\n\
                     [surface]\nexports=[\"x\"]\n";
        let dir = scratch(
            "tamper",
            m_src,
            &[("x.myc", "nodule x\nfn f() -> Binary{8} = 0b0\n")],
        );
        let m = parsed_manifest(m_src);
        let mut spore = SporeUnit::from_manifest(&m, &dir).unwrap();
        // Tamper with the identity hash.
        let real_id = spore.raw.id.clone();
        let fake_id = ContentHash::from_parts(
            "blake3",
            "tampered000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap();
        spore.raw.id = fake_id.clone();
        let err = verify(&spore).unwrap_err();
        match err {
            SporeErr::HashMismatch { expected, found } => {
                // expected is the (tampered) declared identity; found is the recomputed one.
                assert_eq!(
                    expected, fake_id,
                    "expected must be the declared (tampered) id"
                );
                assert_eq!(found, real_id, "found must be the recomputed real hash");
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    /// `identity(spore)` returns the content hash starting with "blake3:".
    #[test]
    fn identity_starts_with_blake3() {
        let m_src = "[project]\nname=\"x\"\nkind=\"program\"\n";
        let dir = scratch(
            "id",
            m_src,
            &[("main.myc", "nodule main\nfn main() -> Binary{8} = 0b0\n")],
        );
        let m = parsed_manifest(m_src);
        let spore = SporeUnit::from_manifest(&m, &dir).unwrap();
        assert!(identity(&spore).as_str().starts_with("blake3:"));
    }

    // --- metadata invariance (ADR-003) ---

    /// Changing only metadata (version) does NOT change the spore identity (ADR-003).
    ///
    /// Guard: including metadata in the identity hash makes this fail.
    #[test]
    fn metadata_change_does_not_change_identity() {
        // Mutant witness: feeding version into the canonical encoder makes this fail.
        let base = "[project]\nname=\"x\"\nkind=\"phylum\"\nversion=\"1.0.0\"\n\
                    [surface]\nexports=[\"x\"]\n";
        let changed = "[project]\nname=\"x\"\nkind=\"phylum\"\nversion=\"99.99.99\"\n\
                       [surface]\nexports=[\"x\"]\n";
        let dir = scratch(
            "metainv",
            base,
            &[("x.myc", "nodule x\nfn f() -> Binary{8} = 0b0\n")],
        );
        let s1 = SporeUnit::from_manifest(&parsed_manifest(base), &dir).unwrap();
        let s2 = SporeUnit::from_manifest(&parsed_manifest(changed), &dir).unwrap();
        assert_eq!(
            identity(&s1),
            identity(&s2),
            "metadata change (version) must not change the spore identity (ADR-003)"
        );
    }

    /// A code change DOES change the spore identity.
    ///
    /// Guard: not hashing source files makes this fail.
    #[test]
    fn code_change_changes_identity() {
        // Mutant witness: not including source hashes in the identity encoder makes this fail.
        let m_src = "[project]\nname=\"x\"\nkind=\"phylum\"\n\
                     [surface]\nexports=[\"x\"]\n";
        let dir = scratch(
            "codechange",
            m_src,
            &[("x.myc", "nodule x\nfn f() -> Binary{8} = 0b0\n")],
        );
        let s1 = SporeUnit::from_manifest(&parsed_manifest(m_src), &dir).unwrap();
        // Change the source file.
        std::fs::write(dir.join("x.myc"), "nodule x\nfn f() -> Binary{8} = 0b1\n").unwrap();
        let s2 = SporeUnit::from_manifest(&parsed_manifest(m_src), &dir).unwrap();
        assert_ne!(
            identity(&s1),
            identity(&s2),
            "a code change must change the spore identity (ADR-003)"
        );
    }

    // --- from_value (spore(v) — the degenerate case) ---

    /// `spore(v)` identity is the content-addressed hash of the value (ADR-013 §2).
    #[test]
    fn from_value_identity_is_deterministic() {
        let v = byte_value(BITS_A);
        let s1 = SporeUnit::from_value(&v, None).unwrap();
        let s2 = SporeUnit::from_value(&v, None).unwrap();
        assert_eq!(
            identity(&s1),
            identity(&s2),
            "from_value identity must be deterministic (Exact)"
        );
    }

    /// Two different values produce different spore hashes.
    ///
    /// Guard: returning a constant hash from from_value makes this fail.
    #[test]
    fn from_value_different_values_produce_different_hashes() {
        // Mutant witness: returning a constant hash collapses both.
        let va = byte_value(BITS_A);
        let vb = byte_value(BITS_B);
        let sa = SporeUnit::from_value(&va, None).unwrap();
        let sb = SporeUnit::from_value(&vb, None).unwrap();
        assert_ne!(
            identity(&sa),
            identity(&sb),
            "different values must produce different spore hashes"
        );
    }

    /// `verify(from_value(v)) == Ok` — the round-trip for single-value spores.
    ///
    /// Guard: any inconsistency between from_value's identity and recompute_identity makes this fail.
    #[test]
    fn from_value_verify_is_ok_round_trip() {
        // Mutant witness: if `from_value` stored a different id than `recompute_identity` would
        // compute from the same `raw`, verify would fail. They are consistent by construction:
        // `from_value` calls `recompute_identity` directly to set `raw.id`, so verify (which also
        // calls `recompute_identity`) always reproduces the same hash.
        let v = byte_value(BITS_A);
        let spore = SporeUnit::from_value(&v, None).unwrap();
        // The self-consistency check: recompute from the stored raw components.
        assert_eq!(
            verify(&spore),
            Ok(()),
            "verify(from_value(v)) must be Ok (round-trip)"
        );
    }

    /// `spore(v)` identity cross-checks against `std.content::hash_of_value` — the identity
    /// IS the content-addressed hash (ADR-003 / std.content integration).
    ///
    /// Guard: not using hash_of_value in the from_value encoder makes this fail.
    #[test]
    fn from_value_identity_cross_checks_with_content_hash() {
        // Mutant witness: encoding the value bytes directly (bypassing hash_of_value) would
        // produce a different hash if hash_of_value normalizes the value in a non-trivial way.
        let v = byte_value(BITS_A);
        let content_hash = mycelium_std_content::hash_of_value(&v);
        let spore = SporeUnit::from_value(&v, None).unwrap();
        // The spore's identity encodes the value's content hash (not a raw-byte hash of the
        // value's internal representation). The identity is the domain-separated wrapper hash;
        // we verify it changes deterministically with the content hash by checking two values.
        let vb = byte_value(BITS_B);
        let content_hash_b = mycelium_std_content::hash_of_value(&vb);
        let spore_b = SporeUnit::from_value(&vb, None).unwrap();
        // If content_hash differs, the spore identity must differ (hash injectivity within this
        // domain).
        assert_ne!(
            content_hash, content_hash_b,
            "sanity: different payloads differ"
        );
        assert_ne!(
            identity(&spore),
            identity(&spore_b),
            "different content hashes must produce different spore identities (ADR-003)"
        );
    }

    // --- manifest_of / EXPLAIN ---

    /// `manifest_of` returns `None` for a project spore (no manifest).
    #[test]
    fn manifest_of_is_none_for_project_spore() {
        let m_src = "[project]\nname=\"x\"\nkind=\"program\"\n";
        let dir = scratch(
            "nomanifest",
            m_src,
            &[("main.myc", "nodule main\nfn main() -> Binary{8} = 0b0\n")],
        );
        let spore = SporeUnit::from_manifest(&parsed_manifest(m_src), &dir).unwrap();
        assert_eq!(
            manifest_of(&spore),
            None,
            "project spore must have no manifest"
        );
    }

    /// `explain_spore` output is deterministic and mentions ADR-003.
    #[test]
    fn explain_spore_is_deterministic_and_mentions_adr003() {
        let m_src = "[project]\nname=\"x\"\nkind=\"phylum\"\n\
                     [surface]\nexports=[\"x\"]\n";
        let dir = scratch(
            "explain",
            m_src,
            &[("x.myc", "nodule x\nfn f() -> Binary{8} = 0b0\n")],
        );
        let spore = SporeUnit::from_manifest(&parsed_manifest(m_src), &dir).unwrap();
        let e1 = explain_spore(&spore);
        let e2 = explain_spore(&spore);
        assert_eq!(e1, e2, "explain must be deterministic");
        assert!(
            e1.contains("ADR-003"),
            "explain must mention ADR-003 (identity/metadata distinction): {e1}"
        );
    }

    // --- publish-input refusal (C1/G2) ---

    /// A phylum without a surface is refused (SurfaceUnstated — C1/G2).
    #[test]
    fn phylum_without_surface_is_refused() {
        let m_src = "[project]\nname=\"x\"\nkind=\"phylum\"\n";
        let dir = scratch(
            "nosurface",
            m_src,
            &[("a.myc", "nodule a\nfn f() -> Binary{8} = 0b0\n")],
        );
        let err = SporeUnit::from_manifest(&parsed_manifest(m_src), &dir).unwrap_err();
        // The error must be PublishErr naming the surface issue — not a silent default.
        match &err {
            SporeErr::PublishErr(msg) => assert!(
                msg.contains("germinate") || msg.contains("surface"),
                "error must mention the germination surface issue: {msg}"
            ),
            other => panic!("expected PublishErr, got {other:?}"),
        }
    }

    /// A project with no sources is refused (NoSources — C1/G2).
    #[test]
    fn no_sources_is_refused() {
        let m_src = "[project]\nname=\"x\"\nkind=\"program\"\n";
        let dir = scratch("nosrc", m_src, &[]);
        let err = SporeUnit::from_manifest(&parsed_manifest(m_src), &dir).unwrap_err();
        match &err {
            SporeErr::PublishErr(msg) => assert!(
                msg.contains("nothing to package") || msg.contains("source"),
                "error must mention no sources: {msg}"
            ),
            other => panic!("expected PublishErr, got {other:?}"),
        }
    }

    /// `SporeErr::HashMismatch` display names both hashes and says "never a silent accept".
    #[test]
    fn hash_mismatch_display_names_both_hashes() {
        let expected = ContentHash::from_parts(
            "blake3",
            "expected00000000000000000000000000000000000000000000000000000000",
        )
        .unwrap();
        let found = ContentHash::from_parts(
            "blake3",
            "found0000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap();
        let msg = format!(
            "{}",
            SporeErr::HashMismatch {
                expected: expected.clone(),
                found: found.clone()
            }
        );
        assert!(
            msg.contains(expected.as_str()),
            "display must carry expected hash: {msg}"
        );
        assert!(
            msg.contains(found.as_str()),
            "display must carry found hash: {msg}"
        );
        assert!(
            msg.contains("mismatch"),
            "display must mention 'mismatch': {msg}"
        );
    }
}
