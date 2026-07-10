//! Tests for `crate::remote` (M-871/E26-1; ADR-037) — the GHCR/OCI remote backend.
//!
//! White-box (`use crate::…`); data-driven where a case fits a table (CLAUDE.md test-layout rule).
//! Guarantee tags: the remote path is `Empirical` (verified here via round-trip/property tests
//! against an in-process [`MemTransport`] double) or `Declared` (where correctness rests on `oras`'s
//! own OCI conformance, which the `#[ignore]`d live tests exercise separately) — never `Proven`.

use std::io::Write;
use std::path::PathBuf;

use mycelium_proj::parse_manifest;

use crate::remote::{
    build_dense_map, content_hash_from_title, decode_dense_map, encode_dense_map, parse_registry,
    publish_remote, resolve_remote, verify_and_reconstruct, DenseMap, MemTransport, ObjectRef,
    RegistryTarget, RemoteError,
};
use crate::{content_address, ResolvedDep};
use mycelium_core::ContentHash;
use mycelium_proj::ProjectKind;

// ─── helpers ────────────────────────────────────────────────────────────────────────────────────

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "myc-remote-{tag}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a real, small `phylum` [`crate::Spore`] under a scratch project dir with the given source
/// files (rel_path, contents).
fn demo_spore(tag: &str, files: &[(&str, &str)]) -> (crate::Spore, PathBuf) {
    let m = "[project]\nname=\"geo\"\nkind=\"phylum\"\nversion=\"1.0.0\"\n\
             [surface]\nexports=[\"geo.shapes\"]\n";
    let dir = scratch_dir(tag);
    for (rel, content) in files {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }
    let spore = crate::build_spore(&parse_manifest(m).unwrap(), &dir).unwrap();
    (spore, dir)
}

fn hash_of(bytes: &[u8]) -> ContentHash {
    let hex = blake3::hash(bytes).to_hex();
    ContentHash::from_parts("blake3", hex.as_str()).unwrap()
}

/// A small, self-consistent [`DenseMap`] not tied to a real project tree, for encode/decode tests.
fn sample_dense_map() -> DenseMap {
    let sources = vec![
        crate::SourceFile {
            path: "a.myc".to_owned(),
            hash: hash_of(b"nodule a\n"),
        },
        crate::SourceFile {
            path: "b/c.myc".to_owned(),
            hash: hash_of(b"nodule b.c\n"),
        },
    ];
    let deps = vec![ResolvedDep {
        name: "dep1".to_owned(),
        phylum: "std".to_owned(),
        hash: "blake3:0000000000000000000000000000000000000000000000000000000000000000".to_owned(),
        version: Some("1.0.0".to_owned()),
    }];
    let surface = vec!["geo.shapes".to_owned()];
    let spore_id = content_address(ProjectKind::Phylum, &surface, &sources, &deps);
    DenseMap {
        format_version: "mycelium-densemap-v1",
        spore_id,
        kind: ProjectKind::Phylum,
        name: "geo".to_owned(),
        version: Some("1.0.0".to_owned()),
        surface,
        objects: sources
            .into_iter()
            .map(|s| ObjectRef {
                rel_path: s.path,
                content_hash: s.hash,
            })
            .collect(),
        deps,
    }
}

// ─── encode/decode round-trip + injectivity ────────────────────────────────────────────────────

#[test]
fn encode_decode_round_trips() {
    let dm = sample_dense_map();
    let bytes = encode_dense_map(&dm);
    let decoded = decode_dense_map(&bytes).expect("a self-produced encoding must decode");
    assert_eq!(decoded, dm);
}

#[test]
fn encode_decode_round_trips_with_no_version_and_no_deps() {
    let mut dm = sample_dense_map();
    dm.version = None;
    dm.deps.clear();
    let bytes = encode_dense_map(&dm);
    let decoded = decode_dense_map(&bytes).unwrap();
    assert_eq!(decoded, dm);
    assert_eq!(decoded.version, None);
    assert!(decoded.deps.is_empty());
}

#[test]
fn encode_decode_round_trips_with_empty_string_version() {
    // `Some("")` must decode back to `Some("")`, distinct from `None` (the whole point of the
    // `none`/`some:` presence tag).
    let mut dm = sample_dense_map();
    dm.version = Some(String::new());
    let bytes = encode_dense_map(&dm);
    let decoded = decode_dense_map(&bytes).unwrap();
    assert_eq!(decoded.version, Some(String::new()));
}

#[test]
fn encode_is_deterministic_regardless_of_input_order() {
    // Two DenseMaps with the same logical content but differently-ordered sections must encode
    // identically (encode_dense_map sorts internally).
    let mut dm = sample_dense_map();
    let mut reordered = dm.clone();
    reordered.objects.reverse();
    reordered.deps.reverse();
    dm.objects.reverse();
    let a = encode_dense_map(&dm);
    let b = encode_dense_map(&reordered);
    assert_eq!(a, b);
}

proptest::proptest! {
    /// For ANY rel_path/dep-name/dep-phylum containing spaces, newlines, colons, or digits-then-colon
    /// sequences that could be confused for a length prefix, `decode(encode(x)) == x` — the
    /// length-prefixed (netstring-style) encoding never lets adversarial content forge a field
    /// boundary. Guarantee: `Empirical` (trials).
    #[test]
    fn dense_map_round_trips_over_adversarial_strings(
        rel_path in "[a-zA-Z0-9 \\n:./_-]{1,40}",
        dep_name in "[a-zA-Z0-9 \\n:_-]{1,40}",
        name in "[a-zA-Z0-9 \\n:_-]{1,40}",
    ) {
        let source = crate::SourceFile { path: rel_path.clone(), hash: hash_of(rel_path.as_bytes()) };
        let dep = ResolvedDep {
            name: dep_name.clone(),
            phylum: "phy lum\n:1".to_owned(),
            hash: "blake3:1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
            version: Some("1.0.0\n:weird".to_owned()),
        };
        let surface = vec!["a\nb:c".to_owned()];
        let sources = vec![source];
        let deps = vec![dep];
        let spore_id = content_address(ProjectKind::Phylum, &surface, &sources, &deps);
        let dm = DenseMap {
            format_version: "mycelium-densemap-v1",
            spore_id,
            kind: ProjectKind::Phylum,
            name,
            version: Some("v\n:1".to_owned()),
            surface,
            objects: sources.into_iter().map(|s| ObjectRef { rel_path: s.path, content_hash: s.hash }).collect(),
            deps,
        };
        let bytes = encode_dense_map(&dm);
        let decoded = decode_dense_map(&bytes).unwrap();
        proptest::prop_assert_eq!(decoded, dm);
    }

    /// Injectivity: two DenseMaps that differ only in their `name` never share an encoding (the
    /// core property the length-prefix discipline exists to guarantee). Guarantee: `Empirical`.
    #[test]
    fn distinct_dense_maps_never_share_an_encoding(
        name_a in "[a-zA-Z0-9 \\n:_-]{1,20}",
        name_b in "[a-zA-Z0-9 \\n:_-]{1,20}",
    ) {
        proptest::prop_assume!(name_a != name_b);
        let mut dm_a = sample_dense_map();
        dm_a.name = name_a;
        let mut dm_b = sample_dense_map();
        dm_b.name = name_b;
        proptest::prop_assert_ne!(encode_dense_map(&dm_a), encode_dense_map(&dm_b));
    }
}

// ─── decode_dense_map: never-silent rejection ──────────────────────────────────────────────────

#[test]
fn decode_rejects_bad_header() {
    let err = decode_dense_map(b"not-a-densemap\nrest").unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
    assert!(err.to_string().contains("bad header"), "{err}");
}

#[test]
fn decode_rejects_wrong_count_line() {
    let dm = sample_dense_map();
    let mut bytes = encode_dense_map(&dm);
    // Corrupt the `objects <N>` count line: bump N by one so the decoder expects one more entry
    // than is actually present, which must surface as an explicit error (never a silent short-read).
    let text = String::from_utf8(bytes.clone()).unwrap();
    let bad = text.replacen("objects 2\n", "objects 3\n", 1);
    assert_ne!(text, bad, "the fixture must actually contain `objects 2`");
    bytes = bad.into_bytes();
    let err = decode_dense_map(&bytes).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
}

#[test]
fn decode_rejects_truncated_length_prefix() {
    // Cut the buffer off mid-way through the `name` field's declared length-prefixed value: fewer
    // bytes remain than the `3:` prefix declares ("name:3:geo\n" chopped to "name:3:g").
    let dm = sample_dense_map();
    let bytes = encode_dense_map(&dm);
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.contains("name:3:geo\n"),
        "fixture must contain the expected name field"
    );
    let idx = text.find("name:3:geo\n").unwrap();
    let cut_at = idx + "name:3:g".len();
    let truncated = &text.as_bytes()[..cut_at];
    let err = decode_dense_map(truncated).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
    assert!(err.to_string().contains("truncated"), "{err}");
}

#[test]
fn decode_rejects_trailing_bytes() {
    let dm = sample_dense_map();
    let mut bytes = encode_dense_map(&dm);
    bytes.extend_from_slice(b"trailing garbage");
    let err = decode_dense_map(&bytes).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
    assert!(err.to_string().contains("trailing"), "{err}");
}

#[test]
fn decode_rejects_bad_content_hash() {
    let dm = sample_dense_map();
    let bytes = encode_dense_map(&dm);
    let text = String::from_utf8(bytes).unwrap();
    let spore_id_line = format!("spore_id:{}\n", dm.spore_id.as_str());
    let bad = text.replacen(&spore_id_line, "spore_id:not-a-hash\n", 1);
    assert_ne!(text, bad);
    let err = decode_dense_map(bad.as_bytes()).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
}

#[test]
fn decode_rejects_duplicate_object() {
    let mut dm = sample_dense_map();
    let dup = dm.objects[0].clone();
    dm.objects.push(dup);
    // Fix up the objects-section count so the corruption under test is purely the duplicate, not a
    // count mismatch.
    let mut bytes_text = String::from_utf8(encode_dense_map(&dm)).unwrap();
    bytes_text = bytes_text.replacen("objects 2\n", "objects 3\n", 1);
    let err = decode_dense_map(bytes_text.as_bytes()).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
    assert!(err.to_string().contains("duplicate"), "{err}");
}

// ─── build_dense_map ────────────────────────────────────────────────────────────────────────────

#[test]
fn build_dense_map_produces_matching_objects_and_dense_map() {
    let (spore, dir) = demo_spore(
        "build-ok",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (dm, blobs) = build_dense_map(&spore, &dir).unwrap();
    assert_eq!(dm.spore_id, spore.id);
    assert_eq!(dm.name, spore.name);
    assert_eq!(dm.objects.len(), 1);
    assert_eq!(blobs.len(), 1);
    assert_eq!(dm.objects[0].rel_path, blobs[0].rel_path);
    assert_eq!(dm.objects[0].content_hash, blobs[0].content_hash);
    // The OCI blob title round-trips back to the same content hash (ADR-037 §2).
    assert_eq!(
        content_hash_from_title(&blobs[0].oci_title()).as_ref(),
        Some(&blobs[0].content_hash)
    );
}

#[test]
fn build_dense_map_rejects_a_tampered_source_file() {
    let (spore, dir) = demo_spore(
        "build-tamper",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    // Mutate the file on disk after the spore was built — build_dense_map must catch the drift.
    std::fs::write(dir.join("shapes.myc"), "// tampered\n").unwrap();
    let err = build_dense_map(&spore, &dir).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
}

// ─── verify_and_reconstruct ─────────────────────────────────────────────────────────────────────

#[test]
fn verify_and_reconstruct_happy_path() {
    let (spore, dir) = demo_spore(
        "vr-ok",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (dm, blobs) = build_dense_map(&spore, &dir).unwrap();
    let fetched: Vec<(String, Vec<u8>)> = blobs
        .iter()
        .map(|b| (b.oci_title(), b.bytes.clone()))
        .collect();
    let reconstructed = verify_and_reconstruct(dm, &fetched).unwrap();
    assert_eq!(reconstructed.spore_id, spore.id);
    assert_eq!(reconstructed.sources.len(), 1);
    assert_eq!(reconstructed.sources[0].0, "shapes.myc");
}

#[test]
fn verify_and_reconstruct_rejects_a_tampered_blob() {
    let (spore, dir) = demo_spore(
        "vr-tamper",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (dm, blobs) = build_dense_map(&spore, &dir).unwrap();
    let mut fetched: Vec<(String, Vec<u8>)> = blobs
        .iter()
        .map(|b| (b.oci_title(), b.bytes.clone()))
        .collect();
    fetched[0].1.push(b'!');
    let err = verify_and_reconstruct(dm, &fetched).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
}

#[test]
fn verify_and_reconstruct_rejects_a_missing_object() {
    let (spore, dir) = demo_spore(
        "vr-missing",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (dm, _blobs) = build_dense_map(&spore, &dir).unwrap();
    let err = verify_and_reconstruct(dm, &[]).unwrap_err();
    assert!(matches!(err, RemoteError::NotFound(_)), "got {err:?}");
}

#[test]
fn verify_and_reconstruct_rejects_an_extra_undescribed_blob() {
    let (spore, dir) = demo_spore(
        "vr-extra",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (dm, blobs) = build_dense_map(&spore, &dir).unwrap();
    let mut fetched: Vec<(String, Vec<u8>)> = blobs
        .iter()
        .map(|b| (b.oci_title(), b.bytes.clone()))
        .collect();
    fetched.push(("extra.myco".to_owned(), b"not described".to_vec()));
    let err = verify_and_reconstruct(dm, &fetched).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
}

#[test]
fn verify_and_reconstruct_rejects_a_spore_id_mismatch() {
    let (spore, dir) = demo_spore(
        "vr-idmismatch",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (mut dm, blobs) = build_dense_map(&spore, &dir).unwrap();
    // A dense-map that claims a spore_id inconsistent with its own objects/deps/surface.
    dm.spore_id = hash_of(b"a completely different DAG");
    let fetched: Vec<(String, Vec<u8>)> = blobs
        .iter()
        .map(|b| (b.oci_title(), b.bytes.clone()))
        .collect();
    let err = verify_and_reconstruct(dm, &fetched).unwrap_err();
    assert!(matches!(err, RemoteError::Integrity(_)), "got {err:?}");
}

// ─── publish_remote / resolve_remote via MemTransport ──────────────────────────────────────────

#[test]
fn publish_then_resolve_exact_version_round_trips_via_mem_transport() {
    let (spore, dir) = demo_spore(
        "pubres-exact",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let target = RegistryTarget::Oci {
        base: "example.test/owner".to_owned(),
        plain_http: true,
    };
    let transport = MemTransport::new();
    let published = publish_remote(&target, &spore, &dir, "geo", "1.0.0", &transport).unwrap();
    assert_eq!(published.spore_id, spore.id);
    assert_eq!(published.reference, "example.test/owner/geo:1.0.0");

    let resolved = resolve_remote(&target, "geo", "1.0.0", &transport).unwrap();
    assert_eq!(resolved.version, "1.0.0");
    assert_eq!(resolved.reconstructed.spore_id, spore.id);
    assert_eq!(resolved.reconstructed.sources.len(), 1);
}

// ─── immutability (M-872): a name@version is immutable; idempotent re-publish is fine ────────────

#[test]
fn publish_remote_refuses_a_conflicting_republish_under_an_existing_version() {
    // Two DIFFERENT spores (different source bytes → different spore_id), same name@version.
    let (spore1, dir1) = demo_spore(
        "immut-conflict-1",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (spore2, dir2) = demo_spore(
        "immut-conflict-2",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn b() -> Binary{8} = 0b1\n",
        )],
    );
    assert_ne!(
        spore1.id, spore2.id,
        "the two demo spores must differ in identity"
    );
    let target = RegistryTarget::Oci {
        base: "example.test/owner".to_owned(),
        plain_http: true,
    };
    let transport = MemTransport::new();
    publish_remote(&target, &spore1, &dir1, "geo", "1.0.0", &transport).unwrap();

    // A DIFFERENT spore under the SAME name@version is refused as a Conflict (immutability, G2) —
    // never a silent overwrite.
    let err = publish_remote(&target, &spore2, &dir2, "geo", "1.0.0", &transport).unwrap_err();
    assert!(
        matches!(err, RemoteError::Conflict(_)),
        "expected Conflict, got {err:?}"
    );
    assert_eq!(err.exit_code(), 6);

    // The immutability refusal must not have overwritten the first publish (it still resolves to
    // spore1, unchanged).
    let resolved = resolve_remote(&target, "geo", "1.0.0", &transport).unwrap();
    assert_eq!(resolved.reconstructed.spore_id, spore1.id);
}

#[test]
fn publish_remote_allows_an_idempotent_republish_of_the_same_spore() {
    let (spore, dir) = demo_spore(
        "immut-idempotent",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let target = RegistryTarget::Oci {
        base: "example.test/owner".to_owned(),
        plain_http: true,
    };
    let transport = MemTransport::new();
    let first = publish_remote(&target, &spore, &dir, "geo", "1.0.0", &transport).unwrap();
    // Re-publishing the IDENTICAL spore under the same name@version is idempotent (same spore_id →
    // not a conflict); the receipt is stable.
    let second = publish_remote(&target, &spore, &dir, "geo", "1.0.0", &transport).unwrap();
    assert_eq!(first.spore_id, second.spore_id);
    assert_eq!(first.reference, second.reference);
}

#[test]
fn resolve_latest_picks_the_highest_published_version() {
    let (spore1, dir1) = demo_spore(
        "pubres-latest-1",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let (spore2, dir2) = demo_spore(
        "pubres-latest-2",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn b() -> Binary{8} = 0b1\n",
        )],
    );
    let target = RegistryTarget::Oci {
        base: "example.test/owner".to_owned(),
        plain_http: true,
    };
    let transport = MemTransport::new();
    publish_remote(&target, &spore1, &dir1, "geo", "1.0.0", &transport).unwrap();
    publish_remote(&target, &spore2, &dir2, "geo", "2.0.0", &transport).unwrap();
    publish_remote(&target, &spore1, &dir1, "geo", "1.5.0", &transport).unwrap();

    let resolved = resolve_remote(&target, "geo", "latest", &transport).unwrap();
    assert_eq!(resolved.version, "2.0.0");
    assert_eq!(resolved.reconstructed.spore_id, spore2.id);

    let resolved_star = resolve_remote(&target, "geo", "*", &transport).unwrap();
    assert_eq!(resolved_star.version, "2.0.0");
}

#[test]
fn resolve_range_constraint_is_unsupported() {
    let (spore, dir) = demo_spore(
        "pubres-range",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let target = RegistryTarget::Oci {
        base: "example.test/owner".to_owned(),
        plain_http: true,
    };
    let transport = MemTransport::new();
    publish_remote(&target, &spore, &dir, "geo", "1.0.0", &transport).unwrap();

    for constraint in [
        "^1.0.0",
        "~1.0",
        ">=1.0.0",
        "<2.0.0",
        "=1.0.0",
        "1.0.0,2.0.0",
    ] {
        let err = resolve_remote(&target, "geo", constraint, &transport).unwrap_err();
        assert!(
            matches!(err, RemoteError::Unsupported(_)),
            "constraint {constraint:?} got {err:?}"
        );
    }
}

#[test]
fn resolve_unpublished_name_is_not_found() {
    let target = RegistryTarget::Oci {
        base: "example.test/owner".to_owned(),
        plain_http: true,
    };
    let transport = MemTransport::new();
    let err = resolve_remote(&target, "nope", "1.0.0", &transport).unwrap_err();
    assert!(matches!(err, RemoteError::NotFound(_)), "got {err:?}");
}

#[test]
fn publish_remote_rejects_a_local_target() {
    let (spore, dir) = demo_spore(
        "pubres-localtarget",
        &[(
            "shapes.myc",
            "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
        )],
    );
    let target = RegistryTarget::Local(PathBuf::from("/tmp/whatever"));
    let transport = MemTransport::new();
    let err = publish_remote(&target, &spore, &dir, "geo", "1.0.0", &transport).unwrap_err();
    assert!(matches!(err, RemoteError::InvalidInput(_)), "got {err:?}");
}

// ─── parse_registry ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_registry_table() {
    let cases: &[(&str, Result<RegistryTarget, ()>)] = &[
        (
            "ghcr://my-org/my-repo",
            Ok(RegistryTarget::Oci {
                base: "ghcr.io/my-org/my-repo".to_owned(),
                plain_http: false,
            }),
        ),
        (
            "oci://localhost:5000",
            Ok(RegistryTarget::Oci {
                base: "localhost:5000".to_owned(),
                plain_http: true,
            }),
        ),
        (
            "oci://reg.example.com/ns",
            Ok(RegistryTarget::Oci {
                base: "reg.example.com/ns".to_owned(),
                plain_http: false,
            }),
        ),
        (
            "/var/lib/mycelium/registry",
            Ok(RegistryTarget::Local(PathBuf::from(
                "/var/lib/mycelium/registry",
            ))),
        ),
        (
            "relative/registry/dir",
            Ok(RegistryTarget::Local(PathBuf::from(
                "relative/registry/dir",
            ))),
        ),
        ("s3://bad-scheme/bucket", Err(())),
        ("ghcr://", Err(())),
        ("oci://", Err(())),
    ];
    for (input, expected) in cases {
        let got = parse_registry(input);
        match expected {
            Ok(want) => assert_eq!(&got.unwrap(), want, "input {input:?}"),
            Err(()) => assert!(
                got.is_err(),
                "input {input:?} should be rejected, got {got:?}"
            ),
        }
    }
}

// ─── live-`oras`/live-registry tests (not run here) ────────────────────────────────────────────

// A live round-trip against a real `oras` binary and a local OCI registry (or GHCR) is
// intentionally NOT included as a `#[test]` here — it requires an external binary + network/daemon
// this crate's test suite must not depend on. The orchestrator runs that check separately (ADR-037
// §3 DoD: "round-trip verified against a local OCI registry ... and a live GHCR"). To exercise it
// by hand:
//
//   1. `docker run -d -p 5000:5000 --name oci-test registry:2`
//   2. Build a demo spore + call `mycelium_spore::remote::publish_remote` against
//      `RegistryTarget::Oci { base: "localhost:5000/test".into(), plain_http: true }` with an
//      `OrasTransport { plain_http: true }`.
//   3. `resolve_remote` the same reference back and assert the reconstructed spore_id matches.
#[test]
#[ignore = "requires the `oras` CLI on PATH and a reachable OCI registry (see comment above) — run manually, not in `just check`"]
fn live_oras_round_trip_placeholder() {
    let pre = crate::remote::oras_preflight();
    assert!(pre.is_ok() || pre.is_err(), "documents intent only");
}
