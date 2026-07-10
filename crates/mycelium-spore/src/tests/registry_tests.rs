//! Tests for `crate::registry` — publish/resolve round-trip, integrity, immutability, security.
//! Extracted from the old inline `#[cfg(test)]` block in `registry.rs` (CLAUDE.md test-layout rule).

use std::io::Write;
use std::path::PathBuf;

use mycelium_proj::parse_manifest;

use crate::registry::{artifact_hash, format_entry, parse_entry, publish, resolve, RegistryError};

// ─── helpers ──────────────────────────────────────────────────────────────────

fn scratch_registry(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "myc-registry-{tag}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a tiny phylum spore under a scratch project dir.
fn demo_spore(tag: &str, body: &str) -> (crate::Spore, Vec<u8>) {
    let m = "[project]\nname=\"geo\"\nkind=\"phylum\"\nversion=\"1.0.0\"\n\
             [surface]\nexports=[\"geo.shapes\"]\n";
    let dir = std::env::temp_dir().join(format!(
        "myc-spore-src-{tag}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut f = std::fs::File::create(dir.join("shapes.myc")).unwrap();
    f.write_all(body.as_bytes()).unwrap();
    let spore = crate::build_spore(&parse_manifest(m).unwrap(), &dir).unwrap();
    let descriptor = crate::explain(&spore).into_bytes();
    (spore, descriptor)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[test]
fn publish_then_resolve_round_trips_and_verifies_integrity() {
    let reg = scratch_registry("roundtrip");
    let (spore, descriptor) = demo_spore(
        "rt",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );

    let receipt = publish(&reg, &spore, &descriptor, "geo", "1.0.0").unwrap();
    assert_eq!(receipt.artifact, artifact_hash(&descriptor));
    assert_eq!(receipt.spore_id, spore.id);
    assert!(!receipt.already_present);

    let got = resolve(&reg, "geo", "1.0.0").unwrap();
    assert_eq!(got.bytes, descriptor);
    assert_eq!(got.spore_id, spore.id);
    assert_eq!(got.artifact, receipt.artifact);

    // `latest` resolves the single published version.
    assert_eq!(resolve(&reg, "geo", "latest").unwrap().version, "1.0.0");
}

#[test]
fn republish_is_idempotent_but_a_different_artifact_conflicts() {
    let reg = scratch_registry("immutable");
    let (s1, d1) = demo_spore(
        "a",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    publish(&reg, &s1, &d1, "geo", "1.0.0").unwrap();
    // Same artifact, same version → idempotent.
    let again = publish(&reg, &s1, &d1, "geo", "1.0.0").unwrap();
    assert!(again.already_present);

    // A different artifact under the SAME name@version is refused, never silently overwritten (G2).
    let (s2, d2) = demo_spore(
        "b",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b1\n",
    );
    let err = publish(&reg, &s2, &d2, "geo", "1.0.0").unwrap_err();
    assert_eq!(err.exit_code(), 6, "{err}");
    assert!(format!("{err}").contains("immutable"), "{err}");
}

#[test]
fn a_tampered_object_is_caught_on_resolve_not_silently_served() {
    let reg = scratch_registry("tamper");
    let (spore, descriptor) = demo_spore(
        "t",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    let receipt = publish(&reg, &spore, &descriptor, "geo", "1.0.0").unwrap();

    // Tamper with the stored object bytes (flip the tail).
    let mut bytes = std::fs::read(&receipt.object_path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xFF;
    std::fs::write(&receipt.object_path, &bytes).unwrap();

    let err = resolve(&reg, "geo", "1.0.0").unwrap_err();
    assert_eq!(err.exit_code(), 5, "{err}");
    assert!(format!("{err}").contains("content address"), "{err}");
}

#[test]
fn an_unpublished_name_or_version_is_not_found_never_invented() {
    let reg = scratch_registry("missing");
    assert_eq!(resolve(&reg, "nope", "1.0.0").unwrap_err().exit_code(), 4);
    let (spore, descriptor) = demo_spore(
        "m",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    publish(&reg, &spore, &descriptor, "geo", "1.0.0").unwrap();
    assert_eq!(resolve(&reg, "geo", "9.9.9").unwrap_err().exit_code(), 4);
}

#[test]
fn a_traversing_name_or_version_is_refused_not_joined_into_a_path() {
    // Security (G2): a name/version with `..` or a path separator must be refused before it can
    // escape the registry root — never silently joined.
    let reg = scratch_registry("traversal");
    let (spore, descriptor) = demo_spore(
        "tv",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    for bad in ["../escape", "a/b", "..", ".", "x\\y", ""] {
        let e = publish(&reg, &spore, &descriptor, bad, "1.0.0").unwrap_err();
        assert_eq!(e.exit_code(), 3, "name {bad:?} should be refused: {e}");
        let e2 = publish(&reg, &spore, &descriptor, "geo", bad).unwrap_err();
        assert_eq!(e2.exit_code(), 3, "version {bad:?} should be refused: {e2}");
    }
    // resolve refuses a traversing name too.
    assert_eq!(
        resolve(&reg, "../escape", "1.0.0").unwrap_err().exit_code(),
        3
    );
    // and a traversing exact-version constraint.
    publish(&reg, &spore, &descriptor, "geo", "1.0.0").unwrap();
    assert_eq!(resolve(&reg, "geo", "../1.0.0").unwrap_err().exit_code(), 3);
}

#[test]
fn a_range_constraint_is_unsupported_not_mis_resolved() {
    // VR-5: v0 must refuse a SemVer range rather than silently pretend to satisfy it.
    let reg = scratch_registry("range");
    let (spore, descriptor) = demo_spore(
        "r",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    publish(&reg, &spore, &descriptor, "geo", "1.2.3").unwrap();
    let err = resolve(&reg, "geo", "^1.0").unwrap_err();
    assert_eq!(err.exit_code(), 64, "{err}");
    assert!(format!("{err}").contains("range"), "{err}");
}

// --- property test: the hash-verification bound (M-732 DoD) ---
proptest::proptest! {
    /// For ANY descriptor bytes, the integrity address is the BLAKE3 of those bytes, and any
    /// single-byte mutation changes the address — so a tampered object can never hash to the
    /// recorded `artifact` (the never-silent integrity guarantee, G2). Guarantee: `Empirical`
    /// (trials) — BLAKE3 collision resistance itself is `Declared` upstream, not re-proven here.
    #[test]
    fn artifact_hash_is_stable_and_mutation_sensitive(
        bytes in proptest::collection::vec(proptest::num::u8::ANY, 1..256),
        idx in 0usize..256,
    ) {
        let h = artifact_hash(&bytes);
        // Deterministic: re-hashing the same bytes yields the same address.
        proptest::prop_assert_eq!(&h, &artifact_hash(&bytes));
        // Mutation-sensitive: flipping one byte changes the address (so resolve's check fires).
        let mut tampered = bytes.clone();
        let i = idx % tampered.len();
        tampered[i] = tampered[i].wrapping_add(1);
        proptest::prop_assert_ne!(h, artifact_hash(&tampered));
    }
}

#[test]
fn latest_picks_the_highest_version() {
    let reg = scratch_registry("latest");
    let (spore, descriptor) = demo_spore(
        "l",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    // Same artifact, multiple version labels (metadata is not identity — ADR-003).
    for v in ["1.0.0", "1.2.0", "1.10.0", "2.0.0"] {
        publish(&reg, &spore, &descriptor, "geo", v).unwrap();
    }
    // Numeric component ordering: 1.10.0 > 1.2.0, and 2.0.0 is the max.
    assert_eq!(resolve(&reg, "geo", "latest").unwrap().version, "2.0.0");
}

// ─── RegistryError display/exit_code surface tests ────────────────────────────

#[test]
fn registry_error_exit_codes_and_display() {
    let cases: &[(RegistryError, u8, &str)] = &[
        (RegistryError::InvalidInput("x".into()), 3, "input-error"),
        (RegistryError::NotFound("x".into()), 4, "not-found"),
        (RegistryError::Integrity("x".into()), 5, "integrity-error"),
        (RegistryError::Conflict("x".into()), 6, "conflict"),
        (RegistryError::Unsupported("x".into()), 64, "unsupported"),
        (RegistryError::Io("x".into()), 66, "io-error"),
    ];
    for (err, code, fragment) in cases {
        assert_eq!(err.exit_code(), *code, "{err}");
        assert!(format!("{err}").contains(fragment), "{err}");
    }
}

// ─── parse_entry: total-and-strict index read-back (DN-40 §3) ─────────────────

#[test]
fn parse_entry_round_trips_a_well_formed_entry() {
    // The happy path: format_entry → parse_entry recovers exactly the (spore_id, artifact) pair.
    let id = artifact_hash(b"identity-dag-bytes");
    let art = artifact_hash(b"descriptor-bytes");
    let text = format_entry(&id, &art);
    let (got_id, got_art) = parse_entry(&text).expect("a canonical entry parses");
    assert_eq!(got_id, id);
    assert_eq!(got_art, art);
}

#[test]
fn parse_entry_rejects_a_duplicate_key_never_last_wins() {
    // DN-40 §3(a): a second occurrence of a key must be a refused Integrity error, NOT silently the
    // last value (never-silent last-wins; G2).
    let id1 = artifact_hash(b"first-id");
    let id2 = artifact_hash(b"second-id");
    let art = artifact_hash(b"descriptor-bytes");
    let dup = format!(
        "spore_id = {}\nspore_id = {}\nartifact = {}\n",
        id1.as_str(),
        id2.as_str(),
        art.as_str()
    );
    let err = parse_entry(&dup).expect_err("a duplicate key is refused, not last-wins");
    assert_eq!(err.exit_code(), 5, "{err}");
    assert!(format!("{err}").contains("duplicate"), "{err}");
    assert!(format!("{err}").contains("spore_id"), "{err}");
}

#[test]
fn parse_entry_rejects_an_unrecognized_line_never_silently_ignored() {
    // DN-40 §3(b): a line that is neither blank nor a known key=value must be refused with an explicit
    // error naming the bad line (never silently ignored; G2). Covers an unknown key AND a non-pair line.
    let id = artifact_hash(b"identity");
    let art = artifact_hash(b"descriptor");

    // (i) an unrecognized key.
    let unknown_key = format!(
        "spore_id = {}\nartifact = {}\nmystery = 42\n",
        id.as_str(),
        art.as_str()
    );
    let e1 = parse_entry(&unknown_key).expect_err("an unrecognized key is refused");
    assert_eq!(e1.exit_code(), 5, "{e1}");
    assert!(format!("{e1}").contains("mystery"), "{e1}");

    // (ii) a line that is not a `key = value` pair at all.
    let junk_line = format!(
        "spore_id = {}\nartifact = {}\nthis is not a pair\n",
        id.as_str(),
        art.as_str()
    );
    let e2 = parse_entry(&junk_line).expect_err("a non-pair line is refused");
    assert_eq!(e2.exit_code(), 5, "{e2}");
    assert!(format!("{e2}").contains("this is not a pair"), "{e2}");
}

#[test]
fn parse_entry_rejects_missing_fields_and_bad_hashes() {
    // A missing required field and a malformed hash value are both explicit refusals (G2), not defaults.
    let id = artifact_hash(b"identity");
    let art = artifact_hash(b"descriptor");

    let missing_artifact = format!("spore_id = {}\n", id.as_str());
    let e1 = parse_entry(&missing_artifact).expect_err("a missing artifact line is refused");
    assert_eq!(e1.exit_code(), 5, "{e1}");
    assert!(format!("{e1}").contains("artifact"), "{e1}");

    let bad_hash = format!("spore_id = {}\nartifact = not-a-hash\n", id.as_str());
    let e2 = parse_entry(&bad_hash).expect_err("a malformed hash value is refused");
    assert_eq!(e2.exit_code(), 5, "{e2}");

    // A blank interior line is benign padding and does not break a valid entry.
    let with_blank = format!(
        "spore_id = {}\n\nartifact = {}\n",
        id.as_str(),
        art.as_str()
    );
    let (got_id, got_art) = parse_entry(&with_blank).expect("a blank padding line is tolerated");
    assert_eq!(got_id, id);
    assert_eq!(got_art, art);
}

#[test]
fn a_duplicate_keyed_index_entry_is_caught_on_resolve_not_last_wins() {
    // End-to-end: a hand-corrupted index file with a duplicate key surfaces as an Integrity refusal at
    // resolve, naming the offending entry — never silently resolved to the last value (G2).
    let reg = scratch_registry("dupkey");
    let (spore, descriptor) = demo_spore(
        "dk",
        "// nodule: geo.shapes\nnodule geo.shapes\nfn a() -> Binary{8} = 0b0\n",
    );
    let receipt = publish(&reg, &spore, &descriptor, "geo", "1.0.0").unwrap();

    // Append a second, conflicting spore_id line to the index entry (simulating corruption / tamper).
    let idx = reg.join("index").join("geo").join("1.0.0");
    let mut entry = std::fs::read_to_string(&idx).unwrap();
    let other = artifact_hash(b"a-different-identity");
    entry.push_str(&format!("spore_id = {}\n", other.as_str()));
    std::fs::write(&idx, entry).unwrap();
    let _ = receipt; // keep the published object around for the resolve attempt.

    let err = resolve(&reg, "geo", "1.0.0").unwrap_err();
    assert_eq!(err.exit_code(), 5, "{err}");
    assert!(format!("{err}").contains("duplicate"), "{err}");
}
