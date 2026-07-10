//! White-box tests for [`crate::id::ContentHash`] — content-address shape (schema-aligned) and the
//! centralized algorithm-aware digest validation (DN-40 wave-2). Extracted from `id.rs` as-touched
//! (test-layout rule; M-797).

use crate::id::{ContentHash, BLAKE3_HEX_LEN};

/// A real, well-formed `blake3` digest: exactly 64 lowercase hex (M-103).
const VALID_BLAKE3: &str =
    "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn parses_well_shaped() {
    assert!(ContentHash::parse(VALID_BLAKE3).is_some());
}

#[test]
fn rejects_malformed_shape() {
    assert!(ContentHash::parse("no-colon").is_none());
    assert!(ContentHash::parse("blake3:").is_none());
    assert!(ContentHash::parse(":digest").is_none());
    assert!(ContentHash::parse("UPPER:abc").is_none());
    assert!(ContentHash::parse("blake3:has space").is_none());
}

/// `parse` is **shape-only** — it matches the normative schema pattern
/// (`provenance.schema.json`: `^[a-z0-9]+:[A-Za-z0-9_-]+$`, "fixes only the shape"), so a
/// shape-valid stub like `"blake3:abc"` is *accepted* here. The algorithm-aware rule is opt-in via
/// [`ContentHash::parse_digest`]. (This keeps the Rust type and the on-wire schema in agreement;
/// see the FLAG in the leaf report re. tightening `parse` itself.)
#[test]
fn parse_is_shape_only_so_stub_is_accepted() {
    assert!(ContentHash::parse("blake3:abc").is_some());
}

/// DN-40 wave-2 — the centralized algorithm-aware check. For the kernel algorithm `blake3` the
/// digest must be a *real* 64-hex digest; a shape-valid-but-bogus stub is rejected, never silently
/// accepted (G2). (A3 had to add this inline in `mycelium-proj`'s manifest parser; it is now
/// centralized in `ContentHash::parse_digest` / `digest_well_formed` so every consumer benefits.)
#[test]
fn parse_digest_requires_64_lower_hex_blake3() {
    // Accepted: exactly 64 lowercase hex.
    assert!(ContentHash::parse_digest(VALID_BLAKE3).is_some());
    // Rejected: too short (the A3 motivating stub).
    assert!(ContentHash::parse_digest("blake3:abc").is_none());
    // Rejected: wrong length — 63 and 65 hex bracket the boundary.
    assert!(
        ContentHash::parse_digest(&format!("blake3:{}", "a".repeat(BLAKE3_HEX_LEN - 1))).is_none()
    );
    assert!(
        ContentHash::parse_digest(&format!("blake3:{}", "a".repeat(BLAKE3_HEX_LEN + 1))).is_none()
    );
    // Accepted: exactly at the boundary.
    assert!(ContentHash::parse_digest(&format!("blake3:{}", "a".repeat(BLAKE3_HEX_LEN))).is_some());
    // Rejected: 64 chars but uppercase hex (the canonical form is lowercase).
    assert!(ContentHash::parse_digest(&format!("blake3:{}", "A".repeat(BLAKE3_HEX_LEN))).is_none());
    // Rejected: 64 chars but a non-hex letter (`g`).
    assert!(ContentHash::parse_digest(&format!("blake3:{}", "g".repeat(BLAKE3_HEX_LEN))).is_none());
    // Shape failures are still rejected by the stricter path too.
    assert!(ContentHash::parse_digest("blake3:has space").is_none());
}

/// Forward-compat policy: an **unknown** algorithm stays shape-only (permissive) even on the
/// algorithm-aware path — its digest is not constrained to the blake3 form, so a short digest is
/// accepted as long as the charset is legal. This keeps a future hash migration a value change, not
/// a type change.
#[test]
fn unknown_algo_is_shape_only_permissive() {
    // A non-blake3 algo with a short digest: accepted (no algo-specific length rule applies).
    assert!(ContentHash::parse_digest("sha256:abc").is_some());
    assert!(ContentHash::parse_digest("rcplan:0000").is_some());
    // The shape charset is still enforced even for unknown algos.
    assert!(ContentHash::parse_digest("sha256:has space").is_none());
    // The centralized predicate agrees with the parse path.
    assert!(ContentHash::digest_well_formed("sha256", "abc"));
}

/// The predicate and the `has_well_formed_digest` accessor are the single source of truth, usable on
/// an already-parsed (e.g. deserialized, shape-only) address without re-parsing.
#[test]
fn has_well_formed_digest_matches_predicate() {
    let good = ContentHash::parse(VALID_BLAKE3).expect("shape-valid");
    assert!(good.has_well_formed_digest());
    assert!(ContentHash::digest_well_formed("blake3", good.digest()));

    let stub = ContentHash::parse("blake3:abc").expect("shape-valid stub");
    assert!(!stub.has_well_formed_digest());
    assert!(!ContentHash::digest_well_formed("blake3", "abc"));
}

#[test]
fn from_parts_splits_back_out() {
    let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let h = ContentHash::from_parts("blake3", digest).expect("valid");
    assert_eq!(h.algo(), "blake3");
    assert_eq!(h.digest(), digest);
    assert_eq!(h.as_str(), VALID_BLAKE3);
    assert!(ContentHash::from_parts("blake3", "has space").is_none());
    assert!(ContentHash::from_parts("UPPER", "abc").is_none());
}
