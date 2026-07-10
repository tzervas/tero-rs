//! Content addresses (RFC-0001 §4.6): `<algo>:<digest>`.

use serde::{Deserialize, Serialize};

/// A content address, e.g. `blake3:af13…` (64 hex). The kernel hash is **BLAKE3** (fixed in M-103),
/// rendered as `blake3:<64-hex>`; this type fixes the shape (`<algo>:<digest>`, matching the
/// `provenance.schema.json` pattern) and stays algorithm-agnostic so a future migration is a value
/// change, not a type change.
///
/// Two validation strengths are offered, deliberately distinct (DN-40 wave-2):
/// - [`ContentHash::parse`] / [`ContentHash::from_parts`] — **shape-only**, exactly matching the
///   normative JSON-Schema pattern `^[a-z0-9]+:[A-Za-z0-9_-]+$` (which documents that it "fixes only
///   the shape"). This is what (de)serialization uses, so the Rust type and the schema agree.
/// - [`ContentHash::parse_digest`] — **algorithm-aware**: additionally requires that a `blake3`
///   address carry a *real* 64-lowercase-hex digest, so a shape-valid-but-bogus stub like
///   `"blake3:abc"` is rejected (G2). This is the **centralized** check consumers that hold a
///   genuine, identity-bearing pin (e.g. a manifest dependency hash — DN-40 A3) should call, instead
///   of each re-deriving the 64-hex rule. See [`ContentHash::digest_well_formed`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash(String);

/// Width of a `blake3` digest in hex characters — `blake3::hash().to_hex()` emits a 256-bit hash as
/// exactly 64 lowercase hex (M-103). `pub(crate)` so the in-crate test module can assert the bound.
pub(crate) const BLAKE3_HEX_LEN: usize = 64;

impl ContentHash {
    /// Parse a content address, validating its **shape only**: `algo` is `[a-z0-9]+`, `digest` is
    /// `[A-Za-z0-9_-]+`, separated by a single `:`. This is exactly the normative schema pattern
    /// (`provenance.schema.json`: `^[a-z0-9]+:[A-Za-z0-9_-]+$`, "fixes only the shape"), so the Rust
    /// type and the on-wire schema agree — (de)serialization routes through here. Returns `None` if
    /// malformed.
    ///
    /// **Note:** this does *not* check that a `blake3` digest is a real 64-hex digest — a stub like
    /// `"blake3:abc"` is shape-valid and accepted here. A caller holding an identity-bearing pin
    /// (not a placeholder) should use [`ContentHash::parse_digest`], which adds the algorithm-aware
    /// digest check (DN-40 wave-2).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let (algo, digest) = s.split_once(':')?;
        if algo.is_empty() || digest.is_empty() {
            return None;
        }
        if !algo
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        {
            return None;
        }
        if !digest
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return None;
        }
        Some(ContentHash(s.to_owned()))
    }

    /// Parse a content address with **algorithm-aware digest validation** (DN-40 wave-2) — the
    /// centralized form. Validates the shape (as [`parse`](Self::parse)) **and**, for the kernel's
    /// fixed algorithm `blake3` (M-103), that the digest is a *real* digest: exactly
    /// [`BLAKE3_HEX_LEN`] lowercase hex `[0-9a-f]` (what `blake3::hash().to_hex()` emits). So a
    /// shape-valid-but-bogus stub like `"blake3:abc"` is **rejected**, not silently accepted (G2).
    /// Unknown algorithms stay **permissive** (shape-only) for forward-compatibility: a future hash
    /// migration is a value change, not a type change, and the kernel mints only `blake3` today.
    ///
    /// This centralizes the check DN-40 A3 first added inline in `mycelium-proj`'s manifest parser,
    /// so every consumer that holds an identity-bearing pin can share one canonical implementation
    /// rather than re-deriving the 64-hex rule. Returns `None` if malformed. **It is kept distinct
    /// from [`parse`](Self::parse)** because the normative JSON Schema is shape-only by design;
    /// (de)serialization must stay schema-aligned, so the stricter rule is opt-in at the call sites
    /// that want it.
    #[must_use]
    pub fn parse_digest(s: &str) -> Option<Self> {
        let h = Self::parse(s)?;
        if Self::digest_well_formed(h.algo(), h.digest()) {
            Some(h)
        } else {
            None
        }
    }

    /// Is `digest` a well-formed digest for `algo`? For `blake3` (M-103): exactly [`BLAKE3_HEX_LEN`]
    /// lowercase hex `[0-9a-f]`. For any other (forward-compat) algorithm: permissive — the address
    /// charset already enforced by [`parse`](Self::parse) is sufficient, so this returns `true`.
    /// This is the single source of truth for the algorithm-aware digest rule (DN-40 wave-2).
    #[must_use]
    pub fn digest_well_formed(algo: &str, digest: &str) -> bool {
        match algo {
            "blake3" => {
                digest.len() == BLAKE3_HEX_LEN
                    && digest
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }
            _ => true,
        }
    }

    /// Does this address carry a well-formed digest for its own algorithm (DN-40 wave-2)? Convenience
    /// over [`digest_well_formed`](Self::digest_well_formed) for an already-parsed address — lets a
    /// consumer that received a [`ContentHash`] via the shape-only path (e.g. `Deserialize`) assert
    /// the algorithm-aware rule without re-parsing.
    #[must_use]
    pub fn has_well_formed_digest(&self) -> bool {
        Self::digest_well_formed(self.algo(), self.digest())
    }

    /// Build a content address from an algorithm tag and digest, validating the **shape** (`algo` is
    /// `[a-z0-9]+`, `digest` is `[A-Za-z0-9_-]+`). Returns `None` if either part is malformed. This
    /// is the constructor the content-addressing pass (M-103) uses after computing a digest. Because
    /// M-103 feeds it a real 64-hex `blake3` digest, the shape check suffices; a caller that wants
    /// the algorithm-aware guarantee on untrusted parts can compose with
    /// [`digest_well_formed`](Self::digest_well_formed) or use [`parse_digest`](Self::parse_digest).
    #[must_use]
    pub fn from_parts(algo: &str, digest: &str) -> Option<Self> {
        Self::parse(&format!("{algo}:{digest}"))
    }

    /// The algorithm tag (the part before `:`), e.g. `blake3`.
    #[must_use]
    pub fn algo(&self) -> &str {
        self.0.split_once(':').map_or("", |(a, _)| a)
    }

    /// The digest (the part after `:`).
    #[must_use]
    pub fn digest(&self) -> &str {
        self.0.split_once(':').map_or("", |(_, d)| d)
    }

    /// The address as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for ContentHash {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Validate the shape on the way in — a malformed address is an error, never silent. The
        // shape-only check matches the normative JSON Schema (`provenance.schema.json`); the
        // algorithm-aware digest rule (DN-40) is opt-in via `parse_digest`/`has_well_formed_digest`.
        let s = String::deserialize(deserializer)?;
        ContentHash::parse(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("malformed content address: {s:?}")))
    }
}
