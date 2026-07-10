//! Content-addressing for the doc-IR (ADR-003). A node's identity is the BLAKE3 hash of its
//! *projected content* — the same `blake3:<hex>` shape the kernel uses ([`mycelium_core::ContentHash`]),
//! so a doc node addresses exactly like a value node. This is what makes the dual human/machine
//! projection (G11) checkable: HTML and JSON are two renderers of the *same* hashed node, and the
//! §4.1 `dual-projection-parity` lint fails the build if they ever desync.
//!
//! The encoding is **canonical + injective**: every field is written tagged and length-prefixed, so
//! two structurally-distinct nodes cannot collide on an encoding (the same discipline as the kernel's
//! private `Canon`, reproduced here because that encoder is `pub(crate)` to core).

use mycelium_core::ContentHash;

/// A canonical, injective content hasher: tagged, length-prefixed writes feed a single BLAKE3 state.
pub struct DocHasher {
    h: blake3::Hasher,
}

impl Default for DocHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl DocHasher {
    /// A fresh hasher.
    #[must_use]
    pub fn new() -> Self {
        DocHasher {
            h: blake3::Hasher::new(),
        }
    }

    /// Absorb a one-byte domain/kind tag.
    pub fn tag(&mut self, t: u8) -> &mut Self {
        self.h.update(&[t]);
        self
    }

    /// Absorb a `u64` (little-endian, fixed width — framing is injective).
    pub fn u64(&mut self, n: u64) -> &mut Self {
        self.h.update(&n.to_le_bytes());
        self
    }

    /// Absorb a length-prefixed string (the prefix makes the framing injective).
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.h.update(s.as_bytes());
        self
    }

    /// Absorb an optional string distinctly from the empty string (tag 0 = none, 1 = some).
    pub fn opt_str(&mut self, s: Option<&str>) -> &mut Self {
        match s {
            None => self.tag(0),
            Some(v) => {
                self.tag(1);
                self.str(v)
            }
        }
    }

    /// Absorb an already-computed child address (a content hash), length-prefixed.
    pub fn child(&mut self, h: &ContentHash) -> &mut Self {
        self.str(h.as_str())
    }

    /// Finalize into the kernel's `blake3:<hex>` content-address shape.
    #[must_use]
    pub fn finish(&self) -> ContentHash {
        let hex = self.h.finalize().to_hex();
        // BLAKE3 hex is always 64 lowercase [0-9a-f] — a well-formed digest by construction.
        ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_address_has_the_kernel_shape() {
        let mut h = DocHasher::new();
        h.tag(1).str("hello");
        let id = h.finish();
        assert_eq!(id.algo(), "blake3");
        assert_eq!(id.digest().len(), 64);
        assert!(id.digest().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn distinct_content_gives_distinct_addresses() {
        let a = {
            let mut h = DocHasher::new();
            h.tag(1).str("a");
            h.finish()
        };
        let b = {
            let mut h = DocHasher::new();
            h.tag(1).str("b");
            h.finish()
        };
        assert_ne!(a.as_str(), b.as_str());
    }

    #[test]
    fn the_encoding_is_injective_across_framing() {
        // "ab" + "c" must not collide with "a" + "bc" — length-prefixing prevents it.
        let ab_c = {
            let mut h = DocHasher::new();
            h.str("ab").str("c");
            h.finish()
        };
        let a_bc = {
            let mut h = DocHasher::new();
            h.str("a").str("bc");
            h.finish()
        };
        assert_ne!(ab_c.as_str(), a_bc.as_str());
    }

    #[test]
    fn an_absent_field_differs_from_an_empty_one() {
        let none = {
            let mut h = DocHasher::new();
            h.opt_str(None);
            h.finish()
        };
        let empty = {
            let mut h = DocHasher::new();
            h.opt_str(Some(""));
            h.finish()
        };
        assert_ne!(none.as_str(), empty.as_str());
    }

    #[test]
    fn hashing_is_deterministic() {
        let mk = || {
            let mut h = DocHasher::new();
            h.tag(7).str("title").opt_str(Some("body")).u64(3);
            h.finish()
        };
        assert_eq!(mk().as_str(), mk().as_str());
    }
}
