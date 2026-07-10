//! `NameRegistry` — the read/write `hash ↔ name` side-table (RFC-0001 §4.6).
//!
//! Names are *metadata*, not identity (ADR-003): they live here, not in the hash. A definition's
//! [`ContentHash`] is unchanged by adding, removing, or renaming its human-readable label. The
//! registry exposes two read-only ops (`resolve_name` / `names_of`) and one write op (`bind`) for
//! building up the table — the spec's C4 guarantee holds because binding or re-binding a name
//! does not touch the hash.
//!
//! # FLAG: Q3 (spec §7-Q3)
//! Whether this registry lives behind `std.content`, behind `core`/the prelude, or behind the
//! toolchain (LSP/registry) is not yet settled. This implementation places it here as the spec
//! describes, flagged for the maintainer's ratification. If the resolution moves the map out of
//! `std.content`, these two ops (`resolve_name` / `names_of`) are removed from this crate.
//!
//! # One-name limitation (FLAG)
//! The kernel's [`mycelium_core::content::Names`] stores at most *one* name per hash. The spec
//! sketch shows `names_of` returning `List<Str>` (potentially multiple names). This crate wraps
//! the kernel's map directly (KC-3: no new trusted code) and therefore surfaces *at most one*
//! name per hash. The multi-name surface is a follow-on design question (FLAG — pending spec §7-Q3
//! resolution + a registry redesign). Until then, `names_of` returns a `Vec<String>` with 0 or 1
//! entries, which is honest and correct for the current kernel.

use mycelium_core::{content::Names, ContentHash};

/// A read/write `hash ↔ name` registry (RFC-0001 §4.6 "names-as-metadata").
///
/// Wraps the kernel's [`Names`] side-table. Names are pure metadata; binding or re-binding a name
/// does not change a hash's identity. The module boundary is the spec §3 read-only surface
/// (`resolve_name` / `names_of`); `bind` is exposed for constructing and populating a registry
/// instance, which is a deliberate ergonomic affordance (spec §7-Q4).
///
/// # FLAG: Q3
/// The *ownership* of this registry (in `std.content` vs. the toolchain) is an open question
/// (spec §7-Q3). See module-level docs.
#[derive(Debug, Clone, Default)]
pub struct NameRegistry {
    inner: Names,
}

impl NameRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        NameRegistry {
            inner: Names::new(),
        }
    }

    /// Bind a human name to a content hash. Re-binding a different name is allowed and does not
    /// change identity (that is the whole point — ADR-003).
    ///
    /// Returns the previous name for that hash, if any.
    pub fn bind(&mut self, hash: ContentHash, name: impl Into<String>) -> Option<String> {
        self.inner.bind(hash, name)
    }

    /// Look up the name bound to `hash`, returning `None` when the name is unbound.
    ///
    /// # C1 compliance
    /// Returns `None` (an honest absence) — never a sentinel hash (RFC-0016 §4.1 C1).
    ///
    /// # Guarantee tag: `Exact`
    /// The result is a deterministic, pure read of the name table (RFC-0001 §4.6).
    #[must_use]
    pub fn resolve_name(&self, hash: &ContentHash) -> Option<&str> {
        self.inner.name_of(hash)
    }

    /// All names bound to `hash`, as a list (0 or 1 entries with the current kernel; see module
    /// FLAG on the one-name limitation).
    ///
    /// Returns an empty `Vec` when no name is bound — never a sentinel (C1).
    ///
    /// # Guarantee tag: `Exact`
    /// Deterministic pure read (RFC-0001 §4.6).
    #[must_use]
    pub fn names_of(&self, hash: &ContentHash) -> Vec<String> {
        match self.inner.name_of(hash) {
            Some(n) => vec![n.to_owned()],
            None => vec![],
        }
    }

    /// Number of names currently bound in the registry.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::NameRegistry;
    use mycelium_core::ContentHash;

    fn h(s: &str) -> ContentHash {
        ContentHash::parse(s).expect("test hash must be well-formed")
    }

    #[test]
    fn resolve_name_returns_none_for_unbound_hash() {
        // C1: None is the honest absence, not a sentinel (guard: returning a default string fails).
        let reg = NameRegistry::new();
        assert_eq!(reg.resolve_name(&h("blake3:unbound")), None);
    }

    #[test]
    fn names_of_returns_empty_for_unbound_hash() {
        // C1: empty Vec, not a sentinel (guard: returning a non-empty Vec fails).
        let reg = NameRegistry::new();
        assert!(reg.names_of(&h("blake3:unbound")).is_empty());
    }

    #[test]
    fn bind_and_resolve_round_trip() {
        let mut reg = NameRegistry::new();
        let hash = h("blake3:abc");
        assert_eq!(reg.bind(hash.clone(), "my_def"), None);
        assert_eq!(reg.resolve_name(&hash), Some("my_def"));
        assert_eq!(reg.names_of(&hash), vec!["my_def".to_owned()]);
    }

    #[test]
    fn rebind_returns_old_name_and_does_not_change_hash() {
        // ADR-003: re-binding a name does not touch the hash.
        let mut reg = NameRegistry::new();
        let hash = h("blake3:abc");
        reg.bind(hash.clone(), "first");
        let prev = reg.bind(hash.clone(), "second");
        assert_eq!(prev, Some("first".to_owned()));
        assert_eq!(reg.resolve_name(&hash), Some("second"));
        // The hash itself is unchanged — assert it still resolves.
        assert_eq!(reg.names_of(&hash), vec!["second".to_owned()]);
    }

    #[test]
    fn names_are_per_hash_not_global() {
        // Two distinct hashes carry independent names.
        let mut reg = NameRegistry::new();
        let h1 = h("blake3:aaa");
        let h2 = h("blake3:bbb");
        reg.bind(h1.clone(), "alpha");
        reg.bind(h2.clone(), "beta");
        assert_eq!(reg.resolve_name(&h1), Some("alpha"));
        assert_eq!(reg.resolve_name(&h2), Some("beta"));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn len_and_is_empty_track_bindings() {
        let mut reg = NameRegistry::new();
        assert!(reg.is_empty());
        reg.bind(h("blake3:x"), "x");
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }
}
