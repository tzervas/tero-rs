//! A **content-addressed build cache** (M-312; ADR-003; RFC-0004 §4).
//!
//! Maps a build *request*'s content address to the [`BuildCertificate`] it produced, so re-building
//! identical inputs reuses the prior certificate instead of re-deciding. The key folds the **whole**
//! request — the component's identity hash *and* the decision inputs (spec ratification, the
//! obligations, the `promote` flag) — so a change in verification state is a **miss**, never a stale
//! hit. Reusing a certificate whose obligations no longer hold would be exactly the silent-staleness
//! failure the never-silent ethos forbids (G2), so the key is honest by construction.

use std::collections::HashMap;

use mycelium_core::{operation_hash, ContentHash};

use crate::{decide, BuildCertificate, Component};

/// The outcome of a cached build — and whether it was served from cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheOutcome {
    /// Served from cache: an identical request was built before (the certificate is reused).
    Hit(BuildCertificate),
    /// Freshly decided and stored (first time this exact request was seen).
    Miss(BuildCertificate),
}

impl CacheOutcome {
    /// The certificate, regardless of hit/miss.
    #[must_use]
    pub fn certificate(&self) -> &BuildCertificate {
        match self {
            CacheOutcome::Hit(c) | CacheOutcome::Miss(c) => c,
        }
    }

    /// Whether this was a cache hit.
    #[must_use]
    pub fn was_hit(&self) -> bool {
        matches!(self, CacheOutcome::Hit(_))
    }
}

/// A content-addressed cache of build certificates, keyed by the build request's content address.
#[derive(Debug, Clone, Default)]
pub struct BuildCache {
    entries: HashMap<ContentHash, BuildCertificate>,
}

impl BuildCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The content address of a build request: the component's identity hash folded with every
    /// input that can change the decision (spec ratification, the three obligations, `promote`).
    /// Two requests collide iff they are byte-identical in all of these (ADR-003).
    #[must_use]
    pub fn request_key(c: &Component, promote: bool) -> ContentHash {
        let o = c.obligations;
        operation_hash(&format!(
            "build-request.v1:{}:{}:{}:{}:{}:{}",
            c.hash.as_str(),
            c.spec_ratified,
            o.swap_certificates_valid,
            o.bound_checks_discharged,
            o.reference_equivalence_checked,
            promote,
        ))
    }

    /// Build `c` (promoting or not), serving the cached certificate on a hit or deciding-then-storing
    /// on a miss. Deterministic: an identical request always yields an equal certificate, so a hit is
    /// observationally identical to re-deciding (it only saves the work).
    pub fn build(&mut self, c: &Component, promote: bool) -> CacheOutcome {
        let key = Self::request_key(c, promote);
        if let Some(cert) = self.entries.get(&key) {
            return CacheOutcome::Hit(cert.clone());
        }
        let cert = decide(c, promote);
        self.entries.insert(key, cert.clone());
        CacheOutcome::Miss(cert)
    }

    /// The number of distinct requests cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Obligations;

    fn component(spec_ratified: bool, obligations: Obligations) -> Component {
        Component {
            hash: operation_hash("def.cache-test"),
            spec_ratified,
            obligations,
        }
    }

    #[test]
    fn an_unchanged_request_second_build_is_a_hit_reusing_the_certificate() {
        let mut cache = BuildCache::new();
        let c = component(true, Obligations::all());
        let first = cache.build(&c, true);
        let second = cache.build(&c, true);
        assert!(!first.was_hit(), "first build is a miss");
        assert!(second.was_hit(), "the unchanged second build is a hit");
        // The hit reuses the prior certificate verbatim (acceptance: M-312).
        assert_eq!(first.certificate(), second.certificate());
        assert_eq!(cache.len(), 1, "an identical request adds no new entry");
    }

    #[test]
    fn a_changed_obligation_is_a_miss_not_a_stale_hit() {
        // Mutant-witness: if request_key ignored the obligations, this would wrongly hit and return
        // the prior (AOT) certificate for a component that is no longer eligible — a silent stale
        // hit (G2 violation).
        let mut cache = BuildCache::new();
        let verified = component(true, Obligations::all());
        let promoted = cache.build(&verified, true);
        assert_eq!(promoted.certificate().route(), crate::ExecutionRoute::Aot);

        let mut weakened = Obligations::all();
        weakened.reference_equivalence_checked = false;
        let regressed = cache.build(&component(true, weakened), true);
        assert!(!regressed.was_hit(), "a changed obligation must miss");
        assert_eq!(
            regressed.certificate().route(),
            crate::ExecutionRoute::Interpreted,
            "the re-decided certificate reflects the weakened obligations"
        );
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn flipping_promote_is_a_distinct_request() {
        // Mutant-witness: dropping `promote` from the key would conflate a promoted and an
        // unpromoted build of the same definition.
        let mut cache = BuildCache::new();
        let c = component(true, Obligations::all());
        let unpromoted = cache.build(&c, false);
        let promoted = cache.build(&c, true);
        assert!(!unpromoted.was_hit() && !promoted.was_hit());
        assert_eq!(
            unpromoted.certificate().route(),
            crate::ExecutionRoute::Interpreted
        );
        assert_eq!(promoted.certificate().route(), crate::ExecutionRoute::Aot);
        assert_eq!(cache.len(), 2);
    }
}
