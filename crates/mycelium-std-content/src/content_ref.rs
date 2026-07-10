//! `ContentRef` — a typed, opaque pointer that cert / policy / provenance / spore artifacts embed
//! to designate a content-addressed target (spec §3; RFC-0001 §4.6; ADR-013).
//!
//! # Design
//! A `ContentRef` wraps a [`ContentHash`] with an explicit `Kind` tag so that downstream modules
//! (cert, policy, `spore`) can distinguish *what* the hash points to without inspecting the digest
//! itself. The tag is identity-bearing metadata **about the pointer** (not the target): two
//! `ContentRef`s with the same hash but different `Kind`s compare unequal, which is the correct
//! behaviour (a cert ref and a value ref to the same digest are different roles).
//!
//! # C3 / no black boxes
//! `ContentRef` is fully inspectable: callers can read both the `Kind` and the inner
//! `ContentHash` (via [`ContentRef::kind`] and [`ContentRef::hash`]). There is no hidden decision.
//!
//! # C4 / value-semantic
//! `ContentRef` is `Clone` + `PartialEq` + `Eq`. Identical `(kind, hash)` pairs are identical
//! refs; the struct carries no identity-independent metadata.

use crate::error::MalformedDigest;
use mycelium_core::ContentHash;

/// The role a [`ContentRef`] points to (the explicit kind tag).
///
/// This is metadata **about the pointer**, not the pointed-to value. It lets cert / policy /
/// `spore` modules distinguish roles without parsing the digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RefKind {
    /// A content-addressed runtime value (the result of [`crate::hash_of_value`]).
    Value,
    /// A content-addressed definition / AST node (the result of [`crate::hash_of_def`]).
    Def,
    /// A primitive operation (the result of [`mycelium_core::content::operation_hash`]).
    Operation,
    /// A policy artifact (RFC-0005 policy content-hash).
    Policy,
    /// A `spore` deployable (ADR-013).
    Spore,
    /// Any other role not covered by the variants above.
    Other,
}

/// A typed, opaque content reference — a `(kind, hash)` pair that cert / policy / provenance /
/// `spore` artifacts embed to designate a content-addressed target.
///
/// Constructed via [`crate::as_ref`] (from a `ContentHash`) or directly via
/// [`ContentRef::new`] when the kind is known at the call site.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentRef {
    kind: RefKind,
    hash: ContentHash,
}

impl ContentRef {
    /// Build a `ContentRef` from an explicit kind and hash.
    ///
    /// This is `Exact`/total — there is no failure mode.
    #[must_use]
    pub fn new(kind: RefKind, hash: ContentHash) -> Self {
        ContentRef { kind, hash }
    }

    /// The role this reference designates.
    #[must_use]
    pub fn kind(&self) -> RefKind {
        self.kind
    }

    /// The content-addressed identity this reference points to.
    #[must_use]
    pub fn hash(&self) -> &ContentHash {
        &self.hash
    }

    /// Consume the ref, returning the inner [`ContentHash`].
    #[must_use]
    pub fn into_hash(self) -> ContentHash {
        self.hash
    }

    /// The canonical string form of this reference: `<kind-prefix>+<algo>:<digest>`.
    ///
    /// Round-trips through the [`FromStr`] impl (`s.parse::<ContentRef>()`): for any `r`,
    /// `r.as_str_repr().parse::<ContentRef>() == Ok(r)`. This is the "machine" side of the G11
    /// dual projection; the human side is [`kind`](Self::kind) + a name lookup via
    /// [`crate::names_of`].
    ///
    /// [`FromStr`]: std::str::FromStr
    #[must_use]
    pub fn as_str_repr(&self) -> String {
        format!("{}+{}", self.kind_prefix(), self.hash.as_str())
    }

    fn kind_prefix(&self) -> &'static str {
        match self.kind {
            RefKind::Value => "value",
            RefKind::Def => "def",
            RefKind::Operation => "op",
            RefKind::Policy => "policy",
            RefKind::Spore => "spore",
            RefKind::Other => "other",
        }
    }
}

impl RefKind {
    /// The inverse of the kind-prefix used in [`ContentRef::as_str_repr`]; `None` for an
    /// unrecognised prefix (kept in lock-step with `ContentRef::kind_prefix`).
    fn from_prefix(prefix: &str) -> Option<RefKind> {
        Some(match prefix {
            "value" => RefKind::Value,
            "def" => RefKind::Def,
            "op" => RefKind::Operation,
            "policy" => RefKind::Policy,
            "spore" => RefKind::Spore,
            "other" => RefKind::Other,
            _ => return None,
        })
    }
}

impl std::str::FromStr for ContentRef {
    type Err = MalformedDigest;

    /// Parse the canonical `<kind-prefix>+<algo>:<digest>` form produced by
    /// [`ContentRef::as_str_repr`] — the exact inverse of that projection.
    ///
    /// # Errors
    /// Returns [`MalformedDigest`] (never a coerced value — C1/G2) when the string lacks the `+`
    /// separator, carries an unknown kind prefix, or its `<algo>:<digest>` tail is malformed.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (prefix, rest) = s.split_once('+').ok_or_else(|| {
            MalformedDigest::new(s, "expected `<kind-prefix>+<algo>:<digest>` (missing `+`)")
        })?;
        let kind = RefKind::from_prefix(prefix)
            .ok_or_else(|| MalformedDigest::new(s, "unknown content-ref kind prefix"))?;
        let hash = ContentHash::parse(rest).ok_or_else(|| {
            MalformedDigest::new(s, "malformed `<algo>:<digest>` content address")
        })?;
        Ok(ContentRef { kind, hash })
    }
}

#[cfg(test)]
mod tests {
    use super::{ContentRef, RefKind};
    use mycelium_core::ContentHash;

    fn example_hash() -> ContentHash {
        ContentHash::parse("blake3:abc123").expect("well-formed")
    }

    #[test]
    fn str_repr_round_trips_through_from_str() {
        // The doc on `as_str_repr` promises r.as_str_repr().parse() == Ok(r) for every kind.
        for kind in [
            RefKind::Value,
            RefKind::Def,
            RefKind::Operation,
            RefKind::Policy,
            RefKind::Spore,
            RefKind::Other,
        ] {
            let r = ContentRef::new(kind, example_hash());
            let parsed = r.as_str_repr().parse::<ContentRef>();
            assert_eq!(parsed, Ok(r), "round-trip must hold for {kind:?}");
        }
    }

    #[test]
    fn from_str_rejects_malformed_with_explicit_error() {
        // C1: each malformed shape is an explicit MalformedDigest, never a coerced value.
        assert!(
            "nokind".parse::<ContentRef>().is_err(),
            "missing '+' separator"
        );
        assert!(
            "bogus+blake3:abc123".parse::<ContentRef>().is_err(),
            "unknown kind prefix"
        );
        assert!(
            "value+nocolon".parse::<ContentRef>().is_err(),
            "malformed <algo>:<digest> tail"
        );
    }

    #[test]
    fn content_ref_kind_and_hash_round_trip() {
        let h = example_hash();
        let r = ContentRef::new(RefKind::Def, h.clone());
        assert_eq!(r.kind(), RefKind::Def);
        assert_eq!(r.hash(), &h);
    }

    #[test]
    fn content_ref_equality_is_kind_and_hash() {
        // Same kind + same hash → equal.
        let r1 = ContentRef::new(RefKind::Value, example_hash());
        let r2 = ContentRef::new(RefKind::Value, example_hash());
        assert_eq!(r1, r2);

        // Different kind → not equal, even for the same hash (guard: changing kind makes this fail).
        let r3 = ContentRef::new(RefKind::Def, example_hash());
        assert_ne!(r1, r3, "kind is part of ContentRef identity");
    }

    #[test]
    fn as_str_repr_includes_kind_prefix_and_hash() {
        let r = ContentRef::new(RefKind::Spore, example_hash());
        let s = r.as_str_repr();
        assert!(
            s.starts_with("spore+"),
            "repr must begin with the kind prefix"
        );
        assert!(
            s.ends_with(example_hash().as_str()),
            "repr must end with the hash string"
        );
    }

    #[test]
    fn into_hash_yields_inner_hash() {
        let h = example_hash();
        let r = ContentRef::new(RefKind::Policy, h.clone());
        assert_eq!(r.into_hash(), h);
    }
}
