//! `std.content` — Ring 1 / Tier A capability surface (M-523).
//!
//! The identity model made into an ergonomic, read-only library. The one load-bearing promise is
//! the **canonical-hash guarantee** (ADR-003 / RFC-0001 §4.6): a value's — or a definition's —
//! identity *is* the content-addressed digest of its **normalized** structure (and
//! types-including-`Repr`, and static contract), and **metadata is not identity**.
//!
//! # What this crate is
//! A Ring-1 ergonomic wrapper over `mycelium-core`'s content-hash surface (M-103). It adds **no
//! new trusted / unsafe code** (KC-3, `#![forbid(unsafe_code)]`): all the trust lives in the
//! kernel's normalizer + digest, which this crate only *reads*.
//!
//! # What this crate is NOT
//! - **Not** hashing-for-maps (`Hash`/`Hasher` for `Map`/`Set` buckets) — that is `collections`
//!   (M-511), a non-identity hash. Keeping this split explicit is a primary obligation
//!   (RFC-0016 §4.3).
//! - **Not** packaging / deployment — `spore` (M-522, ADR-013) *consumes* our digests, but owns
//!   the artifact.
//! - **Not** representation change — `std.swap` (M-516) owns conversion; `content` is purely
//!   observational.
//!
//! # Guarantee matrix
//! Every exported op has a row in [`guarantee_matrix::MATRIX`] (RFC-0016 §4.5 / spec §4).
//! All ops are `Exact` (deterministic) and effect-free. The matrix is asserted in tests, not
//! prose-only (C2 / VR-5).
//!
//! # C1 — never-silent
//! - Total ops (`hash_of_value`, `hash_of_def`, `digest_eq`, `as_ref`, `names_of`) cannot fail.
//! - `parse_ref` returns `Err(`[`MalformedDigest`]`)` — never a coerced/zeroed digest.
//! - `resolve_name` returns `Option` — `None` is an honest "not found", never a sentinel hash.
//!
//! # C4 — content-addressed, value-semantic
//! Identical content collides; metadata is NOT identity; trivial renames do not change identity
//! (ADR-003; RFC-0001 §4.6 / §4.8).
//!
//! # Design spec
//! `docs/spec/stdlib/content.md` (M-523, #164).
//!
//! # Open questions (FLAGs carried from spec §7)
//!
//! - **(Q1)** The concrete digest algorithm (BLAKE3) is fixed by M-103 in the kernel; this crate
//!   re-exports whatever `mycelium-core` provides. The kernel spec notes BLAKE3 as the chosen
//!   algorithm (M-103 acceptance); this crate treats it as an implementation detail, not a
//!   surface commitment.
//!
//! - **(Q2)** `hash_of_value` is exposed as a total op over the kernel's `Value::content_hash`.
//!   The spec FLAGs whether non-`Exact`/mixed-paradigm composite values have a defined identity
//!   or are an explicit refusal (RFC-0001 §4.7 r3 "the interpreter refuses"). Currently the
//!   kernel's `Value::content_hash` is total, so we surface it as total.
//!   // FLAG: Q2 — if the kernel adds a refusal path for mixed-paradigm values, this op becomes
//!   // fallible (`Result<ContentHash, IdentityRefusal>`). Do not silently silentise that path.
//!
//! - **(Q3)** The `hash ↔ name` map ownership is an open question (spec §7-Q3 / RFC-0016 §8-Q2).
//!   The [`NameRegistry`] and `resolve_name` / `names_of` functions are here for now; if the
//!   resolution moves the registry to the toolchain, those two ops move out.
//!   // FLAG: Q3 — NameRegistry placement pending spec §7-Q3 resolution.
//!
//! - **(Q4)** How implicit `ContentHash` derivation may be at the call site is an open ergonomics
//!   question (spec §7-Q4 / RFC-0016 §8-Q3 — tension A). This crate is always-explicit;
//!   the per-ring ergonomics pass (M-540) may introduce implicit derivation later.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A):** Content-addressed operations are representation-agnostic
//! at the content level — the canonical hash of a value includes its `Repr` field as part of the
//! normalized structure (RFC-0001 §4.6), so representation is carried in the identity itself,
//! not implicit. The certificate of identity is always inspectable; metadata (including `Repr`) is
//! never stripped from the hash input (ADR-003).
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/content.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.

#![forbid(unsafe_code)]

pub mod content_ref;
pub mod error;
pub mod guarantee_matrix;
pub mod name_registry;

pub use content_ref::{ContentRef, RefKind};
pub use error::MalformedDigest;
pub use mycelium_core::ContentHash;
pub use name_registry::NameRegistry;

// Re-export the kernel's Names side-table directly so callers who prefer the lower-level API
// can use it without going through the std.content wrapper.
pub use mycelium_core::content::Names;

// ─── Exported operations ─────────────────────────────────────────────────────

/// The content hash of a runtime *value*: its identity-bearing `Repr` + payload, with all dynamic
/// [`Meta`](mycelium_core::Meta) excluded (RFC-0001 §4.6).
///
/// # Guarantee tag: `Exact` (deterministic)
/// A content hash is a pure function of the value's normalized structure (RFC-0001 §4.6 WF4).
/// Two values with identical `Repr` + payload but different provenance/bounds **collide**
/// (same hash — correct, metadata is not identity — ADR-003). Two values differing in `Repr`
/// get different hashes (paradigm is identity-bearing — RFC-0001 §4.6).
///
/// # Fallibility: total
/// Every value has an identity; this op cannot fail. (C1: no failure mode to hide.)
///
/// # Effects: none
/// Pure computation; no IO, time, randomness, or allocation beyond the hash output.
///
/// # FLAG: Q2
/// If the kernel adds a refusal path for mixed-paradigm composite values, this signature will
/// change to `Result<ContentHash, IdentityRefusal>`. See module-level FLAG Q2.
#[must_use]
pub fn hash_of_value(v: &mycelium_core::Value) -> ContentHash {
    v.content_hash()
}

/// The content hash of a definition (hash-of-AST; RFC-0001 §4.6 `hash(def)`):
/// ```text
/// H( normalize(structure(def)) ‖ types_with_repr(def) ‖ static_contract(def) )
/// ```
///
/// # Guarantee tag: `Exact` (deterministic)
/// Identity is over the α-normalized structure, types-with-`Repr`, constant literals, operator
/// names, and swap contracts — **never** over binder names or dynamic value metadata (ADR-003).
/// Consequences (RFC-0001 §4.6 acceptance):
/// - Trivial renames do **not** change the hash (α-renaming a binder is not identity).
/// - Identical definitions **collide** (same hash — correct, not a bug).
/// - A paradigm change **does** change the hash (types include `Repr`).
///
/// # Fallibility: total
/// Every definition node has a content identity; this op cannot fail. (C1.)
///
/// # Effects: none
#[must_use]
pub fn hash_of_def(d: &mycelium_core::Node) -> ContentHash {
    d.content_hash()
}

/// Identity equality by digest: two content hashes are **the same identity** iff their digests
/// are equal.
///
/// # Guarantee tag: `Exact` (deterministic)
/// Digest comparison is a pure byte-equality test; it is reflexive, symmetric, and transitive.
///
/// # Fallibility: total
/// Comparing two digests cannot fail.
///
/// # Effects: none
///
/// # Note
/// This is *not* the same as structural `==` on the values themselves: two values with
/// different runtime state but the same normalized content (e.g. differing only in provenance)
/// are `digest_eq` (same identity) even if `v1 != v2` structurally (ADR-003).
#[must_use]
pub fn digest_eq(a: &ContentHash, b: &ContentHash) -> bool {
    a == b
}

/// Build a typed [`ContentRef`] that cert / policy / provenance / `spore` artifacts embed to
/// designate a content-addressed target.
///
/// The `kind` parameter carries the role of the reference (`Value`, `Def`, `Policy`, …) so
/// downstream modules can distinguish roles without parsing the digest (no black boxes, C3).
///
/// # Guarantee tag: `Exact` (deterministic)
/// A pure construction from a known kind and hash; total and effect-free.
///
/// # Fallibility: total
/// There is no failure mode.
///
/// # Effects: none
#[must_use]
pub fn as_ref(h: ContentHash, kind: RefKind) -> ContentRef {
    ContentRef::new(kind, h)
}

/// Parse a content-address string (`<algo>:<digest>`) into a [`ContentHash`].
///
/// # Guarantee tag: `Exact` (deterministic)
/// If the string is well-formed, the result is always the same for the same input.
///
/// # Fallibility: `Err(MalformedDigest)` on bad shape (C1 — never-silent)
/// A string that does not match the `<algo>:<digest>` pattern returns
/// `Err(`[`MalformedDigest`]`)`, carrying:
/// - the rejected input (so the caller can surface it), and
/// - a description of *why* it was rejected (G11 dual projection).
///
/// A zeroed / synthetic / sentinel digest is **never** returned for malformed input (C1).
///
/// # Effects: none
pub fn parse_ref(s: &str) -> Result<ContentHash, MalformedDigest> {
    ContentHash::parse(s).ok_or_else(|| {
        // Diagnose *which* rule is violated for a more useful error description (G11).
        let description = if !s.contains(':') {
            "missing ':' separator between algo and digest"
        } else {
            let (algo, digest) = s.split_once(':').expect("contains ':'");
            if algo.is_empty() {
                "algo part is empty"
            } else if digest.is_empty() {
                "digest part is empty"
            } else if !algo
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            {
                "algo must be [a-z0-9]+"
            } else {
                // digest contains an invalid character
                "digest must be [A-Za-z0-9_-]+"
            }
        };
        MalformedDigest::new(s, description)
    })
}

/// Look up the name bound to a content hash in `registry`, returning `None` when the name is
/// unbound.
///
/// # Guarantee tag: `Exact` (deterministic)
/// A pure read of the name registry.
///
/// # Fallibility: `None` when name unbound (C1 — never-silent)
/// An unbound name is an honest `None`, **never** a sentinel hash or a fabricated name (C1).
///
/// # Effects: none
/// The name registry is a content-addressed, append-only side-table; reads are effect-free.
///
/// # FLAG: Q3
/// See module-level FLAG Q3 on registry ownership.
#[must_use]
pub fn resolve_name<'r>(registry: &'r NameRegistry, hash: &ContentHash) -> Option<&'r str> {
    registry.resolve_name(hash)
}

/// All names bound to `hash` in `registry`, as a list (0 or 1 entries with the current kernel;
/// see [`name_registry`] module-level FLAG on the one-name limitation).
///
/// Returns an empty `Vec` when no name is bound — **never** a sentinel (C1).
///
/// # Guarantee tag: `Exact` (deterministic)
/// A pure read of the name registry.
///
/// # Fallibility: total (possibly empty list)
/// The op cannot fail; an unbound hash yields an empty list (not an error).
///
/// # Effects: none
///
/// # FLAG: Q3
/// See module-level FLAG Q3 on registry ownership.
#[must_use]
pub fn names_of(registry: &NameRegistry, hash: &ContentHash) -> Vec<String> {
    registry.names_of(hash)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Integration-level tests over the public API surface.
    //!
    //! These tests assert the *behavioural* guarantees stated in the guarantee matrix and spec §4.
    //! The guarantee matrix itself is unit-tested in [`crate::guarantee_matrix::tests`].
    //!
    //! # Randomised identity / collision tests
    //! No sibling crate uses proptest, so we use randomised std tests (spec instruction: "else
    //! randomized std tests"). The randomised tests use a fixed seed (noted in each test) so they
    //! are reproducible; the seed is also the "mutant witness" (dev-workflow banked guard #7).

    use super::*;
    use mycelium_core::{
        meta::{Meta, Provenance},
        node::Node,
        repr::Repr,
        value::{Payload, Value},
    };

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn byte_value(bits: [bool; 8]) -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(bits.to_vec()),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed byte value")
    }

    const BITS_A: [bool; 8] = [true, false, true, true, false, false, true, false];
    const BITS_B: [bool; 8] = [false, true, false, false, true, true, false, true]; // bitwise NOT of A

    fn const_node(bits: [bool; 8]) -> Node {
        Node::Const(byte_value(bits))
    }

    fn let_node(binder: &str) -> Node {
        // let <binder> = BITS_A in <binder>   — used to test α-renaming invariance.
        Node::Let {
            id: binder.to_owned(),
            bound: Box::new(const_node(BITS_A)),
            body: Box::new(Node::Var(binder.to_owned())),
        }
    }

    // ─── hash_of_value tests ─────────────────────────────────────────────────

    /// `hash_of_value` is deterministic: the same value always yields the same hash.
    /// Guard: any randomness in `hash_of_value` makes this fail.
    #[test]
    fn hash_of_value_is_deterministic() {
        let v = byte_value(BITS_A);
        assert_eq!(hash_of_value(&v), hash_of_value(&v));
    }

    /// Two values with identical `Repr` + payload collide regardless of `Meta` (ADR-003).
    /// Guard: hashing Meta in `hash_of_value` makes this fail.
    #[test]
    fn hash_of_value_excludes_meta() {
        let exact_meta = byte_value(BITS_A);
        let derived_meta = Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(BITS_A.to_vec()),
            Meta::new(
                Provenance::Derived {
                    op: ContentHash::parse("blake3:someop").unwrap(),
                    inputs: vec![],
                },
                mycelium_core::GuaranteeStrength::Exact,
                None,
                None,
                None,
                None,
            )
            .expect("valid meta"),
        )
        .expect("well-formed");
        assert!(
            digest_eq(&hash_of_value(&exact_meta), &hash_of_value(&derived_meta)),
            "values differing only in Meta must collide (ADR-003 — metadata is not identity)"
        );
    }

    /// Values differing in payload get different hashes.
    /// Guard: truncating the payload in `hash_of_value` makes this fail.
    #[test]
    fn hash_of_value_different_payloads_differ() {
        assert!(!digest_eq(
            &hash_of_value(&byte_value(BITS_A)),
            &hash_of_value(&byte_value(BITS_B)),
        ));
    }

    /// Values differing in `Repr` paradigm get different hashes (RFC-0001 §4.6).
    /// Guard: excluding Repr from `hash_of_value` makes this fail.
    #[test]
    fn hash_of_value_different_repr_differ() {
        let binary = byte_value(BITS_A);
        let ternary = Value::new(
            Repr::Ternary { trits: 6 },
            Payload::Trits(vec![mycelium_core::value::Trit::Zero; 6]),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed ternary");
        assert!(!digest_eq(
            &hash_of_value(&binary),
            &hash_of_value(&ternary),
        ));
    }

    // ─── hash_of_def tests ───────────────────────────────────────────────────

    /// `hash_of_def` is deterministic.
    /// Guard: any randomness in `hash_of_def` makes this fail.
    #[test]
    fn hash_of_def_is_deterministic() {
        let n = let_node("x");
        assert_eq!(hash_of_def(&n), hash_of_def(&n));
    }

    /// α-renaming a binder does NOT change identity (RFC-0001 §4.6; ADR-003).
    /// Guard: including binder names in the hash makes this fail.
    #[test]
    fn hash_of_def_alpha_renaming_preserves_identity() {
        assert!(
            digest_eq(
                &hash_of_def(&let_node("x")),
                &hash_of_def(&let_node("long_name"))
            ),
            "α-renaming a binder must not change identity (RFC-0001 §4.6)"
        );
    }

    /// Identical definitions collide.
    /// Guard: any non-determinism in `hash_of_def` makes this fail.
    #[test]
    fn hash_of_def_identical_defs_collide() {
        assert!(digest_eq(
            &hash_of_def(&const_node(BITS_A)),
            &hash_of_def(&const_node(BITS_A)),
        ));
    }

    /// Distinct definitions get different hashes.
    /// Guard: any collision in `hash_of_def` beyond the correct ones makes this fail.
    #[test]
    fn hash_of_def_distinct_defs_differ() {
        assert!(!digest_eq(
            &hash_of_def(&const_node(BITS_A)),
            &hash_of_def(&const_node(BITS_B)),
        ));
    }

    /// A paradigm change changes identity (RFC-0001 §4.6).
    /// Guard: excluding Repr from the node hash makes this fail.
    #[test]
    fn hash_of_def_paradigm_change_changes_identity() {
        let bin_node = const_node(BITS_A);
        let tern_node = Node::Const(
            Value::new(
                Repr::Ternary { trits: 6 },
                Payload::Trits(vec![mycelium_core::value::Trit::Zero; 6]),
                Meta::exact(Provenance::Root),
            )
            .unwrap(),
        );
        assert!(!digest_eq(
            &hash_of_def(&bin_node),
            &hash_of_def(&tern_node),
        ));
    }

    // ─── digest_eq tests ─────────────────────────────────────────────────────

    /// `digest_eq` is reflexive.
    /// Guard: returning false for a==a makes this fail.
    #[test]
    fn digest_eq_is_reflexive() {
        let h = hash_of_def(&const_node(BITS_A));
        assert!(digest_eq(&h, &h));
    }

    /// `digest_eq` is symmetric.
    /// Guard: an asymmetric comparison makes this fail.
    #[test]
    fn digest_eq_is_symmetric() {
        let h1 = hash_of_def(&const_node(BITS_A));
        let h2 = hash_of_def(&const_node(BITS_A)); // same def → same hash
        assert_eq!(digest_eq(&h1, &h2), digest_eq(&h2, &h1));
    }

    /// `digest_eq` is transitive.
    /// Guard: a non-transitive equality makes this fail.
    #[test]
    fn digest_eq_is_transitive() {
        let h1 = hash_of_def(&let_node("a"));
        let h2 = hash_of_def(&let_node("b")); // α-equivalent → same hash
        let h3 = hash_of_def(&let_node("c")); // also α-equivalent
        if digest_eq(&h1, &h2) && digest_eq(&h2, &h3) {
            assert!(digest_eq(&h1, &h3), "digest_eq must be transitive");
        }
    }

    // ─── as_ref tests ────────────────────────────────────────────────────────

    /// `as_ref` wraps the hash with the given kind; the kind is inspectable (C3: no black boxes).
    #[test]
    fn as_ref_preserves_kind_and_hash() {
        let h = hash_of_def(&const_node(BITS_A));
        let r = as_ref(h.clone(), RefKind::Def);
        assert_eq!(r.kind(), RefKind::Def);
        assert_eq!(r.hash(), &h);
    }

    /// Two refs with the same hash but different kinds are not equal (kind is identity-bearing for
    /// the ref, not the target).
    /// Guard: ignoring kind in ContentRef equality makes this fail.
    #[test]
    fn as_ref_kind_is_part_of_ref_identity() {
        let h = hash_of_def(&const_node(BITS_A));
        let value_ref = as_ref(h.clone(), RefKind::Value);
        let def_ref = as_ref(h.clone(), RefKind::Def);
        assert_ne!(
            value_ref, def_ref,
            "kind must be part of ContentRef identity"
        );
    }

    // ─── parse_ref tests ─────────────────────────────────────────────────────

    /// `parse_ref` accepts well-formed addresses.
    #[test]
    fn parse_ref_accepts_well_formed() {
        assert!(parse_ref("blake3:Hh3kQ_x-1A").is_ok());
        assert!(parse_ref("sha256:abcdef0123456789").is_ok());
    }

    /// `parse_ref` rejects malformed addresses with a typed error (C1 — never-silent).
    /// The error carries both the rejected input and a description (G11).
    /// Guard: returning Ok for malformed input makes this fail.
    #[test]
    fn parse_ref_rejects_malformed_with_explicit_error() {
        // Mutant witness: replacing parse_ref with a function that returns Ok("") fails here.
        for bad in &[
            "no-colon",
            "blake3:",
            ":digest",
            "UPPER:abc",
            "blake3:has space",
        ] {
            let err = parse_ref(bad).expect_err("must reject malformed input");
            assert_eq!(&err.input, bad, "error must carry the rejected input");
            assert!(
                !err.description.is_empty(),
                "error must carry a description (G11)"
            );
        }
    }

    /// `parse_ref` returns a hash that round-trips back to the original string.
    #[test]
    fn parse_ref_round_trips() {
        let s = "blake3:Hh3kQ_x-1A";
        let h = parse_ref(s).expect("well-formed");
        assert_eq!(h.as_str(), s);
    }

    /// The error description disambiguates *which* rule is violated (G11 dual projection).
    /// Guard: returning the same generic description for all errors makes this fail.
    #[test]
    fn parse_ref_error_description_is_specific() {
        let missing_colon = parse_ref("nocolon").unwrap_err();
        let empty_algo = parse_ref(":digest").unwrap_err();
        // Different violations → different descriptions.
        assert_ne!(
            missing_colon.description, empty_algo.description,
            "error descriptions must distinguish rule violations (G11)"
        );
    }

    // ─── resolve_name / names_of tests ───────────────────────────────────────

    /// `resolve_name` returns `None` for an unbound hash (C1 — not a sentinel).
    #[test]
    fn resolve_name_none_for_unbound() {
        let reg = NameRegistry::new();
        let h = hash_of_def(&const_node(BITS_A));
        assert_eq!(resolve_name(&reg, &h), None);
    }

    /// `names_of` returns an empty list for an unbound hash (C1 — not a sentinel).
    #[test]
    fn names_of_empty_for_unbound() {
        let reg = NameRegistry::new();
        let h = hash_of_def(&const_node(BITS_A));
        assert!(names_of(&reg, &h).is_empty());
    }

    /// Binding a name and resolving it round-trips.
    #[test]
    fn resolve_name_and_names_of_after_bind() {
        let mut reg = NameRegistry::new();
        let h = hash_of_def(&const_node(BITS_A));
        reg.bind(h.clone(), "my_const");
        assert_eq!(resolve_name(&reg, &h), Some("my_const"));
        assert_eq!(names_of(&reg, &h), vec!["my_const".to_owned()]);
    }

    /// Binding a name does not change the hash (ADR-003 — metadata is not identity).
    /// Guard: any hash mutation on bind makes this fail.
    #[test]
    fn bind_does_not_change_hash_identity() {
        let mut reg = NameRegistry::new();
        let h_before = hash_of_def(&const_node(BITS_A));
        reg.bind(h_before.clone(), "before");
        let h_after = hash_of_def(&const_node(BITS_A));
        assert!(
            digest_eq(&h_before, &h_after),
            "binding a name must not change the hash (ADR-003)"
        );
    }

    // ─── Randomised round-trip / collision / identity property tests ──────────
    //
    // No sibling crate uses proptest, so we use randomised std tests (spec instruction).
    // A pseudo-random byte sequence is generated from a fixed seed (the "mutant witness" for
    // these tests is: replacing the fixed seed with a seed that produces identical bytes for
    // BITS_A and BITS_B — which would collapse distinct-payload tests).

    /// Randomised: 256 distinct single-byte payloads all get distinct hashes (collision test).
    /// One deterministic sample of 256 (all possible 8-bit values).
    #[test]
    fn hash_of_def_all_single_byte_payloads_are_distinct() {
        // Mutant witness: returning a constant hash from hash_of_def makes this fail.
        let mut hashes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0u8..=255 {
            let bits: [bool; 8] = [
                (i >> 7) & 1 == 1,
                (i >> 6) & 1 == 1,
                (i >> 5) & 1 == 1,
                (i >> 4) & 1 == 1,
                (i >> 3) & 1 == 1,
                (i >> 2) & 1 == 1,
                (i >> 1) & 1 == 1,
                i & 1 == 1,
            ];
            let h = hash_of_def(&const_node(bits));
            assert!(
                hashes.insert(h.as_str().to_owned()),
                "byte {i} produced a hash collision — content hash must be injective"
            );
        }
        assert_eq!(hashes.len(), 256);
    }

    /// Randomised: `hash_of_value` is deterministic for 16 distinct payloads and each pair of
    /// identical-payload values produces the same hash (C4 / collision property).
    /// (One deterministic sample of 16, seed = 0xCA_FE_BA_BE_DE_AD_BE_EF.)
    ///
    /// # Design note — `hash_of_value` vs `hash_of_def(Const(v))`
    /// These are intentionally **different** hashes (domain-separated in the kernel): `hash_of_value`
    /// hashes the `repr + payload` of the value directly (the `Canon::value` path), while
    /// `hash_of_def(Node::Const(v))` prepends a `CONST` node tag so a constant node and a bare
    /// value cannot collide. Both are correct and `Exact`; they describe *different things* (a value
    /// vs a definition node containing that value). This test confirms determinism and collision
    /// semantics for `hash_of_value` without conflating the two.
    #[test]
    fn hash_of_value_is_deterministic_and_collision_correct_for_16_samples() {
        // Mutant witness: returning a constant hash from hash_of_value collapses the hash set to 1.
        // Seed-derived bit strings (16 values, each 8 bits, from a simple LCG).
        let mut state: u64 = 0xCA_FE_BA_BE_DE_AD_BE_EF_u64;
        let mut seen: Vec<(u8, ContentHash)> = Vec::new();
        for _ in 0..16 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let byte = (state >> 56) as u8;
            let bits: [bool; 8] = [
                (byte >> 7) & 1 == 1,
                (byte >> 6) & 1 == 1,
                (byte >> 5) & 1 == 1,
                (byte >> 4) & 1 == 1,
                (byte >> 3) & 1 == 1,
                (byte >> 2) & 1 == 1,
                (byte >> 1) & 1 == 1,
                byte & 1 == 1,
            ];
            let v1 = byte_value(bits);
            let v2 = byte_value(bits); // identical payload
            let h1 = hash_of_value(&v1);
            let h2 = hash_of_value(&v2);
            assert!(
                digest_eq(&h1, &h2),
                "hash_of_value must be deterministic: identical values must collide (byte={byte:#010b})"
            );
            seen.push((byte, h1));
        }
        // Verify that distinct byte values produce distinct hashes (no spurious collisions in
        // this sample — not a proof of global injectivity, but a collision sanity check).
        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                if seen[i].0 != seen[j].0 {
                    assert!(
                        !digest_eq(&seen[i].1, &seen[j].1),
                        "hash_of_value produced a collision for distinct bytes {} and {}",
                        seen[i].0,
                        seen[j].0
                    );
                }
            }
        }
    }
}
