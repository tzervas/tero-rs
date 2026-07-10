//! Tests for `crate::inject_gate` (M-961; RFC-0038 §4/§7/§8; DN-77 §4).
//!
//! Fixture-driven per the house style: the refusal cases are a data table, a test body is an
//! assert over a case. Guarantee tags per test doc.

use crate::inject_gate::*;

fn signer(s: &str) -> SignerId {
    SignerId::new(s)
}

fn root(names: &[&str]) -> TrustRoot {
    TrustRoot::new(names.iter().map(|n| signer(n)))
}

// ─── germination refusals (never-silent; DN-77 §4 items 1/6) ─────────────────

/// The germination refusal table: (mode, grain, root, expected refusal).
/// `inoculated` + empty root refuses (a silent `loose` downgrade is forbidden — G2/I3);
/// `module`/`call` grains refuse (unenforced — M-847; DN-63 pattern).
/// Guarantee: `Proven` for the refusal firing (direct construction), the *policy* is `Declared`.
#[test]
fn germination_refusals_are_explicit() {
    let cases: &[(InjectMode, EnforcementGrain, TrustRoot, PolicyError)] = &[
        (
            InjectMode::Inoculated,
            EnforcementGrain::Whole,
            TrustRoot::empty(),
            PolicyError::EmptyTrustRoot,
        ),
        (
            InjectMode::Inoculated,
            EnforcementGrain::Module,
            root(&["k1"]),
            PolicyError::GrainNotYetEnforced(EnforcementGrain::Module),
        ),
        (
            InjectMode::Inoculated,
            EnforcementGrain::Call,
            root(&["k1"]),
            PolicyError::GrainNotYetEnforced(EnforcementGrain::Call),
        ),
        (
            InjectMode::Loose,
            EnforcementGrain::Module,
            TrustRoot::empty(),
            PolicyError::GrainNotYetEnforced(EnforcementGrain::Module),
        ),
    ];
    for (mode, grain, tr, expected) in cases {
        let got = InjectPolicy::germinate(*mode, *grain, tr.clone(), Box::new(TestScheme));
        assert_eq!(got.err().as_ref(), Some(expected), "{mode}/{grain}");
    }
    // The refusal explains itself (G2 — never a bare code).
    let e = PolicyError::GrainNotYetEnforced(EnforcementGrain::Call);
    assert!(format!("{e}").contains("M-847"), "{e}");
    assert!(
        format!("{}", PolicyError::EmptyTrustRoot).contains("RFC-0038"),
        "cites its basis"
    );
}

#[test]
fn valid_germinations_carry_their_posture() {
    let p = InjectPolicy::germinate(
        InjectMode::Inoculated,
        EnforcementGrain::Whole,
        root(&["k1", "k2"]),
        Box::new(TestScheme),
    )
    .expect("inoculated/whole with a non-empty root germinates");
    assert_eq!(p.mode(), InjectMode::Inoculated);
    assert_eq!(p.grain(), EnforcementGrain::Whole);
    assert_eq!(p.trust_root().len(), 2);

    // Loose default: empty root, explicit (RFC-0038 §7.1).
    let loose = InjectPolicy::loose();
    assert_eq!(loose.mode(), InjectMode::Loose);
    assert!(loose.trust_root().is_empty());
}

// ─── the verify path (DN-77 §4 items 2/4) ─────────────────────────────────────

/// A trusted signer with a good signature verifies; an untrusted signer and a tampered
/// signature refuse — each with the distinct, inspectable reason (G2).
/// Guarantee: `Proven` for the dispatch into the three arms (direct construction); the test
/// scheme's binding is `Empirical` (property test below); unforgeability is NOT claimed (M-836).
#[test]
fn verify_admits_trusted_and_refuses_untrusted_or_mismatched() {
    let p = InjectPolicy::germinate(
        InjectMode::Inoculated,
        EnforcementGrain::Whole,
        root(&["k1"]),
        Box::new(TestScheme),
    )
    .unwrap();
    let msg = b"blake3:abc\x00attest";
    let good = TestScheme.sign(&signer("k1"), msg);

    assert_eq!(p.verify(&signer("k1"), msg, &good), Ok(()));
    // Untrusted signer — even with a self-consistent signature.
    let foreign = TestScheme.sign(&signer("mallory"), msg);
    assert_eq!(
        p.verify(&signer("mallory"), msg, &foreign),
        Err(VerifyRefusal::UntrustedSigner(signer("mallory")))
    );
    // Trusted signer, tampered signature.
    let mut bad = good.clone();
    bad[0] = bad[0].wrapping_add(1);
    assert_eq!(
        p.verify(&signer("k1"), msg, &bad),
        Err(VerifyRefusal::SignatureMismatch(signer("k1")))
    );
    // Trusted signer, signature over different content.
    let other = TestScheme.sign(&signer("k1"), b"blake3:other\x00attest");
    assert_eq!(
        p.verify(&signer("k1"), msg, &other),
        Err(VerifyRefusal::SignatureMismatch(signer("k1")))
    );
}

// ─── property tests: the verify-path bounds (one per bound — house rule) ─────

proptest::proptest! {
    /// **Bound: signature binding.** For ANY (signer, message), the test scheme's sign→verify
    /// round-trips, and mutating any single byte of the message or the signature makes verify
    /// refuse — a tampered artifact can never verify as the original. Guarantee: `Empirical`
    /// (trials); the scheme's unforgeability is explicitly NOT claimed (`Declared` — M-836).
    #[test]
    fn test_scheme_binding_is_tamper_sensitive(
        name in "[a-z0-9]{1,16}",
        msg in proptest::collection::vec(proptest::num::u8::ANY, 1..128),
        idx in 0usize..128,
    ) {
        let s = signer(&name);
        let sig = TestScheme.sign(&s, &msg);
        proptest::prop_assert!(SignatureScheme::verify(&TestScheme, &s, &msg, &sig));
        // Message tamper.
        let mut m2 = msg.clone();
        let i = idx % m2.len();
        m2[i] = m2[i].wrapping_add(1);
        proptest::prop_assert!(!SignatureScheme::verify(&TestScheme, &s, &m2, &sig));
        // Signature tamper.
        let mut s2 = sig.clone();
        let j = idx % s2.len();
        s2[j] = s2[j].wrapping_add(1);
        proptest::prop_assert!(!SignatureScheme::verify(&TestScheme, &s, &msg, &s2));
    }

    /// **Bound: no admission outside the TrustRoot.** For ANY root and ANY signer not in it,
    /// `verify` refuses with `UntrustedSigner` regardless of the signature bytes — trust is
    /// never inherited or assumed (RFC-0038 I4; G2). Guarantee: `Empirical` (trials).
    #[test]
    fn no_signer_outside_the_root_is_ever_admitted(
        trusted in proptest::collection::btree_set("[a-m][a-z0-9]{0,8}", 0..5),
        outsider in "[n-z][a-z0-9]{0,8}",
        msg in proptest::collection::vec(proptest::num::u8::ANY, 0..64),
        sig in proptest::collection::vec(proptest::num::u8::ANY, 0..16),
    ) {
        // Outsider names start [n-z], trusted start [a-m] — disjoint by construction.
        let tr = TrustRoot::new(trusted.iter().map(|n| signer(n)));
        let p = if tr.is_empty() {
            InjectPolicy::loose()
        } else {
            InjectPolicy::germinate(
                InjectMode::Inoculated, EnforcementGrain::Whole, tr, Box::new(TestScheme),
            ).unwrap()
        };
        proptest::prop_assert_eq!(
            p.verify(&signer(&outsider), &msg, &sig),
            Err(VerifyRefusal::UntrustedSigner(signer(&outsider)))
        );
    }
}

// ─── the default-plus-deviations manifest (DN-77 §4 item 7) ──────────────────

/// The manifest renders the declared default plus every enumerated deviation — and says so
/// explicitly when there are none (never silent either way; RFC-0038 §8.5 / G2).
/// Guarantee: `Proven` for the rendering (direct construction).
#[test]
fn manifest_renders_default_plus_deviations() {
    let base = PolicyManifest {
        mode: InjectMode::Loose,
        grain: EnforcementGrain::Whole,
        scheme: TestScheme.name().to_owned(),
        trusted_signers: vec![signer("k1")],
        deviations: vec![],
    };
    let none = base.explain();
    assert!(none.contains("default: loose / grain whole"), "{none}");
    assert!(none.contains("deviations: none"), "{none}");
    assert!(none.contains("k1"), "{none}");
    assert!(
        none.contains("NOT production crypto"),
        "the scheme's honesty label is on the surface (VR-5): {none}"
    );

    let with = PolicyManifest {
        deviations: vec![PolicyDeviation {
            site: "blake3:abc".to_owned(),
            posture: "signed-and-verified (signer k1)".to_owned(),
            why: "per-inject signing opt-in in a loose context (RFC-0038 §8.7)".to_owned(),
        }],
        ..base
    }
    .explain();
    assert!(with.contains("deviations (1):"), "{with}");
    assert!(with.contains("blake3:abc"), "{with}");
    assert!(with.contains("§8.7"), "{with}");
}

/// The declared digest is deterministic and input-sensitive (the binding the cert message
/// construction relies on). Guarantee: `Empirical`; explicitly non-cryptographic (`Declared`).
#[test]
fn declared_digest_is_deterministic_and_input_sensitive() {
    assert_eq!(declared_digest64(b"a"), declared_digest64(b"a"));
    assert_ne!(declared_digest64(b"a"), declared_digest64(b"b"));
}
