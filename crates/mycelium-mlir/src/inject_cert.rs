//! `InjectCert` — the inject gate's certificate (M-961; RFC-0038 §6.2; DN-77 §4 item 2).
//!
//! The cert binds the **dispatch key itself** (the definition's [`ContentHash`], ADR-003/016/017)
//! to a signing authority and to the VR-4 no-opaque-lowering attestation (DN-18/M-630
//! [`CrossBackendGate`]): the signature is over `content_hash ‖ vr4_attestation_digest`, so the
//! cert asserts "from an authorized party **and** the lowering is auditable" in one artifact
//! (RFC-0038 §6.2 — security and transparency fused).
//!
//! **Honesty / scope (VR-5, G2).**
//! * `issued_at` is carried as the **§L placeholder** it is declared to be — recorded, **not
//!   enforced**: the replay/expiry mechanism is open R&D (**M-837**); the replay gap stays named
//!   and disclosed (RFC-0038 §5.2 / DN-44 §1.1), never silently shipped as closed.
//! * The attestation digest and the [`TestScheme`](crate::inject_gate::TestScheme)
//!   signature are **deterministic but non-cryptographic** (`Declared`) — the production cipher
//!   is M-836 R&D behind the `SignatureScheme` seam, and the canonical wire serialization the
//!   digest should cover is the ADR-013 wire format (**M-839**). Phase I digests the gate's
//!   EXPLAIN rendering (its auditable record).
//! * Production cert **emission** (`myc-prepare` signed spores) is **M-839/M-836** R&D; the
//!   [`InjectCert::issue_with_test_scheme`] constructor is the dev/test issuing path and says so.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

use crate::inject_gate::{declared_digest64, SignerId, TestScheme};
use mycelium_core::ContentHash;

use crate::vr4::CrossBackendGate;

/// The inject certificate (RFC-0038 §6.2) — the spore's signature component (ADR-013 §2
/// component 4; the wire format rides M-839).
#[derive(Debug, Clone)]
pub struct InjectCert {
    /// The dispatch key the cert authorizes (ADR-003/016/017) — the signature is over this key.
    content_hash: ContentHash,
    /// The signing authority's public-key fingerprint.
    signer: SignerId,
    /// The signature over `content_hash ‖ vr4_attestation_digest`.
    signature: Vec<u8>,
    /// The DN-18/M-630 no-opaque-lowering attestation carried inside the cert (RFC-0038 §6.2 —
    /// the cert asserts auditable lowering, not just authorship).
    vr4_attestation: CrossBackendGate,
    /// **§L placeholder** (RFC-0038 §6.2/§L): carried, **not enforced** — replay/expiry is open
    /// R&D (M-837). Seconds since the Unix epoch by convention; no currency check exists yet.
    issued_at: u64,
}

/// The signed message for a cert over `hash` with attestation `vr4`:
/// `content_hash ‖ 0x00 ‖ digest(vr4)` (RFC-0038 §6.2).
///
/// The gate constructs this from the **actual** hash of the code being admitted (never from the
/// cert's claimed hash), so a cert minted for different content fails signature verification
/// naturally — no secondary identity can drift from the dispatch key.
///
/// The digest is [`declared_digest64`] over the gate's EXPLAIN rendering — deterministic,
/// **non-cryptographic** (`Declared`; production digest/serialization ride M-836/M-839).
#[must_use]
pub fn signed_message(hash: &ContentHash, vr4: &CrossBackendGate) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(hash.as_str().as_bytes());
    msg.push(0x00);
    msg.extend_from_slice(&declared_digest64(vr4.explain().as_bytes()));
    msg
}

impl InjectCert {
    /// Issue a cert under the **deterministic test scheme** — the dev/test issuing path
    /// (DN-77 §3.3 B-1). **Not production signing** (VR-5): `TestScheme` is binding-only and
    /// forgeable by construction; `myc-prepare` production emission with a real cipher is
    /// M-839/M-836 R&D. `issued_at` is recorded, not enforced (§L placeholder — M-837).
    #[must_use]
    pub fn issue_with_test_scheme(
        signer: SignerId,
        content_hash: ContentHash,
        vr4_attestation: CrossBackendGate,
        issued_at: u64,
    ) -> Self {
        let signature = TestScheme.sign(&signer, &signed_message(&content_hash, &vr4_attestation));
        InjectCert {
            content_hash,
            signer,
            signature,
            vr4_attestation,
            issued_at,
        }
    }

    /// Assemble a cert from its parts — how a cert arrives as *data* (e.g. parsed from a spore's
    /// signature component; the wire format rides M-839). No validation happens here: a cert is
    /// only ever *trusted* by passing the image's verify gate (RFC-0038 §5.1) — construction is
    /// not admission.
    #[must_use]
    pub fn from_parts(
        signer: SignerId,
        content_hash: ContentHash,
        signature: Vec<u8>,
        vr4_attestation: CrossBackendGate,
        issued_at: u64,
    ) -> Self {
        InjectCert {
            content_hash,
            signer,
            signature,
            vr4_attestation,
            issued_at,
        }
    }

    /// The dispatch key the cert claims to authorize.
    #[must_use]
    pub fn content_hash(&self) -> &ContentHash {
        &self.content_hash
    }
    /// The signing authority.
    #[must_use]
    pub fn signer(&self) -> &SignerId {
        &self.signer
    }
    /// The signature bytes.
    #[must_use]
    pub fn signature(&self) -> &[u8] {
        &self.signature
    }
    /// The carried VR-4 attestation.
    #[must_use]
    pub fn vr4(&self) -> &CrossBackendGate {
        &self.vr4_attestation
    }
    /// The §L placeholder issue time — carried, **not enforced** (M-837; RFC-0038 §L).
    #[must_use]
    pub fn issued_at(&self) -> u64 {
        self.issued_at
    }

    /// The cert's `EXPLAIN` (no black box): what it binds, who signed, and — honestly — which
    /// parts are placeholders (G2/VR-5).
    #[must_use]
    pub fn explain(&self) -> String {
        format!(
            "InjectCert for {} — signer {} — issued_at {} (carried, NOT enforced: replay/expiry \
             is open R&D, M-837/RFC-0038 §L) — VR-4 attestation: {} of {} backend stage(s) dumped",
            self.content_hash.as_str(),
            self.signer,
            self.issued_at,
            self.vr4_attestation.covered(),
            self.vr4_attestation.stages.len(),
        )
    }
}
