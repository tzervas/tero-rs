//! Inject-mode gate core (M-961; RFC-0038 §4/§6–§8; DN-77 §4 — the confirmed Phase-I subset).
//!
//! This module is the **policy vocabulary + verify seam** of the inject-security axis: the
//! `loose`/`inoculated` modes (RFC-0038 §4.2), the [`TrustRoot`] (§7.1), the [`SignatureScheme`]
//! verify seam (DN-77 §3.3 B-1), the enforcement-grain knob (§8.4), the never-silent policy
//! refusals, and the default-plus-deviations [`PolicyManifest`] (§8.5, Phase-I slice per DN-77
//! §4 item 7). The *gate itself* — where a program is loaded/verified — is the dispatch
//! boundary ([`Image`](crate::inject::Image), the DN-77 §2 insertion point); the policy core
//! lives beside it in this crate so the security surface is auditable in one small place (KC-3).
//!
//! **Placement (DN-68).** DN-77 §2 named `mycelium-sec` "a plausible home for verify helpers,
//! not a prebuilt one" — but `mycelium-sec` is a **tools-tier** crate (`xtask/deps-strata.toml`)
//! and the gate's consumer is this **core-tier** crate, so housing the mechanism there would
//! create a `core → tools` edge the DN-68 `no-upward-tier-edges` named rule forbids (checked:
//! `cargo run -p xtask -- deps`). Per that rule's own precedent (M-883/M-884 — the seam moves to
//! the lower tier), the policy core lands here, importable downward by the security tooling.
//!
//! **Honesty / scope (VR-5, G2).** What is built here is exactly the DN-77 §4 subset:
//!
//! * Mode gating (`Loose`/`Inoculated`), `whole`-grain default — **Enacted** by M-961's landed
//!   code + tests.
//! * The verify seam + the deterministic **test scheme** — the *gating mechanism* is Enacted;
//!   **production-grade signing stays `Declared`**: [`TestScheme`] provides *binding only*
//!   (signature ↔ (signer, message)), **zero unforgeability** — the production cipher and the
//!   whole key-management story are open R&D (M-836, RFC-0038 §K.2), and the seam exists so the
//!   production scheme plugs in without gate changes (DN-77 §3.3 B-1).
//! * `module`/`call` enforcement grains are **not enforced** — selecting one refuses
//!   never-silently ([`PolicyError::GrainNotYetEnforced`], the DN-63 pattern); the build-out is
//!   tracked by **M-847** (RFC-0038 §8.4–§8.7). Never a silent downgrade to `whole`.
//! * Replay/expiry (RFC-0038 §L, M-837), the scoping-hierarchy config surface (§M, M-838),
//!   `myc-prepare` signed-spore emission (M-839), the cross-colony mesh flow (M-842), and the
//!   colony trust topology (§8.8, M-849) all stay `Declared` — none is closed here (G2).
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

/// A signing authority's public-key **fingerprint** (RFC-0038 §6.2 `signer`).
///
/// Phase-I this is an opaque string fingerprint; the concrete key format/derivation is part of
/// the M-836 key-management R&D (`Declared`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignerId(String);

impl SignerId {
    /// A signer id from its fingerprint text.
    pub fn new(fingerprint: impl Into<String>) -> Self {
        SignerId(fingerprint.into())
    }
    /// The fingerprint text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SignerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The set of trusted [`SignerId`]s associated with an image (RFC-0038 §7.1).
///
/// Constructed once (at germination) and never mutated through this type — immutability *after
/// germination* (I3) is enforced at the image boundary, which holds the `TrustRoot` privately
/// and refuses runtime change with an explicit error (never a silent downgrade; G2).
/// An **empty** `TrustRoot` means `loose` mode (§7.1) — explicit and inspectable, never silent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrustRoot {
    signers: BTreeSet<SignerId>,
}

impl TrustRoot {
    /// An empty trust root (⇒ `loose` mode, RFC-0038 §7.1).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
    /// A trust root over the given signers (deduplicated, order-independent).
    pub fn new(signers: impl IntoIterator<Item = SignerId>) -> Self {
        TrustRoot {
            signers: signers.into_iter().collect(),
        }
    }
    /// Whether the root trusts `signer`.
    #[must_use]
    pub fn trusts(&self, signer: &SignerId) -> bool {
        self.signers.contains(signer)
    }
    /// Whether the root is empty (⇒ `loose` mode, §7.1).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signers.is_empty()
    }
    /// How many signers are trusted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.signers.len()
    }
    /// The trusted signers, in deterministic (sorted) order — for the EXPLAIN surface.
    pub fn signers(&self) -> impl Iterator<Item = &SignerId> {
        self.signers.iter()
    }
}

/// The two first-class inject modes (RFC-0038 §4.2; DN-77 §4 item 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectMode {
    /// Local-dev: unsigned injection permitted; every unsigned admission is G2-tagged
    /// ([`Admission::Unsigned`]) so unsigned code is **never silent** (I1).
    Loose,
    /// Production: injection requires a valid `InjectCert` from a signer in the image's
    /// [`TrustRoot`]; unsigned code is an explicit refusal (I2), on **both** the compiled and
    /// interpreter-fallback paths (I6).
    Inoculated,
}

impl InjectMode {
    /// The canonical label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            InjectMode::Loose => "loose",
            InjectMode::Inoculated => "inoculated",
        }
    }
}

impl fmt::Display for InjectMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The enforcement-granularity knob (RFC-0038 §8.4) — **at what grain** the signing requirement
/// is checked, orthogonal to the mode.
///
/// Phase I enforces **`Whole`** only (the `inoculated` application default, §8.6): the
/// application signature is checked once at load; its calls are then trusted — declared,
/// mode-tagged, EXPLAIN-able, never a hidden weakening. `Module`/`Call` are the *knob* only:
/// selecting them is a never-silent [`PolicyError::GrainNotYetEnforced`] refusal (DN-63
/// pattern); their enforcement paths are **M-847**'s tracked scope (`Declared`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementGrain {
    /// The full deployed application/spore signature is checked once at compile/load time.
    Whole,
    /// Per-phylum/per-nodule checking — **not yet enforced** (M-847).
    Module,
    /// Per-dispatch re-verification — **not yet enforced** (M-847).
    Call,
}

impl EnforcementGrain {
    /// The canonical label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            EnforcementGrain::Whole => "whole",
            EnforcementGrain::Module => "module",
            EnforcementGrain::Call => "call",
        }
    }
}

impl fmt::Display for EnforcementGrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How an admitted definition entered the image — the per-entry G2 tag carried on every
/// dispatch decision (RFC-0038 I1/I7; surfaced on `Resolution` at the image boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Admitted unsigned (only possible in `loose` mode) — the I1 never-silent unsigned tag.
    Unsigned,
    /// Admitted with a verified `InjectCert` from this trusted signer.
    Verified {
        /// The trusted signer whose signature verified.
        signer: SignerId,
    },
}

impl fmt::Display for Admission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Admission::Unsigned => f.write_str("unsigned (G2-tagged)"),
            Admission::Verified { signer } => write!(f, "signed-and-verified (signer {signer})"),
        }
    }
}

/// The signature-verification **seam** (DN-77 §3.3 B-1): the gate verifies through this
/// interface; the production cipher plugs in here when the M-836 key-management R&D resolves —
/// no gate code changes.
///
/// A scheme is a stateless verifier: given a signer's public fingerprint, a message, and a
/// signature, it answers whether the signature verifies. **No implementation shipped in Phase I
/// is production cryptography** (see [`TestScheme`]); that claim stays `Declared` (VR-5).
pub trait SignatureScheme: Send + Sync {
    /// The scheme's name, for the EXPLAIN/manifest surface (never a hidden choice — G2).
    fn name(&self) -> &'static str;
    /// Whether `signature` verifies for (`signer`, `message`) under this scheme.
    fn verify(&self, signer: &SignerId, message: &[u8], signature: &[u8]) -> bool;
}

/// FNV-1a 64-bit — the deterministic, dependency-free digest under [`TestScheme`] and
/// [`declared_digest64`]. **Not cryptographic** (`Declared` — see those items' honesty notes).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A deterministic 8-byte digest for Phase-I attestation binding (`Declared` — placeholder).
///
/// **Honesty (VR-5):** FNV-1a is **not** a cryptographic hash — this digest provides
/// deterministic *binding* for the conformance suite, not collision resistance. The production
/// digest rides the M-836 cipher decision, and the canonical serialization it digests rides the
/// ADR-013 wire format (M-839). Flagged, never silent (G2).
#[must_use]
pub fn declared_digest64(bytes: &[u8]) -> [u8; 8] {
    fnv1a64(bytes).to_be_bytes()
}

/// The **deterministic test scheme** (DN-77 §3.3 B-1) — exercises the gate mechanism for the
/// conformance suite.
///
/// **Honesty (VR-5, loud):** this scheme provides *binding only* — a signature is a keyed
/// recomputable digest of (signer, message), so tampering with either is detected — but it has
/// **zero unforgeability**: anyone who knows the (public) signer fingerprint can mint a valid
/// signature. It is **NOT** production cryptography and is named accordingly on every
/// EXPLAIN/manifest surface. The production cipher + key management are open R&D (**M-836**,
/// RFC-0038 §K.2) and stay `Declared`; the [`SignatureScheme`] seam exists so that decision
/// plugs in without gate changes.
#[derive(Debug, Clone, Copy, Default)]
pub struct TestScheme;

impl TestScheme {
    const DOMAIN: &'static [u8] = b"myc-inject-test-scheme-v0";

    /// Sign `message` as `signer` under the test scheme (deterministic; **forgeable by
    /// construction** — see the type-level honesty note). The dev/test issuing path only;
    /// production signing (`myc-prepare`) is M-839/M-836 R&D.
    #[must_use]
    pub fn sign(&self, signer: &SignerId, message: &[u8]) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(Self::DOMAIN.len() + signer.as_str().len() + 1 + message.len());
        buf.extend_from_slice(Self::DOMAIN);
        buf.extend_from_slice(signer.as_str().as_bytes());
        buf.push(0x1f);
        buf.extend_from_slice(message);
        fnv1a64(&buf).to_be_bytes().to_vec()
    }
}

impl SignatureScheme for TestScheme {
    fn name(&self) -> &'static str {
        "test-fnv1a64 (Declared: binding only, NOT production crypto — M-836)"
    }
    fn verify(&self, signer: &SignerId, message: &[u8], signature: &[u8]) -> bool {
        self.sign(signer, message) == signature
    }
}

/// Why a presented certificate was refused by [`InjectPolicy::verify`] — always explicit (G2).
///
/// Both cases surface at the image boundary as the RFC-0038 §5.1 `BadSignature` refusal (a cert
/// whose signature does not verify against a trusted key — wrong/untrusted signer); the two
/// inner reasons stay distinct here so the refusal is inspectable, never a folded mystery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyRefusal {
    /// The cert's signer is not in the image's [`TrustRoot`] (RFC-0038 §7.2 — own-root rule).
    UntrustedSigner(SignerId),
    /// The signer is trusted but the signature does not verify over the message.
    SignatureMismatch(SignerId),
}

impl fmt::Display for VerifyRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyRefusal::UntrustedSigner(s) => write!(
                f,
                "signer {s} is not in this image's TrustRoot — trust is never inherited or \
                 assumed (RFC-0038 I4/§7.2; G2)"
            ),
            VerifyRefusal::SignatureMismatch(s) => write!(
                f,
                "signature from trusted signer {s} does not verify over the presented content — \
                 bad, untrusted code is blocked, never admitted (RFC-0038 §8.7; G2)"
            ),
        }
    }
}

impl std::error::Error for VerifyRefusal {}

/// Why a policy could not be germinated — always explicit, never a silent downgrade (G2/I3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// `inoculated` was requested with an **empty** [`TrustRoot`]. An empty root means `loose`
    /// mode (RFC-0038 §7.1); silently downgrading a requested `inoculated` posture to `loose`
    /// would be a hidden weakening, so germination refuses instead (G2/I3).
    EmptyTrustRoot,
    /// A `module`/`call` enforcement grain was selected but its enforcement path is not yet
    /// built (Phase I enforces `whole` only). Refused never-silently rather than silently
    /// running at a different grain (DN-63 pattern); the build-out is **M-847** (RFC-0038
    /// §8.4–§8.7).
    GrainNotYetEnforced(EnforcementGrain),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyError::EmptyTrustRoot => f.write_str(
                "inoculated mode requires a non-empty TrustRoot — an empty TrustRoot means loose \
                 mode (RFC-0038 §7.1); refusing germination rather than silently downgrading the \
                 requested posture (G2/I3)",
            ),
            PolicyError::GrainNotYetEnforced(g) => write!(
                f,
                "enforcement grain `{g}` is not yet enforced — Phase I builds the `whole` grain; \
                 the `module`/`call` enforcement paths are tracked R&D (M-847; RFC-0038 §8.4). \
                 Refused explicitly, never silently run at a different grain (G2)"
            ),
        }
    }
}

impl std::error::Error for PolicyError {}

/// The image's inject-security **policy**: mode × grain × trust root × scheme, fixed at
/// germination (RFC-0038 §7.1 I3 — the image holds it immutably; there is no mutation API).
pub struct InjectPolicy {
    mode: InjectMode,
    grain: EnforcementGrain,
    trust_root: TrustRoot,
    scheme: Box<dyn SignatureScheme>,
}

impl fmt::Debug for InjectPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InjectPolicy")
            .field("mode", &self.mode)
            .field("grain", &self.grain)
            .field("trust_root", &self.trust_root)
            .field("scheme", &self.scheme.name())
            .finish()
    }
}

impl InjectPolicy {
    /// Germinate a policy — the one validation point (RFC-0038 §7.1; DN-77 §4 items 1/3/6).
    ///
    /// # Errors
    /// * [`PolicyError::EmptyTrustRoot`] — `inoculated` with an empty root (a silent `loose`
    ///   downgrade is forbidden; G2/I3).
    /// * [`PolicyError::GrainNotYetEnforced`] — `module`/`call` grain selected (Phase I enforces
    ///   `whole` only; M-847).
    pub fn germinate(
        mode: InjectMode,
        grain: EnforcementGrain,
        trust_root: TrustRoot,
        scheme: Box<dyn SignatureScheme>,
    ) -> Result<Self, PolicyError> {
        if !matches!(grain, EnforcementGrain::Whole) {
            return Err(PolicyError::GrainNotYetEnforced(grain));
        }
        if mode == InjectMode::Inoculated && trust_root.is_empty() {
            return Err(PolicyError::EmptyTrustRoot);
        }
        Ok(InjectPolicy {
            mode,
            grain,
            trust_root,
            scheme,
        })
    }

    /// The development default: `loose`, `whole` grain, empty trust root (⇒ `loose`, RFC-0038
    /// §7.1 — explicit here, never inferred silently), test scheme.
    #[must_use]
    pub fn loose() -> Self {
        InjectPolicy {
            mode: InjectMode::Loose,
            grain: EnforcementGrain::Whole,
            trust_root: TrustRoot::empty(),
            scheme: Box::new(TestScheme),
        }
    }

    /// The policy's mode.
    #[must_use]
    pub fn mode(&self) -> InjectMode {
        self.mode
    }
    /// The policy's enforcement grain (`whole` in Phase I — see [`Self::germinate`]).
    #[must_use]
    pub fn grain(&self) -> EnforcementGrain {
        self.grain
    }
    /// The trust root (read-only — I3 immutability is preserved by the absence of mutators).
    #[must_use]
    pub fn trust_root(&self) -> &TrustRoot {
        &self.trust_root
    }
    /// The active scheme's name (for EXPLAIN — never a hidden choice, G2).
    #[must_use]
    pub fn scheme_name(&self) -> &'static str {
        self.scheme.name()
    }

    /// Verify a presented signature for (`signer`, `message`) against **this image's own**
    /// trust root (RFC-0038 I4 — trust is never inherited) through the scheme seam.
    ///
    /// # Errors
    /// [`VerifyRefusal::UntrustedSigner`] if `signer` is not in the root;
    /// [`VerifyRefusal::SignatureMismatch`] if the signature does not verify. Both are explicit
    /// refusals — a failed verify never admits (G2).
    pub fn verify(
        &self,
        signer: &SignerId,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), VerifyRefusal> {
        if !self.trust_root.trusts(signer) {
            return Err(VerifyRefusal::UntrustedSigner(signer.clone()));
        }
        if !self.scheme.verify(signer, message, signature) {
            return Err(VerifyRefusal::SignatureMismatch(signer.clone()));
        }
        Ok(())
    }
}

/// One enumerated departure from the declared default posture (RFC-0038 §8.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDeviation {
    /// The deviating site (Phase I: the content hash of a per-inject override; the full
    /// nodule/function/line site vocabulary rides the §M config surface — M-838, `Declared`).
    pub site: String,
    /// The posture at that site.
    pub posture: String,
    /// Why the site deviates (author-facing; never silent — G2).
    pub why: String,
}

/// The **default-plus-deviations manifest** (RFC-0038 §8.5; DN-77 §4 item 7 — the Phase-I
/// slice): the effective policy rendered as a declared default plus enumerated deviations,
/// surfaced via EXPLAIN. No site's posture is silent or surprising (G2).
///
/// **Scope (VR-5):** Phase I renders the project-level default with per-inject overrides
/// (§8.7's per-inject signing in an otherwise-`loose` context). The full seven-level scope
/// hierarchy + config surface is §M R&D (**M-838**), and the wider granularity/scope-resolution
/// system is **M-847**'s tracked scope — this slice *coordinates with*, and does not duplicate,
/// those (DN-77 §5/§6 F-1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyManifest {
    /// The declared default mode.
    pub mode: InjectMode,
    /// The declared enforcement grain.
    pub grain: EnforcementGrain,
    /// The active signature scheme's name (incl. its honesty label).
    pub scheme: String,
    /// The trusted signers (deterministic order).
    pub trusted_signers: Vec<SignerId>,
    /// Every site that departs from the default posture — enumerated, never elided.
    pub deviations: Vec<PolicyDeviation>,
}

impl PolicyManifest {
    /// Render the manifest for the EXPLAIN channel (no black box): the declared default plus
    /// every enumerated deviation.
    #[must_use]
    pub fn explain(&self) -> String {
        let mut out = format!(
            "inject policy default: {} / grain {} — scheme: {} — trust root: {} signer(s)",
            self.mode,
            self.grain,
            self.scheme,
            self.trusted_signers.len()
        );
        for s in &self.trusted_signers {
            out.push_str(&format!("\n  trusted signer: {s}"));
        }
        if self.deviations.is_empty() {
            out.push_str("\ndeviations: none — every site runs the declared default");
        } else {
            out.push_str(&format!("\ndeviations ({}):", self.deviations.len()));
            for d in &self.deviations {
                out.push_str(&format!("\n  {}: {} — {}", d.site, d.posture, d.why));
            }
        }
        out
    }
}
