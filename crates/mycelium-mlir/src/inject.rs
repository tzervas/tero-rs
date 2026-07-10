//! In-process **hot-inject** prototype (M-341; ADR-017; ADR-016's call ABI; RFC-0004 §9 the
//! interpreted↔compiled continuum; phase-4).
//!
//! This is the *named first build step* of ADR-017, realized on the M-340 `dlopen` JIT (the
//! prototype substrate). It builds the three pieces ADR-016/017 specify:
//!
//! 1. **A hash-keyed dispatch table** ([`Image`]) — the running image holds a `ContentHash → entry`
//!    map (ADR-016's call ABI: a compiled stable component is invoked by the *content hash of the
//!    definition* it compiles). A [`call`](Image::call) **resolves to a compiled entry if present,
//!    else interprets** the definition (the continuum, RFC-0004 §9.1). A hash with neither a
//!    compiled entry nor an interpretable definition is an **explicit** [`InjectError::DispatchMiss`]
//!    — never a silent guess (G2/SC-3; ADR-017 decision 5).
//! 2. **Load-and-register injection** ([`inject`](Image::inject)) — "injecting" a recompiled
//!    definition compiles its unit (the `dlopen` JIT) and registers a *new* `hash → entry`. It
//!    **never mutates a live entry**: because identity *is* the content hash (ADR-003), a re-inject
//!    of the same definition is the same code under the same key (publish-once, idempotent), and an
//!    *edited* definition is a **new hash under a new entry** (ADR-017 decision 4). The atomicity
//!    hazard dissolves: an in-flight call to the old hash finishes on the old (still-loaded) code,
//!    while a new caller — referencing the new hash — dispatches to the new entry.
//! 3. **The recompile set by hash reachability** ([`recompile_closure`]) — editing a definition
//!    yields a new hash; its transitive *dependents* (which named the old hash) get new hashes too;
//!    everything else keeps its hash *and its already-compiled entry*. So the recompile set is
//!    **exactly the changed dependency-closure**, computed by reachability over the dependency graph
//!    — no AST/file diff (ADR-017 decision 3).
//!
//! **The inject-mode security gate (M-961; RFC-0038; DN-77 §4 — the confirmed Phase-I subset).**
//! On top of the dispatch mechanism, the image carries an
//! [`InjectPolicy`](mycelium_sec::inject_gate::InjectPolicy) fixed at **germination**
//! ([`Image::germinate`]): in **`loose`** mode unsigned registration is permitted and every
//! unsigned admission is G2-tagged ([`Admission::Unsigned`] on every [`Resolution`] — I1/I7); in
//! **`inoculated`** mode unsigned registration is an explicit
//! [`InjectError::UnsignedCode`] refusal — on **both** the compiled path
//! ([`inject`](Image::inject)) and the interpreter-fallback path ([`define`](Image::define)),
//! uniformly (I6). A presented [`InjectCert`] is verified against the image's **own**
//! [`TrustRoot`] (I4) through the `SignatureScheme` seam; a wrong/untrusted signer is an explicit
//! [`InjectError::BadSignature`] (§5.1/§8.7) — presented-but-bad certs are **blocked even in
//! `loose` mode**, never downgraded to unsigned-permitted. The `TrustRoot` is **immutable after
//! germination** (I3): [`Image::set_trust_root`] exists only to refuse explicitly. The effective
//! posture is EXPLAIN-able as a **default-plus-deviations manifest**
//! ([`Image::policy_manifest`], §8.5 Phase-I slice).
//!
//! **Honesty / gate scope (VR-5, G2).** What the gate enacts is exactly the DN-77 §4 subset:
//! mode gating, the `TrustRoot` verify flow, the two never-silent refusals, the `whole`
//! enforcement grain (load-time check; calls then trusted — §8.4/§8.6; `module`/`call` grains
//! refuse never-silently at germination, tracked by M-847), and the Phase-I manifest slice.
//! Production signing/key-management (M-836), replay/expiry (`issued_at` carried, not enforced —
//! M-837), the scoping config surface (M-838), `myc-prepare` emission (M-839), cross-colony mesh
//! flow (M-842), and colony trust topology (M-849) stay **`Declared`** — open, flagged, not
//! closed here.
//!
//! **Verification (NFR-7).** The injected-compiled path is checked **observationally equivalent** to
//! the reference interpreter through the shared M-210 TV checker (`mycelium_cert::check`,
//! `ObservationalEquiv`) — the same checker that validates swaps and the interp↔AOT differential.
//! See `tests/inject_hotswap.rs`.
//!
//! **Scope / honesty (VR-5).** This is the *in-process* proof. A "definition/unit" here is a
//! **closed** bit/trit-subset program (the JIT's domain today, M-340) and the call boundary is
//! ADR-016's call ABI **restricted to nullary units** — the args-carrying value boundary (the
//! RFC-0001 §4.8 wire form) lands with the MLIR→LLVM backend (RFC-0004 §2). Cross-process / native
//! units and the cross-process unit format (RFC-0004 §10 OQ-3) stay deferred. What is proven *now*:
//! hash-keyed dispatch, never-silent resolution, load-and-register injection without live-entry
//! mutation, the dependency-closure recompile set, and interp≡injected-compiled equivalence.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use crate::inject_gate::{
    Admission, InjectMode, InjectPolicy, PolicyDeviation, PolicyManifest, SignerId, TrustRoot,
};
use mycelium_core::{ContentHash, Node, Value};
use mycelium_interp::{EvalError, Interpreter};

use crate::inject_cert::{signed_message, InjectCert};
use crate::jit::{compile_so, JitArtifact};
use crate::llvm::AotError;

/// How a [`ContentHash`] resolves in an [`Image`] — the inspectable/`EXPLAIN`-able dispatch decision
/// (ADR-017 decision 5: which hash resolves to which entry is queryable). Never a hidden choice.
///
/// Per RFC-0038 §7.3 (I7), every resolution carries its **inject-mode dimension**: the image's
/// mode plus the entry's [`Admission`] tag (unsigned vs signed-and-verified), so the security
/// posture of every dispatch decision is inspectable through the same EXPLAIN channel as the
/// execution path (ADR-006/G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// A compiled (injected) entry exists for this hash — the call dispatches to native code.
    Compiled {
        /// The image's inject mode (RFC-0038 §4.2).
        inject_mode: InjectMode,
        /// How this entry was admitted (I1 — unsigned status is never silent).
        admission: Admission,
    },
    /// No compiled entry, but an interpretable definition is registered — the call interprets
    /// (the continuum, RFC-0004 §9.1).
    Interpreted {
        /// The image's inject mode (RFC-0038 §4.2).
        inject_mode: InjectMode,
        /// How this definition was admitted (I1 — unsigned status is never silent).
        admission: Admission,
    },
    /// Neither a compiled entry nor an interpretable definition — a call would be an explicit
    /// [`InjectError::DispatchMiss`], never a guess.
    Miss,
}

/// A failure at the dispatch/injection boundary — every variant is explicit (never a silent pass or
/// a partial registration; G2/SC-3; ADR-017 decision 5). (`PartialEq` but not `Eq`: the wrapped
/// `EvalError` is only `PartialEq`.)
#[derive(Debug, Clone, PartialEq)]
pub enum InjectError {
    /// A call to a hash with no compiled entry and no interpretable definition.
    DispatchMiss(ContentHash),
    /// Compiling/loading the unit failed — no entry is registered (never a partial registration).
    /// Carries the underlying [`AotError`] (incl. a skippable `ToolchainMissing` when `clang` is
    /// absent, so callers can degrade to the interpreter rather than fail).
    Compile(AotError),
    /// The interpreter fallback refused the definition (an explicit `EvalError`, surfaced).
    Interp(EvalError),
    /// **Unsigned code refused** (RFC-0038 I2; M-961): an unsigned registration was attempted on
    /// an `inoculated` image — on either path (compiled *or* interpreter fallback, I6). Carries
    /// the exact rejected hash; nothing is registered (never a partial admission).
    UnsignedCode(ContentHash),
    /// **Bad/untrusted signature refused** (RFC-0038 §5.1/§8.7; M-961): a presented [`InjectCert`]
    /// did not verify against this image's **own** [`TrustRoot`] (I4) — the signer is untrusted,
    /// or the signature does not cover this exact content (the message is built from the *actual*
    /// hash being admitted, so a cert minted for other content fails here too). Blocked in every
    /// mode — presented-but-bad certs are bad, untrusted code even in `loose` (§8.7). Carries the
    /// exact rejected hash and the claimed signer; nothing is registered.
    BadSignature(ContentHash, SignerId),
    /// **`TrustRoot` is immutable after germination** (RFC-0038 §7.1 I3): a runtime change was
    /// attempted and is refused explicitly — never a silent re-root/downgrade (G2).
    TrustRootImmutable,
}

impl std::fmt::Display for InjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InjectError::DispatchMiss(h) => {
                write!(
                    f,
                    "dispatch miss: no compiled entry or definition for {}",
                    h.as_str()
                )
            }
            InjectError::Compile(e) => write!(f, "unit compile/load failed: {e}"),
            InjectError::Interp(e) => write!(f, "interpreter fallback refused: {e}"),
            InjectError::UnsignedCode(h) => write!(
                f,
                "unsigned code refused: this image is inoculated — injection of {} requires a \
                 valid InjectCert from a signer in the image's TrustRoot, on both the compiled \
                 and interpreter paths (RFC-0038 I2/I6; G2 — explicit, never silent)",
                h.as_str()
            ),
            InjectError::BadSignature(h, s) => write!(
                f,
                "bad signature refused for {}: the cert from signer {s} does not verify against \
                 this image's own TrustRoot (wrong or untrusted signer, or a cert for different \
                 content) — bad, untrusted code is blocked in every mode (RFC-0038 §5.1/§8.7/I4; \
                 G2)",
                h.as_str()
            ),
            InjectError::TrustRootImmutable => f.write_str(
                "TrustRoot is immutable after germination — a runtime change is refused \
                 explicitly, never a silent re-root (RFC-0038 §7.1 I3; G2)",
            ),
        }
    }
}

impl std::error::Error for InjectError {}

/// The running **image**: a hash-keyed dispatch table over a compiled overlay + an interpretable
/// base (RFC-0004 §9 continuum), gated by an inject-security policy fixed at germination
/// (RFC-0038; M-961). Definitions are registered interpret-only with [`define`](Self::define)
/// (unsigned) or [`define_signed`](Self::define_signed); [`inject`](Self::inject) /
/// [`inject_signed`](Self::inject_signed) add a compiled entry on top. A [`call`](Self::call)
/// prefers the compiled entry, else interprets — never a silent miss. Under the Phase-I `whole`
/// enforcement grain the gate checks at **registration (load) time**; admitted entries' calls are
/// then trusted (§8.4/§8.6 — declared and EXPLAIN-able, never a hidden weakening; per-call
/// re-verification is the `call` grain, M-847).
pub struct Image {
    /// The interpretable base: every known definition, keyed by its content hash (the continuum's
    /// safe default — ADR-009).
    defs: HashMap<ContentHash, Node>,
    /// The compiled (injected) overlay: `ContentHash → entry`. Injection registers here; a key is
    /// **published once, never overwritten** (ADR-017 decision 4 — content-addressing guarantees a
    /// re-inject under the same key is the same code).
    compiled: HashMap<ContentHash, JitArtifact>,
    /// The trusted reference interpreter for the fallback path (ADR-007).
    interp: Interpreter,
    /// The inject-security policy — fixed at germination, **no mutation API** (I3).
    policy: InjectPolicy,
    /// Per-entry admission tags (I1/I7): how each admitted hash entered the image. An entry only
    /// enters through a gated path, so every key in `defs`/`compiled` has a tag; a missing tag
    /// floors to `Unsigned` (never over-claim `Verified` — VR-5).
    admissions: HashMap<ContentHash, Admission>,
}

impl Default for Image {
    /// The development default: a **`loose`** policy with an **empty `TrustRoot`** (⇒ `loose`,
    /// RFC-0038 §7.1 — explicit here, never inferred silently) over the default interpreter.
    fn default() -> Self {
        Image::germinate(Interpreter::default(), InjectPolicy::loose())
    }
}

impl Image {
    /// An empty image with the default reference interpreter and the `loose` development policy
    /// (empty `TrustRoot` ⇒ `loose`, RFC-0038 §7.1).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an image with a specific interpreter for the fallback path (e.g. one wired with a
    /// certified swap engine), under the `loose` development policy. The cert axis and the inject
    /// axis are orthogonal (RFC-0038 §4/I5): use [`germinate`](Self::germinate) to combine any
    /// interpreter with any inject policy.
    #[must_use]
    pub fn with_interpreter(interp: Interpreter) -> Self {
        Image::germinate(interp, InjectPolicy::loose())
    }

    /// **Germinate** an image with its inject-security policy (RFC-0038 §7.1: the `TrustRoot` is
    /// set at the image start event and is immutable thereafter — I3). The policy was already
    /// validated by `InjectPolicy::germinate` (mode/grain/root consistency, never-silent).
    #[must_use]
    pub fn germinate(interp: Interpreter, policy: InjectPolicy) -> Self {
        Image {
            defs: HashMap::new(),
            compiled: HashMap::new(),
            interp,
            policy,
            admissions: HashMap::new(),
        }
    }

    /// The image's inject mode (RFC-0038 §4.2) — inspectable, never hidden (G2).
    #[must_use]
    pub fn inject_mode(&self) -> InjectMode {
        self.policy.mode()
    }

    /// The image's trust root (read-only — I3).
    #[must_use]
    pub fn trust_root(&self) -> &TrustRoot {
        self.policy.trust_root()
    }

    /// **Refuses** — the `TrustRoot` is immutable after germination (RFC-0038 §7.1 I3). This
    /// method exists so the runtime-change attempt is an *explicit, testable* error rather than
    /// an absent API a caller might work around silently (G2; the RFC-0038 §13(d) conformance
    /// clause).
    ///
    /// # Errors
    /// Always [`InjectError::TrustRootImmutable`].
    pub fn set_trust_root(&mut self, _root: TrustRoot) -> Result<(), InjectError> {
        Err(InjectError::TrustRootImmutable)
    }

    /// Register a definition **interpret-only and unsigned** under its content hash (RFC-0001
    /// §4.6), returning the hash. Re-defining the same definition is idempotent (the same hash
    /// maps to the same node). This is the continuum's safe default: a definition runs
    /// interpreted until it is injected.
    ///
    /// **Gate (M-961):** on an `inoculated` image an unsigned registration is an explicit
    /// [`InjectError::UnsignedCode`] — the interpreter path is gated equally (RFC-0038 I6), no
    /// back door. In `loose` mode the admission is G2-tagged [`Admission::Unsigned`] (I1).
    ///
    /// # Errors
    /// [`InjectError::UnsignedCode`] in `inoculated` mode (nothing is registered).
    pub fn define(&mut self, node: Node) -> Result<ContentHash, InjectError> {
        let hash = node.content_hash();
        if self.policy.mode() == InjectMode::Inoculated {
            return Err(InjectError::UnsignedCode(hash));
        }
        self.defs.entry(hash.clone()).or_insert(node);
        self.admissions
            .entry(hash.clone())
            .or_insert(Admission::Unsigned);
        Ok(hash)
    }

    /// Register a definition interpret-only **with an [`InjectCert`]**, verifying it against the
    /// image's **own** `TrustRoot` (I4) before anything is registered. Works in both modes — in
    /// `loose` this is the per-inject signing opt-in (RFC-0038 §8.7); a cert that does not verify
    /// is **blocked**, never downgraded to unsigned-permitted.
    ///
    /// # Errors
    /// [`InjectError::BadSignature`] — untrusted signer, or a signature that does not cover this
    /// exact content (nothing is registered).
    pub fn define_signed(
        &mut self,
        node: Node,
        cert: &InjectCert,
    ) -> Result<ContentHash, InjectError> {
        let hash = node.content_hash();
        let signer = self.gate_signed(&hash, cert)?;
        self.defs.entry(hash.clone()).or_insert(node);
        self.admissions
            .insert(hash.clone(), Admission::Verified { signer });
        Ok(hash)
    }

    /// **Inject** a recompiled definition **unsigned**: compile its unit (the `dlopen` JIT) and
    /// register a `hash → entry` in the dispatch table, *also* recording it as an interpretable
    /// definition. Returns the definition's content hash (the dispatch key).
    ///
    /// **Gate (M-961):** on an `inoculated` image an unsigned inject is an explicit
    /// [`InjectError::UnsignedCode`] with **no** registration (RFC-0038 I2). In `loose` mode the
    /// admission is G2-tagged [`Admission::Unsigned`] (I1).
    ///
    /// **Never mutates a live entry (ADR-017 decision 4).** If a compiled entry already exists for
    /// this hash, the existing entry is kept (publish-once) and no recompile happens — by
    /// content-addressing it is byte-for-byte the same code. Injecting an *edited* definition is a
    /// **new hash**, so it lands under a new key and the old entry is untouched; an in-flight call to
    /// the old hash continues to dispatch to the old, still-loaded code.
    ///
    /// A failed unit compile/load is an explicit [`InjectError::Compile`] with **no** registration
    /// (never a partial entry); when `clang` is absent it is a skippable `Compile(ToolchainMissing)`.
    ///
    /// # Errors
    /// [`InjectError::UnsignedCode`] in `inoculated` mode; [`InjectError::Compile`] on a failed
    /// unit compile/load.
    pub fn inject(&mut self, node: &Node) -> Result<ContentHash, InjectError> {
        let hash = node.content_hash();
        if self.policy.mode() == InjectMode::Inoculated {
            return Err(InjectError::UnsignedCode(hash));
        }
        self.admissions
            .entry(hash.clone())
            .or_insert(Admission::Unsigned);
        self.register_compiled(hash, node)
    }

    /// **Inject with an [`InjectCert`]**: verify the cert against the image's **own** `TrustRoot`
    /// (I4) **first** — a refused cert registers nothing (never a partial admission) — then
    /// compile and register as [`inject`](Self::inject) does. This is the `inoculated` path, and
    /// in `loose` mode the per-inject signing opt-in (RFC-0038 §8.7).
    ///
    /// # Errors
    /// [`InjectError::BadSignature`] — untrusted signer or a signature that does not cover this
    /// exact content; [`InjectError::Compile`] on a failed unit compile/load.
    pub fn inject_signed(
        &mut self,
        node: &Node,
        cert: &InjectCert,
    ) -> Result<ContentHash, InjectError> {
        let hash = node.content_hash();
        let signer = self.gate_signed(&hash, cert)?;
        self.admissions
            .insert(hash.clone(), Admission::Verified { signer });
        self.register_compiled(hash, node)
    }

    /// The signed-admission gate: verify `cert` for the **actual** `hash` being admitted against
    /// the image's own policy (RFC-0038 §5.1/§7.2/I4). The signed message is rebuilt from the
    /// actual hash + the cert's carried attestation, so a cert minted for different content fails
    /// signature verification naturally — no secondary identity can drift from the dispatch key.
    fn gate_signed(&self, hash: &ContentHash, cert: &InjectCert) -> Result<SignerId, InjectError> {
        let msg = signed_message(hash, cert.vr4());
        self.policy
            .verify(cert.signer(), &msg, cert.signature())
            .map_err(|_refusal| InjectError::BadSignature(hash.clone(), cert.signer().clone()))?;
        Ok(cert.signer().clone())
    }

    /// The shared compile-and-register tail of [`inject`](Self::inject) /
    /// [`inject_signed`](Self::inject_signed) (the gate already ran).
    fn register_compiled(
        &mut self,
        hash: ContentHash,
        node: &Node,
    ) -> Result<ContentHash, InjectError> {
        // The definition is always interpretable (the continuum base) — record it first so a later
        // resolution can fall back even if the compiled overlay is dropped.
        self.defs
            .entry(hash.clone())
            .or_insert_with(|| node.clone());
        if self.compiled.contains_key(&hash) {
            // Publish-once: the key already holds this exact code (content-addressed). Do not
            // recompile and do not overwrite the live entry.
            return Ok(hash);
        }
        let artifact = compile_so(node).map_err(InjectError::Compile)?;
        self.compiled.insert(hash.clone(), artifact);
        Ok(hash)
    }

    /// How `hash` resolves — the `EXPLAIN`-able dispatch decision (ADR-017 decision 5), carrying
    /// the inject-mode dimension (RFC-0038 §7.3 I7): the image's mode + the entry's admission
    /// tag. A missing admission record floors to `Unsigned` (never over-claim `Verified` — VR-5).
    #[must_use]
    pub fn resolve(&self, hash: &ContentHash) -> Resolution {
        let admission = || {
            self.admissions
                .get(hash)
                .cloned()
                .unwrap_or(Admission::Unsigned)
        };
        if self.compiled.contains_key(hash) {
            Resolution::Compiled {
                inject_mode: self.policy.mode(),
                admission: admission(),
            }
        } else if self.defs.contains_key(hash) {
            Resolution::Interpreted {
                inject_mode: self.policy.mode(),
                admission: admission(),
            }
        } else {
            Resolution::Miss
        }
    }

    /// Dispatch a call by content hash (ADR-016's call ABI, nullary-unit restriction). Resolves to
    /// the compiled entry if present, else interprets the registered definition; a hash with neither
    /// is an explicit [`InjectError::DispatchMiss`] (never a silent guess).
    ///
    /// Under the Phase-I `whole` grain the security gate ran at registration time; an admitted
    /// entry's call is trusted (RFC-0038 §8.4/§8.6 — declared and EXPLAIN-able via
    /// [`resolve`](Self::resolve)/[`policy_manifest`](Self::policy_manifest), never a hidden
    /// weakening; per-call re-verification is the deferred `call` grain, M-847).
    ///
    /// # Errors
    /// [`InjectError::DispatchMiss`] for an unknown hash; [`InjectError::Compile`]/
    /// [`InjectError::Interp`] surfaced from the dispatched path.
    pub fn call(&self, hash: &ContentHash) -> Result<Value, InjectError> {
        if let Some(entry) = self.compiled.get(hash) {
            return entry.call().map_err(InjectError::Compile);
        }
        if let Some(node) = self.defs.get(hash) {
            return self.interp.eval(node).map_err(InjectError::Interp);
        }
        Err(InjectError::DispatchMiss(hash.clone()))
    }

    /// Whether a compiled (injected) entry exists for `hash`.
    #[must_use]
    pub fn is_injected(&self, hash: &ContentHash) -> bool {
        self.compiled.contains_key(hash)
    }

    /// The number of compiled (injected) entries — the dispatch table never shrinks on a re-inject
    /// of an existing hash (publish-once), so a stable count witnesses the no-overwrite property.
    #[must_use]
    pub fn injected_count(&self) -> usize {
        self.compiled.len()
    }

    /// The number of known (interpretable) definitions.
    #[must_use]
    pub fn defined_count(&self) -> usize {
        self.defs.len()
    }

    /// The **default-plus-deviations manifest** (RFC-0038 §8.5; DN-77 §4 item 7 — Phase-I slice):
    /// the declared default posture plus every per-inject departure, enumerated. In `loose` mode a
    /// signed-and-verified admission is a deviation (the §8.7 per-inject opt-in); in `inoculated`
    /// mode every admission is verified (the norm), so there are none to enumerate. Deterministic
    /// (sorted by site).
    #[must_use]
    pub fn policy_manifest(&self) -> PolicyManifest {
        let mut deviations: Vec<PolicyDeviation> = Vec::new();
        if self.policy.mode() == InjectMode::Loose {
            for (hash, adm) in &self.admissions {
                if let Admission::Verified { .. } = adm {
                    deviations.push(PolicyDeviation {
                        site: hash.as_str().to_owned(),
                        posture: adm.to_string(),
                        why: "per-inject signing opt-in in a loose context (RFC-0038 §8.7)"
                            .to_owned(),
                    });
                }
            }
        }
        deviations.sort_by(|a, b| a.site.cmp(&b.site));
        PolicyManifest {
            mode: self.policy.mode(),
            grain: self.policy.grain(),
            scheme: self.policy.scheme_name().to_owned(),
            trusted_signers: self.policy.trust_root().signers().cloned().collect(),
            deviations,
        }
    }

    /// Render the image's inject-security posture for the EXPLAIN channel (no black box).
    #[must_use]
    pub fn explain_policy(&self) -> String {
        self.policy_manifest().explain()
    }
}

/// The **recompile set** of a change, by hash reachability (ADR-017 decision 3 — no AST/file diff).
///
/// `deps` is the dependency graph: `deps[h]` is the set of hashes that definition `h` *directly
/// references*. `changed` is the set of edited definitions (each already a *new* hash). The result is
/// the closure that must be recompiled: every `changed` definition **plus** every definition that
/// transitively depends on a changed one (its callers, by reverse reachability) — because each such
/// dependent named a now-changed hash and is therefore itself a new definition. Everything outside
/// the set keeps its hash and its already-compiled entry (never re-injected).
///
/// Pure and deterministic; depends only on the hash graph, never on definition bodies.
#[must_use]
pub fn recompile_closure(
    deps: &HashMap<ContentHash, Vec<ContentHash>>,
    changed: &[ContentHash],
) -> HashSet<ContentHash> {
    // Invert the dependency edges to reverse edges (dependency → its dependents/callers).
    let mut dependents: HashMap<&ContentHash, Vec<&ContentHash>> = HashMap::new();
    for (h, references) in deps {
        for r in references {
            dependents.entry(r).or_default().push(h);
        }
    }
    // BFS the reverse graph from every changed hash; the closure includes the changed set itself.
    let mut closure: HashSet<ContentHash> = HashSet::new();
    let mut frontier: Vec<ContentHash> = changed.to_vec();
    while let Some(h) = frontier.pop() {
        if !closure.insert(h.clone()) {
            continue; // already visited
        }
        if let Some(callers) = dependents.get(&h) {
            for c in callers {
                if !closure.contains(*c) {
                    frontier.push((*c).clone());
                }
            }
        }
    }
    closure
}

// Tests extracted to src/tests/inject_tests.rs + src/tests/inject_gate_tests.rs (CLAUDE.md
// test-layout rule; M-789 as-touched; M-961 conformance suite).
