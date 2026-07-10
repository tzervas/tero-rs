//! **The native-artifact descriptor a deployable Spore embeds** (M-620; ADR-013; ADR-003;
//! RFC-0004 §6; VR-4/VR-5/G2).
//!
//! ADR-013 makes a **Spore** the content-addressed deployable unit — a hash-identified DAG of code +
//! state + metadata. M-620 asks: produce a deployable Spore *from the native-compiled backend*. The
//! full Spore wire-format that embeds compiled native artifacts (and the workspace wiring that makes
//! `mycelium-spore` depend on this crate) is a **design-first decision** recorded in `DN-18`
//! (design-complete, impl-pending) — it is an ADR-level change, not a silent build detail. What this
//! module lands **now** is the buildable, crate-local primitive that the Spore layer will embed: an
//! inspectable, content-addressed [`NativeArtifact`] descriptor of one natively-compiled program,
//! carrying the **VR-4 no-opaque-lowering attestation** into the deployed unit.
//!
//! **Content-addressed identity (ADR-003).** A [`NativeArtifact`]'s identity **is the content hash of
//! the program it compiles**, derived **internally** from [`Node::content_hash`] at build time — never
//! supplied, and so structurally unable to disagree with the program its `lowered_ir`/`vr4` embody.
//! This is the same Unison-style code identity ADR-003 fixes, enforced *by construction*: an artifact
//! whose `id()` names a different program than it compiled is **unrepresentable**, not merely rejected.
//! Everything else the descriptor carries — the dumpable IR text, the toolchain versions, the
//! `EXPLAIN` — is **metadata, not identity** (two builds of the *same* program on different LLVM patch
//! versions share one identity, because `content_hash` is over the program structure, not the lowered
//! text). [`NativeArtifact::id`] returns that canonical identity; [`NativeArtifact::same_identity_as`]
//! compares by it, ignoring metadata — exactly ADR-003's "metadata is not identity".
//!
//! **VR-4 carried into the deployment (the M-620↔M-630 seam).** The descriptor embeds the
//! [`crate::vr4`] cross-backend attestation (every backend's lowering is dumpable — no opaque pass)
//! **and** the program's own dumpable lowered IR, so the no-opaque-lowering guarantee (VR-4) travels
//! *with* the deployed unit and is inspectable at the deployment site, not just at build time
//! (RFC-0004 §6). [`NativeArtifact::explain`] renders the whole attestation.
//!
//! **Never-silent (G2), and *structured*.** A failure is an explicit [`DeployError`], never a guessed
//! default — and each failure keeps a distinct, *structured* signal so a caller branches on it without
//! brittle string-matching: an absent native toolchain is [`DeployError::ToolchainMissing`] (the
//! caller **skips** — the house idiom, mirroring [`AotError::ToolchainMissing`]); a program the native
//! backend cannot lower soundly is [`DeployError::NotDeployable`] carrying the backend's own
//! `EXPLAIN`-able reason (routed to the proven path), never fragile codegen shipped to fill the gap
//! (G2/VR-5).
//!
//! **Honesty (VR-5).** The descriptor's guarantee is `Empirical` — the lowered IR is the real
//! artifact, its faithfulness evidenced by the differentials (M-302/M-602); never `Proven` (no
//! machine-checked end-to-end deployment-correctness theorem; G2/VR-5).
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use mycelium_core::{ContentHash, GuaranteeStrength, Node};

use crate::llvm::{emit_llvm_ir, AotError};
use crate::vr4::{cross_backend_gate, CrossBackendGate};

/// Why producing a deployable native artifact failed — always explicit (G2), never a guessed default,
/// and always *structured* so a caller branches on the case without string-matching.
///
/// Note on "missing identity" (G2, the strongest form): the deploy **identity** is **derived
/// internally** from [`Node::content_hash`] — it is *never an input*, so it can neither be missing nor
/// disagree with the program. An invalid identity is unrepresentable *by construction*, not rejected
/// after the fact (CLAUDE.md banked guard 2). The two never-silent failures below are the real ones,
/// and both keep a structured signal.
#[derive(Debug, Clone, PartialEq)]
pub enum DeployError {
    /// The native toolchain (`llc`/`clang`) is absent — the caller should **skip**, not fail (the
    /// house "skip gracefully when a tool is absent" idiom). A *dedicated, structured* variant
    /// mirroring [`AotError::ToolchainMissing`], so the skip case is detectable without brittle
    /// string-matching of [`DeployError::NotDeployable`]. Carries the missing tool's name.
    ToolchainMissing(String),
    /// The program is outside the fragment the native backend can lower soundly. Carries the
    /// backend's own `EXPLAIN`-able reason; the program runs on the proven (interpreter / richer)
    /// path — fragile codegen is **never** shipped to fill the gap (G2/VR-5).
    NotDeployable(String),
}

impl core::fmt::Display for DeployError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DeployError::ToolchainMissing(t) => {
                write!(
                    f,
                    "native toolchain absent ({t}) — cannot produce a deployable native artifact here \
                     (skip; the program runs on the proven path)"
                )
            }
            DeployError::NotDeployable(m) => {
                write!(
                    f,
                    "program is not natively deployable: {m} (runs on the proven path)"
                )
            }
        }
    }
}

impl std::error::Error for DeployError {}

/// The inspectable, content-addressed descriptor of one natively-compiled program — the unit a
/// deployable Spore embeds (M-620; ADR-013/ADR-003). Identity is the program's content hash; the
/// dumpable IR + VR-4 attestation are carried metadata (not identity, ADR-003).
#[derive(Debug, Clone, PartialEq)]
pub struct NativeArtifact {
    /// The **canonical identity**: the content hash of the program this artifact compiles, derived
    /// internally from [`Node::content_hash`] (ADR-003 code identity) — never an input. Two builds of
    /// the same program are the same artifact, whatever the toolchain: identity is over the program
    /// structure, not the lowered IR text.
    identity: ContentHash,
    /// The program's **dumpable lowered LLVM IR** — the no-opaque-lowering evidence carried into the
    /// deployed unit (VR-4; RFC-0004 §6). Metadata, not identity (a different LLVM patch may render
    /// it differently for the same program).
    lowered_ir: String,
    /// The VR-4 **cross-backend attestation**: every backend's lowering is dumpable (no opaque pass).
    /// Carried so the guarantee travels with the deployment. Metadata, not identity.
    vr4: CrossBackendGate,
    /// The honest strength of "this artifact faithfully runs the program" — `Empirical` (the
    /// differentials), never `Proven` (VR-5).
    faithfulness: GuaranteeStrength,
}

impl NativeArtifact {
    /// Build the native-artifact descriptor for `node`. The content identity is derived **internally**
    /// from [`Node::content_hash`] (ADR-003) — never supplied, so an artifact's `id()` can never name a
    /// different program than its `lowered_ir`/`vr4` embody (that mismatch is unrepresentable, not
    /// merely rejected — the strongest G2 form).
    ///
    /// Lowers `node` to dumpable LLVM IR via the direct-LLVM backend and records the VR-4 cross-backend
    /// attestation. Two never-silent, *structured* refusals (so a caller branches without
    /// string-matching): an absent toolchain is [`DeployError::ToolchainMissing`] — the caller **skips**
    /// (the house idiom, mirroring [`AotError::ToolchainMissing`]); a program the backend cannot lower
    /// soundly is [`DeployError::NotDeployable`] carrying the backend's own `EXPLAIN`-able reason —
    /// never fragile codegen (G2/VR-5).
    pub fn build(node: &Node) -> Result<Self, DeployError> {
        // Identity IS the program's content hash (ADR-003), computed here from the program structure —
        // not a caller's claim, not the lowered text — so `id()` cannot name a different program than
        // the IR/attestation below embody.
        let identity = node.content_hash();
        // The dumpable lowering is the artifact's no-opaque evidence; an out-of-fragment node (or a
        // missing toolchain) is an explicit, *structured* refusal routed to the proven path — never
        // fragile output.
        let lowered_ir = match emit_llvm_ir(node) {
            Ok(ir) => ir,
            // The toolchain-absent case keeps its structured signal (mirrors AotError::ToolchainMissing)
            // so a caller can skip without string-matching — the house "skip gracefully" idiom.
            Err(AotError::ToolchainMissing(t)) => return Err(DeployError::ToolchainMissing(t)),
            Err(e) => return Err(DeployError::NotDeployable(e.to_string())),
        };
        let vr4 = cross_backend_gate(node);
        Ok(NativeArtifact {
            identity,
            lowered_ir,
            vr4,
            // The artifact is a real compiled lowering; its faithfulness is evidenced by the
            // interp↔native differentials (M-302/M-602), never proven end-to-end (VR-5).
            faithfulness: GuaranteeStrength::Empirical,
        })
    }

    /// The canonical content-addressed identity (the program's hash; ADR-003). **This** is the
    /// artifact's identity — metadata is not.
    #[must_use]
    pub fn id(&self) -> &ContentHash {
        &self.identity
    }

    /// The dumpable lowered LLVM IR carried into the deployment (VR-4 evidence). Metadata.
    #[must_use]
    pub fn lowered_ir(&self) -> &str {
        &self.lowered_ir
    }

    /// The VR-4 cross-backend attestation travelling with the deployed unit (no opaque pass anywhere).
    #[must_use]
    pub fn vr4(&self) -> &CrossBackendGate {
        &self.vr4
    }

    /// The honest faithfulness strength — `Empirical` (the differentials), never `Proven` (VR-5).
    #[must_use]
    pub fn faithfulness(&self) -> GuaranteeStrength {
        self.faithfulness
    }

    /// Whether two artifacts have the **same content-addressed identity** (ADR-003) — i.e. compile the
    /// same program — **ignoring metadata** (the IR text, the attestation, the tool versions). This is
    /// the "metadata is not identity" comparison: two builds of the same program are equal here even
    /// if their carried IR differs byte-for-byte.
    #[must_use]
    pub fn same_identity_as(&self, other: &NativeArtifact) -> bool {
        self.identity == other.identity
    }

    /// A human-readable `EXPLAIN` of the deployable artifact: its content identity, the carried-IR
    /// size, the faithfulness tag, and the embedded VR-4 attestation — so the deployed unit's
    /// no-opaque-lowering guarantee is auditable at the deployment site (no black box; RFC-0004 §6).
    #[must_use]
    pub fn explain(&self) -> String {
        format!(
            "NativeArtifact (M-620 deployable; ADR-013/ADR-003):\n  identity: {} (content-addressed \
             code identity — metadata is NOT identity)\n  lowered LLVM IR: {} bytes (dumpable — VR-4 \
             evidence carried into the deployment)\n  faithfulness: {:?} (the differentials; never \
             Proven — VR-5)\n{}",
            self.identity.as_str(),
            self.lowered_ir.len(),
            self.faithfulness,
            self.vr4.explain(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{Meta, Payload, Provenance, Repr, Trit, Value};

    fn byte(bits: [bool; 8]) -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(bits.to_vec()),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    /// Build, or signal a skip (`None`) when the native toolchain is absent — the honest house idiom
    /// (the structured [`DeployError::ToolchainMissing`]) so the deploy tests pass on a machine without
    /// `llc`/`clang` instead of failing. Any *other* error is a real bug and panics.
    fn build_or_skip(node: &Node) -> Option<NativeArtifact> {
        match NativeArtifact::build(node) {
            Ok(a) => Some(a),
            Err(DeployError::ToolchainMissing(_)) => None,
            Err(e) => panic!("unexpected deploy error: {e}"),
        }
    }

    fn in_fragment() -> Node {
        Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Op {
                prim: "bit.xor".into(),
                args: vec![
                    Node::Const(byte([true, false, true, true, false, false, true, false])),
                    Node::Const(byte([false, false, true, false, true, false, true, true])),
                ],
            }],
        }
    }

    #[test]
    fn builds_a_deployable_artifact_carrying_the_vr4_attestation() {
        // M-620: a native artifact built from the in-fragment program carries its content identity,
        // dumpable IR (VR-4 evidence), and the cross-backend no-opaque attestation.
        let prog = in_fragment();
        let Some(art) = build_or_skip(&prog) else {
            return; // toolchain absent — honest structured skip
        };
        // Identity IS the program's content hash (ADR-003), derived internally — not a supplied input.
        assert_eq!(art.id(), &prog.content_hash());
        assert!(
            !art.lowered_ir().is_empty(),
            "dumpable IR is carried (VR-4)"
        );
        // The VR-4 attestation covers all backends (no opaque pass) — the deployed guarantee.
        assert!(art.vr4().covered() >= 5);
        // Honest tag: Empirical, never Proven (VR-5).
        assert_eq!(art.faithfulness(), GuaranteeStrength::Empirical);
        // The EXPLAIN names the identity and the VR-4 obligation.
        let ex = art.explain();
        assert!(ex.contains(art.id().as_str()));
        assert!(ex.contains("VR-4 no-opaque-lowering"));
    }

    #[test]
    fn identity_is_the_program_content_hash_derived_not_supplied() {
        // ADR-003 (Copilot #264 fix): identity is the program's content hash, derived **internally** —
        // the caller cannot supply (and so cannot forge) it. We witness the three invariants this buys.
        let prog_a = in_fragment(); // bit.not(bit.xor(a,b))
        let prog_b = Node::Op {
            // a genuinely different program ⇒ different content hash AND different lowered IR
            prim: "bit.and".into(),
            args: vec![
                Node::Const(byte([true; 8])),
                Node::Const(byte([false, true, false, true, false, true, false, true])),
            ],
        };
        let (Some(a), Some(b)) = (build_or_skip(&prog_a), build_or_skip(&prog_b)) else {
            return; // toolchain absent — honest structured skip
        };
        // (1) Identity IS the program hash — exactly, by construction (not the IR text, not a claim).
        assert_eq!(a.id(), &prog_a.content_hash());
        assert_eq!(b.id(), &prog_b.content_hash());
        // (2) Different programs ⇒ different identity (their lowered IR also differs — real distinctness).
        assert_ne!(
            a.lowered_ir(),
            b.lowered_ir(),
            "the two programs must have different lowered IR (real distinctness)"
        );
        assert!(
            !a.same_identity_as(&b),
            "different programs ⇒ different identity"
        );
        // (3) Metadata is not identity: a second independent build of the SAME program yields the SAME
        // identity. `content_hash` is over the program structure, not the lowered text — so two builds
        // on different LLVM patch versions (whose IR would differ) would still share this identity.
        let Some(a2) = build_or_skip(&prog_a) else {
            return;
        };
        assert!(
            a.same_identity_as(&a2),
            "same program ⇒ same identity (metadata is not identity)"
        );
        assert_eq!(a.id(), a2.id());
    }

    #[test]
    fn identity_cannot_be_forged_and_a_swap_is_a_structured_refusal() {
        // G2 (the strongest form): identity is derived from the program (`content_hash`), never an
        // input — so there is no path by which an artifact's id() disagrees with its program. We pin
        // it directly: the built artifact's id() equals the program hash, full stop (no forgeable
        // parameter exists — Copilot #264).
        let prog = in_fragment();
        if let Some(art) = build_or_skip(&prog) {
            assert_eq!(
                art.id(),
                &prog.content_hash(),
                "id() is the program hash, not a forgeable input"
            );
        }
        // M-852 (E25-1): a `Swap` over the certified binary↔ternary class is now natively deployable
        // (the direct-LLVM backend lowers it). `(8,6)` is a legal pair, so it builds (or skips on a
        // box without the toolchain) — no longer an out-of-fragment refusal.
        let legal_swap = Node::Swap {
            src: Box::new(Node::Const(byte([true; 8]))),
            target: Repr::Ternary { trits: 6 },
            policy: ContentHash::parse("blake3:round_trip_safe").unwrap(),
        };
        match NativeArtifact::build(&legal_swap) {
            Ok(_) | Err(DeployError::ToolchainMissing(_)) => { /* deployable, or honest env skip */
            }
            Err(e) => panic!("a legal binary↔ternary swap must be deployable now (M-852), got {e}"),
        }

        // The remaining never-silent path: a swap we do **not** lower — an **illegal** pair `(8,2)`
        // (2^7=128 > (3^2−1)/2=4) — is refused at compile time (the Recheck mode's side-condition
        // re-check) ⇒ an explicit *structured* refusal, never fragile codegen (G2/VR-5). This refusal
        // is returned during lowering, before the toolchain, so it holds even on a box without llc.
        let illegal_swap = Node::Swap {
            src: Box::new(Node::Const(byte([true; 8]))),
            target: Repr::Ternary { trits: 2 },
            policy: ContentHash::parse("blake3:round_trip_safe").unwrap(),
        };
        match NativeArtifact::build(&illegal_swap) {
            Err(DeployError::NotDeployable(m)) => {
                assert!(!m.is_empty(), "the refusal carries an EXPLAIN-able reason");
            }
            Err(DeployError::ToolchainMissing(_)) => {}
            Ok(_) => {
                panic!("an illegal-pair swap must not be natively deployable (out of fragment)")
            }
        }
    }

    #[test]
    fn a_trit_carry_op_deploys_on_the_direct_llvm_native_path() {
        // The direct-LLVM backend (M-301) DOES natively compile trit carry arithmetic
        // (`trit.add/sub/mul` over Ternary{m}) — so it is genuinely deployable, and the artifact
        // carries its dumpable IR + the VR-4 attestation. (The MLIR-dialect fragment refuses trit
        // carry and routes it here — that boundary lives in vr4.rs, not in the deploy artifact, which
        // uses the richer direct-LLVM backend.) This pins that the native deploy path covers more than
        // the element-wise fragment, honestly.
        let add = Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(
                    Value::new(
                        Repr::Ternary { trits: 3 },
                        Payload::Trits(vec![Trit::Pos, Trit::Neg, Trit::Zero]),
                        Meta::exact(Provenance::Root),
                    )
                    .unwrap(),
                ),
                Node::Const(
                    Value::new(
                        Repr::Ternary { trits: 3 },
                        Payload::Trits(vec![Trit::Zero, Trit::Pos, Trit::Pos]),
                        Meta::exact(Provenance::Root),
                    )
                    .unwrap(),
                ),
            ],
        };
        let Some(art) = build_or_skip(&add) else {
            return; // toolchain absent — honest structured skip
        };
        assert_eq!(art.id(), &add.content_hash());
        assert!(!art.lowered_ir().is_empty(), "carries dumpable IR (VR-4)");
        assert_eq!(art.faithfulness(), GuaranteeStrength::Empirical);
    }

    #[test]
    fn the_artifact_explain_is_deterministic_for_deployment() {
        // The deployed attestation must be byte-deterministic so its identity is stable across runs /
        // machines (ADR-003 content-addressing).
        let (Some(a), Some(b)) = (build_or_skip(&in_fragment()), build_or_skip(&in_fragment()))
        else {
            return; // toolchain absent — honest structured skip
        };
        assert_eq!(a.explain(), b.explain());
    }
}
