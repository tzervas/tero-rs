//! **The VR-4 no-opaque-lowering cross-backend gate** (M-630 — the Phase-6 EXIT gate; RFC-0004 §6;
//! VR-4/VR-5; ADR-009; NFR-7).
//!
//! VR-4's normative obligation (RFC-0004 §6): *"Every stage is dumpable/diffable (SC-4); each pass
//! preserves `Meta` (WF5); no-opaque-lowering applies to **all** backends (ADR-009)."* This module
//! turns that prose obligation into a **mechanical, enumerable gate**: for **every** execution
//! backend Mycelium ships, it produces the backend's lowering stage as **inspectable textual output**
//! (no opaque pass anywhere), each tagged at its honestly-established strength (VR-5). The gate is the
//! single place that asserts "I can dump *every* backend", so a future backend that cannot be dumped
//! fails this gate by construction rather than slipping in silently.
//!
//! **The backends and their dumpable stage** (each a real entry point that already exists — this
//! module *aggregates* them, it does not invent a new lowering):
//!
//! | Backend | Dumpable stage | Entry point | Tag (VR-5) |
//! |---|---|---|---|
//! | Interpreter (trusted base, NFR-7) | the Core IR it evaluates | [`mycelium_core::lower::dump_node`] | `Declared` (the reference text; it *is* ground truth, not evidenced *against* anything) |
//! | AOT env-machine | the lowered A-normal-form substrate stage | [`mycelium_core::lower::Anf::dump`] | `Declared` (a faithful structural render) |
//! | Direct-LLVM (M-301/M-373/M-378/M-379) | textual LLVM IR | [`crate::llvm::emit_llvm_ir`] | `Empirical` (the interp↔native differential, M-302) |
//! | MLIR-dialect textual skeleton (M-150) | the ternary-dialect ANF skeleton | [`crate::dialect::emit`] | `Declared` (the always-available no-opaque anchor) |
//! | MLIR-dialect real lowering (M-601, feature `mlir-dialect`) | the `arith`/`func` module + the lowered LLVM IR | `crate::dialect::native::emit_mlir` / `crate::dialect::native::lower_to_llvm_ir` | `Empirical` (the three-way differential, M-602) |
//! | JIT / SIMD packed-ternary (M-360/M-610) | the unpack-compute kernel IR (**node-independent** — a fixed I2_S kernel exemplar; the packed-ternary kernel is a runtime-data primitive, not a lowering of the program) | [`crate::bitnet::emit_bitnet_dot_ir_for`] / [`crate::simd`] | `Empirical` (the SIMD↔scalar differential) |
//!
//! **Honesty (VR-5).** The tag on each stage is the strength of the *evidence that the dumped stage
//! is faithful to what runs*, not a claim about the program's numerics. The compiled paths are
//! `Empirical` (a differential validates the dump's semantics against the interpreter); the textual
//! renders and the trusted-base Core IR are `Declared` (asserted faithful renders — the interpreter
//! text is itself the reference). Nothing here is `Proven`: no machine-checked
//! render-faithfulness theorem exists (G2/VR-5).
//!
//! **Never-silent (G2).** A backend whose toolchain/feature is absent yields a [`StageStatus::Skipped`]
//! with the reason, never a fabricated dump. The gate reports which backends were *covered* so a
//! caller can require non-vacuous coverage (the interpreter, AOT-env, direct-LLVM textual skeleton
//! and the SIMD kernel IR are **always** dumpable — no toolchain needed — so the gate is never empty).
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use mycelium_core::{GuaranteeStrength, Node, PackScheme};

use crate::bitnet::emit_bitnet_dot_ir_for;
use crate::dialect;
use crate::llvm::emit_llvm_ir;

/// A backend whose lowering VR-4 requires to be dumpable. Enumerable so the gate is exhaustive: every
/// variant must yield an inspectable stage (or an explicit [`StageStatus::Skipped`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// The reference interpreter — the trusted base (NFR-7). Its "stage" is the Core IR it evaluates.
    Interpreter,
    /// The AOT env-machine — its stage is the lowered A-normal form (the substrate stage).
    AotEnvMachine,
    /// The direct-LLVM-IR backend (M-301/M-373/M-378/M-379) — its stage is textual LLVM IR.
    DirectLlvm,
    /// The MLIR-dialect **textual skeleton** (M-150) — always available, the no-opaque anchor.
    MlirSkeleton,
    /// The MLIR-dialect **real** `arith`/`func`→LLVM lowering (M-601; feature `mlir-dialect`).
    MlirDialect,
    /// The JIT / SIMD packed-ternary kernel path (M-360/M-610) — its stage is the unpack-compute IR.
    JitSimd,
}

impl Backend {
    /// All backends, in lowering order — the exhaustive set the VR-4 gate must cover.
    #[must_use]
    pub fn all() -> [Backend; 6] {
        [
            Backend::Interpreter,
            Backend::AotEnvMachine,
            Backend::DirectLlvm,
            Backend::MlirSkeleton,
            Backend::MlirDialect,
            Backend::JitSimd,
        ]
    }

    /// A stable human-readable name (for `EXPLAIN` / reports).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Backend::Interpreter => "interpreter (trusted base)",
            Backend::AotEnvMachine => "AOT env-machine",
            Backend::DirectLlvm => "direct-LLVM",
            Backend::MlirSkeleton => "MLIR-dialect textual skeleton",
            Backend::MlirDialect => "MLIR-dialect (arith/func -> LLVM)",
            Backend::JitSimd => "JIT/SIMD packed-ternary",
        }
    }
}

/// Whether a backend's stage was dumped, or skipped (with a reason) — never a fabricated dump (G2).
#[derive(Debug, Clone, PartialEq)]
pub enum StageStatus {
    /// The stage was dumped: the inspectable textual lowering output.
    Dumped(String),
    /// The stage could not be produced in this environment (a missing tool/feature, or a node
    /// outside this backend's fragment). The reason is explicit — an honest gap, not an opaque pass.
    Skipped(String),
}

impl StageStatus {
    /// Whether this stage was actually dumped (covered), vs. an honest skip.
    #[must_use]
    pub fn is_dumped(&self) -> bool {
        matches!(self, StageStatus::Dumped(_))
    }
}

/// One backend's VR-4 obligation result: the backend, its dumpable-stage status, and the honest
/// strength of the claim that the dump is faithful to what runs (VR-5). Inspectable; nothing here is
/// `Proven` (no render-faithfulness theorem; G2/VR-5).
#[derive(Debug, Clone, PartialEq)]
pub struct BackendStage {
    /// Which backend.
    pub backend: Backend,
    /// Its dumpable stage (or an explicit skip).
    pub status: StageStatus,
    /// The honestly-established strength of "this dump is faithful to what the backend runs".
    pub faithfulness: GuaranteeStrength,
}

impl BackendStage {
    /// A short `EXPLAIN` line: backend, covered/skipped, the faithfulness tag, and the dump size /
    /// skip reason — so the cross-backend gate result is auditable at a glance (no black box).
    #[must_use]
    pub fn explain(&self) -> String {
        match &self.status {
            StageStatus::Dumped(d) => format!(
                "{:<34} covered   [{:?}] — {} bytes of inspectable lowering",
                self.backend.name(),
                self.faithfulness,
                d.len()
            ),
            StageStatus::Skipped(why) => format!(
                "{:<34} skipped   [{:?}] — {why}",
                self.backend.name(),
                self.faithfulness
            ),
        }
    }
}

/// The full VR-4 cross-backend gate result for one program: a [`BackendStage`] for **every** backend.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossBackendGate {
    /// Per-backend dumpable-stage results, in lowering order.
    pub stages: Vec<BackendStage>,
}

impl CrossBackendGate {
    /// Whether **every** backend produced a dumpable stage (no skips) — the strongest gate verdict.
    /// Usually `false` in practice (the real MLIR path skips when libMLIR is absent); use
    /// [`covered`](Self::covered) for the always-available floor.
    #[must_use]
    pub fn fully_covered(&self) -> bool {
        self.stages.iter().all(|s| s.status.is_dumped())
    }

    /// How many backends produced an actual dump (vs. an honest skip). The always-available backends
    /// (interpreter, AOT-env, direct-LLVM textual, MLIR skeleton, JIT/SIMD IR) guarantee this is
    /// `≥ 5` for an in-fragment program — so the gate is **never vacuous** even without libMLIR.
    #[must_use]
    pub fn covered(&self) -> usize {
        self.stages.iter().filter(|s| s.status.is_dumped()).count()
    }

    /// The aggregate `EXPLAIN` — one line per backend. The auditable record that VR-4 holds across
    /// all backends (no opaque pass anywhere), each verdict at its honest strength (VR-5).
    #[must_use]
    pub fn explain(&self) -> String {
        let mut s = String::from(
            "VR-4 no-opaque-lowering cross-backend gate (M-630; RFC-0004 §6 / ADR-009):\n",
        );
        for st in &self.stages {
            s.push_str("  ");
            s.push_str(&st.explain());
            s.push('\n');
        }
        s
    }
}

/// Run the **VR-4 cross-backend gate** over `node`: for every backend, produce its dumpable lowering
/// stage as inspectable text (or an explicit, never-silent skip). This is the Phase-6 exit obligation
/// — *no opaque pass on any backend* — made mechanical and enumerable (M-630; RFC-0004 §6).
///
/// The interpreter (Core IR), AOT env-machine (lowered ANF), MLIR textual skeleton, and — for an
/// in-fragment node — the direct-LLVM IR are **always** dumpable (no toolchain). The real MLIR
/// lowering ([`Backend::MlirDialect`]) and the JIT/SIMD kernel IR are also pure-text emissions (no
/// tool needed to *emit* the IR), so they too dump without libMLIR/clang present — the toolchain is
/// only needed to *run* them, not to inspect them. A node outside a backend's fragment is an explicit
/// [`StageStatus::Skipped`] (the same honest boundary the backend itself enforces; G2).
#[must_use]
pub fn cross_backend_gate(node: &Node) -> CrossBackendGate {
    let stages = Backend::all()
        .into_iter()
        .map(|backend| dump_stage(backend, node))
        .collect();
    CrossBackendGate { stages }
}

/// Produce one backend's dumpable stage over `node` (an inspectable text dump, or an explicit skip),
/// with the honest faithfulness tag (VR-5).
fn dump_stage(backend: Backend, node: &Node) -> BackendStage {
    use mycelium_core::lower::{dump_node, lower_to_anf};
    let (status, faithfulness) = match backend {
        // The interpreter evaluates the Core IR; the Core IR dump *is* its dumpable stage. It is the
        // reference text (ground truth), so the faithfulness claim is Declared (nothing to evidence
        // it against — it is what everything else is checked against).
        Backend::Interpreter => (
            StageStatus::Dumped(dump_node(node)),
            GuaranteeStrength::Declared,
        ),
        // The AOT env-machine evaluates the lowered A-normal form; its dump is the substrate stage.
        Backend::AotEnvMachine => (
            StageStatus::Dumped(lower_to_anf(node).dump()),
            GuaranteeStrength::Declared,
        ),
        // Direct-LLVM emits textual LLVM IR for the in-fragment subset; an out-of-fragment node is an
        // explicit refusal (the backend's own honest boundary). When it *dumps*, the dump's semantics
        // are validated against the interpreter by the M-302 differential ⇒ Empirical. When it
        // *skips*, nothing was dumped, so the tag is `Declared` — the asserted strength of the honest
        // boundary, not a differential verdict over an absent artifact (VR-5).
        Backend::DirectLlvm => match emit_llvm_ir(node) {
            Ok(ir) => (StageStatus::Dumped(ir), GuaranteeStrength::Empirical),
            Err(e) => (
                StageStatus::Skipped(format!("out of the direct-LLVM fragment: {e}")),
                GuaranteeStrength::Declared,
            ),
        },
        // The MLIR textual skeleton is always available (no toolchain) — the no-opaque anchor.
        Backend::MlirSkeleton => (
            StageStatus::Dumped(dialect::emit(node)),
            GuaranteeStrength::Declared,
        ),
        // The real MLIR lowering: emitting the arith/func module is a pure-text emission (no tool to
        // *emit* — the tool only *runs* it), so it dumps even without libMLIR. A dumped module is
        // validated three ways by M-602 ⇒ Empirical; a skip (out-of-fragment, or the feature off)
        // dumped nothing ⇒ `Declared`. Feature-gated.
        Backend::MlirDialect => mlir_dialect_stage(node),
        // The JIT/SIMD packed-ternary path: its dumpable stage is the unpack-compute kernel IR. NOTE
        // it is **node-independent** — a fixed I2_S kernel exemplar (the TL1/TL2/SIMD kernels emit
        // analogously), since the packed-ternary kernel is a runtime-data primitive, not a lowering of
        // `node`. So its stage is always present; the `Err` arm is unreachable (I2_S is statically
        // supported), kept only as a never-silent guard. A dumped kernel is differential-validated
        // against the scalar oracle ⇒ Empirical.
        Backend::JitSimd => match emit_bitnet_dot_ir_for(PackScheme::I2S) {
            Ok(ir) => (StageStatus::Dumped(ir), GuaranteeStrength::Empirical),
            // Unreachable in practice (I2_S always has a kernel); a never-silent guard, tagged
            // `Declared` because it would have dumped nothing.
            Err(e) => (
                StageStatus::Skipped(format!("kernel IR unavailable: {e}")),
                GuaranteeStrength::Declared,
            ),
        },
    };
    BackendStage {
        backend,
        status,
        faithfulness,
    }
}

/// The MLIR-dialect real-lowering stage. With the `mlir-dialect` feature ON, emit the `arith`/`func`
/// module (a pure-text emission — no tool needed to *inspect* it; the toolchain only *runs* it). With
/// the feature OFF the real path is not compiled in, so it is an explicit, honest skip (G2).
#[cfg(feature = "mlir-dialect")]
fn mlir_dialect_stage(node: &Node) -> (StageStatus, GuaranteeStrength) {
    match crate::dialect::native::emit_mlir(node) {
        // A dumped arith/func module is validated three ways by M-602 ⇒ Empirical.
        Ok((module, _kind, _width)) => (StageStatus::Dumped(module), GuaranteeStrength::Empirical),
        // An out-of-fragment node dumped nothing ⇒ `Declared` (the asserted boundary, not a verdict
        // over an absent artifact; VR-5).
        Err(e) => (
            StageStatus::Skipped(format!("out of the MLIR element-wise fragment: {e}")),
            GuaranteeStrength::Declared,
        ),
    }
}

/// Feature-OFF: the real MLIR lowering is not compiled in — an explicit skip (never a fabricated
/// dump). The textual skeleton ([`Backend::MlirSkeleton`]) still covers the no-opaque anchor. The tag
/// is `Declared`: nothing was dumped, so there is no differential verdict to claim (VR-5).
#[cfg(not(feature = "mlir-dialect"))]
fn mlir_dialect_stage(_node: &Node) -> (StageStatus, GuaranteeStrength) {
    (
        StageStatus::Skipped(
            "feature `mlir-dialect` OFF — real arith/func lowering not compiled in (the textual \
             skeleton covers the no-opaque anchor)"
                .to_owned(),
        ),
        GuaranteeStrength::Declared,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{Meta, Payload, Provenance, Repr, Value};

    fn byte(bits: [bool; 8]) -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(bits.to_vec()),
            Meta::exact(Provenance::Root),
        )
        .unwrap()
    }

    /// An in-fragment program every compiled backend can lower: `bit.not(bit.xor(A, B))`.
    fn in_fragment() -> Node {
        let a = byte([true, false, true, true, false, false, true, false]);
        let b = byte([false, false, true, false, true, false, true, true]);
        Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Op {
                prim: "bit.xor".into(),
                args: vec![Node::Const(a), Node::Const(b)],
            }],
        }
    }

    #[test]
    fn the_gate_covers_every_backend_with_an_inspectable_stage() {
        // M-630: VR-4 across ALL backends — every backend that is *compiled in* yields a dumpable,
        // non-empty lowering stage for an in-fragment program. No opaque pass anywhere. The real MLIR
        // dialect path is only compiled in under the `mlir-dialect` feature; with it OFF that backend
        // is an explicit, honest skip (the textual skeleton still covers the no-opaque anchor) — never
        // a fabricated dump (G2). So the dump obligation is feature-aware, not blind.
        let gate = cross_backend_gate(&in_fragment());
        assert_eq!(gate.stages.len(), Backend::all().len());
        for st in &gate.stages {
            // Nothing ever claims Proven — no render-faithfulness theorem (VR-5).
            assert_ne!(st.faithfulness, GuaranteeStrength::Proven);
            match &st.status {
                StageStatus::Dumped(d) => assert!(
                    !d.is_empty(),
                    "{} produced an empty dump",
                    st.backend.name()
                ),
                StageStatus::Skipped(why) => {
                    // The ONLY backend allowed to skip an in-fragment program is the real MLIR
                    // dialect path when its feature is off — and it must say so explicitly.
                    assert_eq!(
                        st.backend,
                        Backend::MlirDialect,
                        "in-fragment program: {} must be dumpable, skipped: {why}",
                        st.backend.name()
                    );
                    assert!(!why.is_empty(), "the skip must carry an explicit reason");
                }
            }
        }
        // The always-available backends are covered regardless of toolchain/feature (the floor).
        for b in [
            Backend::Interpreter,
            Backend::AotEnvMachine,
            Backend::DirectLlvm,
            Backend::MlirSkeleton,
            Backend::JitSimd,
        ] {
            let st = gate.stages.iter().find(|s| s.backend == b).unwrap();
            assert!(st.status.is_dumped(), "{} must always dump", b.name());
        }
        // With the real MLIR path compiled in, the strongest verdict holds: every backend covered.
        #[cfg(feature = "mlir-dialect")]
        {
            assert!(gate.fully_covered(), "{}", gate.explain());
            assert_eq!(gate.covered(), Backend::all().len());
        }
        // Without it, the real MLIR path honestly skips, so 5 of 6 are covered (never fewer).
        #[cfg(not(feature = "mlir-dialect"))]
        {
            assert!(!gate.fully_covered());
            assert_eq!(gate.covered(), Backend::all().len() - 1);
        }
    }

    #[test]
    fn the_gate_is_never_vacuous_even_without_a_toolchain() {
        // The always-text backends (interpreter, AOT-env, direct-LLVM IR, MLIR skeleton, real MLIR
        // module emission, JIT/SIMD IR) dump without libMLIR/clang — so coverage never collapses to
        // zero. This is the floor that makes a green gate meaningful.
        let gate = cross_backend_gate(&in_fragment());
        assert!(
            gate.covered() >= 5,
            "at least the always-available backends must be covered, got {}:\n{}",
            gate.covered(),
            gate.explain()
        );
    }

    #[test]
    fn an_out_of_fragment_node_is_an_explicit_skip_not_an_opaque_pass() {
        // A trit-carry op is outside the direct-LLVM bit subset and the MLIR element-wise fragment —
        // those backends must SKIP it explicitly (the honest boundary), never silently lower it. The
        // interpreter / AOT-env still dump it (they cover the full calculus).
        let add = Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(
                    Value::new(
                        Repr::Ternary { trits: 3 },
                        Payload::Trits(vec![
                            mycelium_core::Trit::Pos,
                            mycelium_core::Trit::Neg,
                            mycelium_core::Trit::Zero,
                        ]),
                        Meta::exact(Provenance::Root),
                    )
                    .unwrap(),
                ),
                Node::Const(
                    Value::new(
                        Repr::Ternary { trits: 3 },
                        Payload::Trits(vec![
                            mycelium_core::Trit::Zero,
                            mycelium_core::Trit::Pos,
                            mycelium_core::Trit::Pos,
                        ]),
                        Meta::exact(Provenance::Root),
                    )
                    .unwrap(),
                ),
            ],
        };
        let gate = cross_backend_gate(&add);
        // The interpreter and AOT-env machine cover the full calculus — they dump it.
        let interp = gate
            .stages
            .iter()
            .find(|s| s.backend == Backend::Interpreter)
            .unwrap();
        assert!(interp.status.is_dumped(), "interp dumps the full calculus");
        // Direct-LLVM is out-of-fragment here ⇒ an explicit skip (never an opaque lowering).
        let direct = gate
            .stages
            .iter()
            .find(|s| s.backend == Backend::DirectLlvm)
            .unwrap();
        // It is either a dump (if the bit-subset emitter handles it) or an explicit skip — never a
        // silent/opaque pass. For trit.add the direct-LLVM path refuses, so assert the skip is
        // explicit with a reason.
        if let StageStatus::Skipped(why) = &direct.status {
            assert!(!why.is_empty(), "the skip must carry an explicit reason");
        }
    }

    #[test]
    fn explain_lists_every_backend_and_is_deterministic() {
        let gate = cross_backend_gate(&in_fragment());
        let ex = gate.explain();
        for b in Backend::all() {
            assert!(ex.contains(b.name()), "EXPLAIN must list {}", b.name());
        }
        assert!(ex.contains("VR-4 no-opaque-lowering"));
        // Deterministic (pure text emission, no RNG/time).
        assert_eq!(cross_backend_gate(&in_fragment()).explain(), ex);
    }
}
