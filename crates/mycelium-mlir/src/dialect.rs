//! Ternary-dialect lowering (M-150 textual skeleton + M-601 real `arith`/`func`→LLVM path).
//!
//! Two artifacts share this module:
//!
//! - [`emit`] — the **textual** ternary-dialect rendering of the lowered A-normal form
//!   (`mycelium-core::lower`): one dialect op per binding, all attributes inline so nothing is
//!   opaque (M-150; RFC-0004 §6). Always available, no toolchain needed — the per-stage-dumpable
//!   anchor and the *shape* of the eventual MLIR path.
//!
//! - `native` (feature `mlir-dialect`, OFF by default) — the **real** lowering (M-601; M-725;
//!   RFC-0004 §2; RFC-0029 §7; ADR-009/ADR-019). For the **bit/trit element-wise fragment**
//!   (`core.id`, `bit.not/and/or/xor`, `trit.neg`) **plus the balanced-ternary additive carry chain**
//!   (`trit.add`/`trit.sub`, widened by M-725) it emits a genuine MLIR module in the
//!   `arith`/`func`/`cf` dialects, runs it through `mlir-opt-<v> --convert-cf-to-llvm
//!   --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts |
//!   mlir-translate-<v> --mlir-to-llvmir` to **real LLVM IR**, then `clang` → native → read-back
//!   (the *same* read-back protocol as [`crate::llvm`], including the shared overflow sentinel, so
//!   the two compiled paths are differential-equivalent by one contract). This is a fourth, genuinely
//!   MLIR-compiled execution path — not the textual skeleton.
//!
//! **Honesty / scope (VR-5; G2).** `native` covers the fragment the **standard** MLIR dialects
//! carry faithfully: the bit/trit element-wise ops **and** (M-725) the additive carry chain
//! `trit.add`/`trit.sub`, lowered as a fixed-width ripple-carry over `arith` with a never-silent
//! overflow read-back. The **new honest boundary** is `trit.mul` (the shifted-accumulate fragment)
//! and every richer node — the data fragment (`Construct`/`Match`), closures (`App`/`Lam`),
//! recursion (`Fix`/`FixGroup`), `Swap`, Dense/VSA — each an **explicit, never-silent**
//! `native::DialectError::Unsupported` with an `EXPLAIN`-able reason that routes it back to the
//! richer direct-LLVM backend ([`crate::llvm`]) or the interpreter. No fragile/divergent codegen is
//! ever shipped to widen coverage (G2/VR-5). The `mlir-opt`/`mlir-translate` tools are **probed at
//! runtime** and their absence is a graceful `native::DialectError::ToolchainMissing` (skip, never
//! fail) — mirroring the `llc`/`clang` idiom — so the feature build/test stays green on a box without
//! libMLIR (ADR-019).
//!
//! **Guarantee tag:** `Empirical` — a genuinely compiled MLIR→LLVM artifact whose correctness is
//! evidenced by the three-way differential (interp ≡ direct-LLVM ≡ MLIR-dialect) over the v0
//! calculus corpus (M-602); never `Proven` without a checked equivalence proof.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use core::fmt::Write as _;

use mycelium_core::lower::{self, Anf, AnfAlt, Rhs};
use mycelium_core::{Node, Payload, Repr, Trit};

fn repr_attr(repr: &Repr) -> String {
    match repr {
        Repr::Binary { width } => format!("binary<{width}>"),
        Repr::Ternary { trits } => format!("ternary<{trits}>"),
        Repr::Dense { dim, .. } => format!("dense<{dim}>"),
        Repr::Vsa { model, dim, .. } => format!("vsa<{model},{dim}>"),
        // RFC-0032 D3 (M-749): the indexed sequence renders its element attr and declared length.
        Repr::Seq { elem, len } => format!("seq<{},{len}>", repr_attr(elem)),
        // RFC-0032 D4 (M-750): the byte string carries no static type parameter.
        Repr::Bytes => "bytes".to_owned(),
        // ADR-040 (M-896): the scalar float renders its frozen width by name (F64-only today).
        // This is the *textual dump* only — float ops/codegen are M-898+; the native path keeps
        // refusing unsupported fragments explicitly (never a silent flatten).
        Repr::Float { .. } => "float<f64>".to_owned(),
    }
}

fn payload_attr(p: &Payload) -> String {
    match p {
        Payload::Bits(b) => b.iter().map(|&x| if x { '1' } else { '0' }).collect(),
        Payload::Trits(t) => t
            .iter()
            .map(|&x| match x {
                Trit::Neg => '-',
                Trit::Zero => '0',
                Trit::Pos => '+',
            })
            .collect(),
        Payload::Scalars(xs) => format!("{xs:?}"),
        Payload::Hypervector(xs) => format!("{xs:?}"),
        // RFC-0032 D3 (M-749): a sequence payload renders its elements' attrs, comma-joined.
        Payload::Seq(elems) => {
            let inner: Vec<String> = elems.iter().map(|e| payload_attr(e.payload())).collect();
            format!("[{}]", inner.join(","))
        }
        // RFC-0032 D4 (M-750): a byte payload renders as a lowercase-hex string.
        Payload::Bytes(bytes) => bytes.iter().map(|b| format!("{b:02x}")).collect(),
        // ADR-040 (M-896): shortest round-trip decimal (`{:?}`) — deterministic; the in-band
        // specials render as `inf`/`-inf`/`NaN` and `-0.0` keeps its sign (a faithful dump).
        Payload::Float(x) => format!("{x:?}"),
    }
}

/// Emit the textual `ternary`-dialect module for `node` (one op per lowered binding). Deterministic.
///
/// The data + recursion fragment (`Construct`/`App`/`Lam`/`Fix`/`Match`, M-342) renders as dialect
/// ops too, with body-bearing ops (closures, match arms) carrying their nested block as a textual
/// **region** — a faithful skeleton, not a silent flatten. This is the dumpable shape of the eventual
/// MLIR path; the executable path for these nodes is the `aot::run` env-machine.
#[must_use]
pub fn emit(node: &Node) -> String {
    let anf = lower::lower_to_anf(node);
    let mut s = String::from("module {\n  func.func @kernel() -> !myc.value {\n");
    emit_block(&anf, 2, &mut s);
    s.push_str("  }\n}");
    s
}

/// Emit one ANF block's ops + terminator at indent `depth` (in 2-space units). Recursive: nested
/// regions (closure/recursion bodies, match arms) emit at a deeper indent.
fn emit_block(anf: &Anf, depth: usize, s: &mut String) {
    let pad = "  ".repeat(depth);
    for b in anf.bindings() {
        let name = b.name.render();
        let layout = b
            .layout
            .map(|l| format!("  // layout = {l:?}"))
            .unwrap_or_default();
        let _ = write!(s, "{pad}{name} = ");
        emit_op(&b.rhs, depth, s);
        let _ = writeln!(s, "{layout}");
    }
    let _ = writeln!(
        s,
        "{pad}\"func.return\"({}) : (!myc.value) -> ()",
        anf.result().render()
    );
}

/// Emit one lowered RHS as a dialect op (no trailing newline). Body-bearing ops embed nested regions.
fn emit_op(rhs: &Rhs, depth: usize, s: &mut String) {
    let pad = "  ".repeat(depth);
    match rhs {
        Rhs::Const(v) => {
            let _ = write!(
                s,
                "\"ternary.const\"() {{repr = \"{}\", value = \"{}\", guarantee = \"{:?}\"}} : () -> !myc.value",
                repr_attr(v.repr()),
                payload_attr(v.payload()),
                v.meta().guarantee()
            );
        }
        Rhs::Alias(a) => {
            let _ = write!(
                s,
                "\"ternary.alias\"({}) : (!myc.value) -> !myc.value",
                a.render()
            );
        }
        Rhs::Op { prim, args } => {
            let operands: Vec<String> = args
                .iter()
                .map(mycelium_core::lower::Atom::render)
                .collect();
            let _ = write!(
                s,
                "\"ternary.op\"({}) {{prim = \"{prim}\"}} : ({}) -> !myc.value",
                operands.join(", "),
                vec!["!myc.value"; operands.len()].join(", ")
            );
        }
        Rhs::Swap {
            src,
            target,
            policy,
        } => {
            let _ = write!(
                s,
                "\"ternary.swap\"({}) {{target = \"{}\", policy = \"{}\"}} : (!myc.value) -> !myc.value",
                src.render(),
                repr_attr(target),
                policy.as_str()
            );
        }
        Rhs::Construct { ctor, args } => {
            let operands: Vec<String> = args
                .iter()
                .map(mycelium_core::lower::Atom::render)
                .collect();
            let _ = write!(
                s,
                "\"myc.construct\"({}) {{ctor = \"{ctor}\"}} : ({}) -> !myc.value",
                operands.join(", "),
                vec!["!myc.value"; operands.len()].join(", ")
            );
        }
        Rhs::App { func, arg } => {
            let _ = write!(
                s,
                "\"myc.app\"({}, {}) : (!myc.value, !myc.value) -> !myc.value",
                func.render(),
                arg.render()
            );
        }
        Rhs::Lam { param, body } => {
            let _ = writeln!(s, "\"myc.lam\"() ({{  // param = \"{param}\"");
            emit_block(body, depth + 1, s);
            let _ = write!(s, "{pad}}}) : () -> !myc.value");
        }
        Rhs::Fix { name, body } => {
            let _ = writeln!(s, "\"myc.fix\"() ({{  // self = \"{name}\"");
            emit_block(body, depth + 1, s);
            let _ = write!(s, "{pad}}}) : () -> !myc.value");
        }
        Rhs::FixGroup { defs, which } => {
            let _ = writeln!(s, "\"myc.fixgroup\"() ({{  // member = \"{which}\"");
            for (member, body) in defs {
                let _ = writeln!(s, "{pad}  // def \"{member}\"");
                emit_block(body, depth + 1, s);
            }
            let _ = write!(s, "{pad}}}) : () -> !myc.value");
        }
        Rhs::Match {
            scrutinee,
            alts,
            default,
        } => {
            let _ = writeln!(s, "\"myc.match\"({}) (", scrutinee.render());
            for alt in alts {
                match alt {
                    AnfAlt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        let _ = writeln!(s, "{pad}  {{  // alt {ctor} ({})", binders.join(" "));
                        emit_block(body, depth + 2, s);
                        let _ = writeln!(s, "{pad}  }},");
                    }
                    AnfAlt::Lit { value, body } => {
                        let _ =
                            writeln!(s, "{pad}  {{  // alt-lit {}", payload_attr(value.payload()));
                        emit_block(body, depth + 2, s);
                        let _ = writeln!(s, "{pad}  }},");
                    }
                }
            }
            match default {
                Some(d) => {
                    let _ = writeln!(s, "{pad}  {{  // default");
                    emit_block(d, depth + 2, s);
                    let _ = writeln!(s, "{pad}  }}");
                }
                None => {
                    let _ = writeln!(s, "{pad}  // no-default");
                }
            }
            let _ = write!(s, "{pad}) : (!myc.value) -> !myc.value");
        }
    }
}

#[cfg(feature = "mlir-dialect")]
pub mod native;
