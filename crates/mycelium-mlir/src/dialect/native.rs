//! The **real** ternary-dialect lowering (M-601; M-725; M-857; RFC-0004 §2; RFC-0029 §7;
//! ADR-009/ADR-019).
//!
//! Feature-gated (`mlir-dialect`, OFF by default). For the **bit/trit element-wise straight-line
//! fragment plus the balanced-ternary fixed-width arithmetic** — the additive carry chain
//! `trit.add`/`trit.sub` (M-725) and the shifted-accumulate multiply `trit.mul` (M-857) — this
//! emits a genuine MLIR module in the `func` + `arith` + `cf` dialects and drives it through the
//! verified libMLIR pipeline
//!
//! ```text
//! mlir-opt-<v> --convert-cf-to-llvm --convert-func-to-llvm --convert-arith-to-llvm
//!   --reconcile-unrealized-casts
//!   | mlir-translate-<v> --mlir-to-llvmir
//! ```
//!
//! to **real LLVM IR**, then `clang` → native executable → run → read-back. It is a fourth,
//! genuinely MLIR-compiled execution path (not the textual [`super::emit`] skeleton, not the
//! [`crate::llvm`] direct-LLVM emitter, not the [`crate::aot`] env-machine).
//!
//! **The fragment, and the moving honest boundary (RFC-0004 §2; M-725; M-857; M-856; VR-5/G2).**
//! RFC-0004 §2 sequences the AOT path as "`ternary` first … lowering progressively to
//! `linalg`/`vector`/`arith`". The element-wise bit/trit ops (`core.id`, `bit.not/and/or/xor`,
//! `trit.neg`) are the sub-fragment the **standard** `arith` dialect carries faithfully — one `arith`
//! op per element, every op dumpable. **M-725 widened this beyond element-wise** to the
//! balanced-ternary *additive* carry chain `trit.add`/`trit.sub`, and **M-857 widens it again to
//! balanced-ternary *multiply* `trit.mul`** — the shifted-accumulate / 2m-trit-buffer fragment. Both
//! lower through the real dialect path over `arith` ops: additive as a fixed-width ripple-carry
//! (`arith.addi`/`arith.remsi`/`arith.divsi`/`arith.subi`, digit-for-digit the same `s + 4 →
//! srem/sdiv 3 − 1` step the direct-LLVM path uses), and multiply as shifted accumulation of `±a`
//! (`arith.muli` per digit) into a 2m-trit buffer with the same shared ripple adder — overflow iff any
//! high trit is non-zero.
//!
//! **M-856 widens the fragment a third time, to the `Construct`/`Match` non-recursive data fragment
//! and the certified binary↔ternary `Swap`.** `Construct`/`Match` mirror the direct-LLVM
//! [`crate::llvm`] Increment-1 fragment (M-373; non-recursive, bounded — no `Fix`/`FixGroup` in
//! scope), but **without** the `alloca`/`getelementptr`/`load`/`store` indirection: because MLIR's
//! structured control flow needs no memory to thread a dominating value into a successor block (block
//! *arguments*, not `phi` nodes, and a value defined in a dominating predecessor is directly
//! referenceable in every block it dominates — verified empirically against `mlir-opt-18` for this
//! module), a constructed value's tag and fields stay plain SSA registers end to end: `Construct`
//! materializes the tag as an `arith.constant … : i64` and carries each field's already-lowered
//! [`Lane`] forward unchanged (no store), and `Match` dispatches on the tag with `cf.switch`,
//! binding each `Ctor` arm's fields by direct SSA reference (no load) and merging the arms' results
//! through a block-argument "phi" at the join point. Only the **`Ctor`-arm** form is covered (data
//! constructed by `Construct`); the narrow `Binary{8}`-packed **`Lit`-arm** form (the Increment-3
//! recursion *branch primitive* the direct-LLVM heap trampoline uses for its base case) stays an
//! explicit refusal here — it is tied to the `Fix`/`FixGroup` recursion fragment, which is out of
//! scope for this increment (flagged for a follow-up alongside Dense/VSA, M-856b). A fixed-width
//! arithmetic op *inside* a `Match` arm has its overflow flag folded **locally** to that arm and
//! re-exported through the merge block's own block argument (never a flag referencing a
//! non-dominating block — a real SSA-dominance hazard the direct-LLVM alloca/phi shape does not
//! share, so this module's design deliberately does not mirror it byte-for-byte; the *value*-level
//! contract — the same read-back protocol, the same overflow semantics — still matches
//! [`crate::llvm`] exactly).
//!
//! `Swap` covers the **certified binary↔ternary class** (RFC-0002 §4) plus same-`Repr` identity,
//! mirroring `crate::swap_codegen`'s algorithm — decode/encode via `arith.extui`/`arith.extsi`/
//! `arith.muli`/`arith.addi`/`arith.subi`/`arith.divsi`/`arith.remsi`/`arith.select`/`arith.trunci`/
//! `arith.shrui`/`arith.andi`/`arith.cmpi`, digit-for-digit the same accumulate/balanced-division
//! transcode `mycelium_core::binary`/`ternary` define (DRY at the *algorithm* level — the
//! direct-LLVM and MLIR-arith emitters are independent textual renderings of the *same* algorithm,
//! never a divergent second one) — always under the **`Recheck`** cert mode (the compile-time
//! independent bijection re-check; `crate::swap_codegen::legal_pair`/`MAX_BINARY_WIDTH_I64`/
//! `MAX_TERNARY_WIDTH_I64` are reused directly, single source of truth). The `ReuseInterp` opt-in mode
//! is not wired here (a small, explicitly deferred gap — the default/safer mode is what
//! [`compile_and_run`] uses). An out-of-range `dec` (`Ternary → Binary`) result and an illegal `(n,m)`
//! pair are refused exactly as `crate::swap_codegen` refuses them (an [`DialectError::Overflow`] read
//! back through the shared sentinel, and a compile-time [`DialectError::Unsupported`], respectively —
//! never silent).
//!
//! Out-of-range results (trit arithmetic *or* the `Swap` `dec` direction) are reported through the
//! **shared** [`crate::llvm::OVERFLOW_SENTINEL`] read-back (a `cf.cond_br` on the runtime overflow
//! flag) — never a silent wrap. The new honest boundary is everything **richer**: closures, recursion
//! (`App`/`Lam`/`Fix`/`FixGroup`), the `Lit`-arm branch primitive, and Dense/VSA (both as `Swap`
//! targets and as a `Repr` in the generic lowering) — each an **explicit, never-silent**
//! [`DialectError::Unsupported`] routing the program to the direct-LLVM backend ([`crate::llvm`]) or
//! the interpreter, which already cover the full v0 calculus. We still ship **no** divergent codegen
//! for the closure/recursion/Dense/VSA fragments here just to widen further. The honest boundary is an
//! explicit refusal, not silent or fragile output.
//!
//! **Read-back protocol — shared with [`crate::llvm`] (single contract).** The emitted `@main`
//! `putchar`s each result element's ASCII char (`'0'`/`'1'` for bits; `'-'`/`'0'`/`'+'` for trits)
//! then a newline, and the result is decoded by the **same** [`crate::llvm::decode_result`] the
//! direct-LLVM path uses. On an arithmetic overflow (`trit.add`/`trit.sub`/`trit.mul` leaving the
//! `m`-trit range) the artifact prints the **shared** [`crate::llvm::OVERFLOW_SENTINEL`] line instead,
//! decoded to an explicit [`DialectError::Overflow`] — byte-for-byte the same sentinel and meaning as
//! the direct-LLVM path's [`crate::llvm::AotError::Overflow`]. So the MLIR-dialect output and the
//! direct-LLVM output are read back identically — the three-way differential (M-602/M-725/M-857)
//! compares like with like, on the result *and* the overflow refusal.
//!
//! **Toolchain probing (skip-gracefully).** `mlir-opt`/`mlir-translate`/`clang` are probed at
//! runtime; absence is a graceful [`DialectError::ToolchainMissing`] (the caller skips, never
//! fails) — mirroring the `llc`/`clang` `ToolchainMissing` idiom in [`crate::llvm`]. So even
//! `cargo test --features mlir-dialect` is green on a box without libMLIR (ADR-019).
//!
//! **Guarantee tag:** `Empirical` — a real compiled artifact, correctness evidenced by the M-602
//! three-way differential over the corpus; never `Proven` without a checked equivalence proof
//! (VR-5).

use std::fmt;
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

use mycelium_core::lower::{self, Anf, AnfAlt, Atom, Rhs};
use mycelium_core::{Node, Payload, Repr, Trit, Value};

use crate::llvm::{decode_result, LaneKind, OVERFLOW_SENTINEL};

/// The dialect-native sibling of [`crate::dense_codegen`] (M-856b) — lowers `DenseProgram` (the
/// element-wise Dense fragment) to `arith`/`func`/`math`/`llvm` MLIR instead of direct LLVM IR text.
pub mod dense;
/// The dialect-native sibling of [`crate::vsa_codegen`] (M-856b) — lowers `VsaProgram` (the
/// MAP-I/BSC/HRR/FHRR real-`Vec<f64>` fragment) to `arith`/`func`/`math`/`llvm` MLIR.
pub mod vsa;

/// An explicit failure of the real MLIR-dialect path. Every unsupported construct, missing tool, or
/// subprocess failure is one of these — the path is **never silent** (G2). Mirrors the contract of
/// [`crate::llvm::AotError`], specialized to the MLIR pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DialectError {
    /// A node/prim/repr outside the fragment the standard `arith`/`func`/`cf` dialects lower here
    /// (the bit/trit element-wise ops **plus** the balanced-ternary fixed-width arithmetic
    /// `trit.add`/`trit.sub`; M-725, and `trit.mul`; M-857). The message names what was refused and
    /// where it should run instead (the direct-LLVM backend [`crate::llvm`] or the interpreter) — an
    /// `EXPLAIN`-able routing, never a silent drop (G2/VR-5).
    Unsupported(String),
    /// A balanced-ternary arithmetic result (additive `trit.add`/`trit.sub`; M-725, or multiplicative
    /// `trit.mul`; M-857) left the fixed `m`-trit range — the MLIR-compiled artifact computed the
    /// overflow at runtime and signalled it through the shared read-back protocol (the
    /// [`crate::llvm::OVERFLOW_SENTINEL`] line). Surfaced as an explicit error mirroring
    /// [`crate::llvm::AotError::Overflow`] and the interpreter's `EvalError::Overflow` — never a
    /// silent wrap (SC-3/G2). So the three-way differential stays honest on overflow too.
    Overflow(String),
    /// An operand atom with no prior binding (an ill-formed lowering — should not occur for a
    /// well-formed ANF program; surfaced explicitly rather than panicking).
    FreeVariable(String),
    /// The MLIR toolchain (`mlir-opt-<v>` / `mlir-translate-<v>`) or `clang` is not installed —
    /// callers should **skip**, not fail (the house "skip gracefully when a tool is absent" idiom;
    /// ADR-019). Carries the missing tool name.
    ToolchainMissing(String),
    /// A pipeline stage (`mlir-opt`, `mlir-translate`, or `clang`) ran but returned a non-zero
    /// status. Carries the stage name + captured stderr (no opaque failure).
    Compile(String),
    /// The compiled artifact failed to run or produced unreadable output.
    Run(String),
    /// The native stdout did not parse back into the expected payload shape.
    Parse(String),
    /// Reconstructing the result [`Value`] failed its well-formedness check.
    Wf(String),
}

impl fmt::Display for DialectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DialectError::Unsupported(m) => {
                write!(f, "unsupported for the MLIR-dialect fragment: {m}")
            }
            DialectError::Overflow(m) => write!(f, "balanced-ternary overflow: {m}"),
            DialectError::FreeVariable(v) => write!(f, "free variable in lowered IR: {v}"),
            DialectError::ToolchainMissing(t) => write!(f, "MLIR toolchain missing: {t}"),
            DialectError::Compile(e) => write!(f, "MLIR pipeline compile failed: {e}"),
            DialectError::Run(e) => write!(f, "MLIR artifact run failed: {e}"),
            DialectError::Parse(e) => write!(f, "MLIR artifact output parse failed: {e}"),
            DialectError::Wf(e) => write!(f, "result not well-formed: {e}"),
        }
    }
}

impl std::error::Error for DialectError {}

/// The representation kind of a lowered result lane — the **public** shape descriptor for the
/// MLIR-dialect path (`Binary{w}` or `Ternary{m}`). Mirrors the internal `crate::llvm::LaneKind`
/// (which is `pub(crate)`); kept distinct so the public API does not leak a crate-private type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultKind {
    /// `Binary{w}` — elements in `{0, 1}`, printed `'0'`/`'1'`.
    Binary,
    /// `Ternary{m}` — balanced-ternary elements in `{-1, 0, 1}`, printed `'-'`/`'0'`/`'+'`.
    Ternary,
}

impl ResultKind {
    fn from_lane(k: LaneKind) -> Self {
        match k {
            LaneKind::Binary => ResultKind::Binary,
            LaneKind::Ternary => ResultKind::Ternary,
        }
    }
    fn to_lane(self) -> LaneKind {
        match self {
            ResultKind::Binary => LaneKind::Binary,
            ResultKind::Ternary => LaneKind::Ternary,
        }
    }
}

/// The resolved MLIR toolchain: the `mlir-opt`/`mlir-translate` binary names (version-matched to the
/// installed LLVM major) plus `clang`. Produced by [`resolve_tools`]; inspectable (no hidden tool
/// choice — the resolved binaries are queryable for `EXPLAIN`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirTools {
    /// The `mlir-opt` binary name, e.g. `mlir-opt-18`.
    pub mlir_opt: String,
    /// The `mlir-translate` binary name, e.g. `mlir-translate-18`.
    pub mlir_translate: String,
    /// The `clang` binary name (`clang-<v>` if present, else `clang`).
    pub clang: String,
    /// The detected LLVM major version the tools are matched to.
    pub llvm_major: u32,
}

impl MlirTools {
    /// Whether the MLIR toolchain resolves in this environment (a convenience over
    /// [`resolve_tools`] for tests/harnesses that want to assert non-vacuous coverage). `true` iff
    /// [`resolve_tools`] succeeds.
    #[must_use]
    pub fn is_available() -> bool {
        resolve_tools().is_ok()
    }
}

/// Parse the LLVM major version from a `--version` banner — either an `… LLVM version NN.…` line
/// (`llc`, `mlir-opt`, `mlir-translate`) **or** a `clang version NN.…` line (`clang`, which does not
/// print "LLVM version"). Returns `None` when no recognized banner is present.
fn parse_llvm_major(s: &str) -> Option<u32> {
    for line in s.lines() {
        for marker in ["LLVM version", "clang version"] {
            if let Some(idx) = line.find(marker) {
                let rest = &line[idx + marker.len()..];
                let tok: String = rest
                    .trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(major) = tok.parse::<u32>() {
                    return Some(major);
                }
            }
        }
    }
    None
}

/// Detect the installed LLVM major version from `llc --version`, falling back to `clang --version`.
/// Returns `None` when neither tool is present or the version line cannot be parsed — the caller
/// turns that into a graceful skip.
fn detect_llvm_major() -> Option<u32> {
    for tool in ["llc", "clang"] {
        if let Ok(out) = Command::new(tool).arg("--version").output() {
            if let Some(major) = parse_llvm_major(&String::from_utf8_lossy(&out.stdout)) {
                return Some(major);
            }
        }
    }
    None
}

/// Probe whether a binary exists + responds to `--version`.
fn tool_present(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The LLVM major a tool reports via `--version`, or `None` if it's absent/unparsable. Used to
/// confirm an *unversioned* fallback binary actually matches the detected LLVM major before it is
/// accepted — never a silent mismatched substitution (G2).
fn tool_major(name: &str) -> Option<u32> {
    let out = Command::new(name).arg("--version").output().ok()?;
    parse_llvm_major(&String::from_utf8_lossy(&out.stdout))
}

/// Resolve the MLIR toolchain, version-matched to the installed LLVM major.
///
/// Tries the versioned binaries first (`mlir-opt-<major>`, `mlir-translate-<major>` — how the distro
/// packages them; ADR-019), then the unversioned fallbacks (`mlir-opt`, `mlir-translate`). Returns
/// [`DialectError::ToolchainMissing`] (a *skip*, not a failure) when the LLVM major cannot be
/// detected or a required tool is absent. **Never** silently substitutes a mismatched-version tool
/// (G2; no silent toolchain bump — CLAUDE.md).
pub fn resolve_tools() -> Result<MlirTools, DialectError> {
    let major = detect_llvm_major().ok_or_else(|| {
        DialectError::ToolchainMissing("llc/clang (LLVM version undetectable)".to_owned())
    })?;

    let opt_versioned = format!("mlir-opt-{major}");
    let tr_versioned = format!("mlir-translate-{major}");
    // Versioned binary first; otherwise an unversioned fallback ONLY when its own `--version`
    // reports the same major (never silently substitute a mismatched toolchain — G2).
    let mlir_opt = if tool_present(&opt_versioned) {
        opt_versioned
    } else if tool_major("mlir-opt") == Some(major) {
        "mlir-opt".to_owned()
    } else {
        return Err(DialectError::ToolchainMissing(format!(
            "mlir-opt-{major} (unversioned `mlir-opt` absent or a different LLVM major — never \
             silently substituted, G2) — run scripts/setup-mlir.sh"
        )));
    };
    let mlir_translate = if tool_present(&tr_versioned) {
        tr_versioned
    } else if tool_major("mlir-translate") == Some(major) {
        "mlir-translate".to_owned()
    } else {
        return Err(DialectError::ToolchainMissing(format!(
            "mlir-translate-{major} (unversioned `mlir-translate` absent or a different LLVM major \
             — never silently substituted, G2) — run scripts/setup-mlir.sh"
        )));
    };
    let clang_versioned = format!("clang-{major}");
    let clang = if tool_present(&clang_versioned) {
        clang_versioned
    } else if tool_major("clang") == Some(major) {
        "clang".to_owned()
    } else {
        return Err(DialectError::ToolchainMissing(format!(
            "clang-{major} (unversioned `clang` absent or a different LLVM major — never silently \
             substituted, G2)"
        )));
    };

    Ok(MlirTools {
        mlir_opt,
        mlir_translate,
        clang,
        llvm_major: major,
    })
}

// ─── SSA naming for the emitted MLIR ──────────────────────────────────────────────────────────

/// A monotone counter minting fresh MLIR SSA names (`%v0`, `%v1`, …). MLIR SSA values are textual
/// `%name`s exactly like LLVM, so the emitter mirrors [`crate::llvm`]'s `Ssa` shape.
#[derive(Default)]
struct Ssa(usize);
impl Ssa {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("%v{n}")
    }
}

/// A computed value lane in the MLIR body: its representation kind and one `i32`-typed SSA operand
/// (or `i32` literal `arith.constant`) per element. The element model is **identical** to
/// [`crate::llvm`]'s `Lane` (Binary in `{0,1}`, Ternary in `{-1,0,1}`), so the read-back is shared.
#[derive(Debug, Clone)]
struct Lane {
    kind: LaneKind,
    /// SSA names of `i32` values, one per element.
    vals: Vec<String>,
}

/// A monotone counter minting fresh MLIR block labels (`bb0`, `bb1`, …; written `^bb0` at use sites).
/// A separate namespace from [`Ssa`] (labels start `^`, values `%`), kept as its own counter for
/// readability — mirrors [`crate::llvm`]'s `Bbc`. M-856 (`Match` dispatch).
#[derive(Default)]
struct Bbc(usize);
impl Bbc {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("bb{n}")
    }
}

/// A constructed data value in the lowered env (M-856; mirrors the *value*-level contract of
/// [`crate::llvm::Datum`], not its physical layout). Produced by `Rhs::Construct`, consumed by a
/// `Rhs::Match` `Ctor` arm.
///
/// **No memory.** Unlike the direct-LLVM `Datum` (a stack `alloca`'d `[N+1 x i64]` struct read back
/// with `getelementptr`/`load`), this carries the tag and every field as **plain SSA values**: MLIR's
/// structured control flow needs no memory to thread a dominating value into a successor block (a
/// value defined in a dominating predecessor block is directly referenceable in every block it
/// dominates — verified against `mlir-opt-18` for this module's shape), so `Construct` never stores
/// and `Match` never loads. The tag is additionally materialized as an `i64` SSA constant (the
/// `cf.switch` discriminant); `_tag` is kept for auditability, mirroring `crate::llvm::Datum::_tag`.
#[derive(Debug, Clone)]
struct Datum {
    /// The constructor tag (`ctor.index()`), retained for auditability.
    _tag: u64,
    /// The tag materialized as an `arith.constant … : i64` SSA name — the `cf.switch` discriminant.
    tag_ssa: String,
    /// Each field's already-lowered lane, in declaration order.
    fields: Vec<Lane>,
}

/// An environment value for the M-856 lowering: either a repr lane (bit/trit) or a constructed
/// [`Datum`]. Mirrors [`crate::llvm::EnvValue`]'s `Repr`/`Datum` variants (closures/Fix/FixGroup are
/// not represented here — `App`/`Lam`/`Fix`/`FixGroup` stay explicit refusals in this fragment).
#[derive(Debug, Clone)]
enum EnvValue {
    Repr(Lane),
    Datum(Datum),
}

impl EnvValue {
    /// Extract the repr lane (cloned), or an explicit refusal if this is a datum (G2 — never a
    /// silent type confusion).
    fn as_lane(&self, ctx: &str) -> Result<Lane, DialectError> {
        match self {
            EnvValue::Repr(l) => Ok(l.clone()),
            EnvValue::Datum(_) => Err(DialectError::Unsupported(format!(
                "{ctx}: expected a repr lane but found a constructed datum (M-856)"
            ))),
        }
    }
}

/// Emit an `arith.constant` for one `i32` element value, returning its SSA name.
fn emit_const_i32(v: i32, ssa: &mut Ssa, body: &mut String) -> String {
    let r = ssa.fresh();
    let _ = writeln!(body, "    {r} = arith.constant {v} : i32");
    r
}

/// Emit an `arith.constant` for one `i64` value, returning its SSA name. M-856 (`Construct` tags,
/// `Swap` transcode arithmetic).
fn emit_const_i64(v: i64, ssa: &mut Ssa, body: &mut String) -> String {
    let r = ssa.fresh();
    let _ = writeln!(body, "    {r} = arith.constant {v} : i64");
    r
}

/// Materialize a constant `Value`'s elements as `arith.constant` SSA values (the entry point for a
/// `Rhs::Const`). Refuses Dense/VSA reprs explicitly (they are not in the element-wise fragment).
fn const_lane(v: &Value, ssa: &mut Ssa, body: &mut String) -> Result<Lane, DialectError> {
    match (v.repr(), v.payload()) {
        (Repr::Binary { .. }, Payload::Bits(b)) => {
            let vals = b
                .iter()
                .map(|&x| emit_const_i32(i32::from(x), ssa, body))
                .collect();
            Ok(Lane {
                kind: LaneKind::Binary,
                vals,
            })
        }
        (Repr::Ternary { .. }, Payload::Trits(t)) => {
            let vals = t
                .iter()
                .map(|&x| {
                    let e = match x {
                        Trit::Neg => -1,
                        Trit::Zero => 0,
                        Trit::Pos => 1,
                    };
                    emit_const_i32(e, ssa, body)
                })
                .collect();
            Ok(Lane {
                kind: LaneKind::Ternary,
                vals,
            })
        }
        (repr, _) => Err(DialectError::Unsupported(format!(
            "repr {repr:?} is not in the element-wise dialect fragment (Dense/VSA stay on the \
             interpreter / direct-LLVM path)"
        ))),
    }
}

/// Require a lane to be of the expected kind, else an explicit refusal (a `bit.*` op on a ternary
/// lane, or `trit.*` on a binary one, is a type error — never silently mis-lowered; G2).
fn require_kind(prim: &str, got: LaneKind, want: LaneKind) -> Result<(), DialectError> {
    if got == want {
        Ok(())
    } else {
        Err(DialectError::Unsupported(format!(
            "{prim} expects a {want:?} operand, got {got:?}"
        )))
    }
}

/// Map a unary `arith` op over a lane's elements (one op per element — dumpable, no opaque pass),
/// `mk` rendering the op line for element SSA `x` into result SSA `r`.
fn map1(a: &Lane, ssa: &mut Ssa, body: &mut String, mk: impl Fn(&str, &str) -> String) -> Lane {
    let vals = a
        .vals
        .iter()
        .map(|x| {
            let r = ssa.fresh();
            let _ = writeln!(body, "{}", mk(x, &r));
            r
        })
        .collect();
    Lane { kind: a.kind, vals }
}

/// Map a binary `arith` op over two equal-width lanes' elements, `mk` rendering the op line for
/// element SSAs `x`,`y` into result SSA `r`. Width mismatch is an explicit refusal (G2).
fn map2(
    prim: &str,
    a: &Lane,
    b: &Lane,
    ssa: &mut Ssa,
    body: &mut String,
    mk: impl Fn(&str, &str, &str) -> String,
) -> Result<Lane, DialectError> {
    if a.vals.len() != b.vals.len() {
        return Err(DialectError::Unsupported(format!(
            "{prim}: width mismatch {} vs {}",
            a.vals.len(),
            b.vals.len()
        )));
    }
    let vals = a
        .vals
        .iter()
        .zip(&b.vals)
        .map(|(x, y)| {
            let r = ssa.fresh();
            let _ = writeln!(body, "{}", mk(x, y, &r));
            r
        })
        .collect();
    Ok(Lane { kind: a.kind, vals })
}

/// Lower one primitive over its operand lanes to `arith` ops, returning the result lane. Covers the
/// element-wise fragment **and** the balanced-ternary fixed-width arithmetic — the additive carry
/// chain `trit.add`/`trit.sub` (M-725) and the shifted-accumulate multiply `trit.mul` (M-857) — all of
/// which push their runtime overflow `i1` SSA name(s) onto `flags` (the caller folds them into the
/// program-level overflow flag that drives the read-back). The new honest boundary is everything
/// **richer** (the data fragment, closures, recursion, `Swap`, Dense/VSA): an explicit
/// [`DialectError::Unsupported`] routing to [`crate::llvm`] / the interpreter (G2). The carry *step*
/// re-emits `mycelium_core::ternary::add_with_carry` and the multiply re-emits
/// `mycelium_core::ternary::mul` digit-for-digit (one source of truth per algorithm, never a divergent
/// second algorithm — DRY).
fn emit_op(
    prim: &str,
    operands: &[&Lane],
    ssa: &mut Ssa,
    body: &mut String,
    flags: &mut Vec<String>,
) -> Result<Lane, DialectError> {
    let arity1 = |p: &str| -> Result<&Lane, DialectError> {
        match operands {
            [a] => Ok(*a),
            _ => Err(DialectError::Unsupported(format!(
                "{p} expects 1 operand, got {}",
                operands.len()
            ))),
        }
    };
    let arity2 = |p: &str| -> Result<(&Lane, &Lane), DialectError> {
        match operands {
            [a, b] => Ok((*a, *b)),
            _ => Err(DialectError::Unsupported(format!(
                "{p} expects 2 operands, got {}",
                operands.len()
            ))),
        }
    };
    match prim {
        // Identity passes the lane through unchanged, any kind.
        "core.id" => Ok(arity1(prim)?.clone()),
        // bit.not x = xor(x, 1) per bit.
        "bit.not" => {
            let a = arity1(prim)?;
            require_kind(prim, a.kind, LaneKind::Binary)?;
            let one = emit_const_i32(1, ssa, body);
            Ok(map1(a, ssa, body, |x, r| {
                format!("    {r} = arith.xori {x}, {one} : i32")
            }))
        }
        "bit.and" | "bit.or" | "bit.xor" => {
            let (a, b) = arity2(prim)?;
            require_kind(prim, a.kind, LaneKind::Binary)?;
            require_kind(prim, b.kind, LaneKind::Binary)?;
            let op = match prim {
                "bit.and" => "arith.andi",
                "bit.or" => "arith.ori",
                _ => "arith.xori",
            };
            map2(prim, a, b, ssa, body, |x, y, r| {
                format!("    {r} = {op} {x}, {y} : i32")
            })
        }
        // Balanced-ternary negation is digit-wise (`-t`), exact, no carry — `0 - x` per trit.
        "trit.neg" => {
            let a = arity1(prim)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            let zero = emit_const_i32(0, ssa, body);
            Ok(map1(a, ssa, body, |x, r| {
                format!("    {r} = arith.subi {zero}, {x} : i32")
            }))
        }
        // Balanced-ternary addition (M-725): a fixed-width ripple-carry over the trits (LSB→MSB),
        // with a runtime overflow `i1` (non-zero final carry ⇒ out of m-trit range). Mirrors
        // `mycelium_core::ternary::add` digit-for-digit — the same `s + 4 → srem/sdiv 3 − 1` step
        // the direct-LLVM path (`crate::llvm::emit_trit_add`) emits, re-expressed in `arith` ops.
        "trit.add" => {
            let (a, b) = arity2(prim)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            require_kind(prim, b.kind, LaneKind::Ternary)?;
            require_width(prim, a, b)?;
            let (lane, ovf) = emit_trit_add(&a.vals, &b.vals, ssa, body);
            flags.push(ovf);
            Ok(lane)
        }
        // Subtraction `a − b` = `add(a, neg(b))`: negate `b`'s trits, then the same ripple adder
        // (exactly `crate::llvm`'s `trit.sub`; DRY).
        "trit.sub" => {
            let (a, b) = arity2(prim)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            require_kind(prim, b.kind, LaneKind::Ternary)?;
            require_width(prim, a, b)?;
            let zero = emit_const_i32(0, ssa, body);
            let neg_b = map1(b, ssa, body, |x, r| {
                format!("    {r} = arith.subi {zero}, {x} : i32")
            });
            let (lane, ovf) = emit_trit_add(&a.vals, &neg_b.vals, ssa, body);
            flags.push(ovf);
            Ok(lane)
        }
        // Balanced-ternary multiplication (M-857): shifted accumulation of `±a` into a 2m-trit
        // buffer (mirrors `mycelium_core::ternary::mul` and the direct-LLVM `crate::llvm::emit_trit_mul`
        // digit-for-digit), then overflow iff any high trit is non-zero. Each `b` digit scales `a` by
        // an `arith.muli` (the digit is ±1/0, so this is exactly ±a / 0 per position). Re-expresses the
        // *same* algorithm in `arith` ops — not a divergent second codegen (DRY: the carry step is the
        // shared `emit_trit_add_step`). The per-accumulation carries plus the high trits become runtime
        // overflow `i1` flags folded into the program-level read-back (never a silent wrap — G2).
        "trit.mul" => {
            let (a, b) = arity2(prim)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            require_kind(prim, b.kind, LaneKind::Ternary)?;
            require_width(prim, a, b)?;
            let (lane, ovfs) = emit_trit_mul(&a.vals, &b.vals, ssa, body);
            flags.extend(ovfs);
            Ok(lane)
        }
        other => Err(DialectError::Unsupported(format!(
            "primitive {other:?} is not in the MLIR-dialect fragment (bit.not/and/or/xor, trit.neg, \
             trit.add, trit.sub, trit.mul, core.id) — it runs on the direct-LLVM backend / interpreter"
        ))),
    }
}

/// Require two lanes to have equal element count, else an explicit width-mismatch refusal (G2).
/// Mirrors [`crate::llvm`]'s `require_width`.
fn require_width(prim: &str, a: &Lane, b: &Lane) -> Result<(), DialectError> {
    if a.vals.len() == b.vals.len() {
        Ok(())
    } else {
        Err(DialectError::Unsupported(format!(
            "{prim}: width mismatch {} vs {}",
            a.vals.len(),
            b.vals.len()
        )))
    }
}

/// Emit a fixed-width balanced-ternary ripple-carry add over MSB-first trit operands `a`/`b` (equal
/// length, caller-checked) in `arith` ops. Returns the sum lane (MSB-first) and the SSA name of an
/// `i1` register set iff the final carry is non-zero (overflow). Digit-for-digit identical to
/// [`crate::llvm`]'s `emit_trit_add` (and thus to `mycelium_core::ternary::add`): with
/// `x = aᵢ + bᵢ + carry + 4` (always ≥ 1 so `arith.remsi`/`arith.divsi` are euclidean), the balanced
/// digit is `x remsi 3 − 1` and the next carry is `x divsi 3 − 1`.
fn emit_trit_add(a: &[String], b: &[String], ssa: &mut Ssa, body: &mut String) -> (Lane, String) {
    let m = a.len();
    // The incoming carry of the LSB step is the constant 0 trit.
    let mut carry = emit_const_i32(0, ssa, body);
    let mut sum_lsb: Vec<String> = Vec::with_capacity(m);
    // Process least-significant first (the tail of the MSB-first strings).
    for i in (0..m).rev() {
        let (digit, next_carry) = emit_trit_add_step(&a[i], &b[i], &carry, ssa, body);
        sum_lsb.push(digit);
        carry = next_carry;
    }
    // Overflow iff the final carry out of the most-significant trit is non-zero (an `i1`).
    let zero = emit_const_i32(0, ssa, body);
    let ovf = ssa.fresh();
    let _ = writeln!(body, "    {ovf} = arith.cmpi ne, {carry}, {zero} : i32");
    let vals: Vec<String> = sum_lsb.into_iter().rev().collect(); // back to MSB-first
    (
        Lane {
            kind: LaneKind::Ternary,
            vals,
        },
        ovf,
    )
}

/// One balanced-ternary add step in `arith`: given operand trits `a`/`b` and the incoming `carry`
/// (all `i32` SSA in `{−1,0,1}`), emit the balanced digit + outgoing carry. Returns
/// `(digit_reg, carry_reg)`. Byte-for-byte the `arith` analogue of [`crate::llvm`]'s
/// `emit_trit_add_step` (the single shared carry primitive — DRY).
fn emit_trit_add_step(
    a: &str,
    b: &str,
    carry: &str,
    ssa: &mut Ssa,
    body: &mut String,
) -> (String, String) {
    let four = emit_const_i32(4, ssa, body);
    let three = emit_const_i32(3, ssa, body);
    let one = emit_const_i32(1, ssa, body);
    let s1 = ssa.fresh();
    let _ = writeln!(body, "    {s1} = arith.addi {a}, {b} : i32");
    let s2 = ssa.fresh();
    let _ = writeln!(body, "    {s2} = arith.addi {s1}, {carry} : i32");
    // x = s + 4 ∈ [1,7], strictly positive ⇒ remsi/divsi coincide with euclidean rem/div by 3.
    let x = ssa.fresh();
    let _ = writeln!(body, "    {x} = arith.addi {s2}, {four} : i32");
    let rem = ssa.fresh();
    let _ = writeln!(body, "    {rem} = arith.remsi {x}, {three} : i32");
    let digit = ssa.fresh();
    let _ = writeln!(body, "    {digit} = arith.subi {rem}, {one} : i32");
    let q = ssa.fresh();
    let _ = writeln!(body, "    {q} = arith.divsi {x}, {three} : i32");
    let next_carry = ssa.fresh();
    let _ = writeln!(body, "    {next_carry} = arith.subi {q}, {one} : i32");
    (digit, next_carry)
}

/// Emit fixed-width balanced-ternary multiplication over MSB-first trit operands `a`/`b` (equal
/// length, caller-checked) in `arith` ops. Mirrors `mycelium_core::ternary::mul` and the direct-LLVM
/// [`crate::llvm`]'s `emit_trit_mul` digit-for-digit: shifted accumulation of `±a` into a 2m-trit
/// buffer, returning the low `m` trits (MSB-first) and the overflow `i1` flags — one per non-zero high
/// trit, plus each accumulation's carry (provably zero, OR-ed in as an honest net so a codegen slip can
/// never pass silently — G2). Each `b` digit (∈ `{−1,0,1}`) scales `a` by an `arith.muli`, so
/// `aⱼ · bₖ` is exactly `±aⱼ / 0` per position — the per-digit factor, no carry yet; the carries are
/// resolved by the shared ripple adder ([`emit_ripple_add_lsb`], built on the shared
/// `emit_trit_add_step` — DRY, never a divergent second algorithm).
fn emit_trit_mul(
    a: &[String],
    b: &[String],
    ssa: &mut Ssa,
    body: &mut String,
) -> (Lane, Vec<String>) {
    let m = a.len();
    if m == 0 {
        return (
            Lane {
                kind: LaneKind::Ternary,
                vals: Vec::new(),
            },
            Vec::new(),
        );
    }
    let wide = 2 * m;
    // LSB-first views of the operands and a 2m-wide accumulator initialised to the zero-trit SSA.
    // (MLIR operands must be SSA values — unlike LLVM IR text, a literal `0` is not a valid operand,
    // so every buffer slot is a materialised `arith.constant 0`.)
    let a_lsb: Vec<&String> = a.iter().rev().collect();
    let b_lsb: Vec<&String> = b.iter().rev().collect();
    let zero = emit_const_i32(0, ssa, body);
    let mut acc: Vec<String> = vec![zero.clone(); wide];
    let mut flags: Vec<String> = Vec::new();

    for (k, &bk) in b_lsb.iter().enumerate() {
        // Partial = (a scaled by digit bk) shifted left by k, in a 2m-wide LSB-first buffer. The
        // digit is ±1/0, so `aⱼ * bk` is exactly ±aⱼ / 0 — the per-digit factor, no carry yet.
        let mut partial: Vec<String> = vec![zero.clone(); wide];
        for (j, &aj) in a_lsb.iter().enumerate() {
            let p = ssa.fresh();
            let _ = writeln!(body, "    {p} = arith.muli {aj}, {bk} : i32");
            partial[k + j] = p;
        }
        let (next_acc, carry) = emit_ripple_add_lsb(&acc, &partial, ssa, body);
        acc = next_acc;
        // The 2m-wide sum cannot truly overflow for m-trit operands; OR the carry in anyway so a
        // codegen slip can never pass silently (honest net, never a fabricated guarantee — G2).
        let zc = emit_const_i32(0, ssa, body);
        let c = ssa.fresh();
        let _ = writeln!(body, "    {c} = arith.cmpi ne, {carry}, {zc} : i32");
        flags.push(c);
    }
    // The product fits in m trits iff the high half (positions [m, 2m)) is all zero.
    for hi in &acc[m..] {
        let zh = emit_const_i32(0, ssa, body);
        let f = ssa.fresh();
        let _ = writeln!(body, "    {f} = arith.cmpi ne, {hi}, {zh} : i32");
        flags.push(f);
    }
    let vals: Vec<String> = acc[..m].iter().rev().cloned().collect(); // low m trits, MSB-first
    (
        Lane {
            kind: LaneKind::Ternary,
            vals,
        },
        flags,
    )
}

/// Ripple-carry add over two equal-length **LSB-first** trit-operand vectors in `arith` ops. Returns
/// the sum (LSB-first) and the final carry SSA register. The shared inner adder for [`emit_trit_mul`],
/// built on the shared [`emit_trit_add_step`] (the single carry primitive — DRY). The incoming carry of
/// the LSB step is the constant 0 trit.
fn emit_ripple_add_lsb(
    a: &[String],
    b: &[String],
    ssa: &mut Ssa,
    body: &mut String,
) -> (Vec<String>, String) {
    let mut carry = emit_const_i32(0, ssa, body);
    let mut sum: Vec<String> = Vec::with_capacity(a.len());
    for (ai, bi) in a.iter().zip(b) {
        let (digit, next_carry) = emit_trit_add_step(ai, bi, &carry, ssa, body);
        sum.push(digit);
        carry = next_carry;
    }
    (sum, carry)
}

/// Fold a list of `i1` overflow flags into one (`arith.ori` chain), or `None` if empty.
/// Deterministic. Mirrors [`crate::llvm`]'s `fold_or`.
fn fold_or(flags: &[String], ssa: &mut Ssa, body: &mut String) -> Option<String> {
    let mut it = flags.iter();
    let mut acc = it.next()?.clone();
    for f in it {
        let r = ssa.fresh();
        let _ = writeln!(body, "    {r} = arith.ori {acc}, {f} : i1");
        acc = r;
    }
    Some(acc)
}

/// The source [`Repr`] a lowered [`Lane`] denotes (mirrors `crate::llvm::lane_repr`). Used to
/// reconstruct a `Swap`'s *source* repr from its already-lowered operand (M-856).
fn lane_repr(lane: &Lane) -> Repr {
    match lane.kind {
        LaneKind::Binary => Repr::Binary {
            width: lane.vals.len() as u32,
        },
        LaneKind::Ternary => Repr::Ternary {
            trits: lane.vals.len() as u32,
        },
    }
}

/// Walk the lowered ANF, emitting one `arith` op per binding into `@main`'s body, and return the
/// result lane **plus** the program-level overflow flag (`Some(i1)` iff any fixed-width arithmetic
/// binding — `trit.add`/`trit.sub` (M-725), `trit.mul` (M-857), a `Swap` `dec`/illegal-final-quotient
/// check, or a `Match` whose arms carry one (M-856) — can overflow/be-out-of-range at runtime, else
/// `None`). Returns an explicit [`DialectError::Unsupported`] for any node outside the fragment —
/// routing the program to the direct-LLVM backend / interpreter (G2). `uses_abort` is set when a
/// `Match` with no default arm lowers a defined trap (so [`emit_mlir`] declares `@abort` only when
/// it is actually called).
fn lower_program(
    node: &Node,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    uses_abort: &mut bool,
) -> Result<(Lane, Option<String>), DialectError> {
    let anf = lower::lower_to_anf(node);
    let mut env: std::collections::HashMap<Atom, EnvValue> = std::collections::HashMap::new();
    lower_block(&anf, &mut env, ssa, bbc, body, uses_abort)
}

/// Lower one ANF block (its bindings + result) into MLIR ops, returning the result lane and the
/// folded **local-to-this-block** overflow flag (`None` when no op in *this* block can
/// overflow/be-out-of-range — so an overflow-free, data-free program emits exactly the M-601
/// module). Recursively callable for `Match` arm/default bodies (M-856): each call gets its own
/// local `flags` list, folded at its end — a nested block's flag is exported to its caller only
/// through a dominating SSA value (a `Match`'s merge-block argument), never left dangling across a
/// non-dominating branch (a real SSA-dominance hazard; see the module doc comment).
fn lower_block(
    anf: &Anf,
    env: &mut std::collections::HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    uses_abort: &mut bool,
) -> Result<(Lane, Option<String>), DialectError> {
    // The per-op overflow `i1` registers accumulated across *this* block only (M-856: a `Match`'s
    // arm/default bodies each fold their own list independently — see the function doc comment).
    // Any fixed-width arithmetic op (`trit.add`/`trit.sub`; M-725, `trit.mul`; M-857) or `Swap`
    // range check (M-856) pushes its overflow condition here; the interpreter errors on the *first*
    // overflow, so the native path being conservative (OR of all of them ⇒ one explicit overflow)
    // gives the same verdict — the meaningless result is never read either way. Mirrors
    // `crate::llvm`'s flags.
    let mut flags: Vec<String> = Vec::new();
    let lookup_ev = |env: &std::collections::HashMap<Atom, EnvValue>,
                     a: &Atom|
     -> Result<EnvValue, DialectError> {
        env.get(a)
            .cloned()
            .ok_or_else(|| DialectError::FreeVariable(a.render()))
    };

    for b in anf.bindings() {
        let ev = match &b.rhs {
            Rhs::Const(v) => EnvValue::Repr(const_lane(v, ssa, body)?),
            Rhs::Alias(a) => lookup_ev(env, a)?,
            Rhs::Op { prim, args } => {
                let operands: Vec<Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("op operand"))
                    .collect::<Result<_, _>>()?;
                let refs: Vec<&Lane> = operands.iter().collect();
                EnvValue::Repr(emit_op(prim, &refs, ssa, body, &mut flags)?)
            }
            // M-856: the certified binary↔ternary `Swap` (RFC-0002 §4) + same-Repr identity, always
            // under the `Recheck` cert mode (the compile-time independent bijection re-check).
            // Dense/VSA and other swap kinds stay an explicit refusal (below).
            Rhs::Swap { src, target, .. } => {
                let src_lane = lookup_ev(env, src)?.as_lane("swap source")?;
                let src_repr = lane_repr(&src_lane);
                let (lane, ovf) = lower_swap_dialect(&src_lane, &src_repr, target, ssa, body)?;
                if let Some(f) = ovf {
                    flags.push(f);
                }
                EnvValue::Repr(lane)
            }
            // M-856: the non-recursive `Construct`/`Match` data fragment (mirrors the direct-LLVM
            // Increment-1 fragment; M-373). `Construct` never emits IR beyond the tag constant — the
            // field lanes are already-lowered SSA values, carried forward unchanged (no memory).
            Rhs::Construct { ctor, args } => {
                let fields: Vec<Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("Construct field"))
                    .collect::<Result<_, _>>()?;
                let tag_ssa = emit_const_i64(i64::from(ctor.index()), ssa, body);
                EnvValue::Datum(Datum {
                    _tag: u64::from(ctor.index()),
                    tag_ssa,
                    fields,
                })
            }
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => {
                let (lane, ovf) =
                    lower_match_dialect(scrutinee, alts, default, env, ssa, bbc, body, uses_abort)?;
                if let Some(f) = ovf {
                    flags.push(f);
                }
                EnvValue::Repr(lane)
            }
            // Everything below is an explicit, never-silent refusal — it runs on the direct-LLVM
            // backend (`crate::llvm`, which covers the full v0 calculus) or the interpreter. The
            // message routes it there (EXPLAIN-able; G2/VR-5).
            Rhs::App { .. } | Rhs::Lam { .. } => {
                return Err(DialectError::Unsupported(
                    "closures (App/Lam) are not in the MLIR-dialect fragment — they are lowered by \
                     the direct-LLVM backend (crate::llvm; M-378) or interpreted"
                        .to_owned(),
                ));
            }
            Rhs::Fix { .. } | Rhs::FixGroup { .. } => {
                return Err(DialectError::Unsupported(
                    "recursion (Fix/FixGroup) is not in the MLIR-dialect fragment — tail recursion \
                     is lowered by the direct-LLVM backend (crate::llvm; M-379); the rest is \
                     interpreted"
                        .to_owned(),
                ));
            }
        };
        env.insert(b.name.clone(), ev);
    }
    let result = lookup_ev(env, anf.result())?.as_lane("block result")?;
    // Fold every per-op overflow `i1` *in this block* into one local flag (or `None`).
    let overflow = fold_or(&flags, ssa, body);
    Ok((result, overflow))
}

// ─── M-856: `Match` (Ctor-arm, non-recursive data fragment) ───────────────────────────────────

/// Lower `Rhs::Match` for a **`Datum` scrutinee + `Ctor` arms** — the M-856 dialect counterpart of
/// [`crate::llvm`]'s `lower_match` Increment-1 (Ctor) form. A `Match` on a bare repr lane (the
/// `Lit`-arm branch primitive `crate::llvm` uses for the Increment-3 recursion base case) is an
/// explicit refusal: it is tied to `Fix`/`FixGroup`, which stays out of the MLIR-dialect fragment.
///
/// Dispatches on the tag with `cf.switch`; each arm's fields are bound by **direct SSA reference**
/// (the `Datum`'s field lanes, computed before the switch, dominate every arm block — no load).
/// Each arm is lowered into its **own local text buffer** first (via a recursive [`lower_block`]
/// call, so nested `Match`/arithmetic inside an arm folds its own overflow flag locally), and only
/// once every arm's shape is known is the merge block's header decided — so the overflow `i1` is
/// threaded through the merge only when at least one arm actually needs it (an overflow-free,
/// arithmetic-free `Match` costs nothing beyond the switch/merge it already needed).
///
/// A no-match with no ANF `default` traps with `@abort` (a defined trap, never raw UB; G2) —
/// `uses_abort` is set so [`emit_mlir`] declares the extern only when it is actually called.
#[allow(clippy::too_many_arguments)]
fn lower_match_dialect(
    scrutinee: &Atom,
    alts: &[AnfAlt],
    default_arm: &Option<Anf>,
    env: &std::collections::HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    uses_abort: &mut bool,
) -> Result<(Lane, Option<String>), DialectError> {
    let scrut = env
        .get(scrutinee)
        .cloned()
        .ok_or_else(|| DialectError::FreeVariable(scrutinee.render()))?;
    let datum = match scrut {
        EnvValue::Datum(d) => d,
        EnvValue::Repr(_) => {
            return Err(DialectError::Unsupported(
                "Match on a repr-lane scrutinee is not in the MLIR-dialect fragment — only \
                 constructor-arm Match on a Construct-built datum is lowered here; literal-arm \
                 matching on a Binary{8} lane is the Increment-3 recursion branch primitive, tied to \
                 Fix/FixGroup and deferred alongside Dense/VSA (M-856; the direct-LLVM backend still \
                 covers it)"
                    .to_owned(),
            ));
        }
    };
    if alts.is_empty() && default_arm.is_none() {
        return Err(DialectError::Unsupported(
            "Match with zero arms and no default (exhaustive coverage requires at least one arm or \
             a default)"
                .to_owned(),
        ));
    }
    for alt in alts {
        if matches!(alt, AnfAlt::Lit { .. }) {
            return Err(DialectError::Unsupported(
                "a literal arm on a constructed-data Match scrutinee is not valid — constructor \
                 arms only for data values (G2)"
                    .to_owned(),
            ));
        }
    }

    let arm_labels: Vec<String> = (0..alts.len()).map(|_| bbc.fresh()).collect();
    let default_label = bbc.fresh();
    let merge_label = bbc.fresh();

    let _ = write!(
        body,
        "    cf.switch {} : i64, [\n      default: ^{default_label}",
        datum.tag_ssa
    );
    for (alt, label) in alts.iter().zip(&arm_labels) {
        let AnfAlt::Ctor { ctor, .. } = alt else {
            unreachable!("Lit arms rejected above")
        };
        let _ = write!(body, ",\n      {}: ^{label}", ctor.index());
    }
    let _ = writeln!(body, "\n    ]");

    struct ArmOut {
        label: String,
        text: String,
        lane: Lane,
        ovf: Option<String>,
    }
    let mut arms: Vec<ArmOut> = Vec::with_capacity(alts.len() + 1);
    for (alt, label) in alts.iter().zip(&arm_labels) {
        let AnfAlt::Ctor {
            binders,
            body: arm_body,
            ..
        } = alt
        else {
            unreachable!("Lit arms rejected above")
        };
        // Never-silent (G2): the interpreter rejects arity mismatch with DataMalformed.
        if binders.len() != datum.fields.len() {
            return Err(DialectError::Unsupported(format!(
                "Match arm binder arity ({}) != constructor field count ({}) — malformed Match \
                 (interpreter rejects with DataMalformed; G2/WF7)",
                binders.len(),
                datum.fields.len()
            )));
        }
        let mut arm_env = env.clone();
        for (binder, field_lane) in binders.iter().zip(&datum.fields) {
            arm_env.insert(
                Atom::Named(binder.clone()),
                EnvValue::Repr(field_lane.clone()),
            );
        }
        let mut text = String::new();
        let (lane, ovf) = lower_block(arm_body, &mut arm_env, ssa, bbc, &mut text, uses_abort)?;
        arms.push(ArmOut {
            label: label.clone(),
            text,
            lane,
            ovf,
        });
    }
    match default_arm {
        Some(default_block) => {
            let mut def_env = env.clone();
            let mut text = String::new();
            let (lane, ovf) =
                lower_block(default_block, &mut def_env, ssa, bbc, &mut text, uses_abort)?;
            arms.push(ArmOut {
                label: default_label.clone(),
                text,
                lane,
                ovf,
            });
        }
        None => {
            *uses_abort = true;
            let _ = writeln!(body, "  ^{default_label}:");
            let _ = writeln!(body, "    func.call @abort() : () -> ()");
            let z = emit_const_i32(0, ssa, body);
            let _ = writeln!(body, "    func.return {z} : i32");
        }
    }

    debug_assert!(
        !arms.is_empty(),
        "at least one Ctor arm or a Some(default) reaches here — checked above"
    );
    let kind = arms[0].lane.kind;
    let width = arms[0].lane.vals.len();
    for a in &arms[1..] {
        if a.lane.kind != kind || a.lane.vals.len() != width {
            return Err(DialectError::Unsupported(
                "Match arms produce lanes of different kind or width — all arms must return the \
                 same repr shape"
                    .to_owned(),
            ));
        }
    }
    let any_ovf = arms.iter().any(|a| a.ovf.is_some());

    let merge_result_names: Vec<String> = (0..width).map(|_| ssa.fresh()).collect();
    let merge_ovf_name = if any_ovf { Some(ssa.fresh()) } else { None };
    for a in &arms {
        let _ = writeln!(body, "  ^{}:", a.label);
        body.push_str(&a.text);
        let mut names = a.lane.vals.clone();
        let mut types: Vec<&str> = vec!["i32"; a.lane.vals.len()];
        if merge_ovf_name.is_some() {
            // Normalize a missing local overflow to a materialized `false` — only when a *sibling*
            // arm needs the i1 slot (never-silent: an arm that cannot overflow contributes `false`,
            // not an omitted operand).
            let ovf_ssa = match &a.ovf {
                Some(s) => s.clone(),
                None => {
                    let r = ssa.fresh();
                    let _ = writeln!(body, "    {r} = arith.constant false");
                    r
                }
            };
            names.push(ovf_ssa);
            types.push("i1");
        }
        let _ = writeln!(
            body,
            "    cf.br ^{merge_label}({} : {})",
            names.join(", "),
            types.join(", ")
        );
    }
    let mut header_parts: Vec<String> = merge_result_names
        .iter()
        .map(|n| format!("{n}: i32"))
        .collect();
    if let Some(ovf_name) = &merge_ovf_name {
        header_parts.push(format!("{ovf_name}: i1"));
    }
    if header_parts.is_empty() {
        let _ = writeln!(body, "  ^{merge_label}:");
    } else {
        let _ = writeln!(body, "  ^{merge_label}({}):", header_parts.join(", "));
    }

    Ok((
        Lane {
            kind,
            vals: merge_result_names,
        },
        merge_ovf_name,
    ))
}

// ─── M-856: `Swap` (certified binary↔ternary + identity) ──────────────────────────────────────

/// Lower `Rhs::Swap` for the **certified binary↔ternary class** (RFC-0002 §4) plus same-`Repr`
/// identity — the MLIR-arith counterpart of `crate::swap_codegen::lower_swap`, always under the
/// `Recheck` cert mode (the compile-time independent bijection re-check; `ReuseInterp` is not wired
/// here, a small explicitly-deferred gap). Reuses `crate::swap_codegen::legal_pair` and the
/// `MAX_BINARY_WIDTH_I64`/`MAX_TERNARY_WIDTH_I64` i64-soundness bounds directly (single source of
/// truth for the side-condition; DRY) but re-derives the transcode **arithmetic** in `arith` ops
/// (the direct-LLVM emitter's textual LLVM instructions cannot be reused — different IR surface —
/// so this is an independent rendering of the *same* `mycelium_core::binary`/`ternary` algorithm,
/// digit-for-digit, never a divergent second algorithm).
///
/// Returns the transcoded lane plus an optional out-of-range `i1` (`Some` for the partial `dec`
/// direction and the `enc` final-quotient honest-net check, mirroring `crate::swap_codegen`;
/// `None` for `enc` on a re-checked-legal pair and for identity). Dense/VSA, a non-bit/trit pair, or
/// an **illegal** `(n,m)` pair are explicit [`DialectError::Unsupported`] refusals — never silently
/// lowered (G2).
fn lower_swap_dialect(
    src_lane: &Lane,
    src_repr: &Repr,
    target: &Repr,
    ssa: &mut Ssa,
    body: &mut String,
) -> Result<(Lane, Option<String>), DialectError> {
    match (src_repr, target) {
        // ── Identity (same Repr): the trivial swap, no transcode. ──
        (a, b) if a == b => Ok((src_lane.clone(), None)),
        // ── Binary{n} → Ternary{m}: enc, total on a legal pair. ──
        (Repr::Binary { width }, Repr::Ternary { trits }) => {
            let (width, trits) = (*width, *trits);
            check_legal_dialect(width, trits)?;
            check_i64_width_dialect(width, trits)?;
            if src_lane.kind != LaneKind::Binary {
                return Err(DialectError::Unsupported(format!(
                    "swap Binary→Ternary: source lane is {:?}, expected Binary (G2)",
                    src_lane.kind
                )));
            }
            let int_reg = emit_swap_bits_to_int(&src_lane.vals, ssa, body);
            let (lane, final_q) = emit_swap_int_to_trits(&int_reg, trits as usize, ssa, body);
            // On a legal (re-checked) pair the final quotient is provably 0; the never-silent
            // final-quotient check is emitted anyway (dead code on the normal path) so a codegen
            // slip can never pass silently — mirrors `crate::swap_codegen`'s honest-net (G2/SC-3).
            let zero = emit_const_i64(0, ssa, body);
            let oor = ssa.fresh();
            let _ = writeln!(body, "    {oor} = arith.cmpi ne, {final_q}, {zero} : i64");
            Ok((lane, Some(oor)))
        }
        // ── Ternary{m} → Binary{n}: dec, PARTIAL — range failure is never-silent. ──
        (Repr::Ternary { trits }, Repr::Binary { width }) => {
            let (width, trits) = (*width, *trits);
            check_legal_dialect(width, trits)?;
            check_i64_width_dialect(width, trits)?;
            if src_lane.kind != LaneKind::Ternary {
                return Err(DialectError::Unsupported(format!(
                    "swap Ternary→Binary: source lane is {:?}, expected Ternary (G2)",
                    src_lane.kind
                )));
            }
            let int_reg = emit_swap_trits_to_int(&src_lane.vals, ssa, body);
            let (lane, oor) = emit_swap_int_to_bits(&int_reg, width as usize, ssa, body);
            Ok((lane, Some(oor)))
        }
        // ── Everything else: an explicit refusal — never a silent mis-lowering (G2). ──
        (a, b) => Err(DialectError::Unsupported(format!(
            "swap {a:?} → {b:?}: only the certified binary↔ternary class (and same-Repr identity) \
             is lowered in the MLIR-dialect fragment (M-856); Dense/VSA and other swap kinds stay \
             explicit refusals — they run on the interpreter / direct-LLVM path"
        ))),
    }
}

/// Re-check the bijection side-condition at compile time (the `Recheck` cert mode — the only mode
/// this module wires). An illegal pair is refused here, never emitted (VR-5/G2). Reuses
/// `crate::swap_codegen::legal_pair` directly (single source of truth for the side-condition).
fn check_legal_dialect(width: u32, trits: u32) -> Result<(), DialectError> {
    if crate::swap_codegen::legal_pair(width, trits) {
        Ok(())
    } else {
        Err(DialectError::Unsupported(format!(
            "swap Binary{{{width}}}↔Ternary{{{trits}}}: the compile-time re-check rejects the \
             bijection side-condition — (n,m) is NOT a legal pair (B_n ⊄ T_m, RFC-0002 §5); the \
             swap is refused, never emitted (M-856; VR-5/G2)"
        )))
    }
}

/// Refuse a bit/trit width the `i64` transcode cannot represent soundly (never a silently-wrong
/// transcode; G2/VR-5). Reuses `crate::swap_codegen`'s bounds directly (single source of truth).
fn check_i64_width_dialect(width: u32, trits: u32) -> Result<(), DialectError> {
    use crate::swap_codegen::{MAX_BINARY_WIDTH_I64, MAX_TERNARY_WIDTH_I64};
    if width > MAX_BINARY_WIDTH_I64 {
        return Err(DialectError::Unsupported(format!(
            "swap binary width {width} exceeds the native i64 transcode bound \
             ({MAX_BINARY_WIDTH_I64}); refused rather than emit a silently-wrong transcode (M-856; \
             G2/VR-5)"
        )));
    }
    if trits > MAX_TERNARY_WIDTH_I64 {
        return Err(DialectError::Unsupported(format!(
            "swap ternary width {trits} exceeds the native i64 transcode bound \
             ({MAX_TERNARY_WIDTH_I64}); refused rather than emit a silently-wrong transcode (M-856; \
             G2/VR-5)"
        )));
    }
    Ok(())
}

/// Decode an MSB-first `Binary` lane (`i32` elements in `{0,1}`) into a two's-complement integer in
/// an `i64` SSA register. Mirrors `mycelium_core::binary::bits_to_int` / `crate::swap_codegen`'s
/// `emit_bits_to_int` digit-for-digit, re-expressed in `arith` ops (M-856; DRY at the *algorithm*
/// level).
fn emit_swap_bits_to_int(bits: &[String], ssa: &mut Ssa, body: &mut String) -> String {
    let n = bits.len();
    let mut acc = emit_const_i64(0, ssa, body);
    for v in bits {
        let z = ssa.fresh();
        let _ = writeln!(body, "    {z} = arith.extui {v} : i32 to i64");
        let two = emit_const_i64(2, ssa, body);
        let sh = ssa.fresh();
        let _ = writeln!(body, "    {sh} = arith.muli {acc}, {two} : i64");
        let next = ssa.fresh();
        let _ = writeln!(body, "    {next} = arith.addi {sh}, {z} : i64");
        acc = next;
    }
    if n == 0 {
        return acc; // empty string denotes 0 (binary::bits_to_int contract)
    }
    let sign = &bits[0];
    let one32 = emit_const_i32(1, ssa, body);
    let is_neg = ssa.fresh();
    let _ = writeln!(body, "    {is_neg} = arith.cmpi eq, {sign}, {one32} : i32");
    debug_assert!(
        n <= crate::swap_codegen::MAX_BINARY_WIDTH_I64 as usize,
        "check_i64_width_dialect guarantees n <= MAX_BINARY_WIDTH_I64"
    );
    let two_pow_n: i64 = 1i64 << n;
    let two_pow_n_ssa = emit_const_i64(two_pow_n, ssa, body);
    let corrected = ssa.fresh();
    let _ = writeln!(
        body,
        "    {corrected} = arith.subi {acc}, {two_pow_n_ssa} : i64"
    );
    let out = ssa.fresh();
    let _ = writeln!(
        body,
        "    {out} = arith.select {is_neg}, {corrected}, {acc} : i64"
    );
    out
}

/// Decode an MSB-first `Ternary` lane (`i32` elements in `{−1,0,1}`) into an integer in an `i64`
/// SSA register. Mirrors `mycelium_core::ternary::trits_to_int` / `crate::swap_codegen`'s
/// `emit_trits_to_int` digit-for-digit (Horner from the MSB), re-expressed in `arith` ops (M-856).
fn emit_swap_trits_to_int(trits: &[String], ssa: &mut Ssa, body: &mut String) -> String {
    let mut acc = emit_const_i64(0, ssa, body);
    for v in trits {
        let ext = ssa.fresh();
        let _ = writeln!(body, "    {ext} = arith.extsi {v} : i32 to i64");
        let three = emit_const_i64(3, ssa, body);
        let mul = ssa.fresh();
        let _ = writeln!(body, "    {mul} = arith.muli {acc}, {three} : i64");
        let next = ssa.fresh();
        let _ = writeln!(body, "    {next} = arith.addi {mul}, {ext} : i64");
        acc = next;
    }
    acc
}

/// Encode an `i64` integer (`int_reg`) into an MSB-first `Ternary` lane of `m` trits, returning the
/// lane and the final-quotient SSA register. Mirrors `mycelium_core::ternary::int_to_trits` /
/// `crate::swap_codegen`'s `emit_int_to_trits` digit-for-digit (balanced remainder + borrow),
/// re-expressed in `arith` ops (M-856). The caller emits the never-silent final-quotient check.
fn emit_swap_int_to_trits(
    int_reg: &str,
    m: usize,
    ssa: &mut Ssa,
    body: &mut String,
) -> (Lane, String) {
    let mut v = int_reg.to_owned();
    let mut lsb_first: Vec<String> = Vec::with_capacity(m);
    for _ in 0..m {
        let three_a = emit_const_i64(3, ssa, body);
        let sr = ssa.fresh();
        let _ = writeln!(body, "    {sr} = arith.remsi {v}, {three_a} : i64");
        let three_b = emit_const_i64(3, ssa, body);
        let plus3 = ssa.fresh();
        let _ = writeln!(body, "    {plus3} = arith.addi {sr}, {three_b} : i64");
        let three_c = emit_const_i64(3, ssa, body);
        let r0 = ssa.fresh();
        let _ = writeln!(body, "    {r0} = arith.remsi {plus3}, {three_c} : i64");
        let two_c = emit_const_i64(2, ssa, body);
        let is_two = ssa.fresh();
        let _ = writeln!(body, "    {is_two} = arith.cmpi eq, {r0}, {two_c} : i64");
        let neg1_c = emit_const_i64(-1, ssa, body);
        let digit64 = ssa.fresh();
        let _ = writeln!(
            body,
            "    {digit64} = arith.select {is_two}, {neg1_c}, {r0} : i64"
        );
        let v_minus = ssa.fresh();
        let _ = writeln!(body, "    {v_minus} = arith.subi {v}, {digit64} : i64");
        let three_d = emit_const_i64(3, ssa, body);
        let v_next = ssa.fresh();
        let _ = writeln!(
            body,
            "    {v_next} = arith.divsi {v_minus}, {three_d} : i64"
        );
        let digit32 = ssa.fresh();
        let _ = writeln!(body, "    {digit32} = arith.trunci {digit64} : i64 to i32");
        lsb_first.push(digit32);
        v = v_next;
    }
    let vals: Vec<String> = lsb_first.into_iter().rev().collect(); // → MSB-first
    (
        Lane {
            kind: LaneKind::Ternary,
            vals,
        },
        v,
    )
}

/// Encode an `i64` integer (`int_reg`) into an MSB-first `Binary` lane of `n` two's-complement
/// bits, plus an `i1` out-of-range register (set iff the value does not fit `B_n`). Mirrors
/// `mycelium_core::binary::int_to_bits` / `crate::swap_codegen`'s `emit_int_to_bits` digit-for-digit,
/// re-expressed in `arith` ops (M-856). The range bit is the never-silent `dec`-partiality signal.
fn emit_swap_int_to_bits(
    int_reg: &str,
    n: usize,
    ssa: &mut Ssa,
    body: &mut String,
) -> (Lane, String) {
    if n == 0 {
        // Zero-width: representable iff v == 0 (binary::int_to_bits n==0 contract).
        let zero = emit_const_i64(0, ssa, body);
        let oor = ssa.fresh();
        let _ = writeln!(body, "    {oor} = arith.cmpi ne, {int_reg}, {zero} : i64");
        return (
            Lane {
                kind: LaneKind::Binary,
                vals: Vec::new(),
            },
            oor,
        );
    }
    debug_assert!(
        n <= crate::swap_codegen::MAX_BINARY_WIDTH_I64 as usize,
        "check_i64_width_dialect guarantees n <= MAX_BINARY_WIDTH_I64"
    );
    let half: i64 = 1i64 << (n - 1);
    let lo = -half;
    let hi = half - 1;
    let lo_c = emit_const_i64(lo, ssa, body);
    let lt_lo = ssa.fresh();
    let _ = writeln!(
        body,
        "    {lt_lo} = arith.cmpi slt, {int_reg}, {lo_c} : i64"
    );
    let hi_c = emit_const_i64(hi, ssa, body);
    let gt_hi = ssa.fresh();
    let _ = writeln!(
        body,
        "    {gt_hi} = arith.cmpi sgt, {int_reg}, {hi_c} : i64"
    );
    let oor = ssa.fresh();
    let _ = writeln!(body, "    {oor} = arith.ori {lt_lo}, {gt_hi} : i1");
    let vals: Vec<String> = (0..n)
        .map(|i| {
            let shift = n - 1 - i;
            let shift_c = emit_const_i64(shift as i64, ssa, body);
            let sh = ssa.fresh();
            let _ = writeln!(body, "    {sh} = arith.shrui {int_reg}, {shift_c} : i64");
            let one_c = emit_const_i64(1, ssa, body);
            let m = ssa.fresh();
            let _ = writeln!(body, "    {m} = arith.andi {sh}, {one_c} : i64");
            let t = ssa.fresh();
            let _ = writeln!(body, "    {t} = arith.trunci {m} : i64 to i32");
            t
        })
        .collect();
    (
        Lane {
            kind: LaneKind::Binary,
            vals,
        },
        oor,
    )
}

/// Emit the print sequence: one `func.call @putchar` per result element (its ASCII char), then a
/// newline `putchar`. The char codes match [`crate::llvm`]'s `emit_char_code` (Binary → `val + 48`;
/// Ternary → `-1→45 ('-')`, `0→48 ('0')`, `1→43 ('+')`) so the read-back is identical across paths.
fn emit_print(lane: &Lane, ssa: &mut Ssa, body: &mut String) {
    for v in &lane.vals {
        let code = emit_char_code(lane.kind, v, ssa, body);
        let r = ssa.fresh();
        let _ = writeln!(body, "    {r} = func.call @putchar({code}) : (i32) -> i32");
    }
    let nl = emit_const_i32(10, ssa, body);
    let r = ssa.fresh();
    let _ = writeln!(body, "    {r} = func.call @putchar({nl}) : (i32) -> i32");
}

/// Emit the `i32` ASCII char code for one element of `kind` (SSA `v`) using `arith` ops, returning
/// the SSA holding the code. The encoding is byte-for-byte the same as [`crate::llvm`]'s
/// `emit_char_code`, so a Binary/Ternary element prints the identical char on both compiled paths.
fn emit_char_code(kind: LaneKind, v: &str, ssa: &mut Ssa, body: &mut String) -> String {
    match kind {
        LaneKind::Binary => {
            let off = emit_const_i32(48, ssa, body);
            let c = ssa.fresh();
            let _ = writeln!(body, "    {c} = arith.addi {v}, {off} : i32");
            c
        }
        LaneKind::Ternary => {
            // isneg = (v == -1); ispos = (v == 1).
            let neg1 = emit_const_i32(-1, ssa, body);
            let pos1 = emit_const_i32(1, ssa, body);
            let isneg = ssa.fresh();
            let _ = writeln!(body, "    {isneg} = arith.cmpi eq, {v}, {neg1} : i32");
            let ispos = ssa.fresh();
            let _ = writeln!(body, "    {ispos} = arith.cmpi eq, {v}, {pos1} : i32");
            // t = ispos ? 43 ('+') : 48 ('0');  c = isneg ? 45 ('-') : t.
            let c43 = emit_const_i32(43, ssa, body);
            let c48 = emit_const_i32(48, ssa, body);
            let c45 = emit_const_i32(45, ssa, body);
            let t = ssa.fresh();
            let _ = writeln!(body, "    {t} = arith.select {ispos}, {c43}, {c48} : i32");
            let c = ssa.fresh();
            let _ = writeln!(body, "    {c} = arith.select {isneg}, {c45}, {t} : i32");
            c
        }
    }
}

/// Emit the full MLIR module for `node`: a `func.func private @putchar` declaration + a
/// `func.func @main` that computes the result lane, prints each element, and returns 0. Deterministic.
/// Returns an explicit [`DialectError::Unsupported`] for an out-of-fragment node (the program then
/// runs on the direct-LLVM backend / interpreter).
///
/// **Overflow read-back (M-725/M-857).** When the program contains a `trit.add`/`trit.sub`
/// (M-725) or `trit.mul` (M-857) that can overflow at runtime, `@main` branches (`cf.cond_br`) on
/// the folded overflow `i1`: on overflow it prints the shared [`OVERFLOW_SENTINEL`] line and
/// returns 0; otherwise it prints the result line — exactly mirroring [`crate::llvm`]'s read-back,
/// so the artifact's stdout means the same on both compiled paths. An **overflow-free** program (no
/// fixed-width arithmetic op) emits the single-block, straight-line module unchanged (byte-for-byte
/// the M-601 shape) — the branch is added only when it is needed.
///
/// The returned `(module, kind, width)` triple carries the lane shape so the read-back
/// ([`crate::llvm::decode_result`]) can parse `@main`'s stdout. Every op is explicit, dumpable MLIR
/// text — no opaque pass (RFC-0004 §6 / VR-4).
pub fn emit_mlir(node: &Node) -> Result<(String, ResultKind, usize), DialectError> {
    let mut ssa = Ssa::default();
    let mut bbc = Bbc::default();
    let mut body = String::new();
    let mut uses_abort = false;
    let (result, overflow) = lower_program(node, &mut ssa, &mut bbc, &mut body, &mut uses_abort)?;

    let kind = ResultKind::from_lane(result.kind);
    let width = result.vals.len();

    let mut module = String::new();
    module.push_str("module {\n");
    module.push_str("  func.func private @putchar(i32) -> i32\n");
    // M-856: `@abort` is declared only when a Match with no default arm actually traps to it — a
    // data-free / total-Match program's module is unaffected (still byte-for-byte the M-601 shape).
    if uses_abort {
        module.push_str("  func.func private @abort() -> ()\n");
    }
    module.push_str("  func.func @main() -> i32 {\n");

    match overflow {
        // No fixed-width arithmetic op (trit.add/sub/mul) ⇒ no overflow path; straight-line
        // print + return (the M-601 module unchanged, so element-wise programs emit byte-for-byte
        // as before).
        None => {
            emit_print(&result, &mut ssa, &mut body);
            module.push_str(&body);
            let r = ssa.fresh();
            let _ = writeln!(module, "    {r} = arith.constant 0 : i32");
            let _ = writeln!(module, "    func.return {r} : i32");
        }
        // Overflow possible ⇒ branch on the runtime flag (`cf.cond_br`): print the sentinel line on
        // overflow, the result line otherwise. The read-back protocol — never a silent wrap (G2).
        // The entry block ends with the body (which computed the `ovf` i1) + the conditional branch;
        // `^ovf` prints the sentinel, `^ok` prints the result. Both return 0.
        Some(ovf) => {
            module.push_str(&body);
            let _ = writeln!(module, "    cf.cond_br {ovf}, ^ovf, ^ok");
            // ^ovf: print the OVERFLOW_SENTINEL char + newline, return 0.
            module.push_str("  ^ovf:\n");
            let sentinel = emit_const_i32(i32::from(OVERFLOW_SENTINEL), &mut ssa, &mut module);
            let s = ssa.fresh();
            let _ = writeln!(
                module,
                "    {s} = func.call @putchar({sentinel}) : (i32) -> i32"
            );
            let nl = emit_const_i32(10, &mut ssa, &mut module);
            let snl = ssa.fresh();
            let _ = writeln!(
                module,
                "    {snl} = func.call @putchar({nl}) : (i32) -> i32"
            );
            let zo = ssa.fresh();
            let _ = writeln!(module, "    {zo} = arith.constant 0 : i32");
            let _ = writeln!(module, "    func.return {zo} : i32");
            // ^ok: print the result line, return 0.
            module.push_str("  ^ok:\n");
            let mut ok = String::new();
            emit_print(&result, &mut ssa, &mut ok);
            module.push_str(&ok);
            let zk = ssa.fresh();
            let _ = writeln!(module, "    {zk} = arith.constant 0 : i32");
            let _ = writeln!(module, "    func.return {zk} : i32");
        }
    }

    module.push_str("  }\n}\n");
    Ok((module, kind, width))
}

// ─── The pipeline: MLIR module → real LLVM IR → native → read-back ────────────────────────────

/// Lower `node` through the real MLIR pipeline to **LLVM IR text**, without compiling/running it.
/// Emits the `arith`/`func`/`cf` MLIR module ([`emit_mlir`]), then runs
/// `mlir-opt --convert-cf-to-llvm --convert-func-to-llvm --convert-arith-to-llvm
/// --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir`. The `--convert-cf-to-llvm` pass
/// lowers the M-725/M-857 overflow-read-back `cf.cond_br` (a no-op for an overflow-free
/// element-wise program, which contains no `cf` ops). Each stage is a real libMLIR pass; the intermediate MLIR and
/// the resulting IR are both dumpable (no opaque pass — VR-4). Returns the LLVM IR text + lane shape,
/// or an explicit [`DialectError`] (skip on `ToolchainMissing`).
pub fn lower_to_llvm_ir(node: &Node) -> Result<(String, ResultKind, usize), DialectError> {
    let (mlir, kind, width) = emit_mlir(node)?;
    let tools = resolve_tools()?;

    let dir = unique_tmp_dir()?;
    let mlir_path = dir.join("kernel.mlir");
    let guard = TmpDir(dir);
    std::fs::write(&mlir_path, mlir.as_bytes())
        .map_err(|e| DialectError::Run(format!("write MLIR: {e}")))?;

    // Stage 1: mlir-opt lowers cf+func+arith → the LLVM dialect. `--convert-cf-to-llvm` handles
    // the M-725/M-857 overflow-read-back branch; it is a no-op for an overflow-free element-wise
    // module.
    let lowered_mlir = run_capture(
        &tools.mlir_opt,
        &[
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            path(&mlir_path)?,
        ],
        "mlir-opt",
    )?;

    // Stage 2: mlir-translate renders the LLVM-dialect module as textual LLVM IR.
    let lowered_path = guard.0.join("kernel.lowered.mlir");
    std::fs::write(&lowered_path, lowered_mlir.as_bytes())
        .map_err(|e| DialectError::Run(format!("write lowered MLIR: {e}")))?;
    let llvm_ir = run_capture(
        &tools.mlir_translate,
        &["--mlir-to-llvmir", path(&lowered_path)?],
        "mlir-translate",
    )?;

    Ok((llvm_ir, kind, width))
}

/// A compiled native artifact from the MLIR-dialect path: the executable on disk (cleaned up on
/// drop) plus the result shape needed to parse its output. Produced by [`compile`]; run with
/// [`Compiled::run`]. The **compile-once / run-many** split mirrors [`crate::llvm::CompiledArtifact`]
/// so a harness can time the one-time AOT cost separately from warm per-invocation cost (M-602).
pub struct Compiled {
    _dir: TmpDir,
    bin: std::path::PathBuf,
    kind: ResultKind,
    width: usize,
    llvm_major: u32,
}

impl Compiled {
    /// The LLVM major version the MLIR toolchain was matched to (for `EXPLAIN`/captions).
    #[must_use]
    pub fn llvm_major(&self) -> u32 {
        self.llvm_major
    }
    /// Execute the compiled artifact and read its result back as an `Exact` `Binary{w}`/`Ternary{m}`
    /// [`Value`] via the **shared** [`crate::llvm::decode_result`] (same read-back as the
    /// direct-LLVM path). A `trit.add`/`trit.sub` (M-725) or `trit.mul` (M-857) that overflowed
    /// prints the shared [`OVERFLOW_SENTINEL`] line ⇒ an explicit [`DialectError::Overflow`], never
    /// a silent wrap (M-725/M-857; mirrors [`crate::llvm::AotError::Overflow`] and the
    /// interpreter's `EvalError::Overflow`).
    pub fn run(&self) -> Result<Value, DialectError> {
        let output = Command::new(&self.bin)
            .output()
            .map_err(|e| DialectError::Run(format!("exec {}: {e}", self.bin.display())))?;
        if !output.status.success() {
            return Err(DialectError::Run(format!(
                "artifact exited {}",
                output.status
            )));
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|e| DialectError::Parse(format!("non-utf8 output: {e}")))?;
        let line = stdout.lines().next().unwrap_or("");
        // Read-back protocol: the sentinel line means the native arithmetic overflowed the m-trit
        // range — an explicit error, never a silently-wrapped result (matches the interpreter's
        // `EvalError::Overflow` and the direct-LLVM path, so the three-way differential stays honest).
        if line.as_bytes() == [OVERFLOW_SENTINEL] {
            return Err(DialectError::Overflow(format!(
                "fixed-width result out of {}-trit range",
                self.width
            )));
        }
        decode_result(self.kind.to_lane(), self.width, line.chars())
            .map_err(|e| DialectError::Parse(e.to_string()))
    }
}

/// Compile `node` through the MLIR pipeline to a native executable (MLIR → LLVM IR → `clang`)
/// without running it. Returns [`DialectError::ToolchainMissing`] when the toolchain is absent so
/// callers can skip; any out-of-fragment construct is an explicit [`DialectError::Unsupported`].
pub fn compile(node: &Node) -> Result<Compiled, DialectError> {
    let (llvm_ir, kind, width) = lower_to_llvm_ir(node)?;
    let tools = resolve_tools()?;

    let dir = unique_tmp_dir()?;
    let ll = dir.join("kernel.ll");
    let bin = dir.join("kernel");
    let guard = TmpDir(dir);
    std::fs::write(&ll, llvm_ir.as_bytes())
        .map_err(|e| DialectError::Run(format!("write LLVM IR: {e}")))?;

    // clang compiles + links the textual LLVM IR directly to a native executable.
    run_ok(
        &tools.clang,
        &[path(&ll)?, "-o", path(&bin)?, "-Wno-override-module"],
        "clang",
    )?;

    Ok(Compiled {
        _dir: guard,
        bin,
        kind,
        width,
        llvm_major: tools.llvm_major,
    })
}

/// Compile + run `node` through the MLIR pipeline and read the result back. The convenience wrapper
/// over [`compile`] + [`Compiled::run`] — the **MLIR-dialect** execution path the M-602 three-way
/// differential checks against the interpreter and the direct-LLVM backend.
pub fn compile_and_run(node: &Node) -> Result<Value, DialectError> {
    compile(node)?.run()
}

// ─── subprocess plumbing (mirrors crate::llvm's tool-probe pattern) ───────────────────────────

/// Run a tool capturing stdout; a missing binary is [`DialectError::ToolchainMissing`] (skip), a
/// non-zero exit is [`DialectError::Compile`] with the captured stderr (no opaque failure).
fn run_capture(tool: &str, args: &[&str], stage: &str) -> Result<String, DialectError> {
    let out = Command::new(tool)
        .args(args)
        .output()
        .map_err(|_| DialectError::ToolchainMissing(tool.to_owned()))?;
    if out.status.success() {
        String::from_utf8(out.stdout)
            .map_err(|e| DialectError::Parse(format!("{stage}: non-utf8 stdout: {e}")))
    } else {
        Err(DialectError::Compile(format!(
            "{stage} ({tool} {}): {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        )))
    }
}

/// Run a tool for its side effect (no stdout needed); same never-silent error contract.
fn run_ok(tool: &str, args: &[&str], stage: &str) -> Result<(), DialectError> {
    let out = Command::new(tool)
        .args(args)
        .output()
        .map_err(|_| DialectError::ToolchainMissing(tool.to_owned()))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(DialectError::Compile(format!(
            "{stage} ({tool} {}): {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        )))
    }
}

fn path(p: &Path) -> Result<&str, DialectError> {
    p.to_str()
        .ok_or_else(|| DialectError::Run(format!("non-utf8 path {}", p.display())))
}

fn unique_tmp_dir() -> Result<std::path::PathBuf, DialectError> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("myc-mlir-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| DialectError::Run(format!("mkdir tmp: {e}")))?;
    Ok(dir)
}

/// Best-effort cleanup of the per-run temp dir.
struct TmpDir(std::path::PathBuf);
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
