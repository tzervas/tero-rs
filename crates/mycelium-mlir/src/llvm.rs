//! Direct-LLVM-IR AOT backend for the kernel **bit/trit subset + non-recursive data fragment**
//! (M-301; M-373; RFC-0004 §2 *direct-LLVM fallback* / §11.2 *Increment-1 sanction*;
//! ADR-007/009; DN-15 §4.1; phase-3.md §1/§9.1).
//!
//! **Scope / honesty.** The ratified AOT path is `MLIR → LLVM` (RFC-0004 §2), but libMLIR is absent
//! in this environment while LLVM 18 tooling (`llc`, `clang`) is present. RFC-0004 §2 explicitly
//! anticipates *"a lighter direct-LLVM backend"* as the revisit; this module is that backend, scoped
//! to a **bit/trit + non-recursive data sub-fragment**: `core.id`, `bit.not/and/or/xor` over
//! `Binary{w}`, `trit.neg/add/sub/mul` over `Ternary{m}`, and (Increment-1) `Construct`/`Match` over
//! tagged stack structs. It is a *genuinely compiled native artifact* — not the textual `dialect::emit`
//! skeleton, and not the `aot::run` env-machine: [`emit_llvm_ir`] renders textual LLVM IR (one op
//! per output element, so nothing is opaque — RFC-0004 §6), and [`compile_and_run`] drives `llc` +
//! `clang` to a native executable, runs it, and reads the result back. This is the third,
//! *compiled*, execution path; the interp↔native differential (M-302) checks it against the
//! reference interpreter (NFR-7/RR-12).
//!
//! **Trit carry arithmetic (M-301 trit slice).** `trit.add/sub/mul` over `Ternary{m}` are lowered as
//! **ripple-carry** / **shifted-accumulate** IR that mirrors `mycelium_core::ternary` digit-for-digit
//! (`s + 4`, then `srem 3 − 1` for the balanced digit and `sdiv 3 − 1` for the carry — euclidean by
//! construction because `s + 4 ≥ 1`). Fixed-width overflow (a non-zero final carry, or non-zero high
//! trits of a product) is **detected at runtime** and signalled through the **read-back protocol**:
//! an out-of-range result prints the `OVERFLOW_SENTINEL` line (AOT) / returns a non-zero status
//! (JIT) and surfaces as an explicit [`AotError::Overflow`] — never a silent wrap (SC-3; G2). This
//! matches the interpreter's `EvalError::Overflow` so the M-302 differential stays honest.
//!
//! **Non-recursive data sub-fragment (Increment-1 — M-373; DN-15 §4.1; RFC-0004 §11.2).**
//! `Construct` and `Match` are now natively compiled for the **non-recursive, bounded** case (no
//! `Fix`/`FixGroup` in scope, so all allocations are statically bounded at codegen time). The
//! representation uses **stack `alloca`** (not `@malloc`) — a deliberate choice grounded in the
//! non-recursive/bounded restriction: because no heap recursion can produce unbounded allocation
//! depth, the alloca frame size is fixed at compile time, and an explicit OOM failure path is
//! unnecessary. Each constructed value is an `[N+1 x i64]` alloca (slot 0 = tag i64; slots 1..N =
//! field elements, one i64 per element laid out consecutively across all fields). `Match` emits an
//! LLVM `switch i64` on the tag with an explicit defined-trap default (never raw `unreachable` UB;
//! G2). Guarantee tag: **Declared** (hand-written textual-IR lowering; the differential against the
//! interpreter is empirical evidence, not a proof — VR-5).
//!
//! **Deliberately out of *this* subset (explicit refusals, never silent — G2):** `App`, `Lam`, `Fix`,
//! `FixGroup` (closures + recursion need closure-conversion + heap, deferred to Increment-2/3),
//! `Swap` to a non-binary/ternary repr, and the `Dense`/`Vsa` representations *in the generic bit/trit
//! `Node` lowering* (the lane model here is i32 bit/trit elements). Each is an explicit [`AotError`].
//! **`Repr::Dense` now has a dedicated native home** — the [`crate::dense_codegen`] direct-LLVM path
//! (M-853; RFC-0039 §5.1) lowers the un-quantized F32/BF16 element-wise surface against the
//! `mycelium-dense` reference; a Dense `Const` reaching [`const_lane`] is routed there (the refusal
//! here names where Dense *is* lowered — ADR-006/G2). VSA + quantized Dense stay refused (RFC-0039
//! §5.1/§5.2; M-854 / E20-1). The MLIR dialect path stays the eventual home for the bit/trit fragment
//! (`dialect::emit` is its dumpable skeleton), deferred until libMLIR exists.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _; // `writeln!` into a String never fails — call sites discard the Result.
use std::path::Path;
use std::process::Command;

use mycelium_core::lower::{self, Atom, Rhs};
use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Trit, Value};

use crate::swap_codegen::{self, SwapCertMode};

/// An explicit failure of the direct-LLVM AOT path. Every non-supported construct, missing tool, or
/// subprocess failure is one of these — the path is **never silent** (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AotError {
    /// A representation outside the subset (only `Binary{w}` / `Ternary{m}` are supported here).
    UnsupportedRepr(String),
    /// A primitive outside the subset (`core.id`, `bit.not/and/or/xor`, `trit.neg`).
    UnsupportedPrim(String),
    /// A Core IR construct the subset backend does not lower (e.g. a swap).
    UnsupportedNode(String),
    /// An operand atom with no prior binding (an ill-formed lowering).
    FreeVariable(String),
    /// A binary op over mismatched widths.
    WidthMismatch {
        /// The primitive name.
        prim: String,
        /// First operand width.
        a: usize,
        /// Second operand width.
        b: usize,
    },
    /// The native toolchain (`llc`/`clang`) is not installed — callers should **skip**, not fail
    /// (the house "skip gracefully when a tool is absent" idiom).
    ToolchainMissing(String),
    /// `llc`/`clang` ran but returned a non-zero status (compile failure).
    Compile(String),
    /// The compiled artifact failed to run or produced unreadable output.
    Run(String),
    /// The native stdout did not parse back into the expected payload shape.
    Parse(String),
    /// Reconstructing the result [`Value`] failed its well-formedness check.
    Wf(String),
    /// A balanced-ternary arithmetic result left the fixed `m`-trit range — the native path computed
    /// the overflow at runtime and signalled it through the read-back protocol (matches the
    /// interpreter's `EvalError::Overflow`; never a silent wrap, SC-3/G2).
    Overflow(String),
    /// A [`PackScheme`](mycelium_core::PackScheme) with no BitNet compute kernel (only the three
    /// bitnet packings I2_S/TL1/TL2 have one). An explicit refusal — never a silent misdecode.
    UnsupportedScheme(String),
    /// A tail-recursive `Fix` loop hit the [`AutoDepthBudget`] ceiling — a graceful explicit refusal.
    /// This mirrors the **AOT env-machine's** `EvalError::DepthLimit` (the env-machine reuses the same
    /// budget trait). The reference **interpreter** is O(1)-host-stack and refuses non-termination via
    /// `EvalError::FuelExhausted` instead — it does **not** raise `DepthLimit`. So the differential
    /// parity is *"every path refuses a non-productive recursion, none silently"* (G2/SC-3), **not**
    /// that all paths raise the same error variant.
    ///
    /// [`AutoDepthBudget`]: crate::budget::AutoDepthBudget
    DepthLimit(String),
}

/// The single byte the native artifact prints (AOT) when a fixed-width trit-arithmetic result
/// overflows the `m`-trit range. Chosen because it is **not** a valid element char (`'0'`/`'1'` for
/// bits, `'-'`/`'0'`/`'+'` for trits), so it can never be confused with a result line.
pub(crate) const OVERFLOW_SENTINEL: u8 = b'!';

/// The single byte the native artifact prints (AOT) when a tail-recursive Fix loop hits the
/// `AutoDepthBudget` ceiling (Increment-3 / DN-05 #1). Distinct from `OVERFLOW_SENTINEL` and all
/// valid element chars — a defined-sentinel, never a silent hang or SIGSEGV (G2/SC-3).
pub(crate) const DEPTHLIMIT_SENTINEL: u8 = b'#';

impl fmt::Display for AotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AotError::UnsupportedRepr(r) => write!(f, "unsupported repr for the AOT subset: {r}"),
            AotError::UnsupportedPrim(p) => write!(f, "unsupported prim for the AOT subset: {p}"),
            AotError::UnsupportedNode(n) => write!(f, "unsupported node for the AOT subset: {n}"),
            AotError::FreeVariable(v) => write!(f, "free variable in lowered IR: {v}"),
            AotError::WidthMismatch { prim, a, b } => {
                write!(f, "{prim}: width mismatch {a} vs {b}")
            }
            AotError::ToolchainMissing(t) => write!(f, "native toolchain missing: {t}"),
            AotError::Compile(e) => write!(f, "native compile failed: {e}"),
            AotError::Run(e) => write!(f, "native run failed: {e}"),
            AotError::Parse(e) => write!(f, "native output parse failed: {e}"),
            AotError::Wf(e) => write!(f, "result not well-formed: {e}"),
            AotError::Overflow(e) => write!(f, "balanced-ternary overflow: {e}"),
            AotError::UnsupportedScheme(s) => write!(f, "no BitNet kernel for packing scheme: {s}"),
            AotError::DepthLimit(e) => write!(f, "tail-recursion depth limit reached: {e}"),
        }
    }
}

impl std::error::Error for AotError {}

/// One element (a bit or a trit), as an LLVM `i32` operand: a literal (`"0"`/`"1"`/`"-1"`) or an
/// SSA register (`"%r3"`).
type Operand = String;

/// Which representation a lane carries — fixes how its elements are computed and printed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneKind {
    /// `Binary{w}` — elements in `{0, 1}`, printed `'0'`/`'1'`.
    Binary,
    /// `Ternary{m}` — balanced-ternary elements in `{-1, 0, 1}`, printed `'-'`/`'0'`/`'+'`.
    Ternary,
}

/// A computed value lane: its representation kind and one `i32` operand per element.
#[derive(Debug, Clone)]
pub(crate) struct Lane {
    pub(crate) kind: LaneKind,
    pub(crate) vals: Vec<Operand>,
}

/// The layout of one field inside a [`Datum`] struct: kind + number of elements. Elements are stored
/// consecutively in the struct's i64 slots starting at `slot_start` (each element occupies one i64).
#[derive(Debug, Clone)]
pub(crate) struct FieldLayout {
    /// Binary or Ternary — determines how elements are interpreted.
    pub(crate) kind: LaneKind,
    /// Number of elements (the `w` of `Binary{w}` or the `m` of `Ternary{m}`).
    pub(crate) elems: usize,
    /// The i64 slot index (1-based; slot 0 is always the tag) of the first element of this field.
    pub(crate) slot_start: usize,
}

/// A constructed data value in the lowered env: a pointer to a stack-`alloca`'d struct (the tag in
/// slot 0, field elements in consecutive i64 slots after it) plus the field layout so that a
/// downstream `Match` can extract the fields without knowing the field types again.
///
/// Representation choice (DN-15 §4.1 / RFC-0004 §11.2): **stack `alloca`** is used instead of
/// `@malloc` because the non-recursive/bounded restriction (no `Fix`/`FixGroup`) means all
/// allocation depth is fixed at codegen time — there is no need for heap allocation or an explicit
/// OOM failure path. `alloca` is simpler, inspectable, and directly auditable in the emitted IR.
#[derive(Debug, Clone)]
pub(crate) struct Datum {
    /// The SSA register holding the `[N+1 x i64]*` alloca pointer.
    pub(crate) ptr: String,
    /// The constructor tag (an i64 discriminant, equal to the `CtorRef::index()`).
    /// Retained for auditability / future diagnostics; not read back during Match lowering because
    /// the tag is re-loaded from the alloca at runtime.
    pub(crate) _tag: u64,
    /// Layout of each field, in declaration order.
    pub(crate) fields: Vec<FieldLayout>,
    /// Total number of i64 slots (1 tag + sum of all field elem counts).
    pub(crate) slots: usize,
}

/// The element width of the **narrow `Binary{8}` recursion-accumulator ABI** used by the heap
/// trampoline ([`crate::trampoline`]): a recursion accumulator is a `Binary{8}` value packed into a
/// single `i64` (DN-15 §7.1/§10). Anything wider/other-repr is an explicit `UnsupportedNode` (never a
/// silent mis-encode — G2). *(M-378 also used this for the narrow closure ABI; M-851 widened closures
/// to inlined any-repr/width values, so this constant now scopes the trampoline accumulator only.)*
pub(crate) const CLOSURE_ABI_WIDTH: usize = 8;

/// A native closure value — **widened, specialize-at-application** representation (M-851; DN-15 §7.1
/// — the "uniform … any repr/width" widening of the M-378 narrow `Binary{8}` ABI). The narrow ABI
/// eagerly emitted a top-level `i64 (i8*,i64)` function + heap record carrying a single packed
/// `Binary{8}`; this widened path instead **suspends** the closure as its un-lowered lambda (`param` +
/// `body` ANF) plus a snapshot of the **captured environment** (each free var's already-lowered
/// [`EnvValue`]), and **specializes (inlines) it at the application site** ([`lower_app`]): the
/// argument pins the param's concrete shape, the body is lowered inline, and the result flows back as
/// an [`EnvValue`] of whatever shape the body computed. This lets closures over **any repr/width**,
/// **curried application** (a body whose result is itself a `ClosureVal`), and **returned closures**
/// (a closure flowing through a `let`/binding, applied later) all lower natively — with no fixed wire
/// ABI, no indirect call, and **no guessed param width** (each shape is statically resolved at the App
/// site; G2/VR-5). A closure that is never applied (left on the printable program result, or
/// `Match`-scrutinized) stays an explicit refusal (a closure is not printable/branchable — DN-15 §7.4).
#[derive(Debug, Clone)]
pub(crate) struct ClosureVal {
    /// The closure parameter name (bound to the application argument when specialized).
    pub(crate) param: String,
    /// The closure body as an un-lowered ANF block, lowered inline at the application site.
    pub(crate) body: mycelium_core::lower::Anf,
    /// The captured environment — each free variable of the body mapped to its already-lowered
    /// [`EnvValue`] from the *defining* scope (a lane of any repr/width, or a nested `ClosureVal`).
    /// Snapshotting the values (not re-looking-them-up at the App site) keeps lexical capture correct
    /// when the closure is applied outside its defining scope (a returned/curried closure).
    pub(crate) captured: Vec<(String, EnvValue)>,
}

/// A suspended `Fix` value (Increment-3 / RFC-0004 §11.6): the Fix body (a `λparam. Match ...`)
/// stored without emitting IR. Consumed by a downstream `App` that triggers `lower_tail_fix`,
/// which rewrites the Fix+App into an iterative LLVM loop. A bare Fix that is never `App`-applied
/// is an explicit `UnsupportedNode` (G2 — it would leave the result on a non-printable value).
#[derive(Debug, Clone)]
pub(crate) struct FixVal {
    /// The self-reference name (the bound variable in `Fix name body`).
    pub(crate) name: String,
    /// The Fix body — a `λparam. Match param { arms }` ANF node (not yet lowered).
    pub(crate) body: mycelium_core::lower::Anf,
}

/// A suspended `FixGroup` member (M-850 / Wave-B; RFC-0004 §11.6; DN-15 §10): the group's lowered
/// member definitions plus which member this binding resolves to. Stored without emitting IR;
/// consumed by a downstream `App(member, init)` that routes to [`crate::trampoline`]. The
/// env-machine analogue is `aot.rs`'s `AotVal::FixGroup` (the focus-suspension unfold).
#[derive(Debug, Clone)]
pub(crate) struct FixGroupVal {
    /// All members of the group, in declaration order: `(member-name, lowered λ.Match body)`.
    pub(crate) defs: Vec<(String, mycelium_core::lower::Anf)>,
    /// Which member name this binding resolves to (the entry when this binding is the `App` func).
    pub(crate) which: String,
}

/// An environment value — a repr-lane (bit/trit), a constructed data value (tagged struct), a
/// native closure (Increment-2), a suspended Fix value (Increment-3), or a suspended FixGroup
/// member (M-850).
///
/// The `lower_program` env maps [`Atom`] → `EnvValue`. Repr-lane values flow into `emit_op`; datum
/// values are produced by `Construct` and consumed by `Match` arm bodies; closure values are
/// produced by `Lam` and consumed by `App`; Fix values are suspended by `Fix` and consumed by the
/// special `App(Fix, init)` dispatch in `lower_app`. Neither a datum, closure, nor Fix is ever a
/// final result (the output protocol prints bits/trits; leaving one on the result atom is refused
/// with an explicit [`AotError::UnsupportedNode`]).
#[derive(Debug, Clone)]
pub(crate) enum EnvValue {
    Repr(Lane),
    Datum(Datum),
    Closure(ClosureVal),
    /// A suspended `Fix` — produced by `Rhs::Fix`, consumed by the special `App(Fix, init)`
    /// dispatch. Never a printable result value (G2).
    Fix(FixVal),
    /// A suspended `FixGroup` member (M-850 / Wave-B) — produced by `Rhs::FixGroup`, consumed by the
    /// special `App(FixGroup-member, init)` dispatch which routes to the heap-trampoline
    /// ([`crate::trampoline`]). Carries the whole group's lowered defs plus which member this is
    /// (so a sibling call resolves the group). Never a printable result value (G2).
    FixGroup(FixGroupVal),
}

impl EnvValue {
    /// Extract the repr lane, or return an explicit error if it is a datum/closure.
    fn into_lane(self, ctx: &str) -> Result<Lane, AotError> {
        match self {
            EnvValue::Repr(l) => Ok(l),
            EnvValue::Datum(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a data value (only repr \
                 values are valid here)"
            ))),
            EnvValue::Closure(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a closure value — a closure is not a \
                 printable/repr value in the native ABI (Increment-2; DN-15 §7.4)"
            ))),
            EnvValue::Fix(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a Fix value — a bare Fix is not a \
                 printable/repr value; it must be applied to an argument (Increment-3; DN-15 §8; G2)"
            ))),
            EnvValue::FixGroup(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a FixGroup member — a bare FixGroup is not a \
                 printable/repr value; it must be applied to an argument (M-850; DN-15 §10; G2)"
            ))),
        }
    }
    fn as_lane(&self, ctx: &str) -> Result<&Lane, AotError> {
        match self {
            EnvValue::Repr(l) => Ok(l),
            EnvValue::Datum(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a data value"
            ))),
            EnvValue::Closure(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a closure value (Increment-2; DN-15 §7.4)"
            ))),
            EnvValue::Fix(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a Fix value — only usable as an App func \
                 (Increment-3; G2)"
            ))),
            EnvValue::FixGroup(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a repr lane but found a FixGroup member — only usable as an App \
                 func (M-850; G2)"
            ))),
        }
    }
    /// Extract the closure, or an explicit refusal if this is not a closure (e.g. `App` applied to a
    /// non-function value — never a silent miscall; G2).
    fn as_closure(&self, ctx: &str) -> Result<&ClosureVal, AotError> {
        match self {
            EnvValue::Closure(c) => Ok(c),
            EnvValue::Fix(_) => Err(AotError::UnsupportedNode(format!(
                "{ctx}: found a Fix value where a closure was expected — an App on a Fix is handled \
                 specially by lower_tail_fix, not via as_closure (Increment-3; G2)"
            ))),
            _ => Err(AotError::UnsupportedNode(format!(
                "{ctx}: expected a closure value but found a non-function value — only a `Lam` \
                 produces a callable closure (Increment-2; DN-15 §7.1)"
            ))),
        }
    }
}

/// SSA-name generator for the emitted IR (monotone counter → deterministic names).
pub(crate) struct Ssa(pub(crate) usize);
impl Ssa {
    pub(crate) fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("%r{n}")
    }
}

/// Basic-block label counter — gives every emitted control-flow label a unique name (monotone,
/// deterministic). Separate from the SSA counter so block names and register names never collide.
pub(crate) struct Bbc(pub(crate) usize);
impl Bbc {
    pub(crate) fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("bb{n}")
    }
}

/// The lowered program: the emitted op `body`, the `result` lane, and the SSA counter to continue
/// from. The **single source of truth** for [`emit_llvm_ir`], [`result_shape`], and the JIT
/// function emitter — so the shape used to parse the output can never disagree with what was emitted.
pub(crate) struct Lowered {
    pub(crate) body: String,
    pub(crate) result: Lane,
    pub(crate) ssa: Ssa,
    /// The combined runtime overflow flag — an `i1` SSA register that is the OR of every
    /// trit-arithmetic op's overflow condition, or `None` for a program that cannot overflow (no
    /// `trit.add/sub/mul`). The AOT/JIT emitters branch on it to drive the read-back protocol.
    pub(crate) overflow: Option<String>,
    /// **Vestigial since M-851 — always empty.** The narrow ABI (M-378) emitted one top-level
    /// `@myc_closureN` function per `Rhs::Lam` here; the widened ABI **inlines** closures at their
    /// application site ([`lower_app`]), so no top-level closure functions are produced. Retained only
    /// because the trampoline-shared lowering signatures thread a `funcs` sink (M-850); nothing writes
    /// to it now. Kept (not removed) to avoid churning the `crate::trampoline` interface.
    pub(crate) funcs: Vec<String>,
}

/// Lower a single field `Lane` into the struct at `ptr`, writing elements starting at `slot_start`
/// (each element occupies one i64 slot). Returns the `FieldLayout` for this field.
fn emit_store_field(
    lane: &Lane,
    ptr: &str,
    slots: usize,
    slot_start: usize,
    ssa: &mut Ssa,
    body: &mut String,
) -> FieldLayout {
    for (i, v) in lane.vals.iter().enumerate() {
        // Sign-extend / zero-extend the i32 element to i64 before storing.
        let ext = ssa.fresh();
        let _ = writeln!(body, "  {ext} = sext i32 {v} to i64");
        let gep = ssa.fresh();
        let slot = slot_start + i;
        let _ = writeln!(
            body,
            "  {gep} = getelementptr inbounds [{slots} x i64], [{slots} x i64]* {ptr}, i64 0, i64 {slot}"
        );
        let _ = writeln!(body, "  store i64 {ext}, i64* {gep}");
    }
    FieldLayout {
        kind: lane.kind,
        elems: lane.vals.len(),
        slot_start,
    }
}

/// Load one field from a struct at `ptr` given its `FieldLayout`, returning a `Lane` of i32
/// register operands (each element truncated from i64). The struct has `slots` total i64 slots.
fn emit_load_field(
    layout: &FieldLayout,
    ptr: &str,
    slots: usize,
    ssa: &mut Ssa,
    body: &mut String,
) -> Lane {
    let vals: Vec<Operand> = (0..layout.elems)
        .map(|i| {
            let slot = layout.slot_start + i;
            let gep = ssa.fresh();
            let _ = writeln!(
                body,
                "  {gep} = getelementptr inbounds [{slots} x i64], [{slots} x i64]* {ptr}, i64 0, i64 {slot}"
            );
            let loaded = ssa.fresh();
            let _ = writeln!(body, "  {loaded} = load i64, i64* {gep}");
            let trunc = ssa.fresh();
            let _ = writeln!(body, "  {trunc} = trunc i64 {loaded} to i32");
            trunc
        })
        .collect();
    Lane {
        kind: layout.kind,
        vals,
    }
}

/// Emit the `i32` ASCII char code for one result element of `kind` (operand `v`), returning the SSA
/// register holding it. Binary → `val + 48` (`'0'`/`'1'`); Ternary → `'-'`(45)/`'0'`(48)/`'+'`(43)
/// via a branch-free `select` chain. **Shared** by the AOT (`putchar`) and JIT (`store`) emitters so
/// their element encodings — and thus the read-back — can never diverge.
pub(crate) fn emit_char_code(kind: LaneKind, v: &str, ssa: &mut Ssa, body: &mut String) -> String {
    match kind {
        LaneKind::Binary => {
            let c = ssa.fresh();
            let _ = writeln!(body, "  {c} = add i32 {v}, 48");
            c
        }
        LaneKind::Ternary => {
            let isneg = ssa.fresh();
            let _ = writeln!(body, "  {isneg} = icmp eq i32 {v}, -1");
            let ispos = ssa.fresh();
            let _ = writeln!(body, "  {ispos} = icmp eq i32 {v}, 1");
            let t = ssa.fresh();
            let _ = writeln!(body, "  {t} = select i1 {ispos}, i32 43, i32 48");
            let c = ssa.fresh();
            let _ = writeln!(body, "  {c} = select i1 {isneg}, i32 45, i32 {t}");
            c
        }
    }
}

/// Decode `width` printed element chars (Binary: `'0'`/`'1'`; Ternary: `'-'`/`'0'`/`'+'`) into an
/// `Exact` `Value`. **Shared** by the AOT stdout read-back and the JIT buffer read-back.
pub(crate) fn decode_result(
    kind: LaneKind,
    width: usize,
    chars: impl Iterator<Item = char>,
) -> Result<Value, AotError> {
    let chars: Vec<char> = chars.collect();
    if chars.len() != width {
        return Err(AotError::Parse(format!(
            "expected {width} elements, got {} ({chars:?})",
            chars.len()
        )));
    }
    match kind {
        LaneKind::Binary => {
            let bits: Vec<bool> = chars
                .into_iter()
                .map(|c| match c {
                    '0' => Ok(false),
                    '1' => Ok(true),
                    other => Err(AotError::Parse(format!("non-bit char {other:?}"))),
                })
                .collect::<Result<_, _>>()?;
            Value::new(
                Repr::Binary {
                    width: width as u32,
                },
                Payload::Bits(bits),
                Meta::exact(Provenance::Root),
            )
            .map_err(|e| AotError::Wf(e.to_string()))
        }
        LaneKind::Ternary => {
            let trits: Vec<Trit> = chars
                .into_iter()
                .map(|c| match c {
                    '-' => Ok(Trit::Neg),
                    '0' => Ok(Trit::Zero),
                    '+' => Ok(Trit::Pos),
                    other => Err(AotError::Parse(format!("non-trit char {other:?}"))),
                })
                .collect::<Result<_, _>>()?;
            Value::new(
                Repr::Ternary {
                    trits: width as u32,
                },
                Payload::Trits(trits),
                Meta::exact(Provenance::Root),
            )
            .map_err(|e| AotError::Wf(e.to_string()))
        }
    }
}

/// Walk the lowered ANF, emitting one op per binding, and return the result lane. Returns an
/// explicit [`AotError`] for anything outside the bit/trit + non-recursive-data subset (M-301;
/// M-373). The env maps each bound atom to an [`EnvValue`] (either a repr lane or a datum struct).
pub(crate) fn lower_program(node: &Node) -> Result<Lowered, AotError> {
    // The default native swap cert mode is `Recheck` (compile-time independent re-check; M-852).
    lower_program_with_swap_mode(node, SwapCertMode::Recheck)
}

/// The source [`Repr`] a lowered [`Lane`] denotes — a `Binary` lane of `N` elements is
/// `Repr::Binary{ width: N }`, a `Ternary` lane of `M` elements is `Repr::Ternary{ trits: M }`. Used
/// to reconstruct the swap's *source* repr (the `Swap` node carries only the target). The element
/// count is exactly the repr width by construction of [`const_lane`]/`emit_op` (each lane element is
/// one bit/trit), so this reconstruction is exact — never a guess (G2).
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

/// Lower a program under an **explicit** native swap cert mode (M-852): `Recheck` (DEFAULT —
/// compile-time independent re-check of the bijection certificate) or `ReuseInterp` (OPT-IN — carry
/// the interpreter's certificate forward). The mode reaches the straight-line / let / non-recursive
/// match-arm `Swap` lowering sites; swaps inside a recursion *base arm* use the `Recheck` default
/// (the trampoline path is not threaded — still correct, never silent). [`lower_program`] delegates
/// here with `Recheck`.
pub(crate) fn lower_program_with_swap_mode(
    node: &Node,
    swap_mode: SwapCertMode,
) -> Result<Lowered, AotError> {
    let anf = lower::lower_to_anf(node);
    let mut env: HashMap<Atom, EnvValue> = HashMap::new();
    let mut ssa = Ssa(0);
    let mut bbc = Bbc(0);
    let mut body = String::new();
    // The per-op overflow `i1` registers, accumulated across the program. Any trit-arithmetic op
    // pushes its overflow condition here; the interpreter errors on the *first* overflow, so the
    // native path being conservative (OR of all of them ⇒ one explicit `Overflow`) gives the same
    // verdict — we never read the meaningless result either way.
    let mut flags: Vec<String> = Vec::new();
    // The emitted closure functions (Increment-2); one per `Rhs::Lam`. Stays empty for closure-free
    // programs, so their emitted module is unchanged.
    let mut funcs: Vec<String> = Vec::new();

    for b in anf.bindings() {
        let ev = match &b.rhs {
            Rhs::Const(v) => EnvValue::Repr(const_lane(v)?),
            Rhs::Alias(a) => lookup_ev(&env, a)?.clone(),
            Rhs::Op { prim, args } => {
                let operands: Vec<&Lane> = args
                    .iter()
                    .map(|a| lookup_ev(&env, a)?.as_lane("op operand"))
                    .collect::<Result<_, _>>()?;
                EnvValue::Repr(emit_op(prim, &operands, &mut ssa, &mut body, &mut flags)?)
            }
            // M-852 (E25-1; RFC-0002 §3/§4; RFC-0004 §6/§11): the **`Swap` node** — the one
            // Repr-changing node (WF1) — lowers natively for the certified binary↔ternary class
            // (+ same-Repr identity). The cert handling is the reified `swap_mode`: DEFAULT
            // `Recheck` re-runs the bijection cert check at compile time; OPT-IN `ReuseInterp`
            // carries the interpreter's cert forward. Range failures on the partial `dec` direction
            // push a never-silent overflow flag (matches `SwapError::OutOfRange`). Dense/VSA and
            // other swap kinds stay explicit `UnsupportedNode` (G2).
            Rhs::Swap { src, target, .. } => {
                let src_lane = lookup_ev(&env, src)?.as_lane("swap source")?;
                let src_repr = lane_repr(src_lane);
                let (lane, _explain) = swap_codegen::lower_swap(
                    src_lane, &src_repr, target, swap_mode, &mut ssa, &mut body, &mut flags,
                )?;
                EnvValue::Repr(lane)
            }
            // Increment-1 (M-373; DN-15 §4.1; RFC-0004 §11.2): Construct and Match are lowered for
            // the NON-RECURSIVE, BOUNDED case. Stack alloca is used (not malloc) because the
            // non-recursive/bounded restriction (no Fix/FixGroup in scope) makes all allocation depth
            // statically known at codegen time — no OOM path needed (G2 is satisfied by the explicit
            // UnsupportedNode refusal for Fix/FixGroup below). Guarantee: Declared (VR-5).
            Rhs::Construct { ctor, args } => {
                // Each field is a Lane; we store each element as one i64 slot after the tag.
                // Layout: [tag(i64), field_0_elem_0(i64), ..., field_0_elem_w-1, field_1_elem_0, ...]
                let field_lanes: Vec<Lane> = args
                    .iter()
                    .map(|a| lookup_ev(&env, a)?.as_lane("Construct field").cloned())
                    .collect::<Result<_, _>>()?;
                let total_elem: usize = field_lanes.iter().map(|l| l.vals.len()).sum();
                let slots = 1 + total_elem; // tag slot + one slot per element across all fields
                                            // Allocate the struct on the stack.
                let ptr = ssa.fresh();
                let _ = writeln!(body, "  {ptr} = alloca [{slots} x i64], align 8");
                // Store the tag (ctor.index() as i64) in slot 0.
                let tag_gep = ssa.fresh();
                let tag_val = ctor.index() as u64;
                let _ = writeln!(
                    body,
                    "  {tag_gep} = getelementptr inbounds [{slots} x i64], [{slots} x i64]* {ptr}, i64 0, i64 0"
                );
                let _ = writeln!(body, "  store i64 {tag_val}, i64* {tag_gep}");
                // Store each field's elements consecutively after the tag.
                let mut slot_start = 1usize;
                let mut field_layouts: Vec<FieldLayout> = Vec::with_capacity(field_lanes.len());
                for lane in &field_lanes {
                    let layout =
                        emit_store_field(lane, &ptr, slots, slot_start, &mut ssa, &mut body);
                    slot_start += lane.vals.len();
                    field_layouts.push(layout);
                }
                EnvValue::Datum(Datum {
                    ptr,
                    _tag: tag_val,
                    fields: field_layouts,
                    slots,
                })
            }
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => lower_match(
                scrutinee, alts, default, &env, &mut ssa, &mut bbc, &mut body, &mut funcs,
                &mut flags, swap_mode,
            )?,
            // M-378 + M-851 (DN-15 §7; RFC-0004 §11.5/§11.7): `Lam` builds a suspended closure value
            // (free-var snapshot) and `App` **specializes (inlines)** it at the call site over the
            // widened any-repr/width boxed-value model — currying + returned closures lower natively.
            // Fix/FixGroup route to the heap trampoline (M-850; DN-05 #1 stack-robustness; G2/VR-5).
            Rhs::Lam {
                param,
                body: lam_body,
            } => lower_lam(param, lam_body, &env)?,
            Rhs::App { func, arg } => lower_app(
                func, arg, &env, &mut ssa, &mut bbc, &mut body, &mut funcs, &mut flags, swap_mode,
            )?,
            Rhs::Fix {
                name,
                body: fix_body,
            } => {
                // Suspend the Fix as a value (no IR yet). Consumed by a downstream App.
                EnvValue::Fix(FixVal {
                    name: name.clone(),
                    body: fix_body.clone(),
                })
            }
            // M-850 (Wave-B): a `FixGroup` binding is now **suspended** as an `EnvValue::FixGroup`
            // (no IR yet), exactly as a `Fix` is suspended — consumed by a downstream
            // `App(member, init)` that routes to the heap-trampoline (RFC-0004 §11.6; DN-15 §10).
            Rhs::FixGroup { defs, which } => EnvValue::FixGroup(FixGroupVal {
                defs: defs.iter().map(|(n, d)| (n.clone(), d.clone())).collect(),
                which: which.clone(),
            }),
        };
        env.insert(b.name.clone(), ev);
    }

    let result_ev = lookup_ev(&env, anf.result())?.clone();
    let result = result_ev.into_lane("final program result")?;
    // Fold the per-op overflow flags into one `i1` (left-associative `or` chain), or `None`.
    let overflow = fold_or(&flags, &mut ssa, &mut body);
    Ok(Lowered {
        body,
        result,
        ssa,
        overflow,
        funcs,
    })
}

/// `pub(crate)` shim so the M-850 heap-trampoline ([`crate::trampoline`]) can reuse the exact
/// straight-line block lowering (DRY/KC-3 — one lowering, never a divergent copy). Same contract as
/// [`lower_anf_block`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_anf_block_pub(
    anf: &lower::Anf,
    env: &mut HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
) -> Result<Lane, AotError> {
    // The trampoline (recursion) path lowers `Binary{8}` straight-line + Match only — a `Swap`
    // there is already refused — so the swap cert mode is the `Recheck` default (M-852).
    lower_anf_block(
        anf,
        env,
        ssa,
        bbc,
        body,
        funcs,
        flags,
        SwapCertMode::Recheck,
    )
}

/// `pub(crate)` shim: lower every **support** binding of `anf` — every binding whose name is neither
/// `call_name` (the recursive `App(member, step)`, which the trampoline emits itself) nor
/// `result_name` (the wrapping `Binary{8}` op the trampoline applies as the defunctionalized
/// continuation) — extending `env`. The support set is the straight-line `Binary{8}` const/alias/op
/// bindings the `step` atom and the continuation's saved operand depend on.
///
/// **Why every binding, not "those before the call":** after ANF flattening the saved continuation
/// operand can be bound *after* the call binding (e.g. `%c=App(self,%s); %k=Const(mask);
/// %r=and(%c,%k)` — the mask `%k` follows the call), so a "stop at the call" walk would miss it and
/// the operand would be a free variable. Skipping exactly the call and result bindings (and lowering
/// everything else regardless of position) lowers each support binding exactly once. Any binding the
/// saved operand transitively needs precedes it in ANF order, so a single forward pass binds them all
/// before the continuation materializes the operand (G2: a still-missing operand is a free-variable
/// error, never silently mis-encoded).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_bindings_before_call_pub(
    anf: &lower::Anf,
    call_name: &Atom,
    result_name: &Atom,
    env: &mut HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    // The support sequence is straight-line `Binary{8}` (const/alias/op only), so block-label and
    // closure-function sinks are unused here — kept in the signature for a uniform lowering surface.
    _bbc: &mut Bbc,
    body: &mut String,
    _funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
) -> Result<(), AotError> {
    for b in anf.bindings() {
        if &b.name == call_name || &b.name == result_name {
            // The recursive-call binding and the wrapping-op result binding are emitted by the
            // trampoline itself (the call + the defunctionalized continuation); skip both, lower the
            // rest (which includes any saved-operand binding that follows the call in ANF order).
            continue;
        }
        let ev = match &b.rhs {
            Rhs::Const(v) => EnvValue::Repr(const_lane(v)?),
            Rhs::Alias(a) => lookup_ev(env, a)?.clone(),
            Rhs::Op { prim, args } => {
                let operands: Vec<&Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("trampoline pre-call op operand"))
                    .collect::<Result<_, _>>()?;
                EnvValue::Repr(emit_op(prim, &operands, ssa, body, flags)?)
            }
            // A non-canonical pre-call construct (Swap/Construct/Match/Lam/Fix/App-to-non-member/
            // FixGroup) is refused — the trampoline pre-call sequence is straight-line Binary{8}
            // only; anything else is routed to the interpreter (never fragile IR — G2/VR-5).
            other => {
                return Err(AotError::UnsupportedNode(format!(
                    "trampoline: a non-straight-line pre-call binding ({}) is not supported in a \
                     recursive arm — only Binary{{8}} const/alias/op before the call (G2)",
                    rhs_kind(other)
                )));
            }
        };
        env.insert(b.name.clone(), ev);
    }
    Ok(())
}

/// A short human label for an `Rhs` variant (diagnostics only).
fn rhs_kind(rhs: &Rhs) -> &'static str {
    match rhs {
        Rhs::Const(_) => "Const",
        Rhs::Alias(_) => "Alias",
        Rhs::Op { .. } => "Op",
        Rhs::Swap { .. } => "Swap",
        Rhs::Construct { .. } => "Construct",
        Rhs::App { .. } => "App",
        Rhs::Lam { .. } => "Lam",
        Rhs::Fix { .. } => "Fix",
        Rhs::FixGroup { .. } => "FixGroup",
        Rhs::Match { .. } => "Match",
    }
}

/// `pub(crate)` shims for the narrow-ABI pack/unpack/typecheck helpers the trampoline shares.
pub(crate) fn pack_binary8_pub(lane: &Lane, ssa: &mut Ssa, body: &mut String) -> String {
    pack_binary8(lane, ssa, body)
}
pub(crate) fn unpack_binary8_pub(src: &str, ssa: &mut Ssa, body: &mut String) -> Lane {
    unpack_binary8(src, ssa, body)
}
pub(crate) fn as_binary8_pub<'a>(ev: &'a EnvValue, ctx: &str) -> Result<&'a Lane, AotError> {
    as_binary8(ev, ctx)
}
pub(crate) fn lit_binary8_packed_pub(value: &Value) -> Result<u64, AotError> {
    lit_binary8_packed(value)
}

/// Lower a nested ANF block (a `Match` arm or similar nested scope) into the ongoing IR stream,
/// extending `env` with any new bindings. Returns the result `Lane` of the nested block (forcing the
/// result to a repr lane). This is the recursive workhorse for `Rhs::Match` arm bodies.
#[allow(clippy::too_many_arguments)]
fn lower_anf_block(
    anf: &lower::Anf,
    env: &mut HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    swap_mode: SwapCertMode,
) -> Result<Lane, AotError> {
    lower_anf_block_ev(anf, env, ssa, bbc, body, funcs, flags, swap_mode)?
        .into_lane("match arm result")
}

/// Lower a nested ANF block returning its raw [`EnvValue`] result — a lane **or** a closure. The
/// closure-result case is what makes **currying / returned closures** lowerable (M-851): a closure
/// body whose result is itself a `Lam` / an `App`-returning-a-closure yields an `EnvValue::Closure`,
/// which the caller (an enclosing `App`, or a `let` binding) consumes directly rather than forcing it
/// to a printable lane. [`lower_anf_block`] is the lane-forcing wrapper for the (common) repr-result.
#[allow(clippy::too_many_arguments)]
fn lower_anf_block_ev(
    anf: &lower::Anf,
    env: &mut HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    swap_mode: SwapCertMode,
) -> Result<EnvValue, AotError> {
    for b in anf.bindings() {
        let ev = match &b.rhs {
            Rhs::Const(v) => EnvValue::Repr(const_lane(v)?),
            Rhs::Alias(a) => lookup_ev(env, a)?.clone(),
            Rhs::Op { prim, args } => {
                let operands: Vec<&Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("op operand"))
                    .collect::<Result<_, _>>()?;
                EnvValue::Repr(emit_op(prim, &operands, ssa, body, flags)?)
            }
            // M-852: native `Swap` lowering inside a nested block (a `let`-bound or match-arm swap).
            Rhs::Swap { src, target, .. } => {
                let src_lane = lookup_ev(env, src)?.as_lane("swap source")?;
                let src_repr = lane_repr(src_lane);
                let (lane, _explain) = swap_codegen::lower_swap(
                    src_lane, &src_repr, target, swap_mode, ssa, body, flags,
                )?;
                EnvValue::Repr(lane)
            }
            Rhs::Construct { ctor, args } => {
                let field_lanes: Vec<Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("Construct field").cloned())
                    .collect::<Result<_, _>>()?;
                let total_elem: usize = field_lanes.iter().map(|l| l.vals.len()).sum();
                let slots = 1 + total_elem;
                let ptr = ssa.fresh();
                let _ = writeln!(body, "  {ptr} = alloca [{slots} x i64], align 8");
                let tag_gep = ssa.fresh();
                let tag_val = ctor.index() as u64;
                let _ = writeln!(
                    body,
                    "  {tag_gep} = getelementptr inbounds [{slots} x i64], [{slots} x i64]* {ptr}, i64 0, i64 0"
                );
                let _ = writeln!(body, "  store i64 {tag_val}, i64* {tag_gep}");
                let mut slot_start = 1usize;
                let mut field_layouts: Vec<FieldLayout> = Vec::with_capacity(field_lanes.len());
                for lane in &field_lanes {
                    let layout = emit_store_field(lane, &ptr, slots, slot_start, ssa, body);
                    slot_start += lane.vals.len();
                    field_layouts.push(layout);
                }
                EnvValue::Datum(Datum {
                    ptr,
                    _tag: tag_val,
                    fields: field_layouts,
                    slots,
                })
            }
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => lower_match(
                scrutinee, alts, default, env, ssa, bbc, body, funcs, flags, swap_mode,
            )?,
            // Increment-2: closures are lowered inside match arms too (a `Lam`/`App` may appear in an
            // arm body). Fix/FixGroup stay explicit UnsupportedNode (Increment-3; G2/VR-5).
            Rhs::Lam {
                param,
                body: lam_body,
            } => lower_lam(param, lam_body, env)?,
            Rhs::App { func, arg } => {
                lower_app(func, arg, env, ssa, bbc, body, funcs, flags, swap_mode)?
            }
            Rhs::Fix {
                name,
                body: fix_body,
            } => {
                // Suspend the Fix as a value (no IR yet). Consumed by a downstream App.
                // (Increment-2/3: closures + Fix are lowered inside nested blocks too.)
                EnvValue::Fix(FixVal {
                    name: name.clone(),
                    body: fix_body.clone(),
                })
            }
            // M-850: suspend the FixGroup member (consumed by a downstream App → heap-trampoline).
            Rhs::FixGroup { defs, which } => EnvValue::FixGroup(FixGroupVal {
                defs: defs.iter().map(|(n, d)| (n.clone(), d.clone())).collect(),
                which: which.clone(),
            }),
        };
        env.insert(b.name.clone(), ev);
    }
    Ok(lookup_ev(env, anf.result())?.clone())
}

/// Compute the **free `Named` variables** of a closure body `body` whose parameter is `param`, in
/// deterministic first-encounter order — the closure's captured set (Increment-2; DN-15 §7.3). A
/// name is free iff it is referenced (directly, or inside a nested lambda / match arm) and is neither
/// `param` nor bound by an enclosing binding / match binder / nested lambda parameter *within*
/// `body`. Lexical scoping is honoured (each nested scope's binders are removed while inside it).
/// Only `Named` atoms are captured; `Temp` operands are always block-local in this ANF, so a closure
/// body never has a free temp — and if one ever did, the closure-body lowering would surface it as an
/// explicit [`AotError::FreeVariable`] (never a silent miscapture; G2).
fn closure_free_vars(body: &lower::Anf, param: &str) -> Vec<String> {
    use std::collections::HashSet;
    let mut bound: HashSet<String> = HashSet::new();
    bound.insert(param.to_owned());
    let mut free: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    free_vars_into(body, &mut bound, &mut free, &mut seen);
    free
}

/// Record `a` as free if it is a `Named` atom not in `bound` and not already captured (dedup via
/// `seen`, preserving first-encounter order).
fn note_free_atom(
    a: &Atom,
    bound: &std::collections::HashSet<String>,
    free: &mut Vec<String>,
    seen: &mut std::collections::HashSet<String>,
) {
    if let Atom::Named(n) = a {
        if !bound.contains(n) && seen.insert(n.clone()) {
            free.push(n.clone());
        }
    }
}

/// The lexical free-variable walk backing [`closure_free_vars`]. Adds a scope's binders to `bound`
/// while descending into it and removes them on the way out, so shadowing is honoured.
fn free_vars_into(
    anf: &lower::Anf,
    bound: &mut std::collections::HashSet<String>,
    free: &mut Vec<String>,
    seen: &mut std::collections::HashSet<String>,
) {
    use mycelium_core::lower::AnfAlt;
    for b in anf.bindings() {
        match &b.rhs {
            Rhs::Const(_) => {}
            Rhs::Alias(a) => note_free_atom(a, bound, free, seen),
            Rhs::Op { args, .. } | Rhs::Construct { args, .. } => {
                for a in args {
                    note_free_atom(a, bound, free, seen);
                }
            }
            Rhs::Swap { src, .. } => note_free_atom(src, bound, free, seen),
            Rhs::App { func, arg } => {
                note_free_atom(func, bound, free, seen);
                note_free_atom(arg, bound, free, seen);
            }
            Rhs::Lam {
                param,
                body: lam_body,
            } => {
                let added = bound.insert(param.clone());
                free_vars_into(lam_body, bound, free, seen);
                if added {
                    bound.remove(param);
                }
            }
            Rhs::Fix { name, body: fbody } => {
                let added = bound.insert(name.clone());
                free_vars_into(fbody, bound, free, seen);
                if added {
                    bound.remove(name);
                }
            }
            Rhs::FixGroup { defs, .. } => {
                let added: Vec<String> = defs
                    .iter()
                    .filter(|(n, _)| bound.insert(n.clone()))
                    .map(|(n, _)| n.clone())
                    .collect();
                for (_, d) in defs {
                    free_vars_into(d, bound, free, seen);
                }
                for n in added {
                    bound.remove(&n);
                }
            }
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => {
                note_free_atom(scrutinee, bound, free, seen);
                for alt in alts {
                    match alt {
                        AnfAlt::Ctor {
                            binders,
                            body: arm_body,
                            ..
                        } => {
                            let added: Vec<String> = binders
                                .iter()
                                .filter(|x| bound.insert((*x).clone()))
                                .cloned()
                                .collect();
                            free_vars_into(arm_body, bound, free, seen);
                            for x in added {
                                bound.remove(&x);
                            }
                        }
                        AnfAlt::Lit { body: arm_body, .. } => {
                            free_vars_into(arm_body, bound, free, seen);
                        }
                    }
                }
                if let Some(d) = default {
                    free_vars_into(d, bound, free, seen);
                }
            }
        }
        // The binding's own name becomes bound for subsequent bindings in this block.
        if let Atom::Named(n) = &b.name {
            bound.insert(n.clone());
        }
    }
    note_free_atom(anf.result(), bound, free, seen);
}

/// Require an [`EnvValue`] to be a `Binary{8}` lane — the only repr that the trampoline
/// recursion-accumulator ABI (DN-15 §7.1/§10) and the `Match` branch-primitive (`Lit`-arm switch;
/// DN-15 §8.3) accept. Explicit refusal otherwise (G2). *(Before M-851 this was also the closure
/// boundary check; the widened closure ABI now carries any repr/width via inlining, so this
/// function's scope is the trampoline + branch-primitive subset only.)*
fn as_binary8<'a>(ev: &'a EnvValue, ctx: &str) -> Result<&'a Lane, AotError> {
    let lane = ev.as_lane(ctx)?;
    if lane.kind != LaneKind::Binary || lane.vals.len() != CLOSURE_ABI_WIDTH {
        return Err(AotError::UnsupportedNode(format!(
            "{ctx}: the native closure ABI (Increment-2) carries only Binary{{{CLOSURE_ABI_WIDTH}}} \
             values packed as one i64; got {:?} of width {}",
            lane.kind,
            lane.vals.len()
        )));
    }
    Ok(lane)
}

/// Pack a `Binary{8}` lane (8 `i32` elements in `{0,1}`) into a single `i64` (element `i` → bit `i`).
/// The inverse of [`unpack_binary8`]; the two define the narrow closure-ABI encoding (DN-15 §7.1).
fn pack_binary8(lane: &Lane, ssa: &mut Ssa, body: &mut String) -> String {
    let mut acc = "0".to_owned();
    for (i, v) in lane.vals.iter().enumerate() {
        let z = ssa.fresh();
        let _ = writeln!(body, "  {z} = zext i32 {v} to i64");
        let sh = ssa.fresh();
        let _ = writeln!(body, "  {sh} = shl i64 {z}, {i}");
        let next = ssa.fresh();
        let _ = writeln!(body, "  {next} = or i64 {acc}, {sh}");
        acc = next;
    }
    acc
}

/// Unpack a single `i64` into a `Binary{8}` lane (bit `i` → element `i`, as an `i32` in `{0,1}`). The
/// inverse of [`pack_binary8`].
fn unpack_binary8(src: &str, ssa: &mut Ssa, body: &mut String) -> Lane {
    let vals = (0..CLOSURE_ABI_WIDTH)
        .map(|i| {
            let sh = ssa.fresh();
            let _ = writeln!(body, "  {sh} = lshr i64 {src}, {i}");
            let m = ssa.fresh();
            let _ = writeln!(body, "  {m} = and i64 {sh}, 1");
            let t = ssa.fresh();
            let _ = writeln!(body, "  {t} = trunc i64 {m} to i32");
            t
        })
        .collect();
    Lane {
        kind: LaneKind::Binary,
        vals,
    }
}

/// Lower `Rhs::Lam` (M-851 widened closure ABI; DN-15 §7.1 widening). Emits **no IR** — it builds a
/// **suspended** [`ClosureVal`] (the `param`, the un-lowered `body`, and a snapshot of each captured
/// free var's already-lowered [`EnvValue`]). The body is lowered later, **inlined at the application
/// site** ([`lower_app`]), where the argument pins the param's concrete shape — so closures over any
/// repr/width, currying, and returned closures lower without a fixed wire ABI or a guessed width.
/// Capturing the values *now* (snapshot) keeps lexical scope correct when the closure is applied
/// outside its defining scope (a returned/curried closure). A free var that is not in the defining
/// env is an explicit [`AotError::FreeVariable`] (never a silent miscapture — G2).
fn lower_lam(
    param: &str,
    body: &lower::Anf,
    env: &HashMap<Atom, EnvValue>,
) -> Result<EnvValue, AotError> {
    let capture_names = closure_free_vars(body, param);
    let mut captured: Vec<(String, EnvValue)> = Vec::with_capacity(capture_names.len());
    for capname in &capture_names {
        // Snapshot the captured value from the *defining* scope. Only repr lanes and (nested) closures
        // may be captured; a datum/Fix/FixGroup capture is an explicit refusal (DN-15 §7.4; G2).
        let cev = lookup_ev(env, &Atom::Named(capname.clone()))?.clone();
        match &cev {
            EnvValue::Repr(_) | EnvValue::Closure(_) => {}
            EnvValue::Datum(_) => {
                return Err(AotError::UnsupportedNode(format!(
                    "closure capture `{capname}`: a constructed data value cannot be captured by a \
                     closure (the widened closure ABI carries repr lanes and closures; DN-15 §7.4; G2)"
                )));
            }
            EnvValue::Fix(_) | EnvValue::FixGroup(_) => {
                return Err(AotError::UnsupportedNode(format!(
                    "closure capture `{capname}`: a Fix/FixGroup value cannot be captured — it must \
                     be applied (App) to run via the trampoline, not captured (M-850; G2)"
                )));
            }
        }
        captured.push((capname.clone(), cev));
    }
    Ok(EnvValue::Closure(ClosureVal {
        param: param.to_owned(),
        body: body.clone(),
        captured,
    }))
}

/// Lower `Rhs::App` (DN-15 §7.3/§8, closure path widened by M-851): paths dispatched on `func`'s
/// `EnvValue`:
/// - **`EnvValue::Fix`/`FixGroup`**: recursion → trampoline / iterative loop (M-850; narrow ABI).
/// - **`EnvValue::Closure`**: the closure is **specialized (inlined) at this site** — the body is
///   lowered into the current block with the param bound to the (concrete-shape) argument and each
///   captured free var restored, returning the body's [`EnvValue`] (a lane of any repr/width, or a
///   nested closure for currying / a returned closure). No fixed wire ABI, no indirect call, no
///   guessed param width — every shape is statically resolved here (G2/VR-5; DN-15 §7.1 widening).
///
/// Anything else is an explicit refusal (G2 — never silent).
#[allow(clippy::too_many_arguments)]
fn lower_app(
    func: &Atom,
    arg: &Atom,
    env: &HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    swap_mode: SwapCertMode,
) -> Result<EnvValue, AotError> {
    let func_ev = lookup_ev(env, func)?;
    // Increment-3: App(Fix, init) → iterative tail-recursion loop (never stack recursion; DN-05 #1).
    // M-850 (Wave-B): a non-tail single `Fix` (or a `Match` in a pre-call binding — DN-15 §8.5) no
    // longer refuses; it routes to the **heap-trampoline** ([`crate::trampoline`]) — the full
    // defunctionalized control-stack lowering DN-15 §4.3 anticipated. The fast iterative tail loop is
    // kept (byte-identical IR) for the pure-tail/base fragment it already handled; anything heavier
    // takes the trampoline. Both are bounded by the SAME `AutoDepthBudget` (DRY/KC-3).
    if let EnvValue::Fix(fixval) = func_ev {
        let members = crate::trampoline::destructure_fix(&fixval.name, &fixval.body)?;
        if crate::trampoline::is_pure_tail_single_fix(&members)? {
            return lower_tail_fix(fixval, arg, env, ssa, bbc, body, funcs, flags);
        }
        return crate::trampoline::lower_recursion_group(
            &members, 0, arg, env, ssa, bbc, body, funcs, flags,
        );
    }
    // M-850: App(FixGroup-member, init) → heap-trampoline over all members; `which` selects the
    // entry. Mutual recursion is resolved by the trampoline's shared member dispatch (RFC-0004
    // §11.6; DN-15 §10). Never the C stack — bounded by the same AutoDepthBudget (DN-05 #1).
    if let EnvValue::FixGroup(fg) = func_ev {
        let members = crate::trampoline::destructure_fixgroup(&fg.defs)?;
        let entry = fg
            .defs
            .iter()
            .position(|(n, _)| n == &fg.which)
            .ok_or_else(|| AotError::FreeVariable(fg.which.clone()))?;
        return crate::trampoline::lower_recursion_group(
            &members, entry, arg, env, ssa, bbc, body, funcs, flags,
        );
    }
    // M-851: specialize (inline) the closure at this application. Build a fresh env from the closure's
    // captured snapshot + the param bound to the argument's already-lowered value (any repr/width, or a
    // nested closure), then lower the closure body into the current block. The argument's concrete
    // shape flows in directly, so no width is ever guessed (G2). A closure cannot see the enclosing
    // function's bindings — only its captures + param (the snapshot is the lexical environment).
    let closure = func_ev.as_closure("App function")?.clone();
    let arg_ev = lookup_ev(env, arg)?.clone();
    let mut cenv: HashMap<Atom, EnvValue> = HashMap::with_capacity(closure.captured.len() + 1);
    for (capname, cev) in &closure.captured {
        cenv.insert(Atom::Named(capname.clone()), cev.clone());
    }
    cenv.insert(Atom::Named(closure.param.clone()), arg_ev);
    lower_anf_block_ev(
        &closure.body,
        &mut cenv,
        ssa,
        bbc,
        body,
        funcs,
        flags,
        swap_mode,
    )
}

/// Pack a `Binary{8}` literal `Value` (a Match `Lit`-arm pattern) into the `u64` the native branch
/// switch compares against the packed scrutinee (DN-15 §8.3). Bit `i` → position `i`, identical to
/// [`pack_binary8`], so the comparison is exact. Other reprs/widths are an explicit refusal (G2).
fn lit_binary8_packed(value: &Value) -> Result<u64, AotError> {
    match (value.repr(), value.payload()) {
        (Repr::Binary { width }, Payload::Bits(bits))
            if *width as usize == CLOSURE_ABI_WIDTH && bits.len() == CLOSURE_ABI_WIDTH =>
        {
            let mut acc = 0u64;
            for (i, &b) in bits.iter().enumerate() {
                if b {
                    acc |= 1u64 << i;
                }
            }
            Ok(acc)
        }
        (repr, _) => Err(AotError::UnsupportedNode(format!(
            "Match Lit-arm pattern must be a Binary{{{CLOSURE_ABI_WIDTH}}} value in the native branch \
             primitive (Increment-3); got {repr:?}"
        ))),
    }
}

/// Lower `Rhs::Match` — shared by [`lower_program`] and nested arm/closure bodies (so the two paths
/// never drift). Two forms, dispatched on the scrutinee's [`EnvValue`]:
/// - **`Datum` scrutinee + `Ctor` arms** (Increment-1): load the tag from slot 0, `switch i64` on the
///   constructor index, bind each arm's fields, phi-merge.
/// - **`Binary{8}` lane scrutinee + `Lit` arms** (Increment-3 — the native **branch primitive**;
///   DN-15 §8.3): pack the lane to `i64`, `switch i64` on the packed literal patterns (`Lit` arms bind
///   nothing), phi-merge. This is the base-case conditional a terminating recursion needs.
///
/// A no-match with no ANF `default` traps with `@abort` (a defined trap, never raw `unreachable`; G2).
/// Mixing forms (a `Lit` arm on a datum, or a `Ctor` arm on a lane) is an explicit refusal.
#[allow(clippy::too_many_arguments)]
fn lower_match(
    scrutinee: &Atom,
    alts: &[lower::AnfAlt],
    default_arm: &Option<lower::Anf>,
    env: &HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    swap_mode: SwapCertMode,
) -> Result<EnvValue, AotError> {
    use mycelium_core::lower::AnfAlt;
    let scrut = lookup_ev(env, scrutinee)?.clone();

    // The discriminant `i64` the switch compares, plus (for a datum) the datum layout for field
    // binding. A Binary lane is the branch-primitive scrutinee; only `Datum`/`Binary` are valid.
    let datum_opt: Option<Datum> = match &scrut {
        EnvValue::Datum(d) => Some(d.clone()),
        EnvValue::Repr(lane) if lane.kind == LaneKind::Binary => None,
        EnvValue::Repr(_) => {
            return Err(AotError::UnsupportedNode(
                "Match on a non-Binary repr lane (the native branch primitive compares Binary{8} \
                 values; Ternary scrutinees are out of scope — G2)"
                    .to_owned(),
            ));
        }
        EnvValue::Closure(_) => {
            return Err(AotError::UnsupportedNode(
                "Match on a closure value is not supported (G2)".to_owned(),
            ));
        }
        EnvValue::Fix(_) => {
            return Err(AotError::UnsupportedNode(
                "Match on a Fix value is not supported — a Fix must be applied (App) before it \
                 can be matched (Increment-3; G2)"
                    .to_owned(),
            ));
        }
        EnvValue::FixGroup(_) => {
            return Err(AotError::UnsupportedNode(
                "Match on a FixGroup member is not supported — it must be applied (App) before it \
                 can be matched (M-850; G2)"
                    .to_owned(),
            ));
        }
    };

    let disc_reg = match &datum_opt {
        Some(datum) => {
            let slots = datum.slots;
            let ptr = &datum.ptr;
            let tag_gep = ssa.fresh();
            let _ = writeln!(
                body,
                "  {tag_gep} = getelementptr inbounds [{slots} x i64], [{slots} x i64]* {ptr}, i64 0, i64 0"
            );
            let tag_reg = ssa.fresh();
            let _ = writeln!(body, "  {tag_reg} = load i64, i64* {tag_gep}");
            tag_reg
        }
        None => {
            // Binary-lane scrutinee: the packed lane is the discriminant the switch compares.
            let lane = as_binary8(&scrut, "Match scrutinee")?;
            pack_binary8(lane, ssa, body)
        }
    };

    let arm_labels: Vec<String> = (0..alts.len()).map(|_| bbc.fresh()).collect();
    let default_label = bbc.fresh();
    let merge_label = bbc.fresh();

    // The switch: per-arm value is the ctor index (datum form) or the packed `Lit` (Binary form).
    let _ = write!(body, "  switch i64 {disc_reg}, label %{default_label} [");
    for (alt, label) in alts.iter().zip(&arm_labels) {
        let sw_val: u64 = match (&datum_opt, alt) {
            (Some(_), AnfAlt::Ctor { ctor, .. }) => u64::from(ctor.index()),
            (None, AnfAlt::Lit { value, .. }) => lit_binary8_packed(value)?,
            (Some(_), AnfAlt::Lit { .. }) => {
                return Err(AotError::UnsupportedNode(
                    "literal arm on a constructed-data Match scrutinee — constructor arms only for \
                     data values (G2)"
                        .to_owned(),
                ));
            }
            (None, AnfAlt::Ctor { .. }) => {
                return Err(AotError::UnsupportedNode(
                    "constructor arm on a Binary Match scrutinee — literal arms only for repr \
                     values (G2)"
                        .to_owned(),
                ));
            }
        };
        let _ = write!(body, " i64 {sw_val}, label %{label}");
    }
    let _ = writeln!(body, " ]");

    // Each arm: bind fields (Ctor) or nothing (Lit), lower the body, collect for the phi.
    let mut phi_entries: Vec<(String, Lane)> = Vec::with_capacity(alts.len());
    for (alt, label) in alts.iter().zip(&arm_labels) {
        let _ = writeln!(body, "{label}:");
        let mut arm_env = env.clone();
        let arm_body = match alt {
            AnfAlt::Ctor {
                binders,
                body: arm_body,
                ..
            } => {
                let datum = datum_opt
                    .as_ref()
                    .expect("a Ctor arm implies a Datum scrutinee (checked in the switch above)");
                // Never-silent (G2): the interpreter rejects arity mismatch with DataMalformed.
                if binders.len() != datum.fields.len() {
                    return Err(AotError::UnsupportedNode(format!(
                        "Match arm binder arity ({}) != constructor field count ({}) — malformed \
                         Match (interpreter rejects with DataMalformed; G2/WF7)",
                        binders.len(),
                        datum.fields.len()
                    )));
                }
                let slots = datum.slots;
                let ptr = &datum.ptr;
                for (binder, field_layout) in binders.iter().zip(&datum.fields) {
                    let field_lane = emit_load_field(field_layout, ptr, slots, ssa, body);
                    arm_env.insert(Atom::Named(binder.clone()), EnvValue::Repr(field_lane));
                }
                arm_body
            }
            // A `Lit` arm binds nothing — the literal is matched by the switch value.
            AnfAlt::Lit { body: arm_body, .. } => arm_body,
        };
        let arm_result = lower_anf_block(
            arm_body,
            &mut arm_env,
            ssa,
            bbc,
            body,
            funcs,
            flags,
            swap_mode,
        )?;
        phi_entries.push((label.clone(), arm_result));
        let _ = writeln!(body, "  br label %{merge_label}");
    }

    // Default block: lower the ANF default (Some) or trap with abort() (None) — never raw UB (G2).
    let _ = writeln!(body, "{default_label}:");
    if let Some(default_block) = default_arm {
        let default_result = lower_anf_block(
            default_block,
            &mut env.clone(),
            ssa,
            bbc,
            body,
            funcs,
            flags,
            swap_mode,
        )?;
        phi_entries.push((default_label.clone(), default_result));
        let _ = writeln!(body, "  br label %{merge_label}");
    } else {
        let _ = writeln!(body, "  call void @abort()");
        let _ = writeln!(body, "  ret i32 0");
    }

    // Merge block: all arms must yield the same lane kind/width; phi per element.
    let _ = writeln!(body, "{merge_label}:");
    if phi_entries.is_empty() {
        return Err(AotError::UnsupportedNode(
            "Match with zero arms (exhaustive coverage requires at least one arm or a default)"
                .to_owned(),
        ));
    }
    let first = &phi_entries[0].1;
    let kind = first.kind;
    let width = first.vals.len();
    for (_, lane) in &phi_entries[1..] {
        if lane.kind != kind || lane.vals.len() != width {
            return Err(AotError::UnsupportedNode(
                "Match arms produce lanes of different kind or width — all arms must return the same \
                 repr shape"
                    .to_owned(),
            ));
        }
    }
    let mut result_vals: Vec<Operand> = Vec::with_capacity(width);
    for elem_idx in 0..width {
        let phi_reg = ssa.fresh();
        let phi_operands: Vec<String> = phi_entries
            .iter()
            .map(|(lbl, lane)| format!("[ {}, %{lbl} ]", lane.vals[elem_idx]))
            .collect();
        let _ = writeln!(body, "  {phi_reg} = phi i32 {}", phi_operands.join(", "));
        result_vals.push(phi_reg);
    }
    Ok(EnvValue::Repr(Lane {
        kind,
        vals: result_vals,
    }))
}

/// Tail-recursion classification of a Fix-Match arm body (Increment-3; RFC-0004 §11.6).
/// Each arm's body is inspected (never lowered yet) to determine whether it is:
/// - **Tail**: the arm body's result atom is bound by `Rhs::App{func: Named(self_name), arg}` and
///   `self_name` appears nowhere else in the arm body. If so, the step atom is extracted.
/// - **Base**: `self_name` does not appear anywhere in the arm body (no recursion in arm).
/// - **NonTail**: `self_name` appears in a non-tail position → `UnsupportedNode` (G2).
#[derive(Debug)]
enum ArmKind {
    /// A tail self-call: `App(self, step)` is the result; carry the `step` atom.
    Tail(Atom),
    /// A base case: no reference to the self-name anywhere.
    Base,
}

/// Scan every binding and the result atom of `anf` for any reference to `self_name` as a
/// `Named` atom. Returns `true` if found anywhere.
fn anf_refs_name(anf: &lower::Anf, self_name: &str) -> bool {
    use mycelium_core::lower::AnfAlt;
    let named = |a: &Atom| matches!(a, Atom::Named(n) if n == self_name);
    for b in anf.bindings() {
        let found = match &b.rhs {
            Rhs::Const(_) => false,
            Rhs::Alias(a) => named(a),
            Rhs::Op { args, .. } | Rhs::Construct { args, .. } => args.iter().any(named),
            Rhs::Swap { src, .. } => named(src),
            Rhs::App { func, arg } => named(func) || named(arg),
            // A nested binder that rebinds `self_name` SHADOWS it — references inside that scope are
            // not the outer self, so do NOT descend (A4-01: respect shadowing, as `free_vars_into`
            // does). Core lowering keeps source `Named` binders un-alpha-renamed, so this is reachable.
            Rhs::Lam { param, body: lb } => {
                param.as_str() != self_name && anf_refs_name(lb, self_name)
            }
            Rhs::Fix { name, body: fb } => {
                name.as_str() != self_name && anf_refs_name(fb, self_name)
            }
            Rhs::FixGroup { defs, .. } => {
                // Every member name is in scope of every def, so if any member shadows `self_name`
                // the whole group rebinds it — skip; otherwise descend into each def.
                !defs.iter().any(|(n, _)| n.as_str() == self_name)
                    && defs.iter().any(|(_, d)| anf_refs_name(d, self_name))
            }
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => {
                named(scrutinee)
                    || alts.iter().any(|alt| match alt {
                        AnfAlt::Ctor { body: ab, .. } | AnfAlt::Lit { body: ab, .. } => {
                            anf_refs_name(ab, self_name)
                        }
                    })
                    || default
                        .as_ref()
                        .is_some_and(|d| anf_refs_name(d, self_name))
            }
        };
        if found {
            return true;
        }
    }
    named(anf.result())
}

/// Classify the arm body of a Fix-Match arm (Increment-3; RFC-0004 §11.6).
///
/// An arm is a **tail call** iff (1) its result atom is bound (in that Anf) by
/// `Rhs::App{func: Named(self_name), arg: step}` and (2) `self_name` does not appear anywhere
/// ELSE in the arm body (only in the final App binding). An arm is a **base case** iff
/// `self_name` does not appear anywhere in the arm body at all. Any other occurrence
/// (self used as non-call, or nested recursion) returns `UnsupportedNode` (G2).
fn classify_arm(arm_body: &lower::Anf, self_name: &str) -> Result<ArmKind, AotError> {
    let result_atom = arm_body.result();
    // Check if the result atom is bound by App(self_name, step) in this Anf.
    let tail_step: Option<Atom> = arm_body.bindings().iter().find_map(|b| {
        if &b.name != result_atom {
            return None;
        }
        if let Rhs::App {
            func: Atom::Named(fname),
            arg: step,
        } = &b.rhs
        {
            if fname == self_name {
                return Some(step.clone());
            }
        }
        None
    });

    if let Some(step) = tail_step {
        // It's a tail call. Verify self_name appears ONLY in the final App binding — not elsewhere.
        // Build a trimmed anf without the final binding to check for stray references.
        let bindings_without_tail: Vec<_> = arm_body
            .bindings()
            .iter()
            .filter(|b| &b.name != result_atom)
            .collect();
        // Check: self_name must not appear in any earlier binding or the step atom itself.
        let stray_ref = bindings_without_tail.iter().any(|b| {
            let in_rhs = match &b.rhs {
                Rhs::Const(_) => false,
                Rhs::Alias(a) => matches!(a, Atom::Named(n) if n == self_name),
                Rhs::Op { args, .. } | Rhs::Construct { args, .. } => args
                    .iter()
                    .any(|a| matches!(a, Atom::Named(n) if n == self_name)),
                Rhs::Swap { src, .. } => matches!(src, Atom::Named(n) if n == self_name),
                Rhs::App { func, arg } => {
                    matches!(func, Atom::Named(n) if n == self_name)
                        || matches!(arg, Atom::Named(n) if n == self_name)
                }
                Rhs::Lam { body: lb, .. } => anf_refs_name(lb, self_name),
                Rhs::Fix { body: fb, .. } => anf_refs_name(fb, self_name),
                Rhs::FixGroup { defs, .. } => defs.iter().any(|(_, d)| anf_refs_name(d, self_name)),
                Rhs::Match {
                    scrutinee,
                    alts,
                    default,
                } => {
                    matches!(scrutinee, Atom::Named(n) if n == self_name)
                        || alts.iter().any(|alt| {
                            let ab = match alt {
                                mycelium_core::lower::AnfAlt::Ctor { body, .. } => body,
                                mycelium_core::lower::AnfAlt::Lit { body, .. } => body,
                            };
                            anf_refs_name(ab, self_name)
                        })
                        || default
                            .as_ref()
                            .is_some_and(|d| anf_refs_name(d, self_name))
                }
            };
            in_rhs || matches!(&b.name, Atom::Named(n) if n == self_name)
        });
        if stray_ref {
            return Err(AotError::UnsupportedNode(format!(
                "non-tail self-reference to `{self_name}` in a tail-Fix arm: the self-name \
                 appears outside the final tail App — only tail self-calls are supported \
                 (Increment-3; RFC-0004 §11.6; G2)"
            )));
        }
        Ok(ArmKind::Tail(step))
    } else {
        // Not a tail call: self_name must not appear anywhere in the arm (base case).
        if anf_refs_name(arm_body, self_name) {
            return Err(AotError::UnsupportedNode(format!(
                "non-tail self-reference to `{self_name}` in a tail-Fix arm: self-name appears \
                 but the arm result is not a direct tail App(self, step) — non-tail recursion is \
                 unsupported (Increment-3; RFC-0004 §11.6; G2)"
            )));
        }
        Ok(ArmKind::Base)
    }
}

/// Lower bindings of `arm_body` up to (but NOT including) the final binding whose name is
/// `result_atom` (the tail App binding). Extends `env` with each intermediate binding.
/// This computes the `step` argument of the tail call without recursing into `lower_app`.
#[allow(clippy::too_many_arguments)]
fn lower_arm_bindings_before_tail(
    arm_body: &lower::Anf,
    env: &mut HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
) -> Result<(), AotError> {
    let result_atom = arm_body.result();
    for b in arm_body.bindings() {
        if &b.name == result_atom {
            // The tail App binding — stop here (caller handles it explicitly).
            break;
        }
        let ev = match &b.rhs {
            Rhs::Const(v) => EnvValue::Repr(const_lane(v)?),
            Rhs::Alias(a) => lookup_ev(env, a)?.clone(),
            Rhs::Op { prim, args } => {
                let operands: Vec<&Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("op operand"))
                    .collect::<Result<_, _>>()?;
                EnvValue::Repr(emit_op(prim, &operands, ssa, body, flags)?)
            }
            // M-852 lowers `Swap` for straight-line / let / non-recursive match-arm positions; a
            // swap inside a tail-Fix *recursion step* stays an explicit refusal (out of scope — the
            // recursion path is not swap-aware). Never a silent mis-lowering (G2).
            Rhs::Swap { target, .. } => {
                return Err(AotError::UnsupportedNode(format!(
                    "swap to {target:?} in a tail-Fix arm body — native swap is lowered outside \
                     recursion steps only (M-852); refused here, never silent (G2)"
                )));
            }
            Rhs::Construct { ctor, args } => {
                let field_lanes: Vec<Lane> = args
                    .iter()
                    .map(|a| lookup_ev(env, a)?.as_lane("Construct field").cloned())
                    .collect::<Result<_, _>>()?;
                let total_elem: usize = field_lanes.iter().map(|l| l.vals.len()).sum();
                let slots = 1 + total_elem;
                let ptr = ssa.fresh();
                let _ = writeln!(body, "  {ptr} = alloca [{slots} x i64], align 8");
                let tag_gep = ssa.fresh();
                let tag_val = ctor.index() as u64;
                let _ = writeln!(
                    body,
                    "  {tag_gep} = getelementptr inbounds [{slots} x i64], [{slots} x i64]* {ptr}, i64 0, i64 0"
                );
                let _ = writeln!(body, "  store i64 {tag_val}, i64* {tag_gep}");
                let mut slot_start = 1usize;
                let mut field_layouts = Vec::with_capacity(field_lanes.len());
                for lane in &field_lanes {
                    let layout = emit_store_field(lane, &ptr, slots, slot_start, ssa, body);
                    slot_start += lane.vals.len();
                    field_layouts.push(layout);
                }
                EnvValue::Datum(Datum {
                    ptr,
                    _tag: tag_val,
                    fields: field_layouts,
                    slots,
                })
            }
            Rhs::Lam {
                param,
                body: lam_body,
            } => lower_lam(param, lam_body, env)?,
            Rhs::App { func, arg } => {
                // Recursion-arm App: the trampoline/tail-loop path; a swap here is refused (the
                // `Rhs::Swap` arm above), so the swap cert mode is the `Recheck` default (M-852).
                lower_app(
                    func,
                    arg,
                    env,
                    ssa,
                    bbc,
                    body,
                    funcs,
                    flags,
                    SwapCertMode::Recheck,
                )?
            }
            Rhs::Fix {
                name,
                body: fix_body,
            } => EnvValue::Fix(FixVal {
                name: name.clone(),
                body: fix_body.clone(),
            }),
            // A `Match` in the pre-tail binding sequence (e.g. computing the next step via a Match)
            // is refused for now. `lower_match` introduces basic blocks, so the loop's back-edge
            // would branch from the Match's merge block rather than the recorded `recur` label — the
            // back-edge `phi` then has stale predecessors (LLVM "PHI node entries do not match
            // predecessors"). Supporting it needs current-block tracking threaded through the
            // back-edge; deferred (DN-15 §8.5). Explicit refusal, never fragile/incorrect IR (G2/VR-5).
            Rhs::Match { .. } => return Err(AotError::UnsupportedNode(
                "a Match in a tail-Fix arm's pre-tail binding sequence (e.g. computing the next \
                     step via a Match) is not yet supported — it introduces basic blocks that \
                     invalidate the loop back-edge phi; deferred (DN-15 §8.5)"
                    .to_owned(),
            )),
            // FixGroup (mutual recursion) stays deferred — explicit refusal, never silent (G2).
            Rhs::FixGroup { .. } => return Err(AotError::UnsupportedNode(
                "FixGroup in a tail-Fix arm binding sequence (mutual recursion — deferred to a \
                     later increment)"
                    .to_owned(),
            )),
        };
        env.insert(b.name.clone(), ev);
    }
    Ok(())
}

/// Emit the iterative LLVM loop for `App(Fix{name, λparam. Match param { <arms> }}, init)`
/// (Increment-3; RFC-0004 §11.6; DN-15 §8; DN-05 #1). Only the canonical shape is supported;
/// out-of-scope cases are refused with explicit `UnsupportedNode` (never silent — G2).
///
/// ## Shape invariant (checked here)
///
/// The Fix body must be exactly `Rhs::Lam{param, lam_body}` where `lam_body` is exactly
/// `Rhs::Match{scrutinee=param, alts, default}`. Each alt/default is tail-classified:
///
/// - **tail arm**: `App(self, step)` is the result — back-edge with the new accumulator.
/// - **base arm**: no self-reference — exit the loop with the base result.
///
/// Non-tail self-references, `FixGroup`, non-Binary{8} types, Ctor arms all return
/// `UnsupportedNode` (G2).
///
/// ## Loop structure (emitted IR)
/// ```text
/// entry:
///   br %header
/// header:   ; loop header — phi nodes accumulate the iteration state
///   %n_packed = phi i64 [%init, %entry], [%next_k, %recur_k]...
///   %depth   = phi i64 [0, %entry],      [%depth1_k, %recur_k]...
///   %over = icmp uge i64 %depth, <ceiling>
///   br i1 %over, label %depthlimit, label %dispatch
/// depthlimit:
///   putchar(DEPTHLIMIT_SENTINEL); putchar(10); ret i32 0
/// dispatch:
///   %n = unpack(%n_packed)
///   switch i64 %n_packed, label %def [ i64 <k>, label %arm_k ... ]
/// arm_k:     ; base: lower arm body; br %exit; or tail: lower step, br %header
/// exit:
///   %res = phi i32 [<base results>...]   (per element)
/// ```
#[allow(clippy::too_many_arguments)]
fn lower_tail_fix(
    fixval: &FixVal,
    init_atom: &Atom,
    env: &HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
) -> Result<EnvValue, AotError> {
    let self_name = &fixval.name;
    let fix_body = &fixval.body;

    // ── Step (a): extract (param, match_node) from the Fix body ───────────────────────────────
    // The Fix body is a single-binding Anf: one `Rhs::Lam{param, lam_body}` binding.
    let (param, lam_body) = {
        let bindings = fix_body.bindings();
        if bindings.len() != 1 {
            return Err(AotError::UnsupportedNode(format!(
                "tail-Fix: Fix body must be exactly `λparam. Match ...` (a single Lam binding); \
                 got {} bindings — nested/complex Fix bodies are outside Increment-3 (G2)",
                bindings.len()
            )));
        }
        match &bindings[0].rhs {
            Rhs::Lam { param, body: lb } => (param.clone(), lb),
            other => {
                return Err(AotError::UnsupportedNode(format!(
                    "tail-Fix: Fix body must be a Lam, got {other:?} — only `Fix{{λ.Match}}` \
                     is supported (Increment-3; RFC-0004 §11.6; G2)"
                )));
            }
        }
    };

    // The Lam body is a single-binding Anf: one `Rhs::Match{scrutinee=param, alts, default}`.
    let (alts, default_anf) = {
        let lam_bindings = lam_body.bindings();
        if lam_bindings.len() != 1 {
            return Err(AotError::UnsupportedNode(format!(
                "tail-Fix: Lam body must be exactly `Match param {{ ... }}` (a single Match \
                 binding); got {} bindings (G2)",
                lam_bindings.len()
            )));
        }
        match &lam_bindings[0].rhs {
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => {
                // Scrutinee must be the param atom.
                if scrutinee != &Atom::Named(param.clone()) {
                    return Err(AotError::UnsupportedNode(format!(
                        "tail-Fix: Match scrutinee must be the Lam param `{param}`; got \
                         {scrutinee:?} (G2)"
                    )));
                }
                (alts.clone(), default.clone())
            }
            other => {
                return Err(AotError::UnsupportedNode(format!(
                    "tail-Fix: Lam body must be a Match, got {other:?} — only `λ.Match` is \
                     supported (Increment-3; G2)"
                )));
            }
        }
    };

    // ── Step (b): tail-classify each arm and the default ──────────────────────────────────────
    // Lit arms only (Ctor arms on the param → UnsupportedNode).
    use mycelium_core::lower::AnfAlt;
    let mut arm_kinds: Vec<(u64, lower::Anf, ArmKind)> = Vec::with_capacity(alts.len());
    for alt in &alts {
        match alt {
            AnfAlt::Lit {
                value,
                body: arm_body,
            } => {
                let packed_key = lit_binary8_packed(value)?;
                let kind = classify_arm(arm_body, self_name)?;
                arm_kinds.push((packed_key, arm_body.clone(), kind));
            }
            AnfAlt::Ctor { .. } => {
                return Err(AotError::UnsupportedNode(
                    "tail-Fix: Ctor arm on the recursion param is not supported — only Lit arms \
                     (Binary{8} literal patterns) are allowed in Increment-3 (G2)"
                        .to_owned(),
                ));
            }
        }
    }
    // Classify the default arm (if present).
    let default_kind: Option<(lower::Anf, ArmKind)> = if let Some(def) = &default_anf {
        let kind = classify_arm(def, self_name)?;
        Some((def.clone(), kind))
    } else {
        None
    };

    // ── Step (c): pack `init` to i64 ──────────────────────────────────────────────────────────
    let init_ev = lookup_ev(env, init_atom)?;
    let init_lane = as_binary8(init_ev, "tail-Fix init argument")?.clone();
    let init_packed = pack_binary8(&init_lane, ssa, body);

    // ── Step (d): emit the loop IR ────────────────────────────────────────────────────────────
    // Resolve the depth ceiling once (via the DepthBudget trait).
    use crate::budget::DepthBudget as _;
    let ceiling = crate::budget::AutoDepthBudget::default()
        .resolve()
        .max_depth;

    // Label pool — all unique (bbc.fresh() is monotone).
    let entry_label = bbc.fresh(); // the pre-header block (already "active" — we emit a br)
    let header_label = bbc.fresh(); // phi / depth-check
    let depthlimit_label = bbc.fresh(); // graceful depth-limit exit
    let dispatch_label = bbc.fresh(); // switch dispatch
    let exit_label = bbc.fresh(); // merge point for base cases

    // Per-arm/default labels.
    let arm_labels: Vec<String> = (0..arm_kinds.len()).map(|_| bbc.fresh()).collect();
    let default_label = bbc.fresh(); // the switch default block

    // Emit the br to the entry pre-header (terminates the current "fall-through" block).
    let _ = writeln!(body, "  br label %{entry_label}");
    let _ = writeln!(body, "{entry_label}:");
    let _ = writeln!(body, "  br label %{header_label}");

    // ── Header: phi nodes ─────────────────────────────────────────────────────────────────────
    // The header phi has one incoming edge from entry + one from each recur (tail-call) arm.
    // We emit placeholder phi strings now and back-patch after classifying arms; instead, we
    // structure the emission so that phi operands are collected first, then emitted.
    //
    // Collect recur arm labels (for the back-edges on the phi).
    let recur_arm_labels: Vec<String> = arm_kinds
        .iter()
        .zip(&arm_labels)
        .filter(|((_, _, kind), _)| matches!(kind, ArmKind::Tail(_)))
        .map(|(_, lbl)| lbl.clone())
        .collect();
    let recur_default_label: Option<String> = default_kind
        .as_ref()
        .and_then(|(_, kind)| matches!(kind, ArmKind::Tail(_)).then(|| default_label.clone()));

    // The packed-n phi: %n_packed = phi i64 [%init, %entry_label], [%next_k, %recur_k]...
    // We emit the phi header now with just the entry edge; back-edges are added after loop body.
    // HOWEVER, LLVM requires complete phi operand lists — so we must use SSA registers that we
    // declare in the recur arm blocks. We do this by pre-assigning SSA names for the "next"
    // registers and the "depth1" registers that will be defined in each recur arm block.
    let n_packed_phi = ssa.fresh();
    let depth_phi = ssa.fresh();

    // Pre-assign "next" packed values + "depth1" values for each recur arm.
    // We know which arms are tail calls; for each we reserve two SSA names now.
    let mut recur_next_regs: Vec<String> = Vec::new();
    let mut recur_depth1_regs: Vec<String> = Vec::new();
    let all_recur_count = recur_arm_labels.len() + recur_default_label.as_ref().map_or(0, |_| 1);
    for _ in 0..all_recur_count {
        recur_next_regs.push(ssa.fresh());
        recur_depth1_regs.push(ssa.fresh());
    }

    let _ = writeln!(body, "{header_label}:");
    // Build the phi for %n_packed.
    {
        let mut phi_args = format!("[ {init_packed}, %{entry_label} ]");
        let mut ri = 0;
        for ((_, _, kind), lbl) in arm_kinds.iter().zip(&arm_labels) {
            if matches!(kind, ArmKind::Tail(_)) {
                phi_args.push_str(&format!(", [ {}, %{lbl} ]", recur_next_regs[ri]));
                ri += 1;
            }
        }
        if let Some((_, kind)) = &default_kind {
            if matches!(kind, ArmKind::Tail(_)) {
                phi_args.push_str(&format!(", [ {}, %{default_label} ]", recur_next_regs[ri]));
            }
        }
        let _ = writeln!(body, "  {n_packed_phi} = phi i64 {phi_args}");
    }
    // Build the phi for %depth.
    {
        let mut phi_args = format!("[ 0, %{entry_label} ]");
        let mut ri = 0;
        for ((_, _, kind), lbl) in arm_kinds.iter().zip(&arm_labels) {
            if matches!(kind, ArmKind::Tail(_)) {
                phi_args.push_str(&format!(", [ {}, %{lbl} ]", recur_depth1_regs[ri]));
                ri += 1;
            }
        }
        if let Some((_, kind)) = &default_kind {
            if matches!(kind, ArmKind::Tail(_)) {
                phi_args.push_str(&format!(
                    ", [ {}, %{default_label} ]",
                    recur_depth1_regs[ri]
                ));
            }
        }
        let _ = writeln!(body, "  {depth_phi} = phi i64 {phi_args}");
    }
    // Depth-limit check.
    let over_reg = ssa.fresh();
    let _ = writeln!(body, "  {over_reg} = icmp uge i64 {depth_phi}, {ceiling}");
    let _ = writeln!(
        body,
        "  br i1 {over_reg}, label %{depthlimit_label}, label %{dispatch_label}"
    );

    // ── DepthLimit block ──────────────────────────────────────────────────────────────────────
    let _ = writeln!(body, "{depthlimit_label}:");
    let dl1 = ssa.fresh();
    let _ = writeln!(
        body,
        "  {dl1} = call i32 @putchar(i32 {})",
        DEPTHLIMIT_SENTINEL
    );
    let dl2 = ssa.fresh();
    let _ = writeln!(body, "  {dl2} = call i32 @putchar(i32 10)");
    let _ = writeln!(body, "  ret i32 0");

    // ── Dispatch block: unpack + switch ───────────────────────────────────────────────────────
    let _ = writeln!(body, "{dispatch_label}:");
    let n_lane = unpack_binary8(&n_packed_phi, ssa, body);
    // Emit switch: default → %default_label; lit_k → %arm_k.
    let _ = write!(
        body,
        "  switch i64 {n_packed_phi}, label %{default_label} ["
    );
    for ((key, _, _), lbl) in arm_kinds.iter().zip(&arm_labels) {
        let _ = write!(body, " i64 {key}, label %{lbl}");
    }
    let _ = writeln!(body, " ]");

    // ── Arm blocks ────────────────────────────────────────────────────────────────────────────
    // Collect (label, lane) pairs for the exit phi (base arms only).
    let mut exit_phi_entries: Vec<(String, Lane)> = Vec::new();
    let mut recur_idx = 0usize;

    for ((_, arm_anf, arm_kind), lbl) in arm_kinds.iter().zip(&arm_labels) {
        let _ = writeln!(body, "{lbl}:");
        // Bind param → n_lane in a child env (clone).
        let mut arm_env = env.clone();
        arm_env.insert(Atom::Named(param.clone()), EnvValue::Repr(n_lane.clone()));

        match arm_kind {
            ArmKind::Base => {
                // Lower the full arm body and collect its result for the exit phi. A base-arm swap
                // uses the `Recheck` default (the recursion path is not mode-threaded; M-852).
                let result_lane = lower_anf_block(
                    arm_anf,
                    &mut arm_env,
                    ssa,
                    bbc,
                    body,
                    funcs,
                    flags,
                    SwapCertMode::Recheck,
                )?;
                exit_phi_entries.push((lbl.clone(), result_lane));
                let _ = writeln!(body, "  br label %{exit_label}");
            }
            ArmKind::Tail(step_atom) => {
                // Lower bindings BEFORE the tail App (the step computation).
                lower_arm_bindings_before_tail(
                    arm_anf,
                    &mut arm_env,
                    ssa,
                    bbc,
                    body,
                    funcs,
                    flags,
                )?;
                // Look up the step atom from the arm env.
                let step_ev = lookup_ev(&arm_env, step_atom)?;
                let step_lane = as_binary8(step_ev, "tail-Fix step argument")?.clone();
                // Emit the "next" packed value and "depth+1" using the pre-reserved SSA regs.
                let next_packed = pack_binary8(&step_lane, ssa, body);
                // Copy packed value into the pre-reserved register via an `or i64 <val>, 0`.
                let _ = writeln!(
                    body,
                    "  {} = or i64 {next_packed}, 0",
                    recur_next_regs[recur_idx]
                );
                let depth1_raw = ssa.fresh();
                let _ = writeln!(body, "  {depth1_raw} = add i64 {depth_phi}, 1");
                let _ = writeln!(
                    body,
                    "  {} = or i64 {depth1_raw}, 0",
                    recur_depth1_regs[recur_idx]
                );
                recur_idx += 1;
                let _ = writeln!(body, "  br label %{header_label}");
            }
        }
    }

    // ── Default block ─────────────────────────────────────────────────────────────────────────
    let _ = writeln!(body, "{default_label}:");
    {
        let mut def_env = env.clone();
        def_env.insert(Atom::Named(param.clone()), EnvValue::Repr(n_lane.clone()));
        match &default_kind {
            Some((def_anf, ArmKind::Base)) => {
                let result_lane = lower_anf_block(
                    def_anf,
                    &mut def_env,
                    ssa,
                    bbc,
                    body,
                    funcs,
                    flags,
                    SwapCertMode::Recheck,
                )?;
                exit_phi_entries.push((default_label.clone(), result_lane));
                let _ = writeln!(body, "  br label %{exit_label}");
            }
            Some((def_anf, ArmKind::Tail(step_atom))) => {
                lower_arm_bindings_before_tail(
                    def_anf,
                    &mut def_env,
                    ssa,
                    bbc,
                    body,
                    funcs,
                    flags,
                )?;
                let step_ev = lookup_ev(&def_env, step_atom)?;
                let step_lane = as_binary8(step_ev, "tail-Fix default step argument")?.clone();
                let next_packed = pack_binary8(&step_lane, ssa, body);
                let _ = writeln!(
                    body,
                    "  {} = or i64 {next_packed}, 0",
                    recur_next_regs[recur_idx]
                );
                let depth1_raw = ssa.fresh();
                let _ = writeln!(body, "  {depth1_raw} = add i64 {depth_phi}, 1");
                let _ = writeln!(
                    body,
                    "  {} = or i64 {depth1_raw}, 0",
                    recur_depth1_regs[recur_idx]
                );
                let _ = writeln!(body, "  br label %{header_label}");
            }
            None => {
                // No default arm: abort() — defined-trap (never raw unreachable UB; G2/WF7).
                let _ = writeln!(body, "  call void @abort()");
                let _ = writeln!(body, "  ret i32 0");
            }
        }
    }

    // ── Exit block: phi over base-case results ────────────────────────────────────────────────
    let _ = writeln!(body, "{exit_label}:");
    if exit_phi_entries.is_empty() {
        // All arms are tail calls — no base case. The loop is diverging; it will always hit the
        // depth-limit ceiling (emitted above in the header) before the exit block is reached.
        // Emit `call void @abort()` + `ret i32 0` as the defined-trap dead-block terminator —
        // never raw `unreachable` UB (G2/DN-05 #1/SC-3). The block is provably dead because the
        // depthlimit path exits before any base-case branch could reach this label.
        let _ = writeln!(body, "  call void @abort()");
        let _ = writeln!(body, "  ret i32 0");
        // Return a dummy zero-width binary lane; the output section downstream is also dead
        // (execution exits via depthlimit first). The dummy lane propagates cleanly through
        // `result_shape` → `CompiledArtifact::run` → sentinel check (returns `DepthLimit`
        // before `decode_result` is invoked, so `width == 0` is never observed by the decoder).
        return Ok(EnvValue::Repr(Lane {
            kind: LaneKind::Binary,
            vals: vec![],
        }));
    }
    // All base arms must agree on kind/width.
    let first = &exit_phi_entries[0].1;
    let kind = first.kind;
    let width = first.vals.len();
    for (_, lane) in &exit_phi_entries[1..] {
        if lane.kind != kind || lane.vals.len() != width {
            return Err(AotError::UnsupportedNode(
                "tail-Fix: base-case arms return lanes of different kind or width — all base \
                 cases must agree on repr shape (G2)"
                    .to_owned(),
            ));
        }
    }
    // Emit phi per element.
    let mut result_vals: Vec<Operand> = Vec::with_capacity(width);
    for elem_idx in 0..width {
        let phi_reg = ssa.fresh();
        let phi_operands: Vec<String> = exit_phi_entries
            .iter()
            .map(|(lbl, lane)| format!("[ {}, %{lbl} ]", lane.vals[elem_idx]))
            .collect();
        let _ = writeln!(body, "  {phi_reg} = phi i32 {}", phi_operands.join(", "));
        result_vals.push(phi_reg);
    }
    Ok(EnvValue::Repr(Lane {
        kind,
        vals: result_vals,
    }))
}

/// Emit textual LLVM IR for the bit/trit + non-recursive-data program `node` — a `main` that
/// computes the result elements and writes them as a line to stdout (Binary: `'0'`/`'1'`;
/// Ternary: `'-'`/`'0'`/`'+'`). Deterministic. One op per output element (no opaque pass —
/// RFC-0004 §6). Returns an explicit [`AotError`] for anything outside the subset.
pub fn emit_llvm_ir(node: &Node) -> Result<String, AotError> {
    // Default native swap cert mode: `Recheck` (compile-time independent re-check; M-852).
    emit_llvm_ir_with_swap_mode(node, SwapCertMode::Recheck)
}

/// Emit textual LLVM IR under an **explicit** native swap cert mode (M-852): `Recheck` (DEFAULT —
/// compile-time independent re-check of the bijection certificate) or `ReuseInterp` (OPT-IN — carry
/// the interpreter's certificate forward). The mode is recorded in the emitted IR (a dumpable swap
/// comment) and selects whether the bijection side-condition is independently re-checked at compile
/// time. For a swap-free program the two modes emit byte-identical IR. [`emit_llvm_ir`] delegates
/// here with `Recheck`.
pub fn emit_llvm_ir_with_swap_mode(
    node: &Node,
    swap_mode: SwapCertMode,
) -> Result<String, AotError> {
    let Lowered {
        body,
        result,
        mut ssa,
        overflow,
        // Vestigial since M-851: the narrow ABI emitted one top-level `@myc_closureN` function per
        // `Rhs::Lam` here; the widened ABI **inlines** closures at their application site
        // ([`lower_app`]), so no top-level closure functions (and no bump arena) are produced. The
        // field is retained (always empty) only because the trampoline-shared lowering signatures
        // thread it (M-850); it carries no closure functions now.
        funcs: _funcs,
    } = lower_program_with_swap_mode(node, swap_mode)?;
    // M-850 (Wave-B): the heap-trampoline lowering emits `@myc_tramp_alloc` calls into `body`; a
    // program with no non-tail-Fix/FixGroup recursion never does, so it emits the same module as
    // before (no trampoline runtime, no extra declares). Detecting via the emitted body keeps the
    // runtime opt-in without threading another flag through every lowering signature (DRY).
    let uses_trampoline = body.contains("@myc_tramp_alloc");
    let mut out =
        String::from("; mycelium direct-LLVM AOT (bit/trit + non-recursive data; M-301; M-373)\n");
    if uses_trampoline {
        out.push_str("; recursion: heap-trampoline control stack (M-850; DN-15 §10)\n");
    }
    // M-851: closures lower by **inlining** at the application site — no top-level closure functions,
    // no bump arena, no closure heap. A closure-only program now emits the same straight-line module
    // as the bit/trit subset (its closures are inlined into `@main`'s body). The only heap is the
    // trampoline frame stack (recursion).
    //
    // `@putchar` for the read-back protocol; `@abort` for the defined-traps (the match no-default
    // trap and the trampoline OOM). `@abort` is declared `noreturn` so LLVM treats every
    // `call @abort` as non-returning: the dead `ret` that follows each trap is provably never taken
    // (G2), and no post-trap path is ever reachable.
    out.push_str("declare i32 @putchar(i32)\n");
    out.push_str("declare void @abort() noreturn\n");
    if uses_trampoline {
        // The trampoline frame stack needs `@malloc`/`@free` + the alloc/free seams. Never silent OOM
        // (defined-trap).
        out.push_str("declare i8* @malloc(i64)\n");
        out.push_str("declare void @free(i8*)\n");
        out.push_str(&crate::trampoline::trampoline_runtime());
    }
    out.push('\n');
    out.push_str("define i32 @main() {\nentry:\n");
    out.push_str(&body);
    match overflow {
        // No trit arithmetic ⇒ no overflow path; emit the result line straight-line (unchanged IR).
        None => {
            emit_result_line(result.kind, &result.vals, &mut ssa, &mut out);
        }
        // Overflow possible ⇒ branch on the runtime flag: print the sentinel line on overflow, the
        // result line otherwise (the read-back protocol — never a silent wrap, G2).
        Some(ovf) => {
            let _ = writeln!(&mut out, "  br i1 {ovf}, label %ovf, label %ok");
            out.push_str("ovf:\n");
            let s = ssa.fresh();
            let _ = writeln!(
                &mut out,
                "  {s} = call i32 @putchar(i32 {})",
                OVERFLOW_SENTINEL
            );
            let snl = ssa.fresh();
            let _ = writeln!(&mut out, "  {snl} = call i32 @putchar(i32 10)");
            out.push_str("  ret i32 0\nok:\n");
            emit_result_line(result.kind, &result.vals, &mut ssa, &mut out);
        }
    }
    out.push_str("}\n");
    Ok(out)
}

// ─── parallel per-function/per-nodule codegen (M-860) ─────────────────────────────────────────

/// Lower `nodes` — a batch of **independent** functions/nodules — in parallel across the native
/// work-stealing [`Scheduler`](mycelium_sched::scheduler::Scheduler) (M-861), returning one
/// [`AotError`]-or-IR result per input, in the **same order as `nodes`**.
///
/// **Determinism (Exact by construction).** Each call to [`emit_llvm_ir_with_swap_mode`] is a pure
/// function of its own `Node` — a fresh [`Ssa`]/[`Bbc`] counter pair starts at zero every time, and
/// no lowering shares mutable state across nodes — so parallelizing the *computation* cannot change
/// any individual node's emitted text. Two further guards make the *batch* result reproducible, not
/// just "happens to work":
/// - the batch is sorted by each node's [`mycelium_core::Node::content_hash`] (RFC-0001 §4.6 — a
///   stable, α-normalized structural key, paired with the original index as a tie-break for two
///   structurally-identical nodes) *before* being submitted, so **which job the scheduler dispatches
///   in which slot** is a pure function of the batch's content, never of wall-clock/thread-arrival
///   order;
/// - [`Scheduler::run_indexed`](mycelium_sched::scheduler::Scheduler::run_indexed) returns its
///   outputs in **spawn order** (its RT2 differential contract — completion order and worker identity
///   are unobservable through that API), and every result is then scattered back to its **original**
///   `nodes` index.
///
/// So `emit_llvm_ir_many(nodes) == nodes.iter().map(emit_llvm_ir)` element-wise, byte-for-byte,
/// regardless of worker count, steal schedule, or scheduling — asserted by
/// [`tests::llvm::parallel_emit_matches_sequential_emit_byte_identical`]. The work-stealing pool is
/// the same one M-861 landed for the runtime scheduler (zero new dependency — `mycelium-mlir`
/// already depends on `mycelium-sched` directly, M-865).
///
/// Uses the default swap-cert mode ([`SwapCertMode::Recheck`]); see
/// [`emit_llvm_ir_many_with_swap_mode`] to select [`SwapCertMode::ReuseInterp`] for the whole batch.
#[must_use]
pub fn emit_llvm_ir_many(nodes: &[Node]) -> Vec<Result<String, AotError>> {
    emit_llvm_ir_many_with_swap_mode(nodes, SwapCertMode::Recheck)
}

/// [`emit_llvm_ir_many`] under an explicit, whole-batch [`SwapCertMode`] (M-852). See
/// [`emit_llvm_ir_many`]'s doc for the determinism contract (content-key-sorted work distribution,
/// spawn-order scheduler results, original-index reassembly).
#[must_use]
pub fn emit_llvm_ir_many_with_swap_mode(
    nodes: &[Node],
    swap_mode: SwapCertMode,
) -> Vec<Result<String, AotError>> {
    use mycelium_sched::scheduler::Scheduler;

    if nodes.is_empty() {
        return Vec::new();
    }

    // Stable work order: sort indices by (content_hash, original_index) — never a `HashMap`/set
    // iteration order, never wall-clock/thread-arrival order (G2 — no incidental nondeterminism).
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    let keys: Vec<mycelium_core::ContentHash> = nodes.iter().map(Node::content_hash).collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));

    // One pure job per node, submitted in content-key order; each job carries its ORIGINAL index so
    // the scatter below can restore `nodes`' own order. `Scheduler::run_indexed` returns outputs in
    // spawn order (RT2), so `computed` is in content-key order — the scatter, not the return order,
    // is what pins the output to the original index.
    //
    // M-864: `run_indexed` now requires `'static` job closures (the persistent pool's worker threads
    // outlive this call), so each job clones its own `Node` up front rather than borrowing
    // `&nodes[i]` — the content-key sort above already ran over the ORIGINAL `nodes`, so cloning
    // afterward changes nothing about which node ends up in which slot, only how it's captured.
    let jobs: Vec<_> = order
        .iter()
        .map(|&i| {
            let node = nodes[i].clone();
            move || (i, emit_llvm_ir_with_swap_mode(&node, swap_mode))
        })
        .collect();
    let computed: Vec<(usize, Result<String, AotError>)> =
        Scheduler::new().run_indexed(jobs, None, None);

    // Scatter back to original position — the output vector's order never depends on completion
    // order or the content-key sort, only on `nodes`' own index (Exact; never-silent: every slot is
    // populated exactly once, or this would panic rather than silently return a short/reordered Vec).
    let mut out: Vec<Option<Result<String, AotError>>> = (0..nodes.len()).map(|_| None).collect();
    for (i, r) in computed {
        out[i] = Some(r);
    }
    out.into_iter()
        .map(|slot| slot.expect("every index populated exactly once by the scatter above"))
        .collect()
}

/// Emit each result element as its ASCII char via `@putchar` (one op per element — a transparent
/// rendering of the computed lane, no opaque pass, RFC-0004 §6), then a trailing newline and `ret`.
fn emit_result_line(kind: LaneKind, vals: &[Operand], ssa: &mut Ssa, out: &mut String) {
    for v in vals {
        let c = emit_char_code(kind, v, ssa, out);
        let p = ssa.fresh();
        let _ = writeln!(out, "  {p} = call i32 @putchar(i32 {c})");
    }
    let nl = ssa.fresh();
    let _ = writeln!(out, "  {nl} = call i32 @putchar(i32 10)");
    out.push_str("  ret i32 0\n");
}

/// The result shape (lane kind + element count) of the program — **derived from the actual
/// lowering** ([`lower_program_with_swap_mode`]) so it can never disagree with what
/// [`emit_llvm_ir_with_swap_mode`] emits. Used by [`compile`] to know how to parse the native
/// output. Threaded with the swap mode so a `Recheck`-refused illegal-pair swap is surfaced
/// identically here and in the emitter (no shape/emit disagreement; M-852).
fn result_shape(node: &Node, swap_mode: SwapCertMode) -> Result<(LaneKind, usize), AotError> {
    let l = lower_program_with_swap_mode(node, swap_mode)?;
    Ok((l.result.kind, l.result.vals.len()))
}

fn lookup_ev<'a>(env: &'a HashMap<Atom, EnvValue>, a: &Atom) -> Result<&'a EnvValue, AotError> {
    env.get(a).ok_or_else(|| AotError::FreeVariable(a.render()))
}

/// The const's elements as `i32` literal operands + its lane kind, or an explicit refusal for an
/// unsupported repr (Dense/VSA).
fn const_lane(v: &Value) -> Result<Lane, AotError> {
    match (v.repr(), v.payload()) {
        (Repr::Binary { .. }, Payload::Bits(b)) => Ok(Lane {
            kind: LaneKind::Binary,
            vals: b
                .iter()
                .map(|&x| if x { "1" } else { "0" }.to_owned())
                .collect(),
        }),
        (Repr::Ternary { .. }, Payload::Trits(t)) => Ok(Lane {
            kind: LaneKind::Ternary,
            vals: t
                .iter()
                .map(|&x| {
                    match x {
                        Trit::Neg => "-1",
                        Trit::Zero => "0",
                        Trit::Pos => "1",
                    }
                    .to_owned()
                })
                .collect(),
        }),
        // M-853 (RFC-0039 §5.1): `Repr::Dense` now has a **native home** — the dedicated
        // `dense_codegen` direct-LLVM path (`dense_compile_and_run`/`emit_dense_llvm_ir`), which lowers
        // the un-quantized F32/BF16 element-wise surface against the `mycelium-dense` reference. Dense
        // is **not** lowered through this generic bit/trit `Node` const-lane (the lane model is i32
        // bit/trit elements; Dense is `f64`), so a Dense `Const` reaching here is routed to that native
        // path, never silently mis-lowered as a bit/trit lane. The refusal is informative (ADR-006/G2):
        // it names where Dense *is* lowered. (Quantized Dense + VSA stay refused per RFC-0039 §5.1/§5.2.)
        (repr @ Repr::Dense { .. }, _) => Err(AotError::UnsupportedRepr(format!(
            "{repr:?}: Dense is lowered by the dedicated dense_codegen direct-LLVM path \
             (dense_compile_and_run; M-853/RFC-0039 §5.1), not this generic bit/trit Node lowering"
        ))),
        (repr, _) => Err(AotError::UnsupportedRepr(format!("{repr:?}"))),
    }
}

/// Require a lane to be of the expected kind, else an explicit refusal (a `bit.*` op on a ternary
/// lane, or `trit.*` on a binary one, is a type error — never silently mis-lowered).
fn require_kind(prim: &str, got: LaneKind, want: LaneKind) -> Result<(), AotError> {
    if got == want {
        Ok(())
    } else {
        Err(AotError::UnsupportedPrim(format!(
            "{prim} expects a {want:?} operand, got {got:?}"
        )))
    }
}

/// Emit the LLVM IR for one bit/trit-subset op, returning the result lane. Trit-arithmetic ops also
/// push their runtime overflow `i1` register(s) onto `flags` (the caller folds them into the
/// program-level overflow flag that drives the read-back protocol).
fn emit_op(
    prim: &str,
    operands: &[&Lane],
    ssa: &mut Ssa,
    body: &mut String,
    flags: &mut Vec<String>,
) -> Result<Lane, AotError> {
    match prim {
        // Identity passes the lane through unchanged, any kind (M-I1 passthrough).
        "core.id" => {
            let [a] = arity1(prim, operands)?;
            Ok((*a).clone())
        }
        "bit.not" => {
            let [a] = arity1(prim, operands)?;
            require_kind(prim, a.kind, LaneKind::Binary)?;
            Ok(map1(a, ssa, body, |x, r| format!("  {r} = xor i32 {x}, 1")))
        }
        "bit.and" | "bit.or" | "bit.xor" => {
            let (a, b) = arity2(prim, operands)?;
            require_kind(prim, a.kind, LaneKind::Binary)?;
            require_kind(prim, b.kind, LaneKind::Binary)?;
            let instr = match prim {
                "bit.and" => "and",
                "bit.or" => "or",
                _ => "xor",
            };
            map2(prim, a, b, ssa, body, |x, y, r| {
                format!("  {r} = {instr} i32 {x}, {y}")
            })
        }
        // Balanced-ternary negation is digit-wise (`-t`), exact, no carry — `0 - x` per trit.
        "trit.neg" => {
            let [a] = arity1(prim, operands)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            Ok(map1(a, ssa, body, |x, r| format!("  {r} = sub i32 0, {x}")))
        }
        // Balanced-ternary addition: a fixed-width ripple-carry over the trits (LSB→MSB), with a
        // runtime overflow flag (non-zero final carry ⇒ out of m-trit range). Mirrors
        // `mycelium_core::ternary::add` digit-for-digit.
        "trit.add" => {
            let (a, b) = arity2(prim, operands)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            require_kind(prim, b.kind, LaneKind::Ternary)?;
            require_width(prim, a, b)?;
            let (lane, ovf) = emit_trit_add(&a.vals, &b.vals, ssa, body);
            flags.push(ovf);
            Ok(lane)
        }
        // Subtraction `a − b` = `add(a, neg(b))`: negate `b`'s trits, then the same ripple adder.
        "trit.sub" => {
            let (a, b) = arity2(prim, operands)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            require_kind(prim, b.kind, LaneKind::Ternary)?;
            require_width(prim, a, b)?;
            let neg_b = map1(b, ssa, body, |x, r| format!("  {r} = sub i32 0, {x}"));
            let (lane, ovf) = emit_trit_add(&a.vals, &neg_b.vals, ssa, body);
            flags.push(ovf);
            Ok(lane)
        }
        // Multiplication: shifted accumulation in a 2m-trit buffer (mirrors
        // `mycelium_core::ternary::mul`), then overflow iff any high trit is non-zero. Each `b` digit
        // scales `a` by an `i32 mul` (the digit is ±1/0, so this is exactly ±a / 0 per position).
        "trit.mul" => {
            let (a, b) = arity2(prim, operands)?;
            require_kind(prim, a.kind, LaneKind::Ternary)?;
            require_kind(prim, b.kind, LaneKind::Ternary)?;
            require_width(prim, a, b)?;
            let (lane, ovfs) = emit_trit_mul(&a.vals, &b.vals, ssa, body);
            flags.extend(ovfs);
            Ok(lane)
        }
        other => Err(AotError::UnsupportedPrim(other.to_owned())),
    }
}

/// Require two lanes to have equal element count, else an explicit [`AotError::WidthMismatch`].
fn require_width(prim: &str, a: &Lane, b: &Lane) -> Result<(), AotError> {
    if a.vals.len() == b.vals.len() {
        Ok(())
    } else {
        Err(AotError::WidthMismatch {
            prim: prim.to_owned(),
            a: a.vals.len(),
            b: b.vals.len(),
        })
    }
}

/// Emit a fixed-width balanced-ternary ripple-carry add over MSB-first trit operands `a`/`b` (equal
/// length, caller-checked). Returns the sum lane (MSB-first) and an `i1` register that is set iff the
/// final carry is non-zero (overflow). Each digit follows `mycelium_core::ternary::add`: with
/// `x = aᵢ + bᵢ + carry + 4` (always ≥ 1 so `srem`/`sdiv` are euclidean), the balanced digit is
/// `x srem 3 − 1` and the next carry is `x sdiv 3 − 1`.
fn emit_trit_add(a: &[Operand], b: &[Operand], ssa: &mut Ssa, body: &mut String) -> (Lane, String) {
    let m = a.len();
    let mut carry = "0".to_owned();
    let mut sum_lsb: Vec<Operand> = Vec::with_capacity(m);
    // Process least-significant first (the tail of the MSB-first strings).
    for i in (0..m).rev() {
        let (digit, next_carry) = emit_trit_add_step(&a[i], &b[i], &carry, ssa, body);
        sum_lsb.push(digit);
        carry = next_carry;
    }
    // Overflow iff the final carry out of the most-significant trit is non-zero.
    let ovf = ssa.fresh();
    let _ = writeln!(body, "  {ovf} = icmp ne i32 {carry}, 0");
    let vals: Vec<Operand> = sum_lsb.into_iter().rev().collect(); // back to MSB-first
    (
        Lane {
            kind: LaneKind::Ternary,
            vals,
        },
        ovf,
    )
}

/// One balanced-ternary add step: given operand trits `a`/`b` and the incoming `carry` (all `i32` in
/// `{−1,0,1}`), emit the digit + outgoing carry. Returns `(digit_reg, carry_reg)`.
fn emit_trit_add_step(
    a: &str,
    b: &str,
    carry: &str,
    ssa: &mut Ssa,
    body: &mut String,
) -> (String, String) {
    let s1 = ssa.fresh();
    let _ = writeln!(body, "  {s1} = add i32 {a}, {b}");
    let s2 = ssa.fresh();
    let _ = writeln!(body, "  {s2} = add i32 {s1}, {carry}");
    // x = s + 4 ∈ [1,7], strictly positive ⇒ srem/sdiv coincide with euclidean rem/div by 3.
    let x = ssa.fresh();
    let _ = writeln!(body, "  {x} = add i32 {s2}, 4");
    let rem = ssa.fresh();
    let _ = writeln!(body, "  {rem} = srem i32 {x}, 3");
    let digit = ssa.fresh();
    let _ = writeln!(body, "  {digit} = sub i32 {rem}, 1");
    let q = ssa.fresh();
    let _ = writeln!(body, "  {q} = sdiv i32 {x}, 3");
    let next_carry = ssa.fresh();
    let _ = writeln!(body, "  {next_carry} = sub i32 {q}, 1");
    (digit, next_carry)
}

/// Emit fixed-width balanced-ternary multiplication over MSB-first trit operands `a`/`b` (equal
/// length, caller-checked). Mirrors `mycelium_core::ternary::mul`: shifted accumulation of `±a` into
/// a 2m-trit buffer, returning the low `m` trits (MSB-first) and the overflow `i1` flags — one per
/// non-zero high trit, plus each accumulation's carry (provably zero, OR-ed in as an honest net).
fn emit_trit_mul(
    a: &[Operand],
    b: &[Operand],
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
    // LSB-first views of the operands and a 2m-wide accumulator initialised to zero.
    let a_lsb: Vec<&Operand> = a.iter().rev().collect();
    let b_lsb: Vec<&Operand> = b.iter().rev().collect();
    let mut acc: Vec<Operand> = vec!["0".to_owned(); wide];
    let mut flags: Vec<String> = Vec::new();

    for (k, &bk) in b_lsb.iter().enumerate() {
        // Partial = (a scaled by digit bk) shifted left by k, in a 2m-wide LSB-first buffer. The
        // digit is ±1/0, so `aⱼ * bk` is exactly ±aⱼ / 0 — the per-digit factor, no carry yet.
        let mut partial: Vec<Operand> = vec!["0".to_owned(); wide];
        for (j, &aj) in a_lsb.iter().enumerate() {
            let p = ssa.fresh();
            let _ = writeln!(body, "  {p} = mul i32 {aj}, {bk}");
            partial[k + j] = p;
        }
        let (next_acc, carry) = emit_ripple_add_lsb(&acc, &partial, ssa, body);
        acc = next_acc;
        // The 2m-wide sum cannot truly overflow for m-trit operands; OR the carry in anyway so a
        // codegen slip can never pass silently (honest net, never a fabricated guarantee).
        let c = ssa.fresh();
        let _ = writeln!(body, "  {c} = icmp ne i32 {carry}, 0");
        flags.push(c);
    }
    // The product fits in m trits iff the high half (positions [m, 2m)) is all zero.
    for hi in &acc[m..] {
        let f = ssa.fresh();
        let _ = writeln!(body, "  {f} = icmp ne i32 {hi}, 0");
        flags.push(f);
    }
    let vals: Vec<Operand> = acc[..m].iter().rev().cloned().collect(); // low m trits, MSB-first
    (
        Lane {
            kind: LaneKind::Ternary,
            vals,
        },
        flags,
    )
}

/// Ripple-carry add over two equal-length **LSB-first** trit-operand vectors. Returns the sum
/// (LSB-first) and the final carry register. The shared inner adder for [`emit_trit_mul`].
fn emit_ripple_add_lsb(
    a: &[Operand],
    b: &[Operand],
    ssa: &mut Ssa,
    body: &mut String,
) -> (Vec<Operand>, String) {
    let mut carry = "0".to_owned();
    let mut sum: Vec<Operand> = Vec::with_capacity(a.len());
    for (ai, bi) in a.iter().zip(b) {
        let (digit, next_carry) = emit_trit_add_step(ai, bi, &carry, ssa, body);
        sum.push(digit);
        carry = next_carry;
    }
    (sum, carry)
}

/// Fold a list of `i1` overflow flags into one (`or i1` chain), or `None` if empty. Deterministic.
fn fold_or(flags: &[String], ssa: &mut Ssa, body: &mut String) -> Option<String> {
    let mut it = flags.iter();
    let mut acc = it.next()?.clone();
    for f in it {
        let r = ssa.fresh();
        let _ = writeln!(body, "  {r} = or i1 {acc}, {f}");
        acc = r;
    }
    Some(acc)
}

/// Emit one IR instruction per element of `a`, returning the result lane (same kind as `a`).
fn map1(a: &Lane, ssa: &mut Ssa, body: &mut String, f: impl Fn(&str, &str) -> String) -> Lane {
    let vals = a
        .vals
        .iter()
        .map(|x| {
            let r = ssa.fresh();
            let _ = writeln!(body, "{}", f(x, &r));
            r
        })
        .collect();
    Lane { kind: a.kind, vals }
}

/// Emit one IR instruction per element pair of `a`/`b` (widths must match), returning the result
/// lane (same kind as `a`).
fn map2(
    prim: &str,
    a: &Lane,
    b: &Lane,
    ssa: &mut Ssa,
    body: &mut String,
    f: impl Fn(&str, &str, &str) -> String,
) -> Result<Lane, AotError> {
    if a.vals.len() != b.vals.len() {
        return Err(AotError::WidthMismatch {
            prim: prim.to_owned(),
            a: a.vals.len(),
            b: b.vals.len(),
        });
    }
    let vals = a
        .vals
        .iter()
        .zip(&b.vals)
        .map(|(x, y)| {
            let r = ssa.fresh();
            let _ = writeln!(body, "{}", f(x, y, &r));
            r
        })
        .collect();
    Ok(Lane { kind: a.kind, vals })
}

fn arity1<'a>(prim: &str, ops: &[&'a Lane]) -> Result<[&'a Lane; 1], AotError> {
    match ops {
        [a] => Ok([a]),
        _ => Err(AotError::UnsupportedPrim(format!(
            "{prim} expects 1 operand, got {}",
            ops.len()
        ))),
    }
}

fn arity2<'a>(prim: &str, ops: &[&'a Lane]) -> Result<(&'a Lane, &'a Lane), AotError> {
    match ops {
        [a, b] => Ok((a, b)),
        _ => Err(AotError::UnsupportedPrim(format!(
            "{prim} expects 2 operands, got {}",
            ops.len()
        ))),
    }
}

/// A compiled native artifact for a bit/trit-subset program: the executable on disk (in a
/// per-artifact temp dir, cleaned up on drop) plus the result shape (lane kind + element count)
/// needed to parse its output. Produced by [`compile`]; run any number of times with
/// [`CompiledArtifact::run`]. The **compile-once / run-many** split is the natural AOT shape and lets
/// a harness time the one-time AOT cost separately from warm per-invocation cost (the E1 perf
/// measurement, M-303).
pub struct CompiledArtifact {
    _dir: TmpDir,
    bin: std::path::PathBuf,
    kind: LaneKind,
    width: usize,
}

impl CompiledArtifact {
    /// Execute the compiled artifact and read its result back as an `Exact` `Binary{w}`/`Ternary{m}`
    /// [`Value`] (bit/`neg` ops are exact; the subset refuses approximate inputs).
    pub fn run(&self) -> Result<Value, AotError> {
        let output = Command::new(&self.bin)
            .output()
            .map_err(|e| AotError::Run(format!("exec {}: {e}", self.bin.display())))?;
        if !output.status.success() {
            return Err(AotError::Run(format!("artifact exited {}", output.status)));
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|e| AotError::Parse(format!("non-utf8 output: {e}")))?;
        let line = stdout.lines().next().unwrap_or("");
        // Read-back protocol: the sentinel line means the native arithmetic overflowed the m-trit
        // range — an explicit error, never a silently-wrapped result (matches `EvalError::Overflow`).
        if line.as_bytes() == [OVERFLOW_SENTINEL] {
            return Err(AotError::Overflow(format!(
                "fixed-width result out of {}-trit range",
                self.width
            )));
        }
        // DepthLimit sentinel: the tail-recursive Fix loop hit the AutoDepthBudget ceiling (DN-05 #1).
        // Graceful explicit refusal — never SIGSEGV, never hang. Mirrors the AOT env-machine's
        // `EvalError::DepthLimit`; the reference interpreter refuses divergence via `FuelExhausted`
        // (it does not raise DepthLimit). Both refuse non-silently — that is the differential parity.
        if line.as_bytes() == [DEPTHLIMIT_SENTINEL] {
            return Err(AotError::DepthLimit(
                "tail-recursive Fix loop exceeded the AutoDepthBudget ceiling".to_owned(),
            ));
        }
        decode_result(self.kind, self.width, line.chars())
    }
}

/// Compile the bit/trit-subset program to a native executable (emit LLVM IR → `llc` → `clang`)
/// without running it. Returns [`AotError::ToolchainMissing`] when `llc`/`clang` are absent so
/// callers can skip; any out-of-subset construct is the same explicit refusal as [`emit_llvm_ir`].
pub fn compile(node: &Node) -> Result<CompiledArtifact, AotError> {
    // Default native swap cert mode: `Recheck` (compile-time independent re-check; M-852).
    compile_with_swap_mode(node, SwapCertMode::Recheck)
}

/// Compile under an **explicit** native swap cert mode (M-852): `Recheck` (DEFAULT — compile-time
/// independent re-check of the bijection certificate) or `ReuseInterp` (OPT-IN — carry the
/// interpreter's certificate forward). The IR and the read-back shape are both threaded with the
/// mode so they never disagree. [`compile`] delegates here with `Recheck`.
pub fn compile_with_swap_mode(
    node: &Node,
    swap_mode: SwapCertMode,
) -> Result<CompiledArtifact, AotError> {
    let ir = emit_llvm_ir_with_swap_mode(node, swap_mode)?;
    let (kind, width) = result_shape(node, swap_mode)?;
    ensure_toolchain()?;

    let dir = unique_tmp_dir()?;
    let ll = dir.join("kernel.ll");
    let obj = dir.join("kernel.o");
    let bin = dir.join("kernel");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    // `-relocation-model=pic`: the trampoline (M-850) and closure-arena modules carry global/`.rodata`
    // references whose default (static) relocations are `R_X86_64_32S`, which a PIE link rejects
    // (`can not be used when making a PIE object`). Modern clang links PIE by default, so we emit a
    // PIC object that is link-compatible with PIE. The byte-for-byte element-wise modules are
    // unaffected by the relocation model (no global refs), so this is a strict superset — never a
    // silent behavioural change (G2); it only lets the heavier recursion/closure modules link.
    run_tool(
        "llc",
        &[
            "-relocation-model=pic",
            "-filetype=obj",
            path(&ll)?,
            "-o",
            path(&obj)?,
        ],
    )?;
    run_tool("clang", &[path(&obj)?, "-o", path(&bin)?])?;

    Ok(CompiledArtifact {
        _dir: guard,
        bin,
        kind,
        width,
    })
}

/// Compile the bit/trit-subset program to a native executable, run it once, and read the result
/// back. The convenience wrapper over [`compile`] + [`CompiledArtifact::run`]; this is the
/// **compiled** execution path the M-302 differential checks against the interpreter.
pub fn compile_and_run(node: &Node) -> Result<Value, AotError> {
    compile(node)?.run()
}

/// Compile + run under an **explicit** native swap cert mode (M-852): `Recheck` (DEFAULT) or
/// `ReuseInterp` (OPT-IN). The compiled-path entry the swap differential exercises in **both** cert
/// modes. [`compile_and_run`] delegates here with `Recheck`.
pub fn compile_and_run_with_swap_mode(
    node: &Node,
    swap_mode: SwapCertMode,
) -> Result<Value, AotError> {
    compile_with_swap_mode(node, swap_mode)?.run()
}

fn ensure_toolchain() -> Result<(), AotError> {
    for tool in ["llc", "clang"] {
        Command::new(tool)
            .arg("--version")
            .output()
            .map_err(|_| AotError::ToolchainMissing(tool.to_owned()))?;
    }
    Ok(())
}

pub(crate) fn run_tool(tool: &str, args: &[&str]) -> Result<(), AotError> {
    let out = Command::new(tool)
        .args(args)
        .output()
        .map_err(|_| AotError::ToolchainMissing(tool.to_owned()))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(AotError::Compile(format!(
            "{tool} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        )))
    }
}

pub(crate) fn path(p: &Path) -> Result<&str, AotError> {
    p.to_str()
        .ok_or_else(|| AotError::Run(format!("non-utf8 path {}", p.display())))
}

pub(crate) fn unique_tmp_dir() -> Result<std::path::PathBuf, AotError> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("myc-aot-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| AotError::Run(format!("mkdir tmp: {e}")))?;
    Ok(dir)
}

/// Best-effort cleanup of the per-run temp dir.
pub(crate) struct TmpDir(pub(crate) std::path::PathBuf);
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
