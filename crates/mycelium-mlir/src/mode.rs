//! **Explicit, named execution modes** — the never-silently-selected dispatcher (M-727; E15-1;
//! RFC-0029 §7.3; ADR-009; G2/VR-5).
//!
//! ADR-009 already sanctions the hybrid execution model (AOT preferred, the **interpreter** the
//! reference, interpreter/**JIT** for dynamic VSA/HDC), and the M-340 in-process `dlopen` JIT
//! ([`crate::jit`]) already exists and is differential-checked against the interpreter
//! (`tests/jit_differential.rs`). What M-727 adds is the **formalization**: a single, named
//! [`ExecMode`] enum and a [`run`] dispatcher whose contract is that **a mode is chosen only by being
//! named** — there is **no default, no heuristic, no fallback** that could engage the JIT (or swap the
//! AOT env-machine for the interpreter) behind the caller's back. This is RFC-0029 §7.3's never-silent
//! selection contract made executable: `run(ExecMode::Jit, …)` is the *only* path to the JIT, and it
//! is unreachable unless the caller writes `ExecMode::Jit`.
//!
//! **No superseding ADR (RFC-0029 §7.3).** ADR-009 already permits interpreter/JIT, so this module is
//! a formalization + hardening, not a deferral lift — recorded honestly here, no decision rewritten.
//!
//! **Honest scope (VR-5/G2).**
//! - [`ExecMode::Interpreter`] — the M-110 reference interpreter (the trusted base, NFR-7); covers the
//!   whole v0 calculus.
//! - [`ExecMode::Aot`] — the [`crate::aot`] env-machine (an independent big-step evaluator over the
//!   lowered ANF); covers the whole v0 calculus.
//! - [`ExecMode::Jit`] — the [`crate::jit`] in-process compiled path; covers the **bit/trit subset**
//!   it compiles (`bit.*`, `trit.neg/add/sub/mul`), refusing the rest *explicitly*
//!   ([`ModeError::Unsupported`]) and the toolchain-absent case *explicitly*
//!   ([`ModeError::ToolchainMissing`]) — never a silent slow-path or wrap.
//!
//! **Selection is the caller's, deliberately.** This module does **not** pick a mode for you; a
//! "smart" auto-selector that fell back interpreter→JIT on its own would be exactly the silent
//! substitution G2 forbids (interpreter↔JIT divergence must always be *detectable*, which it is only
//! if the caller knows which path ran). The correctness bar before a caller may prefer JIT over the
//! interpreter — `JIT == interpreter` — is `Empirical`, exercised by `tests/jit_differential.rs` and
//! the unified three-way differential (M-729).

use mycelium_core::{Node, Value};
use mycelium_interp::{EvalError, Interpreter, PrimRegistry, SwapEngine};

use crate::llvm::AotError;

/// The **named, explicit** execution modes (M-727; RFC-0029 §7.3). Each variant names exactly one
/// execution path; there is no `Default` and no `Auto` variant — a mode is engaged **only** by being
/// named in a [`run`] call, never inferred. This is the type-level form of the never-silent-selection
/// contract: the JIT is reachable *iff* a caller writes [`ExecMode::Jit`] (G2).
///
/// Deliberately **not** `Default`: there is no "the obvious mode" to fall back to — the caller must
/// choose, so that which path ran is always known (and interpreter↔JIT divergence always detectable).
///
/// **`#[non_exhaustive]` tension (G2).** This is marked `#[non_exhaustive]` so a future mode can be
/// added without a breaking change — but that openness carries a never-silent obligation. When a
/// variant is added: (1) every **in-crate** exhaustive match (`name`, `is_always_available`, the
/// [`run`] dispatch, the `ALL` list) must be updated to handle it, and (2) **downstream** code — which
/// `#[non_exhaustive]` forces to carry a wildcard arm — must treat an unknown variant as an **explicit
/// error**, never route it silently to an existing mode (a wildcard that fell through to, say,
/// `Interpreter` would be exactly the silent substitution this enum exists to forbid). The wildcard is
/// for *refusal*, not for *defaulting*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExecMode {
    /// The M-110 **reference interpreter** — the trusted base (NFR-7), small-step substitution over
    /// the whole v0 calculus. Always available (no toolchain).
    Interpreter,
    /// The [`crate::aot`] **env-machine** — an independent big-step evaluator over the lowered ANF,
    /// covering the whole v0 calculus. Always available (no toolchain): a pure-Rust runnable model,
    /// not the native-LLVM artifact. (The native-LLVM/MLIR compiled artifacts are reached through
    /// [`crate::compile_and_run`] / [`crate::mlir_compile_and_run`]; this mode is the *runnable AOT
    /// model* that the three-way differential pins, M-729.)
    Aot,
    /// The [`crate::jit`] **in-process compiled** path (M-340; `compile → dlopen → call`) over the
    /// bit/trit subset. **Never silently selected** — reachable only by naming this variant.
    /// Toolchain-absent (`clang` missing) and out-of-subset nodes are *explicit* refusals
    /// ([`ModeError`]), never a silent fallback.
    Jit,
}

impl ExecMode {
    /// Every named mode, in a stable order — for tooling, differential parameterization, and an
    /// `EXPLAIN` over "which modes exist". Mirrors the `CertMode::ALL` parameterization idiom.
    pub const ALL: [ExecMode; 3] = [ExecMode::Interpreter, ExecMode::Aot, ExecMode::Jit];

    /// The stable, human-readable name of this mode (for `EXPLAIN`/diagnostics/logs — so a record of
    /// "which mode ran" is legible, never an opaque discriminant).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            ExecMode::Interpreter => "interpreter",
            ExecMode::Aot => "aot",
            ExecMode::Jit => "jit",
        }
    }

    /// Whether this mode can run **without an external toolchain**. The interpreter and the AOT
    /// env-machine are pure-Rust (`true`); the JIT needs `clang` (`false`), so a caller can query this
    /// to choose deliberately rather than discover a `ToolchainMissing` at run time — the choice stays
    /// the caller's, never an automatic fallback (G2).
    #[must_use]
    pub fn is_always_available(self) -> bool {
        !matches!(self, ExecMode::Jit)
    }
}

/// An execution-mode failure (M-727): the unified, **explicit** error surface for [`run`]. Every
/// failure mode is a named variant — there is no silent fallback or wrap (G2). Wraps the underlying
/// [`EvalError`] (interpreter/AOT) or [`AotError`] (JIT) so the originating path's diagnostic is
/// preserved, never flattened to an opaque string.
#[derive(Debug)]
#[non_exhaustive]
pub enum ModeError {
    /// The interpreter or AOT env-machine reported an evaluation error (free variable, overflow,
    /// fuel/depth limit, …). Carries which mode produced it for a legible record.
    Eval {
        /// The mode that produced the error (`Interpreter` or `Aot`).
        mode: ExecMode,
        /// The underlying interpreter/AOT error.
        source: EvalError,
    },
    /// The JIT path reported a node outside its compiled bit/trit subset — an **explicit** refusal,
    /// routed here rather than silently mis-compiled (G2). The caller can re-run under
    /// [`ExecMode::Interpreter`]/[`ExecMode::Aot`] (which cover the whole calculus) — but that
    /// re-selection is the caller's deliberate choice, never automatic.
    Unsupported(String),
    /// The JIT toolchain (`clang`) is absent — an **explicit**, skippable signal (the house "skip
    /// gracefully" idiom), never a silent fall-through to a different mode (G2).
    ToolchainMissing(String),
    /// A balanced-ternary overflow on the JIT path — surfaced explicitly (mirrors the interpreter's
    /// `EvalError::Overflow`), never a silently-wrapped value (SC-3/G2).
    Overflow(String),
    /// Any other JIT-path runtime failure (FFI/IO), surfaced explicitly.
    Jit(String),
}

impl std::fmt::Display for ModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModeError::Eval { mode, source } => {
                write!(f, "{} mode evaluation error: {source}", mode.name())
            }
            ModeError::Unsupported(n) => {
                write!(
                    f,
                    "jit mode: node outside the compiled bit/trit subset: {n}"
                )
            }
            ModeError::ToolchainMissing(t) => write!(f, "jit mode: toolchain missing: {t}"),
            ModeError::Overflow(e) => write!(f, "jit mode: balanced-ternary overflow: {e}"),
            ModeError::Jit(e) => write!(f, "jit mode: {e}"),
        }
    }
}

impl std::error::Error for ModeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModeError::Eval { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Map a JIT-path [`AotError`] into the unified [`ModeError`], preserving the *kind* of failure (so a
/// toolchain skip stays a toolchain skip, an overflow stays an overflow — never collapsed to a generic
/// error, G2). `pub(crate)` for white-box testing (`src/tests/mode.rs`).
pub(crate) fn jit_err(e: AotError) -> ModeError {
    match e {
        AotError::ToolchainMissing(t) => ModeError::ToolchainMissing(t),
        AotError::UnsupportedNode(n) => ModeError::Unsupported(n),
        AotError::Overflow(o) => ModeError::Overflow(o),
        other => ModeError::Jit(other.to_string()),
    }
}

/// **Run `node` under the explicitly named `mode`** (M-727; RFC-0029 §7.3) — the
/// never-silently-selected dispatcher.
///
/// The dispatch is a single `match` on the caller-supplied `mode`: the **only** way to reach the JIT
/// is to pass [`ExecMode::Jit`], and no arm falls through to another mode. There is no heuristic, no
/// "try JIT then fall back" — that fall-back would be the silent substitution G2 forbids. A mode that
/// cannot run a given node (the JIT outside its subset, or with `clang` absent) returns an **explicit**
/// [`ModeError`]; it does **not** quietly run a different mode. Re-selecting after a refusal is the
/// caller's deliberate choice.
///
/// `prims` + `swap` configure the two pure-Rust trusted paths identically (the interpreter is built
/// from them; the AOT env-machine borrows them), so [`ExecMode::Interpreter`] and [`ExecMode::Aot`]
/// run with the *same* primitive registry and swap engine — no silent configuration drift between the
/// reference and the model. They are taken **by value** because the reference interpreter owns its
/// config (`Interpreter::new` requires an owned `PrimRegistry` + `Box<dyn SwapEngine>`); a `&`-only
/// signature can't supply that without cloning a trait object, which `SwapEngine` does not support.
/// The JIT is closed over its own emitter (it compiles the same lowering the AOT path lowers), so it
/// ignores `prims`/`swap`. All three modes agree on the observable (`repr + payload + guarantee`) over
/// the shared subset — the M-729 three-way differential (`tests/threeway_codegen_differential.rs`).
///
/// # Errors
/// Returns [`ModeError`] when the chosen mode refuses or fails the node: `Eval` for an
/// interpreter/AOT evaluation error, `Unsupported`/`ToolchainMissing`/`Overflow`/`Jit` for the JIT
/// path — each explicit, never a silent fallback (G2).
pub fn run(
    mode: ExecMode,
    node: &Node,
    prims: PrimRegistry,
    swap: Box<dyn SwapEngine>,
) -> Result<Value, ModeError> {
    match mode {
        // The reference interpreter owns its config; build it from the shared registry + swap.
        ExecMode::Interpreter => {
            Interpreter::new(prims, swap)
                .eval(node)
                .map_err(|source| ModeError::Eval {
                    mode: ExecMode::Interpreter,
                    source,
                })
        }
        // The AOT env-machine borrows the same config (so the two trusted paths are configured
        // identically — no silent drift between reference and model).
        ExecMode::Aot => crate::aot::run(node, &prims, &*swap).map_err(|source| ModeError::Eval {
            mode: ExecMode::Aot,
            source,
        }),
        // The JIT is reached ONLY here, and ONLY because the caller named `ExecMode::Jit`. No other
        // arm can route to it (never-silent selection, G2). It ignores `prims`/`swap` (closed over its
        // own emitter), so they are simply dropped — never silently re-routed to a configured path.
        ExecMode::Jit => crate::jit::jit_run(node).map_err(jit_err),
    }
}
