//! The runnable AOT artifact (M-150 "runnable artifact for the subset"; M-151 differential target).
//!
//! Executes the **lowered A-normal form** (`mycelium-core::lower`) with a **big-step env-machine**:
//! bindings are evaluated in order into an environment, operands are looked up, primitives and swaps
//! are applied via the shared registries. This is an *independent execution path* from the M-110
//! reference interpreter (small-step substitution over the nested tree), so checking the two for
//! observable equivalence (M-151) is a real NFR-7 two-path test — it stands in for "interpreter vs
//! compiled native" until the libMLIR backend lands.
//!
//! **Full v0 calculus (M-342; RFC-0011 §4.4 Q5 closed).** [`run_core`] covers the whole calculus —
//! `Const/Var/Let/Op/Swap` plus the r3/r4 data + recursion nodes: `Construct` builds a [`Datum`],
//! `Match` selects an arm (binding constructor fields), `Lam` is a closure capturing its environment,
//! `App` applies it call-by-value, and `Fix` unfolds under a fuel clock. The three-way differential
//! (L1-eval ≡ L0-interp ≡ AOT) now spans this whole fragment. (The *native LLVM* backend stays the
//! bit/trit subset and refuses the rest with an explicit `UnsupportedNode` — VR-5; data/closure
//! codegen is the deferred MLIR→LLVM work.)
//!
//! **Stack-robust (M-347).** The machine is a **trampoline** over an *explicit heap control stack*
//! (`eval_machine`): `App`/`Match` push a continuation frame and switch blocks; a completed block
//! returns its value, unwinding the stack. So object-level recursion lives on the **heap**, and the
//! host call stack is **O(1)** — like the reference interpreter. Deep recursion is bounded by two
//! **explicit, graceful** budgets — `fuel` (Fix unfolds; time → [`EvalError::FuelExhausted`]) and a
//! control-stack depth ceiling (space → [`EvalError::DepthLimit`]) — never a host-stack abort and
//! never a hang. (Empirically: pre-trampoline this aborted at ~600 unfolds; post-trampoline it is
//! graceful — see `xtask recursion-probe`, DN-05 §1.1.) The depth ceiling is now resolved
//! **dynamically** from detected memory headroom ([`crate::budget`], DN-05 §2.4 / DN05-Q5): with the
//! control stack on the heap, the budget is a policy over memory, derived honestly and `EXPLAIN`-able,
//! with a conservative static fallback. [`run_core_with_budget`] still takes an explicit ceiling.
//!
//! **Tail-call optimization (M-996; maintainer decision 2026-07-06).** The env-machine now applies
//! the same ratified RFC-0041 §4.0 depth metric the L1 interpreter got in M-994 fix (a): **tail
//! iterations do not charge depth**. A continuation that would merely pass the callee's value
//! through unchanged (a *passthrough* [`Cont`] — see [`Cont::is_tail_passthrough`]) is **elided at
//! push time**: no frame, no depth charge, so a `match`-driven tail countdown runs in O(1) control-
//! stack depth on this path exactly as it does interpreted — closing the live §5.1 family-parity
//! violation where the same program at the same budget succeeded on the interpreter but refused
//! `DepthLimit` here. The two (maintainer-authorized) behavior shifts: a deep terminating tail loop
//! is `DepthLimit → Ok(value)`, and a **divergent** tail loop is `DepthLimit → FuelExhausted` (fuel
//! is the designed non-termination backstop, matching the interpreter's long-standing behavior).
//! Non-tail calls are byte-for-byte unchanged and still refuse at the depth ceiling. Every elision
//! is counted in the [`TcoTrace`] witness (house rule #2: never a silent optimization).
//!
//! **Hot-path representations (M-999) — closing the interpreter-vs-env-machine ordering gap.**
//! Same-profile measurement (`tests/aot_vs_interp_bench.rs`, `Empirical`) showed the machine
//! *slower* than the L1 interpreter (~4.4x on the M-987 snoc shape); a callgrind profile put the
//! loss in allocation/copy churn, removed by four sharing changes (semantics untouched — the full
//! differential suite stays green with zero expectation edits):
//! 1. **Environment** ([`Env`]/[`EnvFrame`]): a mutable top segment over an `Rc`-shared chain of
//!    frozen frames — capture (closure/`Fix`/match arm) is an O(1)-amortized [`Env::snapshot`],
//!    not the former whole-`HashMap` clone. Lookup walks newest-first (innermost wins — observably
//!    the map's insert-overwrite); the frozen chain tears down iteratively (never-silent G2, like
//!    [`AotDatum`]).
//! 2. **Prepared program** ([`Code`]): the lowered ANF is mirrored once at entry into an
//!    `Rc`-shared form, so entering a `Lam`/`Fix` body, match arm, or `FixGroup` member is an O(1)
//!    handle — formerly a deep `body.clone()` of the subtree *per execution*.
//! 3. **Interned atoms** (`Rc<Atom>` keys): binding names/params/match binders are prepared once;
//!    a runtime re-binding is a refcount bump, never a per-step `String` alloc.
//! 4. **Shared repr values** (`AotVal::Repr(Rc<Value>)`): a variable reference is a refcount bump
//!    and a `Const` execution is alloc-free — formerly every reference deep-cloned payload plus
//!    `Meta`, and every move copied the ~100+-byte inline `Value`.
//!
//! Result (release, one host, 2026-07-06): the env-machine runs **~1.5-1.7x faster** than the
//! interpreter on the snoc shape and ~1.2-1.4x on a deep tail loop, from 0.22x/0.7x before — the
//! bench file records the tables and how to re-measure.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use std::rc::Rc;

use mycelium_core::lower::{self, Anf, AnfAlt, Atom, Rhs};
use mycelium_core::{
    ContentHash, CoreValue, CtorRef, Datum, GuaranteeStrength, Node, PackScheme, Payload,
    PhysicalLayout, Repr, Value,
};
use mycelium_interp::{Budgets, EffectKind, EvalError, PrimRegistry, SwapEngine};
use mycelium_workstack::{ensure_sufficient_stack, BudgetError, DepthGuard, RecursionBudget};

use crate::budget::{AutoDepthBudget, DepthBudget, DepthResolution, DEFAULT_PER_FRAME_BYTES};
use crate::pack;

/// The default fuel for the env-machine's `Fix` clock — generous; the guard is against a
/// non-productive recursion, surfaced as an explicit [`EvalError::FuelExhausted`], never a hang
/// (mirrors the reference interpreter, RFC-0007 §4.5).
const AOT_FUEL: u64 = 1_000_000;

/// Resolve the **control-stack depth** ceiling for the trampoline (M-347): the *space* analogue of
/// fuel. The machine refuses past this with an explicit [`EvalError::DepthLimit`] — a graceful limit
/// that bounds heap growth, never an OOM/abort. Resolved **dynamically** from detected memory
/// headroom ([`crate::budget`], DN-05 §2.4 / DN05-Q5): a fixed constant is too timid on a large host
/// and too bold on a small one, so the default policy derives it from `MemAvailable`/`RLIMIT_AS` with
/// a conservative static fallback. The basis is `EXPLAIN`-able ([`default_depth_budget`]).
fn resolve_max_depth() -> usize {
    AutoDepthBudget::default().resolve().max_depth
}

/// The default depth-budget resolution — the resolved ceiling **and** its `EXPLAIN`-able basis (no
/// black box, G2). Exposed for tooling/diagnostics (the xtask probe, a future `EXPLAIN`) so the
/// chosen limit and *why* are always inspectable, never an opaque magic number (DN-05 §2.4 / DN05-Q5).
pub fn default_depth_budget() -> DepthResolution {
    AutoDepthBudget::default().resolve()
}

/// A value in the AOT env-machine: a representation value, a **structurally-shared datum**, or a
/// **closure** / **recursive suspension** for the r4 function fragment. Closures capture their
/// defining environment by value (the v0 surface is first-order, so this is a finite capture). Bodies
/// are [`Rc`]-shared so closures/continuation frames don't clone the block tree.
///
/// **M-994 (b) — field-level structural sharing.** A datum is an AOT-local cons cell
/// (`Data(Rc<AotDatum>)`, where [`AotDatum`] holds `ctor` + `fields: Vec<AotVal>` + `guarantee`),
/// **not** an inlined [`mycelium_core::Datum`]. The [`Rc`] around the node makes the whole sub-tree
/// shared, so:
/// - a variable reference (`lookup`'s `.cloned()`) and an environment clone are an **O(1)** refcount
///   bump, not an O(nodes) deep clone of the whole spine; and
/// - a `Match` arm binding a constructor field (`select_arm`) is an **O(1)** `AotVal::clone` (a
///   refcount bump on the sub-tree), not an O(subtree) deep copy out of a `Vec<CoreValue>`.
///
/// This is the AOT analogue of the L1 interpreter's `Arc<Vec<..>>` on `L1Value::Data` (M-987): the
/// frozen `mycelium_core::Datum` (its `fields: Vec<CoreValue>` is inside the DN-56 freeze) is **not**
/// modified — the sharing lives entirely in this crate's env-machine value, and a `Datum`/`CoreValue`
/// is materialised only at the observable boundaries ([`to_core`]: final result, `Op`/`Swap` operand).
/// Building datums *only* from `AotVal` fields (`Construct`) means no `mycelium_core::Datum` is ever
/// converted **into** an `AotVal`, so the sharing is closed. `guarantee` is the same meet-summary
/// [`Datum::new`] computes — carried incrementally so it survives destructure/reconstruct unchanged.
//
// M-999 supersedes the earlier "repr `Value` is intentionally inlined" trade-off note: the callgrind
// profile behind `tests/aot_vs_interp_bench.rs` showed the inline `Value` was the machine's dominant
// memcpy/alloc source (every variable reference deep-cloned payload + `Meta`; every move copied the
// ~100+-byte enum), so a repr value is now `Rc`-shared like `Data` — a reference is a refcount bump,
// a `Const` execution is alloc-free (the prepared program holds the `Rc`), and the enum is pointer-
// sized. The one new cost is a single `Rc` box per *freshly produced* op/swap result (`Empirical`:
// the swap was measured faster, same file).
/// One prepared mutual-recursion group, shared across the group's per-member suspensions:
/// `(member name, interned binding key, prepared body)` per member (M-999).
type CodeDefs = Rc<Vec<(String, Rc<Atom>, Rc<Code>)>>;

#[derive(Clone)]
enum AotVal {
    /// A representation value (a repr normal form), `Rc`-shared (M-999): a reference/clone is an
    /// O(1) refcount bump; an owned `Value` is materialised only at the observable boundaries
    /// ([`to_core`], and never for operands — prims/swaps take `&Value`).
    Repr(Rc<Value>),
    /// A structurally-shared datum: an [`Rc`]-shared [`AotDatum`] (constructor + field values + the
    /// meet-summary guarantee). Cloning is an O(1) `Rc` bump; the recursion-safe iterative `Drop` lives
    /// on [`AotDatum`] (not `AotVal`), so `AotVal` stays freely movable while a deep spine still tears
    /// down without overflowing the host stack (never-silent G2).
    Data(Rc<AotDatum>),
    /// A lambda closure: parameter (an interned key into the prepared program — cloning it is an
    /// `Rc` bump, never a `String` alloc; M-999), body block (a shared handle), and the captured
    /// environment.
    Closure {
        param: Rc<Atom>,
        body: Rc<Code>,
        env: Env,
    },
    /// A `Fix` suspension: unfolds on application under the fuel clock (`name` interned like
    /// `Closure::param`).
    Fix {
        name: Rc<Atom>,
        body: Rc<Code>,
        env: Env,
    },
    /// A mutual-recursion group member (RFC-0001 r5): on application it re-binds every member name to
    /// its own suspension (so siblings can call each other) and enters `which`'s body, under the fuel
    /// clock — the env-machine analogue of the interpreter's `FixGroup` focus unfold.
    FixGroup {
        /// All members of the group `(name, interned binding key, prepared body)`, shared across
        /// the per-member suspensions.
        defs: CodeDefs,
        /// Which member this suspension resolves to on application.
        which: String,
        /// The environment captured at the group's binding site.
        env: Env,
    },
}

/// The payload of an [`AotVal::Data`]: a saturated constructor, its field values, and the meet-summary
/// guarantee (identical to what [`Datum::new`] computes). Kept behind an [`Rc`] in `AotVal` so cloning
/// a datum is an O(1) refcount bump and destructure-binding a field is an O(1) `AotVal::clone`. The
/// recursion-safe iterative [`Drop`] lives **here** (not on `AotVal`) so that `AotVal` carries no
/// `Drop` and its variants stay freely movable (`Rc::try_unwrap`/pattern moves), while a deep datum
/// spine still tears down iteratively (never-silent G2).
struct AotDatum {
    ctor: CtorRef,
    fields: Vec<AotVal>,
    guarantee: GuaranteeStrength,
}

impl AotVal {
    /// This value's guarantee: a repr value's own `Meta.guarantee`, or a datum's carried meet-summary
    /// (equal to what [`Datum::new`] computes). Mirrors [`CoreValue::guarantee`] — the honesty accessor
    /// the `Match` guarantee-meet rule reads (a function has no guarantee; it is never a `Match`
    /// scrutinee in a well-typed program, and the `Match` handler refuses one before this is reached).
    fn guarantee(&self) -> GuaranteeStrength {
        match self {
            AotVal::Repr(v) => v.meta().guarantee(),
            AotVal::Data(d) => d.guarantee,
            // Unreachable for a well-typed scrutinee (the `Match` handler rejects a function value
            // with `FunctionResult` first); `Exact` is the neutral element for the meet, never an
            // upgrade (VR-5).
            AotVal::Closure { .. } | AotVal::Fix { .. } | AotVal::FixGroup { .. } => {
                GuaranteeStrength::Exact
            }
        }
    }
}

/// Materialise an [`AotVal`] as a [`mycelium_core::CoreValue`] at an observable boundary (the final
/// result) — the point where the env-machine's structurally-shared datum becomes a frozen-kernel
/// `Datum`. A bare function value is the explicit [`EvalError::FunctionResult`] (a v0 entry returns a
/// repr/data value, never a function — mirrors the interpreter). **Iterative** (an explicit
/// task/value worklist, exactly like `mycelium_core`'s `clone_core`), so a deep datum spine converts
/// without overflowing the host stack (never-silent G2). The rebuilt `Datum` recomputes the same
/// meet-summary from the same fields, so the result is byte-identical to the pre-M-994 path — the
/// differential is unmoved.
fn to_core(v: AotVal) -> Result<CoreValue, EvalError> {
    enum Task {
        Expand(AotVal),
        Assemble { ctor: CtorRef, arity: usize },
    }
    let mut tasks: Vec<Task> = vec![Task::Expand(v)];
    let mut done: Vec<CoreValue> = Vec::new();
    while let Some(task) = tasks.pop() {
        match task {
            Task::Expand(av) => match av {
                AotVal::Repr(rv) => done.push(CoreValue::Repr(
                    // Own the value without a deep copy when uniquely owned (the common case at
                    // the consuming boundary); a still-shared value falls back to one clone.
                    Rc::try_unwrap(rv).unwrap_or_else(|rc| (*rc).clone()),
                )),
                AotVal::Data(rc) => {
                    // Own the datum without a deep copy when uniquely owned (the common case at a
                    // consuming boundary); a still-shared spine falls back to cloning this node's
                    // `Vec` (O(1) per element — an `AotVal` refcount bump — deep work amortised by the
                    // worklist). `mem::take` (not a field move) sidesteps `AotDatum`'s `Drop`.
                    let mut d = Rc::try_unwrap(rc).unwrap_or_else(|rc| (*rc).clone());
                    let ctor = d.ctor.clone();
                    let fields = std::mem::take(&mut d.fields);
                    tasks.push(Task::Assemble {
                        ctor,
                        arity: fields.len(),
                    });
                    for f in fields {
                        tasks.push(Task::Expand(f));
                    }
                }
                AotVal::Closure { .. } | AotVal::Fix { .. } | AotVal::FixGroup { .. } => {
                    return Err(EvalError::FunctionResult)
                }
            },
            Task::Assemble { ctor, arity } => {
                let mut fields = Vec::with_capacity(arity);
                for _ in 0..arity {
                    fields.push(done.pop().expect("to_core: datum field underflow"));
                }
                // `Datum::new` recomputes the meet-summary — identical to the carried `guarantee`
                // (same fields, same rule), so nothing is upgraded (VR-5).
                done.push(CoreValue::Data(Datum::new(ctor, fields)));
            }
        }
    }
    Ok(done.pop().expect("to_core: exactly one root value remains"))
}

/// Clone an [`AotDatum`] shell-only (its `fields` are O(1)-cloned `AotVal`s — a `Data` field is a `Rc`
/// bump). Used by `to_core`'s shared-spine fallback; the derived `Clone` would be identical but the
/// manual `Drop` below forbids `#[derive(Clone)]`'s field access, so it is spelled out.
impl Clone for AotDatum {
    fn clone(&self) -> Self {
        AotDatum {
            ctor: self.ctor.clone(),
            fields: self.fields.clone(),
            guarantee: self.guarantee,
        }
    }
}

impl Drop for AotDatum {
    /// **Iterative** teardown of a deep `Data` spine (never-silent G2), mirroring
    /// `mycelium_core::Datum::drop`: a length-*n* list is an *n*-deep `AotDatum` chain whose derived
    /// (recursive) drop would `SIGABRT`. Each still-uniquely-owned child `Rc<AotDatum>` is pushed onto
    /// a worklist and its `fields` emptied before the shell drops, so re-entrant drop sees empty
    /// `fields` — bounded reentrancy, never deep recursion. A **shared** child (`Rc` strong-count > 1)
    /// is left for its last owner (`Rc::into_inner` yields `None`), so no double-free and no over-eager
    /// reclaim. `#![forbid(unsafe_code)]` holds — only safe `Rc`/`Vec`/`mem::take`.
    fn drop(&mut self) {
        let mut work: Vec<Rc<AotDatum>> = Vec::new();
        // Seed the worklist with this datum's own `Data` children, taking their `Rc` out.
        for child in self.fields.drain(..) {
            if let AotVal::Data(rc) = child {
                work.push(rc);
            }
        }
        while let Some(rc) = work.pop() {
            // Reclaim (and descend into) the child only if we are its last owner; otherwise another
            // owner keeps it alive and its own final drop handles it.
            if let Some(mut d) = Rc::into_inner(rc) {
                for grandchild in d.fields.drain(..) {
                    if let AotVal::Data(rc) = grandchild {
                        work.push(rc);
                    }
                }
                // `d` drops here as a childless shell (its `fields` are now empty).
            }
        }
    }
}

/// One **frozen** segment of an environment, shared by `Rc`: the bindings captured up to some
/// snapshot point, plus the (also frozen) parent chain below them. Frames are immutable after
/// construction — sharing one is an O(1) refcount bump, never a copy (M-999).
struct EnvFrame {
    /// The segment's bindings, oldest-first (lookup scans newest-first for innermost-wins). Keys
    /// are the **interned** atoms of the prepared program (`Rc<Atom>`), so binding costs a
    /// refcount bump, never a per-step `String` alloc (M-999).
    bindings: Vec<(Rc<Atom>, AotVal)>,
    /// The frozen chain this segment extends (`None` at the root).
    parent: Option<Rc<EnvFrame>>,
}

impl Drop for EnvFrame {
    /// **Iterative** teardown of the frozen parent *chain* (never-silent G2), mirroring
    /// [`AotDatum`]'s drop: a chain of uniquely-owned frames would otherwise drop recursively,
    /// one host-stack frame per env frame. Chain length is lexically bounded in practice (one
    /// frame per capture point executed in a block — program-text, not input, sized), but the
    /// iterative form makes that a non-assumption. Each still-uniquely-owned parent is unlinked
    /// onto a loop before its shell drops; a **shared** parent (`Rc` strong-count > 1) is left
    /// for its last owner. Bindings holding *nested closures* (a closure whose env holds a
    /// closure …) still unwind by bounded reentrancy through `AotVal` — the same pre-existing
    /// property the `HashMap` env had; this `Drop` addresses the new linear chain only.
    fn drop(&mut self) {
        let mut next = self.parent.take();
        while let Some(rc) = next {
            match Rc::into_inner(rc) {
                // We are the last owner: unlink its parent into the loop, then let the
                // now-chainless shell drop (its own Drop sees `parent == None` — no recursion).
                Some(mut frame) => next = frame.parent.take(),
                // Shared tail — another owner keeps it alive; its final drop handles it.
                None => break,
            }
        }
    }
}

/// The env-machine environment (M-999): a small **mutable top** segment (the bindings made since
/// the last capture) over an `Rc`-shared chain of **frozen** [`EnvFrame`]s. This replaces the
/// former `HashMap<Atom, AotVal>` whose *whole-map clone* at every closure capture, `Fix`/
/// `FixGroup` suspension, match-arm entry, and function-value lookup was the measured constant-
/// factor gap between the env-machine and the L1 interpreter (the M-995 report's residual;
/// baseline in `tests/aot_vs_interp_bench.rs`):
/// - [`Env::insert`] is a `Vec` push (amortized O(1); shadowing = later entry wins on lookup,
///   observably identical to the map's overwrite);
/// - [`Env::snapshot`] (capture) freezes the top segment into a shared frame — **O(1) amortized**
///   (each binding is moved into a frozen frame at most once), where the map clone was O(live
///   bindings) *per capture*;
/// - [`Env::get`] scans the top newest-first, then walks the frozen chain — innermost binding
///   wins, exactly the map-insert-overwrite semantics; cost is bounded by lexical scope size
///   (the same discipline as the L1 interpreter's `scope: Vec<(String, L1Value)>` scan).
#[derive(Clone, Default)]
struct Env {
    /// Bindings made since the last snapshot, oldest-first (mutable working segment); keys are
    /// interned (see [`EnvFrame::bindings`]).
    top: Vec<(Rc<Atom>, AotVal)>,
    /// The frozen, shared tail (`None` at the root).
    parent: Option<Rc<EnvFrame>>,
}

impl Env {
    /// An empty environment.
    fn new() -> Self {
        Env::default()
    }

    /// Bind `name := val` — a push; an existing binding of the same name is *shadowed* (lookup
    /// returns the newest), which is observably the former map's insert-overwrite.
    fn insert(&mut self, name: Rc<Atom>, val: AotVal) {
        self.top.push((name, val));
    }

    /// The innermost binding of `a`, if any: scan the mutable top newest-first, then each frozen
    /// frame newest-first down the chain.
    fn get(&self, a: &Atom) -> Option<&AotVal> {
        if let Some((_, v)) = self.top.iter().rev().find(|(n, _)| **n == *a) {
            return Some(v);
        }
        let mut frame = self.parent.as_deref();
        while let Some(f) = frame {
            if let Some((_, v)) = f.bindings.iter().rev().find(|(n, _)| **n == *a) {
                return Some(v);
            }
            frame = f.parent.as_deref();
        }
        None
    }

    /// Capture the current environment by value — the O(1)-amortized replacement for the former
    /// whole-map clone. A non-empty top segment is **frozen** in place (moved into a new shared
    /// [`EnvFrame`]; `self` keeps evaluating on a fresh empty top over that frame), then the
    /// capture is just an `Rc` bump of the frozen chain. Every captured `Env` therefore has an
    /// empty top, so cloning a captured value (`AotVal::Closure`/`Fix`/`FixGroup` in [`lookup`])
    /// stays O(1) too.
    fn snapshot(&mut self) -> Env {
        if !self.top.is_empty() {
            let frame = Rc::new(EnvFrame {
                bindings: std::mem::take(&mut self.top),
                parent: self.parent.take(),
            });
            self.parent = Some(frame);
        }
        Env {
            top: Vec::new(),
            parent: self.parent.clone(),
        }
    }
}

fn lookup(env: &Env, a: &Atom) -> Result<AotVal, EvalError> {
    env.get(a).cloned().ok_or_else(|| match a {
        Atom::Named(x) => EvalError::FreeVariable(x.clone()),
        Atom::Temp(k) => EvalError::FreeVariable(format!("%{k}")),
    })
}

/// The **prepared program** (M-999): the lowered [`Anf`] mirrored **once** at entry into an
/// `Rc`-shared crate-local form, so every nested body a running program re-enters — a `Lam`/`Fix`
/// body, a match arm/default, a `FixGroup` member — is an **O(1) `Rc::clone` handle**, not a deep
/// clone of the subtree. Under the old shape the machine deep-cloned program text *per
/// execution* (`Rc::new(body.clone())` on every closure-binding evaluation and every selected
/// match arm — O(subtree) each, every loop iteration), which — together with the `HashMap` env
/// clone — was the measured constant-factor loss to the L1 interpreter, whose walker borrows
/// `&Node` and never clones code (`tests/aot_vs_interp_bench.rs`). The mirror costs one
/// O(program-text) pass (recursive over lexical nesting, exactly like `lower_to_anf` itself,
/// inside the same `ensure_sufficient_stack` guard); semantics are untouched — this is the same
/// tree, shared instead of re-copied.
pub(crate) struct Code {
    bindings: Vec<CodeBinding>,
    result: Atom,
}

/// One prepared binding: the lowered [`lower::Binding`] minus the `layout` field (scheduling
/// metadata the env-machine never reads — the LLVM backend consumes it from the `Anf` directly).
struct CodeBinding {
    /// The binding's name, interned behind `Rc` so every runtime re-binding of it is a refcount
    /// bump (M-999).
    name: Rc<Atom>,
    rhs: CodeRhs,
}

/// The prepared [`Rhs`]: identical shape, with nested blocks behind [`Rc<Code>`].
enum CodeRhs {
    /// The constant is pre-`Rc`ed so each execution is a refcount bump, alloc-free (M-999).
    Const(Rc<Value>),
    Alias(Atom),
    Op {
        prim: String,
        args: Vec<Atom>,
    },
    Swap {
        src: Atom,
        target: Repr,
        policy: ContentHash,
    },
    Construct {
        ctor: CtorRef,
        args: Vec<Atom>,
    },
    App {
        func: Atom,
        arg: Atom,
    },
    Lam {
        param: Rc<Atom>,
        body: Rc<Code>,
    },
    Fix {
        name: Rc<Atom>,
        body: Rc<Code>,
    },
    FixGroup {
        defs: CodeDefs,
        which: String,
    },
    Match {
        scrutinee: Atom,
        alts: Vec<CodeAlt>,
        default: Option<Rc<Code>>,
    },
}

/// The prepared [`AnfAlt`]: identical shape, arm bodies behind [`Rc<Code>`].
enum CodeAlt {
    Ctor {
        ctor: CtorRef,
        binders: Vec<Rc<Atom>>,
        body: Rc<Code>,
    },
    Lit {
        value: Value,
        body: Rc<Code>,
    },
}

impl Code {
    /// Mirror a lowered block into the shared form (one deep pass; see the type-level doc).
    pub(crate) fn prepare(anf: &Anf) -> Rc<Code> {
        let bindings = anf
            .bindings()
            .iter()
            .map(|b| CodeBinding {
                name: Rc::new(b.name.clone()),
                rhs: match &b.rhs {
                    Rhs::Const(v) => CodeRhs::Const(Rc::new(v.clone())),
                    Rhs::Alias(a) => CodeRhs::Alias(a.clone()),
                    Rhs::Op { prim, args } => CodeRhs::Op {
                        prim: prim.clone(),
                        args: args.clone(),
                    },
                    Rhs::Swap {
                        src,
                        target,
                        policy,
                    } => CodeRhs::Swap {
                        src: src.clone(),
                        target: target.clone(),
                        policy: policy.clone(),
                    },
                    Rhs::Construct { ctor, args } => CodeRhs::Construct {
                        ctor: ctor.clone(),
                        args: args.clone(),
                    },
                    Rhs::App { func, arg } => CodeRhs::App {
                        func: func.clone(),
                        arg: arg.clone(),
                    },
                    Rhs::Lam { param, body } => CodeRhs::Lam {
                        param: Rc::new(Atom::Named(param.clone())),
                        body: Code::prepare(body),
                    },
                    Rhs::Fix { name, body } => CodeRhs::Fix {
                        name: Rc::new(Atom::Named(name.clone())),
                        body: Code::prepare(body),
                    },
                    Rhs::FixGroup { defs, which } => CodeRhs::FixGroup {
                        defs: Rc::new(
                            defs.iter()
                                .map(|(n, d)| {
                                    (n.clone(), Rc::new(Atom::Named(n.clone())), Code::prepare(d))
                                })
                                .collect(),
                        ),
                        which: which.clone(),
                    },
                    Rhs::Match {
                        scrutinee,
                        alts,
                        default,
                    } => CodeRhs::Match {
                        scrutinee: scrutinee.clone(),
                        alts: alts
                            .iter()
                            .map(|alt| match alt {
                                AnfAlt::Ctor {
                                    ctor,
                                    binders,
                                    body,
                                } => CodeAlt::Ctor {
                                    ctor: ctor.clone(),
                                    binders: binders
                                        .iter()
                                        .map(|b| Rc::new(Atom::Named(b.clone())))
                                        .collect(),
                                    body: Code::prepare(body),
                                },
                                AnfAlt::Lit { value, body } => CodeAlt::Lit {
                                    value: value.clone(),
                                    body: Code::prepare(body),
                                },
                            })
                            .collect(),
                        default: default.as_ref().map(Code::prepare),
                    },
                },
            })
            .collect();
        Rc::new(Code {
            bindings,
            result: anf.result().clone(),
        })
    }

    /// The number of bindings in this block (white-box test access — fields stay module-private).
    #[cfg(test)]
    pub(crate) fn bindings_len(&self) -> usize {
        self.bindings.len()
    }

    /// The block's result atom (white-box test access).
    #[cfg(test)]
    pub(crate) fn result(&self) -> &Atom {
        &self.result
    }
}

/// Coerce an [`AotVal`] to a representation [`Value`] (for an `Op`/`Swap` operand): a datum or a
/// function in that position is a type error the checker prevents — explicit, never a guess. A repr
/// `Value` is bounded-depth by construction, so cloning it is cheap (unlike a `Datum` spine); the
/// deep-clone hazard (b) targets only the recursive `Data` case, handled by the shared-field machine.
fn as_repr_value(v: AotVal) -> Result<Rc<Value>, EvalError> {
    match v {
        AotVal::Repr(rv) => Ok(rv),
        AotVal::Data(_) => Err(EvalError::DataMalformed {
            why: "a primitive/swap operand reduced to a data value, not a representation value"
                .to_owned(),
        }),
        AotVal::Closure { .. } | AotVal::Fix { .. } | AotVal::FixGroup { .. } => {
            Err(EvalError::DataMalformed {
                why: "a primitive/swap operand reduced to a function value".to_owned(),
            })
        }
    }
}

/// Run a Core IR program through the AOT path to a [`CoreValue`] (the full v0 calculus — repr, data,
/// and recursion; M-342). Lowers to ANF, then evaluates with a **trampolined** environment machine
/// (an explicit heap control stack — *O(1) host stack*, M-347), an independent path from the M-110
/// small-step interpreter (the NFR-7 two-path check).
pub fn run_core(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
) -> Result<CoreValue, EvalError> {
    run_core_with_budget(node, prims, swap, AOT_FUEL, resolve_max_depth())
}

/// [`run_core`] with an explicit `Fix`-unfold (fuel) budget and the dynamically-resolved depth ceiling.
pub fn run_core_with_fuel(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    fuel: u64,
) -> Result<CoreValue, EvalError> {
    run_core_with_budget(node, prims, swap, fuel, resolve_max_depth())
}

/// [`run_core`] with **both** budgets explicit (M-347): `fuel` bounds `Fix` unfolds (time), `max_depth`
/// bounds the control-stack depth (space). Each overrun is an **explicit, graceful** error
/// ([`EvalError::FuelExhausted`] / [`EvalError::DepthLimit`]) — never a hang and never a host-stack
/// abort. This is the **explicit override**: `max_depth` is whatever the caller passes; the
/// `run_core`/`run_core_with_fuel` entries instead resolve it *dynamically* from detected memory
/// headroom ([`crate::budget`], DN-05 §2.4 / DN05-Q5).
pub fn run_core_with_budget(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    fuel: u64,
    max_depth: usize,
) -> Result<CoreValue, EvalError> {
    // The default entry carries an *empty* effect ledger: no `alloc` effect budget is declared, so the
    // depth ceiling remains the sole space guard (identical pre-RFC-0014-§4.8 behaviour).
    run_core_with_effects(node, prims, swap, fuel, max_depth, &mut Budgets::new())
}

/// [`run_core_with_budget`] with a shared **effect-budget ledger** threaded through the env-machine
/// (RFC-0014 §4.8 — completing the deferred integration). The ledger is the *same*
/// [`mycelium_interp::Budgets`] the recovery driver consumes, and an overrun surfaces as
/// [`EvalError::EffectBudget`] — the effect sibling of `FuelExhausted`/`DepthLimit`, on the **one
/// runtime refusal channel** (RFC-0014 §8: separate named budgets, one enforcement mechanism).
///
/// v0 L0 has **no effect node** (KC-3 — no kernel hook), so the machine spends only the budgets that
/// correspond to costs it *intrinsically* incurs: a declared **`alloc`** budget is charged
/// [`DEFAULT_PER_FRAME_BYTES`] per control-stack frame, at the same frame-push site the depth ceiling
/// guards — making the `alloc` effect budget the **opt-in** sibling of the DN-05 depth ceiling. Absent
/// (the default empty ledger) ⇒ no charge ⇒ unchanged behaviour (I5: a broader bound is opt-in). The
/// `retry`/`cascade` budgets are spent by the recovery *driver* over this same ledger and channel; the
/// concurrency wave (RFC-0008) layers *per-task* ledgers on this seam.
pub fn run_core_with_effects(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    fuel: u64,
    max_depth: usize,
    budgets: &mut Budgets,
) -> Result<CoreValue, EvalError> {
    // The public entry discards the TCO witness (identical semantics); the traced sibling exists so
    // the elision count is *observable* where it matters (white-box tests / diagnostics — M-996).
    run_core_with_effects_traced(node, prims, swap, fuel, max_depth, budgets).0
}

/// [`run_core_with_effects`] plus the [`TcoTrace`] elision witness (M-996; house rule #2). Crate-
/// internal: the white-box tests assert `total_elided` so the TCO is test-witnessed, not inferred;
/// see the `TcoTrace` FLAG about a future user-facing EXPLAIN surface.
pub(crate) fn run_core_with_effects_traced(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    fuel: u64,
    max_depth: usize,
    budgets: &mut Budgets,
) -> (Result<CoreValue, EvalError>, TcoTrace) {
    let mut tco = TcoTrace::default();
    let result = run_core_impl(node, prims, swap, fuel, max_depth, budgets, &mut tco);
    (result, tco)
}

/// The shared implementation behind [`run_core_with_effects`]/[`run_core_with_effects_traced`].
fn run_core_impl(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    fuel: u64,
    max_depth: usize,
    budgets: &mut Budgets,
    tco: &mut TcoTrace,
) -> Result<CoreValue, EvalError> {
    // RFC-0041 W3½: the AOT env-machine now charges the *shared* `mycelium-workstack` recursion budget
    // at each control-stack frame-push, so its never-silent depth ceiling and host-stack guard are the
    // same primitives the L1/L0 machines use. `max_depth` (the DN-05 §2.4 resolved ceiling — the
    // differential floor 4096, the dynamic `[10k,2M]` headroom outside) maps 1:1 to the budget's depth
    // limit, so the accept/reject threshold is byte-for-byte the pre-extraction `stack.len() >= max_depth`
    // boundary (behavior-preserving — the W3½ gate). A `max_depth` above `u32::MAX` saturates to
    // `u32::MAX` (unreachable under the DN-05 `[10k,2M]` clamp; an explicit caller passing more gets a
    // ceiling reported as `u32::MAX`, never a wraparound — never-silent).
    let depth_limit = u32::try_from(max_depth).unwrap_or(u32::MAX);
    // Grow the host stack once at entry (RFC-0041 §4.3 / §4.4). The trampoline already keeps the AOT's
    // *object-level* recursion on the heap (O(1) host stack), so this is the shared-guard consistency
    // wrap and a backstop for any host-recursive callee — the growth is bounded by the depth ceiling.
    // The `RecursionBudget` is created *inside* the closure: it is `Send` but not `Sync`, so it is owned
    // on the worker rather than borrowed across the thread boundary (mir-passes `emit` precedent).
    let sizing = RecursionBudget::new(depth_limit, u64::MAX, u64::MAX);
    ensure_sufficient_stack(&sizing, move || {
        let budget = RecursionBudget::new(depth_limit, u64::MAX, u64::MAX);
        // Lower, then mirror ONCE into the `Rc`-shared prepared form (M-999): both passes are
        // O(program text) and recursive over the same lexical nesting, inside this stack guard.
        let top = Code::prepare(&lower::lower_to_anf(node));
        let mut fuel = fuel;
        let result = eval_machine(
            top,
            Env::new(),
            prims,
            swap,
            &mut fuel,
            &budget,
            budgets,
            tco,
        )?;
        to_core(result)
    })
}

/// Run a Core IR program through the AOT path to a representation [`Value`]. Convenience over
/// [`run_core`] for the repr fragment: a data result is the explicit [`EvalError::DataResult`]
/// (mirrors `Interpreter::eval`), never a silent mishandling.
pub fn run(node: &Node, prims: &PrimRegistry, swap: &dyn SwapEngine) -> Result<Value, EvalError> {
    match run_core(node, prims, swap)? {
        CoreValue::Repr(v) => Ok(v),
        CoreValue::Data(_) => Err(EvalError::DataResult),
    }
}

/// A continuation: where to bind a returned value and resume. The reified caller context.
pub(crate) struct Cont {
    block: Rc<Code>,
    idx: usize,
    env: Env,
    name: Rc<Atom>,
}

impl Cont {
    /// True iff resuming this continuation with a value is the **identity on that value** — the
    /// tail-transparency test of the M-996 TCO (RFC-0041 §4.0/§4.6, mirroring the L1 interpreter's
    /// M-994 fix (a)). A `Resume` of this continuation binds `name := val`, finds the block complete
    /// (`idx` past the last binding), looks up `block.result()` — which **is** `name`, so it reads
    /// back exactly `val` — and passes it to the next frame. That holds *unconditionally* on the Ok
    /// path, and the Err path never resumes any frame (an error aborts the whole machine), so a
    /// passthrough frame is observationally transparent on **both** Ok and Err.
    ///
    /// **Why this is the whole "peek", with no stack walk (the deliberate divergence from the L1
    /// interpreter's shape):** the interpreter discovers a tail call by peeking *down* its stack
    /// through already-pushed binder-restoring frames (`MatchPop`/`LetPop`) to the caller's
    /// `InvokePost`. In this ANF machine the analogous transparent frames are exactly the
    /// passthrough `Resume`s — and because transparency is an intrinsic, O(1)-checkable property of
    /// the continuation *itself*, we never push a passthrough frame in the first place (the "commit"
    /// is eliding the push, which eagerly drops the caller's saved env — the interpreter's
    /// drain-cleanup analog). No transparent frame ever enters the stack, so no drain is needed and
    /// the frame below is always the real (non-transparent) consumer.
    ///
    /// **Cross-module invariant note (PR #1193 review, MEDIUM):** under the *current*
    /// `mycelium_core::lower::lower_to_anf` lowering, `Node::Let` always emits a trailing `Alias`
    /// binding, so every block reachable via the public `Node` API that completes at a `Resume`
    /// already has `result() == name` — making the second conjunct unreachable-false through that
    /// API today. It is **kept as required defense-in-depth**: it is what makes this condition
    /// *locally sound* rather than silently dependent on another module's lowering shape (if a
    /// future lowering emits a block whose result is not the bound name, eliding it would return
    /// the wrong value). Pinned directly by the white-box unit test
    /// `tests::aot::is_tail_passthrough_requires_result_to_be_the_bound_name`.
    pub(crate) fn is_tail_passthrough(&self) -> bool {
        self.idx >= self.block.bindings.len() && self.block.result == *self.name
    }

    /// Test-only constructor for the white-box `is_tail_passthrough` pin (PR #1193 review) —
    /// builds a `Cont` with an empty env without exposing the private `Env`/`AotVal` types.
    #[cfg(test)]
    pub(crate) fn probe(block: Rc<Code>, idx: usize, name: Atom) -> Self {
        Cont {
            block,
            idx,
            env: Env::new(),
            name: Rc::new(name),
        }
    }
}

/// The M-996 TCO elision tally — the env-machine's **observable witness** that tail-call elision
/// actually happened (house rule #2: an optimization that changes the depth accounting must never
/// be a black box; the L1 interpreter's analog is `mycelium_l1::TcoTrace`/`total_elided`, RFC-0041
/// §4.6 tco32). Deliberately minimal — a saturating counter of elided control-stack frames (tail
/// `App` `Resume` frames plus tail-position `Match` `Resume` frames), surfaced through the
/// crate-internal traced runner [`run_core_with_effects_traced`] so white-box tests assert elision
/// happened rather than inferring it from depth behavior.
///
/// FLAG (M-996, for the integrating parent): `run_core` has **no public stats/EXPLAIN surface** to
/// hang a user-facing trace on today (unlike `Evaluator::tco_trace`); whether to expose one (e.g. a
/// per-callee ring like the interpreter's) is a follow-up decision, recorded here rather than
/// silently skipped (G2/VR-5).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TcoTrace {
    /// Total frames elided by TCO (saturating; diagnostic — never load-bearing for semantics).
    pub(crate) total_elided: u64,
}

impl TcoTrace {
    /// Record one elided frame (saturating — the witness never wraps, G2).
    fn record(&mut self) {
        self.total_elided = self.total_elided.saturating_add(1);
    }
}

/// A frame on the explicit **heap** control stack — what makes the machine O(1) host stack. Each frame
/// holds an RAII [`DepthGuard`] (RFC-0041 W3½): the shared [`RecursionBudget`] is charged one unit when
/// the frame is pushed and releases it when the frame is popped/dropped, so `budget.current_depth()`
/// tracks `stack.len()` exactly and the depth ceiling is the shared `mycelium-workstack` primitive —
/// preserving the pre-extraction `stack.len() >= max_depth` accept/reject threshold byte-for-byte.
// `pub(crate)` (fields private) so the in-crate white-box size pin (`src/tests/aot.rs`) can measure it.
pub(crate) struct Frame<'b> {
    /// The per-frame depth reservation on the shared budget; released on pop/drop. Underscored because
    /// it is held purely for its `Drop` side-effect (the release) — the drop *is* the "read".
    _guard: DepthGuard<'b>,
    /// What to do when this frame is resumed.
    kind: FrameKind,
}

/// The action a resumed control-stack [`Frame`] performs.
// `ApplyThen` carries an inlined `AotVal` (see the note on `AotVal`); the size asymmetry with
// `Resume` is the same accepted trade-off.
#[allow(clippy::large_enum_variant)]
enum FrameKind {
    /// Bind the returned value to `name`, then resume `block` at `idx` in `env`.
    Resume(Cont),
    /// The returned value is a function; **apply** it to `arg`, then resume per the continuation.
    /// (The two-step shape of a `Fix` application: unfold the body to a closure, then call it.)
    ApplyThen { arg: AotVal, cont: Cont },
}

/// Map the shared budget's never-silent over-budget error to the AOT env-machine's existing observable
/// (RFC-0041 W3½): a [`BudgetError::DepthExceeded`] becomes the **unchanged** [`EvalError::DepthLimit`]
/// at the *same* limit, so the recursion/AOT differentials are byte-for-byte unmoved (the §5.1 canonical
/// variant unification is W5-gated — the AOT's externally-observed error variant is deliberately *not*
/// changed here). [`RecursionBudget::try_enter`] only ever yields `DepthExceeded` (it charges no
/// bytes/steps), so the `OutOfBudget` arm is unreachable in this crate; it is mapped defensively to a
/// `DepthLimit` on its own ceiling rather than panicking (never-silent, G2).
fn depth_limit_error(e: BudgetError) -> EvalError {
    let limit = match e {
        BudgetError::DepthExceeded { limit } => limit,
        BudgetError::OutOfBudget { limit, .. } => u32::try_from(limit).unwrap_or(u32::MAX),
    };
    EvalError::DepthLimit {
        limit: limit as usize,
    }
}

/// Enter an application `f arg` whose result should resume `ret`: push the right frame and return the
/// `(block, env)` to evaluate next. Closures call their body; a `Fix` unfolds under the fuel clock
/// (binding its name to itself) and re-applies. **The depth ceiling is the shared `mycelium-workstack`
/// budget (RFC-0041 W3½):** one [`RecursionBudget::try_enter`] per frame-push — the only place the
/// control stack grows on a call — with its [`DepthGuard`] moved into the pushed [`Frame`] for the
/// frame's lifetime, so the accept/reject threshold is exactly the pre-extraction `stack.len() >=
/// max_depth`. Applying a non-function is an explicit refusal.
///
/// **TCO (M-996 — the peek-then-commit at call entry, RFC-0041 §4.0/§4.6):** a closure applied with
/// a **passthrough** `ret` ([`Cont::is_tail_passthrough`]) is a genuine tail call — the peek is the
/// O(1) passthrough test (no mutation), the commit is *not pushing* the `Resume` frame (which
/// eagerly drops the caller's saved env). An elided call charges **no depth and no `alloc` bytes**
/// (§4.0: a tail iteration does not charge depth; no frame ⇒ no control-stack memory), so a tail
/// loop runs at O(1) depth and a tail call *at* the ceiling still succeeds. A `Fix`/`FixGroup`
/// unfold keeps its `ApplyThen` frame even in tail position — that frame does real post-work (apply
/// the unfolded closure) — but it is popped before the follow-up closure apply, so its charge is
/// transient and a tail `Fix` loop is **net-zero** on depth per iteration. Non-function operands
/// (`Repr`/`Data`) keep the pre-M-996 charge-then-refuse order byte-for-byte.
#[allow(clippy::too_many_arguments)] // the machine threads its three budgets + the TCO witness
fn enter_apply<'b>(
    f: AotVal,
    arg: AotVal,
    ret: Cont,
    stack: &mut Vec<Frame<'b>>,
    fuel: &mut u64,
    budget: &'b RecursionBudget,
    budgets: &mut Budgets,
    tco: &mut TcoTrace,
) -> Result<(Rc<Code>, Env), EvalError> {
    // TCO peek: only a Closure apply pushes a pure-`Resume(ret)` frame, so only there can eliding
    // the push be the identity. (`Fix`/`FixGroup` push `ApplyThen` — real post-work — and the
    // non-function arms below must keep their charge-then-refuse order.)
    if ret.is_tail_passthrough() && matches!(f, AotVal::Closure { .. }) {
        let AotVal::Closure { param, body, env } = f else {
            unreachable!("matched Closure in the tail-call peek above");
        };
        // Commit: elide the frame. No `try_enter` (no depth charge — §4.0), no `alloc` charge (no
        // frame allocated). Dropping `ret` here releases the caller's saved env eagerly — the
        // interpreter's drain-cleanup analog. NOTE (deliberate non-port of the interpreter's
        // `LetPop` Substrate escape check): the AOT fragment's values are `Repr`/`Data`/functions —
        // there is **no** Substrate-like affine value in `AotVal`, so an eager env drop is a plain
        // `Rc`/`Value` release with no release-on-drain obligation to run (stated, not cargo-culted).
        tco.record();
        drop(ret);
        let mut call_env = env;
        call_env.insert(param, arg);
        return Ok((body, call_env));
    }
    // The source-call/β frame charge: reserve one depth unit on the shared budget. The guard is moved
    // into the frame we push (released on pop), so `budget.current_depth() == stack.len()` at every
    // enter — hence `depth.get() >= depth_limit` (try_enter's refusal) is exactly the prior
    // `stack.len() >= max_depth`. Over-budget is the never-silent `DepthExceeded`, mapped to the
    // unchanged `EvalError::DepthLimit` (behavior-preserving; the depth check precedes the fuel/effect
    // charges, exactly as the ad-hoc ceiling did).
    let guard = budget.try_enter().map_err(depth_limit_error)?;
    // A declared `alloc` effect budget bounds the control-stack *memory* — charged per frame at the
    // DN-05 per-frame rate, the opt-in sibling of the depth ceiling (RFC-0014 §4.8). Absent ⇒ skip
    // (the depth ceiling is the default space guard). An overrun is the unified, graceful
    // `EvalError::EffectBudget` (`?` converts via `From<EffectBudgetExhausted>`) — never an OOM.
    if budgets.remaining(&EffectKind::Alloc).is_some() {
        budgets.consume(EffectKind::Alloc, DEFAULT_PER_FRAME_BYTES)?;
    }
    match f {
        AotVal::Closure { param, body, env } => {
            stack.push(Frame {
                _guard: guard,
                kind: FrameKind::Resume(ret),
            });
            let mut call_env = env;
            call_env.insert(param, arg);
            Ok((body, call_env))
        }
        AotVal::Fix { name, body, env } => {
            *fuel = fuel.checked_sub(1).ok_or(EvalError::FuelExhausted)?;
            stack.push(Frame {
                _guard: guard,
                kind: FrameKind::ApplyThen { arg, cont: ret },
            });
            // A captured env (built by `Env::snapshot`) always has an empty top segment, so this
            // clone — like the `FixGroup` clones below — is an O(1) `Rc` bump (M-999).
            let selfval = AotVal::Fix {
                name: Rc::clone(&name),
                body: Rc::clone(&body),
                env: env.clone(),
            };
            let mut unfold_env = env;
            unfold_env.insert(name, selfval);
            Ok((body, unfold_env))
        }
        AotVal::FixGroup { defs, which, env } => {
            *fuel = fuel.checked_sub(1).ok_or(EvalError::FuelExhausted)?;
            stack.push(Frame {
                _guard: guard,
                kind: FrameKind::ApplyThen { arg, cont: ret },
            });
            // Re-bind every member name to its own focus suspension (so a sibling call resolves the
            // whole group), then enter the focused member's body — mirrors the interpreter's
            // `FixGroup` focus unfold under the same fuel clock.
            let mut unfold_env = env.clone();
            for (member, key, _) in defs.iter() {
                unfold_env.insert(
                    Rc::clone(key),
                    AotVal::FixGroup {
                        defs: Rc::clone(&defs),
                        which: member.clone(),
                        env: env.clone(),
                    },
                );
            }
            let body = defs
                .iter()
                .find(|(n, _, _)| *n == which)
                // M-999: the member body is already a prepared shared block — an O(1) handle,
                // not a per-unfold deep clone of the member's subtree.
                .map(|(_, _, b)| Rc::clone(b))
                .ok_or(EvalError::FreeVariable(which))?;
            Ok((body, unfold_env))
        }
        AotVal::Repr(_) | AotVal::Data(_) => Err(EvalError::ApplyNonFunction),
    }
}

/// Select the first-matching arm (or default) of a lowered `Match`, returning the arm's block (as a
/// fresh [`Rc`]) and the environment to evaluate it in (constructor fields bound left-to-right). No
/// match + no default is an explicit [`EvalError::NonExhaustiveMatch`].
fn select_arm(
    scrut: &AotVal,
    alts: &[CodeAlt],
    default: Option<&Rc<Code>>,
    env: &mut Env,
) -> Result<(Rc<Code>, Env), EvalError> {
    for alt in alts {
        match alt {
            CodeAlt::Ctor {
                ctor,
                binders,
                body,
            } => {
                if let AotVal::Data(d) = scrut {
                    if &d.ctor == ctor {
                        if binders.len() != d.fields.len() {
                            return Err(EvalError::DataMalformed {
                                why: format!(
                                    "constructor arm binds {} of {} field(s) (WF6/WF7)",
                                    binders.len(),
                                    d.fields.len()
                                ),
                            });
                        }
                        // M-999: entering an arm captures the env by O(1) snapshot (formerly a
                        // whole-map clone per match). Binder pushes shadow in the arm env only —
                        // the caller's env (taken for the `Cont` after this returns) never sees
                        // them, exactly as the clone-then-insert did.
                        let mut arm_env = env.snapshot();
                        for (binder, field) in binders.iter().zip(d.fields.iter()) {
                            // The M-994 (b) win: binding a field is an O(1) `AotVal::clone` (a refcount
                            // bump on the shared sub-tree), not a deep copy out of a `Vec<CoreValue>`;
                            // the binder key is interned — an `Rc` bump, no `String` alloc (M-999).
                            arm_env.insert(Rc::clone(binder), field.clone());
                        }
                        // M-999: the arm body is a prepared shared block — an O(1) handle, not a
                        // per-match deep clone of the arm's subtree.
                        return Ok((Rc::clone(body), arm_env));
                    }
                }
            }
            CodeAlt::Lit { value, body } => {
                if let AotVal::Repr(rv) = scrut {
                    if rv.repr() == value.repr() && rv.payload() == value.payload() {
                        return Ok((Rc::clone(body), env.snapshot()));
                    }
                }
            }
        }
    }
    match default {
        Some(d) => Ok((Rc::clone(d), env.snapshot())),
        None => Err(EvalError::NonExhaustiveMatch),
    }
}

/// The result of evaluating one binding's RHS: bind a value and advance, or switch to a new block
/// (a call / match descent) whose continuation is already on the stack.
// `Bind` carries an inlined `AotVal` (see the note on `AotVal`) — same accepted size trade-off.
#[allow(clippy::large_enum_variant)]
enum Step {
    Bind(Rc<Atom>, AotVal),
    Switch(Rc<Code>, Env),
}

/// The trampoline: iterate over blocks with an explicit control stack, so object-level recursion
/// uses **heap**, not the host call stack (O(1) host stack — the M-347 fix). `App`/`Match` push a
/// continuation and switch blocks; a completed block returns its result value, unwinding the stack
/// (an `ApplyThen` frame re-applies). Deep recursion is bounded by `fuel` (time) and the shared
/// [`RecursionBudget`]'s depth ceiling (space) — both explicit graceful errors, never an abort. Each
/// pushed [`Frame`] holds a [`DepthGuard`] borrowed from `budget`, so `budget` must outlive `stack`
/// (RFC-0041 W3½). Tail-passthrough continuations are elided rather than pushed (TCO, M-996 — see
/// [`enter_apply`] and the `Match` arm below), each elision recorded in `tco` (house rule #2).
#[allow(clippy::too_many_arguments)] // the machine threads its three budgets + the TCO witness
fn eval_machine<'b>(
    top: Rc<Code>,
    top_env: Env,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    fuel: &mut u64,
    budget: &'b RecursionBudget,
    budgets: &mut Budgets,
    tco: &mut TcoTrace,
) -> Result<AotVal, EvalError> {
    let mut block = top;
    let mut env = top_env;
    let mut idx = 0usize;
    let mut stack: Vec<Frame<'b>> = Vec::new();

    loop {
        if idx >= block.bindings.len() {
            // Block complete: produce its result and resume the top control-stack frame.
            let val = lookup(&env, &block.result)?;
            match stack.pop() {
                None => return Ok(val),
                Some(Frame { _guard, kind }) => {
                    // Popping the frame releases its depth reservation NOW — mirroring the
                    // pre-extraction `stack.pop()` dropping `stack.len()` by one *before* any re-enter,
                    // so a following `enter_apply` sees the post-pop depth (the exact prior threshold
                    // order: an `ApplyThen`'s pop-then-re-push is net-zero on depth).
                    drop(_guard);
                    match kind {
                        FrameKind::Resume(c) => {
                            let mut e = c.env;
                            e.insert(c.name, val);
                            block = c.block;
                            env = e;
                            idx = c.idx;
                        }
                        FrameKind::ApplyThen { arg, cont } => {
                            // The returned value is the unfolded closure; apply it to the saved arg
                            // (its result flows to `cont`, the frame enter_apply pushes).
                            let (nb, ne) = enter_apply(
                                val, arg, cont, &mut stack, fuel, budget, budgets, tco,
                            )?;
                            block = nb;
                            env = ne;
                            idx = 0;
                        }
                    }
                }
            }
            continue;
        }

        // Evaluate binding `idx`. Compute an owned `Step` inside a scope that borrows `block`, so we
        // can reassign `block`/`env` afterwards without an outstanding borrow.
        let step: Step = {
            let binding = &block.bindings[idx];
            // Interned: re-binding a name is an `Rc` bump, never a `String` alloc (M-999).
            let name = Rc::clone(&binding.name);
            match &binding.rhs {
                CodeRhs::Const(v) => Step::Bind(name, AotVal::Repr(Rc::clone(v))),
                CodeRhs::Alias(a) => Step::Bind(name, lookup(&env, a)?),
                CodeRhs::Op { prim, args } => {
                    let vals: Vec<Rc<Value>> = args
                        .iter()
                        .map(|a| as_repr_value(lookup(&env, a)?))
                        .collect::<Result<_, _>>()?;
                    let refs: Vec<&Value> = vals.iter().map(|rc| &**rc).collect();
                    let f = prims
                        .get(prim)
                        .ok_or_else(|| EvalError::UnknownPrim(prim.clone()))?;
                    Step::Bind(name, AotVal::Repr(Rc::new(f(prim, &refs)?)))
                }
                CodeRhs::Swap {
                    src,
                    target,
                    policy,
                } => {
                    let s = as_repr_value(lookup(&env, src)?)?;
                    Step::Bind(name, AotVal::Repr(Rc::new(swap.swap(&s, target, policy)?)))
                }
                CodeRhs::Construct { ctor, args } => {
                    // Build the datum from the *shared* `AotVal` fields directly — each looked-up field
                    // is an O(1) clone, and `Rc`-wrapping the field vector is the whole per-node cost
                    // (no `Vec<CoreValue>` materialisation). A function-valued field is rejected here,
                    // exactly as the former `as_core` conversion did (`FunctionResult`), preserving the
                    // error and its timing. The carried `guarantee` is the meet-summary `Datum::new`
                    // would compute over the same fields (byte-identical when later materialised).
                    let mut fields: Vec<AotVal> = Vec::with_capacity(args.len());
                    for a in args {
                        let fv = lookup(&env, a)?;
                        if matches!(
                            fv,
                            AotVal::Closure { .. } | AotVal::Fix { .. } | AotVal::FixGroup { .. }
                        ) {
                            return Err(EvalError::FunctionResult);
                        }
                        fields.push(fv);
                    }
                    let guarantee =
                        GuaranteeStrength::meet_all(fields.iter().map(AotVal::guarantee));
                    Step::Bind(
                        name,
                        AotVal::Data(Rc::new(AotDatum {
                            ctor: ctor.clone(),
                            fields,
                            guarantee,
                        })),
                    )
                }
                // M-999: closure/suspension capture is an O(1) `Env::snapshot` (freeze + share),
                // formerly a whole-`HashMap` clone per capture, and the body is an O(1) handle
                // into the prepared program, formerly a per-execution deep clone of the subtree —
                // together the measured constant-factor gap to the interpreter
                // (`tests/aot_vs_interp_bench.rs`).
                CodeRhs::Lam { param, body } => Step::Bind(
                    name,
                    AotVal::Closure {
                        param: Rc::clone(param),
                        body: Rc::clone(body),
                        env: env.snapshot(),
                    },
                ),
                CodeRhs::Fix { name: fname, body } => Step::Bind(
                    name,
                    AotVal::Fix {
                        name: Rc::clone(fname),
                        body: Rc::clone(body),
                        env: env.snapshot(),
                    },
                ),
                CodeRhs::FixGroup { defs, which } => Step::Bind(
                    name,
                    AotVal::FixGroup {
                        defs: Rc::clone(defs),
                        which: which.clone(),
                        env: env.snapshot(),
                    },
                ),
                CodeRhs::App { func, arg } => {
                    let f = lookup(&env, func)?;
                    let a = lookup(&env, arg)?;
                    let ret = Cont {
                        block: Rc::clone(&block),
                        idx: idx + 1,
                        env: std::mem::take(&mut env),
                        name,
                    };
                    let (nb, ne) = enter_apply(f, a, ret, &mut stack, fuel, budget, budgets, tco)?;
                    Step::Switch(nb, ne)
                }
                CodeRhs::Match {
                    scrutinee,
                    alts,
                    default,
                } => {
                    // Match directly on the *shared* scrutinee `AotVal` — no `Datum` materialisation
                    // (which would deep-copy the spine on every match, the (b) cost we removed). A
                    // function scrutinee is the explicit `FunctionResult` (mirrors the former `as_core`
                    // coercion, preserving that error and its ordering before the guarantee check).
                    let scrut = lookup(&env, scrutinee)?;
                    if matches!(
                        scrut,
                        AotVal::Closure { .. } | AotVal::Fix { .. } | AotVal::FixGroup { .. }
                    ) {
                        return Err(EvalError::FunctionResult);
                    }
                    // r3 boundary (RFC-0011 §4.6): the guarantee-meet through Match is the identity
                    // only when the scrutinee is Exact; a non-Exact scrutinee is the explicit deferral
                    // (never a fabricated bound) — mirrors the reference interpreter.
                    let g = scrut.guarantee();
                    if g != GuaranteeStrength::Exact {
                        return Err(EvalError::GuaranteeMeetUnsupported { scrutinee: g });
                    }
                    let (arm_block, arm_env) =
                        select_arm(&scrut, alts, default.as_ref(), &mut env)?;
                    let cont = Cont {
                        block: Rc::clone(&block),
                        idx: idx + 1,
                        env: std::mem::take(&mut env),
                        name,
                    };
                    if cont.is_tail_passthrough() {
                        // TCO (M-996): a tail-position `Match` — the arm's value IS the enclosing
                        // block's result, so the `Resume` frame would be a pure passthrough. Elide
                        // it: no frame, no depth charge (§4.0 — the ANF analog of the interpreter's
                        // tail-transparent `MatchPop`; the passthrough test is checked on BOTH
                        // settle paths — see `is_tail_passthrough`). Dropping `cont` releases the
                        // caller env eagerly. `select_arm` still ran first, so `NonExhaustiveMatch`
                        // and the guarantee-meet refusal surface exactly as before (order preserved).
                        tco.record();
                        drop(cont);
                    } else {
                        // `Match` grows the control stack by one continuation frame — charge the
                        // shared budget here too (the pre-extraction ceiling guarded this site
                        // identically), after `select_arm` so a `NonExhaustiveMatch` still surfaces
                        // first (order preserved).
                        let guard = budget.try_enter().map_err(depth_limit_error)?;
                        stack.push(Frame {
                            _guard: guard,
                            kind: FrameKind::Resume(cont),
                        });
                    }
                    Step::Switch(arm_block, arm_env)
                }
            }
        };

        match step {
            Step::Bind(name, v) => {
                env.insert(name, v);
                idx += 1;
            }
            Step::Switch(nb, ne) => {
                block = nb;
                env = ne;
                idx = 0;
            }
        }
    }
}

/// Run a Core IR program through the AOT path **with a schedule-staged packing layout** (M-251;
/// RFC-0004 §5/§8). The result is first computed by [`run`], then — for a ternary result — its
/// trits are materialized into a physical buffer **packed under `packed_as`** and **read back under
/// the recorded tag `read_as`** (the `Meta.physical` claim), and the layout is recorded on the
/// result's `Meta` (M-I5 lossless, [`Value`]'s `with_physical`).
///
/// When the tag is correct (`packed_as == read_as`) the read-back is the identity, so the result is
/// observably equal to the layout-agnostic reference (the interpreter / [`run`]) — and the M-210
/// observational-equivalence check validates. A **mislabeled** tag (`packed_as != read_as`)
/// misreads the buffer, producing a different payload that the same check rejects (NFR-7) — the E3
/// soundness property: the layout record is trusted *only because a wrong one is caught*.
///
/// Non-ternary results carry no trit-packing layout, so they pass through unchanged.
pub fn run_with_layout(
    node: &Node,
    prims: &PrimRegistry,
    swap: &dyn SwapEngine,
    packed_as: PackScheme,
    read_as: PackScheme,
) -> Result<Value, EvalError> {
    let v = run(node, prims, swap)?;
    match (v.repr(), v.payload()) {
        (Repr::Ternary { .. }, Payload::Trits(trits)) => {
            let read = pack::relayout_trits(trits, packed_as, read_as);
            let meta = v
                .meta()
                .clone()
                .with_physical(PhysicalLayout::TritPacked { scheme: read_as });
            Value::new(v.repr().clone(), Payload::Trits(read), meta)
                .map_err(|e| EvalError::Swap(e.to_string()))
        }
        _ => Ok(v),
    }
}
