//! The **L1 fuel-guarded evaluator** (RFC-0007 §4.6): a big-step environment machine mirroring
//! the M-110 reference interpreter's contract — explicit errors only, a step budget instead of a
//! termination assumption (CakeML-style clocked semantics, T3.4), and the *same* trusted
//! primitive registry and swap engine the L0 interpreter and the AOT path dispatch through, so
//! "two execution paths" can never mean "two semantics" (NFR-7).
//!
//! Programs **inside** the evaluation-complete fragment also elaborate to L0
//! ([`crate::elab::elaborate`]) and must agree with this evaluator on the observable
//! (`repr + payload + guarantee`) — the §4.6 differential obligation, validated through the M-210
//! shared checker (`tests/differential.rs`). Programs **outside** the fragment (recursion, match,
//! data values, dynamic guarantee indices) run *only* here.
//!
//! Honesty:
//! - exhausting the step budget is an explicit [`L1Error::FuelExhausted`], never a hang — and
//!   "checked total" means precisely "terminates for every sufficiently large fuel" (§4.5);
//! - a guarantee index `@ g` is checked **dynamically against `Meta`** (stage 0, RFC-0007 §4.3):
//!   asserting `@ g` on a value whose tag is weaker than `g` is an explicit
//!   [`L1Error::GuaranteeTooWeak`] — the assertion never upgrades the tag (VR-5), and a passing
//!   check leaves the value's own (possibly stronger) tag untouched;
//! - states the typechecker proves unreachable still fail as explicit [`L1Error::Stuck`] errors,
//!   never panics or defaults (S5/G2);
//! - [`Evaluator::call`] runs the recursive evaluation on a deep worker stack (256 MiB, lazily
//!   committed) via [`mycelium_stack::with_deep_stack`], so the **explicit depth budget** — not
//!   the caller's thread stack — is always what bounds a pathological input. Raising
//!   [`DEFAULT_DEPTH`] via [`Evaluator::with_depth`] is now host-stack-safe: the budget refuses
//!   cleanly well before the physical stack limit (banked guard 4; see `DEFAULT_DEPTH`). The
//!   worker stack is the transitional Rust-host adapter; the explicit budget is the portable
//!   primitive that will carry to the self-hosted Mycelium frontend (RFC-0007 §4.5/§4.6).

use std::collections::{BTreeMap, VecDeque};
use std::mem;
use std::sync::Arc;

use mycelium_cert::BinaryTernarySwapEngine;
use mycelium_core::{CoreValue, DataRegistry, Datum, GuaranteeStrength, Value};
use mycelium_interp::{
    Budgets, EffectBudget, EffectBudgetExhausted, EvalError as KernelError, PrimRegistry,
    SwapEngine,
};
// RFC-0041 W5 (M-979): the shared never-silent recursion budget the CEK work-stack charges at each
// source-call/β boundary (§4.0 metric). The canonical `BudgetError::DepthExceeded` maps to
// [`L1Error::DepthExceeded`] at the same threshold (§5.1); `DepthGuard` is the RAII depth
// reservation held for the lifetime of a live source-call frame on the explicit work-stack.
use mycelium_workstack::{BudgetError, DepthGuard, RecursionBudget};
// M-906 (DN-70 D1; RFC-0008 RT3): `@forage`'s D-lite placement decision reuses the existing
// RFC-0005 `SelectionPolicy` machinery verbatim — no new mechanism (DN-70).
use mycelium_select::{
    select_placement, Candidate, CostModel, NodeRef, SelectionInputs, SelectionPolicy,
};

use crate::ast::{Expr, Hypha, Literal, Pattern, Strength};
use crate::checkty::{prim_kernel_name, Env};
use crate::elab::{lit_value, policy_name_ref, type_repr, ElabError};

/// An L1 runtime value: an L0 representation value, or a constructed datum. Data values are
/// immutable and acyclic by construction — a `Construct` value can only contain values that
/// existed before it (RFC-0007 §4.7).
///
/// **`Arc` structural sharing for O(1) clone (M-994 fix (b), M-987).** A `Data` node's `fields` are
/// held behind an [`Arc`], so cloning a `Data` value is a **refcount bump**, not an O(nodes) spine
/// rebuild. This is sound precisely because `Data` is *immutable and acyclic by construction* (the
/// invariant above): no code path mutates a live `Data`'s fields in place, so sharing is
/// observationally identical to copying. `Clone` is therefore **derived** — `Arc::clone` is O(1) and
/// `String` clones are bounded by the (short) type/ctor names, so a whole-value clone is O(1) in the
/// node count. This removes the factor of value-size that made `eval_path`'s per-reference `v.clone()`
/// the ~n^3 L1-eval cost (M-987); a variable reference is now O(1).
///
/// **Iterative destruction (RFC-0041 §4.5, M-979).** `Drop` is still **hand-written and iterative**.
/// A derived recursive `Drop` walks the `Data.fields` spine on the *host* stack, so a deep
/// (uniquely-owned) `Cons`/`Nat` value — the exact shape the depth budget now permits up to 4096 deep
/// — would `SIGABRT` on destruction (RFC-0041 §1's recursive-`Drop` stack bomb). It walks an explicit
/// heap worklist instead, so control recursion is O(1) host stack for any value depth. With `Arc`
/// sharing the deep dismantle only happens when a node is the *last* owner ([`Arc::get_mut`] succeeds);
/// a still-shared subtree drops in O(1) (a refcount decrement), never wrongly torn down. `PartialEq`
/// stays derived — no code path compares deep `L1Value`s, which go through [`L1Value::to_core`] for
/// the differential; a deep-compare iterativisation is a tracked §4.5 residual.
#[derive(Clone, Debug, PartialEq)]
pub enum L1Value {
    /// An L0 value (`repr + payload + Meta`).
    Repr(Value),
    /// A saturated constructor application (W6).
    Data {
        /// The data type's name (v0 keys the registry by name; RFC-0007 §4.2).
        ty: String,
        /// The constructor's name.
        ctor: String,
        /// The constructor's field values, in declaration order. Held behind an [`Arc`] so cloning a
        /// `Data` shares the spine (O(1)) rather than deep-copying it (M-994 fix (b)) — sound because
        /// `Data` is immutable + acyclic (never mutated in place).
        fields: Arc<Vec<L1Value>>,
    },
    /// An affine `Substrate` handle (DN-71 Model S §4.1; M-902) — an opaque, runtime-only
    /// external-resource handle ([`crate::substrate::SubstrateHandle`]). It is **not** a repr value
    /// and **not** algebraic data: it names an external resource (RFC-0006 LR-8), carries no
    /// `Repr`/`Meta`, and never lowers to L0 (no kernel node — KC-3). It lives at this evaluator
    /// level only, is *passed* by the ordinary value-binding machinery, and is *inspected* via its
    /// [`SubstrateHandle`](crate::substrate::SubstrateHandle) accessors. The affine use-once
    /// enforcement is M-903 (a static checker pass plus a runtime backstop) and the `consume`
    /// lowering — M-904, DN-71 §4.3 — now **executes** the checked move (never a silent move —
    /// G2/VR-5); a live, un-`consume`d handle is deterministically released at scope exit and the
    /// release recorded (M-904, DN-71 §8 FLAG-4's v0 posture — never a silent leak).
    Substrate(crate::substrate::SubstrateHandle),
    /// A function-typed value: a reference to an ordinary top-level function, by name (ADR-033/
    /// DN-74, M-923 — the `FieldSpec::Fn` dictionary-dispatch payload). This is deliberately **not**
    /// a general closure: ADR-033 §2.1 identifies the function by its own top-level identity (no
    /// captured environment — a dictionary field is a content-addressed term reference), so a name
    /// is the whole value. Produced when a bare top-level function is referenced in value position
    /// (`crate::elab`'s mirror-image arm); applied by looking the name up in `env.fns` and invoking
    /// it (`eval_app`'s new dispatch-through-a-bound-value case). Never constructed from a `lambda`
    /// literal or a captured value — that remains `mono.rs`'s closure-defunctionalization territory
    /// (RFC-0024 §4A/M-704); this variant is reachable only when `Evaluator` runs directly on a
    /// checked-but-**not**-monomorphized `Env` (mirroring [`crate::elab::elaborate_direct`]).
    Fn(String),
}

/// **Iterative `Drop` (RFC-0041 §4.5; M-994 fix (b)).** A derived recursive `Drop` walks
/// `Data.fields` on the host stack — a deep value SIGABRTs *on destruction* (RFC-0041 §1's
/// recursive-`Drop` stack bomb). This dismantles the value over an explicit heap worklist: each
/// uniquely-owned `Data`'s fields are moved out (leaving an empty `Vec`) before the node itself
/// drops, so no node's `Drop` ever recurses.
///
/// **`Arc` sharing (M-994 fix (b)).** `Data.fields` is now `Arc<Vec<L1Value>>`, so a node's spine can
/// be shared. The dismantle only descends into a node whose `Arc` we are the **last** owner of
/// ([`Arc::get_mut`] returns `Some` iff `strong_count == 1` and there are no weaks — and this
/// evaluator never creates a `Weak`): a still-shared subtree is left intact and drops in O(1) (a
/// refcount decrement) when its owner goes. This keeps the O(1)-host-stack SIGABRT-safety for any
/// depth *and* respects sharing — a shared tail is never wrongly torn down.
///
/// **Allocation (honest scope).** The worklist buffer is seeded by *stealing* the (uniquely-owned)
/// root's existing `fields` `Vec` (`mem::take` through `Arc::get_mut` — no new allocation for the
/// buffer itself); it may grow while draining a wide subtree. A non-`Data`, an empty `Data`, or a
/// still-shared `Data` allocates nothing and returns immediately.
impl Drop for L1Value {
    fn drop(&mut self) {
        // Only a `Data` node we uniquely own can hold a deep spine to dismantle. If shared, the
        // refcount decrement below is all that is needed (the spine stays alive under other owners).
        let L1Value::Data { fields, .. } = self else {
            return;
        };
        // The iterative dismantle relies on `Arc::get_mut` succeeding for a *uniquely-owned* node.
        // A `Weak` ref would make `get_mut` return `None` even at `strong_count == 1`, silently
        // skipping the dismantle so this subtree drops via recursive drop-glue — quietly reopening
        // the deep-spine SIGABRT hole RFC-0041 §6 closed. No code creates a `Weak` against `L1Value`
        // (checked); this makes that safety invariant *checked* in debug rather than a silent
        // convention (PR #1190 review, MEDIUM — defense-in-depth).
        debug_assert_eq!(
            Arc::weak_count(fields),
            0,
            "L1Value::Data.fields must have no Weak refs, else iterative Drop degrades to recursive \
             drop-glue (deep-spine SIGABRT hazard; RFC-0041 §6)"
        );
        let Some(root_fields) = Arc::get_mut(fields) else {
            return; // shared — do not dismantle; the `Arc` drop is O(1).
        };
        if root_fields.is_empty() {
            return;
        }
        // Steal the root's fields into the worklist (reusing the existing allocation).
        let mut stack: Vec<L1Value> = mem::take(root_fields);
        while let Some(mut v) = stack.pop() {
            if let L1Value::Data { fields, .. } = &mut v {
                // Move this (uniquely-owned) node's children onto the worklist, so `v` drops with an
                // empty spine (shallow — no recursion). A shared child is left for its owner.
                if let Some(child_fields) = Arc::get_mut(fields) {
                    stack.append(child_fields);
                }
            }
            // `v` drops here: its own `Drop` re-enters, but its `fields` are now empty (or shared, so
            // O(1)), so it returns immediately — no host-stack recursion.
        }
    }
}

impl L1Value {
    /// The underlying L0 value, if this is a representation value; `None` for data or a `Substrate`
    /// handle (never-silent — neither has a repr value here, G2).
    #[must_use]
    pub fn as_repr(&self) -> Option<&Value> {
        match self {
            L1Value::Repr(v) => Some(v),
            L1Value::Data { .. } | L1Value::Substrate(_) | L1Value::Fn(_) => None,
        }
    }

    /// The affine [`SubstrateHandle`](crate::substrate::SubstrateHandle), if this is a `Substrate`
    /// value; `None` otherwise (never-silent — a non-Substrate has no handle here, G2). The
    /// inspection window onto the opaque handle (its tag, opaque identity, and acquisition
    /// provenance — DN-71 §4.1; M-902).
    #[must_use]
    pub fn as_substrate(&self) -> Option<&crate::substrate::SubstrateHandle> {
        match self {
            L1Value::Substrate(h) => Some(h),
            L1Value::Repr(_) | L1Value::Data { .. } | L1Value::Fn(_) => None,
        }
    }

    /// Project this L1 value onto the L0 [`CoreValue`] domain, resolving each constructor's
    /// name-keyed identity (`ty`/`ctor`) to its content-addressed `#T#i` [`mycelium_core::CtorRef`]
    /// through `registry` — the **same** registry the elaborator built (RFC-0011 §4.3). This is the
    /// bridge that makes the M-210 differential meaningful on the data fragment: an L1-eval result
    /// and an elaborate→L0-interp result become comparable *as the same L0 value* (NFR-7). The data
    /// guarantee is the meet-summary [`Datum::new`] computes from the fields, identical on both
    /// paths. Returns `None` if a constructor is not in the registry (outside the r3 fragment).
    #[must_use]
    pub fn to_core(&self, env: &crate::checkty::Env, registry: &DataRegistry) -> Option<CoreValue> {
        match self {
            L1Value::Repr(v) => Some(CoreValue::Repr(v.clone())),
            L1Value::Data { ty, ctor, fields } => {
                let decl = env.types.get(ty)?;
                let index = decl.ctors.iter().position(|c| c.name == *ctor)?;
                let ctor_ref = registry.ctor_ref(ty, u32::try_from(index).ok()?)?;
                let core_fields = fields
                    .iter()
                    .map(|f| f.to_core(env, registry))
                    .collect::<Option<Vec<_>>>()?;
                Some(CoreValue::Data(Datum::new(ctor_ref, core_fields)))
            }
            // A `Substrate` handle has **no** L0 projection — it is not a kernel value (no `Repr`,
            // no L0 node; DN-71 §4.1). It never participates in the L0/AOT differential, so `None`
            // here is the honest "no core value", never a fabricated lowering (G2). M-904 keeps this
            // property: `consume` lowers through existing paths, and `Substrate` itself stays absent
            // from the L0 value world.
            L1Value::Substrate(_) => None,
            // ADR-033/DN-74 (M-923): a bare function value has no [`CoreValue`] counterpart either
            // — `CoreValue` is `Repr | Data` only (a *fully-applied* result), never an unapplied
            // function. The differential this leaf lands compares `main`'s fully-dispatched
            // (applied) result, never the raw dictionary/function value itself, so this arm is
            // never exercised by that harness; it exists so this match stays exhaustive rather than
            // silently panicking if some future caller ever does `to_core` a bare `Fn` (G2).
            L1Value::Fn(_) => None,
        }
    }
}

/// Whether `v` transitively contains a `Substrate` handle with the given `id` — the M-904 (DN-71
/// §8 FLAG-4) scope-exit-release **escape check**: a handle still reachable from a scope's own
/// result (directly, or nested inside a constructed `Data` value) must never be released, even if
/// it was never explicitly `consume`d. `L1Value` is finite and acyclic by construction (`Data`'s
/// own doc comment — every field existed before its containing value), so this recursion always
/// terminates and is a **precise** (not merely approximating) check for everything the v0 evaluator
/// can construct — never a false negative that would let a live, still-reachable handle be wrongly
/// released (G2).
fn value_contains_substrate_id(v: &L1Value, id: u64) -> bool {
    match v {
        L1Value::Substrate(h) => h.id() == id,
        L1Value::Data { fields, .. } => fields.iter().any(|f| value_contains_substrate_id(f, id)),
        // A function value names a top-level definition, not a runtime handle — it can never carry
        // a `Substrate` (ADR-033 §2.1: no captured environment), so it never contributes an escape.
        L1Value::Repr(_) | L1Value::Fn(_) => false,
    }
}

/// Whether the `popped` scope binding is a live `Substrate` handle that **escapes** into any of the
/// `into` values — the multi-value companion to the single-value `escaping` check inside
/// [`Evaluator::release_if_abandoned`]. Used at the `LetPop` scope-exit in `enter_call`, where a
/// let-bound handle can only reach the callee through the call's arguments (`argv`): if it escapes,
/// the scope-exit release is suppressed (the callee now owns it). Factored out of the previously
/// inlined `matches!` so the escape test lives alongside `value_contains_substrate_id` rather than
/// being duplicated (DRY; PR #1189 nit).
fn substrate_escapes_into(popped: &(String, L1Value), into: &[L1Value]) -> bool {
    matches!(&popped.1, L1Value::Substrate(h)
        if into.iter().any(|a| value_contains_substrate_id(a, h.id())))
}

/// One recorded `@forage(policy)` placement decision (M-906; DN-70 D1; RFC-0008 RT3) — the
/// mandatory RFC-0005 §2.2 EXPLAIN record, `site`-tagged (mirrors
/// [`crate::substrate::ReleaseEvent`]'s `site` field — no fabricated line/column; this evaluator
/// has no source spans, VR-5) so a caller can attribute each decision to the function it occurred
/// in. `explanation` is [`mycelium_select::Explanation`] **verbatim** — the real RFC-0005
/// mechanism's own record, not a reimplementation (DN-70 D1: "no new mechanism").
#[derive(Debug, Clone, PartialEq)]
pub struct ForageDecision {
    /// The function in which the `@forage`-annotated hypha spawned.
    pub site: String,
    /// The `mycelium-select` EXPLAIN record for this decision.
    pub explanation: mycelium_select::Explanation,
}

/// Why a `@forage(policy)` placement failed — always explicit, never a silent hang or a fabricated
/// placement (RT4/G2; DN-63 §3.5 FLAG-14; M-906/DN-70 D1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForageError {
    /// The D-lite worker-availability bitmask has no set bits — the (degenerate, single-node)
    /// candidate set is empty. DN-63 §3.5 FLAG-14's required shape, implemented exactly: a typed,
    /// explicit error, never a silent hang. `elaborate`/`elaborate_colony` refuse the identical
    /// source with an explicit [`ElabError::Residual`] (see `crate::elab::forage_reject_if_empty`)
    /// — never-silent on every path (L1-eval and elaborate→{L0-interp, AOT} agree: none of them
    /// silently accepts a no-candidate forage).
    NoCandidates,
}

impl core::fmt::Display for ForageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ForageError::NoCandidates => write!(
                f,
                "`@forage` has no placement candidates (DN-63 §3.5 FLAG-14): the worker- \
                 availability bitmask is all-zero — an explicit refusal, never a silent placement"
            ),
        }
    }
}

impl std::error::Error for ForageError {}

/// Why L1 evaluation could not produce a value — always explicit (S5/G2).
#[derive(Debug, Clone, PartialEq)]
pub enum L1Error {
    /// The step budget ran out — the non-termination guard (RFC-0007 §4.5/§4.6).
    FuelExhausted,
    /// The recursion-depth budget ran out. This is the **explicit semantic ceiling** (banked guard
    /// 4; see [`DEFAULT_DEPTH`]): the evaluator recurses on the deep worker stack
    /// ([`mycelium_stack`]), so the budget — not a host-stack overflow — is always what stops a
    /// pathological input. Raise with [`Evaluator::with_depth`]; the host stack will not be the
    /// limit.
    DepthExceeded {
        /// The configured depth budget.
        limit: u32,
    },
    /// The trusted kernel (prim registry / swap engine) refused — the refusal is forwarded
    /// verbatim, never softened.
    Kernel(KernelError),
    /// A dynamic guarantee-index check failed: the asserted `@ g` is *stronger* than the value's
    /// actual tag — an assertion may only weaken, never upgrade (VR-5; RFC-0007 §4.3).
    GuaranteeTooWeak {
        /// The function in which the assertion appears.
        site: String,
        /// The asserted strength.
        asserted: Strength,
        /// The value's actual (weaker) strength.
        actual: GuaranteeStrength,
    },
    /// A construct the v0 evaluator does not support (`wild`, `spore`, bare-integer/list
    /// literals…) — refused with its reason, mirroring the typechecker's refusals.
    Unsupported {
        /// The function in which the construct appears.
        site: String,
        /// What was refused, and why.
        what: String,
    },
    /// An evaluation state the typechecker proves unreachable (unknown name, non-exhaustive
    /// match, arity mismatch…). Reported explicitly rather than panicking, so a checker bug can
    /// never become silent misbehavior.
    Stuck {
        /// The function in which evaluation got stuck.
        site: String,
        /// What went wrong.
        why: String,
    },
    /// A declared per-effect budget was exceeded (RFC-0014 §4.5 I4; M-677). The effect analogue
    /// of [`L1Error::FuelExhausted`]: graceful, explicit, never a hang or OOM. The budget is
    /// primed from `FnSig::effect_budgets` at the call site and consumed once per declared effect
    /// per invocation.
    EffectBudget(EffectBudgetExhausted),
    /// A `@forage(policy)` placement decision refused (M-906; DN-70 D1; RFC-0008 RT3) — see
    /// [`ForageError`].
    Forage(ForageError),
}

impl core::fmt::Display for L1Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            L1Error::FuelExhausted => write!(f, "evaluation exceeded its step budget"),
            L1Error::DepthExceeded { limit } => write!(
                f,
                "evaluation exceeded its recursion-depth budget ({limit}) — explicit by design \
                 (raise with `Evaluator::with_depth`; the host stack is not the limit)"
            ),
            L1Error::Kernel(e) => write!(f, "kernel refusal: {e}"),
            L1Error::GuaranteeTooWeak {
                site,
                asserted,
                actual,
            } => write!(
                f,
                "in `{site}`: asserted `@ {asserted:?}` but the value's tag is {actual:?} — an \
                 annotation may only weaken (VR-5)"
            ),
            L1Error::Unsupported { site, what } => write!(f, "in `{site}`: {what}"),
            L1Error::Stuck { site, why } => write!(
                f,
                "in `{site}`: stuck — {why} (the typechecker should have refused this program)"
            ),
            L1Error::EffectBudget(e) => write!(f, "{e}"),
            L1Error::Forage(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for L1Error {}

impl From<KernelError> for L1Error {
    fn from(e: KernelError) -> Self {
        L1Error::Kernel(e)
    }
}

impl From<EffectBudgetExhausted> for L1Error {
    fn from(e: EffectBudgetExhausted) -> Self {
        L1Error::EffectBudget(e)
    }
}

impl From<ForageError> for L1Error {
    fn from(e: ForageError) -> Self {
        L1Error::Forage(e)
    }
}

/// The surface strength keyword's kernel lattice point.
#[must_use]
pub fn strength_of(s: Strength) -> GuaranteeStrength {
    match s {
        Strength::Exact => GuaranteeStrength::Exact,
        Strength::Proven => GuaranteeStrength::Proven,
        Strength::Empirical => GuaranteeStrength::Empirical,
        Strength::Declared => GuaranteeStrength::Declared,
    }
}

/// Default step budget — mirrors the reference interpreter's (M-110).
const DEFAULT_FUEL: u64 = 1_000_000;

/// Default recursion-depth budget — conservative enough for an unoptimized (debug) build.
///
/// [`Evaluator::call`] runs the recursive evaluation on a deep worker stack (256 MiB, lazily
/// committed, via [`mycelium_stack::with_deep_stack`]), so this budget is the **always-binding
/// semantic ceiling** (banked guard 4) — not a stand-in for the host stack. Deep but terminating
/// programs can safely raise it via [`Evaluator::with_depth`]; the host stack will not be the
/// limit. Default is 64 — conservative by design and unchanged. A raised budget refuses cleanly
/// once it trips; the worker stack is the transitional Rust-host adapter (see
/// [`mycelium_stack`]) and is expected to disappear when the frontend self-hosts (the budget
/// carries to the Mycelium-native clocked-computation model; RFC-0007 §4.5/§4.6).
///
/// **Grounding (measured, not guessed).** The 256 MiB worker stack is the same one the checker
/// and elaborator use. The evaluator's `eval` frame is smaller than the checker's (~10.9 KiB):
/// it carries a `u64` fuel counter, a `u32` depth counter, a `&str` site, a `&mut Vec<…>` scope
/// pointer, and a `&Expr` — roughly 2–4 KiB in a debug build. At ~4 KiB/frame the 256 MiB
/// stack supports **~65,000** levels physically; at ~2 KiB/frame **~130,000**. The default
/// budget (64) is therefore a **~1,000× safety margin** below the physical ceiling, and raising
/// it to 4,096 (matching the checker) is safe with ample headroom. An in-process measurement
/// of the *clean-DepthExceeded* property is the regression guard; the physical ceiling estimate
/// is `Empirical` (frame size varies with the Rust optimizer and the IR structure).
///
/// **RFC-0041 W5 (M-979): raised 64 → 4096, and the metric changed.** The evaluator is now an
/// explicit heap **work-stack CEK machine** ([`Evaluator::run_machine`]), so control recursion is
/// O(1) host stack for *any* depth — the depth budget is no longer a host-stack guard but the
/// **semantic** source-call/β ceiling on the §4.0 metric (charged once per `Expr::App` boundary via
/// the shared [`mycelium_workstack::RecursionBudget`], **not** per AST node). The default is the
/// workspace floor [`RecursionBudget::DEFAULT_DEPTH_LIMIT`] (4096), reconciling the L1 path with the
/// interp/AOT paths at one threshold + one canonical variant (§5.1). Tail calls without pending
/// post-work reuse their frame (TCO, §4.6), so a tail-recursive loop runs in bounded depth.
pub(crate) const DEFAULT_DEPTH: u32 = RecursionBudget::DEFAULT_DEPTH_LIMIT;

/// The tunable **budgets** of an [`Evaluator`] — the step (`fuel`) and recursion-depth guards — as
/// a single options struct, an alternative to threading the fluent [`Evaluator::with_fuel`] /
/// [`Evaluator::with_depth`] chain. Applied via [`Evaluator::with_opts`]; the fluent setters stay.
///
/// Only the `Copy` budget knobs live here: the *engines* (`PrimRegistry`, `Box<dyn SwapEngine>`)
/// are not part of `EvaluatorOpts` — they are not `Clone`/`Default` and stay set through
/// [`Evaluator::with_engines`], so this struct is a plain, copyable, defaultable bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluatorOpts {
    /// The step budget (as [`Evaluator::with_fuel`]). [`Default`] is `DEFAULT_FUEL`.
    pub fuel: u64,
    /// The recursion-depth budget (as [`Evaluator::with_depth`]). [`Default`] is `DEFAULT_DEPTH`.
    /// Evaluation runs on the deep worker stack ([`mycelium_stack`]), so a raised budget is
    /// host-stack-safe — the budget, not the host stack, is the ceiling.
    pub depth: u32,
}

/// The defaults mirror [`Evaluator::new`] exactly — `DEFAULT_FUEL` / `DEFAULT_DEPTH` — so
/// `Evaluator::new(env).with_opts(EvaluatorOpts::default())` is a no-op (the budgets are unchanged).
impl Default for EvaluatorOpts {
    fn default() -> Self {
        EvaluatorOpts {
            fuel: DEFAULT_FUEL,
            depth: DEFAULT_DEPTH,
        }
    }
}

impl EvaluatorOpts {
    /// Set the step budget (builder-style), leaving `depth` untouched.
    #[must_use]
    pub fn fuel(mut self, fuel: u64) -> Self {
        self.fuel = fuel;
        self
    }

    /// Set the recursion-depth budget (builder-style), leaving `fuel` untouched.
    #[must_use]
    pub fn depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }
}

/// One elided tail-call frame recorded by the TCO EXPLAIN trail (RFC-0041 §4.6 tco32) — the
/// per-callee identity + the running iteration count at the moment the frame was reused. Not a bare
/// count (house rule #2 in substance, not just letter): a deep tail chain that later errors yields
/// an actionable "who was looping, how many times" trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcoElision {
    /// The callee whose invoke frame was reused (the tail-called function's name).
    pub callee: String,
    /// The 1-based count of consecutive elisions for this callee at the time of this record.
    pub iteration: u64,
}

/// The bounded EXPLAIN record of tail-call optimization (RFC-0041 §4.6 tco32): total elided frames,
/// a per-callee iteration tally, and a **ring buffer of the last [`TcoTrace::RING_CAP`] elisions** —
/// so an over-budget error at the end of a deep tail chain is diagnosable (which callee, how deep)
/// without retaining an unbounded trace. Inspectable via [`Evaluator::tco_trace`] (never a black
/// box — house rule #2).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TcoTrace {
    /// The total number of tail-call frames elided across every [`Evaluator::call`] so far.
    pub total_elided: u64,
    /// Per-callee elision counts (identity + iteration count — §4.6 tco32).
    pub per_callee: BTreeMap<String, u64>,
    /// The last [`Self::RING_CAP`] elisions, newest at the back (the bounded ring buffer).
    pub recent: VecDeque<TcoElision>,
}

impl TcoTrace {
    /// The ring-buffer capacity: the number of most-recent elisions retained for the actionable
    /// trace. Bounded so a deep tail chain cannot grow the trace without limit (§4.6 tco32).
    pub const RING_CAP: usize = 32;

    /// Record one elided tail frame for `callee`: bump the totals and push onto the bounded ring.
    fn record(&mut self, callee: &str) {
        self.total_elided = self.total_elided.saturating_add(1);
        let iteration = {
            let c = self.per_callee.entry(callee.to_owned()).or_insert(0);
            *c = c.saturating_add(1);
            *c
        };
        if self.recent.len() == Self::RING_CAP {
            self.recent.pop_front();
        }
        self.recent.push_back(TcoElision {
            callee: callee.to_owned(),
            iteration,
        });
    }
}

/// The L1 evaluator over a checked [`Env`]. Construction wires the same trusted engines the
/// L0 paths use: the built-in prim registry and the certified binary↔ternary swap engine
/// (M-120/M-210) — override with [`Evaluator::with_engines`] for tests or extensions.
///
/// [`Evaluator::call`] runs the [work-stack CEK machine](Self::run_machine) on a deep worker stack
/// (see [`DEFAULT_DEPTH`]); the swap engine must be `Send + Sync` so `&Evaluator` can be shared
/// across the scoped worker thread (all built-in engines are `Copy`, hence `Send + Sync`).
pub struct Evaluator<'e> {
    env: &'e Env,
    prims: PrimRegistry,
    swap: Box<dyn SwapEngine + Send + Sync>,
    fuel: u64,
    depth: u32,
    /// The M-904 (DN-71 §8 FLAG-4 v0 posture) scope-exit **release log** — every deterministic
    /// release of a live, never-`consume`d `Substrate` binding is recorded here (never a silent
    /// leak — G2), inspectable via [`Self::release_events`]. A [`std::sync::Mutex`] (not a
    /// `RefCell`) so `Evaluator` stays `Sync` — [`Self::call`]'s deep-stack worker closure captures
    /// `&self` (see that method's doc for the full `Sync` argument); the lock is only ever held for
    /// the length of a single `Vec::push`/clone, so it never contends across the recursive walk.
    releases: std::sync::Mutex<Vec<crate::substrate::ReleaseEvent>>,
    /// The M-906 (DN-70 D1; RFC-0008 RT3) **mandatory-EXPLAIN placement trail** — every `@forage`
    /// decision made during evaluation, inspectable via [`Self::forage_decisions`] (never a black
    /// box — house rule 2). Same [`std::sync::Mutex`]-not-`RefCell` pattern as [`Self::releases`]
    /// (`Evaluator` stays `Sync`; the lock is only ever held for a single `Vec::push`/clone).
    forage_trail: std::sync::Mutex<Vec<ForageDecision>>,
    /// The RFC-0041 §4.6 (tco32) **tail-call EXPLAIN trail** — the bounded record of every frame the
    /// CEK machine elided by TCO, inspectable via [`Self::tco_trace`] (never a black box — house
    /// rule #2). Same [`std::sync::Mutex`]-not-`RefCell` pattern as [`Self::releases`] (`Evaluator`
    /// stays `Sync`; the lock is only ever held for a single record/clone).
    tco_trace: std::sync::Mutex<TcoTrace>,
}

impl<'e> Evaluator<'e> {
    /// An evaluator over `env` with the trusted default engines and the default budgets.
    #[must_use]
    pub fn new(env: &'e Env) -> Self {
        Evaluator {
            env,
            prims: PrimRegistry::with_builtins(),
            swap: Box::new(BinaryTernarySwapEngine),
            fuel: DEFAULT_FUEL,
            depth: DEFAULT_DEPTH,
            releases: std::sync::Mutex::new(Vec::new()),
            forage_trail: std::sync::Mutex::new(Vec::new()),
            tco_trace: std::sync::Mutex::new(TcoTrace::default()),
        }
    }

    /// The [`ForageDecision`] EXPLAIN trail accumulated by every `@forage(policy)` placement
    /// decision (M-906; DN-70 D1; RFC-0008 RT3) across every [`Self::call`] made on this
    /// `Evaluator` so far — mandatory EXPLAIN, never a black box (house rule 2; RFC-0005 §2.2).
    /// Empty iff no `@forage`-annotated hypha ever ran. A poisoned lock (only reachable after an
    /// unrelated panic while holding it, which this evaluator never does by design) is recovered
    /// rather than propagated (mirrors [`Self::release_events`]).
    #[must_use]
    pub fn forage_decisions(&self) -> Vec<ForageDecision> {
        self.forage_trail
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The [`TcoTrace`] EXPLAIN record accumulated by tail-call optimization (RFC-0041 §4.6 tco32)
    /// across every [`Self::call`] made on this `Evaluator` so far — the bounded per-callee tally +
    /// ring buffer of the most-recent elided tail frames, so a deep tail chain that ends in an
    /// over-budget error yields an actionable trace (house rule #2 — never a black box). Empty iff
    /// no tail call was ever elided. A poisoned lock (only reachable after an unrelated panic while
    /// holding it, which this evaluator never does by design) is recovered rather than propagated
    /// (mirrors [`Self::release_events`]).
    #[must_use]
    pub fn tco_trace(&self) -> TcoTrace {
        self.tco_trace
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The [`ReleaseEvent`](crate::substrate::ReleaseEvent) log accumulated by scope-exit releases
    /// (M-904; DN-71 §8 FLAG-4's v0 drop-without-consume posture) across every [`Self::call`] made
    /// on this `Evaluator` so far — inspectable, never a black box (house rule 2). Empty iff no live
    /// `Substrate` binding was ever abandoned (every one reached either escaped into its scope's own
    /// result, or had already been explicitly `consume`d). A poisoned lock (only reachable if a prior
    /// panic occurred while holding it, which this evaluator never does by design) is recovered
    /// rather than propagated, so an unrelated panic elsewhere can never make this accessor itself
    /// panic or silently report an empty log (G2).
    #[must_use]
    pub fn release_events(&self) -> Vec<crate::substrate::ReleaseEvent> {
        self.releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// M-904 (DN-71 §8 FLAG-4 v0 posture): release `popped` at its scope-exit iff it is a still-live
    /// `Substrate` handle that does **not** escape into `escaping` (the enclosing scope's own
    /// result — checked via [`value_contains_substrate_id`], not assumed, so a returned handle, or
    /// one nested inside a constructed `Data` value, is never wrongly released). Records a
    /// [`ReleaseEvent`](crate::substrate::ReleaseEvent) into [`Self::releases`] when a release
    /// actually happens. A non-`Substrate` binding, an already-terminal handle (already `consume`d,
    /// or already released through another clone of the same identity), or an escaping handle are
    /// all legitimate no-ops here — nothing to release, never an error.
    ///
    /// **Known v0 limitation (honest, not silently hidden — mirrors `crate::affine`'s documented
    /// loop/closure gap):** this is called at the two scope-exit points M-904 wires it into —
    /// `Expr::Let` and a function's own parameters at the end of [`Self::invoke`] — not at every
    /// binder in the evaluator (e.g. a `match`-arm pattern binder that captures a `Substrate` out of
    /// a data field is not yet covered). A handle abandoned only through such an uncovered binder is
    /// not released here; closing that gap is future work, not silently claimed done.
    fn release_if_abandoned(&self, popped: &(String, L1Value), escaping: Option<&L1Value>) {
        let L1Value::Substrate(handle) = &popped.1 else {
            return;
        };
        if let Some(v) = escaping {
            if value_contains_substrate_id(v, handle.id()) {
                return;
            }
        }
        if let Some(event) = handle.release(popped.0.clone()) {
            self.releases
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(event);
        }
    }

    /// Replace the prim registry and swap engine. The swap engine must be `Send + Sync` (all
    /// built-in engines are `Copy`, hence `Send + Sync`; a custom engine for tests likewise).
    #[must_use]
    pub fn with_engines(
        mut self,
        prims: PrimRegistry,
        swap: Box<dyn SwapEngine + Send + Sync>,
    ) -> Self {
        self.prims = prims;
        self.swap = swap;
        self
    }

    /// Override the step budget.
    #[must_use]
    pub fn with_fuel(mut self, fuel: u64) -> Self {
        self.fuel = fuel;
        self
    }

    /// Override the recursion-depth budget. Evaluation runs on the deep worker stack
    /// ([`mycelium_stack`]), so a raised budget is host-stack-safe — the budget is the ceiling,
    /// not the host stack.
    #[must_use]
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Apply a budget [`EvaluatorOpts`] in one call — equivalent to
    /// `self.with_fuel(opts.fuel).with_depth(opts.depth)`. Additive convenience; the engines are
    /// untouched (configure those with [`Evaluator::with_engines`]).
    #[must_use]
    pub fn with_opts(self, opts: EvaluatorOpts) -> Self {
        self.with_fuel(opts.fuel).with_depth(opts.depth)
    }

    /// Call function `name` with `args`, big-step, under the configured budgets. The result
    /// honors the signature's dynamic guarantee index, if any (RFC-0007 §4.3).
    ///
    /// The recursive evaluation runs on a deep worker stack (256 MiB, lazily committed) via
    /// [`mycelium_stack::with_deep_stack`], so the **explicit [`DEFAULT_DEPTH`] budget** — not
    /// the caller's thread stack — is always the bound. The host stack never overflows for any
    /// budget value: [`L1Error::DepthExceeded`] is always what trips first (banked guard 4). Cost:
    /// one worker-thread spawn per call (~tens of µs); shallow programs touch only a few stack
    /// pages (lazily committed). The worker stack is the transitional Rust-host adapter; the
    /// budget is the portable primitive for the future self-hosted frontend.
    pub fn call(&self, name: &str, args: Vec<L1Value>) -> Result<L1Value, L1Error> {
        // Run the recursive evaluation on the deep worker stack so the explicit depth budget —
        // not the caller's thread stack — is the bound for any budget value. The closure captures
        // `&self`; this is safe because `Evaluator: Sync` (all fields are `Sync`: `&Env`,
        // `PrimRegistry` — a `BTreeMap<String, fn(…)>` — and `Box<dyn SwapEngine + Send + Sync>`).
        mycelium_stack::with_deep_stack(|| {
            // RFC-0041 W5 (M-979): the shared source-call/β depth budget on the §4.0 metric. The L1
            // path charges only the depth ceiling (its memory/work-step ceilings stay unbounded —
            // wiring a real L1 memory ceiling is future §4.2 work, tracked, not silently claimed);
            // the canonical `BudgetError::DepthExceeded { limit }` maps to `L1Error::DepthExceeded`
            // at the same threshold (§5.1). `self.depth` is the configured ceiling (default 4096).
            let budget = RecursionBudget::new(self.depth, u64::MAX, u64::MAX);
            let mut fuel = self.fuel;
            let mut ledger = Budgets::new();
            // The deep worker stack still backstops the *non-control* recursions that remain on the
            // host stack (nested-`Pattern` `try_match`, the deep-`Data` escape check, `L1Value`
            // iterative ops) — the CEK machine removes the primary (control) host-stack consumer, so
            // the explicit budget is what bounds a pathological input on every path (banked guard 4).
            self.run_machine(&budget, &mut fuel, &mut ledger, name, args)
        })
    }

    /// Map a surface effect name to its [`EffectKind`] and create the corresponding
    /// [`EffectBudget`] variant with the given ceiling (RFC-0014 §4.5 I3/I4; M-677).
    /// The mapping is closed: the five built-in kinds plus a fall-through `Named` bucket for any
    /// user-declared name. `"retry"` → `Attempts`, `"alloc"` → `Bytes`, `"io"` → `Ops`,
    /// `"cascade"` → `Depth`, `"time"` → `Fuel`, any other → `Named`.
    fn effect_name_to_budget(name: &str, ceiling: u64) -> EffectBudget {
        match name {
            "retry" => EffectBudget::Attempts(ceiling),
            "alloc" => EffectBudget::Bytes(ceiling),
            "io" => EffectBudget::Ops(ceiling),
            "cascade" => EffectBudget::Depth(ceiling),
            "time" => EffectBudget::Fuel(ceiling),
            other => EffectBudget::Named(other.to_owned(), ceiling),
        }
    }

    /// **The L1 work-stack CEK machine (RFC-0041 §4.1/§4.6, M-979).** The former 7-function
    /// recursive SCC (`invoke`/`eval`/`eval_app`/`eval_match`/`eval_for`/`eval_wild`/
    /// `eval_hypha_forage`) is now **one iterative loop** over an explicit heap work-stack
    /// (`Vec<Frame>`): each [`Frame`] reifies the interleaved *post-child* work (scope push/pop,
    /// the M-904 `release_if_abandoned`, the return-guarantee assert, the swap/ascription checks) as
    /// an explicit continuation, so control recursion is **O(1) host stack** for any input depth —
    /// the depth budget, not the host stack, is what refuses a pathological input (never-silent, G2).
    ///
    /// The environment is a single shared `scope: Vec<(name, value)>` with a per-function `base`
    /// marker (each function sees only `scope[base..]`, preserving lexical scope across calls). The
    /// depth budget is charged **once per `Expr::App` boundary** (the §4.0 source-call/β metric) via
    /// the shared [`RecursionBudget`] — the [`DepthGuard`] is held on the work-stack for the whole
    /// App (its args *and*, for a user call, the invoked body), so a nested call chain grows depth
    /// exactly as the metric predicts and refuses at the ceiling with the canonical
    /// [`L1Error::DepthExceeded`] (§5.1). Tail calls with no pending post-work reuse their frame
    /// (TCO — [`Self::enter_call`]).
    fn run_machine<'b>(
        &self,
        budget: &'b RecursionBudget,
        fuel: &mut u64,
        ledger: &mut Budgets,
        name: &str,
        args: Vec<L1Value>,
    ) -> Result<L1Value, L1Error> {
        let mut regs: Regs<'e, 'b> = Regs {
            site: "",
            base: 0,
            scope: Vec::new(),
            stack: Vec::new(),
        };
        // Charge the top-level source-call frame, then set up its invocation (params, effect ledger,
        // the return-guarantee `InvokePost`). A refusal here is the never-silent depth ceiling.
        let guard = match budget.try_enter() {
            Ok(g) => g,
            Err(e) => return Err(depth_exceeded(e)),
        };
        let mut ctrl = self.enter_call(fuel, ledger, &mut regs, name, args, guard);
        loop {
            ctrl = match ctrl {
                Ctrl::Eval(e) => self.eval_step(budget, fuel, ledger, &mut regs, e),
                Ctrl::Settle(settled) => match regs.stack.pop() {
                    // The work-stack is empty — the top-level call has settled; this is the result.
                    None => return settled,
                    Some(frame) => {
                        self.apply_frame(budget, fuel, ledger, &mut regs, frame, settled)
                    }
                },
            };
        }
    }

    /// One step of the machine in **Eval** mode: dispatch on `e`, either producing a value
    /// (`Ctrl::Settle`) for a leaf, or pushing the continuation [`Frame`] for its post-child work and
    /// descending into the first child (`Ctrl::Eval`). Every node costs one unit of fuel at entry
    /// (the same per-node clock as the former `eval`, so an unproductive recursion is an explicit
    /// [`L1Error::FuelExhausted`], never a hang — §4.5).
    #[allow(clippy::too_many_lines)] // one dispatch arm per surface `Expr`; splitting would obscure it
    fn eval_step<'b>(
        &self,
        budget: &'b RecursionBudget,
        fuel: &mut u64,
        _ledger: &mut Budgets,
        regs: &mut Regs<'e, 'b>,
        e: &'e Expr,
    ) -> Ctrl<'e> {
        // Per-node fuel (matches the former `eval` entry): a wide *and* a deep AST both cost fuel.
        match fuel.checked_sub(1) {
            Some(f) => *fuel = f,
            None => return Ctrl::Settle(Err(L1Error::FuelExhausted)),
        }
        let site = regs.site;
        match e {
            // Context-free repr literals (binary/ternary/bytes/string/float) lower via `lit_value`.
            Expr::Lit(
                l @ (Literal::Bin(_)
                | Literal::Trit(_)
                | Literal::Bytes(_)
                | Literal::Str(_)
                | Literal::Float(_)),
            ) => Ctrl::Settle(
                lit_value(site, l)
                    .map(L1Value::Repr)
                    .map_err(|err| unsupported(site, &err)),
            ),
            // A list literal `[e1, …]` evaluates each element (left-to-right) to a repr, then builds
            // a `Repr::Seq` (RFC-0032 D3). An empty `[]` has no anchoring element repr — refused.
            Expr::Lit(Literal::List(elems)) => {
                if elems.is_empty() {
                    return Ctrl::Settle(Err(L1Error::Unsupported {
                        site: site.to_owned(),
                        what: "an empty list literal `[]` has no element repr to anchor the `Seq` \
                               descriptor at eval (RFC-0032 D3)"
                            .to_owned(),
                    }));
                }
                regs.stack.push(Frame::ListElem {
                    elems,
                    idx: 0,
                    vals: Vec::with_capacity(elems.len()),
                });
                Ctrl::Eval(&elems[0])
            }
            Expr::Lit(_) => Ctrl::Settle(Err(L1Error::Unsupported {
                site: site.to_owned(),
                what: "bare-integer literals have no v0 value form (Q6)".to_owned(),
            })),

            Expr::Path(p) => Ctrl::Settle(self.eval_path(regs, p)),

            Expr::Let {
                name,
                ty,
                bound,
                body,
            } => {
                regs.stack.push(Frame::LetBound {
                    name: name.as_str(),
                    ty_guar: ty.as_ref().and_then(|t| t.guarantee),
                    body,
                });
                Ctrl::Eval(bound)
            }

            Expr::If { cond, conseq, alt } => {
                regs.stack.push(Frame::IfBranch { conseq, alt });
                Ctrl::Eval(cond)
            }

            Expr::Match { scrutinee, arms } => {
                regs.stack.push(Frame::MatchArms {
                    arms: arms.as_slice(),
                });
                Ctrl::Eval(scrutinee)
            }

            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => {
                regs.stack.push(Frame::ForAfterXs {
                    x: x.as_str(),
                    acc: acc.as_str(),
                    init,
                    body,
                });
                Ctrl::Eval(xs)
            }

            Expr::Swap {
                value,
                target,
                policy,
            } => {
                regs.stack.push(Frame::SwapPost { target, policy });
                Ctrl::Eval(value)
            }

            Expr::WithParadigm { .. } => Ctrl::Settle(Err(L1Error::Unsupported {
                site: site.to_owned(),
                what: "internal: a `with paradigm` block reached the evaluator — the ambient \
                       resolution pass strips it (RFC-0012 §4.4)"
                    .to_owned(),
            })),

            // A `wild { name(args…) }` block (the audited FFI floor — M-661/M-721; RFC-0028 §4.3):
            // dispatch the host op through the reserved `wild:` prim namespace after evaluating its
            // arguments left-to-right (CBV). Only the *shape* `name(args…)` / bare `name` is
            // interpreted; any other shape is an explicit refusal (never silent — G2).
            Expr::Wild(body) => {
                let (opname, args): (&str, &'e [Expr]) = match body.as_ref() {
                    Expr::App { head, args } => match head.as_ref() {
                        Expr::Path(p) if p.0.len() == 1 => (p.0[0].as_str(), args.as_slice()),
                        _ => {
                            return Ctrl::Settle(Err(L1Error::Unsupported {
                                site: site.to_owned(),
                                what: "a v0 `wild` block body must be a host-call form \
                                       `name(args…)` with a single, undotted host-operation name \
                                       (RFC-0028 §4.2)"
                                    .to_owned(),
                            }))
                        }
                    },
                    Expr::Path(p) if p.0.len() == 1 => (p.0[0].as_str(), &[]),
                    _ => {
                        return Ctrl::Settle(Err(L1Error::Unsupported {
                            site: site.to_owned(),
                            what: "a v0 `wild` block body must be a host-call form `name(args…)` \
                                   or a bare `name` (RFC-0028 §4.2)"
                                .to_owned(),
                        }))
                    }
                };
                let key = format!("wild:{opname}");
                if args.is_empty() {
                    return self.wild_dispatch(regs, key, Vec::new());
                }
                regs.stack.push(Frame::WildArgs {
                    key,
                    args,
                    idx: 0,
                    argv: Vec::with_capacity(args.len()),
                });
                Ctrl::Eval(&args[0])
            }

            Expr::Spore(_) => Ctrl::Settle(Err(L1Error::Unsupported {
                site: site.to_owned(),
                what: "`spore` is deferred to the reconstruction-manifest work (E2-5/M-260)"
                    .to_owned(),
            })),

            // `consume <expr>` — the M-904 checked identity-move (DN-71 Model S §4.3).
            Expr::Consume(operand) => {
                regs.stack.push(Frame::ConsumePost);
                Ctrl::Eval(operand)
            }

            Expr::Lambda { .. } => Ctrl::Settle(Err(L1Error::Unsupported {
                site: site.to_owned(),
                what:
                    "internal: an `Expr::Lambda` reached the evaluator — closures are lowered by \
                       monomorphization (RFC-0024 §4A / M-704); run eval on the monomorphized env \
                       (never a silent accept, G2)"
                        .to_owned(),
            })),

            // `colony { hypha e1, …, hypha eN }` (RFC-0008 §4.7; M-666): the RT2 spawn-order
            // sequentialization — each hypha body runs in order (its `@forage` consulted first), and
            // the colony's value is the *last* hypha's. The parser guarantees ≥ 1 hypha.
            Expr::Colony(hyphae) => {
                if hyphae.is_empty() {
                    return Ctrl::Settle(Err(L1Error::Unsupported {
                        site: site.to_owned(),
                        what: "internal: an empty `colony` reached the evaluator — the parser \
                               requires ≥ 1 hypha (RFC-0008 §4.7)"
                            .to_owned(),
                    }));
                }
                self.start_hypha(regs, hyphae, 0)
            }

            Expr::Ascribe(inner, t) => {
                // A guarantee index asserts post-work; an unindexed ascription is transparent (no
                // frame — so it stays TCO-eligible when it is the fn's tail).
                if let Some(g) = t.guarantee {
                    regs.stack.push(Frame::AscribePost { guar: g });
                }
                Ctrl::Eval(inner)
            }

            // A function/constructor/prim application: the §4.0 source-call/β boundary. Charge one
            // depth unit (held for the whole App — args *and*, for a user call, the invoked body).
            Expr::App { head, args } => {
                let guard = match budget.try_enter() {
                    Ok(g) => g,
                    Err(err) => return Ctrl::Settle(Err(depth_exceeded(err))),
                };
                let Expr::Path(p) = head.as_ref() else {
                    return Ctrl::Settle(Err(L1Error::Stuck {
                        site: site.to_owned(),
                        why: "v0 application head must be a name (first-order)".to_owned(),
                    }));
                };
                if p.0.len() != 1 {
                    return Ctrl::Settle(Err(L1Error::Stuck {
                        site: site.to_owned(),
                        why: format!("dotted call `{}`", p.0.join(".")),
                    }));
                }
                let opname = p.0[0].as_str();
                if args.is_empty() {
                    return self.app_dispatch(fuel, _ledger, regs, opname, Vec::new(), guard);
                }
                regs.stack.push(Frame::AppArgs {
                    name: opname,
                    args: args.as_slice(),
                    idx: 0,
                    argv: Vec::with_capacity(args.len()),
                    guard,
                });
                Ctrl::Eval(&args[0])
            }

            // `fuse(a, b)` (DN-58 §A): lawful binary merge. Evaluate both operands, then combine
            // (repr → the `fuse_join:binary` prim; data → the user `join` fn).
            Expr::Fuse { left, right } => {
                regs.stack.push(Frame::FuseAfterLeft {
                    left_expr: left,
                    right_expr: right,
                });
                Ctrl::Eval(left)
            }

            // `reclaim(policy) { body }` (DN-58 §B): the sequential reference — evaluate the policy
            // for effect, then the body (whose value is the scope's observable).
            Expr::Reclaim { policy, body } => {
                regs.stack.push(Frame::ReclaimBody { body });
                Ctrl::Eval(policy)
            }

            Expr::TupleLit(_) => Ctrl::Settle(Err(L1Error::Unsupported {
                site: site.to_owned(),
                what: "internal: a TupleLit reached the evaluator — tuple literals are lowered by \
                       the checker to constructor applications (M-826); run eval on a checked, \
                       monomorphized env (never a silent accept, G2)"
                    .to_owned(),
            })),
        }
    }

    /// One step of the machine in **Settle** mode: `frame` is the just-popped continuation and
    /// `settled` is the child's `Ok(value)` / `Err`. Each frame runs its reified post-child work —
    /// including, on the error path, its deterministic scope-exit cleanup (`release_if_abandoned`,
    /// scope restoration), so an error unwinds the work-stack *never-silently* (G2), exactly as the
    /// former recursive evaluator ran cleanup on both the success and error path.
    #[allow(clippy::too_many_lines)] // one arm per continuation frame; splitting would obscure it
    fn apply_frame<'b>(
        &self,
        budget: &'b RecursionBudget,
        fuel: &mut u64,
        ledger: &mut Budgets,
        regs: &mut Regs<'e, 'b>,
        frame: Frame<'e, 'b>,
        settled: Result<L1Value, L1Error>,
    ) -> Ctrl<'e> {
        let site = regs.site;
        match frame {
            Frame::ListElem {
                elems,
                idx,
                mut vals,
            } => {
                let v = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                match v.as_repr() {
                    Some(rv) => vals.push(rv.clone()),
                    None => {
                        return Ctrl::Settle(Err(L1Error::Unsupported {
                            site: site.to_owned(),
                            what: "a list literal element is not a representation value — a v0 \
                                   `Seq` is built from repr elements only (RFC-0032 D3)"
                                .to_owned(),
                        }))
                    }
                }
                if idx + 1 < elems.len() {
                    let next = idx + 1;
                    regs.stack.push(Frame::ListElem {
                        elems,
                        idx: next,
                        vals,
                    });
                    return Ctrl::Eval(&elems[next]);
                }
                // All elements collected — build the `Seq` (the first repr anchors the descriptor).
                let first = vals.first().expect("non-empty: checked at Eval time");
                let elem = first.repr().clone();
                let len = u32::try_from(vals.len()).unwrap_or(u32::MAX);
                Ctrl::Settle(
                    mycelium_core::Value::new(
                        mycelium_core::Repr::Seq {
                            elem: Box::new(elem),
                            len,
                        },
                        mycelium_core::Payload::Seq(vals),
                        mycelium_core::Meta::exact(mycelium_core::Provenance::Root),
                    )
                    .map(L1Value::Repr)
                    .map_err(|err| L1Error::Stuck {
                        site: site.to_owned(),
                        why: format!("malformed sequence literal: {err}"),
                    }),
                )
            }

            Frame::LetBound {
                name,
                ty_guar,
                body,
            } => {
                let bv = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                if let Some(g) = ty_guar {
                    if let Err(e) = self.assert_guarantee(site, &bv, g) {
                        return Ctrl::Settle(Err(e));
                    }
                }
                regs.scope.push((name.to_owned(), bv));
                regs.stack.push(Frame::LetPop);
                Ctrl::Eval(body)
            }

            Frame::LetPop => {
                // The `let` binding's scope ends here (M-904): release it if it is a still-live
                // `Substrate` that does not escape into this `let`'s own result. Runs on the success
                // *and* error path (deterministic scope-exit release, never a silent leak — G2).
                let popped = regs.scope.pop().expect("let binding present");
                self.release_if_abandoned(&popped, settled.as_ref().ok());
                Ctrl::Settle(settled)
            }

            Frame::IfBranch { conseq, alt } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(c) => match c {
                    L1Value::Data { ref ctor, .. } if ctor == "True" => Ctrl::Eval(conseq),
                    L1Value::Data { ref ctor, .. } if ctor == "False" => Ctrl::Eval(alt),
                    other => Ctrl::Settle(Err(L1Error::Stuck {
                        site: site.to_owned(),
                        why: format!("if-condition evaluated to a non-Bool: {other:?}"),
                    })),
                },
            },

            Frame::MatchArms { arms } => {
                let sv = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                // The checker verified exhaustiveness/arity (W7): the first arm whose pattern matches
                // fires; the trailing `Stuck` is the honest never-silent fallback (G2).
                for arm in arms {
                    let mut binds: Vec<(String, L1Value)> = Vec::new();
                    match self.try_match(site, &arm.pattern, &sv, &mut binds) {
                        Err(e) => return Ctrl::Settle(Err(e)),
                        Ok(false) => {}
                        Ok(true) => {
                            let mark = regs.scope.len();
                            regs.scope.extend(binds);
                            regs.stack.push(Frame::MatchPop { mark });
                            return Ctrl::Eval(&arm.body);
                        }
                    }
                }
                Ctrl::Settle(Err(L1Error::Stuck {
                    site: site.to_owned(),
                    why: "no arm matched the scrutinee (W7 — the checker requires coverage)"
                        .to_owned(),
                }))
            }

            Frame::MatchPop { mark } => {
                // Restore the scope to before the arm's binders (on both success and error).
                regs.scope.truncate(mark);
                Ctrl::Settle(settled)
            }

            Frame::SwapPost { target, policy } => {
                let v = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                let Some(src) = v.as_repr() else {
                    return Ctrl::Settle(Err(L1Error::Stuck {
                        site: site.to_owned(),
                        why: "swap source is not a representation value".to_owned(),
                    }));
                };
                let repr = match type_repr(site, target) {
                    Ok(r) => r,
                    Err(err) => return Ctrl::Settle(Err(unsupported(site, &err))),
                };
                let out = match self.swap.swap(src, &repr, &policy_name_ref(policy)) {
                    Ok(o) => L1Value::Repr(o),
                    Err(err) => return Ctrl::Settle(Err(L1Error::from(err))),
                };
                if let Some(g) = target.guarantee {
                    if let Err(e) = self.assert_guarantee(site, &out, g) {
                        return Ctrl::Settle(Err(e));
                    }
                }
                Ctrl::Settle(Ok(out))
            }

            Frame::ConsumePost => {
                let v = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                let Some(handle) = v.as_substrate() else {
                    return Ctrl::Settle(Err(L1Error::Stuck {
                        site: site.to_owned(),
                        why: "internal: `consume`'s operand evaluated to a non-Substrate value — \
                              `check_consume`'s type rule (DN-03 §1 / LR-8) guarantees a \
                              `Substrate{tag}` operand, so this is a staging invariant break, never \
                              a silent move (G2)"
                            .to_owned(),
                    }));
                };
                Ctrl::Settle(handle.try_consume().map(L1Value::Substrate).map_err(|err| {
                    L1Error::Stuck {
                        site: site.to_owned(),
                        why: err.to_string(),
                    }
                }))
            }

            Frame::AscribePost { guar } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(v) => match self.assert_guarantee(site, &v, guar) {
                    Ok(()) => Ctrl::Settle(Ok(v)),
                    Err(e) => Ctrl::Settle(Err(e)),
                },
            },

            Frame::FuseAfterLeft {
                left_expr,
                right_expr,
            } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(lv) => {
                    regs.stack.push(Frame::FuseAfterRight {
                        lv,
                        left_expr,
                        right_expr,
                    });
                    Ctrl::Eval(right_expr)
                }
            },

            Frame::FuseAfterRight {
                lv,
                left_expr,
                right_expr,
            } => {
                let rv = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                match (&lv, &rv) {
                    // Repr fuse = the `Binary` semilattice meet (bitwise-AND) via the shared
                    // `fuse_join:binary` prim (the same prim the L0/AOT paths dispatch — DN-58 §A.5).
                    (L1Value::Repr(lrepr), L1Value::Repr(rrepr)) => {
                        let Some(f) = self.prims.get("fuse_join:binary") else {
                            return Ctrl::Settle(Err(L1Error::Kernel(KernelError::UnknownPrim(
                                "fuse_join:binary".to_owned(),
                            ))));
                        };
                        Ctrl::Settle(
                            f("fuse_join:binary", &[lrepr, rrepr])
                                .map(L1Value::Repr)
                                .map_err(L1Error::from),
                        )
                    }
                    // Data type: dispatch through the user `join` fn (its Fuse instance). Preserves
                    // the former semantics exactly by re-evaluating both operand *expressions* into
                    // the `join(left, right)` call (a fresh source-call/β boundary — depth-charged).
                    (L1Value::Data { .. }, _) => {
                        let guard = match budget.try_enter() {
                            Ok(g) => g,
                            Err(err) => return Ctrl::Settle(Err(depth_exceeded(err))),
                        };
                        regs.stack.push(Frame::FuseJoinLeft { right_expr, guard });
                        Ctrl::Eval(left_expr)
                    }
                    _ => Ctrl::Settle(Err(L1Error::Stuck {
                        site: site.to_owned(),
                        why:
                            "`fuse` applied to mixed repr/data operands — internal type error (the \
                              checker should have rejected this; DN-58 §A.4 — never-silent, G2)"
                                .to_owned(),
                    })),
                }
            }

            Frame::FuseJoinLeft { right_expr, guard } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(lv2) => {
                    regs.stack.push(Frame::FuseJoinRight { lv2, guard });
                    Ctrl::Eval(right_expr)
                }
            },

            Frame::FuseJoinRight { lv2, guard } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(rv2) => self.app_dispatch(fuel, ledger, regs, "join", vec![lv2, rv2], guard),
            },

            Frame::ReclaimBody { body } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(_policy_value) => Ctrl::Eval(body),
            },

            Frame::ColonyForage { hyphae, idx } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(pv) => match self.forage_select(site, &pv) {
                    Err(e) => Ctrl::Settle(Err(e)),
                    Ok(()) => {
                        regs.stack.push(Frame::ColonyBody { hyphae, idx });
                        Ctrl::Eval(&hyphae[idx].body)
                    }
                },
            },

            Frame::ColonyBody { hyphae, idx } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(v) => {
                    if idx + 1 == hyphae.len() {
                        // The last hypha's value is the colony's observable.
                        Ctrl::Settle(Ok(v))
                    } else {
                        // A leading hypha's value is discarded (sequentialized for effect only).
                        drop(v);
                        self.start_hypha(regs, hyphae, idx + 1)
                    }
                }
            },

            Frame::WildArgs {
                key,
                args,
                idx,
                mut argv,
            } => {
                let v = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                argv.push(v);
                if idx + 1 < args.len() {
                    let next = idx + 1;
                    regs.stack.push(Frame::WildArgs {
                        key,
                        args,
                        idx: next,
                        argv,
                    });
                    return Ctrl::Eval(&args[next]);
                }
                self.wild_dispatch(regs, key, argv)
            }

            Frame::AppArgs {
                name,
                args,
                idx,
                mut argv,
                guard,
            } => {
                let v = match settled {
                    Err(e) => return Ctrl::Settle(Err(e)),
                    Ok(v) => v,
                };
                argv.push(v);
                if idx + 1 < args.len() {
                    let next = idx + 1;
                    regs.stack.push(Frame::AppArgs {
                        name,
                        args,
                        idx: next,
                        argv,
                        guard,
                    });
                    return Ctrl::Eval(&args[next]);
                }
                self.app_dispatch(fuel, ledger, regs, name, argv, guard)
            }

            Frame::InvokePost {
                param_base,
                saved_site,
                saved_base,
                ret_guar,
                nparams,
                tco_eligible: _,
                guard: _,
            } => {
                // The function body has settled. Release this call's own parameter scope (M-904) on
                // both the success and error path — a still-live `Substrate` param that does not
                // escape into the call's result is released deterministically, never a silent leak.
                for i in param_base..param_base + nparams {
                    self.release_if_abandoned(&regs.scope[i], settled.as_ref().ok());
                }
                // The return-guarantee index (if any) is checked against the callee's own name
                // (`regs.site` is still the callee here) — before we restore the caller's context.
                let next = match settled {
                    Err(e) => Ctrl::Settle(Err(e)),
                    Ok(v) => match ret_guar {
                        None => Ctrl::Settle(Ok(v)),
                        Some(g) => match self.assert_guarantee(regs.site, &v, g) {
                            Ok(()) => Ctrl::Settle(Ok(v)),
                            Err(e) => Ctrl::Settle(Err(e)),
                        },
                    },
                };
                regs.scope.truncate(param_base);
                regs.site = saved_site;
                regs.base = saved_base;
                // `guard` drops here — this call's depth unit is released.
                next
            }

            Frame::ForAfterXs { x, acc, init, body } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(spine) => {
                    regs.stack.push(Frame::ForAfterInit {
                        x,
                        acc,
                        body,
                        spine,
                    });
                    Ctrl::Eval(init)
                }
            },

            Frame::ForAfterInit {
                x,
                acc,
                body,
                spine,
            } => match settled {
                Err(e) => Ctrl::Settle(Err(e)),
                Ok(accv) => self.for_advance(regs, fuel, x, acc, body, spine, accv),
            },

            Frame::ForStep {
                x,
                acc,
                body,
                spine,
            } => {
                // Remove this element's `x`/`acc` bindings (pushed before the body ran); their values
                // drop. Match-arm/for binders are not `Substrate`-tracked (the documented v0 gap).
                regs.scope.pop();
                regs.scope.pop();
                match settled {
                    Err(e) => Ctrl::Settle(Err(e)),
                    Ok(accv) => self.for_advance(regs, fuel, x, acc, body, spine, accv),
                }
            }
        }
    }

    /// Resolve a variable/constructor/function reference in value position (the former `eval`
    /// `Expr::Path` arm). A single-segment name resolves against the **current function's** scope
    /// (`scope[base..]`, innermost first — so lexical scope is preserved across calls), then a
    /// nullary constructor, then a bare top-level function value (ADR-033/DN-74). Never-silent on an
    /// unresolved name (G2).
    fn eval_path<'b>(&self, regs: &Regs<'e, 'b>, p: &crate::ast::Path) -> Result<L1Value, L1Error> {
        if p.0.len() == 1 {
            let name = &p.0[0];
            if let Some((_, v)) = regs.scope[regs.base..]
                .iter()
                .rev()
                .find(|(n, _)| n == name)
            {
                return Ok(v.clone());
            }
            if let Some((d, i)) = self.env.ctor(name) {
                if d.ctors[i].fields.is_empty() {
                    return Ok(L1Value::Data {
                        ty: d.name.clone(),
                        ctor: name.clone(),
                        fields: Arc::new(vec![]),
                    });
                }
            }
            if self.env.fns.contains_key(name) {
                return Ok(L1Value::Fn(name.clone()));
            }
        }
        Err(L1Error::Stuck {
            site: regs.site.to_owned(),
            why: format!("unresolved name `{}`", p.0.join(".")),
        })
    }

    /// Begin one hypha's evaluation inside a `colony` (its `@forage` placement decision first, if
    /// present, then its body). Shared by the colony's initial entry and each `ColonyBody` advance.
    fn start_hypha<'b>(
        &self,
        regs: &mut Regs<'e, 'b>,
        hyphae: &'e [Hypha],
        idx: usize,
    ) -> Ctrl<'e> {
        if hyphae[idx].forage.is_some() {
            let policy = hyphae[idx]
                .forage
                .as_deref()
                .expect("checked is_some above");
            regs.stack.push(Frame::ColonyForage { hyphae, idx });
            Ctrl::Eval(policy)
        } else {
            regs.stack.push(Frame::ColonyBody { hyphae, idx });
            Ctrl::Eval(&hyphae[idx].body)
        }
    }

    /// Dispatch a saturated application `name(argv)` (the former `eval_app` tail): a scope-bound
    /// function value (ADR-033/DN-74 dictionary dispatch), then a top-level function `invoke`
    /// ([`Self::enter_call`], which applies TCO), then a saturated constructor, then a kernel prim.
    /// `guard` is the App's depth reservation — moved into the callee's frame for a user call, or
    /// dropped here once a constructor/prim result is built (the App boundary is complete).
    fn app_dispatch<'b>(
        &self,
        fuel: &mut u64,
        ledger: &mut Budgets,
        regs: &mut Regs<'e, 'b>,
        name: &str,
        argv: Vec<L1Value>,
        guard: DepthGuard<'b>,
    ) -> Ctrl<'e> {
        let site = regs.site;
        // A scope-bound name (e.g. a match-projected dictionary field holding an `L1Value::Fn`)
        // shadows the global namespace — checked first, exactly as the value-position `Path` arm.
        if let Some((_, v)) = regs.scope[regs.base..]
            .iter()
            .rev()
            .find(|(n, _)| n == name)
        {
            let L1Value::Fn(callee) = v else {
                return Ctrl::Settle(Err(L1Error::Stuck {
                    site: site.to_owned(),
                    why: format!("`{name}` is bound to a non-function value and cannot be applied"),
                }));
            };
            let callee = callee.clone();
            return self.enter_call(fuel, ledger, regs, &callee, argv, guard);
        }
        if self.env.fns.contains_key(name) {
            return self.enter_call(fuel, ledger, regs, name, argv, guard);
        }
        if let Some((d, i)) = self.env.ctor(name) {
            if d.ctors[i].fields.len() != argv.len() {
                return Ctrl::Settle(Err(L1Error::Stuck {
                    site: site.to_owned(),
                    why: format!("unsaturated constructor `{name}` (W6)"),
                }));
            }
            // The App boundary is complete — `guard` drops, releasing this App's depth unit.
            drop(guard);
            return Ctrl::Settle(Ok(L1Value::Data {
                ty: d.name.clone(),
                ctor: name.to_owned(),
                fields: Arc::new(argv),
            }));
        }
        if let Some(kernel) = prim_kernel_name(name) {
            let vals: Result<Vec<&Value>, L1Error> = argv
                .iter()
                .map(|v| {
                    v.as_repr().ok_or_else(|| L1Error::Stuck {
                        site: site.to_owned(),
                        why: format!("prim `{name}` applied to a data value"),
                    })
                })
                .collect();
            let out = match vals {
                Err(e) => Err(e),
                Ok(vals) => match self.prims.get(kernel) {
                    None => Err(L1Error::Kernel(KernelError::UnknownPrim(kernel.to_owned()))),
                    Some(f) => f(kernel, &vals).map(L1Value::Repr).map_err(L1Error::from),
                },
            };
            drop(guard); // prim result built — release this App's depth unit.
            return Ctrl::Settle(out);
        }
        drop(guard);
        Ctrl::Settle(Err(L1Error::Stuck {
            site: site.to_owned(),
            why: format!("unknown function/constructor/prim `{name}`"),
        }))
    }

    /// Dispatch a `wild { name(args…) }` host op through the reserved `wild:` prim namespace (the
    /// former `eval_wild` tail). All arguments must be repr values; an ungranted host op is an
    /// explicit [`KernelError::UnknownPrim`] (never silent — G2). `wild` is not a source-call/β
    /// boundary in the §4.0 metric (a host op never recurses), so it charges no depth.
    fn wild_dispatch<'b>(&self, regs: &Regs<'e, 'b>, key: String, argv: Vec<L1Value>) -> Ctrl<'e> {
        let site = regs.site;
        // `key` is `wild:<name>`; recover the bare op name for diagnostics.
        let opname = key.strip_prefix("wild:").unwrap_or(&key);
        let vals: Result<Vec<&Value>, L1Error> = argv
            .iter()
            .map(|v| {
                v.as_repr().ok_or_else(|| L1Error::Stuck {
                    site: site.to_owned(),
                    why: format!(
                        "`wild` host op `{opname}` applied to a data value (RFC-0028 §4.4)"
                    ),
                })
            })
            .collect();
        Ctrl::Settle(match vals {
            Err(e) => Err(e),
            Ok(vals) => match self.prims.get(&key) {
                None => Err(L1Error::Kernel(KernelError::UnknownPrim(key.clone()))),
                Some(f) => f(&key, &vals).map(L1Value::Repr).map_err(L1Error::from),
            },
        })
    }

    /// Begin one function invocation (the former `invoke`): arity + effect-budget checks, bind the
    /// parameters into the shared scope, and push the [`Frame::InvokePost`] that runs the scope-exit
    /// release + return-guarantee assert. **TCO (RFC-0041 §4.6):** if this call is a *direct tail
    /// call* — the caller's `InvokePost` is the current top of the work-stack — and that caller frame
    /// has **no pending post-work** (no `sig.ret.guarantee` index *and* no `Substrate`-valued
    /// parameter, so its post-work is an observational no-op), the caller frame is **elided** and
    /// this callee reuses its slot: the caller's depth guard drops as this callee's is installed, so
    /// a tail-recursive loop runs in **bounded depth**. Every elision is recorded in the bounded
    /// [`TcoTrace`] EXPLAIN ring (§4.6 tco32). `guard` is this call's already-charged depth unit.
    fn enter_call<'b>(
        &self,
        _fuel: &mut u64,
        ledger: &mut Budgets,
        regs: &mut Regs<'e, 'b>,
        name: &str,
        argv: Vec<L1Value>,
        guard: DepthGuard<'b>,
    ) -> Ctrl<'e> {
        let Some((fname, fd)) = self.env.fns.get_key_value(name) else {
            return Ctrl::Settle(Err(L1Error::Stuck {
                site: name.to_owned(),
                why: format!("unknown function `{name}`"),
            }));
        };
        let fname: &'e str = fname.as_str();
        if fd.sig.value_params.len() != argv.len() {
            return Ctrl::Settle(Err(L1Error::Stuck {
                site: name.to_owned(),
                why: format!(
                    "`{name}` takes {} argument(s), got {}",
                    fd.sig.value_params.len(),
                    argv.len()
                ),
            }));
        }
        // Prime + consume the shared effect-budget ledger (M-677 / RFC-0014 §4.5 I4) before binding
        // params or evaluating the body — exactly as the former `invoke`.
        for (eff_name, &ceiling) in &fd.sig.effect_budgets {
            let budget = Self::effect_name_to_budget(eff_name, ceiling);
            if ledger.remaining(&budget.kind()).is_none() {
                ledger.set(budget);
            }
        }
        for eff_name in fd.sig.effect_budgets.keys() {
            let kind = Self::effect_name_to_budget(eff_name, 0).kind();
            if let Err(e) = ledger.consume(kind, 1) {
                return Ctrl::Settle(Err(L1Error::EffectBudget(e)));
            }
        }

        let ret_guar = fd.sig.ret.guarantee;
        // The precondition the frame-reuse (TCO) of *this* callee is gated on when it later makes its
        // own tail call: no return-guarantee post-work, and no live `Substrate` parameter (whose
        // scope-exit release would otherwise be skipped — the VR-5/leak hazard, §4.6-C4).
        let this_tco_eligible =
            ret_guar.is_none() && !argv.iter().any(|v| matches!(v, L1Value::Substrate(_)));

        // TCO (M-994 fix (a) — RFC-0041 §6 scoped amendment completing the §4.6 TCO intent; widens
        // M-986): elide the caller's frame iff — looking THROUGH any run
        // of binder-restoring frames (`MatchPop`/`LetPop`) — the first non-transparent frame is the
        // caller's tco-eligible `InvokePost`. `MatchPop`/`LetPop` are observationally transparent to
        // the *value* (they only restore scope), so a tail call made under them is still in tail
        // position: the call's result IS the enclosing function's result. We PEEK first (no mutation)
        // so the non-tail case is byte-for-byte the old behavior.
        let tail = {
            let mut i = regs.stack.len();
            let mut found = false;
            while i > 0 {
                match &regs.stack[i - 1] {
                    Frame::MatchPop { .. } | Frame::LetPop => i -= 1,
                    Frame::InvokePost {
                        tco_eligible: true, ..
                    } => {
                        found = true;
                        break;
                    }
                    _ => break,
                }
            }
            found
        };
        let (param_base, saved_site, saved_base) = if tail {
            // Commit: drain the transparent binder-restoring frames, executing each one's scope
            // cleanup eagerly (the M-986 "truncate the scope eagerly before a tail call"), so the
            // caller's `InvokePost` surfaces to the top. A `LetPop` still runs its M-904 scope-exit
            // release for a let-bound `Substrate` that does not escape into the call's arguments
            // (the only channel a value from this scope can reach the callee) — never a silent leak.
            loop {
                match regs.stack.last() {
                    Some(Frame::MatchPop { mark }) => {
                        let mark = *mark;
                        regs.stack.pop();
                        regs.scope.truncate(mark);
                    }
                    Some(Frame::LetPop) => {
                        regs.stack.pop();
                        let popped = regs.scope.pop().expect("let binding present");
                        if !substrate_escapes_into(&popped, &argv) {
                            self.release_if_abandoned(&popped, None);
                        }
                    }
                    _ => break,
                }
            }
            // Pop the caller's `InvokePost` (its depth guard drops — depth released) and reuse its
            // slot for this callee, threading the caller's *own* saved site/base so that when this
            // callee finally returns non-tail, control returns to the caller's caller.
            let Some(Frame::InvokePost {
                param_base: caller_param_base,
                saved_site: caller_saved_site,
                saved_base: caller_saved_base,
                ..
            }) = regs.stack.pop()
            else {
                unreachable!("just matched InvokePost on top");
            };
            self.tco_trace
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .record(fname);
            regs.scope.truncate(caller_param_base);
            (caller_param_base, caller_saved_site, caller_saved_base)
        } else {
            (regs.scope.len(), regs.site, regs.base)
        };

        for (p, a) in fd.sig.value_params.iter().zip(argv) {
            regs.scope.push((p.name.clone(), a));
        }
        regs.stack.push(Frame::InvokePost {
            param_base,
            saved_site,
            saved_base,
            ret_guar,
            nparams: fd.sig.value_params.len(),
            tco_eligible: this_tco_eligible,
            guard,
        });
        regs.site = fname;
        regs.base = param_base;
        Ctrl::Eval(&fd.body)
    }

    /// Split a `for` spine value into its head element + tail (the former `eval_for` cons/nil
    /// analysis). `Ok(None)` is the nil terminator; `Ok(Some((elem, rest)))` is a cons.
    ///
    /// **`Arc` sharing (M-994 fix (b)).** `Data.fields` is `Arc<Vec<L1Value>>`, so `elem`/`rest` are
    /// *cloned* out of the (borrowed) spine rather than moved. This is cheap: the `rest` tail is a
    /// `Data` value, so its clone is an O(1) `Arc` refcount bump (not the O(spine) deep-copy the old
    /// `mem::take` existed to avoid); the `elem` clone is a single small element value. The spine
    /// borrow keeps the tail alive, so no E0509 by-value move-out is needed at all.
    fn split_spine(
        &self,
        site: &str,
        spine: L1Value,
    ) -> Result<Option<(L1Value, L1Value)>, L1Error> {
        let L1Value::Data { ty, ctor, fields } = &spine else {
            return Err(L1Error::Stuck {
                site: site.to_owned(),
                why: "`for` spine is not a data value".to_owned(),
            });
        };
        if fields.is_empty() {
            return Ok(None); // a nil — the spine ends, the fold is the accumulator
        }
        let Some(d) = self.env.types.get(ty) else {
            return Err(L1Error::Stuck {
                site: site.to_owned(),
                why: format!("`for` over unregistered type `{ty}`"),
            });
        };
        let Some(ci) = d.ctors.iter().position(|c| c.name == *ctor) else {
            return Err(L1Error::Stuck {
                site: site.to_owned(),
                why: format!("`for` met unknown constructor `{ctor}` of `{ty}`"),
            });
        };
        let mut elem = None;
        let mut rest = None;
        for (f, v) in d.ctors[ci].fields.iter().zip(fields.iter()) {
            if matches!(f, crate::checkty::Ty::Data(n, _) if *n == *ty) {
                rest = Some(v.clone());
            } else {
                elem = Some(v.clone());
            }
        }
        let (Some(elem), Some(rest)) = (elem, rest) else {
            return Err(L1Error::Stuck {
                site: site.to_owned(),
                why: format!("`{ctor}` is not nil/cons-shaped — the checker should have refused"),
            });
        };
        Ok(Some((elem, rest)))
    }

    /// Advance the `for` fold's iterative spine walk one element (the former `eval_for` loop body):
    /// a nil terminates with the accumulator; a cons charges the per-element fuel, binds `x`/`acc`,
    /// and evaluates the body — its value becomes the next accumulator via a [`Frame::ForStep`].
    /// The walk is **iterative** (no host-stack recursion per element), so a long `for` costs fuel,
    /// never depth (RFC-0007 §4.8).
    #[allow(clippy::too_many_arguments)] // the fold threads its budgets + the form's five parts
    fn for_advance<'b>(
        &self,
        regs: &mut Regs<'e, 'b>,
        fuel: &mut u64,
        x: &'e str,
        acc: &'e str,
        body: &'e Expr,
        spine: L1Value,
        accv: L1Value,
    ) -> Ctrl<'e> {
        match self.split_spine(regs.site, spine) {
            Err(e) => Ctrl::Settle(Err(e)),
            Ok(None) => Ctrl::Settle(Ok(accv)),
            Ok(Some((elem, rest))) => {
                // Each element's body evaluation is clocked (matches the former `eval_for`).
                match fuel.checked_sub(1) {
                    Some(f) => *fuel = f,
                    None => return Ctrl::Settle(Err(L1Error::FuelExhausted)),
                }
                regs.scope.push((x.to_owned(), elem));
                regs.scope.push((acc.to_owned(), accv));
                regs.stack.push(Frame::ForStep {
                    x,
                    acc,
                    body,
                    spine: rest,
                });
                Ctrl::Eval(body)
            }
        }
    }

    /// Evaluate one hypha's already-computed `@forage(policy)` placement decision (the former
    /// `eval_hypha_forage` tail, after the policy value is in hand — M-906/DN-70 D1; RFC-0008 RT3).
    /// It consults the *real* RFC-0005 `SelectionPolicy`/[`select_placement`] machinery over the
    /// D-lite single-node candidate set the bitmask names, records the mandatory EXPLAIN decision
    /// (house rule #2), and is semantics-free (RT3: it changes no value). An all-zero bitmask is an
    /// explicit [`ForageError::NoCandidates`] (never a fabricated placement — DN-63 §3.5 / G2).
    fn forage_select(&self, site: &str, policy_value: &L1Value) -> Result<(), L1Error> {
        let L1Value::Repr(v) = policy_value else {
            return Err(L1Error::Stuck {
                site: site.to_owned(),
                why: "internal: `@forage(policy)` evaluated to a non-repr value — the checker \
                      requires a literal binary bitmask (M-906/DN-70 D1)"
                    .to_owned(),
            });
        };
        let mycelium_core::Payload::Bits(bits) = v.payload() else {
            return Err(L1Error::Stuck {
                site: site.to_owned(),
                why: "internal: `@forage(policy)` evaluated to a non-`Binary` repr — the checker \
                      requires a literal binary bitmask (M-906/DN-70 D1)"
                    .to_owned(),
            });
        };
        let candidates: Vec<Candidate> = bits
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| Candidate::Node(NodeRef(format!("worker-{i}"))))
            .collect();
        if candidates.is_empty() {
            return Err(ForageError::NoCandidates.into());
        }
        let policy = SelectionPolicy::new(
            "forage.dlite.v0",
            candidates,
            Vec::new(),
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .map_err(|e| L1Error::Stuck {
            site: site.to_owned(),
            why: format!(
                "internal: `@forage` D-lite policy construction failed unexpectedly: {e} (the \
                 caller always supplies ≥ 1 candidate and a valid cost weight)"
            ),
        })?;
        let inputs = SelectionInputs {
            src: mycelium_core::Repr::Bytes,
            guarantee: GuaranteeStrength::Declared,
            bound: None,
            sparsity: None,
            decode: None,
        };
        let (_chosen, explanation) =
            select_placement(&policy, &inputs, None).map_err(|e| L1Error::Stuck {
                site: site.to_owned(),
                why: format!(
                    "internal: `@forage` D-lite `select_placement` failed unexpectedly: {e} (the \
                     caller always supplies a `Node` candidate set)"
                ),
            })?;
        self.forage_trail
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(ForageDecision {
                site: site.to_owned(),
                explanation,
            });
        Ok(())
    }

    /// Try to match `val` against `pat`, accumulating the pattern's binders into `binds`
    /// (left-to-right, recursively for nested patterns). Returns whether it matched; on a partial
    /// nested failure the caller discards `binds`, so no rollback is needed. The
    /// constructor/literal/binder resolution mirrors the typechecker's `check_pattern` exactly, so a
    /// checked program never gets stuck (RFC-0007 §4.7).
    fn try_match(
        &self,
        site: &str,
        pat: &Pattern,
        val: &L1Value,
        binds: &mut Vec<(String, L1Value)>,
    ) -> Result<bool, L1Error> {
        match pat {
            Pattern::Wildcard => Ok(true),
            // A bare name is a nullary-constructor alternative iff it names one of the value's data
            // type's constructors; otherwise it binds the whole value.
            Pattern::Ident(n) => match val {
                L1Value::Data { ty, ctor, .. }
                    if self
                        .env
                        .types
                        .get(ty)
                        .is_some_and(|d| d.ctors.iter().any(|c| c.name == *n)) =>
                {
                    Ok(ctor == n)
                }
                _ => {
                    binds.push((n.clone(), val.clone()));
                    Ok(true)
                }
            },
            Pattern::Ctor(n, subs) => match val {
                L1Value::Data { ctor, fields, .. } => {
                    if ctor != n {
                        return Ok(false);
                    }
                    for (sub, fv) in subs.iter().zip(fields.iter()) {
                        if !self.try_match(site, sub, fv, binds)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                // A `Substrate` handle or a bare function value matches no constructor pattern (the
                // checker's type discipline keeps either off a data-ctor arm anyway); never-silent
                // `Ok(false)`, never a panic (G2).
                L1Value::Repr(_) | L1Value::Substrate(_) | L1Value::Fn(_) => Ok(false),
            },
            Pattern::Lit(lit) => match val {
                L1Value::Repr(v) => {
                    let lv = crate::elab::lit_value(site, lit).map_err(|e| L1Error::Stuck {
                        site: site.to_owned(),
                        why: format!("malformed literal pattern: {e}"),
                    })?;
                    Ok(lv.repr() == v.repr() && lv.payload() == v.payload())
                }
                // A `Substrate` handle or a bare function value has no literal form to compare
                // against — never-silent `Ok(false)` (neither has a repr/payload; G2).
                L1Value::Data { .. } | L1Value::Substrate(_) | L1Value::Fn(_) => Ok(false),
            },
            // M-826: a tuple pattern `(x, y, …)` desugars to `Ctor(MkTuple$N, subs)` during
            // checking/resolve. A raw `Pattern::Tuple` here means the evaluator was handed an
            // un-checked pattern (staging bug). Never-silent (G2): fall back to the `Ctor` path
            // by re-calling with the equivalent desugared pattern.
            Pattern::Tuple(subs) => {
                let n = subs.len();
                let ctor_name = crate::checkty::tuple_ctor_name(n);
                let desugared = Pattern::Ctor(ctor_name, subs.clone());
                self.try_match(site, &desugared, val, binds)
            }
            // `Pattern::Or` is desugared in `check_match` before evaluation; reaching here means
            // the program was not checked — an explicit never-silent refusal (G2).
            Pattern::Or(_) => Err(L1Error::Stuck {
                site: site.to_owned(),
                why: "internal: Pattern::Or reached the evaluator — or-patterns must be \
                      desugared by the checker before evaluation (invariant violation)"
                    .to_owned(),
            }),
        }
    }

    /// The stage-0 dynamic guarantee check (RFC-0007 §4.3): the value's actual tag must be **at
    /// least as strong** as the asserted index — an annotation may only weaken (VR-5). The check
    /// never modifies the value: a passing assertion leaves the (possibly stronger) tag in place,
    /// and a failing one is an explicit error, never a downgrade-and-continue.
    pub(crate) fn assert_guarantee(
        &self,
        site: &str,
        v: &L1Value,
        asserted: Strength,
    ) -> Result<(), L1Error> {
        match v {
            L1Value::Repr(value) => {
                let actual = value.meta().guarantee();
                if actual.rank() > strength_of(asserted).rank() {
                    return Err(L1Error::GuaranteeTooWeak {
                        site: site.to_owned(),
                        asserted,
                        actual,
                    });
                }
                Ok(())
            }
            L1Value::Data { .. } => Err(L1Error::Unsupported {
                site: site.to_owned(),
                what: "a guarantee index on a data-typed value has no Meta to check in v0"
                    .to_owned(),
            }),
            // A `Substrate` handle carries no `Meta`/guarantee tag (it names an external resource,
            // not a value — LR-8; DN-71 §4.1). A guarantee index on it has nothing to check: an
            // explicit refusal, never a silently-passed assertion (G2/VR-5).
            L1Value::Substrate(_) => Err(L1Error::Unsupported {
                site: site.to_owned(),
                what:
                    "a guarantee index on a `Substrate` handle has no Meta to check — a Substrate \
                       is an affine external-resource handle, not a repr value (LR-8; DN-71 §4.1)"
                        .to_owned(),
            }),
            // ADR-033/DN-74 (M-923): a bare function value carries no `Meta`/guarantee tag either
            // (it names a top-level definition, not a representation value) — an explicit refusal,
            // never a silently-passed assertion (G2/VR-5).
            L1Value::Fn(_) => Err(L1Error::Unsupported {
                site: site.to_owned(),
                what: "a guarantee index on a function value has no Meta to check in v0".to_owned(),
            }),
        }
    }
}

/// Map the shared [`RecursionBudget`]'s canonical over-budget error to the L1 evaluator's
/// [`L1Error::DepthExceeded`] at the **same threshold** (RFC-0041 §5.1 error parity). The L1 machine
/// charges only the depth ceiling, so [`BudgetError::OutOfBudget`] is never constructed on this path;
/// it is mapped defensively to the same never-silent refusal (never a panic — G2).
fn depth_exceeded(e: BudgetError) -> L1Error {
    match e {
        BudgetError::DepthExceeded { limit } => L1Error::DepthExceeded { limit },
        BudgetError::OutOfBudget { .. } => L1Error::DepthExceeded { limit: 0 },
    }
}

/// The **control** register of the L1 work-stack CEK machine (RFC-0041 §4.1): either *evaluate* an
/// expression (descend), or *settle* a produced value / propagated error up through the continuation
/// stack (ascend). One `enum` avoids two mutually-recursive loop functions — the whole machine is a
/// single `match` on this.
// `Settle` carries a value/error and `Eval` a thin ref; the machine returns `Ctrl` by value on the
// hot path, where boxing the value would add per-step heap churn for no real win (values are moved,
// not copied). The size gap is bounded by `L1Value`'s own (pinned) size.
#[allow(clippy::large_enum_variant)]
enum Ctrl<'e> {
    /// Evaluate `e` under the current `site`/`base` context.
    Eval(&'e Expr),
    /// Propagate a produced value (`Ok`) or an error (`Err`) into the top continuation [`Frame`];
    /// an `Err` unwinds the stack, running each frame's deterministic scope-exit cleanup (G2).
    Settle(Result<L1Value, L1Error>),
}

/// The machine's mutable **registers**: the shared environment (`scope` + the current function's
/// `base` marker — each function sees only `scope[base..]`, so lexical scope survives across calls),
/// the current `site` (function name, for error attribution), and the explicit continuation
/// work-stack. Bundled so the machine's step helpers thread one `&mut` instead of five.
struct Regs<'e, 'b> {
    /// The current function's name — the `site` of every error raised while its body runs.
    site: &'e str,
    /// The scope index at which the current function's parameters begin (its lexical floor).
    base: usize,
    /// The single shared environment: `(name, value)` bindings, innermost last.
    scope: Vec<(String, L1Value)>,
    /// The explicit heap continuation stack (the CEK "K").
    stack: Vec<Frame<'e, 'b>>,
}

/// One **continuation frame** of the L1 work-stack CEK machine (RFC-0041 §4.1/§4.6): the reified
/// *post-child* work for one partially-evaluated construct. Each variant names what to do once the
/// child it descended into settles — the interleaved scope push/pop, `release_if_abandoned`,
/// return-guarantee assert, swap/ascription checks, and (for `App`) the source-call depth
/// reservation ([`DepthGuard`]) that a former recursive stack frame carried implicitly.
#[allow(clippy::large_enum_variant)] // frames are short-lived + depth-bounded (<= budget); boxing
                                     // every value-carrying variant would add per-frame heap churn on the eval hot path for no real win.
enum Frame<'e, 'b> {
    /// A list literal: `elems[idx]` just evaluated → collect its repr, evaluate the next, or build
    /// the `Seq`.
    ListElem {
        elems: &'e [Expr],
        idx: usize,
        vals: Vec<Value>,
    },
    /// A `let`: the bound value just evaluated → assert its optional guarantee index, push the
    /// binding, evaluate the body.
    LetBound {
        name: &'e str,
        ty_guar: Option<Strength>,
        body: &'e Expr,
    },
    /// A `let` body just evaluated → pop the binding + `release_if_abandoned` (M-904 scope-exit).
    LetPop,
    /// An `if` condition just evaluated → choose the consequent/alternate branch (transparent: no
    /// value transform, so the chosen branch stays TCO-eligible).
    IfBranch { conseq: &'e Expr, alt: &'e Expr },
    /// A `match` scrutinee just evaluated → try the arms' patterns.
    MatchArms { arms: &'e [crate::ast::Arm] },
    /// A `match` arm body just evaluated → truncate the scope back to `mark` (drop the arm binders).
    MatchPop { mark: usize },
    /// A `swap` source just evaluated → run the certified swap + optional guarantee assert.
    SwapPost {
        target: &'e crate::ast::TypeRef,
        policy: &'e crate::ast::Path,
    },
    /// A `consume` operand just evaluated → the checked affine Live→Consumed move.
    ConsumePost,
    /// An ascription's inner value just evaluated → assert the guarantee index.
    AscribePost { guar: Strength },
    /// `fuse`'s left operand just evaluated → evaluate the right (carrying the left value + exprs).
    FuseAfterLeft {
        left_expr: &'e Expr,
        right_expr: &'e Expr,
    },
    /// `fuse`'s right operand just evaluated → combine (repr prim) or begin the data `join` dispatch.
    FuseAfterRight {
        lv: L1Value,
        left_expr: &'e Expr,
        right_expr: &'e Expr,
    },
    /// The data-`fuse` `join` call: its (re-evaluated) left operand just evaluated → evaluate right.
    FuseJoinLeft {
        right_expr: &'e Expr,
        guard: DepthGuard<'b>,
    },
    /// The data-`fuse` `join` call: its right operand just evaluated → dispatch `join(left, right)`.
    FuseJoinRight { lv2: L1Value, guard: DepthGuard<'b> },
    /// A `reclaim` policy just evaluated (discarded) → evaluate the supervised body.
    ReclaimBody { body: &'e Expr },
    /// A `colony` hypha's `@forage` policy just evaluated → record the placement decision, then the
    /// hypha's body.
    ColonyForage { hyphae: &'e [Hypha], idx: usize },
    /// A `colony` hypha's body just evaluated → the last hypha's value is the result; a leading
    /// hypha's value is discarded and the next hypha begins.
    ColonyBody { hyphae: &'e [Hypha], idx: usize },
    /// A `wild` host-op argument just evaluated → collect it, evaluate the next, or dispatch the op.
    WildArgs {
        key: String,
        args: &'e [Expr],
        idx: usize,
        argv: Vec<L1Value>,
    },
    /// An application argument just evaluated → collect it, evaluate the next, or dispatch the call.
    /// Holds the App's source-call [`DepthGuard`] (charged at the App boundary, §4.0), moved into the
    /// callee's [`Self::InvokePost`] for a user call or dropped once a ctor/prim result is built.
    AppArgs {
        name: &'e str,
        args: &'e [Expr],
        idx: usize,
        argv: Vec<L1Value>,
        guard: DepthGuard<'b>,
    },
    /// A function body just evaluated → release the call's parameter scope (M-904), assert the
    /// return-guarantee index, and restore the caller's `site`/`base`. `tco_eligible` records whether
    /// *this* frame may be elided when its function makes a direct tail call (no pending post-work);
    /// `guard` is the call's depth reservation, released when this frame is dropped.
    InvokePost {
        param_base: usize,
        saved_site: &'e str,
        saved_base: usize,
        ret_guar: Option<Strength>,
        nparams: usize,
        tco_eligible: bool,
        // Held purely for its `Drop` — releasing this call's source-call depth unit when the frame
        // is dropped (on return, TCO elision, or error unwind). Never read (RAII), hence the allow.
        #[allow(dead_code)]
        guard: DepthGuard<'b>,
    },
    /// A `for`'s spine (`xs`) just evaluated → evaluate the initial accumulator (`init`).
    ForAfterXs {
        x: &'e str,
        acc: &'e str,
        init: &'e Expr,
        body: &'e Expr,
    },
    /// A `for`'s initial accumulator just evaluated → begin the iterative spine walk.
    ForAfterInit {
        x: &'e str,
        acc: &'e str,
        body: &'e Expr,
        spine: L1Value,
    },
    /// A `for` body just evaluated (the next accumulator) → pop the element binders, advance the
    /// spine to `spine` (the tail), and either finish or evaluate the body for the next element.
    ForStep {
        x: &'e str,
        acc: &'e str,
        body: &'e Expr,
        spine: L1Value,
    },
}

/// Forward a bridge refusal (shared with elaboration) as an explicit evaluator refusal.
fn unsupported(site: &str, e: &ElabError) -> L1Error {
    L1Error::Unsupported {
        site: site.to_owned(),
        what: e.to_string(),
    }
}
