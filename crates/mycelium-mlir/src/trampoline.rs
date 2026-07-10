//! Direct-LLVM **heap-trampoline** lowering for full object-level recursion — non-tail single
//! `Fix` and mutual-recursion `FixGroup` — over the narrow `Binary{8}` ABI (M-850; epic E25-1;
//! ADR-034; RFC-0004 §2/§11.6; DN-15 §4.3/§5-table Inc-3/§8.5/§10).
//!
//! ## What this is, and why it is a separate file
//!
//! [`crate::llvm`] already lowers **tail-position** single `Fix` to an iterative LLVM loop
//! ([`crate::llvm::lower_tail_fix`]) — the host C stack is O(1) by construction (DN-15 §8.1). That
//! covers the tail fragment but **refuses** (never-silent `AotError::UnsupportedNode`) non-tail
//! recursion and `FixGroup`. This module is the **full defunctionalized heap trampoline** DN-15 §4.3
//! anticipated: it runs object-level recursion on an **explicit heap control stack** (a `@malloc`'d
//! frame stack), *not* the C stack, so a deep / non-terminating recursion is bounded by the
//! **same [`AutoDepthBudget`]** the env-machine uses (DRY/KC-3 — reused, not re-invented; M-349) and
//! refuses gracefully with the [`crate::llvm::DEPTHLIMIT_SENTINEL`] read-back, never a SIGSEGV or a
//! hang (DN-05 #1; G2/SC-3). Keeping it file-disjoint from the tail-loop code is the M-850 ownership
//! contract (the tail loop stays untouched and keeps its byte-for-byte IR).
//!
//! ## The machine (mirror of [`crate::aot`]'s `Vec<Frame>` trampoline, in emitted IR)
//!
//! The env-machine ([`crate::aot::run`]) runs recursion as a Rust loop over a `Vec<Frame>` control
//! stack: `App` pushes a continuation frame and switches blocks, a completed block pops a frame and
//! resumes. This module emits the **LLVM-IR analogue** of exactly that machine, defunctionalized:
//!
//! - Every recursion **member** (the single `Fix`, or each `FixGroup` member) gets an integer
//!   `member id`. Each member is the canonical `λparam. Match param { Lit-arms / default }` shape.
//! - Each arm's body is analyzed (never the C stack) into a **linear plan**: pre-call straight-line
//!   `Binary{8}` bindings, then either a **base** result, or a **call** `App(callee, step)` whose
//!   result is fed to a **defunctionalized continuation** — at most one *pending unary/binary
//!   `Binary{8}` op* wrapping that call (the "linear non-tail recursion" fragment). A **tail** call is
//!   the degenerate continuation "return the callee's result unchanged".
//! - The trampoline is one LLVM loop with a `@malloc`'d **frame stack**: each frame stores a
//!   *resume continuation id* + a *saved `i64` operand* (the defunctionalized pending op). A call
//!   **pushes** a frame (checking depth) and re-enters dispatch with the callee id + step; a base /
//!   completed continuation **pops** a frame and applies its pending op to the result, looping until
//!   the stack is empty (the final result).
//!
//! Anything outside this shape — a self/sibling reference that is not a saturated `App(callee, step)`
//! whose continuation is a single pending `Binary{8}` op, a Ctor arm on the recursion param, a
//! non-`Binary{8}` width, trit arithmetic in a recursive body, a nested `Fix`/`Lam` in a recursive
//! body, more than one call per arm — is an explicit [`AotError::UnsupportedNode`] (never a silent
//! mis-lowering, never an upgraded guarantee — G2/VR-5). The boundary is **honest**: the reference
//! interpreter still evaluates the refused program; the native path simply declines it and routes to
//! the interpreter.
//!
//! ## Honesty (VR-5)
//! Guarantee tag **Empirical** — hand-written textual LLVM IR with a *checked* empirical basis, not a
//! proof. The basis (M-850): the **interp ≡ direct-LLVM** differential over the recursion corpus
//! (`tests/recursion_trampoline_differential.rs` — non-tail `Cont::{Not,And,Or,Xor}`, two stacked
//! frames, `FixGroup` mutual recursion, deep-recursion `DepthLimit`) is green, **and** a
//! `cargo-mutants` witness of the frame/continuation logic is caught by it (`emit_apply_cont` and
//! `materialize_saved` mutants caught by value-divergence; `emit_push_frame`/`emit_bump_depth → ()`
//! caught by the deep-recursion test no longer terminating — 0 missed on that core). It is **not**
//! `Proven`: there is no machine-checked refinement theorem for the emitted IR, so the tag is never
//! upgraded past `Empirical` (VR-5). The MLIR-dialect leg does **not** run for this corpus —
//! `dialect::native` honestly refuses recursion (`Fix`/`FixGroup` → `UnsupportedNode`), so the
//! differential is two-way (interp ≡ direct-LLVM) with the dialect edge an explicit refusal, not a
//! skipped/vacuous pass.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced (the frame stack is a
//! safe `@malloc`/`@free` heap structure in emitted IR, like the Increment-2 arena).
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt::Write as _;

use mycelium_core::lower::{Anf, AnfAlt, Atom, Rhs};
use mycelium_core::Value;

use crate::budget::DepthBudget as _;
use crate::llvm::{
    as_binary8_pub, lit_binary8_packed_pub, pack_binary8_pub, unpack_binary8_pub, AotError, Bbc,
    EnvValue, Lane, LaneKind, Ssa, CLOSURE_ABI_WIDTH, DEPTHLIMIT_SENTINEL,
};

/// One recursion **member**: its self/own name, the parameter, and its tail-`Match` arms — the
/// canonical `λparam. Match param { … }` shape, already destructured. A single `Fix` is a group of
/// one member; a `FixGroup` is N members sharing one trampoline.
pub(crate) struct Member {
    /// The member's bound name (the self-name for `Fix`; the member name for a `FixGroup` def).
    name: String,
    /// The `λ` parameter (the recursion accumulator; scrutinee of the body `Match`).
    param: String,
    /// `(packed-literal-key, arm-body)` for each `Lit` arm, plus the optional `default` body.
    arms: Vec<(u64, Anf)>,
    /// The `default` arm body, if any.
    default: Option<Anf>,
}

/// A **defunctionalized continuation** applied to a callee's `i64` result: the pending `Binary{8}`
/// op (if any) plus its saved operand atom, encoded so a heap frame can store it as two `i64`s
/// (`tag`, `saved`). The op set is exactly the `Binary{8}` element-wise primitives the narrow ABI
/// supports; `Identity` is the tail-call continuation (return the result unchanged).
///
/// The `Atom` operand is the **analysis-phase** reference into the arm env; it is resolved to a
/// concrete packed-`i64` SSA register only at emission time ([`materialize_saved`]) — so a `Temp`
/// operand is looked up correctly (the prior `String`-name modelling mis-resolved `Temp` atoms).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Cont {
    /// Tail call — return the callee's result unchanged (no pending op). Frame tag `0`.
    Identity,
    /// `bit.not(result)` — unary. Frame tag `1`.
    Not,
    /// `bit.and(result, saved-operand-atom)` — binary. Tag `2`.
    And(Atom),
    /// `bit.or(result, saved-operand-atom)`. Tag `3`.
    Or(Atom),
    /// `bit.xor(result, saved-operand-atom)`. Tag `4`.
    Xor(Atom),
}

impl Cont {
    /// The integer tag stored in a frame's continuation slot (defunctionalization key).
    fn tag(&self) -> u64 {
        match self {
            Cont::Identity => 0,
            Cont::Not => 1,
            Cont::And(_) => 2,
            Cont::Or(_) => 3,
            Cont::Xor(_) => 4,
        }
    }
    /// The saved-operand atom for a binary op, or `None` for identity/unary.
    fn operand(&self) -> Option<&Atom> {
        match self {
            Cont::And(a) | Cont::Or(a) | Cont::Xor(a) => Some(a),
            _ => None,
        }
    }
}

/// The analyzed plan of one arm: the pre-call straight-line bindings to lower, then the terminal —
/// either a base result lane, or a call to `callee` with a `step` atom feeding a `Cont`.
enum ArmPlan {
    /// Base case: no self/sibling reference; the arm result is a straight-line `Binary{8}` lane.
    Base,
    /// A (tail or non-tail) call: `App(callee_member, step_atom)`, result fed to `cont`.
    Call {
        /// The member index (into the group) being called — self or a sibling.
        callee: usize,
        /// The argument atom (the recursion step), bound by the pre-call bindings.
        step: Atom,
        /// The name of the binding holding the `App(callee, step)` — the cut point that separates the
        /// pre-call straight-line bindings (lowered normally) from the call (lowered by the
        /// trampoline). For a tail call this is the arm result; for a non-tail call it is an earlier
        /// binding whose result the wrapping op consumes.
        call_name: Atom,
        /// The defunctionalized continuation applied to the callee's result.
        cont: Cont,
    },
}

/// Entry point: lower `App(group, init)` where `group` is a non-tail single `Fix` or a `FixGroup`,
/// to a heap-trampoline LLVM loop, returning the result lane as an [`EnvValue`]. `group_members` is
/// the destructured group (≥1 member, canonical shape); `entry` is the index of the applied member;
/// `init_atom` is the initial accumulator (must be `Binary{8}` in the env).
///
/// Out-of-scope shapes are explicit [`AotError::UnsupportedNode`] (G2). The depth ceiling is the
/// shared [`crate::budget::AutoDepthBudget`] (M-349, DRY) → graceful [`AotError::DepthLimit`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_recursion_group(
    members: &[Member],
    entry: usize,
    init_atom: &Atom,
    env: &HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
) -> Result<EnvValue, AotError> {
    // ── Analyze every member's arms into plans (no IR yet — fail fast on out-of-scope shapes). ──
    let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
    let mut member_plans: Vec<Vec<(Option<u64>, Anf, ArmPlan)>> = Vec::with_capacity(members.len());
    for m in members {
        let mut plans: Vec<(Option<u64>, Anf, ArmPlan)> = Vec::new();
        for (key, arm) in &m.arms {
            let plan = analyze_arm(arm, &names, &m.param)?;
            plans.push((Some(*key), arm.clone(), plan));
        }
        if let Some(def) = &m.default {
            let plan = analyze_arm(def, &names, &m.param)?;
            plans.push((None, def.clone(), plan));
        } else {
            // No default — a no-match must trap (defined-trap @abort), never raw UB (G2). We model it
            // as a synthetic arm handled in dispatch emission below (no plan needed).
        }
        member_plans.push(plans);
    }

    // ── Resolve the shared depth ceiling once (the SAME trait the env-machine uses; DRY/KC-3). ──
    let ceiling = crate::budget::AutoDepthBudget::default()
        .resolve()
        .max_depth;

    // ── Pack the initial accumulator to i64. ────────────────────────────────────────────────────
    let init_ev = env
        .get(init_atom)
        .ok_or_else(|| AotError::FreeVariable(init_atom.render()))?;
    let init_lane = as_binary8_pub(init_ev, "trampoline init argument")?.clone();
    let init_packed = pack_binary8_pub(&init_lane, ssa, body);

    // ── Frame-stack runtime: a @malloc'd array of {cont_tag:i64, saved:i64} pairs. We emit a single
    //    dedicated stack per group (kept distinct from the closure arena). The stack pointer and a
    //    depth counter live in `alloca` cells so the trampoline loop can mutate them across blocks
    //    without phi gymnastics — all values are local, no `unsafe`, fully dumpable (RFC-0004 §6).
    let stack_bytes = ceiling.saturating_mul(16).max(16); // 2×i64 per frame
    let sp = ssa.fresh(); // alloca i64 — frame count (also the depth)
    let stk = ssa.fresh(); // i8* base of the frame array
    let cur = ssa.fresh(); // alloca i64 — current member id
    let acc = ssa.fresh(); // alloca i64 — current accumulator (packed Binary{8})
    let res = ssa.fresh(); // alloca i64 — the value being returned up the stack
    let _ = writeln!(
        body,
        "  ; ── M-850 heap-trampoline (non-tail Fix / FixGroup) ──"
    );
    let _ = writeln!(body, "  {sp} = alloca i64, align 8");
    let _ = writeln!(body, "  store i64 0, i64* {sp}");
    let _ = writeln!(body, "  {cur} = alloca i64, align 8");
    let _ = writeln!(body, "  store i64 {entry}, i64* {cur}");
    let _ = writeln!(body, "  {acc} = alloca i64, align 8");
    let _ = writeln!(body, "  store i64 {init_packed}, i64* {acc}");
    let _ = writeln!(body, "  {res} = alloca i64, align 8");
    let _ = writeln!(
        body,
        "  {stk} = call i8* @myc_tramp_alloc(i64 {stack_bytes})"
    );

    // ── Labels for the dispatch state machine. ──────────────────────────────────────────────────
    let l_call = bbc.fresh(); // CALL state: depth-check + enter member dispatch
    let l_depthlimit = bbc.fresh(); // graceful depth-limit exit
    let l_dispatch = bbc.fresh(); // switch on current member id
    let l_return = bbc.fresh(); // RETURN state: pop a frame + apply its continuation
    let l_done = bbc.fresh(); // stack empty → final result
    let l_unwind = bbc.fresh(); // apply popped continuation, then loop to RETURN

    // Per-member dispatch blocks.
    let member_labels: Vec<String> = (0..members.len()).map(|_| bbc.fresh()).collect();

    // Enter the CALL state with (cur=entry, acc=init, sp=0).
    let _ = writeln!(body, "  br label %{l_call}");

    // ── CALL state ───────────────────────────────────────────────────────────────────────────────
    // depth check, then switch to the current member's dispatch block.
    let _ = writeln!(body, "{l_call}:");
    let d = ssa.fresh();
    let _ = writeln!(body, "  {d} = load i64, i64* {sp}");
    let over = ssa.fresh();
    let _ = writeln!(body, "  {over} = icmp uge i64 {d}, {ceiling}");
    let _ = writeln!(
        body,
        "  br i1 {over}, label %{l_depthlimit}, label %{l_dispatch}"
    );

    // ── DepthLimit block (graceful, never SIGSEGV/hang; the env-machine DepthLimit twin — G2). ──
    let _ = writeln!(body, "{l_depthlimit}:");
    let _ = writeln!(body, "  call void @myc_tramp_free(i8* {stk})");
    let dl1 = ssa.fresh();
    let _ = writeln!(
        body,
        "  {dl1} = call i32 @putchar(i32 {})",
        DEPTHLIMIT_SENTINEL
    );
    let dl2 = ssa.fresh();
    let _ = writeln!(body, "  {dl2} = call i32 @putchar(i32 10)");
    let _ = writeln!(body, "  ret i32 0");

    // ── DISPATCH: switch on the current member id → that member's block. ─────────────────────────
    let _ = writeln!(body, "{l_dispatch}:");
    let cm = ssa.fresh();
    let _ = writeln!(body, "  {cm} = load i64, i64* {cur}");
    let _ = write!(body, "  switch i64 {cm}, label %{} [", member_labels[0]);
    for (i, lbl) in member_labels.iter().enumerate() {
        let _ = write!(body, " i64 {i}, label %{lbl}");
    }
    let _ = writeln!(body, " ]");

    // ── Each member's dispatch block: switch on the accumulator literal → that arm's plan. ───────
    for (mi, (m, plans)) in members.iter().zip(&member_plans).enumerate() {
        emit_member_block(
            mi,
            m,
            plans,
            &member_labels[mi],
            &l_call,
            &l_return,
            sp.as_str(),
            stk.as_str(),
            cur.as_str(),
            acc.as_str(),
            res.as_str(),
            ssa,
            bbc,
            body,
            funcs,
            flags,
            env,
        )?;
    }

    // ── RETURN state: if stack empty → done; else pop a frame, apply its continuation. ───────────
    let _ = writeln!(body, "{l_return}:");
    let d2 = ssa.fresh();
    let _ = writeln!(body, "  {d2} = load i64, i64* {sp}");
    let empty = ssa.fresh();
    let _ = writeln!(body, "  {empty} = icmp eq i64 {d2}, 0");
    let _ = writeln!(body, "  br i1 {empty}, label %{l_done}, label %{l_unwind}");

    // ── UNWIND: pop the top frame (decrement sp), load its (cont_tag, saved), apply to res. ──────
    let _ = writeln!(body, "{l_unwind}:");
    let d3 = ssa.fresh();
    let _ = writeln!(body, "  {d3} = load i64, i64* {sp}");
    let newsp = ssa.fresh();
    let _ = writeln!(body, "  {newsp} = sub i64 {d3}, 1");
    let _ = writeln!(body, "  store i64 {newsp}, i64* {sp}");
    // frame base = stk + newsp*16. Load cont_tag (slot 0) and saved (slot 1).
    let fbyte = ssa.fresh();
    let _ = writeln!(body, "  {fbyte} = mul i64 {newsp}, 16");
    let fptr = ssa.fresh();
    let _ = writeln!(body, "  {fptr} = getelementptr i8, i8* {stk}, i64 {fbyte}");
    let fi64 = ssa.fresh();
    let _ = writeln!(body, "  {fi64} = bitcast i8* {fptr} to i64*");
    let tag_reg = ssa.fresh();
    let _ = writeln!(body, "  {tag_reg} = load i64, i64* {fi64}");
    let saved_gep = ssa.fresh();
    let _ = writeln!(
        body,
        "  {saved_gep} = getelementptr i64, i64* {fi64}, i64 1"
    );
    let saved_reg = ssa.fresh();
    let _ = writeln!(body, "  {saved_reg} = load i64, i64* {saved_gep}");
    let r_in = ssa.fresh();
    let _ = writeln!(body, "  {r_in} = load i64, i64* {res}");
    // Apply the defunctionalized continuation to r_in by switching on tag_reg.
    let applied = emit_apply_cont(&tag_reg, &saved_reg, &r_in, ssa, bbc, body);
    let _ = writeln!(body, "  store i64 {applied}, i64* {res}");
    // The applied result becomes the value being returned — loop back to RETURN to unwind further.
    let _ = writeln!(body, "  br label %{l_return}");

    // ── DONE: the stack is empty — `res` holds the final accumulator-result. ─────────────────────
    let _ = writeln!(body, "{l_done}:");
    let _ = writeln!(body, "  call void @myc_tramp_free(i8* {stk})");
    let final_packed = ssa.fresh();
    let _ = writeln!(body, "  {final_packed} = load i64, i64* {res}");
    let result_lane = unpack_binary8_pub(&final_packed, ssa, body);
    Ok(EnvValue::Repr(result_lane))
}

/// Emit one member's dispatch block: switch on the accumulator literal → per-arm plan blocks. A
/// base arm stores its result and branches to RETURN; a call arm pushes a continuation frame (if
/// non-identity) and branches to CALL with the callee id + step.
#[allow(clippy::too_many_arguments)]
fn emit_member_block(
    _mi: usize,
    m: &Member,
    plans: &[(Option<u64>, Anf, ArmPlan)],
    member_label: &str,
    l_call: &str,
    l_return: &str,
    sp: &str,
    stk: &str,
    cur: &str,
    acc: &str,
    res: &str,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    env: &HashMap<Atom, EnvValue>,
) -> Result<(), AotError> {
    let _ = writeln!(body, "{member_label}:");
    let acc_reg = ssa.fresh();
    let _ = writeln!(body, "  {acc_reg} = load i64, i64* {acc}");

    // Split plans into literal arms (with a key) and the default (key None).
    let lit_arms: Vec<&(Option<u64>, Anf, ArmPlan)> =
        plans.iter().filter(|(k, _, _)| k.is_some()).collect();
    let default_arm: Option<&(Option<u64>, Anf, ArmPlan)> =
        plans.iter().find(|(k, _, _)| k.is_none());

    let arm_labels: Vec<String> = (0..lit_arms.len()).map(|_| bbc.fresh()).collect();
    let default_label = bbc.fresh();

    let _ = write!(body, "  switch i64 {acc_reg}, label %{default_label} [");
    for (arm, lbl) in lit_arms.iter().zip(&arm_labels) {
        let key = arm.0.expect("lit arm has a key");
        let _ = write!(body, " i64 {key}, label %{lbl}");
    }
    let _ = writeln!(body, " ]");

    for (arm, lbl) in lit_arms.iter().zip(&arm_labels) {
        emit_arm(
            &arm.1, &arm.2, &m.param, lbl, l_call, l_return, sp, stk, cur, acc, res, &acc_reg, ssa,
            bbc, body, funcs, flags, env,
        )?;
    }

    // Default arm: lower its plan, or trap (defined-trap @abort, never raw unreachable — G2).
    let _ = writeln!(body, "{default_label}:");
    match default_arm {
        Some((_, anf, plan)) => {
            emit_arm_inner(
                anf, plan, &m.param, l_call, l_return, sp, stk, cur, acc, res, &acc_reg, ssa, bbc,
                body, funcs, flags, env,
            )?;
        }
        None => {
            let _ = writeln!(body, "  call void @abort()");
            let _ = writeln!(body, "  ret i32 0");
        }
    }
    Ok(())
}

/// Emit the labelled arm block then its inner plan.
#[allow(clippy::too_many_arguments)]
fn emit_arm(
    anf: &Anf,
    plan: &ArmPlan,
    param: &str,
    label: &str,
    l_call: &str,
    l_return: &str,
    sp: &str,
    stk: &str,
    cur: &str,
    acc: &str,
    res: &str,
    acc_reg: &str,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    env: &HashMap<Atom, EnvValue>,
) -> Result<(), AotError> {
    let _ = writeln!(body, "{label}:");
    emit_arm_inner(
        anf, plan, param, l_call, l_return, sp, stk, cur, acc, res, acc_reg, ssa, bbc, body, funcs,
        flags, env,
    )
}

/// Emit an arm's plan (assuming the block label is already open): bind `param`←acc, lower the
/// pre-call straight-line bindings, then the terminal (base store→RETURN, or push-frame→CALL).
#[allow(clippy::too_many_arguments)]
fn emit_arm_inner(
    anf: &Anf,
    plan: &ArmPlan,
    param: &str,
    l_call: &str,
    l_return: &str,
    sp: &str,
    stk: &str,
    cur: &str,
    acc: &str,
    res: &str,
    acc_reg: &str,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
    funcs: &mut Vec<String>,
    flags: &mut Vec<String>,
    env: &HashMap<Atom, EnvValue>,
) -> Result<(), AotError> {
    // Bind the param to the current accumulator (unpacked Binary{8}).
    let param_lane = unpack_binary8_pub(acc_reg, ssa, body);
    let mut arm_env = env.clone();
    arm_env.insert(Atom::Named(param.to_owned()), EnvValue::Repr(param_lane));

    match plan {
        ArmPlan::Base => {
            // Lower the full arm body as a straight-line Binary{8} block; store its result into `res`.
            let result_lane =
                crate::llvm::lower_anf_block_pub(anf, &mut arm_env, ssa, bbc, body, funcs, flags)?;
            require_binary8(&result_lane, "trampoline base-arm result")?;
            let packed = pack_binary8_pub(&result_lane, ssa, body);
            let _ = writeln!(body, "  store i64 {packed}, i64* {res}");
            let _ = writeln!(body, "  br label %{l_return}");
        }
        ArmPlan::Call {
            callee,
            step,
            call_name,
            cont,
        } => {
            // Lower the straight-line *support* bindings so the `step` atom and the continuation's
            // saved operand are available. Every binding except the recursive-call binding
            // (`call_name`) and the wrapping-op result binding (`anf.result()`) is lowered — the
            // trampoline emits those two itself (the call + the defunctionalized continuation). This
            // must lower the saved operand even when it is bound *after* the call in ANF order (e.g.
            // `%c=App(self,%s); %k=Const(mask); %r=and(%c,%k)`), so we skip the two and lower the rest
            // rather than stopping at the call.
            crate::llvm::lower_bindings_before_call_pub(
                anf,
                call_name,
                anf.result(),
                &mut arm_env,
                ssa,
                bbc,
                body,
                funcs,
                flags,
            )?;
            // Materialize the saved continuation operand (binary op) NOW (it depends on pre-call
            // bindings); pack to i64. Identity/unary have no saved operand ⇒ "0".
            let saved_reg = materialize_saved(cont, &arm_env, ssa, body)?;
            // If the continuation is non-identity, push a frame {cont_tag, saved}; a tail (identity)
            // call pushes no frame but still advances the depth counter (so a non-terminating tail
            // recursion hits the SAME graceful DepthLimit — never an unbounded loop; G2/DN-05 #1).
            if cont.tag() != 0 {
                emit_push_frame(cont.tag(), &saved_reg, sp, stk, ssa, body);
            } else {
                emit_bump_depth(sp, ssa, body);
            }
            // Set cur←callee, acc←step, loop to CALL.
            let step_ev = arm_env
                .get(step)
                .ok_or_else(|| AotError::FreeVariable(step.render()))?;
            let step_lane = as_binary8_pub(step_ev, "trampoline call step")?.clone();
            let step_packed = pack_binary8_pub(&step_lane, ssa, body);
            let _ = writeln!(body, "  store i64 {callee}, i64* {cur}");
            let _ = writeln!(body, "  store i64 {step_packed}, i64* {acc}");
            let _ = writeln!(body, "  br label %{l_call}");
        }
    }
    Ok(())
}

/// A tail call pushes no frame but must still advance the depth counter (so a non-terminating tail
/// recursion reaches the graceful `DepthLimit` exactly as a non-tail one does — never an unbounded
/// loop; G2/DN-05 #1).
fn emit_bump_depth(sp: &str, ssa: &mut Ssa, body: &mut String) {
    let d = ssa.fresh();
    let _ = writeln!(body, "  {d} = load i64, i64* {sp}");
    let d1 = ssa.fresh();
    let _ = writeln!(body, "  {d1} = add i64 {d}, 1");
    let _ = writeln!(body, "  store i64 {d1}, i64* {sp}");
}

/// Push a continuation frame `{cont_tag, saved_reg}` onto the heap frame stack and increment `sp`
/// (the depth counter). The depth-limit guard in the CALL state catches an over-deep stack *before*
/// the next dispatch, so a push past the ceiling is never reached as a real store (the guard runs
/// first), and the frame array is sized `ceiling × 16` so this store is always in-bounds.
fn emit_push_frame(
    tag: u64,
    saved_reg: &str,
    sp: &str,
    stk: &str,
    ssa: &mut Ssa,
    body: &mut String,
) {
    let d = ssa.fresh();
    let _ = writeln!(body, "  {d} = load i64, i64* {sp}");
    let fbyte = ssa.fresh();
    let _ = writeln!(body, "  {fbyte} = mul i64 {d}, 16");
    let fptr = ssa.fresh();
    let _ = writeln!(body, "  {fptr} = getelementptr i8, i8* {stk}, i64 {fbyte}");
    let fi64 = ssa.fresh();
    let _ = writeln!(body, "  {fi64} = bitcast i8* {fptr} to i64*");
    let _ = writeln!(body, "  store i64 {tag}, i64* {fi64}");
    let saved_gep = ssa.fresh();
    let _ = writeln!(
        body,
        "  {saved_gep} = getelementptr i64, i64* {fi64}, i64 1"
    );
    let _ = writeln!(body, "  store i64 {saved_reg}, i64* {saved_gep}");
    let d1 = ssa.fresh();
    let _ = writeln!(body, "  {d1} = add i64 {d}, 1");
    let _ = writeln!(body, "  store i64 {d1}, i64* {sp}");
}

/// Emit the application of a defunctionalized continuation (`tag`, `saved`) to a callee result
/// `r_in`, returning the result register. A `switch` on the tag selects the pending `Binary{8}` op;
/// `phi` merges. Identity passes the result through.
fn emit_apply_cont(
    tag: &str,
    saved: &str,
    r_in: &str,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    body: &mut String,
) -> String {
    let l_id = bbc.fresh();
    let l_not = bbc.fresh();
    let l_and = bbc.fresh();
    let l_or = bbc.fresh();
    let l_xor = bbc.fresh();
    let l_merge = bbc.fresh();
    let _ = writeln!(
        body,
        "  switch i64 {tag}, label %{l_id} [ i64 0, label %{l_id} i64 1, label %{l_not} \
         i64 2, label %{l_and} i64 3, label %{l_or} i64 4, label %{l_xor} ]"
    );
    // Identity.
    let _ = writeln!(body, "{l_id}:");
    let _ = writeln!(body, "  br label %{l_merge}");
    // Not: xor with the low-8-bits mask 255 (per-bit complement of the packed Binary{8}).
    let _ = writeln!(body, "{l_not}:");
    let v_not = ssa.fresh();
    let _ = writeln!(body, "  {v_not} = xor i64 {r_in}, 255");
    let _ = writeln!(body, "  br label %{l_merge}");
    // And / Or / Xor with the saved operand.
    let _ = writeln!(body, "{l_and}:");
    let v_and = ssa.fresh();
    let _ = writeln!(body, "  {v_and} = and i64 {r_in}, {saved}");
    let _ = writeln!(body, "  br label %{l_merge}");
    let _ = writeln!(body, "{l_or}:");
    let v_or = ssa.fresh();
    let _ = writeln!(body, "  {v_or} = or i64 {r_in}, {saved}");
    let _ = writeln!(body, "  br label %{l_merge}");
    let _ = writeln!(body, "{l_xor}:");
    let v_xor = ssa.fresh();
    let _ = writeln!(body, "  {v_xor} = xor i64 {r_in}, {saved}");
    let _ = writeln!(body, "  br label %{l_merge}");
    // Merge.
    let _ = writeln!(body, "{l_merge}:");
    let phi = ssa.fresh();
    let _ = writeln!(
        body,
        "  {phi} = phi i64 [ {r_in}, %{l_id} ], [ {v_not}, %{l_not} ], [ {v_and}, %{l_and} ], \
         [ {v_or}, %{l_or} ], [ {v_xor}, %{l_xor} ]"
    );
    phi
}

/// Materialize a [`Cont`]'s runtime saved-operand register from the arm env (the operand may be a
/// pre-call binding, `Named` or `Temp`), returning the packed-`i64` SSA register — or `"0"` for an
/// identity/unary continuation (which stores an ignored saved slot). Explicit refusal if the operand
/// is missing or not a `Binary{8}` value (never a silent mis-encode — G2).
fn materialize_saved(
    cont: &Cont,
    env: &HashMap<Atom, EnvValue>,
    ssa: &mut Ssa,
    body: &mut String,
) -> Result<String, AotError> {
    match cont.operand() {
        None => Ok("0".to_owned()),
        Some(a) => {
            let ev = env
                .get(a)
                .ok_or_else(|| AotError::FreeVariable(a.render()))?;
            let lane = as_binary8_pub(ev, "trampoline continuation operand")?.clone();
            Ok(pack_binary8_pub(&lane, ssa, body))
        }
    }
}

/// Require a lane to be the `Binary{8}` ABI shape (explicit refusal otherwise — G2).
fn require_binary8(lane: &Lane, ctx: &str) -> Result<(), AotError> {
    if lane.kind != LaneKind::Binary || lane.vals.len() != CLOSURE_ABI_WIDTH {
        return Err(AotError::UnsupportedNode(format!(
            "{ctx}: the trampoline ABI carries only Binary{{{CLOSURE_ABI_WIDTH}}} values; got \
             {:?} width {}",
            lane.kind,
            lane.vals.len()
        )));
    }
    Ok(())
}

/// Analyze one arm body into an [`ArmPlan`]: classify as a base case (no member reference) or a call
/// (a single saturated `App(callee, step)` whose continuation is at most one pending `Binary{8}` op).
/// Anything else is an explicit [`AotError::UnsupportedNode`] (G2/VR-5).
fn analyze_arm(anf: &Anf, names: &[&str], _param: &str) -> Result<ArmPlan, AotError> {
    // Does any member name appear anywhere in the arm?
    let any_ref = names.iter().any(|n| anf_refs_any(anf, n));
    if !any_ref {
        return Ok(ArmPlan::Base);
    }

    // The arm result atom must be bound by either:
    //   (a) App(member, step)                            → tail call (Cont::Identity)
    //   (b) Op{prim, [App-result]} / Op{prim, [App-result, other]} or [other, App-result]
    //        where the App-result binding is App(member, step) and `prim` is a Binary{8} op
    //                                                    → non-tail call (Cont::{Not,And,Or,Xor})
    let result_atom = anf.result();
    let bindings = anf.bindings();

    // Locate the binding that defines the result atom.
    let result_binding = bindings
        .iter()
        .find(|b| &b.name == result_atom)
        .ok_or_else(|| {
            AotError::UnsupportedNode(
                "trampoline: arm result atom is not locally bound (non-canonical ANF) — refused (G2)"
                    .to_owned(),
            )
        })?;

    // Helper: is `b` a binding `App(member, step)`? Returns (callee_idx, step).
    let as_member_app = |rhs: &Rhs| -> Option<(usize, Atom)> {
        if let Rhs::App {
            func: Atom::Named(f),
            arg,
        } = rhs
        {
            if let Some(idx) = names.iter().position(|n| n == f) {
                return Some((idx, arg.clone()));
            }
        }
        None
    };

    // Case (a): the result binding itself is the member App → tail call.
    if let Some((callee, step)) = as_member_app(&result_binding.rhs) {
        // No other member reference may appear (the step must not itself recurse — that is a deeper
        // shape we refuse, G2).
        check_single_call(anf, names, result_atom)?;
        return Ok(ArmPlan::Call {
            callee,
            step,
            call_name: result_atom.clone(),
            cont: Cont::Identity,
        });
    }

    // Case (b): the result binding is a Binary{8} Op wrapping exactly one member-App operand.
    if let Rhs::Op { prim, args } = &result_binding.rhs {
        // Find which arg is the member-App result and which (if any) is the saved operand.
        let mut call_info: Option<(usize, Atom, Atom)> = None; // (callee, step, call_binding_name)
        let mut saved: Option<Atom> = None;
        for a in args {
            // Is `a` bound by App(member, _)?
            if let Some(b) = bindings.iter().find(|b| &b.name == a) {
                if let Some((callee, step)) = as_member_app(&b.rhs) {
                    if call_info.is_some() {
                        return Err(AotError::UnsupportedNode(
                            "trampoline: more than one recursive call in one arm — only a single \
                             self/sibling call per arm is supported (linear non-tail recursion; G2)"
                                .to_owned(),
                        ));
                    }
                    call_info = Some((callee, step, a.clone()));
                    continue;
                }
            }
            // Otherwise it is the saved operand (resolved against the arm env at emit time — works
            // for both `Named` and `Temp` atoms).
            saved = Some(a.clone());
        }
        let (callee, step, call_name) = call_info.ok_or_else(|| {
            AotError::UnsupportedNode(
                "trampoline: a member-referencing arm whose result op has no direct recursive-call \
                 operand is too complex to lower — refused (G2)"
                    .to_owned(),
            )
        })?;
        let cont = match (prim.as_str(), &saved) {
            ("bit.not", None) => Cont::Not,
            ("bit.and", Some(s)) => Cont::And(s.clone()),
            ("bit.or", Some(s)) => Cont::Or(s.clone()),
            ("bit.xor", Some(s)) => Cont::Xor(s.clone()),
            (other, _) => {
                return Err(AotError::UnsupportedNode(format!(
                    "trampoline: non-tail continuation op `{other}` is not a supported \
                     Binary{{8}} pending op (only bit.not/and/or/xor wrapping a single recursive \
                     call; G2)"
                )));
            }
        };
        // Exactly one recursive call total in the arm.
        check_single_call(anf, names, result_atom)?;
        return Ok(ArmPlan::Call {
            callee,
            step,
            call_name,
            cont,
        });
    }

    Err(AotError::UnsupportedNode(
        "trampoline: a member-referencing arm in a non-canonical shape (the recursive call is not \
         the result, nor a single Binary{8} op wrapping the result) — refused, never fragile IR \
         (G2/VR-5)"
            .to_owned(),
    ))
}

/// Verify the arm contains **exactly one** member-`App` reference (the one feeding the result/its
/// wrapping op), and no member name appears anywhere else (the step must not itself recurse, no
/// member reference in a pre-call binding other than that single call). Any other shape is refused.
fn check_single_call(anf: &Anf, names: &[&str], call_result: &Atom) -> Result<(), AotError> {
    let bindings = anf.bindings();
    // Identify the call binding (the one bound to App(member, _) whose name participates in result).
    // We count member references across all bindings *except* allow exactly one `App(member, step)`.
    let mut member_app_count = 0usize;
    for b in bindings {
        match &b.rhs {
            Rhs::App {
                func: Atom::Named(f),
                ..
            } if names.iter().any(|n| n == f) => {
                member_app_count += 1;
            }
            // Member name referenced as a non-App operand → too complex / unsupported (G2).
            _ => {
                if rhs_refs_any_member(&b.rhs, names) {
                    return Err(AotError::UnsupportedNode(
                        "trampoline: a member/self name is referenced in a non-call position (or a \
                         nested binder) inside an arm — only a single saturated App(member, step) \
                         per arm is supported (G2)"
                            .to_owned(),
                    ));
                }
            }
        }
    }
    if member_app_count != 1 {
        return Err(AotError::UnsupportedNode(format!(
            "trampoline: an arm must contain exactly one recursive App(member, step); found \
             {member_app_count} (linear non-tail recursion only; G2)"
        )));
    }
    let _ = call_result;
    Ok(())
}

/// Does any member name appear anywhere in `anf` (as a `Named` atom)?
fn anf_refs_any(anf: &Anf, name: &str) -> bool {
    anf.bindings().iter().any(|b| rhs_refs_name(&b.rhs, name))
        || matches!(anf.result(), Atom::Named(n) if n == name)
}

fn rhs_refs_any_member(rhs: &Rhs, names: &[&str]) -> bool {
    names.iter().any(|n| rhs_refs_name(rhs, n))
}

/// Does `rhs` reference `name` (as a `Named` atom) anywhere, including nested bodies? Used to refuse
/// any non-canonical member reference (nested `Lam`/`Fix`/`Match` member use). Conservative: a nested
/// body referencing a member is treated as a reference (→ refusal upstream).
fn rhs_refs_name(rhs: &Rhs, name: &str) -> bool {
    let named = |a: &Atom| matches!(a, Atom::Named(n) if n == name);
    match rhs {
        Rhs::Const(_) => false,
        Rhs::Alias(a) => named(a),
        Rhs::Op { args, .. } | Rhs::Construct { args, .. } => args.iter().any(named),
        Rhs::Swap { src, .. } => named(src),
        Rhs::App { func, arg } => named(func) || named(arg),
        Rhs::Lam { param, body } => param != name && anf_refs_named(body, name),
        Rhs::Fix { name: fname, body } => fname != name && anf_refs_named(body, name),
        Rhs::FixGroup { defs, .. } => {
            !defs.iter().any(|(n, _)| n == name)
                && defs.iter().any(|(_, d)| anf_refs_named(d, name))
        }
        Rhs::Match {
            scrutinee,
            alts,
            default,
        } => {
            named(scrutinee)
                || alts.iter().any(|alt| match alt {
                    AnfAlt::Ctor { body, .. } | AnfAlt::Lit { body, .. } => {
                        anf_refs_named(body, name)
                    }
                })
                || default.as_ref().is_some_and(|d| anf_refs_named(d, name))
        }
    }
}

fn anf_refs_named(anf: &Anf, name: &str) -> bool {
    anf_refs_any(anf, name)
}

/// Destructure a `Fix { name, body }` (the suspended [`crate::llvm::FixVal`]) into a single-member
/// group, or an explicit refusal if the body is not the canonical `λparam. Match param { … }` shape.
pub(crate) fn destructure_fix(self_name: &str, fix_body: &Anf) -> Result<Vec<Member>, AotError> {
    let (param, arms, default) = destructure_lam_match(self_name, fix_body)?;
    Ok(vec![Member {
        name: self_name.to_owned(),
        param,
        arms,
        default,
    }])
}

/// Destructure a `FixGroup`'s member definitions into [`Member`]s. Each member must be the canonical
/// `λparam. Match param { Lit-arms / default }` shape (Binary{8} branch). The `defs` are the lowered
/// `(name, body)` pairs carried by every `Rhs::FixGroup` binding.
pub(crate) fn destructure_fixgroup(defs: &[(String, Anf)]) -> Result<Vec<Member>, AotError> {
    let mut members = Vec::with_capacity(defs.len());
    for (name, def) in defs {
        let (param, arms, default) = destructure_lam_match(name, def)?;
        members.push(Member {
            name: name.clone(),
            param,
            arms,
            default,
        });
    }
    Ok(members)
}

/// The destructured `λparam. Match param { … }` shape: `(param, [(lit-key, arm)], default)`.
type LamMatch = (String, Vec<(u64, Anf)>, Option<Anf>);

/// Shared destructuring of a `λparam. Match param { Lit-arms / default }` body (single `Lam` binding
/// whose body is a single `Match` on `param` with `Lit` arms). Returns `(param, lit-arms, default)`.
/// Any deviation is an explicit [`AotError::UnsupportedNode`] (G2).
fn destructure_lam_match(member_name: &str, member_body: &Anf) -> Result<LamMatch, AotError> {
    let lam_bindings = member_body.bindings();
    if lam_bindings.len() != 1 {
        return Err(AotError::UnsupportedNode(format!(
            "trampoline: member `{member_name}` body must be exactly `λparam. Match …` (a single \
             Lam binding); got {} bindings (G2)",
            lam_bindings.len()
        )));
    }
    let (param, lam_body) = match &lam_bindings[0].rhs {
        Rhs::Lam { param, body } => (param.clone(), body),
        other => {
            return Err(AotError::UnsupportedNode(format!(
                "trampoline: member `{member_name}` body must be a Lam, got {other:?} (G2)"
            )));
        }
    };
    let inner = lam_body.bindings();
    if inner.len() != 1 {
        return Err(AotError::UnsupportedNode(format!(
            "trampoline: member `{member_name}` Lam body must be exactly `Match param {{ … }}`; \
             got {} bindings (G2)",
            inner.len()
        )));
    }
    let (alts, default) = match &inner[0].rhs {
        Rhs::Match {
            scrutinee,
            alts,
            default,
        } => {
            if scrutinee != &Atom::Named(param.clone()) {
                return Err(AotError::UnsupportedNode(format!(
                    "trampoline: member `{member_name}` Match scrutinee must be the param \
                     `{param}`; got {scrutinee:?} (G2)"
                )));
            }
            (alts.clone(), default.clone())
        }
        other => {
            return Err(AotError::UnsupportedNode(format!(
                "trampoline: member `{member_name}` Lam body must be a Match, got {other:?} (G2)"
            )));
        }
    };
    let mut lit_arms: Vec<(u64, Anf)> = Vec::with_capacity(alts.len());
    for alt in &alts {
        match alt {
            AnfAlt::Lit { value, body } => {
                let key = packed_lit(value)?;
                lit_arms.push((key, body.clone()));
            }
            AnfAlt::Ctor { .. } => {
                return Err(AotError::UnsupportedNode(format!(
                    "trampoline: member `{member_name}` has a Ctor arm on the recursion param — \
                     only Binary{{8}} Lit arms are supported (G2)"
                )));
            }
        }
    }
    Ok((param, lit_arms, default))
}

/// Pack a `Binary{8}` literal `Value` to the `u64` switch key (shared with [`crate::llvm`]).
fn packed_lit(value: &Value) -> Result<u64, AotError> {
    lit_binary8_packed_pub(value)
}

/// Classify a destructured group: `true` iff it is a **single member** whose every arm is **tail or
/// base** with **no `Match` in a pre-call binding** — i.e. exactly the fragment the fast iterative
/// tail-loop ([`crate::llvm::lower_tail_fix`]) already handles byte-for-byte. `llvm.rs` uses this to
/// keep the tail loop for that fragment and only reach for the heavier trampoline when it must
/// (non-tail / `FixGroup` / Match-in-pre-call). Returns `Err` only on a genuinely out-of-scope shape
/// that *neither* path can lower (so the caller surfaces one honest refusal — G2).
pub(crate) fn is_pure_tail_single_fix(members: &[Member]) -> Result<bool, AotError> {
    if members.len() != 1 {
        return Ok(false); // a FixGroup is never the tail-loop fragment.
    }
    let m = &members[0];
    let names = [m.name.as_str()];
    let mut all_tail_or_base = true;
    let arms_iter = m.arms.iter().map(|(_, a)| a).chain(m.default.iter());
    for arm in arms_iter {
        // A `Match` in the arm's pre-call bindings forces the trampoline (the tail loop refuses it,
        // DN-15 §8.5). Detect a nested Match anywhere in the arm body's binding RHSs.
        if arm_has_nested_match(arm) {
            all_tail_or_base = false;
            continue;
        }
        match analyze_arm(arm, &names, &m.param)? {
            ArmPlan::Base => {}
            ArmPlan::Call { cont, .. } => {
                if cont != Cont::Identity {
                    all_tail_or_base = false; // a non-tail (pending-op) call ⇒ trampoline.
                }
            }
        }
    }
    Ok(all_tail_or_base)
}

/// Does the arm body contain a nested `Match` in any of its bindings (which the tail loop refuses,
/// DN-15 §8.5, but the trampoline pre-call lowering also refuses — both route honestly)? This only
/// distinguishes *which* path to try; a true result steers to the trampoline.
fn arm_has_nested_match(anf: &Anf) -> bool {
    anf.bindings()
        .iter()
        .any(|b| matches!(&b.rhs, Rhs::Match { .. }))
}

/// The heap-trampoline runtime: a thin `@malloc`/`@free` pair for the frame stack, emitted into the
/// module **only** when a program uses the trampoline (a non-tail-`Fix`/`FixGroup` program). A
/// closure/tail-only program emits byte-for-byte the same module as before (no trampoline runtime).
/// Fully textual / dumpable (no opaque pass — RFC-0004 §6); zero `unsafe`.
pub(crate) fn trampoline_runtime() -> String {
    let mut s = String::new();
    s.push_str("; M-850 heap-trampoline frame-stack runtime (non-tail Fix / FixGroup)\n");
    s.push_str("define i8* @myc_tramp_alloc(i64 %n) {\nentry:\n");
    s.push_str("  %p = call i8* @malloc(i64 %n)\n");
    s.push_str("  %null = icmp eq i8* %p, null\n");
    s.push_str("  br i1 %null, label %oom, label %ok\n");
    // OOM: a defined-trap (@abort is declared noreturn; the trailing ret is provably dead — G2).
    s.push_str("oom:\n  call void @abort()\n  ret i8* null\n");
    s.push_str("ok:\n  ret i8* %p\n}\n\n");
    s.push_str("define void @myc_tramp_free(i8* %p) {\nentry:\n");
    s.push_str("  call void @free(i8* %p)\n");
    s.push_str("  ret void\n}\n");
    s
}
