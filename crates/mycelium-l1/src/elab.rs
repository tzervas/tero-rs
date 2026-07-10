//! **Elaboration to the L0 Core IR** (RFC-0007 §4.6, **retired by RFC-0001 r4**). The
//! evaluation-complete fragment is now the **whole v0 calculus**: representation ops (L0), data +
//! matching (r3, `Construct`/flat `Match`), and **functions + recursion** (r4/r5,
//! `Lam`/`App`/`Fix`/`FixGroup`). So a self- *or* mutually-recursive, data-building, matching program
//! elaborates to a closed L0 term. A guarantee index `@ g` is no longer a `Residual`: since RFC-0018
//! (M-663) it is **statically checked** by [`crate::grade`] and then **erased** here (a grade, like a
//! type, is a compile-time property with no L0 node — KC-3). The remaining `Residual`s are genuine
//! staging (generics/monomorphization, `wild`/FFI, `spore`) — never a partial artifact (G2).
//!
//! This module also owns the shared surface→kernel bridge the evaluator reuses, so the two
//! execution paths cannot drift on the basics: literal values ([`lit_value`]), representation
//! resolution ([`type_repr`]), and the v0 policy-name reference ([`policy_name_ref`]).
//!
//! # How a `match` lowers (RFC-0011 §4.4)
//! Nested surface patterns are compiled to the **flat** kernel `Match` by the **M-320 Maranget
//! decision tree** (`crate::decision`) — the untrusted, inspectable lowering. Each tree `Switch`
//! becomes an L0 `Match` on the occurrence's bound variable; each constructor case becomes an
//! `Alt::Ctor` binding *all* the constructor's fields (so every binder occurrence is available at
//! the leaf), and each leaf elaborates the surface arm's body with its binders mapped to those
//! field variables. `if` desugars to a `Match` on the prelude `Bool`. WF7 coverage is the checker's
//! (the tree is verified `Fail`-free before lowering — defense in depth, never silent).
//!
//! # How recursion lowers (RFC-0001 r4/r5)
//! The reachable call graph is decomposed into strongly-connected components (Tarjan), bound
//! **callee-first**. A **self-recursive singleton** is bound once as `let f = Fix(f, λparams. body)`;
//! a **mutually-recursive group** of ≥2 functions (M-343; R7-Q3) is bound as a single
//! `FixGroup{[(f, λ…), (g, λ…), …]}` whose members are all mutually in scope. A call to any recursive
//! function becomes a curried `App` on its recursion variable; every **other** call still inlines
//! (the residual non-recursive call graph is acyclic). `for` desugars to a synthesized self-recursive
//! `Fix` fold over the linear spine (RFC-0007 §4.8).

use std::collections::{BTreeMap, BTreeSet};

use mycelium_core::{
    operation_hash, Alt, CtorRef, CtorSpec, DataRegistry, DeclSpec, FieldSpec, FieldTyRef,
    FloatWidth, FnSig, Meta, Node, Payload, PolicyRef, Provenance, Repr, ScalarKind, SparsityClass,
    Trit, Value,
};

use crate::ast::{Arm, BaseType, Expr, Literal, Path, Scalar, Sparsity, TypeRef, WidthRef};
use crate::checkty::{infer_type, normalize_pattern, prim_kernel_name, resolve_ty, Env, Ty};
use crate::decision::{self, Head, Tree};

/// Why a definition could not be elaborated to L0 — always explicit, never a partial artifact
/// (RFC-0007 §4.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElabError {
    /// The body (or something it calls) is outside the evaluation-complete fragment; the program
    /// still *runs* — on the L1 fuel-guarded evaluator (RFC-0007 §4.6).
    Residual {
        /// The definition being elaborated when the refusal arose.
        site: String,
        /// Which construct fell outside the fragment, and why.
        what: String,
    },
    /// The requested entry definition does not exist in the checked environment.
    UnknownFn(String),
    /// A pass-internal AST traversal ([`crate::totality::walk_expr`], reused here for call-set
    /// collection over the call graph — M-641/DRY) exceeded its own explicit recursion-depth budget
    /// (M-674) on a pathologically-nested body. An operational resource refusal, distinct from
    /// [`ElabError::Residual`] (a semantic "outside the fragment" verdict) — refused cleanly rather
    /// than overflowing the host stack (banked guard 4).
    DepthExceeded {
        /// The definition being analyzed when the budget was exceeded.
        site: String,
        /// The exceeded budget.
        limit: u32,
    },
}

impl core::fmt::Display for ElabError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ElabError::Residual { site, what } => write!(
                f,
                "`{site}` is outside the evaluation-complete fragment (RFC-0007 §4.6): {what} — \
                 run it on the L1 evaluator"
            ),
            ElabError::UnknownFn(name) => write!(f, "no function `{name}` in the checked nodule"),
            ElabError::DepthExceeded { site, limit } => write!(
                f,
                "`{site}`: AST nesting exceeds the call-graph analysis's own recursion-depth budget \
                 ({limit}) — an explicit budget (banked guard 4; M-674), refused cleanly rather than \
                 overflowing the host stack"
            ),
        }
    }
}

impl std::error::Error for ElabError {}

fn residual<T>(site: &str, what: impl Into<String>) -> Result<T, ElabError> {
    Err(ElabError::Residual {
        site: site.to_owned(),
        what: what.into(),
    })
}

/// Build the L0 [`Value`] of a representation literal (Q6: a literal *is* its representation —
/// width = digit count). Bare integers and lists have no representation family and are refused
/// (the typechecker already refuses them; this refusal keeps the bridge honest on its own).
pub fn lit_value(site: &str, l: &Literal) -> Result<Value, ElabError> {
    match l {
        Literal::Bin(s) => {
            let bits: Vec<bool> = s
                .chars()
                .filter(|c| *c == '0' || *c == '1')
                .map(|c| c == '1')
                .collect();
            let width = u32::try_from(bits.len()).expect("digit count fits u32");
            Value::new(
                Repr::Binary { width },
                Payload::Bits(bits),
                Meta::exact(Provenance::Root),
            )
            .map_or_else(
                |e| residual(site, format!("malformed binary literal: {e}")),
                Ok,
            )
        }
        Literal::Trit(s) => {
            let trits: Vec<Trit> = s
                .chars()
                .map(|c| match c {
                    '+' => Ok(Trit::Pos),
                    '0' => Ok(Trit::Zero),
                    '-' => Ok(Trit::Neg),
                    other => Err(other),
                })
                .collect::<Result<_, _>>()
                .map_or_else(
                    |c| residual(site, format!("non-trit char {c:?} in ternary literal")),
                    Ok,
                )?;
            let width = u32::try_from(trits.len()).expect("trit count fits u32");
            Value::new(
                Repr::Ternary { trits: width },
                Payload::Trits(trits),
                Meta::exact(Provenance::Root),
            )
            .map_or_else(
                |e| residual(site, format!("malformed ternary literal: {e}")),
                Ok,
            )
        }
        // RFC-0032 D4 (M-750): a `0x…` byte-string literal lowers to a `Repr::Bytes` value. The lexer
        // already validated even-hex parity / non-empty, so decoding the (separator-stripped) hex
        // into bytes is total; a stray non-hex char is still a never-silent `Residual` (G2,
        // defense-in-depth — it should never reach here).
        Literal::Bytes(s) => {
            let hex: String = s.chars().filter(|c| *c != '_').collect();
            // Even count is a lexer invariant; chunk into byte pairs.
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            let chars: Vec<char> = hex.chars().collect();
            for pair in chars.chunks(2) {
                let hs: String = pair.iter().collect();
                match u8::from_str_radix(&hs, 16) {
                    Ok(b) => bytes.push(b),
                    Err(_) => {
                        return residual(site, format!("non-hex byte pair {hs:?} in `0x…` literal"))
                    }
                }
            }
            Value::new(
                Repr::Bytes,
                Payload::Bytes(bytes),
                Meta::exact(Provenance::Root),
            )
            .map_or_else(
                |e| residual(site, format!("malformed byte literal: {e}")),
                Ok,
            )
        }
        // M-910/M-911 (kickoff `enb` Phase-I H1): a `"…"` textual string literal lowers to the
        // SAME `Repr::Bytes`/`Payload::Bytes` value form as `Literal::Bytes` above (KC-3 — no new
        // L0 node) — its decoded content, UTF-8-encoded. The lexer already decoded the escape set
        // and validated termination, so the encode is total.
        Literal::Str(s) => Value::new(
            Repr::Bytes,
            Payload::Bytes(s.as_bytes().to_vec()),
            Meta::exact(Provenance::Root),
        )
        .map_or_else(
            |e| residual(site, format!("malformed string literal: {e}")),
            Ok,
        ),
        // ADR-040 (M-897, kickoff `enb` Phase-I H1 Gap A): a decimal float literal lowers to the
        // EXISTING `Repr::Float`/`Payload::Float` scalar value form landed by M-896 (KC-3 — no new
        // L0 node). The single text→f64 conversion happens here, via `f64::from_str` — the
        // **correctly-rounded** (RNE) decimal→binary64 conversion the literal's spec documents
        // (ADR-040 FLAG-3; correct rounding is a Rust-std claim — `Declared` — pinned `Empirical`
        // by the round-trip conformance corpus). Meta is `Exact` **as a definition**: the literal
        // denotes exactly the correctly-rounded binary64 of its decimal text, and this value *is*
        // that binary64 (ADR-040 §2.6 — the definition is Exact; the host-conversion claim is the
        // Empirical residue, tested, never silently upgraded — VR-5). The lexer already validated
        // form + finiteness, so the parse is total; a failure here is a never-silent internal
        // `Residual` (defense in depth, mirroring the `Bytes` arm). `Value::new` canonicalizes any
        // NaN (unreachable for a finite literal) per ADR-040 §2.3.
        Literal::Float(s) => {
            let x: f64 = match s.parse() {
                Ok(x) => x,
                Err(_) => {
                    return residual(
                        site,
                        format!("internal: malformed float literal text {s:?} reached elaboration"),
                    )
                }
            };
            Value::new(
                Repr::Float {
                    width: FloatWidth::F64,
                },
                Payload::Float(x),
                Meta::exact(Provenance::Root),
            )
            .map_or_else(
                |e| residual(site, format!("malformed float literal: {e}")),
                Ok,
            )
        }
        Literal::Int(_) => residual(
            site,
            "a bare integer literal has no representation family (Q6)",
        ),
        Literal::AmbientInt(_, _) => residual(
            site,
            "internal: an unresolved ambient bare decimal reached elaboration — the checker \
             resolves its width before the L0 bridge runs (RFC-0012 §4.3)",
        ),
        // A list literal `[…]` lowers via the elaborator's `expr_inner` (it elaborates each element
        // expression, then builds the `Repr::Seq` const) — `lit_value` only sees context-free
        // literals, so a `Literal::List` reaching here is an internal invariant break (never silent,
        // G2/VR-5).
        Literal::List(_) => residual(
            site,
            "internal: a list literal reached `lit_value` — it lowers through `expr_inner` (the \
             element expressions need elaboration; RFC-0032 D3)",
        ),
    }
}

/// Resolve a surface [`TypeRef`] to a kernel [`Repr`] (swap targets). Only representation types
/// resolve; named/data and `Substrate` types are explicit refusals.
pub fn type_repr(site: &str, t: &TypeRef) -> Result<Repr, ElabError> {
    match &t.base {
        BaseType::Binary(WidthRef::Lit(n)) => Ok(Repr::Binary { width: *n }),
        BaseType::Binary(WidthRef::Name(v)) => Err(ElabError::Residual {
            site: site.to_owned(),
            what: format!(
                "width variable `{v}` reached elaboration — must be monomorphized first \
                 (DN-42 / M-753; Residual)"
            ),
        }),
        BaseType::Ternary(WidthRef::Lit(m)) => Ok(Repr::Ternary { trits: *m }),
        BaseType::Ternary(WidthRef::Name(v)) => Err(ElabError::Residual {
            site: site.to_owned(),
            what: format!(
                "width variable `{v}` reached elaboration — must be monomorphized first \
                 (DN-42 / M-753; Residual)"
            ),
        }),
        BaseType::Dense(d, s) => Ok(Repr::Dense {
            dim: *d,
            dtype: match s {
                Scalar::F16 => ScalarKind::F16,
                Scalar::Bf16 => ScalarKind::Bf16,
                Scalar::F32 => ScalarKind::F32,
                Scalar::F64 => ScalarKind::F64,
            },
        }),
        // RFC-0003 §3 / ADR-008 (M-892): the VSA type resolves to its kernel `Repr` — the lift of
        // the former blanket deferral (the `vsa_*` prims need typable operands). Not a v0 swap
        // target (the checker's swap gate admits only Binary/Ternary/Dense), but `type_repr` is
        // the general surface→`Repr` resolver, so a concrete resolution here is correct and
        // never-silent (the `Bytes`/`Seq`/`Float` posture). The surface model ident is
        // canonicalized to the kernel model id exactly as the checker does (one mapping — the
        // types the checker admits and the reprs elaboration emits must not fork).
        BaseType::Vsa {
            model,
            dim,
            sparsity,
        } => Ok(Repr::Vsa {
            model: crate::checkty::vsa_kernel_model_id(model),
            dim: *dim,
            sparsity: sparsity_class(sparsity),
        }),
        // RFC-0032 D3/D4: the sequence/byte-string reprs resolve to their kernel `Repr` (the element
        // type recurses through `type_repr`). They are not v0 swap targets (the checker refuses a
        // `swap` to them — no swap engine), but `type_repr` is the general surface→`Repr` resolver,
        // so a concrete resolution here is correct and never-silent.
        BaseType::Seq { elem, len } => Ok(Repr::Seq {
            elem: Box::new(type_repr(site, elem)?),
            len: *len,
        }),
        BaseType::Bytes => Ok(Repr::Bytes),
        // ADR-040 (M-897): the nullary scalar-float repr type resolves to its kernel `Repr`
        // (binary64 only — FLAG-1). Not a v0 swap target (the checker's swap gate admits only
        // Binary/Ternary/Dense), but `type_repr` is the general surface→`Repr` resolver, so a
        // concrete resolution here is correct and never-silent (the `Bytes`/`Seq` posture).
        BaseType::Float => Ok(Repr::Float {
            width: FloatWidth::F64,
        }),
        BaseType::Substrate(tag) => residual(
            site,
            format!("Substrate{{{tag}}} is not a representation type"),
        ),
        BaseType::Named(name, _) => residual(
            site,
            format!("`{name}` is not a representation type — no kernel Repr"),
        ),
        BaseType::Ambient(_) => residual(
            site,
            "internal: an unresolved paradigm-less repr `{…}` reached elaboration — the ambient \
             resolution pass fills it first (RFC-0012 §4.3)",
        ),
        // Function types are surface-only in HOF stage 1 (RFC-0024 §3, M-685); elaboration
        // (defunctionalization) is M-687. A function type used as a swap target is a
        // never-silent explicit refusal (G2).
        BaseType::Fn(_, _) => residual(
            site,
            "function types (`A -> B`) are not a representation type and cannot be a swap target \
             (RFC-0024 §3, HOF stage 1 — defunctionalization is M-687)",
        ),
        // M-826: tuple types are not representation types and cannot be swap targets (a swap must
        // target a Binary/Ternary/Dense/Bytes/Seq repr — never a product type; never silent, G2).
        BaseType::Tuple(_) => residual(
            site,
            "tuple types are not a representation type and cannot be a swap target (M-826; G2)",
        ),
    }
}

/// The v0 **policy-name reference**: a deterministic, domain-separated content address derived
/// from the surface policy *name* (`policy: roundtrip`).
///
/// Honesty note (Declared): RFC-0005 policy *objects* are content-addressed over their canonical
/// serialization (`mycelium-select::SelectionPolicy::policy_ref`); binding surface names to
/// registered policy objects is later integration work. Until it lands, this name-derived address
/// keeps `Meta.policy_used` answerable and — because the evaluator and the elaborator share this
/// one function — keeps every execution path's swaps on the *same* `PolicyRef`, so the NFR-7
/// differential is meaningful. Domain-separated (`policy-name.v0:`) so it can never collide with
/// a structural or operation hash.
#[must_use]
pub fn policy_name_ref(policy: &Path) -> PolicyRef {
    operation_hash(&policy_name_preimage(policy))
}

/// The domain-separated **preimage** of [`policy_name_ref`] — `policy-name.v0:<dotted>` — before it
/// is fed to the kernel content hash. Extracted (M-1012) so the self-hosted `semcore.myc` port has a
/// reachable live oracle for the *wild-free* half of the reference (the frontend's name construction);
/// the hashing step ([`operation_hash`], BLAKE3) is the kernel primitive the port defers to the
/// materialization/kernel-hash crossing (DN-26 §10; `.myc` `FLAG-semcore-27`). Behaviour-preserving:
/// `policy_name_ref` is byte-identical to its former inline form.
pub(crate) fn policy_name_preimage(policy: &Path) -> String {
    format!("policy-name.v0:{}", policy.0.join("."))
}

/// A surface name's elaboration binding: `(surface name, kernel variable, v0 type)`. The type lets
/// the elaborator re-infer a `match` scrutinee's type (to lower its patterns) without a second
/// inference pass over the whole body.
type Binding = (String, String, Ty);

/// Elaborate the nullary function `entry` of a checked nodule to a closed L0 [`Node`].
///
/// As of RFC-0001 r4 the evaluation-complete fragment is the **whole v0 calculus**: data + matching
/// (r3) and now **functions + recursion** (`Lam`/`App`/`Fix`). Each reachable **self-recursive**
/// function is bound once as `let f = Fix(f, λparams. body)` (callee-first), and a call to it
/// elaborates to a curried `App`; every other call still inlines (the non-recursive call graph is
/// acyclic). **Mutual recursion** lowers to a `FixGroup` (RFC-0001 r5; M-343 — R7-Q3); top-level
/// functions in a nodule are mutually visible (RP-6 / DN-13 — no surface marker), so a mutual group is
/// *inferred* from the call graph and **materialized as a `FixGroup` node** — inspectable in the
/// elaborated term, never a black box. A guarantee index `@ g` is statically checked + erased since
/// RFC-0018 (M-663), not a `Residual`. On success the result is a closed L0 term whose evaluation
/// must agree with the L1 evaluator (NFR-7; the M-210 differential).
pub fn elaborate(env: &Env, entry: &str) -> Result<Node, ElabError> {
    // Elaboration recurses over the (checked) expression AST; run it on a deep worker stack so deep
    // input never overflows the caller's thread stack. The semantic bound stays the upstream explicit
    // budgets (the parser's nesting cap + the checker's `MAX_CHECK_DEPTH`, both already enforced before
    // a program reaches here); the worker stack is the transitional Rust-only adapter (`mycelium_stack`).
    mycelium_stack::with_deep_stack(|| {
        // M-673: monomorphize first — specialize every reachable generic instantiation and statically
        // resolve every trait-method call to a direct call, yielding a closed monomorphic `Env`. The
        // entry is nullary monomorphic, so its name is byte-identical in the mono'd env (empty type
        // args ⇒ unchanged name), and on a non-generic/non-trait program the pre-pass is a fast clone
        // (so the existing monomorphic differential is observably unchanged — NFR-7). The prelude/SCC
        // machinery below then runs **unchanged** over the specialized env.
        let mono_env = crate::mono::monomorphize(env, entry)?;
        let (mut el, binders, fd) = elab_prelude(&mono_env, entry)?;
        let mut stack = vec![entry.to_owned()];
        let entry_body = el.expr(&mut stack, &[], &fd.body)?;
        Ok(wrap_in_binders(binders, entry_body))
    })
}

/// **ADR-033/DN-74 (M-923):** elaborate `entry` **without** the [`crate::mono::monomorphize`]
/// pre-pass [`elaborate`] always runs first. Intended for an `env` that is already concrete
/// (no generics, no traits, no unresolved widths) and whose `Ty::Fn` record fields should lower
/// through the kernel `FieldSpec::Fn` primitive (`field_spec`'s new arm, Path A — the DN-74-ratified
/// FLAG-1 disposition) rather than through `mono.rs`'s closure defunctionalization.
///
/// **Why this exists, honestly (new evidence surfaced while implementing M-923).** `mono.rs`
/// unconditionally rewrites *every* reachable `Ty::Fn` field into a closure tag-sum before
/// `build_registry` ever runs (RFC-0024 §4A/M-704) — so calling `field_spec`'s new `FieldSpec::Fn`
/// arm through the standard [`elaborate`] entry is unreachable by construction (defunctionalization
/// already produces a closed, executing term for that case). Making `FieldSpec::Fn` reachable from
/// a real, parsed-and-checked program therefore needs an elaboration path that does not run through
/// `mono.rs` at all — this function is that path. It is deliberately **narrow**: it does not
/// monomorphize generics or lower closures (a `lambda` literal reaching [`Elab::expr`] this way is
/// still refused, per the `Expr::Lambda` invariant guard), so it only accepts programs that are
/// already closed and whose function-typed field values are bare references to ordinary top-level
/// functions (the `field_spec`/`expr` `Ty::Fn`/named-fn-as-value arms this leaf adds). This is
/// **additive** — [`elaborate`]'s own behavior, and the existing differential corpus that depends on
/// it (RFC-0024 §4A closures, NFR-7), is unchanged.
///
/// # Errors
/// The same [`ElabError`] variants as [`elaborate`] — most notably [`ElabError::Residual`] for a
/// generic entry/function or a construct this narrower path does not (yet) accept; **never** a
/// silent, half-lowered term (G2/VR-5).
pub fn elaborate_direct(env: &Env, entry: &str) -> Result<Node, ElabError> {
    mycelium_stack::with_deep_stack(|| {
        let (mut el, binders, fd) = elab_prelude(env, entry)?;
        let mut stack = vec![entry.to_owned()];
        let entry_body = el.expr(&mut stack, &[], &fd.body)?;
        Ok(wrap_in_binders(binders, entry_body))
    })
}

/// **Per-hypha elaboration of a `colony` entry** for the *real-concurrency* execution path
/// (RFC-0008 §4.7; M-666 redone with the `mycelium-mlir::runtime` executor). Where [`elaborate`]
/// produces the single L0 `Node` whose body is the **RT2 spawn-order sequentialization** (a `Let`
/// chain — the deterministic *reference* the concurrent run is validated against), this produces one
/// **closed L0 `Node` per `hypha`**: each hypha body, elaborated under the entry's scope and wrapped
/// in the **same** recursive-binder prelude (so a hypha may call the nodule's recursive functions).
/// The `mycelium-mlir` colony driver spawns each as a concurrent `Task` in a `Scope`/`Colony`, runs
/// the structured fork/join, and validates the concurrent observable **equals** the sequential
/// reference (RT2; an inequality is an explicit divergence, never silent — G2/RT4).
///
/// The colony's *observable* is its **last** hypha's value (the type rule, [`crate::checkty`]); the
/// returned vector preserves spawn order, so element `N-1` is that observable. This adds **no L0
/// concurrency node** — the trusted base stays sequential (RFC-0008 §4.2; KC-3); concurrency is
/// scheduling layered *over* unchanged per-hypha L0 terms.
///
/// Refuses with an explicit [`ElabError::Residual`] (never a fabricated accept) when the entry body
/// is **not** a `colony`, or when any hypha body is outside the evaluation-complete fragment.
pub fn elaborate_colony(env: &Env, entry: &str) -> Result<Vec<Node>, ElabError> {
    mycelium_stack::with_deep_stack(|| elaborate_colony_inner(env, entry))
}

fn elaborate_colony_inner(env: &Env, entry: &str) -> Result<Vec<Node>, ElabError> {
    // M-673: monomorphize first (see [`elaborate`]); the per-hypha lowering then runs unchanged over
    // the closed monomorphic env. A colony entry is nullary monomorphic, so its name is preserved.
    let mono_env = crate::mono::monomorphize(env, entry)?;
    let (mut el, binders, fd) = elab_prelude(&mono_env, entry)?;
    let Expr::Colony(hyphae) = &fd.body else {
        return residual(
            entry,
            "the entry body is not a `colony` — `elaborate_colony` lowers a colony to its per-hypha \
             closed L0 programs (the concurrent path); use `elaborate` for the sequentialized form",
        );
    };
    if hyphae.is_empty() {
        return residual(
            entry,
            "internal: an empty `colony` reached elaboration — the parser requires ≥ 1 hypha \
             (RFC-0008 §4.7)",
        );
    }
    let mut stack = vec![entry.to_owned()];
    // One closed L0 program per hypha: the hypha body elaborated under the entry scope, then wrapped
    // in the *shared* recursive prelude (the binders are cloned per hypha so each task is independent
    // — RT1: no shared state crosses a hypha boundary). Spawn order is preserved. Each hypha's
    // `@forage(policy)` (M-906/DN-70 D1), if present, is folded into its own program the same way
    // `elab_colony`'s sequential reference does (`Let{_=policy, body}` — RT3 semantics-free
    // placement), and the DN-63 FLAG-14 empty-candidate-set check runs identically.
    let mut programs = Vec::with_capacity(hyphae.len());
    for h in hyphae {
        let body = el.expr(&mut stack, &[], &h.body)?;
        let node = match &h.forage {
            None => body,
            Some(policy) => {
                forage_reject_if_empty(entry, h)?;
                let policy_node = el.expr(&mut stack, &[], policy)?;
                Node::Let {
                    id: el.fresh("forage_policy"),
                    bound: Box::new(policy_node),
                    body: Box::new(body),
                }
            }
        };
        programs.push(wrap_in_binders(binders.clone(), node));
    }
    Ok(programs)
}

/// **Policy + body elaboration of a `reclaim` entry** for the *real-supervision* execution path
/// (DN-58 §B; RFC-0008 RT7; M-817). Where [`elaborate`] produces the single L0 `Node` whose body is
/// the **sequential reference** (a `Let{_ = policy, body}` — evaluate the policy for its effect, then
/// yield the body; the deterministic reference the supervised run is validated against), this produces
/// the **policy** and **body** as two **closed L0 `Node`s**, each elaborated under the entry's scope
/// and wrapped in the **same** recursive-binder prelude (so each may call the nodule's recursive
/// functions). The `mycelium-mlir` reclaim driver ([`mycelium_mlir::run_reclaim`]) runs the body under
/// `mycelium-std-runtime::supervise_with_restart` (re-evaluating it per restart — which is why the body
/// is returned as an unevaluated node, not a value), threads a `SupervisionRecord` EXPLAIN trail, and
/// validates the supervised observable **equals** the sequential reference on success (RT7). This adds
/// **no L0 supervision node** — the trusted base stays sequential (KC-3); supervision is scheduling
/// layered *over* unchanged body evaluation, exactly as the concurrent `colony` driver layers over
/// per-hypha L0 terms.
///
/// Refuses with an explicit [`ElabError::Residual`] (never a fabricated accept) when the entry body is
/// **not** a `reclaim`, or when the policy/body is outside the evaluation-complete fragment.
pub fn elaborate_reclaim(env: &Env, entry: &str) -> Result<(Node, Node), ElabError> {
    mycelium_stack::with_deep_stack(|| elaborate_reclaim_inner(env, entry))
}

fn elaborate_reclaim_inner(env: &Env, entry: &str) -> Result<(Node, Node), ElabError> {
    // M-673: monomorphize first (see [`elaborate`]); the policy/body lowering then runs unchanged over
    // the closed monomorphic env. A reclaim entry is nullary monomorphic, so its name is preserved.
    let mono_env = crate::mono::monomorphize(env, entry)?;
    let (mut el, binders, fd) = elab_prelude(&mono_env, entry)?;
    let Expr::Reclaim { policy, body } = &fd.body else {
        return residual(
            entry,
            "the entry body is not a `reclaim` — `elaborate_reclaim` lowers a reclaim to its \
             (policy, body) closed L0 programs (the supervised path); use `elaborate` for the \
             sequential-reference form",
        );
    };
    let mut stack = vec![entry.to_owned()];
    // The policy and body, each elaborated under the entry scope and wrapped in the *shared* recursive
    // prelude (binders cloned so the two programs are independent). The driver evaluates the policy
    // once (for its effect) and re-evaluates the body per supervised restart (RT7).
    let policy_node = el.expr(&mut stack, &[], policy)?;
    let body_node = el.expr(&mut stack, &[], body)?;
    Ok((
        wrap_in_binders(binders.clone(), policy_node),
        wrap_in_binders(binders, body_node),
    ))
}

/// **Elaborate a user-defined generative-lowering rule's RHS to a closed L0 [`Node`]** (DN-54
/// §4.1/§6 / M-812-cont). This is the `crate::elab` site that **reads [`Env::lower_rules`]** — the
/// completion the `low` (M-812) landing deferred: a `lower Name[…] = <rhs>` rule is no longer mere
/// stored data, it elaborates to real L0.
///
/// The mechanism is **DRY and observation-faithful by construction**: the rule's RHS is given the
/// **exact same** lowering path a hand-written nullary `fn <rule>%rhs() = <rhs>` would take — it is
/// inserted as a synthetic nullary fn into a *clone* of `env` and run through the ordinary
/// [`elaborate`]. So the §7 differential `observe(derive Name for T) == observe(hand-lowered Name)`
/// holds **because the two go through one code path**, and the rule's observational identity is
/// `Empirical` (earned by running, never self-attested — VR-5).
///
/// **§6 KC-3 (kernel-growth) — `Proven`-by-construction (narrow, checked sense):** the return type
/// is [`Node`], a *closed* Rust enum (the frozen L0 grammar — `mycelium_core::node`). The elaborator
/// can only ever *construct* one of those variants, so a `lower` rule **cannot** introduce a new
/// kernel node — the type system is the side-condition that makes this `Proven`. (The checker's
/// §4.6 `wild`-refusal closes the one *surface*-growth a rule could otherwise smuggle in.) This
/// function does **not** *prove* an arbitrary RHS elaborates — only that **if** it does, it adds no
/// kernel node, which is exactly KC-3.
///
/// **Scope / honest residual (DN-54 underdetermination — FLAGGED).** v0 elaborates a **nullary,
/// monomorphic** rule (the landed surface — `lower Name = <expr>`): its RHS is a closed term. A
/// **parametric** rule (`lower Name[T] = <rhs over T>`) whose RHS mentions the type parameter `T`
/// has no monomorphic L0 form until `derive Name for <concrete>` *instantiates* `T` — and **how a
/// `derive` site's instantiated L0 attaches to / is referenced by the surrounding program is
/// underdetermined by DN-54** (the note's worked example RHS is an `impl` block, which is an *item*,
/// not an [`Expr`], so it is not even expressible as a v0 rule RHS). Rather than invent that
/// consumption semantics (G2/VR-5: a correct partial landing beats a guessed elaboration), this
/// elaborates the rule's RHS *as written* (the param-instantiation of a non-nullary RHS surfaces as
/// the ordinary generic [`ElabError::Residual`] from [`elaborate`]) and leaves the attachment model
/// for the maintainer to ratify. See DN-54 §3.2/§5 and the M-812-cont FLAG.
///
/// # Errors
/// [`ElabError::UnknownFn`] if `rule_name` is not a registered `lower` rule; [`ElabError::Residual`]
/// if the RHS is outside the evaluation-complete fragment (e.g. it mentions an un-instantiated type
/// parameter, or a `wild`/`spore`/`Substrate` site — never a fabricated artifact, G2).
pub fn elaborate_lower_rule(env: &Env, rule_name: &str) -> Result<Node, ElabError> {
    let Some(rule) = env.lower_rules.get(rule_name) else {
        return Err(ElabError::UnknownFn(rule_name.to_owned()));
    };
    // An **item-shaped** rule (DN-54 §10 Model A / M-973 — `lower Name[T] = impl Trait for T { … }`)
    // is **not** elaborated to a nullary-fn value here: its output is a sibling `impl` injected at the
    // `derive` site (checkty.rs), which the ordinary instance/method passes lower — there is no
    // free-standing L0 term for the rule itself. Surfacing it as a never-silent residual keeps this
    // path honest (G2): the caller wanting the derived L0 must go through the injected sibling.
    let rule = match &rule.rhs {
        crate::ast::LowerRhs::Expr(_) => rule,
        crate::ast::LowerRhs::Impl(_) => {
            return residual(
                rule_name,
                "this `lower` rule has an item-shaped (`impl … for …`) RHS — a DN-54 §10 Model A \
                 sibling-injection template (M-973). It has no stand-alone L0 term: its output is \
                 the concrete `impl` injected at each `derive Name for T` site (checked/lowered by \
                 the ordinary instance passes). `elaborate_lower_rule` lowers only expression-shaped \
                 rules; there is nothing to elaborate here (never silent — G2).",
            );
        }
    };
    let rhs_expr = rule
        .expr_rhs()
        .expect("matched the Expr arm just above — item-shaped rules returned early");
    // The synthetic entry name is `%`-prefixed: `%` is not a surface identifier character (the lexer
    // forbids it), so this can never collide with a real fn / rule / constructor name (G2 — no
    // silent shadowing). The RHS becomes its body verbatim.
    let entry = format!("%lower-rhs%{rule_name}");
    let synth = crate::ast::FnDecl {
        vis: crate::ast::Vis::Private,
        thaw: false,
        tier: None,
        sig: crate::ast::FnSig {
            name: entry.clone(),
            params: vec![],
            value_params: vec![],
            // The RHS may have any well-typed result; v0 elaboration does not consult the synthetic
            // entry's declared return type (it elaborates the body), so a placeholder return type is
            // immaterial. Use `Binary{0}` as an inert, always-valid placeholder; the checker has
            // already validated the RHS at definition time (DN-54 §4.1), so this synthetic fn is not
            // re-checked — it is fed straight to `elaborate`.
            ret: crate::ast::TypeRef::unguaranteed(BaseType::Binary(WidthRef::Lit(0))),
            effects: vec![],
            effect_budgets: std::collections::BTreeMap::new(),
        },
        body: rhs_expr.clone(),
    };
    let mut env2 = env.clone();
    env2.fns.insert(entry.clone(), synth);
    elaborate(&env2, &entry)
}

/// Shared front-end of [`elaborate`]/[`elaborate_colony`]: validate the entry is a closed (nullary,
/// no dynamic guarantee) definition, build the data registry, decompose the reachable call graph into
/// callee-first recursive SCCs (Tarjan), and elaborate each SCC's recursive binder. Returns the
/// primed [`Elab`], the callee-first binder list, and the entry's [`FnDecl`]. DRY: the recursion
/// machinery is identical whether the entry body is sequentialized to one `Node` or split per-hypha.
fn elab_prelude<'e>(
    env: &'e Env,
    entry: &str,
) -> Result<(Elab<'e>, Vec<RecBinding>, &'e crate::ast::FnDecl), ElabError> {
    let Some(fd) = env.fns.get(entry) else {
        return Err(ElabError::UnknownFn(entry.to_owned()));
    };
    if !fd.sig.value_params.is_empty() {
        return residual(
            entry,
            "the entry has value parameters — v0 elaborates closed (nullary) entries; \
             apply it from a nullary definition",
        );
    }
    if !fd.sig.params.is_empty() {
        return residual(
            entry,
            "the entry is generic — a generic definition's L0 lowering is staged (monomorphization; \
             RFC-0007 §11.3, the M-657 follow-up). Elaborate a concrete (monomorphic) entry.",
        );
    }
    // RFC-0018 (M-663): the return guarantee index `@ g` is now **statically checked** by the
    // grading pass (`crate::grade`) and **erased** here — like a type, a grade has no L0 form (it is
    // a compile-time property, not a runtime node — KC-3). No `Residual` (the stage-0 dynamic check
    // it replaced remains the runtime fallback only for dynamically-graded values — RFC-0018 §4.7).
    let registry = build_registry(env)?;
    // The recursive strongly-connected components of the reachable call graph, callee-first (Tarjan).
    // A self-recursive singleton stays a `Fix`; a group of ≥2 mutually-recursive functions becomes a
    // `FixGroup` (RFC-0001 r5; M-343 enacts mutual recursion — R7-Q3).
    let sccs = recursive_sccs(env, entry)?;
    let mut el = Elab {
        env,
        registry,
        fresh: 0,
        rec: BTreeMap::new(),
        depth: 0,
    };
    // Every member of a recursive SCC gets a kernel recursion variable — in scope for every recursive
    // body (its own SCC and any callee SCC) and the entry body.
    for scc in &sccs {
        for f in scc {
            let kf = el.fresh(f);
            el.rec.insert(f.clone(), kf);
        }
    }
    // Elaborate each SCC's binding, callee-first: a singleton self-recursion is a `Fix`; a group is a
    // `FixGroup` over the members' curried lambdas (each member sees every name in the group).
    let mut binders: Vec<RecBinding> = Vec::with_capacity(sccs.len());
    for scc in &sccs {
        if scc.len() == 1 {
            let f = &scc[0];
            let kf = el.rec[f].clone();
            let fix = Box::new(el.elab_recursive_fn(f, &kf)?);
            binders.push(RecBinding::Single { var: kf, fix });
        } else {
            let mut defs: Vec<(String, Box<Node>)> = Vec::with_capacity(scc.len());
            for f in scc {
                let kf = el.rec[f].clone();
                defs.push((kf, Box::new(el.elab_fn_lam(f)?)));
            }
            binders.push(RecBinding::Group(defs));
        }
    }
    Ok((el, binders, fd))
}

/// Wrap an elaborated body in a callee-first binder prelude. `binders` is callee-first; fold in
/// reverse so the first (callee) binding ends up outermost (in scope for every later binding and the
/// body). Shared by [`elaborate`] (one body) and [`elaborate_colony`] (each hypha body).
fn wrap_in_binders(binders: Vec<RecBinding>, body: Node) -> Node {
    binders.into_iter().rev().fold(body, |acc, b| match b {
        RecBinding::Single { var, fix } => Node::Let {
            id: var,
            bound: fix,
            body: Box::new(acc),
        },
        RecBinding::Group(defs) => Node::FixGroup {
            defs,
            body: Box::new(acc),
        },
    })
}

/// One recursive binding the entry body is wrapped in: a self-recursive singleton (`Fix`, bound via
/// `Let`) or a mutually-recursive group (`FixGroup`). Built callee-first; see [`elaborate`].
// `Clone` so [`elaborate_colony`] can replay the *same* recursive prelude over each hypha body
// (every concurrent task is an independent closed term — RT1: no shared mutable state).
#[derive(Clone)]
enum RecBinding {
    /// A self-recursive function: its kernel variable and the `Fix` node bound to it (boxed — the
    /// `Group` variant is pointer-sized, so an unboxed `Node` here would unbalance the enum).
    Single { var: String, fix: Box<Node> },
    /// A mutually-recursive group: `(member variable, curried lambda)` pairs, all mutually in scope.
    Group(Vec<(String, Box<Node>)>),
}

/// The **recursive** strongly-connected components of the reachable call graph, **callee-first**
/// (a callee SCC is bound *outside* its callers). A self-recursive singleton (`{f}` with a self-call)
/// and a mutual group (≥2 functions in a cycle) are both recursive SCCs; a function in no cycle
/// inlines and is **not** returned. Computed with Tarjan's algorithm — which finalises each SCC only
/// after all its successor (callee) SCCs, i.e. in reverse-topological = callee-first order. Roots,
/// successors, and each SCC's members are visited/sorted deterministically so the lowering (and thus
/// the content hash) is reproducible. A function is "reachable" if the entry transitively calls it.
fn recursive_sccs(env: &Env, entry: &str) -> Result<Vec<Vec<String>>, ElabError> {
    // BFS the reachable functions (sorted via the BTreeSet).
    let mut reachable: BTreeSet<String> = BTreeSet::new();
    let mut frontier = vec![entry.to_owned()];
    while let Some(f) = frontier.pop() {
        if !reachable.insert(f.clone()) {
            continue;
        }
        if let Some(fd) = env.fns.get(&f) {
            for callee in calls_in_fn(&f, &fd.body)? {
                if env.fns.contains_key(&callee) {
                    frontier.push(callee);
                }
            }
        }
    }

    // Tarjan's SCC over the reachable call graph.
    struct Tarjan<'e> {
        env: &'e Env,
        reachable: &'e BTreeSet<String>,
        index: usize,
        idx: BTreeMap<String, usize>,
        low: BTreeMap<String, usize>,
        on_stack: BTreeSet<String>,
        stack: Vec<String>,
        out: Vec<Vec<String>>,
    }
    // The reachable function callees of `f`, sorted and unique (BTreeSet) for a deterministic walk.
    fn successors(
        env: &Env,
        reachable: &BTreeSet<String>,
        f: &str,
    ) -> Result<BTreeSet<String>, ElabError> {
        Ok(calls_in_fn(f, &env.fns[f].body)?
            .into_iter()
            .filter(|c| reachable.contains(c) && env.fns.contains_key(c))
            .collect())
    }
    // Call-graph recursion (bounded by the reachable function count, not AST nesting — a separate
    // resource from the [`ElabError::DepthExceeded`] AST-traversal budget `successors` may surface).
    fn strongconnect(t: &mut Tarjan, v: &str) -> Result<(), ElabError> {
        t.idx.insert(v.to_owned(), t.index);
        t.low.insert(v.to_owned(), t.index);
        t.index += 1;
        t.stack.push(v.to_owned());
        t.on_stack.insert(v.to_owned());
        for w in successors(t.env, t.reachable, v)? {
            if !t.idx.contains_key(&w) {
                strongconnect(t, &w)?;
                let lw = t.low[&w];
                let lv = t.low.get_mut(v).expect("v indexed");
                *lv = (*lv).min(lw);
            } else if t.on_stack.contains(&w) {
                let iw = t.idx[&w];
                let lv = t.low.get_mut(v).expect("v indexed");
                *lv = (*lv).min(iw);
            }
        }
        if t.low[v] == t.idx[v] {
            let mut scc: Vec<String> = Vec::new();
            loop {
                let w = t.stack.pop().expect("stack non-empty while popping an SCC");
                t.on_stack.remove(&w);
                let is_root = w == v;
                scc.push(w);
                if is_root {
                    break;
                }
            }
            scc.sort(); // deterministic member order (group binding order is observable in the hash)
            t.out.push(scc);
        }
        Ok(())
    }
    let mut t = Tarjan {
        env,
        reachable: &reachable,
        index: 0,
        idx: BTreeMap::new(),
        low: BTreeMap::new(),
        on_stack: BTreeSet::new(),
        stack: Vec::new(),
        out: Vec::new(),
    };
    for f in &reachable {
        if !t.idx.contains_key(f) {
            strongconnect(&mut t, f)?;
        }
    }

    // Keep only the *recursive* SCCs (a multi-member group, or a self-looping singleton), preserving
    // Tarjan's callee-first order.
    let mut sccs = Vec::with_capacity(t.out.len());
    for scc in t.out {
        let recursive =
            scc.len() > 1 || calls_in_fn(&scc[0], &env.fns[&scc[0]].body)?.contains(&scc[0]);
        if recursive {
            sccs.push(scc);
        }
    }
    Ok(sccs)
}

/// The set of function/constructor/prim names a body calls (single-segment heads + bare paths). A
/// superset filter — the caller intersects with `env.fns` to get function calls.
///
/// # Errors
/// [`ElabError::DepthExceeded`] once the shared [`crate::totality::walk_expr`] traversal's own
/// recursion exceeds its explicit budget (M-674) on a pathologically-nested `body` — a clean,
/// explicit refusal rather than a host-stack overflow.
fn calls_in_fn(site: &str, body: &Expr) -> Result<BTreeSet<String>, ElabError> {
    let mut out = BTreeSet::new();
    collect_calls(site, body, &mut out)?;
    Ok(out)
}

fn collect_calls(site: &str, e: &Expr, out: &mut BTreeSet<String>) -> Result<(), ElabError> {
    // Same pre-order traversal totality uses (M-641) — factored into the one shared `walk_expr`;
    // this collector's *action* differs (it gathers **every** single-segment path, the superset
    // filter `calls_in_fn` documents, not just `App` heads), so the visitor closure carries that.
    // `walk_expr` carries its own explicit recursion-depth budget (M-674) — mapped here to the
    // never-silent `ElabError::DepthExceeded`, never a host-stack overflow.
    crate::totality::walk_expr(e, &mut |x| {
        if let Expr::Path(p) = x {
            if p.0.len() == 1 {
                out.insert(p.0[0].clone());
            }
        }
    })
    .map_err(|e| ElabError::DepthExceeded {
        site: site.to_owned(),
        limit: e.limit,
    })
}

/// Build the content-addressed data registry `Σ` (RFC-0001 §4.3 r3) from the checked environment's
/// type declarations, so the elaborator can resolve constructor names to `#T#i` [`CtorRef`]s. A type
/// carrying a field outside the r3 data fragment (e.g. a `Substrate` field) is skipped; if a
/// *reachable* type references it, the registry build fails and the program is honestly `Residual`.
///
/// Public so a differential / a consumer can rebuild the **same** registry the elaborator used
/// (it is a pure, content-addressed function of `env.types`) — e.g. to map an L1 evaluator's
/// name-keyed data value onto the elaborated L0 value's `#T#i` identity (NFR-7).
pub fn build_registry(env: &Env) -> Result<DataRegistry, ElabError> {
    let mut specs: BTreeMap<String, DeclSpec> = BTreeMap::new();
    'types: for (name, d) in &env.types {
        let mut ctors = Vec::with_capacity(d.ctors.len());
        for c in &d.ctors {
            let mut fields = Vec::with_capacity(c.fields.len());
            for f in &c.fields {
                match field_spec(f) {
                    Some(fs) => fields.push(fs),
                    None => continue 'types, // a non-r3 field — skip this type (Residual if used)
                }
            }
            ctors.push(CtorSpec { fields });
        }
        specs.insert(name.clone(), DeclSpec { ctors });
    }
    DataRegistry::build(&specs).map_err(|e| ElabError::Residual {
        site: "<data registry>".to_owned(),
        what: format!("a reachable data type is outside the r3 fragment: {e}"),
    })
}

/// Convert a v0 field type to a registry [`FieldSpec`]; `None` for a type with no monomorphic r3
/// value form. **Stage-1 (RFC-0007 §11.3):** a **generic instantiation** (`Data` with type arguments)
/// or an **abstract type parameter** ([`Ty::Var`]) has no monomorphic registry form — its elaboration
/// is *staged* (monomorphization is the M-657 follow-up). Returning `None` makes [`build_registry`]
/// skip the owning declaration, so any *use* of a generic value surfaces as an explicit `Residual`
/// (never a silent, half-monomorphized artifact — G2/VR-5).
///
/// **`Ty::Fn` (ADR-033/DN-74, M-923).** A function-typed field lowers to the kernel
/// [`FieldSpec::Fn`] primitive — Path A, the type-carrying full-signature encoding DN-74 ratified
/// as the final FLAG-1 disposition (2026-07-02): the parameter type and the return type are each
/// resolved to a [`FieldTyRef`] and folded into an [`FnSig`], so two `Fn`-typed fields with
/// different signatures hash to **distinct** content addresses (the ADR-033 §10.1 same-arity
/// collision is closed at the kernel level; `crates/mycelium-core/src/data.rs` §"ADR-033 FLAG-1
/// Path A"). Curried multi-parameter arrows (`A => B => C`, M-822) compose naturally: each `Ty::Fn`
/// contributes one parameter, with the return position recursing into a nested [`FieldTyRef::Fn`]
/// for the remaining arrow — no separate multi-arity encoding needed. This replaces the former
/// blanket `None` (staged `Residual`) for **concrete** signatures; a signature whose param/return
/// resolves through a still-open type variable, an unresolved width, or a generic `Data`
/// instantiation stays honestly staged (`ty_to_field_ty_ref` returns `None` for those leaves,
/// narrowing — never eliminating — the residual, G2/VR-5).
///
/// **Reachability note (honest scope, DN-74 new evidence).** `crate::mono`'s closure
/// defunctionalization (RFC-0024 §4A/M-704) unconditionally rewrites every reachable `Ty::Fn` field
/// into a closure tag-sum `Ty::Data` reference *before* [`elaborate`] ever calls [`build_registry`]
/// — so through the standard `elaborate` entry point this arm is not reached (defunctionalization
/// already produces a closed, executing L0 term for every such field, verified by the existing
/// closures three-way differential). This `FieldSpec::Fn` lowering is exercised through
/// [`build_registry`] called directly, and through [`elaborate_direct`] (below) — the narrow,
/// additive entry point that targets the kernel primitive `elaborate` cannot reach without also
/// changing `mono.rs`'s defunctionalization scope (out of this leaf's owned files).
pub(crate) fn field_spec(ty: &Ty) -> Option<FieldSpec> {
    Some(match ty {
        Ty::Binary(crate::checkty::Width::Lit(n)) => FieldSpec::Repr(Repr::Binary { width: *n }),
        Ty::Binary(crate::checkty::Width::Var(_)) => return None, // width-var must not reach elab
        Ty::Ternary(crate::checkty::Width::Lit(m)) => FieldSpec::Repr(Repr::Ternary { trits: *m }),
        Ty::Ternary(crate::checkty::Width::Var(_)) => return None, // width-var must not reach elab
        Ty::Dense(d, s) => FieldSpec::Repr(Repr::Dense {
            dim: *d,
            dtype: scalar_kind(*s),
        }),
        // RFC-0003 §3 (M-892): the VSA repr has a monomorphic kernel form (the model id in the
        // checked type is already the canonical kernel id).
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => FieldSpec::Repr(Repr::Vsa {
            model: model.clone(),
            dim: *dim,
            sparsity: sparsity_class(sparsity),
        }),
        // RFC-0032 D3/D4: the sequence/byte-string reprs have monomorphic kernel forms.
        Ty::Seq(elem, n) => FieldSpec::Repr(Repr::Seq {
            elem: Box::new(ty_to_repr(elem)?),
            len: *n,
        }),
        Ty::Bytes => FieldSpec::Repr(Repr::Bytes),
        // ADR-040 (M-897): the scalar float has a monomorphic kernel form (binary64 — FLAG-1).
        Ty::Float => FieldSpec::Repr(Repr::Float {
            width: FloatWidth::F64,
        }),
        Ty::Data(n, args) if args.is_empty() => FieldSpec::Data(n.clone()),
        Ty::Data(_, _) | Ty::Var(_) => return None,
        Ty::Substrate(_) => return None,
        // ADR-033/DN-74 (M-923): a function-typed field lowers to `FieldSpec::Fn { arity, sig }`
        // (Path A — the full param+return signature, see the doc comment above). `arity` is always
        // 1 at this level: `Ty::Fn` is a single-parameter arrow (curried multi-arg values are
        // nested arrows, RFC-0024 §4A.5/M-822), and the nesting composes through the recursive
        // `FieldTyRef::Fn` case in `ty_to_field_ty_ref`.
        Ty::Fn(param, ret) => {
            let param = ty_to_field_ty_ref(param)?;
            let ret = ty_to_field_ty_ref(ret)?;
            FieldSpec::Fn {
                arity: 1,
                sig: FnSig {
                    arity: 1,
                    params: vec![param],
                    ret: Box::new(ret),
                },
            }
        }
    })
}

/// Convert a **representation** [`Ty`] to its kernel [`Repr`] — the element-type resolver for a
/// `Seq{T, N}` field (RFC-0032 D3). Returns `None` for any non-representation type (a `Data`/`Var`/
/// `Fn` element has no monomorphic kernel repr in stage-1), so a `Seq` of such an element is itself
/// staged (`field_spec` returns `None`) rather than half-elaborated (never a silent artifact —
/// G2/VR-5).
pub(crate) fn ty_to_repr(ty: &Ty) -> Option<Repr> {
    Some(match ty {
        Ty::Binary(crate::checkty::Width::Lit(n)) => Repr::Binary { width: *n },
        Ty::Binary(crate::checkty::Width::Var(_)) => return None, // width-var must not reach elab
        Ty::Ternary(crate::checkty::Width::Lit(m)) => Repr::Ternary { trits: *m },
        Ty::Ternary(crate::checkty::Width::Var(_)) => return None, // width-var must not reach elab
        Ty::Dense(d, s) => Repr::Dense {
            dim: *d,
            dtype: scalar_kind(*s),
        },
        // RFC-0003 §3 (M-892): the VSA repr resolves concretely (model already canonical).
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => Repr::Vsa {
            model: model.clone(),
            dim: *dim,
            sparsity: sparsity_class(sparsity),
        },
        Ty::Seq(elem, n) => Repr::Seq {
            elem: Box::new(ty_to_repr(elem)?),
            len: *n,
        },
        Ty::Bytes => Repr::Bytes,
        // ADR-040 (M-897): the scalar float resolves to its kernel `Repr` (binary64 — FLAG-1).
        Ty::Float => Repr::Float {
            width: FloatWidth::F64,
        },
        Ty::Data(_, _) | Ty::Var(_) | Ty::Substrate(_) | Ty::Fn(_, _) => return None,
    })
}

/// Convert a v0 type to a [`FieldTyRef`] — the leaf type a `Fn`-typed field's signature can hold
/// (ADR-033 §10.2 Path A): a `Repr` leaf (via [`ty_to_repr`]), a *monomorphic* `Data` reference (by
/// build-time name — resolved to a `ContentHash` at [`DataRegistry::build`] time, exactly like a
/// top-level [`FieldSpec::Data`]), or a nested [`FieldTyRef::Fn`] for a higher-order/curried
/// parameter or return. `None` for anything with no monomorphic form here — a generic `Data`
/// instantiation, an unresolved [`Ty::Var`]/width, or [`Ty::Substrate`] (M-923; mirrors
/// [`field_spec`]'s own staging: a signature leaf that cannot resolve keeps the *owning* `Fn`
/// field staged, never a half-encoded signature — G2/VR-5).
pub(crate) fn ty_to_field_ty_ref(ty: &Ty) -> Option<FieldTyRef> {
    Some(match ty {
        Ty::Data(n, args) if args.is_empty() => FieldTyRef::Data(n.clone()),
        Ty::Data(_, _) | Ty::Var(_) | Ty::Substrate(_) => return None,
        Ty::Fn(param, ret) => {
            let param = ty_to_field_ty_ref(param)?;
            let ret = ty_to_field_ty_ref(ret)?;
            FieldTyRef::Fn(Box::new(FnSig {
                arity: 1,
                params: vec![param],
                ret: Box::new(ret),
            }))
        }
        // Every other v0 type is a representation type — delegate to the shared resolver so the
        // `Repr` leaf encoding can never drift between a `Fn` signature and an ordinary field.
        _ => FieldTyRef::Repr(ty_to_repr(ty)?),
    })
}

/// The `Scalar` → kernel `ScalarKind` mapping (shared with [`type_repr`]).
pub(crate) fn scalar_kind(s: Scalar) -> ScalarKind {
    match s {
        Scalar::F16 => ScalarKind::F16,
        Scalar::Bf16 => ScalarKind::Bf16,
        Scalar::F32 => ScalarKind::F32,
        Scalar::F64 => ScalarKind::F64,
    }
}

/// The surface `Sparsity` → kernel `SparsityClass` mapping (M-892; shared with [`type_repr`]).
pub(crate) fn sparsity_class(sp: &Sparsity) -> SparsityClass {
    match sp {
        Sparsity::Dense => SparsityClass::Dense,
        Sparsity::Sparse(k) => SparsityClass::Sparse { max_active: *k },
    }
}

/// The elaboration context: the checked environment, the data registry `Σ`, a fresh-name counter
/// (for inlining + match/lambda binders), and the **recursion scope** — the reachable self-recursive
/// functions mapped to their kernel `Fix` variables (RFC-0001 r4). A call to a name in `rec`
/// elaborates to an `App` chain on its `Fix` var; every other function call still **inlines**.
/// The elaborator's **explicit expression-nesting budget** — the elaborator's twin of
/// `checkty::MAX_CHECK_DEPTH` and `parse::MAX_EXPR_DEPTH` (banked guard 4; A4-02). Elaboration runs on
/// the deep worker stack ([`mycelium_stack`]) and only ever sees a *checked* program (so its depth is
/// already ≤ the checker's budget), but it carries its own reified budget so it is **self-sufficient**:
/// fed a hand-built `Env` straight through the API, it refuses past this with a clean [`ElabError`],
/// never a host-stack overflow. Same value as the checker — the deep worker stack accommodates it with
/// the same ~6× margin (measured ~24,600-level physical ceiling).
const MAX_ELAB_DEPTH: u32 = 4096;

pub(crate) struct Elab<'e> {
    pub(crate) env: &'e Env,
    pub(crate) registry: DataRegistry,
    pub(crate) fresh: u32,
    pub(crate) rec: BTreeMap<String, String>,
    /// Live expression-nesting depth for the explicit [`MAX_ELAB_DEPTH`] budget.
    pub(crate) depth: u32,
}

impl Elab<'_> {
    /// A fresh kernel variable for surface name `base`. `%` is not an identifier character in the
    /// surface lexer, so fresh names can never capture or collide with surface binders.
    fn fresh(&mut self, base: &str) -> String {
        let n = self.fresh;
        self.fresh += 1;
        format!("{base}%{n}")
    }

    /// The `#T#i` [`CtorRef`] for constructor `name`, resolved through the same `Env::ctor` lookup
    /// the checker uses (so the elaborator and the L1 evaluator agree on constructor identity).
    fn ctor_ref(&self, name: &str) -> Option<CtorRef> {
        let (d, i) = self.env.ctor(name)?;
        self.registry.ctor_ref(&d.name, u32::try_from(i).ok()?)
    }

    /// The surface→type view of `scope`, for re-inferring a scrutinee/bound type.
    fn ty_scope(scope: &[Binding]) -> Vec<(String, Ty)> {
        scope
            .iter()
            .map(|(s, _, t)| (s.clone(), t.clone()))
            .collect()
    }

    /// Depth-guarded entry to elaboration (banked guard 4): charge one nesting level against the
    /// explicit [`MAX_ELAB_DEPTH`] budget, refuse past it with a clean [`ElabError`] (never a
    /// host-stack overflow), and release the level on every exit path. Mirrors the parser's
    /// `parse_expr` guard and the checker's `Cx::enter`. All recursive elaboration goes through here.
    fn expr(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        e: &Expr,
    ) -> Result<Node, ElabError> {
        self.depth += 1;
        if self.depth > MAX_ELAB_DEPTH {
            self.depth -= 1;
            let site = stack.last().map_or("<elaborate>", String::as_str);
            return residual(
                site,
                format!(
                    "expression nesting exceeds the elaborator depth budget ({MAX_ELAB_DEPTH}) — an \
                     explicit budget (banked guard 4), refused cleanly rather than overflowing the \
                     host stack (RFC-0007 §4.6 clocked-recursion discipline)"
                ),
            );
        }
        let r = self.expr_inner(stack, scope, e);
        self.depth -= 1;
        r
    }

    /// Elaborate `e` under `scope` (surface name → kernel variable + type). `stack` is the call
    /// path — the cycle (recursion) detector and the error site. Always entered via [`Self::expr`]
    /// (the depth-budget guard), so it — and every `self.expr(…)` it makes — is depth-bounded.
    fn expr_inner(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        e: &Expr,
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("stack starts with the entry").clone();
        let site = site.as_str();
        match e {
            // RFC-0032 D3 (M-749): a list literal `[e1, …]` lowers to a `Repr::Seq` const. Each
            // element is elaborated; it must reduce to a `Node::Const` value (the v0 surface only
            // constructs a `Seq` from constant elements — a non-const element is a never-silent
            // `Residual`, G2). The element repr is taken from the first element; the checker has
            // already verified homogeneity, so the `Repr::Seq` well-formedness check is a final
            // never-silent guard (a malformed/heterogeneous seq is refused, never silently built).
            Expr::Lit(Literal::List(elems)) => {
                let mut vals = Vec::with_capacity(elems.len());
                for el in elems {
                    // `mycelium_core::Node` now has a manual `Drop` (RFC-0041 §4.5 iterative
                    // destruction), so a by-value field move-out of a `Node` is E0509 — bind, match
                    // by-ref, and `clone` the constant `Value` (shallow by construction). This is the
                    // W3↔W5 E0509 coupling the RFC flags; W5's eval rewrite may prefer a
                    // `mem::replace` here to avoid the clone.
                    let reduced = self.expr(stack, scope, el)?;
                    match &reduced {
                        Node::Const(v) => vals.push(v.clone()),
                        _ => return residual(
                            site,
                            "a list literal element did not reduce to a constant — the v0 `Seq` \
                                 surface constructs from constant elements only (RFC-0032 D3)",
                        ),
                    }
                }
                // The element repr anchors the seq descriptor. An empty seq has no element to read a
                // repr from; the v0 surface refuses an un-ascribed `[]` at check time, so a `[]`
                // reaching elaboration would be one ascribed to a `Seq{T, 0}` — but `lit_value` does
                // not carry that `T`. We therefore refuse a bare empty list here (never silent, G2);
                // the non-empty case (the tested surface) lowers fully.
                let Some(first) = vals.first() else {
                    return residual(
                        site,
                        "an empty list literal `[]` has no element repr to anchor the `Seq` \
                         descriptor at elaboration (RFC-0032 D3); the empty-seq surface is staged",
                    );
                };
                let elem = first.repr().clone();
                let len = u32::try_from(vals.len()).map_or(u32::MAX, |n| n);
                Value::new(
                    Repr::Seq {
                        elem: Box::new(elem),
                        len,
                    },
                    Payload::Seq(vals),
                    Meta::exact(Provenance::Root),
                )
                .map_or_else(
                    |e| residual(site, format!("malformed sequence literal: {e}")),
                    |v| Ok(Node::Const(v)),
                )
            }
            Expr::Lit(l) => Ok(Node::Const(lit_value(site, l)?)),
            Expr::Path(p) => {
                if p.0.len() == 1 {
                    let name = &p.0[0];
                    if let Some((_, kvar, _)) = scope.iter().rev().find(|(s, _, _)| s == name) {
                        return Ok(Node::Var(kvar.clone()));
                    }
                    // A bare reference to a recursive function is its Fix variable (a nullary
                    // recursive function `loop()` reached this way unfolds when forced — RFC-0001 r4).
                    if let Some(kf) = self.rec.get(name) {
                        return Ok(Node::Var(kf.clone()));
                    }
                    // A bare nullary constructor (Z, Nil, True, …) is a saturated Construct.
                    if self.env.ctor(name).is_some() {
                        let ctor = self.ctor_ref(name).ok_or_else(|| ElabError::Residual {
                            site: site.to_owned(),
                            what: format!("`{name}` is outside the r3 data registry"),
                        })?;
                        return Ok(Node::Construct { ctor, args: vec![] });
                    }
                    // ADR-033/DN-74 (M-923): a bare reference to an ordinary, non-recursive
                    // top-level function in VALUE position (not the head of a call — `app`
                    // handles that case) is the surface form of a `FieldSpec::Fn` payload — e.g.
                    // the argument of `MkDict(eq8)` in a dictionary/dynamic-dispatch construction
                    // (ADR-033 §2.1: "the function is identified by its content hash ... the
                    // actual method body is a value in the term registry"). A top-level function
                    // closes over nothing (RFC-0007 §4.7), so it lowers directly to its own closed
                    // curried lambda term (`Lam`/`App` are already in the RFC-0007 §4.1 node
                    // budget — no new kernel node, KC-3). A **generic** function used this way
                    // stays staged, exactly like a generic function *call* (`app` below) — never a
                    // half-monomorphized artifact (G2/VR-5).
                    //
                    // Reachable only through [`elaborate_direct`]: through the standard
                    // [`elaborate`] entry this arm never fires, because `crate::mono`'s closure
                    // defunctionalization (RFC-0024 §4A/M-704) already rewrites every such
                    // reference into a closure-constructor call before elaboration runs (see the
                    // `field_spec` doc comment for the full reachability note).
                    if let Some(fd) = self.env.fns.get(name) {
                        if !fd.sig.params.is_empty() {
                            return residual(
                                site,
                                format!(
                                    "generic function `{name}<…>` used as a value has no L0 form \
                                     yet (monomorphization staged — RFC-0007 §11.3, the M-657 \
                                     follow-up)"
                                ),
                            );
                        }
                        return self.elab_fn_lam(name);
                    }
                }
                residual(site, format!("unresolved name `{}`", p.0.join(".")))
            }
            Expr::Let {
                name,
                ty: _,
                bound,
                body,
            } => {
                // RFC-0018 (M-663): a `let`'s `@ g` ascription is statically checked + erased (no L0
                // form) — see [`elab_prelude`]. The type part is handled by re-inference below.
                let kbound = self.expr(stack, scope, bound)?;
                // The bound's type (re-inferred) goes into scope so a later `match` on this binding
                // can lower its patterns.
                let bty = infer_type(self.env, &mut Self::ty_scope(scope), bound).map_err(|e| {
                    ElabError::Residual {
                        site: site.to_owned(),
                        what: format!("could not re-infer `let {name}`'s type: {e}"),
                    }
                })?;
                let kvar = self.fresh(name);
                let mut inner = scope.to_vec();
                inner.push((name.clone(), kvar.clone(), bty));
                let kbody = self.expr(stack, &inner, body)?;
                Ok(Node::Let {
                    id: kvar,
                    bound: Box::new(kbound),
                    body: Box::new(kbody),
                })
            }
            Expr::If { cond, conseq, alt } => self.elab_if(stack, scope, cond, conseq, alt),
            Expr::Match { scrutinee, arms } => self.elab_match(stack, scope, scrutinee, arms),
            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => self.elab_for(stack, scope, x, xs, acc, init, body),
            Expr::Swap {
                value,
                target,
                policy,
            } => {
                // RFC-0018 (M-663): a `swap` target's `@ g` is statically checked + erased; the swap
                // is the endorsement point whose certificate is validated at elaboration/runtime
                // (R18-Q4), not represented as a grade node in L0.
                let src = self.expr(stack, scope, value)?;
                let target_repr = type_repr(site, target)?;
                // DN-52 FLAG-1 / freeze-ledger (W5): Dense is accepted by the checker (RFC-0002 /
                // RFC-0005) and `type_repr` resolves it to `Repr::Dense{..}`, but the standard
                // three-way harness uses `BinaryTernarySwapEngine` which only covers Binary↔Ternary.
                // Without this guard, `elaborate` would return `Ok(Node::Swap{Dense})` while every
                // runner (L0-interp, AOT, L1-eval) refuses explicitly — an elaboration-level silent
                // gap in the DN-50 narrow gate. Resolution: emit an explicit `Residual` so EVERY path
                // is consistent (never-silent, G2). A Dense-capable swap engine (E2-1 / ADR-033
                // `FieldSpec::Fn` wave) lifts this `Residual` when it lands.
                if matches!(target_repr, Repr::Dense { .. }) {
                    return residual(
                        site,
                        "Dense swap targets are staged — the standard three-way harness \
                         (BinaryTernarySwapEngine) does not cover Dense conversions; a Dense swap \
                         engine lands with E2-1/ADR-033 (DN-52 FLAG-1 → Explicit-Residual; \
                         freeze-ledger W5)",
                    );
                }
                Ok(Node::Swap {
                    src: Box::new(src),
                    target: target_repr,
                    policy: policy_name_ref(policy),
                })
            }
            Expr::WithParadigm { .. } => residual(
                site,
                "internal: a `with paradigm` block reached elaboration — the ambient resolution \
                 pass strips it (RFC-0012 §4.4)",
            ),
            // `wild` (the audited FFI floor — M-661/M-720) lowers to a host-dispatch `Op` in the
            // reserved `wild:` prim namespace (RFC-0028 §4.2/§4.3) — **no new Core-IR node** (KC-3,
            // reusing `Node::Op`). The body is the trusted/opaque escape (a `name(args…)` host-call
            // form); a body that is not that form is an explicit `Residual`, never a fabricated
            // artifact (G2).
            Expr::Wild(body) => self.elab_wild(stack, scope, body),
            Expr::Spore(_) => residual(site, "`spore` is deferred (E2-5/M-260)"),
            // M-904 (DN-71 Model S §4.3, maintainer-accepted 2026-07-02): `consume <expr>` lowers as
            // the **observational-identity move** through existing paths — the affine obligation is
            // discharged statically at check time (M-903's tracker), and `consume` is already
            // move-transparent in `crate::grade` (it neither upgrades nor downgrades the operand's
            // tag), so there is nothing left for L0 to represent beyond the operand itself. No new L0
            // node (KC-3); `Substrate` itself still has no `Repr`/kernel projection (LR-8) — this arm
            // never claims otherwise, it just stops refusing to elaborate the *move*. This lifts the
            // former M-664 residual for this fragment (the M-904 DoD's "the Residual is gone").
            //
            // AOT posture (DN-71 §8 FLAG-8, recorded, not silently dropped): v0 has no acquisition
            // surface that actually produces a `Substrate` value in a running Mycelium program (the
            // `wild` host-call registry grants no op that returns one yet), so no program reaching
            // this arm today can carry a live `Substrate` value into `mycelium-mlir`'s AOT path in
            // practice. Whether a *future* acquisition surface's handle can cross into the AOT
            // kernel-`Value` world is a separate, still-open question owned by that crate — not
            // reopened or silently assumed answered by this arm (out of this leaf's scope).
            Expr::Consume(operand) => self.expr(stack, scope, operand),
            Expr::Colony(hyphae) => self.elab_colony(stack, scope, hyphae),
            // RFC-0024 §4A (M-704): closures are **lowered by monomorphization** (`mono.rs`) — a
            // lambda becomes a tag-sum constructor application + a generated `apply` dispatcher, so a
            // raw `Expr::Lambda` never survives into elaboration (`elaborate` monomorphizes first).
            // This arm is kept as a **defensive, never-silent** invariant (G2): a lambda reaching
            // elaboration is an internal staging bug, surfaced as an explicit `Residual`, never a
            // fabricated artifact.
            Expr::Lambda { .. } => residual(
                site,
                "internal: an `Expr::Lambda` reached elaboration — closures are lowered by \
                 monomorphization before elaborate (RFC-0024 §4A / M-704); this is a staging \
                 invariant break, never a silent accept (G2)",
            ),
            Expr::Ascribe(inner, _t) => {
                // The type part is static and already checked — elaboration is transparent. RFC-0018
                // (M-663): an `@ g` ascription is likewise statically checked (`crate::grade`) + erased.
                self.expr(stack, scope, inner)
            }
            Expr::App { head, args } => self.app(stack, scope, head, args),
            // DN-58 §A.5 (M-667/M-817): `fuse(a, b)`. The **Data** case is desugared upstream by
            // monomorphization to the resolved `Fuse::join` call (`mono.rs` — so it runs three-way as
            // an ordinary inlined call), so a `Fuse` node reaching here is a **repr** fuse. The
            // `Binary` meet is the registered `fuse_join:binary` prim (bitwise-AND, the boolean-lattice
            // greatest-lower-bound — runs three-way; `Empirical` semilattice laws). The other reprs
            // have **no committed canonical meet** in v0 (DN-58 §A.6 F-A3), and a Data `Fuse` node here
            // means mono did not resolve the instance — both are explicit never-silent residuals, never
            // a fabricated lowering (G2). No new L0 node (KC-3): the meet reuses `Node::Op`.
            Expr::Fuse { left, right } => {
                // Re-infer the left operand type to dispatch the lowering (the checker has already
                // verified homogeneity so the inferred head uniquely determines the meet).
                let lty = infer_type(self.env, &mut Self::ty_scope(scope), left).map_err(|e| {
                    ElabError::Residual {
                        site: site.to_owned(),
                        what: format!(
                            "could not re-infer `fuse` left-operand type: {e} (DN-58 §A.5)"
                        ),
                    }
                })?;
                match &lty {
                    Ty::Binary(_) => {
                        let la = self.expr(stack, scope, left)?;
                        let ra = self.expr(stack, scope, right)?;
                        Ok(Node::Op {
                            prim: "fuse_join:binary".to_owned(),
                            args: vec![la, ra],
                        })
                    }
                    Ty::Ternary(_) | Ty::Dense(_, _) | Ty::Bytes | Ty::Seq(_, _) => residual(
                        site,
                        format!(
                            "`fuse` over `{lty}` has no committed semilattice-meet prim in v0 — only \
                             the `Binary` repr meet (`fuse_join:binary`, bitwise-AND) and a user \
                             `Data` type with a `Fuse` instance execute (DN-58 §A.6 F-A3 defers the \
                             other paradigm meets); declare an `impl Fuse` or fuse `Binary` values \
                             (never a fabricated meet — G2)"
                        ),
                    ),
                    _ => residual(
                        site,
                        format!(
                            "internal: a Data-type `fuse` over `{lty}` reached elaboration — \
                             monomorphization desugars a Data `fuse` to the resolved `Fuse::join` \
                             call (DN-58 §A.5); a `Fuse` node here means the instance was not resolved \
                             (never a fabricated lowering — G2)"
                        ),
                    ),
                }
            }
            // DN-58 §B (M-667/M-817): `reclaim(policy) { body }`. The **trusted base** lowers it to its
            // sequential reference — evaluate `policy` for its effect, then yield `body` — exactly as
            // the L1 evaluator runs it (`eval.rs`) and exactly as `colony` keeps the base sequential
            // (M-666): a `Let` binding the policy to a discarded binder, with the body as the result.
            // This runs **three-way** (L1-eval ≡ L0-interp ≡ AOT) with **no new L0 node** (KC-3) and
            // **no prim** (the trusted base cannot depend on `mycelium-std-runtime`). The **real** RT7
            // supervision — the bounded-restart cascade + the `SupervisionRecord` EXPLAIN trail — is a
            // runtime-tier driver (`mycelium-mlir::run_reclaim`, over the lazy body node from
            // `elaborate_reclaim`), validated equal to this reference on success — the same layering
            // the concurrent `colony` executor uses over unchanged per-hypha L0 terms. Never-silent: a
            // body failure propagates through the normal error path here, and is precisely what the
            // supervisor restarts on there (G2). Guarantee: `Empirical` (M-713).
            Expr::Reclaim { policy, body } => {
                let policy_node = self.expr(stack, scope, policy)?;
                let body_node = self.expr(stack, scope, body)?;
                Ok(Node::Let {
                    id: self.fresh("reclaim_policy"),
                    bound: Box::new(policy_node),
                    body: Box::new(body_node),
                })
            }

            // M-826: `TupleLit` nodes are rewritten to `App { head: Path(MkTuple$N), args }` by
            // the checker (`check_tuple_lit`) and then the App is lowered by the `app` arm above.
            // A surviving `TupleLit` in elaboration is a staging bug — never silent (G2).
            Expr::TupleLit(_) => residual(
                site,
                "internal: TupleLit survived to elaboration — the checker should have rewritten \
                 it to a constructor App (M-826; never silent, G2)",
            ),
        }
    }

    /// Lower a `wild { name(a₁,…,aₙ) }` block to a host-dispatch [`Node::Op`] in the reserved
    /// `wild:` prim namespace (RFC-0028 §4.2/§4.3; M-720). The body is the trusted/opaque FFI escape
    /// (M-661): it is **not** type-checked, so only its *shape* is interpreted here — a single
    /// host-call form `name(args…)` or a bare `name` with an undotted host-operation name. The
    /// arguments are ordinary in-scope expressions, elaborated through the normal path. Any other
    /// body shape is an explicit [`ElabError::Residual`] (never a fabricated lowering — G2).
    ///
    /// The resulting `Op { prim: "wild:name", … }` is resolved at runtime through the interpreter's
    /// prim registry — the capability handle (RFC-0028 §4.3) — which registers **no** `wild:` op by
    /// default, so an ungranted host op is a never-silent `UnknownPrim` (G2). The `wild:` prefix is
    /// reserved: no built-in paradigm primitive uses it, so a `wild:`-prefixed `Op` is unambiguously
    /// a host call. **No new Core-IR node** (KC-3).
    fn elab_wild(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        body: &Expr,
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        let (name, args): (&str, &[Expr]) = match body {
            Expr::App { head, args } => match head.as_ref() {
                Expr::Path(p) if p.0.len() == 1 => (p.0[0].as_str(), args.as_slice()),
                _ => {
                    return residual(
                        &site,
                        "a v0 `wild` block body must be a host-call form `name(args…)` with a \
                         single, undotted host-operation name (RFC-0028 §4.2) — never a guess (G2)",
                    )
                }
            },
            Expr::Path(p) if p.0.len() == 1 => (p.0[0].as_str(), &[]),
            _ => {
                return residual(
                    &site,
                    "a v0 `wild` block body must be a host-call form `name(args…)` or a bare \
                     `name` (RFC-0028 §4.2); other body shapes are a future append-only extension \
                     — never a fabricated lowering (G2)",
                )
            }
        };
        let prim = format!("wild:{name}");
        let mut elab_args = Vec::with_capacity(args.len());
        for a in args {
            elab_args.push(self.expr(stack, scope, a)?);
        }
        Ok(Node::Op {
            prim,
            args: elab_args,
        })
    }

    /// `if c then t else e` desugars to a flat `Match` on the prelude `Bool` (RFC-0007 §4.4; the
    /// constructors `True`/`False` come from the same registry the surface checks against).
    fn elab_if(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        cond: &Expr,
        conseq: &Expr,
        alt: &Expr,
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        let cond_node = self.expr(stack, scope, cond)?;
        let true_ref = self.bool_ctor(&site, "True")?;
        let false_ref = self.bool_ctor(&site, "False")?;
        let conseq_node = self.expr(stack, scope, conseq)?;
        let alt_node = self.expr(stack, scope, alt)?;
        let cond_var = self.fresh("cond");
        let m = Node::Match {
            scrutinee: Box::new(Node::Var(cond_var.clone())),
            alts: vec![
                Alt::Ctor {
                    ctor: true_ref,
                    binders: vec![],
                    body: conseq_node,
                },
                Alt::Ctor {
                    ctor: false_ref,
                    binders: vec![],
                    body: alt_node,
                },
            ],
            default: None,
        };
        Ok(Node::Let {
            id: cond_var,
            bound: Box::new(cond_node),
            body: Box::new(m),
        })
    }

    fn bool_ctor(&self, site: &str, name: &str) -> Result<CtorRef, ElabError> {
        self.ctor_ref(name).ok_or_else(|| ElabError::Residual {
            site: site.to_owned(),
            what: format!("the prelude `Bool` constructor `{name}` is missing from the registry"),
        })
    }

    /// Lower a surface `match` to the flat L0 `Match` via the M-320 Maranget decision tree
    /// (RFC-0011 §4.4). Re-infers the scrutinee type, normalises each arm pattern (collecting binder
    /// occurrences), compiles the verified-`Fail`-free decision tree, and threads it into nested L0
    /// `Match` nodes — binding the scrutinee once in an enclosing `Let`.
    fn elab_match(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        scrutinee: &Expr,
        arms: &[Arm],
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        // 1. Re-infer the scrutinee type (the checker validated it; this recomputes it for lowering).
        let sty = infer_type(self.env, &mut Self::ty_scope(scope), scrutinee).map_err(|e| {
            ElabError::Residual {
                site: site.clone(),
                what: format!("could not re-infer the match scrutinee's type: {e}"),
            }
        })?;
        // 2. Elaborate the scrutinee and bind it once (a Match tests sub-values of one value).
        let scrut_node = self.expr(stack, scope, scrutinee)?;
        let scrut_var = self.fresh("scrut");
        // 3. Normalise every arm's pattern → the coverage matrix + per-arm binder occurrences.
        let mut matrix: Vec<Vec<crate::usefulness::Pat>> = Vec::with_capacity(arms.len());
        let mut arm_binders: Vec<Vec<(String, Ty, Vec<usize>)>> = Vec::with_capacity(arms.len());
        for arm in arms {
            let mut binds = Vec::new();
            let pat =
                normalize_pattern(&self.env.types, &site, &arm.pattern, &sty, &[], &mut binds)
                    .map_err(|e| ElabError::Residual {
                        site: site.clone(),
                        what: format!("could not normalise a match pattern: {e}"),
                    })?;
            matrix.push(vec![pat]);
            arm_binders.push(binds);
        }
        // 4. Compile (and re-verify Fail-free) the Maranget decision tree — the untrusted lowering.
        let arm_ix: Vec<usize> = (0..arms.len()).collect();
        let occ_root = [Vec::<usize>::new()];
        // RFC-0041 §4.7: the decision-tree compilation is budget-charged; an over-budget match is a
        // never-silent refusal (it already passed the checker's own compile, so this is defensive).
        let tree = decision::compile(&self.env.types, &matrix, &arm_ix, &occ_root, &[sty])
            .map_err(|e| ElabError::Residual {
                site: site.clone(),
                what: format!(
                    "match compilation exceeded the recursion budget: {e} (RFC-0041 §4.7)"
                ),
            })?;
        if decision::has_reachable_fail(&tree) {
            return residual(
                &site,
                "the match compiled to a decision tree with a reachable Fail (usefulness and the \
                 Maranget compiler disagree) — refusing to emit an unsound L0 Match",
            );
        }
        // 5. Lower the tree to nested L0 Match nodes; the root occurrence is the bound scrutinee.
        let mut occ_map: BTreeMap<Vec<usize>, String> = BTreeMap::new();
        occ_map.insert(Vec::new(), scrut_var.clone());
        let body = self.lower_tree(stack, scope, &tree, &occ_map, arms, &arm_binders)?;
        Ok(Node::Let {
            id: scrut_var,
            bound: Box::new(scrut_node),
            body: Box::new(body),
        })
    }

    /// Lower a Maranget [`Tree`] into nested L0 `Match` nodes. `occ_map` maps each already-bound
    /// occurrence (a path into the scrutinee) to its kernel variable; a `Switch` matches on the
    /// occurrence's variable, a constructor case binds *all* its fields (extending `occ_map`), and a
    /// leaf elaborates the surface arm body with its binders resolved through `occ_map`.
    fn lower_tree(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        tree: &Tree,
        occ_map: &BTreeMap<Vec<usize>, String>,
        arms: &[Arm],
        arm_binders: &[Vec<(String, Ty, Vec<usize>)>],
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        match tree {
            Tree::Leaf(i) => {
                // Bind the arm's pattern binders to the kernel variables at their occurrences, then
                // elaborate the arm body in that extended scope.
                let mut arm_scope = scope.to_vec();
                for (name, ty, occ) in &arm_binders[*i] {
                    let kvar = occ_map.get(occ).ok_or_else(|| ElabError::Residual {
                        site: site.clone(),
                        what: format!(
                            "internal: binder `{name}` at occurrence {occ:?} was not bound by the \
                             decision tree"
                        ),
                    })?;
                    arm_scope.push((name.clone(), kvar.clone(), ty.clone()));
                }
                self.expr(stack, &arm_scope, &arms[*i].body)
            }
            Tree::Fail => residual(
                &site,
                "internal: the decision tree reached a Fail (a checked-exhaustive match must not)",
            ),
            Tree::Switch {
                occurrence,
                cases,
                default,
            } => {
                let scrut_kvar =
                    occ_map
                        .get(occurrence)
                        .cloned()
                        .ok_or_else(|| ElabError::Residual {
                            site: site.clone(),
                            what: format!(
                                "internal: switch occurrence {occurrence:?} is not bound"
                            ),
                        })?;
                let mut alts = Vec::with_capacity(cases.len());
                for (head, subtree) in cases {
                    match head {
                        Head::Ctor(name, arity) => {
                            let ctor = self.ctor_ref(name).ok_or_else(|| ElabError::Residual {
                                site: site.clone(),
                                what: format!("`{name}` is outside the r3 data registry"),
                            })?;
                            // Bind ALL fields (not just the discriminated ones) so every binder
                            // occurrence below is available at the leaf.
                            let binders: Vec<String> =
                                (0..*arity).map(|_| self.fresh(name)).collect();
                            let mut child_map = occ_map.clone();
                            for (j, b) in binders.iter().enumerate() {
                                let mut child = occurrence.clone();
                                child.push(j);
                                child_map.insert(child, b.clone());
                            }
                            let body = self.lower_tree(
                                stack,
                                scope,
                                subtree,
                                &child_map,
                                arms,
                                arm_binders,
                            )?;
                            alts.push(Alt::Ctor {
                                ctor,
                                binders,
                                body,
                            });
                        }
                        Head::Lit(key) => {
                            let value = lit_key_to_value(&site, key)?;
                            let body =
                                self.lower_tree(stack, scope, subtree, occ_map, arms, arm_binders)?;
                            alts.push(Alt::Lit { value, body });
                        }
                    }
                }
                let default_node = match default {
                    Some(d) => Some(Box::new(self.lower_tree(
                        stack,
                        scope,
                        d,
                        occ_map,
                        arms,
                        arm_binders,
                    )?)),
                    None => None,
                };
                Ok(Node::Match {
                    scrutinee: Box::new(Node::Var(scrut_kvar)),
                    alts,
                    default: default_node,
                })
            }
        }
    }

    /// Elaborate an application: prims become `Op` nodes; saturated constructors become `Construct`
    /// nodes; a call to a recursive function (in `self.rec`) becomes a curried `App` on its recursion
    /// variable (`Fix`/`FixGroup`), and every **other** user-function call **inlines** (the residual
    /// non-recursive call graph is acyclic, so inlining terminates).
    fn app(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        head: &Expr,
        args: &[Expr],
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        let site = site.as_str();
        let Expr::Path(p) = head else {
            return residual(site, "v0 application head must be a name (first-order)");
        };
        if p.0.len() != 1 {
            return residual(site, format!("dotted call `{}`", p.0.join(".")));
        }
        let name = &p.0[0];

        // ADR-033/DN-74 (M-923): dynamic dispatch through a `FieldSpec::Fn` payload — `name` is a
        // scope-bound variable of `Ty::Fn` type (e.g. a `match`-projected dictionary field,
        // `Mk(f) => f(v)`), not a call to a top-level function/constructor/prim. Checked *before*
        // every other resolution (lexical scope shadows the global namespace, matching the
        // `Expr::Path` value-position arm above) — this lowers to an ordinary curried `App` on the
        // bound variable (`Lam`/`App` are already in the RFC-0007 §4.1 node budget — no new kernel
        // node, KC-3). Reachable only through [`elaborate_direct`]: on the standard `elaborate`
        // path `mono.rs` has already rewritten every `Ty::Fn` scope binding into a closure tag-sum
        // (`Ty::Data`), so this arm never fires there (see `field_spec`'s doc comment).
        if let Some((_, kvar, sty)) = scope.iter().rev().find(|(s, _, _)| s == name) {
            if matches!(sty, Ty::Fn(_, _)) {
                let mut node = Node::Var(kvar.clone());
                for a in args {
                    let karg = self.expr(stack, scope, a)?;
                    node = Node::App {
                        func: Box::new(node),
                        arg: Box::new(karg),
                    };
                }
                return Ok(node);
            }
        }

        // A call to a recursive function is a curried `App` on its `Fix` variable (RFC-0001 r4) —
        // never inlined (that would loop). Arguments evaluate left-to-right (CBV).
        if let Some(kf) = self.rec.get(name).cloned() {
            let mut node = Node::Var(kf);
            for a in args {
                let karg = self.expr(stack, scope, a)?;
                node = Node::App {
                    func: Box::new(node),
                    arg: Box::new(karg),
                };
            }
            return Ok(node);
        }

        if let Some(fd) = self.env.fns.get(name) {
            // Stage-1 (RFC-0007 §11.3): a **generic** function call is type-checked (the checker
            // instantiates it) but its elaboration to closed L0 is *staged* — monomorphization is the
            // M-657 follow-up. Refuse explicitly (never a silent or half-monomorphized lowering).
            if !fd.sig.params.is_empty() {
                return residual(
                    site,
                    format!(
                        "generic function `{name}<…>` type-checks, but lowering a generic \
                         instantiation to L0 is staged (monomorphization — RFC-0007 §11.3; the \
                         M-657 follow-up). This call has no L0 form yet."
                    ),
                );
            }
            // A non-recursive call inlines. Any function in a cycle (self or mutual) is in `self.rec`
            // and was handled by the recursion-variable branch above, so reaching here while `name`
            // is on the inline stack would mean a cycle escaped SCC detection — keep an explicit
            // guard as defense in depth (an internal invariant), never a silent inline loop.
            if stack.iter().any(|f| f == name) {
                return residual(
                    site,
                    format!(
                        "`{name}` is in a call cycle that was not registered as recursive — internal \
                         elaboration invariant (every cycle should lower to `Fix`/`FixGroup`)"
                    ),
                );
            }
            // RFC-0018 (M-663): `{name}`'s return/parameter `@ g` guarantees are statically checked
            // (`crate::grade`) and erased here — a grade has no L0 form (KC-3). Inline as usual.
            // Inline: Let-bind each argument left-to-right (preserving CBV evaluation order),
            // then elaborate the callee body with its parameters mapped to the fresh binders.
            // The callee sees *only* its parameters (top-level functions close over nothing).
            let mut bindings = Vec::new();
            for (param, arg) in fd.sig.value_params.iter().zip(args) {
                let karg = self.expr(stack, scope, arg)?;
                // Monomorphic callee (generic callees are staged out above) — no type params in scope.
                let pty = resolve_ty(site, &self.env.types, &[], &param.ty)
                    .map(|(t, _)| t)
                    .map_err(|e| ElabError::Residual {
                        site: site.to_owned(),
                        what: format!("could not resolve `{name}`'s parameter type: {e}"),
                    })?;
                bindings.push((param.name.clone(), self.fresh(&param.name), karg, pty));
            }
            let callee_scope: Vec<Binding> = bindings
                .iter()
                .map(|(s, k, _, t)| (s.clone(), k.clone(), t.clone()))
                .collect();
            stack.push(name.clone());
            let body = self.expr(stack, &callee_scope, &fd.body)?;
            stack.pop();
            // Wrap right-to-left so the leftmost argument's Let is outermost (evaluated first).
            let node = bindings
                .into_iter()
                .rev()
                .fold(body, |acc, (_, kvar, karg, _)| Node::Let {
                    id: kvar,
                    bound: Box::new(karg),
                    body: Box::new(acc),
                });
            return Ok(node);
        }

        // A saturated constructor application builds a data value (W6 saturation is already checked).
        if self.env.ctor(name).is_some() {
            let ctor = self.ctor_ref(name).ok_or_else(|| ElabError::Residual {
                site: site.to_owned(),
                what: format!("`{name}` is outside the r3 data registry"),
            })?;
            let mut kargs = Vec::with_capacity(args.len());
            for a in args {
                kargs.push(self.expr(stack, scope, a)?);
            }
            return Ok(Node::Construct { ctor, args: kargs });
        }

        if let Some(kernel) = prim_kernel_name(name) {
            let mut kargs = Vec::new();
            for a in args {
                kargs.push(self.expr(stack, scope, a)?);
            }
            return Ok(Node::Op {
                prim: kernel.to_owned(),
                args: kargs,
            });
        }

        // An **unqualified trait-method call** (RFC-0019 §4.4) type-checks (the checker resolved the
        // instance / bound), but its L0 lowering is **dictionary-passing**, staged identically to a
        // generic instantiation — RFC-0007 §12.3. Refuse with an explicit `Residual` (never a silent
        // or fabricated artifact — G2); it lands with the monomorphization follow-up (M-673).
        if self
            .env
            .traits
            .values()
            .any(|tr| tr.sigs.iter().any(|s| s.name == *name))
        {
            return residual(
                site,
                format!(
                    "trait-method call `{name}` type-checks, but dictionary-passing lowering to L0 \
                     is staged (RFC-0019 §4.4 / RFC-0007 §12.3; the M-673 follow-up). No L0 form yet."
                ),
            );
        }

        residual(site, format!("unknown function/constructor/prim `{name}`"))
    }

    /// Elaborate a reachable **self-recursive** function `fname` to `Fix(kf, λparams. body)` — the
    /// closed form r4 uses for direct recursion (RFC-0007 §4.1; the v0 surface is first-order, so the
    /// body is closed except for its params, `kf`, and the other recursive functions in scope).
    fn elab_recursive_fn(&mut self, fname: &str, kf: &str) -> Result<Node, ElabError> {
        Ok(Node::Fix {
            name: kf.to_owned(),
            body: Box::new(self.elab_fn_lam(fname)?),
        })
    }

    /// Elaborate `fname` to its curried lambda `λp1. … λpn. body` (params `p1` outermost), with the
    /// body in scope of the params and **every** recursion variable in `self.rec` (its own name plus
    /// any sibling in its group). This is the recursion-variable-agnostic core shared by
    /// [`Self::elab_recursive_fn`] (which wraps it in a `Fix`) and the `FixGroup` group lowering
    /// (which binds the lambdas of a mutually-recursive SCC together — RFC-0001 r5).
    pub(crate) fn elab_fn_lam(&mut self, fname: &str) -> Result<Node, ElabError> {
        let fd = self.env.fns[fname].clone();
        // Stage-1 (RFC-0007 §11.3): a generic (recursive) function type-checks, but lowering it to a
        // closed L0 lambda requires monomorphization (the M-657 follow-up) — staged, never silent.
        if !fd.sig.params.is_empty() {
            return residual(
                fname,
                format!(
                    "generic function `{fname}<…>` type-checks, but its L0 lowering is staged \
                     (monomorphization — RFC-0007 §11.3; the M-657 follow-up)"
                ),
            );
        }
        // RFC-0018 (M-663): `{fname}`'s return/parameter `@ g` guarantees are statically checked
        // (`crate::grade`) and erased here — a grade has no L0 form (KC-3). Lower the body as usual.
        let mut scope: Vec<Binding> = Vec::new();
        let mut param_kvars: Vec<String> = Vec::new();
        for p in &fd.sig.value_params {
            let kp = self.fresh(&p.name);
            // Monomorphic body (generic fns are staged out above) — no type params in scope.
            let pty = resolve_ty(fname, &self.env.types, &[], &p.ty)
                .map(|(t, _)| t)
                .map_err(|e| ElabError::Residual {
                    site: fname.to_owned(),
                    what: format!("could not resolve `{fname}`'s parameter type: {e}"),
                })?;
            scope.push((p.name.clone(), kp.clone(), pty));
            param_kvars.push(kp);
        }
        let mut stack = vec![fname.to_owned()];
        let body = self.expr(&mut stack, &scope, &fd.body)?;
        // Curry: λp1. λp2. … body (p1 outermost).
        Ok(param_kvars
            .into_iter()
            .rev()
            .fold(body, |acc, kp| Node::Lam {
                param: kp,
                body: Box::new(acc),
            }))
    }

    /// Elaborate `for x in xs, acc = init => body` to its synthesized self-recursive fold (RFC-0007
    /// §4.8), as an inline `Fix` over the linearly-recursive spine type:
    ///
    /// ```text
    /// App(App(Fix(fold, λs. λa. Match s {
    ///            Nil          => a,
    ///            Cons(x,rest) => App(App(fold, rest), body[acc↦a]) }),
    ///         xs), init)
    /// ```
    ///
    /// The nil/cons shape was already validated by the checker (`linear_elem_ty`); here we just read
    /// off the element/spine field positions from the registry.
    #[allow(clippy::too_many_arguments)]
    fn elab_for(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        x: &str,
        xs: &Expr,
        acc: &str,
        init: &Expr,
        body: &Expr,
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        let sty = infer_type(self.env, &mut Self::ty_scope(scope), xs).map_err(|e| {
            ElabError::Residual {
                site: site.clone(),
                what: format!("could not infer the `for` spine type: {e}"),
            }
        })?;
        let Ty::Data(tname, _) = &sty else {
            return residual(&site, format!("`for` spine is not a data type: {sty}"));
        };
        let d = self
            .env
            .types
            .get(tname)
            .ok_or_else(|| ElabError::Residual {
                site: site.clone(),
                what: format!("unknown type `{tname}`"),
            })?
            .clone();
        // Find the nil constructor (no fields) and the cons constructor (one spine field of type
        // `tname` + one element field).
        let mut nil_name: Option<String> = None;
        let mut cons: Option<(String, usize, usize, Ty)> = None; // (name, elem_idx, spine_idx, elem_ty)
        for c in &d.ctors {
            if c.fields.is_empty() {
                nil_name = Some(c.name.clone());
                continue;
            }
            let Some(spine_idx) = c
                .fields
                .iter()
                .position(|f| matches!(f, Ty::Data(n, _) if n == tname))
            else {
                return residual(
                    &site,
                    format!("`for` constructor `{}` has no spine field", c.name),
                );
            };
            let Some(elem_idx) = (0..c.fields.len()).find(|&i| i != spine_idx) else {
                return residual(
                    &site,
                    format!("`for` constructor `{}` has no element field", c.name),
                );
            };
            cons = Some((
                c.name.clone(),
                elem_idx,
                spine_idx,
                c.fields[elem_idx].clone(),
            ));
        }
        let (Some(nil_name), Some((cons_name, _elem_idx, spine_idx, elem_ty))) = (nil_name, cons)
        else {
            return residual(
                &site,
                format!("`for` needs a nil + cons shape on `{tname}`"),
            );
        };
        let aty = infer_type(self.env, &mut Self::ty_scope(scope), init).map_err(|e| {
            ElabError::Residual {
                site: site.clone(),
                what: format!("could not infer the `for` accumulator type: {e}"),
            }
        })?;
        let xs_node = self.expr(stack, scope, xs)?;
        let init_node = self.expr(stack, scope, init)?;

        // Fresh kernel vars for the synthesized fold.
        let fold = self.fresh("fold");
        let s_kv = self.fresh("s");
        let a_kv = self.fresh("acc");
        let elem_kv = self.fresh(x);
        let spine_kv = self.fresh("rest");
        let cons_arity = d
            .ctors
            .iter()
            .find(|c| c.name == cons_name)
            .expect("cons ctor present")
            .fields
            .len();
        let binders: Vec<String> = (0..cons_arity)
            .map(|i| {
                if i == spine_idx {
                    spine_kv.clone()
                } else {
                    elem_kv.clone()
                }
            })
            .collect();

        // The loop body, with `x` ↦ the element binder and `acc` ↦ the accumulator parameter.
        let mut body_scope = scope.to_vec();
        body_scope.push((x.to_owned(), elem_kv.clone(), elem_ty));
        body_scope.push((acc.to_owned(), a_kv.clone(), aty));
        let body_node = self.expr(stack, &body_scope, body)?;

        let nil_ref = self
            .ctor_ref(&nil_name)
            .ok_or_else(|| ElabError::Residual {
                site: site.clone(),
                what: format!("`{nil_name}` is outside the r3 data registry"),
            })?;
        let cons_ref = self
            .ctor_ref(&cons_name)
            .ok_or_else(|| ElabError::Residual {
                site: site.clone(),
                what: format!("`{cons_name}` is outside the r3 data registry"),
            })?;
        // Cons arm body: App(App(fold, rest), body[acc↦a]).
        let rec_call = Node::App {
            func: Box::new(Node::App {
                func: Box::new(Node::Var(fold.clone())),
                arg: Box::new(Node::Var(spine_kv)),
            }),
            arg: Box::new(body_node),
        };
        let match_node = Node::Match {
            scrutinee: Box::new(Node::Var(s_kv.clone())),
            alts: vec![
                Alt::Ctor {
                    ctor: nil_ref,
                    binders: vec![],
                    body: Node::Var(a_kv.clone()),
                },
                Alt::Ctor {
                    ctor: cons_ref,
                    binders,
                    body: rec_call,
                },
            ],
            default: None,
        };
        let fix = Node::Fix {
            name: fold,
            body: Box::new(Node::Lam {
                param: s_kv,
                body: Box::new(Node::Lam {
                    param: a_kv,
                    body: Box::new(match_node),
                }),
            }),
        };
        // App(App(fix, xs), init) — walk the spine head-to-tail from the initial accumulator.
        Ok(Node::App {
            func: Box::new(Node::App {
                func: Box::new(fix),
                arg: Box::new(xs_node),
            }),
            arg: Box::new(init_node),
        })
    }

    /// Elaborate `colony { hypha e1, …, hypha eN }` to its **RT2 spawn-order sequentialization**
    /// (RFC-0008 §4.2/RT2; M-666). RFC-0008 makes the *reference semantics* of a deterministic
    /// concurrent program its deterministic sequentialization, and content-addressing/the NFR-7
    /// differential are over that reference — so the honest L0 form is the sequentialization, **not**
    /// a concurrency node (the L0 Core IR has none; the trusted base stays sequential — KC-3). It
    /// lowers to a chain of `Let`s that evaluates each leading hypha for its (sequentialized) effect,
    /// in order, and yields the **last** hypha's value:
    ///
    /// ```text
    /// Let(_1, ⟦e1⟧, Let(_2, ⟦e2⟧, … ⟦eN⟧))      (each _i a fresh, `%`-named unused binder)
    /// ```
    ///
    /// Nothing is dropped silently (G2): every hypha body is elaborated and bound, so a leading
    /// hypha's refusal/divergence is preserved under CBV (RT4/I1). This sequentialization is the
    /// **RT2 reference ORACLE**: the real concurrent executor (`mycelium-mlir::runtime` —
    /// `Scope`/`Colony`/`Task`, structured fork/join, M-357) runs the colony's per-hypha L0 programs
    /// ([`elaborate_colony`]) as concurrent tasks and is **validated equal to this reference** (the
    /// RT2 differential, `mycelium_mlir::run_colony`); a divergence is an explicit error, never a
    /// silent race (G2/RT4). The concurrent run adds **no L0 kernel node** — the trusted base stays
    /// sequential (RFC-0008 §4.2; KC-3).
    ///
    /// Honesty (Declared at the surface; the lowering itself adds no guarantee): this realizes the
    /// *deterministic* R1 fragment (RFC-0008 §4.6 R1) only. With no v0 product type the colony's
    /// observable is the last hypha's value (the sequential reference's final step), never a
    /// fabricated join-product.
    fn elab_colony(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        hyphae: &[crate::ast::Hypha],
    ) -> Result<Node, ElabError> {
        let site = stack.last().expect("non-empty").clone();
        let Some((last, leading)) = hyphae.split_last() else {
            return residual(
                &site,
                "internal: an empty `colony` reached elaboration — the parser requires ≥ 1 hypha \
                 (RFC-0008 §4.7)",
            );
        };
        // The last hypha is the colony's observable (the RT2 sequentialization's final step).
        let mut node = self.elab_hypha_node(stack, scope, &site, last)?;
        // Wrap right-to-left so the first hypha's `Let` ends up outermost (evaluated first, CBV).
        for h in leading.iter().rev() {
            let bound = self.elab_hypha_node(stack, scope, &site, h)?;
            // A fresh `%`-named binder: `%` is not a surface identifier char, so it never captures a
            // surface name, and the binding is intentionally unused (the value is sequentialized for
            // its effect only). The leading hypha is still fully evaluated under CBV.
            let kvar = self.fresh("hypha");
            node = Node::Let {
                id: kvar,
                bound: Box::new(bound),
                body: Box::new(node),
            };
        }
        Ok(node)
    }

    /// Elaborate one hypha's own contribution node, prefixed with its `@forage(policy)` policy
    /// evaluation if present (RFC-0008 RT3; DN-63 §3.5; M-906/DN-70 D1) —
    /// `Let{_=policy_node, node}`, mirroring `reclaim`'s `Let{_=policy, body}` sequential-reference
    /// shape (DN-58 §B) exactly: the policy is evaluated for its effect (semantics-free placement,
    /// RT3 — it never changes `node`'s value) after the static empty-candidate-set check
    /// ([`forage_reject_if_empty`]) has already refused an all-zero bitmask. No new L0 node (KC-3).
    fn elab_hypha_node(
        &mut self,
        stack: &mut Vec<String>,
        scope: &[Binding],
        site: &str,
        h: &crate::ast::Hypha,
    ) -> Result<Node, ElabError> {
        forage_reject_if_empty(site, h)?;
        let node = self.expr(stack, scope, &h.body)?;
        let Some(policy) = &h.forage else {
            return Ok(node);
        };
        let policy_node = self.expr(stack, scope, policy)?;
        Ok(Node::Let {
            id: self.fresh("forage_policy"),
            bound: Box::new(policy_node),
            body: Box::new(node),
        })
    }
}

/// Statically validate a hypha's `@forage(policy)` D-lite bitmask (RFC-0008 RT3; DN-63 §3.5
/// FLAG-14; M-906/DN-70 D1). The checker guarantees `policy` is `Expr::Lit(Literal::Bin(_))` when
/// present ([`crate::checkty::Cx::check_forage_policy`]); a defensive [`ElabError::Residual`]
/// covers the (unreachable-on-a-checked-env) alternative — never a fabricated lowering (G2). An
/// **all-zero mask** is the DN-63 FLAG-14 empty-candidate-set case: refused here, explicitly, as
/// an [`ElabError::Residual`] (so neither elaborated path — L0-interp nor AOT — ever silently
/// accepts a no-candidate forage) — the L1 evaluator refuses the *identical* source with a typed
/// [`crate::eval::L1Error::Forage`] (`ForageError::NoCandidates`); see `differential.rs`'s
/// `forage_no_candidates_is_an_explicit_refusal_on_every_path` for the three-way consistency
/// check.
fn forage_reject_if_empty(site: &str, h: &crate::ast::Hypha) -> Result<(), ElabError> {
    let Some(policy) = &h.forage else {
        return Ok(());
    };
    let Expr::Lit(Literal::Bin(s)) = policy.as_ref() else {
        return residual(
            site,
            "internal: `@forage(policy)` reached elaboration with a non-literal policy — the \
             checker requires a literal binary bitmask (M-906/DN-70 D1); never a fabricated \
             lowering (G2)",
        );
    };
    let popcount = s.chars().filter(|c| *c == '1').count();
    if popcount == 0 {
        return residual(
            site,
            "`@forage(policy)` has an all-zero worker-availability bitmask — the D-lite \
             single-node candidate set is empty (DN-63 §3.5 FLAG-14); the L1 evaluator refuses \
             this identically with an explicit `ForageError::NoCandidates` (never a silent \
             placement — G2/RT4)",
        );
    }
    Ok(())
}

/// Reconstruct the L0 [`Value`] of a literal-pattern key (`b:1010` / `t:+0-`) produced by the
/// checker's `literal_key` (the `_` separators already normalised away). The width is the digit
/// count (Q6: a literal *is* its representation).
fn lit_key_to_value(site: &str, key: &str) -> Result<Value, ElabError> {
    if let Some(bits) = key.strip_prefix("b:") {
        let bits: Vec<bool> = bits.chars().map(|c| c == '1').collect();
        let width = u32::try_from(bits.len()).expect("digit count fits u32");
        Value::new(
            Repr::Binary { width },
            Payload::Bits(bits),
            Meta::exact(Provenance::Root),
        )
        .map_or_else(
            |e| residual(site, format!("malformed binary literal key: {e}")),
            Ok,
        )
    } else if let Some(trits) = key.strip_prefix("t:") {
        let trits: Vec<Trit> = trits
            .chars()
            .map(|c| match c {
                '+' => Ok(Trit::Pos),
                '0' => Ok(Trit::Zero),
                '-' => Ok(Trit::Neg),
                other => Err(other),
            })
            .collect::<Result<_, _>>()
            .map_or_else(
                |c| residual(site, format!("non-trit char {c:?} in ternary literal key")),
                Ok,
            )?;
        let width = u32::try_from(trits.len()).expect("trit count fits u32");
        Value::new(
            Repr::Ternary { trits: width },
            Payload::Trits(trits),
            Meta::exact(Provenance::Root),
        )
        .map_or_else(
            |e| residual(site, format!("malformed ternary literal key: {e}")),
            Ok,
        )
    } else if let Some(text) = key.strip_prefix("f:") {
        // M-897: a float literal pattern is refused by the checker (`normalize_pattern`, ADR-040
        // FLAG-4) before any decision tree is built, so an `f:` key reaching the L0 bridge is an
        // internal invariant break — refused explicitly, never a silently-picked equality (G2).
        residual(
            site,
            format!(
                "internal: a float literal-pattern key `f:{text}` reached elaboration — the \
                 checker refuses float patterns (ADR-040 FLAG-4)"
            ),
        )
    } else {
        residual(site, format!("unrecognised literal key `{key}`"))
    }
}
