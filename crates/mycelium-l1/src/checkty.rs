//! The **v0 monomorphic typechecker** (RFC-0007 §4.4) plus the program environment it checks
//! against: the data-type registry (declarations are registry entries, never term nodes —
//! RFC-0007 §4.2) and the function table. Every refusal is an explicit [`CheckError`] — generics,
//! `spore`, value-level integers without context, and a `wild` block outside the audited FFI floor
//! (the `@std-sys` context gate, M-661/LR-9) are *refused with a reason*, never guessed at. A `wild`
//! inside a `@std-sys` nodule is the audited FFI escape: gated (it must declare the `ffi` effect),
//! its body trusted/opaque (not recursively checked — audited, not verified, VR-5/ADR-014), and its
//! execution staged ([`crate::elab`] lowers it to a `Residual`).

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use crate::affine::{Tracker, UseOutcome};
use crate::ambient::AmbientError;
use crate::ast::{
    Arm, BaseType, DeriveDecl, Expr, FnDecl, FnSig, Hypha, ImplDecl, Item, Literal, LowerDecl,
    Nodule, ObjectDecl, Paradigm, Param, Path, Pattern, Phylum, Scalar, Sparsity, Strength,
    TraitRef, TypeDecl, TypeRef, UsePath, WidthRef,
};

/// The checker's **explicit expression-nesting budget** (the "banked guard 4" discipline; A4-02).
/// Type-checking recurses on the expression AST; rather than rely on the host call stack to bound
/// that recursion (a resource that varies by thread and by IR frame size — never a semantic limit),
/// the checker carries this reified budget and refuses past it with a clean [`CheckError`], exactly as
/// the parser ([`crate::parse::MAX_EXPR_DEPTH`]) and the evaluator (`eval::DEFAULT_DEPTH`) do for their
/// recursions. It is set comfortably **above** the parser's surface-nesting cap (so it never trips for
/// parser-produced ASTs — it is the defense-in-depth ceiling for an AST handed straight to the checker
/// via the API), and the recursion runs on a deep worker stack ([`mycelium_stack`]) so this budget — not a
/// host-stack overflow — is always what bounds a pathological input.
///
/// **Grounding (measured, not guessed).** The 256 MiB worker stack physically supports **~24,600**
/// levels of `check` recursion in a debug build (empirically: 24,589 survives, 24,765 aborts;
/// ~10.9 KiB/frame — release frames are smaller, so the ceiling is *higher* there). This budget
/// (`4096`) is therefore a **~6× safety margin** below the measured physical ceiling, and **16×**
/// above the parser's 256-deep surface cap — so a real (parsed) program is never within ~16× of it,
/// and even a synthetic AST refuses cleanly with ~6× stack headroom to spare. Raising it is safe up to
/// roughly a third of the physical ceiling; widen the worker stack first if more is ever wanted.
///
/// **Self-hosting:** this explicit budget is the portable primitive (it carries over to the
/// Mycelium-native frontend's clocked bounded-computation model); the worker stack is the transitional
/// Rust-only adapter (`mycelium_stack`).
pub const MAX_CHECK_DEPTH: u32 = 4096;

/// RAII depth accounting for the checker's recursive [`Cx::check`] (paired with [`MAX_CHECK_DEPTH`]):
/// increments the live nesting depth on entry and decrements it on drop, so the budget is honoured on
/// **every** exit path (early `return`, `?`, or fall-through) — never a counter that leaks on error.
struct DepthGuard<'a>(&'a Cell<u32>);

impl Drop for DepthGuard<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get().saturating_sub(1));
    }
}

/// A width argument in a [`Ty::Binary`] or [`Ty::Ternary`] — either a concrete literal or an
/// abstract width parameter (DN-42 / M-753). Width variables (`Var`) are introduced in step (b)
/// and survive only during generic-fn checking; they must not reach elaboration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Width {
    /// A concrete width literal (e.g. `8` in `Binary{8}`).
    Lit(u32),
    /// An abstract width parameter (e.g. `N` in `fn f<N>(x: Binary{N})`; DN-42 / M-753 v1).
    Var(String),
}

impl core::fmt::Display for Width {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Width::Lit(n) => write!(f, "{n}"),
            Width::Var(v) => write!(f, "{v}"),
        }
    }
}

/// A checked type. Stage-0 is monomorphic; stage-1 (RFC-0007 §11) adds **type parameters as
/// abstract variables** ([`Ty::Var`]) and **applied data types** ([`Ty::Data`] with arguments).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    /// `Binary{n}`.
    Binary(Width),
    /// `Ternary{m}`.
    Ternary(Width),
    /// `Dense{d, s}`.
    Dense(u32, Scalar),
    /// `VSA{model, dim, sparsity}` (RFC-0003 §3; M-892 — the type-level lift of the former
    /// blanket "VSA types are deferred" refusal). The `model` is the **kernel model id**
    /// (`"MAP-I"`, `"FHRR"`, `"BSC"`, …) — the surface ident is canonicalized by
    /// [`vsa_kernel_model_id`] at resolution (an ident cannot carry the kernel ids' `-`).
    /// Any model id is admissible as a type *mention* (exactly as `mycelium-core` type-checks
    /// hypervector mentions without the algebra — ADR-008); the `vsa_*` prims are additionally
    /// gated to their dispatch set by [`Checker::try_check_vsa_prim`]. Value construction is
    /// injection-only until a surface hypervector form lands (the M-890 Dense posture).
    Vsa {
        /// The kernel model id (canonicalized — ADR-003: the name here is metadata, matching
        /// `Repr::Vsa { model }`).
        model: String,
        /// Hypervector dimensionality.
        dim: u32,
        /// Declared sparsity (the `vsa_*` prims require `Dense` at introduction — the kernel's
        /// Value-level ops construct dense-class results).
        sparsity: Sparsity,
    },
    /// A registered data type applied to type arguments — `Data("List", [Binary(8)])` is
    /// `List<Binary{8}>`; an empty argument vector is a monomorphic/nullary type (`Data("Bool", [])`).
    /// Content addressing of declarations: RFC-0007 §4.2 (parameterized declarations are one registry
    /// entry); the prototype keys by name since v0 is single-nodule.
    Data(String, Vec<Ty>),
    /// `Substrate{tag}` — the affine external-resource kind (LR-8). No value forms exist in v0.
    Substrate(String),
    /// `Seq{T, N}` — a first-class indexed homogeneous sequence (RFC-0032 D3; M-749): `N` elements
    /// each of element type `T`. Constructed by the `[e1, …]` list literal; the surface checks
    /// elements are homogeneous and the count matches `N` (never-silent — G2). Guarantee: `Exact`
    /// (a literal *is* its representation).
    Seq(Box<Ty>, u32),
    /// `Bytes` — a first-class byte string (RFC-0032 D4; M-750). Constructed by the `0x…` literal.
    /// Guarantee: `Exact`.
    Bytes,
    /// `Float` — the first-class scalar float, IEEE-754 binary64 (ADR-040; M-897). A nullary repr
    /// type like [`Ty::Bytes`] (the width set is F64-only at introduction — ADR-040 FLAG-1).
    /// Constructed by the decimal float literal (`1.5`), which denotes the **correctly-rounded**
    /// (RNE) binary64 of its decimal text (FLAG-3). Guarantee: `Exact` as a *definition* (the
    /// literal denotes exactly that binary64 value); the host-conversion claim is pinned
    /// `Empirical` by the round-trip conformance corpus (ADR-040 §2.6 — VR-5, no silent upgrade).
    Float,
    /// An **abstract type parameter** (a skolem variable) — in scope only while checking a generic
    /// declaration's constructors or a generic function's body (RFC-0007 §11.2). Two `Var`s are equal
    /// iff their names match; that structural equality is the engine of parametric checking. A
    /// `Var`-typed value is **representation-opaque**: no representation-specific `Op` may apply to it
    /// (this is the unbounded-case form of RFC-0019 §4.6's Repr-polymorphism restriction — it falls
    /// out of the abstract-variable discipline, restating S1 at the polymorphic level).
    Var(String),
    /// A **function type** `A -> B` (RFC-0024 §3, M-686): the type of a named top-level function
    /// used as a first-class value. Stage-1 supports single-argument arrows only; the parameter
    /// type and return type may themselves be abstract (`Ty::Var`). A `Fn`-typed value is not a
    /// legal instance head (coherence is over concrete types — same posture as `Ty::Var`).
    ///
    /// Guarantee: `Declared` (a type-level contract; no theorem — VR-5).
    Fn(Box<Ty>, Box<Ty>),
}

impl core::fmt::Display for Ty {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Ty::Binary(n) => write!(f, "Binary{{{n}}}"),
            Ty::Ternary(m) => write!(f, "Ternary{{{m}}}"),
            Ty::Dense(d, s) => write!(f, "Dense{{{d}, {s:?}}}"),
            Ty::Data(n, args) if args.is_empty() => write!(f, "{n}"),
            Ty::Data(n, args) => {
                write!(f, "{n}<")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ">")
            }
            Ty::Substrate(t) => write!(f, "Substrate{{{t}}}"),
            Ty::Vsa {
                model,
                dim,
                sparsity,
            } => match sparsity {
                Sparsity::Dense => write!(f, "VSA{{{model}, {dim}, Dense}}"),
                Sparsity::Sparse(k) => write!(f, "VSA{{{model}, {dim}, Sparse{{{k}}}}}"),
            },
            Ty::Seq(elem, n) => write!(f, "Seq{{{elem}, {n}}}"),
            Ty::Bytes => write!(f, "Bytes"),
            Ty::Float => write!(f, "Float"),
            Ty::Var(v) => write!(f, "{v}"),
            // RFC-0024 §3: render as `A -> B` (right-associative). Parenthesize a function-typed
            // LHS so `(A -> B) -> C` is unambiguous in diagnostics, not `A -> B -> C` (Copilot #397).
            Ty::Fn(a, r) if matches!(a.as_ref(), Ty::Fn(_, _)) => write!(f, "({a}) => {r}"),
            Ty::Fn(a, r) => write!(f, "{a} => {r}"),
        }
    }
}

/// An explicit check failure (never a silent pass or a guess — S5/G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckError {
    /// Which function (or item) the error is in, when known.
    pub site: String,
    /// What went wrong.
    pub message: String,
}

impl CheckError {
    pub(crate) fn new(site: &str, message: impl Into<String>) -> Self {
        CheckError {
            site: site.to_owned(),
            message: message.into(),
        }
    }

    /// Public, ergonomic constructor: a check failure at `site` with `message`. Mirrors the
    /// crate-internal `CheckError::new` (which stays private) so external callers — the
    /// toolchain crates that surface L1 diagnostics — can build a [`CheckError`] without reaching
    /// through the struct fields (Law of Demeter). Additive; no existing signature changes.
    #[must_use]
    pub fn at(site: impl Into<String>, message: impl Into<String>) -> Self {
        CheckError {
            site: site.into(),
            message: message.into(),
        }
    }
}

impl core::fmt::Display for CheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "check error in `{}`: {}", self.site, self.message)
    }
}

impl std::error::Error for CheckError {}

/// A resolution-pass refusal flows through the unified [`CheckError`] (the toolchain's one
/// surface→checked-program error), preserving the never-silent site + message (RFC-0012 §4.3/§4.4).
impl From<AmbientError> for CheckError {
    fn from(e: AmbientError) -> Self {
        let site = match &e {
            AmbientError::UnresolvedAmbient { site }
            | AmbientError::ParadigmShapeMismatch { site, .. }
            | AmbientError::BareDecimalNoEncoding { site, .. }
            | AmbientError::DepthExceeded { site, .. } => site.clone(),
            AmbientError::MultipleDefaults { .. } => "<nodule>".to_owned(),
        };
        CheckError {
            site,
            message: e.to_string(),
        }
    }
}

/// One constructor of a registered data type.
#[derive(Debug, Clone, PartialEq)]
pub struct CtorInfo {
    /// Constructor name.
    pub name: String,
    /// Field types.
    pub fields: Vec<Ty>,
}

/// A registered data type. **Stage-1 (RFC-0007 §11):** `params` are the type parameters (empty for a
/// monomorphic type); a constructor's [`CtorInfo::fields`] may reference them as [`Ty::Var`]. The
/// fields are stored *abstractly* (over the parameters) — [`subst_ty`] instantiates them at concrete
/// type arguments when a value is constructed or matched.
#[derive(Debug, Clone, PartialEq)]
pub struct DataInfo {
    /// Type name.
    pub name: String,
    /// Type parameters, in declaration order (empty for a monomorphic type). `List<A>` ⇒ `["A"]`.
    pub params: Vec<String>,
    /// Constructors, in declaration order (the index is the `#type#i` of RFC-0007 §4.2). Field types
    /// are abstract over `params` (may contain [`Ty::Var`]).
    pub ctors: Vec<CtorInfo>,
}

/// A registered **trait** (RFC-0019 §4.2; LR-2). Stage-1 is single-parameter, but the structure
/// carries `params: Vec<String>` uniformly (the trait's type-variable names, in scope as `Ty::Var`
/// while its method signatures are checked). The method `sigs` are stored as their surface
/// [`FnSig`]s; an `impl`'s methods are checked against the trait's sigs with `params ↦ trait_args`.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitInfo {
    /// Trait name.
    pub name: String,
    /// Type-parameter names (single-parameter in stage-1; abstract over the method sigs).
    pub params: Vec<String>,
    /// The required method signatures, in declaration order.
    pub sigs: Vec<FnSig>,
}

/// A registered **instance** `impl Trait<args> for T` (RFC-0019 §4.5). Keyed in [`Env::instances`]
/// by `(trait_name, type_head(for_ty))`; the full `for_ty`/`trait_args` are kept for `EXPLAIN` and
/// the staged dictionary lowering (M-673).
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceInfo {
    /// The trait this instances.
    pub trait_name: String,
    /// The (concrete) trait arguments.
    pub trait_args: Vec<Ty>,
    /// The (concrete) type the instance is for.
    pub for_ty: Ty,
    /// The provided method names (a non-silent record of what the dictionary supplies).
    pub methods: Vec<String>,
}

/// The **coherence key** of a type (RFC-0019 §4.5): the head a `(trait, type-head)` instance key is
/// computed from. Width/shape is intentionally **erased** — stage-1 keys per *head*, conservatively
/// rejecting two instances on the same head even at different widths (a documented, deferrable
/// refinement; the alternative, width-granular keying, needs the role/variance machinery deferred to
/// v2). A bare type **variable** is not a legal instance head in stage-1 (an `impl … for T` over an
/// abstract `T` would be a blanket instance — refused explicitly, never silently), so `Ty::Var`
/// yields `None`.
#[must_use]
pub fn type_head(ty: &Ty) -> Option<String> {
    Some(match ty {
        Ty::Binary(_) => "Binary".to_owned(),
        Ty::Ternary(_) => "Ternary".to_owned(),
        Ty::Dense(_, _) => "Dense".to_owned(),
        // RFC-0003 §3 (M-892): VSA keys per head like Dense (model/dim/sparsity erased — the same
        // documented stage-1 coherence simplification as the width-erased scalar paradigms).
        Ty::Vsa { .. } => "Vsa".to_owned(),
        Ty::Substrate(t) => format!("Substrate:{t}"),
        // RFC-0032 D3/D4: the sequence/byte-string reprs key per head (width/elem erased like the
        // scalar paradigms — a documented stage-1 coherence simplification).
        Ty::Seq(_, _) => "Seq".to_owned(),
        Ty::Bytes => "Bytes".to_owned(),
        // ADR-040 (M-897): the scalar float is a primitive repr head like `Bytes` (F64-only at
        // introduction, so the head carries no width — FLAG-1).
        Ty::Float => "Float".to_owned(),
        Ty::Data(n, _) => format!("Data:{n}"),
        // `Ty::Var` and `Ty::Fn` are not legal instance heads in stage-1 — a blanket instance
        // over an abstract variable or a function type is refused explicitly (RFC-0024 §3 / RFC-0019 §4.5).
        Ty::Var(_) | Ty::Fn(_, _) => return None,
    })
}

/// Substitute type arguments for the abstract parameters in a stage-1 type (RFC-0007 §11.2): replace
/// each [`Ty::Var`] by its binding in `s`, recursing into [`Ty::Data`] arguments. A `Var` with no
/// binding is left as-is (it is a parameter still in scope — e.g. while checking a generic body). This
/// is total and never inserts a `Swap` (S1): it only renames/instantiates type structure.
pub(crate) fn subst_ty(ty: &Ty, s: &BTreeMap<String, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => s.get(v).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Data(n, args) => Ty::Data(n.clone(), args.iter().map(|a| subst_ty(a, s)).collect()),
        // RFC-0024 §3: substitute into both sides of a function type (the param type and return type
        // may contain abstract type-variables — e.g. `f: A -> B` in a generic body).
        Ty::Fn(a, r) => Ty::Fn(Box::new(subst_ty(a, s)), Box::new(subst_ty(r, s))),
        // RFC-0032 D3: substitute into the element type (it may mention an abstract variable in a
        // generic context); the length is a structural constant, never substituted.
        Ty::Seq(elem, n) => Ty::Seq(Box::new(subst_ty(elem, s)), *n),
        // DN-42 / M-753 step-b: width-variable substitution — Binary/Ternary carrying a Width::Var
        // look up the var name in `s` via the carrier convention: a width-var binding is stored as
        // `var_name → Ty::Binary(Width::Lit(n))` regardless of paradigm (Binary or Ternary). On
        // extraction, we produce Binary or Ternary as appropriate. A var with no binding is left
        // as-is (still in scope — e.g. while checking the generic body). Never-silent (G2/VR-5).
        Ty::Binary(Width::Var(v)) => s
            .get(v)
            .map(|carrier| match carrier {
                Ty::Binary(Width::Lit(n)) => Ty::Binary(Width::Lit(*n)),
                _ => ty.clone(), // unrecognised carrier: leave as-is (defensive; should not occur)
            })
            .unwrap_or_else(|| ty.clone()),
        Ty::Ternary(Width::Var(v)) => s
            .get(v)
            .map(|carrier| match carrier {
                Ty::Binary(Width::Lit(n)) => Ty::Ternary(Width::Lit(*n)), // extract width from carrier
                _ => ty.clone(),
            })
            .unwrap_or_else(|| ty.clone()),
        // RFC-0003 §3 (M-892): VSA is fully concrete (model/dim/sparsity are structural
        // constants — no abstract slot to substitute into), like Dense.
        Ty::Binary(Width::Lit(_))
        | Ty::Ternary(Width::Lit(_))
        | Ty::Dense(_, _)
        | Ty::Vsa { .. }
        | Ty::Substrate(_)
        | Ty::Bytes
        | Ty::Float => ty.clone(),
    }
}

/// Build the parameter→argument substitution for a data type's constructor fields (RFC-0007 §11.2):
/// pairs each declared parameter name with the corresponding concrete type argument. A mismatched
/// length yields a partial map (the caller has already arity-checked, or is in a position where the
/// extra/missing entries simply do not substitute — never a panic).
pub(crate) fn param_subst(params: &[String], args: &[Ty]) -> BTreeMap<String, Ty> {
    params.iter().cloned().zip(args.iter().cloned()).collect()
}

/// Does `ty` mention any abstract type parameter ([`Ty::Var`])? Used to decide whether a (partially
/// substituted) declared type is concrete enough to drive a bidirectional check, or must let the
/// argument synthesize its own type so the parameter can be inferred (RFC-0007 §11.3).
pub(crate) fn has_var(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::Data(_, args) => args.iter().any(has_var),
        // RFC-0024 §3: a function type has a variable if either side does.
        Ty::Fn(a, r) => has_var(a) || has_var(r),
        // RFC-0032 D3: a sequence has a variable iff its element type does.
        Ty::Seq(elem, _) => has_var(elem),
        // DN-42 / M-753 step-b: a width-var is abstract — it makes the type have a variable.
        // A concrete width-lit is not abstract (Width::Lit is already resolved).
        Ty::Binary(Width::Var(_)) | Ty::Ternary(Width::Var(_)) => true,
        Ty::Binary(Width::Lit(_))
        | Ty::Ternary(Width::Lit(_))
        | Ty::Dense(_, _)
        // M-892: VSA is fully concrete (no abstract slot), like Dense.
        | Ty::Vsa { .. }
        | Ty::Substrate(_)
        | Ty::Bytes
        | Ty::Float => false,
    }
}

/// One-sided **unification** of a declared type (which may contain [`Ty::Var`] parameters) against a
/// concrete `actual` type, accumulating the parameter substitution `s` (RFC-0007 §11.3). Used to
/// **infer** a generic function's type arguments from its call-site argument types, and a constructor's
/// from its field arguments. A parameter is bound at most once; a second, conflicting binding is an
/// explicit mismatch (never a silent re-coercion — G2/VR-5). No `Swap` is ever inserted: a
/// representation-level disagreement (`Binary{8}` vs `Binary{16}`) is an explicit error, not a
/// conversion (S1).
pub(crate) fn unify(
    site: &str,
    decl: &Ty,
    actual: &Ty,
    s: &mut BTreeMap<String, Ty>,
) -> Result<(), CheckError> {
    match (decl, actual) {
        (Ty::Var(v), _) => match s.get(v) {
            Some(bound) if bound != actual => Err(CheckError::new(
                site,
                format!(
                    "type parameter `{v}` would have to be both {bound} and {actual} — \
                     an ambiguous instantiation, not a guess (RFC-0007 §11.3)"
                ),
            )),
            _ => {
                s.insert(v.clone(), actual.clone());
                Ok(())
            }
        },
        // A parameter appearing on the concrete side (nested generic call inside a generic body):
        // treat as equality (both must already agree).
        (_, Ty::Var(_)) if decl == actual => Ok(()),
        (Ty::Data(n1, a1), Ty::Data(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
            for (d, a) in a1.iter().zip(a2) {
                unify(site, d, a, s)?;
            }
            Ok(())
        }
        // RFC-0024 §3: structural unification of function types — param and return independently
        // (arrow is not covariant/contravariant at this stage; it is structural equality with variable
        // binding). Never a silent coercion — an `A -> B` against a `C -> D` unifies iff A~C and B~D
        // (VR-5, G2).
        (Ty::Fn(a1, r1), Ty::Fn(a2, r2)) => {
            unify(site, a1, a2, s)?;
            unify(site, r1, r2, s)
        }
        // RFC-0032 D3: two sequences unify iff their lengths are equal and their element types
        // unify (never a silent length coercion — G2/VR-5).
        (Ty::Seq(e1, n1), Ty::Seq(e2, n2)) if n1 == n2 => unify(site, e1, e2, s),
        // DN-42 / M-753 step-b: width-variable unification.
        // A width var on the declared side unifies against a concrete literal width of the SAME
        // paradigm. Cross-paradigm (Binary var ~ Ternary lit) falls through to the mismatch error
        // below — never a silent swap (G2/VR-5).
        // The binding is encoded as `var_name → Ty::Binary(Width::Lit(n))` in the shared subst map
        // (carrier convention — always Binary, so `subst_ty` can extract the `u32`
        // for either paradigm). A conflicting second binding is a never-silent error (VR-5/G2).
        (Ty::Binary(Width::Var(v)), Ty::Binary(Width::Lit(n)))
        | (Ty::Ternary(Width::Var(v)), Ty::Ternary(Width::Lit(n))) => {
            let carrier = Ty::Binary(Width::Lit(*n));
            match s.get(v) {
                Some(bound) if *bound != carrier => {
                    // Render the prior width HONESTLY (never-silent / G2): the carrier may be a
                    // concrete `Width::Lit` (another argument pinned it) OR an abstract `Width::Var`
                    // (the M-718 var-var arm bound it to the enclosing scope's width). Print the
                    // carrier's width via its `Display` — never a phantom `0` for the abstract case.
                    let prior = match bound {
                        Ty::Binary(w) => w.to_string(),
                        other => other.to_string(),
                    };
                    Err(CheckError::new(
                        site,
                        format!(
                            "width parameter `{v}` would have to be both {prior} and {n} — a width mismatch, not a coercion (DN-42 §4 / VR-5)"
                        ),
                    ))
                }
                _ => {
                    s.insert(v.clone(), carrier);
                    Ok(())
                }
            }
        }
        // DN-42 / M-753 / M-718: a width var unified against ANOTHER width var — a nested or recursive
        // width-generic call inside a width-generic body (e.g. `map_get`'s recursive `map_get(rest, k)`
        // over `Binary{N}`, or `le<N>` delegating to `cmp<N>`). BIND the declared var to the actual var
        // (carrier holds a `Width::Var`), mirroring the type-var pass-through above (`Ty::Var` arm): this
        // marks the callee's width param "determined by the enclosing scope" — it stays abstract here
        // and is resolved to a concrete `Width::Lit` when the enclosing fn is monomorphized
        // (`mono::infer_fn_targs` re-infers from the substituted scope). Before this binding the
        // determination check in `check_app_generic_fn` refused every recursive/nested width-generic
        // call (the width var was never inserted into `subst`). A prior binding to a *different*
        // concrete width is a never-silent mismatch — never a silent coercion (VR-5/G2/S1).
        (Ty::Binary(Width::Var(v1)), Ty::Binary(Width::Var(v2)))
        | (Ty::Ternary(Width::Var(v1)), Ty::Ternary(Width::Var(v2))) => {
            let carrier = Ty::Binary(Width::Var(v2.clone()));
            match s.get(v1) {
                // A prior binding that differs from this carrier (a concrete `Width::Lit` from another
                // argument, or a different width var) is a never-silent mismatch — exactly the type-var
                // conflict logic above: the param cannot be two widths at once, and an abstract width is
                // never silently coerced to a concrete one (DN-42 §4 / VR-5 / S1).
                Some(bound) if *bound != carrier => Err(CheckError::new(
                    site,
                    format!(
                        "width parameter `{v1}` would have to be both {bound} and the abstract width \
                         `{v2}` — an undetermined width, not a coercion (DN-42 §4 / VR-5)"
                    ),
                )),
                _ => {
                    s.insert(v1.clone(), carrier);
                    Ok(())
                }
            }
        }
        _ if decl == actual => Ok(()),
        _ => Err(CheckError::new(
            site,
            format!("cannot match {decl} against {actual} (RFC-0007 §11.3 — never a silent swap)"),
        )),
    }
}

/// The checked program environment: registry + function table. Built by [`check_nodule`]; the
/// evaluator and elaborator consume it (so nothing runs unchecked).
#[derive(Debug, Clone)]
pub struct Env {
    /// Data registry, keyed by type name.
    pub types: BTreeMap<String, DataInfo>,
    /// Function table, keyed by name.
    pub fns: BTreeMap<String, FnDecl>,
    /// Per-function totality classification (RFC-0007 §4.5), filled by the totality checker.
    pub totality: BTreeMap<String, crate::totality::Totality>,
    /// Trait registry (RFC-0019 §4.2; LR-2), keyed by trait name. A trait is a **registry entry**,
    /// never a kernel node (KC-3). Stored for `EXPLAIN` and for the staged dictionary lowering
    /// (M-673).
    pub traits: BTreeMap<String, TraitInfo>,
    /// Instance registry (RFC-0019 §4.5; the **coherence** key), keyed by `(trait_name, type_head)`.
    /// Head-granular keying is the stage-1 coherence discipline — at most one instance per
    /// `(trait, type-head)` (global uniqueness). Stored for instance resolution + the staged
    /// dictionary lowering (M-673).
    pub instances: BTreeMap<(String, String), InstanceInfo>,
    /// **Retained impl-method bodies** (M-673), keyed `(trait_name, type_head(for_ty))` — parallel to
    /// [`Self::instances`] (head-granular: at most one instance per key by coherence). Each value is
    /// the instance's **resolved** method `FnDecl`s (ambient literals + ctor/binder patterns
    /// normalized — the same canonical form a top-level fn carries). [`InstanceInfo::methods`] keeps
    /// only the method *names* (a non-silent record of the dictionary slots); to **monomorphize** a
    /// trait-method call to a direct call (`crate::mono`) the elaborator needs the method *bodies*, so
    /// they are retained here additively (rather than mutating [`InstanceInfo`], whose `PartialEq` /
    /// equality sites stay untouched). Empty when a nodule declares no `impl`s — a non-trait program is
    /// byte-identical to the pre-M-673 `Env`.
    pub impls: BTreeMap<(String, String), Vec<FnDecl>>,
    /// **User-defined generative-lowering rules** (DN-54 / M-812), keyed by rule name.
    /// Populated by `check_nodule_with` from every `lower Name[params] = <rhs>` declaration in the
    /// nodule; looked up by `derive Name for T` application during elaboration ([`crate::elab`]).
    /// Empty in any nodule that declares no `lower` rules. Acyclicity (DN-54 §4.2) and KC-3 (RHS
    /// must lower to existing L0 nodes, never grow the kernel) are both enforced at check time —
    /// never-silent (G2).
    pub lower_rules: BTreeMap<String, LowerDecl>,
    /// **Provenance of `derive`-generated sibling impls** (DN-54 §10.6 OQ-A / DN-81 §6.5 / RFC-0001
    /// §4.3; M-973). Keyed by the same `(trait_name, type_head)` coherence key as [`Self::instances`],
    /// each value records the `(rule_name, for_ty_render)` a Model-A sibling injection came from — so
    /// a derived impl is distinguishable from a hand-written one (`reveal`/EXPLAIN provenance), and
    /// Model A retains Model B's one intrinsic edge (DN-81 §5.3 Objection 2). Empty in any nodule with
    /// no `derive` sites. **Guarantee: `Declared`** — this is recorded metadata, not a checked
    /// property; ADR-003 content-addressing is the substantive identity, this map is the human-facing
    /// origin label (VR-5 — carried honestly, never upgraded).
    pub derived_provenance: BTreeMap<(String, String), (String, String)>,
    /// **Provenance of `via`-generated forwarding impls** (DN-53 §A.3.2/§A.3.3; M-966), the `via`
    /// analogue of [`Self::derived_provenance`]. Keyed by the same `(trait_name, type_head)`
    /// coherence key as [`Self::instances`]; each value records `(field_idx, object_name)` — which
    /// positional field the forwarding impl delegates through, and which `object` declared it — so
    /// `reveal`/EXPLAIN can label the generated impl `[generated: via <field_idx> : <TraitName>]`
    /// per DN-53 §A.3.3 (never a silent/opaque dispatch, G2). Resolution itself is deterministic by
    /// construction: `via_decls` is an ordered `Vec` (declaration order), never a `HashMap`, and two
    /// `via` clauses on the same object claiming the same trait are refused up front
    /// ([`expand_object_via_decls`]) rather than silently choosing one. Empty in any nodule with no
    /// `via` clauses. **Guarantee: `Declared`** — recorded metadata, not a checked property (VR-5 —
    /// carried honestly, never upgraded).
    pub via_provenance: BTreeMap<(String, String), (u32, String)>,
}

impl Env {
    /// Find the data type owning constructor `ctor`, with its index — `None` if no type has it.
    #[must_use]
    pub fn ctor(&self, ctor: &str) -> Option<(&DataInfo, usize)> {
        self.types
            .values()
            .find_map(|d| d.ctors.iter().position(|c| c.name == ctor).map(|i| (d, i)))
    }

    /// The registered data type named `name`, if any. Additive read-only accessor over the public
    /// [`types`](Env::types) map — a Law-of-Demeter-friendly alternative to `env.types.get(name)`.
    #[must_use]
    pub fn type_info(&self, name: &str) -> Option<&DataInfo> {
        self.types.get(name)
    }

    /// The function declaration named `name`, if any. Additive read-only accessor over the public
    /// [`fns`](Env::fns) map.
    #[must_use]
    pub fn fn_decl(&self, name: &str) -> Option<&FnDecl> {
        self.fns.get(name)
    }

    /// The totality verdict for function `name`, if it has been classified. Additive read-only
    /// accessor over the public [`totality`](Env::totality) map. `Totality` is `Copy`, so this
    /// returns it by value (never silently fabricating a verdict for an unclassified name — `None`).
    #[must_use]
    pub fn fn_totality(&self, name: &str) -> Option<crate::totality::Totality> {
        self.totality.get(name).copied()
    }

    /// The registered trait named `name`, if any (RFC-0019 §4.2). Additive read-only accessor over
    /// the public [`traits`](Env::traits) map.
    #[must_use]
    pub fn trait_info(&self, name: &str) -> Option<&TraitInfo> {
        self.traits.get(name)
    }

    /// The registered instance for `(trait_name, head)`, if any (RFC-0019 §4.5). `head` is a
    /// [`type_head`]. Additive read-only accessor over the public [`instances`](Env::instances) map.
    #[must_use]
    pub fn instance(&self, trait_name: &str, head: &str) -> Option<&InstanceInfo> {
        self.instances
            .get(&(trait_name.to_owned(), head.to_owned()))
    }
}

/// The builtin prelude: `type Bool = False | True` (`if` scrutinizes it; RFC-0007 keeps `if` as
/// elaboration-level sugar over `Match` on this registry entry).
pub(crate) fn prelude() -> DataInfo {
    DataInfo {
        name: "Bool".to_owned(),
        params: vec![],
        ctors: vec![
            CtorInfo {
                name: "False".to_owned(),
                fields: vec![],
            },
            CtorInfo {
                name: "True".to_owned(),
                fields: vec![],
            },
        ],
    }
}

/// Resolve a surface [`TypeRef`] to a checked [`Ty`], with the type parameters `tyvars` in scope
/// (RFC-0007 §11.2): a `Named(name, [])` whose `name` is a type parameter resolves to [`Ty::Var`];
/// any other `Named` is a data type whose **arity is checked** against its declaration (`List<A>`
/// applied to the wrong number of arguments is an explicit error, never a guess). VSA types resolve
/// concretely as of M-892 (their surface model ident canonicalized to the kernel model id). The
/// guarantee index is *allowed* and returned alongside (checked dynamically at
/// stage 0 — RFC-0007 §4.3). `tyvars` is `&[]` in a monomorphic context.
pub(crate) fn resolve_ty(
    site: &str,
    types: &BTreeMap<String, DataInfo>,
    tyvars: &[String],
    t: &TypeRef,
) -> Result<(Ty, Option<Strength>), CheckError> {
    let base = match &t.base {
        BaseType::Binary(WidthRef::Lit(n)) => Ty::Binary(Width::Lit(*n)),
        BaseType::Binary(WidthRef::Name(v)) => Ty::Binary(Width::Var(v.clone())),
        BaseType::Ternary(WidthRef::Lit(m)) => Ty::Ternary(Width::Lit(*m)),
        BaseType::Ternary(WidthRef::Name(v)) => Ty::Ternary(Width::Var(v.clone())),
        BaseType::Dense(d, s) => Ty::Dense(*d, *s),
        BaseType::Substrate(tag) => Ty::Substrate(tag.clone()),
        // RFC-0032 D3/D4: the sequence + byte-string repr types. `Seq{T, N}` resolves its element
        // type recursively under the same `tyvars` scope; `Bytes` is nullary.
        BaseType::Seq { elem, len } => {
            let (elem_ty, _) = resolve_ty(site, types, tyvars, elem)?;
            Ty::Seq(Box::new(elem_ty), *len)
        }
        BaseType::Bytes => Ty::Bytes,
        // ADR-040 (M-897): the nullary scalar-float repr type (binary64 only — FLAG-1).
        BaseType::Float => Ty::Float,
        // RFC-0003 §3 / ADR-008 (M-892): the VSA type resolves concretely — the type-level lift
        // of the former blanket deferral, needed so the `vsa_*` prims are typable. Any model id
        // is admissible as a *mention* (core already type-checks hypervector mentions without
        // the algebra); the prims gate their dispatch set separately (`try_check_vsa_prim`).
        // The surface ident is canonicalized to the kernel model id (`MAP_I` → `MAP-I`, …).
        BaseType::Vsa {
            model,
            dim,
            sparsity,
        } => Ty::Vsa {
            model: vsa_kernel_model_id(model),
            dim: *dim,
            sparsity: sparsity.clone(),
        },
        BaseType::Named(name, args) => {
            // A bare name that is a type parameter in scope is an abstract type variable (§11.2).
            if args.is_empty() && tyvars.iter().any(|v| v == name) {
                Ty::Var(name.clone())
            } else {
                let Some(decl) = types.get(name) else {
                    return Err(CheckError::new(site, format!("unknown type `{name}`")));
                };
                // Arity is checked — never a guess (§11.3). A type parameter cannot be applied.
                if args.len() != decl.params.len() {
                    return Err(CheckError::new(
                        site,
                        format!(
                            "`{name}` takes {} type argument(s), got {} (RFC-0007 §11.3)",
                            decl.params.len(),
                            args.len()
                        ),
                    ));
                }
                let mut resolved = Vec::with_capacity(args.len());
                for a in args {
                    resolved.push(resolve_ty(site, types, tyvars, a)?.0);
                }
                Ty::Data(name.clone(), resolved)
            }
        }
        BaseType::Ambient(_) => {
            // The resolution pass ([`crate::ambient`]) fills every paradigm-less repr before the
            // checker runs; a residual one is an internal invariant break, refused explicitly (never
            // a silent guess) — defense in depth (RFC-0012 §4.3).
            return Err(CheckError::new(
                site,
                "internal: an unresolved paradigm-less repr `{…}` reached the checker — the \
                 ambient resolution pass should have filled it (RFC-0012 §4.3)",
            ));
        }
        // RFC-0024 §3 (M-686): function types are now checked. Resolve both sides recursively
        // under the same `tyvars` scope — the param and return types may themselves be abstract.
        // This node is single-argument by construction; a curried multi-arg value type is nested
        // `Fn`s (`A => B => C`, M-822) and a tuple-domain arrow `(A, B) => C` is this same node
        // with a `BaseType::Tuple` param (M-826) — both compose through this one arm, no separate
        // refusal (DN-73 D1/D2, ratified). Guarantee: Declared (a type-level contract — VR-5).
        BaseType::Fn(param, ret) => {
            let (param_ty, _) = resolve_ty(site, types, tyvars, param)?;
            let (ret_ty, _) = resolve_ty(site, types, tyvars, ret)?;
            Ty::Fn(Box::new(param_ty), Box::new(ret_ty))
        }
        // M-826: a tuple type `(T, U, …)` resolves to `Ty::Data("Tuple$N", [T, U, …])` where
        // `Tuple$N` is the synthetic single-constructor product type registered by
        // `register_tuple_types`. Arity ≥ 2 is enforced by the parser; the registration must have
        // happened before checking (KC-3 — no new kernel node, desugars to existing Construct/Match).
        // Guarantee: `Empirical` (construct→destructure round-trip tested — M-826).
        BaseType::Tuple(elems) => {
            let n = elems.len();
            let tname = tuple_type_name(n);
            if !types.contains_key(&tname) {
                return Err(CheckError::new(
                    site,
                    format!(
                        "internal: synthetic tuple type `{tname}` not registered — \
                         `register_tuple_types` must run before the checker (M-826; never silent)"
                    ),
                ));
            }
            let mut resolved = Vec::with_capacity(n);
            for e in elems {
                resolved.push(resolve_ty(site, types, tyvars, e)?.0);
            }
            Ty::Data(tname, resolved)
        }
    };
    Ok((base, t.guarantee))
}

/// The synthetic name for a tuple type of arity `n` (M-826, KC-3). Injective over n; `$` is not
/// a surface-identifier character, so this name cannot collide with user-defined types.
/// `Tuple$2` = the 2-tuple (pair), `Tuple$3` = the 3-tuple, etc.
#[must_use]
pub(crate) fn tuple_type_name(n: usize) -> String {
    format!("Tuple${n}")
}

/// The synthetic constructor name for a tuple of arity `n` (M-826). Same injective scheme.
#[must_use]
pub(crate) fn tuple_ctor_name(n: usize) -> String {
    format!("MkTuple${n}")
}

/// Synthesize the [`DataInfo`] for a tuple of arity `n` (n ≥ 2; M-826, KC-3). The type has `n`
/// type parameters `T0 … T{n-1}` and a single constructor `MkTuple$N(T0, T1, …)`.
/// This is a pure data declaration (no kernel node — KC-3); the checker registers it on demand.
#[must_use]
pub(crate) fn synthetic_tuple_data(n: usize) -> DataInfo {
    let params: Vec<String> = (0..n).map(|i| format!("T{i}")).collect();
    let fields: Vec<Ty> = params.iter().map(|p| Ty::Var(p.clone())).collect();
    DataInfo {
        name: tuple_type_name(n),
        params: params.clone(),
        ctors: vec![CtorInfo {
            name: tuple_ctor_name(n),
            fields,
        }],
    }
}

/// Walk a nodule and collect all tuple arities referenced (in `BaseType::Tuple` and
/// `Expr::TupleLit`) so `register_types` can pre-register the synthetic `Tuple$N` data types
/// before checking. Uses a recursive AST scan (pre-order, no depth guard needed — it's a scan,
/// not evaluation).
pub(crate) fn collect_tuple_arities(nodule: &Nodule) -> std::collections::BTreeSet<usize> {
    let mut arities = std::collections::BTreeSet::new();
    for item in &nodule.items {
        collect_tuple_arities_item(item, &mut arities);
    }
    arities
}

fn collect_tuple_arities_item(
    item: &crate::ast::Item,
    out: &mut std::collections::BTreeSet<usize>,
) {
    use crate::ast::Item;
    match item {
        Item::Fn(fd) => {
            collect_tuple_arities_sig(&fd.sig, out);
            collect_tuple_arities_expr(&fd.body, out);
        }
        Item::Type(td) => {
            for ctor in &td.ctors {
                for f in &ctor.fields {
                    collect_tuple_arities_typeref(f, out);
                }
            }
        }
        Item::Trait(tr) => {
            for sig in &tr.sigs {
                collect_tuple_arities_sig(sig, out);
            }
        }
        Item::Impl(id) => {
            for t in &id.trait_args {
                collect_tuple_arities_typeref(t, out);
            }
            collect_tuple_arities_typeref(&id.for_ty, out);
            for m in &id.methods {
                collect_tuple_arities_sig(&m.sig, out);
                collect_tuple_arities_expr(&m.body, out);
            }
        }
        Item::Object(od) => {
            for f in &od.ctor.fields {
                collect_tuple_arities_typeref(f, out);
            }
            for impl_d in &od.impls {
                for t in &impl_d.trait_args {
                    collect_tuple_arities_typeref(t, out);
                }
                collect_tuple_arities_typeref(&impl_d.for_ty, out);
                for m in &impl_d.methods {
                    collect_tuple_arities_sig(&m.sig, out);
                    collect_tuple_arities_expr(&m.body, out);
                }
            }
            for f in &od.fns {
                collect_tuple_arities_sig(&f.sig, out);
                collect_tuple_arities_expr(&f.body, out);
            }
        }
        Item::Lower(ld) => match &ld.rhs {
            crate::ast::LowerRhs::Expr(e) => collect_tuple_arities_expr(e, out),
            // Item-shaped RHS (DN-54 §10 Model A, M-973): collect tuple arities from the impl
            // template's method signatures + bodies, exactly as the `Item::Impl` arm above.
            crate::ast::LowerRhs::Impl(id) => {
                for t in &id.trait_args {
                    collect_tuple_arities_typeref(t, out);
                }
                collect_tuple_arities_typeref(&id.for_ty, out);
                for m in &id.methods {
                    collect_tuple_arities_sig(&m.sig, out);
                    collect_tuple_arities_expr(&m.body, out);
                }
            }
        },
        Item::InherentImpl(iid) => {
            collect_tuple_arities_typeref(&iid.for_ty, out);
            for m in &iid.methods {
                collect_tuple_arities_sig(&m.sig, out);
                collect_tuple_arities_expr(&m.body, out);
            }
        }
        Item::Use(_) | Item::Default(_) | Item::Derive(_) => {}
    }
}

fn collect_tuple_arities_typeref(tr: &TypeRef, out: &mut std::collections::BTreeSet<usize>) {
    use crate::ast::BaseType;
    match &tr.base {
        BaseType::Tuple(elems) => {
            out.insert(elems.len());
            for e in elems {
                collect_tuple_arities_typeref(e, out);
            }
        }
        BaseType::Seq { elem, .. } => collect_tuple_arities_typeref(elem, out),
        BaseType::Named(_, args) => {
            for a in args {
                collect_tuple_arities_typeref(a, out);
            }
        }
        BaseType::Fn(a, b) => {
            collect_tuple_arities_typeref(a, out);
            collect_tuple_arities_typeref(b, out);
        }
        _ => {}
    }
}

fn collect_tuple_arities_sig(sig: &crate::ast::FnSig, out: &mut std::collections::BTreeSet<usize>) {
    for p in &sig.value_params {
        collect_tuple_arities_typeref(&p.ty, out);
    }
    collect_tuple_arities_typeref(&sig.ret, out);
}

/// Scan a pattern for tuple arities (M-826) — a `Pattern::Tuple` of arity N requires the synthetic
/// `Tuple$N` data type to be registered before checking, even when no `TupleLit` of that arity is
/// constructed in the nodule. Recurses through constructor sub-patterns and or-pattern alternatives.
fn collect_tuple_arities_pattern(p: &Pattern, out: &mut std::collections::BTreeSet<usize>) {
    match p {
        Pattern::Tuple(subs) => {
            out.insert(subs.len());
            for sub in subs {
                collect_tuple_arities_pattern(sub, out);
            }
        }
        Pattern::Ctor(_, subs) => {
            for sub in subs {
                collect_tuple_arities_pattern(sub, out);
            }
        }
        Pattern::Or(alts) => {
            for alt in alts {
                collect_tuple_arities_pattern(alt, out);
            }
        }
        Pattern::Wildcard | Pattern::Lit(_) | Pattern::Ident(_) => {}
    }
}

fn collect_tuple_arities_expr(e: &crate::ast::Expr, out: &mut std::collections::BTreeSet<usize>) {
    use crate::ast::Expr;
    match e {
        Expr::TupleLit(elems) => {
            out.insert(elems.len());
            for el in elems {
                collect_tuple_arities_expr(el, out);
            }
        }
        Expr::Let {
            bound, body, ty, ..
        } => {
            if let Some(t) = ty {
                collect_tuple_arities_typeref(t, out);
            }
            collect_tuple_arities_expr(bound, out);
            collect_tuple_arities_expr(body, out);
        }
        Expr::If { cond, conseq, alt } => {
            collect_tuple_arities_expr(cond, out);
            collect_tuple_arities_expr(conseq, out);
            collect_tuple_arities_expr(alt, out);
        }
        Expr::Match { scrutinee, arms } => {
            collect_tuple_arities_expr(scrutinee, out);
            for arm in arms {
                // M-826 (self-review M1): a tuple pattern that is only ever *destructured* — never
                // constructed as a `TupleLit` nor named in a type annotation elsewhere in the nodule —
                // still needs its synthetic `Tuple$N` arity registered, or the checker hits an internal
                // "tuple type not registered" panic instead of a user diagnostic. Scan the arm pattern,
                // not just its body.
                collect_tuple_arities_pattern(&arm.pattern, out);
                collect_tuple_arities_expr(&arm.body, out);
            }
        }
        Expr::For { xs, init, body, .. } => {
            collect_tuple_arities_expr(xs, out);
            collect_tuple_arities_expr(init, out);
            collect_tuple_arities_expr(body, out);
        }
        Expr::Swap { value, .. } => collect_tuple_arities_expr(value, out),
        Expr::Wild(b) | Expr::Spore(b) | Expr::Consume(b) => {
            collect_tuple_arities_expr(b, out);
        }
        Expr::Colony(hyphae) => {
            for h in hyphae {
                collect_tuple_arities_expr(&h.body, out);
            }
        }
        Expr::Lambda { params, body } => {
            for p in params {
                collect_tuple_arities_typeref(&p.ty, out);
            }
            collect_tuple_arities_expr(body, out);
        }
        Expr::App { head, args } => {
            collect_tuple_arities_expr(head, out);
            for a in args {
                collect_tuple_arities_expr(a, out);
            }
        }
        Expr::Fuse { left, right } => {
            collect_tuple_arities_expr(left, out);
            collect_tuple_arities_expr(right, out);
        }
        Expr::Reclaim { policy, body } => {
            collect_tuple_arities_expr(policy, out);
            collect_tuple_arities_expr(body, out);
        }
        Expr::Ascribe(inner, t) => {
            collect_tuple_arities_expr(inner, out);
            collect_tuple_arities_typeref(t, out);
        }
        Expr::WithParadigm { body, .. } => collect_tuple_arities_expr(body, out),
        Expr::Path(_) | Expr::Lit(_) => {}
    }
}

/// The checked environments of a whole **phylum** (M-662): one [`Env`] per nodule, paired with the
/// nodule's path. The product of [`check_phylum`]. For a phylum-of-one this holds the single nodule's
/// `Env` (so [`check_nodule`] just unwraps it) — additive, backward-compatible.
#[derive(Debug, Clone)]
pub struct PhylumEnv {
    /// One `(nodule path, checked Env)` per nodule, in source order.
    pub nodules: Vec<(Path, Env)>,
}

impl PhylumEnv {
    /// The single nodule's [`Env`] when this is a phylum-of-one, else `None`. The bridge
    /// [`check_nodule`] uses to keep its single-`Env` return type (M-662).
    #[must_use]
    pub fn single(&self) -> Option<&Env> {
        match self.nodules.as_slice() {
            [(_, env)] => Some(env),
            _ => None,
        }
    }

    /// The checked [`Env`] of the nodule whose path equals `path`, if present.
    #[must_use]
    pub fn nodule(&self, path: &Path) -> Option<&Env> {
        self.nodules.iter().find(|(p, _)| p == path).map(|(_, e)| e)
    }
}

/// The **phylum-wide export table** (M-662): the `pub` items of every nodule, keyed by **qualified
/// name** (`nodule.path` + `.` + item name, e.g. `"std.collections.List"`). This is the *import
/// registry* — the **only-`pub`** view a `use` resolves against (RFC-0006 §4.3). It is kept strictly
/// separate from the pub-blind coherence view (below): conflating the two would let a `use` import a
/// private name or let the orphan rule miss a private declaration (a bug — the two views answer
/// different questions).
#[derive(Debug, Default)]
struct Exports {
    /// Exported data types, by qualified name.
    types: BTreeMap<String, DataInfo>,
    /// Exported functions, by qualified name.
    fns: BTreeMap<String, FnDecl>,
    /// Exported traits, by qualified name.
    traits: BTreeMap<String, TraitInfo>,
    /// **All** declared simple names per nodule-prefix, with their `pub`-ness — used to distinguish
    /// "no such name" from "exists but private" in a `use` refusal (G2 — an honest, helpful
    /// diagnostic). Keyed by qualified name → `is_pub`.
    declared: BTreeMap<String, bool>,
}

/// The resolved imports available to **one** nodule while its bodies are checked (M-662): the
/// imported `pub` declarations, merged by **simple name** at the documented precedence (own decls
/// shadow explicit `use` shadow glob), plus the set of names that a **glob-vs-glob collision** left
/// **ambiguous** (importable only by an explicit `use`). The ambiguous set is consulted at every
/// unresolved-name site so a *reference* to an ambiguous glob name is a never-silent `CheckError`,
/// never a silent winner (G2).
#[derive(Debug, Default, Clone)]
struct NoduleImports {
    /// Imported data types, by simple name.
    types: BTreeMap<String, DataInfo>,
    /// Imported functions, by simple name.
    fns: BTreeMap<String, FnDecl>,
    /// Imported traits, by simple name.
    traits: BTreeMap<String, TraitInfo>,
    /// Names brought in by **two or more** globs (and not resolved by an explicit `use` or a local
    /// decl): a *reference* to one of these is the never-silent glob-vs-glob ambiguity error.
    ambiguous: BTreeSet<String>,
}

impl NoduleImports {
    /// The never-silent glob-vs-glob ambiguity refusal for `name`, if it is ambiguous here (G2).
    /// Returns `None` when the name is unambiguous (so the caller falls through to its own
    /// unknown-name diagnostic).
    fn ambiguity_error(&self, site: &str, name: &str) -> Option<CheckError> {
        self.ambiguous.contains(name).then(|| {
            CheckError::new(
                site,
                format!(
                    "`{name}` is ambiguous — imported by more than one glob `use … .*` in this \
                     nodule; import it explicitly (`use <path>.{name}`) to disambiguate (M-662; \
                     never a silent winner — G2)"
                ),
            )
        })
    }
}

/// Check a whole nodule: build the registry (prelude + declarations), then type every function
/// body against its signature, classify totality. No maturation gate is applied (the scope is
/// treated as non-matured). Returns the checked [`Env`].
///
/// As of M-344 (RFC-0012) the input is first run through the **ambient resolution pass**
/// ([`crate::ambient::resolve`]) — paradigm-less reprs are filled, `with paradigm` blocks stripped,
/// bare decimals tagged — so the checker only ever sees fully-explicit (longhand) forms. A program
/// using no ambient is unchanged (resolution is identity).
///
/// As of M-662 a bare nodule is checked as a **phylum-of-one** ([`check_phylum`]); this is a thin
/// wrapper that unwraps the single [`Env`]. Behavior on every single-nodule program is unchanged
/// (no imports, the orphan rule's locality set is exactly this one nodule).
pub fn check_nodule(nodule: &Nodule) -> Result<Env, CheckError> {
    check_and_resolve(nodule).map(|(env, _)| env)
}

/// Check a whole **phylum** (M-662; RFC-0006 §4.3): build the phylum-wide `pub` **export table** and
/// the pub-blind **coherence view**, then check each nodule's bodies with its resolved `use` imports
/// available and the **phylum-wide orphan rule** enforced. Returns one [`Env`] per nodule.
///
/// Two strictly-separate phylum-wide views (conflating them is a bug — they answer different
/// questions): the **import registry** ([`Exports`]) is `pub`-only (what a `use` may bind); the
/// **coherence view** is pub-blind (every nodule's trait/type declarations are visible to the orphan
/// rule regardless of `pub`). Cross-nodule **execution** is staged — the per-nodule [`Env`]s are real
/// and complete for type-checking; running a `use`d fn across nodules is a follow-up (eval keeps its
/// per-nodule reach; a cross-nodule call lowers to a never-silent `Unsupported`/`Residual`).
///
/// # Errors
/// Any never-silent refusal: an unknown/private/ambiguous import, a duplicate import, a coherence or
/// phylum-wide orphan violation, or any per-nodule type error (all surfaced as [`CheckError`]).
pub fn check_phylum(phylum: &Phylum) -> Result<PhylumEnv, CheckError> {
    check_phylum_matured(phylum, false)
}

/// Like [`check_phylum`] but with the explicit `matured_scope` gate applied to **every** nodule
/// (RFC-0017 §4.2; M-662). When `matured_scope` is `false` this is identical to [`check_phylum`].
///
/// # Errors
/// See [`check_phylum`]; additionally a non-total non-`thaw` definition in any nodule under a matured
/// scope is an explicit [`CheckError`].
pub fn check_phylum_matured(phylum: &Phylum, matured_scope: bool) -> Result<PhylumEnv, CheckError> {
    mycelium_stack::with_deep_stack(|| check_phylum_inner(phylum, matured_scope))
}

fn check_phylum_inner(phylum: &Phylum, matured_scope: bool) -> Result<PhylumEnv, CheckError> {
    // 1. Ambient-resolve every nodule once (RFC-0012): the checker only ever sees longhand forms.
    let resolved: Vec<Nodule> = phylum
        .nodules
        .iter()
        .map(crate::ambient::resolve)
        .collect::<Result<_, _>>()?;

    // Phase 0: structurally expand `object` declarations (DN-53, M-811). Replaces each
    // `Item::Object` with its `TypeDecl + ImplDecl(s) + FnDecl(s)` lowering so the existing
    // type/trait/fn registration passes work without modification (KC-3 — zero kernel growth).
    // `via` delegation impls are NOT generated here — they require the trait registry (built in
    // pass 2, below). The original `ObjectDecl`s with `via_decls` are preserved in a parallel
    // `Vec<Vec<ObjectDecl>>` and passed into `check_nodule_with` for Phase 0b processing.
    let mut via_objects_per_nodule: Vec<Vec<ObjectDecl>> = Vec::with_capacity(resolved.len());
    let mut resolved_expanded: Vec<Nodule> = Vec::with_capacity(resolved.len());
    for n in resolved {
        if !n
            .items
            .iter()
            .any(|i| matches!(i, Item::Object(_) | Item::InherentImpl(_)))
        {
            via_objects_per_nodule.push(vec![]);
            resolved_expanded.push(n);
            continue;
        }
        let mut items = Vec::with_capacity(n.items.len());
        let mut via_objs: Vec<ObjectDecl> = Vec::new();
        for item in n.items {
            match item {
                Item::Object(od) => {
                    if !od.via_decls.is_empty() {
                        via_objs.push(od.clone());
                    }
                    items.extend(desugar_object_structural(&od));
                }
                // M-664: an inherent `impl T { fn … }` block lowers to its methods as top-level
                // `Item::Fn`s, lifted verbatim — methods are ordinary explicitly-typed free
                // functions (the `object` inherent-`fn` model; KC-3 — zero kernel growth). The
                // `for_ty` is organizational metadata in v0 (no qualified `T::m` call syntax yet —
                // `check_path` refuses multi-segment paths), so the association is advisory; a
                // name collision with another top-level fn is caught by the duplicate-fn check (G2).
                //
                // KNOWN GAP (intentional in v0, never-silent by documentation): `for_ty` is **not**
                // validated to resolve to a known type here, so `impl NoSuchType { … }` is accepted
                // (the unknown head is dropped, the methods still check). This is a deliberate scope
                // boundary, not an oversight: at this Phase-0 desugar the type registries do **not**
                // yet exist (`register_nodule_decls`, pass 2, runs *after* this loop — see below), so
                // resolving `for_ty` would require either a separate type-name pre-pass or deferring
                // the entire impl desugar past registration — both larger than the v0 payoff while
                // `for_ty` carries no semantic weight (it is unreachable: no `T::m` call form binds to
                // it). The accepted consequence: an unknown `for_ty` is a no-op, not a refusal. When a
                // qualified-call surface lands (`T::m`), `for_ty` becomes load-bearing and MUST be
                // resolved against the (then-available) registry — at that point this becomes a real
                // never-silent gap to close (G2). Behaviour pinned by
                // `inherent_impl_on_an_unknown_for_ty_is_accepted_in_v0` (tests/check.rs).
                Item::InherentImpl(id) => {
                    for m in id.methods {
                        items.push(Item::Fn(m));
                    }
                }
                other => items.push(other),
            }
        }
        via_objects_per_nodule.push(via_objs);
        resolved_expanded.push(Nodule {
            path: n.path,
            std_sys: n.std_sys,
            items,
        });
    }
    let resolved = resolved_expanded;

    // 2. Register each nodule's declarations into its own registries, and from those build the two
    //    phylum-wide views: the `pub`-only export table (import registry) and the pub-blind coherence
    //    view (all traits/types, for the orphan rule). Registration here is *declaration-level* only
    //    (no body checking) — bodies are checked in pass 3 with imports available.
    let mut per_nodule_regs: Vec<NoduleRegs> = Vec::with_capacity(resolved.len());
    let mut exports = Exports::default();
    let mut coherence = CoherenceView::default();
    for nodule in &resolved {
        let regs = register_nodule_decls(nodule)?;
        let qual = |name: &str| qualify(&nodule.path, name);
        // Export the `pub` items (import registry — pub-only); record every declared name's pub-ness
        // (for the never-silent "no such name" vs "exists but private" distinction).
        for (name, info) in &regs.types {
            exports
                .declared
                .insert(qual(name), info_is_pub_type(nodule, name));
            if info_is_pub_type(nodule, name) {
                exports.types.insert(qual(name), info.clone());
            }
        }
        for (name, fd) in &regs.fns {
            exports.declared.insert(qual(name), fd.vis.is_pub());
            if fd.vis.is_pub() {
                exports.fns.insert(qual(name), fd.clone());
            }
        }
        for (name, info) in &regs.traits {
            exports
                .declared
                .insert(qual(name), info_is_pub_trait(nodule, name));
            if info_is_pub_trait(nodule, name) {
                exports.traits.insert(qual(name), info.clone());
            }
        }
        // Coherence view (pub-blind): every nodule's trait + data-type *names* are visible to the
        // phylum-wide orphan rule regardless of `pub` (coherence is enforcement authority, not the
        // `pub` namespace — RFC-0019 §4.5; M-662).
        for name in regs.traits.keys() {
            coherence.traits.insert(name.clone());
        }
        for name in regs.types.keys() {
            // The prelude `Bool` is registered into every nodule; skip it as a phylum "local" so it
            // does not falsely satisfy the orphan rule for an unrelated impl (it is a primitive-ish
            // builtin, handled by the primitive-repr arm anyway).
            if name != "Bool" {
                coherence.types.insert(name.clone());
            }
        }
        per_nodule_regs.push(regs);
    }

    // 3. Check each nodule's bodies with (a) its resolved `use` imports merged into its registries and
    //    (b) the phylum-wide pub-blind orphan rule. Each yields a checked `Env`.
    let mut out = Vec::with_capacity(resolved.len());
    for (i, (nodule, regs)) in resolved.iter().zip(per_nodule_regs).enumerate() {
        let imports = resolve_imports(nodule, &exports)?;
        // Pass this nodule's via_objects (objects with `via` decls) for Phase 0b expansion of
        // delegation impls (DN-53 M-811). The slice is empty for nodules with no `via` clauses.
        let via_objects = &via_objects_per_nodule[i];
        let env = check_nodule_with(
            nodule,
            regs,
            &imports,
            &coherence,
            matured_scope,
            via_objects,
        )?;
        out.push((nodule.path.clone(), env));
    }
    Ok(PhylumEnv { nodules: out })
}

/// `nodule.path` + `.` + `name` — a top-level item's **qualified name** (the import-registry key;
/// M-662). `nodule a.b` declaring `List` ⇒ `"a.b.List"`.
fn qualify(path: &Path, name: &str) -> String {
    if path.0.is_empty() {
        name.to_owned()
    } else {
        format!("{}.{name}", path.0.join("."))
    }
}

/// Is the named **type** declared `pub` in this (resolved) nodule? (The registry [`DataInfo`] does
/// not carry `Vis` — it is a checked, post-resolution structure — so the surface `Vis` is read back
/// from the nodule's items. The prelude `Bool` is never a surface item ⇒ not `pub`.)
fn info_is_pub_type(nodule: &Nodule, name: &str) -> bool {
    nodule
        .items
        .iter()
        .any(|i| matches!(i, Item::Type(td) if td.name == name && td.vis.is_pub()))
}

/// Is the named **trait** declared `pub` in this (resolved) nodule? (See [`info_is_pub_type`].)
fn info_is_pub_trait(nodule: &Nodule, name: &str) -> bool {
    nodule
        .items
        .iter()
        .any(|i| matches!(i, Item::Trait(td) if td.name == name && td.vis.is_pub()))
}

/// The pub-blind **coherence view** for the phylum-wide orphan rule (M-662; RFC-0019 §4.5): the set
/// of **all** trait names and **all** data-type names declared by **any** nodule of the phylum,
/// regardless of `pub`. Distinct from the [`Exports`] (pub-only) import registry — an `impl` is legal
/// iff its trait OR its `for`-type head is declared *somewhere* in the phylum, and that visibility is
/// pub-blind (coherence is enforcement authority, not the `pub` namespace).
#[derive(Debug, Default)]
pub(crate) struct CoherenceView {
    /// Every trait name declared anywhere in the phylum (pub-blind).
    pub(crate) traits: BTreeSet<String>,
    /// Every data-type name declared anywhere in the phylum (pub-blind), excluding the prelude.
    pub(crate) types: BTreeSet<String>,
}

/// The **declaration-level** registries of one nodule (types/fns/traits), built before any body is
/// checked (M-662). The phylum builds its two cross-nodule views from these, then re-uses them as the
/// per-nodule base when checking that nodule's bodies (so registration runs once per nodule — DRY).
struct NoduleRegs {
    types: BTreeMap<String, DataInfo>,
    fns: BTreeMap<String, FnDecl>,
    traits: BTreeMap<String, TraitInfo>,
}

/// Register one (resolved) nodule's **declarations** — data types (Pass 1), traits (Pass 1b), and
/// function signatures (Pass 2) — into its registries, with the same duplicate/arity refusals as the
/// single-nodule checker (M-662 factors these out of `check_resolved_matured` so the phylum can build
/// its cross-nodule views before checking any body). Bodies and instances are **not** handled here
/// (instances need the phylum-wide orphan view; bodies need imports). The prelude `Bool` is included
/// so intra-nodule resolution is unchanged.
fn register_nodule_decls(nodule: &Nodule) -> Result<NoduleRegs, CheckError> {
    let mut types = BTreeMap::new();
    let p = prelude();
    types.insert(p.name.clone(), p);
    register_types(&mut types, nodule)?;
    let mut traits = register_traits(&types, nodule)?;
    // M-965 (DN-58 §A F-A1): seed the built-in `Fuse` trait — the trait analogue of the `Bool`
    // prelude type above — but only when this nodule actually declares an `impl Fuse[...] for
    // ...` (never unconditionally). A nodule that tries to redeclare it gets an explicit refusal
    // (never a silent shadow of the built-in — G2), exactly as redeclaring `Bool` would collide in
    // `types`.
    //
    // **Why conditional (unlike `Bool`, which is always seeded):** `mono::is_already_monomorphic`
    // (the "is this program already closed — no generics/traits/instances?" fast-path test) and a
    // wide swath of the existing test corpus assert `env.traits.is_empty()` / `mono.traits.is_empty()`
    // for any program that never mentions traits at all. Bool is harmless there (it has empty
    // `params`, so it never trips the *type*-genericity half of that test); an unconditionally
    // seeded `Fuse` trait would trip the *trait*-emptiness half for **every** program, including
    // ones with no `fuse`/`Fuse` in sight — a real regression, not just a test artifact (it would
    // force every trait-free program through mono's slow specializing pass). So `Fuse` is seeded
    // **iff** this nodule's own items need it (an `impl Fuse[...] for ...`) — never based on
    // whether `fuse(a, b)` is *called* (the repr-type fast path in `check_fuse` never touches the
    // trait registry at all, and a Data-type `fuse` call always requires a prior `impl`, which is
    // exactly what this scan detects).
    //
    // FLAG (M-965, narrow, honest residual): a nodule that delegates `Fuse` **only** via `via idx :
    // Fuse` sugar (DN-53) — with no textual `impl Fuse[...] for ...` — is not detected here, because
    // `via`-generated impls are expanded later (`check_nodule_with` Phase 0b, after imports/coherence
    // are available) and are not yet in `nodule.items` at this registration pass. Such a program
    // would see "impl for unknown trait `Fuse`" at via-expansion time — a never-silent refusal, not a
    // silent misbehavior, but a real gap the scan below doesn't yet close (deferred, not hidden).
    let fuse_used = nodule
        .items
        .iter()
        .any(|item| matches!(item, Item::Impl(id) if id.trait_name == crate::fuse::TRAIT_NAME));
    if fuse_used {
        if traits.contains_key(crate::fuse::TRAIT_NAME) {
            return Err(CheckError::new(
                crate::fuse::TRAIT_NAME,
                "cannot redeclare the built-in prelude trait `Fuse` (DN-58 §A / M-965 F-A1) — its \
                 lawful-merge `join` contract is already provided by the prelude; remove this \
                 declaration and `impl Fuse[T] for T { fn join(a: T, b: T) => T = … }` directly",
            ));
        }
        traits.insert(crate::fuse::TRAIT_NAME.to_owned(), crate::fuse::prelude());
    } else if traits.contains_key(crate::fuse::TRAIT_NAME) {
        return Err(CheckError::new(
            crate::fuse::TRAIT_NAME,
            "cannot redeclare the built-in prelude trait `Fuse` (DN-58 §A / M-965 F-A1) — its \
             lawful-merge `join` contract is already provided by the prelude; remove this \
             declaration and `impl Fuse[T] for T { fn join(a: T, b: T) => T = … }` directly",
        ));
    }
    let mut fns: BTreeMap<String, FnDecl> = BTreeMap::new();
    for item in &nodule.items {
        if let Item::Fn(fd) = item {
            if let Some(dup) = first_duplicate(&fd.sig.param_names()) {
                return Err(CheckError::new(
                    &fd.sig.name,
                    format!("duplicate type parameter `{dup}` in `{}`", fd.sig.name),
                ));
            }
            if fns.insert(fd.sig.name.clone(), fd.clone()).is_some() {
                return Err(CheckError::new(&fd.sig.name, "duplicate function"));
            }
        }
    }
    Ok(NoduleRegs { types, fns, traits })
}

/// Resolve one nodule's `use` imports against the phylum-wide [`Exports`] (M-662). Builds the
/// per-nodule [`NoduleImports`] — imported `pub` decls merged by simple name at glob-then-explicit
/// precedence (own decls shadow these later, in [`check_nodule_with`]) — and enforces every
/// never-silent rule:
///
/// - **unknown name/path** → explicit refusal (distinguishing "no such name" from "exists but
///   private", honest + helpful);
/// - **two explicit `use`s binding the same simple name** → duplicate-import refusal;
/// - **glob-vs-glob collision** on a name → recorded `ambiguous` (a *reference* to it is refused at
///   use-site), never a silent winner.
///
/// (A glob over a prefix with zero `pub` names is allowed — an empty import; an unresolved *reference*
/// then surfaces the normal unknown-name error.)
fn resolve_imports(nodule: &Nodule, exports: &Exports) -> Result<NoduleImports, CheckError> {
    let site = qualify(&nodule.path, "<use>");
    let mut imp = NoduleImports::default();
    // Track how each simple name entered (glob vs explicit) so precedence + the dup-explicit and
    // glob-ambiguity rules are enforced deterministically.
    let mut via_explicit: BTreeSet<String> = BTreeSet::new();
    let mut via_glob: BTreeSet<String> = BTreeSet::new();

    // First the globs (lowest precedence), then the explicit `use`s (which shadow a glob name).
    for item in &nodule.items {
        let Item::Use(UsePath { path, glob: true }) = item else {
            continue;
        };
        let prefix = path.0.join(".");
        // Every exported name directly under this prefix (qualified key = prefix + "." + simple, with
        // exactly one trailing segment).
        let mut any = false;
        for qual in exports.declared.keys() {
            let Some(simple) = direct_child(&prefix, qual) else {
                continue;
            };
            // Only `pub` names are importable; a glob silently *skips* a private name (a glob over a
            // path imports its **public** surface — a private name is simply not part of it, which is
            // not a silent swap: it was never importable). Whether it is `pub` is in `exports.*`.
            if !exports_has_pub(exports, qual) {
                continue;
            }
            any = true;
            if via_explicit.contains(&simple) {
                continue; // an explicit use already bound this name (higher precedence)
            }
            if via_glob.contains(&simple) {
                // A second glob brings the same name ⇒ ambiguous (unless later shadowed). Remove the
                // tentative binding; record the ambiguity (never a silent winner — G2).
                imp.ambiguous.insert(simple.clone());
                remove_import(&mut imp, &simple);
                continue;
            }
            via_glob.insert(simple.clone());
            insert_export(&mut imp, exports, qual, &simple);
        }
        let _ = any; // an empty glob (no pub names) is allowed (the reference, if any, fails later)
    }
    // Explicit `use a.b.X` (higher precedence than any glob).
    for item in &nodule.items {
        let Item::Use(UsePath { path, glob: false }) = item else {
            continue;
        };
        // The path's last segment is the imported item; the prefix is its owning nodule path.
        let Some((simple, prefix)) = split_last_seg(path) else {
            return Err(CheckError::new(
                &site,
                "a `use` path must name an item (`use a.b.Item`) — an empty `use` path does not name \
                 a cross-nodule item (M-662)",
            ));
        };
        // A single-segment `use X` names no nodule (prefix empty). Refuse with a teaching diagnostic
        // rather than the confusing downstream "no such name" lookup miss (M-662; never-silent — G2).
        if prefix.is_empty() {
            return Err(CheckError::new(
                &site,
                format!(
                    "`use {simple}`: a cross-nodule import must be nodule-qualified — `{simple}` names \
                     no nodule. Write `use <nodule>.{simple}` (a specific import) or `use <nodule>.*` \
                     (a glob) (M-662)"
                ),
            ));
        }
        let qual = format!("{prefix}.{simple}");
        // Never-silent: unknown path/name vs exists-but-private (honest + helpful — G2).
        match exports.declared.get(&qual) {
            None => {
                return Err(CheckError::new(
                    &site,
                    format!(
                        "`use {}`: no such name `{qual}` in the phylum — no nodule declares it \
                         (M-662; never a silent skip — G2)",
                        path.0.join(".")
                    ),
                ));
            }
            Some(false) => {
                return Err(CheckError::new(
                    &site,
                    format!(
                        "`use {}`: `{qual}` exists but is not `pub` — it is private to its nodule \
                         and not importable (mark it `pub` to export it; M-662)",
                        path.0.join(".")
                    ),
                ));
            }
            Some(true) => {}
        }
        // Duplicate explicit import of the same simple name (ambiguous local binding) — G2.
        if via_explicit.contains(&simple) {
            return Err(CheckError::new(
                &site,
                format!(
                    "duplicate import of `{simple}` — two `use` declarations bind the same name; \
                     import it once (M-662; never a silent shadow — G2)"
                ),
            ));
        }
        via_explicit.insert(simple.clone());
        // An explicit use shadows any glob binding/ambiguity for this name (deterministic precedence).
        imp.ambiguous.remove(&simple);
        insert_export(&mut imp, exports, &qual, &simple);
    }
    Ok(imp)
}

/// Does the export table mark `qual` as a `pub` (importable) name?
fn exports_has_pub(exports: &Exports, qual: &str) -> bool {
    matches!(exports.declared.get(qual), Some(true))
}

/// If `qual` is `prefix` + `.` + a **single** further segment, return that segment (a *direct* child
/// of `prefix`); else `None`. (`"a.b.List"` is a direct child of `"a.b"` ⇒ `List`; `"a.b.c.X"` is
/// not.) Used to expand a glob `use prefix.*` to exactly the names one level under the prefix.
fn direct_child(prefix: &str, qual: &str) -> Option<String> {
    let rest = qual.strip_prefix(prefix)?.strip_prefix('.')?;
    if rest.is_empty() || rest.contains('.') {
        None
    } else {
        Some(rest.to_owned())
    }
}

/// Split a `use` path into `(last_segment, prefix_joined)`; `None` for an empty path. `a.b.Item` ⇒
/// `("Item", "a.b")`; a single-segment `Item` ⇒ `("Item", "")`.
fn split_last_seg(path: &Path) -> Option<(String, String)> {
    let (last, init) = path.0.split_last()?;
    Some((last.clone(), init.join(".")))
}

/// Insert the export `qual` (a `pub` type/fn/trait) into the per-nodule imports under `simple`.
/// Exactly one of the three export tables holds `qual` (a name is one kind); insert from whichever.
fn insert_export(imp: &mut NoduleImports, exports: &Exports, qual: &str, simple: &str) {
    if let Some(info) = exports.types.get(qual) {
        imp.types.insert(simple.to_owned(), info.clone());
    }
    if let Some(fd) = exports.fns.get(qual) {
        imp.fns.insert(simple.to_owned(), fd.clone());
    }
    if let Some(info) = exports.traits.get(qual) {
        imp.traits.insert(simple.to_owned(), info.clone());
    }
}

/// Remove any import binding for `simple` across all three tables (used when a glob-vs-glob collision
/// demotes a tentatively-bound name to `ambiguous`).
fn remove_import(imp: &mut NoduleImports, simple: &str) {
    imp.types.remove(simple);
    imp.fns.remove(simple);
    imp.traits.remove(simple);
}

/// Like [`check_nodule`] but with an explicit `matured_scope` flag (RFC-0017 §4.2): when `true`,
/// every reachable definition whose `thaw == false` must be `Total` (the existing totality
/// classifier, unchanged) — a non-total non-thaw definition is an explicit `CheckError`. Definitions
/// marked `thaw` are exempt from the gate (RFC-0017 §4.3). When `matured_scope` is `false` this
/// is identical to [`check_nodule`].
pub fn check_nodule_matured(nodule: &Nodule, matured_scope: bool) -> Result<Env, CheckError> {
    check_and_resolve_matured(nodule, matured_scope).map(|(env, _)| env)
}

fn check_and_resolve_matured(
    nodule: &Nodule,
    matured_scope: bool,
) -> Result<(Env, Nodule), CheckError> {
    // Run the recursive pass on a deep worker stack so deep-but-valid input never overflows the
    // *caller's* thread stack — the explicit [`MAX_CHECK_DEPTH`] budget, not the host stack, bounds a
    // pathological input (banked guard 4; the worker stack is the transitional Rust-only adapter —
    // see [`mycelium_stack`]). Borrows are fine: the worker is a scoped thread.
    mycelium_stack::with_deep_stack(|| check_and_resolve_matured_inner(nodule, matured_scope))
}

fn check_and_resolve_matured_inner(
    nodule: &Nodule,
    matured_scope: bool,
) -> Result<(Env, Nodule), CheckError> {
    // A bare nodule is a **phylum-of-one** (M-662): route it through the same phylum orchestration so
    // its orphan rule's locality set is exactly this one nodule and its imports are empty — behavior
    // identical to the pre-M-662 single-nodule path, by construction. The `with_deep_stack` is already
    // established by the caller (`check_and_resolve_matured` / `check_phylum_matured`), so call the
    // inner orchestrator directly to avoid nesting worker stacks.
    let resolved = crate::ambient::resolve(nodule)?;
    let phylum = Phylum::of_one(resolved.clone());
    let penv = check_phylum_inner(&phylum, matured_scope)?;
    let env = penv
        .single()
        .expect("a phylum-of-one yields exactly one Env")
        .clone();
    let mut items = Vec::with_capacity(resolved.items.len());
    for item in &resolved.items {
        match item {
            Item::Fn(fd) => {
                let resolved_fd = env
                    .fns
                    .get(&fd.sig.name)
                    .cloned()
                    .unwrap_or_else(|| fd.clone());
                items.push(Item::Fn(resolved_fd));
            }
            other => items.push(other.clone()),
        }
    }
    let twin = Nodule {
        path: resolved.path.clone(),
        // Preserve the `@std-sys` FFI-floor marker (M-661) on the resolved longhand twin — it is
        // header metadata, untouched by ambient resolution / body checking.
        std_sys: resolved.std_sys,
        items,
    };
    Ok((env, twin))
}

/// Like [`check_nodule`], but also returns the **fully-resolved longhand twin** of the program
/// (paradigm tags filled *and* bare-decimal widths resolved from context) — the source the M-142/LSP
/// "expand ambient" projection renders (RFC-0012 §5). The returned [`Nodule`] elaborates to the
/// identical L0 (and content hash) as the original (I2; RFC-0012 §4.3).
pub fn check_and_resolve(nodule: &Nodule) -> Result<(Env, Nodule), CheckError> {
    check_and_resolve_matured(nodule, false)
}

/// Register a (resolved) nodule's **data declarations** into `types` (Pass 1; RFC-0007 §11): a shell
/// per type first (so recursive field references resolve), then its constructors. Duplicate type
/// parameters / duplicate type names are explicit refusals. Extracted (M-662) so the phylum can build
/// its registries once and reuse them; behavior is byte-identical to the inlined Pass 1.
pub(crate) fn register_types(
    types: &mut BTreeMap<String, DataInfo>,
    nodule: &Nodule,
) -> Result<(), CheckError> {
    // M-826: pre-register synthetic Tuple$N types for all arities referenced in this nodule.
    // This must run BEFORE the user-type registration below, because resolve_ty (called during
    // ctor resolution) needs to look up Tuple$N in `types`. Never-silent: resolve_ty checks the
    // table and returns an explicit error if a referenced Tuple$N is missing (G2).
    for arity in collect_tuple_arities(nodule) {
        let tname = tuple_type_name(arity);
        types
            .entry(tname)
            .or_insert_with(|| synthetic_tuple_data(arity));
    }

    for item in &nodule.items {
        if let Item::Type(td) = item {
            if let Some(dup) = first_duplicate(&td.params) {
                return Err(CheckError::new(
                    &td.name,
                    format!("duplicate type parameter `{dup}` in `{}`", td.name),
                ));
            }
            if types.contains_key(&td.name) {
                return Err(CheckError::new(&td.name, "duplicate type declaration"));
            }
            // Insert a shell first so recursive field references resolve.
            types.insert(
                td.name.clone(),
                DataInfo {
                    name: td.name.clone(),
                    params: td.params.clone(),
                    ctors: vec![],
                },
            );
        }
    }
    for item in &nodule.items {
        if let Item::Type(td) = item {
            let ctors = resolve_ctors(types, td)?;
            types.get_mut(&td.name).expect("registered above").ctors = ctors;
        }
    }
    Ok(())
}

// ---- DN-53 M-811: object desugaring helpers ----

/// Structural lowering of an `object` declaration (DN-53 §A.3.2, M-811): expand to the equivalent
/// `Item::Type` + `Item::Impl` + `Item::Fn` items that the existing registration and checking passes
/// already handle.
///
/// The `via` delegation impls are **not** generated here — they require the trait registry (not yet
/// built at the point this is called). They are generated by [`expand_object_via_decls`] and injected
/// as additional `Item::Impl` items before the instance-registration and method-body passes.
///
/// Guarantee: `Declared` (structural one-to-one rewrite; injective by construction — two distinct
/// `object` declarations never produce the same item set, and the set determines the source uniquely;
/// DN-53 §A.3.3). The three-way differential (`tests/object_desugar.rs`) provides `Empirical`
/// agreement between the `object` form and its hand-written lowering.
fn desugar_object_structural(od: &ObjectDecl) -> Vec<Item> {
    // 1. The type declaration: `type Name[params] = CtorName(T1, T2, …)`.
    let type_decl = TypeDecl {
        vis: od.vis,
        name: od.name.clone(),
        params: od.params.clone(),
        ctors: vec![od.ctor.clone()],
    };
    let mut items: Vec<Item> = vec![Item::Type(type_decl)];
    // 2. Each explicit `impl` block, lifted verbatim.
    for impl_decl in &od.impls {
        items.push(Item::Impl(impl_decl.clone()));
    }
    // 3. Each inherent `fn`, lifted verbatim.
    for fn_decl in &od.fns {
        items.push(Item::Fn(fn_decl.clone()));
    }
    // `via_decls` are NOT expanded here; see [`expand_object_via_decls`].
    items
}

/// Generate forwarding `ImplDecl`s for each `via` clause of an `object` declaration
/// (DN-53 §A.3.2, M-811). Called **after** `register_traits` so the trait's method signatures are
/// available.
///
/// For each `via <field_idx> : TraitName[args]`, generates an `impl TraitName for ObjectName` whose
/// methods each destructure the object via a `match` to extract the delegate field, then forward the
/// call. The delegate must implement the trait; the type-checker verifies this when it resolves the
/// forwarding call.
///
/// Guarantee: `Empirical` — the forwarding bodies are synthesized from the trait signatures; the
/// three-way differential (`tests/object_desugar.rs`) pins the agreement
/// `observe(object) == observe(lower(object))` (DN-53 §A.3.3).
///
/// Never-silent (G2): an unknown trait name, an out-of-range field index, or **two `via` clauses on
/// the same object delegating the same trait** (M-966 — an ambiguous choice between two delegates
/// with no deterministic tiebreak) is an explicit [`CheckError`], never a silent accept or a
/// silent first-match pick.
fn expand_object_via_decls(
    od: &ObjectDecl,
    traits: &BTreeMap<String, TraitInfo>,
) -> Result<Vec<ImplDecl>, CheckError> {
    // Ambiguity refusal (M-966, DN-53 §A.3.2): two `via` clauses on the *same* object that both
    // delegate the *same* trait have no deterministic tiebreak between the two candidate delegates
    // — refuse explicitly rather than silently picking the first-declared one (G2). This is checked
    // before any forwarding impl is generated, over `od.via_decls` in **declaration order** (an
    // ordered `Vec`, never a `HashMap`), so the reported candidates are stable across runs — the same
    // ambiguous `object` always names the same two `via` clauses (`Empirical`, pinned by
    // `tests/via_ordering.rs`).
    let mut first_via_for_trait: BTreeMap<&str, u32> = BTreeMap::new();
    for via in &od.via_decls {
        if let Some(&first_idx) = first_via_for_trait.get(via.trait_name.as_str()) {
            return Err(CheckError::new(
                &od.name,
                format!(
                    "`object {}`: ambiguous `via` delegation for trait `{}` — both `via {} : {}` \
                     and `via {} : {}` claim to provide it, and there is no deterministic tiebreak \
                     between two delegates for the same trait (DN-53 §A.3.2; M-966). Fix: keep only \
                     one `via` clause per trait on this object, or replace one of them with an \
                     explicit `impl {} for {}` that picks the winner by hand — never a silent \
                     first-match (G2).",
                    od.name,
                    via.trait_name,
                    first_idx,
                    via.trait_name,
                    via.field_idx,
                    via.trait_name,
                    via.trait_name,
                    od.name
                ),
            ));
        }
        first_via_for_trait.insert(via.trait_name.as_str(), via.field_idx);
    }

    let mut via_impls: Vec<ImplDecl> = Vec::new();
    for via in &od.via_decls {
        // Look up the trait — never-silent (G2).
        let trait_info = traits.get(&via.trait_name).ok_or_else(|| {
            CheckError::new(
                &od.name,
                format!(
                    "`object {}`: `via {} : {}` — trait `{}` is not declared in scope \
                     (DN-53 §A.3.2; `via` requires the trait to be visible — never a silent miss, G2)",
                    od.name, via.field_idx, via.trait_name, via.trait_name
                ),
            )
        })?;
        // Validate the field index — never-silent (G2).
        let n_fields = od.ctor.fields.len();
        if via.field_idx as usize >= n_fields {
            return Err(CheckError::new(
                &od.name,
                format!(
                    "`object {}`: `via {} : {}` — field index {} is out of range; the constructor \
                     `{}` has {} field(s) (0-based; DN-53 §A.3.2; never silent — G2)",
                    od.name, via.field_idx, via.trait_name, via.field_idx, od.ctor.name, n_fields
                ),
            ));
        }
        // Build the object type reference for the `for_ty` of the generated impl.
        // If the object has type params, apply them (`ObjectName[T, U, …]`).
        let obj_ty = TypeRef::unguaranteed(if od.params.is_empty() {
            BaseType::Named(od.name.clone(), vec![])
        } else {
            BaseType::Named(
                od.name.clone(),
                od.params
                    .iter()
                    .map(|p| TypeRef::unguaranteed(BaseType::Named(p.clone(), vec![])))
                    .collect(),
            )
        });
        // Generate binder names for all constructor fields: `_f0`, `_f1`, …
        let binders: Vec<String> = (0..n_fields).map(|i| format!("_f{i}")).collect();
        let delegate_var = binders[via.field_idx as usize].clone();
        // Pattern for the match arm: `CtorName(_f0, _f1, …)`
        let match_pattern = Pattern::Ctor(
            od.ctor.name.clone(),
            binders.iter().map(|b| Pattern::Ident(b.clone())).collect(),
        );
        // Generate one forwarding method per trait signature.
        // Forwarding body for `fn m(self_param, arg2, …) => R`:
        //   m(match self_param { CtorName(_f0, _f1, …) = _fN }, arg2, …)
        // where `_fN` is the delegate field at position `field_idx`.
        let mut methods: Vec<FnDecl> = Vec::new();
        for sig in &trait_info.sigs {
            // Substitute the trait's own type params ↦ this `via` clause's trait args into the
            // method signature (positional zip: `trait_info.params[i]` ↦ `via.trait_args[i]`),
            // reusing the same `subst_type_param_in_sig` machinery `derive`'s Model-A sibling
            // injection uses (M-973). Without this, the generated method's signature would carry
            // the trait's *abstract* parameter type verbatim — but `check_impl_methods` always
            // resolves an impl method's own signature with an EMPTY tyvar scope (an impl is
            // expected to spell out concrete types, exactly as a hand-written
            // `impl Trait[Binary{8}] for T { fn m(x: Binary{8}) => … }` does), so an unsubstituted
            // abstract type would be an "unknown type" `CheckError` for *every* via-delegated
            // parametric trait. An arity mismatch (extra/missing trait args) is left to
            // `register_instances`'s existing arity check — never guessed at here (G2).
            let sig = trait_info
                .params
                .iter()
                .zip(via.trait_args.iter())
                .fold(sig.clone(), |acc, (param, concrete)| {
                    subst_type_param_in_sig(&acc, param, concrete)
                });
            // The first value-parameter is the "self" parameter.
            let self_name = sig
                .value_params
                .first()
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "self".to_string());
            // `match self_param { CtorName(_f0, _f1, …) = _fN }`
            let delegate_expr = Expr::Match {
                scrutinee: Box::new(Expr::Path(Path(vec![self_name]))),
                arms: vec![Arm {
                    pattern: match_pattern.clone(),
                    body: Expr::Path(Path(vec![delegate_var.clone()])),
                }],
            };
            // Call args: the delegate value, then the remaining params by name.
            let mut call_args: Vec<Expr> = vec![delegate_expr];
            for vp in sig.value_params.iter().skip(1) {
                call_args.push(Expr::Path(Path(vec![vp.name.clone()])));
            }
            // Full forwarding call: `method_name(delegate_expr, arg2, …)`
            let body = Expr::App {
                head: Box::new(Expr::Path(Path(vec![sig.name.clone()]))),
                args: call_args,
            };
            methods.push(FnDecl {
                vis: crate::ast::Vis::Private,
                thaw: false,
                tier: None,
                sig,
                body,
            });
        }
        via_impls.push(ImplDecl {
            trait_name: via.trait_name.clone(),
            trait_args: via.trait_args.clone(),
            for_ty: obj_ty,
            methods,
        });
    }
    Ok(via_impls)
}

/// The core checker for **one nodule of a phylum** (M-662), run on an already ambient-resolved nodule
/// with its pre-built declaration registries (`regs`), its resolved cross-nodule `imports`, the
/// phylum-wide pub-blind `coherence` view, and an explicit maturation flag. When `matured_scope` is
/// true, every fn with `thaw == false` must be `Total` (RFC-0017 §4.2).
///
/// The nodule's checking registries = **its own declarations** merged with the **imported `pub`
/// declarations** (own decls take precedence — the documented shadowing). The orphan rule is enforced
/// **phylum-wide** (via `coherence`); name resolution is **never-silent** (an ambiguous glob name is
/// refused at use-site). A phylum-of-one passes empty imports + a coherence view of just this nodule,
/// so this reduces *exactly* to the pre-M-662 single-nodule checker.
fn check_nodule_with(
    nodule: &Nodule,
    regs: NoduleRegs,
    imports: &NoduleImports,
    coherence: &CoherenceView,
    matured_scope: bool,
    via_objects: &[ObjectDecl],
) -> Result<Env, CheckError> {
    // Build the nodule's checking registries: imports first (lower precedence), own decls override
    // (the documented "own decl shadows `use`" precedence — RFC-0006 §4.3). `regs` already holds the
    // prelude + this nodule's own declarations; layering imports *under* them is just inserting any
    // imported name not already present.
    let mut types = imports.types.clone();
    types.extend(regs.types.clone());
    let mut traits = imports.traits.clone();
    traits.extend(regs.traits.clone());
    let mut fns = imports.fns.clone();
    fns.extend(regs.fns.clone());

    // Phase 0b (DN-53 M-811): generate `via` delegation impls now that the trait registry is built.
    // Each `via N : Trait` produces a forwarding `ImplDecl`; these are appended to a working copy of
    // the nodule's items so the instance-registration pass and the impl-method body-checking pass both
    // see them (the structural items — `TypeDecl`, explicit `ImplDecl`, `FnDecl` — were already
    // inlined into the nodule by `check_phylum_inner`'s Phase 0 expansion; only the `via`-derived
    // impls arrive here). Guarantee: `Empirical` (three-way differential, `tests/object_desugar.rs`).
    let via_generated_impls: Vec<ImplDecl> = {
        let mut all = Vec::new();
        for od in via_objects {
            all.extend(expand_object_via_decls(od, &traits)?);
        }
        all
    };
    // EXPLAIN provenance (M-966, DN-53 §A.3.3): record which `via` clause produced each generated
    // forwarding impl, the `via` analogue of `derived_provenance` below. The coherence key matches
    // `register_instances`'s `(trait_name, type_head)`: for an `object`, `type_head` always resolves
    // to `Data:<name>` (an object's own declared type name is never itself a bound type-variable —
    // `resolve_ty`'s `Ty::Var` branch only fires for names in `tyvars`, which an object's own type
    // name is not), so this can be built directly without a `resolve_ty` round-trip. Only reached
    // when `expand_object_via_decls` above did **not** refuse for ambiguity, so at most one `via`
    // entry per `(trait, object)` ever lands here (never overwritten silently).
    let via_provenance: BTreeMap<(String, String), (u32, String)> = {
        let mut all = BTreeMap::new();
        for od in via_objects {
            let head = format!("Data:{}", od.name);
            for via in &od.via_decls {
                all.insert(
                    (via.trait_name.clone(), head.clone()),
                    (via.field_idx, od.name.clone()),
                );
            }
        }
        all
    };
    // Phase 0c (DN-54 §10 Model A / DN-81 §10.4 / M-973): **register `lower` rules and expand
    // `derive` sites to sibling `impl`s BEFORE Pass 2b/3b** — the deliberate affine wiring the
    // ratified correction demands. This is the enactment of the attachment model: a `derive Name for
    // C` whose rule has an item-shaped RHS (`lower Name[T] = impl Trait for T { … }`) is instantiated
    // (substitute `C` for the rule's type param) and its concrete `impl` **injected as a sibling
    // item** into `effective_nodule` — exactly as `via` injects its forwarding impls. Because the
    // injection happens *before* `register_instances` (Pass 2b, coherence/orphan) and
    // `check_impl_methods` (Pass 3b, whose `check_fn_body` runs the **active** M-919 affine tracker),
    // the derived impl gets coherence, orphan, effect, guarantee-grading **and affine use-once**
    // coverage **by construction** — the affine wiring is "process derives before Pass 3b" (DN-81
    // §10.4 option (a)), never an unchecked no-op (proved by the derive-site double-consume reject
    // test). Only the *item-shaped* rules inject a sibling; an expression-shaped rule's `derive` has
    // no sibling (its L0 is the `elaborate_lower_rule` nullary term — unchanged). All never-silent
    // (G2): an unknown rule, an ill-formed target type, or a non-single type-parameter item rule is an
    // explicit refusal here, before any sibling is built.
    //
    // Structural registration + §4.2 acyclicity run here too (moved up from the former Pass 3e) so
    // `lower_rules` is populated before the derive expansion reads it; the §4.1 RHS type-check stays
    // *after* Pass 2b (Pass 3e-late) because it needs the resolved `instances`.
    let mut lower_rules: BTreeMap<String, LowerDecl> = BTreeMap::new();
    for item in &nodule.items {
        if let Item::Lower(ld) = item {
            check_lower_decl_structural(ld, &lower_rules)?;
            lower_rules.insert(ld.name.clone(), ld.clone());
        }
    }
    check_lower_rule_acyclicity(&lower_rules, &types, &fns)?;
    // Expand each `derive` into its concrete sibling impl (item-shaped rules only). De-dup by the
    // ADR-003 content key `(trait, type_head)`: two *identical* `derive`s collapse to one injected
    // impl (a no-op duplicate — DN-54 §10.3); a genuine conflict (a different impl on the same key,
    // or a hand-written impl on that key) is left for `register_instances` to refuse as a coherence
    // violation, never silently (G2). Provenance (OQ-A / DN-81 §6.5) is recorded per injected impl.
    let mut derive_generated_impls: Vec<ImplDecl> = Vec::new();
    let mut derived_provenance: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
    let mut derived_seen: BTreeMap<(String, String), ImplDecl> = BTreeMap::new();
    for item in &nodule.items {
        let Item::Derive(dd) = item else { continue };
        // Rule resolution + target-type well-formedness (never-silent, G2).
        check_derive_application(dd, &lower_rules, &types)?;
        let rule = &lower_rules[&dd.name];
        let Some(template) = rule.impl_rhs() else {
            // An expression-shaped rule: no sibling impl to inject (its L0 is the nullary
            // `elaborate_lower_rule` term). The derive site is validated (above); nothing attaches.
            continue;
        };
        // Model A instantiation requires exactly one type parameter to bind to the `derive`'s target
        // type (`lower Name[T] = impl … for T`; DN-54 §3.2). A nullary or multi-param item rule is a
        // never-silent refusal in v1 (multi-param instantiation is OQ-C, a future extension).
        if rule.params.len() != 1 {
            return Err(CheckError::new(
                &dd.name,
                format!(
                    "`derive {} for …` targets an item-shaped rule with {} type parameter(s), but \
                     Model-A sibling injection binds the single `derive` target type to exactly one \
                     rule parameter (`lower {}[T] = impl … for T`; DN-54 §10.3 / OQ-C multi-param is \
                     a future extension — never a silent mis-instantiation, G2)",
                    dd.name,
                    rule.params.len(),
                    dd.name
                ),
            ));
        }
        let concrete = subst_type_param_in_impl(template, &rule.params[0], &dd.for_ty);
        // The content key = the resolved `(trait, type_head)` of the concrete impl — the same key
        // `register_instances` coheres on. Resolve the head now for the de-dup + provenance key.
        let (for_ty_resolved, _) = resolve_ty(&dd.name, &types, &[], &concrete.for_ty)?;
        let Some(head) = type_head(&for_ty_resolved) else {
            return Err(CheckError::new(
                &dd.name,
                format!(
                    "`derive {} for {for_ty_resolved}` instantiates to an `impl … for \
                     {for_ty_resolved}` whose target is not a concrete instance head (DN-54 §10.2 / \
                     RFC-0019 §4.5 — a derive target must be a concrete type, never a bare type \
                     variable; G2)",
                    dd.name
                ),
            ));
        };
        let key = (concrete.trait_name.clone(), head);
        match derived_seen.get(&key) {
            // Identical prior derive on the same key ⇒ ADR-003 content-key de-dup: a no-op (DN-54
            // §10.3). A *different* impl on the same key is NOT resolved here — it is left to
            // `register_instances`, which refuses it as an overlapping-instance coherence violation
            // (the correct never-silent diagnostic, G2). So we only skip an exact duplicate.
            Some(prev) if *prev == concrete => continue,
            _ => {}
        }
        derived_seen.insert(key.clone(), concrete.clone());
        // Provenance (OQ-A / DN-81 §6.5 / RFC-0001 §4.3): record the originating rule + target so a
        // derived impl is distinguishable from a hand-written one (`Declared` — carried, not upgraded).
        derived_provenance.insert(key, (dd.name.clone(), format!("{for_ty_resolved}")));
        derive_generated_impls.push(concrete);
    }

    // Build the effective nodule (with via impls AND derive-generated sibling impls appended) for the
    // instance + impl-body passes. If neither is present, borrow the original nodule directly (avoids
    // a clone in the common case). The `Option` owner keeps the `Nodule` alive long enough for the
    // borrow `effective_nodule` to be valid across pass 2b and 3b below.
    let effective_nodule_owned: Option<Nodule> =
        if via_generated_impls.is_empty() && derive_generated_impls.is_empty() {
            None
        } else {
            let mut items = nodule.items.clone();
            for id in &via_generated_impls {
                items.push(Item::Impl(id.clone()));
            }
            for id in &derive_generated_impls {
                items.push(Item::Impl(id.clone()));
            }
            Some(Nodule {
                path: nodule.path.clone(),
                std_sys: nodule.std_sys,
                items,
            })
        };
    let effective_nodule: &Nodule = effective_nodule_owned.as_ref().unwrap_or(nodule);

    // Pass 2b: register trait **instances** (RFC-0019 §4.5). Coherence (global uniqueness + the
    // **phylum-wide** orphan rule) is enforced as each instance is registered; ALL instances are
    // registered before any method body is checked (Pass 3b), so a method body may rely on an instance
    // declared by a *later* `impl`. This pass resolves heads + checks coherence; it does not yet check
    // bodies. The orphan rule consults the pub-blind phylum-wide `coherence` view (M-662).
    // Uses `effective_nodule` so via-generated impls are included in the instance registry.
    let instances = register_instances(&types, &traits, coherence, effective_nodule)?;

    // Pass 3: type every (own) body **against** its declared return type (bidirectional, RFC-0012
    // §4.3), with imports available, and resolve any ambient bare-decimal widths from context —
    // rewriting each body so the downstream evaluator/elaborator see only concrete literals. Only this
    // nodule's *own* fns are (re)checked + stored (imported fns were already checked in their home
    // nodule — RFC-0007 §11; a `use`d fn is checked in its home nodule's context, never re-checked
    // here under this nodule's ambient).
    let mut resolved_fns: BTreeMap<String, FnDecl> = fns.clone();
    for fd in regs.fns.values() {
        let (body, _ret) = check_fn_body(
            &types,
            &fns,
            &traits,
            &instances,
            imports,
            nodule.std_sys,
            fd,
        )?;
        resolved_fns.insert(
            fd.sig.name.clone(),
            FnDecl {
                vis: fd.vis,
                thaw: fd.thaw,
                tier: fd.tier,
                sig: fd.sig.clone(),
                body,
            },
        );
    }
    let fns = resolved_fns;

    // Pass 3b: check each `impl` method body against its **expected** signature (the trait sig with
    // the trait's params substituted by this impl's trait_args — RFC-0019 §4.5). The instance set is
    // fully registered (Pass 2b), so a method may use any instance. The method bodies are not re-stored
    // for *elaboration* (the elaborator stages the dictionary lowering — M-673), but their **resolved
    // (canonical) form** is collected so the guarantee-grading pass (3d) walks the same normalized AST
    // as a top-level fn — patterns already ctor/binder-resolved (M-663 / Copilot review).
    let mut resolved_impl_methods: Vec<FnDecl> = Vec::new();
    // M-673: retain each instance's resolved method bodies, keyed `(trait, type_head(for_ty))` —
    // parallel to `instances` (head-granular; coherence guarantees ≤ 1 per key). The grading pass (3d)
    // still consumes the flat `resolved_impl_methods` list; the keyed `impls` map is what the
    // monomorphization pre-pass (`crate::mono`) reads to lower a trait-method call to a direct call.
    // Uses `effective_nodule` so via-generated impl method bodies are also checked (DN-53 M-811).
    let mut impls: BTreeMap<(String, String), Vec<FnDecl>> = BTreeMap::new();
    for item in &effective_nodule.items {
        if let Item::Impl(id) = item {
            let methods = check_impl_methods(
                &types,
                &fns,
                &traits,
                &instances,
                imports,
                effective_nodule.std_sys,
                id,
            )?;
            // The instance head: resolve `for_ty` exactly as `register_instances` did (concretely, no
            // type-vars in scope). Registration already accepted this impl, so resolution + a concrete
            // head are guaranteed here (a `Ty::Var` head was refused at registration); the `expect`
            // documents that invariant rather than silently dropping a method set.
            let (for_ty, _) = resolve_ty(&id.trait_name, &types, &[], &id.for_ty)?;
            let head = type_head(&for_ty)
                .expect("instance registration refused a non-concrete `for` type head");
            impls.insert((id.trait_name.clone(), head), methods.clone());
            resolved_impl_methods.extend(methods);
        }
    }

    // Pass 3b-2 (M-965; DN-58 §A F-A2): the `Fuse` semilattice-law checker. Every `impl Fuse[T]
    // for T` whose `T` is a finitely-enumerable domain (a nullary-constructor `Data` type — the
    // `Bool`-shape) is exhaustively checked against the three semilattice laws (idempotence,
    // commutativity, associativity — RFC-0008 RT6); a violation is an explicit, never-silent
    // `CheckError` naming the failed law + a concrete witness (G2), refused **here at definition**,
    // never reaching a `fuse` call site. A non-enumerable `T` (has fields, is parametric/recursive)
    // is honestly left **unchecked** in v0 (DN-58 §A.6 F-A3 — deferred, never silently claimed
    // lawful; VR-5). Runs after Pass 3b so every instance's resolved method bodies are available.
    crate::fuse::check_fuse_laws(&types, &fns, &traits, &instances, &impls)?;

    // Pass 3e-late: **§4.1 IL-grammar RHS type-check** of each `lower` rule (DN-54 §4.1 / M-812-cont).
    // The structural checks (§3 uniqueness / param-uniqueness / §4.6 `wild`-purity), rule registration,
    // §4.2 acyclicity, and the `derive`-site expansion all ran in **Phase 0c** above (before Pass 2b),
    // so the derived sibling impls are already coherence/orphan-checked (Pass 2b) and their method
    // bodies affine/body-checked (Pass 3b) — the DN-81 §10.4 affine wiring. What remains here is the
    // per-rule RHS type-check, which is deferred to *after* Pass 2b because it needs the resolved
    // `instances` (a rule's RHS may call trait methods). For an **expression**-shaped rule this is the
    // IL-grammar `infer` (with the M-919 active affine tracker); for an **item**-shaped rule it is the
    // template's trait/method-set validation (its full body/affine check already happened at each
    // derive site — DN-54 §10.4). See [`check_lower_rule_rhs_type`].
    //
    // Guarantee posture (VR-5, honest): the RHS type-check is `Declared` (a structural IL-grammar
    // check, not a theorem). KC-3 (§6) holds **by construction** — the elaborator's codomain is the
    // *closed* [`mycelium_core::Node`] enum, so a rule can add no kernel node; the §4.6 `wild`-refusal
    // (Phase 0c) closes the one surface-growth a rule could smuggle in. The elaboration itself
    // ([`crate::elab::elaborate_lower_rule`]) is `Empirical` (earned by the §7 differential).
    for ld in lower_rules.values() {
        check_lower_rule_rhs_type(ld, &types, &fns, &traits, &instances, imports)?;
    }

    // Pass 3c: **effect coverage** (RFC-0014 §3.4/§4.5 I3; M-660 — guarantee: `Declared`, a
    // structural coverage check, not a theorem). Every effect a fn *performs* — the union of the
    // declared effects of every top-level fn it calls — must be in its own *declared* set. An
    // undeclared performed effect is an explicit `CheckError` naming the effect and the callee that
    // introduces it (under-declaration is never silent — G2/RFC-0014 I3). Over-declaration is
    // allowed (a declaration is a contract — RFC-0014 I5). Run after bodies type-check, over the
    // checked `fns` table.
    // Effect coverage runs over this nodule's **own** fn bodies (and its impl methods); an imported
    // fn was already coverage-checked in its home nodule (M-662 — a `use`d fn is checked in its home
    // context, never re-litigated here). The merged `fns`/`traits` give the callee effect lookups.
    // Uses `effective_nodule` so via-generated impl methods' bodies are included in coverage check.
    check_effect_coverage(&fns, &regs.fns, &traits, effective_nodule)?;

    // Pass 3d: **guarantee grading** (RFC-0018 §4.3 stage-1a, Design A — guarantee: `Declared`). The
    // guarantee index `@ g` becomes a statically-enforced constraint over the integrity lattice
    // `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`: every call's argument must satisfy its callee
    // parameter's demand, and each body must satisfy its declared return demand (G-App/G-Weaken). Runs
    // after bodies type-check, over the merged `fns` (so a call to an imported `pub fn` resolves to
    // that callee's declared grades) and this nodule's own fns + impl methods. A violation is an
    // explicit `CheckError` (never silent — G2/VR-5). This is the static successor to RFC-0007 §4.3's
    // stage-0 dynamic check (which remains the runtime semantics); the noninterference *theorem* stays
    // Declared-with-argument (RFC-0018 §11 / `research/09`), not upgraded.
    crate::grade::check_guarantees(&fns, &regs.fns, &resolved_impl_methods)?;

    // Pass 4: totality classification + the scope-quantified matured gate (RFC-0017 §4.2). Classify
    // over the merged `fns` so an own fn calling an imported one classifies against the real callee.
    // When `matured_scope` is true, every **own** fn with `thaw == false` must be `Total`; a non-total
    // non-thaw fn is an explicit error (RFC-0007 §4.5 / RFC-0017 §4.2). A `thaw` fn is exempt.
    // (Imported fns are gated by their *own* nodule's scope, not this one — M-662.)
    let totality = crate::totality::classify_all(&fns)
        .map_err(|e| CheckError::new(&qualify(&nodule.path, "<totality>"), e.to_string()))?;
    if matured_scope {
        for fd in regs.fns.values() {
            if !fd.thaw && totality[&fd.sig.name] != crate::totality::Totality::Total {
                return Err(CheckError::new(
                    &fd.sig.name,
                    format!(
                        "`{}` is in a matured scope and must be total (RFC-0007 §4.5 / \
                         RFC-0017 §4.2) — mark it `thaw fn` to exempt it, or make it total",
                        fd.sig.name
                    ),
                ));
            }
        }
    }

    Ok(Env {
        types,
        fns,
        totality,
        traits,
        instances,
        impls,
        lower_rules,
        derived_provenance,
        via_provenance,
    })
}

// ---- lower / derive validation (DN-54 §4/§6 / M-812-cont) ----

/// **Structural** validation of a `lower Name[params] = <rhs>` declaration (DN-54 §3/§4.6 /
/// M-812-cont) — the part that does **not** need the full typing context. Run in the registration
/// loop *before* acyclicity (§4.2) and the §4.1 RHS type-check (which run over the whole set).
///
/// **Guarantee: `Declared`** (structural checks, not theorems). Checks (all never-silent, G2):
/// - **Uniqueness**: the rule `name` must not already be registered.
/// - **Parameter uniqueness**: duplicate param names in `[…]` are refused.
/// - **§4.6 purity**: the RHS contains no `wild { … }` block — a generative lowering is a *pure*
///   compile-time mechanism (the FFI gate is level-independent — DN-38 §3), refused structurally so
///   the refusal holds even in an `@std-sys` nodule (where the ordinary `wild` gate would accept it).
fn check_lower_decl_structural(
    ld: &LowerDecl,
    lower_rules: &BTreeMap<String, LowerDecl>,
) -> Result<(), CheckError> {
    // (1) Uniqueness — no two `lower` rules may share a name in the same nodule.
    if lower_rules.contains_key(&ld.name) {
        return Err(CheckError::new(
            &ld.name,
            format!(
                "duplicate `lower` rule `{}` — a generative-lowering rule name must be unique \
                 within a nodule (DN-54 §4.2 / M-812; never silent, G2)",
                ld.name
            ),
        ));
    }
    // (2) Parameter uniqueness — `lower Name[T, T, …]` with a repeated param is an error.
    if let Some(dup) = first_duplicate(&ld.params) {
        return Err(CheckError::new(
            &ld.name,
            format!(
                "duplicate type parameter `{dup}` in `lower {}[…]` (DN-54 §3 / M-812; \
                 every param name must be distinct)",
                ld.name
            ),
        ));
    }
    // (3) §4.6 purity — a `lower` rule's RHS must be a pure compile-time term; a `wild { … }` block
    // (the audited FFI floor — LR-9/S6) is refused **structurally** here so the diagnostic names
    // DN-54 §4.6 precisely (and so the refusal holds even in an `@std-sys` nodule). Never-silent (G2).
    // The RHS is pure whether it is expression-shaped (walk the expr) or item-shaped (walk every
    // method body of the `impl` template — DN-54 §10 Model A, M-973). A `wild` anywhere is refused.
    let has_wild = match &ld.rhs {
        crate::ast::LowerRhs::Expr(e) => {
            rhs_contains_wild(e).map_err(|err| CheckError::new(&ld.name, err.to_string()))?
        }
        crate::ast::LowerRhs::Impl(id) => {
            let mut found = false;
            for m in &id.methods {
                if rhs_contains_wild(&m.body)
                    .map_err(|err| CheckError::new(&ld.name, err.to_string()))?
                {
                    found = true;
                    break;
                }
            }
            found
        }
    };
    if has_wild {
        return Err(CheckError::new(
            &ld.name,
            format!(
                "`lower {}`'s RHS contains a `wild {{ … }}` block — a generative-lowering rule is a \
                 pure compile-time mechanism and may not perform host/FFI calls (DN-54 §4.6 / \
                 §3.3; the `wild` gate is level-independent — DN-38 §3; never silent, G2)",
                ld.name
            ),
        ));
    }
    Ok(())
}

/// **§4.1 IL-grammar RHS type-check** of a registered `lower` rule (DN-54 §4.1 / M-812-cont). The
/// RHS must type-check as an L1 expression with the rule's type parameters in scope as abstract
/// type-variables (the rule is parametric over the type it is derived for). An ill-typed / ill-formed
/// RHS is an explicit refusal at definition time — so no `derive` site can invoke a rule that would
/// produce broken L0. Run over the whole rule set *after* §4.2 acyclicity (so a cyclic set reports
/// the cycle, not an "unknown name" for the cycle's rule-reference edge).
///
/// **Guarantee: `Declared`** (a structural type/grammar check, not a theorem — VR-5) **layered with
/// `Empirical`** (M-919; DN-71 Model S §4.2) for any `Substrate`-typed binding the RHS introduces —
/// see the affine-tracker note below. The RHS has no *value parameters* in v0 (`lower Name[T] =
/// <expr>`), so the value scope starts empty; the rule's `[…]` params are the *type* scope only. No
/// expected type is pinned (a rule may lower to any well-typed term) — pure inference. The
/// `@std-sys` gate is held **closed**: a `lower` rule is never an FFI escape (the §4.6 structural
/// check already refused any direct `wild`; this is defense in depth).
///
/// **Affine tracking is ACTIVE here, seeded empty (M-919 fix; DN-71 Model S §4.2).** A `lower`
/// rule's RHS has no *initial* `Substrate` binding (no value params, no literal — DN-71 §4.1), but
/// its RHS **can** legally introduce one mid-body: DN-54 §3.3 permits calling other already-checked
/// top-level `fn`s, and nothing in v0 restricts those callees to non-affine return types, so a `let
/// s = acquire_thing() in …` inside a rule's RHS can bind a real `Substrate{tag}` value (`acquire_thing`
/// itself having gone through the ordinary `check_fn_body`'s active tracker at its own definition
/// site, in an `@std-sys` nodule). An **inert** tracker here would silently exempt every such binding
/// from the double-consume check ordinary function bodies get — a real, verified coverage gap (a
/// probe reproduction confirmed a same-rule double-consume via a helper-fn-acquired `Substrate` type-checked
/// under the pre-fix inert tracker). Seeding an **active-but-empty** tracker (mirroring
/// `check_fn_body`'s `Tracker::seeded`, just from an empty initial scope, since a rule has no value
/// params to seed from) closes this: every `let`/lambda/match binder pushed during the RHS walk is
/// tracked exactly as it would be inside an ordinary function body, so a double-consume of a
/// derive-site-acquired `Substrate` is refused here too, never silently.
fn check_lower_rule_rhs_type(
    ld: &LowerDecl,
    types: &BTreeMap<String, DataInfo>,
    fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    instances: &BTreeMap<(String, String), InstanceInfo>,
    imports: &NoduleImports,
) -> Result<(), CheckError> {
    // An **item-shaped** rule (DN-54 §10 Model A, M-973) is not an expression: its full type-check
    // happens at each `derive` site, over the concrete (substituted) sibling `impl` — through the
    // ordinary `register_instances` (coherence/orphan) + `check_impl_methods` (bodies + the M-919
    // affine tracker) passes. At definition time we validate only what is decidable on the parametric
    // template without a concrete `for` type: the trait must exist and the method **set** must match
    // the trait's requirement set exactly (never silently filled/dropped — G2). A rule that is never
    // derived is thus still refused if its impl names an unknown trait or the wrong method set; a
    // *derived* rule gets the full body/affine check at its use site (DN-54 §10.4). This is honest
    // (VR-5): the template's per-body type-check is `Declared` here and earned `Empirical`/checked at
    // the derive site, never over-claimed at definition.
    if let crate::ast::LowerRhs::Impl(id) = &ld.rhs {
        let Some(tr) = traits.get(&id.trait_name) else {
            return Err(CheckError::new(
                &ld.name,
                format!(
                    "`lower {}`'s item RHS is `impl {} for …`, but trait `{}` is not declared in \
                     scope (DN-54 §10.3 Model A; a derive template's trait must be visible — never \
                     silent, G2)",
                    ld.name, id.trait_name, id.trait_name
                ),
            ));
        };
        check_impl_method_set(tr, id).map_err(|e| {
            CheckError::new(
                &ld.name,
                format!(
                    "`lower {}`'s item RHS `impl {} for …`: {}",
                    ld.name, id.trait_name, e.message
                ),
            )
        })?;
        return Ok(());
    }
    let rhs = ld
        .expr_rhs()
        .expect("item-shaped rules returned above; this is the expression arm");
    let cx = Cx {
        site: &ld.name,
        types,
        fns,
        traits,
        instances,
        imports,
        tyvars: &ld.params,
        bounds: &[],
        std_sys: false,
        depth: Cell::new(0),
        // M-919 / DN-71 Model S §4.2: active, seeded from the (empty) initial value scope — see the
        // doc comment above. A `lower` rule has no value *parameters*, but its RHS can still
        // introduce a `Substrate`-typed local via a helper-fn call, and that binding must be
        // use-once-checked exactly like any other affine binding (never silently exempted).
        affine: Tracker::seeded(&[]),
    };
    let mut scope: Vec<(String, Ty)> = Vec::new();
    cx.infer(&mut scope, rhs).map_err(|e| {
        CheckError::new(
            &ld.name,
            format!(
                "`lower {}`'s RHS fails the IL-grammar / type check (DN-54 §4.1): {}",
                ld.name, e.message
            ),
        )
    })?;
    Ok(())
}

/// Does the expression tree contain a `wild { … }` block anywhere? (DN-54 §4.6 — a `lower` rule's
/// RHS must be pure.) Returns `true` at the first `wild` found. Uses the shared stateless
/// [`crate::totality::walk_expr`] traversal (DRY — same tree walk every pass uses), which carries
/// its own explicit recursion-depth budget (M-674) — a pathologically-nested RHS refuses cleanly
/// (propagated to the caller) rather than overflowing the host stack.
fn rhs_contains_wild(rhs: &Expr) -> Result<bool, crate::totality::WalkDepthExceeded> {
    let mut found = false;
    crate::totality::walk_expr(rhs, &mut |x| {
        if matches!(x, Expr::Wild(_)) {
            found = true;
        }
    })?;
    Ok(found)
}

// ---- DN-54 §10 Model A: derive-site type-parameter instantiation (M-973) ----
//
// A `derive Name for C` instantiates an item-shaped `lower Name[T] = impl … for T { … }` rule by
// substituting the concrete target `C` for the rule's single type parameter `T` throughout the
// `impl` template, then injecting the resulting concrete `ImplDecl` as a sibling item (Phase 0c). The
// substitution is **purely structural** (no typing context needed), **Swap-free** (it never inserts a
// representation change — VR-5/G2), and **hygiene-preserving** (a method that shadows the rule param
// with its own type parameter is left untouched). Any occurrence the substitution does not reach (a
// type-parameter mention the walk misses) is caught **never-silently** downstream: the concrete impl
// is fully re-checked by `register_instances` + `check_impl_methods`, where an unresolved `T` is an
// explicit "unknown type" [`CheckError`] — a missed substitution degrades to a loud refusal, never a
// silent mis-lowering (G2).

/// Substitute the rule's type parameter `param` with the concrete type `concrete` throughout a
/// [`TypeRef`] (DN-54 §10 Model A instantiation; M-973). A bare `Named(param, [])` occurrence becomes
/// `concrete` (keeping the occurrence's own guarantee index if written — `T @ Exact` ↦ `C @ Exact`);
/// every structural form recurses; reprs / `Substrate` / `Bytes` / `Float` / VSA / dense carry no
/// nested type-name and are cloned verbatim. Total, allocation-bounded by the input's size.
fn subst_type_param_in_typeref(tr: &TypeRef, param: &str, concrete: &TypeRef) -> TypeRef {
    match &tr.base {
        BaseType::Named(name, args) if name == param && args.is_empty() => TypeRef {
            base: concrete.base.clone(),
            guarantee: tr.guarantee.or(concrete.guarantee),
        },
        BaseType::Named(name, args) => TypeRef {
            base: BaseType::Named(
                name.clone(),
                args.iter()
                    .map(|a| subst_type_param_in_typeref(a, param, concrete))
                    .collect(),
            ),
            guarantee: tr.guarantee,
        },
        BaseType::Seq { elem, len } => TypeRef {
            base: BaseType::Seq {
                elem: Box::new(subst_type_param_in_typeref(elem, param, concrete)),
                len: *len,
            },
            guarantee: tr.guarantee,
        },
        BaseType::Fn(a, b) => TypeRef {
            base: BaseType::Fn(
                Box::new(subst_type_param_in_typeref(a, param, concrete)),
                Box::new(subst_type_param_in_typeref(b, param, concrete)),
            ),
            guarantee: tr.guarantee,
        },
        BaseType::Tuple(elems) => TypeRef {
            base: BaseType::Tuple(
                elems
                    .iter()
                    .map(|e| subst_type_param_in_typeref(e, param, concrete))
                    .collect(),
            ),
            guarantee: tr.guarantee,
        },
        // No nested type-name that could be the rule parameter (widths are a *separate* parameter
        // space — DN-42; a type parameter never appears in a width slot). Clone verbatim.
        BaseType::Binary(_)
        | BaseType::Ternary(_)
        | BaseType::Dense(_, _)
        | BaseType::Vsa { .. }
        | BaseType::Substrate(_)
        | BaseType::Bytes
        | BaseType::Float
        | BaseType::Ambient(_) => tr.clone(),
    }
}

/// Substitute `param` ↦ `concrete` in a function **signature** (value-parameter types + return type;
/// M-973). The signature's own type parameters (`sig.params`) and effect annotations are cloned
/// verbatim — a method's own generics are a separate, shadowing scope, and effects carry no type.
fn subst_type_param_in_sig(sig: &FnSig, param: &str, concrete: &TypeRef) -> FnSig {
    FnSig {
        name: sig.name.clone(),
        params: sig.params.clone(),
        value_params: sig
            .value_params
            .iter()
            .map(|p| Param {
                name: p.name.clone(),
                ty: subst_type_param_in_typeref(&p.ty, param, concrete),
            })
            .collect(),
        ret: subst_type_param_in_typeref(&sig.ret, param, concrete),
        effects: sig.effects.clone(),
        effect_budgets: sig.effect_budgets.clone(),
    }
}

/// Substitute `param` ↦ `concrete` throughout an **expression** (a derived method body; M-973). The
/// only positions an [`Expr`] carries a [`TypeRef`] are `Let`'s ascription, `Swap`'s target,
/// `Lambda`'s parameter types, and `Ascribe`; everything else recurses structurally. Total; a
/// [`Literal::List`]'s element expressions recurse, other literals are cloned.
fn subst_type_param_in_expr(e: &Expr, param: &str, concrete: &TypeRef) -> Expr {
    let boxed = |x: &Expr| Box::new(subst_type_param_in_expr(x, param, concrete));
    match e {
        Expr::Let {
            name,
            ty,
            bound,
            body,
        } => Expr::Let {
            name: name.clone(),
            ty: ty
                .as_ref()
                .map(|t| subst_type_param_in_typeref(t, param, concrete)),
            bound: boxed(bound),
            body: boxed(body),
        },
        Expr::If { cond, conseq, alt } => Expr::If {
            cond: boxed(cond),
            conseq: boxed(conseq),
            alt: boxed(alt),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: boxed(scrutinee),
            arms: arms
                .iter()
                .map(|a| Arm {
                    pattern: a.pattern.clone(),
                    body: subst_type_param_in_expr(&a.body, param, concrete),
                })
                .collect(),
        },
        Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        } => Expr::For {
            x: x.clone(),
            xs: boxed(xs),
            acc: acc.clone(),
            init: boxed(init),
            body: boxed(body),
        },
        Expr::Swap {
            value,
            target,
            policy,
        } => Expr::Swap {
            value: boxed(value),
            target: subst_type_param_in_typeref(target, param, concrete),
            policy: policy.clone(),
        },
        Expr::WithParadigm { paradigm, body } => Expr::WithParadigm {
            paradigm: *paradigm,
            body: boxed(body),
        },
        Expr::Wild(b) => Expr::Wild(boxed(b)),
        Expr::Spore(b) => Expr::Spore(boxed(b)),
        Expr::Consume(b) => Expr::Consume(boxed(b)),
        Expr::Colony(hyphae) => Expr::Colony(
            hyphae
                .iter()
                .map(|h| Hypha {
                    forage: h.forage.as_ref().map(|f| boxed(f)),
                    body: subst_type_param_in_expr(&h.body, param, concrete),
                })
                .collect(),
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params: params
                .iter()
                .map(|p| Param {
                    name: p.name.clone(),
                    ty: subst_type_param_in_typeref(&p.ty, param, concrete),
                })
                .collect(),
            body: boxed(body),
        },
        Expr::App { head, args } => Expr::App {
            head: boxed(head),
            args: args
                .iter()
                .map(|a| subst_type_param_in_expr(a, param, concrete))
                .collect(),
        },
        Expr::Fuse { left, right } => Expr::Fuse {
            left: boxed(left),
            right: boxed(right),
        },
        Expr::Reclaim { policy, body } => Expr::Reclaim {
            policy: boxed(policy),
            body: boxed(body),
        },
        Expr::Ascribe(inner, ty) => Expr::Ascribe(
            boxed(inner),
            subst_type_param_in_typeref(ty, param, concrete),
        ),
        Expr::TupleLit(elems) => Expr::TupleLit(
            elems
                .iter()
                .map(|x| subst_type_param_in_expr(x, param, concrete))
                .collect(),
        ),
        Expr::Lit(Literal::List(elems)) => Expr::Lit(Literal::List(
            elems
                .iter()
                .map(|x| subst_type_param_in_expr(x, param, concrete))
                .collect(),
        )),
        // A `Path`, or any non-`List` literal, carries no type-parameter occurrence — clone verbatim.
        Expr::Path(_) | Expr::Lit(_) => e.clone(),
    }
}

/// Instantiate an item-shaped `lower`-rule template at a concrete target type — the Model-A
/// substitution `param ↦ concrete` over an entire [`ImplDecl`] (DN-54 §10.3; M-973). Substitutes the
/// trait arguments, the `for` type, and each method's signature + body. **Hygiene (never-silent
/// correctness):** a method that shadows the rule parameter with its **own** type parameter of the
/// same name is left untouched (the inner binding wins; the rule param does not leak past the shadow).
fn subst_type_param_in_impl(id: &ImplDecl, param: &str, concrete: &TypeRef) -> ImplDecl {
    ImplDecl {
        trait_name: id.trait_name.clone(),
        trait_args: id
            .trait_args
            .iter()
            .map(|a| subst_type_param_in_typeref(a, param, concrete))
            .collect(),
        for_ty: subst_type_param_in_typeref(&id.for_ty, param, concrete),
        methods: id
            .methods
            .iter()
            .map(|m| {
                // Shadowing: if the method declares its own type parameter named `param`, that inner
                // scope shadows the rule parameter — do not substitute inside this method (hygiene).
                if m.sig.param_names().iter().any(|n| n == param) {
                    m.clone()
                } else {
                    FnDecl {
                        vis: m.vis,
                        thaw: m.thaw,
                        tier: m.tier,
                        sig: subst_type_param_in_sig(&m.sig, param, concrete),
                        body: subst_type_param_in_expr(&m.body, param, concrete),
                    }
                }
            })
            .collect(),
    }
}

/// Validate a `derive Name for T` use-site application (DN-54 §4 / M-812 / DN-38 §8.1).
///
/// **Guarantee: `Declared`** (name-resolution + target-type + arity checks; the RHS instantiation
/// + elaboration run at elaboration time, `Empirical` — VR-5).
///
/// Checks (all never-silent, G2):
/// - **Rule resolution**: `Name` must resolve to a registered `lower` rule in `lower_rules`
///   (earlier declarations in the same Pass 3e loop are already registered).
/// - **Target type well-formedness**: `for_ty` must resolve to a real type (DN-54 §4).
fn check_derive_application(
    dd: &DeriveDecl,
    lower_rules: &BTreeMap<String, LowerDecl>,
    types: &BTreeMap<String, DataInfo>,
) -> Result<(), CheckError> {
    if !lower_rules.contains_key(&dd.name) {
        return Err(CheckError::new(
            &dd.name,
            format!(
                "`derive {}` references an unknown generative-lowering rule — declare it first \
                 with `lower {} […] = <rhs>` (DN-54 §4 / M-812; rule must be visible before use, \
                 G2). Did you spell the rule name correctly?",
                dd.name, dd.name
            ),
        ));
    }
    // The target type must be a real, well-formed type (a `derive Foo for NoSuchType` is a
    // never-silent refusal — G2). No type-vars are in scope at a `derive` use site (it names a
    // concrete target).
    resolve_ty(&dd.name, types, &[], &dd.for_ty)?;
    Ok(())
}

/// **§4.2 cross-rule acyclicity** (DN-54 §4.2 / M-812-cont). Reject a `lower` rule set in which the
/// rules' RHSs reference one another in a cycle (mutual recursion among rules), which would diverge
/// the elaboration pipeline (the lowering passes must stay a DAG — RFC-0006/DN-38 §2). A rule's
/// edges are the names it references in its RHS **that are themselves rule names** (a conservative
/// superset over the single-segment paths in the RHS — the same edge set [`crate::elab`] would
/// expand). A strongly-connected component of size > 1, or a direct self-reference, is a cycle and
/// is refused with a never-silent diagnostic naming a rule on the cycle (G2). Guarantee: `Declared`
/// (a structural acyclicity check, not a theorem).
fn check_lower_rule_acyclicity(
    lower_rules: &BTreeMap<String, LowerDecl>,
    types: &BTreeMap<String, DataInfo>,
    fns: &BTreeMap<String, FnDecl>,
) -> Result<(), CheckError> {
    // A single-segment path is a **rule-reference edge** only if it names a `lower` rule AND does not
    // *also* resolve as a constructor or a top-level fn. Without this guard a rule that shares a name
    // with a ctor/fn used in another rule's RHS would manufacture a spurious edge → a false-positive
    // cycle that rejects a *valid* program (a niche naming coincidence). A ctor/fn occurrence is an
    // ordinary value reference, never a rule expansion, so it is not an edge. (Safe both ways: this
    // *narrows* the edge set to true rule-refs — it never drops a real rule→rule edge, since a name
    // that is a genuine rule reference is, by §4.1 RHS type-check, not a ctor/fn of the same spelling.)
    let is_ctor = |name: &str| -> bool {
        types
            .values()
            .any(|d| d.ctors.iter().any(|c| c.name == name))
    };
    let is_rule_edge = |name: &str| -> bool {
        lower_rules.contains_key(name) && !fns.contains_key(name) && !is_ctor(name)
    };
    // Edges: rule → the set of *other rule names* its RHS references (single-segment paths). Sorted
    // (BTreeSet) for a deterministic cycle report. `walk_expr` carries its own explicit
    // recursion-depth budget (M-674); a pathologically-nested RHS refuses cleanly here rather than
    // overflowing the host stack.
    let edges: BTreeMap<&str, BTreeSet<String>> = lower_rules
        .iter()
        .map(|(name, ld)| {
            let mut refs = BTreeSet::new();
            let mut collect = |x: &Expr| {
                if let Expr::Path(p) = x {
                    if p.0.len() == 1 && is_rule_edge(&p.0[0]) {
                        refs.insert(p.0[0].clone());
                    }
                }
            };
            // A rule-reference edge can hide in an expression RHS *or* in any method body of an
            // item-shaped RHS (DN-54 §10 Model A, M-973); walk whichever form the rule carries.
            match &ld.rhs {
                crate::ast::LowerRhs::Expr(e) => crate::totality::walk_expr(e, &mut collect)
                    .map_err(|err| CheckError::new(name, err.to_string()))?,
                crate::ast::LowerRhs::Impl(id) => {
                    for m in &id.methods {
                        crate::totality::walk_expr(&m.body, &mut collect)
                            .map_err(|err| CheckError::new(name, err.to_string()))?;
                    }
                }
            }
            Ok::<_, CheckError>((name.as_str(), refs))
        })
        .collect::<Result<_, _>>()?;

    // A direct self-reference is the trivial cycle.
    for (name, refs) in &edges {
        if refs.contains(*name) {
            return Err(CheckError::new(
                name,
                format!(
                    "`lower {name}` references itself — a `lower` rule may not be (mutually) \
                     recursive (DN-54 §4.2: the lowering-rule graph must be acyclic so `derive` \
                     terminates; use `Fix` for user-level recursion *inside* the RHS instead). \
                     Never silent (G2)."
                ),
            ));
        }
    }

    // Iterative DFS cycle detection (color marking) — no host-stack recursion (A4-02 discipline:
    // the rule graph is small, but we stay iterative so a pathological set cannot overflow). A grey
    // node re-encountered on the current path closes a cycle.
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Grey,
        Black,
    }
    let mut color: BTreeMap<&str, Color> = edges.keys().map(|k| (*k, Color::White)).collect();
    // The DFS frame: the node and an iterator position into its (sorted) successors.
    for &root in edges.keys() {
        if color[root] != Color::White {
            continue;
        }
        // Stack of (node, successor-index).
        let mut stack: Vec<(&str, usize)> = vec![(root, 0)];
        color.insert(root, Color::Grey);
        while let Some(&(node, idx)) = stack.last() {
            let succs: Vec<&str> = edges[node].iter().map(String::as_str).collect();
            if idx < succs.len() {
                stack.last_mut().expect("non-empty").1 += 1;
                let next = succs[idx];
                match color[next] {
                    Color::White => {
                        color.insert(next, Color::Grey);
                        stack.push((next, 0));
                    }
                    Color::Grey => {
                        // `next` is on the current DFS path ⇒ a cycle through it.
                        return Err(CheckError::new(
                            next,
                            format!(
                                "`lower` rules form a cycle through `{next}` — mutually-recursive \
                                 lowering rules are refused (DN-54 §4.2: the lowering-rule graph \
                                 must be acyclic so `derive` terminates). Break the cycle, or use \
                                 `Fix` for user-level recursion *inside* a rule's RHS. Never \
                                 silent (G2)."
                            ),
                        ));
                    }
                    Color::Black => {}
                }
            } else {
                color.insert(node, Color::Black);
                stack.pop();
            }
        }
    }
    Ok(())
}

/// **Effect-coverage pass** (RFC-0014 §3.4/§4.5 I3; M-660 — guarantee: `Declared`). For every
/// top-level function **and every `impl`-method body**, the effects it **performs** must be a subset
/// of the effects it **declares**. The *performed* set is the union, over every call in the body, of
/// the declared effects of the callee — a **known top-level fn** `g` (in `fns`) OR an unqualified
/// **trait-method** call (the declaring trait method's effects, from `traits`; an ambiguous method
/// name was already refused at the call site — M-659, so the name maps to one trait in any program
/// reaching here). This is the v0 "manual-declare + compositional-check" line (RFC-0014 §8): the
/// checker *composes* declared effects up the call graph as a **check**, never *infers* an undeclared
/// one. (In M-660 the only effect sources are these declarations: `wild`-sourced effects arrive with
/// M-661, and the runtime budget ledger is the separate M-353 concern.)
///
/// **Under-declaration** — performing an effect not declared — is an explicit [`CheckError`] naming
/// both the missing effect and the callee that introduces it (G2/RFC-0014 I3 "no undeclared
/// effects"). **Over-declaration** is allowed: declaring an effect the body never performs is a
/// *contract*, not an error (RFC-0014 I5 default-tightly-scoped). Checking `impl`-method bodies too
/// (their declared set == the trait method's, by conformance) is what keeps an effect from being
/// **hidden** from a caller — the core RFC-0014 invariant that "an effect a function performs is
/// visible in its signature".
fn check_effect_coverage(
    fns: &BTreeMap<String, FnDecl>,
    own_fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    nodule: &Nodule,
) -> Result<(), CheckError> {
    // Only this nodule's **own** fn bodies are coverage-checked (M-662); the callee effect lookups use
    // the merged `fns` (so an own fn calling an imported `pub` fn sees that callee's declared effects).
    for fd in own_fns.values() {
        check_body_effect_coverage(fns, traits, &fd.sig.name, &fd.sig.effects, &fd.body)?;
    }
    // `impl`-method bodies perform effects too (the RFC-0019 × RFC-0014 surface). Their declared set
    // equals the trait method's (conformance), so a body performing more than that is an undeclared
    // effect that must be refused — otherwise a trait-method/impl effect would be hidden from callers.
    for item in &nodule.items {
        if let Item::Impl(id) = item {
            for m in &id.methods {
                check_body_effect_coverage(fns, traits, &m.sig.name, &m.sig.effects, &m.body)?;
            }
        }
    }
    Ok(())
}

/// What introduced a performed effect, for the coverage diagnostic. `Ord` so the `(effect, source)`
/// set is deterministic (a stable first-miss). A **`Call`** is a top-level fn or trait-method call
/// (M-660); **`Wild`** is the `wild` FFI floor (M-661 — `wild` performs `ffi`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum EffectSource {
    /// A call to a named callee (top-level fn or unqualified trait method) — M-660.
    Call(String),
    /// A `wild` block — the FFI floor; it performs the `ffi` effect (M-661).
    Wild,
}

/// One body's effect coverage: `declared ⊇ performed`, where *performed* is the union of each
/// callee's declared effects — a known top-level fn (`fns`), else an unqualified trait method
/// (`traits`, by name) — **plus the `ffi` effect contributed by any `wild` block** (M-661). Owned-
/// `String` sets keep lifetimes simple; the structural walk is shared with totality (one traversal,
/// no bespoke depth-guarded recursion). Deterministic order ⇒ a stable first-miss diagnostic. The
/// M-353 runtime budget ledger is a separate concern (not consulted here).
fn check_body_effect_coverage(
    fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    name: &str,
    declared_effs: &[String],
    body: &Expr,
) -> Result<(), CheckError> {
    let declared: std::collections::BTreeSet<String> = declared_effs.iter().cloned().collect();
    // Each performed effect is recorded with its **source** so the diagnostic can name it: a
    // `Source::Call(callee)` (a top-level fn or trait-method call — M-660) or the `Source::Wild`
    // FFI floor (M-661). The set is `(effect, source)` for a deterministic, de-duplicated first-miss.
    let mut performed: std::collections::BTreeSet<(String, EffectSource)> =
        std::collections::BTreeSet::new();
    crate::totality::walk_expr(body, &mut |x| {
        match x {
            Expr::App { head, .. } => {
                if let Expr::Path(p) = head.as_ref() {
                    if p.0.len() == 1 {
                        let callee = &p.0[0];
                        if let Some(g) = fns.get(callee) {
                            for eff in &g.sig.effects {
                                performed.insert((eff.clone(), EffectSource::Call(callee.clone())));
                            }
                        } else {
                            // Not a top-level fn ⇒ an unqualified trait-method call: it performs the
                            // declaring trait method's declared effects (the contract).
                            for tr in traits.values() {
                                for s in &tr.sigs {
                                    if &s.name == callee {
                                        for eff in &s.effects {
                                            performed.insert((
                                                eff.clone(),
                                                EffectSource::Call(callee.clone()),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // `wild` is the `ffi` effect **source** (M-661; RFC-0016 §8-Q6 binding to M-660): a fn
            // whose body contains a `wild` block *performs* `ffi`, so it must declare `!{ffi}` — else
            // the coverage check refuses it exactly as for a call-sourced undeclared effect (G2). The
            // `@std-sys` context gate is the typechecker's separate concern ([`Cx::check_wild`]); here
            // `wild` only contributes its effect. (A `wild` that reached this pass already passed the
            // context gate, since coverage runs after bodies type-check.)
            Expr::Wild(_) => {
                performed.insert(("ffi".to_owned(), EffectSource::Wild));
            }
            _ => {}
        }
    })
    .map_err(|e| CheckError::new(name, e.to_string()))?;
    for (eff, source) in &performed {
        if !declared.contains(eff) {
            let via = match source {
                EffectSource::Call(callee) => format!("via calling `{callee}`"),
                EffectSource::Wild => "via a `wild` block (the FFI floor — M-661)".to_owned(),
            };
            return Err(CheckError::new(
                name,
                format!(
                    "`{name}` performs effect `{eff}` ({via}) but does not declare it — add it to \
                     the `!{{…}}` effect annotation (RFC-0014 §4.5 I3: no undeclared effects; never \
                     silent — G2)"
                ),
            ));
        }
    }
    Ok(())
}

/// Render an effect set for a diagnostic in the surface `!{a, b}` form (`!{}` for the empty/pure
/// set). Used only for never-silent error messages (M-660); preserves the written source order.
fn render_effects(effects: &[String]) -> String {
    format!("!{{{}}}", effects.join(", "))
}

pub(crate) fn resolve_ctors(
    types: &BTreeMap<String, DataInfo>,
    td: &TypeDecl,
) -> Result<Vec<CtorInfo>, CheckError> {
    let mut ctors = Vec::new();
    for c in &td.ctors {
        if ctors.iter().any(|x: &CtorInfo| x.name == c.name) {
            return Err(CheckError::new(
                &td.name,
                format!("duplicate constructor `{}`", c.name),
            ));
        }
        let mut fields = Vec::new();
        for f in &c.fields {
            // The declaration's type parameters are in scope, so a field may be an abstract
            // `Ty::Var` (`Cons(A, List<A>)` ⇒ fields `[Var("A"), Data("List", [Var("A")])]`).
            let (ty, _) = resolve_ty(&td.name, types, &td.params, f)?;
            fields.push(ty);
        }
        ctors.push(CtorInfo {
            name: c.name.clone(),
            fields,
        });
    }
    Ok(ctors)
}

/// The first value that appears more than once in `xs` (left to right), if any. Used to reject
/// duplicate type-parameter names — an explicit error, never a silently-shadowed binding (G2).
pub(crate) fn first_duplicate(xs: &[String]) -> Option<&String> {
    let mut seen = std::collections::BTreeSet::new();
    xs.iter().find(|x| !seen.insert((*x).clone()))
}

/// **Trait pass** (RFC-0019 §4.2; LR-2 — guarantee: `Declared`, a structural registry check, not a
/// theorem). Register each `trait Tr<params> { fn … }` as a [`TraitInfo`]: reject a duplicate trait
/// name; reject duplicate type-parameter names and duplicate method names; and check that **every**
/// method signature *resolves* with the trait's params in scope as abstract type-variables (its
/// value-param types and return type via [`resolve_ty`]). A trait is a registry entry, never a
/// kernel node (KC-3). Every refusal is an explicit [`CheckError`] (G2).
pub(crate) fn register_traits(
    types: &BTreeMap<String, DataInfo>,
    nodule: &Nodule,
) -> Result<BTreeMap<String, TraitInfo>, CheckError> {
    let mut traits: BTreeMap<String, TraitInfo> = BTreeMap::new();
    for item in &nodule.items {
        let Item::Trait(td) = item else { continue };
        let site = &td.name;
        if let Some(dup) = first_duplicate(&td.params) {
            return Err(CheckError::new(
                site,
                format!("duplicate type parameter `{dup}` in trait `{}`", td.name),
            ));
        }
        if traits.contains_key(&td.name) {
            return Err(CheckError::new(site, "duplicate trait declaration"));
        }
        // Method names must be distinct within the trait (a duplicated requirement is ambiguous).
        let mut seen_methods: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for s in &td.sigs {
            if !seen_methods.insert(s.name.as_str()) {
                return Err(CheckError::new(
                    site,
                    format!("duplicate method `{}` in trait `{}`", s.name, td.name),
                ));
            }
            // Each method sig must resolve with the trait's params (and the method's own params) in
            // scope as type-variables (RFC-0007 §11.2). A method may carry its own type-params too;
            // bounds on them are validated against the complete trait registry in the pass below (G2).
            let mut tyvars = td.params.clone();
            tyvars.extend(s.param_names());
            check_sig_resolves(types, site, &tyvars, s)?;
        }
        traits.insert(
            td.name.clone(),
            TraitInfo {
                name: td.name.clone(),
                params: td.params.clone(),
                sigs: td.sigs.clone(),
            },
        );
    }
    // The registry is now complete: validate that every trait-method type-parameter BOUND names a
    // KNOWN trait. This is a second pass precisely so a bound may forward-reference a later-declared
    // trait. An unknown bound (`fn f<T: Nope>(…)`) is an explicit error here, never silently
    // registered (G2) — otherwise the ill-formed requirement would surface only at an unrelated
    // later check, or never (if no impl exercises it).
    for tr in traits.values() {
        for s in &tr.sigs {
            for tp in &s.params {
                for b in &tp.bounds {
                    if !traits.contains_key(&b.name) {
                        return Err(CheckError::new(
                            &tr.name,
                            format!(
                                "trait `{}` method `{}`: unknown trait `{}` in the bound `{}: {}` \
                                 (RFC-0019 §4.1 — a method's type-parameter bound must name a \
                                 declared trait)",
                                tr.name, s.name, b.name, tp.name, b.name
                            ),
                        ));
                    }
                }
            }
        }
    }
    Ok(traits)
}

/// Confirm a [`FnSig`]'s value-parameter types and return type all resolve under `tyvars` (the
/// abstract type-variables in scope). Shared by the trait pass (method requirements) and the
/// bounded-fn body checker (signature validation). Does not check the body — only the signature.
fn check_sig_resolves(
    types: &BTreeMap<String, DataInfo>,
    site: &str,
    tyvars: &[String],
    sig: &FnSig,
) -> Result<(), CheckError> {
    for p in &sig.value_params {
        resolve_ty(site, types, tyvars, &p.ty)?;
    }
    resolve_ty(site, types, tyvars, &sig.ret)?;
    Ok(())
}

/// **Impl pass — registration + coherence** (RFC-0019 §4.5; LR-2 — guarantee: `Declared`, the
/// coherence argument is Declared-with-argument per RFC-0019, not machine-checked). For each
/// `impl Trait<args> for T`:
/// - resolve `for_ty` and each `trait_args` to a **concrete** [`Ty`] (no type-variables in scope);
/// - the trait must exist and the argument **arity** must match the trait's params (else explicit);
/// - **global uniqueness:** key `(trait, type_head(for_ty))` must be free, else an overlapping-
///   instance / coherence refusal naming the pair (RFC-0019 §4.5; ADR-003);
/// - **orphan rule (phylum-wide, pub-blind — M-662):** the trait is declared in *some* nodule of the
///   phylum, OR `for_ty`'s head is a `Data` declared in *some* nodule of the phylum, OR `for_ty` is a
///   primitive repr type (`Binary`/`Ternary`/`Dense`/`Substrate`); otherwise an explicit orphan
///   refusal. The locality view is pub-blind (coherence is enforcement authority, not the `pub`
///   namespace — RFC-0019 §4.5). For a phylum-of-one this reduces to the prior single-nodule test.
///
/// Method *bodies* are not checked here (that is [`check_impl_methods`], after the whole instance
/// set is known); method *presence* (exact-match against the trait's sigs) IS checked here.
pub(crate) fn register_instances(
    types: &BTreeMap<String, DataInfo>,
    traits: &BTreeMap<String, TraitInfo>,
    coherence: &CoherenceView,
    nodule: &Nodule,
) -> Result<BTreeMap<(String, String), InstanceInfo>, CheckError> {
    // The orphan-rule locality test is **phylum-wide and pub-blind** (M-662; RFC-0019 §4.5): a trait
    // or `for`-type head is "local" iff *some* nodule of the phylum declares it, regardless of `pub`
    // (coherence is enforcement authority, not the `pub` namespace). The pub-blind `coherence` view
    // holds every nodule's trait/type names. For a phylum-of-one this is exactly this nodule's
    // declarations, so the rule reduces to the pre-M-662 single-nodule test.
    let phylum_traits = &coherence.traits;
    let phylum_types = &coherence.types;

    let mut instances: BTreeMap<(String, String), InstanceInfo> = BTreeMap::new();
    for item in &nodule.items {
        let Item::Impl(id) = item else { continue };
        let site = &id.trait_name;
        // The trait must exist.
        let Some(tr) = traits.get(&id.trait_name) else {
            return Err(CheckError::new(
                site,
                format!(
                    "`impl` for unknown trait `{}` (RFC-0019 §4.5)",
                    id.trait_name
                ),
            ));
        };
        // Resolve the `for` type and the trait arguments concretely (no type-variables in scope —
        // an instance head is a concrete type in stage-1).
        let (for_ty, _) = resolve_ty(site, types, &[], &id.for_ty)?;
        let mut trait_args = Vec::with_capacity(id.trait_args.len());
        for a in &id.trait_args {
            trait_args.push(resolve_ty(site, types, &[], a)?.0);
        }
        // Arity: the trait's params count must equal the written trait-argument count.
        if trait_args.len() != tr.params.len() {
            return Err(CheckError::new(
                site,
                format!(
                    "trait `{}` takes {} type argument(s), but this `impl` supplies {} \
                     (RFC-0019 §4.5)",
                    tr.name,
                    tr.params.len(),
                    trait_args.len()
                ),
            ));
        }
        // Coherence key — the type head (width/shape erased; stage-1 keys per head).
        let Some(head) = type_head(&for_ty) else {
            return Err(CheckError::new(
                site,
                format!(
                    "an `impl … for {for_ty}` over a bare type variable is not a legal instance \
                     head in stage-1 (no blanket instances — RFC-0019 §4.5); the `for` type must \
                     be concrete"
                ),
            ));
        };
        // Orphan rule — **phylum-wide, pub-blind** (RFC-0019 §4.5; M-662): legal iff the trait is
        // declared in *some* nodule of the phylum, OR the `for`-type head is a data type declared in
        // *some* nodule of the phylum, OR a primitive repr type. This generalizes the former
        // nodule-local test to the whole phylum (an impl may be in either the trait's *or* the type's
        // nodule, or any sibling, so long as one head is phylum-local); an impl whose trait **and**
        // type are both outside the phylum still orphan-rejects.
        let trait_local = phylum_traits.contains(id.trait_name.as_str());
        let type_local = match &for_ty {
            Ty::Data(n, _) => phylum_types.contains(n.as_str()),
            // Primitive repr types are "owned by the phylum" for stage-1 (RFC-0019 §4.5) — the
            // RFC-0032 sequence/byte-string reprs and the M-892 VSA repr included.
            Ty::Binary(_)
            | Ty::Ternary(_)
            | Ty::Dense(_, _)
            | Ty::Vsa { .. }
            | Ty::Substrate(_)
            | Ty::Seq(_, _)
            | Ty::Bytes
            | Ty::Float => true,
            // `Ty::Var` and `Ty::Fn` are not legal instance heads; type_head() returns None for them,
            // so this arm is unreachable in practice (the coherence key check rejects them upstream).
            // Kept for exhaustiveness — never a silent accept (G2).
            Ty::Var(_) | Ty::Fn(_, _) => false,
        };
        if !trait_local && !type_local {
            return Err(CheckError::new(
                site,
                format!(
                    "orphan instance: `impl {} for {for_ty}` — neither the trait `{}` nor the type \
                     `{for_ty}` is declared in any nodule of this phylum (RFC-0019 §4.5 orphan rule, \
                     phylum-wide; M-662). Declare one of them in the phylum, or move the impl.",
                    id.trait_name, id.trait_name
                ),
            ));
        }
        // Global uniqueness — at most one instance per `(trait, head)` (RFC-0019 §4.5; ADR-003). A
        // duplicate (even at a different width on the same head — the documented stage-1 over-rejection)
        // is an explicit coherence refusal, never a silent shadow (G2).
        let key = (id.trait_name.clone(), head.clone());
        if instances.contains_key(&key) {
            return Err(CheckError::new(
                site,
                format!(
                    "overlapping instance — coherence/global-uniqueness violation (RFC-0019 §4.5): \
                     a second `impl {} for` a `{head}` type. Stage-1 keys per (trait, type-head), so \
                     two instances on the same head (even at different widths) conflict.",
                    id.trait_name
                ),
            ));
        }
        // Method presence: the impl's method set must EXACTLY match the trait's required sigs.
        check_impl_method_set(tr, id)?;
        instances.insert(
            key,
            InstanceInfo {
                trait_name: id.trait_name.clone(),
                trait_args,
                for_ty,
                methods: id.methods.iter().map(|m| m.sig.name.clone()).collect(),
            },
        );
    }
    Ok(instances)
}

/// The impl's method **set** must match the trait's requirement set **exactly** (RFC-0019 §4.5):
/// a missing method and an extra method are both explicit refusals (never silently filled or
/// dropped — G2). Signature/body agreement is [`check_impl_methods`]; this is only presence/names.
fn check_impl_method_set(tr: &TraitInfo, id: &ImplDecl) -> Result<(), CheckError> {
    let site = &id.trait_name;
    let required: std::collections::BTreeSet<&str> =
        tr.sigs.iter().map(|s| s.name.as_str()).collect();
    let provided: std::collections::BTreeSet<&str> =
        id.methods.iter().map(|m| m.sig.name.as_str()).collect();
    for need in &required {
        if !provided.contains(need) {
            return Err(CheckError::new(
                site,
                format!("impl of `{}` is missing method `{need}`", tr.name),
            ));
        }
    }
    for have in &provided {
        if !required.contains(have) {
            return Err(CheckError::new(
                site,
                format!(
                    "impl of `{}` has method `{have}` not in trait `{}`",
                    tr.name, tr.name
                ),
            ));
        }
    }
    // A method must not be provided twice (ambiguous dictionary slot).
    if let Some(dup) = first_duplicate(
        &id.methods
            .iter()
            .map(|m| m.sig.name.clone())
            .collect::<Vec<_>>(),
    ) {
        return Err(CheckError::new(
            site,
            format!(
                "impl of `{}` provides method `{dup}` more than once",
                tr.name
            ),
        ));
    }
    Ok(())
}

/// Check one `impl`'s method bodies against their **expected** signatures (RFC-0019 §4.5; guarantee:
/// `Declared` — a structural check). The expected signature of method `m` is the trait's sig for `m`
/// with the trait's params substituted by this impl's `trait_args`; the method body is checked the
/// normal fn-body way against those substituted value-param/return types. A signature mismatch
/// (wrong param types or arity, wrong return) is an explicit refusal, never silently accepted.
fn check_impl_methods(
    types: &BTreeMap<String, DataInfo>,
    fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    instances: &BTreeMap<(String, String), InstanceInfo>,
    imports: &NoduleImports,
    std_sys: bool,
    id: &ImplDecl,
) -> Result<Vec<FnDecl>, CheckError> {
    let tr = traits
        .get(&id.trait_name)
        .expect("instance registration checked the trait exists");
    // The resolved method bodies (ambient literals + ctor/binder patterns normalized — the canonical
    // checked form), returned so the late guarantee-grading pass (Pass 3d) walks the *same* canonical
    // AST as a top-level fn rather than the raw registered body (M-663 / Copilot review — grading must
    // not re-derive the ctor/binder ambiguity from a global scan).
    let mut resolved: Vec<FnDecl> = Vec::with_capacity(id.methods.len());
    // The trait-arg substitution `trait.params ↦ impl.trait_args` (resolved concretely).
    let (for_ty, _) = resolve_ty(&id.trait_name, types, &[], &id.for_ty)?;
    let mut trait_args = Vec::with_capacity(id.trait_args.len());
    for a in &id.trait_args {
        trait_args.push(resolve_ty(&id.trait_name, types, &[], a)?.0);
    }
    let s = param_subst(&tr.params, &trait_args);
    for method in &id.methods {
        let req = tr
            .sigs
            .iter()
            .find(|x| x.name == method.sig.name)
            .expect("method-set match checked at registration");
        let site = &method.sig.name;
        // Arity must match the requirement.
        if method.sig.value_params.len() != req.value_params.len() {
            return Err(CheckError::new(
                site,
                format!(
                    "impl method `{}` of `{}` takes {} parameter(s), but the trait requires {}",
                    method.sig.name,
                    tr.name,
                    method.sig.value_params.len(),
                    req.value_params.len()
                ),
            ));
        }
        // Each declared parameter type must equal the (substituted) required type; likewise the
        // return type. The method's own value-param types are resolved concretely (the impl is over a
        // concrete `for_ty`, so the method carries no abstract type-variables in stage-1).
        for (mp, rp) in method.sig.value_params.iter().zip(&req.value_params) {
            let (got, _) = resolve_ty(site, types, &[], &mp.ty)?;
            let want = subst_ty(&resolve_ty(site, types, &tr.params, &rp.ty)?.0, &s);
            if got != want {
                return Err(CheckError::new(
                    site,
                    format!(
                        "impl method `{}` parameter `{}`: {}",
                        method.sig.name,
                        mp.name,
                        edge_mismatch("type", &want, &got)
                    ),
                ));
            }
        }
        let (got_ret, _) = resolve_ty(site, types, &[], &method.sig.ret)?;
        let want_ret = subst_ty(&resolve_ty(site, types, &tr.params, &req.ret)?.0, &s);
        if got_ret != want_ret {
            return Err(CheckError::new(
                site,
                format!(
                    "impl method `{}` return: {}",
                    method.sig.name,
                    edge_mismatch("type", &want_ret, &got_ret)
                ),
            ));
        }
        // Effect conformance (RFC-0014 §4.5 I3; M-660): an impl method's **declared effect set must
        // equal the trait method's** (exact match in stage-1 — an unannotated trait method ⇒ the
        // impl method must also be unannotated/pure). The effect annotation is part of the
        // signature contract, so a divergence is an explicit refusal, never a silent widen/narrow
        // (G2). Set equality (order-insensitive); duplicates within one annotation were already
        // refused at parse time.
        let req_effects: std::collections::BTreeSet<&str> =
            req.effects.iter().map(String::as_str).collect();
        let got_effects: std::collections::BTreeSet<&str> =
            method.sig.effects.iter().map(String::as_str).collect();
        if req_effects != got_effects {
            return Err(CheckError::new(
                site,
                format!(
                    "impl method `{}` declares effects {} but trait `{}` requires {} — an impl \
                     method's effect annotation must match the trait method's exactly (RFC-0014 \
                     §4.5 I3; never silently widened or narrowed — G2)",
                    method.sig.name,
                    render_effects(&method.sig.effects),
                    tr.name,
                    render_effects(&req.effects),
                ),
            ));
        }
        // The body is checked the normal fn-body way (against the method's own — now validated —
        // signature). `for_ty` is concrete, so the body has no abstract type-variables; the full
        // trait/instance context is available so the body may itself call trait methods. The
        // `@std-sys` context (M-661) flows in so a `wild` block inside an impl method is gated
        // exactly as in a top-level fn (an impl in a non-`@std-sys` nodule may not contain `wild`).
        let (body, _ret) = check_fn_body(types, fns, traits, instances, imports, std_sys, method)?;
        resolved.push(FnDecl {
            vis: method.vis,
            thaw: method.thaw,
            tier: method.tier,
            sig: method.sig.clone(),
            body,
        });
    }
    let _ = for_ty; // resolved above for the arg substitution; head reuse is at registration.
    Ok(resolved)
}

/// Check a function (or impl method) body against its declared signature (RFC-0007 §11; RFC-0019
/// §4.1). Validates the type-parameter bounds, builds the `tyvars`/`bounds` scopes, resolves the
/// value-param + return types, and runs the bidirectional [`Cx::check`]. Returns the **resolved**
/// body (ambient bare-decimals filled) and the resolved return type. Shared by Pass 3 (top-level
/// fns) and [`check_impl_methods`] (impl methods) — DRY.
fn check_fn_body(
    types: &BTreeMap<String, DataInfo>,
    fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    instances: &BTreeMap<(String, String), InstanceInfo>,
    imports: &NoduleImports,
    std_sys: bool,
    fd: &FnDecl,
) -> Result<(Expr, Ty), CheckError> {
    let site = &fd.sig.name;
    let tyvars = fd.sig.param_names();
    // Validate every bound names a real trait with the right argument arity (RFC-0019 §4.1). The
    // bound's trait-args may reference the fn's own type-variables (`T: Cmp<T>`), so resolve them
    // with `tyvars` in scope. A bound on an unknown trait / wrong arity is an explicit refusal.
    let bounds = check_bounds(types, traits, site, &tyvars, &fd.sig.params)?;
    let mut scope: Vec<(String, Ty)> = Vec::new();
    for p in &fd.sig.value_params {
        let (ty, _) = resolve_ty(site, types, &tyvars, &p.ty)?;
        scope.push((p.name.clone(), ty));
    }
    let (ret, _) = resolve_ty(site, types, &tyvars, &fd.sig.ret)?;
    let cx = Cx {
        site,
        types,
        fns,
        traits,
        instances,
        imports,
        tyvars: &tyvars,
        bounds: &bounds,
        std_sys,
        depth: Cell::new(0),
        // The one whole-function-body walk — seed the active affine tracker from the parameter
        // scope so a `Substrate`-typed parameter is already tracked as the body starts (M-903;
        // DN-71 §4.2: a parameter pass counts as the caller's move-in).
        affine: Tracker::seeded(&scope),
    };
    let (got, body) = cx.check(&mut scope, &fd.body, Some(&ret))?;
    if got != ret {
        return Err(CheckError::new(site, edge_mismatch("body", &ret, &got)));
    }
    Ok((body, ret))
}

/// Validate a function/method's type-parameter **bounds** (RFC-0019 §4.1): each bound must name a
/// registered trait with the correct type-argument arity, and the bound's trait-args must resolve
/// under `tyvars`. Returns the `(param, bounds)` pairs (only the bounded params) for the checking
/// context. Every refusal is explicit (G2). The dictionary the bound stands for is staged to
/// elaboration (RFC-0007 §12.3 / M-673) — the checker only validates satisfiability ("typing").
fn check_bounds(
    types: &BTreeMap<String, DataInfo>,
    traits: &BTreeMap<String, TraitInfo>,
    site: &str,
    tyvars: &[String],
    params: &[crate::ast::TypeParam],
) -> Result<Vec<(String, Vec<TraitRef>)>, CheckError> {
    let mut bounds: Vec<(String, Vec<TraitRef>)> = Vec::new();
    for p in params {
        for b in &p.bounds {
            let Some(tr) = traits.get(&b.name) else {
                return Err(CheckError::new(
                    site,
                    format!(
                        "bound `{}: {}` names unknown trait `{}` (RFC-0019 §4.1)",
                        p.name, b.name, b.name
                    ),
                ));
            };
            // Arity: written args must match the trait's params — **except** the canonical
            // elided-self form `T: Cmp` on a single-parameter trait, which is sugar for `T: Cmp<T>`
            // (Rust/Haskell `T: Cmp` ⇒ `Cmp T`). That elision is *only* valid for a single-param
            // trait with zero written args; any other count mismatch is an explicit refusal (G2).
            let elided_self = b.args.is_empty() && tr.params.len() == 1;
            if !elided_self && b.args.len() != tr.params.len() {
                return Err(CheckError::new(
                    site,
                    format!(
                        "bound `{}: {}<…>` supplies {} type argument(s), but trait `{}` takes {} \
                         (write `{}: {}` for the single-parameter self-bound, or supply all args)",
                        p.name,
                        b.name,
                        b.args.len(),
                        b.name,
                        tr.params.len(),
                        p.name,
                        b.name
                    ),
                ));
            }
            for a in &b.args {
                resolve_ty(site, types, tyvars, a)?;
            }
        }
        if !p.bounds.is_empty() {
            bounds.push((p.name.clone(), p.bounds.clone()));
        }
    }
    Ok(bounds)
}

/// The checking context for one function body.
struct Cx<'a> {
    site: &'a str,
    types: &'a BTreeMap<String, DataInfo>,
    fns: &'a BTreeMap<String, FnDecl>,
    /// Trait registry (RFC-0019 §4.2) — for resolving bounded-generic-call satisfiability and
    /// unqualified trait-method calls (`Tr::m`). Empty in re-inference (`infer_type`).
    traits: &'a BTreeMap<String, TraitInfo>,
    /// Instance registry (RFC-0019 §4.5), keyed by `(trait, type-head)` — the coherence map a
    /// bounded call / trait-method call resolves against. Empty in re-inference.
    instances: &'a BTreeMap<(String, String), InstanceInfo>,
    /// The nodule's resolved cross-nodule imports (M-662) — consulted **only** at unresolved-name
    /// sites to raise the never-silent glob-vs-glob ambiguity error (a *reference* to a name brought
    /// in by ≥2 globs and not shadowed). Imported `pub` decls themselves are already merged into
    /// `types`/`fns`/`traits` (by simple name), so ordinary resolution sees them directly; this field
    /// only carries the `ambiguous` set so a reference to an ambiguous name is refused, never a silent
    /// winner (G2). Empty (`ambiguous` empty) in re-inference and in a phylum-of-one.
    imports: &'a NoduleImports,
    /// The type parameters in scope for this body (RFC-0007 §11.2) — empty for a monomorphic
    /// function. A bare `Named` type that matches one of these resolves to [`Ty::Var`].
    tyvars: &'a [String],
    /// The **bounds** on the type parameters in scope (`T: Cmp` ⇒ `("T", TraitRef{Cmp})`), so an
    /// unqualified trait-method call inside a bounded body can be typed through a bound (the
    /// dictionary it stands for is staged to elaboration — RFC-0007 §12.3 / M-673). Parallel to
    /// `tyvars`; empty for an unbounded/monomorphic body.
    bounds: &'a [(String, Vec<TraitRef>)],
    /// Whether the enclosing nodule carries the `@std-sys` FFI-floor marker (M-661; RFC-0016 §8-Q6).
    /// A `wild` block (the denied-by-default unsafe escape, LR-9/S6) type-checks **only** when this is
    /// `true`; in a non-`@std-sys` nodule a `wild` is a hard [`CheckError`] (never a silent escape —
    /// G2). Threaded down from the nodule header through [`check_fn_body`] / [`check_impl_methods`].
    std_sys: bool,
    /// Live expression-nesting depth for the explicit [`MAX_CHECK_DEPTH`] budget (interior
    /// mutability so [`Self::check`] stays `&self`). Reset per body; accounted by [`DepthGuard`].
    depth: Cell<u32>,
    /// The **affine `Substrate` use-once tracker** (M-903; DN-71 §4.2) — mirrors `scope` by index
    /// (interior mutability, matching `depth`'s pattern, so [`Self::check`] stays `&self`). **Active**
    /// (seeded via [`Tracker::seeded`]) in both [`check_fn_body`] (seeded from the parameter scope)
    /// **and** [`check_lower_rule_rhs_type`] (seeded from the empty initial scope — M-919 fix, DN-71
    /// Model S §4.2: a `lower` rule's RHS can still bind a `Substrate` via a helper-fn call, so it is
    /// use-once-checked too); **inert** (a guaranteed no-op) only in the `infer_type` context — see
    /// `crate::affine`'s module docs for why. (FLAG-6 fix, M-973: this doc previously read that the
    /// tracker was inert in `check_lower_rule_rhs_type`, stale since M-919 made it active there.)
    affine: Tracker,
}

/// RFC-0040 (M-977): if `expected` is a **cons-list-shaped** user ADT, return its `(nil, cons)`
/// constructor names — the trigger for desugaring a `[…]` literal into a right-nested `Cons` chain
/// (see [`Cx::check_list`]). "Cons-list-shaped" is a purely *structural* recognition, requiring no
/// annotation: the type is a `Ty::Data(name, _)` whose declaration has **exactly two** constructors —
/// one **nullary** (the "nil": `Nil`/`GLNil`/`TNil`/`BNil`/…) and one **binary** whose SECOND field is
/// the recursive `Self` type `Data(name, …)` (the "cons": `Cons(A, Self)`). This matches every
/// `lib/std` list type (`Vec`, `Trits`, `ByteList`, `GRowList`, …) uniformly. A type not of this shape
/// yields `None`, so the `Seq{T,N}` and no-context paths are untouched (never a silent reinterpret).
fn cons_list_ctors(
    types: &std::collections::BTreeMap<String, DataInfo>,
    expected: &Ty,
) -> Option<(String, String)> {
    let Ty::Data(name, _args) = expected else {
        return None;
    };
    let di = types.get(name)?;
    if di.ctors.len() != 2 {
        return None;
    }
    let mut nil: Option<String> = None;
    let mut cons: Option<String> = None;
    for c in &di.ctors {
        match c.fields.len() {
            0 => nil = Some(c.name.clone()),
            2 => {
                // The cons ctor's SECOND field must be the recursive Self reference `Data(name, …)`
                // at the type's own arity — that is what makes this a linked list, not a pair.
                if let Ty::Data(fname, fargs) = &c.fields[1] {
                    if fname == name && fargs.len() == di.params.len() {
                        cons = Some(c.name.clone());
                    }
                }
            }
            _ => {}
        }
    }
    match (nil, cons) {
        (Some(n), Some(c)) => Some((n, c)),
        _ => None,
    }
}

impl Cx<'_> {
    fn err<T>(&self, msg: impl Into<String>) -> Result<T, CheckError> {
        Err(CheckError::new(self.site, msg))
    }

    /// Map a match-analysis over-budget refusal (usefulness / decision-tree, RFC-0041 §4.7) into the
    /// checker's never-silent [`CheckError`] surface — an explicit refusal, never a host-stack
    /// overflow (G2/VR-5). Shared by the exhaustiveness/redundancy (`usefulness::useful`) and the
    /// decision-tree (`decision::compile`) callsites so the mapping is written once (DRY).
    fn match_budget_err(&self, e: mycelium_workstack::BudgetError) -> CheckError {
        CheckError::new(
            self.site,
            format!(
                "match analysis exceeded the recursion budget: {e} — an explicit over-budget refusal \
                 (RFC-0041 §4.7), never a host-stack overflow (G2/VR-5)"
            ),
        )
    }

    /// Enter one level of `check` recursion against the explicit [`MAX_CHECK_DEPTH`] budget
    /// (banked guard 4): charge a level, refuse with a clean [`CheckError`] past the budget (never a
    /// host-stack overflow), and return a [`DepthGuard`] that releases the level on **any** exit path.
    fn enter(&self) -> Result<DepthGuard<'_>, CheckError> {
        let d = self.depth.get() + 1;
        if d > MAX_CHECK_DEPTH {
            return self.err(format!(
                "expression nesting exceeds the checker depth budget ({MAX_CHECK_DEPTH}) — an \
                 explicit budget (banked guard 4), refused cleanly rather than overflowing the \
                 host stack (RFC-0007 §4.6 clocked-recursion discipline)"
            ));
        }
        self.depth.set(d);
        Ok(DepthGuard(&self.depth))
    }

    fn ctor(&self, name: &str) -> Option<(&DataInfo, usize)> {
        self.types
            .values()
            .find_map(|d| d.ctors.iter().position(|c| c.name == name).map(|i| (d, i)))
    }

    /// Infer the type of `e` under `scope` (a lexical stack; shadowing = later wins). A thin wrapper
    /// over the bidirectional [`Self::check`] with no expected type — used where only the type is
    /// wanted and `e` carries no ambient bare-decimal needing context (e.g. the elaborator's
    /// re-inference over already-resolved terms).
    fn infer(&self, scope: &mut Vec<(String, Ty)>, e: &Expr) -> Result<Ty, CheckError> {
        self.check(scope, e, None).map(|(ty, _)| ty)
    }

    /// **Bidirectional check** (RFC-0012 §4.3): type `e` under `scope`, optionally *against* an
    /// `expected` type, and return the type together with a **resolved** expression — any ambient
    /// bare decimal ([`Literal::AmbientInt`]) whose width the context determines is rewritten to a
    /// concrete `Binary`/`Ternary` literal, so the evaluator and elaborator see only explicit forms.
    /// Where the width is *not* determined, the refusal is an explicit `UnresolvedWidth` (never a
    /// built-in default). A cross-paradigm edge surfaces as a `MissingConversion` (never silent).
    fn check(
        &self,
        scope: &mut Vec<(String, Ty)>,
        e: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // Charge one nesting level against the explicit depth budget; released on every exit path.
        // This is what bounds checker recursion — not the host stack (RFC-0007 §4.6; A4-02).
        let _depth = self.enter()?;
        match e {
            Expr::Lit(Literal::AmbientInt(p, v)) => {
                let lit = self.resolve_ambient_int(*p, *v, expected)?;
                let ty = lit_ty_of(self.site, &lit)?;
                Ok((ty, Expr::Lit(lit)))
            }
            // RFC-0032 D3 (M-749): a list literal `[e1, …]` constructs a `Seq{T, N}`. Element type is
            // **inferred from the (homogeneous) elements**; `N` is the count. Checked here (not in the
            // context-free `lit_ty_of`) because the elements are expressions needing the checking
            // context. Never-silent (G2): a heterogeneous element, or a mismatch against an expected
            // `Seq{T, N}`, is an explicit `CheckError`.
            Expr::Lit(Literal::List(elems)) => self.check_list(scope, elems, expected),
            Expr::Lit(l) => Ok((self.lit_ty(l)?, Expr::Lit(l.clone()))),
            Expr::Path(p) => self.check_path(scope, p, e, expected),
            // The heavy, allocation-bearing arms are separate methods so this dispatch frame stays
            // small — a deep but call-light nest (e.g. `not(not(…))`) must fit the host stack the
            // parser's depth bound allows, in debug builds too (A4-02).
            Expr::Let {
                name,
                ty,
                bound,
                body,
            } => self.check_let(scope, name, ty.as_ref(), bound, body, expected),
            Expr::If { cond, conseq, alt } => self.check_if(scope, cond, conseq, alt, expected),
            Expr::Match { scrutinee, arms } => self.check_match(scope, scrutinee, arms, expected),
            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => self.check_for(scope, x, xs, acc, init, body, expected),
            Expr::Swap {
                value,
                target,
                policy,
            } => self.check_swap(scope, value, target, policy),
            Expr::Wild(body) => self.check_wild(body, expected),
            Expr::Spore(_) => {
                self.err("`spore` is deferred to the reconstruction-manifest work (E2-5/M-260)")
            }
            // DN-03 §1 / M-664: `consume <expr>` — affine acquisition of a `Substrate` value (LR-8).
            Expr::Consume(operand) => self.check_consume(scope, operand, expected),
            Expr::Colony(hyphae) => self.check_colony(scope, hyphae, expected),
            // RFC-0024 §4A (M-704): a `lambda(p: A) => body` checks to `Ty::Fn(A, B)` where
            // `B = infer(body)` under `scope ∪ {p: A}`. The closure's *capture set* (free variables
            // of the body, bound in the enclosing scope) is implicit here — it is computed and lowered
            // by monomorphization (`mono.rs`), which reuses this same re-inference. No new `Ty` variant
            // (the closure struct is an ordinary `Ty::Data` after lowering — §4A.6). A **multi-argument**
            // `lambda(p1, p2, …) => body` desugars to nested single-param lambdas (`check_lambda`,
            // M-822) — no tuple-type prerequisite needed (DN-73 D1, ratified); only the zero-argument
            // case stays a never-silent refusal (G2).
            Expr::Lambda { params, body } => self.check_lambda(scope, params, body, expected),
            Expr::WithParadigm { .. } => self.err(
                "internal: a `with paradigm` block reached the checker — the ambient resolution \
                 pass should have stripped it (RFC-0012 §4.4)",
            ),
            Expr::Ascribe(inner, t) => self.check_ascribe(scope, inner, t),
            Expr::App { head, args } => self.check_app(scope, head, args, expected),
            // M-826: `(a, b, …)` — a tuple literal (arity ≥ 2). Desugars to a constructor
            // application `MkTuple$N(a, b, …)` on the synthetic `Tuple$N<A, B, …>` data type
            // (KC-3 — no new L0 node). Guarantee: `Empirical` (round-trip tested in differential.rs).
            Expr::TupleLit(elems) => self.check_tuple_lit(scope, elems, expected),
            // DN-58 §A (M-667): `fuse(a, b)` — lawful binary merge over the `Fuse` semilattice.
            Expr::Fuse { left, right } => self.check_fuse(scope, left, right, expected),
            // DN-58 §B (M-667): `reclaim(policy) { body }` — supervised scope.
            Expr::Reclaim { policy, body } => self.check_reclaim(scope, policy, body, expected),
        }
    }

    /// `consume <expr>` (DN-03 §1 / LR-8 / M-664) — affine acquisition of a `Substrate` value. The
    /// operand must have a `Substrate{tag}` type; any other operand type is an explicit refusal
    /// (never silent — G2). The result is the moved substrate (`Substrate{tag}`), now exclusively
    /// owned by the consumer.
    ///
    /// Guarantee **`Empirical`** (M-903; DN-71 Model S §4.2/§4.4 — updated from the M-664-era
    /// `Declared`, note that stated the pre-M-903 baseline honestly and is stale as of this
    /// affine-tracker landing, VR-5 requires this update rather than leaving it): the *single-use*
    /// property is now **checked**, not merely asserted — every reference to a `Substrate`-typed
    /// binding is recorded in `self.affine` (`crate::affine::Tracker`, wired below via `use_at` at
    /// every scope reference) and a second move on any reachable path is refused here, naming both
    /// sites. `Empirical`, not `Proven`: the tracker's own module docs are explicit that it is a
    /// sound-over-precise conservative approximation (verified by an exhaustive sweep, not a
    /// mechanized soundness proof) with a known, documented loop/closure multiplicity gap closed only
    /// by [`crate::substrate::SubstrateHandle::try_consume`]'s runtime backstop (M-903; wired into
    /// execution by M-904, DN-71 §4.3).
    fn check_consume(
        &self,
        scope: &mut Vec<(String, Ty)>,
        operand: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let (oty, oexpr) = self.check(scope, operand, None)?;
        let Ty::Substrate(tag) = &oty else {
            return self.err(format!(
                "`consume` requires an affine `Substrate{{…}}` operand (DN-03 §1 / LR-8), but its \
                 operand has type `{oty}` — only a `Substrate` value can be consumed \
                 (never silent — G2)"
            ));
        };
        let ty = Ty::Substrate(tag.clone());
        // Bidirectional contract: if a result type is expected, it must equal the moved substrate's
        // type — a mismatch is an explicit refusal, never a silent coercion (G2).
        if let Some(exp) = expected {
            if exp != &ty {
                return self.err(format!(
                    "`consume` of `{ty}` yields `{ty}`, but the context expects `{exp}` \
                     (never silent — G2)"
                ));
            }
        }
        Ok((ty, Expr::Consume(Box::new(oexpr))))
    }

    fn check_path(
        &self,
        scope: &[(String, Ty)],
        p: &Path,
        e: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        if p.0.len() != 1 {
            return self.err(format!(
                "dotted path `{}` does not resolve — multi-segment qualified-path *syntax* is \
                 deferred in v0; bring the name into scope with a `use` (`use {}`) and reference it \
                 by its final segment (M-662)",
                p.0.join("."),
                p.0.join(".")
            ));
        }
        let name = &p.0[0];
        // `rposition` (not `.rev().find()`) so `idx` is the scope's own index — the affine tracker
        // mirrors `scope` index-for-index (shadowing = later wins = the last/rightmost match).
        if let Some(idx) = scope.iter().rposition(|(n, _)| n == name) {
            let ty = &scope[idx].1;
            // M-903 / DN-71 §4.2: a reference to a `Substrate`-typed binding is a **move** — record
            // it in the affine tracker (a no-op for any non-`Substrate` binding, or when this `Cx`
            // is not running the affine pass — `crate::affine` docs). A second move of the same
            // binding on a reachable path is a never-silent double-consume refusal naming **both**
            // sites (RFC-0013 diagnostic style: this checker has no source spans, so "site" is the
            // binding's name plus a stable per-body use ordinal — honest, not a fabricated line/col).
            match self.affine.use_at(idx) {
                UseOutcome::NotAffine | UseOutcome::FirstUse => {}
                UseOutcome::DoubleUse {
                    tag,
                    first_ordinal,
                    this_ordinal,
                } => {
                    return self.err(format!(
                        "double-consume: `{name}` (`Substrate{{{tag}}}`) is used again here \
                         (reference #{this_ordinal} to `{name}` in `{site}`), but it was already \
                         moved earlier (reference #{first_ordinal} to `{name}`) — an affine \
                         `Substrate` value may be consumed, passed, returned, or captured **at \
                         most once** along any execution path (DN-71 Model S §4.2; RFC-0006 LR-8). \
                         This is a static checker refusal naming both the first move and this \
                         violating use — never silent (G2/VR-5).",
                        site = self.site
                    ));
                }
            }
            return Ok((ty.clone(), e.clone()));
        }
        // A reference to a name brought in by ≥2 globs (and not shadowed by a local/own/explicit
        // binding, which would have resolved above) is the never-silent glob-vs-glob ambiguity (G2).
        if let Some(err) = self.imports.ambiguity_error(self.site, name) {
            return Err(err);
        }
        if let Some((d, i)) = self.ctor(name) {
            if d.ctors[i].fields.is_empty() {
                // Nullary constructor as a value. A **generic** type has no fields to infer its type
                // arguments from, so they must come from `expected` (bidirectional) — an absent or
                // mismatched context is an explicit "ascribe it" error, never a guess (§11.3).
                let targs = if d.params.is_empty() {
                    vec![]
                } else {
                    match expected {
                        Some(Ty::Data(en, eargs))
                            if en == &d.name && eargs.len() == d.params.len() =>
                        {
                            eargs.clone()
                        }
                        _ => {
                            return self.err(format!(
                                "constructor `{name}` of generic `{}<…>` needs its type \
                                 argument(s) from context — ascribe the value (RFC-0007 §11.3, \
                                 never a guess)",
                                d.name
                            ))
                        }
                    }
                };
                return Ok((Ty::Data(d.name.clone(), targs), e.clone()));
            }
            return self.err(format!(
                "constructor `{name}` takes {} field(s) — apply it (W6 saturation)",
                d.ctors[i].fields.len()
            ));
        }
        // RFC-0024 §3 (M-686): a bare top-level function name in value position synthesizes
        // `Ty::Fn(param_ty, ret_ty)`. Only single-argument **monomorphic** functions are supported
        // in stage-1 — a generic function referenced bare without an `expected` context that fixes
        // its type arguments is a never-silent refusal (G2/VR-5); a multi-value-param function is
        // refused explicitly (partial application is out-of-scope per RFC-0024 §5).
        if let Some(fd) = self.fns.get(name) {
            // Nullary fn: not a value — cannot be used in function-value position (G2).
            if fd.sig.value_params.is_empty() {
                return self.err(format!(
                    "`{name}` takes 0 value parameters — a nullary function is not a \
                     first-class value; apply it directly (never a silent coercion — G2)"
                ));
            }
            // Multi-parameter monomorphic fn (M-822 / RFC-0024 §4A.5): used as a first-class value,
            // synthesize the curried type `A -> B -> … -> Z` and return a curried lambda expression
            // wrapping the saturated call. Zero-param is refused above; generic multi-param fns
            // need type-arg context from the expected type — deferred (never a silent accept, G2).
            if fd.sig.value_params.len() > 1 {
                if !fd.sig.params.is_empty() {
                    return self.err(format!(
                        "`{name}` is generic and multi-parameter — using a generic multi-parameter \
                         function as a first-class value requires type arguments from context; \
                         ascribe the value (RFC-0024 §4A.5, never a silent coercion — G2)"
                    ));
                }
                // Monomorphic multi-param fn: compute the curried type and return the
                // lambda wrapper expression. `check_path` has a read-only scope, so we build
                // the final checked expression structurally (no re-checking needed for a monomorphic
                // fn — all types are concrete from the declaration).
                let vparams = fd.sig.value_params.clone();
                // Resolve each parameter type and the return type.
                let mut param_tys: Vec<Ty> = Vec::with_capacity(vparams.len());
                for p in &vparams {
                    param_tys.push(resolve_ty(self.site, self.types, &[], &p.ty)?.0);
                }
                let ret_ty = resolve_ty(self.site, self.types, &[], &fd.sig.ret)?.0;
                // Build the curried type: A -> (B -> (… -> Z)) (right-associative).
                let curried_ty = param_tys.iter().rev().fold(ret_ty.clone(), |acc, t| {
                    Ty::Fn(Box::new(t.clone()), Box::new(acc))
                });
                // Bidirectional: if an expected type is given, it must match (never a coercion).
                if let Some(exp) = expected {
                    if exp != &curried_ty {
                        return self.err(format!(
                            "`{name}` has curried type `{curried_ty}`, but the context expects \
                             `{exp}` (type mismatch — RFC-0024 §4A.5, never a silent coercion)"
                        ));
                    }
                }
                // Build the saturated call inside the innermost lambda.
                let call = Expr::App {
                    head: Box::new(e.clone()),
                    args: vparams
                        .iter()
                        .map(|p| Expr::Path(Path(vec![p.name.clone()])))
                        .collect(),
                };
                // Build curried lambda: lambda(p1: A) => lambda(p2: B) => … => name(p1…pN).
                let mut body: Expr = call;
                for p in vparams.iter().rev() {
                    body = Expr::Lambda {
                        params: vec![p.clone()],
                        body: Box::new(body),
                    };
                }
                return Ok((curried_ty, body));
            }
            // Monomorphic callee: resolve the param and return types directly.
            if fd.sig.params.is_empty() {
                let (param_ty, _) =
                    resolve_ty(self.site, self.types, &[], &fd.sig.value_params[0].ty)?;
                let (ret_ty, _) = resolve_ty(self.site, self.types, &[], &fd.sig.ret)?;
                return Ok((Ty::Fn(Box::new(param_ty), Box::new(ret_ty)), e.clone()));
            }
            // Generic callee: type arguments must be fixed by context (`expected`). Attempt to
            // solve them from the expected `Ty::Fn(a, r)` via unification; any unsolved variable
            // is a never-silent refusal (G2/VR-5 — never a guessed default).
            let callee_vars = fd.sig.param_names();
            let mut subst: BTreeMap<String, Ty> = BTreeMap::new();
            if let Some(Ty::Fn(ea, er)) = expected {
                let want_a = resolve_ty(
                    self.site,
                    self.types,
                    &callee_vars,
                    &fd.sig.value_params[0].ty,
                )?
                .0;
                let want_r = resolve_ty(self.site, self.types, &callee_vars, &fd.sig.ret)?.0;
                // Best-effort: ignore unification errors here — unsolved vars are caught below.
                let _ = unify(self.site, &want_a, ea, &mut subst);
                let _ = unify(self.site, &want_r, er, &mut subst);
            }
            for v in &callee_vars {
                if !subst.contains_key(v) {
                    return self.err(format!(
                        "`{name}` is generic over `{v}`, but its type arguments cannot be \
                         determined from context — ascribe the value or the call site \
                         (RFC-0024 §5, RFC-0007 §11.3, never a guessed default)"
                    ));
                }
            }
            let want_a = resolve_ty(
                self.site,
                self.types,
                &callee_vars,
                &fd.sig.value_params[0].ty,
            )?
            .0;
            let want_r = resolve_ty(self.site, self.types, &callee_vars, &fd.sig.ret)?.0;
            let param_ty = subst_ty(&want_a, &subst);
            let ret_ty = subst_ty(&want_r, &subst);
            return Ok((Ty::Fn(Box::new(param_ty), Box::new(ret_ty)), e.clone()));
        }
        self.err(teach_unknown(name, &format!("unknown name `{name}`")))
    }

    fn check_let(
        &self,
        scope: &mut Vec<(String, Ty)>,
        name: &str,
        ty: Option<&TypeRef>,
        bound: &Expr,
        body: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let want = match ty {
            Some(t) => Some(resolve_ty(self.site, self.types, self.tyvars, t)?.0),
            None => None,
        };
        let (bty, bound2) = self.check(scope, bound, want.as_ref())?;
        if let Some(w) = &want {
            if w != &bty {
                return self.err(format!("let `{name}`: {}", edge_mismatch("bound", w, &bty)));
            }
        }
        scope.push((name.to_owned(), bty.clone()));
        self.affine.push(&bty);
        let r = self.check(scope, body, expected);
        scope.pop();
        self.affine.pop();
        let (rty, body2) = r?;
        Ok((
            rty,
            Expr::Let {
                name: name.to_owned(),
                ty: ty.cloned(),
                bound: Box::new(bound2),
                body: Box::new(body2),
            },
        ))
    }

    /// **Lambda / closure typing** (RFC-0024 §4A.6, M-704). A `lambda(p: A) => body` checks to
    /// `Ty::Fn(A, B)` where `B = infer(body)` under `scope ∪ {p: A}`. The capture set (free variables
    /// of `body` bound in the enclosing `scope`) is well-typed *by construction* — each free name is
    /// looked up in `scope` during body checking, so an unbound free variable is a never-silent
    /// `unknown name` refusal from `check_path` (G2), not a guess. The closure's *lowering* (the
    /// tag-sum struct + the generated `apply` dispatcher) is performed by monomorphization (`mono.rs`),
    /// which reuses this typing via re-inference (`infer_type`) — no new `Ty` variant here (the closure
    /// struct is an ordinary `Ty::Data` post-mono; §4A.6).
    ///
    /// **Multi-argument currying (M-822 / RFC-0024 §4A.5/§4A.8).** A multi-parameter lambda
    /// `lambda(p1: A, p2: B, …, pN: Z) => body` is treated as curried:
    /// `lambda(p1: A) => lambda(p2: B) => … => lambda(pN: Z) => body`. This is the RFC-0024 §4A.5
    /// partial-application path — each arrow in the curried chain lowers by the existing §4A single-arg
    /// machinery. Zero-parameter lambdas are a never-silent refusal (G2) — a zero-arg closure has
    /// no type without a unit type, which is a separate surface decision.
    ///
    /// **Bidirectional.** When the context supplies an expected `Ty::Fn(ea, er)`, the written param
    /// type must equal `ea` (a mismatch is a never-silent refusal, not a coercion), and the body is
    /// checked against `er` so a bare-decimal in the body takes its width from the codomain. A
    /// non-arrow expected type for a lambda is an explicit refusal.
    fn check_lambda(
        &self,
        scope: &mut Vec<(String, Ty)>,
        params: &[Param],
        body: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // Zero-parameter lambda: no type without a unit type — never-silent refusal (G2).
        if params.is_empty() {
            return self.err(
                "a `lambda` requires at least 1 parameter — a zero-argument lambda has no type \
                 without a unit/nullary type, which is a separate surface decision (never a silent \
                 accept — G2)"
                    .to_owned(),
            );
        }
        // Multi-argument currying (M-822 / RFC-0024 §4A.5/§4A.8): `lambda(p1, p2, …) => body`
        // desugars to `lambda(p1) => lambda(p2) => … => body`. Build the inner curried body and
        // check this as a single-param lambda whose body is the inner lambda. The curried type
        // is `A -> B -> … -> Z` (right-associative), which is exactly the §4A single-arg arrow
        // machinery applied recursively — no new mechanism needed (Declared; RFC-0024 §4A.5).
        if params.len() > 1 {
            let (first, rest) = params.split_first().expect("len > 1");
            // Build the inner curried lambda: `lambda(rest…) => body`.
            let inner_body = Expr::Lambda {
                params: rest.to_vec(),
                body: Box::new(body.clone()),
            };
            // The expected type for the outer one-param wrapper. If context provides
            // `Ty::Fn(ea, inner_expected)`, pass `inner_expected` to the inner recursion. A
            // non-arrow expected type is refused by the single-param path below (never silent, G2).
            // We do NOT extract `inner_expected` here — we pass `expected` as-is to the outer
            // single-param wrapper so the bidirectional domain-check fires for the first param.
            return self.check_lambda(scope, std::slice::from_ref(first), &inner_body, expected);
        }
        // Exactly one parameter — the base case.
        let [param] = params else {
            unreachable!("len == 1 after the multi-arg and zero-arg branches above")
        };
        // The written parameter type (lambda params are always ascribed — `parse_params_opt`).
        let (param_ty, _) = resolve_ty(self.site, self.types, self.tyvars, &param.ty)?;
        // If an expected arrow type is supplied, the codomain drives body checking and the domain
        // must match the written param type (never a silent coercion — G2).
        let expected_ret: Option<Ty> = match expected {
            Some(Ty::Fn(ea, er)) => {
                if ea.as_ref() != &param_ty {
                    return self.err(format!(
                        "this `lambda`'s parameter `{}` has type `{param_ty}`, but the context \
                         expects a `{ea} => {er}` (arrow-domain mismatch — RFC-0024 §4A.6, never a \
                         silent coercion)",
                        param.name
                    ));
                }
                Some(er.as_ref().clone())
            }
            Some(other) => {
                return self.err(format!(
                    "a `lambda` has function type, but the context expects `{other}` (a `lambda` is \
                     not a `{other}` — RFC-0024 §4A.6, never a silent coercion)"
                ));
            }
            None => None,
        };
        // Check the body under the extended scope; the param shadows any same-named outer binder.
        // (A `Substrate`-typed capture inside this body is tracked as one lexical use — sound only
        // when the closure runs at most once; the runtime backstop covers a closure called more
        // than once — `crate::affine` module docs, the known loop/closure limitation.)
        scope.push((param.name.clone(), param_ty.clone()));
        self.affine.push(&param_ty);
        let r = self.check(scope, body, expected_ret.as_ref());
        scope.pop();
        self.affine.pop();
        let (body_ty, body2) = r?;
        if let Some(er) = &expected_ret {
            if er != &body_ty {
                return self.err(format!(
                    "this `lambda`'s body has type `{body_ty}`, but the context expects the codomain \
                     `{er}` (arrow-codomain mismatch — RFC-0024 §4A.6, never a silent coercion)"
                ));
            }
        }
        Ok((
            Ty::Fn(Box::new(param_ty), Box::new(body_ty)),
            Expr::Lambda {
                params: params.to_vec(),
                body: Box::new(body2),
            },
        ))
    }

    fn check_if(
        &self,
        scope: &mut Vec<(String, Ty)>,
        cond: &Expr,
        conseq: &Expr,
        alt: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let bool_ty = Ty::Data("Bool".to_owned(), vec![]);
        let (c, cond2) = self.check(scope, cond, Some(&bool_ty))?;
        if c != bool_ty {
            return self.err(format!("if-condition must be Bool, got {c}"));
        }
        // M-903 / DN-71 §4.2: `conseq`/`alt` are mutually exclusive at runtime, so each is checked
        // from the *same* pre-branch affine snapshot; the post-branch states are then union-merged
        // (a slot moved in *either* branch counts as moved afterward — conservative, `crate::affine`
        // module docs on why this is sound-over-precise for v0).
        let pre_branch = self.affine.snapshot();
        let (t, conseq2) = self.check(scope, conseq, expected)?;
        let post_conseq = self.affine.snapshot();
        self.affine.restore(&pre_branch);
        // The else-branch may borrow the then-branch's type as its expected (so a bare decimal in
        // one branch can take the other's width).
        let (f, alt2) = self.check(scope, alt, expected.or(Some(&t)))?;
        self.affine.merge_alt(post_conseq);
        if t != f {
            return self.err(format!(
                "if-branches disagree: {}",
                edge_mismatch("else", &t, &f)
            ));
        }
        Ok((
            t,
            Expr::If {
                cond: Box::new(cond2),
                conseq: Box::new(conseq2),
                alt: Box::new(alt2),
            },
        ))
    }

    fn check_swap(
        &self,
        scope: &mut Vec<(String, Ty)>,
        value: &Expr,
        target: &TypeRef,
        policy: &Path,
    ) -> Result<(Ty, Expr), CheckError> {
        // The source repr is unconstrained by the target, so a bare-decimal source has no context
        // here — it must be ascribed (else an explicit UnresolvedWidth).
        let (vty, value2) = self.check(scope, value, None)?;
        if !matches!(vty, Ty::Binary(_) | Ty::Ternary(_) | Ty::Dense(_, _)) {
            return self.err(format!(
                "swap source must be a representation type, got {vty}"
            ));
        }
        let (tty, _) = resolve_ty(self.site, self.types, self.tyvars, target)?;
        if !matches!(tty, Ty::Binary(_) | Ty::Ternary(_) | Ty::Dense(_, _)) {
            return self.err(format!(
                "swap target must be a representation type, got {tty}"
            ));
        }
        Ok((
            tty,
            Expr::Swap {
                value: Box::new(value2),
                target: target.clone(),
                policy: policy.clone(),
            },
        ))
    }

    /// Type a `wild { body }` block — the **audited FFI floor** (M-661; RFC-0016 §8-Q6; LR-9/S6;
    /// ADR-014). Guarantee: **`Declared`** — this is a structural + audited *context* gate, never a
    /// theorem (VR-5). The rule (settled by the maintainer; RFC-0016 §8-Q6 amendment):
    ///
    /// 1. **Context gate.** A `wild` block is legal **only** inside a `@std-sys` nodule. In any other
    ///    nodule it is a hard [`CheckError`] — the audited FFI floor lives only in `std-sys` (LR-9),
    ///    never a silent escape from safe code (G2). This is a hard refusal, **not** a lint.
    /// 2. **Type by ascription, never synthesis.** The `wild` body is the **trusted/opaque FFI
    ///    escape** — it is **not** recursively type-checked (it conforms to the expected type; it is
    ///    *audited*, not *verified* — VR-5/ADR-014). So a result type must be supplied by the context
    ///    (`expected`); in a synthesis position the checker refuses with "ascribe the `wild` block's
    ///    result type" (never a guessed type — G2). The block then **has** that expected type.
    /// 3. **Effect source.** `wild` is the `ffi` effect source (M-660 binding): the enclosing fn must
    ///    declare `!{ffi}`. That is enforced separately, in the effect-coverage pass
    ///    ([`check_body_effect_coverage`]) — which credits a `wild` with performing `ffi` — so it
    ///    composes with the M-660 machinery rather than duplicating it here.
    /// 4. **Execution is staged.** There is no FFI host in v0, so `wild` *type-checks + gates + is
    ///    audited* now; actually *running* it elaborates to an explicit [`crate::elab::ElabError::Residual`]
    ///    (a future capability) — consistent with M-657/659/660 staging. The body is preserved
    ///    **verbatim** in the returned expression (opaque — no interior resolution).
    fn check_wild(&self, body: &Expr, expected: Option<&Ty>) -> Result<(Ty, Expr), CheckError> {
        if !self.std_sys {
            return self.err(
                "`wild` is denied outside a `@std-sys` nodule — the audited FFI floor lives only in \
                 `std-sys` (RFC-0016 §8-Q6, LR-9); never a silent escape — G2. Mark the nodule's \
                 header `@std-sys` to author the FFI floor.",
            );
        }
        // The body is the trusted/opaque FFI escape — NOT recursively checked (audited, not verified;
        // VR-5/ADR-014). It must therefore take its type from the context: refuse in synthesis.
        let Some(want) = expected else {
            return self.err(
                "a `wild` block has no synthesizable type — its body is the trusted/opaque FFI escape \
                 (not type-checked, only audited; ADR-014/VR-5). Ascribe the `wild` block's result \
                 type — `(wild { … }) : Binary{8}` (a special form takes the ascription parenthesized) \
                 — or use it in a typed position (a fn body / a `let` with an annotation, e.g. \
                 `let v: Binary{8} = wild { … }`) — never a guess (G2).",
            );
        };
        // `@std-sys` + a known expected type: the block *has* that type; the body is preserved
        // verbatim (opaque). Effect coverage (`ffi`) is checked by the M-660 pass, not here.
        Ok((want.clone(), Expr::Wild(Box::new(body.clone()))))
    }

    /// Type a `colony { hypha e1, …, hypha eN }` block (RFC-0008 §4.7; M-666). Every `hypha` body
    /// is type-checked under the **current** scope (RT1: hyphae share no state — each closes over
    /// the lexical environment by value, never over a mutable cell), and the colony's result type is
    /// the **last** hypha's type — the **RT2 spawn-order sequentialization**'s final observable
    /// (RFC-0008 §4.6 R1: the deterministic fragment's reference semantics is the sequentialization).
    /// The `expected` type, if any, applies to that last hypha (the colony's value); the earlier
    /// hyphae are checked with no expected. A colony must hold **≥ 1** hypha (defense-in-depth — the
    /// parser already requires it); an empty colony is an explicit refusal, never a silent unit.
    ///
    /// Honesty (Declared): this typing is the conservative v0 surface for RFC-0008 §4.7. With no
    /// product/tuple type in the v0 calculus, the colony cannot yet yield *all* hyphae's joined
    /// results as one heterogeneous value — that is later work (a join-result product; RFC-0008
    /// RT6/§4.7). The last-hypha-as-observable rule keeps the type honest and matches the RT2
    /// sequential reference; it never invents a product type or silently discards a type mismatch.
    fn check_colony(
        &self,
        scope: &mut Vec<(String, Ty)>,
        hyphae: &[Hypha],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let Some((last, leading)) = hyphae.split_last() else {
            return self.err(
                "a `colony` must contain at least one `hypha` (RFC-0008 §4.7 — a colony is a \
                 grouping of *active* hyphae); an empty `colony { }` has no observable",
            );
        };
        let mut checked: Vec<Hypha> = Vec::with_capacity(hyphae.len());
        // Leading hyphae: each is its own computation with no expected type. RT1 — each is checked
        // under the same lexical scope (closed over by value), never mutating it.
        for h in leading {
            let forage = match &h.forage {
                Some(p) => Some(Box::new(self.check_forage_policy(scope, p)?)),
                None => None,
            };
            let (_ty, body2) = self.check(scope, &h.body, None)?;
            checked.push(Hypha {
                forage,
                body: body2,
            });
        }
        // The final hypha carries the colony's observable (the RT2 sequentialization's last step), so
        // the `expected` type applies to it.
        let forage = match &last.forage {
            Some(p) => Some(Box::new(self.check_forage_policy(scope, p)?)),
            None => None,
        };
        let (rty, last_body2) = self.check(scope, &last.body, expected)?;
        checked.push(Hypha {
            forage,
            body: last_body2,
        });
        Ok((rty, Expr::Colony(checked)))
    }

    /// D-lite `@forage(policy)` policy check (RFC-0008 RT3; DN-63 §3.5; M-906/DN-70 D1). The
    /// D-lite subset narrows DN-63's open `policy: PlacementPolicy` expression sketch (§3.5) to a
    /// **literal binary bitmask**: each set bit `i` names one local worker candidate `worker-i`
    /// (DN-63 FLAG-13, narrowed for the subset, not resolved — the full node-level `Meta` signal
    /// set stays open, owned by the H2 `forage` maturity work, DN-70 §5 R-6). This is a
    /// *checker*-level narrowing, not a grammar one (`parse_hypha` accepts any expression),
    /// because the D-lite mechanism builds a `mycelium_select::SelectionPolicy` **statically**, at
    /// elaboration time, straight from the literal's bits (so `elaborate`/`elaborate_colony` and
    /// the L1 evaluator can agree identically on the DN-63 FLAG-14 empty-candidate refusal — see
    /// [`crate::elab`]'s `elab_colony` and [`crate::eval::ForageError`]). Reusing `mycelium-select`
    /// for an *arbitrary evaluated* Mycelium policy value is exactly the DN-70 §5 R-5 mechanized
    /// capture-and-set surface, explicitly deferred to H2. A non-literal or non-`Binary` policy is
    /// refused here, explicitly (never silently coerced or ignored — G2).
    ///
    /// Guarantee: `Declared` — the literal-bitmask convention is this D-lite subset's own,
    /// narrower-than-DN-63 choice (FLAG, per the M-906 task brief: "uses the existing
    /// SelectionPolicy — no new mechanism"; the mechanism itself is unchanged, only its D-lite
    /// input surface is narrowed).
    fn check_forage_policy(
        &self,
        scope: &mut Vec<(String, Ty)>,
        policy: &Expr,
    ) -> Result<Expr, CheckError> {
        let (ty, policy2) = self.check(scope, policy, None)?;
        if !matches!(ty, Ty::Binary(_)) {
            return self.err(format!(
                "`@forage(policy)` requires a binary-bitmask policy in the D-lite subset (got \
                 `{ty}`) — each set bit names one local worker candidate (DN-63 §3.5 FLAG-13, \
                 narrowed; RFC-0008 RT3); the general `PlacementPolicy` expression surface is the \
                 DN-70 §5 R-5 H2 mechanized-capture work (never-silent — G2)"
            ));
        }
        if !matches!(policy2, Expr::Lit(Literal::Bin(_))) {
            return self.err(
                "`@forage(policy)` requires a *literal* binary-bitmask policy in the D-lite \
                 subset (e.g. `@forage(0b101) hypha …`) — a computed `Binary` expression is not \
                 yet supported; the general dynamic-policy surface is the DN-70 §5 R-5 H2 \
                 mechanized `SelectionPolicy` capture-and-set work (never-silent — G2)"
                    .to_owned(),
            );
        }
        Ok(policy2)
    }

    /// DN-58 §A (M-667, M-965): `fuse(a, b)` — lawful binary merge over the `Fuse` semilattice
    /// instance carried by the type. Both operands must have the same type `T`; the result type is
    /// also `T`. Repr types (Binary/Ternary/Dense/Bytes/Seq) are always fusible — the semilattice
    /// join is bitwise-and for Binary/Ternary (commutative/associative/idempotent by semantics —
    /// Empirical). Named Data types require a `Fuse` trait instance; absent one the error is
    /// never-silent (G2).
    ///
    /// Guarantee: `Empirical` — the type-homogeneity check is Exact; the "repr types are fusible"
    /// claim is Empirical (bitwise-and semilattice laws are property-tested, not mechanized-Proven
    /// here). The Data-type Fuse-instance check is Declared (trait registry lookup, not a proof) —
    /// **unless** [`crate::fuse::check_fuse_laws`] (M-965 F-A2) exhaustively verified the instance's
    /// `join` (a finite nullary-constructor `for_ty`), in which case that instance's laws are
    /// `Empirical`-checked, not merely Declared (VR-5: the checker runs at registration, before this
    /// `fuse`-call site is ever reached, so a law-violating instance never gets this far — it was
    /// already refused when it was `impl`ed).
    /// **Landed (M-965):** F-A1 — `Fuse` is now a built-in prelude trait
    /// (`crate::fuse::prelude`), so a program `impl Fuse[T] for T { … }`s directly, with no
    /// `trait Fuse` declaration of its own. F-A2 — the semilattice law checker
    /// (`crate::fuse::check_fuse_laws`) is wired into `check_nodule_with`'s impl pass; it is
    /// exhaustive only over a **finite, enumerable** `for_ty` (a nullary-constructor Data type) —
    /// a composite/parametric `for_ty`'s laws are honestly left unchecked (DN-58 §A.6 F-A3,
    /// deferred), never silently assumed (VR-5).
    fn check_fuse(
        &self,
        scope: &mut Vec<(String, Ty)>,
        left: &Expr,
        right: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // Check the left operand under the expected type.
        let (lty, left2) = self.check(scope, left, expected)?;
        // Check the right operand against the left's type (homogeneity — DN-58 §A.4).
        let (rty, right2) = self.check(scope, right, Some(&lty))?;
        if lty != rty {
            return self.err(format!(
                "`fuse` requires both operands to have the same type; left is `{lty}`, right is \
                 `{rty}` (DN-58 §A.4 — the lawful-merge instance is per type; RFC-0008 RT6; \
                 never-silent — G2)"
            ));
        }
        // Fusibility check (Empirical for repr types; Declared for Data types with a Fuse instance).
        // `Ty::Float` is deliberately NOT in the fusible repr set (M-897): the repr-level meet is
        // bitwise-and, which has no lawful meaning over IEEE-754 bit patterns — a float `fuse`
        // requires an explicit `Fuse` instance (the Data-type path below), never a silent
        // bit-level merge (G2/VR-5).
        let fusible = matches!(
            &lty,
            Ty::Binary(_) | Ty::Ternary(_) | Ty::Dense(_, _) | Ty::Bytes | Ty::Seq(_, _)
        );
        if !fusible {
            // For Data types: look up a `Fuse` instance in the trait registry (Declared).
            let has_fuse_instance = if let Some(head) = type_head(&lty) {
                self.instances
                    .contains_key(&("Fuse".to_owned(), head.clone()))
            } else {
                false
            };
            if !has_fuse_instance {
                return self.err(format!(
                    "`fuse` requires a `Fuse` semilattice instance for `{lty}` — `Fuse` is a \
                     built-in prelude trait (M-965 F-A1), so declare \
                     `impl Fuse[{lty}] for {lty} {{ fn join(a: {lty}, b: {lty}) => {lty} = … }}` \
                     with a commutative/associative/idempotent `join`; RFC-0008 RT6 / DN-58 §A.4 \
                     (never-silent — G2). If `{lty}` is a finite nullary-constructor type, the \
                     three semilattice laws are exhaustively checked at `impl` time (M-965 F-A2) — \
                     a law violation refuses right there, before this `fuse` call is ever reached."
                ));
            }
        }
        Ok((
            lty,
            Expr::Fuse {
                left: Box::new(left2),
                right: Box::new(right2),
            },
        ))
    }

    /// DN-58 §B (M-667): `reclaim(policy) { body }` — attach a reified reclamation/supervision
    /// policy to a structured scope. `policy` is any expression (the policy type is open in v0 —
    /// FLAG F-B2: the `SupervisionPolicy` surface type is not yet committed in the type system);
    /// `body` is the supervised expression. The result type is the body's type. Never-silent on an
    /// ill-typed `policy` or `body` (G2). Guarantee: `Empirical` (M-713 property-tested).
    fn check_reclaim(
        &self,
        scope: &mut Vec<(String, Ty)>,
        policy: &Expr,
        body: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // The policy is any well-typed expression; its type is unconstrained in v0 (FLAG F-B2).
        let (_, policy2) = self.check(scope, policy, None)?;
        // The body is checked under the expected type.
        let (bty, body2) = self.check(scope, body, expected)?;
        Ok((
            bty,
            Expr::Reclaim {
                policy: Box::new(policy2),
                body: Box::new(body2),
            },
        ))
    }

    /// M-826 — check a tuple literal `(a, b, …)` (arity ≥ 2).
    ///
    /// Desugars to a constructor application on the synthetic `Tuple$N<A, B, …>` data type
    /// (KC-3: no new L0 node). Each element is checked, then the whole is rewritten to an
    /// equivalent `Expr::App { head: Path("MkTuple$N"), args }` so the downstream mono/elab passes
    /// see only ordinary data-constructor applications. Guarantee: `Empirical` (round-trip tested).
    ///
    /// Never-silent (G2):
    /// - Arity < 2 cannot arise (the parser rejects it).
    /// - If an expected type is provided and disagrees with the inferred `Tuple$N<…>`, an explicit
    ///   mismatch error is returned.
    fn check_tuple_lit(
        &self,
        scope: &mut Vec<(String, Ty)>,
        elems: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let n = elems.len();
        debug_assert!(n >= 2, "parser must reject 0- and 1-tuples");
        let tname = tuple_type_name(n);
        let ctor_name = tuple_ctor_name(n);

        // Determine expected element types if the context pins a compatible Tuple$N.
        let expected_elems: Option<Vec<Ty>> = match expected {
            Some(Ty::Data(tn, targs)) if *tn == tname && targs.len() == n => Some(targs.clone()),
            _ => None,
        };

        // Check each element; collect inferred element types + rebuilt expressions.
        let mut elem_tys = Vec::with_capacity(n);
        let mut rebuilt = Vec::with_capacity(n);
        for (i, el) in elems.iter().enumerate() {
            let exp_i = expected_elems.as_ref().map(|v| &v[i]);
            let (ety, el2) = self.check(scope, el, exp_i)?;
            elem_tys.push(ety);
            rebuilt.push(el2);
        }

        let result_ty = Ty::Data(tname.clone(), elem_tys);

        // Bidirectional contract: if a type was expected, it must match.
        if let Some(exp) = expected {
            if exp != &result_ty {
                return self.err(format!(
                    "tuple literal has type `{result_ty}`, but the context expects `{exp}` \
                     (M-826 — never a silent coercion, G2)"
                ));
            }
        }

        // Rewrite to `MkTuple$N(a, b, …)` — a Path application that the mono/elab passes
        // handle identically to any user data-constructor call.
        let head = Expr::Path(Path(vec![ctor_name]));
        Ok((result_ty, app_node(&head, rebuilt)))
    }

    fn check_ascribe(
        &self,
        scope: &mut Vec<(String, Ty)>,
        inner: &Expr,
        t: &TypeRef,
    ) -> Result<(Ty, Expr), CheckError> {
        let (want, _) = resolve_ty(self.site, self.types, self.tyvars, t)?;
        let (ity, inner2) = self.check(scope, inner, Some(&want))?;
        if ity != want {
            return self.err(format!(
                "ascription: {}",
                edge_mismatch("expression", &want, &ity)
            ));
        }
        Ok((want, Expr::Ascribe(Box::new(inner2), t.clone())))
    }

    fn check_app(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        args: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // M-826 Part 2 — lift the v0 "head must be a name" restriction to allow `f(x)(y)` and
        // other chained applications (RFC-0007 §4.4 narrowing noted for orchestrator). When the
        // head is not a `Path`, infer its type; if it has a function type `A -> B`, apply it as a
        // HOF call. If it does not have a function type, the error is explicit (never silent — G2).
        let Expr::Path(p) = head else {
            let (hty, h2) = self.check(scope, head, None)?;
            let Ty::Fn(param_ty, ret_ty) = hty else {
                return self.err(format!(
                    "application head is not a function — the expression before `(…)` has type \
                     `{hty}`, which is not callable (M-826 §Part2 — never silent, G2)"
                ));
            };
            if args.len() != 1 {
                return self.err(format!(
                    "higher-order application requires exactly 1 argument in stage-1; \
                     got {} — partial application / multi-arg HOF is deferred (RFC-0024 §5, G2)",
                    args.len()
                ));
            }
            let (got, a2) = self.check(scope, &args[0], Some(&param_ty))?;
            if got != *param_ty {
                return self.err(format!(
                    "higher-order call: argument has type `{got}` but callee expects `{param_ty}` \
                     (arrow-type mismatch — M-826, never a silent coercion, G2)"
                ));
            }
            return Ok((*ret_ty, app_node(&h2, vec![a2])));
        };
        if p.0.len() != 1 {
            return self.err(format!(
                "dotted call `{}` does not resolve — multi-segment qualified-path *syntax* is \
                 deferred in v0; `use {}` and call by its final segment (M-662)",
                p.0.join("."),
                p.0.join(".")
            ));
        }
        let name = &p.0[0];

        // RFC-0024 §3 (M-686): a scope binder of function type `Ty::Fn(a, r)` is applied as a
        // higher-order call. This covers `f(x)` inside a HOF body where `f: A -> B` is a parameter.
        // Single-argument only in stage-1 — applying more or fewer arguments is a never-silent error
        // (RFC-0024 §5, G2). Does NOT apply when the name also shadows a top-level fn (the scope
        // binder takes priority by the `scope.iter().rev()` lookup, already matching above).
        if let Some((_, Ty::Fn(param_ty, ret_ty))) = scope.iter().rev().find(|(n, _)| n == name) {
            let param_ty = param_ty.as_ref().clone();
            let ret_ty = ret_ty.as_ref().clone();
            if args.len() != 1 {
                return self.err(format!(
                    "`{name}` has function type and takes exactly 1 argument in stage-1; \
                     got {} (partial application / multi-arg HOF is deferred — RFC-0024 §5, \
                     never a silent coercion)",
                    args.len()
                ));
            }
            let (got, a2) = self.check(scope, &args[0], Some(&param_ty))?;
            if got != param_ty {
                return self.err(format!(
                    "`{name}` has type `{param_ty} -> {ret_ty}`; argument has type `{got}` \
                     (arrow-type mismatch — RFC-0024 §3, never a silent coercion)"
                ));
            }
            return Ok((ret_ty, app_node(head, vec![a2])));
        }

        // User function: each argument is checked **against** its declared parameter type, so a
        // bare-decimal argument takes the parameter's width.
        if let Some(fd) = self.fns.get(name) {
            if fd.sig.value_params.len() != args.len() {
                return self.err(format!(
                    "`{name}` takes {} argument(s), got {}",
                    fd.sig.value_params.len(),
                    args.len()
                ));
            }
            // Monomorphic callee — unchanged v0 path (exact bidirectional checking + error messages).
            if fd.sig.params.is_empty() {
                let mut rebuilt = Vec::with_capacity(args.len());
                for (pm, a) in fd.sig.value_params.iter().zip(args) {
                    let (want, _) = resolve_ty(self.site, self.types, &[], &pm.ty)?;
                    let (got, a2) = self.check(scope, a, Some(&want))?;
                    if want != got {
                        return self.err(format!(
                            "`{name}` parameter `{}`: {}",
                            pm.name,
                            edge_mismatch("argument", &want, &got)
                        ));
                    }
                    rebuilt.push(a2);
                }
                let (ret, _) = resolve_ty(self.site, self.types, &[], &fd.sig.ret)?;
                return Ok((ret, app_node(head, rebuilt)));
            }
            // Generic callee — extracted to a separate (non-inlined) method so `check_app`'s frame
            // stays small on the common monomorphic/prim path (the A4-02 host-stack-depth bound).
            return self.check_app_generic_fn(scope, head, name, fd, args);
        }

        // Constructor (W6 saturation).
        if let Some((d, i)) = self.ctor(name) {
            let arity = d.ctors[i].fields.len();
            let dname = d.name.clone();
            let params = d.params.clone();
            if arity != args.len() {
                return self.err(format!(
                    "constructor `{name}` takes {arity} field(s), got {} (W6 saturation)",
                    args.len()
                ));
            }
            // Monomorphic data type — the original inline path (small frame, exact v0 errors).
            if params.is_empty() {
                let fields = d.ctors[i].fields.clone();
                let mut rebuilt = Vec::with_capacity(args.len());
                for (want, a) in fields.iter().zip(args) {
                    let (got, a2) = self.check(scope, a, Some(want))?;
                    if want != &got {
                        return self.err(format!(
                            "constructor `{name}` field: {}",
                            edge_mismatch("argument", want, &got)
                        ));
                    }
                    rebuilt.push(a2);
                }
                return Ok((Ty::Data(dname, vec![]), app_node(head, rebuilt)));
            }
            // Generic data type — extracted (frame-size; A4-02).
            let fields = d.ctors[i].fields.clone();
            return self
                .check_app_generic_ctor(scope, head, name, dname, params, fields, args, expected);
        }

        // Unqualified **trait-method** call (RFC-0019 §4.4; RFC-0007 §12.1): if `name` is not a
        // fn/ctor (checked above) but is a method of exactly one registered trait, resolve it
        // through a bound in scope or a concrete instance — extracted (frame-size; A4-02).
        if self.is_trait_method(name) {
            return self.check_trait_method_call(scope, head, name, args, expected);
        }

        // Builtin prim: width-polymorphic and width-preserving, so the result's expected width (or
        // a concrete operand's width) anchors any bare-decimal operand (RFC-0012 §4.3). Inlined
        // (not a separate method) to keep the per-nesting-level host-stack frame count at the
        // pre-M-344 depth — the parser bounds AST nesting, and the checker must fit that bound
        // without overflowing (A4-02).
        // A call to a name brought in by ≥2 globs (unshadowed) is the never-silent glob-vs-glob
        // ambiguity (G2) — refuse before falling through to the prim/unknown diagnostic.
        if let Some(err) = self.imports.ambiguity_error(self.site, name) {
            return Err(err);
        }
        // RFC-0032 D1 (M-747): the comparison prims `eq`/`lt` are **width-collapsing** — two
        // equal-width operands of one paradigm reduce to `Binary{1}` (the realized `Bool`; see
        // `prims.rs`). They do not fit the width-*preserving* `prim_family` path below (whose result
        // width equals the operand width and which can anchor a bare decimal off the *expected* type).
        // Because the result is `Binary{1}` — not the operand width — the expected type cannot anchor a
        // bare-decimal operand here; a *concrete* operand still can, so we anchor a bare decimal off the
        // other operand (consistent with the width-preserving prims). Only when **both** operands are
        // bare decimals is there no anchor at all, and then it is refused honestly (G2 — never a default
        // width); ascribe one operand or write it explicitly.
        // M-767 (RFC-0033 §4.1.2; ADR-028): `lt_s` — the **signed** (two's-complement) order over
        // `Binary{N}` — shares this width-collapsing comparison branch (same anchoring rules, same
        // `Binary{1}` result) but is Binary-only: `lt` reads Binary operands as unsigned
        // magnitudes, so the signed order is a distinct named op (DN-72 `_s`), while balanced
        // ternary's `lt` order is already the signed order (no ternary `lt_s` exists).
        if matches!(name.as_str(), "eq" | "lt" | "lt_s") {
            if args.len() != 2 {
                return self.err(format!(
                    "`{name}` takes two operands (got {}); RFC-0032 D1",
                    args.len()
                ));
            }
            // First pass: type the non-bare-decimal operand(s); the first concrete one anchors width.
            let mut typed: Vec<Option<(Ty, Expr)>> = vec![None, None];
            let mut anchor: Option<Ty> = None;
            for (i, a) in args.iter().enumerate() {
                if matches!(a, Expr::Lit(Literal::AmbientInt(_, _))) {
                    continue;
                }
                let (t, a2) = self.check(scope, a, None)?;
                if anchor.is_none() {
                    anchor = Some(t.clone());
                }
                typed[i] = Some((t, a2));
            }
            // Second pass: resolve each bare decimal against the concrete operand's type, if any.
            let mut tys = Vec::with_capacity(2);
            let mut rebuilt = Vec::with_capacity(2);
            for (i, a) in args.iter().enumerate() {
                let (t, a2) = match typed[i].take() {
                    Some(x) => x,
                    None => {
                        let anchor_ty = anchor.clone().ok_or_else(|| {
                            CheckError::new(
                                self.site,
                                format!(
                                    "both operands of `{name}` are bare decimals, so neither pins a \
                                     width — and because comparison is width-collapsing (RFC-0032 D1) \
                                     the `Binary{{1}}` result cannot anchor them either. Ascribe one \
                                     operand or write it explicitly (RFC-0012 §4.3, never a default \
                                     width)"
                                ),
                            )
                        })?;
                        self.check(scope, a, Some(&anchor_ty))?
                    }
                };
                tys.push(t);
                rebuilt.push(a2);
            }
            match (&tys[0], &tys[1]) {
                (Ty::Binary(x), Ty::Binary(y)) if x == y => {}
                // M-767: `lt_s` on a ternary pair refuses with the real routing — balanced
                // ternary's D1 `lt` order IS the signed order, so a distinct ternary `lt_s`
                // would silently duplicate it (G2: explicit teaching refusal, never two names
                // for one order).
                (Ty::Ternary(x), Ty::Ternary(y)) if x == y && name == "lt_s" => {
                    return self.err(
                        "`lt_s` is the Binary two's-complement order (RFC-0033 §4.1.2; ADR-028); \
                         balanced ternary is inherently signed, so `lt` already orders it by \
                         signed value — use `lt`"
                            .to_owned(),
                    );
                }
                (Ty::Ternary(x), Ty::Ternary(y)) if x == y => {}
                // ADR-040 §2.4 (M-899): float ordering is *partial* (NaN is unordered), so
                // floats never route through the D1 total order — refuse with the real routing
                // (never a silently-invented order for NaN; G2).
                (Ty::Float, Ty::Float) => {
                    return self.err(format!(
                        "`{name}` is the RFC-0032 D1 Binary/Ternary comparison; float ordering \
                         is partial (NaN is unordered — ADR-040 §2.4), so floats have their own \
                         explicit prims: use `flt_lt`/`flt_le`/`flt_gt`/`flt_ge`/`flt_eq` (any \
                         NaN operand → false) or the named total order `flt_total_le` for \
                         sorting/keying (M-899)"
                    ));
                }
                _ => {
                    return self.err(format!(
                        "`{name}` compares two equal-width operands of the same paradigm \
                         (Binary/Ternary), got {:?} and {:?} (RFC-0032 D1)",
                        tys[0], tys[1]
                    ));
                }
            }
            return Ok((Ty::Binary(Width::Lit(1)), app_node(head, rebuilt)));
        }
        // RFC-0032 D3/D4 (M-749/M-750): the never-silent indexing/length prims over `Seq`/`Bytes`.
        // Their signatures are **not** width-preserving (they don't fit `prim_family`), so they are
        // typed in a dedicated branch:
        //   - `seq_len(s: Seq{T, N}) -> Binary{32}`   (the element count as an unsigned 32-bit value)
        //   - `seq_get(s: Seq{T, N}, i: Binary{W}) -> T`   (the element type; out-of-bounds is a
        //     never-silent *runtime* refusal, not a type error)
        //   - `bytes_len(b: Bytes) -> Binary{32}`
        //   - `bytes_get(b: Bytes, i: Binary{W}) -> Binary{8}`
        //   - `bytes_slice(b: Bytes, start: Binary{W}, end: Binary{W}) -> Bytes`  (DN-43; never-silent
        //     out-of-range/inverted-range refusal is a *runtime* contract, not a type error)
        //   - `bytes_concat(b1: Bytes, b2: Bytes) -> Bytes`  (DN-43; total)
        // The index `i` is any `Binary{W}` (an unsigned magnitude); a bare-decimal index has no
        // anchor here (the result is not the index width), so it must be written/ascribed explicitly
        // (G2 — never a default width).
        if let Some(ret) = self.try_check_seq_bytes_prim(scope, head, name, args)? {
            return Ok(ret);
        }
        // RFC-0001 §4.1 / RFC-0002 §5 (M-890, `enb` Gap C): the tensor-valued dense elementwise
        // prims. Dim + dtype live in the type (`Dense{d, s}`), so a shape/dtype mismatch is a
        // *static* explicit refusal here — never a broadcast/coercion (G2). Not width-preserving
        // in the `prim_family` sense (Dense has no bare-decimal encoding to anchor — RFC-0012
        // §4.3 refuses bare decimals under a Dense ambient), so a dedicated branch, like seq/bytes.
        if let Some(ret) = self.try_check_dense_prim(scope, head, name, args)? {
            return Ok(ret);
        }
        // RFC-0003 §3/§4 / ADR-008 (M-892, `enb` Gap C): the model-dispatched VSA bind group.
        // Model + dim live in the type (`VSA{model, dim, sparsity}`), so a model/dim mismatch is
        // a *static* explicit refusal here — never a coercion (G2) — and an out-of-set model is
        // a static refusal naming the dispatch set. A dedicated branch like dense/seq/bytes.
        if let Some(ret) = self.try_check_vsa_prim(scope, head, name, args)? {
            return Ok(ret);
        }
        // ADR-040 §2.5 (M-898, `enb` Gap A): the scalar-float arithmetic prims. `Float` is a
        // nullary type (no width to anchor — every operand and the result are exactly `Float`),
        // so a dedicated branch like dense/seq/bytes, not the width-polymorphic `prim_family`.
        if let Some(ret) = self.try_check_float_prim(scope, head, name, args)? {
            return Ok(ret);
        }
        let Some(fam) = prim_family(name) else {
            return self.err(teach_unknown(
                name,
                &format!("unknown function/constructor/prim `{name}`"),
            ));
        };
        // First, type the operands that are *not* bare decimals; they anchor the width.
        let mut typed: Vec<Option<(Ty, Expr)>> = vec![None; args.len()];
        let mut anchor: Option<u32> = expected.and_then(|t| fam.width_of(t));
        for (i, a) in args.iter().enumerate() {
            if matches!(a, Expr::Lit(Literal::AmbientInt(_, _))) {
                continue;
            }
            let (t, a2) = self.check(scope, a, None)?;
            if anchor.is_none() {
                anchor = fam.width_of(&t);
            }
            typed[i] = Some((t, a2));
        }
        // Then resolve each bare-decimal operand against the anchor.
        let mut arg_tys = Vec::with_capacity(args.len());
        let mut rebuilt = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let (t, a2) = match typed[i].take() {
                Some(x) => x,
                None => {
                    let w = anchor.ok_or_else(|| {
                        CheckError::new(
                            self.site,
                            format!(
                                "UnresolvedWidth: a bare decimal operand of `{name}` has no width \
                                 here — no concrete operand or expected type pins it. Ascribe it, \
                                 or write it explicitly (RFC-0012 §4.3, never a default width)"
                            ),
                        )
                    })?;
                    self.check(scope, a, Some(&fam.ty(w)))?
                }
            };
            arg_tys.push(t);
            rebuilt.push(a2);
        }
        match prim_sig(name, &arg_tys) {
            Some(ret) => Ok((ret, app_node(head, rebuilt))),
            None => self.err(format!(
                "`{name}` does not accept argument types {arg_tys:?} (T-Op; RFC-0007 §4.4)"
            )),
        }
    }

    /// Check a call to a **generic** function (RFC-0007 §11.3): resolve the callee's signature with
    /// the *callee's* type parameters as abstract variables, **unify** the declared parameter types
    /// against the actual argument types to infer the type arguments, and substitute the solution
    /// into the return type. An unsolved type parameter is an explicit error — never a guessed
    /// default (G2/VR-5). Extracted (`#[inline(never)]`) so [`Self::check_app`]'s frame stays within
    /// the A4-02 host-stack-depth bound on the common monomorphic/prim path.
    #[inline(never)]
    fn check_app_generic_fn(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        fd: &FnDecl,
        args: &[Expr],
    ) -> Result<(Ty, Expr), CheckError> {
        let callee_vars = fd.sig.param_names();
        let mut subst: BTreeMap<String, Ty> = BTreeMap::new();
        let mut rebuilt = Vec::with_capacity(args.len());
        for (pm, a) in fd.sig.value_params.iter().zip(args) {
            let want = resolve_ty(self.site, self.types, &callee_vars, &pm.ty)?.0;
            let want_now = subst_ty(&want, &subst);
            // A fully-concrete (post-substitution) expected type drives the argument's check (so a
            // bare decimal takes the width); a still-abstract one lets the argument synthesize.
            let exp = if has_var(&want_now) {
                None
            } else {
                Some(&want_now)
            };
            let (got, a2) = self.check(scope, a, exp)?;
            unify(self.site, &want_now, &got, &mut subst).map_err(|e| {
                CheckError::new(
                    self.site,
                    format!("`{name}` argument `{}`: {}", pm.name, e.message),
                )
            })?;
            rebuilt.push(a2);
        }
        for v in &callee_vars {
            if !subst.contains_key(v) {
                return self.err(format!(
                    "`{name}` is generic over `{v}`, but this call does not determine it — ascribe an argument or the result (RFC-0007 §11.3, never a guessed default)"
                ));
            }
        }
        // DN-42 / M-753 step-b: also check that every width parameter is resolved.
        // A width var is bound as `var_name → Ty::Binary(Width::Lit(n))` in the same subst map.
        // An undetermined width parameter is an explicit error — never a guessed default (G2/VR-5).
        let callee_wvars = fd.sig.width_param_names();
        for v in &callee_wvars {
            if !subst.contains_key(v) {
                return self.err(format!(
                    "`{name}` is width-generic over `{v}`, but this call does not determine the width — ascribe an argument (DN-42 §4, never a guessed default / VR-5)"
                ));
            }
        }
        // Bounded-generic-call satisfiability (RFC-0019 §4.1/§4.5; RFC-0007 §12.1). Now `T = C` is
        // solved, for each bound `T: Tr` on a type-parameter require an instance `(Tr, head(C))` to
        // exist — else an explicit "no instance" refusal. The dictionary VALUE is NOT constructed
        // here (staged to elaboration — RFC-0007 §12.3 / M-673); the checker only validates the bound
        // is *satisfiable* ("dictionary typing"), never a silent skip (G2).
        for p in &fd.sig.params {
            for b in &p.bounds {
                let concrete = subst
                    .get(&p.name)
                    .cloned()
                    .unwrap_or(Ty::Var(p.name.clone()));
                self.require_instance(
                    &b.name,
                    &concrete,
                    &format!("required by `{name}`'s bound `{}: {}`", p.name, b.name),
                )?;
            }
        }
        let ret = subst_ty(
            &resolve_ty(self.site, self.types, &callee_vars, &fd.sig.ret)?.0,
            &subst,
        );
        Ok((ret, app_node(head, rebuilt)))
    }

    /// Require that an instance `(trait_name, type_head(concrete))` exists, or refuse explicitly
    /// (RFC-0019 §4.5; G2). Used for bounded-generic-call satisfiability and concrete trait-method
    /// resolution. If `concrete` is still abstract (a bare `Ty::Var` — the call is itself inside a
    /// bounded generic whose bound already guarantees the instance at the eventual concrete type),
    /// the requirement is **discharged by the bound in scope**: it is satisfied iff that same
    /// `(var: trait)` bound is present in `self.bounds` (else an explicit "no instance/bound" error).
    /// The dictionary value is staged to elaboration (M-673) — this is satisfiability only.
    fn require_instance(
        &self,
        trait_name: &str,
        concrete: &Ty,
        because: &str,
    ) -> Result<(), CheckError> {
        match type_head(concrete) {
            Some(head) => match self.instances.get(&(trait_name.to_owned(), head)) {
                // Head-erasure is the COHERENCE key (≤1 instance per head); RESOLUTION must still
                // match the FULL concrete type — a `Binary{8}` instance does not satisfy a `Binary{4}`
                // call. Head-erasure over-REJECTS duplicates; it must never over-ACCEPT a different
                // type (G2: never silently reuse a mismatched instance).
                Some(info) if info.for_ty == *concrete => Ok(()),
                Some(info) => self.err(format!(
                    "no instance `{trait_name}` for `{concrete}` ({because}) — the `{trait_name}` \
                     instance on this type head is declared for `{}`, not `{concrete}` (RFC-0019 §4.5)",
                    info.for_ty
                )),
                None => self.err(format!(
                    "no instance `{trait_name}` for `{concrete}` ({because}) — declare \
                     `impl {trait_name}<…> for {concrete} {{ … }}` (RFC-0019 §4.5, never assumed)"
                )),
            },
            // `concrete` is a type-variable in scope: discharge via a matching bound (the dictionary
            // is threaded by the eventual caller — RFC-0007 §12.3 / M-673).
            None => {
                let Ty::Var(v) = concrete else {
                    return self.err(format!(
                        "no instance `{trait_name}` for `{concrete}` ({because})"
                    ));
                };
                let satisfied = self
                    .bounds
                    .iter()
                    .any(|(pv, bs)| pv == v && bs.iter().any(|b| b.name == trait_name));
                if satisfied {
                    Ok(())
                } else {
                    self.err(format!(
                        "no instance/bound provides `{trait_name}` for type variable `{v}` \
                         ({because}) — add the bound `{v}: {trait_name}` (RFC-0019 §4.1)"
                    ))
                }
            }
        }
    }

    /// Is `name` a method of some registered trait? (The call-resolution path uses this to decide
    /// whether to try trait-method resolution — fn/ctor names were already dispatched, so a name
    /// reaching here that matches a trait method is an unqualified trait-method call.)
    fn is_trait_method(&self, name: &str) -> bool {
        self.traits
            .values()
            .any(|tr| tr.sigs.iter().any(|s| s.name == name))
    }

    /// Resolve and type an **unqualified trait-method call** `m(args)` (RFC-0019 §4.4; RFC-0007
    /// §12.1; guarantee: `Declared`). The method must belong to **exactly one** trait (ambiguity
    /// across traits is an explicit error — never a guess). The trait's single type-parameter is
    /// determined by **unifying** the trait method's signature against the actual argument types;
    /// then an instance must exist — either a concrete `(Tr, head(C))` or a `T: Tr` **bound in
    /// scope** (dictionary staged to elaboration — M-673). The call types at the (substituted)
    /// method return type. Extracted (`#[inline(never)]`) for the frame-size reason as the other
    /// generic paths (A4-02).
    #[inline(never)]
    fn check_trait_method_call(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        args: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // 1. The trait(s) that declare a method named `name`. Exactly one ⇒ resolve; >1 ⇒ ambiguous.
        let owners: Vec<&TraitInfo> = self
            .traits
            .values()
            .filter(|tr| tr.sigs.iter().any(|s| s.name == name))
            .collect();
        let tr = match owners.as_slice() {
            [one] => *one,
            [] => unreachable!("is_trait_method gated this call"),
            many => {
                let names: Vec<&str> = many.iter().map(|t| t.name.as_str()).collect();
                return self.err(format!(
                    "ambiguous trait-method call `{name}` — declared by multiple traits ({}); \
                     stage-1 has no qualified-call syntax, so this is an explicit refusal, never a \
                     guess (RFC-0019 §4.4)",
                    names.join(", ")
                ));
            }
        };
        if tr.params.len() != 1 {
            return self.err(format!(
                "trait-method resolution for `{name}` needs a single-parameter trait in stage-1 \
                 (trait `{}` has {} parameters — multi-parameter traits are v2, RFC-0019 §10)",
                tr.name,
                tr.params.len()
            ));
        }
        let sig = tr
            .sigs
            .iter()
            .find(|s| s.name == name)
            .expect("owner has the method");
        if sig.value_params.len() != args.len() {
            return self.err(format!(
                "trait method `{}::{name}` takes {} argument(s), got {}",
                tr.name,
                sig.value_params.len(),
                args.len()
            ));
        }
        // 2. Unify the method's (abstract-over-the-trait-param) value-param types against the actual
        //    argument types to solve the trait parameter — never a guess (RFC-0007 §11.3).
        let tparam = &tr.params[0];
        let trait_vars = std::slice::from_ref(tparam);
        let mut subst: BTreeMap<String, Ty> = BTreeMap::new();
        // Seed from `expected` against the (abstract) return type, so a nullary-ish return can pin
        // the parameter even when an argument is a bare decimal.
        if let Some(exp) = expected {
            let ret_abs = resolve_ty(self.site, self.types, trait_vars, &sig.ret)?.0;
            let _ = unify(self.site, &ret_abs, exp, &mut subst);
        }
        let mut rebuilt = Vec::with_capacity(args.len());
        for (pm, a) in sig.value_params.iter().zip(args) {
            let want = resolve_ty(self.site, self.types, trait_vars, &pm.ty)?.0;
            let want_now = subst_ty(&want, &subst);
            let exp = if has_var(&want_now) {
                None
            } else {
                Some(&want_now)
            };
            let (got, a2) = self.check(scope, a, exp)?;
            unify(self.site, &want_now, &got, &mut subst).map_err(|e| {
                CheckError::new(
                    self.site,
                    format!(
                        "trait method `{}::{name}` argument `{}`: {}",
                        tr.name, pm.name, e.message
                    ),
                )
            })?;
            rebuilt.push(a2);
        }
        // 3. The trait parameter must be determined; then an instance (concrete or via a scope bound)
        //    must provide the trait — else an explicit "no instance/bound" refusal (never a guess).
        let Some(receiver) = subst.get(tparam).cloned() else {
            return self.err(format!(
                "trait-method call `{name}` does not determine trait `{}`'s type parameter `{tparam}` \
                 from its arguments — ascribe an argument or the result (RFC-0019 §4.4, never a guess)",
                tr.name
            ));
        };
        self.require_instance(
            &tr.name,
            &receiver,
            &format!(
                "no instance/bound provides `{}::{name}` for these arguments",
                tr.name
            ),
        )?;
        let ret = subst_ty(
            &resolve_ty(self.site, self.types, trait_vars, &sig.ret)?.0,
            &subst,
        );
        Ok((ret, app_node(head, rebuilt)))
    }

    /// Check a saturated application of a **generic** data constructor (RFC-0007 §11.2/§11.3). The
    /// constructor's declared fields are abstract over the type's parameters; the type arguments are
    /// taken from `expected` when it pins this data type (bidirectional), otherwise **inferred** from
    /// the field arguments via [`unify`]. An undetermined parameter is an explicit "ascribe it" error
    /// — never a guess. Extracted (`#[inline(never)]`) for the same frame-size reason as
    /// [`Self::check_app_generic_fn`].
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn check_app_generic_ctor(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        dname: String,
        params: Vec<String>,
        fields: Vec<Ty>,
        args: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let mut subst: BTreeMap<String, Ty> = BTreeMap::new();
        if let Some(Ty::Data(en, eargs)) = expected {
            if *en == dname && eargs.len() == params.len() {
                for (p, ea) in params.iter().zip(eargs) {
                    subst.insert(p.clone(), ea.clone());
                }
            }
        }
        let mut rebuilt = Vec::with_capacity(args.len());
        for (want, a) in fields.iter().zip(args) {
            let want_now = subst_ty(want, &subst);
            let exp = if has_var(&want_now) {
                None
            } else {
                Some(&want_now)
            };
            let (got, a2) = self.check(scope, a, exp)?;
            unify(self.site, &want_now, &got, &mut subst).map_err(|e| {
                CheckError::new(
                    self.site,
                    format!("constructor `{name}` field: {}", e.message),
                )
            })?;
            rebuilt.push(a2);
        }
        let mut targs = Vec::with_capacity(params.len());
        for p in &params {
            match subst.get(p) {
                Some(t) => targs.push(t.clone()),
                None => {
                    return self.err(format!(
                        "constructor `{name}` does not determine type parameter `{p}` of \
                         `{dname}<…>` — ascribe the value (RFC-0007 §11.3, never a guess)"
                    ))
                }
            }
        }
        Ok((Ty::Data(dname, targs), app_node(head, rebuilt)))
    }

    /// T-For (RFC-0007 §4.8): `xs` must be a *linearly recursive* data type (nil/cons shape);
    /// `init : A`; `body : A` under `x : E, acc : A`; the whole expression is `A`. Every shape
    /// violation is an explicit refusal — general catamorphisms are an L2 concern.
    #[allow(clippy::too_many_arguments)]
    fn check_for(
        &self,
        scope: &mut Vec<(String, Ty)>,
        x: &str,
        xs: &Expr,
        acc: &str,
        init: &Expr,
        body: &Expr,
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let (sty, xs2) = self.check(scope, xs, None)?;
        let Ty::Data(tname, targs) = &sty else {
            return self.err(format!(
                "`for` iterates a linearly recursive data value, got {sty} (RFC-0007 §4.8)"
            ));
        };
        // For a generic linear type (`List<Binary{8}>`) the element type is the declared element
        // (`Var("A")`) with the scrutinee's type arguments substituted in (RFC-0007 §11.2).
        let elem = linear_elem_ty(self.site, self.types, tname, targs)?;
        // The accumulator type is the whole expression's type, so the `for`'s expected anchors `init`.
        let (aty, init2) = self.check(scope, init, expected)?;
        // (A `Substrate`-typed `x`/`acc` capture inside `body` is tracked as one lexical use —
        // sound only when `body` runs at most once, which a `for` generally does not guarantee;
        // the runtime backstop covers a repeated iteration — `crate::affine` module docs.)
        scope.push((x.to_owned(), elem.clone()));
        self.affine.push(&elem);
        scope.push((acc.to_owned(), aty.clone()));
        self.affine.push(&aty);
        let r = self.check(scope, body, Some(&aty));
        scope.pop();
        self.affine.pop();
        scope.pop();
        self.affine.pop();
        let (bty, body2) = r?;
        if bty != aty {
            return self.err(format!(
                "`for` body must yield the accumulator type {}",
                edge_mismatch("body", &aty, &bty)
            ));
        }
        Ok((
            aty,
            Expr::For {
                x: x.to_owned(),
                xs: Box::new(xs2),
                acc: acc.to_owned(),
                init: Box::new(init2),
                body: Box::new(body2),
            },
        ))
    }

    /// Type a `match` over a data, `Binary`, or `Ternary` scrutinee with **nested** patterns
    /// (RFC-0007 §4.4/§4.7). Each arm's pattern is checked against the scrutinee type (binders enter
    /// scope at their field types), the arm bodies must agree, and coverage is decided by the
    /// **Maranget usefulness** algorithm ([`crate::usefulness`]): the match must be **exhaustive**
    /// (a `_` is not useful — otherwise the witness names a missing case) and every arm must be
    /// **reachable** (an arm covered by the earlier ones is a redundancy error). This unifies the data
    /// match and the M-320 literal match: a `Binary`/`Ternary` value domain is never enumerated, so a
    /// literal match still needs a `_`/binder default (W7 — coverage is checked, never assumed).
    /// Bare-decimal literal patterns take their width from the scrutinee/field type (RFC-0012 §4.3).
    fn check_match(
        &self,
        scope: &mut Vec<(String, Ty)>,
        scrutinee: &Expr,
        arms: &[crate::ast::Arm],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        let (sty, scrut2) = self.check(scope, scrutinee, None)?;
        if !matches!(sty, Ty::Data(_, _) | Ty::Binary(_) | Ty::Ternary(_)) {
            return self.err(format!(
                "match scrutinee must be a data, Binary, or Ternary type, got {sty}"
            ));
        }
        if arms.is_empty() {
            return self.err("a match needs at least one arm");
        }
        // Or-pattern desugar (RFC-0020 §9 / R20-Q3, KC-3 — zero kernel growth): expand each
        // surface arm whose pattern is `Pattern::Or(alts)` into `|alts|` plain arms that share
        // the same body. Binding-consistency check (never-silent G2): every alternative in an
        // or-pattern must bind the same set of variable names at the same types — a mismatch is a
        // `CheckError`, never a silent accept. After desugar, `Pattern::Or` never appears in the
        // arms the downstream loop sees, so `resolve_pattern`/`check_pattern`/the evaluator/the
        // elaborator all stay Or-free (their never-silent guards below confirm this invariant).
        let flat_arms: Vec<crate::ast::Arm> =
            arms.iter().try_fold(Vec::new(), |mut acc, arm| {
                match &arm.pattern {
                    Pattern::Or(alts) => {
                        // Validate: the or-pattern must have ≥ 2 alternatives (the parser only
                        // builds a `Pattern::Or` when it sees at least one `|`, so ≥ 2 is
                        // guaranteed by construction — this check defends against API misuse).
                        if alts.len() < 2 {
                            return self.err(
                                "internal: Pattern::Or must have at least 2 alternatives \
                                 (or-pattern invariant violation — report this)",
                            );
                        }
                        // Check binding consistency across all alternatives.
                        // Each alternative is resolved + typed against the scrutinee type; their
                        // binder sets must agree (same names, same types — G2/never-silent).
                        // We collect the binders of the FIRST alternative as the reference set,
                        // then verify each subsequent alternative against it.
                        let mut ref_binds: Option<Vec<(String, Ty)>> = None;
                        for (alt_idx, alt) in alts.iter().enumerate() {
                            let resolved = self.resolve_pattern(alt, &sty)?;
                            let mut binds: Vec<(String, Ty, Vec<usize>)> = Vec::new();
                            self.check_pattern(&resolved, &sty, &mut binds)?;
                            let binder_sig: Vec<(String, Ty)> = binds
                                .iter()
                                .map(|(n, t, _)| (n.clone(), t.clone()))
                                .collect();
                            match &ref_binds {
                                None => ref_binds = Some(binder_sig),
                                Some(ref_sig) => {
                                    // Same names + types required (order-insensitive: sort by name).
                                    let mut got = binder_sig.clone();
                                    let mut want = ref_sig.clone();
                                    got.sort_by(|a, b| a.0.cmp(&b.0));
                                    want.sort_by(|a, b| a.0.cmp(&b.0));
                                    if got != want {
                                        return self.err(format!(
                                            "or-pattern alternative {alt_idx} binds {got:?} but \
                                             alternative 0 binds {ref_sig:?} — every alternative \
                                             of an or-pattern must bind the same variable names at \
                                             the same types (RFC-0020 §9 / R20-Q3, never-silent G2)"
                                        ));
                                    }
                                }
                            }
                            // Expand into a plain arm sharing the body.
                            acc.push(crate::ast::Arm {
                                pattern: resolved,
                                body: arm.body.clone(),
                            });
                        }
                    }
                    _ => acc.push(arm.clone()),
                }
                Ok(acc)
            })?;

        let col = [sty.clone()];
        let mut rows: Vec<Vec<crate::usefulness::Pat>> = Vec::new();
        let mut result: Option<Ty> = None;
        let mut arms2: Vec<crate::ast::Arm> = Vec::with_capacity(flat_arms.len());
        // M-903 / DN-71 §4.2: every arm is a mutually exclusive alternative at runtime, so each is
        // checked from the *same* pre-match affine snapshot; `merged` accumulates the union of every
        // arm's post-body outcome (a slot moved in *any* arm counts as moved afterward —
        // conservative, `crate::affine` module docs). Restored as the tracker's live state once the
        // whole match has been checked.
        let pre_match = self.affine.snapshot();
        let mut merged = pre_match.clone();
        for arm in &flat_arms {
            // Normalize the pattern against the scrutinee/field types first — resolve ambient
            // bare-decimal literals to concrete ones, and rewrite nullary-ctor idents to explicit
            // `Ctor(name, [])` — so the matrix, the evaluator, the elaborator, and the type-free
            // grading/totality passes all see one canonical, unambiguous checked pattern.
            let pattern = self.resolve_pattern(&arm.pattern, &sty)?;
            // Type the (possibly nested) pattern against the scrutinee type, collecting its binders.
            let mut binds: Vec<(String, Ty, Vec<usize>)> = Vec::new();
            let pat = self.check_pattern(&pattern, &sty, &mut binds)?;
            self.check_linear(&binds)?;
            // Redundancy (W7): an arm covered by the earlier rows is unreachable.
            if crate::usefulness::useful(self.types, &rows, std::slice::from_ref(&pat), &col)
                .map_err(|e| self.match_budget_err(e))?
                .is_none()
            {
                return self.err(
                    "this arm is unreachable — earlier arms already cover it (W7: every arm must \
                     be reachable)",
                );
            }
            // Type the body with the pattern's binders in scope. Restore the shared pre-match affine
            // baseline first (undoes any Moved marks a *previous* arm left on an outer binding — arms
            // are alternatives, not a sequence), then push this arm's own binder slots (a
            // `Substrate`-typed field capture is tracked exactly like a `let` binder — DN-71 §4.2).
            let depth = scope.len();
            self.affine.restore(&pre_match);
            for (name, ty, _occ) in &binds {
                scope.push((name.clone(), ty.clone()));
                self.affine.push(ty);
            }
            let body_expected = expected.or(result.as_ref());
            let r = self.check(scope, &arm.body, body_expected);
            scope.truncate(depth);
            self.affine.truncate(depth);
            let (bty, body2) = r?;
            // Union-merge this arm's (now-truncated, pre-match-depth) outcome into `merged`.
            if let (Some(m), Some(cur)) = (merged.as_mut(), self.affine.snapshot()) {
                crate::affine::union_merge_into(m, &cur);
            }
            match &result {
                None => result = Some(bty),
                Some(r) if *r != bty => {
                    return self.err(format!(
                        "match arms disagree: {}",
                        edge_mismatch("arm", r, &bty)
                    ))
                }
                Some(_) => {}
            }
            rows.push(vec![pat]);
            arms2.push(crate::ast::Arm {
                pattern,
                body: body2,
            });
        }
        // Install the union-merged post-match affine state as the tracker's live state, so any
        // `Substrate` binding moved in at least one arm is treated as moved by whatever follows this
        // `match` (M-903 / DN-71 §4.2).
        self.affine.restore(&merged);
        // Exhaustiveness (W7): a wildcard must not be useful — else its witness is a missing case.
        if let Some(witness) =
            crate::usefulness::useful(self.types, &rows, &[crate::usefulness::Pat::Wild], &col)
                .map_err(|e| self.match_budget_err(e))?
        {
            return self.err(format!(
                "non-exhaustive match on {sty}: missing {} (W7 — coverage is checked, never assumed)",
                crate::usefulness::render(&witness[0])
            ));
        }
        // Codegen half (M-320; RFC-0007 §3, "compiled away by the elaborator"): compile the checked
        // match to a Maranget decision tree and confirm it is **Fail-free** — an exhaustive match must
        // compile to total coverage, so the usefulness analysis and the tree compiler must agree
        // (defense in depth; never silent if they don't). The tree is the flat-tests form; emitting
        // its leaves as L0 kernel nodes awaits the RFC-0001 revision (RFC-0007 §4.6).
        let arm_ix: Vec<usize> = (0..rows.len()).collect();
        let occ = [Vec::<usize>::new()];
        let tree = crate::decision::compile(self.types, &rows, &arm_ix, &occ, &col)
            .map_err(|e| self.match_budget_err(e))?;
        if crate::decision::has_reachable_fail(&tree) {
            return self.err(
                "internal: an exhaustive match compiled to a decision tree with a reachable Fail \
                 (the usefulness check and the Maranget compiler disagree — report this)",
            );
        }
        let rty =
            result.ok_or_else(|| CheckError::new(self.site, "a match needs at least one arm"))?;
        Ok((
            rty,
            Expr::Match {
                scrutinee: Box::new(scrut2),
                arms: arms2,
            },
        ))
    }

    /// Normalize a surface [`Pattern`] against its `expected` type into the **canonical checked
    /// form** stored in the resolved AST, type-directed at every position (the scrutinee type at the
    /// root, each constructor field's type as it recurses):
    ///
    /// 1. **Ambient literals** — resolve a bare-decimal (`AmbientInt`) pattern to a concrete
    ///    `Binary`/`Ternary` literal at the width `expected` pins. A literal under a
    ///    non-repr/cross-paradigm position is left unchanged so [`normalize_pattern`] raises the
    ///    precise W7 error.
    /// 2. **Nullary-ctor idents** — rewrite a bare `Pattern::Ident(name)` that names a *nullary
    ///    constructor of the scrutinee data type* into an explicit `Pattern::Ctor(name, vec![])`.
    ///    This makes the checked AST **unambiguous**: a residual `Pattern::Ident` is always a true
    ///    binder, a `Pattern::Ctor` always a constructor — so the type-free downstream passes
    ///    (guarantee grading Pass 3d, totality Pass 4) need no type information to tell them apart.
    ///    The checker (which alone knows the *expected scrutinee type*) is the single source of truth
    ///    for this resolution, mirroring [`normalize_pattern`]; a binder whose name merely collides
    ///    with a nullary ctor of an *unrelated* type stays a binder (no global ctor scan — that
    ///    over-broad scan was an unsound grade-upgrade, M-663 / Copilot review).
    fn resolve_pattern(&self, pat: &Pattern, expected: &Ty) -> Result<Pattern, CheckError> {
        Ok(match pat {
            Pattern::Lit(Literal::AmbientInt(p, v)) => {
                Pattern::Lit(self.resolve_ambient_int(*p, *v, Some(expected))?)
            }
            // A bare name is a nullary-constructor pattern iff it names a nullary ctor of the
            // *scrutinee's own* data type; otherwise it is a binder (left as `Ident`).
            Pattern::Ident(name)
                if matches!(expected, Ty::Data(tn, _)
                    if self.types.get(tn).is_some_and(|d|
                        d.ctors.iter().any(|c| c.name == *name && c.fields.is_empty()))) =>
            {
                Pattern::Ctor(name.clone(), vec![])
            }
            Pattern::Ctor(name, subs) => {
                // Recurse with each sub-pattern's field type, when the expected type is the owning
                // data type and the constructor/arity line up; otherwise leave `subs` for the
                // normalizer to diagnose.
                let field_tys = match expected {
                    // The declared field types are abstract over the type's parameters; substitute
                    // the scrutinee's type arguments so a generic field recurses at its concrete
                    // type (RFC-0007 §11.2).
                    Ty::Data(tn, targs) => self.types.get(tn).and_then(|d| {
                        d.ctors
                            .iter()
                            .find(|c| c.name == *name)
                            .filter(|c| c.fields.len() == subs.len())
                            .map(|c| {
                                let s = param_subst(&d.params, targs);
                                c.fields.iter().map(|f| subst_ty(f, &s)).collect::<Vec<_>>()
                            })
                    }),
                    _ => None,
                };
                let mut out = Vec::with_capacity(subs.len());
                for (i, s) in subs.iter().enumerate() {
                    match &field_tys {
                        Some(fts) => out.push(self.resolve_pattern(s, &fts[i])?),
                        None => out.push(s.clone()),
                    }
                }
                Pattern::Ctor(name.clone(), out)
            }
            // M-826: a tuple pattern `(x, y, …)` desugars to a single-constructor `Ctor` pattern
            // on the synthetic `Tuple$N` type — identical to any user-declared product constructor.
            // Never-silent (G2): arity mismatch vs. the expected scrutinee type is an explicit error.
            Pattern::Tuple(subs) => {
                let n = subs.len();
                let ctor_name = tuple_ctor_name(n);
                let tname = tuple_type_name(n);
                // The expected type must be the matching Tuple$N.
                let field_tys = match expected {
                    Ty::Data(tn, targs) if *tn == tname => self.types.get(tn).and_then(|d| {
                        d.ctors
                            .iter()
                            .find(|c| c.name == ctor_name)
                            .filter(|c| c.fields.len() == subs.len())
                            .map(|c| {
                                let s = param_subst(&d.params, targs);
                                c.fields.iter().map(|f| subst_ty(f, &s)).collect::<Vec<_>>()
                            })
                    }),
                    _ => None,
                };
                let mut out = Vec::with_capacity(subs.len());
                for (i, s) in subs.iter().enumerate() {
                    let resolved = match &field_tys {
                        Some(fts) => self.resolve_pattern(s, &fts[i])?,
                        None => s.clone(),
                    };
                    out.push(resolved);
                }
                Pattern::Ctor(ctor_name, out)
            }
            Pattern::Wildcard | Pattern::Lit(_) | Pattern::Ident(_) => pat.clone(),
            // `Pattern::Or` is desugared in `check_match` before `resolve_pattern` is called;
            // reaching here means the invariant was violated — an explicit never-silent refusal (G2).
            Pattern::Or(_) => {
                return Err(CheckError::new(
                    self.site,
                    "internal: Pattern::Or reached resolve_pattern — or-patterns must be desugared \
                     in check_match before any downstream pass (invariant violation — report this)",
                ));
            }
        })
    }

    /// Resolve a bare decimal (`AmbientInt`) to a concrete `Binary`/`Ternary` literal at the width
    /// the `expected` type pins (RFC-0012 §4.3). Never a built-in default: an absent or
    /// cross-paradigm context is an explicit refusal.
    fn resolve_ambient_int(
        &self,
        p: Paradigm,
        v: i64,
        expected: Option<&Ty>,
    ) -> Result<Literal, CheckError> {
        match (p, expected) {
            (Paradigm::Binary, Some(Ty::Binary(Width::Lit(w)))) => encode_binary(self.site, v, *w),
            (Paradigm::Ternary, Some(Ty::Ternary(Width::Lit(w)))) => encode_balanced_ternary(self.site, v, *w),
            (_, Some(other)) => self.err(format!(
                "a bare `{p}` decimal cannot fill a {other} context — {} (RFC-0012 §4.3)",
                match paradigm_name(other) {
                    Some(o) => format!("`{p}` and `{o}` are different paradigms; write an explicit \
                                        `swap` or a tagged literal"),
                    None => "a bare decimal only resolves in a Binary/Ternary context".to_owned(),
                }
            )),
            (_, None) => self.err(format!(
                "UnresolvedWidth: a bare `{p}` decimal `{v}` has no width here — its width must come \
                 from the checked context (an ascription, a parameter/return type, or another \
                 operand). Ascribe it or write it explicitly (RFC-0012 §4.3; never a default width)"
            )),
        }
    }

    /// Type-check `pat` against `expected`, accumulating its binders (`name: ty @ occurrence`) into
    /// `binds`, and return the normalized [`crate::usefulness::Pat`] for the coverage matrix.
    /// Delegates to the free [`normalize_pattern`] (shared with the elaborator), starting at the root
    /// occurrence `[]`.
    fn check_pattern(
        &self,
        pat: &Pattern,
        expected: &Ty,
        binds: &mut Vec<(String, Ty, Vec<usize>)>,
    ) -> Result<crate::usefulness::Pat, CheckError> {
        normalize_pattern(self.types, self.site, pat, expected, &[], binds)
    }

    /// A pattern must bind each name at most once (linearity) — a repeated binder is ambiguous, so it
    /// is an explicit error rather than a silent last-wins.
    fn check_linear(&self, binds: &[(String, Ty, Vec<usize>)]) -> Result<(), CheckError> {
        for (i, (n, _, _)) in binds.iter().enumerate() {
            if binds[..i].iter().any(|(m, _, _)| m == n) {
                return self.err(format!(
                    "pattern binds `{n}` more than once (bindings must be linear)"
                ));
            }
        }
        Ok(())
    }

    /// Check a list literal `[e1, …]` into a [`Ty::Seq`] (RFC-0032 D3; M-749). The element type is
    /// **inferred from the elements**, which must be **homogeneous** (a mismatch is a never-silent
    /// [`CheckError`], never a silent coercion — G2); the length is the element count. When an
    /// `expected` [`Ty::Seq`] type is supplied, each element is checked **against** its element type
    /// and the count must match the expected length (never-silent). An empty `[]` against a
    /// `Seq{T, 0}` is the well-formed empty sequence; an empty `[]` with **no** expected `Seq` type
    /// has no inferable element type and is an explicit refusal (G2 — never a guessed element).
    ///
    /// Returns the resolved literal so any ambient bare-decimal element (resolved against the
    /// element type) is rewritten to a concrete `Binary`/`Ternary` form for the evaluator/elaborator.
    fn check_list(
        &self,
        scope: &mut Vec<(String, Ty)>,
        elems: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<(Ty, Expr), CheckError> {
        // RFC-0040 (M-977): a `[e1, …, en]` literal checked against a **cons-list-shaped** user ADT
        // (exactly two constructors — one nullary "nil" and one binary `Cons(A, Self)` "cons")
        // DESUGARS to the right-nested `Cons(e1, Cons(…, Nil))` chain and is checked as that chain.
        // The desugared expression IS the hand-written chain (same surface AST after desugaring), so
        // elaboration/eval/AOT see byte-for-byte the current representation — behaviour-neutral by
        // construction (the three-property gate: flatten ≡ lower ≡ semantics unchanged). The Seq
        // literal path (below) is unchanged: `Seq{T, N}` is a distinct expected type, so there is no
        // ambiguity. When there is NO expected type, a bare `[…]` still infers a `Seq` as before.
        if let Some(exp) = expected {
            if let Some((nil_ctor, cons_ctor)) = cons_list_ctors(self.types, exp) {
                // RFC-0041 §4.7 (W1, RR-29): the desugared `Cons(e1, Cons(…, Nil))` chain is `n`
                // *data*-levels deep, so checking it via `self.check` walks it with `n` levels of
                // genuine Rust CONTROL recursion — a 4096-element list is then wrongly refused by the
                // control-depth budget (`MAX_CHECK_DEPTH`) as if it were 4096-deep control recursion,
                // and an even larger one risks a host-stack overflow. Route the list's *data*-spine
                // depth through ITERATION instead: a flat loop that charges a work-step (not a control
                // depth level) per element (§4.7 data-vs-control fix), when `exp` is CONCRETE (the
                // corpus). Behaviour is byte-identical to the desugared-chain check — each element is
                // checked against the same element type, and the returned resolved `Cons` chain is the
                // same AST (§the three-property gate). When `exp`'s element type is still ABSTRACT (a
                // generic body — never a large in-corpus list literal), keep the exact original
                // recursive desugar path so its checking behaviour is preserved verbatim.
                let concrete = matches!(exp, Ty::Data(_, args) if !args.iter().any(has_var));
                if concrete {
                    return self.check_cons_list(scope, elems, exp, &nil_ctor, &cons_ctor);
                }
                let mut acc = Expr::Path(Path(vec![nil_ctor]));
                for e in elems.iter().rev() {
                    acc = Expr::App {
                        head: Box::new(Expr::Path(Path(vec![cons_ctor.clone()]))),
                        args: vec![e.clone(), acc],
                    };
                }
                return self.check(scope, &acc, expected);
            }
        }
        // The expected element type + length, when checking against a `Seq{T, N}`.
        let expected_elem: Option<Ty> = match expected {
            Some(Ty::Seq(elem, _)) => Some((**elem).clone()),
            _ => None,
        };
        if let Some(Ty::Seq(_, n)) = expected {
            if *n as usize != elems.len() {
                return self.err(format!(
                    "list literal has {} element(s) but the expected `Seq` length is {} \
                     (RFC-0032 D3 — never a silent truncation/padding)",
                    elems.len(),
                    n
                ));
            }
        }
        let mut elem_ty: Option<Ty> = expected_elem.clone();
        let mut rebuilt = Vec::with_capacity(elems.len());
        for (i, e) in elems.iter().enumerate() {
            // Drive each element against the running element type (expected or the first element's
            // inferred type), so a bare-decimal element takes that width; the first element with no
            // expected type synthesizes its own type and anchors the rest.
            let (t, e2) = self.check(scope, e, elem_ty.as_ref())?;
            match &elem_ty {
                Some(want) if *want != t => {
                    return self.err(format!(
                        "list literal element {i} has type {t}, but the sequence's element type is \
                         {want} — list elements must be homogeneous (RFC-0032 D3, never a silent \
                         coercion)"
                    ));
                }
                Some(_) => {}
                None => elem_ty = Some(t),
            }
            rebuilt.push(e2);
        }
        let Some(elem) = elem_ty else {
            return self.err(
                "an empty list literal `[]` has no inferable element type here — ascribe it to a \
                 `Seq{T, 0}` (RFC-0032 D3, never a guessed element type)",
            );
        };
        let len = u32::try_from(rebuilt.len()).map_err(|_| {
            CheckError::new(self.site, "list literal length exceeds u32 (RFC-0032 D3)")
        })?;
        Ok((
            Ty::Seq(Box::new(elem), len),
            Expr::Lit(Literal::List(rebuilt)),
        ))
    }

    /// Check a list literal `[e1, …, en]` against a **concrete** cons-list ADT (`exp`, a two-ctor
    /// nil/`Cons(A, Self)` type — RFC-0040/M-977), routing the list's *data*-spine depth through
    /// ITERATION rather than control recursion (RFC-0041 §4.7, the data-vs-control fix).
    ///
    /// The desugared value is still the right-nested `Cons(e1, Cons(…, Nil))` chain (so elaboration /
    /// eval / AOT see byte-for-byte the RFC-0040 representation), but it is **built and checked in a
    /// flat loop**: each element is checked against the constructor's (already fully-determined)
    /// element type via a single, shallow `Cons(e, <typed placeholder tail>)` check — so the checker's
    /// control-recursion depth stays O(1) per element and a large list is bounded by the work-step
    /// budget, never refused as if it were `n`-deep control recursion. Behaviour is byte-identical to
    /// the desugared-chain check for the concrete `exp` case (each element's expected type comes from
    /// `exp` alone, not the tail): the produced resolved elements and the produced chain match exactly.
    ///
    /// Never-silent (G2/VR-5): a heterogeneous / mistyped element surfaces the same explicit
    /// `CheckError` the desugared-chain check would (both flow through `check_app`'s constructor path).
    fn check_cons_list(
        &self,
        scope: &mut Vec<(String, Ty)>,
        elems: &[Expr],
        exp: &Ty,
        nil_ctor: &str,
        cons_ctor: &str,
    ) -> Result<(Ty, Expr), CheckError> {
        // A per-list work-step budget (RFC-0041 §4.7): the data-spine width is charged here, NOT as
        // control depth. The default ceiling leaves the work-step limit effectively unbounded in W1
        // (the real memory/work-step ceilings are wired in W2 — mycelium-workstack §4.2); the charge
        // is wired now so the never-silent bounding machinery is in place. A `Cons` head expr, rebuilt
        // once (its `Path` is cloned per layer for the resolved chain).
        let budget = mycelium_workstack::RecursionBudget::default();
        let cons_head = || Expr::Path(Path(vec![cons_ctor.to_owned()]));
        // Resolve the innermost `Nil` once (its resolved form is the chain's tail) — the same check the
        // desugared chain performs at its deepest level.
        let (_nil_ty, nil_expr) = self.check(
            scope,
            &Expr::Path(Path(vec![nil_ctor.to_owned()])),
            Some(exp),
        )?;
        // A synthetic, typed placeholder for the (already-checked) `Cons` tail. Its name uses `$` so it
        // can never collide with — or be referenced by — a surface identifier (`$` is not lexable in an
        // ident), and it is truncated away before returning so it never leaks into the caller's scope.
        // It is pushed onto BOTH `scope` and the affine tracker (kept index-for-index parallel — a
        // `check_path` reference resolves its affine slot by scope index; a non-`Substrate` slot is a
        // `Skip`, so this placeholder is affine-inert), then both are truncated back on every exit.
        const TAIL_SLOT: &str = "$cons_list_tail$";
        let base = scope.len();
        let tail_ref = Expr::Path(Path(vec![TAIL_SLOT.to_owned()]));
        scope.push((TAIL_SLOT.to_owned(), exp.clone()));
        self.affine.push(exp);
        let mut resolved_elems = Vec::with_capacity(elems.len());
        let mut result_ty: Option<Ty> = None;
        for e in elems {
            // Charge one work-step for this element (data-spine width — never a control-depth level).
            if let Err(be) = budget.charge_steps(1) {
                scope.truncate(base);
                self.affine.truncate(base);
                return self.err(format!(
                    "list literal exceeded the work-step budget: {be} — an explicit over-budget \
                     refusal (RFC-0041 §4.7), never a host-stack overflow (G2/VR-5)"
                ));
            }
            // One shallow `Cons(e, $tail)` check: the element is checked against the constructor's
            // element type (fixed by `exp`), the placeholder tail supplies the recursive `Self` field
            // at its known type. Depth is O(1), independent of the list length.
            let layer = Expr::App {
                head: Box::new(cons_head()),
                args: vec![e.clone(), tail_ref.clone()],
            };
            let (lty, lresolved) = match self.check(scope, &layer, Some(exp)) {
                Ok(x) => x,
                Err(err) => {
                    scope.truncate(base);
                    self.affine.truncate(base);
                    return Err(err);
                }
            };
            // Extract this element's resolved form (the `Cons` app's first argument).
            let elem_resolved = match lresolved {
                Expr::App { mut args, .. } if !args.is_empty() => args.swap_remove(0),
                other => other,
            };
            resolved_elems.push(elem_resolved);
            result_ty = Some(lty);
        }
        scope.truncate(base);
        self.affine.truncate(base);
        // Rebuild the right-nested resolved chain `Cons(e1', Cons(…, Nil))` from the resolved elements.
        let mut acc = nil_expr;
        for e in resolved_elems.into_iter().rev() {
            acc = Expr::App {
                head: Box::new(cons_head()),
                args: vec![e, acc],
            };
        }
        // The whole-expression type is the checked ADT type (`exp` for the concrete case, exactly as
        // the outermost `Cons` of the desugared chain would report); an empty `[]` is that `exp`.
        let ty = result_ty.unwrap_or_else(|| exp.clone());
        Ok((ty, acc))
    }

    /// Type a `Seq`/`Bytes` indexing/length/slice/concat/eq/hash prim (RFC-0032 D3/D4; M-749/M-750/
    /// M-799; M-912), or `Ok(None)` if `name` is not one of them (so the caller falls through to the
    /// width-preserving prim path).
    /// The operands are checked, the receiver's repr type is verified (a non-`Seq`/`Bytes` receiver,
    /// or a wrong arity, is an explicit never-silent refusal — G2), and the result type is returned.
    fn try_check_seq_bytes_prim(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        args: &[Expr],
    ) -> Result<Option<(Ty, Expr)>, CheckError> {
        // The index argument's type (any `Binary{W}`) is checked; a bare-decimal index has no anchor
        // (the prim's result is not the index width), so it must be explicit.
        let check_index = |scope: &mut Vec<(String, Ty)>, a: &Expr| -> Result<Expr, CheckError> {
            let (ity, a2) = self.check(scope, a, None)?;
            match ity {
                Ty::Binary(_) => Ok(a2),
                _ => self.err(format!(
                    "`{name}` index must be a `Binary{{W}}` (unsigned magnitude), got {ity} \
                     (RFC-0032 D3/D4)"
                )),
            }
        };
        let arity_err = |want: usize| -> Result<Option<(Ty, Expr)>, CheckError> {
            self.err(format!(
                "`{name}` takes {want} operand(s), got {} (RFC-0032 D3/D4)",
                args.len()
            ))
        };
        match name {
            "seq_len" | "bytes_len" => {
                if args.len() != 1 {
                    return arity_err(1);
                }
                let (recv, recv2) = self.check(scope, &args[0], None)?;
                let ok = match name {
                    "seq_len" => matches!(recv, Ty::Seq(_, _)),
                    _ => matches!(recv, Ty::Bytes),
                };
                if !ok {
                    return self.err(format!(
                        "`{name}` expects a {} operand, got {recv} (RFC-0032 D3/D4)",
                        if name == "seq_len" { "Seq" } else { "Bytes" }
                    ));
                }
                // Length is an unsigned 32-bit count (matches `prims.rs`).
                Ok(Some((
                    Ty::Binary(Width::Lit(32)),
                    app_node(head, vec![recv2]),
                )))
            }
            "seq_get" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (recv, recv2) = self.check(scope, &args[0], None)?;
                let Ty::Seq(elem, _) = recv else {
                    return self.err(format!(
                        "`seq_get` expects a `Seq` operand, got {recv} (RFC-0032 D3)"
                    ));
                };
                let idx2 = check_index(scope, &args[1])?;
                // The result is the element type (out-of-bounds is a runtime refusal, not a type error).
                Ok(Some((*elem, app_node(head, vec![recv2, idx2]))))
            }
            "bytes_get" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (recv, recv2) = self.check(scope, &args[0], None)?;
                if !matches!(recv, Ty::Bytes) {
                    return self.err(format!(
                        "`bytes_get` expects a `Bytes` operand, got {recv} (RFC-0032 D4)"
                    ));
                }
                let idx2 = check_index(scope, &args[1])?;
                // A byte is a `Binary{8}` value.
                Ok(Some((
                    Ty::Binary(Width::Lit(8)),
                    app_node(head, vec![recv2, idx2]),
                )))
            }
            // DN-43 (M-799): `bytes_slice(b: Bytes, start: Binary{W}, end: Binary{W}) -> Bytes` — the
            // never-silent half-open sub-slice `[start, end)` (RFC-0032 D4; the kernel prim already
            // exists). The receiver must be `Bytes`; both bounds are any `Binary{W}` (an unsigned
            // magnitude, checked via `check_index`). An out-of-range or inverted range is a never-silent
            // *runtime* refusal (`prims.rs::prim_bytes_slice`), never a static error and never a silent
            // clamp (G2) — so the static rule only fixes the operand/result shapes.
            "bytes_slice" => {
                if args.len() != 3 {
                    return arity_err(3);
                }
                let (recv, recv2) = self.check(scope, &args[0], None)?;
                if !matches!(recv, Ty::Bytes) {
                    return self.err(format!(
                        "`bytes_slice` expects a `Bytes` operand, got {recv} (RFC-0032 D4)"
                    ));
                }
                let start2 = check_index(scope, &args[1])?;
                let end2 = check_index(scope, &args[2])?;
                Ok(Some((Ty::Bytes, app_node(head, vec![recv2, start2, end2]))))
            }
            // DN-43 (M-799): `bytes_concat(b1: Bytes, b2: Bytes) -> Bytes` — total byte concatenation
            // (RFC-0032 D4; the kernel prim already exists). Both operands must be `Bytes`; a non-`Bytes`
            // operand is an explicit static refusal (G2).
            "bytes_concat" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (a, a2) = self.check(scope, &args[0], None)?;
                if !matches!(a, Ty::Bytes) {
                    return self.err(format!(
                        "`bytes_concat` expects a `Bytes` first operand, got {a} (RFC-0032 D4)"
                    ));
                }
                let (b, b2) = self.check(scope, &args[1], None)?;
                if !matches!(b, Ty::Bytes) {
                    return self.err(format!(
                        "`bytes_concat` expects a `Bytes` second operand, got {b} (RFC-0032 D4)"
                    ));
                }
                Ok(Some((Ty::Bytes, app_node(head, vec![a2, b2]))))
            }
            // DN-41 (M-798): `width_cast(value: Binary{N}, into: Binary{M}) -> Binary{M}` — the
            // never-silent `Binary` re-width. The second operand is a **width witness**: only its
            // `Binary{M}` width is read (its value drives nothing — the runtime ignores its bits), and
            // `M` is the result width. Both operands must be `Binary` (a non-`Binary` is an explicit
            // type refusal — G2). Neither operand is anchored by the *expected* type: the witness
            // already pins `M`, and the value operand must be a concrete `Binary{N}` (a bare decimal
            // has no anchor here — the result is the witness width, not the value width — so it is
            // refused, never defaulted). Whether the cast is exact (widen/identity) or fits (narrow)
            // is a runtime contract, not a static one — narrowing overflow is a never-silent runtime
            // refusal (mirroring `add_u`/`sub_u`), so the static rule only fixes the result width.
            "width_cast" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (vty, v2) = self.check(scope, &args[0], None)?;
                if !matches!(vty, Ty::Binary(_)) {
                    return self.err(format!(
                        "`width_cast` value operand must be a concrete `Binary{{N}}`, got {vty} \
                         (DN-41; never a default width)"
                    ));
                }
                let (wty, w2) = self.check(scope, &args[1], None)?;
                let Ty::Binary(Width::Lit(m)) = wty else {
                    return self.err(format!(
                        "`width_cast` width witness must be a concrete `Binary{{M}}` (only its width is used), \
                         got {wty} (DN-41; a width-variable witness is refused — DN-42/M-753)"
                    ));
                };
                // The result is `Binary{M}` (the witness width); the value→M fit is a runtime contract.
                Ok(Some((
                    Ty::Binary(Width::Lit(m)),
                    app_node(head, vec![v2, w2]),
                )))
            }
            // M-912 (`enb`, folded-in gap): `bytes_eq(a: Bytes, b: Bytes) -> Binary{1}` — byte-wise
            // equality, flagged missing by the diag/error/recover ports (`bytes.*` had len/get/
            // slice/concat but no equality). Both operands must be `Bytes` (a non-`Bytes` operand is
            // an explicit static refusal — G2); the result is the realized `Bool` truth value, total
            // over every pair (`prims.rs::prim_bytes_eq` is a straight `[u8]` comparison).
            "bytes_eq" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (a, a2) = self.check(scope, &args[0], None)?;
                if !matches!(a, Ty::Bytes) {
                    return self.err(format!(
                        "`bytes_eq` expects a `Bytes` first operand, got {a} (M-912)"
                    ));
                }
                let (b, b2) = self.check(scope, &args[1], None)?;
                if !matches!(b, Ty::Bytes) {
                    return self.err(format!(
                        "`bytes_eq` expects a `Bytes` second operand, got {b} (M-912)"
                    ));
                }
                Ok(Some((
                    Ty::Binary(Width::Lit(1)),
                    app_node(head, vec![a2, b2]),
                )))
            }
            // M-912 (`enb`): `hash_blake3(b: Bytes) -> Bytes` — the kernel's own BLAKE3
            // content-addressing hash (M-103) surfaced as a prim: the 32-byte digest of the input
            // byte string. The operand must be `Bytes` (a non-`Bytes` operand is an explicit static
            // refusal — G2); total and deterministic over every `Bytes` input, so there is no
            // runtime failure domain (unlike `bytes_get`/`bytes_slice`).
            "hash_blake3" => {
                if args.len() != 1 {
                    return arity_err(1);
                }
                let (recv, recv2) = self.check(scope, &args[0], None)?;
                if !matches!(recv, Ty::Bytes) {
                    return self.err(format!(
                        "`hash_blake3` expects a `Bytes` operand, got {recv} (M-912)"
                    ));
                }
                Ok(Some((Ty::Bytes, app_node(head, vec![recv2]))))
            }
            _ => Ok(None),
        }
    }

    /// Try to check a call to a **dense prim** (RFC-0001 §4.1 / RFC-0002 §5; M-890/M-891,
    /// `enb` Gap C) — the tensor-valued group `dense_add`/`dense_sub`/`dense_neg`/`dense_scale`
    /// (kernel `dense.add`/`dense.sub`/`dense.neg`/`dense.scale`, the `mycelium-dense` surface)
    /// plus the measurement pair `dense_dot`/`dense_similarity` (kernel `dense.dot`/
    /// `dense.similarity`, M-891). Returns `Ok(None)` if `name` is not one of them.
    ///
    /// Dim and dtype are part of the type (`Dense{d, s}`), so the never-silent shape contract is
    /// **static** here: `dense_add`/`dense_sub` require two `Dense{d, s}` operands with *equal*
    /// dim and dtype (a mismatch is an explicit refusal naming both types — never a broadcast or
    /// re-round; G2), and the result is the same `Dense{d, s}`. `dense_neg` is unary,
    /// dim/dtype-preserving. `dense_scale(a: Dense{d, s}, c: Dense{1, s})` takes its factor as a
    /// **`Dense{1, s}` scalar** of the *same* dtype — the pre-Gap-A scalar form (no scalar-float
    /// type exists until the M-895/M-896 float ADR lands; FLAGged at the kernel wrapper,
    /// `mycelium-interp/src/prims.rs`). `dense_dot`/`dense_similarity` (M-891) take the same
    /// equal-dim/dtype operand pair and return **`Dense{1, F64}`** — the measurement-result form:
    /// the binary64 the kernel computed, delivered exactly (never re-rounded onto the operand
    /// grid), carrying the kernel's `Proven` absolute accumulation bound at runtime; `F64` has no
    /// dense op set (kernel v1 scope), so a measurement cannot silently feed back into dense
    /// arithmetic — re-entry is an explicit runtime `UnsupportedDtype` refusal (G2). The
    /// numeric-domain contracts (off-grid/non-finite
    /// elements, overflow, subnormal results, approximate sources) are *runtime* refusals owned by
    /// the kernel, not static rules — exactly as `add_u`'s overflow is (RFC-0032 D2 precedent).
    /// A bare-decimal operand has no Dense anchor (RFC-0012 §4.3 gives Dense no bare-decimal
    /// encoding), so it is refused by the operand check itself, never defaulted.
    fn try_check_dense_prim(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        args: &[Expr],
    ) -> Result<Option<(Ty, Expr)>, CheckError> {
        if !matches!(
            name,
            "dense_add"
                | "dense_sub"
                | "dense_neg"
                | "dense_scale"
                | "dense_dot"
                | "dense_similarity"
        ) {
            return Ok(None);
        }
        let arity_err = |want: usize| -> Result<Option<(Ty, Expr)>, CheckError> {
            self.err(format!(
                "`{name}` takes {want} operand(s), got {} (RFC-0001 §4.1; M-890)",
                args.len()
            ))
        };
        // Every operand must be a concrete `Dense{d, s}`; check the first and destructure.
        let check_dense = |scope: &mut Vec<(String, Ty)>,
                           a: &Expr,
                           which: &str|
         -> Result<(u32, Scalar, Expr), CheckError> {
            let (ty, a2) = self.check(scope, a, None)?;
            let Ty::Dense(d, s) = ty else {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` {which} operand must be a `Dense{{dim, scalar}}`, got {ty} \
                         (RFC-0001 §4.1; M-890 — never a silent conversion; a cross-paradigm \
                         edge needs an explicit `swap`)"
                    ),
                ));
            };
            Ok((d, s, a2))
        };
        match name {
            "dense_add" | "dense_sub" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (da, sa, a2) = check_dense(scope, &args[0], "first")?;
                let (db, sb, b2) = check_dense(scope, &args[1], "second")?;
                if da != db || sa != sb {
                    // The never-silent shape contract, statically: name both full types.
                    return self.err(format!(
                        "`{name}` operands must share one dim and dtype, got Dense{{{da}, \
                         {sa:?}}} and Dense{{{db}, {sb:?}}} (M-890 — a shape/dtype mismatch is \
                         an explicit refusal, never a broadcast or re-round; G2)"
                    ));
                }
                Ok(Some((Ty::Dense(da, sa), app_node(head, vec![a2, b2]))))
            }
            "dense_dot" | "dense_similarity" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (da, sa, a2) = check_dense(scope, &args[0], "first")?;
                let (db, sb, b2) = check_dense(scope, &args[1], "second")?;
                if da != db || sa != sb {
                    return self.err(format!(
                        "`{name}` operands must share one dim and dtype, got Dense{{{da}, \
                         {sa:?}}} and Dense{{{db}, {sb:?}}} (M-891 — a shape/dtype mismatch is \
                         an explicit refusal, never a broadcast or re-round; G2)"
                    ));
                }
                // The measurement-result form (M-891): Dense{1, F64} — the binary64 the kernel
                // computed, exactly; its Proven accumulation bound rides the runtime value.
                Ok(Some((
                    Ty::Dense(1, Scalar::F64),
                    app_node(head, vec![a2, b2]),
                )))
            }
            "dense_neg" => {
                if args.len() != 1 {
                    return arity_err(1);
                }
                let (d, s, a2) = check_dense(scope, &args[0], "single")?;
                Ok(Some((Ty::Dense(d, s), app_node(head, vec![a2]))))
            }
            "dense_scale" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (d, s, a2) = check_dense(scope, &args[0], "vector")?;
                let (dc, sc, c2) = check_dense(scope, &args[1], "scale-factor")?;
                if dc != 1 {
                    return self.err(format!(
                        "`dense_scale` factor must be a `Dense{{1, scalar}}` (the pre-Gap-A \
                         scalar form), got Dense{{{dc}, {sc:?}}} (M-890)"
                    ));
                }
                if sc != s {
                    return self.err(format!(
                        "`dense_scale` factor dtype {sc:?} must match the vector dtype {s:?} \
                         (M-890 — never a silent re-round; G2)"
                    ));
                }
                Ok(Some((Ty::Dense(d, s), app_node(head, vec![a2, c2]))))
            }
            _ => unreachable!("gated by the matches! above"),
        }
    }

    /// Try to check a call to a **VSA prim** (RFC-0003 §3/§4/§5/§6; ADR-008) — the M-892 bind
    /// group `vsa_bind`/`vsa_unbind` (binary, hypervector × hypervector) and `vsa_permute`
    /// (hypervector × `Binary{W}` shift), kernel `vsa.bind`/`vsa.unbind`/`vsa.permute`, plus the
    /// M-893 **certified superposition** `vsa_bundle` (`Seq{VSA{…}, N≥1}` × `Float` δ →
    /// `VSA{…}`), kernel `vsa.bundle`, plus the M-894 **cleanup/reconstruction pair and the
    /// capacity query** — `vsa_cleanup` (`VSA{…}` × `Seq{VSA{…}, N≥1}` → `Seq{Float, 3}`,
    /// the `[index, confidence, margin]` decision triple), `vsa_reconstruct`
    /// (`VSA{…}` × `VSA{…}` × `Seq{VSA{…}, N≥1}` × `Float` threshold → `Seq{Float, 3}`), and
    /// `vsa_required_dim` (`Binary{W}` items × `Float` δ → `Binary{64}`), kernel
    /// `vsa.cleanup`/`vsa.reconstruct`/`vsa.required_dim` (surface names `_`-joined like
    /// `dense_add`). Returns `Ok(None)` if `name` is not one of them.
    ///
    /// Model + dim + sparsity live in the type (`VSA{model, dim, sparsity}`), so the static rules
    /// are: every hypervector operand must be a `VSA` type with **one shared model + dim +
    /// sparsity** (a mismatch is an explicit refusal, never a coercion — G2); the model must be
    /// in the **introduction dispatch set MAP-I / FHRR / BSC** (an out-of-set model — HRR/MAP-B/
    /// SBC mentions included — is a static refusal naming the set; widening is append-only and
    /// mirrors the interpreter's dispatch); and the sparsity must be `Dense` at introduction
    /// (the kernel's Value-level ops construct dense-class results — a `Sparse` claim would
    /// diverge from the runtime value, so it is refused, not silently re-classed). `vsa_permute`'s
    /// shift is any unsigned `Binary{W}` magnitude (rotation is cyclic — the inverse permute is
    /// the complementary shift `dim − s`); a bare-decimal shift has no width anchor here and must
    /// be written/ascribed explicitly (RFC-0012 §4.3 — never a default width).
    ///
    /// `vsa_bundle` adds two static rules of its own: the model must additionally be in the
    /// **certified-bundle set {MAP-I}** — the only introduction-set model with a *certified*
    /// Value-level bundle (`bundle_values_certified`, the M-131 checked-instantiation pattern);
    /// an FHRR/BSC bundle is a **static** refusal naming the certified set (their kernel bundles
    /// are `Empirical`-profile ops — routing them here would silently re-tag their evidence,
    /// VR-5; surfacing them is a distinct append-only extension) — and the item count `N` lives
    /// in the `Seq` type, so an **empty bundle (`N = 0`) is a static refusal** too (no
    /// superposition is defined). The capacity side-condition `dim ≥ requiredDim(N, δ)` stays
    /// **runtime** (δ is a run-time `Float`): the kernel refuses `InsufficientCapacity` naming
    /// the required dim rather than issuing an unbacked `Proven` bound.
    ///
    /// The M-894 prims add their own static rules: `vsa_cleanup`/`vsa_reconstruct`'s codebook is
    /// a `Seq{VSA{…}, N ≥ 1}` sharing the query/record's model + dim (`N` lives in the type, so
    /// an **empty codebook is a static refusal** — nothing to clean up against); `vsa_reconstruct`
    /// additionally restricts the model to the **reconstruct dispatch set {MAP-I, BSC}** — an
    /// FHRR record is a *static* refusal naming the ground (FHRR's `Empirical` unbind profile is
    /// trial-validated only for a single bind product, not a reconstruction record — VR-5; its
    /// surfacing is an append-only extension under a reconstruction-regime profile of its own),
    /// and its threshold is a `Float` (the RFC-0003 §6 manifest's `cleanup_threshold ∈ [0, 1]`
    /// made an explicit operand — the domain check stays runtime, δ-style). `vsa_required_dim`'s
    /// items ride any unsigned `Binary{W}` magnitude (like `vsa_permute`'s shift — a bare decimal
    /// has no width anchor, RFC-0012 §4.3) and the result is the `Binary{64}` sufficient
    /// dimension carrying the kernel's `Proven` `CapacityBound`. Both retrieval prims return the
    /// `Seq{Float, 3}` decision triple — the retrieval is never a silent nearest-neighbour pick
    /// (FR-S4/G2: confidence + margin are first-class results).
    ///
    /// The numeric-domain contracts stay **runtime**, owned by the kernel and carried through the
    /// interpreter wrapper: alphabet violations, the FHRR unbind regime gate, non-`Exact`
    /// operand composition, the RFC-0010 §4.4 identifiability tie, and the below-threshold
    /// retrieval are explicit runtime refusals; per-op tags ride the runtime value
    /// **per model** (MAP-I/BSC ops `Exact`; FHRR `unbind` `Empirical` with its trial-validated
    /// δ bound; the certified `vsa.bundle` result `Proven` with its checked `CapacityBound`;
    /// the M-894 decision triple carrying the query/record's own (strength, bound) pair —
    /// VR-5, carried, never re-stamped; `mycelium-interp/src/prims.rs`).
    fn try_check_vsa_prim(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        args: &[Expr],
    ) -> Result<Option<(Ty, Expr)>, CheckError> {
        if !matches!(
            name,
            "vsa_bind"
                | "vsa_unbind"
                | "vsa_permute"
                | "vsa_bundle"
                | "vsa_cleanup"
                | "vsa_reconstruct"
                | "vsa_required_dim"
        ) {
            return Ok(None);
        }
        // The M-892 introduction dispatch set (must mirror `vsa_model_of` in
        // `mycelium-interp/src/prims.rs`; widening one without the other is caught by the
        // three-way conformance tests).
        const VSA_PRIM_MODELS: [&str; 3] = ["MAP-I", "FHRR", "BSC"];
        let arity_err = |want: usize| -> Result<Option<(Ty, Expr)>, CheckError> {
            self.err(format!(
                "`{name}` takes {want} operand(s), got {} (RFC-0003 §3; M-892)",
                args.len()
            ))
        };
        // Every hypervector operand must be a concrete `VSA{model, dim, sparsity}` whose model is
        // in the dispatch set and whose sparsity is `Dense` (introduction scope).
        let check_hv = |scope: &mut Vec<(String, Ty)>,
                        a: &Expr,
                        which: &str|
         -> Result<(String, u32, Sparsity, Expr), CheckError> {
            let (ty, a2) = self.check(scope, a, None)?;
            let Ty::Vsa {
                model,
                dim,
                sparsity,
            } = ty
            else {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` {which} operand must be a `VSA{{model, dim, sparsity}}` \
                         hypervector, got {ty} (RFC-0003 §3; M-892 — never a silent conversion; \
                         a cross-paradigm edge needs an explicit `swap`)"
                    ),
                ));
            };
            if !VSA_PRIM_MODELS.contains(&model.as_str()) {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` model {model:?} is outside the vsa prim dispatch set at \
                         introduction (MAP-I, FHRR, BSC — M-892); the type is a legal mention, \
                         but its algebra is not surfaced — widening the set is an append-only \
                         extension, never a guessed algebra (G2)"
                    ),
                ));
            }
            if sparsity != Sparsity::Dense {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` requires a `Dense`-sparsity hypervector at introduction — the \
                         kernel's Value-level ops construct dense-class results, so a Sparse \
                         operand type would diverge from the runtime value (M-892; refused, \
                         never silently re-classed — G2)"
                    ),
                ));
            }
            Ok((model, dim, sparsity, a2))
        };
        // The M-894 codebook operand: a `Seq{VSA{model, dim, Dense}, N ≥ 1}` whose atoms share
        // the query/record's model + dim. `N` lives in the type, so the empty codebook is a
        // *static* refusal (nothing to clean up against — FR-S4; the runtime twin stays in the
        // interpreter wrapper for injected values).
        let check_codebook = |scope: &mut Vec<(String, Ty)>,
                              a: &Expr,
                              model: &str,
                              dim: u32|
         -> Result<Expr, CheckError> {
            let (sty, s2) = self.check(scope, a, None)?;
            let Ty::Seq(elem, n) = sty else {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` codebook must be a `Seq{{VSA{{model, dim, sparsity}}, N}}` of \
                         clean atoms, got {sty} (RFC-0003 §3/§6; M-894 — never a silent \
                         conversion)"
                    ),
                ));
            };
            let Ty::Vsa {
                model: cm,
                dim: cd,
                sparsity: cs,
            } = *elem
            else {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` codebook atoms must be `VSA{{model, dim, sparsity}}` \
                         hypervectors, got a Seq of {elem} (M-894 — never a silent conversion; \
                         a cross-paradigm edge needs an explicit `swap`)"
                    ),
                ));
            };
            if cm != model || cd != dim {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` codebook atoms must share the query's model and dim, got \
                         VSA{{{cm}, {cd}}} atoms against a VSA{{{model}, {dim}}} query/record \
                         (M-894 — a model or dim mismatch is an explicit refusal, never a \
                         cross-model coercion; G2)"
                    ),
                ));
            }
            if cs != Sparsity::Dense {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` requires `Dense`-sparsity codebook atoms at introduction — \
                         the kernel's cleanup memory holds dense-class atoms, so a Sparse atom \
                         type would diverge from the runtime value (M-894; refused, never \
                         silently re-classed — G2)"
                    ),
                ));
            }
            if n == 0 {
                return Err(CheckError::new(
                    self.site,
                    format!(
                        "`{name}` requires at least one codebook atom — an empty cleanup memory \
                         has nothing to clean up against (FR-S4; M-894 — refused statically, \
                         never a defaulted retrieval; G2)"
                    ),
                ));
            }
            Ok(s2)
        };
        // Both retrieval prims return the `[index, confidence, margin]` decision triple — the
        // retrieval decision is a first-class, inspectable value (FR-S4/G2).
        let triple_ty = || Ty::Seq(Box::new(Ty::Float), 3);
        match name {
            "vsa_bind" | "vsa_unbind" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (ma, da, spa, a2) = check_hv(scope, &args[0], "first")?;
                let (mb, db, _spb, b2) = check_hv(scope, &args[1], "second")?;
                if ma != mb || da != db {
                    // The never-silent model/shape contract, statically: name both full types.
                    return self.err(format!(
                        "`{name}` operands must share one model and dim, got VSA{{{ma}, {da}}} \
                         and VSA{{{mb}, {db}}} (M-892 — a model or dim mismatch is an explicit \
                         refusal, never a cross-model coercion; G2)"
                    ));
                }
                Ok(Some((
                    Ty::Vsa {
                        model: ma,
                        dim: da,
                        sparsity: spa,
                    },
                    app_node(head, vec![a2, b2]),
                )))
            }
            "vsa_permute" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (m, d, sp, a2) = check_hv(scope, &args[0], "hypervector")?;
                let (sty, s2) = self.check(scope, &args[1], None)?;
                let Ty::Binary(_) = sty else {
                    return self.err(format!(
                        "`vsa_permute` shift must be an unsigned `Binary{{W}}` magnitude, got \
                         {sty} (M-892 — rotation is cyclic, so the inverse permute is the \
                         complementary shift `dim − s`; a bare decimal has no width anchor here \
                         and must be written/ascribed explicitly, RFC-0012 §4.3)"
                    ));
                };
                Ok(Some((
                    Ty::Vsa {
                        model: m,
                        dim: d,
                        sparsity: sp,
                    },
                    app_node(head, vec![a2, s2]),
                )))
            }
            "vsa_bundle" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                // First operand: `Seq{VSA{model, dim, Dense}, N ≥ 1}` — the items to superpose.
                let (sty, s2) = self.check(scope, &args[0], None)?;
                let Ty::Seq(elem, n) = sty else {
                    return self.err(format!(
                        "`vsa_bundle` first operand must be a `Seq{{VSA{{model, dim, \
                         sparsity}}, N}}` of hypervectors to superpose, got {sty} (RFC-0003 \
                         §4/§5; M-893 — never a silent conversion)"
                    ));
                };
                let Ty::Vsa {
                    model,
                    dim,
                    sparsity,
                } = *elem
                else {
                    return self.err(format!(
                        "`vsa_bundle` items must be `VSA{{model, dim, sparsity}}` \
                         hypervectors, got a Seq of {elem} (RFC-0003 §4/§5; M-893 — never a \
                         silent conversion; a cross-paradigm edge needs an explicit `swap`)"
                    ));
                };
                if !VSA_PRIM_MODELS.contains(&model.as_str()) {
                    return self.err(format!(
                        "`vsa_bundle` model {model:?} is outside the vsa prim dispatch set at \
                         introduction (MAP-I, FHRR, BSC — M-892); the type is a legal mention, \
                         but its algebra is not surfaced — widening the set is an append-only \
                         extension, never a guessed algebra (G2)"
                    ));
                }
                // The certified-bundle set at introduction: {MAP-I} — the only model with a
                // certified Value-level bundle (`bundle_values_certified`). FHRR/BSC kernel
                // bundles are Empirical-profile ops; routing them through the certified prim
                // would silently re-tag their evidence (VR-5), so the refusal is static and
                // names the set (surfacing them is a distinct append-only extension).
                if model != "MAP-I" {
                    return self.err(format!(
                        "`vsa_bundle` is the certified superposition path and its dispatch set \
                         at introduction is the certified singleton {{MAP-I}} — model \
                         {model:?} has no certified Value-level bundle (M-893); the FHRR/BSC \
                         empirical-profile bundles are a distinct, append-only surfacing under \
                         their own name, never a silent re-tag (VR-5)"
                    ));
                }
                if sparsity != Sparsity::Dense {
                    return self.err(
                        "`vsa_bundle` requires `Dense`-sparsity hypervectors at introduction — \
                         the kernel's Value-level ops construct dense-class results, so a \
                         Sparse item type would diverge from the runtime value (M-893; \
                         refused, never silently re-classed — G2)",
                    );
                }
                // N lives in the Seq type, so the empty bundle is a *static* refusal here (the
                // kernel's EmptyBundle stays as the runtime twin for injected values).
                if n == 0 {
                    return self.err(
                        "`vsa_bundle` requires at least one item — no superposition is defined \
                         over a `Seq{…, 0}` (RFC-0003 §4; M-893 — refused statically, never a \
                         defaulted value — G2)"
                            .to_owned(),
                    );
                }
                // Second operand: the target failure probability δ, a `Float` (its (0, 1]
                // domain and the capacity side-condition are runtime, kernel-owned refusals).
                let (dty, d2) = self.check(scope, &args[1], Some(&Ty::Float))?;
                if !matches!(dty, Ty::Float) {
                    return self.err(format!(
                        "`vsa_bundle` target failure probability δ must be a `Float`, got \
                         {dty} (M-893 — the certified capacity bound targets an explicit δ; \
                         never a defaulted or coerced parameter — G2)"
                    ));
                }
                Ok(Some((
                    Ty::Vsa {
                        model,
                        dim,
                        sparsity,
                    },
                    app_node(head, vec![s2, d2]),
                )))
            }
            // M-894: `vsa_cleanup(query, codebook)` — the cleanup-memory retrieval. The query may
            // be any dispatch-set model (the arg-max procedure is model-generic — the model only
            // supplies `similarity`); the codebook must share its model + dim. Result: the
            // decision triple.
            "vsa_cleanup" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (m, d, _sp, q2) = check_hv(scope, &args[0], "query")?;
                let cb2 = check_codebook(scope, &args[1], &m, d)?;
                Ok(Some((triple_ty(), app_node(head, vec![q2, cb2]))))
            }
            // M-894: `vsa_reconstruct(record, role, codebook, threshold)` — the RFC-0003 §6
            // compositional role-reconstruction (unbind → cleanup → an explicit threshold). The
            // model must be in the **reconstruct dispatch set {MAP-I, BSC}** (the self-inverse
            // Exact unbinds): FHRR's Empirical unbind profile is trial-validated only for a
            // single `vsa.fhrr.bind` product, which a reconstruction record is not — stretching
            // it would be an unearned tag (VR-5), so the refusal is static and names the ground;
            // surfacing FHRR reconstruction is a distinct append-only extension under a
            // reconstruction-regime profile of its own.
            "vsa_reconstruct" => {
                if args.len() != 4 {
                    return arity_err(4);
                }
                let (m, d, _sp, r2) = check_hv(scope, &args[0], "record")?;
                if m != "MAP-I" && m != "BSC" {
                    return self.err(format!(
                        "`vsa_reconstruct` model {m:?} is outside the reconstruct dispatch set \
                         at introduction ({{MAP-I, BSC}}) — FHRR's unbind tag is Empirical and \
                         trial-validated only for a single bind product, not a reconstruction \
                         record (M-894; VR-5 — never a stretched profile; surfacing it is a \
                         distinct, append-only extension)"
                    ));
                }
                let (mb, db, _spb, role2) = check_hv(scope, &args[1], "role")?;
                if mb != m || db != d {
                    return self.err(format!(
                        "`vsa_reconstruct` record and role must share one model and dim, got \
                         VSA{{{m}, {d}}} and VSA{{{mb}, {db}}} (M-894 — a model or dim mismatch \
                         is an explicit refusal, never a cross-model coercion; G2)"
                    ));
                }
                let cb2 = check_codebook(scope, &args[2], &m, d)?;
                // The threshold: a `Float` — the RFC-0003 §6 manifest's `cleanup_threshold`
                // made an explicit operand; its [0, 1] domain stays a runtime, kernel-owned
                // refusal (δ-style — a run-time Float).
                let (tty, t2) = self.check(scope, &args[3], Some(&Ty::Float))?;
                if !matches!(tty, Ty::Float) {
                    return self.err(format!(
                        "`vsa_reconstruct` cleanup threshold must be a `Float`, got {tty} \
                         (RFC-0003 §6; M-894 — a below-threshold retrieval is an explicit \
                         refusal, never a defaulted or coerced threshold; G2)"
                    ));
                }
                Ok(Some((
                    triple_ty(),
                    app_node(head, vec![r2, role2, cb2, t2]),
                )))
            }
            // M-894: `vsa_required_dim(items, δ)` — the M-131 capacity-bound query. Items ride
            // any unsigned `Binary{W}` magnitude (like `vsa_permute`'s shift — a bare decimal
            // has no width anchor here, RFC-0012 §4.3); δ is a `Float` whose (0, 1] domain is a
            // runtime refusal. Result: the `Binary{64}` sufficient dimension, carrying the
            // kernel's `Proven` `CapacityBound` (the checked instantiation — inspectable).
            "vsa_required_dim" => {
                if args.len() != 2 {
                    return arity_err(2);
                }
                let (ity, i2) = self.check(scope, &args[0], None)?;
                let Ty::Binary(_) = ity else {
                    return self.err(format!(
                        "`vsa_required_dim` items must be an unsigned `Binary{{W}}` magnitude, \
                         got {ity} (M-894 — a bare decimal has no width anchor here and must be \
                         written/ascribed explicitly, RFC-0012 §4.3)"
                    ));
                };
                let (dty, d2) = self.check(scope, &args[1], Some(&Ty::Float))?;
                if !matches!(dty, Ty::Float) {
                    return self.err(format!(
                        "`vsa_required_dim` target failure probability δ must be a `Float`, \
                         got {dty} (M-894 — the capacity theorem instantiates an explicit δ; \
                         never a defaulted or coerced parameter — G2)"
                    ));
                }
                Ok(Some((
                    Ty::Binary(Width::Lit(64)),
                    app_node(head, vec![i2, d2]),
                )))
            }
            _ => unreachable!("gated by the matches! above"),
        }
    }

    /// Try to check a call to a **scalar-float prim** (ADR-040 §2.4/§2.5; M-898/M-899, `enb`
    /// Gap A) — the arithmetic group `flt_add`/`flt_sub`/`flt_mul`/`flt_div` (binary) and
    /// `flt_neg` (unary), kernel `flt.add`/`flt.sub`/`flt.mul`/`flt.div`/`flt.neg`, plus the
    /// comparison group `flt_lt`/`flt_le`/`flt_gt`/`flt_ge`/`flt_eq` and the named total order
    /// `flt_total_le` (all binary), kernel `flt.lt`/…/`flt.eq`/`flt.total_le` (surface names are
    /// `_`-joined like `dense_add`, since `.` is the path separator in the lexer). Returns
    /// `Ok(None)` if `name` is not one of them.
    ///
    /// `Float` is nullary (binary64 only at introduction — ADR-040 FLAG-1), so the static rule is
    /// the simplest of the prim branches: every operand must be exactly `Float`, and the result
    /// is `Float` for the arithmetic ops and **`Binary{1}`** — the realized `Bool` of the
    /// RFC-0032 D1 engineering note, exactly the `eq`/`lt` shape — for the comparison ops. A
    /// non-`Float` operand is an explicit refusal (a cross-paradigm edge needs an
    /// explicit `swap`; a bare decimal has no `Float` anchor — RFC-0012 §4.3 gives it none — so it
    /// is refused, never defaulted; write a float literal `1.0`, M-897). The numeric-domain
    /// behaviour is deliberately **not** a static rule and **not** a runtime refusal either:
    /// arithmetic specials (overflow → ±inf, `x/0` → ±inf, `0/0` → NaN) are **in-band,
    /// inspectable, propagating values** per the ratified ADR-040 §2.4 FLAG-2 — the ops are total
    /// over `Float` (contrast `div_u`, whose integer div-by-zero has no in-band sentinel and
    /// must refuse at runtime) — and **comparison NaN semantics are value semantics, not types**:
    /// the five predicates deliver the IEEE-754 §5.11 *defined* result `false` on any NaN operand
    /// (NaN is unordered; `flt_eq(NaN, NaN)` is false), while `flt_total_le` is the IEEE-754
    /// §5.10 `totalOrder` — total and reflexive, NaN placed last, `−0` before `+0` (ADR-040
    /// §2.4; M-899). Per-op tag: `Empirical` per ADR-040 §2.6, carried on the runtime value with
    /// its zero-deviation-vs-spec bound; for `flt_total_le` the total-order *property* stays
    /// `Empirical` until the M-511 proof debt is discharged (`mycelium-interp/src/prims.rs`).
    fn try_check_float_prim(
        &self,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        name: &str,
        args: &[Expr],
    ) -> Result<Option<(Ty, Expr)>, CheckError> {
        // ADR-040 §2.4 (CU-3): the never-silent Binary↔Float conversions. Mixed-paradigm operands
        // (unlike the uniform-Float shape the rest of this function checks below), so a dedicated
        // pre-branch — `flt_to_bin` mirrors `width_cast`'s witness-operand shape (DN-41): the
        // second operand's `Binary{M}` *width* is the result width, its bits unused.
        if name == "bin_to_flt" {
            if args.len() != 1 {
                return self.err(format!(
                    "`bin_to_flt` takes 1 operand, got {} (ADR-040 §2.4; CU-3)",
                    args.len()
                ));
            }
            let (vty, v2) = self.check(scope, &args[0], None)?;
            if !matches!(vty, Ty::Binary(_)) {
                return self.err(format!(
                    "`bin_to_flt` operand must be a concrete `Binary{{N}}`, got {vty} (ADR-040 \
                     §2.4 — a checked-exact conversion; never a default width)"
                ));
            }
            return Ok(Some((Ty::Float, app_node(head, vec![v2]))));
        }
        if name == "flt_to_bin" {
            if args.len() != 2 {
                return self.err(format!(
                    "`flt_to_bin` takes 2 operands, got {} (ADR-040 §2.4; CU-3)",
                    args.len()
                ));
            }
            let (vty, v2) = self.check(scope, &args[0], Some(&Ty::Float))?;
            if !matches!(vty, Ty::Float) {
                return self.err(format!(
                    "`flt_to_bin` first (value) operand must be a `Float` (IEEE-754 binary64 — \
                     ADR-040 §2.1), got {vty} (never a silent conversion; write a float literal \
                     like `1.0`)"
                ));
            }
            let (wty, w2) = self.check(scope, &args[1], None)?;
            let Ty::Binary(Width::Lit(m)) = wty else {
                return self.err(format!(
                    "`flt_to_bin` width witness must be a concrete `Binary{{M}}` (only its width \
                     is used), got {wty} (ADR-040 §2.4, the DN-41 witness shape; a width-variable \
                     witness is refused)"
                ));
            };
            return Ok(Some((
                Ty::Binary(Width::Lit(m)),
                app_node(head, vec![v2, w2]),
            )));
        }
        let is_cmp = matches!(
            name,
            "flt_lt" | "flt_le" | "flt_gt" | "flt_ge" | "flt_eq" | "flt_total_le"
        );
        // ADR-040 §2.5 (CU-2): the mandated classification predicates — unary `Float → Binary{1}`.
        let is_class = matches!(name, "flt_is_nan" | "flt_is_finite" | "flt_is_infinite");
        if !is_cmp
            && !is_class
            && !matches!(
                name,
                "flt_add" | "flt_sub" | "flt_mul" | "flt_div" | "flt_neg"
            )
        {
            return Ok(None);
        }
        let want = if name == "flt_neg" || is_class { 1 } else { 2 };
        if args.len() != want {
            return self.err(format!(
                "`{name}` takes {want} operand(s), got {} (ADR-040 §2.4/§2.5; M-898/M-899)",
                args.len()
            ));
        }
        let mut rebuilt = Vec::with_capacity(want);
        for (i, a) in args.iter().enumerate() {
            let (ty, a2) = self.check(scope, a, Some(&Ty::Float))?;
            if !matches!(ty, Ty::Float) {
                let which = match (want, i) {
                    (1, _) => "single",
                    (_, 0) => "first",
                    _ => "second",
                };
                return self.err(format!(
                    "`{name}` {which} operand must be a `Float` (IEEE-754 binary64 — ADR-040 \
                     §2.1), got {ty} (M-898/M-899 — never a silent conversion; a cross-paradigm \
                     edge needs an explicit `swap`, and a bare decimal has no Float anchor — \
                     write a float literal like `1.0`)"
                ));
            }
            rebuilt.push(a2);
        }
        let ret = if is_cmp || is_class {
            Ty::Binary(Width::Lit(1))
        } else {
            Ty::Float
        };
        Ok(Some((ret, app_node(head, rebuilt))))
    }

    /// Literal typing (Q6): a literal *is* its representation — a binary literal's width is its
    /// digit count, a ternary literal's trit count its width. Bare integers need context v0 does not
    /// yet give them → explicit refusal, never a cross-family default. (List literals are checked by
    /// [`Self::check_list`], which has the element-expression context `lit_ty_of` lacks.)
    fn lit_ty(&self, l: &Literal) -> Result<Ty, CheckError> {
        lit_ty_of(self.site, l)
    }
}

/// The literal-typing rule (Q6), as a free function so the elaborator can reuse it without a
/// checking context. A literal *is* its representation (width = digit count); bare integers and
/// lists are explicit refusals.
pub(crate) fn lit_ty_of(site: &str, l: &Literal) -> Result<Ty, CheckError> {
    match l {
        Literal::Bin(s) => {
            let n = s.chars().filter(|c| *c == '0' || *c == '1').count();
            if n == 0 {
                return Err(CheckError::new(site, "empty binary literal"));
            }
            Ok(Ty::Binary(Width::Lit(
                u32::try_from(n).expect("digit count fits u32"),
            )))
        }
        Literal::Trit(s) => {
            if s.is_empty() {
                return Err(CheckError::new(site, "empty ternary literal"));
            }
            Ok(Ty::Ternary(Width::Lit(
                u32::try_from(s.len()).expect("trit count fits u32"),
            )))
        }
        Literal::Int(_) => Err(CheckError::new(
            site,
            "a bare integer literal has no representation family (no cross-family defaulting, \
             Q6) — write a binary/ternary literal, declare a `default paradigm` (RFC-0012), or \
             ascribe a Dense element",
        )),
        Literal::AmbientInt(_, _) => Err(CheckError::new(
            site,
            "internal: an unresolved ambient bare decimal reached `lit_ty_of` — the checker \
             resolves its width from context first (RFC-0012 §4.3)",
        )),
        // RFC-0032 D4 (M-750): a `0x…` byte-string literal *is* a `Bytes` value.
        Literal::Bytes(_) => Ok(Ty::Bytes),
        // ADR-040 (M-897): a decimal float literal types as the scalar-float repr type. The lexer
        // already validated form + binary64 finiteness, so the literal is well-typed context-free.
        Literal::Float(_) => Ok(Ty::Float),
        // M-910/M-911: a `"…"` textual string literal lowers to the SAME `Repr::Bytes` value form
        // as `Literal::Bytes` (UTF-8-encoded; KC-3 — no new value form), so it types as `Bytes` too.
        Literal::Str(_) => Ok(Ty::Bytes),
        // A list literal is checked in `Cx::check_list` (it needs the element-expression context).
        // Reaching `lit_ty_of` means a `Literal::List` appeared where only context-free literals are
        // expected — a **pattern** position (a `Seq` pattern is not a v0 surface). An explicit
        // refusal, never a silent accept (G2).
        Literal::List(_) => Err(CheckError::new(
            site,
            "a list/sequence pattern is not supported in the v0 surface (RFC-0032 D3 — `[…]` is a \
             constructor literal, not a pattern)",
        )),
    }
}

/// Normalize a surface [`Pattern`] against its `expected` type into a [`crate::usefulness::Pat`]
/// (the coverage-matrix shape), collecting its binders as `(name, type, occurrence)` — the
/// **occurrence** is the path of field indices from the scrutinee root to the binder's position.
/// Shared by the checker (`Cx::check_pattern`, occurrence `[]`) and the **elaborator** (which needs
/// the matrix + the binder occurrences to lower a `match` to flat L0 `Match`/binders — RFC-0011
/// §4.4). Nested constructor/literal patterns recurse (RFC-0007 §4.7); a binder is a wildcard for
/// coverage (it refines nothing), a nullary constructor an empty `Ctor`.
pub(crate) fn normalize_pattern(
    types: &BTreeMap<String, DataInfo>,
    site: &str,
    pat: &Pattern,
    expected: &Ty,
    occ: &[usize],
    binds: &mut Vec<(String, Ty, Vec<usize>)>,
) -> Result<crate::usefulness::Pat, CheckError> {
    use crate::usefulness::Pat;
    match pat {
        Pattern::Wildcard => Ok(Pat::Wild),
        Pattern::Ident(n) => {
            // A bare name is a nullary-constructor alternative iff it names one of the expected
            // data type's constructors; otherwise it binds the whole position (at this occurrence).
            if let Ty::Data(tn, _) = expected {
                let d = types.get(tn).expect("registered data type");
                if let Some(c) = d.ctors.iter().find(|c| c.name == *n) {
                    if !c.fields.is_empty() {
                        return Err(CheckError::new(
                            site,
                            format!(
                                "constructor pattern `{n}` must bind its {} field(s) (W7)",
                                c.fields.len()
                            ),
                        ));
                    }
                    return Ok(Pat::Ctor(n.clone(), vec![]));
                }
            }
            binds.push((n.clone(), expected.clone(), occ.to_vec()));
            Ok(Pat::Wild)
        }
        Pattern::Ctor(n, subs) => {
            let Ty::Data(tn, targs) = expected else {
                return Err(CheckError::new(
                    site,
                    format!(
                        "constructor pattern `{n}` on a {expected} scrutinee — match a literal or `_`"
                    ),
                ));
            };
            let d = types.get(tn).expect("registered data type").clone();
            let Some(c) = d.ctors.iter().find(|c| c.name == *n) else {
                return Err(CheckError::new(
                    site,
                    format!("`{n}` is not a constructor of {tn}"),
                ));
            };
            if subs.len() != c.fields.len() {
                return Err(CheckError::new(
                    site,
                    format!(
                        "pattern `{n}` binds {} of {} field(s) (W7: exactly the arity)",
                        subs.len(),
                        c.fields.len()
                    ),
                ));
            }
            // The constructor's field types are abstract over the type's parameters; instantiate them
            // at the scrutinee's type arguments so each binder gets its concrete type (RFC-0007 §11.2).
            let s = param_subst(&d.params, targs);
            let mut out = Vec::with_capacity(subs.len());
            for (i, (sub, fty)) in subs.iter().zip(&c.fields).enumerate() {
                let mut child = occ.to_vec();
                child.push(i);
                let fty = subst_ty(fty, &s);
                out.push(normalize_pattern(types, site, sub, &fty, &child, binds)?);
            }
            Ok(Pat::Ctor(n.clone(), out))
        }
        // M-826: a tuple pattern `(x, y, …)` desugars to a `Ctor` pattern on the synthetic
        // `Tuple$N` type. After `resolve_pattern` has run, all `Pattern::Tuple` nodes should
        // already have been rewritten to `Pattern::Ctor`; this arm handles any that reach here
        // directly (e.g. in standalone normalization). Never-silent (G2): arity or type mismatch
        // is an explicit error.
        Pattern::Tuple(subs) => {
            let n = subs.len();
            let ctor_name = tuple_ctor_name(n);
            // Re-delegate to the Ctor arm by building the equivalent pattern.
            normalize_pattern(
                types,
                site,
                &Pattern::Ctor(ctor_name.clone(), subs.clone()),
                expected,
                occ,
                binds,
            )
            .map_err(|e| {
                // Rephrase the error to mention the tuple surface syntax.
                CheckError::new(
                    site,
                    format!(
                        "tuple pattern `({})` — {}",
                        subs.iter().map(|_| "_").collect::<Vec<_>>().join(", "),
                        e.message
                    ),
                )
            })
        }
        Pattern::Lit(lit) => {
            // ADR-040 FLAG-4 (M-897): float literals are refused in match patterns — an explicit
            // refusal, never a silently-chosen equality. IEEE `==` and bit-level content identity
            // diverge on floats (`+0.0 == -0.0` but two identities; canonical NaN is one identity
            // but `NaN != NaN`), so a literal-pattern arm would have to silently pick one of the
            // two semantics. Float dispatch goes through the comparison prims when they land
            // (M-899), where the choice is a named op, never a hidden pattern semantic (G2/VR-5).
            if matches!(lit, Literal::Float(_)) {
                return Err(CheckError::new(
                    site,
                    "a float literal is not a legal match pattern (ADR-040 FLAG-4: IEEE equality \
                     and content identity diverge on floats — `-0.0`/NaN; match on the result of \
                     an explicit comparison instead — never a silent equality choice, G2)",
                ));
            }
            let lty = lit_ty_of(site, lit)?;
            if lty != *expected {
                return Err(CheckError::new(
                    site,
                    format!(
                        "literal pattern has type {lty} but the scrutinee is {expected} \
                         (W7: a literal arm must match the scrutinee's repr and width)"
                    ),
                ));
            }
            Ok(Pat::Lit(literal_key(lit)))
        }
        // `Pattern::Or` is desugared in `check_match` before `normalize_pattern` is ever called;
        // reaching here means the invariant was violated — an explicit never-silent refusal (G2).
        Pattern::Or(_) => Err(CheckError::new(
            site,
            "internal: Pattern::Or reached normalize_pattern — or-patterns must be desugared \
             in check_match before any downstream pass (invariant violation — report this)",
        )),
    }
}

/// Re-infer an expression's type against a checked [`Env`] (the elaborator needs the scrutinee type
/// to lower a `match`, and a `let`-bound's type to track its scope — RFC-0011). The program is
/// already checked, so this recomputes a type the checker validated; it does not re-litigate errors.
pub(crate) fn infer_type(
    env: &Env,
    scope: &mut Vec<(String, Ty)>,
    e: &Expr,
) -> Result<Ty, CheckError> {
    // Re-inference has no cross-nodule imports (the term is already checked and monomorphic; no
    // glob-ambiguity can arise here — any ambiguity was refused during checking — M-662).
    let no_imports = NoduleImports::default();
    let cx = Cx {
        site: "<elaborate>",
        types: &env.types,
        fns: &env.fns,
        // The trait/instance registries are available for re-inference (a monomorphic body may
        // call a trait method resolved through a concrete instance — RFC-0019 §4.5).
        traits: &env.traits,
        instances: &env.instances,
        imports: &no_imports,
        // Re-inference runs over already-checked, monomorphic terms (a generic *instantiation* is
        // refused at elaboration before re-inference — RFC-0007 §11.3 staging), so no type
        // parameters / bounds are in scope here.
        tyvars: &[],
        bounds: &[],
        // Re-inference is **post-check**: the `@std-sys` gate (M-661) already passed during checking,
        // and a `wild` block lowers to an explicit `Residual` in the elaborator *before* any interior
        // re-inference, so `wild` never reaches here. Setting this `true` keeps re-inference honest
        // (it would never spuriously refuse a `wild` that the program already validated) without
        // re-litigating the gate — the gate is the checker's job, done once.
        std_sys: true,
        depth: Cell::new(0),
        // Post-check re-inference over an already-validated term, invoked repeatedly over
        // partial/overlapping fragments with a scope the elaborator threads itself — not the one
        // whole-function-body walk, so the affine pass stays inert here (`crate::affine` docs: it
        // would risk a false positive on a fragment that isn't the original walk, and the term
        // already passed the real check).
        affine: Tracker::inert(),
    };
    cx.infer(scope, e)
}

/// A canonical key for de-duplicating literal patterns (M-320): normalize away `_` separators so
/// `0b1010` and `0b10_10` collide as the *same* literal. `Bin`/`Trit`/`Bytes`/`Str` reach here (the
/// caller type-checks the literal first, which rejects `Int`/`List`).
fn literal_key(lit: &Literal) -> String {
    match lit {
        Literal::Bin(s) => format!(
            "b:{}",
            s.chars()
                .filter(|c| *c == '0' || *c == '1')
                .collect::<String>()
        ),
        Literal::Trit(s) => format!("t:{s}"),
        Literal::Int(i) => format!("i:{i}"),
        Literal::AmbientInt(p, i) => format!("amb:{p}:{i}"),
        // `Bytes`/`List` literal patterns are not a v0 surface (the type-check rejects them before
        // here); a stable key keeps the function total without a silent collision (G2).
        Literal::Bytes(s) => {
            format!("by:{}", s.chars().filter(|c| *c != '_').collect::<String>())
        }
        // M-910/M-911: a string literal pattern types as `Bytes` (like `Literal::Bytes` above), so
        // it can reach here too; keyed on its decoded content (a distinct prefix — `s:` never
        // collides with `by:`'s hex-digit alphabet).
        Literal::Str(s) => format!("s:{s}"),
        // M-897: float literal patterns are refused by `normalize_pattern` before keying (ADR-040
        // FLAG-4), so this arm is unreachable in practice; a stable key (the verbatim source text,
        // distinct `f:` prefix) keeps the function total without a silent collision (G2).
        Literal::Float(s) => format!("f:{s}"),
        Literal::List(_) => "list".to_owned(),
    }
}

/// The teaching diagnostic for imperative control-flow words used as names (RFC-0007 §4.8):
/// the error was happening anyway (unknown name) — make it teach instead of confuse.
fn teach_unknown(name: &str, base: &str) -> String {
    if matches!(name, "while" | "loop" | "break" | "continue" | "return") {
        format!(
            "{base} — `{name}` is not a Mycelium form; iterate by recursion or \
             `for x in xs, acc = init => body` (RFC-0007 §4.8)"
        )
    } else {
        base.to_owned()
    }
}

/// The v0 linear-recursion shape check (RFC-0007 §4.8): every constructor of `tname` is either
/// a **nil** (no fields) or a **cons** (exactly one spine field of type `tname` + exactly one
/// element field), with one element type across all cons constructors. Returns the element
/// type; anything else is an explicit refusal.
fn linear_elem_ty(
    site: &str,
    types: &BTreeMap<String, DataInfo>,
    tname: &str,
    targs: &[Ty],
) -> Result<Ty, CheckError> {
    let d = types
        .get(tname)
        .ok_or_else(|| CheckError::new(site, format!("unknown type `{tname}`")))?;
    // The declared element type is abstract over the type's parameters; instantiate it at the
    // scrutinee's type arguments (RFC-0007 §11.2) so `for` over a `List<Binary{8}>` binds `Binary{8}`.
    let s = param_subst(&d.params, targs);
    let mut elem: Option<Ty> = None;
    let mut has_cons = false;
    for c in &d.ctors {
        if c.fields.is_empty() {
            continue; // a nil — ends the spine
        }
        let (spine, rest): (Vec<&Ty>, Vec<&Ty>) = c
            .fields
            .iter()
            .partition(|f| matches!(f, Ty::Data(n, _) if n == tname));
        if spine.len() != 1 || rest.len() != 1 {
            return Err(CheckError::new(
                site,
                format!(
                    "`for` needs a linearly recursive type: constructor `{}` of `{tname}` must \
                     have exactly one `{tname}` field and one element field (general \
                     catamorphisms are an L2 concern — RFC-0007 §4.8)",
                    c.name
                ),
            ));
        }
        has_cons = true;
        let elem_ty = subst_ty(rest[0], &s);
        match &elem {
            None => elem = Some(elem_ty),
            Some(e) if *e == elem_ty => {}
            Some(e) => {
                return Err(CheckError::new(
                    site,
                    format!(
                        "`for` needs one element type across `{tname}`'s constructors: \
                         {e} vs {elem_ty}"
                    ),
                ))
            }
        }
    }
    if !has_cons {
        return Err(CheckError::new(
            site,
            format!("`{tname}` has no recursive constructor — nothing for `for` to iterate"),
        ));
    }
    Ok(elem.expect("has_cons implies an element type"))
}

/// Rebuild an [`Expr::App`] node from its head and (resolved) arguments.
fn app_node(head: &Expr, args: Vec<Expr>) -> Expr {
    Expr::App {
        head: Box::new(head.clone()),
        args,
    }
}

/// The paradigm name of a representation type (for the never-silent cross-paradigm framing).
fn paradigm_name(t: &Ty) -> Option<&'static str> {
    match t {
        Ty::Binary(_) => Some("Binary"),
        Ty::Ternary(_) => Some("Ternary"),
        Ty::Dense(_, _) => Some("Dense"),
        // RFC-0001 §4.1's fourth paradigm (M-892): a cross-paradigm edge involving a hypervector
        // gets the same explicit-`swap` framing (no v0 swap engine reaches it — the refusal is
        // the honest message, never a silent conversion).
        Ty::Vsa { .. } => Some("VSA"),
        // `Ty::Fn` is not a representation type — it has no paradigm (RFC-0024 §3). `Seq`/`Bytes`
        // are first-class reprs but not *swap*-paradigms (no cross-paradigm swap edge), so they have
        // no paradigm name here (RFC-0032 D3/D4).
        // `Ty::Float` likewise: a first-class repr but not a v0 swap-paradigm (no swap engine —
        // any future float<->binary/dense conversion is an explicit certed swap, ADR-040 §5).
        Ty::Seq(_, _)
        | Ty::Bytes
        | Ty::Float
        | Ty::Data(_, _)
        | Ty::Substrate(_)
        | Ty::Var(_)
        | Ty::Fn(_, _) => None,
    }
}

/// Frame a type-edge mismatch (RFC-0012 §4.4): a **cross-paradigm** edge (`want` and `got` are
/// representation types of *different* paradigms) becomes an explicit `MissingConversion` pointing
/// at writing a `swap` — the never-silent guarantee. A same-paradigm mismatch (e.g. two `Binary`
/// widths) keeps the plain wording.
fn edge_mismatch(edge: &str, want: &Ty, got: &Ty) -> String {
    match (paradigm_name(want), paradigm_name(got)) {
        (Some(w), Some(g)) if w != g => format!(
            "MissingConversion: {edge} is {got} but {want} is required — a cross-paradigm edge needs \
             an explicit `swap(…, to: {want}, policy: …)` (RFC-0012 §4.4; never silently converted)"
        ),
        _ => format!("{edge} has type {got}, expected {want}"),
    }
}

/// A bare-decimal-bearing prim family — every v0 prim is width-polymorphic and width-preserving, so
/// the result width (or a concrete operand's) anchors a bare-decimal operand (RFC-0012 §4.3).
#[derive(Clone, Copy)]
enum PrimFam {
    Binary,
    Ternary,
}

impl PrimFam {
    /// The width of `t` if it is this family's representation type, else `None`.
    fn width_of(self, t: &Ty) -> Option<u32> {
        match (self, t) {
            (PrimFam::Binary, Ty::Binary(Width::Lit(w)))
            | (PrimFam::Ternary, Ty::Ternary(Width::Lit(w))) => Some(*w),
            _ => None,
        }
    }

    /// This family's type at width `w`.
    fn ty(self, w: u32) -> Ty {
        match self {
            PrimFam::Binary => Ty::Binary(Width::Lit(w)),
            PrimFam::Ternary => Ty::Ternary(Width::Lit(w)),
        }
    }
}

/// The family of a builtin prim, or `None` if `name` is not a known prim.
fn prim_family(name: &str) -> Option<PrimFam> {
    Some(match name {
        // RFC-0032 D2 (M-748): width-preserving binary logical + arithmetic prims.
        // RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): `mul_s` — never-silent two's-complement
        // multiply (distinct surface name from the trit-backed `mul` below, which stays balanced-
        // ternary). M-888 adds `div_u`/`rem_u` — never-silent **unsigned** division/remainder
        // (the signed variant rides M-767 under its own distinct name, per RFC-0033 §4.1.2's
        // signedness-split requirement for division). M-889 adds `shl_u`/`shr_u` — never-silent
        // **logical** left/right shift (the arithmetic/signed right shift rides M-767 likewise).
        // M-766 adds `add_s`/`sub_s`/`neg_s` — never-silent two's-complement add/sub/neg.
        // M-767 adds `div_s`/`rem_s`/`shr_s` — the signed (two's-complement) division/remainder
        // (truncated toward zero, remainder sign following the dividend — SMT-LIB `bvsdiv`/
        // `bvsrem`; the `min ÷ −1` overflow refuses explicitly) and the arithmetic
        // (sign-extending) right shift, completing the §4.1.2 signedness split. (The signed
        // comparison `lt_s` rides the dedicated width-collapsing comparison branch, not this
        // width-preserving family.)
        // **Surface naming (DN-72, ratified 2026-07-02):** every integer prim's surface name
        // carries an explicit signedness suffix — `_u` = unsigned semantics, `_s` =
        // signed/two's-complement (ADR-028: signedness lives in the operation, not the type).
        // This resolved the historical `_bin`/`_tc` naming FLAG (the mixed suffixes the set
        // accumulated as ops landed under M-748/M-887/M-888/M-889/M-766). The `bit.*`/`bin.*`
        // *kernel*-namespace inconsistency one layer down is deliberately NOT touched — kernel
        // names are content-addressed (DN-10 §3.4); see the DN-72 deferred FLAG.
        "not" | "popcount" | "clz" | "ctz" | "xor" | "and" | "or" | "add_u" | "sub_u" | "mul_u"
        | "mul_s" | "div_u" | "rem_u" | "shl_u" | "shr_u" | "add_s" | "sub_s" | "neg_s"
        | "div_s" | "rem_s" | "shr_s" => PrimFam::Binary,
        "add" | "sub" | "mul" | "neg" => PrimFam::Ternary,
        _ => return None,
    })
}

/// Encode a non-negative decimal `v` as an **unsigned** `Binary{width}` literal (MSB-first), or an
/// explicit refusal if it does not fit (RFC-0012 §4.3 — never a silent wrap/truncation).
fn encode_binary(site: &str, v: i64, width: u32) -> Result<Literal, CheckError> {
    if width == 0 {
        return Err(CheckError::new(
            site,
            "cannot encode a decimal at Binary{0} (zero width)",
        ));
    }
    let vu = u128::try_from(v).map_err(|_| {
        CheckError::new(
            site,
            format!("negative decimal `{v}` has no unsigned `Binary` encoding"),
        )
    })?;
    if width < 128 && (vu >> width) != 0 {
        return Err(CheckError::new(
            site,
            format!("decimal `{v}` does not fit Binary{{{width}}} (unsigned range 0..2^{width})"),
        ));
    }
    let mut s = String::with_capacity(width as usize);
    for i in (0..width).rev() {
        let bit = if i < 128 { (vu >> i) & 1 } else { 0 };
        s.push(if bit == 1 { '1' } else { '0' });
    }
    Ok(Literal::Bin(s))
}

/// Encode a decimal `v` as a **balanced-ternary** `Ternary{width}` literal (MSB-first, digits in
/// `{-,0,+}`), or an explicit refusal if it does not fit the symmetric range (RFC-0012 §4.3).
fn encode_balanced_ternary(site: &str, v: i64, width: u32) -> Result<Literal, CheckError> {
    if width == 0 {
        return Err(CheckError::new(
            site,
            "cannot encode a decimal at Ternary{0} (zero width)",
        ));
    }
    let mut n = i128::from(v);
    let mut digits: Vec<i8> = Vec::new(); // LSB-first, each in {-1, 0, 1}
    while n != 0 {
        let mut r = (n % 3) as i8;
        n /= 3;
        if r == 2 {
            r = -1;
            n += 1;
        } else if r == -2 {
            r = 1;
            n -= 1;
        }
        digits.push(r);
    }
    if digits.len() > width as usize {
        return Err(CheckError::new(
            site,
            format!(
                "decimal `{v}` does not fit Ternary{{{width}}} (balanced range ±(3^{width}-1)/2)"
            ),
        ));
    }
    digits.resize(width as usize, 0);
    let s: String = digits
        .iter()
        .rev()
        .map(|d| match d {
            -1 => '-',
            0 => '0',
            1 => '+',
            _ => unreachable!("a balanced trit is in {{-1, 0, 1}}"),
        })
        .collect();
    Ok(Literal::Trit(s))
}

/// The builtin prim signature table `Π` (RFC-0007 §4.4 T-Op), width-polymorphic. Surface names
/// map onto the trusted interpreter's registry (`bit.*`/`trit.*`).
#[must_use]
pub fn prim_sig(name: &str, args: &[Ty]) -> Option<Ty> {
    match (name, args) {
        // M-766: `neg_s` — the two's-complement unary negate joins `not` (unary, width-preserving).
        ("not" | "neg_s" | "popcount" | "clz" | "ctz", [Ty::Binary(w)]) => {
            Some(Ty::Binary(w.clone()))
        }
        ("xor", [Ty::Binary(a), Ty::Binary(b)]) if a == b => Some(Ty::Binary(a.clone())),
        // RFC-0032 D2 (M-748): width-preserving binary arithmetic/logical (never-silent overflow is
        // a runtime contract; the static signature is width-preserving like the trit arithmetic).
        // RFC-0033 §4.1.2/§4.1.3 (M-887): `mul_s` — same width-preserving shape; the never-silent
        // two's-complement overflow bound is likewise a runtime contract (`bin.mul`'s `Overflow`).
        // M-888: `div_u`/`rem_u` — same width-preserving shape; div-by-zero is likewise a
        // runtime contract (`bin.div`/`bin.rem`'s `PrimType` refusal), not a static type error.
        // M-889: `shl_u`/`shr_u` — same width-preserving shape (both operands, including the
        // shift amount, are `Binary{N}`); an out-of-range shift amount is likewise a runtime
        // contract (`bin.shl`/`bin.shr`'s `PrimType` refusal), not a static type error.
        // M-766: `add_s`/`sub_s` — same width-preserving shape; the never-silent two's-complement
        // overflow bound is likewise a runtime contract (`bin.add`/`bin.sub`'s `Overflow`).
        // M-767: `div_s`/`rem_s`/`shr_s` — same width-preserving shape; div-by-zero, the
        // `min ÷ −1` signed-division overflow, and an out-of-range shift amount are likewise
        // runtime contracts, not static type errors.
        (
            "and" | "or" | "add_u" | "sub_u" | "mul_u" | "mul_s" | "div_u" | "rem_u" | "shl_u"
            | "shr_u" | "add_s" | "sub_s" | "div_s" | "rem_s" | "shr_s",
            [Ty::Binary(a), Ty::Binary(b)],
        ) if a == b => Some(Ty::Binary(a.clone())),
        ("add" | "sub" | "mul", [Ty::Ternary(a), Ty::Ternary(b)]) if a == b => {
            Some(Ty::Ternary(a.clone()))
        }
        ("neg", [Ty::Ternary(m)]) => Some(Ty::Ternary(m.clone())),
        _ => None,
    }
}

/// Canonicalize a surface VSA model ident to the **kernel model id** (`Repr::Vsa { model }` —
/// M-892). A surface ident cannot carry the `-` some kernel ids use (`MAP-I`, `MAP-B`), so the
/// underscore spelling maps onto it; every other ident passes through **verbatim** (an unknown
/// model is a legal type *mention* — core type-checks hypervector mentions without the algebra,
/// ADR-008 — and the `vsa_*` prims gate their own dispatch set with an explicit refusal, so
/// nothing here guesses). The mapping is total and append-only.
#[must_use]
pub fn vsa_kernel_model_id(surface: &str) -> String {
    match surface {
        "MAP_I" => "MAP-I".to_owned(),
        "MAP_B" => "MAP-B".to_owned(),
        other => other.to_owned(),
    }
}

/// The surface→kernel prim-name mapping (the `Op` node's `prim` — RFC-0007 §4.1).
#[must_use]
pub fn prim_kernel_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "not" => "bit.not",
        // CU-6: bit-manipulation counts (popcount/clz/ctz), unary Binary{N} -> Binary{N}.
        "popcount" => "bit.popcount",
        "clz" => "bit.clz",
        "ctz" => "bit.ctz",
        "xor" => "bit.xor",
        // RFC-0032 D2 (M-748): surface the already-registered `bit.and`/`bit.or` + never-silent
        // binary `add`/`sub` (distinct surface names from the trit-backed `add`/`sub` below).
        // Surface names here and below follow the DN-72 `_u`/`_s` signedness-suffix convention
        // (ADR-028); the kernel names on the RHS are content-addressed (DN-10 §3.4) and unchanged.
        "and" => "bit.and",
        "or" => "bit.or",
        "add_u" => "bit.add",
        "sub_u" => "bit.sub",
        // RFC-0033 §4.1.2/§4.1.3 (M-887, `enb` Gap B): never-silent two's-complement multiply —
        // the first shared (signedness-agnostic bit-pattern) two's-complement op ADR-028 names.
        "mul_s" => "bin.mul",
        // RFC-0033 §4.1.2 (CU-1): never-silent **unsigned** multiply — overflow-distinct from the
        // signed `mul_s`/`bin.mul` (the `math.myc` FLAG-math-1 missing op).
        "mul_u" => "bit.mul",
        // RFC-0033 §4.1.2/§4.1.3 (M-888, `enb` Gap B): never-silent **unsigned** division/
        // remainder — division must be a distinct-named op per signedness (§4.1.2); the signed
        // reading rides M-767 under its own surface name.
        "div_u" => "bin.div",
        "rem_u" => "bin.rem",
        // RFC-0033 §4.1.2/§4.1.3 (M-889, `enb` Gap B): never-silent **logical** left/right shift —
        // the arithmetic/signed right shift rides M-767 under its own surface name.
        "shl_u" => "bin.shl",
        "shr_u" => "bin.shr",
        // RFC-0033 §4.1.2/§4.1.3 (M-766, `enb` Gap B): never-silent two's-complement add/sub/neg —
        // completes the shared set `mul_s` started. The `_s` suffix marks the signed/two's-
        // complement reading, distinct from the *unsigned* `add_u`/`sub_u` (`bit.add`/`bit.sub`)
        // above (DN-72; see the `prim_family` naming comment).
        "add_s" => "bin.add",
        "sub_s" => "bin.sub",
        "neg_s" => "bin.neg",
        // RFC-0033 §4.1.2/§4.1.3 (M-767, `enb` Gap B): the signedness-split signed op set — the
        // signed truncated division/remainder and the arithmetic (sign-extending) right shift,
        // distinct-named from their `_u` counterparts per ADR-028 (SMT-LIB `bvsdiv`/`bvudiv`,
        // `bvashr`/`bvlshr`). The new kernel names take the `_s` spelling directly; the deferred
        // DN-72 §5 kernel-namespace reconciliation (`bit.*` vs `bin.*`) is not prejudged here.
        "div_s" => "bin.div_s",
        "rem_s" => "bin.rem_s",
        "shr_s" => "bin.shr_s",
        // RFC-0032 D1 (M-747): reduce-to-`Bool` comparison/equality (returns `Binary{1}`).
        "eq" => "cmp.eq",
        "lt" => "cmp.lt",
        // RFC-0033 §4.1.2 (M-767): the signed (two's-complement) order over `Binary{N}` — `lt`
        // reads Binary operands as unsigned magnitudes, so the signed order is a distinct named
        // op (ADR-028's `bvslt`/`bvult` split); Binary-only (ternary's `lt` is already signed).
        "lt_s" => "cmp.lt_s",
        // RFC-0032 D3/D4 (M-749/M-750): never-silent indexing/length over `Seq`/`Bytes` (the surface
        // names are `_`-joined since a `.` is the path separator in the lexer).
        "seq_len" => "seq.len",
        "seq_get" => "seq.get",
        "bytes_len" => "bytes.len",
        "bytes_get" => "bytes.get",
        // DN-43 (M-799): surface the already-landed never-silent byte slice/concat prims (RFC-0032 D4,
        // M-750). Surfacing only — the kernel prims exist + are registered; this maps the surface name.
        "bytes_slice" => "bytes.slice",
        "bytes_concat" => "bytes.concat",
        // M-912 (`enb`, folded-in gap): byte-wise equality — flagged missing by the diag/error/
        // recover ports (`bytes.*` had len/get/slice/concat but no equality).
        "bytes_eq" => "bytes.eq",
        // M-912 (`enb`): the kernel's own BLAKE3 content-addressing hash (M-103;
        // `mycelium-core::content::Canon`/`id::ContentHash`) surfaced as a prim.
        "hash_blake3" => "hash.blake3",
        // DN-41 (M-798): never-silent `Binary` width-cast (zero-extend widen / checked narrow).
        "width_cast" => "bit.width_cast",
        // RFC-0001 §4.1 / RFC-0002 §5 (M-890, `enb` Gap C): the tensor-valued dense elementwise
        // group — kernel `mycelium-dense`, per-op tags carried from `DenseSpace::op_guarantee`
        // (`dense.neg` `Exact`, the rest `Proven`; VR-5). Surface names are `_`-joined like
        // `seq_len` (a `.` is the path separator in the lexer); typed by the dedicated
        // `try_check_dense_prim` branch (shape/dtype in the type ⇒ static never-silent mismatch).
        "dense_add" => "dense.add",
        "dense_sub" => "dense.sub",
        "dense_neg" => "dense.neg",
        "dense_scale" => "dense.scale",
        // M-891: the measurement pair — `Proven`, the kernel's binary64 accumulation bound
        // (absolute/Linf; see `mycelium-interp/src/prims.rs`), result form Dense{1, F64}.
        "dense_dot" => "dense.dot",
        "dense_similarity" => "dense.similarity",
        // ADR-040 §2.5 (M-898, `enb` Gap A): the scalar-float arithmetic group — IEEE-754
        // binary64 under RNE over the nullary `Float` type (M-896/M-897), in-band specials per
        // the ratified FLAG-2, per-op tag `Empirical` per ADR-040 §2.6. Surface names are
        // `_`-joined like `dense_add` (a `.` is the path separator in the lexer); a distinct
        // `flt_*` namespace because floats are their own type — the `_bin`/`_tc` signedness
        // suffixes of the integer set don't apply (there is one float reading, not two). Typed
        // by the dedicated `try_check_float_prim` branch (all operands `Float`, result `Float`).
        "flt_add" => "flt.add",
        "flt_sub" => "flt.sub",
        "flt_mul" => "flt.mul",
        "flt_div" => "flt.div",
        "flt_neg" => "flt.neg",
        // ADR-040 §2.4 (M-899, `enb` Gap A): the scalar-float comparison group — the IEEE-754
        // §5.11 partial-order predicates (NaN explicitly unordered: any NaN operand → the
        // *defined* value false, `flt_eq(NaN, NaN)` = false) plus the **named, opt-in total
        // order** `flt_total_le` (IEEE-754 §5.10 `totalOrder`: −inf < … < −0 < +0 < … < +inf
        // < NaN — reflexive, NaN placed last, the signed zeros directed). Two `Float` operands
        // collapse to `Binary{1}` (the realized `Bool`), typed by `try_check_float_prim`. A
        // distinct namespace from the D1 `eq`/`lt` because float ordering is *partial* — routing
        // floats through the D1 total order would silently invent an order for NaN (G2). Tag:
        // `Empirical` per ADR-040 §2.6; `flt_total_le`'s total-order property stays `Empirical`
        // until the M-511 proof debt is discharged (never `Proven` without the checked theorem).
        "flt_lt" => "flt.lt",
        "flt_le" => "flt.le",
        "flt_gt" => "flt.gt",
        "flt_ge" => "flt.ge",
        "flt_eq" => "flt.eq",
        "flt_total_le" => "flt.total_le",
        // ADR-040 §2.5 (CU-2): the mandated float classification predicates.
        "flt_is_nan" => "flt.is_nan",
        "flt_is_finite" => "flt.is_finite",
        "flt_is_infinite" => "flt.is_infinite",
        // ADR-040 §2.4 (CU-3): the never-silent Binary↔Float conversions — the "target-width
        // prim" shape of `width_cast` (DN-41). `bin_to_flt` is checked-exact (unary,
        // `Binary{N} -> Float`); `flt_to_bin` reads its target width off a witness operand
        // (`Float, Binary{M} -> Binary{M}`), exactly like `width_cast`. Typed by the dedicated
        // mixed-paradigm pre-branch in `try_check_float_prim` (not the uniform-Float loop below).
        "bin_to_flt" => "bin.to_flt",
        "flt_to_bin" => "flt.to_bin",
        // RFC-0003 §3/§4 / ADR-008 (M-892, `enb` Gap C): the model-dispatched VSA bind group —
        // kernel `mycelium-vsa`, per-model tags carried from the model's Value-level op (MAP-I/
        // BSC ops `Exact`; FHRR `unbind` `Empirical` with its trial-validated δ — VR-5, never
        // re-stamped; the Π-table intrinsic is the meet over the MAP-I/FHRR/BSC dispatch set).
        // Surface names are `_`-joined like `dense_add`; typed by the dedicated
        // `try_check_vsa_prim` branch (model + dim in the type ⇒ static never-silent mismatch;
        // an out-of-set model is a static refusal naming the set).
        "vsa_bind" => "vsa.bind",
        "vsa_unbind" => "vsa.unbind",
        "vsa_permute" => "vsa.permute",
        // RFC-0003 §4/§5 / ADR-008 (M-893, `enb` Gap C): the certified VSA superposition —
        // kernel `vsa.bundle` via MAP-I's `bundle_values_certified` (the M-131 checked-
        // instantiation pattern: a `Proven` `CapacityBound` iff `dim ≥ requiredDim(m, δ)` with
        // bipolar + distinct items checked, else an explicit refusal). Typed by the
        // `try_check_vsa_prim` branch: `Seq{VSA{…}, N≥1}` × `Float` δ → `VSA{…}`, with the
        // certified-singleton {MAP-I} dispatch and the empty bundle refused *statically* (N is
        // in the type); FHRR/BSC bundles are Empirical-profile kernel ops — statically refused
        // here naming the certified set, an append-only future surfacing (VR-5).
        "vsa_bundle" => "vsa.bundle",
        // RFC-0003 §3/§5/§6 / ADR-008 (M-894, `enb` Gap C): the cleanup/reconstruction pair +
        // the capacity-bound query. `vsa.cleanup` (query × codebook) and `vsa.reconstruct`
        // (record × role × codebook × threshold) return the `Seq{Float, 3}` `[index, confidence,
        // margin]` decision triple — retrieval is never a silent nearest-neighbour pick
        // (FR-S4/G2), a tie is the RFC-0010 §4.4 identifiability refusal, and the query/record's
        // own (strength, bound) pair carries through (a certified bundle record re-discloses its
        // `Proven` `CapacityBound` — the disclosed bound is the value's own; VR-5).
        // `vsa.required_dim` (items × δ → `Binary{64}`) surfaces the M-131
        // `requiredDim`/`proven_capacity_bound` checked instantiation, its result carrying the
        // kernel's `Proven` `CapacityBound`. Typed by `try_check_vsa_prim`: cleanup dispatches
        // the full MAP-I/FHRR/BSC set (the arg-max is model-generic); reconstruct the
        // self-inverse {MAP-I, BSC} (FHRR statically refused naming its unbind profile's
        // regime); the factor decode (`reconstruct_factors`, RFC-0009/0010) is deliberately not
        // surfaced here — its RFC-0005 selector's mandatory EXPLAIN has no prim-surface carrier
        // yet — a distinct append-only surfacing under its own name.
        "vsa_cleanup" => "vsa.cleanup",
        "vsa_reconstruct" => "vsa.reconstruct",
        "vsa_required_dim" => "vsa.required_dim",
        "add" => "trit.add",
        "sub" => "trit.sub",
        "mul" => "trit.mul",
        "neg" => "trit.neg",
        _ => return None,
    })
}
