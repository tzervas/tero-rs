//! **Ambient representation resolution** (RFC-0012; enacts M-344). The ambient is a *declared,
//! scoped, paradigm-only* default that offsets honesty's verbosity (tension **A**) **without**
//! becoming a black box. This module is the enactment's honest core.
//!
//! # The architecture: a surface→surface "expand to longhand" pass
//! [`resolve`] rewrites a parsed [`Nodule`] into its **longhand twin**: every paradigm-less repr
//! `{…}` is replaced by the concrete `P{…}` of the enclosing ambient, every `with paradigm` block
//! is stripped (after filling its interior tags), and every bare decimal under an ambient is tagged
//! [`Literal::AmbientInt`] (its *width* is resolved later, by the checker, from context — §4.3).
//! The **unchanged** `check_nodule → elaborate` pipeline then runs on the twin. This makes
//! RFC-0012's two normative invariants true *by construction*:
//!
//! - **(I1) the ambient emits no `Swap`** — resolution only fills tags/encodings; it has no rule
//!   that inserts a [`Expr::Swap`] (conversions stay author-written, WF1);
//! - **(I2) resolution is observationally the identity** — `elaborate(p) = elaborate(resolve(p))`,
//!   and `resolve(p)` *is* the longhand twin a reader would write, so the content hashes coincide
//!   (RFC-0001 §4.6). The §4.6 meaning-preservation differential (`tests/ambient.rs`) is the
//!   executable proof.
//!
//! # Never-silent refusals (§4.3/§4.4), the no-black-box guarantee
//! - [`AmbientError::UnresolvedAmbient`] — a `{…}` with no enclosing ambient (no implicit global
//!   fallback — that would be silent).
//! - [`AmbientError::ParadigmShapeMismatch`] — a written shape that does not fit the ambient
//!   paradigm (e.g. `{8}` under a `Dense` ambient) — never a coerced guess.
//! - [`AmbientError::BareDecimalNoEncoding`] — a bare decimal under a `Dense`/`VSA` ambient (those
//!   paradigms have no bare-decimal encoding).
//! - [`AmbientError::MultipleDefaults`] — two nodule-scope `default paradigm` declarations.
//!
//! The cross-paradigm `MissingConversion` refusal (§4.4) and the bare-decimal `UnresolvedWidth`
//! refusal (§4.3) need *types*, so they live in the checker ([`crate::checkty`]); they are the
//! never-silent edge-of-the-feature guarantees.
//!
//! # Provenance / EXPLAIN (§4.3), the no-black-box guarantee #2
//! [`resolve_report`] returns, alongside the twin, a [`ResolutionNote`] per fill — *where did this
//! paradigm come from?* answered for every resolved site. The resolved longhand is also renderable
//! ([`expand_to_source`]) so the elided default is never *hidden*, only *elided* (the M-142/LSP
//! "expand ambient" projection; §5). Realization note (KC-3): the provenance is recorded at the
//! surface/resolution layer rather than as a new `mycelium_core::Provenance` variant — that would
//! change a frozen data-contract schema (`provenance.schema.json`) for metadata that is not hashed
//! (RFC-0001 §4.6) and is fully recoverable here. See the RFC-0012 changelog (append-only).

use crate::ast::{
    AmbientParams, Arm, BaseType, Ctor, DeriveDecl, Expr, FnDecl, FnSig, ImplDecl,
    InherentImplDecl, Item, Literal, LowerDecl, LowerRhs, Nodule, ObjectDecl, Paradigm, Param,
    ParamKind, Pattern, Phylum, Scalar, Sparsity, TraitDecl, TraitRef, TypeDecl, TypeParam,
    TypeRef, UsePath, Vis, WidthRef,
};

/// A never-silent refusal from the resolution pass (§4.3/§4.4) — always explicit, never a guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AmbientError {
    /// Two nodule-scope `default paradigm` declarations — the outer frame is ambiguous.
    MultipleDefaults {
        /// The first declared paradigm.
        first: Paradigm,
        /// The second (rejected) declaration.
        second: Paradigm,
    },
    /// A paradigm-less repr `{…}` with **no enclosing ambient** (§4.3) — there is no implicit
    /// global fallback (that would be silent).
    UnresolvedAmbient {
        /// The item/definition being resolved.
        site: String,
    },
    /// A written shape that does not fit the ambient paradigm (§4.3) — never coerced.
    ParadigmShapeMismatch {
        /// The item/definition being resolved.
        site: String,
        /// The ambient paradigm in force.
        paradigm: Paradigm,
        /// Why the written shape does not fit it.
        detail: String,
    },
    /// A bare decimal under a `Dense`/`VSA` ambient (§4.3): those paradigms have no bare-decimal
    /// encoding (the integer paradigms `Binary`/`Ternary` do).
    BareDecimalNoEncoding {
        /// The item/definition being resolved.
        site: String,
        /// The ambient paradigm in force.
        paradigm: Paradigm,
    },
    /// The resolution pass's OWN recursive descent over `expr`/`type_ref`/`pattern` exceeded its
    /// explicit [`MAX_AMBIENT_DEPTH`] budget (M-674 remaining TODO item 2) — the compiler pass's
    /// analysis recursion over the AST, not a claim about the checked program's semantics. Refused
    /// cleanly here rather than overflowing the host stack (banked guard 4; A4-02), mirroring the
    /// checker's `MAX_CHECK_DEPTH` discipline.
    DepthExceeded {
        /// The item/definition being resolved when the budget was exceeded.
        site: String,
        /// The exceeded budget.
        limit: u32,
    },
}

impl core::fmt::Display for AmbientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AmbientError::MultipleDefaults { first, second } => write!(
                f,
                "two `default paradigm` declarations (`{first}` then `{second}`) — a nodule has one \
                 outer ambient (RFC-0012 §4.2); nest a `with paradigm` block for a local override"
            ),
            AmbientError::UnresolvedAmbient { site } => write!(
                f,
                "`{site}`: a paradigm-less repr `{{…}}` has no enclosing ambient — declare \
                 `default paradigm P` (or wrap in `with paradigm P {{ … }}`), or write the paradigm \
                 explicitly. There is no implicit global default (RFC-0012 §4.3, never-silent)"
            ),
            AmbientError::ParadigmShapeMismatch {
                site,
                paradigm,
                detail,
            } => write!(
                f,
                "`{site}`: the paradigm-less repr does not fit the `{paradigm}` ambient — {detail} \
                 (RFC-0012 §4.3; the shape is never coerced)"
            ),
            AmbientError::BareDecimalNoEncoding { site, paradigm } => write!(
                f,
                "`{site}`: a bare decimal has no `{paradigm}` encoding — only `Binary`/`Ternary` \
                 ambients give a bare decimal a meaning (RFC-0012 §4.3); write the value explicitly"
            ),
            AmbientError::DepthExceeded { site, limit } => write!(
                f,
                "`{site}`: AST nesting exceeds the ambient resolution pass's own recursion-depth \
                 budget ({limit}) — an explicit budget (banked guard 4; M-674), refused cleanly \
                 rather than overflowing the host stack"
            ),
        }
    }
}

impl std::error::Error for AmbientError {}

/// A record of one ambient fill, for EXPLAIN / "where did this paradigm come from?" (§4.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionNote {
    /// The item/definition the fill occurred in.
    pub site: String,
    /// The paradigm supplied by the ambient.
    pub paradigm: Paradigm,
    /// What was filled (a repr tag or a bare-decimal encoding).
    pub detail: String,
}

/// The resolved twin plus its provenance trace.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// The longhand twin (paradigm-less forms filled; `with` blocks stripped; bare decimals tagged).
    pub nodule: Nodule,
    /// One note per ambient fill (provenance / EXPLAIN; §4.3).
    pub notes: Vec<ResolutionNote>,
}

/// Resolve a parsed [`Nodule`] to its longhand twin (RFC-0012 §4.3/§4.4). Identity on a program
/// that uses no ambient — the feature is purely additive (opt-in), so every pre-RFC-0012 program
/// passes through unchanged.
///
/// # Errors
/// Returns an [`AmbientError`] for any never-silent refusal (unresolved/unshaped ambient, a bare
/// decimal with no encoding, or a duplicate nodule default).
pub fn resolve(nodule: &Nodule) -> Result<Nodule, AmbientError> {
    resolve_report(nodule).map(|r| r.nodule)
}

/// Like [`resolve`], but also returns the provenance trace ([`ResolutionNote`]s) for EXPLAIN (§4.3).
///
/// Runs on [`mycelium_stack`]'s deep worker stack (M-674), mirroring the checker's/evaluator's
/// discipline: [`MAX_AMBIENT_DEPTH`] — not a host-stack overflow — is always what bounds a
/// pathologically-nested `nodule`, regardless of the caller's own thread-stack size. Resolution runs
/// *before* `checkty`'s own deep-stack wrapping (`check_phylum_matured`), so it needs its own —
/// nesting worker threads is cheap (tens of microseconds) and harmless.
///
/// # Errors
/// See [`resolve`].
pub fn resolve_report(nodule: &Nodule) -> Result<Resolved, AmbientError> {
    mycelium_stack::with_deep_stack(|| resolve_report_inner(nodule))
}

fn resolve_report_inner(nodule: &Nodule) -> Result<Resolved, AmbientError> {
    // The nodule-scope ambient: at most one `default paradigm`, the outermost frame. It governs all
    // signature types and is the base expression ambient.
    let mut default: Option<Paradigm> = None;
    for item in &nodule.items {
        if let Item::Default(p) = item {
            if let Some(first) = default {
                return Err(AmbientError::MultipleDefaults { first, second: *p });
            }
            default = Some(*p);
        }
    }

    let mut r = Resolver { notes: Vec::new() };
    let mut items = Vec::with_capacity(nodule.items.len());
    for item in &nodule.items {
        match item {
            // The default declaration is consumed (it is metadata for resolution, not a runtime
            // item); the longhand twin carries no ambient.
            Item::Default(_) => {}
            Item::Use(p) => items.push(Item::Use(p.clone())),
            Item::Type(td) => items.push(Item::Type(r.type_decl(default, td)?)),
            Item::Trait(td) => items.push(Item::Trait(r.trait_decl(default, td)?)),
            Item::Impl(id) => items.push(Item::Impl(r.impl_decl(default, id)?)),
            Item::Fn(fd) => items.push(Item::Fn(r.fn_decl(default, fd)?)),
            // DN-53 M-811: resolve ambient representation inside the object body — constructor
            // field types, `via` trait arguments, explicit `impl` method bodies, and inherent `fn`
            // bodies — so that `check_phylum_inner`'s Phase 0 structural expansion produces
            // already-resolved `Item::Type`/`Item::Impl`/`Item::Fn` items. Without this pass the
            // expanded items would carry unresolved paradigm-less repr sites (a never-silent defect
            // — G2). The ambient for the object body is the same nodule-level ambient as for any
            // top-level declaration (there is no inner ambient scope inside an `object` body).
            Item::Object(od) => items.push(Item::Object(r.object_decl(default, od)?)),
            // DN-54 / M-812: `lower`/`derive` declarations carry no ambient-paradigm parameters —
            // the rule's RHS is a typed L1 term that is already unambiguous (no bare repr, no
            // ambient integer). Pass through unchanged; the type-checker validates the RHS.
            Item::Lower(ld) => items.push(Item::Lower(ld.clone())),
            Item::Derive(dd) => items.push(Item::Derive(dd.clone())),
            // M-664: an inherent `impl T { fn … }` block — resolve the target type + method bodies
            // (Phase 0 desugars it to its `Item::Fn`s afterward).
            Item::InherentImpl(id) => {
                items.push(Item::InherentImpl(r.inherent_impl_decl(default, id)?));
            }
        }
    }
    Ok(Resolved {
        nodule: Nodule {
            path: nodule.path.clone(),
            // The `@std-sys` FFI-floor marker (M-661) is carried through resolution unchanged — the
            // checker runs on this longhand twin and gates `wild` on it, so dropping it here would
            // make every `std-sys` `wild` block spuriously refused (the marker is not ambient state).
            std_sys: nodule.std_sys,
            items,
        },
        notes: r.notes,
    })
}

/// Render a (resolved or partly-resolved) [`Nodule`] back to canonical surface text — the M-142/LSP
/// **"expand ambient"** projection (RFC-0012 §5; R12-Q3). Fed the fully-resolved twin (post-checker,
/// so bare decimals are concrete literals), it prints the exact longhand a reader would write; fed
/// the type-form-resolved nodule, it still shows every resolved *paradigm* (an `AmbientInt` renders
/// as its decimal with a paradigm note, since its width is the checker's to fill).
#[must_use]
pub fn expand_to_source(nodule: &Nodule) -> String {
    let mut out = String::new();
    // Re-emit the `@std-sys` FFI-floor marker (M-661) when present: dropping it would silently
    // relocate audited `wild` code into a non-`@std-sys` context (changing program meaning — G2),
    // so the longhand twin must round-trip the header attribute.
    //
    // DN-57 §3 (M-818): the nodule header is a component — it ends with a mandatory `;`. The
    // header/body boundary is the `;` token, so the canonical form is whitespace-independent and
    // re-parses under the mandatory-terminator grammar.
    out.push_str(&format!(
        "nodule {}{};\n",
        path_str(&nodule.path),
        if nodule.std_sys { " @std-sys" } else { "" }
    ));
    for item in &nodule.items {
        out.push('\n');
        // Each `print_*` emits the item text ending in `\n`; DN-57 §3 (M-818) appends the mandatory
        // `;` component terminator via `terminate_item`, *uniformly* — a `}`-closed block (`trait`,
        // `impl`, `object`) still gets the trailing `;`.
        let item_text = match item {
            Item::Use(u) => print_use(u),
            Item::Default(p) => format!("default paradigm {p}\n"),
            Item::Type(td) => print_type_decl(td),
            Item::Trait(td) => print_trait_decl(td),
            Item::Impl(id) => print_impl_decl(id),
            Item::Fn(fd) => print_fn_decl(fd),
            // `object` declarations are rendered in surface form (not desugared) — the LSP
            // "expand ambient" shows the source as-written with paradigms filled in. After
            // Phase 0 expansion in the checker the desugared items (type + impls + fns) are in
            // scope; here we emit the pre-desugar surface so the round-trip is stable.
            Item::Object(od) => print_object_decl(od),
            // DN-54 / M-812: `lower`/`derive` declarations round-trip verbatim through the
            // ambient expansion pass (no ambient state to fill; rule RHS has no bare reprs).
            Item::Lower(ld) => print_lower_decl(ld),
            Item::Derive(dd) => print_derive_decl(dd),
            // M-664: an inherent method block round-trips in surface form (pre-desugar), like
            // `object` — the Phase 0 desugar to `Item::Fn`s happens later in the checker.
            Item::InherentImpl(id) => print_inherent_impl_decl(id),
        };
        out.push_str(&terminate_item(&item_text));
    }
    out
}

/// Append the mandatory `;` component terminator (DN-57 §3, M-818) to a rendered item. Each
/// `print_*` produces text ending in a single trailing `\n`; this replaces that `\n` with `;\n` so
/// the item ends in exactly one `;` (uniform across expression items and `}`-closed blocks). The
/// terminator goes *after* the closing `}` of a block item (`trait`/`impl`/`object`), matching the
/// parser's `expect_terminator` after `}` consumption.
fn terminate_item(item_text: &str) -> String {
    match item_text.strip_suffix('\n') {
        Some(body) => format!("{body};\n"),
        None => format!("{item_text};"),
    }
}

/// Render a whole [`Phylum`] back to canonical surface text (M-662): the optional `phylum <path>`
/// header, then each `nodule` block via [`expand_to_source`]. Round-trips the phylum header, every
/// `pub` marker, and `use` (specific + glob) verbatim, so `parse_phylum → expand → parse_phylum` is
/// stable (the LSP "expand ambient" / EXPLAIN projection over a multi-nodule source — RFC-0012 §5).
#[must_use]
pub fn expand_phylum_to_source(phylum: &Phylum) -> String {
    let mut out = String::new();
    if let Some(path) = &phylum.path {
        out.push_str(&format!("phylum {}\n", path_str(path)));
    }
    for nodule in &phylum.nodules {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&expand_to_source(nodule));
    }
    out
}

/// Explicit depth budget for the ambient resolution pass's OWN recursive descent over
/// `Resolver::expr`/`type_ref`/`pattern` (M-674 remaining TODO item 2) — the compiler pass's
/// analysis recursion, distinct from any RFC-0012 semantic concept (resolution has no notion of
/// non-termination). Mirrors the checker's `MAX_CHECK_DEPTH` discipline (banked guard 4; A4-02):
/// rather than rely on the host call stack (a resource that is not a semantic limit), the pass
/// carries this reified budget and refuses past it with a clean [`AmbientError::DepthExceeded`],
/// never a host-stack overflow. Set comfortably above the parser's `MAX_EXPR_DEPTH` (256)
/// surface-nesting cap, so no parser-produced AST ever approaches it — the ceiling exists as
/// defense-in-depth for a synthetic/API-built tree handed directly to [`resolve`]/[`resolve_report`].
///
/// **Grounding (measured, not guessed).** `resolve_report` runs on [`mycelium_stack`]'s 256 MiB deep
/// worker stack (below); measured empirically, `Resolver::expr`'s own recursion survives at least
/// 1,000,000 levels on that stack without overflowing. This budget (`4096`) is therefore comfortably
/// (>200×) below the measured physical floor, and **16×** above the parser's 256-deep surface cap.
pub(crate) const MAX_AMBIENT_DEPTH: u32 = 4096;

/// The resolution worker: holds the provenance trace; the ambient paradigm is threaded as an
/// argument (innermost-enclosing-wins, a binder-like stack), never as mutable state. The recursion
/// depth (M-674) is threaded the same way — an explicit argument to `expr`/`type_ref`/`pattern`,
/// reset to `0` at the top of each declaration's resolution (mirroring the checker's per-body reset).
struct Resolver {
    notes: Vec<ResolutionNote>,
}

impl Resolver {
    /// Charge one level of nesting against the explicit [`MAX_AMBIENT_DEPTH`] budget (banked guard
    /// 4): returns the incremented depth, or a clean [`AmbientError::DepthExceeded`] past the
    /// budget — never a host-stack overflow.
    fn enter_depth(site: &str, depth: u32) -> Result<u32, AmbientError> {
        let depth = depth + 1;
        if depth > MAX_AMBIENT_DEPTH {
            return Err(AmbientError::DepthExceeded {
                site: site.to_owned(),
                limit: MAX_AMBIENT_DEPTH,
            });
        }
        Ok(depth)
    }
    fn note(&mut self, site: &str, paradigm: Paradigm, detail: impl Into<String>) {
        self.notes.push(ResolutionNote {
            site: site.to_owned(),
            paradigm,
            detail: detail.into(),
        });
    }

    fn type_decl(
        &mut self,
        amb: Option<Paradigm>,
        td: &TypeDecl,
    ) -> Result<TypeDecl, AmbientError> {
        let mut ctors = Vec::with_capacity(td.ctors.len());
        for c in &td.ctors {
            let mut fields = Vec::with_capacity(c.fields.len());
            for f in &c.fields {
                fields.push(self.type_ref(amb, &td.name, f, 0)?);
            }
            ctors.push(Ctor {
                name: c.name.clone(),
                fields,
            });
        }
        Ok(TypeDecl {
            // Visibility is surface metadata, untouched by ambient resolution (M-662).
            vis: td.vis,
            name: td.name.clone(),
            params: td.params.clone(),
            ctors,
        })
    }

    fn trait_decl(
        &mut self,
        amb: Option<Paradigm>,
        td: &TraitDecl,
    ) -> Result<TraitDecl, AmbientError> {
        let mut sigs = Vec::with_capacity(td.sigs.len());
        for s in &td.sigs {
            sigs.push(self.fn_sig(amb, s)?);
        }
        Ok(TraitDecl {
            // Visibility is surface metadata, untouched by ambient resolution (M-662).
            vis: td.vis,
            name: td.name.clone(),
            params: td.params.clone(),
            sigs,
        })
    }

    /// Resolve an `impl Trait<args> for T { fn … }` (RFC-0019 §4.1): the trait arguments, the
    /// `for` type, and each method (signature + body) all resolve under the nodule ambient — the
    /// same passes the surrounding `type`/`fn` use, so an impl never sees an unresolved paradigm-less
    /// repr reach the checker (defense in depth — RFC-0012 §4.3).
    fn impl_decl(
        &mut self,
        amb: Option<Paradigm>,
        id: &ImplDecl,
    ) -> Result<ImplDecl, AmbientError> {
        let site = &id.trait_name;
        let mut trait_args = Vec::with_capacity(id.trait_args.len());
        for a in &id.trait_args {
            trait_args.push(self.type_ref(amb, site, a, 0)?);
        }
        let for_ty = self.type_ref(amb, site, &id.for_ty, 0)?;
        let mut methods = Vec::with_capacity(id.methods.len());
        for m in &id.methods {
            methods.push(self.fn_decl(amb, m)?);
        }
        Ok(ImplDecl {
            trait_name: id.trait_name.clone(),
            trait_args,
            for_ty,
            methods,
        })
    }

    /// Resolve an inherent method block `impl T { fn … }` (M-664): ambient-fill the target type and
    /// every method body, mirroring [`Self::impl_decl`] minus the trait reference. The desugar in
    /// `checkty.rs` Phase 0 then sees already-resolved `for_ty`/methods.
    fn inherent_impl_decl(
        &mut self,
        amb: Option<Paradigm>,
        id: &InherentImplDecl,
    ) -> Result<InherentImplDecl, AmbientError> {
        let site = "impl";
        let for_ty = self.type_ref(amb, site, &id.for_ty, 0)?;
        let mut methods = Vec::with_capacity(id.methods.len());
        for m in &id.methods {
            methods.push(self.fn_decl(amb, m)?);
        }
        Ok(InherentImplDecl { for_ty, methods })
    }

    fn fn_decl(&mut self, amb: Option<Paradigm>, fd: &FnDecl) -> Result<FnDecl, AmbientError> {
        let sig = self.fn_sig(amb, &fd.sig)?;
        // The function body resolves under the nodule ambient as its base frame; `with paradigm`
        // blocks nest *inside* it. Signatures (above) never see a block-scope override. The
        // recursion-depth budget (M-674) resets to `0` per body, mirroring the checker's per-body
        // reset — each declaration's traversal is independently bounded.
        let body = self.expr(amb, &fd.sig.name, &fd.body, 0)?;
        Ok(FnDecl {
            // Visibility is surface metadata, untouched by ambient resolution (M-662).
            vis: fd.vis,
            thaw: fd.thaw,
            // `@tier` is surface metadata too — pass it through unchanged (DN-58 §C; M-667).
            tier: fd.tier,
            sig,
            body,
        })
    }

    /// Resolve an `object` composition surface declaration (DN-53, M-811): ambient-fill the
    /// constructor field types, the `via` trait arguments, and the bodies of every explicit `impl`
    /// and inherent `fn` in the body. Visibility and structural names are surface metadata,
    /// untouched by ambient resolution.
    fn object_decl(
        &mut self,
        amb: Option<Paradigm>,
        od: &ObjectDecl,
    ) -> Result<ObjectDecl, AmbientError> {
        let site = &od.name;
        // Resolve the constructor field types.
        let ctor = {
            let mut fields = Vec::with_capacity(od.ctor.fields.len());
            for f in &od.ctor.fields {
                fields.push(self.type_ref(amb, site, f, 0)?);
            }
            Ctor {
                name: od.ctor.name.clone(),
                fields,
            }
        };
        // Resolve `via` trait arguments.
        let mut via_decls = Vec::with_capacity(od.via_decls.len());
        for via in &od.via_decls {
            let mut trait_args = Vec::with_capacity(via.trait_args.len());
            for a in &via.trait_args {
                trait_args.push(self.type_ref(amb, site, a, 0)?);
            }
            via_decls.push(crate::ast::ViaDecl {
                field_idx: via.field_idx,
                trait_name: via.trait_name.clone(),
                trait_args,
            });
        }
        // Resolve explicit `impl` blocks.
        let mut impls = Vec::with_capacity(od.impls.len());
        for id in &od.impls {
            impls.push(self.impl_decl(amb, id)?);
        }
        // Resolve inherent `fn` declarations.
        let mut fns = Vec::with_capacity(od.fns.len());
        for fd in &od.fns {
            fns.push(self.fn_decl(amb, fd)?);
        }
        Ok(ObjectDecl {
            vis: od.vis,
            name: od.name.clone(),
            params: od.params.clone(),
            ctor,
            via_decls,
            impls,
            fns,
        })
    }

    fn fn_sig(&mut self, amb: Option<Paradigm>, s: &FnSig) -> Result<FnSig, AmbientError> {
        let mut value_params = Vec::with_capacity(s.value_params.len());
        for p in &s.value_params {
            value_params.push(Param {
                name: p.name.clone(),
                ty: self.type_ref(amb, &s.name, &p.ty, 0)?,
            });
        }
        let ret = self.type_ref(amb, &s.name, &s.ret, 0)?;
        Ok(FnSig {
            name: s.name.clone(),
            params: s.params.clone(),
            value_params,
            ret,
            // Effects and their budgets are checker/runtime metadata with no ambient/paradigm
            // resolution — they carry through the ambient pass verbatim (M-660/M-677; RFC-0014 §4.5).
            effects: s.effects.clone(),
            effect_budgets: s.effect_budgets.clone(),
        })
    }

    /// Resolve a [`TypeRef`]: a paradigm-less base is filled from `amb`; everything else passes
    /// through (the guarantee index is unaffected — VR-5).
    ///
    /// # Errors
    /// [`AmbientError::DepthExceeded`] once this traversal's own recursion exceeds
    /// [`MAX_AMBIENT_DEPTH`] (M-674) — a clean, explicit refusal rather than a host-stack overflow
    /// on a pathologically-nested `t`.
    fn type_ref(
        &mut self,
        amb: Option<Paradigm>,
        site: &str,
        t: &TypeRef,
        depth: u32,
    ) -> Result<TypeRef, AmbientError> {
        let depth = Self::enter_depth(site, depth)?;
        let base = match &t.base {
            BaseType::Ambient(params) => {
                let p = amb.ok_or_else(|| AmbientError::UnresolvedAmbient {
                    site: site.to_owned(),
                })?;
                let base = fill_repr(site, p, params)?;
                self.note(site, p, format!("{}", DisplayBase(&base)));
                base
            }
            // Named types may carry paradigm-less type arguments; resolve recursively.
            BaseType::Named(name, args) => {
                let mut out = Vec::with_capacity(args.len());
                for a in args {
                    out.push(self.type_ref(amb, site, a, depth)?);
                }
                BaseType::Named(name.clone(), out)
            }
            // Function types carry two inner TypeRefs; resolve each so that a paradigm-less
            // repr nested inside `A -> B` is filled in context (RFC-0024 §3, M-685).
            BaseType::Fn(arg, ret) => BaseType::Fn(
                Box::new(self.type_ref(amb, site, arg, depth)?),
                Box::new(self.type_ref(amb, site, ret, depth)?),
            ),
            other => other.clone(),
        };
        Ok(TypeRef {
            base,
            guarantee: t.guarantee,
        })
    }

    /// Resolve an expression under the current ambient `amb`. `with paradigm P { e }` recurses with
    /// `amb = Some(P)` and returns the resolved body (the block is stripped — I1: no node inserted).
    ///
    /// # Errors
    /// [`AmbientError::DepthExceeded`] once this traversal's own recursion exceeds
    /// [`MAX_AMBIENT_DEPTH`] (M-674) — a clean, explicit refusal rather than a host-stack overflow
    /// on a pathologically-nested `e`.
    fn expr(
        &mut self,
        amb: Option<Paradigm>,
        site: &str,
        e: &Expr,
        depth: u32,
    ) -> Result<Expr, AmbientError> {
        let depth = Self::enter_depth(site, depth)?;
        Ok(match e {
            Expr::WithParadigm { paradigm, body } => {
                self.expr(Some(*paradigm), site, body, depth)?
            }
            Expr::Lit(l) => Expr::Lit(self.literal(amb, site, l, depth)?),
            Expr::Path(p) => Expr::Path(p.clone()),
            Expr::Let {
                name,
                ty,
                bound,
                body,
            } => Expr::Let {
                name: name.clone(),
                ty: match ty {
                    Some(t) => Some(self.type_ref(amb, site, t, depth)?),
                    None => None,
                },
                bound: Box::new(self.expr(amb, site, bound, depth)?),
                body: Box::new(self.expr(amb, site, body, depth)?),
            },
            Expr::If { cond, conseq, alt } => Expr::If {
                cond: Box::new(self.expr(amb, site, cond, depth)?),
                conseq: Box::new(self.expr(amb, site, conseq, depth)?),
                alt: Box::new(self.expr(amb, site, alt, depth)?),
            },
            Expr::Match { scrutinee, arms } => {
                let mut out = Vec::with_capacity(arms.len());
                for arm in arms {
                    out.push(Arm {
                        pattern: self.pattern(amb, site, &arm.pattern, depth)?,
                        body: self.expr(amb, site, &arm.body, depth)?,
                    });
                }
                Expr::Match {
                    scrutinee: Box::new(self.expr(amb, site, scrutinee, depth)?),
                    arms: out,
                }
            }
            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => Expr::For {
                x: x.clone(),
                xs: Box::new(self.expr(amb, site, xs, depth)?),
                acc: acc.clone(),
                init: Box::new(self.expr(amb, site, init, depth)?),
                body: Box::new(self.expr(amb, site, body, depth)?),
            },
            Expr::Swap {
                value,
                target,
                policy,
            } => Expr::Swap {
                value: Box::new(self.expr(amb, site, value, depth)?),
                target: self.type_ref(amb, site, target, depth)?,
                policy: policy.clone(),
            },
            // `wild` is the audited/opaque FFI escape (M-661): its body is trusted foreign code,
            // preserved **verbatim** — ambient resolution does NOT descend into it. Descending would
            // (a) rewrite the interior (contradicting the "no interior resolution" contract) and
            // (b) raise interior ambient errors (e.g. a bare decimal under a non-integer ambient)
            // from a body that should be opaque/trusted — a surprising refusal. Keeping it a leaf
            // makes `wild` opaque end-to-end, matching `Cx::check_wild` + `totality::walk_expr`
            // (audited, not verified — VR-5/ADR-014; RFC-0016 §8-Q6). `spore(value)` wraps a *real*
            // value expression (deferred — E2-5/M-260), so it still resolves transparently.
            Expr::Wild(b) => Expr::Wild(b.clone()),
            Expr::Spore(b) => Expr::Spore(Box::new(self.expr(amb, site, b, depth)?)),
            // M-664: `consume <expr>` — resolve the operand's ambient (the operand is an ordinary
            // value expression; the `Substrate`-type check is the checker's job).
            Expr::Consume(b) => Expr::Consume(Box::new(self.expr(amb, site, b, depth)?)),
            // A `lambda` body flows transparently under the same ambient (no new ambient frame); the
            // params carry their own explicit types. (Deferred form — M-704 — but resolved like any expr.)
            Expr::Lambda { params, body } => Expr::Lambda {
                params: params.clone(),
                body: Box::new(self.expr(amb, site, body, depth)?),
            },
            // A `colony`'s ambient flows transparently into each `hypha` body (no new ambient frame;
            // RFC-0008 §4.7). Resolve every hypha body under the same `amb`.
            Expr::Colony(hyphae) => {
                let mut out = Vec::with_capacity(hyphae.len());
                for h in hyphae {
                    // M-906 (DN-70 D1): a hypha's optional `@forage(policy)` annotation resolves
                    // under the same ambient as its body (no new ambient frame — it is a plain
                    // value expression, like `reclaim`'s policy).
                    let forage = match &h.forage {
                        Some(p) => Some(Box::new(self.expr(amb, site, p, depth)?)),
                        None => None,
                    };
                    out.push(crate::ast::Hypha {
                        forage,
                        body: self.expr(amb, site, &h.body, depth)?,
                    });
                }
                Expr::Colony(out)
            }
            Expr::App { head, args } => {
                let mut out = Vec::with_capacity(args.len());
                for a in args {
                    out.push(self.expr(amb, site, a, depth)?);
                }
                Expr::App {
                    head: Box::new(self.expr(amb, site, head, depth)?),
                    args: out,
                }
            }
            Expr::Ascribe(inner, t) => Expr::Ascribe(
                Box::new(self.expr(amb, site, inner, depth)?),
                self.type_ref(amb, site, t, depth)?,
            ),
            // DN-58 §A/§B (M-667): `fuse(a, b)` and `reclaim(policy) { body }` — both operands
            // flow under the same ambient frame (no new paradigm context). Resolve transparently.
            Expr::Fuse { left, right } => Expr::Fuse {
                left: Box::new(self.expr(amb, site, left, depth)?),
                right: Box::new(self.expr(amb, site, right, depth)?),
            },
            Expr::Reclaim { policy, body } => Expr::Reclaim {
                policy: Box::new(self.expr(amb, site, policy, depth)?),
                body: Box::new(self.expr(amb, site, body, depth)?),
            },
            // M-826: propagate the ambient paradigm into each element of a tuple literal.
            // A tuple literal `(a, b, …)` may contain repr literals (e.g. bare decimals); each
            // element gets the same ambient as the enclosing context.
            Expr::TupleLit(elems) => {
                let mut out = Vec::with_capacity(elems.len());
                for el in elems {
                    out.push(self.expr(amb, site, el, depth)?);
                }
                Expr::TupleLit(out)
            }
        })
    }

    /// Resolve a literal: a bare decimal under an *integer* ambient becomes [`Literal::AmbientInt`]
    /// (the checker fills its width); under `Dense`/`VSA` it is a never-silent refusal; with no
    /// ambient it stays [`Literal::Int`] (the checker's existing "no representation family" rule
    /// applies — unchanged status quo). Tagged literals are unaffected (§4.3).
    fn literal(
        &mut self,
        amb: Option<Paradigm>,
        site: &str,
        l: &Literal,
        depth: u32,
    ) -> Result<Literal, AmbientError> {
        Ok(match l {
            Literal::Int(v) => match amb {
                Some(p @ (Paradigm::Binary | Paradigm::Ternary)) => {
                    self.note(site, p, format!("bare decimal `{v}` adopts `{p}`"));
                    Literal::AmbientInt(p, *v)
                }
                Some(p) => {
                    return Err(AmbientError::BareDecimalNoEncoding {
                        site: site.to_owned(),
                        paradigm: p,
                    })
                }
                None => Literal::Int(*v),
            },
            Literal::List(elems) => {
                let mut out = Vec::with_capacity(elems.len());
                for e in elems {
                    out.push(self.expr(amb, site, e, depth)?);
                }
                Literal::List(out)
            }
            // Tagged literals already name their paradigm; AmbientInt is only produced here.
            // RFC-0032 D4: a `0x…` byte-string literal is a tagged repr literal (no ambient).
            // M-910/M-911: a `"…"` string literal joins the same group (no ambient either).
            // M-897: a float literal already names its representation (ADR-040) — an integer
            // ambient never touches it (the group below is unchanged-under-ambients).
            Literal::Bin(_)
            | Literal::Trit(_)
            | Literal::Bytes(_)
            | Literal::Str(_)
            | Literal::Float(_)
            | Literal::AmbientInt(_, _) => l.clone(),
        })
    }

    /// Resolve a pattern: a bare-decimal literal pattern adopts the ambient paradigm (its width is
    /// the scrutinee's, checked by `normalize_pattern`). Constructor/binder/wildcard patterns and
    /// tagged-literal patterns are unaffected.
    ///
    /// # Errors
    /// [`AmbientError::DepthExceeded`] once this traversal's own recursion exceeds
    /// [`MAX_AMBIENT_DEPTH`] (M-674) — a clean, explicit refusal rather than a host-stack overflow
    /// on a pathologically-nested `p`.
    fn pattern(
        &mut self,
        amb: Option<Paradigm>,
        site: &str,
        p: &Pattern,
        depth: u32,
    ) -> Result<Pattern, AmbientError> {
        let depth = Self::enter_depth(site, depth)?;
        Ok(match p {
            Pattern::Lit(l) => Pattern::Lit(self.literal(amb, site, l, depth)?),
            Pattern::Ctor(name, subs) => {
                let mut out = Vec::with_capacity(subs.len());
                for s in subs {
                    out.push(self.pattern(amb, site, s, depth)?);
                }
                Pattern::Ctor(name.clone(), out)
            }
            // M-826: propagate the ambient paradigm into sub-patterns of a tuple pattern.
            Pattern::Tuple(subs) => {
                let mut out = Vec::with_capacity(subs.len());
                for s in subs {
                    out.push(self.pattern(amb, site, s, depth)?);
                }
                Pattern::Tuple(out)
            }
            Pattern::Wildcard | Pattern::Ident(_) => p.clone(),
            // `Pattern::Or` alternatives are surface syntax (RFC-0020 §9 / R20-Q3): the ambient
            // pass runs BEFORE the checker (which desugars `Or` into flat arms), so we walk each
            // alternative and apply ambient literal resolution to it — the same transformation
            // applied to any other pattern. The `Or` node itself carries no ambient information.
            Pattern::Or(alts) => {
                let mut resolved = Vec::with_capacity(alts.len());
                for alt in alts {
                    resolved.push(self.pattern(amb, site, alt, depth)?);
                }
                Pattern::Or(resolved)
            }
        })
    }
}

/// Fill a paradigm-less `{params}` with paradigm `p` into a concrete [`BaseType`], or refuse with a
/// [`AmbientError::ParadigmShapeMismatch`] (§4.3) — the shape is never coerced.
fn fill_repr(site: &str, p: Paradigm, params: &AmbientParams) -> Result<BaseType, AmbientError> {
    let mismatch = |detail: &str| AmbientError::ParadigmShapeMismatch {
        site: site.to_owned(),
        paradigm: p,
        detail: detail.to_owned(),
    };
    Ok(match (p, params) {
        (Paradigm::Binary, AmbientParams::Size(n)) => BaseType::Binary(WidthRef::Lit(*n)),
        (Paradigm::Ternary, AmbientParams::Size(n)) => BaseType::Ternary(WidthRef::Lit(*n)),
        (Paradigm::Dense, AmbientParams::Dense(d, s)) => BaseType::Dense(*d, *s),
        (
            Paradigm::Vsa,
            AmbientParams::Vsa {
                model,
                dim,
                sparsity,
            },
        ) => BaseType::Vsa {
            model: model.clone(),
            dim: *dim,
            sparsity: sparsity.clone(),
        },
        (Paradigm::Binary | Paradigm::Ternary, _) => {
            return Err(mismatch("this paradigm takes a single size `{N}`"))
        }
        (Paradigm::Dense, _) => return Err(mismatch("`Dense` takes `{dim, scalar}`")),
        (Paradigm::Vsa, _) => return Err(mismatch("`VSA` takes `{model, dim, sparsity}`")),
    })
}

// --- surface pretty-printer (the "expand ambient" projection) -------------------------------------

fn path_str(p: &crate::ast::Path) -> String {
    p.0.join(".")
}

/// The `pub ` prefix for an exported top-level item, or `""` (M-662). Re-emitting it keeps the
/// longhand twin faithful — dropping `pub` would silently change a name's cross-nodule visibility.
fn pub_str(vis: Vis) -> &'static str {
    if vis.is_pub() {
        "pub "
    } else {
        ""
    }
}

/// Render an `object` composition declaration back to canonical surface text (DN-53, M-811).
/// This is the "expand ambient" projection of an `object` — it prints the surface form, not the
/// desugared lowering. The desugared form (type + impls + fns) is visible after Phase 0 expansion
/// in the checker; the surface form here supports the LSP round-trip and `expand_to_source`.
fn print_object_decl(od: &ObjectDecl) -> String {
    let params = if od.params.is_empty() {
        String::new()
    } else {
        format!("[{}]", od.params.join(", "))
    };
    // Ctor: `CtorName(T1, T2, …)` or `CtorName` for nullary.
    let ctor_str = if od.ctor.fields.is_empty() {
        od.ctor.name.clone()
    } else {
        let fs: Vec<String> = od.ctor.fields.iter().map(print_type_ref).collect();
        format!("{}({})", od.ctor.name, fs.join(", "))
    };
    let mut s = format!(
        "{}object {}{} {{\n  {};\n",
        pub_str(od.vis),
        od.name,
        params,
        ctor_str
    );
    for vd in &od.via_decls {
        let args = if vd.trait_args.is_empty() {
            String::new()
        } else {
            let a: Vec<String> = vd.trait_args.iter().map(print_type_ref).collect();
            format!("[{}]", a.join(", "))
        };
        s.push_str(&format!(
            "  via {} : {}{};\n",
            vd.field_idx, vd.trait_name, args
        ));
    }
    for id in &od.impls {
        // Re-use print_impl_decl but indent each line by 2 spaces; DN-57 §3 (M-818): the object
        // `impl` member is a component — terminated by `;` after its closing `}`.
        for line in terminate_item(&print_impl_decl(id)).lines() {
            s.push_str(&format!("  {line}\n"));
        }
    }
    for fd in &od.fns {
        // DN-57 §3 (M-818): each object `fn` member is a component — terminated by `;`.
        for line in terminate_item(&print_fn_decl(fd)).lines() {
            s.push_str(&format!("  {line}\n"));
        }
    }
    s.push_str("}\n");
    s
}

/// Render a `use` import (specific `use a.b.Item` or glob `use a.b.*`; M-662). Re-emitting the `.*`
/// keeps the glob distinct from a specific import on round-trip.
fn print_use(u: &UsePath) -> String {
    if u.glob {
        format!("use {}.*\n", path_str(&u.path))
    } else {
        format!("use {}\n", path_str(&u.path))
    }
}

fn print_type_decl(td: &TypeDecl) -> String {
    let params = if td.params.is_empty() {
        String::new()
    } else {
        // RFC-0037 D2: data type-parameters render in `[…]` (parsed by `parse_type_params_opt`).
        format!("[{}]", td.params.join(", "))
    };
    let ctors: Vec<String> = td
        .ctors
        .iter()
        .map(|c| {
            if c.fields.is_empty() {
                c.name.clone()
            } else {
                let fs: Vec<String> = c.fields.iter().map(print_type_ref).collect();
                format!("{}({})", c.name, fs.join(", "))
            }
        })
        .collect();
    format!(
        "{}type {}{} = {}\n",
        pub_str(td.vis),
        td.name,
        params,
        ctors.join(" | ")
    )
}

fn print_trait_decl(td: &TraitDecl) -> String {
    let params = if td.params.is_empty() {
        String::new()
    } else {
        // RFC-0037 D2: trait type-parameters render in `[…]` (parsed by `parse_type_params_opt`).
        format!("[{}]", td.params.join(", "))
    };
    let mut s = format!("{}trait {}{} {{\n", pub_str(td.vis), td.name, params);
    for sig in &td.sigs {
        // DN-57 §3 (M-818): each trait signature is a component — terminated by `;`.
        s.push_str(&format!("  fn {};\n", print_sig_tail(sig)));
    }
    s.push_str("}\n");
    s
}

fn print_fn_decl(fd: &FnDecl) -> String {
    format!(
        "{}{}fn {} =\n  {}\n",
        pub_str(fd.vis),
        if fd.thaw { "thaw " } else { "" },
        print_sig_tail(&fd.sig),
        print_expr(&fd.body)
    )
}

fn print_sig_tail(sig: &FnSig) -> String {
    // RFC-0037 D2: type parameters render in `[T]` (may carry bounds — the dictionary site),
    // width/const parameters render in `{N}` (bare names). The two lists print in that order,
    // matching `parse_sig_tail` (which reads `[…]` then `{…}`), so expand→reparse round-trips.
    let type_ps: Vec<String> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Type)
        .map(print_type_param)
        .collect();
    let width_ps: Vec<String> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Width)
        .map(print_type_param)
        .collect();
    let tp = if type_ps.is_empty() {
        String::new()
    } else {
        format!("[{}]", type_ps.join(", "))
    };
    let wp = if width_ps.is_empty() {
        String::new()
    } else {
        format!("{{{}}}", width_ps.join(", "))
    };
    let ps: Vec<String> = sig
        .value_params
        .iter()
        .map(|p| format!("{}: {}", p.name, print_type_ref(&p.ty)))
        .collect();
    format!(
        "{}{}{}({}) => {}",
        sig.name,
        tp,
        wp,
        ps.join(", "),
        print_type_ref(&sig.ret)
    )
}

fn print_type_ref(t: &TypeRef) -> String {
    let base = format!("{}", DisplayBase(&t.base));
    match t.guarantee {
        Some(g) => format!("{base} @ {g:?}"),
        None => base,
    }
}

/// Canonical surface form of a (possibly bounded) function type-parameter (`T` or `T: Cmp + Ord<T>`;
/// RFC-0019 §4.1).
fn print_type_param(p: &TypeParam) -> String {
    if p.bounds.is_empty() {
        p.name.clone()
    } else {
        let bs: Vec<String> = p.bounds.iter().map(print_trait_ref).collect();
        format!("{}: {}", p.name, bs.join(" + "))
    }
}

/// Canonical surface form of a trait reference in a bound (`Cmp` or `Cmp[Binary{8}]`; RFC-0037 D2).
fn print_trait_ref(tr: &TraitRef) -> String {
    if tr.args.is_empty() {
        tr.name.clone()
    } else {
        let args: Vec<String> = tr.args.iter().map(print_type_ref).collect();
        format!("{}[{}]", tr.name, args.join(", "))
    }
}

/// Canonical surface form of an `impl Trait[args] for T { fn … }` (RFC-0019 §4.1 / RFC-0037 D2).
fn print_impl_decl(id: &ImplDecl) -> String {
    let args = if id.trait_args.is_empty() {
        String::new()
    } else {
        let a: Vec<String> = id.trait_args.iter().map(print_type_ref).collect();
        format!("[{}]", a.join(", "))
    };
    let mut s = format!(
        "impl {}{} for {} {{\n",
        id.trait_name,
        args,
        print_type_ref(&id.for_ty)
    );
    for m in &id.methods {
        // DN-57 §3 (M-818): each impl method is a component — terminated by `;`.
        s.push_str(&format!("  {}", terminate_item(&print_fn_decl(m))));
    }
    s.push_str("}\n");
    s
}

/// Canonical surface form of an inherent method block `impl T { fn … }` (DN-03 §1 / M-664).
fn print_inherent_impl_decl(id: &InherentImplDecl) -> String {
    let mut s = format!("impl {} {{\n", print_type_ref(&id.for_ty));
    for m in &id.methods {
        // DN-57 §3 (M-818): each inherent method is a component — terminated by `;`.
        s.push_str(&format!("  {}", terminate_item(&print_fn_decl(m))));
    }
    s.push_str("}\n");
    s
}

/// A [`Display`] for [`BaseType`] in canonical surface form (shared by the printer and the
/// provenance notes).
struct DisplayBase<'a>(&'a BaseType);

impl core::fmt::Display for DisplayBase<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            BaseType::Binary(WidthRef::Lit(n)) => write!(f, "Binary{{{n}}}"),
            BaseType::Binary(WidthRef::Name(v)) => write!(f, "Binary{{{v}}}"),
            BaseType::Ternary(WidthRef::Lit(m)) => write!(f, "Ternary{{{m}}}"),
            BaseType::Ternary(WidthRef::Name(v)) => write!(f, "Ternary{{{v}}}"),
            BaseType::Dense(d, s) => write!(f, "Dense{{{d}, {}}}", scalar_str(*s)),
            BaseType::Vsa {
                model,
                dim,
                sparsity,
            } => write!(f, "VSA{{{model}, {dim}, {}}}", sparsity_str(sparsity)),
            BaseType::Substrate(t) => write!(f, "Substrate{{{t}}}"),
            // RFC-0032 D3/D4: `Seq{T, N}` / nullary `Bytes`.
            BaseType::Seq { elem, len } => {
                write!(f, "Seq{{{}, {len}}}", print_type_ref(elem))
            }
            BaseType::Bytes => write!(f, "Bytes"),
            // ADR-040 (M-897): nullary scalar-float repr keyword.
            BaseType::Float => write!(f, "Float"),
            BaseType::Named(n, args) if args.is_empty() => write!(f, "{n}"),
            BaseType::Named(n, args) => {
                // RFC-0037 D2: type arguments render in `[…]` (parsed by `parse_type_args_opt`).
                let a: Vec<String> = args.iter().map(print_type_ref).collect();
                write!(f, "{n}[{}]", a.join(", "))
            }
            BaseType::Ambient(params) => write!(f, "{{{}}}", ambient_params_str(params)),
            // Function type: `A => B` in canonical surface form (RFC-0024 §3, M-685; RFC-0037 D4
            // retired the `->` glyph in favour of `=>`).
            BaseType::Fn(arg, ret) => {
                // Parenthesize a function-typed LHS so `(A => B) => C` round-trips unambiguously,
                // not as `A => B => C` (Copilot #397).
                let lhs = print_type_ref(arg);
                if matches!(arg.base, BaseType::Fn(_, _)) {
                    write!(f, "({lhs}) => {}", print_type_ref(ret))
                } else {
                    write!(f, "{lhs} => {}", print_type_ref(ret))
                }
            }
            // M-826: tuple type `(T, U, …)` — rendered in canonical surface form.
            BaseType::Tuple(elems) => {
                let parts: Vec<String> = elems.iter().map(print_type_ref).collect();
                write!(f, "({})", parts.join(", "))
            }
        }
    }
}

fn scalar_str(s: Scalar) -> &'static str {
    match s {
        Scalar::F16 => "F16",
        Scalar::Bf16 => "BF16",
        Scalar::F32 => "F32",
        Scalar::F64 => "F64",
    }
}

fn sparsity_str(s: &Sparsity) -> String {
    match s {
        Sparsity::Dense => "Dense".to_owned(),
        Sparsity::Sparse(k) => format!("Sparse{{{k}}}"),
    }
}

fn ambient_params_str(p: &AmbientParams) -> String {
    match p {
        AmbientParams::Size(n) => format!("{n}"),
        AmbientParams::Dense(d, s) => format!("{d}, {}", scalar_str(*s)),
        AmbientParams::Vsa {
            model,
            dim,
            sparsity,
        } => format!("{model}, {dim}, {}", sparsity_str(sparsity)),
    }
}

fn print_expr(e: &Expr) -> String {
    match e {
        Expr::Lit(l) => print_literal(l),
        Expr::Path(p) => path_str(p),
        Expr::Let {
            name,
            ty,
            bound,
            body,
        } => {
            let ann = ty
                .as_ref()
                .map(|t| format!(": {}", print_type_ref(t)))
                .unwrap_or_default();
            format!(
                "let {name}{ann} = {} in {}",
                print_expr(bound),
                print_expr(body)
            )
        }
        Expr::If { cond, conseq, alt } => format!(
            "if {} then {} else {}",
            print_expr(cond),
            print_expr(conseq),
            print_expr(alt)
        ),
        Expr::Match { scrutinee, arms } => {
            let arms: Vec<String> = arms
                .iter()
                .map(|a| format!("{} => {}", print_pattern(&a.pattern), print_expr(&a.body)))
                .collect();
            format!("match {} {{ {} }}", print_expr(scrutinee), arms.join(", "))
        }
        Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        } => format!(
            "for {x} in {}, {acc} = {} => {}",
            print_expr(xs),
            print_expr(init),
            print_expr(body)
        ),
        Expr::Swap {
            value,
            target,
            policy,
        } => format!(
            "swap({}, to: {}, policy: {})",
            print_expr(value),
            print_type_ref(target),
            path_str(policy)
        ),
        Expr::WithParadigm { paradigm, body } => {
            format!("with paradigm {paradigm} {{ {} }}", print_expr(body))
        }
        Expr::Wild(b) => format!("wild {{ {} }}", print_expr(b)),
        Expr::Spore(b) => format!("spore({})", print_expr(b)),
        Expr::Consume(b) => format!("consume {}", print_expr(b)),
        Expr::Lambda { params, body } => format!(
            "lambda({}) => {}",
            params
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
                .join(", "),
            print_expr(body)
        ),
        Expr::Colony(hyphae) => {
            let hs: Vec<String> = hyphae
                .iter()
                .map(|h| format!("hypha {}", print_expr(&h.body)))
                .collect();
            format!("colony {{ {} }}", hs.join(", "))
        }
        Expr::App { head, args } => {
            let args: Vec<String> = args.iter().map(print_expr).collect();
            format!("{}({})", print_expr(head), args.join(", "))
        }
        Expr::Ascribe(inner, t) => format!("{} : {}", print_expr(inner), print_type_ref(t)),
        // DN-58 §A/§B (M-667): `fuse` and `reclaim` print forms.
        Expr::Fuse { left, right } => {
            format!("fuse({}, {})", print_expr(left), print_expr(right))
        }
        Expr::Reclaim { policy, body } => {
            format!("reclaim({}) {{ {} }}", print_expr(policy), print_expr(body))
        }
        // M-826: tuple literal `(a, b, …)`.
        Expr::TupleLit(elems) => {
            let parts: Vec<String> = elems.iter().map(print_expr).collect();
            format!("({})", parts.join(", "))
        }
    }
}

fn print_pattern(p: &Pattern) -> String {
    match p {
        Pattern::Wildcard => "_".to_owned(),
        Pattern::Lit(l) => print_literal(l),
        Pattern::Ctor(n, subs) => {
            let s: Vec<String> = subs.iter().map(print_pattern).collect();
            format!("{n}({})", s.join(", "))
        }
        Pattern::Ident(n) => n.clone(),
        // M-826: tuple pattern `(x, y, …)`.
        Pattern::Tuple(subs) => {
            let s: Vec<String> = subs.iter().map(print_pattern).collect();
            format!("({})", s.join(", "))
        }
        // `Pattern::Or` is surface syntax (RFC-0020 §9 / R20-Q3). The ambient printer runs before
        // the checker desugars `Or`, so round-trip it as `p1 | p2 | …` (the surface form).
        Pattern::Or(alts) => alts
            .iter()
            .map(print_pattern)
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

fn print_literal(l: &Literal) -> String {
    match l {
        Literal::Bin(s) => format!("0b{s}"),
        // RFC-0037 D4: balanced-ternary literals use the `0t…` prefix (the angle form `<…>` was
        // retired with D4 — `<` is operator-only). The printer must emit the active form so the
        // round-trip `parse → expand_to_source → parse` is stable (M-818 exposed the stale `<…>`).
        Literal::Trit(s) => format!("0t{s}"),
        // RFC-0032 D4: a `0x…` byte-string literal round-trips to its source form.
        Literal::Bytes(s) => format!("0x{s}"),
        // M-910/M-911: a `"…"` string literal round-trips by RE-ESCAPING its decoded content (the
        // inverse of `Lexer::lex_string`) — every char the lexer's escape set can decode is
        // re-escaped here, so `parse → expand_to_source → parse` is stable (M-818 discipline).
        Literal::Str(s) => format!("\"{}\"", escape_string_literal(s)),
        Literal::Int(i) => format!("{i}"),
        // M-897: the float literal's surface form IS its stored source text (verbatim round-trip).
        Literal::Float(s) => s.clone(),
        // A still-unresolved ambient decimal: show the decimal + its resolved paradigm (the width is
        // the checker's to fill — this only appears when expanding a type-form-only nodule).
        Literal::AmbientInt(p, i) => format!("{i} /* {p} (width from context) */"),
        Literal::List(es) => {
            let s: Vec<String> = es.iter().map(print_expr).collect();
            format!("[{}]", s.join(", "))
        }
    }
}

/// Re-escape a decoded string literal's content back to its `"…"` surface form (M-910/M-911) — the
/// exact inverse of `Lexer::lex_string`'s decode step, over the same minimal escape set (`\n \t \\
/// \" \0 \r`). Every char the lexer can decode is re-escaped here, so no other char needs escaping.
fn escape_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            other => out.push(other),
        }
    }
    out
}

/// Round-trip a `lower Name[params] = <rhs>` declaration (DN-54 / M-812).
/// The RHS is printed via [`print_expr`]; no ambient state to fill.
fn print_lower_decl(ld: &LowerDecl) -> String {
    let params = if ld.params.is_empty() {
        String::new()
    } else {
        format!("[{}]", ld.params.join(", "))
    };
    // The RHS is either expression-shaped (v0) or item-shaped (`impl … for …`; DN-54 §10 Model A,
    // M-973). Render each in its surface form so the ambient round-trip stays faithful (RFC-0012).
    let rhs = match &ld.rhs {
        LowerRhs::Expr(e) => print_expr(e),
        // `print_impl_decl` emits a trailing newline; trim it so the `= …\n` wrapper below is the
        // single line terminator (the item RHS renders as a multi-line block after `=`).
        LowerRhs::Impl(id) => print_impl_decl(id).trim_end().to_string(),
    };
    format!("lower {}{} = {}\n", ld.name, params, rhs)
}

/// Round-trip a `derive Name for T` declaration (DN-54 / M-812 / DN-38 §8.1).
fn print_derive_decl(dd: &DeriveDecl) -> String {
    format!("derive {} for {}\n", dd.name, print_type_ref(&dd.for_ty))
}
