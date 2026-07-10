//! The L1 recursive-descent parser (RFC-0006; faithful to `docs/spec/grammar/mycelium.ebnf`).
//! Hand-written, no dependencies. Every failure is an explicit [`ParseError`] with a position
//! (never a panic, never a silent accept — S5/G2). v0 covers the L1-facing core.

use crate::ast::{
    AmbientParams, Arm, BaseType, Ctor, DeriveDecl, ExecutionMode, Expr, FnDecl, FnSig, Hypha,
    ImplDecl, InherentImplDecl, Item, Literal, LowerDecl, LowerRhs, Nodule, ObjectDecl, Paradigm,
    Param, ParamKind, Path, Pattern, Phylum, Scalar, Sparsity, Strength, TraitDecl, TraitRef,
    TypeDecl, TypeParam, TypeRef, UsePath, ViaDecl, Vis, WidthRef,
};
use crate::error::ParseError;
use crate::lexer::lex;
use crate::token::{Pos, ScalarTok, Spanned, StrengthTok, Tok};

/// Maximum nesting depth of the expression grammar. Crafted deeply-nested input would otherwise
/// drive the recursive-descent parser (and, over the resulting AST, the typechecker / totality
/// checker / elaborator) into unbounded host-stack recursion and abort the process — `myc-check` is
/// the M-002 oracle and must return an explicit error, never crash (A4-02/B2-01). The limit is well
/// above any realistic L1 program and far below the host stack budget. Bounding the parser bounds
/// the AST depth, so the downstream passes are protected transitively.
///
/// **RFC-0041 §4.2/§7 (W1):** raised `256 → 4096` to unify the parser's ceiling with the shared
/// recursion budget's `DEFAULT_DEPTH_LIMIT` ([`mycelium_workstack::RecursionBudget`]) and the
/// checker's `MAX_CHECK_DEPTH`, so a program the checker would accept is no longer refused *earlier*
/// by a tighter parser cap. The parser runs under the 256 MiB [`mycelium_stack::with_deep_stack`]
/// worker (below), which physically supports far more than 4096 parser frames — so the guard still
/// fires cleanly (an explicit `ParseError`) well before any host-stack overflow (verified by the
/// `deeply_nested_*` regressions in `tests/check.rs`, which exceed 4096). Eval's 64 stays (W5).
const MAX_EXPR_DEPTH: u32 = 4096;

/// Parse a complete **single-`nodule`** program from source — the v0 entry point, unchanged by the
/// phylum work (M-662). A bare `nodule <path> <item>*` parses to a [`Nodule`]; trailing content (a
/// second `nodule`, a `phylum` header) is an explicit error here. Multi-nodule / phylum-headed source
/// uses [`parse_phylum`]; a [`Nodule`] *is* a phylum-of-one ([`Phylum::of_one`]).
pub fn parse(src: &str) -> Result<Nodule, ParseError> {
    let toks = lex(src)?;
    // Run the recursive-descent parser on the managed deep stack (as `eval`/`ambient` do) so the
    // explicit `MAX_EXPR_DEPTH` budget — not the host stack — is the binding limit on nesting depth,
    // independent of per-toolchain frame sizes (A4-02 / DN-40). Regression witness: on MSRV 1.96.1
    // (ADR-041) the larger parser frames overflowed the 2 MB test stack at the 256-deep guard
    // boundary on the `type_args` path, turning an explicit refusal back into a SIGABRT (G2).
    mycelium_stack::with_deep_stack(move || {
        let mut p = Parser {
            toks,
            i: 0,
            depth: 0,
        };
        let nodule = p.parse_nodule()?;
        p.expect(&Tok::Eof, "end of input")?;
        Ok(nodule)
    })
}

/// Parse a complete **phylum** program (M-662; RFC-0006 §4.3): an optional `phylum <path>` header
/// followed by **one-or-more** `nodule` blocks, into a [`Phylum`]. A header-less single `nodule`
/// parses to a phylum-of-one (`path: None`) — so `parse_phylum` is a strict superset of [`parse`]:
/// every program [`parse`] accepts, `parse_phylum` accepts identically (as a phylum-of-one), and it
/// additionally accepts a `phylum` header and multiple nodules. A `phylum` header with **zero**
/// nodules is an explicit error (a phylum groups nodules — there must be at least one).
///
/// # Errors
/// Returns a [`ParseError`] for any malformed header, item, or a `phylum` header followed by no
/// `nodule` (never a panic, never a silent accept — S5/G2).
pub fn parse_phylum(src: &str) -> Result<Phylum, ParseError> {
    let toks = lex(src)?;
    // Deep-stack the recursive descent — see [`parse`] (A4-02 / DN-40 / ADR-041 frame-size note).
    mycelium_stack::with_deep_stack(move || {
        let mut p = Parser {
            toks,
            i: 0,
            depth: 0,
        };
        let phylum = p.parse_phylum()?;
        p.expect(&Tok::Eof, "end of input")?;
        Ok(phylum)
    })
}

struct Parser {
    toks: Vec<Spanned>,
    i: usize,
    /// Current expression-nesting depth, bounded by [`MAX_EXPR_DEPTH`] (A4-02).
    depth: u32,
}

/// Walk a [`TypeRef`] collecting which param names appear in width-slot positions
/// (`Binary{N}` / `Ternary{N}`) vs type-slot positions (`Named(N, [])`).
/// Used by [`classify_params`] to disambiguate width vs type parameters (DN-42 / M-753 v1).
fn collect_name_uses(
    tr: &TypeRef,
    params: &std::collections::BTreeSet<String>,
    width_used: &mut std::collections::BTreeSet<String>,
    type_used: &mut std::collections::BTreeSet<String>,
) {
    collect_base_name_uses(&tr.base, params, width_used, type_used);
}

fn collect_base_name_uses(
    bt: &BaseType,
    params: &std::collections::BTreeSet<String>,
    width_used: &mut std::collections::BTreeSet<String>,
    type_used: &mut std::collections::BTreeSet<String>,
) {
    match bt {
        BaseType::Binary(WidthRef::Name(n)) | BaseType::Ternary(WidthRef::Name(n)) => {
            if params.contains(n) {
                width_used.insert(n.clone());
            }
        }
        BaseType::Named(n, args) => {
            if params.contains(n) {
                type_used.insert(n.clone());
            }
            for a in args {
                collect_name_uses(a, params, width_used, type_used);
            }
        }
        BaseType::Seq { elem, .. } => collect_name_uses(elem, params, width_used, type_used),
        BaseType::Fn(a, b) => {
            collect_name_uses(a, params, width_used, type_used);
            collect_name_uses(b, params, width_used, type_used);
        }
        _ => {}
    }
}

/// Post-parse classification of `<…>` parameters as [`ParamKind::Type`] or [`ParamKind::Width`]
/// by examining how each name is used in `value_params` and `ret` (DN-42 / M-753 v1).
///
/// **Refusals (never-silent — G2 / VR-5):**
/// - A name in both a width slot (`Binary{N}`) and a type slot (`Named(N, [])`) → explicit error.
/// - A bound on a width-classified param → explicit error (DN-42 §7: deferred).
fn classify_params(
    params: Vec<TypeParam>,
    value_params: &[crate::ast::Param],
    ret: &TypeRef,
) -> Result<Vec<TypeParam>, crate::error::ParseError> {
    use std::collections::BTreeSet;
    let param_names: BTreeSet<String> = params.iter().map(|p| p.name.clone()).collect();
    let mut width_used: BTreeSet<String> = BTreeSet::new();
    let mut type_used: BTreeSet<String> = BTreeSet::new();
    for vp in value_params {
        collect_name_uses(&vp.ty, &param_names, &mut width_used, &mut type_used);
    }
    collect_name_uses(ret, &param_names, &mut width_used, &mut type_used);

    let mut result = Vec::with_capacity(params.len());
    for p in params {
        let in_width = width_used.contains(&p.name);
        let in_type = type_used.contains(&p.name);
        let kind = match (in_width, in_type) {
            (true, true) => {
                return Err(crate::error::ParseError::new(
                    crate::token::Pos { line: 0, col: 0 },
                    format!(
                        "parameter `{}` appears in both a width slot (`Binary{{N}}`/`Ternary{{N}}`)                          and a type slot — ambiguous: is it a width param or a type param? Use                          distinct names (DN-42 / M-753; never a silent guess)",
                        p.name
                    ),
                ))
            }
            (true, false) => {
                if !p.bounds.is_empty() {
                    return Err(crate::error::ParseError::new(
                        crate::token::Pos { line: 0, col: 0 },
                        format!(
                            "width parameter `{}` cannot carry trait bounds in v1 — bounds on width                              params are deferred (DN-42 §7; never a silent ignore)",
                            p.name
                        ),
                    ));
                }
                ParamKind::Width
            }
            _ => ParamKind::Type,
        };
        result.push(TypeParam {
            name: p.name,
            kind,
            bounds: p.bounds,
        });
    }
    Ok(result)
}

impl Parser {
    fn cur(&self) -> &Tok {
        &self.toks[self.i].tok
    }

    fn pos(&self) -> Pos {
        self.toks[self.i].pos
    }

    fn at(&self, t: &Tok) -> bool {
        self.cur() == t
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i].tok.clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }

    fn err<T>(&self, what: &str) -> Result<T, ParseError> {
        Err(ParseError::new(
            self.pos(),
            format!("expected {what}, found {:?}", self.cur()),
        ))
    }

    fn expect(&mut self, t: &Tok, what: &str) -> Result<(), ParseError> {
        if self.at(t) {
            self.bump();
            Ok(())
        } else {
            self.err(what)
        }
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Consume the **mandatory** `;` component terminator (DN-57 §3, enacted M-818). Every
    /// component — a top-level item, a trait signature, an `impl`/object method, an object `via`
    /// clause, **and the nodule header itself** — ends with exactly one `;`, *uniformly and
    /// regardless of how its body ends*: a `}`-terminated component (`trait { … }`, `impl { … }`,
    /// `object { … }`) still requires the trailing `;`. This makes the end-of-component a single
    /// terminal *token* (never the *absence* of more tokens / a newline), so whitespace-free source
    /// (`nodule d; fn a() => … = …;`) is legal and a streaming parser can emit a completed
    /// component the instant it sees `;` — the full streaming guarantee.
    ///
    /// Never-silent (G2): a missing terminator is an explicit [`ParseError`] naming the component
    /// and where the `;` belongs — never a silently-accepted run-on. `what` is the component name
    /// (e.g. `"function definition"`).
    fn expect_terminator(&mut self, what: &str) -> Result<(), ParseError> {
        if self.eat(&Tok::Semi) {
            Ok(())
        } else {
            Err(ParseError::new(
                self.pos(),
                format!(
                    "expected `;` to terminate this {what} (DN-57: `;` is the mandatory \
                     component terminator — every component ends with `;`, including a \
                     `}}`-closed block), found {:?}",
                    self.cur()
                ),
            ))
        }
    }

    // ---- recursion budget (A4-02 / DN-40 A1·A2) ----
    //
    // Every recursive-descent entry point that can nest on the *host* stack — the expression
    // grammar (`parse_expr`/`parse_unary`), the **type** subgrammar (`parse_type_ref` →
    // `parse_base_type`/`parse_type_args_opt`), and the **pattern** grammar (`parse_pattern`,
    // which recurses over nested constructor sub-patterns) — charges this single shared budget.
    // Crafted input (`A -> A -> …`, nested `<…>`, `C(C(C(…)))`) would otherwise drive unbounded
    // host recursion and abort the process (SIGABRT) — `myc-check` must return an explicit error,
    // never crash (G2; the module's never-silent contract). A single pair keeps the discipline
    // DRY so no entry point silently drifts off the budget (DN-40 found exactly such a drift).

    /// Charge one level of nesting against the shared [`MAX_EXPR_DEPTH`] budget. Returns an
    /// explicit [`ParseError`] (not a panic) once the limit is exceeded, leaving the budget
    /// *unchanged* on the error path so the failed level is not counted. On success the caller
    /// **must** pair this with exactly one [`leave_depth`](Self::leave_depth) on every exit path.
    fn enter_depth(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            self.depth -= 1;
            return Err(ParseError::new(
                self.pos(),
                format!(
                    "expression nests deeper than the limit of {MAX_EXPR_DEPTH} — refusing to recurse"
                ),
            ));
        }
        Ok(())
    }

    /// Release one level previously charged by [`enter_depth`](Self::enter_depth). Always paired
    /// with a successful `enter_depth`, on every (Ok *and* Err) exit path of the guarded region,
    /// so no increment is ever leaked.
    fn leave_depth(&mut self) {
        self.depth -= 1;
    }

    // ---- separated lists (DRY, M-640) ----
    //
    // The grammar repeats two comma-separated shapes; these helpers are the single code path for
    // each, so every call site consumes byte-identical tokens and raises the identical `ParseError`
    // (the close-delimiter `expect` with its bespoke message stays at the call site). No grammar
    // change — a pure factoring of the hand-rolled loops.

    /// `one (`,` one)*` — a **non-empty** comma list, parsed *between* already-recognized delimiters
    /// (the caller consumed the opener and will `expect` the closer). Parses the first element
    /// unconditionally, then one more after each comma, stopping at the first non-comma. With
    /// `trailing_end = Some(t)`, a comma immediately followed by `t` ends the list (consumed) — the
    /// trailing-comma tolerance of `match` arms; with `None`, no trailing comma is accepted. Mirrors
    /// the bare `push(one); while eat(Comma) { … push(one) }` loop exactly. Used for constructor
    /// fields, type params, type args, sub-patterns (no trailing) and match arms (trailing).
    fn comma_separated<T>(
        &mut self,
        trailing_end: Option<&Tok>,
        mut parse_one: impl FnMut(&mut Self) -> Result<T, ParseError>,
    ) -> Result<Vec<T>, ParseError> {
        let mut items = vec![parse_one(self)?];
        while self.eat(&Tok::Comma) {
            if let Some(end) = trailing_end {
                if self.at(end) {
                    break;
                }
            }
            items.push(parse_one(self)?);
        }
        Ok(items)
    }

    /// `[ one (`,` one)* ]` — a possibly-**empty** comma list bounded by `end`, parsed *inside*
    /// already-opened delimiters (the caller consumed the opener and will `expect`/consume `end`).
    /// Equivalent to `if !at(end) { push(one); while eat(Comma) { push(one) } }`; no trailing comma.
    /// Used for value params, call args, and list-literal elements (each empty-permitting).
    fn comma_separated_until<T>(
        &mut self,
        end: &Tok,
        parse_one: impl FnMut(&mut Self) -> Result<T, ParseError>,
    ) -> Result<Vec<T>, ParseError> {
        if self.at(end) {
            return Ok(Vec::new());
        }
        self.comma_separated(None, parse_one)
    }

    /// `expect` a **leading keyword** whose diagnostic is just its own backtick-quoted spelling
    /// (`expected `let`, found …`), templating the message from one token→spelling table instead of
    /// re-spelling the keyword at each opener (M-640). Byte-identical to the prior
    /// `expect(&Tok::Let, "`let`")` form — same token consumed, same `ParseError` text. Only used at
    /// the self-naming keyword openers; sites whose message carries extra context (e.g.
    /// `"`fn` in the trait body"`) keep their bespoke [`expect`](Self::expect) call unchanged.
    fn expect_keyword(&mut self, kw: &Tok) -> Result<(), ParseError> {
        // Canonical surface spelling of each self-naming keyword opener. Total over exactly the
        // tokens the parser passes here; the `_` fallback keeps this **panic-free** (the parser
        // never panics — module invariant) by falling back to the `{:?}` form for any token outside
        // the set, so even a future miswiring is an explicit diagnostic, never a crash.
        let spelling = match kw {
            Tok::Type => "type",
            Tok::Trait => "trait",
            Tok::Fn => "fn",
            Tok::Let => "let",
            Tok::If => "if",
            Tok::Then => "then",
            Tok::Else => "else",
            Tok::Match => "match",
            Tok::For => "for",
            Tok::Swap => "swap",
            Tok::With => "with",
            Tok::Wild => "wild",
            Tok::Spore => "spore",
            Tok::Colony => "colony",
            // DN-53, M-811: `object` is now an active keyword; `via` is active inside object bodies.
            Tok::Object => "object",
            Tok::Via => "via",
            other => return self.expect(kw, &format!("{other:?}")),
        };
        self.expect(kw, &format!("`{spelling}`"))
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.cur() {
            Tok::Ident(s) => {
                let s = s.clone();
                self.bump();
                Ok(s)
            }
            _ => self.err("an identifier"),
        }
    }

    fn u32_lit(&mut self) -> Result<u32, ParseError> {
        match *self.cur() {
            Tok::Int(n) => {
                self.bump();
                u32::try_from(n)
                    .map_err(|_| ParseError::new(self.pos(), format!("{n} is out of u32 range")))
            }
            _ => self.err("a non-negative integer"),
        }
    }

    // ---- items ----

    /// `phylum_header? nodule+` — a whole phylum program (M-662; RFC-0006 §4.3). An optional
    /// `phylum <path>` header (the library-scale grouping; DN-06) precedes **one-or-more** `nodule`
    /// blocks. `phylum` is a *grouping*, not a container — no `phylum { … }` block; the nodules follow
    /// the header at top level. A header with no following `nodule` is an explicit error.
    fn parse_phylum(&mut self) -> Result<Phylum, ParseError> {
        // Optional `phylum <path>` header. `phylum` activates as a header keyword here (M-662); it was
        // reserved-not-active before. It carries a dotted path and opens no block.
        let path = if self.eat(&Tok::Phylum) {
            Some(self.parse_path()?)
        } else {
            None
        };
        let mut nodules = Vec::new();
        // One-or-more `nodule` blocks. The first must be present (a phylum, headed or not, groups at
        // least one nodule); each `parse_nodule` consumes its items up to the next `nodule`/EOF.
        if !self.at(&Tok::Nodule) {
            return Err(ParseError::new(
                self.pos(),
                if path.is_some() {
                    "a `phylum` header must be followed by at least one `nodule` block \
                     (a phylum groups nodules — RFC-0006 §4.3)"
                        .to_owned()
                } else {
                    "expected a `nodule` header to open the program".to_owned()
                },
            ));
        }
        while self.at(&Tok::Nodule) {
            nodules.push(self.parse_nodule()?);
        }
        Ok(Phylum { path, nodules })
    }

    /// `nodule <path> @std-sys? <item>*` — one nodule block (RFC-0006 §4.3). An optional
    /// **`@std-sys`** marker after the path (M-661; RFC-0016 §8-Q6) tags the nodule as the audited
    /// FFI-floor context: only a `@std-sys` nodule may contain a `wild` block (the checker enforces
    /// this — a `wild` elsewhere is a hard refusal, never silent — G2). The marker is lexed atomically
    /// as [`Tok::AtStdSys`] (so `@std-sys`'s `-` is not a lex error); absent ⇒ a normal nodule.
    ///
    /// Items run until the **next `nodule` header or EOF** (M-662): in a multi-nodule phylum the items
    /// of one nodule end where the next `nodule` begins. For a single-nodule program ([`parse`]) the
    /// loop simply runs to EOF, exactly as before — backward-compatible (a `nodule` *is* a
    /// phylum-of-one).
    fn parse_nodule(&mut self) -> Result<Nodule, ParseError> {
        self.expect(&Tok::Nodule, "a `nodule` header to open the program")?;
        let path = self.parse_path()?;
        // Optional `@std-sys` FFI-floor header marker (M-661). It is the audited-FFI *context* gate;
        // it carries no further syntax (no `: true`/`: false` — its mere presence is the attribute).
        let std_sys = self.eat(&Tok::AtStdSys);
        // DN-57 §3 (M-818): the nodule header is itself a component — it ends with a mandatory `;`.
        // This is what makes fully whitespace-free source (`nodule d; fn a() => … = …;`) legal: the
        // header/body boundary is the `;` token, not a newline.
        self.expect_terminator("nodule header")?;
        let mut items = Vec::new();
        // Stop at the next `nodule` (the start of a sibling nodule in a phylum) or EOF.
        while !self.at(&Tok::Eof) && !self.at(&Tok::Nodule) {
            items.push(self.parse_item()?);
            self.expect_terminator("item")?; // DN-57 §3 (M-818): mandatory component terminator
        }
        Ok(Nodule {
            path,
            std_sys,
            items,
        })
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        // A leading `pub` marks a top-level `fn`/`trait`/`type` (or `thaw fn`) as cross-nodule
        // **exported** (M-662). It is only valid before one of those declarations — a `pub` before
        // `use`/`default`/`impl` (or anything else) is an explicit refusal, never a silent accept
        // (G2). `impl`/`default`/`use` are not part of the `pub` namespace (a `use` imports, it does
        // not re-export). The `pub` is consumed here and threaded into the declaration's `vis`.
        if self.at(&Tok::Pub) {
            return self.parse_pub_item();
        }
        // DN-58 §C (M-667): `@tier(mode)` is a per-definition execution-mode hint that precedes a
        // `fn` declaration. It is parsed here (before the general `match`) so that `@tier` opens a
        // function item at both private and pub-prefixed positions. The `@` that opens `@tier` is
        // the same token as the guarantee-annotation `@` in type position, so we must look ahead:
        // `@` followed by `tier` → parse the tier attribute + fn decl; any other `@` is a syntax
        // error (no other item-level `@` attribute exists in v0 — teaching, never silent, G2).
        if self.at(&Tok::At) {
            return self
                .parse_tier_fn_decl(crate::ast::Vis::Private)
                .map(Item::Fn);
        }
        match self.cur() {
            Tok::Use => self.parse_use().map(Item::Use),
            Tok::Default => {
                self.bump();
                self.expect(&Tok::Paradigm, "`paradigm` after `default` (RFC-0012 §4.2)")?;
                Ok(Item::Default(self.parse_paradigm()?))
            }
            Tok::Type => self.parse_type_decl(Vis::Private).map(Item::Type),
            Tok::Trait => self.parse_trait_decl(Vis::Private).map(Item::Trait),
            // M-659 / RFC-0019 §4.1: `impl Trait<args> for T { fn … }` (the trait-instance
            // production). `impl` was reserved by M-658 (RFC-0007 §12.2); this is the production that
            // consumes it.
            // M-664: `impl Trait for T { … }` (trait instance) and `impl T { … }` (inherent method
            // block) share this dispatcher, which disambiguates on the `for`/`{` follower.
            Tok::Impl => self.parse_impl_item(),
            Tok::Fn | Tok::Thaw => self.parse_fn_decl(Vis::Private).map(Item::Fn),
            // DN-58 §C (M-667): `@tier(compiled)` / `@tier(interpreted)` before a `fn` declaration.
            // The `@` at item position is unambiguous — guarantee annotations only appear in type
            // contexts, not at item position. The `@tier` attribute is parsed here and threaded into
            // `FnDecl.tier` (never-silent on a bad mode name — G2).
            Tok::At => self.parse_tier_fn_decl(Vis::Private).map(Item::Fn),
            Tok::Matured => Err(ParseError::new(
                self.pos(),
                "maturation is declared per `nodule`/`phylum` in the header \
                     (`// @matured: true`) or per program in the manifest — RFC-0017 §4.1; \
                     to keep one definition interpreted inside a matured scope use `thaw fn`"
                    .to_owned(),
            )),
            // M-666 / RFC-0008 §4.7: `colony` and `hypha` are now **active**, but as *expressions*,
            // not top-level items — they live inside a `fn` body. Teaching diagnostics point there
            // (never a silent accept, G2).
            Tok::Colony => Err(ParseError::new(
                self.pos(),
                "`colony { … }` is an expression (a structured-concurrency scope; RFC-0008 §4.7), \
                 not a top-level item — write it inside a `fn` body"
                    .to_owned(),
            )),
            Tok::Hypha => Err(ParseError::new(
                self.pos(),
                "`hypha <expr>` spawns a concurrent task and is only valid inside a `colony { … }` \
                 block (RFC-0008 §4.7 RT7 — an orphan hypha is not expressible), not at item position"
                    .to_owned(),
            )),
            // DN-03 §4 / RFC-0008 §4.5: the remaining runtime-vocabulary reserved words. They lex as
            // keywords (never silent identifiers, G2) but no L1 construct consumes them yet — teaching
            // diagnostic, never a silent accept. (`hypha`/`colony` left the set with M-666;
            // `fuse`/`reclaim`/`tier` left the set with M-667 / DN-58 — they are now active.)
            t @ (Tok::Mesh
            | Tok::Graft
            | Tok::Cyst
            | Tok::Xloc
            | Tok::Forage
            | Tok::Backbone) => Err(ParseError::new(
                self.pos(),
                format!(
                    "`{word}` is reserved for the runtime model (RFC-0008), not yet active — \
                     it cannot open a program or be used as an identifier at this language version",
                    word = runtime_keyword_spelling(t)
                ),
            )),
            // DN-58 §C (M-667): `tier` as a bare item is a teaching diagnostic — `@tier` attaches
            // to a `fn` declaration (write `@tier(compiled)` or `@tier(interpreted)` before `fn`).
            Tok::Tier => Err(ParseError::new(
                self.pos(),
                "`tier` is the DN-58 §C execution-mode hint — write `@tier(compiled)` or \
                 `@tier(interpreted)` before a `fn` declaration (not as a bare item); \
                 e.g. `@tier(compiled) fn hot_path(…) => … = …` (never-silent — G2)"
                    .to_owned(),
            )),
            // DN-58 §A/§B (M-667): `fuse` and `reclaim` are expressions, not top-level items.
            Tok::Fuse => Err(ParseError::new(
                self.pos(),
                "`fuse(a, b)` is an expression (DN-58 §A — lawful binary merge over the Fuse \
                 semilattice; RFC-0008 RT6), not a top-level item — write it inside a `fn` body"
                    .to_owned(),
            )),
            Tok::Reclaim => Err(ParseError::new(
                self.pos(),
                "`reclaim(policy) { body }` is an expression (DN-58 §B — supervised scope; \
                 RFC-0008 RT7), not a top-level item — write it inside a `fn` body"
                    .to_owned(),
            )),
            // DN-03 §1 / M-664: `consume <expr>` is an **expression** (affine acquisition of a
            // `Substrate`, LR-8), not a top-level item — teaching diagnostic points into a fn body
            // (never a silent accept, G2). `grow` is superseded by `derive` (DN-38 §8.1 / M-812).
            Tok::Consume => Err(ParseError::new(
                self.pos(),
                "`consume <expr>` is an expression (DN-03 §1 / M-664 — affine acquisition of a \
                 `Substrate` value, LR-8), not a top-level item — write it inside a `fn` body"
                    .to_owned(),
            )),
            Tok::Grow => Err(ParseError::new(
                self.pos(),
                "`grow` is superseded by `derive` (DN-38 §8.1 / M-812) — write `derive Name for T` \
                 to apply a generative-lowering rule, or `lower Name[params] = <rhs>` to define one; \
                 `grow` can no longer open a program or be used as an identifier (G2)"
                    .to_owned(),
            )),
            // M-662: a `phylum` header must be the *first* token of the program (before the nodule
            // blocks); reaching one at item position means it was misplaced after a nodule began.
            Tok::Phylum => Err(ParseError::new(
                self.pos(),
                "a `phylum <path>` header opens the program — it must come before the first \
                 `nodule` block, not at item position (RFC-0006 §4.3; phylum is a grouping, not a \
                 `phylum { … }` container)"
                    .to_owned(),
            )),
            // RFC-0037 D5: `lambda` is an expression keyword, not a top-level item (teaching, G2).
            Tok::Lambda => Err(ParseError::new(
                self.pos(),
                "`lambda(…) => …` is an expression (RFC-0037 D5), not a top-level item — write it \
                 inside a `fn` body"
                    .to_owned(),
            )),
            // DN-53 / M-811: `object` is now ACTIVE — `object Name[params] { Ctor(…); via …; impl
            // …; fn … }` is a composition surface that desugars to `type`+`impl`+forwarding-impls
            // at check time. Zero kernel growth (KC-3); `reveal`-able per DN-38 §5.
            Tok::Object => self.parse_object_decl(Vis::Private).map(Item::Object),
            // DN-54 / M-812: `lower` and `derive` are now **active** (settles the grow→derive
            // reconciliation per DN-38 §8.1). `lower Name[params] = <rhs>` defines a rule;
            // `derive Name for T` applies one.
            Tok::Lower => self.parse_lower_decl().map(Item::Lower),
            Tok::Derive => self.parse_derive_decl().map(Item::Derive),
            _ => self.err(
                "a top-level item (`use`, `pub`, `default paradigm`, `type`, `trait`, `impl`, \
                 `fn`, `thaw fn`, `lower`, or `derive`)",
            ),
        }
    }

    /// Parse a `pub`-prefixed top-level item (M-662). `pub` exports a top-level `fn`/`trait`/`type`
    /// (or `thaw fn`) to the other nodules of the phylum; it is **only** valid there. A `pub use` /
    /// `pub default` / `pub impl` (or `pub` before anything else) is an explicit refusal — never a
    /// silent accept (G2): a `use` imports rather than re-exports, and `impl`/`default` are not part
    /// of the `pub` namespace.
    fn parse_pub_item(&mut self) -> Result<Item, ParseError> {
        self.expect(&Tok::Pub, "`pub`")?;
        match self.cur() {
            Tok::Type => self.parse_type_decl(Vis::Pub).map(Item::Type),
            Tok::Trait => self.parse_trait_decl(Vis::Pub).map(Item::Trait),
            Tok::Fn | Tok::Thaw => self.parse_fn_decl(Vis::Pub).map(Item::Fn),
            // DN-58 §C (M-667): `pub @tier(mode) fn …` — execution-mode hint on a pub fn.
            Tok::At => self.parse_tier_fn_decl(Vis::Pub).map(Item::Fn),
            // DN-53 / M-811: `pub object` exports the composed type name to other nodules (M-662).
            Tok::Object => self.parse_object_decl(Vis::Pub).map(Item::Object),
            Tok::Use => Err(ParseError::new(
                self.pos(),
                "`pub use` is not a form — a `use` imports a name into this nodule, it does not \
                 re-export it (M-662); drop the `pub`"
                    .to_owned(),
            )),
            Tok::Impl => Err(ParseError::new(
                self.pos(),
                "`pub impl` is not a form — an `impl` is not `pub`-gated (its coherence is \
                 phylum-wide and pub-blind; M-662/RFC-0019 §4.5); drop the `pub`"
                    .to_owned(),
            )),
            Tok::Default => Err(ParseError::new(
                self.pos(),
                "`pub default` is not a form — `default paradigm` is nodule-scope ambient state, \
                 not an exportable item (M-662); drop the `pub`"
                    .to_owned(),
            )),
            _ => self.err(
                "`pub` must be followed by `fn`, `trait`, `type`, `object`, or `thaw fn` \
                 (M-662 — only those top-level items are exportable)",
            ),
        }
    }

    /// `use path` (specific) or `use path.*` (glob) — a cross-nodule import (M-662; RFC-0006 §4.3).
    /// A trailing `.*` makes it a **glob** (import every `pub` name under the path); otherwise the
    /// path's last segment names the imported item. A `*` anywhere but the final segment is an
    /// explicit parse error — the lexer emits `Tok::Star` for any `*`; this production is what
    /// restricts the glob `*` to the final position. `use` is never `pub`-gated.
    fn parse_use(&mut self) -> Result<UsePath, ParseError> {
        self.expect(&Tok::Use, "`use`")?;
        // A `use` path is a dotted path whose final segment may be `*` (the glob). Parse the dotted
        // path, then check for a trailing `.*`.
        let mut segs = vec![self.ident()?];
        let mut glob = false;
        while self.eat(&Tok::Dot) {
            if self.eat(&Tok::Star) {
                glob = true;
                break;
            }
            segs.push(self.ident()?);
        }
        // A glob needs a prefix to glob under (`use *` alone is meaningless).
        if glob && segs.is_empty() {
            return self.err("a glob `use` needs a path prefix (`use a.b.*`), not a bare `*`");
        }
        Ok(UsePath {
            path: Path(segs),
            glob,
        })
    }

    /// A bare paradigm tag (`Binary|Ternary|Dense|VSA`) for an ambient declaration (RFC-0012 §4.2).
    fn parse_paradigm(&mut self) -> Result<Paradigm, ParseError> {
        let p = match self.cur() {
            Tok::Binary => Paradigm::Binary,
            Tok::Ternary => Paradigm::Ternary,
            Tok::Dense => Paradigm::Dense,
            Tok::Vsa => Paradigm::Vsa,
            _ => return self.err("a paradigm (`Binary|Ternary|Dense|VSA`)"),
        };
        self.bump();
        Ok(p)
    }

    fn parse_type_decl(&mut self, vis: Vis) -> Result<TypeDecl, ParseError> {
        self.expect_keyword(&Tok::Type)?;
        let name = self.ident()?;
        let params = self.parse_type_params_opt()?;
        self.expect(&Tok::Eq, "`=` before the constructors")?;
        let mut ctors = vec![self.parse_ctor()?];
        while self.eat(&Tok::Pipe) {
            ctors.push(self.parse_ctor()?);
        }
        Ok(TypeDecl {
            vis,
            name,
            params,
            ctors,
        })
    }

    fn parse_ctor(&mut self) -> Result<Ctor, ParseError> {
        let name = self.ident()?;
        let mut fields = Vec::new();
        if self.eat(&Tok::LParen) {
            fields = self.comma_separated(None, Self::parse_type_ref)?;
            self.expect(&Tok::RParen, "`)` to close the constructor fields")?;
        }
        Ok(Ctor { name, fields })
    }

    fn parse_trait_decl(&mut self, vis: Vis) -> Result<TraitDecl, ParseError> {
        self.expect_keyword(&Tok::Trait)?;
        let name = self.ident()?;
        let params = self.parse_type_params_opt()?;
        self.expect(&Tok::LBrace, "`{` to open the trait body")?;
        let mut sigs = Vec::new();
        while !self.at(&Tok::RBrace) {
            sigs.push(self.parse_fn_sig()?);
            self.expect_terminator("trait signature")?; // DN-57 §3 (M-818)
        }
        self.expect(&Tok::RBrace, "`}` to close the trait body")?;
        Ok(TraitDecl {
            vis,
            name,
            params,
            sigs,
        })
    }

    fn parse_fn_sig(&mut self) -> Result<FnSig, ParseError> {
        self.expect(&Tok::Fn, "`fn` in the trait body")?;
        self.parse_sig_tail()
    }

    /// `thaw? fn …` with the caller-supplied cross-nodule visibility `vis` (M-662). Top-level fns get
    /// `Vis::Pub` iff a `pub` preceded them; impl methods are always parsed with `Vis::Private` (an
    /// `impl` is not `pub`-gated — its method `vis` is inert).
    fn parse_fn_decl(&mut self, vis: Vis) -> Result<FnDecl, ParseError> {
        let thaw = self.eat(&Tok::Thaw);
        self.expect(&Tok::Fn, "`fn`")?;
        let sig = self.parse_sig_tail()?;
        self.expect(&Tok::Eq, "`=` before the function body")?;
        let body = self.parse_expr()?;
        Ok(FnDecl {
            vis,
            thaw,
            tier: None,
            sig,
            body,
        })
    }

    /// DN-58 §C (M-667): `@tier(compiled)` / `@tier(interpreted)` followed by a function
    /// declaration. Syntax: `@tier ( compiled | interpreted ) thaw? fn …`. The mode is recorded on
    /// `FnDecl.tier` (non-semantic / NFR-7 — never a behavioural switch, only a performance hint).
    /// Never-silent on a bad mode name: `compiled` and `interpreted` are the only valid modes (G2).
    ///
    /// The `@` at item position is unambiguous — guarantee annotations live inside type refs, not at
    /// the top-level item grammar — so `@` here means "execution-mode tier attribute" (DN-58 §C).
    fn parse_tier_fn_decl(&mut self, vis: Vis) -> Result<FnDecl, ParseError> {
        self.expect(&Tok::At, "`@` before the `tier` attribute")?;
        self.expect_keyword(&Tok::Tier)?;
        self.expect(&Tok::LParen, "`(` to open the `@tier` mode argument")?;
        let mode = match self.cur().clone() {
            Tok::Ident(ref s) if s == "compiled" => {
                self.bump();
                ExecutionMode::Compiled
            }
            Tok::Ident(ref s) if s == "interpreted" => {
                self.bump();
                ExecutionMode::Interpreted
            }
            _ => {
                return Err(ParseError::new(
                    self.pos(),
                    "`@tier` mode must be `compiled` or `interpreted` (DN-58 §C; RFC-0004 \
                     ExecutionMode — never-silent on a bad mode name, G2)"
                        .to_owned(),
                ));
            }
        };
        self.expect(&Tok::RParen, "`)` to close the `@tier` mode argument")?;
        let thaw = self.eat(&Tok::Thaw);
        self.expect(&Tok::Fn, "`fn` after `@tier(mode)`")?;
        let sig = self.parse_sig_tail()?;
        self.expect(&Tok::Eq, "`=` before the function body")?;
        let body = self.parse_expr()?;
        Ok(FnDecl {
            vis,
            thaw,
            tier: Some(mode),
            sig,
            body,
        })
    }

    /// DN-58 §A (M-667): `fuse ( <left> , <right> )` — lawful binary merge over the `Fuse`
    /// semilattice instance (RFC-0008 RT6). Both operands are expressions; the checker verifies type
    /// homogeneity and fusibility (never-silent on a missing `Fuse` instance, G2). Syntax sugar —
    /// no new L0 node (KC-3).
    fn parse_fuse_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Fuse)?;
        self.expect(&Tok::LParen, "`(` to open `fuse` operands")?;
        let left = self.parse_expr()?;
        self.expect(&Tok::Comma, "`,` between `fuse` operands")?;
        let right = self.parse_expr()?;
        self.expect(&Tok::RParen, "`)` to close `fuse` operands")?;
        Ok(Expr::Fuse {
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// DN-58 §B (M-667): `reclaim ( <policy> ) { <body> }` — attach a reified supervision/
    /// reclamation policy to a structured scope (RFC-0008 RT7). The `policy` is an expression
    /// evaluating to a supervision policy; the `body` is any expression (typically a `colony { … }`
    /// block). Braces are required for `body` clarity (G2 / never-silent on malformed syntax).
    fn parse_reclaim_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Reclaim)?;
        self.expect(&Tok::LParen, "`(` to open the `reclaim` policy argument")?;
        let policy = self.parse_expr()?;
        self.expect(&Tok::RParen, "`)` to close the `reclaim` policy argument")?;
        self.expect(&Tok::LBrace, "`{` to open the `reclaim` supervised body")?;
        let body = self.parse_expr()?;
        self.expect(&Tok::RBrace, "`}` to close the `reclaim` supervised body")?;
        Ok(Expr::Reclaim {
            policy: Box::new(policy),
            body: Box::new(body),
        })
    }

    /// The shared `name <params>? ( value_params? ) -> ret !{effects}?` tail of a signature. A
    /// function's type-parameters may carry **trait bounds** (`<T: Cmp + Ord<T>>`; RFC-0019 §4.1) —
    /// the dictionary site — so this uses
    /// [`parse_type_params_bounded`](Self::parse_type_params_bounded). The optional
    /// `!{ eff1, eff2 }` **effect annotation** (RFC-0014 §3.4; M-660/M-677) follows the return type;
    /// absent ⇒ the empty (pure) effect set. Each effect may carry an optional budget bound
    /// `eff(<=N)` (M-677; RFC-0014 §4.5 I4); `N` may carry a unit suffix (`KiB`, `MiB`, `GiB`).
    fn parse_sig_tail(&mut self) -> Result<FnSig, ParseError> {
        let name = self.ident()?;
        // RFC-0037 D2: type parameters in `[T]` (may carry bounds — the dictionary site), const/width
        // parameters in `{N}` (bare names). Both collect into one `params` list; their `[…]`/`{…}`
        // bracket is the surface kind hint, and `classify_params` resolves the authoritative kind by
        // usage (a `[T]` used only in a width slot is a never-silent error, not a silent reclassify).
        let mut raw_params = self.parse_type_params_bounded()?;
        raw_params.extend(self.parse_const_params_opt()?);
        let names: Vec<String> = raw_params.iter().map(|p| p.name.clone()).collect();
        if let Some(dup) = first_duplicate_str(&names) {
            return Err(ParseError::new(
                self.pos(),
                format!(
                    "parameter `{dup}` is declared twice across the `[…]`/`{{…}}` lists — each \
                     type/const parameter name is unique (RFC-0037 D2; never a silent shadow, G2)"
                ),
            ));
        }
        self.expect(&Tok::LParen, "`(` to open the parameter list")?;
        let value_params = self.parse_params_opt()?;
        self.expect(&Tok::RParen, "`)` to close the parameter list")?;
        self.expect_return_arrow()?;
        let ret = self.parse_type_ref()?;
        let (effects, effect_budgets) = self.parse_effects_opt()?;
        let params = classify_params(raw_params, &value_params, &ret)?;
        Ok(FnSig {
            name,
            params,
            value_params,
            ret,
            effects,
            effect_budgets,
        })
    }

    /// Expect the `=>` return/function arrow (RFC-0037 D4). A leftover `->` (still lexed as
    /// [`Tok::Arrow`]) gets an explicit teaching reject rather than a confusing token error (G2).
    fn expect_return_arrow(&mut self) -> Result<(), ParseError> {
        if self.at(&Tok::Arrow) {
            return Err(ParseError::new(
                self.pos(),
                "the arrow is now `=>`, not `->` (RFC-0037 D4 retired the `->` glyph)".to_owned(),
            ));
        }
        self.expect(&Tok::FatArrow, "`=>` and a result type")?;
        Ok(())
    }

    /// `{ Ident (',' Ident)* }?` — const/width parameter declarations (RFC-0037 D2). Each is a bare
    /// name (no bounds — width params cannot carry trait bounds in v1, DN-42 §7), tagged
    /// [`ParamKind::Width`] as a hint (`classify_params` confirms by usage). This `{…}` is a distinct
    /// position from the `Binary{N}` width *slot* inside a type (parsed in `parse_base_type`), so the
    /// two `{…}` uses never collide — a const-param list is only ever right after the fn name (and an
    /// optional `[…]`), before `(`.
    fn parse_const_params_opt(&mut self) -> Result<Vec<TypeParam>, ParseError> {
        let mut params = Vec::new();
        if self.eat(&Tok::LBrace) {
            params = self.comma_separated(None, |p| {
                let name = p.ident()?;
                if p.at(&Tok::Colon) {
                    return Err(ParseError::new(
                        p.pos(),
                        "const/width parameters cannot carry trait bounds in v1 (DN-42 §7; bounds \
                         live only on `[T]` type parameters) — never a silent drop"
                            .to_owned(),
                    ));
                }
                Ok(TypeParam {
                    name,
                    kind: crate::ast::ParamKind::Width,
                    bounds: Vec::new(),
                })
            })?;
            self.expect(&Tok::RBrace, "`}` to close the const/width parameters")?;
        }
        Ok(params)
    }

    /// `lambda(params) => body` (RFC-0037 D5). Parses the typed parameter list and the body
    /// expression into an [`Expr::Lambda`] node. **Closure semantics — environment capture and
    /// dynamic fn-flow — are implemented (M-704 / RFC-0024 §4A):** the checker types it to `Ty::Fn`
    /// and monomorphization lowers it by Reynolds defunctionalization (a tag-sum struct + a generated
    /// `apply` dispatcher), so it parses, type-checks, **and evaluates**. **Multi-argument lambdas
    /// are now supported via currying (M-822 / RFC-0024 §4A.5/§4A.8):** a `lambda(p1, p2) => body`
    /// desugars downstream to `lambda(p1) => lambda(p2) => body`; partial application (fn-as-value)
    /// likewise yields a curried arrow `A -> B -> Z`. Zero-parameter lambdas are a never-silent
    /// refusal downstream (no type without a unit/nullary type — G2).
    /// Type/const parameters on a lambda (`lambda[T]{N}(…)`) are an explicit never-silent refusal here
    /// (the syntax is reserved by RFC-0037 D5 but the form is not yet wired), not a silent accept.
    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Lambda)?;
        if self.at(&Tok::LBracket) || self.at(&Tok::LBrace) {
            return Err(ParseError::new(
                self.pos(),
                "type/const parameters on a `lambda` (`lambda[T]{N}(…)`) land with M-704 (RFC-0037 \
                 D5 reserves the syntax; full closures are RFC-0024 §5) — never a silent accept"
                    .to_owned(),
            ));
        }
        self.expect(&Tok::LParen, "`(` to open the lambda parameter list")?;
        let params = self.parse_params_opt()?;
        self.expect(&Tok::RParen, "`)` to close the lambda parameter list")?;
        self.expect(
            &Tok::FatArrow,
            "`=>` between the lambda parameter list and its body (RFC-0037 D5)",
        )?;
        let body = self.parse_expr()?;
        Ok(Expr::Lambda {
            params,
            body: Box::new(body),
        })
    }

    /// `!{ eff(<=N)? (, eff(<=N)?)* }?` — the optional **effect annotation** after a signature's
    /// return type (RFC-0014 §3.4/§4.5 I3/I4; M-660/M-677). When the next token is `!`, consume
    /// `!{` … `}` and parse a comma-separated list of effect entries. Each entry is an effect
    /// **name** (plain identifier — the closed kernel kinds `retry|alloc|io|cascade|time` plus user
    /// `Named` effects; RFC-0014 §4.5) followed by an **optional budget bound** `(<=N<unit>?)` where
    /// `N` is a positive integer and `unit` is an optional binary-size suffix (`KiB` = 1024,
    /// `MiB` = 1048576, `GiB` = 1073741824; M-677). The `<=` is parsed as the two-token sequence
    /// `<` `=` (RFC-0037 D1 retired `<=` as an infix operator glyph; in the effect-annotation
    /// context it is a bound delimiter, never an operator).
    ///
    /// The empty set `!{}` is valid (an explicit "declares no effects"). Absent `!` ⇒ pure (empty
    /// set — RFC-0014 I5). A **duplicate** effect name is an explicit refusal (G2 — never silently
    /// deduped). A **zero budget** `(<=0)` is an explicit refusal (a zero budget would exhaust
    /// immediately — never a silent trap). A **negative** budget literal is refused (budgets are
    /// non-negative). Returns the effect name list (in source order) and a separate budget map
    /// (keyed by name, present only for budgeted effects).
    fn parse_effects_opt(
        &mut self,
    ) -> Result<(Vec<String>, std::collections::BTreeMap<String, u64>), ParseError> {
        if !self.eat(&Tok::Bang) {
            return Ok((Vec::new(), std::collections::BTreeMap::new()));
        }
        self.expect(
            &Tok::LBrace,
            "`{` after `!` to open the effect set (RFC-0014 §3.4)",
        )?;
        // The empty written set `!{}` is valid (an explicit "declares no effects").
        let set_start = self.pos();
        let mut effects: Vec<String> = Vec::new();
        let mut effect_budgets: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        if !self.at(&Tok::RBrace) {
            // Parse comma-separated `eff (<=N unit?)?` entries.
            loop {
                let eff_name = self.ident()?;
                // Optional budget bound `(<=N<unit>?)`.
                if self.eat(&Tok::LParen) {
                    // Expect `<=` (two tokens: `<` then `=`; RFC-0037 D1 note above).
                    if !self.eat(&Tok::LAngle) {
                        return self.err("`<=` to open the budget bound (e.g. `retry(<=3)`)");
                    }
                    if !self.eat(&Tok::Eq) {
                        return self.err("`=` after `<` to form the `<=` bound operator in an effect budget (RFC-0014 §4.5 I4)");
                    }
                    let budget_start = self.pos();
                    // Parse the numeric value.
                    let raw: i64 =
                        match *self.cur() {
                            Tok::Int(n) => {
                                self.bump();
                                n
                            }
                            _ => return self.err(
                                "a non-negative integer as the effect budget (e.g. `retry(<=3)`)",
                            ),
                        };
                    // Reject non-positive budgets *before* the `as u64` cast below: a zero budget
                    // would exhaust before any work fires, and guarding `< 0` here keeps a negative
                    // `i64` (defensive — should not arise from `Tok::Int`) from wrapping to a huge
                    // `u64`. Either way it is an explicit, never-silent error (RFC-0014 §4.5 I4, G2).
                    if raw <= 0 {
                        return Err(ParseError::new(
                            budget_start,
                            format!(
                                "effect budget `{raw}` must be positive (> 0) — a zero budget would \
                                 exhaust immediately (and a negative budget is rejected before the \
                                 unsigned cast) (RFC-0014 §4.5 I4; never a silent trap, G2)"
                            ),
                        ));
                    }
                    let raw_u64 = raw as u64;
                    // Optional unit suffix (KiB / MiB / GiB — RFC-0014 §3.4 example `alloc(<=64KiB)`).
                    let budget: u64 = if let Tok::Ident(unit) = self.cur().clone() {
                        let multiplier: u64 = match unit.as_str() {
                            "KiB" => 1024,
                            "MiB" => 1024 * 1024,
                            "GiB" => 1024 * 1024 * 1024,
                            other => return Err(ParseError::new(
                                self.pos(),
                                format!(
                                    "unknown budget unit `{other}` — supported units are `KiB` \
                                     (1024), `MiB` (1048576), `GiB` (1073741824); omit the unit \
                                     for a plain count (RFC-0014 §4.5 I4; never a silent default, G2)"
                                ),
                            )),
                        };
                        self.bump();
                        raw_u64.checked_mul(multiplier).ok_or_else(|| {
                            ParseError::new(
                                budget_start,
                                format!(
                                "effect budget `{raw_u64}{unit}` overflows u64 — use a smaller \
                                 budget (RFC-0014 §4.5 I4)"
                            ),
                            )
                        })?
                    } else {
                        raw_u64
                    };
                    self.expect(&Tok::RParen, "`)` to close the effect budget bound")?;
                    effect_budgets.insert(eff_name.clone(), budget);
                }
                effects.push(eff_name);
                if !self.eat(&Tok::Comma) {
                    break;
                }
                // Trailing comma before `}` is not allowed (keep it strict — G2).
                if self.at(&Tok::RBrace) {
                    return Err(ParseError::new(
                        self.pos(),
                        "trailing comma before `}` in an effect annotation is not allowed — \
                         remove the trailing `,` (RFC-0014 §4.5; never a silent accept)"
                            .to_owned(),
                    ));
                }
            }
        }
        self.expect(&Tok::RBrace, "`}` to close the effect set")?;
        if let Some(dup) = first_duplicate_str(&effects) {
            // Point at the effect set itself (not after the closing `}`) for a clearer diagnostic.
            return Err(ParseError::new(
                set_start,
                format!(
                    "duplicate effect `{dup}` in the effect annotation — list each declared effect \
                     once (RFC-0014 §4.5; a repeated effect is a never-silent refusal, not a silent \
                     dedup)"
                ),
            ));
        }
        Ok((effects, effect_budgets))
    }

    fn parse_params_opt(&mut self) -> Result<Vec<Param>, ParseError> {
        self.comma_separated_until(&Tok::RParen, |p| {
            let name = p.ident()?;
            p.expect(&Tok::Colon, "`:` and the parameter type")?;
            let ty = p.parse_type_ref()?;
            Ok(Param { name, ty })
        })
    }

    /// `< name (, name)* >?` — **unbounded** type-parameter names, for `type`/`trait` declarations
    /// (stage-1: data/trait type-params are unbounded abstractions — RFC-0019 §4.1 / RFC-0007 §12.1).
    /// A bound (`<T: Cmp>`) here is an **explicit refusal** (deferred to a later stage), never
    /// silently dropped — bounds belong only on function type-params (the dictionary site).
    fn parse_type_params_opt(&mut self) -> Result<Vec<String>, ParseError> {
        let mut params = Vec::new();
        if self.eat(&Tok::LBracket) {
            params = self.comma_separated(None, |p| {
                let name = p.ident()?;
                if p.at(&Tok::Colon) {
                    return Err(ParseError::new(
                        p.pos(),
                        "bounds on `type`/`trait` type-parameters are deferred in stage-1 \
                         (RFC-0019 §4.1 — bounds live only on function type-parameters, the \
                         dictionary site); write the bound on the bounded `fn` instead"
                            .to_owned(),
                    ));
                }
                Ok(name)
            })?;
            self.expect(&Tok::RBracket, "`]` to close the type parameters")?;
        }
        Ok(params)
    }

    /// `< type_param (, type_param)* >?` where `type_param ::= Ident (':' bound)?` — **bounded**
    /// type-parameters for **functions** (RFC-0019 §4.1). An unbounded `T` yields
    /// `TypeParam { bounds: [] }` (the §11 identity, so every v0 program still parses).
    fn parse_type_params_bounded(&mut self) -> Result<Vec<TypeParam>, ParseError> {
        let mut params = Vec::new();
        if self.eat(&Tok::LBracket) {
            params = self.comma_separated(None, Self::parse_type_param)?;
            self.expect(&Tok::RBracket, "`]` to close the type parameters")?;
        }
        Ok(params)
    }

    /// One bounded type-parameter `Ident (':' bound)?` (RFC-0019 §4.1).
    fn parse_type_param(&mut self) -> Result<TypeParam, ParseError> {
        let name = self.ident()?;
        let bounds = if self.eat(&Tok::Colon) {
            self.parse_bound()?
        } else {
            Vec::new()
        };
        Ok(TypeParam {
            name,
            kind: crate::ast::ParamKind::Type,
            bounds,
        })
    }

    /// A trait bound `Ident type_args? ('+' Ident type_args?)*` — one or more trait references
    /// (RFC-0019 §4.1 `bound`). Reuses the existing type-argument parser for each trait's `<…>`.
    fn parse_bound(&mut self) -> Result<Vec<TraitRef>, ParseError> {
        let mut bounds = vec![self.parse_trait_ref()?];
        while self.eat(&Tok::Plus) {
            bounds.push(self.parse_trait_ref()?);
        }
        Ok(bounds)
    }

    /// One trait reference in a bound — `Cmp` or `Cmp<Binary{8}>` (RFC-0019 §4.1).
    fn parse_trait_ref(&mut self) -> Result<TraitRef, ParseError> {
        let name = self.ident()?;
        let args = self.parse_type_args_opt()?;
        Ok(TraitRef { name, args })
    }

    /// Top-level `impl` dispatcher (M-664). Disambiguates the **trait-instance** form
    /// `impl Trait[args]? for T { … }` (RFC-0019 §4.1 → [`Item::Impl`]) from the **inherent**
    /// method block `impl T { … }` (DN-03 §1 → [`Item::InherentImpl`]) by parsing the head type
    /// once and branching on the following `for` (trait) vs `{` (inherent). Never silent (G2): any
    /// other follower is an explicit parse error naming both forms.
    fn parse_impl_item(&mut self) -> Result<Item, ParseError> {
        self.expect(&Tok::Impl, "`impl`")?;
        // The head is a base type: a trait ref `Trait[args]?` parses as `Named(trait, args)`, and an
        // inherent target `T` (`Binary{8}`, `Foo[X]`, …) parses as its own base type. Disambiguate
        // *after* the head by the follower.
        let head = self.parse_base_type()?;
        if self.at(&Tok::For) {
            // Trait-instance: `impl Trait[args]? for T { … }`. The head before `for` must be a
            // trait *name* (a `Named` base type) — never silent if it is a repr/structural type (G2).
            self.bump(); // `for`
            let (trait_name, trait_args) = match head {
                BaseType::Named(name, args) => (name, args),
                _ => {
                    return Err(ParseError::new(
                        self.pos(),
                        "the item before `for` in a trait `impl … for …` must be a trait name \
                         (e.g. `impl Cmp[Binary{8}] for T`), not a repr/structural type \
                         (RFC-0019 §4.1; never silent — G2)"
                            .to_owned(),
                    ));
                }
            };
            let for_ty = self.parse_type_ref()?;
            let methods = self.parse_impl_body()?;
            Ok(Item::Impl(ImplDecl {
                trait_name,
                trait_args,
                for_ty,
                methods,
            }))
        } else if self.at(&Tok::LBrace) {
            // Inherent block: `impl T { fn … }` (M-664). The head *is* the target type.
            let for_ty = TypeRef::unguaranteed(head);
            let methods = self.parse_impl_body()?;
            Ok(Item::InherentImpl(InherentImplDecl { for_ty, methods }))
        } else {
            Err(ParseError::new(
                self.pos(),
                "expected `for` (a trait instance: `impl Trait for T { … }`) or `{` (an inherent \
                 method block: `impl T { … }`) after the type in an `impl` (RFC-0019 §4.1 / \
                 DN-03 §1; never silent — G2)"
                    .to_owned(),
            ))
        }
    }

    /// Parse a trait-instance `impl Trait[args]? for T { … }` (RFC-0019 §4.1 → [`ImplDecl`]). Used
    /// where only the trait form is valid — inside an `object` body and the `parse_impl_item`
    /// trait branch share the `{ … }` body via [`Self::parse_impl_body`].
    fn parse_impl_decl(&mut self) -> Result<ImplDecl, ParseError> {
        self.expect(&Tok::Impl, "`impl`")?;
        let trait_name = self.ident()?;
        let trait_args = self.parse_type_args_opt()?;
        self.expect(
            &Tok::For,
            "`for` after the trait in an `impl` (RFC-0019 §4.1)",
        )?;
        let for_ty = self.parse_type_ref()?;
        let methods = self.parse_impl_body()?;
        Ok(ImplDecl {
            trait_name,
            trait_args,
            for_ty,
            methods,
        })
    }

    /// Parse an `impl` body `{ fn … }` (shared by trait-instance and inherent impls; M-664). A
    /// `pub` on a method is an explicit refusal (M-662 — method visibility follows the trait/type,
    /// not the impl; never silent — G2). Methods are parsed with `Vis::Private` (the field is inert).
    fn parse_impl_body(&mut self) -> Result<Vec<FnDecl>, ParseError> {
        self.expect(&Tok::LBrace, "`{` to open the `impl` body")?;
        let mut methods = Vec::new();
        while !self.at(&Tok::RBrace) {
            if self.at(&Tok::Pub) {
                return Err(ParseError::new(
                    self.pos(),
                    "an `impl` method is not `pub`-gated — visibility of a method follows the \
                     trait/type, not the impl (M-662); drop the `pub`"
                        .to_owned(),
                ));
            }
            methods.push(self.parse_fn_decl(Vis::Private)?);
            self.expect_terminator("impl method")?; // DN-57 §3 (M-818)
        }
        self.expect(&Tok::RBrace, "`}` to close the `impl` body")?;
        Ok(methods)
    }

    /// `object Name[params]? { Ctor(T1, T2); (via N : Trait[args]?)* (impl …)* (fn …)* }`
    ///
    /// Parses an object composition surface declaration (DN-53, M-811). The body must open with
    /// exactly one constructor clause (same syntax as a `TypeDecl` constructor, followed by `;`),
    /// then zero-or-more `via` delegation clauses, zero-or-more `impl` blocks, and zero-or-more
    /// `fn` definitions, in any order after the constructor — the constructor must come first.
    ///
    /// `via N : Trait[args]?` — `N` is a decimal field-index literal (`u32`), not an identifier.
    /// `impl` and `fn` clauses follow the same grammar as top-level `impl`/`fn` items.
    ///
    /// Never silent (G2): a duplicate constructor, out-of-range field index (checked at check time),
    /// and unknown body keywords are all explicit parse errors.
    fn parse_object_decl(&mut self, vis: Vis) -> Result<ObjectDecl, ParseError> {
        self.expect_keyword(&Tok::Object)?;
        let name = self.ident()?;
        let params = self.parse_type_params_opt()?;
        self.expect(&Tok::LBrace, "`{` to open the `object` body")?;

        // The first element must be the constructor clause: `Name(T1, T2)` followed by `;`.
        // An `object` body with no constructor is an explicit error — the name is the type name,
        // not a constructor (G2: never-silent).
        if self.at(&Tok::RBrace) {
            return Err(ParseError::new(
                self.pos(),
                "an `object` body must have at least one constructor clause (`Name(T1, T2);`) \
                 before any `via`, `impl`, or `fn` items (DN-53 §A.2.1; never-silent, G2)"
                    .to_owned(),
            ));
        }
        let ctor = self.parse_ctor()?;
        // The constructor is terminated by a mandatory `;` (it is not an expression statement).
        self.expect(
            &Tok::Semi,
            "`;` after the constructor clause in an `object` body (DN-53 §A.2.1)",
        )?;

        // Remaining body items: `via`, `impl`, `fn` (in any order, each terminated by `;`/`}`)
        let mut via_decls: Vec<ViaDecl> = Vec::new();
        let mut impls: Vec<ImplDecl> = Vec::new();
        let mut fns: Vec<FnDecl> = Vec::new();

        while !self.at(&Tok::RBrace) {
            match self.cur() {
                // `via N : Trait[args]?` — delegation clause (DN-53 §A.3.2).
                Tok::Via => {
                    self.bump(); // consume `via`
                                 // The field index is a decimal integer literal (never a name — positional, G2).
                    let field_idx = match self.cur().clone() {
                        Tok::Int(n) if n >= 0 => {
                            let idx = u32::try_from(n).map_err(|_| {
                                ParseError::new(
                                    self.pos(),
                                    format!(
                                        "`via` field index `{n}` overflows u32 — field indices \
                                         must fit in a u32 (DN-53 §A.3.2; G2)"
                                    ),
                                )
                            })?;
                            self.bump();
                            idx
                        }
                        Tok::Int(n) => {
                            return Err(ParseError::new(
                                self.pos(),
                                format!(
                                    "`via` field index must be a non-negative integer; got `{n}` \
                                     (DN-53 §A.3.2; G2)"
                                ),
                            ));
                        }
                        _ => {
                            return Err(ParseError::new(
                                self.pos(),
                                "expected a non-negative decimal field index after `via` \
                                 (e.g. `via 0 : Trait`) — got a non-integer token (DN-53 §A.3.2; G2)"
                                    .to_owned(),
                            ));
                        }
                    };
                    self.expect(
                        &Tok::Colon,
                        "`:` after the field index in a `via` clause (`via N : Trait`)",
                    )?;
                    let trait_name = self.ident()?;
                    let trait_args = self.parse_type_args_opt()?;
                    self.expect_terminator("`via` delegation clause")?; // DN-57 §3 (M-818)
                    via_decls.push(ViaDecl {
                        field_idx,
                        trait_name,
                        trait_args,
                    });
                }
                // `impl Trait for …` — explicit trait instance inside the object body.
                Tok::Impl => {
                    impls.push(self.parse_impl_decl()?);
                    self.expect_terminator("object `impl` member")?; // DN-57 §3 (M-818)
                }
                // `fn …` / `thaw fn …` — inherent function.
                Tok::Fn | Tok::Thaw => {
                    fns.push(self.parse_fn_decl(Vis::Private)?);
                    self.expect_terminator("object `fn` member")?; // DN-57 §3 (M-818)
                }
                // `pub` inside an object body is not supported — methods/impl items are not
                // re-exported independently; the object itself carries its `pub` vis (G2).
                Tok::Pub => {
                    return Err(ParseError::new(
                        self.pos(),
                        "`pub` inside an `object` body is not valid — the object's own `pub` \
                         governs visibility of the composed type name; individual `fn`/`impl` \
                         items inside the body are not independently exported (DN-53 §A.2.1; G2)"
                            .to_owned(),
                    ));
                }
                _ => {
                    return Err(ParseError::new(
                        self.pos(),
                        "unexpected token inside `object` body — expected `via`, `impl`, `fn`, \
                         `thaw`, or `}` (DN-53 §A.2.1)"
                            .to_owned(),
                    ));
                }
            }
        }

        self.expect(&Tok::RBrace, "`}` to close the `object` body")?;
        Ok(ObjectDecl {
            vis,
            name,
            params,
            ctor,
            via_decls,
            impls,
            fns,
        })
    }

    // ---- lower / derive (DN-54 / M-812) ----

    /// `lower Name [params]? = <rhs>` — a user-defined generative-lowering rule (DN-54 §3).
    ///
    /// Grammar:
    /// ```text
    /// lower_decl ::= 'lower' Ident ( '[' ident_list ']' )? '=' expr
    /// ```
    ///
    /// The `lower` keyword is consumed by the caller (`parse_item`); this method starts at the
    /// rule name. The `params` are unbound type-variable names in the RHS — no bounds at this
    /// stage (KC-3: the checker validates the RHS is IL-grammar-clean and lowers only to existing
    /// L0 nodes). Never silent: a missing `=` or a missing name is an explicit [`ParseError`] (G2).
    fn parse_lower_decl(&mut self) -> Result<LowerDecl, ParseError> {
        self.expect(&Tok::Lower, "`lower`")?;
        let name = self.ident()?;
        // Optional type-parameter list `[T, U, …]` — reuses the existing `[…]` form.
        let params = self.parse_type_params_opt()?;
        self.expect(
            &Tok::Eq,
            "`=` after the rule name/params in a `lower` declaration (DN-54 §3)",
        )?;
        // The RHS is either **item-shaped** (`impl Trait for T { … }` — DN-54 §10.1(b) / §10.3
        // Model A, the sibling-injection template; parsed by [`Self::parse_lower_item_rhs`]) or a
        // full **expression** (the v0 form, type-checked against the IL grammar at check time). The
        // leading `impl` keyword disambiguates: only the item form can open with it (M-973).
        let rhs = if self.at(&Tok::Impl) {
            LowerRhs::Impl(self.parse_lower_item_rhs()?)
        } else {
            LowerRhs::Expr(self.parse_expr()?)
        };
        Ok(LowerDecl { name, params, rhs })
    }

    /// Parse an **item-shaped** `lower`-rule RHS (DN-54 §10.1(b) / §10.3 Model A; OQ-B; M-973). The
    /// **only** legal item form in v1 is a trait-instance template `impl Trait[args]? for T { … }`
    /// (DN-54 §10.6 OQ-B — the minimum for the §3.2 worked example; `type` aliases and standalone
    /// `fn` items are deliberately out of v1 scope, YAGNI). The template's `for` type is the rule's
    /// type parameter (e.g. `T`); a `derive Name for C` use site substitutes `C` for it and injects
    /// the concrete impl as a sibling (checked in `checkty.rs`). Reuses [`Self::parse_impl_decl`], so
    /// an inherent `impl T { … }` (no `for`) is a never-silent parse error (G2 — no silent
    /// over-generalization of the legal item set).
    fn parse_lower_item_rhs(&mut self) -> Result<ImplDecl, ParseError> {
        self.parse_impl_decl()
    }

    /// `derive Name for T` — use-site application of a generative-lowering rule (DN-54 / DN-38
    /// §8.1 / M-812; settles the `grow → derive` reconciliation).
    ///
    /// Grammar:
    /// ```text
    /// derive_decl ::= 'derive' Ident 'for' type_ref
    /// ```
    ///
    /// The `derive` keyword is consumed by the caller (`parse_item`). The `name` must resolve to a
    /// `lower`-declared rule in scope; `for_ty` is the target type the rule is instantiated over.
    /// Never silent: a missing `for` or an unknown rule name is an explicit error at check time (G2).
    fn parse_derive_decl(&mut self) -> Result<DeriveDecl, ParseError> {
        self.expect(&Tok::Derive, "`derive`")?;
        let name = self.ident()?;
        self.expect(
            &Tok::For,
            "`for` after the rule name in a `derive` application (DN-54 §4; `derive Name for T`)",
        )?;
        let for_ty = self.parse_type_ref()?;
        Ok(DeriveDecl { name, for_ty })
    }

    // ---- types ----

    fn parse_type_ref(&mut self) -> Result<TypeRef, ParseError> {
        // Depth-guarded (A4-02 / DN-40 A2): a crafted `A -> A -> …` chain recurses right here, and
        // `parse_base_type` recurses into nested `<…>` type arguments — both charge the shared
        // budget so deep type nesting is an explicit error, never a host-stack overflow (G2).
        self.enter_depth()?;
        let r = self.parse_type_ref_guarded();
        self.leave_depth();
        r
    }

    fn parse_type_ref_guarded(&mut self) -> Result<TypeRef, ParseError> {
        // Parse `base [@guarantee]` first.  Then, if a `->` follows, this whole LHS
        // becomes the argument of a function type and we parse the RHS recursively
        // (right-associative; `@` binds tighter than `->` — RFC-0024 §3).
        let lhs = self.parse_type_ref_atom()?;
        if self.at(&Tok::Arrow) {
            // RFC-0037 D4: `->` is retired; the function-type arrow is `=>` too. Teaching reject (G2).
            return Err(ParseError::new(
                self.pos(),
                "the function-type arrow is now `=>`, not `->` (RFC-0037 D4)".to_owned(),
            ));
        }
        if self.eat(&Tok::FatArrow) {
            // Right-associative: recurse for the result type (which may itself be `A => B`).
            let rhs = self.parse_type_ref()?;
            Ok(TypeRef::unguaranteed(BaseType::Fn(
                Box::new(lhs),
                Box::new(rhs),
            )))
        } else {
            Ok(lhs)
        }
    }

    /// Parse a single `base [@guarantee]` atom — without consuming a trailing `->`.  This is
    /// the non-recursive inner step of [`parse_type_ref`]; callers that need to stop *before*
    /// the arrow use this directly (none in v1 — `parse_sig_tail` already consumed its own
    /// `->` before calling `parse_type_ref` for the return type, so there is no ambiguity).
    fn parse_type_ref_atom(&mut self) -> Result<TypeRef, ParseError> {
        let base = self.parse_base_type()?;
        let guarantee = if self.eat(&Tok::At) {
            Some(self.parse_strength()?)
        } else {
            None
        };
        Ok(TypeRef { base, guarantee })
    }

    fn parse_base_type(&mut self) -> Result<BaseType, ParseError> {
        match self.cur().clone() {
            // RFC-0037 D2-b: `bin` is a short repr-keyword alias for `Binary` — it elaborates
            // identically (the exact same `BaseType::Binary`), so both spellings share one arm.
            Tok::Binary | Tok::BinShort => {
                self.bump();
                let w = self.braced_width()?;
                Ok(BaseType::Binary(w))
            }
            // RFC-0037 D2-b: `tern` is a short repr-keyword alias for `Ternary` (see `Tok::Binary`
            // arm above).
            Tok::Ternary | Tok::TernShort => {
                self.bump();
                let t = self.braced_width()?;
                Ok(BaseType::Ternary(t))
            }
            // RFC-0037 D2-b: `emb` is a short repr-keyword alias for `Dense` (see `Tok::Binary` arm
            // above).
            Tok::Dense | Tok::EmbShort => {
                self.bump();
                self.expect(&Tok::LBrace, "`{` after `Dense`/`emb`")?;
                let dim = self.u32_lit()?;
                self.expect(&Tok::Comma, "`,` between dim and dtype")?;
                let scalar = self.parse_scalar()?;
                self.expect(&Tok::RBrace, "`}` to close `Dense{…}`/`emb{…}`")?;
                Ok(BaseType::Dense(dim, scalar))
            }
            // RFC-0037 D2-b: `hvec` is a short repr-keyword alias for `VSA` (see `Tok::Binary` arm
            // above). `vec` was rejected (collides with `std.collections.Vec`) — it is never a
            // keyword, so it cannot reach this arm.
            Tok::Vsa | Tok::HvecShort => {
                self.bump();
                self.expect(&Tok::LBrace, "`{` after `VSA`/`hvec`")?;
                let model = self.ident()?;
                self.expect(&Tok::Comma, "`,` after the model")?;
                let dim = self.u32_lit()?;
                self.expect(&Tok::Comma, "`,` before the sparsity")?;
                let sparsity = self.parse_sparsity()?;
                self.expect(&Tok::RBrace, "`}` to close `VSA{…}`/`hvec{…}`")?;
                Ok(BaseType::Vsa {
                    model,
                    dim,
                    sparsity,
                })
            }
            Tok::Substrate => {
                self.bump();
                self.expect(&Tok::LBrace, "`{` after `Substrate`")?;
                let name = self.ident()?;
                self.expect(&Tok::RBrace, "`}` to close `Substrate{…}`")?;
                Ok(BaseType::Substrate(name))
            }
            // RFC-0032 D3 (M-749): `Seq{T, N}` — a repr-descriptor `{}` like the other repr types.
            // The element type `T` is any `TypeRef` (recursing into `parse_type_ref`); `N` is a
            // `u32` literal. The `{T, N}` order disambiguates trivially: the first descriptor slot is
            // a type, the second a length (separated by `,`).
            Tok::Seq => {
                self.bump();
                self.expect(&Tok::LBrace, "`{` after `Seq`")?;
                let elem = self.parse_type_ref()?;
                self.expect(&Tok::Comma, "`,` between the element type and the length")?;
                let len = self.u32_lit()?;
                self.expect(&Tok::RBrace, "`}` to close `Seq{…}`")?;
                Ok(BaseType::Seq {
                    elem: Box::new(elem),
                    len,
                })
            }
            // RFC-0032 D4 (M-750): `Bytes` — a nullary repr keyword (no descriptor).
            Tok::Bytes => {
                self.bump();
                Ok(BaseType::Bytes)
            }
            // ADR-040 (M-897): `Float` — a nullary repr keyword like `Bytes` (binary64 only at
            // introduction, FLAG-1; a later width extends the surface append-only).
            Tok::Float => {
                self.bump();
                Ok(BaseType::Float)
            }
            Tok::Ident(s) => {
                self.bump();
                let args = self.parse_type_args_opt()?;
                Ok(BaseType::Named(s, args))
            }
            // A paradigm-less repr `{ … }` (RFC-0012 §4.2): the paradigm is supplied later by the
            // enclosing ambient; only the size/shape is written here. The shape (single size vs
            // Dense `{N, scalar}` vs VSA `{model, dim, sparsity}`) is disambiguated by lookahead;
            // whether it *fits* the ambient paradigm is the resolution pass's never-silent check.
            Tok::LBrace => self.parse_ambient_repr().map(BaseType::Ambient),
            // M-826: `(T, U, …)` is a tuple type (arity ≥ 2); a single `(T)` is grouping.
            // A single-element parenthesized type `(T)` stays a bare type (grouping only).
            Tok::LParen => {
                self.bump(); // consume `(`
                let first = self.parse_type_ref()?;
                if self.eat(&Tok::Comma) {
                    let mut elems = vec![first];
                    while !self.at(&Tok::RParen) {
                        elems.push(self.parse_type_ref()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` to close the tuple type")?;
                    if elems.len() < 2 {
                        return Err(ParseError::new(
                            self.pos(),
                            "a tuple type requires arity ≥ 2; a single-element parenthesized \
                             type `(T)` is grouping, not a 1-tuple (M-826)"
                                .to_owned(),
                        ));
                    }
                    Ok(BaseType::Tuple(elems))
                } else {
                    // Single-element — grouping; return the inner type's base unchanged.
                    self.expect(&Tok::RParen, "`)` to close the parenthesized type")?;
                    Ok(first.base)
                }
            }
            _ => self.err("a type"),
        }
    }

    /// Parse a paradigm-less repr's params `{ … }` into [`AmbientParams`] (RFC-0012 §4.2). The
    /// leading token disambiguates: an `Int` opens a size (`{N}`) or a Dense shape (`{N, scalar}`);
    /// an `Ident` opens a VSA shape (`{model, dim, sparsity}`).
    fn parse_ambient_repr(&mut self) -> Result<AmbientParams, ParseError> {
        self.expect(&Tok::LBrace, "`{` to open the paradigm-less repr")?;
        let params = match self.cur() {
            Tok::Int(_) => {
                let n = self.u32_lit()?;
                if self.eat(&Tok::Comma) {
                    let scalar = self.parse_scalar()?;
                    AmbientParams::Dense(n, scalar)
                } else {
                    AmbientParams::Size(n)
                }
            }
            Tok::Ident(_) => {
                let model = self.ident()?;
                self.expect(&Tok::Comma, "`,` after the VSA model")?;
                let dim = self.u32_lit()?;
                self.expect(&Tok::Comma, "`,` before the sparsity")?;
                let sparsity = self.parse_sparsity()?;
                AmbientParams::Vsa {
                    model,
                    dim,
                    sparsity,
                }
            }
            _ => {
                return self
                    .err("a paradigm-less repr param (a size `{N}`, `{N, scalar}`, or VSA shape)")
            }
        };
        self.expect(&Tok::RBrace, "`}` to close the paradigm-less repr")?;
        Ok(params)
    }

    fn braced_u32(&mut self) -> Result<u32, ParseError> {
        self.expect(&Tok::LBrace, "`{` and a width")?;
        let n = self.u32_lit()?;
        self.expect(&Tok::RBrace, "`}` to close the width")?;
        Ok(n)
    }

    /// `'{'` `(u32_lit | Ident)` `'}'` — parses the width slot of `Binary{…}` or `Ternary{…}`:
    /// either a concrete literal (`Binary{8}`) or a width-parameter name (`Binary{N}` — DN-42 /
    /// M-753 v1). Dense/VSA/Seq still use [`Self::braced_u32`] (literal-only; DN-42 §6).
    fn braced_width(&mut self) -> Result<crate::ast::WidthRef, ParseError> {
        self.expect(&Tok::LBrace, "`{` and a width")?;
        let wr = match self.cur().clone() {
            Tok::Ident(s) => {
                self.bump();
                crate::ast::WidthRef::Name(s)
            }
            _ => {
                let v = self.u32_lit()?;
                crate::ast::WidthRef::Lit(v)
            }
        };
        self.expect(&Tok::RBrace, "`}` to close the width")?;
        Ok(wr)
    }

    fn parse_type_args_opt(&mut self) -> Result<Vec<TypeRef>, ParseError> {
        let mut args = Vec::new();
        if self.eat(&Tok::LBracket) {
            args = self.comma_separated(None, Self::parse_type_ref)?;
            self.expect(&Tok::RBracket, "`]` to close the type arguments")?;
        }
        Ok(args)
    }

    fn parse_sparsity(&mut self) -> Result<Sparsity, ParseError> {
        match self.cur() {
            Tok::Dense => {
                self.bump();
                Ok(Sparsity::Dense)
            }
            Tok::Sparse => {
                self.bump();
                let k = self.braced_u32()?;
                Ok(Sparsity::Sparse(k))
            }
            _ => self.err("a sparsity (`Dense` or `Sparse{…}`)"),
        }
    }

    fn parse_scalar(&mut self) -> Result<Scalar, ParseError> {
        match *self.cur() {
            Tok::Scalar(s) => {
                self.bump();
                Ok(match s {
                    ScalarTok::F16 => Scalar::F16,
                    ScalarTok::Bf16 => Scalar::Bf16,
                    ScalarTok::F32 => Scalar::F32,
                    ScalarTok::F64 => Scalar::F64,
                })
            }
            _ => self.err("a scalar kind (`F16|BF16|F32|F64`)"),
        }
    }

    fn parse_strength(&mut self) -> Result<Strength, ParseError> {
        match *self.cur() {
            Tok::Strength(s) => {
                self.bump();
                Ok(match s {
                    StrengthTok::Exact => Strength::Exact,
                    StrengthTok::Proven => Strength::Proven,
                    StrengthTok::Empirical => Strength::Empirical,
                    StrengthTok::Declared => Strength::Declared,
                })
            }
            _ => self.err("a guarantee strength (`Exact|Proven|Empirical|Declared`)"),
        }
    }

    // ---- expressions ----

    /// Depth-guarded entry to the expression grammar: refuses to recurse past [`MAX_EXPR_DEPTH`]
    /// with an explicit error rather than overflowing the host stack on crafted nesting (A4-02).
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.enter_depth()?;
        let r = self.parse_expr_inner();
        self.leave_depth();
        r
    }

    fn parse_expr_inner(&mut self) -> Result<Expr, ParseError> {
        self.teach_imperative()?;
        // M-666 / RFC-0008 §4.7: a bare `hypha <expr>` at expression position is only valid *inside*
        // a `colony { … }` block (RT7 — an orphan hypha is not expressible); `parse_colony` consumes
        // the `hypha` keywords in the block body, so reaching one here means it is unscoped. Explicit
        // teaching diagnostic, never a silent accept (G2).
        if self.at(&Tok::Hypha) {
            return Err(ParseError::new(
                self.pos(),
                "`hypha <expr>` spawns a concurrent task and is only valid inside a `colony { … }` \
                 block (RFC-0008 §4.7 RT7 — an orphan hypha is not expressible)"
                    .to_owned(),
            ));
        }
        // DN-03 §4 / RFC-0008 §4.5: the remaining runtime-vocabulary reserved words produce a
        // teaching diagnostic at expression position (never a silent accept, G2). (`hypha`/`colony`
        // left the reserved set with M-666; `fuse`/`reclaim`/`tier` left with M-667 / DN-58.)
        if let t @ (Tok::Mesh | Tok::Graft | Tok::Cyst | Tok::Xloc | Tok::Forage | Tok::Backbone) =
            self.cur()
        {
            return Err(ParseError::new(
                self.pos(),
                format!(
                    "`{word}` is reserved for the runtime model (RFC-0008), not yet active — \
                     it cannot open a program or be used as an identifier at this language version",
                    word = runtime_keyword_spelling(t)
                ),
            ));
        }
        // DN-58 §C (M-667): `tier` at expression position — teaching diagnostic.
        if self.at(&Tok::Tier) {
            return Err(ParseError::new(
                self.pos(),
                "`tier` is the DN-58 §C execution-mode hint — write `@tier(compiled)` or \
                 `@tier(interpreted)` before a `fn` declaration at item position; it cannot appear \
                 in expression context (never-silent — G2)"
                    .to_owned(),
            ));
        }
        // DN-38 §8.1 / M-812: `grow` is superseded by `derive` — teaching diagnostic points to
        // `derive Name for T` at item position (never a silent accept, G2).
        if self.at(&Tok::Grow) {
            return Err(ParseError::new(
                self.pos(),
                "`grow` is superseded by `derive` (DN-38 §8.1 / M-812) — write `derive Name for T` \
                 at item position to apply a generative-lowering rule, or `lower Name[params] = <rhs>` \
                 to define one; `grow` cannot be used as an identifier (G2)"
                    .to_owned(),
            ));
        }
        match self.cur() {
            Tok::Let => self.parse_let(),
            Tok::If => self.parse_if(),
            Tok::Match => self.parse_match(),
            Tok::For => self.parse_for(),
            Tok::Swap => self.parse_swap(),
            Tok::With => self.parse_with_paradigm(),
            Tok::Wild => self.parse_wild(),
            Tok::Spore => self.parse_spore(),
            // DN-03 §1 / M-664: `consume <expr>` — affine acquisition of a `Substrate` value (LR-8).
            Tok::Consume => self.parse_consume_expr(),
            Tok::Colony => self.parse_colony(),
            // RFC-0037 D5: an anonymous-function expression (parses; semantics deferred to M-704).
            Tok::Lambda => self.parse_lambda(),
            // DN-53 / M-811: `object` is active at **item** position (top-level declarations) but is
            // NOT a valid expression form — it cannot appear in value/expression contexts (G2).
            Tok::Object => Err(ParseError::new(
                self.pos(),
                "`object` opens a composition-surface declaration (DN-53/M-811) — it is valid only \
                 at item position (top level of a nodule), not in expression context; \
                 if you want a value of an object type, construct it with the named constructor"
                    .to_owned(),
            )),
            // DN-54 / M-812: `lower` and `derive` are top-level declaration forms — they cannot
            // appear at expression position (teaching, never silent, G2).
            Tok::Lower => Err(ParseError::new(
                self.pos(),
                "`lower Name[params] = <rhs>` is a top-level declaration (DN-54 §3 / M-812), not \
                 an expression — move it to item position (outside any `fn` body)"
                    .to_owned(),
            )),
            Tok::Derive => Err(ParseError::new(
                self.pos(),
                "`derive Name for T` is a top-level declaration (DN-54 / M-812), not an expression \
                 — move it to item position (outside any `fn` body)"
                    .to_owned(),
            )),
            // DN-58 §A (M-667): `fuse(a, b)` — lawful binary merge over the `Fuse` semilattice
            // instance (RFC-0008 RT6). Both operands must have the same type; the checker verifies
            // the type homogeneity and fusibility. Never-silent on type mismatch (G2).
            Tok::Fuse => self.parse_fuse_expr(),
            // DN-58 §B (M-667): `reclaim(policy) { body }` — attach a reified supervision/
            // reclamation policy to a structured scope (RFC-0008 RT7). The checker verifies the
            // body is well-typed; the policy type is open in v0 (FLAG F-B2). Never-silent (G2).
            Tok::Reclaim => self.parse_reclaim_expr(),
            // RFC-0025 / M-705: the infix-operator layer. A non-keyword expression is an operator
            // expression over unary/applicative operands; each operator desugars to its canonical
            // word function. The keyword-led forms above (let/if/match/…) are statements, not
            // operands; to use one as an operand, parenthesize it.
            _ => self.parse_binexpr(0),
        }
    }

    /// Precedence-climbing parser for infix operator expressions (RFC-0025 / M-705). Each binary
    /// operator desugars to its canonical word function (`a + b` → `add(a, b)`); the word form
    /// remains valid everywhere the sugar is (the sugar is *additive* — words stay canonical, the
    /// kernel is unchanged: this is a frontend-only desugaring, no L0/L1 change, KC-3). `min_bp`
    /// is the minimum binding power this call consumes; left-associativity is encoded by recursing
    /// on the right operand at `bp + 1` so an equal-precedence operator is left for the loop.
    ///
    /// **Stack-safety (A4-02).** This needs no extra depth charge of its own and must not add one
    /// (the enclosing [`parse_expr`](Self::parse_expr) already charges the budget for this
    /// expression level — charging again would double-count and halve the effective nesting limit).
    /// The RHS recursion `parse_binexpr(bp + 1)` strictly *raises* `min_bp` each step, so its
    /// recursion depth is bounded by the fixed number of precedence tiers (a small constant),
    /// independent of input length — it cannot overflow. A flat left-associative chain
    /// `a + a + …` is consumed by the loop (not recursion), so it too stays O(1) deep. The only
    /// genuinely unbounded operator vector is the prefix chain in [`parse_unary`](Self::parse_unary),
    /// which carries its own depth guard. Nested parens route back through `parse_expr` (guarded).
    fn parse_binexpr(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        while let Some((bp, word)) = infix_op(self.cur()) {
            if bp < min_bp {
                break;
            }
            self.bump(); // the operator token
            let rhs = self.parse_binexpr(bp + 1)?;
            lhs = op_call(word, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    /// Prefix unary operators (RFC-0025 / M-705): `-a` → `neg(a)`, `!a` → `not(a)`. Unary binds
    /// tighter than every binary operator and is right-associative (`- - a` → `neg(neg(a))`). A
    /// `!` here is always the unary operator: the effect-set `!{…}` only ever appears in a fn
    /// signature (parsed elsewhere), never at expression position. Any other token delegates to
    /// the applicative layer ([`parse_app`]).
    ///
    /// The prefix recursion participates in the shared [`MAX_EXPR_DEPTH`] budget (A4-02): a crafted
    /// prefix chain (`!!!!…a`, `----a`) is refused with an explicit error past the limit, never a
    /// host-stack overflow (G2 — never crash).
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let word = match self.cur() {
            Tok::Minus => "neg",
            Tok::Bang => "not",
            _ => return self.parse_app(),
        };
        self.enter_depth()?;
        self.bump(); // the prefix operator
        let operand = self.parse_unary();
        self.leave_depth();
        operand.map(|o| op_call(word, vec![o]))
    }

    /// Teaching diagnostic (RFC-0007 §4.8): `while`/`loop`/`break`/`continue`/`return` are not
    /// forms — and juxtaposition (`while cond …`) was never valid syntax anyway, so when one of
    /// these *unreserved* identifiers is immediately followed by an expression opener or `{`,
    /// the (inevitable) error teaches instead of confusing. Any other use stays an ordinary
    /// identifier.
    fn teach_imperative(&mut self) -> Result<(), ParseError> {
        let Tok::Ident(word) = self.cur() else {
            return Ok(());
        };
        if !matches!(
            word.as_str(),
            "while" | "loop" | "break" | "continue" | "return"
        ) {
            return Ok(());
        }
        let word = word.clone();
        let next = &self.toks[(self.i + 1).min(self.toks.len() - 1)].tok;
        let juxtaposed = matches!(
            next,
            Tok::Ident(_)
                | Tok::BinLit(_)
                | Tok::TritLit(_)
                | Tok::Int(_)
                | Tok::LBrace
                | Tok::If
                | Tok::Let
                | Tok::Match
                | Tok::For
                | Tok::Swap
        );
        if juxtaposed {
            return Err(ParseError::new(
                self.pos(),
                format!(
                    "`{word}` is not a Mycelium form — iterate by recursion or `for x in xs, \
                     acc = init => body` (bounded, total by construction; RFC-0007 §4.8)"
                ),
            ));
        }
        Ok(())
    }

    /// `for x in xs, acc = init => body` (RFC-0007 §4.8; spelling adopted at r3).
    fn parse_for(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::For)?;
        let x = self.ident()?;
        self.expect(&Tok::In, "`in` after the element binder")?;
        let xs = Box::new(self.parse_app()?);
        self.expect(&Tok::Comma, "`,` before the accumulator binding")?;
        let acc = self.ident()?;
        self.expect(&Tok::Eq, "`=` and the initial accumulator")?;
        let init = Box::new(self.parse_app()?);
        self.expect(&Tok::FatArrow, "`=>` and the fold body")?;
        let body = Box::new(self.parse_expr()?);
        Ok(Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        })
    }

    fn parse_let(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Let)?;
        let name = self.ident()?;
        let ty = if self.eat(&Tok::Colon) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        self.expect(&Tok::Eq, "`=` in the let binding")?;
        let bound = Box::new(self.parse_expr()?);
        self.expect(&Tok::In, "`in` after the let binding")?;
        let body = Box::new(self.parse_expr()?);
        Ok(Expr::Let {
            name,
            ty,
            bound,
            body,
        })
    }

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::If)?;
        let cond = Box::new(self.parse_expr()?);
        self.expect_keyword(&Tok::Then)?;
        let conseq = Box::new(self.parse_expr()?);
        self.expect_keyword(&Tok::Else)?;
        let alt = Box::new(self.parse_expr()?);
        Ok(Expr::If { cond, conseq, alt })
    }

    fn parse_match(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Match)?;
        let scrutinee = Box::new(self.parse_expr()?);
        self.expect(&Tok::LBrace, "`{` to open the match arms")?;
        // Non-empty (≥ 1 arm), with a trailing comma before `}` tolerated.
        let arms = self.comma_separated(Some(&Tok::RBrace), Self::parse_arm)?;
        self.expect(
            &Tok::RBrace,
            "`}` to close the match (or `,` and another arm)",
        )?;
        Ok(Expr::Match { scrutinee, arms })
    }

    fn parse_arm(&mut self) -> Result<Arm, ParseError> {
        // Parse the first (and possibly only) alternative pattern.
        let first = self.parse_pattern()?;
        // Or-pattern (RFC-0020 §9 / R20-Q3): if `|` follows, parse additional alternatives
        // (`A | B | C => body`). In pattern position `|` is unambiguously the or-separator
        // (not the bitwise-or expression operator `bor`, which is lexed at the same `Tok::Pipe`
        // but can only appear after a full expression, never immediately after a pattern before
        // `=>`). If there are multiple alternatives, they are gathered into `Pattern::Or`, which
        // the checker desugars into repeated arms sharing the same body (KC-3 — zero kernel
        // growth). Never-silent (G2): a `|` in a pattern with no following alternative is an
        // explicit `ParseError`.
        let pattern = if self.at(&Tok::Pipe) {
            let mut alts = vec![first];
            while self.eat(&Tok::Pipe) {
                alts.push(self.parse_pattern()?);
            }
            Pattern::Or(alts)
        } else {
            first
        };
        self.expect(&Tok::FatArrow, "`=>` in the match arm")?;
        let body = self.parse_expr()?;
        Ok(Arm { pattern, body })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        // Depth-guarded (A4-02 / DN-40 A1): a nested constructor pattern `C(C(C(…)))` recurses
        // through `comma_separated(Self::parse_pattern)` below, so it charges the shared budget —
        // deep pattern nesting is an explicit error, never a host-stack overflow (G2).
        self.enter_depth()?;
        let r = self.parse_pattern_guarded();
        self.leave_depth();
        r
    }

    fn parse_pattern_guarded(&mut self) -> Result<Pattern, ParseError> {
        match self.cur().clone() {
            Tok::Ident(s) if s == "_" => {
                self.bump();
                Ok(Pattern::Wildcard)
            }
            Tok::Ident(s) => {
                self.bump();
                if self.eat(&Tok::LParen) {
                    let subs = self.comma_separated(None, Self::parse_pattern)?;
                    self.expect(&Tok::RParen, "`)` to close the constructor pattern")?;
                    Ok(Pattern::Ctor(s, subs))
                } else {
                    Ok(Pattern::Ident(s))
                }
            }
            Tok::BinLit(_)
            | Tok::TritLit(_)
            | Tok::BytesLit(_)
            | Tok::StrLit(_)
            | Tok::FloatLit(_)
            | Tok::Int(_)
            | Tok::LBracket => Ok(Pattern::Lit(self.parse_literal()?)),
            // M-826: `(x, y, …)` is a tuple pattern (arity ≥ 2). A single `(_)` is grouping.
            Tok::LParen => {
                self.bump();
                let first = self.parse_pattern()?;
                if self.eat(&Tok::Comma) {
                    let mut subs = vec![first];
                    while !self.at(&Tok::RParen) {
                        subs.push(self.parse_pattern()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` to close the tuple pattern")?;
                    if subs.len() < 2 {
                        return Err(ParseError::new(
                            self.pos(),
                            "a tuple pattern requires arity ≥ 2 (M-826)".to_owned(),
                        ));
                    }
                    Ok(Pattern::Tuple(subs))
                } else {
                    // Single-element — grouping; unwrap the inner pattern.
                    self.expect(&Tok::RParen, "`)` to close the parenthesized pattern")?;
                    Ok(first)
                }
            }
            _ => self.err("a pattern"),
        }
    }

    fn parse_swap(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Swap)?;
        self.expect(&Tok::LParen, "`(` after `swap`")?;
        let value = Box::new(self.parse_expr()?);
        self.expect(&Tok::Comma, "`,` before the `to:` target")?;
        self.expect(&Tok::To, "the `to:` target label")?;
        self.expect(&Tok::Colon, "`:` after `to`")?;
        let target = self.parse_type_ref()?;
        self.expect(
            &Tok::Comma,
            "`,` before the `policy:` (a swap is never silent — S1)",
        )?;
        self.expect(&Tok::Policy, "the `policy:` label (mandatory — WF2)")?;
        self.expect(&Tok::Colon, "`:` after `policy`")?;
        let policy = self.parse_path()?;
        self.expect(&Tok::RParen, "`)` to close the swap")?;
        Ok(Expr::Swap {
            value,
            target,
            policy,
        })
    }

    /// `with paradigm P { e }` — a block-scope ambient override (RFC-0012 §4.4). Not a conversion:
    /// the resolution pass fills the interior tags and strips the block; an unbridged cross-paradigm
    /// edge is a never-silent `MissingConversion`.
    fn parse_with_paradigm(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::With)?;
        self.expect(&Tok::Paradigm, "`paradigm` after `with` (RFC-0012 §4.4)")?;
        let paradigm = self.parse_paradigm()?;
        self.expect(&Tok::LBrace, "`{` to open the `with paradigm` block")?;
        let body = Box::new(self.parse_expr()?);
        self.expect(&Tok::RBrace, "`}` to close the `with paradigm` block")?;
        Ok(Expr::WithParadigm { paradigm, body })
    }

    fn parse_wild(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Wild)?;
        self.expect(&Tok::LBrace, "`{` to open the wild block")?;
        let body = Box::new(self.parse_expr()?);
        self.expect(&Tok::RBrace, "`}` to close the wild block")?;
        Ok(Expr::Wild(body))
    }

    fn parse_spore(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Spore)?;
        self.expect(&Tok::LParen, "`(` after `spore`")?;
        let value = Box::new(self.parse_expr()?);
        self.expect(&Tok::RParen, "`)` to close `spore(…)`")?;
        Ok(Expr::Spore(value))
    }

    /// `consume <expr>` (DN-03 §1; LR-8; M-664) — affine acquisition of a `Substrate` value. A
    /// prefix form (like the unary `neg`/`not`): the operand is an applicative expression
    /// (`consume s`, `consume f(x)`), depth-guarded so a crafted nesting is an explicit error, never
    /// a host-stack overflow (G2). The operand-type check (`Substrate{tag}`) is the checker's job.
    fn parse_consume_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Consume)?;
        self.enter_depth()?;
        // `consume_expr ::= 'consume' app_expr` (mycelium.ebnf): the operand is an **applicative**
        // expression, matching the doc above and the `for`/`hypha` prefix-operand siblings
        // (`parse_app`). This is narrower than the prior `parse_unary` (which also admitted `neg`/`not`
        // operands like `consume -s`); the grammar does not include those, so `parse_app` is the
        // grammar-faithful production — never-more-permissive (no G2 break).
        let operand = self.parse_app();
        self.leave_depth();
        operand.map(|o| Expr::Consume(Box::new(o)))
    }

    /// `colony { hypha e1, hypha e2, … }` — the structured-concurrency scope (RFC-0008 §4.7; DN-06
    /// §1.3). The block body is a **non-empty** comma-separated list of `hypha <expr>` spawns; an
    /// empty `colony { }` is an explicit error (a colony with no hyphae is meaningless — RT7 names a
    /// *grouping of active hyphae*). Each `hypha` keyword opens one spawn whose body is an
    /// application/expression over immutable values; a trailing comma before `}` is tolerated (the
    /// `match`-arm convention). Deterministic R1 fragment only (RFC-0008 §4.6 R1) — the
    /// arbitration/placement RT3 constructs are separate, later work.
    fn parse_colony(&mut self) -> Result<Expr, ParseError> {
        self.expect_keyword(&Tok::Colony)?;
        self.expect(&Tok::LBrace, "`{` to open the `colony` block")?;
        // Non-empty (≥ 1 hypha), trailing comma before `}` tolerated.
        let hyphae = self.comma_separated(Some(&Tok::RBrace), Self::parse_hypha)?;
        self.expect(
            &Tok::RBrace,
            "`}` to close the `colony` (or `,` and another `hypha`)",
        )?;
        Ok(Expr::Colony(hyphae))
    }

    /// One `hypha <expr>` spawn inside a [`parse_colony`] body, with an optional leading
    /// `@forage(policy) ` placement-policy annotation (RFC-0008 RT3; DN-63 §3.5; D-lite, M-906/
    /// DN-70 D1) — the same `@` **attribute-before-construct** shape as `@tier(mode) fn …`
    /// (DN-58 §C), applied to `hypha`. The `hypha` keyword is mandatory (RT7: every concurrent
    /// unit is named — a bare body would be ambiguous with a value), and its computation is parsed
    /// as an `app_expr` (a call like `compute(x)` — the issue's canonical form), matching `for`'s
    /// use of `app_expr` for its bounded sub-expressions.
    ///
    /// `policy` is parsed as a general expression (mirroring `reclaim(policy) { body }`'s open
    /// policy — DN-58 §B); the D-lite **narrowing** to a literal binary bitmask is a *checker*
    /// rule ([`crate::checkty::Cx::check_forage_policy`]), not a grammar restriction, so a
    /// non-literal policy still parses and gets a precise, checker-level teaching diagnostic
    /// rather than an opaque parse error.
    fn parse_hypha(&mut self) -> Result<Hypha, ParseError> {
        let forage = if self.at(&Tok::At) {
            self.bump(); // `@`
            self.expect_keyword(&Tok::Forage)?;
            self.expect(&Tok::LParen, "`(` to open the `@forage` policy argument")?;
            let policy = self.parse_expr()?;
            self.expect(&Tok::RParen, "`)` to close the `@forage` policy argument")?;
            Some(Box::new(policy))
        } else {
            None
        };
        self.expect(
            &Tok::Hypha,
            "`hypha` to open a concurrent task in the `colony` block (RFC-0008 §4.7)",
        )?;
        let body = self.parse_app()?;
        Ok(Hypha { forage, body })
    }

    fn parse_app(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.parse_primary()?;
        while self.eat(&Tok::LParen) {
            let args = self.parse_args_opt()?;
            self.expect(&Tok::RParen, "`)` to close the call")?;
            e = Expr::App {
                head: Box::new(e),
                args,
            };
        }
        if self.eat(&Tok::Colon) {
            let ty = self.parse_type_ref()?;
            e = Expr::Ascribe(Box::new(e), ty);
        }
        Ok(e)
    }

    fn parse_args_opt(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.comma_separated_until(&Tok::RParen, Self::parse_expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.cur() {
            Tok::BinLit(_)
            | Tok::TritLit(_)
            | Tok::BytesLit(_)
            | Tok::StrLit(_)
            | Tok::FloatLit(_)
            | Tok::Int(_)
            | Tok::LBracket => Ok(Expr::Lit(self.parse_literal()?)),
            Tok::Ident(_) => Ok(Expr::Path(self.parse_path()?)),
            Tok::LParen => {
                // M-826: `(e, e2, …)` is a tuple literal (arity ≥ 2); `(e)` is grouping.
                self.bump();
                let first = self.parse_expr()?;
                if self.eat(&Tok::Comma) {
                    // At least two elements — a tuple literal.
                    let mut elems = vec![first];
                    // parse remaining elements (trailing comma before `)` is tolerated)
                    while !self.at(&Tok::RParen) {
                        elems.push(self.parse_expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` to close the tuple literal")?;
                    if elems.len() < 2 {
                        return Err(ParseError::new(
                            self.pos(),
                            "a tuple literal requires arity ≥ 2; a single-element parenthesized \
                             expression `(e)` is grouping, not a 1-tuple (M-826 — FLAG: unit `()` \
                             and 1-tuples are deferred surface decisions)"
                                .to_owned(),
                        ));
                    }
                    Ok(Expr::TupleLit(elems))
                } else {
                    // Single element — grouping, not a tuple.
                    self.expect(&Tok::RParen, "`)` to close the parenthesized expression")?;
                    Ok(first)
                }
            }
            _ => self.err("an expression"),
        }
    }

    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        match self.cur().clone() {
            Tok::BinLit(s) => {
                self.bump();
                Ok(Literal::Bin(s))
            }
            Tok::TritLit(s) => {
                self.bump();
                Ok(Literal::Trit(s))
            }
            // RFC-0032 D4 (M-750): `0x…` byte-string literal; the lexer already validated even-hex
            // parity / non-empty, so the inner string is stored verbatim.
            Tok::BytesLit(s) => {
                self.bump();
                Ok(Literal::Bytes(s))
            }
            // M-910/M-911: `"…"` textual string literal; the lexer already decoded its escape set
            // and validated termination, so the content is stored verbatim.
            Tok::StrLit(s) => {
                self.bump();
                Ok(Literal::Str(s))
            }
            // ADR-040 (M-897): a decimal float literal; the lexer already validated the form
            // (digits `.` digits and/or an exponent with digits) and binary64 finiteness, so the
            // source text is stored verbatim (the single text→f64 conversion is elaboration's).
            Tok::FloatLit(s) => {
                self.bump();
                Ok(Literal::Float(s))
            }
            Tok::Int(n) => {
                self.bump();
                Ok(Literal::Int(n))
            }
            Tok::LBracket => {
                self.bump();
                let elems = self.comma_separated_until(&Tok::RBracket, Self::parse_expr)?;
                self.expect(&Tok::RBracket, "`]` to close the list literal")?;
                Ok(Literal::List(elems))
            }
            _ => self.err("a literal"),
        }
    }

    fn parse_path(&mut self) -> Result<Path, ParseError> {
        let mut segs = vec![self.ident()?];
        while self.eat(&Tok::Dot) {
            segs.push(self.ident()?);
        }
        Ok(Path(segs))
    }
}

/// The infix binding power and canonical word function for an operator token (RFC-0025 / M-705),
/// or `None` if the token does not open an infix operator. Higher binding power binds tighter;
/// every binary operator is left-associative. The precedence tiers follow **Rust's** table (the
/// implementation language, syntactically adjacent; RFC-0025 §4.1). The angle-bracket/shift
/// operators (`<`, `>`, `<<`, `>>`) are **wired** (M-745, resolved by RFC-0037 D1's `[…]`
/// type-argument kind-split, which frees `<>` for operators-only use — no contextual lexing
/// needed; see `Tok::LAngle`/`RAngle`/`Shl`/`Shr` below). `<=`/`>=` have **no glyph at all** —
/// they are retired (RFC-0037 D1); their word-canonical forms `lte`/`gte` are ordinary calls, not
/// entries in this table (M-916 verified this inventory; no residual glyph wiring remained). The
/// desugaring is purely syntactic: a word target whose prim/stdlib function does not yet exist
/// (`div`, `rem`, `band`, `bor`, `ne`, `and`, `or`, `gt`, `shl`, `shr`) still desugars here and
/// surfaces an explicit "unknown function/prim" refusal downstream (never silent — G2); today
/// `add`/`sub`/`mul`/`xor`/`eq`/`lt` (and unary `neg`/`not`) resolve end-to-end.
fn infix_op(tok: &Tok) -> Option<(u8, &'static str)> {
    Some(match tok {
        Tok::Star => (70, "mul"),
        Tok::Slash => (70, "div"),
        Tok::Percent => (70, "rem"),
        Tok::Plus => (60, "add"),
        Tok::Minus => (60, "sub"),
        // Shift (RFC-0025 §4.1 Tier 4; M-745): binds tighter than the bitwise ops and looser than
        // `+`/`-` — the bp slot (55) reserved between `add` (60) and `band` (50). This follows the
        // ratified §4.1 table (Rust's precedence, the cited source of truth: shift is tighter than
        // `& ^ |`), NOT RFC-0037 §6's *illustrative* sketch, which nested shift adjacent to
        // comparison (looser than `|`) — an internal inconsistency flagged for the spec.
        Tok::Shl => (55, "shl"),
        Tok::Shr => (55, "shr"),
        Tok::Amp => (50, "band"),
        Tok::Caret => (40, "xor"),
        Tok::Pipe => (30, "bor"),
        // Comparison (RFC-0025 §4.1 Tier 8; M-745): looser than the bitwise ops, tighter than
        // equality — the bp slot (25) reserved between `bor` (30) and `eq` (20). `<=`/`>=` have no
        // glyph (retired by RFC-0037 D1); their word forms `lte`/`gte` are ordinary calls.
        Tok::LAngle => (25, "lt"),
        Tok::RAngle => (25, "gt"),
        Tok::EqEq => (20, "eq"),
        Tok::BangEq => (20, "ne"),
        Tok::AmpAmp => (11, "and"),
        Tok::PipePipe => (10, "or"),
        _ => return None,
    })
}

/// Build the canonical word-function application an operator desugars to (RFC-0025 / M-705). The
/// sugar leaves **no separate trace**: the desugared `App` node *is* the audit record — the
/// canonical word form is the inspectable EXPLAIN (ADR-006, no black boxes), so `a + b` and
/// `add(a, b)` are structurally identical after parsing (this resolves RFC-0025 Q5 — no separate
/// `DesugarRecord` is needed; the desugaring target is the record).
fn op_call(word: &str, args: Vec<Expr>) -> Expr {
    Expr::App {
        head: Box::new(Expr::Path(Path(vec![word.to_owned()]))),
        args,
    }
}

/// The first value appearing more than once in `xs` (left to right), if any. Used by the effect
/// annotation parser to reject a duplicate effect name explicitly (M-660; G2 — never a silent
/// dedup). A small, allocation-light scan (effect sets are short); mirrors the checker's
/// `first_duplicate` without coupling the two modules.
fn first_duplicate_str(xs: &[String]) -> Option<&String> {
    let mut seen = std::collections::BTreeSet::new();
    xs.iter().find(|x| !seen.insert((*x).as_str()))
}

/// Return the surface spelling for a DN-03 §4 runtime-vocabulary reserved keyword token.
/// Used in teaching diagnostics so the error message names the actual word, not the enum variant.
/// Total over exactly the runtime-vocabulary tokens; the `_` arm is unreachable in practice
/// (callers only pass one of the ten runtime-vocab arms) but keeps this panic-free (G2).
fn runtime_keyword_spelling(tok: &Tok) -> &'static str {
    match tok {
        Tok::Hypha => "hypha",
        Tok::Fuse => "fuse",
        Tok::Mesh => "mesh",
        Tok::Graft => "graft",
        Tok::Cyst => "cyst",
        Tok::Xloc => "xloc",
        Tok::Forage => "forage",
        Tok::Backbone => "backbone",
        Tok::Tier => "tier",
        Tok::Reclaim => "reclaim",
        _ => "<runtime-keyword>",
    }
}
