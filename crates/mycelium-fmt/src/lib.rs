//! `mycelium-fmt` — **`mycfmt`**, the canonical formatter (M-364; M-142 grows up).
//!
//! Formatting is a **projection**: it rewrites a `.myc` source into one canonical textual normal form
//! and **never changes a definition's content-addressed identity** (RFC-0001 §4.6/§4.8; ADR-003). The
//! contract is `docs/spec/Mycfmt-Formatter-Contract.md`; this crate enacts it. Three invariants hold,
//! the first by a **runtime guard** (so an identity-changing format is never emitted) and all three by
//! test (`tests/`):
//!
//! - **C1 identity-preservation.** The formatted text re-parses to the **same surface AST** as the input
//!   (`parse(out) == parse(src)`), and its header re-parses equal (`parse_header`). Equivalent to
//!   content-hash preservation on the elaborable fragment (ADR-003). Checked at runtime: a mismatch is a
//!   refusal ([`FmtError::OutOfScope`]), never an emitted rewrite.
//! - **C2 idempotence.** `format(format(s)) == format(s)` byte-for-byte (the canonical form is a fixed
//!   point). Tested.
//! - **C3 header-preservation.** The DN-06 `// nodule:` marker + the M-359 `// @key:` structured header
//!   are re-emitted canonically (§4 order); a malformed header is an explicit [`FmtError::Header`], never
//!   a silent drop (G2/VR-5).
//!
//! **Never-silent (G2).** Unparsable input, a malformed header, or a construct outside the round-trip-safe
//! v0 scope (§7 — expressions that do not round-trip) is an **explicit error** with an exit code;
//! `mycfmt` **never** writes a partial or garbled rewrite.  Interior comments — previously refused in v0
//! — are now **preserved** by interleaving from the lexer comment table (M-690; Stage 2): leading
//! comment blocks above items are re-emitted verbatim; trailing `// …` comments on a fn body line or a
//! match arm are re-attached at the end of the rendered line.  A genuinely unplaceable comment is still
//! a [`FmtError::OutOfScope`] refusal — never a silent drop.
//! The load-bearing subtlety: the body is printed from the **raw parse** (`mycelium_l1::parse`), *not*
//! the ambient-resolved twin — so `default paradigm` / `with paradigm` are **preserved**, not expanded
//! (formatting ≠ "expand ambient").
//!
//! KC-3: this lives entirely above the kernel; the trusted base depends on nothing here.
//!
//! ## Comment placement (M-690, Stage 2 — Empirical)
//!
//! Comments are captured by `mycelium_l1::lexer::lex_with_comments` and anchored to items by
//! **source line number** matching against the token stream's `Spanned` positions:
//!
//! - **Leading doc-block**: one or more consecutive `// …` lines (not trailing) immediately before an
//!   item's first token are re-emitted verbatim above that item's canonical rendering.  A stray non-`@key`
//!   comment in the header region is now a leading doc-block on the first item (no longer refused).
//! - **Trailing fn-body comment**: a `// …` on the same source line as the `fn` keyword (i.e. the whole
//!   fn fits on one source line) is re-attached after the body expression on the rendered body line
//!   (`  body  // comment`).
//! - **Trailing match-arm comment**: a `// …` on the same source line as a `=>` token switches that
//!   match to **one-arm-per-line** rendering and places the comment after the arm's canonical expression.
//!   Non-nested matches only; a deeply nested match with arm comments is a [`FmtError::OutOfScope`] (not
//!   a silent drop — see FLAG below).
//!
//! **FLAG (anchoring sufficiency):** Token-position anchoring is sufficient for ALL tested placement
//! cases: leading doc-blocks, trailing fn-body comments, and trailing arm comments on non-nested
//! matches.  **Nested match arm trailing comments** (a `=>` line inside an outer match arm's body) are
//! out-of-scope: the `FatArrow` positions interleave in source order, making it impossible to assign
//! a comment to the correct arm purely from token positions (without AST line numbers from `ast.rs`
//! `Arm`).  Adding line numbers to `ast.rs`/`Arm` collides with the parallel HOF track (M-689
//! stage-1 sibling) and is serialized — flagged up rather than silently worked around.

use mycelium_l1::ast::{Expr, FnDecl, Hypha, ImplDecl, Item, Nodule, Pattern};
use mycelium_l1::lexer::{lex_with_comments, Comment};
use mycelium_l1::token::{Spanned, Tok};
use mycelium_l1::{expand_to_source, parse, parse_phylum};
use mycelium_proj::{parse_header, Deprecated, StructuredHeader};
use mycelium_workstack::{ensure_sufficient_stack, BudgetError, ProcessArena, RecursionBudget};
use std::collections::HashMap;

/// The formatter spelling/version this build implements. The `[toolchain].format` pin (M-359) is a
/// **hard pin** (M-364 §10.3): a mismatch is refused, never formatted with rules the project didn't ask
/// for (G2).
pub const MYCFMT_VERSION: &str = "mycfmt-0";

/// RFC-0041 §4.2/§9 process-wide arena ceiling `mycfmt`'s render family reserves against before
/// rendering. **Declared** — an asserted operational default, not a measured bound: a `.myc` source
/// is untrusted input (a vendored/spore-resolved dependency, or a CI run over an untrusted PR diff —
/// RFC-0041 §5 untrusted-input coverage), and every concurrent format call (batch `mycfmt --check`
/// over a project, or a future LSP format-on-save) charges the *same* process-global counter
/// ([`mycelium_workstack::current_process_bytes`]), so the sum across concurrent renders is what
/// this ceiling bounds. Each consuming crate declares its own operational default (consumer-side
/// wiring); a single shared constant across the workspace is a follow-up centralization item, not
/// introduced silently here (`docs/notes/W7-arena-coverage-audit.md`).
const PROCESS_ARENA_CEILING_BYTES: u64 = 256 * 1024 * 1024;

/// A conservative, **Declared** (not measured) multiplier on the input source length, used only to
/// *size the pre-render reservation* — never to bound the actual output. The readable layout can
/// expand a compact source somewhat (re-indentation, wrapped segments, re-emitted comments); this
/// multiplier is generous relative to observed expansion so the reservation stays a conservative
/// upper bound in practice, without being `Proven`.
const RENDER_BYTES_PER_SRC_BYTE: u64 = 4;

/// The default target line width for the **readable** layout style (M-974; retuned M-976). Purely a
/// *presentation* heuristic: a construct whose compact single-line rendering, placed at its indent
/// column, would exceed this width is broken across lines (line breaks after commas / `|`); a shorter
/// construct stays inline. This is `Declared` — a readability threshold, not a proven bound. It is
/// **functionally inert**: the readable output re-parses to the *same* surface AST as the compact
/// output (the C1/C2 guards enforce it), so the width choice never changes any parse/elaborate/eval
/// behavior — machines ingest the flattened stream / full file either way.
///
/// **Why 100 (M-976).** The earlier 88 was Black's Python default — an arbitrary import with a
/// misleading association for a value-semantics systems language. 100 is **`rustfmt`'s `max_width`
/// default** — the formatter the Mycelium Rust kernel itself already uses — so the threshold is
/// *grounded in the project's own toolchain*, not borrowed. It is overridable per call via
/// [`LayoutCfg::width`]; the shipped default is the single threshold (R0).
const READABLE_WIDTH: usize = 100;

/// The layout style a format pass emits (M-974). Both styles are **identity-preserving projections**
/// (RFC-0001 §4.6/§4.8) — they differ only in *presentation*, never in the surface AST (C1). The
/// distinction is the inverse posture of `--flatten`: `Compact` keeps each item's body inline (the
/// original mycfmt-0 canonical form — the machine-leaning default retained for `examples/` and any
/// generic `.myc` root); `Readable` breaks long argument/field/variant segments across lines with
/// line breaks after commas for human-authored source (the `lib/std/*.myc` canonical — DN-82).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Style {
    /// The original mycfmt-0 canonical form: item bodies rendered inline (one expression per line).
    #[default]
    Compact,
    /// The human-readable multi-line form (M-974/DN-82): long segments wrap after commas / `|`,
    /// nested structure indents, short segments stay inline (the [`READABLE_WIDTH`] heuristic).
    Readable,
}

/// How a right-nested same-head chain (Cons/GLCons/TCons/bool_and/cat …) lays out its per-link
/// **inner** (non-recursive / leading) arguments under the Shape-Dispatched Readable rules (M-976 /
/// DN-82). BOTH styles keep the spine **flat** — every link at ONE fixed indent, the terminal on
/// its own line, all closers coalesced (R1/R5) — so both kill the deepening-Cons pyramid; they
/// differ ONLY in whether a fitting inner call stays inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpineInner {
    /// **Compact (default).** A leading argument that fits under [`LayoutCfg::width`] at its column
    /// stays inline (the confirmed-good canonical rendering of `matrix()`/`guarantee_matrix()`); an
    /// overflowing leading arg blocks per R2, spine indent unchanged.
    #[default]
    InlineWhenFits,
    /// **Expanded house style.** The spine STILL stays flat (each link at one indent, no pyramid),
    /// but every inner nested call is broken onto its own lines (block-indented per R2) even when it
    /// would fit. Both styles are behavior-neutral (C1/C2); this one trades density for
    /// per-argument column comparability.
    AlwaysExpand,
}

/// Presentation tunables for the readable layout (M-976 / DN-82). All fields are **presentation-only**
/// and identity-preserving (C1/C2) — a `LayoutCfg` never changes the surface AST, only whitespace.
///
/// `width` defaults to [`READABLE_WIDTH`] (100 — rustfmt-aligned `Declared` threshold); it is the
/// SOLE inline-vs-break trigger (R0). `spine_inner` selects the house style for same-head chains
/// (R1). [`LayoutCfg::default`] is the compact default the `lib/std` canonical form uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutCfg {
    /// The single break threshold (R0). Chars, not bytes, measured at a node's own indent.
    pub width: usize,
    /// Same-head-chain inner-argument layout (R1): compact (default) vs expanded house style.
    pub spine_inner: SpineInner,
}

impl Default for LayoutCfg {
    fn default() -> Self {
        Self {
            width: READABLE_WIDTH,
            spine_inner: SpineInner::InlineWhenFits,
        }
    }
}

/// A successful format result.
///
/// `Default` is the empty result (no output, unchanged, no notes) — an additive constructor
/// convenience (M-644); the canonical "ends with one newline" output is produced by
/// [`format_source`], not by `Default`/`From`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Formatted {
    /// The output text. For a value produced by [`format_source`] this is the **canonical** form and
    /// always ends with exactly one newline; the additive `Default` / `From<String>` constructors do
    /// **not** normalize (M-644), so a hand-built `Formatted` may not carry that invariant.
    pub output: String,
    /// Whether the output differs from the input (drives `--check`).
    pub changed: bool,
    /// The normalizations applied, named for `EXPLAIN` (no black box).
    pub notes: Vec<String>,
}

impl From<String> for Formatted {
    /// Lift a raw output string into a [`Formatted`] (M-644 ergonomics): `changed` is `false` and
    /// `notes` is empty — a trivial wrapper for callers/tests that already hold canonical text.
    /// `format_source` is the path that computes `changed`/`notes`; this does not.
    fn from(output: String) -> Self {
        Self {
            output,
            changed: false,
            notes: Vec::new(),
        }
    }
}

/// A formatting refusal — never a partial rewrite (G2). Each maps to a CLI exit code.
///
/// `#[non_exhaustive]`: new refusal kinds may be added without a breaking change — an external
/// exhaustive `match` must carry a `_` arm (M-644; additive — no variant removed).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FmtError {
    /// Input is not a valid `.myc` program (exit 2).
    Parse(String),
    /// A malformed `// nodule:` / `// @key:` header (exit 3).
    Header(String),
    /// A construct outside the round-trip-safe v0 scope, or a `[toolchain].format` pin mismatch (exit 4).
    OutOfScope(String),
    /// **RFC-0041 §4.2/§9 (W7 process-arena coverage).** The pre-render reservation against the
    /// shared process-wide memory arena would exceed its ceiling (exit 5) — refused rather than let
    /// this render's memory join an unbounded concurrent sum with other in-flight passes (batch
    /// `mycfmt` runs, a future LSP format-on-save). Never a partial rewrite (G2), same as every other
    /// `FmtError` variant.
    OutOfBudget(BudgetError),
}

impl FmtError {
    /// The CLI exit code for this refusal (contract §5).
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self {
            FmtError::Parse(_) => 2,
            FmtError::Header(_) => 3,
            FmtError::OutOfScope(_) => 4,
            FmtError::OutOfBudget(_) => 5,
        }
    }
}

impl std::fmt::Display for FmtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FmtError::Parse(m) => write!(f, "parse-error: {m}"),
            FmtError::Header(m) => write!(f, "header-error: {m}"),
            FmtError::OutOfScope(m) => write!(f, "refused: {m}"),
            FmtError::OutOfBudget(e) => {
                write!(f, "refused (RFC-0041 §4.2 process-arena ceiling): {e}")
            }
        }
    }
}

impl std::error::Error for FmtError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FmtError::OutOfBudget(e) => Some(e),
            FmtError::Parse(_) | FmtError::Header(_) | FmtError::OutOfScope(_) => None,
        }
    }
}

/// Does `src` open with a `phylum` header — i.e. is the first significant line (after leading blank or
/// `//`-comment lines) the reserved `phylum` keyword at a word boundary? Lets mycfmt refuse a *malformed*
/// phylum (one `parse_phylum` rejects, e.g. a `phylum` header with no `nodule`) as `OutOfScope` rather
/// than a parse error, so a phylum source is never a parse error in v0 (M-662; G2). `phylum` is a
/// reserved keyword (never an identifier), so a leading `phylum` token unambiguously opens a phylum.
fn opens_with_phylum(src: &str) -> bool {
    src.lines()
        .map(str::trim_start)
        .find(|l| !l.is_empty() && !l.starts_with("//"))
        .and_then(|l| l.strip_prefix("phylum"))
        .is_some_and(|rest| {
            rest.is_empty() || !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        })
}

/// Format `src` into its canonical form.
///
/// `pin` is the optional `[toolchain].format` value from `mycelium-proj.toml` (a **hard pin**: a value
/// other than [`MYCFMT_VERSION`] is refused).
///
/// # Errors
/// [`FmtError::Parse`] (unparsable), [`FmtError::Header`] (malformed header), or [`FmtError::OutOfScope`]
/// (a pin mismatch, an unplaceable comment, or a body that does not round-trip — identity could change).
/// On any error nothing is rewritten (G2).
pub fn format_source(src: &str, pin: Option<&str>) -> Result<Formatted, FmtError> {
    format_source_styled(src, pin, Style::Compact)
}

/// Format `src` into its **human-readable** canonical form (M-974/DN-82): the inverse posture of
/// `--flatten`. Long argument / field / variant / arm segments break across lines with line breaks
/// after commas (and `|` for sum-type constructors); nested structure indents; short segments stay
/// inline (the [`READABLE_WIDTH`] readability heuristic). This is the canonical form the `myc-fmt`
/// gate enforces for the **human-authored stdlib** (`lib/std/*.myc`); `Style::Compact`
/// ([`format_source`]) remains the default for `examples/` and any generic `.myc` root, and
/// [`flatten_source`] stays the explicit machine/stream form.
///
/// It is **presentation-only and functionally inert**: the output re-parses to the *same* surface AST
/// as the compact form (the same C1/C2 identity + idempotence guards apply), so it never changes any
/// parse/elaborate/eval behavior — machines ingest the flattened stream / full file either way.
///
/// # Errors
/// Identical to [`format_source`]: [`FmtError::Parse`], [`FmtError::Header`], or
/// [`FmtError::OutOfScope`] (pin mismatch, unplaceable comment, or a body that does not round-trip).
/// On any error nothing is rewritten (G2).
pub fn format_source_readable(src: &str, pin: Option<&str>) -> Result<Formatted, FmtError> {
    format_source_readable_cfg(src, pin, LayoutCfg::default())
}

/// Format `src` into its human-readable canonical form with an explicit [`LayoutCfg`] (M-976/DN-82):
/// same as [`format_source_readable`] but the caller picks the Shape-Dispatched Readable house-style
/// knob (`spine_inner`: compact `InlineWhenFits` default vs the `AlwaysExpand` house style) and the
/// break width. Both configurations are **presentation-only and functionally inert** — the output
/// re-parses to the same surface AST (the same C1/C2 guards apply), so the knob never changes any
/// parse/elaborate/eval behavior.
///
/// # Errors
/// Identical to [`format_source_readable`].
pub fn format_source_readable_cfg(
    src: &str,
    pin: Option<&str>,
    cfg: LayoutCfg,
) -> Result<Formatted, FmtError> {
    format_source_styled_cfg(src, pin, Style::Readable, cfg)
}

/// Shared implementation of [`format_source`] (Compact) and [`format_source_readable`] (Readable).
/// The `style` parameter selects only the *layout* of item bodies — every scope/identity/header
/// guard is style-independent, so both styles emit the same surface AST (C1) from the same input.
///
/// # Errors
/// See [`format_source`].
pub fn format_source_styled(
    src: &str,
    pin: Option<&str>,
    style: Style,
) -> Result<Formatted, FmtError> {
    format_source_styled_cfg(src, pin, style, LayoutCfg::default())
}

/// Shared implementation with an explicit [`LayoutCfg`] (M-976). The `cfg` selects only *presentation*
/// (the same-head-chain house style + break width); every scope/identity/header guard is
/// style- and cfg-independent, so all configurations emit the same surface AST (C1) from the same
/// input. For `Style::Compact` the `cfg` is inert (the compact path renders item bodies inline).
///
/// **RFC-0041 §4.2/§9 W7 (process-arena coverage).** Delegates to
/// [`format_source_styled_cfg_with_arena`] with the crate's declared default ceiling
/// ([`PROCESS_ARENA_CEILING_BYTES`]) — see that function for the reservation.
///
/// # Errors
/// See [`format_source`], plus [`FmtError::OutOfBudget`] if the pre-render reservation would exceed
/// the process-wide arena ceiling.
pub fn format_source_styled_cfg(
    src: &str,
    pin: Option<&str>,
    style: Style,
    cfg: LayoutCfg,
) -> Result<Formatted, FmtError> {
    let arena = ProcessArena::new(PROCESS_ARENA_CEILING_BYTES);
    format_source_styled_cfg_with_arena(src, pin, style, cfg, &arena)
}

/// [`format_source_styled_cfg`], parameterized over an explicit [`ProcessArena`] (`pub(crate)`, so
/// this crate's own tests can inject a tiny-ceiling arena and witness a real refusal — production
/// callers always go through [`format_source_styled_cfg`], which supplies the crate's declared
/// default ceiling).
///
/// # Errors
/// See [`format_source_styled_cfg`].
pub(crate) fn format_source_styled_cfg_with_arena(
    src: &str,
    pin: Option<&str>,
    style: Style,
    cfg: LayoutCfg,
    arena: &ProcessArena,
) -> Result<Formatted, FmtError> {
    // Hard pin (M-364 §10.3): never format with rules the project did not pin.
    if let Some(p) = pin {
        if p != MYCFMT_VERSION {
            return Err(FmtError::OutOfScope(format!(
                "[toolchain].format = {p:?}, but this is {MYCFMT_VERSION} — refusing to format with rules \
                 the project did not pin (hard pin; G2). Align the pin or use the matching mycfmt."
            )));
        }
    }

    // Phylum / multi-nodule sources are outside mycfmt v0 scope (M-662): the formatter v0 canonicalizes a
    // SINGLE nodule. A `phylum` header or multiple `nodule` blocks is an explicit out-of-scope refusal —
    // never a parse error and never a partial rewrite (G2). A WELL-FORMED phylum (header, or >1 nodule) is
    // caught via `parse_phylum`; a MALFORMED one (a `phylum` header `parse_phylum` rejects — e.g. no
    // `nodule`) is caught by its opening keyword, so a phylum source NEVER surfaces as a parse error. A
    // header-less single nodule is a phylum-of-one and formats normally below.
    let is_phylum = parse_phylum(src).is_ok_and(|ph| ph.path.is_some() || ph.nodules.len() > 1)
        || opens_with_phylum(src);
    if is_phylum {
        return Err(FmtError::OutOfScope(
            "phylum / multi-nodule sources are outside mycfmt v0 scope (M-662) — format each nodule's \
             content individually; whole-phylum canonical formatting is future work (refused, never a \
             partial rewrite — G2)"
                .to_string(),
        ));
    }

    // The header (M-358/M-359): a malformed marker/key is explicit, never a silent drop (C3/G2).
    let header = parse_header(src).map_err(|e| FmtError::Header(e.to_string()))?;
    // The body: the RAW parse — preserves `default paradigm`/`with paradigm` (formatting ≠ expand-ambient).
    let nodule = parse(src).map_err(|e| FmtError::Parse(e.to_string()))?;

    let lines: Vec<&str> = src.lines().collect();
    let body_start = body_start_line(&lines);

    // Lex with comments (M-690, Stage 2): capture every `//` comment for interleaving.
    // This is the entry point for the comment-preservation path — replaces the old refusal.
    let (tokens, comments) = lex_with_comments(src).map_err(|e| FmtError::Parse(e.to_string()))?;

    // Build a CommentPlan: classify every comment as leading (attached to an item) or trailing
    // (attached to a fn body line or a match arm).
    let has_structured_header = header.is_some();
    let plan = build_comment_plan(
        &nodule,
        &tokens,
        &comments,
        &lines,
        body_start,
        has_structured_header,
    )?;

    let mut out = String::new();
    let mut notes = Vec::new();

    match &header {
        Some(h) => {
            // Re-emit the structured header canonically.  Any stray non-`@key` comment in the header
            // region is now a leading doc-block on the first item (plan.first_item_stray_comments),
            // so we no longer refuse it.
            out.push_str(&render_header(h));
            notes.push(
                "re-emitted the structured header (// nodule: + // @key:) in canonical order"
                    .to_owned(),
            );
            // Stray header comments — preserved as leading doc-block on the first item (plan handles them).
            if !plan.stray_header_comments.is_empty() {
                notes.push(format!(
                    "preserved {} stray header comment(s) as a leading doc-block on the first item",
                    plan.stray_header_comments.len()
                ));
            }
        }
        None => {
            let leading = leading_comment_block(&lines, body_start);
            if !leading.is_empty() {
                out.push_str(&leading);
                notes.push("preserved the leading comment block".to_owned());
            }
        }
    }

    // RFC-0041 §4.2/§9 (W7): reserve against the shared process-wide arena before the render-family
    // call below — `src` is untrusted input (§5), so this bounds the *concurrent sum* of in-flight
    // renders' estimated cost, not just this one call's. Held for the render's duration; released on
    // drop at the end of this function.
    let estimate = (src.len() as u64).saturating_mul(RENDER_BYTES_PER_SRC_BYTE);
    let _arena_reservation = arena.reserve(estimate).map_err(FmtError::OutOfBudget)?;

    // Render the body: items with their leading/trailing comments interleaved.
    let (body, body_notes) = render_body_with_comments(&nodule, &plan, style, cfg)?;
    out.push_str(&body);
    notes.extend(body_notes);
    notes.push("re-printed the body in canonical surface form".to_owned());

    // Exactly one trailing newline.
    while out.ends_with("\n\n") {
        out.pop();
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }

    // C1 identity guard: the output must re-parse to the SAME surface AST, and the header must survive.
    // A mismatch is a refusal — mycfmt never emits an identity-changing format (round-trip-safe scope, §7).
    let reparsed = parse(&out).map_err(|e| {
        FmtError::OutOfScope(format!(
            "the formatted output did not re-parse ({e}) — refusing (round-trip-safe scope; C1/§7)"
        ))
    })?;
    if reparsed != nodule {
        return Err(FmtError::OutOfScope(
            "formatting would change the program's surface AST — identity not preserved; refusing \
             (round-trip-safe scope; C1/§7). This construct is outside mycfmt v0."
                .to_owned(),
        ));
    }
    let reheader = parse_header(&out).map_err(|e| FmtError::Header(e.to_string()))?;
    if reheader != header {
        return Err(FmtError::OutOfScope(
            "formatting would change the structured header — refusing (C3)".to_owned(),
        ));
    }

    let changed = out != src;
    Ok(Formatted {
        output: out,
        changed,
        notes,
    })
}

/// Flatten `src` into the single-line human↔stream form (M-819; DN-57 §2).
///
/// The mandatory `;` component terminator (M-818) makes the stream form unambiguous: every
/// component ends with `;`, so components can be separated by a single space and the whole
/// nodule fits on one line without any whitespace carrying structural meaning.
///
/// **What this produces:** a single output line (plus the required final `\n`) where the nodule
/// header and every top-level item are emitted in their canonical inline form, joined by `; `
/// separators.  Example:
/// ```text
/// nodule signals.demo; use core.binary; fn f(x: Binary{8}) => Binary{8} = x;
/// ```
///
/// **Round-trip guarantee (`Empirical`):** `parse(flatten(src)) == parse(format(src))` — the
/// flattened output re-parses to the same surface AST as the canonically formatted input.  This
/// is verified by a runtime identity check (C1, same guard as `format_source`) and by the
/// data-driven corpus test in `src/tests.rs`.  The guarantee is `Empirical`: backed by the
/// corpus tests, not a formal proof.
///
/// **Comments and structured header:** the flatten form is the machine/stream form; it does not
/// preserve `//` comments or `// @key:` structured-header metadata.  Both are stripped — they
/// are not part of the Mycelium surface AST and cannot appear mid-line without swallowing
/// subsequent content.  A caller that needs comment preservation must use `format_source`.
///
/// **Scope:** the same v0 scope as `format_source` — single-nodule sources only; a phylum /
/// multi-nodule source is an explicit `OutOfScope` refusal, never a partial rewrite (G2).
///
/// **RFC-0041 §4.2/§9 W7 (process-arena coverage).** Delegates to
/// [`flatten_source_with_arena`] with the crate's declared default ceiling
/// ([`PROCESS_ARENA_CEILING_BYTES`]) — see that function for the reservation.
///
/// # Errors
/// [`FmtError::Parse`] (unparsable), [`FmtError::Header`] (malformed `// nodule:` / `// @key:`
/// header — structurally invalid, not just metadata), [`FmtError::OutOfScope`] (a pin
/// mismatch, a phylum/multi-nodule source, or a body that does not round-trip), or
/// [`FmtError::OutOfBudget`] (the pre-render reservation would exceed the process-wide arena
/// ceiling).  On any error nothing is written (G2).
pub fn flatten_source(src: &str, pin: Option<&str>) -> Result<Formatted, FmtError> {
    let arena = ProcessArena::new(PROCESS_ARENA_CEILING_BYTES);
    flatten_source_with_arena(src, pin, &arena)
}

/// [`flatten_source`], parameterized over an explicit [`ProcessArena`] (`pub(crate)`, so this
/// crate's own tests can inject a tiny-ceiling arena and witness a real refusal — production
/// callers always go through [`flatten_source`], which supplies the crate's declared default
/// ceiling).
///
/// # Errors
/// See [`flatten_source`].
pub(crate) fn flatten_source_with_arena(
    src: &str,
    pin: Option<&str>,
    arena: &ProcessArena,
) -> Result<Formatted, FmtError> {
    // Hard pin: same guard as format_source.
    if let Some(p) = pin {
        if p != MYCFMT_VERSION {
            return Err(FmtError::OutOfScope(format!(
                "[toolchain].format = {p:?}, but this is {MYCFMT_VERSION} — refusing to format with rules \
                 the project did not pin (hard pin; G2). Align the pin or use the matching mycfmt."
            )));
        }
    }

    // Phylum / multi-nodule: same guard as format_source.
    let is_phylum = parse_phylum(src).is_ok_and(|ph| ph.path.is_some() || ph.nodules.len() > 1)
        || opens_with_phylum(src);
    if is_phylum {
        return Err(FmtError::OutOfScope(
            "phylum / multi-nodule sources are outside mycfmt v0 scope (M-662) — format each nodule's \
             content individually; whole-phylum canonical formatting is future work (refused, never a \
             partial rewrite — G2)"
                .to_string(),
        ));
    }

    // A malformed structured header is an explicit error (C3/G2) even in flatten mode —
    // the header being structurally invalid is a content problem, not a metadata-only issue.
    parse_header(src).map_err(|e| FmtError::Header(e.to_string()))?;

    // Parse the nodule body (raw parse — preserves `default paradigm`/`with paradigm`).
    let nodule = parse(src).map_err(|e| FmtError::Parse(e.to_string()))?;

    // RFC-0041 §4.2/§9 (W7): reserve against the shared process-wide arena before the render below —
    // see `format_source_styled_cfg_with_arena`'s identical comment.
    let estimate = (src.len() as u64).saturating_mul(RENDER_BYTES_PER_SRC_BYTE);
    let _arena_reservation = arena.reserve(estimate).map_err(FmtError::OutOfBudget)?;

    // Render the flat form directly from AST (no comments, no multiline layout).
    let out = render_flat(&nodule);

    // C1 identity guard: the flattened output must re-parse to the SAME surface AST.
    // This is the core round-trip guarantee (Empirical — backed by corpus tests).
    let reparsed = parse(&out).map_err(|e| {
        FmtError::OutOfScope(format!(
            "the flattened output did not re-parse ({e}) — refusing (round-trip-safe scope; C1/§7)"
        ))
    })?;
    if reparsed != nodule {
        return Err(FmtError::OutOfScope(
            "flattening would change the program's surface AST — identity not preserved; refusing \
             (round-trip-safe scope; C1/§7). This construct is outside mycfmt v0."
                .to_owned(),
        ));
    }

    let changed = out != src;
    Ok(Formatted {
        output: out,
        changed,
        notes: vec![
            "emitted the single-line stream form (--flatten; M-819/DN-57 §2)".to_owned(),
            "structured header and comments stripped (not part of the Mycelium surface AST)"
                .to_owned(),
        ],
    })
}

/// Render `nodule` as a single-line flat stream: `nodule path; item1; item2; …\n`.
///
/// This is the layout policy for `--flatten`; it reuses the existing item renderers but
/// collapses all whitespace so the whole program fits on one line.  No comments are emitted.
///
/// **RFC-0041 §4.7 (RR-29) guard-hole closure (W1).** This is [`flatten_source`]'s only render-family
/// entry, so it wraps once, here, in [`ensure_sufficient_stack`] — every callee it reaches
/// (`render_item_flat`, `render_impl_flat`, and transitively `render_expr_canonical`, each already
/// self-guarded per item/method too) runs on the grown 256 MiB worker stack rather than the caller's.
/// One worker-thread spawn per `flatten_source` call, not per recursion level (`nodule.items` is a
/// flat loop, not itself recursive). Output is unchanged — only the stack it runs on changed.
fn render_flat(nodule: &mycelium_l1::ast::Nodule) -> String {
    let budget = RecursionBudget::with_depth_default(u64::MAX, u64::MAX);
    ensure_sufficient_stack(&budget, || render_flat_inner(nodule))
}

/// The actual body of [`render_flat`] — see that function's doc comment for the guard-hole wrap.
fn render_flat_inner(nodule: &mycelium_l1::ast::Nodule) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Nodule header component (terminates with `;` per DN-57 §3 / M-818).
    parts.push(format!(
        "nodule {}{}",
        nodule.path.0.join("."),
        if nodule.std_sys { " @std-sys" } else { "" }
    ));

    // Each top-level item — rendered inline (no indentation, no newlines).
    for item in &nodule.items {
        parts.push(render_item_flat(item));
    }

    // Join with `; ` — each part already carries its trailing `;` except the last part
    // of each item (which is included in the part string).  Actually each part already
    // ends with `;` because the item renderers include the terminator.
    // The nodule header part does NOT yet carry `;` — it is added when we join with `; `.
    //
    // Strategy: the nodule header is just "nodule path" (no `;`), and each item part
    // already ends with `;`.  Join all parts with a single space and add the final `\n`.
    //
    // nodule path; item1; item2;   ← each item ends with `;`, nodule header gets `;` from join
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            // nodule header: append `; ` if there are more parts, else `;\n`
            out.push_str(part);
            if parts.len() == 1 {
                out.push_str(";\n");
            } else {
                out.push_str("; ");
            }
        } else {
            // item part already ends with `;`
            let trimmed = part.trim_end();
            out.push_str(trimmed);
            if i + 1 < parts.len() {
                out.push(' ');
            } else {
                out.push('\n');
            }
        }
    }

    out
}

/// Render a single top-level item as an inline flat string (no newlines, no indentation).
/// Each returned string ends with `;` (the mandatory component terminator, DN-57 §3 / M-818).
fn render_item_flat(item: &mycelium_l1::ast::Item) -> String {
    use mycelium_l1::ast::Item;
    match item {
        Item::Fn(fd) => {
            let pub_prefix = if fd.vis.is_pub() { "pub " } else { "" };
            let thaw_prefix = if fd.thaw { "thaw " } else { "" };
            let sig = render_sig_tail(&fd.sig);
            // Guarded entry (RFC-0041 §4.7 RR-29 / W1): each item's body is its own fresh
            // `render_expr_canonical` call, so a pathological single item is stack-safe on its own,
            // independent of `render_flat`'s outer wrap below.
            let body = render_expr_canonical(&fd.body);
            format!("{pub_prefix}{thaw_prefix}fn {sig} = {body};")
        }
        Item::Impl(id) => render_impl_flat(id),
        _ => {
            // For non-fn, non-impl items: reuse the canonical text from render_non_fn_item_compact,
            // then collapse it to a single line (it is already single-line for use/default/type
            // in canonical form, but we normalise whitespace for safety).
            let text = render_non_fn_item_compact(item);
            // The text ends with `\n`; trim and ensure the `;` terminator is present.
            let trimmed = text.trim_end();
            // Collapse any interior newlines (e.g. multiline type declarations) to spaces.
            let flat: String = trimmed
                .split('\n')
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            flat
        }
    }
}

/// Render an `impl` block in flat inline form: `impl Trait for Type { fn …; fn …; };`
fn render_impl_flat(id: &mycelium_l1::ast::ImplDecl) -> String {
    let args = if id.trait_args.is_empty() {
        String::new()
    } else {
        let a: Vec<String> = id.trait_args.iter().map(render_type_ref).collect();
        format!("[{}]", a.join(", "))
    };
    let methods: Vec<String> = id
        .methods
        .iter()
        .map(|m| {
            let sig = render_sig_tail(&m.sig);
            let pub_prefix = if m.vis.is_pub() { "pub " } else { "" };
            let thaw_prefix = if m.thaw { "thaw " } else { "" };
            // Guarded entry (RFC-0041 §4.7 RR-29 / W1) — see `render_item_flat`'s matching comment.
            let body = render_expr_canonical(&m.body);
            format!("{pub_prefix}{thaw_prefix}fn {sig} = {body};")
        })
        .collect();
    let body = if methods.is_empty() {
        String::new()
    } else {
        format!(" {} ", methods.join(" "))
    };
    format!(
        "impl {}{} for {} {{{body}}};",
        id.trait_name,
        args,
        render_type_ref(&id.for_ty)
    )
}

// ================================================================================================
// Comment plan: classify all comments relative to the items they belong to.
// ================================================================================================

/// A classified comment plan for one source file.
///
/// All indices are into the `nodule.items` Vec.
struct CommentPlan {
    /// Stray non-`@key` comments in the header region: they become a leading doc-block on item 0.
    stray_header_comments: Vec<String>,
    /// Leading comment blocks per item index: comments that appear directly above an item.
    /// Key = item index (0-based).
    leading: HashMap<usize, Vec<String>>,
    /// Trailing comment on the fn body expression line, per fn item index.
    /// Key = item index (0-based), value = comment text (the `// …` lexeme).
    fn_trailing: HashMap<usize, String>,
    /// Per-item, per-arm trailing comments: item_idx → arm_idx → comment text.
    /// Populated only for items whose match arms have trailing comments.
    arm_trailing: HashMap<usize, HashMap<usize, String>>,
}

/// Build the [`CommentPlan`] by cross-referencing the comment list and the token stream positions.
///
/// `has_structured_header` is `true` when the source has a `// nodule:` / `// @key:` header.
/// When `false`, pre-body comments are already emitted by `leading_comment_block` in
/// `format_source` and must NOT also be assigned as leading for item 0 (which would double-emit).
///
/// Guarantee: Empirical — anchoring uses token line numbers; the algorithm is validated by the
/// unit tests in this file.
fn build_comment_plan(
    nodule: &Nodule,
    tokens: &[Spanned],
    comments: &[Comment],
    lines: &[&str],
    body_start: usize,
    has_structured_header: bool,
) -> Result<CommentPlan, FmtError> {
    // Collect the source-order line numbers for every top-level item's first token.
    // We scan the token stream for the item-opening keywords (fn, type, trait, impl, use,
    // nodule/default), skipping the nodule header itself.
    let item_first_token_lines = item_first_lines(nodule, tokens);

    // Collect the source-order line numbers of every MATCH-ARM FatArrow token. Since RFC-0037 D4
    // the **return** arrow is also `=>` (FatArrow), but it sits in the fn signature at brace-depth 0
    // (before the return type); match-arm `=>` live inside a `match { … }` block (depth ≥ 1). So we
    // track `{`/`}` depth and count only FatArrows at depth ≥ 1 — the return arrow (depth 0) is
    // excluded, so a trailing comment on a signature line is correctly a fn-body comment, not an
    // orphaned arm comment (the regression RFC-0037's arrow-unification would otherwise cause).
    let fat_arrow_lines: Vec<u32> = {
        let mut arm_lines = Vec::new();
        let mut depth: i32 = 0;
        for s in tokens {
            match s.tok {
                Tok::LBrace => depth += 1,
                Tok::RBrace => depth -= 1,
                Tok::FatArrow if depth >= 1 => arm_lines.push(s.pos.line),
                _ => {}
            }
        }
        arm_lines
    };

    // -----------------------------------------------------------------------
    // Step 1: classify header-region comments.
    //
    // Only when a structured header is present: non-`@key` header comments are
    // stray → leading doc-block on the first item.
    // When there is NO structured header, pre-body comments are handled by
    // `leading_comment_block` in `format_source` — do NOT re-assign them here.
    // -----------------------------------------------------------------------
    let mut stray_header_comments: Vec<String> = Vec::new();
    if has_structured_header {
        let header_comment_lines: Vec<u32> = comments
            .iter()
            .filter(|c| !c.trailing && (c.line as usize) <= body_start)
            .map(|c| c.line)
            .collect();

        for &hline in &header_comment_lines {
            let line_text = lines
                .get((hline as usize).saturating_sub(1))
                .copied()
                .unwrap_or("");
            let trimmed = line_text.trim();
            // A structured header line starts with `// nodule:` or `// @`:
            let is_structured = trimmed.starts_with("// nodule:") || trimmed.starts_with("// @");
            if !is_structured {
                stray_header_comments.push(trimmed.to_owned());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: classify body-region comments.
    //
    // Body comments are those in lines > body_start.
    // A body comment is either:
    //   (a) trailing within a fn's line range → fn trailing comment
    //       (includes not only the fn keyword line but any line in the fn body,
    //       since the canonical render may move the comment to a different line)
    //   (b) trailing on a FatArrow line → match arm trailing comment
    //   (c) non-trailing → a leading comment for the next item
    //   (d) trailing on some other line → unplaceable → refuse (G2)
    // -----------------------------------------------------------------------

    // Build fn line ranges: for each Fn item, record (first_token_line, next_item_first_line).
    // A trailing comment anywhere in [first_token_line, next_item_first_line) that is NOT
    // on a FatArrow line is a fn-body trailing comment.
    //
    // This range-based approach handles idempotence: on the FIRST format pass, the
    // trailing comment may be on the same line as the `fn` keyword (whole fn on one line);
    // on the SECOND pass (formatting the already-formatted output), the fn body is split
    // to two lines so the comment is on the body line, not the `fn` line.
    let fn_line_ranges: Vec<(u32, u32, usize)> = {
        let mut ranges = Vec::new();
        for (idx, item) in nodule.items.iter().enumerate() {
            if matches!(item, Item::Fn(_)) {
                let first = item_first_token_lines.get(idx).copied().unwrap_or(0);
                // The range ends just before the next item's first line (or at u32::MAX for the last).
                let next = item_first_token_lines
                    .get(idx + 1)
                    .copied()
                    .unwrap_or(u32::MAX);
                ranges.push((first, next, idx));
            }
        }
        ranges
    };

    // For each trailing comment, classify it.
    let mut fn_trailing: HashMap<usize, String> = HashMap::new();
    let mut arm_trailing_flat: HashMap<u32, String> = HashMap::new(); // FatArrow line → comment text

    for comment in comments {
        // Skip header-region comments (already classified above, or handled by leading_comment_block).
        if (comment.line as usize) <= body_start {
            continue;
        }
        if !comment.trailing {
            // Non-trailing (leading) comments are handled in step 3.
            continue;
        }
        // Trailing comment in the body region.
        // First check FatArrow lines (match arm comments) — these take priority over fn range.
        if fat_arrow_lines.contains(&comment.line) {
            // Trailing comment on a match arm line.
            arm_trailing_flat.insert(comment.line, comment.text.clone());
        } else if let Some(&(_, _, item_idx)) = fn_line_ranges
            .iter()
            .find(|(first, next, _)| comment.line >= *first && comment.line < *next)
        {
            // Trailing comment within a fn's source line range → fn-body trailing comment.
            // Only record the first such comment per fn (the one closest to the body expression).
            fn_trailing
                .entry(item_idx)
                .or_insert_with(|| comment.text.clone());
        } else {
            // Trailing on an unknown line → unplaceable (G2: never silent drop).
            return Err(FmtError::OutOfScope(format!(
                "line {}: a trailing comment on this line cannot be placed by mycfmt v0 — \
                 the line is not within a `fn` declaration or a match arm (`=>`); refused, never \
                 silently dropped (G2; M-690 stage-2 scope)",
                comment.line
            )));
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: assign arm trailing comments to item and arm indices.
    //
    // We walk the nodule's items and, for each `Item::Fn`, walk the fn's body
    // looking for `Expr::Match` nodes.  For each match, we assign the arm trailing
    // comments by matching their FatArrow source lines to the arms in source order.
    //
    // FLAG: nested match arm trailing comments are out of scope — see module-level
    // FLAG comment.  A trailing comment on a nested match arm is an `OutOfScope`
    // refusal (never a silent drop — G2).
    // -----------------------------------------------------------------------
    let mut arm_trailing: HashMap<usize, HashMap<usize, String>> = HashMap::new();

    if !arm_trailing_flat.is_empty() {
        // For each item that is a Fn or Impl (which contains Fns), scan for matches.
        // Build a "remaining arrows" list in source order to assign to arms.
        let mut remaining_arrow_comments: Vec<(u32, String)> =
            arm_trailing_flat.into_iter().collect();
        remaining_arrow_comments.sort_by_key(|(line, _)| *line);

        for (item_idx, item) in nodule.items.iter().enumerate() {
            match item {
                Item::Fn(fd) => {
                    assign_arm_comments_for_fn(
                        item_idx,
                        &fd.body,
                        &mut remaining_arrow_comments,
                        &mut arm_trailing,
                        &fat_arrow_lines,
                    )?;
                }
                Item::Impl(id) => {
                    for method in &id.methods {
                        assign_arm_comments_for_fn(
                            item_idx,
                            &method.body,
                            &mut remaining_arrow_comments,
                            &mut arm_trailing,
                            &fat_arrow_lines,
                        )?;
                    }
                }
                _ => {}
            }
        }

        // Any remaining unassigned arm comments are unplaceable (G2).
        if let Some((line, _)) = remaining_arrow_comments.first() {
            return Err(FmtError::OutOfScope(format!(
                "line {line}: a trailing match-arm comment could not be assigned to an arm — \
                 possibly a nested match (out of scope for token-position anchoring); \
                 refused, never silently dropped (G2; M-690 FLAG)"
            )));
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: assign leading (non-trailing) body comments to items.
    // -----------------------------------------------------------------------
    let mut leading: HashMap<usize, Vec<String>> = HashMap::new();

    // Stray header comments become leading comments on item 0.
    if !stray_header_comments.is_empty() && !nodule.items.is_empty() {
        leading
            .entry(0)
            .or_default()
            .extend(stray_header_comments.clone());
    }

    // Non-trailing body-region comments: assign each to the first item whose first-token line
    // is > the comment's line (the item immediately following the comment block).
    let body_non_trailing: Vec<&Comment> = comments
        .iter()
        .filter(|c| !c.trailing && (c.line as usize) > body_start)
        .collect();

    for comment in body_non_trailing {
        // Find the item whose first-token line is closest to (and after) this comment's line.
        let target_item = item_first_token_lines
            .iter()
            .enumerate()
            .find(|(_, &first_line)| first_line > comment.line)
            .map(|(idx, _)| idx);

        match target_item {
            Some(item_idx) => {
                leading
                    .entry(item_idx)
                    .or_default()
                    .push(comment.text.clone());
            }
            None => {
                // Comment after the last item — attach it to the last item as a trailing
                // comment block. (We'll handle end-of-nodule comments as trailing on the last item.)
                if let Some(last_idx) = nodule.items.len().checked_sub(1) {
                    leading
                        .entry(last_idx + 1)
                        .or_default()
                        .push(comment.text.clone());
                }
            }
        }
    }

    Ok(CommentPlan {
        stray_header_comments,
        leading,
        fn_trailing,
        arm_trailing,
    })
}

/// Collect, for each item in `nodule.items`, the 1-based source line of its first token.
///
/// We walk the token stream looking for the Nth occurrence of item-opening tokens
/// (after the `nodule` header tokens are consumed).
fn item_first_lines(nodule: &Nodule, tokens: &[Spanned]) -> Vec<u32> {
    let mut result = Vec::with_capacity(nodule.items.len());

    // Skip past the nodule header tokens: `nodule`, the path segments and dots,
    // and the optional `@std-sys` marker.  The items begin after the last path token.
    // We find the `Tok::Nodule` token and skip until we reach the first item keyword.
    let body_tokens: Vec<&Spanned> = {
        let mut found_nodule = false;
        let mut after_header = Vec::new();
        for s in tokens {
            if !found_nodule {
                if s.tok == Tok::Nodule {
                    found_nodule = true;
                }
                continue;
            }
            // After the nodule token: skip the path (Ident, Dot) and optional @std-sys.
            // The first item keyword follows.
            if matches!(s.tok, Tok::Ident(_) | Tok::Dot | Tok::AtStdSys | Tok::Eof)
                && after_header.is_empty()
            {
                continue;
            }
            if s.tok == Tok::Eof {
                break;
            }
            after_header.push(s);
        }
        after_header
    };

    // Now walk the body tokens.  For each item, find the first token that opens it.
    // We match by counting item-opening keywords/tokens in source order.
    //
    // Items in the AST come in the same order as in the source.  We assign the first
    // token (by source order) that could open each item:
    //
    //   Item::Use    → Tok::Use
    //   Item::Default→ Tok::Default
    //   Item::Type   → Tok::Type  (or Tok::Pub + Tok::Type)
    //   Item::Trait  → Tok::Trait (or Tok::Pub + Tok::Trait)
    //   Item::Impl   → Tok::Impl
    //   Item::Fn     → Tok::Fn    (or Tok::Pub + Tok::Fn, or Tok::Thaw + …)
    //
    // We peek one token at a time and consume the first matching opener for each item.
    let mut tok_idx = 0;

    for item in &nodule.items {
        // Find the next opener for this item.
        while tok_idx < body_tokens.len() {
            let s = body_tokens[tok_idx];
            let is_opener = match item {
                Item::Use(_) => s.tok == Tok::Use,
                Item::Default(_) => s.tok == Tok::Default,
                Item::Type(_) => matches!(s.tok, Tok::Type | Tok::Pub),
                Item::Trait(_) => matches!(s.tok, Tok::Trait | Tok::Pub),
                Item::Impl(_) => s.tok == Tok::Impl,
                // M-664: an inherent method block also opens on `impl` (no `for`); the opener
                // keyword is identical to a trait-instance `impl`.
                Item::InherentImpl(_) => s.tok == Tok::Impl,
                Item::Fn(_) => matches!(s.tok, Tok::Fn | Tok::Pub | Tok::Thaw),
                // DN-53 / M-811: `object` opens at item position (optionally `pub object`).
                Item::Object(_) => matches!(s.tok, Tok::Object | Tok::Pub),
                // DN-54 / M-812: `lower`/`derive` open at item position.
                Item::Lower(_) => s.tok == Tok::Lower,
                Item::Derive(_) => s.tok == Tok::Derive,
            };
            if is_opener {
                result.push(s.pos.line);
                tok_idx += 1;
                break;
            }
            tok_idx += 1;
        }
        // If we ran out of tokens without finding an opener, push a sentinel.
        if result.len() < nodule.items.len()
            && result.len()
                == nodule
                    .items
                    .iter()
                    .position(|i| std::ptr::eq(i, item))
                    .map_or(0, |p| p)
        {
            // sentinel: line 0 means no match found
        }
    }

    // Pad if we found fewer lines than items (should not happen for well-formed input).
    while result.len() < nodule.items.len() {
        result.push(0);
    }

    result
}

/// Walk `expr`, find `Expr::Match` nodes (top-level, non-nested), and for each arm whose
/// source `FatArrow` line has a trailing comment, record it in `arm_trailing[item_idx][arm_idx]`.
///
/// This is intentionally limited to the TOP-LEVEL match in `expr`'s body (non-nested).
/// A trailing arm comment inside a nested match's arm body is out of scope (FLAG in module doc).
fn assign_arm_comments_for_fn(
    item_idx: usize,
    expr: &Expr,
    remaining: &mut Vec<(u32, String)>,
    arm_trailing: &mut HashMap<usize, HashMap<usize, String>>,
    fat_arrow_lines: &[u32],
) -> Result<(), FmtError> {
    // Walk to the top-level `match` (possibly wrapped in `let`, `if`, `with paradigm`).
    // We only handle a single level of match; nested matches are out of scope.
    collect_match_arm_comments(item_idx, expr, remaining, arm_trailing, fat_arrow_lines, 0)
}

/// Recursive walker for match arm comment assignment.
/// `depth` tracks how many `match` expressions we've entered; arm trailing comments in matches
/// at depth > 1 (nested) are out of scope.
fn collect_match_arm_comments(
    item_idx: usize,
    expr: &Expr,
    remaining: &mut Vec<(u32, String)>,
    arm_trailing: &mut HashMap<usize, HashMap<usize, String>>,
    fat_arrow_lines: &[u32],
    depth: u32,
) -> Result<(), FmtError> {
    match expr {
        Expr::Match { scrutinee, arms } => {
            // Find the FatArrow lines for these arms.
            // We consume `arms.len()` arrow lines from `remaining` that fall within the
            // expected range.  For depth=0 (top-level match), we consume in order.
            // For depth>0 (nested), we still try, but this is the out-of-scope case.
            for (arm_idx, _arm) in arms.iter().enumerate() {
                if remaining.is_empty() {
                    break;
                }
                let (arrow_line, comment_text) = remaining.remove(0);
                // Verify this is actually a FatArrow line.
                if !fat_arrow_lines.contains(&arrow_line) {
                    // Should not happen (we built remaining from fat_arrow_lines).
                    return Err(FmtError::OutOfScope(format!(
                        "line {arrow_line}: internal error assigning arm comment (G2 never-silent)"
                    )));
                }
                if depth > 0 {
                    // Nested match arm comment: out of scope (FLAG).
                    return Err(FmtError::OutOfScope(format!(
                        "line {arrow_line}: a trailing comment on a nested match arm is out of scope \
                         for token-position anchoring (M-690 FLAG; see module doc); refused, never \
                         silently dropped (G2)"
                    )));
                }
                arm_trailing
                    .entry(item_idx)
                    .or_default()
                    .insert(arm_idx, comment_text);
            }
            // Recurse into arm bodies to catch further nested matches.
            for arm in arms {
                collect_match_arm_comments(
                    item_idx,
                    &arm.body,
                    remaining,
                    arm_trailing,
                    fat_arrow_lines,
                    depth + 1,
                )?;
            }
            // Recurse into scrutinee.
            collect_match_arm_comments(
                item_idx,
                scrutinee,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        // Recurse through expression wrappers that may contain a match.
        Expr::Let { bound, body, .. } => {
            collect_match_arm_comments(
                item_idx,
                bound,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                body,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::If { cond, conseq, alt } => {
            collect_match_arm_comments(
                item_idx,
                cond,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                conseq,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                alt,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::For { xs, init, body, .. } => {
            collect_match_arm_comments(
                item_idx,
                xs,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                init,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                body,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::App { head, args } => {
            collect_match_arm_comments(
                item_idx,
                head,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            for a in args {
                collect_match_arm_comments(
                    item_idx,
                    a,
                    remaining,
                    arm_trailing,
                    fat_arrow_lines,
                    depth,
                )?;
            }
        }
        Expr::WithParadigm { body, .. } | Expr::Wild(body) | Expr::Spore(body) => {
            collect_match_arm_comments(
                item_idx,
                body,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        // `consume e` (DN-03 §1 surface keyword) wraps a single sub-expression — recurse through it
        // transparently to catch any nested match-arm comments.
        Expr::Consume(inner) => {
            collect_match_arm_comments(
                item_idx,
                inner,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::Swap { value, .. } => {
            collect_match_arm_comments(
                item_idx,
                value,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::Colony(hyphae) => {
            for h in hyphae {
                // M-970: recurse into the optional `@forage(policy)` expression too — it is a
                // plain expression (parsed by `parse_expr`, same as `reclaim`'s policy) and could
                // in principle carry a nested match with a trailing comment, so it must not be
                // skipped alongside the body. Destructuring `Hypha { forage, body }` (rather than
                // `h.forage`/`h.body` access) makes a future field on `Hypha` a compile error here
                // instead of a silent skip.
                let Hypha { forage, body } = h;
                if let Some(policy) = forage {
                    collect_match_arm_comments(
                        item_idx,
                        policy,
                        remaining,
                        arm_trailing,
                        fat_arrow_lines,
                        depth,
                    )?;
                }
                collect_match_arm_comments(
                    item_idx,
                    body,
                    remaining,
                    arm_trailing,
                    fat_arrow_lines,
                    depth,
                )?;
            }
        }
        Expr::Ascribe(inner, _) => {
            collect_match_arm_comments(
                item_idx,
                inner,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::Lambda { body, .. } => {
            collect_match_arm_comments(
                item_idx,
                body,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        // DN-58 §A/§B (M-667): `fuse(a, b)` and `reclaim(policy) { body }` — recurse into
        // sub-expressions to collect any match arm comments inside them. Neither form introduces
        // match-arm structure itself; they recurse transparently.
        Expr::Fuse { left, right } => {
            collect_match_arm_comments(
                item_idx,
                left,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                right,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        Expr::Reclaim { policy, body } => {
            collect_match_arm_comments(
                item_idx,
                policy,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
            collect_match_arm_comments(
                item_idx,
                body,
                remaining,
                arm_trailing,
                fat_arrow_lines,
                depth,
            )?;
        }
        // A tuple literal groups sub-expressions — recurse into each (like `App` args) to catch any
        // nested match-arm comments inside an element.
        Expr::TupleLit(elems) => {
            for e in elems {
                collect_match_arm_comments(
                    item_idx,
                    e,
                    remaining,
                    arm_trailing,
                    fat_arrow_lines,
                    depth,
                )?;
            }
        }
        // Leaves: Lit, Path — no subexpressions to recurse into.
        Expr::Lit(_) | Expr::Path(_) => {}
    }
    Ok(())
}

// ================================================================================================
// Body rendering with comment interleaving.
// ================================================================================================

/// Render the nodule body (all items) with leading/trailing comments interleaved.
/// Returns the rendered text and any notes for the caller.
///
/// **RFC-0041 §4.7 (RR-29) guard-hole closure (W1).** This is [`format_source_styled_cfg`]'s only
/// render-family entry, so it wraps once, here, in [`ensure_sufficient_stack`] — every callee it
/// reaches (`render_item_with_comments`, `render_fn_decl_with_comments`, `render_impl_with_comments`,
/// the Shape-Dispatched Readable family `render_expr_readable`/`render_expr_broken`/`render_spine`,
/// `render_expr_with_arm_comments`, and transitively `render_expr_canonical`) runs on the grown
/// 256 MiB worker stack rather than the caller's. One worker-thread spawn per format call, not per
/// recursion level (`nodule.items` is a flat loop over items; the per-item recursion beneath runs
/// unwrapped on this single grown stack). Output is unchanged — only the stack it runs on changed.
fn render_body_with_comments(
    nodule: &Nodule,
    plan: &CommentPlan,
    style: Style,
    cfg: LayoutCfg,
) -> Result<(String, Vec<String>), FmtError> {
    let budget = RecursionBudget::with_depth_default(u64::MAX, u64::MAX);
    ensure_sufficient_stack(&budget, || {
        render_body_with_comments_inner(nodule, plan, style, cfg)
    })
}

/// The actual body of [`render_body_with_comments`] — see that function's doc comment for the
/// guard-hole wrap.
fn render_body_with_comments_inner(
    nodule: &Nodule,
    plan: &CommentPlan,
    style: Style,
    cfg: LayoutCfg,
) -> Result<(String, Vec<String>), FmtError> {
    let mut out = String::new();
    let mut notes = Vec::new();

    // Render the nodule header line (e.g. `nodule signals.demo;`). DN-57 §3 (M-818): the nodule
    // header is a component — it ends with the mandatory `;` terminator (so the canonical output
    // re-parses under the mandatory-terminator grammar; the round-trip self-check at the call site
    // would otherwise reject it).
    let nodule_line = format!(
        "nodule {}{};\n",
        nodule.path.0.join("."),
        if nodule.std_sys { " @std-sys" } else { "" }
    );
    out.push_str(&nodule_line);

    for (item_idx, item) in nodule.items.iter().enumerate() {
        out.push('\n');

        // Emit any leading comments for this item.
        if let Some(leading_comments) = plan.leading.get(&item_idx) {
            let count = leading_comments.len();
            for cmt in leading_comments {
                out.push_str(cmt);
                out.push('\n');
            }
            notes.push(format!(
                "preserved {count} leading comment(s) on item {}",
                item_idx
            ));
        }

        // Render the item itself.
        let item_text = render_item_with_comments(item, item_idx, plan, style, cfg, &mut notes)?;
        out.push_str(&item_text);
    }

    // End-of-nodule trailing comments (attached after the last item, index = items.len()).
    let sentinel_idx = nodule.items.len();
    if let Some(trailing_comments) = plan.leading.get(&sentinel_idx) {
        for cmt in trailing_comments {
            out.push('\n');
            out.push_str(cmt);
            out.push('\n');
        }
        notes.push(format!(
            "preserved {} comment(s) at the end of the nodule",
            trailing_comments.len()
        ));
    }

    Ok((out, notes))
}

/// Render a single top-level item with its trailing comment (if any) and arm comments (if any).
fn render_item_with_comments(
    item: &Item,
    item_idx: usize,
    plan: &CommentPlan,
    style: Style,
    cfg: LayoutCfg,
    notes: &mut Vec<String>,
) -> Result<String, FmtError> {
    match item {
        Item::Fn(fd) => {
            let fn_trailing = plan.fn_trailing.get(&item_idx);
            let arm_map = plan.arm_trailing.get(&item_idx);
            render_fn_decl_with_comments(
                fd,
                fn_trailing.map(String::as_str),
                arm_map,
                style,
                cfg,
                notes,
            )
        }
        Item::Impl(id) => {
            // An impl can contain fns; delegate to render_impl which can attach arm comments
            // (currently: arm comments in impl methods are treated the same as top-level fns).
            render_impl_with_comments(id, item_idx, plan, style, cfg, notes)
        }
        // Other items (use, default, type, trait) are rendered via expand_to_source on a
        // synthetic single-item nodule-like string, or more simply: we re-implement the trivial
        // render inline (DRY note: these are verbatim copies of the ambient.rs private printers,
        // which are NOT accessible from this crate — this crate must duplicate the output).
        _ => Ok(render_non_fn_item(item, style)),
    }
}

/// Render a non-fn item to its canonical surface form.  Mirrors the private ambient.rs printers.
///
/// In `Style::Readable`, a `type` sum-type declaration whose compact single line would exceed
/// [`READABLE_WIDTH`] is re-wrapped one constructor per line (M-974); every other non-fn item
/// (`use`/`default`/`trait`/`object`/`lower`/`derive`) is short by construction, so it uses the
/// compact `expand_to_source` render in both styles.
fn render_non_fn_item(item: &Item, style: Style) -> String {
    if style == Style::Readable {
        if let Item::Type(td) = item {
            if let Some(readable) = render_type_decl_readable(td) {
                return readable;
            }
        }
    }
    render_non_fn_item_compact(item)
}

/// Render a non-fn item to its **compact** canonical surface form.  Mirrors the private ambient.rs
/// printers by round-tripping through `expand_to_source` on a synthetic single-item nodule.
fn render_non_fn_item_compact(item: &Item) -> String {
    // We derive the text by calling `expand_to_source` on a synthetic single-item nodule, then
    // extracting the item's text.  This avoids duplicating the printer logic while staying
    // correct.
    //
    // CAVEAT: `expand_to_source` always outputs `nodule <path>;\n\n<item>\n` (DN-57 §3 / M-818: the
    // header is a component, so it carries its mandatory `;`). We strip the header line.
    let synthetic = mycelium_l1::ast::Nodule {
        path: mycelium_l1::ast::Path(vec!["_".to_owned()]),
        std_sys: false,
        items: vec![item.clone()],
    };
    let full = expand_to_source(&synthetic);
    // `full` is `nodule _;\n\n<item text>\n`; we want just `<item text>\n`.
    // Strip the header line + the blank line (`nodule _;\n\n`) from the front: `splitn(3, '\n')`
    // yields `["nodule _;", "", "<item text>\n"]`, and `nth(2)` is the item text.
    full.splitn(3, '\n')
        .nth(2)
        .map(str::to_owned)
        .unwrap_or_default()
}

/// Render a `fn` declaration with optional fn-body trailing comment and arm comments.
///
/// The canonical form is:
/// ```text
/// fn name(params) -> ret =
///   body_expr
/// ```
/// With a trailing fn-body comment:
/// ```text
/// fn name(params) -> ret =
///   body_expr  // comment
/// ```
/// With match arm comments (multiline match):
/// ```text
/// fn name(params) -> ret =
///   match x {
///     arm0_pat => arm0_body,  // comment
///     arm1_pat => arm1_body
///   }
/// ```
fn render_fn_decl_with_comments(
    fd: &FnDecl,
    fn_trailing: Option<&str>,
    arm_map: Option<&HashMap<usize, String>>,
    style: Style,
    cfg: LayoutCfg,
    notes: &mut Vec<String>,
) -> Result<String, FmtError> {
    let pub_prefix = if fd.vis.is_pub() { "pub " } else { "" };
    let thaw_prefix = if fd.thaw { "thaw " } else { "" };

    let has_arm_comments = arm_map.is_some_and(|m| !m.is_empty());

    // The signature: compact by default. In Readable style, a signature whose header line
    // (`<pub><thaw>fn <sig> =`) would exceed READABLE_WIDTH wraps its value-parameter list one
    // parameter per line (M-974) — the "line breaks after commas for large segments" rule applied
    // to the value-param segment. The arm-comment path keeps the compact signature (its multiline
    // match rendering is comment-anchored and untouched by M-974).
    let compact_sig = render_sig_tail(&fd.sig);
    let sig_text = if style == Style::Readable && !has_arm_comments {
        let header_len = pub_prefix.len() + thaw_prefix.len() + "fn ".len() + compact_sig.len() + 2;
        if header_len > READABLE_WIDTH {
            render_sig_readable(&fd.sig)
        } else {
            compact_sig
        }
    } else {
        compact_sig
    };

    let body_text = if has_arm_comments {
        // The body contains a match with arm trailing comments; render multiline (comment-anchored).
        let amap = arm_map.expect("has_arm_comments implies Some");
        notes.push(format!(
            "preserved {} trailing arm comment(s) with multiline match rendering in `{}`",
            amap.len(),
            fd.sig.name
        ));
        render_expr_with_arm_comments(&fd.body, amap)?
    } else if style == Style::Readable {
        // M-974/M-976: readable body layout — long segments break, short ones stay inline;
        // same-head chains render as a flat spine (R1) per `cfg.spine_inner`.
        render_expr_readable(&fd.body, 2, cfg)
    } else {
        // Guarded entry (RFC-0041 §4.7 RR-29 / W1) — see `render_item_flat`'s matching comment.
        render_expr_canonical(&fd.body)
    };

    // When the body text is multiline (e.g. a multiline match), indent ALL lines by 2 spaces so
    // the canonical form is properly structured relative to the `fn sig =` header line.
    // A single-line body just gets the `  ` prefix directly.
    //
    // Two indentation regimes:
    //   * Compact / arm-comment bodies are rendered relative to column 0, so every line gets a
    //     blanket 2-space prefix to sit under the `fn sig =` header.
    //   * Readable bodies (`render_expr_readable(&fd.body, 2)`) are already *absolutely* indented
    //     for a first line at column 2 — their continuation lines carry their own padding — so only
    //     the first line is prefixed (prefixing the whole string touches just the leading line).
    let readable_body = style == Style::Readable && !has_arm_comments;
    let indented_body = if readable_body {
        format!("  {body_text}")
    } else if body_text.contains('\n') {
        body_text
            .lines()
            .map(|l| {
                if l.is_empty() {
                    String::new()
                } else {
                    format!("  {l}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        format!("  {body_text}")
    };

    // DN-57 §3 (M-818): the fn body ends with the mandatory `;` component terminator. It lands
    // directly after the body expression (and *before* any trailing comment), so the canonical form
    // re-parses under the mandatory-terminator grammar.
    let mut s = format!("{pub_prefix}{thaw_prefix}fn {sig_text} =\n{indented_body};\n");

    // Append fn-body trailing comment if any.
    // A trailing comment on the fn body means the whole fn was on one source line
    // (the body is a simple expression, not a multiline construct) so we append
    // to the (single) body line — between the body expression and the final `\n` (after the `;`).
    if let Some(cmt) = fn_trailing {
        if s.ends_with('\n') {
            s.pop(); // remove the trailing newline
            s.push_str(&format!("  {cmt}"));
            s.push('\n');
        }
        notes.push(format!(
            "preserved trailing comment on fn `{}`: {cmt}",
            fd.sig.name
        ));
    }

    Ok(s)
}

/// Render an `impl` declaration (which may contain fns with arm comments).
fn render_impl_with_comments(
    id: &ImplDecl,
    item_idx: usize,
    plan: &CommentPlan,
    style: Style,
    cfg: LayoutCfg,
    notes: &mut Vec<String>,
) -> Result<String, FmtError> {
    let args = if id.trait_args.is_empty() {
        String::new()
    } else {
        let a: Vec<String> = id.trait_args.iter().map(render_type_ref).collect();
        format!("[{}]", a.join(", "))
    };
    let mut s = format!(
        "impl {}{} for {} {{\n",
        id.trait_name,
        args,
        render_type_ref(&id.for_ty)
    );
    let arm_map = plan.arm_trailing.get(&item_idx);
    for method in &id.methods {
        // Each method is itself a component; `render_fn_decl_with_comments` already appends the
        // method's mandatory `;` (DN-57 §3, M-818).
        let method_text = render_fn_decl_with_comments(method, None, arm_map, style, cfg, notes)?;
        for line in method_text.lines() {
            s.push_str("  ");
            s.push_str(line);
            s.push('\n');
        }
    }
    // DN-57 §3 (M-818): the `impl` block is itself a component — its closing `}` carries the
    // mandatory `;` terminator, *uniformly* with the expression items (a `}`-closed block still
    // ends with `;`).
    s.push_str("};\n");
    Ok(s)
}

// ================================================================================================
// Expression renderers (mirrors of ambient.rs private printers, for the comment-aware path).
// ================================================================================================

/// Render an expression in canonical form (without any comment interleaving).
/// This mirrors `print_expr` in ambient.rs.
///
/// **RFC-0041 §4.7 (RR-29) guard-hole closure (W1).** This is the render family's own recursive
/// entry: [`render_expr_canonical_inner`] recurses over `Expr` directly (`Let`/`If`/`Match`/…), and
/// is reachable **both** from the guarded text pipeline ([`format_source_styled_cfg`] /
/// [`flatten_source`], each wrapped once at their own outer entry below) **and** directly, by any
/// caller that already holds an [`mycelium_l1::ast::Expr`] built some other way (bypassing
/// `mycelium_l1::parse`'s depth guard) — the census regression
/// (`src/tests/guard_hole_census.rs::render_expr_canonical_deep_let_chain`). So this function itself
/// wraps once, on entry, in [`mycelium_workstack::ensure_sufficient_stack`] and delegates to the
/// actual (unwrapped) recursive logic in [`render_expr_canonical_inner`] — every self-recursive call
/// inside `render_expr_canonical_inner` calls itself directly (never back through this wrapper), so
/// a single grown 256 MiB worker stack backs the whole recursive descent, not one worker thread per
/// nesting level. Infallible (returns `String`); the budget's depth/mem/step ceilings are
/// unbounded ([`u64::MAX`]) until W2's per-machine frame census wires real ceilings. Output is
/// byte-identical to before this change — only the stack the recursion runs on changed.
fn render_expr_canonical(e: &Expr) -> String {
    let budget = RecursionBudget::with_depth_default(u64::MAX, u64::MAX);
    ensure_sufficient_stack(&budget, || render_expr_canonical_inner(e))
}

/// The actual (unguarded) recursive descent behind [`render_expr_canonical`] — see that function's
/// doc comment for why the wrap lives one level up, not here.
fn render_expr_canonical_inner(e: &Expr) -> String {
    match e {
        Expr::Lit(l) => render_literal(l),
        Expr::Path(p) => p.0.join("."),
        // v0 tuple literal (M-826): `(a, b, …)`. Desugars to a synthetic `Tuple$N` constructor in
        // the checker/mono; the surface round-trips as the parenthesized, comma-separated list.
        Expr::TupleLit(elems) => {
            let s: Vec<String> = elems.iter().map(render_expr_canonical_inner).collect();
            format!("({})", s.join(", "))
        }
        // RFC-0037 D5 lambda. Closure semantics are deferred to M-704; this canonical render mirrors
        // ambient.rs `print_expr` (param names + `=>` body). Lambdas are absent from the v0 corpus,
        // so the comment-aware token path (not this fallback) drives all current conformance.
        Expr::Lambda { params, body } => format!(
            "lambda({}) => {}",
            params
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
                .join(", "),
            render_expr_canonical_inner(body)
        ),
        Expr::Let {
            name,
            ty,
            bound,
            body,
        } => {
            let ann = ty
                .as_ref()
                .map(|t| format!(": {}", render_type_ref(t)))
                .unwrap_or_default();
            format!(
                "let {name}{ann} = {} in {}",
                render_expr_canonical_inner(bound),
                render_expr_canonical_inner(body)
            )
        }
        Expr::If { cond, conseq, alt } => format!(
            "if {} then {} else {}",
            render_expr_canonical_inner(cond),
            render_expr_canonical_inner(conseq),
            render_expr_canonical_inner(alt)
        ),
        Expr::Match { scrutinee, arms } => {
            let arm_strs: Vec<String> = arms
                .iter()
                .map(|a| {
                    format!(
                        "{} => {}",
                        render_pattern(&a.pattern),
                        render_expr_canonical_inner(&a.body)
                    )
                })
                .collect();
            format!(
                "match {} {{ {} }}",
                render_expr_canonical_inner(scrutinee),
                arm_strs.join(", ")
            )
        }
        Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        } => format!(
            "for {x} in {}, {acc} = {} => {}",
            render_expr_canonical_inner(xs),
            render_expr_canonical_inner(init),
            render_expr_canonical_inner(body)
        ),
        Expr::Swap {
            value,
            target,
            policy,
        } => format!(
            "swap({}, to: {}, policy: {})",
            render_expr_canonical_inner(value),
            render_type_ref(target),
            policy.0.join(".")
        ),
        Expr::WithParadigm { paradigm, body } => {
            format!(
                "with paradigm {paradigm} {{ {} }}",
                render_expr_canonical_inner(body)
            )
        }
        Expr::Wild(b) => format!("wild {{ {} }}", render_expr_canonical_inner(b)),
        Expr::Spore(b) => format!("spore({})", render_expr_canonical_inner(b)),
        Expr::Colony(hyphae) => {
            // M-970 (found by M-914): a hypha's optional `@forage(policy)` placement annotation
            // (RFC-0008 RT3; DN-63 §3.5; M-906/DN-70 D1) must round-trip through the canonical
            // render — this is the exact inverse of `Parser::parse_hypha`
            // (`crates/mycelium-l1/src/parse.rs`): `@forage(<policy>) hypha <body>` when present,
            // else bare `hypha <body>`. Previously this arm dropped `Hypha::forage` entirely,
            // which silently erased the annotation on format (a G2 violation) — the C1 identity
            // guard below caught the mismatch and refused (exit 4), but never emitted the fix.
            //
            // The `Hypha { forage, body }` destructure (rather than `h.forage`/`h.body` field
            // access) is deliberate: it makes a FUTURE field added to `Hypha` a compile error
            // here (non-exhaustive destructure) instead of a silent drop like this one was — the
            // struct-field analogue of an exhaustive enum match with no catch-all arm.
            let hs: Vec<String> = hyphae
                .iter()
                .map(|h| {
                    let Hypha { forage, body } = h;
                    match forage {
                        Some(policy) => format!(
                            "@forage({}) hypha {}",
                            render_expr_canonical_inner(policy),
                            render_expr_canonical_inner(body)
                        ),
                        None => format!("hypha {}", render_expr_canonical_inner(body)),
                    }
                })
                .collect();
            format!("colony {{ {} }}", hs.join(", "))
        }
        Expr::App { head, args } => {
            let arg_strs: Vec<String> = args.iter().map(render_expr_canonical_inner).collect();
            format!(
                "{}({})",
                render_expr_canonical_inner(head),
                arg_strs.join(", ")
            )
        }
        Expr::Ascribe(inner, t) => {
            format!(
                "{} : {}",
                render_expr_canonical_inner(inner),
                render_type_ref(t)
            )
        }
        // DN-58 §A/§B (M-667): canonical rendering for `fuse(a, b)` and `reclaim(pol) { body }`.
        // Mirrors `print_expr` in ambient.rs.
        Expr::Fuse { left, right } => format!(
            "fuse({}, {})",
            render_expr_canonical_inner(left),
            render_expr_canonical_inner(right)
        ),
        Expr::Reclaim { policy, body } => format!(
            "reclaim({}) {{ {} }}",
            render_expr_canonical_inner(policy),
            render_expr_canonical_inner(body)
        ),
        // `consume e` — affine move of a substrate (mirrors `print_expr` in ambient.rs).
        Expr::Consume(b) => format!("consume {}", render_expr_canonical_inner(b)),
    }
}

// ================================================================================================
// Readable (human) layout (M-974 / DN-82).  These renderers break long argument / field / variant
// / arm segments across lines (line breaks after commas / `|`) while keeping short constructs
// inline — the inverse posture of `--flatten`.  Every break is whitespace-only, so the readable
// output re-parses to the SAME surface AST as the compact form (C1) and is a fixed point (C2): the
// style is a presentation projection, functionally inert.
// ================================================================================================

/// Does `compact` (a single-line rendering) fit on one line when placed at column `indent`, against
/// `width` (M-976: [`LayoutCfg::width`], default [`READABLE_WIDTH`])? One threshold governs EVERY
/// construct (R0) — chars (not bytes), measured at the node's own indent. The readability decision is
/// `Declared` — a width threshold, not a proven bound.
fn fits_w(compact: &str, indent: usize, width: usize) -> bool {
    !compact.contains('\n') && indent + compact.chars().count() <= width
}

/// A right-nested same-head chain (R1): `App{head:H, [..lead, App{head:H, ..}]}` repeated ≥ 2 deep.
/// Returns `(head-render, leading-args-per-link in order, terminal, depth)`. Detects
/// Cons/GLCons/TCons/bool_and/cat and any right-fold-encoded variadic uniformly — all are `App` with
/// a `Path` head; there is no separate `Cons` node. Reads ONLY the AST, so C2 holds by construction.
fn same_head_chain(e: &Expr) -> Option<(String, Vec<Vec<&Expr>>, &Expr, usize)> {
    let head_s = match e {
        Expr::App { head, args } if !args.is_empty() => render_expr_canonical_inner(head),
        _ => return None,
    };
    let mut node = e;
    let mut links: Vec<Vec<&Expr>> = Vec::new();
    let terminal: &Expr;
    loop {
        let Expr::App { head, args } = node else {
            return None;
        };
        if args.is_empty() || render_expr_canonical_inner(head) != head_s {
            return None;
        }
        let (lead, last) = args.split_at(args.len() - 1);
        let last = &last[0];
        links.push(lead.iter().collect());
        match last {
            Expr::App { head: h2, args: a2 }
                if !a2.is_empty() && render_expr_canonical_inner(h2) == head_s =>
            {
                node = last; // continue unrolling the chain
            }
            _ => {
                terminal = last; // first non-same-head last-arg = the chain terminal
                break;
            }
        }
    }
    let depth = links.len();
    if depth >= 2 {
        Some((head_s, links, terminal, depth))
    } else {
        None
    }
}

/// Does this arm body force its arm (and hence the enclosing match) open regardless of width (R4c)?
/// True iff the body is (or, through an ascription, wraps) a top-level `let … in …` — the
/// twice-flagged buried-binding pathology (inspect/inspect_err).
fn arm_body_has_top_let(body: &Expr) -> bool {
    match body {
        Expr::Let { .. } => true,
        Expr::Ascribe(inner, _) => arm_body_has_top_let(inner),
        _ => false,
    }
}

/// Does `e` have to break even though its compact render fits (R4c)? A `match` whose any arm body
/// forces open cannot render inline — the forced arm break propagates up to the match node.
fn expr_forces_break(e: &Expr) -> bool {
    match e {
        Expr::Match { arms, .. } => arms.iter().any(|a| arm_body_has_top_let(&a.body)),
        _ => false,
    }
}

/// Render one inner (leading / non-recursive) argument of a same-head spine link, honoring the
/// house-style knob (R1 / [`SpineInner`]). `InlineWhenFits` keeps a fitting call inline (and lets an
/// overflowing one block per R2); `AlwaysExpand` force-breaks every nested call onto its own lines
/// while the spine stays flat.
fn render_spine_inner_arg(a: &Expr, indent: usize, cfg: LayoutCfg) -> String {
    match cfg.spine_inner {
        SpineInner::AlwaysExpand if matches!(a, Expr::App { args, .. } if !args.is_empty()) => {
            render_expr_broken(a, indent, cfg)
        }
        _ => render_expr_readable(a, indent, cfg),
    }
}

/// R1 flat spine + R5 coalesced closers. Every link sits at ONE fixed indent (`indent`, never +2 per
/// link); the terminal begins the final line; all `depth` closers collapse into a single horizontal
/// run on that line. The FIRST line carries no pad (indent contract); the caller places it at
/// `indent`.
fn render_spine(
    head_s: &str,
    links: &[Vec<&Expr>],
    terminal: &Expr,
    depth: usize,
    indent: usize,
    cfg: LayoutCfg,
) -> String {
    let pad = " ".repeat(indent);
    let mut lines: Vec<String> = Vec::with_capacity(depth + 1);
    for (li, lead) in links.iter().enumerate() {
        let lead_s: Vec<String> = lead
            .iter()
            .map(|a| render_spine_inner_arg(a, indent, cfg))
            .collect();
        let link = format!("{head_s}({},", lead_s.join(", "));
        if li == 0 {
            lines.push(link); // indent contract: first line unpadded
        } else {
            lines.push(format!("{pad}{link}"));
        }
    }
    let term_s = render_spine_inner_arg(terminal, indent, cfg);
    let closers = ")".repeat(depth);
    lines.push(format!("{pad}{term_s}{closers}"));
    lines.join("\n")
}

/// Render an expression in the human-readable multi-line style (M-974/M-976).
///
/// **Indent contract:** the returned string's FIRST line carries no leading indentation — the
/// caller places it at column `indent`; any continuation lines are *absolutely* indented as if the
/// first line began at column `indent`. A node whose compact single-line render fits within
/// [`LayoutCfg::width`] at `indent` — AND is not forced open (R4c) — stays inline; otherwise it
/// breaks per the shape-dispatched rules ([`render_expr_broken`]). AST-driven, so idempotent (C2).
fn render_expr_readable(e: &Expr, indent: usize, cfg: LayoutCfg) -> String {
    let compact = render_expr_canonical_inner(e);
    if fits_w(&compact, indent, cfg.width) && !expr_forces_break(e) {
        return compact;
    }
    render_expr_broken(e, indent, cfg)
}

/// The break layouts (Shape-Dispatched Readable, M-976). Dispatch order (R6): (1) same-head chain →
/// R1 spine + R5 closers; (2) else App → R2 wide-flat block; (3) match/if → R3 tree; (4) let → R4
/// binding; (5) tuple/list/ascription → per-kind; (6) compact fallback. Assumes the node must break
/// (the caller verified `!fits || forces_break`); every layout is whitespace-only, so C1/C2 hold.
fn render_expr_broken(e: &Expr, indent: usize, cfg: LayoutCfg) -> String {
    use mycelium_l1::ast::Literal;
    let compact = render_expr_canonical_inner(e);
    let pad = " ".repeat(indent);
    let inner = indent + 2;
    let ipad = " ".repeat(inner);
    match e {
        // R1 (dispatched FIRST, R6): a right-nested same-head chain lays out as a flat spine so a
        // semantically-flat N-element list is never drawn as an N-deep rightward-drifting tree. R2
        // (wide-flat block) otherwise — one arg per line at indent+2, `)` alone on its own line.
        Expr::App { head, args } if !args.is_empty() => {
            if let Some((head_s, links, terminal, depth)) = same_head_chain(e) {
                render_spine(&head_s, &links, terminal, depth, indent, cfg)
            } else {
                let head_s = render_expr_readable(head, indent, cfg);
                let arg_lines: Vec<String> = args
                    .iter()
                    .map(|a| format!("{ipad}{}", render_expr_readable(a, inner, cfg)))
                    .collect();
                format!("{head_s}(\n{}\n{pad})", arg_lines.join(",\n"))
            }
        }
        // R3 genuine-tree indentation: one indent (2 sp) per REAL nesting level; a `{}`-block-former
        // body (nested match/if) RIDES the arm line (case b) so a linear match ladder halves its
        // depth; otherwise `=>` drops to its own line and the body renders one level deeper (case c).
        // R4c: a let-headed arm ALWAYS breaks (body force-broken at inner+2) even when it would fit.
        Expr::Match { scrutinee, arms } => {
            let scrut = render_expr_readable(scrutinee, indent + "match ".len(), cfg);
            let arm_lines: Vec<String> = arms
                .iter()
                .enumerate()
                .map(|(i, arm)| {
                    let is_last = i + 1 == arms.len();
                    let comma = if is_last { "" } else { "," };
                    let patn = render_pattern(&arm.pattern);
                    let forces = arm_body_has_top_let(&arm.body); // R4c
                    let body_compact = render_expr_canonical_inner(&arm.body);
                    let one_line = format!("{patn} => {body_compact}{comma}");
                    // Case (a): the whole arm fits on one line (and is not force-opened).
                    if !forces && fits_w(&one_line, inner, cfg.width) {
                        return format!("{ipad}{one_line}");
                    }
                    // Case (b): a nested match/if body rides the arm line, its inner arms one level
                    // below — reserving depth only for genuine nesting.
                    if !forces && matches!(arm.body, Expr::Match { .. } | Expr::If { .. }) {
                        let body = render_expr_readable(&arm.body, inner, cfg);
                        let first = body.lines().next().unwrap_or("");
                        let ride = format!("{patn} => {first}");
                        if fits_w(&ride, inner, cfg.width) {
                            return format!("{ipad}{patn} => {body}{comma}");
                        }
                    }
                    // Case (c): `=>` on its own line; the body renders one level deeper per its kind
                    // (a same-head-chain body reaches R1 and spines; a wide-flat body → R2 block; a
                    // let-headed arm is force-broken here — R4c).
                    let body = render_expr_broken(&arm.body, inner + 2, cfg);
                    format!("{ipad}{patn} =>\n{ipad}  {body}{comma}")
                })
                .collect();
            format!("match {scrut} {{\n{}\n{pad}}}", arm_lines.join("\n"))
        }
        // R4 binding layout: structural BLOCK indent (never align-to-open-paren). The `bound` breaks
        // at the let's OWN indent (rename-stable, matches R2's closer discipline), and the body
        // always starts on its own line at the let-chain indent (never glued after the last `in`).
        Expr::Let {
            name,
            ty,
            bound,
            body,
        } => {
            let ann = ty
                .as_ref()
                .map(|t| format!(": {}", render_type_ref(t)))
                .unwrap_or_default();
            let head = format!("let {name}{ann} = ");
            let bound_s = render_expr_readable(bound, indent, cfg);
            let body_s = render_expr_readable(body, indent, cfg);
            format!("{head}{bound_s} in\n{pad}{body_s}")
        }
        // `if cond then conseq else alt` — the three parts on their own lines.
        Expr::If { cond, conseq, alt } => {
            let cond_s = render_expr_readable(cond, indent + "if ".len(), cfg);
            let conseq_s = render_expr_readable(conseq, inner, cfg);
            let alt_s = render_expr_readable(alt, inner, cfg);
            format!("if {cond_s} then\n{ipad}{conseq_s}\n{pad}else\n{ipad}{alt_s}")
        }
        // A tuple literal with long elements: one element per line, break after each comma.
        Expr::TupleLit(elems) if !elems.is_empty() => {
            let elem_lines: Vec<String> = elems
                .iter()
                .map(|el| format!("{ipad}{}", render_expr_readable(el, inner, cfg)))
                .collect();
            format!("(\n{}\n{pad})", elem_lines.join(",\n"))
        }
        // A list literal with long elements: one element per line, break after each comma.
        Expr::Lit(Literal::List(es)) if !es.is_empty() => {
            let elem_lines: Vec<String> = es
                .iter()
                .map(|el| format!("{ipad}{}", render_expr_readable(el, inner, cfg)))
                .collect();
            format!("[\n{}\n{pad}]", elem_lines.join(",\n"))
        }
        // Type ascription: wrap the inner expression, keep the type inline.
        Expr::Ascribe(inner_e, t) => {
            format!(
                "{} : {}",
                render_expr_readable(inner_e, indent, cfg),
                render_type_ref(t)
            )
        }
        // Constructs with no comma/`|` segment to break (or an empty one): fall back to the compact
        // single-line rendering. The C1 guard still verifies round-trip; nothing is dropped (G2).
        _ => compact,
    }
}

/// Render one sum-type constructor (`Name` or `Name(T1, T2)`) — mirrors the ctor render in
/// `mycelium_l1::ambient::print_type_decl`, used by [`render_type_decl_readable`].
fn render_ctor(c: &mycelium_l1::ast::Ctor) -> String {
    if c.fields.is_empty() {
        c.name.clone()
    } else {
        let fs: Vec<String> = c.fields.iter().map(render_type_ref).collect();
        format!("{}({})", c.name, fs.join(", "))
    }
}

/// Render a `type` sum-type declaration in the readable style (M-974) — one constructor per line,
/// break after each `|` — **only when** its compact single line would exceed [`READABLE_WIDTH`].
/// Returns `None` when the compact form already fits (the caller then uses the compact render), so a
/// short `type Result[A, E] = Ok(A) | Err(E);` stays inline. The output ends with the mandatory `;`
/// terminator + newline, matching the compact item text (DN-57 §3).
///
/// ```text
/// type Predicate =
///     PAlways
///   | PSrcKindIs(Kind)
///   | PAnd(Predicate, Predicate);
/// ```
fn render_type_decl_readable(td: &mycelium_l1::ast::TypeDecl) -> Option<String> {
    let vis = if td.vis.is_pub() { "pub " } else { "" };
    let params = if td.params.is_empty() {
        String::new()
    } else {
        format!("[{}]", td.params.join(", "))
    };
    let ctors: Vec<String> = td.ctors.iter().map(render_ctor).collect();
    // Fit decision on the compact one-line form (with its `;`) — AST-driven, so idempotent.
    let compact_line = format!("{vis}type {}{} = {};", td.name, params, ctors.join(" | "));
    if compact_line.chars().count() <= READABLE_WIDTH {
        return None;
    }
    // Wrap: first ctor indented 4 (aligned under `= `), each subsequent ctor prefixed `| ` at 2.
    let mut lines: Vec<String> = Vec::with_capacity(ctors.len());
    for (i, c) in ctors.iter().enumerate() {
        if i == 0 {
            lines.push(format!("    {c}"));
        } else {
            lines.push(format!("  | {c}"));
        }
    }
    Some(format!(
        "{vis}type {}{} =\n{};\n",
        td.name,
        params,
        lines.join("\n")
    ))
}

/// Render an expression, using multiline match rendering if the arm_map is non-empty and
/// the expression (or sub-expression) is a match.
///
/// **Comma placement:** the arm separator `,` must go BEFORE any trailing comment on that arm's
/// line.  A `// …` comment runs to end-of-line, so a comma placed AFTER it would be swallowed
/// into the comment text and then lost when the output is re-lexed, breaking the round-trip.
/// Canonical multiline form:
/// ```text
/// match scrutinee {
///   pat0 => body0,  // comment
///   pat1 => body1
/// }
/// ```
fn render_expr_with_arm_comments(
    e: &Expr,
    arm_map: &HashMap<usize, String>,
) -> Result<String, FmtError> {
    match e {
        Expr::Match { scrutinee, arms } => {
            // Multiline match rendering: one arm per line, comma BEFORE any trailing comment.
            let scrutinee_str = render_expr_canonical_inner(scrutinee);
            let last_idx = arms.len().saturating_sub(1);
            let mut arm_strs = Vec::with_capacity(arms.len());
            for (arm_idx, arm) in arms.iter().enumerate() {
                let arm_body = render_expr_canonical_inner(&arm.body);
                let pat = render_pattern(&arm.pattern);
                let is_last = arm_idx == last_idx;
                // The comma (`,`) separates arms and must appear BEFORE any trailing comment
                // so it is not swallowed into the comment text during re-lexing.
                let comma = if is_last { "" } else { "," };
                if let Some(cmt) = arm_map.get(&arm_idx) {
                    arm_strs.push(format!("  {pat} => {arm_body}{comma}  {cmt}"));
                } else {
                    arm_strs.push(format!("  {pat} => {arm_body}{comma}"));
                }
            }
            Ok(format!(
                "match {scrutinee_str} {{\n{}\n}}",
                arm_strs.join("\n")
            ))
        }
        // For non-match expressions, fall through to canonical rendering.
        // (arm_map entries for arm lines that are not a direct top-level match would already have
        // been refused by the comment plan builder.)
        _ => Ok(render_expr_canonical_inner(e)),
    }
}

fn render_pattern(p: &Pattern) -> String {
    match p {
        Pattern::Wildcard => "_".to_owned(),
        Pattern::Lit(l) => render_literal(l),
        // v0 tuple pattern (M-826): `(p1, p2, …)` — desugars to a `MkTuple$N` ctor pattern in the checker.
        Pattern::Tuple(subs) => {
            let s: Vec<String> = subs.iter().map(render_pattern).collect();
            format!("({})", s.join(", "))
        }
        Pattern::Ctor(n, subs) => {
            let s: Vec<String> = subs.iter().map(render_pattern).collect();
            format!("{n}({})", s.join(", "))
        }
        // v0 or-pattern (M-823 / R20-Q3): `A | B | …` — desugars in the checker to one arm per
        // alternative; the surface round-trips as the `|`-separated list of alternatives.
        Pattern::Or(alts) => {
            let s: Vec<String> = alts.iter().map(render_pattern).collect();
            s.join(" | ")
        }
        Pattern::Ident(n) => n.clone(),
    }
}

fn render_literal(l: &mycelium_l1::ast::Literal) -> String {
    use mycelium_l1::ast::Literal;
    match l {
        Literal::Bin(s) => format!("0b{s}"),
        // RFC-0037 D4: balanced-ternary literals render with the `0t…` prefix (the `<…>` form is retired).
        Literal::Trit(s) => format!("0t{s}"),
        // RFC-0032 D4 (M-750): a `0x…` byte-string literal round-trips to its source form.
        Literal::Bytes(s) => format!("0x{s}"),
        Literal::Int(i) => format!("{i}"),
        Literal::AmbientInt(p, i) => format!("{i} /* {p} (width from context) */"),
        Literal::List(es) => {
            let s: Vec<String> = es.iter().map(render_expr_canonical_inner).collect();
            format!("[{}]", s.join(", "))
        }
        // M-910/M-911: a `"…"` textual string literal round-trips to its source form; the
        // escape set here is the exact inverse of `Lexer::lex_string`'s decode table
        // (`\n \t \\ \" \0 \r`) so re-lexing the rendered output reproduces the same
        // `Literal::Str`.
        Literal::Str(s) => format!("\"{}\"", escape_string_literal(s)),
        // ADR-040 (M-897): a decimal float literal stores its source text verbatim, so it
        // round-trips as-is (re-lexing the rendered output reproduces the same `Literal::Float`).
        Literal::Float(s) => s.clone(),
        // `Literal` is `#[non_exhaustive]` (mycelium-l1/src/ast.rs) precisely so a downstream
        // crate like this one is *required* to keep a wildcard arm — every variant as of this
        // writing (Bin/Trit/Int/AmbientInt/List/Bytes/Str/Float) has an explicit arm above, so
        // this only fires for a variant added after this file and not yet wired up here: a
        // never-silent internal error (G2) rather than a silently-wrong render.
        _ => unreachable!(
            "unrecognized Literal variant — mycelium-l1 version mismatch (G2: never silent)"
        ),
    }
}

/// Re-escape a decoded string literal's content back to `.myc` source form. The exact inverse
/// of `mycelium_l1::lexer::Lexer::lex_string`'s decode table (`\n \t \\ \" \0 \r`) — mirrors
/// `mycelium_l1::ambient::escape_string_literal` (private to that crate; mycelium-fmt does not
/// depend on mycelium-l1's private internals, so the escape set is kept in sync by definition
/// rather than by import).
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

/// Renders a [`mycelium_l1::ast::TypeRef`] — the type-keyword render path, including the four
/// paradigm repr keywords (`BaseType::Binary`/`Ternary`/`Dense`/`Vsa`).
///
/// **RFC-0037 D2-b (M-915) canonicalization choice — `Declared`.** The short repr-keyword aliases
/// `bin`/`tern`/`emb`/`hvec` elaborate identically to their long forms at parse time (the parser
/// produces the exact same `BaseType::Binary`/`Ternary`/`Dense`/`Vsa` for either spelling — see
/// `mycelium-l1`'s `Tok::BinShort` doc comment); the AST itself retains **no record** of which
/// spelling was written. Consequently there is no separate short-form render arm here by
/// construction: this function always emits the long form (`Binary`/`Ternary`/`Dense`/`VSA`), so
/// `mycfmt` canonicalizes a short-alias input to its long-form output. This keeps the existing
/// corpus and every pre-existing `mycelium-fmt` fixture byte-identical (no reformat churn) while
/// still accepting the ergonomic short spelling as input — verified by
/// `docs/spec/grammar/conformance/accept/26-short-repr-keywords.myc` round-tripping through both
/// `format_source` (C1/C2) and `flatten_source`.
fn render_type_ref(t: &mycelium_l1::ast::TypeRef) -> String {
    use mycelium_l1::ast::BaseType;
    let base = match &t.base {
        BaseType::Binary(n) => format!("Binary{{{n}}}"),
        BaseType::Ternary(m) => format!("Ternary{{{m}}}"),
        BaseType::Dense(d, s) => format!("Dense{{{d}, {}}}", scalar_str(*s)),
        // v0 tuple type (M-826): `(T, U, …)`.
        BaseType::Tuple(elems) => {
            let s: Vec<String> = elems.iter().map(render_type_ref).collect();
            format!("({})", s.join(", "))
        }
        BaseType::Vsa {
            model,
            dim,
            sparsity,
        } => {
            format!("VSA{{{model}, {dim}, {}}}", sparsity_str(sparsity))
        }
        BaseType::Substrate(t) => format!("Substrate{{{t}}}"),
        // RFC-0032 D3/D4 (M-749/M-750): `Seq{T, N}` / nullary `Bytes`.
        BaseType::Seq { elem, len } => format!("Seq{{{}, {len}}}", render_type_ref(elem)),
        BaseType::Bytes => "Bytes".to_owned(),
        // ADR-040 (M-897): the nullary scalar-float repr keyword (binary64 only — FLAG-1).
        BaseType::Float => "Float".to_owned(),
        BaseType::Named(n, args) if args.is_empty() => n.clone(),
        BaseType::Named(n, args) => {
            // RFC-0037 D1: type arguments in `[…]` (was `<…>`).
            let a: Vec<String> = args.iter().map(render_type_ref).collect();
            format!("{n}[{}]", a.join(", "))
        }
        BaseType::Ambient(params) => format!("{{{}}}", ambient_params_str(params)),
        // RFC-0037 D4: function type `A => B` (right-associative; the `->` glyph is retired). The
        // parser builds `Fn(atom, rhs)` where the left is always a non-`Fn` atom, so rendering both
        // sides recursively and joining with ` => ` round-trips without parentheses (C1).
        BaseType::Fn(a, b) => format!("{} => {}", render_type_ref(a), render_type_ref(b)),
    };
    match t.guarantee {
        Some(g) => format!("{base} @ {g:?}"),
        None => base,
    }
}

fn render_sig_tail(sig: &mycelium_l1::ast::FnSig) -> String {
    use mycelium_l1::ast::ParamKind;
    // RFC-0037 D2: type parameters render in `[…]`, const/width parameters in `{…}` (kind-split).
    let type_ps: Vec<String> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Type)
        .map(render_type_param)
        .collect();
    let const_ps: Vec<String> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Width)
        .map(|p| p.name.clone())
        .collect();
    let tp = if type_ps.is_empty() {
        String::new()
    } else {
        format!("[{}]", type_ps.join(", "))
    };
    let cp = if const_ps.is_empty() {
        String::new()
    } else {
        format!("{{{}}}", const_ps.join(", "))
    };
    let ps: Vec<String> = sig
        .value_params
        .iter()
        .map(|p| format!("{}: {}", p.name, render_type_ref(&p.ty)))
        .collect();
    let eff = render_effects_suffix(sig);
    // RFC-0037 D4: return arrow `=>` (the `->` glyph is retired).
    format!(
        "{}{}{}({}) => {}{}",
        sig.name,
        tp,
        cp,
        ps.join(", "),
        render_type_ref(&sig.ret),
        eff
    )
}

/// The trailing effect-set suffix of a signature — ` !{name, name(<=N), …}` when present, else empty.
///
/// RFC-0014 §4.5 I4 (M-677): render each effect with its budget bound when present — `name(<=N)`.
/// The parser folds any `KiB`/`MiB`/`GiB` suffix into a unit-less byte count
/// (`effect_budgets: BTreeMap<String, u64>`), so the canonical surface is the raw `<=N`; this
/// round-trips AST-equal. Shared by the compact ([`render_sig_tail`]) and readable
/// ([`render_sig_readable`]) signature renderers (DRY) so the two never drift.
fn render_effects_suffix(sig: &mycelium_l1::ast::FnSig) -> String {
    if sig.effects.is_empty() {
        return String::new();
    }
    let rendered: Vec<String> = sig
        .effects
        .iter()
        .map(|e| match sig.effect_budgets.get(e) {
            Some(n) => format!("{e}(<={n})"),
            None => e.clone(),
        })
        .collect();
    format!(" !{{{}}}", rendered.join(", "))
}

/// Render a signature in the **readable** style (M-974): the value-parameter list breaks one
/// parameter per line (line breaks after commas), keeping the type/width parameter lists and the
/// return/effects inline. Returns the signature text *without* the leading `fn ` (the caller adds
/// it) — the first line follows `fn ` at column 3, the parameters indent to column 2, and the
/// closing `) => Ret !{eff}` returns to column 0. Whitespace-only vs [`render_sig_tail`], so it
/// re-parses to the same [`mycelium_l1::ast::FnSig`] (C1).
///
/// ```text
/// name[T]{N}(
///   p1: T1,
///   p2: T2
/// ) => Ret !{eff}
/// ```
fn render_sig_readable(sig: &mycelium_l1::ast::FnSig) -> String {
    use mycelium_l1::ast::ParamKind;
    let type_ps: Vec<String> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Type)
        .map(render_type_param)
        .collect();
    let const_ps: Vec<String> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Width)
        .map(|p| p.name.clone())
        .collect();
    let tp = if type_ps.is_empty() {
        String::new()
    } else {
        format!("[{}]", type_ps.join(", "))
    };
    let cp = if const_ps.is_empty() {
        String::new()
    } else {
        format!("{{{}}}", const_ps.join(", "))
    };
    let eff = render_effects_suffix(sig);
    let ret = render_type_ref(&sig.ret);
    // A niladic fn cannot overflow the width via its (empty) parameter list — fall back to the
    // compact tail so `name() => Ret` never becomes an empty `(\n\n)`.
    if sig.value_params.is_empty() {
        return format!("{}{}{}() => {}{}", sig.name, tp, cp, ret, eff);
    }
    let ps: Vec<String> = sig
        .value_params
        .iter()
        .map(|p| format!("  {}: {}", p.name, render_type_ref(&p.ty)))
        .collect();
    format!(
        "{}{}{}(\n{}\n) => {}{}",
        sig.name,
        tp,
        cp,
        ps.join(",\n"),
        ret,
        eff
    )
}

fn render_type_param(p: &mycelium_l1::ast::TypeParam) -> String {
    if p.bounds.is_empty() {
        p.name.clone()
    } else {
        let bs: Vec<String> = p.bounds.iter().map(render_trait_ref).collect();
        format!("{}: {}", p.name, bs.join(" + "))
    }
}

fn render_trait_ref(tr: &mycelium_l1::ast::TraitRef) -> String {
    if tr.args.is_empty() {
        tr.name.clone()
    } else {
        let args: Vec<String> = tr.args.iter().map(render_type_ref).collect();
        format!("{}[{}]", tr.name, args.join(", "))
    }
}

fn scalar_str(s: mycelium_l1::ast::Scalar) -> &'static str {
    use mycelium_l1::ast::Scalar;
    match s {
        Scalar::F16 => "F16",
        Scalar::Bf16 => "BF16",
        Scalar::F32 => "F32",
        Scalar::F64 => "F64",
    }
}

fn sparsity_str(s: &mycelium_l1::ast::Sparsity) -> String {
    use mycelium_l1::ast::Sparsity;
    match s {
        Sparsity::Dense => "Dense".to_owned(),
        Sparsity::Sparse(k) => format!("Sparse{{{k}}}"),
    }
}

fn ambient_params_str(p: &mycelium_l1::ast::AmbientParams) -> String {
    use mycelium_l1::ast::AmbientParams;
    match p {
        AmbientParams::Size(n) => format!("{n}"),
        AmbientParams::Dense(d, s) => format!("{d}, {}", scalar_str(*s)),
        AmbientParams::Vsa {
            model,
            dim,
            sparsity,
        } => {
            format!("{model}, {dim}, {}", sparsity_str(sparsity))
        }
    }
}

// ================================================================================================
// Helpers (unchanged from the original implementation).
// ================================================================================================

/// The 1-based line index where the body (the `nodule …` code) begins: the first non-blank line that is
/// not a `//` comment. Every header/leading line is a `//` comment or blank, so this cleanly separates the
/// comment/header region from the code. Returns `lines.len()` if there is no code line.
fn body_start_line(lines: &[&str]) -> usize {
    lines
        .iter()
        .position(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with("//")
        })
        .unwrap_or(lines.len())
}

/// The leading comment block (case: no structured header) — the comment lines before the code,
/// each verbatim, one per line, blank lines dropped, terminated by a newline. Empty if there are none.
fn leading_comment_block(lines: &[&str], end: usize) -> String {
    let mut out = String::new();
    for l in &lines[..end.min(lines.len())] {
        if l.trim().starts_with("//") {
            out.push_str(l);
            out.push('\n');
        }
    }
    out
}

/// Render a structured header canonically: the `// nodule:` marker, then present `// @key:` lines in the
/// fixed §4 order (`HEADER_KEYS`), one space after each colon, comma-joined lists. Values are re-emitted
/// as parsed — never fabricated (VR-5).
fn render_header(h: &StructuredHeader) -> String {
    let mut s = h.marker.canonical();
    s.push('\n');
    let f = &h.fields;
    if let Some(v) = &f.version {
        s.push_str(&format!("// @version: {v}\n"));
    }
    if let Some(v) = &f.license {
        s.push_str(&format!("// @license: {v}\n"));
    }
    if let Some(v) = &f.authors {
        s.push_str(&format!("// @authors: {}\n", v.join(", ")));
    }
    if let Some(v) = &f.since {
        s.push_str(&format!("// @since: {v}\n"));
    }
    if let Some(v) = &f.updated {
        s.push_str(&format!("// @updated: {v}\n"));
    }
    if let Some(v) = &f.summary {
        s.push_str(&format!("// @summary: {v}\n"));
    }
    if let Some(v) = &f.repository {
        s.push_str(&format!("// @repository: {v}\n"));
    }
    if let Some(v) = &f.keywords {
        s.push_str(&format!("// @keywords: {}\n", v.join(", ")));
    }
    if let Some(d) = &f.deprecated {
        let v = match d {
            Deprecated::Flag(b) => b.to_string(),
            Deprecated::Reason(r) => r.clone(),
        };
        s.push_str(&format!("// @deprecated: {v}\n"));
    }
    s
}

#[cfg(test)]
mod tests;
