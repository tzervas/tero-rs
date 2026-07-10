//! **Semantic-tokens provider** (M-730; `textDocument/semanticTokens/full`).
//!
//! Classifies the lexical token stream into LSP semantic-token types and emits the protocol's
//! relative-delta encoding. The legend is the **LSP layer of the ratified RFC-0026 §3.2 scope-name
//! table** (Accepted), **plus one additive extension** (M-975, see the [`TOKEN_TYPES`] doc): the
//! standard, *unsuffixed* LSP token types
//! `keyword`/`type`/`enumMember`/`number`/`operator`/`comment`/`variable`/`string` the table maps
//! each lexer bucket to (e.g. the guarantee-strength bucket → `enumMember`, the substrate/scalar
//! types → `type`). This is the broadest-coverage layer — it classifies comments, numbers,
//! operators, strings, and identifiers as well as the keyword buckets, with **every** lexer token
//! kind landing on a legend entry or the documented delimiter/`Eof` no-highlight set (M-975: the
//! `classify` match is exhaustive, no wildcard fallthrough — see its doc comment). The TextMate +
//! tree-sitter layers of the same table are generated under `tools/grammar/` (M-731) from the same
//! lexer `keyword()`; per §3.2 those layers cover a subset today (the tree-sitter scaffold captures
//! only the four word buckets).
//!
//! Scope and honesty (`Declared`): classification is **purely lexical/token-kind** — every
//! identifier is `variable` because the lexer cannot tell a function name from a binding without
//! semantic context. A client should present these as syntax colors, never as type-aware
//! classification (VR-5). A lexer error yields an **empty** token stream (the parse error surfaces
//! on the diagnostics channel), never a fabricated highlight (G2).

use serde_json::{json, Value};

use mycelium_l1::token::Tok;

use crate::span::{lex_items, LexItem, LexKind};

/// The semantic-token **type legend**, in index order. The encoded stream's `tokenType` field is an
/// index into this list (LSP §`semanticTokens`). These are the standard LSP token types the ratified
/// RFC-0026 §3.2 table maps each lexer bucket to (the TextMate/tree-sitter names of the same table
/// live in `tools/grammar/`), **plus `string`** (index 8, M-975): RFC-0026 §3.2's table predates the
/// textual-string-literal token ([`Tok::StrLit`], M-910/M-911) and has no bucket for it. `string` is
/// still a **standard, unsuffixed** LSP semantic-token type (the same naming convention the ratified
/// table already follows), added here **additively** (appended, not renumbering any existing index)
/// so this classifier can honestly tag `StrLit` rather than silently dropping it (G2). This mirrors
/// the §3.2 coverage note's own precedent: `operator`/`identifier` already ship **LSP-only**, ahead
/// of the other layers. FLAG (Declared, not yet ratified): a future RFC-0026 amendment/supersession
/// should fold `string` into the normative table so the TextMate/tree-sitter layers can pick it up too.
pub const TOKEN_TYPES: &[&str] = &[
    "keyword",    // 0 — reserved words (declaration + control + runtime vocabulary)
    "type",       // 1 — substrate/representation types (Binary/Ternary/Dense/VSA/…) + scalars
    "enumMember", // 2 — guarantee-strength lattice members (Exact/Proven/Empirical/Declared)
    "function", // 3 — (reserved in the legend; lexical classification cannot assign it — see note)
    "variable", // 4 — identifiers (lexical: every name, function or binding alike)
    "number",   // 5 — binary / balanced-ternary / integer / byte-string / float literals
    "operator", // 6 — arithmetic/logical/comparison/shift/annotation operators and arrows
    "comment",  // 7 — `//` line comments
    "string",   // 8 — textual string literals (`StrLit`; M-975 legend extension, see doc above)
];

/// LSP semantic-token type indices (into [`TOKEN_TYPES`]). `pub(crate)` (not public API) so the
/// in-crate test module (`src/tests/semantic.rs`) gets white-box access without exposing these to
/// downstream crates.
pub(crate) const T_KEYWORD: u32 = 0;
pub(crate) const T_TYPE: u32 = 1;
pub(crate) const T_ENUM_MEMBER: u32 = 2;
pub(crate) const T_VARIABLE: u32 = 4;
pub(crate) const T_NUMBER: u32 = 5;
pub(crate) const T_OPERATOR: u32 = 6;
pub(crate) const T_COMMENT: u32 = 7;
pub(crate) const T_STRING: u32 = 8;

/// The `legend` advertised in the server's `semanticTokensProvider` capability: the type list above
/// and an empty modifier list (no modifiers are emitted — honest about scope).
#[must_use]
pub fn semantic_tokens_legend() -> Value {
    json!({
        "tokenTypes": TOKEN_TYPES,
        "tokenModifiers": [],
    })
}

/// Classify one lexical item to its [`TOKEN_TYPES`] index, or `None` for items that carry no
/// highlight (delimiters, `Eof`). Delimiters (`()[]{}`, `:` `,` `.`) are intentionally unclassified:
/// editors colour them via the grammar, not semantic tokens. `<`/`>` are **not** in that set — since
/// RFC-0037 D1 moved type-argument lists to `[…]`, [`Tok::LAngle`]/[`Tok::RAngle`] are pure infix
/// comparison operators (`lt`/`gt`), so they classify with the other operators below (never silently
/// dropped, G2).
///
/// **Never-silent completeness (M-975):** every [`Tok`] variant is either matched below or falls into
/// the documented delimiter/`Eof` `None` arm — there is no wildcard catch-all arm, so the match is
/// exhaustive over every [`Tok`] variant the compiler knows about: a future lexer token that isn't
/// added here fails to **compile**, rather than silently falling through unhighlighted (G2).
///
/// `pub(crate)` (not public API) so the in-crate test module (`src/tests/semantic.rs`) can exercise
/// it directly.
pub(crate) fn classify(kind: &LexKind) -> Option<u32> {
    let tok = match kind {
        LexKind::Comment => return Some(T_COMMENT),
        LexKind::Token(t) => t,
    };
    let idx = match tok {
        // Declaration + control + runtime-vocabulary keywords (RFC-0026 §3.2: control vs.
        // declaration keywords are deliberately NOT split — one `keyword` bucket for all of it).
        Tok::Nodule
        | Tok::Phylum
        | Tok::Colony
        | Tok::Hypha
        | Tok::Fuse
        | Tok::Mesh
        | Tok::Graft
        | Tok::Cyst
        | Tok::Xloc
        | Tok::Forage
        | Tok::Backbone
        | Tok::Tier
        | Tok::Reclaim
        | Tok::Consume
        | Tok::Grow
        | Tok::Derive
        | Tok::Use
        | Tok::Pub
        | Tok::Type
        | Tok::Trait
        | Tok::Impl
        | Tok::Fn
        | Tok::Matured
        | Tok::Thaw
        | Tok::Let
        | Tok::In
        | Tok::If
        | Tok::Then
        | Tok::Else
        | Tok::Match
        | Tok::For
        | Tok::Swap
        | Tok::Default
        | Tok::Paradigm
        | Tok::With
        | Tok::Wild
        | Tok::Spore
        | Tok::To
        | Tok::Policy
        | Tok::Lambda
        | Tok::Object
        | Tok::Via
        | Tok::Lower => T_KEYWORD,
        // Substrate / representation types and scalars — incl. the M-915 short repr-keywords
        // (`bin`/`tern`/`emb`/`hvec`, which elaborate identically to their long forms) and the
        // RFC-0032/ADR-040 first-class repr-type keywords `Seq`/`Bytes`/`Float`.
        Tok::Binary
        | Tok::Ternary
        | Tok::Dense
        | Tok::Vsa
        | Tok::BinShort
        | Tok::TernShort
        | Tok::EmbShort
        | Tok::HvecShort
        | Tok::Seq
        | Tok::Bytes
        | Tok::Float
        | Tok::Substrate
        | Tok::Sparse
        | Tok::Scalar(_) => T_TYPE,
        // Guarantee-strength lattice members.
        Tok::Strength(_) => T_ENUM_MEMBER,
        // Identifiers — lexical only (no function/variable distinction available; see module note).
        Tok::Ident(_) => T_VARIABLE,
        // Numeric-shaped literals: binary/balanced-ternary/decimal-int (existing), plus the
        // byte-string (`0x…`) and decimal-float literals (M-975) — all fixed-form digit literals,
        // matching the RFC-0026 §3.2 "numeric" bucket → LSP `number`.
        Tok::BinLit(_) | Tok::TritLit(_) | Tok::Int(_) | Tok::BytesLit(_) | Tok::FloatLit(_) => {
            T_NUMBER
        }
        // A textual string literal (`"…"`) is not numeric — it gets its own `string` legend entry
        // (M-975; see the [`TOKEN_TYPES`] doc comment for why this extends beyond the current
        // RFC-0026 §3.2 table).
        Tok::StrLit(_) => T_STRING,
        // Operators / annotations / arrows / comparisons / shifts.
        Tok::Plus
        | Tok::Minus
        | Tok::Star
        | Tok::Slash
        | Tok::Percent
        | Tok::Caret
        | Tok::Amp
        | Tok::AmpAmp
        | Tok::Eq
        | Tok::EqEq
        | Tok::Arrow
        | Tok::FatArrow
        | Tok::Bang
        | Tok::BangEq
        | Tok::Pipe
        | Tok::PipePipe
        | Tok::At
        | Tok::AtStdSys
        | Tok::LAngle
        | Tok::RAngle
        | Tok::Shl
        | Tok::Shr => T_OPERATOR,
        // Delimiters (`()[]{}`, `:` `,` `.`) and `Eof` carry no semantic-token highlight — an
        // explicit, documented design choice (see the function doc comment), not a silent drop.
        Tok::LParen
        | Tok::RParen
        | Tok::LBrace
        | Tok::RBrace
        | Tok::LBracket
        | Tok::RBracket
        | Tok::Colon
        | Tok::Comma
        | Tok::Semi
        | Tok::Dot
        | Tok::Eof => return None,
    };
    Some(idx)
}

/// Build the `textDocument/semanticTokens/full` result for `src`: the LSP relative-delta encoding
/// (`{ "data": [deltaLine, deltaStartChar, length, tokenType, tokenModifiers, …] }`).
///
/// The encoding is relative to the previous emitted token (LSP §`semanticTokens` "Integer Encoding"):
/// `deltaLine` is the line gap; `deltaStartChar` is the column gap when on the same line, else the
/// absolute 0-based start column; `length`/`tokenType` are the item's char length and legend index;
/// `tokenModifiers` is always `0`. Items are emitted in source order (already sorted by line/col).
/// Unclassified items (delimiters) are skipped. **Never-silent:** an un-lexable document yields
/// `{ "data": [] }` (the diagnostics channel reports the error), never a fabricated stream (G2).
#[must_use]
pub fn semantic_tokens_full(src: &str) -> Value {
    json!({ "data": encode(&lex_items(src)) })
}

/// The flat `u32` delta-stream for `items` (the body of [`semantic_tokens_full`], exposed for tests).
#[must_use]
pub(crate) fn encode(items: &[LexItem]) -> Vec<u32> {
    let mut data = Vec::new();
    let mut prev_line = 0u32;
    let mut prev_col0 = 0u32; // 0-based start column of the previous emitted token
    for it in items {
        let Some(ttype) = classify(&it.kind) else {
            continue;
        };
        let line0 = it.line.saturating_sub(1);
        let col0 = it.col.saturating_sub(1);
        let delta_line = line0 - prev_line;
        let delta_col = if delta_line == 0 {
            col0 - prev_col0
        } else {
            col0
        };
        data.extend_from_slice(&[delta_line, delta_col, it.len, ttype, 0]);
        prev_line = line0;
        prev_col0 = col0;
    }
    data
}
