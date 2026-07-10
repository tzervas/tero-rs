//! **Hover provider** (M-730; `textDocument/hover`).
//!
//! Resolves the token under the cursor to a grounded, human-readable description.
//!
//! Scope and honesty (`Declared` / never-upgraded — VR-5): hover reports only what is **lexically
//! knowable**. For a keyword, a substrate/representation type, or a guarantee-strength token it
//! returns a real, grounded description. For an **identifier** it cannot honestly report a type or
//! guarantee tag — there is no type checker behind this surface yet — so it returns the lexical
//! role and **explicitly flags** that type/guarantee inference is not available, rather than
//! fabricating one (the DoD's "type + guarantee for every symbol" is the north star; this is the
//! honest subset that exists today, flagged not faked). **Never-silent (G2):** a position that is
//! not on a token yields a `null` hover (no result), never an invented one.

use serde_json::{json, Value};

use mycelium_l1::token::{ScalarTok, StrengthTok, Tok};

use crate::span::{item_at, lex_items, LexKind};

/// Build the `textDocument/hover` result for `src` at the 0-based LSP position `(line, character)`.
///
/// Returns an LSP `Hover` (`{ "contents": { kind: "markdown", value }, "range" }`) when the cursor
/// is on a describable token, else [`Value::Null`] (LSP's "no hover"). The `range` is the token's
/// exact character span, so the client can highlight it.
#[must_use]
pub fn hover(src: &str, line: u32, character: u32) -> Value {
    let items = lex_items(src);
    let Some(it) = item_at(&items, line, character) else {
        return Value::Null;
    };
    let Some(markdown) = describe(&it.kind) else {
        return Value::Null;
    };
    let start = it.col.saturating_sub(1);
    json!({
        "contents": { "kind": "markdown", "value": markdown },
        "range": {
            "start": { "line": line, "character": start },
            "end": { "line": line, "character": start + it.len },
        },
    })
}

/// A grounded markdown description for a lexical item, or `None` when there is nothing honest to say
/// (delimiters, operators with no teaching value).
fn describe(kind: &LexKind) -> Option<String> {
    let tok = match kind {
        LexKind::Comment => return Some("`//` line comment".to_owned()),
        LexKind::Token(t) => t,
    };
    let body = match tok {
        // --- guarantee-strength lattice (the honesty rule, VR-5) ---
        Tok::Strength(s) => {
            let (name, gloss) = match s {
                StrengthTok::Exact => (
                    "Exact",
                    "the result is exact — bit-for-bit correct with no loss (strongest tier).",
                ),
                StrengthTok::Proven => (
                    "Proven",
                    "backed by a theorem whose side-conditions are *checked*; allowed only with that checked basis.",
                ),
                StrengthTok::Empirical => (
                    "Empirical",
                    "established by trials/measurement, not a proof — honest about its evidential basis.",
                ),
                StrengthTok::Declared => (
                    "Declared",
                    "asserted by the author, always flagged — the weakest, default-honest tier.",
                ),
            };
            format!(
                "**`{name}`** — guarantee strength\n\nLattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`: {gloss}\n\nA tag is never *upgraded* without a checked basis (VR-5)."
            )
        }
        // --- substrate / representation types ---
        Tok::Binary => ty("Binary", "the binary value-semantics substrate (bit-packed)."),
        Tok::Ternary => ty("Ternary", "the balanced-ternary substrate (trits: −, 0, +)."),
        Tok::Dense => ty("Dense", "the dense numeric substrate (packed scalars)."),
        Tok::Vsa => ty("VSA", "the vector-symbolic-architecture substrate (hypervectors)."),
        Tok::Substrate => ty("Substrate", "the abstract substrate kind (representation family)."),
        Tok::Sparse => ty("Sparse", "the sparse representation modifier."),
        Tok::Scalar(s) => {
            let n = match s {
                ScalarTok::F16 => "F16",
                ScalarTok::Bf16 => "BF16",
                ScalarTok::F32 => "F32",
                ScalarTok::F64 => "F64",
            };
            ty(n, "a floating-point scalar element type.")
        }
        // --- declaration / control keywords (the high-value ones get a gloss) ---
        Tok::Nodule => kw("nodule", "opens a static unit (the basic module of a program)."),
        Tok::Phylum => kw("phylum", "a library/package (reserved-not-active)."),
        Tok::Colony => kw("colony", "a runtime grouping of hyphae (reserved)."),
        Tok::Use => kw("use", "imports names from another nodule."),
        Tok::Pub => kw("pub", "marks a cross-nodule export."),
        Tok::Type => kw("type", "declares a type."),
        Tok::Trait => kw("trait", "declares a trait (behavioural interface)."),
        Tok::Impl => kw("impl", "implements a trait for a type (ratified; not yet active)."),
        Tok::Fn => kw("fn", "declares a function."),
        Tok::Thaw => kw("thaw", "marks a function as thaw-able (`thaw fn …`)."),
        Tok::Matured => kw("matured", "maturity attribute (header form `// @matured: …`)."),
        Tok::Let => kw("let", "binds a value in an expression (`let x = … in …`)."),
        Tok::In => kw("in", "the body separator of a `let … in …`."),
        Tok::If => kw("if", "a conditional expression (`if … then … else …`)."),
        Tok::Then => kw("then", "the consequent of an `if`."),
        Tok::Else => kw("else", "the alternative of an `if`."),
        Tok::Match => kw("match", "pattern-matches a value."),
        Tok::For => kw("for", "an iteration form."),
        Tok::Swap => kw("swap", "an explicit, never-silent representation change."),
        Tok::Spore => kw("spore", "the deployable/published artifact (ADR-013)."),
        Tok::Paradigm => kw("paradigm", "selects a computation paradigm."),
        Tok::Policy => kw("policy", "a reified selection/conversion policy."),
        Tok::Default => kw("default", "a default arm/selection."),
        Tok::With => kw("with", "attaches a clause (e.g. a policy)."),
        Tok::Wild => kw("wild", "a wildcard."),
        Tok::To => kw("to", "the target of a `swap … to …`."),
        // Reserved runtime-vocabulary words (not yet active): one honest note covers them.
        Tok::Hypha
        | Tok::Fuse
        | Tok::Mesh
        | Tok::Graft
        | Tok::Cyst
        | Tok::Xloc
        | Tok::Forage
        | Tok::Backbone
        | Tok::Tier
        | Tok::Reclaim => {
            "**reserved keyword** — a ratified runtime-vocabulary word that lexes as a keyword (so it is never a silent identifier, G2) but has no active L1 construct yet (DN-03/DN-06).".to_owned()
        }
        // --- identifiers: honest about the absence of type/guarantee inference ---
        Tok::Ident(name) => format!(
            "**`{name}`** — identifier\n\n*Lexical hover only:* no type or guarantee tag is shown — type/guarantee inference is not yet wired into this surface, and a fabricated one would violate the honesty rule (VR-5). Use go-to-definition to navigate to its declaration."
        ),
        // --- literals ---
        Tok::BinLit(_) => "a **binary literal** (`0b…`).".to_owned(),
        Tok::TritLit(_) => "a **balanced-ternary literal** (`<…>`, trits −/0/+).".to_owned(),
        Tok::Int(_) => "an **integer literal**.".to_owned(),
        // Delimiters / operators: nothing worth a hover panel.
        _ => return None,
    };
    Some(body)
}

/// A **keyword** hover body: a bold lexeme tag plus a one-line gloss. Use [`ty`] for substrate /
/// scalar *types* so the role label stays accurate (a `Binary`/`F32` token is a type, not a keyword).
fn kw(name: &str, gloss: &str) -> String {
    format!("**`{name}`** — keyword\n\n{gloss}")
}

/// A **type** hover body (substrate / representation / scalar element types — `Binary`, `Ternary`,
/// `Dense`, `VSA`, `Substrate`, `Sparse`, `F32`, …): same shape as [`kw`] but the honest role label.
fn ty(name: &str, gloss: &str) -> String {
    format!("**`{name}`** — type\n\n{gloss}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 0-based column of the first occurrence of `needle` in the single-line-ish `src`.
    fn col_of(src: &str, needle: &str) -> u32 {
        let line = src.lines().next().unwrap();
        line.find(needle).expect("needle present") as u32
    }

    #[test]
    fn hover_on_keyword_is_grounded() {
        let src = "fn f() = 0b0\n";
        let h = hover(src, 0, 0); // on `fn`
        let v = h["contents"]["value"].as_str().unwrap();
        assert!(v.contains("fn"), "{v}");
        assert!(v.contains("function"), "{v}");
        // A real keyword is labelled "keyword", not "type".
        assert!(v.contains("— keyword"), "{v}");
        // The range covers the two-char `fn`.
        assert_eq!(h["range"]["end"]["character"], 2);
    }

    #[test]
    fn hover_on_type_is_labeled_type_not_keyword() {
        // A substrate/representation type (`Binary`) is a TYPE, not a keyword — the role label must
        // say so (the kw()/ty() split). Hover on `Binary` in the return position.
        let src = "fn f() -> Binary{8} = 0b0\n";
        let c = src.find("Binary").unwrap() as u32;
        let h = hover(src, 0, c);
        let v = h["contents"]["value"].as_str().unwrap();
        assert!(v.contains("Binary"), "{v}");
        assert!(
            v.contains("— type"),
            "type token must be labelled 'type': {v}"
        );
        assert!(
            !v.contains("— keyword"),
            "type token must NOT be labelled 'keyword': {v}"
        );
    }

    #[test]
    fn hover_on_strength_explains_the_lattice() {
        let src = "fn f() -> Binary{8} @ Declared = 0b0\n";
        let c = col_of(src, "Declared");
        let v = hover(src, 0, c);
        let s = v["contents"]["value"].as_str().unwrap();
        assert!(s.contains("guarantee strength"), "{s}");
        assert!(s.contains("VR-5"), "{s}");
    }

    #[test]
    fn hover_on_identifier_refuses_to_fabricate_a_type() {
        // VR-5: an identifier hover must NOT claim a type/guarantee it cannot derive.
        let src = "fn myfunc() = 0b0\n";
        let c = col_of(src, "myfunc");
        let v = hover(src, 0, c);
        let s = v["contents"]["value"].as_str().unwrap();
        assert!(s.contains("myfunc"), "{s}");
        assert!(s.contains("honesty rule") || s.contains("VR-5"), "{s}");
    }

    #[test]
    fn hover_off_any_token_is_null_never_fabricated() {
        // G2: a position on whitespace yields a null hover, never a guessed one.
        let src = "fn  f()\n";
        let v = hover(src, 0, 2); // the gap between `fn` and `f`
        assert_eq!(v, Value::Null);
    }
}
