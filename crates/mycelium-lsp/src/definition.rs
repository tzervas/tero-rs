//! **Go-to-definition provider** (M-730; `textDocument/definition`).
//!
//! Resolves an identifier under the cursor to a declaration **within the same document**.
//!
//! Scope and honesty (`Declared`): this is a **lexical, single-document** resolver. It collects the
//! names introduced by the declaration forms `fn <name>`, `type <name>`, and `trait <name>` (the
//! lexer pattern `<keyword> <Ident>`), and navigates an identifier use to the matching declaration.
//! It does **not** cross nodule/phylum boundaries, resolve imports, or do scope/shadowing analysis —
//! a client should treat it as best-effort lexical navigation, not a semantic resolver (VR-5).
//! **Never-silent (G2):** an identifier with no in-document declaration yields a `null` result, never
//! a fabricated location.

use serde_json::{json, Value};

use mycelium_l1::token::Tok;

use crate::span::{item_at, lex_items, LexItem, LexKind};

/// Build the `textDocument/definition` result for `src` at the 0-based position `(line, character)`,
/// for a document identified by `uri`.
///
/// Returns an LSP `Location` (`{ "uri", "range" }`) pointing at the in-document declaration of the
/// identifier under the cursor, or [`Value::Null`] when the cursor is not on an identifier or the
/// name has no `fn`/`type`/`trait` declaration in this document.
#[must_use]
pub fn definition(uri: &str, src: &str, line: u32, character: u32) -> Value {
    let items = lex_items(src);
    let Some(target) = item_at(&items, line, character) else {
        return Value::Null;
    };
    let LexKind::Token(Tok::Ident(name)) = &target.kind else {
        return Value::Null; // only identifiers have a definition to jump to
    };
    let Some(decl) = declarations(&items).into_iter().find(|(n, _)| n == name) else {
        return Value::Null; // never-silent: no fabricated location for an unresolved name
    };
    let (_, span) = decl;
    let start = span.col.saturating_sub(1);
    json!({
        "uri": uri,
        "range": {
            "start": { "line": span.line.saturating_sub(1), "character": start },
            "end": { "line": span.line.saturating_sub(1), "character": start + span.len },
        },
    })
}

/// The `(name, declaration-span)` pairs introduced in `items` by the `fn`/`type`/`trait` declaration
/// forms — the lexer adjacency `<decl-keyword> <Ident>`. First declaration wins on a duplicate name
/// (a redeclaration is a checker concern, not a navigation one).
fn declarations(items: &[LexItem]) -> Vec<(String, LexItem)> {
    let mut out: Vec<(String, LexItem)> = Vec::new();
    for win in items.windows(2) {
        let is_decl_kw = matches!(
            &win[0].kind,
            LexKind::Token(Tok::Fn) | LexKind::Token(Tok::Type) | LexKind::Token(Tok::Trait)
        );
        if !is_decl_kw {
            continue;
        }
        if let LexKind::Token(Tok::Ident(name)) = &win[1].kind {
            if !out.iter().any(|(n, _)| n == name) {
                out.push((name.clone(), win[1].clone()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos_of(src: &str, line_idx: usize, needle: &str) -> u32 {
        src.lines()
            .nth(line_idx)
            .unwrap()
            .find(needle)
            .expect("needle present") as u32
    }

    #[test]
    fn jumps_from_a_use_to_the_fn_declaration() {
        // `g` is declared on line 1 and used on line 2; definition() jumps to the declaration.
        let src = "fn g() -> Binary{8} = 0b0\nfn h() = g()\n";
        let use_col = pos_of(src, 1, "g()"); // the call site on line 2
        let loc = definition("mem://d.myc", src, 1, use_col);
        assert_eq!(loc["uri"], "mem://d.myc");
        assert_eq!(loc["range"]["start"]["line"], 0, "declaration is on line 1");
        // The declaration `g` sits right after `fn ` on line 1.
        assert_eq!(loc["range"]["start"]["character"], 3);
        assert_eq!(loc["range"]["end"]["character"], 4);
    }

    #[test]
    fn resolves_type_and_trait_declarations() {
        let src = "type Shape = Binary{8}\nfn area(s: Shape) = 0b0\n";
        let use_col = pos_of(src, 1, "Shape");
        let loc = definition("mem://d.myc", src, 1, use_col);
        assert_eq!(loc["range"]["start"]["line"], 0);
    }

    #[test]
    fn unknown_name_is_null_never_fabricated() {
        // G2: an identifier with no in-document declaration yields null, not a guessed location.
        let src = "fn h() = undeclared()\n";
        let use_col = pos_of(src, 0, "undeclared");
        assert_eq!(definition("mem://d.myc", src, 0, use_col), Value::Null);
    }

    #[test]
    fn cursor_not_on_an_identifier_is_null() {
        let src = "fn g() = 0b0\n";
        // On the `fn` keyword — not an identifier use.
        assert_eq!(definition("mem://d.myc", src, 0, 0), Value::Null);
    }
}
