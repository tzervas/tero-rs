//! Shared **lexical span resolution** for the position-aware LSP providers (M-730): semantic
//! tokens ([`crate::semantic`]), hover ([`crate::hover`]), and go-to-definition
//! ([`crate::definition`]).
//!
//! Scope and honesty (`Declared`): this is a **purely lexical** layer. It runs the canonical L1
//! lexer ([`mycelium_l1::lexer::lex_with_comments`]) and resolves each token/comment to a
//! character span on a single source line. It has **no** semantic context — it does not know a
//! token's type, its scope, or its binding. The providers built on it therefore classify by token
//! kind only; they never claim type-aware inference (VR-5 / G2).
//!
//! ## Span resolution (no lexer duplication)
//! The L1 lexer records only each token's **start** [`Pos`](mycelium_l1::token::Pos), not its
//! length. Rather than re-implement lexeme scanning (a drift risk), this module recovers each
//! token's length from the **next lexical boundary**: every token in this lexer occupies exactly
//! one line, and the only thing between two adjacent lexical items is whitespace or a comment
//! (itself a boundary). So a token's text runs from its start column to the next item's start (or
//! end-of-line), with trailing whitespace trimmed against the real source line. This is exact for
//! ASCII source.
//!
//! ## Position encoding (UTF-16 stopgap)
//! LSP `Position.character` is **UTF-16 code units** by default; this module measures spans in
//! **Unicode scalar values**, which agree with UTF-16 only for ASCII. The L1 lexer accepts non-ASCII
//! in `//` comments, so rather than hand a standard client mis-counted offsets, [`lex_items`]
//! **refuses** a non-ASCII source — it returns no spans (hover/definition/tokens yield nothing for
//! that file), never a silent mis-offset (G2). The proper fix — UTF-16 length accounting or
//! position-encoding negotiation (`utf-8`/`utf-16`) — is the follow-up; this is the honest interim.

use mycelium_l1::lexer::lex_with_comments;
use mycelium_l1::token::Tok;

/// What a [`LexItem`] is: a real token, or a `//` line comment.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LexKind {
    /// A lexed token (keywords, identifiers, literals, operators, delimiters).
    Token(Tok),
    /// A `//` line comment (runs to end-of-line).
    Comment,
}

/// One lexical item resolved to a character span on a single source line. Columns are **1-based**
/// (matching [`mycelium_l1::token::Pos`]); LSP consumers subtract one to reach 0-based positions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LexItem {
    /// 1-based source line of the item's first character.
    pub line: u32,
    /// 1-based character column of the item's first character.
    pub col: u32,
    /// Length in Unicode scalar values (characters). Exact for ASCII; see module note for non-ASCII.
    pub len: u32,
    /// The token or comment this span covers.
    pub kind: LexKind,
}

/// Resolve every token and comment in `src` to a character span (see module note for the
/// boundary-recovery method).
///
/// **Never-silent (G2):** on a lexer error this returns an **empty** list rather than fabricating
/// spans — the parse error itself is reported on the diagnostics channel ([`crate::sync`]); this
/// layer simply has nothing it can honestly highlight. The `Tok::Eof` sentinel is excluded (it has
/// no source text).
pub(crate) fn lex_items(src: &str) -> Vec<LexItem> {
    // UTF-16 stopgap (never-silent, G2): LSP `Position.character` is UTF-16 code units by default,
    // but this module measures spans in Unicode scalar values. Those agree only while the source is
    // ASCII. The L1 lexer accepts non-ASCII in `//` comments, so a file with any non-ASCII byte
    // would yield offsets a standard LSP client mis-counts. Until proper position-encoding
    // negotiation (`utf-16`/`utf-8`) lands, REFUSE rather than return wrong ranges: a non-ASCII
    // source yields no spans (hover/definition/tokens return nothing for it), never a silent
    // mis-offset. ASCII sources are exact and unaffected.
    if !src.is_ascii() {
        return Vec::new();
    }
    let (toks, comments) = match lex_with_comments(src) {
        Ok(pair) => pair,
        Err(_) => return Vec::new(),
    };

    // Per-line character grids, for trailing-whitespace trimming. A trailing '\r' (CRLF) is dropped
    // so it never inflates a token's length.
    let line_chars: Vec<Vec<char>> = src
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l).chars().collect())
        .collect();
    let line_len = |line: u32| -> u32 {
        line_chars
            .get(line.saturating_sub(1) as usize)
            .map_or(0, |c| c.len() as u32)
    };

    // Merge tokens (minus Eof) and comments into one source-ordered list of raw starts.
    #[derive(Clone)]
    struct Raw {
        line: u32,
        col: u32,
        kind: LexKind,
    }
    let mut raws: Vec<Raw> = Vec::with_capacity(toks.len() + comments.len());
    for s in toks {
        if s.tok == Tok::Eof {
            continue;
        }
        raws.push(Raw {
            line: s.pos.line,
            col: s.pos.col,
            kind: LexKind::Token(s.tok),
        });
    }
    for c in comments {
        raws.push(Raw {
            line: c.line,
            col: c.col,
            kind: LexKind::Comment,
        });
    }
    raws.sort_by_key(|a| (a.line, a.col));

    let mut out = Vec::with_capacity(raws.len());
    for i in 0..raws.len() {
        let r = &raws[i];
        let len = match &r.kind {
            // A comment runs from its `//` opener to end-of-line.
            LexKind::Comment => line_len(r.line).saturating_sub(r.col.saturating_sub(1)),
            LexKind::Token(_) => {
                // The next boundary on this line is the next raw item if it shares the line, else
                // end-of-line.
                let raw_end_col = match raws.get(i + 1) {
                    Some(next) if next.line == r.line => next.col,
                    _ => line_len(r.line) + 1,
                };
                token_len(&line_chars, r.line, r.col, raw_end_col)
            }
        };
        if len == 0 {
            continue; // a zero-length span carries no useful highlight/hover target
        }
        out.push(LexItem {
            line: r.line,
            col: r.col,
            len,
            kind: r.kind.clone(),
        });
    }
    out
}

/// The trimmed character length of a token occupying `[col, raw_end_col)` (1-based, exclusive end)
/// on `line` of `line_chars`: the raw span minus any trailing whitespace (the trivia separating it
/// from the next boundary).
fn token_len(line_chars: &[Vec<char>], line: u32, col: u32, raw_end_col: u32) -> u32 {
    let Some(chars) = line_chars.get(line.saturating_sub(1) as usize) else {
        return 0;
    };
    let start = col.saturating_sub(1) as usize;
    let raw_end = (raw_end_col.saturating_sub(1) as usize).min(chars.len());
    if raw_end <= start {
        return 0;
    }
    let mut end = raw_end;
    while end > start && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    (end - start) as u32
}

/// The [`LexItem`] whose span covers the 0-based LSP position `(line0, char0)`, if any. The span is
/// the half-open character interval `[col-1, col-1+len)` on `line-1`. Returns `None` when the
/// position falls on whitespace, a delimiter gap, or outside the document — **never-silent**: an
/// absent target is an explicit `None`, never a fabricated span (G2).
pub(crate) fn item_at(items: &[LexItem], line0: u32, char0: u32) -> Option<&LexItem> {
    let line = line0.checked_add(1)?;
    items.iter().find(|it| {
        it.line == line
            && char0 >= it.col.saturating_sub(1)
            && char0 < it.col.saturating_sub(1) + it.len
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item_kinds(src: &str) -> Vec<(u32, u32, u32, LexKind)> {
        lex_items(src)
            .into_iter()
            .map(|i| (i.line, i.col, i.len, i.kind))
            .collect()
    }

    #[test]
    fn lexer_error_yields_no_spans_not_a_panic() {
        // G2: an un-lexable source is an empty span list (the error surfaces via diagnostics), never
        // a fabricated span or a panic.
        let items = lex_items("fn f() = §"); // `§` is not a valid L1 character
        assert!(items.is_empty());
    }

    #[test]
    fn non_ascii_source_refuses_spans_utf16_stopgap() {
        // A perfectly valid program whose ONLY non-ASCII is in a `//` comment: scalar-vs-UTF-16
        // offsets would disagree past it, so we refuse (no spans) rather than mis-offset (G2).
        let ascii = "fn f() = y  // ok\n";
        assert!(
            !lex_items(ascii).is_empty(),
            "ASCII source must still resolve spans"
        );
        let non_ascii = "fn f() = y  // café\n"; // non-ASCII 'é' in the comment
        assert!(
            lex_items(non_ascii).is_empty(),
            "non-ASCII source must yield no spans (UTF-16 stopgap), never mis-offset ranges"
        );
    }

    #[test]
    fn token_lengths_are_exact_for_adjacent_and_spaced_tokens() {
        // `a.b` packs three tokens with no spacing; `=>` is one two-char token.
        let items = item_kinds("nodule x\nfn a() = a.b\n");
        // Find the `nodule` keyword span: line 1, col 1, len 6.
        assert!(items
            .iter()
            .any(|(l, c, n, _)| *l == 1 && *c == 1 && *n == 6));
        // `a` and `.` and `b` are adjacent single chars on line 2.
        let line2: Vec<_> = items.iter().filter(|(l, ..)| *l == 2).collect();
        // The `.` between the two `a.b` identifiers is exactly one char.
        assert!(line2
            .iter()
            .any(|(_, _, n, k)| *n == 1 && matches!(k, LexKind::Token(Tok::Dot))));
    }

    #[test]
    fn trailing_comment_does_not_bleed_into_the_prior_token() {
        // `y` must measure one char even with a trailing comment on the same line.
        let items = lex_items("fn f() = y  // why\n");
        let y = items
            .iter()
            .find(|i| matches!(&i.kind, LexKind::Token(Tok::Ident(s)) if s == "y"))
            .expect("y token present");
        assert_eq!(y.len, 1, "trailing comment bled into the token length");
        // The comment itself is captured and spans to end-of-line.
        assert!(items
            .iter()
            .any(|i| i.kind == LexKind::Comment && i.len >= 5));
    }

    #[test]
    fn item_at_is_none_off_token_and_some_on_token() {
        let src = "fn f() = y\n";
        let items = lex_items(src);
        // Position on the `fn` keyword (line0=0, char0=0) hits a token.
        assert!(item_at(&items, 0, 0).is_some());
        // A column past the end of line 1 is whitespace/void → None (never-silent).
        assert!(item_at(&items, 0, 50).is_none());
    }
}
