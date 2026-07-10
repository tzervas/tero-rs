//! The L1 lexer (RFC-0006; DN-02). Hand-written, no dependencies (house style). Produces a token
//! stream or an explicit [`ParseError`] — never a silent skip of an unrecognized character.
//!
//! After RFC-0037 D1/D4 there is no `<` subtlety: type arguments use `[…]` and balanced-ternary
//! literals use the `0t…` prefix (lexed whole like `0b…`/`0x…`), so `<` is always the operator
//! [`Tok::LAngle`]. A `0t` with no trit glyph is an explicit error, never a silent empty literal.
//!
//! ## Comment capture
//!
//! The public entry [`lex_with_comments`] returns the same token stream as [`lex`] **plus** an
//! ordered [`Vec<Comment>`] containing every `//` comment in the source.  The plain [`lex`]
//! function is behavior-identical to before (comments still do not appear in the token stream).
//!
//! ### What `Comment::text` stores
//!
//! The `text` field holds the **full lexeme from `//` through (but not including) the terminating
//! `\n`** (or end-of-file).  The leading `//` is included verbatim; there is no trailing newline
//! or carriage-return in `text`.  This makes round-trip re-emission straightforward: write
//! `comment.text`, then a newline.  No content is omitted and no content is altered (G2 —
//! never-silent capture).

use crate::error::ParseError;
use crate::token::{keyword, Pos, Spanned, Tok};

/// A captured `//` line comment, produced by [`lex_with_comments`].
///
/// Guarantee: `Empirical` — the `text`/`line`/`col`/`trailing` fields are populated by
/// the lexer's own line/column counters, which are validated by the unit tests in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comment {
    /// The full lexeme from `//` through end-of-line content (verbatim, not including the
    /// terminating newline or carriage-return).  The leading `//` is always present.
    /// Example: for the source line `  // nodule: foo`, `text` is `"// nodule: foo"`.
    pub text: String,
    /// 1-based line number of the `//` opener.
    pub line: u32,
    /// 1-based column number of the `//` opener.
    pub col: u32,
    /// `true` iff at least one non-comment token was emitted on **this same source line**
    /// before the comment was encountered — i.e. this is an end-of-line (trailing) comment
    /// like `x => y  // why`.  `false` for a full-line (leading) comment.
    pub trailing: bool,
}

struct Lexer {
    chars: Vec<char>,
    i: usize,
    line: u32,
    col: u32,
    /// Comments collected so far, in source order.  Populated during `skip_trivia` calls
    /// **only when [`capture_comments`](Self::capture_comments) is set** (the `mycfmt` path).
    comments: Vec<Comment>,
    /// The source line on which the most-recently-emitted non-comment token started, or `0`
    /// if no token has been emitted yet.  Used to compute [`Comment::trailing`].
    last_token_line: u32,
    /// When `false` (the plain [`lex`] path), `//` comments are skipped without allocating a
    /// [`Comment`] — the common parse/check front-end pays no comment-capture cost. When `true`
    /// (the [`lex_with_comments`] / `mycfmt` path), each comment is captured into `comments`.
    capture_comments: bool,
}

/// Tokenize `src` into a [`Spanned`] stream terminated by [`Tok::Eof`].
///
/// Comments are **discarded** (behavior-identical to the original implementation).  This is the
/// front-end fast path: `//` comments are *skipped without allocating* a [`Comment`], so a
/// parse/check that never needs comments pays no capture cost (Copilot #397).  Use
/// [`lex_with_comments`] to obtain the comment side-table alongside the same token stream.
pub fn lex(src: &str) -> Result<Vec<Spanned>, ParseError> {
    run_lexer(src, false).map(|(toks, _comments)| toks)
}

/// Tokenize `src`, returning the [`Spanned`] token stream **and** an ordered [`Vec<Comment>`]
/// with every `//` comment in source order.
///
/// The token stream is byte-identical to what [`lex`] returns; no comment appears in the tokens.
/// No comment is silently dropped — every `//`-to-EOL run is captured (G2).
///
/// # Errors
///
/// Returns a [`ParseError`] on any lexically invalid input (e.g. an unrecognized character or a
/// malformed ternary literal) — the same conditions under which [`lex`] returns `Err`.
pub fn lex_with_comments(src: &str) -> Result<(Vec<Spanned>, Vec<Comment>), ParseError> {
    run_lexer(src, true)
}

/// Shared lexer driver. `capture_comments` selects the fast path ([`lex`], comments skipped without
/// allocation) vs the `mycfmt` path ([`lex_with_comments`], comments captured into the side-table).
/// The token stream is identical either way (comments never enter the token stream).
fn run_lexer(
    src: &str,
    capture_comments: bool,
) -> Result<(Vec<Spanned>, Vec<Comment>), ParseError> {
    let mut lx = Lexer {
        chars: src.chars().collect(),
        i: 0,
        line: 1,
        col: 1,
        comments: Vec::new(),
        last_token_line: 0,
        capture_comments,
    };
    let toks = lx.run()?;
    Ok((toks, lx.comments))
}

impl Lexer {
    fn pos(&self) -> Pos {
        Pos {
            line: self.line,
            col: self.col,
        }
    }

    /// Record a token emission so that subsequent comments on the same line are marked trailing.
    fn note_token_at(&mut self, line: u32) {
        self.last_token_line = line;
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.i + 1).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.i).copied()?;
        self.i += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn run(&mut self) -> Result<Vec<Spanned>, ParseError> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia();
            let pos = self.pos();
            let Some(c) = self.peek() else {
                // Eof does not update last_token_line — it carries no source line of its own.
                out.push(Spanned { tok: Tok::Eof, pos });
                return Ok(out);
            };
            let tok = match c {
                '(' => self.single(Tok::LParen),
                ')' => self.single(Tok::RParen),
                '{' => self.single(Tok::LBrace),
                '}' => self.single(Tok::RBrace),
                '[' => self.single(Tok::LBracket),
                ']' => self.single(Tok::RBracket),
                // `>` (gt) / `>>` (shr): comparison and right-shift operators (RFC-0025 §4.1
                // Tiers 8/4; M-745). Lexed whole — no nested-generic `>>` hazard now that type args
                // moved to `[…]` (RFC-0037 D1).
                '>' => self.lex_rangle(),
                // `@` is the guarantee-annotation glyph (`T @ Exact`), but `@std-sys` is the atomic
                // nodule-header FFI-floor marker (M-661): `-` is not an identifier char, so `@std-sys`
                // could never lex as `@` + an identifier — it must be recognized whole here. Only the
                // exact `@std-sys` is special; any other `@…` stays the bare `Tok::At`.
                '@' => self.lex_at(),
                ':' => self.single(Tok::Colon),
                ',' => self.single(Tok::Comma),
                // `;` — the DN-57 component/operation terminator (optional in v0; whitespace-free /
                // streamable source). `,` separates siblings within a component; `;` terminates one.
                ';' => self.single(Tok::Semi),
                '.' => self.single(Tok::Dot),
                // `|` is the sum-type constructor separator (`type T = A | B`) and the bitwise-`bor`
                // operator (RFC-0025 / M-705); `||` is the logical-`or` operator. The parser
                // disambiguates single `|` by position (type-decl separator vs expression operator).
                // (There is no `|`-separated pattern-alternation production in the v0 surface.)
                '|' => self.lex_pipe(),
                // `!` opens the effect annotation `!{ … }` (RFC-0014 §3.4; M-660) and is the unary
                // `not` operator at expression position (RFC-0025 / M-705); `!=` is the `ne`
                // operator. The parser accepts a signature `!` only before `{` (never a silent
                // accept, G2).
                '!' => self.lex_bang(),
                // `+` is the trait-bound separator (`T: A + B`; RFC-0019 §4.1) and the infix `add`
                // operator at expression position (RFC-0025 / M-705). It is also a trit glyph, but a
                // trit literal is only ever scanned *whole* from an opening `<` (in
                // `lex_angle_or_trit`), so a `+` reaching here is one of those two operator tokens.
                '+' => self.single(Tok::Plus),
                // `*` is the glob marker of a wildcard import `use a.b.*` (M-662) and the infix
                // `mul` operator at expression position (RFC-0025 / M-705); the parser disambiguates
                // by position.
                '*' => self.single(Tok::Star),
                // `/` is the infix `div` operator (RFC-0025 / M-705). `//` line comments are
                // consumed by `skip_trivia`, so a `/` reaching here is always the operator.
                '/' => self.single(Tok::Slash),
                // `%` (rem), `^` (xor): infix operators at expression position (RFC-0025 / M-705).
                '%' => self.single(Tok::Percent),
                '^' => self.single(Tok::Caret),
                // `&` (band) / `&&` (and): bitwise- and logical-and operators (RFC-0025 / M-705).
                '&' => self.lex_amp(),
                // `<` is operator-only (RFC-0037 D1): type-args moved to `[…]` and trit literals to
                // `0t…`, so a `<` is always a comparison/shift glyph now — `<` (lt) or, when
                // doubled, `<<` (shl) (RFC-0025 §4.1 Tiers 8/4; M-745).
                '<' => self.lex_langle(),
                '=' => self.lex_eq(),
                '-' => self.lex_dash(),
                '0' if self.peek2() == Some('b') => self.lex_binary(pos)?,
                // `0x…` is a byte-string literal (RFC-0032 D4, M-750), lexed whole like `0b…`. The
                // `0x` prefix is unambiguous: a bare `0` not followed by `b`/`x`/`t` is an int.
                '0' if self.peek2() == Some('x') => self.lex_hex_bytes(pos)?,
                // `0t…` is a balanced-ternary literal (RFC-0037 D4), lexed whole like `0b…`/`0x…`.
                '0' if self.peek2() == Some('t') => self.lex_trit(pos)?,
                // `"…"` is a textual string literal (M-910/M-911, kickoff `enb`): scanned whole,
                // with its minimal escape set decoded inline (see `lex_string`).
                '"' => self.lex_string(pos)?,
                c if c.is_ascii_digit() => self.lex_int(pos)?,
                c if is_ident_start(c) => self.lex_ident(),
                other => {
                    return Err(ParseError::new(
                        pos,
                        format!("unexpected character {other:?}"),
                    ))
                }
            };
            self.note_token_at(pos.line);
            out.push(Spanned { tok, pos });
        }
    }

    /// Lex an `@`: the atomic `@std-sys` nodule-header marker (M-661) if `@` is immediately followed
    /// by the literal `std-sys`, else the bare guarantee-annotation [`Tok::At`]. The match is on the
    /// exact 7-char tail `std-sys`; a longer identifier (`std-system`) is **not** matched (the char
    /// after `-sys` must not be an identifier continuation), so the special case stays maximally
    /// narrow and `@std` / `@Exact` are unaffected. No `unsafe`; a pure lookahead-and-consume.
    fn lex_at(&mut self) -> Tok {
        // Consume '@', then peek the exact `std-sys` tail without consuming unless it matches in full.
        self.bump();
        const MARKER: &[char] = &['s', 't', 'd', '-', 's', 'y', 's'];
        let matches_tail = MARKER
            .iter()
            .enumerate()
            .all(|(k, &want)| self.chars.get(self.i + k).copied() == Some(want));
        // It must be a *whole* word: the char after `std-sys` cannot continue an identifier
        // (so `@std-system` is NOT the marker — it stays `@` + ident, which then fails downstream).
        let next_after = self.chars.get(self.i + MARKER.len()).copied();
        let whole_word = next_after.is_none_or(|c| !is_ident_continue(c));
        if matches_tail && whole_word {
            for _ in 0..MARKER.len() {
                self.bump();
            }
            Tok::AtStdSys
        } else {
            Tok::At
        }
    }

    fn single(&mut self, tok: Tok) -> Tok {
        self.bump();
        tok
    }

    fn lex_eq(&mut self) -> Tok {
        self.bump(); // '='
        match self.peek() {
            Some('>') => {
                self.bump();
                Tok::FatArrow
            }
            // `==` is the infix `eq` operator (RFC-0025 / M-705); `=` stays the binder glyph.
            Some('=') => {
                self.bump();
                Tok::EqEq
            }
            _ => Tok::Eq,
        }
    }

    fn lex_dash(&mut self) -> Tok {
        self.bump(); // '-'
        if self.peek() == Some('>') {
            self.bump();
            // `->` is the **retired** return arrow (RFC-0037 D4 → `=>`); still lexed as [`Tok::Arrow`]
            // so the parser can emit a teaching reject rather than a confusing token error (G2).
            Tok::Arrow
        } else {
            // A bare `-` is the infix sub / unary neg operator (RFC-0025 / M-705); the parser
            // disambiguates binary from prefix by position.
            Tok::Minus
        }
    }

    /// `&` (band) or `&&` (logical and) — RFC-0025 / M-705.
    fn lex_amp(&mut self) -> Tok {
        self.bump(); // '&'
        if self.peek() == Some('&') {
            self.bump();
            Tok::AmpAmp
        } else {
            Tok::Amp
        }
    }

    /// `|` (sum-type constructor separator / `bor`) or `||` (logical or) — RFC-0025 / M-705.
    fn lex_pipe(&mut self) -> Tok {
        self.bump(); // '|'
        if self.peek() == Some('|') {
            self.bump();
            Tok::PipePipe
        } else {
            Tok::Pipe
        }
    }
    /// `<` (lt, RFC-0025 §4.1 Tier 8) / `<<` (shl, Tier 4) — M-745. The doubled form lexes whole;
    /// post-RFC-0037 D1 there is no type-argument `<…>` role, so `<<` is never two type-arg opens.
    fn lex_langle(&mut self) -> Tok {
        self.bump(); // '<'
        if self.peek() == Some('<') {
            self.bump();
            Tok::Shl
        } else {
            Tok::LAngle
        }
    }
    /// `>` (gt, RFC-0025 §4.1 Tier 8) / `>>` (shr, Tier 4) — M-745. The doubled form lexes whole;
    /// no nested-generic `>>` hazard now that type arguments use `[…]` (RFC-0037 D1).
    fn lex_rangle(&mut self) -> Tok {
        self.bump(); // '>'
        if self.peek() == Some('>') {
            self.bump();
            Tok::Shr
        } else {
            Tok::RAngle
        }
    }

    /// `!` (effect-set opener / unary not) or `!=` (ne) — RFC-0014 §3.4 / RFC-0025 / M-705.
    fn lex_bang(&mut self) -> Tok {
        self.bump(); // '!'
        if self.peek() == Some('=') {
            self.bump();
            Tok::BangEq
        } else {
            Tok::Bang
        }
    }

    /// Lex a balanced-ternary literal `0t…` (RFC-0037 D4), mirroring [`Self::lex_binary`]/
    /// [`Self::lex_hex_bytes`]: scan the trit glyphs (`+`/`0`/`-`) verbatim into the inner string
    /// (the same MSB-first `+0-` content the AST/parser expect). Never-silent (G2): an empty `0t`
    /// (no trit glyph) is an explicit [`ParseError`] naming the position — never a silently-empty
    /// literal. (The former `<…>` angle form is retired; `<` is now operator-only.)
    fn lex_trit(&mut self, pos: Pos) -> Result<Tok, ParseError> {
        self.bump(); // '0'
        self.bump(); // 't'
        let mut trits = String::new();
        while let Some(c) = self.peek() {
            if matches!(c, '+' | '-' | '0') {
                trits.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if trits.is_empty() {
            return Err(ParseError::new(
                pos,
                "balanced-ternary literal `0t` has no trits (expected at least one of `+`/`0`/`-`)"
                    .to_owned(),
            ));
        }
        Ok(Tok::TritLit(trits))
    }

    /// Lex a textual string literal `"…"` (M-910, kickoff `enb` Phase-I H1): scan to the closing
    /// `"`, decoding the **explicit, minimal escape set** inline — `\n \t \\ \" \0 \r` (ergonomic,
    /// not expressive; `\xNN` is deliberately NOT included — it would let a literal inject a
    /// non-UTF-8 byte into what is otherwise always-valid-UTF-8 text, so it is left for a follow-up
    /// with its own justification). `Tok::StrLit` carries the **decoded** content, mirroring
    /// [`Self::lex_hex_bytes`]'s "the lexer is the never-silent gate" role: escape errors are lexer
    /// errors, not deferred to elaboration.
    ///
    /// Never-silent (G2): an **unterminated** literal — EOF or a raw newline/CR reached before the
    /// closing `"` (raw newlines are not allowed inside a string; use `\n`), a **trailing `\`**
    /// before EOF, or an **unknown escape** (`\q`, say) — is an explicit [`ParseError`] naming the
    /// offending position; never a silently-truncated or half-escaped token.
    fn lex_string(&mut self, pos: Pos) -> Result<Tok, ParseError> {
        self.bump(); // opening '"'
        let mut out = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(ParseError::new(
                        pos,
                        "unterminated string literal (no closing `\"` before end of file)"
                            .to_owned(),
                    ));
                }
                Some('\n' | '\r') => {
                    return Err(ParseError::new(
                        pos,
                        "unterminated string literal (a raw newline/carriage-return is not \
                         allowed inside \"…\" — use \\n)"
                            .to_owned(),
                    ));
                }
                Some('"') => {
                    self.bump();
                    break;
                }
                Some('\\') => {
                    let esc_pos = self.pos();
                    self.bump(); // consume '\'
                    match self.peek() {
                        Some('n') => {
                            out.push('\n');
                            self.bump();
                        }
                        Some('t') => {
                            out.push('\t');
                            self.bump();
                        }
                        Some('\\') => {
                            out.push('\\');
                            self.bump();
                        }
                        Some('"') => {
                            out.push('"');
                            self.bump();
                        }
                        Some('0') => {
                            out.push('\0');
                            self.bump();
                        }
                        Some('r') => {
                            out.push('\r');
                            self.bump();
                        }
                        Some(other) => {
                            return Err(ParseError::new(
                                esc_pos,
                                format!(
                                    "unknown escape sequence `\\{other}` in string literal \
                                     (supported: \\n \\t \\\\ \\\" \\0 \\r)"
                                ),
                            ));
                        }
                        None => {
                            return Err(ParseError::new(
                                esc_pos,
                                "unterminated string literal (trailing `\\` before end of file)"
                                    .to_owned(),
                            ));
                        }
                    }
                }
                Some(c) => {
                    out.push(c);
                    self.bump();
                }
            }
        }
        Ok(Tok::StrLit(out))
    }

    fn lex_binary(&mut self, pos: Pos) -> Result<Tok, ParseError> {
        self.bump(); // '0'
        self.bump(); // 'b'
        let mut digits = String::new();
        // Track whether any actual binary digit (not just a `_` separator) was scanned: a
        // base-prefixed literal must carry a value. `0b` alone, or `0b_`, is a never-silent
        // lex error (G2) — the literal is parsed only when it has at least one `0`/`1`.
        let mut saw_digit = false;
        while let Some(c) = self.peek() {
            if c == '0' || c == '1' {
                saw_digit = true;
                digits.push(c);
                self.bump();
            } else if c == '_' {
                digits.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if !saw_digit {
            return Err(ParseError::new(
                pos,
                "binary literal `0b` has no digits (expected at least one `0` or `1`)".to_owned(),
            ));
        }
        Ok(Tok::BinLit(digits))
    }

    /// Lex a byte-string literal `0x…` (RFC-0032 D4, M-750), mirroring [`Self::lex_binary`]: scan
    /// hex digits + `_` separators verbatim into the inner string. Never-silent (G2): an empty `0x`
    /// (no hex digit), a non-hex digit, **or an odd number of hex digits** (a byte is two hex chars)
    /// is an explicit [`ParseError`] naming the offending position — never a silently-empty or
    /// half-byte token. The `_` separators are preserved but do not count toward the byte parity.
    fn lex_hex_bytes(&mut self, pos: Pos) -> Result<Tok, ParseError> {
        self.bump(); // '0'
        self.bump(); // 'x'
        let mut digits = String::new();
        // Count actual hex digits (not `_` separators) for the even-parity check; a base-prefixed
        // literal must carry at least one digit (`0x` / `0x_` alone is a never-silent lex error).
        let mut hex_count = 0usize;
        while let Some(c) = self.peek() {
            if c.is_ascii_hexdigit() {
                hex_count += 1;
                digits.push(c);
                self.bump();
            } else if c == '_' {
                digits.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if hex_count == 0 {
            return Err(ParseError::new(
                pos,
                "byte-string literal `0x` has no hex digits (expected at least one, even count)"
                    .to_owned(),
            ));
        }
        if !hex_count.is_multiple_of(2) {
            return Err(ParseError::new(
                pos,
                format!(
                    "byte-string literal `0x` has an odd hex-digit count ({hex_count}) — each byte \
                     is two hex chars (RFC-0032 D4); never a silent half-byte"
                ),
            ));
        }
        Ok(Tok::BytesLit(digits))
    }

    /// Lex a decimal number: an integer literal, or — when the digits continue with a fractional
    /// part (`.` followed by a digit) and/or an exponent (`e`/`E`) — a float literal (ADR-040 /
    /// M-897). The Int-disambiguation is **structural, never a guess**:
    ///
    /// - `1.5` / `0.0` — a `.` followed by a **digit** continues the number as a float. A `.` *not*
    ///   followed by a digit is left unconsumed (`1.` stays `Int(1)` + `Tok::Dot` — `.` remains the
    ///   path/field glyph, so a float always has digits on **both** sides of its dot; there is no
    ///   leading-dot `.5` form either, since a bare `.` never starts a number).
    /// - `1e10` / `2.5e-3` — an `e`/`E` immediately after the digits opens an exponent
    ///   (optional `+`/`-` sign, then **at least one** digit). Today `1e10` could only lex as
    ///   `Int(1)` + `Ident("e10")`, which no production accepts adjacently — so claiming the
    ///   exponent form introduces no grammar ambiguity.
    ///
    /// Never-silent (G2), mirroring [`Tok::Int`]'s out-of-range refusal: an exponent with no digit
    /// (`1e`, `1e+`) is an explicit [`ParseError`], and a literal whose **correctly-rounded**
    /// binary64 value is not finite (`1e999` — magnitude beyond f64::MAX rounds to ±inf) is an
    /// explicit out-of-range error, never a silent ±inf value (ADR-040 §2.4: a literal is a
    /// *conversion boundary*, so out-of-range is an explicit error — in-band IEEE specials arise
    /// only from *arithmetic*). Rounding posture (ADR-040 FLAG-3): the literal denotes the
    /// correctly-rounded (RNE) binary64 of its decimal text; the token carries the text verbatim
    /// and the single conversion happens at elaboration via `f64::from_str` (correct rounding is a
    /// Rust-std claim — `Declared`, pinned `Empirical` by the round-trip conformance corpus).
    fn lex_int(&mut self, pos: Pos) -> Result<Tok, ParseError> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        // Fractional part: only a `.` with a digit right behind it extends the number (see doc).
        let mut is_float = false;
        if self.peek() == Some('.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            s.push('.');
            self.bump(); // '.'
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    s.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        // Exponent part: `e`/`E`, optional sign, then at least one digit (else a never-silent error).
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            s.push(self.bump().expect("peeked exponent char exists"));
            if matches!(self.peek(), Some('+' | '-')) {
                s.push(self.bump().expect("peeked sign char exists"));
            }
            let mut saw_digit = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    saw_digit = true;
                    s.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
            if !saw_digit {
                return Err(ParseError::new(
                    pos,
                    format!(
                        "float literal `{s}` has an exponent with no digits (expected at least \
                         one digit after `e`/`E`, e.g. `1e10`, `2.5e-3`)"
                    ),
                ));
            }
        }
        if is_float {
            // The form is validated above, so `from_str` cannot fail; the parse here exists only to
            // range-check finiteness at the lex gate (defense in depth — never an `unwrap`).
            let x: f64 = s.parse().map_err(|_| {
                ParseError::new(pos, format!("malformed float literal: {s} (internal)"))
            })?;
            if !x.is_finite() {
                return Err(ParseError::new(
                    pos,
                    format!(
                        "float literal out of range: {s} (its correctly-rounded IEEE-754 binary64 \
                         value is not finite — magnitude exceeds ~1.8e308; ADR-040 §2.4)"
                    ),
                ));
            }
            return Ok(Tok::FloatLit(s));
        }
        s.parse::<i64>()
            .map(Tok::Int)
            .map_err(|_| ParseError::new(pos, format!("integer literal out of range: {s}")))
    }

    fn lex_ident(&mut self) -> Tok {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        keyword(&s).unwrap_or(Tok::Ident(s))
    }

    /// Skip whitespace and `//` line comments, capturing each comment into `self.comments`.
    ///
    /// Whitespace skipping is unchanged.  For every `//`-to-EOL run, the text (from `//` through
    /// the last non-newline character) is stored as a [`Comment`] in source order.  No comment
    /// is silently dropped (G2).
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek2() == Some('/') => {
                    // Record position of the `//` opener before consuming anything.
                    let comment_line = self.line;
                    let comment_col = self.col;
                    let trailing = self.last_token_line == comment_line;
                    // Build the comment text only on the capture path; on the plain `lex` path
                    // `text` stays `None`, so we advance past the comment with no `String` alloc
                    // and no `Comment` push (Copilot #397 — the front-end pays no comment cost).
                    let mut text = self.capture_comments.then(String::new);
                    while let Some(c) = self.peek() {
                        // Stop at the line terminator — break on `\r` too so a CRLF source does not
                        // leave a trailing `\r` in the comment text (the `\r\n` is then consumed by
                        // the whitespace arm). Keeps comment text `\r`-free + LF/CRLF round-trip
                        // parity, per the lexer's "no carriage-return" contract (Copilot #397).
                        if c == '\n' || c == '\r' {
                            break;
                        }
                        if let Some(t) = text.as_mut() {
                            t.push(c);
                        }
                        self.bump();
                    }
                    if let Some(text) = text {
                        self.comments.push(Comment {
                            text,
                            line: comment_line,
                            col: comment_col,
                            trailing,
                        });
                    }
                }
                _ => return,
            }
        }
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
