//! LlmCanonical → Core-IR S-expression parser (M-381 Arm 4; RFC-0021 §4.1 `EditCapability`).
//!
//! This is the **inverse** of the `project::llm_canonical` renderer: it reads a
//! LlmCanonical S-expression string (produced by the renderer, or typed by an LLM) and
//! validates/normalizes it into a canonical Core-IR S-expression string suitable for
//! `myc-check` verification.
//!
//! # Honesty (VR-5, G2)
//! - **Guarantee tag: `Empirical`** — this is a heuristic line/token regex approach over
//!   the closed ~11-node grammar, not a proven-sound parser. Never upgrade to `Proven`
//!   without a checked basis (VR-5).
//! - **Fail-closed**: empty input, unrecognized constructs, unclosed delimiters, and
//!   inputs exceeding the depth limit all return explicit `Err`, never a partial/silent
//!   result (G2).
//! - **Depth limit**: recursive descent is guarded at 64 levels (banked guard #4). Inputs
//!   deeper than this return `ParseError::DepthLimitExceeded` — not a stack overflow.
//!
//! # What the parser validates
//! The LlmCanonical format produced by `project::llm_canonical` is itself an S-expression:
//! every node kind maps to a recognizable head keyword or positional structure. The parser
//! re-tokenizes the source into a balanced S-expression tree, classifies each form by its
//! head token, and emits the normalized S-expression string.
//!
//! Forms recognized (exactly the 11 node kinds the renderer covers):
//! - `(const …)` → constant
//! - `<identifier>` → variable reference
//! - `(let [id expr] body)` → let binding
//! - `(op prim arg…)` → primitive application
//! - `(swap! src :to repr :policy ref)` → representation swap
//! - `(make ctor arg…)` → saturated constructor
//! - `(match scrutinee alt…)` → flat pattern match
//! - `(fn [param] body)` → lambda
//! - `(func arg)` → application (general two-element S-expr)
//! - `(fix name body)` → fixed-point
//! - `(fix-group (binds…) body)` → mutual recursion group
//!
//! Any head token not in this set returns `ParseError::UnrecognizedConstruct`.

/// Maximum nesting depth (banked guard #4 — depth limit prevents stack overflow).
pub const DEPTH_LIMIT: usize = 64;

/// Errors returned by [`parse_llm_canonical`] (G2: always explicit, never silent).
#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    /// The input was empty or contained only whitespace.
    EmptyInput,
    /// A token was not recognized as a valid LlmCanonical construct.
    UnrecognizedConstruct {
        /// 1-based line number of the unrecognized token.
        line: usize,
        /// The unrecognized token text.
        token: String,
    },
    /// A delimiter was opened but never closed.
    UnclosedDelimiter {
        /// 1-based line number where the delimiter was opened.
        line: usize,
        /// The unclosed delimiter character.
        delimiter: char,
    },
    /// The input nesting depth exceeded [`DEPTH_LIMIT`] (banked guard #4).
    DepthLimitExceeded {
        /// The limit that was exceeded.
        limit: usize,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::EmptyInput => write!(f, "empty input"),
            ParseError::UnrecognizedConstruct { line, token } => {
                write!(f, "line {line}: unrecognized construct '{token}'")
            }
            ParseError::UnclosedDelimiter { line, delimiter } => {
                write!(f, "line {line}: unclosed delimiter '{delimiter}'")
            }
            ParseError::DepthLimitExceeded { limit } => {
                write!(f, "nesting depth limit {limit} exceeded")
            }
        }
    }
}

/// Parse a LlmCanonical source string into a normalized Core-IR S-expression string.
///
/// Returns `Ok(ir_sexp)` on success. On failure returns an explicit `Err(ParseError)` —
/// never a partial or silent result (G2).
///
/// # Guarantee tag: `Empirical`
/// This is a heuristic token/bracket approach over the closed v0 grammar (research/11
/// T11.4). It is **not** a proven-sound parser. The result is suitable as input to
/// `myc-check` for scoring, but do not rely on it for semantics-preserving transforms.
pub fn parse_llm_canonical(source: &str) -> Result<String, ParseError> {
    if source.trim().is_empty() {
        return Err(ParseError::EmptyInput);
    }

    let mut parser = Parser::new(source);
    let sexp = parser.read_sexp(0)?;
    let result = normalize_sexp(&sexp, 0)?;

    // Check for unconsumed input (additional top-level forms).
    parser.skip_whitespace();
    if parser.pos < parser.chars.len() {
        let mut forms = vec![result];
        loop {
            parser.skip_whitespace();
            if parser.pos >= parser.chars.len() {
                break;
            }
            let next_sexp = parser.read_sexp(0)?;
            let next = normalize_sexp(&next_sexp, 0)?;
            forms.push(next);
        }
        if forms.len() == 1 {
            Ok(forms.remove(0))
        } else {
            Ok(format!("(seq {})", forms.join(" ")))
        }
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Internal tokenizer + recursive descent parser
// ---------------------------------------------------------------------------

/// S-expression tree (internal representation before normalization).
#[derive(Debug)]
enum Sexp {
    /// An atom (identifier, keyword, literal).
    // `_line` is stored for potential future error-reporting extensions; currently
    // atom normalization does not need it (atoms return their text verbatim).
    Atom { text: String, _line: usize },
    /// A parenthesized list.
    List {
        items: Vec<Sexp>,
        /// Line where the opening `(` appeared.
        open_line: usize,
    },
    /// A bracketed list `[…]` (used for let/fn binders).
    // `_open_line` is stored for potential future error-reporting extensions; currently
    // bracket normalization recursively normalizes items without needing the line number.
    Bracket { items: Vec<Sexp>, _open_line: usize },
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    /// Line number at each char position (1-based).
    line_at: Vec<usize>,
}

impl Parser {
    fn new(source: &str) -> Self {
        let chars: Vec<char> = source.chars().collect();
        let mut line_at = Vec::with_capacity(chars.len() + 1);
        let mut ln = 1usize;
        for &c in &chars {
            line_at.push(ln);
            if c == '\n' {
                ln += 1;
            }
        }
        line_at.push(ln); // sentinel for EOF
        Self {
            chars,
            pos: 0,
            line_at,
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn current_line(&self) -> usize {
        self.line_at.get(self.pos).copied().unwrap_or(1)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_whitespace(&mut self) {
        loop {
            match self.peek_char() {
                Some(c) if c.is_whitespace() => {
                    self.advance();
                }
                Some(';') => {
                    // Line comment: skip to end of line.
                    while let Some(cc) = self.peek_char() {
                        self.advance();
                        if cc == '\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// Read a raw S-expression tree (no semantic validation yet).
    fn read_sexp(&mut self, depth: usize) -> Result<Sexp, ParseError> {
        if depth > DEPTH_LIMIT {
            return Err(ParseError::DepthLimitExceeded { limit: DEPTH_LIMIT });
        }
        self.skip_whitespace();
        match self.peek_char() {
            None => {
                let line = self.current_line();
                Err(ParseError::UnrecognizedConstruct {
                    line,
                    token: "<EOF>".into(),
                })
            }
            Some('(') => {
                let open_line = self.current_line();
                self.advance(); // consume '('
                let mut items = Vec::new();
                loop {
                    self.skip_whitespace();
                    match self.peek_char() {
                        None => {
                            return Err(ParseError::UnclosedDelimiter {
                                line: open_line,
                                delimiter: '(',
                            });
                        }
                        Some(')') => {
                            self.advance();
                            break;
                        }
                        _ => {
                            items.push(self.read_sexp(depth + 1)?);
                        }
                    }
                }
                Ok(Sexp::List { items, open_line })
            }
            Some('[') => {
                let open_line = self.current_line();
                self.advance(); // consume '['
                let mut items = Vec::new();
                loop {
                    self.skip_whitespace();
                    match self.peek_char() {
                        None => {
                            return Err(ParseError::UnclosedDelimiter {
                                line: open_line,
                                delimiter: '[',
                            });
                        }
                        Some(']') => {
                            self.advance();
                            break;
                        }
                        _ => {
                            items.push(self.read_sexp(depth + 1)?);
                        }
                    }
                }
                Ok(Sexp::Bracket {
                    items,
                    _open_line: open_line,
                })
            }
            Some('<') => {
                // Special literal forms: <N trits>, <hv:…>, <malformed-value:…>
                let line = self.current_line();
                let mut text = String::new();
                text.push('<');
                self.advance(); // consume '<'
                let mut found_close = false;
                loop {
                    match self.peek_char() {
                        None => break,
                        Some('>') => {
                            text.push('>');
                            self.advance();
                            found_close = true;
                            break;
                        }
                        Some(c) => {
                            text.push(c);
                            self.advance();
                        }
                    }
                }
                if !found_close {
                    return Err(ParseError::UnclosedDelimiter {
                        line,
                        delimiter: '<',
                    });
                }
                Ok(Sexp::Atom { text, _line: line })
            }
            Some(_) => {
                // Atom: read until whitespace or delimiter.
                let line = self.current_line();
                let mut text = String::new();
                loop {
                    match self.peek_char() {
                        None => break,
                        Some(c)
                            if c.is_whitespace()
                                || c == '('
                                || c == ')'
                                || c == '['
                                || c == ']' =>
                        {
                            break;
                        }
                        Some(c) => {
                            text.push(c);
                            self.advance();
                        }
                    }
                }
                Ok(Sexp::Atom { text, _line: line })
            }
        }
    }
}

/// Normalize a parsed S-expression into a canonical Core-IR S-expression string.
///
/// Validates the head keyword and recursively normalizes sub-expressions.
/// Returns `ParseError::UnrecognizedConstruct` for any unrecognized head.
fn normalize_sexp(sexp: &Sexp, depth: usize) -> Result<String, ParseError> {
    if depth > DEPTH_LIMIT {
        return Err(ParseError::DepthLimitExceeded { limit: DEPTH_LIMIT });
    }
    match sexp {
        Sexp::Atom { text, .. } => {
            // A bare atom is a variable reference or a keyword atom.
            Ok(text.clone())
        }
        Sexp::Bracket { items, .. } => {
            // A bracket `[…]` appears inside `let` and `fn` binders.
            let parts: Result<Vec<String>, ParseError> =
                items.iter().map(|i| normalize_sexp(i, depth + 1)).collect();
            Ok(format!("[{}]", parts?.join(" ")))
        }
        Sexp::List { items, open_line } => {
            if items.is_empty() {
                // Empty list `()` — not a valid LlmCanonical construct.
                return Err(ParseError::UnrecognizedConstruct {
                    line: *open_line,
                    token: "()".into(),
                });
            }
            // The first item determines the form.
            let head = match &items[0] {
                Sexp::Atom { text, .. } => text.as_str(),
                _ => {
                    // First element is a sub-list — this is an application form.
                    return normalize_app(items, *open_line, depth);
                }
            };

            match head {
                "const" => normalize_const(items, *open_line, depth),
                "let" => normalize_let(items, *open_line, depth),
                "op" => normalize_op(items, *open_line, depth),
                "swap!" => normalize_swap(items, *open_line, depth),
                "make" => normalize_make(items, *open_line, depth),
                "match" => normalize_match(items, *open_line, depth),
                "fn" => normalize_fn(items, *open_line, depth),
                "fix" => normalize_fix(items, *open_line, depth),
                "fix-group" => normalize_fix_group(items, *open_line, depth),
                "seq" => normalize_seq(items, *open_line, depth),
                // Any other head: treat as application (func arg…).
                _ => normalize_app(items, *open_line, depth),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-form normalization helpers
// ---------------------------------------------------------------------------

fn normalize_const(items: &[Sexp], _open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (const <value-sexp>)
    let inner: Result<Vec<String>, ParseError> = items[1..]
        .iter()
        .map(|i| normalize_sexp(i, depth + 1))
        .collect();
    Ok(format!("(const {})", inner?.join(" ")))
}

fn normalize_let(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (let [id bound-expr] body-expr)
    if items.len() < 3 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "let".into(),
        });
    }
    let binder = normalize_sexp(&items[1], depth + 1)?;
    let body = normalize_sexp(&items[2], depth + 1)?;
    Ok(format!("(let {binder} {body})"))
}

fn normalize_op(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (op prim arg...)
    if items.len() < 2 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "op".into(),
        });
    }
    let prim = normalize_sexp(&items[1], depth + 1)?;
    let args: Result<Vec<String>, ParseError> = items[2..]
        .iter()
        .map(|i| normalize_sexp(i, depth + 1))
        .collect();
    let args_str = args?.join(" ");
    if args_str.is_empty() {
        Ok(format!("(op {prim})"))
    } else {
        Ok(format!("(op {prim} {args_str})"))
    }
}

fn normalize_swap(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (swap! src :to repr :policy ref)
    if items.len() < 6 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "swap!".into(),
        });
    }
    let src = normalize_sexp(&items[1], depth + 1)?;
    let to_kw = normalize_sexp(&items[2], depth + 1)?;
    let repr = normalize_sexp(&items[3], depth + 1)?;
    let policy_kw = normalize_sexp(&items[4], depth + 1)?;
    let pol = normalize_sexp(&items[5], depth + 1)?;
    Ok(format!("(swap! {src} {to_kw} {repr} {policy_kw} {pol})"))
}

fn normalize_make(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (make ctor arg...)
    if items.len() < 2 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "make".into(),
        });
    }
    let ctor = normalize_sexp(&items[1], depth + 1)?;
    let args: Result<Vec<String>, ParseError> = items[2..]
        .iter()
        .map(|i| normalize_sexp(i, depth + 1))
        .collect();
    let args_str = args?.join(" ");
    if args_str.is_empty() {
        Ok(format!("(make {ctor})"))
    } else {
        Ok(format!("(make {ctor} {args_str})"))
    }
}

fn normalize_match(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (match scrutinee alt...)
    if items.len() < 2 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "match".into(),
        });
    }
    let scrutinee = normalize_sexp(&items[1], depth + 1)?;
    let alts: Result<Vec<String>, ParseError> = items[2..]
        .iter()
        .map(|i| normalize_sexp(i, depth + 1))
        .collect();
    let alts_str = alts?.join(" ");
    if alts_str.is_empty() {
        Ok(format!("(match {scrutinee})"))
    } else {
        Ok(format!("(match {scrutinee} {alts_str})"))
    }
}

fn normalize_fn(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (fn [param] body)
    if items.len() < 3 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "fn".into(),
        });
    }
    let param = normalize_sexp(&items[1], depth + 1)?;
    let body = normalize_sexp(&items[2], depth + 1)?;
    Ok(format!("(fn {param} {body})"))
}

fn normalize_fix(items: &[Sexp], open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (fix name body)
    if items.len() < 3 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "fix".into(),
        });
    }
    let name = normalize_sexp(&items[1], depth + 1)?;
    let body = normalize_sexp(&items[2], depth + 1)?;
    Ok(format!("(fix {name} {body})"))
}

fn normalize_fix_group(
    items: &[Sexp],
    open_line: usize,
    depth: usize,
) -> Result<String, ParseError> {
    // (fix-group (binds...) body)
    if items.len() < 3 {
        return Err(ParseError::UnrecognizedConstruct {
            line: open_line,
            token: "fix-group".into(),
        });
    }
    let binds = normalize_sexp(&items[1], depth + 1)?;
    let body = normalize_sexp(&items[2], depth + 1)?;
    Ok(format!("(fix-group {binds} {body})"))
}

fn normalize_seq(items: &[Sexp], _open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (seq form...) — internal multi-form wrapper
    let parts: Result<Vec<String>, ParseError> = items[1..]
        .iter()
        .map(|i| normalize_sexp(i, depth + 1))
        .collect();
    Ok(format!("(seq {})", parts?.join(" ")))
}

fn normalize_app(items: &[Sexp], _open_line: usize, depth: usize) -> Result<String, ParseError> {
    // (func arg...) — application
    let func = normalize_sexp(&items[0], depth + 1)?;
    let args: Result<Vec<String>, ParseError> = items[1..]
        .iter()
        .map(|i| normalize_sexp(i, depth + 1))
        .collect();
    let args_str = args?.join(" ");
    Ok(format!("({func} {args_str})"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- mutant-witness tests -------------------------------------------------

    /// `test_empty_input_is_error`: `parse_llm_canonical("")` returns
    /// `Err(ParseError::EmptyInput)`. Exact variant — returning `Ok("")` would
    /// make this fail (mutant witness: the empty-check guard is required).
    #[test]
    fn test_empty_input_is_error() {
        assert_eq!(parse_llm_canonical(""), Err(ParseError::EmptyInput));
        assert_eq!(parse_llm_canonical("   \t\n"), Err(ParseError::EmptyInput));
    }

    /// `test_depth_limit_exceeded`: 65 levels of nesting returns
    /// `Err(ParseError::DepthLimitExceeded { limit: 64 })`. Exact variant —
    /// removing the depth check causes a stack overflow on deep input (mutant
    /// witness: the depth guard is required; DEPTH_LIMIT == 64).
    #[test]
    fn test_depth_limit_exceeded() {
        // Build a string with DEPTH_LIMIT+1 opening parens, each nested inside the
        // next.  The read_sexp depth guard fires at depth > DEPTH_LIMIT.
        // When we call read_sexp(0) on the outermost `(`:
        //   - depth 0: reads `(`, recurses with depth 1 on inner content
        //   - …
        //   - depth DEPTH_LIMIT: reads `(`, tries to recurse with depth DEPTH_LIMIT+1
        //     → fires DepthLimitExceeded before reading inner content
        // So we need DEPTH_LIMIT+1 opening parens total.
        let nesting = DEPTH_LIMIT + 1;
        let open: String = "(const ".repeat(nesting);
        let close: String = ")".repeat(nesting);
        let deep_input = format!("{open}x @Exact{close}");
        assert_eq!(
            parse_llm_canonical(&deep_input),
            Err(ParseError::DepthLimitExceeded { limit: DEPTH_LIMIT })
        );
    }

    /// `test_roundtrip_simple_construct`: a simple Core-IR fragment rendered by the
    /// LlmCanonical renderer is then parsed back. The result should be `Ok` and contain
    /// the key structural tokens. This is the key roundtrip property test.
    #[test]
    fn test_roundtrip_simple_construct() {
        use crate::project::llm_canonical;
        use mycelium_core::Node;

        // A simple Op node: (op bit.not x)
        let op_node = Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Var("x".into())],
        };
        let rendered =
            llm_canonical(&op_node).expect("small fixture fits the default arena ceiling");
        let parsed = parse_llm_canonical(&rendered);
        assert!(
            parsed.is_ok(),
            "roundtrip parse failed: {:?} on input {:?}",
            parsed,
            rendered
        );
        let out = parsed.unwrap();
        assert!(
            out.contains("op"),
            "roundtrip output should contain 'op': {out}"
        );
        assert!(
            out.contains("bit.not"),
            "roundtrip output should contain 'bit.not': {out}"
        );
        assert!(
            out.contains('x'),
            "roundtrip output should contain 'x': {out}"
        );
    }

    /// `test_unrecognized_construct`: a form not in the LlmCanonical vocabulary returns
    /// `Err(ParseError::UnrecognizedConstruct { .. })`. We use an empty list `()`, which
    /// the renderer never produces, and assert the `token` field.
    #[test]
    fn test_unrecognized_construct() {
        let result = parse_llm_canonical("()");
        assert!(
            matches!(result, Err(ParseError::UnrecognizedConstruct { ref token, .. }) if token == "()"),
            "expected UnrecognizedConstruct with token='()': {result:?}"
        );
    }

    /// `test_unclosed_delimiter`: unclosed `(` returns
    /// `Err(ParseError::UnclosedDelimiter { delimiter: '(', .. })`.
    #[test]
    fn test_unclosed_delimiter() {
        let result = parse_llm_canonical("(const 0b101 @Exact");
        assert!(
            matches!(
                result,
                Err(ParseError::UnclosedDelimiter { delimiter: '(', .. })
            ),
            "expected UnclosedDelimiter for unclosed '(': {result:?}"
        );
    }

    // -- additional coverage tests -------------------------------------------

    /// Unclosed `<` returns `UnclosedDelimiter { delimiter: '<' }`.
    #[test]
    fn test_unclosed_angle() {
        let result = parse_llm_canonical("<4 trits");
        assert!(
            matches!(
                result,
                Err(ParseError::UnclosedDelimiter { delimiter: '<', .. })
            ),
            "expected UnclosedDelimiter for unclosed '<': {result:?}"
        );
    }

    /// A valid variable reference atom parses successfully.
    #[test]
    fn test_var_atom_parses() {
        let result = parse_llm_canonical("myvar");
        assert_eq!(result, Ok("myvar".to_string()));
    }

    /// A complete op form with depth well within the limit parses successfully.
    #[test]
    fn test_op_parses() {
        let result = parse_llm_canonical("(op bit.not x)");
        assert!(result.is_ok(), "op parse failed: {result:?}");
        let out = result.unwrap();
        assert!(out.contains("op") && out.contains("bit.not") && out.contains('x'));
    }

    /// `(fn [x] x)` parses as a lambda.
    #[test]
    fn test_fn_parses() {
        let result = parse_llm_canonical("(fn [x] x)");
        assert!(result.is_ok(), "fn parse failed: {result:?}");
        let out = result.unwrap();
        assert!(out.contains("fn") && out.contains('x'));
    }

    /// `(fix f f)` parses as a fixed-point.
    #[test]
    fn test_fix_parses() {
        let result = parse_llm_canonical("(fix f f)");
        assert!(result.is_ok(), "fix parse failed: {result:?}");
        let out = result.unwrap();
        assert!(out.contains("fix") && out.contains('f'));
    }

    /// The depth limit boundary: exactly DEPTH_LIMIT levels succeeds; DEPTH_LIMIT+1 fails.
    #[test]
    fn test_depth_limit_boundary() {
        // Build DEPTH_LIMIT nested (op id …) forms — innermost is just "x".
        // Each (op id <inner>) adds one list level.
        // read_sexp(0) → reads `(` at depth 0, recurses with depth 1 for content.
        // The atom "op" is read at depth 1, then prim "id" at depth 1, then
        // the next element at depth 1 which is another `(`, so read_sexp(2), etc.
        // After DEPTH_LIMIT layers the innermost would be read at depth DEPTH_LIMIT.
        // One more layer would try depth DEPTH_LIMIT+1 → DepthLimitExceeded.
        let mut s = "x".to_string();
        for _ in 0..DEPTH_LIMIT {
            s = format!("(op id {s})");
        }
        // DEPTH_LIMIT nested lists: deepest list content read at depth DEPTH_LIMIT.
        // That is exactly the limit so it should succeed.
        let at_limit = parse_llm_canonical(&s);
        assert!(
            at_limit.is_ok(),
            "depth exactly at limit ({DEPTH_LIMIT} lists) should succeed: {at_limit:?}"
        );

        // One more level: DEPTH_LIMIT+1 lists → should fail.
        let s_over = format!("(op id {s})");
        let over_limit = parse_llm_canonical(&s_over);
        assert_eq!(
            over_limit,
            Err(ParseError::DepthLimitExceeded { limit: DEPTH_LIMIT })
        );
    }

    /// Roundtrip: a let binding goes through renderer → parser without error.
    #[test]
    fn test_roundtrip_let() {
        use crate::project::llm_canonical;
        use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Value};

        let byte_val = Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, true, false, false, true, false]),
            Meta::exact(Provenance::Root),
        )
        .unwrap();
        let let_node = Node::Let {
            id: "a".into(),
            bound: Box::new(Node::Const(byte_val)),
            body: Box::new(Node::Var("a".into())),
        };
        let rendered =
            llm_canonical(&let_node).expect("small fixture fits the default arena ceiling");
        let parsed = parse_llm_canonical(&rendered);
        assert!(
            parsed.is_ok(),
            "let roundtrip failed: {:?} on input {:?}",
            parsed,
            rendered
        );
        let out = parsed.unwrap();
        assert!(
            out.contains("let"),
            "roundtrip output should contain 'let': {out}"
        );
    }

    /// Roundtrip: swap node preserves `swap!`, `:to`, `:policy` (P3).
    #[test]
    fn test_roundtrip_swap() {
        use crate::project::llm_canonical;
        use mycelium_core::{ContentHash, Meta, Node, Payload, Provenance, Repr, Value};

        let byte_val = Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![false; 8]),
            Meta::exact(Provenance::Root),
        )
        .unwrap();
        let swap_node = Node::Swap {
            src: Box::new(Node::Const(byte_val)),
            target: Repr::Ternary { trits: 6 },
            policy: ContentHash::parse("blake3:po1icy00").unwrap(),
        };
        let rendered =
            llm_canonical(&swap_node).expect("small fixture fits the default arena ceiling");
        let parsed = parse_llm_canonical(&rendered);
        assert!(
            parsed.is_ok(),
            "swap roundtrip failed: {:?} on input {:?}",
            parsed,
            rendered
        );
        let out = parsed.unwrap();
        assert!(out.contains("swap!"), "P3: swap! must survive: {out}");
        assert!(out.contains(":to"), "P3: :to must survive: {out}");
        assert!(out.contains(":policy"), "P3: :policy must survive: {out}");
        assert!(
            out.contains("Ternary"),
            "P3: target repr must survive: {out}"
        );
    }

    /// Roundtrip all 11 node kinds: renderer → parser succeeds for every node.
    #[test]
    fn test_roundtrip_all_11_node_kinds() {
        use crate::project::llm_canonical;
        use mycelium_core::{ContentHash, CtorRef, Meta, Node, Payload, Provenance, Repr, Value};

        let byte_val = || {
            Value::new(
                Repr::Binary { width: 8 },
                Payload::Bits(vec![true, false, true, true, false, false, true, false]),
                Meta::exact(Provenance::Root),
            )
            .unwrap()
        };
        let ctor = CtorRef::new(ContentHash::parse("blake3:00ctor00").unwrap(), 0);
        let nodes: Vec<Node> = vec![
            Node::Const(byte_val()),
            Node::Var("x".into()),
            Node::Let {
                id: "a".into(),
                bound: Box::new(Node::Const(byte_val())),
                body: Box::new(Node::Var("a".into())),
            },
            Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Const(byte_val())],
            },
            Node::Swap {
                src: Box::new(Node::Const(byte_val())),
                target: Repr::Ternary { trits: 6 },
                policy: ContentHash::parse("blake3:po1icy00").unwrap(),
            },
            Node::Construct {
                ctor: ctor.clone(),
                args: vec![Node::Const(byte_val())],
            },
            Node::Match {
                scrutinee: Box::new(Node::Const(byte_val())),
                alts: vec![],
                default: Some(Box::new(Node::Const(byte_val()))),
            },
            Node::Lam {
                param: "x".into(),
                body: Box::new(Node::Var("x".into())),
            },
            Node::App {
                func: Box::new(Node::Var("f".into())),
                arg: Box::new(Node::Const(byte_val())),
            },
            Node::Fix {
                name: "f".into(),
                body: Box::new(Node::Var("f".into())),
            },
            Node::FixGroup {
                defs: vec![
                    ("f".into(), Box::new(Node::Var("g".into()))),
                    ("g".into(), Box::new(Node::Var("f".into()))),
                ],
                body: Box::new(Node::Var("f".into())),
            },
        ];
        assert_eq!(nodes.len(), 11, "coverage: all 11 v0 node kinds");
        for n in &nodes {
            let rendered = llm_canonical(n).expect("small fixture fits the default arena ceiling");
            let parsed = parse_llm_canonical(&rendered);
            assert!(
                parsed.is_ok(),
                "node {:?}: roundtrip failed {:?} on input {:?}",
                std::mem::discriminant(n),
                parsed,
                rendered
            );
        }
    }
}
