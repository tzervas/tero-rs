//! Token-scoped auth for the API fronts (M-1017 / DN-87 §6.4: "token-scoped auth, read-only by
//! default, no secret material served"). A small, never-silent allow-list:
//!
//! - **Tokens are runtime-only** — supplied via `TERO_TOKENS` (an inline `token:scope` list) or
//!   `TERO_TOKENS_FILE` (the same grammar in a file). **Never** committed, logged, or serialized
//!   (the gitleaks gate covers generated artifacts too); this module deliberately has no `Display`
//!   that prints a token.
//! - **Read-only by default** — the `read` scope covers query/cite/explain/identify; the broader
//!   `refresh` scope additionally permits reloading the index. `refresh ⊇ read`.
//! - **Refuse to start without tokens** — [`TokenTable::from_env`] returns an error (not an empty,
//!   accidentally-open table) when no tokens are configured; the binaries surface it on stderr and
//!   exit non-zero. There is no anonymous default in v0 (YAGNI + footgun avoidance).
//!
//! Honesty (VR-5): this is a `Declared` mechanism — a constant-time token comparison is the obvious
//! hardening step and is **not** claimed here (the [`HashMap`] lookup is not constant-time; for the
//! 127.0.0.1 read-only floor the timing channel is out of scope, and TLS/CT-compare belong to a
//! future hardening pass). No cryptographic guarantee is asserted.

use std::collections::HashMap;
use std::env;
use std::fs;

use crate::front::core::FrontError;

/// The access scope a token carries. `Refresh` is a strict superset of `Read` (`refresh ⊇ read`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Read-only: query / cite / explain / identify. The default, per DN-87 §6.4.
    Read,
    /// Read **plus** `refresh` (reload the served index from disk).
    Refresh,
}

impl Scope {
    /// Privilege rank — higher permits more. `Refresh` (1) ⊇ `Read` (0).
    fn rank(self) -> u8 {
        match self {
            Scope::Read => 0,
            Scope::Refresh => 1,
        }
    }

    /// Whether a token of `self` scope may perform an operation requiring `required` scope.
    #[must_use]
    pub fn allows(self, required: Scope) -> bool {
        self.rank() >= required.rank()
    }

    /// The wire keyword (`read` / `refresh`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Read => "read",
            Scope::Refresh => "refresh",
        }
    }

    /// Parse a scope keyword. `None` for any other string (never a silent default — the caller
    /// records the malformed entry).
    fn parse(s: &str) -> Option<Scope> {
        match s {
            "read" => Some(Scope::Read),
            "refresh" => Some(Scope::Refresh),
            _ => None,
        }
    }
}

/// A configuration error building the [`TokenTable`] — surfaced at startup, never swallowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenTableError {
    /// Neither `TERO_TOKENS` nor `TERO_TOKENS_FILE` was set.
    Unset,
    /// The source was set but held no `token:scope` entries (an accidentally-open server — refused).
    Empty,
    /// A `token:scope` entry was malformed (missing `:`, empty token, unknown scope keyword).
    Malformed(String),
    /// `TERO_TOKENS_FILE` could not be read.
    Io(String),
}

impl std::fmt::Display for TokenTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenTableError::Unset => write!(
                f,
                "no API tokens configured — set TERO_TOKENS (a `token:scope` list, e.g. \
                 `s3cr3t:read other:refresh`) or TERO_TOKENS_FILE; the front refuses to start \
                 without tokens (no anonymous default)"
            ),
            TokenTableError::Empty => write!(
                f,
                "the configured token source held no `token:scope` entries — refusing to start an \
                 open server"
            ),
            TokenTableError::Malformed(why) => write!(f, "malformed token entry: {why}"),
            TokenTableError::Io(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for TokenTableError {}

/// A per-request authorization failure, mapped to a `4xx` / JSON-RPC code by the front (via
/// [`FrontError`]). Deliberately coarse — it never echoes the presented token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// No token presented.
    Missing,
    /// A token was presented but is not in the allow-list.
    Invalid,
    /// The token is valid but its scope does not permit the operation.
    InsufficientScope {
        /// The scope the token actually has.
        have: Scope,
        /// The scope the operation requires.
        need: Scope,
    },
}

impl From<AuthError> for FrontError {
    fn from(e: AuthError) -> FrontError {
        match e {
            AuthError::Missing => FrontError::Unauthorized(
                "missing bearer token (Authorization: Bearer <token>)".into(),
            ),
            AuthError::Invalid => FrontError::Unauthorized("invalid token".into()),
            AuthError::InsufficientScope { have, need } => FrontError::Forbidden(format!(
                "token scope `{}` does not permit this operation (requires `{}`)",
                have.as_str(),
                need.as_str()
            )),
        }
    }
}

/// A runtime allow-list of `token -> scope`. Constructed once at startup from the environment; the
/// fronts hold it read-only and consult it on every request.
#[derive(Debug, Clone)]
pub struct TokenTable {
    tokens: HashMap<String, Scope>,
}

impl TokenTable {
    /// Load tokens from the environment: `TERO_TOKENS_FILE` (a path to a `token:scope` list) takes
    /// precedence, else `TERO_TOKENS` (the inline list). Returns [`TokenTableError`] — never an empty
    /// table — when nothing is configured or an entry is malformed (never-silent startup).
    pub fn from_env() -> Result<TokenTable, TokenTableError> {
        let raw = match env::var("TERO_TOKENS_FILE") {
            Ok(path) if !path.is_empty() => fs::read_to_string(&path).map_err(|e| {
                TokenTableError::Io(format!("reading TERO_TOKENS_FILE={path}: {e}"))
            })?,
            _ => env::var("TERO_TOKENS").map_err(|_| TokenTableError::Unset)?,
        };
        Self::parse(&raw)
    }

    /// Parse a whitespace/comma-separated `token:scope` list into a table. Public for testability
    /// (the auth tests exercise it directly without touching process env).
    pub fn parse(raw: &str) -> Result<TokenTable, TokenTableError> {
        let mut tokens = HashMap::new();
        for entry in raw
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
        {
            let (tok, scope) = entry.split_once(':').ok_or_else(|| {
                TokenTableError::Malformed(format!("entry {entry:?} is not `token:scope`"))
            })?;
            if tok.is_empty() {
                return Err(TokenTableError::Malformed(format!(
                    "empty token in entry {entry:?}"
                )));
            }
            let scope = Scope::parse(scope).ok_or_else(|| {
                TokenTableError::Malformed(format!(
                    "unknown scope {scope:?} in entry {entry:?} (expected `read` or `refresh`)"
                ))
            })?;
            tokens.insert(tok.to_owned(), scope);
        }
        if tokens.is_empty() {
            return Err(TokenTableError::Empty);
        }
        Ok(TokenTable { tokens })
    }

    /// Authorize a presented token for an operation requiring `required` scope. Returns the token's
    /// granted scope on success, or a typed [`AuthError`] (missing / invalid / insufficient) —
    /// never a silent allow.
    pub fn authorize(&self, presented: Option<&str>, required: Scope) -> Result<Scope, AuthError> {
        let token = presented.ok_or(AuthError::Missing)?;
        let &have = self.tokens.get(token).ok_or(AuthError::Invalid)?;
        if have.allows(required) {
            Ok(have)
        } else {
            Err(AuthError::InsufficientScope {
                have,
                need: required,
            })
        }
    }

    /// The number of configured tokens (never zero post-construction — see [`TokenTableError::Empty`]).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Always `false` for a constructed table (kept for the `clippy::len_without_is_empty` lint).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
