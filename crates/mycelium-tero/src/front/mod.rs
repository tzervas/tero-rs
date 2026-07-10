//! The API fronts (M-1017 / DN-87 §2.3): **one core, two thin fronts**. The query engine (M-1016)
//! is driven through exactly one framework-agnostic surface ([`core`]) — request → [`crate::Query`]
//! → a stable JSON envelope — and both fronts are thin adapters over it:
//!
//! - [`mcp`] — an MCP server over stdio (native tool ergonomics for MCP-speaking platforms;
//!   newline-delimited JSON-RPC 2.0, modeled on `mycelium-lsp`'s wire loop). *(M-1017 PR-B.)*
//! - [`http`] — a plain versioned HTTP/JSON API (the universal floor: Grok, curl, anything), an
//!   `axum` app on the `tokio` runtime.
//!
//! Because both fronts serialize through the **same** [`core`] envelope, an answer over MCP is
//! byte-identical to the same answer over HTTP (the M-1017 DoD "parity (differential-tested
//! answers)" — see `crate::tests::front_parity`). Access is [`auth`]-gated: token-scoped,
//! read-only by default, and never-silent (a bad/absent token or too-narrow scope is an explicit
//! refusal, never a silent allow).
//!
//! Honesty (VR-5): the fronts are `Declared` — mechanical transport plus a token check; no security
//! *proof* is claimed (see [`auth`] for the constant-time-compare hardening note). The provenance
//! and refusal guarantees are the engine's ([`crate::query`]); the fronts only carry them across the
//! wire without weakening them (a refusal stays a refusal; a citation stays resolvable).

pub(crate) mod auth;
pub(crate) mod core;
pub(crate) mod http;
pub(crate) mod mcp;
