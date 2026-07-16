//! Markdown corpus ingest for tero (self-contained; no mycelium crates).
pub mod corpus;
pub mod ir;

pub use corpus::{ingest, AnchorAlloc};
pub use ir::{Level, Node, Payload, SourceKind};
