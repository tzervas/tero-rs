//! Minimal document IR for tero markdown ingest (no mycelium language-kernel types).

#![allow(dead_code)] // IR mirrors full corpus surface; not every variant is projected today.

use serde::Serialize;

/// Graded depth for progressive disclosure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Minimal,
    Medium,
    Detailed,
}

/// Corpus source family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    Rfc,
    Adr,
    Note,
    Spec,
    Devlog,
    Api,
    Other,
}

impl SourceKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Rfc => "rfc",
            SourceKind::Adr => "adr",
            SourceKind::Note => "note",
            SourceKind::Spec => "spec",
            SourceKind::Devlog => "devlog",
            SourceKind::Api => "api",
            SourceKind::Other => "other",
        }
    }
}

/// Source provenance (path + line).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Provenance {
    pub source: String,
    pub line: u32,
}

/// Cross-reference resolution status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "resolution", rename_all = "kebab-case")]
pub enum XrefResolution {
    Unresolved,
    Internal { anchor: String },
    ExternalUrl,
    OutOfScope,
    Dead { reason: String },
}

/// Cross-reference target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct XrefTarget {
    pub raw: String,
    pub resolution: XrefResolution,
}

/// Kind-specific node content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Payload {
    Document {
        source_kind: SourceKind,
    },
    Section,
    Prose {
        text: String,
    },
    Example {
        lang: String,
        source: String,
        checked: bool,
    },
    Xref {
        target: XrefTarget,
    },
    ApiItem {
        signature: Option<String>,
        summary: Option<String>,
    },
    Undocumented {
        reason: String,
    },
}

/// A node in the doc tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Node {
    pub anchor: String,
    pub title: Option<String>,
    pub level: Option<Level>,
    pub provenance: Provenance,
    pub payload: Payload,
    pub children: Vec<Node>,
}

impl Node {
    /// Construct a node (no content-addressing; anchors are the stable keys for tero).
    #[must_use]
    pub fn new(
        anchor: impl Into<String>,
        title: Option<String>,
        level: Option<Level>,
        provenance: Provenance,
        payload: Payload,
        children: Vec<Node>,
    ) -> Self {
        Self {
            anchor: anchor.into(),
            title,
            level,
            provenance,
            payload,
            children,
        }
    }

    /// Depth-first walk.
    pub fn walk<'a>(&'a self, f: &mut dyn FnMut(&'a Node)) {
        f(self);
        for c in &self.children {
            c.walk(f);
        }
    }
}
