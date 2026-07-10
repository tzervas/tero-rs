//! The **doc-IR**: a typed, content-addressed intermediate the whole corpus (RFCs/ADRs/notes/specs +
//! code + M-359 header metadata) is *projected into* — one navigable model, many renderers. This is a
//! **projection, never a parallel truth** (ADR-003): a node's identity is the hash of its projected
//! content ([`crate::hash`]), so HTML, Typst and JSON are *views of one node* and cannot silently
//! diverge (the §4.1 `dual-projection-parity` lint enforces it).
//!
//! "Undocumented is **flagged**, never invented" (the prose analogue of G2): a missing doc surfaces as
//! an explicit [`Payload::Undocumented`] / an [`Payload::ApiItem`] with `summary: None`, rendered as a
//! visible "undocumented" marker — never papered over with filler.

use std::collections::BTreeMap;

use mycelium_core::ContentHash;
use serde::Serialize;

use crate::hash::DocHasher;

/// Graded depth (RFC-0013's `minimal / medium / detailed` levels, reused for docs — §4.1 progressive
/// disclosure). A reader picks how deep to go over *one* truth, never three divergent ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// A one-line summary.
    Minimal,
    /// The working explanation.
    Medium,
    /// The full normative detail.
    Detailed,
}

impl Level {
    /// The canonical label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Minimal => "minimal",
            Level::Medium => "medium",
            Level::Detailed => "detailed",
        }
    }

    fn tag(self) -> u8 {
        match self {
            Level::Minimal => 1,
            Level::Medium => 2,
            Level::Detailed => 3,
        }
    }
}

/// Which corpus family a [`Payload::Document`] was projected from (drives ordering + the template's
/// section grouping; part of the navigable index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    /// An RFC (normative design).
    Rfc,
    /// An ADR (architecture decision).
    Adr,
    /// A design note (DN-*) or other note.
    Note,
    /// A spec / contract document.
    Spec,
    /// A devlog narrative entry.
    Devlog,
    /// A projected code/API reference unit (a `.myc` nodule or a JSON schema).
    Api,
    /// Other corpus markdown (glossary, index, charter, …).
    Other,
}

impl SourceKind {
    /// The canonical label.
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

/// Where a node was projected from (append-only provenance, §9 — "generated from"). Metadata, **not
/// identity** (ADR-003): provenance is recorded but never perturbs the content hash, so re-flowing a
/// source line does not break a deep link.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Provenance {
    /// The repo-relative source path the node projects.
    pub source: String,
    /// The 1-based source line (0 when not line-addressable).
    pub line: u32,
}

/// How a cross-reference resolved against the model (the §4.1 `no-dead-xref` verdict).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "resolution", rename_all = "kebab-case")]
pub enum XrefResolution {
    /// Not yet resolved (the ingest-time placeholder; the build always resolves before finalizing —
    /// seeing this in the lint is a bug, reported never-silently).
    Unresolved,
    /// Resolved to an intra-model anchor (a content address via the index).
    Internal {
        /// The resolved anchor.
        anchor: String,
    },
    /// An `http(s)://` URL — out of scope here; `links.sh` / the browser own external reachability.
    ExternalUrl,
    /// A non-`.md`, non-corpus relative target (a script, `mailto:`, …) — not a doc xref.
    OutOfScope,
    /// A same-repo `.md`/anchor target that does **not** resolve — a build failure (§4.1 #5).
    Dead {
        /// Why it is dead, in author-facing terms.
        reason: String,
    },
}

/// The resolved-or-not target of a cross-reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct XrefTarget {
    /// The raw link target as authored (e.g. `RFC-0013.md#levels`, `#section`, `https://…`).
    pub raw: String,
    /// The resolution verdict.
    pub resolution: XrefResolution,
}

impl XrefResolution {
    fn tag(&self) -> u8 {
        match self {
            XrefResolution::Unresolved => 0,
            XrefResolution::Internal { .. } => 1,
            XrefResolution::ExternalUrl => 2,
            XrefResolution::OutOfScope => 3,
            XrefResolution::Dead { .. } => 4,
        }
    }
}

/// The kind-specific content of a node. The variant **is** the node's type; shared fields (anchor,
/// title, level, provenance, children) live on [`Node`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Payload {
    /// A whole projected source document (an RFC/ADR/note/spec/devlog file, or an API unit).
    Document {
        /// The corpus family.
        source_kind: SourceKind,
    },
    /// A heading and the block run beneath it.
    Section,
    /// A paragraph / text block.
    Prose {
        /// The prose text (normalized, projected verbatim — never rewritten).
        text: String,
    },
    /// A fenced code example. `checked` examples must type-check (§4.1 #4); illustrative ones are
    /// honestly flagged as not-CI-checked, never silently treated as verified.
    Example {
        /// The fence language tag (`myc`, `text`, …).
        lang: String,
        /// The example source, verbatim.
        source: String,
        /// Whether this example is held to the type-check bar.
        checked: bool,
    },
    /// A cross-reference to another node.
    Xref {
        /// The resolved-or-not target.
        target: XrefTarget,
    },
    /// A projected API item (a `.myc` nodule header, a `fn` signature, or a JSON-schema field).
    /// `summary: None` is the explicit *undocumented* state (G2) — rendered, never invented.
    ApiItem {
        /// The signature / declaration, when one was projected.
        signature: Option<String>,
        /// The `@summary` (or schema `description`) projected from the source, or `None`.
        summary: Option<String>,
    },
    /// An explicit "undocumented" marker — a visible, honest gap, never filler prose (G2).
    Undocumented {
        /// What is undocumented, in author-facing terms.
        what: String,
    },
    /// The generated index root (the navigable index→detail entry point, §4.1 #2).
    Index,
}

impl Payload {
    fn tag(&self) -> u8 {
        match self {
            Payload::Document { .. } => 1,
            Payload::Section => 2,
            Payload::Prose { .. } => 3,
            Payload::Example { .. } => 4,
            Payload::Xref { .. } => 5,
            Payload::ApiItem { .. } => 6,
            Payload::Undocumented { .. } => 7,
            Payload::Index => 8,
        }
    }

    /// The canonical kind label (for diagnostics / the machine projection).
    #[must_use]
    pub fn kind_str(&self) -> &'static str {
        match self {
            Payload::Document { .. } => "document",
            Payload::Section => "section",
            Payload::Prose { .. } => "prose",
            Payload::Example { .. } => "example",
            Payload::Xref { .. } => "xref",
            Payload::ApiItem { .. } => "api-item",
            Payload::Undocumented { .. } => "undocumented",
            Payload::Index => "index",
        }
    }
}

/// One node of the content-addressed doc-IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Node {
    /// The content address (BLAKE3 over the projected content + child ids — ADR-003).
    pub id: ContentHash,
    /// A stable, globally-unique slug used for navigation + deep links.
    pub anchor: String,
    /// The display title, when the node has one.
    pub title: Option<String>,
    /// The graded depth, when this is a level-graded block.
    pub level: Option<Level>,
    /// Where this node was projected from (metadata, not identity).
    pub provenance: Provenance,
    /// The kind-specific content.
    pub payload: Payload,
    /// Child nodes (already content-addressed — their ids feed this node's hash).
    pub children: Vec<Node>,
}

/// Iterative destruction (RFC-0041 §4.5's doc-IR member of the iterative-destruction class).
///
/// The **derived** recursive `Drop` this replaces walks the `children` spine one host-stack frame
/// per depth level, so a deep tree (confirmed empirically down to n=50,000 — `guard_hole_census.rs`,
/// `src/tests/ir.rs::walk_does_not_overflow_on_a_deep_chain`) overflows the stack on drop, independent
/// of (and previously un-closed by) the [`Node::walk`] host-stack guard. `mycelium-doc` is a tooling
/// crate, not the frozen kernel/L1 core §4.5 otherwise tracks, so this lands as a normal fix (no
/// within-freeze channel needed).
///
/// Mechanics: `mem::take` the drop target's `children` into an explicit worklist `Vec<Node>`, then
/// drain it depth-first, at each step also `mem::take`-ing the popped node's own `children` onto the
/// worklist *before* that node is allowed to drop. By the time a worklist node's implicit drop glue
/// re-enters this `impl Drop` (recursively, once per node), its `children` is already empty — so the
/// recursion never goes deeper than one level; the depth that used to live on the host stack now lives
/// in the worklist `Vec` on the heap. **No observable change** (destruction-order-only): `Node` has no
/// `Drop`-visible side effects (no external resources), only memory to reclaim, so reordering *which*
/// sibling frees first is unobservable — only that every node frees exactly once, which this
/// preserves. A worklist-`Vec` allocation during `drop` is acceptable here (unlike the OOM/unwind-
/// critical kernel path in §4.5): `mycelium-doc` is a build-time tooling crate, not on the interpreter's
/// panic/unwind hot path.
impl Drop for Node {
    fn drop(&mut self) {
        let mut worklist: Vec<Node> = std::mem::take(&mut self.children);
        while let Some(mut next) = worklist.pop() {
            // Move this node's children onto the worklist *before* `next` is dropped at the end of
            // the loop body, so the recursive re-entry into `Node::drop` below sees an already-empty
            // `children` and does no further recursion.
            worklist.extend(std::mem::take(&mut next.children));
            // `next` drops here — its `children` is empty, so this is O(1), not recursive depth.
        }
    }
}

impl Node {
    /// Build a node, computing its content address from its content + children (ADR-003). Provenance
    /// is **not** hashed (metadata, not identity) so a re-flowed source line keeps the deep link
    /// stable.
    #[must_use]
    pub fn new(
        anchor: impl Into<String>,
        title: Option<String>,
        level: Option<Level>,
        provenance: Provenance,
        payload: Payload,
        children: Vec<Node>,
    ) -> Node {
        let anchor = anchor.into();
        let mut h = DocHasher::new();
        h.tag(payload.tag());
        // The anchor is part of identity (it is the stable address), the title and level too.
        h.str(&anchor);
        h.opt_str(title.as_deref());
        h.tag(level.map_or(0, Level::tag));
        // Payload content.
        match &payload {
            Payload::Document { source_kind } => {
                h.str(source_kind.as_str());
            }
            Payload::Section | Payload::Index => {}
            Payload::Prose { text } => {
                h.str(text);
            }
            Payload::Example {
                lang,
                source,
                checked,
            } => {
                h.str(lang).str(source).u64(u64::from(*checked));
            }
            Payload::Xref { target } => {
                h.str(&target.raw).tag(target.resolution.tag());
                if let XrefResolution::Internal { anchor } = &target.resolution {
                    h.str(anchor);
                }
                if let XrefResolution::Dead { reason } = &target.resolution {
                    h.str(reason);
                }
            }
            Payload::ApiItem { signature, summary } => {
                h.opt_str(signature.as_deref()).opt_str(summary.as_deref());
            }
            Payload::Undocumented { what } => {
                h.str(what);
            }
        }
        // Child ids (order-sensitive — a section is its ordered block run).
        h.u64(children.len() as u64);
        for c in &children {
            h.child(&c.id);
        }
        Node {
            id: h.finish(),
            anchor,
            title,
            level,
            provenance,
            payload,
            children,
        }
    }

    /// Depth-first pre-order visit of this node and its descendants.
    ///
    /// **RFC-0041 §4.7 guard-hole close (W1, RR-29).** The whole walk runs on
    /// [`mycelium_workstack::ensure_sufficient_stack`]'s grown worker stack (a 256 MiB
    /// lazily-committed thread), so a pathologically deep IR tree (thousands of nested sections)
    /// walks to completion instead of overflowing the caller's host stack — `walk` stays infallible
    /// (`()`); the fix is that it now never `SIGABRT`s. The budget passed is
    /// `mycelium_workstack::RecursionBudget::default()` — its depth/mem/step ceilings play no role in
    /// W1 (the guard body only grows the host stack; it does not charge against the budget), so this
    /// closes the host-stack hole only — it does not introduce a new refusal path, and behavior is
    /// otherwise unchanged (`Declared`: a real depth/work-step ceiling for doc-IR walks is future
    /// work, not introduced silently here).
    ///
    /// The guard is applied **once**, at this public entry — the recursion itself runs through
    /// [`walk_inner`](Self::walk_inner), which does **not** re-guard per level (guarding every
    /// recursive step would spawn a worker thread per node: wrong and needlessly expensive).
    pub fn walk<'a>(&'a self, f: &mut (dyn FnMut(&'a Node) + Send)) {
        let budget = mycelium_workstack::RecursionBudget::default();
        mycelium_workstack::ensure_sufficient_stack(&budget, move || self.walk_inner(f));
    }

    /// The unguarded recursive body of [`walk`](Self::walk). Runs entirely on the worker stack the
    /// public entry already established.
    fn walk_inner<'a>(&'a self, f: &mut dyn FnMut(&'a Node)) {
        f(self);
        for c in &self.children {
            c.walk_inner(f);
        }
    }
}

/// The whole projected corpus: top-level documents plus the navigable index over every node.
#[derive(Debug, Clone, Serialize)]
pub struct DocModel {
    /// The top-level [`Payload::Document`] nodes, in projection order.
    pub documents: Vec<Node>,
    /// anchor → content address, over **every** node (the search/navigation index, §4.1 #2).
    pub anchors: BTreeMap<String, ContentHash>,
}

impl DocModel {
    /// Assemble a model from projected documents, building the anchor index. The builders allocate
    /// globally-unique anchors, so a collision would be an internal bug; if one ever slips through,
    /// the duplicate key collapses here and the **navigability lint catches it** (it errors when the
    /// anchor count is below the node count — §4.1 #2). The detection is at lint time, not here.
    #[must_use]
    pub fn new(documents: Vec<Node>) -> DocModel {
        let mut anchors = BTreeMap::new();
        for d in &documents {
            d.walk(&mut |n| {
                anchors.insert(n.anchor.clone(), n.id.clone());
            });
        }
        DocModel { documents, anchors }
    }

    /// Every node across every document, depth-first (the order a reader meets them).
    #[must_use]
    pub fn all_nodes(&self) -> Vec<&Node> {
        let mut v = Vec::new();
        for d in &self.documents {
            d.walk(&mut |n| v.push(n));
        }
        v
    }

    /// The set of content addresses present in the model (used by the dual-projection-parity lint).
    #[must_use]
    pub fn id_set(&self) -> std::collections::BTreeSet<String> {
        self.all_nodes()
            .iter()
            .map(|n| n.id.as_str().to_owned())
            .collect()
    }
}
