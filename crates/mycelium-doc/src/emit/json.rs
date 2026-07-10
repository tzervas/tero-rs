//! The machine projection (spec §3 — JSON / JSONL for search · tooling · LSP hover, G11). Two views:
//! `doc-model.json` (the whole content-addressed model, serialized) and `search-index.jsonl` (one
//! compact record per node). Both carry each node's content address, so the dual-projection-parity
//! lint can confirm this and the HTML view are two renderers of *one* IR.

use serde::Serialize;

use crate::emit::Artifacts;
use crate::ir::{DocModel, Node};

/// One search-index record (the LSP-hover / search sidecar shape).
#[derive(Debug, Serialize)]
struct IndexRecord<'a> {
    id: &'a str,
    anchor: &'a str,
    kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<&'a str>,
    source: &'a str,
    line: u32,
}

/// Render the machine artifacts: the full model JSON + the JSONL search index.
#[must_use]
pub fn render(model: &DocModel) -> Artifacts {
    let mut arts = Artifacts::new();
    arts.put("doc-model.json", render_model_json(model));
    arts.put("search-index.jsonl", render_search_index(model));
    arts
}

/// The whole model, serialized (pretty) — every node id is present (the parity hook).
#[must_use]
pub fn render_model_json(model: &DocModel) -> String {
    serde_json::to_string_pretty(model).expect("the doc model is always serializable")
}

/// One JSON record per node, newline-delimited (a streamable search/tooling index).
#[must_use]
pub fn render_search_index(model: &DocModel) -> String {
    let mut out = String::new();
    for node in model.all_nodes() {
        let rec = record_for(node);
        out.push_str(&serde_json::to_string(&rec).expect("record is serializable"));
        out.push('\n');
    }
    out
}

fn record_for(node: &Node) -> IndexRecord<'_> {
    IndexRecord {
        id: node.id.as_str(),
        anchor: &node.anchor,
        kind: node.payload.kind_str(),
        title: node.title.as_deref(),
        level: node.level.map(crate::ir::Level::as_str),
        source: &node.provenance.source,
        line: node.provenance.line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{ingest, AnchorAlloc};
    use crate::ir::SourceKind;

    fn model() -> DocModel {
        let mut a = AnchorAlloc::new();
        let src = "# Doc\n\nLead.\n\n## Sec\n\nBody.\n";
        DocModel::new(vec![ingest("d.md", src, SourceKind::Note, &mut a)])
    }

    #[test]
    fn the_model_json_carries_every_node_id() {
        let m = model();
        let json = render_model_json(&m);
        for id in m.id_set() {
            assert!(json.contains(&id), "missing {id} in model JSON");
        }
    }

    #[test]
    fn the_search_index_has_one_line_per_node() {
        let m = model();
        let idx = render_search_index(&m);
        let lines: Vec<&str> = idx.lines().collect();
        assert_eq!(lines.len(), m.all_nodes().len());
        // Each line is valid JSON with an id.
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap()
                .starts_with("blake3:"));
        }
    }
}
