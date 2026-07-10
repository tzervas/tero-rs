//! Renderers — **pure functions of the doc-IR** (spec §3): HTML, Typst (→ PDF) and machine JSON are
//! *views of one content-addressed model*, never parallel truths. Every renderer embeds each node's
//! content address, so the §4.1 `dual-projection-parity` lint can prove HTML and JSON are two views of
//! the same nodes (G11 / ADR-003). EPUB is an honest deferral (see [`mod@crate::build`] notes) — never a
//! half-build.

pub mod html;
pub mod json;
pub mod typst;

use std::collections::BTreeMap;

/// A set of generated artifacts: repo/out-relative path → file contents.
#[derive(Debug, Clone, Default)]
pub struct Artifacts {
    /// path → contents, deterministically ordered.
    pub files: BTreeMap<String, String>,
}

impl Artifacts {
    /// A fresh, empty artifact set.
    #[must_use]
    pub fn new() -> Self {
        Artifacts::default()
    }

    /// Add (or overwrite) one artifact.
    pub fn put(&mut self, path: impl Into<String>, contents: impl Into<String>) {
        self.files.insert(path.into(), contents.into());
    }

    /// Write every artifact under `out_dir`, creating parent directories. Never-silent: returns the
    /// first I/O error with its path.
    ///
    /// # Errors
    /// Propagates the first filesystem error (with the offending path) — never a silent partial write.
    pub fn write_to(&self, out_dir: &std::path::Path) -> std::io::Result<usize> {
        for (rel, contents) in &self.files {
            let path = out_dir.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("creating {}: {e}", parent.display()))
                })?;
            }
            std::fs::write(&path, contents).map_err(|e| {
                std::io::Error::new(e.kind(), format!("writing {}: {e}", path.display()))
            })?;
        }
        Ok(self.files.len())
    }
}

/// Escape text for HTML body content / attribute values.
#[must_use]
pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping_neutralizes_markup() {
        assert_eq!(
            html_escape("<a href=\"x\">&'"),
            "&lt;a href=&quot;x&quot;&gt;&amp;&#39;"
        );
    }

    #[test]
    fn artifacts_keep_insertion_addressable() {
        let mut a = Artifacts::new();
        a.put("index.html", "<x>");
        a.put("a/b.json", "{}");
        assert_eq!(a.files.len(), 2);
        assert_eq!(a.files.get("index.html").unwrap(), "<x>");
    }
}
