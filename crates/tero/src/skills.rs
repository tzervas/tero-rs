//! The `.claude/skills/*/SKILL.md` family — one row per agent skill, from its YAML frontmatter
//! (`name`, `description`, and `when_to_use`).
//!
//! A minimal, purpose-built frontmatter reader (the same honestly-named-subset discipline as
//! `issues.rs`): it reads the `--- … ---` block's top-level scalar / folded-block-scalar keys, not
//! arbitrary YAML. The indexed one-line **summary is the skill's `description`**, verbatim; the
//! `when_to_use` trigger is not separately columned (kept lean) but is one Read away at the row's
//! `file:line` — a recorded scoping choice, not a silent drop (G2). A SKILL.md with no frontmatter
//! or no `name` is **flagged**.

use mycelium_doc::corpus::AnchorAlloc;

use crate::docs::one_line;
use crate::model::{Family, Flagged, TeroIndexItem};
use crate::walk::{collect_ext, repo_rel};

/// Index every `.claude/skills/*/SKILL.md`. `alloc` namespaces skill anchors (`sk--<name>`).
/// Skip-graceful: a missing skills tree yields nothing.
///
/// # Errors
/// Propagates a filesystem error under a present skills tree.
pub fn index_all(
    repo_root: &std::path::Path,
    alloc: &mut AnchorAlloc,
    items: &mut Vec<TeroIndexItem>,
    flagged: &mut Vec<Flagged>,
) -> std::io::Result<()> {
    let root = repo_root.join(".claude/skills");
    for path in collect_ext(&root, "md")? {
        if path.file_name().and_then(|n| n.to_str()) != Some("SKILL.md") {
            continue;
        }
        let rel = repo_rel(repo_root, &path);
        let src = std::fs::read_to_string(&path)?;
        let Some(front) = frontmatter(&src) else {
            flagged.push(Flagged {
                item: rel.clone(),
                reason: "SKILL.md has no `--- … ---` YAML frontmatter — not indexed".to_owned(),
            });
            continue;
        };
        let name = front_field(&front, "name");
        let Some(name) = name.filter(|n| !n.is_empty()) else {
            flagged.push(Flagged {
                item: rel.clone(),
                reason: "SKILL.md frontmatter has no `name:` — not indexed".to_owned(),
            });
            continue;
        };
        let anchor = alloc.alloc(Some("sk"), &name);
        let mut item = TeroIndexItem::new(anchor, Family::Skill, "skill", name, rel.clone(), 1);
        item.summary = front_field(&front, "description").map(|d| one_line(&d, 200));
        items.push(item);
    }
    Ok(())
}

/// The raw frontmatter block: the lines strictly between the first two `---` fence lines. `None`
/// when the file does not open with a `---` fence that later closes.
fn frontmatter(src: &str) -> Option<Vec<String>> {
    let mut lines = src.lines();
    if lines.next().map(str::trim) != Some("---") {
        return None;
    }
    let mut block = Vec::new();
    for line in lines {
        if line.trim() == "---" {
            return Some(block);
        }
        block.push(line.to_owned());
    }
    None
}

/// Read a top-level frontmatter field, joining a folded/literal block scalar (`>`/`>-`/`|`/`|-`)
/// or a plain inline scalar. Continuation lines are the indented lines following the key up to the
/// next top-level (column-0) key. Verbatim from source; `None` when the key is absent.
fn front_field(block: &[String], key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    let start = block.iter().position(|l| l.starts_with(&prefix))?;
    let inline = block[start][prefix.len()..].trim();
    // A non-block, non-empty inline value is the whole field.
    if !inline.is_empty() && !matches!(inline, ">" | ">-" | ">+" | "|" | "|-" | "|+") {
        return Some(dequote(inline));
    }
    // Otherwise gather indented continuation lines (a folded block scalar → space-joined).
    let mut parts = Vec::new();
    for line in &block[start + 1..] {
        if line.trim().is_empty() {
            if !parts.is_empty() {
                // A blank line ends the paragraph for our one-line summary purpose.
                break;
            }
            continue;
        }
        // A new top-level key (no leading whitespace) ends this field.
        if !line.starts_with(char::is_whitespace) {
            break;
        }
        parts.push(line.trim().to_owned());
    }
    let joined = parts.join(" ");
    (!joined.is_empty()).then(|| dequote(&joined))
}

/// Strip a single pair of surrounding quotes (either kind) from a scalar.
fn dequote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if s.len() >= 2
        && ((bytes[0] == b'"' && bytes[s.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[s.len() - 1] == b'\''))
    {
        s[1..s.len() - 1].to_owned()
    } else {
        s.to_owned()
    }
}
