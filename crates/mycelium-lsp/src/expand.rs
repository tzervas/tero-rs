//! **Expand ambient** (M-344; RFC-0012 §5): the toolchain projection that renders a document's
//! resolved *longhand* form on demand — the answer to "what does this paradigm-less `{…}` /
//! `default paradigm` actually mean here?". Because the ambient is pure surface elaboration
//! (RFC-0012 I2), the expanded form is the program a reader would write by hand, and it elaborates
//! to the identical L0 (identical content hash). This is the "expand ambient" the editor surfaces
//! so the elided default is never *hidden*, only *elided* (§5).
//!
//! Width resolution needs the checker, so full expansion runs the parse → resolve → check pipeline
//! ([`mycelium_l1::check_and_resolve`]) and pretty-prints the checker-resolved twin
//! ([`mycelium_l1::expand_to_source`]); a parse/check failure is reported, never a partial render.

use mycelium_l1::{check_and_resolve, expand_to_source, parse, parse_nodule_header};

/// Render `text`'s fully-resolved longhand twin (paradigm tags filled, `with paradigm` blocks
/// stripped, bare-decimal widths resolved from context).
///
/// The canonical surface printer drops comments (they are lexer trivia), so a valid `// nodule:`
/// header marker (DN-06 §6) is **preserved** explicitly here — re-emitted in canonical form as the
/// first line (the M-142 formatter wiring). A malformed *named* marker is reported, never silently
/// dropped (G2).
///
/// # Errors
/// Returns the parse/check diagnostic message if the document does not parse or check (so the
/// expansion is never a partial or guessed artifact — G2/never-silent), or the header diagnostic if
/// the `// nodule:` marker is malformed.
pub fn expand_ambient(text: &str) -> Result<String, String> {
    let header = parse_nodule_header(text).map_err(|e| e.to_string())?;
    let nodule = parse(text).map_err(|e| e.to_string())?;
    let (_, twin) = check_and_resolve(&nodule).map_err(|e| e.to_string())?;
    let body = expand_to_source(&twin);
    Ok(match header {
        Some(h) => format!("{}\n{body}", h.canonical()),
        None => body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_a_binary_ambient_to_longhand() {
        let src = "nodule d;\ndefault paradigm Binary;\nfn main() => {8} = not(0b1011_0010);";
        let out = expand_ambient(src).expect("expands");
        assert!(out.contains("Binary{8}"), "{out}");
        assert!(!out.contains("default paradigm"), "{out}");
    }

    #[test]
    fn a_check_failure_is_reported_not_partially_rendered() {
        // A paradigm-less repr with no ambient cannot be expanded — it is an explicit diagnostic.
        let err = expand_ambient("nodule d;\nfn main() => {8} = 0b1011_0010;").unwrap_err();
        assert!(err.contains("no enclosing ambient"), "{err}");
    }

    #[test]
    fn a_valid_nodule_header_marker_is_preserved() {
        // DN-06 §6: the `// nodule:` marker survives the canonical re-print (M-142), even though
        // comments are otherwise dropped.
        let src = "// nodule: signals.demo\nnodule signals.demo;\ndefault paradigm Binary;\nfn main() => {8} = not(0b1011_0010);";
        let out = expand_ambient(src).expect("expands");
        assert!(out.starts_with("// nodule: signals.demo\n"), "{out}");
        assert!(out.contains("Binary{8}"), "{out}");
    }

    #[test]
    fn a_malformed_header_marker_is_reported() {
        let err = expand_ambient("// nodule: 9bad\nnodule d;\nfn main() => Binary{8} = not(0b0);")
            .unwrap_err();
        assert!(err.contains("nodule"), "{err}");
    }
}
