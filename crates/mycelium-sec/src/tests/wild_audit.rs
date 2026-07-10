//! Tests for the `wild`-block audit (M-367) — extracted from the old inline `#[cfg(test)]`
//! block in `lib.rs` (CLAUDE.md test-layout rule; as-touched by M-961).

use crate::*;

#[test]
fn an_unjustified_wild_block_is_flagged() {
    let a = audit_wild(&[(
        "io.myc".to_owned(),
        "nodule io\nfn read(h: Substrate{Handle}) -> Binary{8} =\n    wild { foreign_read(h) }\n"
            .to_owned(),
    )]);
    assert_eq!(a.inventory.len(), 1);
    assert!(!a.inventory[0].justified);
    assert_eq!(a.findings.len(), 1);
    assert_eq!(a.findings[0].rule, "wild-unjustified");
    assert_eq!(a.findings[0].severity, Severity::Medium);
    assert_eq!(a.inventory[0].line, 3);
}

#[test]
fn a_safety_justified_wild_block_passes() {
    // SAFETY on the preceding comment line.
    let a = audit_wild(&[(
        "io.myc".to_owned(),
        "nodule io\nfn read(h: Substrate{Handle}) -> Binary{8} =\n    // SAFETY: h is an affine handle, read once\n    wild { foreign_read(h) }\n".to_owned(),
    )]);
    assert_eq!(a.inventory.len(), 1);
    assert!(a.inventory[0].justified);
    assert!(a.findings.is_empty());
    // SAFETY as a trailing comment on the same line also counts.
    let b = audit_wild(&[(
        "io.myc".to_owned(),
        "nodule io\nfn r() -> Binary{8} = wild { x } // SAFETY: trivial\n".to_owned(),
    )]);
    assert!(b.inventory[0].justified, "{:?}", b.inventory);
}

#[test]
fn wild_in_prose_or_identifiers_is_not_a_false_positive() {
    // `wild` mentioned in a comment is not a block; `wildcard`/`rewild` are not the keyword.
    let a = audit_wild(&[(
        "x.myc".to_owned(),
        "// the wild { } block is unsafe\nnodule x\nfn f(wildcard: Binary{8}) -> Binary{8} = wildcard\n".to_owned(),
    )]);
    assert!(a.inventory.is_empty(), "{:?}", a.inventory);
}

#[test]
fn a_blank_line_breaks_the_justification_block() {
    // A SAFETY comment separated from the wild by a blank line does NOT justify it (G2 — be strict).
    let a = audit_wild(&[(
        "io.myc".to_owned(),
        "nodule io\n// SAFETY: stale, not adjacent\n\nfn r() -> Binary{8} =\n    wild { x }\n"
            .to_owned(),
    )]);
    assert_eq!(a.unjustified(), 1, "{:?}", a.inventory);
}
