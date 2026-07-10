//! White-box tests for [`crate::header`] (extracted from the inline module — M-790; test-layout rule).

use crate::header::*;
use mycelium_core::cert_mode::CertMode;

#[test]
fn a_full_root_header_parses() {
    let src = "// nodule: geometry.shapes\n\
               // @version: 1.2.0\n\
               // @license: Apache-2.0\n\
               // @authors: Tyler Zervas, A. N. Other\n\
               // @since: 2026-01-10\n\
               // @updated: 2026-06-16\n\
               // @summary: 2D shape primitives.\n\
               // @repository: https://github.com/example/geometry\n\
               // @keywords: geometry, shapes\n\
               // @deprecated: false\n\
               nodule geometry.shapes\n";
    let h = parse_header(src).unwrap().unwrap();
    assert_eq!(h.marker.dotted().as_deref(), Some("geometry.shapes"));
    assert_eq!(h.fields.version.as_deref(), Some("1.2.0"));
    assert_eq!(h.fields.license.as_deref(), Some("Apache-2.0"));
    assert_eq!(h.fields.authors.as_ref().unwrap().len(), 2);
    assert_eq!(h.fields.keywords.as_ref().unwrap(), &["geometry", "shapes"]);
    assert_eq!(h.fields.deprecated, Some(Deprecated::Flag(false)));
}

#[test]
fn a_subnodule_marker_only_has_no_fields() {
    let h = parse_header("// nodule: geometry.shapes.circle\nnodule geometry.shapes.circle\n")
        .unwrap()
        .unwrap();
    assert_eq!(h.fields, HeaderFields::default());
}

#[test]
fn no_marker_means_no_header() {
    assert_eq!(parse_header("fn f() -> Binary{8} = 0b0").unwrap(), None);
}

#[test]
fn an_unknown_key_is_an_explicit_error() {
    let e = parse_header("// nodule: g\n// @authrs: x\n").unwrap_err();
    assert!(e.message.contains("unknown header key"), "{e}");
    assert_eq!(e.line, 2);
}

#[test]
fn a_duplicate_key_is_an_explicit_error() {
    let e = parse_header("// nodule: g\n// @license: MIT\n// @license: MIT\n").unwrap_err();
    assert!(e.message.contains("duplicate"), "{e}");
}

#[test]
fn bad_values_are_explicit_errors() {
    assert!(parse_header("// nodule: g\n// @license: NotARealLicense\n").is_err());
    assert!(parse_header("// nodule: g\n// @since: 2026-13-40\n").is_err());
    assert!(parse_header("// nodule: g\n// @updated: yesterday\n").is_err());
    assert!(parse_header("// nodule: g\n// @version: 1.x\n").is_err());
    assert!(parse_header("// nodule: g\n// @repository: not a url\n").is_err());
}

#[test]
fn deprecated_can_carry_a_reason() {
    let h = parse_header("// nodule: g\n// @deprecated: use geometry.v2 instead\n")
        .unwrap()
        .unwrap();
    assert_eq!(
        h.fields.deprecated,
        Some(Deprecated::Reason("use geometry.v2 instead".to_owned()))
    );
}

// RFC-0017: @matured is a boolean key inherited top-down.
#[test]
fn matured_true_parses() {
    // Mutant-witness: removing the `"matured"` arm in set_field would make this panic.
    let h = parse_header("// nodule: g\n// @matured: true\n")
        .unwrap()
        .unwrap();
    assert_eq!(h.fields.matured, Some(true));
}

#[test]
fn matured_false_parses() {
    // Mutant-witness: swapping true/false arms would make this fail.
    let h = parse_header("// nodule: g\n// @matured: false\n")
        .unwrap()
        .unwrap();
    assert_eq!(h.fields.matured, Some(false));
}

#[test]
fn matured_bad_value_is_explicit_error() {
    // Mutant-witness: removing the bad() call for unknown values would make this succeed.
    let e = parse_header("// nodule: g\n// @matured: yes\n").unwrap_err();
    assert!(
        e.message.contains("a boolean `true` or `false`"),
        "expected boolean error, got: {e}"
    );
    assert_eq!(e.line, 2);
}

// RFC-0034 §6 / M-790: @certification is a closed-set mode key (the nodule tier).
#[test]
fn certification_parses_each_mode() {
    for (word, mode) in [
        ("fast", CertMode::Fast),
        ("balanced", CertMode::Balanced),
        ("certified", CertMode::Certified),
    ] {
        let src = format!("// nodule: g\n// @certification: {word}\n");
        let h = parse_header(&src).unwrap().unwrap();
        assert_eq!(h.fields.certification, Some(mode), "for {word}");
    }
}

#[test]
fn certification_unknown_mode_is_explicit_error() {
    // Never-silent (G2): an out-of-set mode word is rejected, not defaulted.
    let e = parse_header("// nodule: g\n// @certification: turbo\n").unwrap_err();
    assert!(e.message.contains("unknown @certification mode"), "{e}");
    assert_eq!(e.line, 2);
}

#[test]
fn validators_are_honest() {
    assert!(is_iso_date("2026-06-16"));
    assert!(!is_iso_date("2026-6-16"));
    assert!(!is_iso_date("2026-13-01"));
    assert!(is_semver("1.2.0"));
    assert!(is_semver("1.2.0-rc.1"));
    assert!(!is_semver("1.2"));
    assert!(is_spdx("MIT"));
    assert!(is_spdx("MIT OR Apache-2.0"));
    assert!(!is_spdx("Bogus-9.9"));
}
