//! White-box tests for [`crate::manifest`] (extracted from the inline module — M-790; test-layout rule).

use crate::manifest::*;
use mycelium_core::cert_mode::CertMode;

const SAMPLE: &str = r#"
# mycelium-proj.toml
[project]
name        = "geometry"          # the project name
kind        = "phylum"
version     = "1.2.0"
license     = "Apache-2.0"
authors     = ["Tyler Zervas", "A. N. Other"]
since       = "2026-01-10"
summary     = "2D/3D geometry primitives and certified swaps."
repository  = "https://github.com/example/geometry"
keywords    = ["geometry", "linear-algebra"]
lang        = "mycelium-0"

[surface]
exports     = ["geometry.shapes"]

[dependencies]
numerics    = { phylum = "numerics", version = "^2", hash = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" }
"#;

/// A real (64-hex) blake3 pin used by the positive dependency tests.
const VALID_PIN: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn the_sample_manifest_parses() {
    let m = parse_manifest(SAMPLE).unwrap();
    assert_eq!(m.project.name, "geometry");
    assert_eq!(m.project.kind, ProjectKind::Phylum);
    assert_eq!(m.project.version.as_deref(), Some("1.2.0"));
    assert_eq!(m.project.license.as_deref(), Some("Apache-2.0"));
    assert_eq!(m.project.authors.as_ref().unwrap().len(), 2);
    assert_eq!(
        m.project.keywords.as_ref().unwrap(),
        &["geometry", "linear-algebra"]
    );
}

#[test]
fn missing_required_fields_are_explicit_errors() {
    assert!(parse_manifest("[project]\nname = \"x\"\n").is_err()); // no kind
    assert!(parse_manifest("[project]\nkind = \"program\"\n").is_err()); // no name
    assert!(parse_manifest("[surface]\nexports = []\n").is_err()); // no [project]
}

#[test]
fn an_unknown_project_key_is_an_explicit_error() {
    let e = parse_manifest("[project]\nname=\"x\"\nkind=\"script\"\nfoo=\"bar\"\n").unwrap_err();
    assert!(e.message.contains("unknown `[project]` key"), "{e}");
}

#[test]
fn bad_kind_and_values_are_explicit_errors() {
    assert!(parse_manifest("[project]\nname=\"x\"\nkind=\"library\"\n").is_err());
    assert!(parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\nlicense=\"Nope\"\n").is_err());
    assert!(
        parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\nsince=\"2026/01/10\"\n").is_err()
    );
}

#[test]
fn the_toolchain_table_is_interpreted() {
    // M-364: `[toolchain].format` is now read (the manifest's first toolchain consumer).
    let m = parse_manifest(SAMPLE).unwrap();
    assert!(m.toolchain.is_none(), "SAMPLE declares no [toolchain]");
    let m = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[toolchain]\nformat=\"mycfmt-0\"\nlints=\"strict\"\n",
    )
    .unwrap();
    let tc = m.toolchain.unwrap();
    assert_eq!(tc.format.as_deref(), Some("mycfmt-0"));
    assert_eq!(tc.lints.as_deref(), Some("strict"));
}

#[test]
fn the_surface_dependencies_and_spore_tables_are_interpreted() {
    // M-368: the packaging tables are now typed (first consumer).
    let m = parse_manifest(SAMPLE).unwrap();
    assert_eq!(
        m.surface.as_ref().unwrap().exports,
        vec!["geometry.shapes".to_owned()]
    );
    assert_eq!(m.dependencies.len(), 1);
    let d = &m.dependencies[0];
    assert_eq!(d.name, "numerics");
    assert_eq!(d.phylum, "numerics");
    // The pin is parsed into a typed `ContentHash` (DN-40 A3), not free text.
    assert_eq!(d.hash.as_ref().map(|h| h.as_str()), Some(VALID_PIN));
    assert_eq!(d.version.as_deref(), Some("^2"));

    let m =
        parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\n[spore]\ninclude=[\"surface\"]\n")
            .unwrap();
    assert_eq!(m.spore.unwrap().include, vec!["surface".to_owned()]);
}

#[test]
fn a_malformed_dependency_or_unknown_key_is_explicit() {
    // A non-inline-table dependency is an error.
    assert!(parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\nfoo=\"bar\"\n"
    )
    .is_err());
    // An unknown dependency key is an error (closed set).
    let e = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\nfoo={ phylum=\"f\", oops=\"x\" }\n",
    )
    .unwrap_err();
    assert!(e.message.contains("unknown dependency key"), "{e}");
    // An unknown [surface] key is an error.
    assert!(
        parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\n[surface]\nexprts=[\"a\"]\n")
            .is_err()
    );
}

// DN-40 A3: the dependency `hash` is the identity-bearing pin (ADR-003) and is **parsed**, not
// accepted as free text. A malformed pin is an explicit `ManifestError` at manifest-build time, so
// it can never flow downstream into a spore's identity edge.
#[test]
fn a_valid_dependency_hash_is_parsed_into_a_typed_pin() {
    let src = format!(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\n\
         numerics={{ phylum=\"numerics\", hash=\"{VALID_PIN}\" }}\n"
    );
    let m = parse_manifest(&src).unwrap();
    let d = &m.dependencies[0];
    assert_eq!(d.hash.as_ref().map(|h| h.as_str()), Some(VALID_PIN));
    assert_eq!(d.hash.as_ref().unwrap().algo(), "blake3");
}

#[test]
fn a_shape_malformed_dependency_hash_is_an_explicit_error() {
    // No `:` separator — not even an `<algo>:<digest>` shape.
    let e = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\n\
         numerics={ phylum=\"numerics\", hash=\"nocolon\" }\n",
    )
    .unwrap_err();
    assert!(e.message.contains("malformed content-address"), "{e}");
    assert!(e.message.contains("numerics"), "{e}"); // names the offending dependency (G2)
    assert!(e.message.contains("DN-40 A3"), "{e}");
}

#[test]
fn a_bogus_but_shaped_dependency_hash_is_rejected() {
    // `blake3:abc` is shape-valid but is NOT a real 64-hex blake3 digest — it must be refused, not
    // silently accepted as a pin (DN-40 A3: the inverted parse-don't-validate this fix closes).
    let e = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\n\
         numerics={ phylum=\"numerics\", hash=\"blake3:abc\" }\n",
    )
    .unwrap_err();
    assert!(e.message.contains("64 lowercase hex"), "{e}");
    assert!(e.message.contains("numerics"), "{e}");
}

#[test]
fn an_unknown_toolchain_key_is_an_explicit_error() {
    let e = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[toolchain]\nformatt=\"mycfmt-0\"\n",
    )
    .unwrap_err();
    assert!(e.message.contains("unknown `[toolchain]` key"), "{e}");
}

#[test]
fn out_of_subset_constructs_are_explicit_errors() {
    // A bare number is outside the v0 subset — flagged, never silently dropped.
    let e = parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\nversion=12\n").unwrap_err();
    assert!(e.message.contains("v0 manifest reader supports"), "{e}");
}

// RFC-0034 §6 / M-790: `[project].certification` is the phylum-tier mode declaration.
#[test]
fn certification_parses_in_the_project_table() {
    let m = parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\ncertification=\"certified\"\n")
        .unwrap();
    assert_eq!(m.project.certification, Some(CertMode::Certified));
}

#[test]
fn certification_absent_is_none_not_a_silent_default() {
    // Honest absence: the manifest carries `None`; the *resolver* applies the project default,
    // never the parser silently (G2).
    let m = parse_manifest(SAMPLE).unwrap();
    assert_eq!(m.project.certification, None);
}

#[test]
fn certification_unknown_mode_is_explicit_error() {
    let e = parse_manifest("[project]\nname=\"x\"\nkind=\"phylum\"\ncertification=\"turbo\"\n")
        .unwrap_err();
    assert!(e.message.contains("unknown @certification mode"), "{e}");
}

// --- DN-40 §3 input-validation: duplicate keys are rejected, never silently last-wins ---

#[test]
fn a_duplicate_project_key_is_rejected_not_last_wins() {
    // Two `name =` lines would otherwise let the second silently overwrite the first (G2).
    let e = parse_manifest("[project]\nname=\"a\"\nkind=\"phylum\"\nname=\"b\"\n").unwrap_err();
    assert!(e.message.contains("duplicate"), "{e}");
    assert!(
        e.message.contains("name"),
        "should name the offending key: {e}"
    );
}

#[test]
fn a_duplicate_key_in_each_interpreted_table_is_rejected() {
    // toolchain
    assert!(parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[toolchain]\nlints=\"a\"\nlints=\"b\"\n"
    )
    .unwrap_err()
    .message
    .contains("duplicate"));
    // surface
    assert!(parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[surface]\nexports=[]\nexports=[\"a\"]\n"
    )
    .unwrap_err()
    .message
    .contains("duplicate"));
    // dependencies (a repeated dependency name)
    assert!(parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\nd={ phylum=\"d\" }\nd={ phylum=\"d\" }\n"
    )
    .unwrap_err()
    .message
    .contains("duplicate"));
    // spore
    assert!(parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[spore]\ninclude=[]\ninclude=[\"s\"]\n"
    )
    .unwrap_err()
    .message
    .contains("duplicate"));
}

#[test]
fn a_duplicate_table_is_rejected() {
    let e = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[toolchain]\nformat=\"a\"\n[toolchain]\nlints=\"b\"\n",
    )
    .unwrap_err();
    assert!(e.message.contains("duplicate"), "{e}");
    assert!(
        e.message.contains("toolchain"),
        "should name the offending table: {e}"
    );
}

// --- DN-40 §3 input-validation: deeply-nested TOML is refused, never a stack-exhausting crash ---

#[test]
fn deeply_nested_array_is_refused_not_a_crash() {
    // A single-line value nested far past MAX_VALUE_DEPTH (16). The parser must return an explicit
    // error, not recurse to stack exhaustion (DoS bound, G2).
    let nesting = "[".repeat(2_000);
    let src = format!("[project]\nname=\"x\"\nkind=\"phylum\"\nkeywords={nesting}\n");
    let e = parse_manifest(&src).unwrap_err();
    assert!(e.message.contains("nests deeper"), "{e}");
}

#[test]
fn deeply_nested_inline_table_is_refused_not_a_crash() {
    // Deeply-nested inline tables hit the same depth cap.
    let nesting = "{ a = ".repeat(2_000);
    let src = format!("[project]\nname=\"x\"\nkind=\"phylum\"\nlang={nesting}\n");
    let e = parse_manifest(&src).unwrap_err();
    assert!(e.message.contains("nests deeper"), "{e}");
}

#[test]
fn a_shallow_nested_value_within_the_depth_limit_still_parses() {
    // A normal dependency inline-table (depth 1) and a small list are well within the cap — the
    // bound refuses pathological input without rejecting legitimate manifests.
    let m = parse_manifest(
        "[project]\nname=\"x\"\nkind=\"phylum\"\n[dependencies]\nd={ phylum=\"d\", version=\"^1\" }\n",
    )
    .unwrap();
    assert_eq!(m.dependencies.len(), 1);
}
