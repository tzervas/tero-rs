//! End-to-end **conformance fixtures** (M-359): real `mycelium-proj.toml` + `.myc` header files run
//! through the whole pipeline — parse manifest, parse header, resolve inheritance, EXPLAIN — and a
//! malformed header is an explicit error (G2). The fixtures mirror the spec §3/§4 examples.

use std::fs;
use std::path::PathBuf;

use mycelium_proj::{explain, parse_header, parse_manifest, resolve, Origin};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

#[test]
fn a_root_nodule_resolves_locally_and_explains() {
    let manifest = parse_manifest(&fixture("mycelium-proj.toml")).expect("manifest parses");
    let header = parse_header(&fixture("root.myc"))
        .expect("header parses")
        .expect("has a marker");
    let r = resolve(&header, Some(&manifest));

    // The root declares its own license — local, not inherited.
    assert_eq!(r.license.as_ref().unwrap().value, "Apache-2.0");
    assert_eq!(r.license.as_ref().unwrap().origin, Origin::Local);
    assert_eq!(r.name.as_ref().unwrap(), &["geometry", "shapes"]);

    let ex = explain(&r);
    assert!(ex.contains("license: Apache-2.0  [local]"), "{ex}");
}

#[test]
fn a_subnodule_inherits_from_the_manifest() {
    let manifest = parse_manifest(&fixture("mycelium-proj.toml")).expect("manifest parses");
    let header = parse_header(&fixture("subnodule.myc"))
        .expect("header parses")
        .expect("has a marker");
    let r = resolve(&header, Some(&manifest));

    // The subnodule carries only `@updated`; license/version/authors inherit from the manifest.
    assert_eq!(r.license.as_ref().unwrap().origin, Origin::ProjectManifest);
    assert_eq!(r.version.as_ref().unwrap().value, "1.2.0");
    assert_eq!(r.version.as_ref().unwrap().origin, Origin::ProjectManifest);
    assert_eq!(r.updated.as_deref(), Some("2026-06-16"));

    let ex = explain(&r);
    assert!(
        ex.contains("license: Apache-2.0  [mycelium-proj.toml]"),
        "{ex}"
    );
    assert!(ex.contains("updated: 2026-06-16  [local]"), "{ex}");
}

#[test]
fn a_malformed_header_is_an_explicit_error() {
    let err = parse_header(&fixture("bad-header.myc")).unwrap_err();
    assert!(err.message.contains("license"), "{err}");
    assert_eq!(err.line, 2);
}
