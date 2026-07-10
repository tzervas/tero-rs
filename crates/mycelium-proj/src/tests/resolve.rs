//! White-box tests for [`crate::resolve`] (extracted from the inline module — M-790; test-layout rule).

use crate::cert_scope::CertScope;
use crate::header::parse_header;
use crate::manifest::{parse_manifest, Manifest};
use crate::resolve::*;
use mycelium_core::cert_mode::CertMode;

fn manifest() -> Manifest {
    parse_manifest(
        "[project]\nname=\"geometry\"\nkind=\"phylum\"\nversion=\"1.2.0\"\n\
         license=\"Apache-2.0\"\nauthors=[\"Tyler Zervas\"]\nsince=\"2026-01-10\"\n\
         repository=\"https://github.com/example/geometry\"\nkeywords=[\"geometry\"]\n",
    )
    .unwrap()
}

#[test]
fn a_subnodule_inherits_from_the_manifest() {
    let h = parse_header("// nodule: geometry.shapes.circle\n// @updated: 2026-06-16\n")
        .unwrap()
        .unwrap();
    let r = resolve(&h, Some(&manifest()));
    // Inherited from the manifest.
    assert_eq!(r.license.as_ref().unwrap().value, "Apache-2.0");
    assert_eq!(r.license.as_ref().unwrap().origin, Origin::ProjectManifest);
    assert_eq!(r.version.as_ref().unwrap().origin, Origin::ProjectManifest);
    // Per-file: local, never inherited.
    assert_eq!(r.updated.as_deref(), Some("2026-06-16"));
}

#[test]
fn a_local_value_overrides_the_manifest() {
    let h = parse_header("// nodule: geometry.shapes\n// @license: MIT\n")
        .unwrap()
        .unwrap();
    let r = resolve(&h, Some(&manifest()));
    assert_eq!(r.license.as_ref().unwrap().value, "MIT");
    assert_eq!(r.license.as_ref().unwrap().origin, Origin::Local);
}

#[test]
fn explain_names_every_source() {
    let h = parse_header("// nodule: geometry.shapes\n// @license: MIT\n// @updated: 2026-06-16\n")
        .unwrap()
        .unwrap();
    let r = resolve(&h, Some(&manifest()));
    let ex = explain(&r);
    assert!(ex.contains("license: MIT  [local]"), "{ex}");
    assert!(ex.contains("version: 1.2.0  [mycelium-proj.toml]"), "{ex}");
    assert!(ex.contains("updated: 2026-06-16  [local]"), "{ex}");
}

#[test]
fn no_manifest_means_only_local_fields_resolve() {
    let h = parse_header("// nodule: solo\n// @license: MIT\n")
        .unwrap()
        .unwrap();
    let r = resolve(&h, None);
    assert_eq!(r.license.as_ref().unwrap().origin, Origin::Local);
    assert!(r.version.is_none());
}

// --- RFC-0034 §6 / M-790: certification mode resolution through the header resolver ---

fn manifest_with_cert(mode: &str) -> Manifest {
    parse_manifest(&format!(
        "[project]\nname=\"g\"\nkind=\"phylum\"\ncertification=\"{mode}\"\n"
    ))
    .unwrap()
}

#[test]
fn certification_defaults_to_fast_when_unset_everywhere() {
    // No declaration at any scope ⇒ the project default `fast`, source `None` (the default fallback).
    let h = parse_header("// nodule: g\n").unwrap().unwrap();
    let r = resolve(&h, None);
    assert_eq!(r.certification.mode, CertMode::Fast);
    assert_eq!(r.certification.source, None);
}

#[test]
fn certification_inherits_from_the_manifest_phylum_tier() {
    let h = parse_header("// nodule: g\n").unwrap().unwrap();
    let r = resolve(&h, Some(&manifest_with_cert("certified")));
    assert_eq!(r.certification.mode, CertMode::Certified);
    assert_eq!(r.certification.source, Some(CertScope::Phylum));
}

#[test]
fn nodule_certification_overrides_the_phylum_tier() {
    // The DoD precedence law (concrete instance): nodule `@certification` beats the manifest.
    let h = parse_header("// nodule: g\n// @certification: fast\n")
        .unwrap()
        .unwrap();
    let r = resolve(&h, Some(&manifest_with_cert("certified")));
    assert_eq!(r.certification.mode, CertMode::Fast);
    assert_eq!(r.certification.source, Some(CertScope::Nodule));
}

#[test]
fn explain_names_the_certification_source() {
    let h = parse_header("// nodule: g\n// @certification: balanced\n")
        .unwrap()
        .unwrap();
    let r = resolve(&h, Some(&manifest_with_cert("certified")));
    let ex = explain(&r);
    // Most-specific wins, and the source is named — never ambient (G2).
    assert!(ex.contains("certification: balanced  [nodule]"), "{ex}");

    // Inherited case names the phylum tier.
    let h2 = parse_header("// nodule: g\n").unwrap().unwrap();
    let r2 = resolve(&h2, Some(&manifest_with_cert("certified")));
    assert!(
        explain(&r2).contains("certification: certified  [phylum]"),
        "{}",
        explain(&r2)
    );

    // Defaulted case names `default`.
    let r3 = resolve(&h2, None);
    assert!(
        explain(&r3).contains("certification: fast  [default]"),
        "{}",
        explain(&r3)
    );
}
