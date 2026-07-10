use crate::nodule::*;

#[test]
fn named_marker_is_recognised() {
    let h = parse_nodule_header("// nodule: geometry.shapes\nnodule geometry.shapes;\n")
        .unwrap()
        .unwrap();
    assert_eq!(h.dotted().as_deref(), Some("geometry.shapes"));
    assert_eq!(h.canonical(), "// nodule: geometry.shapes");
}

#[test]
fn bare_marker_is_recognised() {
    let h = parse_nodule_header("// nodule\nnodule g.s;\n")
        .unwrap()
        .unwrap();
    assert_eq!(h.name, None);
    assert_eq!(h.canonical(), "// nodule");
}

#[test]
fn leading_blank_lines_are_skipped() {
    let h = parse_nodule_header("\n\n   \n// nodule: a.b\n")
        .unwrap()
        .unwrap();
    assert_eq!(h.dotted().as_deref(), Some("a.b"));
}

#[test]
fn an_ordinary_first_comment_is_not_a_marker() {
    assert_eq!(
        parse_nodule_header("// just a comment\nnodule d;\n").unwrap(),
        None
    );
    // `nodule` mentioned in prose (no colon, not bare) is not a marker — no false positive.
    assert_eq!(
        parse_nodule_header("// nodule is Mycelium's word for module\nnodule d;\n").unwrap(),
        None
    );
}

#[test]
fn code_first_means_no_marker() {
    assert_eq!(
        parse_nodule_header("nodule d;\nfn f() => Binary{8} = 0b0;").unwrap(),
        None
    );
}

#[test]
fn empty_named_marker_is_an_explicit_error() {
    let e = parse_nodule_header("// nodule:\n").unwrap_err();
    assert_eq!(e.line, 1);
    assert!(e.message.contains("must name the nodule"), "{}", e.message);
}

#[test]
fn ill_formed_name_is_an_explicit_error() {
    assert!(parse_nodule_header("// nodule: 9bad\n").is_err());
    assert!(parse_nodule_header("// nodule: a..b\n").is_err());
    assert!(parse_nodule_header("// nodule: a.b.\n").is_err());
    assert!(parse_nodule_header("// nodule: has space\n").is_err());
}

#[test]
fn empty_source_has_no_marker() {
    assert_eq!(parse_nodule_header("").unwrap(), None);
    assert_eq!(parse_nodule_header("\n  \n").unwrap(), None);
}
