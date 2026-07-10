//! White-box tests for [`crate::apiref`] — extracted from the logic file (as-touched, CLAUDE.md
//! test layout rule) when the `=>` return-arrow split bug (M-1004) was fixed. Uses `pub(crate)`
//! access to `nodule_name`, `fn_signatures`, `preceding_doc`, `fn_name`, `path_stem`.

use crate::apiref::*;
use crate::corpus::AnchorAlloc;
use crate::ir::Payload;

const SRC: &str = "// nodule: hello.greeting\n\
                   // @summary: A greeting nodule.\n\
                   nodule hello.greeting\n\
                   \n\
                   fn wave() -> Ternary{4} =\n\
                     <+0-0>\n";

#[test]
fn a_documented_nodule_carries_its_summary() {
    let mut a = AnchorAlloc::new();
    let doc = project_nodule("examples/hello/greeting.myc", SRC, &mut a);
    let nodule_item = doc
        .children
        .iter()
        .find_map(|n| match &n.payload {
            Payload::ApiItem { summary, .. }
                if n.title.as_deref() == Some("nodule hello.greeting") =>
            {
                Some(summary.clone())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(nodule_item.as_deref(), Some("A greeting nodule."));
}

#[test]
fn an_undocumented_fn_is_flagged_never_invented() {
    let mut a = AnchorAlloc::new();
    let doc = project_nodule("x.myc", SRC, &mut a);
    let fn_item = doc
        .children
        .iter()
        .find(|n| n.title.as_deref() == Some("fn wave() -> Ternary{4}"))
        .unwrap();
    match &fn_item.payload {
        Payload::ApiItem { summary, signature } => {
            assert!(summary.is_none(), "undocumented, never invented");
            assert_eq!(signature.as_deref(), Some("fn wave() -> Ternary{4}"));
        }
        _ => panic!("expected an api-item"),
    }
}

#[test]
fn a_fn_with_a_preceding_comment_is_documented_from_source() {
    // The contiguous `//` block above a `fn` becomes its summary (M-736); a `@`/`nodule`
    // header or a blank line bounds the block, so the nodule header never leaks into a fn doc.
    const DOC_SRC: &str = "// nodule: m\n\
                           nodule m\n\
                           \n\
                           // add: combine two bytes. Why: the running total step.\n\
                           // It is total and never-silent.\n\
                           fn add(a: Binary{8}, b: Binary{8}) -> Binary{8} = a\n";
    let mut a = AnchorAlloc::new();
    let doc = project_nodule("x.myc", DOC_SRC, &mut a);
    let fn_item = doc
        .children
        .iter()
        .find(|n| n.title.as_deref() == Some("fn add(a: Binary{8}, b: Binary{8}) -> Binary{8}"))
        .unwrap();
    match &fn_item.payload {
        Payload::ApiItem { summary, .. } => {
            assert_eq!(
                summary.as_deref(),
                Some(
                    "add: combine two bytes. Why: the running total step. It is total and never-silent."
                ),
                "the two-line source comment is joined verbatim — traces to source, never invented"
            );
        }
        _ => panic!("expected an api-item"),
    }
}

#[test]
fn the_whole_source_is_a_checked_example() {
    let mut a = AnchorAlloc::new();
    let doc = project_nodule("x.myc", SRC, &mut a);
    let ex = doc
        .children
        .iter()
        .find_map(|n| match &n.payload {
            Payload::Example {
                checked, source, ..
            } => Some((*checked, source.clone())),
            _ => None,
        })
        .unwrap();
    assert!(ex.0);
    assert!(ex.1.contains("fn wave"));
}

#[test]
fn a_schema_projects_its_fields_with_undocumented_gaps() {
    let mut a = AnchorAlloc::new();
    let schema = r#"{
        "title": "Bound",
        "description": "A numeric bound.",
        "properties": {
            "kind": {"type": "string", "description": "The bound kind."},
            "value": {"type": "number"}
        }
    }"#;
    let doc = project_schema("docs/spec/schemas/bound.schema.json", schema, &mut a).unwrap();
    let documented = doc
        .children
        .iter()
        .filter(|n| {
            matches!(
                &n.payload,
                Payload::ApiItem {
                    summary: Some(_),
                    ..
                }
            )
        })
        .count();
    let undocumented = doc
        .children
        .iter()
        .filter(|n| matches!(&n.payload, Payload::ApiItem { summary: None, .. }))
        .count();
    assert_eq!(documented, 1, "kind is documented");
    assert_eq!(undocumented, 1, "value is an explicit undocumented gap");
}

// --- M-1004 regression: the `=>` fat-arrow return-type split bug ------------------------------

#[test]
fn fn_signatures_keeps_the_fat_arrow_return_type() {
    // Every real `.myc` file under `lib/` (found while building the M-1004 lib-index extractor)
    // writes the return type with `=>`, e.g. `fn is_lt(o: Ordering) => Bool =`. The original
    // `split_once('=')` cut found the `=` *inside* `=>` first and silently dropped `=> Bool` —
    // never caught because the pre-existing test fixtures only used the `->` spelling. Pinned here.
    const SRC: &str = "// nodule: std.cmp\n\
                       nodule std.cmp;\n\
                       \n\
                       fn is_lt(o: Ordering) => Bool =\n\
                         match o { Lt => True, Eq => False, Gt => False };\n";
    let sigs = fn_signatures(SRC);
    assert_eq!(
        sigs,
        vec![("fn is_lt(o: Ordering) => Bool".to_owned(), 4)],
        "the fat-arrow return type must survive the signature split"
    );
}

#[test]
fn fn_signatures_still_handles_the_thin_arrow_spelling() {
    // Backward-compat: the pre-M-1004 fixtures use `->` (no body-separator ambiguity at all —
    // there is exactly one bare `=` on the line). The fix must not regress this spelling.
    const SRC: &str = "fn wave() -> Ternary{4} =\n  <+0-0>\n";
    let sigs = fn_signatures(SRC);
    assert_eq!(sigs, vec![("fn wave() -> Ternary{4}".to_owned(), 1)]);
}

#[test]
fn fn_signatures_handles_an_inline_body_after_the_fat_arrow() {
    // A one-line fn body (`fn zero32() => Binary{32} = 0b0...;`, common in lib/compiler/*.myc)
    // must still split at the body `=`, not the arrow's `=`.
    const SRC: &str = "fn zero32() => Binary{32} = 0b0000_0000;\n";
    let sigs = fn_signatures(SRC);
    assert_eq!(sigs, vec![("fn zero32() => Binary{32}".to_owned(), 1)]);
}

#[test]
fn nodule_name_reads_the_dotted_declaration() {
    assert_eq!(
        nodule_name("// nodule: std.cmp\nnodule std.cmp;\n"),
        Some("std.cmp".to_owned())
    );
}

#[test]
fn path_stem_strips_directory_and_extension() {
    assert_eq!(path_stem("lib/std/cmp.myc"), "cmp");
}

#[test]
fn fn_name_reads_the_leading_identifier() {
    assert_eq!(
        fn_name("fn is_lt(o: Ordering) => Bool"),
        Some("is_lt".to_owned())
    );
}
