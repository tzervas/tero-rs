//! White-box tests for [`crate::lib_index`] (M-1004) — the `docs/lib-index/` extractor. Uses
//! `pub(crate)` access to the per-file extraction internals (`type_declarations`, `full_summary`,
//! `unrecognized_top_level`, `nodule_decl_line`, `index_file`) plus a hermetic temp-dir fixture
//! (the `book.rs`/`build.rs` precedent) for the full `build_lib_index` walk.

use std::fs;
use std::path::{Path, PathBuf};

use crate::lib_index::*;

fn temp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("mycdoc-libindex-{tag}-{nanos}"));
    fs::create_dir_all(&p).unwrap();
    p
}

fn write(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

// ── `full_summary` (the forward-joining multi-line `@summary` reader) ──────────────────────────

#[test]
fn full_summary_joins_multiline_continuation() {
    let src = "// nodule: compiler.ambient\n\
               // @version: 0.1.0\n\
               // @summary: Self-hosted ambient resolution. A\n\
               //   faithful port of the Rust module, over a copy\n\
               //   of the AST vocabulary.\n\
               nodule compiler.ambient;\n";
    assert_eq!(
        full_summary(src).as_deref(),
        Some(
            "Self-hosted ambient resolution. A faithful port of the Rust module, over a copy of \
             the AST vocabulary."
        )
    );
}

#[test]
fn full_summary_stops_at_the_nodule_marker_line() {
    // A single-line `@summary` is followed directly by the `nodule X.Y;` statement (no blank
    // line) — the scan must not swallow it as "continuation prose".
    let src = "// nodule: std.cmp\n// @summary: Ordering surface.\nnodule std.cmp;\n";
    assert_eq!(full_summary(src).as_deref(), Some("Ordering surface."));
}

#[test]
fn full_summary_is_none_without_an_at_summary_line() {
    let src = "// nodule: std.cmp\nnodule std.cmp;\n";
    assert_eq!(full_summary(src), None);
}

// ── `type_declarations` (new extraction — not in `apiref.rs`) ──────────────────────────────────

#[test]
fn type_declarations_reads_a_single_line_enum() {
    let src = "type Ordering = Lt | Eq | Gt;\n";
    let (decls, problems) = type_declarations(src);
    assert!(problems.is_empty());
    assert_eq!(decls.len(), 1);
    let d = &decls[0];
    assert_eq!(d.name, "Ordering");
    assert_eq!(d.start_line, 1);
    assert_eq!(d.text, "type Ordering = Lt | Eq | Gt");
    let names: Vec<&str> = d.ctors.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(names, vec!["Lt", "Eq", "Gt"]);
    // Single-line declaration: every constructor is attributed to that one line.
    assert!(d.ctors.iter().all(|(_, _, line)| *line == 1));
}

#[test]
fn type_declarations_generic_name_stops_at_the_bracket() {
    let src = "type Option[A] = Some(A) | None;\n";
    let (decls, _) = type_declarations(src);
    assert_eq!(decls[0].name, "Option");
}

#[test]
fn type_declarations_reads_a_multiline_enum_with_per_ctor_lines() {
    // The real `lib/compiler/ambient.myc::AmbientParams` shape: one constructor per line,
    // `|`-prefixed continuations. Each constructor must attribute to ITS OWN line, not the
    // type's start line.
    let src = "type AmbientParams =\n\
               \x20   APSize(Binary{32})\n\
               \x20 | APDense(Binary{32}, Scalar)\n\
               \x20 | APVsa(Bytes, Binary{32}, Sparsity);\n";
    let (decls, problems) = type_declarations(src);
    assert!(problems.is_empty());
    assert_eq!(decls.len(), 1);
    let d = &decls[0];
    assert_eq!(d.name, "AmbientParams");
    assert_eq!(d.start_line, 1);
    assert_eq!(
        d.ctors,
        vec![
            ("APSize".to_owned(), "APSize(Binary{32})".to_owned(), 2),
            (
                "APDense".to_owned(),
                "APDense(Binary{32}, Scalar)".to_owned(),
                3
            ),
            (
                "APVsa".to_owned(),
                "APVsa(Bytes, Binary{32}, Sparsity)".to_owned(),
                4
            ),
        ]
    );
}

#[test]
fn type_declarations_drops_embedded_divider_comments_from_body_and_ctors() {
    // The real `lib/compiler/token.myc::Tok` shape: full-line `// section divider` comments
    // INTERLEAVED inside the multi-line type block (PR #1206 review HIGH). Comment text must
    // reach neither the ctor list nor the type's joined `text`; line attribution stays exact.
    let src = "type Tok =\n\
               \x20   // structural keywords (DN-02)\n\
               \x20   Nodule | Phylum | Colony\n\
               \x20   // punctuation\n\
               \x20 | FloatLit(Bytes)\n\
               \x20 | LParen;\n";
    let (decls, problems) = type_declarations(src);
    assert!(problems.is_empty());
    assert_eq!(decls.len(), 1);
    let d = &decls[0];
    assert_eq!(d.name, "Tok");
    // No empty-named ctor, no comment prose spliced onto a neighbor.
    assert_eq!(
        d.ctors,
        vec![
            ("Nodule".to_owned(), "Nodule".to_owned(), 3),
            ("Phylum".to_owned(), "Phylum".to_owned(), 3),
            ("Colony".to_owned(), "Colony".to_owned(), 3),
            ("FloatLit".to_owned(), "FloatLit(Bytes)".to_owned(), 5),
            ("LParen".to_owned(), "LParen".to_owned(), 6),
        ]
    );
    // The joined declaration text is comment-free too.
    assert!(
        !d.text.contains("//"),
        "comment spliced into text: {}",
        d.text
    );
}

#[test]
fn type_declarations_flags_an_unterminated_decl_never_silently() {
    // No terminating `;` before EOF — must be reported as a problem, not silently dropped (G2),
    // and must not crash/hang the scan.
    let src = "type Broken =\n  Foo(Bytes)\n";
    let (decls, problems) = type_declarations(src);
    assert!(decls.is_empty());
    assert_eq!(problems.len(), 1);
    assert_eq!(problems[0].0, "Broken");
    assert!(problems[0].2.contains("never reached a terminating"));
}

#[test]
fn type_declarations_depth_tracks_nested_delimiters_in_ctor_args() {
    // A constructor argument itself contains `|`-free but bracket-heavy types (`Vec[Bytes]`,
    // `Map[K, V]`) — the top-level split must not be fooled by nested `[]`/`()`/`{}`.
    let src = "type Item = Use(UsePath) | Type(TypeDecl) | Fn(FnDecl);\n";
    let (decls, _) = type_declarations(src);
    let names: Vec<&str> = decls[0].ctors.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(names, vec!["Use", "Type", "Fn"]);
}

// ── `unrecognized_top_level` (future-proofing flag, G2) ─────────────────────────────────────────

#[test]
fn unrecognized_top_level_flags_a_use_declaration() {
    let src = "nodule std.cmp;\n\nuse std.core.Bool;\n\nfn f() => Bool = True;\n";
    let flagged = unrecognized_top_level(src);
    assert_eq!(flagged, vec![("use", 3)]);
}

#[test]
fn unrecognized_top_level_ignores_indented_and_blank_lines() {
    // A `use`-shaped line that is NOT at column 0 (inside a fn body, say) is not top-level and
    // must not be flagged; nor should this extractor ever see one today (no `.myc` file indents
    // a real top-level item), but the check must not false-positive if it ever does appear
    // indented for some other (non-Item) reason.
    let src = "fn f() => Bool =\n  use_helper();\n";
    assert!(unrecognized_top_level(src).is_empty());
}

// ── `nodule_decl_line` ───────────────────────────────────────────────────────────────────────────

#[test]
fn nodule_decl_line_finds_the_statement_not_the_header_comment() {
    let src = "// nodule: std.cmp\n// @version: 0.1.0\nnodule std.cmp;\n";
    assert_eq!(nodule_decl_line(src), 3);
}

#[test]
fn nodule_decl_line_falls_back_to_the_marker_comment() {
    let src = "// nodule: std.cmp\nfn f() => Bool = True;\n";
    assert_eq!(nodule_decl_line(src), 1);
}

// ── `index_file` (per-file orchestration: nodule + fn + type + ctor + flags) ────────────────────

#[test]
fn index_file_extracts_nodule_fn_type_and_ctor_rows() {
    let src = "// nodule: std.cmp\n\
               // @summary: Ordering surface.\n\
               nodule std.cmp;\n\
               \n\
               type Ordering = Lt | Eq | Gt;\n\
               \n\
               // is_lt: project an Ordering to Bool.\n\
               fn is_lt(o: Ordering) => Bool =\n\
               \x20 match o { Lt => True, Eq => False, Gt => False };\n";
    let mut items = Vec::new();
    let mut flagged = Vec::new();
    index_file("std", "lib/std/cmp.myc", src, &mut items, &mut flagged);

    assert!(flagged.is_empty(), "a clean file should flag nothing");
    // `index_file`'s own emission order is nodule → every `fn` → every `type` (+ its ctors), an
    // implementation detail re-sorted away by `build_lib_index`'s (phylum, nodule, line, symbol)
    // pass — so compare the *set* of (symbol, kind) pairs, not this internal ordering.
    let mut kinds: Vec<(&str, &str)> = items
        .iter()
        .map(|i| (i.symbol.as_str(), i.kind.as_str()))
        .collect();
    kinds.sort_unstable();
    let mut expected = vec![
        ("std.cmp", "nodule"),
        ("std.cmp::Ordering", "type"),
        ("std.cmp::Ordering::Lt", "ctor"),
        ("std.cmp::Ordering::Eq", "ctor"),
        ("std.cmp::Ordering::Gt", "ctor"),
        ("std.cmp::is_lt", "fn"),
    ];
    expected.sort_unstable();
    assert_eq!(kinds, expected);
    let nodule_item = &items[0];
    assert_eq!(nodule_item.summary.as_deref(), Some("Ordering surface."));
    assert_eq!(nodule_item.tag, ITEM_TAG);
    let fn_item = items.iter().find(|i| i.kind == "fn").unwrap();
    assert_eq!(
        fn_item.signature.as_deref(),
        Some("fn is_lt(o: Ordering) => Bool")
    );
    assert_eq!(
        fn_item.summary.as_deref(),
        Some("is_lt: project an Ordering to Bool.")
    );
}

#[test]
fn index_file_flags_a_missing_nodule_marker_never_silent() {
    let src = "type X = A | B;\n";
    let mut items = Vec::new();
    let mut flagged = Vec::new();
    index_file("std", "lib/std/x.myc", src, &mut items, &mut flagged);
    assert!(
        flagged
            .iter()
            .any(|f| f.reason.contains("no `// nodule:` marker")),
        "a file with no nodule marker must be flagged, not silently grouped as if fine"
    );
    // Still indexed under a filename-derived nodule — never dropped entirely (G2).
    assert!(items.iter().any(|i| i.nodule == "x"));
}

#[test]
fn index_file_flags_a_malformed_header_but_still_extracts_fn_type() {
    // An unknown `@key` is a checked header error (`mycelium_proj::parse_header`); extraction of
    // fn/type must still proceed (a bad header is not a license to drop everything else — G2).
    let src = "// nodule: std.cmp\n\
               // @bogus: nope\n\
               nodule std.cmp;\n\
               \n\
               fn f() => Bool = True;\n";
    let mut items = Vec::new();
    let mut flagged = Vec::new();
    index_file("std", "lib/std/cmp.myc", src, &mut items, &mut flagged);
    assert!(
        flagged
            .iter()
            .any(|f| f.reason.contains("header metadata parse error")),
        "a malformed header must be flagged, not silently ignored"
    );
    assert!(items.iter().any(|i| i.symbol == "std.cmp::f"));
}

#[test]
fn index_file_flags_an_unrecognized_top_level_construct() {
    let src = "nodule std.cmp;\n\nuse std.core.Bool;\n";
    let mut items = Vec::new();
    let mut flagged = Vec::new();
    index_file("std", "lib/std/cmp.myc", src, &mut items, &mut flagged);
    assert!(flagged
        .iter()
        .any(|f| f.reason.contains("unextracted top-level construct")));
}

// ── the full `build_lib_index` walk (hermetic temp-dir fixture) ─────────────────────────────────

fn small_lib() -> PathBuf {
    let root = temp_dir("build");
    write(
        &root,
        "lib/std/cmp.myc",
        "// nodule: std.cmp\n\
         // @summary: Ordering surface.\n\
         nodule std.cmp;\n\
         \n\
         type Ordering = Lt | Eq | Gt;\n\
         \n\
         fn is_lt(o: Ordering) => Bool =\n\
         \x20 match o { Lt => True, Eq => False, Gt => False };\n",
    );
    write(
        &root,
        "lib/compiler/token.myc",
        "// nodule: compiler.token\nnodule compiler.token;\n\ntype Tok = TInt | TEof;\n",
    );
    root
}

#[test]
fn build_lib_index_walks_every_phylum_and_sorts_deterministically() {
    let root = small_lib();
    let report = build_lib_index(&root).unwrap();
    assert!(report.flagged.is_empty());

    let phyla: Vec<&str> = {
        let mut v: Vec<&str> = report.items.iter().map(|i| i.phylum.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    assert_eq!(phyla, vec!["compiler", "std"]);

    // Stable (phylum, nodule, line, symbol) order — never re-derived per run in a different order.
    let mut sorted = report.items.clone();
    sorted.sort_by(|a, b| {
        (&a.phylum, &a.nodule, a.line, &a.symbol).cmp(&(&b.phylum, &b.nodule, b.line, &b.symbol))
    });
    assert_eq!(report.items, sorted);
}

#[test]
fn build_lib_index_is_byte_identical_across_two_runs() {
    let root = small_lib();
    let a = build_lib_index(&root).unwrap();
    let b = build_lib_index(&root).unwrap();
    assert_eq!(a.items, b.items);
    assert_eq!(a.flagged, b.flagged);

    let out_a = temp_dir("out-a");
    let out_b = temp_dir("out-b");
    write_json(&a, &out_a).unwrap();
    write_json(&b, &out_b).unwrap();
    write_markdown(&a, &out_a).unwrap();
    write_markdown(&b, &out_b).unwrap();
    assert_eq!(
        fs::read(out_a.join("index.json")).unwrap(),
        fs::read(out_b.join("index.json")).unwrap()
    );
    assert_eq!(
        fs::read(out_a.join("INDEX.md")).unwrap(),
        fs::read(out_b.join("INDEX.md")).unwrap()
    );
}

#[test]
fn write_markdown_never_silently_drops_the_flagged_section() {
    let root = temp_dir("flagged");
    write(&root, "lib/std/bad.myc", "type X = A | B;\n"); // no nodule marker → flagged
    let report = build_lib_index(&root).unwrap();
    assert!(!report.flagged.is_empty());
    let out = temp_dir("flagged-out");
    write_markdown(&report, &out).unwrap();
    let md = fs::read_to_string(out.join("INDEX.md")).unwrap();
    assert!(md.contains("## Flagged items"));
    assert!(!md.contains("*(none)*"));
}
