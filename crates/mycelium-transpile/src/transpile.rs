//! The driver (M-873): parse one Rust file with `syn`, walk every top-level item exhaustively,
//! and either emit `.myc` text or record a [`Gap`] — **never both-absent** (G2).
//!
//! **Invariant** (checked by `src/tests/invariant.rs`): for every top-level item in
//! `syn::File::items`, its name/index appears in `GapReport::emitted_items` OR at least one
//! [`Gap`] in `GapReport::gaps` — never neither.

use crate::emit::{self, Emitted};
use crate::gap::{Category, Gap, GapReason, GapReport};
use crate::map::tokens_to_string;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use syn::spanned::Spanned;
use syn::Item;

/// Parse `path` and transpile every top-level item. Returns the best-effort `.myc` text plus the
/// structured gap report. I/O and parse failures are returned as `Err` (this is a hard failure
/// distinct from a per-item gap — the file could not be read/parsed at all).
pub fn transpile_file(path: &Path) -> Result<(String, GapReport), String> {
    let source_text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    transpile_source(
        &source_text,
        &path.display().to_string(),
        &derive_nodule_path(path),
    )
}

/// Transpile already-read source text. Split out from [`transpile_file`] so tests can exercise
/// the driver on small inline fixtures without touching the filesystem.
pub fn transpile_source(
    source_text: &str,
    file_label: &str,
    nodule_path: &str,
) -> Result<(String, GapReport), String> {
    let parsed =
        syn::parse_file(source_text).map_err(|e| format!("failed to parse {file_label}: {e}"))?;

    let mut emitted_items = Vec::new();
    let mut gaps = Vec::new();

    if !parsed.attrs.is_empty() {
        // Inner (`#![...]`) attributes live outside `syn::File::items`, so they are outside the
        // per-item invariant's scope by construction — but still recorded, never silently
        // dropped. Doc attrs (`//!`) are folded into the nodule header instead of gapped.
        let non_doc: Vec<String> = parsed
            .attrs
            .iter()
            .filter(|a| !a.path().is_ident("doc"))
            .map(tokens_to_string)
            .collect();
        if !non_doc.is_empty() {
            gaps.push(Gap {
                file: file_label.to_string(),
                line: 1,
                col: 1,
                category: Category::Other,
                rust_construct: Category::Other.as_str().to_string(),
                snippet: non_doc.join(" "),
                reason: "crate/file-level inner attributes (#![...]) are not transpiled (no \
                          nodule-header equivalent for these Rust-specific directives)"
                    .to_string(),
                item_name: None,
            });
        }
    }

    let total = parsed.items.len();
    let mut body_chunks = Vec::new();

    // M-1006 (E33-1): the per-file resolvability set gating named-field-record emission (see
    // `emit::with_resolvable`). Computed once over this file's declarations, then installed for the
    // item loop so a named-field `struct`/enum variant emits only when it introduces no unresolved
    // in-file reference (which would poison the file's `myc check` and cost its clean items).
    let resolvable = resolvable_type_names(&parsed.items);
    let layouts = struct_layouts(&parsed.items);
    crate::emit::with_emit_ctx(resolvable, layouts, || {
        for item in &parsed.items {
            let (line, col) = span_line_col(item);
            let name_hint = item_display_name(item);
            let snippet = tokens_to_string(item);

            match dispatch_item(item) {
                Outcome::Emitted(Emitted {
                    name,
                    myc,
                    sub_gaps,
                }) => {
                    body_chunks.push(myc);
                    emitted_items.push(name.clone());
                    for sg in sub_gaps {
                        gaps.push(Gap {
                            file: file_label.to_string(),
                            line,
                            col,
                            category: sg.category,
                            // `rust_construct` mirrors `category` (the finer, per-failure-reason
                            // taxonomy from `gap.rs`), not the coarse `syn::Item` kind an earlier
                            // iteration used (`Impl`/`Fn`/`Struct`/...) — that coarser string
                            // collapsed e.g. every failing `impl` method to the same "Impl" label
                            // regardless of *why* it failed, hiding exactly the distinction the gap
                            // report exists to surface (G2: the report is the ground truth the
                            // surface-feature backlog is synthesized from, so its categories must be
                            // the real ones, not a re-derived approximation of them).
                            rust_construct: sg.category.as_str().to_string(),
                            snippet: snippet.clone(),
                            reason: sg.reason,
                            item_name: Some(name.clone()),
                        });
                    }
                }
                Outcome::Gap(reason) => {
                    gaps.push(Gap {
                        file: file_label.to_string(),
                        line,
                        col,
                        category: reason.category,
                        rust_construct: reason.category.as_str().to_string(),
                        snippet,
                        reason: reason.reason,
                        item_name: name_hint,
                    });
                }
                Outcome::TestExcluded => {
                    gaps.push(Gap {
                        file: file_label.to_string(),
                        line,
                        col,
                        category: Category::TestItem,
                        rust_construct: Category::TestItem.as_str().to_string(),
                        snippet,
                        reason: "#[cfg(test)] item — out of scope for this PoC's transpilation \
                              surface (excluded from the expressible-fraction denominator, \
                              but recorded, never silently skipped)"
                            .to_string(),
                        item_name: name_hint,
                    });
                }
            }
        }
    });

    let myc_text = render_nodule(nodule_path, &body_chunks, &parsed.attrs);
    let report = GapReport {
        source: file_label.to_string(),
        emitted_items,
        gaps,
        total_top_level_items: total,
    };
    Ok((myc_text, report))
}

/// Compute the set of type names that are **resolvable in this file** — the M-1006 (E33-1) gate for
/// named-field-record emission (consumed via [`crate::emit::with_resolvable`]). A declared
/// `struct`/`enum` is resolvable iff every field type (across all variants, for an enum) *maps* AND
/// every **user** type it references is itself a resolvable in-file type. A reference to a type not
/// declared in this file (e.g. a sibling-crate/kernel type such as `ContentHash`) is never
/// resolvable, so a record depending on it stays gapped rather than emitting a reference that would
/// poison the file's `myc check` (VR-5/G2). Builtins are handled by `map_type` and are not deps.
///
/// This is a **greatest** fixed point (start with every mappable declared type, then iteratively
/// drop any whose deps aren't all resolvable) so **recursive and mutually-recursive** types — a
/// self-referential `type Nat = Z | S(Nat)`, an `FsNode`/`ScopeTree` cycle — are correctly kept
/// resolvable (a least fixed point would wrongly exclude every cycle).
fn resolvable_type_names(items: &[Item]) -> HashSet<String> {
    // Each declared type -> its user-type deps, or `None` if any field is unmappable (that type can
    // then never be resolvable — consistent with `map_type` gapping the field).
    fn collect_field_deps(fields: &syn::Fields, acc: &mut Option<Vec<String>>) {
        let field_iter = match fields {
            syn::Fields::Unit => return,
            syn::Fields::Named(fs) => fs.named.iter(),
            syn::Fields::Unnamed(fs) => fs.unnamed.iter(),
        };
        for f in field_iter {
            match acc.as_mut() {
                None => return,
                Some(v) => {
                    if !crate::map::field_type_user_deps(&f.ty, v) {
                        *acc = None;
                        return;
                    }
                }
            }
        }
    }
    let mut deps: Vec<(String, Option<Vec<String>>)> = Vec::new();
    for item in items {
        match item {
            Item::Struct(s) => {
                let mut acc = Some(Vec::new());
                collect_field_deps(&s.fields, &mut acc);
                deps.push((s.ident.to_string(), acc));
            }
            Item::Enum(e) => {
                let mut acc = Some(Vec::new());
                for v in &e.variants {
                    collect_field_deps(&v.fields, &mut acc);
                    if acc.is_none() {
                        break;
                    }
                }
                deps.push((e.ident.to_string(), acc));
            }
            _ => {}
        }
    }
    // Greatest fixed point: seed with every mappable declared type, then drop any whose deps are not
    // all still in the set (an external/unmapped dep, or one already dropped — cascading out).
    let mut resolvable: HashSet<String> = deps
        .iter()
        .filter(|(_, d)| d.is_some())
        .map(|(n, _)| n.clone())
        .collect();
    loop {
        let mut changed = false;
        let mut to_drop: Vec<String> = Vec::new();
        for (name, d) in &deps {
            if !resolvable.contains(name) {
                continue;
            }
            // `d` is `Some` for every name still in `resolvable` (seeded from `is_some`).
            if let Some(ds) = d {
                if ds.iter().any(|dep| !resolvable.contains(dep)) {
                    to_drop.push(name.clone());
                }
            }
        }
        for name in to_drop {
            resolvable.remove(&name);
            changed = true;
        }
        if !changed {
            break;
        }
    }
    resolvable
}

/// Positional field layouts of every in-file `struct` — the M-1006 field-projection input (Lever 1),
/// consumed via [`crate::emit::with_emit_ctx`]. Each struct maps to its field slots in declaration
/// order (`Some(name)` named, `None` unnamed); the emitted constructor name is the struct's own type
/// name (see `emit::emit_struct`). Only `struct`s are recorded — a `self.<field>` projection or a
/// struct literal is meaningful only on a single-constructor product, not an enum.
fn struct_layouts(items: &[Item]) -> HashMap<String, Vec<Option<String>>> {
    let mut out = HashMap::new();
    for item in items {
        if let Item::Struct(s) = item {
            let fields: Vec<Option<String>> = match &s.fields {
                syn::Fields::Named(fs) => fs
                    .named
                    .iter()
                    .map(|f| f.ident.as_ref().map(ToString::to_string))
                    .collect(),
                syn::Fields::Unnamed(fs) => fs.unnamed.iter().map(|_| None).collect(),
                syn::Fields::Unit => Vec::new(),
            };
            out.insert(s.ident.to_string(), fields);
        }
    }
    out
}

enum Outcome {
    Emitted(Emitted),
    Gap(GapReason),
    TestExcluded,
}

/// Exhaustive dispatch over `syn::Item` (itself `#[non_exhaustive]`). Every arm either calls into
/// `emit.rs` or produces an explicit [`GapReason`] — the trailing `_` arm is the
/// forward-compatibility catch-all, itself a gap, never a silent no-op.
fn dispatch_item(item: &Item) -> Outcome {
    match item {
        Item::Enum(e) => emit::emit_enum(e).map_or_else(Outcome::Gap, Outcome::Emitted),
        Item::Struct(s) => emit::emit_struct(s).map_or_else(Outcome::Gap, Outcome::Emitted),
        Item::Fn(f) => emit::emit_fn(f).map_or_else(Outcome::Gap, Outcome::Emitted),
        Item::Trait(t) => emit::emit_trait(t).map_or_else(Outcome::Gap, Outcome::Emitted),
        Item::Impl(i) => emit::emit_impl(i).map_or_else(Outcome::Gap, Outcome::Emitted),
        Item::Use(u) => dispatch_use(u),
        Item::Mod(m) => {
            if emit::is_cfg_test(&m.attrs) {
                Outcome::TestExcluded
            } else {
                Outcome::Gap(GapReason::new(
                    Category::Other,
                    "`mod` declaration — Mycelium's nodule-per-file model has no nested-module \
                     construct in this grammar fragment",
                ))
            }
        }
        Item::Macro(m) => {
            if m.mac.path.is_ident("macro_rules") {
                Outcome::Gap(GapReason::new(
                    Category::MacroDef,
                    "`macro_rules!` definition — no macro system in this grammar",
                ))
            } else {
                Outcome::Gap(GapReason::new(
                    Category::MacroInvocation,
                    "item-position macro invocation — no macro system in this grammar",
                ))
            }
        }
        Item::Const(c) => Outcome::Gap(GapReason::new(
            Category::Other,
            format!(
                "top-level `const {}` — no const item production in the grammar (`item` covers \
                 use/default/type/trait/impl/fn/object/lower/derive only)",
                c.ident
            ),
        )),
        Item::Static(s) => Outcome::Gap(GapReason::new(
            Category::Other,
            format!(
                "top-level `static {}` — no static item production in the grammar",
                s.ident
            ),
        )),
        Item::Type(t) => Outcome::Gap(GapReason::new(
            Category::Other,
            format!(
                "`type {} = ...` alias — Mycelium's `type_item` always introduces a new nominal \
                 sum type via `'=' constructor ('|' constructor)*`; a bare alias to an existing \
                 type would fabricate a sum type where none exists semantically",
                t.ident
            ),
        )),
        Item::Union(u) => Outcome::Gap(GapReason::new(
            Category::Struct,
            format!("`union {}` — no union construct in the grammar", u.ident),
        )),
        Item::ExternCrate(e) => Outcome::Gap(GapReason::new(
            Category::Other,
            format!(
                "`extern crate {}` — no equivalent (phylum/nodule import model differs)",
                e.ident
            ),
        )),
        Item::ForeignMod(_) => Outcome::Gap(GapReason::new(
            Category::Other,
            "foreign/FFI block — Mycelium's FFI escape is `wild`, legal only inside an \
             `@std-sys` nodule with a declared `!{ffi}` effect; not auto-mapped",
        )),
        Item::TraitAlias(t) => Outcome::Gap(GapReason::new(
            Category::Trait,
            format!("`trait {} = ...` alias — no trait-alias construct", t.ident),
        )),
        Item::Verbatim(_) => Outcome::Gap(GapReason::new(
            Category::Other,
            "unparsed/verbatim item (syn could not fully parse this construct)",
        )),
        _ => Outcome::Gap(GapReason::new(
            Category::Other,
            "unrecognized syn::Item variant (Item is #[non_exhaustive] — forward-compatibility \
             catch-all)",
        )),
    }
}

/// `use` imports are **flagged, not emitted** (M-1001, `Category::Import`).
///
/// Grammar-wise `use foo.bar;` maps fine, but the transpiler has **no cross-nodule symbol table**,
/// so it cannot confirm the imported path resolves to a declared Mycelium nodule — and the M-1000
/// vet loop confirms these imports fail `myc check` name-resolution **every time** (a Rust `use
/// extern_crate::Sym` names a *crate*, not a phylum nodule; even a same-crate `use crate::foo::Bar`
/// has no sibling nodule to resolve against in the single-file emission). Emitting an import we
/// cannot confirm resolves is exactly the "plausible but wrong" emission `map_type`/`emit_expr`
/// already refuse for qualified paths/calls (DN-34 §4/§8.2) — and, worse, a single unresolved `use`
/// **poisons the whole draft's `myc check`** (it was the universal `checked_fraction`-blocker in the
/// §8.7 baseline). So it is recorded as a gap (never silently dropped, G2), never emitted. The gap's
/// `snippet` (built by the driver) carries the original `use …;` text, so the human port still sees
/// exactly what to import.
fn dispatch_use(u: &syn::ItemUse) -> Outcome {
    // Describe the import shape for a precise reason (the full text is in the driver-built snippet).
    let detail = describe_use_tree(&u.tree);
    Outcome::Gap(GapReason::new(
        Category::Import,
        format!(
            "`use` import ({detail}) — the transpiler has no cross-nodule symbol table, so it \
             cannot confirm the imported path resolves to a declared Mycelium nodule; the M-1000 \
             vet loop confirms such imports fail `myc check` name-resolution (a Rust `use \
             extern_crate::Sym` names a crate, not a nodule). Flagged, not emitted — the same \
             flag-don't-guess stance `map_type`/`emit_expr` take on qualified paths/calls (DN-34 \
             §4/§8.2; VR-5/G2)"
        ),
    ))
}

/// A short human description of a `use` tree's shape, for the gap reason.
fn describe_use_tree(tree: &syn::UseTree) -> String {
    match tree {
        syn::UseTree::Path(p) => describe_use_tree(&p.tree),
        syn::UseTree::Name(n) => format!("single path ending `{}`", n.ident),
        syn::UseTree::Glob(_) => "glob `::*`".to_string(),
        syn::UseTree::Rename(r) => format!("rename `{} as {}`", r.ident, r.rename),
        syn::UseTree::Group(_) => "grouped `{{a, b}}`".to_string(),
    }
}

fn item_display_name(item: &Item) -> Option<String> {
    match item {
        Item::Const(i) => Some(i.ident.to_string()),
        Item::Enum(i) => Some(i.ident.to_string()),
        Item::ExternCrate(i) => Some(i.ident.to_string()),
        Item::Fn(i) => Some(i.sig.ident.to_string()),
        Item::ForeignMod(_) => None,
        Item::Impl(i) => Some(tokens_to_string(&*i.self_ty)),
        Item::Macro(i) => i.ident.as_ref().map(|id| id.to_string()),
        Item::Mod(i) => Some(i.ident.to_string()),
        Item::Static(i) => Some(i.ident.to_string()),
        Item::Struct(i) => Some(i.ident.to_string()),
        Item::Trait(i) => Some(i.ident.to_string()),
        Item::TraitAlias(i) => Some(i.ident.to_string()),
        Item::Type(i) => Some(i.ident.to_string()),
        Item::Union(i) => Some(i.ident.to_string()),
        Item::Use(_) => None,
        _ => None,
    }
}

fn span_line_col(item: &Item) -> (usize, usize) {
    let start = item.span().start();
    (start.line, start.column + 1)
}

/// Best-effort nodule-path derivation (Declared heuristic): `crates/mycelium-std-cmp/src/lib.rs`
/// -> `std.cmp`, matching `lib/std/cmp.myc`'s actual header for the crate this PoC targets. Not
/// guaranteed to be meaningful for an arbitrary input path — the CLI documents this.
fn derive_nodule_path(path: &Path) -> String {
    let crate_dir = path
        .parent() // .../src
        .and_then(Path::parent) // .../mycelium-std-cmp
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str());
    match crate_dir {
        Some(dir) => {
            let stripped = dir.strip_prefix("mycelium-").unwrap_or(dir);
            stripped.replace('-', ".")
        }
        None => path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    }
}

fn render_nodule(nodule_path: &str, chunks: &[String], file_attrs: &[syn::Attribute]) -> String {
    let mut out = String::new();
    out.push_str(&format!("// nodule: {nodule_path}\n"));
    for d in emit::doc_lines(file_attrs) {
        out.push_str(&d);
        out.push('\n');
    }
    out.push_str(
        "// @summary: best-effort transpilation via mycelium-transpile (M-873). Declared,\n\
         // unvalidated — no Mycelium parser/typechecker confirms this output; see the\n\
         // accompanying .gap.json for every construct this pass could not express.\n",
    );
    out.push_str(&format!("nodule {nodule_path};\n\n"));
    out.push_str(&chunks.join("\n\n"));
    out.push('\n');
    out
}
