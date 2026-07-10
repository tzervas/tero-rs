//! The construct-mapping table: Rust `syn` types/paths -> Mycelium `type_ref` surface text, or
//! an explicit reason a mapping is not confirmed (never a guess — VR-5, G2).
//!
//! **Guarantee: `Declared`.** Every row here is a heuristic syn -> surface-text mapping verified
//! only against `docs/spec/grammar/mycelium.ebnf` (the grammar text), not against a Mycelium
//! parser or typechecker. Human-auditable: each row below carries a comment citing the grammar
//! fact it relies on.

use crate::gap::{guarded, Category, GapReason};
use quote::ToTokens;
use syn::{PathArguments, Type};

/// Render a `syn` node's tokens back to text, for gap snippets and unmapped-type messages only
/// (never used to build emitted `.myc` output — that always goes through the explicit mapping
/// functions in this module / `emit.rs`).
pub fn tokens_to_string<T: ToTokens>(node: &T) -> String {
    node.to_token_stream().to_string()
}

/// Map a Rust type to its Mycelium `type_ref` text.
///
/// `self_ty` supplies the substitution for `Self` inside an impl/trait body — `None` when there
/// is no enclosing impl/trait (a bare `Self` then has no referent and is a gap).
///
/// Returns `Err(GapReason)` when the type has no confirmed grammar surface. Confirmed rows (see
/// `docs/spec/grammar/mycelium.ebnf` §`base_type`):
/// - `bool` -> the ordinary named type `Bool` (used bare in `lib/std/cmp.myc`; base_type's
///   `Ident type_args?` arm covers an ordinary named type, so this assumes a kernel/prelude
///   `Bool` exists — Declared, not verified against a symbol table).
/// - unsigned integers (`u8`/`u16`/.../`u128`) -> `Binary{N}` (`base_type ::= 'Binary' '{' Int
///   '}'`). `lib/std/cmp.myc`'s own comments describe `Binary{N}` as **unsigned magnitude** —
///   so *signed* integers (`i8`.../`isize`) are intentionally NOT mapped here (would misrepresent
///   twos-complement semantics as an unsigned-magnitude representation); they are a gap.
/// - `isize`/`usize` -> gap (platform-dependent width has no fixed `Binary{N}`).
/// - `f64` -> `Float` (`base_type ::= 'Float'`, `docs/spec/grammar/mycelium.ebnf:251` — a nullary
///   scalar-float type, "IEEE-754 binary64 only at introduction", ADR-040 FLAG-1/M-897; trx2 Lane C
///   Deliverable 2 verify-first correction — `myc check`-confirmed). `f32`/`char` -> gap (`f32` has
///   no confirmed representation, `Float` being binary64-only; `char` has no confirmed base_type
///   arm in this grammar fragment). NOTE: `scalar` (`F16`/`BF16`/`F32`/`F64`) is a *different*,
///   Dense-only production (`Dense{N, scalar}`/`ambient_params`) — unrelated to the bare `Float`
///   value type.
/// - `String`/`str`/`&str` -> `Bytes` (RFC-0033 §3.2: the dedicated, never-silent UTF-8 text repr;
///   grammar `base_type` line 250; a `"…"` StrLit lowers to the same `Repr::Bytes` value form —
///   checkty.rs:6669). Verified `myc check`-clean (DN-34 §8.14). `&str` is erased to `str` by the
///   shared-reference arm below, then mapped.
/// - `()` (unit) -> gap (no unit-value literal in the grammar's `literal`/`primary` productions).
/// - an ordinary zero-argument named type (`Ordering`, a same-crate type, etc.) -> passed through
///   as-is via `base_type`'s `Ident type_args?` arm.
/// - a tuple type of arity >= 2, all of whose elements map -> the grammar's tuple `type_ref` arm
///   (`'(' type_ref ',' type_ref (',' type_ref)* ')'`, M-826).
/// - a **shared** reference `&T` / `&'a T` -> the referent's mapping (the reference is *erased*).
///   Mycelium is value-semantic (ADR-003: no reference types; the grammar's `base_type`/`type_ref`
///   has no `&` form), so a shared borrow denotes the same `T` as the value — the type-position twin
///   of the reference-transparent erasure `emit.rs` already does on `&expr`/`&pat`, and how the
///   hand-port writes Rust `&Ordering` params as value `Ordering` (`lib/std/cmp.myc`). A referent
///   that itself has no mapping still gaps (its own precise reason surfaces — never a partial
///   emission). A **mutable** reference `&mut T` is NOT erased -> gap (in-place mutation has no
///   value-semantic correspondence — same stance as the `&mut self` receiver gap in
///   `emit::map_signature`).
/// - a single-segment named *generic application* (`Result<Duration, TimeErr>`, `Vec<u8>`,
///   `Option<T>`), all of whose angle-bracketed arguments are themselves mappable *types* ->
///   `Head[arg, …]` via `base_type ::= Ident type_args?` + `type_args ::= '[' type_ref (','
///   type_ref)* ']'` (grammar lines 258 + 265; RFC-0037 D1 uses `[]`, not `<>`). Refused as a gap
///   (never a partial emission) if the head is a reserved word, if any argument is a lifetime /
///   const-generic / associated-type binding-or-constraint, or if any argument type itself gaps.
/// - a *qualified* multi-segment path (`std::cmp::Ordering`, `crate::foo::Bar`) -> gap. Mycelium
///   `path`s are dot-joined and this module has no cross-nodule symbol table, so collapsing to
///   the last segment (as it did in an earlier iteration of this function) risked silently
///   conflating a foreign type with an unrelated local type of the same terminal name — a real
///   bug caught by inspecting this transpiler's own output on `std::cmp::Ordering` vs the local
///   `Ordering` (see the transpiler's report). Left an explicit gap rather than guessed (VR-5).
///
/// **RFC-0041 §4.7 (W1):** guarded by the crate-wide recursion budget (`crate::gap::guarded`) —
/// self-recurses over unbounded/attacker-controlled type nesting (a right-nested `Type::Tuple`),
/// so each call consumes one budget frame and refuses with a `Category::RecursionBudget` gap
/// rather than risking a host-stack overflow.
pub fn map_type(ty: &Type, self_ty: Option<&str>) -> Result<String, GapReason> {
    guarded(|| map_type_inner(ty, self_ty))
}

/// The recursion-guarded body of [`map_type`]. Recursive calls use the public `map_type` name so
/// each nested call re-enters the guard.
fn map_type_inner(ty: &Type, self_ty: Option<&str>) -> Result<String, GapReason> {
    match ty {
        Type::Path(tp) if tp.qself.is_none() && tp.path.segments.len() > 1 => Err(GapReason::new(
            Category::Other,
            format!(
                "qualified type path `{}` — collapsing to its last segment would risk colliding \
                 with an unrelated same-named local type (e.g. `std::cmp::Ordering` vs a local \
                 `Ordering`); left an explicit gap rather than guessed (VR-5)",
                tokens_to_string(tp)
            ),
        )),
        Type::Path(tp) if tp.qself.is_none() => {
            let seg =
                tp.path.segments.last().ok_or_else(|| {
                    GapReason::new(Category::Other, "empty type path".to_string())
                })?;
            let name = seg.ident.to_string();
            match name.as_str() {
                "Self" => self_ty.map(str::to_string).ok_or_else(|| {
                    GapReason::new(
                        Category::Other,
                        "`Self` type with no enclosing impl/trait context",
                    )
                }),
                "bool" => Ok("Bool".to_string()),
                "u8" => Ok("Binary{8}".to_string()),
                "u16" => Ok("Binary{16}".to_string()),
                "u32" => Ok("Binary{32}".to_string()),
                "u64" => Ok("Binary{64}".to_string()),
                "u128" => Ok("Binary{128}".to_string()),
                "i8" | "i16" | "i32" | "i64" | "i128" => Err(GapReason::new(
                    Category::Other,
                    format!(
                        "signed integer `{name}` — Binary{{N}} is documented unsigned-magnitude \
                         (lib/std/cmp.myc); mapping a signed type onto it would misrepresent \
                         twos-complement semantics, so this is left an explicit gap rather than \
                         guessed (VR-5)"
                    ),
                )),
                "isize" | "usize" => Err(GapReason::new(
                    Category::Other,
                    format!(
                        "`{name}` has a platform-dependent width; no fixed Binary{{N}} mapping"
                    ),
                )),
                // trx2 Lane C Deliverable 2 (verify-first correction, mitigation #14): the prior
                // "no confirmed base_type arm" reason for `f32`/`f64` was STALE — the grammar DOES
                // have a nullary `Float` base_type (`docs/spec/grammar/mycelium.ebnf:251`: "first-
                // class scalar float, IEEE-754 binary64 only at introduction (ADR-040 FLAG-1;
                // M-897) — nullary like Bytes"). `scalar` (`F16`/`BF16`/`F32`/`F64`) is a DIFFERENT,
                // Dense-only production (`Dense{N, scalar}`/`ambient_params`) — the earlier comment
                // conflated the two. Confirmed `myc check`-clean empirically: `fn f(x: Float) =>
                // Float = 1.5;` and `fn f(x: Float) => Binary{1} = flt_is_nan(x);` both check with
                // no import (`target/debug/myc`, `mycelium-proj.toml` `lang = "mycelium-0"`).
                // `Float` is explicitly "binary64 only at introduction" (a width extension is a
                // future, its-own-decision append — the grammar comment's own words), so `f64` maps
                // faithfully; `f32` still has no confirmed representation and stays a gap (never
                // silently widened/narrowed to `Float`, VR-5).
                "f64" => Ok("Float".to_string()),
                "f32" => Err(GapReason::new(
                    Category::Other,
                    "`f32` has no confirmed Mycelium representation — `Float` \
                     (docs/spec/grammar/mycelium.ebnf:251) is IEEE-754 binary64 only at \
                     introduction (ADR-040 FLAG-1/M-897); a width extension is a future, \
                     separately-decided append, never silently assumed (VR-5)",
                )),
                "char" => Err(GapReason::new(
                    Category::Other,
                    "`char` has no confirmed base_type arm in this grammar fragment",
                )),
                // RFC-0033 §3.2 (grounded via tero, DN-34 §8.14): `Bytes` is the language's
                // *dedicated, never-silent UTF-8* text repr (grammar `base_type` line 250,
                // "first-class byte string"; a `"…"` StrLit lowers to the same `Repr::Bytes` value
                // form — checkty.rs:6669, M-910/M-911). So Rust `String`/`str` map onto `Bytes`
                // faithfully: both denote an owned UTF-8 text value under value semantics (ADR-003),
                // and the earlier "not confirmed equivalent" hedge is resolved by §3.2. Verified
                // `myc check`-clean (a `Bytes`-typed field/param/return and a `"…"` literal all pass
                // — DN-34 §8.14 verify-first). This is the type-position twin of the string-literal
                // value emission `emit.rs` already performs (`Lit::Str` -> `StrLit`). Graded
                // `Declared` like every row here (grammar-text- + oracle-verified, not proven).
                "String" | "str" => Ok("Bytes".to_string()),
                _ if matches!(seg.arguments, PathArguments::None) => {
                    // M-1001: an ordinary named type passed through as-is — but if its name is a
                    // Mycelium reserved word (e.g. a Rust type literally named `Binary`/`Float`), the
                    // bare identifier would lex as a keyword and fail to parse. Gap it (never emit
                    // un-parseable text) rather than guess a rename (VR-5/G2).
                    crate::reserved::guard_ident(&name, "type name")?;
                    Ok(name)
                }
                // A single-segment named *generic application* (`Result<Duration, TimeErr>`,
                // `Vec<u8>`, `Option<T>`). Confirmed surface: `base_type ::= Ident type_args?` with
                // `type_args ::= '[' type_ref (',' type_ref)* ']'`
                // (docs/spec/grammar/mycelium.ebnf lines 258 + 265 — RFC-0037 D1: type arguments in
                // square brackets, not `<…>`). Every scalar/gapped builtin (`bool`/`u8`.../`String`/
                // …) already matched an earlier arm, so a generic application is *never* mapped onto
                // a `Bool`/`Binary{N}`/`String` head here — only ordinary named heads reach this arm
                // (they fall through the builtin name matches, exactly as the bare-named arm above).
                // Graded `Declared` like every row in this module (grammar-text-verified only).
                _ => match &seg.arguments {
                    PathArguments::AngleBracketed(ab) => {
                        // Head maps exactly as the bare-named arm does — a reserved-word head still
                        // gaps (never emit un-lexable text; VR-5/G2), before any argument work.
                        crate::reserved::guard_ident(&name, "type name")?;
                        let mut args = Vec::with_capacity(ab.args.len());
                        for arg in &ab.args {
                            match arg {
                                // Recurse through the *public* `map_type` (not `_inner`) so the
                                // recursion budget re-arms per nested application — same pattern as
                                // the tuple arm below — and, as there, a type argument that itself
                                // gaps propagates its own precise `GapReason` unchanged (`?`), never
                                // a partial emission.
                                syn::GenericArgument::Type(t) => args.push(map_type(t, self_ty)?),
                                // A lifetime / const-generic / associated-type binding-or-constraint
                                // (or any future non-`Type` `GenericArgument`) has no `type_ref`-
                                // shaped `type_args` surface (line 265 admits only `type_ref`s), so
                                // refuse the whole application rather than drop the argument (G2).
                                other => {
                                    return Err(GapReason::new(
                                        Category::GenericBound,
                                        format!(
                                            "generic type path `{}` — type argument `{}` is not a \
                                             type (lifetime / const-generic / associated-type \
                                             binding-or-constraint); `type_args` admits only \
                                             type_refs, so left an explicit gap (VR-5)",
                                            tokens_to_string(tp),
                                            tokens_to_string(other)
                                        ),
                                    ));
                                }
                            }
                        }
                        // `type_args ::= '[' type_ref (',' type_ref)* ']'` requires >= 1 type_ref;
                        // an empty `<>` has no confirmed surface.
                        if args.is_empty() {
                            return Err(GapReason::new(
                                Category::GenericBound,
                                format!(
                                    "generic type path `{}` — empty type-argument list has no \
                                     confirmed `type_args` surface (requires >= 1 type_ref)",
                                    tokens_to_string(tp)
                                ),
                            ));
                        }
                        Ok(format!("{name}[{}]", args.join(", ")))
                    }
                    // Non-angle-bracketed arguments (e.g. an `Fn(..)`-trait parenthesized form) —
                    // no confirmed grammar surface; left an explicit gap.
                    _ => Err(GapReason::new(
                        Category::GenericBound,
                        format!(
                            "generic type path `{}` — type-argument mapping not confirmed",
                            tokens_to_string(tp)
                        ),
                    )),
                },
            }
        }
        Type::Tuple(t) if t.elems.is_empty() => Err(GapReason::new(
            Category::Other,
            "unit type `()` has no representable value in this grammar fragment",
        )),
        Type::Tuple(t) if t.elems.len() >= 2 => {
            let mut parts = Vec::with_capacity(t.elems.len());
            for elem in &t.elems {
                parts.push(map_type(elem, self_ty)?);
            }
            Ok(format!("({})", parts.join(", ")))
        }
        // A **shared** reference type `&T` / `&'a T` has no Mycelium reference-type surface — the
        // grammar's `type_ref`/`base_type` (docs/spec/grammar/mycelium.ebnf §`base_type`) admits no
        // `&` form, and Mycelium is value-semantic (ADR-003: there are no reference types). Under
        // value semantics a shared borrow and the value it borrows denote the *same* `T`, so the
        // reference is **erased** and its referent type mapped. This is the type-position analogue of
        // the reference-transparent erasure `emit.rs` already performs on `&expr` (`Expr::Reference`)
        // and `&pat` (`Pat::Reference`), and it is exactly how the hand-port renders Rust `&Ordering`
        // params as value `Ordering` (`lib/std/cmp.myc`'s `fn cmp(a: Ordering, b: Ordering)` for the
        // Rust `fn cmp(&self, other: &Ordering)`). The lifetime, if any, is erased with the reference
        // (lifetimes have no grammar surface). Recurse through the *public* `map_type` so the
        // recursion budget re-arms per level (same pattern as the tuple arm) — and a referent type
        // that itself has no confirmed mapping propagates its own precise `GapReason` unchanged (`?`),
        // never a partial emission (so `&str`/`&[u8]`/`&dyn T` surface their *referent's* real
        // blocker, not the reference; VR-5/G2).
        Type::Reference(r) if r.mutability.is_none() => map_type(&r.elem, self_ty),
        // A **mutable** reference `&mut T` is NOT erased. In-place mutation through a `&mut` has no
        // value-semantic correspondence (ADR-003) — the same stance the `&mut self` receiver already
        // takes in `emit::map_signature` — so erasing it to a plain value type would silently drop
        // the mutation. Left an explicit gap rather than a misrepresentation (VR-5/G2).
        Type::Reference(_) => Err(GapReason::new(
            Category::Other,
            format!(
                "`{}` is a mutable reference `&mut T` — in-place mutation through a borrow has no \
                 value-semantic correspondence (ADR-003; cf. the `&mut self` receiver gap), so it \
                 is left an explicit gap rather than silently erased to a value type (VR-5)",
                tokens_to_string(ty)
            ),
        )),
        _ => Err(GapReason::new(
            Category::Other,
            format!("unsupported Rust type form `{}`", tokens_to_string(ty)),
        )),
    }
}

/// For the M-1006 **resolvability fixpoint** (`transpile::resolvable_type_names`): collect the bare,
/// single-segment **user** type names `ty` references (the ones [`map_type`] passes through *as-is* —
/// i.e. not builtins), pushing them into `out`. Returns `false` when `ty` has **no** [`map_type`]
/// mapping at all (an unmappable field ⇒ its record can never be resolvable — consistent with
/// `map_type` gapping the field). Builtins (`bool`, `u8..u128`) and tuples/shared-refs/generic-apps
/// of mappables are traversed for their nested user names but are not themselves deps.
///
/// This deliberately **mirrors [`map_type`]'s mappable shapes**; if the two drift, the only cost is a
/// *missed* emission (a struct conservatively left gapped) — never an unsound one (VR-5): the gate is
/// one-sided (it can only *withhold* an emission, so a stale mirror is safe, just less generous).
pub(crate) fn field_type_user_deps(ty: &Type, out: &mut Vec<String>) -> bool {
    match ty {
        Type::Path(tp) if tp.qself.is_none() && tp.path.segments.len() == 1 => {
            let seg = match tp.path.segments.last() {
                Some(s) => s,
                None => return false,
            };
            let name = seg.ident.to_string();
            match name.as_str() {
                // Builtins `map_type` maps directly — mappable, but contribute no user dep.
                // `String`/`str` now map to `Bytes` (RFC-0033 §3.2 — DN-34 §8.14), so they join the
                // builtins here: a `String`-typed field no longer withholds its struct's emission.
                // `f64` now maps to `Float` (trx2 Lane C Deliverable 2 — see `map_type`'s doc); it
                // joins the builtins here too, for the identical reason.
                "bool" | "u8" | "u16" | "u32" | "u64" | "u128" | "String" | "str" | "f64" => {
                    matches!(seg.arguments, PathArguments::None)
                }
                // Shapes `map_type` gaps outright ⇒ unmappable field.
                "Self" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "usize" | "f32"
                | "char" => false,
                _ => {
                    // A reserved-word type name fails to lex ⇒ `map_type` gaps it (unmappable).
                    if crate::reserved::is_reserved(&name) {
                        return false;
                    }
                    match &seg.arguments {
                        PathArguments::None => {
                            out.push(name);
                            true
                        }
                        PathArguments::AngleBracketed(ab) => {
                            out.push(name);
                            !ab.args.is_empty()
                                && ab.args.iter().all(|a| match a {
                                    syn::GenericArgument::Type(t) => field_type_user_deps(t, out),
                                    _ => false,
                                })
                        }
                        _ => false,
                    }
                }
            }
        }
        // Qualified multi-segment path: `map_type` gaps it (unmappable).
        Type::Path(_) => false,
        Type::Tuple(t) if t.elems.is_empty() => false,
        Type::Tuple(t) if t.elems.len() >= 2 => {
            t.elems.iter().all(|e| field_type_user_deps(e, out))
        }
        Type::Reference(r) if r.mutability.is_none() => field_type_user_deps(&r.elem, out),
        _ => false,
    }
}
