//! Unit tests for the `.myc` emitter, over a small fixture corpus (data-driven — per CLAUDE.md
//! "Complex test logic lives in fixtures + parameterization, not in test bodies").

use crate::gap::Category;
use crate::transpile::transpile_source;

/// The expected outcome for one fixture.
enum Expect {
    /// The item is emitted, and the `.myc` text contains this substring.
    Emitted {
        item: &'static str,
        contains: &'static str,
    },
    /// The item is not emitted at all, and at least one gap of this category is recorded.
    Gapped { category: Category },
    /// The item is emitted (containing the substring) AND at least one sub-gap of the given
    /// category is also recorded for it (e.g. a dropped `#[derive(..)]`).
    EmittedAndGapped {
        item: &'static str,
        contains: &'static str,
        sub_gap_category: Category,
    },
}

struct Case {
    name: &'static str,
    rust: &'static str,
    expect: Expect,
}

/// The fixture corpus. Each row cites the grammar production it exercises.
fn cases() -> Vec<Case> {
    vec![
        // `type_item`: C-like enum -> a sum type (grammar §type_item/constructor).
        Case {
            name: "c_like_enum",
            rust: "enum Ordering { Less, Equal, Greater }",
            expect: Expect::Emitted {
                item: "Ordering",
                contains: "type Ordering = Less | Equal | Greater;",
            },
        },
        // `fn_item`: a single-expression body (grammar §fn_item).
        Case {
            name: "simple_fn",
            rust: "fn is_lt(o: bool) -> bool { o }",
            expect: Expect::Emitted {
                item: "is_lt",
                contains: "fn is_lt(o: Bool) => Bool = o;",
            },
        },
        // `match_expr` over bool literal patterns (grammar §match_expr/pattern).
        Case {
            name: "match_expr",
            rust: "fn pick(o: bool) -> bool { match o { true => false, false => true } }",
            expect: Expect::Emitted {
                item: "pick",
                contains: "match o { True => False, False => True }",
            },
        },
        // A `let`-chain + tail expr desugars to nested `let ... in ...` (still a single
        // `fn_item` body expression).
        Case {
            name: "let_chain_body",
            rust: "fn double(x: bool) -> bool { let y = x; y }",
            expect: Expect::Emitted {
                item: "double",
                contains: "let y = x in y",
            },
        },
        // Tuple-variant enum: positional fields map via `constructor`'s optional field list.
        Case {
            name: "tuple_variant_enum",
            rust: "enum Foo { A(u8), B }",
            expect: Expect::Emitted {
                item: "Foo",
                contains: "type Foo = A(Binary{8}) | B;",
            },
        },
        // A tuple struct maps to a single-constructor `type_item`.
        Case {
            name: "tuple_struct",
            rust: "struct Bf16Bits(u16);",
            expect: Expect::Emitted {
                item: "Bf16Bits",
                contains: "type Bf16Bits = Bf16Bits(Binary{16});",
            },
        },
        // KNOWN HARD GAP: `trait` — every realistic trait in the target crate gaps (default
        // bodies, supertraits, or an unresolvable `Self`); this fixture exercises the
        // unresolvable-`self` path specifically (no default body, no supertrait).
        Case {
            name: "trait_self_unresolvable",
            rust: "trait Foo { fn bar(&self) -> bool; }",
            expect: Expect::Gapped {
                category: Category::Trait,
            },
        },
        // KNOWN HARD GAP: `macro_rules!` definitions — no macro system in the grammar.
        Case {
            name: "macro_rules_gap",
            rust: "macro_rules! foo { () => {}; }",
            expect: Expect::Gapped {
                category: Category::MacroDef,
            },
        },
        // Item-position macro invocations are a distinct category from macro *definitions*.
        Case {
            name: "macro_invocation_gap",
            rust: "some_macro!(a, b, c);",
            expect: Expect::Gapped {
                category: Category::MacroInvocation,
            },
        },
        // M-1006 (E33-1): a named-field ("record") struct whose fields all resolve in-file now emits
        // POSITIONALLY (field names dropped + recorded via a `NamedFieldDrop` sub-gap) — the
        // grammar-grounded mapping the `lib/std/*.myc` hand-ports use (`type GuaranteeRow = Row(..)`).
        Case {
            name: "struct_named_fields_emits_positionally",
            rust: "struct Foo { x: u8, y: bool }",
            expect: Expect::EmittedAndGapped {
                item: "Foo",
                contains: "type Foo = Foo(Binary{8}, Bool)",
                sub_gap_category: Category::NamedFieldDrop,
            },
        },
        // M-1006 §8.14: a named-field struct with a `String` field now EMITS — `String` maps to
        // `Bytes` (RFC-0033 §3.2), so the record is fully mappable and emits positionally.
        Case {
            name: "struct_named_field_string_maps_to_bytes",
            rust: "struct WithText { s: String, n: u32 }",
            expect: Expect::EmittedAndGapped {
                item: "WithText",
                contains: "type WithText = WithText(Bytes, Binary{32})",
                sub_gap_category: Category::NamedFieldDrop,
            },
        },
        // M-1006: a named-field struct with an UNMAPPABLE field type (`char`) still gaps — the
        // field's own precise repr reason wins (mapped *before* the resolvability gate), so the gap
        // profile keeps "unmappable field" distinct from "out-of-file reference".
        Case {
            name: "struct_named_field_unmappable_type_still_gaps",
            rust: "struct Bad { c: char }",
            expect: Expect::Gapped {
                category: Category::Struct,
            },
        },
        // M-1006 resolvability gate: a named-field struct whose fields all MAP but reference a type
        // not declared in this file (`Elsewhere`) is gated — emitting it would introduce an
        // unresolved reference that poisons the file's `myc check`. Left an honest `Struct` gap.
        Case {
            name: "struct_named_field_out_of_file_ref_is_gated",
            rust: "struct Ref { h: Elsewhere }",
            expect: Expect::Gapped {
                category: Category::Struct,
            },
        },
        // M-1006 greatest-fixpoint: mutually-recursive named-field structs (`A` <-> `B`) resolve as a
        // group and emit — a *least* fixpoint would wrongly gate both (each waits on the other). Both
        // are declared in-file and reference only each other + builtins, so the cycle is resolvable.
        Case {
            name: "mutually_recursive_named_structs_resolve",
            rust: "struct A { b: B, x: u8 }\nstruct B { a: A }",
            expect: Expect::EmittedAndGapped {
                item: "A",
                contains: "type A = A(B, Binary{8})",
                sub_gap_category: Category::NamedFieldDrop,
            },
        },
        // M-1006 Lever 1: a `self.<field>` projection in an impl body desugars to a `match` on the
        // struct's single (positional) constructor — the faithful equivalent (no projection surface in
        // the grammar). `Perm` is resolvable (its ctor emits), so the projection is gated ON.
        Case {
            name: "field_projection_desugars_to_match",
            rust: "struct Perm { mode: u8 }\nimpl Perm { fn get(self) -> u8 { self.mode } }",
            expect: Expect::Emitted {
                item: "impl Perm",
                contains: "match self { Perm(p0) => p0 }",
            },
        },
        // M-1006 Lever 1: struct-literal construction `Foo { mode: a }` -> the positional ctor call
        // `Foo(a)` (fields ordered by declaration). `Self { .. }` resolves the same way in impl context.
        Case {
            name: "struct_literal_construction_emits_positional_ctor",
            rust: "struct Foo { mode: u8 }\nfn mk(a: u8) -> Foo { Foo { mode: a } }",
            expect: Expect::Emitted {
                item: "mk",
                contains: "Foo(a)",
            },
        },
        // M-1006 Lever 1 gate: a field access on a NON-`self` base gaps — the transpiler tracks no
        // local types, so it cannot resolve the projection to a constructor position (never a guess).
        // (No struct is declared here, so the sole item is the gapping `peek`.)
        Case {
            name: "field_access_on_non_self_base_gaps",
            rust: "fn peek(f: u8) -> u8 { f.mode }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // M-873 follow-on (DN-41): a numeric-widening `impl Widen<..> for ..` whose body is a
        // qualified associated-function call (`u16::from(self)`, the real shape of Rust's
        // widening bodies in `mycelium-std-cmp`) must never be emitted with the *fabricated*
        // `from(self)` text (`from` is not a Mycelium builtin — no grammar production; only prose
        // mentions in `docs/spec/grammar/mycelium.ebnf`). Once both `Self`/target map to
        // `Binary{N}`/`Binary{M}` (unsigned widening), it is now instead emitted **faithfully**
        // via the real DN-41 `width_cast` prim — a strict improvement over the earlier "gap the
        // whole impl" behavior this case originally pinned (see
        // `widen_impls_never_fabricate_from_in_real_crate` in `src/tests/diff.rs` for the
        // real-crate-scale version of this guard).
        Case {
            name: "widen_binary_emits_width_cast",
            rust: "impl Widen<u16> for u8 { fn widen(self) -> u16 { u16::from(self) } }",
            expect: Expect::Emitted {
                item: "impl Widen[Binary{16}] for Binary{8}",
                contains: "width_cast(self, 0b0000_0000_0000_0000)",
            },
        },
        // Widen over a non-`Binary` `Self` (e.g. `bool`) has no `width_cast` witness path (`Self`
        // doesn't map to `Binary{N}` at all) — the qualified `u32::from(self)` call stays an
        // honest gap, unchanged from the pre-DN-41 behavior.
        Case {
            name: "widen_bool_from_call_still_gapped_not_fabricated",
            rust: "impl Widen<u32> for bool { fn widen(self) -> u32 { u32::from(self) } }",
            expect: Expect::Gapped {
                category: Category::Impl,
            },
        },
        // DN-41 §2: `Narrow::narrow` is fallible (`Result<To, NarrowError>`) — no `= expr
        // fn_item` body can express a Result-returning refuse, so it stays an explicit,
        // DN-41-cited gap rather than a forced/fabricated emission.
        Case {
            name: "narrow_gapped_cites_dn41",
            rust: "impl Narrow<u8> for u16 { fn narrow(self) -> Result<u8, NarrowError> { u8::try_from(self) } }",
            expect: Expect::Gapped {
                category: Category::Impl,
            },
        },
        // KNOWN HARD GAP: multi-statement fn body (an interior statement that is neither a
        // simple `let` nor the trailing expression).
        Case {
            name: "multi_stmt_body_gap",
            rust: "fn foo(x: bool) -> bool { let y = x; println!(\"{}\", 1); y }",
            expect: Expect::Gapped {
                category: Category::MultiStmtBody,
            },
        },
        // A string literal maps to a `StrLit` (grammar line 414/430; M-910/M-911) — reachable in
        // an emittable body as a call argument (its type is inferred, not named). The Rust `\n`
        // decodes to a raw newline which is re-escaped back to `\n` in the emitted StrLit.
        Case {
            name: "string_literal_arg_emits_strlit",
            rust: "fn f(x: u8) -> u8 { g(x, \"hi\\n\") }",
            expect: Expect::Emitted {
                item: "f",
                contains: "g(x, \"hi\\n\")",
            },
        },
        // A float literal maps to a `FloatLit` (grammar line 414/443; ADR-040/M-897) when its
        // digit string is a well-formed, finite FloatLit — reachable as a call argument.
        Case {
            name: "float_literal_arg_emits_floatlit",
            rust: "fn f(x: u8) -> u8 { g(x, 1.5) }",
            expect: Expect::Emitted {
                item: "f",
                contains: "g(x, 1.5)",
            },
        },
        // An exponent-form float likewise maps (`syn` normalizes `E`→`e`, drops the `+`).
        Case {
            name: "float_exponent_arg_emits_floatlit",
            rust: "fn f(x: u8) -> u8 { g(x, 2.5E+3) }",
            expect: Expect::Emitted {
                item: "f",
                contains: "g(x, 2.5e3)",
            },
        },
        // An explicit-element array maps to a `ListLit` (grammar line 415; RFC-0032 D3) —
        // reachable as a call argument.
        Case {
            name: "array_literal_arg_emits_listlit",
            rust: "fn f(x: u8) -> u8 { g(x, [x, x]) }",
            expect: Expect::Emitted {
                item: "f",
                contains: "g(x, [x, x])",
            },
        },
        // KNOWN HARD GAP: a string literal carrying a control char with no Mycelium escape
        // (`\x07` bell) — StrLit has no `\xNN` form, so it is never-silently gapped, never emitted
        // as a raw byte (G2/VR-5).
        Case {
            name: "string_control_char_gapped",
            rust: "fn f(x: u8) -> u8 { g(x, \"\\x07\") }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // KNOWN HARD GAP: a Rust-only float shape (trailing-dot `2.` → digit string "2.", empty
        // fraction) has no faithful Mycelium FloatLit spelling — gapped rather than reshaped (VR-5).
        Case {
            name: "float_trailing_dot_gapped",
            rust: "fn f(x: u8) -> u8 { g(x, 2.) }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // KNOWN HARD GAP: a well-shaped float whose value is not finite binary64 (`1e999` → +inf)
        // — a literal is a conversion boundary, so out-of-range is a never-silent refuse, never a
        // silent ±inf (ADR-040 §2.4).
        Case {
            name: "float_non_finite_gapped",
            rust: "fn f(x: u8) -> u8 { g(x, 1e999) }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // KNOWN HARD GAP: an array-repeat `[x; N]` — `ListLit` has no repeat form.
        Case {
            name: "array_repeat_gapped",
            rust: "fn f(x: u8) -> u8 { g(x, [x; 4]) }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // A bounded generic type parameter has no bare-identifier `type_params` mapping.
        Case {
            name: "generic_bound_gap",
            rust: "fn foo<T: Clone>(x: T) -> T { x }",
            expect: Expect::Gapped {
                category: Category::GenericBound,
            },
        },
        // M-1006 (E33-1): a named-field enum variant whose fields resolve now emits POSITIONALLY
        // (`A { x: u8 }` -> `A(Binary{8})`), names dropped + recorded via a `NamedFieldDrop` sub-gap.
        Case {
            name: "payload_variant_named_fields_emits_positionally",
            rust: "enum Foo { A { x: u8 }, B }",
            expect: Expect::EmittedAndGapped {
                item: "Foo",
                contains: "type Foo = A(Binary{8}) | B",
                sub_gap_category: Category::NamedFieldDrop,
            },
        },
        // M-1006 §8.14: a named-field variant with a `String` field now EMITS — `String` maps to
        // `Bytes` (RFC-0033 §3.2), names dropped + recorded via a `NamedFieldDrop` sub-gap.
        Case {
            name: "payload_variant_string_field_maps_to_bytes",
            rust: "enum Msg { Text { s: String }, Empty }",
            expect: Expect::EmittedAndGapped {
                item: "Msg",
                contains: "type Msg = Text(Bytes) | Empty",
                sub_gap_category: Category::NamedFieldDrop,
            },
        },
        // M-1006: a named-field variant with an UNMAPPABLE field type (`char`) still gaps — the
        // variant's own precise reason wins (mapped before the resolvability gate).
        Case {
            name: "payload_variant_unmappable_field_still_gaps",
            rust: "enum Bad { A { c: char } }",
            expect: Expect::Gapped {
                category: Category::PayloadVariant,
            },
        },
        // `#[derive(..)]` (any non-doc attribute) is dropped but recorded — the item is still
        // emitted (structural mapping doesn't need the derive), with a DeriveAttr sub-gap.
        Case {
            name: "derive_attr_sub_gap",
            rust: "#[derive(Debug, Clone)]\nenum Foo { A, B }",
            expect: Expect::EmittedAndGapped {
                item: "Foo",
                contains: "type Foo = A | B;",
                sub_gap_category: Category::DeriveAttr,
            },
        },
        // M-1001: a `use` import is FLAGGED, not emitted — the transpiler has no cross-nodule symbol
        // table so it cannot confirm the path resolves (the vet loop confirms such imports fail
        // `myc check` name-resolution), and an emitted `use` poisons the whole draft's check.
        Case {
            name: "simple_use_gapped",
            rust: "use a::b::C;",
            expect: Expect::Gapped {
                category: Category::Import,
            },
        },
        // Grouped `use` is likewise an Import gap.
        Case {
            name: "grouped_use_gap",
            rust: "use a::{b, c};",
            expect: Expect::Gapped {
                category: Category::Import,
            },
        },
        // M-1001: a type whose name is a Mycelium reserved word (`Float`) can't be emitted verbatim
        // (it would lex as a keyword) — gapped ReservedWord, never renamed (VR-5/G2).
        Case {
            name: "reserved_type_name",
            rust: "enum Float { A, B }",
            expect: Expect::Gapped {
                category: Category::ReservedWord,
            },
        },
        // M-1001: a variant/constructor named a reserved word (`Exact`) — the collision that poisoned
        // `mycelium-l1/src/eval.rs`'s parse in the §8.7 baseline.
        Case {
            name: "reserved_variant",
            rust: "enum GuaranteeStrength { Exact, Loose }",
            expect: Expect::Gapped {
                category: Category::ReservedWord,
            },
        },
        // Shared-reference erasure (this leaf, ADR-003): a fn whose params are `&T` shared references
        // now maps — the references are erased so the signature becomes value params, exactly as the
        // hand-port renders it. This is the item-level effect that unblocks emission (the real-corpus
        // shape: `fn digest_eq(a: &ContentHash, b: &ContentHash) -> bool`).
        Case {
            name: "shared_ref_params_emit",
            rust: "fn digest_eq(a: &Ordering, b: &Ordering) -> bool { a == b }",
            expect: Expect::Emitted {
                item: "digest_eq",
                contains: "fn digest_eq(a: Ordering, b: Ordering) => Bool = a == b;",
            },
        },
        // NEVER-SILENT CASCADE: a fn taking `&mut T` stays gapped — a mutable reference has no
        // value-semantic correspondence (ADR-003), so it is NOT erased. The whole fn gaps (Other),
        // never a partial emission that silently drops the mutation.
        Case {
            name: "mut_ref_param_gapped",
            rust: "fn bump(x: &mut Ordering) -> bool { true }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // M-1006 §8.14: a fn taking `&str` now emits — the reference erases to `str`, which maps to
        // `Bytes` (RFC-0033 §3.2). The real-corpus shape `fn message(&self) -> &str` (a String/`str`
        // accessor) is the class this unblocks.
        Case {
            name: "shared_ref_to_str_emits_bytes",
            rust: "fn tag(msg: &str) -> bool { true }",
            expect: Expect::Emitted {
                item: "tag",
                contains: "fn tag(msg: Bytes) => Bool = True;",
            },
        },
        // NEVER-SILENT CASCADE: a fn taking `&char` still gaps — the reference erases but the referent
        // `char` has no confirmed base_type arm, so the honest deeper blocker surfaces (Other), never
        // a fabricated emission.
        Case {
            name: "shared_ref_to_unmappable_referent_still_gapped",
            rust: "fn is_err(c: &char) -> bool { true }",
            expect: Expect::Gapped {
                category: Category::Other,
            },
        },
        // ── trx2 Lane C Deliverable 1: operand-type-gated operator emission ─────────────────────
        // (verify-first, mitigation #14 — every surface name below was confirmed against the real
        // built `target/debug/myc`/`target/debug/myc-check` toolchain; see this module's
        // `binop_operand_gated_forms_check_clean` live-oracle test for the `myc check`-clean
        // proof, and `emit.rs`'s `Expr::Binary` arm doc for the full citation trail.)
        //
        // Both operands are known `Binary{16}` params (from `MappedSig::params` via `sig_type_env`)
        // -> `&`/`|` rewrite to the bare-call prim forms `and`/`or` (the glyph desugar target
        // `band`/`bor` is NOT a prim — `myc check`-confirmed to fail with no import).
        Case {
            name: "bitand_known_binary_emits_and_call",
            rust: "fn f(a: u16, b: u16) -> u16 { a & b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "and(a, b)",
            },
        },
        Case {
            name: "bitor_known_binary_emits_or_call",
            rust: "fn f(a: u16, b: u16) -> u16 { a | b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "or(a, b)",
            },
        },
        // `^` is already the correct prim name after the parser's glyph desugar (`Tok::Caret` ->
        // word `"xor"`, which IS a bare-call prim) — left as the unchanged glyph; no rewrite.
        Case {
            name: "bitxor_known_binary_stays_glyph",
            rust: "fn f(a: u16, b: u16) -> u16 { a ^ b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "a ^ b",
            },
        },
        // `!=`/`>` desugar to `ne`/`gt`, which are non-`pub` `lib/std/cmp.myc` functions, not
        // prims — a bare `ne(a,b)`/`gt(a,b)` call fails identically to the glyph (both parse to the
        // same `Expr::App`). The verified fix composes them from the `eq`/`lt` prims directly
        // (exactly `cmp.myc`'s own `ne{N}`/`gt{N}` derivation), which DOES check clean with no
        // import.
        Case {
            name: "ne_known_binary_composes_from_eq",
            rust: "fn f(a: u16, b: u16) -> bool { a != b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "(match eq(a, b) { 0b1 => False, _ => True })",
            },
        },
        Case {
            name: "gt_known_binary_composes_from_eq_and_lt",
            rust: "fn f(a: u16, b: u16) -> bool { a > b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "(match eq(a, b) { 0b1 => False, _ => match lt(a, b) { 0b1 => False, \
                            _ => True } })",
            },
        },
        // `==`/`<` are RFC-0032 D1's ratified glyphs — unchanged by this deliverable even though
        // both operands here are known `Binary{16}` (the operand-gate only fires for the
        // `& | != >` arms).
        Case {
            name: "eq_lt_known_binary_stay_glyphs",
            rust: "fn f(a: u16, b: u16) -> bool { a == b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "a == b",
            },
        },
        // Non-`Binary{N}` operand (a `bool` param, mapped to `Bool` — never a `Binary{N}` text per
        // `map_type`) keeps the CURRENT (pre-deliverable) emission unchanged: still the bare glyph,
        // not a call. Proves the gate is genuinely operand-typed, not unconditional.
        Case {
            name: "bitand_non_binary_operand_keeps_glyph",
            rust: "fn f(a: bool, b: bool) -> bool { a & b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "a & b",
            },
        },
        Case {
            name: "gt_non_binary_operand_keeps_glyph",
            rust: "fn f(a: bool, b: bool) -> bool { a > b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "a > b",
            },
        },
        // One operand unknown (a call result, not a bare in-scope identifier) — the gate requires
        // BOTH operands resolved, so this also keeps the glyph (never a half-composed emission).
        Case {
            name: "ne_one_operand_unresolved_keeps_glyph",
            rust: "fn f(a: u16, b: u16) -> bool { a != g(b) }",
            expect: Expect::Emitted {
                item: "f",
                contains: "a != g(b)",
            },
        },
        // A `let`-aliased local of a known `Binary{N}` param is itself recognized as known (the
        // `Stmt::Local` env-extension case (a): "RHS is a bare param already in the env").
        Case {
            name: "let_alias_of_known_binary_extends_env",
            rust: "fn f(a: u16, b: u16) -> bool { let c = a; c > b }",
            expect: Expect::Emitted {
                item: "f",
                contains: "match eq(c, b) { 0b1 => False, _ => match lt(c, b) { 0b1 => False, \
                            _ => True } }",
            },
        },
        // An impl method's `self` parameter is threaded into the env too (via `sig_type_env`
        // already covering the `Receiver` arm's `("self", ty)` entry from `map_signature`) — a
        // `Binary{N}`-mapped `Self` type (here `u16` -> `Binary{16}`) participates in the same
        // operand gate. Uses a non-`Widen` trait name so `try_width_cast_widen_body`'s DN-41
        // special-case (which bypasses this body-emission path entirely) never intercepts it.
        Case {
            name: "impl_method_self_known_binary_participates_in_gate",
            rust: "impl Foo for u16 { fn m(self, b: u16) -> u16 { self & b } }",
            expect: Expect::Emitted {
                item: "impl Foo for Binary{16}",
                contains: "and(self, b)",
            },
        },
    ]
}

fn run(case: &Case) {
    let (myc, report) = transpile_source(case.rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("case `{}` failed to parse/transpile: {e}", case.name));
    match &case.expect {
        Expect::Emitted { item, contains } => {
            assert!(
                report.emitted_items.iter().any(|n| n == item),
                "case `{}`: expected `{item}` in emitted_items, got {:?}",
                case.name,
                report.emitted_items
            );
            assert!(
                myc.contains(contains),
                "case `{}`: expected .myc to contain `{contains}`, got:\n{myc}",
                case.name
            );
        }
        Expect::Gapped { category } => {
            assert!(
                report.emitted_items.is_empty(),
                "case `{}`: expected no emitted items, got {:?}",
                case.name,
                report.emitted_items
            );
            assert!(
                report.gaps.iter().any(|g| g.category == *category),
                "case `{}`: expected a gap of category {:?}, got {:?}",
                case.name,
                category.as_str(),
                report
                    .gaps
                    .iter()
                    .map(|g| g.category.as_str())
                    .collect::<Vec<_>>()
            );
        }
        Expect::EmittedAndGapped {
            item,
            contains,
            sub_gap_category,
        } => {
            assert!(
                report.emitted_items.iter().any(|n| n == item),
                "case `{}`: expected `{item}` in emitted_items, got {:?}",
                case.name,
                report.emitted_items
            );
            assert!(
                myc.contains(contains),
                "case `{}`: expected .myc to contain `{contains}`, got:\n{myc}",
                case.name
            );
            assert!(
                report.gaps.iter().any(|g| g.category == *sub_gap_category),
                "case `{}`: expected a sub-gap of category {:?}, got {:?}",
                case.name,
                sub_gap_category.as_str(),
                report
                    .gaps
                    .iter()
                    .map(|g| g.category.as_str())
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn emit_fixture_corpus() {
    for case in cases() {
        run(&case);
    }
}

/// Regression guard (High finding, G2/DN-34 §4, extended by DN-41/M-873 follow-on): the
/// never-silent gap mechanism means a *gapped* item's `.myc` text is never emitted at all — pin
/// that down directly for the bool-`Self` widen shape (which still has no `width_cast` witness
/// path — `Self` doesn't map to `Binary{N}`) so a future change that started emitting a
/// partial/fallback body for this case would fail loudly here, not just leave `emitted_items`
/// empty while still leaking fabricated text into the `.myc` output.
#[test]
fn widen_bool_from_call_produces_no_fabricated_myc_text() {
    let rust = "impl Widen<u32> for bool { fn widen(self) -> u32 { u32::from(self) } }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.is_empty(),
        "expected the bool Widen impl to be fully gapped, got emitted_items={:?}",
        report.emitted_items
    );
    assert!(
        !myc.contains("from("),
        "emitted .myc text must never contain a fabricated `from(...)` call (from is not a \
         Mycelium builtin — G2/DN-34 §4), got:\n{myc}"
    );
}

/// The DN-41 companion of the guard above: a `Binary{N}`->`Binary{M}` widen must emit a **real**
/// `width_cast(self, ..)` call — never a fabricated `from(...)` call, and never left gapped now
/// that the faithful mapping exists.
#[test]
fn widen_binary_emits_width_cast_not_fabricated_from() {
    let rust = "impl Widen<u16> for u8 { fn widen(self) -> u16 { u16::from(self) } }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report
            .emitted_items
            .iter()
            .any(|n| n == "impl Widen[Binary{16}] for Binary{8}"),
        "expected the Binary widen impl to be emitted via width_cast, got emitted_items={:?}",
        report.emitted_items
    );
    assert!(
        !myc.contains("from("),
        "emitted .myc text must never contain a fabricated `from(...)` call (from is not a \
         Mycelium builtin — G2/DN-34 §4), got:\n{myc}"
    );
    assert!(
        myc.contains("width_cast(self, 0b0000_0000_0000_0000)"),
        "expected a real `width_cast(self, ..)` call with a 16-bit zero witness, got:\n{myc}"
    );
}

/// DN-41 companion: `Narrow::narrow` is fallible and has no `= expr` surface, so it must stay an
/// honest gap whose reason cites DN-41 — never a fabricated `try_from`/`?`-shaped emission.
#[test]
fn narrow_gap_cites_dn41_and_produces_no_fabricated_myc_text() {
    let rust = "impl Narrow<u8> for u16 { fn narrow(self) -> Result<u8, NarrowError> { u8::try_from(self) } }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.is_empty(),
        "expected the Narrow impl to be fully gapped, got emitted_items={:?}",
        report.emitted_items
    );
    assert!(
        !myc.contains("try_from") && !myc.contains("width_cast"),
        "narrow bodies must never be fabricated (no try_from-shaped or width_cast emission), \
         got:\n{myc}"
    );
    assert!(
        report.gaps.iter().any(|g| g.reason.contains("DN-41")),
        "expected the narrow gap's reason to cite DN-41, got {:?}",
        report.gaps.iter().map(|g| &g.reason).collect::<Vec<_>>()
    );
}

/// Never-silent guard (G2/VR-5): a string literal that cannot be faithfully re-escaped (a control
/// char with no Mycelium `\xNN`/`\u{..}` form) is gapped, and its raw byte NEVER leaks into the
/// emitted `.myc` text — a future change that started emitting the raw control byte (or a fabricated
/// `\x07` escape Mycelium's lexer would reject) would fail loudly here.
#[test]
fn string_control_char_never_leaks_raw_byte() {
    let rust = "fn f(x: u8) -> u8 { g(x, \"\\x07\") }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.is_empty(),
        "expected the control-char string body to be fully gapped, got emitted_items={:?}",
        report.emitted_items
    );
    assert!(
        !myc.contains('\u{7}') && !myc.contains("\\x07"),
        "gapped control-char string must never leak a raw byte or a fabricated `\\x07` escape \
         (StrLit has no `\\xNN` form), got:\n{myc}"
    );
}

/// The sharpened `MultiStmtBody` reason (this leaf, E33-1 M-1006 phase-1) names the *kind* of the
/// offending interior statement — a nested item (local `static`/`const`/`fn`), a macro invocation,
/// or a semicolon-terminated statement expression — so the gap report is precise, not generic
/// (G2). Each is a genuinely design-blocked form (no local-item / no macro / value-discard has no
/// grammar surface); this pins the diagnostic text, not any emission.
#[test]
fn multi_stmt_body_reason_names_the_statement_kind() {
    let cases = [
        // A local `static` item statement (the real `mono_nanos` shape).
        (
            "fn f(x: u8) -> u8 { static Z: u8 = 0; x }",
            "nested item declaration",
        ),
        // A macro-invocation statement (the real `rejection_sample_u64` `debug_assert!` shape).
        (
            "fn f(x: u8) -> u8 { debug_assert!(x > 0); x }",
            "macro-invocation statement",
        ),
        // A semicolon-terminated (value-discarding) statement expression.
        (
            "fn f(x: u8) -> u8 { g(x); x }",
            "semicolon-terminated (value-discarding) statement expression",
        ),
    ];
    for (rust, needle) in cases {
        let (_, report) = transpile_source(rust, "fixture.rs", "fixture")
            .unwrap_or_else(|e| panic!("failed to parse/transpile `{rust}`: {e}"));
        assert!(
            report
                .gaps
                .iter()
                .any(|g| g.category == Category::MultiStmtBody && g.reason.contains(needle)),
            "case `{rust}`: expected a MultiStmtBody gap whose reason mentions `{needle}`, got {:?}",
            report
                .gaps
                .iter()
                .map(|g| (g.category.as_str(), g.reason.as_str()))
                .collect::<Vec<_>>()
        );
    }
}

use super::vet::find_myc_check;

/// **The verify-first proof** (mitigation #14) for trx2 Lane C Deliverable 1: every operand-gated
/// rewrite in `Expr::Binary` (`and`/`or` for `&`/`|`, the `eq`/`lt`-composed forms for `!=`/`>`) is
/// run through the REAL `myc-check` oracle here, not just asserted as a substring match (the
/// `emit_fixture_corpus` cases above prove the *text*; this proves the text actually **type-checks**
/// with zero imports — the property the whole deliverable is for). Skips gracefully (never fails)
/// when `myc-check` is not built, exactly like `src/tests/vet.rs`'s `live_myc_check_classifies_clean_and_broken`.
#[test]
fn binop_operand_gated_forms_check_clean() {
    let Some(bin) = find_myc_check() else {
        eprintln!(
            "emit: live oracle test skipped — no runnable myc-check (set MYC_CHECK_CMD or build \
             `cargo build -p mycelium-check --bin myc-check`). The fixture-corpus text assertions \
             above still cover the emitted shape."
        );
        return;
    };

    let dir = std::env::temp_dir().join(format!(
        "mycelium-transpile-emit-binop-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // Every rewrite this deliverable makes, in ONE nodule (mirrors the real driver: one file, no
    // cross-nodule imports) — `and`/`or`/`eq`/`lt` must all resolve as bare-call prims with no
    // `use`, and the composed `!=`/`>` match expressions must type as `Bool`.
    let rust_snippets = [
        "fn f_and(a: u16, b: u16) -> u16 { a & b }",
        "fn f_or(a: u16, b: u16) -> u16 { a | b }",
        "fn f_ne(a: u16, b: u16) -> bool { a != b }",
        "fn f_gt(a: u16, b: u16) -> bool { a > b }",
        // `^` (unchanged glyph) rides along as a negative control — it must ALSO check clean
        // (it already did before this deliverable; this pins that it still does).
        "fn f_xor(a: u16, b: u16) -> u16 { a ^ b }",
    ];
    for (i, rust) in rust_snippets.iter().enumerate() {
        let (myc, report) = transpile_source(rust, "fixture.rs", "oracle")
            .unwrap_or_else(|e| panic!("failed to parse/transpile `{rust}`: {e}"));
        assert!(
            !report.emitted_items.is_empty(),
            "case {i} (`{rust}`) failed to emit at all: gaps={:?}",
            report.gaps
        );
        let path = dir.join(format!("case_{i}.myc"));
        std::fs::write(&path, &myc).expect("write case .myc");

        let checker = crate::vet::MycChecker {
            command: vec![bin.display().to_string()],
            cwd: None,
        };
        let rec = checker.vet_file(&path, "fixture.rs", 1, 1);
        assert_eq!(
            rec.class,
            crate::vet::VetClass::Clean,
            "case {i} (`{rust}`) must check CLEAN with the real myc-check oracle — emitted:\n{myc}\n\
             diagnostic={:?}",
            rec.diagnostic
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Regression guard (HIGH finding, PR #1299 review, fix 1a) for the `Stmt::Local` shadow-
/// invalidation bug: a `let` that **shadows** an existing name with an RHS of *unknown* type left
/// the shadowed name's *stale* prior type in `local_env`, so `Expr::Binary`'s operand-type gate
/// could keep firing using a type that no longer applies to the (now-shadowed) name. Repro: `let x
/// = a;` (RHS is the known `Binary{16}` param `a`, so `x` is recorded as `Binary{16}`), then `let x
/// = true;` shadows `x` with a bool-literal RHS (unknown type to this module — never a `Binary{N}`
/// guess). The tail `x != b` must fall back to the plain `!=` glyph (the shadowed `x`'s type is
/// invalidated), never the `eq`/`lt`-composed form the gate would wrongly emit using the *old*
/// binding.
#[test]
fn let_shadow_with_unknown_type_invalidates_stale_binary_env_entry() {
    let rust = "fn f(a: u16, b: u16) -> bool { let x = a; let x = true; x != b }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.iter().any(|n| n == "f"),
        "expected `f` to emit, got emitted_items={:?}",
        report.emitted_items
    );
    assert!(
        myc.contains("x != b"),
        "expected the shadowed `x != b` tail to fall back to the plain glyph (the shadow \
         invalidates x's known-Binary{{16}} type from the OLD `let x = a;` binding), got:\n{myc}"
    );
    assert!(
        !myc.contains("match eq(x, b)"),
        "the operand-type gate must NOT fire on the shadowed `x` using the stale OLD binding's \
         type — `let x = true;` shadows it with an unknown-type RHS, got:\n{myc}"
    );
}

/// Regression guard (HIGH finding, PR #1299 review, fix 1b) for the match-arm pattern-binding gap:
/// a name a match arm's pattern **binds** (here `Wrap::A(x)`'s `u32` payload `x`) must never
/// inherit an outer local's type through `env` — the outer `x: u16` (`Binary{16}`) parameter must
/// not leak onto the pattern-bound `x`, which is a *different* binding (the enum payload,
/// `Binary{32}`). Before the fix this mis-fired `and(x, b)` using the outer `Binary{16}` — a real
/// `myc check` width-mismatch failure once the pattern-bound `x` (actually `Binary{32}`) is
/// resolved against `b: Binary{16}`. The arm must fall back to the plain `&` glyph.
#[test]
fn match_arm_pattern_bound_name_invalidates_outer_binary_env_entry() {
    let rust = "enum Wrap { A(u32), B } fn f(x: u16, b: u16, w: Wrap) -> u16 { match w { \
                Wrap::A(x) => x & b, Wrap::B => b } }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.iter().any(|n| n == "f"),
        "expected `f` to emit, got emitted_items={:?}",
        report.emitted_items
    );
    assert!(
        myc.contains("x & b"),
        "expected the `Wrap::A(x) => x & b` arm to fall back to the plain glyph (the \
         pattern-bound `x` is a distinct Binary{{32}} payload, not the outer u16 param), \
         got:\n{myc}"
    );
    assert!(
        !myc.contains("and(x, b)"),
        "the operand-type gate must NOT fire using the outer `x: u16` param's type for the \
         pattern-bound `x` (a real Binary{{32}} payload vs Binary{{16}} `b` — a genuine \
         width-mismatch myc-check failure if emitted), got:\n{myc}"
    );
}

/// **The verify-first live-oracle proof** (mitigation #14) for both PR #1299 review fixes above:
/// runs the two repros' emitted `.myc` through the REAL `myc-check` oracle. Honest finding
/// (never a silently-skipped false-green, G2): neither repro's *fixed* (fallen-back-to-glyph)
/// emission is actually `myc check`-clean — but for a completely different, PRE-EXISTING and
/// separately-tracked reason than the bug being fixed here. `!=`/`&` in the un-gated (operand-type
/// unknown) fallback path desugar to the bare word calls `ne`/`band`, which are not resolvable
/// prims with no import (exactly the failure mode this module's `Expr::Binary` doc already
/// documents for every other un-gated `!=`/`&` case, e.g. `bitand_non_binary_operand_keeps_glyph`
/// above) — this is orthogonal to, and unaffected by, the type-env shadow/pattern-binding fixes.
/// What this test proves is the *negative* the fixes exist for: the diagnostic is the KNOWN
/// `ne`/`band` gap, never a mismatched-width `and`/`eq` prim-call failure the pre-fix bug would
/// have risked (or, worse, a coincidentally-succeeding wrong-type `Clean` result). Skips
/// gracefully (never fails) when `myc-check` is not built.
#[test]
fn shadow_and_pattern_bound_fixes_fall_back_to_known_gap_not_wrong_prim_call() {
    let Some(bin) = find_myc_check() else {
        eprintln!(
            "emit: live oracle test skipped — no runnable myc-check (set MYC_CHECK_CMD or build \
             `cargo build -p mycelium-check --bin myc-check`). The fixture-corpus text \
             assertions above still cover the emitted shape."
        );
        return;
    };

    let dir = std::env::temp_dir().join(format!(
        "mycelium-transpile-emit-shadow-pattern-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // (case name, rust source, the un-gated glyph word this fallback desugars to — the honest,
    // pre-existing gap the diagnostic must name; NOT the mismatched-width prim the pre-fix bug
    // would have wrongly emitted).
    let cases = [
        (
            "let_shadow",
            "fn f(a: u16, b: u16) -> bool { let x = a; let x = true; x != b }",
            "`ne`",
        ),
        (
            "match_arm_pattern_bound",
            "enum Wrap { A(u32), B } fn f(x: u16, b: u16, w: Wrap) -> u16 { match w { \
             Wrap::A(x) => x & b, Wrap::B => b } }",
            "`band`",
        ),
    ];
    for (name, rust, expected_gap_word) in cases {
        let (myc, report) = transpile_source(rust, "fixture.rs", "oracle")
            .unwrap_or_else(|e| panic!("case `{name}` (`{rust}`) failed to parse/transpile: {e}"));
        assert!(
            !report.emitted_items.is_empty(),
            "case `{name}` (`{rust}`) failed to emit at all: gaps={:?}",
            report.gaps
        );
        // Never the wrong-type prim call the pre-fix bug would have risked.
        assert!(
            !myc.contains("eq(x, b)") && !myc.contains("and(x, b)"),
            "case `{name}`: must never emit the mismatched-type prim-call form the shadow/\
             pattern-binding bug would have produced, got:\n{myc}"
        );

        let path = dir.join(format!("{name}.myc"));
        std::fs::write(&path, &myc).expect("write case .myc");
        let checker = crate::vet::MycChecker {
            command: vec![bin.display().to_string()],
            cwd: None,
        };
        let rec = checker.vet_file(&path, "fixture.rs", 1, 1);
        assert_eq!(
            rec.class,
            crate::vet::VetClass::CheckError,
            "case `{name}` (`{rust}`) was expected to hit the KNOWN pre-existing {expected_gap_word} \
             gap (never silently `Clean` on a wrong-type basis) — emitted:\n{myc}\ndiagnostic={:?}",
            rec.diagnostic
        );
        assert!(
            rec.diagnostic.contains(expected_gap_word),
            "case `{name}`: expected the diagnostic to name the known pre-existing \
             {expected_gap_word} gap, got: {}",
            rec.diagnostic
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
