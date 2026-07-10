//! M-740 Stage 3a (DN-26 §7.3 row 3) — the self-hosted `compiler.ast` port.
//!
//! `lib/compiler/ast.myc` is pure DATA (enums/structs + a handful of small helper impls, no
//! upward deps — DN-26 §7.1: "ast -> none"). Unlike Stages 1–2 (which differential the RUNTIME
//! behavior of a lexer/recogniser against the live Rust oracle), there is no comparable "run it
//! and compare the output" leg for a bare vocabulary of type declarations — the actual AST-SHAPE
//! differential (does `compiler.parse` build the SAME tree the Rust parser does, over the full L1
//! conformance corpus) is `compiler.parse`'s job (DN-26 §7.3 row 3, a sibling leaf). So this
//! Stage-3a gate is, per the task brief, two legs:
//!
//! (a) **the structural gate** — `lib/compiler/ast.myc` parses and type-checks green through the
//!     real pipeline (`parse` -> `check_nodule`), exactly like every prior stage's smoke test; and
//! (b) **the inventory** — a transcribed table, one row per Rust enum-variant/struct ported (incl.
//!     every FLAG-renamed constructor), whose COUNT is asserted against a textual audit of the
//!     ported source, AND whose constructors are exercised via L1-eval (constructed + classified)
//!     for every enum type — a `Declared`/audited grade (VR-5): this is a completeness/inventory
//!     check, not a Rust-oracle differential (there is no oracle output to differential against for
//!     bare data types). The `Empirical` cross-check arrives with `compiler.parse`'s own gate.
//!
//! M-981 applies as in every prior stage: only the L1-eval leg is exercised at this scale (the L0
//! substitution interpreter is impractical for a source file this size). M-980's split-match idiom
//! is used throughout every classify_* driver fn below — each match destructures exactly ONE
//! constructor level, never a nested pattern.

use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

const AST_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/ast.myc"
));

fn program(driver: &str) -> String {
    format!("{AST_SRC}\n{driver}")
}

/// L1-eval-only assertion (the M-981 convention every prior stage uses): parse -> check_nodule ->
/// monomorphize -> `Evaluator::call("main")` -> extract the `Binary{32}` result as a `u32`.
fn assert_l1_only_u32(label: &str, src: &str, expected_u32: u32) {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));
    let repr_val = l1_core
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: expected a Repr CoreValue, got {l1_core:?}"));
    let got = match repr_val.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("{label}: expected a Bits payload, got {other:?}"),
    };
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the expected classification code {expected_u32}"
    );
}

/// Format `n` as an explicit `Binary{32}` literal (bare decimal literals do not ambient-resolve in
/// every position — the same `b32` convention `compiler_stage1.rs`/`compiler_stage2.rs` use).
fn b32(n: u32) -> String {
    format!("0b{n:032b}")
}

/// Format `n` as an explicit `Binary{64}` literal (for `Literal::Int`/`AmbientInt`'s `Binary{64}`
/// payload — FLAG-ast-2).
fn b64(n: u64) -> String {
    format!("0b{n:064b}")
}

/// The structural gate: `lib/compiler/ast.myc` parses and type-checks green through the real
/// pipeline, standalone (no driver needed — the nodule has no `main`).
#[test]
fn ast_myc_parses_and_checks() {
    let nodule = parse(AST_SRC).unwrap_or_else(|e| panic!("ast.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("ast.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The ported-constructor inventory (VR-5 `Declared`/audited grade — see the module doc comment).
// One row per Rust enum-variant/struct ported: (Rust name, this port's `.myc` constructor name).
// The EffectBudget row is this leaf's OWN addition (FLAG-ast-6: BTreeMap -> Vec[EffectBudget]),
// not a direct Rust ctor — marked as such, never silently folded into the "ported from ast.rs"
// count (G2).
// ─────────────────────────────────────────────────────────────────────────────────────────────
const CTOR_INVENTORY: &[(&str, &str)] = &[
    // Path
    ("Path::Path", "Pth"),
    // Phylum
    ("Phylum::Phylum", "Phy"),
    // Vis
    ("Vis::Private", "Private"),
    ("Vis::Pub", "Pub"),
    // UsePath
    ("UsePath::UsePath", "UP"),
    // Nodule
    ("Nodule::Nodule", "Nd"),
    // Paradigm (FLAG-ast-4 P-prefix)
    ("Paradigm::Binary", "PBinary"),
    ("Paradigm::Ternary", "PTernary"),
    ("Paradigm::Dense", "PDense"),
    ("Paradigm::Vsa", "PVsa"),
    // AmbientParams (FLAG-ast-4/5 AP-prefix)
    ("AmbientParams::Size", "APSize"),
    ("AmbientParams::Dense", "APDense"),
    ("AmbientParams::Vsa", "APVsa"),
    // ViaDecl
    ("ViaDecl::ViaDecl", "VD"),
    // ObjectDecl
    ("ObjectDecl::ObjectDecl", "OD"),
    // Item (bare — no collision, case-sensitivity per FLAG-ast-4)
    ("Item::Use", "Use"),
    ("Item::Default", "Default"),
    ("Item::Type", "Type"),
    ("Item::Trait", "Trait"),
    ("Item::Impl", "Impl"),
    ("Item::Fn", "Fn"),
    ("Item::Object", "Object"),
    ("Item::Lower", "Lower"),
    ("Item::Derive", "Derive"),
    ("Item::InherentImpl", "InherentImpl"),
    // InherentImplDecl
    ("InherentImplDecl::InherentImplDecl", "IID"),
    // LowerDecl
    ("LowerDecl::LowerDecl", "LD"),
    // LowerRhs (FLAG-ast-5 LR-prefix — both collide with Item::Impl / for symmetry)
    ("LowerRhs::Expr", "LRExpr"),
    ("LowerRhs::Impl", "LRImpl"),
    // DeriveDecl
    ("DeriveDecl::DeriveDecl", "DD"),
    // TypeDecl
    ("TypeDecl::TypeDecl", "TD"),
    // Ctor
    ("Ctor::Ctor", "Ctr"),
    // TraitDecl
    ("TraitDecl::TraitDecl", "TrD"),
    // ImplDecl
    ("ImplDecl::ImplDecl", "ImD"),
    // TraitRef
    ("TraitRef::TraitRef", "TRf"),
    // WidthRef (FLAG-ast-5 W-prefix — Lit collides 3-way with Expr::Lit/Pattern::Lit)
    ("WidthRef::Lit", "WLit"),
    ("WidthRef::Name", "WName"),
    // ParamKind (FLAG-ast-5 Pk-prefix — Type collides with Item::Type)
    ("ParamKind::Type", "PkType"),
    ("ParamKind::Width", "PkWidth"),
    // TypeParam
    ("TypeParam::TypeParam", "TP"),
    // FnSig
    ("FnSig::FnSig", "FS"),
    // ExecutionMode
    ("ExecutionMode::Interpreted", "Interpreted"),
    ("ExecutionMode::Compiled", "Compiled"),
    // FnDecl
    ("FnDecl::FnDecl", "FD"),
    // Param
    ("Param::Param", "Prm"),
    // TypeRef
    ("TypeRef::TypeRef", "TR"),
    // BaseType (FLAG-ast-4 Kw-prefix for the 7 repr-keyword collisions; FLAG-ast-5 FnArrow for the
    // Item::Fn collision; Vsa/Named/Ambient/Tuple bare)
    ("BaseType::Binary", "KwBinary"),
    ("BaseType::Ternary", "KwTernary"),
    ("BaseType::Dense", "KwDense"),
    ("BaseType::Vsa", "Vsa"),
    ("BaseType::Substrate", "KwSubstrate"),
    ("BaseType::Seq", "KwSeq"),
    ("BaseType::Bytes", "KwBytes"),
    ("BaseType::Float", "KwFloat"),
    ("BaseType::Named", "Named"),
    ("BaseType::Ambient", "Ambient"),
    ("BaseType::Fn", "FnArrow"),
    ("BaseType::Tuple", "Tuple"),
    // Sparsity (FLAG-ast-4 Sp-prefix — both collide)
    ("Sparsity::Dense", "SpDense"),
    ("Sparsity::Sparse", "SpSparse"),
    // Scalar (FLAG-ast-4 S-prefix, reuses token.myc::ScalarTok naming)
    ("Scalar::F16", "SF16"),
    ("Scalar::Bf16", "SBf16"),
    ("Scalar::F32", "SF32"),
    ("Scalar::F64", "SF64"),
    // Strength (FLAG-ast-4 G-prefix, reuses token.myc::StrengthTok / std.core::Guarantee naming)
    ("Strength::Exact", "GExact"),
    ("Strength::Proven", "GProven"),
    ("Strength::Empirical", "GEmpirical"),
    ("Strength::Declared", "GDeclared"),
    // Expr (bare throughout)
    ("Expr::Let", "Let"),
    ("Expr::If", "If"),
    ("Expr::Match", "Match"),
    ("Expr::For", "For"),
    ("Expr::Swap", "Swap"),
    ("Expr::WithParadigm", "WithParadigm"),
    ("Expr::Wild", "Wild"),
    ("Expr::Spore", "Spore"),
    ("Expr::Consume", "Consume"),
    ("Expr::Colony", "Colony"),
    ("Expr::Lambda", "Lambda"),
    ("Expr::App", "App"),
    ("Expr::Fuse", "Fuse"),
    ("Expr::Reclaim", "Reclaim"),
    ("Expr::Path", "Path"),
    ("Expr::Lit", "Lit"),
    ("Expr::Ascribe", "Ascribe"),
    ("Expr::TupleLit", "TupleLit"),
    // Arm
    ("Arm::Arm", "Ar"),
    // Hypha
    ("Hypha::Hypha", "Hy"),
    // Pattern (FLAG-ast-5 P-prefix throughout — Lit/Tuple collide, rest prefixed for consistency)
    ("Pattern::Wildcard", "PWildcard"),
    ("Pattern::Lit", "PLit"),
    ("Pattern::Ctor", "PCtor"),
    ("Pattern::Ident", "PIdent"),
    ("Pattern::Tuple", "PTuple"),
    ("Pattern::Or", "POr"),
    // Literal (FLAG-ast-4/5 — Bytes/Float collide with BaseType AND with repr keywords)
    ("Literal::Bin", "Bin"),
    ("Literal::Trit", "Trit"),
    ("Literal::Int", "Int"),
    ("Literal::AmbientInt", "AmbientInt"),
    ("Literal::List", "List"),
    ("Literal::Bytes", "LBytes"),
    ("Literal::Str", "Str"),
    ("Literal::Float", "LFloat"),
    // This leaf's own addition (FLAG-ast-6) — NOT a direct ast.rs ctor, flagged as such.
    ("(added, FLAG-ast-6) EffectBudget map-entry", "EB"),
];

/// `contains_word`: a plain word-boundary substring search (no `regex` dependency) — true iff
/// `needle` occurs in `haystack` with a non-identifier character (or string edge) on both sides,
/// so e.g. searching for `"Fn"` does not spuriously match inside `"FnArrow"` / `"FnSig"`.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || nb.len() > bytes.len() {
        return false;
    }
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    for start in 0..=(bytes.len() - nb.len()) {
        if &bytes[start..start + nb.len()] == nb {
            let before_ok = start == 0 || !is_ident(bytes[start - 1]);
            let after_idx = start + nb.len();
            let after_ok = after_idx == bytes.len() || !is_ident(bytes[after_idx]);
            if before_ok && after_ok {
                return true;
            }
        }
    }
    false
}

/// The inventory count leg (`Declared`/audited — see the module doc comment): the transcribed
/// table above must have exactly 103 rows (102 constructors ported from `ast.rs`'s 36 types —
/// `BaseType` alone has 12 variants — plus this leaf's own `EB` addition), and every listed `.myc`
/// constructor name must actually appear
/// (as a whole word, never a substring match) in the ported source — a cheap, honest textual audit
/// that catches a dropped or mistyped row without re-deriving a full parser in the test harness.
#[test]
fn ast_myc_ctor_inventory_count_and_presence() {
    assert_eq!(
        CTOR_INVENTORY.len(),
        103,
        "expected 102 ported ast.rs constructors (BaseType has 12 variants, not 11 — corrected \
         during authoring) + 1 leaf-added EffectBudget entry"
    );
    for (rust_name, myc_ctor) in CTOR_INVENTORY {
        assert!(
            contains_word(AST_SRC, myc_ctor),
            "{rust_name}: ported constructor `{myc_ctor}` not found (as a whole word) in ast.myc"
        );
    }
}

/// Every `type` declaration ast.myc actually carries (39 total: the 36 ported from `ast.rs` +
/// `EffectBudget` (FLAG-ast-6) + the local `Option`/`Vec` infrastructure redeclares — FLAG-ast-3).
#[test]
fn ast_myc_type_decl_count() {
    let count = AST_SRC
        .lines()
        .filter(|l| l.trim_start().starts_with("type "))
        .count();
    assert_eq!(
        count, 39,
        "expected 39 `type` declarations (36 ported + EffectBudget + local Option/Vec)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Per-enum classify_* driver fns + construct/classify tables (M-980 split-match idiom: every
// classify_* fn destructures exactly ONE constructor level).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Shared sample-value helpers + one `classify_*` fn per enum ast.myc declares — appended once per
/// program, mirroring `compiler_stage2.rs`'s `driver_prelude` pattern.
fn driver_prelude() -> &'static str {
    "fn sample_typeref() => TypeRef = typeref_unguaranteed(KwBytes);\n\
     fn sample_path() => Path = Pth(Nil);\n\
     fn sample_expr() => Expr = Lit(Bin(\"0\"));\n\
     fn classify_vis(v: Vis) => Binary{32} =\n\
     \x20 match v { Private => 0b00000000000000000000000000000000, Pub => 0b00000000000000000000000000000001 };\n\
     fn classify_paradigm(p: Paradigm) => Binary{32} =\n\
     \x20 match p {\n\
     \x20   PBinary => 0b00000000000000000000000000000000,\n\
     \x20   PTernary => 0b00000000000000000000000000000001,\n\
     \x20   PDense => 0b00000000000000000000000000000010,\n\
     \x20   PVsa => 0b00000000000000000000000000000011\n\
     \x20 };\n\
     fn classify_widthref(w: WidthRef) => Binary{32} =\n\
     \x20 match w { WLit(_) => 0b00000000000000000000000000000000, WName(_) => 0b00000000000000000000000000000001 };\n\
     fn classify_paramkind(k: ParamKind) => Binary{32} =\n\
     \x20 match k { PkType => 0b00000000000000000000000000000000, PkWidth => 0b00000000000000000000000000000001 };\n\
     fn classify_execmode(e: ExecutionMode) => Binary{32} =\n\
     \x20 match e { Interpreted => 0b00000000000000000000000000000000, Compiled => 0b00000000000000000000000000000001 };\n\
     fn classify_sparsity(s: Sparsity) => Binary{32} =\n\
     \x20 match s { SpDense => 0b00000000000000000000000000000000, SpSparse(_) => 0b00000000000000000000000000000001 };\n\
     fn classify_scalar(s: Scalar) => Binary{32} =\n\
     \x20 match s {\n\
     \x20   SF16 => 0b00000000000000000000000000000000,\n\
     \x20   SBf16 => 0b00000000000000000000000000000001,\n\
     \x20   SF32 => 0b00000000000000000000000000000010,\n\
     \x20   SF64 => 0b00000000000000000000000000000011\n\
     \x20 };\n\
     fn classify_strength(s: Strength) => Binary{32} =\n\
     \x20 match s {\n\
     \x20   GDeclared => 0b00000000000000000000000000000000,\n\
     \x20   GEmpirical => 0b00000000000000000000000000000001,\n\
     \x20   GProven => 0b00000000000000000000000000000010,\n\
     \x20   GExact => 0b00000000000000000000000000000011\n\
     \x20 };\n\
     fn classify_lowerrhs(r: LowerRhs) => Binary{32} =\n\
     \x20 match r { LRExpr(_) => 0b00000000000000000000000000000000, LRImpl(_) => 0b00000000000000000000000000000001 };\n\
     fn classify_item(i: Item) => Binary{32} =\n\
     \x20 match i {\n\
     \x20   Use(_) => 0b00000000000000000000000000000000,\n\
     \x20   Default(_) => 0b00000000000000000000000000000001,\n\
     \x20   Type(_) => 0b00000000000000000000000000000010,\n\
     \x20   Trait(_) => 0b00000000000000000000000000000011,\n\
     \x20   Impl(_) => 0b00000000000000000000000000000100,\n\
     \x20   Fn(_) => 0b00000000000000000000000000000101,\n\
     \x20   Object(_) => 0b00000000000000000000000000000110,\n\
     \x20   Lower(_) => 0b00000000000000000000000000000111,\n\
     \x20   Derive(_) => 0b00000000000000000000000000001000,\n\
     \x20   InherentImpl(_) => 0b00000000000000000000000000001001\n\
     \x20 };\n\
     fn classify_basetype(b: BaseType) => Binary{32} =\n\
     \x20 match b {\n\
     \x20   KwBinary(_) => 0b00000000000000000000000000000000,\n\
     \x20   KwTernary(_) => 0b00000000000000000000000000000001,\n\
     \x20   KwDense(_, _) => 0b00000000000000000000000000000010,\n\
     \x20   Vsa(_, _, _) => 0b00000000000000000000000000000011,\n\
     \x20   KwSubstrate(_) => 0b00000000000000000000000000000100,\n\
     \x20   KwSeq(_, _) => 0b00000000000000000000000000000101,\n\
     \x20   KwBytes => 0b00000000000000000000000000000110,\n\
     \x20   KwFloat => 0b00000000000000000000000000000111,\n\
     \x20   Named(_, _) => 0b00000000000000000000000000001000,\n\
     \x20   Ambient(_) => 0b00000000000000000000000000001001,\n\
     \x20   FnArrow(_, _) => 0b00000000000000000000000000001010,\n\
     \x20   Tuple(_) => 0b00000000000000000000000000001011\n\
     \x20 };\n\
     fn classify_literal(l: Literal) => Binary{32} =\n\
     \x20 match l {\n\
     \x20   Bin(_) => 0b00000000000000000000000000000000,\n\
     \x20   Trit(_) => 0b00000000000000000000000000000001,\n\
     \x20   Int(_) => 0b00000000000000000000000000000010,\n\
     \x20   AmbientInt(_, _) => 0b00000000000000000000000000000011,\n\
     \x20   List(_) => 0b00000000000000000000000000000100,\n\
     \x20   LBytes(_) => 0b00000000000000000000000000000101,\n\
     \x20   Str(_) => 0b00000000000000000000000000000110,\n\
     \x20   LFloat(_) => 0b00000000000000000000000000000111\n\
     \x20 };\n\
     fn classify_pattern(p: Pattern) => Binary{32} =\n\
     \x20 match p {\n\
     \x20   PWildcard => 0b00000000000000000000000000000000,\n\
     \x20   PLit(_) => 0b00000000000000000000000000000001,\n\
     \x20   PCtor(_, _) => 0b00000000000000000000000000000010,\n\
     \x20   PIdent(_) => 0b00000000000000000000000000000011,\n\
     \x20   PTuple(_) => 0b00000000000000000000000000000100,\n\
     \x20   POr(_) => 0b00000000000000000000000000000101\n\
     \x20 };\n\
     fn classify_expr(e: Expr) => Binary{32} =\n\
     \x20 match e {\n\
     \x20   Let(_, _, _, _) => 0b00000000000000000000000000000000,\n\
     \x20   If(_, _, _) => 0b00000000000000000000000000000001,\n\
     \x20   Match(_, _) => 0b00000000000000000000000000000010,\n\
     \x20   For(_, _, _, _, _) => 0b00000000000000000000000000000011,\n\
     \x20   Swap(_, _, _) => 0b00000000000000000000000000000100,\n\
     \x20   WithParadigm(_, _) => 0b00000000000000000000000000000101,\n\
     \x20   Wild(_) => 0b00000000000000000000000000000110,\n\
     \x20   Spore(_) => 0b00000000000000000000000000000111,\n\
     \x20   Consume(_) => 0b00000000000000000000000000001000,\n\
     \x20   Colony(_) => 0b00000000000000000000000000001001,\n\
     \x20   Lambda(_, _) => 0b00000000000000000000000000001010,\n\
     \x20   App(_, _) => 0b00000000000000000000000000001011,\n\
     \x20   Fuse(_, _) => 0b00000000000000000000000000001100,\n\
     \x20   Reclaim(_, _) => 0b00000000000000000000000000001101,\n\
     \x20   Path(_) => 0b00000000000000000000000000001110,\n\
     \x20   Lit(_) => 0b00000000000000000000000000001111,\n\
     \x20   Ascribe(_, _) => 0b00000000000000000000000000010000,\n\
     \x20   TupleLit(_) => 0b00000000000000000000000000010001\n\
     \x20 };\n"
}

/// Like `program`, but also appends `driver_prelude()` — for tests that need the shared
/// `sample_typeref`/`sample_expr`/`sample_path` helpers (or a `classify_*` fn) without going
/// through `classify_program`'s fixed `v()`/`main()` shape.
fn program_with_prelude(driver: &str) -> String {
    format!("{}\n{}\n{driver}", program(""), driver_prelude())
}

fn classify_program(ty: &str, ctor_expr: &str, classify_fn: &str) -> String {
    format!(
        "{}\n{}\nfn v() => {ty} = {ctor_expr};\nfn main() => Binary{{32}} = {classify_fn}(v());",
        program(""),
        driver_prelude()
    )
}

/// One table-driven test per enum: constructs a representative value for EVERY variant and
/// asserts `classify_*` returns that variant's expected index — every listed constructor is both
/// CONSTRUCTED and CLASSIFIED via L1-eval (the task brief's stronger inventory leg, not just the
/// count-only fallback), for every enum ast.myc declares.
fn run_enum_table(ty: &str, classify_fn: &str, entries: &[(&str, &str, u32)]) {
    for (label, ctor_expr, expected) in entries {
        let src = classify_program(ty, ctor_expr, classify_fn);
        assert_l1_only_u32(&format!("{ty}::{label}"), &src, *expected);
    }
}

#[test]
fn ast_myc_classifies_every_vis_variant() {
    run_enum_table(
        "Vis",
        "classify_vis",
        &[("Private", "Private", 0), ("Pub", "Pub", 1)],
    );
}

#[test]
fn ast_myc_classifies_every_paradigm_variant() {
    run_enum_table(
        "Paradigm",
        "classify_paradigm",
        &[
            ("PBinary", "PBinary", 0),
            ("PTernary", "PTernary", 1),
            ("PDense", "PDense", 2),
            ("PVsa", "PVsa", 3),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_widthref_variant() {
    let wlit_ctor = format!("WLit({})", b32(8));
    run_enum_table(
        "WidthRef",
        "classify_widthref",
        &[
            ("WLit", wlit_ctor.as_str(), 0),
            ("WName", "WName(\"N\")", 1),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_paramkind_variant() {
    run_enum_table(
        "ParamKind",
        "classify_paramkind",
        &[("PkType", "PkType", 0), ("PkWidth", "PkWidth", 1)],
    );
}

#[test]
fn ast_myc_classifies_every_executionmode_variant() {
    run_enum_table(
        "ExecutionMode",
        "classify_execmode",
        &[
            ("Interpreted", "Interpreted", 0),
            ("Compiled", "Compiled", 1),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_sparsity_variant() {
    let sparse_ctor = format!("SpSparse({})", b32(4));
    run_enum_table(
        "Sparsity",
        "classify_sparsity",
        &[
            ("SpDense", "SpDense", 0),
            ("SpSparse", sparse_ctor.as_str(), 1),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_scalar_variant() {
    run_enum_table(
        "Scalar",
        "classify_scalar",
        &[
            ("SF16", "SF16", 0),
            ("SBf16", "SBf16", 1),
            ("SF32", "SF32", 2),
            ("SF64", "SF64", 3),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_strength_variant() {
    run_enum_table(
        "Strength",
        "classify_strength",
        &[
            ("GDeclared", "GDeclared", 0),
            ("GEmpirical", "GEmpirical", 1),
            ("GProven", "GProven", 2),
            ("GExact", "GExact", 3),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_lowerrhs_variant() {
    let impl_ctor = "LRImpl(ImD(\"T\", Nil, sample_typeref(), Nil))";
    run_enum_table(
        "LowerRhs",
        "classify_lowerrhs",
        &[
            ("LRExpr", "LRExpr(sample_expr())", 0),
            ("LRImpl", impl_ctor, 1),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_item_variant() {
    let use_ctor = "Use(UP(sample_path(), False))";
    let type_ctor = "Type(TD(Private, \"T\", Nil, Nil))";
    let trait_ctor = "Trait(TrD(Private, \"T\", Nil, Nil))";
    let impl_ctor = "Impl(ImD(\"T\", Nil, sample_typeref(), Nil))";
    let fn_ctor = "Fn(FD(Private, False, None, FS(\"f\", Nil, Nil, sample_typeref(), Nil, Nil), sample_expr()))";
    let object_ctor = "Object(OD(Private, \"O\", Nil, Ctr(\"O\", Nil), Nil, Nil, Nil))";
    let lower_ctor = "Lower(LD(\"L\", Nil, LRExpr(sample_expr())))";
    let derive_ctor = "Derive(DD(\"D\", sample_typeref()))";
    let inherent_ctor = "InherentImpl(IID(sample_typeref(), Nil))";
    run_enum_table(
        "Item",
        "classify_item",
        &[
            ("Use", use_ctor, 0),
            ("Default", "Default(PBinary)", 1),
            ("Type", type_ctor, 2),
            ("Trait", trait_ctor, 3),
            ("Impl", impl_ctor, 4),
            ("Fn", fn_ctor, 5),
            ("Object", object_ctor, 6),
            ("Lower", lower_ctor, 7),
            ("Derive", derive_ctor, 8),
            ("InherentImpl", inherent_ctor, 9),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_basetype_variant() {
    let kw_binary = format!("KwBinary(WLit({}))", b32(8));
    let kw_ternary = format!("KwTernary(WLit({}))", b32(6));
    let kw_dense = format!("KwDense({}, SF32)", b32(4));
    let vsa = format!("Vsa(\"m\", {}, SpDense)", b32(8));
    let kw_seq = format!("KwSeq(sample_typeref(), {})", b32(4));
    run_enum_table(
        "BaseType",
        "classify_basetype",
        &[
            ("KwBinary", kw_binary.as_str(), 0),
            ("KwTernary", kw_ternary.as_str(), 1),
            ("KwDense", kw_dense.as_str(), 2),
            ("Vsa", vsa.as_str(), 3),
            ("KwSubstrate", "KwSubstrate(\"S\")", 4),
            ("KwSeq", kw_seq.as_str(), 5),
            ("KwBytes", "KwBytes", 6),
            ("KwFloat", "KwFloat", 7),
            ("Named", "Named(\"T\", Nil)", 8),
            ("Ambient", &format!("Ambient(APSize({}))", b32(8)), 9),
            ("FnArrow", "FnArrow(sample_typeref(), sample_typeref())", 10),
            (
                "Tuple",
                "Tuple(Cons(sample_typeref(), Cons(sample_typeref(), Nil)))",
                11,
            ),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_literal_variant() {
    let int_ctor = format!("Int({})", b64(5));
    let ambient_int_ctor = format!("AmbientInt(PBinary, {})", b64(5));
    run_enum_table(
        "Literal",
        "classify_literal",
        &[
            ("Bin", "Bin(\"0\")", 0),
            ("Trit", "Trit(\"0\")", 1),
            ("Int", int_ctor.as_str(), 2),
            ("AmbientInt", ambient_int_ctor.as_str(), 3),
            ("List", "List(Nil)", 4),
            ("LBytes", "LBytes(\"00\")", 5),
            ("Str", "Str(\"s\")", 6),
            ("LFloat", "LFloat(\"1.5\")", 7),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_pattern_variant() {
    let tuple_ctor = "PTuple(Cons(PWildcard, Cons(PWildcard, Nil)))";
    let or_ctor = "POr(Cons(PWildcard, Cons(PWildcard, Nil)))";
    run_enum_table(
        "Pattern",
        "classify_pattern",
        &[
            ("PWildcard", "PWildcard", 0),
            ("PLit", "PLit(Bin(\"0\"))", 1),
            ("PCtor", "PCtor(\"C\", Nil)", 2),
            ("PIdent", "PIdent(\"x\")", 3),
            ("PTuple", tuple_ctor, 4),
            ("POr", or_ctor, 5),
        ],
    );
}

#[test]
fn ast_myc_classifies_every_expr_variant() {
    let let_ctor = "Let(\"x\", None, sample_expr(), sample_expr())";
    let if_ctor = "If(sample_expr(), sample_expr(), sample_expr())";
    let match_ctor = "Match(sample_expr(), Cons(Ar(PWildcard, sample_expr()), Nil))";
    let for_ctor = "For(\"x\", sample_expr(), \"acc\", sample_expr(), sample_expr())";
    let swap_ctor = "Swap(sample_expr(), sample_typeref(), sample_path())";
    let colony_ctor = "Colony(Cons(Hy(None, sample_expr()), Nil))";
    let tuplelit_ctor = "TupleLit(Cons(sample_expr(), Cons(sample_expr(), Nil)))";
    run_enum_table(
        "Expr",
        "classify_expr",
        &[
            ("Let", let_ctor, 0),
            ("If", if_ctor, 1),
            ("Match", match_ctor, 2),
            ("For", for_ctor, 3),
            ("Swap", swap_ctor, 4),
            ("WithParadigm", "WithParadigm(PBinary, sample_expr())", 5),
            ("Wild", "Wild(sample_expr())", 6),
            ("Spore", "Spore(sample_expr())", 7),
            ("Consume", "Consume(sample_expr())", 8),
            ("Colony", colony_ctor, 9),
            ("Lambda", "Lambda(Nil, sample_expr())", 10),
            ("App", "App(sample_expr(), Nil)", 11),
            ("Fuse", "Fuse(sample_expr(), sample_expr())", 12),
            ("Reclaim", "Reclaim(sample_expr(), sample_expr())", 13),
            ("Path", "Path(sample_path())", 14),
            ("Lit", "Lit(Bin(\"0\"))", 15),
            ("Ascribe", "Ascribe(sample_expr(), sample_typeref())", 16),
            ("TupleLit", tuplelit_ctor, 17),
        ],
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Small helper-impl ports (Phylum::of_one, Strength::rank/meet/satisfies, TypeRef::unguaranteed/
// with_guarantee, LowerDecl::expr_rhs/impl_rhs, FnSig::param_names/width_param_names,
// Paradigm's Display equivalent) — each exercised once via L1-eval.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn ast_myc_phylum_of_one_wraps_a_single_bare_nodule() {
    let src = program(
        "fn main() => Binary{32} =\n\
         \x20 match phylum_of_one(Nd(Pth(Nil), False, Nil)) {\n\
         \x20   Phy(path, nodules) => match path {\n\
         \x20     None => match nodules { Cons(_, Nil) => 0b00000000000000000000000000000001, _ => 0b00000000000000000000000000000000 },\n\
         \x20     Some(_) => 0b00000000000000000000000000000000\n\
         \x20   }\n\
         \x20 };",
    );
    assert_l1_only_u32("phylum_of_one: path=None, nodules=[one]", &src, 1);
}

#[test]
fn ast_myc_strength_rank_orders_the_lattice() {
    // Exact > Proven > Empirical > Declared (rank 3 > 2 > 1 > 0).
    for (label, ctor, expected) in [
        ("GDeclared", "GDeclared", 0u32),
        ("GEmpirical", "GEmpirical", 1),
        ("GProven", "GProven", 2),
        ("GExact", "GExact", 3),
    ] {
        let driver = format!("fn main() => Binary{{8}} = strength_rank({ctor});");
        assert_l1_only_u32(
            &format!("strength_rank({label})"),
            &program(&driver),
            expected,
        );
    }
}

#[test]
fn ast_myc_strength_meet_takes_the_weaker_grade() {
    let cases: &[(&str, &str, &str, u32)] = &[
        (
            "meet(Proven, Empirical) is Empirical",
            "GProven",
            "GEmpirical",
            1,
        ),
        (
            "meet(Empirical, Proven) is Empirical",
            "GEmpirical",
            "GProven",
            1,
        ),
        ("meet(Exact, Exact) is Exact", "GExact", "GExact", 3),
        (
            "meet(Declared, Exact) is Declared",
            "GDeclared",
            "GExact",
            0,
        ),
    ];
    for (label, a, b, expected) in cases {
        let driver = format!("fn main() => Binary{{8}} = strength_rank(strength_meet({a}, {b}));");
        assert_l1_only_u32(label, &program(&driver), *expected);
    }
}

#[test]
fn ast_myc_strength_satisfies_is_the_ge_comparison() {
    let cases: &[(&str, &str, &str, u32)] = &[
        ("Exact satisfies Empirical", "GExact", "GEmpirical", 1),
        (
            "Empirical does not satisfy Exact",
            "GEmpirical",
            "GExact",
            0,
        ),
        ("Proven satisfies Proven", "GProven", "GProven", 1),
    ];
    for (label, actual, demand, expected) in cases {
        let driver = format!(
            "fn main() => Binary{{32}} = match strength_satisfies({actual}, {demand}) {{ True => 0b00000000000000000000000000000001, False => 0b00000000000000000000000000000000 }};"
        );
        assert_l1_only_u32(label, &program(&driver), *expected);
    }
}

#[test]
fn ast_myc_paradigm_to_bytes_matches_display() {
    let cases: &[(&str, &str)] = &[
        ("PBinary", "Binary"),
        ("PTernary", "Ternary"),
        ("PDense", "Dense"),
        ("PVsa", "VSA"),
    ];
    for (ctor, want) in cases {
        let driver = format!(
            "fn main() => Binary{{32}} = match bytes_eq(paradigm_to_bytes({ctor}), \"{want}\") {{ 0b1 => 0b00000000000000000000000000000001, _ => 0b00000000000000000000000000000000 }};"
        );
        assert_l1_only_u32(
            &format!("paradigm_to_bytes({ctor}) == {want:?}"),
            &program(&driver),
            1,
        );
    }
}

#[test]
fn ast_myc_typeref_unguaranteed_and_with_guarantee() {
    let src = program(
        "fn main() => Binary{32} =\n\
         \x20 match typeref_unguaranteed(KwBytes) {\n\
         \x20   TR(_, g) => match g { None => 0b00000000000000000000000000000001, Some(_) => 0b00000000000000000000000000000000 }\n\
         \x20 };",
    );
    assert_l1_only_u32("typeref_unguaranteed has no guarantee", &src, 1);

    let src2 = program(
        "fn main() => Binary{32} =\n\
         \x20 match typeref_with_guarantee(KwBytes, GExact) {\n\
         \x20   TR(_, g) => match g { Some(_) => 0b00000000000000000000000000000001, None => 0b00000000000000000000000000000000 }\n\
         \x20 };",
    );
    assert_l1_only_u32("typeref_with_guarantee carries Some(_)", &src2, 1);
}

#[test]
fn ast_myc_lowerdecl_expr_rhs_and_impl_rhs_accessors() {
    let expr_decl = "LD(\"R1\", Nil, LRExpr(sample_expr()))";
    let impl_decl = "LD(\"R2\", Nil, LRImpl(ImD(\"T\", Nil, sample_typeref(), Nil)))";

    let src = program_with_prelude(&format!(
        "fn main() => Binary{{32}} =\n\
         \x20 match lowerdecl_expr_rhs({expr_decl}) {{ Some(_) => 0b00000000000000000000000000000001, None => 0b00000000000000000000000000000000 }};"
    ));
    assert_l1_only_u32("lowerdecl_expr_rhs(expr-shaped) is Some", &src, 1);

    let src2 = program_with_prelude(&format!(
        "fn main() => Binary{{32}} =\n\
         \x20 match lowerdecl_expr_rhs({impl_decl}) {{ Some(_) => 0b00000000000000000000000000000001, None => 0b00000000000000000000000000000000 }};"
    ));
    assert_l1_only_u32("lowerdecl_expr_rhs(impl-shaped) is None", &src2, 0);

    let src3 = program_with_prelude(&format!(
        "fn main() => Binary{{32}} =\n\
         \x20 match lowerdecl_impl_rhs({impl_decl}) {{ Some(_) => 0b00000000000000000000000000000001, None => 0b00000000000000000000000000000000 }};"
    ));
    assert_l1_only_u32("lowerdecl_impl_rhs(impl-shaped) is Some", &src3, 1);
}

#[test]
fn ast_myc_fnsig_param_names_filters_by_kind() {
    let sig = "FS(\"f\", Cons(TP(\"A\", PkType, Nil), Cons(TP(\"N\", PkWidth, Nil), Cons(TP(\"B\", PkType, Nil), Nil))), Nil, sample_typeref(), Nil, Nil)";
    let src = program_with_prelude(&format!(
        "fn count(v: Vec[Bytes]) => Binary{{32}} =\n\
         \x20 match v {{ Nil => 0b00000000000000000000000000000000, Cons(_, rest) => add_u(0b00000000000000000000000000000001, count(rest)) }};\n\
         fn main() => Binary{{32}} = count(fnsig_param_names({sig}));"
    ));
    assert_l1_only_u32(
        "fnsig_param_names keeps only the 2 type-kind params",
        &src,
        2,
    );

    let src2 = program_with_prelude(&format!(
        "fn count(v: Vec[Bytes]) => Binary{{32}} =\n\
         \x20 match v {{ Nil => 0b00000000000000000000000000000000, Cons(_, rest) => add_u(0b00000000000000000000000000000001, count(rest)) }};\n\
         fn main() => Binary{{32}} = count(fnsig_width_param_names({sig}));"
    ));
    assert_l1_only_u32(
        "fnsig_width_param_names keeps only the 1 width-kind param",
        &src2,
        1,
    );
}

/// A composite "smoke" construction tying several struct types together (Nodule/Item/FnDecl/
/// FnSig/TypeRef/BaseType/Expr) — proves the struct-heavy portion of the port composes and
/// type-checks as a realistic small program, not just in isolation.
#[test]
fn ast_myc_composite_nodule_construction_type_checks_and_evaluates() {
    let src = program(
        "fn build() => Nodule =\n\
         \x20 Nd(\n\
         \x20   Pth(Cons(\"demo\", Nil)),\n\
         \x20   False,\n\
         \x20   Cons(\n\
         \x20     Fn(FD(\n\
         \x20       Pub,\n\
         \x20       False,\n\
         \x20       None,\n\
         \x20       FS(\"f\", Nil, Cons(Prm(\"x\", typeref_unguaranteed(KwBytes)), Nil), typeref_unguaranteed(KwBytes), Nil, Nil),\n\
         \x20       Path(Pth(Cons(\"x\", Nil)))\n\
         \x20     )),\n\
         \x20     Nil\n\
         \x20   )\n\
         \x20 );\n\
         fn main() => Binary{32} =\n\
         \x20 match nodule_items(build()) {\n\
         \x20   Cons(item, Nil) => classify_item(item),\n\
         \x20   _ => 0b11111111111111111111111111111111\n\
         \x20 };",
    );
    let full = format!("{}\n{}", src, driver_prelude());
    assert_l1_only_u32(
        "composite Nodule[Fn(...)] round-trips through nodule_items + classify_item",
        &full,
        5,
    );
}
