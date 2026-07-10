//! M-740 Stage 5, increment 4 (M-1009; DN-26 §7.3 row 5 / §9 flag-1) — the self-hosted
//! `compiler.semcore` monomorphization name-mangling family: the LIVE-ORACLE differential gate for
//! mono.rs's pure `Ty`→`Bytes` mangling leaves ported into `lib/compiler/semcore.myc`:
//!   * `mangle_ty` / `scalar_tag` (mono.rs) — a `Ty` to a flat identifier-suffix fragment.
//!   * `mangle_decl` / `mangle_ctor` (mono.rs) — a decl/ctor name at concrete type arguments.
//!   * `mangle_method` (mono.rs) — a trait method to `method$Trait$ForTy`.
//!   * `mangle_arrow` / `mangle_ty_or_fn` / `apply_fn_name` (mono.rs, RFC-0024 §4A) — closure-arrow
//!     tag-sum naming + the dispatcher name.
//!   * `mangle_decl_with_wargs` / `mangle_hof_decl` (mono.rs, DN-42/M-753 + RFC-0024 §4) — the
//!     width-argument + HOF-specialization mangles (`scalar_tag`/`mangle_decl_with_wargs` are the
//!     module-private members, exercised transitively through the reachable `pub(crate)` entries).
//!
//! **Live-oracle posture (VR-5).** Every case calls the REAL Rust `mono::mangle_*` on a fixture and
//! asserts the `.myc` port's output equals it via a direct `bytes_eq` (the cleanest witness kind —
//! no equality scaffolding needed). The injectivity boundary is exercised: the nullary-`Data` `#`
//! tag vs. the repr mangle, and the `$`/`%`/`~` joints. All exercised fns are `pub(crate)`,
//! reachable from this in-crate `src/tests/` module (no visibility change to `mono.rs`; only this
//! module + its one `mod` line were added). `mangle_ty_in_ty`/`item_key` are DEFERRED
//! (FLAG-semcore-17: module-private, no reachable oracle — not ported rather than land un-witnessed).
//!
//! M-981 applies: only the L1-eval leg is exercised (small synthetic fixtures, not a corpus
//! program — the L0/AOT leg's marginal value is low relative to its eval-depth cost, M-987).

use crate::ast::{Scalar, Sparsity};
use crate::checkty::{check_nodule, Ty, Width};
use crate::elab::build_registry;
use crate::eval::Evaluator;
use crate::mono::{
    apply_fn_name, mangle_arrow, mangle_ctor, mangle_decl, mangle_hof_decl, mangle_method,
    mangle_ty, mangle_ty_or_fn, monomorphize,
};
use crate::parse;
use mycelium_core::Payload;

/// Extract a `Binary{N}` `CoreValue`'s bits as a `u32` (MSB-first) — the established convention.
fn core_bits_as_u32(v: &mycelium_core::CoreValue) -> u32 {
    let repr_val = v
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr CoreValue, got {v:?}"));
    match repr_val.payload() {
        Payload::Bits(bits) => bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b)),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

const SEMCORE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/semcore.myc"
));

fn program(driver: &str) -> String {
    format!("{SEMCORE_SRC}\n{driver}")
}

/// L1-eval-only assertion (the M-981 convention): parse → check → monomorphize → build_registry →
/// eval `main` → compare the `Binary{32}` result to the LIVE-ORACLE-derived `expected_u32`.
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
    let got = core_bits_as_u32(&l1_core);
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the LIVE-ORACLE expected value {expected_u32}"
    );
}

// ── Rust → `.myc` fixture encoders ────────────────────────────────────────────────────────────────

fn encode_u32(n: u32) -> String {
    let mut s = String::from("0b");
    for (count, i) in (0..32).rev().enumerate() {
        if count != 0 && count % 4 == 0 {
            s.push('_');
        }
        s.push(if (n >> i) & 1 == 1 { '1' } else { '0' });
    }
    s
}

fn encode_bytes(s: &str) -> String {
    format!("{s:?}")
}

fn encode_scalar(s: Scalar) -> &'static str {
    match s {
        Scalar::F16 => "SF16",
        Scalar::Bf16 => "SBf16",
        Scalar::F32 => "SF32",
        Scalar::F64 => "SF64",
    }
}

fn encode_sparsity(sp: &Sparsity) -> String {
    match sp {
        Sparsity::Dense => "SpDense".to_owned(),
        Sparsity::Sparse(k) => format!("SpSparse({})", encode_u32(*k)),
    }
}

fn encode_width(w: &Width) -> String {
    match w {
        Width::Lit(n) => format!("WdLit({})", encode_u32(*n)),
        Width::Var(v) => format!("WdVar({})", encode_bytes(v)),
    }
}

fn encode_ty(t: &Ty) -> String {
    match t {
        Ty::Binary(w) => format!("TyBinary({})", encode_width(w)),
        Ty::Ternary(w) => format!("TyTernary({})", encode_width(w)),
        Ty::Dense(d, s) => format!("TyDense({}, {})", encode_u32(*d), encode_scalar(*s)),
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => format!(
            "TyVsa({}, {}, {})",
            encode_bytes(model),
            encode_u32(*dim),
            encode_sparsity(sparsity)
        ),
        Ty::Data(n, args) => format!("TyData({}, {})", encode_bytes(n), encode_ty_list(args)),
        Ty::Substrate(t) => format!("TySubstrate({})", encode_bytes(t)),
        Ty::Seq(elem, n) => format!("TySeq({}, {})", encode_ty(elem), encode_u32(*n)),
        Ty::Bytes => "TyBytes".to_owned(),
        Ty::Float => "TyFloat".to_owned(),
        Ty::Var(v) => format!("TyVar({})", encode_bytes(v)),
        Ty::Fn(a, r) => format!("TyFn({}, {})", encode_ty(a), encode_ty(r)),
    }
}

fn encode_ty_list(ts: &[Ty]) -> String {
    let mut s = String::from("Nil");
    for t in ts.iter().rev() {
        s = format!("Cons({}, {})", encode_ty(t), s);
    }
    s
}

fn encode_width_list(ws: &[Width]) -> String {
    let mut s = String::from("Nil");
    for w in ws.iter().rev() {
        s = format!("Cons({}, {})", encode_width(w), s);
    }
    s
}

fn encode_pairs(ps: &[(usize, String)]) -> String {
    let mut s = String::from("Nil");
    for (idx, name) in ps.iter().rev() {
        s = format!(
            "Cons(Pr({}, {}), {})",
            encode_u32(u32::try_from(*idx).expect("index fits u32")),
            encode_bytes(name),
            s
        );
    }
    s
}

// Small fixture constructors keeping the test bodies to `assert over a case`.
fn data(n: &str, args: Vec<Ty>) -> Ty {
    Ty::Data(n.to_owned(), args)
}
fn var(n: &str) -> Ty {
    Ty::Var(n.to_owned())
}
fn bin(n: u32) -> Ty {
    Ty::Binary(Width::Lit(n))
}

/// A `bytes_eq` witness: assert the `.myc` mangle expression's output equals the oracle `want`.
fn assert_mangle(label: &str, myc_expr: &str, want: &str) {
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match bytes_eq({}, {}) {{ 0b1 => one32(), _ => zero32() }};\n",
        myc_expr,
        encode_bytes(want)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Structural gate: `semcore.myc` (with the increment-4 additions) parses and type-checks green.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn semcore_mangle_parses_and_checks() {
    let nodule = parse(SEMCORE_SRC).unwrap_or_else(|e| panic!("semcore.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("semcore.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Non-vacuity probe: a `bytes_eq` witness against the WRONG expected string yields 0.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mangle_witness_discriminates() {
    // `mangle_ty(Binary{8})` = "Binary8"; comparing against the wrong "Binary16" must yield 0.
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match bytes_eq(mangle_ty({}), {}) {{ 0b1 => one32(), _ => zero32() }};\n",
        encode_ty(&bin(8)),
        encode_bytes("Binary16")
    );
    assert_l1_only_u32("wrong_mangle_yields_zero", &program(&driver), 0);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// mangle_ty (LIVE — mono::mangle_ty): every Ty kind, incl. the injectivity boundary (`#` nullary tag
// vs. repr mangle) and the VSA `-`→`_` model canonicalization.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mangle_ty_cases() {
    let cases: Vec<Ty> = vec![
        bin(8),
        Ty::Binary(Width::Var("N".to_owned())),
        Ty::Ternary(Width::Lit(6)),
        Ty::Ternary(Width::Var("M".to_owned())),
        Ty::Dense(1024, Scalar::F32),
        Ty::Dense(16, Scalar::Bf16),
        Ty::Vsa {
            model: "MAP-I".to_owned(),
            dim: 256,
            sparsity: Sparsity::Dense,
        },
        Ty::Vsa {
            model: "FHRR".to_owned(),
            dim: 512,
            sparsity: Sparsity::Sparse(8),
        },
        Ty::Substrate("file".to_owned()),
        Ty::Seq(Box::new(bin(8)), 4),
        Ty::Seq(Box::new(data("List", vec![bin(8)])), 2),
        Ty::Bytes,
        Ty::Float,
        // Injectivity boundary: a nullary data type named `Binary8` mangles to `Binary8#`, never
        // colliding with the repr `Binary{8}` → `Binary8`.
        data("Binary8", vec![]),
        data("Bool", vec![]),
        data("List", vec![bin(8)]),
        data("Map", vec![Ty::Bytes, data("List", vec![bin(16)])]),
        var("A"),
        Ty::Fn(Box::new(bin(8)), Box::new(Ty::Bytes)),
    ];
    for (i, t) in cases.iter().enumerate() {
        let want = mangle_ty(t);
        assert_mangle(
            &format!("mangle_ty_{i}"),
            &format!("mangle_ty({})", encode_ty(t)),
            &want,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// mangle_decl / mangle_ctor (LIVE): empty targs → name unchanged; non-empty → `$`-joined args.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mangle_decl_ctor_cases() {
    let cases: Vec<(&str, Vec<Ty>)> = vec![
        ("add", vec![]),
        ("map", vec![bin(8)]),
        ("zip", vec![bin(8), Ty::Bytes]),
        ("wrap", vec![data("List", vec![bin(16)])]),
    ];
    for (i, (name, targs)) in cases.iter().enumerate() {
        let want_decl = mangle_decl(name, targs);
        assert_mangle(
            &format!("mangle_decl_{i}"),
            &format!(
                "mangle_decl({}, {})",
                encode_bytes(name),
                encode_ty_list(targs)
            ),
            &want_decl,
        );
        let want_ctor = mangle_ctor(name, targs);
        assert_mangle(
            &format!("mangle_ctor_{i}"),
            &format!(
                "mangle_ctor({}, {})",
                encode_bytes(name),
                encode_ty_list(targs)
            ),
            &want_ctor,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// mangle_method (LIVE): `method$Trait$ForTy`.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mangle_method_cases() {
    let cases: Vec<(&str, &str, Ty)> = vec![
        ("cmp", "Cmp", bin(8)),
        ("show", "Show", data("List", vec![bin(8)])),
        ("eq", "Eq", Ty::Bytes),
    ];
    for (i, (method, trait_name, for_ty)) in cases.iter().enumerate() {
        let want = mangle_method(method, trait_name, for_ty);
        assert_mangle(
            &format!("mangle_method_{i}"),
            &format!(
                "mangle_method({}, {}, {})",
                encode_bytes(method),
                encode_bytes(trait_name),
                encode_ty(for_ty)
            ),
            &want,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// mangle_arrow / mangle_ty_or_fn / apply_fn_name (LIVE): the closure-arrow tag-sum + dispatcher name,
// incl. a nested arrow (recurses through mangle_ty_or_fn).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mangle_arrow_cases() {
    let arrows: Vec<(Ty, Ty)> = vec![
        (bin(8), Ty::Bytes),
        (data("List", vec![bin(8)]), Ty::Float),
        // A nested arrow codomain: `A => (B => C)`.
        (bin(8), Ty::Fn(Box::new(Ty::Bytes), Box::new(Ty::Float))),
    ];
    for (i, (a, b)) in arrows.iter().enumerate() {
        let want_arrow = mangle_arrow(a, b);
        assert_mangle(
            &format!("mangle_arrow_{i}"),
            &format!("mangle_arrow({}, {})", encode_ty(a), encode_ty(b)),
            &want_arrow,
        );
        // apply_fn_name strips the `Fn$` prefix.
        let want_apply = apply_fn_name(&want_arrow);
        assert_mangle(
            &format!("apply_fn_name_{i}"),
            &format!("apply_fn_name({})", encode_bytes(&want_arrow)),
            &want_apply,
        );
    }
    // mangle_ty_or_fn: a plain (non-Fn) Ty delegates to mangle_ty; a Fn goes to the arrow tag-sum.
    let plain = data("List", vec![bin(8)]);
    assert_mangle(
        "mangle_ty_or_fn_plain",
        &format!("mangle_ty_or_fn({})", encode_ty(&plain)),
        &mangle_ty_or_fn(&plain),
    );
    let arrow = Ty::Fn(Box::new(bin(8)), Box::new(Ty::Bytes));
    assert_mangle(
        "mangle_ty_or_fn_arrow",
        &format!("mangle_ty_or_fn({})", encode_ty(&arrow)),
        &mangle_ty_or_fn(&arrow),
    );
    // apply_fn_name on a string WITHOUT the `Fn$` prefix uses the whole string (strip_prefix noop).
    assert_mangle(
        "apply_fn_name_no_prefix",
        &format!("apply_fn_name({})", encode_bytes("Bytes")),
        &apply_fn_name("Bytes"),
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// mangle_hof_decl (LIVE — mono::mangle_hof_decl, exercising the private mangle_decl_with_wargs +
// scalar_tag transitively): width args (`$Binary{n}`), the `%` static-fn joint, the `~` dynamic joint.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mangle_hof_decl_cases() {
    // Empty fn_args + dyn_fns → delegates to mangle_decl_with_wargs (type args + width args only).
    let hof_expr = |name: &str,
                    targs: &[Ty],
                    wargs: &[Width],
                    fa: &[(usize, String)],
                    df: &[(usize, String)]| {
        format!(
            "mangle_hof_decl({}, {}, {}, {}, {})",
            encode_bytes(name),
            encode_ty_list(targs),
            encode_width_list(wargs),
            encode_pairs(fa),
            encode_pairs(df),
        )
    };

    // (a) width args only.
    let want = mangle_hof_decl("add", &[], &[Width::Lit(8)], &[], &[]);
    assert_mangle(
        "hof_wargs",
        &hof_expr("add", &[], &[Width::Lit(8)], &[], &[]),
        &want,
    );

    // (b) type + width args.
    let want = mangle_hof_decl("op", &[bin(8)], &[Width::Lit(16)], &[], &[]);
    assert_mangle(
        "hof_targs_wargs",
        &hof_expr("op", &[bin(8)], &[Width::Lit(16)], &[], &[]),
        &want,
    );

    // (c) static fn arguments (the `%` joint).
    let fa = vec![(0usize, "callee".to_owned()), (2usize, "other".to_owned())];
    let want = mangle_hof_decl("hof", &[bin(8)], &[], &fa, &[]);
    assert_mangle(
        "hof_fn_args",
        &hof_expr("hof", &[bin(8)], &[], &fa, &[]),
        &want,
    );

    // (d) dynamic fn arguments (the `~` joint).
    let df = vec![(1usize, "Fn$Binary8$Bytes".to_owned())];
    let want = mangle_hof_decl("dyn_hof", &[], &[], &[], &df);
    assert_mangle(
        "hof_dyn_fns",
        &hof_expr("dyn_hof", &[], &[], &[], &df),
        &want,
    );

    // (e) both static and dynamic fn arguments (both joints, order preserved).
    let fa = vec![(0usize, "s".to_owned())];
    let df = vec![(3usize, "Fn$Bytes$Float".to_owned())];
    let want = mangle_hof_decl("both", &[bin(8)], &[Width::Lit(4)], &fa, &df);
    assert_mangle(
        "hof_both",
        &hof_expr("both", &[bin(8)], &[Width::Lit(4)], &fa, &df),
        &want,
    );
}
