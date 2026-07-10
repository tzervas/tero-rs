//! Unit tests for `crate::prim_map` (trx2 Lane C Deliverable 2) — both the emitted-text shape
//! (fixture corpus, data-driven per CLAUDE.md "Complex test logic lives in fixtures +
//! parameterization") and, for the `wired: true` rows, a live-oracle proof against the real
//! `myc-check` toolchain (mirrors `src/tests/emit.rs`'s `binop_operand_gated_forms_check_clean`).

use super::vet::find_myc_check;
use crate::gap::Category;
use crate::transpile::transpile_source;

/// WIRED: a receiver known `Float` (via the `f64` parameter, itself mapped by this leaf's
/// `map_type` fix) triggers the real `flt_is_nan`/`flt_is_finite`/`flt_is_infinite` prim call,
/// bridged `Binary{1}` -> `Bool` (Rust's `f64::is_nan`/… always return `bool`).
#[test]
fn wired_float_classification_methods_emit_bridged_prim_calls() {
    let cases = [
        (
            "fn f(x: f64) -> bool { x.is_nan() }",
            "(match flt_is_nan(x) { 0b1 => True, _ => False })",
        ),
        (
            "fn f(x: f64) -> bool { x.is_finite() }",
            "(match flt_is_finite(x) { 0b1 => True, _ => False })",
        ),
        (
            "fn f(x: f64) -> bool { x.is_infinite() }",
            "(match flt_is_infinite(x) { 0b1 => True, _ => False })",
        ),
    ];
    for (rust, needle) in cases {
        let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
            .unwrap_or_else(|e| panic!("failed to parse/transpile `{rust}`: {e}"));
        assert!(
            report.emitted_items.iter().any(|n| n == "f"),
            "case `{rust}`: expected `f` in emitted_items, got {:?} (gaps={:?})",
            report.emitted_items,
            report.gaps
        );
        assert!(
            myc.contains(needle),
            "case `{rust}`: expected emitted text to contain `{needle}`, got:\n{myc}"
        );
        // The `f64` parameter itself must map to the grammar's real `Float` base_type (this
        // leaf's `map_type` fix) — otherwise the receiver-type gate could never have fired.
        assert!(
            myc.contains("fn f(x: Float)"),
            "case `{rust}`: expected the `f64` param to map to `Float`, got:\n{myc}"
        );
    }
}

/// NOT gated: an `.is_nan()`-named method on a receiver whose type is NOT known to be `Float`
/// (here, an ordinary passed-through named type) must NOT trigger the bridged prim rewrite — the
/// receiver-type gate exists precisely to prevent a coincidentally-same-named method on an
/// unrelated type from being mistranslated (VR-5: never guess the receiver's type). Falls through
/// to the unchanged generic `recv.method(args)` -> `method(recv, args)` desugar.
#[test]
fn is_nan_on_unknown_receiver_type_keeps_generic_desugar() {
    let rust = "fn f(x: Thing) -> bool { x.is_nan() }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.iter().any(|n| n == "f"),
        "expected `f` in emitted_items, got {:?} (gaps={:?})",
        report.emitted_items,
        report.gaps
    );
    assert!(
        myc.contains("is_nan(x)") && !myc.contains("flt_is_nan"),
        "expected the OLD generic bare-call desugar (`is_nan(x)`), not the bridged `flt_is_nan` \
         rewrite, since `Thing` is not a known `Float` receiver — got:\n{myc}"
    );
}

/// PENDING-BACKEND: `.wrapping_add()`/`.wrapping_sub()`/`.wrapping_mul()` on a receiver known to
/// be some concrete `Binary{N}` are recognized (CU-5, RFC-0034 §10/M-791 — the named `wrapping`
/// construct is a decided ruling with no grammar surface or runtime path yet) but ALWAYS refuse —
/// never emitted, per the PENDING-BACKEND contract (VR-5/G2).
#[test]
fn wrapping_methods_on_known_binary_are_pending_backend_gaps() {
    let cases = [
        "fn f(a: u16, b: u16) -> u16 { a.wrapping_add(b) }",
        "fn f(a: u16, b: u16) -> u16 { a.wrapping_sub(b) }",
        "fn f(a: u16, b: u16) -> u16 { a.wrapping_mul(b) }",
    ];
    for rust in cases {
        let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
            .unwrap_or_else(|e| panic!("failed to parse/transpile `{rust}`: {e}"));
        assert!(
            !report.emitted_items.iter().any(|n| n == "f"),
            "case `{rust}`: expected NO emission for a PENDING-BACKEND row, got emitted_items={:?}",
            report.emitted_items
        );
        assert!(
            !myc.contains("wrapping"),
            "case `{rust}`: a PENDING-BACKEND row must never leak emitted text, got:\n{myc}"
        );
        assert!(
            report
                .gaps
                .iter()
                .any(|g| g.category == Category::Conversion
                    && g.reason.contains("PENDING-BACKEND(CU-5)")),
            "case `{rust}`: expected a Category::Conversion gap citing PENDING-BACKEND(CU-5), got \
             {:?}",
            report
                .gaps
                .iter()
                .map(|g| (g.category.as_str(), g.reason.as_str()))
                .collect::<Vec<_>>()
        );
    }
}

/// NOT gated: `.wrapping_add()` on a receiver NOT known to be a concrete `Binary{N}` (here, an
/// unrelated passed-through type) does not fire the PENDING-BACKEND gap either — same
/// receiver-type-gate discipline as the `is_nan` case above, applied to the `AnyBinaryWidth` gate.
#[test]
fn wrapping_add_on_unknown_receiver_type_keeps_generic_desugar() {
    let rust = "fn f(x: Thing, y: Thing) -> Thing { x.wrapping_add(y) }";
    let (myc, report) = transpile_source(rust, "fixture.rs", "fixture")
        .unwrap_or_else(|e| panic!("failed to parse/transpile: {e}"));
    assert!(
        report.emitted_items.iter().any(|n| n == "f"),
        "expected `f` in emitted_items (the generic desugar still emits SOME text), got {:?} \
         (gaps={:?})",
        report.emitted_items,
        report.gaps
    );
    assert!(
        myc.contains("wrapping_add(x, y)"),
        "expected the OLD generic bare-call desugar (`wrapping_add(x, y)`), not a PENDING-BACKEND \
         gap, since `Thing` is not a known `Binary{{N}}` receiver — got:\n{myc}"
    );
}

/// **The verify-first proof** (mitigation #14) for the WIRED rows: every bridged
/// `flt_is_nan`/`flt_is_finite`/`flt_is_infinite` emission is run through the REAL `myc-check`
/// oracle, proving the text actually type-checks with zero imports (not just a substring match).
/// Skips gracefully (never fails) when `myc-check` is not built.
#[test]
fn wired_methods_check_clean_against_real_toolchain() {
    let Some(bin) = find_myc_check() else {
        eprintln!(
            "prim_map: live oracle test skipped — no runnable myc-check (set MYC_CHECK_CMD or \
             build `cargo build -p mycelium-check --bin myc-check`). The fixture-corpus text \
             assertions above still cover the emitted shape."
        );
        return;
    };

    let dir = std::env::temp_dir().join(format!(
        "mycelium-transpile-prim-map-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    let rust_snippets = [
        "fn f_is_nan(x: f64) -> bool { x.is_nan() }",
        "fn f_is_finite(x: f64) -> bool { x.is_finite() }",
        "fn f_is_infinite(x: f64) -> bool { x.is_infinite() }",
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
