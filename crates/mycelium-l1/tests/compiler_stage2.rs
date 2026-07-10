//! M-740 Stage 2 (DN-26 §7.3 row 2) — the self-hosted `compiler.nodule` port.
//!
//! Header-parse differential: `lib/compiler/nodule.myc`'s `parse_nodule_header` vs the live Rust
//! oracle (`mycelium_l1::parse_nodule_header`, `crates/mycelium-l1/src/nodule.rs`) over (a) a
//! synthetic edge battery transcribed from the oracle's own unit tests
//! (`src/tests/nodule.rs`), and (b) every real `.myc` file in the L1 conformance corpus
//! (accept + reject) and under `lib/` — the DN-26 §7.3 Stage-2 gate ("header-parse differential").
//!
//! Comparison shape per input: a 4-way classification code (0 = no marker, 1 = bare marker,
//! 2 = named marker, 3 = explicit error), plus the joined dotted name for named markers and the
//! 1-based error line for errors. Message CONTENT is deliberately not compared
//! (nodule.myc FLAG-nodule-3: static messages; the oracle interpolates).

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::{CoreValue, GuaranteeStrength, Payload};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// Extract a `Binary{N}` `CoreValue`'s bits as a `u32` (MSB-first), ignoring `Meta`/provenance
/// (same helper as `compiler_stage1.rs`).
fn core_bits_as_u32(v: &CoreValue) -> u32 {
    let repr_val = v
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr CoreValue, got {v:?}"));
    match repr_val.payload() {
        Payload::Bits(bits) => bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b)),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

const NODULE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/nodule.myc"
));

/// The shared driver prelude: the classification helpers every test's `main` calls. Each helper
/// unwraps ONE constructor per `match` (the split-match idiom — `compiler_stage1.rs`
/// FLAG-stage1-checker-1 / M-980), never a combined multi-level pattern.
fn driver_prelude() -> String {
    format!(
        "fn ph_code(src: Bytes) => Binary{{32}} =\n\
         \x20 match parse_nodule_header(src) {{\n\
         \x20   Err(_) => {three},\n\
         \x20   Ok(o) => match o {{\n\
         \x20     None => {zero},\n\
         \x20     Some(h) => match nh_name(h) {{ None => {one}, Some(_) => {two} }}\n\
         \x20   }}\n\
         \x20 }};\n\
         fn ph_err_line(src: Bytes) => Binary{{32}} =\n\
         \x20 match parse_nodule_header(src) {{ Err(e) => nhe_line(e), Ok(_) => {zero} }};\n\
         fn ph_dotted_is(src: Bytes, want: Bytes) => Binary{{32}} =\n\
         \x20 match parse_nodule_header(src) {{\n\
         \x20   Err(_) => {zero},\n\
         \x20   Ok(o) => match o {{\n\
         \x20     None => {zero},\n\
         \x20     Some(h) => match dotted(h) {{\n\
         \x20       None => {zero},\n\
         \x20       Some(d) => match bytes_eq(d, want) {{ 0b1 => {one}, _ => {zero} }}\n\
         \x20     }}\n\
         \x20   }}\n\
         \x20 }};\n\
         fn ph_canonical_is(src: Bytes, want: Bytes) => Binary{{32}} =\n\
         \x20 match parse_nodule_header(src) {{\n\
         \x20   Err(_) => {zero},\n\
         \x20   Ok(o) => match o {{\n\
         \x20     None => {zero},\n\
         \x20     Some(h) => match bytes_eq(canonical(h), want) {{ 0b1 => {one}, _ => {zero} }}\n\
         \x20   }}\n\
         \x20 }};\n",
        zero = b32(0),
        one = b32(1),
        two = b32(2),
        three = b32(3),
    )
}

fn program(driver: &str) -> String {
    format!("{NODULE_SRC}\n{}\n{driver}", driver_prelude())
}

/// L1-eval-only assertion (same helper + rationale as `compiler_stage1.rs`): the L0-interp leg is
/// impractical at self-hosted-compiler scale (M-981), so the per-input differential here compares
/// the Rust oracle against the L1-eval leg — a complete "Rust recogniser ≡ self-hosted recogniser"
/// comparison (DN-26 §7.3's Stage-2 requirement). The three-way legs are still exercised once by
/// [`nodule_myc_three_way_smoke`] below, which is small enough to run them.
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
        "{label}: L1-eval result {got} does not match the Rust-oracle-computed expected value {expected_u32}"
    );
}

/// Full three-way assertion (L1-eval ≡ L0-interp ≡ AOT, then vs an expected reference program) —
/// the `std_*.rs` harness shape, run once on a small input to keep the L0/AOT legs exercised for
/// this stage (everything else is L1-only per M-981).
fn assert_three_way_u32(label: &str, src: &str, expected_u32: u32) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

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

    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: elaborate failed: {e}"));
    let l0_core = interp
        .eval_core(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
        .unwrap_or_else(|e| panic!("{label}: AOT run_core failed: {e}"));

    assert_eq!(
        l1_core, l0_core,
        "{label}: L1-eval(mono) vs elaborate->L0-interp diverged"
    );
    assert_eq!(l0_core, aot_core, "{label}: L0-interp vs AOT diverged");
    for (x, y, pair) in [
        (&l1_core, &l0_core, "L1<->interp"),
        (&l0_core, &aot_core, "interp<->AOT"),
    ] {
        assert_eq!(
            check_core(x, y),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "{label}: {pair} differential must validate Exact"
        );
    }
    let got = core_bits_as_u32(&l1_core);
    assert_eq!(
        got, expected_u32,
        "{label}: three-way-agreed result {got} does not match the Rust-oracle-computed expected value {expected_u32}"
    );
}

/// Format `n` as an explicit `Binary{32}` literal (same rationale as `compiler_stage1.rs::b32`:
/// bare decimal literals do not ambient-resolve in every position).
fn b32(n: u32) -> String {
    format!("0b{n:032b}")
}

fn escape_myc_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// The Rust oracle's classification, mapped to the same 4-way code the `.myc` driver computes.
fn oracle_code(src: &str) -> u32 {
    match mycelium_l1::parse_nodule_header(src) {
        Err(_) => 3,
        Ok(None) => 0,
        Ok(Some(h)) => {
            if h.name.is_none() {
                1
            } else {
                2
            }
        }
    }
}

/// Assert the `.myc` port fully agrees with the live Rust oracle on `input`: the classification
/// code, plus the dotted name (named case) or the 1-based error line (error case).
fn assert_header_differential(label: &str, input: &str) {
    let esc = escape_myc_string(input);
    let expected = oracle_code(input);
    let driver = format!("fn main() => Binary{{32}} = ph_code(\"{esc}\");");
    assert_l1_only_u32(
        &format!("{label}: classification code"),
        &program(&driver),
        expected,
    );

    match mycelium_l1::parse_nodule_header(input) {
        Ok(Some(h)) => {
            if let Some(dotted) = h.dotted() {
                let driver = format!(
                    "fn main() => Binary{{32}} = ph_dotted_is(\"{esc}\", \"{}\");",
                    escape_myc_string(&dotted)
                );
                assert_l1_only_u32(
                    &format!("{label}: dotted name is {dotted:?}"),
                    &program(&driver),
                    1,
                );
            }
            let canonical = h.canonical();
            let driver = format!(
                "fn main() => Binary{{32}} = ph_canonical_is(\"{esc}\", \"{}\");",
                escape_myc_string(&canonical)
            );
            assert_l1_only_u32(
                &format!("{label}: canonical spelling is {canonical:?}"),
                &program(&driver),
                1,
            );
        }
        Err(e) => {
            let driver = format!("fn main() => Binary{{32}} = ph_err_line(\"{esc}\");");
            assert_l1_only_u32(
                &format!("{label}: error line is {}", e.line),
                &program(&driver),
                e.line,
            );
        }
        Ok(None) => {}
    }
}

/// One small three-way run (L1 ≡ L0 ≡ AOT) so Stage 2 keeps the full differential legs exercised
/// where scale permits (M-981 narrows the corpus sweep to L1-only, as in Stage 1).
#[test]
fn nodule_myc_three_way_smoke() {
    let input = "// nodule: geometry.shapes\nnodule geometry.shapes;\n";
    assert_eq!(oracle_code(input), 2, "oracle sanity");
    let driver = format!(
        "fn main() => Binary{{32}} = ph_code(\"{}\");",
        escape_myc_string(input)
    );
    assert_three_way_u32("named marker three-way", &program(&driver), 2);
}

/// The synthetic edge battery — every case from the oracle's own unit tests
/// (`src/tests/nodule.rs`) plus the near-miss/whitespace/CRLF edges, each differentialed against
/// the LIVE oracle (never a hand-assumed expectation on the `.myc` side).
#[test]
fn nodule_myc_matches_oracle_on_the_edge_battery() {
    let inputs: &[(&str, &str)] = &[
        // named / bare / skipping — the oracle's positive cases
        (
            "named marker",
            "// nodule: geometry.shapes\nnodule geometry.shapes;\n",
        ),
        ("bare marker", "// nodule\nnodule g.s;\n"),
        ("single-segment name", "// nodule: solo\n"),
        ("deep dotted name", "// nodule: a.b.c.d_e.f0\n"),
        ("no space before name", "// nodule:tight\n"),
        (
            "extra spaces around name",
            "//   nodule:    padded.name   \n",
        ),
        ("no space after slashes", "//nodule\n"),
        ("leading blank lines", "\n\n   \n// nodule: a.b\n"),
        (
            "crlf line endings",
            "\r\n  \r\n// nodule: c.d\r\nnodule c.d;\r\n",
        ),
        ("no trailing newline", "// nodule: end"),
        // non-markers — ordinary comments / code / empty (the oracle's None cases)
        ("ordinary comment", "// just a comment\nnodule d;\n"),
        (
            "prose mentioning nodule",
            "// nodule is Mycelium's word for module\nnodule d;\n",
        ),
        ("doc comment", "/// nodule\n"),
        ("code first", "nodule d;\nfn f() => Binary{8} = 0b0;"),
        ("empty source", ""),
        ("blank-only source", "\n  \n"),
        ("bare slash line", "/\n"),
        ("space before colon is prose", "// nodule : spaced\n"),
        // ill-formed named markers — the oracle's Err cases (G2 near-misses)
        ("empty name", "// nodule:\n"),
        ("empty name after spaces", "// nodule:   \n"),
        ("digit-led segment", "// nodule: 9bad\n"),
        ("doubled dot", "// nodule: a..b\n"),
        ("trailing dot", "// nodule: a.b.\n"),
        ("leading dot", "// nodule: .a\n"),
        ("space in name", "// nodule: has space\n"),
        ("error on line 3", "\n   \n// nodule: bad..name\n"),
    ];
    for (label, input) in inputs {
        assert_header_differential(label, input);
    }
}

/// The FLAG-nodule-2 narrowing, PINNED as a known divergence (review finding, PR #1165 — the
/// ASCII-only trim is a real CLASSIFICATION divergence, not a cosmetic one): a leading line
/// consisting solely of non-ASCII Unicode whitespace (NBSP, U+00A0) is blank to the Unicode-aware
/// Rust oracle (which then finds the marker on the next line) but NOT blank to the byte-wise
/// `.myc` port (which classifies the NBSP line itself and returns "no marker"). This test asserts
/// the DIVERGENT pair explicitly — oracle 2 (named), port 0 (none) — so the gap is loud in the
/// gate, and lifting the narrowing later will fail this pin and force FLAG-nodule-2's removal
/// (G2/VR-5: a known-wrong answer is recorded, never silent).
#[test]
fn nodule_myc_known_divergence_non_ascii_whitespace_only_line() {
    let input = "\u{00A0}\n// nodule: hidden.marker\n";
    assert_eq!(
        oracle_code(input),
        2,
        "oracle: an NBSP-only line is blank; the named marker on line 2 is found"
    );
    let driver = format!(
        "fn main() => Binary{{32}} = ph_code(\"{}\");",
        escape_myc_string(input)
    );
    assert_l1_only_u32(
        "KNOWN divergence (FLAG-nodule-2): NBSP-only leading line hides the marker from the port",
        &program(&driver),
        0,
    );
}

/// The Stage-2 gate (DN-26 §7.3): header-parse differential, Rust oracle vs self-hosted
/// `compiler.nodule`, over EVERY real `.myc` file in the L1 conformance corpus (accept + reject)
/// and every self-hosted nodule under `lib/` (`lib/std/`, `lib/compiler/` — including this port's
/// own source, whose first line is itself a named marker).
#[test]
fn nodule_myc_matches_oracle_over_corpus_and_lib() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dirs = [
        "docs/spec/grammar/conformance/accept",
        "docs/spec/grammar/conformance/reject",
        "lib/std",
        "lib/compiler",
    ];
    let mut total = 0usize;
    for dir in dirs {
        let dir_path = root.join(dir);
        let mut files: Vec<_> = std::fs::read_dir(&dir_path)
            .unwrap_or_else(|e| panic!("cannot read {dir_path:?}: {e}"))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "myc"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no .myc files found under {dir_path:?}");
        for path in &files {
            let source = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
            assert_header_differential(&format!("{dir}/{:?}", path.file_name()), &source);
            total += 1;
        }
    }
    assert!(
        total >= 60,
        "expected the corpus + lib sweep to cover ~60+ files, found {total}"
    );
}
