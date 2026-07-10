//! The RFC-0012 §4.6 **meaning-preservation differential** (M-344; NFR-7). The ambient is *sugar,
//! not behavior*: a program written with the ambient and its explicit longhand twin must elaborate
//! to the **identical** L0 — hence the **identical content hash** (RFC-0001 §4.6) — and, where they
//! run, observe the identical value. This is the executable proof of invariant **I2** (resolution is
//! observationally the identity) and, transitively, **I1** (the ambient inserts no `Swap` — a
//! fabricated conversion would change the elaborated structure and break the hash).
//!
//! The never-silent refusals (§4.3/§4.4) are tested too: an unresolved/ill-shaped ambient, a
//! bare-decimal with no encoding or no width, and a cross-paradigm `MissingConversion` are each an
//! explicit error, never a silent coercion.

use mycelium_core::ContentHash;
use mycelium_interp::Interpreter;
use mycelium_l1::{check_and_resolve, check_nodule, elaborate, expand_to_source, parse, resolve};

/// Elaborate `src`'s `main` to L0 and return its content hash (the identity that I2 preserves).
fn elaborated_hash(src: &str) -> ContentHash {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("parse `{src}`: {e}")))
        .unwrap_or_else(|e| panic!("check `{src}`: {e}"));
    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("elaborate `{src}`: {e}"));
    node.content_hash()
}

/// `(ambient program, explicit longhand twin)` — both must elaborate to the identical L0.
fn pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        // Paradigm-less return + tagged-literal body (the dominant repetition killer).
        (
            "nodule d;\ndefault paradigm Binary;\nfn main() => {8} = xor(0b1111_0000, 0b0000_1111);",
            "nodule d;\nfn main() => Binary{8} = xor(0b1111_0000, 0b0000_1111);",
        ),
        // A bare decimal whose width comes from the return type (5 = 0b0000_0101).
        (
            "nodule d;\ndefault paradigm Binary;\nfn main() => {8} = 5;",
            "nodule d;\nfn main() => Binary{8} = 0b0000_0101;",
        ),
        // A bare decimal under a Ternary ambient (5 = balanced ternary 0+-- at width 4).
        (
            "nodule d;\ndefault paradigm Ternary;\nfn main() => {4} = 5;",
            "nodule d;\nfn main() => Ternary{4} = 0t0+--;",
        ),
        // Paradigm-less parameter + return through a call.
        (
            "nodule d;\ndefault paradigm Binary;\nfn flip(x: {8}) => {8} = not(x);\nfn main() => {8} = flip(0b1010_1010);",
            "nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(0b1010_1010);",
        ),
        // A bare decimal as a call argument (width from the parameter type; 170 = 0b1010_1010).
        (
            "nodule d;\ndefault paradigm Binary;\nfn flip(x: {8}) => {8} = not(x);\nfn main() => {8} = flip(170);",
            "nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(0b1010_1010);",
        ),
        // A `with paradigm` override block + a paradigm-less swap target — the block is pure scoping
        // and elaborates away (I1: the swap is the author's, not the ambient's).
        (
            "nodule d;\ndefault paradigm Binary;\nfn main() => Ternary{6} = with paradigm Ternary { swap(0b1011_0010, to: {6}, policy: rt) };",
            "nodule d;\nfn main() => Ternary{6} = swap(0b1011_0010, to: Ternary{6}, policy: rt);",
        ),
        // A bare-decimal operand whose width is pinned by the *other* operand (concrete anchor).
        (
            "nodule d;\ndefault paradigm Binary;\nfn main() => {8} = xor(0b1111_0000, 15);",
            "nodule d;\nfn main() => Binary{8} = xor(0b1111_0000, 0b0000_1111);",
        ),
    ]
}

#[test]
fn ambient_and_longhand_twins_elaborate_to_the_identical_l0() {
    for (i, (ambient, longhand)) in pairs().iter().enumerate() {
        assert_eq!(
            elaborated_hash(ambient),
            elaborated_hash(longhand),
            "pair #{i}: the ambient program and its longhand twin must share a content hash (I2)\n  \
             ambient:  {ambient}\n  longhand: {longhand}"
        );
    }
}

#[test]
fn the_twins_observe_the_identical_value() {
    // A runnable subset: both twins run on the reference interpreter and agree on the observable.
    let interp = Interpreter::default();
    for (ambient, longhand) in pairs() {
        let run = |src: &str| {
            let env = check_nodule(&parse(src).unwrap()).unwrap();
            let node = elaborate(&env, "main").unwrap();
            interp
                .eval(&node)
                .map(|v| (v.repr().clone(), v.payload().clone()))
        };
        // Only compare where both actually produce a value (the binary→ternary swap pair needs the
        // cert engine, which the default interpreter lacks — its identity engine refuses it equally
        // on both twins, so a matched `Err` is still agreement).
        match (run(ambient), run(longhand)) {
            (Ok(a), Ok(b)) => assert_eq!(
                a, b,
                "twins diverged at runtime:\n  {ambient}\n  {longhand}"
            ),
            (Err(_), Err(_)) => {}
            (a, b) => {
                panic!("twins disagree on runnability: {a:?} vs {b:?}\n  {ambient}\n  {longhand}")
            }
        }
    }
}

#[test]
fn expand_ambient_renders_a_faithful_longhand_twin() {
    // The M-142/LSP "expand ambient" projection: rendering the resolved twin to source and
    // re-elaborating it reproduces the original program's L0 (RFC-0012 §5).
    let ambient = "nodule d;\ndefault paradigm Binary;\nfn flip(x: {8}) => {8} = not(x);\nfn main() => {8} = flip(5);";
    let (_, twin) = check_and_resolve(&parse(ambient).unwrap()).expect("checks");
    let rendered = expand_to_source(&twin);
    assert!(
        !rendered.contains("default paradigm") && rendered.contains("Binary{8}"),
        "the expanded form must be fully longhand (no ambient declaration; concrete reprs):\n{rendered}"
    );
    // Re-parsing the rendered longhand must succeed with no residual paradigm-less repr.
    assert!(
        mycelium_l1::resolve(&parse(&rendered).expect("expanded form reparses")).is_ok(),
        "the expanded longhand must itself be ambient-free:\n{rendered}"
    );
    assert_eq!(
        elaborated_hash(&rendered),
        elaborated_hash(ambient),
        "the expanded longhand must elaborate to the same L0 as the ambient original:\n{rendered}"
    );
}

#[test]
fn a_program_with_no_ambient_is_untouched() {
    // The feature is opt-in: resolution is the identity on a pre-RFC-0012 program.
    let src = "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);";
    let resolved = resolve(&parse(src).unwrap()).expect("resolves");
    assert_eq!(
        resolved,
        parse(src).unwrap(),
        "no-ambient resolution must be the identity"
    );
}

#[test]
fn a_paradigm_less_ascription_states_the_per_use_size() {
    // R12-Q1 (resolved 2026-06-16, M-351): per-use size needs no new sugar — a paradigm-less
    // ascription `e : {N}` already supplies the ambient paradigm + an explicit width at the use
    // site, so a context-free bare decimal is sized without a surrounding annotation, and it
    // elaborates identically to the fully-tagged longhand (I2). Sizes stay explicit; no default
    // width (the v0 honesty principle is preserved).
    assert_eq!(
        elaborated_hash("nodule d;\ndefault paradigm Binary;\nfn main() => Binary{8} = (5 : {8});"),
        elaborated_hash("nodule d;\nfn main() => Binary{8} = (0b0000_0101 : Binary{8});"),
        "a `: {{N}}` ascription must pin the per-use size identically to longhand (R12-Q1)"
    );
}

// --- never-silent refusals (§4.3/§4.4) -----------------------------------------------------------

fn check_err(src: &str) -> String {
    check_nodule(&parse(src).expect("parses"))
        .expect_err("must refuse")
        .message
}

#[test]
fn a_paradigm_less_repr_with_no_ambient_is_unresolved_ambient() {
    let msg = check_err("nodule d;\nfn main() => {8} = 0b1011_0010;");
    assert!(msg.contains("no enclosing ambient"), "got: {msg}");
}

#[test]
fn a_shape_that_does_not_fit_the_ambient_is_a_paradigm_shape_mismatch() {
    // `{8}` (a single size) cannot fill a `Dense` ambient (which needs `{dim, scalar}`).
    let msg = check_err("nodule d;\ndefault paradigm Dense;\nfn main() => {8} = 0b1011_0010;");
    assert!(
        msg.contains("does not fit the `Dense` ambient"),
        "got: {msg}"
    );
}

#[test]
fn a_bare_decimal_under_dense_has_no_encoding() {
    let msg = check_err("nodule d;\ndefault paradigm Dense;\nfn main() => Dense{4, F32} = 5;");
    assert!(msg.contains("no `Dense` encoding"), "got: {msg}");
}

#[test]
fn a_bare_decimal_with_no_width_context_is_unresolved_width() {
    // A swap *source* is unconstrained by the target, so a bare decimal there has no width.
    let msg = check_err(
        "nodule d;\ndefault paradigm Binary;\nfn main() => Ternary{6} = swap(5, to: Ternary{6}, policy: rt);",
    );
    assert!(msg.contains("UnresolvedWidth"), "got: {msg}");
}

#[test]
fn a_cross_paradigm_edge_is_a_missing_conversion() {
    // A Binary body where Ternary is required — never silently converted.
    let msg =
        check_err("nodule d;\ndefault paradigm Binary;\nfn main() => Ternary{6} = 0b1011_0010;");
    assert!(
        msg.contains("MissingConversion") && msg.contains("swap"),
        "got: {msg}"
    );
}

#[test]
fn two_nodule_defaults_are_refused() {
    let msg = check_err(
        "nodule d;\ndefault paradigm Binary;\ndefault paradigm Ternary;\nfn main() => {8} = 0b1011_0010;",
    );
    assert!(msg.contains("two `default paradigm`"), "got: {msg}");
}

#[test]
fn an_overflowing_bare_decimal_is_refused_not_wrapped() {
    // 300 does not fit Binary{8} — an explicit refusal, never a silent wrap (RFC-0012 §4.3).
    let msg = check_err("nodule d;\ndefault paradigm Binary;\nfn main() => {8} = 300;");
    assert!(msg.contains("does not fit"), "got: {msg}");
}
