//! **DN-50 narrow standing gate (W5/freeze-ledger).**
//!
//! RFC-0007 §4.6 / DN-50 OQ-2 obligation: for every construct the parser + checker accept (the
//! "parsable-and-checked" frontier), `elaborate(env, "main")` must return EITHER:
//!   (a) `Ok(node)` — the construct is in the evaluation-complete fragment and runs three-way
//!       (L1-eval ≡ L0-interp ≡ AOT), OR
//!   (b) `Err(ElabError::Residual { .. })` — the construct is accepted but not yet in the
//!       evaluation-complete fragment; it lowers with an explicit staging refusal.
//!
//! What is FORBIDDEN (a G2/DN-50 violation):
//!   - `elaborate` returns `Ok` but every downstream runner refuses (a silent accept-but-unrunnable
//!     with no explicit `Residual` from elaborate) — the silent gap DN-52 was designed to find.
//!   - `elaborate` returns `Err(ElabError::UnknownFn)` for a `main` that the checker accepted —
//!     that would be a semantic inconsistency between the checker and the elaborator.
//!   - `elaborate` panics.
//!
//! This test file is the **standing gate** (DN-52 §4 / DN-56 §5.1): it covers a representative
//! table of accepted-construct categories and asserts the invariant above. It does NOT require
//! every program to *run* — only that the elaborate result is transparent (Ok or explicit Residual).
//! The full three-way differential for the evaluation-complete fragment lives in `differential.rs`.
//!
//! Honesty: the gate is `Empirical` (representative corpus over categories, not exhaustive proof).
//! Coverage of the parsable-vs-runnable frontier is the DN-52 census; this test pins the gate
//! mechanically so the census stays green automatically after each change (DN-56 §5.1/freeze-gate).
//!
//! Classification of each category (matching DN-52 §3):
//!   `Runs`            — elaborate returns Ok; three-way differential confirmed (differential.rs)
//!   `Explicit-Residual` — elaborate returns Err(Residual{..}); never a silent diverge
//!
//! The gate does NOT prove "never a silent gap exists in the full language" — that is the census's
//! scope. It proves that for these recorded categories the invariant holds right now (Empirical).

use mycelium_l1::{check_nodule, elaborate, parse, ElabError};

/// One row in the standing-gate table.
struct GateRow {
    /// Short description of the construct category (for failure messages).
    category: &'static str,
    /// A checker-accepted source program with a nullary `main`.
    src: &'static str,
    /// Expected elaborate result class.
    expect: GateExpect,
}

#[derive(Debug, PartialEq, Eq)]
enum GateExpect {
    /// `elaborate` must return `Ok` — construct is in the evaluation-complete fragment.
    Runs,
    /// `elaborate` must return `Err(ElabError::Residual{..})` — staged, explicit.
    ExplicitResidual,
}

/// The standing-gate corpus: one row per construct category from the DN-52 census.
/// Every row must be parsable, check-checkable, and then classify as Runs or Explicit-Residual.
/// Any other outcome is a gate violation (G2/DN-50).
fn gate_corpus() -> Vec<GateRow> {
    vec![
        // -----------------------------------------------------------------------------------------
        // RUNS: the evaluation-complete Binary/Ternary fragment
        // -----------------------------------------------------------------------------------------
        GateRow {
            category: "Binary literal (bare)",
            src: "nodule d;\nfn main() => Binary{8} = 0b1011_0010;",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "Ternary literal (bare)",
            src: "nodule d;\nfn main() => Ternary{4} = 0t00+-;",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "Binary unary op",
            src: "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "Binary binary op (xor)",
            src: "nodule d;\nfn main() => Binary{8} = xor(0b1011_0010, 0b1111_1111);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "Ternary arithmetic (add)",
            src: "nodule d;\nfn main() => Ternary{4} = add(0t00+-, 0t0+0-);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "let binding",
            src: "nodule d;\nfn main() => Binary{8} = let a = 0b1011_0010 in a;",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "function call (inlined)",
            src: "nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(0b0000_0001);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "Binary→Ternary swap (evaluation-complete)",
            src: "nodule d;\nfn main() => Ternary{6} = swap(0b1011_0010, to: Ternary{6}, policy: rt);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "round-trip swap through let",
            src: "nodule d;\nfn main() => Binary{8} = let b = 0b0010_1010 in swap(swap(b, to: Ternary{6}, policy: rt), to: Binary{8}, policy: rt);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "data type with match",
            src: "nodule d;\ntype Flag = Off | On;\nfn main() => Binary{8} = match On { Off => 0b0000_0000, On => 0b0000_0001 };",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "recursive fn (self-recursion, terminating)",
            src: "nodule d;\nfn f(n: Binary{8}) => Binary{8} = match n { 0b0000_0000 => 0b0000_0000, _ => f(xor(n, 0b0000_0001)) };\nfn main() => Binary{8} = f(0b0000_0001);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "colony with single hypha",
            src: "nodule d;\nfn main() => Binary{8} = colony { hypha not(0b1011_0010) };",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "infix operator sugar (desugars to word call)",
            src: "nodule d;\nfn main() => Binary{8} = 0b1011_0010 ^ 0b1111_1111;",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "generic fn (monomorphized at elaborate time)",
            src: "nodule d;\nfn id[A](x: A) => A = x;\nfn main() => Binary{8} = id(0b0000_0001);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "trait impl (static dispatch, monomorphized)",
            src: "nodule d;\ntrait Flip[A] { fn flip(x: A) => A; };\nimpl Flip[Binary{8}] for Binary{8} { fn flip(x: Binary{8}) => Binary{8} = not(x); };\nfn main() => Binary{8} = flip(0b1011_0010);",
            expect: GateExpect::Runs,
        },
        GateRow {
            category: "for-fold over a list spine (data type + Fix fold desugar)",
            src: "nodule d;\ntype ByteList = End | More(Binary{8}, ByteList);\nfn main() => Binary{8} = let bs = More(0b1111_0000, More(0b0000_1111, End)) in for b in bs, acc = 0b0000_0000 => xor(acc, b);",
            expect: GateExpect::Runs,
        },
        // -----------------------------------------------------------------------------------------
        // EXPLICIT-RESIDUAL: accepted by checker but staged (not in evaluation-complete fragment).
        // Each must return Err(ElabError::Residual{..}) — never Ok or a panic.
        // -----------------------------------------------------------------------------------------
        GateRow {
            // DN-52 §3 row 7 / FLAG-1 RESOLVED (W5): Dense swap is accepted by the checker
            // (RFC-0002/RFC-0005) but `elaborate` returns an explicit Residual (never Ok or silent).
            // The BinaryTernarySwapEngine does not cover Dense conversions; a Dense-capable engine
            // lands with E2-1/ADR-033. DN-52 classification: Explicit-Residual (Empirical).
            category: "Dense swap target (DN-52 FLAG-1 RESOLVED → Explicit-Residual)",
            src: "nodule d;\nfn main() => Dense{4, F32} = swap(0b1011_0010, to: Dense{4, F32}, policy: rt);",
            expect: GateExpect::ExplicitResidual,
        },
        GateRow {
            // `wild` body not in host-call form: the checker accepts `wild { let a = … in a }` in
            // a `@std-sys` nodule (the body is opaque — RFC-0028 §4.2, M-661). The elaborator then
            // inspects the body's SHAPE; only a host-call form `name(args…)` lowers — any other
            // body shape is an explicit Residual (never a fabricated lowering — G2, elab.rs:915).
            // DN-52 §2.8 classification: Explicit-Residual.
            category: "`wild` body not in host-call form (DN-52 §2.8 → Explicit-Residual)",
            src: "nodule std.sys.x @std-sys;\nfn main() => Binary{8} !{ffi} = wild { let a = 0b0000_0000 in a };",
            expect: GateExpect::ExplicitResidual,
        },
    ]
}

/// **DN-50 narrow standing gate — the parsable-vs-runnable fence, mechanically wired.**
///
/// For every row in the table: the program parses, type-checks, and `elaborate` returns EITHER
/// `Ok` (construct is in the evaluation-complete fragment, Runs) OR `Err(ElabError::Residual{..})`
/// (explicitly staged). No other result is acceptable — `ElabError::UnknownFn` would be a
/// checker/elaborator inconsistency, and a panic would be a safety violation.
///
/// Gate: `Empirical` (representative table over categories). The obligation is DN-52 §4 /
/// DN-56 §5.1. Coverage: every accept is in this table's category OR in the three-way
/// differential corpus (`differential.rs`). Together they form the complete ledger.
///
/// DN-56 §5 freeze-gate condition #1: this gate must be green (no Undetermined rows) before the
/// kernel can be declared frozen. W5/freeze-ledger wires it here.
#[test]
fn every_accepted_construct_elaborates_to_ok_or_explicit_residual() {
    for row in gate_corpus() {
        // Step 1: must parse.
        let nodule = parse(row.src).unwrap_or_else(|e| {
            panic!(
                "DN-50 gate row [{cat}]: program must parse — {e}",
                cat = row.category
            )
        });
        // Step 2: must type-check.
        let env = check_nodule(&nodule).unwrap_or_else(|e| {
            panic!(
                "DN-50 gate row [{cat}]: program must type-check — {e}",
                cat = row.category
            )
        });
        // Step 3: elaborate must return Ok or Err(Residual). Any other outcome is a gate violation.
        let result = elaborate(&env, "main");
        match (&result, &row.expect) {
            (Ok(_), GateExpect::Runs) => {
                // Gate passes: elaborate succeeded, construct is in the evaluation-complete fragment.
            }
            (Err(ElabError::Residual { .. }), GateExpect::ExplicitResidual) => {
                // Gate passes: elaborate refused with an explicit Residual — staged, never silent.
            }
            (Ok(_), GateExpect::ExplicitResidual) => {
                panic!(
                    "DN-50 gate VIOLATION [{cat}]: expected Explicit-Residual but elaborate returned Ok — \
                     this construct would silently elaborate to L0 while runners may refuse, violating \
                     G2/DN-50 (the narrow gate). Fix: add an explicit Residual in elab.rs for this category.",
                    cat = row.category
                );
            }
            (Err(ElabError::Residual { site, what }), GateExpect::Runs) => {
                panic!(
                    "DN-50 gate REGRESSION [{cat}]: expected Runs but elaborate returned Residual \
                     (site={site}, what={what}). If this category genuinely became staged, update \
                     the DN-52 census and change the table to ExplicitResidual.",
                    cat = row.category
                );
            }
            (Err(ElabError::UnknownFn(name)), _) => {
                panic!(
                    "DN-50 gate INCONSISTENCY [{cat}]: elaborate returned UnknownFn({name}) for a \
                     `main` that the checker accepted — the checker and elaborator disagree on what \
                     is in scope. This is a semantic bug, not a staging refusal.",
                    cat = row.category
                );
            }
            (Err(ElabError::DepthExceeded { site, limit }), _) => {
                panic!(
                    "DN-50 gate INCONSISTENCY [{cat}]: elaborate hit its own call-graph-analysis \
                     recursion-depth budget ({limit}) at `{site}` (M-674) for a DN-50 corpus \
                     program — none of these fixtures should approach that budget; this indicates a \
                     corpus regression, not a staging refusal.",
                    cat = row.category
                );
            }
        }
    }
}
