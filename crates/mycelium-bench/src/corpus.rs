//! The shared **program corpus** every backend is measured on: v0-calculus source strings, parsed +
//! type-checked + elaborated to a closed Core IR [`Node`] (the universal backend input). Drawn from
//! the same shapes the M-210 three-way differential exercises (`crates/mycelium-l1/tests/
//! differential.rs`), so the corpus is grounded in already-trusted programs.
//!
//! Cases are tagged with the **fragment** they live in, which is what lets the harness *surface*
//! capability losses honestly (G2):
//! - [`Fragment::BitSubset`] — straight-line bit/trit programs the compiled backends (JIT,
//!   direct-LLVM, MLIR-dialect) can lower. These are where a backend can WIN or LOSE on *speed*.
//! - [`Fragment::Recursion`] — programs with `Fix`/`FixGroup` the compiled backends **cannot** lower
//!   (an explicit `Unsupported` refusal). Running these is how the harness records the compiled
//!   paths' **capability loss** with its reason — never omitting it.
//! - [`Fragment::Data`] — `Construct`/`Match` programs. A *flat non-recursive* match/construct to a
//!   repr the compiled backends MAY still lower; a recursive-data result they cannot. Which is which
//!   is a *measured* harness output (VR-5), not a pre-asserted capability claim.
//! - [`Fragment::Swap`] — uses the certified binary<->ternary swap. Since M-852 (PR #823) the compiled
//!   paths natively lower a **legal-pair** swap to a repr value; an illegal pair or unsupported repr
//!   is still an explicit capability loss. Which is which is a *measured* harness output (VR-5), like
//!   [`Fragment::Data`] — not a pre-asserted "always a loss" fact.
//!
//! Every case carries its source so the report is reproducible and the verdict is auditable.

use mycelium_core::Node;
use mycelium_l1::{check_nodule, elaborate, parse};

/// Which evaluation-complete fragment a case lives in — the basis for whether a compiled backend can
/// be expected to run it at all (a capability boundary, reified — no black box).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fragment {
    /// Straight-line bit/trit ops over `Binary{w}` / `Ternary{m}` — lowerable by every backend.
    BitSubset,
    /// Builds/matches algebraic data (`Construct`/`Match`) — interp + AOT env-machine only.
    Data,
    /// Recursion (`Fix`/`FixGroup`) — interp + AOT env-machine only.
    Recursion,
    /// Uses the certified binary<->ternary `Swap` — a legal pair now natively lowers on the compiled
    /// backends too (M-852); an illegal pair/unsupported repr is still a capability loss.
    Swap,
}

impl Fragment {
    /// A short human label for the report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Fragment::BitSubset => "bit-subset",
            Fragment::Data => "data",
            Fragment::Recursion => "recursion",
            Fragment::Swap => "swap",
        }
    }
}

/// One corpus entry: a stable id, the v0-calculus source, its fragment, and a one-line note on what
/// it stresses. The elaborated [`Node`] is produced on demand by [`Case::elaborate`].
#[derive(Debug, Clone)]
pub struct Case {
    /// A stable, human-readable id (used as the report row key; sorted for determinism).
    pub id: &'static str,
    /// The v0-calculus source program (entry point is always `main`).
    pub src: &'static str,
    /// The fragment this case lives in.
    pub fragment: Fragment,
    /// A one-line note: what this case exercises / why it is in the corpus.
    pub note: &'static str,
}

/// An error from turning a corpus source into a Core IR term — kept explicit (never a silent skip)
/// so a corpus regression (a program that no longer parses/checks/elaborates) is loud.
#[derive(Debug)]
pub enum CorpusError {
    /// The source failed to parse.
    Parse(String),
    /// The source failed to type-check.
    Check(String),
    /// The source failed to elaborate to a closed L0 term.
    Elaborate(String),
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorpusError::Parse(m) => write!(f, "corpus parse error: {m}"),
            CorpusError::Check(m) => write!(f, "corpus check error: {m}"),
            CorpusError::Elaborate(m) => write!(f, "corpus elaborate error: {m}"),
        }
    }
}

impl std::error::Error for CorpusError {}

impl Case {
    /// Parse, type-check and elaborate this case's source to a closed Core IR [`Node`] (the input
    /// every backend consumes). Errors are explicit — a corpus program that stops parsing/checking/
    /// elaborating is a loud failure, not a silent drop (G2).
    pub fn elaborate(&self) -> Result<Node, CorpusError> {
        let nodule = parse(self.src).map_err(|e| CorpusError::Parse(e.to_string()))?;
        let env = check_nodule(&nodule).map_err(|e| CorpusError::Check(e.to_string()))?;
        elaborate(&env, "main").map_err(|e| CorpusError::Elaborate(e.to_string()))
    }
}

/// The full corpus, in a stable order. Bit-subset cases first (every backend runs them), then the
/// data / recursion / swap cases (where the compiled backends record a capability loss).
#[must_use]
pub fn corpus() -> Vec<Case> {
    vec![
        // ── Bit subset: lowerable by every backend (the speed WIN/LOSS surface) ──────────────────
        Case {
            id: "bit-literal",
            src: "nodule d;\nfn main() => Binary{8} = 0b1011_0010;",
            fragment: Fragment::BitSubset,
            note: "a bare 8-bit literal — the most trivial kernel (spawn-bound for compiled paths)",
        },
        Case {
            id: "bit-not",
            src: "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);",
            fragment: Fragment::BitSubset,
            note: "single unary bit op",
        },
        Case {
            id: "bit-xor-not",
            src: "nodule d;\nfn main() => Binary{8} = not(xor(0b1011_0010, 0b1111_1111));",
            fragment: Fragment::BitSubset,
            note: "a small straight-line bit pipeline (the xtask-e1 §2 shape)",
        },
        Case {
            id: "bit-let-chain",
            src: "nodule d;\nfn main() => Binary{8} =\n  let a = 0b1011_0010 in let b = xor(a, 0b0000_1111) in not(xor(b, a));",
            fragment: Fragment::BitSubset,
            note: "let-bound straight-line bit computation (a few surface ops: not/xor only)",
        },
        Case {
            id: "trit-neg",
            src: "nodule d;\nfn main() => Ternary{4} = neg(0t00+-);",
            fragment: Fragment::BitSubset,
            note: "element-wise trit negation (the only trit op the MLIR-dialect path lowers)",
        },
        Case {
            id: "trit-add",
            src: "nodule d;\nfn main() => Ternary{4} = add(0t00+-, 0t0+0-);",
            fragment: Fragment::BitSubset,
            note: "balanced-ternary ripple-carry add (direct-LLVM lowers it; MLIR-dialect does NOT — a capability split)",
        },
        // ── Swap: certified binary<->ternary — a legal pair, natively lowered since M-852 ──────────
        Case {
            id: "swap-roundtrip",
            src: "nodule d;\nfn main() => Binary{8} =\n  let b = 0b0010_1010 in swap(swap(b, to: Ternary{6}, policy: rt), to: Binary{8}, policy: rt);",
            fragment: Fragment::Swap,
            note: "a binary->ternary->binary certified round-trip over a LEGAL (8,6) pair — since \
                   M-852 the compiled backends natively lower this to a value (measured, not \
                   assumed); an illegal pair remains a capability loss",
        },
        // ── Data: Construct/Match — interp + AOT env-machine only ─────────────────────────────────
        Case {
            id: "data-match-repr",
            src: "nodule d;\ntype Sign = Neg | Zero | Pos;\nfn label(s: Sign) => Ternary{1} = match s { Neg => 0t-, Zero => 0t0, _ => 0t+ };\nfn main() => Ternary{1} = label(Zero);",
            fragment: Fragment::Data,
            note: "a flat non-recursive data match returning a repr — a data case the compiled \
                   backends MAY still lower (measured, not assumed)",
        },
        Case {
            id: "data-construct",
            src: "nodule d;\ntype Nat = Z | S(Nat);\nfn main() => Nat = S(S(Z));",
            fragment: Fragment::Data,
            note: "a datum result (the program evaluates to constructed data)",
        },
        Case {
            id: "data-nested-match",
            src: "nodule d;\ntype Nat = Z | S(Nat);\nfn pred2(n: Nat) => Nat = match n { Z => Z, S(Z) => Z, S(S(m)) => m };\nfn main() => Nat = pred2(S(S(S(Z))));",
            fragment: Fragment::Data,
            note: "nested (Maranget) patterns returning a datum",
        },
        // ── Recursion: Fix / FixGroup — interp + AOT env-machine only ─────────────────────────────
        Case {
            id: "rec-self",
            src: "nodule d;\ntype Nat = Z | S(Nat);\nfn drop_(n: Nat) => Nat = match n { Z => Z, S(m) => drop_(m) };\nfn main() => Nat = drop_(S(S(S(Z))));",
            fragment: Fragment::Recursion,
            note: "self-recursion consuming a Nat (Fix + App + Match)",
        },
        Case {
            id: "rec-build",
            src: "nodule d;\ntype Nat = Z | S(Nat);\nfn copy(n: Nat) => Nat = match n { Z => Z, S(m) => S(copy(m)) };\nfn main() => Nat = copy(S(S(Z)));",
            fragment: Fragment::Recursion,
            note: "self-recursion building data on the way back",
        },
        Case {
            id: "rec-mutual",
            src: "nodule d;\ntype Nat = Z | S(Nat);\nfn ping(n: Nat) => Nat = match n { Z => Z, S(m) => pong(m) };\nfn pong(n: Nat) => Nat = match n { Z => Z, S(m) => ping(m) };\nfn main() => Nat = ping(S(S(Z)));",
            fragment: Fragment::Recursion,
            note: "mutual recursion (a FixGroup)",
        },
        Case {
            id: "rec-fold",
            src: "nodule d;\ntype ByteList = End | More(Binary{8}, ByteList);\nfn checksum(bs: ByteList) => Binary{8} = for b in bs, acc = 0b0000_0000 => xor(acc, b);\nfn main() => Binary{8} = checksum(More(0b1111_0000, More(0b0000_1111, End)));",
            fragment: Fragment::Recursion,
            note: "a `for` fold over a list spine (a synthesized Fix fold) returning a repr",
        },
    ]
}
