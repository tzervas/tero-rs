//! **M-999 — the AOT-vs-interpreter ordering witness (measurement, not a gate).**
//!
//! Runs the **same workloads** through the L1 interpreter (`mycelium_l1::Evaluator`) and the AOT
//! env-machine (`mycelium_mlir::aot::run_core_with_budget`) in the **same build profile**, and
//! prints per-workload wall-clock times, the interpreter:env-machine ratio, and fitted exponents.
//! This is the recorded apples-to-apples baseline the M-995 report asked for: the earlier numbers
//! (AOT 1.11 s release vs L1 0.375 s debug at snoc n=400) were **cross-profile** and therefore
//! not a valid ordering claim; this harness is the honest replacement.
//!
//! Workloads:
//! 1. **snoc/build** (the M-987 cost shape): `build(n)` counts down, `snoc`-appending a byte to the
//!    end of the accumulator each step — O(n²) `snoc` levels; the classic env/clone hot path.
//! 2. **deep tail loop**: `count(n)` counts down in tail position — pure call/match machinery at
//!    O(1) control-stack depth on both machines (both have TCO: M-994 (a) / M-996).
//!
//! Guarantee tag: **Empirical** — single wall-clock trials on one host; the printed profile note
//! (debug/release) is part of the number (VR-5: never quote these cross-profile). Run explicitly:
//!
//! ```text
//! cargo test -p mycelium-mlir --release --test aot_vs_interp_bench -- --ignored --nocapture
//! ```
//!
//! Sizes via `M999_SIZES=100,200,400` (snoc) and `M999_COUNT_N=50000` (tail loop).
//!
//! ## Recorded baseline (release, one host, 2026-07-06 — `Empirical`)
//!
//! **BEFORE** the M-999 env-representation fix (`type Env = HashMap<Atom, AotVal>`, cloned per
//! closure capture / match arm / function-value lookup) — two runs agreed within ~2%:
//!
//! ```text
//! workload            n     L1 interp   AOT env-machine   interp/AOT
//! snoc/build        100      0.0096 s          0.0379 s        0.25x
//! snoc/build        200      0.0346 s          0.1486 s        0.23x
//! snoc/build        400      0.1378 s          0.6172 s        0.22x
//! tail count      50000      0.2632 s          0.3733 s        0.71x
//! fitted exponent (snoc, [100,400]):  L1 p = 1.92   AOT p = 2.01
//! ```
//!
//! (ratio < 1 = the interpreter is FASTER — the inverted ordering M-999 exists to fix. The M-995
//! structural sharing had already brought the AOT snoc curve to ~n²; the ~4.5x residual was a
//! constant-factor per-frame env overhead, not an exponent gap. Note the honest correction to the
//! earlier cross-profile framing: same-profile, the interpreter beat the env-machine ~4.4x on
//! snoc, not ~3x the raw cross-profile numbers suggested.)
//!
//! **AFTER** the M-999 representation fixes (env snapshot frames + the prepared `Rc`-shared code
//! mirror + interned `Rc<Atom>` keys + `Rc`-shared repr values — see the `aot.rs` module doc):
//!
//! ```text
//! workload            n     L1 interp   AOT env-machine   interp/AOT
//! snoc/build        100      0.0100 s          0.0062 s        1.60x
//! snoc/build        200      0.0348 s          0.0226 s        1.54x
//! snoc/build        400      0.1472 s          0.0876 s        1.68x
//! tail count      50000      0.2967 s          0.2185 s        1.36x
//! fitted exponent (snoc, [100,400]):  L1 p = 1.94   AOT p = 1.91
//! ```
//!
//! (ratio > 1 = the env-machine is FASTER — the required ordering. A second run agreed on snoc
//! within a few percent — 1.50x/1.57x/1.60x — with more jitter on the tail loop, 1.17x; treat the
//! tail-loop margin as ~1.2-1.4x, not a precise constant. Net AOT gain vs the BEFORE table:
//! ~7x on snoc n=400, ~1.7x on the tail loop; the snoc exponent stays ~n² — the M-995 structural
//! sharing already fixed the curve, M-999 removed the constant factor.) Re-measure with the
//! command above rather than trusting this comment as ground truth.

use std::time::Instant;

use mycelium_core::data::{CtorSpec, DataRegistry, DeclSpec, FieldSpec};
use mycelium_core::{Alt, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::{IdentitySwapEngine, PrimRegistry};
use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

// ─── shared value builders (local variants of the aot_data_sharing.rs fixtures) ─────────────────

fn byte8(n: u8) -> Value {
    let bits: Vec<bool> = (0..8).map(|i| (n >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}
fn u32v(n: u32) -> Value {
    // MSB-first (bits[0] is the sign/MSB), matching mycelium_core::binary::bits_to_int.
    let bits: Vec<bool> = (0..32).rev().map(|i| (n >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 32 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}
fn one_true() -> Value {
    Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![true]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `type L = Nil | Cons(Binary{8}, L)`.
fn l_registry() -> DataRegistry {
    let mut m = std::collections::BTreeMap::new();
    m.insert(
        "L".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] },
                CtorSpec {
                    fields: vec![
                        FieldSpec::Repr(Repr::Binary { width: 8 }),
                        FieldSpec::Data("L".to_owned()),
                    ],
                },
            ],
        },
    );
    DataRegistry::build(&m).unwrap()
}

// ─── the AOT-side programs (Core IR `Node`) ──────────────────────────────────────────────────────

/// The M-987 snoc/build cost shape (identical to `aot_data_sharing.rs::snoc_build_program`).
fn aot_snoc_program(n: u32) -> Node {
    let reg = l_registry();
    let nil = reg.ctor_ref("L", 0).unwrap();
    let cons = reg.ctor_ref("L", 1).unwrap();
    let v = |s: &str| Node::Var(s.to_owned());

    let snoc_body = Node::Lam {
        param: "xs".into(),
        body: Box::new(Node::Lam {
            param: "y".into(),
            body: Box::new(Node::Match {
                scrutinee: Box::new(v("xs")),
                alts: vec![
                    Alt::Ctor {
                        ctor: nil.clone(),
                        binders: vec![],
                        body: Node::Construct {
                            ctor: cons.clone(),
                            args: vec![
                                v("y"),
                                Node::Construct {
                                    ctor: nil.clone(),
                                    args: vec![],
                                },
                            ],
                        },
                    },
                    Alt::Ctor {
                        ctor: cons.clone(),
                        binders: vec!["h".into(), "t".into()],
                        body: Node::Construct {
                            ctor: cons.clone(),
                            args: vec![
                                v("h"),
                                Node::App {
                                    func: Box::new(Node::App {
                                        func: Box::new(v("snoc")),
                                        arg: Box::new(v("t")),
                                    }),
                                    arg: Box::new(v("y")),
                                },
                            ],
                        },
                    },
                ],
                default: None,
            }),
        }),
    };
    let build_body = Node::Lam {
        param: "n".into(),
        body: Box::new(Node::Lam {
            param: "acc".into(),
            body: Box::new(Node::Match {
                scrutinee: Box::new(Node::Op {
                    prim: "cmp.eq".into(),
                    args: vec![v("n"), Node::Const(u32v(0))],
                }),
                alts: vec![Alt::Lit {
                    value: one_true(),
                    body: v("acc"),
                }],
                default: Some(Box::new(Node::App {
                    func: Box::new(Node::App {
                        func: Box::new(v("build")),
                        arg: Box::new(Node::Op {
                            prim: "bin.sub".into(),
                            args: vec![v("n"), Node::Const(u32v(1))],
                        }),
                    }),
                    arg: Box::new(Node::App {
                        func: Box::new(Node::App {
                            func: Box::new(v("snoc")),
                            arg: Box::new(v("acc")),
                        }),
                        arg: Box::new(Node::Const(byte8(1))),
                    }),
                })),
            }),
        }),
    };
    Node::Let {
        id: "snoc".into(),
        bound: Box::new(Node::Fix {
            name: "snoc".into(),
            body: Box::new(snoc_body),
        }),
        body: Box::new(Node::Let {
            id: "build".into(),
            bound: Box::new(Node::Fix {
                name: "build".into(),
                body: Box::new(build_body),
            }),
            body: Box::new(Node::App {
                func: Box::new(Node::App {
                    func: Box::new(v("build")),
                    arg: Box::new(Node::Const(u32v(n))),
                }),
                arg: Box::new(Node::Construct {
                    ctor: nil,
                    args: vec![],
                }),
            }),
        }),
    }
}

/// A deep **tail** countdown: `count(n) = match eq(n,0) { 1 => n ; _ => count(n-1) }` — pure
/// call/match machinery, O(1) control-stack depth under the M-996 TCO.
fn aot_count_program(n: u32) -> Node {
    let v = |s: &str| Node::Var(s.to_owned());
    Node::Let {
        id: "count".into(),
        bound: Box::new(Node::Fix {
            name: "count".into(),
            body: Box::new(Node::Lam {
                param: "n".into(),
                body: Box::new(Node::Match {
                    scrutinee: Box::new(Node::Op {
                        prim: "cmp.eq".into(),
                        args: vec![v("n"), Node::Const(u32v(0))],
                    }),
                    alts: vec![Alt::Lit {
                        value: one_true(),
                        body: v("n"),
                    }],
                    default: Some(Box::new(Node::App {
                        func: Box::new(v("count")),
                        arg: Box::new(Node::Op {
                            prim: "bin.sub".into(),
                            args: vec![v("n"), Node::Const(u32v(1))],
                        }),
                    })),
                }),
            }),
        }),
        body: Box::new(Node::App {
            func: Box::new(v("count")),
            arg: Box::new(Node::Const(u32v(n))),
        }),
    }
}

// ─── the L1-side programs (surface source; identical shapes) ─────────────────────────────────────

/// The L1 snoc/build source (identical to `mycelium-l1/tests/spike_m994_cost.rs::program`).
fn l1_snoc_src(n: u32) -> String {
    format!(
        "nodule bench;\n\
         type L = Nil | Cons(Binary{{8}}, L);\n\
         fn snoc(xs: L, y: Binary{{8}}) => L = \
           match xs {{ Nil => Cons(y, Nil), Cons(h, t) => Cons(h, snoc(t, y)) }};\n\
         fn build(n: Binary{{32}}, acc: L) => L = \
           match eq(n, 0b{zero:032b}) {{ \
             0b1 => acc, \
             _ => build(sub_u(n, 0b{one:032b}), snoc(acc, 0b0000_0001)) \
           }};\n\
         fn main() => L = build(0b{n:032b}, Nil);",
        zero = 0u32,
        one = 1u32,
        n = n,
    )
}

/// The L1 tail-countdown source (the shape of [`aot_count_program`]).
fn l1_count_src(n: u32) -> String {
    format!(
        "nodule bench;\n\
         fn count(n: Binary{{32}}) => Binary{{32}} = \
           match eq(n, 0b{zero:032b}) {{ 0b1 => n, _ => count(sub_u(n, 0b{one:032b})) }};\n\
         fn main() => Binary{{32}} = count(0b{n:032b});",
        zero = 0u32,
        one = 1u32,
        n = n,
    )
}

// ─── drivers ─────────────────────────────────────────────────────────────────────────────────────

/// Time one AOT env-machine run (includes the `lower_to_anf` lowering — part of the AOT path).
fn time_aot(prog: &Node, prims: &PrimRegistry) -> f64 {
    let t0 = Instant::now();
    let r = mycelium_mlir::aot::run_core_with_budget(
        prog,
        prims,
        &IdentitySwapEngine,
        u64::MAX,
        100_000_000,
    );
    let d = t0.elapsed().as_secs_f64();
    r.unwrap_or_else(|e| panic!("AOT run failed: {e:?}"));
    d
}

/// Time one L1 interpreter run (parse/check/monomorphize excluded — the timed region is `eval`,
/// matching `spike_m994_cost.rs`; the AOT side's in-timer ANF lowering is O(program text) and
/// negligible against these workloads).
fn time_l1(src: &str) -> f64 {
    let nod = parse(src).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let checked = check_nodule(&nod).unwrap_or_else(|e| panic!("check failed: {e:?}"));
    let mono =
        monomorphize(&checked, "main").unwrap_or_else(|e| panic!("monomorphize failed: {e:?}"));
    let t0 = Instant::now();
    let r = Evaluator::new(&mono)
        .with_fuel(50_000_000_000)
        .call("main", vec![]);
    let d = t0.elapsed().as_secs_f64();
    r.unwrap_or_else(|e| panic!("L1 eval failed: {e}"));
    d
}

/// Fit `t ~ n^p` from the two extreme points.
fn fit_exponent(pts: &[(u32, f64)]) -> Option<f64> {
    let (n1, t1) = pts.first().copied()?;
    let (n2, t2) = pts.last().copied()?;
    (t1 > 0.0 && t2 > 0.0 && n1 != n2)
        .then(|| (t2 / t1).ln() / (f64::from(n2) / f64::from(n1)).ln())
}

// ─── the witness ─────────────────────────────────────────────────────────────────────────────────

/// Prints the same-profile, same-workload comparison table (the M-999 ordering witness). `Empirical`
/// — a measurement, deliberately `#[ignore]`d so the normal gate stays fast; run it explicitly with
/// the command in the module doc.
#[test]
#[ignore = "measurement (M-999 ordering witness); run explicitly with --release --ignored --nocapture"]
fn m999_env_machine_vs_interpreter_ordering() {
    let profile = if cfg!(debug_assertions) {
        "DEBUG"
    } else {
        "RELEASE"
    };
    let sizes: Vec<u32> = std::env::var("M999_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().parse().expect("size"))
                .collect()
        })
        .unwrap_or_else(|| vec![100, 200, 400]);
    let count_n: u32 = std::env::var("M999_COUNT_N")
        .ok()
        .map(|s| s.trim().parse().expect("count n"))
        .unwrap_or(50_000);
    let prims = PrimRegistry::with_builtins();

    println!("\n== M-999 ordering witness ({profile} build; Empirical, single trials) ==");
    println!(
        "  {:<14} {:>7} {:>12} {:>16} {:>12}",
        "workload", "n", "L1 interp", "AOT env-machine", "interp/AOT"
    );
    let mut l1_pts: Vec<(u32, f64)> = Vec::new();
    let mut aot_pts: Vec<(u32, f64)> = Vec::new();
    for &n in &sizes {
        let tl1 = time_l1(&l1_snoc_src(n));
        let taot = time_aot(&aot_snoc_program(n), &prims);
        println!(
            "  {:<14} {:>7} {:>10.4} s {:>14.4} s {:>11.2}x",
            "snoc/build",
            n,
            tl1,
            taot,
            tl1 / taot
        );
        l1_pts.push((n, tl1));
        aot_pts.push((n, taot));
    }
    let tl1 = time_l1(&l1_count_src(count_n));
    let taot = time_aot(&aot_count_program(count_n), &prims);
    println!(
        "  {:<14} {:>7} {:>10.4} s {:>14.4} s {:>11.2}x",
        "tail count",
        count_n,
        tl1,
        taot,
        tl1 / taot
    );
    if let (Some(p1), Some(p2)) = (fit_exponent(&l1_pts), fit_exponent(&aot_pts)) {
        let (lo, hi) = (l1_pts.first().unwrap().0, l1_pts.last().unwrap().0);
        println!("  fitted exponent (snoc, [{lo},{hi}]):  L1 p = {p1:.2}   AOT p = {p2:.2}");
    }
    println!("== end ==\n");
}
