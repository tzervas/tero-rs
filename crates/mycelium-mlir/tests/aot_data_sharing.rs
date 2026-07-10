//! **M-994 (b) AOT analog — structural sharing + iterative destruction (measurement + witness).**
//!
//! The AOT env-machine ([`mycelium_mlir::aot`]) stores a datum as a crate-local structurally-shared
//! cons cell (`AotVal::Data(Rc<AotDatum>)`), **not** an inlined `mycelium_core::Datum`. This is the
//! AOT analog of the L1 interpreter's `Arc<Vec<..>>` on `L1Value::Data` (M-987): a variable reference
//! and a `Match`-arm field binding are O(1) `Rc` bumps rather than O(nodes) deep copies of the frozen
//! `Datum` spine (which lives inside the DN-56 kernel freeze and is **not** modified).
//!
//! Two families here:
//! 1. **Cost curve (`#[ignore]`d measurement, `Empirical`).** Reproduces the M-987 snoc/build shape on
//!    the AOT path and lets the exponent be re-measured. Before the change it is ~n^3 (per-reference +
//!    per-destructure deep clone); after, one factor of n is removed (measured ~13-35x at n=100-400,
//!    debug/release noted at run time). Mirrors `mycelium-l1/tests/spike_m994_cost.rs`.
//! 2. **Deep-datum witness (a real gate).** Builds and tears down a deeply-nested datum through the
//!    env-machine. The structural sharing makes construction cheap; the **iterative** `Drop`/`to_core`
//!    (the analog of `mycelium_core`'s recursion-safe destruction, RFC-0041 §4.5) means a deep spine
//!    never overflows the host stack — a derived (recursive) drop would `SIGABRT` (never-silent G2).
//!
//! Guarantee tag: **Empirical** (wall-clock trials + a depth witness), never a proof (VR-5).

use std::time::Instant;

use mycelium_core::data::{CtorSpec, DataRegistry, DeclSpec, FieldSpec};
use mycelium_core::{Alt, CoreValue, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::{IdentitySwapEngine, PrimRegistry};

// ─── program builders ───────────────────────────────────────────────────────────────────────────

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

/// The M-987 snoc/build cost shape: `build(n)` counts down and `snoc`s a byte onto the growing
/// accumulator each step; `snoc` recurses to the list end, referencing/binding the tail each level.
fn snoc_build_program(n: u32) -> Node {
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

/// A **cheap** deep-list builder (front-cons, O(1) per step, O(n) total): `front(n, acc) =
/// match eq(n,0) { 1 => acc ; _ => front(n-1, Cons(b, acc)) }`, applied to `front(n, Nil)`. Builds an
/// n-deep `Cons(..Cons(..Nil))` without the snoc quadratic — so it exercises deep construction and
/// deep teardown cheaply.
fn front_cons_program(n: u32) -> Node {
    let reg = l_registry();
    let nil = reg.ctor_ref("L", 0).unwrap();
    let cons = reg.ctor_ref("L", 1).unwrap();
    let v = |s: &str| Node::Var(s.to_owned());
    let body = Node::Lam {
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
                        func: Box::new(v("front")),
                        arg: Box::new(Node::Op {
                            prim: "bin.sub".into(),
                            args: vec![v("n"), Node::Const(u32v(1))],
                        }),
                    }),
                    arg: Box::new(Node::Construct {
                        ctor: cons.clone(),
                        args: vec![Node::Const(byte8(1)), v("acc")],
                    }),
                })),
            }),
        }),
    };
    Node::Let {
        id: "front".into(),
        bound: Box::new(Node::Fix {
            name: "front".into(),
            body: Box::new(body),
        }),
        body: Box::new(Node::App {
            func: Box::new(Node::App {
                func: Box::new(v("front")),
                arg: Box::new(Node::Const(u32v(n))),
            }),
            arg: Box::new(Node::Construct {
                ctor: nil,
                args: vec![],
            }),
        }),
    }
}

/// Count the spine length of a `Cons`-list `CoreValue` (iteratively — the value itself is
/// recursion-safe to traverse, RFC-0041 §4.5, but this test walker is explicit regardless).
fn list_len(mut cv: &CoreValue) -> usize {
    let mut n = 0;
    while let CoreValue::Data(d) = cv {
        if d.fields().len() == 2 {
            n += 1;
            cv = &d.fields()[1];
        } else {
            break; // Nil
        }
    }
    n
}

// ─── tests ────────────────────────────────────────────────────────────────────────────────────

/// **Witness (real gate): a deep datum builds and tears down without overflowing the host stack.**
/// A length-`DEEP` `Cons`-list is an `DEEP`-deep `AotVal::Data(Rc<AotDatum>)` chain during evaluation
/// and a `DEEP`-deep `CoreValue::Data` result; a derived (recursive) `Drop` on either would `SIGABRT`
/// well below `DEEP` (~a few thousand frames on a 2 MiB stack). Completing — and getting the right
/// length — witnesses BOTH the O(1) structural sharing (a non-shared build would be far slower/deeper)
/// AND the iterative `Drop`/`to_core` (RFC-0041 §4.5 analog). Mutation-witness: reverting `AotDatum`'s
/// `Drop` to a derived recursive form crashes this test.
#[test]
fn deep_datum_builds_and_drops_without_stack_overflow() {
    const DEEP: u32 = 30_000; // > the ~4-20k native-stack overflow threshold for recursive drop
    let prog = front_cons_program(DEEP);
    // High fuel + an ample depth ceiling. (Since M-996 the env-machine HAS TCO, so the driver's
    // tail re-entry elides and the loop no longer consumes DEEP control-stack depth; the ceiling is
    // kept generous anyway — the property under test is the deep *datum*, not the loop's depth.)
    let result = mycelium_mlir::aot::run_core_with_budget(
        &prog,
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        u64::MAX,
        1_000_000,
    )
    .expect("deep front-cons build must succeed");
    assert_eq!(
        list_len(&result),
        DEEP as usize,
        "the built list must have exactly DEEP Cons cells"
    );
    // `result` (a DEEP-deep CoreValue) drops here — mycelium-core's iterative Datum::drop handles it;
    // the intermediate DEEP-deep AotVal spine already dropped inside the env-machine (iterative Drop).
}

/// **Cost curve (`#[ignore]`d measurement).** Run explicitly:
/// `cargo test -p mycelium-mlir --test aot_data_sharing -- --ignored --nocapture`.
/// Sizes via `M994_AOT_SIZES=100,200,400`. `Empirical`: a wall-clock trial, not a correctness gate.
#[test]
#[ignore = "measurement spike (M-994 AOT (b)); run explicitly with --ignored --nocapture"]
fn m994_aot_cost_curve() {
    let sizes: Vec<u32> = std::env::var("M994_AOT_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().parse().expect("size"))
                .collect()
        })
        .unwrap_or_else(|| vec![100, 200, 400]);
    let prims = PrimRegistry::with_builtins();
    let eng = IdentitySwapEngine;
    println!("\n== M-994 AOT (b) cost curve ==");
    let mut pts: Vec<(u32, f64)> = Vec::new();
    for &n in &sizes {
        let prog = snoc_build_program(n);
        let t0 = Instant::now();
        let r =
            mycelium_mlir::aot::run_core_with_budget(&prog, &prims, &eng, u64::MAX, 100_000_000);
        let d = t0.elapsed().as_secs_f64();
        r.unwrap_or_else(|e| panic!("n={n}: {e:?}"));
        println!("  n = {n:>5} : {d:>10.4} s");
        pts.push((n, d));
    }
    if let (Some(&(n1, t1)), Some(&(n2, t2))) = (pts.first(), pts.last()) {
        if t1 > 0.0 && t2 > 0.0 && n1 != n2 {
            let p = (t2 / t1).ln() / (f64::from(n2) / f64::from(n1)).ln();
            println!("  fitted exponent p (t ~ n^p) over [{n1}, {n2}]: {p:.2}");
        }
    }
    println!("== end ==\n");
}
