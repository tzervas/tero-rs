//! RFC-0041 §4.5 (W3) — iterative, recursion-safe destruction/traversal for the frozen recursive
//! kernel types.
//!
//! Two witness families:
//!
//! 1. **Bit-identical (bar (a)).** The iterative `Clone` / `PartialEq` / `content_hash` produce
//!    results indistinguishable from the derived/recursive forms. The hash is checked against an
//!    independent **recursive reference oracle** ([`ref_hash`], a faithful copy of the *former*
//!    recursive `Canon::node` encoder) over a corpus spanning every node variant and binder-scope
//!    shape — so any change to the iterative byte stream diverges from the oracle and fails
//!    (mutation-witness for the hash conversion).
//!
//! 2. **Depth-safe (the SIGABRT killer).** Deep chains (`DEEP` links) are **constructed, dropped,
//!    cloned, hashed, and unwound** without overflowing the native stack. These are the
//!    mutation-witnesses for `Drop`/`Clone`/hash: weakening the iterative machinery reverts the type
//!    to (auto-)recursive destruction, which `SIGABRT`s on `DEEP` input and crashes the test.
//!
//! Note on `assert!(a == b)` over deep values: the derived `Debug` is still recursive, so `assert_eq!`
//! (which `Debug`-formats on failure) is deliberately avoided for deep inputs.

use crate::content::Canon;
use crate::data::{CtorSpec, DataRegistry, DeclSpec, FieldSpec};
use crate::datum::{CoreValue, Datum};
use crate::id::ContentHash;
use crate::meta::{Meta, Provenance};
use crate::node::{Alt, Node, VarId};
use crate::repr::Repr;
use crate::value::{Payload, Value};
use std::collections::BTreeMap;

/// Deep-chain length. On a test thread's ~2 MiB stack a recursive drop/clone/hash overflows at a few
/// thousand frames, so this depth (`~100k`, a wide margin over the ~4–20k overflow threshold for the
/// three recursion kinds) aborts the process under any regression to recursion — the mutation-witness
/// — while staying fast enough for the change-scoped tier (DN-20). RFC-0041 §4.5 cites 500k as the
/// headline goal; the guarantee (bounded native stack, O(depth) heap) holds identically at any depth
/// (nothing in these loops scales with a fixed bound), and 100k already clears the overflow threshold
/// by >5×. Kept at 100k (not 500k) to bound the debug-build parallel run's time/memory.
const DEEP: usize = 100_000;

// --------------------------------------------------------------------------------------------
// Fixtures
// --------------------------------------------------------------------------------------------

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed byte")
}

/// A `Nat`-like registry: ctor 0 = `Z` (nullary), ctor 1 = `S(Nat)` (one recursive field).
fn nat_registry() -> DataRegistry {
    let mut m = BTreeMap::new();
    m.insert(
        "Nat".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] },
                CtorSpec {
                    fields: vec![FieldSpec::Data("Nat".to_owned())],
                },
            ],
        },
    );
    DataRegistry::build(&m).expect("nat registry")
}

// --------------------------------------------------------------------------------------------
// Recursive reference hash oracle — a faithful copy of the FORMER recursive `Canon::node`
// (content.rs, pre-RFC-0041). It reuses the same `pub(crate)` leaf encoders (`value`/`repr`/
// `ctor_ref`/…), so it differs from the production encoder ONLY in the node-structural recursion
// this change made iterative — i.e. it is an exact oracle for that conversion.
// --------------------------------------------------------------------------------------------

fn ref_hash(n: &Node) -> ContentHash {
    let mut c = Canon::new();
    let mut scope: Vec<VarId> = Vec::new();
    ref_node(&mut c, n, &mut scope);
    c.finish()
}

fn ref_node(c: &mut Canon, n: &Node, scope: &mut Vec<VarId>) {
    use crate::content::tag;
    match n {
        Node::Var(name) => {
            if let Some(pos) = scope.iter().rposition(|b| b == name) {
                c.tag(tag::VAR_BOUND);
                c.u32((scope.len() - 1 - pos) as u32);
            } else {
                c.tag(tag::VAR_FREE);
                c.str(name);
            }
        }
        Node::Const(v) => {
            c.tag(tag::CONST);
            c.value(v);
        }
        Node::Let { id, bound, body } => {
            c.tag(tag::LET);
            ref_node(c, bound, scope);
            scope.push(id.clone());
            ref_node(c, body, scope);
            scope.pop();
        }
        Node::Op { prim, args } => {
            c.tag(tag::OP);
            c.str(prim);
            c.u64(args.len() as u64);
            for a in args {
                ref_node(c, a, scope);
            }
        }
        Node::Swap {
            src,
            target,
            policy,
        } => {
            c.tag(tag::SWAP);
            ref_node(c, src, scope);
            c.repr(target);
            c.str(policy.as_str());
        }
        Node::Construct { ctor, args } => {
            c.tag(tag::CONSTRUCT);
            c.ctor_ref(ctor);
            c.u64(args.len() as u64);
            for a in args {
                ref_node(c, a, scope);
            }
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            c.tag(tag::MATCH);
            ref_node(c, scrutinee, scope);
            c.u64(alts.len() as u64);
            for alt in alts {
                match alt {
                    Alt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        c.tag(tag::ALT_CTOR);
                        c.ctor_ref(ctor);
                        c.u64(binders.len() as u64);
                        let mark = scope.len();
                        scope.extend(binders.iter().cloned());
                        ref_node(c, body, scope);
                        scope.truncate(mark);
                    }
                    Alt::Lit { value, body } => {
                        c.tag(tag::ALT_LIT);
                        c.value(value);
                        ref_node(c, body, scope);
                    }
                }
            }
            match default {
                Some(d) => {
                    c.tag(tag::MATCH_DEFAULT);
                    ref_node(c, d, scope);
                }
                None => c.tag(tag::MATCH_NO_DEFAULT),
            }
        }
        Node::Lam { param, body } => {
            c.tag(tag::LAM);
            scope.push(param.clone());
            ref_node(c, body, scope);
            scope.pop();
        }
        Node::App { func, arg } => {
            c.tag(tag::APP);
            ref_node(c, func, scope);
            ref_node(c, arg, scope);
        }
        Node::Fix { name, body } => {
            c.tag(tag::FIX);
            scope.push(name.clone());
            ref_node(c, body, scope);
            scope.pop();
        }
        Node::FixGroup { defs, body } => {
            c.tag(tag::FIXGROUP);
            c.u64(defs.len() as u64);
            let mark = scope.len();
            for (name, _) in defs {
                scope.push(name.clone());
            }
            for (_, d) in defs {
                ref_node(c, d, scope);
            }
            ref_node(c, body, scope);
            scope.truncate(mark);
        }
    }
}

/// A corpus spanning every `Node` variant and several binder-scope shapes (nested `Let`, `Lam` under
/// `Let`, `Match` alt binders shadowing, `FixGroup` mutual binders, bound-vs-free `Var`s). Kept
/// **shallow** so the recursive oracle is safe to run — depth-safety is witnessed separately.
// Sequential `push`es (rather than one `vec![]`) keep the shared-`clone` construction legible.
#[allow(clippy::vec_init_then_push)]
fn node_corpus() -> Vec<Node> {
    let reg = nat_registry();
    let z = reg.ctor_ref("Nat", 0).expect("Z");
    let s = reg.ctor_ref("Nat", 1).expect("S");
    let policy = ContentHash::parse("blake3:round_trip_safe").expect("policy hash");

    let mut v = Vec::new();

    v.push(Node::Var("free".to_owned()));
    v.push(Node::Const(byte([
        true, false, true, true, false, false, true, false,
    ])));

    // let a = <byte> in swap(a -> Ternary{6}) : bound var resolves to a de Bruijn index.
    v.push(Node::Let {
        id: "a".to_owned(),
        bound: Box::new(Node::Const(byte([false; 8]))),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Var("a".to_owned())),
            target: Repr::Ternary { trits: 6 },
            policy: policy.clone(),
        }),
    });

    // Op with 0 and 2 args.
    v.push(Node::Op {
        prim: "bit.not".to_owned(),
        args: vec![],
    });
    v.push(Node::Op {
        prim: "bit.xor".to_owned(),
        args: vec![Node::Var("x".to_owned()), Node::Var("y".to_owned())],
    });

    // Construct S(Z) and S(S(Z)).
    v.push(Node::Construct {
        ctor: s.clone(),
        args: vec![Node::Construct {
            ctor: z.clone(),
            args: vec![],
        }],
    });

    // Match with a Ctor alt (binders shadow), a Lit alt, and a default.
    v.push(Node::Match {
        scrutinee: Box::new(Node::Var("scrut".to_owned())),
        alts: vec![
            Alt::Ctor {
                ctor: s.clone(),
                binders: vec!["n".to_owned()],
                // body references the binder `n` (bound) and a free var.
                body: Node::Op {
                    prim: "use".to_owned(),
                    args: vec![Node::Var("n".to_owned()), Node::Var("free".to_owned())],
                },
            },
            Alt::Lit {
                value: byte([true; 8]),
                body: Node::Var("free".to_owned()),
            },
        ],
        default: Some(Box::new(Node::Var("d".to_owned()))),
    });
    // Match with no default and no alts (degenerate but well-framed for hashing).
    v.push(Node::Match {
        scrutinee: Box::new(Node::Const(byte([false; 8]))),
        alts: vec![],
        default: None,
    });

    // Lam under a Let (nested binders), App, Fix, and a FixGroup of two mutually-recursive members.
    v.push(Node::Let {
        id: "outer".to_owned(),
        bound: Box::new(Node::Const(byte([false; 8]))),
        body: Box::new(Node::Lam {
            param: "x".to_owned(),
            body: Box::new(Node::App {
                func: Box::new(Node::Var("outer".to_owned())),
                arg: Box::new(Node::Var("x".to_owned())),
            }),
        }),
    });
    v.push(Node::Fix {
        name: "f".to_owned(),
        body: Box::new(Node::Var("f".to_owned())),
    });
    v.push(Node::FixGroup {
        defs: vec![
            ("f".to_owned(), Box::new(Node::Var("g".to_owned()))),
            ("g".to_owned(), Box::new(Node::Var("f".to_owned()))),
        ],
        body: Box::new(Node::App {
            func: Box::new(Node::Var("f".to_owned())),
            arg: Box::new(Node::Var("g".to_owned())),
        }),
    });

    v
}

// --------------------------------------------------------------------------------------------
// 1. Bit-identical witnesses (bar (a))
// --------------------------------------------------------------------------------------------

#[test]
fn iterative_node_hash_matches_the_recursive_oracle() {
    for n in node_corpus() {
        assert!(
            n.content_hash() == ref_hash(&n),
            "iterative Canon::node diverged from the recursive reference encoding"
        );
    }
}

#[test]
fn node_clone_is_structurally_equal_and_identity_preserving() {
    for n in node_corpus() {
        let c = n.clone();
        assert!(
            n == c,
            "clone must be structurally equal (iterative PartialEq)"
        );
        assert!(
            n.content_hash() == c.content_hash(),
            "clone must preserve the content hash"
        );
        // Round-trip through the oracle too: the clone hashes to the same reference bytes.
        assert!(c.content_hash() == ref_hash(&n));
    }
}

#[test]
fn node_partial_eq_distinguishes_distinct_nodes() {
    let corpus = node_corpus();
    for (i, a) in corpus.iter().enumerate() {
        for (j, b) in corpus.iter().enumerate() {
            // Distinct corpus members are all pairwise unequal; equal indices are equal.
            assert_eq!(i == j, a == b);
        }
    }
}

#[test]
fn datum_clone_and_eq_and_hash_agree_on_shallow_data() {
    let reg = nat_registry();
    // S(S(Z)) with an Exact leaf; compare identity across an independent rebuild + a clone.
    let build = || {
        Datum::new(
            reg.ctor_ref("Nat", 1).unwrap(),
            vec![CoreValue::Data(Datum::new(
                reg.ctor_ref("Nat", 1).unwrap(),
                vec![CoreValue::Data(Datum::new(
                    reg.ctor_ref("Nat", 0).unwrap(),
                    vec![],
                ))],
            ))],
        )
    };
    let a = build();
    let b = build();
    let c = a.clone();
    assert!(
        a == b,
        "independent equal datums compare equal (iterative PartialEq)"
    );
    assert!(a == c, "clone equals original");
    assert!(a.content_hash() == b.content_hash());
    assert!(a.content_hash() == c.content_hash());

    // A different field value ⇒ different hash and inequality.
    let d = Datum::new(
        reg.ctor_ref("Nat", 1).unwrap(),
        vec![CoreValue::Repr(byte([true; 8]))],
    );
    assert!(a != d);
    assert!(a.content_hash() != d.content_hash());
}

// --------------------------------------------------------------------------------------------
// 2. Depth-safety witnesses (the SIGABRT killer / mutation-witness)
// --------------------------------------------------------------------------------------------

/// Depth at which a deeply-**nested-binder** term is hashed. `content_hash` on nested binders is
/// O(depth²) — the de Bruijn resolution linear-scans a binder scope that grows with nesting depth
/// (a **pre-existing** characteristic: the former recursive `Canon::node` had the same complexity but
/// `SIGABRT`ed before paying it; RFC-0041 §4.5 is about recursion-safety, not this scan cost — see
/// the leaf report's FLAG). Kept small so the O(depth²) hash stays quick while still overflowing a
/// recursive stack (~a few thousand frames).
const BINDER_DEEP: usize = 12_000;

/// A left-nested `Let` spine of `depth` links (each introduces a binder) over a shared free-var
/// leaf — built iteratively.
fn deep_let(depth: usize) -> Node {
    let mut n = Node::Var("leaf".to_owned());
    for _ in 0..depth {
        n = Node::Let {
            id: "a".to_owned(),
            bound: Box::new(Node::Var("b".to_owned())),
            body: Box::new(n),
        };
    }
    n
}

/// A left-nested `App` spine of `depth` links (**no binders**, so hashing is O(depth)) over a
/// free-var leaf — the binder-free deep spine for the large-depth hash witness.
fn deep_app(depth: usize) -> Node {
    let mut n = Node::Var("leaf".to_owned());
    for _ in 0..depth {
        n = Node::App {
            func: Box::new(n),
            arg: Box::new(Node::Var("arg".to_owned())),
        };
    }
    n
}

/// A `S(S(S(… x)))` chain of `depth` nested `Datum`s whose innermost field is a representation
/// `Value` — exercising the shared `Datum ↔ CoreValue ↔ Value` teardown/traversal worklist across
/// every type hop. Returned as a `Datum`.
fn deep_datum(reg: &DataRegistry, depth: usize) -> Datum {
    let s = reg.ctor_ref("Nat", 1).expect("S");
    let mut cur = CoreValue::Repr(byte([true, false, true, false, true, false, true, false]));
    for _ in 0..depth {
        cur = CoreValue::Data(Datum::new(s.clone(), vec![cur]));
    }
    match cur {
        CoreValue::Data(d) => d,
        CoreValue::Repr(_) => unreachable!("depth >= 1 wraps in a Datum"),
    }
}

#[test]
fn deep_node_constructs_and_destructs_without_sigabrt() {
    // Build + drop a DEEP spine. A recursive `Drop` would overflow the stack here (SIGABRT).
    let deep = deep_let(DEEP);
    drop(deep);
}

#[test]
fn deep_node_clones_and_is_equal_without_sigabrt() {
    // Clone + structural-eq of a DEEP 2-child (`Let`) spine — both O(depth); a recursive `Clone` or
    // `PartialEq` would SIGABRT. (Hashing this binder-nested spine is O(depth²) and is witnessed
    // separately at BINDER_DEEP; here we exercise the clone reassembly + eq at full DEEP.)
    let deep = deep_let(DEEP);
    let cloned = deep.clone(); // recursive Clone would SIGABRT
    assert!(deep == cloned); // recursive PartialEq would SIGABRT
                             // Both shells drop iteratively on the way out.
}

#[test]
fn deep_binder_free_node_hashes_without_sigabrt() {
    // A DEEP binder-free (`App`) spine hashes in O(depth); a recursive `Canon::node` would SIGABRT.
    let deep = deep_app(DEEP);
    let cloned = deep.clone();
    assert!(deep == cloned);
    assert!(deep.content_hash() == cloned.content_hash());
}

#[test]
fn deep_binder_nested_term_hashes_without_sigabrt() {
    // Witness that a deeply-*nested-binder* term (which grows the de Bruijn scope) hashes without
    // overflowing the stack — at BINDER_DEEP, which still overflows a recursive `Canon::node` but
    // keeps the O(depth²) scan quick. The clone hashes identically (identity preserved).
    let deep = deep_let(BINDER_DEEP);
    let cloned = deep.clone();
    assert!(deep.content_hash() == cloned.content_hash());
}

#[test]
fn deep_node_unwind_drops_without_double_fault() {
    // A panic while a DEEP node is a live local unwinds ~DEEP-deep, dropping it. A recursive drop
    // during unwind would overflow → abort (the double-panic SIGABRT we set out to kill).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _deep = deep_let(DEEP);
        panic!("intentional panic while a deep node is live");
    }));
    assert!(result.is_err(), "the panic must be caught, not aborted");
}

#[test]
fn deep_datum_constructs_and_destructs_without_sigabrt() {
    let reg = nat_registry();
    let deep = deep_datum(&reg, DEEP);
    drop(deep);
}

#[test]
fn deep_datum_clones_hashes_and_is_equal_without_sigabrt() {
    let reg = nat_registry();
    let deep = deep_datum(&reg, DEEP);
    let cloned = deep.clone(); // recursive Datum::clone would SIGABRT
    assert!(deep == cloned); // recursive Datum::PartialEq would SIGABRT
    assert!(deep.content_hash() == cloned.content_hash()); // recursive Canon::datum would SIGABRT
}

#[test]
fn deep_core_value_clone_and_eq_without_sigabrt() {
    // Exercise the CoreValue entry points (which delegate into the shared iterative cluster ops).
    let reg = nat_registry();
    let deep = CoreValue::Data(deep_datum(&reg, DEEP));
    let cloned = deep.clone();
    assert!(deep == cloned);
    assert!(deep.content_hash() == cloned.content_hash());
}

#[test]
fn deep_datum_unwind_drops_without_double_fault() {
    let reg = nat_registry();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _deep = deep_datum(&reg, DEEP);
        panic!("intentional panic while a deep datum is live");
    }));
    assert!(result.is_err(), "the panic must be caught, not aborted");
}
