//! The Core IR node grammar (RFC-0001 §4.5).
//!
//! This commits the **core subset** of `SPECIFICATION.md` §10.2; the full term language
//! (abstraction, application, recursion, modules) is layered above this and is a later RFC.
//!
//! ```ebnf
//! Node ::= Const { value: Value }
//!        | Var   { id: VarId }
//!        | Let   { id: VarId, bound: Node, body: Node }
//!        | Op    { prim: Prim, args: [Node] }          (* paradigm-specific primitive *)
//!        | Swap  { src: Node, target: Repr, policy: PolicyRef }  (* the ONLY Repr-changing node *)
//!        | Construct { ctor: CtorRef, args: [Node] }            (* NEW (r3): saturated, W6 *)
//!        | Match { scrutinee: Node, alts: [Alt], default: Option<Node> } (* NEW (r3): flat, W7 *)
//! ```
//!
//! Well-formedness (RFC-0001 §4.5): **WF1** every change of a value's `Repr` is a [`Node::Swap`];
//! **WF2** every [`Node::Swap`] carries a [`PolicyRef`] — enforced *by construction* here, since
//! the `policy` field is mandatory. The r3 nodes carry **WF6** (`Construct` saturation), **WF7**
//! (flat, checked-exhaustive `Match`), and **WF8** (no `Swap` introduced through a `Match`/`Construct`
//! elaboration); WF6/WF7 coverage is *checked* above the kernel (the M-320 usefulness analysis +
//! the data registry [`crate::data::DataRegistry`]), never assumed here (RFC-0011 §4.3).

use crate::data::CtorRef;
use crate::id::ContentHash;
use crate::repr::Repr;
use crate::value::Value;

/// A variable identifier (a name; not part of content identity — RFC-0001 §4.6).
pub type VarId = String;
/// A primitive operator name; each declares its operand/result paradigms (RFC-0001 §4.5).
pub type Prim = String;
/// A reference to the selection policy a swap used (RFC-0005), as a content hash.
pub type PolicyRef = ContentHash;

/// A Core IR node.
///
/// **Recursion-safety (RFC-0041 §4.5, W3):** [`Clone`], [`PartialEq`], and [`Drop`] are **manual,
/// iterative** implementations (below the `impl Node` block), *not* `#[derive]`d — a derived
/// (recursive) `Drop`/`Clone`/`PartialEq` overflows the native stack (`SIGABRT`, which violates the
/// never-silent rule G2) on a deeply-nested node spine (`Let`/`App`/`Fix`/`Construct`/`Match`
/// chains), even on the caller's ~2 MB stack outside any deep-stack worker. Only `Debug` stays
/// derived. The content hash ([`Node::content_hash`] → `Canon::node`) is likewise iterative (see
/// `content.rs`). This is a §6 within-freeze behavior-preserving hardening edit: no observable
/// value/order change — clone/eq/hash are bit-identical to the derived forms (mutation-witnessed).
#[derive(Debug)]
pub enum Node {
    /// A constant value.
    Const(Value),
    /// A variable reference.
    Var(VarId),
    /// A let binding.
    Let {
        /// Bound name.
        id: VarId,
        /// The bound expression.
        bound: Box<Node>,
        /// The body in which `id` is in scope.
        body: Box<Node>,
    },
    /// A paradigm-specific primitive application.
    Op {
        /// The primitive.
        prim: Prim,
        /// Operands.
        args: Vec<Node>,
    },
    /// The only node that changes a value's representation; always carries a policy (WF1/WF2).
    Swap {
        /// The value being converted.
        src: Box<Node>,
        /// The target representation.
        target: Repr,
        /// The policy that chose/justified the swap.
        policy: PolicyRef,
    },
    /// A saturated constructor application (RFC-0011 §4.1, r3): builds a data value. SC-3-transparent
    /// (Repr-transparent — no `Swap`). `args.len()` must equal the constructor's field count (WF6);
    /// saturation is *checked* against the data registry above the kernel.
    Construct {
        /// The constructor (`#T#i`).
        ctor: CtorRef,
        /// The field expressions, in declaration order (saturated, WF6).
        args: Vec<Node>,
    },
    /// A **flat** pattern match (RFC-0011 §4.1, r3): one scrutinee, single-level constructor/literal
    /// alternatives, at most one default. Coverage is *checked* (WF7), never assumed; the Maranget
    /// decision tree that lowers nested surface patterns to this flat form stays an untrusted
    /// artifact above the kernel (RFC-0011 §4.4).
    Match {
        /// The value being scrutinised.
        scrutinee: Box<Node>,
        /// The alternatives, tried first-match, left-to-right.
        alts: Vec<Alt>,
        /// The catch-all branch, taken when no alternative matches.
        default: Option<Box<Node>>,
    },
    /// A **lambda abstraction** (RFC-0001 r4; RFC-0007 §4.1 `Lam`): binds one `param` in `body`. A
    /// `Lam` node *is* a function value (a normal form). The v0 surface is first-order, so an
    /// elaborated `Lam` is **closed** (free only in `param` + global/`Fix` names) — no captured
    /// environment (RFC-0007 §4.7: recursion is through definitions, never heap closures). Multiple
    /// arguments are curried (`λx. λy. …`). The param *type* is checked above the kernel and is **not**
    /// an L0 node field (like `Let`, the post-typecheck core is untyped — identity is structural).
    Lam {
        /// The bound parameter.
        param: VarId,
        /// The body, in scope of `param`.
        body: Box<Node>,
    },
    /// **Application** (RFC-0001 r4; RFC-0007 §4.1 `App`): apply `func` to `arg`, call-by-value. A
    /// saturated multi-arg call is a left-nested chain `App(App(f, a), b)`.
    App {
        /// The function being applied (reduces to a `Lam`).
        func: Box<Node>,
        /// The argument.
        arg: Box<Node>,
    },
    /// **General recursion** (RFC-0001 r4; RFC-0007 §4.1 `Fix`; R7-Q1 — a node, not a recursive-`Let`
    /// flag). `Fix{name, body}` binds `name` to the whole `Fix` in `body` (self-reference), and
    /// unfolds by substitution — `Fix(f, e) ⟶ e[f ↦ Fix(f, e)]` — under the interpreter's fuel clock
    /// (so a non-productive recursion is an explicit budget exhaustion, never a hang; RFC-0007 §4.5,
    /// CakeML). Mutual-recursion `Fix` *groups* are deferred to a later step (R7-Q3); v0 elaborates
    /// only self-recursion.
    Fix {
        /// The self-reference name bound in `body`.
        name: VarId,
        /// The recursive body (typically a `Lam`).
        body: Box<Node>,
    },
    /// **Mutual recursion** — a binding group (RFC-0001 r5; R7-Q3). `FixGroup{defs, body}` binds every
    /// `defs[i].0` to `defs[i].1` **simultaneously**: each definition *and* `body` see all the group's
    /// names, so two functions can call each other. It is the n-way generalisation of [`Node::Fix`]
    /// (the n=1 self-recursion case); the elaborator emits a `FixGroup` only for a strongly-connected
    /// call group of **≥2** functions and leaves direct self-recursion on `Fix` (so the simpler node's
    /// semantics and tests are untouched). Like `Fix`, it unfolds by substitution under the
    /// interpreter's fuel clock — never a hang. Reduction has two cases (mirroring `Fix`'s single
    /// unfold): a **focus** `FixGroup(defs, fᵢ)` unfolds to `defs[i][fⱼ ↦ FixGroup(defs, fⱼ)]` (the
    /// member's definition with the group re-bound), and a **continuation** `FixGroup(defs, e)` (with
    /// `e` not a bare member name) unfolds to `e[fⱼ ↦ FixGroup(defs, fⱼ)]`. The member names are
    /// `%`-fresh (no surface capture), and the group **binds** all of them — substitution shadows
    /// them, so the per-member focus thunks stay intact across the unfold.
    FixGroup {
        /// The mutually-recursive bindings `(name, definition)`, each typically a `Lam`. Order is the
        /// elaborator's callee-stable order; identity is over the α-normalised group (content hash).
        defs: Vec<(VarId, Box<Node>)>,
        /// The continuation, in scope of every bound name in `defs`.
        body: Box<Node>,
    },
}

/// One alternative of a flat [`Node::Match`] (RFC-0011 §4.1): a constructor arm (binding exactly the
/// constructor's arity) or a literal arm (over the non-enumerated `Binary{n}`/`Ternary{m}` domain).
#[derive(Debug, Clone, PartialEq)]
pub enum Alt {
    /// A constructor arm: matches a data value of constructor `ctor`, binding its fields to `binders`
    /// (exactly the constructor's arity — WF7), left-to-right.
    Ctor {
        /// The constructor matched (`#T#i`).
        ctor: CtorRef,
        /// The field binders, in declaration order (length == the constructor's arity).
        binders: Vec<VarId>,
        /// The arm body, in scope of `binders`.
        body: Node,
    },
    /// A literal arm: matches a representation value equal (repr + payload) to `value`. Because the
    /// `Binary{n}`/`Ternary{m}` domain is not enumerated, a `Match` carrying literal arms must also
    /// carry a `default` (WF7) — checked above the kernel.
    Lit {
        /// The literal value to match (a `Binary{n}`/`Ternary{m}` constant).
        value: Value,
        /// The arm body.
        body: Node,
    },
}

impl Node {
    /// Whether this node is the (only) representation-changing node, [`Node::Swap`] (WF1).
    #[must_use]
    pub fn is_repr_changing(&self) -> bool {
        matches!(self, Node::Swap { .. })
    }

    /// Whether this whole node is in the **AOT-lowerable** fragment — i.e. it lowers to ANF and runs
    /// on the AOT path. As of M-342 (RFC-0011 §4.4 Q5 closed) the AOT `aot::run` env-machine covers
    /// the **whole v0 calculus** — `Const/Var/Let/Op/Swap` *and* the r3/r4 data + recursion nodes
    /// (`Construct`/`Match`/`Lam`/`App`/`Fix`) — so every well-formed v0 node is AOT-lowerable, and the
    /// three-way differential (L1-eval ≡ L0-interp ≡ AOT) spans the full calculus. (The *native LLVM*
    /// backend stays the bit/trit subset and refuses the rest with an explicit `UnsupportedNode`,
    /// VR-5; that refusal lives in `mycelium-mlir::llvm`, not here.) Retained as the structural
    /// predicate; it is now total over the v0 node set.
    #[must_use]
    pub fn is_aot_lowerable(&self) -> bool {
        match self {
            Node::Const(_) | Node::Var(_) => true,
            Node::Let { bound, body, .. } => bound.is_aot_lowerable() && body.is_aot_lowerable(),
            Node::Op { args, .. } => args.iter().all(Node::is_aot_lowerable),
            Node::Swap { src, .. } => src.is_aot_lowerable(),
            Node::Construct { args, .. } => args.iter().all(Node::is_aot_lowerable),
            Node::Match {
                scrutinee,
                alts,
                default,
            } => {
                scrutinee.is_aot_lowerable()
                    && alts.iter().all(|a| match a {
                        Alt::Ctor { body, .. } | Alt::Lit { body, .. } => body.is_aot_lowerable(),
                    })
                    && default.as_deref().is_none_or(Node::is_aot_lowerable)
            }
            Node::Lam { body, .. } | Node::Fix { body, .. } => body.is_aot_lowerable(),
            Node::FixGroup { defs, body } => {
                defs.iter().all(|(_, d)| d.is_aot_lowerable()) && body.is_aot_lowerable()
            }
            Node::App { func, arg } => func.is_aot_lowerable() && arg.is_aot_lowerable(),
        }
    }
}

// ---------------------------------------------------------------------------
// Iterative, recursion-safe Drop / Clone / PartialEq (RFC-0041 §4.5, W3).
//
// `Node` is a `Box`-owned, acyclic tree (no `Rc`/`Arc` on the spine, no shared substructure), so
// each node is owned by exactly one parent. Under that invariant an explicit-worklist traversal is
// double-free-safe: every node is visited (and, for `Drop`, freed) exactly once. **Recorded
// precondition (RFC-0041 §4.5, Low freeze11): should a future interning/DAG cache put `Rc`/`Arc` on
// the node spine, these iterative `Drop`s would double-free and MUST be revisited.** It holds today
// by construction — the field types are `Box<Node>` / `Vec<Node>` only.
//
// `#![forbid(unsafe_code)]` (lib.rs) still holds: the worklists use only safe `Box`/`Vec` +
// `std::mem::{replace, take}` take-loops; there is no `next`-pointer trick that needs `unsafe`.
// ---------------------------------------------------------------------------

/// A cheap, allocation-free placeholder swapped into a `Box<Node>` slot while its owned child is
/// moved onto the worklist (`impl Drop` forbids by-value field move-out — E0509 — so every owned
/// `Box<Node>` destructure is a by-ref `mem::replace`). `Var(String::new())` is a leaf and
/// `String::new()` does not allocate.
#[inline]
fn drop_placeholder() -> Node {
    Node::Var(String::new())
}

/// Move every direct **`Node` child** of `n` onto `work`, leaving `n` a childless shell (recursive
/// `Box` slots replaced by [`drop_placeholder`], `Vec`s drained). Contained `Value`s are left to
/// drop in place — a `Value` is bounded-depth by construction (its nested `Seq`/`Repr` are
/// construction-gated; RFC-0041 §4.5 W3 note), so it is not a deep-recursion vector here.
fn detach_node_children(n: &mut Node, work: &mut Vec<Node>) {
    match n {
        Node::Const(_) | Node::Var(_) => {}
        Node::Let { bound, body, .. } => {
            work.push(std::mem::replace(bound.as_mut(), drop_placeholder()));
            work.push(std::mem::replace(body.as_mut(), drop_placeholder()));
        }
        Node::Op { args, .. } | Node::Construct { args, .. } => work.append(args),
        Node::Swap { src, .. } => {
            work.push(std::mem::replace(src.as_mut(), drop_placeholder()));
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            work.push(std::mem::replace(scrutinee.as_mut(), drop_placeholder()));
            // We own the taken `Vec<Alt>`, so its bodies may be moved out by value (no E0509 — the
            // taken vector is a local, not a field of `self`). The alt's non-`Node` fields
            // (`ctor`/`binders`/`value`) drop shallowly when the loop binding goes out of scope.
            for alt in std::mem::take(alts) {
                match alt {
                    Alt::Ctor { body, .. } | Alt::Lit { body, .. } => work.push(body),
                }
            }
            if let Some(d) = default.take() {
                work.push(*d);
            }
        }
        Node::Lam { body, .. } | Node::Fix { body, .. } => {
            work.push(std::mem::replace(body.as_mut(), drop_placeholder()));
        }
        Node::App { func, arg } => {
            work.push(std::mem::replace(func.as_mut(), drop_placeholder()));
            work.push(std::mem::replace(arg.as_mut(), drop_placeholder()));
        }
        Node::FixGroup { defs, body } => {
            for (_, b) in std::mem::take(defs) {
                work.push(*b);
            }
            work.push(std::mem::replace(body.as_mut(), drop_placeholder()));
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Flatten the owned subtree onto an explicit worklist and drop each node as a childless
        // shell — bounded native-stack use regardless of spine depth (RFC-0041 §4.5).
        //
        // Allocation honesty (RFC-0041 §4.5 asks for no allocation during `Drop`): a fully
        // alloc-free iterative drop of a *heterogeneous, non-intrusive* tree in **safe** Rust is not
        // achievable here — `Node` has no spare `next` field to thread an intrusive stack through,
        // and `Drop::drop(&mut self)` cannot be handed a preallocated scratch buffer; the only
        // alloc-free options (an added `next` field, or `unsafe` pointer-reversal) are both barred
        // (a new field is out of the §6 hardening scope; `unsafe` is `forbid`den). This `Vec`
        // worklist starts empty (`Vec::new` does not allocate) and grows only when the spine is
        // actually deep — precisely the case that previously *guaranteed* a multi-MB stack-overflow
        // `SIGABRT`. So the change strictly trades a certain abort for a small pointer-vector
        // allocation that only fails under genuine OOM. (FLAGged up for the integrator.)
        let mut work: Vec<Node> = Vec::new();
        detach_node_children(self, &mut work);
        while let Some(mut n) = work.pop() {
            detach_node_children(&mut n, &mut work);
            // `n` is now a childless shell; dropping it re-enters this `Drop`, but on an
            // already-emptied node — bounded O(1) reentrancy (the placeholders are `Var` leaves),
            // never a deep recursion.
        }
    }
}

impl Clone for Node {
    fn clone(&self) -> Node {
        // Iterative deep clone via an explicit expand/assemble worklist + a value stack, so the
        // front-door `let mut current = node.clone()` no longer `SIGABRT`s on a deep spine
        // (RFC-0041 §4.5). Convention: `Assemble` is pushed first, then each recursive child is
        // `Expand`ed in **forward** order — so `done.pop()` yields children first-to-last.
        enum AltMeta {
            Ctor { ctor: CtorRef, binders: Vec<VarId> },
            Lit { value: Value },
        }
        enum Frame {
            Let {
                id: VarId,
            },
            Op {
                prim: Prim,
                arity: usize,
            },
            Swap {
                target: Repr,
                policy: PolicyRef,
            },
            Construct {
                ctor: CtorRef,
                arity: usize,
            },
            Match {
                metas: Vec<AltMeta>,
                has_default: bool,
            },
            Lam {
                param: VarId,
            },
            App,
            Fix {
                name: VarId,
            },
            FixGroup {
                names: Vec<VarId>,
            },
        }
        enum Task<'a> {
            Expand(&'a Node),
            Assemble(Frame),
        }

        let mut tasks: Vec<Task<'_>> = vec![Task::Expand(self)];
        let mut done: Vec<Node> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Expand(node) => match node {
                    // Leaves clone directly. `Value`/`Repr` are bounded-depth by construction, so
                    // their derived `Clone` is not a deep vector here.
                    Node::Const(v) => done.push(Node::Const(v.clone())),
                    Node::Var(s) => done.push(Node::Var(s.clone())),
                    Node::Let { id, bound, body } => {
                        tasks.push(Task::Assemble(Frame::Let { id: id.clone() }));
                        tasks.push(Task::Expand(bound));
                        tasks.push(Task::Expand(body));
                    }
                    Node::Op { prim, args } => {
                        tasks.push(Task::Assemble(Frame::Op {
                            prim: prim.clone(),
                            arity: args.len(),
                        }));
                        for a in args {
                            tasks.push(Task::Expand(a));
                        }
                    }
                    Node::Swap {
                        src,
                        target,
                        policy,
                    } => {
                        tasks.push(Task::Assemble(Frame::Swap {
                            target: target.clone(),
                            policy: policy.clone(),
                        }));
                        tasks.push(Task::Expand(src));
                    }
                    Node::Construct { ctor, args } => {
                        tasks.push(Task::Assemble(Frame::Construct {
                            ctor: ctor.clone(),
                            arity: args.len(),
                        }));
                        for a in args {
                            tasks.push(Task::Expand(a));
                        }
                    }
                    Node::Match {
                        scrutinee,
                        alts,
                        default,
                    } => {
                        let metas = alts
                            .iter()
                            .map(|a| match a {
                                Alt::Ctor { ctor, binders, .. } => AltMeta::Ctor {
                                    ctor: ctor.clone(),
                                    binders: binders.clone(),
                                },
                                Alt::Lit { value, .. } => AltMeta::Lit {
                                    value: value.clone(),
                                },
                            })
                            .collect();
                        tasks.push(Task::Assemble(Frame::Match {
                            metas,
                            has_default: default.is_some(),
                        }));
                        tasks.push(Task::Expand(scrutinee));
                        for a in alts {
                            match a {
                                Alt::Ctor { body, .. } | Alt::Lit { body, .. } => {
                                    tasks.push(Task::Expand(body));
                                }
                            }
                        }
                        if let Some(d) = default {
                            tasks.push(Task::Expand(d));
                        }
                    }
                    Node::Lam { param, body } => {
                        tasks.push(Task::Assemble(Frame::Lam {
                            param: param.clone(),
                        }));
                        tasks.push(Task::Expand(body));
                    }
                    Node::App { func, arg } => {
                        tasks.push(Task::Assemble(Frame::App));
                        tasks.push(Task::Expand(func));
                        tasks.push(Task::Expand(arg));
                    }
                    Node::Fix { name, body } => {
                        tasks.push(Task::Assemble(Frame::Fix { name: name.clone() }));
                        tasks.push(Task::Expand(body));
                    }
                    Node::FixGroup { defs, body } => {
                        tasks.push(Task::Assemble(Frame::FixGroup {
                            names: defs.iter().map(|(n, _)| n.clone()).collect(),
                        }));
                        for (_, d) in defs {
                            tasks.push(Task::Expand(d));
                        }
                        tasks.push(Task::Expand(body));
                    }
                },
                Task::Assemble(frame) => {
                    // `done.pop()` returns children in forward order (see the push convention above).
                    let node = match frame {
                        Frame::Let { id } => {
                            let bound = done.pop().expect("clone: Let bound");
                            let body = done.pop().expect("clone: Let body");
                            Node::Let {
                                id,
                                bound: Box::new(bound),
                                body: Box::new(body),
                            }
                        }
                        Frame::Op { prim, arity } => {
                            let mut args = Vec::with_capacity(arity);
                            for _ in 0..arity {
                                args.push(done.pop().expect("clone: Op arg"));
                            }
                            Node::Op { prim, args }
                        }
                        Frame::Swap { target, policy } => {
                            let src = done.pop().expect("clone: Swap src");
                            Node::Swap {
                                src: Box::new(src),
                                target,
                                policy,
                            }
                        }
                        Frame::Construct { ctor, arity } => {
                            let mut args = Vec::with_capacity(arity);
                            for _ in 0..arity {
                                args.push(done.pop().expect("clone: Construct arg"));
                            }
                            Node::Construct { ctor, args }
                        }
                        Frame::Match { metas, has_default } => {
                            let scrutinee = done.pop().expect("clone: Match scrutinee");
                            let mut bodies = Vec::with_capacity(metas.len());
                            for _ in 0..metas.len() {
                                bodies.push(done.pop().expect("clone: Match alt body"));
                            }
                            let alts = metas
                                .into_iter()
                                .zip(bodies)
                                .map(|(m, body)| match m {
                                    AltMeta::Ctor { ctor, binders } => Alt::Ctor {
                                        ctor,
                                        binders,
                                        body,
                                    },
                                    AltMeta::Lit { value } => Alt::Lit { value, body },
                                })
                                .collect();
                            let default = if has_default {
                                Some(Box::new(done.pop().expect("clone: Match default")))
                            } else {
                                None
                            };
                            Node::Match {
                                scrutinee: Box::new(scrutinee),
                                alts,
                                default,
                            }
                        }
                        Frame::Lam { param } => {
                            let body = done.pop().expect("clone: Lam body");
                            Node::Lam {
                                param,
                                body: Box::new(body),
                            }
                        }
                        Frame::App => {
                            let func = done.pop().expect("clone: App func");
                            let arg = done.pop().expect("clone: App arg");
                            Node::App {
                                func: Box::new(func),
                                arg: Box::new(arg),
                            }
                        }
                        Frame::Fix { name } => {
                            let body = done.pop().expect("clone: Fix body");
                            Node::Fix {
                                name,
                                body: Box::new(body),
                            }
                        }
                        Frame::FixGroup { names } => {
                            let mut dbodies = Vec::with_capacity(names.len());
                            for _ in 0..names.len() {
                                dbodies.push(done.pop().expect("clone: FixGroup def body"));
                            }
                            let body = done.pop().expect("clone: FixGroup body");
                            let defs = names
                                .into_iter()
                                .zip(dbodies)
                                .map(|(n, d)| (n, Box::new(d)))
                                .collect();
                            Node::FixGroup {
                                defs,
                                body: Box::new(body),
                            }
                        }
                    };
                    done.push(node);
                }
            }
        }
        done.pop().expect("clone: exactly one root node remains")
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        // Iterative structural equality via a pair worklist. Result-identical to the derived
        // (recursive) `PartialEq` — every field is compared and the first mismatch short-circuits;
        // *which* mismatch is found first may differ, but the boolean result cannot (RFC-0041 §4.5,
        // bar (a): observably identical). Bounded native stack regardless of spine depth.
        let mut stack: Vec<(&Node, &Node)> = vec![(self, other)];
        while let Some((a, b)) = stack.pop() {
            match (a, b) {
                (Node::Const(x), Node::Const(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Node::Var(x), Node::Var(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (
                    Node::Let {
                        id: i1,
                        bound: b1,
                        body: y1,
                    },
                    Node::Let {
                        id: i2,
                        bound: b2,
                        body: y2,
                    },
                ) => {
                    if i1 != i2 {
                        return false;
                    }
                    stack.push((b1, b2));
                    stack.push((y1, y2));
                }
                (Node::Op { prim: p1, args: a1 }, Node::Op { prim: p2, args: a2 }) => {
                    if p1 != p2 || a1.len() != a2.len() {
                        return false;
                    }
                    for (x, y) in a1.iter().zip(a2) {
                        stack.push((x, y));
                    }
                }
                (
                    Node::Swap {
                        src: s1,
                        target: t1,
                        policy: pol1,
                    },
                    Node::Swap {
                        src: s2,
                        target: t2,
                        policy: pol2,
                    },
                ) => {
                    if t1 != t2 || pol1 != pol2 {
                        return false;
                    }
                    stack.push((s1, s2));
                }
                (
                    Node::Construct { ctor: c1, args: a1 },
                    Node::Construct { ctor: c2, args: a2 },
                ) => {
                    if c1 != c2 || a1.len() != a2.len() {
                        return false;
                    }
                    for (x, y) in a1.iter().zip(a2) {
                        stack.push((x, y));
                    }
                }
                (
                    Node::Match {
                        scrutinee: s1,
                        alts: al1,
                        default: d1,
                    },
                    Node::Match {
                        scrutinee: s2,
                        alts: al2,
                        default: d2,
                    },
                ) => {
                    if al1.len() != al2.len() {
                        return false;
                    }
                    stack.push((s1, s2));
                    for (x, y) in al1.iter().zip(al2) {
                        match (x, y) {
                            (
                                Alt::Ctor {
                                    ctor: c1,
                                    binders: bd1,
                                    body: bo1,
                                },
                                Alt::Ctor {
                                    ctor: c2,
                                    binders: bd2,
                                    body: bo2,
                                },
                            ) => {
                                if c1 != c2 || bd1 != bd2 {
                                    return false;
                                }
                                stack.push((bo1, bo2));
                            }
                            (
                                Alt::Lit {
                                    value: v1,
                                    body: bo1,
                                },
                                Alt::Lit {
                                    value: v2,
                                    body: bo2,
                                },
                            ) => {
                                if v1 != v2 {
                                    return false;
                                }
                                stack.push((bo1, bo2));
                            }
                            _ => return false,
                        }
                    }
                    match (d1, d2) {
                        (None, None) => {}
                        (Some(x), Some(y)) => stack.push((x, y)),
                        _ => return false,
                    }
                }
                (
                    Node::Lam {
                        param: p1,
                        body: b1,
                    },
                    Node::Lam {
                        param: p2,
                        body: b2,
                    },
                ) => {
                    if p1 != p2 {
                        return false;
                    }
                    stack.push((b1, b2));
                }
                (Node::App { func: f1, arg: a1 }, Node::App { func: f2, arg: a2 }) => {
                    stack.push((f1, f2));
                    stack.push((a1, a2));
                }
                (Node::Fix { name: n1, body: b1 }, Node::Fix { name: n2, body: b2 }) => {
                    if n1 != n2 {
                        return false;
                    }
                    stack.push((b1, b2));
                }
                (Node::FixGroup { defs: d1, body: b1 }, Node::FixGroup { defs: d2, body: b2 }) => {
                    if d1.len() != d2.len() {
                        return false;
                    }
                    for ((n1, x), (n2, y)) in d1.iter().zip(d2) {
                        if n1 != n2 {
                            return false;
                        }
                        stack.push((x, y));
                    }
                    stack.push((b1, b2));
                }
                // Different variants are unequal.
                _ => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::{Meta, Provenance};
    use crate::value::{Payload, Value};

    fn byte() -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, true, false, false, true, false]),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed byte")
    }

    #[test]
    fn builds_a_let_with_a_swap() {
        // let a = 0b1011_0010 in swap(a, to: Ternary{6}, policy: ...)
        let policy = ContentHash::parse("policy:round_trip_safe").expect("hash");
        let node = Node::Let {
            id: "a".to_owned(),
            bound: Box::new(Node::Const(byte())),
            body: Box::new(Node::Swap {
                src: Box::new(Node::Var("a".to_owned())),
                target: Repr::Ternary { trits: 6 },
                policy,
            }),
        };
        // `ref body` — `Node` now has a manual `Drop` (RFC-0041 §4.5), so a by-value field
        // move-out of a `Node` is E0509; borrow instead.
        match node {
            Node::Let { ref body, .. } => assert!(body.is_repr_changing()),
            _ => panic!("expected a Let"),
        }
    }

    #[test]
    fn only_swap_changes_repr() {
        assert!(!Node::Var("x".to_owned()).is_repr_changing());
        assert!(!Node::Op {
            prim: "add_binary".to_owned(),
            args: vec![],
        }
        .is_repr_changing());
    }

    // Mutant-witnesses for Node::is_aot_lowerable (node.rs:183:9 true/false replacements and the
    // structural && → || mutations at lines 185, 195, 198, 202, 204):
    //
    // The function is documented as "now total over the v0 node set" — every well-formed v0 node
    // is AOT-lowerable. Accordingly, the `&&` → `||` mutations at each structural arm are
    // **equivalent** (both sides of `&&` are always true, so `||` produces the same result).
    // The whole-function replacements (`→ true` / `→ false`) ARE killable: replacing with
    // `false` would make every node fail, which these tests detect.
    //
    // These tests also cover every node variant so that any future variant that _isn't_ always
    // lowerable would be detected if the predicate is strengthened later.
    #[test]
    fn is_aot_lowerable_is_total_over_v0_nodes() {
        let policy = ContentHash::parse("policy:round_trip_safe").expect("hash");
        let v = byte();
        // Leaf nodes.
        assert!(Node::Const(v.clone()).is_aot_lowerable());
        assert!(Node::Var("x".to_owned()).is_aot_lowerable());

        // Let — both bound and body must be lowerable (line 185 && mutant).
        let let_node = Node::Let {
            id: "a".to_owned(),
            bound: Box::new(Node::Const(v.clone())),
            body: Box::new(Node::Var("a".to_owned())),
        };
        assert!(let_node.is_aot_lowerable());

        // Op — args (may be empty or non-empty).
        assert!(Node::Op {
            prim: "bit.not".to_owned(),
            args: vec![]
        }
        .is_aot_lowerable());
        assert!(Node::Op {
            prim: "bit.xor".to_owned(),
            args: vec![Node::Var("x".to_owned()), Node::Var("y".to_owned())],
        }
        .is_aot_lowerable());

        // Swap — src must be lowerable.
        let swap = Node::Swap {
            src: Box::new(Node::Const(v.clone())),
            target: Repr::Ternary { trits: 6 },
            policy: policy.clone(),
        };
        assert!(swap.is_aot_lowerable());

        // Lam and Fix — body must be lowerable.
        let lam = Node::Lam {
            param: "x".to_owned(),
            body: Box::new(Node::Var("x".to_owned())),
        };
        assert!(lam.is_aot_lowerable());
        let fix = Node::Fix {
            name: "f".to_owned(),
            body: Box::new(Node::Var("f".to_owned())),
        };
        assert!(fix.is_aot_lowerable());

        // App — both func and arg must be lowerable (line 204 && mutant).
        let app = Node::App {
            func: Box::new(Node::Var("f".to_owned())),
            arg: Box::new(Node::Const(v.clone())),
        };
        assert!(app.is_aot_lowerable());

        // FixGroup — all defs and body (line 202 && mutant).
        let fixgroup = Node::FixGroup {
            defs: vec![
                ("f".to_owned(), Box::new(Node::Var("g".to_owned()))),
                ("g".to_owned(), Box::new(Node::Var("f".to_owned()))),
            ],
            body: Box::new(Node::Var("f".to_owned())),
        };
        assert!(fixgroup.is_aot_lowerable());

        // Match — scrutinee, all alt bodies, and optional default (lines 195, 198 && mutants).
        use crate::data::{CtorSpec, DataRegistry, DeclSpec};
        use std::collections::BTreeMap;
        let mut m = BTreeMap::new();
        m.insert(
            "Unit".to_owned(),
            DeclSpec {
                ctors: vec![CtorSpec { fields: vec![] }],
            },
        );
        let reg = DataRegistry::build(&m).unwrap();
        let cref = reg.ctor_ref("Unit", 0).unwrap();
        let match_node = Node::Match {
            scrutinee: Box::new(Node::Var("x".to_owned())),
            alts: vec![Alt::Ctor {
                ctor: cref,
                binders: vec![],
                body: Node::Const(v.clone()),
            }],
            default: Some(Box::new(Node::Var("x".to_owned()))),
        };
        assert!(match_node.is_aot_lowerable());
    }
}
