//! Inspectable lowering — `≥2` dumpable/diffable stages (M-112; RFC-0004 §5/§6; SC-4; WF5).
//!
//! The interpreter is the reference semantics (M-110); *lowering* is the backend-agnostic path
//! toward codegen, and its defining property here is **inspectability**: every stage has a
//! canonical textual [`dump`](Stage::text) (deterministic — structurally identical programs render
//! identically, SC-4), each pass **preserves `Meta`** (WF5 — guarantee tags survive, never silently
//! dropped), and the packing decision is an **explicit, recorded** schedule choice (RFC-0004 §5; no
//! hidden layout). [`stages`] returns the pipeline so adjacent stages can be diffed.
//!
//! The two stages shipped:
//! - **`core`** — the Core IR node tree (RFC-0001 §4.5), rendered canonically.
//! - **`substrate`** — an **A-normal form**: nested `Op`/`Swap`/`Let` flattened to a linear list of
//!   named bindings (the classic pre-codegen shape every backend consumes), with each binding whose
//!   result representation is *statically known* (a `Const` or a `Swap` target) annotated with its
//!   **scheduled [`PhysicalLayout`]** (the default schedule, RFC-0004 §5 / DN-01).
//!
//! Layout for `Op` results is intentionally left unannotated: the kernel has no operator-typing yet
//! (a later RFC), so inferring it would be a black box (G2). The omission is explicit, not silent.

use core::fmt::Write as _;

use crate::data::CtorRef;
use crate::meta::PackScheme;
use crate::node::{Alt, Node, VarId};
use crate::repr::{FloatWidth, Repr, ScalarKind, SparsityClass};
use crate::value::{Payload, Trit, Value};
use crate::{GuaranteeStrength, PhysicalLayout};

/// One lowering stage: a name and its canonical, diffable textual dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// Stage name (`"core"`, `"substrate"`).
    pub name: &'static str,
    /// The canonical textual rendering (SC-4: deterministic; structurally identical → identical).
    pub text: String,
}

/// The default schedule-staged packing for a representation (RFC-0004 §5; DN-01). The fixed,
/// enumerable layout set keeps selection tractable (T1.4) — `I2_S` is the lossless ternary default.
///
/// Returns `None` for a representation that has **no scheduled physical layout in the fixed set**
/// — currently [`Repr::Seq`] (RFC-0032 D3): a packing schedule for indexed sequences is not yet
/// designed, so rather than silently mislabel a `Seq` with a scalar layout (a black box, G2), the
/// schedule is *explicitly absent*. The binding is then left layout-unannotated, exactly as an
/// `Op` result whose repr is statically unknown is (the omission is explicit, never silent).
#[must_use]
pub fn schedule(repr: &Repr) -> Option<PhysicalLayout> {
    match repr {
        Repr::Binary { .. } => Some(PhysicalLayout::BinaryWords),
        Repr::Ternary { .. } => Some(PhysicalLayout::TritPacked {
            scheme: PackScheme::I2S,
        }),
        Repr::Dense { .. } => Some(PhysicalLayout::DenseArray),
        Repr::Vsa { sparsity, .. } => Some(PhysicalLayout::VsaStore {
            sparse: matches!(sparsity, SparsityClass::Sparse { .. }),
        }),
        // No designed packing schedule for indexed sequences (RFC-0032 D3), byte strings
        // (RFC-0032 D4), or the scalar float (ADR-040/M-896 — layout scheduling is M-898+
        // territory) in the fixed set yet — explicitly absent, never a silently-wrong scalar
        // layout.
        Repr::Seq { .. } | Repr::Bytes | Repr::Float { .. } => None,
    }
}

/// Run the lowering pipeline, returning every stage in order (currently `core` → `substrate`).
#[must_use]
pub fn stages(node: &Node) -> Vec<Stage> {
    vec![
        Stage {
            name: "core",
            text: dump_core(node),
        },
        Stage {
            name: "substrate",
            text: lower_to_anf(node).dump(),
        },
    ]
}

// --- rendering helpers (shared, canonical) -----------------------------------------------------

fn render_scalar_kind(k: ScalarKind) -> &'static str {
    match k {
        ScalarKind::F16 => "F16",
        ScalarKind::Bf16 => "BF16",
        ScalarKind::F32 => "F32",
        ScalarKind::F64 => "F64",
    }
}

fn render_repr(repr: &Repr) -> String {
    match repr {
        Repr::Binary { width } => format!("Binary{{{width}}}"),
        Repr::Ternary { trits } => format!("Ternary{{{trits}}}"),
        Repr::Dense { dim, dtype } => format!("Dense{{{dim}:{}}}", render_scalar_kind(*dtype)),
        Repr::Vsa {
            model,
            dim,
            sparsity,
        } => {
            let s = match sparsity {
                SparsityClass::Dense => "dense".to_owned(),
                SparsityClass::Sparse { max_active } => format!("sparse≤{max_active}"),
            };
            format!("VSA{{{model}:{dim} {s}}}")
        }
        Repr::Seq { elem, len } => format!("Seq{{{}; {len}}}", render_repr(elem)),
        // The frozen width registry has exactly F64 today (ADR-040 FLAG-1); render it by name so a
        // future width renders distinctly by construction.
        Repr::Float { width } => match width {
            FloatWidth::F64 => "Float{F64}".to_owned(),
        },
        Repr::Bytes => "Bytes".to_owned(),
    }
}

fn render_payload(p: &Payload) -> String {
    match p {
        Payload::Bits(b) => {
            let s: String = b.iter().map(|&x| if x { '1' } else { '0' }).collect();
            format!("bits={s}")
        }
        Payload::Trits(t) => {
            let s: String = t
                .iter()
                .map(|&x| match x {
                    Trit::Neg => '-',
                    Trit::Zero => '0',
                    Trit::Pos => '+',
                })
                .collect();
            format!("trits={s}")
        }
        Payload::Scalars(xs) => format!("scalars={xs:?}"),
        Payload::Hypervector(xs) => format!("hv={xs:?}"),
        // Shortest round-trip decimal (`{:?}`) — deterministic and diffable (SC-4); specials
        // render in-band as `inf`/`-inf`/`NaN` (ADR-040 §2.4).
        Payload::Float(x) => format!("float={x:?}"),
        Payload::Seq(elems) => {
            // Render each element by its repr+payload head, comma-joined inside brackets — diffable
            // and deterministic (SC-4), recursing through the same helpers.
            let inner: Vec<String> = elems
                .iter()
                .map(|e| format!("{} {}", render_repr(e.repr()), render_payload(e.payload())))
                .collect();
            format!("seq=[{}]", inner.join(", "))
        }
        Payload::Bytes(bytes) => {
            // Lowercase-hex byte rendering — deterministic and diffable (SC-4).
            let s: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            format!("bytes={s}")
        }
    }
}

fn render_guarantee(g: GuaranteeStrength) -> &'static str {
    match g {
        GuaranteeStrength::Exact => ":exact",
        GuaranteeStrength::Proven => ":proven",
        GuaranteeStrength::Empirical => ":empirical",
        GuaranteeStrength::Declared => ":declared",
    }
}

/// Render a `Const` value head: `const <repr> <payload> <guarantee>` (Meta-preserving — the
/// guarantee tag is always shown, WF5).
fn render_const(v: &Value) -> String {
    format!(
        "const {} {} {}",
        render_repr(v.repr()),
        render_payload(v.payload()),
        render_guarantee(v.meta().guarantee())
    )
}

fn short_hash(h: &crate::ContentHash) -> String {
    let d = h.digest();
    let head: String = d.chars().take(8).collect();
    format!("{}:{head}", h.algo())
}

// --- Stage 0: canonical Core IR dump -----------------------------------------------------------

/// The canonical, deterministic textual rendering of a Core IR node (the `core` stage). A
/// projection: it does not affect content identity (RFC-0001 §4.6/§4.8), and structurally identical
/// nodes render identically (SC-4). Reused as the basis of the formatter (M-142).
#[must_use]
pub fn dump_node(node: &Node) -> String {
    dump_core(node)
}

fn dump_core(node: &Node) -> String {
    let mut s = String::new();
    write_core(node, 0, &mut s);
    s
}

/// The **canonical formatter** (M-142; RFC-0001 §4.8; ADR-003). Like [`dump_node`] but with binder
/// names **α-normalized** to `v0, v1, …` in binding order, so that definitions differing only in
/// names (a "reformatting") render to *identical* canonical text — and, since names are not part of
/// content identity (RFC-0001 §4.6), that shared canonical form carries one shared
/// [`Node::content_hash`]. Formatting is a projection: it never changes identity.
#[must_use]
pub fn format(node: &Node) -> String {
    let mut s = String::new();
    let mut scope: Vec<(String, String)> = Vec::new();
    let mut counter = 0usize;
    write_canon(node, 0, &mut scope, &mut counter, &mut s);
    s
}

fn write_canon(
    node: &Node,
    depth: usize,
    scope: &mut Vec<(String, String)>,
    counter: &mut usize,
    s: &mut String,
) {
    indent(depth, s);
    match node {
        Node::Const(v) => {
            let _ = writeln!(s, "{}", render_const(v));
        }
        Node::Var(x) => {
            // Innermost-first; a bound var renders as its canonical name, a free var keeps its own.
            match scope.iter().rev().find(|(orig, _)| orig == x) {
                Some((_, canon)) => {
                    let _ = writeln!(s, "var {canon}");
                }
                None => {
                    let _ = writeln!(s, "free {x}");
                }
            }
        }
        Node::Let { id, bound, body } => {
            let canon = format!("v{counter}");
            *counter += 1;
            let _ = writeln!(s, "let {canon} =");
            write_canon(bound, depth + 1, scope, counter, s);
            indent(depth, s);
            let _ = writeln!(s, "in");
            scope.push((id.clone(), canon));
            write_canon(body, depth + 1, scope, counter, s);
            scope.pop();
        }
        Node::Op { prim, args } => {
            let _ = writeln!(s, "op {prim}");
            for a in args {
                write_canon(a, depth + 1, scope, counter, s);
            }
        }
        Node::Swap {
            src,
            target,
            policy,
        } => {
            let _ = writeln!(s, "swap -> {} @{}", render_repr(target), short_hash(policy));
            write_canon(src, depth + 1, scope, counter, s);
        }
        Node::Construct { ctor, args } => {
            let _ = writeln!(s, "construct {ctor}");
            for a in args {
                write_canon(a, depth + 1, scope, counter, s);
            }
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let _ = writeln!(s, "match");
            write_canon(scrutinee, depth + 1, scope, counter, s);
            for alt in alts {
                indent(depth, s);
                match alt {
                    Alt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        // α-normalize the binder names to v0, v1, … in binding order (the canonical
                        // dump never leaks source names — §4.8).
                        let canon: Vec<String> = (0..binders.len())
                            .map(|_| {
                                let c = format!("v{counter}");
                                *counter += 1;
                                c
                            })
                            .collect();
                        let _ = writeln!(s, "alt {ctor} ({})", canon.join(" "));
                        let mark = scope.len();
                        for (orig, c) in binders.iter().zip(&canon) {
                            scope.push((orig.clone(), c.clone()));
                        }
                        write_canon(body, depth + 1, scope, counter, s);
                        scope.truncate(mark);
                    }
                    Alt::Lit { value, body } => {
                        let _ = writeln!(s, "alt-lit {}", render_const(value));
                        write_canon(body, depth + 1, scope, counter, s);
                    }
                }
            }
            indent(depth, s);
            match default {
                Some(d) => {
                    let _ = writeln!(s, "default");
                    write_canon(d, depth + 1, scope, counter, s);
                }
                None => {
                    let _ = writeln!(s, "no-default");
                }
            }
        }
        Node::Lam { param, body } => {
            let canon = format!("v{counter}");
            *counter += 1;
            let _ = writeln!(s, "lam {canon} =>");
            scope.push((param.clone(), canon));
            write_canon(body, depth + 1, scope, counter, s);
            scope.pop();
        }
        Node::App { func, arg } => {
            let _ = writeln!(s, "app");
            write_canon(func, depth + 1, scope, counter, s);
            write_canon(arg, depth + 1, scope, counter, s);
        }
        Node::Fix { name, body } => {
            let canon = format!("v{counter}");
            *counter += 1;
            let _ = writeln!(s, "fix {canon} =>");
            scope.push((name.clone(), canon));
            write_canon(body, depth + 1, scope, counter, s);
            scope.pop();
        }
        Node::FixGroup { defs, body } => {
            let _ = writeln!(s, "fixgroup");
            // α-normalise every member name first — the group binds them all mutually, so each is in
            // scope for every definition and the continuation (the canonical dump never leaks names).
            let mark = scope.len();
            for (name, _) in defs {
                let canon = format!("v{counter}");
                *counter += 1;
                scope.push((name.clone(), canon));
            }
            for (i, (_, def)) in defs.iter().enumerate() {
                let canon = scope[mark + i].1.clone();
                indent(depth + 1, s);
                let _ = writeln!(s, "def {canon} =>");
                write_canon(def, depth + 2, scope, counter, s);
            }
            indent(depth + 1, s);
            let _ = writeln!(s, "in");
            write_canon(body, depth + 1, scope, counter, s);
            scope.truncate(mark);
        }
    }
}

fn indent(depth: usize, s: &mut String) {
    for _ in 0..depth {
        s.push_str("  ");
    }
}

fn write_core(node: &Node, depth: usize, s: &mut String) {
    indent(depth, s);
    match node {
        Node::Const(v) => {
            let _ = writeln!(s, "{}", render_const(v));
        }
        Node::Var(x) => {
            let _ = writeln!(s, "var {x}");
        }
        Node::Let { id, bound, body } => {
            let _ = writeln!(s, "let {id} =");
            write_core(bound, depth + 1, s);
            indent(depth, s);
            let _ = writeln!(s, "in");
            write_core(body, depth + 1, s);
        }
        Node::Op { prim, args } => {
            let _ = writeln!(s, "op {prim}");
            for a in args {
                write_core(a, depth + 1, s);
            }
        }
        Node::Swap {
            src,
            target,
            policy,
        } => {
            let _ = writeln!(s, "swap -> {} @{}", render_repr(target), short_hash(policy));
            write_core(src, depth + 1, s);
        }
        Node::Construct { ctor, args } => {
            let _ = writeln!(s, "construct {ctor}");
            for a in args {
                write_core(a, depth + 1, s);
            }
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let _ = writeln!(s, "match");
            write_core(scrutinee, depth + 1, s);
            for alt in alts {
                indent(depth, s);
                match alt {
                    Alt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        let _ = writeln!(s, "alt {ctor} ({})", binders.join(" "));
                        write_core(body, depth + 1, s);
                    }
                    Alt::Lit { value, body } => {
                        let _ = writeln!(s, "alt-lit {}", render_const(value));
                        write_core(body, depth + 1, s);
                    }
                }
            }
            indent(depth, s);
            match default {
                Some(d) => {
                    let _ = writeln!(s, "default");
                    write_core(d, depth + 1, s);
                }
                None => {
                    let _ = writeln!(s, "no-default");
                }
            }
        }
        Node::Lam { param, body } => {
            let _ = writeln!(s, "lam {param} =>");
            write_core(body, depth + 1, s);
        }
        Node::App { func, arg } => {
            let _ = writeln!(s, "app");
            write_core(func, depth + 1, s);
            write_core(arg, depth + 1, s);
        }
        Node::Fix { name, body } => {
            let _ = writeln!(s, "fix {name} =>");
            write_core(body, depth + 1, s);
        }
        Node::FixGroup { defs, body } => {
            let _ = writeln!(s, "fixgroup");
            for (name, def) in defs {
                indent(depth + 1, s);
                let _ = writeln!(s, "def {name} =>");
                write_core(def, depth + 2, s);
            }
            indent(depth + 1, s);
            let _ = writeln!(s, "in");
            write_core(body, depth + 1, s);
        }
    }
}

// --- Stage 1: A-normal-form "substrate" --------------------------------------------------------

/// An operand of a lowered binding: a reference to a named/temp binding. (Public so backends — the
/// MLIR emitter / AOT path, M-150 — can consume the lowered IR.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Atom {
    /// A source `let`-bound name.
    Named(String),
    /// An introduced temporary, `%k`.
    Temp(usize),
}

impl Atom {
    /// The canonical textual rendering of this operand (`name` or `%k`).
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Atom::Named(x) => x.clone(),
            Atom::Temp(k) => format!("%{k}"),
        }
    }
}

/// The right-hand side of a lowered binding.
#[derive(Debug, Clone, PartialEq)]
pub enum Rhs {
    /// A constant value (carries its `Meta`, WF5).
    Const(Value),
    /// An alias to another binding (from a source `let`).
    Alias(Atom),
    /// A primitive application over atoms.
    Op {
        /// The primitive name.
        prim: String,
        /// Operand atoms.
        args: Vec<Atom>,
    },
    /// The representation-changing swap (carries its target and policy, WF1/WF2).
    Swap {
        /// The value being converted.
        src: Atom,
        /// The target representation.
        target: Repr,
        /// The selection policy reference (RFC-0005).
        policy: crate::ContentHash,
    },
    /// A saturated constructor application (RFC-0011 §4.1): builds a data value from field atoms.
    Construct {
        /// The constructor (`#T#i`).
        ctor: CtorRef,
        /// The field operands, in declaration order (saturated, WF6).
        args: Vec<Atom>,
    },
    /// Application of a function atom to an argument atom (RFC-0001 r4; call-by-value).
    App {
        /// The function operand (resolves to a closure).
        func: Atom,
        /// The argument operand.
        arg: Atom,
    },
    /// A lambda abstraction (RFC-0001 r4) — a **closure** value. Its body is a **nested** ANF block
    /// evaluated only on application (lazily), so the linear binding list stays acyclic.
    Lam {
        /// The bound parameter (a `Named` atom inside `body`).
        param: VarId,
        /// The body, lowered to a nested block (shares the program-wide temp counter, so its temps
        /// never collide with the enclosing scope).
        body: Anf,
    },
    /// General recursion (RFC-0001 r4) — its body (typically a [`Rhs::Lam`]) is a nested ANF block;
    /// the env-machine unfolds it under a fuel clock on application.
    Fix {
        /// The self-reference name bound in `body`.
        name: VarId,
        /// The recursive body, lowered to a nested block.
        body: Anf,
    },
    /// One member of a **mutual-recursion group** (RFC-0001 r5; [`Node::FixGroup`]). Lowering emits
    /// one such binding per member, each carrying the whole group's lowered definitions (`defs`) plus
    /// `which` member it is; the env-machine binds it to a suspension that, on application, re-binds
    /// every member name to its own focus suspension (so siblings can call each other) and enters
    /// `which`'s body — the env analogue of the interpreter's focus unfold, under the fuel clock.
    FixGroup {
        /// All members of the group `(name, lowered definition)` — shared by every member binding so
        /// each can resolve its siblings on unfold.
        defs: Vec<(VarId, Anf)>,
        /// Which member name this binding resolves to.
        which: VarId,
    },
    /// A flat pattern match (RFC-0011 §4.1): a scrutinee atom, single-level alternatives whose bodies
    /// are **nested** ANF blocks (evaluated only when selected), and at most one default block.
    Match {
        /// The scrutinised operand.
        scrutinee: Atom,
        /// The alternatives, tried first-match, left-to-right.
        alts: Vec<AnfAlt>,
        /// The catch-all block, taken when no alternative matches.
        default: Option<Anf>,
    },
}

/// One alternative of a lowered [`Rhs::Match`] — the ANF analogue of [`crate::node::Alt`], with the
/// arm body lowered to a nested block.
#[derive(Debug, Clone, PartialEq)]
pub enum AnfAlt {
    /// A constructor arm: matches a data value of `ctor`, binding its fields to `binders`
    /// (left-to-right, exactly the constructor's arity — WF7).
    Ctor {
        /// The constructor matched (`#T#i`).
        ctor: CtorRef,
        /// The field binders (`Named` atoms inside `body`).
        binders: Vec<VarId>,
        /// The arm body, lowered to a nested block (in scope of `binders`).
        body: Anf,
    },
    /// A literal arm: matches a representation value equal (repr + payload) to `value`.
    Lit {
        /// The literal value to match.
        value: Value,
        /// The arm body, lowered to a nested block.
        body: Anf,
    },
}

/// One lowered binding: a name, its right-hand side, and (where statically known) its scheduled
/// physical layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// The binding's name.
    pub name: Atom,
    /// Its right-hand side.
    pub rhs: Rhs,
    /// The scheduled packing, when the result repr is statically known (RFC-0004 §5).
    pub layout: Option<PhysicalLayout>,
}

/// A flattened (A-normal-form) lowering of a Core IR node.
#[derive(Debug, Clone, PartialEq)]
pub struct Anf {
    bindings: Vec<Binding>,
    result: Atom,
}

/// Lower a Core IR node into A-normal form (flatten nested nodes to a linear binding list). Pure and
/// deterministic; `Meta` rides along on `Const` bindings (WF5).
///
/// **Full v0 calculus (RFC-0011 §4.4 Q5 closed; M-342).** The ANF substrate / AOT env-machine path
/// covers the whole v0 calculus: `Const/Var/Let/Op/Swap` plus the r3/r4 data + recursion nodes
/// (`Construct`/`Match`/`Lam`/`App`/`Fix`). Body-bearing nodes (`Lam`/`Fix` bodies, `Match` arm/default
/// bodies) lower to **nested** ANF blocks evaluated lazily by the env-machine (so the binding list
/// stays acyclic and arms/closures are not eagerly run); a single program-wide temp counter keeps
/// every `Temp` globally unique, so a nested scope can never shadow an enclosing temp.
///
/// The native LLVM backend (`mycelium-mlir::llvm`) remains the **bit/trit subset** and refuses
/// data/closure nodes with an explicit `UnsupportedNode` (VR-5); this ANF + the `aot::run` env-machine
/// are the path the three-way differential exercises on the full calculus.
#[must_use]
pub fn lower_to_anf(node: &Node) -> Anf {
    let mut next = 0usize;
    lower_block(node, &mut next)
}

/// Lower a (sub-)expression to its own ANF block, **sharing** the program-wide temp counter `next`
/// so temps stay globally unique across nested blocks (closure/arm bodies).
fn lower_block(node: &Node, next: &mut usize) -> Anf {
    let mut b = Vec::new();
    let result = flatten(node, &mut b, next);
    Anf {
        bindings: b,
        result,
    }
}

fn fresh(next: &mut usize) -> usize {
    let k = *next;
    *next += 1;
    k
}

fn flatten(node: &Node, out: &mut Vec<Binding>, next: &mut usize) -> Atom {
    match node {
        Node::Var(x) => Atom::Named(x.clone()),
        Node::Const(v) => {
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Const(v.clone()),
                layout: schedule(v.repr()),
            });
            name
        }
        Node::Let { id, bound, body } => {
            let ba = flatten(bound, out, next);
            out.push(Binding {
                name: Atom::Named(id.clone()),
                rhs: Rhs::Alias(ba),
                layout: None,
            });
            flatten(body, out, next)
        }
        Node::Op { prim, args } => {
            let atoms: Vec<Atom> = args.iter().map(|a| flatten(a, out, next)).collect();
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Op {
                    prim: prim.clone(),
                    args: atoms,
                },
                layout: None, // Op result repr is not statically known (no operator typing yet).
            });
            name
        }
        Node::Swap {
            src,
            target,
            policy,
        } => {
            let sa = flatten(src, out, next);
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Swap {
                    src: sa,
                    target: target.clone(),
                    policy: policy.clone(),
                },
                layout: schedule(target),
            });
            name
        }
        Node::Construct { ctor, args } => {
            let atoms: Vec<Atom> = args.iter().map(|a| flatten(a, out, next)).collect();
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Construct {
                    ctor: ctor.clone(),
                    args: atoms,
                },
                layout: None, // a datum is not a representation value — no physical layout.
            });
            name
        }
        Node::App { func, arg } => {
            let f = flatten(func, out, next);
            let a = flatten(arg, out, next);
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::App { func: f, arg: a },
                layout: None,
            });
            name
        }
        Node::Lam { param, body } => {
            // The body is a nested block, not flattened into the current one: a closure body runs
            // only on application (lazy). The shared `next` keeps its temps globally unique.
            let body = lower_block(body, next);
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Lam {
                    param: param.clone(),
                    body,
                },
                layout: None,
            });
            name
        }
        Node::FixGroup { defs, body } => {
            // Lower every member definition to a nested block, then emit one `Rhs::FixGroup` binding
            // per member (each carrying the whole group). The member names are `Named` atoms, so the
            // continuation — and each sibling body — resolves them directly from the environment.
            let lowered: Vec<(VarId, Anf)> = defs
                .iter()
                .map(|(name, def)| (name.clone(), lower_block(def, next)))
                .collect();
            for (name, _) in defs {
                out.push(Binding {
                    name: Atom::Named(name.clone()),
                    rhs: Rhs::FixGroup {
                        defs: lowered.clone(),
                        which: name.clone(),
                    },
                    layout: None,
                });
            }
            flatten(body, out, next)
        }
        Node::Fix { name: fname, body } => {
            let body = lower_block(body, next);
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Fix {
                    name: fname.clone(),
                    body,
                },
                layout: None,
            });
            name
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let s = flatten(scrutinee, out, next);
            // Each arm/default body is a nested block (evaluated only when selected, never eagerly).
            let alts: Vec<AnfAlt> = alts
                .iter()
                .map(|alt| match alt {
                    Alt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => AnfAlt::Ctor {
                        ctor: ctor.clone(),
                        binders: binders.clone(),
                        body: lower_block(body, next),
                    },
                    Alt::Lit { value, body } => AnfAlt::Lit {
                        value: value.clone(),
                        body: lower_block(body, next),
                    },
                })
                .collect();
            let default = default.as_ref().map(|d| lower_block(d, next));
            let name = Atom::Temp(fresh(next));
            out.push(Binding {
                name: name.clone(),
                rhs: Rhs::Match {
                    scrutinee: s,
                    alts,
                    default,
                },
                layout: None,
            });
            name
        }
    }
}

fn render_layout(l: PhysicalLayout) -> String {
    match l {
        PhysicalLayout::BinaryWords => "BinaryWords".to_owned(),
        PhysicalLayout::TritPacked { scheme } => format!("TritPacked({scheme:?})"),
        PhysicalLayout::DenseArray => "DenseArray".to_owned(),
        PhysicalLayout::VsaStore { sparse } => format!("VsaStore(sparse={sparse})"),
    }
}

/// Render one lowered RHS into `s`, leaving the cursor at the end of its text (no trailing newline).
/// Flat RHSs render inline; body-bearing RHSs render a header then their nested block(s) indented.
fn write_rhs(rhs: &Rhs, depth: usize, s: &mut String) {
    match rhs {
        Rhs::Const(v) => {
            let _ = write!(s, "{}", render_const(v));
        }
        Rhs::Alias(a) => {
            let _ = write!(s, "{}", a.render());
        }
        Rhs::Op { prim, args } => {
            let a: Vec<String> = args.iter().map(Atom::render).collect();
            let _ = write!(s, "op {prim} {}", a.join(" "));
        }
        Rhs::Swap {
            src,
            target,
            policy,
        } => {
            let _ = write!(
                s,
                "swap {} -> {} @{}",
                src.render(),
                render_repr(target),
                short_hash(policy)
            );
        }
        Rhs::Construct { ctor, args } => {
            let a: Vec<String> = args.iter().map(Atom::render).collect();
            let _ = write!(s, "construct {ctor} {}", a.join(" "));
        }
        Rhs::App { func, arg } => {
            let _ = write!(s, "app {} {}", func.render(), arg.render());
        }
        Rhs::Lam { param, body } => {
            let _ = writeln!(s, "lam {param} =>");
            body.write_block(depth + 1, s);
        }
        Rhs::Fix { name, body } => {
            let _ = writeln!(s, "fix {name} =>");
            body.write_block(depth + 1, s);
        }
        Rhs::FixGroup { defs, which } => {
            let names: Vec<&str> = defs.iter().map(|(n, _)| n.as_str()).collect();
            let _ = writeln!(s, "fixgroup-member {which} of ({})", names.join(", "));
            for (name, body) in defs {
                let _ = writeln!(s, "{}def {name} =>", "  ".repeat(depth + 1));
                body.write_block(depth + 2, s);
            }
        }
        Rhs::Match {
            scrutinee,
            alts,
            default,
        } => {
            let _ = writeln!(s, "match {}", scrutinee.render());
            let pad = "  ".repeat(depth + 1);
            for alt in alts {
                match alt {
                    AnfAlt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        let _ = writeln!(s, "{pad}alt {ctor} ({}) =>", binders.join(" "));
                        body.write_block(depth + 2, s);
                    }
                    AnfAlt::Lit { value, body } => {
                        let _ = writeln!(s, "{pad}alt-lit {} =>", render_const(value));
                        body.write_block(depth + 2, s);
                    }
                }
                s.push('\n');
            }
            match default {
                Some(d) => {
                    let _ = writeln!(s, "{pad}default =>");
                    d.write_block(depth + 2, s);
                }
                None => {
                    let _ = write!(s, "{pad}no-default");
                }
            }
        }
    }
}

impl Anf {
    /// The canonical, diffable dump of the substrate stage (SC-4). Nested blocks (closure/recursion
    /// bodies, match arms) render indented; the flat-fragment output is unchanged.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut s = String::new();
        self.write_block(0, &mut s);
        s
    }

    fn write_block(&self, depth: usize, s: &mut String) {
        let pad = "  ".repeat(depth);
        let inner = "  ".repeat(depth + 1);
        let _ = writeln!(s, "{pad}substrate {{");
        for b in &self.bindings {
            let _ = write!(s, "{inner}{} = ", b.name.render());
            write_rhs(&b.rhs, depth + 1, s);
            if let Some(l) = b.layout {
                let _ = write!(s, "    ; layout={}", render_layout(l));
            }
            s.push('\n');
        }
        let _ = writeln!(s, "{inner}result {}", self.result.render());
        let _ = write!(s, "{pad}}}");
    }

    /// Number of bindings (for tests/tooling).
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether there are no bindings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// The ordered bindings (for backends consuming the lowered IR — M-150).
    #[must_use]
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// The result operand.
    #[must_use]
    pub fn result(&self) -> &Atom {
        &self.result
    }
}
