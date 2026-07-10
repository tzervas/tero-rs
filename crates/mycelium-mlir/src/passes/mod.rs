//! **EXPLAIN-able, never-silent optimization passes** over the lowered IR (M-726; RFC-0029 §7.2;
//! ADR-006/G2; phase-6).
//!
//! Three sanctioned passes — **inlining**, **CSE** (common-subexpression elimination), and **DCE**
//! (dead-code elimination) — run over a flat A-normal-form mirror of the lowered representation
//! (`mycelium_core::lower::Anf`). Each pass is a **pure function** `Program -> (Program, TransformLog)`:
//! it never mutates in place, and every change it makes is **reified** into a queryable
//! [`TransformLog`] of [`TransformRecord`]s — `(pass, rule, site, before → after, reason)`, mirroring
//! M-673's `MonoSelections`. So a user can ask *why* a particular inline/CSE/DCE decision was made; no
//! transform is silent (G2), and no black-box heuristic is hidden inside the rewrite (ADR-006).
//!
//! # What "never-silent" means here (the two halves of G2)
//! 1. **Every transform is recorded.** A pass that removes/folds/inlines a binding without a
//!    corresponding [`TransformRecord`] is a defect; [`run_pipeline`] returns the full log.
//! 2. **No observable change.** A pass is sound **only** if `eval(passes(ir)) == eval(ir)` — running
//!    the optimized program yields the *same* value as the unoptimized one (and as the reference
//!    interpreter). That equivalence is `Empirical` (trials over a corpus), proven by the
//!    `tests/passes_differential.rs` + in-crate `passes` harness — **never** `Proven` absent a
//!    checked equivalence proof (VR-5).
//!
//! # Guarantee tags (honest)
//! - The passes' structural correctness — that the rewrite preserves the IR's meaning — is
//!   **`Declared`** (asserted from the transform rules; not machine-checked).
//! - The `with == without == interp` agreement is **`Empirical`** (the differential corpus).
//!
//! Neither is `Proven`.
//!
//! # Scope (YAGNI; RFC-0029 §7.2)
//! Only inlining/CSE/DCE are sanctioned for 1.0.0. The IR mirror covers the **straight-line +
//! `let`/alias** fragment plus first-order closures (`Lam`/`App`) — exactly where these three passes
//! meaningfully fire — and **conservatively preserves** any node it does not optimize (the pass is a
//! no-op on it, recorded as such by *absence* from the log). Evaluation of both the optimized and the
//! unoptimized program reuses the **trusted** env-machine (`crate::aot::run_core`) and the reference
//! interpreter via a lossless round-trip back to `mycelium_core::Node` — the passes never re-implement
//! evaluation (DRY), so the differential checks them against the trusted base, not a clone of it.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

use mycelium_core::lower::{self, Anf, AnfAlt, Atom, Binding, Rhs};
use mycelium_core::{Node, PhysicalLayout, Value, VarId};

pub mod cse;
pub mod dce;
pub mod inline;

pub use cse::cse;
pub use dce::dce;
pub use inline::inline;

// ─── the IR mirror (owned, constructible — the lowered representation the passes transform) ──────

/// A right-hand side in the pass IR — a structural mirror of [`mycelium_core::lower::Rhs`], owned so
/// the passes can construct and rewrite it freely (the upstream `Anf` has private fields and no
/// constructor, so the passes carry their own form and round-trip to [`Node`] for trusted evaluation).
#[derive(Debug, Clone, PartialEq)]
pub enum PassRhs {
    /// A constant value (carries its `Meta`, WF5).
    Const(Value),
    /// An alias to another binding.
    Alias(Atom),
    /// A primitive application over operand atoms.
    Op {
        /// The primitive name.
        prim: String,
        /// Operand atoms.
        args: Vec<Atom>,
    },
    /// A representation-changing swap (target + policy).
    Swap {
        /// The value being converted.
        src: Atom,
        /// The target representation.
        target: mycelium_core::Repr,
        /// The selection-policy reference.
        policy: mycelium_core::ContentHash,
    },
    /// A saturated constructor application.
    Construct {
        /// The constructor (`#T#i`).
        ctor: mycelium_core::data::CtorRef,
        /// The field operands, in declaration order.
        args: Vec<Atom>,
    },
    /// Application of a function atom to an argument atom (call-by-value).
    App {
        /// The function operand.
        func: Atom,
        /// The argument operand.
        arg: Atom,
    },
    /// A lambda abstraction — a closure whose body is a nested pass-IR block.
    Lam {
        /// The bound parameter.
        param: VarId,
        /// The body block.
        body: Program,
    },
    /// General recursion — its body is a nested pass-IR block.
    Fix {
        /// The self-reference name bound in `body`.
        name: VarId,
        /// The recursive body block.
        body: Program,
    },
    /// One member of a mutual-recursion group.
    FixGroup {
        /// All members `(name, lowered definition)`.
        defs: Vec<(VarId, Program)>,
        /// Which member this binding resolves to.
        which: VarId,
    },
    /// A flat pattern match with nested arm/default blocks.
    Match {
        /// The scrutinised operand.
        scrutinee: Atom,
        /// The alternatives, tried first-match.
        alts: Vec<PassAlt>,
        /// The catch-all block.
        default: Option<Program>,
    },
}

/// One alternative of a [`PassRhs::Match`] — the pass-IR mirror of [`AnfAlt`].
#[derive(Debug, Clone, PartialEq)]
pub enum PassAlt {
    /// A constructor arm.
    Ctor {
        /// The constructor matched.
        ctor: mycelium_core::data::CtorRef,
        /// The field binders.
        binders: Vec<VarId>,
        /// The arm body block.
        body: Program,
    },
    /// A literal arm.
    Lit {
        /// The literal value to match.
        value: Value,
        /// The arm body block.
        body: Program,
    },
}

/// One lowered binding in the pass IR.
#[derive(Debug, Clone, PartialEq)]
pub struct PassBinding {
    /// The binding's name.
    pub name: Atom,
    /// Its right-hand side.
    pub rhs: PassRhs,
    /// The scheduled packing, when statically known (carried through unchanged).
    pub layout: Option<PhysicalLayout>,
}

/// A flat (A-normal-form) pass-IR program: an ordered binding list and a result operand. This is the
/// representation the three passes transform — the owned, constructible mirror of
/// [`mycelium_core::lower::Anf`].
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// The ordered bindings.
    pub bindings: Vec<PassBinding>,
    /// The result operand.
    pub result: Atom,
}

impl Program {
    /// Lift the genuine lowered IR ([`Anf`]) into the owned pass IR (read via the public accessors).
    /// Pure and structural — every binding/result is carried across faithfully.
    #[must_use]
    pub fn from_anf(anf: &Anf) -> Self {
        Program {
            bindings: anf.bindings().iter().map(PassBinding::from_anf).collect(),
            result: anf.result().clone(),
        }
    }

    /// Lower a Core IR [`Node`] to the pass IR, by lowering to ANF first (`lower::lower_to_anf`) then
    /// lifting. The single entry the passes and the differential use.
    #[must_use]
    pub fn lower(node: &Node) -> Self {
        Program::from_anf(&lower::lower_to_anf(node))
    }

    /// Reconstruct a Core IR [`Node`] from this pass IR (the inverse of [`Program::lower`] over the
    /// fragment the passes touch): fold the flat binding list into a nested `let` chain whose
    /// innermost body is the result operand. Used **only** to feed the trusted evaluators
    /// (`crate::aot::run_core` / the reference interpreter), so the differential checks the passes
    /// against the trusted base rather than a re-implemented evaluator (DRY).
    #[must_use]
    pub fn to_node(&self) -> Node {
        let mut body = atom_to_node(&self.result);
        for b in self.bindings.iter().rev() {
            body = Node::Let {
                id: atom_name(&b.name),
                bound: Box::new(b.rhs.to_node()),
                body: Box::new(body),
            };
        }
        body
    }

    /// The number of bindings (for tests/tooling).
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether there are no bindings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl PassBinding {
    fn from_anf(b: &Binding) -> Self {
        PassBinding {
            name: b.name.clone(),
            rhs: PassRhs::from_anf(&b.rhs),
            layout: b.layout,
        }
    }
}

impl PassRhs {
    fn from_anf(rhs: &Rhs) -> Self {
        match rhs {
            Rhs::Const(v) => PassRhs::Const(v.clone()),
            Rhs::Alias(a) => PassRhs::Alias(a.clone()),
            Rhs::Op { prim, args } => PassRhs::Op {
                prim: prim.clone(),
                args: args.clone(),
            },
            Rhs::Swap {
                src,
                target,
                policy,
            } => PassRhs::Swap {
                src: src.clone(),
                target: target.clone(),
                policy: policy.clone(),
            },
            Rhs::Construct { ctor, args } => PassRhs::Construct {
                ctor: ctor.clone(),
                args: args.clone(),
            },
            Rhs::App { func, arg } => PassRhs::App {
                func: func.clone(),
                arg: arg.clone(),
            },
            Rhs::Lam { param, body } => PassRhs::Lam {
                param: param.clone(),
                body: Program::from_anf(body),
            },
            Rhs::Fix { name, body } => PassRhs::Fix {
                name: name.clone(),
                body: Program::from_anf(body),
            },
            Rhs::FixGroup { defs, which } => PassRhs::FixGroup {
                defs: defs
                    .iter()
                    .map(|(n, b)| (n.clone(), Program::from_anf(b)))
                    .collect(),
                which: which.clone(),
            },
            Rhs::Match {
                scrutinee,
                alts,
                default,
            } => PassRhs::Match {
                scrutinee: scrutinee.clone(),
                alts: alts.iter().map(PassAlt::from_anf).collect(),
                default: default.as_ref().map(Program::from_anf),
            },
        }
    }

    /// Reconstruct the Core IR node this RHS denotes, with operand atoms resolved to `var`/`let`
    /// references. The inverse of the lowering for the fragment the passes operate on.
    fn to_node(&self) -> Node {
        match self {
            PassRhs::Const(v) => Node::Const(v.clone()),
            PassRhs::Alias(a) => atom_to_node(a),
            PassRhs::Op { prim, args } => Node::Op {
                prim: prim.clone(),
                args: args.iter().map(atom_to_node).collect(),
            },
            PassRhs::Swap {
                src,
                target,
                policy,
            } => Node::Swap {
                src: Box::new(atom_to_node(src)),
                target: target.clone(),
                policy: policy.clone(),
            },
            PassRhs::Construct { ctor, args } => Node::Construct {
                ctor: ctor.clone(),
                args: args.iter().map(atom_to_node).collect(),
            },
            PassRhs::App { func, arg } => Node::App {
                func: Box::new(atom_to_node(func)),
                arg: Box::new(atom_to_node(arg)),
            },
            PassRhs::Lam { param, body } => Node::Lam {
                param: param.clone(),
                body: Box::new(body.to_node()),
            },
            PassRhs::Fix { name, body } => Node::Fix {
                name: name.clone(),
                body: Box::new(body.to_node()),
            },
            PassRhs::FixGroup { defs, which } => {
                // Reconstruct the whole group (every member name binds mutually), then reference the
                // focused member — the inverse of the per-member lowering.
                Node::Let {
                    id: which.clone(),
                    bound: Box::new(Node::FixGroup {
                        defs: defs
                            .iter()
                            .map(|(n, b)| (n.clone(), Box::new(b.to_node())))
                            .collect(),
                        body: Box::new(Node::Var(which.clone())),
                    }),
                    body: Box::new(Node::Var(which.clone())),
                }
            }
            PassRhs::Match {
                scrutinee,
                alts,
                default,
            } => Node::Match {
                scrutinee: Box::new(atom_to_node(scrutinee)),
                alts: alts.iter().map(PassAlt::to_node).collect(),
                default: default.as_ref().map(|d| Box::new(d.to_node())),
            },
        }
    }
}

impl PassAlt {
    fn from_anf(alt: &AnfAlt) -> Self {
        match alt {
            AnfAlt::Ctor {
                ctor,
                binders,
                body,
            } => PassAlt::Ctor {
                ctor: ctor.clone(),
                binders: binders.clone(),
                body: Program::from_anf(body),
            },
            AnfAlt::Lit { value, body } => PassAlt::Lit {
                value: value.clone(),
                body: Program::from_anf(body),
            },
        }
    }

    fn to_node(&self) -> mycelium_core::node::Alt {
        match self {
            PassAlt::Ctor {
                ctor,
                binders,
                body,
            } => mycelium_core::node::Alt::Ctor {
                ctor: ctor.clone(),
                binders: binders.clone(),
                body: body.to_node(),
            },
            PassAlt::Lit { value, body } => mycelium_core::node::Alt::Lit {
                value: value.clone(),
                body: body.to_node(),
            },
        }
    }
}

/// The textual name of an atom — a source name, or `%k` for a temp. Used as a `let`/`var` identifier
/// when round-tripping the pass IR back to a [`Node`] for trusted evaluation. The `%` prefix is not a
/// surface-identifier character, so a reconstructed temp name never collides with a source binder.
fn atom_name(a: &Atom) -> VarId {
    a.render()
}

/// A node that references an atom: a `Var` of its textual name (the round-trip `let` chain binds
/// every atom under that name).
fn atom_to_node(a: &Atom) -> Node {
    Node::Var(atom_name(a))
}

// ─── the EXPLAIN record: a reified, queryable transform log (RFC-0029 §7.2) ──────────────────────

/// Which optimization pass produced a [`TransformRecord`]. A reified pass identity (no string typos).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pass {
    /// The inlining pass.
    Inline,
    /// Common-subexpression elimination.
    Cse,
    /// Dead-code elimination.
    Dce,
}

impl Pass {
    /// The pass's stable short name (for the EXPLAIN dump / queries).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Pass::Inline => "inline",
            Pass::Cse => "cse",
            Pass::Dce => "dce",
        }
    }
}

impl core::fmt::Display for Pass {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// One reified optimization decision — the `(pass, rule, site, before → after, reason)` tuple of
/// RFC-0029 §7.2. Every transform a pass performs emits exactly one of these (G2: never silent), so
/// the optimization is auditable: the user can read *what* changed, *where*, and *why*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformRecord {
    /// Which pass made the decision.
    pub pass: Pass,
    /// The specific rewrite rule applied (e.g. `"alias-fold"`, `"cse-redirect"`, `"drop-dead"`).
    pub rule: &'static str,
    /// The binding the transform acted on (its textual name) — the *site*.
    pub site: String,
    /// A short textual rendering of the binding **before** the transform.
    pub before: String,
    /// A short textual rendering of the result **after** the transform (e.g. the redirected
    /// operand, or `"<removed>"` for a DCE drop).
    pub after: String,
    /// The human-readable *why* — the reason the rule fired here (no black box, ADR-006).
    pub reason: String,
}

impl core::fmt::Display for TransformRecord {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "[{}/{}] {}: {} -> {}  ({})",
            self.pass, self.rule, self.site, self.before, self.after, self.reason
        )
    }
}

/// The reified, queryable log of every transform a pass (or the pipeline) performed — the EXPLAIN
/// surface of M-726, mirroring M-673's `MonoSelections`. Append-only during a pass; queryable after.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransformLog {
    entries: Vec<TransformRecord>,
}

impl TransformLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        TransformLog::default()
    }

    /// Record one transform decision (the only mutator — every change flows through here, so a pass
    /// physically cannot make a silent transform without a corresponding entry).
    pub fn record(&mut self, record: TransformRecord) {
        self.entries.push(record);
    }

    /// Every recorded transform, in the order performed.
    #[must_use]
    pub fn entries(&self) -> &[TransformRecord] {
        &self.entries
    }

    /// Whether nothing was transformed (the pass was a no-op).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many transforms were recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The records produced by a given pass (an EXPLAIN query: "show me every CSE decision").
    #[must_use]
    pub fn by_pass(&self, pass: Pass) -> Vec<&TransformRecord> {
        self.entries.iter().filter(|r| r.pass == pass).collect()
    }

    /// The records that acted on a given binding name (an EXPLAIN query: "why did binding `%3`
    /// change?"). Matches the record `site`.
    #[must_use]
    pub fn by_site(&self, site: &str) -> Vec<&TransformRecord> {
        self.entries.iter().filter(|r| r.site == site).collect()
    }

    /// Whether any record was produced by `pass` (a quick "did inlining fire?" query).
    #[must_use]
    pub fn fired(&self, pass: Pass) -> bool {
        self.entries.iter().any(|r| r.pass == pass)
    }

    /// Merge another log's entries into this one, preserving order (used by the pipeline to
    /// accumulate every pass's decisions into one auditable record).
    pub fn extend(&mut self, other: TransformLog) {
        self.entries.extend(other.entries);
    }

    /// A multi-line EXPLAIN dump of every recorded decision (one `TransformRecord` per line).
    #[must_use]
    pub fn explain(&self) -> String {
        let mut s = String::new();
        for r in &self.entries {
            s.push_str(&r.to_string());
            s.push('\n');
        }
        s
    }
}

/// Run the full sanctioned pipeline — **inline → CSE → DCE** — over a pass-IR program, accumulating
/// every pass's decisions into one [`TransformLog`]. The order is deliberate and recorded: inlining
/// exposes fresh common subexpressions for CSE, and both can leave bindings dead for DCE to remove.
/// Each stage is the pure `Program -> (Program, TransformLog)` contract; the result is the optimized
/// program plus the merged, queryable log (never silent — every change is in the log).
#[must_use]
pub fn run_pipeline(program: &Program) -> (Program, TransformLog) {
    let mut log = TransformLog::new();
    let (p1, l1) = inline(program);
    log.extend(l1);
    let (p2, l2) = cse(&p1);
    log.extend(l2);
    let (p3, l3) = dce(&p2);
    log.extend(l3);
    (p3, log)
}

/// Lower a [`Node`], run the full pipeline, and return the optimized pass IR + the merged log. The
/// convenience entry the differential and EXPLAIN tooling use.
#[must_use]
pub fn optimize(node: &Node) -> (Program, TransformLog) {
    run_pipeline(&Program::lower(node))
}

// ─── shared rendering / analysis helpers ─────────────────────────────────────────────────────────

/// A short one-line rendering of a pass-IR RHS for a [`TransformRecord`] (never the full nested
/// block — just the head, enough to read the log).
pub(crate) fn render_rhs(rhs: &PassRhs) -> String {
    match rhs {
        PassRhs::Const(v) => format!("const {:?}", v.repr()),
        PassRhs::Alias(a) => format!("alias {}", a.render()),
        PassRhs::Op { prim, args } => {
            let a: Vec<String> = args.iter().map(Atom::render).collect();
            format!("op {prim} {}", a.join(" "))
        }
        PassRhs::Swap { src, target, .. } => {
            format!("swap {} -> {:?}", src.render(), target)
        }
        PassRhs::Construct { ctor, args } => {
            let a: Vec<String> = args.iter().map(Atom::render).collect();
            format!("construct {ctor} {}", a.join(" "))
        }
        PassRhs::App { func, arg } => format!("app {} {}", func.render(), arg.render()),
        PassRhs::Lam { param, .. } => format!("lam {param} => …"),
        PassRhs::Fix { name, .. } => format!("fix {name} => …"),
        PassRhs::FixGroup { which, .. } => format!("fixgroup-member {which}"),
        PassRhs::Match { scrutinee, .. } => format!("match {} …", scrutinee.render()),
    }
}

/// Collect every operand atom a RHS *reads* (its free uses, not its binders) — the use-set the CSE
/// redirect and DCE liveness analyses walk. Nested blocks (closure/recursion/match bodies) contribute
/// their own free atoms minus the binders they introduce; this is computed by [`free_atoms_block`].
pub(crate) fn rhs_uses(rhs: &PassRhs, out: &mut Vec<Atom>) {
    match rhs {
        PassRhs::Const(_) => {}
        PassRhs::Alias(a) => out.push(a.clone()),
        PassRhs::Op { args, .. } | PassRhs::Construct { args, .. } => {
            out.extend(args.iter().cloned());
        }
        PassRhs::Swap { src, .. } => out.push(src.clone()),
        PassRhs::App { func, arg } => {
            out.push(func.clone());
            out.push(arg.clone());
        }
        PassRhs::Lam { param, body } => {
            let mut inner = free_atoms_block(body);
            inner.retain(|a| *a != Atom::Named(param.clone()));
            out.extend(inner);
        }
        PassRhs::Fix { name, body } => {
            let mut inner = free_atoms_block(body);
            inner.retain(|a| *a != Atom::Named(name.clone()));
            out.extend(inner);
        }
        PassRhs::FixGroup { defs, .. } => {
            let names: Vec<Atom> = defs.iter().map(|(n, _)| Atom::Named(n.clone())).collect();
            for (_, body) in defs {
                let mut inner = free_atoms_block(body);
                inner.retain(|a| !names.contains(a));
                out.extend(inner);
            }
        }
        PassRhs::Match {
            scrutinee,
            alts,
            default,
        } => {
            out.push(scrutinee.clone());
            for alt in alts {
                match alt {
                    PassAlt::Ctor { binders, body, .. } => {
                        let mut inner = free_atoms_block(body);
                        inner.retain(|a| !binders.iter().any(|b| *a == Atom::Named(b.clone())));
                        out.extend(inner);
                    }
                    PassAlt::Lit { body, .. } => out.extend(free_atoms_block(body)),
                }
            }
            if let Some(d) = default {
                out.extend(free_atoms_block(d));
            }
        }
    }
}

/// The free atoms of a nested block — every atom it reads that is not bound by one of its own
/// bindings (the block's local binders shadow). The result operand counts as a use.
pub(crate) fn free_atoms_block(program: &Program) -> Vec<Atom> {
    let mut bound: Vec<Atom> = Vec::new();
    let mut free: Vec<Atom> = Vec::new();
    for b in &program.bindings {
        let mut uses = Vec::new();
        rhs_uses(&b.rhs, &mut uses);
        for u in uses {
            if !bound.contains(&u) {
                free.push(u);
            }
        }
        bound.push(b.name.clone());
    }
    if !bound.contains(&program.result) {
        free.push(program.result.clone());
    }
    free
}
