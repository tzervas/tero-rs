//! Content-addressing: hash-of-AST definition identity, with names as separable metadata
//! (RFC-0001 §4.6; ADR-003). M-103.
//!
//! ```text
//! hash(def) = H( normalize(structure(def)) ‖ types_with_repr(def) ‖ static_contract(def) )
//! ```
//!
//! **Hashed (identity-bearing):** the normalized node structure (α-renamed so binder *names* don't
//! matter and bound variables are position-independent de Bruijn indices), the result/operand types
//! *including [`Repr`]*, the literal value of a [`Node::Const`], the operator name of a
//! [`Node::Op`], and the static contract of a [`Node::Swap`] (its target `Repr` and `policy`).
//!
//! **Not hashed (metadata):** human names (binder names, free-variable spellings are kept only
//! because they are part of the *contract* of an open term — see below), source spans, comments,
//! and **all dynamic value metadata** ([`crate::meta::Meta`]: provenance, measured sparsity,
//! realized bounds, `policy_used`). Names are stored separately in a [`Names`] map; renaming a
//! definition does not change its identity.
//!
//! Consequences (the M-103 acceptance, RFC-0001 §4.6): two definitions differing only in
//! representation paradigm get *different* hashes; a definition and any α-renaming/reformatting get
//! the *same* hash; identical definitions collide.
//!
//! The kernel hash is **BLAKE3**; the encoding fed to it is domain-separated and length-prefixed so
//! that distinct structures can never collide by concatenation ambiguity (an injective framing).

use std::collections::HashMap;

use crate::id::ContentHash;
use crate::node::{Alt, Node, VarId};
use crate::repr::{Repr, ScalarKind, SparsityClass};
use crate::value::{Payload, Trit, Value};

/// Domain-separation tags — one byte per syntactic form, so the framing is injective across kinds.
pub(crate) mod tag {
    pub const VAR_BOUND: u8 = 0x01;
    pub const VAR_FREE: u8 = 0x02;
    pub const CONST: u8 = 0x03;
    pub const LET: u8 = 0x04;
    pub const OP: u8 = 0x05;
    pub const SWAP: u8 = 0x06;

    pub const REPR_BINARY: u8 = 0x10;
    pub const REPR_TERNARY: u8 = 0x11;
    pub const REPR_DENSE: u8 = 0x12;
    pub const REPR_VSA: u8 = 0x13;
    /// RFC-0032 D3/D4 (M-749/M-750): the indexed-sequence + byte-string reprs. Appended
    /// (append-only: existing codes are frozen so a definition's identity never shifts when the
    /// registry grows).
    pub const REPR_SEQ: u8 = 0x14;
    pub const REPR_BYTES: u8 = 0x15;
    /// ADR-040 §3 (M-896): the scalar-float repr. Appended (append-only: existing codes are
    /// frozen, so adding this arm changes **no** existing value's address — no rehash is spent;
    /// pinned by the address-stability regression test).
    pub const REPR_FLOAT: u8 = 0x16;

    pub const PAYLOAD_BITS: u8 = 0x20;
    pub const PAYLOAD_TRITS: u8 = 0x21;
    pub const PAYLOAD_SCALARS: u8 = 0x22;
    pub const PAYLOAD_HYPERVECTOR: u8 = 0x23;
    /// RFC-0032 D3/D4 (M-749/M-750): the sequence + byte-string payloads. Appended (append-only).
    pub const PAYLOAD_SEQ: u8 = 0x24;
    pub const PAYLOAD_BYTES: u8 = 0x25;
    /// ADR-040 §3 (M-896): the scalar-float payload. Appended (append-only).
    pub const PAYLOAD_FLOAT: u8 = 0x26;

    pub const SPARSITY_DENSE: u8 = 0x30;
    pub const SPARSITY_SPARSE: u8 = 0x31;

    // r3 (RFC-0001 §4.3/§4.5, RFC-0011): the data registry Σ + the Construct/Match nodes.
    // (0x07 is the standalone `operation_hash` PRIM tag — kept distinct here.)
    pub const CONSTRUCT: u8 = 0x08;
    pub const MATCH: u8 = 0x09;
    pub const ALT_CTOR: u8 = 0x40;
    pub const ALT_LIT: u8 = 0x41;
    pub const MATCH_DEFAULT: u8 = 0x42;
    pub const MATCH_NO_DEFAULT: u8 = 0x43;

    pub const DATADECL: u8 = 0x50;
    pub const CTOR_DECL: u8 = 0x51;
    pub const FIELD_REPR: u8 = 0x52;
    pub const FIELD_DATA: u8 = 0x53; // an out-of-cycle data field: continues with the decl hash
    pub const FIELD_CYCLE: u8 = 0x54; // an in-cycle data field: continues with a placeholder index
    pub const CTOR_REF: u8 = 0x55;
    pub const DATUM: u8 = 0x56;
    /// ADR-033 §10 (FLAG-1 Path A): a function-typed field with full signature encoding.
    /// Precedes `c.u32(arity)` then the `FnSig` encoding. Disjoint from all prior tags.
    pub const FIELD_FN: u8 = 0x57;
    /// Count of parameters in a `FnSig` (precedes each param `FieldTyRef` encoding).
    pub const FN_SIG_PARAMS: u8 = 0x58;
    /// Return-type marker in a `FnSig` (precedes the return `FieldTyRef` encoding).
    pub const FN_SIG_RET: u8 = 0x59;
    /// A `FieldTyRef::Repr` leaf inside a `FnSig` (continues with `Canon::repr`).
    pub const FTR_REPR: u8 = 0x5a;
    /// A `FieldTyRef::Data` leaf (out-of-cycle) inside a `FnSig` (continues with decl hash).
    pub const FTR_DATA: u8 = 0x5b;
    /// A `FieldTyRef::Data` leaf (in-cycle) inside a `FnSig` (continues with placeholder index).
    pub const FTR_DATA_CYCLE: u8 = 0x5c;
    /// A `FieldTyRef::Fn` leaf inside a `FnSig` (continues with a nested `FnSig` encoding).
    pub const FTR_FN: u8 = 0x5d;

    // r4 (RFC-0001 r4; RFC-0007 §4.1): the function/recursion nodes.
    pub const LAM: u8 = 0x0a;
    pub const APP: u8 = 0x0b;
    pub const FIX: u8 = 0x0c;
    pub const FIXGROUP: u8 = 0x0d;

    // R7-Q4 (RFC-0007 §4.4/§8; DN-10 §3): content-addressed prim declarations (the Π table). The
    // prim *operation_hash* PRIM tag (0x07) addresses a prim by name (provenance); these tags
    // address a prim by its signature + intrinsic guarantee (its declaration identity, ADR-003).
    pub const PRIM_DECL: u8 = 0x60;
    pub const PRIM_PARADIGM_ANY: u8 = 0x61;
    pub const PRIM_PARADIGM_BINARY: u8 = 0x62;
    pub const PRIM_PARADIGM_TERNARY: u8 = 0x63;
    pub const PRIM_WIDTH_UNIFORM: u8 = 0x64;
    pub const PRIM_STRENGTH_EXACT: u8 = 0x65;
    pub const PRIM_STRENGTH_PROVEN: u8 = 0x66;
    pub const PRIM_STRENGTH_EMPIRICAL: u8 = 0x67;
    pub const PRIM_STRENGTH_DECLARED: u8 = 0x68;
    /// The width-collapsing comparison rule (RFC-0032 D1). Distinct tag ⇒ a collapsing prim's decl
    /// hash differs from an otherwise-identical uniform one; existing uniform decls are unchanged.
    pub const PRIM_WIDTH_COLLAPSE: u8 = 0x69;
}

/// A canonical, injective, metadata-free byte encoder feeding a [`blake3::Hasher`]. Every write is
/// either a fixed-width integer or a length-prefixed blob, so no two distinct structures share an
/// encoding.
pub(crate) struct Canon {
    h: blake3::Hasher,
}

impl Canon {
    pub(crate) fn new() -> Self {
        Canon {
            h: blake3::Hasher::new(),
        }
    }

    pub(crate) fn tag(&mut self, t: u8) {
        self.h.update(&[t]);
    }

    pub(crate) fn u32(&mut self, n: u32) {
        self.h.update(&n.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, n: u64) {
        self.h.update(&n.to_le_bytes());
    }

    /// A length-prefixed byte blob (the prefix makes the framing injective).
    pub(crate) fn blob(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.h.update(bytes);
    }

    pub(crate) fn str(&mut self, s: &str) {
        self.blob(s.as_bytes());
    }

    /// A finite-precision scalar by its exact bit pattern — deterministic and bit-faithful (so e.g.
    /// `+0.0` and `-0.0` are distinct identities, as they are distinct literals).
    ///
    /// **Known seam, deliberately unchanged here (ADR-040 §2.3 / FLAG-5, `Empirical`):** the
    /// existing `Dense`/`Hypervector` payload paths feed **raw** bits through this encoder with no
    /// NaN canonicalization, so NaN-bearing tensors already have platform-bit-dependent identities.
    /// Settling one uniform NaN rule for those paths is identity-affecting and rides the single
    /// E20-1 rehash (RFC-0033 §7) — NOT this change (M-896), which canonicalizes only the new
    /// scalar [`Payload::Float`] arm and leaves every existing address byte-identical.
    fn f64(&mut self, x: f64) {
        self.h.update(&x.to_bits().to_le_bytes());
    }

    pub(crate) fn finish(self) -> ContentHash {
        let hex = self.h.finalize().to_hex();
        // BLAKE3 hex is 64 lowercase [0-9a-f] chars — always a well-formed digest.
        ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest")
    }

    /// Absorb a [`ContentHash`] (e.g. a referenced data-declaration hash) as a length-prefixed blob.
    pub(crate) fn hash(&mut self, h: &ContentHash) {
        self.str(h.as_str());
    }

    /// Absorb a [`CtorRef`](crate::data::CtorRef): its declaration hash and constructor index. The
    /// constructor *name* is not identity-bearing (ADR-003) — only the `#T#i` pair is.
    pub(crate) fn ctor_ref(&mut self, c: &crate::data::CtorRef) {
        self.tag(tag::CTOR_REF);
        self.hash(c.decl());
        self.u32(c.index());
    }

    /// Absorb a prim declaration's identity-bearing content (R7-Q4; RFC-0007 §4.4): its signature
    /// `(τ₁…τₙ) → τ` (arity-prefixed operand paradigms, the result paradigm, the width relation) and
    /// its intrinsic guarantee `g_f` (RFC-0001 §4.7). The prim *name* is excluded (ADR-003), so two
    /// prims with the same signature and intrinsic collide regardless of name.
    pub(crate) fn prim_decl(
        &mut self,
        sig: &crate::prim::PrimSig,
        intrinsic: crate::guarantee::GuaranteeStrength,
    ) {
        self.tag(tag::PRIM_DECL);
        self.u64(sig.operands.len() as u64);
        for &p in &sig.operands {
            self.prim_paradigm(p);
        }
        self.prim_paradigm(sig.result);
        match sig.width {
            crate::prim::WidthRel::Uniform => self.tag(tag::PRIM_WIDTH_UNIFORM),
            crate::prim::WidthRel::Collapse => self.tag(tag::PRIM_WIDTH_COLLAPSE),
        }
        self.strength(intrinsic);
    }

    fn prim_paradigm(&mut self, p: crate::prim::PrimParadigm) {
        self.tag(match p {
            crate::prim::PrimParadigm::Any => tag::PRIM_PARADIGM_ANY,
            crate::prim::PrimParadigm::Binary => tag::PRIM_PARADIGM_BINARY,
            crate::prim::PrimParadigm::Ternary => tag::PRIM_PARADIGM_TERNARY,
        });
    }

    fn strength(&mut self, s: crate::guarantee::GuaranteeStrength) {
        use crate::guarantee::GuaranteeStrength::{Declared, Empirical, Exact, Proven};
        self.tag(match s {
            Exact => tag::PRIM_STRENGTH_EXACT,
            Proven => tag::PRIM_STRENGTH_PROVEN,
            Empirical => tag::PRIM_STRENGTH_EMPIRICAL,
            Declared => tag::PRIM_STRENGTH_DECLARED,
        });
    }
}

impl Canon {
    fn scalar_kind(&mut self, k: ScalarKind) {
        // The scalar precision is semantically significant (it bounds embedding error) — part of
        // the type, hence identity-bearing (RFC-0001 §4.1).
        self.h.update(&[k.tag()]);
    }

    pub(crate) fn repr(&mut self, r: &Repr) {
        match r {
            Repr::Binary { width } => {
                self.tag(tag::REPR_BINARY);
                self.u32(*width);
            }
            Repr::Ternary { trits } => {
                self.tag(tag::REPR_TERNARY);
                self.u32(*trits);
            }
            Repr::Dense { dim, dtype } => {
                self.tag(tag::REPR_DENSE);
                self.u32(*dim);
                self.scalar_kind(*dtype);
            }
            Repr::Vsa {
                model,
                dim,
                sparsity,
            } => {
                self.tag(tag::REPR_VSA);
                self.str(model);
                self.u32(*dim);
                match sparsity {
                    SparsityClass::Dense => self.tag(tag::SPARSITY_DENSE),
                    SparsityClass::Sparse { max_active } => {
                        self.tag(tag::SPARSITY_SPARSE);
                        self.u32(*max_active);
                    }
                }
            }
            Repr::Seq { elem, len } => {
                // The element type AND the declared length are part of the sequence type's identity
                // (RFC-0032 D3): `Seq<Binary{8}, 3>` and `Seq<Binary{8}, 4>` are distinct types, and
                // `Seq<Binary{8}>` ≠ `Seq<Ternary{8}>`. The nested `elem` recurses through this same
                // injective encoder, so the framing stays collision-free.
                self.tag(tag::REPR_SEQ);
                self.u32(*len);
                self.repr(elem);
            }
            Repr::Float { width } => {
                // ADR-040 §3: identity-bearing = the Float variant tag + the frozen width tag
                // (the `FloatWidth::tag()` registry — append-only, address-stable).
                self.tag(tag::REPR_FLOAT);
                self.h.update(&[width.tag()]);
            }
            Repr::Bytes => {
                // A byte string carries no type parameter (any byte content) — the tag alone fixes
                // the type identity (RFC-0032 D4).
                self.tag(tag::REPR_BYTES);
            }
        }
    }

    fn payload(&mut self, p: &Payload) {
        match p {
            Payload::Bits(bits) => {
                self.tag(tag::PAYLOAD_BITS);
                self.u64(bits.len() as u64);
                for &b in bits {
                    self.h.update(&[u8::from(b)]);
                }
            }
            Payload::Trits(trits) => {
                self.tag(tag::PAYLOAD_TRITS);
                self.u64(trits.len() as u64);
                for &t in trits {
                    let code: u8 = match t {
                        Trit::Neg => 0,
                        Trit::Zero => 1,
                        Trit::Pos => 2,
                    };
                    self.h.update(&[code]);
                }
            }
            Payload::Scalars(xs) => {
                self.tag(tag::PAYLOAD_SCALARS);
                self.u64(xs.len() as u64);
                for &x in xs {
                    self.f64(x);
                }
            }
            Payload::Hypervector(xs) => {
                self.tag(tag::PAYLOAD_HYPERVECTOR);
                self.u64(xs.len() as u64);
                for &x in xs {
                    self.f64(x);
                }
            }
            Payload::Float(x) => {
                // ADR-040 §2.3/§3: identity is the exact bit pattern with NaN canonicalized —
                // `+0.0`/`-0.0` stay distinct addresses; every NaN is ONE address. `Value::new`
                // already canonicalizes on construction, so this re-canonicalization is idempotent
                // there — it makes the one-NaN-address rule hold *by construction* on every hash
                // path, not by trusting the caller.
                self.tag(tag::PAYLOAD_FLOAT);
                self.f64(crate::value::canonical_float(*x));
            }
            Payload::Seq(elems) => {
                // Length-prefixed, then each element's identity-bearing content (repr + payload) via
                // the same `value` encoder — recursive and injective, so two sequences collide iff
                // they are element-wise identical (RFC-0032 D3).
                self.tag(tag::PAYLOAD_SEQ);
                self.u64(elems.len() as u64);
                for e in elems {
                    self.value(e);
                }
            }
            Payload::Bytes(bytes) => {
                // Length-prefixed raw bytes — injective, so two byte strings collide iff identical
                // (RFC-0032 D4).
                self.tag(tag::PAYLOAD_BYTES);
                self.blob(bytes);
            }
        }
    }

    /// The identity-bearing part of a value: its `Repr` (type, incl. paradigm) and its literal
    /// payload. `Meta` is dynamic metadata and is deliberately excluded (RFC-0001 §4.6).
    pub(crate) fn value(&mut self, v: &Value) {
        self.repr(v.repr());
        self.payload(v.payload());
    }

    /// Encode a node under a binder scope (innermost binder last), α-renaming bound variables to
    /// de Bruijn indices so binder *names* never reach the hash.
    ///
    /// **Iterative (RFC-0041 §4.5, W3):** an explicit action-worklist replaces native recursion so a
    /// deeply-nested node spine hashes with **bounded** native stack (a recursive `node` overflows →
    /// `SIGABRT`, which violates never-silent G2). The byte stream is **byte-for-byte identical** to
    /// the former recursive encoding — content addresses are unchanged (bar (a); mutation-witnessed):
    /// every immediate byte (tag / operator / length / repr / policy / literal) is emitted at exactly
    /// the same point, and the de Bruijn scope is pushed/popped by explicit `PushScope`/`PushBinders`/
    /// `PopTo` actions interleaved in the same order the recursion would (so each `Var` sees the same
    /// scope). Scope holds **references** to binder names (never emitted — only used to index), so no
    /// name is cloned.
    fn node(&mut self, root: &Node) {
        // An action processed LIFO. Non-`Visit` actions are the emissions/scope-ops that, in the
        // recursion, happened *between or after* a node's children (e.g. a `Swap`'s `target`/`policy`
        // after its `src`, or a binder scope pop after a body). To schedule a node's forward action
        // sequence on a LIFO stack, we push it **reversed**.
        enum Act<'a> {
            Visit(&'a Node),
            Tag(u8),
            U64(u64),
            Str(&'a str),
            Repr(&'a Repr),
            Value(&'a Value),
            Ctor(&'a crate::data::CtorRef),
            PushScope(&'a VarId),
            PushBinders(&'a [VarId]),
            PopTo(usize),
        }

        let mut scope: Vec<&VarId> = Vec::new();
        let mut acts: Vec<Act<'_>> = vec![Act::Visit(root)];

        while let Some(act) = acts.pop() {
            match act {
                Act::Tag(t) => self.tag(t),
                Act::U64(n) => self.u64(n),
                Act::Str(s) => self.str(s),
                Act::Repr(r) => self.repr(r),
                Act::Value(v) => self.value(v),
                Act::Ctor(c) => self.ctor_ref(c),
                Act::PushScope(name) => scope.push(name),
                Act::PushBinders(bs) => scope.extend(bs.iter()),
                Act::PopTo(mark) => scope.truncate(mark),
                Act::Visit(n) => match n {
                    Node::Var(name) => {
                        // Innermost-first search; a bound var becomes its de Bruijn index, a free var
                        // keeps its name (a free name is part of an open term's contract).
                        if let Some(pos) = scope.iter().rposition(|&b| b == name) {
                            let de_bruijn = (scope.len() - 1 - pos) as u32;
                            self.tag(tag::VAR_BOUND);
                            self.u32(de_bruijn);
                        } else {
                            self.tag(tag::VAR_FREE);
                            self.str(name);
                        }
                    }
                    Node::Const(v) => {
                        self.tag(tag::CONST);
                        self.value(v);
                    }
                    Node::Let { id, bound, body } => {
                        // forward: LET · node(bound) · push(id) · node(body) · pop
                        self.tag(tag::LET);
                        let mark = scope.len();
                        acts.push(Act::PopTo(mark));
                        acts.push(Act::Visit(body));
                        acts.push(Act::PushScope(id));
                        acts.push(Act::Visit(bound));
                    }
                    Node::Op { prim, args } => {
                        self.tag(tag::OP);
                        self.str(prim); // the operator IS identity-bearing
                        self.u64(args.len() as u64);
                        for a in args.iter().rev() {
                            acts.push(Act::Visit(a));
                        }
                    }
                    Node::Swap {
                        src,
                        target,
                        policy,
                    } => {
                        // forward: SWAP · node(src) · repr(target) · str(policy)
                        self.tag(tag::SWAP);
                        acts.push(Act::Str(policy.as_str()));
                        acts.push(Act::Repr(target));
                        acts.push(Act::Visit(src));
                    }
                    Node::Construct { ctor, args } => {
                        self.tag(tag::CONSTRUCT);
                        self.ctor_ref(ctor); // the constructor identity (#T#i) is identity-bearing
                        self.u64(args.len() as u64);
                        for a in args.iter().rev() {
                            acts.push(Act::Visit(a));
                        }
                    }
                    Node::Match {
                        scrutinee,
                        alts,
                        default,
                    } => {
                        self.tag(tag::MATCH);
                        // Build the forward action sequence, then push it reversed. The per-alt binder
                        // scope pops back to the match baseline (scope is balanced across scrutinee
                        // and each alt, so the baseline is constant — matching the recursive `mark`).
                        let baseline = scope.len();
                        let mut seq: Vec<Act<'_>> = Vec::new();
                        seq.push(Act::Visit(scrutinee));
                        seq.push(Act::U64(alts.len() as u64));
                        for alt in alts {
                            match alt {
                                Alt::Ctor {
                                    ctor,
                                    binders,
                                    body,
                                } => {
                                    seq.push(Act::Tag(tag::ALT_CTOR));
                                    seq.push(Act::Ctor(ctor));
                                    // Binder names are α-normalised (not hashed); count + positions
                                    // ride the de Bruijn scope the body is hashed under.
                                    seq.push(Act::U64(binders.len() as u64));
                                    seq.push(Act::PushBinders(binders));
                                    seq.push(Act::Visit(body));
                                    seq.push(Act::PopTo(baseline));
                                }
                                Alt::Lit { value, body } => {
                                    seq.push(Act::Tag(tag::ALT_LIT));
                                    seq.push(Act::Value(value)); // literal is identity-bearing
                                    seq.push(Act::Visit(body));
                                }
                            }
                        }
                        match default {
                            Some(d) => {
                                seq.push(Act::Tag(tag::MATCH_DEFAULT));
                                seq.push(Act::Visit(d));
                            }
                            None => seq.push(Act::Tag(tag::MATCH_NO_DEFAULT)),
                        }
                        for a in seq.into_iter().rev() {
                            acts.push(a);
                        }
                    }
                    Node::Lam { param, body } => {
                        // forward: LAM · push(param) · node(body) · pop
                        self.tag(tag::LAM);
                        let mark = scope.len();
                        acts.push(Act::PopTo(mark));
                        acts.push(Act::Visit(body));
                        acts.push(Act::PushScope(param));
                    }
                    Node::App { func, arg } => {
                        // forward: APP · node(func) · node(arg)
                        self.tag(tag::APP);
                        acts.push(Act::Visit(arg));
                        acts.push(Act::Visit(func));
                    }
                    Node::Fix { name, body } => {
                        // forward: FIX · push(name) · node(body) · pop
                        self.tag(tag::FIX);
                        let mark = scope.len();
                        acts.push(Act::PopTo(mark));
                        acts.push(Act::Visit(body));
                        acts.push(Act::PushScope(name));
                    }
                    Node::FixGroup { defs, body } => {
                        // forward: FIXGROUP · u64(len) · push(all names) · node(each def) · node(body)
                        // · pop. All member names enter scope as one frame before any def/body body
                        // is hashed (mutual recursion), matching the recursive encoder.
                        self.tag(tag::FIXGROUP);
                        self.u64(defs.len() as u64);
                        let mark = scope.len();
                        let mut seq: Vec<Act<'_>> = Vec::new();
                        for (name, _) in defs {
                            seq.push(Act::PushScope(name));
                        }
                        for (_, d) in defs {
                            seq.push(Act::Visit(d));
                        }
                        seq.push(Act::Visit(body));
                        seq.push(Act::PopTo(mark));
                        for a in seq.into_iter().rev() {
                            acts.push(a);
                        }
                    }
                },
            }
        }
    }
}

impl Value {
    /// The content hash of this value's *identity-bearing* content: its [`Repr`] and payload, with
    /// all dynamic [`crate::meta::Meta`] excluded (RFC-0001 §4.6). Two values with identical
    /// repr+payload but different provenance/bounds collide; differing paradigm or literal does not.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        let mut c = Canon::new();
        c.value(self);
        c.finish()
    }
}

impl Node {
    /// The content hash of this definition (RFC-0001 §4.6; ADR-003). Identity is over the
    /// α-normalized structure, types-with-`Repr`, constant literals, operator names, and swap
    /// contracts — never over binder names or dynamic value metadata. Hence: trivial renames do not
    /// change the hash; identical definitions collide; a paradigm change does not.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        let mut c = Canon::new();
        c.node(self);
        c.finish()
    }
}

/// The content address of a *primitive operation* identified by its name — for the `op` field of a
/// [`crate::meta::Provenance::Derived`] record produced by the interpreter (M-110). Domain-separated
/// from node/value hashes so a prim name can never collide with a structural hash.
#[must_use]
pub fn operation_hash(prim: &str) -> ContentHash {
    let mut c = Canon::new();
    c.tag(0x07); // PRIM domain tag (distinct from the node/repr/payload tags above)
    c.str(prim);
    c.finish()
}

/// The separable `hash ↔ name` side-table (RFC-0001 §4.6, "names-as-metadata"). Names live *here*,
/// not in identity, so they can be attached, changed, or dropped without affecting a definition's
/// [`ContentHash`]. This is the kernel-side model of Unison's name store (ADR-003).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Names {
    map: HashMap<ContentHash, String>,
}

impl Names {
    /// An empty name table.
    #[must_use]
    pub fn new() -> Self {
        Names {
            map: HashMap::new(),
        }
    }

    /// Bind a human name to a content hash, returning any previous name for that hash. Re-binding a
    /// different name is allowed and changes nothing about identity (that is the whole point).
    pub fn bind(&mut self, hash: ContentHash, name: impl Into<String>) -> Option<String> {
        self.map.insert(hash, name.into())
    }

    /// The name bound to `hash`, if any.
    #[must_use]
    pub fn name_of(&self, hash: &ContentHash) -> Option<&str> {
        self.map.get(hash).map(String::as_str)
    }

    /// Number of bound names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
