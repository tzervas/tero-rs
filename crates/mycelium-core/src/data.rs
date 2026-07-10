//! The **data registry `Σ`** and constructor references (`#T#i`) — RFC-0001 §4.3 (r3); RFC-0007
//! §4.2; ADR-003 (Unison identity). M-320.
//!
//! r3 folds algebraic data into the Core IR (RFC-0011). A data **value** is built by a
//! [`Node::Construct`](crate::Node) and scrutinised by a [`Node::Match`](crate::Node); the data
//! **declarations** those nodes reference live **here, in the registry — not in the term grammar**
//! (the GHC-Core/Lean/Coq/Unison convergence, RFC-0007 §4.2). Keeping declarations out of the term
//! language is what preserves WF4: a `Construct`/`Match` node hashes over its structure plus the
//! [`CtorRef`] hashes it mentions, and the term grammar does not grow with every data type.
//!
//! # Identity (RFC-0007 §4.2; ADR-003)
//! A declaration `type T<a…> = C₁(τ…) | … | Cₙ(τ…)` is content-addressed over its **α-normalised
//! structure** — constructor order is significant, field types (incl. their `Repr`) are
//! significant, **names are not identity**. A constructor reference is `#T#i` ([`CtorRef`]): the
//! declaration hash ‖ the constructor index. A **self-recursive** declaration hashes its own
//! occurrences as a cycle **placeholder** (the Unison scheme): `Nat = Z | S(Nat)`'s `S` field is a
//! back-reference, so it is encoded as a placeholder, never the (circular) final hash.
//!
//! # Scope (honesty)
//! **Self-recursion is fully realised and tested** (`Nat`, `Bytes`, `List`-shaped types). **Mutual
//! recursion** (a multi-member cycle) now content-addresses **canonically and name-independently**
//! via `canonical_cycle_order` (the Unison recipe; R7-Q3 cycle-ordering closed in RFC-0001 r4).
//! The surface→registry *elaboration* of mutual recursion stays deferred (the L1 prototype accepts
//! only self-recursion), but the **identity** of a mutually-recursive group is now correct, not
//! provisional — so when the surface grows mutual recursion, the hashes do not change underneath it.
//!
//! # ADR-033 FLAG-1 Path A — full function signature encoding (`Empirical` post-test, `Declared` claim)
//! `FieldSpec::Fn` carries a full `FnSig` (all parameter types + return type) so two function-typed
//! fields with different signatures hash to distinct content addresses. The soundness gap described
//! in ADR-033 §10.1 is closed at the kernel level: `MkDict_Eq8 ≠ MkDict_Eq16` as content-addressed
//! declarations, so a type-confused `Match`/projection is a **never-silent no-match** by
//! construction (G2). Tag: `Empirical` (tested — `src/tests/data.rs`); NOT `Proven` (no mechanized
//! injectivity proof; VR-5 forbids the upgrade).

use std::collections::BTreeMap;

use crate::content::Canon;
use crate::id::ContentHash;
use crate::repr::Repr;

/// A constructor reference `#T#i` (RFC-0007 §4.2): the content hash of a data declaration and the
/// constructor's index within it. Two constructors are the *same* constructor iff their declaration
/// hash and index agree — names play no part (ADR-003).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CtorRef {
    decl: ContentHash,
    index: u32,
}

impl CtorRef {
    /// Build a constructor reference from a declaration hash and a constructor index.
    #[must_use]
    pub fn new(decl: ContentHash, index: u32) -> Self {
        CtorRef { decl, index }
    }

    /// The referenced data declaration's content hash (`#T`).
    #[must_use]
    pub fn decl(&self) -> &ContentHash {
        &self.decl
    }

    /// The constructor's index within its declaration (`#i`).
    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl core::fmt::Display for CtorRef {
    /// The Unison spelling `#<declhash>#<i>` (RFC-0007 §4.2).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}#{}", self.decl.as_str(), self.index)
    }
}

// --- Resolved field-type references for function signatures ------------------------------------

/// A resolved field-type reference that can appear inside a function signature: a `Repr` leaf, a
/// `Data` leaf (resolved to its declaration hash), or a nested function type. This is the resolved
/// analogue of [`FieldTyRef`] (ADR-033 §10 PATH-A).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedFieldTyRef {
    /// A representation-typed parameter/return (`Binary{n}` | `Ternary{m}` | …).
    Repr(Repr),
    /// A data-typed parameter/return, resolved to the referenced declaration's content hash.
    Data(ContentHash),
    /// A nested function type — a higher-order parameter or return.
    Fn(Box<ResolvedFnSig>),
}

/// A resolved function signature: the parameter types (in order) and the return type.
/// `arity == params.len()` always holds in a resolved signature (`Declared` well-formedness
/// invariant, checked at build time — [`RegistryError::FnArityMismatch`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFnSig {
    /// Parameter types, in declaration order (`arity == params.len()`).
    pub params: Vec<ResolvedFieldTyRef>,
    /// Return type.
    pub ret: Box<ResolvedFieldTyRef>,
}

/// A field type within a resolved declaration: a representation type, a (possibly cyclic) data
/// type reference, or a function-typed field (ADR-033). This is the *identity-bearing* field shape
/// (RFC-0007 §4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldTy {
    /// A representation-typed field (`Binary{n}` | `Ternary{m}` | `Dense{…}` | `VSA{…}`).
    Repr(Repr),
    /// A data-typed field, referencing another (or the same, recursively) declaration by hash.
    Data(ContentHash),
    /// A function-typed field with a resolved full signature (ADR-033 §10 PATH-A).
    /// The signature encodes all parameter types + return type so distinct fn types hash
    /// distinctly — closing the soundness gap described in ADR-033 §10.1.
    Fn {
        /// The arity (== `sig.params.len()`).
        arity: u32,
        /// The resolved full signature.
        sig: ResolvedFnSig,
    },
}

/// One constructor of a resolved declaration: its field types, in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CtorDecl {
    /// Field types, in declaration order (the index is the field's `#i` position).
    pub fields: Vec<FieldTy>,
}

/// A resolved, content-addressed data declaration: its constructors in declaration order (the index
/// is the `#i` of [`CtorRef`]). Names are stored separately (they are not identity — ADR-003).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataDecl {
    /// Constructors, in declaration order.
    pub ctors: Vec<CtorDecl>,
}

// --- Build-time specs (names are build keys, never hashed) ------------------------------------

/// A build-time field-type reference for function signatures: the same leaf set as a data field
/// can hold, plus nested functions. Every leaf bottoms out in `Repr` (already injective via
/// `Canon::repr`) or `Data(name)` (resolved to `ContentHash` at build time) — no unbounded
/// recursion. `Declared`-with-argument: recursion is well-founded (ADR-033 §10.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldTyRef {
    /// A representation-typed parameter/return.
    Repr(Repr),
    /// A data-typed parameter/return, referenced by build-time name.
    Data(String),
    /// A nested function type (higher-order parameter or return).
    Fn(Box<FnSig>),
}

/// A build-time function signature: the parameter types (in order) and the return type.
/// `arity == params.len()` is the well-formedness invariant; a mismatch is an explicit
/// [`RegistryError::FnArityMismatch`] (never silent — G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    /// Explicit (non-dictionary) parameter count — must equal `params.len()`.
    pub arity: u32,
    /// Parameter types, in declaration order.
    pub params: Vec<FieldTyRef>,
    /// Return type.
    pub ret: Box<FieldTyRef>,
}

/// A build-time field spec: a representation field, a data field referencing another declaration
/// **by name**, or a function-typed field with its full signature (ADR-033 §10 PATH-A).
/// The name is a build key for resolving references — it is *not* hashed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldSpec {
    /// A representation-typed field.
    Repr(Repr),
    /// A data-typed field, by the referenced declaration's build-time name.
    Data(String),
    /// A function-typed field with full resolved signature (ADR-033 §10 PATH-A — FLAG-1 fix).
    /// Two function fields with different parameter/return types produce different content hashes,
    /// so `MkDict_Eq8 ≠ MkDict_Eq16` at the kernel level. `Empirical` guarantee (tested).
    Fn {
        /// Explicit parameter count — must equal `sig.params.len()` (checked at build;
        /// mismatch → [`RegistryError::FnArityMismatch`]).
        arity: u32,
        /// The full function signature (params + return).
        sig: FnSig,
    },
}

/// A build-time constructor spec: its fields, in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CtorSpec {
    /// The fields, in declaration order.
    pub fields: Vec<FieldSpec>,
}

/// A build-time declaration spec: its constructors, in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclSpec {
    /// The constructors, in declaration order.
    pub ctors: Vec<CtorSpec>,
}

/// Why building a [`DataRegistry`] from specs failed — always explicit (never a silent drop).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// A field references a data declaration name that is not in the spec set.
    UnknownTypeRef {
        /// The declaration whose field has the dangling reference.
        in_decl: String,
        /// The unresolved referenced name.
        missing: String,
    },
    /// A `FieldSpec::Fn` (or a `FieldTyRef::Fn` inside a signature) has `arity ≠ params.len()`.
    /// The `arity` field is the stated count; `params_len` is the actual length of `params`.
    /// Never-silent (G2): both values are named so the caller can report the exact mismatch.
    FnArityMismatch {
        /// The declaration containing the malformed field.
        in_decl: String,
        /// The stated arity.
        arity: u32,
        /// The actual number of parameter types in `sig.params`.
        params_len: usize,
    },
}

impl core::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RegistryError::UnknownTypeRef { in_decl, missing } => write!(
                f,
                "data declaration `{in_decl}` references unknown type `{missing}`"
            ),
            RegistryError::FnArityMismatch {
                in_decl,
                arity,
                params_len,
            } => write!(
                f,
                "data declaration `{in_decl}` has a Fn field with arity {arity} \
                 but {params_len} parameter type(s) in its signature"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// The content-addressed data registry `Σ` (RFC-0001 §4.3 r3): the resolved declarations keyed by
/// their content hash, plus the build-time `name → hash` resolution used to form [`CtorRef`]s.
///
/// Built once from a set of [`DeclSpec`]s ([`DataRegistry::build`]); the elaborator and the
/// interpreter share *one* registry so that a constructor's identity (`#T#i`) is the same on every
/// execution path (the NFR-7 differential is about *one* `CtorRef` set, never two).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DataRegistry {
    /// Resolved declarations, keyed by content hash.
    decls: BTreeMap<ContentHash, DataDecl>,
    /// Build-time name → content hash (names are metadata, kept only for reference resolution).
    by_name: BTreeMap<String, ContentHash>,
}

impl DataRegistry {
    /// Build the registry from a set of named declaration specs, computing every declaration's
    /// content hash (cycle-aware: self-references hash as a placeholder, RFC-0007 §4.2). Returns an
    /// explicit [`RegistryError`] for any dangling reference or arity mismatch — never a partial
    /// registry.
    pub fn build(specs: &BTreeMap<String, DeclSpec>) -> Result<Self, RegistryError> {
        // Validate references and arity invariants first (never proceed on a dangling ref or
        // arity mismatch — G2/never-silent).
        for (name, decl) in specs {
            for ctor in &decl.ctors {
                for field in &ctor.fields {
                    validate_field_spec(name, field, specs)?;
                }
            }
        }

        let sccs = strongly_connected_components(specs);
        let mut by_name: BTreeMap<String, ContentHash> = BTreeMap::new();
        let mut decls: BTreeMap<ContentHash, DataDecl> = BTreeMap::new();

        // Process SCCs dependencies-first (the SCC list is already in reverse-topological order),
        // so every out-of-cycle data reference already has a hash when we encode a member.
        for raw_scc in &sccs {
            // r4 (R7-Q3): order the cycle's members **structurally, name-independently** — the Unison
            // recipe — so a mutually-recursive group content-addresses deterministically regardless of
            // declaration names (ADR-003). Each member's *local* hash is computed with all in-cycle
            // references collapsed to a single shared placeholder (so the ordering can't depend on a
            // not-yet-assigned index); members sort by that hash, with the (metadata) name as the only
            // tie-break for the corner case of two structurally identical cyclic members. A singleton
            // (the r3-reachable self-recursion case) is unaffected — one member, order trivial.
            let scc = canonical_cycle_order(raw_scc, specs, &by_name);
            let scc = &scc;
            let in_cycle: BTreeMap<&str, usize> = scc
                .iter()
                .enumerate()
                .map(|(i, n)| (n.as_str(), i))
                .collect();

            // The group hash: encode each member's structure, in the canonical member order, with
            // in-cycle references as their placeholder index and out-of-cycle references as their
            // (already computed) hash.
            let group_hash = {
                let mut c = Canon::new();
                c.tag(crate::content::tag::DATADECL);
                c.u64(scc.len() as u64);
                for name in scc {
                    let decl = &specs[name];
                    encode_decl(&mut c, decl, &in_cycle, &by_name);
                }
                c.finish()
            };

            // Each member's final hash = H(group ‖ member index) — distinguishing members of the
            // same cycle while sharing the cycle's structural identity (the Unison recipe).
            for (i, name) in scc.iter().enumerate() {
                let mut c = Canon::new();
                c.tag(crate::content::tag::DATADECL);
                c.hash(&group_hash);
                c.u32(i as u32);
                let member_hash = c.finish();
                by_name.insert(name.clone(), member_hash.clone());
            }

            // Now that every member has a hash, build the resolved `DataDecl`s (cycle references can
            // now be filled in with the just-computed member hashes).
            for name in scc {
                let decl = &specs[name];
                let resolved = resolve_decl(decl, &by_name);
                let hash = by_name[name].clone();
                decls.insert(hash, resolved);
            }
        }

        Ok(DataRegistry { decls, by_name })
    }

    /// The content hash of the declaration registered under build-time name `name`, if any.
    #[must_use]
    pub fn decl_hash(&self, name: &str) -> Option<&ContentHash> {
        self.by_name.get(name)
    }

    /// A [`CtorRef`] for constructor `index` of the declaration named `name`, if the declaration is
    /// registered and the index is in range.
    #[must_use]
    pub fn ctor_ref(&self, name: &str, index: u32) -> Option<CtorRef> {
        let hash = self.by_name.get(name)?;
        let decl = self.decls.get(hash)?;
        if (index as usize) < decl.ctors.len() {
            Some(CtorRef::new(hash.clone(), index))
        } else {
            None
        }
    }

    /// The resolved declaration at content hash `hash`, if registered.
    #[must_use]
    pub fn decl(&self, hash: &ContentHash) -> Option<&DataDecl> {
        self.decls.get(hash)
    }

    /// The constructor declaration a [`CtorRef`] points at, if registered and in range.
    #[must_use]
    pub fn ctor(&self, ctor: &CtorRef) -> Option<&CtorDecl> {
        self.decls
            .get(ctor.decl())
            .and_then(|d| d.ctors.get(ctor.index() as usize))
    }

    /// The number of fields the referenced constructor takes (its saturation arity, WF6).
    #[must_use]
    pub fn field_count(&self, ctor: &CtorRef) -> Option<usize> {
        self.ctor(ctor).map(|c| c.fields.len())
    }

    /// The number of constructors of the data type the [`CtorRef`] belongs to (for WF7 coverage).
    #[must_use]
    pub fn ctor_count(&self, ctor: &CtorRef) -> Option<usize> {
        self.decls.get(ctor.decl()).map(|d| d.ctors.len())
    }
}

// --- Validation helpers -----------------------------------------------------------------------

/// Validate a single [`FieldSpec`] (and, recursively, any [`FieldTyRef`] nodes inside `Fn`
/// signatures): check that all `Data` references name existing declarations, and that every `Fn`'s
/// `arity == sig.params.len()`. Never-silent on either violation (G2).
fn validate_field_spec(
    in_decl: &str,
    field: &FieldSpec,
    specs: &BTreeMap<String, DeclSpec>,
) -> Result<(), RegistryError> {
    match field {
        FieldSpec::Repr(_) => Ok(()),
        FieldSpec::Data(r) => {
            if !specs.contains_key(r) {
                Err(RegistryError::UnknownTypeRef {
                    in_decl: in_decl.to_owned(),
                    missing: r.clone(),
                })
            } else {
                Ok(())
            }
        }
        FieldSpec::Fn { arity, sig } => {
            // Well-formedness: arity == params.len() (never-silent — RegistryError::FnArityMismatch).
            if *arity as usize != sig.params.len() {
                return Err(RegistryError::FnArityMismatch {
                    in_decl: in_decl.to_owned(),
                    arity: *arity,
                    params_len: sig.params.len(),
                });
            }
            // Validate each param and the return type recursively.
            for param in &sig.params {
                validate_field_ty_ref(in_decl, param, specs)?;
            }
            validate_field_ty_ref(in_decl, &sig.ret, specs)
        }
    }
}

/// Validate a [`FieldTyRef`] (parameter or return inside a `Fn` signature) recursively.
fn validate_field_ty_ref(
    in_decl: &str,
    ty_ref: &FieldTyRef,
    specs: &BTreeMap<String, DeclSpec>,
) -> Result<(), RegistryError> {
    match ty_ref {
        FieldTyRef::Repr(_) => Ok(()),
        FieldTyRef::Data(r) => {
            if !specs.contains_key(r) {
                Err(RegistryError::UnknownTypeRef {
                    in_decl: in_decl.to_owned(),
                    missing: r.clone(),
                })
            } else {
                Ok(())
            }
        }
        FieldTyRef::Fn(nested) => {
            if nested.arity as usize != nested.params.len() {
                return Err(RegistryError::FnArityMismatch {
                    in_decl: in_decl.to_owned(),
                    arity: nested.arity,
                    params_len: nested.params.len(),
                });
            }
            for param in &nested.params {
                validate_field_ty_ref(in_decl, param, specs)?;
            }
            validate_field_ty_ref(in_decl, &nested.ret, specs)
        }
    }
}

// --- Encoding ---------------------------------------------------------------------------------

/// The **canonical, name-independent order** of a strongly-connected declaration group (the Unison
/// cycle recipe, RFC-0007 §4.2; R7-Q3). Each member is keyed by a *local* hash computed with all
/// in-cycle references collapsed to one shared placeholder — so the ordering depends only on
/// structure, never on a member's (not-yet-assigned) cycle index or its name. Members sort by that
/// hash; the metadata name is the deterministic tie-break for the corner case of two structurally
/// identical cyclic members. A singleton group is returned unchanged.
fn canonical_cycle_order(
    scc: &[String],
    specs: &BTreeMap<String, DeclSpec>,
    by_name: &BTreeMap<String, ContentHash>,
) -> Vec<String> {
    if scc.len() == 1 {
        return scc.to_vec();
    }
    // All members map to the *same* placeholder (0) for the ordering pass — structure only.
    let shared: BTreeMap<&str, usize> = scc.iter().map(|n| (n.as_str(), 0usize)).collect();
    let mut keyed: Vec<(ContentHash, String)> = scc
        .iter()
        .map(|name| {
            let mut c = Canon::new();
            c.tag(crate::content::tag::DATADECL);
            encode_decl(&mut c, &specs[name], &shared, by_name);
            (c.finish(), name.clone())
        })
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, n)| n).collect()
}

/// Encode a declaration's identity-bearing structure into `c`: each constructor (order significant)
/// and its fields (order significant), with names excluded. In-cycle data references become a
/// placeholder index; out-of-cycle data references become the referenced hash.
///
/// For `FieldSpec::Fn`, the full signature (params + return) is encoded with fresh tags
/// (`FIELD_FN`, `FN_SIG_*`, `FTR_*`) so distinct fn types never collide (ADR-033 §10 PATH-A).
fn encode_decl(
    c: &mut Canon,
    decl: &DeclSpec,
    in_cycle: &BTreeMap<&str, usize>,
    by_name: &BTreeMap<String, ContentHash>,
) {
    c.tag(crate::content::tag::CTOR_DECL);
    c.u64(decl.ctors.len() as u64);
    for ctor in &decl.ctors {
        c.u64(ctor.fields.len() as u64);
        for field in &ctor.fields {
            encode_field_spec(c, field, in_cycle, by_name);
        }
    }
}

/// Encode a single [`FieldSpec`] into `c`, cycle-aware.
fn encode_field_spec(
    c: &mut Canon,
    field: &FieldSpec,
    in_cycle: &BTreeMap<&str, usize>,
    by_name: &BTreeMap<String, ContentHash>,
) {
    match field {
        FieldSpec::Repr(r) => {
            c.tag(crate::content::tag::FIELD_REPR);
            c.repr(r);
        }
        FieldSpec::Data(name) => {
            if let Some(&idx) = in_cycle.get(name.as_str()) {
                c.tag(crate::content::tag::FIELD_CYCLE);
                c.u32(idx as u32);
            } else {
                c.tag(crate::content::tag::FIELD_DATA);
                // Resolved earlier (dependencies-first); a reference inside the validated
                // spec set with a non-cycle target always has a hash by now.
                c.hash(&by_name[name]);
            }
        }
        FieldSpec::Fn { arity, sig } => {
            // ADR-033 §10 PATH-A: emit FIELD_FN + arity + full signature.
            // The arity is redundant with sig.params.len() (validated at build), but is encoded
            // separately so the saturation invariant is readable without parsing the signature.
            c.tag(crate::content::tag::FIELD_FN);
            c.u32(*arity);
            encode_fn_sig(c, sig, in_cycle, by_name);
        }
    }
}

/// Encode a [`FnSig`] (params + return) into `c`, cycle-aware.
/// Tag layout: `FN_SIG_PARAMS u64(count) [param]… FN_SIG_RET [ret]`
fn encode_fn_sig(
    c: &mut Canon,
    sig: &FnSig,
    in_cycle: &BTreeMap<&str, usize>,
    by_name: &BTreeMap<String, ContentHash>,
) {
    c.tag(crate::content::tag::FN_SIG_PARAMS);
    c.u64(sig.params.len() as u64);
    for param in &sig.params {
        encode_field_ty_ref(c, param, in_cycle, by_name);
    }
    c.tag(crate::content::tag::FN_SIG_RET);
    encode_field_ty_ref(c, &sig.ret, in_cycle, by_name);
}

/// Encode a [`FieldTyRef`] (a param or return leaf) into `c`, cycle-aware.
fn encode_field_ty_ref(
    c: &mut Canon,
    ty_ref: &FieldTyRef,
    in_cycle: &BTreeMap<&str, usize>,
    by_name: &BTreeMap<String, ContentHash>,
) {
    match ty_ref {
        FieldTyRef::Repr(r) => {
            c.tag(crate::content::tag::FTR_REPR);
            c.repr(r);
        }
        FieldTyRef::Data(name) => {
            // A `Data` leaf inside a signature is a genuine declaration edge: if it names an
            // in-cycle declaration, it must use the placeholder, not the (circular) final hash —
            // exactly the same rule as a top-level `FieldSpec::Data` (FLAG-3, ADR-033 §10.2 Q3).
            if let Some(&idx) = in_cycle.get(name.as_str()) {
                c.tag(crate::content::tag::FTR_DATA_CYCLE);
                c.u32(idx as u32);
            } else {
                c.tag(crate::content::tag::FTR_DATA);
                c.hash(&by_name[name]);
            }
        }
        FieldTyRef::Fn(nested) => {
            c.tag(crate::content::tag::FTR_FN);
            encode_fn_sig(c, nested, in_cycle, by_name);
        }
    }
}

// --- Resolution -------------------------------------------------------------------------------

/// Build the resolved [`DataDecl`] for `decl`, with each data field (and each `Data` leaf inside
/// `Fn` signatures) carrying the referenced declaration's (now-computed) hash.
fn resolve_decl(decl: &DeclSpec, by_name: &BTreeMap<String, ContentHash>) -> DataDecl {
    DataDecl {
        ctors: decl
            .ctors
            .iter()
            .map(|ctor| CtorDecl {
                fields: ctor
                    .fields
                    .iter()
                    .map(|f| resolve_field_spec(f, by_name))
                    .collect(),
            })
            .collect(),
    }
}

fn resolve_field_spec(field: &FieldSpec, by_name: &BTreeMap<String, ContentHash>) -> FieldTy {
    match field {
        FieldSpec::Repr(r) => FieldTy::Repr(r.clone()),
        FieldSpec::Data(name) => FieldTy::Data(by_name[name].clone()),
        FieldSpec::Fn { arity, sig } => FieldTy::Fn {
            arity: *arity,
            sig: resolve_fn_sig(sig, by_name),
        },
    }
}

fn resolve_fn_sig(sig: &FnSig, by_name: &BTreeMap<String, ContentHash>) -> ResolvedFnSig {
    ResolvedFnSig {
        params: sig
            .params
            .iter()
            .map(|p| resolve_field_ty_ref(p, by_name))
            .collect(),
        ret: Box::new(resolve_field_ty_ref(&sig.ret, by_name)),
    }
}

fn resolve_field_ty_ref(
    ty_ref: &FieldTyRef,
    by_name: &BTreeMap<String, ContentHash>,
) -> ResolvedFieldTyRef {
    match ty_ref {
        FieldTyRef::Repr(r) => ResolvedFieldTyRef::Repr(r.clone()),
        FieldTyRef::Data(name) => ResolvedFieldTyRef::Data(by_name[name].clone()),
        FieldTyRef::Fn(nested) => ResolvedFieldTyRef::Fn(Box::new(resolve_fn_sig(nested, by_name))),
    }
}

// --- SCC (Tarjan) -----------------------------------------------------------------------------

/// Tarjan's strongly-connected components over the declaration dependency graph (an edge `A → B`
/// when `A` has a data field of type `B`, **or** when a `Fn` signature inside `A` has a `Data`
/// parameter/return referencing `B` — FLAG-3, ADR-033 §10.2 Q3). Returns the SCCs in
/// **reverse-topological order** (dependencies before dependents).
fn strongly_connected_components(specs: &BTreeMap<String, DeclSpec>) -> Vec<Vec<String>> {
    // Successors: the distinct data-declaration references reachable from each declaration,
    // including those inside `Fn` signatures (FLAG-3: Data edges inside sigs are genuine
    // declaration deps and must participate in the SCC graph — ADR-033 §10.2 Q3).
    let succ = |name: &str| -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for ctor in &specs[name].ctors {
            for field in &ctor.fields {
                collect_data_refs_field_spec(field, &mut out);
            }
        }
        out
    };

    struct Tarjan {
        index: BTreeMap<String, usize>,
        low: BTreeMap<String, usize>,
        on_stack: BTreeMap<String, bool>,
        stack: Vec<String>,
        next: usize,
        out: Vec<Vec<String>>,
    }

    impl Tarjan {
        fn run(&mut self, succ: &dyn Fn(&str) -> Vec<String>, v: &str) {
            self.index.insert(v.to_owned(), self.next);
            self.low.insert(v.to_owned(), self.next);
            self.next += 1;
            self.stack.push(v.to_owned());
            self.on_stack.insert(v.to_owned(), true);

            for w in succ(v) {
                if !self.index.contains_key(&w) {
                    self.run(succ, &w);
                    let lw = self.low[&w];
                    let lv = self.low[v];
                    self.low.insert(v.to_owned(), lv.min(lw));
                } else if *self.on_stack.get(&w).unwrap_or(&false) {
                    let iw = self.index[&w];
                    let lv = self.low[v];
                    self.low.insert(v.to_owned(), lv.min(iw));
                }
            }

            if self.low[v] == self.index[v] {
                let mut scc = Vec::new();
                loop {
                    let w = self.stack.pop().expect("non-empty while popping an SCC");
                    self.on_stack.insert(w.clone(), false);
                    scc.push(w.clone());
                    if w == v {
                        break;
                    }
                }
                // Tarjan emits SCCs in reverse-topological order already. A by-name sort gives a
                // deterministic *raw* order; the structural, name-independent canonical ordering is
                // applied later in `build` via `canonical_cycle_order` (R7-Q3).
                scc.sort();
                self.out.push(scc);
            }
        }
    }

    let mut t = Tarjan {
        index: BTreeMap::new(),
        low: BTreeMap::new(),
        on_stack: BTreeMap::new(),
        stack: Vec::new(),
        next: 0,
        out: Vec::new(),
    };
    for name in specs.keys() {
        if !t.index.contains_key(name) {
            t.run(&succ, name);
        }
    }
    t.out
}

/// Collect distinct data-declaration names reachable from a [`FieldSpec`] — including those nested
/// inside `Fn` signatures (FLAG-3, ADR-033 §10.2 Q3). Appends to `out`; deduplicates.
fn collect_data_refs_field_spec(field: &FieldSpec, out: &mut Vec<String>) {
    match field {
        FieldSpec::Repr(_) => {}
        FieldSpec::Data(r) => {
            if !out.contains(r) {
                out.push(r.clone());
            }
        }
        FieldSpec::Fn { sig, .. } => {
            collect_data_refs_fn_sig(sig, out);
        }
    }
}

/// Collect data-declaration names reachable from a [`FnSig`] (params + return).
fn collect_data_refs_fn_sig(sig: &FnSig, out: &mut Vec<String>) {
    for param in &sig.params {
        collect_data_refs_field_ty_ref(param, out);
    }
    collect_data_refs_field_ty_ref(&sig.ret, out);
}

/// Collect data-declaration names reachable from a [`FieldTyRef`] leaf.
fn collect_data_refs_field_ty_ref(ty_ref: &FieldTyRef, out: &mut Vec<String>) {
    match ty_ref {
        FieldTyRef::Repr(_) => {}
        FieldTyRef::Data(r) => {
            if !out.contains(r) {
                out.push(r.clone());
            }
        }
        FieldTyRef::Fn(nested) => {
            collect_data_refs_fn_sig(nested, out);
        }
    }
}
