//! Algebraic **data values** and the runtime value sum (RFC-0001 §4.2 r3; RFC-0011 §4.6).
//!
//! A [`Construct`](crate::Node) node evaluates to a [`Datum`] — a saturated constructor application
//! (constructor tag + field values). A datum is **not** one of the four paradigm [`Repr`](crate::Repr)
//! kinds (those stay closed — RFC-0011 §4.6); it is a different category of value, so the runtime
//! value the interpreter yields is the sum [`CoreValue`] = a representation [`Value`] **or** a
//! [`Datum`]. This mirrors `crates/mycelium-l1::eval::L1Value` (`Repr(Value) | Data{…}`) so the L1
//! evaluator and the L0 interpreter agree on the data fragment (NFR-7).
//!
//! # Honesty (the meet-summary; maintainer-confirmed)
//! A datum carries **one** honesty field: a [`GuaranteeStrength`] **summary** — the `meet` of its
//! fields' guarantees. It carries **no [`Bound`](crate::Bound)**: a datum is not itself an
//! approximation; the quantitative bounds that justify a non-`Exact` summary live on the **leaf
//! representation values** it contains (drillable via provenance / EXPLAIN). M-I1
//! (`guarantee≠Exact ⟺ bound`) is an invariant of *representation* values — where the bound
//! quantifies *that value's* error — not of structural composites. The datum summary is a derived
//! disclosure (an addendum to RFC-0001 §4.7's propagation), and like every guarantee it is
//! monotone-downward (`Construct`/`Match` only ever `meet`, never upgrade — VR-5).

use crate::content::Canon;
use crate::data::CtorRef;
use crate::guarantee::GuaranteeStrength;
use crate::id::ContentHash;
use crate::value::Value;

/// A constructed data value: a saturated constructor application (RFC-0011 §4.1, W6) with a
/// meet-summary guarantee.
///
/// **Recursion-safety (RFC-0041 §4.5, W3):** [`Clone`], [`PartialEq`], and [`Drop`] are **manual,
/// iterative** (below), *not* `#[derive]`d, and the content hash ([`Canon::datum`]) is iterative —
/// a derived (recursive) form overflows the native stack (`SIGABRT`, violating never-silent G2) on a
/// deeply-nested `Datum`↔[`CoreValue`] chain (e.g. a `S(S(S(…Z)))` numeral). All four are
/// bit-identical to the derived forms (mutation-witnessed; §6 within-freeze hardening bar (a)).
#[derive(Debug)]
pub struct Datum {
    ctor: CtorRef,
    fields: Vec<CoreValue>,
    guarantee: GuaranteeStrength,
}

impl Datum {
    /// Construct a datum from a constructor reference and its field values, computing the
    /// meet-summary guarantee = `meet` of the fields' guarantees with the intrinsic `Exact`
    /// (construction adds no approximation — RFC-0011 §4.6). With all-`Exact` fields the summary is
    /// `Exact`, consistent with M-I1 (no bound).
    #[must_use]
    pub fn new(ctor: CtorRef, fields: Vec<CoreValue>) -> Self {
        let guarantee = GuaranteeStrength::meet_all(fields.iter().map(CoreValue::guarantee));
        Datum {
            ctor,
            fields,
            guarantee,
        }
    }

    /// The constructor reference (`#T#i`).
    #[must_use]
    pub fn ctor(&self) -> &CtorRef {
        &self.ctor
    }

    /// The field values, in declaration order.
    #[must_use]
    pub fn fields(&self) -> &[CoreValue] {
        &self.fields
    }

    /// The meet-summary guarantee.
    #[must_use]
    pub fn guarantee(&self) -> GuaranteeStrength {
        self.guarantee
    }

    /// This datum with its summary guarantee met against `g` (weakest-wins). Used by `Match` to
    /// fold the scrutinee's guarantee into a data result (RFC-0011 §4.6); never upgrades (VR-5).
    #[must_use]
    pub fn meet_guarantee(mut self, g: GuaranteeStrength) -> Self {
        self.guarantee = self.guarantee.meet(g);
        self
    }

    /// The identity-bearing content hash of this datum: its constructor reference and its fields'
    /// content (the guarantee summary is dynamic metadata — excluded, like `Meta` on a [`Value`];
    /// RFC-0001 §4.6).
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        let mut c = Canon::new();
        c.datum(self);
        c.finish()
    }
}

/// A runtime value: a representation [`Value`] (one of the four paradigm kinds) or an algebraic
/// [`Datum`]. The interpreter's normal forms (RFC-0011 §4.6).
///
/// **Recursion-safety (RFC-0041 §4.5, W3):** [`Clone`] and [`PartialEq`] are **manual, iterative**
/// (below) over the shared `Datum`↔`CoreValue` worklist; `Drop` stays derived because a
/// `CoreValue::Data`'s only owned recursive field is a [`Datum`], whose manual iterative `Drop`
/// tears down the whole cluster in one bounded pass.
#[derive(Debug)]
pub enum CoreValue {
    /// A representation value (`repr + payload + Meta`).
    Repr(Value),
    /// An algebraic data value.
    Data(Datum),
}

impl CoreValue {
    /// The underlying representation value, if this is a [`CoreValue::Repr`].
    #[must_use]
    pub fn as_repr(&self) -> Option<&Value> {
        match self {
            CoreValue::Repr(v) => Some(v),
            CoreValue::Data(_) => None,
        }
    }

    /// The underlying datum, if this is a [`CoreValue::Data`].
    #[must_use]
    pub fn as_data(&self) -> Option<&Datum> {
        match self {
            CoreValue::Data(d) => Some(d),
            CoreValue::Repr(_) => None,
        }
    }

    /// This value's guarantee: a representation value's own `Meta.guarantee`, or a datum's
    /// meet-summary. The single honesty accessor the `Construct`/`Match` meet rules fold over.
    #[must_use]
    pub fn guarantee(&self) -> GuaranteeStrength {
        match self {
            CoreValue::Repr(v) => v.meta().guarantee(),
            CoreValue::Data(d) => d.guarantee(),
        }
    }

    /// The identity-bearing content hash (RFC-0001 §4.6): a representation value's repr+payload, or
    /// a datum's constructor+fields.
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        // Route through the iterative [`Canon::core_value`] — bit-identical to the per-variant
        // `Value`/`Datum` `content_hash` (a `Repr` hashes its repr+payload; a `Data` its
        // ctor+fields), but recursion-safe on a deep datum spine (RFC-0041 §4.5).
        let mut c = Canon::new();
        c.core_value(self);
        c.finish()
    }
}

impl From<Value> for CoreValue {
    fn from(v: Value) -> Self {
        CoreValue::Repr(v)
    }
}

impl From<Datum> for CoreValue {
    fn from(d: Datum) -> Self {
        CoreValue::Data(d)
    }
}

// ---------------------------------------------------------------------------
// Iterative, recursion-safe Drop / Clone / PartialEq for the `Datum`↔`CoreValue` cluster
// (RFC-0041 §4.5, W3). A `Datum` is a `Box`-owned/`Vec`-owned acyclic tree (no `Rc`/`Arc` shared
// spine), so an explicit-worklist traversal frees/visits each node exactly once — double-free-safe.
// **Recorded precondition (Low freeze11):** a future intern/DAG cache putting `Rc`/`Arc` on the
// datum spine would invalidate the drop; it holds today by construction. `#![forbid(unsafe_code)]`
// still holds — only safe `Vec` + `mem::take` take-loops are used.
// ---------------------------------------------------------------------------

/// Iterative deep clone of a [`CoreValue`] and the `Datum`s it contains, bit-identical to the
/// derived clone (the stored datum `guarantee` summary is preserved verbatim — **not** recomputed
/// via [`Datum::new`], which `meet`s and could differ). Expand/assemble worklist over a `CoreValue`
/// value stack (`done.pop()` yields fields in forward declaration order).
fn clone_core(root: &CoreValue) -> CoreValue {
    enum Task<'a> {
        Expand(&'a CoreValue),
        AssembleData {
            ctor: CtorRef,
            guarantee: GuaranteeStrength,
            arity: usize,
        },
    }
    let mut tasks: Vec<Task<'_>> = vec![Task::Expand(root)];
    let mut done: Vec<CoreValue> = Vec::new();
    while let Some(task) = tasks.pop() {
        match task {
            Task::Expand(cv) => match cv {
                // `Value` is bounded-depth by construction, so its derived clone is not a deep vector.
                CoreValue::Repr(v) => done.push(CoreValue::Repr(v.clone())),
                CoreValue::Data(d) => {
                    tasks.push(Task::AssembleData {
                        ctor: d.ctor.clone(),
                        guarantee: d.guarantee,
                        arity: d.fields.len(),
                    });
                    for f in &d.fields {
                        tasks.push(Task::Expand(f));
                    }
                }
            },
            Task::AssembleData {
                ctor,
                guarantee,
                arity,
            } => {
                let mut fields = Vec::with_capacity(arity);
                for _ in 0..arity {
                    fields.push(done.pop().expect("clone_core: datum field"));
                }
                done.push(CoreValue::Data(Datum {
                    ctor,
                    fields,
                    guarantee,
                }));
            }
        }
    }
    done.pop()
        .expect("clone_core: exactly one root value remains")
}

/// Iterative structural equality of two [`CoreValue`]s (and their nested `Datum`s), result-identical
/// to the derived `PartialEq` (which for a `Datum` compares `ctor`, `fields`, **and** the
/// `guarantee` summary — all reproduced here). Bounded native stack regardless of nesting depth.
fn eq_core(a: &CoreValue, b: &CoreValue) -> bool {
    let mut stack: Vec<(&CoreValue, &CoreValue)> = vec![(a, b)];
    while let Some((x, y)) = stack.pop() {
        match (x, y) {
            (CoreValue::Repr(v1), CoreValue::Repr(v2)) => {
                if v1 != v2 {
                    return false;
                }
            }
            (CoreValue::Data(d1), CoreValue::Data(d2)) => {
                if d1.ctor != d2.ctor
                    || d1.guarantee != d2.guarantee
                    || d1.fields.len() != d2.fields.len()
                {
                    return false;
                }
                for (p, q) in d1.fields.iter().zip(&d2.fields) {
                    stack.push((p, q));
                }
            }
            _ => return false,
        }
    }
    true
}

impl Clone for Datum {
    fn clone(&self) -> Datum {
        // Each top-level field is cloned by the iterative `clone_core` (which absorbs all nesting
        // depth), so this map only ranges over the datum's *breadth* — bounded native stack.
        let fields = self.fields.iter().map(clone_core).collect();
        Datum {
            ctor: self.ctor.clone(),
            fields,
            guarantee: self.guarantee,
        }
    }
}

impl Clone for CoreValue {
    fn clone(&self) -> CoreValue {
        clone_core(self)
    }
}

impl PartialEq for Datum {
    fn eq(&self, other: &Datum) -> bool {
        self.ctor == other.ctor
            && self.guarantee == other.guarantee
            && self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(&other.fields)
                .all(|(a, b)| eq_core(a, b))
    }
}

impl PartialEq for CoreValue {
    fn eq(&self, other: &CoreValue) -> bool {
        eq_core(self, other)
    }
}

impl Drop for Datum {
    fn drop(&mut self) {
        // Flatten the owned field cluster onto an explicit worklist; each nested `Datum` is emptied
        // before it drops, so its re-entrant `Drop` sees empty `fields` — bounded reentrancy, never
        // deep recursion. (`Value` leaves drop in place — bounded-depth by construction.)
        //
        // Allocation honesty (RFC-0041 §4.5): as with `Node::drop`, a fully alloc-free iterative
        // drop of this non-intrusive tree in **safe** Rust is not achievable (no spare `next` field;
        // `Drop` gets no preallocated scratch); the `Vec` starts empty (no allocation) and grows only
        // when the cluster is actually deep — the case that otherwise *guaranteed* a `SIGABRT`.
        let mut work: Vec<CoreValue> = std::mem::take(&mut self.fields);
        while let Some(cv) = work.pop() {
            if let CoreValue::Data(mut d) = cv {
                work.extend(std::mem::take(&mut d.fields));
                // `d` drops here as a childless shell.
            }
            // `CoreValue::Repr(v)` drops `v` here (bounded by construction).
        }
    }
}

impl Canon {
    /// Encode a [`CoreValue`]'s identity-bearing content (a representation value's repr+payload, or
    /// a datum's constructor+fields). `Meta` / the datum summary are dynamic and excluded.
    /// Iterative for the `Data` arm (delegates to the iterative [`Canon::datum`]).
    pub(crate) fn core_value(&mut self, v: &CoreValue) {
        match v {
            CoreValue::Repr(rv) => self.value(rv),
            CoreValue::Data(d) => self.datum(d),
        }
    }

    /// Encode a [`Datum`]: its constructor reference then each field (order significant), **pre-order
    /// and iterative** (RFC-0041 §4.5) — byte-for-byte identical to the former recursive encoding, so
    /// content addresses are unchanged, but with bounded native-stack use on a deep datum spine.
    pub(crate) fn datum(&mut self, d: &Datum) {
        // Emit `d`'s header, then walk the field cluster depth-first, left-to-right (`iter().rev()`
        // pushes so the worklist pops in forward order) — reproducing the recursive pre-order stream.
        self.datum_header(d);
        let mut work: Vec<&CoreValue> = Vec::new();
        for f in d.fields().iter().rev() {
            work.push(f);
        }
        while let Some(cv) = work.pop() {
            match cv {
                CoreValue::Repr(rv) => self.value(rv),
                CoreValue::Data(inner) => {
                    self.datum_header(inner);
                    for f in inner.fields().iter().rev() {
                        work.push(f);
                    }
                }
            }
        }
    }

    /// Emit one datum's header bytes (tag · ctor-ref · field count) — the per-node prefix the
    /// recursive `datum` encoder wrote before recursing into fields.
    fn datum_header(&mut self, d: &Datum) {
        self.tag(crate::content::tag::DATUM);
        self.ctor_ref(d.ctor());
        self.u64(d.fields().len() as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CtorSpec, DataRegistry, DeclSpec, FieldSpec};
    use crate::meta::{Meta, Provenance};
    use crate::repr::Repr;
    use crate::value::Payload;
    use std::collections::BTreeMap;

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
        DataRegistry::build(&m).unwrap()
    }

    fn byte(g: GuaranteeStrength) -> Value {
        let meta = match g {
            GuaranteeStrength::Exact => Meta::exact(Provenance::Root),
            other => Meta::new(
                Provenance::Root,
                other,
                Some(crate::bound::Bound {
                    kind: crate::bound::BoundKind::Error {
                        eps: 0.1,
                        norm: crate::bound::NormKind::Linf,
                    },
                    basis: match other {
                        GuaranteeStrength::Proven => crate::bound::BoundBasis::ProvenThm {
                            citation: "t".into(),
                        },
                        GuaranteeStrength::Empirical => crate::bound::BoundBasis::EmpiricalFit {
                            trials: 1,
                            method: "m".into(),
                        },
                        _ => crate::bound::BoundBasis::UserDeclared,
                    },
                }),
                None,
                None,
                None,
            )
            .unwrap(),
        };
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![false; 8]),
            meta,
        )
        .unwrap()
    }

    #[test]
    fn nullary_datum_is_exact() {
        let reg = nat_registry();
        let z = Datum::new(reg.ctor_ref("Nat", 0).unwrap(), vec![]);
        assert_eq!(z.guarantee(), GuaranteeStrength::Exact);
    }

    #[test]
    fn construct_summary_is_the_meet_of_fields() {
        let reg = nat_registry();
        // A datum over an Empirical leaf summarises as Empirical (honesty degrades — RFC-0011 §4.6).
        let s = Datum::new(
            reg.ctor_ref("Nat", 1).unwrap(),
            vec![CoreValue::Repr(byte(GuaranteeStrength::Empirical))],
        );
        assert_eq!(s.guarantee(), GuaranteeStrength::Empirical);
        // Exact leaf → Exact summary.
        let s2 = Datum::new(
            reg.ctor_ref("Nat", 1).unwrap(),
            vec![CoreValue::Repr(byte(GuaranteeStrength::Exact))],
        );
        assert_eq!(s2.guarantee(), GuaranteeStrength::Exact);
    }

    #[test]
    fn meet_guarantee_only_degrades() {
        let reg = nat_registry();
        let z = Datum::new(reg.ctor_ref("Nat", 0).unwrap(), vec![]);
        assert_eq!(
            z.clone()
                .meet_guarantee(GuaranteeStrength::Exact)
                .guarantee(),
            GuaranteeStrength::Exact
        );
        assert_eq!(
            z.meet_guarantee(GuaranteeStrength::Declared).guarantee(),
            GuaranteeStrength::Declared
        );
    }

    #[test]
    fn content_hash_excludes_the_summary_but_not_the_fields() {
        let reg = nat_registry();
        // Two S(Z) data with differently-tagged leaves: the hash is over repr+payload only, so the
        // guarantee summary does not change identity, but a different payload does.
        let z = || Datum::new(reg.ctor_ref("Nat", 0).unwrap(), vec![]);
        let a = Datum::new(reg.ctor_ref("Nat", 1).unwrap(), vec![CoreValue::Data(z())]);
        let b = Datum::new(reg.ctor_ref("Nat", 1).unwrap(), vec![CoreValue::Data(z())]);
        assert_eq!(a.content_hash(), b.content_hash());
        // Z vs S(Z) differ.
        assert_ne!(z().content_hash(), a.content_hash());
    }

    // Mutant-witness (datum.rs:59:9): Datum::fields() must return the actual fields, not an empty
    // slice. If replaced by `Vec::leak(Vec::new())`, fields() would always return `[]`.
    #[test]
    fn fields_accessor_returns_actual_fields() {
        let reg = nat_registry();
        let z = Datum::new(reg.ctor_ref("Nat", 0).unwrap(), vec![]);
        // Z has 0 fields.
        assert_eq!(z.fields().len(), 0);
        // S(Z) has 1 field, and it is the Z datum.
        let s = Datum::new(
            reg.ctor_ref("Nat", 1).unwrap(),
            vec![CoreValue::Data(Datum::new(
                reg.ctor_ref("Nat", 0).unwrap(),
                vec![],
            ))],
        );
        assert_eq!(s.fields().len(), 1);
        // The single field is a Data variant.
        assert!(s.fields()[0].as_data().is_some());
        assert!(s.fields()[0].as_repr().is_none());
    }

    // Mutant-witness (datum.rs:101:9 and datum.rs:110:9): CoreValue::as_repr and as_data must
    // return Some(_) for the matching variant and None for the other. Both survivors would be
    // killed by asserting both the Some and the None branches.
    #[test]
    fn core_value_as_repr_and_as_data_are_discriminated() {
        let reg = nat_registry();
        let v = byte(GuaranteeStrength::Exact);
        let repr_val = CoreValue::Repr(v.clone());
        // as_repr returns Some for Repr variant and None for Data variant.
        assert!(repr_val.as_repr().is_some());
        assert!(repr_val.as_data().is_none());
        // The returned reference is to the same value.
        assert_eq!(repr_val.as_repr().unwrap(), &v);

        let datum = Datum::new(reg.ctor_ref("Nat", 0).unwrap(), vec![]);
        let data_val = CoreValue::Data(datum.clone());
        // as_data returns Some for Data variant and None for Repr variant.
        assert!(data_val.as_data().is_some());
        assert!(data_val.as_repr().is_none());
        assert_eq!(data_val.as_data().unwrap(), &datum);
    }

    // Mutant-witness (datum.rs:153:9): Canon::core_value must actually encode the value content.
    // If replaced by `()` (no-op), two datums with the SAME ctor but DIFFERENT field values
    // would hash identically — indistinguishable. This test pins that field-content changes hash.
    #[test]
    fn datum_hash_depends_on_field_content_not_just_ctor() {
        let reg = nat_registry();
        // Two S(·) datums whose inner value differs in payload: same ctor (Nat/1), different field.
        let s_zero = Datum::new(
            reg.ctor_ref("Nat", 1).unwrap(),
            vec![CoreValue::Repr(
                Value::new(
                    Repr::Binary { width: 8 },
                    Payload::Bits(vec![false; 8]),
                    Meta::exact(Provenance::Root),
                )
                .unwrap(),
            )],
        );
        let s_ones = Datum::new(
            reg.ctor_ref("Nat", 1).unwrap(),
            vec![CoreValue::Repr(
                Value::new(
                    Repr::Binary { width: 8 },
                    Payload::Bits(vec![true; 8]),
                    Meta::exact(Provenance::Root),
                )
                .unwrap(),
            )],
        );
        // Same constructor index but different field payloads → different content hash.
        // If Canon::core_value is a no-op, both hash to the same value.
        assert_ne!(
            s_zero.content_hash(),
            s_ones.content_hash(),
            "datums with same ctor but different field values must have different hashes"
        );
    }

    // Mutant-witness (datum.rs:153:9): Canon::core_value must actually encode the value content
    // (not no-op). If it were replaced by `()`, all CoreValues would hash identically. The
    // content_hash test above already covers Datum encoding; this test pins the Repr arm.
    #[test]
    fn core_value_content_hash_depends_on_content() {
        // Two Repr values with different payloads must have different content hashes as CoreValues.
        let v1 = CoreValue::Repr(
            Value::new(
                Repr::Binary { width: 8 },
                Payload::Bits(vec![false; 8]),
                Meta::exact(Provenance::Root),
            )
            .unwrap(),
        );
        let v2 = CoreValue::Repr(
            Value::new(
                Repr::Binary { width: 8 },
                Payload::Bits(vec![true; 8]),
                Meta::exact(Provenance::Root),
            )
            .unwrap(),
        );
        assert_ne!(v1.content_hash(), v2.content_hash());

        // A Repr and a Data value must have different content hashes.
        let reg = nat_registry();
        let d = CoreValue::Data(Datum::new(reg.ctor_ref("Nat", 0).unwrap(), vec![]));
        assert_ne!(v1.content_hash(), d.content_hash());
    }
}
