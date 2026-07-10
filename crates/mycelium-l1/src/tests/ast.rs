use crate::ast::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[test]
fn typeref_unguaranteed_matches_field_form() {
    let base = BaseType::Binary(WidthRef::Lit(8));
    assert_eq!(
        TypeRef::unguaranteed(base.clone()),
        TypeRef {
            base,
            guarantee: None
        }
    );
}

#[test]
fn typeref_with_guarantee_matches_field_form() {
    let base = BaseType::Ternary(WidthRef::Lit(3));
    assert_eq!(
        TypeRef::with_guarantee(base.clone(), Strength::Exact),
        TypeRef {
            base,
            guarantee: Some(Strength::Exact)
        }
    );
}

#[test]
fn literal_ctors_match_variants() {
    assert_eq!(Literal::binary("1010"), Literal::Bin("1010".to_owned()));
    assert_eq!(Literal::ternary("+0-"), Literal::Trit("+0-".to_owned()));
    // `impl Into<String>` accepts both `&str` and `String`.
    assert_eq!(
        Literal::binary(String::from("11")),
        Literal::Bin("11".to_owned())
    );
}

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

#[test]
fn scalar_and_strength_hash_is_consistent_with_eq() {
    // The new `Hash` derives must agree with `Eq` (equal values hash equal); enough to confirm
    // the derive is wired and usable as a map/set key.
    assert_eq!(hash_of(&Scalar::F32), hash_of(&Scalar::F32));
    assert_eq!(hash_of(&Strength::Proven), hash_of(&Strength::Proven));
    use std::collections::HashSet;
    let scalars: HashSet<Scalar> = [Scalar::F16, Scalar::Bf16, Scalar::F32, Scalar::F64]
        .into_iter()
        .collect();
    assert_eq!(scalars.len(), 4);
    let strengths: HashSet<Strength> = [
        Strength::Exact,
        Strength::Proven,
        Strength::Empirical,
        Strength::Declared,
    ]
    .into_iter()
    .collect();
    assert_eq!(strengths.len(), 4);
}

#[test]
fn strength_lattice_order_is_the_trust_chain() {
    // The chain `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (RFC-0018 §4.1) — strictly decreasing rank.
    assert!(
        Strength::Exact.rank() > Strength::Proven.rank()
            && Strength::Proven.rank() > Strength::Empirical.rank()
            && Strength::Empirical.rank() > Strength::Declared.rank()
    );
}

#[test]
fn strength_meet_is_the_weaker_grade() {
    // `g₁ ∧ g₂` is the *less trusted* of the two (RFC-0018 §4.1): the pessimistic composition rule.
    assert_eq!(Strength::Exact.meet(Strength::Proven), Strength::Proven);
    assert_eq!(
        Strength::Proven.meet(Strength::Empirical),
        Strength::Empirical
    );
    assert_eq!(
        Strength::Empirical.meet(Strength::Declared),
        Strength::Declared
    );
    // Idempotent, commutative, and `Exact` is the identity (top) of the meet-semilattice.
    for &g in &[
        Strength::Exact,
        Strength::Proven,
        Strength::Empirical,
        Strength::Declared,
    ] {
        assert_eq!(g.meet(g), g, "meet is idempotent");
        assert_eq!(g.meet(Strength::Exact), g, "Exact is the meet identity");
        for &h in &[
            Strength::Exact,
            Strength::Proven,
            Strength::Empirical,
            Strength::Declared,
        ] {
            assert_eq!(g.meet(h), h.meet(g), "meet is commutative");
        }
    }
}

#[test]
fn strength_satisfies_is_at_least_as_trusted() {
    // `self ⊒ demand` (RFC-0018 §4.3): a value satisfies a demand iff it is at least as trusted.
    assert!(Strength::Exact.satisfies(Strength::Exact));
    assert!(Strength::Exact.satisfies(Strength::Declared));
    assert!(Strength::Proven.satisfies(Strength::Empirical));
    // The honesty failure: a weaker value does NOT satisfy a stronger demand (VR-5).
    assert!(!Strength::Empirical.satisfies(Strength::Exact));
    assert!(!Strength::Declared.satisfies(Strength::Proven));
}

#[test]
fn fn_sig_param_names_drops_bounds() {
    // `param_names()` projects the bounded type-params (RFC-0019 §4.1) to their names — the form
    // the §11 generic machinery / checker `tyvars` consume. Bounds are read separately.
    let sig = FnSig {
        name: "f".to_owned(),
        params: vec![
            TypeParam {
                name: "T".to_owned(),
                kind: ParamKind::Type,
                bounds: vec![TraitRef {
                    name: "Cmp".to_owned(),
                    args: vec![],
                }],
            },
            TypeParam {
                name: "U".to_owned(),
                kind: ParamKind::Type,
                bounds: vec![],
            },
        ],
        value_params: vec![],
        ret: TypeRef::unguaranteed(BaseType::Binary(WidthRef::Lit(1))),
        effects: vec![],
        effect_budgets: std::collections::BTreeMap::new(),
    };
    assert_eq!(sig.param_names(), vec!["T".to_owned(), "U".to_owned()]);
}
