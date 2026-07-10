//! White-box tests for Layer-2 **encoding** (M-1018): determinism (pure function of corpus+seed),
//! the never-silent empirical-profile guard, the fixed record dimension, and the recorded top-K cap.

use mycelium_vsa::MapI;

use crate::vsa2::atoms::{atom, TERO_L2_SEED};
use crate::vsa2::encode::{build_codebook, encode_record, tokenize};
use crate::vsa2::profile::{L2_DIM, L2_PROFILE, L2_TERM_CAP};
use crate::{Family, TeroIndexItem};

use super::fixture::corpus_report;

/// A crafted item with a controllable title/summary (for the cap + determinism tests).
fn item_with(title: &str, summary: Option<&str>) -> TeroIndexItem {
    let mut it = TeroIndexItem::new("test-anchor", Family::Doc, "section", title, "docs/x.md", 1);
    it.id = Some("M-0042".to_owned());
    it.summary = summary.map(str::to_owned);
    it
}

#[test]
fn atoms_are_deterministic_for_a_symbol_and_seed() {
    // Encode is a pure function of (symbol, seed): same symbol ⇒ byte-identical atom; different ⇒ not.
    assert_eq!(atom("role-id", 256), atom("role-id", 256));
    assert_ne!(atom("role-id", 256), atom("role-kind", 256));
    // The committed master seed is the one the codebook is built from.
    assert_eq!(TERO_L2_SEED, 0x7E70_1018_5EED_C0DE);
}

#[test]
fn encode_is_deterministic_two_encodes_are_byte_identical() {
    let model = MapI::new(L2_DIM);
    let it = item_with(
        "A Deterministic Encoding Title",
        Some("with a stable summary line"),
    );
    let (v1, s1) = encode_record(&it, &model).expect("in-regime");
    let (v2, s2) = encode_record(&it, &model).expect("in-regime");
    assert_eq!(
        v1, v2,
        "two encodes of the same item must be byte-identical"
    );
    assert_eq!(s1, s2);
}

#[test]
fn record_dimension_is_4096_and_bipolar() {
    let model = MapI::new(L2_DIM);
    let it = item_with("Some Title", Some("a summary"));
    let (v, _) = encode_record(&it, &model).expect("in-regime");
    assert_eq!(v.len(), L2_DIM as usize, "record must be exactly L2_DIM");
    assert!(
        v.iter().all(|&x| x == 1.0 || x == -1.0),
        "signed record must be strictly bipolar (±1)"
    );
}

#[test]
fn empirical_profile_guard_fires_out_of_regime_and_passes_in_regime() {
    // The never-silent capacity guard: too many bundled terms, or too small a dimension, is an
    // explicit refusal — never a silent over-capacity bundle (G2). (The per-field top-K cap keeps
    // real records in-regime, so this guard is the defensive floor beneath that cap.)
    assert!(
        L2_PROFILE.check(65, L2_DIM).is_err(),
        "over max_items refuses"
    );
    assert!(L2_PROFILE.check(10, 1024).is_err(), "below min_dim refuses");
    assert!(L2_PROFILE.check(10, L2_DIM).is_ok(), "in-regime passes");
    // The profile is Declared until trials — its trial count is 0 (honest, no validation yet).
    assert_eq!(L2_PROFILE.trials, 0);
}

#[test]
fn top_k_term_cap_is_recorded_never_silent() {
    let model = MapI::new(L2_DIM);
    // A title with more than L2_TERM_CAP distinct tokens ⇒ truncation is recorded.
    let many: Vec<String> = (0..L2_TERM_CAP + 4).map(|i| format!("term{i}")).collect();
    let long_title = many.join(" ");
    assert!(tokenize(&long_title).len() > L2_TERM_CAP);
    let (_, stats_long) = encode_record(&item_with(&long_title, None), &model).expect("in-regime");
    assert!(stats_long.truncated, "a long title must record truncation");

    // A short title ⇒ no truncation.
    let (_, stats_short) = encode_record(&item_with("short title", None), &model).expect("ok");
    assert!(!stats_short.truncated);
}

#[test]
fn build_codebook_encodes_every_row_none_silently_dropped() {
    let (_root, report) = corpus_report("l2-encode");
    let model = MapI::new(L2_DIM);
    let out = build_codebook(&report, &model);
    // Every fixture row is in-regime, so all encode and none are refused; a refusal (if any) is
    // recorded, never a silent drop.
    assert_eq!(
        out.memory.len() + out.refused.len(),
        report.items.len(),
        "encoded + refused must account for every row (never-silent)"
    );
    assert!(out.refused.is_empty(), "fixture rows are all in-regime");
    assert!(
        out.max_terms <= L2_PROFILE.max_items,
        "within the declared regime"
    );
}
