//! Encoding utility tests — `encode_seq` and `encode_set`.
//!
//! Tests:
//! - C1: explicit errors on empty input.
//! - Guarantee: `encode_seq` distinguishes positions; `encode_set` does not.
//! - Property: `encode_seq` outcome is non-trivially similar to its positionally-correct member
//!   (randomized over 20 seeds to exercise the LCG space).
//! - The tag for compositions is the `bundle` tag (weakest-wins; RFC-0001 §4.7).

use mycelium_std_vsa::{encode_seq, encode_set, similarity, VsaError};
use mycelium_vsa::MapI;

const DIM: u32 = 512;

fn bipolar(dim: u32, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 63) & 1 == 1 {
                1.0_f64
            } else {
                -1.0_f64
            }
        })
        .collect()
}

// --- C1: empty slice is EmptyBundle ---

#[test]
fn encode_seq_empty_is_explicit() {
    let m = MapI::new(DIM);
    assert!(
        matches!(encode_seq(&m, &[]), Err(VsaError::EmptyBundle)),
        "encode_seq over zero items must fail with EmptyBundle"
    );
}

#[test]
fn encode_set_empty_is_explicit() {
    let m = MapI::new(DIM);
    assert!(
        matches!(encode_set(&m, &[]), Err(VsaError::EmptyBundle)),
        "encode_set over zero items must fail with EmptyBundle"
    );
}

// --- Positional distinction: encode_seq vs encode_set ---

/// `encode_seq` is sensitive to order: swapping two items yields a different superposition.
/// `encode_set` is order-insensitive: swapping yields the same result.
///
/// Property checked over 10 seed pairs; each is a one-deterministic-sample check.
#[test]
fn encode_seq_is_order_sensitive() {
    let m = MapI::new(DIM);
    for seed in 0u64..10 {
        let a = bipolar(DIM, seed * 3 + 1);
        let b = bipolar(DIM, seed * 3 + 2);
        let seq_ab = encode_seq(&m, &[&a, &b]).unwrap();
        let seq_ba = encode_seq(&m, &[&b, &a]).unwrap();
        assert_ne!(
            seq_ab, seq_ba,
            "seed {seed}: encode_seq([a,b]) must differ from encode_seq([b,a])"
        );
        let set_ab = encode_set(&m, &[&a, &b]).unwrap();
        let set_ba = encode_set(&m, &[&b, &a]).unwrap();
        // `bundle` is commutative (elementwise sum), so encode_set is order-insensitive.
        assert_eq!(
            set_ab, set_ba,
            "seed {seed}: encode_set([a,b]) should equal encode_set([b,a]) (bundle is commutative)"
        );
    }
}

// --- Property: encode_seq superposition is similar to the member at its position ---
//
// If `s = encode_seq([a, b, c])` then unbinding by `permute(e_i, i)` (where `e_i` is the
// identity role for position `i`) and cleaning up should roughly resemble item `i`.  We verify
// the weaker claim: `similarity(s, permute(a, 0)) > similarity(s, permute(stranger, 0))`.
// This is one deterministic sample per seed (not probabilistic).

#[test]
fn encode_seq_member_similarity_property() {
    let m = MapI::new(DIM);
    for seed in 0u64..20 {
        let items: Vec<Vec<f64>> = (0..3).map(|i| bipolar(DIM, seed * 4 + i)).collect();
        let stranger = bipolar(DIM, seed * 4 + 100);
        let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
        let seq = encode_seq(&m, &refs).unwrap();
        // permute(item[0], 0) == item[0] itself; permute(stranger, 0) == stranger.
        let member_sim = similarity(&m, &seq, &items[0]);
        let stranger_sim = similarity(&m, &seq, &stranger);
        // We only assert a relative claim: the member is closer than the stranger (with 3 items
        // the individual similarity is moderate but consistently above a fresh random vector).
        assert!(
            member_sim > stranger_sim,
            "seed {seed}: encode_seq should be more similar to item[0] ({member_sim:.3}) \
             than to a stranger ({stranger_sim:.3})"
        );
    }
}

// --- encode_set: a bundled singleton is similar to its only member ---

#[test]
fn encode_set_singleton_is_the_item_itself() {
    let m = MapI::new(DIM);
    let a = bipolar(DIM, 77);
    let out = encode_set(&m, &[&a]).unwrap();
    // bundle([a]) == a (single-item superposition is the item).
    assert_eq!(
        out, a,
        "encode_set over a singleton should equal the item itself"
    );
}
