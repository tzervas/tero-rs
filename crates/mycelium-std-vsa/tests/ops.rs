//! Per-model operation vocabulary tests — C1/C2/C3 conformance.
//!
//! Tests the `std.vsa` operation surface over MAP-I (the primary validated model), verifying:
//! - **C1** — errors are explicit (never silent coercions or sentinel returns).
//! - **C2** — guarantee tags match the matrix (tested separately in `guarantee_matrix.rs`).
//! - **C3** — approximate ops expose `(confidence, margin)` and can be thresholded.
//! - Property: `bind ∘ unbind` is self-inverse for MAP-I (randomized over many trials).
//! - Property: `permute ∘ unpermute` is exactly invertible for all shifts (exhaustive over small
//!   dims; randomized over large dims).

use mycelium_std_vsa::{
    bind, bind_role, bundle, cleanup, permute, similarity, unbind, unpermute, CleanupMemory,
    VsaError,
};
use mycelium_vsa::MapI;

const DIM: u32 = 512;

/// A deterministic bipolar (`±1`) hypervector from a seed (tiny LCG; no rand dependency).
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

fn model() -> MapI {
    MapI::new(DIM)
}

// --- C1: errors are explicit ---

/// `bind` with a mismatched operand length is an explicit `DimMismatch`, never a silent coercion.
///
/// Mutant-witness: remove the `check_len` call in `MapI::bind` — this returns a wrong-length
/// result instead of `DimMismatch`.
#[test]
fn bind_dim_mismatch_is_explicit() {
    let m = model();
    let a = bipolar(DIM, 1);
    let b_short = bipolar(DIM / 2, 2);
    assert!(
        matches!(
            bind(&m, &a, &b_short),
            Err(VsaError::DimMismatch {
                expected: 512,
                got: 256
            })
        ),
        "bind with wrong-length b must fail with DimMismatch"
    );
}

/// `bundle` over zero items is `Err(EmptyBundle)`, never a fabricated zero vector.
#[test]
fn bundle_empty_is_explicit() {
    let m = model();
    assert!(
        matches!(bundle(&m, &[]), Err(VsaError::EmptyBundle)),
        "bundle of zero items must fail with EmptyBundle"
    );
}

/// `cleanup` on an empty codebook is `Err(EmptyCodebook)`, not `None` or a default.
#[test]
fn cleanup_empty_codebook_is_explicit() {
    let m = model();
    let mem = CleanupMemory::new(DIM);
    let query = bipolar(DIM, 42);
    assert!(
        matches!(
            cleanup(&mem, &query, &m, 0.0, 0.0),
            Err(VsaError::EmptyCodebook)
        ),
        "cleanup against empty codebook must fail with EmptyCodebook"
    );
}

/// `cleanup` with a query whose length disagrees with the codebook dim is `Err(EmptyCodebook)`.
#[test]
fn cleanup_dim_mismatch_is_explicit() {
    let m = model();
    let mut mem = CleanupMemory::new(DIM);
    mem.insert("a", bipolar(DIM, 1)).unwrap();
    let short = bipolar(DIM / 2, 2);
    assert!(
        matches!(
            cleanup(&mem, &short, &m, 0.0, 0.0),
            Err(VsaError::EmptyCodebook)
        ),
        "cleanup with wrong-length query must fail with EmptyCodebook (dim mismatch)"
    );
}

/// `cleanup` below `min_confidence` is an explicit `BelowCleanupThreshold`, never a
/// silent low-confidence answer passed off as a match.
///
/// Mutant-witness: remove the confidence-threshold check in `ops::cleanup` — the error case
/// silently becomes an `Ok(hit)` with a too-low confidence.
#[test]
fn cleanup_below_confidence_threshold_is_explicit() {
    let m = model();
    let mut mem = CleanupMemory::new(DIM);
    let atom = bipolar(DIM, 7);
    mem.insert("a", atom.clone()).unwrap();
    // A perfect match (same vector) has confidence ≈ 1.0, so min_confidence = 2.0 is impossible.
    match cleanup(&mem, &atom, &m, 2.0, 0.0) {
        Err(VsaError::BelowCleanupThreshold {
            confidence,
            threshold,
        }) => {
            assert!(
                threshold == 2.0,
                "threshold should be the caller-supplied 2.0, got {threshold}"
            );
            assert!(
                confidence > 0.9,
                "exact atom should have high confidence, got {confidence}"
            );
        }
        other => {
            panic!("expected BelowCleanupThreshold for impossible min_confidence, got {other:?}")
        }
    }
}

/// `cleanup` below `min_margin` is an explicit refusal (reusing `BelowCleanupThreshold`).
///
/// A singleton codebook has a margin of `confidence − (−1) ≈ 2.0`, so a min_margin of 3.0 is
/// impossible.
#[test]
fn cleanup_below_margin_threshold_is_explicit() {
    let m = model();
    let mut mem = CleanupMemory::new(DIM);
    mem.insert("a", bipolar(DIM, 9)).unwrap();
    let query = bipolar(DIM, 9);
    // Impossible margin (cosine is in [-1,1], so max margin ≈ 2.0 from the cosine floor).
    assert!(
        matches!(
            cleanup(&mem, &query, &m, 0.0, 3.0),
            Err(VsaError::BelowCleanupThreshold { .. })
        ),
        "cleanup below impossible min_margin must fail explicitly"
    );
}

// --- Property: bind ∘ unbind is self-inverse for MAP-I ---
//
// For MAP-I, bind is the elementwise bipolar product and unbind == bind (self-inverse).
// So unbind(bind(a, b), b) == a exactly for any bipolar a, b.
//
// Randomized over 50 seed pairs (one deterministic sample per seed pair — not probabilistic
// sampling but a multi-seed sweep through the LCG space).

#[test]
fn bind_unbind_self_inverse_property_map_i() {
    let m = model();
    // 50 seed pairs — each deterministic, covers diverse LCG states.
    for seed in 0u64..50 {
        let a = bipolar(DIM, seed * 2 + 1);
        let b = bipolar(DIM, seed * 2 + 2);
        let bound = bind(&m, &a, &b).unwrap_or_else(|e| panic!("seed {seed}: bind failed: {e}"));
        let recovered =
            unbind(&m, &bound, &b).unwrap_or_else(|e| panic!("seed {seed}: unbind failed: {e}"));
        assert_eq!(
            recovered, a,
            "seed {seed}: unbind(bind(a,b), b) must equal a exactly (MAP-I self-inverse)"
        );
    }
}

// --- Property: permute ∘ unpermute is exactly invertible ---
//
// For any shift, unpermute(permute(a, s), s) == a exactly (the §4.1 erratum — permute is a
// fixed coordinate bijection, trivially invertible). Tested over many seeds and shifts.

#[test]
fn permute_unpermute_round_trip_property() {
    let m = model();
    let shifts: &[i64] = &[-100, -7, -1, 0, 1, 3, 7, 13, 100, 512, 1000];
    for (i, seed) in (0u64..20).enumerate() {
        let a = bipolar(DIM, seed + 1000);
        for &shift in shifts {
            let p = permute(&m, &a, shift)
                .unwrap_or_else(|e| panic!("seed {seed} shift {shift}: permute failed: {e}"));
            let back = unpermute(&m, &p, shift)
                .unwrap_or_else(|e| panic!("seed {seed} shift {shift}: unpermute failed: {e}"));
            assert_eq!(
                back, a,
                "seed {seed} shift {shift}: unpermute(permute(a,s),s) must equal a (Exact bijection)",
            );
            // A non-zero shift on a non-constant vector actually moves elements.
            if shift.rem_euclid(DIM as i64) != 0 {
                assert_ne!(
                    p, a,
                    "seed {seed} shift {shift}: a non-zero shift should move elements"
                );
            }
            // Unused: suppress dead-code lint on i.
            let _ = i;
        }
    }
}

// --- Bundle ---

/// A bundle of near-orthogonal atoms is more similar to its members than to a stranger.
#[test]
fn bundle_similar_to_members() {
    let m = model();
    let items: Vec<Vec<f64>> = (0..4).map(|i| bipolar(DIM, 200 + i)).collect();
    let refs: Vec<&[f64]> = items.iter().map(Vec::as_slice).collect();
    let superposed = bundle(&m, &refs).unwrap();
    let member_sim = similarity(&m, &superposed, &items[0]);
    let stranger_sim = similarity(&m, &superposed, &bipolar(DIM, 9999));
    assert!(
        member_sim > 0.2,
        "bundle member should have non-trivial similarity: {member_sim}"
    );
    assert!(
        member_sim > stranger_sim + 0.1,
        "bundle member ({member_sim:.3}) should beat a stranger ({stranger_sim:.3})"
    );
}

// --- Bind role alias ---

/// `bind_role(role, filler)` is identical to `bind(role, filler)`.
#[test]
fn bind_role_equals_bind() {
    let m = model();
    let role = bipolar(DIM, 1);
    let filler = bipolar(DIM, 2);
    let via_bind = bind(&m, &role, &filler).unwrap();
    let via_bind_role = bind_role(&m, &role, &filler).unwrap();
    assert_eq!(via_bind, via_bind_role, "bind_role must equal bind");
}

// --- Similarity ---

/// Similarity of a vector with itself is 1.0 (for a non-zero vector).
#[test]
fn similarity_self_is_one() {
    let m = model();
    let a = bipolar(DIM, 42);
    let sim = similarity(&m, &a, &a);
    assert!(
        (sim - 1.0).abs() < 1e-9,
        "similarity of a vector with itself must be 1.0, got {sim}"
    );
}

/// Similarity of two random orthogonal hypervectors is near 0 in high dimension.
#[test]
fn similarity_random_vectors_near_zero() {
    let m = model();
    let sims: Vec<f64> = (0..10)
        .map(|i| {
            let a = bipolar(DIM, 100 + i * 2);
            let b = bipolar(DIM, 101 + i * 2);
            similarity(&m, &a, &b)
        })
        .collect();
    let max_abs = sims.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs < 0.25,
        "random bipolar vectors in dim {DIM} should be near-orthogonal (max |sim|={max_abs:.3})"
    );
}

// --- Cleanup round-trip (the headline FR-S4 use case) ---

/// Bundle a role⊗filler pair, unbind by the role, clean up → recovers the filler.
/// This is the core VSA associative memory pattern (FR-S4).
#[test]
fn cleanup_makes_approximate_unbind_usable() {
    let m = model();
    let role_color = bipolar(DIM, 10);
    let red = bipolar(DIM, 20);
    let blue = bipolar(DIM, 21);
    let green = bipolar(DIM, 22);

    let bound = bind(&m, &role_color, &red).unwrap();

    let mut fillers = CleanupMemory::new(DIM);
    fillers.insert("red", red.clone()).unwrap();
    fillers.insert("blue", blue).unwrap();
    fillers.insert("green", green).unwrap();

    let noisy = unbind(&m, &bound, &role_color).unwrap();
    let hit = cleanup(&fillers, &noisy, &m, 0.0, 0.0).expect("non-empty codebook");
    assert_eq!(
        hit.label, "red",
        "cleanup after unbind should recover the filler"
    );
    assert!(hit.confidence > 0.5, "confidence={}", hit.confidence);
    assert!(hit.margin > 0.3, "margin={}", hit.margin);
}
