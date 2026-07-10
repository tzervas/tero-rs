//! Deterministic seeded ±1 hypervector **atoms** for the Layer-2 semantic encoding (M-1018).
//!
//! Encoding is a **pure function of `(corpus, seed)`**: every atom is fully determined by its
//! symbol string and the one committed master seed [`TERO_L2_SEED`], via a tiny in-crate seeded LCG.
//! There is no `rand` dependency and no hidden state — so two encodes of the same corpus produce
//! byte-identical hypervectors (the determinism contract the M-1015 Layer-1 index already holds, now
//! extended to Layer 2). The LCG is intentionally the same shape as the `mycelium-vsa` cleanup/mapi
//! test atoms, but kept **in-crate**: `mycelium_vsa::Lcg` is `pub(crate)` there and not reusable.
//!
//! Honesty (VR-5): these atoms are a **`Declared`** construction — a seeded pseudo-random ±1 draw,
//! not a proven near-orthogonal family. The near-orthogonality that makes VSA retrieval work is an
//! `Empirical` property measured by the eval harness, not asserted here.

/// The one committed master seed for Layer-2 atom generation. It is a **versioned constant, not a
/// tunable**: changing it changes every atom and therefore the whole Layer-2 encoding (a new codebook
/// identity). Recorded in every [`crate::vsa2::Layer2Explain`] so a retrieval is reproducible.
pub const TERO_L2_SEED: u64 = 0x7E70_1018_5EED_C0DE;

// ── role symbols ─────────────────────────────────────────────────────────────────────────────────
// The fixed keys under which fillers are bound into a record hypervector. Distinct, stable strings ⇒
// distinct, stable role atoms. A record is `bundle( ROLE ⊗ atom(filler) … )` over these roles.

/// Role atom for a row's source id (`M-1015`, `RFC-0034`, …).
pub const ROLE_ID: &str = "__tero_l2_role_id__";
/// Role atom for a row's family-specific kind (`rfc`/`issue`/`section`/…).
pub const ROLE_KIND: &str = "__tero_l2_role_kind__";
/// Role atom for a row's corpus family (`doc`/`issue`/`changelog`/…).
pub const ROLE_FAMILY: &str = "__tero_l2_role_family__";
/// Role atom for a row's declared status (`Accepted`/`done`/…).
pub const ROLE_STATUS: &str = "__tero_l2_role_status__";
/// Role atom bound over each (capped) title term.
pub const ROLE_TITLE_TERM: &str = "__tero_l2_role_title_term__";
/// Role atom bound over each (capped) summary term.
pub const ROLE_SUMMARY_TERM: &str = "__tero_l2_role_summary_term__";
/// Role atom bound over each `depends_on` id (issues only).
pub const ROLE_DEPENDS_ON: &str = "__tero_l2_role_depends_on__";
/// Role atom bound over each `doc_refs` citation (issues only).
pub const ROLE_DOC_REF: &str = "__tero_l2_role_doc_ref__";
/// Role atom for a row's parent epic (issues only).
pub const ROLE_EPIC: &str = "__tero_l2_role_epic__";
/// Role atom for the cited claim's guarantee tag where the source declares one.
pub const ROLE_GUARANTEE: &str = "__tero_l2_role_guarantee__";

/// FNV-1a 64-bit hash of a symbol — mixes the symbol deterministically into the master seed so each
/// distinct symbol seeds a distinct atom. (FNV-1a is a stable, well-known non-cryptographic hash;
/// its role here is only decorrelation of seeds, not security.)
#[must_use]
pub fn fnv1a(s: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// A deterministic bipolar (`±1`) hypervector of length `dim` for `symbol`, seeded by
/// `TERO_L2_SEED ^ fnv1a(symbol)`. Pure function of `(symbol, dim)` given the fixed master seed.
///
/// The tiny LCG (`x' = a·x + c`, the Numerical-Recipes constants) is drawn one step per component and
/// the sign of the high bit becomes `±1` — the same no-`rand` construction the `mycelium-vsa` tests
/// use, kept in-crate because that crate's `Lcg` is `pub(crate)`.
#[must_use]
pub fn atom(symbol: &str, dim: u32) -> Vec<f64> {
    // Fold the symbol into the seed, then perturb once so a zero fold cannot yield a degenerate state.
    let mut s = (TERO_L2_SEED ^ fnv1a(symbol))
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if (s >> 63) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
