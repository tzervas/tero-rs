//! The tero-declared **empirical profile** for Layer-2 record encoding (M-1018) — the validated
//! regime a record bundle must stay inside, and the dimensionality/term-cap constants the encoder
//! and decoder share.
//!
//! Honesty (VR-5): [`L2_PROFILE`] is tagged **`Declared`** here — its `trials` count is `0` and its
//! `method` says so — because the Layer-2 near-orthogonality/retrieval properties have **not** been
//! trial-validated at this dimension over this corpus yet. The eval harness (`crate::eval`) is what
//! *measures* Layer-2's behaviour; only a recorded trial run would license upgrading the tag to
//! `Empirical`. Until then the profile's job is purely its **never-silent side-condition check**
//! ([`mycelium_vsa::EmpiricalProfile::check`]) at encode time: a record whose bundle exceeds the
//! validated `max_items`, or a dimension below `min_dim`, is **refused explicitly** rather than
//! silently over-capacity-bundled (G2).

use mycelium_vsa::EmpiricalProfile;

/// The Layer-2 hypervector dimensionality. Fixed at 4096 (`16³`, the `MAPI_RESONATOR_PROFILE`
/// operational edge) — high enough that `required_dim(m, δ)` is satisfied for every capped record
/// (so a per-record `Proven` capacity bound is *available*), while staying a single, committed value.
pub const L2_DIM: u32 = 4096;

/// The target per-record bundle failure probability the capacity side-condition is checked against
/// (`δ`). `1e-2` matches the smallest `(m, δ)` setting the M-001 capacity probe checks; at `L2_DIM`
/// it is comfortably satisfied for every capped record (`required_dim(64, 1e-2) ≈ 1753 ≤ 4096`).
pub const L2_DELTA: f64 = 1e-2;

/// The per-field term cap (top-K, first-occurrence order) applied to a record's title/summary term
/// lists and to the `depends_on`/`doc_refs` edge lists. A never-silent cap: the truncation is
/// recorded in the encode result and the [`crate::vsa2::Layer2Explain`], mirroring `query.rs`'s
/// `TEXT_RESULT_LIMIT` posture (a truncated set is reported, never silently exhaustive).
pub const L2_TERM_CAP: usize = 8;

/// The tero-declared empirical profile for a Layer-2 record bundle. `max_items` bounds the number of
/// role-filler binds a single record may superpose (with every list capped at [`L2_TERM_CAP`], the
/// worst-case term count is `4 singletons + 2·8 text + 2·8 edge = 36`, so `64` leaves headroom);
/// `min_dim` pins the encode dimension; `delta` is the checked capacity target. **`Declared`**:
/// `trials = 0`, and `method` states no trial validation has run yet — the harness measures Layer 2,
/// it does not (here) validate this profile into `Empirical`.
pub const L2_PROFILE: EmpiricalProfile = EmpiricalProfile {
    max_items: 64,
    odd_items_only: false,
    min_dim: L2_DIM,
    delta: L2_DELTA,
    trials: 0,
    method:
        "Declared — Layer-2 record-bundle regime; no trial validation discharged yet (M-1018); \
             the eval harness measures retrieval, it does not upgrade this profile to Empirical",
};
