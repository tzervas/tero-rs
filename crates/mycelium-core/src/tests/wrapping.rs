//! White-box tests for [`crate::wrapping`] and the Axis-B `wrapping` opt-out on [`crate::meta::Meta`]
//! (RFC-0034 §10; M-791).
//!
//! **Property:** never-silent failability (Axis B) is DEFAULT-ON — `Meta::wrapping_opt()` returns
//! `None` from every constructor, in every `CertMode`. Absence means "Axis-B failability is active".
//!
//! **Property:** the `wrapping` opt-out is explicit + visible at the use site:
//! - `Meta::wrapping_opt()` returns `Some(WrappingOpt)` *only* after an explicit `with_wrapping()`
//!   call — never ambient, never implicit.
//! - Attaching `WrappingOpt` does **not** alter the guarantee strength (Axis-A) or `cert_mode`
//!   (VR-5: opt-out of one axis never upgrades or silences another).
//!
//! RFC-0034 §13 / conformance: mode-parametric (sweeps `CertMode::ALL`) for the "Axis-B holds in
//! every mode" property; explicit-annotation properties cover the orthogonality invariant.

use crate::cert_mode::CertMode;
use crate::guarantee::GuaranteeStrength;
use crate::meta::{Meta, Provenance};
use crate::wrapping::WrappingOpt;

// ---------------------------------------------------------------------------
// Axis-B default-on: absent from every default-constructed Meta, all CertModes
// ---------------------------------------------------------------------------

/// **Property (Declared, Empirical):** Axis-B failability is default-on. In every `CertMode`,
/// a default-constructed `Meta` carries no `wrapping_opt` — absence is the safe default.
/// Sweeps `CertMode::ALL` for mode-parametric coverage (RFC-0034 §13 conformance).
#[test]
fn axis_b_is_default_on_in_every_cert_mode_exact_meta() {
    // Exact Meta (the simplest constructor path): wrapping_opt absent in every mode.
    for &mode in &CertMode::ALL {
        let m = Meta::exact(Provenance::Root).with_cert_mode(mode);
        assert!(
            m.wrapping_opt().is_none(),
            "Axis-B is default-on: wrapping_opt must be None in mode {mode:?} \
             (RFC-0034 §10 — never-silent failability is the default, not wrapping)"
        );
    }
}

/// The `Meta::new` constructor path also defaults to Axis-B active (no wrapping_opt).
/// Sweeps `CertMode::ALL` for mode-parametric coverage.
#[test]
fn axis_b_is_default_on_in_every_cert_mode_meta_new() {
    for &mode in &CertMode::ALL {
        let m = Meta::new(
            Provenance::Root,
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .expect("valid Exact Meta")
        .with_cert_mode(mode);
        assert!(
            m.wrapping_opt().is_none(),
            "Meta::new must default wrapping_opt to None in mode {mode:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Explicit opt-in: with_wrapping sets the marker; absence ≠ wrapping
// ---------------------------------------------------------------------------

/// `with_wrapping` sets the marker; `wrapping_opt()` returns `Some`.
/// Without calling `with_wrapping`, `wrapping_opt()` stays `None`.
/// This is the use-site visibility / grep-auditability check (RFC-0034 §10).
#[test]
fn wrapping_opt_is_set_only_by_explicit_with_wrapping() {
    let base = Meta::exact(Provenance::Root);
    // Default: absent.
    assert!(
        base.wrapping_opt().is_none(),
        "wrapping_opt must be absent by default"
    );

    // After explicit opt-in: present.
    let with = base.with_wrapping(WrappingOpt::new());
    assert!(
        with.wrapping_opt().is_some(),
        "wrapping_opt must be Some after with_wrapping()"
    );

    // A fresh base (the with_* builder consumes self): still absent, showing no mutation of
    // the source (builder pattern; copy is independent).
    let fresh = Meta::exact(Provenance::Root);
    assert!(
        fresh.wrapping_opt().is_none(),
        "a fresh Meta is independent — wrapping_opt absent"
    );
}

/// `WrappingOpt::new()` is the only constructor; it is idempotent and `PartialEq`-comparable.
/// This checks the type's own stability (no hidden state).
#[test]
fn wrapping_opt_new_is_stable_and_eq() {
    let a = WrappingOpt::new();
    let b = WrappingOpt::new();
    assert_eq!(a, b, "WrappingOpt::new() must be stable / no hidden state");
    // Default is the same as new() (Default impl delegates to new()).
    let d: WrappingOpt = Default::default();
    assert_eq!(a, d, "WrappingOpt::default() must equal WrappingOpt::new()");
}

// ---------------------------------------------------------------------------
// Orthogonality: with_wrapping does NOT silence Axis-A guarantee or cert_mode
// ---------------------------------------------------------------------------

/// **Property:** attaching `WrappingOpt` does NOT change the guarantee strength (Axis-A).
/// Sweeps `CertMode::ALL` × the four `GuaranteeStrength` cases for full coverage.
#[test]
fn wrapping_does_not_silence_axis_a_guarantee_in_any_mode() {
    use crate::bound::{Bound, BoundBasis, BoundKind, NormKind};

    // A representative bound for each non-Exact strength.
    let proven_bound = Bound {
        kind: BoundKind::Error {
            eps: 0.01,
            norm: NormKind::L2,
        },
        basis: BoundBasis::ProvenThm {
            citation: "test theorem".to_owned(),
        },
    };
    let empirical_bound = Bound {
        kind: BoundKind::Probability { delta: 0.05 },
        basis: BoundBasis::EmpiricalFit {
            trials: 1_000,
            method: "test Monte-Carlo".to_owned(),
        },
    };
    let declared_bound = Bound {
        kind: BoundKind::Error {
            eps: 0.2,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    };

    let cases: &[(GuaranteeStrength, Option<Bound>)] = &[
        (GuaranteeStrength::Exact, None),
        (GuaranteeStrength::Proven, Some(proven_bound)),
        (GuaranteeStrength::Empirical, Some(empirical_bound)),
        (GuaranteeStrength::Declared, Some(declared_bound)),
    ];

    for &mode in &CertMode::ALL {
        for (g, b) in cases {
            let base = Meta::new(Provenance::Root, *g, b.clone(), None, None, None)
                .expect("valid Meta for test case")
                .with_cert_mode(mode);

            let before_g = base.guarantee();
            let before_mode = base.cert_mode();

            // Attach wrapping — must not change guarantee or cert_mode.
            let wrapped = base.with_wrapping(WrappingOpt::new());

            assert_eq!(
                wrapped.guarantee(),
                before_g,
                "with_wrapping must not change guarantee (Axis-A) \
                 in mode {mode:?}, intended={g:?}"
            );
            assert_eq!(
                wrapped.cert_mode(),
                before_mode,
                "with_wrapping must not change cert_mode \
                 in mode {mode:?}, intended={g:?}"
            );
            assert!(
                wrapped.wrapping_opt().is_some(),
                "wrapping_opt must be Some after with_wrapping"
            );
        }
    }
}

/// **Property:** `with_wrapping` does NOT touch the bound (Axis-A accuracy/bound is orthogonal).
#[test]
fn wrapping_does_not_alter_bound() {
    use crate::bound::{Bound, BoundBasis, BoundKind, NormKind};

    let b = Bound {
        kind: BoundKind::Error {
            eps: 0.003,
            norm: NormKind::Rel,
        },
        basis: BoundBasis::ProvenThm {
            citation: "orthogonality check".to_owned(),
        },
    };
    let meta = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Proven,
        Some(b.clone()),
        None,
        None,
        None,
    )
    .expect("valid Meta");
    let wrapped = meta.with_wrapping(WrappingOpt::new());

    assert_eq!(
        wrapped.bound(),
        Some(&b),
        "with_wrapping must not alter the bound"
    );
}

// ---------------------------------------------------------------------------
// Mode-parametric conformance sweep (RFC-0034 §13)
// ---------------------------------------------------------------------------

/// RFC-0034 §13 conformance: "Axis-B holds in every mode" — sweeps the `Meta::exact` + every
/// `CertMode` combination to confirm the invariant is cross-mode. A `wrapping` annotation
/// is explicitly absent on the default path and can only appear via `with_wrapping`.
#[test]
fn axis_b_conformance_sweep_no_wrapping_by_default_in_every_mode() {
    // Conformance (RFC-0034 §13, Axis-B invariant): for every mode, the default `Meta` has
    // Axis-B active (wrapping_opt == None). Only an explicit with_wrapping call sets it.
    for &mode in &CertMode::ALL {
        let default_meta = Meta::exact(Provenance::Root).with_cert_mode(mode);
        assert!(
            default_meta.wrapping_opt().is_none(),
            "RFC-0034 §10 / §13 conformance: Axis-B must be default-on \
             (wrapping_opt == None) for mode {mode:?}"
        );

        // After explicit opt-in: present (the opt-in is named and visible, not ambient).
        let explicit_meta = Meta::exact(Provenance::Root)
            .with_cert_mode(mode)
            .with_wrapping(WrappingOpt::new());
        assert!(
            explicit_meta.wrapping_opt().is_some(),
            "RFC-0034 §10: explicit wrapping opt-in must set wrapping_opt in mode {mode:?}"
        );
        // cert_mode is untouched by the wrapping annotation.
        assert_eq!(
            explicit_meta.cert_mode(),
            mode,
            "with_wrapping must not change cert_mode (VR-5 orthogonality)"
        );
    }
}
