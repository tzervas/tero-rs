//! The certification mode (RFC-0034 §3.1, §5) — the tunable certification *policy* a value was
//! produced under.
//!
//! Carried on every [`Meta`](crate::meta::Meta) as a **never-silent** tag (RFC-0034 §3.1) and
//! deliberately **excluded from the content hash**: it rides `Meta`, which RFC-0001 §4.6 excludes
//! wholesale, so switching modes never perturbs a value's content identity (ADR-003). That exclusion
//! is therefore by construction, not a special case — see the `content_hash` exclusion test.
//!
//! Two **first-class** modes — [`Fast`](CertMode::Fast) (the default) and
//! [`Certified`](CertMode::Certified) — with [`Balanced`](CertMode::Balanced) an optional
//! intermediate (RFC-0034 §5). The mode is *disclosure of how much certification ran*, ordered by
//! [`depth`](CertMode::depth) `Fast < Balanced < Certified`. It is **not** a guarantee strength and
//! never upgrades one (VR-5): a `Fast` value sits at the structural `Exact`/`Declared` tags and never
//! claims an `Empirical`/`Proven` it did not earn. M-786 introduces the type + the never-silent tag;
//! M-787 adds [`gate_guarantee`](CertMode::gate_guarantee) — the policy that floors `fast` to the
//! structural tags.

use crate::bound::{Bound, BoundBasis};
use crate::guarantee::GuaranteeStrength;

/// The active certification mode a value was produced under (RFC-0034). Default
/// [`Fast`](CertMode::Fast) — the project default (RFC-0034 §5).
///
/// The `serde` form is the bare string `"Fast" | "Balanced" | "Certified"` (mirroring
/// [`GuaranteeStrength`](crate::guarantee::GuaranteeStrength)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CertMode {
    /// **Fast** (default): no runtime certification machinery — cert-free, memory-safe, inspectable,
    /// and still deployable (spores survive a cert-off runtime, RFC-0034 §8). Provenance tags stay
    /// structural (`Exact`/`Declared`); `Empirical`/`Proven` are not computed (RFC-0034 §7; M-787).
    #[default]
    Fast,
    /// **Balanced** (intermediate): provenance tags propagated and swap certificates *emitted*, but
    /// **unchecked** (RFC-0034 §5).
    Balanced,
    /// **Certified**: the full, checked, certificate-backed auditable framework — today's all-on
    /// behaviour, engaged on request (RFC-0034 §5).
    Certified,
}

impl CertMode {
    /// All three modes, weakest-to-strongest certification depth — for exhaustive iteration in tests
    /// and tooling.
    pub const ALL: [CertMode; 3] = [CertMode::Fast, CertMode::Balanced, CertMode::Certified];

    /// Certification **depth**, `0` = [`Fast`](CertMode::Fast) (least) … `2` =
    /// [`Certified`](CertMode::Certified) (most). Higher = more certification machinery engaged.
    ///
    /// This orders the modes; it is **not** a guarantee strength — a stronger mode never upgrades a
    /// value's [`GuaranteeStrength`](crate::guarantee::GuaranteeStrength) (VR-5). Composing across
    /// modes is an explicit, visible event (RFC-0034 §3.1), never a silent upgrade.
    #[must_use]
    pub fn depth(self) -> u8 {
        match self {
            CertMode::Fast => 0,
            CertMode::Balanced => 1,
            CertMode::Certified => 2,
        }
    }

    /// Gate an operation's *intended* intrinsic guarantee strength by this mode (RFC-0034 §7; M-787).
    ///
    /// `Fast` does not run the trials behind [`Empirical`](GuaranteeStrength::Empirical) or the proof
    /// behind [`Proven`](GuaranteeStrength::Proven), so it **floors both to**
    /// [`Declared`](GuaranteeStrength::Declared) — the honest "computed, but its bound is asserted not
    /// verified in this mode" tag (VR-5: never claim a strength you did not establish). The free,
    /// **structural** [`Exact`](GuaranteeStrength::Exact) (e.g. a bijective binary↔ternary swap) passes
    /// untouched, and an already-`Declared` tag is unchanged. `Balanced` and `Certified` run the
    /// machinery, so they pass every strength through identically (the mechanism is unchanged).
    ///
    /// This is the policy primitive; the operation layer applies it when it constructs a result's
    /// `Meta` (the bound's basis is relabelled to `UserDeclared` in lockstep so M-I1…M-I4 stay
    /// consistent — wired where operations become mode-aware, M-788 onward). It guarantees the M-787
    /// invariant directly: **no `Fast` result ever carries `Empirical`/`Proven`.**
    #[must_use]
    pub fn gate_guarantee(self, intended: GuaranteeStrength) -> GuaranteeStrength {
        use GuaranteeStrength::{Declared, Empirical, Exact, Proven};
        match self {
            CertMode::Fast => match intended {
                Exact => Exact,
                Proven | Empirical | Declared => Declared,
            },
            CertMode::Balanced | CertMode::Certified => intended,
        }
    }

    /// Gate an operation's *intended* `(guarantee, bound)` **pair** by this mode, reconciling the
    /// bound's basis with the floored guarantee so the result still satisfies the `Meta` invariants
    /// **M-I1…M-I4** (RFC-0034 §7; M-788 — the bound/basis half [`gate_guarantee`] explicitly
    /// deferred).
    ///
    /// The guarantee is gated exactly as [`gate_guarantee`](CertMode::gate_guarantee). The subtlety
    /// is the *bound*: when `Fast` floors a would-be [`Proven`](GuaranteeStrength::Proven)/
    /// [`Empirical`](GuaranteeStrength::Empirical) result to [`Declared`](GuaranteeStrength::Declared),
    /// the operation has *computed* a bound value (an ε/δ) but `Fast` ran **no** certification
    /// machinery to check it — so the bound's basis ([`ProvenThm`](BoundBasis::ProvenThm)/
    /// [`EmpiricalFit`](BoundBasis::EmpiricalFit)) is no longer earned. Carrying it unchanged would
    /// violate **M-I4** (`Declared ⟹ UserDeclared`). The honest reconciliation (VR-5: never claim a
    /// basis you did not establish) is to **keep the computed bound *value* but relabel its basis to
    /// [`UserDeclared`](BoundBasis::UserDeclared)** — "computed, but asserted-not-verified in this
    /// mode" — which is exactly the basis M-I4 requires for a `Declared` tag.
    ///
    /// Concretely, mode by intended strength:
    /// - **`Fast` + `Exact`** → `(Exact, None)`. `Exact` is structural and bound-free; the caller's
    ///   `bound` is already `None` by M-I1 (a non-`None` bound on an `Exact` intent is itself a
    ///   bug the `Meta` constructor would reject), and this preserves M-I1.
    /// - **`Fast` + `Proven`/`Empirical`/`Declared`** → `(Declared, bound.relabel(UserDeclared))`.
    ///   The value is kept; only the basis is demoted, satisfying M-I4. (An already-`Declared`
    ///   intent already carries a `UserDeclared` basis, so the relabel is a no-op for it.)
    /// - **`Balanced`/`Certified`** → `(intended, bound)` unchanged — the machinery runs, so the
    ///   computed basis is earned and passes through (the mechanism is preserved).
    ///
    /// The returned pair is **always** one the `Meta` constructor (which enforces M-I1…M-I4) accepts
    /// — that round-trip is the unit-tested contract (`gate_result` then `Meta::new` never errors on
    /// the invariant check). Numeric well-formedness of the bound is unchanged by the relabel (the
    /// payload is untouched), so a well-formed input stays well-formed.
    #[must_use]
    pub fn gate_result(
        self,
        intended_guarantee: GuaranteeStrength,
        intended_bound: Option<Bound>,
    ) -> (GuaranteeStrength, Option<Bound>) {
        use GuaranteeStrength::{Declared, Empirical, Exact, Proven};
        match self {
            CertMode::Fast => match intended_guarantee {
                // Structural exact: bound stays None (M-I1). Any stray bound is dropped — an
                // `Exact` result is bound-free by definition, never a silent carry.
                Exact => (Exact, None),
                // Floored to Declared: keep the computed bound value, relabel its basis to
                // UserDeclared so M-I4 holds (honest: computed, asserted-not-verified in Fast).
                Proven | Empirical | Declared => {
                    (Declared, intended_bound.map(relabel_user_declared))
                }
            },
            // The machinery runs: the intended pair is earned and passes through unchanged.
            CertMode::Balanced | CertMode::Certified => (intended_guarantee, intended_bound),
        }
    }
}

/// Relabel a bound's basis to [`UserDeclared`](BoundBasis::UserDeclared), keeping its `kind`
/// payload (the computed ε/δ value) intact. Used when `Fast` floors a computed-but-unchecked bound
/// to a `Declared` tag (M-788): the value is honest, the *basis* is demoted to "asserted, not
/// verified in this mode" — the only basis M-I4 admits for `Declared`.
fn relabel_user_declared(mut bound: Bound) -> Bound {
    bound.basis = BoundBasis::UserDeclared;
    bound
}
