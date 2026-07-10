//! The named Axis-B `wrapping` opt-out (RFC-0034 §10; M-791).
//!
//! **Never-silent failability (Axis B)** — out-of-range yields `Option`/`Result`/`SwapError` —
//! is **default-on in every mode**, including `fast`. It is O(1) and is the cheapest part of the
//! transparency floor (RFC-0034 §3.3). A developer who genuinely wants wraparound arithmetic
//! opts in via the **named, explicit [`WrappingOpt`]** marker — and choosing it is itself
//! never-silent: the opt-out is visible at the use site and does **not** silence Axis-A
//! (guarantee tags, `cert_mode`) or any other transparency dimension (G2/VR-5).
//!
//! ## Design
//!
//! `WrappingOpt` is a **marker** carried on [`Meta`](crate::meta::Meta) when the operation that
//! produced the value was given an explicit, named `wrapping` opt-out. It is:
//!
//! - **Absent by default** (`Meta::wrapping_opt() == None`) — Axis-B never-silent failability is
//!   the default; no annotation needed for the safe path.
//! - **Explicit to attach** — only [`Meta::with_wrapping`](crate::meta::Meta::with_wrapping)
//!   sets it; there is no ambient "wrapping mode" or implicit coercion.
//! - **Does not affect guarantee strength or cert_mode** — a `wrapping` value can be `Exact` or
//!   `Declared`, `fast` or `certified`; the Axis-A / certification axes are orthogonal (VR-5:
//!   opt-out of one axis never upgrades another).
//! - **Excluded from the content hash** — like all of `Meta`, it rides `Meta` which RFC-0001 §4.6
//!   excludes wholesale. The same value with vs without `wrapping` annotation is content-identical.
//!
//! ## Downstream (FLAG — M-791)
//!
//! The **op-layer wiring** — arithmetic/swap operations that actually honor the `WrappingOpt`
//! marker by electing wraparound instead of returning `Option`/`Result`/`SwapError` — is a
//! downstream task (M-788 onward). This module and the `Meta` tag are the **representation-layer**
//! groundwork; the op layer is wired once arithmetic/swap operations exist.

/// The explicit, named Axis-B opt-out (RFC-0034 §10; M-791).
///
/// Present on a [`Meta`](crate::meta::Meta) only when the producing operation was given a
/// **named, explicit** `wrapping` annotation at the use site. Its absence is the default state
/// (never-silent Axis-B failability applies). Its presence marks "the caller explicitly requested
/// wraparound; this is visible and auditable."
///
/// `WrappingOpt` does **not** alter the guarantee strength (`Exact`/`Proven`/…) or the
/// certification mode (`Fast`/`Certified`/…); those are Axis-A and certification axes and remain
/// orthogonal (VR-5). A `wrapping` annotation is "I chose wraparound here, explicitly" — it is
/// not "I lowered the accuracy bar".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WrappingOpt {
    /// Reason field — reserved for v1 (M-791): a developer-supplied justification string (like the
    /// `// SAFETY:` convention for `unsafe`). In v0 the only legal value is `()` (the unit token is
    /// the explicit acknowledgement: no magic string, no default comment). The field is private so
    /// the stable public surface is just `WrappingOpt::new()`.
    _reason: (),
}

impl WrappingOpt {
    /// Construct the explicit `wrapping` opt-out marker (v0). The call site is the acknowledgement:
    /// the developer wrote `WrappingOpt::new()` — named, visible, grep-auditable (RFC-0034 §10).
    ///
    /// **v1 will require a justification argument** (analogous to `// SAFETY:`). Until then, the
    /// type's existence at the use site is the disclosure.
    #[must_use]
    pub const fn new() -> Self {
        WrappingOpt { _reason: () }
    }
}

impl Default for WrappingOpt {
    fn default() -> Self {
        Self::new()
    }
}
