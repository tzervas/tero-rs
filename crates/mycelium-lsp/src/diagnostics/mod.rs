//! **Structured diagnostics & reified error policy** (M-345; RFC-0013, Accepted 2026-06-16).
//!
//! A *presentation* layer over the explicit, reasoned errors Mycelium already emits (a swap out of
//! range, a failed certificate, a `CheckVerdict::NotValidated`). It renders that one structured truth
//! at graded verbosity, in both human and machine form, and attaches an inspectable per-definition
//! policy that shapes the message, tags, level, and routing of a diagnostic. **Tooling layer only**
//! (no kernel dependency, KC-3; no Python, ADR-007).
//!
//! The governing principle (RFC-0013 §4.1, the operational form of never-silent G2):
//!
//! > **A diagnostic is *additive presentation* over an explicit error — never a substitute for one.**
//!
//! The pieces:
//! - [`registry`] — the error-class registry: names are **looked up, never `eval`-ed** (§4.5 X1).
//! - [`record`] — the content-addressed diagnostic, its **dual human + JSON projection** (G11), the
//!   graded [`Level`]s with an **allowlisted** detailed tier (§4.5 X2), and the never-silent
//!   [`present`] renderer (§4.1: the error is returned **unchanged** alongside the presentation, I1).
//! - [`policy`] — the reified `on <ErrorClass> => {message, tags, level, route}` policy (RFC-0005
//!   pattern; content-addressed `PolicyRef`), presentation/routing **only** (§4.4 I4).
//! - [`sink`] — the **closed v0 route vocabulary** ([`Route`]) and its binding to RFC-0008
//!   observability sinks, each with an **honest delivery guarantee** on the lattice (RT5; M-354,
//!   RFC-0013 §8). Routing never gates propagation (I1).
//! - [`audit`] — the **representation-crossing audit view** (§4.6; routed from RFC-0012 R12-Q2): every
//!   `swap` + its honesty bound (read off, **never upgraded** — VR-5) + its policy, location-independent.

pub mod audit;
pub mod policy;
pub mod record;
pub mod registry;
pub mod sink;

pub use audit::{AuditView, Crossing};
pub use policy::{DiagnosticPolicy, PolicyFile, Rule};
pub use record::{
    present, DiagnosticRecord, Level, Presentation, ReasonedError, DETAILED_ALLOWLIST,
};
pub use registry::{ClassName, ClassRegistry, UnknownClass};
pub use sink::{Delivery, Route, SinkBinding, UnknownRoute};
