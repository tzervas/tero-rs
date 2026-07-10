//! `mycelium-core` — the Core IR (M-101): `Value<Repr, Meta>`, the guarantee lattice, the bound
//! vocabulary, and the node grammar (RFC-0001 r2). The Rust types mirror the ratified
//! data-contract schemas under `docs/spec/schemas/`, and the honesty invariants (M-I1…M-I4) are
//! enforced *by construction* (see [`meta::Meta::new`]).
//!
//! Here so far: the guarantee `meet` composition + laws (M-102) and content-addressing (M-103).
//! Not yet here (own issues): (de)serialization to the schemas (M-104), the reference interpreter
//! (M-110).
//!
//! **Trusted-base discipline (ADR-014 / DN-21 §5 F-1):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

pub mod binary;
pub mod bound;
pub mod cert_mode;
pub mod content;
pub mod data;
pub mod datum;
pub mod guarantee;
pub mod id;
pub mod lower;
pub mod meta;
pub mod node;
pub mod prim;
pub mod recon;
pub mod repr;
pub mod ternary;
pub mod value;
pub mod wrapping;

#[cfg(test)]
mod tests;

pub use bound::{Bound, BoundBasis, BoundKind, NormKind};
pub use cert_mode::CertMode;
pub use content::{operation_hash, Names};
pub use data::{
    CtorDecl, CtorRef, CtorSpec, DataDecl, DataRegistry, DeclSpec, FieldSpec, FieldTy, FieldTyRef,
    FnSig, RegistryError, ResolvedFieldTyRef, ResolvedFnSig,
};
pub use datum::{CoreValue, Datum};
pub use guarantee::GuaranteeStrength;
pub use id::ContentHash;
pub use meta::{Meta, PackScheme, PhysicalLayout, Provenance, SparsityObs};
pub use node::{Alt, Node, PolicyRef, Prim, VarId};
pub use prim::{PrimDecl, PrimParadigm, PrimRef, PrimSig, PrimTable, WidthRel};
pub use recon::{
    CleanupShape, DecodeProcedure, DecodeSpec, InitStrategy, Recipe, ReconInfo, ReconMode,
};
pub use repr::{FloatWidth, Repr, ScalarKind, SparsityClass};
pub use value::{Payload, Trit, Value, CANONICAL_NAN_BITS};
pub use wrapping::WrappingOpt;

/// Well-formedness errors for Core IR construction (RFC-0001 §4.3/§4.5 invariants).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WfError {
    /// The guarantee/bound pairing violates M-I1…M-I4 (the honesty rule).
    GuaranteeBoundMismatch,
    /// A bound's numeric payload is out of range (e.g. `delta ∉ [0,1]`).
    MalformedBound,
    /// A representation has a non-positive width/dim/trits or an empty VSA model id.
    MalformedRepr,
    /// A representation declares a dimension above [`repr::MAX_DIM`] — rejected as an
    /// over-allocation (DoS) guard before any value of that `Repr` is materialized. Names the
    /// offending field, its declared value, and the cap (never-silent, G2). See
    /// [`Repr::check_well_formed`](crate::Repr::check_well_formed).
    DimensionTooLarge {
        /// The offending dimension field (`"width"` / `"trits"` / `"dim"` / `"max_active"`).
        field: &'static str,
        /// The declared (rejected) value.
        value: u32,
        /// The enforced upper bound ([`repr::MAX_DIM`]).
        cap: u32,
    },
    /// A payload does not match its representation (paradigm or length).
    PayloadReprMismatch,
    /// A reconstruction manifest violates its schema invariants (RFC-0003 §6;
    /// `reconstruction-manifest.schema.json`).
    MalformedReconstruction,
    /// A measured sparsity observation is out of range (e.g. `density ∉ [0,1]`). Distinct from
    /// [`WfError::MalformedBound`]: a [`SparsityObs`] is an observation, not a
    /// guarantee bound (A6-08).
    MalformedSparsity,
}

impl core::fmt::Display for WfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `DimensionTooLarge` names the field/value/cap inline (never-silent, G2), so it can't use
        // the static-string arm below.
        if let WfError::DimensionTooLarge { field, value, cap } = self {
            return write!(
                f,
                "representation dimension '{field}' = {value} exceeds the cap {cap} \
                 (over-allocation guard)"
            );
        }
        let s = match self {
            WfError::GuaranteeBoundMismatch => "guarantee/bound inconsistency (M-I1..M-I4)",
            WfError::MalformedBound => "bound payload out of range",
            WfError::MalformedRepr => {
                "representation has non-positive width/dim/trits or empty model"
            }
            WfError::PayloadReprMismatch => "payload does not match its representation",
            WfError::MalformedReconstruction => {
                "reconstruction manifest violates its schema invariants (RFC-0003 §6)"
            }
            WfError::MalformedSparsity => "sparsity observation out of range (density ∉ [0,1])",
            WfError::DimensionTooLarge { .. } => unreachable!("handled above"),
        };
        f.write_str(s)
    }
}

impl std::error::Error for WfError {}
