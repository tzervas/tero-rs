//! The `std-sys` syscall-floor guarantee matrix encoded as **data** (RFC-0016 §4.5; M-722/M-723).
//!
//! Every exported syscall-floor operation has exactly one row in [`MATRIX`]. Until this module
//! existed the per-op tags lived only in each module's prose doc table; RFC-0016 §4.5 / VR-5 require
//! the matrix to be **data, asserted in tests, never prose-only** — that is what this module
//! supplies (closing the M-722/M-723 matrix gap).
//!
//! # Honesty (VR-5) — every row is `Declared`
//!
//! `std-sys` is the audited FFI/OS floor. **No** op here carries a checked theorem (`Proven`) or a
//! measured error corpus (`Empirical`): libm precision, OS entropy quality, filesystem semantics,
//! stream buffering, and clock resolution are all asserted-but-unchecked. So every row is
//! **`Declared`** and is FLAGGED as such — the honest floor for an unaudited host wrapper (RFC-0016
//! §4.1 C2). Promotion of any single op requires its own checked basis (documented `Empirical`
//! coverage or a `Proven` side-condition theorem); none is established in v0. The test
//! [`tests::every_row_is_declared`] is the mechanical guard against an inadvertent upgrade (VR-5).
//!
//! # Never-silent (G2) — fallibility column
//!
//! Every fallible op returns an explicit `Result`/`Option`; its error set is named in `error_set`.
//! The total ops (`mono_nanos`, `sleep_nanos`, `args`, the libm `math::*`, and `fs::exists`) cannot
//! fail for a well-formed input. `fs::exists` follows `std::path::Path::exists` semantics (an OS
//! error reads as `false`); that coalescing is noted in its row rather than hidden.
//!
//! # Effect column (C6 / RFC-0014 §4.5)
//!
//! `"io"` (stream contact) · `"fs"` (filesystem) · `"entropy"` (OS CSPRNG) · `"time"` (clock) ·
//! `"time + entropy"` (wall clock — a civil-time entropy source, RT3) · `"process"` /`"env"`
//! (process/environment) · `"none"` (pure, e.g. the libm transcendentals over their `f64` input).

/// Guarantee tag on the honesty lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (RFC-0016 §4.1 C2;
/// VR-5). A standalone enum whose variants mirror the lattice names; [`GuaranteeTag::as_str`] yields
/// the matching `'static` string, so rows are assertable in tests without depending on the core
/// lattice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuaranteeTag {
    /// No accuracy/precision/probability semantics — the honest floor `Exact` (unused here).
    Exact,
    /// Established by a checked side-condition theorem (VR-5; unused here).
    Proven,
    /// Holds over a generated corpus (proptest); not `Proven` (VR-5; unused here).
    Empirical,
    /// Asserted without a checked basis; always FLAGGED (VR-5). Every `std-sys` floor op is this.
    Declared,
}

impl GuaranteeTag {
    /// Human-readable name matching the lattice notation (`"Declared"`, etc.).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            GuaranteeTag::Exact => "Exact",
            GuaranteeTag::Proven => "Proven",
            GuaranteeTag::Empirical => "Empirical",
            GuaranteeTag::Declared => "Declared",
        }
    }
}

/// Fallibility classification for an exported op (C1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// The op cannot fail for any well-formed input (it has no `Result`/`Option` failure path).
    Total,
    /// The op returns an explicit `Result`/`Option`; the error set is named in `error_set`.
    Fallible,
}

/// One row in the `std-sys` syscall-floor guarantee matrix (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// Module-qualified exported operation name (e.g. `"io::write_out"`).
    pub op: &'static str,
    /// Guarantee tag (VR-5) — `Declared` for every floor op.
    pub guarantee: GuaranteeTag,
    /// Fallibility: total or fallible.
    pub fallibility: Fallibility,
    /// The explicit error set (empty string for total ops).
    pub error_set: &'static str,
    /// Declared effects (C6).
    pub effects: &'static str,
}

/// The `std-sys` syscall-floor guarantee matrix — one row per exported floor op.
///
/// Asserted in [`tests`] — never prose-only (C2 / VR-5 / RFC-0016 §4.5).
pub const MATRIX: &[MatrixRow] = &[
    // ── io (RFC-0028 §4.5; M-722) — standard-stream contact, effect `io` ────────────────────────
    MatrixRow {
        op: "io::read_to_end",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — OS read error (never a silent truncation, G2); \
                    UNBOUNDED alloc (trusted-only — use read_to_end_capped for untrusted input)",
        effects: "io",
    },
    MatrixRow {
        op: "io::read_to_end_capped",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(ReadCappedError) — OS read error / cap exceeded \
                    (P3 bounded; refuses, never truncates, G2)",
        effects: "io",
    },
    MatrixRow {
        op: "io::read_line",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — OS read / UTF-8 error",
        effects: "io",
    },
    MatrixRow {
        op: "io::write_out",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — short/failed write (never silently dropped, G2)",
        effects: "io",
    },
    MatrixRow {
        op: "io::write_err",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — short/failed write",
        effects: "io",
    },
    MatrixRow {
        op: "io::flush_out",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — OS flush error",
        effects: "io",
    },
    // ── fs (RFC-0028 §4.5; M-722) — filesystem contact, effect `fs` ─────────────────────────────
    MatrixRow {
        op: "fs::read",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — open/read error",
        effects: "fs",
    },
    MatrixRow {
        op: "fs::write",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — open/write error",
        effects: "fs",
    },
    MatrixRow {
        op: "fs::exists",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "fs (std::path::Path::exists semantics — an OS error reads as false)",
    },
    MatrixRow {
        op: "fs::create_dir_all",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — mkdir error",
        effects: "fs",
    },
    MatrixRow {
        op: "fs::remove_file",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(io::Error) — unlink error",
        effects: "fs",
    },
    // ── sys (RFC-0028 §4.5; M-722) — process / environment ──────────────────────────────────────
    MatrixRow {
        op: "sys::exit",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "process (diverges — `-> !`)",
    },
    MatrixRow {
        op: "sys::get_env",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "None — unset/invalid var (never a silent empty string, G2)",
        effects: "env",
    },
    MatrixRow {
        op: "sys::args",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(NonUtf8Arg) — non-UTF-8 arg named by index (never a silent drop, G2)",
        effects: "process",
    },
    // ── rand (M-723) — OS entropy (`/dev/urandom`), effect `entropy` ────────────────────────────
    MatrixRow {
        op: "rand::fill_bytes",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set:
            "Err(EntropyError::Unavailable) — no `/dev/urandom` / short read (never zero-fills, G2)",
        effects: "entropy",
    },
    // ── time (M-723) — OS clock, effects `time` / `time + entropy` ──────────────────────────────
    MatrixRow {
        op: "time::wall_nanos",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Fallible,
        error_set: "Err(String) — clock before Unix epoch",
        effects: "time + entropy",
    },
    MatrixRow {
        op: "time::mono_nanos",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "time",
    },
    MatrixRow {
        op: "time::sleep_nanos",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "time",
    },
    // ── math (M-541) — libm transcendental floor; pure over its f64 input, effect `none` ────────
    MatrixRow {
        op: "math::sin",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::cos",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::tan",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::asin",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::acos",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::atan",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::atan2",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::exp",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::exp2",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::ln",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::log2",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::log10",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::sqrt",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
    MatrixRow {
        op: "math::cbrt",
        guarantee: GuaranteeTag::Declared,
        fallibility: Fallibility::Total,
        error_set: "",
        effects: "none",
    },
];

// ── Tests ───────────────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::{Fallibility, GuaranteeTag, MATRIX};

    /// Every exported syscall-floor op appears in the matrix exactly once. Guard: adding or removing
    /// a floor op without updating the matrix makes this fail.
    #[test]
    fn matrix_covers_every_floor_op_once() {
        let expected = [
            "io::read_to_end",
            "io::read_to_end_capped",
            "io::read_line",
            "io::write_out",
            "io::write_err",
            "io::flush_out",
            "fs::read",
            "fs::write",
            "fs::exists",
            "fs::create_dir_all",
            "fs::remove_file",
            "sys::exit",
            "sys::get_env",
            "sys::args",
            "rand::fill_bytes",
            "time::wall_nanos",
            "time::mono_nanos",
            "time::sleep_nanos",
            "math::sin",
            "math::cos",
            "math::tan",
            "math::asin",
            "math::acos",
            "math::atan",
            "math::atan2",
            "math::exp",
            "math::exp2",
            "math::ln",
            "math::log2",
            "math::log10",
            "math::sqrt",
            "math::cbrt",
        ];
        for name in &expected {
            let n = MATRIX.iter().filter(|r| r.op == *name).count();
            assert_eq!(
                n, 1,
                "op {name:?} must appear exactly once in MATRIX (found {n})"
            );
        }
        assert_eq!(
            MATRIX.len(),
            expected.len(),
            "MATRIX has unexpected extra/missing rows"
        );
    }

    /// Every floor op is `Declared` (VR-5). Guard: upgrading any row to `Empirical`/`Proven`/`Exact`
    /// without a checked basis makes this fail — the honest floor for an unaudited host wrapper.
    #[test]
    fn every_row_is_declared() {
        for row in MATRIX {
            assert_eq!(
                row.guarantee,
                GuaranteeTag::Declared,
                "op {:?} is not Declared — std-sys is the unaudited OS floor; an upgrade needs its \
                 own checked Empirical/Proven basis (VR-5)",
                row.op
            );
        }
    }

    /// Fallible ops name a non-empty error set; total ops have an empty one (C1 / never-silent G2).
    #[test]
    fn fallibility_and_error_set_are_consistent() {
        for row in MATRIX {
            match row.fallibility {
                Fallibility::Total => assert!(
                    row.error_set.is_empty(),
                    "total op {:?} must have an empty error_set (C1)",
                    row.op
                ),
                Fallibility::Fallible => assert!(
                    !row.error_set.is_empty(),
                    "fallible op {:?} must name its error set (C1 / G2)",
                    row.op
                ),
            }
        }
    }

    /// Every row declares a non-empty effect (C6) — the floor is the OS-contact surface, so no op is
    /// silently effect-free except the pure libm `math::*` ops (effect `none`).
    #[test]
    fn effects_are_declared() {
        for row in MATRIX {
            assert!(
                !row.effects.is_empty(),
                "op {:?} must declare an effect (C6)",
                row.op
            );
        }
        // The wall clock is a civil-time entropy source (RT3) — it must declare entropy, not just time.
        let wall = MATRIX
            .iter()
            .find(|r| r.op == "time::wall_nanos")
            .expect("wall row");
        assert!(
            wall.effects.contains("entropy"),
            "time::wall_nanos must declare the entropy effect (RT3 — civil time is an entropy source)"
        );
        // The OS entropy draw declares entropy.
        let fill = MATRIX
            .iter()
            .find(|r| r.op == "rand::fill_bytes")
            .expect("fill row");
        assert_eq!(
            fill.effects, "entropy",
            "rand::fill_bytes must declare effect `entropy`"
        );
    }
}
