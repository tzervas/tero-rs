//! `std.fs` guarantee matrix encoded as **data** (RFC-0016 §4.5; spec §4).
//!
//! Every exported operation has one row in [`MATRIX`]. The matrix is the load-bearing C2
//! deliverable (RFC-0016 §4.1 C2 / VR-5): guarantee tags are asserted in tests, not prose-only.
//!
//! # Guarantee tag justification (all `Exact`)
//! `fs` carries **no** accuracy/approximation/probability semantics — there is no ε, no δ, no
//! `BoundBasis` to resolve against (contrast `std.math` M-525). `Exact` is the *honest* tag here,
//! not an upgrade: there is no precision dimension to overclaim. The honesty of `fs` is borne
//! entirely by C1 (explicit fallibility) and C6 (declared effects) — not by a C2 precision tag.
//! Every op either performs its named filesystem effect and returns, or it returns an explicit
//! `FsErr` — there is **no** silently-degraded third outcome (spec §4 tag justification).
//!
//! # Fallibility column
//! Every io-effecting op can fail; path ops are total (Exact, `Option` for `parent`).
//!
//! # `wild`? column
//! Marks which ops bottom out in the audited OS syscall floor (ADR-014). Pure lexical path ops
//! and the UseAfterConsume catch do **not** hit the syscall floor. This column is the `wild`
//! inventory the spec §4 acceptance criterion demands.
//!
//! **FLAG (std-sys / Q1):** The in-memory substrate implementation here does NOT actually invoke
//! the `wild` syscall floor. The `wild?` column marks the *design intent* — which ops WILL hit
//! the floor when `std-sys` (M-541) is wired in. The column is correct for the target architecture;
//! the current implementation uses `InMemoryFs` which has no `unsafe` code.

/// Fallibility classification for an exported op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallibility {
    /// The op is total: it cannot fail for any well-formed input.
    Total,
    /// The op can return `Option` with an honest `None` (not a sentinel).
    OptionFallible,
    /// The op returns `Result`; the error set is described in `error_set`.
    ResultFallible,
}

/// Whether an op reaches the audited OS syscall floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wild {
    /// This op invokes OS syscalls (the audited `wild` block floor — ADR-014).
    Yes,
    /// This op is pure / caught above the OS floor; no syscall.
    No,
}

/// Whether an op has an EXPLAIN obligation (C3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Explainable {
    /// The op surfaces an inspectable RFC-0013 diagnostic record on failure.
    Yes,
    /// The op is total or pure; no diagnostic record needed.
    NotApplicable,
}

/// Declared effects for an op (C6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effects {
    /// No side effects; pure computation.
    None,
    /// Filesystem IO effect (declared; cannot be silent — RFC-0014).
    Io,
}

/// One row in the `std.fs` guarantee matrix (RFC-0016 §4.5 / spec §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation or type.
    pub op: &'static str,
    /// Guarantee tag (all `"Exact"` — see module-level justification).
    pub guarantee: &'static str,
    /// Fallibility classification.
    pub fallibility: Fallibility,
    /// The explicit error / None-case description (empty for total ops).
    pub error_set: &'static str,
    /// Declared effects.
    pub effects: Effects,
    /// Whether the op surfaces an RFC-0013 diagnostic record.
    pub explainable: Explainable,
    /// Whether the op reaches the OS syscall floor (`wild`? column, spec §4).
    pub wild: Wild,
}

/// The `std.fs` guarantee matrix. One row per exported op, encoded as data and asserted in
/// `tests` — never prose-only (RFC-0016 §4.5 / spec §4).
///
/// Every row is `"Exact"` — fs carries no accuracy semantics (spec §4 / VR-5).
/// Every io-effecting row declares `Effects::Io` (C6: no undeclared side effects).
/// Every fallible row has a non-empty `error_set` (C1: never-silent).
/// Every io-effecting row is `Explainable::Yes` (C3: RFC-0013 diagnostic record on failure).
/// The `wild` column marks the OS-floor boundary (ADR-014 inventory — spec §4/§6).
pub const MATRIX: &[MatrixRow] = &[
    // ─── Pure lexical path ops (no IO, no syscall) ───────────────────────────
    MatrixRow {
        op: "Path::new / join",
        guarantee: "Exact",
        fallibility: Fallibility::Total,
        error_set: "",
        effects: Effects::None,
        explainable: Explainable::NotApplicable,
        wild: Wild::No,
    },
    MatrixRow {
        op: "parent",
        guarantee: "Exact",
        fallibility: Fallibility::OptionFallible,
        error_set: "None at root (no parent)",
        effects: Effects::None,
        explainable: Explainable::NotApplicable,
        wild: Wild::No,
    },
    // ─── IO-effecting ops (all `wild`? = yes in real-OS floor) ──────────────
    MatrixRow {
        op: "exists",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(PermDenied)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "open",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotFound | AlreadyExists | PermDenied | IsADirectory | Loop)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "read (io M-514 seam)",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(Io | PermDenied | Interrupted | WouldBlock); short read explicit",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "write (io M-514 seam)",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(Io | DiskFull | PermDenied); short write explicit (no silent partial)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "flush (io M-514 seam)",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(Io | DiskFull)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "close",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(Io | DiskFull); CONSUMES the handle (LR-8)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "stat / metadata",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotFound | PermDenied)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "read_dir / list",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotFound | NotADirectory | PermDenied)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "create_dir",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(AlreadyExists | PermDenied | NotFound)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "remove_file",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotFound | PermDenied | IsADirectory)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "remove_dir",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotEmpty | NotFound | PermDenied)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "rename",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotFound | PermDenied | CrossDevice)",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    MatrixRow {
        op: "copy",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(NotFound | PermDenied | DiskFull); no silent overwrite unless declared",
        effects: Effects::Io,
        explainable: Explainable::Yes,
        wild: Wild::Yes,
    },
    // ─── Affine handle violation (caught above the OS floor) ─────────────────
    MatrixRow {
        op: "(any op on a consumed handle)",
        guarantee: "Exact",
        fallibility: Fallibility::ResultFallible,
        error_set: "Err(UseAfterConsume) (LR-8)",
        effects: Effects::None,
        explainable: Explainable::Yes,
        wild: Wild::No,
    },
];

#[cfg(test)]
mod tests {
    use super::{Effects, Explainable, Fallibility, Wild, MATRIX};

    /// Every op named in the spec §3 surface appears in the matrix.
    /// Guard: removing any op from MATRIX makes this fail.
    #[test]
    fn matrix_contains_all_spec_ops() {
        let expected = [
            "Path::new / join",
            "parent",
            "exists",
            "open",
            "read (io M-514 seam)",
            "write (io M-514 seam)",
            "flush (io M-514 seam)",
            "close",
            "stat / metadata",
            "read_dir / list",
            "create_dir",
            "remove_file",
            "remove_dir",
            "rename",
            "copy",
            "(any op on a consumed handle)",
        ];
        for name in &expected {
            assert!(
                MATRIX.iter().any(|r| r.op == *name),
                "matrix is missing op {:?} (spec §3)",
                name
            );
        }
        assert_eq!(
            MATRIX.len(),
            expected.len(),
            "matrix has unexpected extra rows"
        );
    }

    /// Every row carries the `Exact` guarantee tag (spec §4 / VR-5).
    /// Guard: changing any tag makes this fail.
    #[test]
    fn every_row_is_exact() {
        for row in MATRIX {
            assert_eq!(
                row.guarantee, "Exact",
                "op {:?} must be Exact — fs carries no accuracy semantics (VR-5)",
                row.op
            );
        }
    }

    /// Total ops have an empty error_set; fallible ops have a non-empty one (C1).
    /// Guard: leaving error_set empty for a fallible op makes this fail.
    #[test]
    fn fallibility_and_error_set_are_consistent() {
        for row in MATRIX {
            match row.fallibility {
                Fallibility::Total => assert!(
                    row.error_set.is_empty(),
                    "total op {:?} must have empty error_set",
                    row.op
                ),
                Fallibility::OptionFallible | Fallibility::ResultFallible => assert!(
                    !row.error_set.is_empty(),
                    "fallible op {:?} must name its error set (C1)",
                    row.op
                ),
            }
        }
    }

    /// Every IO-effecting op declares `Effects::Io` (C6: no undeclared side effects).
    /// Guard: marking an IO op as Effects::None makes this fail.
    #[test]
    fn io_ops_declare_io_effect() {
        // The ops that are known IO ops by the spec.
        let io_ops = [
            "exists",
            "open",
            "read (io M-514 seam)",
            "write (io M-514 seam)",
            "flush (io M-514 seam)",
            "close",
            "stat / metadata",
            "read_dir / list",
            "create_dir",
            "remove_file",
            "remove_dir",
            "rename",
            "copy",
        ];
        for name in &io_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *name)
                .unwrap_or_else(|| panic!("op {name:?} not in MATRIX"));
            assert_eq!(
                row.effects,
                Effects::Io,
                "op {:?} must declare Effects::Io (C6)",
                name
            );
        }
    }

    /// Pure lexical path ops and UseAfterConsume declare no effects (C6).
    /// Guard: marking a pure op as Effects::Io makes this fail.
    #[test]
    fn pure_ops_declare_no_effects() {
        let pure_ops = [
            "Path::new / join",
            "parent",
            "(any op on a consumed handle)",
        ];
        for name in &pure_ops {
            let row = MATRIX
                .iter()
                .find(|r| r.op == *name)
                .unwrap_or_else(|| panic!("op {name:?} not in MATRIX"));
            assert_eq!(
                row.effects,
                Effects::None,
                "op {:?} must declare Effects::None (pure)",
                name
            );
        }
    }

    /// Every IO-effecting fallible row is EXPLAIN-able (C3: RFC-0013 diagnostic record).
    /// Guard: marking an IO op Explainable::NotApplicable makes this fail.
    #[test]
    fn io_fallible_rows_are_explainable() {
        for row in MATRIX {
            if row.effects == Effects::Io && row.fallibility == Fallibility::ResultFallible {
                assert_eq!(
                    row.explainable,
                    Explainable::Yes,
                    "IO fallible op {:?} must be Explainable::Yes (C3)",
                    row.op
                );
            }
        }
    }

    /// Pure total ops are NotApplicable for EXPLAIN (no failure to explain).
    #[test]
    fn total_pure_ops_are_not_explainable() {
        for row in MATRIX {
            if row.effects == Effects::None && row.fallibility == Fallibility::Total {
                assert_eq!(
                    row.explainable,
                    Explainable::NotApplicable,
                    "total pure op {:?} must be NotApplicable for EXPLAIN",
                    row.op
                );
            }
        }
    }

    /// IO-effecting ops are marked `wild: Yes` (they hit the syscall floor in the real-OS path).
    /// Pure lexical and UseAfterConsume ops are `wild: No`.
    #[test]
    fn wild_column_matches_spec_inventory() {
        let should_be_wild: Vec<&str> = MATRIX
            .iter()
            .filter(|r| r.wild == Wild::Yes)
            .map(|r| r.op)
            .collect();

        // All IO-effecting ops should be wild.
        for row in MATRIX {
            if row.effects == Effects::Io {
                assert!(
                    row.wild == Wild::Yes,
                    "IO op {:?} must be marked wild: Yes (ADR-014 inventory)",
                    row.op
                );
            }
        }

        // Pure lexical ops and UseAfterConsume are NOT wild.
        for row in MATRIX {
            if row.effects == Effects::None {
                assert!(
                    row.wild == Wild::No,
                    "pure op {:?} must be marked wild: No",
                    row.op
                );
            }
        }

        // The wild column is non-empty (at least one op hits the floor).
        assert!(
            !should_be_wild.is_empty(),
            "at least one op must be wild: Yes (fs cannot be wholly wild-free)"
        );
    }

    /// The UseAfterConsume row is `wild: No` (caught above the OS floor, spec §4).
    #[test]
    fn use_after_consume_is_not_wild() {
        let row = MATRIX
            .iter()
            .find(|r| r.op == "(any op on a consumed handle)")
            .expect("UseAfterConsume row must exist");
        assert_eq!(
            row.wild,
            Wild::No,
            "UseAfterConsume must be wild: No (caught above the floor)"
        );
    }

    /// The number of `wild: Yes` ops matches the number of IO-effecting ops.
    #[test]
    fn wild_count_matches_io_op_count() {
        let io_count = MATRIX.iter().filter(|r| r.effects == Effects::Io).count();
        let wild_count = MATRIX.iter().filter(|r| r.wild == Wild::Yes).count();
        assert_eq!(
            io_count, wild_count,
            "every IO op must be wild: Yes and vice versa"
        );
    }
}
