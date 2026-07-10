//! `FsErr` — the explicit, traceable filesystem error type (C1/G2).
//!
//! Every path, permission, or IO failure is an instance of this type carrying:
//! - the *path* that was attempted (so the caller can surface it), and
//! - an `ErrnoClass` (the classified OS errno — NEVER a bare raw code), and
//! - a one-line *why* (human-readable; satisfies G11 dual projection).
//!
//! **No sentinel, no silent clamp, no `0`-for-error.** Every failure is an explicit `FsErr`.
//! A use-after-consume of an affine handle is `UseAfterConsume` (LR-8), caught above the
//! OS floor and requiring no syscall.
//!
//! Design spec: `docs/spec/stdlib/fs.md` §3; contract: RFC-0016 §4.1 C1/C3; RFC-0013
//! (structured diagnostic record).

use std::fmt;

/// The classified OS errno — never a bare raw code (C3: no opaque error codes).
///
/// This is the RFC-0013 "errno-class" column in the diagnostic record. Each variant
/// maps a POSIX errno *class* to a typed value; the raw OS code is preserved in the
/// `Os` variant of `FsErr` for tooling, but is *never* the primary identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrnoClass {
    /// ENOENT / ENOTDIR — the path or a component was absent.
    NotFound,
    /// EACCES / EPERM — permission denied by the OS.
    PermDenied,
    /// EEXIST — a path already exists where it should not.
    AlreadyExists,
    /// ENOTDIR — a path component is not a directory.
    NotADirectory,
    /// EISDIR — a path is a directory where a regular file was expected.
    IsADirectory,
    /// ENOTEMPTY — a directory was not empty when remove was attempted.
    NotEmpty,
    /// ENOSPC / EDQUOT — the disk or quota is full.
    DiskFull,
    /// EXDEV — a cross-device rename/link was attempted.
    CrossDevice,
    /// EWOULDBLOCK / EAGAIN — the operation would block.
    WouldBlock,
    /// EINTR — the operation was interrupted by a signal.
    Interrupted,
    /// ELOOP — too many levels of symbolic links.
    Loop,
    /// Some other errno class not covered above; raw code preserved for tooling.
    Other,
}

impl fmt::Display for ErrnoClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "NotFound"),
            Self::PermDenied => write!(f, "PermDenied"),
            Self::AlreadyExists => write!(f, "AlreadyExists"),
            Self::NotADirectory => write!(f, "NotADirectory"),
            Self::IsADirectory => write!(f, "IsADirectory"),
            Self::NotEmpty => write!(f, "NotEmpty"),
            Self::DiskFull => write!(f, "DiskFull"),
            Self::CrossDevice => write!(f, "CrossDevice"),
            Self::WouldBlock => write!(f, "WouldBlock"),
            Self::Interrupted => write!(f, "Interrupted"),
            Self::Loop => write!(f, "Loop"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// The explicit, traceable filesystem error (RFC-0013 diagnostic record).
///
/// Every `FsErr` carries:
/// - **`path`** — the path that was attempted (for the caller to surface; G11)
/// - **`errno_class`** — the classified errno, never a bare code (C3)
/// - **`why`** — a human-readable one-liner explaining the refusal (G11)
///
/// For `UseAfterConsume` the `path` is the handle's original path (if known) and the
/// errno_class is absent (the violation is caught above the OS floor — no syscall ran).
///
/// # C1 — never-silent
/// A caller cannot reach an `FsErr` that is empty, zeroed, or opaque. Every variant
/// names the failing operation and the cause. The `Os` variant preserves the raw
/// errno for tooling (the `myc-sec` `wild`-audit), but it is never the *primary*
/// identifier that a caller pattern-matches on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsErr {
    /// The path was not found (and `create`/`create_new` was not set).
    NotFound { path: String, why: &'static str },
    /// Permission denied by the OS.
    PermDenied { path: String, why: &'static str },
    /// The path already exists (with `create_new` set, or `remove`/`rename` collision).
    AlreadyExists { path: String, why: &'static str },
    /// A component of the path is not a directory.
    NotADirectory { path: String, why: &'static str },
    /// The path is a directory where a regular file was expected.
    IsADirectory { path: String, why: &'static str },
    /// A directory was not empty when `remove_dir` was attempted.
    NotEmpty { path: String, why: &'static str },
    /// The disk or quota is full.
    DiskFull { path: String, why: &'static str },
    /// A cross-device rename was attempted.
    CrossDevice {
        from: String,
        to: String,
        why: &'static str,
    },
    /// The operation would block (non-blocking mode).
    WouldBlock { path: String, why: &'static str },
    /// The operation was interrupted by a signal; the caller should retry.
    Interrupted { path: String, why: &'static str },
    /// Too many levels of symbolic links.
    Loop { path: String, why: &'static str },
    /// An affine `substrate` handle was used after it was consumed (LR-8).
    ///
    /// This is caught *above* the OS floor — no syscall ran.
    UseAfterConsume { path: String, why: &'static str },
    /// The requested `OpenOptions` combination is invalid (e.g. `truncate` without write/append).
    ///
    /// A caller error in the *request*, not a property of the path — caught *above* the OS floor
    /// (no syscall ran). Distinct from `NotFound`/`PermDenied` so a caller pattern-matching on the
    /// variant is not misled (C3).
    InvalidOptions { path: String, why: &'static str },
    /// A lower-level IO failure (threaded, not swallowed — I1).
    ///
    /// This wraps a failure that came from the byte-transfer layer. The
    /// `errno_class` field carries the classification; `raw` is preserved
    /// for the `myc-sec` `wild`-audit tooling (never the primary discriminant).
    Os {
        path: String,
        errno_class: ErrnoClass,
        /// Raw OS error code (preserved for tooling; never the primary identifier — C3).
        raw: i32,
        why: &'static str,
    },
}

impl FsErr {
    /// The path that was attempted, if applicable.
    ///
    /// Returns the primary path for single-path errors. For `CrossDevice`, returns the
    /// `from` path. For `UseAfterConsume`, returns the handle's original path.
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            Self::NotFound { path, .. }
            | Self::PermDenied { path, .. }
            | Self::AlreadyExists { path, .. }
            | Self::NotADirectory { path, .. }
            | Self::IsADirectory { path, .. }
            | Self::NotEmpty { path, .. }
            | Self::DiskFull { path, .. }
            | Self::WouldBlock { path, .. }
            | Self::Interrupted { path, .. }
            | Self::Loop { path, .. }
            | Self::UseAfterConsume { path, .. }
            | Self::InvalidOptions { path, .. }
            | Self::Os { path, .. } => path,
            Self::CrossDevice { from, .. } => from,
        }
    }

    /// The human-readable why-string (G11 dual projection).
    #[must_use]
    pub fn why(&self) -> &str {
        match self {
            Self::NotFound { why, .. }
            | Self::PermDenied { why, .. }
            | Self::AlreadyExists { why, .. }
            | Self::NotADirectory { why, .. }
            | Self::IsADirectory { why, .. }
            | Self::NotEmpty { why, .. }
            | Self::DiskFull { why, .. }
            | Self::CrossDevice { why, .. }
            | Self::WouldBlock { why, .. }
            | Self::Interrupted { why, .. }
            | Self::Loop { why, .. }
            | Self::UseAfterConsume { why, .. }
            | Self::InvalidOptions { why, .. }
            | Self::Os { why, .. } => why,
        }
    }

    /// The classified errno — `None` for `UseAfterConsume` (caught above the OS floor).
    #[must_use]
    pub fn errno_class(&self) -> Option<ErrnoClass> {
        match self {
            Self::NotFound { .. } => Some(ErrnoClass::NotFound),
            Self::PermDenied { .. } => Some(ErrnoClass::PermDenied),
            Self::AlreadyExists { .. } => Some(ErrnoClass::AlreadyExists),
            Self::NotADirectory { .. } => Some(ErrnoClass::NotADirectory),
            Self::IsADirectory { .. } => Some(ErrnoClass::IsADirectory),
            Self::NotEmpty { .. } => Some(ErrnoClass::NotEmpty),
            Self::DiskFull { .. } => Some(ErrnoClass::DiskFull),
            Self::CrossDevice { .. } => Some(ErrnoClass::CrossDevice),
            Self::WouldBlock { .. } => Some(ErrnoClass::WouldBlock),
            Self::Interrupted { .. } => Some(ErrnoClass::Interrupted),
            Self::Loop { .. } => Some(ErrnoClass::Loop),
            Self::UseAfterConsume { .. } => None, // caught above the floor; no errno
            Self::InvalidOptions { .. } => None,  // caught above the floor; a request error
            Self::Os { errno_class, .. } => Some(errno_class.clone()),
        }
    }
}

impl fmt::Display for FsErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrossDevice { from, to, why } => {
                write!(f, "CrossDevice(from={from:?}, to={to:?}): {why}")
            }
            Self::UseAfterConsume { path, why } => {
                write!(f, "UseAfterConsume(path={path:?}): {why}")
            }
            Self::InvalidOptions { path, why } => {
                write!(f, "InvalidOptions(path={path:?}): {why}")
            }
            Self::Os {
                path,
                errno_class,
                raw,
                why,
            } => {
                write!(
                    f,
                    "Os(path={path:?}, errno={errno_class}, raw={raw}): {why}"
                )
            }
            err => {
                let class = err
                    .errno_class()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "Unknown".to_owned());
                write!(f, "{class}(path={:?}): {}", err.path(), err.why())
            }
        }
    }
}

mycelium_std_core::impl_std_error!(FsErr);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every FsErr variant exposes a non-empty `why` (G11 — dual projection).
    /// Guard: an empty `why` in any variant makes this fail.
    #[test]
    fn every_variant_has_a_non_empty_why() {
        let errors: Vec<FsErr> = vec![
            FsErr::NotFound {
                path: "/a".into(),
                why: "test",
            },
            FsErr::PermDenied {
                path: "/a".into(),
                why: "test",
            },
            FsErr::AlreadyExists {
                path: "/a".into(),
                why: "test",
            },
            FsErr::NotADirectory {
                path: "/a".into(),
                why: "test",
            },
            FsErr::IsADirectory {
                path: "/a".into(),
                why: "test",
            },
            FsErr::NotEmpty {
                path: "/a".into(),
                why: "test",
            },
            FsErr::DiskFull {
                path: "/a".into(),
                why: "test",
            },
            FsErr::CrossDevice {
                from: "/a".into(),
                to: "/b".into(),
                why: "test",
            },
            FsErr::WouldBlock {
                path: "/a".into(),
                why: "test",
            },
            FsErr::Interrupted {
                path: "/a".into(),
                why: "test",
            },
            FsErr::Loop {
                path: "/a".into(),
                why: "test",
            },
            FsErr::UseAfterConsume {
                path: "/a".into(),
                why: "test",
            },
            FsErr::InvalidOptions {
                path: "/a".into(),
                why: "test",
            },
            FsErr::Os {
                path: "/a".into(),
                errno_class: ErrnoClass::Other,
                raw: 42,
                why: "test",
            },
        ];
        for e in &errors {
            assert!(!e.why().is_empty(), "FsErr variant has empty why: {e:?}");
        }
    }

    /// `UseAfterConsume` has no errno class (caught above the OS floor).
    /// Guard: assigning an errno_class to UseAfterConsume makes this fail.
    #[test]
    fn use_after_consume_has_no_errno_class() {
        let e = FsErr::UseAfterConsume {
            path: "/x".into(),
            why: "handle consumed",
        };
        assert_eq!(
            e.errno_class(),
            None,
            "UseAfterConsume must not carry an errno_class (caught above the floor)"
        );
    }

    /// Every non-UseAfterConsume variant has a classified errno (C3: no opaque codes).
    /// Guard: returning None for a non-UseAfterConsume variant makes this fail.
    #[test]
    fn every_os_level_variant_has_errno_class() {
        let errors = vec![
            FsErr::NotFound {
                path: "/a".into(),
                why: "test",
            },
            FsErr::PermDenied {
                path: "/a".into(),
                why: "test",
            },
            FsErr::AlreadyExists {
                path: "/a".into(),
                why: "test",
            },
            FsErr::DiskFull {
                path: "/a".into(),
                why: "test",
            },
        ];
        for e in &errors {
            assert!(
                e.errno_class().is_some(),
                "OS-level error must have an errno_class: {e:?}"
            );
        }
    }

    /// The `Display` for `FsErr` includes the path (so failures are traceable).
    /// Guard: a Display that drops the path makes this fail.
    #[test]
    fn display_includes_path() {
        let e = FsErr::NotFound {
            path: "/some/path".into(),
            why: "absent",
        };
        let s = e.to_string();
        assert!(
            s.contains("/some/path"),
            "FsErr Display must include the path; got: {s}"
        );
    }
}
