//! \[Declared\] Process / environment floor (RFC-0028 §4.5; M-722). Thin, never-silent wrappers over
//! Rust `std::process` and `std::env`.
//!
//! This is the audited syscall floor for `std.sys`'s OS contact: process exit and environment-variable
//! reads. Per LR-9 / RFC-0016 §8-Q6 all such contact lives in this single `std-sys` phylum.
//!
//! # Honesty (VR-5)
//!
//! Every function carries the **`Declared`** guarantee tag — unaudited `std::process` / `std::env`
//! wrappers; no theorem or measured bound backs OS process/env semantics. Promotion requires a
//! checked or measured basis (none in v0).
//!
//! # Never-silent (G2)
//!
//! `get_env` returns an explicit `Option`: a missing or non-Unicode variable is `None`, never an
//! empty-string stand-in. `exit` does not return (it terminates the process), so its "failure mode"
//! is structural — the explicit, caller-chosen status code is the contract. `args` returns an
//! explicit `Result`: a non-UTF-8 argument is an `Err` naming the offending index, **never** a
//! silently-dropped element that would shift the positions of every following arg (G2).
//!
//! # Guarantee matrix (RFC-0016 §4.5)
//!
//! | op | signature | failure mode | tag |
//! |----|-----------|--------------|-----|
//! | `exit` | `(i32) -> !` | n/a (terminates) | `Declared` |
//! | `get_env` | `(&str) -> Option<String>` | missing/non-Unicode → `None` (never-silent) | `Declared` |
//! | `args` | `() -> Result<Vec<String>, NonUtf8Arg>` | non-UTF-8 arg → `Err` (never-silent) | `Declared` |

/// \[Declared\] Terminate the process with `code`. Does not return. The exit status is the caller's
/// explicit choice — never a silent `0` substituted for an error path (G2): a program that wants to
/// signal failure passes a non-zero `code`.
pub fn exit(code: i32) -> ! {
    std::process::exit(code)
}

/// \[Declared\] Read environment variable `name`. Returns `None` if the variable is unset **or** is
/// not valid Unicode — an explicit absence, never an empty-string stand-in (G2). Use when a missing
/// variable is a recoverable condition; the `None` must be handled, not assumed present.
#[must_use]
pub fn get_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// \[Declared\] A command-line argument that was not valid UTF-8, reported by its **position**
/// (`index` — arg 0 is the program name) so the offending input is named, never silently dropped
/// (G2). Carries the lossy [`String::from_utf8_lossy`] rendering (`lossy`) so an `EXPLAIN`/diagnostic
/// can *show* the bad arg without fabricating a faithful round-trip it cannot deliver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonUtf8Arg {
    /// 0-based position of the offending argument (0 = program name).
    pub index: usize,
    /// `String::from_utf8_lossy` rendering of the raw argument (U+FFFD for each invalid byte).
    pub lossy: String,
}

impl std::fmt::Display for NonUtf8Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "command-line argument at index {} is not valid UTF-8 (lossy: {:?})",
            self.index, self.lossy
        )
    }
}

impl std::error::Error for NonUtf8Arg {}

/// \[Declared\] The process's command-line arguments (including arg 0), parsed to `String`s. Built on
/// `args_os`: a non-UTF-8 argument is an explicit `Err(NonUtf8Arg)` naming the offending **index** —
/// never a silently-dropped element (the previous `filter_map` behaviour) that would shift every
/// following arg's position, and (unlike `std::env::args()`) never a *panic* (G2). The first
/// non-UTF-8 arg short-circuits; callers needing the faithful, position-stable raw form should use
/// `std::env::args_os` directly (this floor is the validated-UTF-8 convenience).
pub fn args() -> Result<Vec<String>, NonUtf8Arg> {
    parse_args(std::env::args_os())
}

/// Core parse step factored over an iterator of [`OsString`](std::ffi::OsString) so it is
/// unit-testable without real `argv`. Maps each arg through `into_string`; the first non-UTF-8 arg
/// becomes an `Err(NonUtf8Arg)` naming its index (never-silent, G2) rather than being dropped.
pub(crate) fn parse_args(
    raw: impl IntoIterator<Item = std::ffi::OsString>,
) -> Result<Vec<String>, NonUtf8Arg> {
    raw.into_iter()
        .enumerate()
        .map(|(index, a)| {
            a.into_string().map_err(|bad| NonUtf8Arg {
                index,
                lossy: bad.to_string_lossy().into_owned(),
            })
        })
        .collect()
}
