//! `mycelium-cli-common` (M-643) — the small, **dependency-free** substrate the four toolchain CLIs
//! (`mycfmt` / `myc-check` / `myc-lint` / `myc-sec`) share.
//!
//! The bins had drifted into three near-identical duplications:
//! * a stdin-or-file `read_input` (`mycfmt.rs:144` ≈ `myc-check.rs:177`, char-for-char modulo the
//!   exit-code newtype),
//! * the `.myc` source-walking (`myc-lint.rs:173` `walk`/`walk_into`, `myc-sec`'s `collect_myc`), and
//! * a hand-rolled `env::args().skip(1)` option loop whose every value-flag repeats the
//!   `match args.next() { Some(v) => …, None => return usage() }` idiom.
//!
//! This crate folds those three out into [`read_source`], [`walk_myc`], and [`Args`]. The contract is
//! **behaviour-preservation**: each helper reproduces the bins' observable behaviour byte-for-byte
//! (same files read, same exit-code mapping at the call site, same stderr text — the tool-name prefix
//! is a parameter, never hard-coded). It is **never-silent** (G2): a missing/unreadable input is a
//! reported, structured outcome the caller maps to its existing exit code, never a hidden empty read.
//!
//! ## Honesty / guarantees
//! * `read_source` — **Exact**: a thin, total wrapper over `std::fs`/`std::io`; it cannot silently
//!   succeed on a missing file (the `Err` is surfaced) nor on ambiguous input (`-` is the *only* stdin
//!   sentinel; every other string is a path, exactly as the bins did).
//! * `walk_myc` — **Exact** over the directory it is given: deterministic (sorted), recursive, and it
//!   skips dotfiles and `target/` exactly as the prior `walk`/`collect_myc` did. It does not follow
//!   symlinks beyond what `std::fs::read_dir` + `is_dir()` already did (unchanged from the originals).
//! * `Args` — **Exact**: a transparent cursor over `std::env::args().skip(1)`; it changes *no* parsing
//!   behaviour, it only names the value-flag idiom so the bins stop repeating it.
//!
//! No new external dependency (KC-3): `std` only.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

/// The conventional sentinel that means "read standard input" across the toolchain CLIs.
pub const STDIN_SENTINEL: &str = "-";

/// The error from [`read_source`]: the diagnostic has **already been emitted** to stderr (never-silent,
/// G2); it carries no payload because there is nothing more to say — the caller's only job is to map it
/// to *its* exit code (the bins use `66` / `EX_IOERR`). A distinct (non-`()`) type keeps the contract
/// legible at the call site and satisfies the no-bare-`Result<_, ()>` lint without inventing data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadError;

/// Read one source: standard input when `path == "-"`, otherwise the file at `path`.
///
/// **Never-silent (G2).** On any failure this prints a diagnostic to **stderr** — using `prefix` as
/// the tool tag so the message is byte-identical to what the bin emitted by hand — and returns
/// [`Err`]`(())`. The caller maps that to its own exit code (the bins use `66` / `EX_IOERR`); this
/// keeps the *exit-code newtype* (`u8` vs `ExitCode`) at the call site, where it belongs, while the
/// *I/O and the message* live here once.
///
/// `prefix` is the leading tool tag exactly as the bin wrote it, e.g. `"mycfmt: io-error"` for
/// `mycfmt`, `"io-error"` (no tool name) for `myc-check`, `"myc-lint: io-error"` for `myc-lint`.
/// The emitted lines are then, verbatim:
/// * stdin failure: `"{prefix}: could not read stdin"`
/// * file failure:  `"{prefix}: {path}: {e}"`
///
/// matching the originals char-for-char.
pub fn read_source(prefix: &str, path: &str) -> Result<String, ReadError> {
    use std::io::Read;
    if path == STDIN_SENTINEL {
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("{prefix}: could not read stdin");
            return Err(ReadError);
        }
        Ok(s)
    } else {
        std::fs::read_to_string(path).map_err(|e| {
            eprintln!("{prefix}: {path}: {e}");
            ReadError
        })
    }
}

/// Collect every `.myc` file under `dir`, recursively, **sorted**.
///
/// Deterministic and total over a readable tree: hidden entries (names starting with `.`) and any
/// `target/` directory are skipped, exactly as the prior `myc-lint` `walk`/`walk_into` and
/// `myc-sec::collect_myc` did. A directory that cannot be read is an explicit [`Err`] carrying the
/// `"{path}: {e}"` message the originals produced — **never** a silently-empty walk (G2).
///
/// The returned paths are *relative to / rooted at* `dir` in the same way `read_dir` yields them
/// (the caller does any `strip_prefix` display work, unchanged).
pub fn walk_myc(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    walk_into(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_into(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if path.is_dir() {
            walk_into(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "myc") {
            out.push(path);
        }
    }
    Ok(())
}

/// A transparent cursor over `std::env::args().skip(1)` that names the one repeated arg-parsing idiom:
/// "this flag takes a value — consume the next token, or it's a usage error".
///
/// It deliberately does **no** flag interpretation of its own; the bin still owns its `match` over
/// option names (their option sets differ). This only removes the duplicated
/// `match args.next() { Some(v) => …, None => return usage() }` boilerplate, behaviour-identically.
///
/// ```
/// # use mycelium_cli_common::Args;
/// // The bin's loop keeps full control of which flags exist and what they mean:
/// let mut args = Args::from_env();
/// while let Some(tok) = args.next() {
///     match tok.as_str() {
///         "--config" => match args.value() {
///             Some(v) => { /* use v */ let _ = v; }
///             None => { /* return usage() */ }
///         },
///         _ => { /* ... */ }
///     }
/// }
/// ```
pub struct Args {
    inner: std::vec::IntoIter<String>,
}

impl Args {
    /// Cursor over the process arguments, skipping `argv[0]` (the program name) — exactly the bins'
    /// `std::env::args().skip(1)`.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_args(std::env::args().skip(1))
    }

    /// Cursor over an explicit argument list (skips nothing — pass the post-`argv[0]` tokens). Used by
    /// the unit tests and available for callers that already hold a token vector.
    #[must_use]
    pub fn from_args<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let v: Vec<String> = args.into_iter().map(Into::into).collect();
        Self {
            inner: v.into_iter(),
        }
    }

    /// The next raw token, or `None` at the end.
    #[must_use]
    #[allow(clippy::should_implement_trait)] // a named `next` mirrors the bins' `args.next()` call sites
    pub fn next(&mut self) -> Option<String> {
        self.inner.next()
    }

    /// The **value** for the just-seen value-flag: the next token, or `None` if the flag was last.
    /// `None` is the caller's cue to `return usage()` — byte-identical to the hand-rolled
    /// `match args.next() { Some(v) => …, None => return usage() }`.
    #[must_use]
    pub fn value(&mut self) -> Option<String> {
        self.inner.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // --- read_source ---------------------------------------------------------

    #[test]
    fn read_source_reads_a_file() {
        let dir = tmpdir("read-file");
        let p = dir.join("a.myc");
        fs::write(&p, "nodule x\n").unwrap();
        let got = read_source("mycfmt: io-error", p.to_str().unwrap());
        assert_eq!(got, Ok("nodule x\n".to_owned()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_source_missing_file_is_err_not_silent() {
        // G2: a missing path is a reported Err, never an empty success.
        let got = read_source("mycfmt: io-error", "definitely/not/here.myc");
        assert_eq!(got, Err(ReadError));
    }

    #[test]
    fn stdin_sentinel_is_exactly_dash() {
        // Only "-" is stdin; every other string is a path (here, a missing one ⇒ Err, not stdin).
        assert_eq!(STDIN_SENTINEL, "-");
        assert_eq!(read_source("io-error", "-stdin-look-alike"), Err(ReadError));
    }

    // --- walk_myc ------------------------------------------------------------

    #[test]
    fn walk_myc_is_sorted_recursive_and_filters() {
        let dir = tmpdir("walk");
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::create_dir_all(dir.join("target")).unwrap();
        fs::create_dir_all(dir.join(".hidden")).unwrap();
        fs::write(dir.join("b.myc"), "").unwrap();
        fs::write(dir.join("a.myc"), "").unwrap();
        fs::write(dir.join("note.txt"), "").unwrap(); // wrong extension — skipped
        fs::write(dir.join("sub/c.myc"), "").unwrap();
        fs::write(dir.join("target/skip.myc"), "").unwrap(); // under target/ — skipped
        fs::write(dir.join(".hidden/skip.myc"), "").unwrap(); // hidden dir — skipped

        let got = walk_myc(&dir).unwrap();
        let rel: Vec<String> = got
            .iter()
            .map(|p| {
                p.strip_prefix(&dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(rel, vec!["a.myc", "b.myc", "sub/c.myc"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn walk_myc_unreadable_dir_is_err_not_silent() {
        // G2: a directory we cannot read is an explicit Err, never an empty list.
        let got = walk_myc(Path::new("definitely/not/a/dir"));
        assert!(got.is_err());
    }

    #[test]
    fn walk_myc_empty_tree_is_ok_empty() {
        let dir = tmpdir("walk-empty");
        let got = walk_myc(&dir).unwrap();
        assert!(got.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    // --- Args ----------------------------------------------------------------

    #[test]
    fn args_value_consumes_next_token() {
        let mut a = Args::from_args(["--config", "proj.toml", "file.myc"]);
        assert_eq!(a.next().as_deref(), Some("--config"));
        assert_eq!(a.value().as_deref(), Some("proj.toml"));
        assert_eq!(a.next().as_deref(), Some("file.myc"));
        assert_eq!(a.next(), None);
    }

    #[test]
    fn args_value_missing_is_none_for_usage() {
        // A trailing value-flag ⇒ value() is None ⇒ the caller's `return usage()` cue.
        let mut a = Args::from_args(["--config"]);
        assert_eq!(a.next().as_deref(), Some("--config"));
        assert_eq!(a.value(), None);
    }

    // --- helpers -------------------------------------------------------------

    fn tmpdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("mycelium-cli-common-{tag}-{nonce}"));
        fs::create_dir_all(&p).unwrap();
        p
    }
}
