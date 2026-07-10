//! `std.fs` — filesystem over affine `substrate` handles (M-528).
//!
//! Ring 2 / Tier B. The filesystem surface — open a path to an affine handle, read/write/append
//! bytes through it, `stat` metadata, `list` a directory, and `create`/`remove`/`rename` entries
//! — layered on the `io` byte-stream surface (M-514, FLAGGED seam below).
//!
//! # Honesty crux (C1/G2)
//! Every path, permission, or IO failure is an explicit `Result::Err` carrying an RFC-0013
//! diagnostic record that names *the path* and the *errno-class cause* — there is:
//! - **no silent create-on-write**: opening for write without `create` is `Err(NotFound)`,
//! - **no silent truncation**: an existing file is not zeroed unless the caller declared `truncate`,
//! - **no silent partial write**: a short write is surfaced as the actual count, never swallowed,
//! - **no silent overwrite**: `copy`/`rename` to an existing target without declared intent is an error.
//!
//! A `File` handle is **affine** (LR-8): consumed exactly once; use-after-consume is
//! `Err(UseAfterConsume)`, never undefined behaviour.
//!
//! # Guarantee tag
//! Every exported op tags `Exact` — `fs` carries no accuracy/approximation semantics (VR-5:
//! `Exact` is the honest floor, not an overclaim). The honesty is borne by C1 (fallibility)
//! and C6 (declared effects), not by a C2 precision tag.
//!
//! # In-memory substrate
//! The real OS syscall floor (`open`/`read`/`write`/`stat`/…) is deferred to `std-sys` (M-541 —
//! **FLAG Q1**). This crate implements the honest surface over a fully-testable `InMemoryFs`
//! substrate. The `FsBackend` trait is the seam; `RealFs` wires in when M-541 lands.
//!
//! # FLAGs (see spec §7)
//! - **Q1** — Real-OS syscall floor (`RealFs`) deferred to `std-sys` (M-541 / RFC-0016 §8-Q6).
//! - **Q2** — The `io` seam: local `ByteSink`/`ByteSource` stand-ins replace M-514 `Read`/`Write`.
//!   Thread M-514's surface when it lands; do NOT pre-commit io's signatures from `fs`.
//! - **Q3** — Path model + portability: `Path` is currently a UTF-8 newtype (conservative). Non-UTF-8
//!   OS paths are an open question (spec §7-Q3 / RFC-0016 §8-Q3); a lossy decode would violate C1.
//! - **Q4** — Atomicity, symlinks, and `walk_dir`: cross-platform `rename`/`copy` atomicity,
//!   symlink follow-vs-nofollow as declared intent, and recursive `walk_dir` (with a traversal
//!   budget) are unsettled (spec §7-Q4). In-memory substrate always follows no symlinks (flat map).
//! - **Q5** — Capability-scoped filesystem effect: whether the `io` effect is ambient or a scoped
//!   capability (WASI-style preopens) is deferred (spec §7-Q5 / RFC-0016 §8-Q3/Q6).
//!
//! # Design spec
//! `docs/spec/stdlib/fs.md` (M-528, #169). Contract: RFC-0016 §4.1 (C1–C6). Guarantee matrix:
//! spec §4 / [`guarantee_matrix::MATRIX`].
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** FS ops are representation-opaque at the byte level —
//! filesystem handles move raw bytes without interpreting any `Repr`. Encoding is always the
//! caller's responsibility; no silent re-encoding occurs at the FS layer. The `Path` type is a
//! UTF-8 newtype (conservative — see FLAG Q3); no non-UTF-8 path is silently coerced (C1).
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/fs.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod error;
pub mod guarantee_matrix;
pub mod metadata;
pub mod options;
pub mod path;
pub(crate) mod substrate;

// ─── Public re-exports ───────────────────────────────────────────────────────

pub use error::{ErrnoClass, FsErr};
pub use guarantee_matrix::{Effects, Explainable, Fallibility, MatrixRow, Wild, MATRIX};
pub use metadata::{FileKind, Metadata, Permissions};
pub use options::OpenOptions;
pub use path::Path;

// ─── File handle — the affine substrate handle (LR-8) ───────────────────────

use substrate::{InMemHandle, InMemoryFs};

/// An affine open-file handle (LR-8: consumed exactly once).
///
/// Obtained from [`Fs::open`]. Must be explicitly closed with [`Fs::close`]; any operation on a
/// consumed handle returns `Err(FsErr::UseAfterConsume)`. This structural constraint closes the
/// only resource-leak vector (LR-9): an unclosed handle is a leak; close errors are surfaced (C1).
///
/// # FLAG (LR-8 / Rust linear types)
/// Rust does not have native linear types, so the single-consumption invariant is enforced at
/// runtime via a `consumed` flag. A future Mycelium-lang implementation can enforce it at
/// compile time (RFC-0006 LR-8). Until then, every op checks `consumed` first.
#[derive(Debug)]
pub struct File {
    handle: InMemHandle,
}

impl File {
    /// The original path this handle was opened for (for diagnostics).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.handle.path
    }

    /// Whether this handle has been consumed.
    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.handle.consumed
    }
}

/// An open directory iterator handle.
///
/// Produced by [`Fs::read_dir`]. Yields child paths as `String`s. This is an in-memory
/// snapshot iterator; in the real-OS path it would iterate `readdir(3)` entries.
#[derive(Debug)]
pub struct DirIter {
    entries: std::vec::IntoIter<String>,
}

impl Iterator for DirIter {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next()
    }
}

// ─── The `Fs` context (the effectful operations are methods on this) ─────────

/// The filesystem context: holds the substrate and exposes all effectful fs ops.
///
/// All effectful operations (`open`, `stat`, `read_dir`, `create_dir`, `remove_file`,
/// `remove_dir`, `rename`, `copy`) are methods on `Fs`. This makes the filesystem effect
/// explicit and inspectable: a function that does not hold an `&mut Fs` cannot perform
/// filesystem IO (partial C6 capability scoping — see FLAG Q5).
///
/// Currently backed by `InMemoryFs`. Wire `RealFs` (M-541) when the `std-sys` phylum lands.
pub struct Fs {
    backend: InMemoryFs,
}

impl Fs {
    /// Create a new `Fs` over a fresh in-memory substrate.
    ///
    /// # Effects: none (pure construction)
    /// The in-memory substrate is fully testable without OS interaction.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            backend: InMemoryFs::new(),
        }
    }

    /// Create a new `Fs` with a simulated disk limit (for testing `DiskFull` paths).
    #[must_use]
    pub fn in_memory_with_limit(bytes: u64) -> Self {
        Self {
            backend: InMemoryFs::new().with_disk_limit(bytes),
        }
    }

    // ─── Path ops (pure, total, no IO) ───────────────────────────────────────

    /// Construct a `Path` from a UTF-8 string.
    ///
    /// # Guarantee: `Exact`, total, effects: none
    #[must_use]
    pub fn path(&self, s: impl Into<String>) -> Path {
        Path::new(s)
    }

    // ─── Effectful ops (effects: io — declared on each) ──────────────────────

    /// Check whether a path exists.
    ///
    /// # Guarantee: `Exact`, `Result<bool, FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `stat`/`access` syscall; in-memory: map lookup)
    ///
    /// Returns `Err(PermDenied)` when the OS denies access to the path. **Never** returns `Ok(false)`
    /// as a substitute for a permission failure (C1: no silent clamp).
    ///
    /// # FLAG (Q5): the `io` effect is currently ambient on this `&mut Fs`; the capability-scoped
    /// design (WASI-style) is deferred (spec §7-Q5).
    pub fn exists(&self, path: &Path) -> Result<bool, FsErr> {
        self.backend.exists(path.as_str())
    }

    /// Get filesystem metadata for a path.
    ///
    /// # Guarantee: `Exact`, `Result<Metadata, FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `stat`/`fstat` syscall)
    /// # Errors: `NotFound` | `PermDenied`
    pub fn stat(&self, path: &Path) -> Result<Metadata, FsErr> {
        self.backend.stat(path.as_str())
    }

    /// Open a path to an affine `File` handle under an explicit `OpenOptions`.
    ///
    /// # Guarantee: `Exact`, `Result<File, FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `open`/`openat` syscall)
    /// # Errors
    /// - `NotFound` — path absent and `create`/`create_new` not set (**no silent create**, C1)
    /// - `AlreadyExists` — `create_new` set and path exists (**no silent overwrite**, C1)
    /// - `PermDenied` — OS access denied
    /// - `IsADirectory` — path is a directory; open a file with `read_dir` for directories
    /// - `NotFound` — parent directory does not exist (create/create_new path)
    pub fn open(&mut self, path: &Path, opts: &OpenOptions) -> Result<File, FsErr> {
        let handle = self.backend.open(path.as_str(), opts)?;
        Ok(File { handle })
    }

    /// Read bytes from an open `File` handle into `buf`. Returns the bytes read.
    ///
    /// A short read (fewer than `buf.len()` bytes) is returned explicitly — not swallowed (C1).
    /// `0` means EOF.
    ///
    /// # Guarantee: `Exact`, `Result<usize, FsErr>`
    /// # Effects: io (io M-514 seam — FLAG Q2)
    /// # `wild`?: yes (real-OS: `read` syscall)
    /// # Errors: `UseAfterConsume` | `PermDenied` | `NotFound` | `IsADirectory`
    pub fn read(&self, file: &mut File, buf: &mut [u8]) -> Result<usize, FsErr> {
        self.backend.read(&mut file.handle, buf)
    }

    /// Write bytes to an open `File` handle. Returns the bytes written.
    ///
    /// A short write (fewer than `buf.len()` bytes) is returned explicitly — never swallowed (C1).
    ///
    /// # Guarantee: `Exact`, `Result<usize, FsErr>`
    /// # Effects: io (io M-514 seam — FLAG Q2)
    /// # `wild`?: yes (real-OS: `write` syscall)
    /// # Errors: `UseAfterConsume` | `PermDenied` | `DiskFull` | `NotFound`
    pub fn write(&mut self, file: &mut File, buf: &[u8]) -> Result<usize, FsErr> {
        self.backend.write(&mut file.handle, buf)
    }

    /// Flush deferred write state for a `File` handle.
    ///
    /// In the real-OS path this calls `fsync`. In the in-memory substrate it is a consistency
    /// check (still surfaces `UseAfterConsume` if the handle is consumed).
    ///
    /// # Guarantee: `Exact`, `Result<(), FsErr>`
    /// # Effects: io (io M-514 seam — FLAG Q2)
    /// # `wild`?: yes (real-OS: `fsync`/`write` syscall)
    /// # Errors: `UseAfterConsume` | `Io` | `DiskFull`
    pub fn flush(&self, file: &File) -> Result<(), FsErr> {
        self.backend.flush(&file.handle)
    }

    /// Close (consume) a `File` handle. **This consumes the handle (LR-8).**
    ///
    /// After `close`, any subsequent use of `file` returns `Err(UseAfterConsume)`. A close error
    /// is surfaced, not dropped (C1).
    ///
    /// # Guarantee: `Exact`, `Result<(), FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `close` syscall)
    /// # Errors: `UseAfterConsume` | `Io` | `DiskFull`
    pub fn close(&self, file: &mut File) -> Result<(), FsErr> {
        self.backend.close(&mut file.handle)
    }

    /// List the entries in a directory.
    ///
    /// Returns a `DirIter` that yields child path strings. The iterator is a snapshot — it is not
    /// backed by a live `readdir` stream (in the in-memory substrate; the real-OS seam will differ
    /// once M-541 lands).
    ///
    /// # Guarantee: `Exact`, `Result<DirIter, FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `opendir`/`readdir` syscall)
    /// # Errors: `NotFound` | `NotADirectory` | `PermDenied`
    pub fn read_dir(&self, path: &Path) -> Result<DirIter, FsErr> {
        let entries = self.backend.read_dir(path.as_str())?;
        Ok(DirIter {
            entries: entries.into_iter(),
        })
    }

    /// Create a directory at `path`.
    ///
    /// # Guarantee: `Exact`, `Result<(), FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `mkdir` syscall)
    /// # Errors: `AlreadyExists` | `PermDenied` | `NotFound` (parent absent)
    pub fn create_dir(&mut self, path: &Path) -> Result<(), FsErr> {
        self.backend.create_dir(path.as_str())
    }

    /// Remove a regular file at `path`.
    ///
    /// # Guarantee: `Exact`, `Result<(), FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `unlink` syscall)
    /// # Errors: `NotFound` | `PermDenied` | `IsADirectory` (use `remove_dir` for directories)
    pub fn remove_file(&mut self, path: &Path) -> Result<(), FsErr> {
        self.backend.remove_file(path.as_str())
    }

    /// Remove an **empty** directory at `path`.
    ///
    /// # Guarantee: `Exact`, `Result<(), FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `rmdir` syscall)
    /// # Errors: `NotEmpty` | `NotFound` | `PermDenied`
    pub fn remove_dir(&mut self, path: &Path) -> Result<(), FsErr> {
        self.backend.remove_dir(path.as_str())
    }

    /// Rename / move `from` to `to`.
    ///
    /// **No silent overwrite**: if `to` already exists, returns `Err(AlreadyExists)` (C1).
    ///
    /// # Guarantee: `Exact`, `Result<(), FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `rename`/`renameat` syscall)
    /// # Errors: `NotFound` | `PermDenied` | `CrossDevice` | `AlreadyExists`
    /// # FLAG (Q4): cross-device rename atomicity is unsettled; in-memory ignores it.
    pub fn rename(&mut self, from: &Path, to: &Path) -> Result<(), FsErr> {
        self.backend.rename(from.as_str(), to.as_str())
    }

    /// Copy `from` to `to`. Returns the number of bytes copied.
    ///
    /// **No silent overwrite**: if `to` already exists, returns `Err(AlreadyExists)` (C1).
    ///
    /// # Guarantee: `Exact`, `Result<u64, FsErr>`
    /// # Effects: io
    /// # `wild`?: yes (real-OS: `open`+`read`+`write` syscalls)
    /// # Errors: `NotFound` | `PermDenied` | `DiskFull` | `AlreadyExists`
    pub fn copy(&mut self, from: &Path, to: &Path) -> Result<u64, FsErr> {
        self.backend.copy(from.as_str(), to.as_str())
    }
}

// ─── Integration tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fs() -> Fs {
        Fs::in_memory()
    }

    // ─── Open / read / write / close lifecycle ────────────────────────────────

    /// Full create-write-close-read round-trip over the public API.
    #[test]
    fn create_write_read_round_trip() {
        let mut fs = fs();
        let p = Path::new("/hello.txt");

        // Write.
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut file = fs.open(&p, &opts_w).expect("open for write");
        let n = fs.write(&mut file, b"hello world").expect("write");
        assert_eq!(n, 11);
        fs.close(&mut file).expect("close");

        // Read.
        let opts_r = OpenOptions::read_only();
        let mut rfile = fs.open(&p, &opts_r).expect("open for read");
        let mut buf = vec![0u8; 16];
        let rn = fs.read(&mut rfile, &mut buf).expect("read");
        assert_eq!(rn, 11);
        assert_eq!(&buf[..11], b"hello world");
        fs.close(&mut rfile).expect("close");
    }

    /// `stat` returns correct metadata after creating a file.
    #[test]
    fn stat_returns_correct_metadata() {
        let mut fs = fs();
        let p = Path::new("/meta.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_w).expect("open");
        fs.write(&mut h, b"abc").expect("write");
        fs.close(&mut h).expect("close");

        let meta = fs.stat(&p).expect("stat");
        assert!(meta.is_file());
        assert_eq!(meta.size, 3);
    }

    /// `exists` is `true` after creation, `false` before.
    #[test]
    fn exists_reflects_creation_and_removal() {
        let mut fs = fs();
        let p = Path::new("/exist_test.txt");
        assert!(!fs.exists(&p).expect("exists before create"));

        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_w).expect("open");
        fs.close(&mut h).expect("close");
        assert!(fs.exists(&p).expect("exists after create"));

        fs.remove_file(&p).expect("remove");
        assert!(!fs.exists(&p).expect("exists after remove"));
    }

    // ─── C1 — never-silent property tests ─────────────────────────────────────

    /// Opening an absent file without create returns `NotFound` (no silent create, C1).
    /// Guard: returning Ok or a default handle makes this fail.
    #[test]
    fn open_absent_without_create_returns_not_found() {
        let mut fs = fs();
        let p = Path::new("/no-such.txt");
        let err = fs
            .open(&p, &OpenOptions::read_only())
            .expect_err("must fail");
        assert!(
            matches!(err, FsErr::NotFound { .. }),
            "expected NotFound; got {err:?}"
        );
    }

    /// `create_new` on an existing file returns `AlreadyExists` (no silent overwrite, C1).
    #[test]
    fn create_new_on_existing_returns_already_exists() {
        let mut fs = fs();
        let p = Path::new("/dup.txt");
        let opts_c = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_c).expect("create");
        fs.close(&mut h).expect("close");

        let opts_cn = OpenOptions::new().with_write(true).with_create_new(true);
        let err = fs.open(&p, &opts_cn).expect_err("must fail");
        assert!(
            matches!(err, FsErr::AlreadyExists { .. }),
            "expected AlreadyExists; got {err:?}"
        );
    }

    /// Use-after-consume returns `UseAfterConsume` (LR-8 — never UB).
    /// Guard: returning Ok or panicking on a consumed handle makes this fail.
    #[test]
    fn use_after_consume_is_explicit_error() {
        let mut fs = fs();
        let p = Path::new("/consumed.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_w).expect("open");
        fs.close(&mut h).expect("close");

        // Any subsequent op must return UseAfterConsume.
        let err = fs.write(&mut h, b"data").expect_err("must fail");
        assert!(
            matches!(err, FsErr::UseAfterConsume { .. }),
            "use after consume must return UseAfterConsume (LR-8); got {err:?}"
        );
    }

    /// Rename to an existing target returns `AlreadyExists` (no silent overwrite, C1).
    #[test]
    fn rename_no_silent_overwrite() {
        let mut fs = fs();
        let opts_c = OpenOptions::new().with_write(true).with_create(true);
        let a = Path::new("/a.txt");
        let b = Path::new("/b.txt");
        let mut ha = fs.open(&a, &opts_c).expect("create a");
        fs.close(&mut ha).expect("close");
        let mut hb = fs.open(&b, &opts_c.clone()).expect("create b");
        fs.close(&mut hb).expect("close");

        let err = fs.rename(&a, &b).expect_err("must fail");
        assert!(
            matches!(err, FsErr::AlreadyExists { .. }),
            "rename to existing must return AlreadyExists; got {err:?}"
        );
    }

    /// Copy to an existing target returns `AlreadyExists` (no silent overwrite, C1).
    #[test]
    fn copy_no_silent_overwrite() {
        let mut fs = fs();
        let opts_c = OpenOptions::new().with_write(true).with_create(true);
        let a = Path::new("/src.txt");
        let b = Path::new("/dst.txt");
        let mut ha = fs.open(&a, &opts_c).expect("create src");
        fs.close(&mut ha).expect("close");
        let mut hb = fs.open(&b, &opts_c.clone()).expect("create dst");
        fs.close(&mut hb).expect("close");

        let err = fs.copy(&a, &b).expect_err("must fail");
        assert!(
            matches!(err, FsErr::AlreadyExists { .. }),
            "copy to existing must return AlreadyExists; got {err:?}"
        );
    }

    /// Truncate without write is rejected at validation time (no silent data loss).
    #[test]
    fn truncate_without_write_is_rejected() {
        let opts = OpenOptions::new().with_truncate(true);
        assert!(
            opts.validate().is_err(),
            "truncate without write must be invalid (C1 — no silent truncation)"
        );
    }

    /// DiskFull is an explicit error (no silent partial write, C1).
    #[test]
    fn disk_full_is_explicit_error() {
        let mut fs = Fs::in_memory_with_limit(3);
        let p = Path::new("/big.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_w).expect("open");
        let err = fs.write(&mut h, b"hello").expect_err("must fail");
        assert!(
            matches!(err, FsErr::DiskFull { .. }),
            "disk full must return DiskFull; got {err:?}"
        );
        fs.close(&mut h).ok();
    }

    // ─── Directory ops ────────────────────────────────────────────────────────

    /// Creating and listing a directory works end-to-end.
    #[test]
    fn create_dir_and_list() {
        let mut fs = fs();
        let dir = Path::new("/mydir");
        fs.create_dir(&dir).expect("mkdir");

        let p = Path::new("/mydir/f.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_w).expect("create file in dir");
        fs.close(&mut h).expect("close");

        let entries: Vec<String> = fs.read_dir(&dir).expect("read_dir").collect();
        assert!(
            entries.contains(&"/mydir/f.txt".to_owned()),
            "directory must list the created file"
        );
    }

    /// `remove_dir` on a non-empty directory returns `NotEmpty` (C1).
    #[test]
    fn remove_dir_nonempty_is_error() {
        let mut fs = fs();
        let dir = Path::new("/full-dir");
        fs.create_dir(&dir).expect("mkdir");
        let child = Path::new("/full-dir/child.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&child, &opts_w).expect("create child");
        fs.close(&mut h).expect("close");

        let err = fs.remove_dir(&dir).expect_err("must fail");
        assert!(
            matches!(err, FsErr::NotEmpty { .. }),
            "expected NotEmpty; got {err:?}"
        );
    }

    /// `read_dir` on a file returns `NotADirectory` (C1).
    #[test]
    fn read_dir_on_file_is_error() {
        let mut fs = fs();
        let p = Path::new("/just-a-file.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&p, &opts_w).expect("create");
        fs.close(&mut h).expect("close");

        let err = fs.read_dir(&p).expect_err("must fail");
        assert!(
            matches!(err, FsErr::NotADirectory { .. }),
            "expected NotADirectory; got {err:?}"
        );
    }

    // ─── Path ops (pure, total) ───────────────────────────────────────────────

    /// Path join and parent are value-semantic and deterministic (Exact, total).
    #[test]
    fn path_ops_are_deterministic() {
        let base = Path::new("/root");
        for _ in 0..3 {
            assert_eq!(
                base.join("child"),
                base.join("child"),
                "join must be deterministic"
            );
        }
        assert_eq!(base.join("child").parent(), Some(base.clone()));
    }

    // ─── Property: short-read is explicit, not swallowed ─────────────────────

    /// A short read (less than buf.len()) returns the actual count — never silently zero-padded.
    /// Guard: zero-padding the buffer and returning buf.len() makes this fail.
    #[test]
    fn short_read_is_explicit() {
        let mut fs = fs();
        let p = Path::new("/short.txt");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut hw = fs.open(&p, &opts_w).expect("create");
        fs.write(&mut hw, b"hi").expect("write 2 bytes");
        fs.close(&mut hw).expect("close");

        let opts_r = OpenOptions::read_only();
        let mut hr = fs.open(&p, &opts_r).expect("open for read");
        let mut buf = vec![0u8; 16]; // larger than the file
        let n = fs.read(&mut hr, &mut buf).expect("read");
        assert_eq!(n, 2, "short read must return 2, not 16 (the buffer size)");
        assert_eq!(&buf[..2], b"hi");
        fs.close(&mut hr).expect("close");
    }

    // ─── Guarantee matrix assertions ─────────────────────────────────────────

    /// The guarantee matrix covers all spec §3 ops.
    #[test]
    fn guarantee_matrix_is_complete() {
        assert_eq!(MATRIX.len(), 16, "spec §3 lists 16 rows; matrix must match");
    }

    /// Every row in the matrix is Exact (VR-5 — fs has no accuracy semantics).
    #[test]
    fn guarantee_matrix_all_rows_exact() {
        for row in MATRIX {
            assert_eq!(row.guarantee, "Exact", "op {:?} must be Exact", row.op);
        }
    }

    /// Every fallible IO row carries a non-empty error_set (C1 — never-silent contract).
    #[test]
    fn guarantee_matrix_fallible_rows_have_error_sets() {
        for row in MATRIX {
            if row.fallibility == Fallibility::ResultFallible {
                assert!(
                    !row.error_set.is_empty(),
                    "fallible op {:?} must name its error set (C1)",
                    row.op
                );
            }
        }
    }

    // ─── Append mode ─────────────────────────────────────────────────────────

    /// Append mode writes to the end of an existing file (not the beginning).
    #[test]
    fn append_mode_writes_at_end() {
        let mut fs = fs();
        let p = Path::new("/append_test.txt");

        // Initial write.
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h1 = fs.open(&p, &opts_w).expect("create");
        fs.write(&mut h1, b"hello").expect("write first");
        fs.close(&mut h1).expect("close");

        // Append.
        let opts_a = OpenOptions::new().with_append(true);
        let mut h2 = fs.open(&p, &opts_a).expect("open for append");
        fs.write(&mut h2, b" world").expect("append");
        fs.close(&mut h2).expect("close");

        // Read the full content.
        let opts_r = OpenOptions::read_only();
        let mut h3 = fs.open(&p, &opts_r).expect("open for read");
        let mut buf = vec![0u8; 32];
        let n = fs.read(&mut h3, &mut buf).expect("read");
        fs.close(&mut h3).expect("close");

        assert_eq!(
            &buf[..n],
            b"hello world",
            "append must produce 'hello world'"
        );
    }

    // ─── Rename moves and removes ─────────────────────────────────────────────

    /// After rename, source is gone and target has the original content.
    #[test]
    fn rename_moves_content() {
        let mut fs = fs();
        let src = Path::new("/src_rename.txt");
        let dst = Path::new("/dst_rename.txt");

        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open(&src, &opts_w).expect("create");
        fs.write(&mut h, b"moved").expect("write");
        fs.close(&mut h).expect("close");

        fs.rename(&src, &dst).expect("rename");
        assert!(
            !fs.exists(&src).unwrap(),
            "source must be gone after rename"
        );
        assert!(fs.exists(&dst).unwrap(), "dest must exist after rename");

        let opts_r = OpenOptions::read_only();
        let mut rh = fs.open(&dst, &opts_r).expect("open renamed");
        let mut buf = vec![0u8; 8];
        let n = fs.read(&mut rh, &mut buf).expect("read");
        fs.close(&mut rh).expect("close");
        assert_eq!(&buf[..n], b"moved");
    }
}
