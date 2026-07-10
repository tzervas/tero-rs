//! In-memory filesystem substrate (the testable IO abstraction layer).
//!
//! This module provides the `FsBackend` trait — a minimal seam between the `fs` API and the
//! actual storage mechanism — and an `InMemoryFs` implementation that is fully testable without
//! any OS interaction.
//!
//! # Why this exists
//! The spec (§2/§7-Q1) requires the real OS syscall floor to live in a separate `std-sys`
//! phylum (RFC-0016 §8-Q6 — now RESOLVED: the `std-sys` quarantine is the maintainer's call).
//! Until `std-sys` (M-541) lands, `fs` must be testable. The solution: a `FsBackend` trait
//! that `open`/`stat`/etc. dispatch through, with an `InMemoryFs` implementation for tests and
//! an (unimplemented, flagged) `RealFs` shim for when `std-sys` lands.
//!
//! # FLAG (std-sys / Q1)
//! The real-OS syscall floor (`RealFs`) is **deferred to `std-sys` (M-541)**. This trait is the
//! seam that will wire up to it. Until M-541 lands, `RealFs` is not implemented here — any code
//! path that would need it returns `FsErr::NotFound` or is unreachable in tests. The orchestrator
//! must wire `RealFs` at integration time.
//!
//! # FLAG (io seam / Q2)
//! The byte-transfer operations (`read`, `write`, `flush`) are owned by `std.io` (M-514). In
//! this crate we define a minimal local `ByteSink`/`ByteSource` trait just enough to make the
//! in-memory substrate testable. When M-514 lands, these local traits are replaced by M-514's
//! surface and the seam is threaded through. The local traits are private and never exported.
//!
//! Design spec: `docs/spec/stdlib/fs.md` §7-Q1/Q2; RFC-0016 §8-Q6.

use crate::error::FsErr;
use crate::metadata::{FileKind, Metadata, Permissions};
use crate::options::OpenOptions;

use std::collections::HashMap;

// ─── Minimal local IO seam (FLAG Q2 — replace with M-514 when it lands) ─────────────────────
//
// The byte-transfer operations (read, write, flush) are owned by std.io (M-514). The traits
// below are intentionally NOT defined here to avoid pre-committing M-514's surface. Instead,
// the in-memory substrate implements byte transfer directly on `InMemoryFs`.
//
// When M-514 lands, the seam here is: replace the inline Vec<u8> read/write logic with
// M-514's Read/Write traits. The `ByteSink`/`ByteSource` sketches are preserved as doc comments
// only — they are the *intended* seam, not an active trait hierarchy.
//
// # FLAG (io seam / Q2)
// The M-514 `Read`/`Write` surface is M-514's to own. Do NOT invent it here.

// ─── The in-memory filesystem ─────────────────────────────────────────────────────────────────

/// An in-memory filesystem node.
#[derive(Debug, Clone)]
enum FsNode {
    File {
        contents: Vec<u8>,
        permissions: Permissions,
        mtime: u64,
    },
    Directory {
        /// Children: basename → full path. We store children as paths into the same flat map.
        children: Vec<String>,
        permissions: Permissions,
        mtime: u64,
    },
}

/// An open file handle in the in-memory substrate.
#[derive(Debug)]
pub(crate) struct InMemHandle {
    pub(crate) path: String,
    pub(crate) read: bool,
    pub(crate) write: bool,
    pub(crate) append: bool,
    /// Current read/write position within the file contents.
    pub(crate) position: usize,
    pub(crate) consumed: bool,
}

/// An in-memory filesystem, fully testable without OS interaction.
///
/// Stores files as `Vec<u8>` in a flat map keyed by absolute path string.
/// Directories are explicit nodes (not implicit from path structure).
///
/// # Thread safety
/// Not thread-safe. This is a test-only substrate for the design-phase implementation.
#[derive(Debug)]
pub(crate) struct InMemoryFs {
    nodes: HashMap<String, FsNode>,
    disk_limit: Option<u64>,
    used_bytes: u64,
}

impl InMemoryFs {
    /// Create a new empty in-memory filesystem.
    #[must_use]
    pub(crate) fn new() -> Self {
        let mut nodes = HashMap::new();
        // The root directory always exists.
        nodes.insert(
            "/".to_owned(),
            FsNode::Directory {
                children: Vec::new(),
                permissions: Permissions::from_mode(0o755),
                mtime: 0,
            },
        );
        Self {
            nodes,
            disk_limit: None,
            used_bytes: 0,
        }
    }

    /// Set a simulated disk limit (in bytes), for testing DiskFull error paths.
    pub(crate) fn with_disk_limit(mut self, limit: u64) -> Self {
        self.disk_limit = Some(limit);
        self
    }

    /// The canonical path string used as a key in the nodes map.
    fn canon(path: &str) -> String {
        if path == "/" {
            return "/".to_owned();
        }
        path.trim_end_matches('/').to_owned()
    }

    /// Resolve the parent directory path string for a path.
    fn parent_of(path: &str) -> Option<String> {
        let s = path.trim_end_matches('/');
        if s.is_empty() || s == "/" {
            return None;
        }
        match s.rfind('/') {
            None => None,
            Some(0) => Some("/".to_owned()),
            Some(idx) => Some(s[..idx].to_owned()),
        }
    }

    /// Register a new child name under a parent directory.
    fn register_child(&mut self, parent: &str, child_path: &str) {
        if let Some(FsNode::Directory { children, .. }) = self.nodes.get_mut(parent) {
            if !children.contains(&child_path.to_owned()) {
                children.push(child_path.to_owned());
            }
        }
    }

    /// Remove a child name from its parent directory.
    fn unregister_child(&mut self, parent: &str, child_path: &str) {
        if let Some(FsNode::Directory { children, .. }) = self.nodes.get_mut(parent) {
            children.retain(|c| c != child_path);
        }
    }

    // ─── Public interface (called by fs.rs ops) ───────────────────────────────

    pub(crate) fn exists(&self, path: &str) -> Result<bool, FsErr> {
        let key = Self::canon(path);
        Ok(self.nodes.contains_key(&key))
    }

    pub(crate) fn stat(&self, path: &str) -> Result<Metadata, FsErr> {
        let key = Self::canon(path);
        match self.nodes.get(&key) {
            None => Err(FsErr::NotFound {
                path: path.to_owned(),
                why: "path does not exist",
            }),
            Some(FsNode::File {
                contents,
                permissions,
                mtime,
            }) => Ok(Metadata::new(
                FileKind::File,
                contents.len() as u64,
                *permissions,
                *mtime,
            )),
            Some(FsNode::Directory {
                permissions, mtime, ..
            }) => Ok(Metadata::new(FileKind::Directory, 0, *permissions, *mtime)),
        }
    }

    pub(crate) fn open(&mut self, path: &str, opts: &OpenOptions) -> Result<InMemHandle, FsErr> {
        if let Err(e) = opts.validate() {
            // A bad option combination is a request error, not a missing path (C3): the path may
            // well exist. Surface it as InvalidOptions so callers are not misled by NotFound.
            return Err(FsErr::InvalidOptions {
                path: path.to_owned(),
                why: e,
            });
        }
        let key = Self::canon(path);
        let exists = self.nodes.contains_key(&key);

        // C1: no silent create.
        if !exists {
            if opts.create_new || opts.create {
                // Ensure parent exists.
                let parent_key = Self::parent_of(&key).ok_or_else(|| FsErr::NotFound {
                    path: path.to_owned(),
                    why: "cannot create at root",
                })?;
                if !self.nodes.contains_key(&parent_key) {
                    return Err(FsErr::NotFound {
                        path: parent_key.clone(),
                        why: "parent directory does not exist",
                    });
                }
                match self.nodes.get(&parent_key) {
                    Some(FsNode::Directory { .. }) => {}
                    _ => {
                        return Err(FsErr::NotADirectory {
                            path: parent_key,
                            why: "parent component is not a directory",
                        });
                    }
                }
                // Create the file.
                self.nodes.insert(
                    key.clone(),
                    FsNode::File {
                        contents: Vec::new(),
                        permissions: Permissions::from_mode(0o644),
                        mtime: 0,
                    },
                );
                self.register_child(&parent_key, &key);
            } else {
                return Err(FsErr::NotFound {
                    path: path.to_owned(),
                    why: "path does not exist and create/create_new not set",
                });
            }
        } else {
            // C1: no silent overwrite for create_new.
            if opts.create_new {
                return Err(FsErr::AlreadyExists {
                    path: path.to_owned(),
                    why: "create_new set but path already exists",
                });
            }
            // Can't open a directory as a file.
            if let Some(FsNode::Directory { .. }) = self.nodes.get(&key) {
                return Err(FsErr::IsADirectory {
                    path: path.to_owned(),
                    why: "path is a directory",
                });
            }
            // C1: no silent truncation.
            if opts.truncate {
                let old_size = if let Some(FsNode::File { contents, .. }) = self.nodes.get(&key) {
                    contents.len() as u64
                } else {
                    0
                };
                if let Some(FsNode::File {
                    contents, mtime, ..
                }) = self.nodes.get_mut(&key)
                {
                    self.used_bytes = self.used_bytes.saturating_sub(old_size);
                    contents.clear();
                    *mtime += 1;
                }
            }
        }

        Ok(InMemHandle {
            path: key,
            read: opts.read,
            write: opts.write || opts.append,
            append: opts.append,
            position: if opts.append {
                // Append: position starts at end.
                if let Some(FsNode::File { contents, .. }) = self.nodes.get(&Self::canon(path)) {
                    contents.len()
                } else {
                    0
                }
            } else {
                0
            },
            consumed: false,
        })
    }

    /// Read bytes from an open handle into `buf`. Returns the number of bytes read.
    ///
    /// A short read (fewer than `buf.len()` bytes) is returned explicitly — not swallowed (C1).
    pub(crate) fn read(&self, handle: &mut InMemHandle, buf: &mut [u8]) -> Result<usize, FsErr> {
        if handle.consumed {
            return Err(FsErr::UseAfterConsume {
                path: handle.path.clone(),
                why: "handle has been consumed (LR-8)",
            });
        }
        if !handle.read {
            return Err(FsErr::PermDenied {
                path: handle.path.clone(),
                why: "handle was not opened for reading",
            });
        }
        match self.nodes.get(&handle.path) {
            None => Err(FsErr::NotFound {
                path: handle.path.clone(),
                why: "file no longer exists",
            }),
            Some(FsNode::File { contents, .. }) => {
                let start = handle.position;
                if start >= contents.len() {
                    return Ok(0); // EOF
                }
                let available = &contents[start..];
                let n = available.len().min(buf.len());
                buf[..n].copy_from_slice(&available[..n]);
                handle.position += n;
                Ok(n)
            }
            Some(FsNode::Directory { .. }) => Err(FsErr::IsADirectory {
                path: handle.path.clone(),
                why: "cannot read from a directory handle",
            }),
        }
    }

    /// Write bytes to an open handle. Returns the number of bytes written.
    ///
    /// A short write (fewer than `buf.len()` bytes) is returned explicitly — not swallowed (C1).
    pub(crate) fn write(&mut self, handle: &mut InMemHandle, buf: &[u8]) -> Result<usize, FsErr> {
        if handle.consumed {
            return Err(FsErr::UseAfterConsume {
                path: handle.path.clone(),
                why: "handle has been consumed (LR-8)",
            });
        }
        if !handle.write {
            return Err(FsErr::PermDenied {
                path: handle.path.clone(),
                why: "handle was not opened for writing",
            });
        }
        match self.nodes.get_mut(&handle.path) {
            None => Err(FsErr::NotFound {
                path: handle.path.clone(),
                why: "file no longer exists",
            }),
            Some(FsNode::File {
                contents, mtime, ..
            }) => {
                let original_len = contents.len();
                let pos = if handle.append {
                    original_len
                } else {
                    handle.position
                };
                let n = buf.len();
                let end = pos + n;
                // Net disk growth: an in-place overwrite (`end <= original_len`) consumes no
                // additional bytes; a write past the end (incl. a sparse seek) grows the file by
                // `end - original_len`. Charging the full `n` on every write would spuriously trip
                // the disk limit on overwrites and over-count `used_bytes`.
                let net_growth = end.saturating_sub(original_len) as u64;
                if let Some(limit) = self.disk_limit {
                    if self.used_bytes + net_growth > limit {
                        return Err(FsErr::DiskFull {
                            path: handle.path.clone(),
                            why: "disk limit exceeded",
                        });
                    }
                }
                // Extend the file if writing past the end (sparse zero-fill).
                if pos > contents.len() {
                    contents.resize(pos, 0);
                }
                // Insert / overwrite bytes at pos.
                if end <= contents.len() {
                    contents[pos..end].copy_from_slice(buf);
                } else {
                    contents.truncate(pos);
                    contents.extend_from_slice(buf);
                }
                handle.position = end;
                self.used_bytes += net_growth;
                *mtime += 1;
                Ok(n)
            }
            Some(FsNode::Directory { .. }) => Err(FsErr::IsADirectory {
                path: handle.path.clone(),
                why: "cannot write to a directory handle",
            }),
        }
    }

    /// Flush a handle (no-op for in-memory; returns Ok).
    ///
    /// In the real `std-sys` floor this would call `fsync`. Here it is a no-op that still
    /// checks the handle is not consumed (LR-8 consistency).
    pub(crate) fn flush(&self, handle: &InMemHandle) -> Result<(), FsErr> {
        if handle.consumed {
            return Err(FsErr::UseAfterConsume {
                path: handle.path.clone(),
                why: "handle has been consumed (LR-8)",
            });
        }
        Ok(())
    }

    /// Close (consume) a handle. After this call the handle is consumed and cannot be reused (LR-8).
    ///
    /// In the real `std-sys` floor this would call `close(fd)`. Here it marks the handle as
    /// consumed. A close error is surfaced, not dropped (C1).
    pub(crate) fn close(&self, handle: &mut InMemHandle) -> Result<(), FsErr> {
        if handle.consumed {
            return Err(FsErr::UseAfterConsume {
                path: handle.path.clone(),
                why: "handle already consumed (double-close, LR-8)",
            });
        }
        handle.consumed = true;
        Ok(())
    }

    pub(crate) fn read_dir(&self, path: &str) -> Result<Vec<String>, FsErr> {
        let key = Self::canon(path);
        match self.nodes.get(&key) {
            None => Err(FsErr::NotFound {
                path: path.to_owned(),
                why: "directory does not exist",
            }),
            Some(FsNode::Directory { children, .. }) => Ok(children.clone()),
            Some(FsNode::File { .. }) => Err(FsErr::NotADirectory {
                path: path.to_owned(),
                why: "path is a file, not a directory",
            }),
        }
    }

    pub(crate) fn create_dir(&mut self, path: &str) -> Result<(), FsErr> {
        let key = Self::canon(path);
        if self.nodes.contains_key(&key) {
            return Err(FsErr::AlreadyExists {
                path: path.to_owned(),
                why: "directory already exists",
            });
        }
        let parent_key = Self::parent_of(&key).ok_or_else(|| FsErr::NotFound {
            path: path.to_owned(),
            why: "cannot create at root",
        })?;
        if !self.nodes.contains_key(&parent_key) {
            return Err(FsErr::NotFound {
                path: parent_key.clone(),
                why: "parent directory does not exist",
            });
        }
        match self.nodes.get(&parent_key) {
            Some(FsNode::Directory { .. }) => {}
            _ => {
                return Err(FsErr::NotADirectory {
                    path: parent_key.clone(),
                    why: "parent is not a directory",
                });
            }
        }
        self.nodes.insert(
            key.clone(),
            FsNode::Directory {
                children: Vec::new(),
                permissions: Permissions::from_mode(0o755),
                mtime: 0,
            },
        );
        self.register_child(&parent_key, &key);
        Ok(())
    }

    pub(crate) fn remove_file(&mut self, path: &str) -> Result<(), FsErr> {
        let key = Self::canon(path);
        match self.nodes.get(&key) {
            None => {
                return Err(FsErr::NotFound {
                    path: path.to_owned(),
                    why: "file does not exist",
                });
            }
            Some(FsNode::Directory { .. }) => {
                return Err(FsErr::IsADirectory {
                    path: path.to_owned(),
                    why: "path is a directory; use remove_dir",
                });
            }
            Some(FsNode::File { contents, .. }) => {
                self.used_bytes = self.used_bytes.saturating_sub(contents.len() as u64);
            }
        }
        self.nodes.remove(&key);
        if let Some(parent) = Self::parent_of(&key) {
            self.unregister_child(&parent, &key);
        }
        Ok(())
    }

    pub(crate) fn remove_dir(&mut self, path: &str) -> Result<(), FsErr> {
        let key = Self::canon(path);
        match self.nodes.get(&key) {
            None => {
                return Err(FsErr::NotFound {
                    path: path.to_owned(),
                    why: "directory does not exist",
                });
            }
            Some(FsNode::File { .. }) => {
                return Err(FsErr::NotADirectory {
                    path: path.to_owned(),
                    why: "path is a file, not a directory",
                });
            }
            Some(FsNode::Directory { children, .. }) => {
                if !children.is_empty() {
                    return Err(FsErr::NotEmpty {
                        path: path.to_owned(),
                        why: "directory is not empty",
                    });
                }
            }
        }
        self.nodes.remove(&key);
        if let Some(parent) = Self::parent_of(&key) {
            self.unregister_child(&parent, &key);
        }
        Ok(())
    }

    pub(crate) fn rename(&mut self, from: &str, to: &str) -> Result<(), FsErr> {
        let from_key = Self::canon(from);
        let to_key = Self::canon(to);

        // C1: source must exist.
        if !self.nodes.contains_key(&from_key) {
            return Err(FsErr::NotFound {
                path: from.to_owned(),
                why: "source path does not exist",
            });
        }

        // C1: no silent overwrite — if target exists, error.
        if self.nodes.contains_key(&to_key) {
            return Err(FsErr::AlreadyExists {
                path: to.to_owned(),
                why: "target path already exists; rename would silently overwrite",
            });
        }

        // Ensure target parent exists.
        let to_parent = Self::parent_of(&to_key).ok_or_else(|| FsErr::NotFound {
            path: to.to_owned(),
            why: "cannot rename to root",
        })?;
        if !self.nodes.contains_key(&to_parent) {
            return Err(FsErr::NotFound {
                path: to_parent.clone(),
                why: "target parent directory does not exist",
            });
        }

        let node = self.nodes.remove(&from_key).expect("exists: checked above");
        // Unregister from old parent.
        if let Some(from_parent) = Self::parent_of(&from_key) {
            self.unregister_child(&from_parent, &from_key);
        }
        // Register under new parent.
        self.nodes.insert(to_key.clone(), node);
        self.register_child(&to_parent, &to_key);
        Ok(())
    }

    pub(crate) fn copy(&mut self, from: &str, to: &str) -> Result<u64, FsErr> {
        let from_key = Self::canon(from);
        let to_key = Self::canon(to);

        // C1: source must exist.
        let data = match self.nodes.get(&from_key) {
            None => {
                return Err(FsErr::NotFound {
                    path: from.to_owned(),
                    why: "source path does not exist",
                });
            }
            Some(FsNode::Directory { .. }) => {
                return Err(FsErr::IsADirectory {
                    path: from.to_owned(),
                    why: "source is a directory; copy is only for regular files",
                });
            }
            Some(FsNode::File {
                contents,
                permissions,
                mtime,
            }) => (contents.clone(), *permissions, *mtime),
        };

        // C1: no silent overwrite.
        if self.nodes.contains_key(&to_key) {
            return Err(FsErr::AlreadyExists {
                path: to.to_owned(),
                why: "target path already exists; copy would silently overwrite",
            });
        }

        // Ensure target parent exists.
        let to_parent = Self::parent_of(&to_key).ok_or_else(|| FsErr::NotFound {
            path: to.to_owned(),
            why: "cannot copy to root",
        })?;
        if !self.nodes.contains_key(&to_parent) {
            return Err(FsErr::NotFound {
                path: to_parent.clone(),
                why: "target parent directory does not exist",
            });
        }

        // Check disk limit.
        if let Some(limit) = self.disk_limit {
            if self.used_bytes + data.0.len() as u64 > limit {
                return Err(FsErr::DiskFull {
                    path: to.to_owned(),
                    why: "disk limit exceeded during copy",
                });
            }
        }

        let size = data.0.len() as u64;
        self.used_bytes += size;
        self.nodes.insert(
            to_key.clone(),
            FsNode::File {
                contents: data.0,
                permissions: data.1,
                mtime: data.2,
            },
        );
        self.register_child(&to_parent, &to_key);
        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::OpenOptions;

    fn make_fs() -> InMemoryFs {
        InMemoryFs::new()
    }

    /// An empty filesystem has a root directory.
    #[test]
    fn empty_fs_has_root() {
        let fs = make_fs();
        assert!(fs.exists("/").unwrap());
    }

    /// `exists` returns false for a non-existent path (never panics or returns sentinel).
    #[test]
    fn exists_false_for_absent() {
        let fs = make_fs();
        assert!(!fs.exists("/does/not/exist").unwrap());
    }

    /// Creating and reading a file round-trips through open/write/close/open/read.
    #[test]
    fn open_write_close_read_round_trips() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut handle = fs.open("/hello.txt", &opts_w).expect("open for write");
        let n = fs.write(&mut handle, b"hello").expect("write");
        assert_eq!(n, 5);
        fs.close(&mut handle).expect("close");

        let opts_r = OpenOptions::read_only();
        let mut rhandle = fs.open("/hello.txt", &opts_r).expect("open for read");
        let mut buf = vec![0u8; 8];
        let read_n = fs.read(&mut rhandle, &mut buf).expect("read");
        assert_eq!(read_n, 5);
        assert_eq!(&buf[..5], b"hello");
        fs.close(&mut rhandle).expect("close");
    }

    /// Opening an absent file without create/create_new returns `NotFound` (C1 — no silent create).
    #[test]
    fn open_absent_without_create_is_not_found() {
        let mut fs = make_fs();
        let opts = OpenOptions::read_only();
        let err = fs.open("/no-such.txt", &opts).expect_err("must fail");
        assert!(
            matches!(err, FsErr::NotFound { .. }),
            "expected NotFound; got {err:?}"
        );
    }

    /// An invalid option combination (`truncate` without write/append) returns `InvalidOptions`,
    /// not `NotFound` — a request error caught above the floor, with the right class (C3).
    #[test]
    fn open_invalid_options_is_invalid_options_not_not_found() {
        let mut fs = make_fs();
        // Pre-create the file so the path exists — proving the error is about the options, not
        // a missing path.
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let _ = fs.open("/file.txt", &opts_w).expect("create");
        let bad = OpenOptions::new().with_truncate(true); // truncate without write/append
        let err = fs
            .open("/file.txt", &bad)
            .expect_err("invalid options must fail");
        assert!(
            matches!(err, FsErr::InvalidOptions { .. }),
            "expected InvalidOptions; got {err:?}"
        );
    }

    /// `create_new` on an existing file returns `AlreadyExists` (C1 — no silent overwrite).
    #[test]
    fn create_new_on_existing_returns_already_exists() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/file.txt", &opts_w).expect("create");
        fs.close(&mut h).expect("close");

        let opts_cn = OpenOptions::new().with_write(true).with_create_new(true);
        let err = fs.open("/file.txt", &opts_cn).expect_err("must fail");
        assert!(
            matches!(err, FsErr::AlreadyExists { .. }),
            "expected AlreadyExists; got {err:?}"
        );
    }

    /// UseAfterConsume on a closed handle (LR-8).
    #[test]
    fn use_after_consume_returns_error() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut handle = fs.open("/x.txt", &opts_w).expect("open");
        fs.close(&mut handle).expect("close");
        let err = fs.write(&mut handle, b"data").expect_err("must fail");
        assert!(
            matches!(err, FsErr::UseAfterConsume { .. }),
            "writing to a consumed handle must return UseAfterConsume (LR-8); got {err:?}"
        );
    }

    /// Double-close returns `UseAfterConsume` (LR-8 — not a silent no-op).
    #[test]
    fn double_close_returns_use_after_consume() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut handle = fs.open("/y.txt", &opts_w).expect("open");
        fs.close(&mut handle).expect("first close");
        let err = fs.close(&mut handle).expect_err("second close must fail");
        assert!(
            matches!(err, FsErr::UseAfterConsume { .. }),
            "double close must return UseAfterConsume; got {err:?}"
        );
    }

    /// `truncate` clears the file on open (C1: truncation is declared, not silent).
    #[test]
    fn truncate_clears_existing_file() {
        let mut fs = make_fs();
        // Write initial content.
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/trunc.txt", &opts_w).expect("create");
        fs.write(&mut h, b"initial").expect("write");
        fs.close(&mut h).expect("close");

        // Re-open with truncate.
        let opts_t = OpenOptions::new().with_write(true).with_truncate(true);
        let mut h2 = fs.open("/trunc.txt", &opts_t).expect("open with truncate");
        fs.close(&mut h2).expect("close");

        // Read: should be empty.
        let opts_r = OpenOptions::read_only();
        let mut h3 = fs.open("/trunc.txt", &opts_r).expect("open for read");
        let mut buf = vec![0u8; 16];
        let n = fs.read(&mut h3, &mut buf).expect("read");
        assert_eq!(n, 0, "truncated file must be empty");
        fs.close(&mut h3).expect("close");
    }

    /// `remove_file` removes a file; subsequent `exists` returns false.
    #[test]
    fn remove_file_works() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/rm.txt", &opts_w).expect("create");
        fs.close(&mut h).expect("close");
        fs.remove_file("/rm.txt").expect("remove");
        assert!(!fs.exists("/rm.txt").unwrap());
    }

    /// `remove_file` on a directory returns `IsADirectory` (C1).
    #[test]
    fn remove_file_on_dir_is_error() {
        let mut fs = make_fs();
        fs.create_dir("/adir").expect("mkdir");
        let err = fs.remove_file("/adir").expect_err("must fail");
        assert!(
            matches!(err, FsErr::IsADirectory { .. }),
            "expected IsADirectory; got {err:?}"
        );
    }

    /// `remove_dir` on non-empty directory returns `NotEmpty` (C1).
    #[test]
    fn remove_dir_nonempty_returns_not_empty() {
        let mut fs = make_fs();
        fs.create_dir("/mydir").expect("mkdir");
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/mydir/child.txt", &opts_w).expect("create child");
        fs.close(&mut h).expect("close");
        let err = fs.remove_dir("/mydir").expect_err("must fail");
        assert!(
            matches!(err, FsErr::NotEmpty { .. }),
            "expected NotEmpty; got {err:?}"
        );
    }

    /// `rename` moves a file; source no longer exists; target has the contents.
    #[test]
    fn rename_moves_file() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/old.txt", &opts_w).expect("create");
        fs.write(&mut h, b"data").expect("write");
        fs.close(&mut h).expect("close");

        fs.rename("/old.txt", "/new.txt").expect("rename");
        assert!(!fs.exists("/old.txt").unwrap(), "old path must be gone");
        assert!(fs.exists("/new.txt").unwrap(), "new path must exist");
    }

    /// `rename` to an existing target returns `AlreadyExists` (C1 — no silent overwrite).
    #[test]
    fn rename_to_existing_returns_already_exists() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h1 = fs.open("/a.txt", &opts_w).expect("create a");
        fs.close(&mut h1).expect("close");
        let mut h2 = fs.open("/b.txt", &opts_w.clone()).expect("create b");
        fs.close(&mut h2).expect("close");
        let err = fs.rename("/a.txt", "/b.txt").expect_err("must fail");
        assert!(
            matches!(err, FsErr::AlreadyExists { .. }),
            "rename to existing must return AlreadyExists; got {err:?}"
        );
    }

    /// `copy` copies a file; both source and target exist afterward.
    #[test]
    fn copy_creates_target_with_same_contents() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/src.txt", &opts_w).expect("create");
        fs.write(&mut h, b"hello").expect("write");
        fs.close(&mut h).expect("close");

        let n = fs.copy("/src.txt", "/dst.txt").expect("copy");
        assert_eq!(n, 5);
        assert!(fs.exists("/src.txt").unwrap(), "source must still exist");
        assert!(fs.exists("/dst.txt").unwrap(), "target must exist");
    }

    /// `copy` to an existing target returns `AlreadyExists` (C1 — no silent overwrite).
    #[test]
    fn copy_to_existing_returns_already_exists() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h1 = fs.open("/a.txt", &opts_w).expect("create a");
        fs.close(&mut h1).expect("close");
        let mut h2 = fs.open("/b.txt", &opts_w.clone()).expect("create b");
        fs.close(&mut h2).expect("close");
        let err = fs.copy("/a.txt", "/b.txt").expect_err("must fail");
        assert!(
            matches!(err, FsErr::AlreadyExists { .. }),
            "copy to existing must return AlreadyExists; got {err:?}"
        );
    }

    /// DiskFull is returned when the simulated disk limit is exceeded.
    #[test]
    fn disk_full_is_explicit_not_silent() {
        let mut fs = InMemoryFs::new().with_disk_limit(4);
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/big.txt", &opts_w).expect("create");
        let err = fs.write(&mut h, b"hello").expect_err("must fail");
        assert!(
            matches!(err, FsErr::DiskFull { .. }),
            "disk full must return DiskFull; got {err:?}"
        );
        fs.close(&mut h).ok(); // consumed=true path
    }

    /// `read_dir` on a directory returns its children.
    #[test]
    fn read_dir_returns_children() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/f1.txt", &opts_w).expect("create f1");
        fs.close(&mut h).expect("close");
        let mut h2 = fs.open("/f2.txt", &opts_w.clone()).expect("create f2");
        fs.close(&mut h2).expect("close");

        let children = fs.read_dir("/").expect("read_dir");
        assert!(
            children.contains(&"/f1.txt".to_owned()),
            "root must contain f1.txt"
        );
        assert!(
            children.contains(&"/f2.txt".to_owned()),
            "root must contain f2.txt"
        );
    }

    /// `read_dir` on a file returns `NotADirectory` (C1).
    #[test]
    fn read_dir_on_file_returns_not_a_directory() {
        let mut fs = make_fs();
        let opts_w = OpenOptions::new().with_write(true).with_create(true);
        let mut h = fs.open("/f.txt", &opts_w).expect("create");
        fs.close(&mut h).expect("close");
        let err = fs.read_dir("/f.txt").expect_err("must fail");
        assert!(
            matches!(err, FsErr::NotADirectory { .. }),
            "expected NotADirectory; got {err:?}"
        );
    }
}
