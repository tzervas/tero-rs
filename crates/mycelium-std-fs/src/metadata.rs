//! `Metadata`, `FileKind`, `Permissions` — read-only filesystem metadata values (C4/ADR-003).
//!
//! Metadata is a **value snapshot**, not an identity: two equal-bytes files may have different
//! `mtime`s, but they are the same Mycelium value. Metadata is not part of path identity (ADR-003).
//!
//! Timestamps are carried as **opaque** OS metadata (`u64` since-epoch): `fs` does not interpret
//! them into monotonic/wall distinctions — that typed distinction belongs to `std.time` (M-529).
//! Carrying an opaque `u64` is the honest narrowing: it preserves the value without overclaiming
//! that `fs` understands its clock semantics (VR-5).
//!
//! Design spec: `docs/spec/stdlib/fs.md` §3; contract: RFC-0016 §4.1 C2/C4.

/// The kind of filesystem entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileKind {
    /// A regular file.
    File,
    /// A directory.
    Directory,
    /// A symbolic link (target not resolved; `fs` does not follow silently).
    Symlink,
    /// Some other OS-specific entry kind.
    Other,
}

/// Read/write/execute permission bits for owner, group, and others.
///
/// Represented as a plain value; no implicit behavior. Stored as the Unix permission bits
/// (a `u32` in `rwxrwxrwx` form) but exposed through typed accessors — no caller should
/// interpret raw bits (C3: no opaque bit patterns).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Permissions {
    /// Raw Unix mode bits (for preservation / display). Not the primary interface.
    mode: u32,
}

impl Permissions {
    /// Construct from raw Unix mode bits.
    #[must_use]
    pub fn from_mode(mode: u32) -> Self {
        Self { mode }
    }

    /// The raw mode bits (preserved for tooling; not the primary interface — C3).
    #[must_use]
    pub fn raw_mode(self) -> u32 {
        self.mode
    }

    /// Whether the owner has read permission.
    #[must_use]
    pub fn owner_read(self) -> bool {
        self.mode & 0o400 != 0
    }

    /// Whether the owner has write permission.
    #[must_use]
    pub fn owner_write(self) -> bool {
        self.mode & 0o200 != 0
    }

    /// Whether the owner has execute permission.
    #[must_use]
    pub fn owner_execute(self) -> bool {
        self.mode & 0o100 != 0
    }

    /// Whether the group has read permission.
    #[must_use]
    pub fn group_read(self) -> bool {
        self.mode & 0o040 != 0
    }

    /// Whether others have read permission.
    #[must_use]
    pub fn others_read(self) -> bool {
        self.mode & 0o004 != 0
    }

    /// Whether this entry is read-only for the owner (write bit not set).
    #[must_use]
    pub fn is_readonly(self) -> bool {
        !self.owner_write()
    }
}

/// A read-only snapshot of filesystem entry metadata (C4 / ADR-003 — metadata is a value).
///
/// `Metadata` is **not** an identity: two identical-content files may have different `mtime`s.
/// A path's identity is its content, not its metadata. `stat`/`exists` return this as a value
/// snapshot at the time of the call; the OS may change it immediately after (TOCTOU — see spec
/// §7-Q4).
///
/// Timestamps are opaque `u64` (seconds since epoch). Interpretation of wall vs monotonic is
/// `std.time` (M-529), not `fs` (spec §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    /// The kind of filesystem entry.
    pub kind: FileKind,
    /// The size in bytes (for files; 0 for directories and other kinds).
    pub size: u64,
    /// The Unix permission bits.
    pub permissions: Permissions,
    /// Modification time as seconds since Unix epoch (opaque; see FLAG below).
    ///
    /// # FLAG (timestamp semantics)
    /// This is a raw `u64` seconds-since-epoch. Interpreting it as monotonic/wall/calendar is
    /// `std.time` (M-529). `fs` carries it opaquely (VR-5: no overclaim of clock semantics).
    pub mtime_secs: u64,
}

impl Metadata {
    /// Construct a metadata value directly (used by the in-memory substrate).
    #[must_use]
    pub fn new(kind: FileKind, size: u64, permissions: Permissions, mtime_secs: u64) -> Self {
        Self {
            kind,
            size,
            permissions,
            mtime_secs,
        }
    }

    /// Whether this entry is a regular file.
    #[must_use]
    pub fn is_file(&self) -> bool {
        self.kind == FileKind::File
    }

    /// Whether this entry is a directory.
    #[must_use]
    pub fn is_dir(&self) -> bool {
        self.kind == FileKind::Directory
    }

    /// Whether this entry is a symbolic link.
    #[must_use]
    pub fn is_symlink(&self) -> bool {
        self.kind == FileKind::Symlink
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metadata is a value: equal fields → equal Metadata (C4).
    /// Guard: pointer-identity equality breaks this.
    #[test]
    fn metadata_equality_is_value_semantic() {
        let m1 = Metadata::new(FileKind::File, 42, Permissions::from_mode(0o644), 1_000_000);
        let m2 = Metadata::new(FileKind::File, 42, Permissions::from_mode(0o644), 1_000_000);
        assert_eq!(m1, m2, "metadata with equal fields must be equal (C4)");
    }

    /// Different mtime yields different Metadata (metadata is not filtered out of value equality).
    #[test]
    fn different_mtime_means_different_metadata() {
        let m1 = Metadata::new(FileKind::File, 42, Permissions::from_mode(0o644), 1_000_000);
        let m2 = Metadata::new(FileKind::File, 42, Permissions::from_mode(0o644), 2_000_000);
        assert_ne!(
            m1, m2,
            "metadata with different mtime must differ (mtime is part of the snapshot value)"
        );
    }

    /// Permission bit accessors are consistent with the raw mode.
    #[test]
    fn permission_bits_are_consistent_with_mode() {
        let p = Permissions::from_mode(0o755);
        assert!(p.owner_read());
        assert!(p.owner_write());
        assert!(p.owner_execute());
        assert!(p.group_read());
        assert!(p.others_read());
        assert!(!p.is_readonly());
    }

    /// Read-only permission (0o444) is correctly detected.
    #[test]
    fn readonly_permission_correctly_detected() {
        let p = Permissions::from_mode(0o444);
        assert!(p.is_readonly(), "0o444 must be read-only");
        assert!(!p.owner_write());
    }

    /// `is_file`, `is_dir`, `is_symlink` are mutually exclusive for the common kinds.
    #[test]
    fn file_kind_accessors_are_correct() {
        let file_meta = Metadata::new(FileKind::File, 0, Permissions::from_mode(0o644), 0);
        let dir_meta = Metadata::new(FileKind::Directory, 0, Permissions::from_mode(0o755), 0);
        let sym_meta = Metadata::new(FileKind::Symlink, 0, Permissions::from_mode(0o777), 0);

        assert!(file_meta.is_file() && !file_meta.is_dir() && !file_meta.is_symlink());
        assert!(!dir_meta.is_file() && dir_meta.is_dir() && !dir_meta.is_symlink());
        assert!(!sym_meta.is_file() && !sym_meta.is_dir() && sym_meta.is_symlink());
    }

    /// The raw mode is preserved round-trip.
    #[test]
    fn raw_mode_round_trips() {
        for mode in [0o000u32, 0o644, 0o755, 0o777, 0o400] {
            let p = Permissions::from_mode(mode);
            assert_eq!(p.raw_mode(), mode, "raw mode must round-trip for {mode:o}");
        }
    }
}
