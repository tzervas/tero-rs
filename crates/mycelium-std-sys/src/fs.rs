//! \[Declared\] Filesystem syscall floor. Thin wrappers over Rust `std::fs`.
//!
//! Declared — no audit of OS FS semantics; wiring from `std-fs` deferred to a future wave.
//!
//! Every operation returns an explicit `Result` on failure — never-silent (G2). No silent
//! creates, no silent truncations, no silent overwrites; the OS error is propagated as-is
//! via `std::io::Error`.
//!
//! RFC-0016 §9: once `std-fs` routes its `RealFs` backend exclusively through this module,
//! the pure `std-fs` crate earns a `wild`-free badge.

use std::io;
use std::path::Path;

/// \[Declared\] Read the entire contents of a file at `path`.
///
/// Returns `Err` on any OS error — never-silent (G2).
pub fn read(path: &Path) -> Result<Vec<u8>, io::Error> {
    std::fs::read(path)
}

/// \[Declared\] Write `contents` to a file at `path`, creating or truncating it.
///
/// Returns `Err` on any OS error — never-silent (G2).
pub fn write(path: &Path, contents: &[u8]) -> Result<(), io::Error> {
    std::fs::write(path, contents)
}

/// \[Declared\] Check whether a path exists on the filesystem.
///
/// Returns `false` on any OS error (consistent with `std::path::Path::exists`).
/// If distinguishing "does not exist" from "permission denied" is required, use
/// `std::fs::metadata` directly.
pub fn exists(path: &Path) -> bool {
    path.exists()
}

/// \[Declared\] Create a directory and all its parents.
///
/// Returns `Err` on any OS error — never-silent (G2).
pub fn create_dir_all(path: &Path) -> Result<(), io::Error> {
    std::fs::create_dir_all(path)
}

/// \[Declared\] Remove a file.
///
/// Returns `Err` on any OS error — never-silent (G2).
pub fn remove_file(path: &Path) -> Result<(), io::Error> {
    std::fs::remove_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Round-trip: write bytes to a temp file, read them back, verify equality.
    #[test]
    fn write_read_roundtrip() {
        let dir = env::temp_dir();
        let path = dir.join("mycelium_std_sys_fs_test_roundtrip.bin");

        let contents: &[u8] = b"Mycelium std-sys fs round-trip test data \x00\x01\x02\xFF";

        // Clean up any leftover from a previous run.
        let _ = remove_file(&path);

        write(&path, contents).expect("write failed");
        let read_back = read(&path).expect("read failed");
        assert_eq!(
            read_back, contents,
            "round-trip mismatch: wrote {contents:?}, read back {read_back:?}"
        );

        // Cleanup.
        remove_file(&path).expect("cleanup remove_file failed");
    }

    #[test]
    fn exists_and_remove() {
        let dir = env::temp_dir();
        let path = dir.join("mycelium_std_sys_fs_test_exists.bin");

        // Should not exist initially (clean up just in case).
        let _ = remove_file(&path);
        assert!(!exists(&path), "expected path to not exist before write");

        write(&path, b"hello").expect("write failed");
        assert!(exists(&path), "expected path to exist after write");

        remove_file(&path).expect("remove_file failed");
        assert!(!exists(&path), "expected path to not exist after remove");
    }

    #[test]
    fn create_dir_all_creates_nested() {
        let dir = env::temp_dir().join("mycelium_std_sys_fs_test_nested_dir");
        let nested = dir.join("a").join("b").join("c");

        // Ignore error if already exists.
        let _ = std::fs::remove_dir_all(&dir);

        create_dir_all(&nested).expect("create_dir_all failed");
        assert!(nested.is_dir(), "expected nested dir to exist");

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_nonexistent_is_err() {
        let path = env::temp_dir().join("mycelium_std_sys_fs_test_NONEXISTENT_XYZ.bin");
        let _ = remove_file(&path); // ensure it's gone
        let result = read(&path);
        assert!(result.is_err(), "expected Err for nonexistent file, got Ok");
    }
}
