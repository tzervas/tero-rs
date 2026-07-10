//! `Path` — a pure, immutable, value-semantic filesystem path (C4/ADR-003).
//!
//! A `Path` is a content-addressable, UTF-8 immutable value. Two equal paths are the same value;
//! metadata is not part of its identity (ADR-003). Path operations are **purely lexical** — they
//! involve no syscalls (no `wild`? = no) and are total (no `effects: io`).
//!
//! # FLAG (Q3 — path model + portability)
//! The spec FLAGs whether `Path` should be a `Text` newtype, a structured component list, or an
//! OS-string that may not be UTF-8 (the classic portability hazard). Non-UTF-8 paths are a
//! never-silent concern (C1): a lossy path decode must be an explicit error, not a replacement
//! char. For now `Path` is a validated UTF-8 newtype; the separator / normalization / case-fold
//! policy is conservative (no implicit normalization). This decision is **FLAGGED** — coordinate
//! with `text`/`string` (M-524) and RFC-0016 §8-Q3 before ratification.
//!
//! # Guarantee tag: `Exact`
//! `Path::new`, `join`, `parent`, `exists_in` (lexical check only) are all pure, total, and
//! deterministic — no precision dimension, no approximation (VR-5: Exact is honest here).

use std::fmt;

/// An immutable, content-addressable UTF-8 filesystem path (C4 / ADR-003).
///
/// A `Path` is a **value**: two `Path`s with the same string are the same value regardless of
/// where they were constructed. Metadata (mtime, inode) is NOT part of its identity.
///
/// # FLAG (Q3 — portability)
/// Currently backed by a `String` (UTF-8 guarantee). Non-UTF-8 OS paths are an open question
/// (spec §7-Q3 / RFC-0016 §8-Q3); a lossy decode would violate C1 — the maintainer's call.
///
/// # PATH SEPARATOR
/// Component separator is `/`. No implicit OS-specific normalization (no `\\` on Windows).
/// This is a conservative default pending the Q3 portability decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    inner: String,
}

impl Path {
    /// Construct a `Path` from a UTF-8 string slice.
    ///
    /// # Guarantee tag: `Exact`, total
    /// Accepts any UTF-8 string. Currently no normalization is applied (conservative; see FLAG Q3).
    /// An empty string is a valid path (the empty / relative path).
    ///
    /// # Effects: none (pure lexical; no syscall)
    /// # FLAG (Q3): non-UTF-8 OS paths are not handled; pending ratification.
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self { inner: s.into() }
    }

    /// The path as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Lexically join a child component onto this path.
    ///
    /// A `/`-separated join: `join("/foo", "bar")` = `/foo/bar`.
    /// If `child` is an absolute path (starts with `/`), it replaces `self` (POSIX convention).
    ///
    /// # Guarantee tag: `Exact`, total
    /// Pure lexical computation; no syscall, no normalization beyond joining.
    ///
    /// # Effects: none
    #[must_use]
    pub fn join(&self, child: &str) -> Self {
        if child.starts_with('/') {
            // Absolute child replaces the base (POSIX convention).
            Self::new(child)
        } else if self.inner.is_empty() || self.inner == "/" {
            // Root or empty: just append without double-slash.
            let mut s = self.inner.clone();
            if !s.ends_with('/') {
                s.push('/');
            }
            s.push_str(child);
            Self { inner: s }
        } else {
            Self {
                inner: format!("{}/{}", self.inner.trim_end_matches('/'), child),
            }
        }
    }

    /// The parent directory of this path, or `None` at the root.
    ///
    /// `parent("/foo/bar")` = `Some("/foo")`.
    /// `parent("/")` = `None` (at the root; nowhere to ascend).
    /// `parent("foo")` = `None` (relative single component; no parent in this path string).
    ///
    /// # Guarantee tag: `Exact` (total, returns `Option` — `None` is explicit, not a sentinel)
    /// # Effects: none (pure lexical)
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        let s = self.inner.trim_end_matches('/');
        if s.is_empty() || s == "/" {
            return None;
        }
        match s.rfind('/') {
            None => None,                            // no separator: single component
            Some(0) => Some(Self::new("/")),         // "/foo" → "/"
            Some(idx) => Some(Self::new(&s[..idx])), // "/a/b" → "/a"
        }
    }

    /// The final component of the path (the file/directory name), or `None` for root.
    ///
    /// # Guarantee tag: `Exact`, total (returns `Option`)
    /// # Effects: none
    #[must_use]
    pub fn file_name(&self) -> Option<&str> {
        let s = self.inner.trim_end_matches('/');
        if s.is_empty() {
            return None;
        }
        match s.rfind('/') {
            None => Some(s),
            Some(idx) => {
                let name = &s[idx + 1..];
                if name.is_empty() {
                    None
                } else {
                    Some(name)
                }
            }
        }
    }

    /// Whether this path starts with `/` (i.e. is an absolute path).
    ///
    /// # Guarantee tag: `Exact`, total
    /// # Effects: none
    #[must_use]
    pub fn is_absolute(&self) -> bool {
        self.inner.starts_with('/')
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl From<&str> for Path {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Path {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Path::new ───────────────────────────────────────────────────────────

    /// `Path::new` is pure and total: any UTF-8 string is accepted.
    /// Guard: panicking on any UTF-8 string makes this fail.
    #[test]
    fn path_new_is_total() {
        let _ = Path::new("");
        let _ = Path::new("/");
        let _ = Path::new("/foo/bar");
        let _ = Path::new("relative");
    }

    /// `Path` is value-semantic: two paths with the same string are equal.
    /// Guard: path identity using pointer equality makes this fail.
    #[test]
    fn path_equality_is_value_semantic() {
        let a = Path::new("/foo/bar");
        let b = Path::new("/foo/bar");
        assert_eq!(a, b, "paths with equal strings must be equal (C4)");
    }

    /// Different strings → different paths.
    /// Guard: a constant hash that ignores the string makes this fail.
    #[test]
    fn different_strings_are_different_paths() {
        let a = Path::new("/foo/bar");
        let b = Path::new("/foo/baz");
        assert_ne!(a, b);
    }

    // ─── join ────────────────────────────────────────────────────────────────

    /// join appends a relative child with a slash separator.
    #[test]
    fn join_appends_relative_child() {
        let p = Path::new("/foo");
        let q = p.join("bar");
        assert_eq!(q.as_str(), "/foo/bar");
    }

    /// Joining an absolute child replaces the base (POSIX convention).
    /// Guard: concatenating regardless of child absoluteness makes this fail.
    #[test]
    fn join_absolute_child_replaces_base() {
        let p = Path::new("/foo");
        let q = p.join("/absolute");
        assert_eq!(q.as_str(), "/absolute");
    }

    /// join from root produces a clean path (no double slash).
    #[test]
    fn join_from_root_no_double_slash() {
        let p = Path::new("/");
        let q = p.join("bar");
        assert_eq!(q.as_str(), "/bar");
    }

    /// join is associative: `p.join("a").join("b")` == `p.join("a/b")`.
    /// Guard: a non-associative implementation makes this fail.
    #[test]
    fn join_associativity() {
        let p = Path::new("/root");
        let two_steps = p.join("a").join("b");
        let one_step = p.join("a/b");
        assert_eq!(two_steps, one_step, "join must be associative");
    }

    // ─── parent ──────────────────────────────────────────────────────────────

    /// `parent` returns the parent directory for a multi-component path.
    #[test]
    fn parent_returns_parent_of_nested_path() {
        assert_eq!(Path::new("/foo/bar").parent(), Some(Path::new("/foo")));
    }

    /// `parent` at root returns `None` (not a sentinel).
    /// Guard: returning `Some("")` instead of `None` makes this fail.
    #[test]
    fn parent_at_root_returns_none() {
        assert_eq!(Path::new("/").parent(), None, "root has no parent");
    }

    /// `parent` of a single-level absolute path returns root.
    #[test]
    fn parent_of_single_level_absolute_returns_root() {
        assert_eq!(Path::new("/foo").parent(), Some(Path::new("/")));
    }

    /// `parent` of a relative single-component path returns `None`.
    #[test]
    fn parent_of_relative_single_component_returns_none() {
        assert_eq!(Path::new("foo").parent(), None);
    }

    /// `parent` of an empty path returns `None`.
    #[test]
    fn parent_of_empty_returns_none() {
        assert_eq!(Path::new("").parent(), None);
    }

    // ─── file_name ───────────────────────────────────────────────────────────

    /// `file_name` returns the final path component.
    #[test]
    fn file_name_returns_final_component() {
        assert_eq!(Path::new("/foo/bar.txt").file_name(), Some("bar.txt"));
    }

    /// `file_name` of root returns `None`.
    #[test]
    fn file_name_of_root_returns_none() {
        assert_eq!(Path::new("/").file_name(), None);
    }

    /// `file_name` of an empty path returns `None`.
    #[test]
    fn file_name_of_empty_returns_none() {
        assert_eq!(Path::new("").file_name(), None);
    }

    // ─── is_absolute ─────────────────────────────────────────────────────────

    /// Absolute paths start with `/`.
    #[test]
    fn is_absolute_for_slash_prefixed() {
        assert!(Path::new("/foo").is_absolute());
        assert!(Path::new("/").is_absolute());
    }

    /// Relative paths are not absolute.
    #[test]
    fn is_not_absolute_for_relative() {
        assert!(!Path::new("foo").is_absolute());
        assert!(!Path::new("").is_absolute());
    }

    // ─── Property: join preserves lexical determinism ────────────────────────

    /// Joining the same child always produces the same result (deterministic — Exact).
    /// Guard: any nondeterminism in join makes this fail.
    #[test]
    fn join_is_deterministic() {
        let base = Path::new("/base");
        for child in &["a", "b", "c/d", "e/f/g"] {
            assert_eq!(
                base.join(child),
                base.join(child),
                "join must be deterministic"
            );
        }
    }

    /// Parent and join are inverse for well-formed paths: `parent(p.join(c)) == p`.
    /// Guard: a non-inverse pair makes this fail.
    #[test]
    fn parent_is_inverse_of_join_for_simple_components() {
        let base = Path::new("/root/sub");
        let joined = base.join("file.txt");
        assert_eq!(
            joined.parent(),
            Some(base.clone()),
            "parent(base.join(child)) must equal base for a single-component child"
        );
    }
}
