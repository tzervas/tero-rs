//! `OpenOptions` — explicit, declared open intent (C1/G2 — never a silent default).
//!
//! Every field defaults to `false`. Opening an absent path without `create` is `Err(NotFound)`,
//! not a conjured file. Opening an existing file without `truncate` preserves its bytes, not a
//! silent zero-out. The mutation is the caller's **declared intent or it does not happen** — this
//! is the spec §4.4 honesty crux made structural.
//!
//! Design spec: `docs/spec/stdlib/fs.md` §3; contract: RFC-0016 §4.1 C1/G2.

/// Declared open intent for `open` (spec §3).
///
/// All fields default `false`. Every create/truncate option is **opt-in, never a default**:
///
/// | Field | When `true` |
/// |---|---|
/// | `read` | Allow reading through the handle |
/// | `write` | Allow writing through the handle |
/// | `append` | Allow appending (implies write; position always at end) |
/// | `create` | Create the file if absent; open existing file if present |
/// | `create_new` | Create the file, failing with `AlreadyExists` if it exists |
/// | `truncate` | Zero-out the file's contents on open; requires `write` |
///
/// # C1 — never-silent
/// - Opening absent path, `create = false`, `create_new = false` → `Err(NotFound)` (no conjured file).
/// - Opening existing path, `truncate = false` → existing bytes preserved (no silent zero-out).
/// - `create_new = true` on an existing path → `Err(AlreadyExists)` (no silent overwrite).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenOptions {
    /// Allow reading through the handle.
    pub read: bool,
    /// Allow writing through the handle.
    pub write: bool,
    /// Append mode: writes always go to the end; position is at end.
    pub append: bool,
    /// Create the file if absent; open it if present. No-op if already exists.
    pub create: bool,
    /// Create the file; fail with `AlreadyExists` if it already exists.
    pub create_new: bool,
    /// Truncate the file's contents to zero on open. Requires `write = true`.
    pub truncate: bool,
}

impl OpenOptions {
    /// All-false options: pure open (no create, no truncate, no write).
    ///
    /// This is the **only** constructor. Every flag must be explicitly enabled by the caller;
    /// there is no convenience constructor that secretly enables a subset of flags.
    #[must_use]
    pub fn new() -> Self {
        Self {
            read: false,
            write: false,
            append: false,
            create: false,
            create_new: false,
            truncate: false,
        }
    }

    /// A read-only open (the most common case made ergonomic, while staying honest).
    ///
    /// Equivalent to `new().with_read(true)`.
    #[must_use]
    pub fn read_only() -> Self {
        Self {
            read: true,
            ..Self::new()
        }
    }

    /// Builder: enable reading.
    #[must_use]
    pub fn with_read(mut self, v: bool) -> Self {
        self.read = v;
        self
    }

    /// Builder: enable writing.
    #[must_use]
    pub fn with_write(mut self, v: bool) -> Self {
        self.write = v;
        self
    }

    /// Builder: enable append mode.
    #[must_use]
    pub fn with_append(mut self, v: bool) -> Self {
        self.append = v;
        self
    }

    /// Builder: enable create-if-absent.
    #[must_use]
    pub fn with_create(mut self, v: bool) -> Self {
        self.create = v;
        self
    }

    /// Builder: enable create-and-fail-if-exists.
    #[must_use]
    pub fn with_create_new(mut self, v: bool) -> Self {
        self.create_new = v;
        self
    }

    /// Builder: enable truncate.
    #[must_use]
    pub fn with_truncate(mut self, v: bool) -> Self {
        self.truncate = v;
        self
    }

    /// Validate that the option combination is coherent.
    ///
    /// Returns `Err(reason)` if the combination is self-contradictory (e.g. `truncate` without
    /// `write`). This is an internal consistency check — the `open` op calls it before touching
    /// any state.
    ///
    /// # Errors
    /// - `"truncate requires write or append"` — truncate is a write-side intent.
    /// - `"create_new and create are mutually exclusive"` — they disagree on whether an existing file is an error.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.truncate && !self.write && !self.append {
            return Err("truncate requires write or append");
        }
        if self.create_new && self.create {
            return Err("create_new and create are mutually exclusive");
        }
        Ok(())
    }

    /// Whether this intent requests any write capability (write or append).
    #[must_use]
    pub fn wants_write(&self) -> bool {
        self.write || self.append
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All fields default to `false` (C1 — no silent defaults).
    /// Guard: any non-false default makes this fail.
    #[test]
    fn all_fields_default_false() {
        let o = OpenOptions::new();
        assert!(!o.read, "read must default false");
        assert!(!o.write, "write must default false");
        assert!(!o.append, "append must default false");
        assert!(!o.create, "create must default false");
        assert!(!o.create_new, "create_new must default false");
        assert!(!o.truncate, "truncate must default false");
    }

    /// `read_only()` sets only `read = true`.
    #[test]
    fn read_only_sets_only_read() {
        let o = OpenOptions::read_only();
        assert!(o.read);
        assert!(!o.write);
        assert!(!o.create);
        assert!(!o.truncate);
    }

    /// truncate without write or append is invalid (C1: no silent truncation without declared intent).
    /// Guard: allowing truncate without write makes this fail.
    #[test]
    fn truncate_without_write_is_invalid() {
        let o = OpenOptions::new().with_truncate(true);
        assert!(
            o.validate().is_err(),
            "truncate without write/append must be invalid"
        );
    }

    /// truncate with write is valid.
    #[test]
    fn truncate_with_write_is_valid() {
        let o = OpenOptions::new().with_write(true).with_truncate(true);
        assert!(o.validate().is_ok());
    }

    /// `create_new` and `create` together is invalid (they are mutually exclusive intents).
    /// Guard: allowing both makes this fail.
    #[test]
    fn create_new_and_create_are_mutually_exclusive() {
        let o = OpenOptions::new().with_create(true).with_create_new(true);
        assert!(
            o.validate().is_err(),
            "create_new and create are mutually exclusive"
        );
    }

    /// A valid write-and-create option set passes validation.
    #[test]
    fn write_create_options_are_valid() {
        let o = OpenOptions::new().with_write(true).with_create(true);
        assert!(o.validate().is_ok());
    }

    /// `wants_write` is true for write or append, false otherwise.
    #[test]
    fn wants_write_reflects_write_capability() {
        assert!(!OpenOptions::new().wants_write());
        assert!(OpenOptions::new().with_write(true).wants_write());
        assert!(OpenOptions::new().with_append(true).wants_write());
    }

    /// Builder pattern is composable (each builder step returns a new value).
    #[test]
    fn builder_is_composable() {
        let o = OpenOptions::new()
            .with_read(true)
            .with_write(true)
            .with_create(true);
        assert!(o.read);
        assert!(o.write);
        assert!(o.create);
        assert!(!o.truncate);
    }
}
