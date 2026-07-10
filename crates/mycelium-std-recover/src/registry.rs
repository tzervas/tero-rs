//! A **minimal error-class registry** for the Rust-first `std.recover` surface (M-520).
//!
//! In the full Mycelium runtime the shared error-class registry is owned by `std.diag` (M-510,
//! RFC-0013 §4.5). For the Rust-first half we provide a thin, self-contained registry so
//! `std.recover` can be built + tested without depending on `std.diag`'s implementation details.
//! The full `std.diag` registry (M-510 leaf) will supersede this once both crates are complete and
//! integrated; this is not a duplication — it is a scoped contract that flags itself accordingly.
//!
//! X1 invariant: a class is a **name** resolved through the registry — never an evaluated string.

use std::collections::BTreeSet;

/// A registry-resolved error class name (RFC-0013 §4.5 — X1).
///
/// Opaque by construction: the only way to obtain a [`ClassName`] is through
/// [`ClassRegistry::resolve`], which checks the name exists. This makes "unknown class" a static
/// impossibility once a `ClassName` is in hand.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassName(String);

impl ClassName {
    /// The string representation of this name (for display and hashing only — not for equality).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ClassName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The explicit error returned by [`ClassRegistry::resolve`] when a name is not registered (X1).
///
/// This is a configuration error, not a runtime panic — the caller must handle it explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownClass {
    /// The attempted class name.
    pub name: String,
}

impl std::fmt::Display for UnknownClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown error class {:?}: not in the registry (RFC-0013 §4.5 X1); \
             register it first or check the spelling",
            self.name
        )
    }
}

mycelium_std_core::impl_std_error!(UnknownClass);

/// A simple, append-only **error-class registry** (RFC-0013 §4.5 / X1).
///
/// A class name is resolved only if it has been explicitly registered; unregistered names return
/// [`UnknownClass`] — never a silent fabrication (G2).
///
/// # Design note
/// This is the Rust-first scoped form of the shared registry that `std.diag` (M-510) owns in the
/// full runtime. The M-510 leaf will provide the canonical form; this one lives here so
/// `std.recover` is buildable and testable in isolation. FLAG to orchestrator: integration seam
/// between `mycelium-std-diag`'s `ClassRegistry` and this one must be reconciled once M-510
/// lands.
#[derive(Debug, Clone, Default)]
pub struct ClassRegistry {
    names: BTreeSet<String>,
}

impl ClassRegistry {
    /// An empty registry (no classes registered yet).
    #[must_use]
    pub fn new() -> Self {
        ClassRegistry::default()
    }

    /// Register a class name. Idempotent — registering the same name twice is a no-op.
    pub fn register(&mut self, name: impl Into<String>) {
        self.names.insert(name.into());
    }

    /// Builder: register a name.
    #[must_use]
    pub fn with(mut self, name: impl Into<String>) -> Self {
        self.register(name);
        self
    }

    /// Resolve a string to a [`ClassName`] if it is registered.
    ///
    /// # Errors
    /// Returns [`UnknownClass`] if `name` is not in the registry (X1 — never an eval'd string).
    pub fn resolve(&self, name: &str) -> Result<ClassName, UnknownClass> {
        if self.names.contains(name) {
            Ok(ClassName(name.to_owned()))
        } else {
            Err(UnknownClass {
                name: name.to_owned(),
            })
        }
    }

    /// Whether a name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}
