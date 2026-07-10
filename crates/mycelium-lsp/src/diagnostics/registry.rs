//! The **error-class registry** (RFC-0013 §4.5, exclusion X1) — error-class names resolve through a
//! **known set**, never an evaluated string.
//!
//! DynEL turned config strings into exception classes with `eval(exception_str)` — arbitrary code
//! execution from a config file (DN-04 §6). Mycelium maps an error-class *name* through this registry:
//! a [`ClassName`] can be obtained **only** via [`ClassRegistry::resolve`], so there is no path from
//! configuration text to a class without a lookup, and **no `eval`** anywhere. An unknown name is an
//! explicit [`UnknownClass`] error — never silently ignored, never coerced (G2).

use std::collections::BTreeSet;
use std::fmt;

/// A **resolved** error-class name. The only constructor is [`ClassRegistry::resolve`]; you cannot
/// fabricate one from arbitrary text without a registry lookup (X1). This makes "looked up, never
/// evaluated" a type-level property, not a convention.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassName(String);

impl ClassName {
    /// The class name as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ClassName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Resolving an error-class name not in the registry — an **explicit** configuration error (X1: never
/// silently ignored, never coerced into code).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownClass {
    /// The unrecognized name as written.
    pub name: String,
}

impl fmt::Display for UnknownClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown error class {:?}: not in the registry (names are looked up, never evaluated — \
             RFC-0013 §4.5 X1); register it first or correct the name",
            self.name
        )
    }
}

impl std::error::Error for UnknownClass {}

/// The known set of error-class names a policy may name (RFC-0013 §4.5). The membership and extension
/// discipline is the implementation-task-open question RFC-0013 §8 left; v0 seeds the set from the
/// explicit errors the kernel + checker + linter already emit (the lint codes and the `SwapError`
/// family), and allows a downstream module to [`register`](Self::register) its own — still by
/// known-set lookup, never `eval`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassRegistry {
    classes: BTreeSet<String>,
}

/// The v0 built-in error classes — grounded in the explicit errors Mycelium already surfaces:
/// the `mycelium-lsp` lint codes, the `mycelium-cert::SwapError` family, and the
/// `CheckVerdict::NotValidated` refusal (RFC-0001/0002; RFC-0013 §1).
const BUILTIN_CLASSES: &[&str] = &[
    // Swap / representation-crossing refusals (mycelium-cert::SwapError; RFC-0002 §5).
    "SwapOutOfRange",
    "SwapIllegalPair",
    "SwapWrongSource",
    "SwapNonFinite",
    "SwapApproximateSource",
    "SwapInsufficientCapacity",
    "SwapAmbiguousDecode",
    "UnsupportedSwapPair",
    // Linter / authoring-invariant findings (mycelium-lsp::lint; SC-3, G2, VR-5).
    "ImplicitSwap",
    "UnverifiedBound",
    "PlaceholderPolicy",
    "FreeVariable",
    "PolicyDivergence",
    // Static-check / validation refusals (RFC-0001; CheckVerdict::NotValidated).
    "NotValidated",
    "TypeMismatch",
    "UnresolvedName",
];

impl ClassRegistry {
    /// An empty registry — resolves nothing until classes are registered.
    #[must_use]
    pub fn new() -> Self {
        ClassRegistry {
            classes: BTreeSet::new(),
        }
    }

    /// The registry seeded with the v0 built-in classes (`BUILTIN_CLASSES`).
    #[must_use]
    pub fn with_builtins() -> Self {
        ClassRegistry {
            classes: BUILTIN_CLASSES.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Register a downstream error class. Returns `true` if newly added. Extension is still
    /// known-set membership — never an `eval` path (X1).
    pub fn register(&mut self, name: impl Into<String>) -> bool {
        self.classes.insert(name.into())
    }

    /// Whether `name` is a known class.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.classes.contains(name)
    }

    /// Resolve a name to a [`ClassName`] **through the registry** — the only way to obtain one. An
    /// unknown name is an explicit [`UnknownClass`] error (X1; never silent).
    ///
    /// # Errors
    /// Returns [`UnknownClass`] if `name` is not registered.
    pub fn resolve(&self, name: &str) -> Result<ClassName, UnknownClass> {
        if self.classes.contains(name) {
            Ok(ClassName(name.to_owned()))
        } else {
            Err(UnknownClass {
                name: name.to_owned(),
            })
        }
    }

    /// The known class names, sorted (deterministic).
    pub fn classes(&self) -> impl Iterator<Item = &str> {
        self.classes.iter().map(String::as_str)
    }
}

impl Default for ClassRegistry {
    fn default() -> Self {
        Self::new()
    }
}
