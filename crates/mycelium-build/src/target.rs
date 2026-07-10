//! **Build-target profiles** (M-301/RFC-0004 §9.2/§9.3; r2): *which platforms* a build targets —
//! orthogonal to the §4 stable-component gate (which *definitions* are compiled, [`crate::decide`]).
//!
//! A build's target set is an explicit, flexible choice, opt-in to breadth, never forced to it
//! (RFC-0004 §9.2):
//! - [`BuildProfile::Interpret`] — no targets (runs on the reference interpreter; the dev default);
//! - [`BuildProfile::Slim`] — exactly one `(os, arch)`;
//! - [`BuildProfile::Selective`] — a chosen subset of `(os, arch)`;
//! - [`BuildProfile::Fat`] — **all** supported targets (universal), first-class but optional.
//!
//! The slim/selective/fat artifacts share **one** shape — a content-addressed per-target
//! [`VariantTable`] (RFC-0004 §9.3) — parameterized only by how many variants it carries, so moving
//! slim → selective → fat is a flag change, not a re-architecture. At run time [`VariantTable::select`]
//! picks the host's variant; an unmatched host is an **explicit miss** ([`DispatchMiss`]) the caller
//! resolves by interpreter fallback or explicit refusal — **never** a wrong-target variant (G2/SC-3).
//!
//! **Honest scope (VR-5):** cross-target *codegen* rides RFC-0004 §2's MLIR→LLVM path, which is
//! **deferred** (awaiting libMLIR). So [`realizable_targets`] admits only the **host** target (and
//! `interpret`) today; a non-host `--slim`/`--target`/`--fat` is an explicit
//! [`BuildError::CrossTargetDeferred`], never a host-only build mislabeled as fat. This module is the
//! build-orchestration layer that is *ready* for that backend, not the backend.

use std::collections::{BTreeMap, BTreeSet};

use mycelium_core::ContentHash;
use serde::{Deserialize, Serialize};

/// A supported operating system (the build-target OS dimension). Closed set; adding one is a
/// deliberate change (KC-3), like the paradigm kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Os {
    /// Linux.
    Linux,
    /// macOS / Darwin.
    MacOs,
    /// Windows.
    Windows,
}

/// A supported instruction-set architecture (the build-target arch dimension).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Arch {
    /// x86-64 / AMD64.
    X86_64,
    /// AArch64 / ARM64.
    Aarch64,
}

/// A build target: an `(os, arch)` pair. Sub-arch CPU features (AVX2, NEON, …) are a **runtime
/// dispatch** refinement *within* an arch (the M-360 SIMD feature-dispatch precedent), not part of
/// the build-target granularity here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Target {
    /// The operating system.
    pub os: Os,
    /// The architecture.
    pub arch: Arch,
}

impl Target {
    /// Construct a target.
    #[must_use]
    pub fn new(os: Os, arch: Arch) -> Self {
        Target { os, arch }
    }

    /// The target the build tool is itself running on, if it is a supported `(os, arch)` — `None`
    /// otherwise (an unsupported host is explicit, never guessed). Detected from the standard
    /// `cfg!` target macros.
    #[must_use]
    pub fn host() -> Option<Target> {
        let os = if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(target_os = "macos") {
            Os::MacOs
        } else if cfg!(target_os = "windows") {
            Os::Windows
        } else {
            return None;
        };
        let arch = if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else {
            return None;
        };
        Some(Target { os, arch })
    }
}

impl core::fmt::Display for Target {
    /// The `<os>-<arch>` spelling used by `--slim`/`--target` (RFC-0004 §9.2).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let os = match self.os {
            Os::Linux => "linux",
            Os::MacOs => "macos",
            Os::Windows => "windows",
        };
        let arch = match self.arch {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
        };
        write!(f, "{os}-{arch}")
    }
}

/// All supported `(os, arch)` targets — the universe `--fat` builds for. The full cross product of
/// the closed [`Os`]/[`Arch`] sets (a deliberate, enumerable matrix — KC-3).
#[must_use]
pub fn supported_targets() -> BTreeSet<Target> {
    let mut s = BTreeSet::new();
    for os in [Os::Linux, Os::MacOs, Os::Windows] {
        for arch in [Arch::X86_64, Arch::Aarch64] {
            s.insert(Target { os, arch });
        }
    }
    s
}

/// A build's **target-set profile** (RFC-0004 §9.2): how many platforms to build for. Flexible —
/// `Fat` is first-class but never forced; `Interpret`/`Slim`/`Selective` cost only what they ask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildProfile {
    /// No compiled targets — run on the reference interpreter (the dev default, zero build step).
    Interpret,
    /// Exactly one target — the smallest artifact.
    Slim(Target),
    /// A chosen subset of targets — "support these, no more".
    Selective(BTreeSet<Target>),
    /// All supported targets — the universal (fat) build.
    Fat,
}

impl BuildProfile {
    /// The concrete target set this profile resolves to (`Interpret` → empty; `Fat` →
    /// [`supported_targets`]).
    #[must_use]
    pub fn targets(&self) -> BTreeSet<Target> {
        match self {
            BuildProfile::Interpret => BTreeSet::new(),
            BuildProfile::Slim(t) => BTreeSet::from([*t]),
            BuildProfile::Selective(s) => s.clone(),
            BuildProfile::Fat => supported_targets(),
        }
    }

    /// Whether this profile compiles anything at all (`false` for `Interpret`).
    #[must_use]
    pub fn is_compiled(&self) -> bool {
        !matches!(self, BuildProfile::Interpret) && !self.targets().is_empty()
    }
}

/// Why a profile's targets cannot be *realized* yet (RFC-0004 §9.3, honest scope).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// One or more requested targets are **not the host**, and cross-target codegen is deferred
    /// (the MLIR→LLVM backend awaits libMLIR — RFC-0004 §2). Explicit: never a host-only build
    /// silently mislabeled as fat/cross.
    CrossTargetDeferred {
        /// The non-host targets that cannot be built today.
        requested: Vec<Target>,
        /// The detected host (what *can* be built).
        host: Target,
    },
    /// The build tool's own `(os, arch)` is not a supported target, so even a host build is refused
    /// (explicit, never guessed).
    UnsupportedHost,
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BuildError::CrossTargetDeferred { requested, host } => {
                let list: Vec<String> = requested.iter().map(ToString::to_string).collect();
                write!(
                    f,
                    "cross-target codegen is deferred (RFC-0004 §2, MLIR→LLVM backend): cannot build \
                     {} on host {host} yet — build the host target, or run on the interpreter",
                    list.join(", ")
                )
            }
            BuildError::UnsupportedHost => {
                write!(f, "the host (os, arch) is not a supported build target")
            }
        }
    }
}

impl std::error::Error for BuildError {}

/// The targets a profile can be **realized** to *today* (RFC-0004 §9.3). Until the MLIR→LLVM backend
/// lands, only the host target is buildable: an `Interpret` profile yields the empty set, a profile
/// asking for **only** the host yields `{host}`, and any **non-host** target is an explicit
/// [`BuildError::CrossTargetDeferred`] — never a silent host-only substitution.
pub fn realizable_targets(
    profile: &BuildProfile,
    host: Target,
) -> Result<BTreeSet<Target>, BuildError> {
    if !supported_targets().contains(&host) {
        return Err(BuildError::UnsupportedHost);
    }
    let targets = profile.targets();
    let non_host: Vec<Target> = targets.iter().copied().filter(|t| *t != host).collect();
    if non_host.is_empty() {
        Ok(targets)
    } else {
        Err(BuildError::CrossTargetDeferred {
            requested: non_host,
            host,
        })
    }
}

/// A **fat (multi-target) artifact's** per-target variant table (RFC-0004 §9.3): each compiled
/// definition variant keyed by its [`Target`], pointing at the variant's content-addressed artifact
/// (an entry in the object store, OQ-3). A `--slim`/`--target` artifact is the identical structure
/// with fewer variants.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantTable {
    variants: BTreeMap<Target, ContentHash>,
}

/// A runtime dispatch **miss**: the host matched no present variant (RFC-0004 §9.3). The caller
/// resolves it explicitly — fall back to the interpreter if it is in the image, else refuse — and
/// must **never** run a variant built for another target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchMiss {
    /// The host that found no matching variant.
    pub host: Target,
}

impl core::fmt::Display for DispatchMiss {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "no compiled variant for host {} — fall back to the interpreter or refuse (never run a \
             wrong-target variant)",
            self.host
        )
    }
}

impl std::error::Error for DispatchMiss {}

impl VariantTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        VariantTable::default()
    }

    /// Record a target's compiled-variant artifact hash.
    pub fn insert(&mut self, target: Target, artifact: ContentHash) {
        self.variants.insert(target, artifact);
    }

    /// The targets this artifact carries variants for.
    pub fn targets(&self) -> impl Iterator<Item = &Target> {
        self.variants.keys()
    }

    /// The number of variants (1 for slim, |targets| for fat).
    #[must_use]
    pub fn len(&self) -> usize {
        self.variants.len()
    }

    /// Whether the table is empty (an interpret-only artifact).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    /// **Runtime variant dispatch** (RFC-0004 §9.3): the artifact hash for `host`, or an explicit
    /// [`DispatchMiss`] — never a wrong-target variant (G2/SC-3). Inspectable like every selection in
    /// the system.
    pub fn select(&self, host: Target) -> Result<&ContentHash, DispatchMiss> {
        self.variants.get(&host).ok_or(DispatchMiss { host })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(os: Os, arch: Arch) -> Target {
        Target::new(os, arch)
    }

    #[test]
    fn profiles_resolve_to_the_right_target_sets() {
        let lx = t(Os::Linux, Arch::X86_64);
        let mac = t(Os::MacOs, Arch::Aarch64);
        assert!(BuildProfile::Interpret.targets().is_empty());
        assert!(!BuildProfile::Interpret.is_compiled());
        assert_eq!(BuildProfile::Slim(lx).targets(), BTreeSet::from([lx]));
        assert_eq!(
            BuildProfile::Selective(BTreeSet::from([lx, mac])).targets(),
            BTreeSet::from([lx, mac])
        );
        assert_eq!(BuildProfile::Fat.targets(), supported_targets());
        assert!(BuildProfile::Fat.is_compiled());
    }

    #[test]
    fn slim_and_selective_share_the_fat_shape_with_fewer_variants() {
        // The artifact shape is one VariantTable; slim is just |targets| = 1.
        let h = ContentHash::parse("blake3:abc").unwrap();
        let mut slim = VariantTable::new();
        slim.insert(t(Os::Linux, Arch::X86_64), h.clone());
        assert_eq!(slim.len(), 1);
        let mut fat = VariantTable::new();
        for tgt in supported_targets() {
            fat.insert(tgt, h.clone());
        }
        assert_eq!(fat.len(), supported_targets().len());
    }

    #[test]
    fn runtime_dispatch_is_never_silent_on_a_miss() {
        let h = ContentHash::parse("blake3:abc").unwrap();
        let lx = t(Os::Linux, Arch::X86_64);
        let win = t(Os::Windows, Arch::X86_64);
        let mut table = VariantTable::new();
        table.insert(lx, h.clone());
        assert_eq!(table.select(lx).unwrap(), &h);
        // A host with no variant is an explicit miss — never a wrong-target variant.
        assert_eq!(table.select(win), Err(DispatchMiss { host: win }));
    }

    #[test]
    fn interpret_and_host_slim_realize_but_cross_target_is_deferred() {
        let host = t(Os::Linux, Arch::X86_64);
        let other = t(Os::MacOs, Arch::Aarch64);
        // Interpret realizes to nothing (always fine).
        assert!(realizable_targets(&BuildProfile::Interpret, host)
            .unwrap()
            .is_empty());
        // Host slim realizes.
        assert_eq!(
            realizable_targets(&BuildProfile::Slim(host), host).unwrap(),
            BTreeSet::from([host])
        );
        // A non-host slim is the explicit deferred boundary.
        assert_eq!(
            realizable_targets(&BuildProfile::Slim(other), host).unwrap_err(),
            BuildError::CrossTargetDeferred {
                requested: vec![other],
                host
            }
        );
        // Fat asks for every target → deferred (lists the non-host ones).
        assert!(matches!(
            realizable_targets(&BuildProfile::Fat, host).unwrap_err(),
            BuildError::CrossTargetDeferred { .. }
        ));
    }

    #[test]
    fn the_host_is_a_supported_target_in_ci() {
        // CI runs on a supported (os, arch); if this ever fails, the host matrix needs a new entry.
        if let Some(host) = Target::host() {
            assert!(
                supported_targets().contains(&host),
                "host {host} unsupported"
            );
        }
    }
}
