//! `mycelium-spore` — **`spore`**, packaging & publishing (M-368; ADR-013).
//!
//! Builds a **content-addressed `spore`** from a `mycelium-proj.toml` project — the deployable unit that
//! germinates into a colony (DN-06/Glossary). The load-bearing rule is ADR-003: **identity is the
//! content-addressed DAG** (the source code by hash + the resolved dependency edges + the germination
//! surface); **metadata is not identity** (`version`/`authors`/`summary`/… travel with the spore but never
//! define it). Two builds of the same code+deps produce the **same spore hash** regardless of the version
//! label. A missing or ambiguous publish input is an **explicit error**, never a guess (G2): a phylum with
//! no surface, a project with no sources, or a dependency with no `hash` is refused — no partial artifact.
//!
//! v0 scope (honest; contract §7): a **single project** with **hash-pinned** dependencies; the on-disk
//! encoding is a **named-provisional** reproducible form (M-368 §9.1), superseded append-only when the
//! RFC-0008 R2 wire-schema lands (the signing + germination contract are deferred there per ADR-013 §4).
//! Source code is content-addressed by **raw-byte BLAKE3**; canonicalized (mycfmt) hashing is a later
//! refinement. KC-3: above the kernel.

use std::path::{Path, PathBuf};

use mycelium_core::ContentHash;
use mycelium_proj::{Manifest, ProjectKind};

/// The content-addressed registry (M-732): `publish` / `resolve` over a local store (ADR-003).
pub mod registry;
pub use registry::{artifact_hash, publish, resolve, PublishReceipt, RegistryError, Resolved};

/// The remote OCI/GHCR registry backend (M-871/E26-1; ADR-037): `publish_remote` / `resolve_remote`
/// distributing the DN-28 dense-map over OCI, behind an [`remote::OciTransport`] (`oras` v0).
pub mod remote;
pub use remote::{
    publish_remote, resolve_remote, DenseMap, ObjectBlob, ObjectRef, OciTransport, OrasTransport,
    Reconstructed, RegistryTarget, RemoteError, RemotePublishReceipt, RemoteResolved,
};

/// A project source file, content-addressed (raw-byte BLAKE3; ADR-003).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    /// Path relative to the project root (forward-slashed, deterministic).
    pub path: String,
    /// `blake3:<hex>` of the file bytes.
    pub hash: ContentHash,
}

/// A resolved dependency edge — pinned by content hash (authoritative, ADR-003).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDep {
    /// The dependency's local name.
    pub name: String,
    /// The depended-on phylum.
    pub phylum: String,
    /// The content-address pin (`blake3:…`).
    pub hash: String,
    /// The human version requirement (metadata; not identity).
    pub version: Option<String>,
}

/// A built spore: its content-addressed identity plus the components that define it and the metadata that
/// travels with (but does not define) it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spore {
    /// The spore identity — `blake3` over the canonical DAG (code + deps + surface), **excluding**
    /// metadata (ADR-003).
    pub id: ContentHash,
    /// The project shape.
    pub kind: ProjectKind,
    /// The germination surface (sorted public export names).
    pub surface: Vec<String>,
    /// The content-addressed source files (sorted by path).
    pub sources: Vec<SourceFile>,
    /// The resolved dependency edges (sorted by name).
    pub deps: Vec<ResolvedDep>,
    /// The project name (metadata — carried, not identity).
    pub name: String,
    /// The project version, if any (metadata — carried, not identity).
    pub version: Option<String>,
}

/// A spore-build refusal — never a partial artifact (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SporeError {
    /// A missing/ambiguous publish input (no surface, no sources, a hashless dep, a bad include) (exit 3).
    Publish(String),
    /// An I/O error reading the project (exit 66).
    Io(String),
}

impl SporeError {
    /// The CLI exit code for this refusal.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self {
            SporeError::Publish(_) => 3,
            SporeError::Io(_) => 66,
        }
    }
}

impl std::fmt::Display for SporeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SporeError::Publish(m) => write!(f, "publish-error: {m}"),
            SporeError::Io(m) => write!(f, "io-error: {m}"),
        }
    }
}

impl std::error::Error for SporeError {}

/// Build a [`Spore`] from a parsed manifest and the project root directory.
///
/// # Errors
/// [`SporeError::Publish`] when a publish input is missing/ambiguous (no germination surface for a phylum,
/// no `.myc` sources, a dependency without a `hash`, or an `[spore].include` naming a non-exported nodule),
/// or [`SporeError::Io`] on a read failure. No partial artifact is produced (G2).
pub fn build_spore(manifest: &Manifest, project_dir: &Path) -> Result<Spore, SporeError> {
    let kind = manifest.project.kind;

    // 1. The germination surface: `[spore].include` (default `["surface"]`) resolved against
    //    `[surface].exports`. A `phylum` must expose a non-empty surface — nothing to germinate is an error.
    let exports = manifest
        .surface
        .as_ref()
        .map(|s| s.exports.clone())
        .unwrap_or_default();
    let include = manifest
        .spore
        .as_ref()
        .map(|s| s.include.clone())
        .filter(|i| !i.is_empty())
        .unwrap_or_else(|| vec!["surface".to_owned()]);

    let mut surface: Vec<String> = Vec::new();
    for entry in &include {
        if entry == "surface" {
            surface.extend(exports.iter().cloned());
        } else {
            // An explicit include must name a declared export (when a surface is declared) — a typo'd
            // surface would ship the wrong thing (G2).
            if manifest.surface.is_some() && !exports.contains(entry) {
                return Err(SporeError::Publish(format!(
                    "[spore].include names `{entry}`, which is not in [surface].exports — refusing to \
                     guess the germination surface (G2)"
                )));
            }
            surface.push(entry.clone());
        }
    }
    surface.sort();
    surface.dedup();
    if kind == ProjectKind::Phylum && surface.is_empty() {
        return Err(SporeError::Publish(
            "a phylum must declare its public [surface].exports (or [spore].include) — there is nothing \
             to germinate; the surface is never guessed (G2)"
                .to_owned(),
        ));
    }

    // 2. The code: every `.myc` source, content-addressed by raw-byte BLAKE3 (sorted, deterministic).
    let mut sources = collect_sources(project_dir)?;
    sources.sort_by(|a, b| a.path.cmp(&b.path));
    if sources.is_empty() {
        return Err(SporeError::Publish(format!(
            "no `.myc` sources under {} — nothing to package",
            project_dir.display()
        )));
    }

    // 3. The dependency edges: each pinned by `hash` (authoritative, ADR-003); a hashless dep is refused.
    let mut deps = Vec::with_capacity(manifest.dependencies.len());
    for d in &manifest.dependencies {
        // The manifest reader has already parsed this into a typed, well-formed `ContentHash`
        // (DN-40 A3) — here we only enforce that the pin is *present* (an unpinned dep is refused).
        let hash = d.hash.as_ref().ok_or_else(|| {
            SporeError::Publish(format!(
                "dependency `{}` has no `hash` — an unpinned dependency is not reproducible; pin it \
                 (`hash = \"blake3:…\"`, ADR-003/G2)",
                d.name
            ))
        })?;
        deps.push(ResolvedDep {
            name: d.name.clone(),
            phylum: d.phylum.clone(),
            hash: hash.as_str().to_owned(),
            version: d.version.clone(),
        });
    }
    deps.sort_by(|a, b| a.name.cmp(&b.name));

    // 4. Content-address the DAG (code + deps + surface + kind) — metadata excluded (ADR-003).
    let id = content_address(kind, &surface, &sources, &deps);

    Ok(Spore {
        id,
        kind,
        surface,
        sources,
        deps,
        name: manifest.project.name.clone(),
        version: manifest.project.version.clone(),
    })
}

/// The canonical, deterministic identity encoding (ADR-003) — **the single source of truth for spore
/// identity**. Metadata (`name`/`version`/`authors`/…) is **excluded** — only the code-by-hash DAG, the
/// dependency hash edges, the germination surface, and the project kind bear identity. Two builds of the
/// same code+deps yield the same spore hash. **Downstream verifiers (e.g. `mycelium-std-spore::verify`)
/// MUST call this function — never re-implement the encoding:** a parallel copy is exactly how the
/// `v0`/`v1` split arose (DRY; the verify path stamped a stale `v0` while `build_spore` stamped `v1`).
pub fn content_address(
    kind: ProjectKind,
    surface: &[String],
    sources: &[SourceFile],
    deps: &[ResolvedDep],
) -> ContentHash {
    // **Injectivity is the whole contract** (ADR-003): distinct (kind, surface, sources, deps) MUST
    // map to distinct addresses, or the content-addressed supply chain (dep pinning, resolve-by-hash,
    // immutability detection) can be substituted under. The original `v0` encoding emitted every
    // author-influenced field space/newline-delimited with **no length-prefix or escaping** — so a
    // crafted source path or dep field containing a space/newline could shift a field boundary and
    // alias two distinct DAGs onto one pre-image string (a second-pre-image collision; all three
    // `ResolvedDep` fields are free-text manifest strings, so this needed no preimage or filesystem).
    // `v1` **length-prefixes every variable-length field** (`<bytelen>:<bytes>`) — the load-bearing
    // part: a field spans exactly its byte count, so no embedded space/newline can forge a boundary
    // (netstring-style). Each section's record count is also recorded (defense-in-depth). Together the
    // pre-image is uniquely decodable ⇒ the encoding is injective by construction. Property-tested over adversarial
    // inputs (paths/names with spaces/newlines) in `src/tests/lib_tests.rs`. The version header bumps
    // `v0 -> v1`, which **re-addresses every spore** (append-only supersession of the explicitly
    // provisional format; acceptable pre-1.0 — no live registry). KEEP-OUT of the kernel (KC-3):
    // identity is a deterministic, *verifiable* encoding — it must be verified, never trusted.
    let mut s = String::from("mycelium-spore-v1\n");
    s.push_str(&format!("kind:{}\n", kind_str(kind)));
    s.push_str(&format!("surface:{}\n", surface.len()));
    for name in surface {
        push_field(&mut s, name);
    }
    s.push_str(&format!("code:{}\n", sources.len()));
    for f in sources {
        push_field(&mut s, &f.path);
        push_field(&mut s, f.hash.as_str());
    }
    s.push_str(&format!("deps:{}\n", deps.len()));
    for d in deps {
        // The hash is identity; the version requirement is metadata and is excluded here.
        push_field(&mut s, &d.name);
        push_field(&mut s, &d.phylum);
        push_field(&mut s, &d.hash);
    }
    let hex = blake3::hash(s.as_bytes()).to_hex();
    ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest")
}

/// Append one length-prefixed field to the canonical pre-image: `<byte-length>:<bytes>\n`. The byte
/// count removes all delimiter ambiguity — the field spans exactly `v.len()` bytes, so an embedded
/// space or newline cannot forge a record/field boundary (netstring-style canonicalization). This is
/// what makes [`content_address`]'s `v1` encoding **injective** where `v0` was not.
fn push_field(s: &mut String, v: &str) {
    s.push_str(&format!("{}:{}\n", v.len(), v));
}

/// The canonical `[project].kind` spelling.
#[must_use]
pub fn kind_str(kind: ProjectKind) -> &'static str {
    match kind {
        ProjectKind::Phylum => "phylum",
        ProjectKind::Program => "program",
        ProjectKind::Script => "script",
    }
}

/// The maximum directory nesting the source walk will descend before refusing (never-silent; G2).
///
/// A project tree this deep is **not** a legitimate `mycelium` layout — it is the signature of a
/// pathological or adversarial input (a symlink cycle, or a generated tree built to exhaust the
/// stack). The walk already refuses to follow symlinked directory entries (the primary cycle defence
/// below), so this cap is the **defence-in-depth** bound that turns "unbounded recursion ⇒ stack
/// overflow / infinite walk (build DoS)" into an **explicit, exit-coded refusal** (DN-40 §3). The
/// value is far above any plausible real nodule hierarchy.
const MAX_WALK_DEPTH: usize = 64;

/// Collect every `.myc` source under `dir` (recursively), content-addressed by raw-byte BLAKE3. Skips
/// hidden entries, `target/`, and the temp files a formatter might leave — deterministic and reproducible.
///
/// **Bounded + symlink-safe (DN-40 §3).** The walk **does not descend into symlinked directory
/// entries** (a symlinked-directory cycle would otherwise be an unbounded/infinite walk — a build
/// DoS), and it caps nesting at [`MAX_WALK_DEPTH`], returning an explicit [`SporeError::Publish`]
/// (never-silent; G2) rather than overflowing the stack.
fn collect_sources(dir: &Path) -> Result<Vec<SourceFile>, SporeError> {
    let mut out = Vec::new();
    walk(dir, dir, 0, &mut out)?;
    Ok(out)
}

fn walk(
    root: &Path,
    dir: &Path,
    depth: usize,
    out: &mut Vec<SourceFile>,
) -> Result<(), SporeError> {
    // Defence-in-depth bound: an over-deep tree (the signature of a symlink cycle or an adversarial
    // input built to exhaust the stack) is refused explicitly — never an unbounded recursion (G2).
    if depth > MAX_WALK_DEPTH {
        return Err(SporeError::Publish(format!(
            "source tree under {} nests deeper than {MAX_WALK_DEPTH} directories at {} — refusing \
             to recurse further (a tree this deep is not a valid layout; likely a symlink cycle or \
             an adversarial input). Flatten the tree or remove the offending link (DN-40/G2)",
            root.display(),
            dir.display(),
        )));
    }

    let entries =
        std::fs::read_dir(dir).map_err(|e| SporeError::Io(format!("{}: {e}", dir.display())))?;
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
        // Classify the entry via `symlink_metadata` — which stats the **link itself**, not its
        // target — so a symlink is detected as a symlink. `Path::is_dir()` (and `metadata`) follow
        // the link and would report a symlinked-directory cycle as an ordinary directory, recursing
        // forever. A symlinked entry is **skipped** here (deterministically), which by construction
        // means no directory cycle can be re-entered (the only path back into an ancestor is via a
        // link). Real directories and real files are handled exactly as before (DN-40 §3).
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| SporeError::Io(format!("{}: {e}", path.display())))?;
        if meta.file_type().is_symlink() {
            // Never silently follow a symlink (cycle / tree-escape risk); a symlinked source is not
            // part of the deterministic content-addressed tree. Skipped, not an error — a benign
            // convenience link must not fail the build, but it is never traversed.
            continue;
        }
        if meta.is_dir() {
            walk(root, &path, depth + 1, out)?;
        } else if meta.is_file() && path.extension().is_some_and(|x| x == "myc") {
            let bytes = std::fs::read(&path)
                .map_err(|e| SporeError::Io(format!("{}: {e}", path.display())))?;
            let hex = blake3::hash(&bytes).to_hex();
            let hash = ContentHash::from_parts("blake3", hex.as_str())
                .expect("blake3 hex is a valid digest");
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(SourceFile { path: rel, hash });
        }
    }
    Ok(())
}

/// The `EXPLAIN` of a built spore (no black box): the identity receipt, the surface, the code by hash, the
/// dependency edges, and the metadata — the metadata explicitly marked *not* identity (ADR-003).
#[must_use]
pub fn explain(spore: &Spore) -> String {
    let mut out = format!("spore: {}  →  {}\n", spore.name, spore.id.as_str());
    out.push_str(&format!("  kind:    {}\n", kind_str(spore.kind)));
    out.push_str(&format!("  surface: {}\n", spore.surface.join(", ")));
    out.push_str(&format!(
        "  code:    {} source file(s)\n",
        spore.sources.len()
    ));
    for f in &spore.sources {
        out.push_str(&format!("    {} {}\n", f.path, f.hash.as_str()));
    }
    out.push_str(&format!("  deps:    {}\n", spore.deps.len()));
    for d in &spore.deps {
        let v = d.version.as_deref().unwrap_or("*");
        out.push_str(&format!(
            "    {} → {} {} (version {v})\n",
            d.name, d.phylum, d.hash
        ));
    }
    let ver = spore.version.as_deref().unwrap_or("—");
    out.push_str(&format!(
        "  metadata: name={}, version={ver}  [not identity — ADR-003]\n",
        spore.name
    ));
    out
}

#[cfg(test)]
mod tests;
