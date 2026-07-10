//! `spore` **remote backend** — GHCR/OCI, distributing the DN-28 dense-map (ADR-037; M-871/E26-1).
//!
//! Extends [`crate::registry`]'s local, file-based store (M-732) with a **networked** sibling: a
//! published spore is distributed as an **OCI 1.1 artifact**, decomposed per DN-28's dense-map shape
//! (ADR-037 §2):
//!
//! ```text
//! each source object (bytes, BLAKE3 == SourceFile.hash)  ──►  one OCI blob (dedup by digest)
//! the dense-map { spore_id, kind, surface, objects, deps } ──►  the OCI config blob
//! name@version                                            ──►  the OCI tag
//! ```
//!
//! `oras` is the **v0 wire-transport driver** — it owns only the OCI HTTP mechanics; this module
//! owns the *design*: the dense-map decomposition, addressing, and fetch-and-verify. The transport
//! sits behind [`OciTransport`] so a future pure-Rust client is an append-only swap (ADR-037 §4).
//!
//! **Two content addresses, both checked, carried over from M-732 verbatim** (ADR-037 §2): every
//! fetched object's bytes must BLAKE3 to its declared `content_hash` (integrity), and the
//! reconstructed source set must recompute — via the **single canonical**
//! [`crate::content_address`] (never re-implemented; DRY, the historic v0/v1-split lesson) — to the
//! dense-map's recorded `spore_id` (identity). A missing object, a byte mismatch, an
//! extra/undescribed blob, or a `spore_id` mismatch is an explicit [`RemoteError::Integrity`], never
//! a silent partial (G2).
//!
//! **Guarantee posture (VR-5/transparency):** the whole remote path is **`Empirical`** (verified by
//! round-trip tests — a local OCI registry double here, `oras`/GHCR live elsewhere) or **`Declared`**
//! where it rests on `oras`/GHCR behavior this crate does not itself prove. Never `Proven`. `oras`
//! absent, or any nonzero exit, is an explicit, actionable [`RemoteError::ToolMissing`] /
//! [`RemoteError::Transport`] — never a silent skip.
//!
//! **KC-3:** no new runtime dependency. The dense-map codec is hand-rolled (no serde, mirroring
//! [`crate::content_address`]); the transport shells to the `oras` CLI as a subprocess.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use mycelium_core::ContentHash;
use mycelium_proj::ProjectKind;

use crate::{content_address, kind_str, ResolvedDep, SourceFile, Spore};

// ─── errors ─────────────────────────────────────────────────────────────────────────────────────

/// A remote-backend operation refusal — always explicit, never a partial/silent result (G2). Mirrors
/// [`crate::registry::RegistryError`]'s exit-code + `Display` style so a `spore` CLI failure looks
/// the same regardless of backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteError {
    /// A malformed/unsafe input: an empty or unsafe `name`/`version`/reference component, an
    /// unrecognized registry scheme, or an empty version constraint (exit 3).
    InvalidInput(String),
    /// No matching tag/artifact found (exit 4).
    NotFound(String),
    /// A content-integrity failure: a bad `ContentHash`, a hash mismatch, a missing/extra object, a
    /// malformed dense-map encoding, or a recomputed-`spore_id` mismatch (exit 5).
    Integrity(String),
    /// An immutability conflict (exit 6): [`publish_remote`] refuses to republish a **different**
    /// spore under an existing `name@version` (M-872) — parity with the local store's
    /// [`crate::registry::RegistryError::Conflict`] (ADR-003 / M-732). **Best-effort ceiling
    /// (Declared, VR-5):** it is a *client-side* pre-check (list-tags → compare `spore_id`); OCI tags
    /// are server-side mutable, so it is not a proven server invariant — never claimed `Proven`.
    Conflict(String),
    /// A request this v0 backend honestly cannot satisfy — a SemVer range constraint (exit 64).
    Unsupported(String),
    /// The `oras` CLI is not on `PATH` (exit 69).
    ToolMissing(String),
    /// `oras` ran but failed, or the OCI wire exchange otherwise failed (exit 74).
    Transport(String),
    /// A local filesystem error (temp files, materializing a resolved tree) (exit 66).
    Io(String),
}

impl RemoteError {
    /// The CLI exit code for this refusal.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self {
            RemoteError::InvalidInput(_) => 3,
            RemoteError::NotFound(_) => 4,
            RemoteError::Integrity(_) => 5,
            RemoteError::Conflict(_) => 6,
            RemoteError::Unsupported(_) => 64,
            RemoteError::ToolMissing(_) => 69,
            RemoteError::Transport(_) => 74,
            RemoteError::Io(_) => 66,
        }
    }
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoteError::InvalidInput(m) => write!(f, "input-error: {m}"),
            RemoteError::NotFound(m) => write!(f, "not-found: {m}"),
            RemoteError::Integrity(m) => write!(f, "integrity-error: {m}"),
            RemoteError::Conflict(m) => write!(f, "conflict: {m}"),
            RemoteError::Unsupported(m) => write!(f, "unsupported: {m}"),
            RemoteError::ToolMissing(m) => write!(f, "tool-missing: {m}"),
            RemoteError::Transport(m) => write!(f, "transport-error: {m}"),
            RemoteError::Io(m) => write!(f, "io-error: {m}"),
        }
    }
}

impl std::error::Error for RemoteError {}

fn io<E: std::fmt::Display>(ctx: &str, e: E) -> RemoteError {
    RemoteError::Io(format!("{ctx}: {e}"))
}

// ─── the dense-map ──────────────────────────────────────────────────────────────────────────────

/// One entry in a [`DenseMap`]'s object list: a source file's repo-relative path plus its
/// content-address (ADR-003 raw-byte BLAKE3, same as [`SourceFile`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRef {
    /// Path relative to the project root (forward-slashed; matches [`SourceFile::path`]).
    pub rel_path: String,
    /// `blake3:<hex>` of the object's bytes.
    pub content_hash: ContentHash,
}

impl ObjectRef {
    /// The OCI blob title this object maps to on push/pull (ADR-037 §2): `<blake3-hex>.myco`.
    #[must_use]
    pub fn oci_title(&self) -> String {
        title_from_hash(&self.content_hash)
    }
}

/// The DN-28 dense-map: the spore's DAG (spore identity, project kind, germination surface,
/// content-addressed objects, dependency edges) plus the carried (non-identity) `name`/`version`
/// metadata — the payload that becomes the OCI **config** blob (ADR-037 §2).
///
/// **Not identity-bearing on its own** — `spore_id` is *recorded* here but is only trustworthy once
/// [`verify_and_reconstruct`] has recomputed it from the fetched objects via
/// [`crate::content_address`] (the single canonical encoding; never re-implemented, DRY).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMap {
    /// The dense-map wire-format tag (`"mycelium-densemap-v1"`), carried so a decoder that reads a
    /// mismatched version fails loudly rather than mis-parsing (G2).
    pub format_version: &'static str,
    /// The spore identity this dense-map claims (ADR-003) — verified, not trusted, by
    /// [`verify_and_reconstruct`].
    pub spore_id: ContentHash,
    /// The project shape.
    pub kind: ProjectKind,
    /// The published package name (metadata; not identity).
    pub name: String,
    /// The published version label, if any (metadata; not identity).
    pub version: Option<String>,
    /// The germination surface (sorted).
    pub surface: Vec<String>,
    /// The content-addressed objects (sorted by `rel_path`) — one per OCI blob layer.
    pub objects: Vec<ObjectRef>,
    /// The resolved dependency edges (sorted by name).
    pub deps: Vec<ResolvedDep>,
}

/// The dense-map wire-format tag. Bumping this (e.g. to `-v2`) is an append-only supersession, same
/// discipline as [`crate::content_address`]'s `v0`→`v1` header.
const DENSE_MAP_HEADER: &str = "mycelium-densemap-v1\n";

/// The blob title an object with this content hash maps to on the OCI wire (ADR-037 §2):
/// `<blake3-hex>.myco`. The single source of truth [`ObjectRef::oci_title`] / [`ObjectBlob::oci_title`]
/// both call.
fn title_from_hash(hash: &ContentHash) -> String {
    format!("{}.myco", hash.digest())
}

/// Parse an OCI blob title (as minted by [`title_from_hash`]) back into the content hash it names.
/// `None` if `title` does not have the `.myco` suffix or the hex before it is not a well-formed
/// BLAKE3 digest (DN-40 wave-2 algorithm-aware check) — never a silent best-effort guess.
#[must_use]
pub fn content_hash_from_title(title: &str) -> Option<ContentHash> {
    let hex = title.strip_suffix(".myco")?;
    ContentHash::parse_digest(&format!("blake3:{hex}"))
}

/// Append one length-prefixed (netstring-style) field: `<byte-length>:<bytes>`. Same discipline as
/// [`crate::push_field`] — the byte count removes all delimiter ambiguity, so an embedded space or
/// newline in a free-text field (a `rel_path`, dependency `name`, …) cannot forge a field boundary.
/// **No trailing `\n`** is appended here (callers place the terminator explicitly, since some
/// callers chain a key prefix before this and a literal after).
fn push_len_field(out: &mut String, v: &str) {
    out.push_str(&format!("{}:{v}\n", v.len()));
}

/// Encode a [`DenseMap`] into its canonical, deterministic, injective, length-prefixed bytes
/// (ADR-037 §2) — the OCI config blob payload. **Injectivity is the whole contract**, same as
/// [`crate::content_address`]: every free-text field (`name`, `version`, `rel_path`, dependency
/// `name`/`phylum`/`hash`/`version`) is length-prefixed so an embedded space/newline/colon cannot
/// alias two distinct dense-maps onto one encoding. Fixed-vocabulary fields (`kind`) and
/// already-charset-restricted fields (`ContentHash`, which cannot contain a newline by construction
/// — [`ContentHash::parse`]) are written as plain lines. Sections are sorted here (`surface`,
/// `objects` by `rel_path`, `deps` by name) regardless of input order, so two [`DenseMap`]s with the
/// same logical content always encode identically.
#[must_use]
pub fn encode_dense_map(dm: &DenseMap) -> Vec<u8> {
    let mut surface = dm.surface.clone();
    surface.sort();
    let mut objects = dm.objects.clone();
    objects.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    let mut deps = dm.deps.clone();
    deps.sort_by(|a, b| a.name.cmp(&b.name));

    let mut out = String::from(DENSE_MAP_HEADER);
    out.push_str(&format!("spore_id:{}\n", dm.spore_id.as_str()));
    out.push_str(&format!("kind:{}\n", kind_str(dm.kind)));
    out.push_str("name:");
    push_len_field(&mut out, &dm.name);
    push_opt_field(&mut out, "version:", dm.version.as_deref());

    out.push_str(&format!("surface {}\n", surface.len()));
    for s in &surface {
        push_len_field(&mut out, s);
    }

    out.push_str(&format!("objects {}\n", objects.len()));
    for o in &objects {
        out.push_str("objpath:");
        push_len_field(&mut out, &o.rel_path);
        out.push_str(&format!("objhash:{}\n", o.content_hash.as_str()));
    }

    out.push_str(&format!("deps {}\n", deps.len()));
    for d in &deps {
        out.push_str("depname:");
        push_len_field(&mut out, &d.name);
        out.push_str("depphylum:");
        push_len_field(&mut out, &d.phylum);
        out.push_str("dephash:");
        push_len_field(&mut out, &d.hash);
        push_opt_field(&mut out, "depversion:", d.version.as_deref());
    }

    out.into_bytes()
}

/// Encode an `Option<&str>` field as `<key><"none"|"some:"+len-prefixed>\n`. The `none`/`some`
/// literals are fixed vocabulary (never user data), so no injectivity risk; this is what lets
/// [`decode_dense_map`] distinguish `None` from `Some("")` unambiguously.
fn push_opt_field(out: &mut String, key: &str, v: Option<&str>) {
    out.push_str(key);
    match v {
        None => out.push_str("none\n"),
        Some(s) => {
            out.push_str("some:");
            push_len_field(out, s);
        }
    }
}

/// A byte-cursor over an encoded dense-map. Operates on raw bytes (not `str::lines()`) because
/// length-prefixed field *values* may legitimately contain `\n` — only the cursor's own structural
/// bytes (headers, counts, key literals) are guaranteed newline-free.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    fn at_end(&self) -> bool {
        self.pos == self.buf.len()
    }

    /// Consume an exact literal, or fail with a positioned [`RemoteError::Integrity`].
    fn expect_literal(&mut self, lit: &str) -> Result<(), RemoteError> {
        let b = lit.as_bytes();
        if self.remaining().len() >= b.len() && &self.remaining()[..b.len()] == b {
            self.pos += b.len();
            Ok(())
        } else {
            Err(RemoteError::Integrity(format!(
                "dense-map decode: expected {lit:?} at byte offset {} (malformed or truncated encoding)",
                self.pos
            )))
        }
    }

    /// Try to consume `lit`; `true`/advances on match, `false`/no-op otherwise (never errors).
    fn try_consume(&mut self, lit: &str) -> bool {
        let b = lit.as_bytes();
        if self.remaining().len() >= b.len() && &self.remaining()[..b.len()] == b {
            self.pos += b.len();
            true
        } else {
            false
        }
    }

    /// Read raw bytes up to (and consuming) the next `\n`, as a UTF-8 `String`. For structural
    /// lines only (`spore_id:`/`kind:`/section-count headers) — these are charset-restricted by
    /// construction and never contain an embedded newline.
    fn read_raw_line(&mut self) -> Result<String, RemoteError> {
        let rest = self.remaining();
        let nl = rest.iter().position(|&b| b == b'\n').ok_or_else(|| {
            RemoteError::Integrity(format!(
                "dense-map decode: unterminated line at byte offset {} (missing newline — truncated encoding)",
                self.pos
            ))
        })?;
        let s = std::str::from_utf8(&rest[..nl])
            .map_err(|_| {
                RemoteError::Integrity(format!(
                    "dense-map decode: non-UTF-8 line at byte offset {}",
                    self.pos
                ))
            })?
            .to_owned();
        self.pos += nl + 1;
        Ok(s)
    }

    /// Read a `<key> <N>\n` section-count header, validating `key`. Returns `N`.
    fn read_count_line(&mut self, key: &str) -> Result<usize, RemoteError> {
        let line = self.read_raw_line()?;
        let prefix = format!("{key} ");
        let rest = line.strip_prefix(prefix.as_str()).ok_or_else(|| {
            RemoteError::Integrity(format!(
                "dense-map decode: expected `{key} <N>` section header, got {line:?}"
            ))
        })?;
        rest.parse::<usize>().map_err(|_| {
            RemoteError::Integrity(format!(
                "dense-map decode: bad count {rest:?} in `{key}` section header"
            ))
        })
    }

    /// Read a `<byte-length>:<bytes>\n` length-prefixed field, returning the decoded value. Strict:
    /// a missing `:`, a non-digit/overflowing length, a truncated value, non-UTF-8 bytes, or a
    /// missing trailing `\n` are each an explicit, positioned error (never silent truncation/default).
    fn read_len_field(&mut self) -> Result<String, RemoteError> {
        let rest = self.remaining();
        let colon = rest.iter().position(|&b| b == b':').ok_or_else(|| {
            RemoteError::Integrity(format!(
                "dense-map decode: malformed length-prefixed field at byte offset {} — missing `:`",
                self.pos
            ))
        })?;
        let len_bytes = &rest[..colon];
        if len_bytes.is_empty() || !len_bytes.iter().all(u8::is_ascii_digit) {
            return Err(RemoteError::Integrity(format!(
                "dense-map decode: malformed length prefix {:?} at byte offset {}",
                String::from_utf8_lossy(len_bytes),
                self.pos
            )));
        }
        let len_str = std::str::from_utf8(len_bytes).expect("ascii digits are valid UTF-8");
        let len: usize = len_str.parse().map_err(|_| {
            RemoteError::Integrity(format!(
                "dense-map decode: length prefix {len_str:?} at byte offset {} does not fit a usize",
                self.pos
            ))
        })?;
        let value_start = colon + 1;
        if rest.len() < value_start + len + 1 {
            return Err(RemoteError::Integrity(format!(
                "dense-map decode: truncated length-prefixed field (declared {len} byte(s)) at byte offset {}",
                self.pos
            )));
        }
        let value = std::str::from_utf8(&rest[value_start..value_start + len])
            .map_err(|_| {
                RemoteError::Integrity(format!(
                    "dense-map decode: length-prefixed field at byte offset {} is not valid UTF-8",
                    self.pos
                ))
            })?
            .to_owned();
        if rest[value_start + len] != b'\n' {
            return Err(RemoteError::Integrity(format!(
                "dense-map decode: length-prefixed field at byte offset {} is missing its trailing newline \
                 (a corrupt length prefix would otherwise misalign every following field)",
                self.pos
            )));
        }
        self.pos += value_start + len + 1;
        Ok(value)
    }
}

/// Read a `<key>none\n` / `<key>some:<len-prefixed>\n` optional field, the decode counterpart of
/// [`push_opt_field`].
fn read_opt_field(c: &mut Cursor<'_>, key: &str) -> Result<Option<String>, RemoteError> {
    c.expect_literal(key)?;
    if c.try_consume("none\n") {
        Ok(None)
    } else if c.try_consume("some:") {
        Ok(Some(c.read_len_field()?))
    } else {
        Err(RemoteError::Integrity(format!(
            "dense-map decode: expected `none` or `some:` after `{key}` at byte offset {}",
            c.pos
        )))
    }
}

fn kind_from_str(s: &str) -> Option<ProjectKind> {
    match s {
        "phylum" => Some(ProjectKind::Phylum),
        "program" => Some(ProjectKind::Program),
        "script" => Some(ProjectKind::Script),
        _ => None,
    }
}

/// Decode bytes produced by [`encode_dense_map`] back into a [`DenseMap`]. **Strict, total, and
/// never-silent** (G2), modeled on [`crate::registry::parse_entry`]: a bad header, a bad/duplicate
/// count, a malformed length prefix, a truncated or non-UTF-8 field, a malformed `ContentHash`, a
/// duplicate object/dependency entry, or trailing bytes after the last section is each an explicit,
/// positioned [`RemoteError::Integrity`] naming the fault — never a silent default, truncation, or
/// last-wins.
///
/// # Errors
/// [`RemoteError::Integrity`] on any malformed input (see above); this function performs no I/O.
pub fn decode_dense_map(bytes: &[u8]) -> Result<DenseMap, RemoteError> {
    let mut c = Cursor { buf: bytes, pos: 0 };

    c.expect_literal(DENSE_MAP_HEADER).map_err(|_| {
        RemoteError::Integrity(
            "dense-map decode: bad header — the first line is not `mycelium-densemap-v1` (not a \
             mycelium dense-map, or an incompatible format version)"
                .to_owned(),
        )
    })?;

    let spore_id_line = c.read_raw_line()?;
    let spore_id_str = spore_id_line.strip_prefix("spore_id:").ok_or_else(|| {
        RemoteError::Integrity(format!(
            "dense-map decode: expected a `spore_id:<hash>` line, got {spore_id_line:?}"
        ))
    })?;
    let spore_id = ContentHash::parse_digest(spore_id_str).ok_or_else(|| {
        RemoteError::Integrity(format!(
            "dense-map decode: `spore_id` value {spore_id_str:?} is not a well-formed content hash"
        ))
    })?;

    let kind_line = c.read_raw_line()?;
    let kind_val = kind_line.strip_prefix("kind:").ok_or_else(|| {
        RemoteError::Integrity(format!(
            "dense-map decode: expected a `kind:<phylum|program|script>` line, got {kind_line:?}"
        ))
    })?;
    let kind = kind_from_str(kind_val).ok_or_else(|| {
        RemoteError::Integrity(format!(
            "dense-map decode: unrecognized `kind` value {kind_val:?}"
        ))
    })?;

    c.expect_literal("name:")?;
    let name = c.read_len_field()?;
    let version = read_opt_field(&mut c, "version:")?;

    let n_surface = c.read_count_line("surface")?;
    let mut surface = Vec::with_capacity(n_surface);
    for _ in 0..n_surface {
        surface.push(c.read_len_field()?);
    }

    let n_objects = c.read_count_line("objects")?;
    let mut objects = Vec::with_capacity(n_objects);
    let mut seen_paths: HashSet<String> = HashSet::with_capacity(n_objects);
    for _ in 0..n_objects {
        c.expect_literal("objpath:")?;
        let rel_path = c.read_len_field()?;
        let hash_line = c.read_raw_line()?;
        let hash_str = hash_line.strip_prefix("objhash:").ok_or_else(|| {
            RemoteError::Integrity(format!(
                "dense-map decode: expected an `objhash:<hash>` line, got {hash_line:?}"
            ))
        })?;
        let content_hash = ContentHash::parse_digest(hash_str).ok_or_else(|| {
            RemoteError::Integrity(format!(
                "dense-map decode: object {rel_path:?} has a malformed content hash {hash_str:?}"
            ))
        })?;
        if !seen_paths.insert(rel_path.clone()) {
            return Err(RemoteError::Integrity(format!(
                "dense-map decode: duplicate object entry for rel_path {rel_path:?} — a dense-map's \
                 objects are a set, not a multiset (G2)"
            )));
        }
        objects.push(ObjectRef {
            rel_path,
            content_hash,
        });
    }

    let n_deps = c.read_count_line("deps")?;
    let mut deps = Vec::with_capacity(n_deps);
    let mut seen_deps: HashSet<String> = HashSet::with_capacity(n_deps);
    for _ in 0..n_deps {
        c.expect_literal("depname:")?;
        let dep_name = c.read_len_field()?;
        c.expect_literal("depphylum:")?;
        let phylum = c.read_len_field()?;
        c.expect_literal("dephash:")?;
        let hash = c.read_len_field()?;
        let dep_version = read_opt_field(&mut c, "depversion:")?;
        if !seen_deps.insert(dep_name.clone()) {
            return Err(RemoteError::Integrity(format!(
                "dense-map decode: duplicate dependency entry for name {dep_name:?} (G2)"
            )));
        }
        deps.push(ResolvedDep {
            name: dep_name,
            phylum,
            hash,
            version: dep_version,
        });
    }

    if !c.at_end() {
        return Err(RemoteError::Integrity(format!(
            "dense-map decode: {} trailing byte(s) after the `deps` section — the encoding is not \
             self-terminating for this input (G2)",
            c.buf.len() - c.pos
        )));
    }

    Ok(DenseMap {
        format_version: "mycelium-densemap-v1",
        spore_id,
        kind,
        name,
        version,
        surface,
        objects,
        deps,
    })
}

// ─── objects, build, and verify-and-reconstruct ────────────────────────────────────────────────

/// One source object's bytes, ready to push as an OCI blob layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectBlob {
    /// `blake3:<hex>` of `bytes` (ADR-003).
    pub content_hash: ContentHash,
    /// Path relative to the project root (matches the corresponding [`ObjectRef::rel_path`]).
    pub rel_path: String,
    /// The object's raw bytes.
    pub bytes: Vec<u8>,
}

impl ObjectBlob {
    /// The OCI blob title this object pushes/pulls under (ADR-037 §2): `<blake3-hex>.myco`.
    #[must_use]
    pub fn oci_title(&self) -> String {
        title_from_hash(&self.content_hash)
    }
}

/// Build a [`DenseMap`] + its [`ObjectBlob`]s from a built [`Spore`], reading each source file's
/// bytes from `project_dir`. Every object's on-disk bytes are re-hashed and asserted to match the
/// spore's recorded [`SourceFile::hash`] — a project directory that has drifted since the spore was
/// built (edited/deleted source) is refused, never silently published under a stale address (G2).
///
/// # Errors
/// [`RemoteError::Io`] if a source file cannot be read; [`RemoteError::Integrity`] if a source
/// file's current bytes do not hash to the spore's recorded address for it.
pub fn build_dense_map(
    spore: &Spore,
    project_dir: &Path,
) -> Result<(DenseMap, Vec<ObjectBlob>), RemoteError> {
    let mut objects = Vec::with_capacity(spore.sources.len());
    let mut blobs = Vec::with_capacity(spore.sources.len());
    for src in &spore.sources {
        let full = project_dir.join(&src.path);
        let bytes = std::fs::read(&full).map_err(|e| io(&full.display().to_string(), e))?;
        let hex = blake3::hash(&bytes).to_hex();
        let actual =
            ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest");
        if actual != src.hash {
            return Err(RemoteError::Integrity(format!(
                "source {} now hashes to {} but the spore records {} — the project directory has \
                 drifted since the spore was built; refusing to publish a stale/mismatched object (G2)",
                src.path,
                actual.as_str(),
                src.hash.as_str()
            )));
        }
        objects.push(ObjectRef {
            rel_path: src.path.clone(),
            content_hash: src.hash.clone(),
        });
        blobs.push(ObjectBlob {
            content_hash: src.hash.clone(),
            rel_path: src.path.clone(),
            bytes,
        });
    }
    objects.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    blobs.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let dense_map = DenseMap {
        format_version: "mycelium-densemap-v1",
        spore_id: spore.id.clone(),
        kind: spore.kind,
        name: spore.name.clone(),
        version: spore.version.clone(),
        surface: spore.surface.clone(),
        objects,
        deps: spore.deps.clone(),
    };
    Ok((dense_map, blobs))
}

/// The verified, reconstructed result of a resolve: the recovered source tree plus the dense-map it
/// came from (its `spore_id` is now trustworthy — every object was hash-checked and the DAG
/// recomputed to match, per [`verify_and_reconstruct`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconstructed {
    /// The verified spore identity (ADR-003).
    pub spore_id: ContentHash,
    /// The recovered `(rel_path, bytes)` source files.
    pub sources: Vec<(String, Vec<u8>)>,
    /// The dense-map this was reconstructed from.
    pub dense_map: DenseMap,
}

/// Fetch-and-verify (ADR-037 §2, the load-bearing check): given a decoded `dense_map` and the
/// `fetched` `(title, bytes)` blobs a transport pulled, verify **every** declared object is present
/// with matching bytes, that no undescribed/duplicate blob was fetched, and that the reconstructed
/// source set recomputes — via [`crate::content_address`] (the single canonical encoding) — to the
/// dense-map's declared `spore_id`. **Never-silent (G2):** a missing object, a byte mismatch, an
/// extra/duplicate blob, or a `spore_id` mismatch is an explicit [`RemoteError`], never a silent
/// partial reconstruction.
///
/// # Errors
/// [`RemoteError::Integrity`] for a duplicate fetched title, a duplicate dense-map object, a byte
/// mismatch, an extra/undescribed blob, or a `spore_id` mismatch; [`RemoteError::NotFound`] if a
/// declared object has no matching fetched blob.
pub fn verify_and_reconstruct(
    dense_map: DenseMap,
    fetched: &[(String, Vec<u8>)],
) -> Result<Reconstructed, RemoteError> {
    let mut by_title: HashMap<&str, &[u8]> = HashMap::with_capacity(fetched.len());
    for (title, bytes) in fetched {
        if by_title.insert(title.as_str(), bytes.as_slice()).is_some() {
            return Err(RemoteError::Integrity(format!(
                "fetched blob title {title:?} appears more than once among the pulled layers (G2)"
            )));
        }
    }

    let mut declared_titles: HashSet<String> = HashSet::with_capacity(dense_map.objects.len());
    let mut sources: Vec<(String, Vec<u8>)> = Vec::with_capacity(dense_map.objects.len());
    let mut rebuilt: Vec<SourceFile> = Vec::with_capacity(dense_map.objects.len());

    for obj in &dense_map.objects {
        let title = obj.oci_title();
        if !declared_titles.insert(title.clone()) {
            return Err(RemoteError::Integrity(format!(
                "dense-map declares object {:?} more than once (title {title:?}) — a corrupt or \
                 adversarial dense-map (G2)",
                obj.rel_path
            )));
        }
        let bytes = by_title.get(title.as_str()).ok_or_else(|| {
            RemoteError::NotFound(format!(
                "dense-map object {} ({}) has no matching fetched blob (title {title:?}) — a missing \
                 object (G2)",
                obj.rel_path,
                obj.content_hash.as_str()
            ))
        })?;
        let hex = blake3::hash(bytes).to_hex();
        let actual =
            ContentHash::from_parts("blake3", hex.as_str()).expect("blake3 hex is a valid digest");
        if actual != obj.content_hash {
            return Err(RemoteError::Integrity(format!(
                "fetched object {} does not hash to its declared content address {} (got {}) — \
                 tampered or corrupt (G2)",
                obj.rel_path,
                obj.content_hash.as_str(),
                actual.as_str()
            )));
        }
        sources.push((obj.rel_path.clone(), bytes.to_vec()));
        rebuilt.push(SourceFile {
            path: obj.rel_path.clone(),
            hash: obj.content_hash.clone(),
        });
    }

    if let Some((extra_title, _)) = fetched.iter().find(|(t, _)| !declared_titles.contains(t)) {
        return Err(RemoteError::Integrity(format!(
            "fetched blob {extra_title:?} is not described by any dense-map object — an \
             extra/undescribed layer (G2)"
        )));
    }

    rebuilt.sort_by(|a, b| a.path.cmp(&b.path));
    let recomputed = content_address(
        dense_map.kind,
        &dense_map.surface,
        &rebuilt,
        &dense_map.deps,
    );
    if recomputed != dense_map.spore_id {
        return Err(RemoteError::Integrity(format!(
            "reconstructed spore recomputes to spore_id {} but the dense-map declares {} — an \
             identity mismatch (a substituted object, a tampered dense-map, or drifted deps/surface; G2)",
            recomputed.as_str(),
            dense_map.spore_id.as_str()
        )));
    }

    Ok(Reconstructed {
        spore_id: dense_map.spore_id.clone(),
        sources,
        dense_map,
    })
}

// ─── transport ──────────────────────────────────────────────────────────────────────────────────

/// The fetched-layer shape a [`OciTransport::pull`] returns: one `(title, bytes)` pair per OCI blob
/// layer. Factored out (clippy `type_complexity`) — see [`ObjectBlob::oci_title`] for the title
/// convention.
pub type PulledLayers = Vec<(String, Vec<u8>)>;

/// The OCI wire-transport driver, behind a trait so `oras` (v0) can be replaced by a future
/// pure-Rust client append-only, without touching the registry design above (ADR-037 §4/KC-3).
pub trait OciTransport {
    /// Push `config` (the encoded dense-map) and `layers` (the objects) to `reference`
    /// (`<repo>:<tag>`), returning the manifest digest.
    ///
    /// # Errors
    /// A transport-specific [`RemoteError`] (never a silent partial push).
    fn push(
        &self,
        reference: &str,
        config: &[u8],
        layers: &[ObjectBlob],
    ) -> Result<String, RemoteError>;

    /// Pull `reference`, returning the config bytes (the encoded dense-map) and the fetched
    /// `(title, bytes)` layers.
    ///
    /// # Errors
    /// A transport-specific [`RemoteError`] (never a silent partial pull).
    fn pull(&self, reference: &str) -> Result<(Vec<u8>, PulledLayers), RemoteError>;

    /// List the tags published under `repo` (`<base>/<name>`, no tag).
    ///
    /// # Errors
    /// A transport-specific [`RemoteError`].
    fn list_tags(&self, repo: &str) -> Result<Vec<String>, RemoteError>;
}

/// Spawn `oras`, mapping a missing binary to [`RemoteError::ToolMissing`] (never a silent skip) and
/// any other spawn failure to [`RemoteError::Transport`].
fn spawn_oras(mut cmd: Command) -> Result<std::process::Output, RemoteError> {
    cmd.output().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            RemoteError::ToolMissing(
                "the `oras` CLI is not on PATH — it is the v0 OCI wire-transport driver (ADR-037 §1); \
                 install it (https://oras.land) or publish/resolve against a local (non-`oci://`/`ghcr://`) \
                 registry path instead"
                    .to_owned(),
            )
        } else {
            RemoteError::Transport(format!("failed to spawn `oras`: {e}"))
        }
    })
}

/// Assert `output` succeeded, else an explicit [`RemoteError::Transport`] naming `what` and the
/// stderr tail (never a silent skip of a failed OCI operation).
fn require_success(output: &std::process::Output, what: &str) -> Result<(), RemoteError> {
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(RemoteError::Transport(format!(
        "`{what}` exited with {:?}: {}",
        output.status.code(),
        tail_lines(&stderr, 20)
    )))
}

/// The last `n` lines of `s` (for a bounded, readable error message from a possibly-long `oras`
/// stderr stream).
fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

/// Map a **failed** `oras` [`std::process::Output`] to a [`RemoteError`], distinguishing an OCI
/// "not found" (a missing repository or tag — a *normal, expected* condition when checking whether
/// something is already published) from a genuine transport/auth failure. Used where "absent" is a
/// legitimate answer (the immutability pre-check, resolve of a missing tag), so a first publish
/// isn't mistaken for an error.
///
/// The distinction is **grounded** in the OCI distribution spec's `NAME_UNKNOWN`/`MANIFEST_UNKNOWN`
/// errors, which both `registry:2` and GHCR surface with the substrings matched below (verified
/// 2026-07-01 against each). An auth/permission failure (`unauthorized`/`denied`/`401`/`403`)
/// deliberately does **not** match, so it stays a propagated [`RemoteError::Transport`] — a missing
/// credential is **never silently read as "nothing is published"** (G2/VR-5), which would otherwise
/// let the immutability check be bypassed by an auth error.
fn classify_oras_failure(output: &std::process::Output, what: &str) -> RemoteError {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let low = stderr.to_ascii_lowercase();
    let not_found = low.contains("not found")
        || low.contains("name unknown")
        || low.contains("not known to registry")
        || low.contains("name_unknown")
        || low.contains("manifest unknown")
        || low.contains("manifest_unknown");
    let msg = format!(
        "`{what}` exited with {:?}: {}",
        output.status.code(),
        tail_lines(&stderr, 20)
    );
    if not_found {
        RemoteError::NotFound(msg)
    } else {
        RemoteError::Transport(msg)
    }
}

/// Preflight-check that `oras` is on `PATH` and runnable — a small, explicit "is the v0 transport
/// prerequisite present" probe (ADR-037 §5), usable ahead of a push/pull for a friendlier failure.
///
/// # Errors
/// [`RemoteError::ToolMissing`] if `oras` is absent; [`RemoteError::Transport`] if it is present but
/// `oras version` fails.
pub fn oras_preflight() -> Result<(), RemoteError> {
    let mut cmd = Command::new("oras");
    cmd.arg("version");
    let out = spawn_oras(cmd)?;
    require_success(&out, "oras version")
}

fn unique_temp_dir(tag: &str) -> Result<PathBuf, RemoteError> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("myc-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| io(&dir.display().to_string(), e))?;
    Ok(dir)
}

/// A best-effort `rm -rf` on drop — the push/pull staging directory is scratch, so cleanup failure
/// is not itself an error (the operation's own result already carries the meaningful outcome).
struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Find a `Digest:` line in `oras push`'s combined output (its convention for reporting the pushed
/// manifest digest).
fn extract_digest(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        line.trim()
            .strip_prefix("Digest:")
            .map(|d| d.trim().to_owned())
    })
}

/// The `oras`-CLI-driven [`OciTransport`] (ADR-037 §1/§4 — v0's transport). Every operation
/// preflights `oras`'s presence and surfaces a nonzero exit as an explicit [`RemoteError::Transport`]
/// (never a silent skip). Guarantee posture: **`Declared`** — correctness rests on `oras`'s own OCI
/// conformance, which this crate does not itself prove.
pub struct OrasTransport {
    /// Pass `--plain-http` to every `oras` invocation (for a local/dev HTTP registry, never a
    /// production GHCR target — [`parse_registry`] sets this only for `localhost`/`127.*` hosts).
    pub plain_http: bool,
}

impl OrasTransport {
    fn maybe_plain_http(&self, cmd: &mut Command) {
        if self.plain_http {
            cmd.arg("--plain-http");
        }
    }
}

impl OciTransport for OrasTransport {
    fn push(
        &self,
        reference: &str,
        config: &[u8],
        layers: &[ObjectBlob],
    ) -> Result<String, RemoteError> {
        oras_preflight()?;
        let tmp = unique_temp_dir("oras-push")?;
        let _guard = TempDirGuard(tmp.clone());

        let cfg_name = "config.densemap";
        let cfg_path = tmp.join(cfg_name);
        std::fs::write(&cfg_path, config).map_err(|e| io(&cfg_path.display().to_string(), e))?;
        for blob in layers {
            let p = tmp.join(blob.oci_title());
            std::fs::write(&p, &blob.bytes).map_err(|e| io(&p.display().to_string(), e))?;
        }

        let mut cmd = Command::new("oras");
        cmd.current_dir(&tmp);
        cmd.arg("push");
        self.maybe_plain_http(&mut cmd);
        cmd.arg(reference);
        cmd.arg("--artifact-type")
            .arg("application/vnd.mycelium.spore.v1");
        cmd.arg("--config")
            .arg(format!("{cfg_name}:application/vnd.mycelium.densemap.v1"));
        for blob in layers {
            cmd.arg(format!(
                "{}:application/vnd.mycelium.spore.object.v1",
                blob.oci_title()
            ));
        }

        let out = spawn_oras(cmd)?;
        require_success(&out, "oras push")?;
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        extract_digest(&text).ok_or_else(|| {
            RemoteError::Transport(format!(
                "oras push succeeded but reported no `Digest:` line — cannot confirm the manifest \
                 digest (G2); output tail:\n{}",
                tail_lines(&text, 30)
            ))
        })
    }

    fn pull(&self, reference: &str) -> Result<(Vec<u8>, PulledLayers), RemoteError> {
        oras_preflight()?;

        let mut cfg_cmd = Command::new("oras");
        cfg_cmd.arg("manifest").arg("fetch-config");
        self.maybe_plain_http(&mut cfg_cmd);
        cfg_cmd.arg(reference);
        let cfg_out = spawn_oras(cfg_cmd)?;
        // A missing manifest/tag is classified NotFound (an expected "not published" answer) vs a
        // real transport/auth failure — so resolve and the immutability pre-check distinguish them.
        if !cfg_out.status.success() {
            return Err(classify_oras_failure(
                &cfg_out,
                "oras manifest fetch-config",
            ));
        }
        let config = cfg_out.stdout;

        let tmp = unique_temp_dir("oras-pull")?;
        let _guard = TempDirGuard(tmp.clone());
        let mut pull_cmd = Command::new("oras");
        pull_cmd.arg("pull");
        self.maybe_plain_http(&mut pull_cmd);
        pull_cmd.arg(reference);
        pull_cmd.arg("-o").arg(&tmp);
        let pull_out = spawn_oras(pull_cmd)?;
        require_success(&pull_out, "oras pull")?;

        let mut layers = Vec::new();
        let rd = std::fs::read_dir(&tmp).map_err(|e| io(&tmp.display().to_string(), e))?;
        for entry in rd {
            let entry = entry.map_err(|e| io(&tmp.display().to_string(), e))?;
            let path = entry.path();
            if path.is_file() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_owned();
                let bytes = std::fs::read(&path).map_err(|e| io(&path.display().to_string(), e))?;
                layers.push((name, bytes));
            }
        }
        Ok((config, layers))
    }

    fn list_tags(&self, repo: &str) -> Result<Vec<String>, RemoteError> {
        oras_preflight()?;
        let mut cmd = Command::new("oras");
        cmd.arg("repo").arg("tags");
        self.maybe_plain_http(&mut cmd);
        cmd.arg(repo);
        let out = spawn_oras(cmd)?;
        // A missing repository is a NORMAL answer here ("no tags yet") — classify it as NotFound so
        // the immutability pre-check can proceed with a first publish; a real transport/auth failure
        // still propagates as Transport (never silently read as "no tags", G2).
        if !out.status.success() {
            return Err(classify_oras_failure(&out, "oras repo tags"));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect())
    }
}

/// An in-memory [`OciTransport`] test double — no `oras`/network involved, for pure integration
/// tests of the dense-map + verify-and-reconstruct pipeline. Guarantee posture: this is a **test
/// fixture**, not a claim about real OCI-registry behavior.
#[derive(Default)]
pub struct MemTransport {
    store: RefCell<HashMap<String, (Vec<u8>, PulledLayers)>>,
}

impl MemTransport {
    /// A fresh, empty in-memory registry double.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl OciTransport for MemTransport {
    fn push(
        &self,
        reference: &str,
        config: &[u8],
        layers: &[ObjectBlob],
    ) -> Result<String, RemoteError> {
        let layer_pairs: PulledLayers = layers
            .iter()
            .map(|b| (b.oci_title(), b.bytes.clone()))
            .collect();
        self.store
            .borrow_mut()
            .insert(reference.to_owned(), (config.to_vec(), layer_pairs));
        // A deterministic stand-in digest (not a real OCI manifest digest — this is a test double).
        Ok(format!("sha256:{}", blake3::hash(config).to_hex()))
    }

    fn pull(&self, reference: &str) -> Result<(Vec<u8>, PulledLayers), RemoteError> {
        self.store.borrow().get(reference).cloned().ok_or_else(|| {
            RemoteError::NotFound(format!(
                "no artifact pushed for reference {reference:?} in this in-memory transport"
            ))
        })
    }

    fn list_tags(&self, repo: &str) -> Result<Vec<String>, RemoteError> {
        let prefix = format!("{repo}:");
        Ok(self
            .store
            .borrow()
            .keys()
            .filter_map(|r| r.strip_prefix(prefix.as_str()).map(str::to_owned))
            .collect())
    }
}

// ─── registry target + publish/resolve ─────────────────────────────────────────────────────────

/// Where a `spore` registry operation routes — decided **once, from an explicit scheme, never
/// guessed** (ADR-037 §1/§5): a bare path is the M-732 local store; `ghcr://`/`oci://` select the
/// remote OCI backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryTarget {
    /// The M-732 local, file-based store.
    Local(PathBuf),
    /// The remote OCI/GHCR backend: `base` is the OCI repository namespace the `name` is joined
    /// under (`<base>/<name>:<version>`); `plain_http` selects unencrypted HTTP (only ever set for
    /// a `localhost`/`127.*` `oci://` host — never for `ghcr://`).
    Oci { base: String, plain_http: bool },
}

/// Parse a `--registry` value into a [`RegistryTarget`] (ADR-037 §1/§5): `ghcr://<owner>/<repo>` →
/// GHCR at `ghcr.io/<owner>/<repo>`; `oci://<host>[/<path>]` → that host (plain HTTP auto-selected
/// for `localhost`/`127.*`, so a local dev registry needs no extra flag); any other `<scheme>://` is
/// an explicit [`RemoteError::InvalidInput`] (an unrecognized scheme is never silently treated as a
/// local path); no scheme at all is the local store. The route is decided by the scheme alone —
/// never guessed from the target's reachability (G2).
///
/// # Errors
/// [`RemoteError::InvalidInput`] for an empty `ghcr://`/`oci://` authority, or an unrecognized
/// `<scheme>://`.
pub fn parse_registry(s: &str) -> Result<RegistryTarget, RemoteError> {
    if let Some(rest) = s.strip_prefix("ghcr://") {
        if rest.is_empty() {
            return Err(RemoteError::InvalidInput(
                "ghcr:// requires an owner/repo path, e.g. `ghcr://my-org/my-repo`".to_owned(),
            ));
        }
        return Ok(RegistryTarget::Oci {
            base: format!("ghcr.io/{rest}"),
            plain_http: false,
        });
    }
    if let Some(rest) = s.strip_prefix("oci://") {
        if rest.is_empty() {
            return Err(RemoteError::InvalidInput(
                "oci:// requires a host[/path], e.g. `oci://localhost:5000` or `oci://reg.example.com/ns`"
                    .to_owned(),
            ));
        }
        let host = rest.split('/').next().unwrap_or("");
        let plain_http =
            host == "localhost" || host.starts_with("localhost:") || host.starts_with("127.");
        return Ok(RegistryTarget::Oci {
            base: rest.to_owned(),
            plain_http,
        });
    }
    if let Some((scheme, _)) = s.split_once("://") {
        return Err(RemoteError::InvalidInput(format!(
            "unrecognized registry scheme `{scheme}://` — only `ghcr://` and `oci://` select the remote \
             backend; a bare path is the local store (the route is never guessed, ADR-037 §1/§5)"
        )));
    }
    Ok(RegistryTarget::Local(PathBuf::from(s)))
}

/// Reject a `name`/`version` that would not survive as an OCI reference component: empty, or
/// containing whitespace, `/`, or `:` (which would break `<repo>/<name>:<version>` parsing). This is
/// a **minimal, honest** check — it does not attempt to enforce the full OCI reference grammar;
/// anything past this is left for `oras`/the registry to report (never silently rewritten).
fn validate_oci_component(kind: &str, value: &str) -> Result<(), RemoteError> {
    let bad = value.is_empty()
        || value.chars().any(char::is_whitespace)
        || value.contains('/')
        || value.contains(':');
    if bad {
        return Err(RemoteError::InvalidInput(format!(
            "{kind} {value:?} is not a safe OCI reference component — it must be non-empty and contain \
             no whitespace, `/`, or `:` (never guessed or silently rewritten; G2)"
        )));
    }
    Ok(())
}

/// The receipt of a successful [`publish_remote`] — the never-silent EXPLAIN of what was pushed and
/// where (G2; no black box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemotePublishReceipt {
    /// The published package name.
    pub name: String,
    /// The published version label.
    pub version: String,
    /// The spore identity (ADR-003).
    pub spore_id: ContentHash,
    /// The OCI manifest digest the transport reported.
    pub manifest_digest: String,
    /// The full OCI reference (`<repo>:<tag>`) pushed to.
    pub reference: String,
}

/// **Publish** `spore` to the remote OCI backend `target` (must be [`RegistryTarget::Oci`]):
/// build the dense-map + objects from `project_dir` ([`build_dense_map`]), then push them via
/// `transport` to `<target.base>/<name>:<version>` (ADR-037 §2).
///
/// # Errors
/// [`RemoteError::InvalidInput`] if `target` is [`RegistryTarget::Local`] or `name`/`version` is not
/// a safe reference component; propagates [`build_dense_map`]'s [`RemoteError::Io`] /
/// [`RemoteError::Integrity`], and the `transport`'s own errors (e.g. [`RemoteError::ToolMissing`]).
pub fn publish_remote(
    target: &RegistryTarget,
    spore: &Spore,
    project_dir: &Path,
    name: &str,
    version: &str,
    transport: &dyn OciTransport,
) -> Result<RemotePublishReceipt, RemoteError> {
    let RegistryTarget::Oci { base, .. } = target else {
        return Err(RemoteError::InvalidInput(
            "publish_remote requires an `Oci` registry target — a `Local` target uses \
             `registry::publish` instead (the route is decided once, by scheme; ADR-037 §5)"
                .to_owned(),
        ));
    };
    validate_oci_component("name", name)?;
    validate_oci_component("version", version)?;

    let (dense_map, blobs) = build_dense_map(spore, project_dir)?;
    let config = encode_dense_map(&dense_map);
    let repo = format!("{base}/{name}");
    let reference = format!("{repo}:{version}");

    // Immutability pre-check (M-872) — best-effort parity with the local store's `Conflict` semantics
    // (M-732). A registry `name@version` is immutable: republishing a DIFFERENT spore under an
    // existing tag is refused; an identical re-publish is idempotent; a first publish proceeds.
    //
    // **Honest ceiling (Declared, VR-5):** OCI tags are server-side MUTABLE — this is a *client-side*
    // guard (a racing or hostile client could still overwrite the tag out from under us), so it is a
    // best-effort consistency check, **not** a proven server invariant. It is never claimed `Proven`.
    // Never-silent (G2): a real transport/auth error is propagated, never swallowed into "proceed".
    match transport.list_tags(&repo) {
        Ok(tags) if tags.iter().any(|t| t == version) => {
            // The tag already exists — compare identity by reading its recorded `spore_id`.
            let (existing_config, _existing_layers) = transport.pull(&reference)?;
            let existing = decode_dense_map(&existing_config)?;
            if existing.spore_id != spore.id {
                return Err(RemoteError::Conflict(format!(
                    "{name}@{version} is already published at {reference} with a DIFFERENT spore \
                     (existing spore_id={}, publishing {}) — a registry name@version is immutable; \
                     publish under a new version (ADR-003 / M-732 parity, G2). Note: OCI tags are \
                     server-side mutable, so this is a best-effort client-side guard (Declared), not \
                     a proven server invariant.",
                    existing.spore_id.as_str(),
                    spore.id.as_str(),
                )));
            }
            // Same `spore_id` → an identical re-publish is idempotent; fall through to a harmless
            // re-push (content-addressed, so the manifest digest is unchanged).
        }
        Ok(_) => {} // repo exists but this tag is absent → proceed
        Err(RemoteError::NotFound(_)) => {} // repo/tags don't exist yet (first publish) → proceed
        Err(e) => return Err(e), // a real transport/auth failure → never swallowed (G2)
    }

    let manifest_digest = transport.push(&reference, &config, &blobs)?;

    Ok(RemotePublishReceipt {
        name: name.to_owned(),
        version: version.to_owned(),
        spore_id: spore.id.clone(),
        manifest_digest,
        reference,
    })
}

/// The result of a successful [`resolve_remote`]: the concrete version selected and the
/// fetch-and-verified reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteResolved {
    /// The resolved name.
    pub name: String,
    /// The concrete version selected (after resolving `latest`/`*`).
    pub version: String,
    /// The verified, reconstructed spore.
    pub reconstructed: Reconstructed,
}

/// **Resolve** `name` at `constraint` against the remote OCI backend `target` (must be
/// [`RegistryTarget::Oci`]): `constraint` is an exact version, or `latest`/`*` (the highest tag by
/// [`crate::registry`]'s version-sort key, reused here — DRY). A range constraint (`^`/`~`/`>`/`<`/
/// `=`/a comma-separated list) is an explicit [`RemoteError::Unsupported`] — v0 never mis-resolves a
/// SemVer range it cannot honestly evaluate (the deferred ADR-018 work). Pulls via `transport`, then
/// [`decode_dense_map`]s the config and [`verify_and_reconstruct`]s against the fetched layers.
///
/// # Errors
/// [`RemoteError::InvalidInput`] if `target` is [`RegistryTarget::Local`] or the constraint is
/// empty; [`RemoteError::Unsupported`] for a range constraint; [`RemoteError::NotFound`] if no tag
/// matches; propagates `transport`'s errors and [`decode_dense_map`]/[`verify_and_reconstruct`]'s
/// integrity errors.
pub fn resolve_remote(
    target: &RegistryTarget,
    name: &str,
    constraint: &str,
    transport: &dyn OciTransport,
) -> Result<RemoteResolved, RemoteError> {
    let RegistryTarget::Oci { base, .. } = target else {
        return Err(RemoteError::InvalidInput(
            "resolve_remote requires an `Oci` registry target — a `Local` target uses \
             `registry::resolve` instead (the route is decided once, by scheme; ADR-037 §5)"
                .to_owned(),
        ));
    };
    validate_oci_component("name", name)?;
    let c = constraint.trim();
    if c.is_empty() {
        return Err(RemoteError::InvalidInput(format!(
            "{name}: an empty version constraint resolves nothing (it is never guessed)"
        )));
    }
    if c.starts_with(['^', '~', '>', '<', '=']) || c.contains(',') {
        return Err(RemoteError::Unsupported(format!(
            "version constraint {c:?} is a range — the remote backend resolves an exact version or \
             `latest`/`*` only; SemVer range resolution is the deferred ADR-018 work, not silently \
             approximated (VR-5)"
        )));
    }

    let repo = format!("{base}/{name}");
    let version = if c == "latest" || c == "*" {
        let mut tags = transport.list_tags(&repo)?;
        tags.sort_by_key(|v| crate::registry::version_key(v));
        tags.pop()
            .ok_or_else(|| RemoteError::NotFound(format!("{name}: no tags found at {repo}")))?
    } else {
        c.to_owned()
    };

    let reference = format!("{repo}:{version}");
    let (config, layers) = transport.pull(&reference)?;
    let dense_map = decode_dense_map(&config)?;
    let reconstructed = verify_and_reconstruct(dense_map, &layers)?;

    Ok(RemoteResolved {
        name: name.to_owned(),
        version,
        reconstructed,
    })
}
