//! `spore` — packaging & publishing CLI (M-368/M-732/M-871; contract
//! `docs/spec/Spore-Build-and-Publish-Contract.md`, remote backend ADR-037).
//!
//! Builds a content-addressed `spore` from a `mycelium-proj.toml` project (ADR-013) and
//! publishes/resolves it against a content-addressed registry: a **local** file store (M-732) or
//! the **remote OCI/GHCR** dense-map backend (M-871/E26-1, ADR-037) — selected once, by an explicit
//! `--registry` scheme, never guessed. Identity is the code+deps DAG (ADR-003); metadata is not
//! identity. A missing/ambiguous input is an explicit error — **no partial artifact** is ever
//! written, and a registry never silently overwrites or mis-resolves (G2).
//!
//! ```text
//! spore build    [--config <manifest>] [-o <out>]                                # build + write the spore descriptor
//! spore explain  [--config <manifest>]                                           # the identity receipt; write nothing
//! spore publish  --registry <dir|oci://…|ghcr://…> [--config <manifest>]         # publish (local store or remote OCI/GHCR)
//!                [--name <n>] [--version <v>]
//! spore resolve  <name> <version|latest> --registry <dir|oci://…|ghcr://…> [-o <out>]   # fetch a hash-verified artifact
//! ```
//!
//! `--registry` scheme routing (ADR-037 §1/§5): a bare path is the local store; `ghcr://<owner>/<repo>`
//! and `oci://<host>[/<path>]` route to the remote OCI backend (driven by the `oras` CLI — absent or
//! failing is an explicit, actionable error, never a silent skip).
//!
//! Exit codes: 0 ok · 2 manifest error · 3 publish-input · 4 not-found · 5 integrity · 6 conflict ·
//! 64 usage/unsupported · 66 I/O · 69 tool missing (`oras`) · 74 transport error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mycelium_proj::parse_manifest;
use mycelium_spore::remote::{self, RegistryTarget};
use mycelium_spore::{build_spore, explain, registry, Spore};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         spore build   [--config <manifest>] [-o <out>]\n  \
         spore explain [--config <manifest>]\n  \
         spore publish --registry <dir|oci://…|ghcr://…> [--config <manifest>] [--name <n>] [--version <v>]\n  \
         spore resolve <name> <version|latest> --registry <dir|oci://…|ghcr://…> [-o <out>]"
    );
    ExitCode::from(64)
}

/// The flags shared across subcommands, parsed once.
#[derive(Default)]
struct Opts {
    config: Option<String>,
    out: Option<String>,
    registry: Option<String>,
    name: Option<String>,
    version: Option<String>,
    positionals: Vec<String>,
}

fn parse_opts(mut args: std::env::Args) -> Option<Opts> {
    let mut o = Opts::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => o.config = Some(args.next()?),
            "-o" => o.out = Some(args.next()?),
            "--registry" => o.registry = Some(args.next()?),
            "--name" => o.name = Some(args.next()?),
            "--version" => o.version = Some(args.next()?),
            s if s.starts_with('-') => return None, // an unknown flag is a usage error, never ignored
            _ => o.positionals.push(a),
        }
    }
    Some(o)
}

fn main() -> ExitCode {
    let mut args = std::env::args();
    let _argv0 = args.next();
    let Some(cmd) = args.next() else {
        return usage();
    };
    let Some(opts) = parse_opts(args) else {
        return usage();
    };

    match cmd.as_str() {
        "build" | "explain" | "publish" | "resolve" => {
            // Reject irrelevant flags / stray positionals per subcommand before dispatch — the
            // never-ignored posture (G2): an input a subcommand does not use is a usage error.
            if let Err(code) = validate_opts(&cmd, &opts) {
                return code;
            }
            if cmd == "resolve" {
                run_resolve(&opts)
            } else {
                run_with_spore(&cmd, &opts)
            }
        }
        _ => usage(),
    }
}

/// Reject inputs a subcommand does not use — never silently ignored (G2). Each subcommand declares
/// exactly the flags it accepts and whether it takes positional arguments; anything else (an
/// irrelevant flag like `spore explain -o out`, or a stray positional like `spore build garbage`) is
/// a usage error rather than a silent no-op.
fn validate_opts(cmd: &str, opts: &Opts) -> Result<(), ExitCode> {
    let (allowed, takes_positionals): (&[&str], bool) = match cmd {
        "explain" => (&["--config"], false),
        "build" => (&["--config", "-o"], false),
        "publish" => (&["--config", "--registry", "--name", "--version"], false),
        "resolve" => (&["--registry", "-o"], true),
        _ => (&[], false),
    };
    let present = [
        ("--config", opts.config.is_some()),
        ("-o", opts.out.is_some()),
        ("--registry", opts.registry.is_some()),
        ("--name", opts.name.is_some()),
        ("--version", opts.version.is_some()),
    ];
    let bad: Vec<&str> = present
        .iter()
        .filter(|(flag, set)| *set && !allowed.contains(flag))
        .map(|(flag, _)| *flag)
        .collect();
    if !bad.is_empty() {
        eprintln!(
            "spore: `{cmd}` does not accept {} — irrelevant flag(s) are never silently ignored (G2)",
            bad.join(", ")
        );
        return Err(usage());
    }
    if !takes_positionals && !opts.positionals.is_empty() {
        eprintln!(
            "spore: `{cmd}` takes no positional arguments, got: {}",
            opts.positionals.join(" ")
        );
        return Err(usage());
    }
    Ok(())
}

/// The subcommands that first build a spore from a manifest (`build`/`explain`/`publish`). Argument
/// relevance is validated up-front by [`validate_opts`] (irrelevant flags / stray positionals are
/// already rejected before this runs).
fn run_with_spore(cmd: &str, opts: &Opts) -> ExitCode {
    let manifest_path = opts
        .config
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("mycelium-proj.toml"));
    let project_dir = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let text = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("spore: io-error: {}: {e}", manifest_path.display());
            return ExitCode::from(66);
        }
    };
    let manifest = match parse_manifest(&text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("spore: manifest-error: {}: {e}", manifest_path.display());
            return ExitCode::from(2);
        }
    };
    let spore = match build_spore(&manifest, &project_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("spore: {e}");
            return ExitCode::from(e.exit_code());
        }
    };

    match cmd {
        "explain" => {
            print!("{}", explain(&spore));
            ExitCode::SUCCESS
        }
        "build" => emit_build(&spore, opts.out.as_deref()),
        "publish" => run_publish(&spore, &project_dir, opts),
        _ => usage(),
    }
}

/// `spore publish` — publish the built spore under `name@version`. The `--registry` value's scheme
/// decides the backend, once, never guessed (ADR-037 §1/§5): a bare path is the M-732 local store;
/// `ghcr://`/`oci://` route to the remote OCI/GHCR backend (ADR-037).
fn run_publish(spore: &Spore, project_dir: &Path, opts: &Opts) -> ExitCode {
    let Some(registry_str) = opts.registry.as_deref() else {
        eprintln!("spore: usage: publish requires --registry <dir|oci://…|ghcr://…>");
        return ExitCode::from(64);
    };
    let target = match remote::parse_registry(registry_str) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("spore: {e}");
            return ExitCode::from(e.exit_code());
        }
    };
    match &target {
        RegistryTarget::Local(dir) => run_publish_local(spore, dir, opts),
        RegistryTarget::Oci { .. } => run_publish_remote(spore, project_dir, &target, opts),
    }
}

/// The `name`/`version` a publish runs under: `--name`/`--version` win, else the manifest's; the
/// version is never guessed (ADR-003) — a missing version is an explicit publish-input error.
fn publish_name_version(spore: &Spore, opts: &Opts) -> Result<(String, String), ExitCode> {
    let name = opts.name.clone().unwrap_or_else(|| spore.name.clone());
    let Some(version) = opts.version.clone().or_else(|| spore.version.clone()) else {
        eprintln!(
            "spore: publish-input-error: no version to publish under — pass --version or set \
             [project].version (it is never guessed; ADR-003)"
        );
        return Err(ExitCode::from(3));
    };
    Ok((name, version))
}

/// `spore publish --registry <dir>` — the M-732 local, file-based store.
fn run_publish_local(spore: &Spore, registry_dir: &Path, opts: &Opts) -> ExitCode {
    let (name, version) = match publish_name_version(spore, opts) {
        Ok(nv) => nv,
        Err(code) => return code,
    };
    let descriptor = explain(spore).into_bytes();
    match registry::publish(registry_dir, spore, &descriptor, &name, &version) {
        Ok(r) => {
            let state = if r.already_present {
                "already present"
            } else {
                "published"
            };
            eprintln!(
                "spore: {state} {name}@{version}\n  spore_id: {}\n  artifact: {}\n  object:   {}",
                r.spore_id.as_str(),
                r.artifact.as_str(),
                r.object_path.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("spore: {e}");
            ExitCode::from(e.exit_code())
        }
    }
}

/// `spore publish --registry oci://…|ghcr://…` — the ADR-037 remote OCI/GHCR backend, driven by
/// `oras` ([`remote::OrasTransport`]). `oras` absent or failing is an explicit, actionable error
/// (never a silent skip — ADR-037 §5).
fn run_publish_remote(
    spore: &Spore,
    project_dir: &Path,
    target: &RegistryTarget,
    opts: &Opts,
) -> ExitCode {
    let (name, version) = match publish_name_version(spore, opts) {
        Ok(nv) => nv,
        Err(code) => return code,
    };
    let plain_http = matches!(
        target,
        RegistryTarget::Oci {
            plain_http: true,
            ..
        }
    );
    let transport = remote::OrasTransport { plain_http };
    match remote::publish_remote(target, spore, project_dir, &name, &version, &transport) {
        Ok(r) => {
            eprintln!(
                "spore: published {name}@{version}\n  reference:        {}\n  manifest digest:  {}\n  spore_id:         {}",
                r.reference,
                r.manifest_digest,
                r.spore_id.as_str()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("spore: {e}");
            ExitCode::from(e.exit_code())
        }
    }
}

/// `spore resolve <name> <version|latest>` — fetch a hash-verified artifact. The `--registry`
/// scheme routes to the local store or the remote OCI/GHCR backend, same as `publish` (ADR-037).
fn run_resolve(opts: &Opts) -> ExitCode {
    let Some(registry_str) = opts.registry.as_deref() else {
        eprintln!("spore: usage: resolve requires --registry <dir|oci://…|ghcr://…>");
        return ExitCode::from(64);
    };
    let [name, constraint] = match opts.positionals.as_slice() {
        [n, c] => [n.clone(), c.clone()],
        _ => {
            eprintln!(
                "spore: usage: resolve <name> <version|latest> --registry <dir|oci://…|ghcr://…> [-o <out>]"
            );
            return ExitCode::from(64);
        }
    };
    let target = match remote::parse_registry(registry_str) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("spore: {e}");
            return ExitCode::from(e.exit_code());
        }
    };
    match &target {
        RegistryTarget::Local(dir) => run_resolve_local(dir, &name, &constraint, opts),
        RegistryTarget::Oci { .. } => run_resolve_remote(&target, &name, &constraint, opts),
    }
}

/// `spore resolve --registry <dir>` — the M-732 local, file-based store.
fn run_resolve_local(registry_dir: &Path, name: &str, constraint: &str, opts: &Opts) -> ExitCode {
    match registry::resolve(registry_dir, name, constraint) {
        Ok(r) => {
            eprintln!(
                "spore: resolved {name}@{} (spore_id {}, artifact {})",
                r.version,
                r.spore_id.as_str(),
                r.artifact.as_str()
            );
            match opts.out.as_deref() {
                Some(path) => match std::fs::write(path, &r.bytes) {
                    Ok(()) => {
                        eprintln!("spore: wrote {path}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("spore: io-error: {path}: {e}");
                        ExitCode::from(66)
                    }
                },
                None => {
                    // The descriptor is UTF-8 text (the explain receipt); stream it to stdout.
                    print!("{}", String::from_utf8_lossy(&r.bytes));
                    ExitCode::SUCCESS
                }
            }
        }
        Err(e) => {
            eprintln!("spore: {e}");
            ExitCode::from(e.exit_code())
        }
    }
}

/// `spore resolve --registry oci://…|ghcr://…` — the ADR-037 remote OCI/GHCR backend. With `-o
/// <dir>`, materializes the fetch-and-verified source tree under `<dir>` (plus a
/// `<dir>/mycelium-densemap` receipt); without `-o`, prints the verified receipt to stderr only —
/// never writes a partial tree (G2).
fn run_resolve_remote(
    target: &RegistryTarget,
    name: &str,
    constraint: &str,
    opts: &Opts,
) -> ExitCode {
    let plain_http = matches!(
        target,
        RegistryTarget::Oci {
            plain_http: true,
            ..
        }
    );
    let transport = remote::OrasTransport { plain_http };
    match remote::resolve_remote(target, name, constraint, &transport) {
        Ok(r) => {
            let obj_count = r.reconstructed.sources.len();
            match opts.out.as_deref() {
                Some(dir) => match materialize(&r.reconstructed, Path::new(dir)) {
                    Ok(()) => {
                        eprintln!(
                            "spore: resolved {name}@{} (spore_id {}, {obj_count} object(s)) -> {dir}",
                            r.version,
                            r.reconstructed.spore_id.as_str()
                        );
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("spore: {e}");
                        ExitCode::from(e.exit_code())
                    }
                },
                None => {
                    eprintln!(
                        "spore: resolved {name}@{} (spore_id {}, {obj_count} object(s)) — pass -o <dir> to materialize",
                        r.version,
                        r.reconstructed.spore_id.as_str()
                    );
                    ExitCode::SUCCESS
                }
            }
        }
        Err(e) => {
            eprintln!("spore: {e}");
            ExitCode::from(e.exit_code())
        }
    }
}

/// Write a [`remote::Reconstructed`] source tree under `out_dir` (creating parent directories) plus
/// its dense-map receipt at `<out_dir>/mycelium-densemap`. Every `rel_path` is validated by
/// [`safe_join`] before any write — a resolved object naming an absolute path or a `..` component
/// never escapes `out_dir` (G2, the same traversal spirit as `registry::safe_component`).
fn materialize(
    r: &mycelium_spore::remote::Reconstructed,
    out_dir: &Path,
) -> Result<(), mycelium_spore::remote::RemoteError> {
    use mycelium_spore::remote::RemoteError;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| RemoteError::Io(format!("{}: {e}", out_dir.display())))?;
    for (rel_path, bytes) in &r.sources {
        let target = safe_join(out_dir, rel_path)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| RemoteError::Io(format!("{}: {e}", parent.display())))?;
        }
        std::fs::write(&target, bytes)
            .map_err(|e| RemoteError::Io(format!("{}: {e}", target.display())))?;
    }
    let dm_bytes = mycelium_spore::remote::encode_dense_map(&r.dense_map);
    let dm_path = out_dir.join("mycelium-densemap");
    std::fs::write(&dm_path, dm_bytes)
        .map_err(|e| RemoteError::Io(format!("{}: {e}", dm_path.display())))?;
    Ok(())
}

/// Join `root` with the forward-slashed relative path `rel`, refusing any component that would
/// escape `root` (absolute path, `.`/`..`, an embedded NUL, or an empty component) — never silently
/// joined (G2), same traversal-refusal spirit as `registry::safe_component`.
fn safe_join(root: &Path, rel: &str) -> Result<PathBuf, mycelium_spore::remote::RemoteError> {
    use mycelium_spore::remote::RemoteError;
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\0') {
        return Err(RemoteError::Integrity(format!(
            "resolved object path {rel:?} is not a safe relative path — refusing to write outside {} (G2)",
            root.display()
        )));
    }
    let mut out = root.to_path_buf();
    for comp in rel.split('/') {
        if comp.is_empty() || comp == "." || comp == ".." {
            return Err(RemoteError::Integrity(format!(
                "resolved object path {rel:?} contains an unsafe component ({comp:?}) — refusing to \
                 write outside {} (G2)",
                root.display()
            )));
        }
        out.push(comp);
    }
    Ok(out)
}

/// Emit the spore descriptor (the named-provisional v0 encoding; M-368 §9.1) — the EXPLAIN body prefixed
/// with the identity line, written to `-o <out>` or stdout. (The R2 wire-schema supersedes this.)
fn emit_build(spore: &Spore, out: Option<&str>) -> ExitCode {
    let descriptor = explain(spore);
    match out {
        Some(path) => match std::fs::write(path, &descriptor) {
            Ok(()) => {
                eprintln!("spore: wrote {} ({})", path, spore.id.as_str());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("spore: io-error: {path}: {e}");
                ExitCode::from(66)
            }
        },
        None => {
            print!("{descriptor}");
            ExitCode::SUCCESS
        }
    }
}
