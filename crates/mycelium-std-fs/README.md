# mycelium-std-fs

> Filesystem access over affine handles for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.fs` is the Ring-2/Tier-B filesystem surface. It provides an `Fs` context struct, an
affine `File` handle (LR-8: single-consumption by Rust move semantics), a `DirIter` for
directory traversal, and `OpenOptions` for explicit open-mode declaration. No op silently
creates, truncates, overwrites, or partially writes — every failure is an explicit
`Err(FsErr)`. The in-memory substrate (`Fs::in_memory()` backed by `InMemoryFs`) is fully
testable without OS facilities; `RealFs` is deferred to `std-sys` (M-541).

## Key items

- `Fs` — filesystem context; constructed via `Fs::in_memory()` or (future) `Fs::real()`.
- `File` — affine file handle (LR-8); consumed exactly once by `read_all`/`write`/`close`.
- `DirIter` — fallible directory iterator returning `(Path, FileKind)` entries.
- `OpenOptions` — explicit open-mode builder (`read`, `write`, `create`, `truncate`, `append`).
- `Path` — UTF-8 path newtype; construction fails on non-UTF-8 input.
- `Metadata` — file metadata (size, kind, permissions, timestamps).
- `FileKind` — `File`, `Dir`, `Symlink`, or `Other`.
- `Permissions` — readable/writable permission flags.
- `FsErr` — explicit filesystem error; `ErrnoClass` classifies the underlying OS error kind.

## Design references

- RFC-0016 (core + standard library contract, C1–C6); LR-8 (affine single-consumption handles).
- Spec: `docs/spec/stdlib/fs.md` (M-528).

## Role in the workspace

Filesystem I/O layer for the stdlib; builds on `std.io` `Source`/`Sink` abstractions. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-fs).
