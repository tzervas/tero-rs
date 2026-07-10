# tero-rs container image — wraps the mycelium-tero fronts (tero-index/tero-http/tero-mcp/
# tero-eval) built from this Cargo workspace. Primary consumer: tero-mcp (the MCP stdio front),
# run as the image entrypoint; the other three bins ship alongside it in /usr/local/bin.
#
# WHAT: two-stage build — a `rust` builder compiles the workspace in release mode, then only the
# four built binaries are copied into a slim Debian runtime image (no toolchain/source in the
# shipped image).
# WHY: the workspace pins rust-version = "1.96.1" (Cargo.toml [workspace.package]) as its MSRV;
# the `rust:1` tag tracks current stable, which satisfies that floor. No crate in mycelium-tero's
# own dependency graph (mycelium-core/mycelium-l1/mycelium-doc/mycelium-vsa) uses a nightly
# `#![feature(...)]` gate (verified: grep for `feature(` across those crates found none), so a
# stable-toolchain image builds it cleanly.
# WHY NOT distroless/scratch: the runtime binaries are plain dynamically-linked glibc executables
# (not statically linked musl); debian-slim keeps libc present with a small footprint rather than
# adding a musl target cross-compile step this pass.

FROM rust:1-bookworm AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p mycelium-tero

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/tero-mcp /usr/local/bin/tero-mcp
COPY --from=builder /build/target/release/tero-index /usr/local/bin/tero-index
COPY --from=builder /build/target/release/tero-http /usr/local/bin/tero-http
COPY --from=builder /build/target/release/tero-eval /usr/local/bin/tero-eval

# tero-mcp is a stdio MCP front (no listening port); it is the default entrypoint because it is
# the binary tero-mcp/cabal-devmelopner consume downstream. Override the command to run
# tero-index/tero-http/tero-eval instead: `docker run ghcr.io/tzervas/tero-rs:<tag> tero-http ...`.
ENTRYPOINT ["tero-mcp"]
