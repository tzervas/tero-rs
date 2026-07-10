//! RFC-0041 §4.7/§5 — the guard-hole **census** (W0 safety net; RR-29 guard-hole inventory turned
//! into a tracked failing test for this crate's hole).
//!
//! **`mycelium-core` is FROZEN (DN-56/M-969, KC-3).** This test lives ONLY in the black-box `tests/`
//! directory — no `src/` file is touched, not even to register a test submodule (the W0 leaf brief
//! for this hole is explicit: "for `write_canon`, use ONLY a black-box `tests/` file"). The private
//! `write_canon` (`crates/mycelium-core/src/lower.rs:212`) is reached only through the public
//! [`mycelium_core::lower::format`] wrapper.
//!
//! Real repro: builds a genuinely deep [`Node`] and calls [`format`]. Rust's default stack-overflow
//! handler aborts the process directly (never through panic/unwind), so this is not
//! `catch_unwind`-able — the test stays `#[ignore = "Wn"]`d; running it for real would crash the
//! whole test binary.

use mycelium_core::lower::format;
use mycelium_core::{ContentHash, CtorRef, Meta, Node, Payload, Provenance, Repr, Value};

/// A right-nested `Node::Construct` chain, `n` deep.
fn deep_construct(n: usize) -> Node {
    let byte = Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        Meta::exact(Provenance::Root),
    )
    .expect("a well-formed Binary{8} const");
    let ctor = CtorRef::new(
        ContentHash::parse("blake3:round_trip_safe").expect("a well-formed content hash"),
        0,
    );
    let mut acc = Node::Const(byte);
    for _ in 0..n {
        acc = Node::Construct {
            ctor: ctor.clone(),
            args: vec![acc],
        };
    }
    acc
}

/// Hole: `write_canon` (`crates/mycelium-core/src/lower.rs:212`), reached via the public
/// [`format`] wrapper (`lower.rs:204`).
///
/// **Honesty (FLAG, VR-5):** `format` returns a plain `String` — infallible today, so this test
/// cannot assert a "clean refusal". It constructs the real repro (the call itself, if unignored on a
/// large enough `n`, is the SIGABRT) and documents that this dump/debug renderer is expected to be
/// routed through the shared work-step budget — via the §6 within-freeze behavior-preserving-hardening
/// channel, since `write_canon` lives in the **frozen core**. Deferred from W1 to the frozen-core wave
/// (W3) precisely because that edit requires the maintainer checkpoint (a trusted-base change).
#[test]
#[ignore = "W3"] // RFC-0041 §6 within-freeze channel: write_canon is frozen mycelium-core — deferred W1→W3 (maintainer checkpoint before trusted-base edits).
fn write_canon_deep_construct_chain() {
    let deep = deep_construct(200_000);
    let _ = format(&deep);
}
