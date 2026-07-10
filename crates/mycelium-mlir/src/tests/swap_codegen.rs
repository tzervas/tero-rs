//! In-crate white-box tests for `swap_codegen.rs` (M-852; CLAUDE.md test-layout rule). These are
//! pure **emission** + **logic** checks (no toolchain): the `legal_pair` side-condition, the
//! `SwapCertMode` EXPLAIN record/comment, the never-silent refusals, and that the emitted IR carries
//! the dumpable cert-basis comment (RFC-0004 §6). The compiled-path differential (interp ≡ native,
//! both cert modes, M-210-checked) lives in `tests/swap_differential.rs`.

use crate::llvm::{emit_llvm_ir, emit_llvm_ir_with_swap_mode};
use crate::swap_codegen::{legal_pair, SwapCertMode, SwapExplain};
use mycelium_core::{ContentHash, Meta, Node, Payload, Provenance, Repr, Trit, Value};

fn binary(bits: Vec<bool>) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn ternary(trits: Vec<Trit>) -> Value {
    let m = trits.len() as u32;
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(trits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn policy() -> ContentHash {
    ContentHash::parse("blake3:round_trip_safe").unwrap()
}

fn swap_b_to_t(bits: Vec<bool>, m: u32) -> Node {
    Node::Swap {
        src: Box::new(Node::Const(binary(bits))),
        target: Repr::Ternary { trits: m },
        policy: policy(),
    }
}

// ─── legal_pair (the re-checked RFC-0002 §5 side-condition) ──────────────────────────────────────

/// `legal_pair` matches the published worked pairs: `(8,6)` legal (128 ≤ 364), `(8,4)` illegal
/// (128 > 40), `(4,4)` legal (8 ≤ 40), `(4,3)` legal (8 ≤ 13), `(8,3)` illegal (128 > 13).
#[test]
fn legal_pair_matches_the_side_condition() {
    assert!(legal_pair(8, 6), "(8,6) is legal: 2^7=128 ≤ (3^6−1)/2=364");
    assert!(!legal_pair(8, 4), "(8,4) is illegal: 128 > (3^4−1)/2=40");
    assert!(legal_pair(4, 4), "(4,4) is legal: 2^3=8 ≤ 40");
    assert!(legal_pair(4, 3), "(4,3) is legal: 8 ≤ (3^3−1)/2=13");
    assert!(!legal_pair(8, 3), "(8,3) is illegal: 128 > 13");
}

/// `legal_pair` is the **independent** re-implementation of `mycelium-cert::legal_pair` — it agrees
/// with the cert crate's verdict over a small grid (so the `Recheck` basis is the same side-condition,
/// just computed in this crate without importing cert).
#[test]
fn legal_pair_agrees_with_the_cert_crate() {
    for n in 1u32..=12 {
        for m in 1u32..=10 {
            assert_eq!(
                legal_pair(n, m),
                mycelium_cert::legal_pair(n, m),
                "legal_pair({n},{m}) disagrees with mycelium-cert"
            );
        }
    }
}

// ─── EXPLAIN record + the dumpable IR comment (RFC-0004 §6, no black box) ─────────────────────────

/// The emitted IR carries the dumpable swap cert-basis comment for **both** modes, naming the cert
/// source (never hidden — G2). The two modes record distinct cert sources.
#[test]
fn emitted_ir_records_the_cert_mode_and_source() {
    let prog = swap_b_to_t(vec![true, false, true, true, false, false, true, false], 6);
    let recheck = emit_llvm_ir_with_swap_mode(&prog, SwapCertMode::Recheck).unwrap();
    let reuse = emit_llvm_ir_with_swap_mode(&prog, SwapCertMode::ReuseInterp).unwrap();

    // The dumpable comment names the swap, the (n,m) pair, legal=true, the mode, and the source.
    assert!(
        recheck.contains("; swap") && recheck.contains("legal=true"),
        "recheck IR must carry the dumpable swap comment:\n{recheck}"
    );
    assert!(
        recheck.contains("cert-source=compile-time-rechecked"),
        "recheck IR must record the compile-time-rechecked cert source:\n{recheck}"
    );
    assert!(
        reuse.contains("cert-source=interp-carried"),
        "reuse IR must record the interp-carried cert source:\n{reuse}"
    );
    // The transcode itself (the value-preserving enc) is the same in both modes — the mode only
    // changes the recorded basis, not the emitted arithmetic.
    let strip = |s: &str| {
        s.lines()
            .filter(|l| !l.trim_start().starts_with("; swap"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(
        strip(&recheck),
        strip(&reuse),
        "the two cert modes must emit identical transcode IR (only the comment differs)"
    );
}

/// The default `emit_llvm_ir` uses the `Recheck` mode (the project default — compile-time re-check).
#[test]
fn default_emit_uses_recheck_mode() {
    let prog = swap_b_to_t(vec![false; 8], 6);
    let default = emit_llvm_ir(&prog).unwrap();
    let recheck = emit_llvm_ir_with_swap_mode(&prog, SwapCertMode::Recheck).unwrap();
    assert_eq!(default, recheck, "default emit must be the Recheck mode");
    assert!(default.contains("cert-source=compile-time-rechecked"));
}

/// `SwapCertMode` labels/sources are stable, distinct, and never empty (the EXPLAIN strings — G2).
#[test]
fn swap_cert_mode_labels_are_distinct_and_nonempty() {
    assert_ne!(
        SwapCertMode::Recheck.label(),
        SwapCertMode::ReuseInterp.label()
    );
    assert_ne!(
        SwapCertMode::Recheck.cert_source(),
        SwapCertMode::ReuseInterp.cert_source()
    );
    assert!(!SwapCertMode::Recheck.label().is_empty());
    assert!(!SwapCertMode::ReuseInterp.cert_source().is_empty());
    assert_eq!(SwapCertMode::default(), SwapCertMode::Recheck);
}

/// The `SwapExplain` is constructible and its fields round-trip the mode/source pairing (the
/// inspectable record an EXPLAIN consumer reads).
#[test]
fn swap_explain_pairs_mode_and_source() {
    let e = SwapExplain {
        src: "Binary { width: 8 }".into(),
        target: "Ternary { trits: 6 }".into(),
        width: 8,
        trits: 6,
        legal_pair: true,
        mode: SwapCertMode::ReuseInterp,
        cert_source: SwapCertMode::ReuseInterp.cert_source(),
        identity: false,
    };
    assert_eq!(e.cert_source, "interp-carried");
    assert!(!e.identity);
}

// ─── the IR transcode shape (no opaque pass — §6) ────────────────────────────────────────────────

/// The Binary→Ternary transcode emits the explicit decode (`zext`/`mul`/`add` for `bits_to_int`) and
/// the balanced-division encode (`srem`/`sdiv`/`select` for `int_to_trits`) — every step visible IR.
#[test]
fn b_to_t_emits_explicit_transcode_ir() {
    let prog = swap_b_to_t(vec![true, false, true, true, false, false, true, false], 6);
    let ir = emit_llvm_ir(&prog).unwrap();
    assert!(ir.contains("zext i32"), "bits_to_int zext missing:\n{ir}");
    assert!(ir.contains("srem i64"), "int_to_trits srem missing:\n{ir}");
    assert!(ir.contains("sdiv i64"), "int_to_trits sdiv missing:\n{ir}");
    assert!(
        ir.contains("select i1"),
        "balanced fold select missing:\n{ir}"
    );
    // The ternary output read-back uses the '-'(45)/'0'(48)/'+'(43) select chain.
    assert!(ir.contains("i32 45") && ir.contains("i32 43"));
}

/// The Ternary→Binary transcode emits the `dec` decode (`sext`/`mul`/`add` for `trits_to_int`) and
/// the range-check (`icmp slt`/`icmp sgt`/`or i1`) that drives the never-silent out-of-range read-back.
#[test]
fn t_to_b_emits_range_checked_transcode_ir() {
    let prog = Node::Swap {
        src: Box::new(Node::Const(ternary(vec![Trit::Zero, Trit::Pos, Trit::Neg]))),
        target: Repr::Binary { width: 4 },
        policy: policy(),
    };
    let ir = emit_llvm_ir(&prog).unwrap();
    assert!(ir.contains("sext i32"), "trits_to_int sext missing:\n{ir}");
    // The never-silent range check + the overflow read-back branch.
    assert!(
        ir.contains("icmp slt i64") && ir.contains("icmp sgt i64"),
        "range check missing:\n{ir}"
    );
    assert!(
        ir.contains("br i1") && ir.contains("ovf:"),
        "the never-silent out-of-range read-back branch must be emitted:\n{ir}"
    );
}

// ─── emission determinism ────────────────────────────────────────────────────────────────────────

#[test]
fn swap_emission_is_deterministic() {
    let prog = swap_b_to_t(vec![true, false, true, true, false, false, true, false], 6);
    assert_eq!(emit_llvm_ir(&prog), emit_llvm_ir(&prog));
}

// ─── never-silent refusals (G2) ──────────────────────────────────────────────────────────────────

/// An **illegal pair** is refused at compile time in `Recheck` mode (`UnsupportedNode`) — never
/// emitted (VR-5/G2). The `emit_llvm_ir` default (Recheck) surfaces the refusal during lowering.
#[test]
fn illegal_pair_is_refused_in_recheck_emit() {
    let prog = swap_b_to_t(vec![true, false, true, true, false, false, true, false], 4); // (8,4) illegal
    match emit_llvm_ir(&prog) {
        Err(crate::llvm::AotError::UnsupportedNode(msg)) => {
            assert!(
                msg.contains("legal pair") || msg.contains("recheck"),
                "refusal must name the side-condition; got: {msg}"
            );
        }
        other => panic!("Recheck emit must refuse the illegal pair (8,4), got {other:?}"),
    }
}

/// An identity swap (same `Repr`) lowers to a pass-through (no transcode), recorded as `identity` in
/// the dumpable comment — still never silent (the comment is present).
#[test]
fn identity_swap_passes_through_with_an_explain_comment() {
    let prog = Node::Swap {
        src: Box::new(Node::Const(binary(vec![true, false, true, false]))),
        target: Repr::Binary { width: 4 },
        policy: policy(),
    };
    let ir = emit_llvm_ir(&prog).unwrap();
    assert!(
        ir.contains("identity"),
        "the identity swap must record the identity comment:\n{ir}"
    );
    // No balanced-division transcode is emitted for an identity (the lane passes through).
    assert!(
        !ir.contains("srem i64"),
        "an identity swap must not emit a transcode:\n{ir}"
    );
}

/// A binary width too wide for the i64 transcode is refused **never-silently** (G2/VR-5) rather than
/// emit a silently-wrong transcode — the native-path width bound (`check_i64_width`), which guards the
/// i64 path in **both** cert modes. This matters most in `ReuseInterp` mode: there `check_legal` does
/// **not** refuse an illegal pair, so an over-wide `(64, 2)` would otherwise reach the i64 decode and
/// overflow `1i64 << 64`. The width bound catches it explicitly. (A *legal* pair always has `n ≤ 61`
/// and `m ≤ 39` with the i64 `max_magnitude`, so this bound only ever fires on an illegal/over-wide
/// pair — but firing it is what keeps the i64 path sound, never a silent overflow.)
#[test]
fn over_wide_pair_is_refused_not_silently_transcoded() {
    use crate::llvm::emit_llvm_ir_with_swap_mode;
    // 64 bits → 2 trits: an illegal, over-wide pair. In ReuseInterp the legal-pair re-check does not
    // refuse, so the i64 width bound must catch n=64 > 62 — never a silent `1i64 << 64` overflow.
    let prog = swap_b_to_t(vec![false; 64], 2);
    assert!(!mycelium_cert::legal_pair(64, 2), "(64,2) is illegal");
    match emit_llvm_ir_with_swap_mode(&prog, SwapCertMode::ReuseInterp) {
        Err(crate::llvm::AotError::UnsupportedNode(msg)) => {
            assert!(
                msg.contains("i64") && (msg.contains("62") || msg.contains("width")),
                "the refusal must name the i64 transcode width bound; got: {msg}"
            );
        }
        other => panic!("an over-wide (64,2) swap must be refused in ReuseInterp, got {other:?}"),
    }
    // In the default Recheck mode the illegal pair is refused even earlier (the legal-pair re-check),
    // so it is still never silently transcoded.
    assert!(matches!(
        emit_llvm_ir(&prog),
        Err(crate::llvm::AotError::UnsupportedNode(_))
    ));
}

/// An **in-i64-bound illegal pair** in `ReuseInterp` mode emits the `enc` transcode IR **plus** a
/// never-silent final-quotient overflow flag — so a value outside the ternary range produces an
/// explicit `AotError::Overflow` rather than silently wrong trits (G2/SC-3; M-852). The IR contains
/// the `icmp ne` final-quotient check (the never-silent guard). Pair `(8, 4)` is illegal:
/// `2^7 = 128 > (3^4−1)/2 = 40`. The width guard (`check_i64_width`) does **not** block it (8 ≤ 62,
/// 4 ≤ 39), so this test exercises the final-quotient path (not `check_i64_width`).
#[test]
fn in_i64_bound_illegal_pair_enc_emits_final_quotient_overflow_flag() {
    use crate::llvm::emit_llvm_ir_with_swap_mode;
    // (8,4): 8 ≤ 62 and 4 ≤ 39 — passes check_i64_width. Illegal (128 > 40) but in-i64-bound.
    let prog = swap_b_to_t(vec![true, false, true, true, false, false, true, false], 4); // bit pattern = −78
    assert!(!mycelium_cert::legal_pair(8, 4), "(8,4) is illegal");
    // In Recheck mode the illegal pair is refused at compile time (before any IR is emitted).
    match emit_llvm_ir(&prog) {
        Err(crate::llvm::AotError::UnsupportedNode(_)) => { /* expected */ }
        other => panic!("Recheck must refuse (8,4) before any IR, got {other:?}"),
    }
    // In ReuseInterp mode the IR is emitted (the pair's legality is not re-checked at compile time),
    // and the emitted IR carries the never-silent final-quotient check (`icmp ne i64 …, 0`) that
    // catches an out-of-range value at runtime. For the in-range value −78 (which fits in T_4:
    // |−78| = 78 > 40 — actually also out of range for T_4; max_magnitude(4)=40), the runtime
    // would produce AotError::Overflow (the read-back sentinel). The key assertion here is that
    // the IR is emitted (no UnsupportedNode) and contains the overflow guard.
    match emit_llvm_ir_with_swap_mode(&prog, SwapCertMode::ReuseInterp) {
        Ok(ir) => {
            assert!(
                ir.contains("; swap") && ir.contains("srem i64"),
                "ReuseInterp must emit the transcode IR (not refuse at compile time):\n{ir}"
            );
            assert!(
                ir.contains("icmp ne i64"),
                "the enc direction must emit the never-silent final-quotient overflow guard \
                 (G2/SC-3; M-852):\n{ir}"
            );
        }
        Err(crate::llvm::AotError::UnsupportedNode(msg)) => {
            panic!("ReuseInterp must not refuse an in-i64-bound pair at compile time; got: {msg}")
        }
        Err(other) => panic!("unexpected error for (8,4) in ReuseInterp: {other}"),
    }
}

/// The width bound is `>` not `>=`: a binary width of **exactly** `MAX_BINARY_WIDTH_I64 = 62` is
/// **accepted** (it is the widest the i64 transcode handles soundly — `1i64 << 62` fits i64), not
/// refused. Exercised via `ReuseInterp` (an illegal but in-i64-bound pair `(62, 2)` reaches the
/// transcode). Pins the off-by-one: a `> → >=` mutation would wrongly refuse width 62.
#[test]
fn width_at_the_i64_bound_is_accepted_not_refused() {
    use crate::llvm::emit_llvm_ir_with_swap_mode;
    // Binary side: width 62 == MAX_BINARY_WIDTH_I64 must be accepted (pins `>` not `>=`).
    let bw = swap_b_to_t(vec![false; 62], 2); // (62,2) illegal but in-i64-bound
    let ir = emit_llvm_ir_with_swap_mode(&bw, SwapCertMode::ReuseInterp)
        .expect("binary width == 62 is exactly at the i64 bound and must lower (not refused)");
    assert!(
        ir.contains("; swap") && ir.contains("srem i64"),
        "the width-62 swap must emit the transcode (accepted at the boundary):\n{ir}"
    );
    // Ternary side: trits 39 == MAX_TERNARY_WIDTH_I64 must be accepted (pins the ternary `>` bound).
    let tw = swap_b_to_t(vec![false; 2], 39); // (2,39) in-i64-bound; enc to 39 trits
    let ir_t = emit_llvm_ir_with_swap_mode(&tw, SwapCertMode::ReuseInterp)
        .expect("ternary width == 39 is exactly at the i64 bound and must lower (not refused)");
    assert!(
        ir_t.contains("; swap"),
        "the trits-39 swap must emit the transcode (accepted at the boundary):\n{ir_t}"
    );
}
