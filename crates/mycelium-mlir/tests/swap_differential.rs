//! M-852 — the **`Swap`-node native-codegen differential** (E25-1; ADR-034; RFC-0002 §3/§4;
//! RFC-0004 §3/§6; NFR-7; VR-4; the M-210 shared checker).
//!
//! Extends the M-302 interp↔native differential to the **`Swap` node** — the one Repr-changing node
//! (WF1) — for the certified **binary↔ternary** swap class. The programs run under:
//!
//! 1. the **reference interpreter** with the certified [`BinaryTernarySwapEngine`] (the trusted base —
//!    the *reference the native swap must be observationally equivalent to*), and
//! 2. the **direct-LLVM** backend ([`mycelium_mlir::compile_and_run`] / its `_with_swap_mode`
//!    variant; `swap_codegen.rs`) in **both** cert modes (`Recheck` DEFAULT, `ReuseInterp` OPT-IN),
//!
//! and each pair is **validated through the single shared M-210 checker**
//! ([`RefinementRelation::ObservationalEquiv`]) — a deliberately divergent lowering is caught, so a
//! passing differential is meaningful, not vacuous.
//!
//! **M-856 widens the third (MLIR-dialect) leg to a real lowering** of the certified
//! binary↔ternary class (+ same-`Repr` identity), always under the `Recheck` cert mode (the
//! `ReuseInterp` opt-in mode is not wired in the dialect path). So `b_to_t_three_way_including_mlir_dialect`
//! / `t_to_b_three_way_including_mlir_dialect` are a genuine three-way for this class, each pair
//! M-210-checked. Dense/VSA targets and an **illegal** `(n,m)` pair stay explicit dialect
//! refusals ([`swap_to_dense_is_refused_by_the_mlir_dialect_path`] /
//! [`swap_illegal_pair_is_refused_by_the_mlir_dialect_path`]) — never silently lowered — and the
//! out-of-range `dec` direction is refused the same way as direct-LLVM
//! ([`swap_out_of_range_dec_is_refused_by_the_mlir_dialect_path`]).
//!
//! **Out-of-range (`dec` partiality, RFC-0002 §4 P4).** A `Ternary → Binary` swap whose value leaves
//! `B_n` is refused **never-silently** by both paths: the interpreter errors (`SwapError::OutOfRange`
//! via `EvalError`), the native path returns `AotError::Overflow` (the read-back sentinel) — never a
//! silent wrap (SC-3/G2). [`out_of_range_dec_is_refused_non_silently`] asserts the parity.
//!
//! **Illegal pair (compile-time re-check, `Recheck` mode).** An illegal `(n,m)` pair is refused at
//! compile time by the `Recheck` mode's independent side-condition re-check (`AotError::UnsupportedNode`),
//! exactly where the interpreter's certified engine raises `SwapError::IllegalPair` — the compile-time
//! re-check is an independent basis, not a trust of the interpreter's cert
//! ([`illegal_pair_is_refused_at_compile_time_in_recheck_mode`]).
//!
//! **Toolchain skip.** The direct-LLVM path needs `llc`/`clang`; where absent it returns
//! `ToolchainMissing` and the path **skips** (the house idiom) — never a false failure.
//!
//! **Guarantee:** `Empirical` — the differential is empirical evidence the native swap agrees with the
//! trusted interpreter over the corpus, in both cert modes; never upgraded to `Proven` without a
//! checked proof object linked into codegen (VR-5).

use mycelium_cert::{
    binary_to_ternary, check, ternary_to_binary, BinaryTernarySwapEngine, CheckVerdict, Evidence,
    RefinementRelation,
};
use mycelium_core::{
    ContentHash, GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Trit, Value,
};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_mlir::{compile_and_run_with_swap_mode, AotError, SwapCertMode};
use mycelium_numerics::Certificate;

// ─── helpers ───────────────────────────────────────────────────────────────────────────────────

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

/// The NFR-7 observable: `(repr, payload, guarantee)`. The native read-back reconstructs
/// `Meta::exact`, and the certified swap also tags `Exact` (it is value-preserving within range), so
/// the two observables coincide for an in-range swap.
type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);
fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}

/// The reference interpreter wired with the **certified binary↔ternary swap engine** (the reference
/// the native swap must match — M-120; RFC-0002 §4).
fn interp_swap(node: &Node) -> Result<Value, mycelium_interp::EvalError> {
    Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    )
    .eval(node)
}

/// A `Swap{src → target}` node over a constant source.
fn swap_node(src: Value, target: Repr) -> Node {
    Node::Swap {
        src: Box::new(Node::Const(src)),
        target,
        policy: policy(),
    }
}

const BOTH_MODES: [SwapCertMode; 2] = [SwapCertMode::Recheck, SwapCertMode::ReuseInterp];

// ─── corpus ────────────────────────────────────────────────────────────────────────────────────

/// In-range binary↔ternary swaps over legal pairs. `(8, 6)` is legal: `2^7 = 128 ≤ (3^6−1)/2 = 364`.
/// Each case is a known pair where the swap stays within range; the named integer oracle is checked
/// against `mycelium-cert` directly in [`b_to_t_matches_cert`].
fn b_to_t_corpus() -> Vec<(Node, Value)> {
    // The certified reference value (computed via the cert crate), paired with the program.
    let mk = |bits: Vec<bool>, m: u32| {
        let src = binary(bits);
        let (cert_val, _cert) =
            binary_to_ternary(&src, m, &policy()).expect("legal pair, in range");
        (swap_node(src.clone(), Repr::Ternary { trits: m }), cert_val)
    };
    vec![
        // 0b1011_0010 (MSB-first) = −78 → 6 trits.
        mk(vec![true, false, true, true, false, false, true, false], 6),
        // 0 → 6 trits.
        mk(vec![false; 8], 6),
        // 127 (max positive byte) → 6 trits (364 ≥ 127, in range).
        mk(vec![false, true, true, true, true, true, true, true], 6),
        // −128 (min byte) → 6 trits (364 ≥ 128, in range).
        mk(
            vec![true, false, false, false, false, false, false, false],
            6,
        ),
        // A 4-bit value into 4 trits: (4,4) legal (2^3=8 ≤ (3^4−1)/2=40). 0b1011 = −5.
        mk(vec![true, false, true, true], 4),
    ]
}

/// In-range ternary→binary swaps over legal pairs (the partial `dec`, staying in range). The
/// certified reference value is computed via `ternary_to_binary`.
fn t_to_b_corpus() -> Vec<(Node, Value)> {
    let mk = |trits: Vec<Trit>, n: u32| {
        let src = ternary(trits);
        let (cert_val, _cert) =
            ternary_to_binary(&src, n, &policy()).expect("legal pair, in range");
        (swap_node(src.clone(), Repr::Binary { width: n }), cert_val)
    };
    vec![
        // 6-trit value of −78 → 8 bits (round-trips the b_to_t −78 case).
        {
            let t6 = binary_to_ternary(
                &binary(vec![true, false, true, true, false, false, true, false]),
                6,
                &policy(),
            )
            .unwrap()
            .0;
            let Payload::Trits(ts) = t6.payload() else {
                unreachable!()
            };
            mk(ts.clone(), 8)
        },
        // small in-range: 2 = [0,+,-] (3 trits) → 4 bits. `(4,3)` is legal: 2^3=8 ≤ (3^3−1)/2=13,
        // and 2 ∈ B_4 = [−8,7].
        mk(vec![Trit::Zero, Trit::Pos, Trit::Neg], 4),
        // −5 over 4 trits → 4 bits (in range for B_4 = [−8,7]).
        {
            let t = binary_to_ternary(&binary(vec![true, false, true, true]), 4, &policy())
                .unwrap()
                .0;
            let Payload::Trits(ts) = t.payload() else {
                unreachable!()
            };
            mk(ts.clone(), 4)
        },
    ]
}

// ─── the differential ────────────────────────────────────────────────────────────────────────────

/// Binary→Ternary: interp(certified) ≡ direct-LLVM, in **both** cert modes, validated through the
/// M-210 `ObservationalEquiv` checker. The reference value is also pinned to `mycelium-cert`.
#[test]
fn b_to_t_three_way_in_both_cert_modes() {
    for (i, (node, cert_val)) in b_to_t_corpus().iter().enumerate() {
        let interp = interp_swap(node).expect("certified interp must evaluate the in-range swap");
        // The interpreter result must equal the cert crate's own enc (pin the reference).
        assert_eq!(
            observable(&interp),
            observable(cert_val),
            "program #{i}: interp swap != mycelium-cert binary_to_ternary"
        );
        for mode in BOTH_MODES {
            match compile_and_run_with_swap_mode(node, mode) {
                Ok(native) => {
                    assert_eq!(
                        observable(&interp),
                        observable(&native),
                        "program #{i} mode {mode:?}: interp vs direct-LLVM swap diverged"
                    );
                    // M-210: the interp↔native pair validates through the single shared TV checker.
                    assert_eq!(
                        check(
                            &interp,
                            &native,
                            RefinementRelation::ObservationalEquiv,
                            Certificate::exact(),
                            &Evidence::Observational,
                        ),
                        CheckVerdict::Validated {
                            strength: GuaranteeStrength::Exact
                        },
                        "program #{i} mode {mode:?}: shared checker must validate interp↔native"
                    );
                }
                Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
                Err(e) => panic!("program #{i} mode {mode:?}: direct-LLVM swap errored: {e}"),
            }
        }
    }
}

/// Ternary→Binary (the partial `dec`, in range): interp(certified) ≡ direct-LLVM, both modes,
/// M-210-checked, reference pinned to `mycelium-cert`.
#[test]
fn t_to_b_three_way_in_both_cert_modes() {
    for (i, (node, cert_val)) in t_to_b_corpus().iter().enumerate() {
        let interp = interp_swap(node).expect("certified interp must evaluate the in-range dec");
        assert_eq!(
            observable(&interp),
            observable(cert_val),
            "program #{i}: interp swap != mycelium-cert ternary_to_binary"
        );
        for mode in BOTH_MODES {
            match compile_and_run_with_swap_mode(node, mode) {
                Ok(native) => {
                    assert_eq!(
                        observable(&interp),
                        observable(&native),
                        "program #{i} mode {mode:?}: interp vs direct-LLVM dec diverged"
                    );
                    assert_eq!(
                        check(
                            &interp,
                            &native,
                            RefinementRelation::ObservationalEquiv,
                            Certificate::exact(),
                            &Evidence::Observational,
                        ),
                        CheckVerdict::Validated {
                            strength: GuaranteeStrength::Exact
                        },
                        "program #{i} mode {mode:?}: shared checker must validate interp↔native"
                    );
                }
                Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
                Err(e) => panic!("program #{i} mode {mode:?}: direct-LLVM dec errored: {e}"),
            }
        }
    }
}

/// A swap chained through a `let` and a `bit.*` op, then swapped — exercises the *runtime-lane* swap
/// (the source is a computed lane, not a constant), in both modes. `not(A)` then swap to ternary.
#[test]
fn computed_lane_swap_matches_interp() {
    // not(0b1011_0010) = 0b0100_1101 = 77, then swap to 6 trits.
    let a = binary(vec![true, false, true, true, false, false, true, false]);
    let prog = Node::Swap {
        src: Box::new(Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Const(a)],
        }),
        target: Repr::Ternary { trits: 6 },
        policy: policy(),
    };
    let interp = interp_swap(&prog).expect("interp eval");
    for mode in BOTH_MODES {
        match compile_and_run_with_swap_mode(&prog, mode) {
            Ok(native) => assert_eq!(
                observable(&interp),
                observable(&native),
                "mode {mode:?}: computed-lane swap diverged"
            ),
            Err(AotError::ToolchainMissing(_)) => {}
            Err(e) => panic!("mode {mode:?}: computed-lane swap errored: {e}"),
        }
    }
}

/// Out-of-range `dec` (`Ternary → Binary` whose value leaves `B_n`): both paths refuse non-silently
/// (SC-3/G2). The interpreter errors (`SwapError::OutOfRange`), the native path returns
/// `AotError::Overflow` (the read-back sentinel). Never a silent wrap on either path.
#[test]
fn out_of_range_dec_is_refused_non_silently() {
    // A value that fits 6 trits but not B_4 = [−8, 7]: 100 → 6 trits, then dec to 4 bits.
    let big = binary_to_ternary(
        &binary(vec![false, true, true, false, false, true, false, false]), // 0b0110_0100 = 100
        6,
        &policy(),
    )
    .unwrap()
    .0;
    let Payload::Trits(ts) = big.payload() else {
        unreachable!()
    };
    // (4, 6) is a legal pair (B_4 ⊆ T_6), so the swap is *legal* — only the *value* is out of range,
    // which is the partial-`dec` path we want to exercise.
    let prog = swap_node(ternary(ts.clone()), Repr::Binary { width: 4 });

    // Interp: certified engine errors (OutOfRange surfaced as EvalError::Swap).
    let interp = interp_swap(&prog);
    assert!(
        interp.is_err(),
        "interp must refuse the out-of-range dec, got {:?}",
        interp.ok().map(|v| v.payload().clone())
    );

    // Native: AotError::Overflow (the never-silent read-back), or skip if toolchain absent.
    for mode in BOTH_MODES {
        match compile_and_run_with_swap_mode(&prog, mode) {
            Err(AotError::Overflow(_)) => { /* expected explicit refusal */ }
            Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
            Ok(v) => panic!(
                "mode {mode:?}: out-of-range dec must refuse, got {:?}",
                v.payload()
            ),
            Err(e) => panic!("mode {mode:?}: unexpected native error on out-of-range dec: {e}"),
        }
    }
}

/// An **illegal pair** `(n, m)` (where `B_n ⊄ T_m`) is refused at compile time by the `Recheck`
/// mode's independent side-condition re-check (`AotError::UnsupportedNode`) — never emitted (VR-5/G2).
/// `(8, 4)` is illegal: `2^7 = 128 > (3^4−1)/2 = 40`. The certified interpreter also refuses
/// (`SwapError::IllegalPair`), so the compile-time re-check matches the reference verdict.
#[test]
fn illegal_pair_is_refused_at_compile_time_in_recheck_mode() {
    let prog = swap_node(
        binary(vec![true, false, true, true, false, false, true, false]),
        Repr::Ternary { trits: 4 },
    );
    // The certified interpreter refuses the illegal pair (the reference verdict).
    assert!(
        interp_swap(&prog).is_err(),
        "certified interp must refuse the illegal pair (8,4)"
    );
    // Recheck mode refuses at compile time — an explicit UnsupportedNode, not a ToolchainMissing skip
    // and not a silent emission. (This refusal happens during lowering, before llc/clang, so it is
    // returned even on a box without the toolchain.)
    match compile_and_run_with_swap_mode(&prog, SwapCertMode::Recheck) {
        Err(AotError::UnsupportedNode(msg)) => {
            assert!(
                msg.contains("legal pair") || msg.contains("recheck"),
                "the refusal must name the re-check / legal-pair side-condition; got: {msg}"
            );
        }
        other => panic!("Recheck mode must refuse the illegal pair at compile time, got {other:?}"),
    }
}

// ─── M-856: the third leg — the MLIR-dialect path now LOWERS the certified swap class ─────────
//
// M-856 widens the MLIR-dialect fragment to the certified binary↔ternary `Swap` class (always
// under the `Recheck` cert mode — `ReuseInterp` is not wired in the dialect path, a small
// explicitly-deferred gap). So the three-way differential is now a REAL three-way for this class:
// interp(certified) ≡ direct-LLVM ≡ MLIR-dialect, each pair M-210-checked. Dense/VSA targets, a
// non-bit/trit pair, and an **illegal** `(n,m)` pair stay explicit dialect refusals (covered by
// [`swap_illegal_pair_is_refused_by_the_mlir_dialect_path`] and
// [`swap_to_dense_is_refused_by_the_mlir_dialect_path`]) — never silently lowered.

#[cfg(feature = "mlir-dialect")]
#[test]
fn b_to_t_three_way_including_mlir_dialect() {
    for (i, (node, cert_val)) in b_to_t_corpus().iter().enumerate() {
        let interp = interp_swap(node).expect("certified interp must evaluate the in-range swap");
        assert_eq!(
            observable(&interp),
            observable(cert_val),
            "program #{i}: interp swap != mycelium-cert binary_to_ternary"
        );
        match mycelium_mlir::mlir_compile_and_run(node) {
            Ok(native) => {
                assert_eq!(
                    observable(&interp),
                    observable(&native),
                    "program #{i}: interp vs MLIR-dialect swap diverged"
                );
                assert_eq!(
                    check(
                        &interp,
                        &native,
                        RefinementRelation::ObservationalEquiv,
                        Certificate::exact(),
                        &Evidence::Observational,
                    ),
                    CheckVerdict::Validated {
                        strength: GuaranteeStrength::Exact
                    },
                    "program #{i}: shared checker must validate interp↔MLIR-dialect"
                );
            }
            Err(mycelium_mlir::DialectError::ToolchainMissing(_)) => { /* env skip */ }
            Err(e) => panic!("program #{i}: MLIR-dialect swap errored: {e}"),
        }
    }
}

#[cfg(feature = "mlir-dialect")]
#[test]
fn t_to_b_three_way_including_mlir_dialect() {
    for (i, (node, cert_val)) in t_to_b_corpus().iter().enumerate() {
        let interp = interp_swap(node).expect("certified interp must evaluate the in-range dec");
        assert_eq!(
            observable(&interp),
            observable(cert_val),
            "program #{i}: interp swap != mycelium-cert ternary_to_binary"
        );
        match mycelium_mlir::mlir_compile_and_run(node) {
            Ok(native) => {
                assert_eq!(
                    observable(&interp),
                    observable(&native),
                    "program #{i}: interp vs MLIR-dialect dec diverged"
                );
                assert_eq!(
                    check(
                        &interp,
                        &native,
                        RefinementRelation::ObservationalEquiv,
                        Certificate::exact(),
                        &Evidence::Observational,
                    ),
                    CheckVerdict::Validated {
                        strength: GuaranteeStrength::Exact
                    },
                    "program #{i}: shared checker must validate interp↔MLIR-dialect"
                );
            }
            Err(mycelium_mlir::DialectError::ToolchainMissing(_)) => { /* env skip */ }
            Err(e) => panic!("program #{i}: MLIR-dialect dec errored: {e}"),
        }
    }
}

/// Out-of-range `dec` on the MLIR-dialect path: `DialectError::Overflow` (the shared sentinel
/// read-back), mirroring `out_of_range_dec_is_refused_non_silently`'s direct-LLVM assertion —
/// never a silent wrap (SC-3/G2).
#[cfg(feature = "mlir-dialect")]
#[test]
fn swap_out_of_range_dec_is_refused_by_the_mlir_dialect_path() {
    use mycelium_mlir::DialectError;
    let big = binary_to_ternary(
        &binary(vec![false, true, true, false, false, true, false, false]), // 100
        6,
        &policy(),
    )
    .unwrap()
    .0;
    let Payload::Trits(ts) = big.payload() else {
        unreachable!()
    };
    let prog = swap_node(ternary(ts.clone()), Repr::Binary { width: 4 });
    match mycelium_mlir::mlir_compile_and_run(&prog) {
        Err(DialectError::Overflow(_)) => { /* expected explicit refusal */ }
        Err(DialectError::ToolchainMissing(_)) => { /* env skip */ }
        Ok(v) => panic!(
            "the MLIR-dialect path must refuse the out-of-range dec, got {:?}",
            v.payload()
        ),
        Err(e) => panic!("unexpected MLIR-dialect error on out-of-range dec: {e}"),
    }
}

/// An **illegal** `(n, m)` pair — `(8, 4)`: `2^7 = 128 > (3^4−1)/2 = 40` — is refused by the
/// MLIR-dialect path's `Recheck`-mode compile-time re-check (`DialectError::Unsupported`), the
/// dialect analogue of `illegal_pair_is_refused_at_compile_time_in_recheck_mode`. Never emitted —
/// the illegal pair fails before the toolchain is even invoked (so this refusal is returned even
/// on a box without libMLIR — checked without a `ToolchainMissing` escape hatch).
#[cfg(feature = "mlir-dialect")]
#[test]
fn swap_illegal_pair_is_refused_by_the_mlir_dialect_path() {
    use mycelium_mlir::DialectError;
    let prog = swap_node(
        binary(vec![true, false, true, true, false, false, true, false]),
        Repr::Ternary { trits: 4 },
    );
    match mycelium_mlir::mlir_compile_and_run(&prog) {
        Err(DialectError::Unsupported(msg)) => {
            assert!(
                msg.contains("legal pair") || msg.contains("recheck"),
                "the refusal must name the legal-pair re-check; got: {msg}"
            );
        }
        other => panic!("the illegal pair must be Unsupported at compile time, got {other:?}"),
    }
}

/// A `Swap` to `Dense` stays an explicit MLIR-dialect refusal — only the certified binary↔ternary
/// class (+ same-Repr identity) is lowered here (M-856); Dense/VSA are out of scope (deferred
/// alongside the dialect legs of `dense_differential.rs`/`vsa_differential.rs`, unaffected by this
/// change).
#[cfg(feature = "mlir-dialect")]
#[test]
fn swap_to_dense_is_refused_by_the_mlir_dialect_path() {
    use mycelium_mlir::DialectError;
    let prog = Node::Swap {
        src: Box::new(Node::Const(binary(vec![
            true, false, true, true, false, false, true, false,
        ]))),
        target: Repr::Dense {
            dim: 8,
            dtype: mycelium_core::ScalarKind::F32,
        },
        policy: policy(),
    };
    match mycelium_mlir::mlir_compile_and_run(&prog) {
        Err(DialectError::Unsupported(_)) => { /* expected explicit refusal */ }
        Err(DialectError::ToolchainMissing(_)) => { /* env skip — still no silent success */ }
        Ok(v) => panic!(
            "a Swap to Dense must be refused by the MLIR-dialect path, got {:?}",
            v.payload()
        ),
        Err(e) => panic!("unexpected MLIR-dialect error on Swap-to-Dense: {e}"),
    }
}

/// Sanity: the native swap actually **discriminates** — two different sources do not produce equal
/// results (so the equivalence above is non-vacuous). −78 and 0 swap to different ternary values.
#[test]
fn native_swap_distinguishes_different_sources() {
    let s1 = swap_node(
        binary(vec![true, false, true, true, false, false, true, false]),
        Repr::Ternary { trits: 6 },
    );
    let s2 = swap_node(binary(vec![false; 8]), Repr::Ternary { trits: 6 });
    let (a, b) = match (
        compile_and_run_with_swap_mode(&s1, SwapCertMode::Recheck),
        compile_and_run_with_swap_mode(&s2, SwapCertMode::Recheck),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(AotError::ToolchainMissing(_)), _) | (_, Err(AotError::ToolchainMissing(_))) => return,
        (a, b) => panic!("native swap errored: {a:?} / {b:?}"),
    };
    assert_ne!(observable(&a), observable(&b), "swap(−78) != swap(0)");
    // The shared checker rejects the divergent pair (never a vacuous pass).
    assert!(
        matches!(
            check(
                &a,
                &b,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::NotValidated { .. }
        ),
        "the checker must reject the divergent swap pair"
    );
}
