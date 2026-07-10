//! M-222 acceptance — selection wired into the **swap-target site** (RFC-0005 §4; RFC-0002):
//! an auto-selected target drives a real `Node::Swap` through the reference interpreter; the
//! result records `Meta.policy_used = PolicyRef` and the selection emitted its mandatory EXPLAIN;
//! an override forces the alternate target deterministically. (The packing site consumes the same
//! `select` via `select_packing` — wired for real in E2-7/M-250.)
//!
//! **M-881 (dev-cycle break):** this test used to run the `Node::Swap` through
//! `mycelium_cert::CertifiedSwapEngine`, which put `mycelium-cert` in `mycelium-select`'s
//! `[dev-dependencies]` — closing a `select →[dev] cert → vsa → select` cycle. What this test
//! actually verifies is the **wiring**: that select's chosen target and `PolicyRef` drive a real
//! `Node::Swap` through the interpreter's `SwapEngine` extension point and land correctly on the
//! result's `Meta.policy_used` (ADR-006) — not the certified engine's numerics (that's
//! `mycelium-cert`'s own test suite). So the one conversion this test exercises (Dense
//! `F32 → BF16`) is reproduced locally via `TestSwapEngine` below, a minimal test double against
//! `mycelium_interp::SwapEngine` (already public, dependency-free of `cert`); same-repr swaps
//! keep using `mycelium-interp`'s own `IdentitySwapEngine`. Coverage is unchanged: every assertion
//! below is identical to the pre-M-881 version.

use mycelium_core::{
    Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, Node, NormKind, Payload,
    Provenance, Repr, ScalarKind, Value,
};
use mycelium_interp::{EvalError, IdentitySwapEngine, Interpreter, SwapEngine};
use mycelium_select::{
    select_swap_target, Action, Candidate, CostModel, Predicate, Rule, SelectionInputs,
    SelectionPolicy,
};

/// Round one `f32` (carried as `f64`, per `Payload::Scalars`) to `bf16` precision by round-to-
/// nearest-even truncation of the low 16 mantissa bits — the standard bf16-rounding algorithm
/// (not cert-specific numerics; `mycelium-cert`'s own `dense_f32_to_bf16` proves the checked ε
/// bound for the *real* engine). Widened back to `f32`/`f64` for storage, matching `Payload`'s
/// wire representation.
#[allow(clippy::cast_possible_truncation)]
fn round_f32_to_bf16(x: f64) -> f64 {
    let bits = (x as f32).to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16);
    f64::from(rounded)
}

/// A minimal test-local stand-in for `mycelium_cert::CertifiedSwapEngine` (see the module doc for
/// why): it supports exactly the one cross-repr conversion this wiring test exercises (Dense
/// `F32 → BF16`), delegates same-repr swaps to `IdentitySwapEngine`, and refuses everything else
/// — never a silent coercion (G2). The result is honestly tagged `Declared` (M-I4): this is a
/// hand-rolled test fixture with no checked/cited theorem behind its bound, so it must not claim
/// the `Proven` strength the real certified engine earns.
struct TestSwapEngine;

impl SwapEngine for TestSwapEngine {
    fn swap(&self, src: &Value, target: &Repr, policy: &ContentHash) -> Result<Value, EvalError> {
        if src.repr() == target {
            return IdentitySwapEngine.swap(src, target, policy);
        }
        match (src.repr(), target) {
            (
                Repr::Dense {
                    dim: src_dim,
                    dtype: ScalarKind::F32,
                },
                Repr::Dense {
                    dim: tgt_dim,
                    dtype: ScalarKind::Bf16,
                },
            ) if src_dim == tgt_dim => {
                let Payload::Scalars(xs) = src.payload() else {
                    return Err(EvalError::UnsupportedSwap {
                        from: src.repr().clone(),
                        to: target.clone(),
                    });
                };
                let out: Vec<f64> = xs.iter().map(|&x| round_f32_to_bf16(x)).collect();
                let bound = Bound {
                    kind: BoundKind::Error {
                        eps: 2f64.powi(-8),
                        norm: NormKind::Rel,
                    },
                    basis: BoundBasis::UserDeclared,
                };
                let meta = Meta::new(
                    Provenance::Derived {
                        op: mycelium_core::operation_hash("test.swap.f32_to_bf16"),
                        inputs: vec![src.content_hash()],
                    },
                    GuaranteeStrength::Declared,
                    Some(bound),
                    src.meta().sparsity(),
                    src.meta().physical(),
                    Some(policy.clone()),
                )
                .map_err(EvalError::Wf)?;
                Value::new(target.clone(), Payload::Scalars(out), meta).map_err(EvalError::Wf)
            }
            _ => Err(EvalError::UnsupportedSwap {
                from: src.repr().clone(),
                to: target.clone(),
            }),
        }
    }
}

/// The worked policy: an exact Dense F32 value swaps to BF16 (halve the storage); otherwise stay.
fn policy() -> SelectionPolicy {
    SelectionPolicy::new(
        "bf16-when-exact",
        vec![
            Candidate::Repr(Repr::Dense {
                dim: 3,
                dtype: ScalarKind::Bf16,
            }),
            Candidate::Repr(Repr::Dense {
                dim: 3,
                dtype: ScalarKind::F32,
            }),
        ],
        vec![Rule {
            when: Predicate::DtypeIs(ScalarKind::F32),
            action: Action::Cheapest, // BF16 wins on the explicit storage cost
        }],
        1,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .unwrap()
}

fn f32_value() -> Value {
    Value::new(
        Repr::Dense {
            dim: 3,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.5, -2.25, 0.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// **The headline wiring:** select → build the `Swap` node with the policy's `PolicyRef` → run
/// the interpreter — the result's `Meta.policy_used` is exactly the recorded `PolicyRef`, and the
/// EXPLAIN trace exists for the selection that chose the target.
#[test]
fn auto_selected_swap_records_policy_ref_and_explains() {
    let policy = policy();
    let v = f32_value();
    let inputs = SelectionInputs::of_value(&v);
    let (target, explanation) = select_swap_target(&policy, &inputs, None).unwrap();
    assert_eq!(
        target,
        Repr::Dense {
            dim: 3,
            dtype: ScalarKind::Bf16
        }
    );
    // The mandatory EXPLAIN: the policy hash matches and every candidate was costed.
    assert_eq!(explanation.policy, policy.policy_ref());
    assert_eq!(explanation.costs.len(), 2);
    assert!(!explanation.overridden);

    // Wire it: the Swap node carries the policy's content hash (WF2), the test engine runs it.
    let node = Node::Swap {
        src: Box::new(Node::Const(v)),
        target,
        policy: policy.policy_ref(),
    };
    let interp = Interpreter::new(
        mycelium_interp::PrimRegistry::with_builtins(),
        Box::new(TestSwapEngine),
    );
    let out = interp.eval(&node).unwrap();
    assert_eq!(
        out.repr(),
        &Repr::Dense {
            dim: 3,
            dtype: ScalarKind::Bf16
        }
    );
    // "Which policy chose this?" — answerable from the value alone (RFC-0005 §3).
    assert_eq!(out.meta().policy_used(), Some(&policy.policy_ref()));
}

/// The first-class override forces the alternate target deterministically — and the run still
/// records the policy that was (overridden but) in charge.
#[test]
fn override_forces_the_alternate_target() {
    let policy = policy();
    let v = f32_value();
    let inputs = SelectionInputs::of_value(&v);
    let (target, explanation) = select_swap_target(&policy, &inputs, Some(1)).unwrap();
    assert_eq!(
        target,
        Repr::Dense {
            dim: 3,
            dtype: ScalarKind::F32
        }
    );
    assert!(explanation.overridden);
    // Determinism of the override across repeated calls.
    for _ in 0..50 {
        let (t2, e2) = select_swap_target(&policy, &inputs, Some(1)).unwrap();
        assert_eq!(t2, target);
        assert_eq!(e2, explanation);
    }
    // Same-repr target → `TestSwapEngine`'s `IdentitySwapEngine` delegation still runs it fine.
    let node = Node::Swap {
        src: Box::new(Node::Const(v.clone())),
        target,
        policy: policy.policy_ref(),
    };
    let interp = Interpreter::new(
        mycelium_interp::PrimRegistry::with_builtins(),
        Box::new(TestSwapEngine),
    );
    let out = interp.eval(&node).unwrap();
    assert_eq!(out.repr(), v.repr());
    assert_eq!(out.payload(), v.payload());
}
