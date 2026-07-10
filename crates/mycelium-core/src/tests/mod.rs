//! In-crate white-box test modules (test layout rule: no tests in logic files; one submodule per
//! source module under test). Extracted as-touched (M-797); new modules land here directly.

mod binary;
mod cert_mode;
mod content;
mod data;
mod id;
/// RFC-0041 §4.5 (W3) — recursion-safe iterative Drop / Clone / PartialEq / content-hash for the
/// frozen recursive types (`Node`, `Datum`/`CoreValue`): deep-chain construct/destruct/clone/unwind
/// without `SIGABRT`, and bit-identical hashes/eq vs a recursive reference oracle.
mod iter_destruction;
#[path = "lib.rs"]
mod lib_root;
mod lower;
mod meta;
/// Shared mode-parametric test harness (M-795; RFC-0034 §13; DN-20). Provides canonical bound
/// fixtures, `for_each_mode`, `ModeScope`, and `assert_mode_scope` for the cross-mode negative
/// pattern. Used by `cert_mode` and `mode_tests`; available to any in-crate test module.
pub(super) mod mode_harness;
mod mode_tests;
mod prim;
mod repr;
/// RFC-0041 §4.5 (W7, item #10) — Box-owned/acyclic spine tripwire for the frozen iterative-`Drop`
/// types (`Node`/`Alt`, `Datum`/`CoreValue`): fails if `Rc`/`Arc` shared ownership appears on the
/// value spine, which would silently break the double-free-safe teardown (DN-56 §6 within-freeze).
mod spine_ownership_tripwire;
mod ternary;
mod value;
/// RFC-0041 §4.5 (W7, item #6) — construction-gate census for `Value`/`Repr`: grounds (Empirical) that
/// a deeply-nested `Value` is unbuildable (every path routes through the depth-walking `Value::new`
/// gate; wire is 128-capped), the tripwire for a future ungated constructor, and the bare-`Repr` FLAG.
mod value_construction_census;
mod wrapping;
