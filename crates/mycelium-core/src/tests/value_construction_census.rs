//! RFC-0041 §4.5 (W7 / item #6) — **construction-gate census** for [`Value`]/[`Repr`].
//!
//! `Value`/`Repr` keep **derived (recursive)** `Drop`/`Clone`/`PartialEq`/hash (they were *deferred*
//! from the W3 recursion→iteration transform to a coordinated W3b — RFC-0041 §4.5 Status). That
//! deferral is only sound if a **deeply-nested `Value` cannot be built** in the first place: every
//! path that produces a `Value` must itself refuse/recurse over the full nesting depth, so no
//! constructor can hand back a value deep enough to arm the derived recursive `Drop` (a `SIGABRT`
//! bomb). Before this module that "unbuildable" claim was **`Declared`**. This census **upgrades it
//! to `Empirical`** for `Value` — and stands as a **tripwire** that fails loudly if a future
//! constructor bypasses the [`Value::new`] gate.
//!
//! ## What is grounded here (and what is not — VR-5)
//! * **`Empirical` (checked below):** the *only* public path that yields an owned `Value` is
//!   [`Value::new`] (plus `<Value as Deserialize>`, which routes *through* `Value::new`), and that
//!   gate **walks the full nesting depth** — a malformed leaf buried `depth` levels down is rejected,
//!   which is only possible if `check_well_formed` recursed to that leaf. So the native-stack use of
//!   *construction* scales with nesting depth exactly as the derived `Drop` does: a `Value` you can
//!   hold is one whose depth-`D` gate already ran to completion. The wire path is additionally capped
//!   by `serde_json`'s 128-deep recursion limit (never-silent `Err`, not overflow).
//! * **Still `Declared` (NOT upgraded here):** that construction overflows *no later than* the derived
//!   `Drop` would (i.e. the gate's per-frame stack ≥ `Drop`'s). Both are `O(depth)` over the same
//!   spine, but their per-frame stack constants are not compared by any test (a stack overflow aborts
//!   the process — it cannot be probed in-suite). The residual is defended in depth by the
//!   language-level deterministic depth budget (DN-84) and the wire 128-cap; W3b closes it outright by
//!   making `Value`/`Repr` iterative. Stating this rather than overclaiming `Empirical` for the
//!   *ordering* is the VR-5 posture.
//!
//! ## FLAG (material — see the leaf report): a **bare `Repr` is ungated**
//! `Repr` is a `pub enum` whose recursive `Seq { elem: Box<Repr>, .. }` variant is constructible by
//! **direct variant literals** with *no* well-formedness gate ([`Repr::check_well_formed`] runs only
//! inside [`Value::new`]). A bare, arbitrarily-deep `Repr` therefore *can* be built in first-party
//! Rust, and its derived recursive `Drop`/`Clone`/`PartialEq` would `SIGABRT` — so the RFC-0041 §4.5
//! wording "`Value`/`Repr` … construction-gated, thus unbuildable" is **precise for `Value` but
//! overclaims for a bare `Repr`**. Reachability is first-party-Rust-only: the **wire** path is capped
//! by `serde_json`'s 128-limit, and `.myc`/interpreter values only ever exist as gated `Value`s. The
//! practical risk is nil (kernel `Repr`s are shallow descriptors), but the claim's *precision* is not
//! — W3b must either make `Repr` iterative or the amendment must scope the claim to `Value`. This
//! module encodes the gap as a standing, visible assertion ([`bare_repr_construction_is_ungated`]).

use crate::meta::{Meta, Provenance};
use crate::repr::Repr;
use crate::value::{Payload, Value};
use crate::WfError;

/// A modest depth: deep enough that a *depth-coupled* gate demonstrably walks many levels, yet far
/// below the recursive-overflow threshold (~a few thousand frames on a 2 MiB test stack — see
/// `iter_destruction::BINDER_DEEP`) so these tests over the *recursive* `Value`/`Repr` never abort.
const CENSUS_DEPTH: usize = 256;

fn leaf_meta() -> Meta {
    Meta::exact(Provenance::Root)
}

/// A well-formed depth-1 leaf `Value` (an 8-bit `Binary`).
fn leaf_byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        leaf_meta(),
    )
    .expect("leaf byte is well-formed")
}

/// A `Repr` nested `depth` `Seq` levels deep over an inner leaf repr — built by direct public variant
/// literals (i.e. **ungated**; no `Value::new`/`check_well_formed` on the path).
fn nested_seq_repr(inner: Repr, depth: usize) -> Repr {
    let mut r = inner;
    for _ in 0..depth {
        r = Repr::Seq {
            elem: Box::new(r),
            len: 1,
        };
    }
    r
}

/// A well-formed `Value` nested `depth` singleton-`Seq` levels deep, built **bottom-up through the
/// [`Value::new`] gate** (each level re-runs `check_well_formed` + `payload_matches`).
fn nested_seq_value(depth: usize) -> Value {
    let mut v = leaf_byte();
    for _ in 0..depth {
        let elem_repr = v.repr().clone();
        v = Value::new(
            Repr::Seq {
                elem: Box::new(elem_repr),
                len: 1,
            },
            Payload::Seq(vec![v]),
            leaf_meta(),
        )
        .expect("each singleton-seq wrap is well-formed and homogeneous");
    }
    v
}

/// Count how many `Seq` levels a value nests, by walking element 0 down to the leaf.
fn measured_depth(root: &Value) -> usize {
    let mut d = 0;
    let mut cur = root;
    while let Some(inner) = cur.seq_get(0) {
        d += 1;
        cur = inner;
    }
    d
}

// --------------------------------------------------------------------------------------------
// Census / tripwire 1 — the ONLY owned-`Value` constructor is the gated `Value::new`.
// --------------------------------------------------------------------------------------------

/// Collect the names of every `pub fn` in `src` whose signature returns an **owned** `Value`
/// (`-> Value`, `-> Result<Value…>`, or `-> Result<Self…>`). Accessors returning `&Repr` /
/// `Option<…>` / `&[…]` are excluded — they hand out borrows/queries, never a fresh value.
fn owned_value_constructor_fns(src: &str) -> Vec<String> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if let Some(pos) = lines[i].find("pub fn ") {
            // Reassemble the signature (it may span lines) up to the body `{` or a `;`.
            let mut sig = String::new();
            let mut j = i;
            loop {
                sig.push_str(lines[j]);
                sig.push(' ');
                if lines[j].contains('{')
                    || lines[j].trim_end().ends_with(';')
                    || j + 1 >= lines.len()
                {
                    break;
                }
                j += 1;
            }
            let after = &lines[i][pos + "pub fn ".len()..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let returns_owned_value = sig.contains("-> Value")
                || sig.contains("Result<Value")
                || sig.contains("Result<Self");
            if returns_owned_value {
                out.push(name);
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn value_new_is_the_sole_owned_value_constructor() {
    let src = include_str!("../value.rs");

    // (a) The only public fn that returns an owned `Value` is `new` (the gate). If a future
    // `Value::from_raw`/builder is added, it appears here and this fails LOUDLY — forcing the author
    // to prove the new path is depth-gated (and to update this allowlist deliberately).
    let ctors = owned_value_constructor_fns(src);
    assert_eq!(
        ctors,
        vec!["new".to_owned()],
        "a new owned-Value constructor bypassing the Value::new gate was added to value.rs \
         (RFC-0041 §4.5 W3b deferral assumes Value::new is the sole gate): {ctors:?}"
    );

    // (b) `Value`'s fields stay private — otherwise a struct literal would sidestep the gate.
    let start = src
        .find("pub struct Value {")
        .expect("value.rs defines `pub struct Value {`");
    let rest = &src[start..];
    let open = rest.find('{').expect("struct body opens");
    let close = rest[open..].find('}').expect("struct body closes");
    let body = &rest[open + 1..open + close];
    assert!(
        !body.contains("pub"),
        "Value's fields must stay private so Value::new is the only constructor; found `pub` in: {body:?}"
    );

    // (c) The `Deserialize` (wire) path routes *through* `Value::new` — it is not an ungated bypass.
    assert!(
        src.contains("Value::new(w.repr, w.payload, w.meta)"),
        "the Value Deserialize impl must route through the Value::new gate (never-silent §4.8)"
    );
}

// --------------------------------------------------------------------------------------------
// Census 2 — the gate walks the FULL nesting depth (anti-vacuity: construction is depth-coupled).
// --------------------------------------------------------------------------------------------

#[test]
fn value_new_gate_walks_the_full_nesting_depth() {
    // A `Seq` spine of `depth` well-formed levels wrapping a *malformed* innermost leaf
    // (`Binary { width: 0 }`). Every `Seq` wrapper is itself well-formed (len 1 ≤ MAX_DIM), so the
    // ONLY way `Value::new` can reject is by recursing all the way to the depth-`D` leaf. A rejection
    // therefore *witnesses* that the gate examined the full depth — construction cannot outrun its own
    // recursion, so it cannot hand back a value deeper than the native stack allows.
    for &depth in &[0usize, 1, 8, 64, CENSUS_DEPTH] {
        let malformed = nested_seq_repr(Repr::Binary { width: 0 }, depth);
        // The payload is irrelevant here: `check_well_formed` runs *before* `payload_matches`.
        let got = Value::new(malformed.clone(), Payload::Bits(Vec::new()), leaf_meta());
        assert!(
            matches!(got, Err(WfError::MalformedRepr)),
            "a malformed leaf {depth} levels deep must be rejected (MalformedRepr) — \
             the gate must walk the full depth; got {got:?}"
        );
        // The `bool`/never-silent predicates agree, and both walked the same depth.
        assert!(!malformed.well_formed());
        assert!(matches!(
            malformed.check_well_formed(),
            Err(WfError::MalformedRepr)
        ));
    }
}

#[test]
fn well_formed_deep_value_is_gated_but_constructible_at_safe_depth() {
    // A well-formed depth-`CENSUS_DEPTH` value *does* build — through `CENSUS_DEPTH` gate passes — and
    // reports exactly that depth. This exercises the real recursive `check_well_formed` +
    // `payload_matches` (homogeneity) at depth, and the value drops safely on the way out (below the
    // overflow threshold). It anchors the Empirical claim: constructibility is depth-coupled to the
    // gate, not free.
    let v = nested_seq_value(CENSUS_DEPTH);
    assert_eq!(
        measured_depth(&v),
        CENSUS_DEPTH,
        "the gated construction produced the expected nesting depth"
    );
    // Homogeneity was enforced at every level: the outermost repr is a `Seq`.
    assert!(matches!(v.repr(), Repr::Seq { len: 1, .. }));
}

// --------------------------------------------------------------------------------------------
// Census 3 — the wire (`Deserialize`) path refuses over-limit nesting never-silently.
// --------------------------------------------------------------------------------------------

#[test]
fn wire_deserialize_refuses_over_limit_nesting() {
    // `serde_json`'s default 128-deep recursion limit caps nested containers, so a deep `Seq` wire
    // form is refused with an `Err` (never-silent) long before any deep `Value`/`Repr` materializes —
    // this is the gate on the *wire* construction path (RFC-0041 §4.5 relies on the 128 limit).
    const OVER_LIMIT: usize = 300;

    // A deeply-nested bare `Repr` wire form: {"kind":"Seq","elem":<inner>,"len":1} × OVER_LIMIT.
    let mut repr_json = String::from(r#"{"kind":"Binary","width":8}"#);
    for _ in 0..OVER_LIMIT {
        repr_json = format!(r#"{{"kind":"Seq","elem":{repr_json},"len":1}}"#);
    }
    let repr_res: Result<Repr, _> = serde_json::from_str(&repr_json);
    assert!(
        repr_res.is_err(),
        "an over-128-deep Repr wire form must be refused (serde_json recursion limit), not built"
    );

    // A deeply-nested `Value` wire form nests even more containers per level, so it is refused too.
    let mut value_json = String::from(
        r#"{"repr":{"kind":"Binary","width":8},"payload":{"bits":"00000000"},"meta":{"provenance":"Root","guarantee":"Exact"}}"#,
    );
    for _ in 0..OVER_LIMIT {
        value_json = format!(
            r#"{{"repr":{{"kind":"Seq","elem":{{"kind":"Binary","width":8}},"len":1}},"payload":{{"seq":[{value_json}]}},"meta":{{"provenance":"Root","guarantee":"Exact"}}}}"#
        );
    }
    let value_res: Result<Value, _> = serde_json::from_str(&value_json);
    assert!(
        value_res.is_err(),
        "an over-128-deep Value wire form must be refused (never-silent), not built"
    );
}

// --------------------------------------------------------------------------------------------
// Census 4 — the FLAG, encoded as a standing assertion: a **bare `Repr` is ungated**.
// --------------------------------------------------------------------------------------------

#[test]
fn bare_repr_construction_is_ungated() {
    // A bare `Repr` nested `CENSUS_DEPTH` levels deep is built by *direct public variant literals*
    // with **no** gate on the path — no `Value::new`, no `check_well_formed`. This is the documented
    // asymmetry vs `Value` (which cannot be held without its gate having run): the derived recursive
    // `Drop` of a *bare* deep `Repr` is armable from first-party Rust. We construct at a SAFE depth
    // (so its own derived `Drop` does not overflow here) purely to witness that construction is
    // ungated. NOT reachable from wire (128-cap, tested above) or `.myc` (values are gated `Value`s).
    let deep = nested_seq_repr(Repr::Binary { width: 8 }, CENSUS_DEPTH);

    // Well-formedness is an *opt-in* query, NOT enforced at construction — the crux of the FLAG.
    assert!(
        deep.well_formed(),
        "this particular deep repr is dimensionally well-formed — but nothing *checked* that at \
         construction; `well_formed()` is a separate opt-in call"
    );
    // Count the ungated depth to prove it really is `CENSUS_DEPTH` deep with no gate having run.
    let mut d = 0usize;
    let mut cur = &deep;
    while let Repr::Seq { elem, .. } = cur {
        d += 1;
        cur = elem;
    }
    assert_eq!(
        d, CENSUS_DEPTH,
        "a bare Repr was nested this deep with zero construction-time gating (RFC-0041 §4.5 FLAG: \
         'Repr construction-gated' overclaims for a bare Repr — W3b must make Repr iterative or \
         scope the claim to Value)"
    );
    drop(deep); // safe at CENSUS_DEPTH; would SIGABRT (derived recursive Drop) at large depth.
}
