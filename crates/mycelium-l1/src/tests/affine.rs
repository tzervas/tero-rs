//! White-box + property-style tests for the static affine `Substrate` use-once tracker
//! (M-903; DN-71 Model S §4.2 — `crate::affine`, wired into `crate::checkty::Cx`).
//!
//! Layout: single-binding straight-line sweep, an exhaustive `if`-branch merge sweep (the
//! property-test surface the M-903 DoD asks for — "no path consumes a handle twice undetected"),
//! `match`-arm coverage, constructor/field-capture, the honest loop/closure limitation, and the
//! both-sites diagnostic content. **No `proptest` dependency added** — this crate's established
//! idiom is a deterministic, exhaustively-enumerated sweep (see `tests/check.rs`'s coherence/effect
//! sweeps, `src/tests/lexer.rs` l.600, `src/tests/elab.rs` l.277): every case here is generated from
//! a small combinatorial model and checked against a hand-verified oracle, so the sweep is fully
//! reproducible and reviewable, not randomized.

use crate::checkty::{check_nodule, CheckError};
use crate::parse;

fn check(src: &str) -> Result<crate::checkty::Env, CheckError> {
    check_nodule(&parse(src).expect("parses"))
}

fn is_double_consume(err: &CheckError) -> bool {
    err.message.contains("double-consume")
}

// ---- single-binding straight-line: consume/use accepted once, refused twice --------------

#[test]
fn a_substrate_param_used_once_checks() {
    check("nodule d;\nfn f(s: Substrate{Sock}) => Bool = let _ = s in True;")
        .expect("a single use of a Substrate binding checks");
}

#[test]
fn a_substrate_param_used_twice_is_refused_naming_both_sites() {
    let err = check("nodule d;\nfn f(s: Substrate{Sock}) => Bool = let _ = s in s;")
        .expect_err("a second use of the same binding must be refused");
    assert!(is_double_consume(&err), "got: {}", err.message);
    assert!(
        err.message.contains('s'),
        "must name the binding: {}",
        err.message
    );
    assert!(
        err.message.contains("Substrate{Sock}"),
        "must name the affine type: {}",
        err.message
    );
    // Both-sites (RFC-0013 style): the message names the first move and this violating use, each
    // via a stable use ordinal (this checker has no source spans — VR-5, `crate::affine` docs).
    assert!(
        err.message.contains("first move") || err.message.contains("already moved"),
        "must reference the earlier move: {}",
        err.message
    );
    assert!(
        err.message.contains("reference #0") && err.message.contains("reference #1"),
        "must name both use ordinals: {}",
        err.message
    );
}

#[test]
fn consume_of_the_same_substrate_twice_is_refused() {
    // The surface `consume` keyword is just one kind of use — the tracker doesn't special-case it.
    let err = check(
        "nodule d;\nfn f(s: Substrate{Sock}) => Substrate{Sock} = \
         let a = consume s in consume s;",
    )
    .expect_err("consuming the same binding twice must be refused");
    assert!(is_double_consume(&err), "got: {}", err.message);
}

#[test]
fn two_distinct_substrate_bindings_each_used_once_checks() {
    // Two *different* bindings, each used exactly once — no interference between them (tracked
    // independently by scope index, not by shared tag).
    check(
        "nodule d;\nfn f(a: Substrate{Sock}, b: Substrate{Sock}) => Bool = \
         let _ = a in let _ = b in True;",
    )
    .expect("independent single-use bindings both check");
}

// ---- exhaustive straight-line sweep: the property-test surface ---------------------------
//
// Model: a straight-line body references a chosen sequence of two Substrate-typed params (`a`,
// `b`) in order, then returns `True`. The oracle: the program must check iff neither `a` nor `b`
// is referenced more than once in the sequence — every path is single, so "twice in the sequence"
// *is* "twice on the (only) path". Every length-0..=3 sequence over {a, b} is enumerated (15
// cases): this is the exhaustive form of "no path consumes a handle twice undetected" for the
// straight-line fragment.

fn straight_line_src(seq: &[&str]) -> String {
    let mut body = "True".to_owned();
    for name in seq.iter().rev() {
        body = format!("let _ = {name} in {body}");
    }
    format!("nodule d;\nfn f(a: Substrate{{Sock}}, b: Substrate{{Sock}}) => Bool = {body};")
}

fn count(seq: &[&str], name: &str) -> usize {
    seq.iter().filter(|n| **n == name).count()
}

#[test]
fn straight_line_sweep_matches_the_use_count_oracle() {
    let alphabet = ["a", "b"];
    let mut cases: Vec<Vec<&str>> = vec![vec![]];
    for len in 1..=3usize {
        for mask in 0..(1u32 << len) {
            let seq: Vec<&str> = (0..len)
                .map(|i| alphabet[((mask >> i) & 1) as usize])
                .collect();
            cases.push(seq);
        }
    }
    assert_eq!(
        cases.len(),
        1 + 2 + 4 + 8,
        "15 exhaustive sequences up to length 3"
    );

    for seq in &cases {
        let src = straight_line_src(seq);
        let expect_reject = count(seq, "a") >= 2 || count(seq, "b") >= 2;
        let result = check(&src);
        match (expect_reject, result) {
            (false, Ok(_)) => {}
            (true, Err(e)) => assert!(
                is_double_consume(&e),
                "seq {seq:?}: expected a double-consume refusal, got: {}",
                e.message
            ),
            (false, Err(e)) => panic!("seq {seq:?}: expected accept, got refusal: {}", e.message),
            (true, Ok(_)) => panic!("seq {seq:?}: expected a double-consume refusal, but checked"),
        }
    }
}

// ---- exhaustive if-branch merge sweep: the conservative-union property -------------------
//
// Model: `if cond then B1 else B2` where each branch independently references a subset of {a, b}
// (at most once each, so no *intra*-branch double-use confounds the merge being tested), followed
// by an optional `POST` chain after the `if`. The union-merge oracle: a name is "moved after the
// if" iff it was referenced in *either* branch; `POST` must then check iff it references no
// already-moved name and does not reference any name twice itself. This is the exhaustive form of
// DN-71 §4.2's conservative branch-merge rule (`crate::affine` module docs) — 4 branch-content
// choices × 4 branch-content choices × 3 POST choices = 48 cases.

fn branch_chain(names: &[&str]) -> String {
    let mut body = "True".to_owned();
    for name in names.iter().rev() {
        body = format!("let _ = {name} in {body}");
    }
    body
}

fn if_merge_src(b1: &[&str], b2: &[&str], post: &[&str]) -> String {
    let post_chain = branch_chain(post);
    format!(
        "nodule d;\nfn f(cond: Bool, a: Substrate{{Sock}}, b: Substrate{{Sock}}) => Bool = \
         let _ = if cond then {} else {} in {post_chain};",
        branch_chain(b1),
        branch_chain(b2),
    )
}

#[test]
fn if_branch_merge_sweep_matches_the_union_oracle() {
    // Each branch draws from these 4 (at-most-once-per-name) contents.
    let branch_choices: [&[&str]; 4] = [&[], &["a"], &["b"], &["a", "b"]];
    let post_choices: [&[&str]; 3] = [&[], &["a"], &["b"]];

    let mut n_cases = 0usize;
    for b1 in &branch_choices {
        for b2 in &branch_choices {
            for post in &post_choices {
                n_cases += 1;
                let moved_after_if: std::collections::BTreeSet<&str> =
                    b1.iter().chain(b2.iter()).copied().collect();
                let expect_reject = post.iter().any(|n| moved_after_if.contains(n));
                let src = if_merge_src(b1, b2, post);
                let result = check(&src);
                match (expect_reject, result) {
                    (false, Ok(_)) => {}
                    (true, Err(e)) => assert!(
                        is_double_consume(&e),
                        "b1={b1:?} b2={b2:?} post={post:?}: expected a double-consume refusal, \
                         got: {}",
                        e.message
                    ),
                    (false, Err(e)) => panic!(
                        "b1={b1:?} b2={b2:?} post={post:?}: expected accept, got refusal: {}",
                        e.message
                    ),
                    (true, Ok(_)) => panic!(
                        "b1={b1:?} b2={b2:?} post={post:?}: expected a double-consume refusal, \
                         but checked"
                    ),
                }
            }
        }
    }
    assert_eq!(
        n_cases,
        4 * 4 * 3,
        "48 exhaustive (b1, b2, post) combinations"
    );
}

#[test]
fn using_a_handle_in_only_one_branch_and_never_again_checks() {
    // A single spot-check pulled out of the sweep above, spelled out for readability: consuming in
    // only the `then` branch, with no further use anywhere, is fine — the union-merge rule only
    // bites when a *later* reference exists (module docs on why this is sound-over-precise, not
    // over-eager).
    check(
        "nodule d;\nfn f(cond: Bool, s: Substrate{Sock}) => Bool = \
         if cond then let _ = s in True else True;",
    )
    .expect("a substrate used in only one branch, with no later use, checks");
}

// ---- match-arm merge (N-way generalization of the if-merge rule) -------------------------

const TRI: &str = "type Tri = T0 | T1 | T2;\n";

#[test]
fn each_match_arm_may_independently_use_the_scrutinee_free_substrate() {
    check(&format!(
        "nodule d;\n{TRI}fn f(t: Tri, s: Substrate{{Sock}}) => Bool = \
         match t {{ T0 => let _ = s in True, T1 => True, T2 => True }};"
    ))
    .expect("a substrate used in exactly one arm, with no later use, checks");
}

#[test]
fn a_substrate_used_in_one_arm_and_reused_after_the_match_is_refused() {
    let err = check(&format!(
        "nodule d;\n{TRI}fn f(t: Tri, s: Substrate{{Sock}}) => Bool = \
         let _ = match t {{ T0 => let _ = s in True, T1 => True, T2 => True }} in s;"
    ))
    .expect_err("reusing a handle moved in even one arm, after the match, must be refused");
    assert!(is_double_consume(&err), "got: {}", err.message);
}

#[test]
fn a_substrate_used_in_every_arm_independently_checks() {
    // Every arm uses `s` exactly once (mutually exclusive at runtime) — no arm uses it twice, and
    // nothing after the match reuses it, so this is a legitimate single-use-per-path program.
    check(&format!(
        "nodule d;\n{TRI}fn f(t: Tri, s: Substrate{{Sock}}) => Bool = \
         match t {{ T0 => let _ = s in True, T1 => let _ = s in True, T2 => let _ = s in True }};"
    ))
    .expect("each arm's own single use of the substrate checks independently");
}

#[test]
fn a_substrate_used_twice_within_a_single_arm_is_refused() {
    let err = check(&format!(
        "nodule d;\n{TRI}fn f(t: Tri, s: Substrate{{Sock}}) => Bool = \
         match t {{ T0 => let _ = s in s, T1 => True, T2 => True }};"
    ))
    .expect_err("a double use inside one arm must still be refused");
    assert!(is_double_consume(&err), "got: {}", err.message);
}

// ---- constructor / field capture: a "use" is any move, not just `consume` ----------------

#[test]
fn passing_a_substrate_as_a_constructor_field_is_a_use() {
    // `type Box = Mk(Substrate{Sock});` — building a `Box` captures the substrate; a second
    // reference afterward is a double-consume (DN-71 §4.2: "constructor/field capture" is a move).
    let err = check(
        "nodule d;\ntype Box = Mk(Substrate{Sock});\n\
         fn f(s: Substrate{Sock}) => Bool = let _ = Mk(s) in let _ = s in True;",
    )
    .expect_err("reusing a substrate already captured into a constructor must be refused");
    assert!(is_double_consume(&err), "got: {}", err.message);
}

#[test]
fn passing_a_substrate_as_a_constructor_field_once_checks() {
    check(
        "nodule d;\ntype Box = Mk(Substrate{Sock});\n\
         fn f(s: Substrate{Sock}) => Bool = let _ = Mk(s) in True;",
    )
    .expect("a single constructor-field capture is a legitimate single use");
}

#[test]
fn passing_a_substrate_as_a_function_argument_twice_is_refused() {
    let err = check(
        "nodule d;\nfn take(s: Substrate{Sock}) => Bool = True;\n\
         fn f(s: Substrate{Sock}) => Bool = let _ = take(s) in take(s);",
    )
    .expect_err("passing the same substrate as an argument twice must be refused");
    assert!(is_double_consume(&err), "got: {}", err.message);
}

// ---- leak-without-consume: NOT a static refusal — released at runtime instead (M-904) -----

#[test]
fn a_never_consumed_substrate_binding_checks_the_static_pass_does_not_reject_leaks() {
    // M-903's static use-once tracker enforces only the *upper* bound (at most one move) and stays
    // silent on the *lower* bound (zero moves is not a checker error) — a never-consumed binding
    // still type-checks. This is not a gap: DN-71 §8 FLAG-4's v0 drop-without-consume posture
    // (accepted 2026-07-02, delegated to the M-904 integrator) closes the *lower* bound at
    // **runtime** instead — a live, un-consumed `Substrate` is deterministically released and the
    // release recorded at scope exit (`crate::eval`'s `release_if_abandoned`, exercised in
    // `tests/substrate.rs`), never a silent leak (G2). So "checks" here is deliberately not "runs
    // clean" — the static pass's job is only the upper bound; the runtime v0 posture is the other
    // half of the story, landed separately from the checker (KC-3: two concerns, two mechanisms).
    check("nodule d;\nfn f(s: Substrate{Sock}) => Bool = True;")
        .expect("a Substrate binding that is never consumed still type-checks in v0");
}

// ---- a known, honest limitation: a lambda/for body is one lexical use ---------------------

#[test]
fn a_substrate_captured_once_by_a_lambda_checks() {
    // A single lexical reference inside a lambda body is fine (ordinary single-use case); the
    // *multiplicity* gap (a closure that might be called >1 time at runtime) is a static-pass
    // limitation `crate::affine`'s module docs name explicitly — the runtime backstop
    // (`SubstrateHandle::try_consume`, tested in `tests/substrate.rs`) is what catches an actual
    // repeated runtime move that this lexical check cannot see.
    check(
        "nodule d;\nfn f(s: Substrate{Sock}) => Bool = \
         let g = lambda(_x: Bool) => let _ = s in True in g(True);",
    )
    .expect("a substrate captured once by a lambda body checks");
}

#[test]
fn a_substrate_used_once_in_a_for_body_checks() {
    check(
        "nodule d;\ntype ByteList = Nil | Cons(Binary{8}, ByteList);\n\
         fn f(s: Substrate{Sock}, bs: ByteList) => Bool = \
         for _x in bs, acc = True => let _ = s in acc;",
    )
    .expect("a substrate referenced once lexically inside a for-body checks");
}
