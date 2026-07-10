//! Tests for the reserved-word collision guard (`src/reserved.rs`, M-1001), incl. the **drift
//! guard** against `mycelium-l1`'s real lexer.
//!
//! **Guarantee: `Empirical`** for the drift test (it runs the real `mycelium_l1::token::keyword`
//! over every snapshot word); pure/`Declared` for the guard behaviour tests.

use crate::gap::Category;
use crate::reserved::{guard_ident, is_reserved, RESERVED};

/// **Drift guard.** Every word in the [`RESERVED`] snapshot must still be rejected as an identifier
/// by the *real* `mycelium-l1` lexer (`mycelium_l1::token::keyword` returns `Some` for a keyword). A
/// snapshot word that drifts to a *non*-reserved word — the direction that would make the guard
/// over-gap and regress a valid emission — fails here. (The under-gap direction — a new keyword `l1`
/// adds that this snapshot misses — is a residual the vet loop catches as a parse error, not a
/// silent bad emission; it is not asserted here because this crate should not have to track every
/// future keyword addition to stay correct — over-gap is the only regressing direction.)
#[test]
fn snapshot_words_are_all_still_reserved_in_the_lexer() {
    for word in RESERVED {
        assert!(
            mycelium_l1::token::keyword(word).is_some(),
            "reserved-word snapshot drift: `{word}` is in crate::reserved::RESERVED but the real \
             mycelium-l1 lexer no longer treats it as a keyword — the snapshot must be re-synced \
             with crates/mycelium-l1/src/token.rs `fn keyword` (this would otherwise over-gap a \
             now-valid identifier)"
        );
    }
}

/// The snapshot is non-empty and free of accidental duplicates (a duplicate is harmless for
/// `contains` but signals a copy error).
#[test]
fn snapshot_is_nonempty_and_deduplicated() {
    assert!(
        !RESERVED.is_empty(),
        "reserved-word snapshot must not be empty"
    );
    let mut seen = std::collections::BTreeSet::new();
    for w in RESERVED {
        assert!(
            seen.insert(*w),
            "duplicate reserved word in snapshot: `{w}`"
        );
    }
}

/// `is_reserved` / `guard_ident` accept ordinary identifiers and reject reserved words, tagging the
/// rejection [`Category::ReservedWord`] with a diagnostic that names the colliding word (G2).
#[test]
fn guard_rejects_reserved_accepts_ordinary() {
    // Ordinary Rust identifiers that are NOT Mycelium reserved words → accepted.
    for ok in [
        "Ordering",
        "ForageError",
        "reverse",
        "is_lt",
        "NoCandidates",
        "Foo",
        "my_fn",
    ] {
        assert!(!is_reserved(ok), "`{ok}` should not be reserved");
        assert!(
            guard_ident(ok, "test position").is_ok(),
            "`{ok}` should pass the guard"
        );
    }
    // Reserved words that a Rust enum/variant/type could easily be named → rejected as ReservedWord.
    for bad in [
        "Exact",
        "Proven",
        "Empirical",
        "Declared",
        "F16",
        "Binary",
        "swap",
        "fuse",
    ] {
        assert!(is_reserved(bad), "`{bad}` should be reserved");
        let err = guard_ident(bad, "match pattern").expect_err("reserved word must be gapped");
        assert_eq!(err.category, Category::ReservedWord);
        assert!(
            err.reason.contains(bad),
            "the gap reason names the colliding word `{bad}` (never silent): {}",
            err.reason
        );
    }
}

/// **Declaration-site coverage (PR #1207 review HIGH).** A reserved word used as an *unused* fn
/// parameter name never flows through `Expr::Path`'s use-site guard, and its name is emitted
/// verbatim into `param ::= Ident ':' type_ref` — so the guard must fire in `map_signature`
/// itself. Repro from the review: `fn set_default(default: u32)` emitted, then failed
/// `myc check` at parse (`expected an identifier, found Default`). Now it must be GAPPED as
/// `ReservedWord`, never emitted.
#[test]
fn unused_reserved_fn_parameter_is_gapped_not_emitted() {
    let src = "pub fn set_default(default: u32) -> u32 { 42 }\n";
    let (myc, report) = crate::transpile::transpile_source(src, "reserved_param", "test")
        .expect("transpile_source itself succeeds; the item is gapped");
    assert!(
        report.emitted_items.is_empty(),
        "the fn must not be emitted: {myc}"
    );
    let gap = report
        .gaps
        .iter()
        .find(|g| g.category == Category::ReservedWord)
        .expect("a ReservedWord gap for the parameter");
    assert!(
        gap.reason.contains("default"),
        "the gap names the colliding word (never silent): {}",
        gap.reason
    );
}

/// Same exposure for an unused generic type-parameter name (declaration-site guard in
/// `plain_type_params`) — defensive twin of the fn-parameter case.
#[test]
fn reserved_type_parameter_is_gapped_not_emitted() {
    let err = guard_ident("Binary", "type parameter").expect_err("reserved type param must gap");
    assert_eq!(err.category, Category::ReservedWord);
}
