//! Mycelium reserved-word snapshot + the identifier collision guard (M-1001).
//!
//! A Rust identifier that is a Mycelium **reserved word**, emitted verbatim into
//! constructor/variant/pattern/type/fn position, fails to **parse** — the lexer tokenizes it as a
//! keyword, not an `Ident` (observed by the M-1000 vet loop as
//! `parse-error: expected a pattern, found Strength(Exact)` on `mycelium-l1/src/eval.rs`, and
//! `expected an identifier, found Binary` on `checkty.rs`). That is a "plausible but wrong" emission
//! the DN-34 §4/§8 flag-don't-guess principle forbids. The transpiler has **no sanctioned renaming
//! scheme** — the self-hosted port's per-type ctor prefixing (`lib/compiler/README.md`
//! FLAG-ast-5/FLAG-parse-2) is a *human* decision, not a mechanical one — so a collision is
//! **gapped** ([`crate::gap::Category::ReservedWord`]), never silently emitted or auto-renamed
//! (G2/VR-5).
//!
//! # Guarantee: `Declared`
//!
//! [`RESERVED`] is a verbatim **snapshot** of `mycelium-l1`'s lexer keyword table
//! (`crates/mycelium-l1/src/token.rs` `fn keyword`) as of **2026-07-06**, copied row-for-row. It is
//! `Declared`, not authoritative — the lexer is ground truth. A drift test
//! (`src/tests/reserved.rs`, a dev-dependency on `mycelium-l1`) asserts every word here is still
//! rejected by the real `mycelium_l1::token::keyword`, so a snapshot that drifts to a *non*-reserved
//! word is caught (the **over-gap** direction — the one that would regress a valid emission). The
//! **under-gap** direction — a *new* keyword `l1` adds that this list misses — is a residual the
//! vet loop catches as a parse error, never a silent bad emission.

use crate::gap::{Category, GapReason};

/// The Mycelium reserved-word set — a verbatim snapshot of `mycelium-l1`'s `token::keyword` table
/// (2026-07-06). Grouped as in the source: active keywords, reserved-not-active runtime/surface
/// terms, the repr-type keywords, the scalar-float keywords, and the guarantee-strength keywords.
pub const RESERVED: &[&str] = &[
    // Active + reserved-not-active structural/surface keywords.
    "nodule",
    "phylum",
    "colony",
    "hypha",
    "fuse",
    "mesh",
    "graft",
    "cyst",
    "xloc",
    "forage",
    "backbone",
    "tier",
    "reclaim",
    "consume",
    "grow",
    "lambda",
    "object",
    "via",
    "lower",
    "derive",
    "use",
    "pub",
    "type",
    "trait",
    "impl",
    "fn",
    "matured",
    "thaw",
    "let",
    "in",
    "if",
    "then",
    "else",
    "match",
    "for",
    "swap",
    "default",
    "paradigm",
    "with",
    "wild",
    "spore",
    "to",
    "policy",
    // Repr-type keywords + their RFC-0037 short aliases.
    "Binary",
    "Ternary",
    "Dense",
    "VSA",
    "bin",
    "tern",
    "emb",
    "hvec",
    "Seq",
    "Bytes",
    "Float",
    "Substrate",
    "Sparse",
    // Scalar-float keywords.
    "F16",
    "BF16",
    "F32",
    "F64",
    // Guarantee-strength keywords.
    "Exact",
    "Proven",
    "Empirical",
    "Declared",
];

/// Whether `word` is a Mycelium reserved word (would not lex as an `Ident`).
pub fn is_reserved(word: &str) -> bool {
    RESERVED.contains(&word)
}

/// Guard an identifier the emitter is about to place into `.myc` surface text. `Ok(())` when it is a
/// legal identifier; `Err(GapReason)` (category [`Category::ReservedWord`]) when it collides with a
/// reserved word — so the caller gaps the construct rather than emit un-parseable text. `context`
/// names the position (e.g. `"enum variant"`, `"match pattern"`, `"type name"`) for the diagnostic.
pub fn guard_ident(name: &str, context: &str) -> Result<(), GapReason> {
    if is_reserved(name) {
        Err(GapReason::new(
            Category::ReservedWord,
            format!(
                "{context} `{name}` collides with a Mycelium reserved word — emitting it verbatim \
                 would fail to parse (the lexer tokenizes it as a keyword, not an identifier); no \
                 sanctioned auto-rename in this PoC, so flagged rather than emitted (G2/VR-5)"
            ),
        ))
    } else {
        Ok(())
    }
}
