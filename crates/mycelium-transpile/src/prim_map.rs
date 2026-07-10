//! Forward-map the **known kernel Π prim/API surface** into Rust `Expr::MethodCall` intrinsic
//! patterns, AHEAD of any backend that isn't fully wired yet (trx2 Lane C Deliverable 2).
//!
//! **VERIFY-FIRST (mitigation #14) — every row here traces to a checked citation, never a
//! doc-derived guess.** Two states, both never-silent (G2):
//!
//! - **`wired: true`** — the kernel prim is landed (confirmed in
//!   `crates/mycelium-l1/src/checkty.rs`'s `prim_kernel_name`/`prim_sig` tables) AND this crate
//!   independently confirmed the *exact emitted surface text* checks clean with the real built
//!   `target/debug/myc` (see each row's citation for the probe). [`emit_expr_inner`]'s
//!   `Expr::MethodCall` arm emits the real call for these.
//! - **`wired: false`** — a **PENDING-BACKEND** row: the mapping is *known* (a decided ADR/RFC/DN
//!   ruling), but the runtime/grammar backend is not landed yet. The emitter NEVER emits text for
//!   these — it always refuses with a structured [`crate::gap::GapReason`] citing the ruling
//!   (VR-5/G2: a forward-declared mapping is documentation + a precise gap, never a fabricated
//!   success).
//!
//! # What is deliberately **excluded** from this table (verify-first findings, not oversights)
//!
//! The kickoff brief's WIRED list also named `bit.mul` (`mul_u`) and `bit.popcount`/`bit.clz`/
//! `bit.ctz` (CU-1/CU-6) — both genuinely kernel-landed
//! (`docs/notes/DN-34-Rust-to-Mycelium-Transpiler-Strategy.md` §8.16: Π 59→66, PRs #1273/#1275/
//! #1291). They are **not** rows here because no Rust `Expr::MethodCall` pattern was found that is
//! *faithful* to their calling convention:
//! - `mul_u(a, b) -> Binary{N}` refuses (a runtime `Overflow`) rather than returning `Option` —
//!   Rust's semantically-closest never-silent method, `.checked_mul(rhs) -> Option<T>`, is a real
//!   corpus idiom (`crates/mycelium-std-math/src/exact.rs:273`), but its VALUE SHAPE does not match
//!   (`Option[Binary{N}]` vs bare `Binary{N}`) — mapping the isolated call node would emit a
//!   *type-mismatched* body wherever the `Option` is actually consumed (the realistic case). Rust's
//!   `.wrapping_mul()` is the wrong direction entirely (silently wraps — the G2 anti-pattern this
//!   whole project exists to avoid).
//! - `popcount`/`clz`/`ctz` are **width-preserving** (`Binary{N} -> Binary{N}`,
//!   `crates/mycelium-l1/src/checkty.rs:7164`), but Rust's `.count_ones()`/`.leading_zeros()`/
//!   `.trailing_zeros()` **always return a fixed `u32`** regardless of receiver width — so any
//!   *real, compiling* Rust source using them has an enclosing `u32`-typed context, which maps to
//!   `Binary{32}` and mismatches a `Binary{N}`-shaped body for every `N != 32`.
//!
//! Under `src/tests/vet.rs`'s **file-gated all-or-nothing** `myc check` classification
//! (`checked_clean_items_is_file_gated_all_or_nothing`), a wrongly-typed `wired: true` emission
//! does not just miss an opportunity — it can cost a whole file's `checked_fraction`. Both CU-1 and
//! CU-6 are already reachable through the sanctioned route instead: Deliverable 1's `Expr::Binary`
//! operand-type-gated rewrite (`&`/`|` -> `and`/`or`; DN-34 §8.16 item 4 names exactly this as
//! *the* transpiler-side path for these units). Forcing a second, unfaithful Call/MethodCall
//! binding here would be the "guessed API" VR-5 forbids — reported as a FLAG, not guessed.
//!
//! CU-3 (float<->int conversion) is also excluded: DN-34 §8.16 records a *directional* ruling
//! ("prims for the total directions") but no confirmed prim **name**, and Rust's natural spelling
//! for a value conversion is the `as` cast (`syn::Expr::Cast`), which this emitter has no arm for
//! at all (out of a `Call`/`MethodCall` table's scope). CU-7 (arbitrary-width ternary) is excluded
//! because its natural Rust shape (`BigTernary::add/sub/mul/neg`,
//! `crates/mycelium-core/src/ternary/big_ternary.rs`) uses fully generic method names that collide
//! with an enormous space of unrelated Rust code (`.add()`/`.sub()`/`.mul()`/`.neg()` are common
//! method names on user types) — keying a table row on the bare name alone would misattribute
//! unrelated calls, and the target surface name is itself undecided (DN-34: "needs the
//! growable-`Repr::Ternary` decision"). CU-8 (atomics) and CU-9 (Dense dtype/quant) are excluded
//! per DN-34's own explicit rulings: CU-8 "needs a memory-model RFC ... mint a tracked issue + an
//! RFC stub; do **not** scope a partial stub" (an explicit no-half-measures instruction this row
//! would violate), and CU-9 "rides the E20-1 content-address rehash" (blocked on an unrelated
//! architectural decision, no shape decided at all). All five are reported as FLAGs in the leaf's
//! final report rather than forward-mapped as guesses.

use crate::gap::Category;

/// The gate on a `wired: true` row's **receiver** — required so a coincidentally-same-named method
/// on an unrelated Rust type never triggers a wrong emission (VR-5: never fire on an unconfirmed
/// operand type). Checked against the receiver's [`crate::emit::TypeEnv`] entry (only a bare
/// identifier already known in scope can ever match — never a guess).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverGate {
    /// The receiver's mapped type text must equal exactly this string (e.g. `"Float"`).
    Exact(&'static str),
    /// The receiver's mapped type text must be some concrete `Binary{N}` (any width).
    AnyBinaryWidth,
}

/// One forward-mapped Rust `Expr::MethodCall` pattern (see module docs).
#[derive(Debug, Clone, Copy)]
pub struct PrimMapping {
    /// The Rust method name this row matches (`syn::ExprMethodCall::method`).
    pub rust_method: &'static str,
    /// The Mycelium prim/surface call name — a bare, no-import-needed identifier when `wired`; a
    /// forward-declared (not-yet-real) name otherwise (still cited, never fabricated wholesale —
    /// see each row's `citation`).
    pub myc_prim: &'static str,
    /// Kernel backend landed? `true` -> emit the real call; `false` -> always refuse
    /// (PENDING-BACKEND, never emitted).
    pub wired: bool,
    /// The operand-type gate gating whether this row applies at a given call site.
    pub receiver_gate: ReceiverGate,
    /// When `wired` and the prim's own return is `Binary{1}` where Rust's method returns `bool`
    /// (e.g. `flt_is_nan`), bridge it via the Deliverable-1-proven
    /// `(match <call> { 0b1 => True, _ => False })` composition rather than a bare call (a bare
    /// `Binary{1}` value fails `myc check` against a `Bool`-typed context — confirmed empirically,
    /// same class of gap as `eq`/`lt`'s `Binary{1}` result in `emit.rs`'s `Expr::Binary` docs).
    pub bridge_binary1_to_bool: bool,
    /// The gap category to file a PENDING-BACKEND refusal under (irrelevant for `wired: true`
    /// rows, which never gap).
    pub pending_category: Category,
    /// The `<M-id or slug>` used in the `PENDING-BACKEND(<slug>)` annotation / gap reason, and the
    /// human citation trail backing this row's mapping decision.
    pub slug: &'static str,
    pub citation: &'static str,
}

/// The table. Order is insertion order; [`lookup`] does a linear scan (small, fixed table — no
/// need for a map).
///
/// **WIRED rows (CU-2, ADR-040 §2.5 / `checkty.rs:7325-7327` / DN-34 §8.16 "CU-2 ... landed
/// #1274"):** `flt_is_nan`/`flt_is_finite`/`flt_is_infinite` are confirmed bare-call prims (no
/// import) whose Rust intrinsics (`f64::is_nan`/`is_finite`/`is_infinite`) are real, attested
/// corpus usage (`crates/mycelium-std-math/src/approx.rs`, `exact.rs`). Verified `myc check`-clean
/// with the receiver typed `Float` and the `Binary{1}`->`Bool` bridge applied (probed against
/// `target/debug/myc` — `fn f(x: Float) => Bool = (match flt_is_nan(x) { 0b1 => True, _ => False
/// });` checks clean; see this crate's `src/tests/prim_map.rs` for the committed regression).
/// Requires `crate::map::map_type`'s companion fix (this leaf) mapping Rust `f64` -> the grammar's
/// real nullary `Float` base_type (`docs/spec/grammar/mycelium.ebnf:251`, ADR-040 FLAG-1/M-897) —
/// without that fix these rows are reachable in the table but never actually applicable (no `Float`
/// receiver can ever appear in `env`).
///
/// **PENDING-BACKEND rows (CU-5, RFC-0034 §10 / M-791 / DN-34 §8.16 item 2):** the named
/// `wrapping` construct is a **decided** ruling ("implement the M-791 named construct, no new
/// `wrapping_*` prims... wire the construct to modular evaluation over `bin.add`/`sub`/`mul`"), but
/// has **no grammar surface at all yet** (confirmed: `wrapping` does not appear anywhere in
/// `docs/spec/grammar/mycelium.ebnf`) and no wired runtime evaluation path — per
/// `crates/mycelium-core/src/wrapping.rs`'s module doc, the op-layer wiring (arithmetic/swap
/// operations that actually honor the `WrappingOpt` marker) is a downstream task, and "the op
/// layer is wired once arithmetic/swap operations exist". Gated on the receiver being a known
/// `Binary{N}` (any width) so an unrelated user type's `.wrapping_add()`-named method never
/// produces a misleading citation.
pub const TABLE: &[PrimMapping] = &[
    PrimMapping {
        rust_method: "is_nan",
        myc_prim: "flt_is_nan",
        wired: true,
        receiver_gate: ReceiverGate::Exact("Float"),
        bridge_binary1_to_bool: true,
        pending_category: Category::Other,
        slug: "CU-2",
        citation: "ADR-040 §2.5; checkty.rs:7325 (\"flt_is_nan\" => \"flt.is_nan\"); DN-34 §8.16 \
                   (landed #1274); f64::is_nan real corpus usage \
                   crates/mycelium-std-math/src/approx.rs",
    },
    PrimMapping {
        rust_method: "is_finite",
        myc_prim: "flt_is_finite",
        wired: true,
        receiver_gate: ReceiverGate::Exact("Float"),
        bridge_binary1_to_bool: true,
        pending_category: Category::Other,
        slug: "CU-2",
        citation: "ADR-040 §2.5; checkty.rs:7326 (\"flt_is_finite\" => \"flt.is_finite\"); DN-34 \
                   §8.16 (landed #1274); f64::is_finite real corpus usage \
                   crates/mycelium-std-math/src/approx.rs",
    },
    PrimMapping {
        rust_method: "is_infinite",
        myc_prim: "flt_is_infinite",
        wired: true,
        receiver_gate: ReceiverGate::Exact("Float"),
        bridge_binary1_to_bool: true,
        pending_category: Category::Other,
        slug: "CU-2",
        citation: "ADR-040 §2.5; checkty.rs:7327 (\"flt_is_infinite\" => \"flt.is_infinite\"); \
                   DN-34 §8.16 (landed #1274); f64::is_infinite real corpus usage \
                   crates/mycelium-std-math/src/approx.rs",
    },
    PrimMapping {
        rust_method: "wrapping_add",
        myc_prim: "wrapping(bin.add) [name TBD]",
        wired: false,
        receiver_gate: ReceiverGate::AnyBinaryWidth,
        bridge_binary1_to_bool: false,
        pending_category: Category::Conversion,
        slug: "CU-5",
        citation: "RFC-0034 §10; M-791; DN-34 §8.16 item 2 (\"implement the M-791 named \
                   construct, no new wrapping_* prims ... wire the construct to modular \
                   evaluation over bin.add/sub/mul\"); no `wrapping` token in \
                   docs/spec/grammar/mycelium.ebnf (grammar surface unwired) and no wired \
                   runtime evaluation path per crates/mycelium-core/src/wrapping.rs's module \
                   doc (op-layer wiring is a downstream task; \"the op layer is wired once \
                   arithmetic/swap operations exist\")",
    },
    PrimMapping {
        rust_method: "wrapping_sub",
        myc_prim: "wrapping(bin.sub) [name TBD]",
        wired: false,
        receiver_gate: ReceiverGate::AnyBinaryWidth,
        bridge_binary1_to_bool: false,
        pending_category: Category::Conversion,
        slug: "CU-5",
        citation: "RFC-0034 §10; M-791; DN-34 §8.16 item 2; see wrapping_add's citation \
                   (identical basis)",
    },
    PrimMapping {
        rust_method: "wrapping_mul",
        myc_prim: "wrapping(bin.mul) [name TBD]",
        wired: false,
        receiver_gate: ReceiverGate::AnyBinaryWidth,
        bridge_binary1_to_bool: false,
        pending_category: Category::Conversion,
        slug: "CU-5",
        citation: "RFC-0034 §10; M-791; DN-34 §8.16 item 2; see wrapping_add's citation \
                   (identical basis)",
    },
];

/// Look up `rust_method` in [`TABLE`] (first match; the table has no duplicate `rust_method`
/// entries by construction). `None` for any method name this table doesn't cover — the caller's
/// existing (unchanged) generic method-call desugar applies.
#[must_use]
pub fn lookup(rust_method: &str) -> Option<&'static PrimMapping> {
    TABLE.iter().find(|row| row.rust_method == rust_method)
}

/// Whether `receiver_ty` (a resolved [`crate::emit::TypeEnv`] entry, when the receiver is a known
/// bare identifier — see [`crate::emit::expr_env_type`]) satisfies `gate`.
#[must_use]
pub fn receiver_gate_matches(gate: ReceiverGate, receiver_ty: Option<&str>) -> bool {
    match (gate, receiver_ty) {
        (ReceiverGate::Exact(want), Some(got)) => want == got,
        (ReceiverGate::AnyBinaryWidth, Some(got)) => crate::emit::binary_width(got).is_some(),
        (_, None) => false,
    }
}
