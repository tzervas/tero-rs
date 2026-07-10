//! Structured, never-silent gap report (G2 / M-873).
//!
//! Every Rust construct this PoC transpiler cannot (or, absent a confirmed Mycelium grammar
//! mapping, will not) express in `.myc` surface syntax is recorded here — never dropped
//! silently. This is the mechanism that keeps the transpiler honest: a construct that has no
//! entry here and no entry in [`crate::gap::GapReport::emitted_items`] would be a silent drop,
//! which the driver's invariant (see `src/transpile.rs`, `src/tests/invariant.rs`) forbids.

use serde::Serialize;

/// The category of an unsupported/uncertain Rust construct, so gaps can be grouped and counted.
///
/// This is a **closed, PoC-scoped** set (not exhaustive of every Rust construct) — a construct
/// that fits none of these still gets [`Category::Other`] plus a free-text `reason`, never a
/// silent drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Category {
    Trait,
    Impl,
    Struct,
    MacroDef,
    MacroInvocation,
    MultiStmtBody,
    GenericBound,
    AssocConst,
    DeriveAttr,
    WhereClause,
    PayloadVariant,
    /// A `#[cfg(test)]` item — explicitly out of scope for this PoC's transpilation surface, but
    /// still recorded (never silently skipped). Excluded from the "expressible fraction"
    /// denominator (see [`GapReport::non_test_item_count`]).
    TestItem,
    /// A `Widen`/`Narrow` conversion-op body this pass deliberately left a gap even though a real
    /// DN-41 `width_cast` prim exists — specifically the `Narrow::narrow` case (DN-41's narrowing
    /// is fallible, `Result<To, NarrowError>`, and this grammar fragment's `fn_item` body has no
    /// `= expr`-shaped Result surface to express a refuse), and the defensive fallback for a
    /// `Widen::widen` body over `Binary{N}`/`Binary{M}` whose target width could not be resolved
    /// from the impl's trait-generic argument (never guessed — VR-5). Distinct from the general
    /// `Impl`/`Other` buckets so the union-backlog can rank "conversion-op gaps" on their own.
    Conversion,
    /// RFC-0041 §4.7/§5.1 (W1): a recursive mapping/emit function (`emit_expr`/`emit_block_as_expr`/
    /// `map_pattern`/`map_type`) refused because the input's nesting exceeded the shared
    /// [`mycelium_workstack::RecursionBudget`]'s depth ceiling — a never-silent refusal in place of
    /// an unguarded host-stack overflow (RR-29 guard-hole inventory). Distinct from `Other` so a
    /// pathological-depth refusal is distinguishable from an ordinary unmapped Rust construct.
    RecursionBudget,
    /// M-1001: a `use` import. The transpiler has **no cross-nodule symbol table**, so it cannot
    /// confirm the imported path resolves to a declared Mycelium nodule — and the M-1000 vet loop
    /// confirms these imports fail `myc check` name-resolution every time (a Rust `use
    /// extern_crate::Sym` names a crate, not a nodule). Emitting an import we cannot confirm resolves
    /// is the same "plausible but wrong" emission `map_type`/`emit_expr` already refuse for
    /// qualified paths/calls (DN-34 §4/§8.2), so a `use` is flagged here, not emitted (VR-5/G2).
    /// Distinct from `Other` so the union backlog can rank import gaps on their own.
    Import,
    /// M-1001: a Rust identifier that is a **Mycelium reserved word** (`Exact`, `F16`, `Binary`, …
    /// — `crate::reserved`), which emitted verbatim into constructor/pattern/type/fn position fails
    /// to **parse** (the lexer tokenizes it as a keyword, not an `Ident`). The transpiler has no
    /// sanctioned auto-rename (the port's per-type ctor prefixing is a human decision —
    /// `lib/compiler/README.md` FLAG-ast-5/FLAG-parse-2), so a collision is gapped, never silently
    /// emitted or renamed (G2/VR-5). Distinct so reserved-word collisions rank on their own.
    ReservedWord,
    /// M-1006 (kickoff `trx2`, E33-1): a `struct`/enum-variant **named-field record** whose fields
    /// all map — emitted as the grammar's **positional** `constructor` (`Ident '(' type_ref,* ')'`),
    /// with the field *names* dropped (Mycelium's `constructor` is positional-only —
    /// `docs/spec/grammar/mycelium.ebnf` §`constructor`; there is no record surface). This is **not**
    /// a refusal: the item IS emitted (its product structure is preserved, faithfully, exactly as the
    /// `lib/std/*.myc` hand-ports render a Rust record — e.g. `type GuaranteeRow = Row(Bytes, …)`),
    /// so this rides on the item's `sub_gaps` as a never-silent fidelity note recording *which* field
    /// names were dropped (G2). Distinct from `Struct`/`PayloadVariant` (which remain hard refusals
    /// for a field whose *type* has no mapping) so the emitted-with-names-dropped set is countable on
    /// its own and never conflated with an un-emitted struct.
    NamedFieldDrop,
    Other,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Category::Trait => "Trait",
            Category::Impl => "Impl",
            Category::Struct => "Struct",
            Category::MacroDef => "MacroDef",
            Category::MacroInvocation => "MacroInvocation",
            Category::MultiStmtBody => "MultiStmtBody",
            Category::GenericBound => "GenericBound",
            Category::AssocConst => "AssocConst",
            Category::DeriveAttr => "DeriveAttr",
            Category::WhereClause => "WhereClause",
            Category::PayloadVariant => "PayloadVariant",
            Category::TestItem => "TestItem",
            Category::Conversion => "Conversion",
            Category::RecursionBudget => "RecursionBudget",
            Category::Import => "Import",
            Category::ReservedWord => "ReservedWord",
            Category::NamedFieldDrop => "NamedFieldDrop",
            Category::Other => "Other",
        }
    }
}

/// One construct this transpiler could not (or would not) express in Mycelium surface syntax.
#[derive(Debug, Clone, Serialize)]
pub struct Gap {
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub category: Category,
    /// `Category::as_str()` for [`Gap::category`] — kept as its own (string-typed) field for
    /// serialization stability, but **always derived from `category`**, never a separately
    /// re-derived coarse `syn::Item`-kind label (an earlier iteration used e.g. `"Impl"`/`"Fn"`
    /// regardless of *why* an item failed; the finer per-reason `Category` taxonomy is the ground
    /// truth the committed `.gap.json` is synthesized from — G2, no divergence between the
    /// category actually assigned and the string reported for it).
    pub rust_construct: String,
    pub snippet: String,
    pub reason: String,
    /// Best-effort item name, when the Rust construct has one (functions/types/traits/impls/…).
    /// `None` for anonymous constructs (e.g. a bare item-position macro invocation with no
    /// binding name).
    pub item_name: Option<String>,
}

/// Internal helper carrying a [`Category`] + reason before a [`Gap`] is materialized with its
/// span/snippet/name. Used by `emit.rs`'s per-construct mapping functions so a failure's
/// category survives from the point of detection up to the driver.
#[derive(Debug, Clone)]
pub struct GapReason {
    pub category: Category,
    pub reason: String,
}

impl GapReason {
    pub fn new(category: Category, reason: impl Into<String>) -> Self {
        GapReason {
            category,
            reason: reason.into(),
        }
    }
}

/// The full report for one transpiled source file.
///
/// **Transparency (VR-5):** `emitted_items` records that *some* `.myc` text was produced for an
/// item — it is `Declared` (heuristic, unvalidated by any Mycelium parser/typechecker), never a
/// claim that the output is well-typed Mycelium.
#[derive(Debug, Clone, Serialize)]
pub struct GapReport {
    pub source: String,
    pub emitted_items: Vec<String>,
    pub gaps: Vec<Gap>,
    /// `syn::File::items.len()` — every top-level item in the parsed file, test items included.
    pub total_top_level_items: usize,
}

impl GapReport {
    /// Count of gaps tagged [`Category::TestItem`] — `#[cfg(test)]` items excluded from scope.
    pub fn test_item_count(&self) -> usize {
        self.gaps
            .iter()
            .filter(|g| g.category == Category::TestItem)
            .count()
    }

    /// `total_top_level_items` minus test items — the denominator for the expressible fraction.
    pub fn non_test_item_count(&self) -> usize {
        self.total_top_level_items
            .saturating_sub(self.test_item_count())
    }

    /// Fraction of non-test top-level items for which some `.myc` text was emitted.
    /// `Declared` (see struct docs) — a ratio over a heuristic classification, not a guarantee.
    pub fn expressible_fraction(&self) -> f64 {
        let denom = self.non_test_item_count();
        if denom == 0 {
            return 0.0;
        }
        self.emitted_items.len() as f64 / denom as f64
    }

    /// Per-category gap counts, for reporting.
    pub fn category_counts(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut m = std::collections::BTreeMap::new();
        for g in &self.gaps {
            *m.entry(g.category.as_str()).or_insert(0) += 1;
        }
        m
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// RFC-0041 §4.7/§5.1 (W1) — the shared recursion-budget guard for `emit.rs`/`map.rs`'s mutual/self
// recursion over the `syn` AST (RR-29 guard-hole inventory).
//
// `emit_expr`/`emit_block_as_expr`/`map_pattern` (mutually recursive, `emit.rs`) and `map_type`
// (self-recursive, `map.rs`) previously recursed on unbounded attacker/user-controlled input depth
// with no guard: Rust's default stack-overflow handler aborts the process directly (never through
// panic/unwind, so `catch_unwind` cannot help) — a `SIGABRT`, not a `Result`. This crate-wide,
// per-thread [`mycelium_workstack::RecursionBudget`] closes that hole: every recursive function
// enters one guarded frame via [`guarded`] at its own call site (not just the outermost public
// entry), so a pathological/attacker-controlled AST depth refuses with a
// `Category::RecursionBudget` [`GapReason`] once the shared depth ceiling
// ([`mycelium_workstack::RecursionBudget::DEFAULT_DEPTH_LIMIT`] = 4096) is reached — never a panic,
// abort, or silent drop (G2).
//
// One budget instance is shared across `emit.rs` and `map.rs` (rather than one per function) —
// simpler, and correct because the mutually-/self-recursive groups never run *concurrently* within
// a single transpile pass on one thread: each call chain fully unwinds (every [`DepthGuard`] drops)
// before the next top-level item's chain begins, so a shared counter never conflates two unrelated
// passes.
thread_local! {
    static RECURSION_BUDGET: mycelium_workstack::RecursionBudget =
        mycelium_workstack::RecursionBudget::default();
}

/// Map a recursion-budget refusal onto this crate's own never-silent [`GapReason`] surface
/// (RFC-0041 §5.1's canonical `BudgetError` reconciles here). `DepthExceeded` is the variant this
/// crate can actually hit (depth-only guarding, W1); `OutOfBudget` is mapped too for completeness
/// even though this crate does not currently charge bytes/work-steps.
fn budget_err_to_gap(e: mycelium_workstack::BudgetError) -> GapReason {
    match e {
        mycelium_workstack::BudgetError::DepthExceeded { limit } => GapReason::new(
            Category::RecursionBudget,
            format!(
                "recursion depth budget exceeded (limit {limit} source-call frames) — refused \
                 before a host-stack overflow, per RFC-0041 §4.7/§5.1 (RR-29 guard-hole close, W1)"
            ),
        ),
        mycelium_workstack::BudgetError::OutOfBudget {
            kind,
            limit,
            requested,
        } => GapReason::new(
            Category::RecursionBudget,
            format!(
                "{} budget exhausted (needed {requested}, ceiling {limit})",
                kind.label()
            ),
        ),
    }
}

/// Run `body` guarded by one entered depth frame of the crate-wide [`RECURSION_BUDGET`] (RFC-0041
/// §4.7, W1). Call this at the top of every mutually-/self-recursive function in `emit.rs`/`map.rs`
/// (not just the outermost public entry) so each recursion step consumes budget and a
/// pathological-depth input refuses cleanly with a `Category::RecursionBudget` gap instead of
/// risking a host-stack-overflow `SIGABRT`.
pub(crate) fn guarded<R>(body: impl FnOnce() -> Result<R, GapReason>) -> Result<R, GapReason> {
    RECURSION_BUDGET.with(|budget| {
        let _guard = budget.try_enter().map_err(budget_err_to_gap)?;
        body()
    })
}
