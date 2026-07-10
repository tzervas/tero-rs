//! MEM-4 measurement corpus — the DN-33 §8.1 Q5 gate and its Increment-2/audit-trail companion.
//!
//! Q5 requires that, before Increment 2 is committed, Increment 1's effect is **measured**: a
//! reduction in emitted `Dup` count over a **representative** corpus of Core IR terms (not a
//! cherry-picked best case). This module supplies that corpus and the measurement
//! ([`measure`] / [`measure_standard`] → [`CorpusReport`]).
//!
//! Now that Increment 2 (`rc == 1` reuse) and the AOT audit-trail wiring are built, a companion
//! measurement ([`measure_mem4`] / [`measure_mem4_standard`] → [`Mem4Report`]) covers the rest of
//! the analysis tier over the **same** corpus: the count of reuse sites (`MoveUnique` annotations)
//! and of reclamation records (the RFC-0027 §9 audit-trail size), with each reuse annotation
//! machine-verified sound (reached at rc == 1) by the reference evaluator.
//!
//! For each term the measurement lowers it both ways — [`crate::emit::emit_owned`] (naive,
//! fully-owned) and [`crate::emit::emit_elided`] (borrow elision) — counts the `Dup`s in each, and
//! runs the [`crate::eval::differential`] to confirm the elision is **semantics-preserving** (same
//! reclamation multiset, no use-after-free). The aggregate `Dup`-reduction ratio is the Q5 number.
//!
//! # Honest scope (VR-5)
//!
//! - The `Dup` **count** is **`Exact`** (read off the IR); the aggregate ratio over this corpus is
//!   therefore an exact measurement *of this corpus*.
//! - That this corpus is **representative** of real Mycelium programs is **`Declared`** — there is no
//!   Mycelium program population to sample yet. The corpus is deliberately a **mix** of elision-
//!   friendly (reader-heavy `let`s) and elision-neutral (escaping / single-use) terms, so the ratio
//!   is honest, not inflated.
//! - The translation of "fewer `Dup`s" into a runtime **performance** win stays **`Declared`** (no
//!   Mycelium runtime benchmark yet) — the gate is about the analysis doing real work, not a perf SLO.
//!
//! The corpus is the straight-line fragment (so every term is also differential-checkable).

use mycelium_core::{Meta, Node, Payload, Provenance, Repr, Value};

use crate::emit::{emit_elided, emit_owned, emit_reuse, EmitError};
use crate::eval::{count_dups, count_move_unique, differential, eval, RcError};

// ── small Core IR builders ────────────────────────────────────────────────────

fn bit(b: bool) -> Value {
    Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![b]),
        Meta::exact(Provenance::Root),
    )
    .expect("1-bit value is well-formed")
}

fn konst() -> Node {
    Node::Const(bit(true))
}

fn var(name: &str) -> Node {
    Node::Var(name.to_owned())
}

fn let_(name: &str, bound: Node, body: Node) -> Node {
    Node::Let {
        id: name.to_owned(),
        bound: Box::new(bound),
        body: Box::new(body),
    }
}

/// A reader-primitive application of `x` repeated `k` times: `op(x, x, …, x)`.
fn reads(x: &str, k: usize) -> Node {
    Node::Op {
        prim: "read".to_owned(),
        args: (0..k).map(|_| var(x)).collect(),
    }
}

// ── the corpus ────────────────────────────────────────────────────────────────

/// A representative, **mixed** corpus of straight-line Core IR terms (named for the report).
///
/// Includes elision-friendly terms (a `let` read `k≥2` times by reader primitives — where elision
/// removes `k-1` `Dup`s) **and** elision-neutral terms (a value that escapes to the result or into
/// another binding, which stays fully owned). The mix keeps the measured ratio honest.
#[must_use]
pub fn standard_corpus() -> Vec<(&'static str, Node)> {
    vec![
        // ── elision-friendly: reader-heavy lets (the win cases) ───────────────
        ("reader_x2", let_("x", konst(), reads("x", 2))),
        ("reader_x4", let_("x", konst(), reads("x", 4))),
        ("reader_x8", let_("x", konst(), reads("x", 8))),
        (
            "nested_readers",
            // a read twice (1 dup owned), b read 3× (2 dups owned) → 3 owned dups, 0 elided.
            let_("a", konst(), let_("b", reads("a", 2), reads("b", 3))),
        ),
        (
            "chain_of_readers",
            let_(
                "x",
                konst(),
                let_("y", reads("x", 3), let_("z", reads("y", 3), reads("z", 3))),
            ),
        ),
        // ── elision-neutral: single use (k=1 → 0 dups either way) ─────────────
        ("single_read", let_("x", konst(), reads("x", 1))),
        // ── reuse-bearing (Increment 2): sole-owned single moves → MoveUnique ──
        // x is the result → a move, used exactly once → `rc == 1` reuse site (and dup-neutral).
        ("result_move", let_("x", konst(), var("x"))),
        // y is a sole-owned move (the tail); x is fully borrowable (2 reads → DropAfter). So this
        // term carries BOTH a borrow-elision (on x) and a reuse annotation (on y).
        (
            "borrow_then_sole_move",
            let_("x", konst(), let_("y", reads("x", 2), var("y"))),
        ),
        // nested sole moves: x unused (Drop), y a sole-owned move (MoveUnique).
        (
            "sole_move_after_drop",
            let_("x", konst(), let_("y", konst(), var("y"))),
        ),
        // x is read twice AND escapes as the result → owned (2 dups both ways, no win).
        (
            "partial_escape",
            let_("x", konst(), let_("_t", reads("x", 2), var("x"))),
        ),
        // a constant with no bindings at all — zero RC ops either way.
        ("bare_const", konst()),
    ]
}

// ── measurement ───────────────────────────────────────────────────────────────

/// The per-corpus measurement: the Q5 number plus the semantics-preservation guarantee.
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusReport {
    /// Number of terms measured.
    pub n_terms: usize,
    /// Total `Dup`s emitted by the naive owned lowering across the corpus.
    pub owned_dups: usize,
    /// Total `Dup`s emitted by the borrow-elided lowering across the corpus.
    pub elided_dups: usize,
    /// Whether **every** term's elision was semantics-preserving (same reclamations, no UAF).
    pub all_semantics_preserved: bool,
    /// Per-term `(name, owned_dups, elided_dups, preserved)` rows (for an EXPLAIN-able report).
    pub rows: Vec<(&'static str, usize, usize, bool)>,
}

impl CorpusReport {
    /// The aggregate `Dup`-reduction ratio in `[0, 1]` — `(owned - elided) / owned`, or `0.0` if the
    /// owned lowering emitted no `Dup`s. **`Exact`** for this corpus (a count ratio).
    #[must_use]
    pub fn reduction_ratio(&self) -> f64 {
        if self.owned_dups == 0 {
            0.0
        } else {
            (self.owned_dups - self.elided_dups) as f64 / self.owned_dups as f64
        }
    }

    /// Total `Dup`s removed by elision across the corpus.
    #[must_use]
    pub fn dups_removed(&self) -> usize {
        self.owned_dups.saturating_sub(self.elided_dups)
    }
}

/// Errors a corpus measurement can surface (never-silent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusError {
    /// A term failed to lower.
    Emit(EmitError),
    /// A term's differential run errored (a use-after-free / double-free would be a soundness bug).
    Eval(RcError),
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorpusError::Emit(e) => write!(f, "corpus emit error: {e}"),
            CorpusError::Eval(e) => write!(f, "corpus eval error: {e}"),
        }
    }
}

impl std::error::Error for CorpusError {}

/// Measure Increment 1 over `corpus`: aggregate `Dup` counts (owned vs elided) and confirm every
/// term's elision is semantics-preserving (the Q5 gate).
///
/// Returns a [`CorpusReport`]. Errors (never-silent) if any term fails to lower or its differential
/// run faults — either would be a soundness failure, not a measurement.
pub fn measure(corpus: &[(&'static str, Node)]) -> Result<CorpusReport, CorpusError> {
    let mut owned_dups = 0;
    let mut elided_dups = 0;
    let mut all_preserved = true;
    let mut rows = Vec::with_capacity(corpus.len());

    for (name, term) in corpus {
        let owned = emit_owned(term).map_err(CorpusError::Emit)?;
        let elided = emit_elided(term).map_err(CorpusError::Emit)?;
        let o = count_dups(&owned);
        let e = count_dups(&elided);
        let diff = differential(&owned, &elided).map_err(CorpusError::Eval)?;
        owned_dups += o;
        elided_dups += e;
        all_preserved &= diff.is_semantics_preserving();
        rows.push((*name, o, e, diff.is_semantics_preserving()));
    }

    Ok(CorpusReport {
        n_terms: corpus.len(),
        owned_dups,
        elided_dups,
        all_semantics_preserved: all_preserved,
        rows,
    })
}

/// Measure the [`standard_corpus`] — the canonical Q5 measurement.
pub fn measure_standard() -> Result<CorpusReport, CorpusError> {
    measure(&standard_corpus())
}

// ── Increment 2 + audit-trail measurement (beyond Q5's dup-only number) ─────────
//
// Q5 ([`measure`]) measures only Increment 1's `Dup` reduction. This second measurement covers the
// rest of the MEM-4 analysis tier now that it is built: Increment 2's **`rc == 1` reuse sites**
// (`RcNode::MoveUnique` annotations under [`emit_reuse`]) and the **reclamation-record count** (the
// size of the RFC-0027 §9 audit trail the AOT tier emits — the number of `rc → 0` reclamations the
// reference RC machine performs over the elided emission). Both counts are `Exact`; their soundness
// (every reuse annotation reached at rc == 1) is **machine-verified** by the reference evaluator.

/// One term's row in a [`Mem4Report`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mem4Row {
    /// The corpus term's name.
    pub name: &'static str,
    /// `RcNode::MoveUnique` annotations emitted by [`emit_reuse`] for this term (Increment 2).
    pub reuse_sites: usize,
    /// `rc → 0` reclamations over the elided emission (this term's RFC-0027 §9 audit-trail size).
    pub reclamations: usize,
    /// Whether the reuse emission evaluated soundly (no `UnsoundUnique` / use-after-free) —
    /// machine-verified: every `MoveUnique` was reached at a reference count of exactly 1.
    pub sound: bool,
    /// Whether the reuse emission reclaimed the **same multiset** as the owned emission.
    pub preserved: bool,
}

/// The Increment-2 + audit-trail measurement over a corpus (the companion to the Q5 [`CorpusReport`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mem4Report {
    /// Number of terms measured.
    pub n_terms: usize,
    /// Total `rc == 1` reuse sites (`MoveUnique` annotations) emitted across the corpus.
    pub reuse_sites: usize,
    /// Total `rc → 0` reclamations across the corpus (the aggregate §9 audit-trail size).
    pub reclamations: usize,
    /// Whether **every** term's reuse annotations were machine-verified sound (rc == 1 at each).
    pub all_reuse_sound: bool,
    /// Whether **every** term's reuse emission preserved the owned reclamation multiset.
    pub all_semantics_preserved: bool,
    /// Per-term rows (for an EXPLAIN-able report).
    pub rows: Vec<Mem4Row>,
}

/// Measure Increment 2 + the audit trail over `corpus`: aggregate the reuse-site and
/// reclamation-record counts and confirm every reuse annotation is machine-verified sound and
/// semantics-preserving.
///
/// A term that fails to **lower** is a real error (never-silent → [`CorpusError::Emit`]). A reuse
/// emission that **evaluates** unsoundly is **not** an abort: it is recorded as `sound = false` in
/// its row (so the report honestly carries the failure rather than hiding it behind an early return);
/// the aggregate `all_reuse_sound` then reflects it. The straight-line corpus is sound by
/// construction, so `all_reuse_sound` is the *evidence* of that, measured rather than assumed.
pub fn measure_mem4(corpus: &[(&'static str, Node)]) -> Result<Mem4Report, CorpusError> {
    let mut reuse_sites = 0;
    let mut reclamations = 0;
    let mut all_sound = true;
    let mut all_preserved = true;
    let mut rows = Vec::with_capacity(corpus.len());

    for (name, term) in corpus {
        let owned = emit_owned(term).map_err(CorpusError::Emit)?;
        let elided = emit_elided(term).map_err(CorpusError::Emit)?;
        let reuse = emit_reuse(term).map_err(CorpusError::Emit)?;

        let sites = count_move_unique(&reuse);
        // Sound iff the reuse emission evaluates without faulting — i.e. every `MoveUnique` was
        // reached at rc == 1 (else `RcError::UnsoundUnique`) and no use-after-free occurred.
        let reuse_eval = eval(&reuse);
        let sound = reuse_eval.is_ok();
        // The audit-trail size: reclamations performed over the elided emission.
        let records = eval(&elided).map_err(CorpusError::Eval)?.reclaimed.len();
        // Semantics-preservation: the reuse emission reclaims the same multiset as the owned one.
        let preserved = match (&reuse_eval, eval(&owned)) {
            (Ok(r), Ok(o)) => r.reclaimed_sorted() == o.reclaimed_sorted(),
            _ => false,
        };

        reuse_sites += sites;
        reclamations += records;
        all_sound &= sound;
        all_preserved &= preserved;
        rows.push(Mem4Row {
            name,
            reuse_sites: sites,
            reclamations: records,
            sound,
            preserved,
        });
    }

    Ok(Mem4Report {
        n_terms: corpus.len(),
        reuse_sites,
        reclamations,
        all_reuse_sound: all_sound,
        all_semantics_preserved: all_preserved,
        rows,
    })
}

/// Measure the [`standard_corpus`] for Increment 2 + the audit trail.
pub fn measure_mem4_standard() -> Result<Mem4Report, CorpusError> {
    measure_mem4(&standard_corpus())
}
