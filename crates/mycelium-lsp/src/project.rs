//! The **`LlmCanonical` projection** (RFC-0021 §4.6; M-380) — a v0 prototype renderer.
//!
//! A *projection* (RFC-0021 §3.1) is a total, inspectable function from a content-addressed L1 node
//! tree to a rendered surface; identity stays the content hash, never the rendering (P4). This module
//! is the **LLM-facing canonical** projection (FR-S5): an s-expression rendering chosen for low
//! out-of-distribution token overhead and maximal regularity, intended as the machine-co-authoring
//! surface. It lives **above the kernel** (KC-3) — in the dual-intelligibility surface crate, not in
//! `mycelium-core`.
//!
//! # What this prototype demonstrates (the RFC-0021 §9 ergonomics gate / RP-4 sub-q 1)
//! `research/11` *assessed* that authoring a total, honesty-preserving projection over the closed
//! ~11-node L1 grammar is feasible at single-engineer scale (T11.4). This module turns that assessment
//! into **demonstrated, tested** evidence: it is one total `match` over every [`Node`] kind (the
//! compiler enforces totality), it preserves the honesty overlay by construction, and its tests check
//! that overlay holds. It does **not** address the *empirical* LLM-leverage gate (RP-1 / RFC-0021's
//! §9 second prompt) — that needs LLM runs and is honestly out of scope (`research/11` T11.6).
//!
//! # The honesty overlay (RFC-0021 §4.3 P1–P6), enforced here
//! - **P2 (honest tags survive):** every `Const`'s guarantee tag is rendered (`@Exact`/`@Declared`/…),
//!   and an approximate value's bound presence is surfaced (`:bound`).
//! - **P3 (`Swap` never elided):** the `Swap` node renders explicitly as `(swap! …)`, always carrying
//!   its target and policy — it can never be dropped (it is a match arm).
//! - **P1/P4:** this is a *view*; it changes no node and re-uses the kernel's `CtorRef`/`PrimRef`
//!   `#…` content-addresses as identity.
//!
//! # Honest scope of the prototype (VR-5)
//! The mapping rules here are a Rust `match`, not the *declared, dumpable rule table* RFC-0021 §4.2
//! ultimately wants (so the rules are inspectable as source, but not yet as data); and this is a
//! **read-only** projection (no `RoundTrip` parse-back — RFC-0021 §4.1 `EditCapability::ReadOnly`).
//! Both are noted as follow-ups, not claimed as done.

use mycelium_core::{
    Alt, GuaranteeStrength, Node, Payload, Repr, ScalarKind, SparsityClass, Trit, Value,
};
use mycelium_workstack::{ensure_sufficient_stack, BudgetError, ProcessArena, RecursionBudget};

/// RFC-0041 §4.2/§9 process-wide arena ceiling this projection reserves against before rendering.
/// **Declared** — an asserted operational default, not a measured bound: every concurrent
/// [`llm_canonical`] call (LSP re-analyses over untrusted editor buffers, per §5 untrusted-input
/// coverage) charges the *same* process-global counter
/// ([`mycelium_workstack::current_process_bytes`]), so the sum across concurrent renders — not just
/// one render's own size — is what this ceiling bounds. Each consuming crate declares its own
/// operational default (consumer-side wiring, mirroring the per-invocation `RecursionBudget`
/// pattern); a single shared constant is a follow-up centralization item, not introduced silently
/// here (`docs/notes/W7-arena-coverage-audit.md`).
const PROCESS_ARENA_CEILING_BYTES: u64 = 256 * 1024 * 1024;

/// A conservative, **Declared** (not measured) per-node byte estimate for the rendered s-expression,
/// used only to *size the pre-render reservation* — never to bound the actual output. An
/// under-estimate would only under-charge the arena (never cause a false refusal); this constant is
/// picked generously relative to the shortest possible per-node rendering (a bare variable name or
/// `(const ...)`) so the reservation stays a conservative upper bound in practice, without being
/// `Proven`.
const ESTIMATED_BYTES_PER_NODE: u64 = 64;

/// Render a closed Core IR [`Node`] as the `LlmCanonical` s-expression surface (RFC-0021 §4.6).
/// **Total** over every node kind (enforced by the exhaustive `match`) and **deterministic** (a pure
/// function of the node). Preserves the honesty overlay (P2/P3).
///
/// **RFC-0041 §4.7 W1 (RR-29 guard-hole census):** `Node` is a post-elaboration tree that can exceed
/// the L1 parser's 256-frame depth (an editor buffer is untrusted input), and `render_node` recurses
/// one stack frame per nesting level. This is the LSP's **outermost public entry point** for the
/// render, so it — and only it — wraps the recursion in [`ensure_sufficient_stack`]: a deep buffer
/// now renders on the grown worker stack instead of a SIGABRT that would crash the language server.
/// The budget's depth ceiling is not yet consulted by the W1 host-stack grow (that lands in W2); this
/// call site exists so W2's fine-grained guard swaps in with **no signature change** here.
///
/// **RFC-0041 §4.2/§9 W7 (process-arena coverage).** This projection allocates a `String` proportional
/// to the node count — the process-wide arena hole the §9 DoD names (`docs/notes/W7-arena-coverage-audit.md`
/// item 1): an editor buffer is untrusted, and many concurrent LSP re-analyses could otherwise sum past
/// any single-pass ceiling. **Signature change (contained):** `llm_canonical` was infallible (`String`)
/// before this wave; it is now `Result<String, BudgetError>` so an over-ceiling reservation refuses
/// never-silently (G2) instead of proceeding unbounded. This is a **local, contained** change — the
/// function is `pub` but not re-exported at the crate root, and every call site is inside this crate
/// (tests) — not a ripple into the trusted base (mycelium-interp/mycelium-mlir).
///
/// # Errors
/// [`BudgetError::OutOfBudget`] (`kind = Bytes`) when the pre-render reservation would push the
/// process-wide arena total over its ceiling.
pub fn llm_canonical(node: &Node) -> Result<String, BudgetError> {
    let arena = ProcessArena::new(PROCESS_ARENA_CEILING_BYTES);
    llm_canonical_with_arena(node, &arena)
}

/// [`llm_canonical`], parameterized over an explicit [`ProcessArena`] (`pub(crate)`, so this crate's
/// own tests can inject a tiny-ceiling arena and witness a real refusal — production callers always
/// go through [`llm_canonical`], which supplies the crate's declared default ceiling).
pub(crate) fn llm_canonical_with_arena(
    node: &Node,
    arena: &ProcessArena,
) -> Result<String, BudgetError> {
    let budget = RecursionBudget::with_depth_default(u64::MAX, u64::MAX);
    ensure_sufficient_stack(&budget, || {
        // Pre-flight estimate: computed on the same grown worker stack as the render itself, since a
        // pathologically deep `Node` would overflow an un-guarded count just as readily as the render.
        let estimate = node_count(node).saturating_mul(ESTIMATED_BYTES_PER_NODE);
        let _reservation = arena.reserve(estimate)?;
        Ok(render_node(node))
    })
}

/// Total node count over the same 11-node grammar `render_node` matches — the pre-flight sizing basis
/// for the arena reservation above. `Exact` (a deterministic structural count); the *bytes-per-node*
/// multiplier applied to it is the `Declared` part, not this count itself.
fn node_count(n: &Node) -> u64 {
    1 + match n {
        Node::Const(_) | Node::Var(_) => 0,
        Node::Let { bound, body, .. } => node_count(bound) + node_count(body),
        Node::Op { args, .. } | Node::Construct { args, .. } => {
            args.iter().map(node_count).sum::<u64>()
        }
        Node::Swap { src, .. } => node_count(src),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            node_count(scrutinee)
                + alts.iter().map(alt_node_count).sum::<u64>()
                + default.as_deref().map_or(0, node_count)
        }
        Node::Lam { body, .. } | Node::Fix { body, .. } => node_count(body),
        Node::App { func, arg } => node_count(func) + node_count(arg),
        Node::FixGroup { defs, body } => {
            defs.iter().map(|(_, d)| node_count(d)).sum::<u64>() + node_count(body)
        }
    }
}

fn alt_node_count(a: &Alt) -> u64 {
    match a {
        Alt::Ctor { body, .. } | Alt::Lit { body, .. } => node_count(body),
    }
}

fn render_node(n: &Node) -> String {
    match n {
        Node::Const(v) => format!("(const {})", render_value(v)),
        Node::Var(x) => x.clone(),
        Node::Let { id, bound, body } => {
            format!("(let [{id} {}] {})", render_node(bound), render_node(body))
        }
        Node::Op { prim, args } => format!("(op {prim}{})", render_args(args)),
        // P3: a Swap is ALWAYS rendered, explicitly, with its target + policy — never elided.
        Node::Swap {
            src,
            target,
            policy,
        } => format!(
            "(swap! {} :to {} :policy {})",
            render_node(src),
            render_repr(target),
            policy.as_str()
        ),
        Node::Construct { ctor, args } => format!("(make {ctor}{})", render_args(args)),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => render_match(scrutinee, alts, default.as_deref()),
        Node::Lam { param, body } => format!("(fn [{param}] {})", render_node(body)),
        Node::App { func, arg } => format!("({} {})", render_node(func), render_node(arg)),
        Node::Fix { name, body } => format!("(fix {name} {})", render_node(body)),
        Node::FixGroup { defs, body } => {
            let binds: String = defs
                .iter()
                .map(|(name, def)| format!("[{name} {}]", render_node(def)))
                .collect::<Vec<_>>()
                .join(" ");
            format!("(fix-group ({binds}) {})", render_node(body))
        }
    }
}

/// Space-prefixed operand list, e.g. ` a b c` (empty for no args).
fn render_args(args: &[Node]) -> String {
    args.iter()
        .map(|a| format!(" {}", render_node(a)))
        .collect()
}

fn render_match(scrutinee: &Node, alts: &[Alt], default: Option<&Node>) -> String {
    let mut arms: Vec<String> = alts.iter().map(render_alt).collect();
    if let Some(d) = default {
        arms.push(format!("(_ {})", render_node(d)));
    }
    format!("(match {} {})", render_node(scrutinee), arms.join(" "))
}

fn render_alt(alt: &Alt) -> String {
    match alt {
        Alt::Ctor {
            ctor,
            binders,
            body,
        } => {
            let pat = if binders.is_empty() {
                format!("{ctor}")
            } else {
                format!("({ctor} {})", binders.join(" "))
            };
            format!("({pat} {})", render_node(body))
        }
        Alt::Lit { value, body } => format!("({} {})", render_value(value), render_node(body)),
    }
}

/// Render a constant value: its literal (repr + payload) plus the honesty overlay — the guarantee tag
/// always (P2), and bound presence when approximate (P6-adjacent).
fn render_value(v: &Value) -> String {
    let lit = render_payload(v.repr(), v.payload());
    let guar = guar_str(v.meta().guarantee());
    let bound = if v.meta().bound().is_some() {
        " :bound"
    } else {
        ""
    };
    format!("{lit} @{guar}{bound}")
}

fn guar_str(g: GuaranteeStrength) -> &'static str {
    match g {
        GuaranteeStrength::Exact => "Exact",
        GuaranteeStrength::Proven => "Proven",
        GuaranteeStrength::Empirical => "Empirical",
        GuaranteeStrength::Declared => "Declared",
    }
}

/// Above this width, a literal payload is summarized by its length rather than inlined element-wise.
/// `Binary{width}`/`Ternary{trits}` are only constrained `> 0` (`mycelium-core::repr`), so a `Const`
/// can carry an arbitrarily large payload; rendering every element would be O(width). The summary is
/// honest (the value is never dropped — it states its shape), mirroring the VSA-hypervector case.
const INLINE_MAX: usize = 64;

fn render_payload(repr: &Repr, payload: &Payload) -> String {
    match (repr, payload) {
        (Repr::Binary { .. }, Payload::Bits(bits)) => {
            if bits.len() > INLINE_MAX {
                format!("0b<{} bits>", bits.len())
            } else {
                let s: String = bits.iter().map(|&b| if b { '1' } else { '0' }).collect();
                format!("0b{s}")
            }
        }
        (Repr::Ternary { .. }, Payload::Trits(trits)) => {
            if trits.len() > INLINE_MAX {
                format!("<{} trits>", trits.len())
            } else {
                let s: String = trits
                    .iter()
                    .map(|t| match t {
                        Trit::Neg => '-',
                        Trit::Zero => '0',
                        Trit::Pos => '+',
                    })
                    .collect();
                format!("<{s}>")
            }
        }
        (Repr::Dense { dtype, .. }, Payload::Scalars(xs)) => {
            let s: String = xs
                .iter()
                .map(|x| format!("{x}"))
                .collect::<Vec<_>>()
                .join(" ");
            format!("[{s}]:{}", scalar_str(*dtype))
        }
        (
            Repr::Vsa {
                model, sparsity, ..
            },
            Payload::Hypervector(xs),
        ) => {
            // The hypervector content is not literal-inlined (it is high-dimensional); render its
            // shape honestly so the projection stays total and never silently drops the value.
            format!("<hv:{model}/{}{}>", xs.len(), sparsity_str(sparsity))
        }
        // A payload that does not match its repr cannot occur for a well-formed `Value` (constructed
        // through `Value::new`); render it explicitly rather than panicking (never silent).
        (r, _) => format!("<malformed-value:{}>", render_repr(r)),
    }
}

fn render_repr(r: &Repr) -> String {
    match r {
        Repr::Binary { width } => format!("Binary{{{width}}}"),
        Repr::Ternary { trits } => format!("Ternary{{{trits}}}"),
        Repr::Dense { dim, dtype } => format!("Dense{{{dim},{}}}", scalar_str(*dtype)),
        Repr::Vsa {
            model,
            dim,
            sparsity,
        } => format!("VSA{{{model},{dim}{}}}", sparsity_str(sparsity)),
        // RFC-0032 D3 (M-749): the indexed-sequence repr renders its element type and length.
        Repr::Seq { elem, len } => format!("Seq{{{},{len}}}", render_repr(elem)),
        // RFC-0032 D4 (M-750): the byte-string repr.
        Repr::Bytes => "Bytes".to_owned(),
        // ADR-040 (M-896): the scalar-float repr renders its frozen width by name (F64-only today).
        Repr::Float { .. } => "Float{F64}".to_owned(),
    }
}

fn scalar_str(k: ScalarKind) -> &'static str {
    match k {
        ScalarKind::F16 => "F16",
        ScalarKind::Bf16 => "BF16",
        ScalarKind::F32 => "F32",
        ScalarKind::F64 => "F64",
    }
}

fn sparsity_str(s: &SparsityClass) -> String {
    match s {
        SparsityClass::Dense => String::new(),
        SparsityClass::Sparse { max_active } => format!(",sparse<={max_active}"),
    }
}
