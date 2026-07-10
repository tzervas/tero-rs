//! The **invariant linter** (M-141; SC-3; G2; FR-M3; VR-5).
//!
//! Static, inspectable checks over a Core IR [`Node`] that surface the house honesty rules as
//! [`Diagnostic`]s for authoring tools (the LSP, M-140). The lints:
//!
//! - **`implicit-swap`** (error) — an `Op` whose `Const` operands span *more than one paradigm*
//!   (binary/ternary/dense/vsa). A representation change must be an explicit [`Node::Swap`], never
//!   implied by feeding mixed-paradigm operands to an op (FR-M3; SC-3 "no implicit conversion").
//! - **`unverified-bound`** (warning) — a value carrying a `Declared` guarantee. A user-asserted,
//!   unvalidated bound must *always* be surfaced; it is never silently trusted (M-I4; VR-5).
//! - **`placeholder-policy`** (error) — a [`Node::Swap`] whose `policy` is a stub (an all-zero
//!   digest, or `todo`/`tbd`/`none`/`placeholder`) rather than a real `PolicyRef` (G2: a swap's
//!   selection must be reified, not faked).
//! - **`free-variable`** (error) — a `Var` with no enclosing binder (an open term the interpreter
//!   cannot run).
//!
//! Note WF1 (only `Swap` changes a representation) and WF2 (every `Swap` carries a `PolicyRef`) are
//! enforced *by construction* in the `Node` grammar, so a literally repr-changing non-`Swap` node or
//! a policy-less swap is unrepresentable; these lints catch the *spirit* of those rules at the level
//! where authoring mistakes actually occur (mixed-paradigm ops, stub policies).

use mycelium_core::{GuaranteeStrength, Node, Repr, Value};

/// Severity of a [`Diagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A house-rule violation that should block (honesty / never-silent).
    Error,
    /// An advisory the author must see (e.g. an unverified `Declared` value).
    Warning,
}

/// A single lint finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Stable lint code (e.g. `"implicit-swap"`).
    pub code: &'static str,
    /// Severity.
    pub severity: Severity,
    /// A breadcrumb path to the offending node (e.g. `"let a/swap/op f"`).
    pub at: String,
    /// Human-readable explanation.
    pub message: String,
}

impl Diagnostic {
    /// The breadcrumb [`Self::at`] as a structured, navigable path (split on `/`) — so a client can
    /// locate the offending node step-by-step rather than parsing the string (M-310). An empty
    /// breadcrumb (the program root) yields an empty path.
    #[must_use]
    pub fn path(&self) -> Vec<&str> {
        if self.at.is_empty() {
            Vec::new()
        } else {
            self.at.split('/').collect()
        }
    }
}

/// Lint a (closed) Core IR program, returning all findings in deterministic traversal order.
#[must_use]
pub fn lint(node: &Node) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let mut scope: Vec<&str> = Vec::new();
    walk(node, "", &mut scope, &mut out);
    out
}

/// Whether `lint` found at least one `Error`-severity diagnostic.
#[must_use]
pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.severity == Severity::Error)
}

/// The **source-text** companion lint (M-141; DN-06 §6): recognise the `// nodule:` header marker
/// on a document's first non-blank line and surface a malformed *named* marker as an explicit
/// `Error` (never silently dropped — G2). A well-formed marker (or its absence) yields nothing;
/// this is a *recogniser*, not a requirement (a file need not declare a nodule). The Core-IR [`lint`]
/// runs over the elaborated program; this runs over raw text, where the comment marker lives.
#[must_use]
pub fn lint_nodule_header(src: &str) -> Vec<Diagnostic> {
    match mycelium_l1::parse_nodule_header(src) {
        Ok(_) => Vec::new(),
        Err(e) => vec![Diagnostic {
            code: "nodule-header",
            severity: Severity::Error,
            at: format!("line {}", e.line),
            message: format!(
                "malformed `// nodule:` header marker (DN-06 §6): {} — a near-miss marker is \
                 flagged, never silently ignored (G2)",
                e.message
            ),
        }],
    }
}

/// The **structured-header** lint (M-141; M-359 / spec §3): parse the `// @key: value` header and
/// surface any malformed marker, unknown/duplicate key, or bad value (non-SPDX `@license`, non-ISO
/// `@since`/`@updated`, ill-formed `@version`) as an explicit `Error` — never silently ignored (G2).
/// A well-formed header (or a file with none) yields nothing. Supersedes the bare-marker
/// [`lint_nodule_header`] when the richer M-359 metadata is wanted; both run over raw text.
#[must_use]
pub fn lint_structured_header(src: &str) -> Vec<Diagnostic> {
    match mycelium_proj::parse_header(src) {
        Ok(_) => Vec::new(),
        Err(e) => vec![Diagnostic {
            code: "nodule-header",
            severity: Severity::Error,
            at: format!("line {}", e.line),
            message: format!(
                "malformed nodule header (DN-06 §6 / M-359 spec §3): {} — a header defect is flagged, \
                 never silently ignored (G2)",
                e.message
            ),
        }],
    }
}

fn paradigm(repr: &Repr) -> &'static str {
    match repr {
        Repr::Binary { .. } => "binary",
        Repr::Ternary { .. } => "ternary",
        Repr::Dense { .. } => "dense",
        Repr::Vsa { .. } => "vsa",
        // RFC-0032 D3 (M-749): the indexed-sequence repr.
        Repr::Seq { .. } => "seq",
        // RFC-0032 D4 (M-750): the byte-string repr.
        Repr::Bytes => "bytes",
        // ADR-040 (M-896): the scalar-float repr.
        Repr::Float { .. } => "float",
    }
}

/// A policy reference that is a stub rather than a real reified policy.
fn is_placeholder_policy(policy: &mycelium_core::ContentHash) -> bool {
    let d = policy.digest();
    d.bytes().all(|b| b == b'0') || matches!(d, "todo" | "tbd" | "none" | "placeholder")
}

fn here(prefix: &str, step: &str) -> String {
    if prefix.is_empty() {
        step.to_owned()
    } else {
        format!("{prefix}/{step}")
    }
}

fn check_value(v: &Value, at: &str, out: &mut Vec<Diagnostic>) {
    if v.meta().guarantee() == GuaranteeStrength::Declared {
        out.push(Diagnostic {
            code: "unverified-bound",
            severity: Severity::Warning,
            at: at.to_owned(),
            message: "value carries a Declared (user-asserted, unvalidated) bound — surface it; \
                      never trust it silently (VR-5/M-I4)"
                .to_owned(),
        });
    }
}

fn walk<'a>(node: &'a Node, prefix: &str, scope: &mut Vec<&'a str>, out: &mut Vec<Diagnostic>) {
    match node {
        Node::Const(v) => check_value(v, &here(prefix, "const"), out),
        Node::Var(x) => {
            if !scope.iter().rev().any(|b| b == x) {
                out.push(Diagnostic {
                    code: "free-variable",
                    severity: Severity::Error,
                    at: here(prefix, &format!("var {x}")),
                    message: format!("`{x}` is not bound by any enclosing `let` (open term)"),
                });
            }
        }
        Node::Let { id, bound, body } => {
            let at = here(prefix, &format!("let {id}"));
            walk(bound, &at, scope, out);
            scope.push(id);
            walk(body, &at, scope, out);
            scope.pop();
        }
        Node::Op { prim, args } => {
            let at = here(prefix, &format!("op {prim}"));
            // implicit-swap: mixed-paradigm Const operands imply a conversion the author must make
            // explicit with a Swap.
            let mut paradigms: Vec<&str> = args
                .iter()
                .filter_map(|a| match a {
                    Node::Const(v) => Some(paradigm(v.repr())),
                    _ => None,
                })
                .collect();
            paradigms.sort_unstable();
            paradigms.dedup();
            if paradigms.len() > 1 {
                out.push(Diagnostic {
                    code: "implicit-swap",
                    severity: Severity::Error,
                    at: at.clone(),
                    message: format!(
                        "op `{prim}` mixes paradigms [{}] — insert an explicit `swap` (no implicit conversion; FR-M3/SC-3)",
                        paradigms.join(", ")
                    ),
                });
            }
            for a in args {
                walk(a, &at, scope, out);
            }
        }
        Node::Swap {
            src,
            target,
            policy,
        } => {
            let at = here(prefix, &format!("swap -> {}", paradigm(target)));
            if is_placeholder_policy(policy) {
                out.push(Diagnostic {
                    code: "placeholder-policy",
                    severity: Severity::Error,
                    at: at.clone(),
                    message: format!(
                        "swap references a placeholder policy `{}` — a swap must cite a real reified PolicyRef (G2)",
                        policy.as_str()
                    ),
                });
            }
            walk(src, &at, scope, out);
        }
        // r3 (RFC-0011): the data nodes are Repr-transparent (WF8 — no swap to lint), so the walk
        // just descends; a `Match` constructor arm binds its field variables for the body's scope
        // (so a binder use is not mis-flagged as a free variable).
        Node::Construct { ctor, args } => {
            let at = here(prefix, &format!("construct {ctor}"));
            for a in args {
                walk(a, &at, scope, out);
            }
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let at = here(prefix, "match");
            walk(scrutinee, &at, scope, out);
            for alt in alts {
                match alt {
                    mycelium_core::Alt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        let arm_at = here(&at, &format!("alt {ctor}"));
                        let mark = scope.len();
                        for b in binders {
                            scope.push(b);
                        }
                        walk(body, &arm_at, scope, out);
                        scope.truncate(mark);
                    }
                    mycelium_core::Alt::Lit { body, .. } => {
                        walk(body, &here(&at, "alt-lit"), scope, out);
                    }
                }
            }
            if let Some(d) = default {
                walk(d, &here(&at, "default"), scope, out);
            }
        }
        // r4 (RFC-0001 r4): a Lam/Fix binder enters scope for its body (so a bound use is not
        // mis-flagged as a free variable); App just descends. Repr-transparent (no swap to lint).
        Node::Lam { param, body } => {
            let at = here(prefix, &format!("lam {param}"));
            scope.push(param);
            walk(body, &at, scope, out);
            scope.pop();
        }
        Node::App { func, arg } => {
            let at = here(prefix, "app");
            walk(func, &at, scope, out);
            walk(arg, &at, scope, out);
        }
        Node::Fix { name, body } => {
            let at = here(prefix, &format!("fix {name}"));
            scope.push(name);
            walk(body, &at, scope, out);
            scope.pop();
        }
        Node::FixGroup { defs, body } => {
            // r5: the group binds every member name; they enter scope together for all definitions
            // and the continuation (so a sibling use is not mis-flagged free). Repr-transparent.
            let at = here(prefix, "fixgroup");
            let mark = scope.len();
            for (name, _) in defs {
                scope.push(name);
            }
            for (name, def) in defs {
                walk(def, &here(&at, &format!("def {name}")), scope, out);
            }
            walk(body, &at, scope, out);
            scope.truncate(mark);
        }
    }
}
