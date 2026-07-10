//! The **representation-crossing audit view** (RFC-0013 §4.6) — routed here from RFC-0012 R12-Q2
//! (M-351): *"where do my lossy / precision-changing conversions live, and under what honesty
//! bound?"*, delivered as a structured diagnostic **without constraining where crossings live**.
//!
//! It enumerates **every `swap`** in a program (every representation crossing), wherever it sits
//! (I5: location-independent — it does not care whether a crossing is at a block edge or buried in an
//! expression), and reports each crossing's location, from/to representation, **honesty bound** on
//! the lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`, and selection policy. The honesty bound is
//! **read off** each crossing's certificate and is **never upgraded** (VR-5): a crossing whose bound
//! cannot be derived statically reports `None` (unknown), never a fabricated `Exact`.

use mycelium_cert::{binary_to_ternary, ternary_to_binary, SwapCertificate};
use mycelium_core::{GuaranteeStrength, Node, Repr};
use serde::Serialize;

/// One representation crossing (`swap` site) and what the audit can read off it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Crossing {
    /// Breadcrumb to the crossing.
    pub site: String,
    /// The source representation, when statically known (`None` for a non-constant source).
    pub from: Option<Repr>,
    /// The target representation (always known — it is the swap's static contract).
    pub to: Repr,
    /// The crossing's honesty bound, **read off** its certificate and **never upgraded** (VR-5).
    /// `None` when no certificate is statically derivable here (unknown ≠ exact).
    pub honesty: Option<GuaranteeStrength>,
    /// The selection policy that chose/justified the target (the RFC-0005 `PolicyRef`).
    pub policy: String,
}

/// The audit view: every crossing in a program, in deterministic traversal order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AuditView {
    /// Every representation crossing, wherever it sits (I5).
    pub crossings: Vec<Crossing>,
}

impl AuditView {
    /// Build the audit view for a Core IR program — enumerating **every** `swap` (I5).
    #[must_use]
    pub fn of(node: &Node) -> Self {
        let mut crossings = Vec::new();
        walk(node, "", &mut crossings);
        AuditView { crossings }
    }

    /// The JSON projection (§4.3 dual-projection form — this view is read-only structured output).
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("an audit view serializes")
    }

    /// The human projection: one line per crossing, honesty bound named (or `unknown`, never faked).
    #[must_use]
    pub fn to_human(&self) -> String {
        if self.crossings.is_empty() {
            return "no representation crossings".to_owned();
        }
        let mut out = String::new();
        for (i, c) in self.crossings.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let from = c
                .from
                .as_ref()
                .map_or_else(|| "?".to_owned(), |r| format!("{r:?}"));
            let honesty = c
                .honesty
                .map_or_else(|| "unknown".to_owned(), |g| format!("{g:?}"));
            out.push_str(&format!(
                "{}: {} → {:?}  [honesty: {}]  policy: {}",
                c.site, from, c.to, honesty, c.policy
            ));
        }
        out
    }
}

/// The honesty bound a swap certificate justifies — **read, never upgraded** (VR-5). A `Bijective`
/// (exact-within-range) crossing is `Exact`; a `Bounded` crossing reports the *basis* strength of its
/// bound (`Proven`/`Empirical`/`Declared`) — exactly what the kernel recorded, never stronger.
fn honesty_of(cert: &SwapCertificate) -> GuaranteeStrength {
    match cert {
        SwapCertificate::Bijective { .. } => GuaranteeStrength::Exact,
        SwapCertificate::Bounded { bound, .. } => bound.basis.strength(),
    }
}

/// Derive the crossing's honesty from a statically-known source, or `None` when it cannot be derived
/// here (a non-constant source, or a pair/instance with no implemented certifier). `None` is honest
/// "unknown" — never silently upgraded to `Exact` (VR-5).
fn static_honesty(
    src: &Node,
    target: &Repr,
    policy: &mycelium_core::ContentHash,
) -> Option<GuaranteeStrength> {
    let Node::Const(v) = src else {
        return None;
    };
    let cert = match (v.repr(), target) {
        (Repr::Binary { .. }, Repr::Ternary { trits }) => binary_to_ternary(v, *trits, policy).ok(),
        (Repr::Ternary { .. }, Repr::Binary { width }) => ternary_to_binary(v, *width, policy).ok(),
        _ => None,
    };
    cert.map(|(_, c)| honesty_of(&c))
}

fn here(prefix: &str, step: &str) -> String {
    if prefix.is_empty() {
        step.to_owned()
    } else {
        format!("{prefix}/{step}")
    }
}

/// Walk every node, recording a [`Crossing`] at each `Swap` — wherever it sits (I5).
fn walk(node: &Node, prefix: &str, out: &mut Vec<Crossing>) {
    match node {
        Node::Const(_) | Node::Var(_) => {}
        Node::Let { id, bound, body } => {
            let at = here(prefix, &format!("let {id}"));
            walk(bound, &at, out);
            walk(body, &at, out);
        }
        Node::Op { prim, args } => {
            let at = here(prefix, &format!("op {prim}"));
            for a in args {
                walk(a, &at, out);
            }
        }
        Node::Swap {
            src,
            target,
            policy,
        } => {
            let at = here(prefix, "swap");
            let from = match src.as_ref() {
                Node::Const(v) => Some(v.repr().clone()),
                _ => None,
            };
            out.push(Crossing {
                site: at.clone(),
                from,
                to: target.clone(),
                honesty: static_honesty(src, target, policy),
                policy: policy.as_str().to_owned(),
            });
            walk(src, &at, out);
        }
        Node::Construct { ctor, args } => {
            let at = here(prefix, &format!("construct {ctor}"));
            for a in args {
                walk(a, &at, out);
            }
        }
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let at = here(prefix, "match");
            walk(scrutinee, &at, out);
            for alt in alts {
                match alt {
                    mycelium_core::Alt::Ctor { ctor, body, .. } => {
                        walk(body, &here(&at, &format!("alt {ctor}")), out);
                    }
                    mycelium_core::Alt::Lit { body, .. } => {
                        walk(body, &here(&at, "alt-lit"), out);
                    }
                }
            }
            if let Some(d) = default {
                walk(d, &here(&at, "default"), out);
            }
        }
        Node::Lam { body, .. } => walk(body, &here(prefix, "lam"), out),
        Node::App { func, arg } => {
            let at = here(prefix, "app");
            walk(func, &at, out);
            walk(arg, &at, out);
        }
        Node::Fix { body, .. } => walk(body, &here(prefix, "fix"), out),
        Node::FixGroup { defs, body } => {
            let at = here(prefix, "fixgroup");
            for (name, def) in defs {
                walk(def, &here(&at, &format!("def {name}")), out);
            }
            walk(body, &at, out);
        }
    }
}
