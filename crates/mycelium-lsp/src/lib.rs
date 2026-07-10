//! `mycelium-lsp` — the **minimal toolchain surface** (FR-S5; Foundation §5.8): the invariant
//! linter (M-141), the canonical formatter (M-142), and the LSP feedback facade (M-140) that
//! exposes the semantic-feedback artifact kinds over one surface (SC-5 channel) — the four
//! Phase-1 kinds plus, since M-221, the **selection EXPLAIN channel** (RFC-0005 §4: "why was
//! this representation chosen?").
//!
//! This is a *toolchain* crate, deliberately kept out of the small auditable kernel (KC-3): it
//! depends on `mycelium-core`/`-interp`/`-cert`/`-select` but nothing depends on it.

pub mod baseline;
pub mod completions;
pub mod definition;
pub mod diagnostics;
pub mod expand;
pub mod feedback;
pub mod fmt;
pub mod hover;
pub mod lint;
pub mod llm_canonical_parser;
pub mod project;
pub mod recover;
pub mod semantic;
mod span;
pub mod sync;
pub mod wire;

#[cfg(test)]
mod tests;

pub use baseline::{
    baseline_for_class, derive_baseline, derive_baseline_for, explain_baseline, recovery_profile,
    BaselineRule, RecoveryProfile, RESILIENT_MAX_ATTEMPTS,
};
pub use completions::{completion_list, CompletionItem, KEYWORD_COMPLETIONS, SNIPPET_COMPLETIONS};
pub use definition::definition;
pub use diagnostics::{
    present, AuditView, ClassRegistry, Crossing, DiagnosticPolicy, DiagnosticRecord, Level,
    Presentation, ReasonedError, Rule, UnknownClass,
};
pub use expand::expand_ambient;
pub use feedback::{
    analyze, analyze_with, ExplainSite, Feedback, FeedbackSummary, GuaranteeAnnotation, SwapSite,
};
pub use fmt::format;
pub use hover::hover;
pub use lint::{
    has_errors, lint, lint_nodule_header, lint_structured_header, Diagnostic, Severity,
};
pub use llm_canonical_parser::{parse_llm_canonical, ParseError, DEPTH_LIMIT};
pub use recover::{
    check_effects, handle, EffectBudget, EffectBudgetExhausted, EffectKind, Outcome,
    RecoveryPolicy, Resolution, StructuredError, UndeclaredEffect as RecoverUndeclaredEffect,
};
pub use semantic::{semantic_tokens_full, semantic_tokens_legend, TOKEN_TYPES};
pub use sync::{
    publish_for_source, resilient_publish_for_source, resilient_source_diagnostics,
    source_diagnostics, DocumentStore,
};
pub use wire::{
    publish_diagnostics_notification, read_message, serve, serve_stdio, to_lsp_diagnostic,
    write_message,
};
