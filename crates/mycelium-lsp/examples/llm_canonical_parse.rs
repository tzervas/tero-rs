//! `llm_canonical_parse` — stdin→stdout LlmCanonical S-expression validator (M-381 Arm 4).
//!
//! Reads LlmCanonical source from stdin, validates it with `parse_llm_canonical`, and writes
//! the normalized Core-IR S-expression to stdout. Exits 0 on success, 1 on parse error.
//!
//! Used by the Grok harness (Arm 4) to validate LLM-generated LlmCanonical output before
//! scoring with `myc-check`. The harness calls:
//!
//! ```text
//! echo "$llm_output" | cargo run --example llm_canonical_parse -p mycelium-lsp -q
//! ```
//!
//! Exit codes:
//!   0  → valid LlmCanonical; normalized S-expression written to stdout
//!   1  → parse error; human-readable message written to stderr
//!
//! # Honesty (G2 / VR-5)
//! This binary is a thin wrapper around `parse_llm_canonical` (Empirical guarantee tag).
//! It never suppresses errors and never writes partial output on failure.

use std::io::Read;
use std::process::ExitCode;

use mycelium_lsp::parse_llm_canonical;

fn main() -> ExitCode {
    let mut source = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut source) {
        eprintln!("llm-canonical-parse: I/O error reading stdin: {e}");
        return ExitCode::from(1);
    }
    match parse_llm_canonical(&source) {
        Ok(sexp) => {
            println!("{sexp}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("llm-canonical-parse: {e}");
            ExitCode::from(1)
        }
    }
}
