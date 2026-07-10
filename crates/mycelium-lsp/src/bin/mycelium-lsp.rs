//! The `mycelium-lsp` **executable** (M-310; FR-S5; SC-5): a thin entry point that runs the
//! JSON-RPC server loop ([`mycelium_lsp::serve`]) over the process's real stdin/stdout — the
//! transport every LSP client speaks. All protocol behaviour lives in the library
//! ([`mycelium_lsp::wire`]); this binary only wires it to stdio and maps the loop's outcome to a
//! process exit code.
//!
//! **Never-silent at the transport edge (G2 / RFC-0013 I1).** A document that fails analysis is a
//! *diagnostic*, not a crash — the loop surfaces it and stays alive. A *transport* failure (a
//! malformed JSON-RPC frame, a broken pipe) is unrecoverable: it is reported on stderr and the
//! process exits non-zero, never dropped silently.

use std::process::ExitCode;

fn main() -> ExitCode {
    match mycelium_lsp::serve_stdio() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mycelium-lsp: transport error: {e}");
            ExitCode::FAILURE
        }
    }
}
