//! **LSP check shim** (M-330; SC-5b; NFR-2): reads Mycelium source from stdin, runs
//! `publish_for_source`, and writes the full `textDocument/publishDiagnostics` JSON
//! notification to stdout — one line, no framing.
//!
//! This is the subprocess oracle that `tools/llm-harness/coauthor.py` calls on every
//! generate→check round. No LSP server is started; the analysis is in-process (pure lib
//! call), so latency is minimal and there is no socket lifecycle to manage.
//!
//! Design posture:
//! - **Never-silent (G2)**: any I/O error exits non-zero with a diagnostic on stderr.
//! - **No extra deps**: `publish_for_source` is already in `mycelium-lsp::sync`; this
//!   binary adds zero new dependencies.
//! - **Honest output**: the notification's `diagnostics` array is empty for a clean
//!   nodule and non-empty for parse/type errors — the caller checks `diagnostics.len()`.

use std::io::{self, Read};

fn main() {
    let mut src = String::new();
    if let Err(e) = io::stdin().lock().read_to_string(&mut src) {
        eprintln!("check: failed to read stdin: {e}");
        std::process::exit(2);
    }
    let note = mycelium_lsp::sync::publish_for_source("mem://coauthor", &src);
    println!("{note}");
}
