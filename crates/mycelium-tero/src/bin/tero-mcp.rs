//! `tero-mcp` — the MCP (Model Context Protocol) front for the mycelium-tero memory engine
//! (M-1017 / DN-87 §2.3). Speaks newline-delimited JSON-RPC 2.0 over stdio, so an MCP client
//! launches it as a subprocess and calls its tools natively. Token-scoped, read-only by default.
//!
//! Usage (an MCP client sets `TERO_TOKENS` in the launched process's environment):
//! ```text
//!   TERO_TOKENS='s3cr3t:read admin:refresh' tero-mcp [--index docs/tero-index/index.json]
//! ```
//!
//! The server **refuses to start** with no tokens configured. Exit codes: `0` ok · `64` usage ·
//! `66` I/O · `78` config (no tokens). Never-silent (G2): failures are explicit stderr messages.

use std::path::PathBuf;
use std::process::ExitCode;

use mycelium_tero::{load_report, serve_mcp_stdio, tool_descriptors, TokenTable, SERVER_NAME};
use serde_json::json;

const EX_OK: u8 = 0;
const EX_USAGE: u8 = 64;
const EX_IO: u8 = 66;
const EX_CONFIG: u8 = 78;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => ExitCode::from(code),
        Err((code, msg)) => {
            eprintln!("tero-mcp: {msg}");
            ExitCode::from(code)
        }
    }
}

fn run(args: &[String]) -> Result<u8, (u8, String)> {
    let mut index = PathBuf::from("docs/tero-index/index.json");
    let mut describe = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => {
                i += 1;
                index = PathBuf::from(args.get(i).ok_or((EX_USAGE, usage()))?);
            }
            "--describe" => {
                describe = true;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(EX_OK);
            }
            other => return Err((EX_USAGE, format!("unknown argument: {other}\n{}", usage()))),
        }
        i += 1;
    }

    if describe {
        // Discovery mode for Python wrapper / dynamic tool generation.
        // Does not require TERO_TOKENS or loading the index (static surface).
        let spec = json!({
            "name": SERVER_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "tools": tool_descriptors(),
            // Future: "categories", "groups", full operations beyond MCP tools, etc.
        });
        println!("{}", serde_json::to_string_pretty(&spec).unwrap());
        return Ok(EX_OK);
    }

    // Never-silent: refuse to start without tokens (no anonymous default).
    let tokens = TokenTable::from_env().map_err(|e| (EX_CONFIG, e.to_string()))?;
    let report =
        load_report(&index).map_err(|e| (EX_IO, format!("loading {}: {e}", index.display())))?;

    serve_mcp_stdio(report, tokens, false, index)
        .map_err(|e| (EX_IO, format!("mcp stdio: {e}")))?;
    Ok(EX_OK)
}

fn usage() -> String {
    "usage: TERO_TOKENS='<token>:<read|refresh> ...' tero-mcp [--index <index.json>] [--describe]".to_owned()
}
