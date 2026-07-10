//! `tero-http` — the HTTP/JSON front for the mycelium-tero memory engine (M-1017 / DN-87 §2.3).
//! Serves the M-1016 query engine over a plain, versioned, curl-able HTTP API (the universal floor
//! for any agent platform). Token-scoped, read-only by default.
//!
//! Usage:
//! ```text
//!   TERO_TOKENS='s3cr3t:read admin:refresh' tero-http [--index docs/tero-index/index.json] [--addr 127.0.0.1:8787]
//! ```
//!
//! Tokens are supplied at runtime via `TERO_TOKENS` (a `token:scope` list) or `TERO_TOKENS_FILE`;
//! the server **refuses to start** with no tokens configured (never an accidentally-open server).
//!
//! Exit codes: `0` ok · `64` usage · `66` I/O (index load / server) · `78` config (no tokens).
//! Never-silent (G2): every failure is an explicit stderr message, never a panic.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use mycelium_tero::{load_report, serve_http, AppState, TokenTable};

const EX_OK: u8 = 0;
const EX_USAGE: u8 = 64;
const EX_IO: u8 = 66;
const EX_CONFIG: u8 = 78;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args).await {
        Ok(code) => ExitCode::from(code),
        Err((code, msg)) => {
            eprintln!("tero-http: {msg}");
            ExitCode::from(code)
        }
    }
}

async fn run(args: &[String]) -> Result<u8, (u8, String)> {
    let mut index = PathBuf::from("docs/tero-index/index.json");
    let mut addr_str = String::from("127.0.0.1:8787");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => {
                i += 1;
                index = PathBuf::from(args.get(i).ok_or((EX_USAGE, usage()))?);
            }
            "--addr" => {
                i += 1;
                addr_str = args.get(i).ok_or((EX_USAGE, usage()))?.clone();
            }
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(EX_OK);
            }
            other => return Err((EX_USAGE, format!("unknown argument: {other}\n{}", usage()))),
        }
        i += 1;
    }

    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| (EX_USAGE, format!("bad --addr {addr_str:?}: {e}")))?;

    // Never-silent: refuse to start without tokens (no anonymous default).
    let tokens = TokenTable::from_env().map_err(|e| (EX_CONFIG, e.to_string()))?;
    let ntok = tokens.len();

    let report =
        load_report(&index).map_err(|e| (EX_IO, format!("loading {}: {e}", index.display())))?;
    let nrows = report.items.len();

    let state = Arc::new(AppState::new(report, tokens, false, index));
    eprintln!(
        ">> tero-http: serving {nrows} rows to {ntok} token(s) on http://{addr}/v1 \
         (token-scoped, read-only by default; Layer-1 only — Layer-2 gate closed)"
    );
    serve_http(addr, state)
        .await
        .map_err(|e| (EX_IO, format!("server: {e}")))?;
    Ok(EX_OK)
}

fn usage() -> String {
    "usage: TERO_TOKENS='<token>:<read|refresh> ...' tero-http [--index <index.json>] \
     [--addr <ip:port>]"
        .to_owned()
}
