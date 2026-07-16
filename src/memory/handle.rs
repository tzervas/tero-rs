//! Runtime handle for memory-gate-rs (`SqliteVecStore` + [`PassthroughAdapter`] gateway).

use std::path::Path;
use std::sync::OnceLock;

use memory_gate_rs::adapters::PassthroughAdapter;
use memory_gate_rs::facade::{
    consolidate_once, for_tero_learn, gateway_with_store, learn, open_prod_sqlite, retrieve,
};
use memory_gate_rs::storage::SqliteVecStore;
use memory_gate_rs::{
    AgentDomain, ConsolidationStats, GatewayConfig, LearningContext, MemoryGateway,
    SupportedEmbeddingModel,
};
use serde_json::{json, Value};
use tokio::runtime::Runtime;

/// Configuration / open failure for the optional memory layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryOpenError {
    /// `TERO_MEMORY_ENABLED` is set but `TERO_MEMORY_DB` is missing or empty.
    MissingDb,
    /// `TERO_MEMORY_MODEL` is not a supported catalog id.
    BadModel(String),
    /// Underlying open / gateway init failed.
    Open(String),
}

impl std::fmt::Display for MemoryOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryOpenError::MissingDb => write!(
                f,
                "TERO_MEMORY_ENABLED is set but TERO_MEMORY_DB is unset or empty"
            ),
            MemoryOpenError::BadModel(id) => {
                write!(f, "TERO_MEMORY_MODEL {id:?} is not a supported catalog id")
            }
            MemoryOpenError::Open(why) => write!(f, "memory-gate open failed: {why}"),
        }
    }
}

impl std::error::Error for MemoryOpenError {}

/// Shared tokio runtime for blocking MG async calls from the sync MCP stdio loop.
fn shared_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("memory feature tokio runtime"))
}

/// An opened memory-gate gateway (optional at process level).
pub struct MemoryHandle {
    gateway: MemoryGateway<PassthroughAdapter, SqliteVecStore>,
    default_tero_index: Option<String>,
}

impl MemoryHandle {
    /// Open memory when `TERO_MEMORY_ENABLED` is `1` or `true`; otherwise `Ok(None)`.
    ///
    /// When enabled, `TERO_MEMORY_DB` must name an on-disk sqlite path. `TERO_MEMORY_MODEL`
    /// selects the embedding catalog (default: `SupportedEmbeddingModel::DEFAULT`).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryOpenError`] when enabled but misconfigured or the store cannot open.
    pub fn try_open_from_env() -> Result<Option<Self>, MemoryOpenError> {
        if !memory_enabled_from_env() {
            return Ok(None);
        }
        let db = std::env::var("TERO_MEMORY_DB")
            .ok()
            .filter(|s| !s.is_empty())
            .ok_or(MemoryOpenError::MissingDb)?;
        let model = parse_model_from_env()?;
        Self::open_at(&db, model, None)
    }

    /// Open with an explicit DB path (tests / callers with index metadata).
    ///
    /// # Errors
    ///
    /// Same as [`Self::try_open_from_env`].
    pub fn open_at(
        db: impl AsRef<Path>,
        model: SupportedEmbeddingModel,
        tero_index: Option<String>,
    ) -> Result<Option<Self>, MemoryOpenError> {
        let rt = shared_runtime();
        let store = rt
            .block_on(open_prod_sqlite(db.as_ref(), model))
            .map_err(|e| MemoryOpenError::Open(e.to_string()))?;
        let gateway = gateway_with_store(store, GatewayConfig::default());
        Ok(Some(MemoryHandle {
            gateway,
            default_tero_index: tero_index,
        }))
    }

    /// Attach the L1 index path used for `tero_index` metadata on learns.
    #[must_use]
    pub fn with_tero_index(mut self, index_path: impl Into<String>) -> Self {
        self.default_tero_index = Some(index_path.into());
        self
    }

    /// Persist an experience via memory-gate `learn`.
    ///
    /// # Errors
    ///
    /// Propagates memory-gate errors as strings (surfaced by the MCP front).
    pub fn store(
        &self,
        content: &str,
        anchors: Option<&str>,
        importance: Option<f32>,
        tero_index: Option<&str>,
    ) -> Result<String, String> {
        let anchor_str = anchors.unwrap_or("");
        let index = tero_index
            .map(str::to_owned)
            .or_else(|| self.default_tero_index.clone());
        let ctx = for_tero_learn(content, anchor_str, importance, index);
        shared_runtime()
            .block_on(learn(&self.gateway, ctx, None))
            .map_err(|e| e.to_string())
    }

    /// Ranked retrieval over the `Tero` domain.
    ///
    /// # Errors
    ///
    /// Propagates memory-gate errors as strings.
    pub fn retrieve(&self, query: &str, k: Option<usize>) -> Result<Vec<LearningContext>, String> {
        shared_runtime()
            .block_on(retrieve(&self.gateway, query, k, Some(AgentDomain::Tero)))
            .map_err(|e| e.to_string())
    }

    /// Run consolidation once.
    ///
    /// # Errors
    ///
    /// Propagates memory-gate errors as strings.
    pub fn consolidate_once(&self) -> Result<ConsolidationStats, String> {
        shared_runtime()
            .block_on(consolidate_once(&self.gateway))
            .map_err(|e| e.to_string())
    }
}

/// JSON envelope for a successful store (`kind: memory_stored`).
#[must_use]
pub fn envelope_stored(id: &str) -> Value {
    json!({ "kind": "memory_stored", "id": id })
}

/// JSON envelope for retrieve hits (`kind: memory_hits` — not L1 citations).
#[must_use]
pub fn envelope_hits(contexts: &[LearningContext]) -> Value {
    let items: Vec<Value> = contexts
        .iter()
        .map(|ctx| {
            json!({
                "content": ctx.content,
                "domain": ctx.domain.as_str(),
                "importance": ctx.importance,
                "metadata": ctx.metadata,
            })
        })
        .collect();
    json!({ "kind": "memory_hits", "items": items })
}

/// JSON envelope for consolidation (`kind: memory_consolidated`).
#[must_use]
pub fn envelope_consolidated(stats: &ConsolidationStats) -> Value {
    json!({
        "kind": "memory_consolidated",
        "stats": {
            "items_processed": stats.items_processed,
            "items_deleted": stats.items_deleted,
            "duration_secs": stats.duration.as_secs_f64(),
            "errors": stats.errors,
        }
    })
}

/// Typed refusal when memory tools are compiled in but not enabled at runtime.
#[must_use]
pub fn envelope_memory_disabled() -> Value {
    json!({
        "kind": "refusal",
        "refusal": { "variant": "memory_disabled" },
        "message": "memory tools require TERO_MEMORY_ENABLED=1 (or true) and TERO_MEMORY_DB"
    })
}

fn memory_enabled_from_env() -> bool {
    matches!(
        std::env::var("TERO_MEMORY_ENABLED")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn parse_model_from_env() -> Result<SupportedEmbeddingModel, MemoryOpenError> {
    match std::env::var("TERO_MEMORY_MODEL") {
        Ok(id) if !id.trim().is_empty() => {
            SupportedEmbeddingModel::parse(id.trim()).map_err(|_| MemoryOpenError::BadModel(id))
        }
        _ => Ok(SupportedEmbeddingModel::DEFAULT),
    }
}
