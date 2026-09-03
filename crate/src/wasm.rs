//! WASM-SIMD exports. Same `Model` as the native bins — not a second forward.
//!
//! Built with `+simd128` (see `crate/.cargo/config.toml`). Consumers never
//! compile this; they load the prebuilt `wasm/milton_bg.wasm`.

use wasm_bindgen::prelude::*;

use crate::gguf::GgufFile;
use crate::model::{EmbedConfig, ForwardFault, Model};
use crate::Prefix;

fn fault_from(s: &str) -> Result<ForwardFault, JsError> {
    match s {
        "" | "none" => Ok(ForwardFault::None),
        "layernorm" | "wrong-layernorm" => Ok(ForwardFault::WrongLayernorm),
        "pooling" | "wrong-pooling" => Ok(ForwardFault::WrongPooling),
        "dropped-prefix" | "drop-prefix" => Ok(ForwardFault::DroppedPrefix),
        other => Err(JsError::new(&format!(
            "fail-closed: unknown fault {other:?}"
        ))),
    }
}

/// In-process embedder loaded from GGUF bytes.
#[wasm_bindgen]
pub struct Milton {
    model: Model,
}

#[wasm_bindgen]
impl Milton {
    /// `gguf` is the raw nomic-embed-text-v1.5 GGUF (architecture from the file).
    #[wasm_bindgen(constructor)]
    pub fn new(gguf: Vec<u8>) -> Result<Milton, JsError> {
        if gguf.is_empty() {
            return Err(JsError::new("fail-closed: GGUF bytes are empty"));
        }
        let file =
            GgufFile::from_bytes(gguf).map_err(|e| JsError::new(&format!("fail-closed: {e}")))?;
        let model = Model::from_gguf(&file, EmbedConfig::default())
            .map_err(|e| JsError::new(&format!("fail-closed: {e}")))?;
        Ok(Milton { model })
    }

    /// `embed(text, prefix) -> Float32Array`. Prefix kind is `document` |
    /// `query` | `none`. Templates (`search_document: ` / `search_query: `)
    /// are config on the Rust side — space after the colon is load-bearing.
    #[wasm_bindgen]
    pub fn embed(&self, text: &str, prefix: &str) -> Result<Vec<f32>, JsError> {
        self.embed_with_fault(text, prefix, "none")
    }

    #[wasm_bindgen(js_name = embedWithFault)]
    pub fn embed_with_fault(
        &self,
        text: &str,
        prefix: &str,
        fault: &str,
    ) -> Result<Vec<f32>, JsError> {
        let f = fault_from(fault)?;
        let p = Prefix::parse(prefix).map_err(|e| JsError::new(&format!("fail-closed: {e}")))?;
        self.model
            .embed_with_fault(text, p, f)
            .map_err(|e| JsError::new(&format!("fail-closed: {e}")))
    }

    #[wasm_bindgen(js_name = embeddingLength)]
    pub fn embedding_length(&self) -> u32 {
        self.model.meta.embedding_length as u32
    }
}

/// Harness / JS glue: force a Q4_K inner-loop variant (`perk` | `bprime` | `auto`).
/// `allk` is not shipped — JS warns and falls through. Does not live in the
/// wasm as an env-var string — JS reads `MILTON_Q4K_VARIANT`.
#[wasm_bindgen(js_name = q4kSetForce)]
pub fn q4k_set_force(name: &str) {
    crate::qmatmul_simd128::q4k_set_force(name);
}

#[wasm_bindgen(js_name = q4kSetThreshold)]
pub fn q4k_set_threshold(t: u32) {
    crate::qmatmul_simd128::q4k_set_threshold(t);
}

#[wasm_bindgen(js_name = q4kThreshold)]
pub fn q4k_threshold() -> u32 {
    crate::qmatmul_simd128::q4k_threshold()
}

/// One synthetic superblock × `n_tokens` of the shipped per-k tile.
#[wasm_bindgen(js_name = q4kRunPerk)]
pub fn q4k_run_perk(n_tokens: u32) {
    crate::qmatmul_simd128::q4k_run_perk(n_tokens);
}

/// One synthetic superblock × `n_tokens` of the (b′) lane-wise tile.
#[wasm_bindgen(js_name = q4kRunBprime)]
pub fn q4k_run_bprime(n_tokens: u32) {
    crate::qmatmul_simd128::q4k_run_bprime(n_tokens);
}

#[cfg(feature = "wasm-threads")]
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Threaded artifact only. JS starts `W` `worker_threads` then calls this.
/// `W=1` keeps the serial `matmul_ggml` path inside the shared-memory module.
#[cfg(feature = "wasm-threads")]
#[wasm_bindgen(js_name = miltonSetWorkers)]
pub fn milton_set_workers(n: u32) {
    crate::wasm_pool::set_workers(n);
}

#[cfg(feature = "wasm-threads")]
#[wasm_bindgen(js_name = miltonWorkerCount)]
pub fn milton_worker_count() -> u32 {
    crate::wasm_pool::worker_count()
}

/// Worker entry: never returns. Parks on the shared epoch.
#[cfg(feature = "wasm-threads")]
#[wasm_bindgen(js_name = miltonWorkerEnter)]
pub fn milton_worker_enter(id: u32) {
    crate::wasm_pool::worker_enter(id);
}

#[cfg(feature = "profile")]
#[wasm_bindgen]
impl Milton {
    /// Harness-only. Default wasm:build does not compile this export.
    #[wasm_bindgen(js_name = embedProfiled)]
    pub fn embed_profiled(&self, text: &str, prefix: &str) -> Result<String, JsError> {
        let p = Prefix::parse(prefix).map_err(|e| JsError::new(&format!("fail-closed: {e}")))?;
        let (v, snap) = self
            .model
            .embed_profiled(text, p)
            .map_err(|e| JsError::new(&format!("fail-closed: {e}")))?;
        serde_json::to_string(&serde_json::json!({
            "vector": v,
            "profile": snap,
        }))
        .map_err(|e| JsError::new(&format!("fail-closed: {e}")))
    }
}
