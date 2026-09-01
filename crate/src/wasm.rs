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
        let file = GgufFile::from_bytes(gguf)
            .map_err(|e| JsError::new(&format!("fail-closed: {e}")))?;
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
