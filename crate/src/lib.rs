//! Milton crate — tokenizer + GGUF dequant for nomic-embed-text-v1.5.
//! Forward / mean-pool / L2 and WASM packaging are later slices.
//!
//! Tokenizer core (`tokenize`, `apply_prefix`) is pure. Vocab is embedded
//! from the pinned `vocab.txt`. Dequant correctness is defined against
//! llama.cpp of the pinned GGUF (`harness/goldens/pin.json`). camelid
//! patterns are mirrored, not vendored. An unverified path refuses.

mod prefix;
mod tokenizer;
mod vocab;

#[cfg(test)]
mod conformance;

pub mod dequant;
pub mod error;
pub mod gguf;
pub mod meta;

pub use prefix::{apply_prefix, Prefix, PrefixError};
pub use tokenizer::{tokenize, tokenize_kind, tokenize_prefixed, TokenizeError};

pub use dequant::{dequantize, dequantize_wrong_block_scale, dequantize_wrong_type, f16_to_f32};
pub use error::{Error, Result};
pub use gguf::{GgufFile, MetadataValue, TensorInfo, TensorType};
pub use meta::ModelMeta;

impl GgufFile {
    /// Load the pinned nomic-embed-text-v1.5 GGUF and refuse unknown architectures.
    pub fn load_nomic_v15(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = Self::open(path)?;
        file.nomic_v15_meta()?;
        Ok(file)
    }

    /// Architecture / dims / pooling / layer-norm as read from this file.
    pub fn nomic_v15_meta(&self) -> Result<ModelMeta> {
        ModelMeta::from_gguf(self)
    }

    /// Dequantize one named tensor to a flat f32 vector. Refuses unknown types.
    pub fn dequantize_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        let bytes = self.tensor_bytes(info)?;
        dequantize(info.tensor_type, bytes, info.n_elements() as usize, name)
    }

    pub fn model_meta(&self) -> Result<ModelMeta> {
        ModelMeta::from_gguf(self)
    }
}

/// Cosine similarity. Identical to the harness gate.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let x = f64::from(a[i]);
        let y = f64::from(b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return Some(if na == 0.0 && nb == 0.0 { 1.0 } else { 0.0 });
    }
    Some((dot / denom) as f32)
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut max = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max {
            max = d;
        }
    }
    Some(max)
}

#[derive(Debug, Clone)]
pub struct Compare {
    pub pass: bool,
    pub reason: Option<String>,
    pub cosine: Option<f32>,
    pub cos_dist: Option<f32>,
    pub max_abs: Option<f32>,
}

pub fn compare_vectors(got: &[f32], expected: &[f32], epsilon: f32, epsilon_abs: f32) -> Compare {
    if got.len() != expected.len() {
        return Compare {
            pass: false,
            reason: Some(format!("dim_mismatch:{}->{}", expected.len(), got.len())),
            cosine: None,
            cos_dist: None,
            max_abs: None,
        };
    }
    let cos = cosine(got, expected).unwrap();
    let cos_dist = (1.0 - cos).max(0.0);
    let max_abs = max_abs_diff(got, expected).unwrap();
    let pass = cos >= 1.0 - epsilon && max_abs <= epsilon_abs;
    let reason = if pass {
        None
    } else {
        let mut parts = Vec::new();
        if cos < 1.0 - epsilon {
            parts.push(format!("cos_dist={cos_dist}"));
        }
        if max_abs > epsilon_abs {
            parts.push(format!("max_abs={max_abs}"));
        }
        Some(parts.join(","))
    };
    Compare {
        pass,
        reason,
        cosine: Some(cos),
        cos_dist: Some(cos_dist),
        max_abs: Some(max_abs),
    }
}
