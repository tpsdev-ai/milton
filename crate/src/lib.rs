//! Milton crate — tokenizer + GGUF dequant for nomic-embed-text-v1.5.
//! Forward / mean-pool / L2 and WASM packaging are later slices.

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
        let file = Self::load(path)?;
        file.nomic_v15_meta()?;
        Ok(file)
    }

    /// Architecture / dims / pooling / layer-norm as read from this file.
    pub fn nomic_v15_meta(&self) -> Result<ModelMeta> {
        ModelMeta::from_gguf(self)
    }
}

/// Cosine distance `1 - cos(a, b)`. Length mismatch → 1.0 (fail closed).
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (1.0 - dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Compare one dequantized row to a golden. Same formula as `harness/lib/compare.mjs`.
pub fn compare_vectors(
    got: &[f32],
    expected: &[f32],
    epsilon: f64,
    epsilon_abs: f64,
) -> CompareResult {
    if got.len() != expected.len() {
        return CompareResult {
            ok: false,
            max_abs: f64::INFINITY,
            mean_abs: f64::INFINITY,
            cosine_distance: 1.0,
            mismatch_reason: Some(format!(
                "length mismatch: got {} expected {}",
                got.len(),
                expected.len()
            )),
        };
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for i in 0..got.len() {
        let d = (got[i] as f64 - expected[i] as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d;
    }
    let mean_abs = if got.is_empty() {
        0.0
    } else {
        sum_abs / got.len() as f64
    };
    let cosine_distance = cosine_distance(got, expected) as f64;
    let ok = max_abs <= epsilon_abs && cosine_distance <= epsilon;
    CompareResult {
        ok,
        max_abs,
        mean_abs,
        cosine_distance,
        mismatch_reason: if ok {
            None
        } else {
            Some(format!(
                "max_abs={max_abs} (eps_abs={epsilon_abs}) cosine_distance={cosine_distance} (eps={epsilon})"
            ))
        },
    }
}

#[derive(Debug, Clone)]
pub struct CompareResult {
    pub ok: bool,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub cosine_distance: f64,
    pub mismatch_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_milton() {
        assert_eq!(env!("CARGO_PKG_NAME"), "milton");
    }
}
