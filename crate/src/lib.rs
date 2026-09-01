//! Milton crate — tokenizer + GGUF dequant + forward for nomic-embed-text-v1.5.
//! Same lib is compiled native (bins) and WASM-SIMD (`wasm/`). Not two forwards.
//!
//! Tokenizer core (`tokenize`, `apply_prefix`) is pure. Vocab is embedded
//! from the pinned `vocab.txt`. Forward is GGUF-driven (layer count, dims,
//! pooling, RoPE, LN eps). Prefixes are config. Dequant + forward correctness
//! is defined against llama.cpp of the pinned GGUF. An unverified path refuses.

mod prefix;
mod tokenizer;
mod vocab;
mod ops;
mod qmatmul;
#[cfg(target_arch = "x86_64")]
mod q4k_avx2;
#[cfg(target_arch = "wasm32")]
mod qmatmul_simd128;
mod weights;
mod model;

#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(test)]
mod conformance;

pub mod dequant;
pub mod error;
pub mod gguf;
pub mod meta;

pub use prefix::{apply_prefix, Prefix, PrefixConfig, PrefixError};
pub use tokenizer::{tokenize, tokenize_kind, tokenize_prefixed, TokenizeError};
pub use model::{embed, EmbedConfig, ForwardFault, Model};

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

/// F32-discriminator decision. Both gate implementations (this crate's
/// `embed-gate` and `harness/scripts/discriminate-f32.mjs`) must use this
/// predicate and read `ratio_max` from `quant-budget.json`.
#[derive(Debug, Clone, Copy)]
pub struct F32GateDecision {
    pub pass: bool,
    pub within_absolute: bool,
    pub within_ratio: bool,
    pub ratio: f32,
}

/// `ratio = cos_dist / quant_budget` (∞ if the budget is 0 and cos_dist > 0).
pub fn f32_case_ratio(cos_dist: f32, quant_budget_cos_dist: f32) -> f32 {
    if quant_budget_cos_dist > 0.0 {
        cos_dist / quant_budget_cos_dist
    } else if cos_dist == 0.0 {
        0.0
    } else {
        f32::INFINITY
    }
}

/// Pass only if within the loose absolute ceiling AND the per-case ratio.
pub fn f32_gate_pass(
    cos_dist: f32,
    quant_budget_cos_dist: f32,
    gate_cos_dist: f32,
    ratio_max: f32,
) -> F32GateDecision {
    let ratio = f32_case_ratio(cos_dist, quant_budget_cos_dist);
    let within_absolute = cos_dist <= gate_cos_dist;
    let within_ratio = ratio <= ratio_max;
    F32GateDecision {
        pass: within_absolute && within_ratio,
        within_absolute,
        within_ratio,
        ratio,
    }
}

#[cfg(test)]
mod f32_gate_tests {
    use super::*;

    #[test]
    fn synthetic_under_absolute_over_ratio_fails() {
        let gate = 0.30818967_f32;
        let ratio_max = 1.5_f32;
        let qb = 0.05_f32;
        let cd = 0.10_f32;
        assert!(cd < gate, "under the loose absolute (old gate would PASS)");
        assert!(cd / qb > ratio_max);
        let r = f32_gate_pass(cd, qb, gate, ratio_max);
        assert!(r.within_absolute);
        assert!(!r.within_ratio);
        assert!(!r.pass);
        assert!((r.ratio - 2.0).abs() < 1e-6);
    }

    #[test]
    fn recorded_gated_ratios_0_90_to_1_08_pass() {
        let gate = 0.30818967_f32;
        for ratio in [0.90_f32, 1.00, 1.08] {
            let qb = 0.05_f32;
            let r = f32_gate_pass(ratio * qb, qb, gate, 1.5);
            assert!(r.pass, "ratio {ratio}");
            assert!(r.within_absolute);
            assert!(r.within_ratio);
        }
    }

    #[test]
    fn tight_tier_1e6_floor_is_tighter_than_any_ratio() {
        let r = f32_gate_pass(1e-6, 0.10272989, 0.30818967, 1.5);
        assert!(r.pass);
        assert!(r.ratio < 1e-4);
    }

    #[test]
    fn zero_budget_with_distance_fails_closed() {
        let r = f32_gate_pass(1e-3, 0.0, 0.30818967, 1.5);
        assert!(!r.within_ratio);
        assert!(!r.pass);
        assert!(r.ratio.is_infinite());
    }

    #[test]
    fn rust_gate_reads_ratio_max_from_quant_budget_json() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let raw = std::fs::read_to_string(root.join("harness/goldens/quant-budget.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let ratio_max = v["ratio_max"].as_f64().expect("ratio_max") as f32;
        let gate = v["gate_cos_dist"].as_f64().expect("gate_cos_dist") as f32;
        assert!(
            (ratio_max - 1.5).abs() < 1e-6,
            "ratio_max must be pinned at 1.5, got {ratio_max}"
        );
        // Synthetic: under absolute, over 1.5× own budget → FAIL.
        let r = f32_gate_pass(0.10, 0.05, gate, ratio_max);
        assert!(r.within_absolute);
        assert!(!r.within_ratio);
        assert!(!r.pass);
    }
}
