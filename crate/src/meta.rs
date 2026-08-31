//! Model metadata as read from the GGUF. Confirm against the file; do not assume.
//!
//! Pooling and normalization are recorded *exactly* as the KV store states them.
//! llama.cpp `--pooling mean` / `--embd-normalize 2` are runtime flags (see
//! harness pin); they are not invented here if the file does not carry them.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::gguf::{GgufFile, MetadataValue};

/// llama.cpp `llama_pooling_type` (include/llama.h).
pub const LLAMA_POOLING_UNSPECIFIED: i64 = -1;
pub const LLAMA_POOLING_NONE: i64 = 0;
pub const LLAMA_POOLING_MEAN: i64 = 1;
pub const LLAMA_POOLING_CLS: i64 = 2;
pub const LLAMA_POOLING_LAST: i64 = 3;
pub const LLAMA_POOLING_RANK: i64 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelMeta {
    pub architecture: String,
    pub name: Option<String>,
    pub block_count: u64,
    pub embedding_length: u64,
    pub context_length: u64,
    pub feed_forward_length: Option<u64>,
    pub attention_head_count: Option<u64>,
    pub layer_norm_epsilon: Option<f64>,
    /// Raw `*.pooling_type` integer as stored in the GGUF, if present.
    pub pooling_type: Option<i64>,
    /// Human label derived from `pooling_type` only (1 → "mean"). Never assumed.
    pub pooling: Option<String>,
    /// Raw key that supplied pooling_type.
    pub pooling_key: Option<String>,
    /// Every metadata key whose name mentions norm/normalize, recorded as-read.
    pub normalization: BTreeMap<String, String>,
    /// True iff no metadata key mentioned norm/normalize. The harness still
    /// applies `--embd-normalize 2` at *embed* time; that is not a GGUF fact.
    pub normalization_absent_from_gguf: bool,
}

impl ModelMeta {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| Error::UnsupportedModel("missing general.architecture".into()))?
            .to_string();

        // v1 is nomic-embed-text-v1.5 only. nomic GGUFs use architecture
        // "nomic-bert" (confirm against the file).
        if architecture != "nomic-bert" {
            return Err(Error::UnsupportedModel(format!(
                "v1 supports nomic-embed-text-v1.5 only (architecture nomic-bert); got {architecture}"
            )));
        }

        let name = gguf.metadata_string("general.name").map(str::to_string);
        let prefix = architecture.as_str();

        let block_count = required_u64(gguf, &format!("{prefix}.block_count"))?;
        let embedding_length = required_u64(gguf, &format!("{prefix}.embedding_length"))?;
        let context_length = required_u64(gguf, &format!("{prefix}.context_length"))?;
        let feed_forward_length = gguf.metadata_i64(&format!("{prefix}.feed_forward_length")).map(|v| v as u64);
        let attention_head_count = gguf
            .metadata_i64(&format!("{prefix}.attention.head_count"))
            .map(|v| v as u64);
        let layer_norm_epsilon = gguf.metadata_f64(&format!("{prefix}.attention.layer_norm_epsilon"))
            .or_else(|| gguf.metadata_f64(&format!("{prefix}.layer_norm_epsilon")));

        let pooling_key = format!("{prefix}.pooling_type");
        let pooling_type = gguf.metadata_i64(&pooling_key);
        let pooling = pooling_type.map(pooling_label);
        let pooling_key = pooling_type.map(|_| pooling_key);

        let mut normalization = BTreeMap::new();
        for (k, v) in &gguf.metadata {
            let kl = k.to_ascii_lowercase();
            if kl.contains("norm") || kl.contains("normaliz") {
                normalization.insert(k.clone(), v.as_display());
            }
        }
        let normalization_absent_from_gguf = normalization.is_empty();

        Ok(Self {
            architecture,
            name,
            block_count,
            embedding_length,
            context_length,
            feed_forward_length,
            attention_head_count,
            layer_norm_epsilon,
            pooling_type,
            pooling,
            pooling_key,
            normalization,
            normalization_absent_from_gguf,
        })
    }
}

fn required_u64(gguf: &GgufFile, key: &str) -> Result<u64> {
    gguf.metadata_i64(key)
        .map(|v| v as u64)
        .ok_or_else(|| Error::InvalidGguf(format!("required metadata {key} is missing")))
}

fn pooling_label(ty: i64) -> String {
    match ty {
        LLAMA_POOLING_UNSPECIFIED => "unspecified".into(),
        LLAMA_POOLING_NONE => "none".into(),
        LLAMA_POOLING_MEAN => "mean".into(),
        LLAMA_POOLING_CLS => "cls".into(),
        LLAMA_POOLING_LAST => "last".into(),
        LLAMA_POOLING_RANK => "rank".into(),
        other => format!("unknown({other})"),
    }
}

/// Flatten metadata to JSON-friendly strings (as-read, no interpretation).
pub fn metadata_as_strings(gguf: &GgufFile) -> BTreeMap<String, String> {
    gguf.metadata
        .iter()
        .map(|(k, v)| (k.clone(), metadata_value_string(v)))
        .collect()
}

fn metadata_value_string(v: &MetadataValue) -> String {
    match v {
        MetadataValue::Array(items) if items.len() > 32 => {
            format!("array(len={})", items.len())
        }
        other => other.as_display(),
    }
}
