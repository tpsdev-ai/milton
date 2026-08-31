//! GGUF-driven embedder: tokenize → embeddings → BERT layers → pool → L2.
//!
//! Architecture (layer count, dims, pooling, RoPE, causal, LN eps) is read
//! from the GGUF. Prefix templates are config. v1 verifies nomic-embed-text-v1.5
//! only. Unverified pooling / activation / arch refuse.

use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::meta::{
    ModelMeta, LLAMA_POOLING_CLS, LLAMA_POOLING_MEAN,
};
use crate::ops::{
    attention, cls_pool, l2_normalize_inplace, layer_norm, matmul, mean_pool, rope_neox_inplace,
    silu,
};
use crate::prefix::{Prefix, PrefixConfig};
use crate::weights::Weights;

/// Embed-time knobs that are **not** GGUF facts.
///
/// L2 (`embd_normalize = 2`) is absent from the nomic GGUF; the harness pin
/// still applies `--embd-normalize 2`. Prefix templates are config.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    pub prefix: PrefixConfig,
    /// llama.cpp `--embd-normalize`. 2 = Euclidean / L2. -1 = none.
    pub embd_normalize: i32,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            prefix: PrefixConfig::default(),
            embd_normalize: 2,
        }
    }
}

/// Deliberately-wrong forwards for the must-fail control. Not a fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardFault {
    None,
    WrongLayernorm,
    WrongPooling,
    DroppedPrefix,
}

pub struct Model {
    pub meta: ModelMeta,
    pub weights: Weights,
    pub config: EmbedConfig,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    n_layer: usize,
    head_dim: usize,
    n_rot: usize,
    ln_eps: f32,
    rope_freq_base: f32,
    causal: bool,
}

impl Model {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::load_with_config(path, EmbedConfig::default())
    }

    pub fn load_with_config(path: impl AsRef<std::path::Path>, config: EmbedConfig) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf, config)
    }

    pub fn from_gguf(gguf: &GgufFile, config: EmbedConfig) -> Result<Self> {
        let meta = ModelMeta::from_gguf(gguf)?;
        let n_embd = meta.embedding_length as usize;
        let n_layer = meta.block_count as usize;
        let n_head = meta.attention_head_count.ok_or_else(|| {
            Error::InvalidGguf("required metadata attention.head_count is missing".into())
        })? as usize;
        let n_ff = meta.feed_forward_length.ok_or_else(|| {
            Error::InvalidGguf("required metadata feed_forward_length is missing".into())
        })? as usize;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(Error::InvalidGguf(format!(
                "embedding_length {n_embd} is not divisible by head_count {n_head}"
            )));
        }
        let head_dim = n_embd / n_head;
        let n_rot = meta.rope_dimension_count.unwrap_or(head_dim as u64) as usize;
        if n_rot != head_dim {
            return Err(Error::UnsupportedModel(format!(
                "fail-closed: rope.dimension_count {n_rot} != head_dim {head_dim}"
            )));
        }
        let ln_eps = meta.layer_norm_epsilon.ok_or_else(|| {
            Error::InvalidGguf("required metadata attention.layer_norm_epsilon is missing".into())
        })? as f32;
        let rope_freq_base = meta.rope_freq_base.ok_or_else(|| {
            Error::InvalidGguf(
                "required metadata rope.freq_base is missing (nomic-bert uses RoPE; refuse to guess 10000)"
                    .into(),
            )
        })? as f32;
        let causal = meta.causal_attn.ok_or_else(|| {
            Error::InvalidGguf("required metadata attention.causal is missing".into())
        })?;
        if causal {
            return Err(Error::UnsupportedModel(
                "fail-closed: causal attention is unverified for v1 embeddings".into(),
            ));
        }
        match meta.pooling_type {
            Some(LLAMA_POOLING_MEAN) | Some(LLAMA_POOLING_CLS) => {}
            Some(other) => {
                return Err(Error::UnsupportedPooling(format!(
                    "pooling_type={other} (v1 implements mean and cls from the GGUF; nothing else)"
                )));
            }
            None => {
                return Err(Error::UnsupportedPooling(
                    "GGUF has no pooling_type; refuse to assume mean".into(),
                ));
            }
        }

        let weights = Weights::load(gguf, &meta)?;
        if let Some(ref types) = weights.token_types {
            if types.len() < n_embd {
                return Err(Error::InvalidGguf("token_types.weight shorter than n_embd".into()));
            }
        }

        Ok(Self {
            meta,
            weights,
            config,
            n_embd,
            n_head,
            n_ff,
            n_layer,
            head_dim,
            n_rot,
            ln_eps,
            rope_freq_base,
            causal,
        })
    }

    /// `embed(text, prefix) -> f32 vector`. Prefix templates come from config.
    pub fn embed(&self, text: &str, prefix: Prefix) -> Result<Vec<f32>> {
        self.embed_with_fault(text, prefix, ForwardFault::None)
    }

    pub fn embed_kind(&self, text: &str, kind: &str) -> Result<Vec<f32>> {
        let prefix = Prefix::parse(kind).map_err(|e| Error::Prefix(e.to_string()))?;
        self.embed(text, prefix)
    }

    pub fn embed_with_fault(&self, text: &str, prefix: Prefix, fault: ForwardFault) -> Result<Vec<f32>> {
        let prefix = match fault {
            ForwardFault::DroppedPrefix => Prefix::None,
            _ => prefix,
        };
        // Config applies the prefix; tokenize_prefixed does not apply it again.
        let ids = crate::tokenizer::tokenize_prefixed(&self.config.prefix.apply(text, prefix));
        self.embed_ids(&ids, fault)
    }

    pub fn embed_ids(&self, ids: &[u32], fault: ForwardFault) -> Result<Vec<f32>> {
        if ids.len() as u64 > self.meta.context_length {
            return Err(Error::ContextLength {
                n_tokens: ids.len(),
                context_length: self.meta.context_length,
            });
        }
        if ids.is_empty() {
            return Err(Error::InvalidTensorData("refuse to guess a vector for zero tokens".into()));
        }
        let hidden = self.forward_hidden(ids, fault)?;
        let mut pooled = vec![0.0f32; self.n_embd];
        let pooling = match fault {
            ForwardFault::WrongPooling => LLAMA_POOLING_CLS,
            _ => self.meta.pooling_type.unwrap(),
        };
        match pooling {
            LLAMA_POOLING_MEAN => mean_pool(&hidden, ids.len(), self.n_embd, &mut pooled),
            LLAMA_POOLING_CLS => cls_pool(&hidden, self.n_embd, &mut pooled),
            other => {
                return Err(Error::UnsupportedPooling(format!("pooling_type={other}")));
            }
        }
        match self.config.embd_normalize {
            -1 => {}
            2 => l2_normalize_inplace(&mut pooled),
            other => {
                return Err(Error::UnsupportedModel(format!(
                    "fail-closed: unverified embd_normalize={other} (v1 is 2 / L2 or -1 / none)"
                )));
            }
        }
        Ok(pooled)
    }

    fn forward_hidden(&self, ids: &[u32], fault: ForwardFault) -> Result<Vec<f32>> {
        let n_tok = ids.len();
        let n_embd = self.n_embd;
        let mut x = vec![0.0f32; n_tok * n_embd];
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.weights.token_embd_cols {
                return Err(Error::InvalidTensorData(format!(
                    "token id {id} >= vocab {}",
                    self.weights.token_embd_cols
                )));
            }
            let src = &self.weights.token_embd[id * n_embd..(id + 1) * n_embd];
            x[t * n_embd..(t + 1) * n_embd].copy_from_slice(src);
            if let Some(ref types) = self.weights.token_types {
                // llama.cpp hardcodes type 0 ("Sentence A")
                let row = &types[0..n_embd];
                for i in 0..n_embd {
                    x[t * n_embd + i] += row[i];
                }
            }
        }

        let ln_eps = match fault {
            ForwardFault::WrongLayernorm => 1.0,
            _ => self.ln_eps,
        };
        let mut y = vec![0.0f32; n_tok * n_embd];
        for t in 0..n_tok {
            layer_norm(
                &x[t * n_embd..(t + 1) * n_embd],
                &self.weights.token_embd_norm.weight,
                &self.weights.token_embd_norm.bias,
                ln_eps,
                &mut y[t * n_embd..(t + 1) * n_embd],
            );
        }
        x.copy_from_slice(&y);

        let mut qkv = vec![0.0f32; n_tok * 3 * n_embd];
        let mut attn_out = vec![0.0f32; n_tok * n_embd];
        let mut proj = vec![0.0f32; n_tok * n_embd];
        let mut ffn_up = vec![0.0f32; n_tok * self.n_ff];
        let mut ffn_gate = vec![0.0f32; n_tok * self.n_ff];
        let mut ffn_hid = vec![0.0f32; n_tok * self.n_ff];
        let mut ffn_down = vec![0.0f32; n_tok * n_embd];
        let mut q = vec![0.0f32; n_tok * n_embd];
        let mut k = vec![0.0f32; n_tok * n_embd];
        let mut v = vec![0.0f32; n_tok * n_embd];

        let _ = (self.n_rot, self.causal, self.n_layer);

        for layer in &self.weights.layers {
            matmul(&x, &layer.attn_qkv, n_tok, n_embd, 3 * n_embd, &mut qkv);
            if let Some(ref b) = layer.attn_qkv_bias {
                for t in 0..n_tok {
                    for i in 0..3 * n_embd {
                        qkv[t * 3 * n_embd + i] += b[i];
                    }
                }
            }
            for t in 0..n_tok {
                let src = &qkv[t * 3 * n_embd..(t + 1) * 3 * n_embd];
                q[t * n_embd..(t + 1) * n_embd].copy_from_slice(&src[0..n_embd]);
                k[t * n_embd..(t + 1) * n_embd].copy_from_slice(&src[n_embd..2 * n_embd]);
                v[t * n_embd..(t + 1) * n_embd].copy_from_slice(&src[2 * n_embd..3 * n_embd]);
            }
            rope_neox_inplace(&mut q, n_tok, self.n_head, self.head_dim, self.rope_freq_base);
            rope_neox_inplace(&mut k, n_tok, self.n_head, self.head_dim, self.rope_freq_base);
            attention(&q, &k, &v, n_tok, self.n_head, self.head_dim, &mut attn_out);
            matmul(&attn_out, &layer.attn_output, n_tok, n_embd, n_embd, &mut proj);
            if let Some(ref b) = layer.attn_output_bias {
                for t in 0..n_tok {
                    for i in 0..n_embd {
                        proj[t * n_embd + i] += b[i];
                    }
                }
            }
            for t in 0..n_tok {
                for i in 0..n_embd {
                    proj[t * n_embd + i] += x[t * n_embd + i];
                }
                layer_norm(
                    &proj[t * n_embd..(t + 1) * n_embd],
                    &layer.attn_output_norm.weight,
                    &layer.attn_output_norm.bias,
                    ln_eps,
                    &mut y[t * n_embd..(t + 1) * n_embd],
                );
            }
            x.copy_from_slice(&y);

            match &layer.ffn_gate {
                Some(gate_w) => {
                    matmul(&x, &layer.ffn_up, n_tok, n_embd, self.n_ff, &mut ffn_up);
                    matmul(&x, gate_w, n_tok, n_embd, self.n_ff, &mut ffn_gate);
                    for i in 0..ffn_hid.len() {
                        ffn_hid[i] = ffn_up[i] * silu(ffn_gate[i]);
                    }
                }
                None => {
                    return Err(Error::UnsupportedModel(
                        "fail-closed: ffn_gate missing; v1 nomic-bert is SwiGLU, refuse to guess GELU"
                            .into(),
                    ));
                }
            }
            matmul(&ffn_hid, &layer.ffn_down, n_tok, self.n_ff, n_embd, &mut ffn_down);
            for t in 0..n_tok {
                for i in 0..n_embd {
                    ffn_down[t * n_embd + i] += x[t * n_embd + i];
                }
                layer_norm(
                    &ffn_down[t * n_embd..(t + 1) * n_embd],
                    &layer.layer_output_norm.weight,
                    &layer.layer_output_norm.bias,
                    ln_eps,
                    &mut y[t * n_embd..(t + 1) * n_embd],
                );
            }
            x.copy_from_slice(&y);
        }
        Ok(x)
    }
}

/// Convenience: load + embed. Prefix is the kind (`document` | `query` | `none`).
pub fn embed(model: &Model, text: &str, prefix: Prefix) -> Result<Vec<f32>> {
    model.embed(text, prefix)
}
