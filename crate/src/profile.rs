//! Feature-gated stage timers + counters (issue #28).
//!
//! Compiled only with `--features profile`. Default `wasm:build` / shipped
//! `wasm/milton_bg.wasm` do not include this module. The instrumented
//! forward calls the same kernels as `Model::embed`; it does not rewrite
//! them. Numeric output is checked bit-exact against `embed` in the
//! feature-gated test.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::exp::vmix_axpy;
use crate::meta::{LLAMA_POOLING_CLS, LLAMA_POOLING_MEAN};
use super::{forward_variant, mean_pool_skip, split_qkv, ForwardVariant, Model};
use crate::ops::{
    cls_pool, l2_normalize_inplace, layer_norm, mean_pool, rope_neox_inplace,
    rope_norm_inplace, softmax_inplace, swiglu,
};
use crate::prefix::Prefix;
use crate::qmatmul::{matmul_ggml, QuantMat};

#[derive(Clone, Debug, Default, Serialize)]
pub struct Counters {
    pub n_tokens: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub head_dim: usize,
    /// One GEMV per token per `matmul_ggml` (the live loop in `qmatmul.rs`).
    pub gemv_calls: u64,
    /// `n_tokens * weight_bytes` per matmul — H1: each token re-reads weights.
    pub gemv_weight_bytes: u64,
    pub q8k_rows: u64,
    /// Equivalent f32 MAC FLOPs: `2 * n_tok * n_in * n_out` per matmul.
    pub matmul_flops: u64,
    pub attn_qk_dots: u64,
    pub attn_qk_flops: u64,
    pub attn_softmax_rows: u64,
    pub attn_vmix_axpy: u64,
    pub attn_vmix_flops: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub n_tokens: usize,
    pub stages_ms: BTreeMap<String, f64>,
    pub per_layer: Vec<BTreeMap<String, f64>>,
    pub counters: Counters,
    pub total_ms: f64,
}

struct Acc {
    stages: BTreeMap<String, f64>,
    layer: Vec<BTreeMap<String, f64>>,
    counters: Counters,
}

impl Acc {
    fn new(n_layer: usize) -> Self {
        Self {
            stages: BTreeMap::new(),
            layer: vec![BTreeMap::new(); n_layer],
            counters: Counters::default(),
        }
    }

    fn add(&mut self, name: &str, ms: f64) {
        *self.stages.entry(name.to_string()).or_insert(0.0) += ms;
    }

    fn add_layer(&mut self, li: usize, name: &str, ms: f64) {
        self.add(name, ms);
        if li < self.layer.len() {
            *self.layer[li].entry(name.to_string()).or_insert(0.0) += ms;
        }
    }

    fn finish(self, n_tokens: usize, total_ms: f64) -> Snapshot {
        Snapshot {
            n_tokens,
            stages_ms: self.stages,
            per_layer: self.layer,
            counters: self.counters,
            total_ms,
        }
    }
}

fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        now_js()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = performance, js_name = now)]
    fn now_js() -> f64;
}

fn timed_matmul(
    acc: &mut Acc,
    stage: &str,
    layer: Option<usize>,
    x: &[f32],
    w: &QuantMat,
    n_tokens: usize,
    y: &mut [f32],
) {
    acc.counters.gemv_calls += n_tokens as u64;
    acc.counters.gemv_weight_bytes += (n_tokens as u64) * (w.bytes.len() as u64);
    acc.counters.q8k_rows += n_tokens as u64;
    acc.counters.matmul_flops +=
        2u64 * n_tokens as u64 * w.n_in as u64 * w.n_out as u64;
    let t0 = now_ms();
    matmul_ggml(x, w, n_tokens, y);
    let dt = now_ms() - t0;
    match layer {
        Some(li) => acc.add_layer(li, stage, dt),
        None => acc.add(stage, dt),
    }
}

/// Same loops as `attention_named` (no dump). Split timers for H2.
fn timed_attention(
    acc: &mut Acc,
    li: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; n_tokens];
    acc.counters.attn_qk_dots += (n_heads * n_tokens * n_tokens) as u64;
    acc.counters.attn_qk_flops +=
        2u64 * n_heads as u64 * n_tokens as u64 * n_tokens as u64 * head_dim as u64;
    acc.counters.attn_softmax_rows += (n_heads * n_tokens) as u64;
    acc.counters.attn_vmix_axpy += (n_heads * n_tokens * n_tokens) as u64;
    acc.counters.attn_vmix_flops +=
        2u64 * n_heads as u64 * n_tokens as u64 * n_tokens as u64 * head_dim as u64;
    let mut qk_ms = 0.0;
    let mut sm_ms = 0.0;
    let mut v_ms = 0.0;
    for h in 0..n_heads {
        for tq in 0..n_tokens {
            let qoff = (tq * n_heads + h) * head_dim;
            let qh = &q[qoff..qoff + head_dim];
            let t_qk = now_ms();
            for tk in 0..n_tokens {
                let koff = (tk * n_heads + h) * head_dim;
                let kh = &k[koff..koff + head_dim];
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += qh[i] * kh[i];
                }
                scores[tk] = dot * scale;
            }
            qk_ms += now_ms() - t_qk;
            let t_sm = now_ms();
            softmax_inplace(&mut scores);
            sm_ms += now_ms() - t_sm;
            let ooff = tq * n_heads * head_dim + h * head_dim;
            let t_v = now_ms();
            for i in 0..head_dim {
                out[ooff + i] = 0.0;
            }
            for tk in 0..n_tokens {
                let a = scores[tk];
                let voff = (tk * n_heads + h) * head_dim;
                vmix_axpy(&mut out[ooff..ooff + head_dim], a, &v[voff..voff + head_dim]);
            }
            v_ms += now_ms() - t_v;
        }
    }
    acc.add_layer(li, "attn_qk", qk_ms);
    acc.add_layer(li, "attn_softmax", sm_ms);
    acc.add_layer(li, "attn_vmix", v_ms);
}

impl Model {
    /// Instrumented embed. Same kernels and graph as `embed`; timers only.
    pub fn embed_profiled(&self, text: &str, prefix: Prefix) -> Result<(Vec<f32>, Snapshot)> {
        let t_all = now_ms();
        let mut acc = Acc::new(self.n_layer);
        acc.counters.n_layer = self.n_layer;
        acc.counters.n_embd = self.n_embd;
        acc.counters.n_head = self.n_head;
        acc.counters.n_ff = self.n_ff;
        acc.counters.head_dim = self.head_dim;

        let t0 = now_ms();
        let prefixed = self.config.prefix.apply(text, prefix);
        let ids = crate::tokenizer::tokenize_prefixed(&prefixed);
        acc.add("tokenize", now_ms() - t0);
        if ids.len() as u64 > self.meta.context_length {
            return Err(Error::ContextLength {
                n_tokens: ids.len(),
                context_length: self.meta.context_length,
            });
        }
        if ids.is_empty() {
            return Err(Error::InvalidTensorData(
                "refuse to guess a vector for zero tokens".into(),
            ));
        }
        acc.counters.n_tokens = ids.len();

        let hidden = self.forward_hidden_profiled(&ids, &mut acc)?;
        let n_embd = self.n_embd;
        let mut pooled = vec![0.0f32; n_embd];
        let t0 = now_ms();
        let pooling = self.meta.pooling_type.unwrap();
        match pooling {
            LLAMA_POOLING_MEAN => match forward_variant() {
                ForwardVariant::PoolNoCls => {
                    mean_pool_skip(&hidden, ids.len(), n_embd, true, false, &mut pooled)
                }
                ForwardVariant::PoolNoSep => {
                    mean_pool_skip(&hidden, ids.len(), n_embd, false, true, &mut pooled)
                }
                ForwardVariant::PoolNoSpecial => {
                    mean_pool_skip(&hidden, ids.len(), n_embd, true, true, &mut pooled)
                }
                _ => mean_pool(&hidden, ids.len(), n_embd, &mut pooled),
            },
            LLAMA_POOLING_CLS => cls_pool(&hidden, n_embd, &mut pooled),
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
        acc.add("pooling", now_ms() - t0);
        let total_ms = now_ms() - t_all;
        Ok((pooled, acc.finish(ids.len(), total_ms)))
    }

    fn forward_hidden_profiled(&self, ids: &[u32], acc: &mut Acc) -> Result<Vec<f32>> {
        let n_tok = ids.len();
        let n_embd = self.n_embd;
        let mut x = vec![0.0f32; n_tok * n_embd];
        let t0 = now_ms();
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
                if !matches!(forward_variant(), ForwardVariant::NoType) {
                    let row = &types[0..n_embd];
                    for i in 0..n_embd {
                        x[t * n_embd + i] += row[i];
                    }
                }
            }
        }
        acc.add("embedding_lookup", now_ms() - t0);

        let ln_eps = self.ln_eps;
        let mut y = vec![0.0f32; n_tok * n_embd];
        let t0 = now_ms();
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
        acc.add("layernorm", now_ms() - t0);

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
        let variant = forward_variant();

        for (li, layer) in self.weights.layers.iter().enumerate() {
            timed_matmul(acc, "qkv", Some(li), &x, &layer.attn_qkv, n_tok, &mut qkv);
            if let Some(ref b) = layer.attn_qkv_bias {
                let t0 = now_ms();
                for t in 0..n_tok {
                    for i in 0..3 * n_embd {
                        qkv[t * 3 * n_embd + i] += b[i];
                    }
                }
                acc.add_layer(li, "qkv", now_ms() - t0);
            }
            let t0 = now_ms();
            split_qkv(
                &qkv,
                &mut q,
                &mut k,
                &mut v,
                n_tok,
                n_embd,
                self.n_head,
                self.head_dim,
                variant,
            );
            match variant {
                ForwardVariant::NoRope => {}
                ForwardVariant::RopeNorm => {
                    rope_norm_inplace(&mut q, n_tok, self.n_head, self.head_dim, self.rope_freq_base);
                    rope_norm_inplace(&mut k, n_tok, self.n_head, self.head_dim, self.rope_freq_base);
                }
                _ => {
                    rope_neox_inplace(&mut q, n_tok, self.n_head, self.head_dim, self.rope_freq_base);
                    rope_neox_inplace(&mut k, n_tok, self.n_head, self.head_dim, self.rope_freq_base);
                }
            }
            acc.add_layer(li, "rope", now_ms() - t0);

            timed_attention(
                acc,
                li,
                &q,
                &k,
                &v,
                n_tok,
                self.n_head,
                self.head_dim,
                &mut attn_out,
            );

            timed_matmul(
                acc,
                "out_proj",
                Some(li),
                &attn_out,
                &layer.attn_output,
                n_tok,
                &mut proj,
            );
            if let Some(ref b) = layer.attn_output_bias {
                let t0 = now_ms();
                for t in 0..n_tok {
                    for i in 0..n_embd {
                        proj[t * n_embd + i] += b[i];
                    }
                }
                acc.add_layer(li, "out_proj", now_ms() - t0);
            }
            let t0 = now_ms();
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
            acc.add_layer(li, "layernorm", now_ms() - t0);

            match &layer.ffn_gate {
                Some(gate_w) => {
                    timed_matmul(acc, "ffn_up", Some(li), &x, &layer.ffn_up, n_tok, &mut ffn_up);
                    timed_matmul(acc, "ffn_gate", Some(li), &x, gate_w, n_tok, &mut ffn_gate);
                    let t0 = now_ms();
                    let swap = matches!(variant, ForwardVariant::SwapSwiglu);
                    if swap {
                        swiglu(&ffn_up, &ffn_gate, &mut ffn_hid);
                    } else {
                        swiglu(&ffn_gate, &ffn_up, &mut ffn_hid);
                    }
                    acc.add_layer(li, "ffn_swiglu", now_ms() - t0);
                }
                None => {
                    return Err(Error::UnsupportedModel(
                        "fail-closed: ffn_gate missing; v1 nomic-bert is SwiGLU, refuse to guess GELU"
                            .into(),
                    ));
                }
            }
            timed_matmul(
                acc,
                "ffn_down",
                Some(li),
                &ffn_hid,
                &layer.ffn_down,
                n_tok,
                &mut ffn_down,
            );
            let t0 = now_ms();
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
            acc.add_layer(li, "layernorm", now_ms() - t0);
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Prefix;

    fn gguf_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf")
    }

    #[test]
    fn profiled_embed_matches_default_bit_exact() {
        let path = gguf_path();
        if !path.exists() {
            return;
        }
        let model = Model::load(&path).unwrap();
        for (text, kind) in [("hello", Prefix::None), ("hello", Prefix::Document)] {
            let a = model.embed(text, kind).unwrap();
            let (b, snap) = model.embed_profiled(text, kind).unwrap();
            assert_eq!(a, b, "profiled forward drifted from embed ({kind:?})");
            assert!(snap.n_tokens > 0);
            assert!(snap.counters.gemv_calls > 0);
            assert!(snap.total_ms >= 0.0);
        }
    }
}
