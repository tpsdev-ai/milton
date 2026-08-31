//! Dequantized weight tables, named as stored in the GGUF.
//!
//! Tensor names follow the file (`token_embd`, `blk.%d.attn_qkv`, …).
//! Layer count / dims come from `ModelMeta`. Missing required tensors refuse.

use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::meta::ModelMeta;

#[derive(Debug, Clone)]
pub struct LayerNormW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_qkv: Vec<f32>,
    pub attn_qkv_bias: Option<Vec<f32>>,
    pub attn_output: Vec<f32>,
    pub attn_output_bias: Option<Vec<f32>>,
    pub attn_output_norm: LayerNormW,
    pub ffn_up: Vec<f32>,
    pub ffn_gate: Option<Vec<f32>>,
    pub ffn_down: Vec<f32>,
    pub layer_output_norm: LayerNormW,
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub token_embd: Vec<f32>,
    pub token_embd_cols: usize,
    pub token_types: Option<Vec<f32>>,
    pub token_types_cols: usize,
    pub token_embd_norm: LayerNormW,
    pub layers: Vec<LayerWeights>,
}

impl Weights {
    pub fn load(gguf: &GgufFile, meta: &ModelMeta) -> Result<Self> {
        let n_embd = meta.embedding_length as usize;
        let n_layer = meta.block_count as usize;
        let n_ff = meta.feed_forward_length.ok_or_else(|| {
            Error::InvalidGguf("required metadata feed_forward_length is missing".into())
        })? as usize;

        let token_embd_info = gguf
            .tensor("token_embd.weight")
            .ok_or_else(|| Error::MissingTensor("token_embd.weight".into()))?;
        if token_embd_info.dimensions.first().copied() != Some(n_embd as u64) {
            return Err(Error::InvalidGguf(format!(
                "token_embd.weight first dim {:?} != embedding_length {n_embd}",
                token_embd_info.dimensions
            )));
        }
        let token_embd_cols = token_embd_info.dimensions.get(1).copied().unwrap_or(1) as usize;
        let token_embd = gguf.dequantize_tensor("token_embd.weight")?;

        let (token_types, token_types_cols) = if gguf.tensor("token_types.weight").is_some() {
            let info = gguf.tensor("token_types.weight").unwrap();
            let cols = info.dimensions.get(1).copied().unwrap_or(1) as usize;
            (Some(gguf.dequantize_tensor("token_types.weight")?), cols)
        } else {
            (None, 0)
        };

        let token_embd_norm = load_norm(gguf, "token_embd_norm", n_embd)?;

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(load_layer(gguf, i, n_embd, n_ff)?);
        }

        Ok(Self {
            token_embd,
            token_embd_cols,
            token_types,
            token_types_cols,
            token_embd_norm,
            layers,
        })
    }
}

fn load_norm(gguf: &GgufFile, stem: &str, n_embd: usize) -> Result<LayerNormW> {
    let weight_name = format!("{stem}.weight");
    let bias_name = format!("{stem}.bias");
    let weight = gguf.dequantize_tensor(&weight_name)?;
    let bias = gguf.dequantize_tensor(&bias_name)?;
    if weight.len() != n_embd || bias.len() != n_embd {
        return Err(Error::InvalidGguf(format!(
            "{stem} expected {n_embd} elems, got weight={} bias={}",
            weight.len(),
            bias.len()
        )));
    }
    Ok(LayerNormW { weight, bias })
}

fn load_layer(gguf: &GgufFile, i: usize, n_embd: usize, n_ff: usize) -> Result<LayerWeights> {
    let qkv = format!("blk.{i}.attn_qkv.weight");
    let qkv_b = format!("blk.{i}.attn_qkv.bias");
    let wo = format!("blk.{i}.attn_output.weight");
    let wo_b = format!("blk.{i}.attn_output.bias");
    let up = format!("blk.{i}.ffn_up.weight");
    let gate = format!("blk.{i}.ffn_gate.weight");
    let down = format!("blk.{i}.ffn_down.weight");

    let attn_qkv = gguf.dequantize_tensor(&qkv)?;
    expect_len(&qkv, attn_qkv.len(), n_embd * 3 * n_embd)?;
    let attn_qkv_bias = optional_tensor(gguf, &qkv_b)?;
    if let Some(ref b) = attn_qkv_bias {
        expect_len(&qkv_b, b.len(), 3 * n_embd)?;
    }

    let attn_output = gguf.dequantize_tensor(&wo)?;
    expect_len(&wo, attn_output.len(), n_embd * n_embd)?;
    let attn_output_bias = optional_tensor(gguf, &wo_b)?;
    if let Some(ref b) = attn_output_bias {
        expect_len(&wo_b, b.len(), n_embd)?;
    }

    let ffn_up = gguf.dequantize_tensor(&up)?;
    expect_len(&up, ffn_up.len(), n_embd * n_ff)?;
    let ffn_gate = optional_tensor(gguf, &gate)?;
    if let Some(ref g) = ffn_gate {
        expect_len(&gate, g.len(), n_embd * n_ff)?;
    }
    let ffn_down = gguf.dequantize_tensor(&down)?;
    expect_len(&down, ffn_down.len(), n_embd * n_ff)?;

    Ok(LayerWeights {
        attn_qkv,
        attn_qkv_bias,
        attn_output,
        attn_output_bias,
        attn_output_norm: load_norm(gguf, &format!("blk.{i}.attn_output_norm"), n_embd)?,
        ffn_up,
        ffn_gate,
        ffn_down,
        layer_output_norm: load_norm(gguf, &format!("blk.{i}.layer_output_norm"), n_embd)?,
    })
}

fn optional_tensor(gguf: &GgufFile, name: &str) -> Result<Option<Vec<f32>>> {
    if gguf.tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(gguf.dequantize_tensor(name)?))
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::InvalidGguf(format!(
            "tensor {name} has {got} elems, expected {want}"
        )));
    }
    Ok(())
}
