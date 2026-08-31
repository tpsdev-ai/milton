//! Dequant-to-f32 for the types present in nomic-embed-text-v1.5 Q4_K_M
//! plus the issue-required Q8_0 / F16 kernels.
//!
//! Kernels mirror llama.cpp `dequantize_row_*` (`ggml/src/ggml-quants.c`) and
//! camelid's `decode_q*_tensor` dispatch (MIT — github.com/timtoole02/Camelid).
//! llama.cpp remains the correctness oracle; these are the compute path.
//!
//! Unverified types refuse. Never guess.

use crate::error::{Error, Result};
use crate::gguf::TensorType;

const QK8_0: usize = 32;
const QK8_0_BYTES: usize = 34;
const QK_K: usize = 256;
const Q4_K_BYTES: usize = 144;
const Q5_K_BYTES: usize = 176;
const Q6_K_BYTES: usize = 210;

/// ggml `ggml_compute_fp16_to_fp32` (ggml-impl.h) — Maratyszcza half→float.
/// Scales in GGUF are ggml_half; a conversion that is merely "close" is wrong.
pub fn f16_to_f32(h: u16) -> f32 {
    let w = u32::from(h) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    let exp_offset = 0xE0u32 << 23;
    let exp_scale = f32::from_bits(0x0780_0000); // 0x1.0p-112
    let normalized_value = f32::from_bits((two_w >> 4).wrapping_add(exp_offset)) * exp_scale;
    let magic_mask = 126u32 << 23;
    let magic_bias = 0.5f32;
    let denormalized_value = f32::from_bits((two_w >> 17) | magic_mask) - magic_bias;
    let denormalized_cutoff = 1u32 << 27;
    let result = sign
        | if two_w < denormalized_cutoff {
            denormalized_value.to_bits()
        } else {
            normalized_value.to_bits()
        };
    f32::from_bits(result)
}

pub fn dequantize(tensor_type: TensorType, bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    match tensor_type {
        TensorType::F32 => dequantize_f32(bytes, n_elements, name),
        TensorType::F16 => dequantize_f16(bytes, n_elements, name),
        TensorType::Q8_0 => dequantize_q8_0(bytes, n_elements, name),
        TensorType::Q4K => dequantize_q4_k(bytes, n_elements, name),
        TensorType::Q5K => dequantize_q5_k(bytes, n_elements, name),
        TensorType::Q6K => dequantize_q6_k(bytes, n_elements, name),
        TensorType::Unknown(id) => Err(Error::UnsupportedTensorType {
            name: name.to_string(),
            type_id: id,
            type_name: tensor_type.name(),
        }),
    }
}

fn expect_len(actual: usize, expected: usize, name: &str, ty: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidTensorData(format!(
            "tensor {name} {ty} byte length {actual} does not match expected {expected}"
        )));
    }
    Ok(())
}

fn dequantize_f32(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    expect_len(bytes.len(), n_elements * 4, name, "F32")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dequantize_f16(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    expect_len(bytes.len(), n_elements * 2, name, "F16")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// llama.cpp `dequantize_row_q8_0`: `y[i] = d * qs[i]` per 32-value block.
fn dequantize_q8_0(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    if n_elements % QK8_0 != 0 {
        return Err(Error::InvalidTensorData(format!(
            "tensor {name} Q8_0 n_elements {n_elements} not divisible by {QK8_0}"
        )));
    }
    let n_blocks = n_elements / QK8_0;
    expect_len(bytes.len(), n_blocks * QK8_0_BYTES, name, "Q8_0")?;
    let mut out = Vec::with_capacity(n_elements);
    for block in bytes.chunks_exact(QK8_0_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &q in &block[2..] {
            out.push(d * (q as i8 as f32));
        }
    }
    Ok(out)
}

/// llama.cpp `get_scale_min_k4` — 6-bit packed scales/mins in 12 bytes.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// llama.cpp `dequantize_row_q4_K` (QK_K=256, 144-byte super-block).
///
/// ```text
/// block_q4_K { d:f16, dmin:f16, scales[12], qs[128] }
/// for each 64-wide pair: y = d*sc * nibble - dmin*m
/// ```
fn dequantize_q4_k(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    dequantize_q4_k_scaled(bytes, n_elements, name, 1.0)
}

fn dequantize_q4_k_scaled(
    bytes: &[u8],
    n_elements: usize,
    name: &str,
    scale_mul: f32,
) -> Result<Vec<f32>> {
    if n_elements % QK_K != 0 {
        return Err(Error::InvalidTensorData(format!(
            "tensor {name} Q4_K n_elements {n_elements} not divisible by {QK_K}"
        )));
    }
    let n_blocks = n_elements / QK_K;
    expect_len(bytes.len(), n_blocks * Q4_K_BYTES, name, "Q4_K")?;
    let mut out = Vec::with_capacity(n_elements);
    for block in bytes.chunks_exact(Q4_K_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]])) * scale_mul;
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]])) * scale_mul;
        let scales = &block[4..16];
        let mut q = &block[16..];
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let m1 = min * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2 = min * f32::from(m2);
            for l in 0..32 {
                out.push(d1 * f32::from(q[l] & 0x0f) - m1);
            }
            for l in 0..32 {
                out.push(d2 * f32::from(q[l] >> 4) - m2);
            }
            q = &q[32..];
            is += 2;
        }
    }
    Ok(out)
}

/// llama.cpp `dequantize_row_q5_K` (QK_K=256, 176-byte super-block).
///
/// This GGUF's Q4_K_M mix stores attn_qkv as Q5_K (confirmed against the file).
fn dequantize_q5_k(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    dequantize_q5_k_scaled(bytes, n_elements, name, 1.0)
}

fn dequantize_q5_k_scaled(
    bytes: &[u8],
    n_elements: usize,
    name: &str,
    scale_mul: f32,
) -> Result<Vec<f32>> {
    if n_elements % QK_K != 0 {
        return Err(Error::InvalidTensorData(format!(
            "tensor {name} Q5_K n_elements {n_elements} not divisible by {QK_K}"
        )));
    }
    let n_blocks = n_elements / QK_K;
    expect_len(bytes.len(), n_blocks * Q5_K_BYTES, name, "Q5_K")?;
    let mut out = Vec::with_capacity(n_elements);
    for block in bytes.chunks_exact(Q5_K_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]])) * scale_mul;
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]])) * scale_mul;
        let scales = &block[4..16];
        let qh = &block[16..48];
        let mut ql = &block[48..];
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = min * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = min * f32::from(m2);
            for l in 0..32 {
                let q = f32::from(ql[l] & 0x0f) + if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * q - min1);
            }
            for l in 0..32 {
                let q = f32::from(ql[l] >> 4) + if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d2 * q - min2);
            }
            ql = &ql[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

/// llama.cpp `dequantize_row_q6_K`.
/// Q4_K_M files typically keep a few tensors (token embd / output / attn_v /
/// ffn_down) as Q6_K. Confirm against the file; implement because it is in
/// the mix, not because we are generalizing past nomic v1.5.
fn dequantize_q6_k(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    if n_elements % QK_K != 0 {
        return Err(Error::InvalidTensorData(format!(
            "tensor {name} Q6_K n_elements {n_elements} not divisible by {QK_K}"
        )));
    }
    let n_blocks = n_elements / QK_K;
    expect_len(bytes.len(), n_blocks * Q6_K_BYTES, name, "Q6_K")?;
    let mut out = vec![0.0f32; n_elements];
    let mut y_off = 0usize;
    for block in bytes.chunks_exact(Q6_K_BYTES) {
        // block_q6_K { ql[128], qh[64], scales[16], d:f16 }
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        for _ in 0..(QK_K / 128) {
            for l in 0..32 {
                let is = l / 16;
                // C: (int8_t)((ql & 0xF) | ((qh >> k) & 3) << 4) - 32
                let q1 = ((ql[ql_off + l] & 0x0f) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0f) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                let s1 = sc[sc_off + is] as i8 as f32;
                let s2 = sc[sc_off + is + 2] as i8 as f32;
                let s3 = sc[sc_off + is + 4] as i8 as f32;
                let s4 = sc[sc_off + is + 6] as i8 as f32;
                out[y_off + l] = d * s1 * (q1 as f32);
                out[y_off + l + 32] = d * s2 * (q2 as f32);
                out[y_off + l + 64] = d * s3 * (q3 as f32);
                out[y_off + l + 96] = d * s4 * (q4 as f32);
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    Ok(out)
}

/// Deliberately-wrong dequant: multiply the Q4_K super-block scale (`d` and
/// `dmin`) by 2. The gate must catch this RED and name the tensor.
pub fn dequantize_wrong_block_scale(
    tensor_type: TensorType,
    bytes: &[u8],
    n_elements: usize,
    name: &str,
) -> Result<Vec<f32>> {
    match tensor_type {
        TensorType::Q4K => dequantize_q4_k_scaled(bytes, n_elements, name, 2.0),
        TensorType::Q5K => dequantize_q5_k_scaled(bytes, n_elements, name, 2.0),
        TensorType::Q6K | TensorType::Q8_0 | TensorType::F16 | TensorType::F32 => {
            let mut out = dequantize(tensor_type, bytes, n_elements, name)?;
            for v in &mut out {
                *v *= 2.0;
            }
            Ok(out)
        }
        other => dequantize(other, bytes, n_elements, name),
    }
}

/// Deliberately-wrong dequant: decode the bytes as Q8_0 regardless of the
/// recorded type (or as F16 if the tensor *is* Q8_0). The gate must catch
/// this RED and name the tensor.
pub fn dequantize_wrong_type(
    tensor_type: TensorType,
    bytes: &[u8],
    n_elements: usize,
    name: &str,
) -> Result<Vec<f32>> {
    match tensor_type {
        TensorType::Q8_0 => dequantize_f16(bytes, n_elements, name),
        _ => {
            // Force Q8_0 decode. Length will usually fail closed; if it
            // happens to be a multiple of 34 we still produce wrong values.
            if bytes.len() % QK8_0_BYTES == 0 {
                let n = (bytes.len() / QK8_0_BYTES) * QK8_0;
                dequantize_q8_0(bytes, n, name)
            } else {
                Err(Error::InvalidTensorData(format!(
                    "wrong-type: refusing to decode {name} ({}) as Q8_0 (len {})",
                    tensor_type.name(),
                    bytes.len()
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn f32_roundtrips_bit_exact() {
        let vals = [-1.5f32, 0.0, 3.25, f32::MIN_POSITIVE, -0.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = dequantize(TensorType::F32, &bytes, vals.len(), "t").unwrap();
        for (a, b) in out.iter().zip(vals.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn unknown_type_refused() {
        let err = dequantize(TensorType::Unknown(99), &[0u8; 4], 1, "blk.0").unwrap_err();
        assert!(matches!(err, Error::UnsupportedTensorType { type_id: 99, .. }));
    }

    #[test]
    fn wrong_length_fails_closed() {
        let err = dequantize(TensorType::F32, &[0u8; 6], 2, "t").unwrap_err();
        assert!(matches!(err, Error::InvalidTensorData(_)));
    }
}
