//! llama.cpp quantized `ggml_mul_mat` for the K-quants in this GGUF.
//!
//! Goldens were produced by llama-embedding on Q4_K_M weights. That path
//! quantizes activations to Q8_K and uses `vec_dot_q*_K_q8_K` — it is not
//! dequant-to-f32 then f32 GEMM. Matching the gate (1e-5 max_abs) requires
//! the same integer-dot path. Kernels mirror
//! `quantize_row_q8_K_ref` + `ggml_vec_dot_q{4,5,6}_K_q8_K_generic`.

use crate::dequant::{f16_to_f32, get_scale_min_k4};
use crate::gguf::TensorType;
use crate::ops::matmul;

pub const QK_K: usize = 256;
const Q4_K_BYTES: usize = 144;
const Q5_K_BYTES: usize = 176;
const Q6_K_BYTES: usize = 210;

#[derive(Clone, Debug)]
pub struct QuantMat {
    pub ty: TensorType,
    pub bytes: Vec<u8>,
    pub f32: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
}

#[derive(Clone, Copy, Debug)]
struct BlockQ8K {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

/// ggml `nearest_int`: magic-number round, memcpy float bits → int.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0;
    let i = i32::from_le_bytes(val.to_le_bytes());
    (i & 0x007f_ffff) - 0x0040_0000
}

/// llama.cpp `quantize_row_q8_K_ref`.
fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) {
    debug_assert_eq!(x.len(), out.len() * QK_K);
    for (block, chunk) in out.iter_mut().zip(x.chunks_exact(QK_K)) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in chunk {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            *block = BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; QK_K / 16],
            };
            continue;
        }
        let iscale = -127.0 / max;
        let mut qs = [0i8; QK_K];
        for j in 0..QK_K {
            let v = nearest_int(iscale * chunk[j]);
            qs[j] = v.min(127) as i8;
        }
        let mut bsums = [0i16; QK_K / 16];
        for j in 0..QK_K / 16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += i32::from(qs[j * 16 + ii]);
            }
            bsums[j] = sum as i16;
        }
        *block = BlockQ8K {
            d: 1.0 / iscale,
            qs,
            bsums,
        };
    }
}

fn vec_dot_q4_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q4_K_BYTES);
    let mut sumf = 0.0f32;
    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q4_K_BYTES..(i + 1) * Q4_K_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut aux8 = [0i8; QK_K];
        let mut a = 0usize;
        let mut qoff = 0usize;
        for _ in 0..QK_K / 64 {
            for l in 0..32 {
                aux8[a + l] = (qs[qoff + l] & 0x0f) as i8;
            }
            a += 32;
            for l in 0..32 {
                aux8[a + l] = (qs[qoff + l] >> 4) as i8;
            }
            a += 32;
            qoff += 32;
        }
        let mut sumi = 0i32;
        for j in 0..QK_K / 16 {
            let (_, m) = get_scale_min_k4(j / 2, scales);
            sumi += i32::from(yb.bsums[j]) * i32::from(m);
        }
        let mut acc = 0i32;
        for group in 0..8 {
            let (sc, _) = get_scale_min_k4(group, scales);
            let sc = i32::from(sc);
            let base = group * 32;
            for l in 0..32 {
                acc += sc * i32::from(aux8[base + l]) * i32::from(yb.qs[base + l]);
            }
        }
        sumf += d * yb.d * acc as f32;
        sumf -= dmin * yb.d * sumi as f32;
    }
    sumf
}

fn vec_dot_q5_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q5_K_BYTES);
    let mut sumf = 0.0f32;
    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q5_K_BYTES..(i + 1) * Q5_K_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut aux8 = [0i8; QK_K];
        let mut a = 0usize;
        let mut qoff = 0usize;
        let mut mbit: u8 = 1;
        for _ in 0..QK_K / 64 {
            for l in 0..32 {
                let mut v = (qs[qoff + l] & 0x0f) as i8;
                if qh[l] & mbit != 0 {
                    v += 16;
                }
                aux8[a + l] = v;
            }
            a += 32;
            mbit <<= 1;
            for l in 0..32 {
                let mut v = (qs[qoff + l] >> 4) as i8;
                if qh[l] & mbit != 0 {
                    v += 16;
                }
                aux8[a + l] = v;
            }
            a += 32;
            mbit <<= 1;
            qoff += 32;
        }
        let mut sumi = 0i32;
        for j in 0..QK_K / 16 {
            let (_, m) = get_scale_min_k4(j / 2, scales);
            sumi += i32::from(yb.bsums[j]) * i32::from(m);
        }
        let mut acc = 0i32;
        for group in 0..8 {
            let (sc, _) = get_scale_min_k4(group, scales);
            let sc = i32::from(sc);
            let base = group * 32;
            for l in 0..32 {
                acc += sc * i32::from(aux8[base + l]) * i32::from(yb.qs[base + l]);
            }
        }
        sumf += d * yb.d * acc as f32;
        sumf -= dmin * yb.d * sumi as f32;
    }
    sumf
}

fn vec_dot_q6_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q6_K_BYTES);
    let mut sumf = 0.0f32;
    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut aux8 = [0i8; QK_K];
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut a = 0usize;
        for _ in 0..QK_K / 128 {
            for l in 0..32 {
                aux8[a + l] = ((ql[ql_off + l] & 0x0f) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                aux8[a + l + 32] =
                    ((ql[ql_off + l + 32] & 0x0f) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[a + l + 64] =
                    ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[a + l + 96] =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
            }
            a += 128;
            ql_off += 64;
            qh_off += 32;
        }
        let mut acc = 0i32;
        // 16 groups of 16: each scale is int8
        for group in 0..16 {
            let scale = i32::from(sc[group] as i8);
            let base = group * 16;
            for l in 0..16 {
                acc += scale * i32::from(aux8[base + l]) * i32::from(yb.qs[base + l]);
            }
        }
        sumf += d * yb.d * acc as f32;
    }
    sumf
}

fn col_bytes(ty: TensorType, n_in: usize) -> usize {
    let nb = n_in / QK_K;
    match ty {
        TensorType::Q4K => nb * Q4_K_BYTES,
        TensorType::Q5K => nb * Q5_K_BYTES,
        TensorType::Q6K => nb * Q6_K_BYTES,
        _ => 0,
    }
}

fn q8k_enabled() -> bool {
    match std::env::var("MILTON_Q8K") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            false
        }
        _ => true,
    }
}

/// ggml `mul_mat`: dst[o, t] = vec_dot(W_col[o], Q8_K(x[t])).
pub fn matmul_ggml(x: &[f32], w: &QuantMat, n_tokens: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), n_tokens * w.n_in);
    debug_assert_eq!(y.len(), n_tokens * w.n_out);
    debug_assert_eq!(w.f32.len(), w.n_out * w.n_in);
    let use_q8k = q8k_enabled()
        && matches!(w.ty, TensorType::Q4K | TensorType::Q5K | TensorType::Q6K)
        && w.n_in % QK_K == 0;
    if !use_q8k {
        matmul(x, &w.f32, n_tokens, w.n_in, w.n_out, y);
        return;
    }
    let n_blocks = w.n_in / QK_K;
    let cb = col_bytes(w.ty, w.n_in);
    debug_assert_eq!(w.bytes.len(), cb * w.n_out);
    let mut qrows = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        n_tokens * n_blocks
    ];
    for t in 0..n_tokens {
        quantize_row_q8_k(
            &x[t * w.n_in..(t + 1) * w.n_in],
            &mut qrows[t * n_blocks..(t + 1) * n_blocks],
        );
    }
    for t in 0..n_tokens {
        let yrow = &qrows[t * n_blocks..(t + 1) * n_blocks];
        for o in 0..w.n_out {
            let col = &w.bytes[o * cb..(o + 1) * cb];
            y[t * w.n_out + o] = match w.ty {
                TensorType::Q4K => vec_dot_q4_k_q8_k(col, yrow),
                TensorType::Q5K => vec_dot_q5_k_q8_k(col, yrow),
                TensorType::Q6K => vec_dot_q6_k_q8_k(col, yrow),
                _ => unreachable!(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_int_matches_magic() {
        assert_eq!(nearest_int(0.0), 0);
        assert_eq!(nearest_int(1.0), 1);
        assert_eq!(nearest_int(-1.0), -1);
        assert_eq!(nearest_int(1.4), 1);
        assert_eq!(nearest_int(1.6), 2);
        assert_eq!(nearest_int(-1.6), -2);
    }

    #[test]
    fn q8k_zero_row_is_zero() {
        let x = vec![0.0f32; QK_K];
        let mut out = [BlockQ8K {
            d: 1.0,
            qs: [1; QK_K],
            bsums: [1; QK_K / 16],
        }];
        quantize_row_q8_k(&x, &mut out);
        assert_eq!(out[0].d, 0.0);
        assert!(out[0].qs.iter().all(|&q| q == 0));
    }
}
