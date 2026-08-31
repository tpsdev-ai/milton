//! llama.cpp quantized `ggml_mul_mat` for the K-quants in this GGUF.
//!
//! Goldens were produced by llama-embedding on Q4_K_M weights with
//! `CPU_REPACK` (AVX2 `q4_K_8x8`). That path quantizes activations to Q8_K.
//! Q4_K uses `make_block_q4_Kx8` + `ggml_gemv_q4_K_8x8_q8_K`.
//! Q5_K uses AVX2 `ggml_vec_dot_q5_K_q8_K` (`quants.c`) when AVX2+FMA is
//! present — generic `vec_dot` is a different numeric path and seeds a
//! ~2e-6 residual that later Q8_K `nearest_int` flips compound. Q6_K
//! uses AVX2 `ggml_vec_dot_q6_K_q8_K` the same way (generic 8-lane is
//! fallback only). Matching the gate (1e-5 max_abs) requires this path,
//! not dequant-to-f32.
//!
//! Q4_K residual after the Q6_K AVX2 port (`509c57a`): GEMV-only
//! (`quantize_row_q8_K` + `ggml_gemv_q4_K_8x8_q8_K`) is what the
//! pinned llama.cpp *graph* emits for this GGUF. Layer-dump on
//! `short-hello-document` (n=7): Q/K/V and layer-0 Q4_K match; the
//! first gate-breaking Q4_K fail is `kqv_out-1` token 0. DSO
//! `ggml_gemm_q4_K_8x8_q8_K` (4x8) on that tile disagrees with the
//! graph dump (max_abs=3.49e-4); DSO GEMV + row Q8_K matches the dump
//! (1.5e-7). AVX2 4x8 kernels in `q4k_avx2` stay bit-exact vs the DSO
//! and are not used on the embed path. Do not invent another quant
//! type. Do not touch `quantize_row_q8_K` or Q5_K `vec_dot`.

use crate::dequant::{f16_to_f32, get_scale_min_k4};
use crate::gguf::TensorType;
use crate::ops::matmul;

pub const QK_K: usize = 256;
const Q4_K_BYTES: usize = 144;
const Q5_K_BYTES: usize = 176;
const Q6_K_BYTES: usize = 210;

const Q4_KX8_BYTES: usize = 1152; // 8*f16 d + 8*f16 dmin + 96 scales + 1024 qs

#[derive(Clone, Debug)]
pub struct QuantMat {
    pub ty: TensorType,
    pub bytes: Vec<u8>,
    pub f32: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
    /// llama.cpp `CPU_REPACK` Q4_K → `block_q4_Kx8` when `n_out % 8 == 0`.
    /// Goldens were produced with AVX2 `ggml_gemv_q4_K_8x8_q8_K`, not generic `vec_dot`.
    pub q4k_8x8: Option<Vec<u8>>,
}

impl QuantMat {
    pub fn new(ty: TensorType, bytes: Vec<u8>, f32: Vec<f32>, n_in: usize, n_out: usize) -> Self {
        let q4k_8x8 = if ty == TensorType::Q4K && n_out % 8 == 0 && n_in % QK_K == 0 {
            Some(repack_q4_k_8x8(&bytes, n_in, n_out))
        } else {
            None
        };
        Self {
            ty,
            bytes,
            f32,
            n_in,
            n_out,
            q4k_8x8,
        }
    }
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

fn unpack_scale_min_k4(scales12: &[u8]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes(scales12[0..4].try_into().unwrap());
    utmp[1] = u32::from_le_bytes(scales12[4..8].try_into().unwrap());
    utmp[2] = u32::from_le_bytes(scales12[8..12].try_into().unwrap());
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    let mut sc = [0u8; 8];
    let mut mins = [0u8; 8];
    sc[0..4].copy_from_slice(&utmp[0].to_le_bytes());
    sc[4..8].copy_from_slice(&utmp[1].to_le_bytes());
    mins[0..4].copy_from_slice(&utmp[2].to_le_bytes());
    mins[4..8].copy_from_slice(&utmp[3].to_le_bytes());
    (sc, mins)
}

fn vec_dot_q5_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q5_k_q8_k_avx2(w, y) };
        }
    }
    vec_dot_q5_k_q8_k_generic(w, y)
}

/// llama.cpp `ggml_vec_dot_q5_K_q8_K` AVX2 (`quants.c` ~1919).
///
/// Bit-exact: `maddubs` + `madd_epi16`, `_mm256_fmadd_ps` of `cvtepi32_ps(sumi)`
/// with broadcast `d` into an 8-wide `acc` across superblocks, mins via
/// `hadd`/`madd` into scalar `summs`, `hsum_float_8` pairwise order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5_k_q8_k_avx2(w: &[u8], y: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(w.len(), y.len() * Q5_K_BYTES);

    // llama.cpp `get_scale_shuffle_k4` — 8×32-byte rows, each row broadcasts
    // one u16 scale across the ymm (`_mm256_shuffle_epi8` is two 128-bit shuffles).
    #[rustfmt::skip]
    const K_SHUFFLE: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    ];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4 = _mm256_set1_epi8(0x0F);
    let mzero = _mm_setzero_si128();
    let mone = _mm256_set1_epi8(1);

    let mut acc = _mm256_setzero_ps();
    let mut summs = 0.0f32;

    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q5_K_BYTES..(i + 1) * Q5_K_BYTES];
        let d = yb.d * f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = -yb.d * f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes(block[4..8].try_into().unwrap());
        utmp[1] = u32::from_le_bytes(block[8..12].try_into().unwrap());
        utmp[2] = u32::from_le_bytes(block[12..16].try_into().unwrap());
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        let q8sums = _mm256_loadu_si256(yb.bsums.as_ptr().cast());
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256::<0>(q8sums),
            _mm256_extracti128_si256::<1>(q8sums),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256::<1>(mins_and_scales), q8s);
        let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (_mm_extract_epi32::<0>(hsum) as f32);

        let sc128 = _mm256_extracti128_si256::<0>(mins_and_scales);
        // MM256_SET_M128I(sc128, sc128)
        let scales = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(sc128), sc128);

        let hbits = _mm256_loadu_si256(block[16..48].as_ptr().cast());
        let mut hmask = mone;
        let mut sumi = _mm256_setzero_si256();
        let mut bit = 0i32;
        let q5 = block[48..176].as_ptr();
        let q8 = yb.qs.as_ptr();

        for j in 0..QK_K / 64 {
            let scale_0 = _mm256_shuffle_epi8(
                scales,
                _mm256_loadu_si256(K_SHUFFLE.as_ptr().add((2 * j) * 32).cast()),
            );
            let scale_1 = _mm256_shuffle_epi8(
                scales,
                _mm256_loadu_si256(K_SHUFFLE.as_ptr().add((2 * j + 1) * 32).cast()),
            );

            let q5bits = _mm256_loadu_si256(q5.add(j * 32).cast());

            let q5l_0 = _mm256_and_si256(q5bits, m4);
            let q5h_0 = _mm256_slli_epi16::<4>(_mm256_srl_epi16(
                _mm256_and_si256(hbits, hmask),
                _mm_cvtsi32_si128(bit),
            ));
            bit += 1;
            let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16::<1>(hmask);

            let q5l_1 = _mm256_and_si256(_mm256_srli_epi16::<4>(q5bits), m4);
            let q5h_1 = _mm256_slli_epi16::<4>(_mm256_srl_epi16(
                _mm256_and_si256(hbits, hmask),
                _mm_cvtsi32_si128(bit),
            ));
            bit += 1;
            let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16::<1>(hmask);

            let q8_0 = _mm256_loadu_si256(q8.add(j * 64).cast());
            let q8_1 = _mm256_loadu_si256(q8.add(j * 64 + 32).cast());

            let mut p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    // llama.cpp `hsum_float_8`: high128+low128 → movehl add → movehdup add_ss
    let mut res = _mm256_extractf128_ps::<1>(acc);
    res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res) + summs
}

fn vec_dot_q5_k_q8_k_generic(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q5_K_BYTES);
    // llama.cpp generic: 8 int32 lanes, `sums[l] += d * aux32[l]` per superblock.
    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q5_K_BYTES..(i + 1) * Q5_K_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_scale_min_k4(&block[4..16]);
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
            sumi += i32::from(yb.bsums[j]) * i32::from(mins[j / 2]);
        }
        // llama.cpp generic: 8 stride-8 int32 lanes, then `sums[l] += d * aux32[l]`.
        let mut aux32 = [0i32; 8];
        let mut q8off = 0usize;
        let mut aoff = 0usize;
        for group in 0..8 {
            let scale = i32::from(scales[group]);
            for _chunk in 0..4 {
                for l in 0..8 {
                    let prod = i32::from(yb.qs[q8off + l]) * i32::from(aux8[aoff + l]);
                    aux32[l] += scale * prod;
                }
                q8off += 8;
                aoff += 8;
            }
        }
        let d = d * yb.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        sumf -= dmin * yb.d * sumi as f32;
    }
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

fn vec_dot_q6_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q6_k_q8_k_avx2(w, y) };
        }
    }
    vec_dot_q6_k_q8_k_generic(w, y)
}

/// llama.cpp `ggml_vec_dot_q6_K_q8_K` AVX2 (`quants.c` ~2129).
///
/// Bit-exact: 6-bit reconstruct, (−32) via `maddubs(m32s, q8)` subtract,
/// `madd_epi16` of `cvtepi8_epi16` scales, `_mm256_fmadd_ps` of
/// `broadcast_ss(d)` into 8-wide `acc` across superblocks, `hsum_float_8`
/// pairwise order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q6_k_q8_k_avx2(w: &[u8], y: &[BlockQ8K]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(w.len(), y.len() * Q6_K_BYTES);

    // llama.cpp `get_scale_shuffle` — 8×16-byte rows (`_mm_shuffle_epi8`).
    #[rustfmt::skip]
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15,
    ];

    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(3);
    let m32s = _mm256_set1_epi8(32);

    let mut acc = _mm256_setzero_ps();

    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let d = yb.d * f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        let q4 = block[0..128].as_ptr();
        let qh = block[128..192].as_ptr();
        let q8 = yb.qs.as_ptr();
        let scales = _mm_loadu_si128(block[192..208].as_ptr().cast());

        let mut sumi = _mm256_setzero_si256();
        let mut is = 0usize;

        for j in 0..QK_K / 128 {
            let scale_0 = _mm_shuffle_epi8(
                scales,
                _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is + 0) * 16).cast()),
            );
            let scale_1 = _mm_shuffle_epi8(
                scales,
                _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is + 1) * 16).cast()),
            );
            let scale_2 = _mm_shuffle_epi8(
                scales,
                _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is + 2) * 16).cast()),
            );
            let scale_3 = _mm_shuffle_epi8(
                scales,
                _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is + 3) * 16).cast()),
            );
            is += 4;

            let q4bits1 = _mm256_loadu_si256(q4.add(j * 64).cast());
            let q4bits2 = _mm256_loadu_si256(q4.add(j * 64 + 32).cast());
            let q4bits_h = _mm256_loadu_si256(qh.add(j * 32).cast());

            let q4h_0 = _mm256_slli_epi16::<4>(_mm256_and_si256(q4bits_h, m2));
            let q4h_1 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                _mm256_srli_epi16::<2>(q4bits_h),
                m2,
            ));
            let q4h_2 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                _mm256_srli_epi16::<4>(q4bits_h),
                m2,
            ));
            let q4h_3 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                _mm256_srli_epi16::<6>(q4bits_h),
                m2,
            ));

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            let q4_2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits1), m4),
                q4h_2,
            );
            let q4_3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits2), m4),
                q4h_3,
            );

            let q8_0 = _mm256_loadu_si256(q8.add(j * 128).cast());
            let q8_1 = _mm256_loadu_si256(q8.add(j * 128 + 32).cast());
            let q8_2 = _mm256_loadu_si256(q8.add(j * 128 + 64).cast());
            let q8_3 = _mm256_loadu_si256(q8.add(j * 128 + 96).cast());

            let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    // llama.cpp `hsum_float_8`: high128+low128 → movehl add → movehdup add_ss
    let mut res = _mm256_extractf128_ps::<1>(acc);
    res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

fn vec_dot_q6_k_q8_k_generic(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q6_K_BYTES);
    // llama.cpp generic Q6_K: 8 int32 lanes, float acc per lane, then sum.
    let mut sums = [0.0f32; 8];
    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]])) * yb.d;
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
        let mut aux32 = [0i32; 8];
        for group in 0..16 {
            let scale = i32::from(sc[group] as i8);
            let base = group * 16;
            for l in 0..8 {
                aux32[l] += scale * i32::from(aux8[base + l]) * i32::from(yb.qs[base + l]);
                aux32[l] += scale * i32::from(aux8[base + 8 + l]) * i32::from(yb.qs[base + 8 + l]);
            }
        }
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }
    sums.iter().sum()
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

fn repack_enabled() -> bool {
    match std::env::var("MILTON_REPACK") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            false
        }
        _ => true,
    }
}

/// llama.cpp `make_block_q4_Kx8` (`repack.cpp`), `blck_size_interleave = 8`.
fn make_block_q4_kx8(cols: [&[u8]; 8]) -> [u8; Q4_KX8_BYTES] {
    let mut out = [0u8; Q4_KX8_BYTES];
    for i in 0..8 {
        out[i * 2] = cols[i][0];
        out[i * 2 + 1] = cols[i][1];
        out[16 + i * 2] = cols[i][2];
        out[16 + i * 2 + 1] = cols[i][3];
    }
    // Interleave 8-byte qs chunks, round-robin across the 8 columns.
    for i in 0..128 {
        let src_id = i % 8;
        let src_offset = (i / 8) * 8;
        let dst_offset = i * 8;
        let qs = &cols[src_id][16..144];
        out[128 + dst_offset..128 + dst_offset + 8]
            .copy_from_slice(&qs[src_offset..src_offset + 8]);
    }
    let mut scales_out = [0u8; 96];
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    for i in 0..4 {
        for j in 0..8 {
            let sc = &cols[j][4..16];
            s[j] = sc[i] & 63;
            m[j] = sc[i + 4] & 63;
        }
        let base = i * 12;
        scales_out[base] = (s[0] & 63).wrapping_add((s[4] & 48) << 2);
        scales_out[base + 1] = (s[1] & 63).wrapping_add((s[5] & 48) << 2);
        scales_out[base + 2] = (s[2] & 63).wrapping_add((s[6] & 48) << 2);
        scales_out[base + 3] = (s[3] & 63).wrapping_add((s[7] & 48) << 2);
        scales_out[base + 4] = (m[0] & 63).wrapping_add((m[4] & 48) << 2);
        scales_out[base + 5] = (m[1] & 63).wrapping_add((m[5] & 48) << 2);
        scales_out[base + 6] = (m[2] & 63).wrapping_add((m[6] & 48) << 2);
        scales_out[base + 7] = (m[3] & 63).wrapping_add((m[7] & 48) << 2);
        scales_out[base + 8] = (s[4] & 15).wrapping_add((m[4] & 15) << 4);
        scales_out[base + 9] = (s[5] & 15).wrapping_add((m[5] & 15) << 4);
        scales_out[base + 10] = (s[6] & 15).wrapping_add((m[6] & 15) << 4);
        scales_out[base + 11] = (s[7] & 15).wrapping_add((m[7] & 15) << 4);
    }
    for i in 0..4 {
        for j in 0..8 {
            let sc = &cols[j][4..16];
            s[j] = ((sc[i] & 192) >> 2) | (sc[i + 8] & 15);
            m[j] = ((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4);
        }
        let base = i * 12 + 48;
        scales_out[base] = (s[0] & 63).wrapping_add((s[4] & 48) << 2);
        scales_out[base + 1] = (s[1] & 63).wrapping_add((s[5] & 48) << 2);
        scales_out[base + 2] = (s[2] & 63).wrapping_add((s[6] & 48) << 2);
        scales_out[base + 3] = (s[3] & 63).wrapping_add((s[7] & 48) << 2);
        scales_out[base + 4] = (m[0] & 63).wrapping_add((m[4] & 48) << 2);
        scales_out[base + 5] = (m[1] & 63).wrapping_add((m[5] & 48) << 2);
        scales_out[base + 6] = (m[2] & 63).wrapping_add((m[6] & 48) << 2);
        scales_out[base + 7] = (m[3] & 63).wrapping_add((m[7] & 48) << 2);
        scales_out[base + 8] = (s[4] & 15).wrapping_add((m[4] & 15) << 4);
        scales_out[base + 9] = (s[5] & 15).wrapping_add((m[5] & 15) << 4);
        scales_out[base + 10] = (s[6] & 15).wrapping_add((m[6] & 15) << 4);
        scales_out[base + 11] = (s[7] & 15).wrapping_add((m[7] & 15) << 4);
    }
    out[32..128].copy_from_slice(&scales_out);
    out
}

fn repack_q4_k_8x8(bytes: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    let n_blocks = n_in / QK_K;
    let cb = n_blocks * Q4_K_BYTES;
    debug_assert_eq!(bytes.len(), cb * n_out);
    let n_groups = n_out / 8;
    let mut dst = Vec::with_capacity(n_groups * n_blocks * Q4_KX8_BYTES);
    for g in 0..n_groups {
        for x in 0..n_blocks {
            let col = |j: usize| {
                let off = (g * 8 + j) * cb + x * Q4_K_BYTES;
                &bytes[off..off + Q4_K_BYTES]
            };
            let blk = make_block_q4_kx8([
                col(0),
                col(1),
                col(2),
                col(3),
                col(4),
                col(5),
                col(6),
                col(7),
            ]);
            dst.extend_from_slice(&blk);
        }
    }
    dst
}

/// llama.cpp `ggml_gemv_q4_K_8x8_q8_K_generic` with AVX2-style per-superblock
/// float accum (`iacc * (d * a.d)` once per QK_K, not once per 8-wide chunk).
fn gemv_q4_k_8x8_q8_k(repack: &[u8], y: &[BlockQ8K], n_out: usize, out: &mut [f32]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let n_blocks = y.len();
    debug_assert_eq!(repack.len(), (n_out / 8) * n_blocks * Q4_KX8_BYTES);
    debug_assert_eq!(out.len(), n_out);
    let n_groups = n_out / 8;
    for x in 0..n_groups {
        let mut sumf = [0.0f32; 8];
        let mut sum_minf = [0.0f32; 8];
        for l in 0..n_blocks {
            let off = (x * n_blocks + l) * Q4_KX8_BYTES;
            let blk = &repack[off..off + Q4_KX8_BYTES];
            let mut d = [0.0f32; 8];
            let mut dmin = [0.0f32; 8];
            for j in 0..8 {
                d[j] = f16_to_f32(u16::from_le_bytes([blk[j * 2], blk[j * 2 + 1]]));
                dmin[j] = f16_to_f32(u16::from_le_bytes([
                    blk[16 + j * 2],
                    blk[16 + j * 2 + 1],
                ]));
            }
            let scales = &blk[32..128];
            let qs = &blk[128..];
            let yb = &y[l];

            let mut utmp = [0u32; 32];
            for sb in 0..8 {
                let base = sb * 12;
                utmp[sb * 4] = u32::from_le_bytes(scales[base..base + 4].try_into().unwrap());
                utmp[sb * 4 + 1] =
                    u32::from_le_bytes(scales[base + 4..base + 8].try_into().unwrap());
                utmp[sb * 4 + 2] =
                    u32::from_le_bytes(scales[base + 8..base + 12].try_into().unwrap());
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & KMASK2)
                    | (((utmp[sb * 4 + 1] >> 6) & KMASK3) << 4);
                let uaux_0 = utmp[sb * 4 + 1] & KMASK1;
                utmp[sb * 4 + 1] =
                    (utmp[sb * 4 + 2] & KMASK2) | (((utmp[sb * 4] >> 6) & KMASK3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4] &= KMASK1;
            }
            let mut ub = [0u8; 128];
            for i in 0..32 {
                ub[i * 4..i * 4 + 4].copy_from_slice(&utmp[i].to_le_bytes());
            }

            let mut iacc = [0i32; 8];
            for k in 0..16 {
                let scale_base = (k / 4) * 32;
                for j in 0..8 {
                    let mut sumi = 0i32;
                    for i in 0..8 {
                        let qbyte = qs[k * 64 + j * 8 + i];
                        let v0 = i32::from(qbyte & 0x0f);
                        let v1 = i32::from(qbyte >> 4);
                        let a_off = (k >> 2) * 64 + (k % 4) * 8 + i;
                        let a0 = i32::from(yb.qs[a_off]);
                        let a1 = i32::from(yb.qs[a_off + 32]);
                        let s0 = i32::from(ub[scale_base + j]);
                        let s1 = i32::from(ub[scale_base + 16 + j]);
                        sumi += v0 * a0 * s0 + v1 * a1 * s1;
                    }
                    iacc[j] += sumi;
                }
            }
            let mut iacc_min = [0i32; 8];
            for sb in 0..8 {
                let bsum = i32::from(yb.bsums[sb * 2]) + i32::from(yb.bsums[sb * 2 + 1]);
                for j in 0..8 {
                    iacc_min[j] += i32::from(ub[8 + sb * 16 + j]) * bsum;
                }
            }
            for j in 0..8 {
                let ds = d[j] * yb.d;
                sumf[j] += iacc[j] as f32 * ds;
                sum_minf[j] += iacc_min[j] as f32 * (dmin[j] * yb.d);
            }
        }
        for j in 0..8 {
            out[x * 8 + j] = sumf[j] - sum_minf[j];
        }
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
    let use_repack = repack_enabled() && w.ty == TensorType::Q4K && w.q4k_8x8.is_some();
    if use_repack {
        let packed = w.q4k_8x8.as_ref().unwrap();
        // GEMV + row Q8_K for every token. The 4x8 GEMM path matches the
        // llama.cpp DSO bit-exactly but *not* the embedding graph dump
        // (see crate docs). Stay on GEMV.
        for t in 0..n_tokens {
            let yrow = &qrows[t * n_blocks..(t + 1) * n_blocks];
            gemv_q4_k_8x8_q8_k(packed, yrow, w.n_out, &mut y[t * w.n_out..(t + 1) * w.n_out]);
        }
        return;
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

    #[test]
    fn q5k_column0_matches_dequant_and_vec_dot_tracks_f32() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let gguf_path = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if !gguf_path.exists() {
            return;
        }
        let gguf = crate::gguf::GgufFile::open(&gguf_path).unwrap();
        let info = gguf.tensor("blk.0.attn_qkv.weight").unwrap();
        let n_in = info.dimensions[0] as usize;
        let n_out = info.dimensions[1] as usize;
        let bytes = gguf.tensor_bytes(info).unwrap();
        let f32w = gguf.dequantize_tensor("blk.0.attn_qkv.weight").unwrap();
        let cb = col_bytes(TensorType::Q5K, n_in);
        let col0 = &bytes[0..cb];
        // Dequant column 0 via the existing kernel and compare to f32 layout.
        let deq = crate::dequant::dequantize(TensorType::Q5K, col0, n_in, "col0").unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..n_in {
            max_abs = max_abs.max((deq[i] - f32w[i]).abs());
        }
        assert!(
            max_abs < 1e-6,
            "column-0 dequant vs f32 layout max_abs={max_abs} (byte order wrong?)"
        );

        // Random-ish but deterministic activation.
        let x: Vec<f32> = (0..n_in).map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0).collect();
        let mut y_q = vec![0.0f32; n_out];
        let mut y_f = vec![0.0f32; n_out];
        let w = QuantMat::new(TensorType::Q5K, bytes.to_vec(), f32w, n_in, n_out);
        std::env::set_var("MILTON_Q8K", "1");
        matmul_ggml(&x, &w, 1, &mut y_q);
        std::env::set_var("MILTON_Q8K", "0");
        matmul_ggml(&x, &w, 1, &mut y_f);
        let mut dot = 0.0f64;
        let mut nq = 0.0f64;
        let mut nf = 0.0f64;
        let mut maxd = 0.0f32;
        for i in 0..n_out {
            dot += f64::from(y_q[i]) * f64::from(y_f[i]);
            nq += f64::from(y_q[i]) * f64::from(y_q[i]);
            nf += f64::from(y_f[i]) * f64::from(y_f[i]);
            maxd = maxd.max((y_q[i] - y_f[i]).abs());
        }
        let cos = dot / (nq.sqrt() * nf.sqrt());
        eprintln!("q5k vs f32 matmul cos={cos} max_abs={maxd} y_q0={} y_f0={}", y_q[0], y_f[0]);
        assert!(
            cos > 0.99,
            "Q8_K matmul drifted from dequant-f32: cos={cos} max_abs={maxd}"
        );
    }

    #[test]
    fn q6k_column0_matches_dequant_and_vec_dot_tracks_f32() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let gguf_path = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if !gguf_path.exists() {
            return;
        }
        let gguf = crate::gguf::GgufFile::open(&gguf_path).unwrap();
        let info = gguf
            .tensors
            .iter()
            .find(|t| t.tensor_type == TensorType::Q6K)
            .expect("Q4_K_M mix includes Q6_K");
        let name = info.name.clone();
        let n_in = info.dimensions[0] as usize;
        let n_out = info.dimensions[1] as usize;
        let bytes = gguf.tensor_bytes(info).unwrap();
        let f32w = gguf.dequantize_tensor(&name).unwrap();
        let cb = col_bytes(TensorType::Q6K, n_in);
        let col0 = &bytes[0..cb];
        let deq = crate::dequant::dequantize(TensorType::Q6K, col0, n_in, "col0").unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..n_in {
            max_abs = max_abs.max((deq[i] - f32w[i]).abs());
        }
        assert!(
            max_abs < 1e-6,
            "Q6_K {name} column-0 layout max_abs={max_abs}"
        );

        let x: Vec<f32> = (0..n_in).map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0).collect();
        let mut y_q = vec![0.0f32; n_out];
        let mut y_f = vec![0.0f32; n_out];
        let w = QuantMat::new(TensorType::Q6K, bytes.to_vec(), f32w, n_in, n_out);
        std::env::set_var("MILTON_Q8K", "1");
        matmul_ggml(&x, &w, 1, &mut y_q);
        std::env::set_var("MILTON_Q8K", "0");
        matmul_ggml(&x, &w, 1, &mut y_f);
        let mut dot = 0.0f64;
        let mut nq = 0.0f64;
        let mut nf = 0.0f64;
        let mut maxd = 0.0f32;
        for i in 0..n_out {
            dot += f64::from(y_q[i]) * f64::from(y_f[i]);
            nq += f64::from(y_q[i]) * f64::from(y_q[i]);
            nf += f64::from(y_f[i]) * f64::from(y_f[i]);
            maxd = maxd.max((y_q[i] - y_f[i]).abs());
        }
        let cos = dot / (nq.sqrt() * nf.sqrt());
        eprintln!(
            "q6k {name} vs f32 matmul cos={cos} max_abs={maxd} y_q0={} y_f0={}",
            y_q[0], y_f[0]
        );
        assert!(
            cos > 0.99,
            "Q6_K AVX2 matmul drifted from dequant-f32: cos={cos} max_abs={maxd}"
        );
    }

    #[test]
    fn q4k_column0_matches_dequant() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let gguf_path = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if !gguf_path.exists() {
            return;
        }
        let gguf = crate::gguf::GgufFile::open(&gguf_path).unwrap();
        let info = gguf.tensor("blk.0.attn_output.weight").unwrap();
        let n_in = info.dimensions[0] as usize;
        let n_out = info.dimensions[1] as usize;
        let bytes = gguf.tensor_bytes(info).unwrap();
        let f32w = gguf.dequantize_tensor("blk.0.attn_output.weight").unwrap();
        let cb = col_bytes(TensorType::Q4K, n_in);
        let col0 = &bytes[0..cb];
        let deq = crate::dequant::dequantize(TensorType::Q4K, col0, n_in, "col0").unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..n_in {
            max_abs = max_abs.max((deq[i] - f32w[i]).abs());
        }
        assert!(max_abs < 1e-6, "Q4_K column-0 layout max_abs={max_abs}");
        let x: Vec<f32> = (0..n_in).map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0).collect();
        let mut y_q = vec![0.0f32; n_out];
        let mut y_f = vec![0.0f32; n_out];
        let w = QuantMat::new(TensorType::Q4K, bytes.to_vec(), f32w, n_in, n_out);
        assert!(w.q4k_8x8.is_some(), "Q4_K n_out={n_out} should REPACK 8x8");
        std::env::set_var("MILTON_Q8K", "1");
        std::env::set_var("MILTON_REPACK", "1");
        matmul_ggml(&x, &w, 1, &mut y_q);
        std::env::set_var("MILTON_REPACK", "0");
        let mut y_generic = vec![0.0f32; n_out];
        matmul_ggml(&x, &w, 1, &mut y_generic);
        std::env::set_var("MILTON_Q8K", "0");
        matmul_ggml(&x, &w, 1, &mut y_f);
        let mut dot = 0.0f64;
        let mut nq = 0.0f64;
        let mut nf = 0.0f64;
        let mut maxd = 0.0f32;
        for i in 0..n_out {
            dot += f64::from(y_q[i]) * f64::from(y_f[i]);
            nq += f64::from(y_q[i]) * f64::from(y_q[i]);
            nf += f64::from(y_f[i]) * f64::from(y_f[i]);
            maxd = maxd.max((y_q[i] - y_f[i]).abs());
        }
        let cos = dot / (nq.sqrt() * nf.sqrt());
        let mut d8 = 0.0f64;
        let mut n8 = 0.0f64;
        let mut ng = 0.0f64;
        let mut max8 = 0.0f32;
        for i in 0..n_out {
            d8 += f64::from(y_q[i]) * f64::from(y_generic[i]);
            n8 += f64::from(y_q[i]) * f64::from(y_q[i]);
            ng += f64::from(y_generic[i]) * f64::from(y_generic[i]);
            max8 = max8.max((y_q[i] - y_generic[i]).abs());
        }
        let cos8 = d8 / (n8.sqrt() * ng.sqrt());
        eprintln!(
            "q4k 8x8 vs f32 cos={cos} max_abs={maxd} y8={} yf={}; 8x8 vs generic vec_dot cos={cos8} max_abs={max8}",
            y_q[0], y_f[0]
        );
        assert!(cos > 0.99, "Q4_K 8x8 matmul drifted from f32: cos={cos} max_abs={maxd}");
        assert!(
            cos8 > 0.999,
            "Q4_K 8x8 drifted from generic vec_dot (repack bug?): cos={cos8} max_abs={max8}"
        );
    }
}
