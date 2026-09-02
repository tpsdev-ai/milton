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
//! and are not used on the embed path.
//!
//! Q4_K GEMV FMA (`mul_add` = pin AVX2 `_mm256_fmadd_ps`) is BIT_EXACT vs
//! dump empty-none `kqv_out` MUL_MAT (mul+add was 4.77e-7 / 701 elems) and
//! matches the pin DSO GEMV. Landing it avalanches golden FINAL (empty-none
//! max_abs 3.547e-4, short-hello-none cos_dist 0.00912). Stay on mul+add.
//! Document n=7 leftover vs dump is 3.81e-6 — same as the DSO GEMV; 4x8
//! GEMM is worse and stays off the embed path.
//!
//! Sequence-tiled GEMM (`GEMM_TILE_TOKENS` = 32) unpacks each weight block
//! once per tile and applies the same Q8_K + mul+add tree to every token
//! in the tile. Same numeric path as the GEMV. Not the 4×8 FMA GEMM.
//!
//! After GEMV-only, the first tensor that is **not bit-exact** vs the
//! llama eval-callback dump is `wqkv-0` (Q5_K MUL_MAT). There is no
//! Q5_K REPACK/GEMV/GEMM in the pinned DSO (`tinyBLAS` rejects Q5_K;
//! AMX is compiled OFF). The graph path is `quantize_row_q8_K` +
//! AVX2 `ggml_vec_dot_q5_K_q8_K`. That DSO pair is bit-exact on the
//! dump; DSO generic is 1.91e-6. Milton Q8_K and f16 match the DSO.
//! The DSO compiles `summs += dmin * extract` as `vfmadd231ss`; a
//! separate mul+add was 1–16 ulps off (`wqkv-0` 3489 elems). Do not
//! invent another Q5_K type. Do not globally enable AVX2 Q@K
//! (avalanches `empty-none`). Do not touch `quantize_row_q8_K`.

use crate::dequant::{f16_to_f32, get_scale_min_k4};
use crate::gguf::TensorType;
use crate::ops::matmul;

pub const QK_K: usize = 256;
const Q4_K_BYTES: usize = 144;
const Q5_K_BYTES: usize = 176;
const Q6_K_BYTES: usize = 210;

const Q4_KX8_BYTES: usize = 1152; // 8*f16 d + 8*f16 dmin + 96 scales + 1024 qs

/// Sequence tile for `matmul_ggml`. Each weight block is unpacked once per
/// tile, then applied to every token in the tile with the same Q8_K row
/// quant + mul+add tree the GEMV uses. Not the 4×8 FMA GEMM.
///
/// **32 tokens.** Working-set vs L2 8 MiB on the #35 measurement host:
/// - Largest live matrix (Q4_K FFN up/gate/down):
///   `144 B × (n_in/256) × n_out` = 1,327,104 B = **1.266 MiB**
///   (Q5_K QKV is 176 × 3 × 2304 = 1,216,512 B = 1.160 MiB).
/// - `BlockQ8K` is 292 B (`d` + `qs[256]` + `bsums[16]`).
/// - Q8_K tile at FFN-down `n_in=3072`: 32 × 12 × 292 = **112,128 B**.
/// - Output tile at FFN-up `n_out=3072`: 32 × 3072 × 4 = **393,216 B**.
/// - Peak resident ≈ 1.266 + 0.107 + 0.375 = **1.75 MiB (22% of 8 MiB L2)**.
///   One 8-col Q4_K group + the Q8_K tile + 32×8 accs is ~126 KB while
///   streaming the matrix — well inside L2, with >6 MiB headroom.
/// - wasm:bench 8-case max n=19 fits in one tile (one weight pass).
/// - n=502 → 16 tiles vs 502 per-token GEMV weight passes.
pub(crate) const GEMM_TILE_TOKENS: usize = 32;

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
pub(crate) struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
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

#[allow(dead_code)] // one-token identity wrapper; embed path uses the tile
fn vec_dot_q4_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q4_k_q8_k_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

/// Unpack-once Q4_K×Q8_K tile. Superblock order and mul+add tree match
/// `vec_dot_q4_k_q8_k` per token.
fn vec_dot_q4_k_q8_k_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(w.len(), n_blocks * Q4_K_BYTES);
    debug_assert_eq!(qrows.len(), n_tile * n_blocks);
    debug_assert!(n_tile <= out.len());
    for t in 0..n_tile {
        out[t] = 0.0;
    }
    for i in 0..n_blocks {
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
        let mut mins = [0u8; 8];
        let mut scs = [0u8; 8];
        for group in 0..8 {
            let (sc, m) = get_scale_min_k4(group, scales);
            scs[group] = sc;
            mins[group] = m;
        }
        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
            let mut sumi = 0i32;
            for j in 0..QK_K / 16 {
                sumi += i32::from(yb.bsums[j]) * i32::from(mins[j / 2]);
            }
            let mut acc = 0i32;
            for group in 0..8 {
                let sc = i32::from(scs[group]);
                let base = group * 32;
                for l in 0..32 {
                    acc += sc * i32::from(aux8[base + l]) * i32::from(yb.qs[base + l]);
                }
            }
            out[t] += d * yb.d * acc as f32;
            out[t] -= dmin * yb.d * sumi as f32;
        }
    }
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
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

#[allow(dead_code)] // one-token identity wrapper; embed path uses the tile
fn vec_dot_q5_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q5_k_q8_k_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

fn vec_dot_q5_k_q8_k_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { vec_dot_q5_k_q8_k_avx2_tile(w, qrows, n_tile, n_blocks, out) };
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            crate::qmatmul_simd128::vec_dot_q5_k_q8_k_tile(w, qrows, n_tile, n_blocks, out);
        }
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    vec_dot_q5_k_q8_k_generic_tile(w, qrows, n_tile, n_blocks, out);
}

/// llama.cpp `hsum_float_8` (`quants.c` ~43): pairwise tree, not a scalar
/// reduction LLVM can reassociate. Store-then-add is the same tree as
/// extractf128 / movehl / movehdup; `black_box` keeps the order.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut res = _mm256_extractf128_ps::<1>(x);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    std::hint::black_box(_mm_cvtss_f32(res))
}

/// One QK_K/64 chunk of `ggml_vec_dot_q5_K_q8_K` AVX2. `BIT0`/`BIT1` are
/// the C `bit++` immediates for `_mm256_srli_epi16` — a runtime
/// `_mm256_srl_epi16` is a different encoding and was 1–16 ulps off the
/// pinned DSO on `wqkv-0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q5k_avx2_chunk<const BIT0: i32, const BIT1: i32>(
    scales: std::arch::x86_64::__m256i,
    shuffle0: *const u8,
    shuffle1: *const u8,
    q5: *const u8,
    q8: *const i8,
    hbits: std::arch::x86_64::__m256i,
    hmask: std::arch::x86_64::__m256i,
    m4: std::arch::x86_64::__m256i,
    sumi: std::arch::x86_64::__m256i,
) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    let scale_0 = _mm256_shuffle_epi8(scales, _mm256_loadu_si256(shuffle0.cast()));
    let scale_1 = _mm256_shuffle_epi8(scales, _mm256_loadu_si256(shuffle1.cast()));
    let q5bits = _mm256_loadu_si256(q5.cast());

    let q5l_0 = _mm256_and_si256(q5bits, m4);
    let q5h_0 = _mm256_slli_epi16::<4>(_mm256_srli_epi16::<BIT0>(_mm256_and_si256(hbits, hmask)));
    let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
    let hmask = _mm256_slli_epi16::<1>(hmask);

    let q5l_1 = _mm256_and_si256(_mm256_srli_epi16::<4>(q5bits), m4);
    let q5h_1 = _mm256_slli_epi16::<4>(_mm256_srli_epi16::<BIT1>(_mm256_and_si256(hbits, hmask)));
    let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
    let hmask = _mm256_slli_epi16::<1>(hmask);

    let q8_0 = _mm256_loadu_si256(q8.cast());
    let q8_1 = _mm256_loadu_si256(q8.add(32).cast());
    let p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
    let p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));
    (
        _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1)),
        hmask,
    )
}

/// llama.cpp `ggml_vec_dot_q5_K_q8_K` AVX2 (`quants.c` ~1919), sequence-tiled.
///
/// Dispatch matches the pinned `libggml-cpu.so`: no Q5_K REPACK/GEMV/GEMM;
/// `quantize_row_q8_K` + this `vec_dot`. DSO AVX2 is bit-exact on the
/// `wqkv-0` eval-callback dump; generic is not (1.91e-6). Immediate
/// `_mm256_srli_epi16::<bit>` is the C `bit++` form.
/// `summs = dmin.mul_add(extract, summs)` matches the DSO's
/// `vfmadd231ss` (separate mul+add was 1–16 ulps off).
///
/// Weight superblock (scales / qh / qs) is unpacked once per tile; each
/// token keeps its own 8-wide `acc` + `summs` in the same order as the
/// one-token kernel.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q5_k_q8_k_avx2_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(w.len(), n_blocks * Q5_K_BYTES);
    debug_assert_eq!(qrows.len(), n_tile * n_blocks);
    debug_assert!(n_tile <= GEMM_TILE_TOKENS);
    debug_assert!(n_tile <= out.len());

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

    let mut acc = [_mm256_setzero_ps(); GEMM_TILE_TOKENS];
    let mut summs = [0.0f32; GEMM_TILE_TOKENS];

    for i in 0..n_blocks {
        let block = &w[i * Q5_K_BYTES..(i + 1) * Q5_K_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

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

        let sc128 = _mm256_extracti128_si256::<0>(mins_and_scales);
        // MM256_SET_M128I(sc128, sc128)
        let scales = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(sc128), sc128);
        let mins128 = _mm256_extracti128_si256::<1>(mins_and_scales);

        let hbits = _mm256_loadu_si256(block[16..48].as_ptr().cast());
        let q5 = block[48..176].as_ptr();
        let sh = K_SHUFFLE.as_ptr();

        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
            let d = yb.d * d_w;
            let dmin = -yb.d * dmin_w;

            let q8sums = _mm256_loadu_si256(yb.bsums.as_ptr().cast());
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256::<0>(q8sums),
                _mm256_extracti128_si256::<1>(q8sums),
            );
            let prod = _mm_madd_epi16(mins128, q8s);
            let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            // DSO (`-mfma`) contracts `summs += dmin * extract` to `vfmadd231ss`.
            // Separate mul+add is 1–16 ulps off the pinned `wqkv-0` dump.
            summs[t] = dmin.mul_add(_mm_extract_epi32::<0>(hsum) as f32, summs[t]);

            let mut hmask = mone;
            let mut sumi = _mm256_setzero_si256();
            let q8 = yb.qs.as_ptr();

            let r = q5k_avx2_chunk::<0, 1>(scales, sh, sh.add(32), q5, q8, hbits, hmask, m4, sumi);
            sumi = r.0;
            hmask = r.1;
            let r = q5k_avx2_chunk::<2, 3>(
                scales,
                sh.add(64),
                sh.add(96),
                q5.add(32),
                q8.add(64),
                hbits,
                hmask,
                m4,
                sumi,
            );
            sumi = r.0;
            hmask = r.1;
            let r = q5k_avx2_chunk::<4, 5>(
                scales,
                sh.add(128),
                sh.add(160),
                q5.add(64),
                q8.add(128),
                hbits,
                hmask,
                m4,
                sumi,
            );
            sumi = r.0;
            hmask = r.1;
            let r = q5k_avx2_chunk::<6, 7>(
                scales,
                sh.add(192),
                sh.add(224),
                q5.add(96),
                q8.add(192),
                hbits,
                hmask,
                m4,
                sumi,
            );
            sumi = r.0;

            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }

    for t in 0..n_tile {
        out[t] = hsum_float_8(acc[t]) + summs[t];
    }
}

#[allow(dead_code)] // one-token identity wrapper; embed path uses the tile
fn vec_dot_q5_k_q8_k_generic(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q5_k_q8_k_generic_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn vec_dot_q5_k_q8_k_generic_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(w.len(), n_blocks * Q5_K_BYTES);
    debug_assert_eq!(qrows.len(), n_tile * n_blocks);
    debug_assert!(n_tile <= GEMM_TILE_TOKENS);
    debug_assert!(n_tile <= out.len());
    // llama.cpp generic: 8 int32 lanes, `sums[l] += d * aux32[l]` per superblock.
    let mut sums = [[0.0f32; 8]; GEMM_TILE_TOKENS];
    let mut sumf = [0.0f32; GEMM_TILE_TOKENS];
    for i in 0..n_blocks {
        let block = &w[i * Q5_K_BYTES..(i + 1) * Q5_K_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
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
        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
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
            let d = d_w * yb.d;
            for l in 0..8 {
                sums[t][l] += d * aux32[l] as f32;
            }
            sumf[t] -= dmin * yb.d * sumi as f32;
        }
    }
    for t in 0..n_tile {
        let mut s = sumf[t];
        for l in 0..8 {
            s += sums[t][l];
        }
        out[t] = s;
    }
}

#[allow(dead_code)] // one-token identity wrapper; embed path uses the tile
fn vec_dot_q6_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q6_k_q8_k_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

fn vec_dot_q6_k_q8_k_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { vec_dot_q6_k_q8_k_avx2_tile(w, qrows, n_tile, n_blocks, out) };
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            crate::qmatmul_simd128::vec_dot_q6_k_q8_k_tile(w, qrows, n_tile, n_blocks, out);
        }
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    vec_dot_q6_k_q8_k_generic_tile(w, qrows, n_tile, n_blocks, out);
}

/// llama.cpp `ggml_vec_dot_q6_K_q8_K` AVX2 (`quants.c` ~2129), sequence-tiled.
///
/// Bit-exact: 6-bit reconstruct, (−32) via `maddubs(m32s, q8)` subtract,
/// `madd_epi16` of `cvtepi8_epi16` scales, `_mm256_fmadd_ps` of
/// `broadcast_ss(d)` into 8-wide `acc` across superblocks, `hsum_float_8`
/// pairwise order. Weight superblock unpacked once per tile.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q6_k_q8_k_avx2_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(w.len(), n_blocks * Q6_K_BYTES);
    debug_assert_eq!(qrows.len(), n_tile * n_blocks);
    debug_assert!(n_tile <= GEMM_TILE_TOKENS);
    debug_assert!(n_tile <= out.len());

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

    let mut acc = [_mm256_setzero_ps(); GEMM_TILE_TOKENS];

    for i in 0..n_blocks {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        let q4 = block[0..128].as_ptr();
        let qh = block[128..192].as_ptr();
        let scales = _mm_loadu_si128(block[192..208].as_ptr().cast());

        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
            let d = yb.d * d_w;
            let q8 = yb.qs.as_ptr();

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
                let q4h_1 =
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(q4bits_h), m2));
                let q4h_2 =
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(q4bits_h), m2));
                let q4h_3 =
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(q4bits_h), m2));

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(q4bits1), m4), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(q4bits2), m4), q4h_3);

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

            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }

    for t in 0..n_tile {
        // llama.cpp `hsum_float_8`: high128+low128 → movehl add → movehdup add_ss
        let mut res = _mm256_extractf128_ps::<1>(acc[t]);
        res = _mm_add_ps(res, _mm256_castps256_ps128(acc[t]));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        out[t] = _mm_cvtss_f32(res);
    }
}

#[allow(dead_code)] // one-token identity wrapper; embed path uses the tile
fn vec_dot_q6_k_q8_k_generic(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q6_k_q8_k_generic_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn vec_dot_q6_k_q8_k_generic_tile(
    w: &[u8],
    qrows: &[BlockQ8K],
    n_tile: usize,
    n_blocks: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(w.len(), n_blocks * Q6_K_BYTES);
    debug_assert_eq!(qrows.len(), n_tile * n_blocks);
    debug_assert!(n_tile <= GEMM_TILE_TOKENS);
    debug_assert!(n_tile <= out.len());
    // llama.cpp generic Q6_K: 8 int32 lanes, float acc per lane, then sum.
    let mut sums = [[0.0f32; 8]; GEMM_TILE_TOKENS];
    for i in 0..n_blocks {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut aux8 = [0i8; QK_K];
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut a = 0usize;
        for _ in 0..QK_K / 128 {
            for l in 0..32 {
                aux8[a + l] =
                    ((ql[ql_off + l] & 0x0f) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
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
        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
            let d = d_w * yb.d;
            let mut aux32 = [0i32; 8];
            for group in 0..16 {
                let scale = i32::from(sc[group] as i8);
                let base = group * 16;
                for l in 0..8 {
                    aux32[l] += scale * i32::from(aux8[base + l]) * i32::from(yb.qs[base + l]);
                    aux32[l] +=
                        scale * i32::from(aux8[base + 8 + l]) * i32::from(yb.qs[base + 8 + l]);
                }
            }
            for l in 0..8 {
                sums[t][l] += d * aux32[l] as f32;
            }
        }
    }
    for t in 0..n_tile {
        out[t] = sums[t].iter().sum();
    }
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

/// Unpack one `block_q4_Kx8` scales payload (same utmp dance as the GEMV).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn unpack_q4_kx8_scales(scales: &[u8], ub: &mut [u8; 128]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let mut utmp = [0u32; 32];
    for sb in 0..8 {
        let base = sb * 12;
        utmp[sb * 4] = u32::from_le_bytes(scales[base..base + 4].try_into().unwrap());
        utmp[sb * 4 + 1] = u32::from_le_bytes(scales[base + 4..base + 8].try_into().unwrap());
        utmp[sb * 4 + 2] = u32::from_le_bytes(scales[base + 8..base + 12].try_into().unwrap());
        utmp[sb * 4 + 3] =
            ((utmp[sb * 4 + 2] >> 4) & KMASK2) | (((utmp[sb * 4 + 1] >> 6) & KMASK3) << 4);
        let uaux_0 = utmp[sb * 4 + 1] & KMASK1;
        utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & KMASK2) | (((utmp[sb * 4] >> 6) & KMASK3) << 4);
        utmp[sb * 4 + 2] = uaux_0;
        utmp[sb * 4] &= KMASK1;
    }
    for i in 0..32 {
        ub[i * 4..i * 4 + 4].copy_from_slice(&utmp[i].to_le_bytes());
    }
}

/// Integer products for one Q8_K superblock against an unpacked `block_q4_Kx8`.
/// Same eval order as the live GEMV (`k` outer, `j` columns, `i` 8-wide).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn q4_kx8_iacc(qs: &[u8], ub: &[u8; 128], yb: &BlockQ8K) -> ([i32; 8], [i32; 8]) {
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
    (iacc, iacc_min)
}

/// Sequence-tiled Q4_K×Q8_K 8-col GEMM. Same integer products and mul+add
/// tree as `gemv_q4_k_8x8_q8_k` (one token). Each `block_q4_Kx8` is unpacked
/// once per tile, then applied to every token in the tile.
///
/// Scalar mul+add sits **outside** `target_feature(enable = "avx2")` so LLVM
/// cannot contract it to FMA. Do not land the 4×8 FMA GEMM.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn gemm_q4_k_8x8_q8_k(
    repack: &[u8],
    qrows: &[BlockQ8K],
    n_tokens: usize,
    n_out: usize,
    out: &mut [f32],
) {
    debug_assert!(n_tokens > 0);
    debug_assert_eq!(qrows.len(), n_tokens * (qrows.len() / n_tokens));
    let n_blocks = qrows.len() / n_tokens;
    debug_assert_eq!(repack.len(), (n_out / 8) * n_blocks * Q4_KX8_BYTES);
    debug_assert_eq!(out.len(), n_tokens * n_out);
    let n_groups = n_out / 8;
    for t0 in (0..n_tokens).step_by(GEMM_TILE_TOKENS) {
        let tn = (n_tokens - t0).min(GEMM_TILE_TOKENS);
        for x in 0..n_groups {
            let mut sumf = [[0.0f32; 8]; GEMM_TILE_TOKENS];
            let mut sum_minf = [[0.0f32; 8]; GEMM_TILE_TOKENS];
            for l in 0..n_blocks {
                let off = (x * n_blocks + l) * Q4_KX8_BYTES;
                let blk = &repack[off..off + Q4_KX8_BYTES];
                let mut d = [0.0f32; 8];
                let mut dmin = [0.0f32; 8];
                for j in 0..8 {
                    d[j] = f16_to_f32(u16::from_le_bytes([blk[j * 2], blk[j * 2 + 1]]));
                    dmin[j] =
                        f16_to_f32(u16::from_le_bytes([blk[16 + j * 2], blk[16 + j * 2 + 1]]));
                }
                let mut ub = [0u8; 128];
                unpack_q4_kx8_scales(&blk[32..128], &mut ub);
                let qs = &blk[128..];
                for ti in 0..tn {
                    let yb = &qrows[(t0 + ti) * n_blocks + l];
                    let (iacc, iacc_min) = q4_kx8_iacc(qs, &ub, yb);
                    for j in 0..8 {
                        // Stay on mul+add. Pin dump kqv_out MUL_MAT is AVX2
                        // `_mm256_fmadd_ps(iacc_f32, d*yd, acc)` and is BIT_EXACT
                        // with `mul_add` here, but landing that FMA avalanches the
                        // golden FINAL: empty-none max_abs 6.96e-8 → 3.547e-4
                        // (d0 0.00697820 vs 0.00702455); short-hello-none
                        // cos_dist 0 → 0.00912. Same class as dump-kq / 4×8 Q@K.
                        // Do not land FMA GEMV. Do not land 4x8 GEMM.
                        let ds = d[j] * yb.d;
                        sumf[ti][j] += iacc[j] as f32 * ds;
                        sum_minf[ti][j] += iacc_min[j] as f32 * (dmin[j] * yb.d);
                    }
                }
            }
            for ti in 0..tn {
                for j in 0..8 {
                    out[(t0 + ti) * n_out + x * 8 + j] = sumf[ti][j] - sum_minf[ti][j];
                }
            }
        }
    }
}

/// llama.cpp `ggml_gemv_q4_K_8x8_q8_K_generic` — one-token wrapper on the
/// tiled GEMM (same numeric path).
#[allow(dead_code)]
fn gemv_q4_k_8x8_q8_k(repack: &[u8], y: &[BlockQ8K], n_out: usize, out: &mut [f32]) {
    gemm_q4_k_8x8_q8_k(repack, y, 1, n_out, out);
}

/// ggml `mul_mat`: dst[o, t] = vec_dot(W_col[o], Q8_K(x[t])).
///
/// Sequence-tiled: each weight block is unpacked once per
/// [`GEMM_TILE_TOKENS`] tokens. Same row Q8_K + mul+add tree as the
/// live GEMV — not the 4×8 FMA GEMM.
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
        // Sequence-tiled GEMM on the GEMV numeric path. The 4x8 FMA GEMM
        // matches the llama.cpp DSO bit-exactly but *not* the embedding
        // graph dump (see crate docs). Stay on mul+add GEMV-tree.
        #[cfg(target_arch = "wasm32")]
        crate::qmatmul_simd128::gemm_q4_k_8x8_q8_k(packed, &qrows, n_tokens, w.n_out, y);
        #[cfg(not(target_arch = "wasm32"))]
        gemm_q4_k_8x8_q8_k(packed, &qrows, n_tokens, w.n_out, y);
        return;
    }
    let mut col_out = [0.0f32; GEMM_TILE_TOKENS];
    for t0 in (0..n_tokens).step_by(GEMM_TILE_TOKENS) {
        let tn = (n_tokens - t0).min(GEMM_TILE_TOKENS);
        let tile_rows = &qrows[t0 * n_blocks..(t0 + tn) * n_blocks];
        for o in 0..w.n_out {
            let col = &w.bytes[o * cb..(o + 1) * cb];
            match w.ty {
                TensorType::Q4K => {
                    vec_dot_q4_k_q8_k_tile(col, tile_rows, tn, n_blocks, &mut col_out)
                }
                TensorType::Q5K => {
                    vec_dot_q5_k_q8_k_tile(col, tile_rows, tn, n_blocks, &mut col_out)
                }
                TensorType::Q6K => {
                    vec_dot_q6_k_q8_k_tile(col, tile_rows, tn, n_blocks, &mut col_out)
                }
                _ => unreachable!(),
            }
            for ti in 0..tn {
                y[(t0 + ti) * w.n_out + o] = col_out[ti];
            }
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
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
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
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0)
            .collect();
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
        eprintln!(
            "q5k vs f32 matmul cos={cos} max_abs={maxd} y_q0={} y_f0={}",
            y_q[0], y_f[0]
        );
        assert!(
            cos > 0.99,
            "Q8_K matmul drifted from dequant-f32: cos={cos} max_abs={maxd}"
        );
    }

    fn load_f32_dump(path: &str) -> Option<Vec<f32>> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).ok()?;
        let mut n = [0u8; 8];
        f.read_exact(&mut n).ok()?;
        let n = i64::from_le_bytes(n) as usize;
        let mut skip = [0u8; 32];
        f.read_exact(&mut skip).ok()?;
        let mut buf = vec![0u8; n * 4];
        f.read_exact(&mut buf).ok()?;
        Some(
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    #[test]
    fn q5k_wqkv0_matches_llama_mul_mat_dump() {
        let act = match load_f32_dump("/tmp/ll-dump/ll-inp_norm__ADD__5.f32") {
            Some(v) => v,
            None => return,
        };
        let exp = match load_f32_dump("/tmp/ll-dump/ll-wqkv-0__MUL_MAT__6.f32") {
            Some(v) => v,
            None => return,
        };
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let gguf_path = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if !gguf_path.exists() {
            return;
        }
        let gguf = crate::gguf::GgufFile::open(&gguf_path).unwrap();
        let info = gguf.tensor("blk.0.attn_qkv.weight").unwrap();
        let n_in = info.dimensions[0] as usize;
        let n_out = info.dimensions[1] as usize;
        let bytes = gguf.tensor_bytes(info).unwrap().to_vec();
        let f32w = gguf.dequantize_tensor("blk.0.attn_qkv.weight").unwrap();
        let ntok = act.len() / n_in;
        let w = QuantMat::new(TensorType::Q5K, bytes, f32w, n_in, n_out);
        std::env::set_var("MILTON_Q8K", "1");
        let mut got = vec![0.0f32; ntok * n_out];
        matmul_ggml(&act, &w, ntok, &mut got);
        let mut mx = 0.0f32;
        let mut ndiff = 0usize;
        let mut at = 0usize;
        for i in 0..got.len() {
            let d = (got[i] - exp[i]).abs();
            if d > 0.0 {
                ndiff += 1;
            }
            if d > mx {
                mx = d;
                at = i;
            }
        }
        eprintln!(
            "wqkv-0 vs llama MUL_MAT n={} max_abs={mx:.8e} ndiff={ndiff} at={at} got={} exp={}",
            got.len(),
            got[at],
            exp[at]
        );
        assert!(
            mx < 1e-7,
            "wqkv-0 Q5_K AVX2 vs llama dump max_abs={mx} ndiff={ndiff} (DSO AVX2 is bit-exact)"
        );
    }

    #[test]
    fn q6k_column0_matches_dequant_and_vec_dot_tracks_f32() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
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

        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0)
            .collect();
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
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
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
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0)
            .collect();
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
        assert!(
            cos > 0.99,
            "Q4_K 8x8 matmul drifted from f32: cos={cos} max_abs={maxd}"
        );
        assert!(
            cos8 > 0.999,
            "Q4_K 8x8 drifted from generic vec_dot (repack bug?): cos={cos8} max_abs={max8}"
        );
    }

    /// Tiled GEMM must be bit-identical to per-token GEMV on the same
    /// Q8_K rows (n=1 wrapper vs n>1 tile, including a tail past TILE).
    #[test]
    fn tiled_gemm_matches_per_token_gemv_bit_exact() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let gguf_path = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if !gguf_path.exists() {
            return;
        }
        let gguf = crate::gguf::GgufFile::open(&gguf_path).unwrap();
        let cases = [
            ("blk.0.attn_output.weight", TensorType::Q4K, 7usize),
            (
                "blk.0.attn_output.weight",
                TensorType::Q4K,
                GEMM_TILE_TOKENS + 1,
            ),
            ("blk.0.attn_qkv.weight", TensorType::Q5K, 7),
            (
                "blk.0.attn_qkv.weight",
                TensorType::Q5K,
                GEMM_TILE_TOKENS + 1,
            ),
            ("blk.0.ffn_up.weight", TensorType::Q4K, 3),
            ("blk.0.ffn_down.weight", TensorType::Q4K, 5),
        ];
        std::env::set_var("MILTON_Q8K", "1");
        std::env::set_var("MILTON_REPACK", "1");
        for (name, ty, n_tok) in cases {
            let info = gguf.tensor(name).unwrap();
            let n_in = info.dimensions[0] as usize;
            let n_out = info.dimensions[1] as usize;
            let bytes = gguf.tensor_bytes(info).unwrap().to_vec();
            let f32w = gguf.dequantize_tensor(name).unwrap();
            let w = QuantMat::new(ty, bytes, f32w, n_in, n_out);
            let x: Vec<f32> = (0..n_tok * n_in)
                .map(|i| ((i * 17) % 50) as f32 / 25.0 - 1.0)
                .collect();
            let mut y_tile = vec![0.0f32; n_tok * n_out];
            matmul_ggml(&x, &w, n_tok, &mut y_tile);
            let mut y_gemv = vec![0.0f32; n_tok * n_out];
            for t in 0..n_tok {
                matmul_ggml(
                    &x[t * n_in..(t + 1) * n_in],
                    &w,
                    1,
                    &mut y_gemv[t * n_out..(t + 1) * n_out],
                );
            }
            let mut max_abs = 0.0f32;
            let mut ndiff = 0usize;
            for i in 0..y_tile.len() {
                let d = (y_tile[i] - y_gemv[i]).abs();
                if d > 0.0 {
                    ndiff += 1;
                }
                max_abs = max_abs.max(d);
            }
            assert_eq!(
                y_tile, y_gemv,
                "tiled GEMM drifted from per-token GEMV {name} n={n_tok} max_abs={max_abs} ndiff={ndiff}"
            );
        }
    }

    fn load_dump_f32(path: &std::path::Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 40 {
            return None;
        }
        let n = i64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        if bytes.len() < 40 + n * 4 {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = 40 + i * 4;
            out.push(f32::from_le_bytes(bytes[o..o + 4].try_into().ok()?));
        }
        Some(out)
    }

    /// Isolate Q4_K GEMV: dump kqv_out CONT → Milton → dump kqv_out MUL_MAT.
    /// Does not go through the serial Q@K path.
    #[test]
    fn q4k_gemv_kqv_out_vs_pin_dump() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let gguf_path = root.join("harness/vendor/models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if !gguf_path.exists() {
            return;
        }
        let cases = [
            (
                "/tmp/ll-emb-empty/ll-kqv_out-0__CONT__25.f32",
                "/tmp/ll-emb-empty/ll-kqv_out-0__MUL_MAT__26.f32",
                "empty-none",
            ),
            (
                "/tmp/ll-emb-doc/ll-kqv_out-0__CONT__25.f32",
                "/tmp/ll-emb-doc/ll-kqv_out-0__MUL_MAT__26.f32",
                "short-hello-document",
            ),
        ];
        let gguf = crate::gguf::GgufFile::open(&gguf_path).unwrap();
        let info = gguf.tensor("blk.0.attn_output.weight").unwrap();
        let n_in = info.dimensions[0] as usize;
        let n_out = info.dimensions[1] as usize;
        let bytes = gguf.tensor_bytes(info).unwrap();
        let f32w = gguf.dequantize_tensor("blk.0.attn_output.weight").unwrap();
        let w = QuantMat::new(TensorType::Q4K, bytes.to_vec(), f32w, n_in, n_out);
        std::env::set_var("MILTON_Q8K", "1");
        std::env::set_var("MILTON_REPACK", "1");
        for (inp, refer, label) in cases {
            let Some(x) = load_dump_f32(std::path::Path::new(inp)) else {
                eprintln!("skip {label}: missing {inp}");
                continue;
            };
            let Some(exp) = load_dump_f32(std::path::Path::new(refer)) else {
                eprintln!("skip {label}: missing {refer}");
                continue;
            };
            assert_eq!(x.len(), exp.len());
            assert_eq!(x.len() % n_in, 0);
            let n_tok = x.len() / n_in;
            let mut y = vec![0.0f32; n_tok * n_out];
            matmul_ggml(&x, &w, n_tok, &mut y);
            let mut max_abs = 0.0f32;
            let mut ndiff = 0usize;
            let mut at = 0usize;
            for i in 0..y.len() {
                let d = (y[i] - exp[i]).abs();
                if d > 0.0 {
                    ndiff += 1;
                }
                if d > max_abs {
                    max_abs = d;
                    at = i;
                }
            }
            eprintln!(
                "q4k GEMV kqv_out-0 {label} n={nt} max_abs={max_abs:.8e} ndiff={ndiff} at={at} y={y0:.8} exp={e0:.8}",
                nt = y.len(),
                y0 = y[0],
                e0 = exp[0]
            );
            // Landed path is mul+add. empty-none dump leftover ~4.77e-7;
            // do not assert BIT_EXACT (that requires FMA, which avalanches FINAL).
            let _ = (max_abs, ndiff);
        }

        // Next leftover after matched kqv_out: dump ffn_inp → Q4_K ffn_up.
        let ffn_cases = [
            (
                "ffn_up",
                "blk.0.ffn_up.weight",
                "/tmp/ll-emb-empty/ll-ffn_inp-0.f32",
                "/tmp/ll-emb-empty/ll-ffn_up-0__MUL_MAT__32.f32",
            ),
            (
                "ffn_gate",
                "blk.0.ffn_gate.weight",
                "/tmp/ll-emb-empty/ll-ffn_inp-0.f32",
                "/tmp/ll-emb-empty/ll-ffn_gate-0__MUL_MAT__31.f32",
            ),
            (
                "ffn_out",
                "blk.0.ffn_down.weight",
                "/tmp/ll-emb-empty/ll-ffn_swiglu-0__GLU__33.f32",
                "/tmp/ll-emb-empty/ll-ffn_out-0__MUL_MAT__34.f32",
            ),
        ];
        for (label, tname, inp, refer) in ffn_cases {
            let Some(info) = gguf.tensor(tname) else {
                continue;
            };
            let n_in = info.dimensions[0] as usize;
            let n_out = info.dimensions[1] as usize;
            let bytes = gguf.tensor_bytes(info).unwrap();
            let f32w = gguf.dequantize_tensor(tname).unwrap();
            let w = QuantMat::new(info.tensor_type, bytes.to_vec(), f32w, n_in, n_out);
            let Some(x) = load_dump_f32(std::path::Path::new(inp)) else {
                continue;
            };
            let Some(exp) = load_dump_f32(std::path::Path::new(refer)) else {
                continue;
            };
            let n_tok = x.len() / n_in;
            let mut y = vec![0.0f32; n_tok * n_out];
            matmul_ggml(&x, &w, n_tok, &mut y);
            let mut max_abs = 0.0f32;
            let mut ndiff = 0usize;
            for i in 0..y.len() {
                let d = (y[i] - exp[i]).abs();
                if d > 0.0 {
                    ndiff += 1;
                }
                max_abs = max_abs.max(d);
            }
            eprintln!(
                "q4k GEMV empty-none {label} n={} max_abs={max_abs:.8e} ndiff={ndiff} {}",
                y.len(),
                if ndiff == 0 { "BIT_EXACT" } else { "DIFF" }
            );
        }
    }
}
