//! WASM SIMD128 Q4_K / Q5_K / Q6_K kernels.
//!
//! Same crate, same `Model`. AVX2 stays `#[cfg(target_arch = "x86_64")]`.
//! These are the `+simd128` twins of the AVX2 vec_dot paths in `qmatmul.rs`
//! (2× v128 = one AVX2 ymm, same integer tree + same FMA-shaped scale)
//! plus a bit-exact SIMD128 Q4_K GEMV (native GEMV is already portable
//! mul+add — vectorizing it must not change the integer products).
//!
//! Portable scalar remains the fallback when this module is not compiled.

use std::arch::wasm32::*;
use std::ptr;

use crate::dequant::f16_to_f32;
use crate::qmatmul::{BlockQ8K, QK_K};

const Q5_K_BYTES: usize = 176;
const Q6_K_BYTES: usize = 210;
const Q4_KX8_BYTES: usize = 1152;

#[inline(always)]
unsafe fn load16(p: *const u8) -> v128 {
    ptr::read_unaligned(p.cast::<v128>())
}

#[inline(always)]
unsafe fn load16_i8(p: *const i8) -> v128 {
    ptr::read_unaligned(p.cast::<v128>())
}

/// SSE `_mm_maddubs_epi16`: unsigned u8 × signed i8, adjacent pair, sat i16.
#[inline(always)]
fn maddubs_epi16(u: v128, s: v128) -> v128 {
    let u_lo = u16x8_extend_low_u8x16(u);
    let u_hi = u16x8_extend_high_u8x16(u);
    let s_lo = i16x8_extend_low_i8x16(s);
    let s_hi = i16x8_extend_high_i8x16(s);
    let p_lo = i16x8_mul(u_lo, s_lo);
    let p_hi = i16x8_mul(u_hi, s_hi);
    let evens = i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(p_lo, p_hi);
    let odds = i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(p_lo, p_hi);
    i16x8_add_sat(evens, odds)
}

/// SSE `_mm_madd_epi16` / WASM `i32x4.dot_i16x8`.
#[inline(always)]
fn madd_epi16(a: v128, b: v128) -> v128 {
    i32x4_dot_i16x8(a, b)
}

/// SSE `_mm_hadd_epi16`.
#[inline(always)]
fn hadd_epi16(a: v128, b: v128) -> v128 {
    let evens = i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b);
    let odds = i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(a, b);
    i16x8_add(evens, odds)
}

/// llama.cpp `hsum_float_8` pairwise tree on 2× f32x4 (the two AVX2 lanes).
#[inline(always)]
fn hsum_float_8(lo: v128, hi: v128) -> f32 {
    let sum4 = f32x4_add(lo, hi);
    // f32x4 shuffle is i32x4 shuffle of the same lanes.
    let movehl = i32x4_shuffle::<2, 3, 2, 3>(sum4, sum4);
    let sum2 = f32x4_add(sum4, movehl);
    let movehdup = i32x4_shuffle::<1, 1, 3, 3>(sum2, sum2);
    let sum1 = f32x4_add(sum2, movehdup);
    f32x4_extract_lane::<0>(sum1)
}

#[inline(always)]
fn fmaf32(a: f32, b: f32, c: f32) -> f32 {
    crate::ops::fmaf32(a, b, c)
}

#[inline(always)]
fn fma_f32x4(a: f32, b: v128, c: v128) -> v128 {
    f32x4(
        fmaf32(a, f32x4_extract_lane::<0>(b), f32x4_extract_lane::<0>(c)),
        fmaf32(a, f32x4_extract_lane::<1>(b), f32x4_extract_lane::<1>(c)),
        fmaf32(a, f32x4_extract_lane::<2>(b), f32x4_extract_lane::<2>(c)),
        fmaf32(a, f32x4_extract_lane::<3>(b), f32x4_extract_lane::<3>(c)),
    )
}

#[inline(always)]
fn hsum_i32x4(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v)
        + i32x4_extract_lane::<1>(v)
        + i32x4_extract_lane::<2>(v)
        + i32x4_extract_lane::<3>(v)
}

/// Horizontal sum of 8 i16 products (fits i32).
#[inline(always)]
fn hsum_i16x8(v: v128) -> i32 {
    hsum_i32x4(i32x4_add(i32x4_extend_low_i16x8(v), i32x4_extend_high_i16x8(v)))
}

/// llama.cpp `ggml_vec_dot_q5_K_q8_K` AVX2 (`quants.c` ~1919) as 2× SIMD128.
///
/// Integer tree matches the ymm split (maddubs + madd + scale shuffle).
/// Float acc uses the same 8-wide FMA-then-`hsum_float_8` shape.
pub unsafe fn vec_dot_q5_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q5_K_BYTES);

    // 8×16-byte rows — each broadcasts one i16 scale (AVX2 K_SHUFFLE low half).
    #[rustfmt::skip]
    const K_SHUFFLE: [u8; 128] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    ];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4 = u8x16_splat(0x0F);
    let mut acc_lo = f32x4_splat(0.0);
    let mut acc_hi = f32x4_splat(0.0);
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

        let mut sc_mins = [0u8; 16];
        sc_mins[0..4].copy_from_slice(&utmp[0].to_le_bytes());
        sc_mins[4..8].copy_from_slice(&utmp[1].to_le_bytes());
        sc_mins[8..12].copy_from_slice(&utmp[2].to_le_bytes());
        sc_mins[12..16].copy_from_slice(&utmp[3].to_le_bytes());
        // cvtepu8_epi16 of 16 u8 → 16 i16. Low 8 = scales, high 8 = mins.
        let packed = load16(sc_mins.as_ptr());
        let scales = u16x8_extend_low_u8x16(packed);
        let mins = u16x8_extend_high_u8x16(packed);

        let bsums_lo = load16(yb.bsums.as_ptr().cast());
        let bsums_hi = load16(yb.bsums.as_ptr().add(8).cast());
        let q8s = hadd_epi16(bsums_lo, bsums_hi);
        let prod = madd_epi16(mins, q8s);
        let extract = hsum_i32x4(prod) as f32;
        summs = fmaf32(dmin, extract, summs);

        let hbits_lo = load16(block[16..48].as_ptr());
        let hbits_hi = load16(block[16..48].as_ptr().add(16));
        let mut hmask = u8x16_splat(1);
        let mut sumi_lo = i32x4_splat(0);
        let mut sumi_hi = i32x4_splat(0);
        let q5 = block[48..176].as_ptr();
        let q8 = yb.qs.as_ptr();
        let sh = K_SHUFFLE.as_ptr();

        q5k_chunk::<0, 1>(
            scales,
            load16(sh),
            load16(sh.add(16)),
            q5,
            q8,
            hbits_lo,
            hbits_hi,
            &mut hmask,
            m4,
            &mut sumi_lo,
            &mut sumi_hi,
        );
        q5k_chunk::<2, 3>(
            scales,
            load16(sh.add(32)),
            load16(sh.add(48)),
            q5.add(32),
            q8.add(64),
            hbits_lo,
            hbits_hi,
            &mut hmask,
            m4,
            &mut sumi_lo,
            &mut sumi_hi,
        );
        q5k_chunk::<4, 5>(
            scales,
            load16(sh.add(64)),
            load16(sh.add(80)),
            q5.add(64),
            q8.add(128),
            hbits_lo,
            hbits_hi,
            &mut hmask,
            m4,
            &mut sumi_lo,
            &mut sumi_hi,
        );
        q5k_chunk::<6, 7>(
            scales,
            load16(sh.add(96)),
            load16(sh.add(112)),
            q5.add(96),
            q8.add(192),
            hbits_lo,
            hbits_hi,
            &mut hmask,
            m4,
            &mut sumi_lo,
            &mut sumi_hi,
        );

        acc_lo = fma_f32x4(d, f32x4_convert_i32x4(sumi_lo), acc_lo);
        acc_hi = fma_f32x4(d, f32x4_convert_i32x4(sumi_hi), acc_hi);
    }

    hsum_float_8(acc_lo, acc_hi) + summs
}

/// One QK_K/64 chunk. `BIT0`/`BIT1` are the AVX2 `_mm256_srli_epi16` immediates.
#[inline(always)]
unsafe fn q5k_chunk<const BIT0: u32, const BIT1: u32>(
    scales: v128,
    sh0: v128,
    sh1: v128,
    q5: *const u8,
    q8: *const i8,
    hbits_lo: v128,
    hbits_hi: v128,
    hmask: &mut v128,
    m4: v128,
    sumi_lo: &mut v128,
    sumi_hi: &mut v128,
) {
    let scale_0 = i8x16_swizzle(scales, sh0);
    let scale_1 = i8x16_swizzle(scales, sh1);
    let hmask0 = *hmask;
    let hmask1 = i16x8_shl(hmask0, 1);
    *hmask = i16x8_shl(hmask1, 1);

    let q5_lo = load16(q5);
    let q5_hi = load16(q5.add(16));
    let q8_0_lo = load16_i8(q8);
    let q8_0_hi = load16_i8(q8.add(16));
    let q8_1_lo = load16_i8(q8.add(32));
    let q8_1_hi = load16_i8(q8.add(48));

    *sumi_lo = i32x4_add(
        *sumi_lo,
        q5k_half::<BIT0, BIT1>(scale_0, scale_1, q5_lo, q8_0_lo, q8_1_lo, hbits_lo, hmask0, hmask1, m4),
    );
    *sumi_hi = i32x4_add(
        *sumi_hi,
        q5k_half::<BIT0, BIT1>(scale_0, scale_1, q5_hi, q8_0_hi, q8_1_hi, hbits_hi, hmask0, hmask1, m4),
    );
}

#[inline(always)]
fn q5k_half<const BIT0: u32, const BIT1: u32>(
    scale_0: v128,
    scale_1: v128,
    q5: v128,
    q8_0: v128,
    q8_1: v128,
    hbits: v128,
    hmask0: v128,
    hmask1: v128,
    m4: v128,
) -> v128 {
    let q5l_0 = v128_and(q5, m4);
    let q5h_0 = i16x8_shl(u16x8_shr(v128_and(hbits, hmask0), BIT0), 4);
    let q5_0 = i8x16_add(q5l_0, q5h_0);

    let q5l_1 = v128_and(u16x8_shr(q5, 4), m4);
    let q5h_1 = i16x8_shl(u16x8_shr(v128_and(hbits, hmask1), BIT1), 4);
    let q5_1 = i8x16_add(q5l_1, q5h_1);

    let p16_0 = madd_epi16(scale_0, maddubs_epi16(q5_0, q8_0));
    let p16_1 = madd_epi16(scale_1, maddubs_epi16(q5_1, q8_1));
    i32x4_add(p16_0, p16_1)
}

/// llama.cpp `ggml_vec_dot_q6_K_q8_K` AVX2 (`quants.c` ~2129) as 2× SIMD128.
pub unsafe fn vec_dot_q6_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(w.len(), y.len() * Q6_K_BYTES);

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

    let m4 = u8x16_splat(0x0F);
    let m2 = u8x16_splat(3);
    let m32s = u8x16_splat(32);
    let mut acc_lo = f32x4_splat(0.0);
    let mut acc_hi = f32x4_splat(0.0);

    for (i, yb) in y.iter().enumerate() {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let d = yb.d * f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let q4 = block[0..128].as_ptr();
        let qh = block[128..192].as_ptr();
        let q8 = yb.qs.as_ptr();
        let scales = load16(block[192..208].as_ptr());

        let mut sumi_lo = i32x4_splat(0);
        let mut sumi_hi = i32x4_splat(0);
        let mut is = 0usize;

        for j in 0..QK_K / 128 {
            let scale_0 = i8x16_swizzle(scales, load16(K_SHUFFLE.as_ptr().add((is + 0) * 16)));
            let scale_1 = i8x16_swizzle(scales, load16(K_SHUFFLE.as_ptr().add((is + 1) * 16)));
            let scale_2 = i8x16_swizzle(scales, load16(K_SHUFFLE.as_ptr().add((is + 2) * 16)));
            let scale_3 = i8x16_swizzle(scales, load16(K_SHUFFLE.as_ptr().add((is + 3) * 16)));
            is += 4;

            let ql1_lo = load16(q4.add(j * 64));
            let ql1_hi = load16(q4.add(j * 64 + 16));
            let ql2_lo = load16(q4.add(j * 64 + 32));
            let ql2_hi = load16(q4.add(j * 64 + 48));
            let qh_lo = load16(qh.add(j * 32));
            let qh_hi = load16(qh.add(j * 32 + 16));

            // q4_0: low nibble of ql1, qh bits 0..1. First 16 → scale[even]/sumi_lo.
            q6_acc16(ql1_lo, qh_lo, load16_i8(q8.add(j * 128)), scale_0, m4, m2, m32s, 0, false, true, &mut sumi_lo);
            q6_acc16(ql1_hi, qh_hi, load16_i8(q8.add(j * 128 + 16)), scale_0, m4, m2, m32s, 0, false, false, &mut sumi_hi);
            // q4_1: low nibble of ql2, qh bits 2..3
            q6_acc16(ql2_lo, qh_lo, load16_i8(q8.add(j * 128 + 32)), scale_1, m4, m2, m32s, 2, false, true, &mut sumi_lo);
            q6_acc16(ql2_hi, qh_hi, load16_i8(q8.add(j * 128 + 48)), scale_1, m4, m2, m32s, 2, false, false, &mut sumi_hi);
            // q4_2: high nibble of ql1, qh bits 4..5
            q6_acc16(ql1_lo, qh_lo, load16_i8(q8.add(j * 128 + 64)), scale_2, m4, m2, m32s, 4, true, true, &mut sumi_lo);
            q6_acc16(ql1_hi, qh_hi, load16_i8(q8.add(j * 128 + 80)), scale_2, m4, m2, m32s, 4, true, false, &mut sumi_hi);
            // q4_3: high nibble of ql2, qh bits 6..7
            q6_acc16(ql2_lo, qh_lo, load16_i8(q8.add(j * 128 + 96)), scale_3, m4, m2, m32s, 6, true, true, &mut sumi_lo);
            q6_acc16(ql2_hi, qh_hi, load16_i8(q8.add(j * 128 + 112)), scale_3, m4, m2, m32s, 6, true, false, &mut sumi_hi);
        }

        acc_lo = fma_f32x4(d, f32x4_convert_i32x4(sumi_lo), acc_lo);
        acc_hi = fma_f32x4(d, f32x4_convert_i32x4(sumi_hi), acc_hi);
    }

    hsum_float_8(acc_lo, acc_hi)
}

/// 16 q6 values: nibble from `ql`, 2 high bits from `qh` shifted by `qh_shift`.
#[inline(always)]
fn q6_reconstruct(ql: v128, qh: v128, m4: v128, m2: v128, qh_shift: u32, hi_nibble: bool) -> v128 {
    let q4h = i16x8_shl(v128_and(u16x8_shr(qh, qh_shift), m2), 4);
    let q4l = if hi_nibble {
        v128_and(u16x8_shr(ql, 4), m4)
    } else {
        v128_and(ql, m4)
    };
    v128_or(q4l, q4h)
}

#[inline(always)]
fn q6_dot16(q6: v128, q8: v128, scale_i16: v128, m32s: v128) -> v128 {
    let mut p16 = maddubs_epi16(q6, q8);
    let q8s = maddubs_epi16(m32s, q8);
    p16 = i16x8_sub(p16, q8s);
    madd_epi16(scale_i16, p16)
}

/// One 16-value half of a 32-wide Q6_K reconstruct+dot.
/// `scale_low` picks `cvtepi8_epi16` low (first 16 of the ymm) vs high.
#[inline(always)]
fn q6_acc16(
    ql: v128,
    qh: v128,
    q8: v128,
    scale: v128,
    m4: v128,
    m2: v128,
    m32s: v128,
    qh_shift: u32,
    hi_nibble: bool,
    scale_low: bool,
    sumi: &mut v128,
) {
    let q6 = q6_reconstruct(ql, qh, m4, m2, qh_shift, hi_nibble);
    let sc = if scale_low {
        i16x8_extend_low_i8x16(scale)
    } else {
        i16x8_extend_high_i8x16(scale)
    };
    *sumi = i32x4_add(*sumi, q6_dot16(q6, q8, sc, m32s));
}

/// Bit-exact SIMD128 `ggml_gemv_q4_K_8x8_q8_K` (same integer products + mul+add
/// as the portable GEMV — do not switch this to FMA).
pub fn gemv_q4_k_8x8_q8_k(repack: &[u8], y: &[BlockQ8K], n_out: usize, out: &mut [f32]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let n_blocks = y.len();
    debug_assert_eq!(repack.len(), (n_out / 8) * n_blocks * Q4_KX8_BYTES);
    debug_assert_eq!(out.len(), n_out);
    let n_groups = n_out / 8;
    let m4 = i16x8_splat(0x0F);

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
            unsafe {
                for k in 0..16 {
                    let scale_base = (k / 4) * 32;
                    let a_off = (k >> 2) * 64 + (k % 4) * 8;
                    let a0 = i16x8_extend_low_i8x16(u64x2(
                        ptr::read_unaligned(yb.qs.as_ptr().add(a_off).cast::<u64>()),
                        0,
                    ));
                    let a1 = i16x8_extend_low_i8x16(u64x2(
                        ptr::read_unaligned(yb.qs.as_ptr().add(a_off + 32).cast::<u64>()),
                        0,
                    ));
                    // 2 columns at a time (16 qs bytes).
                    for pair in 0..4 {
                        let j = pair * 2;
                        let packed = load16(qs.as_ptr().add(k * 64 + j * 8));
                        let qs_lo = u16x8_extend_low_u8x16(packed);
                        let qs_hi = u16x8_extend_high_u8x16(packed);
                        let v0_lo = v128_and(qs_lo, m4);
                        let v0_hi = v128_and(qs_hi, m4);
                        let v1_lo = u16x8_shr(qs_lo, 4);
                        let v1_hi = u16x8_shr(qs_hi, 4);
                        let s0a = i32::from(ub[scale_base + j]);
                        let s1a = i32::from(ub[scale_base + 16 + j]);
                        let s0b = i32::from(ub[scale_base + j + 1]);
                        let s1b = i32::from(ub[scale_base + 16 + j + 1]);
                        let sum0a = hsum_i16x8(i16x8_mul(v0_lo, a0));
                        let sum1a = hsum_i16x8(i16x8_mul(v1_lo, a1));
                        let sum0b = hsum_i16x8(i16x8_mul(v0_hi, a0));
                        let sum1b = hsum_i16x8(i16x8_mul(v1_hi, a1));
                        iacc[j] += s0a * sum0a + s1a * sum1a;
                        iacc[j + 1] += s0b * sum0b + s1b * sum1b;
                    }
                }
            }
            let mut iacc_min = [0i32; 8];
            for sb in 0..8 {
                let bsum = i32::from(yb.bsums[sb * 2]) + i32::from(yb.bsums[sb * 2 + 1]);
                let mins = unsafe {
                    u16x8_extend_low_u8x16(u64x2(
                        ptr::read_unaligned(ub.as_ptr().add(8 + sb * 16).cast::<u64>()),
                        0,
                    ))
                };
                let prod_lo = i32x4_mul(i32x4_extend_low_i16x8(mins), i32x4_splat(bsum));
                let prod_hi = i32x4_mul(i32x4_extend_high_i16x8(mins), i32x4_splat(bsum));
                iacc_min[0] += i32x4_extract_lane::<0>(prod_lo);
                iacc_min[1] += i32x4_extract_lane::<1>(prod_lo);
                iacc_min[2] += i32x4_extract_lane::<2>(prod_lo);
                iacc_min[3] += i32x4_extract_lane::<3>(prod_lo);
                iacc_min[4] += i32x4_extract_lane::<0>(prod_hi);
                iacc_min[5] += i32x4_extract_lane::<1>(prod_hi);
                iacc_min[6] += i32x4_extract_lane::<2>(prod_hi);
                iacc_min[7] += i32x4_extract_lane::<3>(prod_hi);
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
