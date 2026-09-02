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
use std::sync::atomic::{AtomicU32, Ordering};

use crate::dequant::f16_to_f32;
use crate::qmatmul::{BlockQ8K, GEMM_TILE_TOKENS, QK_K};

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
/// (a): `i32x4.dot` + add, not extend-low/high. Same integer total.
#[inline(always)]
fn hsum_i16x8(v: v128) -> i32 {
    hsum_i32x4(i32x4_dot_i16x8(v, i16x8_splat(1)))
}

/// Signed i8×i8 adjacent pair-add → 8 i16. Sat never fires for Q6_K
/// (`|q|≤32`, `|q8|≤127`, pair ≤ 8128). Same pairing as `maddubs` then
/// the m32s subtract: `(q6-32)*q8`.
#[inline(always)]
fn madd_i8_pair_i16(a: v128, b: v128) -> v128 {
    let a_lo = i16x8_extend_low_i8x16(a);
    let a_hi = i16x8_extend_high_i8x16(a);
    let b_lo = i16x8_extend_low_i8x16(b);
    let b_hi = i16x8_extend_high_i8x16(b);
    let p_lo = i16x8_mul(a_lo, b_lo);
    let p_hi = i16x8_mul(a_hi, b_hi);
    let evens = i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(p_lo, p_hi);
    let odds = i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(p_lo, p_hi);
    i16x8_add(evens, odds)
}

/// llama.cpp `ggml_vec_dot_q5_K_q8_K` AVX2 (`quants.c` ~1919) as 2× SIMD128.
///
/// Integer tree matches the ymm split (maddubs + madd + scale shuffle).
/// Float acc uses the same 8-wide FMA-then-`hsum_float_8` shape.
pub unsafe fn vec_dot_q5_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q5_k_q8_k_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

/// Sequence-tiled Q5_K×Q8_K. Weight superblock unpacked once per tile. Precondition: `n_tile <= GEMM_TILE_TOKENS` — caller must clamp (`min`); the `debug_assert!` is compiled out in release.
pub unsafe fn vec_dot_q5_k_q8_k_tile(
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

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4 = u8x16_splat(0x0F);
    let mut acc_lo = [f32x4_splat(0.0); GEMM_TILE_TOKENS];
    let mut acc_hi = [f32x4_splat(0.0); GEMM_TILE_TOKENS];
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

        let mut sc_mins = [0u8; 16];
        sc_mins[0..4].copy_from_slice(&utmp[0].to_le_bytes());
        sc_mins[4..8].copy_from_slice(&utmp[1].to_le_bytes());
        sc_mins[8..12].copy_from_slice(&utmp[2].to_le_bytes());
        sc_mins[12..16].copy_from_slice(&utmp[3].to_le_bytes());
        // cvtepu8_epi16 of 16 u8 → 16 i16. Low 8 = scales, high 8 = mins.
        let packed = load16(sc_mins.as_ptr());
        let mins = u16x8_extend_high_u8x16(packed);

        let hbits_lo = load16(block[16..48].as_ptr());
        let hbits_hi = load16(block[16..48].as_ptr().add(16));
        let q5 = block[48..176].as_ptr();

        // Reconstruct 5-bit quants once per superblock (unpack hoist).
        // q5v[group][0=lo / 1=hi]: 16 unsigned 5-bit values.
        let mut q5v = [[i8x16_splat(0); 2]; 8];
        let mut hmask = u8x16_splat(1);
        let (e0, o0) = q5_precompute::<0, 1>(q5, hbits_lo, hbits_hi, &mut hmask, m4);
        q5v[0] = e0;
        q5v[1] = o0;
        let (e1, o1) = q5_precompute::<2, 3>(q5.add(32), hbits_lo, hbits_hi, &mut hmask, m4);
        q5v[2] = e1;
        q5v[3] = o1;
        let (e2, o2) = q5_precompute::<4, 5>(q5.add(64), hbits_lo, hbits_hi, &mut hmask, m4);
        q5v[4] = e2;
        q5v[5] = o2;
        let (e3, o3) = q5_precompute::<6, 7>(q5.add(96), hbits_lo, hbits_hi, &mut hmask, m4);
        q5v[6] = e3;
        q5v[7] = o3;

        // One i16 splat per 32-element group (byte broadcast + widen), not
        // widen-all + K_SHUFFLE i16 broadcast.
        let sc_i16 = [
            i16x8_splat(i16::from(sc_mins[0])),
            i16x8_splat(i16::from(sc_mins[1])),
            i16x8_splat(i16::from(sc_mins[2])),
            i16x8_splat(i16::from(sc_mins[3])),
            i16x8_splat(i16::from(sc_mins[4])),
            i16x8_splat(i16::from(sc_mins[5])),
            i16x8_splat(i16::from(sc_mins[6])),
            i16x8_splat(i16::from(sc_mins[7])),
        ];

        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
            let d = yb.d * d_w;
            let dmin = -yb.d * dmin_w;

            let bsums_lo = load16(yb.bsums.as_ptr().cast());
            let bsums_hi = load16(yb.bsums.as_ptr().add(8).cast());
            let q8s = hadd_epi16(bsums_lo, bsums_hi);
            let prod = madd_epi16(mins, q8s);
            let extract = hsum_i32x4(prod) as f32;
            summs[t] = fmaf32(dmin, extract, summs[t]);

            let mut sumi_lo = i32x4_splat(0);
            let mut sumi_hi = i32x4_splat(0);
            let q8 = yb.qs.as_ptr();

            for g in 0..8 {
                let q8_off = g * 32;
                let q8_lo = load16_i8(q8.add(q8_off));
                let q8_hi = load16_i8(q8.add(q8_off + 16));
                sumi_lo = i32x4_add(
                    sumi_lo,
                    madd_epi16(sc_i16[g], maddubs_epi16(q5v[g][0], q8_lo)),
                );
                sumi_hi = i32x4_add(
                    sumi_hi,
                    madd_epi16(sc_i16[g], maddubs_epi16(q5v[g][1], q8_hi)),
                );
            }

            acc_lo[t] = fma_f32x4(d, f32x4_convert_i32x4(sumi_lo), acc_lo[t]);
            acc_hi[t] = fma_f32x4(d, f32x4_convert_i32x4(sumi_hi), acc_hi[t]);
        }
    }

    for t in 0..n_tile {
        out[t] = hsum_float_8(acc_lo[t], acc_hi[t]) + summs[t];
    }
}

/// Reconstruct two 32-value Q5 groups (low / high nibble) × two 16-byte
/// lanes. Weight-only — hoisted out of the token loop.
#[inline(always)]
unsafe fn q5_precompute<const BIT0: u32, const BIT1: u32>(
    q5: *const u8,
    hbits_lo: v128,
    hbits_hi: v128,
    hmask: &mut v128,
    m4: v128,
) -> ([v128; 2], [v128; 2]) {
    let hmask0 = *hmask;
    let hmask1 = i16x8_shl(hmask0, 1);
    *hmask = i16x8_shl(hmask1, 1);
    let q5_lo = load16(q5);
    let q5_hi = load16(q5.add(16));
    (
        [
            q5_reconstruct::<BIT0>(q5_lo, hbits_lo, hmask0, m4, false),
            q5_reconstruct::<BIT0>(q5_hi, hbits_hi, hmask0, m4, false),
        ],
        [
            q5_reconstruct::<BIT1>(q5_lo, hbits_lo, hmask1, m4, true),
            q5_reconstruct::<BIT1>(q5_hi, hbits_hi, hmask1, m4, true),
        ],
    )
}

#[inline(always)]
fn q5_reconstruct<const BIT: u32>(
    q5: v128,
    hbits: v128,
    hmask: v128,
    m4: v128,
    hi_nibble: bool,
) -> v128 {
    let q5l = if hi_nibble {
        v128_and(u16x8_shr(q5, 4), m4)
    } else {
        v128_and(q5, m4)
    };
    let q5h = i16x8_shl(u16x8_shr(v128_and(hbits, hmask), BIT), 4);
    i8x16_add(q5l, q5h)
}

/// llama.cpp `ggml_vec_dot_q6_K_q8_K` AVX2 (`quants.c` ~2129) as 2× SIMD128.
pub unsafe fn vec_dot_q6_k_q8_k(w: &[u8], y: &[BlockQ8K]) -> f32 {
    let mut out = [0.0f32; 1];
    vec_dot_q6_k_q8_k_tile(w, y, 1, y.len(), &mut out);
    out[0]
}

/// Sequence-tiled Q6_K×Q8_K. Weight superblock unpacked once per tile. Precondition: `n_tile <= GEMM_TILE_TOKENS` — caller must clamp (`min`); the `debug_assert!` is compiled out in release.
pub unsafe fn vec_dot_q6_k_q8_k_tile(
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

    let m4 = u8x16_splat(0x0F);
    let m2 = u8x16_splat(3);
    let m32s = u8x16_splat(32);
    let mut acc_lo = [f32x4_splat(0.0); GEMM_TILE_TOKENS];
    let mut acc_hi = [f32x4_splat(0.0); GEMM_TILE_TOKENS];

    for i in 0..n_blocks {
        let block = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let q4 = block[0..128].as_ptr();
        let qh = block[128..192].as_ptr();
        let scale_bytes = &block[192..208];
        // 16 signed i8 scales → i16 splat (byte broadcast + widen). Lo/hi
        // halves of each 32-wide reconstruct use consecutive scale bytes.
        let sc_i16 = [
            i16x8_splat(i16::from(scale_bytes[0] as i8)),
            i16x8_splat(i16::from(scale_bytes[1] as i8)),
            i16x8_splat(i16::from(scale_bytes[2] as i8)),
            i16x8_splat(i16::from(scale_bytes[3] as i8)),
            i16x8_splat(i16::from(scale_bytes[4] as i8)),
            i16x8_splat(i16::from(scale_bytes[5] as i8)),
            i16x8_splat(i16::from(scale_bytes[6] as i8)),
            i16x8_splat(i16::from(scale_bytes[7] as i8)),
            i16x8_splat(i16::from(scale_bytes[8] as i8)),
            i16x8_splat(i16::from(scale_bytes[9] as i8)),
            i16x8_splat(i16::from(scale_bytes[10] as i8)),
            i16x8_splat(i16::from(scale_bytes[11] as i8)),
            i16x8_splat(i16::from(scale_bytes[12] as i8)),
            i16x8_splat(i16::from(scale_bytes[13] as i8)),
            i16x8_splat(i16::from(scale_bytes[14] as i8)),
            i16x8_splat(i16::from(scale_bytes[15] as i8)),
        ];

        // Signed (q6-32) reconstruct once per superblock. 16 × 16-value halves.
        let mut q6h = [i8x16_splat(0); 16];
        for j in 0..QK_K / 128 {
            let ql1_lo = load16(q4.add(j * 64));
            let ql1_hi = load16(q4.add(j * 64 + 16));
            let ql2_lo = load16(q4.add(j * 64 + 32));
            let ql2_hi = load16(q4.add(j * 64 + 48));
            let qh_lo = load16(qh.add(j * 32));
            let qh_hi = load16(qh.add(j * 32 + 16));
            let base = j * 8;
            q6h[base] = q6_signed(ql1_lo, qh_lo, m4, m2, m32s, 0, false);
            q6h[base + 1] = q6_signed(ql1_hi, qh_hi, m4, m2, m32s, 0, false);
            q6h[base + 2] = q6_signed(ql2_lo, qh_lo, m4, m2, m32s, 2, false);
            q6h[base + 3] = q6_signed(ql2_hi, qh_hi, m4, m2, m32s, 2, false);
            q6h[base + 4] = q6_signed(ql1_lo, qh_lo, m4, m2, m32s, 4, true);
            q6h[base + 5] = q6_signed(ql1_hi, qh_hi, m4, m2, m32s, 4, true);
            q6h[base + 6] = q6_signed(ql2_lo, qh_lo, m4, m2, m32s, 6, true);
            q6h[base + 7] = q6_signed(ql2_hi, qh_hi, m4, m2, m32s, 6, true);
        }

        for t in 0..n_tile {
            let yb = &qrows[t * n_blocks + i];
            let d = yb.d * d_w;
            let q8 = yb.qs.as_ptr();

            let mut sumi_lo = i32x4_splat(0);
            let mut sumi_hi = i32x4_splat(0);

            for g in 0..16 {
                let q8g = load16_i8(q8.add(g * 16));
                let p16 = madd_i8_pair_i16(q6h[g], q8g);
                let addend = madd_epi16(sc_i16[g], p16);
                if g % 2 == 0 {
                    sumi_lo = i32x4_add(sumi_lo, addend);
                } else {
                    sumi_hi = i32x4_add(sumi_hi, addend);
                }
            }

            acc_lo[t] = fma_f32x4(d, f32x4_convert_i32x4(sumi_lo), acc_lo[t]);
            acc_hi[t] = fma_f32x4(d, f32x4_convert_i32x4(sumi_hi), acc_hi[t]);
        }
    }

    for t in 0..n_tile {
        out[t] = hsum_float_8(acc_lo[t], acc_hi[t]);
    }
}

/// 16 signed q6 values (`ql` nibble + `qh` bits − 32). Weight-only.
#[inline(always)]
fn q6_signed(
    ql: v128,
    qh: v128,
    m4: v128,
    m2: v128,
    m32s: v128,
    qh_shift: u32,
    hi_nibble: bool,
) -> v128 {
    let q4h = i16x8_shl(v128_and(u16x8_shr(qh, qh_shift), m2), 4);
    let q4l = if hi_nibble {
        v128_and(u16x8_shr(ql, 4), m4)
    } else {
        v128_and(ql, m4)
    };
    i8x16_sub(v128_or(q4l, q4h), m32s)
}

/// Bit-exact SIMD128 `ggml_gemv_q4_K_8x8_q8_K` (same integer products + mul+add
/// as the portable GEMV — do not switch this to FMA). One-token wrapper.
pub fn gemv_q4_k_8x8_q8_k(repack: &[u8], y: &[BlockQ8K], n_out: usize, out: &mut [f32]) {
    gemm_q4_k_8x8_q8_k(repack, y, 1, n_out, out);
}

/// Load-time Q4_K policy. `FORCE` is harness-only (`perk` / `allk`); default
/// `auto` uses `THRESHOLD` (token count). Uncalibrated default is 33 = always
/// per-k (#45 HEAD). Calibration cannot change numeric results.
const Q4K_FORCE_AUTO: u32 = 0;
const Q4K_FORCE_PERK: u32 = 1;
const Q4K_FORCE_ALLK: u32 = 2;
/// `tn < threshold` → per-k; `tn >= threshold` → all-k. 33 = never all-k.
static Q4K_FORCE: AtomicU32 = AtomicU32::new(Q4K_FORCE_AUTO);
static Q4K_THRESHOLD: AtomicU32 = AtomicU32::new(33);

pub(crate) fn q4k_set_force(name: &str) {
    let v = match name {
        "perk" | "per-k" | "per_k" => Q4K_FORCE_PERK,
        "allk" | "all-k" | "all_k" => Q4K_FORCE_ALLK,
        "auto" | "" => Q4K_FORCE_AUTO,
        _ => Q4K_FORCE_AUTO,
    };
    Q4K_FORCE.store(v, Ordering::Relaxed);
}

pub(crate) fn q4k_set_threshold(t: u32) {
    Q4K_THRESHOLD.store(t.clamp(1, 33), Ordering::Relaxed);
}

pub(crate) fn q4k_threshold() -> u32 {
    Q4K_THRESHOLD.load(Ordering::Relaxed)
}

#[inline(always)]
fn q4k_use_allk(tn: usize) -> bool {
    match Q4K_FORCE.load(Ordering::Relaxed) {
        Q4K_FORCE_PERK => false,
        Q4K_FORCE_ALLK => true,
        _ => (tn as u32) >= Q4K_THRESHOLD.load(Ordering::Relaxed),
    }
}

/// Sequence-tiled SIMD128 Q4_K×Q8_K 8-col GEMM. Two inner-loop variants,
/// same integer tree and same f32 mul+add stage as #45 (`24c8801`):
/// - **per-k hoist** — #45 HEAD: four pair-loads in small locals, tokens inner
/// - **all-k unpack** — `Q4kUnpacked` (the struct that won long-n on M4)
///
/// Selection is by the cached tile-size threshold (or a harness force).
/// Do not switch this to FMA. (b) per-32 scale-as-built is out.
pub fn gemm_q4_k_8x8_q8_k(
    repack: &[u8],
    qrows: &[BlockQ8K],
    n_tokens: usize,
    n_out: usize,
    out: &mut [f32],
) {
    debug_assert!(n_tokens > 0);
    let n_blocks = qrows.len() / n_tokens;
    debug_assert_eq!(repack.len(), (n_out / 8) * n_blocks * Q4_KX8_BYTES);
    debug_assert_eq!(out.len(), n_tokens * n_out);
    let n_groups = n_out / 8;
    let m4 = i16x8_splat(0x0F);

    for t0 in (0..n_tokens).step_by(GEMM_TILE_TOKENS) {
        let tn = (n_tokens - t0).min(GEMM_TILE_TOKENS);
        if q4k_use_allk(tn) {
            q4k_tile_allk(repack, qrows, t0, tn, n_blocks, n_groups, n_out, m4, out);
        } else {
            q4k_tile_perk(repack, qrows, t0, tn, n_blocks, n_groups, n_out, m4, out);
        }
    }
}

/// #45 per-k hoist. Unpack one 16-byte pair-row into `[v128; 4]` locals,
/// tokens inner. Same `hsum_i16x8` (`i32x4.dot`) + per-stripe scale.
#[inline(never)]
fn q4k_tile_perk(
    repack: &[u8],
    qrows: &[BlockQ8K],
    t0: usize,
    tn: usize,
    n_blocks: usize,
    n_groups: usize,
    n_out: usize,
    m4: v128,
    out: &mut [f32],
) {
    for x in 0..n_groups {
        let mut sumf = [[0.0f32; 8]; GEMM_TILE_TOKENS];
        let mut sum_minf = [[0.0f32; 8]; GEMM_TILE_TOKENS];
        for l in 0..n_blocks {
            let off = (x * n_blocks + l) * Q4_KX8_BYTES;
            let (d, dmin, ub, qs) = unpack_q4_kx8_header(&repack[off..off + Q4_KX8_BYTES]);
            let mut iacc = [[0i32; 8]; GEMM_TILE_TOKENS];
            for k in 0..16 {
                let mut v0_lo = [m4; 4];
                let mut v0_hi = [m4; 4];
                let mut v1_lo = [m4; 4];
                let mut v1_hi = [m4; 4];
                let (s0a, s1a, s0b, s1b) = stripe_scales(&ub, k);
                unsafe {
                    unpack_stripe(qs, k, m4, &mut v0_lo, &mut v0_hi, &mut v1_lo, &mut v1_hi);
                }
                for ti in 0..tn {
                    let yb = &qrows[(t0 + ti) * n_blocks + l];
                    unsafe {
                        q4k_stripe_iacc(
                            &v0_lo,
                            &v0_hi,
                            &v1_lo,
                            &v1_hi,
                            &s0a,
                            &s1a,
                            &s0b,
                            &s1b,
                            yb,
                            k,
                            &mut iacc[ti],
                        );
                    }
                }
            }
            q4k_mins_f32(
                qrows,
                t0,
                tn,
                n_blocks,
                l,
                &d,
                &dmin,
                &ub,
                &iacc,
                &mut sumf,
                &mut sum_minf,
            );
        }
        write_tile_out(out, t0, tn, n_out, x, &sumf, &sum_minf);
    }
}

/// All-k unpack: one `Q4kUnpacked` per column group (256 v128 nibble
/// vectors). Same stripe products and per-stripe scale as per-k. No
/// column-group chunk. No per-32 scale hoist.
#[inline(never)]
fn q4k_tile_allk(
    repack: &[u8],
    qrows: &[BlockQ8K],
    t0: usize,
    tn: usize,
    n_blocks: usize,
    n_groups: usize,
    n_out: usize,
    m4: v128,
    out: &mut [f32],
) {
    for x in 0..n_groups {
        let mut sumf = [[0.0f32; 8]; GEMM_TILE_TOKENS];
        let mut sum_minf = [[0.0f32; 8]; GEMM_TILE_TOKENS];
        for l in 0..n_blocks {
            let off = (x * n_blocks + l) * Q4_KX8_BYTES;
            let unpacked = Q4kUnpacked::from_block(&repack[off..off + Q4_KX8_BYTES], m4);
            let mut iacc = [[0i32; 8]; GEMM_TILE_TOKENS];
            for k in 0..16 {
                let (s0a, s1a, s0b, s1b) = stripe_scales(&unpacked.ub, k);
                for ti in 0..tn {
                    let yb = &qrows[(t0 + ti) * n_blocks + l];
                    unsafe {
                        q4k_stripe_iacc(
                            &unpacked.v0_lo[k],
                            &unpacked.v0_hi[k],
                            &unpacked.v1_lo[k],
                            &unpacked.v1_hi[k],
                            &s0a,
                            &s1a,
                            &s0b,
                            &s1b,
                            yb,
                            k,
                            &mut iacc[ti],
                        );
                    }
                }
            }
            q4k_mins_f32(
                qrows,
                t0,
                tn,
                n_blocks,
                l,
                &unpacked.d,
                &unpacked.dmin,
                &unpacked.ub,
                &iacc,
                &mut sumf,
                &mut sum_minf,
            );
        }
        write_tile_out(out, t0, tn, n_out, x, &sumf, &sum_minf);
    }
}

/// One `block_q4_Kx8`. qs stays in the #40 / #45 pair layout (2 columns × 8
/// nibbles per load) — no 8×8 transpose. Bounds: 1024-byte `qs`, 128-byte
/// unpacked scale/min header.
struct Q4kUnpacked {
    d: [f32; 8],
    dmin: [f32; 8],
    ub: [u8; 128],
    /// `[k][pair]` = column `2*pair` (lo) / `2*pair+1` (hi), 8 i16 nibbles.
    v0_lo: [[v128; 4]; 16],
    v0_hi: [[v128; 4]; 16],
    v1_lo: [[v128; 4]; 16],
    v1_hi: [[v128; 4]; 16],
}

impl Q4kUnpacked {
    fn from_block(blk: &[u8], m4: v128) -> Self {
        let (d, dmin, ub, qs) = unpack_q4_kx8_header(blk);
        let mut v0_lo = [[i16x8_splat(0); 4]; 16];
        let mut v0_hi = [[i16x8_splat(0); 4]; 16];
        let mut v1_lo = [[i16x8_splat(0); 4]; 16];
        let mut v1_hi = [[i16x8_splat(0); 4]; 16];
        unsafe {
            for k in 0..16 {
                unpack_stripe(
                    qs,
                    k,
                    m4,
                    &mut v0_lo[k],
                    &mut v0_hi[k],
                    &mut v1_lo[k],
                    &mut v1_hi[k],
                );
            }
        }
        Self {
            d,
            dmin,
            ub,
            v0_lo,
            v0_hi,
            v1_lo,
            v1_hi,
        }
    }
}

fn unpack_q4_kx8_header(blk: &[u8]) -> ([f32; 8], [f32; 8], [u8; 128], *const u8) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    debug_assert_eq!(blk.len(), Q4_KX8_BYTES);
    let mut d = [0.0f32; 8];
    let mut dmin = [0.0f32; 8];
    for j in 0..8 {
        d[j] = f16_to_f32(u16::from_le_bytes([blk[j * 2], blk[j * 2 + 1]]));
        dmin[j] = f16_to_f32(u16::from_le_bytes([blk[16 + j * 2], blk[16 + j * 2 + 1]]));
    }
    let scales = &blk[32..128];
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
    let mut ub = [0u8; 128];
    for i in 0..32 {
        ub[i * 4..i * 4 + 4].copy_from_slice(&utmp[i].to_le_bytes());
    }
    (d, dmin, ub, blk[128..].as_ptr())
}

/// 16 bytes = 2 columns × 8 packed nibbles. In-bounds:
/// `k*64 + pair*16 ≤ 15*64 + 48 = 1008`, `+16 = 1024`.
#[inline(always)]
unsafe fn unpack_stripe(
    qs: *const u8,
    k: usize,
    m4: v128,
    v0_lo: &mut [v128; 4],
    v0_hi: &mut [v128; 4],
    v1_lo: &mut [v128; 4],
    v1_hi: &mut [v128; 4],
) {
    for pair in 0..4 {
        let j = pair * 2;
        let packed = load16(qs.add(k * 64 + j * 8));
        let qs_lo = u16x8_extend_low_u8x16(packed);
        let qs_hi = u16x8_extend_high_u8x16(packed);
        v0_lo[pair] = v128_and(qs_lo, m4);
        v0_hi[pair] = v128_and(qs_hi, m4);
        v1_lo[pair] = u16x8_shr(qs_lo, 4);
        v1_hi[pair] = u16x8_shr(qs_hi, 4);
    }
}

#[inline(always)]
fn stripe_scales(ub: &[u8; 128], k: usize) -> ([i32; 4], [i32; 4], [i32; 4], [i32; 4]) {
    let scale_base = (k / 4) * 32;
    let mut s0a = [0i32; 4];
    let mut s1a = [0i32; 4];
    let mut s0b = [0i32; 4];
    let mut s1b = [0i32; 4];
    for pair in 0..4 {
        let j = pair * 2;
        s0a[pair] = i32::from(ub[scale_base + j]);
        s1a[pair] = i32::from(ub[scale_base + 16 + j]);
        s0b[pair] = i32::from(ub[scale_base + j + 1]);
        s1b[pair] = i32::from(ub[scale_base + 16 + j + 1]);
    }
    (s0a, s1a, s0b, s1b)
}

/// Shared integer products + per-stripe scale. Same tree as #45 HEAD.
/// `a_off+32+7 ≤ 255` (`k≤15`).
#[inline(always)]
unsafe fn q4k_stripe_iacc(
    v0_lo: &[v128; 4],
    v0_hi: &[v128; 4],
    v1_lo: &[v128; 4],
    v1_hi: &[v128; 4],
    s0a: &[i32; 4],
    s1a: &[i32; 4],
    s0b: &[i32; 4],
    s1b: &[i32; 4],
    yb: &BlockQ8K,
    k: usize,
    iacc: &mut [i32; 8],
) {
    let a_off = (k >> 2) * 64 + (k % 4) * 8;
    let a0 = i16x8_extend_low_i8x16(u64x2(
        ptr::read_unaligned(yb.qs.as_ptr().add(a_off).cast::<u64>()),
        0,
    ));
    let a1 = i16x8_extend_low_i8x16(u64x2(
        ptr::read_unaligned(yb.qs.as_ptr().add(a_off + 32).cast::<u64>()),
        0,
    ));
    for pair in 0..4 {
        let j = pair * 2;
        let sum0a = hsum_i16x8(i16x8_mul(v0_lo[pair], a0));
        let sum1a = hsum_i16x8(i16x8_mul(v1_lo[pair], a1));
        let sum0b = hsum_i16x8(i16x8_mul(v0_hi[pair], a0));
        let sum1b = hsum_i16x8(i16x8_mul(v1_hi[pair], a1));
        iacc[j] += s0a[pair] * sum0a + s1a[pair] * sum1a;
        iacc[j + 1] += s0b[pair] * sum0b + s1b[pair] * sum1b;
    }
}

fn q4k_mins_f32(
    qrows: &[BlockQ8K],
    t0: usize,
    tn: usize,
    n_blocks: usize,
    l: usize,
    d: &[f32; 8],
    dmin: &[f32; 8],
    ub: &[u8; 128],
    iacc: &[[i32; 8]; GEMM_TILE_TOKENS],
    sumf: &mut [[f32; 8]; GEMM_TILE_TOKENS],
    sum_minf: &mut [[f32; 8]; GEMM_TILE_TOKENS],
) {
    for ti in 0..tn {
        let yb = &qrows[(t0 + ti) * n_blocks + l];
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
            sumf[ti][j] += iacc[ti][j] as f32 * ds;
            sum_minf[ti][j] += iacc_min[j] as f32 * (dmin[j] * yb.d);
        }
    }
}

fn write_tile_out(
    out: &mut [f32],
    t0: usize,
    tn: usize,
    n_out: usize,
    x: usize,
    sumf: &[[f32; 8]; GEMM_TILE_TOKENS],
    sum_minf: &[[f32; 8]; GEMM_TILE_TOKENS],
) {
    for ti in 0..tn {
        for j in 0..8 {
            out[(t0 + ti) * n_out + x * 8 + j] = sumf[ti][j] - sum_minf[ti][j];
        }
    }
}

/// Synthetic one-superblock / 8-col / 1-block payload for the load-time
/// micro-bench. Deterministic; not a real weight. Both variants must agree.
fn synth_repack() -> [u8; Q4_KX8_BYTES] {
    let mut blk = [0u8; Q4_KX8_BYTES];
    // f16 1.0 = 0x3C00
    for j in 0..8 {
        blk[j * 2] = 0x00;
        blk[j * 2 + 1] = 0x3C;
        blk[16 + j * 2] = 0x00;
        blk[16 + j * 2 + 1] = 0x3C;
    }
    for i in 32..128 {
        blk[i] = (i as u8).wrapping_mul(17);
    }
    for i in 128..Q4_KX8_BYTES {
        blk[i] = (i as u8).wrapping_mul(13);
    }
    blk
}

fn synth_q8(n: usize) -> Vec<BlockQ8K> {
    (0..n)
        .map(|t| {
            let mut qs = [0i8; QK_K];
            let mut bsums = [0i16; 16];
            for i in 0..QK_K {
                qs[i] = ((t.wrapping_mul(17) + i.wrapping_mul(3)) % 127) as i8 - 63;
            }
            for g in 0..16 {
                let mut s = 0i16;
                for i in 0..16 {
                    s += i16::from(qs[g * 16 + i]);
                }
                bsums[g] = s;
            }
            BlockQ8K {
                d: 0.125,
                qs,
                bsums,
            }
        })
        .collect()
}

fn run_forced_variant(allk: bool, n_tokens: usize, out: &mut [f32]) {
    let repack = synth_repack();
    let qrows = synth_q8(n_tokens);
    let m4 = i16x8_splat(0x0F);
    if allk {
        q4k_tile_allk(&repack, &qrows, 0, n_tokens, 1, 1, 8, m4, out);
    } else {
        q4k_tile_perk(&repack, &qrows, 0, n_tokens, 1, 1, 8, m4, out);
    }
}

/// One superblock × `n_tokens` of the named variant. Calibrator times this.
pub(crate) fn q4k_run_variant(allk: bool, n_tokens: u32) {
    let n = (n_tokens as usize).clamp(1, GEMM_TILE_TOKENS);
    let mut out = [0.0f32; GEMM_TILE_TOKENS * 8];
    run_forced_variant(allk, n, &mut out);
    std::hint::black_box(out[0]);
}

/// `max_abs` between per-k and all-k on the synthetic 32-token tile.
/// Must be 0 — calibration is not allowed to change results.
pub(crate) fn q4k_variant_max_abs() -> f32 {
    let mut perk = [0.0f32; GEMM_TILE_TOKENS * 8];
    let mut allk = [0.0f32; GEMM_TILE_TOKENS * 8];
    run_forced_variant(false, GEMM_TILE_TOKENS, &mut perk);
    run_forced_variant(true, GEMM_TILE_TOKENS, &mut allk);
    let mut max = 0.0f32;
    for i in 0..perk.len() {
        let d = (perk[i] - allk[i]).abs();
        if d > max {
            max = d;
        }
    }
    max
}
