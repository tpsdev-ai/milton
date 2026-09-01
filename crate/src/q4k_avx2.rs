#![allow(dead_code)] // DSO-matched kernels; embed path stays on GEMV (see qmatmul.rs).
//! AVX2 Q4_K 4x8 kernels, ported bit-exactly from llama.cpp
//! `harness/vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp`:
//! `ggml_quantize_mat_q8_K_4x8` (~290) and the AVX2 4-row tile of
//! `ggml_gemm_q4_K_8x8_q8_K` (~3073). Not the generic 4x8 (`repack.cpp`
//! ~127 / ~730). Not `e9f5f4e`.
//!
//! `block_q8_Kx4`: `d[4] f32` + `qs[QK_K*4] i8` + `bsums[QK_K/4] i16`
//! = 1168 bytes (`repack.h`).

pub const QK_K: usize = 256;
pub const BLOCK_Q8_KX4_BYTES: usize = 16 + QK_K * 4 + (QK_K / 4) * 2;
const OFF_QS: usize = 16;
const OFF_BSUMS: usize = 16 + QK_K * 4;

pub fn available() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

/// llama.cpp `ggml_quantize_mat_q8_K_4x8` AVX2 (`arch/x86/repack.cpp` ~290).
///
/// `x` is 4 rows of `k` floats (row-major). `vy` is `k/QK_K` packed
/// `block_q8_Kx4`. Rounding is `_mm256_round_ps` + `packs` (saturate),
/// not `quantize_row_q8_K`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn quantize_mat_q8_k_4x8(x: *const f32, vy: *mut u8, k: i64) {
    use std::arch::x86_64::*;
    debug_assert_eq!(k % QK_K as i64, 0);
    let nb = (k as usize) / QK_K;
    let mut iscale = [0.0f32; 4];
    let mut srcv = [[_mm256_setzero_ps(); 32]; 4];
    let mut iscale_vec = [_mm256_setzero_ps(); 4];

    for i in 0..nb {
        let y = vy.add(i * BLOCK_Q8_KX4_BYTES);
        for row_iter in 0..4 {
            let base = x.add(row_iter * (k as usize) + i * 256);
            let v0 = _mm256_loadu_ps(base);
            let v1 = _mm256_loadu_ps(base.add(8));
            let v2 = _mm256_loadu_ps(base.add(16));
            let v3 = _mm256_loadu_ps(base.add(24));

            let sign_bit = _mm256_set1_ps(-0.0);
            let abs0 = _mm256_andnot_ps(sign_bit, v0);
            let abs1 = _mm256_andnot_ps(sign_bit, v1);
            let abs2 = _mm256_andnot_ps(sign_bit, v2);
            let abs3 = _mm256_andnot_ps(sign_bit, v3);

            let mut max_abs = _mm256_max_ps(abs0, abs1);
            max_abs = _mm256_max_ps(max_abs, abs2);
            max_abs = _mm256_max_ps(max_abs, abs3);

            let mut mask0 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v0);
            let mut mask1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v1);
            let mut mask2 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v2);
            let mut mask3 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v3);
            let mut mask_abs = _mm256_or_ps(
                _mm256_or_ps(mask0, mask1),
                _mm256_or_ps(mask2, mask3),
            );

            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;

            for sb in 1..8 {
                let temp_abs = max_abs;
                let v0 = _mm256_loadu_ps(base.add(sb * 32));
                let v1 = _mm256_loadu_ps(base.add(sb * 32 + 8));
                let v2 = _mm256_loadu_ps(base.add(sb * 32 + 16));
                let v3 = _mm256_loadu_ps(base.add(sb * 32 + 24));

                let abs0 = _mm256_andnot_ps(sign_bit, v0);
                let abs1 = _mm256_andnot_ps(sign_bit, v1);
                let abs2 = _mm256_andnot_ps(sign_bit, v2);
                let abs3 = _mm256_andnot_ps(sign_bit, v3);

                max_abs = _mm256_max_ps(max_abs, abs0);
                max_abs = _mm256_max_ps(max_abs, abs1);
                max_abs = _mm256_max_ps(max_abs, abs2);
                max_abs = _mm256_max_ps(max_abs, abs3);

                let mask_prev = _mm256_cmp_ps::<_CMP_EQ_OQ>(temp_abs, max_abs);
                mask_abs = _mm256_and_ps(mask_abs, mask_prev);

                mask0 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v0);
                mask1 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v1);
                mask2 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v2);
                mask3 = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_abs, v3);
                let mask_curr = _mm256_or_ps(
                    _mm256_or_ps(mask0, mask1),
                    _mm256_or_ps(mask2, mask3),
                );
                mask_abs = _mm256_or_ps(mask_abs, mask_curr);

                srcv[row_iter][sb * 4] = v0;
                srcv[row_iter][sb * 4 + 1] = v1;
                srcv[row_iter][sb * 4 + 2] = v2;
                srcv[row_iter][sb * 4 + 3] = v3;
            }

            let mut max4 = _mm_max_ps(
                _mm256_extractf128_ps::<1>(max_abs),
                _mm256_castps256_ps128(max_abs),
            );
            max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            let max_scalar = _mm_cvtss_f32(max4);

            let max_scalar_vec = _mm256_set1_ps(max_scalar);
            let mask_next = _mm256_cmp_ps::<_CMP_EQ_OQ>(max_scalar_vec, max_abs);
            let final_mask = _mm256_and_ps(mask_abs, mask_next);
            let mask = _mm256_movemask_ps(final_mask);

            iscale[row_iter] = if max_scalar != 0.0 {
                127.0 / max_scalar
            } else {
                0.0
            };
            if mask != 0 {
                iscale[row_iter] = if max_scalar != 0.0 {
                    -127.0 / max_scalar
                } else {
                    0.0
                };
            }

            let d = if max_scalar != 0.0 {
                1.0 / iscale[row_iter]
            } else {
                0.0
            };
            (y as *mut f32).add(row_iter).write(d);
            iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
        }

        let mut quants_interleaved = [_mm256_setzero_si256(); 32];
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let qs = y.add(OFF_QS);
        for j in 0..32 {
            let mut v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
            let mut v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
            let mut v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
            let mut v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

            v0 = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(v0);
            v1 = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(v1);
            v2 = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(v2);
            v3 = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(v3);

            let mut i0 = _mm256_cvtps_epi32(v0);
            let i1 = _mm256_cvtps_epi32(v1);
            let mut i2 = _mm256_cvtps_epi32(v2);
            let i3 = _mm256_cvtps_epi32(v3);

            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            i0 = _mm256_permutevar8x32_epi32(i0, perm);

            _mm256_storeu_si256(qs.add(32 * j).cast(), i0);
            quants_interleaved[j] = i0;
        }

        let mut shuffle_mask_sb2 = _mm256_castsi128_si256(_mm_setr_epi8(
            0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15,
        ));
        shuffle_mask_sb2 = _mm256_permute2f128_si256::<0>(shuffle_mask_sb2, shuffle_mask_sb2);
        let mut shuffle_mask_sb3 = _mm256_castsi128_si256(_mm_setr_epi8(
            0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15,
        ));
        shuffle_mask_sb3 = _mm256_permute2f128_si256::<0>(shuffle_mask_sb3, shuffle_mask_sb3);
        let mut shuffle_mask_sb4 = _mm256_castsi128_si256(_mm_setr_epi8(
            0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9,
        ));
        shuffle_mask_sb4 = _mm256_permute2f128_si256::<0>(shuffle_mask_sb4, shuffle_mask_sb4);

        let bsums = y.add(OFF_BSUMS);
        for k in 0..4 {
            let mut q0 = quants_interleaved[k * 8];
            let mut q1 = quants_interleaved[k * 8 + 1];
            let mut q2 = quants_interleaved[k * 8 + 2];
            let mut q3 = quants_interleaved[k * 8 + 3];
            let mut q4 = quants_interleaved[k * 8 + 4];
            let mut q5 = quants_interleaved[k * 8 + 5];
            let mut q6 = quants_interleaved[k * 8 + 6];
            let mut q7 = quants_interleaved[k * 8 + 7];

            let mut sb2_h1 = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
            let mut sb_h1 = _mm256_blend_epi16::<34>(q0, sb2_h1);
            let mut sb3_h1 = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
            sb_h1 = _mm256_blend_epi16::<68>(sb_h1, sb3_h1);
            let mut sb4_h1 = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
            sb_h1 = _mm256_blend_epi16::<136>(sb_h1, sb4_h1);

            let one = _mm256_set1_epi8(1);
            let mut bsums_r1 = _mm256_maddubs_epi16(one, sb_h1);

            for _ in 0..3 {
                q0 = _mm256_srli_epi64::<16>(q0);
                q2 = _mm256_srli_epi64::<16>(q2);
                q4 = _mm256_srli_epi64::<16>(q4);
                q6 = _mm256_srli_epi64::<16>(q6);

                sb2_h1 = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
                sb_h1 = _mm256_blend_epi16::<34>(q0, sb2_h1);
                sb3_h1 = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
                sb_h1 = _mm256_blend_epi16::<68>(sb_h1, sb3_h1);
                sb4_h1 = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
                sb_h1 = _mm256_blend_epi16::<136>(sb_h1, sb4_h1);

                bsums_r1 =
                    _mm256_add_epi16(bsums_r1, _mm256_maddubs_epi16(one, sb_h1));
            }

            let mut sb2_h2 = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
            let mut sb_h2 = _mm256_blend_epi16::<34>(q1, sb2_h2);
            let mut sb3_h2 = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
            sb_h2 = _mm256_blend_epi16::<68>(sb_h2, sb3_h2);
            let mut sb4_h2 = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
            sb_h2 = _mm256_blend_epi16::<136>(sb_h2, sb4_h2);

            let mut bsums_r2 = _mm256_maddubs_epi16(one, sb_h2);

            for _ in 0..3 {
                q1 = _mm256_srli_epi64::<16>(q1);
                q3 = _mm256_srli_epi64::<16>(q3);
                q5 = _mm256_srli_epi64::<16>(q5);
                q7 = _mm256_srli_epi64::<16>(q7);

                sb2_h2 = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
                sb_h2 = _mm256_blend_epi16::<34>(q1, sb2_h2);
                sb3_h2 = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
                sb_h2 = _mm256_blend_epi16::<68>(sb_h2, sb3_h2);
                sb4_h2 = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
                sb_h2 = _mm256_blend_epi16::<136>(sb_h2, sb4_h2);

                bsums_r2 =
                    _mm256_add_epi16(bsums_r2, _mm256_maddubs_epi16(one, sb_h2));
            }

            let bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
            _mm256_storeu_si256((bsums as *mut i16).add(16 * k).cast(), bsums_r);
        }
    }
}

#[inline(always)]
unsafe fn f32cx8_load(p: *const u8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_cvtph_ps(_mm_loadu_si128(p.cast()))
}

#[inline(always)]
unsafe fn maddubs4(
    rhs: [std::arch::x86_64::__m256i; 4],
    lhs: [std::arch::x86_64::__m256i; 4],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs[3], lhs[3]),
                _mm256_maddubs_epi16(rhs[2], lhs[2]),
            ),
            _mm256_maddubs_epi16(rhs[1], lhs[1]),
        ),
        _mm256_maddubs_epi16(rhs[0], lhs[0]),
    )
}

/// llama.cpp AVX2 `ggml_gemm_q4_K_8x8_q8_K` 4-row tile (`arch/x86/repack.cpp` ~3073).
///
/// The 16-row unroll (~2732) is the same per-row FMA order applied to four
/// independent 4-row groups; this tile is the kernel goldens used for
/// `nr % 16 != 0` (hello: nr=4) and is bit-identical per output row.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemm_q4_k_8x8_q8_k(
    n: i32,
    s: *mut f32,
    bs: usize,
    vx: *const u8,
    vy: *const u8,
    nr: i32,
    nc: i32,
) {
    use std::arch::x86_64::*;
    const Q4_KX8: usize = 1152;
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    debug_assert_eq!(n as usize % QK_K, 0);
    debug_assert_eq!(nr % 4, 0);
    debug_assert_eq!(nc % 8, 0);

    let nb = (n as usize) / QK_K;
    let m4b = _mm256_set1_epi8(0x0F);
    let required_order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

    let mut y = 0i64;
    while y < (nr as i64) / 4 {
        let a_ptr = vy.add((y as usize) * nb * BLOCK_Q8_KX4_BYTES);
        let mut x = 0i64;
        while x < (nc as i64) / 8 {
            let b_ptr = vx.add((x as usize) * nb * Q4_KX8);
            let mut acc_rows = [_mm256_setzero_ps(); 4];
            let mut acc_min_rows = [_mm256_setzero_ps(); 4];

            for b in 0..nb {
                let blk = b_ptr.add(b * Q4_KX8);
                let col_scale_f32 = f32cx8_load(blk);
                let col_dmin_f32 = f32cx8_load(blk.add(16));
                let a_blk = a_ptr.add(b * BLOCK_Q8_KX4_BYTES);
                let a_qs = a_blk.add(OFF_QS);
                let a_bsums = a_blk.add(OFF_BSUMS) as *const i16;
                let a_d = a_blk as *const f32;

                for sb in 0..(QK_K / 64) {
                    let b_qs = blk.add(128);
                    let mut rhs_0145 = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367 = [_mm256_setzero_si256(); 4];
                    for i in 0..4 {
                        let raw_0123 =
                            _mm256_loadu_si256(b_qs.add(sb * 256 + i * 64).cast());
                        let raw_4567 =
                            _mm256_loadu_si256(b_qs.add(sb * 256 + 32 + i * 64).cast());
                        rhs_0145[i] = _mm256_blend_epi32::<240>(
                            raw_0123,
                            _mm256_permutevar8x32_epi32(raw_4567, required_order),
                        );
                        rhs_2367[i] = _mm256_blend_epi32::<240>(
                            _mm256_permutevar8x32_epi32(raw_0123, required_order),
                            raw_4567,
                        );
                    }

                    let mut rhs_0145_lo = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367_lo = [_mm256_setzero_si256(); 4];
                    let mut rhs_0145_hi = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367_hi = [_mm256_setzero_si256(); 4];
                    for i in 0..4 {
                        rhs_0145_lo[i] = _mm256_and_si256(rhs_0145[i], m4b);
                        rhs_2367_lo[i] = _mm256_and_si256(rhs_2367[i], m4b);
                        rhs_0145_hi[i] =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_0145[i]), m4b);
                        rhs_2367_hi[i] =
                            _mm256_and_si256(_mm256_srli_epi16::<4>(rhs_2367[i]), m4b);
                    }

                    let mut rhs_0145_lo_sp1 = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367_lo_sp1 = [_mm256_setzero_si256(); 4];
                    let mut rhs_0145_hi_sp1 = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367_hi_sp1 = [_mm256_setzero_si256(); 4];
                    let mut rhs_0145_lo_sp2 = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367_lo_sp2 = [_mm256_setzero_si256(); 4];
                    let mut rhs_0145_hi_sp2 = [_mm256_setzero_si256(); 4];
                    let mut rhs_2367_hi_sp2 = [_mm256_setzero_si256(); 4];
                    for i in 0..4 {
                        rhs_0145_lo_sp1[i] = _mm256_shuffle_epi32::<136>(rhs_0145_lo[i]);
                        rhs_2367_lo_sp1[i] = _mm256_shuffle_epi32::<136>(rhs_2367_lo[i]);
                        rhs_0145_hi_sp1[i] = _mm256_shuffle_epi32::<136>(rhs_0145_hi[i]);
                        rhs_2367_hi_sp1[i] = _mm256_shuffle_epi32::<136>(rhs_2367_hi[i]);
                        rhs_0145_lo_sp2[i] = _mm256_shuffle_epi32::<221>(rhs_0145_lo[i]);
                        rhs_2367_lo_sp2[i] = _mm256_shuffle_epi32::<221>(rhs_2367_lo[i]);
                        rhs_0145_hi_sp2[i] = _mm256_shuffle_epi32::<221>(rhs_0145_hi[i]);
                        rhs_2367_hi_sp2[i] = _mm256_shuffle_epi32::<221>(rhs_2367_hi[i]);
                    }

                    let scales = blk.add(32);
                    let mut utmp_0 = [0u32; 4];
                    let mut utmp_1 = [0u32; 4];
                    std::ptr::copy_nonoverlapping(
                        scales.add(24 * sb) as *const u8,
                        utmp_0.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_0[3] = ((utmp_0[2] >> 4) & KMASK2) | (((utmp_0[1] >> 6) & KMASK3) << 4);
                    let uaux_0 = utmp_0[1] & KMASK1;
                    utmp_0[1] = (utmp_0[2] & KMASK2) | (((utmp_0[0] >> 6) & KMASK3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= KMASK1;

                    std::ptr::copy_nonoverlapping(
                        scales.add(12 + sb * 24) as *const u8,
                        utmp_1.as_mut_ptr() as *mut u8,
                        12,
                    );
                    utmp_1[3] = ((utmp_1[2] >> 4) & KMASK2) | (((utmp_1[1] >> 6) & KMASK3) << 4);
                    let uaux_1 = utmp_1[1] & KMASK1;
                    utmp_1[1] = (utmp_1[2] & KMASK2) | (((utmp_1[0] >> 6) & KMASK3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= KMASK1;

                    let mins_and_scales_0 = _mm_set_epi32(
                        utmp_0[3] as i32,
                        utmp_0[2] as i32,
                        utmp_0[1] as i32,
                        utmp_0[0] as i32,
                    );
                    let scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        mins_and_scales_0,
                        mins_and_scales_0,
                    ));
                    let mins_and_scales_1 = _mm_set_epi32(
                        utmp_1[3] as i32,
                        utmp_1[2] as i32,
                        utmp_1[1] as i32,
                        utmp_1[0] as i32,
                    );
                    let scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        mins_and_scales_1,
                        mins_and_scales_1,
                    ));
                    let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        _mm_shuffle_epi32::<78>(mins_and_scales_0),
                        _mm_shuffle_epi32::<78>(mins_and_scales_1),
                    ));
                    let scale_0145_0 = _mm256_shuffle_epi32::<68>(scales_0);
                    let scale_2367_0 = _mm256_shuffle_epi32::<238>(scales_0);
                    let scale_0145_1 = _mm256_shuffle_epi32::<68>(scales_1);
                    let scale_2367_1 = _mm256_shuffle_epi32::<238>(scales_1);

                    let mut lhs_01_lo = [_mm256_setzero_si256(); 4];
                    let mut lhs_23_lo = [_mm256_setzero_si256(); 4];
                    let mut lhs_01_hi = [_mm256_setzero_si256(); 4];
                    let mut lhs_23_hi = [_mm256_setzero_si256(); 4];
                    for i in 0..4 {
                        let lo = _mm256_loadu_si256(a_qs.add(256 * sb + i * 32).cast());
                        lhs_01_lo[i] = _mm256_permute2f128_si256::<0>(lo, lo);
                        lhs_23_lo[i] = _mm256_permute2f128_si256::<17>(lo, lo);
                        let hi = _mm256_loadu_si256(a_qs.add(128 + 256 * sb + i * 32).cast());
                        lhs_01_hi[i] = _mm256_permute2f128_si256::<0>(hi, hi);
                        lhs_23_hi[i] = _mm256_permute2f128_si256::<17>(hi, hi);
                    }

                    let lhs_bsums = _mm256_loadu_si256(a_bsums.add(16 * sb).cast());
                    let mut lhs_bsums_hsum = _mm256_castsi128_si256(_mm_hadd_epi16(
                        _mm256_castsi256_si128(lhs_bsums),
                        _mm256_extractf128_si256::<1>(lhs_bsums),
                    ));
                    lhs_bsums_hsum =
                        _mm256_permute2x128_si256::<0>(lhs_bsums_hsum, lhs_bsums_hsum);

                    let mut lhs_01_lo_sp1 = [_mm256_setzero_si256(); 4];
                    let mut lhs_23_lo_sp1 = [_mm256_setzero_si256(); 4];
                    let mut lhs_01_hi_sp1 = [_mm256_setzero_si256(); 4];
                    let mut lhs_23_hi_sp1 = [_mm256_setzero_si256(); 4];
                    let mut lhs_01_lo_sp2 = [_mm256_setzero_si256(); 4];
                    let mut lhs_23_lo_sp2 = [_mm256_setzero_si256(); 4];
                    let mut lhs_01_hi_sp2 = [_mm256_setzero_si256(); 4];
                    let mut lhs_23_hi_sp2 = [_mm256_setzero_si256(); 4];
                    for i in 0..4 {
                        lhs_01_lo_sp1[i] = _mm256_shuffle_epi32::<160>(lhs_01_lo[i]);
                        lhs_23_lo_sp1[i] = _mm256_shuffle_epi32::<160>(lhs_23_lo[i]);
                        lhs_01_hi_sp1[i] = _mm256_shuffle_epi32::<160>(lhs_01_hi[i]);
                        lhs_23_hi_sp1[i] = _mm256_shuffle_epi32::<160>(lhs_23_hi[i]);
                        lhs_01_lo_sp2[i] = _mm256_shuffle_epi32::<245>(lhs_01_lo[i]);
                        lhs_23_lo_sp2[i] = _mm256_shuffle_epi32::<245>(lhs_23_lo[i]);
                        lhs_01_hi_sp2[i] = _mm256_shuffle_epi32::<245>(lhs_01_hi[i]);
                        lhs_23_hi_sp2[i] = _mm256_shuffle_epi32::<245>(lhs_23_hi[i]);
                    }

                    let iacc_mat_00_0_sp1 = maddubs4(rhs_0145_lo_sp1, lhs_01_lo_sp1);
                    let iacc_mat_01_0_sp1 = maddubs4(rhs_2367_lo_sp1, lhs_01_lo_sp1);
                    let iacc_mat_10_0_sp1 = maddubs4(rhs_0145_lo_sp1, lhs_23_lo_sp1);
                    let iacc_mat_11_0_sp1 = maddubs4(rhs_2367_lo_sp1, lhs_23_lo_sp1);
                    let iacc_mat_00_1_sp1 = maddubs4(rhs_0145_hi_sp1, lhs_01_hi_sp1);
                    let iacc_mat_01_1_sp1 = maddubs4(rhs_2367_hi_sp1, lhs_01_hi_sp1);
                    let iacc_mat_10_1_sp1 = maddubs4(rhs_0145_hi_sp1, lhs_23_hi_sp1);
                    let iacc_mat_11_1_sp1 = maddubs4(rhs_2367_hi_sp1, lhs_23_hi_sp1);

                    let iacc_mat_00_0_sp2 = maddubs4(rhs_0145_lo_sp2, lhs_01_lo_sp2);
                    let iacc_mat_01_0_sp2 = maddubs4(rhs_2367_lo_sp2, lhs_01_lo_sp2);
                    let iacc_mat_10_0_sp2 = maddubs4(rhs_0145_lo_sp2, lhs_23_lo_sp2);
                    let iacc_mat_11_0_sp2 = maddubs4(rhs_2367_lo_sp2, lhs_23_lo_sp2);
                    let iacc_mat_00_1_sp2 = maddubs4(rhs_0145_hi_sp2, lhs_01_hi_sp2);
                    let iacc_mat_01_1_sp2 = maddubs4(rhs_2367_hi_sp2, lhs_01_hi_sp2);
                    let iacc_mat_10_1_sp2 = maddubs4(rhs_0145_hi_sp2, lhs_23_hi_sp2);
                    let iacc_mat_11_1_sp2 = maddubs4(rhs_2367_hi_sp2, lhs_23_hi_sp2);

                    let mut iacc_mat_00_0 =
                        _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    let mut iacc_mat_01_0 =
                        _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    let mut iacc_mat_10_0 =
                        _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    let mut iacc_mat_11_0 =
                        _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);
                    let mut iacc_mat_00_1 =
                        _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    let mut iacc_mat_01_1 =
                        _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    let mut iacc_mat_10_1 =
                        _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    let mut iacc_mat_11_1 =
                        _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);
                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    let iacc_row_0_0 = _mm256_blend_epi32::<204>(
                        iacc_mat_00_0,
                        _mm256_shuffle_epi32::<78>(iacc_mat_01_0),
                    );
                    let iacc_row_1_0 = _mm256_blend_epi32::<204>(
                        _mm256_shuffle_epi32::<78>(iacc_mat_00_0),
                        iacc_mat_01_0,
                    );
                    let iacc_row_2_0 = _mm256_blend_epi32::<204>(
                        iacc_mat_10_0,
                        _mm256_shuffle_epi32::<78>(iacc_mat_11_0),
                    );
                    let iacc_row_3_0 = _mm256_blend_epi32::<204>(
                        _mm256_shuffle_epi32::<78>(iacc_mat_10_0),
                        iacc_mat_11_0,
                    );
                    let iacc_row_0_1 = _mm256_blend_epi32::<204>(
                        iacc_mat_00_1,
                        _mm256_shuffle_epi32::<78>(iacc_mat_01_1),
                    );
                    let iacc_row_1_1 = _mm256_blend_epi32::<204>(
                        _mm256_shuffle_epi32::<78>(iacc_mat_00_1),
                        iacc_mat_01_1,
                    );
                    let iacc_row_2_1 = _mm256_blend_epi32::<204>(
                        iacc_mat_10_1,
                        _mm256_shuffle_epi32::<78>(iacc_mat_11_1),
                    );
                    let iacc_row_3_1 = _mm256_blend_epi32::<204>(
                        _mm256_shuffle_epi32::<78>(iacc_mat_10_1),
                        iacc_mat_11_1,
                    );

                    let iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    let iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    let iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    let iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    let row_scale_f32_sse = _mm_loadu_ps(a_d);
                    let row_scale_f32 = _mm256_insertf128_ps::<1>(
                        _mm256_castps128_ps256(row_scale_f32_sse),
                        row_scale_f32_sse,
                    );

                    acc_rows[0] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_0),
                        _mm256_mul_ps(
                            col_scale_f32,
                            _mm256_shuffle_ps::<0>(row_scale_f32, row_scale_f32),
                        ),
                        acc_rows[0],
                    );
                    acc_rows[1] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_1),
                        _mm256_mul_ps(
                            col_scale_f32,
                            _mm256_shuffle_ps::<85>(row_scale_f32, row_scale_f32),
                        ),
                        acc_rows[1],
                    );
                    acc_rows[2] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_2),
                        _mm256_mul_ps(
                            col_scale_f32,
                            _mm256_shuffle_ps::<170>(row_scale_f32, row_scale_f32),
                        ),
                        acc_rows[2],
                    );
                    acc_rows[3] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_3),
                        _mm256_mul_ps(
                            col_scale_f32,
                            _mm256_shuffle_ps::<255>(row_scale_f32, row_scale_f32),
                        ),
                        acc_rows[3],
                    );

                    let iacc_row_min_0 =
                        _mm256_madd_epi16(_mm256_shuffle_epi32::<0>(lhs_bsums_hsum), mins_01);
                    let iacc_row_min_1 =
                        _mm256_madd_epi16(_mm256_shuffle_epi32::<85>(lhs_bsums_hsum), mins_01);
                    let iacc_row_min_2 =
                        _mm256_madd_epi16(_mm256_shuffle_epi32::<170>(lhs_bsums_hsum), mins_01);
                    let iacc_row_min_3 =
                        _mm256_madd_epi16(_mm256_shuffle_epi32::<255>(lhs_bsums_hsum), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_min_0),
                        _mm256_mul_ps(
                            col_dmin_f32,
                            _mm256_shuffle_ps::<0>(row_scale_f32, row_scale_f32),
                        ),
                        acc_min_rows[0],
                    );
                    acc_min_rows[1] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_min_1),
                        _mm256_mul_ps(
                            col_dmin_f32,
                            _mm256_shuffle_ps::<85>(row_scale_f32, row_scale_f32),
                        ),
                        acc_min_rows[1],
                    );
                    acc_min_rows[2] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_min_2),
                        _mm256_mul_ps(
                            col_dmin_f32,
                            _mm256_shuffle_ps::<170>(row_scale_f32, row_scale_f32),
                        ),
                        acc_min_rows[2],
                    );
                    acc_min_rows[3] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iacc_row_min_3),
                        _mm256_mul_ps(
                            col_dmin_f32,
                            _mm256_shuffle_ps::<255>(row_scale_f32, row_scale_f32),
                        ),
                        acc_min_rows[3],
                    );
                }
            }

            for i in 0..4 {
                _mm256_storeu_ps(
                    s.add(((y * 4 + i as i64) as usize) * bs + (x as usize) * 8),
                    _mm256_sub_ps(acc_rows[i], acc_min_rows[i]),
                );
            }
            x += 1;
        }
        y += 1;
    }
}
