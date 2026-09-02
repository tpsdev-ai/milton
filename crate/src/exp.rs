//! Shared permissive f32 `exp` for native and WASM.
//!
//! From-scratch Cody-Waite range reduction + Horner minimax. Evaluation is
//! **mul+add, not FMA** on every backend (same discipline as the Q4_K GEMV).
//! Minimax coefficients are published mathematical constants (SLEEF `xexpf`
//! under BSL-1.0; coefficients themselves are not copyrightable). Same
//! polynomial, same op order, same rounding ⇒ native vs WASM bit-identical
//! by construction.
//!
//! Do not call `f32::exp` on the embed path: native is glibc `expf`, WASM is
//! compiler-builtins, and they split at 1 ULP (issue #24 residual 0.0105).
//! Do not copy glibc (LGPL). Do not swap in musl/`libm::expf`.

#[cfg(target_arch = "wasm32")]
use std::ptr;

/// `log2(e)` as f32.
const LOG2_E: f32 = f32::from_bits(0x3fb8aa3b);
/// Cody-Waite `ln(2)` high part (lower 12 bits cleared).
const LN2_HI: f32 = f32::from_bits(0x3f317200);
/// Cody-Waite `ln(2)` low part.
const LN2_LO: f32 = f32::from_bits(0x35bfbe8e);
/// Horner coefficients for `exp(r)` on `|r| ≤ ln(2)/2`.
const C0: f32 = f32::from_bits(0x39502bda);
const C1: f32 = f32::from_bits(0x3ab6a000);
const C2: f32 = f32::from_bits(0x3c0888b3);
const C3: f32 = f32::from_bits(0x3d2aaaa8);
const C4: f32 = f32::from_bits(0x3e2aaaab);
const C5: f32 = 0.5;
/// `expf` overflow threshold (SLEEF `xexpf`).
const OVERFLOW: f32 = f32::from_bits(0x42b17218);
/// Below this, `expf` underflows to 0.
const UNDERFLOW: f32 = -104.0;
/// `2^23` — magic for round-ties-to-even on `|x| < 2^23`.
const MAGIC: f32 = 8388608.0;
const SIGN_BITS: u32 = 0x8000_0000;

/// Round `x` to nearest integer, ties to even. Domain `|x| < 2^23`.
#[inline(always)]
fn rint_ne(x: f32) -> f32 {
    let magic = f32::from_bits(MAGIC.to_bits() | (x.to_bits() & SIGN_BITS));
    (x + magic) - magic
}

/// `2^n * u` by adding `n` to the exponent field. `u` is a normal in ~[0.7, 1.5].
///
/// The encoding is only valid while the result stays a normal (biased
/// exponent in 1..=254). `UNDERFLOW` is `-104`, so x in about
/// `[-104, -88.4]` still reaches here with `n` ≤ `-128`. A wrapping add
/// then flips the sign bit and returns a huge negative or `-Inf` instead
/// of a tiny positive. Flush those lanes to 0 / +Inf — never wrap.
#[inline(always)]
fn ldexp_n(u: f32, n: i32) -> f32 {
    let exp = ((u.to_bits() >> 23) & 0xff) as i32;
    let e = exp.saturating_add(n);
    if e <= 0 {
        return 0.0;
    }
    if e >= 255 {
        return f32::INFINITY;
    }
    f32::from_bits(u.to_bits().wrapping_add((n as u32) << 23))
}

/// Shared f32 `exp`. Mul+add Horner, magic rint, bit-ldexp. Not libm.
/// `inline(never)` so a small softmax loop cannot be auto-vectorized into a
/// different tree than the explicit SIMD kernels (native-vs-WASM split).
#[inline(never)]
pub fn expf_shared(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > OVERFLOW {
        return f32::INFINITY;
    }
    if x < UNDERFLOW {
        return 0.0;
    }
    let n_f = rint_ne(x * LOG2_E);
    let n = n_f as i32;
    // r = x - n*ln2  as two mul+add steps (not FMA).
    let r = (x + n_f * (-LN2_HI)) + n_f * (-LN2_LO);
    let mut u = C0;
    u = u * r + C1;
    u = u * r + C2;
    u = u * r + C3;
    u = u * r + C4;
    u = u * r + C5;
    u = u * r + 1.0;
    u = u * r + 1.0;
    ldexp_n(u, n)
}

/// In-place `exp` over a slice. SIMD 4-wide (WASM) / 8-wide (AVX2), scalar tail.
/// Each lane is the scalar polynomial, so SIMD and scalar are bit-identical.
/// Scalar tail stays *outside* `#[target_feature(enable = "avx2")]` so LLVM
/// cannot contract the Horner into FMA (which WASM SIMD128 cannot match).
pub fn expf_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "wasm32")]
    let i = unsafe { expf_inplace_simd128(x) };
    #[cfg(target_arch = "x86_64")]
    let i = if is_x86_feature_detected!("avx2") {
        unsafe { expf_inplace_avx(x) }
    } else {
        0
    };
    #[cfg(not(any(target_arch = "wasm32", target_arch = "x86_64")))]
    let i = 0usize;
    for v in &mut x[i..] {
        *v = expf_shared(*v);
    }
}

/// `out[i] = up[i] * silu(gate[i])` with `silu(g) = g / (1 + exp(-g))`.
/// Same shared `exp`, same mul+add, vectorized on both backends.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "wasm32")]
    let i = unsafe { swiglu_simd128(gate, up, out) };
    #[cfg(target_arch = "x86_64")]
    let i = if is_x86_feature_detected!("avx2") {
        unsafe { swiglu_avx(gate, up, out) }
    } else {
        0
    };
    #[cfg(not(any(target_arch = "wasm32", target_arch = "x86_64")))]
    let i = 0usize;
    swiglu_scalar(&gate[i..], &up[i..], &mut out[i..]);
}

/// `silu(x) = x / (1 + exp(-x))` with the shared permissive `exp`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + expf_shared(-x))
}

fn swiglu_scalar(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..gate.len() {
        out[i] = up[i] * silu(gate[i]);
    }
}

/// `out[i] = a * v[i] + out[i]` — mul+add, not FMA. SIMD on both backends.
///
/// Replaces the scalar `fmaf32` V-mix. Same op order on native and WASM so
/// they match; mul+add (not FMA) so WASM SIMD128 can do it with `f32x4.mul`+
/// `add` (no IEEE FMA on the WASM spec). Scalar tail is outside AVX2
/// `target_feature` so it cannot contract to `vfmadd`.
pub fn vmix_axpy(out: &mut [f32], a: f32, v: &[f32]) {
    debug_assert_eq!(out.len(), v.len());
    #[cfg(target_arch = "wasm32")]
    let i = unsafe { vmix_axpy_simd128(out, a, v) };
    #[cfg(target_arch = "x86_64")]
    let i = if is_x86_feature_detected!("avx2") {
        unsafe { vmix_axpy_avx(out, a, v) }
    } else {
        0
    };
    #[cfg(not(any(target_arch = "wasm32", target_arch = "x86_64")))]
    let i = 0usize;
    for j in i..out.len() {
        out[j] = a * v[j] + out[j];
    }
}

#[cfg(target_arch = "wasm32")]
unsafe fn expf_inplace_simd128(x: &mut [f32]) -> usize {
    use std::arch::wasm32::*;
    let n = x.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let v = ptr::read_unaligned(x.as_ptr().add(i).cast::<v128>());
        ptr::write_unaligned(x.as_mut_ptr().add(i).cast::<v128>(), expf4_simd128(v));
        i += 4;
    }
    i
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn expf4_simd128(x: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
    let t = f32x4_mul(x, f32x4_splat(LOG2_E));
    let sign = v128_and(t, i32x4_splat(SIGN_BITS as i32));
    let m = v128_or(sign, f32x4_splat(MAGIC));
    let n_f = f32x4_sub(f32x4_add(t, m), m);
    let r = f32x4_add(
        f32x4_add(x, f32x4_mul(n_f, f32x4_splat(-LN2_HI))),
        f32x4_mul(n_f, f32x4_splat(-LN2_LO)),
    );
    let mut u = f32x4_splat(C0);
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(C1));
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(C2));
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(C3));
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(C4));
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(C5));
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(1.0));
    u = f32x4_add(f32x4_mul(u, r), f32x4_splat(1.0));
    // Same ldexp_n as scalar: exponent add only when the result is normal.
    let n_i = i32x4_trunc_sat_f32x4(n_f);
    let n = i32x4_shl(n_i, 23);
    let mut y = i32x4_add(u, n);
    let exp_u = u32x4_shr(v128_and(u, i32x4_splat(0x7f80_0000u32 as i32)), 23);
    let e = i32x4_add(exp_u, n_i);
    let denorm = i32x4_lt(e, i32x4_splat(1));
    let ovf_exp = i32x4_gt(e, i32x4_splat(254));
    y = v128_bitselect(f32x4_splat(0.0), y, denorm);
    y = v128_bitselect(f32x4_splat(f32::INFINITY), y, ovf_exp);
    let overflow = f32x4_gt(x, f32x4_splat(OVERFLOW));
    let underflow = f32x4_lt(x, f32x4_splat(UNDERFLOW));
    y = v128_bitselect(f32x4_splat(f32::INFINITY), y, overflow);
    y = v128_bitselect(f32x4_splat(0.0), y, underflow);
    y
}

#[cfg(target_arch = "wasm32")]
unsafe fn swiglu_simd128(gate: &[f32], up: &[f32], out: &mut [f32]) -> usize {
    use std::arch::wasm32::*;
    let n = gate.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let g = ptr::read_unaligned(gate.as_ptr().add(i).cast::<v128>());
        let u = ptr::read_unaligned(up.as_ptr().add(i).cast::<v128>());
        let e = expf4_simd128(f32x4_neg(g));
        let s = f32x4_div(g, f32x4_add(f32x4_splat(1.0), e));
        ptr::write_unaligned(out.as_mut_ptr().add(i).cast::<v128>(), f32x4_mul(u, s));
        i += 4;
    }
    i
}

#[cfg(target_arch = "wasm32")]
unsafe fn vmix_axpy_simd128(out: &mut [f32], a: f32, v: &[f32]) -> usize {
    use std::arch::wasm32::*;
    let n = out.len();
    let av = f32x4_splat(a);
    let mut i = 0usize;
    while i + 4 <= n {
        let o = ptr::read_unaligned(out.as_ptr().add(i).cast::<v128>());
        let vv = ptr::read_unaligned(v.as_ptr().add(i).cast::<v128>());
        let y = f32x4_add(f32x4_mul(av, vv), o);
        ptr::write_unaligned(out.as_mut_ptr().add(i).cast::<v128>(), y);
        i += 4;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn expf_inplace_avx(x: &mut [f32]) -> usize {
    use std::arch::x86_64::*;
    let n = x.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), expf8_avx(v));
        i += 8;
    }
    i
}

/// AVX2, **not** FMA — mul+add so the polynomial matches WASM SIMD128.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn expf8_avx(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let t = _mm256_mul_ps(x, _mm256_set1_ps(LOG2_E));
    let sign = _mm256_and_ps(t, _mm256_castsi256_ps(_mm256_set1_epi32(SIGN_BITS as i32)));
    let m = _mm256_or_ps(sign, _mm256_set1_ps(MAGIC));
    let n_f = _mm256_sub_ps(_mm256_add_ps(t, m), m);
    let r = _mm256_add_ps(
        _mm256_add_ps(x, _mm256_mul_ps(n_f, _mm256_set1_ps(-LN2_HI))),
        _mm256_mul_ps(n_f, _mm256_set1_ps(-LN2_LO)),
    );
    let mut u = _mm256_set1_ps(C0);
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(C1));
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(C2));
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(C3));
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(C4));
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(C5));
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(1.0));
    u = _mm256_add_ps(_mm256_mul_ps(u, r), _mm256_set1_ps(1.0));
    // Same ldexp_n as scalar: exponent add only when the result is normal.
    let n_i = _mm256_cvttps_epi32(n_f);
    let n = _mm256_slli_epi32(n_i, 23);
    let mut y = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(u), n));
    let exp_u = _mm256_srli_epi32(
        _mm256_and_si256(_mm256_castps_si256(u), _mm256_set1_epi32(0x7f80_0000u32 as i32)),
        23,
    );
    let e = _mm256_add_epi32(exp_u, n_i);
    let denorm = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e); // e < 1
    let ovf_exp = _mm256_cmpgt_epi32(e, _mm256_set1_epi32(254)); // e > 254
    y = _mm256_blendv_ps(y, _mm256_set1_ps(0.0), _mm256_castsi256_ps(denorm));
    y = _mm256_blendv_ps(y, _mm256_set1_ps(f32::INFINITY), _mm256_castsi256_ps(ovf_exp));
    let overflow = _mm256_cmp_ps(x, _mm256_set1_ps(OVERFLOW), _CMP_GT_OQ);
    let underflow = _mm256_cmp_ps(x, _mm256_set1_ps(UNDERFLOW), _CMP_LT_OQ);
    y = _mm256_blendv_ps(y, _mm256_set1_ps(f32::INFINITY), overflow);
    y = _mm256_blendv_ps(y, _mm256_set1_ps(0.0), underflow);
    y
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn swiglu_avx(gate: &[f32], up: &[f32], out: &mut [f32]) -> usize {
    use std::arch::x86_64::*;
    let n = gate.len();
    let mut i = 0usize;
    let one = _mm256_set1_ps(1.0);
    while i + 8 <= n {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u = _mm256_loadu_ps(up.as_ptr().add(i));
        let e = expf8_avx(_mm256_xor_ps(g, _mm256_set1_ps(-0.0)));
        let s = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(u, s));
        i += 8;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vmix_axpy_avx(out: &mut [f32], a: f32, v: &[f32]) -> usize {
    use std::arch::x86_64::*;
    let n = out.len();
    let av = _mm256_set1_ps(a);
    let mut i = 0usize;
    while i + 8 <= n {
        let o = _mm256_loadu_ps(out.as_ptr().add(i));
        let vv = _mm256_loadu_ps(v.as_ptr().add(i));
        let y = _mm256_add_ps(_mm256_mul_ps(av, vv), o);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), y);
        i += 8;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expf_shared_specials() {
        assert_eq!(expf_shared(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(expf_shared(-0.0).to_bits(), 1.0f32.to_bits());
        assert!(expf_shared(f32::NAN).is_nan());
        assert!(expf_shared(100.0).is_infinite());
        assert_eq!(expf_shared(-200.0), 0.0);
    }

    /// Bugbot #26: bit-ldexp with n ≤ -128 wrapped the sign bit. x in
    /// [-104, -88.4] must stay a tiny positive or 0 — never a huge
    /// negative / -Inf that softmax and silu would treat as a weight.
    #[test]
    fn expf_shared_deep_underflow_does_not_wrap_sign() {
        let mut x = -104.0f32;
        let mut xs = Vec::new();
        while x <= -88.4 {
            xs.push(x);
            let y = expf_shared(x);
            assert!(
                y.is_finite() && y >= 0.0,
                "x={x} y={y} bits={:#010x} (huge negative / -Inf wrap)",
                y.to_bits()
            );
            assert!(y < 1e-30, "x={x} y={y} expected tiny positive or 0");
            x += 0.0625;
        }
        // SIMD + tail must match the scalar flush (nomic silu hits this band).
        let mut got = xs.clone();
        expf_inplace(&mut got);
        for i in 0..xs.len() {
            assert_eq!(
                got[i].to_bits(),
                expf_shared(xs[i]).to_bits(),
                "simd lane {i} x={}",
                xs[i]
            );
            assert!(got[i].is_finite() && got[i] >= 0.0 && got[i] < 1e-30);
        }
        // n = -127 (just above -88.4) used to produce NaN for some Horner u.
        for x in [-88.3f32, -88.0, -87.5] {
            let y = expf_shared(x);
            assert!(
                y.is_finite() && y >= 0.0,
                "x={x} y={y} bits={:#010x}",
                y.to_bits()
            );
        }
    }

    #[test]
    fn expf_shared_within_a_few_ulps_of_libm() {
        let mut max_ulps = 0u32;
        let mut x = -80.0f32;
        while x <= 80.0 {
            let got = expf_shared(x);
            let want = x.exp();
            if got.is_finite() && want.is_finite() && got != 0.0 && want != 0.0 {
                let d = got.to_bits().abs_diff(want.to_bits());
                if d > max_ulps {
                    max_ulps = d;
                }
            }
            x += 0.125;
        }
        // A 1-ULP-class minimax; a few ULPs vs glibc is expected, not a gate.
        assert!(max_ulps <= 8, "max ulps vs libm expf = {max_ulps}");
    }

    #[test]
    fn expf_inplace_matches_scalar() {
        let mut xs = Vec::new();
        let mut x = -12.0f32;
        while x <= 8.0 {
            xs.push(x);
            x += 0.0625;
        }
        // lengths that hit SIMD + tail (3, 7, 16, 17)
        for n in [3usize, 7, 16, 17, xs.len()] {
            let mut got = xs[..n].to_vec();
            expf_inplace(&mut got);
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    expf_shared(xs[i]).to_bits(),
                    "lane {i} n={n} x={}",
                    xs[i]
                );
            }
        }
    }

    #[test]
    fn swiglu_matches_scalar_identity() {
        let gate: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let up: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let mut out = vec![0.0f32; 64];
        swiglu(&gate, &up, &mut out);
        for i in 0..64 {
            let expect = up[i] * (gate[i] / (1.0 + expf_shared(-gate[i])));
            assert_eq!(out[i].to_bits(), expect.to_bits(), "i={i}");
        }
    }

    #[test]
    fn vmix_axpy_is_mul_add_not_fma() {
        let a = f32::from_bits(0x3f2aaaab);
        let v = [f32::from_bits(0x40490fdb); 8];
        let acc = f32::from_bits(0x3f800001);
        let mut out = [acc; 8];
        vmix_axpy(&mut out, a, &v);
        let mul_add = a * v[0] + acc;
        assert_eq!(out[0].to_bits(), mul_add.to_bits());
        assert_eq!(out[7].to_bits(), mul_add.to_bits());
    }
}
