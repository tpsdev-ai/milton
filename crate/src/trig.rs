//! Shared permissive f32 `sin`/`cos` for native and WASM.
//!
//! From-scratch Cody-Waite range reduction + Horner minimax (SLEEF `xsincosf`
//! coefficients, BSL-1.0; coefficients themselves are not copyrightable).
//! Evaluation is **mul+add, not FMA** on every backend — same discipline as
//! `crate/src/exp.rs`. Same polynomial, same op order, same rounding ⇒
//! native vs WASM bit-identical by construction.
//!
//! Do not call `f32::sin` / `f32::cos` on the embed path: native is glibc
//! `sinf`/`cosf`, WASM is compiler-builtins, and they split at 1 ULP
//! (issue #26 residual at Qcur-0 t=6 i=14: 0x3E937217 vs 0x3E937216).
//! Do not copy glibc (LGPL). Do not swap in musl / `libm::sinf` / `libm::cosf`.
//!
//! # Domain
//!
//! RoPE angles are `θ_i = pos × θ_scale^i` with `pos < context_length` and
//! `0 < θ_scale < 1`, so `|θ| ≤ pos < context_length`. nomic-embed-text-v1.5
//! stores `nomic-bert.context_length = 2048`. The 4-part Cody-Waite reduction
//! (SLEEF `PI_A/B/C/Df`) is valid for `|x| < 39000` (`TRIGRANGEMAXf`). We
//! assert the RoPE bound (`|θ| ≤ 2048`) and refuse the magic-rint bit-trick
//! when `|x · 2/π| ≥ 2^23` (the encoding is only an integer rounding trick
//! inside that range — wrapping it would silently return the wrong quadrant).

#[cfg(target_arch = "wasm32")]
use std::ptr;

/// `2/π` as f32 (SLEEF `M_2_PI`).
const INV_2PI: f32 = f32::from_bits(0x3f22f983);
/// Cody-Waite `π` parts ÷ 2 (SLEEF `PI_Af..PI_Df` × 0.5). Last bits of A/B/C
/// are zero so `q * PI_*` stays exact for the integer `q` we produce.
const PI_A_HALF: f32 = f32::from_bits(0x3fc90000);
const PI_B_HALF: f32 = f32::from_bits(0x39fd8000);
const PI_C_HALF: f32 = f32::from_bits(0x34a88000);
const PI_D_HALF: f32 = f32::from_bits(0x2e85a309);
/// SLEEF `xsincosf` sin (odd) Horner coefficients.
const SINC_C0: f32 = f32::from_bits(0xb94ca65b);
const SINC_C1: f32 = f32::from_bits(0x3c08839a);
const SINC_C2: f32 = f32::from_bits(0xbe2aaaa2);
/// SLEEF `xsincosf` cos (even) Horner coefficients.
const COSC_C0: f32 = f32::from_bits(0xb491ed89);
const COSC_C1: f32 = f32::from_bits(0x37d0078b);
const COSC_C2: f32 = f32::from_bits(0xbab60b58);
const COSC_C3: f32 = f32::from_bits(0x3d2aaaaa);
const COSC_C4: f32 = -0.5;
/// nomic-embed-text-v1.5 `context_length`. Max RoPE `|θ|` is `pos < 2048`.
pub const ROPE_THETA_MAX: f32 = 2048.0;
/// 4-part Cody-Waite is valid below SLEEF `TRIGRANGEMAXf`.
const REDUCTION_VALID: f32 = 39000.0;
/// `2^23` — magic for round-ties-to-even on `|x| < 2^23`.
const MAGIC: f32 = 8388608.0;
const SIGN_BITS: u32 = 0x8000_0000;
/// Magic rint is only an integer trick while `|t| < 2^23`.
const RINT_LIMIT: f32 = MAGIC;

/// Round `x` to nearest integer, ties to even. Domain `|x| < 2^23`.
///
/// This is a bit-trick on the exponent field (add `2^23` so the ulp is 1).
/// Callers must check `|x| < 2^23` — wrapping the add would pick the wrong
/// quadrant. See `sincosf_shared_rint_domain_refuses_wrap`.
#[inline(always)]
fn rint_ne(x: f32) -> f32 {
    let magic = f32::from_bits(MAGIC.to_bits() | (x.to_bits() & SIGN_BITS));
    (x + magic) - magic
}

/// `freq_base^(-2 / head_dim)` without libm `powf`.
///
/// For nomic, `head_dim = 64` so `head_dim/2 = 32 = 2^5`: five IEEE `sqrt`
/// then a reciprocal. `sqrt` and `/` are correctly rounded and bit-identical
/// on native and WASM. `f32::powf` is glibc vs compiler-builtins — it happened
/// to match on this pair (luck, not a guarantee).
///
/// Fail-closed unless `head_dim/2` is a power of two (v1 is nomic only).
pub fn rope_theta_scale(freq_base: f32, head_dim: usize) -> f32 {
    assert!(
        freq_base.is_finite() && freq_base > 0.0,
        "fail-closed: rope freq_base must be a positive finite (got {freq_base})"
    );
    assert!(
        head_dim >= 2 && head_dim % 2 == 0,
        "fail-closed: rope head_dim must be even and ≥ 2 (got {head_dim})"
    );
    let half = head_dim / 2;
    assert!(
        half.is_power_of_two(),
        "fail-closed: theta_scale via IEEE sqrt requires head_dim/2 to be a power of two (got {head_dim})"
    );
    let n = half.trailing_zeros();
    let mut x = freq_base;
    for _ in 0..n {
        x = x.sqrt();
    }
    1.0 / x
}

/// Test-only: put libm `sinf`/`cosf` back on the **native** backend.
///
/// `MILTON_ROPE_LIBM_SIN=1` on native `milton-embed` turns `wasm:compare`
/// RED (one backend on glibc, the other on the shared kernel). WASM never
/// honours the switch — a gate nobody has seen fail is not a gate.
#[inline]
pub fn rope_use_libm_sin() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        return *ON.get_or_init(|| match std::env::var("MILTON_ROPE_LIBM_SIN") {
            Ok(v) => v == "1",
            Err(_) => false,
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// Shared f32 `sin`/`cos`. Mul+add Horner, magic rint, 4-part Cody-Waite.
/// `inline(never)` so a small RoPE loop cannot be auto-vectorized into a
/// different tree than the explicit SIMD kernels (native-vs-WASM split).
#[inline(never)]
pub fn sincosf_shared(x: f32) -> (f32, f32) {
    if x.is_nan() {
        return (x, x);
    }
    let t = x * INV_2PI;
    // Magic rint is a bit-trick on the exponent. Refuse — never wrap.
    if !t.is_finite() || t.abs() >= RINT_LIMIT {
        return (f32::NAN, f32::NAN);
    }
    if x.abs() >= REDUCTION_VALID {
        return (f32::NAN, f32::NAN);
    }
    let n_f = rint_ne(t);
    let n = n_f as i32;
    // r = x - n*π/2 as four mul+add steps (not FMA).
    let r = (((x + n_f * (-PI_A_HALF)) + n_f * (-PI_B_HALF)) + n_f * (-PI_C_HALF))
        + n_f * (-PI_D_HALF);
    let s = r * r;
    let mut us = SINC_C0;
    us = us * s + SINC_C1;
    us = us * s + SINC_C2;
    us = (us * s) * r;
    let mut sn = r + us;
    let mut uc = COSC_C0;
    uc = uc * s + COSC_C1;
    uc = uc * s + COSC_C2;
    uc = uc * s + COSC_C3;
    uc = uc * s + COSC_C4;
    let mut cs = uc * s + 1.0;
    if (n & 1) != 0 {
        core::mem::swap(&mut sn, &mut cs);
    }
    if (n & 2) != 0 {
        sn = -sn;
    }
    if ((n + 1) & 2) != 0 {
        cs = -cs;
    }
    if x.to_bits() == (-0.0f32).to_bits() {
        sn = -0.0;
    }
    (sn, cs)
}

/// Shared f32 `sin`. Same kernel as `sincosf_shared`.
#[inline]
pub fn sinf_shared(x: f32) -> f32 {
    sincosf_shared(x).0
}

/// Shared f32 `cos`. Same kernel as `sincosf_shared`.
#[inline]
pub fn cosf_shared(x: f32) -> f32 {
    sincosf_shared(x).1
}

/// `sins[i], coses[i] = sincos(thetas[i])`. SIMD 4-wide (WASM) / 8-wide (AVX2),
/// scalar tail. Each lane is the scalar polynomial, so SIMD and scalar are
/// bit-identical. Scalar tail stays *outside* `#[target_feature(enable = "avx2")]`
/// so LLVM cannot contract the Horner into FMA (which WASM SIMD128 cannot match).
pub fn sincosf_inplace(thetas: &[f32], sins: &mut [f32], coses: &mut [f32]) {
    debug_assert_eq!(thetas.len(), sins.len());
    debug_assert_eq!(thetas.len(), coses.len());
    #[cfg(target_arch = "wasm32")]
    let i = unsafe { sincosf_inplace_simd128(thetas, sins, coses) };
    #[cfg(target_arch = "x86_64")]
    let i = if is_x86_feature_detected!("avx2") {
        unsafe { sincosf_inplace_avx(thetas, sins, coses) }
    } else {
        0
    };
    #[cfg(not(any(target_arch = "wasm32", target_arch = "x86_64")))]
    let i = 0usize;
    for j in i..thetas.len() {
        let (s, c) = sincosf_shared(thetas[j]);
        sins[j] = s;
        coses[j] = c;
    }
}

#[cfg(target_arch = "wasm32")]
unsafe fn sincosf_inplace_simd128(
    thetas: &[f32],
    sins: &mut [f32],
    coses: &mut [f32],
) -> usize {
    use std::arch::wasm32::*;
    let n = thetas.len();
    let mut i = 0usize;
    while i + 4 <= n {
        let v = ptr::read_unaligned(thetas.as_ptr().add(i).cast::<v128>());
        let (sn, cs) = sincos4_simd128(v);
        ptr::write_unaligned(sins.as_mut_ptr().add(i).cast::<v128>(), sn);
        ptr::write_unaligned(coses.as_mut_ptr().add(i).cast::<v128>(), cs);
        i += 4;
    }
    i
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sincos4_simd128(x: std::arch::wasm32::v128) -> (std::arch::wasm32::v128, std::arch::wasm32::v128) {
    use std::arch::wasm32::*;
    let t = f32x4_mul(x, f32x4_splat(INV_2PI));
    let sign = v128_and(t, i32x4_splat(SIGN_BITS as i32));
    let m = v128_or(sign, f32x4_splat(MAGIC));
    let n_f = f32x4_sub(f32x4_add(t, m), m);
    let r = f32x4_add(
        f32x4_add(
            f32x4_add(
                f32x4_add(x, f32x4_mul(n_f, f32x4_splat(-PI_A_HALF))),
                f32x4_mul(n_f, f32x4_splat(-PI_B_HALF)),
            ),
            f32x4_mul(n_f, f32x4_splat(-PI_C_HALF)),
        ),
        f32x4_mul(n_f, f32x4_splat(-PI_D_HALF)),
    );
    let s = f32x4_mul(r, r);
    let mut us = f32x4_splat(SINC_C0);
    us = f32x4_add(f32x4_mul(us, s), f32x4_splat(SINC_C1));
    us = f32x4_add(f32x4_mul(us, s), f32x4_splat(SINC_C2));
    us = f32x4_mul(f32x4_mul(us, s), r);
    let mut sn = f32x4_add(r, us);
    let mut uc = f32x4_splat(COSC_C0);
    uc = f32x4_add(f32x4_mul(uc, s), f32x4_splat(COSC_C1));
    uc = f32x4_add(f32x4_mul(uc, s), f32x4_splat(COSC_C2));
    uc = f32x4_add(f32x4_mul(uc, s), f32x4_splat(COSC_C3));
    uc = f32x4_add(f32x4_mul(uc, s), f32x4_splat(COSC_C4));
    let mut cs = f32x4_add(f32x4_mul(uc, s), f32x4_splat(1.0));
    let n_i = i32x4_trunc_sat_f32x4(n_f);
    let odd = i32x4_ne(v128_and(n_i, i32x4_splat(1)), i32x4_splat(0));
    let sn_keep = sn;
    sn = v128_bitselect(cs, sn, odd);
    cs = v128_bitselect(sn_keep, cs, odd);
    let flip_sin = i32x4_ne(v128_and(n_i, i32x4_splat(2)), i32x4_splat(0));
    let flip_cos = i32x4_ne(
        v128_and(i32x4_add(n_i, i32x4_splat(1)), i32x4_splat(2)),
        i32x4_splat(0),
    );
    let signbit = f32x4_splat(-0.0);
    sn = v128_bitselect(v128_xor(sn, signbit), sn, flip_sin);
    cs = v128_bitselect(v128_xor(cs, signbit), cs, flip_cos);
    // Refuse lanes where the magic-rint domain does not hold.
    let bad = v128_or(
        v128_or(f32x4_ne(t, t), f32x4_ge(f32x4_abs(t), f32x4_splat(RINT_LIMIT))),
        f32x4_ge(f32x4_abs(x), f32x4_splat(REDUCTION_VALID)),
    );
    let nan = f32x4_splat(f32::NAN);
    sn = v128_bitselect(nan, sn, bad);
    cs = v128_bitselect(nan, cs, bad);
    (sn, cs)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sincosf_inplace_avx(thetas: &[f32], sins: &mut [f32], coses: &mut [f32]) -> usize {
    use std::arch::x86_64::*;
    let n = thetas.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(thetas.as_ptr().add(i));
        let (sn, cs) = sincos8_avx(v);
        _mm256_storeu_ps(sins.as_mut_ptr().add(i), sn);
        _mm256_storeu_ps(coses.as_mut_ptr().add(i), cs);
        i += 8;
    }
    i
}

/// AVX2, **not** FMA — mul+add so the polynomial matches WASM SIMD128.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sincos8_avx(x: std::arch::x86_64::__m256) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;
    let t = _mm256_mul_ps(x, _mm256_set1_ps(INV_2PI));
    let sign = _mm256_and_ps(t, _mm256_castsi256_ps(_mm256_set1_epi32(SIGN_BITS as i32)));
    let m = _mm256_or_ps(sign, _mm256_set1_ps(MAGIC));
    let n_f = _mm256_sub_ps(_mm256_add_ps(t, m), m);
    let r = _mm256_add_ps(
        _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(x, _mm256_mul_ps(n_f, _mm256_set1_ps(-PI_A_HALF))),
                _mm256_mul_ps(n_f, _mm256_set1_ps(-PI_B_HALF)),
            ),
            _mm256_mul_ps(n_f, _mm256_set1_ps(-PI_C_HALF)),
        ),
        _mm256_mul_ps(n_f, _mm256_set1_ps(-PI_D_HALF)),
    );
    let s = _mm256_mul_ps(r, r);
    let mut us = _mm256_set1_ps(SINC_C0);
    us = _mm256_add_ps(_mm256_mul_ps(us, s), _mm256_set1_ps(SINC_C1));
    us = _mm256_add_ps(_mm256_mul_ps(us, s), _mm256_set1_ps(SINC_C2));
    us = _mm256_mul_ps(_mm256_mul_ps(us, s), r);
    let mut sn = _mm256_add_ps(r, us);
    let mut uc = _mm256_set1_ps(COSC_C0);
    uc = _mm256_add_ps(_mm256_mul_ps(uc, s), _mm256_set1_ps(COSC_C1));
    uc = _mm256_add_ps(_mm256_mul_ps(uc, s), _mm256_set1_ps(COSC_C2));
    uc = _mm256_add_ps(_mm256_mul_ps(uc, s), _mm256_set1_ps(COSC_C3));
    uc = _mm256_add_ps(_mm256_mul_ps(uc, s), _mm256_set1_ps(COSC_C4));
    let mut cs = _mm256_add_ps(_mm256_mul_ps(uc, s), _mm256_set1_ps(1.0));
    let n_i = _mm256_cvttps_epi32(n_f);
    let odd = _mm256_cmpgt_epi32(_mm256_and_si256(n_i, _mm256_set1_epi32(1)), _mm256_set1_epi32(0));
    let odd_ps = _mm256_castsi256_ps(odd);
    let sn_keep = sn;
    sn = _mm256_blendv_ps(sn, cs, odd_ps);
    cs = _mm256_blendv_ps(cs, sn_keep, odd_ps);
    let flip_sin = _mm256_cmpgt_epi32(_mm256_and_si256(n_i, _mm256_set1_epi32(2)), _mm256_set1_epi32(0));
    let flip_cos = _mm256_cmpgt_epi32(
        _mm256_and_si256(_mm256_add_epi32(n_i, _mm256_set1_epi32(1)), _mm256_set1_epi32(2)),
        _mm256_set1_epi32(0),
    );
    let signbit = _mm256_set1_ps(-0.0);
    sn = _mm256_xor_ps(sn, _mm256_and_ps(signbit, _mm256_castsi256_ps(flip_sin)));
    cs = _mm256_xor_ps(cs, _mm256_and_ps(signbit, _mm256_castsi256_ps(flip_cos)));
    let bad = _mm256_or_ps(
        _mm256_or_ps(
            _mm256_cmp_ps(t, t, _CMP_UNORD_Q),
            _mm256_cmp_ps(_mm256_andnot_ps(signbit, t), _mm256_set1_ps(RINT_LIMIT), _CMP_GE_OQ),
        ),
        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x), _mm256_set1_ps(REDUCTION_VALID), _CMP_GE_OQ),
    );
    let nan = _mm256_set1_ps(f32::NAN);
    sn = _mm256_blendv_ps(sn, nan, bad);
    cs = _mm256_blendv_ps(cs, nan, bad);
    (sn, cs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincosf_shared_specials() {
        let (s, c) = sincosf_shared(0.0);
        assert_eq!(s.to_bits(), 0.0f32.to_bits());
        assert_eq!(c.to_bits(), 1.0f32.to_bits());
        assert_eq!(sinf_shared(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(cosf_shared(0.0).to_bits(), 1.0f32.to_bits());
        let (s, c) = sincosf_shared(-0.0);
        assert_eq!(s.to_bits(), (-0.0f32).to_bits());
        assert_eq!(c.to_bits(), 1.0f32.to_bits());
        let (s, c) = sincosf_shared(f32::NAN);
        assert!(s.is_nan() && c.is_nan());
    }

    #[test]
    fn sincosf_shared_unit_circle() {
        let mut x = -8.0f32;
        while x <= 8.0 {
            let (s, c) = sincosf_shared(x);
            let n = s * s + c * c;
            assert!((n - 1.0).abs() < 2e-6, "x={x} sin²+cos²={n}");
            x += 0.0625;
        }
    }

    /// Magic rint is only valid for `|x · 2/π| < 2^23`. Crossing that bound
    /// used to wrap the exponent trick and pick a garbage quadrant. Refuse.
    #[test]
    fn sincosf_shared_rint_domain_refuses_wrap() {
        // RoPE |θ|≤2048 is far inside both the 4-part reduction (|x|<39000)
        // and the magic-rint bit-trick (|x·2/π| < 2^23 ≈ 1.32e7).
        assert!(ROPE_THETA_MAX < REDUCTION_VALID);
        assert!(ROPE_THETA_MAX * INV_2PI < RINT_LIMIT);
        let (s, c) = sincosf_shared(ROPE_THETA_MAX);
        assert!(s.is_finite() && c.is_finite(), "RoPE domain edge must stay finite");
        // rustc 1.83 has no f32::next_down; 1 ulp below 39000 via bits.
        let just_inside = f32::from_bits(REDUCTION_VALID.to_bits() - 1);
        let (s, c) = sincosf_shared(just_inside);
        assert!(
            s.is_finite() && c.is_finite(),
            "just inside TRIGRANGEMAXf must stay finite"
        );
        // At / above the reduction or rint limit: NaN, never a wrapped quadrant.
        let rint_x = RINT_LIMIT / INV_2PI;
        for x in [REDUCTION_VALID, rint_x, rint_x * 1.01, f32::INFINITY, f32::NEG_INFINITY] {
            let (s, c) = sincosf_shared(x);
            assert!(
                s.is_nan() && c.is_nan(),
                "x={x} must refuse wrap, got sin={s} cos={c}"
            );
        }
    }

    #[test]
    fn sincosf_shared_within_a_few_ulps_of_libm() {
        let mut max_ulps = 0u32;
        let mut x = -ROPE_THETA_MAX;
        while x <= ROPE_THETA_MAX {
            let (gs, gc) = sincosf_shared(x);
            let ws = x.sin();
            let wc = x.cos();
            if gs.is_finite() && ws.is_finite() {
                max_ulps = max_ulps.max(gs.to_bits().abs_diff(ws.to_bits()));
            }
            if gc.is_finite() && wc.is_finite() {
                max_ulps = max_ulps.max(gc.to_bits().abs_diff(wc.to_bits()));
            }
            x += 0.5;
        }
        // A 1-ULP-class minimax; a few ULPs vs glibc is expected, not a gate.
        assert!(max_ulps <= 16, "max ulps vs libm sinf/cosf = {max_ulps}");
    }

    #[test]
    fn sincosf_inplace_matches_scalar() {
        let mut xs = Vec::new();
        let mut x = -12.0f32;
        while x <= 12.0 {
            xs.push(x);
            x += 0.125;
        }
        for n in [3usize, 7, 16, 17, 32, xs.len()] {
            let mut sins = vec![0.0f32; n];
            let mut coses = vec![0.0f32; n];
            sincosf_inplace(&xs[..n], &mut sins, &mut coses);
            for i in 0..n {
                let (s, c) = sincosf_shared(xs[i]);
                assert_eq!(sins[i].to_bits(), s.to_bits(), "sin lane {i} n={n} x={}", xs[i]);
                assert_eq!(coses[i].to_bits(), c.to_bits(), "cos lane {i} n={n} x={}", xs[i]);
            }
        }
    }

    #[test]
    fn rope_theta_scale_nomic_is_ieee_sqrt_chain() {
        let got = rope_theta_scale(1000.0, 64);
        let mut x = 1000.0f32;
        for _ in 0..5 {
            x = x.sqrt();
        }
        let want = 1.0 / x;
        assert_eq!(got.to_bits(), want.to_bits());
        assert!(got > 0.0 && got < 1.0);
        // Distinct from libm powf is allowed (1 ULP on this pair); identity
        // across backends is the requirement.
        let libm = 1000.0f32.powf(-2.0 / 64.0);
        let d = got.to_bits().abs_diff(libm.to_bits());
        assert!(d <= 2, "theta_scale vs libm powf ulps={d}");
    }

    /// The #26 residual: glibc `sinf` vs WASM compiler-builtins at
    /// `θ = 6 · θ_scale^14`. Shared sin is the same bits on both backends;
    /// libm on *one* backend is the test-only switch that must turn the
    /// compare RED.
    #[test]
    fn libm_sinf_diverges_from_shared_on_rope_grid() {
        let scale = rope_theta_scale(1000.0, 64);
        let mut mismatches = 0u32;
        for t in 0..32u32 {
            let mut theta = t as f32;
            for _i in 0..32 {
                let shared = sinf_shared(theta);
                let libm = theta.sin();
                if shared.to_bits() != libm.to_bits() {
                    mismatches += 1;
                }
                theta *= scale;
            }
        }
        assert!(
            mismatches > 0,
            "libm sinf matched shared on the RoPE grid — the test-only switch would be a no-op"
        );
    }
}
