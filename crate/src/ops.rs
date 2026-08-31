//! Forward kernels. Match llama.cpp ggml ops (norm, silu, softmax, NEOX RoPE).
//!
//! These are the compute path. The oracle stays llama.cpp. Unverified
//! activations / pooling types refuse in `model.rs`, not here.

/// ggml `ggml_compute_forward_norm_f32` + affine: (x-mean)/sqrt(var+eps) * w + b.
/// Sums use f64 (`ggml_float`). Variance divides by n, not n-1.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), bias.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();
    let mut sum = 0.0f64;
    for &v in x {
        sum += f64::from(v);
    }
    let mean = (sum / n as f64) as f32;
    let mut sum2 = 0.0f64;
    for i in 0..n {
        let v = x[i] - mean;
        out[i] = v;
        sum2 += f64::from(v * v);
    }
    let variance = (sum2 / n as f64) as f32;
    let scale = 1.0 / (variance + eps).sqrt();
    for i in 0..n {
        out[i] = out[i] * scale * weight[i] + bias[i];
    }
}

/// ggml `ggml_silu_f32`: x / (1 + exp(-x)). Scalar tail / tests.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// llama.cpp `ggml_vec_swiglu_f32`: `up * silu(gate)`.
/// AVX2+FMA uses `ggml_v_silu` / `ggml_v_expf` (not libm `expf`).
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { swiglu_avx2(gate, up, out) };
            return;
        }
    }
    for i in 0..out.len() {
        out[i] = up[i] * silu(gate[i]);
    }
}

/// llama.cpp `ggml_v_expf` + `ggml_v_silu` (`vec.h` AVX2+FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let mut i = 0usize;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    while i + 8 <= n {
        let x = _mm256_loadu_ps(gate.as_ptr().add(i));
        let g = _mm256_loadu_ps(up.as_ptr().add(i));
        let silu = _mm256_div_ps(x, _mm256_add_ps(one, ggml_v_expf(_mm256_sub_ps(zero, x))));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(silu, g));
        i += 8;
    }
    for j in i..n {
        out[j] = up[j] * silu(gate[j]);
    }
}

/// llama.cpp `ggml_v_expf` (`vec.h`, AVX2+FMA). Max error ~1.45 + 0.5 ulp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ggml_v_expf(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let r = _mm256_set1_ps(f32::from_bits(0x4b40_0000)); // 0x1.8p23
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3fb8_aa3b)), r); // 0x1.715476p+0
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(f32::from_bits(0x35bf_be8e)), // 0x1.7f7d1cp-20
        _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3f31_7200)), x), // 0x1.62e4p-1
    );
    let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
    let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0))));
    let c = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(126.0),
    ));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3c07_2010)), // 0x1.0e4020p-7
                b,
                _mm256_set1_ps(f32::from_bits(0x3d2b_9f17)), // 0x1.573e2ep-5
            ),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3e2a_af33)), // 0x1.555e66p-3
                b,
                _mm256_set1_ps(f32::from_bits(0x3eff_fedb)), // 0x1.fffdb6p-2
            ),
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7f_fff6)), b), // 0x1.ffffecp-1
    );
    if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps())),
        _mm256_set1_epi32(0x8200_0000u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000u32 as i32)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(192.0),
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1),
                ),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

/// x: [n_tokens, n_in], w: [n_out, n_in] row-major (GGUF [n_in, n_out] columns).
/// y: [n_tokens, n_out]
pub fn matmul(x: &[f32], w: &[f32], n_tokens: usize, n_in: usize, n_out: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), n_tokens * n_in);
    debug_assert_eq!(w.len(), n_out * n_in);
    debug_assert_eq!(y.len(), n_tokens * n_out);
    for t in 0..n_tokens {
        let xrow = &x[t * n_in..(t + 1) * n_in];
        for o in 0..n_out {
            let wrow = &w[o * n_in..(o + 1) * n_in];
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += xrow[i] * wrow[i];
            }
            y[t * n_out + o] = acc;
        }
    }
}

/// ggml softmax: max-subtract, exp, f64 sum, scale.
pub fn softmax_inplace(logits: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for v in logits.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += f64::from(e);
    }
    let inv = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

/// llama.cpp `LLAMA_ROPE_TYPE_NEOX` + `ggml_rope_cache_init` with ext_factor=0,
/// freq_scale=1, attn_factor=1 (no YaRN). Pairs (x[i], x[i + n_dims/2]).
pub fn rope_neox_inplace(x: &mut [f32], n_tokens: usize, n_heads: usize, head_dim: usize, freq_base: f32) {
    debug_assert_eq!(x.len(), n_tokens * n_heads * head_dim);
    debug_assert!(head_dim % 2 == 0);
    let half = head_dim / 2;
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut cache = vec![0.0f32; head_dim];
    for t in 0..n_tokens {
        let mut theta = t as f32;
        for i in 0..half {
            cache[2 * i] = theta.cos();
            cache[2 * i + 1] = theta.sin();
            theta *= theta_scale;
        }
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                let cos_t = cache[2 * i];
                let sin_t = cache[2 * i + 1];
                x[base + i] = x0 * cos_t - x1 * sin_t;
                x[base + i + half] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}

/// GPT-J / NORM RoPE: consecutive pairs (x[2i], x[2i+1]). Probe only.
pub fn rope_norm_inplace(x: &mut [f32], n_tokens: usize, n_heads: usize, head_dim: usize, freq_base: f32) {
    debug_assert_eq!(x.len(), n_tokens * n_heads * head_dim);
    debug_assert!(head_dim % 2 == 0);
    let half = head_dim / 2;
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut cache = vec![0.0f32; head_dim];
    for t in 0..n_tokens {
        let mut theta = t as f32;
        for i in 0..half {
            cache[2 * i] = theta.cos();
            cache[2 * i + 1] = theta.sin();
            theta *= theta_scale;
        }
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let x0 = x[base + 2 * i];
                let x1 = x[base + 2 * i + 1];
                let cos_t = cache[2 * i];
                let sin_t = cache[2 * i + 1];
                x[base + 2 * i] = x0 * cos_t - x1 * sin_t;
                x[base + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}

/// Bidirectional attention. Q/K/V are [n_tokens, n_heads, head_dim].
/// Out is [n_tokens, n_embd] with heads concatenated (token-major).
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; n_tokens];
    for h in 0..n_heads {
        for tq in 0..n_tokens {
            let qoff = (tq * n_heads + h) * head_dim;
            let qh = &q[qoff..qoff + head_dim];
            for tk in 0..n_tokens {
                let koff = (tk * n_heads + h) * head_dim;
                let kh = &k[koff..koff + head_dim];
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += qh[i] * kh[i];
                }
                scores[tk] = dot * scale;
            }
            softmax_inplace(&mut scores);
            let ooff = tq * n_heads * head_dim + h * head_dim;
            for i in 0..head_dim {
                out[ooff + i] = 0.0;
            }
            for tk in 0..n_tokens {
                let a = scores[tk];
                let voff = (tk * n_heads + h) * head_dim;
                for i in 0..head_dim {
                    out[ooff + i] += a * v[voff + i];
                }
            }
        }
    }
}

/// Mean over the token axis. `x` is [n_tokens, n_embd].
pub fn mean_pool(x: &[f32], n_tokens: usize, n_embd: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), n_tokens * n_embd);
    debug_assert_eq!(out.len(), n_embd);
    out.fill(0.0);
    if n_tokens == 0 {
        return;
    }
    for t in 0..n_tokens {
        let row = &x[t * n_embd..(t + 1) * n_embd];
        for i in 0..n_embd {
            out[i] += row[i];
        }
    }
    let inv = 1.0 / n_tokens as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// CLS = first token. Must-fail / GGUF pooling_type=2.
pub fn cls_pool(x: &[f32], n_embd: usize, out: &mut [f32]) {
    debug_assert!(x.len() >= n_embd);
    debug_assert_eq!(out.len(), n_embd);
    out.copy_from_slice(&x[..n_embd]);
}

/// llama.cpp `--embd-normalize 2` / `common_embd_normalize` (euclidean, f64 sum).
pub fn l2_normalize_inplace(x: &mut [f32]) {
    let mut sum = 0.0f64;
    for &v in x.iter() {
        sum += f64::from(v) * f64::from(v);
    }
    let norm = sum.sqrt();
    let scale = if norm > 0.0 { (1.0 / norm) as f32 } else { 0.0 };
    for v in x.iter_mut() {
        *v *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_matches_ggml() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        let x = 1.0f32;
        let expect = x / (1.0 + (-x).exp());
        assert!((silu(x) - expect).abs() < 1e-7);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        rope_neox_inplace(&mut x, 1, 1, 4, 1000.0);
        assert_eq!(x, orig);
    }

    #[test]
    fn mean_pool_averages_tokens() {
        let x = [1.0, 3.0, 5.0, 7.0];
        let mut out = [0.0; 2];
        mean_pool(&x, 2, 2, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn l2_unit_length() {
        let mut x = [3.0f32, 4.0];
        l2_normalize_inplace(&mut x);
        assert!((x[0] - 0.6).abs() < 1e-6);
        assert!((x[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}
