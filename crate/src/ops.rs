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

/// ggml `ggml_silu_f32`: x / (1 + exp(-x)).
///
/// AVX2 `ggml_vec_silu_f32` / `ggml_v_expf` is a different numeric path:
/// it broke `empty-none` vs the pinned goldens (cos_dist 0 → 0.025).
/// Stay on libm. Do not re-enable vec-silu without an `empty-none` receipt.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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
///
/// Cache matches C libm `powf`/`cosf`/`sinf` (Rust `powf`/`cos`/`sin` is
/// bit-exact vs those on this toolchain). Token 0 is identity. The first
/// bit miss vs the eval-callback dump was the apply: ggml's AVX2 build
/// contracts
///   `x0*cos - x1*sin` → `fmaf(x0, cos, -(x1*sin))`
///   `x0*sin + x1*cos` → `fmaf(x0, sin,  x1*cos)`
/// The other order (`fmaf(-x1, sin, x0*cos)`) disagrees with the dump.
/// Do not revert to mul+sub.
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
                x[base + i] = x0.mul_add(cos_t, -(x1 * sin_t));
                x[base + i + half] = x0.mul_add(sin_t, x1 * cos_t);
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
///
/// Q@K stays serial f32. `wqkv-0` Q5_K is bit-exact vs the dump after
/// matching DSO `vfmadd231ss` on `summs`. `Qcur-0`/`Kcur-0` RoPE is
/// bit-exact after matching ggml's apply FMA
/// (`fmaf(x0,c,-(x1*s))` / `fmaf(x0,s,x1*c)`). `Vcur-0` is bit-exact.
/// First remaining dump DIFF is `kq-0` serial f32 (document 6.10e-5 /
/// empty-none 1.53e-5). On bit-exact `Qcur-0`/`Kcur-0` the first miss
/// is serial `dot += q*k` vs ggml's 4×8 `mul_add` + pairwise hadd
/// (`GGML_F32_STEP=32`). That tree, AVX2 intrinsics, and DSO
/// `ggml_vec_dot_f32` are BIT_EXACT vs the `kq` dump (n=2 and n=7).
/// Porting the tree (portable or AVX2) avalanches `empty-none`:
/// expected PASS cos_dist≈0 max_abs=7.86e-8; got cos_dist=0.03114
/// max_abs=0.03411 d0 0.00697820 vs 0.02447182. Keep serial Q@K.
/// Do not land `2d36deb`. Do not dispatch AVX2 `ggml_vec_dot_f32`.
#[allow(dead_code)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    attention_named(q, k, v, n_tokens, n_heads, head_dim, out, None);
}

/// Same as `attention`, with optional dump tag (`MILTON_DUMP=1` writes kq / softmax).
pub fn attention_named(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
    dump_tag: Option<&str>,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; n_tokens];
    let dump = dump_tag.is_some() && std::env::var("MILTON_DUMP").ok().as_deref() == Some("1");
    let mut kq_raw = if dump {
        vec![0.0f32; n_heads * n_tokens * n_tokens]
    } else {
        Vec::new()
    };
    let mut kq_sm = if dump {
        vec![0.0f32; n_heads * n_tokens * n_tokens]
    } else {
        Vec::new()
    };
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
                if dump {
                    // llama kq is {n_kv, n_q, n_heads} = tk + tq*n + h*n*n
                    kq_raw[tk + tq * n_tokens + h * n_tokens * n_tokens] = dot;
                }
            }
            softmax_inplace(&mut scores);
            if dump {
                for tk in 0..n_tokens {
                    kq_sm[tk + tq * n_tokens + h * n_tokens * n_tokens] = scores[tk];
                }
            }
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
    if let Some(tag) = dump_tag {
        if dump {
            dump_f32(&format!("kq-{tag}"), &kq_raw);
            dump_f32(&format!("kq_soft_max-{tag}"), &kq_sm);
        }
    }
}

fn dump_f32(name: &str, x: &[f32]) {
    let path = format!("/tmp/ml-{name}.f32");
    let mut bytes = Vec::with_capacity(8 + x.len() * 4);
    bytes.extend_from_slice(&(x.len() as i64).to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&0i64.to_le_bytes());
    for &v in x {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(&path, bytes);
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
    fn rope_apply_uses_ggml_fma_pairing() {
        // Dump pair t=1 h=0 i=2 on short-hello-document. ggml contracts
        // `x0*c - x1*s` as fmaf(x0, c, -(x1*s)), not fmaf(-x1, s, x0*c).
        let x0 = f32::from_bits(0xbfc83027); // -0x1.90604ep+0
        let x1 = f32::from_bits(0xbef72f7c); // -0x1.ee5ef8p-2
        let c = f32::from_bits(0x3f4be4aa); //  0x1.97c954p-1
        let s = f32::from_bits(0x3f1acd39); //  0x1.359a72p-1
        let y0 = x0.mul_add(c, -(x1 * s));
        assert_eq!(y0.to_bits(), 0xbf7425a1); // -0x1.e84b42p-1 (dump)
        assert_ne!(y0.to_bits(), (x0 * c - x1 * s).to_bits());
        assert_ne!(y0.to_bits(), (-x1).mul_add(s, x0 * c).to_bits());
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
