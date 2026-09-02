# Stage profile tables (issue #28)

Host: 4-core Intel Xeon (KVM), AVX2+FMA+AVX512, 2.4 GHz assumed for roofline.
Roofline: AVX2 76.8 GFLOP/s (2 FMA ports × 8-wide × 2 flop), SIMD128 19.2 GFLOP/s (4-wide mul+add, no FMA).
Corpus: `harness/corpus/corpus.json` (same 18 cases as `wasm:gate` / `wasm:bench` batched). Single-thread.
Timers: `--features profile` only. Default `wasm:build` artifact is bit-identical to 49a6e8d2.

## Native (AVX2, `--features profile`)

| case | n | total ms | matmul % | attn % | qkv ms | ffn* ms | qk ms | softmax ms | V-mix ms | RoPE ms | LN ms | tok+emb+pool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short-hello-document | 7 | 245.39 | 99.6% | 0.1% | 8.48 | 214.79 | 0.16 | 0.10 | 0.09 | 0.099 | 0.200 | 0.022 |
| short-hello-query | 7 | 244.34 | 99.6% | 0.1% | 8.30 | 213.97 | 0.16 | 0.11 | 0.08 | 0.077 | 0.197 | 0.020 |
| short-hello-none | 3 | 105.69 | 99.5% | 0.1% | 3.96 | 92.22 | 0.04 | 0.03 | 0.03 | 0.046 | 0.097 | 0.009 |
| unicode-cjk-emoji | 19 | 657.12 | 99.4% | 0.3% | 18.98 | 577.48 | 1.00 | 0.27 | 0.45 | 0.301 | 0.512 | 0.046 |
| unicode-nfc | 9 | 314.97 | 99.6% | 0.2% | 10.43 | 276.13 | 0.25 | 0.10 | 0.13 | 0.112 | 0.256 | 0.030 |
| unicode-nfd | 9 | 314.05 | 99.6% | 0.1% | 10.31 | 275.37 | 0.25 | 0.10 | 0.12 | 0.098 | 0.248 | 0.027 |
| whitespace-padded | 9 | 313.75 | 99.6% | 0.2% | 10.23 | 275.17 | 0.25 | 0.10 | 0.12 | 0.097 | 0.249 | 0.024 |
| whitespace-only | 6 | 209.95 | 99.6% | 0.1% | 7.30 | 183.74 | 0.12 | 0.08 | 0.06 | 0.069 | 0.172 | 0.020 |
| long-repeated | 502 | 18346.29 | 94.0% | 5.6% | 488.70 | 15255.72 | 672.52 | 93.80 | 267.29 | 11.408 | 18.317 | 0.804 |
| empty-document | 6 | 210.30 | 99.6% | 0.1% | 7.62 | 183.80 | 0.12 | 0.08 | 0.06 | 0.067 | 0.174 | 0.019 |
| empty-none | 2 | 71.86 | 99.5% | 0.1% | 3.26 | 62.21 | 0.02 | 0.02 | 0.02 | 0.030 | 0.077 | 0.006 |
| newlines-tabs | 14 | 484.72 | 99.5% | 0.2% | 14.43 | 426.05 | 0.57 | 0.21 | 0.25 | 0.145 | 0.372 | 0.029 |
| punctuation-only | 9 | 314.31 | 99.6% | 0.1% | 10.43 | 275.53 | 0.25 | 0.10 | 0.12 | 0.107 | 0.252 | 0.024 |
| prefix-already-present | 13 | 453.30 | 99.5% | 0.2% | 14.34 | 397.69 | 0.50 | 0.18 | 0.24 | 0.217 | 0.383 | 0.037 |
| numeric-ids | 58 | 2034.79 | 99.0% | 0.7% | 58.40 | 1781.09 | 9.10 | 1.49 | 3.69 | 0.788 | 1.655 | 0.092 |
| flair-memory-preference | 28 | 979.92 | 99.3% | 0.4% | 29.06 | 859.19 | 2.23 | 0.49 | 0.87 | 0.466 | 0.815 | 0.069 |
| flair-memory-decision | 49 | 1705.76 | 99.1% | 0.6% | 49.35 | 1494.48 | 6.54 | 1.07 | 2.65 | 0.701 | 1.322 | 0.087 |
| flair-query-recall | 17 | 592.75 | 99.5% | 0.2% | 18.27 | 520.38 | 0.82 | 0.22 | 0.37 | 0.216 | 0.474 | 0.042 |

`ffn*` = gate + up + down + swiglu.

## WASM (SIMD128, `--features profile`, separate artifact)

| case | n | total ms | matmul % | attn % | qkv ms | ffn* ms | qk ms | softmax ms | V-mix ms | RoPE ms | LN ms | tok+emb+pool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short-hello-document | 7 | 190.96 | 99.1% | 0.3% | 38.30 | 139.77 | 0.22 | 0.16 | 0.13 | 0.188 | 0.346 | 0.030 |
| short-hello-query | 7 | 189.70 | 99.1% | 0.3% | 38.16 | 138.83 | 0.21 | 0.18 | 0.12 | 0.137 | 0.349 | 0.027 |
| short-hello-none | 3 | 81.23 | 99.1% | 0.2% | 16.40 | 59.34 | 0.05 | 0.06 | 0.04 | 0.063 | 0.135 | 0.010 |
| unicode-cjk-emoji | 19 | 524.74 | 98.9% | 0.4% | 103.80 | 385.47 | 1.30 | 0.38 | 0.61 | 0.355 | 0.793 | 0.068 |
| unicode-nfc | 9 | 249.16 | 99.1% | 0.3% | 50.46 | 182.25 | 0.33 | 0.13 | 0.18 | 0.169 | 0.388 | 0.037 |
| unicode-nfd | 9 | 244.10 | 99.1% | 0.3% | 49.15 | 178.50 | 0.35 | 0.16 | 0.21 | 0.176 | 0.383 | 0.035 |
| whitespace-padded | 9 | 244.26 | 99.1% | 0.3% | 49.15 | 178.71 | 0.33 | 0.13 | 0.19 | 0.171 | 0.382 | 0.038 |
| whitespace-only | 6 | 162.11 | 99.2% | 0.2% | 32.53 | 118.77 | 0.16 | 0.09 | 0.10 | 0.112 | 0.259 | 0.022 |
| long-repeated | 502 | 14954.38 | 90.3% | 9.1% | 2736.72 | 9961.93 | 880.91 | 98.15 | 378.94 | 15.585 | 24.937 | 0.584 |
| empty-document | 6 | 163.45 | 99.2% | 0.2% | 32.86 | 119.78 | 0.16 | 0.09 | 0.10 | 0.115 | 0.264 | 0.021 |
| empty-none | 2 | 55.21 | 99.1% | 0.1% | 11.00 | 40.44 | 0.03 | 0.03 | 0.02 | 0.049 | 0.101 | 0.008 |
| newlines-tabs | 14 | 379.57 | 99.1% | 0.4% | 76.27 | 277.75 | 0.73 | 0.24 | 0.36 | 0.256 | 0.584 | 0.045 |
| punctuation-only | 9 | 244.98 | 99.1% | 0.3% | 49.51 | 179.24 | 0.33 | 0.13 | 0.19 | 0.168 | 0.387 | 0.031 |
| prefix-already-present | 13 | 354.47 | 99.1% | 0.3% | 71.30 | 259.50 | 0.64 | 0.20 | 0.32 | 0.244 | 0.545 | 0.040 |
| numeric-ids | 58 | 1579.81 | 98.2% | 1.2% | 314.86 | 1145.51 | 12.05 | 1.88 | 5.48 | 1.060 | 2.359 | 0.167 |
| flair-memory-preference | 28 | 767.21 | 98.7% | 0.6% | 154.08 | 558.29 | 2.83 | 0.55 | 1.39 | 0.600 | 1.224 | 0.094 |
| flair-memory-decision | 49 | 1334.23 | 98.3% | 1.1% | 266.82 | 968.35 | 8.59 | 1.38 | 4.22 | 1.048 | 2.096 | 0.118 |
| flair-query-recall | 17 | 462.34 | 98.9% | 0.5% | 92.47 | 338.37 | 1.05 | 0.58 | 0.54 | 0.313 | 0.703 | 0.054 |

## Per-stage detail: short-hello-document (n=7) and long-repeated (n=502)

### Native n=7 (245.39 ms)

| stage | ms | % |
| --- | ---: | ---: |
| tokenize | 0.013 | 0.005% |
| embedding_lookup | 0.007 | 0.003% |
| qkv | 8.480 | 3.46% |
| rope | 0.099 | 0.041% |
| attn_qk | 0.164 | 0.067% |
| attn_softmax | 0.095 | 0.039% |
| attn_vmix | 0.088 | 0.036% |
| out_proj | 21.023 | 8.57% |
| ffn_gate | 84.020 | 34.24% |
| ffn_up | 83.759 | 34.13% |
| ffn_swiglu | 0.249 | 0.10% |
| ffn_down | 47.011 | 19.16% |
| layernorm | 0.200 | 0.081% |
| pooling | 0.002 | 0.001% |

### Native n=502 (18346.29 ms)

| stage | ms | % |
| --- | ---: | ---: |
| tokenize | 0.111 | 0.001% |
| embedding_lookup | 0.631 | 0.003% |
| qkv | 488.70 | 2.66% |
| rope | 11.41 | 0.062% |
| attn_qk | 672.52 | 3.67% |
| attn_softmax | 93.80 | 0.51% |
| attn_vmix | 267.29 | 1.46% |
| out_proj | 1504.41 | 8.20% |
| ffn_gate | 5982.78 | 32.61% |
| ffn_up | 5988.76 | 32.64% |
| ffn_swiglu | 22.62 | 0.12% |
| ffn_down | 3284.19 | 17.90% |
| layernorm | 18.32 | 0.10% |
| pooling | 0.063 | 0.000% |

### WASM n=7 (190.96 ms)

| stage | ms | % |
| --- | ---: | ---: |
| tokenize | 0.014 | 0.007% |
| embedding_lookup | 0.011 | 0.006% |
| qkv | 38.304 | 20.06% |
| rope | 0.188 | 0.099% |
| attn_qk | 0.225 | 0.118% |
| attn_softmax | 0.156 | 0.082% |
| attn_vmix | 0.126 | 0.066% |
| out_proj | 11.086 | 5.81% |
| ffn_gate | 44.142 | 23.12% |
| ffn_up | 44.224 | 23.16% |
| ffn_swiglu | 0.463 | 0.24% |
| ffn_down | 51.402 | 26.92% |
| layernorm | 0.346 | 0.181% |
| pooling | 0.005 | 0.002% |

### WASM n=502 (14954.38 ms)

| stage | ms | % |
| --- | ---: | ---: |
| tokenize | 0.255 | 0.002% |
| embedding_lookup | 0.181 | 0.001% |
| qkv | 2736.72 | 18.30% |
| rope | 15.59 | 0.10% |
| attn_qk | 880.91 | 5.89% |
| attn_softmax | 98.15 | 0.66% |
| attn_vmix | 378.94 | 2.53% |
| out_proj | 805.16 | 5.38% |
| ffn_gate | 3137.76 | 20.98% |
| ffn_up | 3133.83 | 20.96% |
| ffn_swiglu | 34.13 | 0.23% |
| ffn_down | 3690.34 | 24.68% |
| layernorm | 24.94 | 0.17% |
| pooling | 0.147 | 0.001% |

## Roofline (achieved GFLOP/s vs single-thread peak)

FLOPs = `2 * n_tok * n_in * n_out` per matmul (f32-MAC equivalent). QKV is Q5_K (native AVX2 / WASM SIMD128). FFN + out-proj are Q4_K (native portable scalar GEMV / WASM SIMD128 `gemv_q4_k_8x8_q8_k`).

| stage | n=7 native GFLOP/s (% AVX2) | n=7 WASM GFLOP/s (% SIMD128) | n=502 native GFLOP/s (% AVX2) | n=502 WASM GFLOP/s (% SIMD128) |
| --- | ---: | ---: | ---: | ---: |
| qkv | 35.06 (45.6%) | 7.76 (40.4%) | 43.62 (56.8%) | 7.79 (40.6%) |
| out_proj | 4.71 (6.1%) | 8.94 (46.6%) | 4.72 (6.2%) | 8.83 (46.0%) |
| ffn_gate | 4.72 (6.1%) | 8.98 (46.8%) | 4.75 (6.2%) | 9.06 (47.2%) |
| ffn_up | 4.73 (6.2%) | 8.96 (46.7%) | 4.75 (6.2%) | 9.07 (47.2%) |
| ffn_down | 8.43 (11.0%) | 7.71 (40.2%) | 8.66 (11.3%) | 7.70 (40.1%) |
| attn_qk | 5.52 (7.2%) | 4.02 (20.9%) | 6.91 (9.0%) | 5.27 (27.5%) |
| attn_vmix | 10.28 (13.4%) | 7.17 (37.3%) | 17.38 (22.6%) | 12.26 (63.8%) |
| matmul_all | 6.49 | 8.38 | 6.59 | 8.42 |

## Native vs llama.cpp n_threads=1 (same host)

Default `milton-embed` (no timers), wasm:bench single set (first 8 corpus cases):

| | n cases | ms | emb/s |
| --- | ---: | ---: | ---: |
| Milton native | 8 | 2422.98 | 3.30 |
| llama.cpp `-t 1` (sum of prompt-eval) | 8 | 262.68 | 30.46 |
| **llama / Milton-native** | | | **9.22×** |

Per-case prompt-eval (llama.cpp `-t 1`) vs Milton:

| case | n | llama prompt-eval ms | Milton native ms | Milton WASM ms | llama ms/tok | Milton native ms/tok | Milton WASM ms/tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short-hello-document | 7 | 30.20 | 245.39 | 190.96 | 4.31 | 35.06 | 27.28 |
| long-repeated | 502 | 1710.92 | 18346.29 | 14954.38 | 3.41 | 36.55 | 29.79 |

llama.cpp improves ms/token with n (4.31 → 3.41). Milton stays flat (~35 native / ~27 WASM). That is the GEMV-vs-GEMM tell.

WASM is *faster* than Milton native on this host because native Q4_K GEMV is the portable scalar kernel; WASM Q4_K GEMV is SIMD128. The 15× WASM-vs-llama gap is **not** "WASM is slow" — Milton's structure is 9.2× behind llama.cpp even in native AVX2.

Do not mix this host's llama 30.46 emb/s with the stored CI/other-host baseline 65.86 emb/s.

## Hypotheses

- **H1 CONFIRMED (dominant).** `matmul_ggml` loops `for t in 0..n_tokens` and re-reads weights each token. 60 GEMV calls/token (5 mats × 12 layers), 70,004,736 weight bytes re-read per token. Native matmul ms/token 34.90 (n=7) vs 34.36 (n=502). Time ratio long/short = 70.61 vs token ratio 71.71 (`linear_in_n: true`).
- **H2 ruled out as dominant.** Attention grows ~n² (ms ratio 2978 vs n² 5143) but is 0.14% short / 5.6% long native (0.27% / 9.1% WASM). One `Vec<n_tokens>` scores alloc, not per token.
- **H3 ruled out.** JS−Rust = 0.32 ms short / 0.23 ms long. Rust is 99.998% of JS wall on long. One call, 3072-byte embedding.
- **H4 ruled out for WASM.** 299 fns, 49 with SIMD, 1796 SIMD insns. Hot: `vec_dot_q5_k_q8_k`, `vec_dot_q6_k_q8_k`, `gemv_q4_k_8x8_q8_k`, `swiglu`, `sincosf_inplace`, `attention_named`. Large no-SIMD fns are fmt/alloc/vocab. Native Q4_K being scalar is a native-only fact, not a missing WASM SIMD128.

## Named cause

Per-token quantized GEMV (`matmul_ggml` `for t in 0..n_tokens`). 60 GEMVs/token, ~66.8 MiB weights re-read every token. Matmul is 90–99% of wall. Milton native is 9.2× behind llama.cpp n_threads=1 — Milton's structure is slow; WASM is not the gap.

## Predicted next slice (do not implement here)

Sequence-tiled GEMM for all five `matmul_ggml` sites (Q4_K FFN up/gate/down, Q4_K out-proj, Q5_K QKV), same numeric path as the live GEMV (row Q8_K + mul+add). Do **not** land the 4x8 FMA GEMM that avalanches FINAL.

**Predicted gain:** 3.0× ± 0.5× WASM single-thread emb/s on the wasm:bench 8-case set (this host); 5.0× ± 1× on long-repeated. Implementing PR must hit 2.5–3.5× single / 4–6× long or explain the miss.

Rationale: llama is 6.3× faster than WASM on n=7 and 8.7× on n=502; leftover after GEMM is roughly SIMD128 vs AVX2 (~2×).
