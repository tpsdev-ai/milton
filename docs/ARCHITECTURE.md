# Milton architecture — proposed (for Nathan's nod)

## Decision: Rust → WASM (SIMD), shipped as a prebuilt `.wasm` + thin TS glue

The npm package is JS/TS glue plus one small prebuilt `.wasm`. **`npm i` downloads the
wasm — there is no native compile, no node-gyp, no per-platform build step at install.**
That zero-native-install property is the entire reason Milton exists; it is not
negotiable for v1.

### Options considered

| approach | native install? | speed | verdict |
|---|---|---|---|
| Pure JS/TS | none | too slow (no SIMD matmul) | out |
| Native N-API (napi-rs/C++) | yes (prebuilds or node-gyp) | fastest | reintroduces the footprint we're escaping — out for v1 |
| **Rust → WASM-SIMD** | **none** | fast enough for a small embed model | **chosen** |
| WebGPU | heavy/immature in Node | fast | bounded follow-on, not v1 |

### Why Rust as the WASM source

The real work is GGUF: loading the file and **dequantizing quant tensor types**
(Q4_K_M block structure, Q8_0, F16…) then matmul. [camelid](https://github.com/timtoole02/Camelid)
is Rust and already does exactly this (GGUF tensor types → compute, evidence-gated). It's
MIT and a known quantity internally — Rust→WASM lets Milton mirror those dequant/kernel
patterns rather than reinvent them. The correctness oracle stays llama.cpp regardless:
Milton's WASM output must match reference vectors within epsilon (see HARNESS-SPEC.md).

### Shape

```
src/            TS glue: public API (embed(text, {prefix}) -> Float32Array), wasm loader
crate/          Rust: GGUF load + dequant + forward (arch from the file)  → compiled to wasm/
wasm/           the prebuilt .wasm (SIMD), committed or built in CI, NOT compiled at user install
harness/        the reference oracle (llama.cpp) + gate — NOT shipped in the package
```

- **Architecture from the file:** dequant + forward read layer count, dims, pooling,
  and normalization from the GGUF. nomic-embed-text-v1.5 is the v1 verified path
  (Flair's model today). BERT-family files (bge / gte / e5 / nomic-v2) are the same
  shape parameterized by that metadata. Model #2 is pin goldens + prefix config +
  any GGUF-flagged variant (CLS vs mean) — not a rewrite. No model registry.
- **Tokenizer:** BERT-style WordPiece, from the GGUF's tokenizer data. A small
  pure-Rust (→wasm) or tiny TS implementation — cheap relative to the matmuls.
- **Prefix:** config, not code. v1's config is Flair's `search_query:` /
  `search_document:` convention so the golden-vector gate stays honest.
- **Pooling:** whatever the GGUF flags (mean for nomic-v1.5; CLS when the file
  says CLS). Do not assume.

### Honest scope

WASM-SIMD is **CPU-only** — no Metal/CUDA. For a ~137M embed-only model this is very
likely plenty. If throughput ever needs to rival llama.cpp-on-Metal, an optional
native/WebGPU accel path is a *bounded follow-on*, gated the same way (must match the
reference within epsilon). v1 is Rust→WASM-SIMD, CPU, correct-and-light.

### The invariant that governs everything

Nothing ships that hasn't cleared the golden-vector gate (HARNESS-SPEC.md). The
implementation tech is a means; the reference match is the definition of done.
