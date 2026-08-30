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
crate/          Rust: GGUF load + dequant + forward pass + mean pool  → compiled to wasm/
wasm/           the prebuilt .wasm (SIMD), committed or built in CI, NOT compiled at user install
harness/        the reference oracle (llama.cpp) + gate — NOT shipped in the package
```

- **Tokenizer:** nomic uses a BERT-style WordPiece tokenizer. A small pure-Rust (→wasm)
  or tiny TS implementation — cheap relative to the matmuls. Must reproduce Flair's
  `search_query:` / `search_document:` prefix handling exactly.
- **Pooling:** mean (nomic-v1.5's actual pooling — confirm against the GGUF metadata, do
  not assume).

### Honest scope

WASM-SIMD is **CPU-only** — no Metal/CUDA. For a ~137M embed-only model this is very
likely plenty. If throughput ever needs to rival llama.cpp-on-Metal, an optional
native/WebGPU accel path is a *bounded follow-on*, gated the same way (must match the
reference within epsilon). v1 is Rust→WASM-SIMD, CPU, correct-and-light.

### The invariant that governs everything

Nothing ships that hasn't cleared the golden-vector gate (HARNESS-SPEC.md). The
implementation tech is a means; the reference match is the definition of done.
