# Prebuilt WASM-SIMD

This directory is **the shipped embedder**. `npm i` uses these files as-is.
Consumers do not need Rust, `wasm-pack`, or a compile step.

| file | role |
|---|---|
| `milton_bg.wasm` | same `crate/` lib compiled `wasm32-unknown-unknown` + `simd128` |
| `milton.js` | wasm-bindgen `--target web` glue (Node loads it via `src/index.js`) |

## How the wasm is produced (builder-side only)

```sh
# rustc 1.83+, wasm-bindgen-cli 0.2.100 (pinned to crate/Cargo.toml)
cargo install wasm-bindgen-cli --version 0.2.100 --locked
npm run wasm:build
```

`scripts/build-wasm.sh` runs:

```
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --manifest-path crate/Cargo.toml \
    --target wasm32-unknown-unknown --release --lib
wasm-bindgen crate/target/wasm32-unknown-unknown/release/milton.wasm \
  --out-dir wasm --target web --out-name milton --omit-default-module-path
```

That is **the same forward pass** as `milton-embed` / `embed-gate`, not a
re-implementation. AVX2 kernels (`q4k_avx2`, `#[cfg(target_arch = "x86_64")]`)
are not in this binary — WASM-SIMD is CPU-portable. An AVX2-vs-WASM-SIMD
numerical close is a later chip.

The GGUF is **not** inside the `.wasm`. JS reads the file and passes bytes
into `new Milton(ggufBytes)`.
