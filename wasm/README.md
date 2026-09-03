# Prebuilt WASM-SIMD

This directory is **the shipped embedder**. `npm i` uses these files as-is.
Consumers do not need Rust, `wasm-pack`, or a compile step.

| file | role |
|---|---|
| `milton_bg.wasm` | same `crate/` lib compiled `wasm32-unknown-unknown` + `simd128` (ordinary path when SAB is absent) |
| `milton.js` | wasm-bindgen `--target web` glue (Node loads it via `src/index.js`) |
| `milton_threads_bg.wasm` | same crate + `wasm-threads`, `+atomics,+bulk-memory`, **shared** memory import |
| `milton_threads.js` | wasm-bindgen glue for the threaded artifact |

The JS loader (`src/index.js`) picks the threaded module only when
`SharedArrayBuffer` + `Atomics` exist, `WebAssembly.validate` accepts a
shared-memory probe, **and** the pool would be larger than 1.
`MILTON_THREADS=1` forces `milton_bg.wasm` (not threads-with-W=1).
`MILTON_THREADS=<n>` sizes the pool when the threads artifact is selected.
`lastThreadReport` records `{artifact, workers, availableParallelism, sabAvailable}`
after load (`sabAvailable` is the capability probe, not the pick).
Absence of SAB is the ordinary path, not an error. A shared-memory module
cannot instantiate where SAB is absent — that is why there are two artifacts.

`milton_threads_bg.wasm` is **reproducible only from CI**. The threads
build uses `-Z build-std`, so crate-hash suffixes (`::hXXXX` in the name
section) follow the rust-src host path even after `--remap-path-prefix`.
A local `npm run wasm:build` is expected to differ in those hashes.
CI is the build of record: do not replace a CI-matching blob with a
local rebuild unless the CI byte-compare is red and you are committing
CI's own output.

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
