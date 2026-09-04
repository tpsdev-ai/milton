# Prebuilt WASM-SIMD

This directory is **the shipped embedder**. `npm i` uses these files as-is.
Consumers do not need Rust, `wasm-pack`, or a compile step.

| file | role |
|---|---|
| `milton_bg.wasm` | same `crate/` lib compiled `wasm32-unknown-unknown` + `simd128` (bun / probe-fail / `MILTON_RELAXED_SIMD=0`) |
| `milton.js` | wasm-bindgen `--target web` glue for the simd128 artifact |
| `milton_relaxed_bg.wasm` | same crate + feature `relaxed-simd` (Q4_K / Q5_K i8×i7 dot). Loaded when the relaxed-dot probe passes and threads are off (`MILTON_THREADS=1` or pool ≤1). |
| `milton_relaxed.js` | wasm-bindgen glue for the relaxed single-thread artifact |
| `milton_threads_bg.wasm` | same crate + `wasm-threads`, `+atomics,+bulk-memory`, **shared** memory import (simd128 fallback when probe fails or `MILTON_RELAXED_SIMD=0`) |
| `milton_threads.js` | wasm-bindgen glue for the threaded simd128 artifact |
| `milton_threads_relaxed_bg.wasm` | threads + `relaxed-simd` — loaded when **both** SAB/Atomics and the relaxed-dot probe pass |
| `milton_threads_relaxed.js` | wasm-bindgen glue for the threaded relaxed artifact |

The JS loader (`src/index.js`) picks the threaded module only when
`SharedArrayBuffer` + `Atomics` exist, `WebAssembly.validate` accepts a
shared-memory probe, **and** the pool would be larger than 1.
`MILTON_THREADS=1` forces the single-thread module (not threads-with-W=1).
`MILTON_THREADS=<n>` sizes the pool when the threads artifact is selected.
`lastThreadReport` records `{artifact, workers, availableParallelism, sabAvailable, wasm, attnMinTokens}`
after load (`sabAvailable` is the capability probe, not the pick; `wasm` is
the basename actually instantiated; `attnMinTokens` is the effective A2
serial→parallel gate after `MILTON_ATTN_MIN_TOKENS` / default 32). A failed load publishes the same grain
with an `error` string and `wasm` set to the artifact that was attempted
(never a prior success).
`MILTON_ATTN_MIN_TOKENS` overrides that gate (non-numeric / out-of-range → 32, warn once).
This is an env override only — not a measured load-time crossover.
Absence of SAB is the ordinary path, not an error. A shared-memory module
cannot instantiate where SAB is absent — that is why there are separate
single-thread and threaded artifacts, each with a simd128 and relaxed variant.

## Relaxed SIMD (issue #43)

Fast-path floor is **Node ≥ 22** (V8 12.4). Node 22 / 24 / 26 validate
`i16x8.relaxed_dot_i8x16_i7x16_s` with no flags. Node 20 predates
default-on relaxed SIMD.

Detection is `WebAssembly.validate` of a one-function probe containing
that instruction (`0xfd 0x112`) — **never a runtime or version sniff**.
Bun (1.3.10, JSC) validates SIMD128 and rejects the relaxed probe, so
every bun consumer — including flair's own test suites — takes
`milton_bg.wasm`. The simd128 kernel is a first-class path, not a corner
case.

`lastQmatmulKernel` records `{kernel: 'relaxed' | 'simd128', probe, forced}`
after load (`probe` is the capability, not the pick). On a failed load it
also carries `error` and `wasm` (the attempted artifact). On the threaded
path, `lastThreadReport.artifact` is `'threads'` at the same time.
`MILTON_RELAXED_SIMD=0` forces simd128 even when the probe passes.
`MILTON_RELAXED_SIMD=1` fail-closes if the probe rejects.

This sandbox's Node 22.14 has no `--no-wasm-relaxed-simd` (the V8 flag
is gone once relaxed SIMD is default-on). The named fallback compare
lane is `npm run wasm:compare:simd128` (`MILTON_RELAXED_SIMD=0`).

Tight bit-exact verdict (`wasm:compare-verdict`) runs on **both**
artifacts in `wasm-compare` (Node 22) and `wasm-compare-node26`:

| npm script | file | artifact | W | kernel |
|---|---|---|---:|---|
| `wasm:compare-verdict` | `milton_relaxed_bg.wasm` | `single` | 1 | `relaxed` |
| `wasm:compare-verdict:simd128` | `milton_bg.wasm` | `single` | 1 | `simd128` |
| `wasm:compare-verdict:threads` | `milton_threads_relaxed_bg.wasm` | `threads` | 4 | `relaxed` |
| `wasm:compare-verdict:threads:simd128` | `milton_threads_bg.wasm` | `threads` | 4 | `simd128` |

Threaded scripts use `env MILTON_WASM_THREADS=1 MILTON_THREADS=4` so a
workflow-level `MILTON_WASM_THREADS=0` cannot neuter them. The verdict
is RED unless the receipt shows that file, `thread_report.workers > 1`,
and `max_abs=0`. A labeled `artifact=threads` with W=1 fails the lane.

Q6_K stays on the base SIMD128 kernel because of its −32 reconstruction
offset, not because 0..63 is out of i7 range (`63 = 0x3F`).

`milton_threads_bg.wasm` is **reproducible only from CI**. The threads
build uses `-Z build-std`, so crate-hash suffixes (`::hXXXX` in the name
section) follow the rust-src host path even after `--remap-path-prefix`.
A local `npm run wasm:build` is expected to differ in those hashes.
CI is the build of record: do not replace a CI-matching blob with a
local rebuild unless the CI byte-compare is red and you are committing
CI's own output.

`milton_bg.wasm` and `milton_relaxed_bg.wasm` do not use `build-std`.
If a local rebuild of either ever diverges from CI (the #50 lesson),
CI is still the build of record — document the delta here rather than
fighting the hash. Do not weaken the byte-compare.

#43 run `33774688123` (HEAD `e2af574`) was that case. Committed here
are CI's rebuilt blobs from that run's `milton_bg.wasm` artifact:

| artifact | sha256 | bytes |
|---|---|---:|
| `milton_bg.wasm` | `836fab097009f110e2bd53928a1aac74d275b16350c7cc021456422241a08a1a` | 636581 |
| `milton_relaxed_bg.wasm` | `f7750b2635bd8bc9e86226828bb0658862ad11274e38b5850f4b23587564caa9` | 638171 |
| `milton_threads_bg.wasm` | `ccad4d27f7e7def502d74a4ead55ad85c86ccb14e556e9d9f948be85279cc884` | 619768 |
| `milton_threads_relaxed_bg.wasm` | `d31d3b5c8708c67393205bb13add5fac3f1c3569b2d4e9a550188cfd7da13d36` | 620820 |

`milton_threads_relaxed_bg.wasm` is new in the threads-relaxed fix (`bc35f6b`).
CI run `33777431688` is the build of record for both threaded blobs.
Glue (`milton_threads.js` / `milton_threads_relaxed.js`) is byte-identical.

#59 adds `attnSetMinTokens` / `attnMinTokens` (env override of the A2 gate).
That forces a remapped rebuild. CI run `33832387376` is the build of record
for the four blobs below (local remap differed in bytes with no section-size
delta — same #50 lesson). Glue is unchanged from the local wasm-bindgen emit.

| artifact | sha256 | bytes |
|---|---|---:|
| `milton_bg.wasm` | `166e01edaa15d17da40c48c34be8f5b00b010e99cc249f93474e2a9628ad1935` | 636781 |
| `milton_relaxed_bg.wasm` | `48e4e2c401c509f8b7230de91fd9a163953f032b69f49a1015495246dc9c7d4f` | 638371 |
| `milton_threads_bg.wasm` | `602b8e600f18df73abe9c83ab8b68e7fa845a23f5d7a339aeea948479a9da579` | 621158 |
| `milton_threads_relaxed_bg.wasm` | `22808601605268961688a33d989cf9937328a7178290bb6963cbbb9090d36e64` | 622210 |

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
