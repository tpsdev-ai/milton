# #44 plan — column split, per-layer join, barrier accounting

Plan-only. No thread / kernel / loader code in this commit.
Derived from `main` `741e88ba` (`crate/src/model.rs` `forward_hidden`,
`crate/src/qmatmul.rs` `matmul_ggml`, `crate/src/qmatmul_simd128.rs`,
`crate/src/wasm.rs`). Not guessed.

Held numbers in the issue body (≤9.0 ms/token after #42 / ≤6.5 after #43)
are stale: #43 closed without landing; #45 / #47 / #49 landed after that
writeup. Cos re-derived vs this SHA: 4-worker `wasm:bench` ≥2.5× the
single-thread baseline on the same host; single-thread artifact must not
regress within 2% vs `741e88ba`.

## The five `matmul_ggml` sites (one BERT layer)

From `Model::forward_hidden` (`crate/src/model.rs`). nomic-bert dims from
the GGUF (also recorded on `QuantMat` and the #39 tile comment):
`n_embd=768`, `n_head=12`, `n_ff=3072`, `n_layer=12`. Q4_K_M mix:
QKV is Q5_K ×12; out-proj + FFN up/gate are Q4_K ×12; FFN-down is Q6_K
on blk.0/4/7/8/9/11 and Q4_K on the other six.

| # | site | call | n_in | n_out | ty | input |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1 | `attn_qkv` | `matmul_ggml(&x, &layer.attn_qkv, …)` | 768 | 2304 | Q5_K | residual `x` after LN |
| 2 | `attn_output` | `matmul_ggml(&attn_out, &layer.attn_output, …)` | 768 | 768 | Q4_K | attention output |
| 3 | `ffn_up` | `matmul_ggml(&x, &layer.ffn_up, …)` | 768 | 3072 | Q4_K | post-attn residual+LN |
| 4 | `ffn_gate` | `matmul_ggml(&x, gate_w, …)` | 768 | 3072 | Q4_K | same `x` as up |
| 5 | `ffn_down` | `matmul_ggml(&ffn_hid, &layer.ffn_down, …)` | 3072 | 768 | Q6_K or Q4_K | SwiGLU hidden |

No other `matmul_ggml` in the layer loop. Embedding lookup is gather, not
a site.

## Happens-before (why 5 sites are not 5 independent jobs)

```
x ──► [1 QKV] ──► bias, split_qkv, RoPE, attention ──► attn_out
attn_out ──► [2 out-proj] ──► residual + LN ──► x'
x' ──► [3 up] ─┐
x' ──► [4 gate] ┴──► SwiGLU ──► [5 down] ──► residual + LN ──► next layer
```

Sites 3 and 4 share the same immutable `x'` and write disjoint buffers
(`ffn_up`, `ffn_gate`). They are the only pair that may share one
dispatch. Site 2 cannot start until attention has written `attn_out`.
Site 5 cannot start until SwiGLU has written `ffn_hid`. Attention / RoPE
/ LN / SwiGLU are not column-split in this chip.

A single barrier after all five sites would let workers compute site 2
from uninitialized `attn_out`. That is a data race and a bit-exact miss.
The issue's "one job list, join once" is the JS grain, not a claim that
the five sites commute.

## Column split (W workers)

Split each site on **output columns**. No cross-column dependency:
`dst[t, o] = vec_dot(W_col[o], Q8_K(x[t]))`. Worker `i` owns
`[col_start(i), col_end(i))` and writes only that range of `y`.

Range (contiguous, leftover on the last worker):

```
start(i) = i * n_out / W
end(i)   = (i + 1) * n_out / W
```

Q4_K 8×8 (`n_out % 8 == 0`) splits on **8-col groups** so a worker never
owns a partial `block_q4_Kx8`:

```
g = n_out / 8
start(i) = (i * g / W) * 8
end(i)   = ((i + 1) * g / W) * 8
```

For default `W = min(4, os.availableParallelism())` every site divides
evenly and every Q4_K range is 8-aligned:

| site | n_out | cols / worker (W=4) | Q4_K groups / worker |
| --- | ---: | ---: | ---: |
| QKV | 2304 | 576 | n/a (Q5_K per-col) |
| out-proj | 768 | 192 | 24 |
| FFN up | 3072 | 768 | 96 |
| FFN gate | 3072 | 768 | 96 |
| FFN down | 768 | 192 | 24 (Q4_K layers) |

`W=1` is the full range (same as today's single-thread loop). `W=2,3`
also land on 8-col boundaries for 768/2304/3072.

Per-column arithmetic is unchanged: same Q8_K row, same SIMD128 tree,
same mul+add. The only difference is which `o` a worker iterates.
That is the bit-exact argument.

## Shared-memory races (argued from the split)

Reads (immutable for the duration of a dispatch):

- Weight bytes / Q4_K 8×8 repack (`QuantMat`) — load-time, never written
  during embed.
- Q8_K tile — quantized once on the coordinator before the dispatch;
  workers do not write it. No per-call copies of rows or weights.

Writes:

- Each worker writes `y[t * n_out + o]` only for `o` in its half-open
  range. Ranges are disjoint → no write-write race.
- Coordinator writes Q8_K and the job list, then `Release` on the epoch
  before `notify`. Workers `Acquire` the epoch before reading. After
  `workers_done == W`, the coordinator `Acquire`s before reading `y`.

No new unbounded `unsafe`. Existing SIMD128 loads stay bounded. Atomics
are `core::sync::atomic` + `memory_atomic_wait32` / `notify` on a
fixed-size control block.

## Per-layer join (one JS join; four atomic phases)

Naive JS `postMessage` + `Promise` per site: **5 × 12 = 60** joins/token.
At the issue's ~20 µs that is **1.2 ms/token** — a third of the old
target, and the reason Cos forbade per-site joins.

Design:

1. One `worker_threads` pool per module instance. Size
   `min(MILTON_THREADS || 4, os.availableParallelism())`. Workers
   instantiate the **shared-memory** module against the same
   `WebAssembly.Memory` and park on `Atomics.wait`.
2. The coordinator (main WASM thread inside `embed`) writes a **layer
   job list** of the five sites (ptrs, dims, type, per-worker column
   ranges) once at the start of the layer.
3. Four **atomic** phases walk that list. One join per phase, not per
   site and not per column:

   | phase | jobs in the list | then (coordinator, single-thread) |
   | --- | --- | --- |
   | A | site 1 QKV | bias, split, RoPE, attention |
   | B | site 2 out-proj | residual + LN |
   | C | sites 3+4 up and gate | SwiGLU |
   | D | site 5 down | residual + LN |

4. JS does not join per phase. `embed()` is one wasm-bindgen call.
   Workers are already running. The only JS-visible join is the return
   of that call (and pool teardown if the instance is dropped).

`W=1` (or the single-thread artifact) never enters the pool: the
existing `matmul_ggml` loop runs on the coordinator.

## Barrier / dispatch accounting

| grain | joins / token | cost model | µs / token |
| --- | ---: | --- | ---: |
| JS postMessage per site (forbidden) | 60 | ~20 µs | **1200** |
| JS postMessage per layer | 12 | ~20 µs | 240 |
| Atomic phase join (this design) | **48** (4×12) | ~1 µs `wait`/`notify` | **~48** |
| JS join per `embed()` | 1 | already in the call | 0 extra |

Committed `#28` WASM profile (`harness/profile/stage-profile.tables.md`,
n=7 `short-hello-document`): matmul is **99.1%** of wall (qkv 20.1% +
out 5.8% + gate 23.1% + up 23.2% + down 26.9%). Sequential leftover
(attn + RoPE + LN + SwiGLU + tok/emb/pool) is **<1%**. Those tables
predate #45/#47/#49; the **share** is what the roofline uses. Absolute
ms/token is re-measured on this PR against `741e88ba`.

Roofline on the matmul share, W=4, no cross-column dep, ~3.3× on that
90%+ (issue text). Amdahl:

```
speedup ≈ 1 / ( (1 − m) + m/3.3 )
```

- If `m = 0.90`: **2.70×**
- If `m = 0.99` (n=7 profile): **3.20×**
- Held: **≥ 2.5×** `wasm:bench` 8-case single vs the same-host
  single-thread baseline.

Atomic overhead vs matmul: 48 µs on a token that is several
milliseconds of matmul is **< 2%** of wall. The 1.2 ms JS-per-site
figure would have been material; that is why the join is inside WASM.

A miss versus 2.5× is explained here (phase count, leftover sequential,
dispatch) — not waved, not "widen the held".

## Two artifacts, one loader (not this commit)

- `wasm/milton_bg.wasm` — today's `+simd128` single-thread module.
  Ordinary path when SAB is absent. Must not regress within 2% vs
  `741e88ba` on `wasm:bench`. `wasm:compare` tight `max_abs=0`.
- `wasm/milton_threads_bg.wasm` — `+simd128,+atomics,+bulk-memory`
  shared-memory import. Instantiation **requires** SAB; that is why
  there are two blobs, not a flag on one.
- Loader picks the threaded module only when `SharedArrayBuffer` and
  `Atomics` exist **and** `WebAssembly.validate` accepts a shared-memory
  probe. Absence of SAB is the ordinary path, not an error.
- Both rebuilt by the remapped `scripts/build-wasm.sh` and
  byte-compared in CI. Second compare lane `wasm:compare:threads`
  (tight `max_abs=0`). `MILTON_THREADS=1` forces the single-thread
  artifact (`milton_bg.wasm`), not threads-with-W=1. `MILTON_THREADS=<n>`
  sizes the pool when the threads artifact is selected. SAB-absent and
  `MILTON_WASM_THREADS=0` also pick single so the existing compare lane
  stays honest on Node (where SAB exists). `lastThreadReport` records
  `{artifact, workers, availableParallelism, sabAvailable}`.

Native AVX2 stays where #40/#49 left it. No relaxed-simd (#43). No
further kernel letters. Product is WASM. v1 CPU-portable.

## Host for the held table

Cursor cloud VM: Linux x86_64, 4× Intel Xeon (KVM), Node v22.14.0,
`SharedArrayBuffer` present, `os.availableParallelism() = 4`. Flint
pairs M4. This PR reports the x86 VM.
