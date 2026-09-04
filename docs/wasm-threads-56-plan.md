# #56 plan — parallelize coordinator attention (head-split), leave RoPE/LN/SwiGLU

Plan-only. No thread / kernel / coordinator code in this commit.
Derived from `main` `b89d446` (`crate/src/ops.rs` `attention_named`,
`crate/src/model.rs` `forward_hidden`, `crate/src/wasm_pool.rs`,
`docs/wasm-threads-44-plan.md`, `harness/profile/stage-profile.tables.md`).
Not guessed. Not a join-micro-opt. Not a retune of #50/#53 held numbers.

Held (Cos-proposed vs this SHA's product path — auto threads, relaxed
when the probe passes). Flint may re-derive; Flint pairs M4. This PR
reports the x86 VM.

1. Long n≈502 `wasm:bench` / `profile:stages` ≥ **1.35×** same-host main long-n.
2. Short / 8-case must not regress more than **2%** vs main on the same host.
3. Single-thread artifact within **2%** of main.
4. Fallback lanes (`MILTON_THREADS=1`, `MILTON_RELAXED_SIMD=0`, SAB-absent)
   still green and within 2% where applicable.

A miss versus 1.35× is explained by the stage table — not waved, not
"widen the held".

## Lever choice (first that is bit-exact and measurable)

**Lever 1: parallelize attention / scores across workers with the same
column-split discipline (head-aligned).** That is the chip.

| lever | stage (WASM #28 n=502) | share of single-thread wall | on coordinator today? | this PR |
| --- | --- | ---: | --- | --- |
| 1 | attn_qk + softmax + V-mix | **9.1%** (880.91 + 98.15 + 378.94 ms) | yes | **move to workers** |
| 2 | RoPE | 0.10% (15.59 ms) | yes | **leave** — not material |
| 3 | LN | 0.17% (24.94 ms) | yes | **leave** — not material |
| 3 | SwiGLU | 0.23% (34.13 ms) | yes | **leave** — not material |
| 4 | 48 phase joins | amortized at long-n (#50) | n/a | **do not chase** |

Why not 2/3/4 as the headline:

- RoPE / LN / SwiGLU together are **0.50%** of single-thread long-n and
  shrink as a share of wall only if we ignore attention. After #50/#53
  they are still tens of milliseconds against **~1.4 s** of serial
  attention. `profile:stages` will reprint them; they stay on the
  coordinator.
- Joins already amortize at long-n (issue text / Flint #50). Adding one
  attention phase is +12 waits/token (~12 µs). That is not the limiter
  and not the claim.
- Retuning #50/#53 kernel letters is out of scope.

Short-n (WASM #28 n=7): attention is **0.27%** of wall (qk 0.118% +
softmax 0.082% + V-mix 0.066%). Parallelizing it cannot regress the
8-case set by more than timer noise — the work is not there.

## Stage-time accounting, short vs long-n (a-priori, #28 tables)

Committed `#28` WASM SIMD128 profile (`harness/profile/stage-profile.tables.md`).
Those tables predate #45/#47/#49/#50/#53; the **shares** are the
roofline input. Absolute ms/token is re-measured on this PR against
`b89d446`.

### Single-thread leftover (coordinator today)

| stage | n=7 ms | n=7 % | n=502 ms | n=502 % |
| --- | ---: | ---: | ---: | ---: |
| attn_qk | 0.225 | 0.118 | 880.91 | 5.89 |
| attn_softmax | 0.156 | 0.082 | 98.15 | 0.66 |
| attn_vmix | 0.126 | 0.066 | 378.94 | 2.53 |
| **attn total** | **0.507** | **0.27** | **1358.00** | **9.08** |
| rope | 0.188 | 0.099 | 15.59 | 0.10 |
| layernorm | 0.346 | 0.181 | 24.94 | 0.17 |
| ffn_swiglu | 0.463 | 0.24 | 34.13 | 0.23 |
| matmul (5 sites) | 189.16 | 99.1 | 13503.81 | 90.3 |

Attention is the only sequential stage that **grows with n²**
(ms ratio 1358/0.507 ≈ 2679 vs n² 5143). RoPE/LN/SwiGLU stay linear
in n and stay <0.3% each.

### After #50/#53 the leftover is a *larger* share of threaded wall

#50 column-split the five `matmul_ggml` sites. #53 sped the product
kernel (~1.28× kernel / ~1.2× threaded). Attention / RoPE / LN /
SwiGLU did not move. Amdahl on the #28 shares, W=4, ~3.3× on the
matmul slice (the #44 roofline), then #53's extra ~1.2× on that
already-parallel slice:

```
T_matmul_par ≈ 0.903 / 3.3 / 1.2 = 0.228 of single-thread wall
T_attn       = 0.091
T_other_seq  = 0.006   (rope + LN + swiglu + tok/emb/pool)
T_main_thr   ≈ 0.325 of single-thread wall
attn share of main threaded wall ≈ 0.091 / 0.325 = 28%
```

If attention itself takes S× on W=4:

```
speedup vs main product-path long-n ≈ 1 / ( (1 − a) + a/S )
```

| S on attention | a = 0.28 | implied vs 1.35× held |
| ---: | ---: | --- |
| 2.0 | 1.16× | miss — explain |
| 3.3 (same as matmul) | 1.24× | miss — explain |
| 4.0 (ideal, no contention) | 1.27× | miss — explain |
| 4.0 and a = 0.40 (matmul even faster on this host) | 1.38× | in range |

**The Cos 1.35× held is optimistic versus the #28 shares.** Q@K is
compute-ish (serial f32 MAC, working set is Q/K/V activations, not
the 66 MiB weight reread), so S can beat the matmul 3.3× — but L2
contention across workers still applies. This PR does **not**
promise 1.35× from the #28 table. It promises a measured stage
table (1 and 4 workers; short and long-n) on this host, and a
held-met **or** a miss explained by that table.

Short-n: a ≈ 0.003. Even S = 4 is a 0.2% wall move. Held #2 (no
>2% 8-case regression) is the constraint, not the gain.

## Split / fuse design (before code)

### What moves

One new atomic phase between today's A and B. Same join primitive
as #44 (`epoch` Release / `workers_done` Acquire). No JS
`postMessage` per site.

| phase | jobs | then (coordinator, still single-thread) |
| --- | --- | --- |
| A | site 1 QKV | bias, `split_qkv`, **RoPE** (stays) |
| **A2** | **attention, head-split** | — |
| B | site 2 out-proj | residual + **LN** (stays) |
| C | sites 3+4 up and gate | **SwiGLU** (stays) |
| D | site 5 down | residual + **LN** (stays) |

Joins / token: **60** (5×12) vs today's 48. Cost model from #44:
~1 µs `wait`/`notify` → **~60 µs/token**. At n=502 that is
immaterial next to hundreds of ms of Q@K.

### Column-split = head-split

`attn_out` is `[n_tokens, n_embd]` with heads concatenated
(`n_embd = n_heads * head_dim` = 12 × 64 = 768). There is no
cross-head dependency: each `(h, tq)` softmax is private, each
output head-slice is private.

Reuse `column_range(n_out, worker, W, align)` from #44 with
`n_out = n_embd` and `align = head_dim` (64). Worker `i` owns
output columns `[col_start, col_end)` and therefore heads
`[col_start / head_dim, col_end / head_dim)`.

For default `W = min(4, os.availableParallelism())`:

| W | heads / worker | cols / worker | leftover |
| ---: | ---: | ---: | --- |
| 1 | 12 (serial `attention_named`) | 768 | none — pool not entered |
| 2 | 6 / 6 | 384 / 384 | none |
| 3 | 4 / 4 / 4 | 256 / 256 / 256 | none |
| 4 | 3 / 3 / 3 / 3 | 192 / 192 / 192 / 192 | none |

`W=1` and the single-thread artifact never enter the pool: the
existing `attention_named` loop runs on the coordinator. Detection
/ fusion must not tax that path (held #3).

Token-tiled (query-token split) also preserves `max_abs=0` — each
`(h, tq)` is independent — but it is a **row** split of `attn_out`,
not the column-split discipline the issue named first. Head-split
is the column-split. We do not need both.

### Arithmetic (bit-exact argument)

Per `(h, tq)` the worker runs **exactly** today's body
(`ops.rs` `attention_named`):

1. Serial `dot += qh[i] * kh[i]` then `* scale`. No 4×8 tree, no
   AVX2 `ggml_vec_dot_f32` (issue #24 avalanche).
2. `softmax_inplace` on a worker-private `scores[n_tokens]` row
   (shared `crate::exp`, f64 sum).
3. `vmix_axpy` (mul+add, not FMA — issue #25).

The only difference is which `h` a worker iterates. Same Q, K, V
bytes. Same op order. That is the `max_abs=0` argument.

Scores scratch is coordinator-owned, `W * n_tokens` f32, one
allocation in `forward_hidden` next to the existing Q/K/V buffers.
Worker `i` writes only `scores[i * n_tokens .. (i+1) * n_tokens]`.
No allocation on the worker path. No shared scores row.

### Shared-memory races (argued from the split)

Reads (immutable for the duration of phase A2):

- `q`, `k`, `v` — coordinator finished RoPE before the epoch
  `Release`. Workers do not write Q/K/V.
- `scale`, `n_tokens`, `n_heads`, `head_dim` — job integers.

Writes:

- Each worker writes `attn_out[tq * n_embd + o]` only for `o` in
  its half-open column range. Ranges disjoint → no write-write.
- Each worker writes only its scores row. Disjoint.
- Same epoch / `workers_done` happens-before as #44. Coordinator
  `Acquire`s `workers_done` before phase B reads `attn_out`.

No new unbounded `unsafe`. Existing SIMD128 loads stay bounded.
Pointers in the job are the coordinator's allocations and stay
valid until the join. `SiteJob` gains an attention opcode
(`TY_ATTN`) plus V / scores pointers; matmul sites ignore them.

RoPE is **not** fused into the Q/K write and is **not**
worker-local in this chip. Fusing it into A2 (worker applies
NEOX only on owned heads) is bit-exact and would cost zero extra
joins, but 0.10% of single-thread long-n cannot close a 1.35×
gap. Leave it; say so in the table. If the measured miss is
"attention S was 2.x and RoPE showed up as material on the
product path," that is a finding, not a silent fuse.

## Two-artifact / four-artifact discipline (unchanged)

- `milton_bg.wasm` / `milton_relaxed_bg.wasm` — no `wasm-threads`.
  Serial `attention_named`. Must stay within 2% of `b89d446`.
- `milton_threads_bg.wasm` / `milton_threads_relaxed_bg.wasm` —
  new A2 phase only when `pool_live()` (`W > 1`).
- Loader picks unchanged. SAB-absent / `MILTON_WASM_THREADS=0` /
  `MILTON_THREADS=1` still pick the single-thread artifact
  (Flint #50 ASK). No new unbounded `unsafe`.
- Committed wasm: remapped `scripts/build-wasm.sh`. CI is the
  build of record if a local rebuild diverges (build-std lesson).

## CI (Flint ASK 2 lesson from #53)

Existing `wasm-compare` / `wasm-compare-node26` already force
`MILTON_WASM_THREADS=1 MILTON_THREADS=4` on the product path.
Threaded verdicts must keep showing `wasm_artifact=threads`,
`workers>1`, `max_abs=0` under Node 22 and 26. Tight lane. No
epsilon / goldens / `expected.json` change. Must-fail stays RED.

`profile:stages` on this PR adds 1-worker and 4-worker tables for
short-hello-document and long-repeated on the product path
(threads + relaxed when the probe passes). The #28 single-thread
script stays; the new tables are the held evidence.

## Host for the held table

Cursor cloud VM: Linux x86_64, 4× Intel Xeon (KVM), Node v22.14.0,
`SharedArrayBuffer` present, `os.availableParallelism() = 4`. Flint
pairs M4. This PR reports the x86 VM.

## Measured addendum (after A2 landed)

A2 join on this KVM host is **not** ~1 µs. Unconditional A2 at n≤19
(the 8-case set) added ~24 ms/embed and **regressed short-n ~60%**.
`ATTN_PARALLEL_MIN_TOKENS = 32` keeps A2 off below that floor
(same `attention_named` body — bit-exact). Long-n n=502 always
takes the split. Not a join micro-opt as the headline; it is the
short-n held.

Quiet product-path pair vs `b89d446` on this VM (auto threads,
relaxed): long 3242.01 → 2287.39 ms = **1.42×** (held 1.35×);
short 297.23 → 294.34 ms (no >2% regression). Stage table:
`harness/profile/threads-stage-profile.tables.md`.

## Out of scope

GPU. Bun-only. Q6_K relaxed offset-fold. Native AMX. Flair batch
ingest API. Barrier/join micro-opts as the headline. Kernel-letter
retunes of #50/#53.
