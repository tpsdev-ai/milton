# #43 plan — relaxed SIMD i8×i7 dot, exactness by value range

Plan-only. No kernel / loader / probe code in this commit.
Derived from `main` `e842447` (`crate/src/qmatmul_simd128.rs`,
`crate/src/qmatmul.rs`, `src/index.js`, `src/wasm-threads.js`).
Not estimated. Not "the integer dot is deterministic."

Sherlock 2026-09-02 + Flint reopen + Nathan 14:49Z runtime-floor
update are one spec. This file is the first-comment text.

## Exactness — by value range, not by spec determinism

`i16x8.relaxed_dot_i8x16_i7x16_s` multiplies 16 i8×i7 lane pairs and
adds adjacent products into i16. `i32x4.relaxed_dot_i8x16_i7x16_add_s`
sums pairs of those i16 into i32 and adds the accumulator.

Two things in those instructions are **implementation-defined by the
spec**:

1. The i7 operand's meaning when its high bit (`0x80`) is set.
2. Whether the pairwise i16 sum wraps or saturates.

**Exactness holds because our values never reach those
implementation-defined boundaries**, so every engine returns the
mullo/madd result. It does **not** hold because "the integer dot is
deterministic."

Worst-case numeric bounds (Q4_K / Q5_K × Q8_K):

| quantity | bound | why |
| --- | ---: | --- |
| Q4_K nibble | 0..15 | 4-bit, bit 7 never set |
| Q5_K quant | 0..31 | 5-bit, bit 7 never set |
| Q8_K activation | −128..127 | i8 |
| \|product\| Q4_K | ≤ 15×128 = 1920 | |
| \|product\| Q5_K | ≤ 31×128 = 3968 | |
| pairwise i16 sum Q4_K | ≤ 2×1920 = 3840 | |
| pairwise i16 sum Q5_K | ≤ 2×3968 = 7936 | |
| i16 saturation / wrap | 32767 | 3840 and 7936 ≪ 32767 |

Neither the i7 high-bit branch nor the wrap-vs-saturate branch is
ever exercised. The f32 scale stage stays in today's order
(`q4k_mins_f32` mul+add, `fma_f32x4` on Q5_K), so `wasm:compare`
tight `max_abs=0` against native holds without touching native.

## Per-quant scope

| quant | values | i7 high bit | relaxed-dot? | reason |
| --- | --- | --- | --- | --- |
| Q4_K | 0..15 | never set | **yes** | in range |
| Q5_K | 0..31 | never set | **yes** | in range |
| Q6_K | 0..63 = `0x3F` | never set | **no** | −32 offset, not range |

Q6_K **does** fit the unsigned i7 operand (`63 = 0x3F`). Range is
not the blocker. Reconstruction is `q6 − 32`. A relaxed-dot Q6_K
path is exact only if the offset is folded out:

```
Σ q8·(v − 32) = Σ q8·v − 32·Σ q8
```

`Σ q8` is the block's existing `bsums` (`BlockQ8K.bsums`, already
used in `q4k_mins_f32` and the Q5_K min term). This chip does
**not** take that fold. Q6_K stays on the base SIMD128 kernel
(`madd_i8_pair_i16` in `qmatmul_simd128.rs`). The reason is the
offset, not the range.

## Instruction count per 32 products

Counted from `crate/src/qmatmul_simd128.rs` at `e842447`. Not the
issue-body "~4 per 16" sketch.

### Q4_K perk — `q4k_stripe_iacc`

One `pair` iteration is 32 products (2 columns × 2 halves × 8-wide
`i16x8.mul`). The integer tree is:

```
i16x8_mul(v, a)           × 4
hsum_i16x8 = i32x4_dot_i16x8(v, splat(1))
             + hsum_i32x4   × 4
hsum_i32x4 = 4× extract_lane + 3× i32 add
```

| op | count / 32 products |
| --- | ---: |
| `i16x8.mul` | 4 |
| `i32x4.dot_i16x8` | 4 |
| `i32x4.extract_lane` | 16 |
| i32 add | 12 |
| **total** | **36** |

Activation widen (`i16x8.extend_low_i8x16` × 2 for `a0`/`a1`) is
hoisted across 4 pairs (128 products) and is not in this tree.

After — restructure to 16-wide i8×i7, `i16x8.relaxed_dot_i8x16_i7x16_s`
then the same `hsum_i16x8`:

| op | count / 32 products |
| --- | ---: |
| `i16x8.relaxed_dot_i8x16_i7x16_s` | 2 |
| `i32x4.dot_i16x8` | 2 |
| `i32x4.extract_lane` | 8 |
| i32 add | 6 |
| **total** | **18** |

**36 → 18 (2.00×) on the perk integer tree.**

### Q4_K (b′) — `q4k_stripe_lane_acc`

Keeps `i32x4` lanes; no hsum in the stripe.

| op | before / 32 products | after |
| --- | ---: | ---: |
| `i16x8.mul` | 4 | 0 |
| `i32x4.dot_i16x8` | 4 | 0 |
| `i32x4.add` | 4 | 0 |
| `i32x4.relaxed_dot_i8x16_i7x16_add_s` | 0 | 2 |
| **total** | **12** | **2** |

**12 → 2 (6.00×) on the (b′) accumulate.**

### Q5_K — `maddubs_epi16` + `madd_epi16`

`maddubs_epi16` (lines 34–44) per 16 products: 2× u8 widen, 2× i8
widen, 2× `i16x8.mul`, 2× shuffle, 1× `i16x8.add_sat` = 9, plus
`i32x4.dot_i16x8` (scale) + `i32x4.add` = 11 per 16, **22 per 32**.

After: `i16x8.relaxed_dot_i8x16_i7x16_s` + `i32x4.dot_i16x8` +
`i32x4.add` = 3 per 16, **6 per 32**.

**22 → 6 (3.67×) on the Q5_K integer tree.**

Q6_K `madd_i8_pair_i16` is unchanged.

## Roofline (before any speedup claim)

nomic-bert Q4_K_M, one token, 12 layers. MAC = `2 × n_in × n_out`.

| site | n_in × n_out | ty | FLOP/token |
| --- | --- | --- | ---: |
| QKV ×12 | 768 × 2304 | Q5_K | 42,467,328 |
| out-proj ×12 | 768 × 768 | Q4_K | 14,155,776 |
| FFN up ×12 | 768 × 3072 | Q4_K | 56,623,104 |
| FFN gate ×12 | 768 × 3072 | Q4_K | 56,623,104 |
| FFN down ×6 | 3072 × 768 | Q4_K | 28,311,552 |
| FFN down ×6 | 3072 × 768 | Q6_K | 28,311,552 |
| **total** | | | **226,492,416** |

Q4_K = 155,713,536 (68.7%). Q5_K = 18.7%. Q6_K stays base (12.5%).
Sites this chip touches: **87.5%** of matmul FLOP.

#42 left the integer-dot tree as the bound on those sites. Instruction
reduction on that tree is 2.00× (Q4_K perk) / 6.00× (Q4_K b′) /
3.67× (Q5_K). Taking a conservative **1.5×** on the 87.5% share
(unpack / scale / mins / Q6_K still run):

```
wall' / wall ≈ 0.125 + 0.875 / 1.5 = 0.708  →  1.41×
```

At 2.00× on the share: `0.125 + 0.875 / 2 = 0.562` → 1.78×.

Held, vs current main `e842447` on the **same host** (Flint pairs M4;
this PR reports the x86 VM only):

- `wasm:bench` 8-case single **≥ 1.4×** on an engine with relaxed-simd
- `profile:stages` n=7 **≤ 17.0 ms/token** on that engine
- Fallback engine (forced simd128) **within 2%** of `e842447` — no
  regression from the detection branch itself

A miss is explained by this table (roofline vs measured), not waved.

## Feature detection — probe, not a version sniff

Nathan 14:49Z: Node floor is **22** (V8 12.4). Confirmed on this host:
Node v22.14.0 `WebAssembly.validate`s a one-function module containing
`i16x8.relaxed_dot_i8x16_i7x16_s` (`0xfd 0x112` = `fd 92 02`). Bun
1.3.10 (JSC) validates SIMD128 and **rejects** the relaxed probe.
Version-sniffing Node would mark bun as "new enough" and then fail
instantiate. Detection is the probe, never `process.versions`.

A wasm **module** that contains any relaxed opcode fails
`WebAssembly.validate` on bun. Both kernels therefore cannot live in
one artifact. Two single-thread artifacts, one loader:

| file | rustc features | when loaded |
| --- | --- | --- |
| `wasm/milton_bg.wasm` | `+simd128` (today) | probe fail, or `MILTON_RELAXED_SIMD=0` |
| `wasm/milton_relaxed_bg.wasm` | `+simd128` + crate feature `relaxed-simd` | probe pass (default on Node ≥22) |
| `wasm/milton_threads_bg.wasm` | unchanged | threads path; **not this chip** |

The relaxed crate feature compiles the Q4_K / Q5_K integer trees with
`#[target_feature(enable = "relaxed-simd")]` only on those functions.
`+relaxed-simd` is **not** a global RUSTFLAG — LLVM must not rewrite
the f32 scale stage (`relaxed_madd` is mul+add on V8; we still do
not want a second f32 path). Q6_K and everything else stay the
simd128 code.

Pick once at load. No per-superblock / per-call branch.

`MILTON_RELAXED_SIMD=1` forces the relaxed artifact and **fail-closes**
if the probe rejects. `MILTON_RELAXED_SIMD=0` forces simd128 even
when the probe passes (the fallback compare lane).

This sandbox has Node v22.14.0 and **no bun**. Node 22.14 has no
`--no-wasm-relaxed-simd` (the V8 flag is gone once relaxed SIMD is
default-on). The first-class fallback lane is therefore
`MILTON_RELAXED_SIMD=0` on Node — same artifact bun would load.
Stated here so the compare lane is not an afterthought.

## Observability

Same grain as `lastThreadReport` / `lastQ4kCalibration`:

```
lastQmatmulKernel = {
  kernel: 'relaxed' | 'simd128',
  probe: boolean,   // capability, not the pick
  forced: boolean,
}
```

`kernel` is the pick. `probe` is whether `WebAssembly.validate`
accepted the relaxed-dot module. `MILTON_RELAXED_SIMD=0` on this
Node is `{kernel:'simd128', probe:true, forced:true}`.

## Bit-exactness gate

- `wasm:compare` on Node ≥22 (auto → relaxed): tight `max_abs=0`
- `wasm:compare:simd128` (`MILTON_RELAXED_SIMD=0`): tight `max_abs=0`
- `wasm:gate` 18/18 covering both
- `wasm:must-fail` RED
- No epsilon / goldens / expected.json / dequant-epsilon.json change

A lane that only runs the simd128 artifact proves nothing about the
relaxed path. Both run in CI. Node 22 is the setup-ci pin
(`package.json` `engines.node` `>=22`). Node 26 is added as a
compare-only matrix job if the workflow stays within the 25-minute
timeout; otherwise the sandbox Node 22 run is recorded and the
matrix gap is stated.

## Out of scope

- Threads (`#44` / `#50` / `#51` already shipped). Threads artifact
  stays simd128.
- Calibration framework variants (perk / b′). Both stay; each gets
  a relaxed integer tree when the relaxed artifact is loaded.
- f32 stage order / FMA.
- Native AVX2 (`#40` / `#49`).
- Q6_K offset fold (optional; not taken).
- Other model architectures.
