# Milton conformance + bench harness

The keystone artifact. Built **before** the embedder. Three things at once:

- correctness **oracle** (is a vector right?)
- optimization **target** (is it lighter/faster?)
- iteration **signal** (am I converging?)

Full spec: [`docs/HARNESS-SPEC.md`](../docs/HARNESS-SPEC.md). This directory implements it. Nothing in `src/` is an embedder — that is the next issue.

## Reference (the oracle)

Pinned path, not a vibe:

| pin | value |
|---|---|
| model | nomic-embed-text-v1.5 |
| GGUF | `nomic-embed-text-v1.5.Q4_K_M.gguf` |
| GGUF sha256 | `d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac` (same file Flair/HFE ship) |
| pooling | mean (GGUF `nomic-bert.pooling_type = 1`; llama.cpp `--pooling mean`) |
| normalize | L2 (`--embd-normalize 2`, matching HFE) |
| prefixes | Flair/HFE: `search_document: {text}` / `search_query: {text}` / omitted = passthrough |
| llama.cpp | commit + digest in [`goldens/pin.json`](goldens/pin.json) |

The GGUF and the llama.cpp tree live in `harness/vendor/` (gitignored). They are **not** shipped in `@tpsdev-ai/milton`.

```sh
npm run harness:setup          # checkout pin.json's llama.cpp commit (fail-closed on leftover drift), build llama-embedding, fetch+verify GGUF
npm run harness:goldens        # write goldens/vectors.json + pin.json + CLS control
npm run harness:f16            # llama-embedding on original F16 GGUF → vectors-f16.json
npm run harness:quant-budget   # derive cos_dist(ref_f32, q_llama) per case
npm run harness:epsilon        # run reference twice; derive EPSILON / EPSILON_ABS
```

## Gate

Official `embed-gate` (Rust): Milton Q4_K_M vs `ref_f32` (llama-embedding on
the original HF F16 GGUF). Pass when `cos_dist(milton, ref_f32) <=`
`quant-budget.json`'s `gate_cos_dist` (max llama Q4-vs-F16 error × 3).
`epsilon.json` is unchanged — it is the Q4-vs-Q4 run-to-run floor used to
lock empty-none / short-hello-none and for `embed-must-fail`.

JS replay gate (wiring) still uses the Q4 goldens:

```
cosine(got, expected) >= 1 - EPSILON
AND
max_i |got_i - expected_i| <= EPSILON_ABS
```

Both required for the JS replay. Fail closed: one miss fails the run and names the item. Receipts:

```
{corpus_digest, reference_digest, n, max_cos_dist, mean_cos_dist, max_abs, pass|fail, failures[]}
```

```sh
npm run harness:gate                 # replay goldens (wiring) — must PASS
npm run harness:gate -- --wrong perturb
npm run harness:gate -- --wrong drop-prefix
npm run harness:gate -- --wrong swap-pooling
npm run harness:gate -- --wrong truncate
MILTON_GATE_EMBEDDER=reference npm run harness:gate   # live llama.cpp vs goldens
```

`EPSILON` / `EPSILON_ABS` are **derived** from the reference's own run-to-run delta (`goldens/epsilon.json`). Do not loosen them to pass.

Dequant (issue #4) has its own fixture + gate, still llama.cpp-oracled, not a second embedder:

```sh
npm run dequant:goldens     # llama.cpp ggml to_float → goldens/dequant.json
npm run dequant:gate        # Milton vs fixture (N tensors, max/mean abs)
npm run dequant:must-fail   # wrong block scale / wrong type must go RED
```

## Must-fail control

`npm test` / `npm run harness:must-fail` ships four wrong embedders. Each must turn the run RED and name the failure before the gate is trusted:

1. perturb a dimension
2. drop the nomic prefix
3. swap pooling (committed CLS vector vs mean)
4. truncate a dimension

## Bench

```sh
npm run harness:footprint   # shipped MB + assert zero native binary in src/
npm run harness:bench       # current Flair path (HFE / Harper models.embed) on this host
```

If the Flair path cannot run here, the bench script exits `2` and writes `BLOCKED` with what was attempted. It will not invent numbers.

## Layout

```
harness/corpus/corpus.json     fixed cases; each documents the failure mode it traps
harness/goldens/vectors.json   pinned Q4_K_M llama-embedding vectors (q_llama)
harness/goldens/vectors-f16.json  F16/F32 llama-embedding oracle (ref_f32)
harness/goldens/quant-budget.json  derived cos_dist(ref_f32, q_llama) + gate
harness/goldens/tokens.json    pinned reference token-ID sequences (tokenizer slice)
harness/goldens/tokenizer-pin.json  HF nomic tokenizer source + file digests
harness/goldens/pin.json       Q4_K_M GGUF digest + llama.cpp commit/digest
harness/goldens/pin-f16.json   F16 GGUF digest + llama.cpp commit
harness/goldens/epsilon.json   Q4-vs-Q4 run-to-run floor (not the F32 gate)
harness/goldens/controls.json  CLS-pooled must-fail fixture
harness/lib/                   gate, metrics, prefixes, receipts, reference runner
harness/scripts/               setup / generate / derive / gate / bench
harness/test/                  must-fail + unit tests (node:test)
harness/vendor/                llama.cpp + GGUF (not committed, not shipped)
```

