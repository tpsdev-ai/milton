#!/usr/bin/env node
/**
 * WASM-SIMD cold-start + throughput vs the captured Flair-path baseline.
 * A delta against baseline-bench.json, not a claim.
 */
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { loadCorpus } from "../lib/corpus.js";
import { lastQ4kCalibration, lastThreadReport, resolveGguf } from "../../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");
const BASELINE_PATH = join(HERE, "..", "goldens", "baseline-bench.json");

const corpus = loadCorpus();
const baseline = JSON.parse(readFileSync(BASELINE_PATH, "utf8"));
const gguf = resolveGguf();
if (!existsSync(gguf)) {
  console.error(`fail-closed: GGUF not found at ${gguf}`);
  process.exit(2);
}

const tImport0 = performance.now();
const milton = await import("../../src/index.js");
const tImport1 = performance.now();

const first = corpus.cases[0];
const t0 = performance.now();
const v = await milton.embed(first.text, { prefix: first.prefix });
const t1 = performance.now();
const coldStartMs = t1 - t0;

const singleN = baseline.single?.n ?? 8;
const singleCases = corpus.cases.slice(0, singleN);
const tSingle0 = performance.now();
for (const c of singleCases) {
  await milton.embed(c.text, { prefix: c.prefix });
}
const tSingle1 = performance.now();
const singleMs = tSingle1 - tSingle0;

const batchedN = corpus.cases.length;
const tBatch0 = performance.now();
for (const c of corpus.cases) {
  await milton.embed(c.text, { prefix: c.prefix });
}
const tBatch1 = performance.now();
const batchedMs = tBatch1 - tBatch0;

function delta(ours, theirs) {
  if (!(theirs > 0) || !Number.isFinite(ours)) return null;
  return {
    ours,
    baseline: theirs,
    delta: ours - theirs,
    ratio: ours / theirs,
    gap: theirs / ours,
  };
}

function flintInterpretation(singleOurs, batchedOurs, multiSingle, multiBatched, matchedSingle, matchedBatched) {
  const vsMulti = multiSingle > 0 && Number.isFinite(singleOurs) ? multiSingle / singleOurs : null;
  const vsMatched = matchedSingle > 0 && Number.isFinite(singleOurs) ? matchedSingle / singleOurs : null;
  const vsMultiBatched =
    multiBatched > 0 && Number.isFinite(batchedOurs) ? multiBatched / batchedOurs : null;
  const vsMatchedBatched =
    matchedBatched > 0 && Number.isFinite(batchedOurs) ? matchedBatched / batchedOurs : null;
  const threadingFactor =
    vsMulti != null && vsMatched != null && vsMatched > 0 ? vsMulti / vsMatched : null;
  return {
    note: "issue #23 addendum: (multi-thread gap ÷ thread-matched gap) ≈ remaining threading factor (WASM-threads deferred). Thread-matched residual ≈ SIMD128-vs-AVX2 + inherent WASM overhead (~1.5–2× floor).",
    single: {
      vs_multi_thread_gap: vsMulti,
      vs_thread_matched_gap: vsMatched,
      threading_factor: threadingFactor,
      first_principles: {
        vs_multi_thread_target: "about 10-15x",
        vs_thread_matched_target: "low single digits",
        wasm_overhead_floor: "about 1.5-2x",
        vs_multi_thread_in_band: vsMulti != null && vsMulti >= 8 && vsMulti <= 16,
        vs_thread_matched_low_single_digits: vsMatched != null && vsMatched < 5,
        look_if_matched_far_worse:
          vsMatched != null && vsMatched >= 5
            ? "thread-matched gap is far worse than low single digits — confirm SIMD128 is actually vectorizing before concluding viability"
            : null,
      },
    },
    batched: {
      vs_multi_thread_gap: vsMultiBatched,
      vs_thread_matched_gap: vsMatchedBatched,
    },
  };
}

const receipt = {
  schema: "milton.bench-wasm/1",
  backend: "wasm-simd",
  host: {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
  },
  dims: v.length,
  import_ms: tImport1 - tImport0,
  cold_start_ms: coldStartMs,
  cold_start_vs_flair: delta(coldStartMs, baseline.cold_start_ms),
  single: {
    n: singleCases.length,
    ms: singleMs,
    embeddings_per_sec: (singleCases.length / singleMs) * 1000,
    vs_flair_multi_thread: delta(
      (singleCases.length / singleMs) * 1000,
      baseline.single?.embeddings_per_sec,
    ),
    vs_flair_thread_matched: delta(
      (singleCases.length / singleMs) * 1000,
      baseline.single_thread?.single?.embeddings_per_sec,
    ),
  },
  batched: {
    n: batchedN,
    ms: batchedMs,
    embeddings_per_sec: (batchedN / batchedMs) * 1000,
    note: "sequential embed() calls on one WASM instance (no native threads)",
    vs_flair_multi_thread: delta((batchedN / batchedMs) * 1000, baseline.batched?.embeddings_per_sec),
    vs_flair_thread_matched: delta(
      (batchedN / batchedMs) * 1000,
      baseline.single_thread?.batched?.embeddings_per_sec,
    ),
  },
  baseline: {
    source: "harness/goldens/baseline-bench.json",
    measured: baseline.measured,
    cold_start_ms: baseline.cold_start_ms,
    single_embeddings_per_sec: baseline.single?.embeddings_per_sec,
    batched_embeddings_per_sec: baseline.batched?.embeddings_per_sec,
    single_thread: baseline.single_thread ?? null,
  },
  q4k_calibration: lastQ4kCalibration,
  thread_report: lastThreadReport,
  flint_addendum: flintInterpretation(
    (singleCases.length / singleMs) * 1000,
    (batchedN / batchedMs) * 1000,
    baseline.single?.embeddings_per_sec,
    baseline.batched?.embeddings_per_sec,
    baseline.single_thread?.single?.embeddings_per_sec,
    baseline.single_thread?.batched?.embeddings_per_sec,
  ),
};

mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(join(ROOT, "harness", "receipts", "wasm-bench.json"), `${JSON.stringify(receipt, null, 2)}\n`);
process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
