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
import { resolveGguf } from "../../src/index.js";

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
    vs_flair: delta(
      (singleCases.length / singleMs) * 1000,
      baseline.single?.embeddings_per_sec,
    ),
  },
  batched: {
    n: batchedN,
    ms: batchedMs,
    embeddings_per_sec: (batchedN / batchedMs) * 1000,
    note: "sequential embed() calls on one WASM instance (no native threads)",
    vs_flair: delta((batchedN / batchedMs) * 1000, baseline.batched?.embeddings_per_sec),
  },
  baseline: {
    source: "harness/goldens/baseline-bench.json",
    measured: baseline.measured,
    cold_start_ms: baseline.cold_start_ms,
    single_embeddings_per_sec: baseline.single?.embeddings_per_sec,
    batched_embeddings_per_sec: baseline.batched?.embeddings_per_sec,
  },
};

mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(join(ROOT, "harness", "receipts", "wasm-bench.json"), `${JSON.stringify(receipt, null, 2)}\n`);
process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
