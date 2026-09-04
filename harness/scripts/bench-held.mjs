#!/usr/bin/env node
/**
 * #56 held pairs: 8-case short + long-repeated on the live product path.
 * Prints lastThreadReport / lastQmatmulKernel so the lane is visible.
 */
import { performance } from "node:perf_hooks";
import { loadCorpus } from "../lib/corpus.js";
import { lastQmatmulKernel, lastThreadReport } from "../../src/index.js";

const milton = await import("../../src/index.js");
const corpus = loadCorpus();
const single = corpus.cases.slice(0, 8);
const long = corpus.cases.find((c) => c.id === "long-repeated");
if (!long) throw new Error("missing long-repeated");

await milton.embed(single[0].text, { prefix: single[0].prefix });

const t0 = performance.now();
for (const c of single) {
  await milton.embed(c.text, { prefix: c.prefix });
}
const shortMs = performance.now() - t0;

const t1 = performance.now();
await milton.embed(long.text, { prefix: long.prefix });
const longMs = performance.now() - t1;

const report = {
  env: {
    MILTON_WASM_THREADS: process.env.MILTON_WASM_THREADS ?? "(auto)",
    MILTON_THREADS: process.env.MILTON_THREADS ?? "(auto)",
    MILTON_RELAXED_SIMD: process.env.MILTON_RELAXED_SIMD ?? "(auto)",
  },
  thread_report: lastThreadReport,
  qmatmul_kernel: lastQmatmulKernel,
  short_8_ms: shortMs,
  long_repeated_ms: longMs,
};
console.log(JSON.stringify(report, null, 2));
