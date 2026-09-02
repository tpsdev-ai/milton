#!/usr/bin/env node
/**
 * Native-Rust vs WASM-SIMD: same crate, two compiles.
 * Gate: cosine >= 1-EPSILON AND max_abs <= EPSILON_ABS (epsilon.json, unchanged).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { loadEpsilon } from "../lib/goldens.js";
import { compareVectors } from "../lib/metrics.js";
import { createNativeEmbedder } from "../lib/milton-native.js";
import { embed as wasmEmbed } from "../../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");

const corpus = loadCorpus();
const eps = loadEpsilon();
const native = createNativeEmbedder();
const nativeEmbed = native.embed;

const rows = [];
let maxCos = 0;
let sumCos = 0;
let maxAbs = 0;
let failed = 0;

for (const c of corpus.cases) {
  const native = await nativeEmbed(c.text, { prefix: c.prefix });
  const wasm = await wasmEmbed(c.text, { prefix: c.prefix });
  const cmp = compareVectors(wasm, native, {
    epsilon: eps.epsilon,
    epsilonAbs: eps.epsilon_abs,
  });
  const row = {
    id: c.id,
    prefix: c.prefix,
    dims: wasm.length,
    cosine: cmp.cosine,
    cos_dist: cmp.cos_dist,
    max_abs: cmp.max_abs,
    pass: cmp.pass,
    reason: cmp.reason ?? null,
  };
  rows.push(row);
  if (cmp.cos_dist > maxCos) maxCos = cmp.cos_dist;
  if (Number.isFinite(cmp.max_abs) && cmp.max_abs > maxAbs) maxAbs = cmp.max_abs;
  sumCos += Number.isFinite(cmp.cos_dist) ? cmp.cos_dist : 1;
  if (!cmp.pass) failed += 1;
}

const receipt = {
  schema: "milton.native-vs-wasm/1",
  result: failed === 0 ? "pass" : "fail",
  n: rows.length,
  failed,
  max_cos_dist: maxCos,
  mean_cos_dist: rows.length ? sumCos / rows.length : 0,
  max_abs: maxAbs,
  epsilon: eps.epsilon,
  epsilon_abs: eps.epsilon_abs,
  note: "Same crate compiled native (AVX2 kernels when present) and wasm32 +simd128. Q4_K/Q5_K/Q6_K and Q@K dots are bit-exact; first residual is softmax f32::exp (glibc expf vs WASM compiler-builtins). epsilon.json is not rewritten.",
  cases: rows,
};

mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(
  join(ROOT, "harness", "receipts", "native-vs-wasm.json"),
  `${JSON.stringify(receipt, null, 2)}\n`,
);
native.close();
process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
if (receipt.result === "fail") process.exitCode = 1;
