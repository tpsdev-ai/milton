#!/usr/bin/env node
/**
 * Native-Rust vs WASM-SIMD: same crate, two compiles.
 * Gate: cosine >= 1-EPSILON AND max_abs <= EPSILON_ABS (epsilon.json, unchanged).
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "../lib/corpus.js";
import { loadEpsilon } from "../lib/goldens.js";
import { compareVectors } from "../lib/metrics.js";
import { createNativeEmbedder } from "../lib/milton-native.js";
import {
  embed as wasmEmbed,
  lastThreadCount,
  lastThreadReport,
  lastWasmArtifact,
} from "../../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "../..");

// Must-fire receipt: MILTON_ROPE_LIBM_SIN=1 npm run wasm:compare.
// Default wasm:compare / wasm:gate / embed:gate do not set this and must
// keep using the feature-less binary (no env-var string in the image).
if (process.env.MILTON_ROPE_LIBM_SIN === "1") {
  const script = join(ROOT, "scripts", "build-embed-rope-libm-sin.sh");
  const bin = execFileSync("bash", [script], {
    cwd: ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  }).trim();
  if (!bin || !existsSync(bin)) {
    throw new Error(
      "fail-closed: rope-libm-sin milton-embed build produced no binary",
    );
  }
  process.env.MILTON_EMBED_BIN = bin;
}

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

let result = failed === 0 ? "pass" : "fail";
const tight =
  process.env.MILTON_ROPE_LIBM_SIN !== "1" && process.env.MILTON_COMPARE_TIGHT !== "0";
let tightBlock = null;
if (tight && maxAbs !== 0) {
  result = "fail";
  tightBlock = { expected: 0, got: maxAbs };
  process.stderr.write(`BLOCK: tight max_abs expected 0 got ${maxAbs}\n`);
}

const receipt = {
  schema: "milton.native-vs-wasm/1",
  result,
  n: rows.length,
  failed,
  max_cos_dist: maxCos,
  mean_cos_dist: rows.length ? sumCos / rows.length : 0,
  max_abs: maxAbs,
  epsilon: eps.epsilon,
  epsilon_abs: eps.epsilon_abs,
  wasm_artifact: lastWasmArtifact,
  wasm_threads: lastThreadCount,
  thread_report: lastThreadReport,
  tight_max_abs: tight ? { expected: 0, got: maxAbs, pass: maxAbs === 0 } : null,
  note: "Same crate compiled native (AVX2 integer kernels + shared mul+add exp/sin/cos) and wasm32 +simd128. Q4_K/Q5_K/Q6_K, Q@K dots, softmax/silu/V-mix, and RoPE use the same math on both backends. epsilon.json is not rewritten.",
  cases: rows,
};
if (tightBlock) receipt.block = { tight_max_abs: tightBlock };

mkdirSync(join(ROOT, "harness", "receipts"), { recursive: true });
writeFileSync(
  join(ROOT, "harness", "receipts", "native-vs-wasm.json"),
  `${JSON.stringify(receipt, null, 2)}\n`,
);
native.close();
process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
if (receipt.result === "fail") process.exitCode = 1;
